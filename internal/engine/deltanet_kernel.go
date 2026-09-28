//go:build mlx

package engine

import (
	"os"
	"strconv"

	"github.com/vaktex/vakt/internal/engine/mlx"
)

// The Metal path of the Gated DeltaNet layer runs as four custom kernels
// between the in_proj and out_proj matmuls:
//
//	proj ─► convSilu ─► deltaPrep ─► gatedDelta ─► gatedNorm ─► out_proj
//	  └─────(b, a)──────┘                        (z)
//
// Each kernel reads its slice of the fused projection in place (row stride
// P = 6144+2048+16+16), so none of the column splits, dtype casts, l2norms,
// sigmoids or reshapes of the plain-ops path is materialised. On the plain-ops
// path those were ~20 full passes over [B*T, 2048..6144] tensors per layer,
// which made the layer memory-bound rather than matmul-bound.
//
// All arithmetic is float32; results are rounded to the compute dtype only
// where the plain-ops path rounds them.

// kernelHeader holds the scalar helpers shared by the kernels. They are the
// numerically stable forms MLX's own ops use, with precise transcendentals
// (runtime-compiled Metal kernels default to fast math).
const kernelHeader = `
inline float vakt_sigmoid(float x) {
    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
    return (x < 0.0f) ? y : 1.0f - y;
}
inline float vakt_log1p(float x) {
    const float u = 1.0f + x;
    return (u == 1.0f) ? x : metal::precise::log(u) * x / (u - 1.0f);
}
inline float vakt_softplus(float x) {
    // logaddexp(x, 0)
    const float mx = metal::max(x, 0.0f);
    const float mn = metal::min(x, 0.0f);
    return metal::isinf(mx) ? mx : mx + vakt_log1p(metal::precise::exp(mn - mx));
}
`

// convSiluSource is the causal depthwise conv (kernel K) followed by SiLU,
// y[b,t,c] = silu(sum_k w[k,c] * x[b,t-K+1+k,c]),
// accumulated in float32, four channels and TT consecutive timesteps per
// thread (a sliding window over the input rows). x is the fused
// projection [B,T,P] (only the first C columns are read), w is [K,C], y is
// [B,T,C].
const convSiluSource = `
    const uint c0 = thread_position_in_grid.x * 4;
    const int t0 = int(thread_position_in_grid.y) * TT;
    const uint b = thread_position_in_grid.z;
    const int T = x_shape[1];
    const int P = x_shape[2];
    if (int(c0) >= C || t0 >= T) { return; }
    for (int e = 0; e < 4; ++e) {
        const uint c = c0 + e;
        float wv[K];
        for (int k = 0; k < K; ++k) { wv[k] = float(w[k * C + c]); }
        // Sliding window: input rows t0-(K-1) .. t0+TT-1, read once for TT
        // outputs (4 reads per output before).
        float win[TT + K - 1];
        for (int i = 0; i < TT + K - 1; ++i) {
            const int src = t0 - (K - 1) + i;
            win[i] = (src >= 0 && src < T) ? float(x[(size_t(b) * T + src) * P + c]) : 0.0f;
        }
        for (int tt = 0; tt < TT; ++tt) {
            const int t = t0 + tt;
            if (t >= T) { break; }
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                if (t - (K - 1) + k >= 0) { acc += wv[k] * win[tt + k]; }
            }
            const float s = acc / (1.0f + metal::precise::exp(-acc));
            y[(size_t(b) * T + t) * C + c] = static_cast<OutT>(s);
        }
    }
`

// deltaPrepSource computes the per-(token, head) scalars of the gated delta
// rule once, one SIMD group per (token, head), into scal [B,T,H,4]:
//
//	[0] qinv  = rsqrt(|q|^2 + 1e-6)      (l2norm of the conv q)
//	[1] kinv  = rsqrt(|k|^2 + 1e-6)      (l2norm of the conv k)
//	[2] beta  = sigmoid(b)
//	[3] decay = exp(-exp(A_log) * softplus(a + dt_bias))   (aneg = -exp(A_log))
//
// They used to be recomputed inside the recurrence by all 128 value-column
// SIMD groups of a head, which made that kernel ALU-bound on transcendentals.
// conv is [B,T,3*H*Dk] (q|k|v); proj is the fused projection [B,T,P] with b
// at column BOFF and a at BOFF+H.
const deltaPrepSource = `
    const uint lane = thread_position_in_grid.x;
    const uint h = thread_position_in_grid.y;
    const uint row = thread_position_in_grid.z;
    const int CW = conv_shape[2];
    const int P = proj_shape[2];
    constexpr int NP = Dk / 32;
    const device InT* cr = conv + size_t(row) * CW;
    float qss = 0.0f, kss = 0.0f;
    for (int i = 0; i < NP; ++i) {
        const int s = lane * NP + i;
        const float q = float(cr[h * Dk + s]);
        const float k = float(cr[H * Dk + h * Dk + s]);
        qss += q * q;
        kss += k * k;
    }
    qss = simd_sum(qss);
    kss = simd_sum(kss);
    if (lane == 0) {
        const device ProjT* pr = proj + size_t(row) * P;
        device float* o = scal + (size_t(row) * H + h) * 4;
        o[0] = metal::precise::rsqrt(qss + 1e-6f);
        o[1] = metal::precise::rsqrt(kss + 1e-6f);
        o[2] = vakt_sigmoid(float(pr[BOFF + h]));
        o[3] = metal::precise::exp(aneg[h] * vakt_softplus(float(pr[BOFF + H + h]) + dtbias[h]));
    }
`

// gatedDeltaSource is the gated delta rule recurrence
// (torch_recurrent_gated_delta_rule, token by token, float32 state).
//
// Layout: a SIMD group covers 8 value columns of one (batch, head); each
// column is owned by a quad of 4 lanes, each lane holding Dk/4 rows of the
// state column in registers. The reductions over Dk are then two quad
// shuffles instead of a five-level simd_sum, and each token's q and k are
// read by 8 columns at once. The l2norm factors and decay come from
// deltaPrep's scalars and are applied to the dot products, not per element:
//
//	kv    = decay * kinv * sum(S * k)
//	S     = decay * S + k * ((v - kv) * beta * kinv)
//	out   = qinv / sqrt(Dk) * sum(S * q)
//
// which is the reference with q, k l2-normalised (q scaled by 1/sqrt(Dk)).
// conv is [B,T,3*H*Dk] (q|k|v), scal is deltaPrep's [B,T,H,4]; y is float32
// [B,T,H,Dv].
const gatedDeltaSource = `
    const uint lane = thread_position_in_grid.x;
    const uint jg = thread_position_in_grid.y;
    const uint bh = thread_position_in_grid.z;
    const uint b = bh / H;
    const uint h = bh % H;
    // T is read at runtime (not a template arg) so one compiled kernel
    // serves every sequence length.
    const int T = conv_shape[1];
    const int CW = conv_shape[2];
    constexpr int R = Dk / 4;
    const uint part = lane % 4;
    const uint j = jg * 8 + lane / 4;
    const float qscale = metal::precise::rsqrt(float(Dk));
    float state[R];
    for (int i = 0; i < R; ++i) { state[i] = 0.0f; }
    for (int t = 0; t < T; ++t) {
        const size_t row = size_t(b) * T + t;
        const device InT* cr = conv + row * CW;
        const device InT* kp = cr + H * Dk + h * Dk + part * R;
        const device InT* qp = cr + h * Dk + part * R;
        const device float* sc = scal + (row * H + h) * 4;
        const float qinv = sc[0];
        const float kinv = sc[1];
        const float bt = sc[2];
        const float dcy = sc[3];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int i = 0; i < R; i += 4) {
            a0 += state[i] * float(kp[i]);
            a1 += state[i + 1] * float(kp[i + 1]);
            a2 += state[i + 2] * float(kp[i + 2]);
            a3 += state[i + 3] * float(kp[i + 3]);
        }
        float kv = (a0 + a1) + (a2 + a3);
        kv += simd_shuffle_xor(kv, 1);
        kv += simd_shuffle_xor(kv, 2);
        kv *= dcy * kinv;
        const float coef = (float(cr[2 * H * Dk + h * Dv + j]) - kv) * bt * kinv;
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int i = 0; i < R; i += 4) {
            state[i] = state[i] * dcy + float(kp[i]) * coef;
            state[i + 1] = state[i + 1] * dcy + float(kp[i + 1]) * coef;
            state[i + 2] = state[i + 2] * dcy + float(kp[i + 2]) * coef;
            state[i + 3] = state[i + 3] * dcy + float(kp[i + 3]) * coef;
            o0 += state[i] * float(qp[i]);
            o1 += state[i + 1] * float(qp[i + 1]);
            o2 += state[i + 2] * float(qp[i + 2]);
            o3 += state[i + 3] * float(qp[i + 3]);
        }
        float o = (o0 + o1) + (o2 + o3);
        o += simd_shuffle_xor(o, 1);
        o += simd_shuffle_xor(o, 2);
        if (part == 0) { y[(row * H + h) * Dv + j] = o * (qinv * qscale); }
    }
`

// gatedNormSource is RMSNormGated followed by the cast for out_proj, one
// SIMD group per (token, head): out = rmsnorm(core) * w * silu(z), in
// float32, rounded to OutT. core is float32 [B,T,H,Dv]; z is read from the
// fused projection [B,T,P] at column ZOFF; out is [B,T,H*Dv].
const gatedNormSource = `
    const uint lane = thread_position_in_grid.x;
    const uint h = thread_position_in_grid.y;
    const uint row = thread_position_in_grid.z;
    const int P = proj_shape[2];
    constexpr int NP = Dv / 32;
    const device float* cr = core + (size_t(row) * H + h) * Dv;
    const device ProjT* zr = proj + size_t(row) * P + ZOFF + h * Dv;
    float v[NP];
    float ss = 0.0f;
    for (int i = 0; i < NP; ++i) {
        v[i] = cr[lane * NP + i];
        ss += v[i] * v[i];
    }
    const float inv = metal::precise::rsqrt(simd_sum(ss) / float(Dv) + 1e-6f); // rmsEps
    for (int i = 0; i < NP; ++i) {
        const int s = lane * NP + i;
        const float zf = float(zr[s]);
        const float n = w[s] * (v[i] * inv);
        out[(size_t(row) * H + h) * Dv + s] = static_cast<OutT>(n * (zf * vakt_sigmoid(zf)));
    }
`

// swigluSource is the MLP activation silu(gate) * up over the fused
// gate|up projection [N, 2I], four output elements per thread. It rounds to
// OutT after silu and after the product, as PyTorch does in bf16.
const swigluSource = `
    const uint c = thread_position_in_grid.x * 4;
    const uint r = thread_position_in_grid.y;
    if (int(c) >= I) { return; }
    const size_t base = size_t(r) * (2 * I);
    for (int e = 0; e < 4; ++e) {
        const float g = float(gu[base + c + e]);
        const float u = float(gu[base + I + c + e]);
        const float a = float(static_cast<OutT>(g * vakt_sigmoid(g)));
        out[size_t(r) * I + c + e] = static_cast<OutT>(a * u);
    }
`

// addNormSource is the residual add plus the next RMSNorm, one threadgroup
// of 256 threads per token (4 elements each, hidden = 1024):
//
//	h = r + float(d)                          (the float32 residual stream)
//	n = OutT(w * (h * rsqrt(mean(h^2) + eps)))  (the next matmul's input)
//
// It replaces a cast, an add, an RMSNorm and a cast (four launches and
// three round trips through memory) per residual.
const addNormSource = `
    const uint lid = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_grid.y;
    const size_t base = size_t(row) * Hd + lid * 4;
    threadgroup float part[8];
    float hv[4];
    float ss = 0.0f;
    for (int e = 0; e < 4; ++e) {
        hv[e] = r[base + e] + float(d[base + e]);
        h[base + e] = hv[e];
        ss += hv[e] * hv[e];
    }
    ss = simd_sum(ss);
    if (thread_index_in_simdgroup == 0) { part[simdgroup_index_in_threadgroup] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) { tot += part[i]; }
    const float inv = metal::precise::rsqrt(tot / float(Hd) + 1e-6f); // rmsEps
    for (int e = 0; e < 4; ++e) {
        n[base + e] = static_cast<OutT>(w[lid * 4 + e] * (hv[e] * inv));
    }
`

// attnPrepSource lays out full attention's inputs from the fused q|k|v
// projection [B,T,QW+2*KW] in one pass, one SIMD group per (token, slot),
// slots being the 8 query heads, then 2 key heads, then 2 value heads:
//
//	q, k: per-head RMSNorm (float32, (1+w) folded), rounded to OutT, then
//	      partial RoPE (first RD of D dims, rotate_half, theta = 1e7) in
//	      float32 as MLX's fast.rope computes it, rounded to OutT
//	v:    copied
//
// Outputs are [B,heads,T,D], the layout SDPA wants. Lane l owns elements
// l + 32*e, so RoPE's pair (i, i + RD/2) is in one lane.
const attnPrepSource = `
    const uint lane = thread_position_in_grid.x;
    const uint slot = thread_position_in_grid.y;
    const uint row = thread_position_in_grid.z;
    const int T = proj_shape[1];
    const int PW = proj_shape[2];
    const uint b = row / T;
    const uint t = row % T;
    constexpr int NE = D / 32;
    const device InT* pr = proj + size_t(row) * PW;
    if (slot >= HQ + HK) {
        const uint hv = slot - HQ - HK;
        const device InT* src = pr + HQ * 2 * D + HK * D + hv * D;
        device OutT* dst = v + ((size_t(b) * HK + hv) * T + t) * D;
        for (int e = 0; e < NE; ++e) { dst[lane + 32 * e] = static_cast<OutT>(src[lane + 32 * e]); }
        return;
    }
    const bool isq = slot < HQ;
    const uint hh = isq ? slot : slot - HQ;
    const device InT* src = isq ? pr + hh * 2 * D : pr + HQ * 2 * D + hh * D;
    const device float* w = isq ? qw : kw;
    float x[NE];
    float ss = 0.0f;
    for (int e = 0; e < NE; ++e) {
        x[e] = float(src[lane + 32 * e]);
        ss += x[e] * x[e];
    }
    const float inv = metal::precise::rsqrt(simd_sum(ss) / float(D) + 1e-6f); // rmsEps
    for (int e = 0; e < NE; ++e) {
        const int j = lane + 32 * e;
        x[e] = float(static_cast<OutT>(w[j] * (x[e] * inv)));
    }
    if (int(lane) < RD / 2) {
        // RoPE pair (lane, lane + RD/2) = (x[0], x[1]) since RD/2 == 32.
        const float dd = float(lane) / float(RD / 2);
        const float inv_freq = metal::precise::exp2(-dd * 23.2534966642f); // log2(ropeTheta)
        const float theta = float(t) * inv_freq;
        const float c = metal::fast::cos(theta);
        const float sn = metal::fast::sin(theta);
        const float x1 = x[0], x2 = x[1];
        x[0] = x1 * c - x2 * sn;
        x[1] = x1 * sn + x2 * c;
    }
    device OutT* dst = (isq ? q + (size_t(b) * HQ + hh) * T * D : k + (size_t(b) * HK + hh) * T * D) + size_t(t) * D;
    for (int e = 0; e < NE; ++e) { dst[lane + 32 * e] = static_cast<OutT>(x[e]); }
`

// attnGateSource multiplies SDPA's output [B,HQ,T,D] by sigmoid(gate), the
// gate being the second half of each query head's 2*D columns in proj, and
// writes [B,T,HQ*D] for o_proj. Rounds like the plain ops: sigmoid to OutT,
// then the product.
const attnGateSource = `
    const uint c = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    const int T = proj_shape[1];
    const int PW = proj_shape[2];
    const uint b = row / T;
    const uint t = row % T;
    const uint hh = c / D;
    const uint d = c % D;
    const float g = float(proj[size_t(row) * PW + hh * 2 * D + D + d]);
    const float sg = float(static_cast<OutT>(vakt_sigmoid(g)));
    const float ov = float(o[((size_t(b) * HQ + hh) * T + t) * D + d]);
    out[size_t(row) * HQ * D + c] = static_cast<OutT>(ov * sg);
`

// kernels are the Metal kernels of the fused path (nil off Metal).
type kernels struct {
	conv, prep, delta, norm, swiglu *mlx.Kernel
	convSteps                       int
	addNorm, attnPrep, attnGate     *mlx.Kernel
}

func newKernels() *kernels {
	return &kernels{
		convSteps: convSteps(),
		conv:      mlx.NewKernel("vakt_causal_conv_silu", []string{"x", "w"}, []string{"y"}, convSiluSource, ""),
		prep:      mlx.NewKernel("vakt_gated_delta_prep", []string{"conv", "proj", "aneg", "dtbias"}, []string{"scal"}, deltaPrepSource, kernelHeader),
		delta:     mlx.NewKernel("vakt_gated_delta_quads", []string{"conv", "scal"}, []string{"y"}, gatedDeltaSource, ""),
		norm:      mlx.NewKernel("vakt_gated_rmsnorm", []string{"core", "proj", "w"}, []string{"out"}, gatedNormSource, kernelHeader),
		swiglu:    mlx.NewKernel("vakt_swiglu", []string{"gu"}, []string{"out"}, swigluSource, kernelHeader),

		addNorm:  mlx.NewKernel("vakt_add_rmsnorm", []string{"r", "d", "w"}, []string{"h", "n"}, addNormSource, ""),
		attnPrep: mlx.NewKernel("vakt_attn_prep", []string{"proj", "qw", "kw"}, []string{"q", "k", "v"}, attnPrepSource, ""),
		attnGate: mlx.NewKernel("vakt_attn_gate", []string{"o", "proj"}, []string{"out"}, attnGateSource, kernelHeader),
	}
}

func (k *kernels) free() {
	if k == nil {
		return
	}
	k.conv.Free()
	k.prep.Free()
	k.delta.Free()
	k.norm.Free()
	k.swiglu.Free()
	k.addNorm.Free()
	k.attnPrep.Free()
	k.attnGate.Free()
}

// Column offsets of the fused linear-attention projection qkv|z|b|a.
const (
	projZOff = linQKVDim                 // 6144
	projBOff = linQKVDim + linValueWidth // 8192 (b), then a at +linHeads
)

// convSteps is the number of consecutive timesteps one conv thread computes
// from a sliding window of convSteps+3 input rows: 1 (default) or
// VAKT_CONV_STEPS. More steps read each input row fewer times but leave
// fewer threads; which wins is a measurement (scripts/abbench.sh).
func convSteps() int {
	if v, err := strconv.Atoi(os.Getenv("VAKT_CONV_STEPS")); err == nil && v >= 1 && v <= 16 {
		return v
	}
	return 1
}

// convSilu runs the depthwise conv + SiLU over the q|k|v columns of proj
// [B,T,P]. The output [B,T,6144] keeps proj's dtype.
func (k *kernels) convSilu(x *mlx.Ctx, proj, taps *mlx.Array, B, T int) *mlx.Array {
	dt := proj.Dtype()
	steps := k.convSteps
	return x.Apply(k.conv, []*mlx.Array{proj, taps}, mlx.KernelLaunch{
		Grid:           [3]int{linQKVDim / 4, (T + steps - 1) / steps, B}, // 4 channels x steps timesteps per thread
		ThreadGroup:    [3]int{256, 1, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, linQKVDim}, Dtype: dt}},
		TemplateInts:   map[string]int{"K": convKernel, "C": linQKVDim, "TT": steps},
		TemplateDtypes: map[string]mlx.DType{"OutT": dt},
	})[0]
}

// deltaPrep computes the per-(token, head) scalars: float32 [B,T,16,4].
func (k *kernels) deltaPrep(x *mlx.Ctx, conv, proj *mlx.Array, L layerW, B, T int) *mlx.Array {
	return x.Apply(k.prep, []*mlx.Array{conv, proj, L.aLogNeg, L.dtBias}, mlx.KernelLaunch{
		Grid:           [3]int{32, linHeads, B * T},
		ThreadGroup:    [3]int{32, linHeads, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, linHeads, 4}, Dtype: mlx.Float32}},
		TemplateInts:   map[string]int{"H": linHeads, "Dk": linKeyDim, "BOFF": projBOff},
		TemplateDtypes: map[string]mlx.DType{"InT": conv.Dtype(), "ProjT": proj.Dtype()},
	})[0]
}

// gatedDelta runs the recurrence over conv with deltaPrep's scalars.
// Returns float32 [B,T,16,128].
func (k *kernels) gatedDelta(x *mlx.Ctx, conv, scal *mlx.Array, B, T int) *mlx.Array {
	return x.Apply(k.delta, []*mlx.Array{conv, scal}, mlx.KernelLaunch{
		Grid:           [3]int{32, linValDim / 8, B * linHeads}, // 8 columns per SIMD group
		ThreadGroup:    [3]int{32, 4, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, linHeads, linValDim}, Dtype: mlx.Float32}},
		TemplateInts:   map[string]int{"H": linHeads, "Dk": linKeyDim, "Dv": linValDim},
		TemplateDtypes: map[string]mlx.DType{"InT": conv.Dtype()},
	})[0]
}

// gatedNorm applies RMSNormGated with silu(z) from proj and casts to the
// compute dtype: [B,T,2048].
func (k *kernels) gatedNorm(x *mlx.Ctx, core, proj, w *mlx.Array, out mlx.DType, B, T int) *mlx.Array {
	return x.Apply(k.norm, []*mlx.Array{core, proj, w}, mlx.KernelLaunch{
		Grid:           [3]int{32, linHeads, B * T},
		ThreadGroup:    [3]int{32, linHeads, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, linValueWidth}, Dtype: out}},
		TemplateInts:   map[string]int{"H": linHeads, "Dv": linValDim, "ZOFF": projZOff},
		TemplateDtypes: map[string]mlx.DType{"OutT": out, "ProjT": proj.Dtype()},
	})[0]
}

// swigluOp is silu(gate) * up over the fused [.., 2I] projection.
func (k *kernels) swigluOp(x *mlx.Ctx, gu *mlx.Array, B, T int) *mlx.Array {
	dt := gu.Dtype()
	return x.Apply(k.swiglu, []*mlx.Array{gu}, mlx.KernelLaunch{
		Grid:           [3]int{intermediate / 4, B * T, 1}, // 4 elements per thread
		ThreadGroup:    [3]int{256, 1, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, intermediate}, Dtype: dt}},
		TemplateInts:   map[string]int{"I": intermediate},
		TemplateDtypes: map[string]mlx.DType{"OutT": dt},
	})[0]
}

// addRMSNorm returns h = r + d (float32) and the RMSNorm of h with scale
// (1+w folded) in dtype out. r is float32 [B,T,H].
func (k *kernels) addRMSNorm(x *mlx.Ctx, r, d, scale *mlx.Array, out mlx.DType, B, T int) (*mlx.Array, *mlx.Array) {
	o := x.Apply(k.addNorm, []*mlx.Array{r, d, scale}, mlx.KernelLaunch{
		Grid:        [3]int{256, B * T, 1},
		ThreadGroup: [3]int{256, 1, 1},
		Outputs: []mlx.KernelOutput{
			{Shape: []int{B, T, hidden}, Dtype: mlx.Float32},
			{Shape: []int{B, T, hidden}, Dtype: out},
		},
		TemplateInts:   map[string]int{"Hd": hidden},
		TemplateDtypes: map[string]mlx.DType{"OutT": out},
	})
	return o[0], o[1]
}

// attnPrepOp returns q [B,8,T,256], k and v [B,2,T,256] from the fused q|k|v
// projection, normed and rotated.
func (k *kernels) attnPrepOp(x *mlx.Ctx, proj, qw, kw *mlx.Array, B, T int) (q, kk, v *mlx.Array) {
	dt := proj.Dtype()
	o := x.Apply(k.attnPrep, []*mlx.Array{proj, qw, kw}, mlx.KernelLaunch{
		Grid:        [3]int{32, attnHeads + 2*kvHeads, B * T},
		ThreadGroup: [3]int{32, attnHeads + 2*kvHeads, 1},
		Outputs: []mlx.KernelOutput{
			{Shape: []int{B, attnHeads, T, headDim}, Dtype: dt},
			{Shape: []int{B, kvHeads, T, headDim}, Dtype: dt},
			{Shape: []int{B, kvHeads, T, headDim}, Dtype: dt},
		},
		TemplateInts: map[string]int{
			"HQ": attnHeads, "HK": kvHeads, "D": headDim, "RD": ropeDims,
		},
		TemplateDtypes: map[string]mlx.DType{"InT": dt, "OutT": dt},
	})
	return o[0], o[1], o[2]
}

// attnPrepSource keeps RoPE's pair in one lane only when RD/2 == 32.
var (
	_ [ropeDims - 64]struct{}
	_ [64 - ropeDims]struct{}
)

// attnGateOp is o * sigmoid(gate) laid out [B,T,8*256] for o_proj.
func (k *kernels) attnGateOp(x *mlx.Ctx, o, proj *mlx.Array, B, T int) *mlx.Array {
	dt := proj.Dtype()
	return x.Apply(k.attnGate, []*mlx.Array{o, proj}, mlx.KernelLaunch{
		Grid:           [3]int{attnHeads * headDim, B * T, 1},
		ThreadGroup:    [3]int{256, 1, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, attnHeads * headDim}, Dtype: dt}},
		TemplateInts:   map[string]int{"HQ": attnHeads, "D": headDim},
		TemplateDtypes: map[string]mlx.DType{"OutT": dt},
	})[0]
}
