//go:build mlx

package engine

import (
	"github.com/vaktex/vakt/internal/engine/mlx"
)

// The Metal path of the Gated DeltaNet layer runs as three custom kernels
// between the in_proj and out_proj matmuls:
//
//	proj ─► convSilu ─► gatedDelta ─► gatedNorm ─► out_proj
//	  └──────────────────┴──(b, a)──────┘ (z)
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
// one thread per output element: y[b,t,c] = silu(sum_k w[k,c] * x[b,t-K+1+k,c]),
// accumulated in float32. x is the fused projection [B,T,P] (only the first
// C columns are read), w is [K,C], y is [B,T,C].
const convSiluSource = `
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    const int T = x_shape[1];
    const int P = x_shape[2];
    if (int(c) >= C || int(t) >= T) { return; }
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int src = int(t) - (K - 1) + k;
        if (src >= 0) {
            acc += float(w[k * C + c]) * float(x[(size_t(b) * T + src) * P + c]);
        }
    }
    const float s = acc / (1.0f + metal::precise::exp(-acc));
    y[(size_t(b) * T + t) * C + c] = static_cast<OutT>(s);
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

// gatedDeltaSource is the gated delta rule recurrence, after the design of
// mlx-lm's gated_delta kernel (MIT, ml-explore/mlx-lm): one SIMD group per
// (batch, head, CPG value columns); each of the 32 lanes owns Dk/32 rows of
// those state columns in registers, and the reductions over Dk are
// simd_sums. It computes exactly torch_recurrent_gated_delta_rule, token by
// token, in float32:
//
//	q, k = l2norm(conv q, k), q *= 1/sqrt(Dk)   (norms from scal)
//
// Handling CPG columns per SIMD group loads each token's q and k once per
// CPG columns instead of once per column. conv is [B,T,3*H*Dk] (q|k|v);
// scal is the deltaPrep output [B,T,H,4]. Output y is float32 [B,T,H,Dv].
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
    constexpr int NP = Dk / 32;
    const float qscale = metal::precise::rsqrt(float(Dk));
    float state[CPG][NP];
    for (int c = 0; c < CPG; ++c) {
        for (int i = 0; i < NP; ++i) { state[c][i] = 0.0f; }
    }
    for (int t = 0; t < T; ++t) {
        const size_t row = size_t(b) * T + t;
        const device InT* cr = conv + row * CW;
        const device float* sc = scal + (row * H + h) * 4;
        const float qinv = sc[0];
        const float kinv = sc[1];
        const float bt = sc[2];
        const float dcy = sc[3];
        float kr[NP], qr[NP];
        for (int i = 0; i < NP; ++i) {
            const int s = lane * NP + i;
            qr[i] = (float(cr[h * Dk + s]) * qinv) * qscale;
            kr[i] = float(cr[H * Dk + h * Dk + s]) * kinv;
        }
        float kv[CPG];
        for (int c = 0; c < CPG; ++c) {
            kv[c] = 0.0f;
            for (int i = 0; i < NP; ++i) {
                state[c][i] *= dcy;
                kv[c] += state[c][i] * kr[i];
            }
        }
        float o[CPG];
        for (int c = 0; c < CPG; ++c) {
            const int j = jg * CPG + c;
            const float dl = (float(cr[2 * H * Dk + h * Dv + j]) - simd_sum(kv[c])) * bt;
            o[c] = 0.0f;
            for (int i = 0; i < NP; ++i) {
                state[c][i] += kr[i] * dl;
                o[c] += state[c][i] * qr[i];
            }
        }
        for (int c = 0; c < CPG; ++c) {
            const float oc = simd_sum(o[c]);
            if (lane == 0) { y[(row * H + h) * Dv + jg * CPG + c] = oc; }
        }
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

// kernels are the Metal kernels of the fused path (nil off Metal).
type kernels struct {
	conv, prep, delta, norm, swiglu *mlx.Kernel
}

func newKernels() *kernels {
	return &kernels{
		conv:   mlx.NewKernel("vakt_causal_conv_silu", []string{"x", "w"}, []string{"y"}, convSiluSource, ""),
		prep:   mlx.NewKernel("vakt_gated_delta_prep", []string{"conv", "proj", "aneg", "dtbias"}, []string{"scal"}, deltaPrepSource, kernelHeader),
		delta:  mlx.NewKernel("vakt_gated_delta_cols", []string{"conv", "scal"}, []string{"y"}, gatedDeltaSource, ""),
		norm:   mlx.NewKernel("vakt_gated_rmsnorm", []string{"core", "proj", "w"}, []string{"out"}, gatedNormSource, kernelHeader),
		swiglu: mlx.NewKernel("vakt_swiglu", []string{"gu"}, []string{"out"}, swigluSource, kernelHeader),
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
}

// Column offsets of the fused linear-attention projection qkv|z|b|a.
const (
	projZOff = linQKVDim                 // 6144
	projBOff = linQKVDim + linValueWidth // 8192 (b), then a at +linHeads
)

// convSilu runs the depthwise conv + SiLU over the q|k|v columns of proj
// [B,T,P]. The output [B,T,6144] keeps proj's dtype.
func (k *kernels) convSilu(x *mlx.Ctx, proj, taps *mlx.Array, B, T int) *mlx.Array {
	dt := proj.Dtype()
	return x.Apply(k.conv, []*mlx.Array{proj, taps}, mlx.KernelLaunch{
		Grid:           [3]int{linQKVDim, T, B},
		ThreadGroup:    [3]int{256, 1, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, linQKVDim}, Dtype: dt}},
		TemplateInts:   map[string]int{"K": convKernel, "C": linQKVDim},
		TemplateDtypes: map[string]mlx.DType{"OutT": dt},
	})[0]
}

// deltaCols is the number of value columns one SIMD group of the
// recurrence carries (state registers per lane: deltaCols * Dk/32).
const deltaCols = 4

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
		Grid:           [3]int{32, linValDim / deltaCols, B * linHeads},
		ThreadGroup:    [3]int{32, 8, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, linHeads, linValDim}, Dtype: mlx.Float32}},
		TemplateInts:   map[string]int{"H": linHeads, "Dk": linKeyDim, "Dv": linValDim, "CPG": deltaCols},
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
