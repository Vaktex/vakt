//go:build mlx

package engine

import (
	"github.com/vaktex/vakt/internal/engine/mlx"
)

// gatedDeltaSource is the gated delta rule recurrence as a Metal kernel,
// after the design of mlx-lm's gated_delta kernel (MIT, ml-explore/mlx-lm):
// one SIMD group per (batch, head, value column); each of the 32 lanes owns
// Dk/32 rows of that state column in registers, and the two reductions over
// Dk (S^T k and S^T q) are simd_sums. It computes exactly
// torch_recurrent_gated_delta_rule, token by token, in float32.
//
// Inputs are contiguous float32: q, k [B,T,H,Dk], v [B,T,H,Dv], g, beta
// [B,T,H] (g in log space; decay = exp(g)). Output y [B,T,H,Dv].
const gatedDeltaSource = `
    const uint lane = thread_position_in_grid.x;
    const uint j = thread_position_in_grid.y;
    const uint bh = thread_position_in_grid.z;
    const uint b = bh / H;
    const uint h = bh % H;
    constexpr int NP = Dk / 32;
    float state[NP];
    for (int i = 0; i < NP; ++i) { state[i] = 0.0f; }
    for (int t = 0; t < T; ++t) {
        const size_t base = (size_t(b) * T + t) * H + h;
        const float dcy = metal::precise::exp(g[base]);
        const float bt = beta[base];
        const device float* kt = k + base * Dk;
        const device float* qt = q + base * Dk;
        float kv = 0.0f;
        for (int i = 0; i < NP; ++i) {
            const int s = lane * NP + i;
            state[i] *= dcy;
            kv += state[i] * kt[s];
        }
        kv = simd_sum(kv);
        const float dl = (v[base * Dv + j] - kv) * bt;
        float o = 0.0f;
        for (int i = 0; i < NP; ++i) {
            const int s = lane * NP + i;
            state[i] += kt[s] * dl;
            o += state[i] * qt[s];
        }
        o = simd_sum(o);
        if (lane == 0) { y[base * Dv + j] = o; }
    }
`

// convSiluSource is the causal depthwise conv (kernel K) followed by SiLU,
// one thread per output element: y[b,t,c] = silu(sum_k w[k,c] * x[b,t-K+1+k,c]),
// accumulated in float32. x is [B,T,C], w is [K,C].
const convSiluSource = `
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint b = thread_position_in_grid.z;
    if (c >= C || t >= T) { return; }
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int src = int(t) - (K - 1) + k;
        if (src >= 0) {
            acc += float(w[k * C + c]) * float(x[(size_t(b) * T + src) * C + c]);
        }
    }
    const float s = acc / (1.0f + metal::precise::exp(-acc));
    y[(size_t(b) * T + t) * C + c] = static_cast<OutT>(s);
`

func newConvKernel() *mlx.Kernel {
	return mlx.NewKernel("vakt_causal_conv_silu", []string{"x", "w"}, []string{"y"}, convSiluSource, "")
}

func (m *model) convSilu(x *mlx.Ctx, in *mlx.Array, taps *mlx.Array, B, T, C int) *mlx.Array {
	out := x.Apply(m.convKern, []*mlx.Array{x.Contiguous(in), taps}, mlx.KernelLaunch{
		Grid:           [3]int{C, T, B},
		ThreadGroup:    [3]int{256, 1, 1},
		Outputs:        []mlx.KernelOutput{{Shape: []int{B, T, C}, Dtype: in.Dtype()}},
		TemplateInts:   map[string]int{"T": T, "C": C, "K": convKernel},
		TemplateDtypes: map[string]mlx.DType{"OutT": in.Dtype()},
	})
	return out[0]
}

func newDeltaKernel() *mlx.Kernel {
	return mlx.NewKernel("vakt_gated_delta", []string{"q", "k", "v", "g", "beta"}, []string{"y"}, gatedDeltaSource, "")
}

func (m *model) deltaKernel(x *mlx.Ctx, q, k, v, g, beta *mlx.Array, B, T int) *mlx.Array {
	c := x.Contiguous
	out := x.Apply(m.kern, []*mlx.Array{c(q), c(k), c(v), c(g), c(beta)}, mlx.KernelLaunch{
		Grid:        [3]int{32, linValDim, B * linHeads},
		ThreadGroup: [3]int{32, 4, 1},
		Outputs:     []mlx.KernelOutput{{Shape: []int{B, T, linHeads, linValDim}, Dtype: mlx.Float32}},
		TemplateInts: map[string]int{
			"T": T, "H": linHeads, "Dk": linKeyDim, "Dv": linValDim,
		},
	})
	return out[0]
}
