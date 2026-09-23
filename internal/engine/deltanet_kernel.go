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
