//go:build mlx

package engine

import (
	"fmt"
	"math"
	"strconv"

	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine/mlx"
)

// Qwen3.5-0.8B text tower constants (Qwen3.5-0.8B-Base/config.json).
const (
	hidden        = core.HiddenSize // 1024
	intermediate  = 3584
	numLayers     = core.NumLayers // 24
	fullInterval  = 4              // layers 3,7,...,23 are full attention
	rmsEps        = 1e-6
	attnHeads     = 8
	kvHeads       = 2
	headDim       = 256
	ropeDims      = 64 // partial_rotary_factor 0.25 * head_dim
	ropeTheta     = 1e7
	linHeads      = 16 // linear_num_{key,value}_heads
	linKeyDim     = 128
	linValDim     = 128
	convKernel    = 4
	linQKVDim     = 2*linHeads*linKeyDim + linHeads*linValDim // 6144
	linValueWidth = linHeads * linValDim                      // 2048
)

func isFull(layer int) bool { return layer%fullInterval == fullInterval-1 }

// weights holds the model tensors in their compute dtype, with the fused and
// transposed layouts the forward uses.
type weights struct {
	embed   *mlx.Array // [V, H]
	norm    *mlx.Array // [H] (1+w applied)
	layers  []layerW
	binW    *mlx.Array // [H, 1] f32
	binB    *mlx.Array // [1]
	auxW    *mlx.Array // [H, 18] f32
	auxB    *mlx.Array // [18]
	all     []*mlx.Array
	compute mlx.DType
}

type layerW struct {
	inNorm, postNorm *mlx.Array // [H], (1+w)
	gateUp           *mlx.Array // [H, 2*I] fused gate|up, transposed
	down             *mlx.Array // [I, H]

	// linear attention
	inProj  *mlx.Array // [H, 6144+2048+16+16] fused qkv|z|b|a, transposed
	convW   *mlx.Array // [6144, K, 1] (MLX depthwise layout)
	aLogNeg *mlx.Array // [16] = -exp(A_log), f32
	dtBias  *mlx.Array // [16] f32
	gnorm   *mlx.Array // [128] plain w (RMSNormGated)
	outProj *mlx.Array // [2048, H]

	// full attention
	qkv   *mlx.Array // [H, 4096+512+512] fused q(+gate)|k|v, transposed
	qNorm *mlx.Array // [256], (1+w)
	kNorm *mlx.Array // [256], (1+w)
	o     *mlx.Array // [2048, H]
}

// loadWeights builds compute-layout weights from raw tensors. prefix is
// "backbone." (published model) or "model.language_model." (base model; no
// heads). Raw arrays are released as they are consumed.
func loadWeights(x *mlx.Ctx, raw map[string]*mlx.Array, prefix string, compute mlx.DType, heads bool) (*weights, error) {
	get := func(name string) *mlx.Array {
		a := raw[name]
		if a == nil {
			x.Fail(fmt.Errorf("missing tensor %s", name))
			return x.Zeros(mlx.Float32, 1)
		}
		return a
	}
	w := &weights{compute: compute}
	keep := func(a *mlx.Array) *mlx.Array {
		x.Keep(a)
		w.all = append(w.all, a)
		return a
	}
	cast := func(a *mlx.Array) *mlx.Array { return x.AsType(a, compute) }
	f32 := func(a *mlx.Array) *mlx.Array { return x.AsType(a, mlx.Float32) }
	// Qwen3_5RMSNorm multiplies by (1 + w), computed in float32.
	onePlus := func(a *mlx.Array) *mlx.Array { return x.Add(f32(a), x.Scalar(1)) }
	lin := func(names ...string) *mlx.Array { // PyTorch [out,in] -> fused [in, sum(out)]
		parts := make([]*mlx.Array, len(names))
		for i, n := range names {
			parts[i] = get(n)
		}
		cat := parts[0]
		if len(parts) > 1 {
			cat = x.Concatenate(0, parts...)
		}
		return x.Contiguous(x.Transpose(cast(cat), 1, 0))
	}
	p := prefix
	w.embed = keep(x.Contiguous(cast(get(p + "embed_tokens.weight"))))
	w.norm = keep(onePlus(get(p + "norm.weight")))
	for i := 0; i < numLayers; i++ {
		lp := p + "layers." + strconv.Itoa(i) + "."
		L := layerW{
			inNorm:   keep(onePlus(get(lp + "input_layernorm.weight"))),
			postNorm: keep(onePlus(get(lp + "post_attention_layernorm.weight"))),
			gateUp:   keep(lin(lp+"mlp.gate_proj.weight", lp+"mlp.up_proj.weight")),
			down:     keep(lin(lp + "mlp.down_proj.weight")),
		}
		if isFull(i) {
			a := lp + "self_attn."
			L.qkv = keep(lin(a+"q_proj.weight", a+"k_proj.weight", a+"v_proj.weight"))
			L.qNorm = keep(onePlus(get(a + "q_norm.weight")))
			L.kNorm = keep(onePlus(get(a + "k_norm.weight")))
			L.o = keep(lin(a + "o_proj.weight"))
		} else {
			a := lp + "linear_attn."
			L.inProj = keep(lin(a+"in_proj_qkv.weight", a+"in_proj_z.weight", a+"in_proj_b.weight", a+"in_proj_a.weight"))
			// PyTorch depthwise conv weight [C, 1, K] -> MLX [C, K, 1].
			L.convW = keep(x.Contiguous(x.Transpose(cast(get(a+"conv1d.weight")), 0, 2, 1)))
			L.aLogNeg = keep(x.Negative(x.Exp(f32(get(a + "A_log")))))
			L.dtBias = keep(f32(get(a + "dt_bias")))
			L.gnorm = keep(f32(get(a + "norm.weight")))
			L.outProj = keep(lin(a + "out_proj.weight"))
		}
		w.layers = append(w.layers, L)
	}
	if heads {
		w.binW = keep(x.Contiguous(x.Transpose(f32(get("binary_head.weight")), 1, 0)))
		w.binB = keep(f32(get("binary_head.bias")))
		w.auxW = keep(x.Contiguous(x.Transpose(f32(get("auxiliary_head.weight")), 1, 0)))
		w.auxB = keep(f32(get("auxiliary_head.bias")))
	}
	if err := x.Eval(w.all...); err != nil {
		return nil, err
	}
	x.Free() // intermediates (raw casts, concatenations) are no longer needed
	for _, a := range raw {
		a.Free()
	}
	return w, nil
}

func (w *weights) free() {
	for _, a := range w.all {
		a.Free()
	}
}

// deltaMode selects the gated-delta implementation.
type deltaMode int

const (
	deltaKernel deltaMode = iota // Metal kernel (GPU on darwin)
	deltaScan                    // exact per-token scan with plain ops (any device)
)

// model is the forward pass.
type model struct {
	w     *weights
	delta deltaMode
	kern  *mlx.Kernel
}

// forward returns the float32 last hidden state after the final norm,
// [B, T, H], for right-padded ids [B, T] with per-row lengths.
func (m *model) forward(x *mlx.Ctx, ids *mlx.Array, mask *mlx.Array, lengths []int, B, T int) *mlx.Array {
	w := m.w
	h := x.Take(w.embed, ids, 0) // [B, T, H]
	for i, L := range w.layers {
		r := h
		n := m.rmsNorm(x, h, L.inNorm)
		var mixed *mlx.Array
		if isFull(i) {
			mixed = m.attention(x, n, L, B, T)
		} else {
			mixed = m.linearAttention(x, n, mask, L, lengths, B, T)
		}
		h = x.Add(r, mixed)
		r = h
		n = m.rmsNorm(x, h, L.postNorm)
		h = x.Add(r, m.mlp(x, n, L))
	}
	return m.rmsNorm(x, h, w.norm)
}

// rmsNorm matches Qwen3_5RMSNorm: normalise in float32, multiply by (1+w)
// (pre-folded into scale), cast back to the input dtype.
func (m *model) rmsNorm(x *mlx.Ctx, h, scale *mlx.Array) *mlx.Array {
	dt := h.Dtype()
	return x.AsType(x.RMSNorm(x.AsType(h, mlx.Float32), scale, rmsEps), dt)
}

func (m *model) mlp(x *mlx.Ctx, h *mlx.Array, L layerW) *mlx.Array {
	gu := x.Matmul(h, L.gateUp) // [B, T, 2I]
	parts := x.SplitAt(gu, -1, intermediate)
	return x.Matmul(x.Multiply(x.Silu(parts[0]), parts[1]), L.down)
}

// attention is Qwen3_5Attention: gated output, q/k RMSNorm per head, partial
// RoPE (first 64 of 256 dims, rotate_half), GQA, causal.
func (m *model) attention(x *mlx.Ctx, h *mlx.Array, L layerW, B, T int) *mlx.Array {
	proj := x.Matmul(h, L.qkv) // [B, T, 4096+512+512]
	p := x.SplitAt(proj, -1, attnHeads*headDim*2, attnHeads*headDim*2+kvHeads*headDim)
	// q_proj output is viewed as [.., heads, 2*head_dim] then chunked: the
	// first head_dim of each head is the query, the second is the gate.
	qg := x.Reshape(p[0], B, T, attnHeads, 2*headDim)
	qgs := x.SplitAt(qg, -1, headDim)
	q, gate := qgs[0], x.Reshape(qgs[1], B, T, attnHeads*headDim)
	k := x.Reshape(p[1], B, T, kvHeads, headDim)
	v := x.Reshape(p[2], B, T, kvHeads, headDim)

	q = m.rmsNorm(x, q, L.qNorm)
	k = m.rmsNorm(x, k, L.kNorm)
	q = x.Transpose(q, 0, 2, 1, 3) // [B, Hq, T, D]
	k = x.Transpose(k, 0, 2, 1, 3)
	v = x.Transpose(v, 0, 2, 1, 3)
	q = x.RoPE(q, ropeDims, false, ropeTheta, 1, 0)
	k = x.RoPE(k, ropeDims, false, ropeTheta, 1, 0)

	// Right padding + causal mask: real queries never see padded keys.
	o := x.SDPA(q, k, v, float32(1/math.Sqrt(headDim)), "causal", nil)
	o = x.Reshape(x.Transpose(o, 0, 2, 1, 3), B, T, attnHeads*headDim)
	o = x.Multiply(o, x.Sigmoid(gate))
	return x.Matmul(o, L.o)
}

// linearAttention is Qwen3_5GatedDeltaNet.
func (m *model) linearAttention(x *mlx.Ctx, h, mask *mlx.Array, L layerW, lengths []int, B, T int) *mlx.Array {
	// apply_mask_to_padding_states: zero padded positions first.
	h = x.Multiply(h, x.AsType(mask, h.Dtype()))
	proj := x.Matmul(h, L.inProj) // [B, T, 6144+2048+16+16]
	p := x.SplitAt(proj, -1, linQKVDim, linQKVDim+linValueWidth, linQKVDim+linValueWidth+linHeads)
	mixed, z, bb, aa := p[0], p[1], p[2], p[3]

	// Depthwise causal conv (kernel 4) + SiLU over the q|k|v channels.
	conv := x.Conv1d(mixed, L.convW, 1, convKernel-1, 1, linQKVDim)
	conv = x.Slice(conv, []int{0, 0, 0}, []int{B, T, linQKVDim}, nil)
	conv = x.Silu(conv)

	qkv := x.SplitAt(conv, -1, linHeads*linKeyDim, 2*linHeads*linKeyDim)
	q := x.Reshape(x.AsType(qkv[0], mlx.Float32), B, T, linHeads, linKeyDim)
	k := x.Reshape(x.AsType(qkv[1], mlx.Float32), B, T, linHeads, linKeyDim)
	v := x.Reshape(x.AsType(qkv[2], mlx.Float32), B, T, linHeads, linValDim)

	beta := x.Sigmoid(x.AsType(bb, mlx.Float32))                                        // [B, T, 16]
	g := x.Multiply(L.aLogNeg, x.Softplus(x.Add(x.AsType(aa, mlx.Float32), L.dtBias))) // [B, T, 16]

	// l2norm(q), l2norm(k) with eps 1e-6, then q *= 1/sqrt(dk).
	q = l2norm(x, q)
	k = l2norm(x, k)
	q = x.Multiply(q, x.Scalar(float32(1/math.Sqrt(linKeyDim))))

	var core *mlx.Array // [B, T, 16, 128] f32
	if m.delta == deltaKernel && m.kern != nil {
		core = m.deltaKernel(x, q, k, v, g, beta, B, T)
	} else {
		core = deltaScanOps(x, q, k, v, g, beta, B, T)
	}

	// Qwen3_5RMSNormGated: rmsnorm(core) * w (plain) * silu(z), in float32.
	zf := x.Reshape(x.AsType(z, mlx.Float32), B, T, linHeads, linValDim)
	o := x.Multiply(x.RMSNorm(core, L.gnorm, rmsEps), x.Silu(zf))
	o = x.AsType(x.Reshape(o, B, T, linValueWidth), h.Dtype())
	return x.Matmul(o, L.outProj)
}

func l2norm(x *mlx.Ctx, a *mlx.Array) *mlx.Array {
	ss := x.Sum(x.Square(a), true, -1)
	return x.Multiply(a, x.Rsqrt(x.Add(ss, x.Scalar(1e-6))))
}

// deltaScanOps is the exact recurrence (torch_recurrent_gated_delta_rule)
// written with plain ops, one token at a time. It is the portable path and
// the reference for the kernel. S is [B, Hv, Dk, Dv].
func deltaScanOps(x *mlx.Ctx, q, k, v, g, beta *mlx.Array, B, T int) *mlx.Array {
	S := x.Zeros(mlx.Float32, B, linHeads, linKeyDim, linValDim)
	decay := x.Exp(g) // [B, T, H]
	outs := make([]*mlx.Array, T)
	for t := 0; t < T; t++ {
		at := func(a *mlx.Array, d int) *mlx.Array { // [B, 1, H, d] -> [B, H, d]
			return x.Reshape(x.Slice(a, []int{0, t, 0, 0}, []int{B, t + 1, linHeads, d}, nil), B, linHeads, d)
		}
		qt, kt, vt := at(q, linKeyDim), at(k, linKeyDim), at(v, linValDim)
		dt := x.Reshape(x.Slice(decay, []int{0, t, 0}, []int{B, t + 1, linHeads}, nil), B, linHeads, 1, 1)
		bt := x.Reshape(x.Slice(beta, []int{0, t, 0}, []int{B, t + 1, linHeads}, nil), B, linHeads, 1)
		S = x.Multiply(S, dt)
		kcol := x.Reshape(kt, B, linHeads, linKeyDim, 1)
		kvMem := x.Sum(x.Multiply(S, kcol), false, 2) // [B, H, Dv]
		delta := x.Multiply(x.Subtract(vt, kvMem), bt)
		S = x.Add(S, x.Multiply(kcol, x.Reshape(delta, B, linHeads, 1, linValDim)))
		qcol := x.Reshape(qt, B, linHeads, linKeyDim, 1)
		outs[t] = x.Reshape(x.Sum(x.Multiply(S, qcol), false, 2), B, 1, linHeads, linValDim)
	}
	return x.Concatenate(1, outs...)
}
