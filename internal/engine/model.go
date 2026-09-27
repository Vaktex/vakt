//go:build mlx

package engine

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine/mlx"
)

// Text tower constants, from the base model config.json.
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

// poolW is AttentionPool (experiment/pooling.py), all float32.
type poolW struct {
	query  *mlx.Array // [4, 256]
	keyT   *mlx.Array // [H, H] (transposed, no bias)
	valueT *mlx.Array // [H, H] (transposed, no bias)
	projT  *mlx.Array // [H, H] (transposed)
	projB  *mlx.Array // [H]
	normW  *mlx.Array // [H]
	normB  *mlx.Array // [H]
}

// headW is MLPHead (experiment/heads.py), all float32:
// LayerNorm -> Linear -> GELU(erf) -> Dropout(identity) -> Linear.
type headW struct {
	lnW, lnB *mlx.Array // [H]
	w1T, b1  *mlx.Array // [H, 2H], [2H]
	w2T, b2  *mlx.Array // [2H, out], [out]
}

// weights holds the model tensors in their compute dtype, with the fused and
// transposed layouts the forward uses.
type weights struct {
	embed   *mlx.Array // [V, H]
	norm    *mlx.Array // [H] (1+w applied)
	layers  []layerW
	pool    poolW
	bin     headW // severity (1 output)
	aux     headW // CWE families (18 outputs)
	heads   bool
	all     []*mlx.Array
	compute mlx.DType
}

type layerW struct {
	inNorm, postNorm *mlx.Array // [H], (1+w)
	gateUp           *mlx.Array // [H, 2*I] fused gate|up, transposed
	down             *mlx.Array // [I, H]

	// linear attention
	inProj     *mlx.Array   // [H, 6144+2048+16+16] fused qkv|z|b|a, transposed
	convTaps   []*mlx.Array // K arrays of [6144]: tap k of the depthwise conv
	convTapsKC *mlx.Array   // the same taps stacked [K, 6144] (Metal kernel)
	aLogNeg    *mlx.Array   // [16] = -exp(A_log), f32
	dtBias     *mlx.Array   // [16] f32
	gnorm      *mlx.Array   // [128] plain w (RMSNormGated)
	outProj    *mlx.Array   // [2048, H]

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
	var used []string // raw tensors consumed since the last materialise
	if raw["binary_head.weight"] != nil && raw["pool.query"] == nil {
		return nil, errors.New("engine: this checkpoint uses the pre-release head layout (mean pool + linear heads); this version of vakt needs the published DOM-0.8B (attention pool + MLP heads)")
	}
	get := func(name string) *mlx.Array {
		a := raw[name]
		if a == nil {
			x.Fail(fmt.Errorf("missing tensor %s", name))
			return x.Zeros(mlx.Float32, 1)
		}
		used = append(used, name)
		return a
	}
	w := &weights{compute: compute}
	// materialise evaluates the weights built so far and releases the raw
	// tensors they came from, so peak memory is the laid-out model plus one
	// layer's raw tensors rather than twice the model (a 3 GB fp32
	// checkpoint otherwise peaked at 6.6 GB and was OOM-killed on 7 GB
	// hosts).
	materialise := func() error {
		if err := x.Eval(w.all...); err != nil {
			return err
		}
		for _, n := range used {
			if a := raw[n]; a != nil {
				a.Free()
				delete(raw, n)
			}
		}
		used = used[:0]
		x.Free() // intermediates (casts, concatenations)
		return nil
	}
	keep := func(a *mlx.Array) *mlx.Array {
		x.Keep(a)
		w.all = append(w.all, a)
		return a
	}
	cast := func(a *mlx.Array) *mlx.Array { return x.AsType(a, compute) }
	f32 := func(a *mlx.Array) *mlx.Array { return x.AsType(a, mlx.Float32) }
	// The reference RMSNorm multiplies by (1 + w), computed in float32.
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
			// PyTorch depthwise conv weight [C, 1, K] -> K contiguous [C] taps.
			cw := cast(get(a + "conv1d.weight"))
			for k := 0; k < convKernel; k++ {
				tap := x.Reshape(x.Slice(cw, []int{0, 0, k}, []int{linQKVDim, 1, k + 1}, nil), linQKVDim)
				L.convTaps = append(L.convTaps, keep(x.Contiguous(tap)))
			}
			L.convTapsKC = keep(x.Contiguous(x.Transpose(x.Reshape(cw, linQKVDim, convKernel), 1, 0)))
			L.aLogNeg = keep(x.Negative(x.Exp(f32(get(a + "A_log")))))
			L.dtBias = keep(f32(get(a + "dt_bias")))
			L.gnorm = keep(f32(get(a + "norm.weight")))
			L.outProj = keep(lin(a + "out_proj.weight"))
		}
		w.layers = append(w.layers, L)
		if err := materialise(); err != nil {
			return nil, err
		}
	}
	if heads {
		// Pooling and heads stay float32 in every precision mode (the Python
		// side forces fp32 here: at 16k tokens a bf16 softmax keeps only 7%
		// of attention weights distinct).
		t32 := func(name string) *mlx.Array { return keep(x.Contiguous(x.Transpose(f32(get(name)), 1, 0))) }
		v32 := func(name string) *mlx.Array { return keep(f32(get(name))) }
		w.pool = poolW{
			query: v32("pool.query"), keyT: t32("pool.key.weight"), valueT: t32("pool.value.weight"),
			projT: t32("pool.project.weight"), projB: v32("pool.project.bias"),
			normW: v32("pool.norm.weight"), normB: v32("pool.norm.bias"),
		}
		head := func(p string) headW {
			return headW{
				lnW: v32(p + ".net.0.weight"), lnB: v32(p + ".net.0.bias"),
				w1T: t32(p + ".net.1.weight"), b1: v32(p + ".net.1.bias"),
				w2T: t32(p + ".net.4.weight"), b2: v32(p + ".net.4.bias"),
			}
		}
		w.bin, w.aux = head("binary_head"), head("auxiliary_head")
		w.heads = true
	}
	if err := x.Eval(w.all...); err != nil {
		return nil, err
	}
	if w.heads {
		// A NaN/Inf anywhere in the pool or heads turns every score into
		// NaN, which the report would show as 0 ("not vulnerable"). They
		// are small (~7M floats): check them all once at load.
		for name, a := range map[string]*mlx.Array{
			"pool.query": w.pool.query, "pool.key.weight": w.pool.keyT, "pool.value.weight": w.pool.valueT,
			"pool.project.weight": w.pool.projT, "pool.project.bias": w.pool.projB,
			"pool.norm.weight": w.pool.normW, "pool.norm.bias": w.pool.normB,
			"binary_head.net.0.weight": w.bin.lnW, "binary_head.net.0.bias": w.bin.lnB,
			"binary_head.net.1.weight": w.bin.w1T, "binary_head.net.1.bias": w.bin.b1,
			"binary_head.net.4.weight": w.bin.w2T, "binary_head.net.4.bias": w.bin.b2,
			"auxiliary_head.net.0.weight": w.aux.lnW, "auxiliary_head.net.0.bias": w.aux.lnB,
			"auxiliary_head.net.1.weight": w.aux.w1T, "auxiliary_head.net.1.bias": w.aux.b1,
			"auxiliary_head.net.4.weight": w.aux.w2T, "auxiliary_head.net.4.bias": w.aux.b2,
		} {
			v, err := x.Float32s(a)
			if err != nil {
				return nil, err
			}
			for _, f := range v {
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					return nil, fmt.Errorf("engine: tensor %s contains NaN or Inf", name)
				}
			}
		}
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
	deltaKernel    deltaMode = iota // fused Metal kernels (GPU on darwin)
	deltaChunkMode                  // chunked algorithm with plain ops (portable default)
	deltaScan                       // per-token scan with plain ops (test reference only)
)

// model is the forward pass.
type model struct {
	w         *weights
	cancelled func() bool // checked between layers
	delta     deltaMode
	kern      *kernels  // fused Metal kernels (nil off Metal)
	evalEvery int       // materialise the residual every N layers (0 = one graph)
	prof      *profiler // per-stage timing (VAKT_PROFILE=1), nil otherwise
}

// forward returns the float32 last hidden state after the final norm,
// [B, T, H], for right-padded ids [B, T] with per-row lengths.
//
// Layers are built in a scratch Ctx that is evaluated and released every
// evalEvery layers. A single lazy graph over all 24 layers keeps every
// intermediate alive until the end (gigabytes at long lengths) and was ~90x
// slower in practice; materialising the residual stream per layer keeps the
// working set to one layer's activations.
func (m *model) forward(x *mlx.Ctx, ids *mlx.Array, mask *mlx.Array, lengths []int, B, T int) *mlx.Array {
	w := m.w
	// The residual stream is carried in float32 in every precision mode
	// (as HF does under autocast); only matmul inputs use the compute dtype.
	m.prof.start()
	h := m.prof.mark(x, "embed", x.AsType(x.Take(w.embed, ids, 0), mlx.Float32)) // [B, T, H], owned by x
	lx := mlx.NewCtx(x.S)                                                        // per-segment scratch
	defer lx.Free()
	for i, L := range w.layers {
		if m.cancelled != nil && m.cancelled() {
			x.Fail(errCancelled)
			return x.Zeros(mlx.Float32, B, T, hidden)
		}
		r := h
		n := m.prof.mark(lx, "norm", m.rmsNorm(lx, h, L.inNorm))
		var mixed *mlx.Array
		if isFull(i) {
			mixed = m.attention(lx, n, L, B, T)
		} else {
			mixed = m.linearAttention(lx, n, mask, L, lengths, B, T)
		}
		h = m.prof.mark(lx, "residual", lx.Add(r, lx.AsType(mixed, mlx.Float32)))
		r = h
		n = m.prof.mark(lx, "norm", m.rmsNorm(lx, h, L.postNorm))
		h = m.prof.mark(lx, "residual", lx.Add(r, lx.AsType(m.mlp(lx, n, L, B, T), mlx.Float32)))
		if m.evalEvery > 0 && (i+1)%m.evalEvery == 0 {
			if err := lx.Eval(h); err != nil {
				x.Fail(err)
				return x.Zeros(mlx.Float32, B, T, hidden)
			}
			// Hand the residual to the caller's Ctx, drop everything else.
			lx.Keep(h)
			x.Adopt(h)
			lx.Free()
		}
	}
	if err := lx.Err(); err != nil {
		x.Fail(err)
	}
	out := m.prof.mark(lx, "norm", m.rmsNorm(lx, h, w.norm))
	if err := lx.Eval(out); err != nil {
		x.Fail(err)
		return x.Zeros(mlx.Float32, B, T, hidden)
	}
	lx.Keep(out)
	return x.Adopt(out)
}

// rmsNorm matches the reference RMSNorm: normalise in float32, multiply by (1+w)
// (pre-folded into scale). The output feeds matmuls, so it is cast to the
// compute dtype.
func (m *model) rmsNorm(x *mlx.Ctx, h, scale *mlx.Array) *mlx.Array {
	return x.AsType(x.RMSNorm(x.AsType(h, mlx.Float32), scale, rmsEps), m.w.compute)
}

func (m *model) mlp(x *mlx.Ctx, h *mlx.Array, L layerW, B, T int) *mlx.Array {
	gu := m.prof.mark(x, "mlp.gate_up", x.Matmul(h, L.gateUp)) // [B, T, 2I]
	if m.kern != nil {
		a := m.prof.mark(x, "mlp.act", m.kern.swigluOp(x, gu, B, T))
		return m.prof.mark(x, "mlp.down", x.Matmul(a, L.down))
	}
	parts := x.SplitAt(gu, -1, intermediate)
	return x.Matmul(x.Multiply(x.Silu(parts[0]), parts[1]), L.down)
}

// attention is gated attention: sigmoid-gated output, q/k RMSNorm per head, partial
// RoPE (first 64 of 256 dims, rotate_half), GQA, causal.
func (m *model) attention(x *mlx.Ctx, h *mlx.Array, L layerW, B, T int) *mlx.Array {
	proj := m.prof.mark(x, "attn.qkv", x.Matmul(h, L.qkv)) // [B, T, 4096+512+512]
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
	v = x.AsType(v, m.w.compute)
	q = x.Transpose(q, 0, 2, 1, 3) // [B, Hq, T, D]
	k = x.Transpose(k, 0, 2, 1, 3)
	v = x.Transpose(v, 0, 2, 1, 3)
	q = x.RoPE(q, ropeDims, false, ropeTheta, 1, 0)
	k = x.RoPE(k, ropeDims, false, ropeTheta, 1, 0)
	m.prof.mark(x, "attn.norm_rope", q)
	m.prof.mark(x, "attn.norm_rope", k)

	// Right padding + causal mask: real queries never see padded keys.
	o := m.prof.mark(x, "attn.sdpa", blockedCausalSDPA(x, q, k, v, float32(1/math.Sqrt(headDim)), B, T))
	o = x.Reshape(x.Transpose(o, 0, 2, 1, 3), B, T, attnHeads*headDim)
	o = m.prof.mark(x, "attn.gate", x.Multiply(o, x.Sigmoid(gate)))
	return m.prof.mark(x, "attn.o", x.Matmul(o, L.o))
}

// attnBlock bounds the query block of full attention. MLX 0.31.1 has no fused
// Metal kernel for head_dim 256 beyond 8 queries and falls back to
// materialising the [B, Hq, Tq, Tk] score matrix: at T=16k that is ~8.6 GB
// per layer. Blocking queries keeps it at [B, Hq, attnBlock, <=T].
const attnBlock = 1024

// blockedCausalSDPA computes causal attention one query block at a time.
// For queries [s, e) it attends to keys [0, e); MLX aligns the causal mask
// bottom-right when Tq < Tk, which is exactly this.
func blockedCausalSDPA(x *mlx.Ctx, q, k, v *mlx.Array, scale float32, B, T int) *mlx.Array {
	if T <= attnBlock {
		return x.SDPA(q, k, v, scale, "causal", nil)
	}
	var outs []*mlx.Array
	for s := 0; s < T; s += attnBlock {
		e := min(s+attnBlock, T)
		qb := x.Slice(q, []int{0, 0, s, 0}, []int{B, attnHeads, e, headDim}, nil)
		kb := x.Slice(k, []int{0, 0, 0, 0}, []int{B, kvHeads, e, headDim}, nil)
		vb := x.Slice(v, []int{0, 0, 0, 0}, []int{B, kvHeads, e, headDim}, nil)
		ob := x.SDPA(qb, kb, vb, scale, "causal", nil)
		// Evaluate each block so only one score matrix is alive at a time.
		if x.Eval(ob) != nil {
			return ob
		}
		outs = append(outs, ob)
	}
	return x.Concatenate(2, outs...)
}

// linearAttention is the Gated DeltaNet layer.
func (m *model) linearAttention(x *mlx.Ctx, h, mask *mlx.Array, L layerW, lengths []int, B, T int) *mlx.Array {
	// apply_mask_to_padding_states: zero padded positions first.
	h = x.Multiply(h, x.AsType(mask, h.Dtype()))
	proj := m.prof.mark(x, "lin.in_proj", x.Matmul(h, L.inProj)) // [B, T, 6144+2048+16+16]
	if m.delta == deltaKernel && m.kern != nil {
		// Metal: three fused kernels read their columns of proj in place.
		conv := m.prof.mark(x, "lin.conv", m.kern.convSilu(x, proj, L.convTapsKC, B, T))
		core := m.prof.mark(x, "lin.delta", m.kern.gatedDelta(x, conv, proj, L, B, T))
		o := m.prof.mark(x, "lin.gated_norm", m.kern.gatedNorm(x, core, proj, L.gnorm, h.Dtype(), B, T))
		return m.prof.mark(x, "lin.out_proj", x.Matmul(o, L.outProj))
	}
	p := x.SplitAt(proj, -1, linQKVDim, linQKVDim+linValueWidth, linQKVDim+linValueWidth+linHeads)
	mixed, z, bb, aa := p[0], p[1], p[2], p[3]

	// Depthwise causal conv (kernel 4) + SiLU over the q|k|v channels,
	// written as K shifted multiply-adds: y[t] = sum_k w[k] * x[t-(K-1)+k].
	// MLX's general conv1d is ~10x slower for this tiny depthwise kernel.
	conv := x.Silu(causalDepthwise(x, mixed, L.convTaps, B, T, linQKVDim))

	qkv := x.SplitAt(conv, -1, linHeads*linKeyDim, 2*linHeads*linKeyDim)
	q := x.Reshape(x.AsType(qkv[0], mlx.Float32), B, T, linHeads, linKeyDim)
	k := x.Reshape(x.AsType(qkv[1], mlx.Float32), B, T, linHeads, linKeyDim)
	v := x.Reshape(x.AsType(qkv[2], mlx.Float32), B, T, linHeads, linValDim)

	beta := x.Sigmoid(x.AsType(bb, mlx.Float32))                                       // [B, T, 16]
	g := x.Multiply(L.aLogNeg, x.Softplus(x.Add(x.AsType(aa, mlx.Float32), L.dtBias))) // [B, T, 16]

	// l2norm(q), l2norm(k) with eps 1e-6, then q *= 1/sqrt(dk).
	q = l2norm(x, q)
	k = l2norm(x, k)
	q = x.Multiply(q, x.Scalar(float32(1/math.Sqrt(linKeyDim))))

	var core *mlx.Array // [B, T, 16, 128] f32
	switch {
	case m.delta == deltaScan:
		core = deltaScanOps(x, q, k, v, g, beta, B, T)
	default:
		core = deltaChunked(x, q, k, v, g, beta, B, T)
	}

	// Gated RMSNorm: rmsnorm(core) * w (plain) * silu(z), in float32.
	zf := x.Reshape(x.AsType(z, mlx.Float32), B, T, linHeads, linValDim)
	o := x.Multiply(x.RMSNorm(core, L.gnorm, rmsEps), x.Silu(zf))
	o = x.AsType(x.Reshape(o, B, T, linValueWidth), h.Dtype())
	return x.Matmul(o, L.outProj)
}

// causalDepthwise computes a causal depthwise conv over [B, T, C] with taps
// [K][C] (taps[k] multiplies the input shifted by K-1-k), summing in the
// input's dtype like PyTorch's conv1d.
func causalDepthwise(x *mlx.Ctx, in *mlx.Array, taps []*mlx.Array, B, T, C int) *mlx.Array {
	K := len(taps)
	padded := x.Concatenate(1, x.Zeros(in.Dtype(), B, K-1, C), in) // [B, T+K-1, C]
	var acc *mlx.Array
	for k := 0; k < K; k++ {
		win := x.Slice(padded, []int{0, k, 0}, []int{B, k + T, C}, nil)
		term := x.Multiply(win, taps[k])
		if acc == nil {
			acc = term
		} else {
			acc = x.Add(acc, term)
		}
	}
	return acc
}

func l2norm(x *mlx.Ctx, a *mlx.Array) *mlx.Array {
	ss := x.Sum(x.Square(a), true, -1)
	return x.Multiply(a, x.Rsqrt(x.Add(ss, x.Scalar(1e-6))))
}

// deltaChunk is the chunk size of the portable chunked gated delta rule.
const deltaChunkLen = 64

// deltaChunked is torch_chunk_gated_delta_rule (the algorithm HF runs for
// prefill) in plain MLX ops, evaluated chunk by chunk so memory is bounded
// by one chunk's working set regardless of T. It is the portable path
// (CPU, CUDA without a kernel). q, k: [B,T,H,Dk]; v: [B,T,H,Dv]; g, beta:
// [B,T,H]; all f32. Returns [B,T,H,Dv].
func deltaChunked(x *mlx.Ctx, q, k, v, g, beta *mlx.Array, B, T int) *mlx.Array {
	const C, H, Dk, Dv = deltaChunkLen, linHeads, linKeyDim, linValDim
	pad := (C - T%C) % C
	Tp := T + pad
	N := Tp / C
	// [B,T,H,d] -> [B,H,Tp,d] (zero-padded; padded rows have beta=0, g=0).
	toBH := func(a *mlx.Array, d int) *mlx.Array {
		a = x.Transpose(a, 0, 2, 1, 3)
		if pad > 0 {
			a = x.Concatenate(2, a, x.Zeros(mlx.Float32, B, H, pad, d))
		}
		return a
	}
	toBH3 := func(a *mlx.Array) *mlx.Array { // [B,T,H] -> [B,H,Tp]
		a = x.Transpose(a, 0, 2, 1)
		if pad > 0 {
			a = x.Concatenate(2, a, x.Zeros(mlx.Float32, B, H, pad))
		}
		return a
	}
	qh, kh, vh := toBH(q, Dk), toBH(k, Dk), toBH(v, Dv)
	bh, gh := toBH3(beta), toBH3(g)

	// Constant masks for one chunk.
	ones := x.Add(x.Zeros(mlx.Float32, C, C), x.Scalar(1))
	lowerIncl := x.Tril(ones, 0)              // j <= i
	strictLower := x.Tril(ones, -1)           // j < i
	eye := x.Subtract(lowerIncl, strictLower) // identity
	upperMaskNeg := x.Multiply(x.Subtract(ones, lowerIncl), x.Scalar(-1e30))

	S := x.Zeros(mlx.Float32, B, H, Dk, Dv)
	outs := make([]*mlx.Array, 0, N)
	cx := mlx.NewCtx(x.S) // per-chunk scratch
	defer cx.Free()
	for n := 0; n < N; n++ {
		s0, s1 := n*C, (n+1)*C
		sl4 := func(a *mlx.Array, d int) *mlx.Array {
			return cx.Slice(a, []int{0, 0, s0, 0}, []int{B, H, s1, d}, nil)
		}
		sl3 := func(a *mlx.Array) *mlx.Array {
			return cx.Slice(a, []int{0, 0, s0}, []int{B, H, s1}, nil)
		}
		qc, kc, vc := sl4(qh, Dk), sl4(kh, Dk), sl4(vh, Dv)
		bc, gc := sl3(bh), sl3(gh) // [B,H,C]

		cum := cx.Cumsum(gc, 2, false, true) // [B,H,C]
		ci := cx.ExpandDims(cum, 3)          // [B,H,C,1]
		cj := cx.ExpandDims(cum, 2)          // [B,H,1,C]
		// exp(cum_i - cum_j) for j <= i, 0 above the diagonal (masked in log
		// space first so exp never overflows).
		decay := cx.Exp(cx.Add(cx.Subtract(ci, cj), upperMaskNeg)) // [B,H,C,C]

		b1 := cx.ExpandDims(bc, 3)
		kBeta := cx.Multiply(kc, b1)
		vBeta := cx.Multiply(vc, b1)
		kT := cx.SwapAxes(kc, 2, 3)

		// HF's UT system is L = (kBeta k^T * decay) strictly lower; the
		// solve applies (I + L)^-1. Its forward substitution (the export
		// path) starts from X = -L and does row_i += sum_{j<i} X[i,j]*row_j.
		X := cx.Negative(cx.Multiply(cx.Multiply(cx.Matmul(kBeta, kT), decay), strictLower))
		Tm, err := forwardSubstitute(cx, X, eye, B, H, C)
		if err != nil {
			x.Fail(err)
			return x.Zeros(mlx.Float32, B, T, H, Dv)
		}
		newV := cx.Matmul(Tm, vBeta)                                          // u
		kCum := cx.Matmul(Tm, cx.Multiply(kBeta, cx.Exp(ci)))                 // w
		attn := cx.Multiply(cx.Multiply(cx.Matmul(qc, kT), decay), lowerIncl) // intra-chunk
		qDec := cx.Multiply(qc, cx.Exp(ci))
		last := cx.Slice(cum, []int{0, 0, C - 1}, []int{B, H, C}, nil) // [B,H,1]
		kDec := cx.Multiply(kc, cx.Exp(cx.Subtract(cx.ExpandDims(last, 3), ci)))

		vNew := cx.Subtract(newV, cx.Matmul(kCum, S))
		o := cx.Add(cx.Matmul(qDec, S), cx.Matmul(attn, vNew)) // [B,H,C,Dv]
		S = cx.Add(cx.Multiply(S, cx.Exp(cx.ExpandDims(last, 3))), cx.Matmul(cx.SwapAxes(kDec, 2, 3), vNew))

		if err := cx.Eval(o, S); err != nil {
			x.Fail(err)
			return x.Zeros(mlx.Float32, B, T, H, Dv)
		}
		cx.Keep(o)
		cx.Keep(S)
		x.Adopt(o)
		x.Adopt(S)
		cx.Free()
		outs = append(outs, o)
	}
	out := x.Concatenate(2, outs...) // [B,H,Tp,Dv]
	if pad > 0 {
		out = x.Slice(out, []int{0, 0, 0, 0}, []int{B, H, T, Dv}, nil)
	}
	return x.Transpose(out, 0, 2, 1, 3)
}

// forwardSubstitute returns (I - L)^-1 for strictly lower-triangular L
// ([B,H,C,C]) row by row, the same recurrence as HF's non-solver path.
func forwardSubstitute(x *mlx.Ctx, L, eye *mlx.Array, B, H, C int) (*mlx.Array, error) {
	rows := make([]*mlx.Array, C)
	for i := 0; i < C; i++ {
		ri := x.Slice(L, []int{0, 0, i, 0}, []int{B, H, i + 1, C}, nil) // [B,H,1,C]
		if i > 0 {
			prev := x.Concatenate(2, rows[:i]...)                          // [B,H,i,C]
			coef := x.Slice(ri, []int{0, 0, 0, 0}, []int{B, H, 1, i}, nil) // [B,H,1,i]
			ri = x.Add(ri, x.Matmul(coef, prev))
		}
		rows[i] = ri
	}
	m := x.Add(x.Concatenate(2, rows...), eye)
	return m, x.Err()
}

// deltaScanOps is the exact recurrence (torch_recurrent_gated_delta_rule)
// written with plain ops, one token at a time. It is only a test reference
// (per-token graphs are slow and memory-hungry at long T). S is
// [B, Hv, Dk, Dv].
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

var errCancelled = errors.New("engine: cancelled")
