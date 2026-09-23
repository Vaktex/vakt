//go:build mlx

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"

static mlx_optional_float vakt_opt_float(float v, bool has) {
	mlx_optional_float o; o.value = v; o.has_value = has; return o;
}
static mlx_array vakt_null_array(void) { mlx_array a; a.ctx = NULL; return a; }
*/
import "C"

import (
	"unsafe"
)

// ---------------------------------------------------------------- elementwise

func (x *Ctx) Add(a, b *Array) *Array      { return x.binary("add", fAdd, a, b) }
func (x *Ctx) Subtract(a, b *Array) *Array { return x.binary("subtract", fSub, a, b) }
func (x *Ctx) Multiply(a, b *Array) *Array { return x.binary("multiply", fMul, a, b) }
func (x *Ctx) Divide(a, b *Array) *Array   { return x.binary("divide", fDiv, a, b) }
func (x *Ctx) Maximum(a, b *Array) *Array  { return x.binary("maximum", fMax, a, b) }
func (x *Ctx) Matmul(a, b *Array) *Array   { return x.binary("matmul", fMatmul, a, b) }
func (x *Ctx) Less(a, b *Array) *Array     { return x.binary("less", fLess, a, b) }
func (x *Ctx) Greater(a, b *Array) *Array  { return x.binary("greater", fGreater, a, b) }
func (x *Ctx) Equal(a, b *Array) *Array    { return x.binary("equal", fEqual, a, b) }
func (x *Ctx) LogAddExp(a, b *Array) *Array {
	return x.binary("logaddexp", fLogAddExp, a, b)
}

func (x *Ctx) Rsqrt(a *Array) *Array    { return x.unary("rsqrt", fRsqrt, a) }
func (x *Ctx) Sqrt(a *Array) *Array     { return x.unary("sqrt", fSqrt, a) }
func (x *Ctx) Exp(a *Array) *Array      { return x.unary("exp", fExp, a) }
func (x *Ctx) Log(a *Array) *Array      { return x.unary("log", fLog, a) }
func (x *Ctx) Log1p(a *Array) *Array    { return x.unary("log1p", fLog1p, a) }
func (x *Ctx) Sigmoid(a *Array) *Array  { return x.unary("sigmoid", fSigmoid, a) }
func (x *Ctx) Negative(a *Array) *Array { return x.unary("negative", fNeg, a) }
func (x *Ctx) Square(a *Array) *Array   { return x.unary("square", fSquare, a) }

// Silu is x * sigmoid(x).
func (x *Ctx) Silu(a *Array) *Array { return x.Multiply(a, x.Sigmoid(a)) }

// Softplus is log(1 + exp(x)), computed stably as logaddexp(x, 0).
func (x *Ctx) Softplus(a *Array) *Array {
	return x.LogAddExp(a, x.AsType(x.Scalar(0), a.Dtype()))
}

// Where selects x where cond is true, else y.
func (x *Ctx) Where(cond, a, b *Array) *Array {
	if !x.ok("where", cond, a, b) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("where", C.mlx_where(&res, cond.c, a.c, b.c, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// AsType casts to dt.
func (x *Ctx) AsType(a *Array, dt DType) *Array {
	if !x.ok("astype", a) {
		return x.empty()
	}
	// Always return a new handle (MLX makes same-dtype astype a no-op view),
	// so the result's lifetime is independent of a's.
	res := C.mlx_array_new()
	if !x.check("astype", C.mlx_astype(&res, a.c, C.mlx_dtype(dt), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// ---------------------------------------------------------------- reductions

// Sum reduces over axes.
func (x *Ctx) Sum(a *Array, keepdims bool, axes ...int) *Array {
	return x.reduce("sum", a, keepdims, axes, true)
}

// Mean reduces over axes.
func (x *Ctx) Mean(a *Array, keepdims bool, axes ...int) *Array {
	return x.reduce("mean", a, keepdims, axes, false)
}

func (x *Ctx) reduce(op string, a *Array, keepdims bool, axes []int, sum bool) *Array {
	if !x.ok(op, a) {
		return x.empty()
	}
	ap, an := cInts(axes)
	defer freeInts(ap)
	res := C.mlx_array_new()
	var rc C.int
	if sum {
		rc = C.mlx_sum_axes(&res, a.c, ap, an, C.bool(keepdims), x.S.c)
	} else {
		rc = C.mlx_mean_axes(&res, a.c, ap, an, C.bool(keepdims), x.S.c)
	}
	if !x.check(op, rc) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// ---------------------------------------------------------------- shape

// Reshape to shape (-1 allowed once).
func (x *Ctx) Reshape(a *Array, shape ...int) *Array {
	if !x.ok("reshape", a) {
		return x.empty()
	}
	sp, sn := cInts(shape)
	defer freeInts(sp)
	res := C.mlx_array_new()
	if !x.check("reshape", C.mlx_reshape(&res, a.c, sp, sn, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Transpose permutes axes.
func (x *Ctx) Transpose(a *Array, axes ...int) *Array {
	if !x.ok("transpose", a) {
		return x.empty()
	}
	ap, an := cInts(axes)
	defer freeInts(ap)
	res := C.mlx_array_new()
	if !x.check("transpose", C.mlx_transpose_axes(&res, a.c, ap, an, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// SwapAxes swaps two axes.
func (x *Ctx) SwapAxes(a *Array, i, j int) *Array {
	if !x.ok("swapaxes", a) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("swapaxes", C.mlx_swapaxes(&res, a.c, C.int(i), C.int(j), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// ExpandDims inserts a size-1 axis.
func (x *Ctx) ExpandDims(a *Array, axis int) *Array {
	if !x.ok("expand_dims", a) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("expand_dims", C.mlx_expand_dims(&res, a.c, C.int(axis), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Squeeze removes the given size-1 axes.
func (x *Ctx) Squeeze(a *Array, axes ...int) *Array {
	if !x.ok("squeeze", a) {
		return x.empty()
	}
	ap, an := cInts(axes)
	defer freeInts(ap)
	res := C.mlx_array_new()
	if !x.check("squeeze", C.mlx_squeeze_axes(&res, a.c, ap, an, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// BroadcastTo broadcasts a to shape.
func (x *Ctx) BroadcastTo(a *Array, shape ...int) *Array {
	if !x.ok("broadcast_to", a) {
		return x.empty()
	}
	sp, sn := cInts(shape)
	defer freeInts(sp)
	res := C.mlx_array_new()
	if !x.check("broadcast_to", C.mlx_broadcast_to(&res, a.c, sp, sn, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Concatenate joins arrays along axis.
func (x *Ctx) Concatenate(axis int, arrays ...*Array) *Array {
	if !x.ok("concatenate", arrays...) {
		return x.empty()
	}
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, a := range arrays {
		C.mlx_vector_array_append_value(vec, a.c)
	}
	res := C.mlx_array_new()
	if !x.check("concatenate", C.mlx_concatenate_axis(&res, vec, C.int(axis), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// SplitAt splits a along axis at the given indices (numpy semantics).
func (x *Ctx) SplitAt(a *Array, axis int, indices ...int) []*Array {
	n := len(indices) + 1
	out := make([]*Array, n)
	fill := func() []*Array {
		for i := range out {
			if out[i] == nil {
				out[i] = x.empty()
			}
		}
		return out
	}
	if !x.ok("split", a) {
		return fill()
	}
	ip, in := cInts(indices)
	defer freeInts(ip)
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	if !x.check("split", C.mlx_split_sections(&vec, a.c, ip, in, C.int(axis), x.S.c)) {
		return fill()
	}
	for i := 0; i < n; i++ {
		el := C.mlx_array_new()
		if !x.check("split", C.mlx_vector_array_get(&el, vec, C.size_t(i))) {
			C.mlx_array_free(el)
			return fill()
		}
		out[i] = x.track(el)
	}
	return out
}

// Slice takes a[start:stop:strides] on every axis (len == ndim).
func (x *Ctx) Slice(a *Array, start, stop, strides []int) *Array {
	if !x.ok("slice", a) {
		return x.empty()
	}
	if strides == nil {
		strides = make([]int, len(start))
		for i := range strides {
			strides[i] = 1
		}
	}
	p1, n1 := cInts(start)
	defer freeInts(p1)
	p2, n2 := cInts(stop)
	defer freeInts(p2)
	p3, n3 := cInts(strides)
	defer freeInts(p3)
	res := C.mlx_array_new()
	if !x.check("slice", C.mlx_slice(&res, a.c, p1, n1, p2, n2, p3, n3, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Take gathers along axis.
func (x *Ctx) Take(a, indices *Array, axis int) *Array {
	if !x.ok("take", a, indices) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("take", C.mlx_take_axis(&res, a.c, indices.c, C.int(axis), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Arange returns [start, stop) with step, of dtype dt.
func (x *Ctx) Arange(start, stop, step float64, dt DType) *Array {
	if !x.ok("arange") {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("arange", C.mlx_arange(&res, C.double(start), C.double(stop), C.double(step), C.mlx_dtype(dt), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Zeros returns a zero array.
func (x *Ctx) Zeros(dt DType, shape ...int) *Array {
	if !x.ok("zeros") {
		return x.empty()
	}
	sp, sn := cInts(shape)
	defer freeInts(sp)
	res := C.mlx_array_new()
	if !x.check("zeros", C.mlx_zeros(&res, sp, sn, C.mlx_dtype(dt), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Contiguous returns a row-contiguous copy (no-op if already contiguous).
func (x *Ctx) Contiguous(a *Array) *Array {
	if !x.ok("contiguous", a) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("contiguous", C.mlx_contiguous(&res, a.c, C.bool(false), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Cumsum along axis.
func (x *Ctx) Cumsum(a *Array, axis int, reverse, inclusive bool) *Array {
	if !x.ok("cumsum", a) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("cumsum", C.mlx_cumsum(&res, a.c, C.int(axis), C.bool(reverse), C.bool(inclusive), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// Tril keeps the lower triangle (k = diagonal offset).
func (x *Ctx) Tril(a *Array, k int) *Array {
	if !x.ok("tril", a) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("tril", C.mlx_tril(&res, a.c, C.int(k), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// ---------------------------------------------------------------- conv

// Conv1d: input [N, L, C_in], weight [C_out, K, C_in/groups] (MLX's NLC/OKI
// layout; PyTorch's [C_out, C_in/groups, K] must be transposed at load).
func (x *Ctx) Conv1d(in, w *Array, stride, padding, dilation, groups int) *Array {
	if !x.ok("conv1d", in, w) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("conv1d", C.mlx_conv1d(&res, in.c, w.c, C.int(stride), C.int(padding), C.int(dilation), C.int(groups), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// ---------------------------------------------------------------- fast

// RMSNorm computes x * rsqrt(mean(x^2) + eps) * w over the last axis. w may
// be nil (no scale).
func (x *Ctx) RMSNorm(a, w *Array, eps float32) *Array {
	if !x.ok("rms_norm", a) || (w != nil && !x.ok("rms_norm", w)) {
		return x.empty()
	}
	wc := C.vakt_null_array()
	if w != nil {
		wc = w.c
	}
	res := C.mlx_array_new()
	if !x.check("rms_norm", C.mlx_fast_rms_norm(&res, a.c, wc, C.float(eps), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// RoPE applies rotary embeddings to the first dims features of the last axis
// of a [B, H, T, D] tensor, positions offset..offset+T-1. traditional=false
// is the rotate_half (GPT-NeoX / HF) convention.
func (x *Ctx) RoPE(a *Array, dims int, traditional bool, base, scale float32, offset int) *Array {
	if !x.ok("rope", a) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check("rope", C.mlx_fast_rope(&res, a.c, C.int(dims), C.bool(traditional),
		C.vakt_opt_float(C.float(base), C.bool(true)), C.float(scale), C.int(offset), C.vakt_null_array(), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// SDPA is fused scaled dot-product attention over [B, H, T, D] tensors
// (K/V may have fewer heads: GQA). mode is "causal" or "" (use mask, which
// may be nil).
func (x *Ctx) SDPA(q, k, v *Array, scale float32, mode string, mask *Array) *Array {
	if !x.ok("sdpa", q, k, v) || (mask != nil && !x.ok("sdpa", mask)) {
		return x.empty()
	}
	cm := C.CString(mode)
	defer C.free(unsafe.Pointer(cm))
	mc := C.vakt_null_array()
	if mask != nil {
		mc = mask.c
	}
	res := C.mlx_array_new()
	if !x.check("sdpa", C.mlx_fast_scaled_dot_product_attention(&res, q.c, k.c, v.c, C.float(scale), cm, mc, C.vakt_null_array(), x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// ---------------------------------------------------------------- function table
// cgo cannot take the address of C functions directly as Go func values, so
// each op gets a tiny adapter.

func fAdd(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int { return C.mlx_add(r, a, b, s) }
func fSub(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int {
	return C.mlx_subtract(r, a, b, s)
}
func fMul(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int {
	return C.mlx_multiply(r, a, b, s)
}
func fDiv(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int { return C.mlx_divide(r, a, b, s) }
func fMax(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int {
	return C.mlx_maximum(r, a, b, s)
}
func fMatmul(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int {
	return C.mlx_matmul(r, a, b, s)
}
func fLess(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int { return C.mlx_less(r, a, b, s) }
func fGreater(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int {
	return C.mlx_greater(r, a, b, s)
}
func fEqual(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int { return C.mlx_equal(r, a, b, s) }
func fLogAddExp(r *C.mlx_array, a, b C.mlx_array, s C.mlx_stream) C.int {
	return C.mlx_logaddexp(r, a, b, s)
}
func fRsqrt(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int    { return C.mlx_rsqrt(r, a, s) }
func fSqrt(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int     { return C.mlx_sqrt(r, a, s) }
func fExp(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int      { return C.mlx_exp(r, a, s) }
func fLog(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int      { return C.mlx_log(r, a, s) }
func fLog1p(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int    { return C.mlx_log1p(r, a, s) }
func fSigmoid(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int  { return C.mlx_sigmoid(r, a, s) }
func fNeg(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int      { return C.mlx_negative(r, a, s) }
func fSquare(r *C.mlx_array, a C.mlx_array, s C.mlx_stream) C.int   { return C.mlx_square(r, a, s) }
