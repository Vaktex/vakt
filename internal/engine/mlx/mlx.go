//go:build mlx

// Package mlx is a thin, safe wrapper over mlx-c, covering exactly what the
// DOM-0.8B forward pass needs.
//
// Error model: every op runs through a *Ctx. mlx-c reports failures through a
// process-wide error handler plus a non-zero return code; the first failure
// is recorded on the Ctx (sticky) and later ops become no-ops that return an
// empty array. Callers build a whole graph and check ctx.Err() (or the error
// from Eval) once. Nothing panics across the cgo boundary.
//
// Memory model: every *Array owns one mlx_array handle. Arrays created
// through a Ctx are tracked and released by ctx.Free(), which callers defer;
// runtime cleanups are a safety net only. MLX arrays are reference counted,
// so freeing a handle never invalidates arrays computed from it.
//
// Concurrency: MLX graph construction and evaluation are not safe for
// concurrent use from multiple goroutines on the same stream. The engine
// drives one Ctx from one goroutine (see core.Engine).
package mlx

/*
#include <stdlib.h>
#include <string.h>
#include "mlx/c/mlx.h"

extern void vaktMLXErrorHandler(char* msg, void* data);

static void vakt_install_error_handler(void) {
	mlx_set_error_handler((mlx_error_handler_func)vaktMLXErrorHandler, NULL, NULL);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// DType mirrors mlx_dtype for the types the engine uses.
type DType int

const (
	Bool     DType = C.MLX_BOOL
	Int32    DType = C.MLX_INT32
	Float16  DType = C.MLX_FLOAT16
	Float32  DType = C.MLX_FLOAT32
	BFloat16 DType = C.MLX_BFLOAT16
)

func (d DType) String() string {
	switch d {
	case Bool:
		return "bool"
	case Int32:
		return "int32"
	case Float16:
		return "float16"
	case Float32:
		return "float32"
	case BFloat16:
		return "bfloat16"
	}
	return fmt.Sprintf("dtype(%d)", int(d))
}

// ---------------------------------------------------------------- errors

var (
	lastErrMu sync.Mutex
	lastErr   string
)

//export vaktMLXErrorHandler
func vaktMLXErrorHandler(msg *C.char, _ unsafe.Pointer) {
	s := C.GoString(msg)
	lastErrMu.Lock()
	lastErr = s
	lastErrMu.Unlock()
}

func takeLastErr() string {
	lastErrMu.Lock()
	defer lastErrMu.Unlock()
	s := lastErr
	lastErr = ""
	return s
}

// Error is an MLX failure with the op that raised it.
type Error struct {
	Op  string
	Msg string
}

func (e *Error) Error() string { return "mlx: " + e.Op + ": " + e.Msg }

var initOnce sync.Once

// Init installs the error handler and registers the embedded Metal library.
// It is idempotent and called by every entry point.
//
// It also pins MLX_ENABLE_TF32=0 unless the caller set it: MLX defaults to
// TF32 tensor-core matmuls for float32 on the GPU, which silently drops fp32
// to ~10 mantissa bits and breaks parity with the PyTorch reference. MLX
// reads the variable once, on first use, so it must be set before any op.
// The bf16 precision mode is unaffected (its matmuls are bf16 anyway).
func Init() {
	initOnce.Do(func() {
		if _, set := os.LookupEnv("MLX_ENABLE_TF32"); !set {
			_ = os.Setenv("MLX_ENABLE_TF32", "0")
		}
		C.vakt_install_error_handler()
		registerMetallib()
	})
}

// ---------------------------------------------------------------- streams

// Stream is an MLX stream (a device plus a queue).
type Stream struct{ c C.mlx_stream }

// MetalAvailable reports whether a Metal GPU can be used.
func MetalAvailable() bool {
	Init()
	var ok C.bool
	if C.mlx_metal_is_available(&ok) != 0 {
		return false
	}
	return bool(ok)
}

// CPU returns the default CPU stream.
func CPU() *Stream { Init(); return &Stream{C.mlx_default_cpu_stream_new()} }

// GPUDevice returns a new stream on GPU number i.
func GPUDevice(i int) (*Stream, error) {
	Init()
	var n C.int
	if C.mlx_device_count(&n, C.MLX_GPU) != 0 || i < 0 || i >= int(n) {
		return nil, fmt.Errorf("mlx: GPU %d not available (%d found)", i, int(n))
	}
	d := C.mlx_device_new_type(C.MLX_GPU, cint(i))
	defer C.mlx_device_free(d)
	return &Stream{C.mlx_stream_new_device(d)}, nil
}

// GPU returns the default GPU stream (Metal or CUDA).
func GPU() *Stream { Init(); return &Stream{C.mlx_default_gpu_stream_new()} }

// Free releases the stream handle.
func (s *Stream) Free() {
	if s != nil && s.c.ctx != nil {
		C.mlx_stream_free(s.c)
		s.c.ctx = nil
	}
}

// TestStream picks the stream for tests: VAKT_TEST_DEVICE=cpu forces CPU
// (hosted CI macOS runners have no usable Metal compute), otherwise GPU when
// available.
func TestStream() *Stream {
	if os.Getenv("VAKT_TEST_DEVICE") == "cpu" || !gpuAvailable() {
		return CPU()
	}
	return GPU()
}

// ---------------------------------------------------------------- arrays

// handle is the freeable state shared by an Array and its GC cleanup, so an
// explicit Free and the cleanup can never both release the same handle.
type handle struct {
	c     C.mlx_array
	freed atomic.Bool
}

func (h *handle) release() {
	if h.freed.CompareAndSwap(false, true) && h.c.ctx != nil {
		C.mlx_array_free(h.c)
	}
}

// Array owns one mlx_array handle.
type Array struct {
	h *handle
	c C.mlx_array // == h.c; kept for direct use in cgo calls
}

func newArray(c C.mlx_array) *Array {
	h := &handle{c: c}
	a := &Array{h: h, c: c}
	// Safety net only: arrays are normally released by Ctx.Free.
	runtime.AddCleanup(a, (*handle).release, h)
	return a
}

// Free releases the handle now. Safe to call twice.
func (a *Array) Free() {
	if a != nil && a.h != nil {
		a.h.release()
	}
}

// Valid reports whether the array holds a live handle.
func (a *Array) Valid() bool {
	return a != nil && a.h != nil && !a.h.freed.Load() && a.c.ctx != nil
}

// Shape returns the array's dimensions.
func (a *Array) Shape() []int {
	if !a.Valid() {
		return nil
	}
	n := goint(C.mlx_array_ndim(a.c))
	if n == 0 {
		return []int{}
	}
	p := unsafe.Slice(C.mlx_array_shape(a.c), n)
	out := make([]int, n)
	for i, v := range p {
		out[i] = int(v)
	}
	runtime.KeepAlive(a) // a's cleanup must not run while p is read
	return out
}

// Size is the number of elements.
func (a *Array) Size() int {
	if !a.Valid() {
		return 0
	}
	return goint(C.mlx_array_size(a.c))
}

// Dtype returns the element type.
func (a *Array) Dtype() DType {
	if !a.Valid() {
		return -1
	}
	return DType(C.mlx_array_dtype(a.c))
}

// ---------------------------------------------------------------- Ctx

// Ctx records the first error and tracks arrays for bulk release.
type Ctx struct {
	S     *Stream
	err   error
	owned []*Array
}

// NewCtx returns a context that builds graphs on stream s.
func NewCtx(s *Stream) *Ctx { Init(); return &Ctx{S: s} }

// Err returns the first error recorded by an op, if any.
func (x *Ctx) Err() error { return x.err }

// Fail records err if no error is recorded yet.
func (x *Ctx) Fail(err error) {
	if x.err == nil {
		x.err = err
	}
}

// Keep removes a from the Ctx's ownership so Free does not release it.
func (x *Ctx) Keep(a *Array) *Array {
	for i, o := range x.owned {
		if o == a {
			x.owned = append(x.owned[:i], x.owned[i+1:]...)
			break
		}
	}
	return a
}

// Adopt makes the Ctx responsible for releasing a (the inverse of Keep).
func (x *Ctx) Adopt(a *Array) *Array {
	x.owned = append(x.owned, a)
	return a
}

// Free releases every array created through this Ctx (except kept ones).
func (x *Ctx) Free() {
	for _, a := range x.owned {
		a.Free()
	}
	x.owned = x.owned[:0]
}

func (x *Ctx) track(c C.mlx_array) *Array {
	a := newArray(c)
	x.owned = append(x.owned, a)
	return a
}

// empty is returned from ops after a failure; it is a valid, freeable handle.
func (x *Ctx) empty() *Array { return x.track(C.mlx_array_new()) }

// check converts an mlx-c return code into the sticky error.
func (x *Ctx) check(op string, rc C.int) bool {
	if rc == 0 {
		return true
	}
	msg := takeLastErr()
	if msg == "" {
		msg = "failed"
	}
	x.Fail(&Error{Op: op, Msg: msg})
	return false
}

// ok reports whether all inputs are usable and no error is recorded yet.
func (x *Ctx) ok(op string, in ...*Array) bool {
	if x.err != nil {
		return false
	}
	for _, a := range in {
		if !a.Valid() {
			x.Fail(&Error{Op: op, Msg: "invalid (nil or freed) input array"})
			return false
		}
	}
	return true
}

// unary / binary helpers keep each wrapper to one line.
type unaryFn func(*C.mlx_array, C.mlx_array, C.mlx_stream) C.int
type binaryFn func(*C.mlx_array, C.mlx_array, C.mlx_array, C.mlx_stream) C.int

func (x *Ctx) unary(op string, f unaryFn, a *Array) *Array {
	if !x.ok(op, a) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check(op, f(&res, a.c, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

func (x *Ctx) binary(op string, f binaryFn, a, b *Array) *Array {
	if !x.ok(op, a, b) {
		return x.empty()
	}
	res := C.mlx_array_new()
	if !x.check(op, f(&res, a.c, b.c, x.S.c)) {
		C.mlx_array_free(res)
		return x.empty()
	}
	return x.track(res)
}

// ---------------------------------------------------------------- constructors

// cInts copies v into C memory. Every value that crosses into mlx-c is a
// shape, axis or index derived from validated model constants and bounded
// token counts (<= core.MaxTokens); values outside int32 are clamped so a
// logic error surfaces as an MLX shape error rather than silent wraparound.
// cint converts one value for mlx-c, mapping anything outside int32 to -1
// (rejected by every mlx-c shape/axis/grid parameter).
func cint(v int) C.int {
	if v > math.MaxInt32 || v < math.MinInt32 {
		return -1
	}
	return C.int(v) // #nosec G115 -- range checked above
}

// cdtype converts one of the DType constants above (a closed enum).
func cdtype(d DType) C.mlx_dtype {
	return C.mlx_dtype(d) // #nosec G115 -- closed enum of mlx_dtype values
}

// csize converts a non-negative Go length/index.
func csize(n int) C.size_t {
	if n < 0 {
		return 0
	}
	return C.size_t(n) // #nosec G115 -- n >= 0
}

// goint converts an mlx-c size_t count (bounded by array sizes <= 2^34).
func goint(n C.size_t) int {
	return int(n) // #nosec G115 -- MLX array sizes are far below MaxInt
}

func cInts(v []int) (*C.int, C.size_t) {
	if len(v) == 0 {
		return nil, 0
	}
	for i, x := range v {
		if x > math.MaxInt32 || x < math.MinInt32 {
			v[i] = -1 // invalid for every mlx-c shape/axis parameter
		}
	}
	p := (*C.int)(C.malloc(csize(len(v)) * C.size_t(unsafe.Sizeof(C.int(0)))))
	s := unsafe.Slice(p, len(v))
	for i, x := range v {
		s[i] = C.int(x) // #nosec G115 -- clamped to int32 range above
	}
	return p, csize(len(v))
}

func freeInts(p *C.int) {
	if p != nil {
		C.free(unsafe.Pointer(p))
	}
}

func checkShape(n int, shape []int) error {
	const maxElems = 1 << 34
	total := 1
	for _, d := range shape {
		if d < 0 || d > 1<<30 {
			return fmt.Errorf("invalid dimension %d", d)
		}
		// Check before multiplying so the product can never wrap.
		if d != 0 && total > maxElems/d {
			return errors.New("shape too large")
		}
		total *= d
	}
	if total != n {
		return fmt.Errorf("data length %d does not match shape %v", n, shape)
	}
	return nil
}

// FromFloat32 copies data into a new float32 array.
func (x *Ctx) FromFloat32(data []float32, shape ...int) *Array {
	return x.fromData("from_float32", unsafe.Pointer(unsafe.SliceData(data)), len(data), shape, Float32)
}

// FromInt32 copies data into a new int32 array.
func (x *Ctx) FromInt32(data []int32, shape ...int) *Array {
	return x.fromData("from_int32", unsafe.Pointer(unsafe.SliceData(data)), len(data), shape, Int32)
}

// FromBool copies data into a new bool array.
func (x *Ctx) FromBool(data []bool, shape ...int) *Array {
	return x.fromData("from_bool", unsafe.Pointer(unsafe.SliceData(data)), len(data), shape, Bool)
}

func (x *Ctx) fromData(op string, p unsafe.Pointer, n int, shape []int, dt DType) *Array {
	if !x.ok(op) {
		return x.empty()
	}
	if err := checkShape(n, shape); err != nil {
		x.Fail(&Error{Op: op, Msg: err.Error()})
		return x.empty()
	}
	if n == 0 {
		p = nil
	}
	sp, sn := cInts(shape)
	defer freeInts(sp)
	// mlx_array_new_data copies the buffer, so the Go memory need only live
	// for the duration of the call (cgo pins it).
	return x.track(C.mlx_array_new_data(p, sp, C.int(sn), cdtype(dt))) // #nosec G115 -- sn = len(shape) <= 8
}

// Scalar returns a 0-d float32 array.
func (x *Ctx) Scalar(v float32) *Array { return x.track(C.mlx_array_new_float32(C.float(v))) }

// ScalarInt returns a 0-d int32 array.
func (x *Ctx) ScalarInt(v int) *Array { return x.track(C.mlx_array_new_int(cint(v))) }

// ---------------------------------------------------------------- evaluation

// Eval evaluates the arrays and returns the Ctx's error, if any.
func (x *Ctx) Eval(arrays ...*Array) error {
	if x.err != nil {
		return x.err
	}
	if !x.ok("eval", arrays...) {
		return x.err
	}
	vec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vec)
	for _, a := range arrays {
		C.mlx_vector_array_append_value(vec, a.c)
	}
	x.check("eval", C.mlx_eval(vec))
	return x.err
}

// Float32s evaluates a (casting to float32 if needed) and copies it out.
func (x *Ctx) Float32s(a *Array) ([]float32, error) {
	if a.Valid() && a.Dtype() != Float32 {
		a = x.AsType(a, Float32)
	}
	// The raw data pointer is in the array's own memory layout: transposes
	// and strided slices are views over another buffer. Always copy out a
	// row-contiguous array so element i is logical element i.
	a = x.Contiguous(a)
	if err := x.Eval(a); err != nil {
		return nil, err
	}
	n := a.Size()
	p := C.mlx_array_data_float32(a.c)
	if p == nil && n > 0 {
		err := &Error{Op: "data_float32", Msg: "no data pointer (" + takeLastErr() + ")"}
		x.Fail(err)
		return nil, err
	}
	out := make([]float32, n)
	if n > 0 {
		copy(out, unsafe.Slice((*float32)(unsafe.Pointer(p)), n))
	}
	runtime.KeepAlive(a) // a's cleanup must not run while p is read
	return out, nil
}

// ---------------------------------------------------------------- memory

// SetCacheLimit bounds MLX's buffer cache (bytes) and returns the old limit.
func SetCacheLimit(n uint64) uint64 {
	var old C.size_t
	C.mlx_set_cache_limit(&old, C.size_t(n))
	return uint64(old)
}

// SetMemoryLimit sets MLX's soft memory limit (bytes).
func SetMemoryLimit(n uint64) uint64 {
	var old C.size_t
	C.mlx_set_memory_limit(&old, C.size_t(n))
	return uint64(old)
}

// SetWiredLimit sets the wired (resident) memory limit on Metal (bytes).
func SetWiredLimit(n uint64) uint64 {
	var old C.size_t
	C.mlx_set_wired_limit(&old, C.size_t(n))
	return uint64(old)
}

// MemoryLimit returns MLX's memory limit for the default device (bytes); on
// Metal this defaults to the GPU's recommended working set.
func MemoryLimit() uint64 {
	Init()
	var n C.size_t
	C.mlx_get_memory_limit(&n)
	return uint64(n)
}

// ActiveMemory returns bytes currently held by live arrays.
func ActiveMemory() uint64 {
	var n C.size_t
	C.mlx_get_active_memory(&n)
	return uint64(n)
}

// PeakMemory returns the peak of ActiveMemory since start or last reset.
func PeakMemory() uint64 {
	var n C.size_t
	C.mlx_get_peak_memory(&n)
	return uint64(n)
}

// ResetPeakMemory resets the peak counter.
func ResetPeakMemory() { C.mlx_reset_peak_memory() }

// ClearCache drops cached (free) buffers.
func ClearCache() { C.mlx_clear_cache() }
