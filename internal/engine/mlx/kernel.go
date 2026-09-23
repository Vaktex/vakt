//go:build mlx

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"unsafe"
)

// Kernel is a custom Metal kernel (mlx.fast.metal_kernel). It is only
// usable on a Metal stream.
type Kernel struct {
	c       C.mlx_fast_metal_kernel
	name    string
	inputs  int
	outputs int
}

// NewKernel compiles lazily (on first Apply) a Metal kernel body. source is
// the body only; MLX generates the signature from the input/output names.
func NewKernel(name string, inputNames, outputNames []string, source, header string) *Kernel {
	Init()
	in := C.mlx_vector_string_new()
	defer C.mlx_vector_string_free(in)
	for _, s := range inputNames {
		cs := C.CString(s)
		C.mlx_vector_string_append_value(in, cs)
		C.free(unsafe.Pointer(cs))
	}
	out := C.mlx_vector_string_new()
	defer C.mlx_vector_string_free(out)
	for _, s := range outputNames {
		cs := C.CString(s)
		C.mlx_vector_string_append_value(out, cs)
		C.free(unsafe.Pointer(cs))
	}
	cn, csrc, chdr := C.CString(name), C.CString(source), C.CString(header)
	defer C.free(unsafe.Pointer(cn))
	defer C.free(unsafe.Pointer(csrc))
	defer C.free(unsafe.Pointer(chdr))
	k := C.mlx_fast_metal_kernel_new(cn, in, out, csrc, chdr, C.bool(true), C.bool(false))
	return &Kernel{c: k, name: name, inputs: len(inputNames), outputs: len(outputNames)}
}

// Free releases the kernel.
func (k *Kernel) Free() {
	if k != nil && k.c.ctx != nil {
		C.mlx_fast_metal_kernel_free(k.c)
		k.c.ctx = nil
	}
}

// KernelOutput describes one output buffer.
type KernelOutput struct {
	Shape []int
	Dtype DType
}

// KernelLaunch configures one application.
type KernelLaunch struct {
	Grid, ThreadGroup [3]int
	Outputs           []KernelOutput
	TemplateInts      map[string]int
	TemplateDtypes    map[string]DType
	InitValue         *float32 // fill outputs before launch (nil = uninitialised)
}

// Apply runs the kernel.
func (x *Ctx) Apply(k *Kernel, inputs []*Array, l KernelLaunch) []*Array {
	outs := make([]*Array, len(l.Outputs))
	fill := func() []*Array {
		for i := range outs {
			if outs[i] == nil {
				outs[i] = x.empty()
			}
		}
		return outs
	}
	op := "kernel:" + k.name
	if !x.ok(op, inputs...) {
		return fill()
	}
	if len(inputs) != k.inputs || len(l.Outputs) != k.outputs {
		x.Fail(&Error{Op: op, Msg: "input/output count mismatch"})
		return fill()
	}
	cfg := C.mlx_fast_metal_kernel_config_new()
	defer C.mlx_fast_metal_kernel_config_free(cfg)
	for _, o := range l.Outputs {
		sp, sn := cInts(o.Shape)
		C.mlx_fast_metal_kernel_config_add_output_arg(cfg, sp, sn, cdtype(o.Dtype))
		freeInts(sp)
	}
	C.mlx_fast_metal_kernel_config_set_grid(cfg, cint(l.Grid[0]), cint(l.Grid[1]), cint(l.Grid[2]))
	C.mlx_fast_metal_kernel_config_set_thread_group(cfg, cint(l.ThreadGroup[0]), cint(l.ThreadGroup[1]), cint(l.ThreadGroup[2]))
	if l.InitValue != nil {
		C.mlx_fast_metal_kernel_config_set_init_value(cfg, C.float(*l.InitValue))
	}
	for name, v := range l.TemplateInts {
		cs := C.CString(name)
		C.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, cs, cint(v))
		C.free(unsafe.Pointer(cs))
	}
	for name, dt := range l.TemplateDtypes {
		cs := C.CString(name)
		C.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, cs, cdtype(dt))
		C.free(unsafe.Pointer(cs))
	}
	in := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(in)
	for _, a := range inputs {
		C.mlx_vector_array_append_value(in, a.c)
	}
	res := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(res)
	if !x.check(op, C.mlx_fast_metal_kernel_apply(&res, k.c, in, cfg, x.S.c)) {
		return fill()
	}
	for i := range outs {
		el := C.mlx_array_new()
		if !x.check(op, C.mlx_vector_array_get(&el, res, csize(i))) {
			C.mlx_array_free(el)
			return fill()
		}
		outs[i] = x.track(el)
	}
	return outs
}
