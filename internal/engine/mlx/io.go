//go:build mlx

package mlx

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"sync"
	"unsafe"
)

// LoadSafetensors loads every tensor in path (lazily: data is read when first
// evaluated). File reads always run on the CPU stream (MLX has no GPU Load
// primitive); the arrays are then usable from any stream. The header must already have been validated with
// internal/engine/safetensors: MLX's own parser is not hardened against
// hostile files. Returned arrays are NOT tracked by the Ctx; the caller owns
// them (typically for the life of the model).
func (x *Ctx) LoadSafetensors(path string) (map[string]*Array, error) {
	if !x.ok("load_safetensors") {
		return nil, x.err
	}
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	m := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(m)
	meta := C.mlx_map_string_to_string_new()
	defer C.mlx_map_string_to_string_free(meta)
	// The Load primitives run when the arrays are first evaluated, possibly
	// on another OS thread, so they need a stream usable from any thread.
	loadStreamOnce.Do(func() { loadStream = CPU() })
	if !x.check("load_safetensors", C.mlx_load_safetensors(&m, &meta, cp, loadStream.c)) {
		return nil, x.err
	}
	out := map[string]*Array{}
	it := C.mlx_map_string_to_array_iterator_new(m)
	defer C.mlx_map_string_to_array_iterator_free(it)
	for {
		var key *C.char
		val := C.mlx_array_new()
		if C.mlx_map_string_to_array_iterator_next(&key, &val, it) != 0 {
			C.mlx_array_free(val)
			break
		}
		out[C.GoString(key)] = newArray(val)
	}
	return out, nil
}

// loadStream is the CPU stream every file load runs on (never freed).
var (
	loadStream     *Stream
	loadStreamOnce sync.Once
)
