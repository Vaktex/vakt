//go:build mlx && darwin

package mlx

/*
#include <stdlib.h>
#include <string.h>
void mlx_vakt_set_embedded_metallib(const void* data, size_t size);
*/
import "C"

import (
	_ "embed"
	"unsafe"
)

// The MLX Metal kernels, compiled at build time (third_party/mlx/build.sh)
// and copied here by `go generate` / the Makefile. Embedding them keeps the
// release a single file: no colocated mlx.metallib, no temp files.
//
//go:generate cp ../../../third_party/mlx/install/darwin_arm64_metal/lib/mlx.metallib metallib/mlx.metallib
//go:embed metallib/mlx.metallib
var metallib []byte

// registerMetallib hands MLX a C-heap copy of the library. The copy is never
// freed: MLX may (re)create the library for each device, and the pointer
// must stay valid and immovable for the life of the process.
func registerMetallib() {
	if len(metallib) == 0 {
		return
	}
	p := C.malloc(C.size_t(len(metallib)))
	C.memcpy(p, unsafe.Pointer(unsafe.SliceData(metallib)), C.size_t(len(metallib)))
	C.mlx_vakt_set_embedded_metallib(p, C.size_t(len(metallib)))
}

func gpuAvailable() bool { return MetalAvailable() }
