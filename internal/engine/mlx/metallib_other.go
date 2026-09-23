//go:build mlx && !darwin

package mlx

/*
#include "mlx/c/mlx.h"
*/
import "C"

// No Metal off darwin; CUDA kernels are compiled into libmlx (NVRTC JIT).
func registerMetallib() {}

// gpuAvailable reports whether a CUDA GPU stream can be created.
func gpuAvailable() bool {
	Init()
	var n C.int
	if C.mlx_device_count(&n, C.MLX_GPU) != 0 {
		return false
	}
	return n > 0
}
