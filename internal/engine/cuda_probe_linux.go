//go:build mlx && linux && cuda

package engine

import "github.com/vaktex/vakt/internal/engine/mlx"

// cudaAvailable reports whether MLX sees at least one CUDA device. MLX's
// CUDA backend loads cuBLAS/cuDNN/NVRTC dynamically; if the host lacks the
// CUDA 13 runtime, device discovery fails and we fall back to CPU.
func cudaAvailable() bool { return mlx.GPUCount() > 0 }
