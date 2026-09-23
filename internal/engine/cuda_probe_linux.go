//go:build mlx && linux && cuda

package engine

import "github.com/vaktex/vakt/internal/engine/mlx"

// cudaAvailable reports whether MLX sees at least one CUDA device.
//
// The CUDA build links libcuda, cuBLAS(Lt), NVRTC and cuDNN as shared
// libraries (only libcudart is static), so the binary needs the NVIDIA
// driver (>= 580) and the CUDA 13 runtime libraries present to start at all:
// the dynamic loader resolves them before main. install.sh checks for them
// and installs the CPU build when they are missing; this probe only covers a
// driver that is present but has no usable device.
func cudaAvailable() bool { return mlx.GPUCount() > 0 }
