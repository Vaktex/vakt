//go:build mlx && linux && !cuda

package engine

const gpuBackend = "cpu"

func gpuOK() bool { return false }

func gpuName(int) string { return cpuName() }
