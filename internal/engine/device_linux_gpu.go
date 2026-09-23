//go:build mlx && linux && cuda

package engine

import (
	"fmt"
	"os/exec"
	"strings"
)

const gpuBackend = "cuda"

func gpuOK() bool { return cudaAvailable() }

func gpuName(i int) string {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader", fmt.Sprintf("--id=%d", i)).Output() // #nosec G204 -- fixed binary, int arg
	if err != nil {
		return fmt.Sprintf("NVIDIA GPU %d", i)
	}
	return fmt.Sprintf("%s (%d)", strings.TrimSpace(string(out)), i)
}
