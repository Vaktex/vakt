//go:build mlx && darwin

package engine

import (
	"os/exec"
	"strings"

	"github.com/vaktex/vakt/internal/engine/mlx"
)

const gpuBackend = "metal"

func gpuOK() bool { return mlx.MetalAvailable() }

func cpuName() string {
	out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output() // #nosec G204 -- fixed argv
	if err != nil {
		return "Apple Silicon"
	}
	return strings.TrimSpace(string(out))
}

func gpuName(int) string { return cpuName() + " GPU" }
