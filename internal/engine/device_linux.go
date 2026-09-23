//go:build mlx && linux

package engine

import (
	"bufio"
	"os"
	"strings"
)

func cpuName() string {
	f, err := os.Open("/proc/cpuinfo")
	if err != nil {
		return "CPU"
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if k, v, ok := strings.Cut(sc.Text(), ":"); ok && strings.TrimSpace(k) == "model name" {
			return strings.TrimSpace(v)
		}
	}
	return "CPU"
}
