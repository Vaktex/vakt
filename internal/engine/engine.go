// Package engine runs DOM-0.8B. The real implementation is MLX behind cgo
// (Metal on darwin, CUDA or CPU on linux) and is selected by build tags; the
// fake engine is always available for tests and for developing the pipeline
// without model weights.
package engine

import (
	"errors"

	"github.com/vaktex/vakt/internal/core"
)

// Options configure Open.
type Options struct {
	// ModelPath is a local model.safetensors path (already downloaded).
	ModelPath string
	// Precision is "fp32" (default, parity-grade) or "bf16".
	Precision string
	// Device is "auto", "gpu" or "cpu".
	Device string
	// DeviceIndex selects a GPU when several are present (CUDA only).
	DeviceIndex int
	// ModelSHA is the file's sha256 if already known (hub computes it while
	// downloading); empty means the engine hashes the file itself.
	ModelSHA string
}

// ErrUnavailable is returned by Open when the binary was built without a
// native backend or the requested device cannot be used.
var ErrUnavailable = errors.New("engine: native backend unavailable")

// Open loads the model. It is replaced per build tag; see open_*.go.
func Open(opts Options) (core.Engine, error) { return open(opts) }
