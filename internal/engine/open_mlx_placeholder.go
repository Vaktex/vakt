//go:build mlx && !mlxnative

package engine

import "github.com/vaktex/vakt/internal/core"

// Placeholder so every documented tag combination builds before the native
// engine lands. The engine stream replaces this file with the MLX
// implementation (and drops the mlxnative guard).
func open(Options) (core.Engine, error) { return nil, ErrUnavailable }
