//go:build !mlx

package engine

import "github.com/vaktex/vakt/internal/core"

// Without the mlx build tag there is no native backend.
func open(Options) (core.Engine, error) { return nil, ErrUnavailable }
