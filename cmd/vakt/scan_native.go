//go:build mlx

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/vaktex/vakt/internal/ast"
	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine"
	"github.com/vaktex/vakt/internal/hub"
	"github.com/vaktex/vakt/internal/pipeline"
	"github.com/vaktex/vakt/internal/report"
	"github.com/vaktex/vakt/internal/tokenize"
	"github.com/vaktex/vakt/internal/walk"
)

// The native build links the real pipeline and engine into the CLI.
func init() {
	runScan = nativeScan
	doctorBench = nativeBench
}

func nativeScan(ctx context.Context, o ScanOptions, prog *report.Progress) (*report.Report, error) {
	path, sha, err := resolveModel(ctx, o, os.Stderr)
	if err != nil {
		return nil, err
	}
	tok, err := tokenize.New()
	if err != nil {
		return nil, err
	}
	defer tok.Close()

	devices := o.Devices
	if len(devices) == 0 {
		devices = []int{0}
	}
	var engines []core.Engine
	defer func() {
		for _, e := range engines {
			_ = e.Close() // scan result or error already decided
		}
	}()
	// VAKT_OPEN_EARLY=1 loads the model and compiles its GPU kernels (with
	// a warm-up batch) while the pipeline walks, parses and tokenizes,
	// instead of before it starts. Off by default until measured
	// (scripts/abbench.sh): one run looked slower, but the same code also
	// varied by 15% between sessions.
	early := os.Getenv("VAKT_OPEN_EARLY") == "1"
	open := func(ctx context.Context) ([]core.Engine, error) {
		for _, d := range devices {
			e, err := engine.Open(engine.Options{ModelPath: path, ModelSHA: sha, Precision: o.Precision, Device: o.Device, DeviceIndex: d})
			if err != nil {
				return nil, err
			}
			engines = append(engines, e)
			if early {
				if err := warmUp(ctx, e); err != nil {
					return nil, err
				}
			}
			if o.Device == "cpu" || e.Info().Backend != "cuda" {
				break // one engine per GPU; CPU and Metal have a single device
			}
		}
		return engines, nil
	}

	rev := o.ModelRevision
	repo := o.ModelRepo
	if o.ModelPath != "" {
		repo, rev = "local:"+filepath.Base(o.ModelPath), ""
	}
	iso, err := ast.NewIsolated(0)
	if err != nil {
		return nil, err
	}
	defer iso.Close()
	cfg := pipeline.Config{
		Isolate:     iso,
		Root:        o.Root,
		Walk:        walk.Options{Include: o.Include, Exclude: o.Exclude, MaxFileBytes: o.MaxFileBytes, FollowSymlinks: o.FollowSymlinks, NoRepoIgnores: o.NoRepoIgnores, Jobs: o.Jobs},
		AST:         ast.Options{NoResidual: !o.TopLevel},
		BatchTokens: o.BatchTokens,
		Jobs:        o.Jobs,
		MinTokens:   o.MinTokens,
		Threshold:   o.Threshold,
		NoCache:     o.NoCache,
		CacheDir:    hub.ScoresDir(),
		ModelRepo:   repo, ModelRevision: rev, Precision: o.Precision,
	}
	if !early {
		es, err := open(ctx)
		if err != nil {
			return nil, err
		}
		return pipeline.Run(ctx, cfg, es, tok, prog)
	}
	return pipeline.RunOpening(ctx, cfg, open, tok, prog)
}

// warmUp scores one short sequence so the GPU kernels are compiled before
// the first real batch.
func warmUp(ctx context.Context, e core.Engine) error {
	ids := make([]int32, 64)
	for i := range ids {
		ids[i] = int32(1000 + i) // #nosec G115 -- small constants
	}
	_, err := e.Score(ctx, [][]int32{ids})
	return err
}

// nativeBench loads the cached model and times one forward pass.
func nativeBench(ctx context.Context) (string, error) {
	path, sha, err := hub.Cached(hub.Options{Repo: brand.ModelRepo, Revision: "main"})
	if err != nil {
		return "", errors.New("model not cached; run `" + brand.Binary + " summon` first")
	}
	e, err := engine.Open(engine.Options{ModelPath: path, ModelSHA: sha, Precision: defaultPrecision, Device: "auto"})
	if err != nil {
		return "", err
	}
	defer e.Close()
	const T, B = 1024, 4
	batch := make([][]int32, B)
	for i := range batch {
		batch[i] = make([]int32, T)
		for j := range batch[i] {
			batch[i][j] = int32((i*7919 + j*104729) % 150000) // #nosec G115 -- small constants
		}
	}
	if _, err := e.Score(ctx, batch); err != nil { // warm-up
		return "", err
	}
	start := time.Now()
	if _, err := e.Score(ctx, batch); err != nil {
		return "", err
	}
	d := time.Since(start)
	info := e.Info()
	return fmt.Sprintf("%.0f tokens/s on %s (%s, %s)", float64(T*B)/d.Seconds(), info.Device, info.Backend, info.Precision), nil
}
