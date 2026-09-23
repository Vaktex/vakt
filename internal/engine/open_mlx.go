//go:build mlx

package engine

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine/mlx"
	"github.com/vaktex/vakt/internal/engine/safetensors"
)

// padID fills padded positions. Its value never matters: embeddings of
// padded positions are only ever combined with real positions through
// causal (left-to-right) operations, and pooling masks them out.
const padID = 248044

type mlxEngine struct {
	s     *mlx.Stream
	m     *model
	info  core.EngineInfo
	maxBT int
}

func open(opts Options) (core.Engine, error) {
	if opts.ModelPath == "" {
		return nil, errors.New("engine: no model path")
	}
	prec := opts.Precision
	if prec == "" {
		prec = "fp32"
	}
	switch prec {
	case "fp32", "bf16":
	case "tf32":
		// fp32 weights and activations with TF32 tensor-core matmuls on the
		// GPU: ~1.7x faster than strict fp32, ~1e-3 drift. MLX reads this
		// once, before its first op, so it must be set before anything
		// below touches MLX.
		_ = os.Setenv("MLX_ENABLE_TF32", "1")
	default:
		return nil, fmt.Errorf("engine: precision must be fp32, tf32 or bf16, got %q", prec)
	}

	// 1. The file is untrusted: validate its header before MLX parses it.
	hdr, err := safetensors.ReadHeader(opts.ModelPath, safetensors.Limits{})
	if err != nil {
		return nil, fmt.Errorf("engine: %s: %w", brand.ModelFile, err)
	}
	prefix, heads := safetensors.BackbonePrefix, true
	if _, ok := hdr.Tensors[safetensors.BackbonePrefix+"embed_tokens.weight"]; !ok {
		prefix, heads = safetensors.BaseModelPrefix, false // base checkpoint (dev only)
	}
	if err := hdr.Require(safetensors.DOMExpectations(prefix)); err != nil {
		return nil, fmt.Errorf("engine: %s is not a DOM-0.8B checkpoint: %w", brand.ModelFile, err)
	}
	sha := opts.ModelSHA
	if sha == "" {
		if sha, err = safetensors.SHA256File(context.Background(), opts.ModelPath); err != nil {
			return nil, err
		}
	}

	// 2. Device.
	var s *mlx.Stream
	backend, device := "cpu", cpuName()
	switch opts.Device {
	case "", "auto":
		if gpuOK() {
			s, backend, device = mlx.GPU(), gpuBackend, gpuName(opts.DeviceIndex)
		} else {
			s = mlx.CPU()
		}
	case "gpu":
		if !gpuOK() {
			return nil, fmt.Errorf("%w: no usable %s GPU", ErrUnavailable, gpuBackend)
		}
		s, backend, device = mlx.GPU(), gpuBackend, gpuName(opts.DeviceIndex)
	case "cpu":
		s = mlx.CPU()
	default:
		return nil, fmt.Errorf("engine: device must be auto, gpu or cpu, got %q", opts.Device)
	}
	configureMemory(backend)

	// 3. Load and lay out the weights.
	x := mlx.NewCtx(s)
	defer x.Free()
	raw, err := x.LoadSafetensors(opts.ModelPath)
	if err != nil {
		s.Free()
		return nil, err
	}
	compute := mlx.Float32
	if prec == "bf16" {
		compute = mlx.BFloat16
	}
	w, err := loadWeights(x, raw, prefix, compute, heads)
	if err != nil {
		s.Free()
		return nil, err
	}
	m := &model{w: w, delta: deltaScan, evalEvery: evalEveryFromEnv(1)}
	if backend == "metal" && os.Getenv("VAKT_DELTANET") != "scan" {
		m.delta, m.kern = deltaKernel, newDeltaKernel()
		m.convKern = newConvKernel()
	}
	return &mlxEngine{
		s: s, m: m, maxBT: maxBatchTokens(backend),
		info: core.EngineInfo{Backend: backend, Device: device, Precision: prec, ModelSHA: strings.ToLower(sha)},
	}, nil
}

func (e *mlxEngine) Info() core.EngineInfo { return e.info }

func (e *mlxEngine) MaxBatchTokens() int { return e.maxBT }

func (e *mlxEngine) Close() error {
	if e.m != nil {
		e.m.w.free()
		e.m.kern.Free()
		e.m.convKern.Free()
		e.m = nil
	}
	e.s.Free()
	mlx.ClearCache()
	return nil
}

// Score runs one right-padded batch.
func (e *mlxEngine) Score(ctx context.Context, batch [][]int32) ([]core.Scores, error) {
	if e.m == nil {
		return nil, errors.New("engine: closed")
	}
	if len(batch) == 0 {
		return nil, nil
	}
	B, T := len(batch), 0
	lengths := make([]int, B)
	for i, ids := range batch {
		if len(ids) == 0 || len(ids) > core.MaxTokens {
			return nil, fmt.Errorf("engine: sequence %d has %d tokens (want 1..%d)", i, len(ids), core.MaxTokens)
		}
		for _, id := range ids {
			if id < 0 || int(id) >= core.VocabSize {
				return nil, fmt.Errorf("engine: sequence %d has token id %d outside the vocabulary", i, id)
			}
		}
		lengths[i] = len(ids)
		T = max(T, len(ids))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// MLX graph building/eval pins the OS thread for the Metal command queue.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	x := mlx.NewCtx(e.s)
	defer x.Free()

	flat := make([]int32, B*T)
	maskv := make([]bool, B*T)
	for i, ids := range batch {
		copy(flat[i*T:], ids)
		for t := range T {
			if t < len(ids) {
				maskv[i*T+t] = true
			} else {
				flat[i*T+t] = padID
			}
		}
	}
	ids := x.FromInt32(flat, B, T)
	mask3 := x.FromBool(maskv, B, T, 1)

	hidden := e.m.forward(x, ids, mask3, lengths, B, T) // [B, T, H] f32

	// Masked mean pool in float32; where() so padded NaN/Inf can't leak.
	hf := x.AsType(hidden, mlx.Float32)
	sum := x.Sum(x.Where(mask3, hf, x.Scalar(0)), false, 1) // [B, H]
	cnt := make([]float32, B)
	for i, n := range lengths {
		cnt[i] = float32(n)
	}
	pooled := x.Divide(sum, x.FromFloat32(cnt, B, 1))

	w := e.m.w
	if w.binW == nil {
		return nil, errors.New("engine: checkpoint has no classification heads (base model)")
	}
	sev := x.Sigmoid(x.Add(x.Matmul(pooled, w.binW), w.binB)) // [B, 1]
	fam := x.Sigmoid(x.Add(x.Matmul(pooled, w.auxW), w.auxB)) // [B, 18]
	if err := x.Eval(sev, fam); err != nil {
		return nil, err
	}
	sv, err := x.Float32s(sev)
	if err != nil {
		return nil, err
	}
	fv, err := x.Float32s(fam)
	if err != nil {
		return nil, err
	}
	out := make([]core.Scores, B)
	for i := range out {
		out[i].Severity = sv[i]
		copy(out[i].Families[:], fv[i*core.NumFamilies:(i+1)*core.NumFamilies])
	}
	return out, nil
}

// pooledForTest returns the masked mean pool for one sequence (parity tests).
func (e *mlxEngine) pooledForTest(ids []int32) ([]float32, []float32, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	x := mlx.NewCtx(e.s)
	defer x.Free()
	T := len(ids)
	a := x.FromInt32(ids, 1, T)
	mv := make([]bool, T)
	for i := range mv {
		mv[i] = true
	}
	mask3 := x.FromBool(mv, 1, T, 1)
	h := e.m.forward(x, a, mask3, []int{T}, 1, T)
	norm, err := x.Float32s(h)
	if err != nil {
		return nil, nil, err
	}
	pooled, err := x.Float32s(x.Mean(x.AsType(h, mlx.Float32), false, 1))
	return pooled, norm, err
}

func maxBatchTokens(backend string) int {
	if backend == "cpu" {
		return 8192
	}
	return 32768
}

func configureMemory(backend string) {
	if backend == "cpu" {
		return
	}
	// Keep a bounded buffer cache so long scans don't grow without limit.
	mlx.SetCacheLimit(2 << 30)
}

func evalEveryFromEnv(def int) int {
	if v := os.Getenv("VAKT_EVAL_EVERY"); v != "" {
		var n int
		if _, err := fmt.Sscan(v, &n); err == nil && n >= 0 {
			return n
		}
	}
	return def
}
