//go:build mlx

package engine

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"

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
	default:
		return nil, fmt.Errorf("engine: precision must be fp32, tf32 or bf16, got %q", prec)
	}

	// MLX reads MLX_ENABLE_TF32 once per process, so the matmul mode is fixed
	// by the first engine; a later engine asking for another mode is refused
	// rather than silently running in the wrong one.
	if err := pinMatmulMode(prec == "tf32"); err != nil {
		return nil, err
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
	// MLX's CUDA build allocates every array (CPU stream included) with
	// cudaMallocManaged, so without a working NVIDIA driver it cannot run
	// at all: say so instead of failing on the first array.
	if gpuBackend == "cuda" && !gpuOK() {
		return nil, fmt.Errorf("%w: this is the CUDA build of %s and no usable NVIDIA GPU/driver was found; install the CPU build (install.sh picks it automatically)", ErrUnavailable, brand.Binary)
	}
	var s *mlx.Stream
	backend, device := "cpu", cpuName()
	switch opts.Device {
	case "", "auto":
		if gpuOK() {
			if s, err = mlx.GPUDevice(opts.DeviceIndex); err != nil {
				return nil, err
			}
			backend, device = gpuBackend, gpuName(opts.DeviceIndex)
		} else {
			s = mlx.CPU()
		}
	case "gpu":
		if !gpuOK() {
			return nil, fmt.Errorf("%w: no usable %s GPU", ErrUnavailable, gpuBackend)
		}
		if s, err = mlx.GPUDevice(opts.DeviceIndex); err != nil {
			return nil, err
		}
		backend, device = gpuBackend, gpuName(opts.DeviceIndex)
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
	// Re-check what MLX actually loaded: the file is opened twice (header
	// validation, then MLX), so a same-user swap in between must not be
	// able to shrink a tensor below the shape the forward pass indexes.
	if err := checkLoaded(raw, safetensors.DOMExpectations(prefix)); err != nil {
		for _, a := range raw {
			a.Free()
		}
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
	m := &model{w: w, delta: deltaChunkMode, evalEvery: evalEveryFromEnv(1)}
	switch os.Getenv("VAKT_DELTANET") {
	case "scan":
		m.delta = deltaScan
	case "chunked":
		m.delta = deltaChunkMode
	}
	if backend == "metal" && os.Getenv("VAKT_DELTANET") == "" {
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
	// Bound the padded batch: the pipeline sizes batches with
	// MaxBatchTokens, but Score must be safe on its own.
	if B > maxBatchSeqs || B*T > max(e.maxBT, core.MaxTokens) {
		return nil, fmt.Errorf("engine: batch of %d x %d padded tokens exceeds the limit (%d tokens, %d sequences)", B, T, max(e.maxBT, core.MaxTokens), maxBatchSeqs)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// MLX graph building/eval pins the OS thread for the Metal command queue.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	x := mlx.NewCtx(e.s)
	defer x.Free()
	if B*T > core.MaxTokens/2 {
		// Long batches leave large activation buffers in MLX's cache;
		// release them so the footprint falls back between big batches.
		defer mlx.ClearCache()
	}

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

	e.m.cancelled = func() bool { return ctx.Err() != nil }
	defer func() { e.m.cancelled = nil }()
	hidden := e.m.forward(x, ids, mask3, lengths, B, T) // [B, T, H] f32
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	w := e.m.w
	if !w.heads {
		return nil, errors.New("engine: checkpoint has no classification heads (base model)")
	}
	pooled, _ := attentionPool(x, w.pool, hidden, mask3, B, T) // [B, H] f32
	sev := x.Sigmoid(mlpHead(x, w.bin, pooled))                // [B, 1]
	fam := x.Sigmoid(mlpHead(x, w.aux, pooled))                // [B, 18]
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
	for _, v := range sv {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("engine: model produced a non-finite score (corrupt weights or numerical overflow)")
		}
	}
	for _, v := range fv {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("engine: model produced a non-finite family probability")
		}
	}
	out := make([]core.Scores, B)
	for i := range out {
		out[i].Severity = sv[i]
		copy(out[i].Families[:], fv[i*core.NumFamilies:(i+1)*core.NumFamilies])
	}
	return out, nil
}

// Attention-pool constants (experiment/pooling.py).
const (
	poolHeads   = 4
	poolHeadDim = hidden / poolHeads // 256
	poolMask    = -1e4               // masked logit: zero weight, finite in fp16
	layerNormEp = 1e-5               // PyTorch LayerNorm default (pool.norm, net.0)
)

// attentionPool is AttentionPool.forward in float32: per-head scores
// K·q/sqrt(256), masked to -1e4 BEFORE the softmax, then the weighted sum of
// V, concatenated, projected and LayerNormed. It also returns the per-token
// weights averaged over heads ([B, T]) for parity tests.
func attentionPool(x *mlx.Ctx, p poolW, hidden, mask3 *mlx.Array, B, T int) (*mlx.Array, *mlx.Array) {
	// Zero padded positions first: they get attention weight 0, but
	// 0 x NaN is NaN, so a non-finite padded state would poison the row.
	h := x.Where(mask3, x.AsType(hidden, mlx.Float32), x.Scalar(0))     // [B, T, H]
	k := x.Reshape(x.Matmul(h, p.keyT), B, T, poolHeads, poolHeadDim)   // [B, T, 4, 256]
	v := x.Reshape(x.Matmul(h, p.valueT), B, T, poolHeads, poolHeadDim) // [B, T, 4, 256]
	// scores[b,h,t] = sum_d k[b,t,h,d] * q[h,d]
	q := x.Reshape(p.query, 1, 1, poolHeads, poolHeadDim)
	scores := x.Sum(x.Multiply(k, q), false, -1) // [B, T, 4]
	scores = x.Multiply(scores, x.Scalar(float32(1/math.Sqrt(poolHeadDim))))
	// Mask before the softmax (padded positions get -1e4, as in training).
	scores = x.Where(mask3, scores, x.Scalar(poolMask))
	wts := x.Softmax(scores, 1) // over T: [B, T, 4]
	// pooled[b,h,d] = sum_t w[b,t,h] * v[b,t,h,d]
	pooled := x.Sum(x.Multiply(x.ExpandDims(wts, 3), v), false, 1) // [B, 4, 256]
	pooled = x.Reshape(pooled, B, poolHeads*poolHeadDim)
	pooled = x.Add(x.Matmul(pooled, p.projT), p.projB)
	pooled = x.LayerNorm(pooled, p.normW, p.normB, layerNormEp)
	meanW := x.Mean(wts, false, 2) // [B, T], averaged over heads
	return pooled, meanW
}

// mlpHead is MLPHead: LayerNorm -> Linear -> exact GELU -> Linear (dropout
// is the identity at inference). Returns logits.
func mlpHead(x *mlx.Ctx, p headW, in *mlx.Array) *mlx.Array {
	z := x.LayerNorm(in, p.lnW, p.lnB, layerNormEp)
	z = x.Gelu(x.Add(x.Matmul(z, p.w1T), p.b1))
	return x.Add(x.Matmul(z, p.w2T), p.b2)
}

// pooledForTest returns, for one sequence, the attention-pooled vector, the
// final-norm hidden states and the per-token pool weights (parity tests).
func (e *mlxEngine) pooledForTest(ids []int32) ([]float32, []float32, error) {
	pooled, norm, _, err := e.poolForTest(ids)
	return pooled, norm, err
}

func (e *mlxEngine) poolForTest(ids []int32) (pooled, norm, weights []float32, err error) {
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
	if norm, err = x.Float32s(h); err != nil {
		return nil, nil, nil, err
	}
	if !e.m.w.heads {
		pooled, err = x.Float32s(x.Mean(x.AsType(h, mlx.Float32), false, 1))
		return pooled, norm, nil, err
	}
	p, wts := attentionPool(x, e.m.w.pool, h, mask3, 1, T)
	if pooled, err = x.Float32s(p); err != nil {
		return nil, nil, nil, err
	}
	weights, err = x.Float32s(wts)
	return pooled, norm, weights, err
}

// maxBatchTokens sizes the padded-token budget of one Score call to the
// device's memory. Measured on the fp32 mock: weights ~3 GB, and a full
// 32k-token batch peaks near 15 GB (~0.37 MB per padded token), so the
// budget is (limit - weights - headroom) / 0.4 MB, clamped to
// [MaxTokens, 32768]. A single MaxTokens sequence is always allowed.
func maxBatchTokens(backend string) int {
	if backend == "cpu" {
		return 8192
	}
	const perToken = 400 << 10
	const reserve = 4 << 30 // weights + headroom
	limit := mlx.MemoryLimit()
	if limit <= reserve {
		return core.MaxTokens
	}
	// Clamp before converting (limit is a uint64 byte count).
	n := min((limit-reserve)/perToken, 32768)
	return max(int(n), core.MaxTokens) // #nosec G115 -- n <= 32768
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

// maxBatchSeqs caps the number of sequences in one Score call.
const maxBatchSeqs = 256

// MaxBatchSeqs is the most sequences one Score call accepts.
func (e *mlxEngine) MaxBatchSeqs() int { return maxBatchSeqs }

// checkLoaded verifies every expected tensor's loaded shape and dtype.
func checkLoaded(raw map[string]*mlx.Array, want []safetensors.Expect) error {
	for _, e := range want {
		a := raw[e.Name]
		if a == nil {
			return fmt.Errorf("engine: model changed while loading: %s missing", e.Name)
		}
		got := a.Shape()
		if len(got) != len(e.Shape) {
			return fmt.Errorf("engine: model changed while loading: %s has shape %v", e.Name, got)
		}
		for i := range got {
			if int64(got[i]) != e.Shape[i] {
				return fmt.Errorf("engine: model changed while loading: %s has shape %v", e.Name, got)
			}
		}
		switch a.Dtype() {
		case mlx.Float32, mlx.BFloat16, mlx.Float16:
		default:
			return fmt.Errorf("engine: model changed while loading: %s has dtype %s", e.Name, a.Dtype())
		}
	}
	return nil
}

var (
	matmulModeSet  bool
	matmulModeTF32 bool
	matmulModeMu   sync.Mutex
)

func pinMatmulMode(tf32 bool) error {
	matmulModeMu.Lock()
	defer matmulModeMu.Unlock()
	if !matmulModeSet {
		matmulModeSet, matmulModeTF32 = true, tf32
		v := "0"
		if tf32 {
			v = "1"
		}
		// Must happen before MLX's first matmul (mlx.Init also defaults it).
		_ = os.Setenv("MLX_ENABLE_TF32", v)
		return nil
	}
	if matmulModeTF32 != tf32 {
		return errors.New("engine: fp32 and tf32 engines cannot share a process (MLX fixes the matmul mode on first use)")
	}
	return nil
}
