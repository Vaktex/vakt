// Package pipeline runs a scan: walk the tree, extract units, tokenize,
// consult the cache, batch by length, score on the engine, build the report.
//
//	walk ─► parse workers ─► tokenize workers ─► batcher ─► engine (1 goroutine per device)
//	                                         └─► cache hits ─────────────┐
//	                                                                     ▼
//	                                                                  results
//
// CPU stages run with cfg.Jobs workers. The engine is driven by exactly one
// goroutine per device (core.Engine is not concurrent).
package pipeline

import (
	"context"
	"errors"
	"fmt"
	"golang.org/x/sync/semaphore"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/vaktex/vakt/internal/ast"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/report"
	"github.com/vaktex/vakt/internal/tokenize"
	"github.com/vaktex/vakt/internal/walk"
)

// Config configures Run (CONTRACTS.md).
type Config struct {
	Root        string
	Walk        walk.Options
	AST         ast.Options
	BatchTokens int // padded tokens per batch (0: engine's MaxBatchTokens)
	Jobs        int
	MinTokens   int     // units with fewer prompt tokens are skipped (0: keep all)
	Threshold   float64 // for the report's flagged counts
	NoCache     bool
	CacheDir    string

	ModelRepo, ModelRevision, Precision string
}

// Run scans cfg.Root with the given engines (one per device) and tokenizer.
func Run(ctx context.Context, cfg Config, engines []core.Engine, tok core.Tokenizer, prog *report.Progress) (*report.Report, error) {
	if len(engines) == 0 {
		return nil, errors.New("pipeline: no engine")
	}
	if prog == nil {
		prog = &report.Progress{}
	}
	if prog.Started.IsZero() {
		prog.Started = time.Now()
	}
	if cfg.Jobs <= 0 {
		cfg.Jobs = 8
	}
	info := engines[0].Info()
	budget := cfg.BatchTokens
	if bs, ok := engines[0].(core.BatchSizer); ok && (budget <= 0 || bs.MaxBatchTokens() < budget) {
		budget = bs.MaxBatchTokens()
	}
	if budget <= 0 {
		budget = 4096 // measured: small padded batches beat large ones (less padding, less memory pressure)
	}
	maxSeqs := 256
	if bs, ok := engines[0].(interface{ MaxBatchSeqs() int }); ok && bs.MaxBatchSeqs() > 0 {
		maxSeqs = bs.MaxBatchSeqs()
	}

	var cache *Cache
	if !cfg.NoCache && cfg.CacheDir != "" {
		c, err := OpenCache(cfg.CacheDir)
		if err == nil {
			cache = c
			defer cache.Close()
		}
	}

	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)

	// Each walk.File carries its contents (up to MaxFileBytes), so the files
	// channel holds only a couple of files per parse worker; with the walker's
	// own in-flight sends that bounds file bytes to roughly
	// (2*Jobs + walk jobs) x MaxFileBytes.
	files := make(chan walk.File, cfg.Jobs)
	skipsC := make(chan walk.Skip, 256)
	// Whole-file units carry up to MaxFileBytes of code; keep the buffer
	// small enough that it cannot pin gigabytes.
	units := make(chan core.Unit, 256)
	// Source bytes of units waiting to be tokenized are bounded separately:
	// whole-file units are up to MaxFileBytes each.
	inflight := semaphore.NewWeighted(maxInflightSourceBytes)
	encoded := make(chan core.Encoded, 1024)
	results := make(chan report.Result, 1024)

	var skips []report.Skip
	var skipsMu sync.Mutex
	addSkip := func(file, reason string) {
		skipsMu.Lock()
		skips = append(skips, report.Skip{File: file, Reason: reason})
		skipsMu.Unlock()
	}

	// Every goroutine started below is tracked: Run must not return while
	// any of them can still touch tok or the engines (the caller closes
	// them right after).
	var all sync.WaitGroup
	defer all.Wait()
	defer cancel(nil) // runs first: unblocks everything, then all.Wait

	// 1. Walk.
	var walkErr error
	var nFiles atomic.Int64
	var wgSkip sync.WaitGroup
	wgSkip.Add(1)
	all.Add(1)
	go func() {
		defer all.Done()
		defer wgSkip.Done()
		for s := range skipsC {
			addSkip(s.Rel, s.Reason)
		}
	}()
	walkDone := make(chan struct{})
	all.Add(1)
	go func() {
		defer all.Done()
		defer close(walkDone)
		defer close(files)
		wo := cfg.Walk
		if wo.Jobs <= 0 || wo.Jobs > 2*cfg.Jobs {
			// Walker goroutines each hold one read file while blocked on send.
			wo.Jobs = max(4, 2*cfg.Jobs)
		}
		walkErr = walk.Walk(ctx, cfg.Root, wo, files, skipsC)
	}()

	// 2. Parse.
	var wgParse sync.WaitGroup
	for range cfg.Jobs {
		wgParse.Add(1)
		all.Add(1)
		go func() {
			defer all.Done()
			defer wgParse.Done()
			for f := range files {
				prog.FilesFound.Add(1)
				// walk read the file through its root-bound handle; never
				// reopen f.Path (a path swap could escape the root).
				src := f.Data
				lang, ok := ast.Detect(f.Rel, src[:min(len(src), 8192)])
				if !ok {
					addSkip(f.Rel, "not source code")
					continue
				}
				us, err := ast.Extract(ctx, f.Rel, lang, src, cfg.AST)
				f.Data, src = nil, nil // units hold copies; let the file go
				if ctx.Err() != nil {
					return
				}
				if errors.Is(err, ast.ErrParseTimeout) {
					addSkip(f.Rel, "parse timed out; scored as a whole file")
				}
				prog.FilesParsed.Add(1)
				nFiles.Add(1)
				for _, u := range us {
					w := unitWeight(u)
					if err := inflight.Acquire(ctx, w); err != nil {
						return
					}
					select {
					case units <- u:
						prog.Units.Add(1)
					case <-ctx.Done():
						inflight.Release(w)
						return
					}
				}
			}
		}()
	}
	all.Add(1)
	go func() {
		defer all.Done()
		wgParse.Wait()
		<-walkDone // walk may still send skips until it returns
		close(skipsC)
		close(units)
	}()

	// 3. Tokenize (render, split oversize, encode, truncate).
	var wgTok sync.WaitGroup
	for range cfg.Jobs {
		wgTok.Add(1)
		all.Add(1)
		go func() {
			defer all.Done()
			defer wgTok.Done()
			for u := range units {
				w := unitWeight(u)
				es := encodeUnit(u, tok)
				inflight.Release(w)
				for _, e := range es {
					if cfg.MinTokens > 0 && len(e.IDs) < cfg.MinTokens {
						continue
					}
					prog.TokensTotal.Add(int64(len(e.IDs)))
					select {
					case encoded <- e:
					case <-ctx.Done():
						return
					}
				}
			}
		}()
	}
	all.Add(1)
	go func() { defer all.Done(); wgTok.Wait(); close(encoded) }()

	// 4. Cache lookup + batching + scoring.
	modelKey := info.ModelSHA
	scoreErr := make(chan error, len(engines))
	all.Add(1)
	go func() {
		defer all.Done()
		defer close(results)
		b := newBatcher(budget, maxSeqs)
		work := make(chan []core.Encoded, len(engines))
		var wgEng sync.WaitGroup
		for _, eng := range engines {
			wgEng.Add(1)
			go func(eng core.Engine) {
				defer wgEng.Done()
				for batch := range work {
					if err := scoreBatch(ctx, eng, batch, cache, modelKey, info.Precision, results, prog); err != nil {
						scoreErr <- err
						cancel(err)
						// Drain so the dispatcher never blocks.
						for range work {
						}
						return
					}
				}
			}(eng)
		}
		dispatch := func(bs [][]core.Encoded) bool {
			for _, bt := range bs {
				select {
				case work <- bt:
				case <-ctx.Done():
					return false
				}
			}
			return true
		}
		pending := make([]core.Encoded, 0, 256)
		flushCache := func() bool {
			if len(pending) == 0 {
				return true
			}
			keys := make([]string, len(pending))
			for i, e := range pending {
				keys[i] = core.CacheKey(modelKey, info.Precision, e.Key)
			}
			hits := cache.Get(keys)
			for i, e := range pending {
				if r, ok := hits[keys[i]]; ok {
					prog.CacheHits.Add(1)
					prog.UnitsScored.Add(1)
					prog.TokensScored.Add(int64(len(e.IDs)))
					select {
					case results <- report.Result{Unit: e.Unit, Tokens: r.Tokens, Scores: r.Scores, Cached: true, Truncated: r.Truncated}:
					case <-ctx.Done():
						return false
					}
					continue
				}
				if !dispatch(b.add(e)) {
					return false
				}
			}
			pending = pending[:0]
			return true
		}
		for e := range encoded {
			if ctx.Err() != nil {
				continue // drain so the tokenizers can exit
			}
			pending = append(pending, e)
			if len(pending) == cap(pending) {
				flushCache()
			}
		}
		if ctx.Err() == nil && flushCache() {
			dispatch(b.flush())
		}
		close(work)
		wgEng.Wait()
	}()

	// 5. Collect (always drain so every stage can finish).
	var out []report.Result
	for r := range results {
		out = append(out, r)
	}
	all.Wait()
	select {
	case err := <-scoreErr:
		return nil, err
	default:
	}
	if err := context.Cause(ctx); err != nil && !errors.Is(err, context.Canceled) {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if walkErr != nil {
		return nil, walkErr
	}

	sort.Slice(skips, func(i, j int) bool { return skips[i].File < skips[j].File })
	meta := report.Meta{
		Root: cfg.Root, StartedAt: prog.Started, Duration: time.Since(prog.Started),
		Files: int(nFiles.Load()), CacheHits: int(prog.CacheHits.Load()), Skipped: skips,
		ModelRepo: cfg.ModelRepo, ModelRevision: cfg.ModelRevision, ModelSHA: info.ModelSHA,
		Backend: info.Backend, Device: info.Device, Precision: info.Precision,
	}
	return report.Build(meta, out, cfg.Threshold), nil
}

// maxInflightSourceBytes bounds unit source waiting for the tokenizers.
const maxInflightSourceBytes = 64 << 20

// unitWeight is a unit's weight against maxInflightSourceBytes (at least 1,
// at most the whole budget so one huge unit can always proceed).
func unitWeight(u core.Unit) int64 {
	return min(max(int64(len(u.Code)), 1), maxInflightSourceBytes)
}

// encodeUnit renders the training prompt, splits units whose prompt exceeds
// the context, and truncates as a last resort (as training did).
func encodeUnit(u core.Unit, tok core.Tokenizer) []core.Encoded {
	// Fast path: most units fit, so tokenize once and done. Only units whose
	// prompt exceeds the context pay for SplitOversize's search.
	text := tokenize.Render(u.Language, u.Code)
	if ids, err := tok.Encode(text); err == nil && len(ids) > 0 && len(ids) <= core.MaxTokens {
		u.Code = "" // downstream needs only IDs; never retain the source
		return []core.Encoded{{Unit: u, IDs: ids, Key: tokenize.PromptKey(text)}}
	}
	// Oversize: SplitOversize bounds its search window near the budget, so
	// each probe tokenizes ~budget-sized text.
	count := func(code string) int { return tokenize.Count(tok, tokenize.Render(u.Language, code)) }
	parts := ast.SplitOversize(u, nil, count, core.MaxTokens)
	out := make([]core.Encoded, 0, len(parts))
	for _, p := range parts {
		text := tokenize.Render(p.Language, p.Code)
		ids, err := tok.Encode(text)
		if err != nil || len(ids) == 0 {
			continue
		}
		p.Code = "" // parts are substrings of the parent: drop them all
		e := core.Encoded{Unit: p, IDs: ids, Key: tokenize.PromptKey(text)}
		if len(ids) > core.MaxTokens {
			e.IDs, e.Truncated = ids[:core.MaxTokens], true
		}
		out = append(out, e)
	}
	return out
}

// scoreBatch runs one batch and emits results (and writes them to cache).
func scoreBatch(ctx context.Context, eng core.Engine, batch []core.Encoded, cache *Cache, modelKey, prec string, results chan<- report.Result, prog *report.Progress) error {
	ids := make([][]int32, len(batch))
	toks := 0
	for i, e := range batch {
		ids[i] = e.IDs
		toks += len(e.IDs)
	}
	scores, err := eng.Score(ctx, ids)
	if err != nil {
		return fmt.Errorf("pipeline: scoring %d units: %w", len(batch), err)
	}
	put := make(map[string]cached, len(batch))
	for i, e := range batch {
		put[core.CacheKey(modelKey, prec, e.Key)] = cached{Tokens: len(e.IDs), Truncated: e.Truncated, Scores: scores[i]}
		select {
		case results <- report.Result{Unit: e.Unit, Tokens: len(e.IDs), Scores: scores[i], Truncated: e.Truncated}:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	_ = cache.Put(put)
	prog.UnitsScored.Add(int64(len(batch)))
	prog.TokensScored.Add(int64(toks))
	return nil
}
