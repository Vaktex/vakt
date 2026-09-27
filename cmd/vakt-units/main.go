// Command vakt-units prints the units `vakt patrol` would score, as JSONL.
//
// It runs the scan's own walk (ignore files, skip dirs, size limits) and
// ast.Extract with the same options `vakt patrol` uses by default, and stops
// there: no tokenizer, no engine. Training data is cut with it, so the model
// is trained on exactly the spans, kinds and language tags it will be asked
// to score.
//
//	vakt-units [--top-level] [--max-file-bytes N] <root> > units.jsonl
//
// One JSON object per unit: file, language, kind, name, start_line,
// end_line, code (exactly what the scanner renders, before PromptCode's
// TrimSpace), and parse ("ok", or why the file fell back to a whole-file
// unit).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/vaktex/vakt/internal/ast"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/walk"
)

type row struct {
	File      string `json:"file"`
	Language  string `json:"language"`
	Kind      string `json:"kind"`
	Name      string `json:"name,omitempty"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Code      string `json:"code"`
	Parse     string `json:"parse"`
}

func main() {
	// ast.NewIsolated re-executes this binary as a parse worker; serve and
	// exit when that is what we were started as.
	ast.MaybeServeWorker()

	topLevel := flag.Bool("top-level", false, "also emit top-level residual units (vakt patrol --top-level)")
	maxBytes := flag.Int64("max-file-bytes", 0, "skip larger files (0: the scan's default)")
	jobs := flag.Int("jobs", runtime.NumCPU(), "parse workers")
	limit := flag.Int64("limit", 0, "stop after emitting this many units (0: unlimited)")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: vakt-units [--top-level] <root>")
		os.Exit(2)
	}

	ctx, stopSignal := signal.NotifyContext(context.Background(), os.Interrupt)
	ctx, stop := context.WithCancel(ctx)
	defer stopSignal()
	defer stop()

	iso, err := ast.NewIsolated(0) // same memory/time budget as a scan
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer iso.Close()

	files := make(chan walk.File, *jobs)
	skips := make(chan walk.Skip, 256)
	go func() {
		for range skips {
		}
	}()

	var walkErr error
	go func() {
		defer close(files)
		walkErr = walk.Walk(ctx, flag.Arg(0), walk.Options{MaxFileBytes: *maxBytes, Jobs: *jobs}, files, skips)
	}()

	out := bufio.NewWriterSize(os.Stdout, 1<<20)
	defer out.Flush()
	var mu sync.Mutex
	enc := json.NewEncoder(out)

	opts := ast.Options{NoResidual: !*topLevel}
	var emitted atomic.Int64
	var wg sync.WaitGroup
	for range *jobs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range files {
				lang, ok := ast.Detect(f.Rel, f.Data[:min(len(f.Data), 8192)])
				if !ok {
					continue
				}
				units, err := iso.Extract(ctx, f.Rel, lang, f.Data, opts)
				parse := "ok"
				switch {
				case errors.Is(err, ast.ErrParseTimeout):
					parse = "timeout"
				case errors.Is(err, ast.ErrParseLimit):
					parse = "limit"
				case err != nil && len(units) == 0:
					continue
				case err != nil:
					parse = "recovered"
				}
				mu.Lock()
				for _, u := range units {
					if *limit > 0 && emitted.Load() >= *limit {
						stop()
						break
					}
					_ = enc.Encode(toRow(u, parse))
					if *limit > 0 && emitted.Add(1) >= *limit {
						stop() // stops the walk and every parse worker promptly
						break
					}
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	close(skips)
	if walkErr != nil && !errors.Is(walkErr, context.Canceled) {
		fmt.Fprintln(os.Stderr, walkErr)
		os.Exit(1)
	}
}

func toRow(u core.Unit, parse string) row {
	return row{File: u.File, Language: u.Language, Kind: u.Kind, Name: u.Name,
		StartLine: u.StartLine, EndLine: u.EndLine, Code: u.Code, Parse: parse}
}
