package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/hub"
	"github.com/vaktex/vakt/internal/report"
	"github.com/vaktex/vakt/internal/results"
	"github.com/vaktex/vakt/internal/walk"
)

// serveScorer scores in-memory files with a model that stays loaded between
// calls. It is used from one goroutine at a time.
type serveScorer interface {
	Score(ctx context.Context, root string, files []walk.File) (*report.Report, error)
	Close() error
}

// openServeScorer loads the model for serve. The native build replaces it;
// like runScan, the default explains what is missing.
var openServeScorer = func(ctx context.Context, o ScanOptions, stderr io.Writer) (serveScorer, error) {
	return nil, fmt.Errorf("the scanning pipeline isn't linked into this build of %s (backend: %s); "+
		"install a release build with a native engine (Metal or CUDA)", brand.Binary, brand.Backend)
}

// serveRequest asks for one file to be scored. Content, when present, is
// scored instead of the file on disk (an unsaved editor buffer).
type serveRequest struct {
	ID      json.RawMessage `json:"id"`
	Root    string          `json:"root"`
	File    string          `json:"file"`
	Content *string         `json:"content"`
}

func newServeCmd() *cobra.Command {
	o := ScanOptions{
		Root: ".", Format: "pretty", Top: defaultTop, MinTokens: defaultMinTokens,
		BatchTokens: defaultBatchTokens, Jobs: runtime.NumCPU(),
	}
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Score files on request with the model kept loaded (for editors)",
		Long: "serve loads " + brand.ModelName + " once and scores files sent to it, one JSON request\n" +
			"per line on stdin, answering one JSON line per request on stdout. Editors\n" +
			"use it to score code as it is typed: only functions that changed are run\n" +
			"through the model; the rest come from the score cache.\n\n" +
			"Request:  {\"id\":1,\"root\":\"/abs/project\",\"file\":\"src/app.py\",\"content\":\"...\"}\n" +
			"          content is optional: without it the file is read from disk.\n" +
			"Replies:  {\"type\":\"start\",...}                          once, on startup\n" +
			"          {\"id\":1,\"type\":\"result\",\"file\":...,\"units\":[...]}\n" +
			"          {\"id\":1,\"type\":\"error\",\"message\":\"...\"}\n" +
			"          {\"id\":1,\"type\":\"superseded\"}   a newer request for the same file arrived first\n\n" +
			"The model must already be downloaded (`" + brand.Binary + " summon`): serve never prompts.",
		Example: "  echo '{\"id\":1,\"root\":\"'$PWD'\",\"file\":\"app.py\"}' | " + brand.Binary + " serve",
		Args:    cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if err := o.finish("", false, cmd.Flags().Changed("revision")); err != nil {
				return &exitCodeError{code: exitError, err: err}
			}
			return serve(cmd.Context(), cmd.InOrStdin(), cmd.OutOrStdout(), cmd.ErrOrStderr(), o)
		},
	}
	f := cmd.Flags()
	f.Float64Var(&o.Threshold, "threshold", defaultThreshold, "flag functions with severity >= this score")
	f.Float64Var(&o.MinConfidence, "min-confidence", defaultMinConfidence, "minimum selected family score for findings")
	f.StringVar(&o.Model, "model", defaultModel, "a local model.safetensors path, or hf:owner/name[@revision]")
	f.StringVar(&o.ModelRevision, "revision", "", "model revision (overrides @revision in --model)")
	f.StringVar(&o.Precision, "precision", defaultPrecision, "compute precision: auto, fp32, tf32, bf16 or fp16")
	f.StringVar(&o.Device, "device", "auto", "device: auto, gpu or cpu")
	f.BoolVar(&o.TopLevel, "top-level", false, "also score top-level code outside functions")
	f.BoolVar(&o.NoCache, "no-cache", false, "don't read or write the score cache")
	f.Int64Var(&o.MaxFileBytes, "max-file-bytes", defaultMaxFileBytes, "refuse files larger than this")
	f.BoolVar(&o.SaveResults, "save-results", false, "store each result in .vakt under its root, keyed by the content hash")
	return cmd
}

// serveOut writes one JSON object per line; safe for concurrent use.
type serveOut struct {
	mu sync.Mutex
	w  io.Writer
}

func (s *serveOut) send(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_, _ = s.w.Write(append(b, '\n'))
}

type serveReply struct {
	ID      json.RawMessage `json:"id,omitempty"`
	Type    string          `json:"type"`
	File    string          `json:"file,omitempty"`
	Message string          `json:"message,omitempty"`
}

// serveResult is a "result" reply. Units is always present: an empty list
// means the file was scored and has nothing to report.
type serveResult struct {
	ID    json.RawMessage `json:"id,omitempty"`
	Type  string          `json:"type"`
	File  string          `json:"file"`
	Units []report.Unit   `json:"units"`
	Model report.Model    `json:"model"`
}

// serve answers requests from in until it is closed or ctx ends. Requests run
// one at a time; when several are queued for the same file, only the newest
// runs and the older ones are answered "superseded".
func serve(ctx context.Context, in io.Reader, stdout, stderr io.Writer, o ScanOptions) error {
	out := &serveOut{w: stdout}
	out.send(struct {
		Type          string      `json:"type"`
		SchemaVersion string      `json:"schema_version"`
		Tool          report.Tool `json:"tool"`
	}{"start", report.SchemaVersion, report.Tool{Name: brand.Product, Binary: brand.Binary, Version: brand.Version, Commit: brand.Commit}})

	var (
		mu      sync.Mutex
		pending = map[string]serveRequest{}
		order   []string
		closed  bool
		wake    = make(chan struct{}, 1)
	)
	signal := func() {
		select {
		case wake <- struct{}{}:
		default:
		}
	}
	go func() {
		sc := bufio.NewScanner(in)
		// A request carries a whole file, JSON-escaped: allow a few times the file cap.
		sc.Buffer(make([]byte, 0, 64<<10), int(min(4*o.MaxFileBytes+(1<<20), 256<<20)))
		for sc.Scan() {
			var req serveRequest
			if err := json.Unmarshal(sc.Bytes(), &req); err != nil {
				out.send(serveReply{Type: "error", Message: "invalid request: " + err.Error()})
				continue
			}
			key := req.Root + "\x00" + req.File
			mu.Lock()
			if old, ok := pending[key]; ok {
				out.send(serveReply{ID: old.ID, Type: "superseded"})
			} else {
				order = append(order, key)
			}
			pending[key] = req
			mu.Unlock()
			signal()
		}
		if err := sc.Err(); err != nil {
			out.send(serveReply{Type: "error", Message: "read requests: " + err.Error()})
		}
		mu.Lock()
		closed = true
		mu.Unlock()
		signal()
	}()

	var scorer serveScorer
	defer func() {
		if scorer != nil {
			_ = scorer.Close()
		}
	}()
	var key []byte
	for {
		mu.Lock()
		var req serveRequest
		have := len(order) > 0
		if have {
			req = pending[order[0]]
			delete(pending, order[0])
			order = order[1:]
		}
		done := closed && !have
		mu.Unlock()
		if done {
			return nil
		}
		if !have {
			select {
			case <-wake:
				continue
			case <-ctx.Done():
				return nil
			}
		}
		if scorer == nil {
			s, err := openServeScorer(ctx, o, stderr)
			if err != nil {
				// Not cached: the next request tries again (after `summon`).
				out.send(serveReply{ID: req.ID, Type: "error", Message: err.Error()})
				continue
			}
			scorer = s
		}
		units, model, data, err := serveOne(ctx, scorer, o, req)
		if ctx.Err() != nil {
			return nil
		}
		if err != nil {
			out.send(serveReply{ID: req.ID, Type: "error", File: req.File, Message: err.Error()})
			continue
		}
		if units == nil {
			units = []report.Unit{}
		}
		out.send(serveResult{ID: req.ID, Type: "result", File: req.File, Units: units, Model: model})
		if o.SaveResults {
			if key == nil {
				if key, err = results.Key(hub.RootDir()); err != nil {
					fmt.Fprintf(stderr, "warning: results key: %v\n", err)
					continue
				}
			}
			entry := map[string]results.Entry{req.File: {SHA256: results.Hash(data), Units: units}}
			if err := results.Update(req.Root, key, model, entry, false); err != nil {
				fmt.Fprintf(stderr, "warning: could not save results to %s: %v\n", results.Dir, err)
			}
		}
	}
}

// serveOne validates a request and scores its file.
func serveOne(ctx context.Context, scorer serveScorer, o ScanOptions, req serveRequest) ([]report.Unit, report.Model, []byte, error) {
	var none report.Model
	if !filepath.IsAbs(req.Root) {
		return nil, none, nil, errors.New("root must be an absolute path")
	}
	if !results.CleanRel(req.File) {
		return nil, none, nil, fmt.Errorf("file must be a clean path relative to root (got %q)", req.File)
	}
	var data []byte
	if req.Content != nil {
		data = []byte(*req.Content)
	} else {
		r, err := os.OpenRoot(req.Root)
		if err != nil {
			return nil, none, nil, err
		}
		st, err := r.Stat(req.File)
		if err == nil && !st.Mode().IsRegular() {
			err = errors.New("not a regular file")
		}
		if err == nil && st.Size() > o.MaxFileBytes {
			err = fmt.Errorf("larger than %d bytes", o.MaxFileBytes)
		}
		if err == nil {
			data, err = r.ReadFile(req.File)
		}
		_ = r.Close()
		if err != nil {
			return nil, none, nil, fmt.Errorf("%s: %w", req.File, err)
		}
	}
	if int64(len(data)) > o.MaxFileBytes {
		return nil, none, nil, fmt.Errorf("%s is larger than %d bytes", req.File, o.MaxFileBytes)
	}
	rep, err := scorer.Score(ctx, req.Root, []walk.File{{Path: filepath.Join(req.Root, filepath.FromSlash(req.File)), Rel: req.File, Size: int64(len(data)), Data: data}})
	if err != nil {
		return nil, none, nil, err
	}
	report.ApplyConfidence(rep, o.MinConfidence)
	return rep.Units, rep.Model, data, nil
}
