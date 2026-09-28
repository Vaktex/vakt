package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/spf13/cobra"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/hub"
	"github.com/vaktex/vakt/internal/report"
)

// Defaults for patrol flags.
const (
	defaultThreshold   = 0.5
	defaultTop         = 25
	defaultMinTokens   = 16
	defaultBatchTokens = 4096
	// auto resolves to fp16 on Metal and tf32 on CUDA (score drift ~1e-3
	// against the fp32 reference) and exact fp32 on CPU; --precision fp32
	// keeps parity grade.
	defaultPrecision    = "auto"
	defaultMaxFileBytes = 2 << 20
	defaultOut          = brand.Binary + "-report.json"
	// Pinned to the verified launch weights (engine parity 2.6e-6 against
	// the Python model at fp32), so a push to the repo never silently
	// changes users' scores. Bump deliberately, after re-running parity.
	defaultModel = "hf:" + brand.ModelRepo + "@" + brand.ModelCommit
)

// ScanOptions is everything patrol was asked to do. The integrator's runScan
// maps it onto pipeline.Config, walk.Options and engine.Options.
type ScanOptions struct {
	Root string // absolute path of the directory to scan

	// Output.
	Format  string  // "pretty", "json" or "both"
	Out     string  // JSON destination for json/both; "-" is stdout
	Top     int     // rows in the pretty top findings table
	Quiet   bool    // summary line only, no progress
	Verbose bool    // model/backend stats and per-finding scores
	NoColor bool    // plain output even on a TTY
	FailOn  float64 // exit 1 if any unit scores >= FailOn (only when FailOnSet)
	// FailOnSet records whether --fail-on was given.
	FailOnSet bool

	// Scoring.
	Threshold   float64 // flag units with severity >= Threshold
	MinTokens   int     // skip units whose rendered prompt is shorter
	BatchTokens int     // padded tokens per engine batch
	Jobs        int     // walk/parse/tokenize workers
	NoCache     bool

	// File selection (walk.Options).
	Include, Exclude []string
	MaxFileBytes     int64
	FollowSymlinks   bool
	NoRepoIgnores    bool
	TopLevel         bool // also score top-level code outside functions

	// Model. Exactly one of ModelPath or ModelRepo is set.
	Model         string // the raw --model value
	ModelPath     string // a local model.safetensors
	ModelRepo     string // Hugging Face repo id, e.g. vaktex/dom-oss-0.8b
	ModelRevision string // branch, tag or commit (default "main")
	Precision     string // "auto", "fp32", "tf32", "bf16", "fp16" or "int8"
	Device        string // "auto", "gpu" or "cpu"
	Devices       []int  // GPU indices; empty means the engine's default

	Demo bool // hidden: synthetic report from engine.Fake
}

// runScan runs a scan. The integrator replaces it with the real pipeline in
// a file built with the engine; until then it explains what is missing.
// prog is updated live and rendered on stderr by patrol.
var runScan = func(ctx context.Context, opts ScanOptions, prog *report.Progress) (*report.Report, error) {
	return nil, fmt.Errorf("the scanning pipeline isn't linked into this build of %s (backend: %s); "+
		"install a release build with a native engine (Metal or CUDA)", brand.Binary, brand.Backend)
}

func newPatrolCmd() *cobra.Command {
	var o ScanOptions
	var devices string
	cmd := &cobra.Command{
		Use:     "patrol [path]",
		Aliases: []string{"scan"},
		Short:   "Scan a codebase and report the functions most likely to be vulnerable",
		Long: "patrol walks a directory (default: the current one), splits source files into\n" +
			"functions, scores each with " + brand.ModelName + " and prints the top findings.\n\n" +
			"Exit status: 0 when there are no findings at or above the threshold,\n" +
			"1 when there are findings, 2 on error, 130 when interrupted.\n" +
			"With --fail-on, exit 1 only if any function scores at or above that value.",
		Example: "  " + brand.Binary + " .\n" +
			"  " + brand.Binary + " patrol src --threshold 0.7 --top 10\n" +
			"  " + brand.Binary + " patrol . --format json --out - | jq .summary\n" +
			"  " + brand.Binary + " patrol . --fail-on 0.9   # CI: exit 1 if severity >= 0.9",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			o.Root = "."
			if len(args) == 1 {
				o.Root = args[0]
			}
			f := cmd.Flags()
			o.FailOnSet = f.Changed("fail-on")
			if err := o.finish(devices, f.Changed("out"), f.Changed("revision")); err != nil {
				return &exitCodeError{code: exitError, err: err}
			}
			return patrol(cmd.Context(), cmd.OutOrStdout(), cmd.ErrOrStderr(), o)
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.Format, "format", "pretty", "output format: pretty, json or both")
	f.StringVarP(&o.Out, "out", "o", defaultOut, "JSON report path for --format json/both (- for stdout)")
	f.Float64Var(&o.Threshold, "threshold", defaultThreshold, "flag functions with severity >= this score")
	f.IntVar(&o.Top, "top", defaultTop, "number of findings to print")
	f.IntVar(&o.MinTokens, "min-tokens", defaultMinTokens, "skip functions shorter than this many tokens")
	f.IntVar(&o.BatchTokens, "batch-tokens", defaultBatchTokens, "padded tokens per model batch")
	f.IntVarP(&o.Jobs, "jobs", "j", runtime.NumCPU(), "parallel walk/parse/tokenize workers")
	f.StringArrayVar(&o.Include, "include", nil, "only scan paths matching this glob (repeatable)")
	f.StringArrayVar(&o.Exclude, "exclude", nil, "skip paths matching this glob (repeatable)")
	f.BoolVar(&o.NoCache, "no-cache", false, "don't read or write the score cache")
	f.StringVar(&o.Model, "model", defaultModel, "a local model.safetensors path, or hf:owner/name[@revision]")
	f.StringVar(&o.ModelRevision, "revision", "", "model revision (overrides @revision in --model)")
	f.StringVar(&o.Precision, "precision", defaultPrecision, "compute precision: auto (fp16 on Metal, tf32 on CUDA, fp32 on CPU), fp32 (exact), tf32, bf16, fp16, or int8 (int8 matmuls; M5-class Apple GPUs; larger score drift)")
	f.StringVar(&o.Device, "device", "auto", "device: auto, gpu or cpu")
	f.StringVar(&devices, "devices", "", "comma-separated GPU indices to use (e.g. 0,1)")
	f.Int64Var(&o.MaxFileBytes, "max-file-bytes", defaultMaxFileBytes, "skip files larger than this")
	f.BoolVar(&o.FollowSymlinks, "follow-symlinks", false, "follow symlinks (never outside the scan root)")
	f.BoolVar(&o.TopLevel, "top-level", false, "also score top-level code outside functions (imports, globals); the model was trained on functions, so these scores are less reliable")
	f.BoolVar(&o.NoRepoIgnores, "no-repo-ignores", false, "ignore the scanned tree's .gitignore/.vaktignore (for untrusted code: the tree cannot hide files)")
	f.BoolVarP(&o.Quiet, "quiet", "q", false, "print only the summary lines")
	f.BoolVarP(&o.Verbose, "verbose", "v", false, "show model/backend stats and per-finding scores")
	f.BoolVar(&o.NoColor, "no-color", false, "disable colour (also NO_COLOR)")
	f.Float64Var(&o.FailOn, "fail-on", 0, "exit with status 1 if any function scores >= this")
	f.BoolVar(&o.Demo, "demo", false, "render a synthetic report from the fake engine")
	_ = f.MarkHidden("demo")
	return cmd
}

// finish validates flags and fills in derived fields.
func (o *ScanOptions) finish(devices string, outSet, revisionSet bool) error {
	switch o.Format {
	case "pretty":
		if outSet {
			return errors.New("--out only applies to --format json or both")
		}
	case "json", "both":
		if o.Out == "" {
			return errors.New("--out must not be empty")
		}
		if o.Format == "both" && o.Out == "-" {
			return errors.New("--format both writes the pretty report to stdout; give --out a file path")
		}
	default:
		return fmt.Errorf("--format must be pretty, json or both (got %q)", o.Format)
	}
	switch {
	case !(o.Threshold > 0 && o.Threshold <= 1):
		return errors.New("--threshold must be in (0, 1]")
	case o.FailOnSet && !(o.FailOn >= 0 && o.FailOn <= 1):
		return errors.New("--fail-on must be in [0, 1]")
	case o.Top < 1:
		return errors.New("--top must be at least 1")
	case o.MinTokens < 0 || o.MinTokens > core.MaxTokens:
		return fmt.Errorf("--min-tokens must be in [0, %d]", core.MaxTokens)
	case o.BatchTokens < 1:
		return errors.New("--batch-tokens must be positive")
	case o.Jobs < 1:
		return errors.New("--jobs must be at least 1")
	case o.MaxFileBytes < 1:
		return errors.New("--max-file-bytes must be positive")
	}
	switch o.Precision {
	case "auto", "fp32", "tf32", "bf16", "fp16", "int8":
	default:
		return fmt.Errorf("--precision must be auto, fp32, tf32, bf16, fp16 or int8 (got %q)", o.Precision)
	}
	switch o.Device {
	case "auto", "gpu", "cpu":
	default:
		return fmt.Errorf("--device must be auto, gpu or cpu (got %q)", o.Device)
	}
	if devices != "" {
		if o.Device == "cpu" {
			return errors.New("--devices selects GPUs and cannot be used with --device cpu")
		}
		seen := map[int]bool{}
		for _, s := range strings.Split(devices, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil || n < 0 || n > 1023 {
				return fmt.Errorf("--devices: %q is not a GPU index", s)
			}
			if !seen[n] {
				seen[n] = true
				o.Devices = append(o.Devices, n)
			}
		}
	}
	for _, g := range append(append([]string{}, o.Include...), o.Exclude...) {
		if g == "" {
			return errors.New("--include/--exclude patterns must not be empty")
		}
	}
	rev := ""
	if revisionSet {
		rev = o.ModelRevision
	}
	spec, err := parseModelSpec(o.Model, rev)
	if err != nil {
		return err
	}
	o.ModelPath, o.ModelRepo, o.ModelRevision = spec.Path, spec.Repo, spec.Revision

	if !o.Demo {
		abs, err := filepath.Abs(o.Root)
		if err != nil {
			return err
		}
		st, err := os.Stat(abs)
		if err != nil {
			return fmt.Errorf("cannot scan %s: %w", o.Root, errors.Unwrap(err))
		}
		if !st.IsDir() {
			return fmt.Errorf("cannot scan %s: not a directory", o.Root)
		}
		o.Root = abs
	}
	return nil
}

// modelSpec is a parsed --model value.
type modelSpec struct {
	Path           string // local file
	Repo, Revision string // Hugging Face
}

// parseModelSpec parses "hf:owner/name[@rev]" or a local path. A non-empty
// revision overrides the one in the spec.
func parseModelSpec(s, revision string) (modelSpec, error) {
	if s == "" {
		s = defaultModel
	}
	if rest, ok := strings.CutPrefix(s, "hf:"); ok {
		repo, rev, _ := strings.Cut(rest, "@")
		if revision != "" {
			rev = revision
		}
		if rev == "" {
			rev = "main"
		}
		if err := hub.ValidateRepo(repo); err != nil {
			return modelSpec{}, fmt.Errorf("--model: %w", err)
		}
		if err := hub.ValidateRevision(rev); err != nil {
			return modelSpec{}, fmt.Errorf("--revision: %w", err)
		}
		return modelSpec{Repo: repo, Revision: rev}, nil
	}
	if revision != "" {
		return modelSpec{}, errors.New("--revision only applies to hf: models")
	}
	st, err := os.Stat(s)
	if err != nil {
		return modelSpec{}, fmt.Errorf("--model: %s: %w", s, errors.Unwrap(err))
	}
	if !st.Mode().IsRegular() {
		return modelSpec{}, fmt.Errorf("--model: %s is not a file", s)
	}
	abs, err := filepath.Abs(s)
	if err != nil {
		return modelSpec{}, err
	}
	return modelSpec{Path: abs}, nil
}

// resolveModel returns the local weights path and its sha256 ("" for a local
// path: the engine hashes it). For hf: models it downloads into the cache,
// drawing a progress bar on stderr. The integrator calls this from runScan.
func resolveModel(ctx context.Context, o ScanOptions, stderr io.Writer) (path, sha string, err error) {
	if o.ModelPath != "" {
		return o.ModelPath, "", nil
	}
	type got struct{ path, sha string }
	res, err := withSignIn(ctx, o.ModelRepo, os.Stdin, stderr, func() (got, error) {
		bar := newDownloadBar(stderr, o.Quiet)
		defer bar.done()
		p, s, err := hub.Resolve(ctx, hub.Options{Repo: o.ModelRepo, Revision: o.ModelRevision, Progress: bar.update})
		return got{p, s}, err
	})
	return res.path, res.sha, err
}

func patrol(ctx context.Context, stdout, stderr io.Writer, o ScanOptions) error {
	var (
		rep *report.Report
		err error
	)
	if o.Demo {
		rep, err = demoReport(ctx, o)
	} else {
		prog := &report.Progress{Started: time.Now()}
		pctx, stop := context.WithCancel(ctx)
		var wg sync.WaitGroup
		if !o.Quiet {
			wg.Add(1)
			go func() { defer wg.Done(); report.RenderProgress(pctx, stderr, prog) }()
		}
		rep, err = runScan(ctx, o, prog)
		stop()
		wg.Wait()
	}
	if err != nil {
		return &exitCodeError{code: exitError, err: err}
	}
	return emit(stdout, stderr, rep, o)
}

// emit writes the report in the requested formats and sets the exit code.
func emit(stdout, stderr io.Writer, rep *report.Report, o ScanOptions) error {
	if o.Format == "json" || o.Format == "both" {
		if o.Out == "-" {
			if err := report.WriteJSON(stdout, rep); err != nil {
				return err
			}
		} else {
			if err := report.WriteFile(o.Out, rep); err != nil {
				return err
			}
			if !o.Quiet {
				fmt.Fprintf(stderr, "report written to %s\n", o.Out)
			}
		}
	}
	if o.Format == "pretty" || o.Format == "both" {
		po := report.PrettyOptions{
			Top: o.Top, Threshold: o.Threshold, Quiet: o.Quiet, Verbose: o.Verbose,
			Color: useColor(stdout, o.NoColor), Hyperlinks: useHyperlinks(stdout),
			Width: termWidth(stdout),
		}
		if err := report.Pretty(stdout, rep, po); err != nil {
			return err
		}
	}
	failAt := o.Threshold
	if o.FailOnSet {
		failAt = o.FailOn
	}
	n := 0
	var maxSev float64
	for _, u := range rep.Units {
		if u.Severity > maxSev {
			maxSev = u.Severity
		}
		if u.Severity >= failAt {
			n++
		}
	}
	if n > 0 {
		msg := fmt.Sprintf("%d finding%s at or above %.2f (max %.2f)", n, plural(n), failAt, maxSev)
		if o.FailOnSet {
			return &exitCodeError{code: exitFindings, err: fmt.Errorf("--fail-on %.2f: %s", o.FailOn, msg)}
		}
		// Findings at the scan threshold: exit 1 with no extra stderr noise;
		// the pretty report already spelled them out.
		return &exitCodeError{code: exitFindings}
	}
	return nil
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
