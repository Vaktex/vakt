// Package walk finds the source files to scan under a root directory.
//
// It walks directories in parallel, honours nested .gitignore files,
// .git/info/exclude and .vaktignore files (same syntax as .gitignore,
// higher priority than a .gitignore in the same directory), and applies
// Include/Exclude doublestar globs. Dependency and build directories
// (DefaultSkipDirs), binary files, minified files and oversized files are
// skipped and reported on the skipped channel.
//
// Everything under the root is untrusted: names, ignore files, symlinks and
// contents. Every directory and file is opened through an os.Root bound to
// the resolved root, so even a path swapped for a symlink mid-walk cannot
// escape it (the kernel refuses the traversal). Symlinks are not followed
// unless Options.FollowSymlinks is set, and even then only within the root.
// File contents are read once, through the same handle that was checked, and
// delivered in File.Data: consumers must not reopen File.Path.
//
// Ignore files inside the scanned tree are chosen by whoever wrote the
// tree. Every path they hide is reported as a Skip (ReasonIgnored), and
// Options.NoRepoIgnores disables them entirely for untrusted scans.
package walk

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"

	"github.com/bmatcuk/doublestar/v4"
)

// DefaultMaxFileBytes is used when Options.MaxFileBytes is zero.
const DefaultMaxFileBytes int64 = 2 << 20

const (
	binarySniffBytes  = 8 << 10  // NUL in the first 8 KiB => binary
	minifySniffBytes  = 64 << 10 // average line length is measured on this prefix
	minifiedAvgLine   = 500      // average line length above this => minified
	minifiedMinSample = 1 << 10  // shorter files are never called minified
)

// Skip reasons. Reasons for I/O errors carry the error text after the prefix.
const (
	ReasonSkipDir    = "dependency or build directory"
	ReasonBinary     = "binary file"
	ReasonMinified   = "minified file"
	ReasonGenerated  = "generated file"
	ReasonTooLarge   = "file too large"
	ReasonSymlink    = "symlink not followed"
	ReasonEscape     = "symlink escapes root"
	ReasonLoop       = "symlink loop or directory already walked"
	ReasonBroken     = "broken symlink"
	ReasonNotRegular = "not a regular file"
	ReasonError      = "unreadable" // "unreadable: <error>"
	ReasonIgnored    = "ignored by the repository's ignore files"
	ReasonTooDeep    = "directory nesting too deep"
	ReasonIgnoreCap  = "ignore file truncated (too many patterns)"
	// ReasonIgnoreOff is reported once (on ".") when the tree's ignore files
	// are too large or costly to honour: they are then disregarded for the
	// rest of the walk, so more files are scanned, never fewer.
	ReasonIgnoreOff = "repository ignore files disregarded (too large or too costly to evaluate)"
)

// maxDepth bounds directory nesting (a directory swapped for an in-root
// symlink mid-walk could otherwise recurse until the path is too long).
const maxDepth = 128

// DefaultSkipDirs are directory base names that are never descended into,
// at any depth: VCS metadata, dependency trees, virtualenvs, build outputs
// and tool caches. They contain third-party or generated code that would
// swamp a scan without being the project's own code.
var DefaultSkipDirs = []string{
	// VCS metadata and git worktree checkouts
	".git", ".hg", ".svn", ".bzr", "_darcs", ".worktrees",
	// JavaScript / TypeScript
	"node_modules", "bower_components", "jspm_packages", ".pnpm-store", ".yarn",
	".next", ".nuxt", ".svelte-kit", ".angular", ".parcel-cache", ".turbo",
	".docusaurus", ".nyc_output", ".serverless",
	// Python
	".venv", "venv", "__pycache__", ".tox", ".nox", ".eggs", ".mypy_cache",
	".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "site-packages",
	// Go / PHP / Ruby
	"vendor", ".bundle",
	// JVM
	".gradle", ".m2",
	// Apple
	"Pods", "Carthage", "DerivedData",
	// Rust / Zig / Dart / Haskell / Elm / Elixir
	"target", "zig-cache", ".zig-cache", "zig-out", ".dart_tool", ".stack-work",
	"elm-stuff", "_build",
	// Build outputs and caches
	"dist", "build", ".cache", ".terraform",
}

var skipDirSet = func() map[string]bool {
	m := make(map[string]bool, len(DefaultSkipDirs))
	for _, d := range DefaultSkipDirs {
		m[d] = true
	}
	return m
}()

// Options configures Walk.
type Options struct {
	Include, Exclude []string // doublestar globs relative to root
	MaxFileBytes     int64    // default 2 MiB; larger files skipped
	FollowSymlinks   bool     // default false; never escapes root even if true
	// NoRepoIgnores disables .gitignore, .vaktignore and .git/info/exclude
	// from the scanned tree; only Include/Exclude apply. Use it when the
	// tree is untrusted and must not decide what gets scanned.
	NoRepoIgnores bool
	Jobs          int
}

// File is a file to scan.
type File struct {
	Path string // absolute path, for display only: do NOT reopen it (use Data)
	Rel  string // path relative to the root, forward slashes
	Size int64
	Data []byte // full contents, read through the checked handle
}

// Skip records a path that was deliberately not scanned.
type Skip struct{ Rel, Reason string }

type walker struct {
	ctx     context.Context
	root    string   // resolved absolute root
	fs      *os.Root // all access goes through this handle
	opts    Options
	out     chan<- File
	skipped chan<- Skip
	sem     chan struct{}
	wg      sync.WaitGroup
	mu      sync.Mutex
	visited map[string]bool // real dirs already walked (FollowSymlinks only)
	budget  atomic.Int64    // ignore patterns still allowed in this walk
	meter   matchMeter      // ignore matching work still allowed in this walk
	dry     bool            // budget probe: evaluate ignores only, no reads or output
	ignMu   sync.Mutex
	ignores map[string]*ignoreFile // parsed by the dry pass, reused by the real walk
}

// Walk walks root and sends every scannable file on out and every skipped
// path on skipped. It never closes either channel. It returns ctx.Err() if
// the context is cancelled, an error if root cannot be used or a glob is
// malformed, and nil otherwise; per-file problems become Skips.
func Walk(ctx context.Context, root string, opts Options, out chan<- File, skipped chan<- Skip) error {
	if opts.MaxFileBytes <= 0 {
		opts.MaxFileBytes = DefaultMaxFileBytes
	}
	if opts.Jobs <= 0 {
		opts.Jobs = min(max(runtime.GOMAXPROCS(0)*2, 4), 64)
	}
	for _, g := range slices.Concat(opts.Include, opts.Exclude) {
		if !doublestar.ValidatePattern(g) {
			return fmt.Errorf("walk: invalid glob %q", g)
		}
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return fmt.Errorf("walk: %w", err)
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return fmt.Errorf("walk: %w", err)
	}
	st, err := os.Stat(real)
	if err != nil {
		return fmt.Errorf("walk: %w", err)
	}
	if !st.IsDir() {
		return fmt.Errorf("walk: %s is not a directory", root)
	}
	fsRoot, err := os.OpenRoot(real)
	if err != nil {
		return fmt.Errorf("walk: %w", err)
	}
	defer fsRoot.Close()
	var dryIgnores map[string]*ignoreFile
	if !opts.NoRepoIgnores {
		// The tree's ignore files decide what is hidden, under a work
		// budget. A dry pass (no reads, nothing emitted) finds out whether
		// they stay within it; if not, the real walk disregards them all,
		// so the result never depends on where or when the budget ran out.
		dry := newWalker(ctx, real, fsRoot, opts, nil, nil)
		dry.dry = true
		dry.ignores = map[string]*ignoreFile{}
		dry.run()
		dryIgnores = dry.ignores
		if dry.meter.out.Load() {
			opts.NoRepoIgnores = true
			if skipped != nil {
				select {
				case skipped <- Skip{Rel: ".", Reason: ReasonIgnoreOff}:
				case <-ctx.Done():
				}
			}
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
	w := newWalker(ctx, real, fsRoot, opts, out, skipped)
	if !opts.NoRepoIgnores {
		w.ignores = dryIgnores
	}
	w.run()
	return ctx.Err()
}

func newWalker(ctx context.Context, real string, fsRoot *os.Root, opts Options, out chan<- File, skipped chan<- Skip) *walker {
	w := &walker{
		ctx:     ctx,
		root:    real,
		fs:      fsRoot,
		opts:    opts,
		out:     out,
		skipped: skipped,
		sem:     make(chan struct{}, opts.Jobs),
		visited: map[string]bool{real: true},
	}
	w.budget.Store(maxPatternsPerWalk)
	w.meter.left.Store(maxMatchSteps)
	return w
}

func (w *walker) run() {
	var chain *ignoreChain
	if !w.opts.NoRepoIgnores {
		// Only a real .git directory: a .git symlink could point at another
		// tree's exclude file.
		if st, err := w.fs.Lstat(".git"); err == nil && st.IsDir() {
			chain = chain.push(w.readIgnore(".git/info/exclude", ""))
		}
	}
	w.wg.Add(1)
	w.walkDir(".", "", chain, 0)
	w.wg.Wait()
}

// walkDir processes one directory. dir is the real path to read, rel its
// slash path relative to the root ("" for the root). It hands
// subdirectories to new goroutines while semaphore slots are free and walks
// them inline otherwise, so the number of goroutines stays bounded.
func (w *walker) walkDir(dir, rel string, chain *ignoreChain, depth int) {
	defer w.wg.Done()
	if w.ctx.Err() != nil {
		return
	}
	if depth > maxDepth {
		w.skip(rel, ReasonTooDeep)
		return
	}
	d, err := w.fs.Open(dir)
	if err != nil {
		w.skip(rel, errReason(err))
		return
	}
	entries, err := d.ReadDir(-1)
	_ = d.Close()
	if err != nil && len(entries) == 0 {
		w.skip(rel, errReason(err))
		return
	}
	if !w.opts.NoRepoIgnores {
		chain = chain.push(w.readIgnore(joinRel(dir, ".gitignore"), rel))
		chain = chain.push(w.readIgnore(joinRel(dir, ".vaktignore"), rel))
	}
	for _, e := range entries {
		if w.ctx.Err() != nil {
			return
		}
		name := e.Name()
		childRel := name
		if rel != "" {
			childRel = rel + "/" + name
		}
		path := joinRel(dir, name)
		typ := e.Type()
		if typ&fs.ModeSymlink != 0 {
			w.symlink(path, childRel, chain, depth)
			continue
		}
		if typ.IsDir() {
			w.dir(path, childRel, name, chain, depth)
			continue
		}
		w.file(path, childRel, typ, chain)
	}
}

func (w *walker) dir(path, rel, name string, chain *ignoreChain, depth int) {
	if skipDirSet[name] {
		w.skip(rel, ReasonSkipDir)
		return
	}
	if w.excluded(rel) {
		return
	}
	if w.ignored(chain, rel, true) {
		w.skip(rel, ReasonIgnored)
		return
	}
	if w.opts.FollowSymlinks && !w.markVisited(filepath.Join(w.root, filepath.FromSlash(path))) {
		return // already reached through a followed symlink
	}
	w.wg.Add(1)
	select {
	case w.sem <- struct{}{}:
		go func() {
			defer func() { <-w.sem }()
			w.walkDir(path, rel, chain, depth+1)
		}()
	default:
		w.walkDir(path, rel, chain, depth+1)
	}
}

// joinRel joins root-relative slash paths ("." is the root).
func joinRel(dir, name string) string {
	if dir == "." || dir == "" {
		return name
	}
	return dir + "/" + name
}

func (w *walker) symlink(path, rel string, chain *ignoreChain, depth int) {
	// Ignore rules see the link itself; decide file vs dir from the target.
	// Stat only inspects the target's metadata, it never reads it. Through
	// the Root, a target outside the root is an error here.
	abs := filepath.Join(w.root, filepath.FromSlash(path))
	st, terr := os.Stat(abs) // metadata only; containment is decided below
	isDir := terr == nil && st.IsDir()
	if w.excluded(rel) {
		return
	}
	if w.ignored(chain, rel, isDir) {
		w.skip(rel, ReasonIgnored)
		return
	}
	if isDir && skipDirSet[filepath.Base(rel)] {
		w.skip(rel, ReasonSkipDir)
		return
	}
	if !w.opts.FollowSymlinks {
		w.skip(rel, ReasonSymlink)
		return
	}
	if terr != nil {
		w.skip(rel, ReasonBroken)
		return
	}
	target, err := filepath.EvalSymlinks(abs)
	if err != nil {
		w.skip(rel, ReasonBroken)
		return
	}
	if !within(w.root, target) {
		w.skip(rel, ReasonEscape)
		return
	}
	// Walk the target by its root-relative path; the Root re-checks
	// containment at open time, so a swap after EvalSymlinks cannot escape.
	tr, err := filepath.Rel(w.root, target)
	if err != nil {
		w.skip(rel, ReasonEscape)
		return
	}
	tr = filepath.ToSlash(tr)
	if !isDir {
		if !st.Mode().IsRegular() {
			w.skip(rel, ReasonNotRegular)
			return
		}
		if !w.included(rel) {
			return
		}
		w.check(tr, rel)
		return
	}
	if !w.markVisited(target) {
		w.skip(rel, ReasonLoop)
		return
	}
	w.wg.Add(1)
	w.walkDir(tr, rel, chain, depth+1)
}

// markVisited records a real directory path and reports whether it was new.
func (w *walker) markVisited(real string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.visited[real] {
		return false
	}
	w.visited[real] = true
	return true
}

func (w *walker) file(path, rel string, typ fs.FileMode, chain *ignoreChain) {
	if w.excluded(rel) || !w.included(rel) {
		return
	}
	if w.ignored(chain, rel, false) {
		w.skip(rel, ReasonIgnored)
		return
	}
	if !typ.IsRegular() {
		w.skip(rel, ReasonNotRegular)
		return
	}
	lower := strings.ToLower(rel)
	if strings.HasSuffix(lower, ".min.js") || strings.HasSuffix(lower, ".min.css") ||
		strings.HasSuffix(lower, ".min.mjs") || strings.HasSuffix(lower, "-min.js") {
		w.skip(rel, ReasonMinified)
		return
	}
	if IsGeneratedName(lower) {
		w.skip(rel, ReasonGenerated)
		return
	}
	w.check(path, rel)
}

// check opens a regular file through the Root, applies the size, binary
// and minified tests and emits it with its contents. path is relative to
// the root.
func (w *walker) check(path, rel string) {
	if w.dry {
		return
	}
	// O_NONBLOCK keeps a FIFO swapped in after the type check from blocking;
	// the Root refuses any path that resolves outside the root.
	f, err := w.fs.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		w.skip(rel, errReason(err))
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		w.skip(rel, errReason(err))
		return
	}
	if !st.Mode().IsRegular() {
		w.skip(rel, ReasonNotRegular)
		return
	}
	if st.Size() > w.opts.MaxFileBytes {
		w.skip(rel, ReasonTooLarge)
		return
	}
	// Read everything now, through this checked handle (the file may have
	// grown since fstat, so the read is capped too).
	buf := make([]byte, st.Size())
	n, err := io.ReadFull(f, buf)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		w.skip(rel, errReason(err))
		return
	}
	data := buf[:n]
	head := data[:min(len(data), minifySniffBytes)]
	if IsBinary(head) {
		w.skip(rel, ReasonBinary)
		return
	}
	if IsMinified(head) {
		w.skip(rel, ReasonMinified)
		return
	}
	if IsGenerated(head) {
		w.skip(rel, ReasonGenerated)
		return
	}
	w.emit(File{Path: filepath.Join(w.root, filepath.FromSlash(path)), Rel: rel, Size: int64(n), Data: data})
}

// IsBinary reports whether data has a NUL byte in its first 8 KiB.
func IsBinary(data []byte) bool {
	return bytes.IndexByte(data[:min(len(data), binarySniffBytes)], 0) >= 0
}

// generatedSniffBytes is how much of a file's head is searched for a
// generator marker. Markers sit in the first comment block by convention.
const generatedSniffBytes = 2 << 10

// generatedMarkers are the conventional headers of machine-written code:
// Go's `// Code generated ... DO NOT EDIT.`, protoc, Thrift, Bazel,
// JetBrains and most codegen tools. Matched case-insensitively.
// Bare "do not edit" is deliberately absent: hand-written files use it for
// one section ("do not edit below this line").
var generatedMarkers = [][]byte{
	[]byte("code generated by"), []byte("@generated"), []byte("<auto-generated"),
	[]byte("this file is auto-generated"), []byte("this file was auto-generated"),
	[]byte("this file was automatically generated"), []byte("this file is automatically generated"),
	[]byte("generated by the protocol buffer compiler"), []byte("autogenerated by thrift"),
	[]byte("do not edit! generated"), []byte("generated file. do not edit"),
	[]byte("generated code - do not edit"), []byte("generated code. do not edit"),
}

// generatedSuffixes are file names that are generated by construction.
var generatedSuffixes = []string{
	".pb.go", ".pb.cc", ".pb.h", "_pb2.py", "_pb2_grpc.py", ".pb.swift", "_grpc.pb.go",
	".g.dart", ".freezed.dart", ".designer.cs", ".g.cs", ".generated.ts", ".generated.js",
	"_generated.go", "_generated.rs", ".gen.go", "zz_generated.deepcopy.go",
}

// IsGeneratedName reports whether a lowercase path names a generated file.
func IsGeneratedName(lower string) bool {
	for _, s := range generatedSuffixes {
		if strings.HasSuffix(lower, s) {
			return true
		}
	}
	return false
}

// IsGenerated reports whether the file's head carries a generator marker.
//
// Generated code is not what a reviewer can fix, and scoring it costs scan
// time and pushes machine-written boilerplate into the ranked list handed
// to deeper review. The training data excludes it for the same reason, so
// these files are also outside what the model has learned to rank.
func IsGenerated(head []byte) bool {
	h := bytes.ToLower(head[:min(len(head), generatedSniffBytes)])
	for _, m := range generatedMarkers {
		if bytes.Contains(h, m) {
			return true
		}
	}
	return false
}

// IsMinified reports whether the sample's average line length exceeds 500
// bytes. Samples under 1 KiB are never considered minified.
func IsMinified(sample []byte) bool {
	if len(sample) < minifiedMinSample {
		return false
	}
	lines := bytes.Count(sample, []byte{'\n'}) + 1
	return len(sample)/lines > minifiedAvgLine
}

func (w *walker) excluded(rel string) bool {
	for _, g := range w.opts.Exclude {
		if doublestar.MatchUnvalidated(g, rel) {
			return true
		}
	}
	return false
}

func (w *walker) included(rel string) bool {
	if len(w.opts.Include) == 0 {
		return true
	}
	for _, g := range w.opts.Include {
		if doublestar.MatchUnvalidated(g, rel) {
			return true
		}
	}
	return false
}

func (w *walker) emit(f File) {
	select {
	case w.out <- f:
	case <-w.ctx.Done():
	}
}

func (w *walker) skip(rel, reason string) {
	if w.dry {
		return
	}
	if rel == "" {
		rel = "."
	}
	if w.skipped == nil {
		return
	}
	select {
	case w.skipped <- Skip{Rel: rel, Reason: reason}:
	case <-w.ctx.Done():
	}
}

func errReason(err error) string {
	var pe *fs.PathError
	if errors.As(err, &pe) {
		err = pe.Err // drop the absolute path; Rel already identifies the file
	}
	return ReasonError + ": " + err.Error()
}

// within reports whether path is root or lies beneath it. Both must be
// cleaned absolute paths.
func within(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)))
}

// readIgnore reads an ignore file (path relative to the root), returning nil
// if it is missing, unreadable, not a regular file, or a symlink. It opens
// through the Root with O_NONBLOCK and checks the opened handle, so neither
// a symlink nor a FIFO swapped in after the Lstat can escape or block.
func (w *walker) readIgnore(path, base string) *ignoreFile {
	if w.ignores != nil && !w.dry {
		// The real walk uses exactly the ignore files the dry pass read and
		// metered, so a file rewritten between the passes cannot swap in
		// costlier rules. Files the dry pass never saw are disregarded.
		w.ignMu.Lock()
		ig := w.ignores[path]
		w.ignMu.Unlock()
		if ig != nil && ig.truncated {
			w.skip(path, ReasonIgnoreCap) // the dry pass cannot report
		}
		return ig
	}
	ig := w.readIgnoreFile(path, base)
	if w.dry {
		w.ignMu.Lock()
		w.ignores[path] = ig
		w.ignMu.Unlock()
	}
	return ig
}

func (w *walker) readIgnoreFile(path, base string) *ignoreFile {
	st, err := w.fs.Lstat(path)
	if err != nil || !st.Mode().IsRegular() {
		return nil
	}
	f, err := w.fs.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil
	}
	defer f.Close()
	if st, err := f.Stat(); err != nil || !st.Mode().IsRegular() {
		return nil
	}
	data, err := io.ReadAll(io.LimitReader(f, maxIgnoreFileBytes+1)) // +1: detect truncation
	if err != nil {
		return nil
	}
	ig, truncated := parseIgnore(base, data, &w.budget)
	if truncated {
		if ig != nil {
			ig.truncated = true
		}
		w.skip(path, ReasonIgnoreCap)
	}
	if w.budget.Load() < 0 {
		// The walk-wide pattern budget ran out. Which files got patterns
		// would depend on goroutine timing, so stop honouring all of them.
		w.meter.out.Store(true)
	}
	return ig
}

// ignored applies the repository's ignore rules under the walk's work
// budget, reporting once when they are switched off.
func (w *walker) ignored(chain *ignoreChain, rel string, isDir bool) bool {
	if w.dry {
		// Probe the budget; the meter is monotone, so the total charged is
		// the same in any order and the verdict is deterministic.
		return chain.ignoredMetered(rel, isDir, &w.meter)
	}
	// The dry pass proved these rules fit the budget (the real walk visits
	// the same paths with the same rules), so no metering here.
	return chain.ignored(rel, isDir)
}
