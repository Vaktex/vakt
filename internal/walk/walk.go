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
// contents. Symlinks are not followed unless Options.FollowSymlinks is set,
// and even then nothing outside the resolved root is ever read.
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
	ReasonTooLarge   = "file too large"
	ReasonSymlink    = "symlink not followed"
	ReasonEscape     = "symlink escapes root"
	ReasonLoop       = "symlink loop or directory already walked"
	ReasonBroken     = "broken symlink"
	ReasonNotRegular = "not a regular file"
	ReasonError      = "unreadable" // "unreadable: <error>"
)

// DefaultSkipDirs are directory base names that are never descended into,
// at any depth: VCS metadata, dependency trees, virtualenvs, build outputs
// and tool caches. They contain third-party or generated code that would
// swamp a scan without being the project's own code.
var DefaultSkipDirs = []string{
	// VCS and harness metadata
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
	Jobs             int
}

// File is a file to scan.
type File struct {
	Path string // absolute path to read (the resolved target for followed symlinks)
	Rel  string // path relative to the root, forward slashes
	Size int64
}

// Skip records a path that was deliberately not scanned.
type Skip struct{ Rel, Reason string }

type walker struct {
	ctx     context.Context
	root    string // resolved absolute root
	opts    Options
	out     chan<- File
	skipped chan<- Skip
	sem     chan struct{}
	wg      sync.WaitGroup
	mu      sync.Mutex
	visited map[string]bool // real dirs already walked (FollowSymlinks only)
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
	w := &walker{
		ctx:     ctx,
		root:    real,
		opts:    opts,
		out:     out,
		skipped: skipped,
		sem:     make(chan struct{}, opts.Jobs),
		visited: map[string]bool{real: true},
	}
	var chain *ignoreChain
	chain = chain.push(readIgnore(filepath.Join(real, ".git", "info", "exclude"), ""))
	w.wg.Add(1)
	w.walkDir(real, "", chain)
	w.wg.Wait()
	return ctx.Err()
}

// walkDir processes one directory. dir is the real path to read, rel its
// slash path relative to the root ("" for the root). It hands
// subdirectories to new goroutines while semaphore slots are free and walks
// them inline otherwise, so the number of goroutines stays bounded.
func (w *walker) walkDir(dir, rel string, chain *ignoreChain) {
	defer w.wg.Done()
	if w.ctx.Err() != nil {
		return
	}
	entries, err := os.ReadDir(dir)
	if err != nil && len(entries) == 0 {
		w.skip(rel, errReason(err))
		return
	}
	chain = chain.push(readIgnore(filepath.Join(dir, ".gitignore"), rel))
	chain = chain.push(readIgnore(filepath.Join(dir, ".vaktignore"), rel))
	for _, e := range entries {
		if w.ctx.Err() != nil {
			return
		}
		name := e.Name()
		childRel := name
		if rel != "" {
			childRel = rel + "/" + name
		}
		path := filepath.Join(dir, name)
		typ := e.Type()
		if typ&fs.ModeSymlink != 0 {
			w.symlink(path, childRel, chain)
			continue
		}
		if typ.IsDir() {
			w.dir(path, childRel, name, chain)
			continue
		}
		w.file(path, childRel, typ, chain)
	}
}

func (w *walker) dir(path, rel, name string, chain *ignoreChain) {
	if skipDirSet[name] {
		w.skip(rel, ReasonSkipDir)
		return
	}
	if chain.ignored(rel, true) || w.excluded(rel) {
		return
	}
	if w.opts.FollowSymlinks && !w.markVisited(path) {
		return // already reached through a followed symlink
	}
	w.wg.Add(1)
	select {
	case w.sem <- struct{}{}:
		go func() {
			defer func() { <-w.sem }()
			w.walkDir(path, rel, chain)
		}()
	default:
		w.walkDir(path, rel, chain)
	}
}

func (w *walker) symlink(path, rel string, chain *ignoreChain) {
	// Ignore rules see the link itself; decide file vs dir from the target.
	// Stat only inspects the target's metadata, it never reads it.
	st, terr := os.Stat(path)
	isDir := terr == nil && st.IsDir()
	if chain.ignored(rel, isDir) || w.excluded(rel) {
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
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		w.skip(rel, ReasonBroken)
		return
	}
	if !within(w.root, target) {
		w.skip(rel, ReasonEscape)
		return
	}
	if !isDir {
		if !st.Mode().IsRegular() {
			w.skip(rel, ReasonNotRegular)
			return
		}
		if !w.included(rel) {
			return
		}
		w.check(target, rel)
		return
	}
	if !w.markVisited(target) {
		w.skip(rel, ReasonLoop)
		return
	}
	w.wg.Add(1)
	w.walkDir(target, rel, chain)
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
	if chain.ignored(rel, false) || w.excluded(rel) || !w.included(rel) {
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
	w.check(path, rel)
}

// check opens a regular file, applies the size, binary and minified tests
// and emits it.
func (w *walker) check(path, rel string) {
	// O_NONBLOCK keeps a FIFO swapped in after the type check from blocking.
	f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0) // #nosec G304 -- path comes from walking the root
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
	buf := make([]byte, min(st.Size(), minifySniffBytes))
	n, err := io.ReadFull(f, buf)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		w.skip(rel, errReason(err))
		return
	}
	head := buf[:n]
	if IsBinary(head) {
		w.skip(rel, ReasonBinary)
		return
	}
	if IsMinified(head) {
		w.skip(rel, ReasonMinified)
		return
	}
	w.emit(File{Path: path, Rel: rel, Size: st.Size()})
}

// IsBinary reports whether data has a NUL byte in its first 8 KiB.
func IsBinary(data []byte) bool {
	return bytes.IndexByte(data[:min(len(data), binarySniffBytes)], 0) >= 0
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

// readIgnore reads an ignore file, returning nil if it is missing,
// unreadable, not a regular file, or a symlink (a symlinked ignore file
// could point outside the root).
func readIgnore(path, base string) *ignoreFile {
	st, err := os.Lstat(path)
	if err != nil || !st.Mode().IsRegular() {
		return nil
	}
	f, err := os.Open(path) // #nosec G304 -- ignore file inside the walked root
	if err != nil {
		return nil
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxIgnoreFileBytes))
	if err != nil {
		return nil
	}
	return parseIgnore(base, data)
}
