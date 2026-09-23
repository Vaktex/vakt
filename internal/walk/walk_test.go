package walk

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

func writeFile(t testing.TB, root, rel string, data []byte) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

type result struct {
	files map[string]File
	skips map[string]string
	err   error
}

func collect(t testing.TB, ctx context.Context, root string, opts Options) result {
	t.Helper()
	out := make(chan File)
	skipped := make(chan Skip)
	r := result{files: map[string]File{}, skips: map[string]string{}}
	var wg sync.WaitGroup
	wg.Go(func() {
		for f := range out {
			if _, dup := r.files[f.Rel]; dup {
				t.Errorf("duplicate file %s", f.Rel)
			}
			r.files[f.Rel] = f
		}
	})
	wg.Go(func() {
		for s := range skipped {
			r.skips[s.Rel] = s.Reason
		}
	})
	r.err = Walk(ctx, root, opts, out, skipped)
	close(out)
	close(skipped)
	wg.Wait()
	return r
}

func keys[M ~map[string]V, V any](m M) []string {
	var k []string
	for s := range m {
		k = append(k, s)
	}
	slices.Sort(k)
	return k
}

func TestWalkBasics(t *testing.T) {
	root := t.TempDir()
	code := []byte("package main\n\nfunc main() {}\n")
	writeFile(t, root, "main.go", code)
	writeFile(t, root, "pkg/a.py", []byte("print(1)\n"))
	writeFile(t, root, "pkg/deep/b.js", []byte("x()\n"))
	writeFile(t, root, "node_modules/lib/index.js", code)
	writeFile(t, root, "sub/vendor/x.go", code)
	writeFile(t, root, ".git/config", code)
	writeFile(t, root, "dist/out.js", code)
	writeFile(t, root, "__pycache__/x.pyc", code)
	writeFile(t, root, "bin.dat", []byte("abc\x00def"))
	writeFile(t, root, "app.min.js", []byte("x()"))
	writeFile(t, root, "packed.js", []byte(strings.Repeat("a", 5000)))
	writeFile(t, root, "big.txt", make([]byte, 100))
	writeFile(t, root, "ignored.log", code)
	writeFile(t, root, "pkg/keep.log", code)
	writeFile(t, root, "pkg/gen/x.go", code)
	writeFile(t, root, ".gitignore", []byte("*.log\n"))
	writeFile(t, root, "pkg/.gitignore", []byte("!keep.log\n/gen/\n"))
	writeFile(t, root, ".vaktignore", []byte("secret.py\n"))
	writeFile(t, root, "secret.py", code)
	writeFile(t, root, ".git/info/exclude", []byte("local.go\n"))
	writeFile(t, root, "local.go", code)

	r := collect(t, context.Background(), root, Options{MaxFileBytes: 64})
	if r.err != nil {
		t.Fatal(r.err)
	}
	wantFiles := []string{".gitignore", ".vaktignore", "main.go", "pkg/.gitignore", "pkg/a.py", "pkg/deep/b.js", "pkg/keep.log"}
	if got := keys(r.files); !slices.Equal(got, wantFiles) {
		t.Errorf("files = %v\nwant    %v", got, wantFiles)
	}
	wantSkips := map[string]string{
		".git":         ReasonSkipDir,
		"node_modules": ReasonSkipDir,
		"sub/vendor":   ReasonSkipDir,
		"dist":         ReasonSkipDir,
		"__pycache__":  ReasonSkipDir,
		"bin.dat":      ReasonBinary,
		"app.min.js":   ReasonMinified,
		"packed.js":    ReasonTooLarge, // 5000 > 64
		"big.txt":      ReasonTooLarge,
	}
	for rel, reason := range wantSkips {
		if r.skips[rel] != reason {
			t.Errorf("skip[%s] = %q, want %q", rel, r.skips[rel], reason)
		}
	}
	f := r.files["main.go"]
	if f.Size != int64(len(code)) || !filepath.IsAbs(f.Path) {
		t.Errorf("main.go = %+v", f)
	}
	if f := r.files["pkg/deep/b.js"]; !strings.HasSuffix(f.Path, filepath.Join("pkg", "deep", "b.js")) {
		t.Errorf("path %q", f.Path)
	}

	r = collect(t, context.Background(), root, Options{})
	if r.skips["packed.js"] != ReasonMinified {
		t.Errorf("packed.js reason %q", r.skips["packed.js"])
	}
}

func TestWalkGlobs(t *testing.T) {
	root := t.TempDir()
	for _, p := range []string{"a.go", "a_test.go", "x/b.go", "x/y/c.py", "docs/d.md"} {
		writeFile(t, root, p, []byte("x\n"))
	}
	r := collect(t, context.Background(), root, Options{Include: []string{"**/*.go", "**/*.py"}, Exclude: []string{"**/*_test.go", "x/y"}})
	if r.err != nil {
		t.Fatal(r.err)
	}
	if got, want := keys(r.files), []string{"a.go", "x/b.go"}; !slices.Equal(got, want) {
		t.Errorf("files = %v, want %v", got, want)
	}
	r = collect(t, context.Background(), root, Options{Include: []string{"[bad"}})
	if r.err == nil {
		t.Error("expected error for bad glob")
	}
}

func TestWalkSymlinks(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks need privileges on windows")
	}
	outside := t.TempDir()
	writeFile(t, outside, "secret.go", []byte("package s\n"))
	root := t.TempDir()
	writeFile(t, root, "real/a.go", []byte("package a\n"))
	mustLink := func(target, rel string) {
		if err := os.Symlink(target, filepath.Join(root, rel)); err != nil {
			t.Fatal(err)
		}
	}
	mustLink(filepath.Join(outside, "secret.go"), "escape.go")
	mustLink(outside, "escapedir")
	mustLink("real", "alias")                     // dir inside root
	mustLink(filepath.Join("real", "a.go"), "b.go") // file inside root
	mustLink(".", "loop")                           // loop to root
	mustLink("missing", "broken.go")

	r := collect(t, context.Background(), root, Options{})
	if got, want := keys(r.files), []string{"real/a.go"}; !slices.Equal(got, want) {
		t.Errorf("no-follow files = %v", got)
	}
	for _, rel := range []string{"escape.go", "escapedir", "alias", "b.go", "loop", "broken.go"} {
		if r.skips[rel] != ReasonSymlink {
			t.Errorf("no-follow skip[%s] = %q", rel, r.skips[rel])
		}
	}

	r = collect(t, context.Background(), root, Options{FollowSymlinks: true})
	if r.err != nil {
		t.Fatal(r.err)
	}
	files := keys(r.files)
	// real/ and alias/ are the same directory: it is walked exactly once.
	var aCount int
	for _, f := range files {
		if strings.HasSuffix(f, "/a.go") {
			aCount++
		}
	}
	if aCount != 1 || !slices.Contains(files, "b.go") {
		t.Errorf("follow files = %v", files)
	}
	for _, f := range r.files {
		if !strings.HasPrefix(f.Path, mustReal(t, root)) {
			t.Errorf("file outside root: %+v", f)
		}
	}
	for rel, want := range map[string]string{"escape.go": ReasonEscape, "escapedir": ReasonEscape, "loop": ReasonLoop, "broken.go": ReasonBroken} {
		if r.skips[rel] != want {
			t.Errorf("follow skip[%s] = %q, want %q", rel, r.skips[rel], want)
		}
	}
}

func mustReal(t *testing.T, p string) string {
	t.Helper()
	r, err := filepath.EvalSymlinks(p)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestWalkRootIsSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip()
	}
	real := t.TempDir()
	writeFile(t, real, "a.go", []byte("x\n"))
	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}
	r := collect(t, context.Background(), link, Options{})
	if r.err != nil || len(r.files) != 1 {
		t.Fatalf("err=%v files=%v", r.err, keys(r.files))
	}
}

func TestWalkPermissionDenied(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("needs a non-root unix user")
	}
	root := t.TempDir()
	writeFile(t, root, "ok.go", []byte("x\n"))
	writeFile(t, root, "locked/x.go", []byte("x\n"))
	writeFile(t, root, "noread.go", []byte("x\n"))
	if err := os.Chmod(filepath.Join(root, "locked"), 0); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(filepath.Join(root, "noread.go"), 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(filepath.Join(root, "locked"), 0o700) }) // #nosec G302 -- restore for cleanup
	r := collect(t, context.Background(), root, Options{})
	if r.err != nil {
		t.Fatal(r.err)
	}
	if !strings.HasPrefix(r.skips["locked"], ReasonError) || !strings.HasPrefix(r.skips["noread.go"], ReasonError) {
		t.Errorf("skips = %v", r.skips)
	}
	if strings.Contains(r.skips["locked"], root) {
		t.Errorf("reason leaks absolute path: %q", r.skips["locked"])
	}
	if _, ok := r.files["ok.go"]; !ok {
		t.Error("ok.go missing")
	}
}

func TestWalkCancel(t *testing.T) {
	root := t.TempDir()
	for i := range 200 {
		writeFile(t, root, filepath.Join("d", strings.Repeat("x", i%7+1), "f"+string(rune('a'+i%26))+".go"), []byte("x\n"))
	}
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan File) // nobody reads: Walk must still return
	skipped := make(chan Skip)
	done := make(chan error, 1)
	go func() { done <- Walk(ctx, root, Options{}, out, skipped) }()
	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("err = %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Walk did not return after cancel")
	}
}

func TestWalkErrors(t *testing.T) {
	if err := Walk(context.Background(), filepath.Join(t.TempDir(), "nope"), Options{}, make(chan File), nil); err == nil {
		t.Error("missing root: want error")
	}
	f := filepath.Join(t.TempDir(), "f")
	writeFile(t, filepath.Dir(f), "f", []byte("x"))
	if err := Walk(context.Background(), f, Options{}, make(chan File), nil); err == nil {
		t.Error("file root: want error")
	}
}

func TestSniffers(t *testing.T) {
	if !IsBinary([]byte("a\x00")) || IsBinary([]byte("abc")) {
		t.Error("IsBinary")
	}
	late := append(make([]byte, 0, 9000), []byte(strings.Repeat("a", 8200))...)
	late = append(late, 0)
	if IsBinary(late) {
		t.Error("NUL after 8 KiB must not count")
	}
	if IsMinified([]byte(strings.Repeat("a\n", 1000))) {
		t.Error("short lines are not minified")
	}
	if !IsMinified([]byte(strings.Repeat("a", 2000))) {
		t.Error("one long line is minified")
	}
	if IsMinified([]byte(strings.Repeat("a", 900))) {
		t.Error("tiny sample never minified")
	}
}

func benchWalk(b *testing.B, root string) {
	if _, err := os.Stat(root); err != nil {
		b.Skipf("%s not present", root)
	}
	var files int
	for b.Loop() {
		r := collect(b, context.Background(), root, Options{})
		if r.err != nil {
			b.Fatal(r.err)
		}
		files = len(r.files)
	}
	b.ReportMetric(float64(files), "files/op")
	b.ReportMetric(float64(files)*float64(b.N)/b.Elapsed().Seconds(), "files/s")
}

func BenchmarkWalkLlamaCpp(b *testing.B) { benchWalk(b, "/Users/shearer/vaktex/llama.cpp") }
func BenchmarkWalkJuiceShop(b *testing.B) { benchWalk(b, "/Users/shearer/vaktex/juice-shop") }
