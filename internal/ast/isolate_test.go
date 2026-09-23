package ast

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/vaktex/vakt/internal/core"
)

func TestMain(m *testing.M) {
	MaybeServeWorker() // the test binary doubles as the parse worker
	os.Exit(m.Run())
}

func newIso(t *testing.T, mem int64) *Isolated {
	iso, err := NewIsolated(mem)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(iso.Close)
	return iso
}

// Isolated parses must match in-process parses exactly.
func TestIsolatedMatchesInProcess(t *testing.T) {
	iso := newIso(t, 0)
	for _, c := range []struct{ lang, file, src string }{
		{"Python", "a.py", "import os\n\n@dec\ndef run(cmd):\n    return os.system(cmd)\n\nclass A:\n    def m(self):\n        return 1\n\nx = 1\ny = 2\nz = 3\n"},
		{"JavaScript", "a.js", "app.get('/x', (req, res) => {\n  a()\n  b()\n})\nfunction f() {\n  return 1\n}\n"},
		{"Go", "a.go", "package a\n\nfunc (s *S) Get() int {\n\treturn 1\n}\n"},
	} {
		want, werr := Extract(context.Background(), c.file, c.lang, []byte(c.src), Options{})
		got, gerr := iso.Extract(context.Background(), c.file, c.lang, []byte(c.src), Options{})
		if (werr == nil) != (gerr == nil) || len(got) != len(want) {
			t.Fatalf("%s: got %v %v, want %v %v", c.lang, got, gerr, want, werr)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("%s unit %d:\n got  %+v\n want %+v", c.lang, i, got[i], want[i])
			}
		}
	}
}

// Inputs that blow up tree-sitter's error recovery without tripping the
// cheap pre-scan are contained by the worker's memory budget / deadline.
func TestIsolatedContainsBlowups(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns workers that allocate")
	}
	iso := newIso(t, 256<<20)
	kotlin := strings.Repeat("when (x) \n", 200_000) // 2 MB, no brackets unbalanced
	start := time.Now()
	units, err := iso.Extract(context.Background(), "a.kt", "Kotlin", []byte(kotlin), Options{ParseTimeout: 30 * time.Second})
	if err != ErrParseLimit && err != ErrParseTimeout {
		t.Fatalf("err %v", err)
	}
	if len(units) != 1 || units[0].Kind != core.KindFile {
		t.Fatalf("units %+v", units)
	}
	t.Logf("contained in %v (%v)", time.Since(start), err)
	// The pool keeps working afterwards.
	u, err := iso.Extract(context.Background(), "b.py", "Python", []byte("def f():\n    return 1\n"), Options{})
	if err != nil || len(u) == 0 {
		t.Fatalf("after kill: %v %v", u, err)
	}
}

func TestIsolatedConcurrentAndCancel(t *testing.T) {
	iso := newIso(t, 0)
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Go(func() {
			u, err := iso.Extract(context.Background(), "a.py", "Python", []byte("def f():\n    return 1\n"), Options{})
			if err != nil || len(u) != 1 || u[0].Name != "f" {
				t.Errorf("%v %v", u, err)
			}
		})
	}
	wg.Wait()
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	_, err := iso.Extract(ctx, "a.kt", "Kotlin", []byte(strings.Repeat("when (x) \n", 200_000)), Options{ParseTimeout: time.Minute})
	if err != context.Canceled {
		t.Fatalf("cancel: %v", err)
	}
}

// A worker that returns garbage spans is not trusted.
func TestRebuildValidates(t *testing.T) {
	src := []byte("def f():\n    return 1\n")
	units, _ := rebuild(response{Units: []wireUnit{{Unit: core.Unit{Kind: core.KindFunction}, StartByte: 0, EndByte: 1 << 30}}}, "a.py", "Python", src)
	if len(units) != 1 || units[0].Kind != core.KindFile {
		t.Fatalf("%+v", units)
	}
}

// A worker whose parent dies exits on its own instead of running on as an
// orphan (the parent can be SIGKILLed mid-scan).
func TestWorkerExitsWhenOrphaned(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// sh starts the worker in the background with stdin on a FIFO that we
	// hold open for writing, prints its pid and exits, orphaning it. The
	// worker never sees EOF, so only the parent watch can make it exit.
	fifo := filepath.Join(t.TempDir(), "in")
	if err := syscall.Mkfifo(fifo, 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("/bin/sh", "-c", `"$0" < "$1" > /dev/null 2>&1 & echo $!`, exe, fifo)
	cmd.Env = append(os.Environ(), WorkerEnv+"=1")
	res := make(chan []byte, 1)
	go func() { out, _ := cmd.Output(); res <- out }()
	w, err := os.OpenFile(fifo, os.O_WRONLY, 0) // blocks until the worker opens it
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	out := <-res
	pid, err := strconv.Atoi(strings.TrimSpace(string(out)))
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if syscall.Kill(pid, 0) != nil {
			return // gone
		}
		time.Sleep(50 * time.Millisecond)
	}
	_ = syscall.Kill(pid, syscall.SIGKILL)
	t.Fatalf("orphaned worker %d still running", pid)
}

func TestWorkerEnvAllowlist(t *testing.T) {
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "x")
	t.Setenv("HF_TOKEN", "x")
	t.Setenv("GITHUB_TOKEN", "x")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "x")
	for _, kv := range workerEnv() {
		k, _, _ := strings.Cut(kv, "=")
		switch k {
		case "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "GODEBUG", "GOMAXPROCS", "GOGC", "GOMEMLIMIT":
		default:
			t.Errorf("worker env leaks %s", k)
		}
	}
}
