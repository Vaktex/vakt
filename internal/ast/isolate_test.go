package ast

import (
	"context"
	"os"
	"strings"
	"sync"
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
