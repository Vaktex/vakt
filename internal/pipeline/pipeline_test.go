package pipeline

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine"
	"github.com/vaktex/vakt/internal/report"
	"github.com/vaktex/vakt/internal/tokenize"
)

func writeTree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, body := range files {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// countingEngine wraps Fake, counting calls and checking the contract.
type countingEngine struct {
	engine.Fake
	calls, seqs atomic.Int64
	budget      int
	fail        error
	t           *testing.T
}

func (c *countingEngine) MaxBatchTokens() int { return c.budget }

func (c *countingEngine) Score(ctx context.Context, batch [][]int32) ([]core.Scores, error) {
	if c.fail != nil {
		return nil, c.fail
	}
	c.calls.Add(1)
	c.seqs.Add(int64(len(batch)))
	longest := 0
	for _, ids := range batch {
		if len(ids) == 0 || len(ids) > core.MaxTokens {
			c.t.Errorf("engine got a sequence of %d tokens", len(ids))
		}
		longest = max(longest, len(ids))
	}
	if padded := longest * len(batch); len(batch) > 1 && padded > c.budget {
		c.t.Errorf("batch of %d padded tokens exceeds budget %d", padded, c.budget)
	}
	return c.Fake.Score(ctx, batch)
}

func newTok(t *testing.T) core.Tokenizer {
	tok, err := tokenize.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { tok.Close() })
	return tok
}

func sampleTree(t *testing.T) string {
	var big strings.Builder
	big.WriteString("def huge():\n")
	for i := 0; i < 9000; i++ {
		fmt.Fprintf(&big, "    value_%d = transform(value_%d, %d)  # step %d\n", i, i-1, i, i)
	}
	return writeTree(t, map[string]string{
		"app/main.py":         "import os\n\ndef run(cmd):\n    return os.system(cmd)\n\n\ndef ok():\n    return 1\n",
		"app/big.py":          big.String(),
		"web/server.js":       "function h(req, res) { res.end(req.url) }\nmodule.exports = h\n",
		"README.md":           "# docs\n",
		"node_modules/x/i.js": "function ignored() {}\n",
		"data.bin":            "\x00\x01\x02binary",
		".gitignore":          "generated/\n",
		"generated/g.py":      "def gen(): pass\n",
	})
}

func TestRunEndToEnd(t *testing.T) {
	root := sampleTree(t)
	eng := &countingEngine{budget: 32768, t: t}
	prog := &report.Progress{}
	rep, err := Run(context.Background(), Config{Root: root, Jobs: 4, Threshold: 0.5, CacheDir: t.TempDir()}, []core.Engine{eng}, newTok(t), prog)
	if err != nil {
		t.Fatal(err)
	}
	files := map[string]bool{}
	parts := 0
	for _, u := range rep.Units {
		files[u.File] = true
		if u.File == "app/big.py" && u.SplitOf > 1 {
			parts = u.SplitOf
		}
	}
	for _, want := range []string{"app/main.py", "app/big.py", "web/server.js"} {
		if !files[want] {
			t.Errorf("missing %s in report (files: %v)", want, files)
		}
	}
	for _, bad := range []string{"README.md", "node_modules/x/i.js", "data.bin", "generated/g.py"} {
		if files[bad] {
			t.Errorf("%s should not be scored", bad)
		}
	}
	if parts < 2 {
		t.Errorf("oversize unit was not split (split_of=%d)", parts)
	}
	if rep.Scan.Files != 3 || prog.UnitsScored.Load() == 0 || prog.TokensScored.Load() != prog.TokensTotal.Load() {
		t.Errorf("counters: files=%d scored=%d tokens %d/%d", rep.Scan.Files, prog.UnitsScored.Load(), prog.TokensScored.Load(), prog.TokensTotal.Load())
	}
	if rep.Model.Backend != "fake" {
		t.Errorf("backend %q", rep.Model.Backend)
	}
}

func TestCacheReuse(t *testing.T) {
	root := sampleTree(t)
	dir := t.TempDir()
	tok := newTok(t)
	e1 := &countingEngine{budget: 32768, t: t}
	r1, err := Run(context.Background(), Config{Root: root, Jobs: 2, CacheDir: dir}, []core.Engine{e1}, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	e2 := &countingEngine{budget: 32768, t: t}
	r2, err := Run(context.Background(), Config{Root: root, Jobs: 2, CacheDir: dir}, []core.Engine{e2}, tok, nil)
	if err != nil {
		t.Fatal(err)
	}
	if e2.calls.Load() != 0 {
		t.Errorf("second run scored %d batches; everything should be cached", e2.calls.Load())
	}
	if r2.Scan.CacheHits != int(e1.seqs.Load()) {
		t.Errorf("cache hits %d, want %d", r2.Scan.CacheHits, e1.seqs.Load())
	}
	if len(r1.Units) != len(r2.Units) {
		t.Fatalf("unit count differs: %d vs %d", len(r1.Units), len(r2.Units))
	}
	for i := range r1.Units {
		if r1.Units[i].Severity != r2.Units[i].Severity {
			t.Fatalf("cached severity differs for %s", r1.Units[i].File)
		}
	}
	// A different precision must not reuse fp32 entries.
	e3 := &countingEngine{budget: 32768, t: t}
	e3p := &precEngine{countingEngine: e3, prec: "bf16"}
	if _, err := Run(context.Background(), Config{Root: root, Jobs: 2, CacheDir: dir}, []core.Engine{e3p}, tok, nil); err != nil {
		t.Fatal(err)
	}
	if e3.calls.Load() == 0 {
		t.Error("bf16 run reused fp32 cache entries")
	}
}

type precEngine struct {
	*countingEngine
	prec string
}

func (p *precEngine) Info() core.EngineInfo {
	i := p.countingEngine.Info()
	i.Precision = p.prec
	return i
}

func TestBatchBudgetRespected(t *testing.T) {
	files := map[string]string{}
	for i := 0; i < 60; i++ {
		files[fmt.Sprintf("m%d.py", i)] = fmt.Sprintf("def f%d(x):\n    return x + %d\n", i, i)
	}
	root := writeTree(t, files)
	eng := &countingEngine{budget: 256, t: t}
	if _, err := Run(context.Background(), Config{Root: root, Jobs: 4, NoCache: true}, []core.Engine{eng}, newTok(t), nil); err != nil {
		t.Fatal(err)
	}
	if eng.seqs.Load() != 60 {
		t.Errorf("scored %d sequences, want 60", eng.seqs.Load())
	}
	if eng.calls.Load() < 2 {
		t.Errorf("expected several batches under a small budget, got %d", eng.calls.Load())
	}
}

func TestEngineErrorAborts(t *testing.T) {
	root := sampleTree(t)
	boom := errors.New("gpu fell over")
	eng := &countingEngine{budget: 32768, t: t, fail: boom}
	_, err := Run(context.Background(), Config{Root: root, Jobs: 2, NoCache: true}, []core.Engine{eng}, newTok(t), nil)
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want engine error", err)
	}
}

func TestCancel(t *testing.T) {
	root := sampleTree(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Run(ctx, Config{Root: root, Jobs: 2, NoCache: true}, []core.Engine{&countingEngine{budget: 32768, t: t}}, newTok(t), nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestMultipleEngines(t *testing.T) {
	files := map[string]string{}
	for i := 0; i < 40; i++ {
		files[fmt.Sprintf("m%d.go", i)] = fmt.Sprintf("package m\n\nfunc F%d() int { return %d }\n", i, i)
	}
	root := writeTree(t, files)
	a := &countingEngine{budget: 128, t: t}
	b := &countingEngine{budget: 128, t: t}
	rep, err := Run(context.Background(), Config{Root: root, Jobs: 4, NoCache: true}, []core.Engine{a, b}, newTok(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	if a.seqs.Load()+b.seqs.Load() != int64(len(rep.Units)) || len(rep.Units) < 40 {
		t.Errorf("engines scored %d+%d, report has %d units", a.seqs.Load(), b.seqs.Load(), len(rep.Units))
	}
}

func TestCacheRejectsCorruptEntries(t *testing.T) {
	c, err := OpenCache(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	var s core.Scores
	s.Severity = 2 // out of range
	if err := c.Put(map[string]cached{"k": {Tokens: 5, Scores: s}}); err != nil {
		t.Fatal(err)
	}
	if got := c.Get([]string{"k"}); len(got) != 0 {
		t.Fatalf("corrupt entry returned: %+v", got)
	}
}

func TestBatcher(t *testing.T) {
	b := newBatcher(core.MaxTokens, 256)
	var all [][]core.Encoded
	for i := 1; i <= 500; i++ {
		all = append(all, b.add(core.Encoded{IDs: make([]int32, 1+(i*37)%3000)})...)
	}
	all = append(all, b.flush()...)
	n := 0
	for _, bt := range all {
		longest := 0
		for _, e := range bt {
			longest = max(longest, len(e.IDs))
		}
		if longest*len(bt) > core.MaxTokens || len(bt) > 256 {
			t.Fatalf("batch %d x %d over budget", len(bt), longest)
		}
		n += len(bt)
	}
	if n != 500 {
		t.Fatalf("batched %d of 500", n)
	}
}
