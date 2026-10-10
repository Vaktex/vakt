package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/report"
	"github.com/vaktex/vakt/internal/results"
	"github.com/vaktex/vakt/internal/walk"
)

// fakeScorer reports one unit per file, named after the content it was given.
type fakeScorer struct{ calls int }

func (f *fakeScorer) Score(_ context.Context, root string, files []walk.File) (*report.Report, error) {
	f.calls++
	var rs []report.Result
	for _, fl := range files {
		r := report.Result{Tokens: 10}
		r.Unit.File, r.Unit.Name, r.Unit.StartLine, r.Unit.EndLine = fl.Rel, strings.TrimSpace(string(fl.Data)), 1, 1
		r.Scores.Severity = 0.99
		r.Scores.Families[0] = 0.99
		rs = append(rs, r)
	}
	return report.Build(report.Meta{Root: root, ModelRepo: "fake"}, rs, 0.5), nil
}

func (f *fakeScorer) Close() error { return nil }

func TestServe(t *testing.T) {
	isolate(t)
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.py"), []byte("on_disk\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	fs := &fakeScorer{}
	prev := openServeScorer
	openServeScorer = func(context.Context, ScanOptions, io.Writer) (serveScorer, error) { return fs, nil }
	t.Cleanup(func() { openServeScorer = prev })

	in := strings.Join([]string{
		`{"id":1,"root":"` + root + `","file":"a.py"}`,
		`{"id":2,"root":"` + root + `","file":"a.py","content":"unsaved"}`,
		`{"id":3,"root":"` + root + `","file":"../x.py"}`,
		`{"id":4,"root":"relative","file":"a.py"}`,
		`not json`,
	}, "\n") + "\n"
	var out bytes.Buffer
	o := ScanOptions{Threshold: 0.5, MinConfidence: 0, MaxFileBytes: 1 << 20, SaveResults: true}
	if err := serve(context.Background(), strings.NewReader(in), &out, io.Discard, o); err != nil {
		t.Fatal(err)
	}
	got := map[string]string{}
	for _, l := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		var m struct {
			ID    json.RawMessage `json:"id"`
			Type  string          `json:"type"`
			Units []report.Unit   `json:"units"`
		}
		if err := json.Unmarshal([]byte(l), &m); err != nil {
			t.Fatalf("%q: %v", l, err)
		}
		v := m.Type
		if len(m.Units) > 0 {
			v += ":" + m.Units[0].Name
		}
		got[string(m.ID)] += v
	}
	// Requests 1 and 2 may arrive before or after the worker picks up 1: either
	// 1 was superseded, or it ran on disk contents.
	if got["2"] != "result:unsaved" || (got["1"] != "superseded" && got["1"] != "result:on_disk") {
		t.Errorf("replies = %v", got)
	}
	if got["3"] != "error" || got["4"] != "error" || !strings.Contains(got[""], "starterror") {
		t.Errorf("replies = %v", got)
	}
	// The unsaved buffer's result is stored under its own hash: it is not
	// "fresh" for the file on disk.
	key, _ := results.Key(os.Getenv("VAKT_CACHE"))
	d, _ := results.Load(root, key)
	if e, ok := d.Files["a.py"]; !ok || e.SHA256 != results.Hash([]byte("unsaved")) {
		t.Errorf("stored entry = %+v", d.Files)
	}
}
