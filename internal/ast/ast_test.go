package ast

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/vaktex/vakt/internal/core"
)

var update = flag.Bool("update", false, "rewrite golden files")

type summary struct {
	Kind  string `json:"kind"`
	Name  string `json:"name"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

// TestGolden extracts every sample and compares (kind, name, lines) with
// testdata/golden.json. Run with -update after reviewing a change.
func TestGolden(t *testing.T) {
	files, err := filepath.Glob("testdata/src/*")
	if err != nil || len(files) == 0 {
		t.Fatal("no samples")
	}
	got := map[string][]summary{}
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		rel := filepath.Base(f)
		lang, ok := Detect(rel, src)
		if !ok {
			t.Fatalf("%s: not detected", rel)
		}
		units, err := Extract(context.Background(), rel, lang, src, Options{})
		if err != nil {
			t.Fatalf("%s: %v", rel, err)
		}
		var ss []summary
		for _, u := range units {
			ss = append(ss, summary{u.Kind, u.Name, u.StartLine, u.EndLine})
			if u.Code != string(src[u.StartByte:u.EndByte]) && u.Kind != core.KindResidual {
				t.Errorf("%s %s: Code is not the source slice", rel, u.Name)
			}
			if u.Language != lang || u.File != rel {
				t.Errorf("%s: unit metadata %+v", rel, u)
			}
		}
		got[lang+" "+rel] = ss
	}
	path := "testdata/golden.json"
	if *update {
		b, _ := json.MarshalIndent(got, "", "  ")
		if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
		return
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run with -update)", err)
	}
	var want map[string][]summary
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	keys := make([]string, 0, len(got))
	for k := range got {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if fmt.Sprint(got[k]) != fmt.Sprint(want[k]) {
			t.Errorf("%s:\n got  %v\n want %v", k, got[k], want[k])
		}
	}
}

// Every grammar must load and parse its own sample into at least one
// function/method unit (catches wrong node-kind tables).
func TestEveryGrammarFindsFunctions(t *testing.T) {
	samples := map[string]string{}
	files, _ := filepath.Glob("testdata/src/*")
	for _, f := range files {
		src, _ := os.ReadFile(f)
		if lang, ok := Detect(filepath.Base(f), src); ok {
			samples[lang] = f
		}
	}
	for _, lang := range Languages() {
		if lang == "HCL" {
			continue // shares the Terraform sample
		}
		f, ok := samples[lang]
		if !ok {
			t.Errorf("%s: no sample file", lang)
			continue
		}
		src, _ := os.ReadFile(f)
		units, err := Extract(context.Background(), filepath.Base(f), lang, src, Options{})
		if err != nil {
			t.Errorf("%s: %v", lang, err)
			continue
		}
		n := 0
		for _, u := range units {
			if u.Kind == core.KindFunction || u.Kind == core.KindMethod {
				n++
			}
		}
		if n == 0 {
			t.Errorf("%s: no function units in %s (kinds table wrong?): %+v", lang, f, units)
		}
	}
}

func TestDetect(t *testing.T) {
	for _, c := range []struct {
		rel, head, want string
		ok              bool
	}{
		{"a/b.py", "", "Python", true},
		{"x.TSX", "", "TypeScript", true},
		{"inc/x.h", "class A {};", "C++", true},
		{"inc/x.h", "int f(void);", "C", true},
		{"bin/tool", "#!/usr/bin/env python3\n", "Python", true},
		{"bin/run", "#!/bin/sh\n", "Bash", true},
		{"README.md", "", "", false},
		{"package-lock.json", "", "", false},
		{"go.sum", "", "", false},
		{"Dockerfile", "", "Bash", true},
		{"config.yml", "", "YAML", true},
		{"x.unknownext", "", "", false},
		{"noext", "plain text", "", false},
	} {
		got, ok := Detect(c.rel, []byte(c.head))
		if got != c.want || ok != c.ok {
			t.Errorf("Detect(%q) = %q,%v want %q,%v", c.rel, got, ok, c.want, c.ok)
		}
	}
}

func TestNoGrammarIsWholeFile(t *testing.T) {
	src := []byte("key: value\nother: 1\n")
	units, err := Extract(context.Background(), "a.yaml", "YAML", src, Options{})
	if err != nil || len(units) != 1 || units[0].Kind != core.KindFile || units[0].EndLine != 2 {
		t.Fatalf("got %+v %v", units, err)
	}
}

// Split parts cover the unit's code exactly, in order, each within budget,
// with correct line numbers and parent span.
func TestSplitOversize(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 400; i++ {
		fmt.Fprintf(&b, "    x_%d = compute(%d)  # line %d\n", i, i, i)
		if i%37 == 0 {
			b.WriteString("\n")
		}
	}
	b.WriteString(strings.Repeat("é", 3000) + "\n") // one pathological long line of multibyte runes
	code := b.String()
	u := core.Unit{File: "f.py", Language: "Python", Kind: core.KindFunction, Name: "big", StartLine: 10, StartByte: 100, Code: code}
	u.EndLine = u.StartLine + strings.Count(code, "\n") - 1
	count := func(s string) int { return len(s) / 4 } // a stand-in token counter
	const budget = 500
	parts := SplitOversize(u, nil, count, budget)
	if len(parts) < 2 {
		t.Fatalf("not split: %d parts", len(parts))
	}
	var joined strings.Builder
	line, off := u.StartLine, u.StartByte
	for i, p := range parts {
		if count(p.Code) > budget {
			t.Errorf("part %d over budget: %d", i, count(p.Code))
		}
		if !utf8.ValidString(p.Code) {
			t.Errorf("part %d split a rune", i)
		}
		if p.SplitPart != i+1 || p.SplitOf != len(parts) || p.ParentStartLine != u.StartLine || p.ParentEndLine != u.EndLine {
			t.Errorf("part %d metadata %+v", i, p)
		}
		if p.StartLine != line || p.StartByte != off {
			t.Errorf("part %d starts at line %d byte %d, want %d/%d", i, p.StartLine, p.StartByte, line, off)
		}
		line += strings.Count(p.Code, "\n")
		off += len(p.Code)
		joined.WriteString(p.Code)
	}
	if joined.String() != code {
		t.Fatal("parts do not reassemble the unit")
	}
	if got := SplitOversize(core.Unit{Code: "short"}, nil, count, budget); len(got) != 1 || got[0].SplitOf != 0 {
		t.Fatalf("small unit was split: %+v", got)
	}
}

func TestHostileInput(t *testing.T) {
	ctx := context.Background()
	for _, lang := range Languages() {
		for _, src := range [][]byte{
			{0xff, 0xfe, 0x00, 'd', 'e', 'f'},
			[]byte(strings.Repeat("(", 20000)),
			[]byte(strings.Repeat("{", 20000) + strings.Repeat("}", 20000)),
			[]byte("def \x00\x01 f():\n  pass\n"),
		} {
			if _, err := Extract(ctx, "x", lang, src, Options{ParseTimeout: 2 * time.Second}); err != nil && err != ErrParseTimeout {
				t.Errorf("%s: %v", lang, err)
			}
		}
	}
}

func TestUnitCap(t *testing.T) {
	var b strings.Builder
	for i := 0; i < MaxUnitsPerFile+10; i++ {
		fmt.Fprintf(&b, "def f%d():\n    pass\n", i)
	}
	units, err := Extract(context.Background(), "gen.py", "Python", []byte(b.String()), Options{})
	if err != nil || len(units) != 1 || units[0].Kind != core.KindFile {
		t.Fatalf("expected whole-file fallback, got %d units %v", len(units), err)
	}
}

func FuzzExtract(f *testing.F) {
	files, _ := filepath.Glob("testdata/src/*")
	for _, p := range files {
		b, _ := os.ReadFile(p)
		f.Add(b, uint8(0))
	}
	langs := Languages()
	sort.Strings(langs)
	f.Fuzz(func(t *testing.T, src []byte, li uint8) {
		lang := langs[int(li)%len(langs)]
		units, _ := Extract(context.Background(), "f", lang, src, Options{ParseTimeout: time.Second})
		for _, u := range units {
			if u.StartByte < 0 || u.EndByte > len(src) || u.StartByte > u.EndByte || u.StartLine < 1 || u.EndLine < u.StartLine {
				t.Fatalf("bad unit %+v for %d-byte input", u, len(src))
			}
		}
	})
}

func BenchmarkExtractRepo(b *testing.B) {
	for _, root := range []string{"/Users/shearer/vaktex/llama.cpp/src", "/Users/shearer/vaktex/juice-shop/routes"} {
		type file struct {
			rel, lang string
			src       []byte
		}
		var files []file
		_ = filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			src, err := os.ReadFile(p)
			if err != nil {
				return nil
			}
			if lang, ok := Detect(p, src); ok {
				files = append(files, file{p, lang, src})
			}
			return nil
		})
		if len(files) == 0 {
			continue
		}
		b.Run(filepath.Base(filepath.Dir(root))+"/"+filepath.Base(root), func(b *testing.B) {
			var units, bytes int
			for range b.N {
				units, bytes = 0, 0
				for _, f := range files {
					us, _ := Extract(context.Background(), f.rel, f.lang, f.src, Options{})
					units += len(us)
					bytes += len(f.src)
				}
			}
			b.ReportMetric(float64(len(files)*b.N)/b.Elapsed().Seconds(), "files/s")
			b.ReportMetric(float64(units*b.N)/b.Elapsed().Seconds(), "units/s")
			b.ReportMetric(float64(bytes*b.N)/b.Elapsed().Seconds()/1e6, "MB/s")
		})
	}
}
