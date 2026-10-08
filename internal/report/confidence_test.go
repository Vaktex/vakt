package report

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/labels"
)

func confidenceReport() *Report {
	r := &Report{
		Units: []Unit{
			{File: "low.go", Name: "low_confidence", Severity: .99, TopFamily: "memory_safety", TopFamilyProb: .8999},
			{File: "mixed.go", Name: "boundary", Severity: .95, TopFamily: "memory_safety", TopFamilyProb: .90},
			{File: "mixed.go", Name: "below_severity", Severity: .9499, TopFamily: "memory_safety", TopFamilyProb: .99},
			{File: "other.go", Name: "strong", Severity: .96, TopFamily: "authorization", TopFamilyProb: .99},
		},
		Scan:    Scan{Files: 3, Units: 4, Tokens: 100},
		Summary: Summary{Threshold: .95},
	}
	sortUnits(r.Units)
	return r
}

func TestApplyConfidence(t *testing.T) {
	r := confidenceReport()
	ApplyConfidence(r, .90)
	if r.Summary.MinConfidence != .90 || r.Summary.FlaggedUnits != 2 || r.Summary.FlaggedFiles != 2 {
		t.Fatalf("summary = %+v", r.Summary)
	}
	if r.Summary.ByFamily[labels.Index("memory_safety")] != 1 || r.Summary.ByFamily[labels.Index("authorization")] != 1 {
		t.Fatalf("family counts = %+v", r.Summary.ByFamily)
	}
	want := map[string]bool{"boundary": true, "strong": true}
	for _, u := range r.Units {
		if u.Flagged != want[u.Name] {
			t.Errorf("%s flagged = %v", u.Name, u.Flagged)
		}
	}
	if len(r.Units) != 4 || r.Scan.Units != 4 || r.Scan.Files != 3 || r.Scan.Tokens != 100 {
		t.Fatalf("raw scan changed: %+v", r.Scan)
	}
	counts := map[string]int{"low.go": 0, "mixed.go": 1, "other.go": 1}
	if len(r.Files) != 3 {
		t.Fatalf("files = %+v", r.Files)
	}
	for _, f := range r.Files {
		if f.FlaggedUnits != counts[f.File] {
			t.Errorf("file counts = %+v", f)
		}
		if f.File == "mixed.go" && f.Units != 2 {
			t.Errorf("all file units not retained: %+v", f)
		}
	}
	var b bytes.Buffer
	if err := WriteJSON(&b, r); err != nil {
		t.Fatal(err)
	}
	var decoded Report
	if err := json.Unmarshal(b.Bytes(), &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Units) != 4 || decoded.Summary.MinConfidence != .90 {
		t.Fatalf("JSON lost raw scores or gate: %+v", decoded)
	}
	ApplyConfidence(r, .99)
	if r.Summary.FlaggedUnits != 1 || r.Summary.FlaggedFiles != 1 {
		t.Fatalf("reapplied summary = %+v", r.Summary)
	}
	ApplyConfidence(r, 0)
	if r.Summary.FlaggedUnits != 3 || r.Summary.FlaggedFiles != 3 {
		t.Fatalf("zero confidence must restore severity-only behavior: %+v", r.Summary)
	}
	b.Reset()
	if err := WriteJSON(&b, r); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(b.String(), "min_confidence") {
		t.Fatal("zero confidence should be omitted from JSON")
	}
}

func TestConfidenceFilterPrettyAndShow(t *testing.T) {
	r := confidenceReport()
	ApplyConfidence(r, .90)
	f := Filter(r, .94, []string{"memory_safety"})
	if f.Summary.MinConfidence != .90 || f.Summary.FlaggedUnits != 2 || f.Summary.FlaggedFiles != 1 || len(f.Units) != 3 {
		t.Fatalf("filtered summary = %+v; units = %d", f.Summary, len(f.Units))
	}
	if r.Summary.FlaggedUnits != 2 || len(r.Units) != 4 {
		t.Fatal("Filter mutated input")
	}
	v := newView(r, PrettyOptions{Threshold: .94, Families: []string{"memory_safety"}})
	if len(v.flagged) != f.Summary.FlaggedUnits || len(v.files) != f.Summary.FlaggedFiles || FamilyCounts(v.byFamily) != f.Summary.ByFamily {
		t.Fatalf("view and Filter disagree: %+v versus %+v", v, f.Summary)
	}
	for _, min := range []float64{0, .80, .90} {
		if got := len(newView(r, PrettyOptions{MinConfidence: min}).flagged); got != 2 {
			t.Errorf("minimum %v bypassed report floor: %d", min, got)
		}
	}
	if got := len(newView(r, PrettyOptions{MinConfidence: .99}).flagged); got != 1 {
		t.Errorf("stricter view minimum = %d", got)
	}
	u, err := FlaggedAt(r, 1, 0)
	if err != nil || u.Name != "strong" {
		t.Fatalf("Finding = %+v, %v", u, err)
	}
	if _, err := FlaggedAt(r, 3, 0); err == nil {
		t.Fatal("Finding exposed low-confidence unit")
	}
	var b bytes.Buffer
	if err := Pretty(&b, r, PrettyOptions{Quiet: true}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), "2 units flagged across 2 files") {
		t.Fatalf("pretty summary = %s", b.String())
	}
	b.Reset()
	if err := Pretty(&b, r, PrettyOptions{MinConfidence: 1}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), "All clear") || !strings.Contains(b.String(), "confidence") || strings.Contains(b.String(), "SEVERITY") {
		t.Fatalf("empty confidence view = %s", b.String())
	}
	r.Summary.MinConfidence = 0 // Report CLI can explicitly override the persisted gate.
	if got := len(newView(r, PrettyOptions{}).flagged); got != 3 {
		t.Errorf("explicit summary reset failed: %d", got)
	}
}
