package main

import (
	"context"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/report"
)

func TestStrictFindingDefaults(t *testing.T) {
	isolate(t)
	old := runScan
	t.Cleanup(func() { runScan = old })
	runScan = func(_ context.Context, o ScanOptions, _ *report.Progress) (*report.Report, error) {
		if o.Threshold != 0.95 {
			t.Errorf("threshold = %v", o.Threshold)
		}
		var scores core.Scores
		scores.Severity = 0.99
		for i := range scores.Families {
			scores.Families[i] = 0.01
		}
		return report.Build(report.Meta{}, []report.Result{{Unit: core.Unit{File: "noise.go", Name: "noise"}, Scores: scores}}, o.Threshold), nil
	}
	code, out, errout := runCLI(t, "", "patrol", t.TempDir(), "--format", "json", "--out", "-")
	if code != exitOK {
		t.Fatalf("low confidence caused exit %d: %s", code, errout)
	}
	if !strings.Contains(out, `"flagged_units": 0`) || !strings.Contains(out, "noise.go") {
		t.Fatalf("raw result missing or flagged: %s", out)
	}
	code, _, errout = runCLI(t, "", "patrol", t.TempDir(), "--min-confidence", "0", "--quiet")
	if code != exitFindings {
		t.Fatalf("override exit=%d %s", code, errout)
	}
	code, _, errout = runCLI(t, "", "patrol", t.TempDir(), "--fail-on", "0.5", "--quiet")
	if code != exitOK {
		t.Fatalf("fail-on ignored confidence gate: %d %s", code, errout)
	}
	code, broad, errout := runCLI(t, out, "report", "-", "--min-confidence", "0", "--format", "json")
	if code != exitOK || !strings.Contains(broad, `"flagged_units": 1`) {
		t.Fatalf("saved report override: %d %s %s", code, broad, errout)
	}
}

func TestMinConfidenceValidation(t *testing.T) {
	isolate(t)
	for _, value := range []string{"-0.1", "1.1", "NaN"} {
		code, _, errout := runCLI(t, "", "patrol", t.TempDir(), "--min-confidence", value)
		if code != exitError || !strings.Contains(errout, "--min-confidence") {
			t.Errorf("%s: %d %s", value, code, errout)
		}
	}
}
