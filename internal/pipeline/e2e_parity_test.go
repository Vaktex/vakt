//go:build mlx

package pipeline

// End-to-end parity: write each fixture's code to a file, scan the tree
// through the real pipeline (walk, extract, render, tokenize, batch, MLX
// engine) and compare every scored unit whose code is exactly the fixture's
// code with the Python reference. This covers what engine parity cannot:
// that the CLI feeds the model the same text training did.

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/ast"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/engine"
	"github.com/vaktex/vakt/internal/report"
)

var extFor = map[string]string{
	"Python": ".py", "JavaScript": ".js", "TypeScript": ".ts", "Go": ".go", "Java": ".java",
	"C": ".c", "C++": ".cpp", "C#": ".cs", "PHP": ".php", "Ruby": ".rb", "Rust": ".rs",
	"Bash": ".sh", "Kotlin": ".kt", "Scala": ".scala", "Lua": ".lua", "Solidity": ".sol",
}

func TestEndToEndParity(t *testing.T) {
	root := filepath.Join("..", "..", "testdata")
	b, err := os.ReadFile(filepath.Join(root, "parity", "fixtures.json"))
	if err != nil {
		t.Skip("fixtures.json not present")
	}
	model := filepath.Join(root, "models", "mock-dom-0.8b", "model.safetensors")
	if _, err := os.Stat(model); err != nil {
		t.Skip("mock model not present")
	}
	var fx struct {
		Samples []fixture `json:"samples"`
	}
	if err := json.Unmarshal(b, &fx); err != nil {
		t.Fatal(err)
	}
	// Fixtures whose code already has training form (no surrounding
	// whitespace) are compared with their reference score. The others were
	// scored by the reference with a trailing newline, which the CLI now
	// deliberately drops; they only have to run.
	tree := t.TempDir()
	want := map[string]float64{} // file -> reference severity
	for _, s := range fx.Samples {
		ext, ok := extFor[s.Language]
		if !ok {
			continue
		}
		name := s.Name + ext
		// The file ends in a newline like real source (whatever the
		// fixture had); the pipeline must feed the model the trimmed,
		// training-form text.
		if err := os.WriteFile(filepath.Join(tree, name), []byte(strings.TrimSpace(s.Code)+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		want[name] = s.Severity
	}
	if len(want) < 10 { // all 48 fixtures are written; only trimmed ones are compared
		t.Fatalf("only %d fixtures usable", len(want))
	}

	dev := "auto"
	if os.Getenv("VAKT_TEST_DEVICE") == "cpu" {
		dev = "cpu"
	}
	eng, err := engine.Open(engine.Options{ModelPath: model, Precision: "fp32", Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	rep, err := Run(context.Background(), Config{Root: tree, Jobs: 4, NoCache: true, AST: ast.Options{MinLines: 1}},
		[]core.Engine{eng}, newTok(t), &report.Progress{})
	if err != nil {
		t.Fatal(err)
	}

	// A file whose single scored unit spans the whole fixture is the
	// fixture text; compare those.
	byFile := map[string][]report.Unit{}
	for _, u := range rep.Units {
		byFile[u.File] = append(byFile[u.File], u)
	}
	compared, worst := 0, 0.0
	for name, ref := range want {
		us := byFile[name]
		if len(us) != 1 || us[0].SplitOf > 0 {
			continue // extraction split the fixture into several units
		}
		lines := strings.Count(strings.TrimSpace(fx0(fx.Samples, name)), "\n") + 1
		if us[0].StartLine != 1 || us[0].EndLine != lines {
			continue // unit is a sub-span (e.g. a function inside the file)
		}
		if strings.TrimSpace(fx0(fx.Samples, name)) != fx0(fx.Samples, name) {
			continue
		}
		d := math.Abs(us[0].Severity - ref)
		worst = max(worst, d)
		// The report rounds scores to 4 decimals; the engine itself is
		// exact to ~1e-7 on these inputs (engine parity tests).
		if d > 5e-5+1e-6 {
			t.Errorf("%s: CLI severity %.5f, reference %.5f (|d| %.5f)", name, us[0].Severity, ref, d)
		}
		compared++
	}
	t.Logf("compared %d whole-fixture units end to end; max |d severity| %.2e", compared, worst)
	// Fixtures are in training form (the reference fixture generator strips them);
	// most are single definitions or whole files that extract to one unit.
	if compared < 20 {
		t.Fatalf("only %d fixtures compared end to end", compared)
	}
}

type fixture struct {
	Name, Language, Code string
	Severity             float64
}

func fx0(samples []fixture, file string) string {
	base := strings.TrimSuffix(file, filepath.Ext(file))
	for _, s := range samples {
		if s.Name == base {
			return s.Code
		}
	}
	return ""
}
