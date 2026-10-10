package results

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vaktex/vakt/internal/report"
)

func write(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestRoundTripFreshnessAndSignature(t *testing.T) {
	root := t.TempDir()
	key, err := Key(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	write(t, root, "a.py", "def a(): pass\n")
	write(t, root, "b.py", "def b(): pass\n")
	model := report.Model{Repo: "r", Revision: "x", Precision: "fp16"}
	units := []report.Unit{{File: "a.py", Name: "a", StartLine: 1, EndLine: 1, Severity: 0.9}, {File: "b.py", Name: "b", StartLine: 1, EndLine: 1, Severity: 0.2}}
	hashes := map[string]string{"a.py": Hash([]byte("def a(): pass\n")), "b.py": Hash([]byte("def b(): pass\n"))}
	if err := Update(root, key, model, ByFile(units, hashes), true); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(root, Dir, ".gitignore")); string(b) == "" {
		t.Error("no .gitignore in .vakt")
	}
	fresh := func(k []byte) int {
		d, err := Load(root, k)
		if err != nil {
			t.Fatal(err)
		}
		us, err := Fresh(root, d, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		return len(us)
	}
	if n := fresh(key); n != 2 {
		t.Fatalf("fresh units = %d, want 2", n)
	}
	write(t, root, "b.py", "def b(): return 1\n")
	if n := fresh(key); n != 1 {
		t.Errorf("after editing b.py: %d units, want 1", n)
	}
	other, _ := Key(t.TempDir())
	if n := fresh(other); n != 0 {
		t.Errorf("results signed with another key were read: %d units", n)
	}

	// A partial update keeps other files; a different model drops them.
	if err := Update(root, key, model, map[string]Entry{"c.py": {SHA256: "x"}}, false); err != nil {
		t.Fatal(err)
	}
	if d, _ := Load(root, key); len(d.Files) != 3 {
		t.Errorf("partial update: %d files, want 3", len(d.Files))
	}
	if err := Update(root, key, report.Model{Repo: "other"}, map[string]Entry{"c.py": {SHA256: "x"}}, false); err != nil {
		t.Fatal(err)
	}
	if d, _ := Load(root, key); len(d.Files) != 1 {
		t.Errorf("model change: %d files, want 1", len(d.Files))
	}
}

// A repository that ships .vakt as a symlink cannot make vakt write outside it.
func TestSymlinkedDirStaysInRoot(t *testing.T) {
	root, outside := t.TempDir(), t.TempDir()
	if err := os.Symlink(outside, filepath.Join(root, Dir)); err != nil {
		t.Skip(err)
	}
	key, _ := Key(t.TempDir())
	_ = Update(root, key, report.Model{}, map[string]Entry{"a.py": {SHA256: "x"}}, true)
	if ents, _ := os.ReadDir(outside); len(ents) != 0 {
		t.Fatalf("wrote outside the root: %v", ents)
	}
}

func TestCleanRel(t *testing.T) {
	for _, ok := range []string{"a.py", "src/a.py", "a/.b/c"} {
		if !CleanRel(ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"", ".", "..", "../a", "/a", "a/../b", "a//b", `a\b`, "./a"} {
		if CleanRel(bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}
