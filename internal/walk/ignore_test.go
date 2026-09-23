package walk

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestWildmatch(t *testing.T) {
	cases := []struct {
		pat, name string
		want      bool
	}{
		{"*.go", "main.go", true},
		{"*.go", "main.gox", false},
		{"a?c", "abc", true},
		{"a?c", "ac", false},
		{"[abc]x", "bx", true},
		{"[!abc]x", "bx", false},
		{"[^abc]x", "dx", true},
		{"[a-c]x", "cx", true},
		{"[a-c]x", "dx", false},
		{"[]]", "]", true},
		{"[!]]", "a", true},
		{"[[:digit:]]*", "9lives", true},
		{"[[:upper:]]", "a", false},
		{"[abc", "a", false}, // malformed: matches nothing
		{`\*`, "*", true},
		{`\*`, "a", false},
		{`\#x`, "#x", true},
		{`a\?`, "a?", true},
		{`a\?`, "ab", false},
		{"*a*b*c*", "xaxbxcx", true},
		{"*a*b*c*", "xaxcxbx", false},
		{"**", "anything", true},
		{"é*", "éclair", true},
		{"?", "é", true},
		{"foo", "foo", true},
		{"foo", "Foo", false},
		{"", "", true},
	}
	for _, c := range cases {
		if got := wildmatch(c.pat, c.name); got != c.want {
			t.Errorf("wildmatch(%q, %q) = %v, want %v", c.pat, c.name, got, c.want)
		}
	}
}

func TestWildmatchPathological(t *testing.T) {
	pat := strings.Repeat("*a", 50) + "b"
	name := strings.Repeat("a", 5000)
	if wildmatch(pat, name) {
		t.Fatal("unexpected match")
	}
	segs := strings.Split(strings.Repeat("**/a/", 40)+"b", "/")
	path := strings.Split(strings.Repeat("a/", 2000)+"c", "/")
	if matchSegments(segs, path) {
		t.Fatal("unexpected segment match")
	}
}

type probe struct {
	path  string
	isDir bool
	want  bool
}

func TestIgnoreChain(t *testing.T) {
	root := parseIgnore("", []byte(strings.Join([]string{
		"# comment",
		"",
		"*.log",
		"!keep.log",
		"/rootonly.txt",
		"build/",
		"docs/**/*.tmp",
		"**/gen/*.go",
		"a/**/z",
		"trailing/**",
		`\!bang`,
		`\#hash`,
		`space\ `,
		"tabs   ",
		"deep/nested/file.txt",
		"secret/",
		"!secret/public.txt", // cannot re-include inside an excluded dir (dir pruned)
	}, "\n")))
	sub := parseIgnore("pkg", []byte("/local.txt\n!*.log\nonly-here\n*.gen\n!keep.gen\n"))
	var c *ignoreChain
	c = c.push(root).push(sub)

	probes := []probe{
		{"x.log", false, true},
		{"a/b/x.log", false, true},
		{"keep.log", false, false},
		{"rootonly.txt", false, true},
		{"sub/rootonly.txt", false, false},
		{"build", true, true},
		{"build", false, false}, // dir-only pattern does not match a file
		{"src/build", true, true},
		{"docs/x.tmp", false, true},
		{"docs/a/b/x.tmp", false, true},
		{"other/docs/x.tmp", false, false},
		{"gen/x.go", false, true},
		{"a/b/gen/x.go", false, true},
		{"a/z", false, true},
		{"a/b/c/z", false, true},
		{"b/a/z", false, false},
		{"trailing", true, false},
		{"trailing/x", false, true},
		{"!bang", false, true},
		{"#hash", false, true},
		{"space ", false, true},
		{"space", false, false},
		{"tabs", false, true},
		{"deep/nested/file.txt", false, true},
		{"x/deep/nested/file.txt", false, false},
		{"secret", true, true},
		// nested file: relative to pkg/
		{"pkg/local.txt", false, true},
		{"pkg/sub/local.txt", false, false},
		{"local.txt", false, false},
		{"pkg/x.log", false, false}, // re-included by deeper !*.log
		{"pkg/a/x.log", false, false},
		{"only-here", false, false},
		{"pkg/q/only-here", false, true},
		{"pkg/x.gen", false, true},
		{"pkg/keep.gen", false, false},
		{"pkgx/x.gen", false, false},
	}
	for _, p := range probes {
		if got := c.ignored(p.path, p.isDir); got != p.want {
			t.Errorf("ignored(%q, dir=%v) = %v, want %v", p.path, p.isDir, got, p.want)
		}
	}
}

func TestParsePatternEdgeCases(t *testing.T) {
	for _, line := range []string{"", "#x", "   ", "/", "!", "!/", "//"} {
		if _, ok := parsePattern(line); ok {
			t.Errorf("parsePattern(%q) should be empty", line)
		}
	}
	p, ok := parsePattern("foo/")
	if !ok || !p.dirOnly || p.anchored {
		t.Errorf("foo/ => %+v", p)
	}
	p, _ = parsePattern("/foo")
	if !p.anchored {
		t.Errorf("/foo should be anchored")
	}
	p, _ = parsePattern("a/**/**/b")
	if len(p.segs) != 3 {
		t.Errorf("collapsed ** segs = %v", p.segs)
	}
	f := parseIgnore("", []byte("\xef\xbb\xbfa.txt\r\nb.txt\r\n"))
	if len(f.pats) != 2 || f.pats[0].segs[0] != "a.txt" || f.pats[1].segs[0] != "b.txt" {
		t.Errorf("BOM/CRLF handling: %+v", f.pats)
	}
}

// TestIgnoreMatchesGit compares the matcher with `git check-ignore` on a
// real repository with nested ignore files.
func TestIgnoreMatchesGit(t *testing.T) {
	git, err := exec.LookPath("git")
	if err != nil {
		t.Skip("git not installed")
	}
	dir := t.TempDir()
	run := func(args ...string) []byte {
		cmd := exec.Command(git, args...) // #nosec G204 -- test helper
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GIT_CONFIG_NOSYSTEM=1", "HOME="+dir)
		out, _ := cmd.CombinedOutput()
		return out
	}
	run("init", "-q")
	write := func(rel, data string) {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	rootIgn := "*.log\n!keep.log\n/top.txt\nout/\ndocs/**/*.tmp\n**/gen/*.go\na/**/z\n\\!bang\n*.[oa]\n!lib.a\nnested/*.c\n"
	subIgn := "/local.txt\n!*.log\n*.gen\n!keep.gen\nsub/**\n"
	write(".gitignore", rootIgn)
	write("pkg/.gitignore", subIgn)
	write(".git/info/exclude", "excluded.md\n")

	paths := []string{
		"x.log", "a/b/x.log", "keep.log", "top.txt", "d/top.txt", "out/f.c", "d/out/f.c",
		"docs/x.tmp", "docs/a/b/x.tmp", "o/docs/x.tmp", "gen/x.go", "a/b/gen/x.go",
		"a/z", "a/b/c/z", "b/a/z", "!bang", "x.o", "x.a", "lib.a", "nested/f.c",
		"nested/d/f.c", "d/nested/f.c", "pkg/local.txt", "pkg/s/local.txt", "local.txt",
		"pkg/x.log", "pkg/x.gen", "pkg/keep.gen", "pkg/sub/f", "pkg/sub/d/f",
		"excluded.md", "d/excluded.md",
	}
	var c *ignoreChain
	c = c.push(parseIgnore("", []byte("excluded.md\n")))
	c = c.push(parseIgnore("", []byte(rootIgn)))
	c = c.push(parseIgnore("pkg", []byte(subIgn)))

	for _, p := range paths {
		write(p, "x")
	}
	out := run(append([]string{"check-ignore", "--no-index", "--"}, paths...)...)
	gitIgnored := map[string]bool{}
	for line := range bytes.SplitSeq(bytes.TrimSpace(out), []byte("\n")) {
		gitIgnored[string(line)] = true
	}
	for _, p := range paths {
		if got := ignoredWithParents(c, p); got != gitIgnored[p] {
			t.Errorf("%s: ours=%v git=%v", p, got, gitIgnored[p])
		}
	}
}

// ignoredWithParents mimics the walker, which never descends into an
// ignored directory.
func ignoredWithParents(c *ignoreChain, p string) bool {
	parts := strings.Split(p, "/")
	for i := 1; i < len(parts); i++ {
		if c.ignored(strings.Join(parts[:i], "/"), true) {
			return true
		}
	}
	return c.ignored(p, false)
}
