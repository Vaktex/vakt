package mcp

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/labels"
	"github.com/vaktex/vakt/internal/report"
)

// withDir adds "dir" to a JSON argument object.
func withDir(dir, args string) string {
	inner := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(strings.TrimSpace(args), "{"), "}"))
	d := `"dir":"` + escape(dir) + `"`
	if inner == "" {
		return "{" + d + "}"
	}
	return "{" + d + "," + inner + "}"
}

// call runs one tool by name and returns its result, failing the test if the
// tool errored.
func call(t *testing.T, d Deps, name, args string) any {
	t.Helper()
	out, err := callErr(t, d, name, args)
	if err != nil {
		t.Fatalf("%s(%s): %v", name, args, err)
	}
	return out
}

// callErr runs one tool and returns its error, for the paths that must fail.
func callErr(t *testing.T, d Deps, name, args string) (any, error) {
	t.Helper()
	for _, tl := range Tools(d) {
		if tl.Name == name {
			return tl.Handler(context.Background(), json.RawMessage(args))
		}
	}
	t.Fatalf("no tool named %q", name)
	return nil, nil
}

// asMap round-trips a result through JSON, which is what the agent actually
// sees, so the test checks the wire shape rather than the Go struct.
func asMap(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

// fixture writes a source file and a report describing it, and returns Deps
// and the tree's directory.
func fixture(t *testing.T) (Deps, string) {
	t.Helper()
	dir := t.TempDir()
	// t.TempDir is under /var on macOS, which is a symlink to /private/var;
	// resolve it so confinement compares like with like.
	if real, err := filepath.EvalSymlinks(dir); err == nil {
		dir = real
	}
	src := "package main\n" + // 1
		"\n" + // 2
		"func query(id string) {\n" + // 3
		"\tdb.Exec(\"SELECT * FROM t WHERE id = \" + id)\n" + // 4
		"}\n" + // 5
		"\n" + // 6
		"func fine() {}\n" // 7
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(src), 0o600); err != nil {
		t.Fatal(err)
	}
	rep := &report.Report{
		SchemaVersion: report.SchemaVersion,
		Scan:          report.Scan{Root: dir, Files: 1, Units: 2, Skipped: []report.Skip{}},
		Units: []report.Unit{
			unit("main.go", "query", 3, 5, 0.93, "data_neutralization"),
			unit("main.go", "fine", 7, 7, 0.61, "file_and_path"),
		},
		Files:   []report.File{{File: "main.go", Language: "go", MaxSeverity: 0.93, FlaggedUnits: 2, Units: 2}},
		Summary: report.Summary{Threshold: 0.5, FlaggedUnits: 2, FlaggedFiles: 1},
	}
	path := filepath.Join(dir, "vakt-report.json")
	if err := report.WriteFile(path, rep); err != nil {
		t.Fatal(err)
	}
	return Deps{}, dir
}

func unit(file, name string, start, end int, sev float64, family string) report.Unit {
	u := report.Unit{
		File: file, Language: "go", Kind: "function", Name: name,
		StartLine: start, EndLine: end, Tokens: 40,
		Severity: sev, TopFamily: family, TopFamilyProb: sev, Flagged: true,
	}
	u.Families[labels.Index(family)] = sev
	// A second, lower family, so other_families has something to report.
	u.Families[labels.Index("web_security")] = 0.21
	return u
}

func TestFindingsListsWorstFirstWithNumbers(t *testing.T) {
	d, dir := fixture(t)
	m := asMap(t, call(t, d, "findings", withDir(dir, `{}`)))
	if got := m["total"].(float64); got != 2 {
		t.Fatalf("total = %v, want 2", got)
	}
	list := m["findings"].([]any)
	first := list[0].(map[string]any)
	if first["number"].(float64) != 1 {
		t.Errorf("first finding is number %v, want 1", first["number"])
	}
	if first["severity"].(float64) != 0.93 {
		t.Errorf("findings are not worst-first: %v", first["severity"])
	}
	// The CWE and a readable issue name must be present: they are why the
	// agent does not have to look the family up.
	if first["cwe"] != "CWE-89" || first["issue"] != "SQL injection" {
		t.Errorf("issue/cwe = %v/%v, want SQL injection/CWE-89", first["issue"], first["cwe"])
	}
	if first["severity_label"] != "CRITICAL" {
		t.Errorf("label = %v, want CRITICAL", first["severity_label"])
	}
	if first["lines"] != "3-5" {
		t.Errorf("lines = %v, want 3-5", first["lines"])
	}
}

func TestFindingsFiltersByThresholdFamilyAndFile(t *testing.T) {
	d, dir := fixture(t)

	m := asMap(t, call(t, d, "findings", withDir(dir, `{"threshold":0.9}`)))
	if m["total"].(float64) != 1 {
		t.Errorf("threshold 0.9: total = %v, want 1", m["total"])
	}

	m = asMap(t, call(t, d, "findings", withDir(dir, `{"family":["file_and_path"]}`)))
	if m["total"].(float64) != 1 {
		t.Errorf("family filter: total = %v, want 1", m["total"])
	}

	// A file suffix is accepted, because the agent usually has the path it is
	// reviewing, not the scan-root-relative one.
	for _, f := range []string{"main.go", "./main.go"} {
		m = asMap(t, call(t, d, "findings", withDir(dir, `{"file":"`+strings.TrimPrefix(f, "./")+`"}`)))
		if m["total"].(float64) != 2 {
			t.Errorf("file %q: total = %v, want 2", f, m["total"])
		}
	}
	m = asMap(t, call(t, d, "findings", withDir(dir, `{"file":"nope.go"}`)))
	if m["total"].(float64) != 0 {
		t.Errorf("unknown file: total = %v, want 0", m["total"])
	}
	if !strings.Contains(m["note"].(string), "No findings match") {
		t.Errorf("empty result has no guidance: %v", m["note"])
	}
}

func TestFindingsPaginates(t *testing.T) {
	d, dir := fixture(t)
	m := asMap(t, call(t, d, "findings", withDir(dir, `{"limit":1}`)))
	if n := len(m["findings"].([]any)); n != 1 {
		t.Fatalf("limit 1 returned %d", n)
	}
	// The note must tell the agent how to get the rest, or it will assume the
	// list is complete.
	if !strings.Contains(m["note"].(string), "offset=1") {
		t.Errorf("note does not offer the next page: %v", m["note"])
	}
	m = asMap(t, call(t, d, "findings", withDir(dir, `{"limit":1,"offset":1}`)))
	got := m["findings"].([]any)[0].(map[string]any)
	if got["number"].(float64) != 2 {
		t.Errorf("offset 1 gave number %v, want 2", got["number"])
	}
}

// The number from findings must resolve to the same unit in finding, for the
// same filters. If these ever disagree the agent reviews the wrong function.
func TestFindingNumbersAgreeWithFindings(t *testing.T) {
	d, dir := fixture(t)
	for _, args := range []string{`{}`, `{"threshold":0.9}`, `{"family":["file_and_path"]}`} {
		list := asMap(t, call(t, d, "findings", withDir(dir, args)))["findings"].([]any)
		for _, row := range list {
			b := row.(map[string]any)
			// Re-ask for that number with the same filters.
			detailArgs := `{"number":` + strconv.Itoa(int(b["number"].(float64)))
			if args != `{}` {
				detailArgs += "," + strings.Trim(args, "{}")
			}
			detailArgs += "}"
			det := asMap(t, call(t, d, "finding", withDir(dir, detailArgs)))
			if det["file"] != b["file"] || det["family"] != b["family"] {
				t.Errorf("%s number %v: findings said %v/%v, finding said %v/%v",
					args, b["number"], b["file"], b["family"], det["file"], det["family"])
			}
		}
	}
}

func TestFindingReturnsSourceAndCWE(t *testing.T) {
	d, dir := fixture(t)
	m := asMap(t, call(t, d, "finding", withDir(dir, `{"number":1}`)))
	if m["cwe"] != "CWE-89" {
		t.Errorf("cwe = %v", m["cwe"])
	}
	code, _ := m["code"].(string)
	if !strings.Contains(code, "SELECT * FROM t") {
		t.Fatalf("code does not contain the flagged line: %q", code)
	}
	// With 3 lines of context, a function at line 3 starts the snippet at 1.
	if m["code_start_line"].(float64) != 1 {
		t.Errorf("code_start_line = %v, want 1", m["code_start_line"])
	}
	if _, ok := m["other_families"]; !ok {
		t.Error("other_families missing; a borderline finding cannot be judged")
	}
	// The note must say a score is not a proof, so the agent reports it as a
	// lead rather than a confirmed vulnerability.
	if note := m["note"].(string); !strings.Contains(note, "not a proven bug") {
		t.Errorf("note = %q", note)
	}
}

func TestFindingContextWidensTheSnippet(t *testing.T) {
	d, dir := fixture(t)
	narrow := asMap(t, call(t, d, "finding", withDir(dir, `{"number":1,"context":0}`)))
	if narrow["code_start_line"].(float64) != 3 {
		t.Errorf("context 0: code_start_line = %v, want 3", narrow["code_start_line"])
	}
	if strings.Contains(narrow["code"].(string), "package main") {
		t.Error("context 0 leaked surrounding lines")
	}
}

func TestFindingOutOfRange(t *testing.T) {
	d, dir := fixture(t)
	_, err := callErr(t, d, "finding", withDir(dir, `{"number":99}`))
	if err == nil {
		t.Fatal("finding 99 succeeded")
	}
	// The message must say how many there are, so the agent can recover
	// without another call.
	if !strings.Contains(err.Error(), "2 findings match") {
		t.Errorf("error = %q", err)
	}
}

func TestFamiliesListsTheTaxonomy(t *testing.T) {
	m := asMap(t, call(t, Deps{}, "families", `{}`))
	fams := m["families"].([]any)
	if len(fams) != len(labels.Families) {
		t.Fatalf("got %d families, want %d", len(fams), len(labels.Families))
	}
	for _, f := range fams {
		e := f.(map[string]any)
		if e["family"] == "" || e["issue"] == "" {
			t.Errorf("incomplete entry: %v", e)
		}
	}
}

// Bad arguments must be reported, not ignored: a silently dropped typo means
// the agent believes it filtered when it did not.
func TestToolsRejectBadArguments(t *testing.T) {
	d, dir := fixture(t)
	cases := []struct{ tool, args, want string }{
		{"findings", `{"treshold":0.9}`, "unknown field"},
		{"findings", `{"threshold":2}`, "threshold must be in [0, 1]"},
		{"findings", `{"family":["not_a_family"]}`, "unknown family"},
		{"findings", `{"limit":9999}`, "limit must be in"},
		{"findings", `{"offset":-1}`, "offset must not be negative"},
		{"finding", `{"number":0}`, "number must be at least 1"},
		{"finding", `{"number":1,"context":999}`, "context must be in [0, 50]"},
		{"scan", `{"threshold":0}`, "threshold must be in (0, 1]"},
		{"scan", `{"include":["["]}`, "not a valid glob"},
	}
	for _, c := range cases {
		_, err := callErr(t, d, c.tool, withDir(dir, c.args))
		if err == nil {
			t.Errorf("%s(%s): no error, want %q", c.tool, c.args, c.want)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s(%s): error = %q, want it to mention %q", c.tool, c.args, err, c.want)
		}
	}
	// A family error should point at the tool that lists valid names.
	_, err := callErr(t, d, "findings", withDir(dir, `{"family":["nope"]}`))
	if !strings.Contains(err.Error(), "families") {
		t.Errorf("family error does not mention the families tool: %q", err)
	}
}

// The server must not become a way to read arbitrary files. Paths from the
// agent and from the report are both untrusted.
func TestPathsAreConfinedToTheWorkingDirectory(t *testing.T) {
	d, dir := fixture(t)
	outside := filepath.Join(filepath.Dir(dir), "outside.json")
	if err := os.WriteFile(outside, []byte(`{"schema_version":"1"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		`{"report":"../outside.json"}`,
		`{"report":"` + escape(outside) + `"}`,
		`{"report":"../../etc/hosts"}`,
	} {
		_, err := callErr(t, d, "findings", withDir(dir, bad))
		if err == nil {
			t.Errorf("findings%s was allowed to read outside the working directory", bad)
			continue
		}
		if !strings.Contains(err.Error(), "outside the working directory") {
			t.Errorf("findings%s: error = %q", bad, err)
		}
	}
	_, err := callErr(t, d, "scan", withDir(dir, `{"path":".."}`))
	if err == nil || !strings.Contains(err.Error(), "outside the working directory") {
		t.Errorf("scan of .. : error = %v, want a confinement refusal", err)
	}
}

// A symlink out of the tree must not smuggle a read past confinement.
func TestSymlinkEscapeIsRefused(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks need privilege on Windows")
	}
	d, dir := fixture(t)
	secret := filepath.Join(t.TempDir(), "secret.json")
	if err := os.WriteFile(secret, []byte(`{"schema_version":"1"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.json")
	if err := os.Symlink(secret, link); err != nil {
		t.Skipf("symlink: %v", err)
	}
	_, err := callErr(t, d, "findings", withDir(dir, `{"report":"link.json"}`))
	if err == nil || !strings.Contains(err.Error(), "outside the working directory") {
		t.Errorf("a symlink out of the tree was followed: %v", err)
	}
}

// confine is the whole security boundary, so it is tested directly as well as
// through the tools. The case that matters most is a symlink out of the tree
// with components that do not exist yet: resolving the path lexically would
// make it look contained, because Clean cannot know `esc` is a link.
func TestConfineResolvesSymlinksUnderMissingComponents(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlinks need privilege on Windows")
	}
	work := evalOrSkip(t, t.TempDir())
	outside := evalOrSkip(t, t.TempDir())
	if err := os.WriteFile(filepath.Join(outside, "secret"), []byte("s"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(work, "esc")); err != nil {
		t.Skipf("symlink: %v", err)
	}
	refuse := []string{
		filepath.Join(work, "esc", "secret"),            // exists through the link
		filepath.Join(work, "esc", "new.json"),          // one missing component
		filepath.Join(work, "esc", "brand", "new.json"), // several missing components
		filepath.Join(work, "esc", "a", "b", "c", "d.json"),
		filepath.Join(work, "..", filepath.Base(outside)), // plain traversal
		filepath.Join(work, "a", "..", "..", "etc", "hosts"),
		outside,
		"/etc/hosts",
	}
	for _, p := range refuse {
		if got, err := confine(work, p); err == nil {
			t.Errorf("confine(%q) allowed %q; it escapes the working directory", p, got)
		}
	}
	// Legitimate paths, including files not yet written, must still pass.
	allow := []string{
		work,
		filepath.Join(work, "vakt-report.json"),
		filepath.Join(work, "sub", "dir", "later.json"),
	}
	for _, p := range allow {
		if _, err := confine(work, p); err != nil {
			t.Errorf("confine(%q) refused a legitimate path: %v", p, err)
		}
	}
}

func evalOrSkip(t *testing.T, dir string) string {
	t.Helper()
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Skipf("EvalSymlinks(%q): %v", dir, err)
	}
	return real
}

// A report whose unit names a path outside the tree must not produce a
// snippet. The report is user data, not trusted input.
func TestSnippetFromAHostileReportPathIsRefused(t *testing.T) {
	d, dir := fixture(t)
	rep := &report.Report{
		SchemaVersion: report.SchemaVersion,
		Scan:          report.Scan{Root: dir},
		Units:         []report.Unit{unit("../../../../etc/passwd", "x", 1, 3, 0.9, "file_and_path")},
		Summary:       report.Summary{Threshold: 0.5, FlaggedUnits: 1},
	}
	path := filepath.Join(dir, "hostile.json")
	if err := report.WriteFile(path, rep); err != nil {
		t.Fatal(err)
	}
	m := asMap(t, call(t, d, "finding", withDir(dir, `{"number":1,"report":"hostile.json"}`)))
	if code, _ := m["code"].(string); code != "" {
		t.Errorf("read a file outside the tree: %q", code)
	}
	if note, _ := m["code_note"].(string); !strings.Contains(note, "outside the working directory") {
		t.Errorf("code_note = %q, want a confinement explanation", note)
	}
}

func TestMissingReportTellsTheAgentToScan(t *testing.T) {
	dir := t.TempDir()
	_, err := callErr(t, Deps{}, "findings", withDir(dir, `{}`))
	if err == nil || !strings.Contains(err.Error(), "call the `scan` tool first") {
		t.Errorf("error = %v, want a pointer to the scan tool", err)
	}
}

// Without a native engine the scan tool must explain that, not panic or
// return an empty report.
func TestScanWithoutAnEngineExplainsItself(t *testing.T) {
	d, dir := fixture(t)
	d.Scan = nil
	_, err := callErr(t, d, "scan", withDir(dir, `{}`))
	if err == nil || !strings.Contains(err.Error(), "no scanning engine") {
		t.Errorf("error = %v, want an explanation that no engine is linked", err)
	}
}

func TestScanSummaryGuidesTheAgent(t *testing.T) {
	d, dir := fixture(t)
	d.Scan = func(_ context.Context, req ScanRequest) (*report.Report, error) {
		return &report.Report{
			SchemaVersion: report.SchemaVersion,
			Scan:          report.Scan{Root: req.Root, Files: 1, Units: 2},
			Units:         []report.Unit{unit("main.go", "query", 3, 5, 0.93, "data_neutralization")},
			Summary:       report.Summary{Threshold: req.Threshold, FlaggedUnits: 1, FlaggedFiles: 1},
		}, nil
	}
	m := asMap(t, call(t, d, "scan", withDir(dir, `{"threshold":0.7}`)))
	if m["threshold"].(float64) != 0.7 {
		t.Errorf("threshold not passed through: %v", m["threshold"])
	}
	if m["flagged"].(float64) != 1 {
		t.Errorf("flagged = %v", m["flagged"])
	}
	// The summary carries the worst findings, so a single call often answers
	// the question without a follow-up.
	if len(m["top_findings"].([]any)) != 1 {
		t.Errorf("top_findings = %v", m["top_findings"])
	}
	if !strings.Contains(m["note"].(string), "lead to verify") {
		t.Errorf("note = %v", m["note"])
	}

	// A clean scan must not read as a safety guarantee.
	d.Scan = func(_ context.Context, req ScanRequest) (*report.Report, error) {
		return &report.Report{
			SchemaVersion: report.SchemaVersion,
			Scan:          report.Scan{Root: req.Root, Files: 3, Units: 9},
			Summary:       report.Summary{Threshold: req.Threshold},
		}, nil
	}
	m = asMap(t, call(t, d, "scan", withDir(dir, `{}`)))
	if note := m["note"].(string); !strings.Contains(note, "not a proof of safety") {
		t.Errorf("clean note = %q", note)
	}
}

// scan must persist its report where the read-only tools look, or the agent's
// next call fails.
func TestScanWritesTheReportTheOtherToolsRead(t *testing.T) {
	dir := t.TempDir()
	if real, err := filepath.EvalSymlinks(dir); err == nil {
		dir = real
	}
	reportPath := filepath.Join(dir, "vakt-report.json")
	d := Deps{}
	d.Scan = func(_ context.Context, req ScanRequest) (*report.Report, error) {
		if req.Out != reportPath {
			t.Errorf("scan asked to write %q, want %q", req.Out, reportPath)
		}
		rep := &report.Report{
			SchemaVersion: report.SchemaVersion,
			Scan:          report.Scan{Root: req.Root},
			Units:         []report.Unit{unit("main.go", "q", 1, 2, 0.9, "data_neutralization")},
			Summary:       report.Summary{Threshold: req.Threshold, FlaggedUnits: 1},
		}
		// The CLI writes the file; mimic that here.
		if err := report.WriteFile(req.Out, rep); err != nil {
			t.Fatal(err)
		}
		return rep, nil
	}
	if m := asMap(t, call(t, d, "scan", withDir(dir, `{}`))); m["report_path"] != reportPath {
		t.Errorf("report_path = %v, want %v", m["report_path"], reportPath)
	}
	if m := asMap(t, call(t, d, "findings", withDir(dir, `{}`))); m["total"].(float64) != 1 {
		t.Errorf("findings after scan: total = %v, want 1", m["total"])
	}
}

// escape makes a path safe to embed in a JSON string literal in these tests.
func escape(s string) string {
	b, _ := json.Marshal(s)
	return string(b[1 : len(b)-1])
}

// Every file tool needs an absolute, existing dir: without one the agent
// would be scanning wherever the server happened to start.
func TestDirIsRequiredAndAbsolute(t *testing.T) {
	_, dir := fixture(t)
	for _, tool := range []string{"scan", "findings"} {
		if _, err := callErr(t, Deps{}, tool, `{}`); err == nil || !strings.Contains(err.Error(), "dir is required") {
			t.Errorf("%s without dir: %v", tool, err)
		}
		if _, err := callErr(t, Deps{}, tool, `{"dir":"relative/path"}`); err == nil || !strings.Contains(err.Error(), "absolute") {
			t.Errorf("%s with a relative dir: %v", tool, err)
		}
		if _, err := callErr(t, Deps{}, tool, withDir(filepath.Join(dir, "missing"), `{}`)); err == nil {
			t.Errorf("%s with a missing dir succeeded", tool)
		}
	}
}

// With no root, any directory works; with one, only directories under it.
func TestRootLimitsWhichDirsTheAgentMayUse(t *testing.T) {
	_, root := fixture(t)
	inside := root // the root itself is allowed
	outside := evalOrSkip(t, t.TempDir())

	if _, err := callErr(t, Deps{}, "findings", withDir(inside, `{}`)); err != nil {
		t.Fatalf("no root: %v", err)
	}
	d := Deps{Root: root, Scan: func(_ context.Context, req ScanRequest) (*report.Report, error) {
		return &report.Report{SchemaVersion: report.SchemaVersion, Scan: report.Scan{Root: req.Root}}, nil
	}}
	if _, err := callErr(t, d, "findings", withDir(inside, `{}`)); err != nil {
		t.Errorf("dir under root refused: %v", err)
	}
	for _, bad := range []string{outside, filepath.Join(root, "..")} {
		_, err := callErr(t, d, "scan", withDir(bad, `{}`))
		if err == nil || !strings.Contains(err.Error(), "only tree this server was started to allow") {
			t.Errorf("dir %s outside root: %v", bad, err)
		}
	}
	// A symlink under root pointing out of it must not get the agent out.
	link := filepath.Join(inside, "escape")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlink: %v", err)
	}
	if _, err := callErr(t, d, "scan", withDir(link, `{}`)); err == nil {
		t.Error("a symlink under root carried dir outside it")
	}
}
