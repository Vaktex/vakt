package report

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/labels"
)

var update = flag.Bool("update", false, "rewrite golden files")

func scores(sev float32, fam map[string]float32) core.Scores {
	var s core.Scores
	s.Severity = sev
	for i := range s.Families {
		s.Families[i] = 0.01
	}
	for k, v := range fam {
		s.Families[labels.Index(k)] = v
	}
	return s
}

func fixture() (Meta, []Result) {
	meta := Meta{
		Root:          "/src/app",
		StartedAt:     time.Date(2026, 9, 23, 6, 0, 0, 0, time.UTC),
		Duration:      2500 * time.Millisecond,
		Files:         5,
		CacheHits:     1,
		Skipped:       []Skip{{File: "assets/app.min.js", Reason: "minified"}},
		ModelRepo:     brand.ModelRepo,
		ModelRevision: "main",
		ModelSHA:      strings.Repeat("ab", 32),
		Backend:       "metal",
		Device:        "Apple M5 Pro",
		Precision:     "fp32",
	}
	u := func(file, lang, kind, name string, s, e int) core.Unit {
		return core.Unit{File: file, Language: lang, Kind: kind, Name: name, StartLine: s, EndLine: e}
	}
	part := func(p, s, e int) core.Unit {
		x := u("src/parse.c", "C", core.KindFunction, "parse_header", s, e)
		x.SplitPart, x.SplitOf, x.ParentStartLine, x.ParentEndLine = p, 2, 10, 400
		return x
	}
	res := []Result{
		{Unit: u("src/net.c", "C", core.KindFunction, "read_packet", 12, 48), Tokens: 900, Scores: scores(0.93, map[string]float32{"memory_safety": 0.91})},
		{Unit: u("src/net.c", "C", core.KindFunction, "checksum", 50, 70), Tokens: 300, Scores: scores(0.12, nil), Cached: true},
		{Unit: u("app/views.py", "Python", core.KindFunction, "search", 5, 30), Tokens: 400, Scores: scores(0.78, map[string]float32{"data_neutralization": 0.82})},
		{Unit: u("app/views.py", "Python", core.KindMethod, "Admin.delete", 40, 60), Tokens: 350, Scores: scores(0.61, map[string]float32{"authorization": 0.7})},
		{Unit: u("app/util.py", "Python", core.KindResidual, "", 1, 20), Tokens: 120, Scores: scores(0.2, nil)},
		{Unit: part(1, 10, 200), Tokens: 16000, Scores: scores(0.55, map[string]float32{"memory_safety": 0.4, "file_and_path": 0.6})},
		{Unit: part(2, 201, 400), Tokens: 15000, Scores: scores(0.88, map[string]float32{"memory_safety": 0.9}), Truncated: true},
	}
	return meta, res
}

func TestBuildSortingAndGrouping(t *testing.T) {
	meta, res := fixture()
	r := Build(meta, res, 0.5)
	if len(r.Units) != 6 {
		t.Fatalf("units = %d, want 6 (two parts grouped)", len(r.Units))
	}
	want := []string{"read_packet", "parse_header", "search", "Admin.delete", "", "checksum"}
	for i, w := range want {
		if r.Units[i].Name != w {
			t.Errorf("unit %d = %q, want %q", i, r.Units[i].Name, w)
		}
	}
	p := r.Units[1]
	if p.StartLine != 10 || p.EndLine != 400 || p.SplitOf != 2 || len(p.Parts) != 2 {
		t.Fatalf("parent = %+v", p)
	}
	if p.Severity != 0.88 || p.Tokens != 31000 || !p.Truncated {
		t.Errorf("parent severity/tokens/truncated = %v/%v/%v", p.Severity, p.Tokens, p.Truncated)
	}
	if got := p.Families[labels.Index("file_and_path")]; got != 0.6 {
		t.Errorf("parent file_and_path = %v, want element-wise max 0.6", got)
	}
	if p.TopFamily != "memory_safety" || p.TopFamilyProb != 0.9 {
		t.Errorf("parent top = %s %v", p.TopFamily, p.TopFamilyProb)
	}
	if r.Summary.FlaggedUnits != 4 || r.Summary.FlaggedFiles != 3 {
		t.Errorf("summary = %+v", r.Summary)
	}
	if r.Summary.ByFamily[labels.Index("memory_safety")] != 2 {
		t.Errorf("by_family = %v", r.Summary.ByFamily)
	}
	if r.Files[0].File != "src/net.c" || r.Files[len(r.Files)-1].File != "app/util.py" {
		t.Errorf("file order = %+v", r.Files)
	}
	if r.Scan.Tokens != 33070 || r.Scan.Units != 6 || r.Scan.TokensPerSec != 13228 {
		t.Errorf("scan = %+v", r.Scan)
	}
	if r.Model.Backend != "metal" || r.SchemaVersion != "1" {
		t.Errorf("model = %+v", r.Model)
	}
}

func TestFilter(t *testing.T) {
	meta, res := fixture()
	r := Build(meta, res, 0.5)
	f := Filter(r, 0.8, []string{"memory_safety"})
	if len(f.Units) != 2 || f.Summary.FlaggedUnits != 2 || f.Summary.FlaggedFiles != 2 || f.Summary.Threshold != 0.8 {
		t.Fatalf("filtered = %+v", f.Summary)
	}
	if len(r.Units) != 6 || r.Summary.Threshold != 0.5 {
		t.Error("Filter mutated its input")
	}
	g := Filter(r, 0.9, nil)
	if g.Summary.FlaggedUnits != 1 || len(g.Units) != 6 {
		t.Errorf("threshold only: %+v", g.Summary)
	}
	if MaxSeverity(r) != 0.93 {
		t.Errorf("MaxSeverity = %v", MaxSeverity(r))
	}
	if ValidateFamilies([]string{"memory_safety"}) != nil || ValidateFamilies([]string{"nope"}) == nil {
		t.Error("ValidateFamilies")
	}
}

func TestBuildEmpty(t *testing.T) {
	r := Build(Meta{}, nil, 0.5)
	var b bytes.Buffer
	if err := WriteJSON(&b, r); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{`"units": []`, `"files": []`, `"by_family": {}`, `"skipped": []`} {
		if !strings.Contains(b.String(), k) {
			t.Errorf("empty report missing %s:\n%s", k, b.String())
		}
	}
}

func TestJSONRoundTripAndOrder(t *testing.T) {
	meta, res := fixture()
	r := Build(meta, res, 0.5)
	var b bytes.Buffer
	if err := WriteJSON(&b, r); err != nil {
		t.Fatal(err)
	}
	if !json.Valid(b.Bytes()) {
		t.Fatal("invalid JSON")
	}
	// Families are in head order.
	s := b.String()
	last := -1
	for _, f := range labels.Families {
		i := strings.Index(s, `"`+f+`"`)
		if i < 0 || i < last {
			t.Fatalf("family %s out of order", f)
		}
		last = i
	}
	got, err := ReadJSON(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	var b2 bytes.Buffer
	if err := WriteJSON(&b2, got); err != nil {
		t.Fatal(err)
	}
	if b.String() != b2.String() {
		t.Fatalf("round trip differs:\n%s\n---\n%s", b.String(), b2.String())
	}
	checkGolden(t, "report.json", b.Bytes())
}

func TestFloatsRounded(t *testing.T) {
	r := Build(Meta{}, []Result{{Unit: core.Unit{File: "a"}, Scores: core.Scores{Severity: 0.123456789}}}, 0.5)
	var b bytes.Buffer
	_ = WriteJSON(&b, r)
	if !strings.Contains(b.String(), `"severity": 0.1235`) {
		t.Fatalf("severity not rounded:\n%s", b.String())
	}
}

func TestReadJSONRejects(t *testing.T) {
	cases := map[string]string{
		"missing":    `{"units":[]}`,
		"wrong":      `{"schema_version":"2"}`,
		"garbage":    `not json`,
		"bad family": `{"schema_version":"1","units":[{"families":{"nope":1}}]}`,
	}
	for name, in := range cases {
		if _, err := ReadJSON(strings.NewReader(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	old := readLimit
	readLimit = 64
	defer func() { readLimit = old }()
	big := `{"schema_version":"1","x":"` + strings.Repeat("a", 100) + `"}`
	if _, err := ReadJSON(strings.NewReader(big)); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Errorf("oversize accepted: %v", err)
	}
}

func TestWriteFileAtomic0600(t *testing.T) {
	meta, res := fixture()
	path := filepath.Join(t.TempDir(), "out.json")
	if err := WriteFile(path, Build(meta, res, 0.5)); err != nil {
		t.Fatal(err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if st.Mode().Perm() != 0o600 {
		t.Errorf("mode = %v", st.Mode().Perm())
	}
	ents, _ := os.ReadDir(filepath.Dir(path))
	if len(ents) != 1 {
		t.Errorf("temp files left: %v", ents)
	}
	f, _ := os.Open(path) // #nosec G304 -- test path
	defer f.Close()
	if _, err := ReadJSON(f); err != nil {
		t.Fatal(err)
	}
}

func TestPrettyGolden(t *testing.T) {
	meta, res := fixture()
	r := Build(meta, res, 0.5)
	cases := map[string]PrettyOptions{
		"pretty.txt":        {Width: 100},
		"pretty_narrow.txt": {Width: 60, Top: 2},
		"pretty_quiet.txt":  {Quiet: true},
		"pretty_family.txt": {Width: 100, Families: []string{"memory_safety"}},
		"pretty_clean.txt":  {Width: 100, Threshold: 0.95},
	}
	for name, o := range cases {
		var b bytes.Buffer
		if err := Pretty(&b, r, o); err != nil {
			t.Fatal(err)
		}
		if strings.Contains(b.String(), "\x1b") {
			t.Errorf("%s: colour escapes in plain output", name)
		}
		for _, line := range strings.Split(b.String(), "\n") {
			if w := o.Width; w > 0 && len([]rune(line)) > w {
				t.Errorf("%s: line wider than %d: %q", name, w, line)
			}
		}
		checkGolden(t, name, b.Bytes())
	}
}

func TestPrettyColour(t *testing.T) {
	meta, res := fixture()
	var b bytes.Buffer
	if err := Pretty(&b, Build(meta, res, 0.5), PrettyOptions{Color: true, Width: 100}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.String(), "\x1b[") || !strings.Contains(b.String(), "╭") {
		t.Error("colour output has no styling or banner")
	}
}

func TestPrettySanitizesHostileStrings(t *testing.T) {
	evil := "evil\x1b]0;pwned\x07\x1b[31mred\u202Egnp.exe\u200B\x00\r\nX"
	res := []Result{{
		Unit:   core.Unit{File: evil + ".c", Language: "C\x1b[2J", Kind: "function", Name: evil, StartLine: 1, EndLine: 2},
		Scores: scores(0.99, map[string]float32{"memory_safety": 0.9}),
	}}
	r := Build(Meta{Root: "/x", Files: 1}, res, 0.5)
	var b bytes.Buffer
	if err := Pretty(&b, r, PrettyOptions{Width: 200}); err != nil {
		t.Fatal(err)
	}
	out := b.String()
	for _, bad := range []string{"\x1b", "\x07", "\u202E", "\u200B", "\x00", "\r", "pwned"} {
		if strings.Contains(out, bad) {
			t.Errorf("output contains %q:\n%q", bad, out)
		}
	}
	if !strings.Contains(out, "evilredgnp.exe  X") {
		t.Errorf("sanitised name missing:\n%s", out)
	}
	var j bytes.Buffer
	if err := WriteJSON(&j, r); err != nil {
		t.Fatal(err)
	}
	if !json.Valid(j.Bytes()) {
		t.Fatal("JSON invalid")
	}
	for _, c := range j.Bytes() {
		if c < 0x20 && c != '\n' {
			t.Fatalf("raw control byte %#x in JSON", c)
		}
	}
}

func TestSanitize(t *testing.T) {
	cases := map[string]string{
		"plain/path.go":                 "plain/path.go",
		"a\x1b[1;31mb\x1b[0m":           "ab",
		"a\x1b]8;;http://x\x1b\\b":      "ab",
		"a\x1b]0;title\x07b":            "ab",
		"a\x1bPdcs\x1b\\b":              "ab",
		"a\u009b31mb":                   "ab",
		"a\u009dosc\u009cb":             "ab",
		"a\u202Eb\u2066c\u2069d\u202Ae": "abcde",
		"zero\u200Bwidth\uFEFF\u2060":   "zerowidth",
		"tab\tnew\nline":                "tab new line",
		"bell\x07del\x7f":               "belldel",
		"bad\xffutf8":                   "bad\uFFFDutf8",
		"日本語/ファイル.py":                   "日本語/ファイル.py",
		"trailing\x1b":                  "trailing",
		"trailing\x1b[":                 "trailing",
	}
	for in, want := range cases {
		if got := sanitize(in); got != want {
			t.Errorf("sanitize(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestProgressLine(t *testing.T) {
	var p Progress
	p.Started = time.Unix(0, 0)
	p.FilesFound.Store(10)
	p.FilesParsed.Store(10)
	p.Units.Store(100)
	p.UnitsScored.Store(50)
	p.TokensTotal.Store(20000)
	p.TokensScored.Store(10000)
	p.CacheHits.Store(3)
	got := ProgressLine(&p, time.Unix(2, 0))
	want := "⠿ files 10/10 · units 50/100 · tokens 10.0k/20.0k · 5,000 tok/s · ETA 2.0s · 3 cached"
	if got != want {
		t.Errorf("got  %q\nwant %q", got, want)
	}
}

func TestRenderProgressNonTTY(t *testing.T) {
	var b bytes.Buffer
	ctx, cancel := contextWithCancel()
	done := make(chan struct{})
	go func() { RenderProgress(ctx, &b, &Progress{Started: time.Now()}); close(done) }()
	time.Sleep(250 * time.Millisecond)
	cancel()
	<-done
	if b.Len() != 0 {
		t.Errorf("wrote to non-TTY: %q", b.String())
	}
}

func TestRenderProgressLoopClears(t *testing.T) {
	var b syncBuffer
	ctx, cancel := contextWithCancel()
	done := make(chan struct{})
	p := &Progress{Started: time.Now()}
	go func() { renderProgressLoop(ctx, &b, p, 5*time.Millisecond, func() int { return 40 }); close(done) }()
	time.Sleep(40 * time.Millisecond)
	cancel()
	<-done
	s := b.String()
	if !strings.Contains(s, "files") || !strings.HasSuffix(s, "\r\x1b[2K") {
		t.Errorf("unexpected progress output %q", s)
	}
}

func checkGolden(t *testing.T, name string, got []byte) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if *update {
		if err := os.MkdirAll("testdata", 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o600); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path) // #nosec G304 -- fixed testdata path
	if err != nil {
		t.Fatalf("%v (run go test -update)", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("%s mismatch:\n--- got\n%s\n--- want\n%s", name, got, want)
	}
}

// A report read from disk is untrusted: an unknown top_family (which is
// rendered verbatim) must be rejected, and terminal escapes must never reach
// the output even if validation were bypassed.
func TestReadJSONRejectsHostileTopFamily(t *testing.T) {
	hostile := "\u001b]52;c;Y3VybCBldmlsfHNo\u0007\u001b]0;pwned\u0007"
	for _, doc := range []string{
		`{"schema_version":"1","units":[{"file":"a.c","severity":0.99,"top_family":` + jsonString(hostile) + `,"families":{}}]}`,
		`{"schema_version":"1","units":[{"file":"a.c","severity":0.99,"top_family":"web_security","families":{},"parts":[{"split_part":1,"top_family":` + jsonString(hostile) + `}]}]}`,
	} {
		if _, err := ReadJSON(strings.NewReader(doc)); err == nil || !strings.Contains(err.Error(), "unknown top_family") {
			t.Fatalf("err = %v, want unknown top_family", err)
		} else if strings.ContainsRune(err.Error(), '\x1b') || strings.ContainsRune(err.Error(), '\a') {
			t.Fatalf("error message carries control characters: %q", err.Error())
		}
	}

	// Defence in depth: Pretty sanitises TopFamily even on an in-memory report.
	r := &Report{SchemaVersion: SchemaVersion, Units: []Unit{{File: "a.c", Severity: 0.99, TopFamily: hostile, Flagged: true}}}
	r.Summary.Threshold = 0.5
	var buf bytes.Buffer
	if err := Pretty(&buf, r, PrettyOptions{Top: 10, Threshold: 0.5, Width: 120}); err != nil {
		t.Fatal(err)
	}
	if out := buf.String(); strings.ContainsRune(out, '\x1b') || strings.ContainsRune(out, '\a') {
		t.Fatalf("pretty output carries escapes: %q", out)
	}
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestReadJSONRejectsOutOfRangeScores(t *testing.T) {
	for _, doc := range []string{
		`{"schema_version":"1","units":[{"file":"a.c","severity":1e9,"families":{}}]}`,
		`{"schema_version":"1","units":[{"file":"a.c","severity":-0.1,"families":{}}]}`,
		`{"schema_version":"1","units":[{"file":"a.c","severity":0.5,"top_family_prob":2,"families":{}}]}`,
		`{"schema_version":"1","units":[{"file":"a.c","severity":0.5,"families":{"web_security":3}}]}`,
		`{"schema_version":"1","units":[],"files":[{"file":"a.c","max_severity":7}]}`,
		`{"schema_version":"1","units":[],"summary":{"threshold":5}}`,
	} {
		if _, err := ReadJSON(strings.NewReader(doc)); err == nil || !strings.Contains(err.Error(), "outside [0,1]") {
			t.Errorf("%s: err = %v", doc, err)
		}
	}
}
