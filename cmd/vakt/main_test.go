package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/report"
)

func runCLI(t *testing.T, stdin string, args ...string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errb bytes.Buffer
	code = run(context.Background(), args, strings.NewReader(stdin), &out, &errb)
	return code, out.String(), errb.String()
}

func isolate(t *testing.T) {
	t.Helper()
	t.Setenv("VAKT_CACHE", t.TempDir())
	t.Setenv("HF_HOME", t.TempDir())
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")
	t.Setenv("NO_COLOR", "")
	t.Setenv("COLUMNS", "")
}

func TestRootHelpIsClean(t *testing.T) {
	code, stdout, stderr := runCLI(t, "", "--help")
	if code != exitOK || stderr != "" {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
	want := `Vaktex OSS (vakt)

Usage:
  vakt [command]

Available Commands:
  doctor      Check this machine, the engine and the model cache
  help        Help about any command
  mcp         Serve Vaktex OSS to a coding agent over the Model Context Protocol
  patrol      Scan a codebase and report the functions most likely to be vulnerable
  report      Re-render a saved JSON report
  show        Print full details for finding n from a report
  summon      Download DOM-0.8B into the local cache
  version     Print the version, commit, backend and model

Flags:
  -h, --help      help for vakt
  -v, --version   version for vakt

Use "vakt [command] --help" for more information about a command.
`
	if stdout != want {
		t.Errorf("root help:\n--- got ---\n%s--- want ---\n%s", stdout, want)
	}
	for _, slop := range []string{
		"scans a codebase for vulnerable code",
		"It splits source into functions",
		"Exit status:",
		"short for",
	} {
		if strings.Contains(stdout, slop) {
			t.Errorf("root help contains removed prose %q", slop)
		}
	}
}

func TestDefaultToPatrol(t *testing.T) {
	root := newRootCmd(nil, nil, nil)
	cases := map[string]string{
		".":               "patrol .",
		"src --top 3":     "patrol src --top 3",
		"--format json .": "patrol --format json .",
		"patrol .":        "patrol .",
		"scan .":          "scan .",
		"doctor":          "doctor",
		"report r.json":   "report r.json",
		"show 1":          "show 1",
		"summon":          "summon",
		"version":         "version",
		"--version":       "--version",
		"--help":          "--help",
		"help patrol":     "help patrol",
		"summon-typo":     "patrol summon-typo",
	}
	for in, want := range cases {
		if got := strings.Join(defaultToPatrol(root, strings.Fields(in)), " "); got != want {
			t.Errorf("%q -> %q, want %q", in, got, want)
		}
	}
}

func TestPatrolWithoutPipeline(t *testing.T) {
	isolate(t)
	code, _, stderr := runCLI(t, "", t.TempDir())
	if code != exitError || !strings.Contains(stderr, "pipeline isn't linked") {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
}

func TestPatrolHookReceivesOptions(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	var got ScanOptions
	old := runScan
	t.Cleanup(func() { runScan = old })
	runScan = func(ctx context.Context, o ScanOptions, prog *report.Progress) (*report.Report, error) {
		got = o
		if prog == nil {
			return nil, errors.New("nil progress")
		}
		prog.FilesFound.Add(1)
		return report.Build(report.Meta{Backend: "cpu"}, nil, o.Threshold), nil
	}
	code, stdout, stderr := runCLI(t, "", dir, "--threshold", "0.7", "--top", "5", "--min-tokens", "8",
		"--batch-tokens", "4096", "--jobs", "3", "--include", "**/*.go", "--include", "**/*.c", "--exclude", "vendor/**",
		"--no-cache", "--revision", "v2", "--precision", "bf16", "--device", "gpu", "--devices", "1,0,1",
		"--max-file-bytes", "1000", "--follow-symlinks", "--no-color")
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr)
	}
	abs, _ := filepath.Abs(dir)
	if got.Root != abs || got.Threshold != 0.7 || got.Top != 5 || got.MinTokens != 8 || got.BatchTokens != 4096 ||
		got.Jobs != 3 || len(got.Include) != 2 || got.Exclude[0] != "vendor/**" || !got.NoCache ||
		got.ModelRepo != brand.ModelRepo || got.ModelRevision != "v2" || got.ModelPath != "" ||
		got.Precision != "bf16" || got.Device != "gpu" || len(got.Devices) != 2 || got.Devices[0] != 1 ||
		got.MaxFileBytes != 1000 || !got.FollowSymlinks || !got.NoColor || got.Format != "pretty" || got.Demo {
		t.Errorf("options = %+v", got)
	}
	if !strings.Contains(stdout, "All clear") {
		t.Errorf("stdout = %q", stdout)
	}
}

func TestPatrolFlagValidation(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	bad := [][]string{
		{"--format", "xml"},
		{"--threshold", "0"},
		{"--threshold", "1.5"},
		{"--top", "0"},
		{"--jobs", "0"},
		{"--precision", "fp16"},
		{"--device", "tpu"},
		{"--devices", "a,b"},
		{"--device", "cpu", "--devices", "0"},
		{"--model", "hf:../../etc@main"},
		{"--model", "hf:vaktex/DOM-0.8B@../x"},
		{"--model", "/does/not/exist.safetensors"},
		{"--fail-on", "2"},
		{"--out", "x.json"}, // pretty + --out
		{"--format", "both", "--out", "-"},
		{"--include", ""},
	}
	for _, args := range bad {
		code, _, stderr := runCLI(t, "", append([]string{"patrol", dir}, args...)...)
		if code != exitError || stderr == "" {
			t.Errorf("%v: code=%d stderr=%q", args, code, stderr)
		}
	}
	if code, _, stderr := runCLI(t, "", "patrol", filepath.Join(dir, "missing")); code != exitError || !strings.Contains(stderr, "cannot scan") {
		t.Errorf("missing dir: %d %q", code, stderr)
	}
}

func TestParseModelSpec(t *testing.T) {
	m, err := parseModelSpec("hf:owner/name@v1", "")
	if err != nil || m.Repo != "owner/name" || m.Revision != "v1" {
		t.Errorf("%+v %v", m, err)
	}
	m, _ = parseModelSpec("hf:owner/name", "")
	if m.Revision != "main" {
		t.Errorf("default revision %q", m.Revision)
	}
	m, _ = parseModelSpec("hf:owner/name@v1", "abc")
	if m.Revision != "abc" {
		t.Errorf("--revision override %q", m.Revision)
	}
	p := filepath.Join(t.TempDir(), "model.safetensors")
	_ = os.WriteFile(p, []byte("x"), 0o600)
	if m, err := parseModelSpec(p, ""); err != nil || m.Path != p {
		t.Errorf("local: %+v %v", m, err)
	}
	if _, err := parseModelSpec(p, "v1"); err == nil {
		t.Error("--revision with a local model accepted")
	}
	if _, err := parseModelSpec(filepath.Dir(p), ""); err == nil {
		t.Error("directory accepted as model")
	}
}

func TestDemoPrettyAndFailOn(t *testing.T) {
	isolate(t)
	code, stdout, stderr := runCLI(t, "", "patrol", "--demo", "--no-color")
	if code != exitFindings {
		t.Fatalf("code=%d stderr=%s", code, stderr)
	}
	for _, want := range []string{brand.Product, "fake", "Top Findings", "SEVERITY", "FAMILY", "CONF", "LOCATION", "UNIT",
		"Flagged files", "units flagged across"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("demo output missing %q:\n%s", want, stdout)
		}
	}
	// memory_safety has two flagged findings in the demo, so the section shows.
	if !strings.Contains(stdout, "Families") {
		t.Error("Families missing although memory_safety repeats")
	}
	if strings.Contains(stdout, "\x1b") {
		t.Error("--no-color output has escapes")
	}
	code, _, stderr = runCLI(t, "", "patrol", "--demo", "--quiet", "--fail-on", "0.5")
	if code != exitFindings || !strings.Contains(stderr, "--fail-on 0.50") {
		t.Errorf("fail-on: code=%d stderr=%q", code, stderr)
	}
	if code, _, _ := runCLI(t, "", "patrol", "--demo", "--quiet", "--fail-on", "0.99"); code != 0 {
		t.Errorf("fail-on above max: code=%d", code)
	}
	// NO_COLOR is honoured even without --no-color (the buffer is not a TTY anyway).
	t.Setenv("NO_COLOR", "1")
	if _, out, _ := runCLI(t, "", "--demo"); strings.Contains(out, "\x1b") {
		t.Error("NO_COLOR ignored")
	}
}

func TestDemoJSONAndReportCmd(t *testing.T) {
	isolate(t)
	code, stdout, stderr := runCLI(t, "", "patrol", "--demo", "--format", "json", "--out", "-")
	if code != exitFindings {
		t.Fatalf("code=%d stderr=%s", code, stderr)
	}
	var raw map[string]any
	if err := json.Unmarshal([]byte(stdout), &raw); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	rep, err := report.ReadJSON(strings.NewReader(stdout))
	if err != nil {
		t.Fatal(err)
	}
	if rep.Model.Backend != "fake" || rep.SchemaVersion != "1" {
		t.Errorf("model = %+v", rep.Model)
	}

	path := filepath.Join(t.TempDir(), "r.json")
	if code, _, stderr := runCLI(t, "", "patrol", "--demo", "--format", "both", "--out", path, "--no-color"); code != exitFindings {
		t.Fatalf("both: %d %s", code, stderr)
	}
	if st, err := os.Stat(path); err != nil || st.Mode().Perm() != 0o600 {
		t.Fatalf("report file: %v", err)
	}

	code, out, stderr := runCLI(t, "", "report", path, "--no-color", "--top", "2", "--family", "memory_safety")
	if code != 0 || !strings.Contains(out, "memory_safety") || strings.Contains(out, "file_and_path") {
		t.Errorf("report --family: %d %s\n%s", code, stderr, out)
	}
	code, out, _ = runCLI(t, "", "report", path, "--format", "json", "--threshold", "0.75")
	if code != 0 {
		t.Fatal(code)
	}
	r2, err := report.ReadJSON(strings.NewReader(out))
	if err != nil || r2.Summary.Threshold != 0.75 || r2.Summary.FlaggedUnits >= rep.Summary.FlaggedUnits {
		t.Errorf("rethreshold: %+v %v", r2.Summary, err)
	}
	data, _ := os.ReadFile(path) // #nosec G304 -- test file
	if code, out, _ := runCLI(t, string(data), "report", "-", "--quiet"); code != 0 || !strings.Contains(out, "units flagged") {
		t.Errorf("stdin: %d %q", code, out)
	}
	if code, _, stderr := runCLI(t, "", "report", path, "--family", "nope"); code != exitError || !strings.Contains(stderr, "unknown family") {
		t.Errorf("bad family: %d %q", code, stderr)
	}
	if code, _, stderr := runCLI(t, `{"schema_version":"9"}`, "report", "-"); code != exitError || !strings.Contains(stderr, "schema_version") {
		t.Errorf("bad schema: %d %q", code, stderr)
	}

	code, out, stderr = runCLI(t, "", "show", "1", "--report", path, "--no-color")
	if code != 0 || !strings.Contains(out, "score") || !strings.Contains(out, "family") {
		t.Errorf("show: %d %s\n%s", code, stderr, out)
	}
	if code, _, stderr := runCLI(t, "", "show", "999", "--report", path); code != exitError || !strings.Contains(stderr, "not found") {
		t.Errorf("show OOB: %d %q", code, stderr)
	}
}

func TestVersion(t *testing.T) {
	for _, args := range [][]string{{"version"}, {"--version"}} {
		code, out, _ := runCLI(t, "", args...)
		first, _, _ := strings.Cut(out, "\n")
		if code != 0 || !strings.Contains(out, brand.Product) || !strings.Contains(out, brand.ModelRepo) ||
			!regexp.MustCompile(`version=[^ ]+ commit=[^ ]+ backend=[^ ]+`).MatchString(first) {
			t.Errorf("%v: %d %q", args, code, out)
		}
	}
}

func TestDoctor(t *testing.T) {
	isolate(t)
	oldCmd, oldLook, oldBench := commandOutput, lookPath, doctorBench
	t.Cleanup(func() { commandOutput, lookPath, doctorBench = oldCmd, oldLook, oldBench })
	commandOutput = func(_ context.Context, name string, args ...string) (string, error) {
		switch name {
		case "sysctl":
			return "Apple M5 Pro", nil
		case "nvidia-smi":
			return "575.10, NVIDIA H100 80GB HBM3\n575.10, NVIDIA H100 80GB HBM3", nil
		case "ldconfig":
			return "\tlibcublas.so.13 (libc6,x86-64) => /usr/lib/libcublas.so.13\n\tlibcudnn.so.9 (libc6,x86-64) => /x", nil
		}
		return "", errors.New("unexpected")
	}
	lookPath = func(string) (string, error) { return "/usr/bin/nvidia-smi", nil }

	var b bytes.Buffer
	if err := runDoctor(context.Background(), &b, "darwin", true); err != nil {
		t.Fatal(err)
	}
	out := b.String()
	for _, want := range []string{"Apple M5 Pro", "engine not linked", "not cached", "hf token         no", brand.ModelRepo} {
		if !strings.Contains(out, want) {
			t.Errorf("darwin doctor missing %q:\n%s", want, out)
		}
	}

	t.Setenv("HF_TOKEN", "hf_SECRET_doctor_token")
	doctorBench = func(context.Context) (string, error) { return "42 units/s", nil }
	b.Reset()
	_ = runDoctor(context.Background(), &b, "linux", true)
	out = b.String()
	for _, want := range []string{"gpu 0", "NVIDIA H100", "575.10 (CUDA 13 needs >= 580", "missing libcublasLt.so.13, libnvrtc.so.13", "42 units/s", "hf token         yes"} {
		if !strings.Contains(out, want) {
			t.Errorf("linux doctor missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "hf_SECRET") {
		t.Error("doctor printed the token")
	}
}

func TestSummonErrors(t *testing.T) {
	isolate(t)
	if code, _, stderr := runCLI(t, "", "summon", "--token-stdin"); code != exitError || !strings.Contains(stderr, "no token") {
		t.Errorf("empty stdin: %d %q", code, stderr)
	}
	if code, _, stderr := runCLI(t, "", "summon", "--offline"); code != exitError || !strings.Contains(stderr, "summon") {
		t.Errorf("offline empty cache: %d %q", code, stderr)
	}
	if code, _, stderr := runCLI(t, "", "summon", "--revision", "../x"); code != exitError || !strings.Contains(stderr, "invalid revision") {
		t.Errorf("bad revision: %d %q", code, stderr)
	}
	if _, err := readToken(strings.NewReader("hf_abc\n")); err != nil {
		t.Error(err)
	}
	if _, err := readToken(strings.NewReader("hf abc\n")); err == nil {
		t.Error("token with space accepted")
	}
}

func TestResolveModel(t *testing.T) {
	isolate(t)
	p := filepath.Join(t.TempDir(), "model.safetensors")
	if path, sha, err := resolveModel(context.Background(), ScanOptions{ModelPath: p}, &bytes.Buffer{}); err != nil || path != p || sha != "" {
		t.Errorf("local: %q %q %v", path, sha, err)
	}
	t.Setenv("HF_HUB_OFFLINE", "1")
	_, _, err := resolveModel(context.Background(), ScanOptions{ModelRepo: brand.ModelRepo, ModelRevision: "main"}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "summon") {
		t.Errorf("offline, empty cache: %v", err)
	}
}

func TestBarLine(t *testing.T) {
	got := barLine(512<<20, 1<<30, 0, 80)
	if !strings.Contains(got, " 50% 512.0 MB/1.00 GB") || !strings.HasPrefix(got, "summoning model ▕") {
		t.Errorf("bar = %q", got)
	}
}

// An interrupted scan exits 130 even though the cancellation surfaces as a
// wrapped scan error.
func TestInterruptExits130(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	old := runScan
	t.Cleanup(func() { runScan = old })
	ctx, cancel := context.WithCancel(context.Background())
	runScan = func(c context.Context, _ ScanOptions, _ *report.Progress) (*report.Report, error) {
		cancel()
		<-c.Done()
		return nil, fmt.Errorf("pipeline: scoring 2 units: %w", c.Err())
	}
	var out, errb bytes.Buffer
	if err := os.WriteFile(filepath.Join(dir, "m.safetensors"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if code := run(ctx, []string{"patrol", dir, "--no-color", "--model", filepath.Join(dir, "m.safetensors")}, strings.NewReader(""), &out, &errb); code != exitSignals {
		t.Fatalf("exit %d, want %d; stderr %q", code, exitSignals, errb.String())
	}
	if !strings.Contains(errb.String(), "interrupted") {
		t.Errorf("stderr %q", errb.String())
	}
}

func TestDoctorStrict(t *testing.T) {
	isolate(t)
	oldCmd, oldLook, oldBench := commandOutput, lookPath, doctorBench
	t.Cleanup(func() { commandOutput, lookPath, doctorBench = oldCmd, oldLook, oldBench })
	commandOutput = func(context.Context, string, ...string) (string, error) { return "", errors.New("none") }
	lookPath = func(string) (string, error) { return "", errors.New("none") }
	var out, errb bytes.Buffer
	if code := run(context.Background(), []string{"doctor", "--no-bench"}, strings.NewReader(""), &out, &errb); code != exitOK {
		t.Fatalf("plain doctor exit %d", code)
	}
	out.Reset()
	if code := run(context.Background(), []string{"doctor", "--no-bench", "--strict"}, strings.NewReader(""), &out, &errb); code != exitError {
		t.Fatalf("strict doctor with warnings exit %d:\n%s", code, out.String())
	}
}
