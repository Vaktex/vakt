package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/report"
)

// mcpSession runs `vakt mcp` over a scripted stdin and returns the decoded
// replies, so the test exercises the same path an agent does.
func mcpSession(t *testing.T, dir string, requests ...string) []map[string]any {
	t.Helper()
	code, stdout, stderr := runCLI(t, strings.Join(requests, "\n")+"\n", "mcp", "--root", dir)
	if code != exitOK {
		t.Fatalf("mcp exited %d: %s", code, stderr)
	}
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(stdout), "\n") {
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("stdout line is not JSON (%v): %q\nstderr: %s", err, line, stderr)
		}
		out = append(out, m)
	}
	return out
}

const mcpInit = `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`

// The command must speak the protocol on stdout and keep its own chatter on
// stderr: one stray byte on stdout breaks the session.
func TestMCPServesProtocolOnStdoutOnly(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	code, stdout, stderr := runCLI(t, mcpInit+"\n", "mcp", "--root", dir)
	if code != exitOK {
		t.Fatalf("code=%d stderr=%q", code, stderr)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(stdout)), &m); err != nil {
		t.Fatalf("stdout is not a single JSON message: %v (%q)", err, stdout)
	}
	if m["result"] == nil {
		t.Errorf("no result in %v", m)
	}
	// The startup notice belongs on stderr, where it cannot corrupt the
	// protocol stream.
	if !strings.Contains(stderr, "serving") {
		t.Errorf("no startup notice on stderr: %q", stderr)
	}
}

func TestMCPListsTheDocumentedTools(t *testing.T) {
	isolate(t)
	got := mcpSession(t, t.TempDir(), mcpInit, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
	if len(got) != 2 {
		t.Fatalf("want 2 replies, got %d", len(got))
	}
	result := got[1]["result"].(map[string]any)
	names := map[string]bool{}
	for _, tl := range result["tools"].([]any) {
		names[tl.(map[string]any)["name"].(string)] = true
	}
	// These four are what the help text and README promise.
	for _, want := range []string{"scan", "findings", "finding", "families"} {
		if !names[want] {
			t.Errorf("tool %q missing; tools = %v", want, names)
		}
	}
}

// The scan tool must reach the CLI's own scan path, with the flags the agent
// asked for, and write the report the read-only tools then read.
func TestMCPScanUsesTheCLIScanPath(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte("package a\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var got ScanOptions
	old := runScan
	t.Cleanup(func() { runScan = old })
	runScan = func(_ context.Context, o ScanOptions, _ *report.Progress) (*report.Report, error) {
		got = o
		return report.Build(report.Meta{Root: o.Root, Backend: "fake"}, nil, o.Threshold), nil
	}

	replies := mcpSession(t, dir, mcpInit,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"scan","arguments":{"dir":"`+jsonEsc(dir)+`","threshold":0.8,"include":["**/*.go"]}}}`)
	res := replies[1]["result"].(map[string]any)
	if res["isError"] == true {
		t.Fatalf("scan failed: %v", res["content"])
	}
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got.Root != real {
		t.Errorf("scanned %q, want %q", got.Root, real)
	}
	if got.Threshold != 0.8 || len(got.Include) != 1 || got.Include[0] != "**/*.go" {
		t.Errorf("options not passed through: %+v", got)
	}
	// Progress rendering would write to stderr and slow a tool call down for
	// no one's benefit: the agent reads the result, not a bar.
	if !got.Quiet {
		t.Error("scan should run quiet under MCP")
	}
	// The report must land on disk so findings/finding and `vakt show` agree.
	if _, err := os.Stat(filepath.Join(real, defaultOut)); err != nil {
		t.Errorf("scan did not write %s: %v", defaultOut, err)
	}
}

// Without an engine, the tool must return the explanation as a tool result,
// so the agent can report it, rather than killing the session.
func TestMCPScanWithoutEngineIsAToolError(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	replies := mcpSession(t, dir, mcpInit,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"scan","arguments":{"dir":"`+jsonEsc(dir)+`"}}}`)
	res := replies[1]["result"].(map[string]any)
	if res["isError"] != true {
		t.Fatalf("want a tool error, got %v", res)
	}
	// In a real binary runScan is always bound: without the engine it is the
	// stub that explains the build. Either way the agent must be told that the
	// build cannot scan, and be told which backend it has.
	text := res["content"].([]any)[0].(map[string]any)["text"].(string)
	if !strings.Contains(text, "isn't linked") && !strings.Contains(text, "no scanning engine") {
		t.Errorf("text does not explain that this build cannot scan: %q", text)
	}
	if !strings.Contains(text, brand.Backend) {
		t.Errorf("text does not name the backend (%s): %q", brand.Backend, text)
	}
}

func TestMCPRootMustExist(t *testing.T) {
	isolate(t)
	code, _, stderr := runCLI(t, "", "mcp", "--root", filepath.Join(t.TempDir(), "nope"))
	if code != exitError {
		t.Fatalf("code = %d, want %d", code, exitError)
	}
	if !strings.Contains(stderr, "--root") {
		t.Errorf("stderr = %q", stderr)
	}
	// A file is not a directory to confine to.
	f := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(f, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	code, _, stderr = runCLI(t, "", "mcp", "--root", f)
	if code != exitError || !strings.Contains(stderr, "not a directory") {
		t.Errorf("code=%d stderr=%q", code, stderr)
	}
}

// Without --root the server starts and accepts any directory.
func TestMCPWithoutRootStarts(t *testing.T) {
	isolate(t)
	code, _, stderr := runCLI(t, mcpInit+"\n", "mcp")
	if code != exitOK || !strings.Contains(stderr, "any directory") {
		t.Errorf("code=%d stderr=%q", code, stderr)
	}
}

func jsonEsc(s string) string {
	b, _ := json.Marshal(s)
	return string(b[1 : len(b)-1])
}
