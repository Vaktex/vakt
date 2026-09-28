package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

// session drives a server over a pipe of newline-delimited JSON, the way a
// client does, and returns one decoded reply per request that has an id.
func session(t *testing.T, tools []Tool, requests ...string) []response {
	t.Helper()
	in := strings.NewReader(strings.Join(requests, "\n") + "\n")
	var out strings.Builder
	srv := New(tools, func(string, ...any) {})
	if err := srv.Serve(context.Background(), in, &out); err != nil {
		t.Fatalf("Serve: %v", err)
	}
	var got []response
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		if line == "" {
			continue
		}
		var r response
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("reply %q is not JSON: %v", line, err)
		}
		got = append(got, r)
	}
	return got
}

// initReq and a tool call, the shortest useful session.
const initReq = `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`

// echoTool is a tool that returns its arguments, or fails on demand.
func echoTool(fail error) Tool {
	return Tool{
		Name:     "echo",
		Schema:   json.RawMessage(`{"type":"object"}`),
		ReadOnly: true,
		Handler: func(_ context.Context, args json.RawMessage) (any, error) {
			if fail != nil {
				return nil, fail
			}
			return map[string]any{"got": string(args)}, nil
		},
	}
}

func TestInitializeAdvertisesToolsAndInstructions(t *testing.T) {
	got := session(t, []Tool{echoTool(nil)}, initReq)
	if len(got) != 1 {
		t.Fatalf("want 1 reply, got %d", len(got))
	}
	var res struct {
		ProtocolVersion string `json:"protocolVersion"`
		Capabilities    struct {
			Tools *struct{} `json:"tools"`
		} `json:"capabilities"`
		ServerInfo   struct{ Name string } `json:"serverInfo"`
		Instructions string                `json:"instructions"`
	}
	decodeResult(t, got[0], &res)
	if res.ProtocolVersion != ProtocolVersion {
		t.Errorf("protocolVersion = %q, want %q", res.ProtocolVersion, ProtocolVersion)
	}
	if res.Capabilities.Tools == nil {
		t.Error("tools capability not advertised")
	}
	if res.ServerInfo.Name != "vakt" {
		t.Errorf("serverInfo.name = %q", res.ServerInfo.Name)
	}
	// The instructions are what stop an agent re-scanning for every question,
	// so their absence is a real regression, not cosmetic.
	if !strings.Contains(res.Instructions, "scan") || !strings.Contains(res.Instructions, "not proofs") &&
		!strings.Contains(res.Instructions, "not proof") {
		t.Errorf("instructions do not explain scanning cost and uncertainty: %q", res.Instructions)
	}
}

func TestToolsRequireInitialize(t *testing.T) {
	// A client that skips the handshake gets a clear protocol error rather
	// than results it never negotiated.
	for _, method := range []string{"tools/list", "tools/call"} {
		got := session(t, []Tool{echoTool(nil)},
			fmt.Sprintf(`{"jsonrpc":"2.0","id":7,"method":%q}`, method))
		if len(got) != 1 || got[0].Error == nil {
			t.Fatalf("%s without initialize: want an error, got %+v", method, got)
		}
		if got[0].Error.Code != codeInvalidRequest {
			t.Errorf("%s: code = %d, want %d", method, got[0].Error.Code, codeInvalidRequest)
		}
	}
}

func TestToolsListReportsSchemaAndAnnotations(t *testing.T) {
	tools := []Tool{
		{Name: "ro", Schema: json.RawMessage(`{"type":"object"}`), ReadOnly: true, Handler: nil},
		{Name: "rw", Schema: json.RawMessage(`{"type":"object"}`), Handler: nil},
	}
	got := session(t, tools, initReq, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
	var res struct{ Tools []toolInfo }
	decodeResult(t, got[1], &res)
	if len(res.Tools) != 2 {
		t.Fatalf("want 2 tools, got %d", len(res.Tools))
	}
	if !res.Tools[0].Annotations.ReadOnlyHint {
		t.Error("read-only tool is not hinted read-only; clients would prompt for it")
	}
	if res.Tools[1].Annotations.ReadOnlyHint {
		t.Error("a tool that runs a scan must not be hinted read-only")
	}
	for _, tl := range res.Tools {
		if !json.Valid(tl.InputSchema) {
			t.Errorf("%s: input schema is not valid JSON", tl.Name)
		}
	}
}

// Every shipped tool must advertise a schema a client can actually consume.
func TestShippedToolSchemasAreValid(t *testing.T) {
	for _, tl := range Tools(Deps{}) {
		if !json.Valid(tl.Schema) {
			t.Errorf("%s: schema is not valid JSON: %s", tl.Name, tl.Schema)
			continue
		}
		var s struct {
			Type                 string `json:"type"`
			AdditionalProperties *bool  `json:"additionalProperties"`
		}
		if err := json.Unmarshal(tl.Schema, &s); err != nil {
			t.Errorf("%s: %v", tl.Name, err)
			continue
		}
		if s.Type != "object" {
			t.Errorf("%s: schema type = %q, want object", tl.Name, s.Type)
		}
		// Arguments are decoded strictly, so the schema must say so or a
		// client will happily send fields the tool then rejects.
		if s.AdditionalProperties == nil || *s.AdditionalProperties {
			t.Errorf("%s: schema should set additionalProperties:false", tl.Name)
		}
		if tl.Description == "" {
			t.Errorf("%s: no description; the agent has nothing to choose on", tl.Name)
		}
	}
}

func TestToolCallReturnsStructuredContent(t *testing.T) {
	got := session(t, []Tool{echoTool(nil)}, initReq,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"echo","arguments":{"a":1}}}`)
	var res callResult
	decodeResult(t, got[1], &res)
	if res.IsError {
		t.Fatalf("unexpected tool error: %+v", res.Content)
	}
	if len(res.Content) != 1 || res.Content[0].Type != "text" {
		t.Fatalf("content = %+v, want one text block", res.Content)
	}
	// Both renderings must be present: text for clients that only read text,
	// structuredContent for those that parse.
	if res.StructuredContent == nil {
		t.Error("no structuredContent")
	}
	// echo returns its raw arguments as a string, so the text block contains
	// them JSON-escaped.
	if !strings.Contains(res.Content[0].Text, `\"a\":1`) {
		t.Errorf("arguments did not reach the handler: %q", res.Content[0].Text)
	}
}

// A tool that fails is a result with isError, not a JSON-RPC error: the agent
// must see the message and be able to correct itself.
func TestToolFailureIsAResultNotAProtocolError(t *testing.T) {
	got := session(t, []Tool{echoTool(errors.New("threshold must be in (0, 1]"))}, initReq,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"echo","arguments":{}}}`)
	if got[1].Error != nil {
		t.Fatalf("tool failure became a protocol error: %+v", got[1].Error)
	}
	var res callResult
	decodeResult(t, got[1], &res)
	if !res.IsError {
		t.Error("isError not set")
	}
	if !strings.Contains(res.Content[0].Text, "threshold must be in") {
		t.Errorf("error text lost: %q", res.Content[0].Text)
	}
}

func TestUnknownToolAndMethod(t *testing.T) {
	got := session(t, []Tool{echoTool(nil)}, initReq,
		`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"nope","arguments":{}}}`,
		`{"jsonrpc":"2.0","id":3,"method":"resources/list"}`)
	for i, want := range []int{codeMethodNotFound, codeMethodNotFound} {
		r := got[i+1]
		if r.Error == nil || r.Error.Code != want {
			t.Errorf("reply %d: error = %+v, want code %d", i+1, r.Error, want)
		}
	}
}

// A malformed message must not end the session: the next request still works.
func TestMalformedMessageDoesNotKillTheSession(t *testing.T) {
	got := session(t, []Tool{echoTool(nil)},
		`{"jsonrpc":"2.0","id":1,"method":"initialize"}`,
		`not json at all`,
		`{"jsonrpc":"1.0","id":9,"method":"old"}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)
	if len(got) != 4 {
		t.Fatalf("want 4 replies, got %d: %+v", len(got), got)
	}
	if got[1].Error == nil || got[1].Error.Code != codeParse {
		t.Errorf("bad JSON: %+v, want parse error", got[1].Error)
	}
	if got[2].Error == nil {
		t.Error("a non-2.0 request was accepted")
	}
	var res struct{ Tools []toolInfo }
	decodeResult(t, got[3], &res)
	if len(res.Tools) != 1 {
		t.Errorf("session did not recover: %+v", res.Tools)
	}
}

// Notifications get no reply, by specification. A client that sees one for
// notifications/initialized may treat the stream as corrupt.
func TestNotificationsGetNoReply(t *testing.T) {
	got := session(t, []Tool{echoTool(nil)},
		initReq,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":null,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":2,"method":"ping"}`)
	if len(got) != 2 {
		t.Fatalf("want 2 replies (initialize, ping), got %d: %+v", len(got), got)
	}
	if string(got[1].ID) != "2" {
		t.Errorf("second reply is for id %s, want 2", got[1].ID)
	}
}

func TestBlankLinesAreSkipped(t *testing.T) {
	got := session(t, []Tool{echoTool(nil)}, initReq, "", "   ",
		`{"jsonrpc":"2.0","id":2,"method":"ping"}`)
	if len(got) != 2 {
		t.Fatalf("blank lines produced replies: %+v", got)
	}
}

// Serve returns cleanly when the client hangs up: an EOF is how a session
// normally ends, and must not look like a failure to the CLI.
func TestEOFIsACleanShutdown(t *testing.T) {
	srv := New([]Tool{echoTool(nil)}, nil)
	var out strings.Builder
	if err := srv.Serve(context.Background(), strings.NewReader(""), &out); err != nil {
		t.Fatalf("Serve on empty input: %v", err)
	}
}

func TestCancelledContextStopsServing(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	srv := New([]Tool{echoTool(nil)}, nil)
	var out strings.Builder
	if err := srv.Serve(ctx, strings.NewReader(initReq+"\n"), &out); err != nil {
		t.Fatalf("Serve after cancel: %v", err)
	}
}

func decodeResult(t *testing.T, r response, v any) {
	t.Helper()
	if r.Error != nil {
		t.Fatalf("unexpected error reply: %+v", r.Error)
	}
	b, err := json.Marshal(r.Result)
	if err != nil {
		t.Fatalf("marshal result: %v", err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("decode result %s: %v", b, err)
	}
}
