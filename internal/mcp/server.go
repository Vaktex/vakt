package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/vaktex/vakt/internal/brand"
)

// ProtocolVersion is the MCP revision this server implements.
const ProtocolVersion = "2025-06-18"

// Tool is one callable tool. Schema is the raw JSON Schema for the tool's
// arguments, which clients use to build the call.
type Tool struct {
	Name        string
	Description string
	Schema      json.RawMessage
	// ReadOnly marks a tool that cannot change the machine, so an agent can
	// call it without asking the user first. Only scan is not read-only:
	// it writes a report file and can spend minutes of GPU time.
	ReadOnly bool
	Handler  func(ctx context.Context, args json.RawMessage) (any, error)
}

// Server serves MCP over one stdio connection.
type Server struct {
	tools  []Tool
	byName map[string]*Tool
	// initialized guards tool calls until the client has completed the
	// handshake, so a confused client gets a clear error rather than results
	// it never negotiated for.
	initialized bool
	// logf receives protocol-level diagnostics. It must never write to the
	// connection's stdout: that stream carries framed JSON only.
	logf func(format string, args ...any)
}

// New builds a server exposing tools.
func New(tools []Tool, logf func(string, ...any)) *Server {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	s := &Server{tools: tools, byName: make(map[string]*Tool, len(tools)), logf: logf}
	for i := range s.tools {
		s.byName[s.tools[i].Name] = &s.tools[i]
	}
	return s
}

// Serve reads requests from r and writes replies to w until r reaches EOF or
// the context is cancelled. A client hanging up is a clean shutdown, so EOF
// returns nil.
func (s *Server) Serve(ctx context.Context, r io.Reader, w io.Writer) error {
	c := newConn(r, w)
	for {
		if err := ctx.Err(); err != nil {
			return nil // interrupted: the CLI reports the signal
		}
		req, err := c.read()
		switch {
		case errors.Is(err, io.EOF):
			return nil
		case err != nil:
			// A malformed message is the client's bug, not a reason to drop
			// the session: report it and read the next one.
			var pe *parseError
			var ie *invalidError
			if errors.As(err, &pe) || errors.As(err, &ie) {
				s.logf("mcp: %v", err)
				if werr := c.replyError(nil, errorf(codeParse, "%v", err)); werr != nil {
					return werr
				}
				continue
			}
			return err
		}
		if err := s.handle(ctx, c, req); err != nil {
			return err
		}
	}
}

// handle dispatches one request and writes its reply.
func (s *Server) handle(ctx context.Context, c *conn, req *request) error {
	result, rerr := s.dispatch(ctx, req)
	if req.isNotification() {
		// Notifications get no reply, by specification, even on error.
		if rerr != nil {
			s.logf("mcp: notification %s: %s", req.Method, rerr.Message)
		}
		return nil
	}
	if rerr != nil {
		return c.replyError(req.ID, rerr)
	}
	return c.reply(req.ID, result)
}

// dispatch routes a method to its handler.
func (s *Server) dispatch(ctx context.Context, req *request) (any, *rpcError) {
	switch req.Method {
	case "initialize":
		s.initialized = true
		return s.describe(), nil
	case "notifications/initialized":
		return nil, nil
	case "ping":
		// Health check: an empty result is the specified reply.
		return struct{}{}, nil
	case "tools/list":
		if !s.initialized {
			return nil, errorf(codeInvalidRequest, "initialize must be called first")
		}
		return s.listTools(), nil
	case "tools/call":
		if !s.initialized {
			return nil, errorf(codeInvalidRequest, "initialize must be called first")
		}
		return s.callTool(ctx, req.Params)
	default:
		// Optional features (resources, prompts, sampling) are not served.
		return nil, errorf(codeMethodNotFound, "unsupported method %q", req.Method)
	}
}

// describe is the initialize result: what this server is and what it can do.
func (s *Server) describe() map[string]any {
	return map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities": map[string]any{
			// listChanged false: the tool set is fixed at startup.
			"tools": map[string]any{"listChanged": false},
		},
		"serverInfo": map[string]any{
			"name":    brand.Binary,
			"title":   brand.Product,
			"version": brand.Version,
		},
		"instructions": instructions,
	}
}

// instructions tells the agent how to use these tools well. Without it, an
// agent tends to call scan on a whole monorepo and then re-scan for each
// question, which costs minutes of GPU time per answer.
const instructions = brand.Product + " scores every function in a codebase with " +
	brand.ModelName + ", a code-security classifier, and reports the ones most likely " +
	"to be vulnerable.\n\n" +
	"Every tool except families takes dir: the absolute path of the project you " +
	"are working in. Pass the same dir to findings and finding that you scanned.\n\n" +
	"Use it like this:\n" +
	"  1. Call scan once for a directory. It is the expensive step (a model runs " +
	"over every function), so scan the narrowest path that answers the question " +
	"and reuse the report.\n" +
	"  2. Call findings to list what it flagged, filtered by severity or CWE family.\n" +
	"  3. Call finding for one result: its code, its CWE and its score.\n" +
	"  4. Call families for the CWE taxonomy the scores use.\n\n" +
	"Scores are probabilities from a model, not proofs. Treat a finding as a lead " +
	"to verify by reading the code, and say so when reporting it. A high score on " +
	"code that is unreachable or already validated upstream is a false positive; " +
	"the model sees one function at a time and cannot know the caller."

// toolInfo is one entry in tools/list.
type toolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
	Annotations *annotations    `json:"annotations,omitempty"`
}

// annotations are the behavioural hints a client uses to decide whether a
// call needs the user's approval.
type annotations struct {
	ReadOnlyHint    bool `json:"readOnlyHint"`
	DestructiveHint bool `json:"destructiveHint"`
	IdempotentHint  bool `json:"idempotentHint"`
}

func (s *Server) listTools() map[string]any {
	list := make([]toolInfo, 0, len(s.tools))
	for i := range s.tools {
		t := &s.tools[i]
		list = append(list, toolInfo{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Schema,
			Annotations: &annotations{
				ReadOnlyHint: t.ReadOnly,
				// Nothing here deletes or overwrites the user's source. scan
				// writes only its own report file, when asked to.
				DestructiveHint: false,
				IdempotentHint:  t.ReadOnly,
			},
		})
	}
	return map[string]any{"tools": list}
}

// callParams are the arguments of tools/call.
type callParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// content is one block of a tool result.
type content struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// callResult is a tools/call reply. Every tool here returns structured data:
// Content holds its JSON rendering, for clients that only read text, and
// StructuredContent holds the same value as JSON for those that parse it.
type callResult struct {
	Content           []content `json:"content"`
	StructuredContent any       `json:"structuredContent,omitempty"`
	IsError           bool      `json:"isError,omitempty"`
}

// callTool runs one tool.
//
// A tool that fails returns a result with isError set, not a JSON-RPC error:
// the specification reserves protocol errors for protocol problems, and this
// way the agent sees the message and can correct itself (a bad path, a family
// name that does not exist) instead of the client reporting a broken server.
func (s *Server) callTool(ctx context.Context, params json.RawMessage) (any, *rpcError) {
	var p callParams
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, errorf(codeInvalidParams, "invalid tools/call params: %v", err)
	}
	t, ok := s.byName[p.Name]
	if !ok {
		return nil, errorf(codeMethodNotFound, "unknown tool %q", p.Name)
	}
	args := p.Arguments
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	out, err := t.Handler(ctx, args)
	if err != nil {
		// Cancellation is the session ending, not a tool result.
		if ctx.Err() != nil {
			return nil, errorf(codeInternal, "cancelled")
		}
		s.logf("mcp: %s: %v", p.Name, err)
		return &callResult{
			Content: []content{{Type: "text", Text: err.Error()}},
			IsError: true,
		}, nil
	}
	text, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return nil, errorf(codeInternal, "encode %s result: %v", p.Name, err)
	}
	return &callResult{
		Content:           []content{{Type: "text", Text: string(text)}},
		StructuredContent: out,
	}, nil
}

// ErrUsage is returned by a tool whose arguments are wrong. It reaches the
// agent as the tool's error text.
func ErrUsage(format string, args ...any) error {
	return fmt.Errorf(format, args...)
}
