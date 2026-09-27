// Package mcp serves the Model Context Protocol over stdio, so a coding agent
// can run a vakt scan and read its findings as structured data instead of
// scraping the terminal report.
//
// The protocol surface an agent needs is small -- initialize, tools/list,
// tools/call -- so this is a hand-rolled JSON-RPC 2.0 implementation rather
// than a dependency. vakt pins and audits every dependency it ships (see
// docs/BUILD.md), and the official SDK would add eight modules, including JWT
// and OAuth2 code that a stdio server never executes.
package mcp

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// JSON-RPC 2.0 error codes. The first four are from the specification; the
// rest are the application codes this server returns.
const (
	codeParse          = -32700
	codeInvalidRequest = -32600
	codeMethodNotFound = -32601
	codeInvalidParams  = -32602
	codeInternal       = -32603
)

// MaxMessageBytes caps a single incoming message. An MCP client is local and
// its requests are small; this stops a runaway or hostile writer on stdin
// from growing the read buffer without bound.
const MaxMessageBytes = 8 << 20

// request is an incoming JSON-RPC message. A request without an ID is a
// notification and gets no reply.
type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// isNotification reports whether the message must not be replied to. A null
// id is a notification too: the spec forbids using it to match a response.
func (r *request) isNotification() bool {
	return len(r.ID) == 0 || string(r.ID) == "null"
}

// response is an outgoing JSON-RPC reply. Exactly one of Result and Error is
// set, which is why both are pointers.
type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// rpcError is a JSON-RPC error object.
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string { return e.Message }

// errorf builds an rpcError. The message reaches the agent verbatim, so
// callers must not interpolate anything secret into it.
func errorf(code int, format string, args ...any) *rpcError {
	return &rpcError{Code: code, Message: fmt.Sprintf(format, args...)}
}

// conn is a framed JSON-RPC connection over a pair of streams. MCP frames
// messages as newline-delimited JSON on stdio.
//
// Writes are serialised: the server answers one request at a time today, but
// a half-written message would corrupt the stream for good, so the lock is
// cheap insurance rather than something a future concurrent handler must
// remember to add.
type conn struct {
	in  *bufio.Scanner
	out io.Writer
	mu  sync.Mutex
}

func newConn(r io.Reader, w io.Writer) *conn {
	sc := bufio.NewScanner(r)
	// Scanner's default 64 KiB limit is too small for a tools/call that
	// carries a file list; grow to MaxMessageBytes and no further.
	sc.Buffer(make([]byte, 0, 64<<10), MaxMessageBytes)
	return &conn{in: sc, out: w}
}

// read returns the next message. It reports io.EOF when the client hangs up,
// which is the normal way an MCP session ends.
func (c *conn) read() (*request, error) {
	for {
		if !c.in.Scan() {
			if err := c.in.Err(); err != nil {
				if errors.Is(err, bufio.ErrTooLong) {
					return nil, fmt.Errorf("message is larger than the %d byte limit", MaxMessageBytes)
				}
				return nil, err
			}
			return nil, io.EOF
		}
		line := c.in.Bytes()
		if len(trimSpace(line)) == 0 {
			continue // keep-alive blank lines are not messages
		}
		var req request
		if err := json.Unmarshal(line, &req); err != nil {
			return nil, &parseError{err}
		}
		if req.JSONRPC != "2.0" || req.Method == "" {
			return nil, &invalidError{}
		}
		return &req, nil
	}
}

// parseError marks a message that was not JSON, so the loop can answer with
// codeParse and keep the session alive.
type parseError struct{ err error }

func (e *parseError) Error() string { return "parse error: " + e.err.Error() }

// invalidError marks a well-formed JSON message that is not a JSON-RPC 2.0
// request.
type invalidError struct{}

func (e *invalidError) Error() string { return "not a JSON-RPC 2.0 request" }

// reply writes a result for id.
func (c *conn) reply(id json.RawMessage, result any) error {
	return c.write(&response{JSONRPC: "2.0", ID: id, Result: result})
}

// replyError writes an error for id.
func (c *conn) replyError(id json.RawMessage, e *rpcError) error {
	if id == nil {
		id = json.RawMessage("null")
	}
	return c.write(&response{JSONRPC: "2.0", ID: id, Error: e})
}

func (c *conn) write(resp *response) error {
	// Marshal before taking the lock: a message that cannot be encoded must
	// not leave a partial line behind.
	b, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	c.mu.Lock()
	defer c.mu.Unlock()
	_, err = c.out.Write(b)
	return err
}

// trimSpace drops leading and trailing ASCII whitespace.
func trimSpace(b []byte) []byte {
	i, j := 0, len(b)
	for i < j && isSpace(b[i]) {
		i++
	}
	for j > i && isSpace(b[j-1]) {
		j--
	}
	return b[i:j]
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}
