package ast

// Parse isolation.
//
// tree-sitter's error recovery can use superlinear time and memory on
// hostile input, and it does not consult the progress callback while
// recovering, so nothing in-process can bound it (a 120 KB Java file reached
// 8 GB in under a second). Isolated parses therefore run in worker
// processes: re-executions of the current binary that serve Extract
// requests over stdin/stdout. The parent kills a worker that exceeds its
// wall-clock deadline or memory budget and falls back to a whole-file unit.
//
// Protocol (little-endian, one request in flight per worker):
//
//	request:  u32 len | JSON header {rel, lang, min_lines, no_residual, timeout_ms} | u32 len | src
//	response: u32 len | JSON {units, err}
//
// The worker's unit Code fields are omitted from the wire and rebuilt by
// the parent from StartByte/EndByte (residual units carry their Code, since
// it is not a contiguous span).

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/vaktex/vakt/internal/core"
)

// WorkerEnv marks a process as a parse worker. Binaries that call
// MaybeServeWorker first thing in main become workers when it is set.
const WorkerEnv = "VAKT_AST_WORKER"

// DefaultWorkerMemory is the per-worker memory budget (resident set).
const DefaultWorkerMemory = 1 << 30

// maxWireBytes bounds any frame read from either side.
const maxWireBytes = 64 << 20

// maxConcurrentParses bounds simultaneous isolated parses (tree-sitter
// parsing is fast; the cap only matters when hostile files pile up).
const maxConcurrentParses = 4

// ErrParseLimit is returned when an isolated parse exceeded its memory
// budget or deadline and the worker was killed; the result is a whole-file
// unit.
var ErrParseLimit = errors.New("ast: parse exceeded its resource limit")

// MaybeServeWorker turns the current process into a parse worker when
// WorkerEnv is set, and never returns in that case. Call it at the start of
// main (and of TestMain in tests that use Isolated).
func MaybeServeWorker() {
	if os.Getenv(WorkerEnv) != "1" {
		return
	}
	limitSelf()
	// Exit if the parent dies mid-parse (it would otherwise run to
	// completion as an orphan): our parent pid changes when reparented.
	ppid := os.Getppid()
	go func() {
		for {
			time.Sleep(100 * time.Millisecond)
			// Reparented (to init/launchd, or a subreaper) or born orphaned.
			if p := os.Getppid(); p != ppid || p == 1 {
				os.Exit(3)
			}
		}
	}()
	// serve returns only when stdin fails; EOF is the parent closing it.
	if err := serve(os.Stdin, os.Stdout); !errors.Is(err, io.EOF) {
		fmt.Fprintln(os.Stderr, "vakt ast worker:", err)
		os.Exit(1)
	}
	os.Exit(0)
}

type request struct {
	Rel        string `json:"rel"`
	Lang       string `json:"lang"`
	MinLines   int    `json:"min_lines"`
	NoResidual bool   `json:"no_residual"`
	TimeoutMS  int64  `json:"timeout_ms"`
}

type wireUnit struct {
	core.Unit
	StartByte int    `json:"sb"`
	EndByte   int    `json:"eb"`
	Code      string `json:"code,omitempty"` // residual only
}

type response struct {
	Units   []wireUnit `json:"units"`
	Err     string     `json:"err,omitempty"`
	Timeout bool       `json:"timeout,omitempty"`
}

func serve(r io.Reader, w io.Writer) error {
	br := bufio.NewReader(r)
	bw := bufio.NewWriter(w)
	for {
		hdr, err := readFrame(br)
		if err != nil {
			return err
		}
		src, err := readFrame(br)
		if err != nil {
			return err
		}
		var req request
		if err := json.Unmarshal(hdr, &req); err != nil {
			return err
		}
		units, err := Extract(context.Background(), req.Rel, req.Lang, src,
			Options{MinLines: req.MinLines, NoResidual: req.NoResidual, ParseTimeout: time.Duration(req.TimeoutMS) * time.Millisecond})
		resp := response{Units: make([]wireUnit, len(units))}
		for i, u := range units {
			wu := wireUnit{Unit: u, StartByte: u.StartByte, EndByte: u.EndByte}
			if u.Kind == core.KindResidual {
				wu.Code = u.Code
			}
			wu.Unit.Code = ""
			resp.Units[i] = wu
		}
		if err != nil {
			resp.Err = err.Error()
			resp.Timeout = errors.Is(err, ErrParseTimeout)
		}
		b, err := json.Marshal(resp)
		if err != nil {
			return err
		}
		if err := writeFrame(bw, b); err != nil {
			return err
		}
		if err := bw.Flush(); err != nil {
			return err
		}
	}
}

func readFrame(r io.Reader) ([]byte, error) {
	var n [4]byte
	if _, err := io.ReadFull(r, n[:]); err != nil {
		return nil, err
	}
	size := binary.LittleEndian.Uint32(n[:])
	if size > maxWireBytes {
		return nil, fmt.Errorf("ast worker: frame of %d bytes", size)
	}
	b := make([]byte, size)
	_, err := io.ReadFull(r, b)
	return b, err
}

func writeFrame(w io.Writer, b []byte) error {
	if len(b) > maxWireBytes {
		return fmt.Errorf("ast worker: frame of %d bytes", len(b))
	}
	var n [4]byte
	binary.LittleEndian.PutUint32(n[:], uint32(len(b))) // #nosec G115 -- bounded above
	if _, err := w.Write(n[:]); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

// Isolated is a pool of parse worker processes.
type Isolated struct {
	exe    string
	mem    int64
	slots  chan struct{} // bounds concurrent isolated parses
	mu     sync.Mutex
	idle   []*worker
	closed bool
}

type worker struct {
	cmd  *exec.Cmd
	in   io.WriteCloser
	out  *bufio.Reader
	done chan struct{} // closed when the process has exited
}

// NewIsolated returns a worker pool that re-executes the current binary.
// memBytes is each worker's resident-memory budget (0: DefaultWorkerMemory).
// The binary must call MaybeServeWorker at the start of main.
func NewIsolated(memBytes int64) (*Isolated, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("ast: locating own executable: %w", err)
	}
	if memBytes <= 0 {
		memBytes = DefaultWorkerMemory
	}
	// Memory can overshoot the budget between polls, so bound how many
	// parses run at once (worst case ~slots x budget).
	return &Isolated{exe: exe, mem: memBytes, slots: make(chan struct{}, maxConcurrentParses)}, nil
}

// Close stops all idle workers. Busy workers stop when their parse ends.
func (iso *Isolated) Close() {
	iso.mu.Lock()
	idle := iso.idle
	iso.idle, iso.closed = nil, true
	iso.mu.Unlock()
	for _, w := range idle {
		w.stop()
	}
}

func (iso *Isolated) get() (*worker, error) {
	iso.mu.Lock()
	if n := len(iso.idle); n > 0 {
		w := iso.idle[n-1]
		iso.idle = iso.idle[:n-1]
		iso.mu.Unlock()
		return w, nil
	}
	iso.mu.Unlock()
	cmd := exec.Command(iso.exe) // #nosec G204 -- our own binary, no arguments
	cmd.Env = append(workerEnv(), WorkerEnv+"=1", fmt.Sprintf("VAKT_AST_WORKER_MEM=%d", iso.mem))
	cmd.Stderr = nil
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("ast: starting parse worker: %w", err)
	}
	w := &worker{cmd: cmd, in: in, out: bufio.NewReader(out), done: make(chan struct{})}
	go func() { _ = cmd.Wait(); close(w.done) }()
	return w, nil
}

func (iso *Isolated) put(w *worker) {
	iso.mu.Lock()
	if !iso.closed && len(iso.idle) < 64 {
		iso.idle = append(iso.idle, w)
		iso.mu.Unlock()
		return
	}
	iso.mu.Unlock()
	w.stop()
}

func (w *worker) stop() {
	_ = w.in.Close()
	select {
	case <-w.done:
	case <-time.After(time.Second):
		w.kill()
	}
}

func (w *worker) kill() {
	if w.cmd.Process != nil {
		_ = w.cmd.Process.Kill()
	}
	<-w.done
}

// Extract parses src in a worker process. It behaves like the package-level
// Extract, except that a parse which exceeds the memory budget or the
// deadline (ParseTimeout plus a grace period) kills the worker and yields a
// whole-file unit with ErrParseLimit.
func (iso *Isolated) Extract(ctx context.Context, rel, lang string, src []byte, opts Options) ([]core.Unit, error) {
	opts = opts.withDefaults()
	// Cheap cases need no isolation.
	if grammarForFile(rel, lang) == nil || len(src) == 0 || pathological(lang, src) {
		// In-process: nothing to parse, or scored whole anyway.
		return Extract(ctx, rel, lang, src, opts)
	}
	select {
	case iso.slots <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-iso.slots }()
	w, err := iso.get()
	if err != nil {
		return nil, err
	}
	hdr, _ := json.Marshal(request{Rel: rel, Lang: lang, MinLines: opts.MinLines, NoResidual: opts.NoResidual, TimeoutMS: opts.ParseTimeout.Milliseconds()})
	type result struct {
		resp response
		err  error
	}
	resc := make(chan result, 1)
	go func() {
		var r result
		defer func() { resc <- r }()
		if r.err = writeFrame(w.in, hdr); r.err != nil {
			return
		}
		if r.err = writeFrame(w.in, src); r.err != nil {
			return
		}
		b, err := readFrame(w.out)
		if err != nil {
			r.err = err
			return
		}
		r.err = json.Unmarshal(b, &r.resp)
	}()

	// Extraction after the parse is linear, so the parse timeout plus a
	// generous grace period bounds the whole request.
	deadline := time.NewTimer(opts.ParseTimeout + 5*time.Second + time.Duration(len(src)/(64<<10))*time.Second)
	defer deadline.Stop()
	poll := time.NewTicker(10 * time.Millisecond)
	defer poll.Stop()
	for {
		select {
		case r := <-resc:
			if r.err != nil {
				w.kill()
				if ctx.Err() != nil {
					return nil, ctx.Err()
				}
				// The worker died (OOM-killed, crashed): contain it.
				return []core.Unit{fileUnit(rel, lang, src)}, ErrParseLimit
			}
			// A worker that grew during a legitimate parse keeps that memory;
			// reusing it would make the next file's fate depend on which
			// worker it lands on. Retire it instead.
			if rss, ok := processRSS(w.cmd.Process.Pid); !ok || rss > iso.mem/4 {
				w.stop()
			} else {
				iso.put(w)
			}
			return rebuild(r.resp, rel, lang, src)
		case <-poll.C:
			if rss, ok := processRSS(w.cmd.Process.Pid); ok && rss > iso.mem {
				w.kill()
				<-resc
				return []core.Unit{fileUnit(rel, lang, src)}, ErrParseLimit
			}
		case <-deadline.C:
			w.kill()
			<-resc
			return []core.Unit{fileUnit(rel, lang, src)}, ErrParseLimit
		case <-ctx.Done():
			w.kill()
			<-resc
			return nil, ctx.Err()
		}
	}
}

// rebuild validates a worker response (the worker is only as trustworthy
// as tree-sitter on hostile input) and restores unit code from src.
func rebuild(resp response, rel, lang string, src []byte) ([]core.Unit, error) {
	if len(resp.Units) > MaxUnitsPerFile+1 {
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	units := make([]core.Unit, 0, len(resp.Units))
	for _, wu := range resp.Units {
		u := wu.Unit
		u.File, u.Language = rel, lang
		u.StartByte, u.EndByte = wu.StartByte, wu.EndByte
		if u.StartByte < 0 || u.EndByte > len(src) || u.StartByte > u.EndByte {
			return []core.Unit{fileUnit(rel, lang, src)}, nil
		}
		if u.Kind == core.KindResidual {
			if len(wu.Code) > len(src) {
				return []core.Unit{fileUnit(rel, lang, src)}, nil
			}
			u.Code = wu.Code
		} else {
			u.Code = string(src[u.StartByte:u.EndByte])
		}
		u.Name = clean(u.Name)
		units = append(units, u)
	}
	var err error
	switch {
	case resp.Timeout:
		err = ErrParseTimeout
	case resp.Err != "":
		err = errors.New(resp.Err)
	}
	return units, err
}

// workerEnv is the environment passed to workers: the parent's, minus
// anything that would make the worker do more than parse (credentials are
// not needed to parse).
func workerEnv() []string {
	// Allowlist: a parser needs nothing else (no credentials, no sockets).
	var env []string
	for _, kv := range os.Environ() {
		k, _, _ := strings.Cut(kv, "=")
		switch k {
		case "PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "GODEBUG", "GOMAXPROCS", "GOGC", "GOMEMLIMIT":
			env = append(env, kv)
		}
	}
	return env
}
