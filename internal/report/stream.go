package report

import (
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/vaktex/vakt/internal/brand"
)

// Stream turns pipeline results into finished report units as they arrive,
// for `patrol --format jsonl`. An unsplit unit is ready as soon as it is
// scored; a split unit once its last part is, assembled exactly as Build
// would assemble it, so a streamed unit equals the one in the final report.
type Stream struct {
	threshold, minConfidence float64
	parts                    map[groupKey][]Result
}

// NewStream returns a Stream flagging units at threshold and minConfidence
// (the same gates as Build and ApplyConfidence).
func NewStream(threshold, minConfidence float64) *Stream {
	return &Stream{threshold: threshold, minConfidence: minConfidence, parts: map[groupKey][]Result{}}
}

// Add records a result and returns the unit it completes, if any.
func (s *Stream) Add(r Result) (Unit, bool) {
	group := []Result{r}
	if u := r.Unit; u.SplitOf > 0 {
		k := groupKey{u.File, u.Kind, u.Name, u.ParentStartLine, u.ParentEndLine}
		s.parts[k] = append(s.parts[k], r)
		if len(s.parts[k]) < u.SplitOf {
			return Unit{}, false
		}
		group = s.parts[k]
		delete(s.parts, k)
	}
	out := Build(Meta{}, group, s.threshold).Units[0]
	out.Flagged = out.Severity >= s.threshold && out.TopFamilyProb >= s.minConfidence
	return out, true
}

// JSONL writes `patrol --format jsonl`: one JSON object per line, so a
// reader can act on each unit as soon as it is scored.
//
//	{"type":"start","schema_version":"1","tool":{...},"root":"/abs/path"}
//	{"type":"unit","unit":{...}}                  one per unit, in scoring order
//	{"type":"progress","files_found":12,...}      at most every ProgressEvery
//	{"type":"report","report":{...}}              the full report, last
//
// Units use the report's unit schema. The final report is authoritative: it
// holds every unit (including any a reader missed) and the scan summary. Safe
// for concurrent use; each event is a single Write.
type JSONL struct {
	mu   sync.Mutex
	w    io.Writer
	err  error
	last time.Time
}

// ProgressEvery is the minimum interval between progress events.
const ProgressEvery = 250 * time.Millisecond

// NewJSONL returns a writer of JSONL events to w.
func NewJSONL(w io.Writer) *JSONL {
	return &JSONL{w: w}
}

func (j *JSONL) write(v any) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.err != nil {
		return j.err
	}
	// encoding/json escapes control characters and replaces invalid UTF-8,
	// so repo strings cannot break a line.
	b, err := json.Marshal(v)
	if err != nil {
		j.err = err
		return err
	}
	_, j.err = j.w.Write(append(b, '\n'))
	return j.err
}

// Start writes the opening event.
func (j *JSONL) Start(root string) error {
	return j.write(struct {
		Type          string `json:"type"`
		SchemaVersion string `json:"schema_version"`
		Tool          Tool   `json:"tool"`
		Root          string `json:"root"`
	}{"start", SchemaVersion, Tool{Name: brand.Product, Binary: brand.Binary, Version: brand.Version, Commit: brand.Commit}, root})
}

// Unit writes one finished unit.
func (j *JSONL) Unit(u Unit) error {
	return j.write(struct {
		Type string `json:"type"`
		Unit Unit   `json:"unit"`
	}{"unit", u})
}

// Progress writes the live counters, unless one was written less than
// ProgressEvery ago.
func (j *JSONL) Progress(p *Progress, now time.Time) error {
	j.mu.Lock()
	if now.Sub(j.last) < ProgressEvery {
		j.mu.Unlock()
		return nil
	}
	j.last = now
	j.mu.Unlock()
	return j.write(struct {
		Type         string `json:"type"`
		FilesFound   int64  `json:"files_found"`
		FilesParsed  int64  `json:"files_parsed"`
		Units        int64  `json:"units"`
		UnitsScored  int64  `json:"units_scored"`
		TokensTotal  int64  `json:"tokens_total"`
		TokensScored int64  `json:"tokens_scored"`
	}{"progress", p.FilesFound.Load(), p.FilesParsed.Load(), p.Units.Load(), p.UnitsScored.Load(), p.TokensTotal.Load(), p.TokensScored.Load()})
}

// Report writes the closing event with the full report.
func (j *JSONL) Report(r *Report) error {
	return j.write(struct {
		Type   string  `json:"type"`
		Report *Report `json:"report"`
	}{"report", r})
}
