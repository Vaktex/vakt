package report

import (
	"bufio"
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Streaming every result must yield exactly the units Build assembles, with
// the same flags ApplyConfidence sets, and a split unit only once complete.
func TestStreamMatchesBuild(t *testing.T) {
	meta, res := fixture()
	const threshold, minConf = 0.5, 0.75
	s := NewStream(threshold, minConf)
	var got []Unit
	splitSeen := 0
	for _, r := range res {
		u, ok := s.Add(r)
		if r.Unit.SplitOf > 0 {
			splitSeen++
			if ok != (splitSeen == r.Unit.SplitOf) {
				t.Fatalf("split part %d/%d: ready=%v", splitSeen, r.Unit.SplitOf, ok)
			}
		} else if !ok {
			t.Fatalf("unsplit unit %s not ready", r.Unit.Name)
		}
		if ok {
			got = append(got, u)
		}
	}
	want := Build(meta, res, threshold)
	ApplyConfidence(want, minConf)
	sortUnits(got)
	if !reflect.DeepEqual(got, want.Units) {
		t.Fatalf("streamed units differ from Build:\n got %+v\nwant %+v", got, want.Units)
	}
}

func TestJSONLEvents(t *testing.T) {
	meta, res := fixture()
	var buf bytes.Buffer
	j := NewJSONL(&buf)
	if err := j.Start("/src/app"); err != nil {
		t.Fatal(err)
	}
	evil := res[0]
	evil.Unit.Name = "x\n\x1b[31m\u202e"
	if u, ok := NewStream(0.5, 0).Add(evil); !ok || j.Unit(u) != nil {
		t.Fatal("unit")
	}
	p := &Progress{}
	p.Units.Store(7)
	now := time.Now()
	_ = j.Progress(p, now)
	_ = j.Progress(p, now.Add(ProgressEvery/2)) // throttled
	if err := j.Report(Build(meta, res, 0.5)); err != nil {
		t.Fatal(err)
	}

	var types []string
	sc := bufio.NewScanner(&buf)
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		var ev struct {
			Type   string          `json:"type"`
			Unit   json.RawMessage `json:"unit"`
			Report json.RawMessage `json:"report"`
			Units  int             `json:"units"`
		}
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			t.Fatalf("line %q: %v", sc.Text(), err)
		}
		types = append(types, ev.Type)
		switch ev.Type {
		case "progress":
			if ev.Units != 7 {
				t.Errorf("progress units = %d", ev.Units)
			}
		case "report":
			if _, err := ReadJSON(bytes.NewReader(ev.Report)); err != nil {
				t.Errorf("report event is not a valid report: %v", err)
			}
		}
	}
	if got := strings.Join(types, ","); got != "start,unit,progress,report" {
		t.Errorf("events = %s", got)
	}
}
