package report

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/labels"
)

// MaxReportBytes caps the size of a report ReadJSON will accept.
const MaxReportBytes = 512 << 20

// readLimit is MaxReportBytes; tests lower it.
var readLimit int64 = MaxReportBytes

// Families holds per-family probabilities in labels.Families order. It
// marshals as a JSON object whose keys follow that order.
type Families [core.NumFamilies]float64

// MarshalJSON emits {"access_control":0.1234,...} in head order.
func (f Families) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	for i, name := range labels.Families {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Quote(name))
		b.WriteByte(':')
		b.WriteString(strconv.FormatFloat(round4(f[i]), 'f', -1, 64))
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// UnmarshalJSON accepts an object keyed by family name. Unknown names are
// rejected so a report from an incompatible model is not silently misread.
func (f *Families) UnmarshalJSON(data []byte) error {
	m := map[string]float64{}
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	*f = Families{}
	for k, v := range m {
		i := labels.Index(k)
		if i < 0 {
			return fmt.Errorf("unknown family %q", k)
		}
		f[i] = v
	}
	return nil
}

// FamilyCounts counts flagged units by top family, in labels.Families order.
type FamilyCounts [core.NumFamilies]int

// MarshalJSON emits only the non-zero counts, in head order.
func (c FamilyCounts) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteByte('{')
	first := true
	for i, name := range labels.Families {
		if c[i] == 0 {
			continue
		}
		if !first {
			b.WriteByte(',')
		}
		first = false
		b.WriteString(strconv.Quote(name))
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(c[i]))
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// UnmarshalJSON is the inverse of MarshalJSON.
func (c *FamilyCounts) UnmarshalJSON(data []byte) error {
	m := map[string]int{}
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	*c = FamilyCounts{}
	for k, v := range m {
		i := labels.Index(k)
		if i < 0 {
			return fmt.Errorf("unknown family %q", k)
		}
		c[i] = v
	}
	return nil
}

// WriteJSON writes the report as indented JSON. encoding/json escapes
// control characters and replaces invalid UTF-8, so strings taken from the
// scanned repository cannot break the document.
func WriteJSON(w io.Writer, r *Report) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// ReadJSON parses a report, rejecting other schema versions and inputs over
// MaxReportBytes.
func ReadJSON(r io.Reader) (*Report, error) {
	lr := &io.LimitedReader{R: r, N: readLimit + 1}
	data, err := io.ReadAll(lr)
	if err != nil {
		return nil, fmt.Errorf("read report: %w", err)
	}
	if int64(len(data)) > readLimit {
		return nil, fmt.Errorf("report is larger than the %d byte limit", readLimit)
	}
	var probe struct {
		SchemaVersion *string `json:"schema_version"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("parse report: %w", err)
	}
	if probe.SchemaVersion == nil {
		return nil, errors.New("not a vakt report: schema_version missing")
	}
	if *probe.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("unsupported report schema_version %q (want %q)", sanitize(*probe.SchemaVersion), SchemaVersion)
	}
	var rep Report
	if err := json.Unmarshal(data, &rep); err != nil {
		return nil, fmt.Errorf("parse report: %w", err)
	}
	if rep.Units == nil {
		rep.Units = []Unit{}
	}
	if rep.Files == nil {
		rep.Files = []File{}
	}
	// top_family is rendered verbatim in several places; a report read from
	// disk is untrusted, so it must name a known family (or be empty).
	for i := range rep.Units {
		if err := checkFamily(rep.Units[i].TopFamily); err != nil {
			return nil, fmt.Errorf("unit %d: %w", i, err)
		}
		for j := range rep.Units[i].Parts {
			if err := checkFamily(rep.Units[i].Parts[j].TopFamily); err != nil {
				return nil, fmt.Errorf("unit %d part %d: %w", i, j, err)
			}
		}
	}
	return &rep, nil
}

func checkFamily(name string) error {
	if name == "" || labels.Index(name) >= 0 {
		return nil
	}
	return fmt.Errorf("unknown top_family %q", sanitize(name))
}

// WriteFile writes the report atomically (temp file + rename) with mode 0600.
func WriteFile(path string, r *Report) (err error) {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tmp.Close()
			_ = os.Remove(tmp.Name())
		}
	}()
	if err = tmp.Chmod(0o600); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	bw := bufio.NewWriter(tmp)
	if err = WriteJSON(bw, r); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	if err = bw.Flush(); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	if err = tmp.Sync(); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	if err = os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	return nil
}
