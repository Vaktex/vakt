// Package report turns scored units into the vakt report: a stable JSON
// document (schema version 1) and a compact terminal rendering.
package report

import (
	"math"
	"sort"
	"time"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/core"
	"github.com/vaktex/vakt/internal/labels"
)

// SchemaVersion is the version of the JSON report schema.
const SchemaVersion = "1"

// SeverityCutoffs map a unit's severity score to CRITICAL / HIGH / MEDIUM / LOW.
// Colour bands use the same thresholds (red / orange / yellow / dim).
const (
	Critical = 0.85 // red
	High     = 0.70 // orange
	Medium   = 0.50 // yellow; below Medium is LOW
)

// SeverityLabel returns CRITICAL, HIGH, MEDIUM or LOW for score.
func SeverityLabel(score float64) string {
	switch {
	case score >= Critical:
		return "CRITICAL"
	case score >= High:
		return "HIGH"
	case score >= Medium:
		return "MEDIUM"
	default:
		return "LOW"
	}
}

// Result is one scored sequence as produced by the pipeline.
type Result struct {
	Unit      core.Unit
	Tokens    int
	Scores    core.Scores
	Cached    bool
	Truncated bool
}

// Skip records a file the walk or the parser skipped.
type Skip struct {
	File   string `json:"file"`
	Reason string `json:"reason"`
}

// Meta describes the scan and the model that produced the results.
type Meta struct {
	Root      string
	StartedAt time.Time
	Duration  time.Duration
	Files     int
	CacheHits int
	Skipped   []Skip

	ModelRepo     string
	ModelRevision string
	ModelSHA      string
	Backend       string
	Device        string
	Precision     string
}

// Report is the JSON document.
type Report struct {
	SchemaVersion string  `json:"schema_version"`
	Tool          Tool    `json:"tool"`
	Model         Model   `json:"model"`
	Scan          Scan    `json:"scan"`
	Units         []Unit  `json:"units"`
	Files         []File  `json:"files"`
	Summary       Summary `json:"summary"`
}

// Tool identifies the binary that wrote the report.
type Tool struct {
	Name    string `json:"name"`
	Binary  string `json:"binary"`
	Version string `json:"version"`
	Commit  string `json:"commit"`
}

// Model identifies the weights and the device they ran on.
type Model struct {
	Repo      string `json:"repo"`
	Revision  string `json:"revision"`
	SHA256    string `json:"sha256"`
	Backend   string `json:"backend"`
	Device    string `json:"device"`
	Precision string `json:"precision"`
}

// Scan holds run statistics.
type Scan struct {
	Root         string    `json:"root"`
	StartedAt    time.Time `json:"started_at"`
	DurationMS   int64     `json:"duration_ms"`
	Files        int       `json:"files"`
	Units        int       `json:"units"`
	Tokens       int64     `json:"tokens"`
	TokensPerSec float64   `json:"tokens_per_sec"`
	CacheHits    int       `json:"cache_hits"`
	Skipped      []Skip    `json:"skipped"`
}

// Unit is one scored unit. A unit that was split to fit the context window
// appears once, spanning the original (parent) lines, with SplitOf set, its
// scores the maximum over its parts, and the parts listed in Parts.
type Unit struct {
	File          string   `json:"file"`
	Language      string   `json:"language"`
	Kind          string   `json:"kind"`
	Name          string   `json:"name,omitempty"`
	StartLine     int      `json:"start_line"`
	EndLine       int      `json:"end_line"`
	Tokens        int      `json:"tokens"`
	Severity      float64  `json:"severity"`
	Families      Families `json:"families"`
	TopFamily     string   `json:"top_family"`
	TopFamilyProb float64  `json:"top_family_prob"`
	Flagged       bool     `json:"flagged"`
	// Explanation is a short, function-specific reason for the finding.
	// Nothing fills it yet: DOM-0.8B scores but does not explain. The field
	// is in schema v1 so a model that does can add it without a bump.
	Explanation string `json:"explanation,omitempty"`
	Cached      bool   `json:"cached,omitempty"`
	Truncated   bool   `json:"truncated,omitempty"`
	SplitPart   int    `json:"split_part,omitempty"`
	SplitOf     int    `json:"split_of,omitempty"`
	Parts       []Part `json:"parts,omitempty"`
}

// Part is one piece of a split unit.
type Part struct {
	SplitPart     int     `json:"split_part"`
	StartLine     int     `json:"start_line"`
	EndLine       int     `json:"end_line"`
	Tokens        int     `json:"tokens"`
	Severity      float64 `json:"severity"`
	TopFamily     string  `json:"top_family"`
	TopFamilyProb float64 `json:"top_family_prob"`
	Cached        bool    `json:"cached,omitempty"`
	Truncated     bool    `json:"truncated,omitempty"`
}

// File aggregates the units of one file.
type File struct {
	File         string  `json:"file"`
	Language     string  `json:"language"`
	MaxSeverity  float64 `json:"max_severity"`
	FlaggedUnits int     `json:"flagged_units"`
	Units        int     `json:"units"`
}

// Summary holds the headline numbers.
type Summary struct {
	Threshold    float64      `json:"threshold"`
	FlaggedUnits int          `json:"flagged_units"`
	FlaggedFiles int          `json:"flagged_files"`
	ByFamily     FamilyCounts `json:"by_family"`
}

// round4 rounds to 4 decimals and maps NaN/Inf and out-of-range values into
// [0,1] so the JSON is always encodable.
func round4(x float64) float64 {
	if math.IsNaN(x) {
		return 0
	}
	x = math.Max(0, math.Min(1, x))
	return math.Round(x*1e4) / 1e4
}

// topFamily names the family that most clears its published cut-off
// (labels.DefaultThresholds), not simply the largest probability: families
// have very different base rates, and the model card routes on the cut-offs.
func topFamily(f Families) (string, float64) {
	i, p := labels.Top([core.NumFamilies]float64(f), labels.DefaultThresholds)
	return labels.Families[i], p
}

type groupKey struct {
	file, kind, name string
	start, end       int
}

// Build assembles a report. Split parts are grouped under their parent.
func Build(meta Meta, results []Result, threshold float64) *Report {
	r := &Report{
		SchemaVersion: SchemaVersion,
		Tool:          Tool{Name: brand.Product, Binary: brand.Binary, Version: brand.Version, Commit: brand.Commit},
		Model: Model{
			Repo: meta.ModelRepo, Revision: meta.ModelRevision, SHA256: meta.ModelSHA,
			Backend: meta.Backend, Device: meta.Device, Precision: meta.Precision,
		},
		Units: []Unit{},
		Files: []File{},
	}

	groups := map[groupKey]int{} // index into r.Units
	var tokens int64
	for _, res := range results {
		tokens += int64(res.Tokens)
		u := res.Unit
		var fam Families
		for i, p := range res.Scores.Families {
			fam[i] = round4(float64(p))
		}
		sev := round4(float64(res.Scores.Severity))
		top, topP := topFamily(fam)

		if u.SplitOf <= 0 {
			r.Units = append(r.Units, Unit{
				File: u.File, Language: u.Language, Kind: u.Kind, Name: u.Name,
				StartLine: u.StartLine, EndLine: u.EndLine, Tokens: res.Tokens,
				Severity: sev, Families: fam, TopFamily: top, TopFamilyProb: topP,
				Cached: res.Cached, Truncated: res.Truncated,
			})
			continue
		}

		ps, pe := u.ParentStartLine, u.ParentEndLine
		if ps == 0 && pe == 0 {
			ps, pe = -1, -1 // unknown parent span: widen from the parts below
		}
		k := groupKey{u.File, u.Kind, u.Name, ps, pe}
		part := Part{
			SplitPart: u.SplitPart, StartLine: u.StartLine, EndLine: u.EndLine, Tokens: res.Tokens,
			Severity: sev, TopFamily: top, TopFamilyProb: topP, Cached: res.Cached, Truncated: res.Truncated,
		}
		idx, ok := groups[k]
		if !ok {
			start, end := ps, pe
			if start < 0 {
				start, end = u.StartLine, u.EndLine
			}
			r.Units = append(r.Units, Unit{
				File: u.File, Language: u.Language, Kind: u.Kind, Name: u.Name,
				StartLine: start, EndLine: end, SplitOf: u.SplitOf, Cached: true,
			})
			idx = len(r.Units) - 1
			groups[k] = idx
		}
		p := &r.Units[idx]
		if ps < 0 {
			p.StartLine = min(p.StartLine, u.StartLine)
			p.EndLine = max(p.EndLine, u.EndLine)
		}
		p.SplitOf = max(p.SplitOf, u.SplitOf)
		p.Tokens += res.Tokens
		p.Severity = max(p.Severity, sev)
		for i := range fam {
			p.Families[i] = max(p.Families[i], fam[i])
		}
		p.Cached = p.Cached && res.Cached
		p.Truncated = p.Truncated || res.Truncated
		p.Parts = append(p.Parts, part)
	}

	files := map[string]*File{}
	for i := range r.Units {
		u := &r.Units[i]
		if len(u.Parts) > 0 {
			sort.Slice(u.Parts, func(a, b int) bool { return u.Parts[a].SplitPart < u.Parts[b].SplitPart })
			u.SplitOf = max(u.SplitOf, len(u.Parts))
			u.TopFamily, u.TopFamilyProb = topFamily(u.Families)
		}
		u.Flagged = u.Severity >= threshold
		f := files[u.File]
		if f == nil {
			f = &File{File: u.File, Language: u.Language}
			files[u.File] = f
		}
		f.Units++
		f.MaxSeverity = max(f.MaxSeverity, u.Severity)
		if u.Flagged {
			f.FlaggedUnits++
			r.Summary.FlaggedUnits++
			r.Summary.ByFamily[labels.Index(u.TopFamily)]++
		}
	}
	sortUnits(r.Units)
	for _, f := range files {
		r.Files = append(r.Files, *f)
		if f.FlaggedUnits > 0 {
			r.Summary.FlaggedFiles++
		}
	}
	sort.Slice(r.Files, func(a, b int) bool {
		fa, fb := r.Files[a], r.Files[b]
		if fa.MaxSeverity != fb.MaxSeverity {
			return fa.MaxSeverity > fb.MaxSeverity
		}
		return fa.File < fb.File
	})
	r.Summary.Threshold = threshold

	skipped := append([]Skip{}, meta.Skipped...)
	sort.SliceStable(skipped, func(a, b int) bool { return skipped[a].File < skipped[b].File })
	var tps float64
	if meta.Duration > 0 {
		tps = math.Round(float64(tokens)/meta.Duration.Seconds()*10) / 10
	}
	r.Scan = Scan{
		Root:         meta.Root,
		StartedAt:    meta.StartedAt.UTC().Truncate(time.Millisecond),
		DurationMS:   meta.Duration.Milliseconds(),
		Files:        meta.Files,
		Units:        len(r.Units),
		Tokens:       tokens,
		TokensPerSec: tps,
		CacheHits:    meta.CacheHits,
		Skipped:      skipped,
	}
	return r
}

func sortUnits(us []Unit) {
	sort.SliceStable(us, func(a, b int) bool {
		ua, ub := us[a], us[b]
		if ua.Severity != ub.Severity {
			return ua.Severity > ub.Severity
		}
		if ua.File != ub.File {
			return ua.File < ub.File
		}
		if ua.StartLine != ub.StartLine {
			return ua.StartLine < ub.StartLine
		}
		return ua.Name < ub.Name
	})
}
