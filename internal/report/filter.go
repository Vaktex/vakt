package report

import (
	"fmt"
	"sort"

	"github.com/vaktex/vakt/internal/labels"
)

// ValidateFamilies checks family names against labels.Families.
func ValidateFamilies(names []string) error {
	for _, n := range names {
		if labels.Index(n) < 0 {
			return fmt.Errorf("unknown family %q", sanitize(n))
		}
	}
	return nil
}

// Filter returns a copy of r keeping only units whose top family is in
// families (all when empty), with Flagged, Files and Summary recomputed at
// threshold (r's threshold when <= 0).
func Filter(r *Report, threshold float64, families []string) *Report {
	if threshold <= 0 {
		threshold = r.Summary.Threshold
	}
	want := map[string]bool{}
	for _, f := range families {
		want[f] = true
	}
	out := *r
	out.Units = []Unit{}
	out.Summary.Threshold = threshold
	for _, u := range r.Units {
		if len(want) > 0 && !want[u.TopFamily] {
			continue
		}
		out.Units = append(out.Units, u)
	}
	ApplyConfidence(&out, r.Summary.MinConfidence)
	sortUnits(out.Units)
	out.Scan.Units = len(out.Units)
	return &out
}

// ApplyConfidence recomputes flags and file/summary aggregates without removing
// scored units or changing scan statistics. A zero minimum disables the gate.
func ApplyConfidence(r *Report, minConfidence float64) {
	r.Summary = Summary{Threshold: r.Summary.Threshold, MinConfidence: minConfidence}
	files := map[string]*File{}
	var order []string
	for i := range r.Units {
		u := &r.Units[i]
		u.Flagged = u.Severity >= r.Summary.Threshold && u.TopFamilyProb >= minConfidence
		f := files[u.File]
		if f == nil {
			f = &File{File: u.File, Language: u.Language}
			files[u.File] = f
			order = append(order, u.File)
		}
		f.Units++
		f.MaxSeverity = max(f.MaxSeverity, u.Severity)
		if u.Flagged {
			f.FlaggedUnits++
			r.Summary.FlaggedUnits++
			if i := labels.Index(u.TopFamily); i >= 0 {
				r.Summary.ByFamily[i]++
			}
		}
	}
	r.Files = []File{}
	for _, name := range order {
		f := files[name]
		r.Files = append(r.Files, *f)
		if f.FlaggedUnits > 0 {
			r.Summary.FlaggedFiles++
		}
	}
	sort.SliceStable(r.Files, func(a, b int) bool {
		fa, fb := r.Files[a], r.Files[b]
		if fa.MaxSeverity != fb.MaxSeverity {
			return fa.MaxSeverity > fb.MaxSeverity
		}
		return fa.File < fb.File
	})
}

// MaxSeverity returns the highest unit severity in r.
func MaxSeverity(r *Report) float64 {
	m := 0.0
	for _, u := range r.Units {
		m = max(m, u.Severity)
	}
	return m
}
