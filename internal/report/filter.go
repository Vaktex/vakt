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
	out.Summary = Summary{Threshold: threshold}
	files := map[string]*File{}
	var order []string
	for _, u := range r.Units {
		if len(want) > 0 && !want[u.TopFamily] {
			continue
		}
		u.Flagged = u.Severity >= threshold
		out.Units = append(out.Units, u)
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
			out.Summary.FlaggedUnits++
			if i := labels.Index(u.TopFamily); i >= 0 {
				out.Summary.ByFamily[i]++
			}
		}
	}
	out.Files = []File{}
	for _, name := range order {
		f := files[name]
		out.Files = append(out.Files, *f)
		if f.FlaggedUnits > 0 {
			out.Summary.FlaggedFiles++
		}
	}
	sort.SliceStable(out.Files, func(a, b int) bool {
		fa, fb := out.Files[a], out.Files[b]
		if fa.MaxSeverity != fb.MaxSeverity {
			return fa.MaxSeverity > fb.MaxSeverity
		}
		return fa.File < fb.File
	})
	sortUnits(out.Units)
	out.Scan.Units = len(out.Units)
	return &out
}

// MaxSeverity returns the highest unit severity in r.
func MaxSeverity(r *Report) float64 {
	m := 0.0
	for _, u := range r.Units {
		m = max(m, u.Severity)
	}
	return m
}
