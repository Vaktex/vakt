package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/bmatcuk/doublestar/v4"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/labels"
	"github.com/vaktex/vakt/internal/report"
)

// maxSnippetLines caps the code a single finding returns. A whole function is
// usually far shorter; the cap stops one generated file from filling the
// agent's context window.
const maxSnippetLines = 200

// defaultLimit is how many findings a list returns when the agent does not
// say. Enough to triage, small enough to read.
const defaultLimit = 20

// maxLimit caps the findings one call can return.
const maxLimit = 200

// ScanFunc runs a scan of root and returns the report. The CLI supplies this,
// because only the native build can score code.
type ScanFunc func(ctx context.Context, req ScanRequest) (*report.Report, error)

// ScanRequest is what the scan tool asks the CLI to do.
type ScanRequest struct {
	Root      string
	Threshold float64
	Include   []string
	Exclude   []string
	TopLevel  bool
	Out       string // report path to write; empty writes nothing
}

// Deps are the capabilities the tools need from the CLI.
type Deps struct {
	Scan ScanFunc
	// ReportName is the report file each scan writes inside the directory it
	// scanned, and the one the other tools read from there (vakt-report.json).
	ReportName string
	// Root, when set, limits the directories an agent may pass as dir to this
	// tree (for example the user's home or a single projects folder). Empty
	// allows any directory.
	Root string
}

// dirSchema is the `dir` property every tool that touches files takes. The
// agent names the directory on each call; everything the call reads or
// writes stays inside it.
const dirSchema = `"dir": {
      "type": "string",
      "description": "Absolute path of the project directory to work in. Every file this call reads or writes stays inside it."
    }`

// Tools builds the tool set.
//
// The shape is deliberate: one expensive tool that produces a report, and
// cheap tools that read it. An agent reviewing a change scans once, then
// queries the result as often as it likes.
func Tools(d Deps) []Tool {
	return []Tool{
		scanTool(d),
		findingsTool(d),
		findingTool(d),
		familiesTool(),
	}
}

// ---------------------------------------------------------------- scan

type scanArgs struct {
	Dir       string   `json:"dir"`
	Path      string   `json:"path"`
	Threshold *float64 `json:"threshold"`
	Include   []string `json:"include"`
	Exclude   []string `json:"exclude"`
	TopLevel  bool     `json:"top_level"`
}

func scanTool(d Deps) Tool {
	return Tool{
		Name: "scan",
		Description: "Scan a directory with " + brand.ModelName + " and return a summary of what it flagged. " +
			"This is the expensive tool: it runs a model over every function, which takes " +
			"seconds to minutes depending on the size of the tree, so scan the narrowest " +
			"path that answers the question and then use `findings` and `finding` to read " +
			"the result rather than scanning again. Writes a report file the other tools read.",
		Schema: json.RawMessage(`{
  "type": "object",
  "properties": {
    ` + dirSchema + `,
    "path": {
      "type": "string",
      "description": "Subdirectory to scan, relative to dir. Defaults to dir itself."
    },
    "threshold": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": 0,
      "maximum": 1,
      "description": "Flag functions scoring at or above this (default 0.5). Raise it to 0.7+ for a shorter, higher-confidence list."
    },
    "include": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Only scan paths matching these globs, e.g. [\"**/*.go\"]."
    },
    "exclude": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Skip paths matching these globs."
    },
    "top_level": {
      "type": "boolean",
      "description": "Also score code outside functions (imports, globals). Off by default: the model was trained on functions and scores these fragments unreliably."
    }
  },
  "required": ["dir"],
  "additionalProperties": false
}`),
		Handler: func(ctx context.Context, raw json.RawMessage) (any, error) {
			// Validate before reporting a missing engine: an agent that sent a
			// bad argument should hear about the argument, which it can fix,
			// rather than about the build, which it cannot.
			var a scanArgs
			if err := decode(raw, &a); err != nil {
				return nil, err
			}
			work, err := d.workDir(a.Dir)
			if err != nil {
				return nil, err
			}
			root, err := resolveDir(work, a.Path)
			if err != nil {
				return nil, err
			}
			reportPath := filepath.Join(work, d.reportName())
			threshold := 0.5
			if a.Threshold != nil {
				threshold = *a.Threshold
				if !(threshold > 0 && threshold <= 1) {
					return nil, ErrUsage("threshold must be in (0, 1] (got %g)", threshold)
				}
			}
			if err := checkGlobs(a.Include, a.Exclude); err != nil {
				return nil, err
			}
			if d.Scan == nil {
				return nil, ErrUsage("this build of %s has no scanning engine linked (backend: %s); "+
					"install a release build with a native engine (Metal or CUDA) to scan",
					brand.Binary, brand.Backend)
			}
			rep, err := d.Scan(ctx, ScanRequest{
				Root: root, Threshold: threshold,
				Include: a.Include, Exclude: a.Exclude,
				TopLevel: a.TopLevel, Out: reportPath,
			})
			if err != nil {
				return nil, err
			}
			return summarize(rep, reportPath), nil
		},
	}
}

// scanSummary is what scan returns: the headline numbers plus the worst
// findings, so a single call is often enough to answer "is this code safe".
type scanSummary struct {
	Root         string         `json:"root"`
	ReportPath   string         `json:"report_path,omitempty"`
	Threshold    float64        `json:"threshold"`
	FilesScanned int            `json:"files_scanned"`
	Functions    int            `json:"functions_scored"`
	Flagged      int            `json:"flagged"`
	FlaggedFiles int            `json:"flagged_files"`
	MaxSeverity  float64        `json:"max_severity"`
	ByFamily     map[string]int `json:"by_family,omitempty"`
	Top          []findingBrief `json:"top_findings"`
	Skipped      int            `json:"files_skipped"`
	Note         string         `json:"note"`
}

func summarize(rep *report.Report, path string) scanSummary {
	s := scanSummary{
		Root:         rep.Scan.Root,
		ReportPath:   path,
		Threshold:    rep.Summary.Threshold,
		FilesScanned: rep.Scan.Files,
		Functions:    rep.Scan.Units,
		Flagged:      rep.Summary.FlaggedUnits,
		FlaggedFiles: rep.Summary.FlaggedFiles,
		MaxSeverity:  report.MaxSeverity(rep),
		Skipped:      len(rep.Scan.Skipped),
		ByFamily:     familyCounts(rep),
	}
	s.Top = briefs(flagged(rep), 0, 5)
	switch {
	case s.Flagged == 0:
		s.Note = fmt.Sprintf("Nothing scored at or above %.2f. That is not a proof of safety: "+
			"the model scores one function at a time and misses flaws that span several.", s.Threshold)
	default:
		s.Note = fmt.Sprintf("%d of %d functions scored at or above %.2f. Each is a lead to verify "+
			"by reading the code, not a confirmed vulnerability. Use `finding` for the code and CWE "+
			"of one, or `findings` to page through them.", s.Flagged, s.Functions, s.Threshold)
	}
	return s
}

// ---------------------------------------------------------------- findings

type findingsArgs struct {
	Dir       string   `json:"dir"`
	Report    string   `json:"report"`
	Threshold *float64 `json:"threshold"`
	Family    []string `json:"family"`
	File      string   `json:"file"`
	Limit     int      `json:"limit"`
	Offset    int      `json:"offset"`
}

func findingsTool(d Deps) Tool {
	return Tool{
		Name: "findings",
		Description: "List what the last scan flagged, worst first, without re-scanning. " +
			"Filter by severity threshold, CWE family or file. Each entry carries the number " +
			"to pass to `finding` for the code behind it.",
		ReadOnly: true,
		Schema: json.RawMessage(`{
  "type": "object",
  "properties": {
    ` + dirSchema + `,
    "report": {"type": "string", "description": "Report file to read, relative to dir. Defaults to the one the last scan of dir wrote."},
    "threshold": {
      "type": "number", "minimum": 0, "maximum": 1,
      "description": "Only list findings scoring at or above this. Defaults to the scan's threshold. Raise it to triage the worst first."
    },
    "family": {
      "type": "array", "items": {"type": "string"},
      "description": "Only list these CWE families, e.g. [\"data_neutralization\"]. Call the families tool for the list."
    },
    "file": {"type": "string", "description": "Only list findings in this file (exact path as reported, or a suffix of it)."},
    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Maximum entries to return (default 20)."},
    "offset": {"type": "integer", "minimum": 0, "description": "Skip this many entries, to page through a long list."}
  },
  "required": ["dir"],
  "additionalProperties": false
}`),
		Handler: func(_ context.Context, raw json.RawMessage) (any, error) {
			var a findingsArgs
			if err := decode(raw, &a); err != nil {
				return nil, err
			}
			work, err := d.workDir(a.Dir)
			if err != nil {
				return nil, err
			}
			rep, path, err := loadReport(d, work, a.Report)
			if err != nil {
				return nil, err
			}
			threshold := 0.0 // 0 means "the report's own"
			if a.Threshold != nil {
				threshold = *a.Threshold
				if threshold < 0 || threshold > 1 {
					return nil, ErrUsage("threshold must be in [0, 1] (got %g)", threshold)
				}
			}
			if err := report.ValidateFamilies(a.Family); err != nil {
				return nil, ErrUsage("%v; call the `families` tool for the 18 valid names", err)
			}
			limit := a.Limit
			switch {
			case limit == 0:
				limit = defaultLimit
			case limit < 0 || limit > maxLimit:
				return nil, ErrUsage("limit must be in [1, %d] (got %d)", maxLimit, limit)
			}
			if a.Offset < 0 {
				return nil, ErrUsage("offset must not be negative (got %d)", a.Offset)
			}

			// Filter recomputes Flagged at the threshold, so the numbering
			// below matches what `finding` resolves for the same arguments.
			filtered := report.Filter(rep, threshold, a.Family)
			list := flagged(filtered)
			if a.File != "" {
				list = byFile(list, a.File)
			}
			out := findingsResult{
				ReportPath: path,
				Threshold:  filtered.Summary.Threshold,
				Total:      len(list),
				Offset:     a.Offset,
				Findings:   briefs(list, a.Offset, limit),
			}
			out.Note = pageNote(len(list), a.Offset, len(out.Findings), a.Family, a.File)
			return out, nil
		},
	}
}

type findingsResult struct {
	ReportPath string         `json:"report_path"`
	Threshold  float64        `json:"threshold"`
	Total      int            `json:"total"`
	Offset     int            `json:"offset"`
	Findings   []findingBrief `json:"findings"`
	Note       string         `json:"note,omitempty"`
}

// findingBrief is one row of a list: enough to decide what to look at, and
// the number to look at it with.
type findingBrief struct {
	// Number is the 1-based rank in severity order, for the `finding` tool.
	// It is only stable for the same report, threshold and family filter,
	// which is why those are echoed alongside it.
	Number   int     `json:"number"`
	Severity float64 `json:"severity"`
	Label    string  `json:"severity_label"`
	Issue    string  `json:"issue"`
	CWE      string  `json:"cwe,omitempty"`
	Family   string  `json:"family"`
	File     string  `json:"file"`
	Lines    string  `json:"lines"`
	Function string  `json:"function,omitempty"`
	Language string  `json:"language,omitempty"`
}

func briefs(us []report.Unit, offset, limit int) []findingBrief {
	out := []findingBrief{}
	for i := offset; i < len(us) && len(out) < limit; i++ {
		u := us[i]
		iss := labels.IssueFor(u.TopFamily)
		out = append(out, findingBrief{
			Number:   i + 1,
			Severity: u.Severity,
			Label:    report.SeverityLabel(u.Severity),
			Issue:    iss.Title,
			CWE:      iss.CWE,
			Family:   u.TopFamily,
			File:     u.File,
			Lines:    fmt.Sprintf("%d-%d", u.StartLine, u.EndLine),
			Function: u.Name,
			Language: u.Language,
		})
	}
	return out
}

func pageNote(total, offset, shown int, family []string, file string) string {
	var b strings.Builder
	if total == 0 {
		b.WriteString("No findings match")
		if len(family) > 0 {
			fmt.Fprintf(&b, " family %s", strings.Join(family, ", "))
		}
		if file != "" {
			fmt.Fprintf(&b, " in %s", file)
		}
		b.WriteString(". Lower the threshold or widen the filter to see more.")
		return b.String()
	}
	if end := offset + shown; end < total {
		fmt.Fprintf(&b, "Showing %d-%d of %d. Pass offset=%d for the next page. ", offset+1, end, total, end)
	}
	b.WriteString("Call `finding` with a number for its code and CWE. " +
		"Scores are model probabilities: verify each by reading the code.")
	return b.String()
}

// ---------------------------------------------------------------- finding

type findingArgs struct {
	Dir       string   `json:"dir"`
	Number    int      `json:"number"`
	Report    string   `json:"report"`
	Threshold *float64 `json:"threshold"`
	Family    []string `json:"family"`
	// Context is a pointer so that 0 ("just the function, no surrounding
	// lines") is distinguishable from the field being absent.
	Context *int `json:"context"`
}

func findingTool(d Deps) Tool {
	return Tool{
		Name: "finding",
		Description: "Show one finding in full: its score, CWE family, location and the source " +
			"code of the function, so it can be judged without a separate file read. " +
			"`number` comes from `findings`, and must be paired with the same threshold " +
			"and family filter that produced it.",
		ReadOnly: true,
		Schema: json.RawMessage(`{
  "type": "object",
  "properties": {
    ` + dirSchema + `,
    "number": {"type": "integer", "minimum": 1, "description": "The finding's number from the findings tool (1 is the most severe)."},
    "report": {"type": "string", "description": "Report file to read, relative to dir. Defaults to the one the last scan of dir wrote."},
    "threshold": {"type": "number", "minimum": 0, "maximum": 1, "description": "The threshold the number was listed at. Defaults to the scan's."},
    "family": {"type": "array", "items": {"type": "string"}, "description": "The family filter the number was listed with, if any."},
    "context": {"type": "integer", "minimum": 0, "maximum": 50, "description": "Extra source lines to include either side of the function (default 3; 0 for the function alone)."}
  },
  "required": ["dir", "number"],
  "additionalProperties": false
}`),
		Handler: func(_ context.Context, raw json.RawMessage) (any, error) {
			var a findingArgs
			if err := decode(raw, &a); err != nil {
				return nil, err
			}
			if a.Number < 1 {
				return nil, ErrUsage("number must be at least 1 (got %d)", a.Number)
			}
			ctxLines := 3
			if a.Context != nil {
				if *a.Context < 0 || *a.Context > 50 {
					return nil, ErrUsage("context must be in [0, 50] (got %d)", *a.Context)
				}
				ctxLines = *a.Context
			}
			work, err := d.workDir(a.Dir)
			if err != nil {
				return nil, err
			}
			rep, path, err := loadReport(d, work, a.Report)
			if err != nil {
				return nil, err
			}
			threshold := 0.0
			if a.Threshold != nil {
				threshold = *a.Threshold
				if threshold < 0 || threshold > 1 {
					return nil, ErrUsage("threshold must be in [0, 1] (got %g)", threshold)
				}
			}
			if err := report.ValidateFamilies(a.Family); err != nil {
				return nil, ErrUsage("%v; call the `families` tool for the 18 valid names", err)
			}
			filtered := report.Filter(rep, threshold, a.Family)
			list := flagged(filtered)
			if a.Number > len(list) {
				return nil, ErrUsage("finding %d does not exist: %d findings match at threshold %.2f",
					a.Number, len(list), filtered.Summary.Threshold)
			}
			u := list[a.Number-1]
			iss := labels.IssueFor(u.TopFamily)
			det := findingDetail{
				Number:      a.Number,
				ReportPath:  path,
				Severity:    u.Severity,
				Label:       report.SeverityLabel(u.Severity),
				Issue:       iss.Title,
				CWE:         iss.CWE,
				Family:      u.TopFamily,
				Confidence:  u.TopFamilyProb,
				File:        u.File,
				StartLine:   u.StartLine,
				EndLine:     u.EndLine,
				Function:    u.Name,
				Kind:        u.Kind,
				Language:    u.Language,
				Explanation: u.Explanation,
				Truncated:   u.Truncated,
				Families:    topFamilies(u, 5),
			}
			// The report records paths relative to the scan root, so read
			// through it, and keep the result inside the working directory.
			det.Code, det.CodeStartLine, det.CodeNote = snippet(work, filtered.Scan.Root, u, ctxLines)
			det.Note = fmt.Sprintf("%s is what the model predicts, not a proven bug. Check whether the "+
				"input is actually attacker-controlled and whether a caller already validates it; "+
				"if it is safe, say why and move on.", iss.DisplayTitle())
			return det, nil
		},
	}
}

type findingDetail struct {
	Number      int     `json:"number"`
	ReportPath  string  `json:"report_path"`
	Severity    float64 `json:"severity"`
	Label       string  `json:"severity_label"`
	Issue       string  `json:"issue"`
	CWE         string  `json:"cwe,omitempty"`
	Family      string  `json:"family"`
	Confidence  float64 `json:"family_confidence"`
	File        string  `json:"file"`
	StartLine   int     `json:"start_line"`
	EndLine     int     `json:"end_line"`
	Function    string  `json:"function,omitempty"`
	Kind        string  `json:"kind,omitempty"`
	Language    string  `json:"language,omitempty"`
	Explanation string  `json:"explanation,omitempty"`
	Truncated   bool    `json:"truncated,omitempty"`
	// Families are the next most likely families, for a finding whose top
	// family is arguable.
	Families      []familyProb `json:"other_families,omitempty"`
	Code          string       `json:"code,omitempty"`
	CodeStartLine int          `json:"code_start_line,omitempty"`
	CodeNote      string       `json:"code_note,omitempty"`
	Note          string       `json:"note"`
}

type familyProb struct {
	Family      string  `json:"family"`
	Issue       string  `json:"issue"`
	Probability float64 `json:"probability"`
}

// topFamilies lists the n most probable families after the top one, so an
// agent can see when a finding is borderline between, say, path traversal and
// unsafe data processing.
func topFamilies(u report.Unit, n int) []familyProb {
	type fp struct {
		i int
		p float64
	}
	all := make([]fp, 0, len(u.Families))
	for i, p := range u.Families {
		if labels.Families[i] == u.TopFamily || p <= 0 {
			continue
		}
		all = append(all, fp{i, p})
	}
	sort.Slice(all, func(a, b int) bool {
		if all[a].p != all[b].p {
			return all[a].p > all[b].p
		}
		return all[a].i < all[b].i
	})
	out := []familyProb{}
	for _, f := range all {
		if len(out) >= n {
			break
		}
		name := labels.Families[f.i]
		out = append(out, familyProb{Family: name, Issue: labels.IssueFor(name).Title, Probability: f.p})
	}
	return out
}

// ---------------------------------------------------------------- families

func familiesTool() Tool {
	return Tool{
		Name: "families",
		Description: "List the 18 CWE families " + brand.ModelName + " predicts, with the CWE id and " +
			"plain-language issue each maps to. Use it to pick a valid `family` filter or to " +
			"explain what a finding's family means.",
		ReadOnly: true,
		Schema:   json.RawMessage(`{"type": "object", "properties": {}, "additionalProperties": false}`),
		Handler: func(_ context.Context, _ json.RawMessage) (any, error) {
			out := make([]familyInfo, 0, len(labels.Families))
			for _, name := range labels.Families {
				iss := labels.IssueFor(name)
				out = append(out, familyInfo{Family: name, Issue: iss.Title, CWE: iss.CWE})
			}
			return map[string]any{
				"model":    brand.ModelName,
				"families": out,
				"note": "A finding's family is the one that most clears its own published cut-off, " +
					"not simply the largest probability: rare families have cut-offs well below 0.5, " +
					"so the raw maximum would over-report the common ones.",
			}, nil
		},
	}
}

type familyInfo struct {
	Family string `json:"family"`
	Issue  string `json:"issue"`
	CWE    string `json:"cwe,omitempty"`
}

// ---------------------------------------------------------------- helpers

// checkGlobs rejects a malformed include/exclude pattern up front, with the
// pattern named, rather than letting the walk fail once a scan is underway.
func checkGlobs(include, exclude []string) error {
	for _, g := range slices.Concat(include, exclude) {
		if g == "" {
			return ErrUsage("include/exclude patterns must not be empty")
		}
		if !doublestar.ValidatePattern(g) {
			return ErrUsage("%q is not a valid glob (doublestar syntax, e.g. \"**/*.go\")", g)
		}
	}
	return nil
}

// decode parses tool arguments strictly, so a misspelled field is reported
// rather than silently ignored.
func decode(raw json.RawMessage, v any) error {
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return ErrUsage("invalid arguments: %v", err)
	}
	return nil
}

// flagged returns the units at or above the report's threshold, in the order
// the pretty report and `vakt show` number them.
func flagged(r *report.Report) []report.Unit {
	out := make([]report.Unit, 0, len(r.Units))
	for _, u := range r.Units {
		if u.Flagged {
			out = append(out, u)
		}
	}
	return out
}

func byFile(us []report.Unit, want string) []report.Unit {
	want = filepath.ToSlash(want)
	out := []report.Unit{}
	for _, u := range us {
		f := filepath.ToSlash(u.File)
		if f == want || strings.HasSuffix(f, "/"+want) {
			out = append(out, u)
		}
	}
	return out
}

func familyCounts(r *report.Report) map[string]int {
	out := map[string]int{}
	for i, n := range r.Summary.ByFamily {
		if n > 0 {
			out[labels.Families[i]] = n
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// loadReport reads the report the agent named, or the last scan of work's.
func loadReport(d Deps, work, name string) (*report.Report, string, error) {
	if name == "" {
		name = d.reportName()
	}
	path, err := resolvePath(work, name)
	if err != nil {
		return nil, "", err
	}
	f, err := os.Open(path) // #nosec G304 -- confined to work by resolvePath
	if err != nil {
		if os.IsNotExist(err) {
			return nil, "", ErrUsage("no report at %s: call the `scan` tool first", path)
		}
		return nil, "", ErrUsage("cannot read %s: %v", path, errText(err))
	}
	defer f.Close()
	rep, err := report.ReadJSON(f)
	if err != nil {
		return nil, "", ErrUsage("%s is not a usable %s report: %v", path, brand.Binary, err)
	}
	return rep, path, nil
}

// snippet reads the source of a unit, with ctxLines either side.
//
// Paths come from a report, which is user data rather than trusted input, so
// the result is confined to the working directory: a report naming
// ../../.ssh/id_rsa must not make this server read it.
func snippet(workDir, root string, u report.Unit, ctxLines int) (code string, start int, note string) {
	rel := filepath.FromSlash(u.File)
	path := rel
	if !filepath.IsAbs(rel) {
		base := root
		if base == "" {
			base = workDir
		}
		path = filepath.Join(base, rel)
	}
	path, err := confine(workDir, path)
	if err != nil {
		return "", 0, "Source not shown: " + err.Error()
	}
	data, err := os.ReadFile(path) // #nosec G304 -- confined to the call.s dir above
	if err != nil {
		return "", 0, fmt.Sprintf("Source not shown (%s); the file may have changed since the scan.", errText(err))
	}
	lines := strings.Split(string(data), "\n")
	from := max(1, u.StartLine-ctxLines)
	to := min(len(lines), u.EndLine+ctxLines)
	if u.StartLine < 1 || from > len(lines) {
		return "", 0, "Source not shown: the report's line numbers do not match the file on disk."
	}
	if to-from+1 > maxSnippetLines {
		to = from + maxSnippetLines - 1
		note = fmt.Sprintf("Truncated to %d lines; the function continues to line %d.", maxSnippetLines, u.EndLine)
	}
	return strings.Join(lines[from-1:to], "\n"), from, note
}

func (d Deps) reportName() string {
	if d.ReportName == "" {
		return "vakt-report.json"
	}
	return d.ReportName
}

// workDir validates the directory the agent passed: it must be absolute,
// exist, and lie inside Root when one is set. Symlinks are resolved first, so
// a link cannot carry dir out of Root.
func (d Deps) workDir(dir string) (string, error) {
	if dir == "" {
		return "", ErrUsage("dir is required: pass the absolute path of the project directory")
	}
	if !filepath.IsAbs(dir) {
		return "", ErrUsage("dir must be an absolute path (got %q)", dir)
	}
	st, err := os.Stat(dir)
	if err != nil {
		return "", ErrUsage("dir %s: %v", dir, errText(err))
	}
	if !st.IsDir() {
		return "", ErrUsage("dir %s: not a directory", dir)
	}
	real := filepath.Clean(dir)
	if r, err := filepath.EvalSymlinks(dir); err == nil {
		real = r
	}
	if d.Root != "" {
		if _, err := confine(d.Root, real); err != nil {
			return "", ErrUsage("dir %s is outside %s, the only tree this server was started to allow", dir, d.Root)
		}
	}
	return real, nil
}

// resolveDir resolves a directory argument and checks it is one.
func resolveDir(workDir, name string) (string, error) {
	if name == "" {
		name = "."
	}
	path, err := resolvePath(workDir, name)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(path)
	if err != nil {
		return "", ErrUsage("cannot scan %s: %v", name, errText(err))
	}
	if !st.IsDir() {
		return "", ErrUsage("cannot scan %s: not a directory", name)
	}
	return path, nil
}

// resolvePath makes name absolute against workDir and confines it there.
func resolvePath(workDir, name string) (string, error) {
	path := name
	if !filepath.IsAbs(path) {
		path = filepath.Join(workDir, path)
	}
	return confine(workDir, path)
}

// confine returns path if it is inside workDir, following symlinks so a link
// out of the tree is caught too.
func confine(workDir, path string) (string, error) {
	base := workDir
	if base == "" {
		return filepath.Clean(path), nil // no working directory set: nothing to confine to
	}
	realBase, err := filepath.EvalSymlinks(base)
	if err != nil {
		realBase = filepath.Clean(base)
	}
	real := resolveExisting(path)
	rel, err := filepath.Rel(realBase, real)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", ErrUsage("%s is outside the working directory (%s); a call only reads and writes inside its dir", path, base)
	}
	return real, nil
}

// resolveExisting resolves symlinks in the deepest part of path that exists,
// then re-appends the components that do not.
//
// EvalSymlinks fails outright on a path with any missing component, so
// resolving the whole string is not an option: a report has yet to be written
// when its path is checked. Falling back to a plain Clean is not an option
// either -- Clean is purely lexical, so `<work>/link/a/b` with `link` a symlink
// out of the tree would look contained when it is not. Walking up to the last
// existing ancestor resolves every symlink that could redirect the path, which
// is the part that matters: the components that do not exist yet cannot be
// links, and creating one later cannot change a check made now.
func resolveExisting(path string) string {
	path = filepath.Clean(path)
	rest := ""
	for {
		if real, err := filepath.EvalSymlinks(path); err == nil {
			return filepath.Join(real, rest)
		}
		parent := filepath.Dir(path)
		if parent == path { // reached the root without finding anything that exists
			return filepath.Join(path, rest)
		}
		rest = filepath.Join(filepath.Base(path), rest)
		path = parent
	}
}

// errText unwraps an os error to its bare cause ("no such file or
// directory"), so a message does not repeat the path it already names.
func errText(err error) string {
	var pe *os.PathError
	if e, ok := err.(*os.PathError); ok {
		pe = e
		return pe.Err.Error()
	}
	return err.Error()
}
