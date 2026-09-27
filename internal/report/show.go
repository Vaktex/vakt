package report

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/vaktex/vakt/internal/labels"
)

// ShowOptions control ShowFinding.
type ShowOptions struct {
	Color bool
	Width int
}

// FlaggedAt returns the 1-based flagged finding at index n (severity order),
// or an error if n is out of range.
func FlaggedAt(r *Report, n int, threshold float64) (Unit, error) {
	if n < 1 {
		return Unit{}, fmt.Errorf("finding number must be >= 1")
	}
	v := newView(r, PrettyOptions{Threshold: threshold})
	if n > len(v.flagged) {
		return Unit{}, fmt.Errorf("finding %d not found (%d flagged)", n, len(v.flagged))
	}
	return v.flagged[n-1], nil
}

// ShowFinding writes the full detail for one flagged unit, including a code
// snippet read from disk when the scan root is available.
func ShowFinding(w io.Writer, r *Report, u Unit, o ShowOptions) error {
	if o.Width <= 0 {
		o.Width = defaultWidth
	}
	p := newPalette(w, o.Color)
	iss := labels.IssueFor(sanitize(u.TopFamily))
	label := SeverityLabel(u.Severity)

	var b strings.Builder
	b.WriteString(fmt.Sprintf("%s  %s\n", p.sevStyle(label).Render(label), iss.DisplayTitle()))
	b.WriteString(fmt.Sprintf("file      %s\n", locationRange(u)))
	b.WriteString(fmt.Sprintf("function  %s\n", unitName(u)))
	b.WriteString(fmt.Sprintf("score     %.4f\n", u.Severity))
	b.WriteString(fmt.Sprintf("conf      %.4f\n", u.TopFamilyProb))
	b.WriteString(fmt.Sprintf("family    %s\n", sanitize(u.TopFamily)))
	if iss.CWE != "" {
		b.WriteString(fmt.Sprintf("cwe       %s\n", iss.CWE))
	}
	if why := strings.TrimSpace(sanitize(u.Explanation)); why != "" {
		b.WriteString(fmt.Sprintf("why       %s\n", why))
	} else {
		b.WriteString(p.dim.Render("why       (none yet)") + "\n")
	}
	if snippet := readSnippet(r.Scan.Root, u); snippet != "" {
		b.WriteString("\n")
		b.WriteString(p.dim.Render(fmt.Sprintf("--- %s ---", locationRange(u))))
		b.WriteString("\n")
		b.WriteString(snippet)
		if !strings.HasSuffix(snippet, "\n") {
			b.WriteString("\n")
		}
	}
	_, err := io.WriteString(w, b.String())
	return err
}

func locationRange(u Unit) string {
	return fmt.Sprintf("%s:%d-%d", sanitize(u.File), u.StartLine, u.EndLine)
}

func readSnippet(root string, u Unit) string {
	rel := filepath.FromSlash(sanitize(u.File))
	path := rel
	if root != "" && !filepath.IsAbs(rel) {
		path = filepath.Join(root, rel)
	}
	f, err := os.Open(path) // #nosec G304 -- path from the user's own scan report
	if err != nil {
		return ""
	}
	defer f.Close()

	start, end := u.StartLine, u.EndLine
	if start < 1 {
		start = 1
	}
	if end < start {
		end = start
	}
	const maxLines = 80
	if end-start+1 > maxLines {
		end = start + maxLines - 1
	}

	var b strings.Builder
	sc := bufio.NewScanner(f)
	// Allow long lines in source.
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		if lineNo < start {
			continue
		}
		if lineNo > end {
			break
		}
		text := sc.Text()
		// Strip C0 controls so a hostile file cannot break the terminal.
		text = sanitize(text)
		fmt.Fprintf(&b, "%6d | %s\n", lineNo, text)
	}
	return b.String()
}
