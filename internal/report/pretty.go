package report

import (
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
	"github.com/muesli/termenv"

	"github.com/vaktex/vakt/internal/brand"
	"github.com/vaktex/vakt/internal/labels"
)

// PrettyOptions control the terminal rendering.
type PrettyOptions struct {
	Top       int      // rows in the top findings table (0 = 25)
	Threshold float64  // flag threshold (0 = the report's threshold)
	Families  []string // only show units whose top family is one of these
	Color     bool     // ANSI colour and banner art; false gives plain text
	Width     int      // terminal width (0 = 100)
	Quiet     bool     // summary line only
}

const (
	defaultWidth = 100
	minWidth     = 60
	barWidth     = 10
)

type palette struct {
	r                                *lipgloss.Renderer
	crit, high, med, low             lipgloss.Style
	title, dim, bold, accent, border lipgloss.Style
}

func newPalette(w io.Writer, color bool) *palette {
	r := lipgloss.NewRenderer(w)
	if color {
		r.SetColorProfile(termenv.ANSI256)
	} else {
		r.SetColorProfile(termenv.Ascii)
	}
	return &palette{
		r:      r,
		crit:   r.NewStyle().Foreground(lipgloss.Color("196")).Bold(true),
		high:   r.NewStyle().Foreground(lipgloss.Color("208")),
		med:    r.NewStyle().Foreground(lipgloss.Color("220")),
		low:    r.NewStyle().Foreground(lipgloss.Color("244")),
		title:  r.NewStyle().Bold(true).Foreground(lipgloss.Color("45")),
		dim:    r.NewStyle().Foreground(lipgloss.Color("244")),
		bold:   r.NewStyle().Bold(true),
		accent: r.NewStyle().Foreground(lipgloss.Color("45")),
		border: r.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("45")).Padding(0, 1),
	}
}

func (p *palette) sev(s, thr float64) lipgloss.Style {
	switch {
	case s >= Critical:
		return p.crit
	case s >= High:
		return p.high
	case s >= thr:
		return p.med
	default:
		return p.low
	}
}

// view is the filtered set of units Pretty renders.
type view struct {
	thr      float64
	flagged  []Unit // sorted as in the report
	byFamily [len(labels.Families)]int
	files    []File // flagged files in report order
	perFile  map[string][]Unit
}

func newView(r *Report, o PrettyOptions) *view {
	v := &view{thr: o.Threshold, perFile: map[string][]Unit{}}
	if v.thr <= 0 {
		v.thr = r.Summary.Threshold
	}
	want := map[string]bool{}
	for _, f := range o.Families {
		want[f] = true
	}
	for _, u := range r.Units {
		if u.Severity < v.thr || (len(want) > 0 && !want[u.TopFamily]) {
			continue
		}
		v.flagged = append(v.flagged, u)
		if i := labels.Index(u.TopFamily); i >= 0 {
			v.byFamily[i]++
		}
		v.perFile[u.File] = append(v.perFile[u.File], u)
	}
	for _, f := range r.Files {
		if n := len(v.perFile[f.File]); n > 0 {
			f.FlaggedUnits = n
			v.files = append(v.files, f)
		}
	}
	return v
}

// Pretty renders the report for a terminal. All strings that came from the
// scanned repository are sanitised before they are written.
func Pretty(w io.Writer, r *Report, o PrettyOptions) error {
	if o.Top <= 0 {
		o.Top = 25
	}
	if o.Width <= 0 {
		o.Width = defaultWidth
	}
	o.Width = max(o.Width, minWidth)
	p := newPalette(w, o.Color)
	v := newView(r, o)

	var b strings.Builder
	if !o.Quiet {
		b.WriteString(banner(p, r, o))
		b.WriteString("\n")
		if len(v.flagged) > 0 {
			b.WriteString(topTable(p, r, v, o))
			b.WriteString("\n")
			b.WriteString(histogram(p, v, o))
			b.WriteString("\n")
			b.WriteString(fileTree(p, v, o))
			b.WriteString("\n")
		}
		if n := len(r.Scan.Skipped); n > 0 {
			b.WriteString(p.dim.Render(fmt.Sprintf("%d file%s skipped (see the JSON report for reasons)", n, plural(n))))
			b.WriteString("\n")
		}
	}
	b.WriteString(summaryLine(p, r, v, o.Width))
	b.WriteString("\n")
	_, err := io.WriteString(w, b.String())
	return err
}

func banner(p *palette, r *Report, o PrettyOptions) string {
	name := sanitize(r.Tool.Name)
	if name == "" {
		name = brand.Product
	}
	rev := sanitize(r.Model.Revision)
	if sha := sanitize(r.Model.SHA256); len(sha) >= 12 {
		rev += " " + sha[:12]
	}
	model := strings.TrimSpace(brand.ModelName + " " + rev)
	dev := sanitize(r.Model.Backend)
	if dev == "" {
		dev = "unknown backend"
	}
	if d := sanitize(r.Model.Device); d != "" && d != "none" {
		dev += " · " + d
	}
	if pr := sanitize(r.Model.Precision); pr != "" {
		dev += " · " + pr
	}
	stats := fmt.Sprintf("%s files · %s units · %s · %s tok/s",
		commas(int64(r.Scan.Files)), commas(int64(r.Scan.Units)),
		fmtDuration(time.Duration(r.Scan.DurationMS)*time.Millisecond), human(r.Scan.TokensPerSec))
	if r.Scan.CacheHits > 0 {
		stats += fmt.Sprintf(" · %s cached", commas(int64(r.Scan.CacheHits)))
	}
	if !o.Color {
		line1 := fmt.Sprintf("%s %s · %s · %s", name, sanitize(r.Tool.Version), model, dev)
		return fit(line1, o.Width) + "\n" + fit(stats, o.Width) + "\n"
	}
	head := p.title.Render("◆ "+name) + " " + p.dim.Render(sanitize(r.Tool.Version)+" · "+brand.Tagline)
	body := fit(model+" · "+dev, o.Width-4) + "\n" + fit(stats, o.Width-4)
	return p.border.Render(head+"\n"+p.dim.Render(body)) + "\n"
}

func topTable(p *palette, r *Report, v *view, o PrettyOptions) string {
	n := min(o.Top, len(v.flagged))
	var b strings.Builder
	title := fmt.Sprintf("Top findings  %d of %d flagged (severity ≥ %.2f)", n, len(v.flagged), v.thr)
	b.WriteString(p.title.Render(title) + "\n")

	rows := v.flagged[:n]
	famW := len("FAMILY")
	langW := len("LANG")
	for _, u := range rows {
		famW = max(famW, len(u.TopFamily)+5)
		langW = max(langW, ansi.StringWidth(sanitize(u.Language)))
	}
	langW = min(langW, 12)
	const sevW = barWidth + 5 // bar + space + 0.00
	rest := o.Width - 2 - sevW - famW - langW - 4*2
	showLang := rest >= 24
	if !showLang {
		rest += langW + 2
	}
	locW := max(10, rest*3/5)
	nameW := max(6, rest-locW)

	hdr := "  " + pad("SEVERITY", sevW) + "  " + pad("FAMILY", famW) + "  " + pad("LOCATION", locW) + "  " + pad("UNIT", nameW)
	if showLang {
		hdr += "  LANG"
	}
	b.WriteString(p.dim.Render(strings.TrimRight(hdr, " ")) + "\n")
	for _, u := range rows {
		st := p.sev(u.Severity, v.thr)
		sev := st.Render(bar(u.Severity)) + " " + st.Render(fmt.Sprintf("%.2f", u.Severity))
		fam := pad(fmt.Sprintf("%s %.2f", u.TopFamily, u.TopFamilyProb), famW)
		loc := pad(truncLeft(location(u), locW), locW)
		name := pad(truncRight(unitName(u), nameW), nameW)
		line := "  " + sev + "  " + fam + "  " + p.accent.Render(loc) + "  " + name
		if showLang {
			line += "  " + p.dim.Render(truncRight(sanitize(u.Language), langW))
		}
		b.WriteString(strings.TrimRight(line, " ") + "\n")
	}
	return b.String()
}

func histogram(p *palette, v *view, o PrettyOptions) string {
	var b strings.Builder
	b.WriteString(p.title.Render("Families") + p.dim.Render("  top family of each flagged unit") + "\n")
	most, nameW := 0, 0
	for i, c := range v.byFamily {
		if c > 0 {
			most = max(most, c)
			nameW = max(nameW, len(labels.Families[i]))
		}
	}
	width := min(40, o.Width-nameW-12)
	type row struct{ i, c int }
	var rows []row
	for i, c := range v.byFamily {
		if c > 0 {
			rows = append(rows, row{i, c})
		}
	}
	// Highest count first, ties in head order.
	for a := 1; a < len(rows); a++ {
		for c := a; c > 0 && rows[c].c > rows[c-1].c; c-- {
			rows[c], rows[c-1] = rows[c-1], rows[c]
		}
	}
	for _, rw := range rows {
		n := max(1, int(math.Round(float64(rw.c)/float64(most)*float64(width))))
		b.WriteString("  " + pad(labels.Families[rw.i], nameW) + "  " + p.accent.Render(strings.Repeat("▇", n)) + " " + strconv.Itoa(rw.c) + "\n")
	}
	return b.String()
}

func fileTree(p *palette, v *view, o PrettyOptions) string {
	var b strings.Builder
	b.WriteString(p.title.Render("Flagged files") + "\n")
	for _, f := range v.files {
		us := v.perFile[f.File]
		st := p.sev(f.MaxSeverity, v.thr)
		meta := fmt.Sprintf("  %d of %d unit%s", len(us), f.Units, plural(f.Units))
		path := truncLeft(sanitize(f.File), o.Width-2-5-ansi.StringWidth(meta))
		b.WriteString("  " + st.Render(fmt.Sprintf("%.2f", f.MaxSeverity)) + " " + p.bold.Render(path) + p.dim.Render(meta) + "\n")
		for i, u := range us {
			branch := "├─"
			if i == len(us)-1 {
				branch = "└─"
			}
			lines := fmt.Sprintf("%d-%d", u.StartLine, u.EndLine)
			if u.SplitOf > 0 {
				lines += fmt.Sprintf(" (%d parts)", u.SplitOf)
			}
			tail := "  " + lines + "  " + u.TopFamily
			nameW := max(8, o.Width-4-3-5-ansi.StringWidth(tail))
			b.WriteString("    " + p.dim.Render(branch) + " " + p.sev(u.Severity, v.thr).Render(fmt.Sprintf("%.2f", u.Severity)) +
				" " + truncRight(unitName(u), nameW) + p.dim.Render(tail) + "\n")
		}
	}
	return b.String()
}

func summaryLine(p *palette, r *Report, v *view, width int) string {
	if len(v.flagged) == 0 {
		msg := fmt.Sprintf("✓ All clear: no units at or above %.2f across %s file%s (%s unit%s scanned).",
			v.thr, commas(int64(r.Scan.Files)), plural(r.Scan.Files), commas(int64(r.Scan.Units)), plural(r.Scan.Units))
		return p.r.NewStyle().Foreground(lipgloss.Color("42")).Bold(true).Render(msg)
	}
	var parts []string
	type fc struct{ i, c int }
	var fcs []fc
	for i, c := range v.byFamily {
		if c > 0 {
			fcs = append(fcs, fc{i, c})
		}
	}
	for a := 1; a < len(fcs); a++ {
		for c := a; c > 0 && fcs[c].c > fcs[c-1].c; c-- {
			fcs[c], fcs[c-1] = fcs[c-1], fcs[c]
		}
	}
	head := fmt.Sprintf("%d unit%s flagged across %d file%s",
		len(v.flagged), plural(len(v.flagged)), len(v.files), plural(len(v.files)))
	// Add families while the line fits the terminal; always show at least one.
	for i, x := range fcs {
		next := append(parts, fmt.Sprintf("%s %d", labels.Families[x.i], x.c))
		tail := ""
		if i < len(fcs)-1 {
			tail = ", …"
		}
		if i > 0 && ansi.StringWidth(head+" ("+strings.Join(next, ", ")+tail+")") > width {
			parts = append(parts, "…")
			break
		}
		parts = next
	}
	msg := head + " (" + strings.Join(parts, ", ") + ")"
	return p.sev(v.flagged[0].Severity, v.thr).Bold(true).Render(msg)
}

func unitName(u Unit) string {
	n := sanitize(u.Name)
	if n == "" {
		n = "<" + sanitize(u.Kind) + ">"
	}
	return n
}

func location(u Unit) string {
	return fmt.Sprintf("%s:%d-%d", sanitize(u.File), u.StartLine, u.EndLine)
}

func bar(s float64) string {
	n := int(math.Round(math.Max(0, math.Min(1, s)) * barWidth))
	return strings.Repeat("█", n) + strings.Repeat("░", barWidth-n)
}

func pad(s string, w int) string {
	if d := w - ansi.StringWidth(s); d > 0 {
		return s + strings.Repeat(" ", d)
	}
	return s
}

func fit(s string, w int) string { return truncRight(s, w) }

func truncRight(s string, w int) string {
	if ansi.StringWidth(s) <= w {
		return s
	}
	return ansi.Truncate(s, max(1, w), "…")
}

func truncLeft(s string, w int) string {
	sw := ansi.StringWidth(s)
	if sw <= w {
		return s
	}
	return ansi.TruncateLeft(s, sw-w+1, "…")
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

func commas(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

func human(x float64) string {
	switch {
	case x >= 1e6:
		return fmt.Sprintf("%.1fM", x/1e6)
	case x >= 1e4:
		return fmt.Sprintf("%.1fk", x/1e3)
	default:
		return commas(int64(math.Round(x)))
	}
}

func fmtDuration(d time.Duration) string {
	switch {
	case d < time.Second:
		return fmt.Sprintf("%dms", d.Milliseconds())
	case d < time.Minute:
		return fmt.Sprintf("%.1fs", d.Seconds())
	default:
		return d.Round(time.Second).String()
	}
}
