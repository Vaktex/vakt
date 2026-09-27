package report

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync/atomic"
	"time"

	"golang.org/x/term"
)

// Progress holds live counters the pipeline updates while it runs.
type Progress struct {
	FilesFound, FilesParsed, Units, UnitsScored, TokensTotal, TokensScored, CacheHits atomic.Int64
	Started                                                                           time.Time
}

// progressInterval is the redraw period (10x/s).
const progressInterval = 100 * time.Millisecond

// isTerminal reports whether w is a terminal.
func isTerminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	return ok && term.IsTerminal(int(f.Fd())) // #nosec G115 -- file descriptors fit in int
}

// RenderProgress draws a single-line status on w (normally stderr) until ctx
// is cancelled, then clears the line. It draws nothing when w is not a TTY.
// It blocks; run it in its own goroutine and wait for it to return before
// printing the report.
func RenderProgress(ctx context.Context, w io.Writer, p *Progress) {
	if p == nil || !isTerminal(w) {
		<-ctx.Done()
		return
	}
	renderProgressLoop(ctx, w, p, progressInterval, func() int {
		f, _ := w.(*os.File)
		width, _, err := term.GetSize(int(f.Fd())) // #nosec G115 -- file descriptors fit in int
		if err != nil || width <= 0 {
			return 80
		}
		return width
	})
}

func renderProgressLoop(ctx context.Context, w io.Writer, p *Progress, every time.Duration, width func() int) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			_, _ = io.WriteString(w, "\r\x1b[2K")
			return
		case <-t.C:
			_, _ = io.WriteString(w, "\r\x1b[2K"+truncRight(ProgressLine(p, time.Now()), width()-1))
		}
	}
}

// ProgressLine formats the status line for time now.
func ProgressLine(p *Progress, now time.Time) string {
	found, parsed := p.FilesFound.Load(), p.FilesParsed.Load()
	units, scored := p.Units.Load(), p.UnitsScored.Load()
	tokTotal, tokScored := p.TokensTotal.Load(), p.TokensScored.Load()
	elapsed := now.Sub(p.Started)
	var rate float64
	if elapsed > 0 {
		rate = float64(tokScored) / elapsed.Seconds()
	}
	eta := "--"
	if rate > 0 && tokTotal > tokScored {
		eta = fmtDuration(time.Duration(float64(tokTotal-tokScored) / rate * float64(time.Second)))
	} else if tokTotal > 0 && tokScored >= tokTotal {
		eta = "0s"
	}
	line := fmt.Sprintf("⠿ files %s/%s · units %s/%s · tokens %s/%s · %s tok/s · ETA %s",
		commas(parsed), commas(found), commas(scored), commas(units),
		human(float64(tokScored)), human(float64(tokTotal)), human(rate), eta)
	if c := p.CacheHits.Load(); c > 0 {
		line += fmt.Sprintf(" · %s cached", commas(c))
	}
	return line
}
