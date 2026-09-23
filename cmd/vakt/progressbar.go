package main

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// downloadBar draws a single-line download bar on a TTY. It is a no-op when
// quiet or when w is not a terminal.
type downloadBar struct {
	w       io.Writer
	on      bool
	mu      sync.Mutex
	last    time.Time
	started time.Time
	drawn   bool
}

func newDownloadBar(w io.Writer, quiet bool) *downloadBar {
	return &downloadBar{w: w, on: !quiet && isTTY(w), started: time.Now()}
}

func (b *downloadBar) update(done, total int64) {
	if !b.on {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	now := time.Now()
	if done < total && now.Sub(b.last) < 100*time.Millisecond {
		return
	}
	b.last = now
	b.drawn = true
	_, _ = io.WriteString(b.w, "\r\x1b[2K"+barLine(done, total, now.Sub(b.started), termWidth(b.w)))
}

func (b *downloadBar) done() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.drawn {
		_, _ = io.WriteString(b.w, "\r\x1b[2K")
		b.drawn = false
	}
}

// barLine renders "summoning model ▕████░░░░▏ 42% 1.3/3.2 GB 85 MB/s".
func barLine(done, total int64, elapsed time.Duration, width int) string {
	frac := 0.0
	if total > 0 {
		frac = min(1, float64(done)/float64(total))
	}
	rate := ""
	if s := elapsed.Seconds(); s > 0.5 {
		rate = " " + bytesHuman(float64(done)/s) + "/s"
	}
	tail := fmt.Sprintf(" %3.0f%% %s/%s%s", frac*100, bytesHuman(float64(done)), bytesHuman(float64(total)), rate)
	head := "summoning model "
	n := max(10, min(40, width-len(head)-len(tail)-3))
	fill := int(frac * float64(n))
	return head + "▕" + strings.Repeat("█", fill) + strings.Repeat("░", n-fill) + "▏" + tail
}

func bytesHuman(b float64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.2f GB", b/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", b/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.0f KB", b/(1<<10))
	default:
		return fmt.Sprintf("%.0f B", b)
	}
}
