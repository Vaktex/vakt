//go:build mlx

package engine

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/vaktex/vakt/internal/engine/mlx"
)

// profiler attributes forward-pass time to stages when VAKT_PROFILE=1. Each
// mark evaluates the stage's output and charges the wall time since the
// previous mark to it, so the forward is serialised stage by stage: totals
// run slower than an unprofiled scan, but the shares show where time goes.
type profiler struct {
	last   time.Time
	stages map[string]time.Duration
	calls  int
	real   int64 // real tokens scored
	padded int64 // padded tokens computed
	total  time.Duration
}

func newProfilerFromEnv() *profiler {
	if os.Getenv("VAKT_PROFILE") == "" {
		return nil
	}
	return &profiler{stages: map[string]time.Duration{}}
}

func (p *profiler) start() {
	if p != nil {
		p.last = time.Now()
	}
}

// mark evaluates a and charges the time since the previous mark to stage.
func (p *profiler) mark(x *mlx.Ctx, stage string, a *mlx.Array) *mlx.Array {
	if p == nil {
		return a
	}
	_ = x.Eval(a) // errors surface through x.Err as usual
	now := time.Now()
	p.stages[stage] += now.Sub(p.last)
	p.last = now
	return a
}

func (p *profiler) batch(real, padded int, d time.Duration) {
	if p == nil {
		return
	}
	p.calls++
	p.real += int64(real)
	p.padded += int64(padded)
	p.total += d
}

func (p *profiler) report(w io.Writer, info string) {
	if p == nil || p.calls == 0 {
		return
	}
	var sum time.Duration
	names := make([]string, 0, len(p.stages))
	for n, d := range p.stages {
		names = append(names, n)
		sum += d
	}
	sort.Slice(names, func(i, j int) bool { return p.stages[names[i]] > p.stages[names[j]] })
	var b strings.Builder
	fmt.Fprintf(&b, "vakt profile (%s): %d batches, %d real / %d padded tokens (%.1f%% padding), %.0f real tok/s while profiling\n",
		info, p.calls, p.real, p.padded, 100*float64(p.padded-p.real)/float64(max(p.padded, 1)), float64(p.real)/p.total.Seconds())
	for _, n := range names {
		d := p.stages[n]
		fmt.Fprintf(&b, "  %-16s %9.1f ms  %5.1f%%\n", n, float64(d.Microseconds())/1000, 100*d.Seconds()/sum.Seconds())
	}
	fmt.Fprintf(&b, "  %-16s %9.1f ms  (Score wall time; the rest is host prep and readback)\n", "total", float64(p.total.Microseconds())/1000)
	_, _ = io.WriteString(w, b.String())
}
