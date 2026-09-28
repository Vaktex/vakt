package pipeline

import (
	"sort"

	"github.com/vaktex/vakt/internal/core"
)

// batcher groups encoded units into padded batches under a token budget
// (longest sequence x count). Units are held in narrow length buckets (see
// bucketOf) so sequences in one batch have similar lengths and little
// padding: padded positions cost the engine as much as real ones.
type batcher struct {
	budget  int
	maxSeqs int
	buckets map[int][]core.Encoded
	held    int // total held units
}

// holdLimit bounds how many units wait in buckets before the fullest
// bucket is flushed, keeping memory bounded on huge scans.
const holdLimit = 4096

// A single sequence is always allowed even if it alone exceeds the budget
// (the engine accepts one MaxTokens sequence); take() guarantees >= 1.
func newBatcher(budget, maxSeqs int) *batcher {
	return &batcher{budget: max(budget, 1), maxSeqs: max(maxSeqs, 1), buckets: map[int][]core.Encoded{}}
}

// bucketOf rounds a length up to its bucket: multiples of 16 up to 128,
// then eight equal steps per octave, so padding within a bucket is at most
// 12.5% (power-of-two buckets from 64 wasted up to half of every batch, and
// most of a batch of short functions).
func bucketOf(n int) int {
	if n <= 128 {
		return max(16, (n+15)/16*16)
	}
	o := 128
	for o*2 < n {
		o <<= 1
	}
	step := o / 8
	return (n + step - 1) / step * step
}

// add queues e and returns any batches that became full.
func (b *batcher) add(e core.Encoded) [][]core.Encoded {
	k := bucketOf(len(e.IDs))
	b.buckets[k] = append(b.buckets[k], e)
	b.held++
	var out [][]core.Encoded
	// A bucket is full when its units at the bucket's length fill the budget.
	if per := min(b.budget/k, b.maxSeqs); len(b.buckets[k]) >= max(per, 1) {
		out = append(out, b.take(k))
	}
	if b.held > holdLimit {
		out = append(out, b.take(b.fullest()))
	}
	return out
}

// flush returns everything left. Neighbouring buckets are merged (padding
// up to flushSlack) rather than flushed one small batch per bucket.
func (b *batcher) flush() [][]core.Encoded {
	var rest []core.Encoded
	for k, q := range b.buckets {
		rest = append(rest, q...)
		delete(b.buckets, k)
	}
	b.held = 0
	sort.Slice(rest, func(i, j int) bool { return len(rest[i].IDs) > len(rest[j].IDs) })
	var out [][]core.Encoded
	for len(rest) > 0 {
		width := len(rest[0].IDs)
		limit := max(1, min(b.budget/max(width, 1), b.maxSeqs))
		n := 1
		for n < len(rest) && n < limit && len(rest[n].IDs)*flushSlack >= width*(flushSlack-1) {
			n++
		}
		out = append(out, rest[:n:n])
		rest = rest[n:]
	}
	return out
}

// flushSlack bounds padding in flushed batches: every unit is at least
// (flushSlack-1)/flushSlack of the batch's longest.
const flushSlack = 4

// take removes one batch from bucket k: as many units as fit the budget
// when padded to the longest unit taken.
func (b *batcher) take(k int) []core.Encoded {
	q := b.buckets[k]
	// Longest first so the padded width is known up front.
	sort.Slice(q, func(i, j int) bool { return len(q[i].IDs) > len(q[j].IDs) })
	width := len(q[0].IDs)
	n := max(1, min(len(q), b.budget/max(width, 1), b.maxSeqs))
	batch := append([]core.Encoded(nil), q[:n]...)
	b.buckets[k] = q[n:]
	if len(b.buckets[k]) == 0 {
		delete(b.buckets, k)
	}
	b.held -= n
	return batch
}

func (b *batcher) fullest() int {
	best, bestN := 0, -1
	for k, q := range b.buckets {
		if len(q)*k > bestN {
			best, bestN = k, len(q)*k
		}
	}
	return best
}
