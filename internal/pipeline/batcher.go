package pipeline

import (
	"sort"

	"github.com/vaktex/vakt/internal/core"
)

// batcher groups encoded units into padded batches under a token budget
// (longest sequence x count). Units are held in length buckets (powers of
// two) so sequences in one batch have similar lengths and little padding.
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

func bucketOf(n int) int {
	b := 64
	for b < n {
		b <<= 1
	}
	return b
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

// flush returns everything left, one or more batches per bucket.
func (b *batcher) flush() [][]core.Encoded {
	keys := make([]int, 0, len(b.buckets))
	for k := range b.buckets {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	var out [][]core.Encoded
	for _, k := range keys {
		for len(b.buckets[k]) > 0 {
			out = append(out, b.take(k))
		}
	}
	return out
}

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
