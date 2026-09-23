package ast

import (
	"bytes"
	"unicode/utf8"

	"github.com/vaktex/vakt/internal/core"
)

// SplitOversize cuts u into consecutive parts that each fit maxTokens, where
// countTokens(code) returns the token count of the fully rendered prompt for
// that code (see CONTRACTS.md). Units that fit are returned unchanged.
//
// Cuts prefer blank lines (paragraph / statement-group boundaries, which in
// practice are AST sibling boundaries), then any line boundary, then a
// UTF-8 rune boundary inside a pathological line. Parts cover u.Code exactly
// and in order.
func SplitOversize(u core.Unit, _ []byte, countTokens func(code string) int, maxTokens int) []core.Unit {
	if len(u.Code) <= maxTokens*16 && countTokens(u.Code) <= maxTokens {
		return []core.Unit{u}
	}
	parts := splitCode(u.Code, countTokens, maxTokens)
	out := make([]core.Unit, 0, len(parts))
	line, off := u.StartLine, u.StartByte
	for i, p := range parts {
		n := bytes.Count([]byte(p), []byte("\n"))
		end := line + n
		if len(p) > 0 && p[len(p)-1] == '\n' {
			end--
		}
		out = append(out, core.Unit{
			File: u.File, Language: u.Language, Kind: u.Kind, Name: u.Name,
			StartLine: line, EndLine: max(line, end),
			StartByte: off, EndByte: off + len(p),
			Code:      p,
			SplitPart: i + 1, SplitOf: len(parts),
			ParentStartLine: u.StartLine, ParentEndLine: u.EndLine,
		})
		line += n
		off += len(p)
	}
	return out
}

// splitCode greedily packs lines into parts under budget, cutting at the
// best boundary available.
func splitCode(code string, count func(string) int, budget int) []string {
	var parts []string
	rest := code
	for len(rest) > 0 {
		// Only test the whole remainder when it could plausibly fit (a token
		// is at least one byte, so > budget*16 bytes essentially never does);
		// re-tokenizing a multi-MiB remainder before every cut was quadratic.
		if len(rest) <= budget*16 && count(rest) <= budget {
			parts = append(parts, rest)
			break
		}
		cut := bestCut(rest, count, budget)
		parts = append(parts, rest[:cut])
		rest = rest[cut:]
	}
	return parts
}

// bestCut returns the largest prefix length of s that fits the budget,
// preferring a blank-line boundary, then a line boundary, then a rune
// boundary. It always makes progress (returns >= 1).
func bestCut(s string, count func(string) int, budget int) int {
	// Binary search the largest fitting byte prefix (token count is
	// monotone in prefix length up to tokenizer merges, which only ever
	// lower counts; the search is conservative).
	//
	// Bound the search window first so each probe tokenizes a prefix near
	// the budget, not the whole remaining unit: start at ~8 bytes per token
	// and double until the prefix no longer fits (or covers s). Probing a
	// 2 MiB unit then costs O(log) tokenizations of ~budget-sized text.
	lo, hi := 0, min(len(s), max(budget*8, 64))
	for hi < len(s) && count(s[:hi]) <= budget {
		lo = hi
		hi = min(len(s), hi*2)
	}
	// Narrow with one proportional estimate: prefix length scales ~linearly
	// with token count, so aim 3% under the budget from the window's
	// measured density. If it fits, the answer lies in [est, est+band];
	// the binary search below then needs only a few probes.
	if hi > lo+1024 {
		if n := count(s[:hi]); n > budget {
			est := int(float64(hi) * float64(budget) / float64(n) * 0.97)
			if est > lo && est < hi && count(s[:est]) <= budget {
				lo = est
				if band := est + max(hi/32, 1024); band < hi && count(s[:band]) > budget {
					hi = band
				}
			}
		}
	}
	// Coarse is fine: the final cut snaps back to a line boundary anyway.
	for lo < hi && hi-lo > 64 {
		mid := (lo + hi + 1) / 2
		if count(s[:mid]) <= budget {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	limit := lo
	if limit == 0 {
		// Not even one rune fits (budget is tiny): take one rune.
		_, size := utf8.DecodeRuneInString(s)
		return max(1, size)
	}
	window := s[:limit]
	// Only accept a boundary in the back half so parts stay balanced.
	minCut := limit / 2
	if i := bytes.LastIndex([]byte(window), []byte("\n\n")); i >= minCut {
		return i + 2
	}
	if i := bytes.LastIndexByte([]byte(window), '\n'); i >= minCut {
		return i + 1
	}
	// Pathological long line: cut on a rune boundary.
	for limit > 0 && !utf8.RuneStart(s[limit]) {
		limit--
	}
	return max(1, limit)
}
