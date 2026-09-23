package walk

import (
	"bytes"
	"strings"
	"sync/atomic"
	"unicode/utf8"
)

// This file implements gitignore(5) matching without depending on go-git.
//
// Supported semantics:
//   - blank lines and lines starting with '#' are ignored; "\#" is a literal '#'
//   - trailing spaces are dropped unless escaped ("\ ")
//   - a leading '!' negates; "\!" is a literal '!'
//   - a trailing '/' matches directories only
//   - a pattern containing a '/' (other than a trailing one) is anchored to
//     the directory holding the ignore file; a leading '/' is dropped
//   - a pattern without '/' matches the base name at any depth
//   - '*', '?', '[...]' (with '!' or '^' negation, ranges, [:class:]) and
//     backslash escapes inside a path segment
//   - "**/" leading, "/**/" inner and "/**" trailing segments
//   - deeper ignore files override shallower ones; within a file the last
//     matching pattern wins
//
// Matching is case sensitive (git's default outside core.ignorecase).
// Everything is linear or O(len(pattern)*len(path)); nothing backtracks
// exponentially, because ignore files come from untrusted repositories.

// Limits on untrusted ignore files. Real repositories stay far below them
// (the largest .gitignore in common use has a few hundred patterns).
const (
	maxIgnoreFileBytes    = 1 << 20 // bytes read from one ignore file
	maxPatternsPerFile    = 10_000
	maxSegmentsPerPattern = 64
	maxPatternsPerWalk    = 100_000
	maxPatternRunes       = 256
	// maxMatchSteps bounds the total wildcard-matching work of one walk
	// (roughly pattern runes x name runes summed over lookups). Real trees
	// use a tiny fraction; a hostile tree that exhausts it just stops
	// having its ignore files honoured, which scans more, never less.
	maxMatchSteps = 200_000_000
)

type ignorePattern struct {
	segs     []seg // anchored: path segments; unanchored: exactly one segment
	anchored bool
	negate   bool
	dirOnly  bool
}

// seg is one pattern segment, pre-converted for matching.
type seg struct {
	raw   string
	runes []rune // nil when raw has no wildcard (plain string compare)
	star  bool   // "*" or "**" (matches any single segment)
	glob2 bool   // "**"
}

func newSeg(s string) seg {
	g := seg{raw: s, star: s == "*" || s == "**", glob2: s == "**"}
	if !g.star && strings.ContainsAny(s, "*?[\\") {
		g.runes = []rune(s)
	}
	return g
}

// match tests one path segment. nameR is name as runes, converted by the
// caller once per lookup (nil means convert on demand).
func (g *seg) match(name string, nameR []rune) bool {
	switch {
	case g.star:
		return true
	case g.runes == nil:
		return g.raw == name
	}
	if nameR == nil {
		nameR = []rune(name)
	}
	return wildmatchRunes(g.runes, nameR)
}

// ignoreFile holds the patterns of one ignore file. base is the directory the
// file lives in, relative to the repository root, with forward slashes ("" for
// the repository root itself).
type ignoreFile struct {
	base         string
	pats         []ignorePattern
	cost         int // sum of unanchored pattern rune lengths (work per base-name rune)
	anchoredCost int // sum of anchored pattern rune lengths x segments (work per path rune)
	truncated    bool
}

// ignoreChain is an immutable linked list of ignore files, deepest first.
type ignoreChain struct {
	parent *ignoreChain
	file   *ignoreFile
}

func (c *ignoreChain) push(f *ignoreFile) *ignoreChain {
	if f == nil || len(f.pats) == 0 {
		return c
	}
	return &ignoreChain{parent: c, file: f}
}

// ignored reports whether full (a slash path relative to the repository
// root) is ignored.
func (c *ignoreChain) ignored(full string, isDir bool) bool {
	return c.ignoredMetered(full, isDir, nil)
}

// matchMeter counts matching work across a walk. Once exhausted, every
// lookup reports "not ignored".
type matchMeter struct {
	left atomic.Int64
	out  atomic.Bool
}

func (m *matchMeter) spend(n int) bool {
	if m == nil {
		return true
	}
	if m.out.Load() {
		return false
	}
	if m.left.Add(int64(-n)) < 0 {
		m.out.Store(true)
		return false
	}
	return true
}

func (c *ignoreChain) ignoredMetered(full string, isDir bool, m *matchMeter) bool {
	if m != nil && m.out.Load() {
		return false
	}
	base := full
	if i := strings.LastIndexByte(full, '/'); i >= 0 {
		base = full[i+1:]
	}
	baseR := []rune(base) // converted once, not once per pattern
	for n := c; n != nil; n = n.parent {
		p, ok := relTo(n.file.base, full)
		if !ok {
			continue
		}
		pats := n.file.pats
		// Charge this file's worst-case matching cost up front: unanchored
		// patterns see the base name, anchored ones (and "**") every
		// segment of the path relative to the ignore file.
		if !m.spend(n.file.cost*(len(baseR)+1) + n.file.anchoredCost*(utf8.RuneCountInString(p)+1)) {
			return false
		}
		for i := len(pats) - 1; i >= 0; i-- {
			if pats[i].match(p, base, baseR, isDir) {
				return !pats[i].negate
			}
		}
	}
	return false
}

func relTo(base, full string) (string, bool) {
	if base == "" {
		return full, full != ""
	}
	if len(full) > len(base) && full[len(base)] == '/' && strings.HasPrefix(full, base) {
		return full[len(base)+1:], true
	}
	return "", false
}

// match tests rel (relative to the ignore file's directory); base/baseR are
// its last segment, precomputed by the caller.
func (p *ignorePattern) match(rel, base string, baseR []rune, isDir bool) bool {
	if p.dirOnly && !isDir {
		return false
	}
	if !p.anchored {
		return p.segs[0].match(base, baseR)
	}
	return matchSegments(p.segs, strings.Split(rel, "/"))
}

// matchSegments matches pattern segments against path segments, treating a
// "**" segment as zero or more directories (one or more when trailing). It
// uses dynamic programming with two rolling rows (O(len(path)) memory), so
// adversarial "**/**/**" patterns stay cheap.
func matchSegments(pat []seg, path []string) bool {
	m, n := len(pat), len(path)
	if m-countGlob2(pat) > n {
		return false // more literal segments than path segments
	}
	next := make([]bool, n+1) // row i+1
	cur := make([]bool, n+1)  // row i
	next[n] = true            // empty pattern matches empty path
	for i := m - 1; i >= 0; i-- {
		for j := n; j >= 0; j-- {
			var v bool
			switch {
			case pat[i].glob2 && i == m-1:
				v = j < n
			case pat[i].glob2:
				v = next[j] || (j < n && cur[j+1])
			default:
				v = j < n && next[j+1] && pat[i].match(path[j], nil)
			}
			cur[j] = v
		}
		next, cur = cur, next
	}
	return next[0]
}

func countGlob2(pat []seg) int {
	c := 0
	for i := range pat {
		if pat[i].glob2 {
			c++
		}
	}
	return c
}

// parseIgnore parses the contents of a gitignore-style file. budget (may be
// nil) is the number of patterns still allowed across the whole walk and is
// decremented. truncated reports that patterns were dropped because a cap was
// hit (per-file, per-walk, or an over-long pattern).
func parseIgnore(base string, data []byte, budget *atomic.Int64) (f *ignoreFile, truncated bool) {
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	if len(data) > maxIgnoreFileBytes {
		data = data[:maxIgnoreFileBytes]
		// Read was cut at the cap: the last line may be half a pattern
		// (e.g. `**/*aaaa` of a longer one), which could hide more than
		// intended. Drop it.
		if i := bytes.LastIndexByte(data, '\n'); i >= 0 {
			data = data[:i]
		} else {
			data = nil
		}
		truncated = true
	}
	f = &ignoreFile{base: base}
	for len(data) > 0 {
		if len(f.pats) >= maxPatternsPerFile {
			return f, true
		}
		var line []byte
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			line, data = data[:i], data[i+1:]
		} else {
			line, data = data, nil
		}
		line = bytes.TrimSuffix(line, []byte("\r"))
		if utf8.RuneCount(line) > maxPatternRunes {
			truncated = true
			continue
		}
		p, ok := parsePattern(string(line))
		if !ok {
			continue
		}
		if len(p.segs) > maxSegmentsPerPattern {
			truncated = true
			continue
		}
		if budget != nil && budget.Add(-1) < 0 {
			return f, true
		}
		f.pats = append(f.pats, p)
		pc := 0
		for _, sg := range p.segs {
			pc += len(sg.runes) + 1
		}
		if p.anchored {
			f.anchoredCost += pc * len(p.segs)
		} else {
			f.cost += pc
		}
	}
	return f, truncated
}

func parsePattern(line string) (ignorePattern, bool) {
	var p ignorePattern
	if line == "" || line[0] == '#' {
		return p, false
	}
	line = trimTrailingSpaces(line)
	if line == "" {
		return p, false
	}
	if line[0] == '!' {
		p.negate = true
		line = line[1:]
	}
	for strings.HasSuffix(line, "/") && !escapedAt(line, len(line)-1) {
		p.dirOnly = true
		line = line[:len(line)-1]
	}
	p.anchored = strings.Contains(line, "/")
	line = strings.TrimLeft(line, "/")
	var segs []seg
	for s := range strings.SplitSeq(line, "/") {
		if s == "" {
			continue
		}
		if s == "**" && len(segs) > 0 && segs[len(segs)-1].glob2 {
			continue
		}
		segs = append(segs, newSeg(s))
		if len(segs) > maxSegmentsPerPattern {
			break // rejected by the caller; stop allocating
		}
	}
	if len(segs) == 0 {
		return p, false
	}
	if !p.anchored && len(segs) != 1 {
		return p, false
	}
	p.segs = segs
	return p, true
}

// escapedAt reports whether s[i] is preceded by an odd number of backslashes.
func escapedAt(s string, i int) bool {
	n := 0
	for j := i - 1; j >= 0 && s[j] == '\\'; j-- {
		n++
	}
	return n%2 == 1
}

func trimTrailingSpaces(s string) string {
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t') {
		if escapedAt(s, len(s)-1) {
			break
		}
		s = s[:len(s)-1]
	}
	return s
}

// wildmatch matches one path segment against a glob segment. '*' and '?'
// never match '/', which cannot occur in a segment anyway. A malformed
// bracket expression makes the pattern match nothing, like git.
func wildmatch(pattern, name string) bool {
	g := newSeg(pattern)
	return g.match(name, nil)
}

// wildmatchRunes is wildmatch with pattern and name already converted.
func wildmatchRunes(p, n []rune) bool {
	px, nx := 0, 0
	starPx, starNx := -1, -1
	for px < len(p) || nx < len(n) {
		if px < len(p) {
			switch c := p[px]; c {
			case '*':
				for px < len(p) && p[px] == '*' {
					px++
				}
				starPx, starNx = px, nx
				continue
			case '?':
				if nx < len(n) {
					px++
					nx++
					continue
				}
			case '[':
				if nx < len(n) {
					ok, next, valid := matchClass(p, px, n[nx])
					if !valid {
						return false
					}
					if ok {
						px = next
						nx++
						continue
					}
				}
			case '\\':
				lit, width := '\\', 1
				if px+1 < len(p) {
					lit, width = p[px+1], 2
				}
				if nx < len(n) && n[nx] == lit {
					px += width
					nx++
					continue
				}
			default:
				if nx < len(n) && n[nx] == c {
					px++
					nx++
					continue
				}
			}
		}
		if starPx >= 0 && starNx < len(n) {
			starNx++
			px, nx = starPx, starNx
			continue
		}
		return false
	}
	return true
}

// matchClass matches r against the bracket expression starting at p[start]
// ('['). It returns whether r matched, the index just past ']', and whether
// the expression was well formed.
func matchClass(p []rune, start int, r rune) (matched bool, next int, valid bool) {
	i := start + 1
	negate := false
	if i < len(p) && (p[i] == '!' || p[i] == '^') {
		negate = true
		i++
	}
	first := true
	for i < len(p) {
		c := p[i]
		if c == ']' && !first {
			return matched != negate, i + 1, true
		}
		first = false
		if c == '[' && i+1 < len(p) && p[i+1] == ':' {
			end := indexRunes(p, i+2, ":]")
			if end < 0 {
				return false, 0, false
			}
			if classMatch(string(p[i+2:end]), r) {
				matched = true
			}
			i = end + 2
			continue
		}
		if c == '\\' {
			i++
			if i >= len(p) {
				return false, 0, false
			}
			c = p[i]
		}
		lo, hi := c, c
		if i+2 < len(p) && p[i+1] == '-' && p[i+2] != ']' {
			hi = p[i+2]
			i += 2
			if hi == '\\' {
				i++
				if i >= len(p) {
					return false, 0, false
				}
				hi = p[i]
			}
		}
		if lo <= r && r <= hi {
			matched = true
		}
		i++
	}
	return false, 0, false
}

func indexRunes(p []rune, from int, sub string) int {
	s := []rune(sub)
	for i := from; i+len(s) <= len(p); i++ {
		ok := true
		for k := range s {
			if p[i+k] != s[k] {
				ok = false
				break
			}
		}
		if ok {
			return i
		}
	}
	return -1
}

func classMatch(name string, r rune) bool {
	if r >= utf8.RuneSelf {
		return false
	}
	c := r
	switch name {
	case "alnum":
		return isAlpha(c) || isDigit(c)
	case "alpha":
		return isAlpha(c)
	case "blank":
		return c == ' ' || c == '\t'
	case "cntrl":
		return c < 32 || c == 127
	case "digit":
		return isDigit(c)
	case "graph":
		return c > 32 && c < 127
	case "lower":
		return c >= 'a' && c <= 'z'
	case "print":
		return c >= 32 && c < 127
	case "punct":
		return c > 32 && c < 127 && !isAlpha(c) && !isDigit(c)
	case "space":
		return c == ' ' || (c >= '\t' && c <= '\r')
	case "upper":
		return c >= 'A' && c <= 'Z'
	case "xdigit":
		return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
	}
	return false
}

func isAlpha(c rune) bool { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') }
func isDigit(c rune) bool { return c >= '0' && c <= '9' }
