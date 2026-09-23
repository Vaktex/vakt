package walk

import (
	"bytes"
	"strings"
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

// maxIgnoreFileBytes caps how much of one ignore file is read.
const maxIgnoreFileBytes = 1 << 20

type ignorePattern struct {
	segs     []string // anchored: path segments; unanchored: exactly one segment
	anchored bool
	negate   bool
	dirOnly  bool
}

// ignoreFile holds the patterns of one ignore file. base is the directory the
// file lives in, relative to the repository root, with forward slashes ("" for
// the repository root itself).
type ignoreFile struct {
	base string
	pats []ignorePattern
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
	for n := c; n != nil; n = n.parent {
		p, ok := relTo(n.file.base, full)
		if !ok {
			continue
		}
		pats := n.file.pats
		for i := len(pats) - 1; i >= 0; i-- {
			if pats[i].match(p, isDir) {
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

func (p *ignorePattern) match(rel string, isDir bool) bool {
	if p.dirOnly && !isDir {
		return false
	}
	if !p.anchored {
		name := rel
		if i := strings.LastIndexByte(rel, '/'); i >= 0 {
			name = rel[i+1:]
		}
		return wildmatch(p.segs[0], name)
	}
	return matchSegments(p.segs, strings.Split(rel, "/"))
}

// matchSegments matches pattern segments against path segments, treating a
// "**" segment as zero or more directories (one or more when trailing). It
// uses dynamic programming so adversarial "**/**/**" patterns stay cheap.
func matchSegments(pat, path []string) bool {
	m, n := len(pat), len(path)
	w := n + 1
	dp := make([]bool, (m+1)*w)
	dp[m*w+n] = true
	for i := m - 1; i >= 0; i-- {
		for j := n; j >= 0; j-- {
			var v bool
			switch {
			case pat[i] == "**" && i == m-1:
				v = j < n
			case pat[i] == "**":
				v = dp[(i+1)*w+j] || (j < n && dp[i*w+j+1])
			default:
				v = j < n && dp[(i+1)*w+j+1] && wildmatch(pat[i], path[j])
			}
			dp[i*w+j] = v
		}
	}
	return dp[0]
}

// parseIgnore parses the contents of a gitignore-style file.
func parseIgnore(base string, data []byte) *ignoreFile {
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	f := &ignoreFile{base: base}
	for len(data) > 0 {
		var line []byte
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			line, data = data[:i], data[i+1:]
		} else {
			line, data = data, nil
		}
		if p, ok := parsePattern(string(bytes.TrimSuffix(line, []byte("\r")))); ok {
			f.pats = append(f.pats, p)
		}
	}
	return f
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
	var segs []string
	for s := range strings.SplitSeq(line, "/") {
		if s == "" {
			continue
		}
		if s == "**" && len(segs) > 0 && segs[len(segs)-1] == "**" {
			continue
		}
		segs = append(segs, s)
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
	if pattern == "*" || pattern == "**" {
		return true
	}
	if !strings.ContainsAny(pattern, "*?[\\") {
		return pattern == name
	}
	p := []rune(pattern)
	n := []rune(name)
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
