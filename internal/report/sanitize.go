package report

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// sanitize makes a string taken from a scanned repository safe to print on a
// terminal: it removes ANSI CSI/OSC/DCS escape sequences, C0/C1 control
// characters, bidi overrides and isolates (U+202A-U+202E, U+2066-U+2069) and
// other zero-width/format characters (Unicode category Cf), and replaces
// invalid UTF-8 with U+FFFD. Tabs and newlines become spaces.
func sanitize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case r == utf8.RuneError && size == 1:
			b.WriteRune(utf8.RuneError)
			i++
			continue
		case r == 0x1b: // ESC: skip the whole sequence
			i = skipEscape(s, i+1)
			continue
		case r == 0x9b: // C1 CSI
			i = skipCSI(s, i+size)
			continue
		case r == 0x9d || r == 0x90 || r == 0x98 || r == 0x9e || r == 0x9f: // C1 OSC, DCS, SOS, PM, APC
			i = skipString(s, i+size)
			continue
		case r == '\t' || r == '\n' || r == '\r':
			b.WriteByte(' ')
		case unicode.IsControl(r), unicode.Is(unicode.Cf, r), isBidi(r):
			// dropped
		default:
			b.WriteRune(r)
		}
		i += size
	}
	return b.String()
}

func isBidi(r rune) bool {
	return (r >= 0x202A && r <= 0x202E) || (r >= 0x2066 && r <= 0x2069) || r == 0x200E || r == 0x200F || r == 0x061C
}

// skipEscape skips the body of an escape sequence starting after ESC.
func skipEscape(s string, i int) int {
	if i >= len(s) {
		return i
	}
	switch s[i] {
	case '[':
		return skipCSI(s, i+1)
	case ']', 'P', 'X', '^', '_':
		return skipString(s, i+1)
	default:
		// Two-byte sequences (ESC c, ESC 7, ...) plus intermediates.
		for i < len(s) && s[i] >= 0x20 && s[i] <= 0x2f {
			i++
		}
		if i < len(s) {
			i++
		}
		return i
	}
}

// skipCSI skips parameters and intermediates up to the final byte.
func skipCSI(s string, i int) int {
	for i < len(s) {
		c := s[i]
		i++
		if c < 0x20 || c > 0x3f { // final byte, or a malformed sequence
			return i
		}
	}
	return i
}

// skipString skips an OSC/DCS/SOS/PM/APC body up to BEL, ST (ESC \) or C1 ST.
func skipString(s string, i int) int {
	for i < len(s) {
		if s[i] == 0x07 {
			return i + 1
		}
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '\\' {
			return i + 2
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == 0x9c {
			return i + size
		}
		i += size
	}
	return i
}
