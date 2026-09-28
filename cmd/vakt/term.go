package main

import (
	"io"
	"os"
	"strconv"

	"golang.org/x/term"
)

const fallbackWidth = 80

// isTTY reports whether w is a terminal.
func isTTY(w io.Writer) bool {
	f, ok := w.(*os.File)
	return ok && term.IsTerminal(int(f.Fd())) // #nosec G115 -- file descriptors fit in int
}

// useColor decides whether to style output written to w.
func useColor(w io.Writer, noColor bool) bool {
	if noColor || os.Getenv("NO_COLOR") != "" || os.Getenv("TERM") == "dumb" {
		return false
	}
	return isTTY(w)
}

// useHyperlinks enables OSC 8 file:// links when stdout is a capable TTY.
// Independent of colour: NO_COLOR still allows jumpable paths.
func useHyperlinks(w io.Writer) bool {
	if os.Getenv("TERM") == "dumb" || os.Getenv("VAKT_NO_HYPERLINKS") != "" {
		return false
	}
	return isTTY(w)
}

// termWidth is the width of w's terminal, $COLUMNS, or a fallback.
func termWidth(w io.Writer) int {
	if f, ok := w.(*os.File); ok {
		if width, _, err := term.GetSize(int(f.Fd())); err == nil && width > 0 { // #nosec G115 -- fds fit in int
			return width
		}
	}
	if c, err := strconv.Atoi(os.Getenv("COLUMNS")); err == nil && c > 0 {
		return c
	}
	return fallbackWidth
}
