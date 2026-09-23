package ast

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	ts "github.com/tree-sitter/go-tree-sitter"

	"github.com/vaktex/vakt/internal/core"
)

// Options configure Extract.
type Options struct {
	// MinLines is the minimum number of non-blank lines for a residual
	// (top-level) unit. Default 3.
	MinLines int
	// ParseTimeout bounds one file's parse. Default 5s.
	ParseTimeout time.Duration
}

// MaxUnitsPerFile caps the units one file can produce.
const MaxUnitsPerFile = 5000

// ErrParseTimeout is returned when a file takes too long to parse; the
// caller should fall back to a whole-file unit.
var ErrParseTimeout = errors.New("ast: parse timed out")

func (o Options) withDefaults() Options {
	if o.MinLines <= 0 {
		o.MinLines = 3
	}
	if o.ParseTimeout <= 0 {
		o.ParseTimeout = 5 * time.Second
	}
	return o
}

// Extract returns the scorable units of one file. src must be the file's
// bytes (invalid UTF-8 is tolerated; tree-sitter handles it byte-wise).
// Languages without a grammar, or files that fail to parse, yield a single
// whole-file unit, so every scanned file is scored.
func Extract(ctx context.Context, rel, lang string, src []byte, opts Options) (units []core.Unit, err error) {
	opts = opts.withDefaults()
	if len(bytes.TrimSpace(src)) == 0 {
		return nil, nil
	}
	g := grammarFor(lang)
	if g == nil {
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	defer func() {
		// A grammar bug must cost one file, not the scan.
		if r := recover(); r != nil {
			units, err = []core.Unit{fileUnit(rel, lang, src)}, fmt.Errorf("ast: %s: recovered: %v", rel, r)
		}
	}()

	p, _ := g.pool.Get().(*ts.Parser)
	if p == nil {
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	defer g.pool.Put(p)

	pctx, cancel := context.WithTimeout(ctx, opts.ParseTimeout)
	defer cancel()
	tree := p.ParseCtx(pctx, src, nil)
	if tree == nil {
		p.Reset() // clear the cancellation state before reuse
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return []core.Unit{fileUnit(rel, lang, src)}, ErrParseTimeout
	}
	defer tree.Close()

	x := &extractor{g: g, src: src, rel: rel, lang: lang}
	x.walk(tree.RootNode(), "", 0)
	if len(x.units) > MaxUnitsPerFile {
		// Pathological file (e.g. generated code): score it as a whole
		// (split later by SplitOversize) rather than as thousands of units.
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	if res, ok := x.residual(opts.MinLines); ok {
		x.units = append(x.units, res)
	}
	if len(x.units) == 0 {
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	return x.units, nil
}

type extractor struct {
	g       *grammar
	src     []byte
	rel     string
	lang    string
	units   []core.Unit
	covered [][2]int // byte ranges claimed by units (for the residual)
}

const maxDepth = 256

// walk visits n looking for definitions. class is the enclosing class name
// ("" at top level).
func (x *extractor) walk(n *ts.Node, class string, depth int) {
	if n == nil || depth > maxDepth || len(x.units) > MaxUnitsPerFile {
		return
	}
	kind := n.Kind()
	switch {
	case x.g.wraps[kind]:
		// A wrapper (decorator/export/template) around a definition: emit
		// the definition with the wrapper's span.
		if def := x.findDef(n); def != nil {
			if x.g.classes[def.Kind()] {
				x.classUnit(def, n, class, depth)
			} else {
				x.emit(def, n, class)
			}
			return
		}
	case x.g.funcs[kind]:
		x.emit(n, n, class)
		return
	case x.g.classes[kind]:
		x.classUnit(n, n, class, depth)
		return
	case x.g.assign[kind]:
		if fn := x.boundFunction(n); fn != nil {
			x.emitNamed(n, x.declName(n), class, kindFor(class))
			return
		}
	}
	for i := uint(0); i < n.ChildCount(); i++ {
		x.walk(n.Child(i), class, depth+1)
	}
}

// classUnit walks a class body for methods; a class with no methods becomes
// one unit of its own.
func (x *extractor) classUnit(cls, span *ts.Node, outer string, depth int) {
	name := x.nameOf(cls)
	full := name
	if outer != "" && name != "" {
		full = outer + "." + name
	} else if name == "" {
		full = outer
	}
	before := len(x.units)
	for i := uint(0); i < cls.ChildCount(); i++ {
		x.walk(cls.Child(i), full, depth+1)
	}
	if len(x.units) == before && cls.Kind() != "namespace_definition" && cls.Kind() != "namespace_declaration" &&
		cls.Kind() != "file_scoped_namespace_declaration" && cls.Kind() != "module" {
		x.emitNamed(span, full, "", core.KindClass)
	}
}

func kindFor(class string) string {
	if class != "" {
		return core.KindMethod
	}
	return core.KindFunction
}

// findDef returns the first function or class inside a wrapper.
func (x *extractor) findDef(n *ts.Node) *ts.Node {
	for i := uint(0); i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if c == nil {
			continue
		}
		k := c.Kind()
		if x.g.funcs[k] || x.g.classes[k] {
			return c
		}
		if x.g.wraps[k] || x.g.assign[k] {
			if d := x.findDef(c); d != nil {
				return d
			}
			if x.g.assign[k] && x.boundFunction(c) != nil {
				return c
			}
		}
	}
	return nil
}

// boundFunction returns the anonymous function bound by a declaration
// (const f = () => {}), or nil.
func (x *extractor) boundFunction(n *ts.Node) *ts.Node {
	for i := uint(0); i < n.NamedChildCount(); i++ {
		d := n.NamedChild(i)
		if d == nil || d.Kind() != "variable_declarator" {
			continue
		}
		if v := d.ChildByFieldName("value"); v != nil && x.g.anon[v.Kind()] {
			return v
		}
	}
	return nil
}

func (x *extractor) declName(n *ts.Node) string {
	for i := uint(0); i < n.NamedChildCount(); i++ {
		if d := n.NamedChild(i); d != nil && d.Kind() == "variable_declarator" {
			if nm := d.ChildByFieldName("name"); nm != nil {
				return x.text(nm)
			}
		}
	}
	return ""
}

func (x *extractor) emit(def, span *ts.Node, class string) {
	name := x.nameOf(def)
	if x.lang == "Terraform" || x.lang == "HCL" {
		name = x.hclName(def)
	}
	if class != "" && name != "" {
		name = class + "." + name
	}
	x.emitNamed(span, name, class, kindFor(class))
}

// emitNamed records a unit spanning span plus any directly preceding
// comments (doc comments / license headers belong to the first definition).
func (x *extractor) emitNamed(span *ts.Node, name, class, kind string) {
	if class != "" && kind == core.KindFunction {
		kind = core.KindMethod
	}
	start := x.leadingComments(span)
	sb, eb := int(start.StartByte()), int(span.EndByte()) // #nosec G115 -- byte offsets within src
	if sb < 0 || eb > len(x.src) || sb >= eb {
		return
	}
	// Extend to the start of the line so indentation is preserved.
	for sb > 0 && x.src[sb-1] != '\n' && (x.src[sb-1] == ' ' || x.src[sb-1] == '\t') {
		sb--
	}
	x.covered = append(x.covered, [2]int{sb, eb})
	x.units = append(x.units, core.Unit{
		File:      x.rel,
		Language:  x.lang,
		Kind:      kind,
		Name:      clean(name),
		StartLine: lineOf(x.src, sb),
		EndLine:   lineOf(x.src, eb-1),
		StartByte: sb,
		EndByte:   eb,
		Code:      string(x.src[sb:eb]),
	})
}

// leadingComments walks back over comment siblings immediately above n
// (no blank line in between).
func (x *extractor) leadingComments(n *ts.Node) *ts.Node {
	first := n
	for p := n.PrevSibling(); p != nil && x.g.comment[p.Kind()]; p = p.PrevSibling() {
		gap := x.src[p.EndByte():first.StartByte()]
		if bytes.Count(gap, []byte("\n")) > 1 {
			break
		}
		first = p
	}
	return first
}

// nameOf finds a definition's name: the "name" field, else the first
// identifier-like named child, else a declarator chain (C/C++).
func (x *extractor) nameOf(n *ts.Node) string {
	if nm := n.ChildByFieldName("name"); nm != nil {
		return x.text(nm)
	}
	if d := n.ChildByFieldName("declarator"); d != nil {
		for d != nil {
			if inner := d.ChildByFieldName("declarator"); inner != nil {
				d = inner
				continue
			}
			if nm := d.ChildByFieldName("name"); nm != nil {
				return x.text(nm)
			}
			return x.text(d)
		}
	}
	for i := uint(0); i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if c == nil {
			continue
		}
		switch c.Kind() {
		case "identifier", "type_identifier", "simple_identifier", "name", "constant", "field_identifier", "property_identifier":
			return x.text(c)
		}
	}
	return ""
}

// hclName renders `resource "aws_s3_bucket" "logs"` as aws_s3_bucket.logs.
func (x *extractor) hclName(n *ts.Node) string {
	var parts []string
	for i := uint(0); i < n.NamedChildCount(); i++ {
		c := n.NamedChild(i)
		if c == nil || c.Kind() == "body" || c.Kind() == "block_start" {
			break
		}
		parts = append(parts, strings.Trim(x.text(c), `"`))
	}
	return strings.Join(parts, ".")
}

func (x *extractor) text(n *ts.Node) string {
	sb, eb := int(n.StartByte()), int(n.EndByte()) // #nosec G115 -- byte offsets within src
	if sb < 0 || eb > len(x.src) || sb > eb {
		return ""
	}
	return string(x.src[sb:eb])
}

// residual returns the top-level code not covered by any unit, if it has
// at least minLines non-blank lines. Lines keep their text; gaps between
// kept regions are joined with a newline.
func (x *extractor) residual(minLines int) (core.Unit, bool) {
	if len(x.units) == 0 {
		return core.Unit{}, false // whole-file unit will be used instead
	}
	covered := make([]bool, len(x.src))
	for _, r := range x.covered {
		for i := r[0]; i < r[1] && i < len(covered); i++ {
			covered[i] = true
		}
	}
	var buf bytes.Buffer
	first, last, nonBlank := -1, -1, 0
	lineStart := 0
	for lineStart < len(x.src) {
		end := bytes.IndexByte(x.src[lineStart:], '\n')
		if end < 0 {
			end = len(x.src)
		} else {
			end += lineStart
		}
		// Keep the uncovered part of the line.
		var seg []byte
		for i := lineStart; i < end; i++ {
			if !covered[i] {
				seg = append(seg, x.src[i])
			}
		}
		if len(bytes.TrimSpace(seg)) > 0 {
			if first < 0 {
				first = lineStart
			}
			last = end
			nonBlank++
			buf.Write(seg)
			buf.WriteByte('\n')
		}
		lineStart = end + 1
	}
	if nonBlank < minLines || first < 0 {
		return core.Unit{}, false
	}
	return core.Unit{
		File:      x.rel,
		Language:  x.lang,
		Kind:      core.KindResidual,
		Name:      "<top-level>",
		StartLine: lineOf(x.src, first),
		EndLine:   lineOf(x.src, max(first, last-1)),
		StartByte: first,
		EndByte:   last,
		Code:      buf.String(),
	}, true
}

func fileUnit(rel, lang string, src []byte) core.Unit {
	return core.Unit{
		File: rel, Language: lang, Kind: core.KindFile,
		StartLine: 1, EndLine: lineOf(src, max(0, len(src)-1)),
		StartByte: 0, EndByte: len(src), Code: string(src),
	}
}

// lineOf returns the 1-based line of byte offset off.
func lineOf(src []byte, off int) int {
	if off > len(src) {
		off = len(src)
	}
	return bytes.Count(src[:off], []byte("\n")) + 1
}

// clean makes a unit name safe and short: first line only, valid UTF-8,
// at most 200 bytes.
func clean(s string) string {
	if i := strings.IndexAny(s, "\r\n"); i >= 0 {
		s = s[:i]
	}
	s = strings.ToValidUTF8(strings.TrimSpace(s), "\uFFFD")
	for len(s) > 200 {
		_, size := utf8.DecodeLastRuneInString(s)
		s = s[:len(s)-size]
	}
	return s
}
