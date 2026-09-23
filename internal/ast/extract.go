package ast

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode"
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
	g := grammarForFile(rel, lang)
	if g == nil {
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	defer func() {
		// A grammar bug must cost one file, not the scan.
		if r := recover(); r != nil {
			units, err = []core.Unit{fileUnit(rel, lang, src)}, fmt.Errorf("ast: %s: recovered: %v", rel, r)
		}
	}()

	// tree-sitter's error recovery on deeply unbalanced input is
	// superlinear in time AND memory and does not call the progress
	// callback while recovering (a 64 KiB Java file of `A<` reached 9 GB).
	// Such files are not meaningful code: score them as a whole instead.
	if pathological(src) {
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	p, _ := g.pool.Get().(*ts.Parser)
	if p == nil {
		return []core.Unit{fileUnit(rel, lang, src)}, nil
	}
	// Bound the parse with tree-sitter's progress callback rather than
	// ParseCtx: ParseCtx cancels through a flag written from a watcher
	// goroutine that can outlive the parse and race with the parser's
	// reuse from the pool.
	deadline := time.Now().Add(opts.ParseTimeout)
	timedOut := false
	popts := &ts.ParseOptions{ProgressCallback: func(ts.ParseState) bool {
		if ctx.Err() != nil || time.Now().After(deadline) {
			timedOut = true
			return true // stop parsing
		}
		return false
	}}
	tree := p.ParseWithOptions(func(off int, _ ts.Point) []byte {
		if off >= len(src) {
			return nil
		}
		return src[off:]
	}, nil, popts)
	if tree == nil || timedOut {
		if tree != nil {
			tree.Close()
		}
		// Never reuse a parser whose parse was stopped: ts_parser_reset does
		// not clear its canceled-balancing state, and the next parse on it
		// aborts the process (a C assert recover cannot catch).
		p.Close()
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return []core.Unit{fileUnit(rel, lang, src)}, ErrParseTimeout
	}
	defer g.pool.Put(p)
	defer tree.Close()

	x := &extractor{g: g, src: src, rel: rel, lang: lang, root: tree.RootNode()}
	x.walk(x.root, "", 0)
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

// Bracket excess limits above which a file is scored whole. Measured on
// llama.cpp and juice-shop project code: ( [ { never exceed ~60 open at
// once; '<' (also the less-than operator) peaks at 624. Cython/generated C
// in vendored virtualenvs exceeds them and is not project code anyway.
const (
	maxUnbalanced      = 1000 // ( [ {
	maxUnbalancedAngle = 4000 // <
	maxNesting         = 512  // ( [ { together
)

// pathological reports whether src has bracket structure that would drive
// tree-sitter's error recovery into superlinear time/memory.
func pathological(src []byte) bool {
	var open [4]int // ( [ { <
	var worst int
	for _, c := range src {
		switch c {
		case '(':
			open[0]++
		case ')':
			open[0]--
		case '[':
			open[1]++
		case ']':
			open[1]--
		case '{':
			open[2]++
		case '}':
			open[2]--
		case '<':
			open[3]++
		case '>':
			open[3]--
		default:
			continue
		}
		depth := open[0] + open[1] + open[2]
		if depth > maxNesting {
			return true
		}
		worst = max(worst, open[0], open[1], open[2])
		if worst > maxUnbalanced || open[3] > maxUnbalancedAngle {
			return true
		}
	}
	return false
}

type extractor struct {
	g       *grammar
	src     []byte
	rel     string
	lang    string
	units   []core.Unit
	covered [][2]int // byte ranges claimed by units (for the residual)
	lines   []int    // byte offset of each line start (built lazily)
	root    *ts.Node
}

// maxErrorUnitLines: a definition containing a parse error that spans more
// than this is assumed to have swallowed its neighbours (e.g. an unknown
// export macro before a prototype) and is descended into instead.
const maxErrorUnitLines = 400

// line returns the 1-based line of byte offset off in O(log lines).
func (x *extractor) line(off int) int {
	if x.lines == nil {
		x.lines = append(x.lines, 0)
		for i, c := range x.src {
			if c == '\n' {
				x.lines = append(x.lines, i+1)
			}
		}
	}
	return sort.SearchInts(x.lines, off+1)
}

const maxDepth = 256

// walk visits n looking for definitions. class is the enclosing class name
// ("" at top level).
func (x *extractor) walk(n *ts.Node, class string, depth int) {
	if n == nil || depth > maxDepth || len(x.units) > MaxUnitsPerFile {
		return
	}
	if !n.IsNamed() {
		return // keywords and punctuation (e.g. the `class` token)
	}
	kind := n.Kind()
	switch {
	case x.g.wraps[kind]:
		// A wrapper (decorator/export/template) around a definition: emit
		// the definition with the wrapper's span.
		if def := x.findDef(n); def != nil {
			switch {
			case x.g.classes[def.Kind()]:
				x.classUnit(def, n, class, depth)
			case x.g.assign[def.Kind()]:
				x.emitNamed(n, x.declName(def), class, kindFor(class))
			default:
				x.emit(def, n, class)
			}
			return
		}
	case x.g.funcs[kind]:
		if x.swallowed(n) {
			break // recurse into the children instead
		}
		if x.isIIFEBody(n) {
			break
		}
		if (kind == "property_declaration" || kind == "indexer_declaration") && n.EndPosition().Row-n.StartPosition().Row < 2 {
			return // auto-property: stays in the residual
		}
		x.emit(n, n, class)
		return
	case x.g.classes[kind]:
		if !x.hasBody(n) {
			return // forward declaration or `struct X *p` type reference
		}
		if x.swallowed(n) {
			break
		}
		x.classUnit(n, n, class, depth)
		return
	case x.g.assign[kind]:
		if fn := x.boundFunction(n); fn != nil {
			x.emitNamed(n, x.declName(n), class, kindFor(class))
			return
		}
	case x.g.anon[kind]:
		// An anonymous function not bound by a declaration: a callback
		// (app.get(path, (req, res) => ...)), an assignment
		// (exports.x = function ...), an object property or a class field.
		// Only substantial ones become units; tiny lambdas stay in their
		// parent's code.
		if n.EndPosition().Row-n.StartPosition().Row >= 2 {
			if x.isIIFE(n) || x.swallowed(n) {
				break // module wrapper: its contents are the units
			}
			name := x.anonName(n)
			if cls := strings.TrimPrefix(class, nsMark); cls != "" && name != "" {
				name = cls + "." + name
			}
			span := x.statementOf(n)
			if x.coveredSpan(span) {
				span = n
			}
			if x.coveredSpan(span) {
				return
			}
			x.emitNamed(span, name, class, kindFor(class))
			return
		}
	}
	for _, c := range children(n, true) {
		x.walk(c, class, depth+1)
	}
}

// statementOf widens an anonymous function to its enclosing expression
// statement when the function is the statement's main content (so
// `exports.x = function(){}` and `app.get(..., cb)` keep their call site).
func (x *extractor) statementOf(fn *ts.Node) *ts.Node {
	for p := fn.Parent(); p != nil; p = p.Parent() {
		switch p.Kind() {
		case "expression_statement":
			if x.countAnon(p, 0) == 1 {
				return p
			}
			return fn // several callbacks: each is its own unit
		case "assignment_expression", "call_expression", "arguments", "await_expression",
			"parenthesized_expression", "member_expression":
			continue
		}
		break
	}
	return fn
}

// countAnon counts substantial anonymous functions directly in n (not
// nested inside another function), stopping at 2.
func (x *extractor) countAnon(n *ts.Node, depth int) int {
	if depth > 64 {
		return 0
	}
	c := 0
	for _, ch := range children(n, true) {
		if x.g.anon[ch.Kind()] {
			if ch.EndPosition().Row-ch.StartPosition().Row >= 2 {
				c++
			}
		} else {
			c += x.countAnon(ch, depth+1)
		}
		if c >= 2 {
			return c
		}
	}
	return c
}

// coveredSpan reports whether span overlaps a unit already emitted.
func (x *extractor) coveredSpan(n *ts.Node) bool {
	sb, eb := int(n.StartByte()), int(n.EndByte()) // #nosec G115 -- byte offsets within src
	for _, r := range x.covered {
		if sb < r[1] && r[0] < eb {
			return true
		}
	}
	return false
}

// hasBody reports whether a class-like node has a body (C/C++ `struct X;`
// and `struct X *p` are references, not definitions).
func (x *extractor) hasBody(n *ts.Node) bool {
	switch n.Kind() {
	case "class_specifier", "struct_specifier", "union_specifier", "enum_specifier":
		return n.ChildByFieldName("body") != nil
	}
	return true
}

// swallowed reports a definition that contains a parse error and is far
// longer than a real function: tree-sitter's recovery glued the rest of the
// file onto it.
func (x *extractor) swallowed(n *ts.Node) bool {
	return n.HasError() && n.EndPosition().Row-n.StartPosition().Row > maxErrorUnitLines
}

// isIIFE reports an anonymous function that is immediately invoked at
// statement level (UMD/module wrappers): `(function(g){ ... })(this);`.
func (x *extractor) isIIFE(fn *ts.Node) bool {
	p := fn.Parent()
	for p != nil && p.Kind() == "parenthesized_expression" {
		p = p.Parent()
	}
	if p == nil || p.Kind() != "call_expression" {
		return false
	}
	callee := p.ChildByFieldName("function")
	if callee == nil || !(callee.StartByte() <= fn.StartByte() && fn.EndByte() <= callee.EndByte()) {
		return false
	}
	return fn.EndPosition().Row-fn.StartPosition().Row > maxErrorUnitLines/4
}

// isIIFEBody: a large named function whose body defines several functions
// (constructor-style JS modules like THREE.WebGLRenderer). Its members are
// extracted individually instead of as one giant unit.
func (x *extractor) isIIFEBody(n *ts.Node) bool {
	if x.lang != "JavaScript" && x.lang != "TypeScript" {
		return false
	}
	if n.EndPosition().Row-n.StartPosition().Row <= maxErrorUnitLines {
		return false
	}
	body := n.ChildByFieldName("body")
	return body != nil && x.countAnon(body, 0) >= 2
}

// anonName names an anonymous function from its context: the assignment
// target, the property/field name, or the callee of the call it is passed to.
func (x *extractor) anonName(fn *ts.Node) string {
	for p := fn.Parent(); p != nil; p = p.Parent() {
		switch p.Kind() {
		case "assignment_expression":
			if l := p.ChildByFieldName("left"); l != nil {
				return x.text(l)
			}
		case "pair", "field_definition", "public_field_definition", "property_signature":
			if k := p.ChildByFieldName("key"); k != nil {
				return x.text(k)
			}
			if k := p.ChildByFieldName("name"); k != nil {
				return x.text(k)
			}
			if k := p.ChildByFieldName("property"); k != nil {
				return x.text(k)
			}
		case "call_expression":
			if f := p.ChildByFieldName("function"); f != nil {
				callee := x.text(f)
				if f.Kind() == "member_expression" {
					if pr := f.ChildByFieldName("property"); pr != nil && strings.ContainsAny(callee, "\n(") {
						callee = x.text(pr) // chained call: name by the method (then, catch, ...)
					}
				}
				if a := p.ChildByFieldName("arguments"); a != nil {
					for _, c := range children(a, true) {
						if c.Kind() == "string" || c.Kind() == "template_string" {
							return callee + " " + x.text(c)
						}
					}
				}
				return callee + " callback"
			}
		case "export_statement":
			if strings.Contains(x.text(p), "default") {
				return "default"
			}
		case "statement_block", "program", "class_body":
			return "<anonymous>"
		}
	}
	return "<anonymous>"
}

// classUnit walks a class body for methods; a class with no methods becomes
// one unit of its own.
func (x *extractor) classUnit(cls, span *ts.Node, outer string, depth int) {
	name := x.nameOf(cls)
	outerName := strings.TrimPrefix(outer, nsMark)
	full := name
	if outerName != "" && name != "" {
		full = outerName + "." + name
	} else if name == "" {
		full = outerName
	}
	scope := full
	if isNamespace(cls.Kind()) && full != "" && (outer == "" || strings.HasPrefix(outer, nsMark)) {
		scope = nsMark + full
	}
	before := len(x.units)
	for _, c := range children(cls, true) {
		x.walk(c, scope, depth+1)
	}
	if len(x.units) == before && !isNamespace(cls.Kind()) {
		x.emitNamed(span, full, "", core.KindClass)
	}
}

func kindFor(class string) string {
	if class != "" && !strings.HasPrefix(class, nsMark) {
		return core.KindMethod
	}
	return core.KindFunction
}

// nsMark prefixes scope names that come from namespaces/modules rather than
// classes, so their functions stay KindFunction.
const nsMark = "\x00ns:"

func isNamespace(kind string) bool {
	switch kind {
	case "namespace_definition", "namespace_declaration", "file_scoped_namespace_declaration", "module":
		return true
	}
	return false
}

// findDef returns the first function or class inside a wrapper.
func (x *extractor) findDef(n *ts.Node) *ts.Node {
	for _, c := range children(n, true) {
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
	for _, d := range children(n, true) {
		if d.Kind() != "variable_declarator" {
			continue
		}
		if v := d.ChildByFieldName("value"); v != nil && x.g.anon[v.Kind()] {
			return v
		}
	}
	return nil
}

func (x *extractor) declName(n *ts.Node) string {
	for _, d := range children(n, true) {
		if d.Kind() == "variable_declarator" {
			if nm := d.ChildByFieldName("name"); nm != nil {
				return x.text(nm)
			}
		}
	}
	return ""
}

func (x *extractor) emit(def, span *ts.Node, class string) {
	name := x.nameOf(def)
	// Go methods: Receiver.Method.
	if x.lang == "Go" && def.Kind() == "method_declaration" && class == "" {
		if r := def.ChildByFieldName("receiver"); r != nil {
			if t := goReceiverType(x, r); t != "" {
				name = t + "." + name
			}
		}
	}
	if x.lang == "Terraform" || x.lang == "HCL" {
		name = x.hclName(def)
	}
	if cls := strings.TrimPrefix(class, nsMark); cls != "" && name != "" && !strings.Contains(name, ".") {
		name = cls + "." + name
	}
	x.emitNamed(span, name, class, kindFor(class))
}

// emitNamed records a unit spanning span plus any directly preceding
// comments (doc comments / license headers belong to the first definition).
func (x *extractor) emitNamed(span *ts.Node, name, class, kind string) {
	if class != "" && !strings.HasPrefix(class, nsMark) && kind == core.KindFunction {
		kind = core.KindMethod
	}
	sb, eb := x.leadingComments(span), int(span.EndByte()) // #nosec G115 -- byte offsets within src
	if sb < 0 || eb > len(x.src) || sb >= eb {
		return
	}
	// leadingComments already starts at a line boundary, so indentation is
	// preserved.
	x.covered = append(x.covered, [2]int{sb, eb})
	x.units = append(x.units, core.Unit{
		File:      x.rel,
		Language:  x.lang,
		Kind:      kind,
		Name:      clean(name),
		StartLine: x.line(sb),
		EndLine:   x.line(eb - 1),
		StartByte: sb,
		EndByte:   eb,
		Code:      string(x.src[sb:eb]),
	})
}

// leadingComments returns the byte offset where n's leading comment block
// starts: consecutive comment lines directly above n (no blank line in
// between). It scans the source text backwards (at most 64 lines) instead of
// walking PrevSibling, which is O(index) per call in tree-sitter and made
// extraction quadratic on files with many siblings.
func (x *extractor) leadingComments(n *ts.Node) int {
	start := int(n.StartByte()) // #nosec G115 -- byte offset within src
	lineStart := bytes.LastIndexByte(x.src[:start], '\n') + 1
	if len(bytes.TrimSpace(x.src[lineStart:start])) != 0 {
		// The node starts mid-line (`}).catch(function ...`): what precedes
		// it on the line belongs to something else, and so do comments above.
		return start
	}
	for k := 0; k < 64 && lineStart > 0; k++ {
		prevEnd := lineStart - 1 // the '\n' ending the previous line
		prevStart := bytes.LastIndexByte(x.src[:prevEnd], '\n') + 1
		line := bytes.TrimSpace(x.src[prevStart:prevEnd])
		if len(line) == 0 || !x.isCommentLine(line) || !x.isCommentNode(prevStart, prevEnd) {
			break
		}
		lineStart = prevStart
	}
	return lineStart
}

// attrKinds are syntax node kinds that belong to the definition below them.
var attrKinds = map[string]bool{
	"attribute_item": true, "inner_attribute_item": true, "attribute_list": true, "attribute": true,
	"decorator": true, "annotation": true, "marker_annotation": true, "modifiers": true,
	"template_parameter_list": true, "template_declaration": true, "attribute_group": true,
}

// isCommentNode confirms, from the syntax tree, that the line [start,end) is
// a comment or an attribute/decorator/template header and not code that
// merely starts like one (`*p = 0;`, `@ beta)`, `template void f<int>();`).
func (x *extractor) isCommentNode(start, end int) bool {
	lo := start
	for lo < end && (x.src[lo] == ' ' || x.src[lo] == '\t') {
		lo++
	}
	if lo >= end || x.root == nil {
		return false
	}
	n := x.root.DescendantForByteRange(uint(lo), uint(lo+1)) // #nosec G115 -- offsets within src
	for d := 0; n != nil && d < 8; d++ {
		k := n.Kind()
		if x.g.comment[k] {
			return true
		}
		if attrKinds[k] {
			// A template header only counts if the template's definition
			// starts after this line (not `template void f<int>(int);`).
			if k == "template_declaration" {
				return int(n.EndByte()) > end // #nosec G115
			}
			return true
		}
		if n.StartByte() != uint(lo) { // #nosec G115
			return false // the line starts inside some other construct
		}
		n = n.Parent()
	}
	return false
}

// isCommentLine reports whether a trimmed line is a comment in the file's
// language (line comments, doc comments, block-comment bodies, attributes).
func (x *extractor) isCommentLine(line []byte) bool {
	for _, pfx := range commentPrefixes[x.lang] {
		if bytes.HasPrefix(line, []byte(pfx)) {
			return true
		}
	}
	return false
}

var (
	cLike    = []string{"//", "/*", "*", "*/"}
	hashLike = []string{"#"}
)

// commentPrefixes also lists attribute/annotation/decorator prefixes that
// belong to the definition below them.
var commentPrefixes = map[string][]string{
	"C": cLike, "C++": append([]string{"template"}, cLike...), "C#": append([]string{"["}, cLike...),
	"Go": cLike, "Java": append([]string{"@"}, cLike...), "JavaScript": append([]string{"@"}, cLike...),
	"TypeScript": append([]string{"@"}, cLike...), "Kotlin": append([]string{"@"}, cLike...),
	"Scala": append([]string{"@"}, cLike...), "Rust": append([]string{"#[", "#!["}, cLike...),
	"Solidity": cLike, "PHP": append([]string{"#[", "#"}, cLike...), "Swift": append([]string{"@"}, cLike...),
	"Python": append([]string{"@"}, hashLike...), "Ruby": hashLike, "Bash": hashLike,
	"Terraform": append([]string{"//"}, hashLike...), "HCL": append([]string{"//"}, hashLike...),
	"Lua": {"--"}, "VBA": {"'", "Rem ", "REM "}, "SQL": {"--", "/*", "*"},
}

// nameOf finds a definition's name: the "name" field, else the first
// identifier-like named child, else a declarator chain (C/C++).
func (x *extractor) nameOf(n *ts.Node) string {
	switch n.Kind() {
	case "impl_item":
		// impl<T> Trait for Type<T> / impl Type: name by the base type.
		return x.baseType(n.ChildByFieldName("type"))
	case "operator_declaration", "conversion_operator_declaration":
		if op := n.ChildByFieldName("operator"); op != nil {
			return "operator " + x.text(op)
		}
		if t := n.ChildByFieldName("type"); t != nil {
			return "operator " + x.text(t)
		}
		return "operator"
	case "constructor_definition":
		return "constructor"
	case "fallback_receive_definition":
		if strings.HasPrefix(strings.TrimSpace(x.text(n)), "receive") {
			return "receive"
		}
		return "fallback"
	case "struct_specifier", "class_specifier", "union_specifier", "enum_specifier":
		// typedef struct { ... } name;
		if n.ChildByFieldName("name") == nil {
			if p := n.Parent(); p != nil && p.Kind() == "type_definition" {
				if d := p.ChildByFieldName("declarator"); d != nil {
					return x.text(d)
				}
			}
		}
	case "destructor_declaration":
		if nm := n.ChildByFieldName("name"); nm != nil {
			return "~" + x.text(nm)
		}
	}
	if nm := n.ChildByFieldName("name"); nm != nil {
		return x.text(nm)
	}
	if d := n.ChildByFieldName("declarator"); d != nil {
		// Unwrap pointer/reference/function declarators down to the name;
		// C++ out-of-line definitions keep their qualifier (Foo::bar).
		for d != nil {
			switch d.Kind() {
			case "qualified_identifier", "identifier", "field_identifier", "destructor_name", "operator_name", "template_function", "operator_cast":
				if d.Kind() == "operator_cast" {
					if t := d.ChildByFieldName("type"); t != nil {
						return "operator " + x.text(t)
					}
				}
				return strings.ReplaceAll(x.text(d), "::", ".")
			}
			if inner := d.ChildByFieldName("declarator"); inner != nil {
				d = inner
				continue
			}
			if nm := d.ChildByFieldName("name"); nm != nil {
				return x.text(nm)
			}
			// reference_declarator etc. hold the next declarator positionally.
			var next *ts.Node
			for _, c := range children(d, true) {
				if strings.HasSuffix(c.Kind(), "declarator") || c.Kind() == "qualified_identifier" || c.Kind() == "identifier" {
					next = c
					break
				}
			}
			if next == nil {
				break
			}
			d = next
		}
	}
	for _, c := range children(n, true) {
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
	for _, c := range children(n, true) {
		if c.Kind() == "body" || c.Kind() == "block_start" {
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
		StartLine: x.line(first),
		EndLine:   x.line(max(first, last-1)),
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
	// Names reach logs, JSON and terminals: drop control characters (C0, C1,
	// ESC) and format characters (bidi overrides, zero-width), which could
	// otherwise forge terminal output or disguise a name.
	s = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) || unicode.Is(unicode.Cf, r) {
			return -1
		}
		return r
	}, s)
	for len(s) > 200 {
		_, size := utf8.DecodeLastRuneInString(s)
		s = s[:len(s)-size]
	}
	return s
}

// children returns n's children (or named children) in order using a tree
// cursor: Node.Child(i) is O(i) in tree-sitter, so index loops are
// quadratic on wide nodes (e.g. 20k siblings in hostile input).
func children(n *ts.Node, namedOnly bool) []*ts.Node {
	c := n.Walk()
	defer c.Close()
	if !c.GotoFirstChild() {
		return nil
	}
	var out []*ts.Node
	for {
		node := c.Node()
		if !namedOnly || node.IsNamed() {
			out = append(out, node)
		}
		if !c.GotoNextSibling() {
			return out
		}
	}
}

// baseType returns the base type name of a Rust type node
// (Deserializer<R> -> Deserializer, a::b::C -> C, &'a T -> T).
func (x *extractor) baseType(t *ts.Node) string {
	for d := 0; t != nil && d < 16; d++ {
		switch t.Kind() {
		case "type_identifier", "primitive_type":
			return x.text(t)
		case "generic_type", "scoped_type_identifier":
			if nm := t.ChildByFieldName("type"); nm != nil && t.Kind() == "generic_type" {
				t = nm
				continue
			}
			if nm := t.ChildByFieldName("name"); nm != nil {
				return x.text(nm)
			}
		case "reference_type", "pointer_type":
			t = t.ChildByFieldName("type")
			continue
		}
		cs := children(t, true)
		if len(cs) == 0 {
			return x.text(t)
		}
		t = cs[0]
	}
	return ""
}

// goReceiverType extracts T from a receiver list like (s *T) or (T[K]).
func goReceiverType(x *extractor, recv *ts.Node) string {
	var find func(n *ts.Node) string
	find = func(n *ts.Node) string {
		if n.Kind() == "type_identifier" {
			return x.text(n)
		}
		for _, c := range children(n, true) {
			if t := find(c); t != "" {
				return t
			}
		}
		return ""
	}
	return find(recv)
}
