package ast

import (
	"sync"
	"unsafe"

	ts "github.com/tree-sitter/go-tree-sitter"

	tshcl "github.com/tree-sitter-grammars/tree-sitter-hcl/bindings/go"
	tskotlin "github.com/tree-sitter-grammars/tree-sitter-kotlin/bindings/go"
	tslua "github.com/tree-sitter-grammars/tree-sitter-lua/bindings/go"
	tsbash "github.com/tree-sitter/tree-sitter-bash/bindings/go"
	tsc "github.com/tree-sitter/tree-sitter-c/bindings/go"
	tscsharp "github.com/tree-sitter/tree-sitter-c-sharp/bindings/go"
	tscpp "github.com/tree-sitter/tree-sitter-cpp/bindings/go"
	tsgo "github.com/tree-sitter/tree-sitter-go/bindings/go"
	tsjava "github.com/tree-sitter/tree-sitter-java/bindings/go"
	tsjs "github.com/tree-sitter/tree-sitter-javascript/bindings/go"
	tsphp "github.com/tree-sitter/tree-sitter-php/bindings/go"
	tspython "github.com/tree-sitter/tree-sitter-python/bindings/go"
	tsruby "github.com/tree-sitter/tree-sitter-ruby/bindings/go"
	tsrust "github.com/tree-sitter/tree-sitter-rust/bindings/go"
	tsscala "github.com/tree-sitter/tree-sitter-scala/bindings/go"
	tsts "github.com/tree-sitter/tree-sitter-typescript/bindings/go"

	tssol "github.com/vaktex/vakt/third_party/tree-sitter-solidity"
	tsvba "github.com/vaktex/vakt/third_party/tree-sitter-vba"
)

// spec describes how to find units in one grammar.
type spec struct {
	lang func() unsafe.Pointer
	// funcs are node kinds that are function-like definitions.
	funcs []string
	// classes are container kinds whose function children are methods.
	classes []string
	// wrappers are kinds that wrap a definition and should be included in
	// its span (decorators, export statements, templates).
	wrappers []string
	// assign are kinds where a named variable is bound to an anonymous
	// function (const f = () => {}); the function inside counts as a unit.
	assign []string
	// anon are anonymous function kinds (only units when bound by assign).
	anon []string
	// comments are kinds of comment nodes to attach as leading docs.
	comments []string
}

var specs = map[string]*spec{
	"Python": {lang: tspython.Language, funcs: []string{"function_definition"}, classes: []string{"class_definition"},
		wrappers: []string{"decorated_definition"}, comments: []string{"comment"}},
	"JavaScript": {lang: tsjs.Language, funcs: []string{"function_declaration", "generator_function_declaration", "method_definition"},
		classes: []string{"class_declaration", "class"}, wrappers: []string{"export_statement"},
		assign: []string{"lexical_declaration", "variable_declaration"}, anon: []string{"arrow_function", "function_expression", "function"},
		comments: []string{"comment"}},
	"TypeScript": {lang: tsts.LanguageTSX, funcs: []string{"function_declaration", "generator_function_declaration", "method_definition", "function_signature"},
		classes: []string{"class_declaration", "abstract_class_declaration", "class"}, wrappers: []string{"export_statement"},
		assign: []string{"lexical_declaration", "variable_declaration"}, anon: []string{"arrow_function", "function_expression", "function"},
		comments: []string{"comment"}},
	"Go": {lang: tsgo.Language, funcs: []string{"function_declaration", "method_declaration"}, comments: []string{"comment"}},
	"Rust": {lang: tsrust.Language, funcs: []string{"function_item"}, classes: []string{"impl_item", "trait_item"},
		wrappers: []string{"attribute_item"}, comments: []string{"line_comment", "block_comment"}},
	"Java": {lang: tsjava.Language, funcs: []string{"method_declaration", "constructor_declaration"},
		classes: []string{"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}, comments: []string{"line_comment", "block_comment"}},
	"C": {lang: tsc.Language, funcs: []string{"function_definition"}, comments: []string{"comment"}},
	"C++": {lang: tscpp.Language, funcs: []string{"function_definition"}, classes: []string{"class_specifier", "struct_specifier", "namespace_definition"},
		wrappers: []string{"template_declaration"}, comments: []string{"comment"}},
	"C#": {lang: tscsharp.Language, funcs: []string{"method_declaration", "constructor_declaration", "local_function_statement", "operator_declaration"},
		classes: []string{"class_declaration", "struct_declaration", "interface_declaration", "record_declaration", "namespace_declaration", "file_scoped_namespace_declaration"},
		comments: []string{"comment"}},
	"PHP": {lang: tsphp.LanguagePHP, funcs: []string{"function_definition", "method_declaration"},
		classes: []string{"class_declaration", "trait_declaration", "interface_declaration"}, comments: []string{"comment"}},
	"Ruby": {lang: tsruby.Language, funcs: []string{"method", "singleton_method"}, classes: []string{"class", "module"}, comments: []string{"comment"}},
	"Bash": {lang: tsbash.Language, funcs: []string{"function_definition"}, comments: []string{"comment"}},
	"Kotlin": {lang: tskotlin.Language, funcs: []string{"function_declaration"}, classes: []string{"class_declaration", "object_declaration"}, comments: []string{"line_comment", "block_comment"}},
	"Scala": {lang: tsscala.Language, funcs: []string{"function_definition"}, classes: []string{"class_definition", "object_definition", "trait_definition"}, comments: []string{"comment", "block_comment"}},
	"Lua": {lang: tslua.Language, funcs: []string{"function_declaration"}, comments: []string{"comment"}},
	"Solidity": {lang: tssol.Language, funcs: []string{"function_definition", "modifier_definition", "constructor_definition", "fallback_receive_definition"},
		classes: []string{"contract_declaration", "library_declaration", "interface_declaration"}, comments: []string{"comment"}},
	"Terraform": {lang: tshcl.Language, funcs: []string{"block"}, comments: []string{"comment"}},
	"HCL":       {lang: tshcl.Language, funcs: []string{"block"}, comments: []string{"comment"}},
	"VBA": {lang: tsvba.Language, funcs: []string{"sub_declaration", "function_declaration", "property_declaration", "sub_statement", "function_statement", "property_statement"},
		comments: []string{"comment"}},
}

// grammar is a language with a pool of parsers (parsers are not safe for
// concurrent use; languages are).
type grammar struct {
	spec    *spec
	lang    *ts.Language
	pool    sync.Pool
	funcs   map[string]bool
	classes map[string]bool
	wraps   map[string]bool
	assign  map[string]bool
	anon    map[string]bool
	comment map[string]bool
}

var (
	grammarsOnce sync.Once
	grammars     map[string]*grammar
)

func set(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

func loadGrammars() {
	grammars = make(map[string]*grammar, len(specs))
	for name, sp := range specs {
		g := &grammar{
			spec: sp, lang: ts.NewLanguage(sp.lang()),
			funcs: set(sp.funcs), classes: set(sp.classes), wraps: set(sp.wrappers),
			assign: set(sp.assign), anon: set(sp.anon), comment: set(sp.comments),
		}
		g.pool.New = func() any {
			p := ts.NewParser()
			if err := p.SetLanguage(g.lang); err != nil {
				p.Close()
				return nil
			}
			return p
		}
		grammars[name] = g
	}
}

func grammarFor(lang string) *grammar {
	grammarsOnce.Do(loadGrammars)
	return grammars[lang]
}

// Languages returns the languages that have a grammar.
func Languages() []string {
	out := make([]string, 0, len(specs))
	for name := range specs {
		out = append(out, name)
	}
	return out
}
