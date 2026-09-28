// Package ast turns source files into scorable units with tree-sitter.
//
// Detect maps a path (and the first bytes of the file) to a canonical
// language name as used in training (language_normalizer.py). Extract parses
// files for languages with a grammar and emits one unit per outermost
// function or method, one per method-less class, and one residual unit for
// substantial top-level code. Languages without a grammar become a single
// whole-file unit. SplitOversize cuts units that exceed the model's context
// at AST child boundaries.
package ast

import (
	"bytes"
	"path"
	"strings"
)

// extLang maps lowercase file extensions to canonical language names.
var extLang = map[string]string{
	".py": "Python", ".pyw": "Python", ".pyi": "Python",
	".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript", ".jsx": "JavaScript",
	".ts": "TypeScript", ".mts": "TypeScript", ".cts": "TypeScript", ".tsx": "TypeScript",
	".go":   "Go",
	".rs":   "Rust",
	".java": "Java",
	".c":    "C", ".h": "C",
	".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".c++": "C++", ".hh": "C++", ".hpp": "C++", ".hxx": "C++", ".ipp": "C++", ".inl": "C++",
	".cs":  "C#",
	".php": "PHP", ".phtml": "PHP",
	".rb": "Ruby", ".rake": "Ruby",
	".sh": "Bash", ".bash": "Bash", ".zsh": "Bash",
	".kt": "Kotlin", ".kts": "Kotlin",
	".swift": "Swift",
	".scala": "Scala", ".sc": "Scala",
	".lua": "Lua",
	".sol": "Solidity",
	".sql": "SQL",
	".tf":  "Terraform", ".tfvars": "Terraform", ".hcl": "HCL",
	".bas": "VBA", ".cls": "VBA", ".frm": "VBA", ".vba": "VBA",
	".vb": "VB.NET", ".vbs": "VBScript",
	".m": "Objective-C", ".mm": "Objective-C",
	".dart": "Dart", ".ex": "Elixir", ".exs": "Elixir",
	".groovy": "Groovy", ".gradle": "Groovy", ".jl": "Julia", ".nim": "Nim",
	".pl": "Perl", ".pm": "Perl", ".ps1": "PowerShell", ".psm1": "PowerShell",
	".r": "R", ".zig": "Zig", ".v": "V", ".vy": "Vyper", ".move": "Move", ".cairo": "Cairo",
	// Trained from Experiment 10's generated data (no grammar here yet, so
	// each file is scored whole, as for other grammar-less languages).
	".hs": "Haskell", ".ml": "OCaml", ".mli": "OCaml", ".erl": "Erlang", ".hrl": "Erlang",
	".clj": "Clojure", ".cljs": "Clojure", ".cljc": "Clojure", ".fs": "F#", ".fsx": "F#",
	".d": "D", ".cu": "CUDA", ".cuh": "CUDA", ".sv": "Verilog", ".svh": "Verilog",
	".vhd": "VHDL", ".vhdl": "VHDL", ".nix": "Nix", ".elm": "Elm", ".hx": "Haxe",
	".cr": "Crystal", ".gleam": "Gleam", ".mojo": "Mojo", ".odin": "Odin",
	".pas": "Pascal", ".pp": "Pascal", ".dpr": "Pascal", ".adb": "Ada", ".ads": "Ada",
	".f": "Fortran", ".f90": "Fortran", ".f95": "Fortran", ".for": "Fortran",
	".cob": "COBOL", ".cbl": "COBOL", ".abap": "ABAP", ".apex": "Apex", ".cfm": "ColdFusion", ".cfc": "ColdFusion",
	".coffee": "CoffeeScript", ".tcl": "Tcl", ".as": "ActionScript", ".asm": "Assembly", ".s": "Assembly",
	".bicep": "Bicep", ".rpgle": "RPG",
	".jsp": "JSP", ".ftl": "FreeMarker", ".hbs": "Handlebars", ".handlebars": "Handlebars", ".twig": "Twig",
	".vue": "Vue", ".svelte": "Svelte",
	".html": "HTML", ".htm": "HTML", ".xml": "XML", ".xsl": "XML", ".xsd": "XML", ".xaml": "XML", ".plist": "XML",
	".css": "CSS", ".scss": "CSS", ".less": "CSS",
	".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
}

// nameLang maps exact (lowercase) base names.
var nameLang = map[string]string{
	"dockerfile": "Bash", "makefile": "Bash", "gnumakefile": "Bash",
	"jenkinsfile": "Groovy", "rakefile": "Ruby", "gemfile": "Ruby", "vagrantfile": "Ruby",
	".bashrc": "Bash", ".zshrc": "Bash", ".profile": "Bash",
}

// noScan are files that are never code worth scoring (docs, locks, data).
var noScan = map[string]bool{
	".md": true, ".markdown": true, ".rst": true, ".txt": true, ".adoc": true,
	".lock": true, ".sum": true, ".mod": false, ".csv": true, ".tsv": true, ".log": true,
	".svg": true, ".map": true, ".snap": true, ".ipynb": true,
}

var lockNames = map[string]bool{
	"package-lock.json": true, "yarn.lock": true, "pnpm-lock.yaml": true, "cargo.lock": true,
	"poetry.lock": true, "gemfile.lock": true, "composer.lock": true, "go.sum": true, "uv.lock": true,
	"pipfile.lock": true, "flake.lock": true, "bun.lockb": true, "mix.lock": true,
}

// dataDirs are path segments whose JSON/YAML/XML content is data, not
// configuration: translation catalogues, fixtures, static assets, test
// corpora. Scoring them costs a large share of a scan for no security
// signal. Code files in these directories are still scanned.
var dataDirs = map[string]bool{
	"i18n": true, "l10n": true, "locale": true, "locales": true, "translations": true, "lang": true,
	"fixtures": true, "__fixtures__": true, "testdata": true, "test-data": true, "__snapshots__": true,
	"static": true, "assets": true, "public": true, "data": true, "datasets": true, "samples": true,
}

var dataLangs = map[string]bool{"JSON": true, "YAML": true, "XML": true, "HTML": true, "CSS": true}

// Detect returns the canonical language of the file at rel (a
// slash-separated relative path) given its first bytes, and whether it
// should be scanned at all.
func Detect(rel string, head []byte) (string, bool) {
	lang, ok := detect(rel, head)
	if ok && dataLangs[lang] {
		for _, seg := range strings.Split(strings.ToLower(path.Dir(rel)), "/") {
			if dataDirs[seg] {
				return "", false
			}
		}
	}
	return lang, ok
}

func detect(rel string, head []byte) (string, bool) {
	base := strings.ToLower(path.Base(rel))
	if lockNames[base] {
		return "", false
	}
	if l, ok := nameLang[base]; ok {
		return l, true
	}
	if strings.HasPrefix(base, "dockerfile.") {
		return "Bash", true
	}
	ext := strings.ToLower(path.Ext(base))
	if noScan[ext] {
		return "", false
	}
	if l, ok := extLang[ext]; ok {
		if ext == ".cls" && looksApex(head) {
			return "Apex", true
		}
		if ext == ".h" && looksCPP(head) {
			return "C++", true
		}
		if ext == ".v" && looksVerilog(head) {
			return "Verilog", true // .v is shared by V and Verilog
		}
		if ext == ".d" && looksMakeDeps(head) {
			return "", false // gcc -MD dependency file, not D source
		}
		if ext == ".m" && bytes.Contains(head, []byte("function ")) && !bytes.Contains(head, []byte("@interface")) && !bytes.Contains(head, []byte("#import")) {
			return "", false // MATLAB, not trained
		}
		return l, true
	}
	if ext == "" {
		return shebang(head)
	}
	return "", false
}

// looksApex tells Salesforce Apex classes (.cls) from VBA class modules.
func looksApex(head []byte) bool {
	h := bytes.ToLower(head)
	if bytes.Contains(h, []byte("attribute vb_")) || bytes.Contains(h, []byte("version 1.0 class")) {
		return false
	}
	for _, k := range [][]byte{[]byte("public class "), []byte("global class "), []byte("with sharing"), []byte("without sharing"), []byte("@istest"), []byte("private class ")} {
		if bytes.Contains(h, k) {
			return true
		}
	}
	return false
}

// looksVerilog tells Verilog (.v) from the V language (.v): Verilog files
// declare `module ... ;` and end with `endmodule`; V uses `fn` and `module x`
// without a semicolon.
func looksVerilog(head []byte) bool {
	return bytes.Contains(head, []byte("endmodule")) ||
		bytes.Contains(head, []byte("always @")) || bytes.Contains(head, []byte("`timescale"))
}

// looksMakeDeps spots compiler dependency files (`foo.o: foo.c foo.h \`),
// which share the .d extension with D source.
func looksMakeDeps(head []byte) bool {
	line := head
	if i := bytes.IndexByte(head, '\n'); i >= 0 {
		line = head[:i]
	}
	return bytes.Contains(line, []byte(".o:")) || bytes.Contains(line, []byte(".o :")) ||
		(bytes.HasSuffix(bytes.TrimSpace(line), []byte("\\")) && bytes.Contains(line, []byte(":")))
}

func looksCPP(head []byte) bool {
	for _, k := range [][]byte{[]byte("class "), []byte("namespace "), []byte("template<"), []byte("template <"), []byte("std::"), []byte("public:"), []byte("private:")} {
		if bytes.Contains(head, k) {
			return true
		}
	}
	return false
}

func shebang(head []byte) (string, bool) {
	if !bytes.HasPrefix(head, []byte("#!")) {
		return "", false
	}
	line := head
	if i := bytes.IndexByte(head, '\n'); i >= 0 {
		line = head[:i]
	}
	s := string(line)
	switch {
	case strings.Contains(s, "python"):
		return "Python", true
	case strings.Contains(s, "node") || strings.Contains(s, "deno") || strings.Contains(s, "bun"):
		return "JavaScript", true
	case strings.Contains(s, "ruby"):
		return "Ruby", true
	case strings.Contains(s, "perl"):
		return "Perl", true
	case strings.Contains(s, "php"):
		return "PHP", true
	case strings.Contains(s, "lua"):
		return "Lua", true
	case strings.Contains(s, "pwsh") || strings.Contains(s, "powershell"):
		return "PowerShell", true
	case strings.Contains(s, "sh"): // sh, bash, zsh, dash, ksh
		return "Bash", true
	}
	return "", false
}
