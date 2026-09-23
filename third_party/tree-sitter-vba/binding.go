// Package tree_sitter_vba is the VBA tree-sitter grammar from
// github.com/harumiWeb/tree-sitter-vba (MIT, see LICENSE), vendored at commit
// a67fd2ddeb577730c363cbbdb1405274594d8a3f. parser.c and tree_sitter/*.h are
// copied unmodified from upstream bindings/go (sha256 1ec8fc0c…a0e81e0a); cgo compiles parser.c with this package.
package tree_sitter_vba

// #cgo CFLAGS: -std=c11 -I${SRCDIR}
// #cgo !windows CFLAGS: -fPIC
// const void *tree_sitter_vba(void);
import "C"
import "unsafe"

// Language returns the tree-sitter Language for VBA.
func Language() unsafe.Pointer {
	return unsafe.Pointer(C.tree_sitter_vba())
}
