// Package tree_sitter_solidity is the Solidity tree-sitter grammar from
// github.com/JoranHonig/tree-sitter-solidity v1.2.13 (MIT, see LICENSE).
// That release ships no Go binding, so parser.c and tree_sitter/*.h are
// copied unmodified from its src directory; cgo compiles parser.c with this
// package.
package tree_sitter_solidity

// #cgo CFLAGS: -std=c11 -I${SRCDIR}
// #cgo !windows CFLAGS: -fPIC
// const void *tree_sitter_solidity(void);
import "C"
import "unsafe"

// Language returns the tree-sitter Language for Solidity.
func Language() unsafe.Pointer {
	return unsafe.Pointer(C.tree_sitter_solidity())
}
