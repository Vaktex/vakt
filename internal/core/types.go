// Package core holds the contracts every other package builds against.
//
// The data flow is:
//
//	walk -> ast (Unit) -> tokenize (Encoded) -> pipeline -> engine (Scores) -> report
//
// Nothing in this package does work; it only fixes the shapes that cross
// package boundaries so the streams can be built independently.
package core

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"strings"
)

// Model architecture constants, taken from the base model config.json and
// experiment/model.py. The engine verifies them against tensor shapes at load.
const (
	MaxTokens   = 16384 // training/inference context (experiment/config.py max_length)
	HiddenSize  = 1024  // pooled representation width
	NumFamilies = 18    // auxiliary_head out_features (labels.CWE_FAMILY_NAMES)
	NumLayers   = 24
	VocabSize   = 248320
)

// CacheVersion is mixed into every cache key. Bump it whenever tokenization,
// prompt rendering, pooling or head semantics change.
const CacheVersion = "2" // 2: prompt code is whitespace-trimmed as in training

// Unit kinds.
const (
	KindFunction = "function"
	KindMethod   = "method"
	KindClass    = "class"    // a class/impl/module body scored because it has no methods
	KindResidual = "residual" // top-level code left over after functions are removed
	KindFile     = "file"     // whole file, used when no grammar is available
)

// Unit is one scorable span of source code.
type Unit struct {
	File      string `json:"file"`     // path relative to the scan root, forward slashes
	Language  string `json:"language"` // canonical name, e.g. "Python", "C++" (as in language_normalizer.py)
	Kind      string `json:"kind"`
	Name      string `json:"name,omitempty"`
	StartLine int    `json:"start_line"` // 1-based, inclusive
	EndLine   int    `json:"end_line"`   // 1-based, inclusive
	StartByte int    `json:"-"`
	EndByte   int    `json:"-"`
	Code      string `json:"-"`
	// SplitPart is 0 for an unsplit unit, otherwise 1..SplitOf when a unit
	// larger than MaxTokens was cut at AST child boundaries. Parts of one
	// unit share ParentStartLine/ParentEndLine (the original unit's span),
	// which is how the report regroups them: the parent's scores are the
	// maximum severity and element-wise maximum family probabilities over
	// its parts.
	SplitPart       int `json:"split_part,omitempty"`
	SplitOf         int `json:"split_of,omitempty"`
	ParentStartLine int `json:"parent_start_line,omitempty"`
	ParentEndLine   int `json:"parent_end_line,omitempty"`
}

// Encoded is a unit rendered through the training prompt and tokenized.
// len(IDs) is always in 1..MaxTokens: the pipeline truncates anything longer
// (as training did with truncation=True) after AST splitting has had its go.
type Encoded struct {
	Unit      Unit
	IDs       []int32
	Truncated bool
	// Key is sha256(prompt text). Combined with the model identity it is the
	// cache key; the source itself is never stored.
	Key [32]byte
}

// Scores are the two heads' outputs after sigmoid.
type Scores struct {
	Severity float32              `json:"severity"`
	Families [NumFamilies]float32 `json:"families"`
}

// EngineInfo describes the loaded model and device.
type EngineInfo struct {
	Backend   string // "metal", "cuda", "cpu", "fake"
	Device    string // human readable, e.g. "Apple M5 Pro" or "NVIDIA H100 (0)"
	Precision string // "fp32" or "bf16" compute
	ModelSHA  string // sha256 of model.safetensors, lowercase hex
}

// Engine scores batches of token id sequences.
//
// Concurrency: a process opens one Engine per device and drives it from a
// single goroutine. Parallelism elsewhere (walk, parse, tokenize) never
// creates extra engines.
//
// Score must return len(batch) results in order. Every sequence must have
// 1..MaxTokens ids; the engine returns an error otherwise. Sequences in a
// batch may differ in length; padding and masking are the engine's job, the
// padding mask comes from sequence lengths (never from a pad token id), and
// a sequence's result must not depend on the rest of the batch beyond
// floating point noise (fp32 <= 1e-4, bf16 <= 1e-2 absolute).
type Engine interface {
	Info() EngineInfo
	Score(ctx context.Context, batch [][]int32) ([]Scores, error)
	Close() error
}

// BatchSizer is optionally implemented by engines to advertise the largest
// padded batch (longest sequence x batch size) they can run at once.
type BatchSizer interface {
	MaxBatchTokens() int
}

// Tokenizer turns rendered prompt text into token ids with exactly the ids
// the HF fast tokenizer produced during training: special tokens that appear
// literally in the text are matched as HF matches them, and none are added
// (the model has no BOS). Invalid UTF-8 is replaced with U+FFFD before
// encoding. Implementations must be safe for concurrent use.
type Tokenizer interface {
	Encode(text string) ([]int32, error)
	Close() error
}

// CacheKey combines the model identity, compute precision and cache version
// with a prompt hash. Each variable-length field is length-prefixed, so
// distinct inputs cannot produce the same preimage.
func CacheKey(modelSHA, precision string, promptKey [32]byte) string {
	h := sha256.New()
	for _, field := range []string{CacheVersion, strings.ToLower(modelSHA), precision} {
		var n [8]byte
		binary.BigEndian.PutUint64(n[:], uint64(len(field)))
		h.Write(n[:])
		h.Write([]byte(field))
	}
	h.Write(promptKey[:])
	return hex.EncodeToString(h.Sum(nil))
}
