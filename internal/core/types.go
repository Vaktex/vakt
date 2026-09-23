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
)

// Model architecture constants, taken from Qwen3.5-0.8B-Base/config.json and
// experiment/model.py. The engine verifies them against tensor shapes at load.
const (
	MaxTokens   = 16384 // training/inference context (experiment/config.py max_length)
	HiddenSize  = 1024  // pooled representation width
	NumFamilies = 18    // auxiliary_head out_features (labels.CWE_FAMILY_NAMES)
	NumLayers   = 24
	VocabSize   = 248320
)

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
	Language  string `json:"language"` // canonical name, e.g. "Python", "C++" (see ast.Canonical)
	Kind      string `json:"kind"`
	Name      string `json:"name,omitempty"`
	StartLine int    `json:"start_line"` // 1-based, inclusive
	EndLine   int    `json:"end_line"`   // 1-based, inclusive
	StartByte int    `json:"-"`
	EndByte   int    `json:"-"`
	Code      string `json:"-"`
	// SplitPart is 0 for an unsplit unit, otherwise 1..SplitOf when a unit
	// larger than MaxTokens was cut at AST child boundaries.
	SplitPart int `json:"split_part,omitempty"`
	SplitOf   int `json:"split_of,omitempty"`
}

// Encoded is a unit rendered through the training prompt and tokenized.
type Encoded struct {
	Unit Unit
	IDs  []int32
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
	ModelSHA  string // sha256 of model.safetensors, hex
}

// Engine scores batches of token id sequences. Implementations must be safe
// for use by a single goroutine; the pipeline owns one goroutine per Engine.
//
// Score must return len(batch) results in order. Sequences in a batch may have
// different lengths; padding and masking are the engine's job, and the result
// for a sequence must not depend on what else is in the batch.
type Engine interface {
	Info() EngineInfo
	Score(ctx context.Context, batch [][]int32) ([]Scores, error)
	Close() error
}

// Tokenizer turns rendered prompt text into token ids with exactly the ids
// the HF fast tokenizer produced during training (no BOS, no special tokens).
// Implementations must be safe for concurrent use.
type Tokenizer interface {
	Encode(text string) ([]int32, error)
	Close() error
}

// CacheKey combines the model identity with a prompt hash.
func CacheKey(modelSHA string, promptKey [32]byte) string {
	h := sha256.New()
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], uint64(len(modelSHA)))
	h.Write(n[:])
	h.Write([]byte(modelSHA))
	h.Write(promptKey[:])
	return hex.EncodeToString(h.Sum(nil))
}
