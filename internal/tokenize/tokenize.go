// Package tokenize turns rendered prompts into Qwen3.5 token ids.
//
// The ids match, exactly, what training produced with
//
//	AutoTokenizer.from_pretrained(model, use_fast=True)(text)["input_ids"]
//
// The work is done by HuggingFace tokenizers (Rust) through
// github.com/daulet/tokenizers, a cgo binding to a static libtokenizers.a
// built by third_party/tokenizers/build.sh. The tokenizer definition is
// embedded (assets/tokenizer.json.zst, see gen/) so the binary needs no
// model files to tokenize.
//
// Behaviour that is easy to get wrong, all covered by the parity fixtures:
//
//   - No special tokens are added. bos_token is None and the post-processor
//     is ByteLevel, so tokenizer(text) adds nothing.
//   - Added tokens that appear in raw text are matched as single tokens, as
//     in HF (split_special_tokens=False). "<|im_start|>" inside source code
//     becomes id 248045, not "<", "|", "im", ... The set is the 33 tokens of
//     tokenizer_config.json, which includes <think> and friends that the raw
//     tokenizer.json lacks.
//   - The pre-tokenizer regex is the one Qwen2Tokenizer hard-codes, not the
//     one in tokenizer.json (they split Devanagari, Thai and combining marks
//     differently). The embedded file is the effective HF config; see gen/.
//   - Invalid UTF-8 is replaced with U+FFFD, one per maximal invalid
//     subsequence, which is exactly bytes.decode("utf-8", errors="replace")
//     in Python. See Sanitize.
//   - NUL bytes are tokenized (id 188). The C boundary cannot carry them, so
//     they are swapped for U+0001 before the call and restored after; the two
//     bytes are treated identically by every stage of this tokenizer.
package tokenize

//go:generate go run ./gen -out assets/tokenizer.json.zst

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"errors"
	"fmt"
	"strings"
	"sync"
	"unicode/utf8"

	hf "github.com/daulet/tokenizers"
	"github.com/klauspost/compress/zstd"
	"golang.org/x/sync/semaphore"

	"github.com/vaktex/vakt/internal/core"
)

// MaxInputBytes is the largest text Encode accepts. Units are cut at
// core.MaxTokens long before this; the cap only bounds memory on hostile
// input.
const MaxInputBytes = 4 << 20

// MaxInFlightBytes bounds the total input size being encoded at once across
// all goroutines. The Rust encoder holds roughly 180 bytes of intermediate
// state per input byte (a 4 MiB input peaks near 740 MB), so without this 16
// workers on oversize input could need over 10 GB. Encode blocks until its
// share is free. Typical units (a few KB to ~64 KB) never wait.
const MaxInFlightBytes = 16 << 20

// A single input must always fit the in-flight budget, or Acquire would
// block forever. Fails to compile if the constants are changed badly.
const _ = uint(MaxInFlightBytes - MaxInputBytes)

// ErrTooLarge is returned by Encode for text over MaxInputBytes.
var ErrTooLarge = errors.New("tokenize: input exceeds 4 MiB")

// ErrClosed is returned by Encode after Close.
var ErrClosed = errors.New("tokenize: tokenizer is closed")

//go:embed assets/tokenizer.json.zst
var compressed []byte

// Byte-level ids of the bytes 0x00 and 0x01. Neither byte occurs in any
// merge, so each is always its own token.
const (
	idNUL = 188
	idSOH = 189
)

// maxDecompressed bounds the embedded asset after decompression (10 MB today).
const maxDecompressed = 64 << 20

var (
	assetOnce sync.Once
	assetJSON []byte
	assetErr  error
)

// tokenizerJSON decompresses the embedded tokenizer.json once.
func tokenizerJSON() ([]byte, error) {
	assetOnce.Do(func() {
		dec, err := zstd.NewReader(nil, zstd.WithDecoderConcurrency(1), zstd.WithDecoderMaxMemory(maxDecompressed))
		if err != nil {
			assetErr = err
			return
		}
		defer dec.Close()
		assetJSON, assetErr = dec.DecodeAll(compressed, nil)
		if assetErr != nil {
			assetErr = fmt.Errorf("tokenize: decompress embedded tokenizer.json: %w", assetErr)
		}
	})
	return assetJSON, assetErr
}

// Tokenizer is the Qwen3.5 tokenizer. It is safe for concurrent use; share
// one instance between all workers. Aggregate throughput from one instance
// flattens out around 8 to 16 goroutines at about 5x a single core. Separate
// instances scale further, so the limit is state shared inside one Rust
// tokenizer (most likely the BPE word cache's RwLock). The cost is about
// 70 MB and 200 ms per instance.
type Tokenizer struct {
	mu     sync.RWMutex // guards tk against Close; Encode holds it shared
	tk     *hf.Tokenizer
	budget *semaphore.Weighted // bytes in flight, see MaxInFlightBytes
}

var _ core.Tokenizer = (*Tokenizer)(nil)

// New loads the embedded tokenizer. Loading takes about 200 ms and ~100 MB,
// so create one per process and share it.
func New() (core.Tokenizer, error) {
	data, err := tokenizerJSON()
	if err != nil {
		return nil, err
	}
	tk, err := hf.FromBytes(data)
	if err != nil {
		return nil, fmt.Errorf("tokenize: load tokenizer: %w", err)
	}
	return &Tokenizer{tk: tk, budget: semaphore.NewWeighted(MaxInFlightBytes)}, nil
}

// Encode returns the token ids of text. Invalid UTF-8 is replaced first (see
// Sanitize). Text longer than MaxInputBytes is rejected with ErrTooLarge.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	if len(text) > MaxInputBytes {
		return nil, fmt.Errorf("%w (%d bytes)", ErrTooLarge, len(text))
	}
	text = Sanitize(text)
	// Replacing invalid bytes with U+FFFD can triple the size: enforce the
	// cap on what actually reaches the tokenizer too.
	if len(text) > MaxInputBytes {
		return nil, fmt.Errorf("%w (%d bytes after UTF-8 repair)", ErrTooLarge, len(text))
	}
	if text == "" {
		return []int32{}, nil
	}
	hasNUL := strings.IndexByte(text, 0) >= 0
	in := text
	if hasNUL {
		in = strings.ReplaceAll(text, "\x00", "\x01")
	}

	// Take the memory budget before the read lock so a waiting Encode never
	// holds up Close.
	weight := int64(len(in))
	if err := t.budget.Acquire(context.Background(), weight); err != nil {
		return nil, err
	}
	defer t.budget.Release(weight)

	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.tk == nil {
		return nil, ErrClosed
	}
	var opts []hf.EncodeOption
	if hasNUL {
		opts = append(opts, hf.WithReturnOffsets())
	}
	// add_special_tokens=true mirrors tokenizer(text); it adds nothing here.
	enc, err := t.tk.EncodeWithOptionsErr(in, true, opts...)
	if err != nil {
		return nil, fmt.Errorf("tokenize: encode: %w", err)
	}
	if len(enc.IDs) == 0 {
		return nil, errors.New("tokenize: encode returned no tokens for non-empty text")
	}

	ids := make([]int32, len(enc.IDs))
	for i, id := range enc.IDs {
		ids[i] = int32(id) // #nosec G115 -- ids are < core.VocabSize (248320)
	}
	if hasNUL {
		if len(enc.Offsets) != len(ids) {
			return nil, errors.New("tokenize: offsets missing for NUL restoration")
		}
		for i, off := range enc.Offsets {
			// Offsets are byte offsets into in, which has text's layout.
			if ids[i] == idSOH && off[0] < uint(len(text)) && text[off[0]] == 0 {
				ids[i] = idNUL
			}
		}
	}
	return ids, nil
}

// Close releases the native tokenizer. It waits for in-flight Encode calls
// and is safe to call more than once.
func (t *Tokenizer) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.tk != nil {
		err := t.tk.Close()
		t.tk = nil
		return err
	}
	return nil
}

// Render is the single prompt every model sees (experiment/encoding.py).
func Render(language, code string) string {
	return "Language: " + language + "\nCode:\n" + code
}

// PromptKey is sha256 of the rendered prompt text.
func PromptKey(text string) [32]byte {
	return sha256.Sum256([]byte(text))
}

// Count returns the number of tokens in s. If s cannot be encoded (it is over
// MaxInputBytes, or the tokenizer is closed) it returns an upper bound
// instead: every token covers at least one byte of the sanitized text, so its
// byte length bounds the count. Oversize input therefore always looks too
// long to fit, never too short.
//
// Errors other than ErrTooLarge (e.g. a closed tokenizer) are programming
// errors, not size signals, so they panic rather than report a small count.
func Count(t core.Tokenizer, s string) int {
	if len(s) > MaxInputBytes {
		return len(s) // cheap: never sanitize a huge input just to say "too big"
	}
	ids, err := t.Encode(s)
	if errors.Is(err, ErrTooLarge) {
		return len(Sanitize(s))
	}
	if err != nil {
		panic("tokenize: Count: " + err.Error())
	}
	return len(ids)
}

// Sanitize replaces invalid UTF-8 with U+FFFD the way Python's
// bytes.decode("utf-8", errors="replace") (and Rust's from_utf8_lossy) does:
// one U+FFFD per maximal subpart of an ill-formed sequence (Unicode 15, 3.9,
// "U+FFFD Substitution of Maximal Subparts"). A truncated 4-byte sequence
// F0 9F 98 becomes one U+FFFD; C0 AF becomes two; an encoded surrogate
// ED A0 80 becomes three.
//
// This differs from strings.ToValidUTF8 (which merges a whole run into one
// U+FFFD) and from ranging over the string (one U+FFFD per byte), both of
// which would change token ids. Valid input is returned unchanged without
// copying.
func Sanitize(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) + 16)
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			b.WriteByte(c)
			i++
			continue
		}
		n := validPrefix(s[i:])
		if n > 0 {
			b.WriteString(s[i : i+n])
			i += n
			continue
		}
		b.WriteRune(utf8.RuneError)
		i += maximalSubpart(s[i:])
	}
	return b.String()
}

// validPrefix returns the length of the valid multi-byte sequence at the
// start of s, or 0.
func validPrefix(s string) int {
	r, n := utf8.DecodeRuneInString(s)
	if r == utf8.RuneError && n <= 1 {
		return 0
	}
	return n
}

// maximalSubpart returns how many bytes of the ill-formed sequence at the
// start of s form one maximal subpart (always >= 1).
func maximalSubpart(s string) int {
	lead := s[0]
	var need int
	lo, hi := byte(0x80), byte(0xBF) // allowed range of the second byte
	switch {
	case lead >= 0xC2 && lead <= 0xDF:
		need = 1
	case lead == 0xE0:
		need, lo = 2, 0xA0
	case lead >= 0xE1 && lead <= 0xEC, lead == 0xEE, lead == 0xEF:
		need = 2
	case lead == 0xED:
		need, hi = 2, 0x9F
	case lead == 0xF0:
		need, lo = 3, 0x90
	case lead >= 0xF1 && lead <= 0xF3:
		need = 3
	case lead == 0xF4:
		need, hi = 3, 0x8F
	default: // 80..C1, F5..FF: never a valid lead
		return 1
	}
	i := 1
	for k := 0; k < need && i < len(s); k++ {
		c := s[i]
		if k == 0 {
			if c < lo || c > hi {
				break
			}
		} else if c < 0x80 || c > 0xBF {
			break
		}
		i++
	}
	return i
}
