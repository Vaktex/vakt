// Command gen builds internal/tokenize/assets/tokenizer.json.zst from a
// base model directory. Run it through `go generate ./internal/tokenize`.
//
// The embedded file is the tokenizer that transformers actually runs, not the
// tokenizer.json on disk. AutoTokenizer resolves this model to
// its slow tokenizer class (transformers 5.17), which rebuilds the Rust backend from
// vocab and merges and differs from tokenizer.json in two ways that change
// ids:
//
//  1. Pre-tokenizer regex. That class hard-codes PRETOKENIZE_REGEX
//     (splitRegex below), which has no \p{M}. tokenizer.json ships a newer
//     pattern with [\p{L}\p{M}]+. They split Devanagari, Thai, combining
//     accents and apostrophes differently.
//  2. Added tokens. Every entry of tokenizer_config.json "added_tokens_decoder"
//     is registered, but tokenizer.json lists only 22 of the 33. The other 11
//     (<think>, </think>, <tool_response>, </tool_response>, <|audio_*|>,
//     <tts_*>) would otherwise be split into ordinary BPE pieces.
//
// The normalizer (NFC), ByteLevel stage, post-processor, vocab and merges are
// copied byte for byte. testdata/gen_cases.py records the regex and added
// tokens of the live HF backend in cases.json, and the Go tests check the
// embedded file against them, so a transformers upgrade that changes either
// shows up as a test failure and not as silently different ids.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/klauspost/compress/zstd"
)

// splitRegex is the transformers tokenizer class's hard-coded
// PRETOKENIZE_REGEX (transformers 5.17.0), the split used in training.
const splitRegex = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type addedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
}

func main() {
	model := flag.String("model", os.Getenv("VAKT_TOKENIZER_DIR"),
		"model directory containing tokenizer.json and tokenizer_config.json (env VAKT_TOKENIZER_DIR)")
	out := flag.String("out", "assets/tokenizer.json.zst", "output path")
	flag.Parse()
	if *model == "" {
		log.Fatal("gen: set -model or VAKT_TOKENIZER_DIR to the base model directory")
	}
	if err := run(*model, *out); err != nil {
		log.Fatalf("gen: %v", err)
	}
}

func run(modelDir, out string) error {
	out = filepath.Clean(out)
	sumPath := filepath.Clean(out + ".sha256")
	tokJSON, err := os.ReadFile(filepath.Join(filepath.Clean(modelDir), "tokenizer.json")) // #nosec G703 -- dev tool: reads the model dir its own user passes
	if err != nil {
		return err
	}
	cfgJSON, err := os.ReadFile(filepath.Join(filepath.Clean(modelDir), "tokenizer_config.json")) // #nosec G703 -- as above
	if err != nil {
		return err
	}
	effective, err := effectiveTokenizer(tokJSON, cfgJSON)
	if err != nil {
		return err
	}

	enc, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBestCompression), zstd.WithEncoderConcurrency(1))
	if err != nil {
		return err
	}
	compressed := enc.EncodeAll(effective, nil)
	if err := enc.Close(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o750); err != nil {
		return err
	}
	if err := os.WriteFile(out, compressed, 0o600); err != nil {
		return err
	}
	srcSum := sha256.Sum256(tokJSON)
	effSum := sha256.Sum256(effective)
	meta := fmt.Sprintf("source tokenizer.json sha256 %x\nembedded tokenizer.json sha256 %x\n", srcSum, effSum)
	// #nosec G703 -- dev-time generator; the path is the developer's own -out flag.
	if err := os.WriteFile(sumPath, []byte(meta), 0o600); err != nil {
		return err
	}
	fmt.Printf("wrote %s: %d -> %d bytes\n%s", out, len(effective), len(compressed), meta)
	return nil
}

// effectiveTokenizer rewrites only the top-level "added_tokens" array of
// tokenizer.json to the union with tokenizer_config.json added_tokens_decoder.
func effectiveTokenizer(tokJSON, cfgJSON []byte) ([]byte, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(tokJSON, &top); err != nil {
		return nil, fmt.Errorf("tokenizer.json: %w", err)
	}
	var existing []addedToken
	if err := json.Unmarshal(top["added_tokens"], &existing); err != nil {
		return nil, fmt.Errorf("tokenizer.json added_tokens: %w", err)
	}
	var cfg struct {
		AddedTokensDecoder map[string]addedToken `json:"added_tokens_decoder"`
	}
	if err := json.Unmarshal(cfgJSON, &cfg); err != nil {
		return nil, fmt.Errorf("tokenizer_config.json: %w", err)
	}
	if len(cfg.AddedTokensDecoder) == 0 {
		return nil, errors.New("tokenizer_config.json has no added_tokens_decoder")
	}

	byID := map[int]addedToken{}
	for _, t := range existing {
		byID[t.ID] = t
	}
	for k, t := range cfg.AddedTokensDecoder {
		id, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("added_tokens_decoder key %q: %w", k, err)
		}
		t.ID = id
		if old, ok := byID[id]; ok && old != t {
			return nil, fmt.Errorf("added token %d differs: tokenizer.json %+v, config %+v", id, old, t)
		}
		byID[id] = t
	}
	merged := make([]addedToken, 0, len(byID))
	for _, t := range byID {
		merged = append(merged, t)
	}
	sort.Slice(merged, func(i, j int) bool { return merged[i].ID < merged[j].ID })
	raw, err := json.Marshal(merged)
	if err != nil {
		return nil, err
	}
	top["added_tokens"] = raw

	pre, err := overrideRegex(top["pre_tokenizer"])
	if err != nil {
		return nil, err
	}
	top["pre_tokenizer"] = pre

	// Re-emit in the original key order so the diff against the source stays small.
	order := []string{"version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model"}
	seen := map[string]bool{}
	var buf bytes.Buffer
	buf.WriteByte('{')
	write := func(k string) error {
		v, ok := top[k]
		if !ok || seen[k] {
			return nil
		}
		seen[k] = true
		if buf.Len() > 1 {
			buf.WriteByte(',')
		}
		kb, err := json.Marshal(k)
		if err != nil {
			return err
		}
		buf.Write(kb)
		buf.WriteByte(':')
		buf.Write(v)
		return nil
	}
	for _, k := range order {
		if err := write(k); err != nil {
			return nil, err
		}
	}
	rest := make([]string, 0, len(top))
	for k := range top {
		rest = append(rest, k)
	}
	sort.Strings(rest)
	for _, k := range rest {
		if err := write(k); err != nil {
			return nil, err
		}
	}
	buf.WriteByte('}')
	if !json.Valid(buf.Bytes()) {
		return nil, errors.New("generated tokenizer.json is not valid JSON")
	}
	return buf.Bytes(), nil
}

// overrideRegex replaces the pattern of the Split step in a
// Sequence[Split, ByteLevel] pre-tokenizer and fails on any other shape.
func overrideRegex(raw json.RawMessage) (json.RawMessage, error) {
	var seq struct {
		Type          string                       `json:"type"`
		Pretokenizers []map[string]json.RawMessage `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &seq); err != nil {
		return nil, fmt.Errorf("pre_tokenizer: %w", err)
	}
	if seq.Type != "Sequence" || len(seq.Pretokenizers) != 2 {
		return nil, fmt.Errorf("pre_tokenizer: want Sequence[Split, ByteLevel], got %s", raw)
	}
	var typ string
	if err := json.Unmarshal(seq.Pretokenizers[0]["type"], &typ); err != nil || typ != "Split" {
		return nil, fmt.Errorf("pre_tokenizer[0]: want Split, got %s", seq.Pretokenizers[0]["type"])
	}
	pat, err := json.Marshal(map[string]string{"Regex": splitRegex})
	if err != nil {
		return nil, err
	}
	seq.Pretokenizers[0]["pattern"] = pat
	return json.Marshal(seq)
}
