package tokenize

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"unicode/utf8"

	"github.com/vaktex/vakt/internal/core"
)

type parityCase struct {
	Name     string  `json:"name"`
	Text     *string `json:"text"`
	RawB64   string  `json:"raw_b64"`
	Language *string `json:"language"`
	Code     *string `json:"code"`
	IDs      []int32 `json:"ids"`
}

type parityFile struct {
	Cases []parityCase `json:"cases"`
}

// input returns the exact string handed to Encode: text cases verbatim, raw
// cases as the undecoded bytes (Encode must sanitize them itself).
func (c parityCase) input(t *testing.T) string {
	t.Helper()
	if c.Text != nil {
		return *c.Text
	}
	raw, err := base64.StdEncoding.DecodeString(c.RawB64)
	if err != nil {
		t.Fatalf("%s: bad raw_b64: %v", c.Name, err)
	}
	return string(raw)
}

func loadCases(t *testing.T, path string) []parityCase {
	t.Helper()
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		t.Fatal(err)
	}
	var f parityFile
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	if len(f.Cases) == 0 {
		t.Fatalf("%s: no cases", path)
	}
	return f.Cases
}

var (
	sharedOnce sync.Once
	sharedTok  core.Tokenizer
	sharedErr  error
)

func tok(tb testing.TB) core.Tokenizer {
	tb.Helper()
	sharedOnce.Do(func() { sharedTok, sharedErr = New() })
	if sharedErr != nil {
		tb.Fatal(sharedErr)
	}
	return sharedTok
}

func checkParity(t *testing.T, cases []parityCase) {
	tk := tok(t)
	pass := 0
	for _, c := range cases {
		got, err := tk.Encode(c.input(t))
		if err != nil {
			t.Errorf("%s: %v", c.Name, err)
			continue
		}
		if !slices.Equal(got, c.IDs) {
			i := 0
			for i < len(got) && i < len(c.IDs) && got[i] == c.IDs[i] {
				i++
			}
			t.Errorf("%s: mismatch at index %d (got %d ids, want %d)\n got[%d:]  %v\n want[%d:] %v",
				c.Name, i, len(got), len(c.IDs), i, head(got[i:]), i, head(c.IDs[i:]))
			continue
		}
		if c.Language != nil && c.Code != nil && c.Text != nil {
			if r := Render(*c.Language, *c.Code); r != *c.Text {
				t.Errorf("%s: Render differs from Python PROMPT.format", c.Name)
				continue
			}
		}
		pass++
	}
	t.Logf("parity %d/%d", pass, len(cases))
}

func head(ids []int32) []int32 {
	if len(ids) > 12 {
		return ids[:12]
	}
	return ids
}

func TestParity(t *testing.T) {
	cases := loadCases(t, "testdata/cases.json")
	if len(cases) < 250 {
		t.Fatalf("only %d cases", len(cases))
	}
	checkParity(t, cases)
}

// TestParityShared runs the repository-wide tokenizer cases from
// testdata/parity, produced by the reference fixture generator. It skips when the
// file is absent.
func TestParityShared(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "parity", "tokenizer_cases.json")
	if env := os.Getenv("VAKT_TOKENIZER_CASES"); env != "" {
		path = env
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("no shared fixtures at %s", path)
	}
	checkParity(t, loadCases(t, path))
}

// TestEmbeddedMatchesBackend checks the embedded tokenizer.json against the
// configuration of the live HF backend recorded by gen_cases.py, so a
// transformers change to the regex or added tokens is caught directly.
func TestEmbeddedMatchesBackend(t *testing.T) {
	data, err := os.ReadFile("testdata/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var fix struct {
		Backend struct {
			Normalizer    json.RawMessage   `json:"normalizer"`
			SplitRegex    string            `json:"split_regex"`
			PostProcessor json.RawMessage   `json:"post_processor"`
			AddedTokens   []json.RawMessage `json:"added_tokens"`
		} `json:"backend"`
	}
	if err := json.Unmarshal(data, &fix); err != nil {
		t.Fatal(err)
	}
	raw, err := tokenizerJSON()
	if err != nil {
		t.Fatal(err)
	}
	var emb struct {
		Normalizer    json.RawMessage   `json:"normalizer"`
		PostProcessor json.RawMessage   `json:"post_processor"`
		AddedTokens   []json.RawMessage `json:"added_tokens"`
		PreTokenizer  struct {
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern struct {
					Regex string `json:"Regex"`
				} `json:"pattern"`
			} `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
	}
	if err := json.Unmarshal(raw, &emb); err != nil {
		t.Fatal(err)
	}
	if got := emb.PreTokenizer.Pretokenizers[0].Pattern.Regex; got != fix.Backend.SplitRegex {
		t.Errorf("split regex\n embedded %q\n backend  %q", got, fix.Backend.SplitRegex)
	}
	if !jsonEqual(t, emb.Normalizer, fix.Backend.Normalizer) {
		t.Errorf("normalizer: embedded %s, backend %s", emb.Normalizer, fix.Backend.Normalizer)
	}
	if !jsonEqual(t, emb.PostProcessor, fix.Backend.PostProcessor) {
		t.Errorf("post_processor: embedded %s, backend %s", emb.PostProcessor, fix.Backend.PostProcessor)
	}
	if len(emb.AddedTokens) != len(fix.Backend.AddedTokens) {
		t.Fatalf("added tokens: embedded %d, backend %d", len(emb.AddedTokens), len(fix.Backend.AddedTokens))
	}
	for i := range emb.AddedTokens {
		if !jsonEqual(t, emb.AddedTokens[i], fix.Backend.AddedTokens[i]) {
			t.Errorf("added token %d: embedded %s, backend %s", i, emb.AddedTokens[i], fix.Backend.AddedTokens[i])
		}
	}
}

func jsonEqual(t *testing.T, a, b json.RawMessage) bool {
	t.Helper()
	var x, y any
	if err := json.Unmarshal(a, &x); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &y); err != nil {
		t.Fatal(err)
	}
	xs, _ := json.Marshal(x)
	ys, _ := json.Marshal(y)
	return string(xs) == string(ys)
}

func TestNoSpecialTokensAdded(t *testing.T) {
	ids, err := tok(t).Encode("hello")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, []int32{14556}) {
		t.Fatalf("got %v, want [14556] (no BOS/EOS)", ids)
	}
	ids, err = tok(t).Encode("")
	if err != nil || ids == nil || len(ids) != 0 {
		t.Fatalf("empty: %v %v", ids, err)
	}
}

func TestAddedTokensInText(t *testing.T) {
	cases := map[string][]int32{
		"<|im_start|>":            {248045},
		"<|endoftext|>":           {248044},
		"<think>":                 {248068},
		"</think>":                {248069},
		"<tool_call>":             {248058},
		"<|audio_pad|>":           {248076},
		"x<|im_start|>y":          {87, 248045, 88},
		"<|im_start":              {27, 91, 316, 4747},
		"a\x00b<|endoftext|>\x00": {64, 188, 65, 248044, 188},
	}
	for in, want := range cases {
		got, err := tok(t).Encode(in)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("%q: got %v want %v", in, got, want)
		}
	}
}

func TestNULAndSOH(t *testing.T) {
	// Expectations from HF: "x   \x00" -> [87 256 220 188], and 0x01 -> 189.
	cases := map[string][]int32{
		"x   \x00":     {87, 256, 220, idNUL},
		"x   \x01":     {87, 256, 220, idSOH},
		"\x00\x01\x00": {idNUL, idSOH, idNUL},
	}
	for in, want := range cases {
		got, err := tok(t).Encode(in)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("%q: got %v want %v", in, got, want)
		}
	}
}

func TestSanitize(t *testing.T) {
	cases := []struct{ in, want string }{
		{"plain", "plain"},
		{"a\xe2\x82x", "a\uFFFDx"},
		{"\xed\xa0\x80z", "\uFFFD\uFFFD\uFFFDz"},
		{"\xf0\x9f\x98", "\uFFFD"},
		{"\xc0\xaf", "\uFFFD\uFFFD"},
		{"\xf4\x90\x80\x80", "\uFFFD\uFFFD\uFFFD\uFFFD"},
		{"\xff\xfe", "\uFFFD\uFFFD"},
		{"\xe0\x80\x80", "\uFFFD\uFFFD\uFFFD"},
		{"\xe0\xa0", "\uFFFD"},
		{"\xf0\x90\x80\x80", "\U00010000"},
		{"\xf0\x90\x80", "\uFFFD"},
		{"\xf0\x90\x80x", "\uFFFDx"},
		{"é\x80", "é\uFFFD"},
	}
	for _, c := range cases {
		if got := Sanitize(c.in); got != c.want {
			t.Errorf("Sanitize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func FuzzSanitize(f *testing.F) {
	for _, s := range []string{"", "a", "\xff", "\xf0\x9f\x98", "é\x80", "\xed\xa0\x80"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		out := Sanitize(s)
		if !utf8.ValidString(out) {
			t.Fatalf("invalid output for %q", s)
		}
		if utf8.ValidString(s) && out != s {
			t.Fatalf("valid input changed: %q", s)
		}
		if strings.ReplaceAll(out, "\uFFFD", "") != strings.ReplaceAll(strings.ToValidUTF8(s, ""), "\uFFFD", "") {
			t.Fatalf("valid content not preserved for %q", s)
		}
	})
}

func TestInvalidUTF8NeverFails(t *testing.T) {
	var all []byte
	for i := range 256 {
		all = append(all, byte(i), byte(255-i), 0xF0, byte(i))
	}
	ids, err := tok(t).Encode(string(all))
	if err != nil || len(ids) == 0 {
		t.Fatalf("got %d ids, err %v", len(ids), err)
	}
}

func TestInputCap(t *testing.T) {
	ok := strings.Repeat("a", MaxInputBytes)
	if _, err := tok(t).Encode(ok); err != nil {
		t.Fatalf("at cap: %v", err)
	}
	_, err := tok(t).Encode(ok + "a")
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("over cap: got %v", err)
	}
	if n := Count(tok(t), ok+"a"); n != MaxInputBytes+1 {
		t.Fatalf("Count over cap = %d", n)
	}
}

func TestRenderAndKey(t *testing.T) {
	if got := Render("Go", "package x\n"); got != "Language: Go\nCode:\npackage x\n" {
		t.Fatalf("Render = %q", got)
	}
	k := PromptKey("")
	if fmt.Sprintf("%x", k) != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" {
		t.Fatalf("PromptKey(\"\") = %x", k)
	}
	if n := Count(tok(t), "hello world"); n != 2 {
		t.Fatalf("Count = %d", n)
	}
}

func TestClose(t *testing.T) {
	tk, err := New()
	if err != nil {
		t.Fatal(err)
	}
	if err := tk.Close(); err != nil {
		t.Fatal(err)
	}
	if err := tk.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := tk.Encode("x"); !errors.Is(err, ErrClosed) {
		t.Fatalf("after close: %v", err)
	}
}

// TestConcurrent encodes every fixture from many goroutines at once against
// one shared tokenizer, with a Close racing at the end. Run with -race.
func TestConcurrent(t *testing.T) {
	cases := loadCases(t, "testdata/cases.json")
	tk, err := New()
	if err != nil {
		t.Fatal(err)
	}
	const workers = 16
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range cases {
				c := cases[(i+w*17)%len(cases)]
				got, err := tk.Encode(c.input(t))
				if err != nil {
					errs <- fmt.Errorf("%s: %w", c.Name, err)
					return
				}
				if !slices.Equal(got, c.IDs) {
					errs <- fmt.Errorf("%s: mismatch under concurrency", c.Name)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	// Close while encoders are running: every call either succeeds or
	// returns ErrClosed, never crashes.
	var wg2 sync.WaitGroup
	for range workers {
		wg2.Add(1)
		go func() {
			defer wg2.Done()
			for range 50 {
				if _, err := tk.Encode("int main() { return 0; }"); err != nil && !errors.Is(err, ErrClosed) {
					t.Error(err)
					return
				}
			}
		}()
	}
	if err := tk.Close(); err != nil {
		t.Error(err)
	}
	wg2.Wait()
}

// benchSource builds ~100 KB of real C++ from the fixtures.
func benchSource(b *testing.B) string {
	b.Helper()
	data, err := os.ReadFile("testdata/cases.json")
	if err != nil {
		b.Fatal(err)
	}
	var f parityFile
	if err := json.Unmarshal(data, &f); err != nil {
		b.Fatal(err)
	}
	var sb strings.Builder
	for sb.Len() < 100_000 {
		for _, c := range f.Cases {
			if c.Code != nil && (strings.Contains(c.Name, "C++") || strings.Contains(c.Name, "file_")) {
				sb.WriteString(*c.Code)
			}
		}
	}
	return Render("C++", sb.String()[:100_000])
}

func BenchmarkEncode100KB(b *testing.B) {
	tk := tok(b)
	src := benchSource(b)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for b.Loop() {
		if _, err := tk.Encode(src); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncode100KBParallel16 splits b.N encodes over exactly 16
// goroutines sharing one tokenizer. ns/op is wall time divided by encodes,
// so MB/s is aggregate throughput.
func BenchmarkEncode100KBParallel16(b *testing.B) {
	tk := tok(b)
	src := benchSource(b)
	b.SetBytes(int64(len(src)))
	const workers = 16
	b.ResetTimer()
	var next atomic.Int64
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for next.Add(1) <= int64(b.N) {
				if _, err := tk.Encode(src); err != nil {
					b.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func BenchmarkSanitizeValid100KB(b *testing.B) {
	src := benchSource(b)
	b.SetBytes(int64(len(src)))
	for b.Loop() {
		_ = Sanitize(src)
	}
}
