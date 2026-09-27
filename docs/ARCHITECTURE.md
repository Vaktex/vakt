# Architecture

How the packages fit together, and the invariants each one is expected to keep. If you change one of these signatures or rules, update this file in the same change.

Module: `github.com/vaktex/vakt` (Go 1.27). Shared types live in `internal/core`.

```
walk ─► ast (parse, extract units) ─► tokenize ─► pipeline (cache, batch) ─► engine ─► report
```

## Backends and the fake engine

- Build tags: `mlx` gives the Metal build on darwin and the CPU build on linux; `mlx,cuda` gives the CUDA build on linux.
- **A build with no tags has no model backend.** `engine.Open` returns `engine.ErrUnavailable`, and the CLI must exit non-zero with a clear message.
- **Nothing may fall back to `engine.Fake` automatically.** `Fake` is used only in tests, and behind the hidden `--demo` flag, which labels the report `backend: fake`.

## Token budget

- **What counts:** a unit's size is `len(tok.Encode(tokenize.Render(lang, code)))`, i.e. the whole rendered prompt. Counting the code alone and adding a prefix length gives the wrong number, because tokens merge across the join.
- **The limit:** that count is compared against `core.MaxTokens` (16384).
- **Splitting:** `ast.SplitOversize` gets `countTokens func(code string) int`, which renders the prompt and counts it. It splits until every part fits.
- **Truncation guard:** after splitting, the pipeline truncates any sequence still over `MaxTokens` to `MaxTokens` ids and sets `Encoded.Truncated`. Training did the same (`truncation=True`).
- **Engine checks:** the engine rejects empty sequences and sequences over `MaxTokens`.

## Concurrency and batching

- **One engine per device.** Each device gets one `core.Engine`, driven by a single goroutine. `--jobs` controls only walk, parse and tokenize.
- **Batch size:** `BatchTokens` counts padded tokens, i.e. longest sequence × batch size. An engine may implement `core.BatchSizer` to set a lower limit.
- **Tolerances:** the batch-invariance and parity tolerance is 1e-4 in fp32 and 1e-2 in bf16 (absolute, on sigmoid outputs).

## Cache

- **Key:** `core.CacheKey(modelSHA, precision, sha256(prompt))`, which also includes `core.CacheVersion`.
- **Contents:** scores and token counts only, never source code.
- **Storage:** bbolt, in a 0700 directory with a 0600 file. It opens with a 1s lock timeout, so a second concurrent `vakt` runs without the cache instead of hanging.

## Untrusted input

Everything read from a scanned repo is untrusted: paths, names and contents.

- Invalid UTF-8 is replaced with U+FFFD, not rejected.
- Terminal output strips control characters, ANSI/OSC escapes and bidi/format characters.
- The model file is also untrusted input. Validate the safetensors header before using it.

## internal/walk
```go
type Options struct {
    Include, Exclude []string // doublestar globs relative to root
    MaxFileBytes     int64    // default 2 MiB; larger files skipped
    FollowSymlinks   bool     // default false; never escapes root even if true
    Jobs             int
}
type File struct { Path string /*abs*/; Rel string /*slash*/; Size int64 }
type Skip struct { Rel, Reason string }
func Walk(ctx context.Context, root string, opts Options, out chan<- File, skipped chan<- Skip) error
```
Honours nested `.gitignore`, `.git/info/exclude` and `.vaktignore`. Always skips dependency and build directories, binary files (a NUL byte in the first 8 KiB), minified files and oversized files. `Walk` does not close the channels; the caller does.

## internal/ast
```go
func Detect(rel string, head []byte) (lang string, ok bool) // canonical names as in language_normalizer.py
type Options struct { MinLines int; ParseTimeout time.Duration }
func Extract(ctx context.Context, rel, lang string, src []byte, opts Options) ([]core.Unit, error)
func SplitOversize(u core.Unit, src []byte, countTokens func(code string) int, maxTokens int) []core.Unit
func Languages() []string // languages with a grammar
```
- **Function-like units:** functions and methods are emitted at their outermost level. Methods are named `Class.method`.
- **Classes:** a class with no methods becomes a `KindClass` unit.
- **Top-level code:** leftover top-level code becomes one `KindResidual` unit if it has at least `MinLines` non-blank lines.
- **No grammar:** a language without a grammar produces one `KindFile` unit.
- **Split parts:** each part sets `SplitPart`/`SplitOf` and `ParentStartLine`/`ParentEndLine`.

## internal/tokenize
```go
func New() (core.Tokenizer, error)          // embedded tokenizer.json
func Render(language, code string) string   // "Language: {language}\nCode:\n{code}"
func PromptKey(text string) [32]byte        // sha256
func Count(t core.Tokenizer, s string) int
```

## internal/engine
`engine.Open(Options) (core.Engine, error)`; `engine.Fake{}` exists for tests and `--demo` only.

## internal/hub
```go
type Options struct { Repo, Revision, CacheDir, Token, Endpoint string; Offline bool; Progress func(done, total int64) }
func Resolve(ctx context.Context, o Options) (path, sha256hex string, err error)
func CacheDir() string // $VAKT_CACHE or os.UserCacheDir()/vakt/hub
```
- **Integrity:** the download goes to a temp file. The sha256 is checked against the LFS etag before the file is renamed into place.
- **One hash:** hub returns the sha, and the engine reuses it instead of re-hashing the file.
- **Scope:** only `brand.ModelFile` is ever fetched.
- **Token:** never logged, never put in an error, never written to a report.

## internal/report
```go
const SchemaVersion = "1"
type Result struct { Unit core.Unit; Tokens int; Scores core.Scores; Cached, Truncated bool }
type Meta struct {
    Root string; StartedAt time.Time; Duration time.Duration
    Files, CacheHits int; Skipped []walk.Skip // or an equivalent {File, Reason} type
    ModelRepo, ModelRevision, ModelSHA, Backend, Device, Precision string
}
func Build(meta Meta, results []Result, threshold float64) *Report
func WriteJSON(w io.Writer, r *Report) error
func ReadJSON(r io.Reader) (*Report, error)
func WriteFile(path string, r *Report) error // atomic, 0600
func Pretty(w io.Writer, r *Report, o PrettyOptions) error
type Progress struct { FilesFound, FilesParsed, Units, UnitsScored, TokensTotal, TokensScored, CacheHits atomic.Int64; Started time.Time }
```

## internal/pipeline
```go
type Config struct {
    Root string; Walk walk.Options; AST ast.Options
    BatchTokens, Jobs, MinTokens int; Threshold float64
    NoCache bool; CacheDir string
    ModelRepo, ModelRevision, Precision string
}
func Run(ctx context.Context, cfg Config, eng core.Engine, tok core.Tokenizer, prog *report.Progress) (*report.Report, error)
```
