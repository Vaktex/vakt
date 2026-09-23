# Package contracts

Every stream builds against these signatures. Change them only in Phase 2
(integration), never inside a stream.

Module: `github.com/vaktex/vakt` (Go 1.27). Shared types live in
`internal/core`. The CUDA build uses the `mlx,cuda` tags, the Metal build
uses `mlx`, and a build with no tags uses the fake engine with no native deps.

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
Honours `.gitignore` (nested), `.git/info/exclude`, and a `.vaktignore`. It always skips
`.git`, `node_modules`, `vendor`, `dist`, `build`, binary files (a NUL byte in the
first 8 KiB), and minified files (average line length over 500). Both channels are
closed by the caller, not by `Walk`.

## internal/ast
```go
func Detect(rel string, head []byte) (lang string, ok bool) // canonical names as in language_normalizer.py
type Options struct { MinLines int; ParseTimeout time.Duration }
func Extract(ctx context.Context, rel, lang string, src []byte, opts Options) ([]core.Unit, error)
func SplitOversize(u core.Unit, src []byte, countTokens func(string) int, maxTokens int) []core.Unit
func Languages() []string // languages with a grammar
```
- Languages without a grammar return one `KindFile` unit.
- Units are functions and methods, keeping nested ones only at their outermost scorable level.
- Top-level code left over after the functions are removed becomes one `KindResidual` unit, but only if it has at least `MinLines` non-blank lines.

## internal/tokenize
```go
func New() (core.Tokenizer, error)          // embedded Qwen3.5 tokenizer.json
func Render(language, code string) string   // "Language: {language}\nCode:\n{code}"
func PromptKey(text string) [32]byte
```

## internal/engine
`engine.Open(Options) (core.Engine, error)`. `engine.Fake{}` is always available.

## internal/hub
```go
type Options struct { Repo, Revision, CacheDir, Token string; Progress func(done, total int64) }
func Resolve(ctx context.Context, o Options) (path string, sha string, err error) // download if needed
func CacheDir() string // $VAKT_CACHE or os.UserCacheDir()/vakt
```

## internal/report
```go
const SchemaVersion = "1"
type Report struct { ... } // JSON schema in the plan
func Build(meta Meta, results []Result) *Report
func WriteJSON(w io.Writer, r *Report) error
func Pretty(w io.Writer, r *Report, o PrettyOptions) error
```

## internal/pipeline
```go
type Config struct { Root string; Walk walk.Options; AST ast.Options; BatchTokens, Jobs int; NoCache bool; ... }
func Run(ctx context.Context, cfg Config, eng core.Engine, tok core.Tokenizer, progress func(Progress)) (*report.Report, error)
```
