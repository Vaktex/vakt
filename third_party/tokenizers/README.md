# tokenizers (HuggingFace, via daulet/tokenizers)

`internal/tokenize` calls the Rust HuggingFace `tokenizers` crate through
[github.com/daulet/tokenizers](https://github.com/daulet/tokenizers), a cgo
binding to a static library, `libtokenizers.a`. `build.sh` builds that
library from source.

| | |
|---|---|
| Go module | `github.com/daulet/tokenizers v1.27.0` (go.mod) |
| Source tag | `v1.27.0` = `f678a7768d5479d9d5a5161c4fc45c8a5ba46146` (checked by build.sh) |
| Rust crate | `tokenizers-ffi` 1.26.0 → `tokenizers` 0.22.0 (upstream `Cargo.lock`, `--locked`) |
| Output | `lib/<goos>_<goarch>/libtokenizers.a` (gitignored), plus a `VERSION` stamp |
| Platforms | darwin/arm64, linux/amd64, linux/arm64 |

The Go package and the library must come from the same tag. The binding has a
link-time check, `tokenizers_version_1_26_0` (the crate version at tag
v1.27.0). A mismatch fails at link time and never becomes a wrong-ids bug at
runtime. To upgrade, bump go.mod and `TAG`/`COMMIT` in `build.sh` together,
then re-run the parity tests.

## Build

```sh
./third_party/tokenizers/build.sh          # no-op if the stamped version is already built
./third_party/tokenizers/build.sh --force  # rebuild
make deps-tokenizers                       # same, via the Makefile
```

It needs git, a C compiler (the `onig_sys` crate compiles C), and cargo. It
looks for cargo on PATH, then in `~/.cargo/bin`. Set `INSTALL_RUST=1` to have
it install a minimal rustup toolchain when cargo is missing. A cold build takes
about 15 s on an M5 Pro and a few minutes in an amd64 container under
emulation.

### Linux (CI, `nvidia/cuda:*-ubuntu24.04` or `ubuntu:24.04`)

```sh
apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git build-essential
# Rust: either the pinned rustup in scripts/docker/Dockerfile.linux, or:
INSTALL_RUST=1 ./third_party/tokenizers/build.sh
# -> third_party/tokenizers/lib/linux_amd64/libtokenizers.a (or linux_arm64)
go test ./internal/tokenize/
```

Tested in `ubuntu:24.04` on linux/arm64 (native) and linux/amd64 (emulated):
parity is 312/312 on both.

## Linking

daulet's `tokenizer.go` hard-codes `#cgo LDFLAGS: -ltokenizers -ldl -lm
-lstdc++` with no `-L`. cgo merges LDFLAGS from every package into one link,
so `internal/tokenize/cgo_link.go` supplies the search path:

```go
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/../../third_party/tokenizers/lib/darwin_arm64
#cgo linux,amd64  LDFLAGS: -L${SRCDIR}/../../third_party/tokenizers/lib/linux_amd64
#cgo linux,arm64  LDFLAGS: -L${SRCDIR}/../../third_party/tokenizers/lib/linux_arm64
```

A plain `go build` / `go test` therefore works with no environment variables.
The Makefile also exports `CGO_LDFLAGS=-L$(TOK_LIB)` pointing at the same
directory. That is harmless (duplicate `-L` for the same path) and can stay or go.

`${SRCDIR}` expands at build time. With `-trimpath` the path never reaches the
binary: `-L` affects the link, not what gets embedded.

## Path hygiene

`build.sh` adds `--remap-path-prefix` for `$HOME`, the source checkout and
`$CARGO_HOME/{registry/src,git/checkouts}` unless `RUSTFLAGS` already contains
a remap (as `make deps` does). rustc applies the last matching remap, so the
broad `$HOME` mapping comes first. With this, a `-trimpath -ldflags '-s -w'`
binary has no `/Users/`, `.cargo/` or `third_party` strings from the Rust
code, which is what `make audit` checks.

## Why Rust and not pure Go

Exact parity with the training tokenizer is the requirement. Running the same
Rust code that `transformers` runs gives that by construction, including NFC
normalization, the Unicode-class regex split, and added-token matching. A
pure-Go BPE would be a reimplementation that only the fixtures could check.
Throughput is the same as Python (about 16 ms per 100 KB on one core), so the
binding adds no measurable cost.
