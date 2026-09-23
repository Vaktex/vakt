# Building Vaktex OSS (`vakt`)

## Prerequisites

| | macOS (Apple Silicon) | Linux (amd64 / arm64) |
|---|---|---|
| Go | 1.27 | 1.27 |
| garble | `go install mvdan.cc/garble@v0.18.0` | same |
| C/C++ | Xcode (with the `metal` compiler) | build-essential |
| cmake | >= 3.27 | >= 3.27 |
| Rust | stable (for HF tokenizers) | stable |
| GPU | Metal (built in) | CUDA 13 toolkit + cuDNN 9 (`BACKEND=cuda`) |
| CPU BLAS | Accelerate (built in) | OpenBLAS + LAPACK(E) dev packages (`BACKEND=cpu`) |

## Targets

```sh
make deps              # MLX + tokenizers static libs for BACKEND
make dev               # bin/vakt, unobfuscated
make prod              # dist/<asset>, garble-obfuscated, stripped, signed (mac)
make test | lint | parity | bench
make audit             # release hygiene checks on dist/<asset>
make checksums         # dist/SHA256SUMS
```

`BACKEND` is one of `auto` (the default), `metal`, `cuda`, `cpu` and `fake`.

- `auto` picks `metal` on macOS. On Linux it picks `cuda` when `nvcc` is on PATH, otherwise `cpu`.
- `fake` builds without the native model engine, for CI linting and pipeline development. It produces `vakt-<os>-<arch>-fake`, which is never a release asset and is left out of `SHA256SUMS`.

| Backend | Build tags | Asset |
|---|---|---|
| metal | `mlx` | `vakt-darwin-arm64` |
| cuda | `mlx cuda` | `vakt-linux-<arch>-cuda13` |
| cpu | `mlx` | `vakt-linux-<arch>-cpu` |

`VERSION` defaults to `git describe`, and `COMMIT` to the short HEAD. Both are stamped into `internal/brand` along with `BACKEND`, using `-X`. That still works under garble. `vakt version` prints `version=… commit=… backend=…`, and `make audit` fails unless those match what the Makefile passed in.

## Linux via Docker

```sh
scripts/docker/build-linux.sh cuda amd64   # or: all all
```

The build uses `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` (CUDA, pinned by digest) or `ubuntu:22.04` (CPU). The 22.04 base keeps the glibc floor at 2.35, which matches `install.sh`, and `make audit` enforces it. cudart is linked statically. cuBLAS, cuBLASLt, NVRTC and cuDNN come from the host, which needs the CUDA 13 runtime and driver >= 580.

## Obfuscation and its limits

`make prod` runs `garble -literals -tiny -seed=random build -trimpath -buildvcs=false` with `-ldflags=-s -w`:

- Go package paths, identifiers and file names are hashed. String literals are encrypted and decoded at runtime. Panic and trace metadata are removed (`-tiny`).
- `GOGARBLE=*` obfuscates dependencies too. If a dependency fails to build under garble, narrow it with, for example, `GOGARBLE=github.com/vaktex/*`.
- **Not obfuscated:** the statically linked C/C++ code (MLX, tree-sitter grammars, tokenizers). It is stripped of symbols and debug info (`strip -x` on macOS, `--strip-unneeded` on Linux) but its logic is readable. That code is open source anyway; our IP is the Go orchestration and the model weights.
- **Weights are not in the binary.** They are downloaded from `vaktex/DOM-0.8B` with the user's HF credentials, and access control happens there.

`make audit` fails the build in any of these cases:

- `github.com/vaktex` appears in the build info or strings;
- build-machine home paths or source-layout paths appear;
- anything that looks like an HF token appears;
- Linux container build paths (`/root/`, `.cargo/registry`, `/src/third_party`) appear. Native objects are built with `-ffile-prefix-map` and Rust with `--remap-path-prefix`, so these paths stay out;
- DWARF or debug sections are present;
- on macOS, the code signature does not verify.

## Reproducibility

The toolchains are pinned and checksum-verified: Go 1.27.1, rustup 1.29.1 with Rust 1.98.1, and the CUDA 13.0.3 image. Release binaries are still deliberately **not** bit-reproducible. `garble -seed=random` picks new name hashes on every build, so symbols can't be diffed across releases. The chosen seed is printed in the build log, so keep CI logs private. Check releases through `SHA256SUMS` and the GitHub build provenance attestations, not by rebuilding.

## Signing (macOS)

The default is an ad-hoc signature. To sign with a real identity:

```sh
make prod CODESIGN_IDENTITY="Developer ID Application: Vaktex (TEAMID)" \
          NOTARIZE=1 NOTARY_PROFILE=vaktex-notary
```

A real identity turns on the hardened runtime with a secure timestamp, which notarization requires. No entitlements are needed: Metal shaders are compiled out of process by MTLCompilerService, not in `vakt`'s own address space.
