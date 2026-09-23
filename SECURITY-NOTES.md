# Security notes

Accepted findings and design decisions from the review gates. Critical and high findings are fixed, never accepted.

## gosec G115 (integer conversions) in `internal/engine/mlx`

mlx-c takes `int` (int32) shapes, axes and indices. Every value that crosses into it comes from:

- model constants (1024, 3584, 6144 and so on), or
- batch dimensions bounded by `core.MaxTokens` (16384) and the engine's batch-token budget, or
- tensor shapes already checked by `internal/engine/safetensors` against `DOMExpectations`.

Every conversion goes through a checked helper:

- `cint` and `cInts` map anything outside int32 to -1, which mlx-c rejects as an invalid shape, axis or grid. A logic error therefore shows up as an MLX error instead of silently wrapping around.
- `csize` rejects negative values.
- `cdtype` only takes values from the closed `DType` enum.
- `goint` only converts MLX array sizes, which are below 2^34.

`internal/engine` and `internal/engine/safetensors` pass gosec with no exclusions. In `internal/engine/mlx`, four G115 findings remain. gosec reports them at line numbers in cgo-generated files, where the `#nosec` comments on the helpers can't attach. So `-exclude=G115` is applied to that one directory only. The engine security review confirmed that all four are wrappers around the checked helpers.

## Model file is untrusted input

MLX's safetensors loader is not hardened against hostile files. `engine.Open` always runs `safetensors.ReadHeader`, which checks:

- size limits and overflow-safe byte counts;
- that no tensors overlap and they fill the data section exactly;
- a dtype allowlist;
- `Require(DOMExpectations)`.

Only after that does MLX see the file. The file's sha256 goes into the cache key, and it comes from `hub` (verified against the LFS etag) or is computed locally.

## Release binaries are not bit-reproducible

`garble -seed=random` is used on purpose; see `docs/BUILD.md`. Check releases with `SHA256SUMS` and the build provenance attestations.

## Self-hosted GPU runner

It runs code from anyone with push access to `vaktex/vakt`. It must be ephemeral and belong to a runner group scoped to this repository. See `docs/CI.md`.

## CUDA 13 runtime installer

`install.sh` installs NVIDIA's `cuda-keyring` package as root only after the user agrees, and only if it matches a pinned SHA-256. Distro/arch combinations with no pin fall back to printed manual instructions.
