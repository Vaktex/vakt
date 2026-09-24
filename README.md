# Vaktex OSS (`vakt`)

`vakt` scans a codebase for vulnerable code. It splits every source file into
functions, methods and classes with tree-sitter, scores each one with the
DOM-0.8B code-security classifier, and prints the units most likely to be
vulnerable, with the CWE family the model suspects. It writes a JSON report
next to the pretty one for CI and tooling.

It runs locally: on Apple Silicon GPUs (Metal), NVIDIA GPUs (CUDA 13) and CPUs
(Linux). Your code never leaves the machine.

## Install

```sh
curl -fsSL https://github.com/vaktex/vakt/releases/latest/download/install.sh | sh
```

The installer picks the right build for the machine: Metal on macOS (Apple
Silicon), CUDA 13 on Linux with an NVIDIA driver 580 or newer, and the CPU
build otherwise. It verifies the download against `SHA256SUMS`, offers to
install missing runtime libraries (CUDA 13 runtime, OpenBLAS/LAPACK), and runs
`vakt doctor`. `install.sh --help` lists its options (`--cpu`, `--prefix`,
`--dry-run`, `--uninstall`, ...).

The DOM-0.8B weights are private on Hugging Face. Set `HF_TOKEN` (or run
`hf auth login`) before the first scan, then:

```sh
vakt summon      # download vaktex/dom-oss-0.8b (~1.5 GB, fp16) into the local cache
vakt doctor      # check the machine, engine and model
```

## Scan

```sh
vakt .                                    # scan the current directory
vakt patrol src --threshold 0.7 --top 10  # stricter, shorter table
vakt . --format json --out - | jq .summary
vakt . --fail-on 0.9                      # CI: exit 2 if any unit scores >= 0.9
```

| Option | Meaning |
|---|---|
| `--precision fp32\|tf32\|bf16` | `fp32` is exact; `tf32` (~1.7x faster on GPU) and `bf16` (~2x) trade a little precision for speed |
| `--include`, `--exclude` | doublestar globs, repeatable |
| `--no-repo-ignores` | ignore the scanned tree's `.gitignore`/`.vaktignore`, so an untrusted repo cannot hide files |
| `--device cpu\|gpu`, `--devices 0,1` | device selection; several GPUs split the work |
| `--no-cache` | don't read or write the score cache |
| `--model path/to/model.safetensors` | use a local checkpoint instead of the Hub |

Exit status: 0 on success, 1 on error, 2 when `--fail-on` is reached, 130 when
interrupted.

`vakt report vakt-report.json` re-renders a saved report.

### What gets scored

- Files are walked in parallel. Dependency and build directories (`node_modules`,
  `vendor`, `.venv`, `target`, `dist`, ...), binary, minified and oversized
  files (> 2 MiB) are skipped, and every skip is listed in the report. Paths
  hidden by the repository's own ignore files are listed too.
- Each file is parsed with tree-sitter in a separate worker process with a
  memory and time budget, so hostile input cannot exhaust the machine; a file
  that exceeds it is scored as a whole.
- Functions and methods become units, with their doc comments, decorators and
  attributes. Top-level code outside them becomes one residual unit per file.
  Languages without a grammar are scored file by file.
- A unit longer than the model's 16,384-token context is split at blank lines
  and statement boundaries; the report shows it once, with the highest score
  of its parts.
- Scores are cached per model, precision and prompt, so re-scans only score
  what changed.

### Output

The pretty report ranks the flagged units (severity at or above
`--threshold`, default 0.5) with their file, lines, name, language and the
most likely CWE family, followed by the riskiest files and anything skipped.
The JSON report (`--format json|both`, `--out`) has every unit's severity and
all 18 family probabilities.

## Where things live

| What | Where |
|---|---|
| Everything | `$VAKT_CACHE`, else `<user cache dir>/vakt` (macOS `~/Library/Caches/vakt`, Linux `~/.cache/vakt`) |
| Model weights | `hub/` inside it |
| Score cache | `scores/scores.db` inside it (mode 0600) |

`install.sh --uninstall` removes the binary and, if you ask, the caches.

## Building from source

See [docs/BUILD.md](docs/BUILD.md) (toolchains, `make dev`, `make prod`,
obfuscation, Linux builds in Docker) and [docs/CI.md](docs/CI.md) (release
pipeline, parity fixtures, the self-hosted GPU runner).
[SECURITY-NOTES.md](SECURITY-NOTES.md) records the security decisions.

## Launch day: switching to the published model

The harness implements the launch architecture of DOM-0.8B: attention pooling
(`pool.*`) and MLP heads (`binary_head.net.*`, `auxiliary_head.net.*`). It
refuses checkpoints with the pre-release layout (`binary_head.weight`) with a
clear error. When the launch weights are pushed:

The model is published to `vaktex/dom-oss-0.8b` (private) by
`experiment.publish` at the end of training: fp16 safetensors, with the
contrastive `projection.*` head removed. The harness is tested against that
exact format: `tools/make_mock_checkpoint.py --dtype fp16` builds a mock with
`publish.inference_weights` itself, and `TestParityFP16Release` checks the
engine against the Python reference on it (4.8e-7 at fp32 compute, 8e-4 at
bf16).

1. **Check the layout.** Once `vaktex/dom-oss-0.8b` has the weights:

   ```sh
   vakt summon && vakt doctor --strict
   vakt . --precision fp32
   ```

   `summon` downloads them; `doctor` confirms they load. A layout mismatch
   fails in the loader and names the tensor.

2. **Pin the revision.** Pin the default to the launch commit so releases are
   reproducible. Put the commit SHA in `cmd/vakt/patrol.go` (`defaultModel`,
   `hf:vaktex/dom-oss-0.8b@<sha>`); the downloader verifies the file's sha256
   against the Hub's LFS pointer for that revision.

3. **Check parity against the real weights.** Regenerate fixtures from the
   published file on the reference machine:

   ```sh
   python tools/parity_fixtures.py --checkpoint <path to the downloaded model.safetensors> \
       --out-dir testdata/parity/release
   make parity         # engine parity: 1e-4 fp32, 1e-2 bf16
   ```

   Then upload the bundle and set the `PARITY_FIXTURES_URL` and
   `PARITY_FIXTURES_SHA256` repository variables (see docs/CI.md) so the GPU
   parity job runs against it.

4. **Release.** `git tag vX.Y.Z && git push origin vX.Y.Z`. CI builds,
   obfuscates, audits, attests and publishes all five binaries with
   `install.sh` and `SHA256SUMS`.
