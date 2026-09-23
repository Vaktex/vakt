# Reference tooling

These Python scripts produce the ground truth that the Go engine (`vakt`) is
tested against. They import the training package
(`$VAKT_CLASSIFICATION_MODELS/experiment`) directly, so the
tensor layout, prompt, tokenisation, pooling and heads are exactly what
training and `experiment.publish` produce. They are not reimplemented here.

All outputs go to the shared, gitignored `Harness/testdata/` directory. The
worktrees all use that one directory.

| Env var | Default |
|---|---|
| `VAKT_CLASSIFICATION_MODELS` | the nearest parent dir holding `experiment/` and `Qwen3.5-0.8B-Base/` |
| `VAKT_TESTDATA` | `$VAKT_CLASSIFICATION_MODELS/Harness/testdata` |
| `VAKT_LLAMA_CPP`, `VAKT_LLAMA_CPP_SHA` | `../llama.cpp` next to classification_models, commit `3173a564` |
| `VAKT_JUICE_SHOP`, `VAKT_JUICE_SHOP_SHA` | `../juice-shop`, commit `a520e158` |

Source code for the long samples and tokenizer cases is read with
`git show <sha>:<path>` at the pinned commits, never from the working tree.
Fixtures are therefore reproducible from upstream and can't pick up local,
uncommitted edits. Both JSON files record the source SHAs under `sources`.

## Regenerate everything

```sh
PY=$VAKT_CLASSIFICATION_MODELS/.venv/bin/python

$PY tools/make_mock_checkpoint.py                 # fp32 -> testdata/models/mock-dom-0.8b/model.safetensors        (~1 min)
$PY tools/make_mock_checkpoint.py --dtype bf16    # bf16 -> testdata/models/mock-dom-0.8b-bf16/model.safetensors   (~1 min)
$PY tools/parity_fixtures.py                      # fixtures from the fp32 mock, CPU            (~3 min on an M-series CPU)
```

The outputs are deterministic, so the same inputs give the same checkpoint
sha256 on the same torch/transformers versions. `parity_fixtures.py` flags:

- `--checkpoint PATH` (default: the fp32 mock)
- `--device cpu|mps` (default `cpu`, which is the ground truth; MPS is only for quick checks)
- `--out-dir DIR` (default `testdata/parity`)
- `--skip-long` drops the ~2k, ~8k and ~12k-token samples

## `make_mock_checkpoint.py`

This builds `experiment.model.QwenClassifier(ExperimentConfig(model_path=<base>),
detect_backend(prefer="cpu"), auxiliary_outputs=18)`. That is the Qwen3.5-0.8B
base text tower, loaded as float32 by the CPU backend. The script then
re-initialises the heads and calls `safetensors.torch.save_model`, the same
call `publish.py` makes.

Head init: `torch.manual_seed(1234)`, and the weights are drawn from
`N(0, 0.05²)`. Pooled vectors from the base model have a norm of about 80 and
share a large common component. Without a correction, every sample would
score about the same, and the scores would often saturate. To avoid that,
each head's bias is set to `-(W · mean_pooled)` over a fixed 12-snippet
calibration set. This centres the logits at zero. Severity over the 48
fixture samples ranges from 0.005 to 0.934 (mean 0.49, std 0.28).

`--dtype bf16` casts the **backbone only** to bfloat16. The heads stay
float32, which is what a CUDA publish produces: the backbone is loaded as
bf16 and the heads are float32 `nn.Linear`s. The engine must accept both
layouts.

The script prints the tensor count, the dtype counts per prefix, the file
size and the sha256.

### Tensor layout

There are 324 tensors: 320 `backbone.*` tensors, plus
`binary_head.{weight,bias}` `[1,1024]`/`[1]` and
`auxiliary_head.{weight,bias}` `[18,1024]`/`[18]`. The file has no
`__metadata__`. The 24 layers follow `layer_types`: layer `i` is full
attention when `i % 4 == 3` and linear attention (Gated DeltaNet) otherwise.
The word embeddings are tied, and there is no `lm_head`.

Notes for the implementer (see `transformers/models/qwen3_5/modeling_qwen3_5.py`):

- `Qwen3_5RMSNorm` (the `input_layernorm`, `post_attention_layernorm`,
  `norm`, `q_norm` and `k_norm` weights) computes `x_norm * (1 + w)`, not
  `x_norm * w`. The weights are zero-centred. It is computed in float32,
  with eps 1e-6.
- `linear_attn.norm` (`Qwen3_5RMSNormGated`, dim 128) multiplies by plain
  `w`, then by `silu(z)`.
- `self_attn.q_proj` is `[4096,1024]` = 8 heads × 256 × 2, because it holds
  the query and the output gate (`attn_output_gate: true`). k and v have 2
  heads × 256. Partial RoPE: 64 of the 256 dims, theta 1e7.
- `linear_attn.in_proj_qkv` is `[6144,1024]` = q(2048) | k(2048) | v(2048).
  It is followed by a depthwise causal `conv1d` (`[6144,1,4]`, no bias) and
  `silu`.
- The inputs to `linear_attn` are multiplied by the attention mask. The
  padding is on the right, so padded positions never influence real tokens.

## `parity_fixtures.py`

This loads the mock into a freshly built `QwenClassifier` with
`load_state_dict(strict=True)`. It runs in float32 and eval mode (no
dropout). The scores come from `model.encode` + `model._heads`:

`last_hidden_state` → masked mean pool (float32) → `binary_head` → sigmoid =
severity, and `auxiliary_head` → sigmoid = 18 family probabilities.

Tokenisation is `experiment.encoding.encode`, the loader's own call:
`AutoTokenizer(use_fast=True)`, no BOS, truncation at 16384, and right
padding with pad = eos = 248044.

### `testdata/parity/fixtures.json`

The top-level keys are `model_sha256`, `checkpoint_dtypes`, `dtype`
(`float32`), `device`, `torch_version`, `transformers_version`,
`python_version`, `prompt_template`, `max_length`, `pad_token_id`,
`eos_token_id`, `families_order` (= `experiment.labels.CWE_FAMILY_NAMES`),
`samples` and `batch`.

Each entry in `samples` has `name`, `language`, `code`, `prompt`, `ids`,
`severity`, `families[18]`, `pooled[1024]`, `binary_logit` and
`family_logits[18]`. Every sample is scored **on its own, without padding**.
There are 48 samples:

- short snippets in Python, C, C++, Java, JavaScript, TypeScript, Go, Rust,
  PHP, Ruby, C#, Solidity, SQL and Bash
- unicode: emoji, CJK, combining marks, RTL overrides, astral-plane code
  points and C0 control characters
- empty-ish cases: `""`, `" "`, `"\n"` and mixed whitespace
- CRLF, tabs, a ~2k-token single line, and an unknown language
- `long_2k`, `long_8k` and `long_12k`, built from real llama.cpp and
  juice-shop code cut at a line boundary

`batch` holds one padded batch of 4 samples of mixed length
(`py_os_system`, `c_gets`, `empty`, `java_sql`). It stores the
`input_ids`/`attention_mask` that were fed in and the outputs, plus
`max_abs_diff_vs_single`. The measured maxima are 1.44e-5 on pooled and
family logits, 7.9e-6 on the binary logit, 2.7e-6 on family probabilities and
2.7e-7 on severity. Use these as the tolerance
scale: batch and single agree to float32 reduction-order noise.

### `testdata/parity/tokenizer_cases.json`

`{tokenizer, tokenizer_sha256, transformers_version, cases: [{text, ids}]}`
with 240 cases. They cover every short sample's code and its rendered
prompt, whitespace, tab and CRLF edge cases, numbers and punctuation,
identifier casing, many scripts and emoji (ZWJ sequences, flags, skin
tones), very long runs and lines, ~27 other languages and file formats,
slices of real files, and strings that look like special tokens. The ids
are `tokenizer(text)["input_ids"]` with the default `add_special_tokens`,
which adds nothing for this tokenizer. None of the texts contains a NUL.
Note that HF turns `<|endoftext|>` written in the text into id 248044. The
Go tokenizer must either do the same or document where it differs.

### `testdata/parity/layers.safetensors`

This file holds hidden states for the single sample `py_os_system`
(T = 17 tokens, batch 1, no padding). The float tensors are float32
`[T,1024]`. They are captured with forward hooks:

| name | what |
|---|---|
| `input_ids` | int64 `[T]`, the ids that were fed in |
| `embeddings` | output of `backbone.embed_tokens` |
| `layers.00` … `layers.23` | output of `backbone.layers[i]` (the residual stream after the block's MLP) |
| `norm` | output of `backbone.norm` (= `last_hidden_state`) |
| `pooled` | float32 `[1024]`, `mean(norm, axis=0)`; equals that sample's `pooled` in fixtures.json |

The metadata is `{"sample": "py_os_system"}`. To find where the engine
diverges, compare it against this file layer by layer: the first layer
whose diff jumps is the broken one. Layers 03, 07, 11, 15, 19 and 23 are
full attention; the rest are Gated DeltaNet.

## Regenerating for the real model

Once the trained `model.safetensors` exists, run
`parity_fixtures.py --checkpoint /path/to/model.safetensors --out-dir <dir>`.
If the published file is bf16, the loader upcasts it to float32 before
scoring, so the fixtures show what an fp32 engine should compute from those
bf16 weights.
