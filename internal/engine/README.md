# DOM-0.8B engine (MLX)

`engine.Open` loads `model.safetensors` and returns a `core.Engine`. The native engine is only built with `-tags mlx`. Without it, `Open` returns `ErrUnavailable`.

```sh
third_party/mlx/build.sh metal          # static MLX v0.31.1 + mlx-c v0.6.0; also copies mlx.metallib for embedding (required before go build -tags mlx)
go test -tags mlx ./internal/engine/... # parity and unit tests
go build -tags mlx ./cmd/vakt
```

## Architecture mapping (PyTorch reference → `model.go`)

| Reference | Here |
|---|---|
| RMSNorm: `x̂ · (1 + w)`, computed in f32 | `rmsNorm`, with `(1 + w)` folded in at load |
| Decoder layer: pre-norm residual | `forward` |
| Gated attention: q_proj holds query and gate per head; q/k RMSNorm; partial RoPE (64 of 256, θ = 1e7, rotate_half); GQA 8/2; output × sigmoid(gate) | `attention` (fused q\|k\|v projection, MLX `fast.rope`, `fast.scaled_dot_product_attention` in causal mode) |
| Gated DeltaNet: mask padding; in_proj_qkv; causal depthwise conv (k=4) + SiLU; l2norm q/k; q × 1/√128; β = σ(b); g = −exp(A_log) · softplus(a + dt_bias); gated delta rule; RMSNormGated (plain w) · SiLU(z); out_proj | `linearAttention` (fused qkv\|z\|b\|a projection, `convSilu` Metal kernel, `deltaKernel` Metal kernel) |
| `AttentionPool` (f32): 4 queries × 256, keys/values without bias, scores × 1/16 masked to −1e4 before the softmax, weighted sum of values, project + LayerNorm (eps 1e-5) | `attentionPool` |
| `MLPHead` (f32) for binary_head and auxiliary_head: LayerNorm → Linear 1024→2048 → exact (erf) GELU → Linear, then sigmoid | `mlpHead`, `mlxEngine.Score` |

The mRoPE sections reduce to 1D RoPE for text-only input: all three position rows are equal.

## Precision

| `--precision` | Weights and activations | Matmuls | Parity vs PyTorch fp32 (48 fixtures) | Metal throughput, T=2048×4 |
|---|---|---|---|---|
| `fp32` (default) | f32 | strict f32 (TF32 off) | max\|Δs\| 3.3e-6, max\|Δp\| 5.6e-6 | ~4.3k tok/s |
| `tf32` | f32 | TF32 tensor cores | max\|Δs\| 1.3e-3 | ~7.1k tok/s |
| `bf16` | bf16 matmul inputs | bf16 | max\|Δs\| 1.0e-2, batch invariance 2.7e-3 | ~9k tok/s |

In every mode the residual stream, norms, the DeltaNet state, pooling and the heads run in f32. The bf16 checkpoint layout (bf16 backbone, f32 heads) loads in every mode.

MLX fixes the matmul mode (TF32 or not) once per process, so one process can't mix `fp32` and `tf32` engines. `Open` refuses the second one.

MLX turns on TF32 for f32 GPU matmuls by default (`MLX_ENABLE_TF32=1`). `mlx.Init` switches it off unless the environment sets it, because TF32 alone moves scores by about 1e-3.

## Kernels

- `deltaKernel`: the gated delta recurrence. It uses one SIMD group per (batch, head, value column) and keeps the state in registers. It follows the design of mlx-lm's `gated_delta` kernel (MIT) and matches `torch_recurrent_gated_delta_rule` exactly in f32.
- `convSilu`: causal depthwise conv plus SiLU in one pass. MLX's general `conv1d` was about 10× slower here.
- `deltaChunked`: the portable path, used on CPU and CUDA and when `VAKT_DELTANET=chunked` is set. It is HF's `torch_chunk_gated_delta_rule` (chunk 64, forward-substitution inverse) in plain ops. It is evaluated chunk by chunk, so memory stays bounded at any length: 12.5 GB peak at T=16384 on CPU, including 3 GB of weights.
- `deltaScanOps`: the per-token recurrence, kept as a test reference only (`VAKT_DELTANET=scan`).

## Batching and memory

- **Padding:** sequences are right-padded, and results don't depend on what else is in the batch. `TestBatchInvariance` checks this.
- **Pooling:** it uses `where(mask, h, 0)`, so a NaN from padding can never leak into the pooled vector.
- **Per-layer evaluation:** the residual stream is evaluated after each layer (`VAKT_EVAL_EVERY`, default 1), and that layer's intermediates are released. A single 24-layer lazy graph was about 90× slower.
- **Batch size:** `MaxBatchTokens()` is 32768 padded tokens on GPU and 8192 on CPU.

## CUDA builds

`libcudart` is linked statically. The NVIDIA driver (`libcuda`), cuBLAS/cuBLASLt, NVRTC and cuDNN are shared libraries, so a CUDA binary only starts on a host that has driver 580 or newer and the CUDA 13 runtime installed. `install.sh` checks for these and installs the CPU build when they're missing. `--devices`/`DeviceIndex` selects the GPU.

## Loading

`Open` validates the safetensors header before MLX parses the file (see docs/SECURITY-DESIGN.md). The model sha256 comes from `Options.ModelSHA` (hub), or is computed if missing.
