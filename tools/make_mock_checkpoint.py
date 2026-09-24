#!/usr/bin/env python
"""Build a mock DOM-0.8B checkpoint with the exact published tensor layout.

The real model does not exist yet. This builds an untrained
`experiment.model.QwenClassifier(..., auxiliary_outputs=18)` from the Qwen3.5
base weights, re-initialises the two heads deterministically, and saves it with
`safetensors.torch.save_model`, exactly as `experiment.publish` does.

Head initialisation
-------------------
The heads are two-layer MLPs (LayerNorm -> Linear -> GELU -> Dropout ->
Linear), not the single `nn.Linear` they used to be, and pooling is a
learned 4-head attention layer rather than a masked mean. Both are
initialised here.

Pooled vectors from the base backbone share a large common component, so
plain random heads give logits dominated by that shared direction and every
sample scores about the same. To get scores that are spread out and not
saturated:

* every `Linear` in each head is drawn from N(0, HEAD_STD^2) under
  `torch.manual_seed(1234)`, and the LayerNorm is left at its identity
  init so the head sees the pooled vector at unit scale;
* the *final* layer's bias is set to minus that head's response to the mean
  pooled vector of a fixed calibration set, so the logits are centred near
  zero and what remains is the sample-dependent part. Only the final bias
  can do this: it is the one that shifts the output directly.

The pooling layer is initialised first and held fixed across both, because
the calibration pooled vectors depend on it.

The contrastive projection head is *not* initialised or saved: it exists
only for the training loss and `experiment.publish` strips it from the
published checkpoint. What this mock contains is exactly what ships.

The calibration run is deterministic (CPU, float32, fixed inputs).

`--dtype bf16` casts the backbone (only) to bfloat16 before saving; the
pooling layer and the heads stay float32, matching a CUDA publish where the
backbone was loaded bf16 and everything above it was created float32.
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from _common import MOCK_BF16, MOCK_FP16, MOCK_FP32, build_classifier, sha256_file  # noqa: E402

SEED = 1234
# Two-layer heads compose their weights, so the per-layer scale that gives a
# usable logit spread is smaller than the 0.05 a single Linear needed.
HEAD_STD = 0.02
# Pooling weights. The attention scores are scaled by 1/sqrt(head_dim)
# inside the layer, so this only has to avoid saturating the softmax.
POOL_STD = 0.02

CALIBRATION = [
    ("Python", "def run(cmd):\n    os.system(cmd)"),
    ("Python", "import pickle\n\ndef load(blob):\n    return pickle.loads(blob)"),
    ("C", "int main(void) {\n    char buf[8];\n    gets(buf);\n    return 0;\n}"),
    ("C", "size_t len(const char *s) { size_t n = 0; while (s[n]) n++; return n; }"),
    ("Go", "func add(a, b int) int {\n\treturn a + b\n}"),
    ("SQL", "SELECT * FROM users WHERE id = 1;"),
    ("Rust", "fn main() {\n    let v = vec![1, 2, 3];\n    println!(\"{:?}\", v);\n}"),
    ("JavaScript", "app.get('/x', (req, res) => res.send(eval(req.query.x)));"),
    ("Java", "public int sum(int[] xs) { int s = 0; for (int x : xs) s += x; return s; }"),
    ("PHP", "<?php echo $_GET['name']; ?>"),
    ("Bash", "rm -rf \"$1\"/*"),
    ("Solidity", "function withdraw() public { msg.sender.call{value: balances[msg.sender]}(\"\"); balances[msg.sender] = 0; }"),
]


@torch.inference_mode()
def calibration_pooled(model, tokenizer):
    from experiment.encoding import PROMPT

    pooled = []

    for language, code in CALIBRATION:
        encoded = tokenizer(PROMPT.format(language=language, code=code), return_tensors="pt")
        pooled.append(model.encode(encoded["input_ids"], encoded["attention_mask"])[0])

    return torch.stack(pooled)


def linear_layers(head):
    """The `nn.Linear`s inside an `MLPHead`, in order."""
    return [m for m in head.net if isinstance(m, torch.nn.Linear)]


def reinit_pool(model):
    """Deterministic pooling weights.

    Done before calibration because every pooled vector depends on these.
    """
    torch.manual_seed(SEED)

    with torch.no_grad():
        model.pool.query.normal_(0.0, POOL_STD)

        for module in (model.pool.key, model.pool.value, model.pool.project):
            module.weight.normal_(0.0, POOL_STD)

            if module.bias is not None:
                module.bias.zero_()


def reinit_heads(model, pooled):
    """Deterministic head weights, centred on the calibration set.

    Each head is an MLP, so 'centre the output' means setting the bias of
    the *final* layer: it is the only one that shifts the logit directly.
    The earlier layers are randomised and then held, and the head's own
    response to the mean pooled vector is measured through them.
    """
    torch.manual_seed(SEED)

    with torch.no_grad():
        for head in (model.binary_head, model.auxiliary_head):
            for layer in linear_layers(head):
                layer.weight.normal_(0.0, HEAD_STD)
                layer.bias.zero_()

            # Measured through the whole head, in eval mode so dropout is
            # off, then subtracted at the output.
            final = linear_layers(head)[-1]
            final.bias.copy_(-head(pooled.mean(0, keepdim=True)).squeeze(0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
                        help="fp16 = exactly what experiment.publish uploads (every tensor .half())")
    args = parser.parse_args()

    out = args.out or {"bf16": MOCK_BF16, "fp16": MOCK_FP16}.get(args.dtype, MOCK_FP32)

    from experiment.encoding import load_tokenizer

    model, config, _ = build_classifier()
    model.eval()
    tokenizer = load_tokenizer(config)

    # Pooling first: the calibration vectors are produced by it.
    reinit_pool(model)
    pooled = calibration_pooled(model, tokenizer)
    reinit_heads(model, pooled)

    with torch.inference_mode():
        severity = torch.sigmoid(model.binary_head(pooled).squeeze(-1))
        families = torch.sigmoid(model.auxiliary_head(pooled))

    print("calibration severities:", [round(v, 4) for v in severity.tolist()])
    print(
        f"severity min={severity.min():.4f} max={severity.max():.4f} "
        f"std={severity.std():.4f}"
    )
    print(
        f"family   min={families.min():.4f} max={families.max():.4f} "
        f"std={families.std():.4f}"
    )

    if args.dtype == "bf16":
        model.backbone.to(torch.bfloat16)

    out.parent.mkdir(parents=True, exist_ok=True)

    # Save the published tensor set, not the training one: the contrastive
    # projection is dropped here exactly as `experiment.publish` drops it,
    # so the mock and the real checkpoint have identical keys.
    from experiment.publish import TRAINING_ONLY_PREFIXES, inference_weights

    if args.dtype == "fp16":
        # The release format: publish.inference_weights itself (fp16 for
        # every tensor, pooling and heads included; projection dropped).
        state = inference_weights(model)
    else:
        state = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if not name.startswith(TRAINING_ONLY_PREFIXES)
        }
    save_file(state, str(out))

    dtypes = collections.Counter()

    with safe_open(str(out), framework="pt") as handle:
        names = list(handle.keys())

        for name in names:
            tensor = handle.get_slice(name)
            dtypes[(name.split(".")[0], tensor.get_dtype())] += 1

    print(f"wrote {out}")
    print(f"tensors: {len(names)}")

    for (prefix, dtype), count in sorted(dtypes.items()):
        print(f"  {prefix:<16} {dtype:<5} x{count}")

    print(f"size: {out.stat().st_size} bytes")
    print(f"sha256: {sha256_file(out)}")


if __name__ == "__main__":
    main()
