#!/usr/bin/env python
"""Build a mock DOM-0.8B checkpoint with the exact published tensor layout.

The real model does not exist yet. This builds an untrained
`experiment.model.QwenClassifier(..., auxiliary_outputs=18)` from the Qwen3.5
base weights, re-initialises the two heads deterministically, and saves it with
`safetensors.torch.save_model`, exactly as `experiment.publish` does.

Head initialisation
-------------------
Pooled vectors from the base backbone have norms around 80 and share a large
common component (mean-vector norm ~75). Plain random heads therefore give
logits dominated by that shared direction, so every sample scores about the
same. To get scores that are spread out and not saturated:

* weights are drawn from N(0, HEAD_STD^2) under `torch.manual_seed(1234)`;
* each bias is set to minus the head's response to the mean pooled vector of
  a fixed calibration set, so the logits are centred near zero and what
  remains is the sample-dependent part.

The calibration run is deterministic (CPU, float32, fixed inputs).

`--dtype bf16` casts the backbone (only) to bfloat16 before saving; the heads
stay float32, matching a CUDA publish where the backbone was loaded bf16 and
the heads were created as float32 `nn.Linear`s.
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_model  # noqa: E402

from _common import MOCK_BF16, MOCK_FP32, build_classifier, sha256_file  # noqa: E402

SEED = 1234
HEAD_STD = 0.05

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


def reinit_heads(model, pooled):
    torch.manual_seed(SEED)

    with torch.no_grad():
        for head in (model.binary_head, model.auxiliary_head):
            head.weight.normal_(0.0, HEAD_STD)
            head.bias.copy_(-(pooled.mean(0) @ head.weight.T))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    out = args.out or (MOCK_BF16 if args.dtype == "bf16" else MOCK_FP32)

    from experiment.encoding import load_tokenizer

    model, config, _ = build_classifier()
    model.eval()
    tokenizer = load_tokenizer(config)

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
    save_model(model, str(out))

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
