#!/usr/bin/env python
"""Generate numerical parity fixtures for the Go engine from the mock checkpoint.

Outputs (in the shared testdata dir, default ../testdata/parity/):

* fixtures.json          per-sample prompt, ids, pooled vector, logits, scores,
                         plus one padded batch showing batch == single
* tokenizer_cases.json   ~200 {text, ids} pure tokenizer cases
* layers.safetensors     per-layer hidden states for one short sample

Everything runs through the training package: prompts are rendered with
`experiment.encoding.PROMPT`, tokenised with `experiment.encoding.encode`
(the exact call the data loader makes), and scored with
`QwenClassifier.encode` + `QwenClassifier._heads`.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from _common import MOCK_FP32, PARITY_DIR, build_classifier, sha256_file  # noqa: E402

import os  # noqa: E402
import subprocess  # noqa: E402

from _common import CLASSIFICATION_MODELS  # noqa: E402


class PinnedRepo:
    """Read files from a git repository at a pinned commit.

    Reading through `git show <sha>:<path>` rather than the working tree makes
    fixtures reproducible from upstream and keeps uncommitted local edits out
    of committed test data.
    """

    def __init__(self, name: str, root: Path, sha: str, subdir: str = ""):
        self.name, self.root, self.sha, self.subdir = name, root, sha, subdir.strip("/")

    def _path(self, rel: str) -> str:
        return f"{self.subdir}/{rel}" if self.subdir else rel

    def read_text(self, rel: str) -> str:
        blob = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self.sha}:{self._path(rel)}"],
            check=True, capture_output=True,
        ).stdout
        return blob.decode("utf-8", errors="strict")

    def glob(self, rel_dir: str, suffix: str) -> list[str]:
        out = subprocess.run(
            ["git", "-C", str(self.root), "ls-tree", "--name-only", f"{self.sha}:{self._path(rel_dir)}"],
            check=True, capture_output=True, text=True,
        ).stdout.split()
        return sorted(f"{rel_dir.rstrip('/')}/{n}" for n in out if n.endswith(suffix))

    def describe(self) -> dict:
        return dict(name=self.name, sha=self.sha)


_VAKTEX = CLASSIFICATION_MODELS.parent
LLAMA = PinnedRepo(
    "llama.cpp", Path(os.environ.get("VAKT_LLAMA_CPP", _VAKTEX / "llama.cpp")),
    os.environ.get("VAKT_LLAMA_CPP_SHA", "3173a56471c1753650cd806694145ffd6dcace67"), subdir="src",
)
JUICE = PinnedRepo(
    "juice-shop", Path(os.environ.get("VAKT_JUICE_SHOP", _VAKTEX / "juice-shop")),
    os.environ.get("VAKT_JUICE_SHOP_SHA", "a520e158cb65c43d24e2c55d84f09b05a2511a03"),
)

# ----------------------------------------------------------------------------
# Samples
# ----------------------------------------------------------------------------

SAMPLES = [
    ("py_os_system", "Python", "def run(cmd):\n    os.system(cmd)"),
    ("py_pickle", "Python", "import pickle\n\n\ndef load(blob: bytes):\n    \"\"\"Deserialize a blob.\"\"\"\n    return pickle.loads(blob)\n"),
    ("py_sql_fstring", "Python", "def find(conn, name):\n    cur = conn.cursor()\n    cur.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n    return cur.fetchall()\n"),
    ("py_safe", "Python", "def add(a: int, b: int) -> int:\n    return a + b\n"),
    ("c_gets", "C", "#include <stdio.h>\n\nint main(void) {\n    char buf[8];\n    gets(buf);\n    printf(\"%s\\n\", buf);\n    return 0;\n}\n"),
    ("c_strcpy", "C", "void copy(char *dst, const char *src) {\n\tstrcpy(dst, src);\n}\n"),
    ("c_uaf", "C", "void f(void) {\n    char *p = malloc(16);\n    free(p);\n    p[0] = 'x';\n}\n"),
    ("cpp_vector", "C++", "#include <vector>\nint at(const std::vector<int>& v, size_t i) {\n    return v[i];\n}\n"),
    ("cpp_template", "C++", "template <typename T>\nT max_of(T a, T b) { return a > b ? a : b; }\n"),
    ("java_sql", "Java", "public User find(Connection c, String id) throws SQLException {\n    Statement s = c.createStatement();\n    ResultSet r = s.executeQuery(\"SELECT * FROM users WHERE id=\" + id);\n    return map(r);\n}\n"),
    ("java_sum", "Java", "public int sum(int[] xs) {\n    int s = 0;\n    for (int x : xs) s += x;\n    return s;\n}\n"),
    ("js_eval", "JavaScript", "app.get('/calc', (req, res) => {\n  res.send(String(eval(req.query.expr)));\n});\n"),
    ("js_innerhtml", "JavaScript", "function show(msg) {\n  document.getElementById('out').innerHTML = msg;\n}\n"),
    ("ts_typed", "TypeScript", "export function clamp(x: number, lo: number, hi: number): number {\n  return Math.min(hi, Math.max(lo, x));\n}\n"),
    ("ts_express", "TypeScript", "router.post('/login', async (req: Request, res: Response) => {\n  const user = await db.query(`SELECT * FROM Users WHERE email = '${req.body.email}'`);\n  res.json(user);\n});\n"),
    ("go_exec", "Go", "func run(w http.ResponseWriter, r *http.Request) {\n\tout, _ := exec.Command(\"sh\", \"-c\", r.URL.Query().Get(\"cmd\")).Output()\n\tw.Write(out)\n}\n"),
    ("go_add", "Go", "func add(a, b int) int {\n\treturn a + b\n}\n"),
    ("rust_unsafe", "Rust", "fn get(v: &[u8], i: usize) -> u8 {\n    unsafe { *v.get_unchecked(i) }\n}\n"),
    ("rust_main", "Rust", "fn main() {\n    let v = vec![1, 2, 3];\n    println!(\"{:?}\", v);\n}\n"),
    ("php_echo", "PHP", "<?php\necho \"Hello, \" . $_GET['name'];\n?>\n"),
    ("php_include", "PHP", "<?php\ninclude($_GET['page'] . '.php');\n"),
    ("ruby_system", "Ruby", "def ping(host)\n  system(\"ping -c 1 #{host}\")\nend\n"),
    ("ruby_yaml", "Ruby", "def load(s)\n  YAML.load(s)\nend\n"),
    ("cs_process", "C#", "public void Run(string arg) {\n    Process.Start(\"cmd.exe\", \"/c \" + arg);\n}\n"),
    ("cs_linq", "C#", "public int Total(IEnumerable<int> xs) => xs.Where(x => x > 0).Sum();\n"),
    ("sol_reentrancy", "Solidity", "function withdraw() public {\n    (bool ok, ) = msg.sender.call{value: balances[msg.sender]}(\"\");\n    require(ok);\n    balances[msg.sender] = 0;\n}\n"),
    ("sol_owner", "Solidity", "function setOwner(address o) public {\n    owner = o;\n}\n"),
    ("sql_select", "SQL", "SELECT id, name FROM users WHERE id = 1;\n"),
    ("sql_grant", "SQL", "GRANT ALL PRIVILEGES ON *.* TO 'app'@'%' WITH GRANT OPTION;\n"),
    ("bash_rm", "Bash", "#!/bin/bash\nrm -rf \"$1\"/*\n"),
    ("bash_eval", "Bash", "read -r input\neval \"$input\"\n"),
    # Unicode.
    ("uni_emoji", "Python", "def greet():\n    return \"hello 👋🏽 world 🌍🚀\"  # emoji ✨\n"),
    ("uni_cjk", "JavaScript", "// 用户输入验证\nfunction 验证(输入) {\n  return 输入.length > 0; // 日本語のコメント、한국어 주석\n}\n"),
    ("uni_combining", "Python", "name = \"e\u0301le\u0300ve Zo\u00eb \u212b \ufb01le\"\nzw = \"a\u200bb\u200dc\ufeffd\"\n"),
    ("uni_rtl_math", "C", "/* \u202eevil\u202c \u05e9\u05dc\u05d5\u05dd \u0645\u0631\u062d\u0628\u0627 \u2211\u222b\u221a\u2260 */\nint x = 1;\n"),
    ("uni_astral", "Rust", "let s = \"\U0001d11e \U0001f9ec \U00020000 \U0010fffd\";\n"),
    ("uni_controls", "Go", "s := \"bell\\a \x07 esc \x1b[31m del \x7f vt \x0b ff \x0c\"\n"),
    # Empty-ish.
    ("empty", "Python", ""),
    ("space", "C", " "),
    ("newline", "Go", "\n"),
    ("whitespace_mix", "Java", " \t\r\n\t  \n\n\n    "),
    ("crlf", "C#", "int F()\r\n{\r\n\treturn 1;\r\n}\r\n"),
    ("tabs", "Go", "func f() {\n\tif x {\n\t\tif y {\n\t\t\treturn\n\t\t}\n\t}\n}\n"),
    ("long_line", "JavaScript", "const data = [" + ", ".join(str(i * 7919 % 1000) for i in range(400)) + "];\n"),
    ("unknown_lang", "Unknown", "some text that is not code at all."),
]

LONG_TARGETS = [("long_2k", 2_000), ("long_8k", 8_000), ("long_12k", 12_000)]

LONG_SOURCES = [
    ("C++", [(LLAMA, "llama-vocab.cpp"), (LLAMA, "llama-model.cpp")]),
    ("TypeScript", None),  # juice-shop routes/*.ts, resolved lazily from the pinned tree
    ("C++", [(LLAMA, "llama-context.cpp"), (LLAMA, "llama-model.cpp")]),
]

BATCH_NAMES = ["py_os_system", "c_gets", "empty", "java_sql"]

LAYER_SAMPLE = "py_os_system"


def long_code(tokenizer, language, files, target_tokens):
    """Real code, cut at a line boundary so the rendered prompt has ~target tokens."""
    from experiment.encoding import PROMPT

    if files is None:
        files = [(JUICE, rel) for rel in JUICE.glob("routes", ".ts")]

    text = ""

    for repo, rel in files * 8:
        text += repo.read_text(rel)
        if len(text) > target_tokens * 6:
            break

    prompt = PROMPT.format(language=language, code=text)
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    cut = offsets[min(target_tokens, len(offsets)) - 1][1]
    cut = prompt.rfind("\n", 0, cut) + 1
    prefix = len(PROMPT.format(language=language, code=""))

    return text[: cut - prefix]


def build_samples(tokenizer):
    # Training-form code: every snippet in the training corpus has no
    # leading/trailing whitespace (str.strip() is a no-op on all 1.1M rows),
    # and the Go pipeline trims units the same way (tokenize.PromptCode).
    samples = [dict(name=n, language=l, code=c.strip()) for n, l, c in SAMPLES]

    for (name, target), (language, files) in zip(LONG_TARGETS, LONG_SOURCES):
        code = long_code(tokenizer, language, files, target).strip()
        samples.append(dict(name=name, language=language, code=code))

    return samples


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


@torch.inference_mode()
def score(model, encoded, device):
    """Pooled vector and both heads, exactly as inference computes them.

    `_heads` returns a dict keyed by head name and also carries the
    training-only `projection`, which is ignored here and is absent from
    the published checkpoint.
    """
    encoded = encoded.to(device)
    pooled = model.encode(encoded["input_ids"], encoded["attention_mask"])
    heads = model._heads(pooled)
    binary, auxiliary = heads["binary"], heads["auxiliary"]

    return {
        "pooled": pooled.float().cpu(),
        "binary_logit": binary.float().cpu(),
        "family_logits": auxiliary.float().cpu(),
        "severity": torch.sigmoid(binary).float().cpu(),
        "families": torch.sigmoid(auxiliary).float().cpu(),
    }


def floats(tensor):
    return [float(v) for v in tensor.reshape(-1).tolist()]


# ----------------------------------------------------------------------------
# Layer dumps
# ----------------------------------------------------------------------------


@torch.inference_mode()
def dump_layers(model, encoded, device, out):
    backbone = model.backbone
    captured = {}
    handles = []

    def hook(name):
        def capture(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor[0].detach().float().cpu().contiguous()

        return capture

    handles.append(backbone.embed_tokens.register_forward_hook(hook("embeddings")))

    for index, layer in enumerate(backbone.layers):
        handles.append(layer.register_forward_hook(hook(f"layers.{index:02d}")))

    handles.append(backbone.norm.register_forward_hook(hook("norm")))

    try:
        encoded = encoded.to(device)
        pooled = model.encode(encoded["input_ids"], encoded["attention_mask"])
    finally:
        for handle in handles:
            handle.remove()

    captured["pooled"] = pooled[0].float().cpu().contiguous()
    captured["input_ids"] = encoded["input_ids"][0].cpu().to(torch.int64).contiguous()

    # Pooling is learned, so `mean(norm)` no longer reproduces `pooled` and
    # a reimplementation cannot be checked against it. The per-token
    # attention weights are the intermediate that makes the pooling step
    # verifiable on its own: a port that gets these right and does the
    # weighted sum correctly will land on `pooled`.
    with torch.inference_mode():
        weights = model.pool.token_weights(
            model.backbone(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).last_hidden_state,
            encoded["attention_mask"],
        )

    captured["pool_weights"] = weights[0].float().cpu().contiguous()
    save_file(captured, str(out), metadata={"sample": LAYER_SAMPLE})

    return captured


# ----------------------------------------------------------------------------
# Tokenizer cases
# ----------------------------------------------------------------------------


def tokenizer_texts(samples):
    from experiment.encoding import PROMPT

    texts = []

    # Every fixture sample's raw code and rendered prompt (short ones only).
    for sample in samples:
        if sample["name"].startswith("long_"):
            continue
        texts.append(sample["code"])
        texts.append(PROMPT.format(language=sample["language"], code=sample["code"]))

    # Whitespace edge cases.
    texts += [
        "", " ", "  ", "   ", "\t", "\t\t", "\n", "\n\n", "\n\n\n", "\r\n", "\r\n\r\n", "\r", " \n", "\n ",
        " \t \t ", "a  b", "a   b", "a\tb", "a\t\tb", "a\nb", "a\r\nb", "a \n b", "x" + " " * 17 + "y",
        " " * 64, "\t" * 16, "\n" * 33, "    return x", "\treturn x", "  \n  \n", "end   \n",
        "trailing spaces   ", "   leading spaces", "\u00a0nbsp\u00a0", "\u3000ideographic space\u3000",
        "\u2028line sep\u2029para sep", "\v\f",
    ]

    # Numbers and punctuation.
    texts += [
        "0", "1234567890", "3.14159265358979", "0xDEADBEEF", "1e-10", "-42", "1_000_000", "12345678901234567890",
        "a+b-c*d/e%f", "===!==<=>=", "->=>::..", "&&||!", "{}[]()<>", "#!@$%^&*", "``` code ```", "/* */ // #",
        "i++; --j; k <<= 2; m >>>= 3;", "\"\\\"escaped\\\"\"", "'\\''", "\\n\\t\\r\\0", "\\u00e9 \\x41",
    ]

    # Identifiers and casing.
    texts += [
        "camelCaseIdentifier", "PascalCaseIdentifier", "snake_case_identifier", "SCREAMING_SNAKE_CASE",
        "kebab-case-identifier", "__dunder__", "$jqueryVar", "@decorator", "x1y2z3", "HTTPServerError",
        "getHTTPResponseCode", "I'm don't we'll they've", "ALLCAPS WORDS HERE", "MiXeD CaSe",
    ]

    # Unicode.
    texts += [
        "héllo wörld", "naïve café résumé", "日本語のテキスト", "中文字符测试", "한국어 텍스트", "Привет мир",
        "Γειά σου κόσμε", "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "สวัสดีชาวโลก",
        "😀😃😄😁", "👨‍👩‍👧‍👦", "🏳️‍🌈", "👍🏽👍🏿", "🇮🇪🇺🇸", "a😀b", "∀x∈ℝ: x²≥0", "→←↑↓⇒",
        "\U0001d400\U0001d401", "\U00020000\U0002a6d6", "\ufeffBOM start", "zero\u200bwidth", "e\u0301",
        "Ω\u2126", "ﬁ ligature", "\U0010fffd", "\u00ff\u0100\u07ff\u0800\uffff",
        "💩" * 20, "中" * 50, "é" * 40,
    ]

    # Very long lines and runs.
    texts += [
        "a" * 1000, "ab" * 700, " a" * 500, "x = " + "1 + " * 300 + "1", "-" * 400, "=" * 257,
        "/" * 100 + " comment", "0" * 500, "base64:" + "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 30,
        "".join(chr(0x4E00 + (i * 37) % 2000) for i in range(600)),
    ]

    # Code in more languages.
    texts += [
        "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hi\")\n}\n",
        "#include <iostream>\nint main() { std::cout << \"hi\" << std::endl; }\n",
        "fn main() -> Result<(), Box<dyn Error>> {\n    Ok(())\n}\n",
        "SELECT u.id, COUNT(*) FROM users u JOIN orders o ON o.uid = u.id GROUP BY u.id HAVING COUNT(*) > 5;",
        "<?php $x = $_POST['x'] ?? 'default'; ?>",
        "defmodule M do\n  def f(x), do: x * 2\nend\n",
        "(defn square [x] (* x x))",
        "main :: IO ()\nmain = putStrLn \"hi\"\n",
        "let rec fact n = if n = 0 then 1 else n * fact (n - 1)",
        "local function f(t) for k, v in pairs(t) do print(k, v) end end",
        "sub f { my ($x) = @_; return $x =~ s/a/b/gr; }",
        "fun main() { val xs = listOf(1, 2, 3); println(xs.map { it * 2 }) }",
        "func f() -> Int { guard let x = y else { return 0 }; return x }",
        "object Main extends App { println(\"hi\") }",
        "import 'package:flutter/material.dart';\nvoid main() => runApp(MyApp());\n",
        "<html><body><script>alert(document.cookie)</script></body></html>",
        "<?xml version=\"1.0\"?>\n<!DOCTYPE foo [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>\n<foo>&xxe;</foo>",
        "{\"key\": [1, 2, {\"nested\": null}], \"t\": true}",
        "key: value\nlist:\n  - a\n  - b\n",
        "[section]\nkey = \"value\"\n",
        "FROM python:3.12\nRUN pip install -r requirements.txt\nCMD [\"python\", \"app.py\"]\n",
        "all:\n\t$(CC) -o $@ $^ $(CFLAGS)\n",
        "mov eax, 1\nint 0x80\n",
        "module top(input clk, output reg q); always @(posedge clk) q <= ~q; endmodule",
        "% LaTeX\n\\begin{equation} E = mc^2 \\end{equation}",
        "# Markdown\n\n* item\n* **bold** _it_\n",
        "diff --git a/x b/x\n@@ -1,2 +1,2 @@\n-old\n+new\n",
    ]

    # Special-token-looking strings. HF parses these as special tokens even in
    # untrusted text; the Go tokenizer must match whatever HF does here.
    texts += [
        "<|endoftext|>", "a<|endoftext|>b", "<|im_start|>user\nhi<|im_end|>", "<think>x</think>",
        "<|vision_start|>", "< |endoftext| >", "<|notatoken|>",
    ]

    # Slices of real files at varied offsets.
    for path, starts in [
        ((LLAMA, "llama-vocab.cpp"), [0, 5000, 40000]),
        ((LLAMA, "llama-model.cpp"), [1000, 60000]),
        ((JUICE, "routes/chat.ts"), [0, 3000]),
        # lib/insecurity.ts opens with juice-shop's public demo RSA key; start
        # past it so secret scanners don't flag committed fixtures.
        ((JUICE, "lib/insecurity.ts"), [4000, 6500]),
    ]:
        repo, rel = path
        content = repo.read_text(rel)

        for start in starts:
            texts.append(content[start:start + 1500])

    seen, unique = set(), []

    for text in texts:
        assert "\x00" not in text

        if text not in seen:
            seen.add(text)
            unique.append(text)

    return unique


# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=MOCK_FP32)
    parser.add_argument("--out-dir", type=Path, default=PARITY_DIR)
    parser.add_argument("--device", default="cpu", help="cpu (ground truth) or mps")
    parser.add_argument("--skip-long", action="store_true", help="omit the three long samples")
    args = parser.parse_args()

    from experiment import publish
    from experiment.encoding import PROMPT, encode, load_tokenizer
    from experiment.labels import CWE_FAMILY_NAMES

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, config, _ = build_classifier()
    state = load_file(str(args.checkpoint))
    dtypes = sorted({str(t.dtype).removeprefix("torch.") for t in state.values()})

    # The checkpoint carries the published tensor set, which omits the
    # training-only contrastive projection, so the load is not strict in
    # that direction. Everything the checkpoint *does* carry must match,
    # and nothing outside the projection may be missing.
    incompatible = model.load_state_dict(
        {k: v.float() for k, v in state.items()}, strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        k for k in incompatible.missing_keys
        if not k.startswith(publish.TRAINING_ONLY_PREFIXES)
    ]
    assert not unexpected, f"checkpoint has tensors the model does not: {unexpected}"
    assert not missing, f"checkpoint is missing inference tensors: {missing}"

    del state
    model.float().eval().to(device)
    tokenizer = load_tokenizer(config)

    # The final Linear of the auxiliary MLP head decides the family count.
    family_outputs = [
        m for m in model.auxiliary_head.net if isinstance(m, torch.nn.Linear)
    ][-1].out_features
    assert len(CWE_FAMILY_NAMES) == family_outputs

    samples = build_samples(tokenizer)

    if args.skip_long:
        samples = [s for s in samples if not s["name"].startswith("long_")]

    by_name = {}

    for sample in samples:
        prompt = PROMPT.format(language=sample["language"], code=sample["code"])
        encoded = encode([prompt], tokenizer, config)
        started = time.perf_counter()
        result = score(model, encoded, device)
        elapsed = time.perf_counter() - started
        ids = encoded["input_ids"][0].tolist()

        record = dict(
            name=sample["name"],
            language=sample["language"],
            code=sample["code"],
            prompt=prompt,
            ids=ids,
            severity=float(result["severity"][0]),
            families=floats(result["families"][0]),
            pooled=floats(result["pooled"][0]),
            binary_logit=float(result["binary_logit"][0]),
            family_logits=floats(result["family_logits"][0]),
        )
        by_name[sample["name"]] = record
        print(f"{sample['name']:<16} tokens={len(ids):>6} severity={record['severity']:.6f} ({elapsed:.1f}s)", flush=True)

    # Padded batch: right padding with pad = eos, exactly as the data loader.
    batch_prompts = [by_name[name]["prompt"] for name in BATCH_NAMES]
    batch_encoded = encode(batch_prompts, tokenizer, config)
    batch_result = score(model, batch_encoded, device)

    diffs = {"severity": 0.0, "families": 0.0, "pooled": 0.0, "binary_logit": 0.0, "family_logits": 0.0}

    for row, name in enumerate(BATCH_NAMES):
        single = by_name[name]
        assert batch_encoded["input_ids"][row][: len(single["ids"])].tolist() == single["ids"]

        for key in diffs:
            got = torch.tensor(floats(batch_result[key][row]))
            want = torch.tensor(single[key] if isinstance(single[key], list) else [single[key]])
            diffs[key] = max(diffs[key], float((got - want).abs().max()))

    print("batch vs single max abs diff:", json.dumps(diffs))

    batch = dict(
        names=BATCH_NAMES,
        padding_side=tokenizer.padding_side,
        pad_token_id=tokenizer.pad_token_id,
        input_ids=batch_encoded["input_ids"].tolist(),
        attention_mask=batch_encoded["attention_mask"].tolist(),
        severity=floats(batch_result["severity"]),
        binary_logit=floats(batch_result["binary_logit"]),
        families=[floats(r) for r in batch_result["families"]],
        family_logits=[floats(r) for r in batch_result["family_logits"]],
        pooled=[floats(r) for r in batch_result["pooled"]],
        max_abs_diff_vs_single=diffs,
    )

    severities = torch.tensor([r["severity"] for r in by_name.values()])
    print(
        f"severity over {len(severities)} samples: min={severities.min():.4f} "
        f"max={severities.max():.4f} mean={severities.mean():.4f} std={severities.std():.4f}"
    )

    fixtures = dict(
        model_sha256=sha256_file(args.checkpoint),
        checkpoint_dtypes=dtypes,
        dtype="float32",
        device=str(device),
        torch_version=torch.__version__,
        transformers_version=__import__("transformers").__version__,
        python_version=platform.python_version(),
        prompt_template=PROMPT,
        max_length=config.max_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        families_order=list(CWE_FAMILY_NAMES),
        sources=[LLAMA.describe(), JUICE.describe()],
        samples=list(by_name.values()),
        batch=batch,
    )

    fixtures_path = args.out_dir / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures, ensure_ascii=False))
    print(f"wrote {fixtures_path}")

    # Layer-by-layer hidden states for one short sample.
    layer_prompt = by_name[LAYER_SAMPLE]["prompt"]
    captured = dump_layers(model, encode([layer_prompt], tokenizer, config), device, args.out_dir / "layers.safetensors")
    pooled_diff = float((captured["pooled"] - torch.tensor(by_name[LAYER_SAMPLE]["pooled"])).abs().max())
    weights = captured["pool_weights"]
    print(
        f"wrote {args.out_dir / 'layers.safetensors'} ({len(captured)} tensors, T={captured['norm'].shape[0]}); "
        f"pooled vs fixture diff={pooled_diff:.3g}, "
        f"pool weights sum={float(weights.sum()):.6f} max={float(weights.max()):.4f}"
    )

    # Pure tokenizer cases.
    cases = [
        dict(text=text, ids=tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in tokenizer_texts(samples)
    ]
    tokenizer_path = args.out_dir / "tokenizer_cases.json"
    tokenizer_path.write_text(
        json.dumps(
            dict(tokenizer="Qwen3.5-0.8B-Base/tokenizer.json",
                 tokenizer_sha256=sha256_file(Path(config.model_path) / "tokenizer.json"),
                 sources=[LLAMA.describe(), JUICE.describe()],
                 transformers_version=fixtures["transformers_version"],
                 cases=cases),
            ensure_ascii=False,
        )
    )
    print(f"wrote {tokenizer_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
