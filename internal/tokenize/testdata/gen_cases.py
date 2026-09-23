#!/usr/bin/env python3
"""Generate tokenizer parity fixtures for internal/tokenize.

Requires transformers==5.17.0 and tokenizers==0.23.2 (requirements.txt).

Ground truth is exactly what training used:

    AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)(text)["input_ids"]

Usage (from the repo root):

    /Users/shearer/vaktex/classification_models/.venv/bin/python \
        internal/tokenize/testdata/gen_cases.py \
        --model /Users/shearer/vaktex/classification_models/Qwen3.5-0.8B-Base \
        --src /Users/shearer/vaktex/llama.cpp --src /Users/shearer/vaktex/juice-shop

Output: internal/tokenize/testdata/cases.json

Each case has "name", "ids" and either "text" or "raw_b64". A raw_b64 case is
arbitrary bytes (usually invalid UTF-8); its ids come from tokenizing
bytes.decode("utf-8", errors="replace"), which is what reading the file in
Python with errors="replace" gives. Rendered cases also carry "language" and
"code" so the Go test can check Render() produced the same text.

Snippet selection is seeded and depends only on the file list of the source
trees, so reruns against the same checkouts give the same fixtures.
"""

import argparse
import base64
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer

PROMPT = "Language: {language}\nCode:\n{code}"  # experiment/encoding.py

# extension -> language name used in the rendered prompt
LANGS = {
    ".c": "C",
    ".cpp": "C++",
    ".h": "C++",
    ".hpp": "C++",
    ".cu": "CUDA",
    ".cuh": "CUDA",
    ".py": "Python",
    ".ts": "TypeScript",
    ".js": "JavaScript",
    ".mjs": "JavaScript",
    ".svelte": "Svelte",
    ".sh": "Bash",
    ".cl": "OpenCL",
    ".comp": "GLSL",
    ".glsl": "GLSL",
    ".wgsl": "WGSL",
    ".metal": "Metal",
    ".cmake": "CMake",
    ".kt": "Kotlin",
    ".swift": "Swift",
    ".nix": "Nix",
    ".json": "JSON",
    ".yml": "YAML",
    ".yaml": "YAML",
    ".gbnf": "GBNF",
    ".jinja": "Jinja",
    ".bat": "Batch",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sol": "Solidity",
    ".pug": "Pug",
    ".md": "Markdown",
    ".hbs": "Handlebars",
}

MAX_FILE = 256 * 1024
MAX_OUTPUT = 2 * 1024 * 1024


def git_files(root: Path):
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return sorted(p for p in out.decode("utf-8", "replace").split("\0") if p)


def pick_snippets(roots, rng, per_lang):
    by_lang = {}
    for root in roots:
        for rel in git_files(root):
            ext = Path(rel).suffix.lower()
            if ext in LANGS:
                by_lang.setdefault(LANGS[ext], []).append(root / rel)

    snippets = []
    for lang in sorted(by_lang):
        files = by_lang[lang]
        rng.shuffle(files)
        taken = 0
        for path in files:
            if taken >= per_lang:
                break
            try:
                if not path.is_file() or path.stat().st_size > MAX_FILE:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:8192] or not data.strip():
                continue
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            n = rng.choice([3, 8, 20, 40, 80])
            start = rng.randrange(max(1, len(lines) - n + 1))
            code = "".join(lines[start:start + n])
            if not code.strip():
                continue
            snippets.append((lang, str(path.name), code))
            taken += 1
    return snippets, by_lang


ADDED_SAMPLES = [
    'prompt = "<|im_start|>user\\n" + msg + "<|im_end|>\\n"',
    "if tok == '<|endoftext|>': break",
    "abc<|endoftext|>def",
    "<|im_start|><|im_start|><|im_end|>",
    "<|im_start",
    "|im_start|>",
    "<|IM_START|> <|Im_Start|>",
    "< |im_start| >",
    "<<|im_start|>>",
    "  <|im_start|>  \n\t<|im_end|>\n",
    "<think>\nreasoning\n</think>\n\nanswer",
    "html = '<think>' + '</think>'",
    "<tool_call>\n{\"name\": \"f\"}\n</tool_call>",
    "<tool_response>ok</tool_response>",
    "x<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>    return 1",
    "<|repo_name|>vaktex/vakt\n<|file_sep|>main.go\npackage main",
    "<|vision_start|><|image_pad|><|vision_end|>",
    "<|audio_start|><|audio_pad|><|audio_end|>",
    "<tts_pad><tts_text_bos><tts_text_eod><tts_text_bos_single>",
    "<|object_ref_start|>a<|object_ref_end|><|box_start|>(1,2)<|box_end|>",
    "<|quad_start|><|quad_end|><|video_pad|><|vision_pad|><|fim_pad|>",
    "// <|endoftext|> should never appear in user input\nint x = 0;",
    "é<|im_start|>é",  # NFC around an added token
    "<|endoftext|>" * 5,
]


def edge_cases():
    c = {}
    c["empty"] = ""
    c["space"] = " "
    c["newline"] = "\n"
    c["crlf_code"] = "int main() {\r\n\treturn 0;\r\n}\r\n"
    c["crlf_blank_lines"] = "a\r\n\r\n\r\nb\r\n"
    c["cr_only"] = "line1\rline2\r\rline3"
    c["mixed_eol"] = "a\nb\r\nc\rd\n\r"
    c["tabs"] = "\tif (x) {\n\t\tfoo();\n\t}\n"
    c["tab_run"] = "a" + "\t" * 37 + "b"
    c["spaces_1000"] = " " * 1000
    c["spaces_mid"] = "x" + " " * 257 + "y"
    c["spaces_trailing"] = "x = 1    \n    \n\ny = 2   "
    c["mixed_ws"] = " \t \t\n \n\t\t \r\n   \f\v x"
    c["indent_deep"] = "".join(" " * (4 * i) + f"level{i}\n" for i in range(24))
    c["blank_lines_200"] = "\n" * 200
    c["unicode_ws"] = "a\u00a0b\u3000c\u2028d\u2029e\u0085f\u200bg\ufeffh"
    c["bom_start"] = "\ufeffpackage main\n"
    c["emoji"] = "print('🚀🔥 done ✅')  # 👨‍👩‍👧‍👦 🇮🇪 🏳️‍🌈"
    c["emoji_skin"] = "👍🏽👍🏿✋🏻"
    c["cjk"] = "// 这是一个测试函数，用于计算总和\nfunction 合計(配列) { return 配列.reduce((a, b) => a + b); }"
    c["japanese"] = "# 日本語のコメント：ひらがなとカタカナ\nprint('こんにちは世界')"
    c["korean"] = "// 한국어 주석입니다\nlet 이름 = \"홍길동\";"
    c["arabic"] = "# تعليق باللغة العربية\nx = 'مرحبا بالعالم'"
    c["hebrew"] = "// הערה בעברית\nconst s = \"שלום\";"
    c["devanagari"] = "नमस्ते दुनिया # हिन्दी टिप्पणी"
    c["thai"] = "ไทย ภาษา // ความคิดเห็น"
    c["greek_cyrillic"] = "λ = lambda x: x  # Привет мир, Ωμέγα"
    c["combining_nfd"] = "cafe\u0301 re\u0301sume\u0301 n\u0303"
    c["combining_stack"] = "e\u0301\u0302\u0323z a\u0308\u0304"
    c["lone_combining"] = "\u0301x \u0301 \u0301\u0301"
    c["hangul_jamo"] = "\u1100\u1161\u11a8 \u1112\u1161\u11ab"
    c["compat_chars"] = "ﬁ ﬂ ① ² ㎏ Ⅻ ｆｕｌｌｗｉｄｔｈ"
    c["kelvin_long_s"] = "it'S x'ſ y'K we'LL they'Re I'M DON'T"
    c["contractions"] = "don't won't it's they're we've I'm you'll he'd"
    c["numbers"] = "0 1 12 123 1234567890 3.14159 -42 1e-9 6.02e23 0xDEADBEEF 0b1010 1_000_000"
    c["numbers_unicode"] = "٠١٢٣ ۴۵۶ १२३ １２３ ½ ⅓ ²³"
    c["long_number"] = "".join(str(i % 10) for i in range(3000))
    c["long_word"] = "a" * 5000
    c["long_identifier"] = "very_long_identifier_" * 200
    c["punct_run"] = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" * 40
    c["arrows"] = "a => b -> c <- d <=> e >>= f :: g ?. h ?? i ..."
    c["html_entities"] = "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &#x27; &amp;"
    c["url"] = "https://example.com/path?q=1&r=%20x#frag user@example.com"
    c["base64_blob"] = base64.b64encode(hashlib.sha256(b"x").digest() * 40).decode()
    c["hex_dump"] = " ".join(f"{b:02x}" for b in range(256))
    c["control_chars"] = "a\x01b\x02c\x07d\x1be[0m\x7ff"
    c["nul_single"] = "a\x00b"
    c["nul_whitespace"] = "x   \x00"
    c["nul_many"] = "\x00\x00 \x00x\x00\n\x00\x00\x00"
    c["nul_and_soh"] = "a\x00b\x01c\x00\x01\x00"
    c["nul_in_code"] = "char buf[] = \"abc\x00def\";\n"
    c["surrogate_like_text"] = "\\ud83d\\ude80 \\u0000 \\x00"
    c["replacement_char"] = "bad \ufffd byte \ufffd\ufffd"
    c["private_use"] = "\ue000\uf8ff\U000f0000"
    c["astral"] = "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟘𝟙𝟚 𓀀 𐍈"
    c["math"] = "∀x∈ℝ: x² ≥ 0 ∧ ∑ᵢ aᵢ ≤ ∞ ≠ ≈"
    c["box_drawing"] = "┌──┐\n│ok│\n└──┘"
    c["sql"] = "SELECT * FROM users WHERE name = '' OR '1'='1'; -- comment\nDROP TABLE x;"
    c["regex_like"] = r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*)$"
    c["prompt_marker_in_code"] = "Language: Python\nCode:\nLanguage: C\nCode:\n"
    for i, s in enumerate(ADDED_SAMPLES):
        c[f"added_token_{i:02d}"] = s
    return c


INVALID_UTF8 = {
    "trunc_3byte": b"a\xe2\x82x",
    "surrogate_encoded": b"\xed\xa0\x80z",
    "trunc_4byte_end": b"ok \xf0\x9f\x98",
    "overlong": b"\xc0\xaf/etc/passwd",
    "above_max": b"\xf4\x90\x80\x80!",
    "ff_fe": b"\xff\xfe int x;",
    "latin1_file": "caf\xe9 na\xefve r\xe9sum\xe9\n".encode("latin-1"),
    "lone_continuation": b"x\x80\x80\x80y",
    "mixed_valid_invalid": "héllo ".encode() + b"\xe9\x80" + " 世界 ".encode() + b"\xf0\x9f" + "🚀".encode(),
    "invalid_then_added": b"\xff<|im_start|>\xfe",
    "invalid_at_start": b"\x80\x81def f(): pass",
    "cp1252_quotes": b"\x93quoted\x94 \x96 dash",
    "invalid_run": bytes(range(0x80, 0xc0)),
    "all_bytes": bytes(range(256)),
    "utf16le_bom": "\ufeffhi".encode("utf-16-le"),
    "e0_bad_second": b"\xe0\x80\x80 \xe0\xa0\x80",
    "f0_bad_second": b"\xf0\x80\x80\x80 \xf0\x90\x80\x80",
    "f4_edge": b"\xf4\x8f\xbf\xbf \xf4\x90",
    "nul_and_invalid": b"a\x00\xffb",
}


def fuzz_cases(rng, n):
    pool = (
        list("abcXYZ019 _-+*/=<>()[]{};:'\",.!?@#$%^&|\\`~")
        + list("\t\n\r\x0b\x0c ")
        + ["\u00a0", "\u3000", "\u2028", "\u0085", "\u200b", "\ufeff", "\x00", "\x01"]
        + list("éüßøñçÅ")
        + ["e\u0301", "\u0301", "\u0308", "\u093f", "\u094d"]
        + list("中文字日本語한국어ไทยहिंदीعربيעבריתΩЖ")
        + ["🚀", "👍🏽", "🇮🇪", "👨‍👩‍👧", "𝔘", "①", "½", "٣", "１", "ﬁ", "ſ", "K"]
        + ["'s", "'T", "'re", "'LL", "'d", "'M", "'Ve"]
        + ["<|im_start|>", "<|endoftext|>", "<think>", "</tool_call>", "<|im_", "|>"]
        + ["    ", "\r\n", "\n\n", "  \n"]
    )
    out = {}
    for i in range(n):
        k = rng.choice([5, 20, 60, 200, 600])
        out[f"fuzz_{i:02d}"] = "".join(rng.choice(pool) for _ in range(k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/shearer/vaktex/classification_models/Qwen3.5-0.8B-Base")
    ap.add_argument("--src", action="append", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("cases.json"))
    ap.add_argument("--per-lang", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    roots = args.src or [Path("/Users/shearer/vaktex/llama.cpp"), Path("/Users/shearer/vaktex/juice-shop")]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    rng = random.Random(args.seed)
    cases = []

    def add(name, text, **extra):
        ids = tok(text)["input_ids"]
        cases.append({"name": name, "text": text, "ids": ids, **extra})

    snippets, by_lang = pick_snippets(roots, rng, args.per_lang)
    for i, (lang, fname, code) in enumerate(snippets):
        name = f"code_{lang}_{i:03d}_{fname}"
        if i % 2 == 0:
            add(name + "_prompt", PROMPT.format(language=lang, code=code), language=lang, code=code)
        else:
            add(name, code)

    # A few whole files (rendered), medium to large, to exercise long inputs.
    big = []
    for lang in ("C++", "TypeScript", "Python", "CUDA"):
        files = sorted(by_lang.get(lang, []), key=lambda p: str(p))
        sized = [p for p in files if 12_000 <= p.stat().st_size <= 40_000]
        if sized:
            big.append((lang, rng.choice(sized)))
    for lang, path in big:
        code = path.read_bytes().decode("utf-8", errors="replace")
        add(f"file_{lang}_{path.name}_prompt", PROMPT.format(language=lang, code=code), language=lang, code=code)

    for name, text in edge_cases().items():
        add(name, text)
    # Rendered prompts around edge content, incl. language names with odd chars.
    for lang, code in [
        ("C#", "Console.WriteLine(\"<|im_end|>\");\r\n"),
        ("C++", "\tauto x = u8\"日本\";\n"),
        ("PL/SQL", "BEGIN NULL; END;\n/\n"),
        ("Objective-C", ""),
        ("Python", "\n\n\n"),
        ("Shell", "echo 🚀\x00done\n"),
    ]:
        add(f"prompt_edge_{lang}", PROMPT.format(language=lang, code=code), language=lang, code=code)

    for name, raw in INVALID_UTF8.items():
        ids = tok(raw.decode("utf-8", errors="replace"))["input_ids"]
        cases.append({"name": "invalid_" + name, "raw_b64": base64.b64encode(raw).decode(), "ids": ids})

    for name, text in fuzz_cases(rng, 40).items():
        add(name, text)

    names = [c["name"] for c in cases]
    assert len(names) == len(set(names)), "duplicate case names"

    # The live backend differs from tokenizer.json on disk (Qwen2Tokenizer
    # rebuilds it); record the parts that decide ids so the Go test can check
    # the embedded asset against them.
    backend = json.loads(tok.backend_tokenizer.to_str())
    split = backend["pre_tokenizer"]["pretokenizers"][0]

    doc = {
        "generator": "internal/tokenize/testdata/gen_cases.py",
        "backend": {
            "normalizer": backend["normalizer"],
            "split_regex": split["pattern"]["Regex"],
            "post_processor": backend["post_processor"],
            "added_tokens": [
                {k: a[k] for k in ("id", "content", "single_word", "lstrip", "rstrip", "normalized", "special")}
                for a in backend["added_tokens"]
            ],
        },
        "model": Path(args.model).name,
        "tokenizer_json_sha256": hashlib.sha256((Path(args.model) / "tokenizer.json").read_bytes()).hexdigest(),
        "transformers_version": transformers.__version__,
        "call": "AutoTokenizer.from_pretrained(model, use_fast=True)(text)['input_ids']",
        "languages": sorted({c["language"] for c in cases if "language" in c} | {lang for lang, _, _ in snippets}),
        "cases": cases,
    }
    data = json.dumps(doc, ensure_ascii=True, separators=(",", ":")).encode()
    if len(data) > MAX_OUTPUT:
        sys.exit(f"cases.json would be {len(data)} bytes (> {MAX_OUTPUT}); lower --per-lang")
    args.out.write_bytes(data + b"\n")
    print(f"wrote {len(cases)} cases, {len(doc['languages'])} languages, {len(data)} bytes -> {args.out}")


if __name__ == "__main__":
    main()
