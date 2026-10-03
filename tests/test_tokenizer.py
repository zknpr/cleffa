"""Token-id parity of the C tokenizer against the HF tokenizer used by the reference.

Usage: test_tokenizer.py MODEL.gguf HF_TOKENIZER_DIR
Compares tokenizer(text, add_special_tokens=False).input_ids for: every text fragment
encode_record produces for the request corpus, edge-case strings, and random fuzz.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
import corpus  # noqa: E402

PIECES = [
    "a", "Z", "é", "é", "É", "q̣̇", "ậ", "각", "가",
    "한국어", "顧客", "ひらがな", "カタカナ", "عربي", "١٢", "１２３", "Ⅻ", "½", "²", "Ω", "Å", "Å",
    "ﬁ", "ſ", "'s", "'S", "'ſ", "'ll", "'LL", "'re", "'ve", "'m", "'d", "'t", "'x", "''", "'",
    " ", "  ", "   ", "\t", "\n", "\r\n", "\n\n", " \n", " ", "　", " ", "​", "﻿",
    "!", "?!", "...", "—", "“", "”", "«", "»", "()", "{}", "[]", "==", "->", "#", "@", "$", "%", "€",
    "0", "7", "42", "3.14", "1e-9", "-1", "🚨", "👍🏽", "👨‍👩‍👧", "🇮🇹", "\U0001f600",
    "<|im_end|>", "<|im_start|>", "<think>", "</think>", "<|im_", "<|endoftext|>", "<tool_call>", "<", "<<|im_end|>",
    "hello", "Hello", "HELLO", "world", "the", "def", "return", "\\n", "\\u00e9", '"', "\\",
]


def fuzz(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        k = rng.randint(1, 24)
        s = "".join(rng.choice(PIECES) for _ in range(k))
        if rng.random() < 0.3:
            s = " ".join(s.split(" "))
        out.append(s)
    for _ in range(n // 4):
        # random scalar values from all planes
        cps = []
        while len(cps) < rng.randint(1, 16):
            cp = rng.choice([rng.randint(0x20, 0x7f), rng.randint(0x80, 0x2fff), rng.randint(0x3000, 0xd7ff),
                             rng.randint(0xe000, 0xffff), rng.randint(0x10000, 0x10ffff)])
            cps.append(chr(cp))
        out.append("".join(cps))
    return out


def corpus_fragments(tokenizer) -> list[str]:
    """Every string encode_record passes to the tokenizer, for the whole corpus."""
    sys.path.insert(0, str(ROOT / "model-flash"))
    from joint_schema_model import render, question_options, SYSTEM_PROMPT

    frags = [
        "\n\nSCHEMA FIELDS:\n", "\nALLOWED OPTIONS:\n", "\n", "END FIELD\n",
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    ]
    for record in corpus.build():
        frags.append(render(record["state"]))
        for qi, (qid, q) in enumerate(record["questions"].items()):
            frags.append(f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
            ins = q.get("instructions")
            frags.append(render(str(qid) if ins is None or ins == "" else ins))
            for oi, (oid, desc) in enumerate(question_options(q)):
                frags.append(f"OPTION {oi + 1}: ")
                sem = {"option_id": oid}
                if desc is not None:
                    sem["description"] = desc
                frags.append(render(sem))
    return frags


def main() -> None:
    gguf_path, tok_dir = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    rng = random.Random(7)
    texts = corpus_fragments(tokenizer)
    n_corpus = len(texts)
    texts += PIECES + fuzz(rng, 20_000)
    # long runs of combining marks in non-canonical order (NFC reordering at scale)
    marks = ["\u0301", "\u0316", "\u0323", "\u0307", "\u0334", "\u05b0", "\u0345", "\u302a", "\u0f71"]
    texts += ["a" + "\u0301" * 2000 + "\u0316" * 2000]
    texts += ["".join(rng.choice(["a", "e", "o", " ", "x"] + marks * 3) for _ in range(rng.randint(50, 3000))) for _ in range(100)]
    texts += ["a" * 20000, "=" * 8000, " " * 5000 + "x", "\n" * 3000, "ab" * 6000, "é" * 4000,
              (README := (ROOT / "model-flash" / "README.md").read_text())]

    t0 = time.perf_counter()
    proc = subprocess.run([ROOT / "clef-tool", "tok", gguf_path],
                          input="\n".join(json.dumps(t) for t in texts) + "\n",
                          capture_output=True, text=True, check=True)
    c_s = time.perf_counter() - t0
    got = proc.stdout.split("\n")[: len(texts)]

    t0 = time.perf_counter()
    want = [tokenizer(t, add_special_tokens=False).input_ids for t in texts]
    hf_s = time.perf_counter() - t0

    bad = 0
    for i, (t, w, g) in enumerate(zip(texts, want, got)):
        gi = [int(x) for x in g.split()] if g and not g.startswith("ERR") else g
        if gi != w:
            bad += 1
            if bad <= 15:
                print(f"  mismatch #{i} {'(corpus)' if i < n_corpus else ''} text={t[:60]!r}")
                print(f"    hf={w[:20]}\n    c ={gi[:20] if isinstance(gi, list) else gi}")
    print(f"tokenizer: {len(texts) - bad}/{len(texts)} match ({n_corpus} corpus fragments); "
          f"C {c_s:.2f}s incl. load, HF {hf_s:.2f}s")
    if bad:
        sys.exit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
