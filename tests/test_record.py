"""Parity of the C request encoder and response builder against joint_schema_model.

Usage: test_record.py MODEL.gguf HF_TOKENIZER_DIR REFERENCE_CODE_DIR
  encode : C encode_request == Python systemone() validation + encode_record (ids, spans, option ids)
  respond: C response JSON == Python systemone_answer over the same float32 probabilities
"""

from __future__ import annotations

import json
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
import corpus  # noqa: E402

KEYS = ["a", "b", "B", "é", "é", "10", "2", "id with space", "顧客", "🚨", "", "\"q\"", "x" * 40]
VALUES = [None, True, False, 0, -0.0, 1e-9, 3, 12345678901234567890, "", "text", "Ünï\ncode", [1, "x"], {"z": 1, "a": [None]}]


def rand_question(rng: random.Random) -> dict:
    t = rng.choice(["noul", "choice", "score", "noul", "choice", "score", "bogus"])
    q: dict = {"type": t}
    r = rng.random()
    if r < 0.25:
        pass
    elif r < 0.35:
        q["instructions"] = ""
    elif r < 0.45:
        q["instructions"] = None
    elif r < 0.55:
        q["instructions"] = rng.choice(VALUES[1:])
    else:
        q["instructions"] = rng.choice(["Is it urgent?", "Pick one.", "Rate it", "顧客の意図は?"])
    if t == "choice":
        keys = rng.sample(KEYS, rng.randint(0, 6))
        q["criteria"] = {k: rng.choice(VALUES + ["desc " + k]) for k in keys}
    elif t == "score":
        q["criteria"] = [rng.choice(VALUES + ["level"]) for _ in range(rng.randint(0, 6))]
    elif t == "noul" and rng.random() < 0.4:
        q["criteria"] = rng.choice([{}, {"true": "Yes."}, {"false": None}, {"true": 1, "false": [2], "other": 3}])
    return q


def rand_request(rng: random.Random) -> dict:
    qs = {}
    for _ in range(rng.randint(0, 5)):
        qs[rng.choice(KEYS + ["q1", "q2", "q3"])] = rand_question(rng)
    return {
        "model": rng.choice(["clef", "clef-flash"]),
        "state": rng.choice(VALUES + ["Short state.", {"nested": {"b": 2, "a": 1}}, "word " * rng.randint(0, 3000)]),
        "questions": qs,
    }


def py_encode(tok, request):
    from joint_schema_model import encode_record, QUESTION_TYPES

    questions = request.get("questions")
    if not isinstance(request.get("model"), str) or "state" not in request:
        return "ERR"
    if not isinstance(questions, dict) or not questions:
        return "ERR"
    for question in questions.values():
        if question.get("type") not in QUESTION_TYPES:
            return "ERR"
        if question["type"] != "noul" and not question.get("criteria"):
            return "ERR"
        # documented divergence: score criteria must be a list in C
        if question["type"] == "score" and not isinstance(question["criteria"], list):
            return "ERR"
    for qid, question in questions.items():
        # documented divergence: an empty id with no instructions is an empty span (NaN in the reference)
        ins = question.get("instructions")
        if qid == "" and (ins is None or ins == ""):
            return "ERR"
    try:
        e = encode_record(tok, request, max_length=16384)
    except Exception:
        return "ERR"
    return {
        "input_ids": list(e.input_ids),
        "questions": [
            {"id": q.question_id, "type": q.question_type, "span": list(q.question_span),
             "option_spans": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
            for q in e.questions
        ],
    }


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def main() -> None:
    gguf_path, tok_dir, code_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, code_dir)
    from joint_schema_model import encode_record, systemone_answer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_dir)
    rng = random.Random(11)
    requests = corpus.build() + [rand_request(rng) for _ in range(3000)]
    # states far beyond max_length: the C encoder stops tokenizing early (review #2, M3); the
    # kept prefix must equal Python's tokenize-everything-then-truncate result
    for n in (20000, 60000):
        words = ["alpha", "é", "e\u0301", "顧客", "  ", "\n", "1234", "!!", "<|im_end|>", "don't"]
        big = "".join(rng.choice(words) + rng.choice([" ", "", "\t"]) for _ in range(n))
        requests.append({"model": "clef", "state": big, "questions": {"q": {"type": "noul"}}})
        requests.append({"model": "clef", "state": {"log": big}, "questions": {"q": {"type": "score", "criteria": ["a", "b"]}}})
    # schema larger than max_length must be rejected the same way
    requests.append({"model": "clef", "state": "s", "questions": {
        f"q{i}": {"type": "choice", "criteria": {f"o{j}": "x " * 50 for j in range(40)}} for i in range(12)}})

    lines = [json.dumps(r, ensure_ascii=False) for r in requests]
    out = subprocess.run([ROOT / "clef-tool", "encode", gguf_path], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")[: len(lines)]
    bad = 0
    n_ok = 0
    for i, (req, got) in enumerate(zip(requests, out)):
        want = py_encode(tok, req)
        g = "ERR" if got.startswith("ERR") else json.loads(got)
        if g != want:
            bad += 1
            if bad <= 8:
                print(f"  encode mismatch #{i}: python={str(want)[:200]}\n    c={got[:200]}")
        elif want != "ERR":
            n_ok += 1
    print(f"encode: {len(requests) - bad}/{len(requests)} match ({n_ok} encoded, rest rejected by both)")

    # Responses over float32 probabilities, as model(...).float().softmax(-1).tolist() yields.
    resp_lines, wants = [], []
    for req in requests:
        enc = py_encode(tok, req)
        if enc == "ERR":
            continue
        probs = []
        for q in enc["questions"]:
            raw = [rng.random() ** rng.choice([1, 4, 12]) for _ in q["option_ids"]]
            if rng.random() < 0.2 and len(raw) > 1:
                raw[1] = raw[0]  # ties: first maximum must win
            s = sum(raw) or 1.0
            probs.append([f32(x / s) for x in raw])
        answers = {
            q["id"]: systemone_answer(req["questions"][q["id"]], dict(zip(q["option_ids"], p)))
            for q, p in zip(enc["questions"], probs)
        }
        wants.append(json.dumps({"model": req["model"], "answers": answers,
                                 "usage": {"input_tokens": len(enc["input_ids"]), "output_tokens": 0}},
                                ensure_ascii=False, separators=(",", ":")))
        resp_lines.append(json.dumps({"request": req, "probs": probs}, ensure_ascii=False))
    got = subprocess.run([ROOT / "clef-tool", "respond", gguf_path], input="\n".join(resp_lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.split("\n")[: len(resp_lines)]
    rbad = 0
    for w, g in zip(wants, got):
        if w != g:
            rbad += 1
            if rbad <= 5:
                print(f"  respond mismatch\n    python={w[:300]}\n    c     ={g[:300]}")
    print(f"respond: {len(wants) - rbad}/{len(wants)} match")
    if bad or rbad:
        sys.exit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
