"""Over-long state: rejected exactly when the reference would silently drop state tokens.

Usage: test_truncation.py MODEL.gguf HF_DIR
For states around the 16,384-token limit, `clef-tool encode-notrunc` must fail if and only if
Python's encode_record keeps fewer state tokens than the full tokenization has, and when it
succeeds its ids must equal encode_record's. Default (reference) encoding must still truncate
like Python.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tool(mode, gguf, reqs):
    out = subprocess.run([ROOT / "clef-tool", mode, gguf], input="\n".join(json.dumps(r) for r in reqs) + "\n",
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [None if o.startswith("ERR") else json.loads(o)["input_ids"] for o in out]


def main() -> None:
    gguf, hf_dir = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, str(hf_dir))
    from joint_schema_model import encode_record, render
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_dir)
    q = {"q": {"type": "choice", "instructions": "Which service?", "criteria": {"a": "api", "b": "db"}}}
    probe = encode_record(tok, {"state": "", "questions": q})
    fixed = len(probe.input_ids)
    word = "alpha "                                      # one token per repetition
    reqs, full_counts = [], []
    for n in list(range(16384 - fixed - 3, 16384 - fixed + 4)) + [100, 20000]:
        state = word * n
        reqs.append({"model": "clef", "state": state, "questions": q})
        full_counts.append(len(tok(render(state), add_special_tokens=False).input_ids))
    strict_ids, ref_ids = tool("encode-notrunc", gguf, reqs), tool("encode", gguf, reqs)
    fails = 0
    for r, full, s_ids, r_ids in zip(reqs, full_counts, strict_ids, ref_ids):
        want = list(encode_record(tok, r).input_ids)
        truncated = len(want) - fixed < full
        ok = (s_ids is None) == truncated and (s_ids is None or s_ids == want) and r_ids == want
        fails += not ok
        print(f"state tokens {full:6d} (fit {16384 - fixed}): reference truncates={truncated!s:5}  "
              f"--no-truncate {'rejects' if s_ids is None else 'accepts'}  {'OK' if ok else 'FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
