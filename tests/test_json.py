"""Byte-for-byte checks of the C JSON layer against CPython's json / repr / round."""

from __future__ import annotations

import json
import math
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "clef-tool"


def run(mode: str, lines: list[str]) -> list[str]:
    out = subprocess.run([TOOL, mode], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    return out.stdout.split("\n")[: len(lines)]


def bits(x: float) -> str:
    return struct.pack(">d", x).hex()


def float_cases(rng: random.Random, n: int) -> list[float]:
    xs: list[float] = []
    for e in range(-1074, 1024):
        xs += [math.ldexp(1.0, e), -math.ldexp(1.0, e)]
    for _ in range(n):
        kind = rng.randrange(6)
        if kind == 0:
            x = struct.unpack(">d", rng.getrandbits(64).to_bytes(8, "big"))[0]
            if math.isfinite(x):
                xs.append(x)
        elif kind == 1:
            xs.append(float(f"{rng.randint(1, 10**rng.randint(1, 17))}e{rng.randint(-30, 30)}"))
        elif kind == 2:
            xs.append(rng.random())
        elif kind == 3:
            xs.append(float(rng.randint(-10**18, 10**18)))
        elif kind == 4:
            xs.append(10.0 ** rng.randint(-20, 22) * rng.choice([1, 9.999999999999998, 1.0000000000000002]))
        else:
            xs.append(struct.unpack(">d", (rng.getrandbits(52)).to_bytes(8, "big"))[0])  # subnormals
    xs += [0.0, -0.0, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5, 0.0001, 5e-324, 1.7976931348623157e308]
    return xs


def test_float_repr(rng: random.Random) -> int:
    xs = float_cases(rng, 400_000)
    got = run("float", [bits(x) for x in xs])
    bad = [(x, g) for x, g in zip(xs, got) if g != repr(x)]
    for x, g in bad[:10]:
        print(f"  repr mismatch {x.hex()}: python={x!r} c={g}")
    print(f"float repr: {len(xs) - len(bad)}/{len(xs)} match")
    return len(bad)


def test_round(rng: random.Random) -> int:
    cases = []
    for _ in range(200_000):
        cases.append((rng.random(), 4))
    for k in range(0, 20001):
        cases.append((k / 20000 + 0.00005, 4))  # ties and near-ties at the 4th decimal
        cases.append((k * 0.00005, 4))
    got = run("round", [f"{bits(x)} {d}" for x, d in cases])
    bad = [(x, d, g) for (x, d), g in zip(cases, got) if g != repr(round(x, d))]
    for x, d, g in bad[:10]:
        print(f"  round mismatch {x!r},{d}: python={round(x, d)!r} c={g}")
    print(f"round: {len(cases) - len(bad)}/{len(cases)} match")
    return len(bad)


ALPHABET = ["a", "Z", " ", "\n", "\t", "\x00", "\x1f", "\x7f", '"', "\\", "/", "é", "é", "顧",
            "🚨", " ", "﻿", "ſ", "\U0010ffff"]


def rand_str(rng: random.Random) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 12)))


def rand_value(rng: random.Random, depth: int = 0):
    kind = rng.randrange(9 if depth < 4 else 6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.choice([True, False])
    if kind == 2:
        return rng.choice([0, -1, 7, 10**30, -(10**25), rng.randint(-10**6, 10**6)])
    if kind == 3:
        return rng.choice([0.5, -0.0, 1e-9, 1250.0, 3.14159, 1e300, rng.random() * 10 ** rng.randint(-8, 20)])
    if kind in (4, 5):
        return rand_str(rng)
    if kind == 6:
        return [rand_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rand_str(rng): rand_value(rng, depth + 1) for _ in range(rng.randint(0, 5))}


def py_dump(text: str) -> str:
    try:
        return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (ValueError, RecursionError):
        return "ERR"


def test_json(rng: random.Random) -> int:
    docs: list[str] = []
    for _ in range(20_000):
        v = rand_value(rng)
        docs.append(json.dumps(v, ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 1]),
                               separators=rng.choice([None, (",", ":"), (" , ", " : ")])).replace("\n", " "))
    docs += [
        '{"a":1,"b":2,"a":3}', '{"b":1,"a":{"y":1,"x":[1,2,{"d":0,"c":-0}]}}', "-0", "-0.0", "1E400",
        "-1e400", "NaN", "Infinity", "-Infinity", "[1.0e2, 1e-7, 12345678901234567890123]",
        '"\\ud83d\\ude80"', '"\\u00e9"', '"\\u0000"', '  {"k" : [ ] }  ', "1.5e", "01", "[1,]",
        '{"a":1,}', '"\\ud800"', '"\\udc00x"', "tru", '"\x01"', "[" * 200 + "]" * 200, "",
        '{"é":1,"e":2,"z":3,"Z":4,"\\u0000":5}', "1" * 5000,
    ]
    # duplicate keys in objects large enough to use the hashed key index (> 16 keys)
    rng2 = random.Random(5)
    for _ in range(300):
        keys = [f"k{rng2.randint(0, 40)}" for _ in range(rng2.randint(17, 120))]
        docs.append("{" + ",".join(f'"{k}":{i}' for i, k in enumerate(keys)) + "}")
    # Deliberate divergences: these must be errors in C even though json.loads accepts them.
    #  - lone surrogates: the reference accepts them in json.loads but its tokenizer then
    #    raises, so the request fails either way; C fails at parse time.
    #  - nesting > 512: bounded recursion for server thread stacks (Python allows ~1000).
    must_error = ['"\\ud800"', '"\\udc00x"', "[" * 600 + "]" * 600]
    docs += ["[" * 500 + "]" * 500]
    got_err = run("json", must_error)
    bad = [(d, "ERR (expected divergence)", g) for d, g in zip(must_error, got_err) if not g.startswith("ERR ")]
    docs = [d for d in docs if d not in must_error]
    got = run("json", docs)
    for d, g in zip(docs, got):
        want = py_dump(d)
        if (want == "ERR") != g.startswith("ERR ") or (want != "ERR" and g != want):
            bad.append((d, want, g))
    for d, want, g in bad[:10]:
        print(f"  json mismatch {d[:80]!r}\n    python={want[:120]!r}\n    c     ={g[:120]!r}")
    print(f"json: {len(docs) - len(bad)}/{len(docs)} match")
    return len(bad)


def main() -> None:
    rng = random.Random(1234)
    failures = test_float_repr(rng) + test_round(rng) + test_json(rng)
    if failures:
        sys.exit(f"FAIL: {failures} mismatches")
    print("PASS")


if __name__ == "__main__":
    main()
