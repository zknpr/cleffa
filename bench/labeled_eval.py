"""Labeled intent-classification evals (BANKING77, CLINC150+OOS), paired across arms.

Every arm answers the same requests: cleffa (C), Cloudflare hosted (H, optionally two passes)
and stock HF BF16 PyTorch (B). Labels live in a separate file and never reach an arm.

Requests are built exactly as the community Decision Index 0.2.1 harness builds them
(github.com/apolinario/decision-index, MIT, commit 87d4650, suite/build/normalize_direct.py
and layout.choice_row): empty state, one choice question q1 whose instructions carry the user
text, options keyed option_0..option_N. The model cards' BANKING77/CLINC150+OOS macro-F1 come
from Cloudflare's self-reported run of that suite, so full-split scores here are directly
comparable to them (Cloudflare's run artifacts are unpublished, so not row by row).

  fetch DIR                       download the pinned sources and verify their SHA-256
  build DATASET SRC OUT           write OUT/{requests,labels}.jsonl and OUT/manifest.json
  capture-local REQ OUT ...       run ./clef over the requests (GPU; take the shared lock)
  capture-pytorch REQ OUT ...     run the snapshot's own systemone() on stock BF16 (GPU)
  plan-hosted REQ OUT ...         plan, or with --run collect, hosted answers (bills neurons)
  score LABELS OUT --arm NAME=FILE [--arm ...]

Response files from capture-* carry {"id", "request_sha256", "response"} per line. The hash
is over the exact bytes sent to that arm, so a response cannot be attributed to a request
it was not produced from. Hosted journals are the bench/cloudflare_checkout.collect format.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import urllib.request

import numpy as np
from compare_cloudflare import load_hosted_bytes, distribution, decision, confidence_value

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "banking77": {
        "repo": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "commit": "57ec275d8078af65b7731c2a98be812d844a6d6b",
        "files": {
            "test.csv": ("banking_data/test.csv",
                         "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
            "categories.json": ("banking_data/categories.json",
                                "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63"),
        },
    },
    "clinc150": {
        "repo": "https://github.com/clinc/oos-eval",
        "commit": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
        "files": {
            "data_full.json": ("data/data_full.json",
                               "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"),
        },
    },
}

# Decision Index 0.2.1 instruction templates and benchmark names, verbatim.
TEMPLATES = {
    "banking77": ("BANKING77", "Classify the banking intent of this user request:\n"),
    "clinc150": ("CLINC150+OOS", "Classify the intent of this user request, or choose out of scope if none applies:\n"),
}
OOS_DESCRIPTION = "out of scope: none of the listed intents"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(out: Path) -> None:
    for name, src in SOURCES.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for local, (remote, digest) in src["files"].items():
            path = folder / local
            if not path.exists():
                raw = src["repo"].replace("github.com", "raw.githubusercontent.com")
                url = f"{raw}/{src['commit']}/{remote}"
                with urllib.request.urlopen(url, timeout=60) as r, path.open("xb") as f:
                    f.write(r.read(64 << 20))
            if sha256_file(path) != digest:
                raise ValueError(f"{path}: SHA-256 differs from the pinned source")
    print(f"sources verified in {out}")


def load_examples(dataset: str, src: Path) -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    """Return (split, text, label) rows, the label order and the option descriptions, as the
    Decision Index builder derives them."""
    for local, (_, digest) in SOURCES[dataset]["files"].items():
        if sha256_file(src / dataset / local) != digest:
            raise ValueError(f"{dataset}/{local}: SHA-256 differs from the pinned source")
    if dataset == "banking77":
        with (src / dataset / "test.csv").open(newline="", encoding="utf-8") as f:
            rows = [("test", r["text"], r["category"]) for r in csv.DictReader(f)]
        labels = json.loads((src / dataset / "categories.json").read_text(encoding="utf-8"))
        descriptions = list(labels)
    else:
        data = json.loads((src / dataset / "data_full.json").read_text(encoding="utf-8"))
        rows = [(split, t, l) for split in ("test", "oos_test") for t, l in data[split]]
        labels = sorted({l for _, l in data["train"]} | {"oos"})
        descriptions = [OOS_DESCRIPTION if l == "oos" else l.replace("_", " ") for l in labels]
    if len(labels) != {"banking77": 77, "clinc150": 151}[dataset] or {l for *_, l in rows} - set(labels):
        raise ValueError(f"{dataset}: label set differs from the Decision Index builder's")
    return rows, labels, descriptions


def build(dataset: str, src: Path, out: Path, per_label: int | None, seed: int) -> None:
    rows, labels, descriptions = load_examples(dataset, src)
    # Source ids restart per split, as in the harness ("CLINC150+OOS:oos_test:0").
    seen: dict[str, int] = {}
    indexed = []
    for split, text, label in rows:
        indexed.append((split, seen.get(split, 0), text, label))
        seen[split] = seen.get(split, 0) + 1
    if per_label is not None:
        # Stratified: the same number per class keeps macro-F1 from being dominated by a
        # few large classes when only a subsample fits the hosted allowance.
        rng = random.Random(seed)
        by_label: dict[str, list] = {}
        for item in indexed:
            by_label.setdefault(item[3], []).append(item)
        indexed = []
        for label in labels:
            pool = by_label.get(label, [])
            if len(pool) < per_label:
                raise ValueError(f"{dataset}: {label} has {len(pool)} examples, fewer than {per_label}")
            indexed.extend(rng.sample(pool, per_label))
        indexed.sort(key=lambda x: (x[0] != "test", x[1]))
    name, template = TEMPLATES[dataset]
    keys = [chr(65 + i) if len(labels) <= 26 else f"option_{i}" for i in range(len(labels))]
    criteria = dict(zip(keys, descriptions))
    out.mkdir(parents=True, exist_ok=False)
    with (out / "requests.jsonl").open("x", encoding="utf-8") as req, \
            (out / "labels.jsonl").open("x", encoding="utf-8") as lab:
        for split, i, text, label in indexed:
            rid = f"{name}:{split}:{i}"
            question = {"type": "choice", "instructions": template + text, "criteria": criteria}
            req.write(json.dumps({"id": rid, "state": {}, "questions": {"q1": question}},
                                 ensure_ascii=False) + "\n")
            lab.write(json.dumps({"id": rid, "label": keys[labels.index(label)]}) + "\n")
    manifest = {
        "dataset": dataset, "split": "test" + ("+oos_test" if dataset == "clinc150" else ""),
        "source": {k: v for k, v in SOURCES[dataset].items()},
        "examples": len(indexed), "labels": len(labels),
        "sampling": {"per_label": per_label, "seed": seed} if per_label else "full test split",
        "harness": "decision-index 0.2.1 (apolinario/decision-index@87d4650)",
        "instruction_template": template, "criteria": criteria,
        "requests_sha256": sha256_file(out / "requests.jsonl"),
        "labels_sha256": sha256_file(out / "labels.jsonl"),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{dataset}: {len(indexed)} requests, {len(labels)} labels -> {out}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def arm_request(row: dict, model: str) -> dict:
    """The request body an arm receives: the stored request plus the model selector."""
    return {**{k: v for k, v in row.items() if k != "id"}, "model": model}


def capture_local(req: Path, out: Path, binary: Path, gguf: Path, model: str) -> None:
    if out.exists():raise FileExistsError(out)
    rows = read_jsonl(req)
    # Match the hosted collector's exact JSON bytes, including Unicode escapes.
    lines = [json.dumps(arm_request(r, model)) for r in rows]
    # --strict mirrors the server default; benign requests encode identically either way.
    # Resolve: Path("./clef") normalizes to "clef", which subprocess would look up on PATH.
    proc = subprocess.run([str(binary.resolve()), "-m", str(gguf), "--strict"], input="\n".join(lines) + "\n",
                          text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"clef exited {proc.returncode}: {proc.stderr[-2000:]}")
    responses = proc.stdout.splitlines()
    if len(responses) != len(rows):
        raise RuntimeError(f"clef returned {len(responses)} responses for {len(rows)} requests")
    with out.open("x", encoding="utf-8") as f:
        for r, line, resp in zip(rows, lines, responses):
            body = json.loads(resp)
            if "error" in body:
                raise RuntimeError(f"{r['id']}: {body['error']}")
            f.write(json.dumps({"id": r["id"], "request_sha256": hashlib.sha256(line.encode()).hexdigest(),
                                "response": body}) + "\n")
    print(f"{len(rows)} local responses -> {out}")


def capture_pytorch(req: Path, out: Path, model_dir: Path, model: str, device: str, dtype: str) -> None:
    # Stock path: the snapshot's own loader and systemone(), default attention and experts
    # implementation, no repo patches. BF16 is what a user runs; float32 is the control arm. Verify the snapshot first
    # (tools/verify_snapshot.py) and run with python -B.
    if out.exists():raise FileExistsError(out)
    import torch
    sys.path.insert(0, str(model_dir.resolve()))
    from joint_schema_model import load_release_model, systemone
    net, processor = load_release_model(model_dir, device=device, dtype=getattr(torch, dtype))
    rows = read_jsonl(req)
    with out.open("x", encoding="utf-8") as f, torch.inference_mode():
        for r in rows:
            body = arm_request(r, model)
            line = json.dumps(body)
            f.write(json.dumps({"id": r["id"], "request_sha256": hashlib.sha256(line.encode()).hexdigest(),
                                "response": systemone(net, processor, json.loads(line))}) + "\n")
            f.flush()
    print(f"{len(rows)} PyTorch responses -> {out}")


def plan_hosted(req: Path, out: Path, model: str, passes: int, run: bool, account: str | None,
                clef_tool: Path, gguf: Path | None, paid_max: float | None) -> None:
    sys.path.insert(0, str(ROOT / "bench"))
    from cloudflare_checkout import collect
    from cloudflare_corpus import check_budget, credentials, daily_usage, final_usage, hosted_payload
    # Pricing page, checked 2026-10-10. The flash rate fell on 2026-10-09.
    rates = {"clef-flash": 3455, "clef": 21818, "clef-omni": 13636}
    rows = read_jsonl(req)
    payloads = [hosted_payload(arm_request(r, model), model) for r in rows]
    # Omni needs a clef-tool and GGUF from a checkout that supports it; pass both.
    gguf = gguf or ROOT / "gguf" / f"{model}.gguf"
    enc = subprocess.run([str(clef_tool.resolve()), "encode-strict", str(gguf)],
                         input="".join(json.dumps(p) + "\n" for p in payloads),
                         text=True, capture_output=True, check=True)
    counts = [len(json.loads(l)["input_ids"]) for l in enc.stdout.splitlines()]
    if len(counts) != len(rows):
        raise RuntimeError("encoder returned a different number of records")
    requests = [{"model": model, "id": r["id"], "request": p,
                 "request_sha256": hashlib.sha256(json.dumps(p).encode()).hexdigest(),
                 "full_input_tokens": n} for r, p, n in zip(rows, payloads, counts)]
    tokens = counts[0] + passes * sum(counts)
    estimate = tokens * rates[model] / 1e6
    plan = {"type": "plan", "purpose": "labeled eval", "requests_file_sha256": sha256_file(req),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "passes": passes, "warmups_per_model": 1, "client_retries": 0,
            "planned_calls": 1 + passes * len(requests), "requests": requests,
            "neurons_per_million_input_tokens": rates[model], "full_token_neuron_estimate": estimate}
    print(f"{model}: {plan['planned_calls']} calls, {tokens} input tokens, about {estimate:.1f} neurons")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        f.write(json.dumps({**plan, "mode": "collect" if run else "plan-only"}) + "\n")
        f.flush()
        if not run:
            return
        # Default: stay inside the free daily allowance (cloudflare_corpus.check_budget). A
        # billed run needs the user's explicit approval and an explicit ceiling.
        def budget(used: float) -> None:
            if paid_max is None:
                check_budget(used, estimate)
            elif not (0 < estimate <= paid_max):
                raise ValueError(f"estimate {estimate:.0f} neurons exceeds --paid-max-neurons {paid_max:.0f}")
        budget(0)
        assert account is not None
        acct, token = credentials(account)
        usage = daily_usage(acct, token)
        f.write(json.dumps({**usage, "before_model": model, "paid_max_neurons": paid_max}) + "\n")
        f.flush()
        budget(usage["used_neurons"])
        collect(plan, f, acct, token)
        f.write(json.dumps({"type": "complete", "timestamp":
                            datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
        f.flush()
        final_usage(f, acct, token)


def round_robin(rows: list[dict], gold: dict[str, str]) -> list[dict]:
    """Every label's first example in file order, then every label's second, and so on."""
    rank: dict[str, int] = {}
    order = []
    for r in rows:
        k = rank.get(gold[r["id"]], 0)
        rank[gold[r["id"]]] = k + 1
        order.append((k, r))
    order.sort(key=lambda x: x[0])                     # stable: file order within a round
    return [r for _, r in order]

def hosted_free(req: Path, outdir: Path, model: str, accounts: list[str], clef_tool: Path,
                gguf: Path | None, run: bool) -> None:
    """Collect hosted answers inside each account's free daily allowance, in resumable chunks.

    Requests run label round-robin (every label's first example, then every label's second,
    ...), so a partial run is still a label-balanced subset. Each chunk is one complete
    collect() journal (warm-up included) of at most cloudflare_corpus.RUN_LIMIT neurons on the
    account with the most free allowance left. Completed chunks are skipped on restart; an
    incomplete one is renamed *.failed and its requests are redone. Stops, without error, when
    no listed account can afford the next request; rerun after 00:00 UTC to continue."""
    sys.path.insert(0, str(ROOT / "bench"))
    from cloudflare_checkout import collect
    from cloudflare_corpus import FREE_NEURONS, HEADROOM, RUN_LIMIT, credentials, daily_usage, hosted_payload
    rates = {"clef-flash": 3455, "clef": 21818, "clef-omni": 13636}
    gold = {r["id"]: r["label"] for r in read_jsonl(req.with_name("labels.jsonl"))}
    rows = round_robin(read_jsonl(req), gold)
    payloads = [hosted_payload(arm_request(r, model), model) for r in rows]
    gguf = gguf or ROOT / "gguf" / f"{model}.gguf"
    enc = subprocess.run([str(clef_tool.resolve()), "encode-strict", str(gguf)],
                         input="".join(json.dumps(p) + "\n" for p in payloads),
                         text=True, capture_output=True, check=True)
    counts = [len(json.loads(l)["input_ids"]) for l in enc.stdout.splitlines()]
    if len(counts) != len(rows):
        raise RuntimeError("encoder returned a different number of records")
    planned = [{"model": model, "id": r["id"], "request": p,
                "request_sha256": hashlib.sha256(json.dumps(p).encode()).hexdigest(),
                "full_input_tokens": n} for r, p, n in zip(rows, payloads, counts)]
    cost = {x["id"]: x["full_input_tokens"] * rates[model] / 1e6 for x in planned}
    outdir.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for path in sorted(outdir.glob("chunk-*.jsonl")):
        chunk = read_jsonl(path)
        if not any(r.get("type") == "complete" for r in chunk):
            path.rename(path.with_suffix(".failed"))
            continue
        done |= {r["id"] for r in chunk[0]["requests"]}
    todo = [x for x in planned if x["id"] not in done]
    total = sum(cost[x["id"]] for x in todo)
    print(f"{model}: {len(done)} done, {len(todo)} to go, about {total:.0f} neurons")
    if not run or not todo:
        return
    creds = {a: credentials(a) for a in accounts}
    # Analytics lag: never trust a reading below what this process has already spent.
    spent = {a: 0.0 for a in accounts}
    start = {a: daily_usage(*creds[a])["used_neurons"] for a in accounts}
    n_chunk = len(list(outdir.glob("chunk-*")))
    while todo:
        def left(a: str) -> float:
            used = max(daily_usage(*creds[a])["used_neurons"], start[a] + spent[a])
            return FREE_NEURONS - HEADROOM - used
        room = {a: left(a) for a in accounts}
        acct = max(room, key=lambda a: room[a])
        cap = min(RUN_LIMIT, room[acct])
        chunk, est = [], 0.0
        for x in todo:
            # The warm-up repeats the first request, so it costs twice.
            extra = cost[x["id"]] * (2 if not chunk else 1)
            if est + extra > cap:
                break
            chunk.append(x)
            est += extra
        if not chunk:
            print(f"free allowance exhausted on {', '.join(accounts)}; "
                  f"{len(todo)} requests left; rerun after 00:00 UTC")
            return
        name = "".join(c if c.isalnum() else "-" for c in acct.lower())
        path = outdir / f"chunk-{n_chunk:04d}-{name}.jsonl"
        n_chunk += 1
        plan = {"type": "plan", "purpose": "labeled eval, free-allowance chunk", "account_name": acct,
                "requests_file_sha256": sha256_file(req), "passes": 1, "warmups_per_model": 1,
                "client_retries": 0, "planned_calls": len(chunk) + 1, "requests": chunk,
                "neurons_per_million_input_tokens": rates[model], "full_token_neuron_estimate": est,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        with path.open("x") as f:
            f.write(json.dumps(plan) + "\n")
            f.write(json.dumps({"type": "usage", "account_name": acct, "free_left_estimate": room[acct]}) + "\n")
            f.flush()
            collect(plan, f, creds[acct][0], creds[acct][1])
            f.write(json.dumps({"type": "complete", "timestamp":
                                datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")
        spent[acct] += est
        ids = {x["id"] for x in chunk}
        todo = [x for x in todo if x["id"] not in ids]
        print(f"chunk {path.name}: {len(chunk)} requests, {est:.0f} neurons; {len(todo)} left", flush=True)


def load_arm(path: Path, requests: dict[str, dict], bindings: dict | None = None) -> dict[str, list[dict]]:
    """Map request id -> list of answer objects (one per pass). Rejects duplicates, errors,
    unknown ids and responses bound to a different request body. A directory is read as the
    complete chunk journals of hosted-free (*.failed files are ignored)."""
    if bindings is None:bindings = {}
    if path.is_dir():
        merged: dict[str, list[dict]] = {}
        for chunk in sorted(path.glob("chunk-*.jsonl")):
            for rid, answers in load_arm(chunk, requests, bindings).items():
                if rid in merged:
                    raise ValueError(f"{chunk}: {rid} answered in more than one chunk")
                merged[rid] = answers
        return merged
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    out: dict[str, list[dict]] = {}
    if rows and rows[0].get("type") == "plan":
        try:
            plan, calls = load_hosted_bytes(raw)
        except ValueError as exc:
            raise ValueError(f"{path}: incomplete or invalid hosted journal: {exc}") from exc
        planned = {(r['model'],r["id"]): r for r in plan["requests"]}
        for _, r in sorted(calls.items()):
            if r["warmup"]:
                continue
            p = planned[r['model'],r["id"]]
            sent = p["request"]
            if r['id'] not in requests or sent != arm_request(requests[r['id']], r['model']):
                raise ValueError(f"{path}: {r['id']} request differs from the requests file")
            if r.get("full_input_reported") is not True:
                raise ValueError(f"{path}: {r['id']} hosted reported a different token count")
            if r['answer'].get('model') != r['model']:
                raise ValueError('Hosted response model differs from request')
            out.setdefault(r["id"], []).append(r["answer"]["answers"])
            bindings[r['id']] = (r['model'], r['request_sha256'])
    else:
        for r in rows:
            if r["id"] not in requests:
                raise ValueError(f"{path}: unknown id {r['id']}")
            model = r["response"].get("model")
            # Older local captures used literal Unicode. They remain valid standalone
            # evidence, but score() refuses to pair them with differently escaped wire bytes.
            hashes = {hashlib.sha256(json.dumps(arm_request(requests[r['id']],model),
                      ensure_ascii=ascii_only).encode()).hexdigest() for ascii_only in (True,False)}
            if r["request_sha256"] not in hashes:
                raise ValueError(f"{path}: {r['id']} is not bound to the stored request")
            if r["id"] in out:
                raise ValueError(f"{path}: duplicate id {r['id']}")
            out[r["id"]] = [r["response"]["answers"]]
            bindings[r['id']] = (model,r['request_sha256'])
    for rid, passes in out.items():
        questions = requests[rid]['questions']
        for answers in passes:
            if set(answers) != set(questions):raise ValueError('Missing or unexpected answer questions')
            for qid,question in questions.items():
                probs = distribution(answers[qid], list(question['criteria']), question['type'])
                decision(answers[qid],probs);confidence_value(answers[qid])
    return out


def macro_f1(gold: list[str], pred: list[str], labels: list[str]) -> float:
    """Decision Index metrics.macro_f1: the mean over labels seen in gold or predictions
    (`labels` is accepted for call symmetry and ignored)."""
    labels = sorted(set(gold) | set(pred))
    g_counts,p_counts = Counter(gold),Counter(pred)
    correct = Counter(g for g,p in zip(gold,pred) if g==p)
    total = 0.0
    for l in labels:
        tp = correct[l]
        total += 0.0 if tp == 0 else 2 * tp / (g_counts[l] + p_counts[l])
    return total / len(labels)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def score(labels_path: Path, out: Path, arms: list[str], boot: int, seed: int,
          only_answered_by: str | None = None) -> None:
    if out.exists():raise FileExistsError(out)
    if boot<40:raise ValueError('at least 40 bootstrap replicates required')
    labels_rows = read_jsonl(labels_path)
    gold = {r["id"]: r["label"] for r in labels_rows}
    req_path = labels_path.with_name("requests.jsonl")
    request_rows = read_jsonl(req_path)
    requests = {r["id"]: r for r in request_rows}
    if len(gold)!=len(labels_rows) or len(requests)!=len(request_rows):
        raise ValueError('duplicate request or label ID')
    if set(requests) != set(gold):
        raise ValueError("requests and labels name different ids")
    label_set = sorted(set(next(iter(requests.values()))["questions"]["q1"]["criteria"]))
    ids = sorted(gold)
    if only_answered_by:
        # Pair the same observed rows. An incomplete first label round does not
        # cover every class, so report observed coverage separately from the schema.
        spec = dict(a.split("=", 1) for a in arms)[only_answered_by]
        ids = sorted(load_arm(Path(spec), requests))
    preds: dict[str, list[str]] = {}
    probs: dict[str, list[dict]] = {}
    passes: dict[str, list[list[str]]] = {}
    bindings = {}
    def evidence(path):
        path=Path(path)
        return {str(p):sha256_file(p) for p in (sorted(path.glob('chunk-*.jsonl')) if path.is_dir() else [path])}
    sources = {name:evidence(path) for name,path in (spec.split('=',1) for spec in arms)}
    if len(sources)!=len(arms):raise ValueError('duplicate arm name')
    for spec in arms:
        name, path = spec.split("=", 1)
        bindings[name] = {}
        answers = load_arm(Path(path), requests, bindings[name])
        missing = [i for i in ids if i not in answers]
        if missing:
            raise ValueError(f"{name}: {len(missing)} requests have no answer, e.g. {missing[:3]}")
        n_pass = {len(v) for v in answers.values()}
        if len(n_pass) != 1:
            raise ValueError(f"{name}: unequal pass counts")
        passes[name] = [[answers[i][k]["q1"]["choice"] for i in ids] for k in range(n_pass.pop())]
        preds[name] = passes[name][0]
        probs[name] = [answers[i][0]["q1"]["probabilities"] for i in ids]
    if not ids:raise ValueError('no paired labeled requests')
    for rid in ids:
        if len({binding[rid] for binding in bindings.values()})!=1:
            raise ValueError(f'{rid}: paired payload bytes differ between arms')
    g = [gold[i] for i in ids]
    rng = random.Random(seed)
    samples = [[rng.randrange(len(ids)) for _ in ids] for _ in range(boot)]
    report = {"examples": len(ids), "of_total": len(gold), "restricted_to_ids_answered_by": only_answered_by,
              "labels": len(label_set), "bootstrap": boot, "seed": seed,
              "observed_gold_labels": len(set(g)), "gold_label_counts": dict(sorted(Counter(g).items())),
              "bootstrap_unit": "paired request rows, pooled resampling without label stratification",
              "requests_sha256": sha256_file(req_path), "labels_sha256": sha256_file(labels_path),
              "arms": {}, "pairs": {}, "source_sha256":sources,
              "identical_paired_payload_bytes":True, "probability_p99_method":"linear"}
    for name, p in preds.items():
        acc = sum(a == b for a, b in zip(g, p)) / len(ids)
        f1 = macro_f1(g, p, label_set)
        f1s = sorted(macro_f1([g[j] for j in s], [p[j] for j in s], label_set) for s in samples)
        entry = {"accuracy": acc, "accuracy_ci95": wilson(sum(a == b for a, b in zip(g, p)), len(ids)),
                 "macro_f1": f1, "macro_f1_ci95": (f1s[int(0.025 * boot)], f1s[int(0.975 * boot) - 1])}
        if len(passes[name]) > 1:
            same = sum(a == b for a, b in zip(passes[name][0], passes[name][1]))
            entry["pass_agreement"] = {"agree": same, "n": len(ids), "ci95": wilson(same, len(ids))}
        report["arms"][name] = entry
    names = list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = preds[a], preds[b]
            agree = sum(x == y for x, y in zip(pa, pb))
            only_a = sum(x == t and y != t for x, y, t in zip(pa, pb, g))
            only_b = sum(y == t and x != t for x, y, t in zip(pa, pb, g))
            dp = [max(abs(qa.get(k, 0.0) - qb.get(k, 0.0)) for k in set(qa) | set(qb))
                  for qa, qb in zip(probs[a], probs[b])]
            diffs = sorted(macro_f1(g_s := [g[j] for j in s], [pa[j] for j in s], label_set) -
                           macro_f1(g_s, [pb[j] for j in s], label_set) for s in samples)
            dps = sorted(dp)
            report["pairs"][f"{a} vs {b}"] = {
                "decision_agreement": agree, "n": len(ids), "agreement_ci95": wilson(agree, len(ids)),
                "correct_only_first": only_a, "correct_only_second": only_b,
                "mcnemar_exact_p": mcnemar_exact(only_a, only_b),
                "macro_f1_diff": macro_f1(g, pa, label_set) - macro_f1(g, pb, label_set),
                "macro_f1_diff_ci95": (diffs[int(0.025 * boot)], diffs[int(0.975 * boot) - 1]),
                "max_abs_dp": {"mean": sum(dp) / len(dp), "p99": float(np.quantile(dp,.99,method='linear')),
                               "max": dps[-1]},
                "disagreements": [i for i, x, y in zip(ids, pa, pb) if x != y],
            }
    if sources!={name:evidence(path) for name,path in (spec.split('=',1) for spec in arms)}:
        raise ValueError('evaluation evidence changed during scoring')
    out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    for name, e in report["arms"].items():
        print(f"{name:12s} acc {e['accuracy']:.4f}  macro-F1 {e['macro_f1']:.4f} "
              f"[{e['macro_f1_ci95'][0]:.4f}, {e['macro_f1_ci95'][1]:.4f}]")
    for pair, e in report["pairs"].items():
        print(f"{pair:24s} agree {e['decision_agreement']}/{e['n']}  McNemar p {e['mcnemar_exact_p']:.3g}  "
              f"dF1 {e['macro_f1_diff']:+.4f} [{e['macro_f1_diff_ci95'][0]:+.4f}, {e['macro_f1_diff_ci95'][1]:+.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("fetch"); s.add_argument("dir", type=Path)
    s = sub.add_parser("build"); s.add_argument("dataset", choices=sorted(SOURCES))
    s.add_argument("src", type=Path); s.add_argument("out", type=Path)
    s.add_argument("--per-label", type=int); s.add_argument("--seed", type=int, default=20261010)
    s = sub.add_parser("capture-local"); s.add_argument("requests", type=Path); s.add_argument("out", type=Path)
    s.add_argument("--binary", type=Path, default=ROOT / "clef"); s.add_argument("--gguf", type=Path, required=True)
    s.add_argument("--model", required=True, choices=["clef-flash", "clef", "clef-omni"])
    s = sub.add_parser("capture-pytorch"); s.add_argument("requests", type=Path); s.add_argument("out", type=Path)
    s.add_argument("--model-dir", type=Path, required=True); s.add_argument("--device", default="mps")
    s.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    s.add_argument("--model", required=True, choices=["clef-flash", "clef", "clef-omni"])
    s = sub.add_parser("plan-hosted"); s.add_argument("requests", type=Path); s.add_argument("out", type=Path)
    s.add_argument("--model", required=True, choices=["clef-flash", "clef", "clef-omni"])
    s.add_argument("--passes", type=int, choices=[1, 2], default=2)
    s.add_argument("--run", action="store_true"); s.add_argument("--account")
    s.add_argument("--clef-tool", type=Path, default=ROOT / "clef-tool"); s.add_argument("--gguf", type=Path)
    s.add_argument("--paid-max-neurons", type=float, help="billed run ceiling; only with the user's approval")
    s = sub.add_parser("hosted-free"); s.add_argument("requests", type=Path); s.add_argument("outdir", type=Path)
    s.add_argument("--model", required=True, choices=["clef-flash", "clef", "clef-omni"])
    s.add_argument("--accounts", nargs="+", required=True, help="exact Wrangler account names")
    s.add_argument("--clef-tool", type=Path, default=ROOT / "clef-tool"); s.add_argument("--gguf", type=Path)
    s.add_argument("--run", action="store_true")
    s = sub.add_parser("score"); s.add_argument("labels", type=Path); s.add_argument("out", type=Path)
    s.add_argument("--arm", action="append", required=True, help="NAME=FILE")
    s.add_argument("--bootstrap", type=int, default=2000); s.add_argument("--seed", type=int, default=20261010)
    s.add_argument("--only-answered-by", help="restrict to the ids this arm answered (partial hosted-free run)")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.dir)
    elif a.cmd == "build":
        build(a.dataset, a.src, a.out, a.per_label, a.seed)
    elif a.cmd == "capture-local":
        capture_local(a.requests, a.out, a.binary, a.gguf, a.model)
    elif a.cmd == "capture-pytorch":
        capture_pytorch(a.requests, a.out, a.model_dir, a.model, a.device, a.dtype)
    elif a.cmd == "plan-hosted":
        if a.run and not a.account:
            ap.error("--run requires --account")
        plan_hosted(a.requests, a.out, a.model, a.passes, a.run, a.account, a.clef_tool, a.gguf, a.paid_max_neurons)
    elif a.cmd == "hosted-free":
        hosted_free(a.requests, a.outdir, a.model, a.accounts, a.clef_tool, a.gguf, a.run)
    else:
        score(a.labels, a.out, a.arm, a.bootstrap, a.seed, a.only_answered_by)


if __name__ == "__main__":
    main()
