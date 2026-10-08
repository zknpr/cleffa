"""Alternate vision implementations on the same warm CLI requests.

The CLI's --time covers inference, excluding image decode,
request encoding, model loading and response serialization. Run one GPU job at a time.
  .venv/bin/python -B bench/vision_latency.py golden/vision-latency.json
  Add --baseline-binary PATH to compare a saved engine with the current build.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", nargs="+", default=["gguf/clef-flash.gguf", "gguf/clef.gguf"])
    parser.add_argument("--requests", type=Path, default=ROOT / "golden/clef-flash-vision-f32/requests.jsonl")
    parser.add_argument("--ids", nargs="+", default=["v001", "v009"])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--baseline-binary", type=Path,
                        help="compare this saved engine against ./clef instead of toggling vision attention")
    parser.add_argument("--candidate-binary", type=Path, default=ROOT / "clef",
                        help="candidate engine; useful for isolated experimental builds")
    parser.add_argument("--allow-other-engines", action="store_true",
                        help="record provisional timings even if other clef processes are running")
    args = parser.parse_args()
    if min(args.samples, args.warmup, args.rounds) < 1:
        parser.error("samples, warmup and rounds must be positive")
    candidate = args.candidate_binary.resolve()
    binaries = {"baseline": args.baseline_binary.resolve(), "candidate": candidate} if args.baseline_binary else {
        "mma": candidate, "mpp": candidate}
    arms = list(binaries)
    processes = subprocess.run(["ps", "-axo", "pid=,state=,comm="], capture_output=True, text=True, check=True)
    active = []
    for line in processes.stdout.splitlines():
        pid, state, command = line.strip().split(None, 2)
        if "T" not in state and Path(command).name in {"clef", "clef-server", *(p.name for p in binaries.values())}:
            active.append({"pid": int(pid), "state": state, "command": command})
    if active and not args.allow_other_engines:
        parser.error(f"other engines may contend for the GPU: {active}; stop/pause them or use --allow-other-engines")
    rows = {json.loads(line)["id"]: line for line in args.requests.read_text().splitlines(keepends=True)}
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}
    report = {
        "metric": "warm CLI inference ms; excludes decode, encoding, loading and response serialization",
        "samples_per_arm_round": args.samples, "warmups_per_arm_round": args.warmup,
        "rounds": args.rounds, "other_active_engines": active, "provisional": bool(active),
        "binary_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "binaries": {arm: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for arm, path in binaries.items()},
        "source_sha256": hashlib.sha256((ROOT / "metal/clef.metal").read_bytes()).hexdigest(),
        "runs": [], "summary": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for rid in args.ids:
            line = rows[rid]
            if not line.endswith("\n"):
                line += "\n"
            seen = {}
            counts = set()
            for round_no in range(args.rounds):
                for arm in (arms if round_no % 2 == 0 else list(reversed(arms))):
                    run_env = env if args.baseline_binary else {**env, "CLEF_VIS_MPP": "0" if arm == "mma" else "1"}
                    run = subprocess.run(
                        [str(binaries[arm]), "-m", model, "--time", "--batch", "1"],
                        input=line * (args.samples + args.warmup), text=True, capture_output=True,
                        cwd=ROOT, env=run_env,
                        check=True, timeout=300,
                    )
                    timings = re.findall(r"batch of 1 \((\d+) tokens\) in ([\d.]+) ms", run.stderr)
                    responses = run.stdout.splitlines()
                    if len(timings) != args.samples + args.warmup or len(responses) != len(timings):
                        raise RuntimeError(f"missing timings/responses for {model} {rid} {arm}: {run.stderr}")
                    for response in responses:
                        if "error" in json.loads(response):
                            raise RuntimeError(f"engine error: {response}")
                    if len(set(responses)) != 1 or (arm in seen and responses[0] != seen[arm]):
                        raise RuntimeError(f"non-repeatable response for {model} {rid} {arm}")
                    seen[arm] = responses[0]
                    counts.update(int(n) for n, _ in timings)
                    if len(counts) != 1:
                        raise RuntimeError("arms processed different token counts")
                    entry = {"model": model, "id": rid, "arm": arm, "round": round_no,
                             "tokens": int(timings[0][0]),
                             "request_sha256": hashlib.sha256(line.encode()).hexdigest(),
                             "warmup_ms": [float(t) for _, t in timings[:args.warmup]],
                             "ms": [float(t) for _, t in timings[args.warmup:]],
                             "response": json.loads(responses[0]), "stderr": run.stderr}
                    report["runs"].append(entry)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"{model} {rid} {arm} round {round_no}: {statistics.median(entry['ms']):.1f} ms", flush=True)
            medians = {arm: statistics.median(t for r in report["runs"]
                       if r["model"] == model and r["id"] == rid and r["arm"] == arm for t in r["ms"])
                       for arm in arms}
            report["summary"].append({"model": model, "id": rid, "median_ms": medians,
                                      "reduction_percent": 100 * (1 - medians[arms[1]] / medians[arms[0]]),
                                      "response_bytes_equal_between_arms": seen[arms[0]] == seen[arms[1]]})
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
