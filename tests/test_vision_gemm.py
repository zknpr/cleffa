"""Compensated vision GEMM overflow must rerun only through the safe FP32 tower.

Usage: .venv/bin/python -B tests/test_vision_gemm.py MODEL.gguf VISION_REQUESTS.jsonl
The kernel benchmark separately checks real out-of-range/NaN inputs and flag isolation.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    model, requests = sys.argv[1:]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLEF_")}

    def run(mask, limit=None, batch=1, dump=None):
        mode = {**env, "CLEF_ACT_F16": str(mask)}
        if limit is not None:
            mode["CLEF_DEBUG_F16_LIMIT"] = str(limit)
        command = [str(ROOT / "clef"), "-m", model, "--logits", "--batch", str(batch)]
        if dump is not None:
            command += ["--dump", str(dump)]
            text = Path(requests).read_text().splitlines()[0] + "\n"
            command_input = {"input": text}
        else:
            command += [requests]
            command_input = {}
        return subprocess.run(command, cwd=ROOT, env=mode, capture_output=True, text=True,
                              check=True, timeout=300, **command_input).stdout

    reference = run(0)
    compensated = run(16)
    if reference == compensated:
        raise AssertionError("ACT_VIS alone has no effect; this fixture cannot test compensation")
    # All backbone producers use BF16 here: only the new vision split can flag the record.
    for batch in (1, 8):
        assert run(16, 1e-6, batch) == reference, f"vision overflow failed to rerun at batch {batch}"
    with tempfile.TemporaryDirectory(prefix="vision-gemm-", dir=ROOT / "golden") as tmp:
        a, b = Path(tmp) / "reference.bin", Path(tmp) / "overflow.bin"
        assert run(0, dump=a) == run(16, 1e-6, dump=b), "overflow dump logits differ"
        assert a.read_bytes() == b.read_bytes(), "dump retained discarded compensated activations"
    print("vision-only overflow: batch 1/8 and dumped residuals match the safe rerun exactly")


if __name__ == "__main__":
    main()
