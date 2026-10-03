"""Load Clef from ./model on Apple Silicon (MPS) and run the README's SystemOne example."""

import sys
import time
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).parent / "model"
# joint_schema_model.py is imported from the pinned, reviewed snapshot (rev 2f3de3dd).
sys.path.insert(0, str(MODEL_DIR))
from joint_schema_model import load_release_model, systemone  # noqa: E402

device = "mps" if torch.backends.mps.is_available() else "cpu"
start = time.perf_counter()
model, processor = load_release_model(MODEL_DIR, device=device)
print(f"loaded on {device} in {time.perf_counter() - start:.1f}s")

request = {
    "model": "clef",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle the message?",
            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
}
start = time.perf_counter()
response = systemone(model, processor, request)
print(f"inference {time.perf_counter() - start:.2f}s")
print(response)
