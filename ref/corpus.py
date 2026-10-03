"""Fixed SystemOne request set used as the parity and latency corpus.

Deterministic, hand-built, and chosen to cover what the native engine must reproduce exactly:
all three question types, 1..N questions, 2..12 options, string vs JSON state, missing
instructions, custom noul criteria, tokenizer edge cases (CJK, emoji, combining marks, code,
whitespace runs, numbers), and state lengths from a few tokens to several thousand.
"""

from __future__ import annotations

import json
import random


def _support(state: str) -> dict:
    return {
        "model": "clef",
        "state": state,
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


def _long_log(lines: int, seed: int) -> str:
    rng = random.Random(seed)
    services = ["api", "auth", "billing", "search", "cdn-edge", "queue", "db-primary"]
    levels = ["INFO", "INFO", "INFO", "WARN", "ERROR"]
    out = []
    for i in range(lines):
        out.append(
            f"2026-09-30T12:{i // 60:02d}:{i % 60:02d}Z {rng.choice(levels)} "
            f"{rng.choice(services)} req_id={rng.getrandbits(48):012x} "
            f"latency_ms={rng.randint(1, 4000)} status={rng.choice([200, 200, 201, 404, 500, 503])}"
        )
    return "\n".join(out)


def build() -> list[dict]:
    requests: list[dict] = []
    add = requests.append

    # README examples.
    add(_support("Our checkout started returning errors and orders are blocked."))
    add({
        "model": "clef",
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
            },
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
    })

    # Short support variants.
    for text in [
        "Hi, I was charged twice for my subscription this month.",
        "Could you update the billing address on my account? No rush.",
        "The dashboard is slow to load but it works eventually.",
        "EVERYTHING IS DOWN!!! 503 on every endpoint since 09:14 UTC",
        "Thanks for the quick fix yesterday, all good now.",
    ]:
        add(_support(text))

    # Minimal question: no instructions (question id used), single noul.
    add({"model": "clef", "state": "The door is open.", "questions": {"is_door_open": {"type": "noul"}}})

    # Custom noul criteria override.
    add({
        "model": "clef",
        "state": "Patient reports chest pain radiating to the left arm, onset 20 minutes ago.",
        "questions": {
            "escalate": {
                "type": "noul",
                "instructions": "Should this be escalated to emergency services?",
                "criteria": {"true": "Escalate immediately.", "false": "Routine triage is fine."},
            }
        },
    })

    # Many options (intent classification, banking77-like).
    intents = {
        f"intent_{i:02d}": desc
        for i, desc in enumerate([
            "Card arrival", "Card lost or stolen", "Exchange rate", "Top up failed", "Transfer pending",
            "Refund not showing", "Wrong amount of cash received", "PIN blocked", "Change personal details",
            "Direct debit not recognised", "Virtual card request", "Account closure",
        ])
    }
    for text in [
        "I ordered my card two weeks ago and it still hasn't come.",
        "My PIN got blocked after 3 tries, how do I reset it?",
        "The ATM gave me 40 instead of 60 euros.",
        "Why is my bank transfer still pending after 3 days?",
    ]:
        add({
            "model": "clef",
            "state": text,
            "questions": {"intent": {"type": "choice", "instructions": "Classify the customer intent.", "criteria": intents}},
        })

    # Score with 5 and 10 levels.
    add({
        "model": "clef",
        "state": {"review": "Battery life is fine, screen is gorgeous, but the speakers crackle at high volume."},
        "questions": {
            "sentiment": {"type": "score", "instructions": "Overall sentiment.", "criteria": [
                "Very negative", "Negative", "Mixed", "Positive", "Very positive"]},
            "mentions_audio": {"type": "noul", "instructions": "Does the review mention audio quality?"},
        },
    })
    add({
        "model": "clef",
        "state": "Rate the risk: user downloaded an unsigned binary from a pastebin link and ran it as root.",
        "questions": {"risk": {"type": "score", "criteria": [str(i) for i in range(10)]}},
    })

    # Tokenizer edge cases in state and criteria.
    add({
        "model": "clef",
        "state": "顧客は請求書の金額が間違っていると言っています。🚨 Ünïcödé façade naïve — “quotes” 'single' \t\ttabs   spaces\n\n\nnewlines",
        "questions": {
            "language": {"type": "choice", "instructions": "Primary language of the message?", "criteria": {
                "ja": "Japanese", "en": "English", "de": "German", "zh": "Chinese"}},
            "complaint": {"type": "noul", "instructions": "Is the customer complaining?"},
        },
    })
    add({
        "model": "clef",
        "state": "def transfer(a, b, amt):\n    if amt <= 0:\n        raise ValueError('amt')\n    a.balance -= amt\n    b.balance += amt  # no lock!\n",
        "questions": {
            "bug": {"type": "choice", "instructions": "Main defect in this code?", "criteria": {
                "race": "Race condition / missing synchronization", "overflow": "Integer overflow",
                "none": "No defect", "validation": "Missing input validation"}},
            "severity": {"type": "score", "criteria": ["low", "medium", "high", "critical"]},
        },
    })
    add({
        "model": "clef",
        "state": {"numbers": [3.14159, -0.0, 1e-9, 12345678901234567890, True, None], "nested": {"a": {"b": {"c": []}}}},
        "questions": {"has_null": {"type": "noul", "instructions": "Does the state contain a null value?"}},
    })

    # Many questions jointly.
    add({
        "model": "clef",
        "state": {
            "ticket": "Customer on Enterprise plan reports SSO login loop after IdP certificate rotation; 400 users blocked.",
            "plan": "enterprise", "region": "eu-west",
        },
        "questions": {
            "team": {"type": "choice", "criteria": {"identity": "SSO / auth", "billing": "Billing", "network": "Network"}},
            "priority": {"type": "score", "criteria": ["P4", "P3", "P2", "P1"]},
            "outage": {"type": "noul", "instructions": "Is this a customer-facing outage?"},
            "enterprise": {"type": "noul", "instructions": "Is the customer on an enterprise plan?"},
            "needs_eng": {"type": "noul", "instructions": "Does this need an engineer?"},
            "region": {"type": "choice", "criteria": {"us": "United States", "eu": "Europe", "apac": "Asia Pacific"}},
        },
    })

    # Long states: ~1k, ~4k, ~8k tokens.
    for lines, seed in [(40, 1), (160, 2), (330, 3)]:
        add({
            "model": "clef",
            "state": _long_log(lines, seed),
            "questions": {
                "incident": {"type": "noul", "instructions": "Do these logs indicate an ongoing incident?"},
                "service": {"type": "choice", "instructions": "Which service is most affected?", "criteria": {
                    s: s for s in ["api", "auth", "billing", "search", "cdn-edge", "queue", "db-primary"]}},
            },
        })

    for i, request in enumerate(requests):
        request["id"] = f"r{i:03d}"
    return requests


if __name__ == "__main__":
    for request in build():
        print(json.dumps(request, ensure_ascii=False))
