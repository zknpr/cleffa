"""Golden-data serialization shared by ref/oracle.py, ref/oracle_f32_stream.py and
ref/write_encoded.py, so every golden directory's encoded.jsonl has one format: the request id,
the reference's token ids, and per question its type, span, option spans and option token ids.
tests/test_parity.py reads it for the token-id check and the expected question/option shape."""

from __future__ import annotations

import json


def encoded_line(request: dict, encoded) -> str:
    return json.dumps({
        "id": request["id"],
        "input_ids": list(encoded.input_ids),
        "questions": [
            {
                "id": q.question_id,
                "type": q.question_type,
                "span": list(q.question_span),
                "option_spans": [list(s) for s in q.option_spans],
                "option_ids": list(q.option_ids),
            }
            for q in encoded.questions
        ],
    }, ensure_ascii=False) + "\n"
