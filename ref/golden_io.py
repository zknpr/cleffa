"""Golden-data serialization shared by ref/oracle.py, ref/oracle_f32_stream.py and
ref/write_encoded.py, so every golden directory's encoded.jsonl has one format: the request id,
the reference's token ids, and per question its type, span, option spans and option token ids.
tests/test_parity.py reads it for the token-id check and the expected question/option shape."""

from __future__ import annotations

import json


def encoded_line(request: dict, encoded, position_ids=None) -> str:
    """position_ids: the [3][T] rotary positions of a record with images (Qwen3_5Model.get_rope_index);
    with them the line also lists each image's first token index and patch grid."""
    line = {
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
    }
    if position_ids is not None:
        starts, t = [], encoded.media["token_offset"]
        for g in encoded.media["image_grid_thw"].tolist():
            t += 1   # <|vision_start|>
            starts.append([t, g[1], g[2]])
            t += g[1] * g[2] // 4 + 1
        line["images"] = starts
        line["position_ids"] = position_ids
    return json.dumps(line, ensure_ascii=False) + "\n"
