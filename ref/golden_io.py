"""Golden-data serialization shared by ref/oracle.py, ref/oracle_f32_stream.py and
ref/write_encoded.py, so every golden directory's encoded.jsonl has one format: the request id,
the reference's token ids, and per question its type, span, option spans and option token ids.
tests/test_parity.py reads it for the token-id check and the expected question/option shape."""

from __future__ import annotations

import json


def encoded_line(request: dict, encoded, position_ids=None) -> str:
    """[3][T] rotary positions and the first token/grid of each still image or video frame pair."""
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
        starts = []
        images = encoded.media["image_grid_thw"].tolist() if "image_grid_thw" in encoded.media else []
        videos = encoded.media["video_grid_thw"].tolist() if "video_grid_thw" in encoded.media else []
        grids = {1: iter(images), 2: iter([[1, h, w] for t, h, w in videos for _ in range(t)])}
        types = encoded.media["mm_token_type_ids"]
        for i, kind in enumerate(types):
            if kind and (i == 0 or types[i - 1] != kind):
                g = next(grids[int(kind)])
                starts.append([encoded.media["token_offset"] + i, g[1], g[2]])
        line["images"] = starts
        line["position_ids"] = position_ids
    return json.dumps(line, ensure_ascii=False) + "\n"
