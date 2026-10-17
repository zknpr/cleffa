"""Model-free checks for bench/labeled_eval.py: Decision Index request construction, arm
binding (local and hosted journal formats), and the paired statistics.

  .venv/bin/python -B tests/test_labeled_eval.py
"""

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import labeled_eval as le


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def request(rid, text):
    q = {"type": "choice", "instructions": "Classify:\n" + text,
         "criteria": {"option_0": "a", "option_1": "b", "option_2": "c"}}
    return {"id": rid, "state": {}, "questions": {"q1": q}}


def answer(choice):
    probs = {k: (0.8 if k == choice else 0.1) for k in ("option_0", "option_1", "option_2")}
    return {"q1": {"type": "choice", "choice": choice, "probabilities": probs, "confidence": 0.7}}


def local_rows(requests, choices, model="clef-flash"):
    out = []
    for r, c in zip(requests, choices):
        line = json.dumps(le.arm_request(r, model), ensure_ascii=False)
        out.append({"id": r["id"], "request_sha256": hashlib.sha256(line.encode()).hexdigest(),
                    "response": {"model": model, "answers": answer(c)}})
    return out


def hosted_journal(requests, passes, model="clef-flash", complete=True):
    planned = []
    for r in requests:
        p = le.arm_request(r, model)
        planned.append({"model": model, "id": r["id"], "request": p,
                        "request_sha256": hashlib.sha256(json.dumps(p).encode()).hexdigest(),
                        "full_input_tokens": 10})
    rows = [{"type": "plan", "requests": planned, "passes": len(passes),
             "planned_calls": 1 + len(planned)*len(passes)}]
    def call(p, c, index):
        body = {"model":model,"answers":answer(c),"usage":{"input_tokens":10}}
        return {"type":"call","model":model,"id":p["id"],"warmup":index==-1,"pass":index,
                "request_sha256":p["request_sha256"],"answer":body,"response":{"result":body},
                "status":200,"reported_input_tokens":10,"full_input_reported":True}
    rows.append(call(planned[0], "option_2", -1))
    for k, choices in enumerate(passes):
        for p, c in zip(planned, choices):
            rows.append(call(p, c, k))
    if complete:
        rows.append({"type": "complete"})
    return rows


class Build(unittest.TestCase):
    def test_decision_index_construction(self):
        src = Path(__file__).resolve().parents[1] / "golden" / "labeled-evals-20261010" / "sources"
        if not (src / "banking77" / "test.csv").exists():
            self.skipTest("sources not fetched (bench/labeled_eval.py fetch)")
        with tempfile.TemporaryDirectory() as d:
            le.build("banking77", src, Path(d) / "b", None, 1)
            le.build("clinc150", src, Path(d) / "c", None, 1)
            b = le.read_jsonl(Path(d) / "b" / "requests.jsonl")
            c = le.read_jsonl(Path(d) / "c" / "requests.jsonl")
            cl = le.read_jsonl(Path(d) / "c" / "labels.jsonl")
        # normalize_direct.banking77 / clinc150 and layout.choice_row at 87d4650.
        self.assertEqual(len(b), 3080)
        self.assertEqual(b[0]["id"], "BANKING77:test:0")
        self.assertEqual(b[0]["state"], {})
        q = b[0]["questions"]["q1"]
        self.assertEqual(q["instructions"], "Classify the banking intent of this user request:\nHow do I locate my card?")
        self.assertEqual(list(q["criteria"])[:2], ["option_0", "option_1"])
        self.assertEqual(q["criteria"]["option_0"], "card_arrival")   # raw label, underscores kept
        self.assertEqual(len(c), 5500)
        self.assertEqual(c[4500]["id"], "CLINC150+OOS:oos_test:0")    # ids restart per split
        crit = c[0]["questions"]["q1"]["criteria"]
        self.assertEqual(len(crit), 151)
        names = list(crit.values())
        self.assertIn("out of scope: none of the listed intents", names)
        self.assertNotIn("oos", names)
        self.assertTrue(all("_" not in n for n in names))
        oos_key = next(k for k, v in crit.items() if v.startswith("out of scope"))
        self.assertTrue(all(r["label"] == oos_key for r in cl[4500:]))


class Arms(unittest.TestCase):
    def test_missing_duplicate_or_unbound_hosted_calls_fail(self):
        import copy
        original=hosted_journal(self.reqs,[["option_0"]*6]*2)
        for change in ('missing-pass','duplicate','forged-hash'):
            rows=copy.deepcopy(original)
            if change=='missing-pass':rows=[r for r in rows if r.get('pass')!=1]
            elif change=='duplicate':rows.insert(-1,copy.deepcopy(rows[2]))
            else:
                for row in rows[0]['requests']:row['request_sha256']='forged'
                for row in rows[1:]:
                    if row.get('type')=='call':row['request_sha256']='forged'
            with self.subTest(change=change):
                write_jsonl(self.d/'bad.jsonl',rows)
                with self.assertRaises(ValueError):le.load_arm(self.d/'bad.jsonl',self.by_id)

    def test_same_unicode_value_with_different_wire_bytes_cannot_be_paired(self):
        rows=[request('u','caf\u00e9')]
        write_jsonl(self.d/'requests.jsonl',rows)
        write_jsonl(self.d/'labels.jsonl',[dict(id='u',label='option_0')])
        native=local_rows(rows,['option_0'])
        write_jsonl(self.d/'c.jsonl',native)
        write_jsonl(self.d/'h.jsonl',hosted_journal(rows,[['option_0']]))
        with self.assertRaisesRegex(ValueError,'payload bytes differ'):
            le.score(self.d/'labels.jsonl',self.d/'score.json',
                     [f'C={self.d/"c.jsonl"}',f'H={self.d/"h.jsonl"}'],50,1)
        native[0]['request_sha256']=hashlib.sha256(json.dumps(le.arm_request(rows[0],'clef-flash')).encode()).hexdigest()
        write_jsonl(self.d/'c.jsonl',native)
        le.score(self.d/'labels.jsonl',self.d/'score.json',
                 [f'C={self.d/"c.jsonl"}',f'H={self.d/"h.jsonl"}'],50,1)

    def test_nonfinite_or_incomplete_probabilities_fail_before_scoring(self):
        rows=local_rows(self.reqs,['option_0']*6)
        rows[0]['response']['answers']['q1']['probabilities']['option_0']=float('nan')
        write_jsonl(self.d/'bad.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'probability'):le.load_arm(self.d/'bad.jsonl',self.by_id)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.reqs = [request(f"T:test:{i}", f"text {i}") for i in range(6)]
        self.by_id = {r["id"]: r for r in self.reqs}

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_binding_rejects_other_request(self):
        rows = local_rows(self.reqs, ["option_0"] * 6)
        rows[2]["request_sha256"] = rows[3]["request_sha256"]
        write_jsonl(self.d / "a.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "not bound"):
            le.load_arm(self.d / "a.jsonl", self.by_id)

    def test_hosted_passes_and_warmup(self):
        write_jsonl(self.d / "h.jsonl", hosted_journal(self.reqs, [["option_0"] * 6, ["option_1"] * 6]))
        arm = le.load_arm(self.d / "h.jsonl", self.by_id)
        self.assertEqual(len(arm), 6)
        self.assertTrue(all(len(v) == 2 for v in arm.values()))   # warm-up excluded
        self.assertEqual(arm["T:test:0"][1]["q1"]["choice"], "option_1")

    def test_hosted_incomplete_or_mismatched(self):
        write_jsonl(self.d / "h.jsonl", hosted_journal(self.reqs, [["option_0"] * 6], complete=False))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            le.load_arm(self.d / "h.jsonl", self.by_id)
        rows = hosted_journal(self.reqs, [["option_0"] * 6])
        rows[2]["full_input_reported"] = False
        rows[2]["reported_input_tokens"] = 9
        rows[2]["answer"]["usage"]["input_tokens"] = 9
        write_jsonl(self.d / "h2.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "token count"):
            le.load_arm(self.d / "h2.jsonl", self.by_id)
        other = dict(self.by_id)
        other["T:test:1"] = request("T:test:1", "different text")
        write_jsonl(self.d / "h3.jsonl", hosted_journal(self.reqs, [["option_0"] * 6]))
        with self.assertRaisesRegex(ValueError, "differs from the requests file"):
            le.load_arm(self.d / "h3.jsonl", other)


class Chunks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.reqs = [request(f"T:test:{i}", f"text {i}") for i in range(6)]
        self.by_id = {r["id"]: r for r in self.reqs}

    def tearDown(self):
        self.tmp.cleanup()

    def test_directory_merge(self):
        out = self.d / "hosted"
        out.mkdir()
        write_jsonl(out / "chunk-0000-a.jsonl", hosted_journal(self.reqs[:3], [["option_0"] * 3]))
        write_jsonl(out / "chunk-0001-b.jsonl", hosted_journal(self.reqs[3:], [["option_1"] * 3]))
        write_jsonl(out / "chunk-0002-c.failed", hosted_journal(self.reqs[:1], [["option_2"]], complete=False))
        arm = le.load_arm(out, self.by_id)
        self.assertEqual(sorted(arm), sorted(self.by_id))
        self.assertEqual(arm["T:test:4"][0]["q1"]["choice"], "option_1")
        write_jsonl(out / "chunk-0003-d.jsonl", hosted_journal(self.reqs[:1], [["option_2"]]))
        with self.assertRaisesRegex(ValueError, "more than one chunk"):
            le.load_arm(out, self.by_id)

    def test_round_robin_and_partial_scoring(self):
        gold = {r["id"]: l for r, l in zip(self.reqs, ["option_0", "option_0", "option_1", "option_1", "option_2", "option_2"])}
        order = [r["id"] for r in le.round_robin(self.reqs, gold)]
        self.assertEqual(order, ["T:test:0", "T:test:2", "T:test:4", "T:test:1", "T:test:3", "T:test:5"])
        write_jsonl(self.d / "requests.jsonl", self.reqs)
        write_jsonl(self.d / "labels.jsonl", [{"id": k, "label": v} for k, v in gold.items()])
        write_jsonl(self.d / "c.jsonl", local_rows(self.reqs, [gold[r["id"]] for r in self.reqs]))
        out = self.d / "hosted"
        out.mkdir()
        first = [r for r in self.reqs if r["id"] in order[:3]]     # one finished round
        write_jsonl(out / "chunk-0000-a.jsonl", hosted_journal(first, [[gold[r["id"]] for r in first]]))
        le.score(self.d / "labels.jsonl", self.d / "out.json", [f"C={self.d / 'c.jsonl'}", f"H={out}"], 50, 1,
                 only_answered_by="H")
        rep = json.loads((self.d / "out.json").read_text())
        self.assertEqual((rep["examples"], rep["of_total"]), (3, 6))
        self.assertEqual(rep["observed_gold_labels"], 3)
        self.assertEqual(rep["gold_label_counts"], {f'option_{i}': 1 for i in range(3)})
        self.assertEqual(rep["pairs"]["C vs H"]["decision_agreement"], 3)
        with self.assertRaisesRegex(ValueError, "have no answer"):
            le.score(self.d / "labels.jsonl", self.d / "out2.json", [f"C={self.d / 'c.jsonl'}", f"H={out}"], 50, 1)

class Stats(unittest.TestCase):
    def test_macro_f1_matches_decision_index(self):
        # metrics.macro_f1 averages over labels seen in gold or predictions.
        gold = ["a", "a", "b", "b"]
        pred = ["a", "c", "b", "b"]
        f1 = {"a": 2 / 3, "b": 1.0, "c": 0.0}
        self.assertAlmostEqual(le.macro_f1(gold, pred, []), sum(f1.values()) / 3)

    def test_mcnemar_and_wilson(self):
        self.assertEqual(le.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(le.mcnemar_exact(0, 5), 2 / 32)
        self.assertAlmostEqual(le.mcnemar_exact(3, 3), 1.0)
        lo, hi = le.wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertTrue(all(math.isnan(x) for x in le.wilson(0, 0)))

    def test_score_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            reqs = [request(f"T:test:{i}", f"text {i}") for i in range(4)]
            write_jsonl(d / "requests.jsonl", reqs)
            write_jsonl(d / "labels.jsonl", [{"id": r["id"], "label": l} for r, l in
                                              zip(reqs, ["option_0", "option_0", "option_1", "option_2"])])
            write_jsonl(d / "c.jsonl", local_rows(reqs, ["option_0", "option_0", "option_1", "option_2"]))
            write_jsonl(d / "h.jsonl", hosted_journal(reqs, [["option_0", "option_1", "option_1", "option_2"]] * 2))
            le.score(d / "labels.jsonl", d / "out.json", [f"C={d / 'c.jsonl'}", f"H={d / 'h.jsonl'}"], 200, 1)
            rep = json.loads((d / "out.json").read_text())
        self.assertEqual(rep["arms"]["C"]["accuracy"], 1.0)
        self.assertEqual(rep["arms"]["H"]["accuracy"], 0.75)
        self.assertEqual(rep["arms"]["H"]["pass_agreement"]["agree"], 4)
        pair = rep["pairs"]["C vs H"]
        self.assertEqual((pair["decision_agreement"], pair["correct_only_first"], pair["correct_only_second"]), (3, 1, 0))
        self.assertEqual(pair["disagreements"], ["T:test:1"])
        self.assertAlmostEqual(pair["max_abs_dp"]["max"], 0.7)


if __name__ == "__main__":
    unittest.main()
