"""Ensure hosted benchmark errors and missing token usage cannot look like equal work."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from cloudflare_checkout import collect, response_info


class ResponseTests(unittest.TestCase):
    def test_budget_stop_prevents_dispatch_and_resume_does_not_repeat_calls(self):
        import hashlib
        payload={'model':'clef','state':'test','questions':{'q':{}}}
        row=dict(model='clef',id='probe',request=payload,full_input_tokens=10,
                 request_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest())
        with patch('cloudflare_checkout.http.client.HTTPSConnection') as connection:
            def stop(*args): raise ValueError('free allowance exhausted')
            with self.assertRaisesRegex(ValueError,'free allowance'):
                collect(dict(requests=[row],passes=2),io.StringIO(),'a'*32,'secret',before_call=stop)
            connection.return_value.request.assert_not_called()
        out=io.StringIO();seen=[]
        with patch('cloudflare_checkout.http.client.HTTPSConnection') as connection:
            response=connection.return_value.getresponse.return_value
            response.status=422;response.read.return_value=b'{"errors":["invalid request"]}'
            response.getheader.return_value=None
            collect(dict(requests=[row],passes=2,completed_calls=[['clef','probe',-1],['clef','probe',0]],
                         record_validation_failures=True),out,'a'*32,'secret',after_call=seen.append)
            connection.return_value.request.assert_called_once()
        self.assertEqual(len(seen),1)
        self.assertEqual(json.loads(out.getvalue())['pass'],1)
        self.assertIn('error',seen[0])

    def test_fresh_passes_record_timestamps_revisions_and_gap(self):
        import hashlib
        payload = {'model':'clef','state':'test','questions':{'q':{'type':'noul'}}}
        row = dict(model='clef', id='probe', request=payload, full_input_tokens=10,
                   request_sha256=hashlib.sha256(json.dumps(payload).encode()).hexdigest())
        out = io.StringIO()
        with patch('cloudflare_checkout.http.client.HTTPSConnection') as connection, \
             patch('cloudflare_checkout.time.sleep') as sleep:
            response = connection.return_value.getresponse.return_value
            response.status = 200
            response.read.return_value = json.dumps(dict(answers={'q':{'type':'noul','noul':.5}},
                                                        usage={'input_tokens':10})).encode()
            response.getheader.side_effect = lambda name: 'revision-fake-secret' if name == 'x-model-version' else None
            collect(dict(requests=[row],passes=2,pass_gap_seconds=60),out,'a'*32,'fake-secret')
            sleep.assert_called_once_with(60)
        calls = list(map(json.loads,out.getvalue().splitlines()))
        self.assertEqual([call['pass'] for call in calls],[-1,0,1])
        for call in calls:
            self.assertLessEqual(call['started_at'],call['completed_at'])
            self.assertEqual(call['model_revision_headers']['x-model-version'],'revision-[REDACTED]')

    def setUp(self):
        self.expected = {"request": {"questions": {"urgent": {}}}, "full_input_tokens": 4510}
        self.answer = {"answers": {"urgent": {"noul": 0.9}}, "usage": {"input_tokens": 4510}}

    def test_direct_and_wrapped_responses(self):
        for response in [self.answer, {"success": True, "result": self.answer, "errors": []}]:
            self.assertIs(response_info(200, response, self.expected)["full_input_reported"], True)

    def test_short_usage_is_not_equal_work(self):
        self.answer["usage"]["input_tokens"] = 2382
        self.assertIs(response_info(200, self.answer, self.expected)["full_input_reported"], False)

    def test_missing_or_invalid_usage_is_unknown(self):
        for value in [None, {}, {"input_tokens": "4510"}, {"input_tokens": True}, {"input_tokens": -1}]:
            self.answer["usage"] = value
            self.assertIsNone(response_info(200, self.answer, self.expected)["full_input_reported"])

    def test_failed_or_incomplete_answers_are_errors(self):
        for status, response in [(429, self.answer), (200, {"success": False, "result": self.answer}),
                                 (200, {"errors": ["error"], "result": self.answer}),
                                 (200, []), (200, {}), (200, {"answers": {}})]:
            with self.subTest(status=status, response=response), self.assertRaises(ValueError):
                response_info(status, response, self.expected)

    def test_failed_call_is_recorded_without_token_or_retry(self):
        import hashlib
        payload = {"model": "clef", "state": "public fixture", "questions": {"urgent": {}}}
        row = {"model": "clef", "id": "test", "request": payload, "full_input_tokens": 4510,
               "request_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
        run_plan = {"requests": [row], "passes": 3}
        out = io.StringIO()
        with patch("cloudflare_checkout.http.client.HTTPSConnection") as connection:
            conn = connection.return_value
            conn.request.side_effect = OSError("fake-secret-token failure")
            with self.assertRaisesRegex(RuntimeError, "collection failed"):
                collect(run_plan, out, "a" * 32, "fake-secret-token")
            conn.request.assert_called_once()
            conn.close.assert_called_once()
        self.assertNotIn("fake-secret-token", out.getvalue())
        record = json.loads(out.getvalue())
        self.assertIn("[REDACTED]", record["error"])
        self.assertNotIn("full_input_reported", record)

    def test_escaped_token_in_a_failed_body_is_redacted(self):
        # A JSON-escaped token survives a textual replace on the raw body and is reassembled by
        # json.loads; the failed-call path writes that decoded response to the journal.
        import hashlib
        payload = {"model": "clef", "state": "public fixture", "questions": {"urgent": {}}}
        row = {"model": "clef", "id": "test", "request": payload, "full_input_tokens": 4510,
               "request_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest()}
        run_plan = {"requests": [row], "passes": 3}
        out = io.StringIO()
        escaped = "fake\\u002dsecret\\u002dtoken"   # "fake-secret-token" with the hyphens JSON-escaped
        with patch("cloudflare_checkout.http.client.HTTPSConnection") as connection:
            conn = connection.return_value
            result = conn.getresponse.return_value
            result.status = 429
            result.read.return_value = ('{"errors": [{"message": "bad token ' + escaped + '"}]}').encode()
            result.getheader.return_value = None
            with self.assertRaisesRegex(RuntimeError, "collection failed"):
                collect(run_plan, out, "a" * 32, "fake-secret-token")
        self.assertNotIn("fake-secret-token", out.getvalue())
        self.assertIn("[REDACTED]", out.getvalue())


if __name__ == "__main__":
    unittest.main()
