"""CPU regressions for cache-eval input ordering and exact-logit acceptance."""
import json
import struct
import unittest

from test_prefix_cache import cold_reference, exact, rows


class CacheEvalChecks(unittest.TestCase):
    def test_reference_dedup_preserves_question_and_option_order(self):
        requests = [
            {'state': 'state', 'questions': {'z': {'enum': ['y', 'a']}, 'a': {'enum': ['b', 'a']}}},
            {'state': 'state', 'questions': {'a': {'enum': ['b', 'a']}, 'z': {'enum': ['y', 'a']}}},
        ]
        requests += [requests[0], requests[1]]
        calls = []

        def run(unique, settings=None):
            calls.append((unique, settings))
            return ''.join(json.dumps(r)+'\n' for r in unique), []

        result = cold_reference(requests, run, {'mode': 'test'})
        self.assertEqual(result, ''.join(json.dumps(r)+'\n' for r in requests))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0][0]), 2)
        self.assertEqual(calls[0][1], {'mode': 'test'})

    def test_incomplete_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete cold reference'):
            cold_reference([{'state': 'x'}], lambda *args, **kwargs: ('', []))

    def test_json_spacing_is_not_a_logit_difference(self):
        self.assertEqual(exact('{"q":[1.0,-2.5]}\n', '{"q": [1.0, -2.5]}\n'), 2)

    def test_one_fp32_ulp_is_rejected(self):
        next_float = struct.unpack('f', struct.pack('I', 0x3f800001))[0]
        with self.assertRaisesRegex(ValueError, 'logits differ'):
            exact(json.dumps({'q': [next_float]}), '{"q":[1.0]}')

    def test_signed_zero_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'logits differ'):
            exact('{"q":[-0.0]}', '{"q":[0.0]}')

    def test_nonfinite_and_missing_coverage_are_rejected(self):
        for actual in ('{"q":[NaN]}', '{"q":[Infinity]}', '{"q":[]}', '{}', '', '{"other":[1]}'):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                exact(actual, '{"q":[1.0]}')

    def test_option_order_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'logits differ'):
            exact('{"q":[2.0,1.0]}', '{"q":[1.0,2.0]}')

    def test_only_expected_errors_may_be_skipped(self):
        self.assertEqual(exact('{"error":"prefix cache entry cannot grow"}', '{"q":[1]}', errors=True), 0)
        for data in ('{"error":"unrelated"}', '{"error":"prefix cache entry cannot grow","q":[1]}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                exact(data, '{"q":[1]}', errors=True)

    def test_unicode_line_separators_stay_inside_json(self):
        data = json.dumps({'q\u0085\u2028\u2029': [1.0]}, ensure_ascii=False)+'\n'
        self.assertEqual(len(rows(data)), 1)
        self.assertEqual(exact(data, data), 1)


if __name__ == '__main__':
    unittest.main()
