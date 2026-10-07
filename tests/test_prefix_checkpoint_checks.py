"""Checkpoint test mode selection must survive CLEF_* environment sanitization."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('test_prefix_checkpoints.py')


class InferenceReached(Exception):
    pass


class ModeChecks(unittest.TestCase):
    def select(self, mode, inherited):
        captured = []

        def intercept(command, **kwargs):
            captured.append((command, kwargs))
            raise InferenceReached

        # Only fixture selection runs before the intercepted inference call.
        # No model, oracle, GPU or real request corpus is needed.
        corpus = [{'state': 'public line\n' * n,
                   'questions': {'q': {'type': 'choice', 'choices': {'a': 'A', 'b': str(n)}}}}
                  for n in (200, 220, 240)]
        with tempfile.TemporaryDirectory(prefix='clef-checkpoint-mode-') as directory:
            requests = Path(directory) / 'requests.jsonl'
            requests.write_text(''.join(json.dumps(row) + '\n' for row in corpus))
            binary = Path(directory) / 'unused-clef'
            args = [str(SCRIPT), str(Path(directory) / 'clef-flash.gguf'), str(requests),
                    '--binary', str(binary)]
            if mode is not None:
                args += ['--attention', mode]
            with patch.object(sys, 'argv', args), patch.object(subprocess, 'run', intercept), \
                    patch.dict(os.environ, {'CLEF_ATTN_TU': inherited, 'CLEF_TEST_UNRELATED': '1'}), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(InferenceReached):
                    runpy.run_path(str(SCRIPT), run_name='__main__')
            self.assertEqual(len(captured), 1)
            command, settings = captured[0]
            self.assertEqual(command[0], str(binary.resolve()))
            self.assertIn('--strict', command)
            self.assertIn('--no-truncate', command)
            return {k: v for k, v in settings['env'].items() if k.startswith('CLEF_')}

    def test_explicit_fp32_overrides_inherited_tu(self):
        self.assertEqual(self.select('fp32', '1'), {'CLEF_ATTN_TU': '0'})

    def test_explicit_tu_overrides_inherited_fp32(self):
        self.assertEqual(self.select('tu', '0'), {'CLEF_ATTN_TU': '1'})

    def test_default_is_qualified_attention_not_inherited_environment(self):
        self.assertEqual(self.select(None, '0'), {'CLEF_ATTN_TU': '1'})


if __name__ == '__main__':
    unittest.main()
