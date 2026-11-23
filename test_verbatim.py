"""Verbatim mode: faster-whisper kwargs only, mocked model, no real audio."""
import os, sys, unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import raven_ears_plus as R


def run(env, nk):
    seen = {}

    def transcribe(y, **kwargs):
        seen.update(kwargs)
        return [], None
    model = SimpleNamespace(transcribe=transcribe)
    notes = []
    with patch.dict(os.environ, env, clear=False), \
         patch.object(R, '_preload_cuda12'), patch.object(R, '_pick_device', return_value='cpu'), \
         patch.object(R, '_torch', return_value=None), patch('lyrics_shared.whisper_name_kwargs', return_value=nk), \
         patch('lyrics_shared.effective_names', return_value=['Zibble']), \
         patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=Mock(return_value=model))}):
        R._whisper_words(np.zeros(16000), notes)
    return seen, notes


class VerbatimTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop('RAVEN_EARS_VERBATIM', None)

    def tearDown(self):
        os.environ.pop('RAVEN_EARS_VERBATIM', None)
        if self._saved is not None:
            os.environ['RAVEN_EARS_VERBATIM'] = self._saved

    def test_off_by_default_keeps_current_kwargs(self):
        kw, notes = run({}, {'hotwords': 'Zibble'})
        self.assertEqual(kw['hotwords'], 'Zibble')
        self.assertTrue(kw['vad_filter'])
        self.assertFalse(kw['condition_on_previous_text'])
        for k in ('initial_prompt', 'vad_parameters', 'hallucination_silence_threshold'):
            self.assertNotIn(k, kw)
        self.assertFalse(any('verbatim' in n for n in notes))

    def test_on_adds_style_vad_and_silence_guard_and_keeps_hotwords(self):
        kw, notes = run({'RAVEN_EARS_VERBATIM': '1'}, {'hotwords': 'Zibble'})
        self.assertEqual(kw['hotwords'], 'Zibble')
        self.assertEqual(kw['initial_prompt'], R.VERBATIM_STYLE)
        self.assertTrue(kw['vad_filter'])
        self.assertEqual(kw['vad_parameters']['min_silence_duration_ms'], 500)
        self.assertEqual(kw['hallucination_silence_threshold'], 2.0)
        self.assertFalse(kw['condition_on_previous_text'])
        self.assertTrue(kw['word_timestamps'])
        self.assertEqual(sum('verbatim mode' in n for n in notes), 1)

    def test_style_is_joined_to_a_names_prompt(self):
        kw, _ = run({'RAVEN_EARS_VERBATIM': 'true'}, {'initial_prompt': 'Zibble.'})
        self.assertTrue(kw['initial_prompt'].startswith(R.VERBATIM_STYLE))
        self.assertTrue(kw['initial_prompt'].endswith('Zibble.'))
        self.assertNotIn('hotwords', kw)

    def test_flag_is_parsed_and_sets_env(self):
        self.assertFalse(R.verbatim_on())
        with patch.dict(os.environ, {'RAVEN_EARS_VERBATIM': '0'}):
            self.assertFalse(R.verbatim_on())
        with patch.dict(os.environ, {'RAVEN_EARS_VERBATIM': 'on'}):
            self.assertTrue(R.verbatim_on())
        from pathlib import Path
        self.assertIn('"--verbatim"', Path(R.__file__).read_text(encoding='utf-8'))


class VerbatimStyleTest(unittest.TestCase):
    def test_custom_style_text_and_file(self):
        import tempfile, os as _os
        from unittest.mock import patch as _patch
        with _patch.dict(_os.environ, {"RAVEN_EARS_VERBATIM_STYLE": "so, um, yeah   like"}):
            self.assertEqual(R.verbatim_style(), "so, um, yeah like")
            self.assertTrue(R.verbatim_kwargs()["initial_prompt"].startswith("so, um, yeah like"))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("okay okay, uh,\nI mean " + "x" * 2000)
        try:
            with _patch.dict(_os.environ, {"RAVEN_EARS_VERBATIM_STYLE": f.name}):
                style = R.verbatim_style()
            self.assertTrue(style.startswith("okay okay, uh, I mean"))
            self.assertEqual(len(style), 600)
        finally:
            _os.unlink(f.name)
        with _patch.dict(_os.environ, {"RAVEN_EARS_VERBATIM_STYLE": ""}):
            self.assertEqual(R.verbatim_style(), R.VERBATIM_STYLE)


if __name__ == '__main__':
    unittest.main()
