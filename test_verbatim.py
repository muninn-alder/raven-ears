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
        self.assertTrue(kw['initial_prompt'].startswith('Zibble.'))      # names first: the tail survives truncation
        self.assertTrue(kw['initial_prompt'].endswith(R.VERBATIM_STYLE))
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


class PromptBudgetTests(unittest.TestCase):
    """Whisper keeps only the LAST 223 prompt tokens; the style must be the tail and a trim must be said."""

    class FakeTok:                                  # one token per whitespace/comma-separated word
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=text.replace(",", " ").split())

    def model(self):
        return SimpleNamespace(hf_tokenizer=self.FakeTok(), max_length=448)

    def test_short_prompt_is_untouched_apart_from_order(self):
        import lyrics_shared as LS
        p, note = LS.fit_prompt(self.model(), ["Zibble", "Aoife"], "Umm, like, you know.")
        self.assertEqual(p, "Zibble, Aoife. Umm, like, you know.")
        self.assertIsNone(note)
        self.assertEqual(LS.fit_prompt(self.model(), ["Zibble"])[0], "Zibble.")   # names-only: same as before

    def test_long_style_plus_names_keeps_style_and_says_so(self):
        import lyrics_shared as LS
        style = " ".join(f"s{i}" for i in range(200))
        names = [f"n{i}" for i in range(30)]
        p, note = LS.fit_prompt(self.model(), names, style)
        self.assertTrue(p.endswith(style))
        self.assertLessEqual(len(p.replace(",", " ").split()), 223)
        self.assertIn("names prompt trimmed", note)
        self.assertIn("of 30 names dropped", note)
        self.assertIn("223", note)
        self.assertTrue(p.startswith("n0, n1"))     # names drop from the END of the list

    def test_style_alone_over_budget_is_trimmed_and_reported(self):
        import lyrics_shared as LS
        p, note = LS.fit_prompt(self.model(), ["A"], " ".join(f"w{i}" for i in range(300)))
        self.assertEqual(len(p.split()), 223)
        self.assertIn("style prompt trimmed, 77 tokens over", note)

    def test_names_only_over_budget_is_trimmed_and_reported(self):
        import lyrics_shared as LS
        notes = []
        model = SimpleNamespace(hf_tokenizer=self.FakeTok(), max_length=448, transcribe=lambda y, initial_prompt=None: 0)
        names = [f"n{i} x y z" for i in range(100)]
        nk = LS.whisper_name_kwargs(model, names, mode="prompt", notes=notes)
        self.assertLessEqual(len(nk["initial_prompt"].replace(",", " ").split()), 223)
        self.assertTrue(notes and "names prompt trimmed" in notes[0])

    def test_no_tokenizer_means_no_trim(self):
        import lyrics_shared as LS
        p, note = LS.fit_prompt(SimpleNamespace(), ["A", "B"], "style")
        self.assertEqual((p, note), ("A, B. style", None))

    def test_verbatim_kwargs_puts_style_last_and_notes_trim(self):
        names = [f"n{i}" for i in range(30)]
        style = " ".join(f"s{i}" for i in range(200))
        notes = []
        with patch.object(R, 'verbatim_style', return_value=style):
            kw = R.verbatim_kwargs({'initial_prompt': ', '.join(names) + '.'}, self.model(), names, notes)
        self.assertTrue(kw['initial_prompt'].endswith(style))
        self.assertTrue(any('names prompt trimmed' in n for n in notes))

    def test_real_whisper_tokenizer_if_cached(self):
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel("large-v3", device="cpu", compute_type="int8", local_files_only=True)
        except Exception as e:
            self.skipTest(f"real tokenizer unavailable offline ({type(e).__name__})")
        import lyrics_shared as LS
        names = [f"Zibblewick{i}" for i in range(30)]
        style = ("Umm, let me think like, hmm... okay so here's what I'm, like, thinking, you know? " * 8)[:600]
        p, note = LS.fit_prompt(m, names, style)
        self.assertTrue(p.endswith(style))
        self.assertLessEqual(LS._token_counter(m)(p), 223)
        self.assertIn("names prompt trimmed", note)
