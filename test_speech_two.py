"""Speech bugs found on real notes: case-sensitive speaker names, and Whisper end-of-clip filler.
Synthetic words and fake front-end only; a temp RAVEN_EARS_HOME."""
from contextlib import redirect_stdout
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import raven_ears_plus as R
import voice_book as V
from test_cpu_profile import card_fixture

ROW = json.dumps({'pitch_hz': 160., 'loud_db': -22.}) + '\n'


def fe_with(voiced_from=None, voiced_to=None, seconds=60):
    n = int(seconds * R.SR / R.HOP) + 2
    voiced = np.zeros(n, bool)
    voiced[int(voiced_from * R.SR / R.HOP):int(voiced_to * R.SR / R.HOP)] = True
    return {'voiced': voiced}


class SpeakerCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        env = patch.dict(os.environ, {'RAVEN_EARS_HOME': self.d.name})
        env.start()
        self.addCleanup(env.stop)

    def test_any_case_is_one_lowercase_file(self):
        self.assertEqual(R.baseline_path('Fox').name, 'fox.jsonl')
        self.assertEqual(R.baseline_path('FOX'), R.baseline_path('fox'))

    def test_lowercase_history_is_found_from_typed_case(self):
        (R.baseline_dir() / 'fox.jsonl').write_text(ROW * 13)
        self.assertEqual(len(R._read_baseline(R.baseline_path('Fox'))), 13)
        self.assertEqual(R.baseline_conflict('Fox'), [])

    def test_lone_legacy_file_is_read_and_appended_not_duplicated(self):
        legacy = R.baseline_dir() / 'Fox.jsonl'
        legacy.write_text(ROW * 3)
        self.assertEqual(R.baseline_path('fox'), legacy)
        self.assertEqual(R.baseline_path('Fox'), legacy)
        with open(R.baseline_path('FOX'), 'a') as fh:
            fh.write(ROW)
        self.assertEqual(sorted(p.name for p in R.baseline_dir().glob('*.jsonl')), ['Fox.jsonl'])
        self.assertEqual(len(legacy.read_text().splitlines()), 4)

    def test_both_cases_use_lowercase_untouched_and_conflict_is_reported(self):
        (R.baseline_dir() / 'fox.jsonl').write_text(ROW * 13)
        (R.baseline_dir() / 'Fox.jsonl').write_text(ROW * 2)
        self.assertEqual(R.baseline_path('Fox').name, 'fox.jsonl')
        self.assertEqual(R.baseline_conflict('Fox'), ['Fox.jsonl', 'fox.jsonl'])
        self.assertEqual(len((R.baseline_dir() / 'Fox.jsonl').read_text().splitlines()), 2)

    def test_card_says_still_learning_no_more_and_warns_on_both(self):
        (R.baseline_dir() / 'fox.jsonl').write_text(ROW * 13)
        (R.baseline_dir() / 'Fox.jsonl').write_text(ROW * 2)
        with card_fixture() as (path, *_):
            out = io.StringIO()
            with redirect_stdout(out):
                R.main([str(path), '--quiet', '--no-picture', '--speaker', 'Fox', '--no-baseline-update'])
        text = out.getvalue()
        self.assertNotIn('still learning', text)
        self.assertIn('differ only by letter case; using fox.jsonl, not merged', text)

    def test_voice_book_finds_existing_print_in_any_case(self):
        with patch('voice_book.home', return_value=R.baseline_dir() / 'voices'):
            V.home().mkdir()
            (V.home() / 'Fox.json').write_text('{}')
            self.assertEqual(V.voice_path('fox').name, 'Fox.json')
            self.assertEqual(V.voice_path('Fox').name, 'Fox.json')
            self.assertEqual(V.voice_path('new').name, 'new.json')


class TailFiller(unittest.TestCase):
    DUR = 40.
    SPEECH = [('i', 5., 5.2), ('love', 5.2, 5.5), ('you', 5.5, 5.8), ('too', 5.8, 6.1)]

    def split(self, words, fe=None, dur=DUR):
        return R.split_tail_filler(words, fe, dur)

    def test_tail_thank_you_over_silence_is_marked_not_dropped(self):
        words = self.SPEECH + [('i', 34., 34.2), ('love', 34.2, 34.4), ('you', 34.4, 34.6), ('too', 34.6, 34.8),
                               ('Thank', 36.0, 36.3), ('you!', 36.3, 36.6)]
        spoken, filler = self.split(words, fe_with(34., 34.8))
        self.assertEqual([w[0] for w in filler], ['Thank', 'you!'])
        self.assertEqual(spoken[-1][0], 'too')

    def test_thanks_for_watching_after_a_gap_is_marked(self):
        words = self.SPEECH + [('and', 30., 30.2), ('why', 30.2, 30.5), ('Thank', 36., 36.2), ('you', 36.2, 36.4),
                               ('for', 36.4, 36.6), ('watching!', 36.6, 37.)]
        spoken, filler = self.split(words, None)
        self.assertEqual(' '.join(w[0] for w in filler), 'Thank you for watching!')

    def test_real_thank_you_over_voice_with_no_gap_is_kept(self):
        words = [('so', 33., 33.2), ('thank', 33.3, 33.6), ('you', 33.6, 33.9)]
        spoken, filler = self.split(words, fe_with(33., 34.))
        self.assertEqual(filler, [])
        self.assertEqual(len(spoken), 3)

    def test_mid_clip_thank_you_is_never_marked_even_after_a_gap_and_silence(self):
        words = [('hi', 1., 1.2), ('Thank', 10., 10.3), ('you', 10.3, 10.6), ('ok', 30., 30.3), ('then', 30.3, 30.6)]
        spoken, filler = self.split(words, fe_with(30., 31.))
        self.assertEqual(filler, [])
        self.assertEqual(spoken, words)

    def test_tail_phrase_without_evidence_when_voiced_and_close_is_kept(self):
        words = [('well', 33., 33.3), ('thanks', 33.4, 33.8)]
        self.assertEqual(self.split(words, fe_with(33., 34.))[1], [])

    def _card(self, words):
        with card_fixture() as (path, model, *_):
            out = io.StringIO()
            with patch.object(R, '_whisper_words', return_value=words), redirect_stdout(out):
                with patch.object(R, 'split_tail_filler', wraps=R.split_tail_filler):
                    R.main([str(path), '--quiet', '--no-picture'])
        return out.getvalue()

    def test_card_marks_filler_notes_it_and_pace_excludes_it(self):
        base = [('Check', 1., 1.4), ('the', 1.4, 1.6), ('signal', 1.6, 2.0), ('now', 2.0, 2.4)]
        plain = self._card(base)
        withf = self._card(base + [('Thank', 56., 56.3), ('you!', 56.3, 56.6)])
        pace = lambda t: next(l for l in t.splitlines() if l.startswith('  PACE'))
        self.assertEqual(pace(plain), pace(withf))
        self.assertIn('⟪≈ likely Whisper filler, not spoken⟫ [0:56] Thank you!', withf)
        self.assertIn("looks like Whisper's end-of-clip", withf)
        self.assertNotIn('likely Whisper filler', plain)
        self.assertIn('  WORDS:   4 words', withf)

    def test_card_keeps_mid_clip_thank_you(self):
        words = [('Check', 1., 1.4), ('thank', 20., 20.3), ('you', 20.3, 20.6), ('now', 21., 21.4)]
        text = self._card(words)
        self.assertNotIn('likely Whisper filler', text)
        self.assertIn('  WORDS:   4 words', text)


if __name__ == '__main__':
    unittest.main()
