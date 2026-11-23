"""Synthetic pitch tracks for the speech TUNE card."""
import unittest

import numpy as np

import raven_ears_plus as R


def track(st, times):
    f0 = 160. * np.exp2(np.asarray(st) / 12.)
    return dict(f0=f0, voiced=np.isfinite(f0), active=np.ones(len(times), bool), times=times)


class SpeechTuneTests(unittest.TestCase):
    def test_rise_fall_level_and_unvoiced_end(self):
        times = np.arange(5 * int(R.FPS * 2)) / R.FPS
        words = [('paper', 2. * i, 2. * i + 1.2) for i in range(4)]
        st = np.full(len(times), np.nan)
        for i, (a, b) in enumerate(((0, 8), (8, 0), (2, 2), (0, 8))):
            sel = (times >= 2. * i) & (times < 2. * i + (1.2 if i < 3 else .8))
            st[sel] = np.linspace(a, b, sel.sum())
        fe = track(st, times)
        pitch = R.analyze_pitch(fe, words)
        tune = R.analyze_tune(fe, words, pitch, [{'range_st': 6.}, {'range_st': 8.}, {'range_st': None}])
        self.assertEqual([p['ending'] for p in tune['phrases']], ['↑', '↓', '→', '?'])
        self.assertEqual(tune['usual_range_st'], 7.)
        self.assertEqual(tune['baseline_n'], 2)
        self.assertEqual(tune['range_st'], pitch['range_st'])
        lines = R.tune_lines(tune, words)
        self.assertIn('usual 7.0 st (n=2)', lines[0])
        self.assertIn('↗+', lines[1])
        self.assertIn('↘-', lines[2])

    def test_arch_jump_word_and_single_frame_octave_spike(self):
        times = np.arange(100) / R.FPS
        words = [('fold', 0., .38), ('paper', .38, .74), ('into', .74, 1.03), ('stars', 1.03, 1.3)]
        st = np.where((times >= .42) & (times < .75), 8., 0.)
        st[5] = 12.
        fe = track(st, times)
        tune = R.analyze_tune(fe, words, R.analyze_pitch(fe, words), [])
        phrase, = tune['phrases']
        self.assertEqual(phrase['points_st'], [0., 8., 0.])
        self.assertEqual(phrase['jump'], dict(word_index=1, st=8.))
        line = R.tune_lines(tune, words)[1]
        self.assertIn('"…into stars" +0↗+8↘+0 st', line)
        self.assertIn('(lift on "paper")', line)
        self.assertNotIn('fold', line)
        self.assertLessEqual(len(line), R.MAX_LINE)

    def test_keeps_all_shapes_and_prints_only_six_in_time_order(self):
        times = np.arange(16 * R.FPS) / R.FPS
        st = np.full(len(times), np.nan)
        words = []
        for i, height in enumerate([1, 9, 2, 6, 5, 8, 3, 7]):
            start = 2. * i
            sel = (times >= start) & (times < start + 1.25)
            st[sel] = np.linspace(0., height, sel.sum())
            words.append(('lantern', start, start + 1.25))
        fe = track(st, times)
        tune = R.analyze_tune(fe, words, R.analyze_pitch(fe, words), [])
        self.assertEqual(len(tune['phrases']), 8)
        self.assertEqual([line.split()[0] for line in R.tune_lines(tune, words)[1:]],
                         ['0:02', '0:06', '0:08', '0:10', '0:12', '0:14'])

    def test_framed_phrases_without_words_and_no_voiced_track(self):
        times = np.arange(100) / R.FPS
        fe = track(np.linspace(-6., 6., len(times)), times)
        tune = R.analyze_tune(fe, None, R.analyze_pitch(fe, None), [])
        line = R.tune_lines(tune, None)[1]
        self.assertIn(' st ↑', line)
        self.assertNotIn('"', line)
        fe = track(np.full(len(times), np.nan), times)
        self.assertIsNone(R.analyze_tune(fe, None, R.analyze_pitch(fe, None), []))


def run_checks(check):
    for case in unittest.defaultTestLoader.loadTestsFromTestCase(SpeechTuneTests):
        result = unittest.TestResult()
        case.run(result)
        check('TUNE: ' + case._testMethodName, result.wasSuccessful(),
              '\n'.join(detail for _, detail in result.failures + result.errors))


if __name__ == '__main__':
    unittest.main()
