"""Invented chroma shaped like a measured bounded lean and later re-entry."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import music as M
import key_opening
from music import reentry_changes as refine


def case(shift=0):
    fps = M.SR / M.HOP
    home = np.array([.40, .19, .18, .10, .16, .18, .36, .45, .24, .32, .20, .28])
    lean = np.array([.42, .42, .56, .42, .60, .30, .36, .39, .28, .49, .23, .42])
    gap = np.array([.44, .18, .30, .12, .47, .30, .21, .34, .22, .65, .24, .31])
    arrival = 1.3 * np.array([.26, .47, .57, .40, .43, .33, .29, .20, .36, .45, .20, .34])
    chroma = np.repeat(home[:, None], int(220 * fps), axis=1)
    for a, b, vector in ((100, 140, lean), (148, 156, gap), (168, 220, arrival)):
        chroma[:, int(a * fps):int(b * fps)] = vector[:, None]
    chroma = np.roll(chroma, shift, axis=0)
    # The input is the raw estimate after home fusion has rechecked fifth pulls.
    info = dict(key=M._kname(12 + (4 + shift) % 12),
                changes=[dict(at=99.8, to=M._kname((9 + shift) % 12))])
    stops = dict(status='ran', events=[dict(start_s=166., length_s=2., kind='band cuts, voice holds')])
    return info, chroma, 220., ((4 + shift) % 12, 'minor'), stops


class KeyReentry(unittest.TestCase):
    def test_measured_shape_demotes_lean_and_recovers_the_later_arrival(self):
        args = case()
        before = copy.deepcopy(args[0])
        raw = M.analyze_key(args[1], args[2])
        self.assertEqual([c['at'] for c in raw['changes'] if c['at'] >= 25], [99.8])
        result = refine(*args)
        self.assertEqual(args[0], before)
        self.assertEqual(result['changes'][0]['suppressed'], 'bounded key lean')
        self.assertEqual(result['changes'][1]['at'], 168)
        settled = key_opening.settled(result, args[3], lambda s: (M.NAMES.index(s.split()[0]), s.split()[1]))
        self.assertEqual([(c['at'], c['to']) for c in settled['changes']], [(168, 'A major')])
        self.assertEqual(len(settled['raw_changes']), 2)
        self.assertEqual(refine(result, *args[1:]), result)

    def test_transposition_and_relative_home_do_not_change_the_rule(self):
        for shift in range(12):
            args = case(shift)
            result = refine(*args)
            self.assertEqual(result['changes'][-1]['at'], 168, shift)
            self.assertEqual(result['changes'][-1]['to'], M._kname((9 + shift) % 12))
            major = list(args)
            major[3] = ((7 + shift) % 12, 'major')
            self.assertEqual(refine(*major), result)

    def test_missing_uncertain_failed_or_partial_stops_preserve_the_estimate(self):
        for kind in ('uncertain', 'not a stop'):
            args = case()
            args[-1]['events'][0]['kind'] = kind
            self.assertEqual(refine(*args), args[0])
        for status in ('failed', 'unavailable'):
            args = case()
            args[-1]['status'] = status
            self.assertEqual(refine(*args), args[0])
        args = case()
        args[-1]['events'][0]['arrangement'] = 'partial cut'
        self.assertEqual(refine(*args), args[0])
        self.assertEqual(refine(*case()[:-1], None), case()[0])

    def test_a_supported_earlier_modulation_keeps_its_time(self):
        args = case()
        fps = M.SR / M.HOP
        args[1][1, int(100 * fps):int(140 * fps)] = .46
        args[1][8, int(100 * fps):int(140 * fps)] = .44
        for a in (int(100 * fps), int(110 * fps), int(120 * fps)):
            self.assertEqual(int(M._key_scores(args[1][:, a:a + int(20 * fps)].mean(axis=1)).argmax()), 9)
        self.assertEqual(refine(*args), args[0])

    def test_no_return_home_or_no_interruption_preserves_the_estimate(self):
        fps = M.SR / M.HOP
        for start in (100, 156):
            args = case()
            args[1][:, int(start * fps):int(168 * fps)] = args[1][:, int(180 * fps), None]
            self.assertEqual(refine(*args), args[0])

    def test_a_long_sustained_later_key_and_new_signature_are_required(self):
        fps = M.SR / M.HOP
        args = case()
        self.assertEqual(refine(args[0], args[1][:, :int(200 * fps)], 200, *args[3:]), args[0])
        args = case()
        # The target still wins, but one new signature note is below the old note.
        args[1][8, int(168 * fps):] = .24
        args[1][7, int(168 * fps):] = .27
        self.assertEqual(int(M._key_scores(args[1][:, int(168 * fps):int(188 * fps)].mean(axis=1)).argmax()), 9)
        self.assertEqual(refine(*args), args[0])

    def test_intervening_change_and_existing_suppression_are_preserved(self):
        args = case()
        args[0]['changes'].append(dict(at=145, to='C major'))
        self.assertEqual(refine(*args), args[0])
        args = case()
        args[0]['changes'][0]['suppressed'] = 'dominant pull'
        self.assertEqual(refine(*args), args[0])

    def test_missing_short_nonfinite_and_opening_evidence_do_not_correct(self):
        args = case()
        self.assertEqual(refine(dict(key='E minor'), *args[1:]), dict(key='E minor'))
        self.assertEqual(refine(args[0], args[1][:, :20], *args[2:]), args[0])
        args[1][0, 100] = np.nan
        self.assertEqual(refine(*args), args[0])
        args = case()
        args[0]['changes'][0]['at'] = 0
        self.assertEqual(refine(*args), args[0])

    def test_harmony_uses_only_complete_cached_stems(self):
        import harmony as H
        args = case()
        with patch.object(H, 'vocal_stem_path', return_value='invented-stems/vocals.wav'), \
             patch.object(H.os.path, 'isfile', return_value=True), \
             patch('stops.analyse', return_value=args[-1]) as energy:
            result = H.reentry_key_changes('invented.wav', args[0], {'chroma': args[1]}, *args[2:4])
        energy.assert_called_once_with('invented-stems')
        self.assertEqual(result, refine(*args))
        for vocal, complete in ((None, True), ('invented-stems/vocals.wav', False)):
            with patch.object(H, 'vocal_stem_path', return_value=vocal), \
                 patch.object(H.os.path, 'isfile', return_value=complete), patch('stops.analyse') as energy:
                self.assertEqual(H.reentry_key_changes('invented.wav', args[0], {'chroma': args[1]}, *args[2:4]), args[0])
                energy.assert_not_called()

    def test_failed_stop_measurement_preserves_keys_and_reports_failure(self):
        import harmony as H
        args = case()
        with patch.object(H, 'vocal_stem_path', return_value='invented-stems/vocals.wav'), \
             patch.object(H.os.path, 'isfile', return_value=True), \
             patch('stops.analyse', return_value={'status': 'failed', 'reason': 'invented decoder failure'}), \
             patch.object(H.ST, 'mark') as mark:
            self.assertEqual(H.reentry_key_changes('invented.wav', args[0], {'chroma': args[1]}, *args[2:4]), args[0])
        mark.assert_called_once_with('key boundary', 'failed', 'invented decoder failure')


if __name__ == '__main__':
    unittest.main()
