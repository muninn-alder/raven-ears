"""Invented vectors and fake inference, never synthetic-audio calibration.

Print schema follows the real Resemblyzer measurement (256 values/count/seconds).
CLI cards reuse the existing fixture's real speech-card numeric/column shapes.
"""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import raven_ears_plus as R
import voice_book as V
from test_cpu_profile import card_fixture


def vector(a=1., b=0.):
    v = np.zeros(256, np.float32)
    v[:2] = a, b
    return v / np.linalg.norm(v)


class VoiceBookTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(os.environ, {'RAVEN_EARS_HOME': self.directory.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.encoder = Mock()
        self.encoder.embed_utterance.return_value = vector()
        self.backend = patch.object(V, '_backend', return_value=(self.encoder, lambda w, **kw: w))
        self.backend.start()
        self.addCleanup(self.backend.stop)

    def print_voice(self, name='Aster', embedding=None):
        path = V.voice_path(name)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(embedding=(vector() if embedding is None else embedding).tolist(),
                                        count=2, seconds=20.)))
        return path

    def cli(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(R.main(args), 0)
        return output.getvalue()

    def test_empty_book_and_missing_encoder_preserve_default_card_bytes(self):
        with card_fixture(voice_home=V.home()) as (path, *_):
            expected = Path('fixtures/speech-default.card.txt').read_bytes()
            actual = self.cli([str(path), '--quiet', '--no-picture'])
            self.assertEqual(actual.encode(), expected)
            V._backend.assert_not_called()
            self.print_voice()
            with patch.object(V, '_backend', side_effect=ImportError('optional encoder absent')):
                self.assertEqual(self.cli([str(path), '--quiet', '--no-picture']).encode(), expected)

    def test_enrol_cli_keeps_only_mean_count_seconds_permissions_and_forget(self):
        with (patch.object(R, 'load_audio', return_value=np.ones(6*V.SR, np.float32)),
              redirect_stderr(io.StringIO()) as notices):
            self.assertIn('Enrolled Aster', self.cli(['--enrol', 'Aster', 'first.webm', 'second.m4a']))
        path = V.voice_path('Aster')
        with tempfile.NamedTemporaryFile(dir=path.parent) as probe:
            os.fchmod(probe.fileno(), 0o644)
            before_mode = stat.S_IMODE(os.fstat(probe.fileno()).st_mode)
            os.fchmod(probe.fileno(), 0o600)
            after_mode = stat.S_IMODE(os.fstat(probe.fileno()).st_mode)
        ignores_chmod = before_mode == after_mode != 0o600
        warning = 'Warning: the voiceprint could not be made private'
        self.assertEqual(notices.getvalue().count(warning), int(ignores_chmod))
        row = json.loads(path.read_text())
        self.assertEqual(set(row), {'embedding', 'count', 'seconds'})
        self.assertEqual((row['count'], row['seconds']), (2, 12.))
        with self.subTest(mode='new voiceprint'):
            if ignores_chmod:
                self.skipTest('this filesystem leaves the probe mode unchanged after chmod 0600')
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(len(row['embedding']), 256)
        self.assertIn('Aster: 2 clips, 12.0s', self.cli(['--voices']))
        # Replacement cannot leave an old vector's count or permissions behind.
        path.chmod(0o644)
        with (patch.object(R, 'load_audio', return_value=np.ones(10*V.SR, np.float32)),
              redirect_stderr(io.StringIO()) as notices):
            self.cli(['--enrol', 'Aster', 'third.mp3'])
        self.assertEqual(json.loads(path.read_text())['count'], 1)
        self.assertEqual(notices.getvalue().count(warning), int(ignores_chmod))
        with self.subTest(mode='replacement voiceprint'):
            if ignores_chmod:
                self.skipTest('this filesystem leaves the probe mode unchanged after chmod 0600')
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        baseline = R.baseline_path('Aster')
        baseline.write_text('separate history\n')
        self.cli(['--forget', 'Aster'])
        self.assertFalse(path.exists())
        self.assertEqual(baseline.read_text(), 'separate history\n')

    def test_failed_enrol_does_not_replace_existing_print(self):
        path = self.print_voice()
        before = path.read_bytes()
        for wav in (np.zeros(12*V.SR, np.float32), np.ones(2*V.SR, np.float32)):
            with self.subTest(seconds=len(wav)/V.SR), self.assertRaises(ValueError):
                V.enrol('Aster', ['invented'], lambda *a: wav)
            self.assertEqual(path.read_bytes(), before)

    def test_names_and_damaged_prints_cannot_escape_or_break_listening(self):
        for name in ('../escape', '.', '/outside', 'A/B', 'A B', '', 'x'*65):
            with self.subTest(name=name), self.assertRaises(ValueError):
                V.voice_path(name)
        path = self.print_voice()
        good = json.loads(path.read_text())
        for bad in ({}, dict(good, embedding=[1., 0.]), dict(good, embedding=[float('nan')]*256),
                    dict(good, embedding=[0.]*256), dict(good, seconds=-1), dict(good, count=True)):
            path.write_text(json.dumps(bad))
            self.assertEqual(V.voices(), {})
        path.write_text('broken JSON')
        self.assertIsNone(V.identify(np.ones(10*V.SR)))

    def test_score_floor_gap_and_short_voice_abstain(self):
        self.print_voice(embedding=vector(.7, np.sqrt(1-.7**2)))
        result = V.identify(np.ones(10*V.SR))
        self.assertEqual(result['status'], 'unknown')
        self.assertIn('closest Aster 0.70', V.voice_line(result))
        self.print_voice(embedding=vector())
        self.print_voice('Birch', vector(.99, .1))
        self.assertEqual(V.identify(np.ones(10*V.SR))['status'], 'unknown')
        result = V.identify(np.ones(V.SR))
        self.assertEqual(result['ranked'], [])
        self.assertIn('too little voice', V.voice_line(result))
        self.assertTrue(V._confident([('Aster', .80), ('Birch', .69)]))
        self.assertFalse(V._confident([('Aster', .80), ('Birch', .71)]))

    def test_conflicting_segments_veto_whole_clip_guess(self):
        self.print_voice()
        self.print_voice('Birch', vector(0., 1.))
        self.encoder.embed_utterance.side_effect = [vector(), vector(), vector(0., 1.)]
        result = V.identify(np.ones(10*V.SR))
        self.assertEqual(result['status'], 'unknown')
        self.assertIn('segments disagree', V.voice_line(result))

    def test_cli_guess_reads_history_and_serializes_field_but_never_teaches(self):
        self.print_voice()
        baseline = R.baseline_path('Aster')
        before = ''.join(json.dumps({'pitch_hz': 160., 'loud_db': -22.})+'\n' for _ in range(12))
        baseline.write_text(before)
        with card_fixture(voice_home=V.home()) as (path, *_):
            output = self.cli([str(path), '--quiet'])
            self.assertIn('VOICE: sounds like Aster (1.00; only enrolled voice)', output)
            self.assertIn('VS USUAL (speaker from the voice book)', output)
            self.assertIn('comparison only, guess never saved', output)
            self.assertNotIn('still learning', output)
            data = json.loads(self.cli([str(path), '--json', '--quiet']))
            self.assertEqual(data['voice']['status'], 'matched')
            self.assertEqual(data['voice']['ranked'][0], ['Aster', 1.])
            self.assertEqual(data['baseline_n_before'], 12)
        self.assertEqual(baseline.read_text(), before)

    def test_guess_with_no_history_does_not_create_one_and_unknown_never_reads(self):
        self.print_voice()
        with card_fixture(voice_home=V.home()) as (path, *_):
            self.cli([str(path), '--quiet'])
            self.assertFalse(R.baseline_path('Aster').exists())
            self.print_voice('Birch', vector(.99, .1))
            with patch.object(R, '_read_baseline', side_effect=AssertionError('unknown must not read history')):
                output = self.cli([str(path), '--quiet'])
                self.assertIn('not sure who this is', output)
                self.assertIn('no speaker given, no comparison', output)

    def test_explicit_speaker_wins_and_can_still_teach(self):
        self.print_voice()
        with card_fixture() as (path, *_), patch.object(V, 'identify', side_effect=AssertionError('explicit wins')), \
             patch.dict(os.environ, {'RAVEN_EARS_NO_BASELINE': '0'}):
            output = self.cli([str(path), '--speaker', 'Birch', '--quiet'])
        self.assertFalse(any(line.startswith('  VOICE:') for line in output.splitlines()))
        self.assertTrue(R.baseline_path('Birch').exists())
        self.assertFalse(R.baseline_path('Aster').exists())

    def test_singer_does_not_consult_the_book(self):
        import singer
        self.print_voice()
        with card_fixture() as (path, *_), patch.object(singer, '_free_gpu'), \
             patch.object(V, 'identify', side_effect=AssertionError('songs excluded')):
            _, data, _ = singer.singer_block(path, 60)
        self.assertNotIn('voice', data)

    def test_management_conflicts_are_errors(self):
        for args in (['--enrol','Aster'], ['--voices','--forget','Aster'],
                     ['--voices','clip.wav'], ['--voices','--serve'], ['--voices','--setup']):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                R.main(args)
            self.assertEqual(error.exception.code, 2)


def run_checks(check):
    for name in unittest.defaultTestLoader.getTestCaseNames(VoiceBookTests):
        result = unittest.TestResult()
        VoiceBookTests(name).run(result)
        check('voice book: ' + name, result.wasSuccessful(), '\n'.join(t for _, t in result.errors + result.failures))


if __name__ == '__main__':
    unittest.main()
