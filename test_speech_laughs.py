"""Speech laugh rules and subprocess/eval contracts. No recordings or models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import eval_laughs as E
import laugh_models as M
import raven_ears_plus as R


class SpeechLaughTests(unittest.TestCase):
    def test_hint_thresholds_missing_pitch_and_long_multisyllable_words(self):
        words = [('like', 12., 12.4), ('um', 31., 31.4), ('paper', 32., 32.4), ('a', 33., 33.4)]
        feats = [dict(pitch_st=8., loud_db=None, dur_ratio=1.6) for _ in words]
        self.assertEqual([h['start'] for h in R.laugh_hint(words, feats)], [12., 31., 33.])
        for key, value in (('pitch_st', 7.99), ('pitch_st', None), ('pitch_st', float('nan')),
                           ('dur_ratio', 1.59), ('dur_ratio', None)):
            changed = [dict(f, **{key: value}) for f in feats]
            self.assertFalse(R.laugh_hint(words, changed))
        self.assertFalse(R.laugh_hint(words[:3], feats[:3]))
        short = [(w, s, s + .1) for w, s, _ in words]
        self.assertFalse(R.laugh_hint(short, feats))

    def test_hint_line_has_times_no_words_and_is_suppressed_by_laughter(self):
        hint = [dict(start=12.), dict(start=31.)]
        empty = dict(laughs=[], snorts=[])
        self.assertIn('none heard · laughing? big pitch jumps on short words at 0:12, 0:31', R.laughs_line(empty, hint))
        positive = dict(laughs=[dict(start=1., end=2., total=.5)], snorts=[])
        self.assertNotIn('laughing?', R.laughs_line(positive, hint))
        long = R.laughs_line(empty, [dict(start=float(i * 10000)) for i in range(100)])
        self.assertIn('laughing?', long)
        self.assertLessEqual(len(long), R.MAX_LINE)

    def test_model_switch_default_and_invalid(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(M.selected(), 'off')
        for name in M.MODELS:
            with patch.dict(os.environ, {'RAVEN_EARS_LAUGH_MODEL': name}):
                self.assertEqual(M.selected(), name)
        with patch.dict(os.environ, {'RAVEN_EARS_LAUGH_MODEL': 'unknown'}):
            with self.assertRaises(ValueError):
                M.selected()

    def test_optional_model_and_hint_both_fit_the_card(self):
        for spans in ([], [[1., 2.], [4., 5.], [6., 7.]]):
            line = R.laughs_line(dict(laughs=[], snorts=[]),
                                 [dict(start=12.), dict(start=31.)],
                                 dict(model='gillick', status='ran', spans=spans))
            # The model leads when it hears laughter (Fox's timed note, 17 Nov 2026); the
            # pitch hint is only offered when the model heard nothing.
            (self.assertNotIn if spans else self.assertIn)('laughing?', line)
            self.assertIn('gillick', line)
            self.assertIn('CNN14 none heard', line)
            self.assertLessEqual(len(line), R.MAX_LINE)

    def test_worker_scope_thread_caps_interpreter_and_numeric_result(self):
        process = MagicMock()
        process.communicate.return_value = (json.dumps(dict(status='ran', duration=3., spans=[[1., 2.]])), None)
        process.returncode = 0
        with patch.object(M.subprocess, 'Popen') as popen, patch.dict(os.environ, {'RAVEN_EARS_OMINE_PY': '/selected/python'}):
            popen.return_value.__enter__.return_value = process
            result = M.run_detector('omine', Path('invented.wav'))
        self.assertEqual(result['spans'], [[1., 2.]])
        cmd = popen.call_args.args[0]
        self.assertIn('MemoryMax=10G', cmd)
        self.assertIn('MemorySwapMax=0', cmd)
        self.assertIn('/selected/python', cmd)
        env = popen.call_args.kwargs['env']
        self.assertEqual(env['OMP_NUM_THREADS'], '4')
        self.assertEqual(env['RAVEN_EARS_NO_BASELINE'], '1')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')

    def test_failed_worker_is_not_silence_and_raw_output_is_discarded(self):
        for code, output in ((1, 'invented private diagnostics'), (0, 'invalid protocol')):
            process = MagicMock(returncode=code)
            process.communicate.return_value = (output, None)
            with patch.object(M.subprocess, 'Popen') as popen:
                popen.return_value.__enter__.return_value = process
                result = M.run_detector('gillick', Path('invented.wav'))
            self.assertEqual(result['status'], 'failed')
            self.assertNotIn('spans', result)
            self.assertNotIn(output, json.dumps(result))

    def test_timeout_kills_worker_group(self):
        process = MagicMock(pid=1234)
        process.communicate.side_effect = [subprocess.TimeoutExpired('worker', 1), ('', None)]
        with patch.object(M.subprocess, 'Popen') as popen, patch.object(M.os, 'killpg') as kill:
            popen.return_value.__enter__.return_value = process
            result = M.run_detector('omine', Path('invented.wav'), timeout=1)
        self.assertEqual(result['reason'], 'worker timed out')
        kill.assert_called_once_with(1234, M.signal.SIGKILL)

    def test_clamp_padded_spans_and_union(self):
        self.assertEqual(M.clean_spans([[2., 7.], [-1., 1.], [.5, 2.5], [8., 9.]], 5.), [[0., 5.]])
        with self.assertRaises(ValueError):
            M.clean_spans([[float('nan'), 1.]], 5.)

    def test_worker_hides_upstream_output_exceptions_and_uses_disposable_home(self):
        code = '''
import os
from pathlib import Path
import laugh_models as M
original = os.environ.get('RAVEN_EARS_HOME')
M.gpu_ready = lambda: None
def fake(name, audio):
    assert os.environ['RAVEN_EARS_HOME'] != original
    print('invented sensitive marker')
    os.write(1, b'native sensitive marker\\n')
    raise RuntimeError('invented sensitive exception')
M._measure = fake
M.worker('hint', Path('invented.wav'))
'''
        env = dict(os.environ, RAVEN_EARS_HOME='unreadable-invented-home')
        completed = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout), dict(status='failed', reason='RuntimeError'))
        self.assertEqual(completed.stderr, '')
        self.assertNotIn('sensitive', completed.stdout)

    def test_eval_sparse_labels_and_unavailable_not_empty(self):
        labels = {'invented.webm': dict(laugh_under_speech_whole=True, not_sneeze=[[24., 25.5]])}
        def runner(task, path):
            if task == 'panns':
                return dict(model=task, status='ran', duration=40., spans=[[1., 2.]], seconds=2.,
                            thresholds={'0.2': [[1., 2.]], '0.15': [[1., 3.]], '0.1': [[1., 4.]]},
                            dogs=[], sneezes=[[24., 25.]])
            return dict(model=task, status='failed', reason='unavailable', seconds=.1)
        rows, measurements = E.evaluate([Path('invented.webm')], labels, runner)
        self.assertEqual(len(rows), 9)
        self.assertIn('2.5%', rows[0]['agreement'])
        self.assertIn('not-sneeze: 1', rows[0]['agreement'])
        self.assertIsNone(rows[-1]['spans'])
        self.assertIn('unmeasured', E.table(rows))
        self.assertNotIn('invented.webm', json.dumps(measurements))
        self.assertIn('not scored', E.agreement('omine', [], 40., dict(smiling_whole=True)))
        self.assertIn('not scored', E.agreement('hint', [], 40., dict(laugh_under_speech_whole=True)))

    def test_timed_labels_and_matches_are_reported_by_time(self):
        data = {'invented.wav': dict(laughs=[[4., 5.], [1., 2.], [5., 5.]],
                                     dog_barks_faint=[[7., 8.], [9., 10.]],
                                     free_text='this should remain private')}
        label = E.read_labels(MagicMock(**{'read_text.return_value': json.dumps(data)}))['invented.wav']
        self.assertEqual(label, dict(laughs=[[1., 2.], [4., 5.]], dog_barks_faint=[[7., 8.], [9., 10.]]))
        text = E.agreement('omine', [[1.5, 3.], [5., 6.]], 10., label)
        self.assertIn('1/2 hits: 1.00-2.00 via 1.50-3.00', text)
        self.assertIn('misses: 4.00-5.00', text)
        self.assertIn('unmatched: 5.00-6.00', text)
        self.assertEqual(len(E.timed_matches([[6., 11.]], label['dog_barks_faint'])['hits']), 1)
        self.assertIn('control dog spans: none', E.agreement('ROOM dog (raw)', [], 10., {}))

    def test_eval_compares_both_faint_passes_and_keeps_room_control_line(self):
        span = dict(start=4.75, end=5.75, score=.12, time=5.)
        row = dict(status='ran', dogs=[span], room_line='  ROOM: dog, faint (0:05)')
        def runner(task, path):
            if task == 'panns':
                return dict(model=task, status='ran', duration=10., spans=[], seconds=2.,
                            thresholds={'0.2': [], '0.15': [], '0.1': []}, dogs=[],
                            room_variants={'raw': row, 'voice-reduced': row})
            return dict(model=task, status='ran', duration=10., spans=[], seconds=1.)

        labels = {'invented-a.wav': dict(dog_barks_faint=[[5., 6.], [7., 8.]])}
        rows, _ = E.evaluate([Path('invented-a.wav'), Path('invented-b.wav')], labels, runner)
        self.assertIn('0/2 hits', rows[3]['agreement'])
        self.assertIn('1/2 hits', rows[4]['agreement'])
        table = E.table(rows)
        self.assertIn('raw + voice-reduced', table)
        self.assertIn('control dog spans: 4.75-5.75', table)
        self.assertIn('dog, faint (0:05)', table)
        self.assertNotIn('invented-', table)


def run_checks(check):
    for case in unittest.defaultTestLoader.loadTestsFromTestCase(SpeechLaughTests):
        result = unittest.TestResult()
        case.run(result)
        check('speech laughs: ' + case._testMethodName, result.wasSuccessful(),
              '\n'.join(detail for _, detail in result.failures + result.errors))


if __name__ == '__main__':
    unittest.main()
