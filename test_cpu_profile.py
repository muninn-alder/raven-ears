"""CPU policy and speech cards: fake inference in the saved speech-4 card's shapes."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.signal import resample_poly  # Load extension modules before sys.modules is patched.
import ears_profile as EP
import raven_ears_plus as R
import stage_status as ST

ROOT = Path(__file__).resolve().parent
POLICY_ENV = ('RAVEN_EARS_PROFILE', 'RAVEN_EARS_WHISPER', 'WHISPER_MODEL', 'RAVEN_EARS_DEVICE')


@contextmanager
def card_fixture(engine=R, *, voice_home=None):
    # Numeric fields and line shapes checked against the real speech-4.card.txt.
    # Words are invented; no saved transcript or private fixture is needed.
    words = [('Check', 1.68, 2.1), ('the', 2.1, 2.3), ('signal.', 2.3, 2.8),
             ('Another', 12.26, 12.7), ('channel', 12.7, 13.1), ('opens.', 13.1, 13.6)]
    n = int(60 * R.FPS) + 1
    fe = dict(active=np.ones(n, bool), voiced=np.ones(n, bool), f0=np.full(n, 163.8),
              voiced_prob=np.full(n, .8), rms_db=np.full(n, -22.),
              times=np.arange(n) / R.FPS, n=n, thr=-60.)
    affect = dict(ok=True, device='cpu', n_windows=38,
                  whole=dict(arousal=.452, dominance=.533, valence=.424),
                  thirds=[dict(arousal=a, dominance=.53, valence=v) for a, v in
                          ((.421, .494), (.439, .400), (.499, .365))],
                  centers=np.array([1., 30., 59.]), P=np.full((3, 3), .45))
    events = dict(events=[], device='cpu', breath={'skipped': 'the event model does not hear breathing here'})
    laughs = dict(duration=60., laughs=[], snorts=[dict(start=48., end=49., label='snort', peak=.13, total=.13)])
    def enrich(d, *_):
        d['room'] = dict(hits=[], total_windows=117, missing_labels=[], reason='')
        d['whisper'] = dict(spans=[], reason='no qualifying word run', withheld=0, withheld_spans=[])
    def transcribe(*_, **__):
        return [SimpleNamespace(words=[SimpleNamespace(word=w, start=a, end=b) for w, a, b in words])], None
    model = Mock(return_value=SimpleNamespace(transcribe=transcribe))
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        path = Path(directory) / 'example.wav'
        path.touch()
        stack.enter_context(patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in POLICY_ENV}, clear=True))
        stack.enter_context(patch('voice_book.home', return_value=voice_home or Path(directory) / 'voices'))
        for name, value in (('load_audio', np.ones(60 * R.SR_SED, np.float32)),
                            ('_source_sample_rate', 16000), ('_front_end', fe), ('_torch', None),
                            ('analyze_channel', dict(classification='wideband-limited', rolloff_99_hz=5781.25)),
                            ('analyze_quality', dict(ok=False, reason='synthetic fixture'))):
            stack.enter_context(patch.object(engine, name, return_value=value))
        af = stack.enter_context(patch.object(engine, 'analyze_affect', side_effect=lambda *_: copy.deepcopy(affect)))
        ev = stack.enter_context(patch.object(engine, 'analyze_events', side_effect=lambda *_: copy.deepcopy(events)))
        laugh = stack.enter_context(patch('room_laugh.detect', side_effect=lambda *a, **k: copy.deepcopy(laughs)))
        stack.enter_context(patch.object(engine.RW, 'enrich', side_effect=enrich))
        for name in ('release_models', '_note_vram', '_preload_cuda12'):
            stack.enter_context(patch.object(engine, name))
        stack.enter_context(patch.object(engine, '_load_tagger', return_value=(Mock(), [])))
        stack.enter_context(patch.object(EP, '_torch', return_value=None))
        stack.enter_context(patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=model)}))
        stack.enter_context(patch('lyrics_shared.whisper_name_kwargs', return_value={}))
        stack.enter_context(patch('lyrics_shared.effective_names', return_value=[]))
        stack.enter_context(patch('socket.create_connection', side_effect=AssertionError('network forbidden')))
        yield path, model, af, ev, laugh


class CpuProfileTests(unittest.TestCase):
    def test_singer_keeps_loudness_pitch_and_recorded_affect_failure(self):
        import singer
        with card_fixture() as (path, _, affect, *_), patch.object(singer, '_free_gpu'):
            affect.side_effect = RuntimeError('missing affect model')
            lines, data, _ = singer.singer_block(path, 60)
        peak = next(line for line in lines if line.startswith('most intense moment:'))
        self.assertIsNone(data['peak']['arousal'])
        self.assertIsNone(data['peak']['note_arousal'])
        self.assertIn("dB louder than the voice's median", peak)
        self.assertIn('pitch there ~E3', peak)
        self.assertNotIn('arousal', peak)
        self.assertTrue(any(line.startswith('pitch: median') for line in lines))
        card = '\n'.join(ST.add_notes(['SINGER:'] + ['  ' + line for line in lines], data['stage_status']))
        self.assertIn('missing affect model', card)
        self.assertRegex(card, r'(?i)affect: failed')

    def test_song_cards_do_not_print_voice_events_but_keep_them(self):
        # 12 Nov 2026: on sung stems the speech event model said Snort x16, Burp x1.
        import singer
        heard = [dict(label='Snort', name='Snort', start=146.5, end=148.5, score=0.31),
                 dict(label='Burp', name='Burp', start=40.0, end=42.0, score=0.22)]
        peak = dict(ok=True, start=146., end=151., loud_db=-14., note_loud_db=-22.,
                    arousal=.92, note_arousal=.71)
        with card_fixture() as (path, _, _, ev, _), patch.object(singer, '_free_gpu'), \
             patch.object(R, 'analyze_peak', return_value=peak):
            ev.side_effect = lambda *_: dict(events=copy.deepcopy(heard), device='cpu', breath={})
            lines, data, _ = singer.singer_block(path, 249)
        text = '\n'.join(lines)
        self.assertNotIn('sounds the voice makes', text)
        self.assertNotIn('event there', text)
        self.assertNotIn('Snort', text)
        self.assertLessEqual({'Snort', 'Burp'}, {e['name'] for e in data['events']['events']})

    def test_singer_peak_with_affect_keeps_exact_card_bytes(self):
        import singer
        # The successful real singer peak's numeric and line shape, with a fake
        # front end from the existing speech fixture (no audio inference).
        peak = dict(ok=True, start=146., end=151., loud_db=-14., note_loud_db=-22.,
                    arousal=.92, note_arousal=.71)
        with card_fixture() as (path, *_), patch.object(singer, '_free_gpu'), \
             patch.object(R, 'analyze_peak', return_value=peak):
            lines, _, _ = singer.singer_block(path, 249)
        actual = next(line for line in lines if line.startswith('most intense moment:'))
        expected = ("most intense moment: 2:26–2:31 · 8 dB louder than the voice's median · "
                    "arousal 0.92 vs 0.71 average " + singer.MOOD_CAVEAT)
        self.assertEqual(actual.encode(), expected.encode())

    def test_profile_off_card_bytes(self):
        with card_fixture() as (path, model, *_):
            result = R.analyze(path, update_baseline=False)
            expected = (ROOT / 'fixtures/speech-default.card.txt').read_bytes()
            self.assertEqual(('\n'.join(result['lines']) + '\n').encode(), expected)
            self.assertEqual(model.call_args.args[0], 'large-v3')
            self.assertNotIn('cpu_threads', model.call_args.kwargs)
            self.assertEqual(result['lines'], R.analyze(path, fast=False, update_baseline=False)['lines'])

    def test_profile_and_model_overrides_reach_whisper(self):
        for env, fast, expected in (({}, False, 'large-v3'),
                                    ({'RAVEN_EARS_PROFILE': 'cpu'}, False, 'large-v3-turbo'),
                                    ({'RAVEN_EARS_PROFILE': 'cpu'}, True, 'small'),
                                    ({'RAVEN_EARS_PROFILE': 'cpu', 'WHISPER_MODEL': 'medium'}, False, 'medium'),
                                    ({'WHISPER_MODEL': 'medium'}, True, 'medium'),
                                    ({'RAVEN_EARS_PROFILE': 'cpu', 'WHISPER_MODEL': 'base',
                                      'RAVEN_EARS_WHISPER': 'large-v3-turbo'}, True, 'large-v3-turbo')):
            with self.subTest(env=env, fast=fast), card_fixture() as (path, model, *_), patch.dict(os.environ, env):
                R.analyze(path, fast=fast, update_baseline=False)
                self.assertEqual(model.call_args.args[0], expected)
                if fast or env.get('RAVEN_EARS_PROFILE'):
                    self.assertLessEqual(model.call_args.kwargs['cpu_threads'], 4)

    def test_fast_card_keeps_words_and_stages_and_explains_affect(self):
        with card_fixture() as (path, model, affect, events, laughs):
            before = dict(os.environ)
            result = R.analyze(path, fast=True, update_baseline=False)
            self.assertEqual(dict(os.environ), before)
            affect.assert_not_called()
            events.assert_called_once()
            laughs.assert_called_once()
            text = '\n'.join(result['lines'])
            for line in ('fast mode: whisper small; affect skipped; threads <=4',
                         '  AFFECT:  skipped: --fast', '  PITCH:', '  PACE:',
                         '  WORDLESS:', 'SILENT:', '  WORDS:', 'Check the signal.', '[0:12] Another channel opens.'):
                self.assertIn(line, text)
            self.assertIn(dict(stage='affect', status='skipped', kind='audio', reason='--fast'), result['stage_status'])
            self.assertTrue(result['data']['words'])
            self.assertIsNone(result['data']['peak']['arousal'])
            self.assertNotIn('arousal', next(line for line in result['lines'] if 'PEAK:' in line))

    def test_fast_supplied_words_disabled_whisper_and_silence_are_honest(self):
        for kwargs, reason in (({'words': [('Calibration', .1, .4)]}, 'supplied word times'),
                               ({'use_whisper': False}, 'Whisper disabled (--no-whisper)')):
            with card_fixture() as (path, model, *_):
                result = R.analyze(path, fast=True, update_baseline=False, **kwargs)
                model.assert_not_called()
                self.assertIn('fast mode: ' + reason, '\n'.join(result['lines']))
        with card_fixture() as (path, model, *_), patch.object(R, 'load_audio', return_value=np.zeros(32000)):
            text = '\n'.join(R.analyze(path, fast=True)['lines'])
            self.assertIn('fast mode:', text)
            self.assertIn('digital silence', text)
            model.assert_not_called()

    def test_fast_cli_and_json_keep_status(self):
        with card_fixture() as (path, model, *_):
            for json_flag in ([], ['--json']):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(R.main([str(path), '--fast', '--quiet', '--no-baseline-update', *json_flag]), 0)
                if json_flag:
                    data = json.loads(output.getvalue())
                    self.assertTrue(any(row['stage'] == 'fast mode' for row in data['stage_status']))
                else:
                    self.assertIn('fast mode: whisper small; affect skipped', output.getvalue())

    def test_resident_whisper_reuses_weights_and_matches_one_shot(self):
        with card_fixture() as (path, model, *_), patch.dict(R._WHISPER_MODELS, {}, clear=True):
            one = R.analyze(path, fast=True, update_baseline=False)
            first = R.analyze(path, fast=True, keep_models=True, update_baseline=False)
            second = R.analyze(path, fast=True, keep_models=True, update_baseline=False)
            self.assertEqual(model.call_count, 2)  # One-shot plus one resident load.
            self.assertEqual(one['lines'], first['lines'])
            self.assertEqual(first['lines'], second['lines'])

    def test_failed_resident_whisper_is_evicted_and_next_clip_retries(self):
        with card_fixture() as (_, model, *_), patch.dict(R._WHISPER_MODELS, {}, clear=True):
            good = model.return_value
            broken = SimpleNamespace(transcribe=Mock(side_effect=RuntimeError('fixture failure')))
            model.side_effect = [broken, good]
            audio = np.zeros(R.SR, np.float32)
            self.assertIsNone(R._whisper_words(audio, [], fast=True, keep_model=True))
            self.assertFalse(R._WHISPER_MODELS)
            self.assertTrue(R._whisper_words(audio, [], fast=True, keep_model=True))
            self.assertEqual(model.call_count, 2)

    def test_fast_laughs_receive_events_instance_before_release(self):
        with card_fixture() as (path, _, _, events, laughs), patch.dict(R._MODELS, {}, clear=True):
            tagger = (Mock(), ['Laughter'])
            event_result = events.side_effect()
            def event_pass(*_):
                R._MODELS[('tagger', 'cpu')] = tagger
                return copy.deepcopy(event_result)
            def load(device):
                return R._MODELS[('tagger', device)]
            events.side_effect = event_pass
            with patch.object(R, '_load_tagger', side_effect=load):
                R.analyze(path, fast=True, update_baseline=False)
            self.assertIs(laughs.call_args.kwargs['tagger'], tagger)
            R.release_models.assert_called()  # Release after the finer pass.

    def test_serve_recovers_from_bad_path_and_keeps_protocol_and_status_isolated(self):
        with card_fixture() as (path, _, _, events, _), patch.dict(R._WHISPER_MODELS, {}, clear=True):
            expected = R.analyze(path, fast=True, update_baseline=False)['lines']
            original = events.side_effect
            def noisy_events(*args):
                print('third-party loader diagnostic')
                return original(*args)
            events.side_effect = noisy_events
            output, diagnostic = io.StringIO(), io.StringIO()
            from contextlib import redirect_stderr
            with redirect_stdout(output), redirect_stderr(diagnostic):
                status = R.serve(['\n', str(path) + '\n', str(path) + '.missing\n', str(path) + '\n'],
                                 update_baseline=False)
            self.assertEqual(status, 0)
            responses = [json.loads(row) for row in output.getvalue().splitlines()]
            self.assertEqual([r['ok'] for r in responses], [True, False, True])
            self.assertEqual(responses[0]['lines'], expected)
            self.assertEqual(responses[2]['lines'], expected)
            self.assertIn('third-party loader diagnostic', diagnostic.getvalue())
            R.release_models.assert_called()

    def test_serve_cli_implies_fast_and_json_keeps_data(self):
        with card_fixture() as (path, *_), patch.dict(R._WHISPER_MODELS, {}, clear=True), \
             patch.object(sys, 'stdin', io.StringIO(str(path) + '\n')):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(R.main(['--serve', '--json', '--no-baseline-update']), 0)
            response = json.loads(output.getvalue())
            self.assertTrue(response['ok'])
            self.assertIn('fast mode', [r['stage'] for r in response['data']['stage_status']])

    def test_cpu_bootstrap_caps_pools_and_preserves_smaller_limits(self):
        code = '''import os, json
import ears_profile
import torch
print(json.dumps([os.environ['RAVEN_EARS_DEVICE'],
    [os.environ[k] for k in ears_profile.THREAD_VARS], torch.get_num_threads(), torch.get_num_interop_threads()]))
'''
        env = {**os.environ, 'RAVEN_EARS_PROFILE': 'cpu', 'RAVEN_EARS_DEVICE': 'cuda',
               'OMP_NUM_THREADS': '16', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '8', 'NUMBA_NUM_THREADS': '8'}
        result = subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), ['cpu', ['4', '2', '4', '4'], 4, 1])

    def test_fast_torch_threads_are_restored_after_failure(self):
        torch = Mock()
        torch.get_num_threads.return_value = 16
        with patch.object(EP, '_torch', return_value=torch), patch.dict(os.environ, {'OMP_NUM_THREADS': '16'}):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with EP._fast_threads():
                    raise RuntimeError('fixture')
            self.assertEqual([call.args[0] for call in torch.set_num_threads.call_args_list], [4, 16])
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '16')

    def test_lyrics_model_never_shrinks(self):
        import lyrics_shared as LS
        model = Mock()
        with patch.dict(os.environ, {'RAVEN_EARS_PROFILE': 'cpu', 'RAVEN_EARS_WHISPER': 'small'}), \
             patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=model)}):
            LS._load_model(R, [])
        self.assertEqual(model.call_args.args[0], 'large-v3')

    def test_lyrics_override_reaches_loader_and_diagnostics(self):
        import lyrics_shared as LS
        model = Mock(side_effect=RuntimeError('fake load failure'))
        notes = []
        with patch.dict(os.environ, {'RAVEN_LYRICS_WHISPER': 'large-v3-turbo'}), \
             patch.object(R, '_pick_device', return_value='cpu'), \
             patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=model)}):
            self.assertEqual(LS._load_model(R, notes), (None, None, None))
            self.assertEqual(model.call_args.args[0], 'large-v3-turbo')
            self.assertIn('large-v3-turbo on cpu FAILED', notes[0])
            self.assertEqual(LS._transcription_note('cpu', 'int8'),
                             'lyrics: whisper large-v3-turbo on cpu (int8), no VAD')

    def test_music_lyric_label_uses_recorded_model_not_current_environment(self):
        import music
        data = dict(file='invented.mp3', duration=60., lyrics=[dict(t=3., text='Paper moons')],
                    notes=['lyrics: whisper large-v3-turbo on cpu (int8), no VAD'])
        with patch.dict(os.environ, {'RAVEN_LYRICS_WHISPER': 'large-v3'}):
            self.assertIn('whisper large-v3-turbo, rough times', music.format_card(data))
        data.pop('notes')
        with patch.dict(os.environ, {'RAVEN_LYRICS_WHISPER': 'large-v3-turbo'}):
            self.assertIn('whisper large-v3, rough times', music.format_card(data))

    def test_lyrics_cache_separates_models_devices_and_precision(self):
        import lyrics_shared as LS
        with tempfile.TemporaryDirectory() as directory, patch.object(LS, 'CACHE', directory), \
             patch.object(LS, 'effective_names', return_value=[]):
            audio = Path(directory) / 'invented.wav'
            audio.write_bytes(b'not audio; cache identity only')
            def transcribe(path, notes):
                notes.append(LS._transcription_note(*LS._model_attempts(R)[0]))
                return [(LS.effective_model(), 10., 10.5)]
            paths = []
            with patch.object(LS, 'transcribe', side_effect=transcribe) as heard:
                for model in ('large-v3', 'large-v3-turbo'):
                    for attempts in ([('cpu', 'int8')], [('cuda', 'float16')], [('cuda', 'int8_float16')]):
                        with patch.dict(os.environ, {'RAVEN_LYRICS_WHISPER': model}), \
                             patch.object(LS, '_model_attempts', return_value=attempts):
                            paths.append(LS.cache_file(audio))
                            self.assertEqual(LS.get_words(audio), [(model, 10., 10.5)])
                            notes = []
                            self.assertEqual(LS.get_words(audio, notes), [(model, 10., 10.5)])
                            self.assertTrue(any('[shared cache]' in n for n in notes))
                self.assertEqual(heard.call_count, 6)
            self.assertEqual(len(set(paths)), 6)

    def test_lyrics_fallback_does_not_poison_preferred_precision_cache(self):
        import lyrics_shared as LS
        with tempfile.TemporaryDirectory() as directory, patch.object(LS, 'CACHE', directory), \
             patch.object(LS, '_model_attempts', return_value=[('cuda', 'float16'), ('cpu', 'int8')]):
            audio = Path(directory) / 'invented.wav'
            audio.write_bytes(b'not audio; fallback cache only')
            def fallback(path, notes):
                notes.append(LS._transcription_note('cpu', 'int8'))
                return [('paper', 10., 10.5)]
            with patch.object(LS, 'transcribe', side_effect=fallback) as heard:
                LS.get_words(audio)
                LS.get_words(audio)
                self.assertEqual(heard.call_count, 2)
            self.assertFalse(Path(LS.cache_file(audio)).exists())

    def test_fresh_stem_choice_provenance_cache_and_six_stem_model(self):
        import singer
        import soundfile
        import torch
        for profile, six, local, expected in (('', False, False, 'htdemucs_ft'),
                                              ('cpu', False, False, 'htdemucs'),
                                              ('cpu', True, False, 'htdemucs_6s'),
                                              ('', False, True, 'htdemucs_ft')):
            with self.subTest(profile=profile, six=six, local=local), tempfile.TemporaryDirectory() as directory:
                sources = singer.SRC6 if six else singer.SRC
                model = Mock(sources=sources)
                model.to.return_value = model
                model.eval.return_value = model
                get_model = Mock(return_value=model, side_effect=AssertionError('unexpected remote resolver') if local else None)
                local_model = Mock(return_value=model)
                apply_model = Mock(return_value=torch.zeros(1, len(sources), 2, 800))
                modules = {'demucs': SimpleNamespace(),
                           'demucs.pretrained': SimpleNamespace(get_model=get_model),
                           'demucs.apply': SimpleNamespace(apply_model=apply_model)}
                @ST.engine
                def run():
                    stem_dir, cached = singer.get_stems('invented.wav', {}, six=six)
                    return dict(stem_dir=stem_dir, cached=cached)
                with patch.dict(os.environ, {'RAVEN_EARS_PROFILE': profile, 'RAVEN_EARS_DEVICE': 'cpu'}), \
                     patch.dict(sys.modules, modules), patch.object(singer, 'CACHE', directory), \
                     patch.object(singer, 'file_hash', return_value='invented'), \
                     patch.object(singer, 'offline', return_value=local), \
                     patch.object(singer, '_local_stem_model', local_model), patch.object(singer, '_free_gpu'), \
                     patch.object(singer.subprocess, 'run', return_value=SimpleNamespace(stdout=np.ones((800, 2), np.float32).tobytes())):
                    result = run()
                    self.assertFalse(result['cached'])
                    if local:
                        get_model.assert_not_called()
                        local_model.assert_called_once_with(expected, torch=torch)
                    else:
                        get_model.assert_called_once_with(expected)
                        local_model.assert_not_called()
                    self.assertEqual((Path(result['stem_dir']) / 'model.txt').read_text().strip(), expected)
                    if expected == 'htdemucs':
                        self.assertIn('htdemucs instead of htdemucs_ft', ST.summary(result['stage_status']))
                        cached = run()
                        self.assertTrue(cached['cached'])
                        self.assertIn('CPU profile chose htdemucs instead of htdemucs_ft', ST.summary(cached['stage_status']))
                        self.assertNotIn('cached stems', ST.summary(result['stage_status']))
                        self.assertEqual(ST.summary(ST.merge(result['stage_status'], cached['stage_status'])).count('CPU profile chose'), 1)
                        # The default profile upgrades the cache and its model marker.
                        with patch.dict(os.environ, {'RAVEN_EARS_PROFILE': ''}):
                            upgraded = run()
                            self.assertFalse(upgraded['cached'])
                            self.assertTrue(run()['cached'])
                        self.assertEqual(get_model.call_count, 2)
                        get_model.assert_called_with('htdemucs_ft')
                        self.assertEqual((Path(upgraded['stem_dir']) / 'model.txt').read_text().strip(), 'htdemucs_ft')
                        self.assertNotIn('degraded', ST.summary(upgraded['stage_status']))

    def test_optional_quality_is_a_note_but_installed_backend_failure_is_degraded(self):
        # Use the same frame layout as card_fixture's saved-card-derived speech front end.
        n = 600
        fe = dict(active=np.ones(n, bool), voiced=np.ones(n, bool), f0=np.full(n, 163.8),
                  times=np.arange(n) / R.FPS, rms_db=np.full(n, -22.))
        vector = np.full(n, 12.)
        for parselmouth, state in ((None, 'ran'), (Mock(Sound=Mock(side_effect=RuntimeError('broken backend'))), 'degraded')):
            with self.subTest(state=state), patch.object(R, '_parselmouth', return_value=(parselmouth, None)), \
                 patch.object(R, '_frame_spectra', return_value=(vector, vector)), \
                 patch.object(R, '_np_hnr_cpp', return_value=(vector, vector)):
                result = R.analyze_quality(np.zeros(R.SR * 10), fe, [])
            self.assertTrue(result['ok'])
            self.assertEqual(result['engine'], 'numpy (approximate)')
            self.assertEqual(result['stage_status'][0]['status'], state)
            text = '\n'.join(ST.add_notes([R.quality_line(result)], result['stage_status']))
            self.assertIn('numpy approximations', text)
            if state == 'ran':
                self.assertIn('optional praat-parselmouth', text)
                self.assertNotIn('degraded', text)

    def test_stem_cache_profile_matrix(self):
        import singer
        for profile in ('', 'cpu'):
            for six in (False, True):
                for marker in (None, 'htdemucs', 'htdemucs_ft', 'htdemucs_6s'):
                    for complete in (False, True):
                        with self.subTest(profile=profile, six=six, marker=marker, complete=complete), \
                             tempfile.TemporaryDirectory() as directory:
                            stem_dir = Path(directory) / ('invented-6s' if six else 'invented')
                            stem_dir.mkdir()
                            for source in (singer.SRC6 if six else singer.SRC):
                                if complete or source != 'vocals':
                                    (stem_dir / (source + '.wav')).touch()
                            if marker:
                                (stem_dir / 'model.txt').write_text(marker + '\n')
                            reuse = complete and (six or profile == 'cpu' or marker != 'htdemucs')
                            stop = Mock(side_effect=RuntimeError('fake separation boundary'))
                            modules = {'demucs': SimpleNamespace(), 'demucs.apply': SimpleNamespace(apply_model=Mock()),
                                       'demucs.pretrained': SimpleNamespace(get_model=stop)}
                            with patch.dict(os.environ, {'RAVEN_EARS_PROFILE': profile}), \
                                 patch.dict(sys.modules, modules), \
                                 patch.object(singer, 'CACHE', directory), \
                                 patch.object(singer, 'file_hash', return_value='invented'):
                                if reuse:
                                    self.assertEqual(singer.get_stems('invented.wav', {}, six=six), (str(stem_dir), True))
                                    stop.assert_not_called()
                                else:
                                    with self.assertRaisesRegex(RuntimeError, 'fake separation boundary'):
                                        singer.get_stems('invented.wav', {}, six=six)
                                    stop.assert_called()

    def test_profile_off_bootstrap_does_not_touch_environment_or_import_torch(self):
        code = '''import os, sys
before = dict(os.environ)
import ears_profile
assert dict(os.environ) == before
assert 'torch' not in sys.modules
'''
        env = {k: v for k, v in os.environ.items() if k != 'RAVEN_EARS_PROFILE'}
        subprocess.run([sys.executable, '-c', code], env=env, cwd=ROOT, check=True)


def run_checks(check):
    for test in unittest.defaultTestLoader.loadTestsFromTestCase(CpuProfileTests):
        result = unittest.TestResult()
        test.run(result)
        check('CPU/fast: ' + test._testMethodName.removeprefix('test_').replace('_', ' '),
              result.wasSuccessful(), '\n'.join(message for _, message in result.errors + result.failures))


if __name__ == '__main__':
    unittest.main()
