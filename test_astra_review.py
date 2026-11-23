"""Release-review regressions: synthetic audio, fake models, no remote services."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import harmony
import listen
import network_policy
import raven_ears_plus as R
import singer
import song_length
import stage_status as ST
from test_cpu_profile import card_fixture
from test_setup import fresh_home


class ReviewTests(unittest.TestCase):
    def test_song_affect_flag_forwarding_and_saved_options(self):
        args = listen.parser().parse_args(['invented.wav', '--no-affect'])
        self.assertIn('--no-affect', listen.pass_args('singer', args))
        self.assertNotIn('--no-affect', listen.pass_args('music', args))
        self.assertNotIn('--no-affect', listen.pass_args('harmony', args))
        with patch.dict(os.environ, {'RAVEN_EARS_NO_AFFECT': '1'}):
            opts = listen.options(args)
        self.assertTrue(opts['no_affect'])
        self.assertEqual(opts['environment']['RAVEN_EARS_NO_AFFECT'], '1')
        self.assertIn('--no-affect', listen.changed_analysis_flags(args, opts, {}))

    def test_singer_cli_passes_affect_switch(self):
        # Stop at the singer boundary, before any model or acoustic measurement.
        with patch.object(singer, 'get_stems', return_value=('stems', True)), \
             patch.object(singer.os.path, 'exists', return_value=True), \
             patch('soundfile.info', return_value=SimpleNamespace(duration=10)), \
             patch('song_extras.prepare', return_value=None), \
             patch('credits.configure_from_args'), patch('lrclib.configure_from_args'), \
             patch.object(singer, 'vocal_presence', return_value=(.8, 10)), \
             patch('lrclib.resolve', return_value={'words': []}), patch('lrclib.card_lines', return_value=[]), \
             patch.object(singer, 'singer_block', side_effect=RuntimeError('test boundary')) as block:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(singer.main(['invented.wav', '--no-affect']), 1)
        self.assertFalse(block.call_args.kwargs['affect'])

    def test_singer_block_passes_affect_switch_to_speech(self):
        with patch.object(R, 'analyze', side_effect=RuntimeError('test boundary')) as analyze:
            with self.assertRaisesRegex(RuntimeError, 'test boundary'):
                singer.singer_block('vocals.wav', 10, affect=False)
        self.assertFalse(analyze.call_args.kwargs['affect'])

    def test_affect_environment_blocks_inference_and_names_reason(self):
        with card_fixture(R) as (path, _, affect, _, _), \
             patch.dict(os.environ, {'RAVEN_EARS_NO_AFFECT': '1'}):
            result = R.analyze(str(path), identify_voice=False)
        affect.assert_not_called()
        self.assertIn('RAVEN_EARS_NO_AFFECT=1', '\n'.join(result['lines']))

    def test_direct_affect_environment_blocks_model_loading(self):
        with patch.dict(os.environ, {'RAVEN_EARS_NO_AFFECT': '1'}), \
             patch.object(R, '_affect_windows') as windows:
            result = R.analyze_affect(np.ones(16000), {}, 1, [])
        self.assertFalse(result['ok'])
        windows.assert_not_called()

    def test_affect_setup_cli_and_environment(self):
        for flag, env in ((['--no-affect'], '0'), ([], '1')):
            with self.subTest(flag=flag, env=env), fresh_home(), \
                 patch.dict(os.environ, {'RAVEN_EARS_NO_AFFECT': env}), \
                 patch.object(R, '_setup_affect') as affect, redirect_stdout(io.StringIO()):
                self.assertEqual(R.main(['--setup', 'speech', *flag]), 0)
                self.assertEqual(R.main(['--setup', 'speech', '--check', *flag]), 0)
            affect.assert_not_called()

    def test_speech_no_baseline_environment_prevents_append(self):
        with card_fixture(R) as (path, *_), \
             patch.object(R, 'baseline_path', return_value=path.parent / 'baseline.jsonl'), \
             patch.dict(os.environ, {'RAVEN_EARS_NO_BASELINE': '1'}):
            result = R.analyze(str(path), speaker='example', identify_voice=False)
            self.assertFalse((path.parent / 'baseline.jsonl').exists())
            self.assertIn('baseline_n_before', result['data'])
            with patch.dict(os.environ, {'RAVEN_EARS_NO_BASELINE': '0'}):
                R.analyze(str(path), speaker='example', identify_voice=False)
            self.assertEqual(len((path.parent / 'baseline.jsonl').read_text().splitlines()), 1)

    def test_short_audible_clip_is_not_silence(self):
        for seconds, level, expected in ((.4, .1, 'too short to analyse'), (.4, 0, 'digital silence'),
                                          (1., 0, 'digital silence')):
            with self.subTest(seconds=seconds, level=level), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'tone.wav'
                path.touch()
                y = np.full(int(seconds * R.SR_SED), level, np.float32)
                with patch.object(R, 'load_audio', return_value=y), patch.object(R, '_torch', return_value=None):
                    result = R.analyze(str(path), identify_voice=False)
                text = '\n'.join(result['lines'])
                self.assertIn(expected, text)
                if level:
                    self.assertNotIn('digital silence', text)

    def test_speech_names_disclose_count_only_when_passed(self):
        import lyrics_shared as LS
        for supports_prompt in (True, False):
            with self.subTest(supports_prompt=supports_prompt), tempfile.TemporaryDirectory() as directory:
                names = Path(directory) / 'names.txt'
                names.write_text('InventedOne\nInventedTwo\nInventedOne\n')
                def transcribe(y, *, initial_prompt=None, **kwargs):
                    return [], None
                if not supports_prompt:
                    def transcribe(y, **kwargs):
                        return [], None
                model = SimpleNamespace(transcribe=transcribe)
                with patch.object(LS, 'NAMES_FILE', str(names)), patch.object(LS, 'EXTRA_NAMES', []), \
                     patch.object(R, '_preload_cuda12'), patch.object(R, '_pick_device', return_value='cpu'), \
                     patch.object(R, '_torch', return_value=None), \
                     patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=Mock(return_value=model))}):
                    notes = []
                    R._whisper_words(np.zeros(16000), notes)
                count_notes = [n for n in notes if 'names list fed to Whisper' in n]
                self.assertEqual(count_notes, [ST.whisper_names_note(2)] if supports_prompt else [])
                self.assertNotIn('InventedOne', '\n'.join(notes))

    def test_protocol_inputs_rejected_before_decoder_or_probe(self):
        for value in ('https://example.invalid/a', 'file:///song.wav', 'custom://song',
                      'concat:https://example.invalid/a', 'pipe:0', '//server/share/a.wav'):
            with self.subTest(value=value), patch.object(R.subprocess, 'run') as run:
                for reader in (lambda: R.load_audio(value, 16000), lambda: R._source_sample_rate(value),
                               lambda: R.analyze(value), lambda: singer.get_stems(value, {})):
                    with self.assertRaisesRegex(ValueError, 'local file'):
                        reader()
                run.assert_not_called()
        self.assertEqual(network_policy.local_audio_path('C:\\audio\\tone.wav'), 'C:\\audio\\tone.wav')

    def test_audio_clis_reject_urls_before_any_subprocess(self):
        import credits
        import instruments
        import melody_abc
        import more_ears
        import music
        import stops
        import conversation_sheet
        import raven_listen_modes
        from tools import measure_doctored
        for main, argv in ((R.main, []), (singer.main, []), (music.main, []), (harmony.main, []),
                           (instruments.main, []), (credits.main, []), (more_ears.main, []),
                           (stops.main, []), (melody_abc.main, ['--vocals']),
                           (conversation_sheet.main, []), (raven_listen_modes.main, []),
                           (measure_doctored.main, [])):
            with self.subTest(main=main.__module__), redirect_stderr(io.StringIO()) as err, \
                 patch('subprocess.run') as run:
                with self.assertRaises(SystemExit) as error:
                    main([*argv, 'https://example.invalid/tone.wav'])
                self.assertEqual(error.exception.code, 2)
                self.assertIn('local file', err.getvalue())
                run.assert_not_called()
        with redirect_stderr(io.StringIO()) as err, patch.object(listen, 'home_context'):
            self.assertEqual(listen.main(['https://example.invalid/tone.wav']), 2)
        self.assertIn('local file', err.getvalue())

    def test_queue_rejects_protocols_in_targets_and_lists(self):
        import ears_queue
        with tempfile.TemporaryDirectory() as directory:
            listing = Path(directory) / 'songs.txt'
            listing.write_text('https://example.invalid/tone.wav\n')
            for targets in (['https://example.invalid/tone.wav'], [str(listing)]):
                with self.subTest(targets=targets), self.assertRaisesRegex(ValueError, 'local file'):
                    ears_queue.expand_targets(targets)

    def test_path_arguments_reject_schemes_before_path_normalization(self):
        import conversation_sheet
        import eval_laughs
        import runpy
        from tools import measure_doctored
        for main, argv in ((conversation_sheet.main, ['c://host/tone.wav']),
                           (measure_doctored.main, ['c://host/tone.wav'])):
            with self.subTest(main=main.__module__), redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit):
                    main(argv)
                self.assertIn('local file', err.getvalue())
        with patch.object(sys, 'argv', ['eval_laughs.py', '--clips', 'c://host/clips']), \
             redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit):
                eval_laughs.main()
            self.assertIn('local file', err.getvalue())
        with patch.object(sys, 'argv', ['madmom_key.py', 'https://example.invalid/a']), \
             patch.dict(sys.modules, {'madmom': None}), redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit):
                runpy.run_path(str(Path(__file__).parent / 'madmom_key.py'), run_name='__main__')
            self.assertIn('local file', err.getvalue())

    def test_missing_ffmpeg_does_not_fall_back_to_unrestricted_decoder(self):
        with patch.object(R, '_ffmpeg_bin', return_value=None), \
             patch('librosa.load') as fallback:
            with self.assertRaisesRegex(RuntimeError, 'FFmpeg is required'):
                R.load_audio('invented.wav', 16000)
        fallback.assert_not_called()

    def test_decode_and_probe_restrict_protocols_and_select_first_stream(self):
        with patch.object(R, '_ffmpeg_bin', return_value='ffmpeg'), \
             patch.object(R.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=np.ones(10, np.float32).tobytes())) as run:
            R.load_audio('invented.wav', 16000)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('-map') + 1], '0:a:0')
        self.assertEqual(command[command.index('-protocol_whitelist') + 1], 'file,pipe')
        with tempfile.NamedTemporaryFile(suffix='.m4a') as source, \
             patch.object(R.shutil, 'which', return_value='ffprobe'), \
             patch.object(R.subprocess, 'run', return_value=SimpleNamespace(stdout=b'{"streams":[{"sample_rate":"44100"}]}')) as run:
            self.assertEqual(R._source_sample_rate(source.name), 44100)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('-select_streams') + 1], 'a:0')
        self.assertEqual(command[command.index('-protocol_whitelist') + 1], 'file,pipe')

    def test_old_downbeat_failure_is_retried_without_rewriting_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = str(Path(directory) / 'features')
            legacy = Path(cp + '.downbeats.pkl')
            legacy.write_bytes(pickle.dumps(None))
            with patch.object(singer, 'beats_downbeats', side_effect=[RuntimeError('temporary'),
                    ([], np.arange(4.), 'beat_this')]) as tracker:
                self.assertIsNone(harmony.tracked_downbeats('invented.wav', cp))
                self.assertEqual(harmony.tracked_downbeats('invented.wav', cp), [0., 1., 2., 3.])
            self.assertEqual(tracker.call_count, 2)
            self.assertIsNone(pickle.loads(legacy.read_bytes()))
            self.assertEqual(list(Path(directory).iterdir()), [legacy])

    def test_downbeat_cache_retries_failures_and_checks_identity_and_provenance(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            cp = str(Path(directory) / 'feature')
            stages = [ST.outcome('beat tracker', 'ran', 'synthetic tracker')]
            success = {'data': (np.arange(8.), np.arange(4.), 'beat_this'), 'stage_status': stages}
            @ST.engine
            def run():
                return {'beats': harmony.tracked_downbeats('invented.wav', cp)}
            with patch.object(harmony, 'cache_path', return_value=cp), \
                 patch.object(torch.hub, 'get_dir', return_value=directory), \
                 patch.object(singer, 'offline', return_value=False), \
                 patch.object(singer, '_tracked_beats', side_effect=[RuntimeError('temporary'), success, success]) as tracker:
                self.assertIsNone(run()['beats'])
                self.assertFalse(Path(cp + '.beats.pkl').exists())
                self.assertEqual(run()['beats'], [0., 1., 2., 3.])
                cached = run()
                self.assertEqual(tracker.call_count, 2)
                self.assertIn(stages[0], cached['stage_status'])
                checkpoint = Path(directory) / 'checkpoints' / 'beat_this-final0.ckpt'
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b'fake weights, never loaded')
                self.assertEqual(run()['beats'], [0., 1., 2., 3.])
                self.assertEqual(tracker.call_count, 3)

    def test_feature_cache_omits_waveform_and_cache_hit_can_reload_melody(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = str(Path(directory) / 'features')
            features = dict(y=np.ones(22050), y_len=22050, rms_db=np.array([-20., -30.]))
            with patch.object(harmony, 'cache_path', return_value=cp), \
                 patch.object(harmony, 'compute_features', return_value=features), \
                 patch.object(harmony, 'gather_key_inputs', return_value={}), \
                 patch.object(harmony, 'resolve_home', return_value=[{'key': 'C'}]):
                harmony.key_claim_for('invented.wav')
                with open(cp + '.feat.pkl', 'rb') as cache:
                    self.assertNotIn('y', pickle.load(cache))
                with patch.object(harmony, 'gather_key_inputs', return_value={}) as gather:
                    harmony.key_claim_for('invented.wav')
                self.assertIsNone(gather.call_args.args[-1])
                harmony.key_claim_for('invented.wav', features=features)
                with open(cp + '.feat.pkl', 'rb') as cache:
                    self.assertNotIn('y', pickle.load(cache))
            self.assertIn('y', features)
            with patch.object(harmony, 'vocal_stem_path', return_value=None), \
                 patch.object(R, 'load_audio', return_value=np.ones(22050)) as decode, \
                 patch.object(harmony, 'compute_melody', return_value={'source': 'full mix'}):
                harmony.load_melody('invented.wav', cp)
                harmony.load_melody('invented.wav', cp)
            decode.assert_called_once_with('invented.wav', harmony.SR)

    def test_conflicting_lengths_are_distinct_from_no_evidence(self):
        rows = [{'source': 'MusicBrainz', 'duration_s': 180}, {'source': 'Discogs', 'duration_s': 240}]
        self.assertIsNone(song_length.finding(180, []))
        found = song_length.finding(180, rows)
        self.assertEqual(found['status'], 'conflict')
        text = song_length.annotate('MUSIC: invented', found)
        self.assertIn('studio lengths conflict', text)
        self.assertIn('MusicBrainz 3:00', text)
        self.assertIn('Discogs 4:00', text)
        self.assertNotIn('may be a video', text)
        # A majority is useful evidence but must not hide a disagreeing service.
        self.assertIn('conflict', song_length.annotate('', song_length.finding(180,
            rows + [{'source': 'LRCLIB', 'duration_s': 181}])))

    def test_saved_card_conflict_survives_every_view(self):
        song = listen.load(Path(__file__).parent / 'fixtures/first-listen-song.json')
        song['cards']['music']['json']['studio_durations'] = [
            {'source': 'MusicBrainz', 'duration_s': 180}, {'source': 'Discogs', 'duration_s': 240}]
        song_length.update_song(song)
        with patch('urllib.request.urlopen', side_effect=AssertionError('no requests')):
            for view in listen.VIEWS:
                text = listen.render(song, view)[0]
                self.assertEqual(text.count('studio lengths conflict'), 1, view)


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg unavailable')
class SyntheticDecodeTests(unittest.TestCase):
    def ffmpeg(self, *args):
        result = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '4',
                                 '-filter_threads', '4', *map(str, args)], capture_output=True)
        if result.returncode:
            self.skipTest('FFmpeg cannot generate synthetic fixture: ' + result.stderr.decode(errors='replace')[-200:])

    def test_m4a_original_envelope_uses_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'synthetic.m4a'
            self.ffmpeg('-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:a', 'aac', audio)
            with patch.object(singer, 'load_mono', side_effect=AssertionError('original must use FFmpeg')):
                envelope, fps = singer._env(str(audio))
            self.assertGreater(len(envelope), 40)
            self.assertTrue(np.isfinite(envelope).all())
            self.assertEqual(fps, 22050 / 256)

    def test_generated_stem_envelope_keeps_soundfile(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'drums.wav'
            sf.write(audio, np.sin(np.arange(22050) * .1), 22050)
            with patch.object(R, 'load_audio', side_effect=AssertionError('stem reader is SoundFile')):
                envelope, _ = singer._env(str(audio), stem=True)
            self.assertGreater(len(envelope), 40)

    def test_multistream_decode_and_separation_hear_first_stream(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'two-streams.mka'
            self.ffmpeg('-f', 'lavfi', '-i', 'sine=frequency=220:duration=0.4',
                        '-f', 'lavfi', '-i', 'sine=frequency=880:duration=0.4',
                        '-map', '0:a', '-map', '1:a', '-ac:a:0', '2', '-ac:a:1', '6', '-c:a', 'pcm_s16le', audio)
            def peak(y, sr):
                return np.argmax(abs(np.fft.rfft(y))) * sr / len(y)
            self.assertAlmostEqual(peak(R.load_audio(str(audio), 16000), 16000), 220, delta=3)
            model = Mock(sources=singer.SRC)
            model.to.return_value = model
            model.eval.return_value = model
            def apply(model, wav, **kwargs):
                self.assertAlmostEqual(peak(wav[0, 0].numpy(), singer.SR_ST), 220, delta=3)
                return torch.zeros(1, 4, 2, wav.shape[-1])
            with patch.object(singer, 'CACHE', str(Path(directory) / 'stems')), \
                 patch.object(singer, '_free_gpu'), patch.object(singer, 'offline', return_value=False), \
                 patch.dict(os.environ, {'RAVEN_EARS_DEVICE': 'cpu'}), \
                 patch.dict(sys.modules, {'demucs.apply': SimpleNamespace(apply_model=apply),
                     'demucs.pretrained': SimpleNamespace(get_model=Mock(return_value=model))}):
                stem_dir, cached = singer.get_stems(str(audio), {})
            self.assertFalse(cached)
            self.assertEqual(Path(stem_dir).name, hashlib.sha256(audio.read_bytes()).hexdigest()[:16] + '-a0')
            self.assertTrue(Path(stem_dir, 'vocals.wav').is_file())


if __name__ == '__main__':
    unittest.main()
