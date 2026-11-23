"""Setup/offline contract: temporary homes, fake downloads, no real model loads."""
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch  # Load native modules before fresh_home patches sys.modules.
import raven_ears_plus as R
import room_laugh
import singer
import model_compat
from test_cpu_profile import card_fixture

BEAT_TEST_BYTES = b'fake checkpoint, never loaded'


@contextmanager
def fresh_home():
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        home = Path(directory)
        env = {k: v for k, v in os.environ.items() if k not in (
            'RAVEN_EARS_OFFLINE', 'RAVEN_EARS_PROFILE', 'RAVEN_EARS_WHISPER', 'WHISPER_MODEL',
            'RAVEN_BEAT_THIS_SHA256', 'RAVEN_BEAT_THIS_SIZE')}
        stack.enter_context(patch.dict(os.environ, dict(env, HOME=str(home)), clear=True))
        stack.enter_context(patch.object(Path, 'home', return_value=home))
        stack.enter_context(patch.dict(os.environ, {'TORCH_HOME': str(home / 'torch')}))
        for name, path in (('PANNS_CKPT', home / 'panns/Cnn14.pth'),
                           ('PANNS_LABELS', home / 'panns/class_labels_indices.csv')):
            stack.enter_context(patch.object(R, name, path))
        labels = list(dict.fromkeys([*R.EVENT_LABELS, 'Snicker']))
        labels += [f'Class {i}' for i in range(527 - len(labels))]
        def retrieve(url, dest):
            if url == R.PANNS_CKPT_URL:
                with Path(dest).open('wb') as f:
                    f.truncate(300_000_000)  # Sparse placeholder, never passed to torch.load.
            else:
                with Path(dest).open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['index', 'mid', 'display_name'])
                    writer.writerows((i, f'/m/fake_{i}', label) for i, label in enumerate(labels))
        download = stack.enter_context(patch('urllib.request.urlretrieve', side_effect=retrieve))
        stack.enter_context(patch.object(singer, 'BEAT_CHECKPOINT_SHA256', hashlib.sha256(BEAT_TEST_BYTES).hexdigest()))
        stack.enter_context(patch.object(singer, 'BEAT_CHECKPOINT_SIZE', len(BEAT_TEST_BYTES)))
        download.beat_opener = stack.enter_context(patch('urllib.request.urlopen',
            side_effect=lambda *args, **kwargs: io.BytesIO(BEAT_TEST_BYTES)))
        def snapshot(model, *, local_files_only, allow_patterns=None):
            root = home / 'hf' / model.replace('/', '--')
            if not local_files_only:
                root.mkdir(parents=True, exist_ok=True)
                for name in ('config.json', 'pytorch_model.bin', 'tokenizer.json'):
                    (root / name).write_bytes(b'fake')
            if not root.is_dir():
                raise FileNotFoundError('no cached snapshot')
            return str(root)
        affect = Mock(side_effect=snapshot)
        def whisper_snapshot(size, *, local_files_only):
            root = home / 'whisper' / size
            if not local_files_only:
                root.mkdir(parents=True, exist_ok=True)
                for name in ('config.json', 'model.bin', 'tokenizer.json', 'vocabulary.json'):
                    (root / name).write_bytes(b'fake')
            if not root.is_dir():
                raise FileNotFoundError('no cached Whisper snapshot')
            return str(root)
        whisper = Mock(side_effect=whisper_snapshot)
        stack.enter_context(patch.dict(sys.modules, {
            'huggingface_hub': SimpleNamespace(snapshot_download=affect),
            'faster_whisper.utils': SimpleNamespace(download_model=whisper)}))
        stack.enter_context(patch('socket.create_connection', side_effect=AssertionError('network forbidden')))
        yield home, download, whisper, affect


class SetupTests(unittest.TestCase):
    def test_progress_precedes_download_and_is_flushed(self):
        with fresh_home() as (_, download, _, _):
            output = Mock(wraps=io.StringIO())
            original = download.side_effect
            def retrieve(url, dest):
                if url == R.PANNS_CKPT_URL:
                    self.assertIn('PANNs checkpoint (~330 MB)', output.getvalue())
                    self.assertIn('several minutes', output.getvalue())
                    self.assertTrue(output.flush.called)
                return original(url, dest)
            download.side_effect = retrieve
            with redirect_stdout(output):
                self.assertTrue(R.setup('speech'))

    def test_music_fetches_beat_checkpoint_once_and_check_never_downloads(self):
        with fresh_home() as (_, download, _, _), redirect_stderr(io.StringIO()) as output:
            self.assertEqual(self.run_setup('music')[0], 0)
            download.beat_opener.assert_called_once_with(singer.BEAT_CHECKPOINT_URL, timeout=60)
            self.assertIn('Beat This! checkpoint from official server, sha256 verified', output.getvalue())
            download.beat_opener.reset_mock()
            download.reset_mock()
            self.assertEqual(self.run_setup('music', '--check')[0], 0)
            self.assertEqual(self.run_setup('music')[0], 0)
            download.assert_not_called()
            download.beat_opener.assert_not_called()
            singer.setup_beat_checkpoint(check=True).unlink()
            status, text = self.run_setup('music', '--check')
            self.assertEqual(status, 1)
            self.assertIn('Beat This! checkpoint missing', text)
            download.assert_not_called()

    def test_music_prepares_sentiment_and_checks_tokenizer_locally(self):
        import harmony

        with fresh_home() as (home, _, _, snapshots), redirect_stderr(io.StringIO()):
            status, text = self.run_setup('music')
            self.assertEqual(status, 0)
            self.assertLess(text.index('prepare RoBERTa lyric sentiment'), text.index('ok   RoBERTa lyric sentiment'))
            fetch = snapshots.call_args
            self.assertEqual(fetch.args, (harmony._SENTIMENT_MODEL,))
            self.assertFalse(fetch.kwargs['local_files_only'])
            self.assertIn('tokenizer.json', fetch.kwargs['allow_patterns'])
            tokenizer = home / 'hf' / harmony._SENTIMENT_MODEL.replace('/', '--') / 'tokenizer.json'
            tokenizer.unlink()
            status, text = self.run_setup('music', '--check')
            self.assertEqual(status, 1)
            self.assertIn('missing RoBERTa lyric sentiment', text)
            snapshots.assert_called_with(harmony._SENTIMENT_MODEL, local_files_only=True)

    def test_beat_http_error_is_visible_and_partial_download_is_removed(self):
        from urllib.error import HTTPError
        with fresh_home() as (_, download, _, _), redirect_stderr(io.StringIO()):
            def open_url(url, **kwargs):
                raise HTTPError(url, 500, 'Internal Server Error', {}, None)
            download.beat_opener.side_effect = open_url
            status, text = self.run_setup('music')
            self.assertEqual(status, 1)
            self.assertIn('checkpoint download failed', text)
            self.assertIn('HTTP Error 500', text)
            urls = [singer.BEAT_CHECKPOINT_URL, *(url for _, url in singer.BEAT_CHECKPOINT_MIRRORS)]
            self.assertEqual([call.args[0] for call in download.beat_opener.call_args_list], urls)
            for url in urls:
                self.assertIn(url, text)
            self.assertFalse(list(Path(os.environ['TORCH_HOME']).rglob('*.ckpt')))
            self.assertFalse(list(Path(os.environ['TORCH_HOME']).rglob('*.part')))

    def test_official_500_uses_first_mirror_and_reports_verified_source(self):
        from urllib.error import HTTPError
        with fresh_home() as (_, download, _, _), redirect_stderr(io.StringIO()) as output:
            download.beat_opener.side_effect = [
                HTTPError(singer.BEAT_CHECKPOINT_URL, 500, 'Internal Server Error', {}, None),
                io.BytesIO(BEAT_TEST_BYTES)]
            checkpoint = singer.setup_beat_checkpoint()
            self.assertEqual(checkpoint.read_bytes(), BEAT_TEST_BYTES)
            self.assertEqual([call.args[0] for call in download.beat_opener.call_args_list],
                             [singer.BEAT_CHECKPOINT_URL, singer.BEAT_CHECKPOINT_MIRRORS[0][1]])
            self.assertIn('Beat This! checkpoint from Hugging Face mirror (oulianov/beat_this), sha256 verified', output.getvalue())
            self.assertFalse(list(checkpoint.parent.glob('*.part')))

    def test_hash_mismatch_from_official_and_first_mirror_uses_second_mirror(self):
        with fresh_home() as (_, download, _, _), redirect_stderr(io.StringIO()) as output:
            wrong = b'x' * len(BEAT_TEST_BYTES)  # Correct size alone must never pass.
            download.beat_opener.side_effect = [io.BytesIO(wrong), io.BytesIO(wrong), io.BytesIO(BEAT_TEST_BYTES)]
            checkpoint = singer.setup_beat_checkpoint()
            self.assertEqual(checkpoint.read_bytes(), BEAT_TEST_BYTES)
            self.assertEqual(download.beat_opener.call_count, 3)
            self.assertIn('sha256 ' + hashlib.sha256(wrong).hexdigest(), output.getvalue())
            self.assertIn('RAVEN_BEAT_THIS_SHA256', output.getvalue())
            self.assertIn('Hugging Face mirror (teamup-tech/Beat-This-weights), sha256 verified', output.getvalue())

    def test_wrong_size_rejected_even_with_matching_hash(self):
        with fresh_home() as (_, download, _, _), redirect_stderr(io.StringIO()):
            wrong = BEAT_TEST_BYTES + b'extra'
            with patch.object(singer, 'BEAT_CHECKPOINT_SHA256', hashlib.sha256(wrong).hexdigest()):
                download.beat_opener.side_effect = lambda *args, **kwargs: io.BytesIO(wrong)
                with self.assertRaisesRegex(RuntimeError, 'verification failed: size'):
                    singer.setup_beat_checkpoint()
            root = Path(os.environ['TORCH_HOME'])
            self.assertFalse(list(root.rglob('*.ckpt')))
            self.assertFalse(list(root.rglob('*.part')))

    def test_interrupted_response_never_exposes_partial_file(self):
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError('connection interrupted')
                return super().read(4)
        with fresh_home() as (_, download, _, _), redirect_stderr(io.StringIO()):
            def open_url(*args, **kwargs):
                root = Path(os.environ['TORCH_HOME'])
                self.assertFalse(list(root.rglob('*.ckpt')))
                return Interrupted(BEAT_TEST_BYTES)
            download.beat_opener.side_effect = open_url
            with self.assertRaisesRegex(RuntimeError, 'connection interrupted'):
                singer.setup_beat_checkpoint()
            self.assertFalse(list(Path(os.environ['TORCH_HOME']).rglob('*.part')))

    def test_check_rejects_corrupt_cache_without_network_and_override_accepts_new_file(self):
        with fresh_home() as (_, download, _, _):
            checkpoint = singer.setup_beat_checkpoint()
            replacement = b'verified new upstream version'
            checkpoint.write_bytes(replacement)
            download.beat_opener.reset_mock()
            with self.assertRaisesRegex(ValueError, 'RAVEN_BEAT_THIS_SHA256'):
                singer.setup_beat_checkpoint(check=True)
            download.beat_opener.assert_not_called()
            with patch.dict(os.environ, {'RAVEN_BEAT_THIS_SHA256': hashlib.sha256(replacement).hexdigest(),
                                         'RAVEN_BEAT_THIS_SIZE': str(len(replacement))}):
                self.assertEqual(singer.setup_beat_checkpoint(check=True), checkpoint)
                checkpoint.unlink()
                download.beat_opener.side_effect = lambda *args, **kwargs: io.BytesIO(replacement)
                self.assertEqual(singer.setup_beat_checkpoint().read_bytes(), replacement)
            download.beat_opener.assert_called_once_with(singer.BEAT_CHECKPOINT_URL, timeout=60)

    def test_malformed_override_preserves_cache_without_downloading(self):
        with fresh_home() as (_, download, _, _):
            checkpoint = singer.setup_beat_checkpoint()
            download.beat_opener.reset_mock()
            with patch.dict(os.environ, {'RAVEN_BEAT_THIS_SHA256': 'typo'}):
                with self.assertRaisesRegex(ValueError, '64 hex digits'):
                    singer.setup_beat_checkpoint()
            self.assertEqual(checkpoint.read_bytes(), BEAT_TEST_BYTES)
            download.beat_opener.assert_not_called()

    def test_online_beats_load_prepared_checkpoint_directly(self):
        with fresh_home() as (_, download, _, _):
            checkpoint = singer.setup_beat_checkpoint()
            download.reset_mock()
            download.beat_opener.reset_mock()
            def tracker(*, checkpoint_path, device):
                self.assertFalse(checkpoint_path.closed)
                self.assertEqual(checkpoint_path.read(), checkpoint.read_bytes())
                self.assertEqual(device, 'cpu')
                return lambda y, sr: ([1, 2], [1])
            with patch.dict(os.environ, {'RAVEN_EARS_DEVICE': 'cpu'}), \
                 patch.dict(sys.modules, {'beat_this.inference': SimpleNamespace(Audio2Beats=tracker)}), \
                 patch.object(singer, '_free_gpu'), patch.object(R, 'load_audio', return_value=np.zeros(22050)):
                b, d, engine = singer.beats_downbeats('invented.wav', use_cache=False)
            self.assertEqual((b.tolist(), d.tolist(), engine), ([1, 2], [1], 'beat_this'))
            download.assert_not_called()
            download.beat_opener.assert_not_called()

    def run_setup(self, *args):
        output = io.StringIO()
        with redirect_stdout(output):
            status = R.main(['--setup', *args])
        return status, output.getvalue()

    def test_fresh_fast_setup_then_offline_fast_card_uses_prepared_files(self):
        # Reuse the numeric/word shapes taken from the real speech-4 card.
        events, laughs = R.analyze_events, room_laugh.detect
        with card_fixture() as (path, model, affect_stage, event_stage, laugh_stage), fresh_home() as (
                home, download, whisper, affect_download), ExitStack() as stack:
            self.assertEqual(self.run_setup('fast')[0], 0)
            self.assertEqual(download.call_count, 2)
            whisper.assert_called_once_with('small', local_files_only=False)
            affect_download.assert_not_called()
            self.assertTrue(R._setup_labels_valid(home / 'panns_data/class_labels_indices.csv'))
            import torch
            tagger = Mock(side_effect=lambda x: {'clipwise_output': torch.zeros((len(x), 527))})
            tagger.to.return_value = tagger
            tagger.eval.return_value = tagger
            cnn = Mock(return_value=tagger)
            # Model construction is fake; the real offline guards, label import,
            # event/laugh windowing, and card orchestration are exercised.
            real_import = __import__
            def local_import(name, *args, **kwargs):
                if name == 'panns_inference.models':
                    return SimpleNamespace(Cnn14=cnn)
                if name == 'panns_inference.config':
                    with (home / 'panns_data/class_labels_indices.csv').open() as f:
                        labels = [row[2] for row in list(csv.reader(f))[1:]]
                    config = SimpleNamespace(labels=labels, classes_num=len(labels))
                    sys.modules[name] = config
                    return config
                return real_import(name, *args, **kwargs)
            stack.enter_context(patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '1', 'RAVEN_EARS_DEVICE': 'cpu'}))
            stack.enter_context(patch.dict(sys.modules, {'panns_inference.config': None}))
            stack.enter_context(patch('builtins.__import__', side_effect=local_import))
            stack.enter_context(patch.object(R, '_torch', return_value=torch))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
            stack.enter_context(patch.object(torch, 'load', return_value={'model': {}}))
            stack.enter_context(patch.object(R, '_MODELS', {}))
            # card_fixture replaces the loader; restore it to test the actual guard.
            stack.enter_context(patch.object(R, '_load_tagger', wraps=LOAD_TAGGER))
            event_stage.side_effect = events
            laugh_stage.side_effect = laughs
            transcription = model.return_value
            def local_whisper(size, **kwargs):
                self.assertTrue(kwargs['local_files_only'])
                R._setup_whisper(size, True)
                return transcription
            model.side_effect = local_whisper
            result = R.analyze(path, fast=True, update_baseline=False)
            affect_stage.assert_not_called()
            affect_download.assert_not_called()
            self.assertEqual(download.call_count, 2)  # Nothing fetched on first card.
            cnn.assert_called_once()
            self.assertEqual(cnn.call_args.kwargs['classes_num'], 527)
            self.assertTrue(result['data']['words'])
            self.assertIsNotNone(result['data']['events'])
            self.assertIsNotNone(result['data']['laughs'])
            self.assertIn('Check the signal.', '\n'.join(result['lines']))
            self.assertNotIn('run --setup', '\n'.join(result['lines']))
            # A fresh process with one required file missing must keep its skips
            # and an actionable remedy, rather than claiming empty event results.
            R.PANNS_CKPT.unlink()
            R._MODELS.clear()
            missing = R.analyze(path, fast=True, update_baseline=False)
            self.assertIsNone(missing['data']['events'])
            self.assertIsNone(missing['data']['laughs'])
            self.assertIn('run --setup (PANNs checkpoint missing)', '\n'.join(missing['lines']))
            self.assertEqual(download.call_count, 2)

    def test_missing_upstream_labels_preflight_and_offline_guard_then_repair(self):
        with fresh_home() as (home, download, whisper, affect):
            self.assertEqual(self.run_setup('fast')[0], 0)
            upstream = home / 'panns_data/class_labels_indices.csv'
            upstream.unlink()
            with patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '1'}), \
                 patch.dict(sys.modules, {'panns_inference.config': None}), patch.object(R, '_MODELS', {}):
                status, text = self.run_setup('fast', '--check')
                self.assertEqual(status, 1)
                self.assertIn('PANNs labels missing or invalid', text)
                self.assertIn('run --setup fast while online', text)
                with self.assertRaisesRegex(RuntimeError, 'upstream labels missing'):
                    R._load_tagger('cpu')
            self.assertFalse(upstream.exists())  # Check-only never repairs files.
            self.assertEqual(download.call_count, 2)
            self.assertEqual(self.run_setup('fast')[0], 0)
            self.assertTrue(R._setup_labels_valid(upstream))

    def test_check_only_fresh_home_never_downloads_or_writes(self):
        with fresh_home() as (home, download, whisper, affect), \
             patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '1'}):
            status, text = self.run_setup('--check')
            self.assertEqual(status, 1)
            for name in ('PANNs checkpoint', 'PANNs labels', 'Whisper large-v3', 'AFFECT'):
                self.assertIn('missing ' + name, text)
            download.assert_not_called()
            whisper.assert_called_once_with('large-v3', local_files_only=True)
            affect.assert_called_once_with(R.AFFECT_MODEL, local_files_only=True)
            self.assertEqual(list(home.iterdir()), [])

    def test_targets_match_policy_and_preserve_default_affect(self):
        cases = [('fast', {}, 'small', False), ('speech', {}, 'large-v3', True),
                 ('speech', {'RAVEN_EARS_PROFILE': 'cpu'}, 'large-v3-turbo', True),
                 ('fast', {'RAVEN_EARS_PROFILE': 'cpu'}, 'small', False),
                 ('fast', {'WHISPER_MODEL': 'base', 'RAVEN_EARS_WHISPER': 'medium'}, 'medium', False),
                 ('speech', {'WHISPER_MODEL': 'base'}, 'base', True),
                 ('music', {'RAVEN_EARS_PROFILE': 'cpu', 'RAVEN_EARS_WHISPER': 'small'}, 'large-v3', True),
                 ('--fast', {}, 'small', False),
                 (None, {}, 'large-v3', True)]
        for target, env, size, uses_affect in cases:
            with self.subTest(target=target, env=env), fresh_home() as (_, _, whisper, affect), patch.dict(os.environ, env):
                args = [] if target is None else [target]
                status, text = self.run_setup(*args)
                self.assertEqual(status, 0)
                source = next((key for key in ('RAVEN_EARS_WHISPER', 'WHISPER_MODEL') if key in env), None)
                label = f'prepare Whisper {size}'
                if target != 'music' and source:
                    label += f' (from {source})'
                self.assertIn(label + ';', text)
                whisper.assert_called_once_with(size, local_files_only=False)
                self.assertEqual(affect.called, uses_affect)
                self.assertEqual(self.run_setup(*args, '--check')[0], 0)
                self.assertTrue(whisper.call_args.kwargs['local_files_only'])

    def test_missing_whisper_tokenizer_and_affect_weights_fail_preflight(self):
        with fresh_home() as (home, _, _, _):
            self.assertEqual(self.run_setup('speech')[0], 0)
            (home / 'whisper/large-v3/tokenizer.json').unlink()
            (home / 'hf' / R.AFFECT_MODEL.replace('/', '--') / 'pytorch_model.bin').unlink()
            status, text = self.run_setup('speech', '--check')
            self.assertEqual(status, 1)
            self.assertIn('tokenizer.json', text)
            self.assertIn('AFFECT config or weights missing', text)
            self.assertNotIn('setup files ready', text)

    def test_invalid_labels_and_truncated_download_do_not_claim_ready(self):
        with fresh_home() as (home, download, _, _):
            self.assertEqual(self.run_setup('fast')[0], 0)
            R.PANNS_LABELS.write_text('index,mid,display_name\n0,/m/fake,One class\n')
            upstream = home / 'panns_data/class_labels_indices.csv'
            upstream.write_text('<html>not labels</html>')
            self.assertEqual(self.run_setup('fast', '--check')[0], 1)
            self.assertEqual(self.run_setup('fast')[0], 0)
            self.assertTrue(R._setup_labels_valid(upstream))
            R.PANNS_CKPT.unlink()
            download.side_effect = lambda url, dest: Path(dest).write_bytes(b'truncated')
            status, text = self.run_setup('fast')
            self.assertEqual(status, 1)
            self.assertIn('download incomplete or invalid', text)
            self.assertFalse(R.PANNS_CKPT.exists())
            self.assertNotIn('setup files ready', text)
            # An HTTP error page is not a valid labels download either.
            R.PANNS_LABELS.unlink()
            status, text = self.run_setup('fast')
            self.assertEqual(status, 1)
            self.assertIn('failed PANNs labels', text)
            self.assertFalse(R.PANNS_LABELS.exists())

    def test_offline_setup_remains_blocked_and_check_needs_setup(self):
        with fresh_home() as (_, download, _, _), patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'RAVEN_EARS_OFFLINE'):
                R.setup('fast')
            download.assert_not_called()
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                R.main(['--check'])
            self.assertEqual(error.exception.code, 2)

    def test_offline_sentiment_uses_only_complete_local_snapshot(self):
        import harmony
        import stage_status as ST
        from huggingface_hub.utils import LocalEntryNotFoundError

        pipeline = Mock(return_value=lambda texts, **kw: [
            [dict(label='positive', score=.8), dict(label='negative', score=.2)] for _ in texts])

        @ST.engine
        def run():
            lift = np.array([-1., -1.])
            lines = [dict(t=0., end=1., text='The lantern makes me glad.')]
            return dict(rows=harmony.contrast(lines, lift, {'energy': lift}, 2.))

        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '1'}), \
             patch.object(harmony, '_SENT', None), \
             patch('huggingface_hub.snapshot_download', side_effect=LocalEntryNotFoundError('missing')) as snapshot, \
             patch.dict(sys.modules, {'transformers': SimpleNamespace(pipeline=pipeline)}):
            result = run()
            self.assertEqual(result['rows'], [])
            self.assertIn('sentiment: skipped: sentiment weights or tokenizer not cached',
                          ST.summary(result['stage_status']))
            snapshot.assert_called_once_with(harmony._SENTIMENT_MODEL, local_files_only=True)
            snapshot.side_effect = None
            snapshot.return_value = directory
            self.assertEqual(run()['rows'], [])
            pipeline.assert_not_called()
            for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
                (Path(directory) / name).write_bytes(b'fake file')
            self.assertAlmostEqual(run()['rows'][0]['words'], .6)
            pipeline.assert_called_once_with('text-classification', model=directory, tokenizer=directory,
                                             top_k=None, device=-1)


LOAD_TAGGER = R._load_tagger


class ModelCompatibilityTests(unittest.TestCase):
    def test_installed_requirements_checked_before_either_transformers_import(self):
        import harmony
        for torch, transformers, required, works in (
                ('2.5.1+cpu', '5.23.0', 'torch>=2.6; extra == "torch"', False),
                ('2.8.0+cpu', '5.23.0', 'torch>=2.6; extra == "torch"', True),
                ('2.2.2', '4.49.0', 'torch>=2.0; extra == "torch"', True)):
            with self.subTest(torch=torch, transformers=transformers), \
                 patch.object(model_compat.metadata, 'version', side_effect=lambda name: torch if name == 'torch' else transformers), \
                 patch.object(model_compat.metadata, 'requires', return_value=[required]), \
                 patch.object(R, '_MODELS', {}), patch.object(harmony, '_SENT', None), \
                 patch.object(harmony, 'offline', return_value=False):
                if works:
                    model_compat.check_transformers_torch()
                else:
                    with self.assertRaisesRegex(model_compat.ModelVersionError, 'your torch .* does not match transformers .*see INSTALL.md'):
                        R._load_affect('cpu')
                    with self.assertRaisesRegex(model_compat.ModelVersionError, 'your torch .* does not match transformers .*see INSTALL.md'):
                        harmony.sentiment(['invented words'])
                    with card_fixture() as (path, _, affect, *_):
                        affect.side_effect = lambda *_: R._load_affect('cpu')
                        text = '\n'.join(R.analyze(path, update_baseline=False)['lines'])
                    self.assertIn('your torch 2.5.1+cpu does not match transformers 5.23.0', text)
                    self.assertIn('see INSTALL.md', text)
                    self.assertNotIn('NameError', text)

    def test_platform_requirements_admit_documented_pairs_and_reject_old_linux_torch(self):
        from packaging.requirements import Requirement
        requirements = [Requirement(line) for line in Path('requirements-music.txt').read_text().splitlines()
                        if line and not line.startswith(('#', '-'))]
        for system, machine, versions in (
                ('linux', 'x86_64', {'torch': '2.8.0+cpu', 'torchaudio': '2.8.0+cpu', 'transformers': '5.23.0'}),
                ('darwin', 'arm64', {'torch': '2.8.0', 'torchaudio': '2.8.0', 'transformers': '5.23.0'}),
                ('darwin', 'x86_64', {'torch': '2.2.2', 'torchaudio': '2.2.2', 'transformers': '4.49.0'})):
            with self.subTest(system=system, machine=machine):
                selected = [r for r in requirements if r.name in versions and
                            (not r.marker or r.marker.evaluate({'sys_platform': system, 'platform_machine': machine}))]
                self.assertEqual(len(selected), 3)
                self.assertTrue(all(versions[r.name] in r.specifier for r in selected))
                for r in selected:
                    if system == 'linux' and r.name == 'torch':
                        self.assertNotIn('2.5.1+cpu', r.specifier)
                    if system == 'darwin' and machine == 'x86_64' and r.name == 'transformers':
                        self.assertNotIn('5.23.0', r.specifier)

if __name__ == '__main__':
    unittest.main()
