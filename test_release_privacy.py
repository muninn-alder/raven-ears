"""Release boundaries: no metadata requests offline, no private fixture dependence."""
import ast
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import credits
import lrclib
import network_policy


class ReleaseTests(unittest.TestCase):
    def test_allowlist_contains_local_import_closure(self):
        root = Path(__file__).resolve().parent
        allowed = set((root / 'RELEASE-ALLOWLIST.txt').read_text().splitlines())
        pending = [name for name in allowed if name.endswith('.py')]
        seen, missing = set(), set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            tree = ast.parse((root / name).read_text(encoding='utf-8'), filename=name)
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ''
                    if node.level:
                        package = Path(name).parent.parts
                        base = '.'.join((*package[:len(package) - node.level + 1], base)).strip('.')
                    modules.append(base)
                    modules.extend(base + '.' + alias.name for alias in node.names)
                elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                      and isinstance(node.args[0].value, str)
                      and (isinstance(node.func, ast.Name) and node.func.id == '__import__'
                           or isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module')):
                    modules.append(node.args[0].value)
            for module in modules:
                parts = module.split('.')
                for i in range(1, len(parts) + 1):
                    base = Path(*parts[:i])
                    for dependency in (base.with_suffix('.py'), base / '__init__.py'):
                        if not (root / dependency).is_file():
                            continue
                        local = dependency.as_posix()
                        if local not in allowed:
                            missing.add(f'{name} imports {local}')
                        pending.append(local)
        self.assertFalse(missing, '\n'.join(sorted(missing)))


class OfflineTests(unittest.TestCase):
    def test_offline_blocks_public_and_low_level_metadata_requests(self):
        with patch.dict(os.environ, {"RAVEN_EARS_OFFLINE": "1", "DISCOGS_TOKEN": "fake"}), \
             patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")) as request, \
             patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
            self.assertTrue(credits.get("missing.wav", "Example", "Calibration")["off"])
            self.assertEqual(lrclib.get("missing.wav", "Example", "Calibration")["status"], "off")
            # Also protect against a warm in-process cache after toggling offline.
            with patch.dict(credits._MEMO, {"missing.wav": {"entries": ["cached"]}}):
                self.assertTrue(credits.for_song("missing.wav")["off"])
            for call in (lambda: credits._get_json(credits.MB),
                         lambda: credits._discogs_json(credits.DISCOGS, "fake"),
                         lambda: lrclib._get_json("get", {})):
                with self.assertRaisesRegex(RuntimeError, "RAVEN_EARS_OFFLINE"):
                    call()
            request.assert_not_called()

    def test_offline_never_launches_external_resolvers(self):
        import ears_queue
        import raven_listen_modes
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"RAVEN_EARS_OFFLINE": "1"}):
            queue = ears_queue.Queue(Path(directory) / "state", Path(directory) / "cards")
            with patch.object(queue, "command", side_effect=AssertionError("external command forbidden")):
                with self.assertRaisesRegex(ears_queue.QueueError, "RAVEN_EARS_OFFLINE"):
                    queue.resolve({"target": "Example - Calibration"}, None)
            import contextlib
            import io
            with patch.object(raven_listen_modes.subprocess, "run", side_effect=AssertionError("external command forbidden")), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    raven_listen_modes.main(["Example - Calibration"])
                self.assertEqual(error.exception.code, 2)

    def test_switch_is_explicit_and_rechecked(self):
        for value, expected in (("1", True), ("true", True), ("0", False), ("", False)):
            with patch.dict(os.environ, {"RAVEN_EARS_OFFLINE": value}):
                self.assertEqual(network_policy.offline(), expected)

    def test_generated_fixture_has_enough_energy_runs(self):
        # Only PCM and the lightweight energy front end; no audio models or GPU.
        from scipy.io import wavfile
        import numpy as np
        import raven_ears_plus as engine
        import test_ears_plus as runner
        raw_sr, raw = wavfile.read(runner.DEFAULT_TTS)
        # The shipped fixture is full-band 44.1 kHz so the first card is not flagged narrowband.
        self.assertEqual(raw_sr, 44100)
        pcm = (engine.load_audio(runner.DEFAULT_TTS, engine.SR) * 32768).astype(np.float32)
        with patch("librosa.pyin", return_value=(np.zeros(len(pcm)//engine.HOP+1), np.zeros(len(pcm)//engine.HOP+1, dtype=bool), None)):
            front = engine._front_end(pcm.astype(np.float32) / 32768)
        runs = engine._merge_runs(engine._runs(front["active"]), int(0.12 * engine.FPS))
        self.assertEqual(sum((b-a)/engine.FPS >= .12 for a,b in runs), 32)
        # A missing fixture must also leave the full runner's fallback usable.
        pcm, _ = runner.synth_speech(30, np.random.default_rng(3))
        with patch("librosa.pyin", return_value=(np.zeros(len(pcm)//engine.HOP+1), np.zeros(len(pcm)//engine.HOP+1, dtype=bool), None)):
            front = engine._front_end(pcm)
        runs = engine._merge_runs(engine._runs(front["active"]), int(.12 * engine.FPS))
        self.assertGreaterEqual(sum((b-a)/engine.FPS >= .12 for a,b in runs), 8)

    def test_shipped_fixture_is_not_flagged_as_a_phone_call(self):
        # INSTALL.md's first card runs on this file; CHANNEL must stay quiet on it.
        import numpy as np
        import raven_ears_plus as engine
        import test_ears_plus as runner
        y16 = engine.load_audio(runner.DEFAULT_TTS, engine.SR)
        y32 = engine.load_audio(runner.DEFAULT_TTS, engine.SR_SED)
        with patch("librosa.pyin", return_value=(np.zeros(len(y16)//engine.HOP+1), np.zeros(len(y16)//engine.HOP+1, dtype=bool), None)):
            front = engine._front_end(y16)
        channel = engine.analyze_channel(y32, 44100, front)
        self.assertIsNone(channel["classification"], channel)
        self.assertIsNone(engine.channel_line(channel))

    def test_offline_blocks_explicit_model_download(self):
        import raven_ears_plus as engine
        import harmony
        with patch.dict(os.environ, {"RAVEN_EARS_OFFLINE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "RAVEN_EARS_OFFLINE"):
                engine._download("https://example.invalid", Path("unused"))
        # Model unavailability remains visible; it is never scored as neutral text.
        with patch.object(harmony, "sentiment", side_effect=RuntimeError("missing model")):
            self.assertEqual(harmony.contrast([{"text":"neutral","t":0,"end":1}], [], {}, 1), [])

    def test_offline_tagger_checks_checkpoint_and_upstream_labels_before_import(self):
        import raven_ears_plus as engine
        import sys
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            checkpoint = home / 'Cnn14.pth'
            model = Mock()
            model.to.return_value = model
            cnn = Mock(return_value=model)
            torch = SimpleNamespace(load=Mock(return_value={'model': {}}))
            with patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '1'}), \
                 patch.object(engine, '_MODELS', {}), patch.object(engine, 'PANNS_CKPT', checkpoint), \
                 patch.object(Path, 'home', return_value=home), patch.object(engine, '_torch', return_value=torch), \
                 patch.dict(sys.modules, {'panns_inference': None, 'panns_inference.config': None}):
                with self.assertRaisesRegex(FileNotFoundError, '--setup'):
                    engine._load_tagger('cpu')
                # Sparse placeholder, never loaded by a real model.
                with checkpoint.open('wb') as f:
                    f.truncate(300_000_000)
                with self.assertRaisesRegex(RuntimeError, 'cached weights not used offline.*RAVEN_EARS_OFFLINE'):
                    engine._load_tagger('cpu')
                cnn.assert_not_called()
                torch.load.assert_not_called()
                labels = home / 'panns_data' / 'class_labels_indices.csv'
                labels.parent.mkdir()
                labels.write_text('index,mid,display_name\n0,/test,test\n')
                real_import = __import__
                def local_import(name, *args, **kwargs):
                    if name == 'panns_inference.models':
                        return SimpleNamespace(Cnn14=cnn)
                    if name == 'panns_inference.config':
                        config = SimpleNamespace(labels=['test'], classes_num=1)
                        sys.modules[name] = config
                        return config
                    return real_import(name, *args, **kwargs)
                with patch('builtins.__import__', side_effect=local_import):
                    loaded, names = engine._load_tagger('cpu')
                    self.assertIs(loaded, model)
                    self.assertEqual(names, ['test'])
                    torch.load.assert_called_once_with(str(checkpoint), map_location='cpu', weights_only=False)
                    # Already imported labels do not trigger another download.
                    labels.unlink()
                    engine._MODELS.clear()
                    self.assertIs(engine._load_tagger('cpu')[0], model)
                    # A warm in-memory model does not need files at all.
                    checkpoint.unlink()
                    self.assertIs(engine._load_tagger('cpu')[0], model)

    def test_offline_beats_use_only_open_local_checkpoint_and_do_not_fall_back(self):
        import singer
        import sys
        import stage_status as S
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoints' / 'beat_this-final0.ckpt'
            torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: directory))
            def local_tracker(*, checkpoint_path, device):
                self.assertFalse(checkpoint_path.closed)
                self.assertEqual(checkpoint_path.read(), b'cached weights')
                self.assertEqual(device, 'cpu')
                return lambda y, sr: ([1, 2], [1])
            tracker = Mock(side_effect=local_tracker)
            @S.engine
            def run():
                try:
                    singer.beats_downbeats('neutral.wav')
                except (RuntimeError, ValueError):
                    pass
                return {}
            with patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '1', 'RAVEN_EARS_DEVICE': 'cpu'}), \
                 patch.dict(sys.modules, {'torch': torch, 'beat_this': SimpleNamespace(),
                     'beat_this.inference': SimpleNamespace(Audio2Beats=tracker),
                     'madmom': None}), patch.object(singer, '_free_gpu'), \
                 patch('raven_ears_plus.load_audio', return_value=[0.] * 22050), \
                 patch.object(singer, 'BEAT_CHECKPOINT_SIZE', len(b'cached weights')), \
                 patch.object(singer, 'BEAT_CHECKPOINT_SHA256', hashlib.sha256(b'cached weights').hexdigest()), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')), \
                 patch('socket.create_connection', side_effect=AssertionError('network forbidden')):
                self.assertIn('weights missing from Torch cache', S.summary(run()['stage_status']))
                tracker.assert_not_called()
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b'cached weights')
                beats, downbeats, backend = singer.beats_downbeats('neutral.wav')
                self.assertEqual((beats.tolist(), downbeats.tolist(), backend), ([1, 2], [1], 'beat_this'))
                tracker.side_effect = RuntimeError('corrupt local weights')
                self.assertIn('beat tracker: failed: RuntimeError: corrupt local weights',
                              S.summary(run()['stage_status']))
                tracker.reset_mock()
                checkpoint.write_bytes(b'wrong weights')
                self.assertIn('checkpoint verification failed', S.summary(run()['stage_status']))
                tracker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
