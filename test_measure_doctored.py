"""Analytic signals and fake filenames; no models, services or personal audio."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import doctored_audio as Q
from tools.measure_doctored import input_files, measure


class MeasurementTests(unittest.TestCase):
    sr = 32000

    def test_clean_sine_level_crest_and_frequency(self):
        t = np.arange(self.sr * 2) / self.sr
        result = measure(.25 * np.sin(2 * np.pi * 1000 * t), self.sr)
        self.assertAlmostEqual(result['rms_dbfs'], 20 * np.log10(.25 / np.sqrt(2)), places=6)
        self.assertAlmostEqual(result['crest_db'], 20 * np.log10(np.sqrt(2)), places=6)
        self.assertLess(result['bass_share'], 1e-20)
        self.assertEqual(result['full_scale_pct'], 0)
        self.assertEqual(result['near_runs'], 0)

    def test_clipped_sine_has_rail_runs_and_lower_crest(self):
        t = np.arange(self.sr * 2) / self.sr
        sine = np.sin(2 * np.pi * 100 * t)
        clean = measure(.5 * sine, self.sr)
        clipped = measure(np.clip(2 * sine, -1, 1), self.sr)
        self.assertGreater(clipped['full_scale_pct'], 65)
        self.assertLess(clipped['full_scale_pct'], 68)
        self.assertEqual(clipped['near_runs'], 400)
        self.assertLess(clipped['crest_db'], clean['crest_db'])

    def test_bass_shelved_noise_and_gain_invariance(self):
        noise = np.random.default_rng(41).normal(size=self.sr * 8)
        spectrum = np.fft.rfft(noise)
        frequency = np.fft.rfftfreq(len(noise), 1 / self.sr)
        spectrum[frequency < 150] *= 24
        shelved = np.fft.irfft(spectrum, n=len(noise))
        white = measure(noise * .01, self.sr)
        bass = measure(shelved * .01, self.sr)
        quieter = measure(shelved * .001, self.sr)
        self.assertLess(white['bass_share'], .012)
        self.assertGreater(bass['bass_share'], .8)
        self.assertAlmostEqual(bass['bass_share'], quieter['bass_share'], places=12)
        self.assertAlmostEqual(bass['crest_db'], quieter['crest_db'], places=12)
        self.assertAlmostEqual(bass['tilt_db_per_oct'], quieter['tilt_db_per_oct'], places=12)
        self.assertAlmostEqual(white['tilt_db_per_oct'], 0, delta=.15)
        self.assertIsNone(Q.warning(dict(status='ran', sample_rate=self.sr, **white)))
        self.assertIn('energy below 150 Hz', Q.warning(dict(status='ran', sample_rate=self.sr, **bass)))

    def test_clipping_alone_is_not_a_doctoring_rule(self):
        t = np.arange(self.sr * 2) / self.sr
        clipped = measure(np.clip(2 * np.sin(2 * np.pi * 1000 * t), -1, 1), self.sr)
        self.assertGreater(clipped['full_scale_pct'], 60)
        self.assertIsNone(Q.warning(dict(status='ran', sample_rate=self.sr, **clipped)))

    def test_threshold_missing_and_malformed_measurements(self):
        for share in (.79999, None, float('nan'), float('inf'), -1, 1.1, '0.95', True):
            self.assertIsNone(Q.warning(dict(status='ran', sample_rate=self.sr, bass_share=share)))
        for value in (None, {}, dict(status='failed', bass_share=.95),
                      dict(status='ran', sample_rate=22050, bass_share=.95)):
            self.assertIsNone(Q.warning(value))
        self.assertIsNotNone(Q.warning(dict(status='ran', sample_rate=self.sr, bass_share=.8)))

    def test_synthetic_wav_uses_real_ears_decoder_and_failure_is_visible(self):
        from scipy.io import wavfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invented.wav'
            t = np.arange(self.sr * 2) / self.sr
            wavfile.write(path, self.sr, (.25 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
            result = Q.inspect(path)
            self.assertEqual(result['status'], 'ran')
            self.assertAlmostEqual(result['crest_db'], 3.0103, places=3)
            self.assertIsNone(Q.warning(result))
        with patch('raven_ears_plus.load_audio', side_effect=RuntimeError('decoder failed')):
            failed = Q.inspect('invented.wav')
        self.assertEqual(failed, dict(status='failed', reason='RuntimeError'))
        self.assertIsNone(Q.warning(failed))

    def test_saved_real_bundle_all_views_only_gain_note(self):
        import listen
        song = json.loads((Path(__file__).parent / 'fixtures/first-listen-song.json').read_text())
        with patch('raven_ears_plus.load_audio', side_effect=AssertionError('saved views never decode')):
            for view in listen.VIEWS:
                before, _ = listen.render(song, view)
                quality = dict(status='ran', sample_rate=self.sr, bass_share=.9320565685)
                after, _ = listen.render(dict(song, audio_quality=quality), view)
                note = Q.annotate('', quality).splitlines()
                self.assertIn('93.2%', after)
                self.assertEqual([line for line in after.splitlines() if line not in note], before.splitlines())
                self.assertEqual(listen.render(dict(song, audio_quality=dict(quality, bass_share=.62368)), view)[0], before)

    def test_runs_split_at_sign_changes_and_include_endpoints(self):
        signal = np.zeros(self.sr)
        signal[:6] = [1, 1, 1, -1, -1, -1]
        signal[100:102] = 1.1  # overshoots count as full scale, but run is too short
        signal[-3:] = .995
        result = measure(signal, self.sr)
        self.assertEqual(result['near_runs'], 3)
        self.assertEqual(result['near_runs_per_s'], 3)
        self.assertAlmostEqual(result['full_scale_pct'], 100 * 8 / self.sr)
        self.assertAlmostEqual(result['near_scale_pct'], 100 * 11 / self.sr)

    def test_silence_and_invalid_input(self):
        result = measure(np.zeros(self.sr), self.sr)
        for key in ('rms_dbfs', 'crest_db', 'bass_share', 'tilt_db_per_oct'):
            self.assertIsNone(result[key])
        for signal, sr in (([], self.sr), ([np.nan] * self.sr, self.sr),
                           (np.zeros((self.sr, 2)), self.sr), (np.zeros(100), self.sr),
                           (np.zeros(self.sr), 16000)):
            with self.assertRaises(ValueError):
                measure(signal, sr)

    def test_shelf_is_nonrecursive_and_explicit_pair_is_first(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('alpha.mp3', 'beta.MP3', 'ignore.wav'):
                (root / name).touch()
            for name in ('stems', 'incoming', 'rots-opening'):
                (root / name).mkdir()
                (root / name / 'skip.mp3').touch()
            files = list(input_files([root / 'beta.MP3', root]))
            self.assertEqual([p.name for p in files], ['beta.MP3', 'alpha.mp3'])


if __name__ == '__main__':
    unittest.main()
