#!/usr/bin/env python3
"""Self-test for raven_ears_plus. Synthetic audio + (optionally) one TTS clip. Nothing leaves the machine.

    python test_ears_plus.py                      # full run (models used if installed)
    python test_ears_plus.py --tts /path/neutral.wav # speech source for the composite
    python test_ears_plus.py --quick              # skip the 11-run baseline seeding
    python test_ears_plus.py --fast               # no decoding, librosa/Praat or audio models

Speech source: generated neutral formant tones, or an explicitly supplied local clip. Words for the composite are derived from the clip's own energy runs, so the
test never depends on Whisper. Whisper-on-composite is run as an informational extra.
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import raven_ears_plus as R  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import lfilter  # noqa: E402

SR = 16000
DEFAULT_TTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "neutral-voiced.wav")  # generated tones; no recorded voice or words
RESULTS = []   # (name, status, detail)


def check(name, ok, detail=""):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def info(name, detail):
    RESULTS.append((name, "INFO", detail))
    print(f"  [info] {name}: {detail}")


# --------------------------------------------------------------------------- #
#  synthesis
# --------------------------------------------------------------------------- #
def _resonate(x, freqs_bws):
    for f, bw in freqs_bws:
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * f / SR
        x = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    return x


def vowel(dur, f0=140.0, contour=None, vib=0.012, kind="ah", level=0.25):
    """Sustained vowel / hum. contour(t_norm 0..1) -> semitone offset."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    st = contour(t / dur) if contour else 0.0
    f = f0 * 2 ** (np.asarray(st) / 12.0) * (1 + vib * np.sin(2 * np.pi * 5.2 * t))
    phase = np.cumsum(f) / SR
    pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    src = lfilter([1], [1, -0.95], pulses)
    form = {"ah": [(700, 110), (1200, 130), (2600, 200)],
            "mm": [(250, 80), (1100, 250), (2200, 300)]}[kind]
    y = _resonate(src, form)
    env = np.minimum(1, np.minimum(t / 0.15, (dur - t) / 0.15)) * (1 + 0.15 * np.sin(2 * np.pi * 0.4 * t))
    y = y * env
    return (y / (np.abs(y).max() + 1e-9) * level).astype(np.float32)


def room(dur, db=-80.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return (rng.standard_normal(int(dur * SR)) * 10 ** (db / 20)).astype(np.float32)


def synth_speech(dur, rng):
    """Fallback speech-like audio: noisy-formant syllables with gaps. Used only if no TTS clip."""
    out, words, t = [], [], 0.0
    while t < dur:
        d = rng.uniform(0.18, 0.4)
        syl = vowel(d, f0=rng.uniform(110, 170), contour=lambda u: -2 * u, vib=0.02,
                    kind=rng.choice(["ah", "mm"]), level=0.2)
        out.append(syl)
        words.append(("w", t, t + d))
        g = rng.uniform(0.30, 0.45)
        out.append(room(g, rng=rng))
        t += d + g
    return np.concatenate(out), words


def breathing(dur=24.0, period=4.0, rng=None):
    """Synthetic breaths: band-passed noise, ~1.0s inhale swell + ~1.3s exhale fall every `period` s."""
    rng = rng or np.random.default_rng(1)
    n = int(dur * SR)
    y = room(dur, db=-70, rng=rng)
    t0 = 0.5
    while t0 + 2.4 < dur:
        for (a, b, rise) in ((t0, t0 + 1.0, True), (t0 + 1.2, t0 + 2.5, False)):
            m = int((b - a) * SR)
            noise = rng.standard_normal(m)
            noise = _resonate(noise, [(1800, 900), (3500, 1500)])
            u = np.linspace(0, 1, m)
            env = (np.sin(np.pi * u) ** 1.5) * (0.4 + 0.6 * (u if rise else 1 - u))
            seg = noise * env
            seg = seg / (np.abs(seg).max() + 1e-9) * 0.08
            i = int(a * SR)
            y[i:i + m] += seg.astype(np.float32)
        t0 += period
    return y


def write_wav(y, path):
    wavfile.write(path, SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))


# --------------------------------------------------------------------------- #
#  speech source -> pieces with word times derived from energy runs
# --------------------------------------------------------------------------- #
def load_speech(tts_path):
    if tts_path and os.path.exists(tts_path):
        y = R.load_audio(tts_path, SR)
        src = f"TTS clip {os.path.basename(tts_path)}"
    else:
        y, _ = synth_speech(30.0, np.random.default_rng(3))
        src = "synthetic syllables (no TTS clip found)"
    fe = R._front_end(y)
    runs = R._merge_runs(R._runs(fe["active"]), int(0.12 * R.FPS))
    runs = [(a / R.FPS, b / R.FPS) for a, b in runs if (b - a) / R.FPS >= 0.12]
    return y, runs, src


def cut_piece(y, runs, t_from, t_to):
    """Cut [t_from, t_to] at the midpoints of the nearest gaps; return audio, words (rel. to piece)."""
    mids = [(runs[i][1] + runs[i + 1][0]) / 2 for i in range(len(runs) - 1)]

    def near(t):
        # Guard: with no TTS clip on this machine the synthetic
        # speech can merge into <2 runs, leaving no gaps to cut at. Cut at t itself.
        return min(mids, key=lambda m: abs(m - t)) if mids else t
    a, b = (0.0 if t_from <= 0 else near(t_from)), near(t_to)
    seg = y[int(a * SR):int(b * SR)]
    words = []
    for s, e in runs:                      # chunk long runs into short pseudo-words (<= 0.4s)
        if s >= a - 1e-6 and e <= b + 1e-6:
            x = s
            while x < e - 1e-6:
                y2 = min(e, x + 0.4)
                words.append(("w", x - a, y2 - a))
                x = y2
    return seg, words, b - a


def compose(y, runs, hum_s=5.0, sil_s=3.0, hum_f0=140.0, kind="ah", stretch=False, seed=0):
    """A | 0.4 | HUM | 0.4 | B | SILENCE | C. Returns audio, words, truth{hum:(a,b), silence:(a,b)}."""
    rng = np.random.default_rng(seed)
    A, wa, da = cut_piece(y, runs, 0, 4.0)
    B, wb, db_ = cut_piece(y, runs, 8.0, 12.0)
    C, wc, dc = cut_piece(y, runs, 16.0, 19.0)
    parts, words, t = [], [], 0.0
    parts.append(A); words += [(w, s + t, e + t) for w, s, e in wa]; t += da
    parts.append(room(0.4, rng=rng)); t += 0.4
    hum0 = t
    parts.append(vowel(hum_s, f0=hum_f0, kind=kind, contour=lambda u: 2 * np.sin(2 * np.pi * 0.7 * u))); t += hum_s
    hum1 = t
    parts.append(room(0.4, rng=rng)); t += 0.4
    if stretch:      # simulate Whisper stretching the word before the hum over it
        w, s, e = words[-1]
        words[-1] = (w, s, hum1 - 0.3)
    parts.append(B); words += [(w, s + t, e + t) for w, s, e in wb]; t += db_
    sil0 = t
    parts.append(room(sil_s, rng=rng)); t += sil_s
    sil1 = t
    parts.append(C); words += [(w, s + t, e + t) for w, s, e in wc]; t += dc
    return np.concatenate(parts), words, {"hum": (hum0, hum1), "silence": (sil0, sil1)}


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def show(res, title):
    print(f"\n--- {title}")
    for ln in res["lines"]:
        print(ln)
    for n in res["notes"]:
        print(f"   [note] {n}")
    print("   [timings]", res["timings"])


def lines_with(res, key):
    return [l for l in res["lines"] if l.strip().startswith(key)]


# --------------------------------------------------------------------------- #
#  tests
# --------------------------------------------------------------------------- #
class ChannelNotesTests(unittest.TestCase):
    """Synthetic spectra and mocked inference only; also included in --fast."""

    @classmethod
    def setUpClass(cls):
        from scipy.signal import butter, sosfilt, resample_poly
        cls.rate = 44100
        t = np.arange(cls.rate * 2) / cls.rate
        rng = np.random.default_rng(48)
        signal = .1 * sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 60))
        signal += rng.normal(0, .04, len(t))
        cls.audio = {}
        for cutoff in (None, 3400, 7000):
            y = signal if cutoff is None else sosfilt(butter(20, cutoff, fs=cls.rate, output='sos'), signal)
            cls.audio[cutoff] = resample_poly(y, 320, 441).astype(np.float32)

    def front(self, y):
        n = len(y) // (R.HOP * 2) + 1
        return dict(active=np.ones(n, bool), voiced=np.zeros(n, bool), f0=np.full(n, np.nan),
                    rms_db=np.full(n, -20.), times=np.arange(n) / R.FPS, n=n, thr=-60.)

    def channel(self, y, rate=44100):
        return R.analyze_channel(y, rate, self.front(y))

    def test_synthetic_bandwidths_and_numbers(self):
        for cutoff, expected in ((None, None), (3400, 'narrowband'), (7000, 'wideband-limited')):
            with self.subTest(cutoff=cutoff):
                result = self.channel(self.audio[cutoff])
                self.assertEqual(result['classification'], expected)
                self.assertEqual(result['sample_rate_hz'], 44100)
                self.assertEqual(result['analysis_rate_hz'], 32000)
                self.assertGreater(result['sounding_frames'], 0)
                self.assertGreater(result['rolloff_99_hz'], 0)
                for key in ('energy_above_3600_fraction', 'energy_above_7500_fraction'):
                    self.assertTrue(0 <= result[key] <= 1)
                line = R.channel_line(result)
                if expected:
                    self.assertIn(expected, line)
                else:
                    self.assertIsNone(line)
                if cutoff == 3400:
                    self.assertLessEqual(result['rolloff_99_hz'], 4000)
                    self.assertTrue(result['shelf_above_3600'])
                elif cutoff == 7000:
                    self.assertGreater(result['rolloff_99_hz'], 4000)
                    self.assertLessEqual(result['rolloff_99_hz'], 8000)
                    self.assertTrue(result['shelf_above_7500'])
                else:
                    self.assertGreater(result['rolloff_99_hz'], 8000)

    def test_source_rate_limits_survive_resampling(self):
        from scipy.signal import resample_poly
        for divisor, rate, expected in ((4, 8000, 'narrowband'), (2, 16000, 'wideband-limited')):
            y = resample_poly(resample_poly(self.audio[None], 1, divisor), divisor, 1)
            self.assertEqual(self.channel(y, rate)['classification'], expected)

    def test_only_sounding_frames_include_unvoiced_sound(self):
        y = np.concatenate([self.audio[3400], self.audio[None]])
        fe = self.front(y)  # entirely unvoiced: sound must still count
        fe['active'] = fe['times'] < 1.9  # leave the splice outside all selected windows
        result = R.analyze_channel(y, self.rate, fe)
        self.assertEqual(result['classification'], 'narrowband')
        self.assertEqual(result['sounding_frames'], int(fe['active'].sum()))
        self.assertIsNone(self.channel(y)['classification'])

    def test_no_sounding_spectrum_is_not_a_channel_diagnosis(self):
        y = self.audio[3400]
        fe = self.front(y)
        fe['active'][:] = False
        for result in (R.analyze_channel(y, self.rate, fe), self.channel(np.zeros_like(y))):
            self.assertIsNone(result['rolloff_99_hz'])
            self.assertIsNone(R.channel_line(result))

    def test_rolloff_alone_does_not_establish_a_shelf(self):
        y = self.audio[3400] + .22 * self.audio[None]
        result = self.channel(y)
        self.assertLessEqual(result['rolloff_99_hz'], 4000)
        self.assertFalse(result['shelf_above_3600'])
        self.assertNotEqual(result['classification'], 'narrowband')

    def test_source_rate_reads_synthetic_wav_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.wav'
            wavfile.write(path, self.rate, np.zeros((100, 2), dtype=np.int16))
            self.assertEqual(R._source_sample_rate(path), self.rate)

    def test_source_rate_general_probe_requires_safe_reader(self):
        from unittest.mock import patch, Mock
        from types import SimpleNamespace
        import wave
        with patch('wave.open', side_effect=wave.Error('not PCM')), \
             patch.object(R.shutil, 'which', return_value='ffprobe'), \
             patch.object(R.subprocess, 'run', return_value=SimpleNamespace(
                 stdout=b'{"streams":[{"sample_rate":"48000"}]}')) as probe:
            self.assertEqual(R._source_sample_rate('calibration.audio'), 48000)
            self.assertIn('a:0', probe.call_args.args[0])
        librosa = Mock()
        librosa.get_samplerate.return_value = 22050
        with patch('wave.open', side_effect=wave.Error('not PCM')), \
             patch.object(R.shutil, 'which', return_value=None), patch.dict(sys.modules, {'librosa': librosa}):
            with self.assertRaisesRegex(RuntimeError, 'ffprobe is required'):
                R._source_sample_rate('calibration.audio')
        librosa.get_samplerate.assert_not_called()

    def test_notes_join_trim_and_omission_count(self):
        self.assertIsNone(R.notes_line([]))
        self.assertEqual(R.notes_line(['  AFFECT skipped\n(model not loaded) ', ' GPU→CPU fallback\t']),
                         'NOTES  AFFECT skipped (model not loaded); GPU→CPU fallback')
        notes = ['AFFECT skipped (model not loaded)', 'GPU→CPU fallback', 'x' * 100, 'last diagnostic']
        line = R.notes_line(notes)
        self.assertTrue(line.endswith('; +2 more'), line)
        self.assertIn('GPU→CPU fallback', line)
        self.assertLessEqual(len(line), 118)

    def test_oversized_first_note_is_visible_and_bounded(self):
        for notes in (['AFFECT skipped: ' + 'x' * 250], ['AFFECT skipped: ' + 'x' * 250, 'other', 'last']):
            line = R.notes_line(notes)
            self.assertTrue(line.startswith('NOTES  AFFECT skipped: '))
            self.assertIn('…', line)
            self.assertLessEqual(len(line), 118)
            if len(notes) > 1:
                self.assertTrue(line.endswith('; +2 more'))

    def analysis_context(self, directory, cutoff):
        from contextlib import ExitStack
        from unittest.mock import patch
        stack = ExitStack()
        self.addCleanup(stack.close)
        path = Path(directory) / 'calibration.wav'
        path.touch()
        y = self.audio[cutoff]
        for name, value in (('load_audio', y), ('_source_sample_rate', self.rate),
                            ('_front_end', self.front(y)), ('_torch', None),
                            ('analyze_quality', {'ok': False, 'reason': 'synthetic fixture'})):
            stack.enter_context(patch.object(R, name, return_value=value))
        for name in ('release_models', '_note_vram'):
            stack.enter_context(patch.object(R, name))
        return path

    def analyze(self, path, **kwargs):
        return R.analyze(path, words=[('calibration', .1, .4), ('signal', .6, 1.2)],
                         events=False, laughs=False, use_whisper=False, update_baseline=False, **kwargs)

    def test_channel_is_first_on_card_and_in_data(self):
        for cutoff, kind in ((None, None), (3400, 'narrowband'), (7000, 'wideband-limited')):
            with tempfile.TemporaryDirectory() as directory:
                path = self.analysis_context(directory, cutoff)
                result = self.analyze(path, affect=False)
                self.assertEqual(result['data']['channel']['classification'], kind)
                self.assertEqual(result['lines'][0].startswith('CHANNEL  '), bool(kind))
                self.assertEqual(result['lines'][-1], 'and now, you')

    def test_chosen_affect_skip_does_not_add_a_raw_note(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.analysis_context(directory, None)
            result = self.analyze(path, affect=False)
            self.assertEqual(result['notes'], [])
            self.assertFalse(any(line.startswith('NOTES  ') for line in result['lines']))
            self.assertEqual(sum(line.startswith('NOTES') for line in result['lines']), 1)

    def test_channel_probe_failure_is_honest(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = self.analysis_context(directory, None)
            with patch.object(R, '_source_sample_rate', side_effect=RuntimeError('metadata unavailable')):
                result = self.analyze(path, affect=False)
            self.assertIsNone(result['data']['channel'])
            notes = [line for line in result['lines'] if line.startswith('NOTES')]
            self.assertEqual(len(notes), 1)
            self.assertTrue(notes[0].startswith('NOTES: ') and 'channel' in notes[0] and 'metadata unavailable' in notes[0])

    def test_affect_load_failure_and_quiet_cli_notes(self):
        import contextlib
        import io
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = self.analysis_context(directory, None)
            words = Path(directory) / 'words.json'
            words.write_text(json.dumps([['calibration', .1, .4], ['signal', .6, 1.2]]))
            with patch.object(R, '_load_affect', side_effect=RuntimeError('model not loaded')):
                result = self.analyze(path, affect=True)
                lines = [line for line in result['lines'] if line.startswith('NOTES')]
                self.assertEqual(len(lines), 1)  # one NOTES line, not a raw digest plus a status line
                line = lines[0]
                self.assertIn('affect', line)
                self.assertIn('model not loaded', line)
                self.assertLessEqual(len(line), 118)
                for quiet in (False, True):
                    out, err = io.StringIO(), io.StringIO()
                    args = [str(path), '--words', str(words), '--no-events', '--no-laughs', '--no-whisper']
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        self.assertEqual(R.main(args + (['--quiet'] if quiet else [])), 0)
                    self.assertIn(line, out.getvalue())
                    self.assertEqual('model not loaded' in err.getvalue(), not quiet)

    def test_successful_cpu_fallback_is_in_raw_card_notes(self):
        from unittest.mock import patch
        def affect(y, fe, dur, notes):
            def worker(device):
                if device == 'cuda':
                    raise RuntimeError('synthetic GPU failure')
                return {'ok': False}
            return R._with_fallback('affect', worker, notes)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = self.analysis_context(directory, None)
            with patch.object(R, '_pick_device', return_value='cuda'), patch.object(R, 'analyze_affect', affect):
                result = self.analyze(path, affect=True)
            lines = [line for line in result['lines'] if line.startswith('NOTES')]
            self.assertEqual(len(lines), 1)
            line = lines[0]
            self.assertIn('retried on CPU', line)
            self.assertLessEqual(len(line), 118)
            self.assertIn('retried on CPU', '; '.join(result['notes']))


class ContourTests(unittest.TestCase):
    """CONTOUR: synthetic F0 tracks only (no audio, no librosa); included in --fast."""

    FPS = 62.5

    def track(self, seconds, shape, n=None):
        n = n or int(seconds * self.FPS)
        x = np.linspace(0.0, 1.0, n)
        return 196.0 * 2.0 ** (np.array([shape(v) for v in x]) / 12.0)       # 196 Hz = G3

    def test_steady_hum_is_held_and_named(self):
        import raven_ears_plus as R
        wobble = self.track(2.0, lambda v: 0.25 * math.sin(40 * v))          # small vibrato is not movement
        c = R.describe_contour(wobble)
        self.assertEqual(c["kind"], "held", c)
        self.assertIn("held near G3", c["phrase"])

    def test_rise_then_hold_names_the_interval(self):
        import raven_ears_plus as R
        c = R.describe_contour(self.track(3.0, lambda v: 4.0 * min(v / 0.4, 1.0)))
        self.assertEqual(c["kind"], "rise", c)
        self.assertIn("rises ~4 st (about a major third), then holds near B3", c["phrase"])

    def test_hold_then_fall(self):
        import raven_ears_plus as R
        c = R.describe_contour(self.track(3.0, lambda v: -7.0 * max(0.0, (v - 0.5) / 0.5)))
        self.assertEqual(c["kind"], "fall", c)
        self.assertIn("holds near G3, then falls ~7 st (about a fifth)", c["phrase"])

    def test_plain_fall_and_over_an_octave(self):
        import raven_ears_plus as R
        self.assertEqual(R.describe_contour(self.track(2.0, lambda v: -3.0 * v))["kind"], "fall")
        big = R.describe_contour(self.track(2.0, lambda v: 15.0 * v))
        self.assertIn("over an octave", big["phrase"])

    def test_arch_and_dip(self):
        import raven_ears_plus as R
        arch = R.describe_contour(self.track(3.0, lambda v: 5.0 * math.sin(math.pi * v)))
        self.assertEqual(arch["kind"], "arch", arch)
        self.assertIn("rises", arch["phrase"]); self.assertIn("then falls", arch["phrase"])
        dip = R.describe_contour(self.track(3.0, lambda v: -5.0 * math.sin(math.pi * v)))
        self.assertEqual(dip["kind"], "dip", dip)

    def test_wavering_without_direction(self):
        import raven_ears_plus as R
        c = R.describe_contour(self.track(3.0, lambda v: 1.2 * math.sin(2 * math.pi * 3 * v)))
        self.assertEqual(c["kind"], "wavers", c)

    def test_declines_when_pitch_is_thin_or_jumps(self):
        import raven_ears_plus as R
        thin = self.track(2.0, lambda v: 0.0)
        thin[::2] = np.nan
        c = R.describe_contour(thin)
        self.assertFalse(c["ok"]); self.assertIn("pitch unclear (voiced 50%)", c["reason"])
        rng = np.random.default_rng(3)
        jumpy = self.track(2.0, lambda v: 0.0) * np.where(rng.random(int(2.0 * self.FPS)) < 0.5, 1.0, 2.0)
        c = R.describe_contour(jumpy)
        self.assertFalse(c["ok"]); self.assertIn("jumps", c["reason"])
        self.assertFalse(R.describe_contour(np.array([]))["ok"])

    def test_one_octave_blip_does_not_break_a_held_note(self):
        import raven_ears_plus as R
        hum = self.track(2.0, lambda v: 0.0)
        hum[40] *= 2.0
        self.assertEqual(R.describe_contour(hum)["kind"], "held")

    def test_only_long_wordless_runs_get_a_contour_and_the_line_fits(self):
        import raven_ears_plus as R
        self.assertAlmostEqual(R.FPS, self.FPS)
        f0 = np.full(int(40 * self.FPS), np.nan)
        spans = [(1.0, 1.6), (5.0, 8.0), (12.0, 14.0), (20.0, 23.0), (30.0, 34.0)]
        for a, b in spans:
            i, j = int(a * self.FPS), int(b * self.FPS)
            f0[i:j] = self.track(b - a, lambda v: 4.0 * v, n=j - i)
        c = R.analyze_contour({"f0": f0}, {"wordless": spans})
        self.assertEqual([r["start"] for r in c["runs"]], [5.0, 12.0, 20.0, 30.0])   # the 0.6 s one is left out
        lines = R.contour_lines(c)
        self.assertEqual(len(lines), 3, lines)                    # the three longest, one line each, in time order
        self.assertTrue(lines[0].startswith("  CONTOUR: 0:05 (3.0s) rises ~4 st (about a major third)"), lines)
        self.assertTrue(lines[1].startswith(" " * 11 + "0:20"), lines)
        self.assertIn("0:30 (4.0s) rises ~4 st (about a major third) · +1 more", lines[2])
        self.assertTrue(all(len(line) <= R.MAX_LINE for line in lines), lines)

    def test_no_line_without_word_times_or_long_runs(self):
        import raven_ears_plus as R
        self.assertIsNone(R.analyze_contour({"f0": np.array([])}, {"wordless": None}))
        self.assertEqual(R.contour_lines(None), [])
        self.assertEqual(R.contour_lines(R.analyze_contour({"f0": np.full(100, 200.0)}, {"wordless": [(0.0, 0.5)]})), [])

    def test_an_unclear_run_says_so_on_the_line(self):
        import raven_ears_plus as R
        f0 = np.full(int(4 * self.FPS), np.nan)
        f0[::3] = 200.0
        lines = R.contour_lines(R.analyze_contour({"f0": f0}, {"wordless": [(0.0, 4.0)]}))
        self.assertIn("pitch unclear", lines[0])


def run_contour_checks(check):
    for test in unittest.defaultTestLoader.loadTestsFromTestCase(ContourTests):
        result = unittest.TestResult()
        test.run(result)
        details = '\n'.join(detail for _, detail in result.errors + result.failures)
        check('contour: ' + test._testMethodName, result.wasSuccessful(), details)


def run_channel_notes_checks(check):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ChannelNotesTests)
    for test in suite:
        result = unittest.TestResult()
        # Each case runs separately; initialize the shared mathematical signals once.
        if not hasattr(ChannelNotesTests, 'audio'):
            ChannelNotesTests.setUpClass()
        test.run(result)
        details = '\n'.join(detail for _, detail in result.errors + result.failures)
        check('channel/notes: ' + test._testMethodName, result.wasSuccessful(), details)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default=os.environ.get("RAVEN_EARS_TEST_TTS", DEFAULT_TTS))
    ap.add_argument("--quick", action="store_true", help="skip baseline seeding")
    ap.add_argument("--fast", action="store_true",
                    help="skip librosa/Praat, audio models and decoding; keep text, arithmetic and formatting checks")
    ap.add_argument("--no-whisper-test", action="store_true")
    a = ap.parse_args()
    skipped = 0

    def audio_tests():
        # Count groups, not assertions: available models change how many checks run.
        nonlocal skipped
        if a.fast:
            skipped += 1
            return False
        return True

    tmp = Path(tempfile.mkdtemp(prefix="ears_plus_test_"))
    t_start = time.time()
    print("raven_ears_plus self-test; scratch:", tmp)
    run_channel_notes_checks(check)
    run_contour_checks(check)
    from test_room_whisper import run_checks as room_whisper_checks
    room_whisper_checks(check)
    from test_speech_laughs import run_checks as speech_laugh_checks
    speech_laugh_checks(check)
    from test_speech_tune import run_checks as speech_tune_checks
    speech_tune_checks(check)
    if not a.fast:
        y_sp, runs, src = load_speech(a.tts)
        print("speech source:", src, f"({len(y_sp) / SR:.1f}s, {len(runs)} energy runs)")
        if len(runs) < 8:
            # The composite tests cut real phrases out of a speech clip. Without one there's
            # nothing to cut, so say so plainly rather than crash if there are too few separate runs.
            print("STOP: the composite tests need a TTS speech clip. One ships at "
                  "fixtures/neutral-voiced.wav; pass --tts <clip> if you moved it.")
            return 2

    # ---- unit: word parsing --------------------------------------------------
    print("\n[1] word parsing")
    w1 = R._parse_words([{"word": " hi", "startTimeMs": 100, "endTimeMs": 400}, ("yo", 1.0, 1.2)])
    w2 = R._parse_words({"raw_stt": {"wordTimestamps": [{"word": "a", "startTimeMs": 0, "endTimeMs": 200}]}})
    check("parse hear_core dicts + tuples", w1 == [("hi", 0.1, 0.4), ("yo", 1.0, 1.2)], str(w1))
    check("parse whole hear() result", w2 == [("a", 0.0, 0.2)], str(w2))

    # ---- unit: pitch ---------------------------------------------------------
    if audio_tests():
        print("\n[2] PITCH on synthetic vowels (known answer)")
        up = vowel(3.0, f0=150, kind="ah", contour=lambda u: np.where(u > 0.75, 5 * (u - 0.75) / 0.25, 0))
        dn = vowel(3.0, f0=150, kind="ah", contour=lambda u: np.where(u > 0.75, -5 * (u - 0.75) / 0.25, 0))
        for nm, sig, want in (("rising end", up, "up"), ("falling end", dn, "down")):
            p = R.analyze_pitch(R._front_end(sig), [("w", 0.1, 2.9)])
            check(f"pitch median ~150Hz, phrase {nm} -> {want}",
                  p.get("ok") and 138 <= p["median_hz"] <= 168 and p["phrase_ends"] == [want],
                  f"median={p.get('median_hz')} ends={p.get('phrase_ends')}")

    # ---- the important one: hum vs silence ----------------------------------
    if audio_tests():
        print("\n[3] WORDLESS vs SILENT: ~5s hum between two phrases + ~3s true silence (THE key check)")
        comp, words, truth = compose(y_sp, runs, hum_s=5.0, sil_s=3.0)
        cpath = str(tmp / "composite.wav")
        write_wav(comp, cpath)
        print(f"  composite {len(comp) / SR:.1f}s, hum at {truth['hum'][0]:.1f}-{truth['hum'][1]:.1f}s, "
              f"silence at {truth['silence'][0]:.1f}-{truth['silence'][1]:.1f}s, {len(words)} words")
        t0 = time.time()
        r = R.analyze(cpath, words=words, use_whisper=False, keep_models=True)
        show(r, f"full card on composite ({time.time() - t0:.1f}s)")
        g = r["data"]["gaps"]
        if not words or g.get("wordless") is None:
            # Guard (16 Oct): no speech runs to cut pseudo-words from (no TTS clip, and the
            # synthetic fallback merged into one run), so WORDLESS can't be tested here.
            print("  SKIP: no word times for the composite, so the WORDLESS check needs a real TTS clip (--tts)")
            g = {**g, "wordless": [], "wordless_s": None, "silent": g.get("silent") or []}
            words_ok = False
        else:
            words_ok = True
        wl_hit = sum(overlap(s, truth["hum"]) for s in g["wordless"])
        if words_ok:
            check("WORDLESS total ~5s", 4.0 <= (g["wordless_s"] or 0) <= 5.6, f"{g['wordless_s']}s")
            check("WORDLESS lies inside the hum", wl_hit >= 0.9 * (g["wordless_s"] or 1e9),
                  f"{wl_hit:.1f}s of {g['wordless_s']}s overlaps the hum")
        sil_hit = sum(overlap(s, truth["silence"]) for s in g["silent"])
        check("SILENT covers the ~3s of true silence (minus edge margins)", 2.2 <= sil_hit <= 3.2, f"{sil_hit:.2f}s inside it; "
              f"total SILENT {g['silent_s']}s also counts the TTS clip's own natural pauses")
        check("SILENT never lands in the hum", sum(overlap(s, truth["hum"]) for s in g["silent"]) < 0.2,
              str([(round(a, 1), round(b, 1)) for a, b in g["silent"]]))
        if words_ok:
            check("WORDLESS line printed", bool(lines_with(r, "WORDLESS")))
        # every line present
        for key, label in (("PITCH", "PITCH"), ("SHAPE", "SHAPE"), ("PEAK", "PEAK")):
            check(f"{label} line present", bool(lines_with(r, key)))
        if r["data"].get("events"):
            check("EVENTS line present", bool(lines_with(r, "EVENTS")), lines_with(r, "EVENTS")[0].strip()[:90])
            names = [e["name"] for e in r["data"]["events"]["events"]]
            info("event labels fired on synthetic hum+speech", ", ".join(sorted(set(names))) or "none (a synthetic vowel is not a human hum to CNN14)")
            info("AudioSet has a literal 'Moan' label", str(r["data"]["events"]["has_moan_label"]) +
                 f" (has 'Wail, moan': {r['data']['events']['wail_moan_label']})")
        else:
            info("EVENTS", "SKIPPED (model not available): " + "; ".join(r["notes"]))
        if r["data"].get("affect") and r["data"]["affect"].get("ok"):
            check("AFFECT line present", bool(lines_with(r, "AFFECT")), lines_with(r, "AFFECT")[0].strip()[:90])
            w = r["data"]["affect"]["whole"]
            check("AFFECT values are finite and sane (-0.3..1.3)", all(-0.3 <= v <= 1.3 for v in w.values()), str(w))
        else:
            info("AFFECT", "SKIPPED (model not available): " + "; ".join(r["notes"]))
        pk = r["data"]["peak"]
        check("PEAK is a time window inside the note", pk.get("ok") and 0 <= pk["start"] < pk["end"] <= len(comp) / SR + 0.01)
        check("all lines <= 120 chars", all(len(l) <= 120 for l in r["lines"]), str(max(len(l) for l in r["lines"])))

    # ---- stretched word (whisper-style) -------------------------------------
    if audio_tests():
        print("\n[4] same, but the word before the hum is stretched over it (Whisper habit)")
        _c, words_s, _t = compose(y_sp, runs, hum_s=5.0, sil_s=3.0, stretch=True)
        r2 = R.analyze(cpath, words=words_s, use_whisper=False, events=False, affect=False)
        g2 = r2["data"]["gaps"]
        print("  ", lines_with(r2, "WORDLESS")[0].strip())
        check("hum still caught as WORDLESS (>=3.5s)", (g2["wordless_s"] or 0) >= 3.5, f"{g2['wordless_s']}s")

    # ---- no words ----------------------------------------------------------------
    if audio_tests():
        print("\n[5] no word times (degrade gracefully)")
        r3 = R.analyze(cpath, words=None, use_whisper=False, events=False, affect=False)
        show(r3, "no words, no models")
        check("SILENT still works without words", bool(lines_with(r3, "SILENT")) and r3["data"]["gaps"]["silent_s"] >= 2.5,
              f"{r3['data']['gaps']['silent_s']}s")
        check("WORDLESS declines honestly without words", r3["data"]["gaps"]["wordless"] is None and not lines_with(r3, "WORDLESS"))
        check("PITCH works without words", bool(lines_with(r3, "PITCH")))
        check("no wpm in SHAPE without words", "wpm" not in "".join(lines_with(r3, "SHAPE")))

    # ---- pure silence / pure hum edge cases ------------------------------------
    if audio_tests():
        print("\n[6] edge cases: all-silent file, pure hum file")
        write_wav(room(4.0), str(tmp / "silent.wav"))
        r4 = R.analyze(str(tmp / "silent.wav"), words=None, use_whisper=False, events=False, affect=False)
        check("near-silent file does not crash", bool(r4["lines"]), r4["lines"][0].strip()[:80])
        write_wav(vowel(6.0), str(tmp / "hum.wav"))
        r5 = R.analyze(str(tmp / "hum.wav"), words=[("x", 0.0, 0.2)], use_whisper=False, events=False, affect=False)
        check("6s hum with one early word -> WORDLESS ~5.5s", (r5["data"]["gaps"]["wordless_s"] or 0) >= 4.5,
              f"{r5['data']['gaps']['wordless_s']}s")

    # ---- breath + real non-speech sounds -------------------------------------
    if audio_tests():
        print("\n[7] BREATH / EVENTS")
        bpath = str(tmp / "breath.wav")
        write_wav(breathing(), bpath)
        rb = R.analyze(bpath, words=[], use_whisper=False, affect=False, keep_models=True)
        if rb["data"].get("events"):
            info("real model on synthetic noise-breaths",
                 "BREATH " + str(rb["data"]["events"]["breath"]) + " (CNN14 is not fooled by band-passed noise)")
            # (a) rate logic with the tagger MOCKED to say 'Breathing' everywhere: 24s of breaths, 4s apart = 15/min
            real_tag = R._tag_windows

            def mock_tag(y32, device):
                centers, probs, labels = real_tag(y32, device)
                probs = probs.copy()
                probs[:, labels.index("Breathing")] = 0.8
                return centers, probs, labels
            R._tag_windows = mock_tag
            try:
                rm = R.analyze(bpath, words=[], use_whisper=False, affect=False)
            finally:
                R._tag_windows = real_tag
            br = rm["data"]["events"]["breath"]
            print("   (mocked tagger) ", (lines_with(rm, "BREATH") or ["(no BREATH line) " + str(br)])[0].strip())
            check("BREATH rate logic: 15/min synthetic -> 12..18 (tagger mocked)", "per_min" in br and 12 <= br["per_min"] <= 18, str(br))
        else:
            info("BREATH", "SKIPPED: event model unavailable")
        pd = Path(os.environ.get("RAVEN_EARS_TEST_PD", HERE / "pd_samples"))
        want = {"Four_sighs.ogg": "Sigh", "Crying_newborn_baby.ogg": "Crying", "Zombie_moan.ogg": "Grunt"}
        if pd.exists() and rb["data"].get("events"):
            print(f"   real public-domain/CC0 samples from {pd} (Wikimedia Commons; optional)")
            for fn, lab in want.items():
                f = pd / fn
                if f.exists():
                    rr = R.analyze(str(f), words=[], use_whisper=False, affect=False, keep_models=True)
                    got = {e["name"] for e in rr["data"]["events"]["events"]}
                    print("    ", fn, "->", lines_with(rr, "EVENTS")[0].strip())
                    check(f"{fn}: '{lab}' detected", lab in got, str(sorted(got)))
        else:
            info("real non-speech samples", "not present (optional); see README")

    # ---- CPU path -------------------------------------------------------------
    if audio_tests():
        print("\n[8] CPU-only path (RAVEN_EARS_DEVICE=cpu)")
        R.release_models()
        os.environ["RAVEN_EARS_DEVICE"] = "cpu"
        t0 = time.time()
        rc = R.analyze(cpath, words=words, use_whisper=False)
        cpu_s = time.time() - t0
        os.environ.pop("RAVEN_EARS_DEVICE")
        show(rc, f"CPU run ({cpu_s:.1f}s)")
        check("CPU run catches the hum", abs((rc["data"]["gaps"]["wordless_s"] or 0) - g["wordless_s"]) < 0.3)
        devs = [rc["data"][k]["device"] for k in ("events", "affect") if rc["data"].get(k)]
        check("models ran on cpu", all(d == "cpu" for d in devs), str(devs))

    # ---- baseline --------------------------------------------------------------
    if not a.quick and audio_tests():
        print("\n[9] BASELINE: seed 10 synthetic notes, then VS USUAL (temp speaker; file deleted at the end)")
        spk = f"_ears_test_{os.getpid()}"
        bpath_file = R.baseline_path(spk)
        try:
            outs = []
            for i in range(10):
                rng = np.random.default_rng(100 + i)
                c, w, _ = compose(y_sp, runs, hum_s=float(rng.uniform(4.0, 6.0)), sil_s=float(rng.uniform(2.5, 3.5)),
                                  hum_f0=float(rng.uniform(130, 150)), seed=i)
                p = str(tmp / f"seed{i}.wav")
                write_wav(c, p)
                outs.append(R.analyze(p, words=w, speaker=spk, use_whisper=False, keep_models=True))
                if i == 0:
                    print("  note 1 ->", lines_with(outs[0], "VS USUAL")[0].strip())
            check("first notes say 'still learning'", "still learning" in lines_with(outs[0], "VS USUAL")[0])
            check("baseline file written outside the vault", bpath_file.exists() and "raven-ears" in str(bpath_file) and
                  "Documents" not in str(bpath_file), str(bpath_file))
            n_lines = len(bpath_file.read_text().splitlines())
            check("10 rows on disk", n_lines == 10, str(n_lines))
            # a typical 11th note: should be unremarkable
            c, w, _ = compose(y_sp, runs, hum_s=5.0, sil_s=3.0, hum_f0=140, seed=50)
            p = str(tmp / "typ.wav"); write_wav(c, p)
            rt = R.analyze(p, words=w, speaker=spk, use_whisper=False, keep_models=True, update_baseline=False)
            print("  typical  ->", lines_with(rt, "VS USUAL")[0].strip())
            # an unusual one: 12s hum an octave-ish higher, no silence
            c, w, _ = compose(y_sp, runs, hum_s=12.0, sil_s=0.6, hum_f0=260, kind="mm", seed=51)
            p = str(tmp / "odd.wav"); write_wav(c, p)
            ro = R.analyze(p, words=w, speaker=spk, use_whisper=False, update_baseline=False)
            vl = lines_with(ro, "VS USUAL")[0]
            print("  unusual  ->", vl.strip())
            flagged = {f[0] for f in ro["data"]["vs_usual"]}
            check("unusual note flags wordless and pitch", {"wordless_s", "pitch_hz"} <= flagged, str(sorted(flagged)))
            check("VS USUAL line <= 120 chars", all(len(l) <= 120 for l in (vl, lines_with(rt, "VS USUAL")[0])))
            check("typical note flags fewer things than the unusual one",
                  len(rt["data"]["vs_usual"]) < len(ro["data"]["vs_usual"]), f"{len(rt['data']['vs_usual'])} vs {len(flagged)}")
        finally:
            R.release_models()
            if bpath_file.exists():
                bpath_file.unlink()
            check("test baseline file deleted", not bpath_file.exists())

    # ---- WORDS card lines, say-none lines, VS USUAL without writing ------------------
    print("\n[9b] WORDS lines, 'none' said out loud, --no-baseline-update still compares")
    ws = [("hello", 1.0, 1.4), ("there", 1.5, 1.9), ("my", 4.0, 4.2), ("love", 4.3, 4.8), ("bye", 5.0, 5.4)]
    wl = R.words_lines(ws, 6.0, {"ok": True, "start": 3.9, "end": 5.0},
                       {"events": [{"label": "Sigh", "name": "Sigh", "start": 2.6, "end": 3.2, "score": 0.9}]},
                       {"wordless": [(2.0, 2.6)]})
    txt = "\n".join(wl)
    check("WORDS: phrases split on the gap, timestamped", "[0:01] hello there" in txt and "[0:04]" in txt, txt[:80].replace("\n", " | "))
    check("WORDS: pause, sigh, wordless voice written inline", "…(2.1 s)" in txt and "[sigh]" in txt and "(voice, no words 0.6 s)" in txt)
    check("WORDS: PEAK markers around the peak span", "⟪PEAK⟫ my" in txt and "bye ⟪/PEAK⟫" in txt)
    check("WORDS lines <= 120 chars", all(len(l) <= 120 for l in R.words_lines([(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(200)], 70.0)))
    long_ws = [(f"w{i}", i * 2.0, i * 2.0 + 0.3) for i in range(150)]
    ll = R.words_lines(long_ws, 300.0, {"ok": True, "start": 150.0, "end": 155.0})
    check("long clip: head+peak+tail with an omission note", any("omitted" in l for l in ll) and len(ll) < 80, str(len(ll)))
    check("WORDS FAILED is loud and says not silence",
          "WORDS FAILED" in R.words_lines(None, 5.0, status="failed: no decoder")[0] and "NOT SILENCE" in R.words_lines(None, 5.0, status="failed: x")[0])
    pl = R.pace_line(ws)
    check("PACE names the longest pause and where it falls", "longest pause 2.1s at 0:02" in pl, pl.strip())
    check("no speech says 'none heard'", "none heard" in R.words_lines(None, 5.0)[0])

    # ---- EMPHASIS: synthetic per-word features -------------------------------------
    ew = [("I", 0.0, 0.1), ("would", 0.2, 0.45), ("never", 0.5, 1.0), ("have", 1.1, 1.3), ("said", 1.35, 1.8),
          ("that", 1.9, 2.2), ("to", 2.25, 2.35), ("him", 2.4, 2.65)]
    flat = {"pitch_st": 0.0, "loud_db": 0.0, "dur_ratio": 1.0}
    ef = [dict(flat) for _ in ew]
    ef[2] = {"pitch_st": 4.2, "loud_db": 5.0, "dur_ratio": 2.0}
    top = R.rank_emphasis(ew, ef)
    check("EMPHASIS: a clearly stressed word is found, and only it", [r["word"] for r in top] == ["never"], str([r["word"] for r in top]))
    el = R.emphasis_line({"ok": True, "words": top})
    check("EMPHASIS line names the cues", el.startswith('  EMPHASIS: "never" (pitch +4 st, loud, long)') and el.endswith("."), el.strip())
    check("EMPHASIS: flat delivery -> none, said out loud",
          R.rank_emphasis(ew, [dict(flat) for _ in ew]) == []
          and "none stands out, even delivery" in R.emphasis_line({"ok": True, "words": []}))
    ef2 = [dict(flat) for _ in ew]
    ef2[6] = {"pitch_st": 4.0, "loud_db": 5.0, "dur_ratio": 2.0}          # "to": long-ish on paper, but a function word
    ef2[0] = {"pitch_st": 6.0, "loud_db": 7.0, "dur_ratio": 3.0}          # "I", 0.1 s: too short to trust
    check("EMPHASIS: a short function word is not over-marked", R.rank_emphasis(ew, ef2) == [], str(R.rank_emphasis(ew, ef2)))
    ef3 = [dict(flat) for _ in ew]
    ef3[5] = {"pitch_st": 5.0, "loud_db": 6.0, "dur_ratio": 2.5}          # "that" (0.3 s) clears all three by a mile
    check("EMPHASIS: an extreme function word can still be marked", [r["word"] for r in R.rank_emphasis(ew, ef3)] == ["that"],
          str([r["word"] for r in R.rank_emphasis(ew, ef3)]))
    ef4 = [dict(flat) for _ in ew]
    ef4[4] = {"pitch_st": None, "loud_db": 5.0, "dur_ratio": 2.0}         # unvoiced: no pitch, still loud + long
    ef4[2] = {"pitch_st": None, "loud_db": 5.0, "dur_ratio": 1.0}         # unvoiced and only one cue
    t4 = R.rank_emphasis(ew, ef4)
    check("EMPHASIS: missing pitch is skipped, not zero; loud+long still counts, one cue alone does not",
          [r["word"] for r in t4] == ["said"] and "pitch" not in R.emphasis_line({"ok": True, "words": t4}), str(t4))
    ef5 = [dict(flat) for _ in ew]
    for i in (1, 2, 3, 4, 5):
        ef5[i] = {"pitch_st": 4.0, "loud_db": 5.0, "dur_ratio": 2.0}
    check("EMPHASIS: keeps only the top few of a run of candidates", len(R.rank_emphasis(ew, ef5)) == 1,
          str(len(R.rank_emphasis(ew, ef5))))
    ewl = R.words_lines(ew, 3.0, emphasis={"ok": True, "words": [{"i": 2, "word": "never"}, {"i": 4, "word": "said,"}]})
    check("EMPHASIS: starred inline in WORDS", "*never*" in "\n".join(ewl), "\n".join(ewl).strip())
    check("EMPHASIS: too few words says so", "none (fewer than 4 words)" in R.emphasis_line(R.analyze_emphasis(ew[:3], {})))
    fe_t = {"times": np.arange(0, 2.7, 1 / R.FPS), "rms_db": np.full(len(np.arange(0, 2.7, 1 / R.FPS)), -30.0),
            "f0": np.full(len(np.arange(0, 2.7, 1 / R.FPS)), np.nan)}
    ft = R.emphasis_features(ew, fe_t)
    check("EMPHASIS: real-feature path copes with all-unvoiced audio (no pitch, flat -> none)",
          all(f["pitch_st"] is None for f in ft) and R.rank_emphasis(ew, ft) == [], str(ft[2]))
    if audio_tests():
        check("no speaker -> 'no comparison' line", any("no speaker given, no comparison" in l for l in r3["lines"]))
        check("skipped sections are named, not omitted", bool(lines_with(r3, "EVENTS")) and bool(lines_with(r3, "BREATH")) and bool(lines_with(r3, "AFFECT")))
        spk2 = f"_ears_test_ro_{os.getpid()}"
        bp2 = R.baseline_path(spk2)
        bp2.parent.mkdir(parents=True, exist_ok=True)
        bp2.write_text("".join(json.dumps({"pitch_hz": 140 + i % 3, "wordless_s": 1.0, "silent_s": 1.0}) + "\n" for i in range(12)))
        try:
            rr = R.analyze(cpath, words=words, speaker=spk2, use_whisper=False, events=False, affect=False, update_baseline=False)
            vl = lines_with(rr, "VS USUAL")[0]
            check("--no-baseline-update still computes VS USUAL", "still learning" not in vl, vl.strip()[:90])
            check("--no-baseline-update did not write", len(bp2.read_text().splitlines()) == 12)
        finally:
            bp2.unlink()

    # ---- unit: cadences on synthetic chord sequences ---------------------------
    print("\n[11] CADENCE / RESOLVES on synthetic chords")
    import harmony as H
    C, D, E, F_, G, A = 0, 2, 4, 5, 7, 9
    CM, GM = ("maj", C), ("maj", G)

    def evs_of(seq, conf=0.9):
        # seq: [(root, quality)], 2 s each; bass on the root
        return [{"t0": 2.0 * i, "t1": 2.0 * i + 2.0, "root": r, "q": q, "bass": r, "bass_share": 0.6, "conf": conf}
                for i, (r, q) in enumerate(seq)]

    def tc_of(vals, step=2.0):
        return {"step": step, "T": np.array(vals, dtype=float)}

    def verdict(seq, key, conf=0.9, tension=None, downbeats=None):
        ev = evs_of(seq, conf)
        end = ev[-1]["t1"]
        tc = tc_of(tension if tension is not None else [0.5] * len(seq))
        return H.find_cadences(ev, [(0.0, key)], [(end, "section")], tc, downbeats)[0]

    cmaj, amin = (C, "major"), (A, "minor")
    check("V-I is authentic", verdict([(F_, "maj"), (G, "maj"), (C, "maj")], cmaj)["cadence"] == "authentic")
    check("V7-I is authentic", verdict([(F_, "maj"), (G, "dom7"), (C, "maj")], cmaj)["cadence"] == "authentic")
    check("IV-I is plagal", verdict([(G, "maj"), (F_, "maj"), (C, "maj")], cmaj)["cadence"] == "plagal")
    check("V-vi is deceptive", verdict([(F_, "maj"), (G, "maj"), (A, "min")], cmaj)["cadence"] == "deceptive")
    hc = verdict([(C, "maj"), (F_, "maj"), (G, "maj")], cmaj)
    check("ending on V is a half cadence", hc["cadence"] == "half" and hc["to"] == "V", hc.get("to", ""))
    check("minor key: V-i authentic", verdict([(D, "min"), (E, "maj"), (A, "min")], amin)["cadence"] == "authentic")
    check("minor key: iv-i plagal", verdict([(E, "maj"), (D, "min"), (A, "min")], amin)["cadence"] == "plagal")
    check("minor key: V-VI deceptive", verdict([(D, "min"), (E, "maj"), (F_, "maj")], amin)["cadence"] == "deceptive")
    check("ii-IV ending is no cadence", verdict([(C, "maj"), (D, "min"), (F_, "maj")], cmaj)["cadence"] == "open")
    lo = verdict([(F_, "maj"), (G, "maj"), (C, "maj")], cmaj, conf=0.4)
    check("low chord confidence still calls it but says likely", lo["cadence"] == "authentic" and lo["likely"])
    vlo = verdict([(F_, "maj"), (G, "maj"), (C, "maj")], cmaj, conf=0.1)
    check("very low confidence is skipped, not guessed", vlo["cadence"] == "unclear")
    dbs = [4.0]
    on = verdict([(F_, "maj"), (G, "maj"), (C, "maj")], cmaj, downbeats=dbs)
    off = verdict([(F_, "maj"), (G, "maj"), (C, "maj")], cmaj, downbeats=[3.0])
    check("arrival on a bar start outweighs one off it", on["strength"] > off["strength"], f"{on['strength']} vs {off['strength']}")
    sq = verdict([(F_, "maj"), (G, "maj"), (C, "maj")], cmaj, tension=[0.8, 0.85, 0.2])
    check("tension before/after the arrival is reported", sq["before"] is not None and sq["before"] > 0.8 and sq["after"] < 0.25,
          f"{sq['before']}->{sq['after']}")
    db = H.find_downbeats([int(i * 20) for i in range(32)], np.array([1.0 if (i // 20) % 4 == 2 and i % 20 == 0 else 0.2
                                                                       for i in range(32 * 20)]))
    check("downbeats follow the strongest beat phase", db is not None and abs(db[0] - 40 / H.FPS) < 1e-6, str(db and db[:2]))
    check("flat beat strengths give no downbeats", H.find_downbeats(list(range(0, 640, 20)), np.ones(700)) is None)
    # the card line, end to end from synthetic data
    seq = [(F_, "maj"), (G, "maj"), (C, "maj"), (F_, "maj"), (C, "maj"), (F_, "maj"), (G, "maj"), (A, "min"), (C, "maj"), (G, "maj")]
    ev = evs_of(seq)
    ends = [(6.0, "section"), (10.0, "section"), (16.0, "section"), (20.0, "section")]
    Rr = {"cads": H.find_cadences(ev, [(0.0, cmaj)], ends, tc_of([0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5]), None),
          "secs": [{"start": a, "end": b, "level": "loud", "db": -20.0} for a, b in ((0, 6), (6, 10), (10, 16), (16, 20))]}
    card = (H.block_resolves(Rr) or [""])[0]
    check("RESOLVES line: tally, hanging end and biggest release", card.startswith("RESOLVES: 2 of 4 phrase ends")
          and "half cadence on V and hangs" in card and "Biggest release" in card, card[:200])
    check("RESOLVES line is silent with nothing to read", H.block_resolves({"cads": [], "secs": []}) == [])

    # ---- unit: meter denominators and the TEMPO/METER agreement -----------------
    print("\n[12] METER denominator + TEMPO line agrees with METER")
    import singer as SG
    import music as MU
    check("7 pulses at 125/min, 4+3 -> 7/4", SG._meter_name(7, (2, 2, 3), 125) == "7/4")
    check("5 pulses at 143/min, 2+3 -> 5/4", SG._meter_name(5, (2, 3), 143) == "5/4")
    check("10 fast pulses with uneven 3+3+2+2 -> 10/8 (tolerate it)", SG._meter_name(10, (3, 3, 2, 2), 150) == "10/8")
    check("10 pulses all 2s -> 5/4 (even pairs)", SG._meter_name(10, (2, 2, 2, 2, 2), 150) == "5/4")
    check("5 pulses faster than any quarter note -> 5/8", SG._meter_name(5, (2, 3), 260) == "5/8")
    check("pulse too slow to name -> no denominator", SG._meter_name(7, (2, 2, 3), 50) is None)
    check("6 pulses in 2+2+2 at 100/min -> 6/4", SG._meter_name(6, (2, 2, 2), 100) == "6/4")
    mclaim = SG.meter_claim({"verdict": "7/4 (4+3)", "confidence": 0.8})
    check("meter_claim quotes a confident verdict", mclaim == "7/4", str(mclaim))
    check("meter_claim declines a weak or denominator-free verdict",
          SG.meter_claim({"verdict": "7 beats per bar, grouping weak (best fit 4+3)", "confidence": 0.2}) is None
          and SG.meter_claim({"verdict": "7/4 (4+3)", "confidence": 0.2}) is None)
    check("meter_claim joins sections", SG.meter_claim({"sections": [
        {"start": 0, "end": 60, "data": {"verdict": "7/4 (4+3)", "confidence": 0.8, "beats_per_bar": 7, "bar_consistency": 1, "bar_s": 3.5}},
        {"start": 60, "end": 120, "data": {"verdict": "4/4 (simple)", "confidence": 0.7, "beats_per_bar": 4, "bar_consistency": 1, "bar_s": 2}}]}) == "7/4 then 4/4")
    tcard = MU.format_card({"file": "x.mp3", "duration": 100.0,
                            "tempo": {"bpm": 143.6, "regularity": 1.0, "beat_strength": 3.0, "meter": "4/4"}, "meter_claim": "5/4"})
    check("TEMPO line quotes the METER claim, never its own 4/4 guess", "meter 5/4" in tcard and "4/4" not in tcard, tcard)
    tcard2 = MU.format_card({"file": "x.mp3", "duration": 100.0,
                             "tempo": {"bpm": 143.6, "regularity": 1.0, "beat_strength": 3.0, "meter": "4/4"}, "meter_claim": None})
    check("TEMPO line drops the meter guess when METER does not commit", "leans" not in tcard2 and "4/4" not in tcard2, tcard2)

    # ---- unit: meter CHANGES inside a song (windowed, synthetic accents) -----------
    print("\n[13] METER changes: windows of the accent series, two-window rule, range seams")
    rng_ = np.random.RandomState(7)

    def synth(parts, p=0.5, noise=0.25):
        """parts = [(pulses, pattern or None)]; None = pure noise. Returns (beats, accents)."""
        a_ = []
        for n_, pat in parts:
            for i_ in range(n_):
                a_.append((pat[i_ % len(pat)] if pat else 0.0) + rng_.randn() * noise)
        a_ = np.array(a_)
        return np.arange(len(a_)) * p + 1.0, a_
    P7, P4 = [3, 0, 1, 0, 2, 0, 0], [3, 0, 1, 0]
    b_, a_ = synth([(900, P7)])
    w_ = SG.window_labels(a_, b_, 0.5)
    sg_, ch_ = SG.detect_meter_changes(w_)
    check("steady 7 -> every window reads 7, no change", all(w["label"] == 7 for w in w_) and ch_ == [] and len(sg_) == 1,
          str([w["label"] for w in w_]))
    b_, a_ = synth([(300, P7), (150, P4), (300, P7)])
    w_ = SG.window_labels(a_, b_, 0.5)
    sg_, ch_ = SG.detect_meter_changes(w_)
    t4a, t4b = b_[300], b_[450]
    check("7 then 4 then 7 -> two changes, 7->4 and 4->7", [(c["from"], c["to"]) for c in ch_] == [(7, 4), (4, 7)], str(ch_))
    check("the seams are ranges that contain the true change times",
          len(ch_) == 2 and ch_[0]["lo"] - 1 <= t4a <= ch_[0]["hi"] + 1 and ch_[1]["lo"] - 1 <= t4b <= ch_[1]["hi"] + 1,
          f"{[(round(c['lo']), round(c['hi'])) for c in ch_]} vs {t4a},{t4b}")
    txt_ = SG.meter_change_text(ch_, "7/4 (2+2+3)", 120.0)
    check("card line names the stretch as a range and says 'then 7/4 again'",
          txt_.startswith("METER: 7/4 (2+2+3), with a 4/4 stretch ≈") and "–" in txt_ and " to ≈" in txt_
          and txt_.endswith(", then 7/4 again."), txt_)
    b_, a_ = synth([(300, P7), (150, P4)])
    ch_ = SG.detect_meter_changes(SG.window_labels(a_, b_, 0.5))[1]
    check("7 then 4 to the end -> one change, text says 'to the end'", len(ch_) == 1
          and SG.meter_change_text(ch_, "7/4", 120.0).endswith("to the end."), str(ch_))
    b_, a_ = synth([(300, P7), (150, None), (300, P7)])
    ch_ = SG.detect_meter_changes(SG.window_labels(a_, b_, 0.5))[1]
    check("a stretch of noise (no meter at all) is not a change", ch_ == [], str(ch_))

    def W(*pairs):
        return [{"t0": 10.0 * i, "t1": 10.0 * i + 30, "label": l, "conf": c} for i, (l, c) in enumerate(pairs)]
    check("a single odd window between 7s is noise, not a change",
          SG.detect_meter_changes(W((7, .9), (7, .9), (4, .9), (7, .9), (7, .9)))[1] == [])
    check("two odd windows in a row do count",
          len(SG.detect_meter_changes(W((7, .9), (7, .9), (4, .9), (4, .9), (7, .9), (7, .9)))[1]) == 2)
    check("low-confidence windows never make a change",
          SG.detect_meter_changes(W((7, .9), (7, .9), (4, .2), (4, .2), (4, .3), (7, .9), (7, .9)))[1] == [])
    check("a low-confidence window inside the run breaks it (two must be consecutive)",
          SG.detect_meter_changes(W((7, .9), (7, .9), (4, .9), (4, .1), (4, .9), (7, .9), (7, .9)))[1] == [])
    check("5 read as 10 (the same meter at two bars) is not a change",
          SG.detect_meter_changes(W((5, .9), (5, .9), (10, .8), (10, .8), (5, .9), (5, .9)))[1] == [])
    check("undecidable windows (no label) are ignored", SG.detect_meter_changes(W((7, .9), (None, 0), (None, 0), (7, .9)))[1] == [])
    N_ = (None, 0)
    ch_ = SG.detect_meter_changes(W((7, .9), (7, .9), (4, .9), (4, .9), (4, .9), (4, .9)))[1]
    check("a stretch with confident windows to the true end still says 'to the end'",
          len(ch_) == 1 and SG.meter_change_text(ch_, "7/4", 120.0).endswith("to the end."), str(ch_))
    ch_ = SG.detect_meter_changes(W((7, .9), (7, .9), (4, .9), (4, .9), (4, .9), N_, N_, N_))[1]
    t_ = SG.meter_change_text(ch_, "7/4", 120.0)
    check("a stretch followed by undecidable windows says its end isn't clear and where the beats blur",
          len(ch_) == 1 and "to the end" not in t_ and "its end isn't clear (beats blur ≈0:50–1:40)" in t_, t_)
    ch_ = SG.detect_meter_changes(W((7, .9), (7, .9), (4, .9), (4, .9), (4, .9), N_, N_, (7, .9), (4, .9), (4, .9)))[1]
    t_ = SG.meter_change_text(ch_, "7/4", 120.0)
    check("blur then a lone opposite window: both mentioned, and the lone window is not counted as a change",
          len(ch_) == 1 and "to the end" not in t_ and "beats blur" in t_ and "a 7/4 window at ≈1:10" in t_, t_)
    ch_ = SG.detect_meter_changes(W((7, .9), (7, .9), (4, .9), (4, .9), (4, .9), (7, .9)))[1]
    t_ = SG.meter_change_text(ch_, "7/4", 120.0)
    check("a lone opposite window right after the stretch is mentioned, not a return",
          len(ch_) == 1 and "to the end" not in t_ and "a 7/4 window at ≈0:50" in t_ and "again" not in t_, t_)
    check("meter_claim declines a change chain without supporting windows",
          SG.meter_claim({"change_chain": ["7/4", "4/4", "7/4"]}) is None)
    # The listening path must retain local counts so a real change can survive
    # a whole-song median dominated by the original meter.
    from unittest.mock import patch
    grid_ = np.arange(240) * 0.5
    downs_ = np.r_[grid_[:120:4], grid_[120::3]]
    candidate_windows_ = [{"t0": i * 15, "t1": i * 15 + 29,
                           "label": 4 if i < 4 else 3, "conf": 0.95} for i in range(7)]
    candidate_segs_, candidate_changes_ = SG.detect_meter_changes(candidate_windows_)
    with patch.object(SG, "accent_series", return_value=(np.zeros(240), [])), \
            patch.object(SG, "meter_changes", return_value=(candidate_windows_, candidate_segs_, candidate_changes_)):
        heard_ = SG._window_changes((grid_, downs_, "fake", np.zeros(1), 1, {}),
                                   ["METER: 4/4"], {"verdict": "4/4", "confidence": 0.95})[1]
    check("new meter evidence saves local four/three bar counts and keeps the strong change",
          heard_["window_bars"][0]["beats_per_bar"] == 4
          and heard_["window_bars"][-1]["beats_per_bar"] == 3
          and SG.meter_claim(heard_) == "4/4 then 3/4", str(heard_["window_bars"]))

    # ---- voice QUALITY -----------------------------------------------------------
    if audio_tests():
        print("\n[9c] voice QUALITY: clean tone vs the same tone with noise, unvoiced input, baseline wording")
        rngq = np.random.default_rng(7)

        def qual(y):
            y = y.astype(np.float32)
            return R.analyze_quality(y, R._front_end(y))

        clean = np.concatenate([vowel(2.0, f0=130 + 10 * i, vib=0.003) if i % 2 == 0 else vowel(2.0, f0=125, vib=0.003) for i in range(4)])
        noisy = clean + (rngq.standard_normal(len(clean)) * 0.12 * np.abs(clean).max()).astype(np.float32)
        qc, qn = qual(clean), qual(noisy)
        check("QUALITY: clean harmonic tone measured", qc.get("ok") and qc["whole"]["hnr_db"] is not None, str(qc.get("whole")))
        if qc.get("ok") and qn.get("ok"):
            wc, wn = qc["whole"], qn["whole"]
            check("QUALITY: clean tone has high HNR", wc["hnr_db"] > 15, f"HNR {wc['hnr_db']:.1f}")
            check("QUALITY: added noise lowers HNR", wn["hnr_db"] < wc["hnr_db"] - 5, f"{wn['hnr_db']:.1f} vs {wc['hnr_db']:.1f}")
            check("QUALITY: added noise lowers CPPS", wn["cpps_db"] < wc["cpps_db"] - 1.5, f"{wn['cpps_db']:.1f} vs {wc['cpps_db']:.1f}")
            lc = R.quality_line(qc)
            check("QUALITY: no baseline says so and uses wide limits", lc and "no baseline yet" in lc and "clear" in lc, lc)
            lnb = R.quality_line(qn)
            check("QUALITY: no-baseline noisy tone is called breathy or noisy", lnb and "breathy or noisy" in lnb, lnb)
            # baseline wording: history of the clean voice, then a noisy note
            rows = []
            for i in range(12):
                r = {k: (wc[k] + float(rngq.normal(0, 0.02 * abs(wc[k]) + 0.05))) for k in R.QUALITY_KEYS if wc.get(k) is not None}
                r["loud_db"] = -20.0 + float(rngq.normal(0, 0.5))
                rows.append(r)
            ln = R.quality_line(qn, {"loud_db": -20.0}, rows)
            check("QUALITY: noisy note vs clean baseline reads breathy", ln and "breathy" in ln and "no baseline" not in ln, ln)
            lsame = R.quality_line(qc, {"loud_db": -20.0}, rows)
            check("QUALITY: same voice vs own baseline says nothing unusual", lsame and "nothing unusual" in lsame, lsame)
            # clear first two thirds, noisy last third -> the card marks it
            mix = np.concatenate([clean[:int(len(clean) * 2 / 3)], noisy[int(len(clean) * 2 / 3):]])
            lm = R.quality_line(qual(mix))
            check("QUALITY: noise only in the last third is marked", lm and "breathier at the end" in lm, lm)
        check("QUALITY: pure noise (no voicing) is handled", qual(room(3.0, db=-30, rng=rngq)).get("ok") is False)
        check("QUALITY: silence is handled", qual(np.zeros(SR * 2)).get("ok") is False)

    check("QUALITY: skipped result gives no line", R.quality_line({"ok": False}) is None)
    # baseline record keeps the measures and --no-baseline-update still compares
    if audio_tests():
        spk3 = f"__qtest_{os.getpid()}"
        bp3 = R.baseline_path(spk3)
        try:
            cp = os.path.join(tempfile.gettempdir(), f"{spk3}.wav")
            write_wav(clean, cp)
            R.analyze(cp, words=[], speaker=spk3, events=False, affect=False, laughs=False, use_whisper=False)
            r1 = R._read_baseline(bp3)
            check("QUALITY: measures are stored in the baseline row", len(r1) == 1 and r1[0].get("cpps_db") is not None, str(list(r1[0])) if r1 else "")
            R.analyze(cp, words=[], speaker=spk3, events=False, affect=False, laughs=False, use_whisper=False, update_baseline=False)
            check("QUALITY: --no-baseline-update leaves the file alone", len(R._read_baseline(bp3)) == 1)
            os.remove(cp)
        finally:
            if bp3.exists():
                bp3.unlink()

    # ---- --lyrics alignment and the names list (synthetic text only) -------------------------
    print("\n[12] LYRICS in hand: alignment, ≈ interpolation, names list")
    import lyric_align as LA
    import lyrics_shared as LS

    def say(text, t0, per=0.4):
        return [(w, t0 + i * per, t0 + i * per + 0.3) for i, w in enumerate(text.split())]
    lyr = ["The counter logs seven blue bins tonight", "Check the tray and count", "Des cubes naïfs sur cette table",
           "Check the tray and count", "The end", "A completely different line here"]
    heard = (say("the counter logs seven blue bins tonite", 10) + say("yeah yeah", 14.5)      # fuzzy match + an ad-lib
             + say("check the tray and count", 20)
             + say("check the try and count", 32)                                         # one word off, second chorus
             + say("the end", 40) + say("banana smoothie operator loud", 45))
    rows, sm = LA.align_lyrics(lyr, heard, dur=60)
    check("align: lines keep the writer's text", [r["text"] for r in rows] == lyr)
    check("align: fuzzy token (tonite~tonight) and an ad-lib still align", not rows[0]["approx"] and abs(rows[0]["t"] - 10) < 0.6, str(rows[0]))
    check("align: the French line nobody heard is approx", rows[2]["approx"] and rows[2]["matched"] == 0, str(rows[2]))
    check("align: ≈ line sits between its heard neighbours", rows[1]["end"] <= rows[2]["t"] and rows[2]["end"] <= rows[3]["t"] + 0.1,
          f"{rows[1]['end']}..{rows[2]['t']}-{rows[2]['end']}..{rows[3]['t']}")
    check("align: repeated chorus lands on the right occurrence", abs(rows[1]["t"] - 20) < 0.6 and abs(rows[3]["t"] - 32) < 0.6,
          f"{rows[1]['t']} and {rows[3]['t']}")
    check("align: times never run backwards", all(a["t"] <= b["t"] + 1e-6 for a, b in zip(rows, rows[1:])))
    check("align: a line Whisper heard as different words is counted, not printed",
          rows[5]["differs"] and rows[5]["approx"] and "banana" not in rows[5]["text"], str(rows[5]))
    # 10 Nov 2026: the summary and MISHEARD now share one rule: the placed (≈) banana line no longer counts, "try" for "tray" does
    check("align: summary counts", sm == {"total": 6, "approx": 2, "differs": 1, "heard": 4}, str(sm))
    sl = LA.summary_line(dict(sm, differs=1))
    check("align: summary line reads like the card", sl.startswith("LYRICS: writer's text, 4 of 6 lines heard, 2 placed between neighbours (≈)")
          and "1 where Whisper heard different words" in sl, sl)
    r0, s0 = LA.align_lyrics(["one line", "another line"], [], dur=20)
    check("align: no heard words at all -> everything ≈, nothing crashes", s0["approx"] == 2 and all(0 <= r["t"] <= 20 for r in r0), str(s0))
    r1, _ = LA.align_lyrics(["opening words here", "second line now"], say("second line now", 30), dur=60)
    check("align: an unheard first line is placed BEFORE the first heard one", r1[0]["approx"] and r1[0]["end"] <= r1[1]["t"] + 0.01, str(r1[0]))
    check("align: accents and punctuation fold (voilà~voila)", LA.norm_tokens("“Te voilà, mon ami” — I’ve") == ["te", "voila", "mon", "ami", "ive"])
    check("align: a one-word line only counts if that word matched", LA.align_lyrics(["Tokenbeam"], say("tokenbeam", 5))[1]["heard"] == 1
          and LA.align_lyrics(["Tokenbeam"], say("the", 5))[1]["heard"] == 0)
    lf = os.path.join(tempfile.gettempdir(), f"__lyr_{os.getpid()}.txt")
    open(lf, "w", encoding="utf-8").write("first line\n\n   \nsecond line  \n")
    check("--lyrics file: one line per line, blanks ignored", LA.read_lyrics_file(lf) == ["first line", "second line"])
    os.remove(lf)
    check("lyric timeline: quote_line tidies curly quotes", LA.quote_line("“Te voilà” — hi") == "\"'Te voilà' — hi\"")

    # timeline helpers on a hand-made R (chords, tension, cadence)
    spans_ = [(0.0, (C, "major"))]
    evs_ = [{"t0": 0.0, "t1": 4.0, "root": F_, "q": "maj", "bass": F_, "conf": 0.9},
            {"t0": 4.0, "t1": 8.0, "root": C, "q": "maj", "bass": C, "conf": 0.9}]
    Rfake = {"evs": evs_, "spans": spans_, "tc": {"step": 2.0, "T": np.array([0.71, 0.7, 0.6, 0.22, 0.2])},
             "cads": [{"cadence": "plagal", "t": 5.0, "before": 0.71, "after": 0.22}]}
    check("lyric timeline: chord numerals over a line", H._line_chords(Rfake, 1.0, 7.0) == "IV→I", H._line_chords(Rfake, 1.0, 7.0))
    lt = H._line_tension(Rfake, 1.0, 7.0)
    check("lyric timeline: tension fall with the release named", lt.startswith("tension falls 0.7") and "(the release)" in lt and "→0.22" in lt, lt)
    check("lyric timeline: a flat stretch says nothing", H._line_tension({**Rfake, "tc": {"step": 2.0, "T": np.array([0.4] * 5)}}, 1.0, 7.0) == "")
    check("no --lyrics: the lyric block is empty (card unchanged)", H.block_lyric_timeline({"align": None}) == [])
    import music as MU
    card = MU.format_card({"file": "x.mp3", "duration": 60.0, "lyrics": [{"t": 3.0, "text": "a line", "approx": True}],
                           "lyrics_align": "LYRICS: writer's text, 0 of 1 lines heard, 1 placed between neighbours (≈)"})
    check("music card: ≈ shows on an interpolated line", "[0:03≈] a line" in card and "writer's text" in card, card[-120:])
    card0 = MU.format_card({"file": "x.mp3", "duration": 60.0, "lyrics": [{"t": 3.0, "text": "a line"}]})
    check("music card: without --lyrics the old LYRICS line is unchanged", "LYRICS:   1 lines (whisper large-v3, rough times)" in card0)

    # MISHEARD block: heard vs wrote, from the same alignment (synthetic words only)
    clock = lambda t: f"{int(round(t)) // 60}:{int(round(t)) % 60:02d}"
    mr, _ = LA.align_lyrics(["Pole pole my love", "Walk with me tonight"], say("paul paul my love", 52) + say("walk with me tonight", 60), dur=90)
    check("misheard: one differing line is found, with what was heard and what was written",
          bool(mr[0]["misheard"]) and not mr[1]["misheard"] and mr[0]["misheard"]["heard_core"] == "paul paul"
          and mr[0]["misheard"]["wrote_core"] == "Pole pole", str(mr[0]["misheard"]))
    mb = LA.misheard_block(mr, clock)
    check("misheard: block header and row read like the card",
          mb == ["MISHEARD (Whisper vs the writer's words)", '  0:52  heard "paul paul my love"   wrote "Pole pole my love"'], str(mb))
    check("misheard: feel-card line names the count and the shortest example",
          LA.feel_line(mr) == 'MISHEARD: 1 line · e.g. 0:52 heard "paul paul" for "Pole pole"', str(LA.feel_line(mr)))
    check("misheard: the summary count equals the block's rows", _["differs"] == len(LA.misheard_lines(mr)) == 1, str(_))
    ml = ["Pole pole my love", "Hold the lantern high", "Carry water to the river", "Hold the lantern high"]
    mh = (say("paul paul my love", 10) + say("hold the lantern high", 20) + say("carry order to the river", 30) + say("hold the lantern high", 40))
    mr2, _ = LA.align_lyrics(ml, mh, dur=60)
    mb2 = LA.misheard_block(mr2, clock)
    check("misheard: several lines come out in time order, one row each",
          [r["t"] for r in LA.misheard_lines(mr2)] == sorted(r["t"] for r in LA.misheard_lines(mr2)) and len(mb2) == 3
          and mb2[2].startswith("  0:30") and "water" in mb2[2] and "order" in mb2[2], str(mb2))
    check("misheard: an identical repeat of a line is not a mishearing", not mr2[3]["misheard"], str(mr2[3]["misheard"]))
    check("misheard: feel-card example is the line that differs most, not the shortest",
          LA.feel_line(mr2) == 'MISHEARD: 2 lines · e.g. 0:10 heard "paul paul" for "Pole pole"', str(LA.feel_line(mr2)))
    pr, _ = LA.align_lyrics(["Hello, World! It's me again."], say("hello world its me again", 5), dur=30)
    check("misheard: case and punctuation differences are not mishearings", not pr[0]["misheard"] and LA.misheard_block(pr, clock) == [])
    tr, _ = LA.align_lyrics(["He'd be gonna sing, oh the song"], say("he be going sing ooh the song", 5), dur=30)
    check("misheard: he'd/he, gonna/going and oh/ooh are the trivial kind, not counted", not tr[0]["misheard"], str(tr[0]["misheard"]))
    nr, ns = LA.align_lyrics(["Walk with me tonight", "Another line goes here"], say("walk with me tonight", 10) + say("another line goes here", 20), dur=40)
    check("misheard: nothing differs -> no block, no feel line", LA.misheard_block(nr, clock) == [] and LA.feel_line(nr) is None)
    ur, _ = LA.align_lyrics(["Pole pole my love", "Une ligne que personne"], say("paul paul my love", 10) + say("banana smoothie operator loud", 30), dur=60)
    ub = LA.misheard_block(ur, clock)
    check("misheard: a placed (≈) line is not misheard; it is counted separately",
          ur[1]["approx"] and not ur[1]["misheard"] and len(LA.misheard_lines(ur)) == 1
          and ub[-1] == "  +1 line Whisper didn't hear (placed ≈, not counted as mishearings)", str(ub))
    longw = " ".join(f"word{c}" for c in "abcdefghijklmnopqrstuvwxyz")
    lw = ["alpha bravo charlie delta " + longw + " echo foxtrot golf hotel"]
    hv = say("alpha bravo charlie delta " + longw.replace("worda", "xylophone") + " echo foxtrot golf hotel", 5)
    lr, _ = LA.align_lyrics(lw, hv, dur=40)
    lb = LA.misheard_block(lr, clock)
    check("misheard: a long line wraps within 118 columns and keeps every word", all(len(x) <= LA.MAX_LINE for x in lb) and len(lb) >= 2
          and "xylophone" in " ".join(lb), str([len(x) for x in lb]))
    mc = MU.format_card({"file": "x.mp3", "duration": 90.0, "lyrics": mr, "lyrics_source": {"status": "file"},
                         "lyrics_align": LA.summary_line(_)})
    check("music card: MISHEARD block follows the lyric rows when the writer's lyrics were supplied",
          "  MISHEARD (Whisper vs the writer's words)" in mc and mc.index("[0:52]") < mc.index("MISHEARD"), mc[-300:])
    mc2 = MU.format_card({"file": "x.mp3", "duration": 90.0, "lyrics": mr, "lyrics_source": {"status": "synced"}})
    check("music card: no MISHEARD block against a database's lyrics, only the writer's", "MISHEARD" not in mc2)

    # names list
    nf = os.path.join(tempfile.gettempdir(), f"__names_{os.getpid()}.txt")
    open(nf, "w", encoding="utf-8").write("# a comment\nZibble\n\n  Žibble  \nZibble\nTokenlet # trailing note\n")
    check("names file: comments/blank/duplicates dropped", LS.load_names(nf) == ["Zibble", "Žibble", "Tokenlet"], str(LS.load_names(nf)))
    check("names file: a missing file is an empty list", LS.load_names(nf + ".nope") == [])
    old_home = os.environ.get("RAVEN_EARS_HOME")
    check("names file: default sits in the ears home", LS.default_names_path().endswith("names.txt") and LS.ears_home() in LS.default_names_path())
    ex = LS.load_names(str(HERE / "names.example.txt"))
    check("names.example.txt ships commented with no names active", ex == [], str(ex))
    try:
        LS.configure(None, ())
        base_tag = LS.names_tag([])
        LS.configure(nf, ["Voilà"])
        eff = LS.effective_names()
        check("names: file plus per-run extras, de-duplicated", eff == ["Zibble", "Žibble", "Tokenlet", "Voilà"], str(eff))
        t1 = LS.names_tag()
        LS.configure(nf, [])
        check("names: cache key changes when the names change", t1 != LS.names_tag() and LS.names_tag() != "" and base_tag == "")
        hash_input = tmp / "hash-input.bin"
        hash_input.write_bytes(b"neutral cache-key test bytes")
        pa = LS.cache_file(str(hash_input))
        LS.configure(None, ())
        pb = LS.cache_file(str(hash_input))
        check("names: lyric cache file name differs with and without names", pa != pb and pa.endswith(".json")
              and "-" in os.path.basename(pa).split("-lang-")[0]
              and "-" not in os.path.basename(pb).split("-lang-")[0])
    finally:
        LS.configure(None, ())
        os.remove(nf)

    class _M1:                           # a faster-whisper with hotwords
        def transcribe(self, audio, initial_prompt=None, hotwords=None): ...

    class _M2:                           # an older one without
        def transcribe(self, audio, initial_prompt=None): ...
    nm = ["Zibble", "Žibble"]
    check("names: hotwords when asked for and supported", LS.whisper_name_kwargs(_M1(), nm, mode="hotwords") == {"hotwords": "Zibble, Žibble"})
    check("names: initial_prompt when hotwords is not supported", LS.whisper_name_kwargs(_M2(), nm, mode="hotwords") == {"initial_prompt": "Zibble, Žibble."})
    check("names: prompt mode uses initial_prompt even if hotwords exists", LS.whisper_name_kwargs(_M1(), nm, mode="prompt") == {"initial_prompt": "Zibble, Žibble."})
    check("names: empty list adds nothing", LS.whisper_name_kwargs(_M1(), []) == {})
    if audio_tests():
        import faster_whisper as _fw, inspect as _insp
        info("faster-whisper hotwords", f"{_fw.__version__}: hotwords supported = {'hotwords' in _insp.signature(_fw.WhisperModel.transcribe).parameters}")

    rw_ = LS.rare_words(["We label the Zibble component", "Žibble increments", "I've counted, we’re ready", "“Te voilà, mon ami”"])
    check("names: rare words from lyrics are proper nouns and non-ASCII, not contractions", rw_ == ["Zibble", "Žibble", "voilà"], str(rw_))
    check("names: a segment that is only the list read back is dropped", LS.names_echo("Zibble, Zibble, Zibble", ["Zibble"])
          and LS.names_echo("Zibbel", ["Zibble"], doubtful=True) and not LS.names_echo("Zibble", ["Zibble"])
          and not LS.names_echo("beside the Zibble counter", ["Zibble"]))

    import music as MU
    if audio_tests():
        print("\n[9d] sung-voice VOCAL line per section: clean vs noisy across two sections, vibrato, no vocal")
        rngs = np.random.default_rng(11)
        note = lambda vib: np.concatenate([vowel(3.0, f0=220 if i % 2 == 0 else 247, vib=vib) for i in range(4)])
        cl_a, cl_b = note(0.02), note(0.02)
        nz_b = cl_b + (rngs.standard_normal(len(cl_b)) * 0.12 * np.abs(cl_b).max()).astype(np.float32)
        secs_ = [0.0, 12.0, 24.0]
        sq_clean = R.sung_quality(np.concatenate([cl_a, cl_b]), [(0.0, 24.0)], secs_)
        sq_mixed = R.sung_quality(np.concatenate([cl_a, nz_b]), [(0.0, 24.0)], secs_)
        lm = R.sung_quality_line(sq_mixed)
        check("VOCAL: a noisier second section reads breathier, the clean first reads clearer",
              lm is not None and "breathier 0:12\u20130:24" in lm and "clearer 0:00\u20130:12" in lm, lm)
        check("VOCAL: numbers ride along in brackets", lm is not None and "[CPPS" in lm, lm)
        lc_ = R.sung_quality_line(sq_clean)
        check("VOCAL: two equally clean sections give no breathy or clear claim",
              lc_ is not None and "breathier" not in lc_ and "clearer" not in lc_ and "no clear contrast" in lc_, lc_)
        vib_a, vib_b = note(0.045), note(0.045)
        lv = R.sung_quality_line(R.sung_quality(np.concatenate([vib_a, vib_b]), [(0.0, 24.0)], secs_))
        check("VOCAL: vibrato in both sections is not read as rough or breathy",
              lv is not None and "rough" not in lv and "breathier" not in lv and "clearer" not in lv, lv)
    if a.fast:
        cl_a, secs_ = None, [0.0, 12.0, 24.0]     # no waveform needed for this early exit
    check("VOCAL: no vocal spans -> no line", R.sung_quality_line(R.sung_quality(cl_a, [], secs_)) is None)
    if audio_tests():
        check("VOCAL: silence inside claimed spans -> no line",
              R.sung_quality_line(R.sung_quality(np.zeros(SR * 24, np.float32), [(0.0, 24.0)], secs_)) is None)
        check("VOCAL: voice present in one section only -> nothing to compare, no line",
              R.sung_quality_line(R.sung_quality(np.concatenate([cl_a, np.zeros(len(cl_b), np.float32)]), [(0.0, 12.0)], secs_)) is None)

    check("VOCAL: no result at all -> no line", R.sung_quality_line(None) is None)
    check("VOCAL: an instrumental card never runs the measurement",
          MU.analyze_vocal_quality("x.mp3", [{"start": 0, "end": 30}], {"instrumental": True, "spans": [], "method": "vocal stem"}, []) is None)
    if audio_tests():
        sq_bl = dict(sq_mixed, whole=dict(sq_mixed["whole"], hnr_db=3.0))
        check("VOCAL: low HNR across the whole song adds the stem-bleed caveat",
              "bleed" in (R.sung_quality_line(sq_bl) or ""), R.sung_quality_line(sq_bl))

    # ---- six stems, keyboard family, synth bass, pad, one key per card, half-time tempo ------------------
    print("\n[11] instruments: keyboard family, piano-stem preference, synth bass, pad fallback (mocked tagger numbers)")
    import instruments as INS
    import singer as SG
    import harmony as HM
    check("six stems: its own cache folder, distinct from the four-stem one",
          SG.stems6_dir("/dev/null") != os.path.join(SG.CACHE, SG.file_hash("/dev/null")) and SG.stems6_dir("/dev/null").endswith("-6s")
          and SG.SRC6[:4] == SG.SRC and SG.SRC6[4:] == ["guitar", "piano"])
    labs = ["Piano", "Electric piano", "Keyboard (musical)", "Organ", "Hammond organ", "Electronic organ", "Synthesizer", "Harpsichord", "Bass guitar"]
    ixk = {l: i for i, l in enumerate(labs)}

    def tagp(**kw):
        p = np.zeros((20, len(labs)), np.float32)
        for k, v in kw.items():
            p[:, ixk[k.replace("_", " ")]] = v
        return p
    on20 = np.ones(20, bool)
    m_close = INS.keys_class_means([tagp(Piano=0.40, Organ=0.33)], ixk, on20)
    vn, _ = INS.keys_verdict(m_close)
    check("keys: piano 0.40 vs organ 0.33 is not clearly ahead -> 'keys (piano or organ?)'", vn == "keys (piano or organ?)", f"{m_close} -> {vn}")
    m_clear = INS.keys_class_means([tagp(Piano=0.60, Organ=0.20)], ixk, on20)
    check("keys: piano 0.60 vs organ 0.20 is clearly ahead -> 'piano'", INS.keys_verdict(m_clear)[0] == "piano", str(m_clear))
    m_alone = INS.keys_class_means([tagp(Hammond_organ=0.5)], ixk, on20)
    check("keys: organ with no rival -> 'organ'; every organ class counts", INS.keys_verdict(m_alone)[0] == "organ", str(m_alone))
    check("keys: nothing tagged -> no name", INS.keys_verdict(INS.keys_class_means([tagp()], ixk, on20))[0] is None)
    m_two = INS.keys_class_means([tagp(Piano=0.2), tagp(Electric_piano=0.5)], ixk, on20)
    check("keys: best stem per class (piano stem and other stem both consulted)", m_two["piano"] == 0.5, str(m_two))
    g = np.array([1, 1, 1, 0, 1, 1], bool)
    rp, ro = np.array([-30, -30, -45, -30, -30, -50.0]), np.array([-32, -32, -32, -32, -32, -32.0])
    pk = np.array([0.5, 0.5, 0.5, 0.5, 0.05, 0.5])
    car = INS.piano_carries(rp, ro, g, pk)
    check("piano stem carries the keys only with level, not far under the other stem, and a keyboard tag",
          car.tolist() == [True, True, False, False, False, False], str(car.tolist()))
    org = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
    check("piano stem carrying the energy overrules an 'organ' tag on the other stem (where it carries, not elsewhere)",
          np.where(car, 0.0, org).tolist() == [0.0, 0.0, 0.6, 0.6, 0.6, 0.6])
    on_ = np.ones(6, bool)
    check("piano: low tagger score -> weak", INS.piano_is_weak(0.32, rp, ro, on_))
    check("piano: far quieter than the other stem -> weak even with a decent score", INS.piano_is_weak(0.6, rp - 20, ro, on_))
    check("piano: decent score and level -> not weak", not INS.piano_is_weak(0.6, rp, ro, on_))
    # synth bass vs plucked bass, from synthetic bass stems (no tagger needed for the shape)
    t_ = np.arange(int(INS.SR * 40)) / INS.SR
    sine = (0.3 * np.sin(2 * np.pi * 55 * t_) * (1 + 0.03 * np.sin(2 * np.pi * 0.2 * t_))).astype(np.float32)
    rngb = np.random.default_rng(5)
    pl = np.zeros_like(t_)
    for t0 in np.arange(0, 40, 0.2):
        i0 = int(t0 * INS.SR)
        seg = np.arange(int(0.2 * INS.SR)) / INS.SR
        f = 82.4
        pl[i0:i0 + len(seg)] += 0.4 * np.exp(-seg * 9) * (np.sin(2 * np.pi * f * seg) + 0.6 * np.sin(2 * np.pi * 2 * f * seg) + 0.4 * np.sin(2 * np.pi * 3 * f * seg))[:len(pl) - i0]
    pl = pl.astype(np.float32) + (rngb.standard_normal(len(pl)) * 1e-4).astype(np.float32)
    mask40 = np.ones(36, bool)
    fs, fp = INS.bass_features(sine, mask40), INS.bass_features(pl, mask40)
    check("bass: a steady sub-bass has a flat envelope and a strong fundamental", fs is not None and fs["fund"] >= 0.9 and fs["swing"] <= INS.BASS_SMOOTH_SWING, str(fs))
    check("bass: a plucked decaying bass has a steep attack and a wide swing", fp is not None and fp["attack"] >= INS.BASS_PLUCK_ATTACK and fp["swing"] >= INS.BASS_PLUCK_SWING, str(fp))
    check("bass: smooth + strong fundamental + tagger leaning Synthesizer -> 'bass (synth-like)'", INS.bass_kind(fs, 0.30, 0.05) == "bass (synth-like)")
    check("bass: smooth shape but the tagger clearly says Bass guitar -> not called synth", INS.bass_kind(fs, 0.02, 0.40) != "bass (synth-like)")
    check("bass: steep, wide-swinging, tagger not synth -> 'bass guitar (plucked)'", INS.bass_kind(fp, 0.02, 0.15) == "bass guitar (plucked)")
    check("bass: plucked shape but tagger says Synthesizer -> stays unclear", INS.bass_kind(fp, 0.40, 0.05) is None)
    check("bass: in-between evidence -> unclear (None)", INS.bass_kind({"attack": 3.5, "swing": 7.0, "fund": 0.5}, 0.0, 0.0) is None)
    check("bass: no features -> unclear (None)", INS.bass_kind(None, 0.5, 0.0) is None)
    if audio_tests():
        # pad fallback on synthetic 'other' stems: a sustained chord vs a clicky one
        pad = sum(0.1 * np.sin(2 * np.pi * f * t_ + ph) for f, ph in ((220, 0), (277, 1), (330, 2), (440, 3))).astype(np.float32)
        pad *= (1 + 0.05 * np.sin(2 * np.pi * 0.1 * t_)).astype(np.float32)
        clk = np.zeros_like(t_)
        for t0 in np.arange(0, 40, 0.25):
            i0 = int(t0 * INS.SR)
            clk[i0:i0 + 400] += (rngb.standard_normal(400) * 0.4).astype(np.float32) * np.exp(-np.arange(400) / 60)[:len(clk) - i0]
        clk = clk.astype(np.float32)
        fpad, fclk = INS.pad_features(pad, mask40), INS.pad_features(clk, mask40)
        check("pad: a sustained smooth chord plus a faint tagger synth hint is called 'synth/pad-like layer (unconfirmed)'",
              INS.pad_kind(fpad, 0.10) == "synth/pad-like layer (unconfirmed)", str(fpad))
        check("pad: the same smooth shape with no tagger hint stays unclear (a sustained guitar looks the same)", INS.pad_kind(fpad, 0.0) is None)
        check("pad: a clicky percussive layer is not, hint or no hint", INS.pad_kind(fclk, 0.2) is None, str(fclk))
        check("pad: thin evidence (too little audio) stays unclear", INS.pad_kind(INS.pad_features(pad, np.ones(3, bool)), 0.2) is None)

    base_res = {"duration": 60.0, "n": 58, "hop": 1.0, "unclear": {"other": 0.8}, "hints": {}, "other_on": [(0.0, 58.0)], "vocal_on": [],
                "families": {"drum machine": {"stem": "drums", "spans": [(0.0, 58.0)], "conf": 0.8, "peak": 0.9, "share": 1.0}}}
    txt_pad = "\n".join(INS.block(dict(base_res, other_character={"label": "synth/pad-like layer (unconfirmed)", "feat": {}})))
    txt_no = "\n".join(INS.block(dict(base_res, other_character={"label": None, "feat": {}})))
    check("card: pad-like layer replaces 'a melodic layer I cannot name' only when the shape supports it",
          "synth/pad-like layer (unconfirmed)" in txt_pad and "cannot name" not in txt_pad and "cannot name" in txt_no and "unclear" in txt_no, txt_pad)

    print("\n[12] one key per card: chords against the spectral reading")

    def evs_of(seq):                       # [(root, quality, seconds)]
        t0, out = 0.0, []
        for r_, q_, d_ in seq:
            out.append({"t0": t0, "t1": t0 + d_, "root": r_, "q": q_})
            t0 += d_
        return out
    ev_d = evs_of([(2, "maj", 20), (7, "maj", 20), (9, "maj", 10), (11, "min", 10)])      # D G A Bm: D major only
    kv = HM.key_verdict(ev_d, (2, "major"), {"key": "A major", "r": 0.5, "margin": 0.02})
    check("key: chords decisive -> the chord key is the claim, even against a thin spectral reading",
          kv["state"] == "chords" and kv["key"] == "D major", str(kv))
    ev_sp = evs_of([(7, "maj", 25), (4, "maj", 25), (2, "maj", 25), (9, "maj", 25)])      # G is D-area only, E is A-area only
    kv2 = HM.key_verdict(ev_sp, (2, "major"), {"key": "E major", "r": 0.5, "margin": 0.02})
    check("key: split evidence says so in ONE line and claims no key", kv2["state"] == "split" and kv2["key"] is None
          and kv2["line"].startswith("key unclear") and "\n" not in kv2["line"] and "never settles" in kv2["line"], str(kv2))
    ev_a = evs_of([(9, "maj", 20), (2, "maj", 20), (4, "maj", 20), (6, "min", 20)])       # A D E F#m: A major
    kv3 = HM.key_verdict(ev_a, (9, "major"), {"key": "F# minor", "r": 0.5, "margin": 0.05})
    check("key: chords decisive in the spectral key's own area -> claimed once, spelled with sharps", kv3["key"] in ("A major", "F# minor") and "G♭" not in kv3["line"], kv3["line"])
    ev_w = evs_of([(0, "maj", 30), (7, "maj", 30), (2, "min", 20), (5, "maj", 20), (1, "maj", 6), (6, "maj", 6)])   # chromatic mess
    kv4 = HM.key_verdict(ev_w, (0, "major"), {"key": "C major", "r": 0.8, "margin": 0.2})
    check("key: weak chords but a confident spectral reading in the same area -> that key", kv4["key"] == "C major" and kv4["state"] in ("agree", "chords"), str(kv4))
    kv5 = HM.key_verdict(ev_w, (0, "major"), {"key": "E major", "r": 0.8, "margin": 0.25})
    check("key: weak chords, confident spectral reading elsewhere -> the spectral key, said as such",
          kv5["state"] in ("spectral", "chords") and kv5["key"] is not None, str(kv5))
    ev_tie = evs_of([(0, "maj", 30), (9, "min", 30), (5, "maj", 30), (2, "min", 30)])             # C Am F Dm: C major and F major both fit fully
    kv6 = HM.key_verdict(ev_tie, (5, "major"), {"key": "C major", "r": 0.9, "margin": 0.3})
    check("key: when several areas fit the chords equally, the spectral area wins the tie", kv6["key"] in ("C major", "A minor") and kv6["state"] in ("chords", "agree"), str(kv6))
    check("key: card line quotes the claim (music.py KEY line is the verdict line)",
          "KEY:      " + kv["line"] in MU.format_card({"file": "x.mp3", "duration": 60.0, "key_claim": kv, "key": {"key": "F# minor", "r": 0.48, "margin": 0.05}})
          and "F# minor" not in MU.format_card({"file": "x.mp3", "duration": 60.0, "key_claim": kv, "key": {"key": "F# minor", "r": 0.48, "margin": 0.05}}))
    check("key: spelling is shared with music.py", HM.key_label((6, "minor")) == "F# minor" and HM.key_label((3, "major")) == "Eb major")
    print("\n[13] double-time tempo: say the half-time reading beside the number")
    md_third = {"sections": [{"data": {"split_votes_ternary": 0.5}}]}
    ht = MU.half_time_note(161.5, md_third)
    check("tempo: fast pulse that sits in thirds -> 'felt beat likely ~81'", ht is not None and "~81" in ht and "161.5" in ht, str(ht))
    check("tempo: a fast pulse that splits in twos gets no half-time note", MU.half_time_note(161.5, {"sections": [{"data": {"split_votes_ternary": 0.1}}]}) is None)
    check("tempo: a moderate pulse in thirds gets none (6/8 at 100 is just 100)", MU.half_time_note(100.0, md_third) is None)
    check("tempo: no meter data -> none", MU.half_time_note(170.0, None) is None)
    check("tempo: an odd committed meter (5/4, 10/8) gets no 12/8 half-time note; 4/4 or 12/8 still does",
          MU.half_time_note(150.0, md_third, "5/4") is None and MU.half_time_note(150.0, md_third, "10/8") is None
          and MU.half_time_note(150.0, md_third, "4/4") is not None and MU.half_time_note(150.0, md_third, "12/8") is not None)
    check("tempo: the number itself is never changed by the note",
          "161.5 BPM" in MU.format_card({"file": "x.mp3", "duration": 60.0, "tempo": {"bpm": 161.5, "regularity": 1.0, "beat_strength": 2.8, "half_time": ht}}) and "~81" in MU.format_card({"file": "x.mp3", "duration": 60.0, "tempo": {"bpm": 161.5, "regularity": 1.0, "beat_strength": 2.8, "half_time": ht}}))

    print("\n[13b] tempo: steady beat_this bars lead; librosa is the fallback")
    import tempo_trust as TT, slim as SL
    bt_ok = {"engine": "beat_this", "beats_per_bar": 4, "bar_consistency": 1.0, "tracker_bpm": 111.1, "split_votes_ternary": 0.42}
    rd = TT.lead_reading(143.6, bt_ok)
    check("tempo trust: disagreeing, confident beat_this leads with its BPM and meter, names librosa once",
          rd is not None and TT.lead_line(rd) == "111.1 BPM · 4 beats per bar · bars steady 1.00 (beat tracker) · librosa heard 143.6", str(rd))
    check("tempo trust: a committed meter claim is the meter named", TT.lead_reading(143.6, bt_ok, "4/4")["meter"] == "4/4")
    check("tempo trust: agreement still leads, without redundant librosa provenance",
          all(TT.lead_reading(b, bt_ok)["bpm"] == 111.1 and "librosa heard" not in TT.lead_line(TT.lead_reading(b, bt_ok))
              for b in (111.1, 113.0)))
    check("tempo trust: 2x / 0.5x / 1.5x / 2:3 relations still lead",
          all(TT.lead_reading(b, bt_ok)["bpm"] == 111.1 for b in (55.55, 222.2, 74.07, 166.65)))
    check("tempo trust: low bar consistency, other engine, no data, untimed sections -> fallback",
          TT.lead_reading(143.6, dict(bt_ok, bar_consistency=0.48)) is None and TT.lead_reading(143.6, dict(bt_ok, engine="madmom")) is None
          and TT.lead_reading(143.6, None) is None and TT.lead_reading(143.6, {"sections": [{"data": bt_ok}, {"data": bt_ok}]}) is None
          and TT.lead_reading(143.6, {"sections": [{"data": bt_ok}]}) is not None)
    t_dis = {"bpm": 143.6, "regularity": 0.78, "beat_strength": 3.08}
    fc = lambda t: MU.format_card({"file": "x.mp3", "duration": 60.0, "tempo": t})
    check("tempo trust: the full card prints the lead line and no felt-beat hedge",
          "TEMPO:    111.1 BPM" in fc(dict(t_dis, lead=TT.lead_line(rd))) and "felt beat" not in fc(dict(t_dis, lead=TT.lead_line(rd))))
    check("tempo trust: without a lead the full card labels the estimate", "TEMPO:    143.6 BPM (librosa estimate) · steady 0.78 · beat strength 3.08x" in fc(t_dis))
    def feel(tempo_bpm, meter):
        txt = "MUSIC: x.mp3 · 1:00\n  TEMPO:    %g BPM · steady 0.78 · beat strength 3.08x\n" % tempo_bpm
        out = SL.render_first_listen({"music": txt, "singer": "", "harmony": ""},
                                     {"music": {"tempo": {"bpm": tempo_bpm, "regularity": 0.78}}, "singer": {"meter": meter}, "harmony": {}})
        heading = re.search(r"(?m)^TEMPO:.*(?:\n  .*)*", out)[0]
        return " ".join(heading.split())
    check("tempo trust, feel card: disagreement -> beat_this leads",
          feel(143.6, bt_ok) == "TEMPO: 111.1 BPM · 4 beats per bar · bars steady 1.00 (beat tracker) · librosa heard 143.6", feel(143.6, bt_ok))
    check("tempo trust, feel card: agreement -> beat_this leads", feel(111.1, bt_ok) == "TEMPO: 111.1 BPM · 4 beats per bar · bars steady 1.00 (beat tracker)", feel(111.1, bt_ok))
    check("tempo trust, feel card: low consistency -> labelled estimates", feel(117.5, dict(bt_ok, bar_consistency=0.48, tracker_bpm=176.5)) == "TEMPO: 117.5 BPM (librosa estimate) · steady 0.78 · beat strength 3.08x · beat tracker heard 176.5 BPM (bars steady 0.48)")

    def tempo_section(start, end, bpm, bpb=4, bar=1.38):
        return {"start": start, "end": end, "data": dict(bt_ok, tracker_bpm=bpm, beats_per_bar=bpb, bar_s=bar)}
    multi = {"sections": [tempo_section(0, 30, 176.5), tempo_section(30, 50, 85.7, 2), tempo_section(50, 173, 166.7)]}
    reading = TT.lead_reading(117.5, multi)
    check("tempo trust: consistent section bars yield 173.9, without averaging the half-time pulse",
          reading is not None and reading["bpm"] == 173.9 and reading["bpb"] == 4
          and "4 beats × 60 / 1.38s" in TT.lead_line(reading), str(reading))
    check("tempo trust: same count across steady sections also derives the bar tempo",
          TT.lead_reading(117.5, {"sections": [tempo_section(0, 60, 170), tempo_section(60, 120, 175)]})["bpm"] == 173.9)
    check("tempo trust: inconsistent section bars -> fallback",
          TT.lead_reading(117.5, {"sections": [tempo_section(0, 100, 170), tempo_section(100, 120, 120, bar=2)]}) is None)
    check("tempo trust: no dominant beat count -> fallback",
          TT.lead_reading(117.5, {"sections": [tempo_section(0, 60, 170), tempo_section(60, 120, 85, 2)]}) is None)
    check("tempo trust: whole-song data takes precedence, even over coherent sections",
          TT.lead_reading(117.5, dict(multi, whole_song_data=bt_ok))["bpm"] == 111.1
          and TT.lead_reading(117.5, dict(multi, whole_song_data=dict(bt_ok, bar_consistency=0.4))) is None)
    check("tempo trust: absent or invalid librosa cannot veto steady beat_this",
          all(TT.lead_reading(b, bt_ok)["bpm"] == 111.1 for b in (None, 0, float("nan"))))
    check("tempo trust: nonfinite or missing beat_this measurements -> fallback",
          all(TT.lead_reading(117.5, dict(bt_ok, **bad)) is None for bad in
              ({"tracker_bpm": float("inf")}, {"tracker_bpm": None}, {"bar_consistency": float("nan")}, {"beats_per_bar": True})))

    import copy, listen, listen_melody
    song = {"cards": {"music": {"text": "MUSIC: invented.mp3 · 2:53\n  TEMPO:    117.5 BPM · felt beat likely ~59 (12/8-type feel)\nSHAPE: keep me\n",
                                 "json": {"tempo": {"bpm": 117.5, "regularity": 1.0}, "meter_claim": "8/8 then 2/4 then 4/4"}},
                      "singer": {"text": "", "json": {"meter": multi}}, "harmony": {"text": "", "json": {}}},
            "manifest": {"passes": {n: {"status": "ran", "reason": ""} for n in ("music", "singer", "harmony")}},
            "melody": {"status": "ran", "reason": "", "text": "rhythm: old saved reading\nABC stays\n",
                       "json": {"hook_text": "rhythm: old saved reading\nhook stays\n"}}}
    original = copy.deepcopy(song)
    full, feel_text, melody = (listen.render(song, view)[0] for view in ("full", "first-listen", "melody"))
    check("tempo trust: saved full/feel/melody agree; obsolete hedge/claim gone, captured data untouched",
          all("173.9 BPM" in text and "felt beat" not in text and "12/8-type" not in text
              and "8/8 then" not in text for text in (full, feel_text, melody))
          and "SHAPE: keep me" in full and "ABC stays" in melody and "hook stays" in feel_text and song == original)
    no_lib = copy.deepcopy(song)
    no_lib["cards"]["music"].update(text="MUSIC: invented.mp3 · 2:53\n",
                                    json={"stage_status": [{"stage": "tempo", "status": "failed"}]})
    check("tempo trust: saved views still lead when librosa failed and left no tempo heading",
          all("173.9 BPM" in listen.render(no_lib, view)[0] for view in ("full", "first-listen", "melody")))
    check("tempo trust: a fresh full card needs no librosa measurement to print its lead",
          "TEMPO:    173.9 BPM" in fc({"lead": TT.lead_line(reading)}))
    for failure in ("failed", "skipped", "not run"):
        song["cards"]["singer"]["json"]["stage_status"] = [{"stage": "meter", "status": failure}]
        check("tempo trust: saved views reject a " + failure + " meter stage",
              all("173.9 BPM" not in listen.render(song, view)[0] for view in ("full", "first-listen", "melody")))

    # ---- credits: parsing, mapping, tie-break, block, off switch, network failure (HTTP layer mocked; nothing touches the net) ----
    print("\n[12] credits: file + MusicBrainz parsing (mocked), word mapping, tie-break only on stems with energy, three-part block")
    import credits as CR
    os.environ.pop("DISCOGS_TOKEN", None); os.environ.pop("RAVEN_DISCOGS_ENV", None)      # tests never use a real token
    ents = CR.parse_credit_text("# c\nPlayer One: drum programming, electric guitar, piano, synth\nbass, vocals\n\nPlayer Two - synth # note\nExample Artist: vocals")
    check("credits file: 'Name: a, b' and bare 'a, b' and 'Name - a' lines parse; comments/blanks dropped",
          [e["who"] for e in ents] == ["Player One", None, "Player Two", "Example Artist"] and ents[1]["classes"] == ["bass guitar", "vocals"], str(ents))
    check("credit words: drum programming -> drum machine, synthesizer/keyboards/bass/drums/vocals map to the tagger's names",
          [CR.map_word(w) for w in ("drum programming", "Synthesizer", "keyboards", "bass", "drums (drum set)", "lead vocals", "electric guitar", "grand piano", "mixing")]
          == ["drum machine", "synth", "keyboard", "bass guitar", "drums", "vocals", "electric guitar", "piano", None])
    credit_examples = {
        "organizer": None, "processing": None, "licensing": None, "disc jockeys": None,
        "bass clarinet": "woodwinds", "bass trombone": "brass", "bass vocals": "vocals",
        "contrabassoon": "woodwinds", "acoustic piano": "piano", "nylon-string harp": "harp",
        "12xstring": None, "12-string banjo": "banjo/mandolin/ukulele", "twelve-string banjo": "banjo/mandolin/ukulele",
        "English horn": "woodwinds", "piano accordion": "woodwinds",
        "guitar technician": None, "piano tuning": None, "electronic production": None,
        "organ": "organ", "singing": "vocals", "double bass": "double bass", "contrabass": "double bass",
        "acoustic guitar": "acoustic guitar", "nylon-string guitar": "acoustic guitar",
        "12-string guitar": "acoustic guitar", "twelve-string guitar": "acoustic guitar",
        "synth programming": "synth", "drums programming": "drum machine", "electronic organ": "organ",
    }
    credit_mismatches = [(w, CR.map_word(w), expected) for w, expected in credit_examples.items() if CR.map_word(w) != expected]
    check("credit words: whole instrument names keep their family; substrings and support roles name no instrument",
          not credit_mismatches, str(credit_mismatches))
    omni = CR.parse_credit_text("A. Player: Omnichord, synth bass, keyboards")
    check("credit words: Omnichord maps to synth, keeps its credited word, and leaves synth bass specific",
          [CR.map_word(w) for w in ("omnichord", "Suzuki Omnichord", "OMNICHORD")] == ["synth", "synth", "synth"]
          and omni == [{"who": "A. Player", "words": ["Omnichord", "synth bass", "keyboards"], "classes": ["synth", "synth bass", "keyboard"]}], str(omni))
    rels = [{"type": "instrument", "attributes": ["piano", "synthesizer"], "artist": {"name": "A. Player"}},
            {"type": "vocal", "attributes": ["lead vocals"], "artist": {"name": "S. Singer"}},
            {"type": "programming", "attributes": ["drums"], "artist": {"name": "A. Player"}},
            {"type": "performer", "attributes": ["additional", "cello"], "artist": {"name": "C. Ellist"}},
            {"type": "producer", "attributes": [], "artist": {"name": "P. Roducer"}}]
    ee = CR.entries_from_relations(rels)
    check("musicbrainz relations: instrument/vocal/programming/performer collected per artist, producer ignored",
          [e["who"] for e in ee] == ["A. Player", "S. Singer", "C. Ellist"] and ee[0]["classes"] == ["piano", "synth", "drum machine"]
          and ee[1]["classes"] == ["vocals"] and ee[2]["classes"] == ["strings"] and "additional" not in ee[2]["words"], str(ee))
    check("credits.example.txt ships commented with nothing active", CR.read_credits_file(str(HERE / "credits.example.txt")) == [])

    def fake_web(responses, calls):
        def g(url):
            calls.append(url)
            for key, val in responses.items():
                if key in url:
                    if isinstance(val, Exception):
                        raise val
                    return val
            raise CR.LookupFailed("no such url in the mock")
        return g
    rec = lambda i, title, art, score=100, dis="", rg=None: {"id": i, "title": title, "score": score, "disambiguation": dis,
                                                           "artist-credit": [{"name": art}],
                                                           "releases": [{"id": "r" + i, "status": "Official",
                                                                         "release-group": rg or {"primary-type": "Album"}}]}
    cands = CR.pick_candidates([rec("live1", "Calibration Example", "Example Artist", dis="live"), rec("remx", "Calibration Example (remix)", "Example Artist"),
                                rec("stud", "Calibration Example", "Example Artist", score=90), rec("other", "Calibration", "Someone Else")],
                               "Example Artist", "Calibration Example")
    check("match: the studio recording beats live/remix; a different artist/title is not a match", cands and cands[0][1]["id"] == "stud", str([(round(s, 2), r["id"]) for s, r in cands]))
    check("match: 'live' is allowed when the wanted name says so",
          CR.pick_candidates([rec("live1", "Song (Live)", "Band"), rec("st", "Song", "Band")], "Band", "Song (Live)")[0][1]["id"] == "live1")
    check("match: nothing close enough -> no candidates", CR.pick_candidates([rec("x", "Totally Other", "Nobody")], "Band", "Song") == [])
    check("file name: 'Artist - Title.mp3' is read; a plain name gives nothing",
          CR.name_from_file("/m/Example Band - Sample Recording.mp3") == ("Example Band", "Sample Recording") and CR.name_from_file("/m/sample-recording.mp3") == (None, None))
    tmpc = tempfile.mkdtemp(prefix="__cr_")
    calls = []
    web = {"recording?": {"recordings": [rec("m1", "Calibration Example", "Example Artist")]},
           "recording/m1": {"relations": [{"type": "instrument", "attributes": ["piano"], "artist": {"name": "A. Dessner"}},
                                          {"type": "vocal", "attributes": [], "artist": {"name": "T. Swift"}}]}}
    old_get = CR._get_json
    try:
        CR._get_json = fake_web(web, calls)
        g1 = CR.get("/m/x.mp3", "Example Artist", "Calibration Example", cache_dir=tmpc)
        n1 = len(calls)
        g2 = CR.get("/m/x.mp3", "Example Artist", "Calibration Example", cache_dir=tmpc)
        check("musicbrainz lookup: search then recording fetch, credits parsed; the second ask is served from the disk cache",
              g1["source"] == "musicbrainz" and n1 == 2 and len(calls) == n1 and g2["entries"] == g1["entries"]
              and calls[1].startswith(CR.MB + "recording/m1?") and "artist-rels" in calls[1], str(calls))
        calls.clear()
        web2 = {"recording?": {"recordings": [rec("m2", "Song", "Band")]}, "recording/m2": {"relations": []},
                "release/rm2": {"media": [{"tracks": [{"recording": {"id": "m2", "relations": [
                    {"type": "performer", "attributes": ["bass guitar"], "artist": {"name": "B. Ass"}}]}}]}]}}
        CR._get_json = fake_web(web2, calls)
        g3 = CR.get("/m/y.mp3", "Band", "Song", cache_dir=tmpc)
        check("musicbrainz lookup: no credits on the recording -> the release's recording-level relations are tried",
              g3["source"] == "musicbrainz" and g3["entries"][0]["classes"] == ["bass guitar"] and any("release/rm2" in c for c in calls), str(calls))
        CR._get_json = fake_web({"recording?": {"recordings": []}}, [])
        g4 = CR.get("/m/z.mp3", "Nobody", "Nothing", cache_dir=tmpc)
        check("musicbrainz lookup: no match -> one honest note, no entries", g4["source"] is None and "no match" in g4["note"] and g4["entries"] == [], str(g4))
        CR._get_json = fake_web({"recording?": CR.LookupFailed("URLError: timed out")}, [])
        g5 = CR.get("/m/w.mp3", "Net", "Down", cache_dir=tmpc)
        check("network failure never breaks the card: a note, no entries, no exception", g5["source"] is None and "unreachable" in g5["note"] and "timed out" in g5["note"], str(g5))
        check("a failed lookup is not cached (the next run asks again)", not os.path.exists(CR.cache_path("Net", "Down", tmpc)))
        calls.clear()
        CR._get_json = fake_web(web, calls)
        g6 = CR.get("/m/x.mp3", "Example Artist", "Calibration Example", off=True, cache_dir=tmpc)
        check("--no-credits: no lookup is made and nothing is returned", g6.get("off") and not calls and g6["entries"] == [])
        gf = CR.get("/m/x.mp3", credits_file=str(HERE / "credits.example.txt"))
        check("--credits FILE replaces the lookup (no web call)", not calls and gf["source"] == "file" and gf["note"])
    finally:
        CR._get_json = old_get
        shutil_ = __import__("shutil")
        shutil_.rmtree(tmpc, ignore_errors=True)

    # ---- Discogs as the second source (mocked HTTP only; added with the discogs-credits branch) ----
    print("\n[12b] credits: Discogs fallback (MusicBrainz timeout / empty), no-token note, per-track beats album-level, cache, token never leaks")
    import json as _json
    _env_keep = {k: os.environ.pop(k, None) for k in ("DISCOGS_TOKEN", "RAVEN_DISCOGS_ENV")}
    FAKE_TOKEN = "TOK-sEcReT-123"
    d_old_mb, d_old_dc = CR._get_json, CR._discogs_json
    d_tmp = tempfile.mkdtemp(prefix="__dc_")
    dcalls, mbcalls = [], []
    release = {"title": "Example Album", "year": 2025, "artists": [{"name": "Band Example"}],
               "tracklist": [{"position": "1", "type_": "track", "title": "Other Song"},
                             {"position": "2", "type_": "track", "title": "Sample Song",
                              "extraartists": [{"name": "Pat Per (2)", "role": "Omnichord"}]},
                             {"position": "3", "type_": "track", "title": "Third Song"}],
               "extraartists": [{"name": "Pat Per (2)", "role": "Piano", "tracks": "1-3"},
                                {"name": "Alex Album", "role": "Cello", "tracks": "2"},
                                {"name": "Sam Skip", "role": "Bass", "tracks": "1"},
                                {"name": "Mo Mixer", "role": "Mixed By"},
                                {"name": "Gil Global", "role": "Vocals [Backing]"}]}

    def fake_dc(url, token):
        dcalls.append((url, token))
        if "database/search" in url:
            return {"results": [{"id": 5, "title": "Band Example - Example Album", "year": "2025", "master_id": 9, "format": ["Album"]}]}
        if "/masters/" in url:
            return {"main_release": 77}
        if "/releases/77" in url:
            return release
        raise CR.LookupFailed("no such url in the mock")

    def mb_web(responses):
        def g(url):
            mbcalls.append(url)
            for key, val in responses.items():
                if key in url:
                    if isinstance(val, Exception):
                        raise val
                    return val
            raise CR.LookupFailed("no such url in the mock")
        return g
    try:
        # roles -> words
        check("discogs roles: instruments kept, bracket modifiers folded in, production/mix roles dropped",
              CR.discogs_words("Omnichord, Bass [Fretless]") == ["Omnichord", "Bass"] and CR.discogs_words("Vocals [Backing]") == ["Backing Vocals"]
              and CR.discogs_words("Mixed By") == [] and CR.discogs_words("Vocal Engineer") == [] and CR.discogs_words("Clarinet") == ["Clarinet"])
        ents, pos = CR.entries_from_discogs(release, "Sample Song")
        by = {e["who"]: e for e in ents}
        check("per-track extraartists beat album-level for the same person (omnichord, not piano)",
              pos == "2" and by["Pat Per"]["words"] == ["Omnichord"] and by["Pat Per"]["classes"] == ["synth"], str(ents))
        check("album-level credit counts only for the tracks it names; unrestricted ones count; mix roles never",
              "Alex Album" in by and "Sam Skip" not in by and "Gil Global" in by and "Mo Mixer" not in by, str(sorted(by)))

        # MusicBrainz times out -> Discogs answers
        os.environ["DISCOGS_TOKEN"] = FAKE_TOKEN
        CR._discogs_json = fake_dc
        CR._get_json = mb_web({"recording?": CR.LookupFailed("timed out after 6 s")})
        r1 = CR.get("/m/x.mp3", "Band Example", "Sample Song", cache_dir=d_tmp)
        check("MusicBrainz timeout falls through to Discogs; the label says which failed and which answered",
              r1["source"] == "discogs" and "MusicBrainz timed out after 6 s" in r1["label"] and "Discogs: Band Example - Example Album (2025)" in r1["label"],
              str(r1.get("label")))
        # cache hit makes no network call, remembers the source
        dcalls.clear(); mbcalls.clear()
        r1b = CR.get("/m/x.mp3", "Band Example", "Sample Song", cache_dir=d_tmp)
        check("cache hit: no network call at all, same entries and source", not dcalls and not mbcalls and r1b["source"] == "discogs" and r1b["entries"] == r1["entries"])
        cached = open(CR.cache_path("Band Example", "Sample Song", d_tmp), encoding="utf-8").read()
        check("cache records the source and never holds the token", _json.loads(cached)["source"] == "discogs" and FAKE_TOKEN not in cached)

        # MusicBrainz answers but has no performer credits -> Discogs
        dcalls.clear()
        CR._get_json = mb_web({"recording?": {"recordings": [rec("e1", "Sample Song", "Band Example")]}, "recording/e1": {"relations": []},
                               "release/re1": {"media": []}})
        r2 = CR.get("/m/y.mp3", "Band Example", "Sample Song", cache_dir=tempfile.mkdtemp(prefix="__dc2_"))
        check("MusicBrainz with no performer credits -> Discogs answers and the label says why",
              r2["source"] == "discogs" and "no performer credits" in r2["label"] and dcalls, str(r2.get("label")))
        check("token travels in the header call, never inside a URL", all(FAKE_TOKEN not in u for u, _ in dcalls) and all(tk == FAKE_TOKEN for _, tk in dcalls))

        # MusicBrainz answers with credits -> Discogs untouched
        dcalls.clear()
        CR._get_json = mb_web({"recording?": {"recordings": [rec("e2", "Sample Song", "Band Example")]},
                               "recording/e2": {"relations": [{"type": "instrument", "attributes": ["piano"], "artist": {"name": "P"}}]}})
        r3 = CR.get("/m/z.mp3", "Band Example", "Sample Song", cache_dir=tempfile.mkdtemp(prefix="__dc3_"))
        check("MusicBrainz with credits never calls Discogs", r3["source"] == "musicbrainz" and not dcalls)

        # no token -> honest skip note
        del os.environ["DISCOGS_TOKEN"]
        CR._get_json = mb_web({"recording?": CR.LookupFailed("timed out after 6 s")})
        dcalls.clear()
        d4 = tempfile.mkdtemp(prefix="__dc4_")
        r4 = CR.get("/m/w.mp3", "Band Example", "Sample Song", cache_dir=d4)
        check("no token: Discogs is skipped with one honest line, MusicBrainz's failure still named, no Discogs call",
              r4["source"] is None and "MusicBrainz timed out after 6 s" in r4["note"] and "Discogs skipped" in r4["note"] and "DISCOGS_TOKEN" in r4["note"] and not dcalls, str(r4))
        check("a failed lookup is not cached as 'no credits'", not os.path.exists(CR.cache_path("Band Example", "Sample Song", d4)))

        # both fail -> both named
        os.environ["DISCOGS_TOKEN"] = FAKE_TOKEN
        CR._discogs_json = lambda url, token: (_ for _ in ()).throw(CR.LookupFailed("timed out after 6 s"))
        r5 = CR.get("/m/v.mp3", "Band Example", "Sample Song", cache_dir=tempfile.mkdtemp(prefix="__dc5_"))
        check("both sources failing: both named, never 'no credits'", r5["entries"] == [] and "MusicBrainz timed out" in r5["note"] and "Discogs timed out" in r5["note"], str(r5))
        check("the token never appears in any note or label", FAKE_TOKEN not in str(r1) + str(r2) + str(r4) + str(r5))

        # token loader: env file, never a default path
        envf = os.path.join(d_tmp, "d.env")
        open(envf, "w").write("# c\nDISCOGS_TOKEN='" + FAKE_TOKEN + "'\n")
        del os.environ["DISCOGS_TOKEN"]
        check("no env and no file path -> no token", CR.discogs_token() is None)
        os.environ["RAVEN_DISCOGS_ENV"] = envf
        check("RAVEN_DISCOGS_ENV names a file with a DISCOGS_TOKEN= line", CR.discogs_token() == FAKE_TOKEN)
    finally:
        CR._get_json, CR._discogs_json = d_old_mb, d_old_dc
        for k in ("DISCOGS_TOKEN", "RAVEN_DISCOGS_ENV"):
            os.environ.pop(k, None)
            if _env_keep[k] is not None:
                os.environ[k] = _env_keep[k]
        __import__("shutil").rmtree(d_tmp, ignore_errors=True)

    # the tie-break
    cr_all = ["piano", "synth", "electric guitar", "drum machine", "vocals"]
    check("tie-break: 'keys (organ or piano?)' + credited piano -> 'piano (credited)'", CR.resolve("keys (organ or piano?)", cr_all) == "piano (credited)")
    check("tie-break: both candidates credited -> neither is picked", CR.resolve("keys (organ or piano?)", ["organ", "piano"]) == "organ or piano (both credited)")
    check("tie-break: keys with no matching credit stay as they were", CR.resolve("keys (organ or piano?)", ["synth", "electric guitar"]) is None)
    check("tie-break: only a 'keyboards' credit gives 'keyboard (credited)'", CR.resolve("keys (organ or piano?)", ["keyboard"]) == "keyboard (credited)")
    check("tie-break: unconfirmed pad + credited synth -> 'synth (credited)'", CR.resolve("synth/pad-like layer (unconfirmed)", cr_all) == "synth (credited)")
    check("tie-break: guitar (type unclear) + credited electric guitar -> 'electric guitar (credited)'", CR.resolve("guitar (type unclear)", cr_all) == "electric guitar (credited)")
    check("tie-break: both electric and acoustic credited -> left unclear", CR.resolve("guitar (type unclear)", ["electric guitar", "acoustic guitar"]) is None)
    check("tie-break: a confident tag is never overruled (drum machine, 'organ', 'acoustic guitar' stay)",
          CR.resolve("drum machine", cr_all) is None and CR.resolve("organ", ["piano"]) is None and CR.resolve("acoustic guitar", ["electric guitar"]) is None)
    check("tie-break: a different guitar than the tagger leaned to does not win", CR.resolve("acoustic guitar (probably)", ["electric guitar"]) is None
          and CR.resolve("electric guitar (probably)", ["electric guitar"]) == "electric guitar (credited)")
    check("tie-break: synth-like bass only becomes 'synth bass' when synth bass is credited, not for a plain synth credit",
          CR.resolve("bass (synth-like)", ["synth"]) is None and CR.resolve("bass (synth-like)", ["synth bass"]) == "synth bass (credited)")
    check("tie-break: 'bass (type unclear)' + credited bass guitar; 'drums (type unclear)' + drum programming",
          CR.resolve("bass (type unclear)", ["bass guitar"]) == "bass guitar (credited)" and CR.resolve("drums (type unclear)", ["drum machine"]) == "drum machine (credited)")
    check("unnamed melodic layer: one credited melodic candidate names it; several leave it nameless",
          CR.resolve_unnamed(["synth", "drum machine", "vocals"]) == "synth (credited)" and CR.resolve_unnamed(["synth", "piano"]) is None)

    # a hand-made instruments result: 100 s, 2 s hop = 50 windows (hop 2.0)
    def fake_res(**kw):
        res = {"version": 5, "duration": 100.0, "hop": 2.0, "n": 50, "unclear": {"other": 0.5, "drums": 0.0, "bass": 0.0}, "hints": {},
               "other_on": [(0.0, 100.0)], "vocal_on": [(10.0, 90.0)], "other_character": {"label": "synth/pad-like layer (unconfirmed)", "feat": {}, "hint": 0.1},
               "families": {"keys (organ or piano?)": {"stem": "other", "spans": [(20.0, 30.0)], "conf": 0.43, "peak": 0.5, "share": 0.1},
                            "drum machine": {"stem": "drums", "spans": [(0.0, 100.0)], "conf": 0.5, "peak": 0.6, "share": 1.0},
                            "bass (synth-like)": {"stem": "bass", "spans": [(10.0, 60.0)], "conf": 0.0, "peak": 0.0, "share": 0.5, "timbre_only": True}}}
        res.update(kw)
        return res
    crd = {"source": "file", "label": "credits file", "note": None,
           "entries": CR.parse_credit_text("A: piano, synth\nB: electric guitar\nC: vocals")}
    out = CR.apply(fake_res(), crd)
    L = "\n".join(out["credit_lines"])
    check("apply: tie-breaks name the ears' own windows ('piano (credited)', 'synth (credited)') and keep what the ears heard",
          "piano (credited)" in out["families"] and out["families"]["piano (credited)"]["credited_from"] == "keys (organ or piano?)"
          and out["other_character"]["label"] == "synth (credited)" and "keys (organ or piano?)" not in out["families"])
    check("apply: the three honest parts (credited+heard with times / credited, not clearly heard / heard, not credited)",
          "credited and heard:" in L and "piano (A): 0:20–0:30" in L and "synth (A): " in L and "vocals (C): 0:10–1:30" in L
          and "electric guitar (B): credited, not clearly heard" in L and "heard, not credited:" in L
          and "bass (synth-like): not credited; possibly synth bass" in L and "drum machine: not credited" in L, L)
    check("apply: the original result is not mutated (the cache stays the ears' own)", "keys (organ or piano?)" in fake_res()["families"] and
          crd["entries"][0]["classes"] == ["piano", "synth"])
    # never inventing
    silent = fake_res(families={"drum machine": fake_res()["families"]["drum machine"]}, other_on=[], other_character=None, vocal_on=[],
                      unclear={"other": 0.0, "drums": 0.0, "bass": 0.0})
    o2 = CR.apply(silent, crd)
    check("never inventing: a silent melodic stem gains no credited instrument; credited ones are 'not clearly heard'",
          list(o2["families"]) == ["drum machine"] and o2["other_character"] is None and "piano (A): credited, not clearly heard" in "\n".join(o2["credit_lines"])
          and "synth (A): credited, not clearly heard" in "\n".join(o2["credit_lines"]) and "vocals (C): credited, not clearly heard" in "\n".join(o2["credit_lines"]),
          "\n".join(o2["credit_lines"]))
    quiet = fake_res(other_on=[], other_character=None)
    o3 = CR.apply(quiet, {"entries": CR.parse_credit_text("synth"), "label": "credits file"})
    check("never inventing: no energy on the other stem -> a lone credited synth is not placed anywhere",
          "synth (credited)" not in o3["families"] and o3["other_character"] is None)
    nameless = fake_res(other_character={"label": None, "feat": {}, "hint": 0.0})
    o4 = CR.apply(nameless, {"entries": CR.parse_credit_text("synth"), "label": "credits file"})
    o5 = CR.apply(nameless, crd)
    check("unnamed layer with energy: named only when the credits leave exactly one melodic candidate", o4["other_character"]["label"] == "synth (credited)"
          and o5["other_character"]["label"] is None)
    # label renders in the card
    import instruments as INS2
    card = "\n".join(INS2.block(out))
    check("card: who-is-playing uses the credited names and the CREDITS block follows it",
          "synth (credited) + voice" in card and card.index("who is playing") < card.index("CREDITS (credits file") and 'the ears heard "keys (organ or piano?)"' in card, card[-400:])
    evs = " ".join(x for _, x in INS2.events(out))
    check("timeline events carry the credited name", "piano (credited) enters" in evs, evs)
    # notes
    on_ = CR.apply(fake_res(), {"entries": [], "note": "MusicBrainz unreachable (timed out); ears only"})
    check("no credits found: the card gets one honest line and the ears' result is untouched",
          on_["credit_lines"] == ["CREDITS: MusicBrainz unreachable (timed out); ears only"] and "keys (organ or piano?)" in on_["families"])
    check("--no-credits: apply leaves the result alone, no CREDITS line", CR.apply(fake_res(), {"off": True, "entries": [], "note": None}) == fake_res())
    CR.configure_from_args(argparse.Namespace(credits=None, no_credits=True, artist=None, title=None))
    check("--no-credits reaches the module's run configuration", CR.for_song("/m/off.mp3").get("off") is True)
    CR.configure_from_args(argparse.Namespace(credits=None, no_credits=False, artist=None, title=None))

    # ---- key fusion: signature vs tonic, voters, bass root constraint, loop vote, madmom order (synthetic; no network) ----
    print("\n[14] key fusion: signature and tonic as two parts, voters, bass root constraint, loop vote")
    import keyfuse as KF
    import music as MU

    def lp_for(*pairs):                       # a voter that favours the given key indices, flat otherwise
        v = np.zeros(24)
        for k, w in pairs:
            v[k] += w
        return KF._norm(v)
    A_MAJ, FS_MIN, B_MIN, D_MAJ = 9, 12 + 6, 12 + 11, 2
    check("key: A major and F# minor share a signature (3 sharps); Ab major is 4 flats; C is none",
          KF.sig_text(A_MAJ) == "3♯" and KF.sig_text(FS_MIN) == "3♯" and KF.sig_text(8) == "4♭" and KF.sig_text(0) == "no ♯/♭", KF.sig_text(8))
    check("key: MIREX relations (same / fifth / relative / parallel / other) and weights",
          KF.relation(FS_MIN, A_MAJ) == "relative" and KF.relation(4, A_MAJ) == "fifth" and KF.relation(12 + 9, A_MAJ) == "parallel"
          and KF.relation(B_MIN, A_MAJ) == "other" and KF.mirex_score(FS_MIN, A_MAJ) == 0.3 and KF.mirex_score(A_MAJ, A_MAJ) == 1.0
          and KF.mirex_score(4, A_MAJ) == 0.5 and KF.mirex_score(12 + 9, A_MAJ) == 0.2 and KF.mirex_score(B_MIN, A_MAJ) == 0.0)
    try:
        import mir_eval
        check("key: our formula agrees with mir_eval.key.weighted_score on the relations",
              all(abs(KF.mirex_score(KF.NAMES.index(e.split()[0]) + (0 if e.endswith("major") else 12), A_MAJ) - mir_eval.key.weighted_score("A major", e)) < 1e-9
                  for e in ("A major", "F# minor", "E major", "A minor", "B minor")))   # mir_eval 0.8 scores only a fifth ABOVE the reference; MIREX since 2017 also takes one below (ours does)
    except ImportError:
        info("mir_eval", "not installed: formula checked by hand only")
    check("key: madmom's order (A, Bb, B, C ... then A minor ...) maps to ours (C ... B; minors after)",
          np.argmax(KF.madmom_to_ours([1] + [0] * 23)) == A_MAJ and np.argmax(KF.madmom_to_ours([0] * 12 + [1] + [0] * 11)) == 12 + 9
          and np.argmax(KF.madmom_to_ours([0, 1] + [0] * 22)) == 10)
    vc = KF.verdict({"spectral": lp_for((A_MAJ, 4)), "cnn": lp_for((A_MAJ, 8)), "chords": lp_for((A_MAJ, 6)), "bass": lp_for((A_MAJ, 4))})
    check("key: agreeing voters -> one KEY line with signature and tonic, no votes line",
          vc["state"] == "clear" and vc["line"].startswith("A major (3♯) · tonic") and vc["votes_line"] is None, vc["line"])
    vt = KF.verdict({"spectral": lp_for((FS_MIN, 2), (A_MAJ, 1.9)), "chords": lp_for((A_MAJ, 2), (FS_MIN, 1.9)), "bass": lp_for((FS_MIN, 1), (A_MAJ, 0.9))})
    check("key: relative keys close -> the signature is claimed, the tonic is not, and the card says so in one line",
          vt["state"] == "tonic" and vt["key"] is None and vt["line"].startswith("3♯ (A major or F# minor") and "tonic unclear" in vt["line"]
          and "\n" not in vt["line"] and vt["votes_line"], str(vt))
    vs = KF.verdict({"spectral": lp_for((A_MAJ, 2), (D_MAJ, 1.9)), "chords": lp_for((D_MAJ, 2), (A_MAJ, 1.9))})
    check("key: a close call between different signatures says 'unclear between' with both", vs["state"] == "split" and "unclear between" in vs["line"]
          and "A major" in vs["line"] and "D major" in vs["line"], vs["line"])
    vd = KF.verdict({"spectral": lp_for((FS_MIN, 3)), "cnn": lp_for((A_MAJ, 9)), "chords": lp_for((A_MAJ, 6)), "bass": lp_for((A_MAJ, 4))})
    check("key: a heavyweight voter that dissents puts the votes on one evidence line", vd["votes_line"] and "spectral F#m" in vd["votes_line"], str(vd["votes_line"]))
    check("key: fusion works without the CNN voter (it is optional)", KF.verdict({"spectral": lp_for((A_MAJ, 3)), "chords": lp_for((A_MAJ, 3))})["working"] == "A major")
    # chord voter: the Bm D A E loop is A major, not B minor / D major (the misheard G is what made it B minor)
    loop = [{"t0": 3.0 * i, "t1": 3.0 * i + 3.0, "root": r, "q": q} for i, (r, q) in enumerate([(11, "min"), (2, "maj"), (9, "maj"), (4, "maj")] * 6)]
    Lc, _ = KF.v_chords(loop)
    top2 = [int(k) for k in np.argsort(Lc)[::-1][:2]]
    check("key: chord voter reads Bm D A E as A major first, never B minor or D major",
          top2[0] == A_MAJ and B_MIN not in top2 and D_MAJ not in top2, str(top2))
    wrong = [dict(e, root=(7 if e["root"] == 4 else e["root"]), q="maj") for e in loop]       # G for E: the old mishearing
    check("key: the misheard G for E (G A Bm D) moves the chord voter off A major (what the bass constraint prevents)",
          int(np.argmax(KF.v_chords(wrong)[0])) != A_MAJ)
    check("key: a 3-chord loop is little evidence (distinct chords scale it)",
          np.ptp(KF.v_chords(loop[:3] * 4)[0]) <= np.ptp(KF.v_chords(loop)[0]) + 1e-9)
    # bass voter: tonic-heavy bass, downbeats and the final note count
    fps = 43.0
    bpc = np.repeat(np.array([9, 9, 4, 2, 9, 9, 11, 4]), 40).astype(np.int8)
    Lb, bi = KF.v_bass(bpc, np.ones(len(bpc)), fps, downbeats=[0.0, 3.7], ends=[len(bpc) / fps - 1], dur=len(bpc) / fps)
    check("key: bass voter puts the bass-heavy tonic first (A major/minor side) and reports its coverage", int(np.argmax(Lb)) in (A_MAJ, 12 + 9) and bi["cover"] > 0.9, str(bi))
    check("key: no bass stem energy -> no bass vote", KF.v_bass(np.full(500, -1, dtype=np.int8), np.zeros(500), fps)[0] is None)
    # melody voter: long phrase-end notes
    hop_s = 0.046
    f0 = []
    for note in (440.0, 554.4, 440.0, 659.3, 440.0, 554.4):                      # A, C#, A, E, A, C#: each 0.8 s then a 0.5 s breath
        f0 += [note] * int(0.8 / hop_s) + [np.nan] * int(0.5 / hop_s)
    rn = KF.resting_notes(f0, hop_s)
    check("key: resting notes = long notes followed by a pause (A C# A E A C#)", [p_ for p_, _ in rn] == [9, 1, 9, 4, 9, 1], str(rn))
    check("key: melody voter leans A major on those resting notes", int(np.argmax(KF.v_melody(f0, hop_s)[0])) == A_MAJ)
    check("key: too few resting notes -> no melody vote", KF.v_melody([440.0] * 30 + [np.nan] * 30, hop_s)[0] is None)
    # cadence voter: V->I counts; a loop that never lands V->I gives no vote at all (flat, and that is fine)
    cad = [{"t0": 0, "t1": 2, "root": 7, "q": "maj"}, {"t0": 2, "t1": 4, "root": 0, "q": "maj"}] * 3
    Lcad, ci = KF.v_cadence(cad)
    check("key: V->I landings (G to C, three times) vote C major", int(np.argmax(Lcad)) == 0 and ci["landings"] >= 3, str(ci))
    check("key: a loop with no V->I landing (Bm D A E) gives the cadence voter nothing to say on A", KF.v_cadence(loop)[0] is None or int(np.argmax(KF.v_cadence(loop)[0])) != A_MAJ)
    if audio_tests():
        # the decoder: bass stem as a root constraint (synthetic chroma where the E bar smears toward G: shared note B only)
        fps_h = HM.FPS

        def synth_F(chord_pcs, bass_pcs, n_per=40, reps=6):
            cols, bcols, bpcs = [], [], []
            for _ in range(reps):
                for pcs, b in zip(chord_pcs, bass_pcs):
                    for _ in range(n_per):
                        c_ = np.zeros(12)
                        c_[pcs] = 1.0 / 3.0
                        bc = np.zeros(12)
                        bc[b] = 1.0
                        cols.append(c_)
                        bcols.append(bc)
                        bpcs.append(b)
            n = len(cols)
            return {"chroma": np.array(cols).T, "bass": np.array(bcols).T, "rms_db": np.full(n, -20.0),
                    "bass_trk": {"pc": np.array(bpcs, dtype=np.int8), "rel": np.ones(n, np.float32)}}, n / fps_h
        Fs, dur_s = synth_F([[11, 2, 6], [2, 6, 9], [9, 1, 4], [11, 7, 2]], [11, 2, 9, 4])       # Bm D A, then (B G D) over a bass E
        a_ = HM.decode_chords(Fs, None, (9, "major"), dur_s, bass_pen=False, loop=False)
        b_ = HM.decode_chords(Fs, None, (9, "major"), dur_s, bass_pen=True, loop=False)
        chk = lambda evs: " ".join(HM.chord_name(e["root"], e["q"]) for e in evs[:4] if e["root"] is not None)
        check("decoder: without the bass stem the smeared fourth bar is not an E chord; with it (bass on E) the chord is rooted on E",
              a_[3]["root"] != 4 and b_[3]["root"] == 4, f"without: {chk(a_)} | with: {chk(b_)}")
        check("decoder: bass constraint is a no-op without a bass track",
              [(e["root"], e["q"]) for e in HM.decode_chords(dict(Fs, bass_trk=None), None, (9, "major"), dur_s, loop=False)] == [(e["root"], e["q"]) for e in a_])

    Ul = np.tile(np.eye(12)[[0, 3, 7, 9]], (6, 1)) + 0.02
    Ul = Ul / np.linalg.norm(Ul, axis=1, keepdims=True)
    check("decoder: the period of a 4-unit loop repeated 6 times is found (4)", HM.loop_period(Ul, lo=3, hi=10) == 4)
    lp0 = np.log(np.full((24, 5), 0.01))
    lp0[:, 0] = np.log(0.96)                                # every unit says state 0 ...
    lp0[5] = np.log(0.01)
    lp0[5, 3] = np.log(0.96)                                # ... except unit 5, misheard as state 3
    voted = HM.loop_vote(Ul, lp0, 4)
    check("decoder: loop vote outvotes one misheard repeat of a loop position", int(np.argmax(voted[5])) == 0 and int(np.argmax(lp0[5])) == 3)
    # madmom: subprocess in its own venv; the suite needs neither the venv nor audio
    if KF.madmom_python():
        info("madmom", f"found at {KF.madmom_python()} (not run in the suite: it needs a real audio file)")
    else:
        info("madmom", "no ./venv-madmom: the CNN voter is skipped, everything else runs")
    old_py = os.environ.get("RAVEN_MADMOM_PY")
    os.environ["RAVEN_MADMOM_PY"] = "/nonexistent/python"
    try:
        if a.fast:
            # Otherwise an installed ./venv-madmom is tried after the broken override.
            from unittest.mock import patch
            with patch.object(KF, "madmom_python", return_value="/nonexistent/python"):
                r_ = KF.cnn_probs("/nonexistent.mp3")
        else:
            r_ = KF.cnn_probs("/nonexistent.mp3")
    finally:
        if old_py is None:
            os.environ.pop("RAVEN_MADMOM_PY", None)
        else:
            os.environ["RAVEN_MADMOM_PY"] = old_py
    check("key: a broken madmom python gives None (voter skipped), not an exception", r_ is None or isinstance(r_, list))
    check("card: the KEY line is the claim, and the one votes line follows it only when there is one",
          MU.format_card({"file": "x.mp3", "duration": 60.0, "key_claim": vd}).count("KEY") == 2
          and MU.format_card({"file": "x.mp3", "duration": 60.0, "key_claim": vc}).count("KEY") == 1)

    # ---- 1 Nov 2026 (Example Track): upload words off the title, a fade can't make a full-band section "quiet", guitar steps ----
    print("\n[15] title cleaning for the credits lookup; section level ignores silence; guitar-stem steps")
    cl = CR.clean_title
    check("clean_title: '(Lyric Video)' goes, '(Example's Version)' stays",
          cl("Example Track (Example's Version) (Lyric Video)") == "Example Track (Example's Version)", cl("Example Track (Example's Version) (Lyric Video)"))
    check("clean_title: official/audio/visualizer/HD/4K/remastered noise, any case, () and []",
          [cl(x) for x in ("Song (Official Audio)", "Song [OFFICIAL LYRIC VIDEO]", "Song (official music video)", "Song (Official Video)",
                           "Song (Audio)", "Song (Visualizer)", "Song HD", "Song [4K]", "Song (2011 Remastered)", "Song (Remastered)", "Song - 4K")]
          == ["Song"] * 11, str([cl(x) for x in ("Song (Official Audio)", "Song [OFFICIAL LYRIC VIDEO]", "Song HD", "Song (2011 Remastered)")]))
    check("clean_title: meaningful version markers survive (Live, Acoustic, Remix, Example's Version), even beside noise",
          cl("Song (Live) [Official Video]") == "Song (Live)" and cl("Song (Acoustic)") == "Song (Acoustic)"
          and cl("Song (Remix) (Lyric Video)") == "Song (Remix)" and cl("Hello (Example's Version)") == "Hello (Example's Version)")
    check("clean_title: words inside a name are not noise; an all-noise title is left alone",
          cl("Audio Girl") == "Audio Girl" and cl("HD Radio Blues") == "HD Radio Blues" and cl("(Lyric Video)") == "(Lyric Video)")
    check("cache key keeps bracketed version words ('Song' and 'Song (Live)' are different questions); case/punctuation do not matter",
          CR.cache_path("A", "Song", "/c") != CR.cache_path("A", "Song (Live)", "/c") and CR.cache_path("A", "Song!", "/c") == CR.cache_path("a", "song", "/c"))
    check("drop_version: '(Example's Version)' comes off; nothing to drop -> None; (Live) is not a version marker",
          CR.drop_version("Example Track (Example's Version)") == "Example Track" and CR.drop_version("Example Track") is None and CR.drop_version("Song (Live)") is None)

    tmpv = tempfile.mkdtemp(prefix="__cr2_")
    callsv = []
    fr = lambda i, title, art: {"id": i, "title": title, "score": 100, "disambiguation": "", "artist-credit": [{"name": art}],
                                "releases": [{"id": "r" + i, "status": "Official", "release-group": {"primary-type": "Album"}}]}
    perf = {"relations": [{"type": "instrument", "attributes": ["electric guitar"], "artist": {"name": "G. Player"}}]}
    old_get2 = CR._get_json
    try:
        def fw(table):                        # fake web: answers by the quoted title in the search; records every URL
            def f(url):
                callsv.append(url)
                if "recording?" in url:
                    q = urllib_parse_unquote(url)
                    for k, v in table.items():
                        if f'recording:"{k}"' in q:
                            return v
                    return {"recordings": []}
                return perf
            return f
        from urllib.parse import unquote_plus as urllib_parse_unquote
        CR._get_json = fw({"Example Track (Example's Version)": {"recordings": [fr("tv", "Example Track (Example's Version)", "Example Artist")]}})
        v1 = CR.get("/m/Example Artist - Example Track (Example's Version) (Lyric Video).mp3", "Example Artist", "Example Track (Example's Version) (Lyric Video)", cache_dir=tmpv)
        check("lookup searches the CLEANED title, never the upload noise",
              v1["source"] == "musicbrainz" and "Lyric Video" not in callsv[0].replace("%20", " ").replace("+", " ") and "Version" in urllib_parse_unquote(callsv[0]), callsv[0])
        check("a clean match on '(Example's Version)' says nothing about dropping it", "without the" not in v1["label"])
        callsv.clear()
        CR._get_json = fw({"Example Track": {"recordings": [fr("orig", "Example Track", "Example Artist")]}})
        v2 = CR.get("/m/f.mp3", "Example Artist", "Example Track (Example's Version) (Lyric Video)", cache_dir=tempfile.mkdtemp(prefix="__cr3_"))
        searches = [u for u in callsv if "recording?" in u]
        check("no match on '(Example's Version)' -> asked once more without it, and the card says so",
              v2["source"] == "musicbrainz" and len(searches) == 2 and "recording:\"Example Track\"" in urllib_parse_unquote(searches[1])
              and "without the '(... Version)' marker" in v2["label"], str(v2.get("label")))
        callsv.clear()
        CR._get_json = fw({})
        v3 = CR.get("/m/g.mp3", "Example Artist", "Example Track (Example's Version) (Lyric Video)", cache_dir=tempfile.mkdtemp(prefix="__cr4_"))
        check("no match at all -> one honest line that names what was searched",
              v3["source"] is None and "no match" in v3["note"] and "searched as 'Example Track (Example's Version)'" in v3["note"], str(v3["note"]))
        callsv.clear()
        v4 = CR.get("/m/h.mp3", "Band", "Song (Live)", cache_dir=tempfile.mkdtemp(prefix="__cr5_"))
        check("a (Live) title with no match is searched once (only a '... Version' marker earns the second try)",
              len([u for u in callsv if "recording?" in u]) == 1, str(callsv))
    finally:
        CR._get_json = old_get2

    if audio_tests():
        # sections: 40 s moderate, 53 s loud with a dip, 7 s of digital silence at the end -> the last section is not "quiet"
        import librosa
        sr_s = MU.SR
        ts = np.arange(int(100 * sr_s)) / sr_s
        amp = np.where(ts < 40, 0.05, 0.18)
        amp = np.where((ts > 70) & (ts < 74), 0.02, amp)
        amp = np.where(ts >= 93, 0.0, amp)
        ys = (amp * np.sin(2 * np.pi * np.where(ts < 40, 220.0, 330.0) * ts)).astype(np.float32)
        S_s = librosa.stft(ys, hop_length=MU.HOP)
        chroma_s = librosa.feature.chroma_cqt(y=ys, sr=sr_s, hop_length=MU.HOP)
        rdb = 20 * np.log10(librosa.feature.rms(S=np.abs(S_s), hop_length=MU.HOP)[0] + 1e-6)
        secs_s = MU.analyze_sections(ys, chroma_s, rdb, 100.0)
        last = secs_s[-1]
        check("section level is the median of frames with sound: a 7 s digital-silence tail does not make a loud last section 'quiet'",
              last["level"] != "quiet" and last["db"] > -30, str(secs_s))
        check("the shape line never calls that return a quiet outro", "quiet outro" not in MU.shape_phrase(secs_s), MU.shape_phrase(secs_s))

    # guitar stem: a one-go step up that holds is reported; a single loud hit and a steady stem are not
    import instruments as INS
    srg = INS.SR
    tg = np.arange(int(60 * srg)) / srg
    rng = np.random.default_rng(3)
    nz = rng.standard_normal(len(tg)).astype(np.float32)
    step = (nz * np.where(tg < 30, 0.01, 0.12)).astype(np.float32)
    gs = INS.guitar_surges(step, srg, -22.0, np.where((np.arange(60) >= 28) & (np.arange(60) <= 31), 0.45, 0.05))
    check("guitar step: ~21 dB step at 30 s is found near 30 s, with the tagger's electric score before/after",
          len(gs) == 1 and abs(gs[0]["t"] - 30.0) <= 1.0 and gs[0]["db"] > 15 and gs[0]["elec_after"] >= 0.4 and gs[0]["elec_before"] <= 0.1, str(gs))
    hit = (nz * 0.05).astype(np.float32)
    hit[int(30 * srg):int(30.4 * srg)] *= 20
    check("guitar step: one loud hit is not a step; a steady stem has none",
          INS.guitar_surges(hit, srg, -22.0) == [] and INS.guitar_surges((nz * 0.05).astype(np.float32), srg, -22.0) == [])
    check("guitar step: a rise inside an empty stem (far under its loud level) is ignored",
          INS.guitar_surges(step, srg, 20.0) == [])
    evg = INS.events({"duration": 60.0, "families": {}, "guitar_surges": gs})
    check("events: the step becomes a timeline line with its level source, and the electric reading is marked local/unconfirmed",
          len(evg) == 1 and "guitar steps up" in evg[0][1] and "guitar stem level" in evg[0][1] and "type unconfirmed" in evg[0][1], str(evg))
    # two strums, a short gap, a rest, then the held band: 40 s quiet, hits at 30.0 and 31.2 (0.4 s each), silence to 32.5, held from 32.5
    tt2 = np.arange(int(60 * srg)) / srg
    a2 = np.where(tt2 < 30, 0.01, 0.0)
    a2 = np.where(((tt2 >= 30.0) & (tt2 < 30.4)) | ((tt2 >= 31.2) & (tt2 < 31.6)) | (tt2 >= 32.5), 0.12, a2)
    two = (rng.standard_normal(len(tt2)) * a2).astype(np.float32)
    g2 = INS.guitar_surges(two, srg, -22.0)
    check("guitar arrival: two separate strums before the held entrance are told apart, with the hold time",
          len(g2) == 1 and len(g2[0]["hits"]) == 2 and abs(g2[0]["hits"][0] - 30.0) <= 0.3 and abs(g2[0]["hits"][1] - 31.25) <= 0.3
          and abs(g2[0]["sustained_from"] - 32.5) <= 0.3, str(g2))
    ev2 = INS.events({"duration": 60.0, "families": {}, "guitar_surges": g2})
    check("events: 'arrives as 2 separate hits (...) and holds from ...' carries the stem-level source",
          len(ev2) == 1 and "2 separate hits" in ev2[0][1] and "holds from 0:32.5" in ev2[0][1], str(ev2))
    evn = INS.events({"duration": 60.0, "families": {"guitar (type unclear)": {"stem": "guitar", "spans": [(30.0, 55.0)], "conf": 0.5, "peak": 0.6, "share": 0.4}},
                      "guitar_surges": gs})
    check("events: a surge beside an existing 'guitar enters' is not said twice", sum(1 for t, x in evn if "guitar" in x and ("steps" in x or "enters" in x)) == 1, str(evn))

    # ---- ACHE hook: fixture/text arithmetic only, including --fast ----------------
    import test_ache
    test_ache.run_checks(check, tmp)

    # ---- whisper integration (info only) -----------------------------------------
    if not a.no_whisper_test and audio_tests():
        print("\n[10] (info) own-Whisper words on the composite")
        try:
            rw = R.analyze(cpath, words=None, use_whisper=True, events=False, affect=False)
            gw = rw["data"]["gaps"]
            info("whisper path", f"{len(rw['data']['words'] or [])} words; WORDLESS {gw['wordless_s']}s, SILENT {gw['silent_s']}s; "
                 + "; ".join(n for n in rw["notes"] if n.startswith("words")))
            print("  ", (lines_with(rw, "WORDLESS") or ["(none)"])[0].strip())
        except Exception as e:
            info("whisper path", f"unavailable: {type(e).__name__}: {e}")

    # ---- eval shelf: isolated text/JSON checks (no audio or network) -------------
    from test_eval_shelf import run_checks as run_shelf_checks
    run_shelf_checks(check)

    # ---- listening modes: text-only checks, including --fast ---------------------
    import test_modes
    test_modes.run_tests(check)

    # ---- listening sessions: isolated JSON/wrapper checks, including --fast ----
    import test_listening_sessions
    test_listening_sessions.run_tests(check)

    # ---- saved songs: fake passes and disposable homes only, including --fast --
    import test_listen
    test_listen.run_tests(check)

    # ---- overnight queue: fake subprocesses and temporary state, also --fast ----
    import test_ears_queue
    test_ears_queue.run_tests(check)

    # ---- slim card hook: saved/synthetic text only, including --fast ------------
    import test_slim
    test_slim.run_checks(check, tmp)

    import test_meter_trust
    test_meter_trust.run_tests(check)

    import test_key_opening
    test_key_opening.run_tests(check)

    import test_key_changes
    test_key_changes.run_tests(check)

    # ---- LRCLIB: mocked HTTP + synthetic timing only, including --fast -----------
    import test_lrclib
    test_lrclib.run_tests(check)
    import test_song_length
    test_song_length.run_tests(check)

    # ---- Jev: mocked HTTP and saved lyric cards only, including --fast ---------
    import test_jev_reads
    test_jev_reads.run_tests(check)
    import test_jev_more
    test_jev_more.run_tests(check)
    import test_song_extras
    test_song_extras.run_tests(check)
    import test_feeling_card
    test_feeling_card.run_tests(check)

    # ---- MORE EARS: model-free synthetic probes (also run under --fast) ----------
    from test_more_ears import run_checks as more_ears_checks
    more_ears_checks(check)

    # ---- HONEST CARD: synthetic statuses and mocked failures, including --fast ---
    from test_stage_status import run_checks as status_checks
    status_checks(check, tmp)
    from test_honest_bugs import run_checks as honest_bug_checks
    honest_bug_checks(check, tmp)
    from test_cpu_profile import run_checks as cpu_profile_checks
    cpu_profile_checks(check)
    from test_voice_book import run_checks as voice_book_checks
    voice_book_checks(check)

    from test_bandout import run_checks as bandout_checks
    bandout_checks(check, tmp)

    # ---- STOPS: synthetic cached stems, no inference or network -----------------
    from test_stops import run_checks as stops_checks
    stops_checks(check, tmp)

    # ---- summary ---------------------------------------------------------------
    if not a.fast:
        R.release_models()
    if skipped:
        print(f"\n--fast: skipped {skipped} audio/model test groups (including integration info).")
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n=== {sum(1 for r in RESULTS if r[1] == 'PASS')} passed, {len(fails)} failed, "
          f"{sum(1 for r in RESULTS if r[1] == 'INFO')} info  ({time.time() - t_start:.0f}s) ===")
    for f in fails:
        print("  FAIL:", f[0], f[2])
    return 1 if fails else 0


def main():
    # Release tests never read recognition hints or baselines from a real user home.
    from unittest.mock import patch
    with tempfile.TemporaryDirectory(prefix="ears_test_home_") as home:
        with patch.dict(os.environ, {"RAVEN_EARS_HOME": home}):
            RESULTS.clear()
            return _main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
