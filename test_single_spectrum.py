"""Single-note listening sheets from generated samples and fake inference."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import listening_sheet as S
import raven_ears_plus as R
from test_cpu_profile import card_fixture


class SingleSpectrumTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name) / "sheet.png"
        self.data = dict(duration_s=1., words=[("Signal", .2, .4)],
                         events=dict(events=[dict(start=.45, end=.6, label="Tap")]),
                         gaps=dict(silent=[(.7, .9)]))
        times = np.arange(100) / 100
        self.measured = dict(times=times, rms_db=np.full_like(times, -15.),
                             f0=np.where(times < .7, 250., np.nan),
                             usual_pitch=dict(low_hz=230., high_hz=270., notes=10))

    def assert_png_dimensions(self):
        png = self.output.read_bytes()
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (2400, 1200))

    def test_synthetic_tone_draws_spectrum_above_pitch_events_and_words(self):
        from matplotlib.collections import QuadMesh

        sr = 16000
        samples = .25 * np.sin(2 * np.pi * 250 * np.arange(sr) / sr)
        samples[int(.7 * sr):int(.9 * sr)] = 0
        self.measured["spectrum"] = S.speech_spectrum(samples, sr)
        finish = S._finish

        def inspect(figure, axes, duration, output, title, subtitle, footer):
            self.assertEqual([ax.get_ylabel() for ax in axes],
                             ["Spectrum (Hz)", "Pitch (Hz)", "Card events", "Words"])
            spec, pitch, events, words = axes
            self.assertGreater(spec.get_position().y0, pitch.get_position().y1)
            self.assertEqual(spec.get_yscale(), "log")
            mesh, = spec.collections
            self.assertIsInstance(mesh, QuadMesh)
            self.assertEqual(mesh.get_cmap().name, "magma")
            self.assertEqual(mesh.get_clim(), (-75, -15))
            self.assertEqual(len(figure.axes), 5)
            loud = figure.axes[-1]
            self.assertEqual(loud.get_ylabel(), "RMS (dB)")
            self.assertEqual(loud.yaxis.get_label_position(), "right")
            self.assertEqual(loud.get_position().bounds, spec.get_position().bounds)
            self.assertEqual(loud.lines[0].get_color(), S.GREEN)
            np.testing.assert_array_equal(loud.lines[0].get_ydata(), self.measured["rms_db"])
            self.assertTrue(any("usual median pitch" in p.get_label() for p in pitch.patches))
            self.assertIn("Tap", [text.get_text() for text in events.texts])
            self.assertIn("Signal", [text.get_text() for text in words.texts])
            self.assertIn("spectrogram", subtitle)
            self.assertIn("green line: measured RMS", footer)
            return finish(figure, axes, duration, output, title, subtitle, footer)

        with patch.object(S, "_finish", side_effect=inspect) as checked:
            S.speech_sheet(self.data, self.measured, self.output)
            checked.assert_called_once()
        self.assert_png_dimensions()

    def test_missing_or_empty_spectrum_keeps_the_energy_layout(self):
        finish = S._finish

        def inspect(figure, axes, duration, output, title, subtitle, footer):
            self.assertEqual([ax.get_ylabel() for ax in figure.axes],
                             ["Energy (dB)", "Pitch (Hz)", "Card events", "Words"])
            self.assertEqual(axes[0].get_gridspec().get_height_ratios(), [2, 1.6, 1.2, 2.1])
            self.assertEqual(len(axes[0].collections), 0)
            self.assertEqual(axes[0].lines[0].get_linewidth(), 1.3)
            np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), self.measured["rms_db"])
            self.assertEqual(subtitle, "Listening sheet · existing speech measurements")
            self.assertTrue(footer.startswith("Energy: the card's RMS"))
            return finish(figure, axes, duration, output, title, subtitle, footer)

        with patch.object(S, "_finish", side_effect=inspect) as checked:
            S.speech_sheet(self.data, self.measured, self.output)
            legacy = self.output.read_bytes()
            self.measured["spectrum"] = S.speech_spectrum(np.array([]), 16000)
            S.speech_sheet(self.data, self.measured, self.output)
            self.assertEqual(checked.call_count, 2)
        self.assertEqual(self.output.read_bytes(), legacy)
        self.assert_png_dimensions()

    def test_cli_passes_captured_spectrum_and_frames_to_the_sheet(self):
        with patch.dict(os.environ, {"RAVEN_EARS_HOME": str(self.output.parent),
                                     "RAVEN_EARS_NO_BASELINE": "1"}), \
             card_fixture() as (audio, *_), redirect_stdout(io.StringIO()), \
             patch.object(S, "speech_sheet", return_value=self.output) as draw:
            self.assertEqual(R.main([str(audio), "--quiet", "--picture-out", str(self.output)]), 0)
        draw.assert_called_once()
        data, measured, output = draw.call_args.args
        self.assertEqual(output, self.output)
        self.assertNotIn("spectrum", data)
        edges, frequencies, power = measured["spectrum"]
        self.assertEqual(edges[-1], data["duration_s"])
        self.assertEqual(power.shape, (len(frequencies) - 1, 2400))
        self.assertEqual(len(measured["times"]), len(measured["rms_db"]))
        self.assertEqual(len(measured["times"]), len(measured["f0"]))


if __name__ == "__main__":
    unittest.main()
