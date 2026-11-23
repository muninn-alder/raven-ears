"""Synthetic cached-beat fixtures; python3 test_meter_compound.py.

These exercise meter_block's real accent/subdivision path without audio or
models. A three-pulse accent repeat alone does not label its musical bar.
"""
import unittest

import numpy as np

import singer


def fixture(profile, tracked_bar, pulse_s=.34, subdivision=None):
    # Same _pre shape as the meter worker: beats, downbeats, engine, envelope,
    # fps, and named envelopes. No precomputed meter verdict or mocked ACF.
    fps = 1000
    b = np.arange(360) * pulse_s + .5
    env = np.zeros(int((b[-1] + pulse_s) * fps) + 1)
    for i, t in enumerate(b):
        env[round(t * fps)] = profile[i % len(profile)]
        if subdivision:
            for fraction, strength in subdivision:
                env[round((t + fraction * pulse_s) * fps)] = strength
    return b, b[::tracked_bar], "beat_this", env, fps, {"mix": env}


def read(pre):
    return singer.meter_block("unused", "unused", float(pre[0][-1] + 1), _pre=pre)


class CompoundMeterTests(unittest.TestCase):
    def test_slow_six_eighth_march_with_two_ternary_strides(self):
        lines, data = read(fixture([1, .2, .2, .65, .2, .2], 6))
        self.assertEqual(data["beats_per_bar"], 6)
        self.assertAlmostEqual(data["pulse_bpm"], 58.8, places=1)
        self.assertEqual(singer.meter_claim(data), "6/8")
        self.assertIn("METER: 6/8 (compound, beats group in 3)", lines[0])

    def test_three_quarter_waltz_with_binary_subdivision(self):
        lines, data = read(fixture([1, .3, .3], 3, subdivision=[(.5, .6)]))
        self.assertEqual(singer.meter_claim(data), "3/4")
        self.assertIn("METER: 3/4 (simple, each beat splits in 2)", lines[0])
        self.assertNotIn("6/8", lines[0])

    def test_nine_eighths_have_three_ternary_strides(self):
        lines, data = read(fixture([1, .2, .2, .7, .2, .2, .7, .2, .2], 9))
        self.assertEqual(data["beats_per_bar"], 9)
        self.assertEqual(singer.meter_claim(data), "9/8")
        self.assertIn("METER: 9/8 (compound, beats group in 3)", lines[0])
        self.assertNotIn("6/8", lines[0])

    def test_nine_eighths_tracked_as_three_compound_beats(self):
        lines, data = read(fixture([1, .7, .7], 3, pulse_s=1.02,
                                   subdivision=[(1 / 3, .6), (2 / 3, .6)]))
        self.assertEqual(singer.meter_claim(data), "9/8")
        self.assertIn("9/8 (compound, each beat splits in 3)", lines[0])

    def test_equal_ternary_strides_do_not_establish_a_paired_bar(self):
        # The same three-pulse sequence could be a waltz bar or one stride of
        # a march. Without a paired-bar cue neither reading is established.
        lines, data = read(fixture([1, .3, .3], 3))
        self.assertAlmostEqual(data["cycle_acf"]["3"], data["cycle_acf"]["6"])
        self.assertIsNone(singer.meter_claim(data))
        self.assertIn("ambiguous", lines[0])


if __name__ == "__main__":
    unittest.main()
