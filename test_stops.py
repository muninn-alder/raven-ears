"""Synthetic stems only; runnable with python3 test_stops.py. No model or network."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import stops as S


class Stops(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.sr = 8000

    def stems(self, gaps=((3, 4),), voice="holds", bleed=0, duration=9, scale=1,
              fade=0, tail=0, stereo=False, vocal_band_bleed=0, reverb=0):
        t = np.arange(round(duration * self.sr)) / self.sr
        # Sines plus invented noise; deterministic, with integral periods per hop.
        rng = np.random.default_rng(431)
        signals = [np.sin(2 * np.pi * f * t) + .02 * rng.normal(size=len(t)) for f in (100, 200, 300, 400)]
        env = np.ones(len(t))
        for start, end in gaps:
            inside = (t >= start) & (t < end)
            env[inside] = 0
            if fade:
                ramp = (t >= start - fade) & (t < start)
                env[ramp] = 10 ** (-24 * (t[ramp] - start + fade) / fade / 20)
            if tail:
                env[inside] = np.exp(-(t[inside] - start) / tail)
            if reverb:
                ringing = inside & (t < start + reverb)
                env[ringing] = np.exp(-np.log(1000) * (t[ringing] - start) / reverb)
        venv = np.ones(len(t))
        if voice == "falls":
            venv = env.copy()
        elif voice == "absent":
            venv[:] = 0
        elif voice == "partial":
            venv[env == 0] = 10 ** (-9 / 20)
        elif voice == "enters":
            venv[:] = 0
            for start, end in gaps:
                venv[(t >= start) & (t < end)] = 1
        elif voice == "weak":
            venv[:] = .01
        samples = [.15 * signals[i] * env for i in range(3)]
        vocal = .15 * signals[3] * venv
        samples[2] += bleed * vocal
        vocal += vocal_band_bleed * samples[0]
        samples.append(vocal)
        for stem, y in zip(S.STEMS, samples):
            y = y * scale
            if stereo:
                y = np.column_stack((y, -y))
            sf.write(Path(self.tmp) / (stem + ".wav"), y, self.sr, subtype="FLOAT")
        return self.tmp

    def analyse(self, **kwargs):
        result = S.analyse(self.stems(**kwargs), tempo=120)
        self.assertEqual(result["status"], "ran", result)
        return result

    def one(self, result, kind):
        self.assertEqual(len(result["events"]), 1, result)
        e = result["events"][0]
        self.assertEqual(e["kind"], kind, e)
        self.assertAlmostEqual(e["start_s"], 3, delta=.03)
        self.assertAlmostEqual(e["length_s"], 1, delta=.03)
        self.assertLessEqual(e["fall_s"], S.MAX_FALL_S)
        self.assertGreater(e["fall_db_per_s"], 0)
        return e

    def test_band_cuts_voice_holds(self):
        e = self.one(self.analyse(), S.KINDS[0])
        self.assertGreaterEqual(e["drop_db"], S.DROP_DB)
        self.assertAlmostEqual(e["voice_change_db"], 0, delta=.2)

    def rising_voice_cut(self, band_depth=34.6, voice_scale=1, rise=8.6):
        # Invented PCM following the measured 1.56 s cut's shape: a local
        # -28.5 dB vocal reference, a brief low, then mostly +8.6 dB singing.
        # Of 148 settled frames, 23 are below -6, 101 above +6, 24 within ±6.
        n, hop = 900, 80
        levels = np.tile(np.array([-73.4, -17.8, -30.0, -28.5])[:, None], (1, n))
        fall = np.interp(np.arange(156), [0, 8, 23, 43, 78, 79, 155],
                         [6 / 34.6, 12 / 34.6, .55, .70, .92, 1, 1])
        levels[:3, 306:462] -= band_depth * fall
        levels[3, 306:314] += 5
        levels[3, 314:326] -= 10
        levels[3, 326:337] -= 20
        levels[3, 337:438] += rise + 1.5 * np.sin(np.arange(101) / 8)
        levels[3, 438:462] += np.linspace(-5, 5, 24)
        levels[3, 500:] = -17.9     # the song-wide level is not the cut reference
        t = np.arange(n * hop) / self.sr
        for s, stem in enumerate(S.STEMS):
            energy = np.repeat(10 ** (levels[s] / 10), hop)
            y = np.sqrt(2 * energy) * np.sin(2 * np.pi * (s + 1) * 100 * t)
            if stem == "vocals":
                y *= voice_scale
            sf.write(Path(self.tmp) / (stem + ".wav"), y, self.sr, subtype="FLOAT")
        return S.analyse(self.tmp)

    def test_voice_rise_and_short_low_hold_through_measured_shape(self):
        result = self.rising_voice_cut()
        self.assertEqual(result["status"], "ran", result)
        self.assertEqual(len(result["events"]), 1, result)
        e = result["events"][0]
        self.assertEqual(e["kind"], S.KINDS[0], e)
        self.assertAlmostEqual(e["start_s"], 3.06, delta=.02)
        self.assertAlmostEqual(e["length_s"], 1.56, delta=.02)
        self.assertAlmostEqual(e["drop_db"], 34.6, delta=.1)
        self.assertAlmostEqual(e["voice_change_db"], 8.6, delta=1)
        self.assertAlmostEqual(e["reference_db"]["vocals"], -28.5, delta=.1)
        self.assertAlmostEqual(e["voice_hold_fraction"], 125 / 148, delta=.01)
        self.assertNotIn("voice_break", e)
        self.assertIn("voice holds", "\n".join(S.card_lines(result)))

    def test_rising_voice_does_not_confirm_shallow_or_bleed_cut(self):
        for options, reason in ((dict(band_depth=9), "accompaniment not 12 dB down"),
                                (dict(voice_scale=.01), "pre-cut voice too weak")):
            with self.subTest(**options):
                events = self.rising_voice_cut(**options)["events"]
                self.assertTrue(events)
                self.assertTrue(all(e["kind"] == "uncertain" for e in events), events)
                self.assertTrue(any(reason in e["reason"] for e in events), events)

    def test_a_rise_as_big_as_a_band_drop_is_not_a_held_voice(self):
        # Rich Girl 0:19.8 shape (11 Nov 2026): vocals +15.9 dB while the band falls; the vocal stem
        # probably took over band energy. Father Figure's +8.6 dB still holds (test above).
        e = self.rising_voice_cut(rise=15.9)["events"][0]
        self.assertEqual(e["kind"], "uncertain", e)
        self.assertIn("band energy may have moved into the vocal stem", e["reason"])
        self.assertGreaterEqual(e["voice_change_db"], S.VOICE_RISE_MAX_DB)

    def test_full_silence(self):
        self.one(self.analyse(voice="falls"), "full silence")

    def test_band_break(self):
        self.one(self.analyse(voice="absent"), "band break")

    def test_strum_gap_intro_is_not_a_stop(self):
        # Regular strums decay into roughly 0.4 s dips before the singer enters.
        # Bass joins partway through; one-stem-only filtering is insufficient.
        t = np.arange(1600) * S.HOP_S
        phase = t % 1.3
        energy = np.exp(-phase)
        energy[phase >= .9] *= .001
        energy[t >= 14] = 1
        curves = dict(drums=np.zeros(len(t)), bass=energy * (t >= 6.5),
                      other=energy, vocals=(t >= 14).astype(float))
        events = S.detect(curves)
        self.assertEqual(events, [])

    def test_band_break_minimum_does_not_raise_vocal_cut_minimum(self):
        for length in (.3, .45, .49, .5):
            for voice, kind in (("absent", "band break"), ("holds", S.KINDS[0]),
                                ("falls", "full silence")):
                with self.subTest(length=length, voice=voice):
                    result = self.analyse(voice=voice, gaps=((3, 3 + length),))
                    if voice == "absent" and length < .5:
                        self.assertTrue(all(e["kind"] == "uncertain" for e in result["events"]))
                        self.assertIn("shorter than 0.5 s", result["events"][0]["reason"])
                    else:
                        self.assertEqual(len(result["events"]), 1, result)
                        self.assertEqual(result["events"][0]["kind"], kind)

    def test_early_stem_exit_cannot_lengthen_short_band_break(self):
        self.stems(voice="absent", gaps=((4, 4.4),))
        for stem in ("drums", "bass"):
            path = Path(self.tmp) / (stem + ".wav")
            y, sr = sf.read(path)
            if stem == "drums":
                y[:] = 0
            else:
                y[3 * sr:4 * sr] = 0
            sf.write(path, y, sr, subtype="FLOAT")
        events = S.analyse(self.tmp)["events"]
        self.assertTrue(all(e["kind"] == "uncertain" for e in events))
        self.assertIn("shorter than 0.5 s", events[0]["reason"])

    def test_band_bleed_in_vocals_is_uncertain_not_full_silence(self):
        self.assertIn("pre-cut voice", self.one(self.analyse(voice="absent", vocal_band_bleed=.05), "uncertain")["reason"])

    def test_slow_fade_is_not_stop(self):
        events = self.analyse(fade=1.2)["events"]
        self.assertTrue(events)
        self.assertTrue(all(e["kind"] == "uncertain" and "fade not ruled out" in e["reason"] for e in events))

    def test_three_second_fade_is_not_stop(self):
        result = self.analyse(fade=3, gaps=((5, 7),), duration=10)
        self.assertTrue(result["events"], result)
        self.assertTrue(all(e["kind"] == "uncertain" for e in result["events"]))
        self.assertIn("fade not ruled out", result["events"][0]["reason"])

    def test_modulated_fade_is_not_stop(self):
        self.stems(fade=1.2)
        for stem in S.STEMS[:3]:
            path = Path(self.tmp) / (stem + ".wav")
            y, sr = sf.read(path)
            t = np.arange(len(y)) / sr
            y *= 10 ** (1.5 * np.sin(2 * np.pi * 7 * t) / 20)
            sf.write(path, y, sr, subtype="FLOAT")
        events = S.analyse(self.tmp)["events"]
        self.assertTrue(events)
        self.assertTrue(all(e["kind"] == "uncertain" for e in events), events)

    def test_half_second_reverb_is_stop(self):
        result = self.analyse(reverb=.5, gaps=((3, 5),))
        self.assertEqual(len(result["events"]), 1, result)
        e = result["events"][0]
        self.assertEqual(e["kind"], S.KINDS[0], e)
        self.assertAlmostEqual(e["start_s"], 3, delta=.10)
        self.assertGreater(e["fall_s"], .15)
        self.assertLess(e["fall_s"], .5)

    def test_short_and_long_gaps_belong_elsewhere(self):
        for end in (3.2, 12):
            self.assertEqual(self.analyse(gaps=((3, end),), duration=15)["events"], [])

    def test_five_and_six_second_stops_with_return(self):
        for length in (5, 6):
            result = self.analyse(gaps=((3, 3 + length),), duration=12)
            self.assertEqual(len(result["events"]), 1, result)
            self.assertEqual(result["events"][0]["kind"], S.KINDS[0])
            self.assertAlmostEqual(result["events"][0]["length_s"], length, delta=.02)

    def test_duration_bounds_inclusive(self):
        for length in (S.MIN_STOP_S, S.MAX_STOP_S):
            result = self.analyse(gaps=((3, 3 + length),), duration=14)
            self.assertEqual(len(result["events"]), 1, result)
            self.assertAlmostEqual(result["events"][0]["length_s"], length, delta=.02)

    def test_vocal_bleed_in_other(self):
        # -20 dB vocal leakage into other; raw summed band energy still falls.
        e = self.one(self.analyse(bleed=.1), S.KINDS[0])
        self.assertGreater(e["drop_db"], 20)
        self.assertLess(e["drop_db"], 30)

    def test_only_other_active_before_cut(self):
        # Intro shape: drums and bass absent on both sides; only other cuts.
        self.stems()
        for stem in ("drums", "bass"):
            sf.write(Path(self.tmp) / (stem + ".wav"), np.zeros(9 * self.sr), self.sr, subtype="FLOAT")
        e = self.one(S.analyse(self.tmp), S.KINDS[0])
        self.assertGreater(e["drop_db"], S.DROP_DB)

    def test_only_other_decays_while_voice_holds(self):
        # Invented ringing layer: do not equate the slow energy edge with
        # the moment an instrument stopped being played.
        self.stems(tail=.45)
        for stem in ("drums", "bass"):
            sf.write(Path(self.tmp) / (stem + ".wav"), np.zeros(9 * self.sr), self.sr, subtype="FLOAT")
        result = S.analyse(self.tmp)
        self.assertEqual(len(result["events"]), 1, result)
        e = result["events"][0]
        self.assertEqual(e["kind"], "uncertain")
        self.assertEqual(e["voice_behavior"], S.KINDS[0])
        self.assertIn("sustained tail", e["reason"])
        self.assertIn("steady fade", e["reason"])
        self.assertGreater(e["fall_s"], S.FADE_RAMP_S)

    def test_slow_tail_does_not_lower_its_own_reference(self):
        self.stems(tail=.8, gaps=((3, 6),), duration=10)
        result = S.analyse(self.tmp)
        self.assertEqual(len(result["events"]), 1, result)
        e = result["events"][0]
        self.assertEqual(e["kind"], "uncertain")
        self.assertIn("slow edge", e["reason"])
        self.assertLess(e["start_s"], 3.7)
        self.assertGreater(e["drop_db"], S.DROP_DB)

    def test_long_approach_keeps_the_armed_edge_origin(self):
        self.stems(tail=1.5, gaps=((3, 8),), duration=11)
        result = S.analyse(self.tmp)
        self.assertEqual(len(result["events"]), 1, result)
        e = result["events"][0]
        self.assertEqual(e["kind"], "uncertain")
        self.assertGreater(e["fall_s"], 1)
        self.assertIn("slow edge", e["reason"])

    def test_short_reverb_tail(self):
        e = self.one(self.analyse(tail=.035), S.KINDS[0])
        self.assertGreater(e["fall_s"], S.HOP_S)

    def test_ending_is_not_stop(self):
        self.assertEqual(self.analyse(gaps=((3, 9),))["events"], [])

    def test_long_ending_is_not_stop(self):
        self.assertEqual(self.analyse(gaps=((3, 15),), duration=15)["events"], [])

    def test_short_transient_does_not_end_a_long_stop(self):
        self.stems(gaps=((3, 9),), duration=12)
        for stem in S.STEMS[:3]:
            path = Path(self.tmp) / (stem + ".wav")
            y, sr = sf.read(path)
            t = np.arange(round(.05 * sr)) / sr
            y[6 * sr:6 * sr + len(t)] = .15 * np.sin(2 * np.pi * 100 * t)
            sf.write(path, y, sr, subtype="FLOAT")
        e = S.analyse(self.tmp)["events"][0]
        self.assertEqual(e["kind"], S.KINDS[0])
        self.assertAlmostEqual(e["length_s"], 6, delta=.02)

    def test_staggered_stem_exits_keep_both_boundaries(self):
        self.stems(gaps=((4, 6),), duration=10)
        # Drums leave first. Bass/other hold another second, then decay quickly.
        path = Path(self.tmp) / "drums.wav"
        y, sr = sf.read(path)
        y[3 * sr:4 * sr] = 0
        sf.write(path, y, sr, subtype="FLOAT")
        e = S.analyse(self.tmp)["events"][0]
        self.assertEqual(e["kind"], S.KINDS[0])
        self.assertAlmostEqual(e["start_s"], 3, delta=.02)
        self.assertAlmostEqual(e["band_start_s"], 4, delta=.02)
        self.assertEqual(e["first_exit_stem"], "drums")
        self.assertIn("first exit: drums", "\n".join(S.card_lines(S.analyse(self.tmp))))

    def test_already_quiet_stem_does_not_create_exit_at_lookback_bound(self):
        self.stems(gaps=((5, 6),), duration=10)
        path = Path(self.tmp) / "drums.wav"
        y, sr = sf.read(path)
        y[2 * sr:] = 0
        sf.write(path, y, sr, subtype="FLOAT")
        e = S.analyse(self.tmp)["events"][0]
        self.assertEqual(e["kind"], S.KINDS[0])
        self.assertAlmostEqual(e["start_s"], 5, delta=.02)
        self.assertIsNone(e["first_exit_stem"])

    def test_voice_fade_after_return_is_separate_from_stop(self):
        self.stems()
        path = Path(self.tmp) / "vocals.wav"
        y, sr = sf.read(path)
        t = np.arange(len(y)) / sr
        y *= 10 ** (-30 * np.clip((t - 4) / 1.2, 0, 1) / 20)
        sf.write(path, y, sr, subtype="FLOAT")
        result = S.analyse(self.tmp)
        e = self.one(result, S.KINDS[0])
        self.assertGreater(e["voice_after"]["start_s"], 4)
        self.assertLess(e["voice_after"]["end_s"], 5.1)
        self.assertIn("voice then fades 6→24 dB below pre-cut", "\n".join(S.card_lines(result)))

    def test_uncertain_voice_is_named_and_not_counted_as_confirmed(self):
        result = self.analyse(voice="partial")
        e = self.one(result, "uncertain")
        self.assertIn("voice neither", e["reason"])
        card = "\n".join(S.card_lines(result))
        self.assertIn("STOPS: 0", card)
        self.assertIn("uncertain: 1 candidates not called", card)
        self.assertIn("uncertain candidates: 1", "\n".join(S.card_lines(result, detail=True)))

    def test_voice_entry_does_not_claim_sung_pitch(self):
        result = self.analyse(voice="enters")
        self.one(result, S.ENTRY_KIND)
        self.assertIn("singing/pitch not checked", "\n".join(S.card_lines(result)))

    def test_opening_cut_is_scanned(self):
        # The one new synthetic smoke alarm: it chooses no detector thresholds.
        result = self.analyse(gaps=((.4, 1.4),))
        self.assertTrue(any(abs(e["start_s"] - .4) < .03 for e in result["events"]))

    def test_saved_curves_keep_precision_and_invalid_evidence_fails_visibly(self):
        # Persistence only: mock detection so these invented numbers never
        # calibrate an energy, depth, duration or recurrence rule.
        curves = {s: np.array([1.234567891e-9, 2.345678912e-8]) for s in S.STEMS}
        with patch.object(S, "_curves", return_value=(curves, .02)), \
             patch.object(S, "detect", return_value=[]):
            result = S.analyse("unused")
        saved = json.loads(json.dumps(result))
        self.assertEqual(saved["curves"]["encoding"], "f64le-zlib-base64")
        restored = S._unpack_curves(saved["curves"])
        for stem in S.STEMS:
            np.testing.assert_array_equal(restored[stem], curves[stem])
        with patch.object(S, "detect", return_value=[]) as detect:
            self.assertEqual(S.refresh(saved)["events"], [])
            for stem in S.STEMS:
                np.testing.assert_array_equal(detect.call_args.args[0][stem], curves[stem])
        for change in ({"encoding": "unknown"}, {"frames": 3}, {"stems": list(reversed(S.STEMS))},
                       {"hop_s": .02}, {"data": "!invalid"}, {"data": "AAAA"}):
            with self.subTest(change=change):
                broken = dict(saved, curves=dict(saved["curves"], **change))
                self.assertEqual(S.refresh(broken)["status"], "failed")
        # Full-precision JSON lists from version 2 remain readable.
        legacy = {s: c.tolist() for s, c in curves.items()}
        with patch.object(S, "detect", return_value=[]) as detect:
            S.refresh(dict(saved, curves=legacy))
            self.assertEqual(detect.call_args.args[0], legacy)
        saved["curves"] = {s: [float("nan")] for s in S.STEMS}
        refreshed = S.refresh(saved)
        self.assertEqual(refreshed["status"], "failed")
        self.assertIn("re-listen to update STOPS", S.card_lines(refreshed)[0])

    def level_stems(self, levels):
        """Invented 10 ms mean-square curves, with aligned four-stem PCM."""
        hop = self.sr // 100
        t = np.arange(levels.shape[1] * hop) / self.sr
        for s, stem in enumerate(S.STEMS):
            energy = np.repeat(10 ** (levels[s] / 10), hop)
            y = np.sqrt(2 * energy) * np.sin(2 * np.pi * (s + 1) * 100 * t)
            sf.write(Path(self.tmp) / (stem + ".wav"), y, self.sr, subtype="FLOAT")
        return S.analyse(self.tmp)

    def faded_levels(self):
        # Invented half-second levels following Strawberry Fields' measured
        # drums/other fade, trough and return, shifted 180 s earlier. Bass and
        # voice sit near their measured inactive floors; no real samples.
        t = np.arange(4500) * S.HOP_S
        knots = [0, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 32.5, 33,
                 33.5, 34, 34.5, 35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5,
                 39, 39.5, 40, 40.5, 41, 42, 44.99]
        drums = [-28, -28, -32, -35, -32, -36, -33, -37, -39, -39, -44, -52,
                 -55, -55, -58, -61, -65, -59, -59, -56, -51, -49, -55, -43,
                 -45, -37, -38, -32, -30, -26, -22, -22]
        other = [-28, -28, -27, -28, -28, -30, -33, -37, -37, -43, -43, -49,
                 -56, -51, -56, -60, -60, -79, -52, -51, -46, -48, -44, -44,
                 -37, -37, -34, -33, -27, -27, -25, -25]
        levels = np.array([np.interp(t, knots, drums), np.full(len(t), -90.),
                           np.interp(t, knots, other), np.full(len(t), -94.)])
        levels[:3] += np.sin(2 * np.pi * 7 * t)
        return levels

    def test_measured_fade_shape_has_its_own_kind_and_card(self):
        result = self.level_stems(self.faded_levels())
        self.assertEqual(result["status"], "ran", result)
        self.assertEqual(len(result["events"]), 1, result)
        e = result["events"][0]
        self.assertEqual(e["kind"], S.FADED_KIND)
        self.assertAlmostEqual(e["start_s"], 33.77, delta=.05)
        self.assertAlmostEqual(e["drop_db"], 17.4, delta=.3)
        self.assertAlmostEqual(e["low_length_s"], 1.59, delta=.05)
        self.assertGreater(e["fall_s"], S.MAX_FALL_S)
        card = "\n".join(S.card_lines(result))
        self.assertIn("STOPS: 1", card)
        self.assertIn("faded stop: 1", card)
        self.assertIn("total stem energy falls", card)
        self.assertNotIn("band cuts 17", card)
        self.assertTrue(all(len(line) <= 118 for line in card.splitlines()))

    def test_faded_stop_requires_return_depth_and_two_stems_beyond_bleed(self):
        for change in ("terminal", "shallow", "one stem", "bleed", "brief return", "quiet passage"):
            with self.subTest(change=change):
                levels = self.faded_levels()
                if change == "terminal":
                    levels[:, 3500:] = levels[:, 3499:3500]
                elif change == "shallow":
                    levels[:3] = -28 + (levels[:3] + 28) * .4
                elif change == "one stem":
                    levels[0] = -90
                elif change == "bleed":
                    levels[3] = levels[2] + 30
                    levels[0] = levels[2]
                elif change == "brief return":
                    levels[:, 3500:] = -100
                    levels[2, 3600:3605] = -20
                else:
                    # A long quieter section that does return, rather than a
                    # short trough. The wider reference follows its new level.
                    t = np.arange(levels.shape[1]) * S.HOP_S
                    section = np.interp(t, [0, 20, 24, 36, 40, 45], [-28, -28, -48, -48, -28, -28])
                    levels[0] = levels[2] = section
                events = self.level_stems(levels)["events"]
                self.assertFalse(any(e["kind"] == S.FADED_KIND for e in events), events)

    def test_existing_slow_candidates_are_never_promoted_to_faded(self):
        levels = self.faded_levels()
        levels[:3] -= np.sin(2 * np.pi * 7 * np.arange(levels.shape[1]) * S.HOP_S)
        events = self.level_stems(levels)["events"]
        self.assertTrue(events)
        self.assertTrue(all(e["kind"] == "uncertain" for e in events), events)

    def test_padding_recovery_after_absent_reference_is_not_a_stop(self):
        # Strawberry Fields' terminal shape: -91.6/-99/-99/-96.5 dB
        # reference, digital zero, then flat -93.3 dB bass residue. A relative
        # 6 dB / 100 ms return alone accepts that residue as a band return.
        levels = np.full((4, 900), -25.)
        levels[:, 500:] = np.array([-91.6, -99., -99., -96.5])[:, None]
        levels[:, 650:744] = -200
        levels[:, 744:] = -200
        levels[1, 744:] = -93.3
        self.assertEqual(self.level_stems(levels)["events"], [])

    def test_ringing_shapes_do_not_establish_a_new_confirmed_kind(self):
        # Invented 100 ms stem levels following the three measured residuals:
        # steady other; a decaying bass under repeatedly rising other; rapidly
        # decaying other. Vocal reference and entry follow the real shapes.
        shapes = [
            ([-33, -32, -33, -31, -31, -31, -31, -32, -30, -33, -30, -33, -34],
             [-80, -67, -61, -59, -56, -54, -53, -52, -51, -51, -51, -51, -52],
             [-27, -20, -23, -51, -27, -20, -24, -28, -33, -56, -29, -21, -20], -85),
            ([-38, -36, -29, -27, -27, -24, -38, -36, -32, -35, -37, -27, -22, -48, -53, -59, -56],
             [-41, -43, -43, -44, -45, -48, -49, -49, -51, -50, -52, -51, -52, -55, -56, -59, -60],
             [-24, -24, -26, -33, -22, -20, -20, -22, -31, -26, -21, -20, -19, -27, -29, -22, -24], -22),
            ([-36, -38, -43, -50, -49], [-88] * 5, [-85, -84, -46, -26, -23], -86)]
        for other, bass, voice, vref in shapes:
            levels = np.tile(np.array([-80., -80., -24., vref])[:, None], (1, 900))
            if vref == -22:
                levels[1] = -30
            end = 300 + len(other) * 10
            for s, values in ((1, bass), (2, other), (3, voice)):
                levels[s, 300:end] = np.repeat(values, 10)
            events = self.level_stems(levels)["events"]
            self.assertTrue(events)
            self.assertTrue(all(e["kind"] == "uncertain" for e in events), events)

    def test_quiet_voice_is_not_guessed(self):
        self.assertIn("pre-cut voice", self.one(self.analyse(voice="weak"), "uncertain")["reason"])

    def test_scale_and_stereo_do_not_change_kind(self):
        for options in (dict(scale=.0001), dict(stereo=True)):
            self.one(self.analyse(**options), S.KINDS[0])

    def test_sustained_quiet_layer_is_uncertain_despite_summed_drop(self):
        self.stems()
        t = np.arange(9 * self.sr) / self.sr
        # A quiet but continuous instrument: 20 dB under a single loud stem.
        sf.write(Path(self.tmp) / "other.wav", .015 * np.sin(2 * np.pi * 300 * t), self.sr, subtype="FLOAT")
        e = self.one(S.analyse(self.tmp), "uncertain")
        self.assertGreater(e["drop_db"], S.DROP_DB)
        self.assertIn("other fell only", e["reason"])

    def test_three_kinds_remain_separate(self):
        self.stems(gaps=((3, 4), (7, 8), (11, 12)), duration=15)
        path = Path(self.tmp) / "vocals.wav"
        y, sr = sf.read(path)
        y[7 * sr:8 * sr] = 0            # everything cuts in the second gap
        y[9 * sr:13 * sr] = 0           # no singer before or during the third
        sf.write(path, y, sr, subtype="FLOAT")
        result = S.analyse(self.tmp)
        self.assertEqual([e["kind"] for e in result["events"]], list(S.KINDS))
        self.assertIn("STOPS: 3 · band cuts, voice holds: 1 · full silence: 1 · band break: 1", S.card_lines(result))

    def test_a_cappella_bleed_cannot_claim_full_silence(self):
        self.stems(voice="falls")
        voice, sr = sf.read(Path(self.tmp) / "vocals.wav")
        for stem in S.STEMS[:3]:
            y = voice * .05 if stem == "other" else np.zeros_like(voice)
            sf.write(Path(self.tmp) / (stem + ".wav"), y, sr, subtype="FLOAT")
        e = self.one(S.analyse(self.tmp), "uncertain")
        self.assertIn("could be vocal bleed alone", e["reason"])

    def test_quiet_band_independent_of_loud_voice_is_stop(self):
        self.stems()
        voice, sr = sf.read(Path(self.tmp) / "vocals.wav")
        for stem in S.STEMS[:3]:
            path = Path(self.tmp) / (stem + ".wav")
            y, _ = sf.read(path)
            y *= .02                     # band >18 dB below voice before cut
            if stem == "other":
                y += .001 * voice        # residual voice leakage is allowed
            sf.write(path, y, sr, subtype="FLOAT")
        e = self.one(S.analyse(self.tmp), S.KINDS[0])
        band_ref = S._db(sum(10 ** (e["reference_db"][s] / 10) for s in S.STEMS[:3]))
        self.assertLess(band_ref, e["reference_db"]["vocals"] - S.BLEED_DB)
        self.assertGreaterEqual(e["stem_independent_fraction"]["other"], S.LOW_FRACTION)

    def test_parallel_vocal_leakage_falls_are_not_independent(self):
        self.stems(voice="partial")
        path = Path(self.tmp) / "vocals.wav"
        y, sr = sf.read(path)
        y[3 * sr:4 * sr] *= .1            # nonzero vocal fall, about 29 dB
        sf.write(path, y, sr, subtype="FLOAT")
        for stem in S.STEMS[:3]:
            sf.write(Path(self.tmp) / (stem + ".wav"), .02 * y, sr, subtype="FLOAT")
        e = self.one(S.analyse(self.tmp), "uncertain")
        self.assertEqual(e["voice_behavior"], "full silence")
        self.assertIn("without an independent stem fall", e["reason"])
        self.assertTrue(all(f == 0 for f in e["stem_independent_fraction"].values()))

    def test_independent_fall_does_not_waive_sustained_stem_guard(self):
        self.stems()
        t = np.arange(9 * self.sr) / self.sr
        for stem in S.STEMS[:3]:
            path = Path(self.tmp) / (stem + ".wav")
            y, sr = sf.read(path)
            if stem == "other":
                y = .015 * np.sin(2 * np.pi * 300 * t)
            sf.write(path, .02 * y, sr, subtype="FLOAT")
        e = self.one(S.analyse(self.tmp), "uncertain")
        self.assertIn("other fell only", e["reason"])
        self.assertNotIn("could be vocal bleed alone", e["reason"])

    def partial_cut(self, voice_break=True):
        # Bass/other leave, drums carry on then make a short rhythmic dip. The
        # voice returns before that dip: settled-only classification misses it.
        self.stems(gaps=((4.4, 4.7),), duration=8)
        for stem in ("bass", "other", "vocals"):
            path = Path(self.tmp) / (stem + ".wav")
            y, sr = sf.read(path)
            if stem == "vocals":
                if voice_break:
                    y[round(3.2 * sr):round(3.6 * sr)] = 0
            else:
                y[3 * sr:round(4.4 * sr)] = 0
            sf.write(path, y, sr, subtype="FLOAT")
        result = S.analyse(self.tmp)
        self.assertEqual(len(result["events"]), 1, result)
        return result

    def test_staggered_partial_cut_with_voice_break_is_not_voice_holds(self):
        result = self.partial_cut()
        e = result["events"][0]
        self.assertEqual(e["kind"], "uncertain", e)
        self.assertEqual(e["arrangement"], "partial cut")
        self.assertEqual(e["continuing_stems"], ["drums"])
        self.assertEqual(e["cutting_stems"], ["bass", "other"])
        self.assertEqual(e["voice_behavior"], "breaks then returns")
        self.assertAlmostEqual(e["voice_break"]["start_s"], 3.2, delta=.02)
        self.assertAlmostEqual(e["voice_break"]["length_s"], .4, delta=.02)
        card = "\n".join(S.card_lines(result, detail=True))
        self.assertIn("partial cut", card)
        self.assertIn("voice breaks:", card)
        self.assertNotIn(", voice holds\n", card)
        self.assertIn("STOPS: 0", card)

    def test_partial_cut_without_voice_break_does_not_invent_one(self):
        e = self.partial_cut(voice_break=False)["events"][0]
        self.assertEqual(e["kind"], "uncertain")
        self.assertEqual(e["arrangement"], "partial cut")
        self.assertEqual(e["voice_behavior"], S.KINDS[0])
        self.assertNotIn("voice_break", e)

    def test_sustained_voice_break_cannot_hide_in_hold_fraction(self):
        self.stems(gaps=((3, 6),), duration=9)
        path = Path(self.tmp) / "vocals.wav"
        y, sr = sf.read(path)
        y[4 * sr:round(4.4 * sr)] = 0
        sf.write(path, y, sr, subtype="FLOAT")
        e = S.analyse(self.tmp)["events"][0]
        self.assertGreater(e["voice_hold_fraction"], S.LOW_FRACTION)
        self.assertEqual(e["kind"], "uncertain")
        self.assertEqual(e["voice_behavior"], "breaks then returns")

    def test_short_vocal_low_is_not_a_sustained_break(self):
        self.stems(gaps=((3, 6),), duration=9)
        path = Path(self.tmp) / "vocals.wav"
        y, sr = sf.read(path)
        y[4 * sr:round(4.2 * sr)] = 0
        sf.write(path, y, sr, subtype="FLOAT")
        e = S.analyse(self.tmp)["events"][0]
        self.assertEqual(e["kind"], S.KINDS[0])
        self.assertNotIn("voice_break", e)

    def test_voice_break_requires_sustained_return(self):
        self.stems(gaps=((3, 6),), duration=9)
        path = Path(self.tmp) / "vocals.wav"
        y, sr = sf.read(path)
        y[4 * sr:round(5.95 * sr)] = 0
        sf.write(path, y, sr, subtype="FLOAT")
        e = S.analyse(self.tmp)["events"][0]
        self.assertEqual(e["kind"], "uncertain")
        self.assertNotIn("voice_break", e)  # 50 ms spike is not a vocal return

    def test_a_new_level_after_long_break_can_have_stops(self):
        self.stems(gaps=((3, 12), (15, 16)), duration=20)
        for stem in S.STEMS[:3]:
            path = Path(self.tmp) / (stem + ".wav")
            y, sr = sf.read(path)
            y[12 * sr:] *= .1           # a different section, 20 dB quieter
            sf.write(path, y, sr, subtype="FLOAT")
        result = S.analyse(self.tmp)
        self.assertEqual(len(result["events"]), 1, result)
        self.assertAlmostEqual(result["events"][0]["start_s"], 15, delta=.02)

    def test_beats_rounding_and_no_unsupplied_beats(self):
        self.assertEqual(S.half_beats(.9, 117.5), 2)
        self.assertEqual(S.half_beats(.625, 120), 1.5)
        self.assertEqual(S.half_beats(.61, 120), 1)
        self.assertNotIn("beats_approx", S.analyse(self.stems())["events"][0])
        self.assertIn("≈2 beats", "\n".join(S.card_lines(self.analyse())))

    def test_recurrence_requires_measurements_and_explicit_meter(self):
        gaps = ((3, 4), (7, 8), (11, 12), (15, 16))
        result = S.analyse(self.stems(gaps=gaps, duration=19), 120, 4)
        self.assertEqual(result["recurrence"], [dict(kind=S.KINDS[0], interval_s=4, intervals=3,
                                                     beats_approx=8, bars_approx=2, count=4,
                                                     start_s=3, end_s=16, event_starts_s=[3, 7, 11, 15])])
        self.assertIn("every 2 bars ≈", "\n".join(S.card_lines(result)))
        self.assertIn("repeated cuts:", "\n".join(S.card_lines(result)))
        self.assertIn("pulse not checked", "\n".join(S.card_lines(result)))
        self.assertNotIn("stop-time:", "\n".join(S.card_lines(result)))
        self.assertNotIn("bars_approx", S.analyse(self.tmp, 120)["recurrence"][0])
        self.assertEqual(S.analyse(self.stems(gaps=gaps[:3], duration=19), 120)["recurrence"], [])
        gaps = ((3, 4), (7, 8), (12, 13), (15, 16))
        self.assertEqual(S.analyse(self.stems(gaps=gaps, duration=19), 120)["recurrence"], [])

    def live_cut_rows(self, gaps, start=2.6):
        # Invented event rows, following Clapton 1994's measured nine-cut
        # shape: two unvoiced cuts, then vocal entries; no curve calibration.
        starts = np.r_[start, start + np.cumsum(gaps)].round(2).tolist()
        return [dict(start_s=t, length_s=1.8, drop_db=40.1,
                     fall_s=.64, fall_to_db=36, fall_db_per_s=51.6,
                     kind="band break" if i < 2 else S.ENTRY_KIND)
                for i, t in enumerate(starts)]

    def test_live_recurrence_keeps_fundamental_and_last_cut(self):
        events = self.live_cut_rows([3.10, 3.22, 3.08, 3.12, 3.15, 3.09, 3.03, 3.17])
        events += self.live_cut_rows([3.18, 3.11, 3.07, 3.09, 3.07, 3.05], start=143)
        runs = S._recurrence(events)
        self.assertEqual([r["count"] for r in runs], [9, 7])
        self.assertEqual([r["interval_s"] for r in runs], [3.11, 3.08])
        self.assertEqual([r["event_starts_s"] for r in runs],
                         [[e["start_s"] for e in events[:9]], [e["start_s"] for e in events[9:]]])

    def test_live_recurrence_uses_complete_run_median_and_checks_limit(self):
        # Muddy 1977: short first gap, two longer gaps, then settling. The
        # third-gap prefix fails the median check; the complete run passes.
        events = self.live_cut_rows([2.98, 3.36, 3.44, 3.20, 3.10])
        runs = S._recurrence(events)
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertEqual((run["count"], run["interval_s"]), (6, 3.2))
        self.assertEqual(run["event_starts_s"], [e["start_s"] for e in events])
        # A feasible midpoint is insufficient: the actual median must fit.
        self.assertEqual(S._recurrence(events[:4]), [])
        outside = self.live_cut_rows([2.98, 3.36, 3.45, 3.20, 3.10])
        self.assertFalse(any(r["count"] == 6 for r in S._recurrence(outside)))

    def test_shorter_recurrence_wins_over_more_half_rate_cuts(self):
        events = self.live_cut_rows([3.10, 3.12, 3.08, 3.10, 6.22, 6.18, 6.24])
        run, = S._recurrence(events)
        self.assertEqual(run["count"], 5)
        self.assertEqual(run["interval_s"], 3.1)
        self.assertEqual(run["event_starts_s"], [e["start_s"] for e in events[:5]])

    def test_live_recurrence_survives_worker_and_song_save(self):
        import listen
        # Exercise run_pass -> worker capture -> save/path scrub -> song.json.
        # Stub only detection and the engine import: no real model or service.
        events = self.live_cut_rows([3.10, 3.22, 3.08, 3.12, 3.15, 3.09, 3.03, 3.17])
        curves = {s: np.array([1.234567891e-9, 2.345678912e-8]) for s in S.STEMS}
        def worker_main(argv, capture):
            result = S.analyse("invented-stems")
            capture({"stops": result})
            print("\n".join(S.card_lines(result)))
        def subprocess_fake(command, **kwargs):
            with patch.object(listen.importlib, "import_module") as engine, \
                 redirect_stdout(io.StringIO()) as out:
                engine.return_value.main = worker_main
                rc = listen.worker(command[3], command[4], command[5:])
            return listen.subprocess.CompletedProcess(command, rc, out.getvalue(), "")
        with patch.object(listen, "store_home", return_value=Path(self.tmp)), \
             patch.object(listen.subprocess, "run", side_effect=subprocess_fake), \
             patch.object(S, "_curves", return_value=(curves, 32)), \
             patch.object(S, "detect", return_value=events):
            card = listen.run_pass("singer", listen.parser().parse_args(["invented.wav"]))
        self.assertEqual(card["status"], "ran", card)
        path = Path(self.tmp) / "song.json"
        listen.save(path, {"cards": {"singer": card}})
        saved = json.loads(path.read_text())["cards"]["singer"]["json"]["stops"]
        self.assertEqual(saved["recurrence"], S._recurrence(events))
        self.assertEqual(saved["recurrence"][0]["count"], 9)
        for stem in S.STEMS:
            np.testing.assert_array_equal(S._unpack_curves(saved["curves"])[stem], curves[stem])

    def test_missing_stem_fails_loud(self):
        self.stems()
        (Path(self.tmp) / "bass.wav").unlink()
        result = S.analyse(self.tmp)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing stem: bass.wav", result["reason"])
        self.assertIn("STOPS: failed", S.card_lines(result)[0])
        self.assertNotIn("none found", "\n".join(S.card_lines(result)))

    def test_misaligned_stem_and_bad_tempo_fail(self):
        self.stems()
        sf.write(Path(self.tmp) / "bass.wav", np.ones(300), self.sr)
        self.assertIn("unaligned stem", S.analyse(self.tmp)["reason"])
        for value in (0, -1, float("nan")):
            self.assertEqual(S.analyse(self.tmp, value)["reason"], "tempo must be finite and positive")

    def test_card_width_empty_summary_and_json(self):
        from raven_ears_plus import MAX_LINE
        result = self.analyse(gaps=())
        self.assertEqual(S.card_lines(result), ["STOPS: none found (0.3 to 8 s; accompaniment vs its own level)",
                                               "  opening 0:00–0:00.30 not checked (short reference history)"])
        result = self.analyse(voice="partial")
        self.assertTrue(all(len(line) <= MAX_LINE for line in S.card_lines(result)))
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(S._time(59.999), "1:00.00")

    def test_shallow_uncertain_candidates_stay_in_json(self):
        result = self.analyse(voice="partial")
        e = result["events"][0]
        e["drop_db"] = 11.9
        card = "\n".join(S.card_lines(result, detail=True))
        self.assertNotIn("0:03.00", card)
        self.assertIn("1 shallower uncertain candidates in JSON only", card)
        self.assertIn("uncertain: 1 candidates not called", "\n".join(S.card_lines(result)))
        self.assertEqual(json.loads(json.dumps(result))["events"], [e])
        e["drop_db"] = 12
        self.assertIn("0:03.00", "\n".join(S.card_lines(result)))
        e["kind"] = S.KINDS[0]
        e["drop_db"] = 11.9
        self.assertIn("0:03.00", "\n".join(S.card_lines(result)))

    def rendering_result(self):
        # Invented rows with the saved Take Five field/line shapes, including
        # first-exit and wrapped rejection diagnostics; no audio or lyrics.
        event = dict(start_s=34.19, length_s=.76, drop_db=13.2, kind="uncertain",
                     fall_to_db=8.4, fall_s=.55, fall_db_per_s=9.8,
                     first_exit_stem="bass", band_start_s=34.27,
                     reason="accompaniment not 12 dB down for 80% of settled cut (tail, bleed or shallow dip); "
                            "drums fell only -2.6 dB; sustained layer or bleed")
        known = dict(event, start_s=229.72, length_s=.89, drop_db=17.7, kind="band break",
                     fall_to_db=13.4, fall_s=.66, fall_db_per_s=15.7,
                     first_exit_stem="drums", band_start_s=229.91, reason="")
        return dict(status="ran", events=[event, dict(event, start_s=50.12, drop_db=11.9), known], recurrence=[])

    def test_default_counts_all_candidates_after_unchanged_confident_event(self):
        result = self.rendering_result()
        before = json.dumps(result, sort_keys=True)
        lines = S.card_lines(result)
        self.assertEqual(lines, [
            "STOPS: 1 · band cuts, voice holds: 0 · full silence: 0 · band break: 1",
            "  3:49.72 (0.89 s) band cuts 17.7 dB, band break",
            "    fall 3→13.4 dB in 0.66 s (15.7 dB/s)",
            "    first exit: drums; summed band crosses −6 dB at 3:49.91",
            "  uncertain: 2 candidates not called (see --view detail / JSON)"])
        self.assertEqual(json.dumps(result, sort_keys=True), before)

    def test_detail_is_byte_identical_to_legacy_candidate_block(self):
        self.assertEqual("\n".join(S.card_lines(self.rendering_result(), detail=True)),
            "STOPS: 1 · band cuts, voice holds: 0 · full silence: 0 · band break: 1\n"
            "  uncertain candidates: 1 (band fall ≥12 dB; full list in JSON)\n"
            "  1 shallower uncertain candidates in JSON only (band fall <12 dB)\n"
            "  0:34.19 (0.76 s) summed band falls 13.2 dB, uncertain\n"
            "    fall 3→8.4 dB in 0.55 s (9.8 dB/s)\n"
            "    first exit: bass; summed band crosses −6 dB at 0:34.27\n"
            "    uncertain: accompaniment not 12 dB down for 80% of settled cut (tail, bleed or shallow dip); drums fell only -2.6\n"
            "    dB; sustained layer or bleed\n"
            "  3:49.72 (0.89 s) band cuts 17.7 dB, band break\n"
            "    fall 3→13.4 dB in 0.66 s (15.7 dB/s)\n"
            "    first exit: drums; summed band crosses −6 dB at 3:49.91")

    def test_zero_confident_reports_strongest_even_below_detail_threshold(self):
        result = self.rendering_result()
        result["events"] = result["events"][:2]
        result["events"].append(dict(result["events"][0], start_s=70, drop_db=9.8))
        for shallow in (False, True):
            if shallow:
                result["events"][0]["drop_db"] = 8
            lines = S.card_lines(result)
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], "STOPS: 0 confident · uncertain: 3 candidates not called (see --view detail / JSON)")
            self.assertNotIn("none", "\n".join(lines))
            times = ("0:50.12", "1:10.00") if shallow else ("0:34.19", "0:50.12")
            self.assertTrue(all(time in lines[1] for time in times))
            self.assertIn("strongest band falls:", lines[1])
            if shallow:
                self.assertNotIn("summed band falls", "\n".join(S.card_lines(result, detail=True)))

    def test_cli_detail_and_json_keep_saved_evidence(self):
        result = self.rendering_result()
        for flags, expected in (([], "\n".join(S.card_lines(result))),
                                (["--view", "detail"], "\n".join(S.card_lines(result, detail=True))),
                                (["--json"], json.dumps(result, indent=2, ensure_ascii=False))):
            with patch.object(S, "analyse", return_value=result), redirect_stdout(io.StringIO()) as out:
                self.assertEqual(S.main(["invented-stems", *flags]), 0)
            self.assertEqual(out.getvalue(), expected + "\n")

    def singer_run(self, *extra):
        import singer
        # Stub every existing stage; singer hook reads only invented local stems.
        with patch.object(singer, "get_stems", return_value=(self.tmp, True)), \
             patch.object(singer, "stems_timeline", return_value=(["  invented curves"], [], {}, {})), \
             patch.dict(os.environ, {"RAVEN_NO_STOPS": "0"}), \
             redirect_stdout(io.StringIO()) as out:
            rc = singer.main([str(Path(self.tmp) / "vocals.wav"), "--no-singer", "--no-meter",
                              "--no-instruments", *extra])
        self.assertEqual(rc, 0, out.getvalue())
        return out.getvalue()

    def test_singer_hook_order_json_and_failure(self):
        self.stems()
        card = self.singer_run("--tempo", "120")
        self.assertLess(card.index("STEMS OVER TIME"), card.index("STOPS: 1"))
        self.assertIn("Shorter cuts: see STOPS.", card)
        self.assertIn("≈2 beats", card)
        result = json.loads(self.singer_run("--json"))
        self.assertEqual(len(result["stops"]["events"]), 1)
        self.assertIn(dict(stage="stops", status="ran", kind="audio"), result["stage_status"])
        (Path(self.tmp) / "bass.wav").unlink()
        self.assertIn("STOPS: failed: missing stem: bass.wav", self.singer_run())

    def test_off_switches_skip_detector(self):
        import singer
        self.stems()
        with patch.object(S, "analyse", side_effect=AssertionError("detector must not run")):
            self.assertNotIn("STOPS:", self.singer_run("--no-stops"))
            with patch.object(singer, "get_stems", return_value=(self.tmp, True)), \
                 patch.dict(os.environ, {"RAVEN_NO_STOPS": "1"}), redirect_stdout(io.StringIO()) as out:
                singer.main([str(Path(self.tmp) / "vocals.wav"), "--no-singer", "--no-meter",
                             "--no-instruments", "--no-stems-timeline", "--json"])
            result = json.loads(out.getvalue())
            self.assertNotIn("stops", result)
            self.assertTrue(any(r["stage"] == "stops" and r["status"] == "skipped" for r in result["stage_status"]))


def run_checks(check, tmp=None):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Stops)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    check("STOPS: synthetic cuts, uncertainty, timing, card and off switches",
          result.wasSuccessful(), "\n".join(t for _, t in result.failures + result.errors))


if __name__ == "__main__":
    unittest.main()
