#!/usr/bin/env python3
"""Synthetic-only checks. Run with plain Python plus numpy; no models or services."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ["RAVEN_EARS_NO_BASELINE"] = "1"

from melody_abc import PROFILES, decode_abc, lyric_line_boundaries, main, self_check, transcribe


def hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi) - 69) / 12)


def track(*segments, fps=100):
    """(MIDI or None, seconds); invented pitches only."""
    return np.concatenate([np.full(round(duration * fps), np.nan if pitch is None else hz(pitch))
                           for pitch, duration in segments])


class MelodyTests(unittest.TestCase):
    def check_result(self, abc, result):
        # Independent accounting and emitted-subset checks, NOT a full ABC parser.
        json.dumps(result, allow_nan=False)
        accepted = sum(n["frames"] for n in result["notes"])
        declined = sum(d["frames"] for d in result["declined"] if d["scope"] == "pitch")
        expected = round(result["settings"]["duration_s"] * result["settings"]["fps"])
        self.assertEqual(accepted + declined, expected)
        for n in result["notes"]:
            self.assertGreaterEqual(n["end"] - n["start"] + 1e-10, result["settings"]["min_note_s"])
            if result["settings"]["pitch_rule"] == "strict":
                self.assertLessEqual(n["max_deviation_cents"], result["settings"]["tolerance_cents"] + 1e-8)
            else:
                self.assertLessEqual(abs(n["cents_off"]), result["settings"]["tolerance_cents"] + 1e-8)
                self.assertLessEqual(n["spread_cents"], result["settings"]["spread_cents"] + 1e-8)
                self.assertLessEqual(n["max_deviation_cents"], result["settings"]["max_excursion_cents"] + 1e-8)
                if n["directionality"] >= 0.8:
                    self.assertLessEqual(n["drift_cents"], result["settings"]["directional_drift_cents"] + 1e-8)
            self.assertLessEqual(n["drift_cents"], result["settings"]["max_drift_cents"] + 1e-8)
        token = re.compile(r'[=^_]?[A-Ga-g][,\']*\d*-?|z\d*|\|')
        for line in result["lines"]:
            music = line["music"]
            tokens = token.findall(music)
            self.assertEqual("".join(tokens), music.replace(" ", ""))
            pitches = [t for t in tokens if re.match(r"[=^_]?[A-Ga-g]", t)]
            self.assertEqual(len(pitches), len(line["lyrics"].split()))
            timed = [t.rstrip("-") for t in tokens if t != "|"]
            total = sum(int(re.search(r"\d+$", t).group()) if re.search(r"\d+$", t) else 1 for t in timed)
            expected = line["tick_end"] - line["tick_start"] if line["tick_start"] is not None else 0
            self.assertEqual(total, expected)
            for i in line["note_indices"]:
                note = result["notes"][i]
                self.assertGreaterEqual(note["tick_start"], line["tick_start"])
                self.assertLessEqual(note["tick_end"], line["tick_end"])
        self.assertIn("approximate transcription from a pitch track", abc)

    def reasons(self, result, scope=None):
        return [d["reason"] for d in result["declined"] if scope is None or d["scope"] == scope]

    def test_stepwise_scale_one_word_per_note(self):
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        words = [(w, i * 0.5, (i + 1) * 0.5)
                 for i, w in enumerate("dax vem lor nif pesh talu zim wo".split())]
        abc, result = transcribe(track(*[(p, 0.5) for p in pitches]), 100, words)
        self.assertEqual([n["midi"] for n in result["notes"]], pitches)
        self.assertEqual([n["word"] for n in result["notes"]], [w[0] for w in words])
        self.assertEqual(result["lines"][0]["lyrics"], "dax vem lor nif pesh talu zim wo")
        self.assertEqual(result["declined"], [])
        self.check_result(abc, result)

    def test_line_timing_uses_only_in_span_note_onsets(self):
        f0 = track(*[(p, .25) for p in (60, 62, 64, 65, 67, 69)])
        abc, result = transcribe(f0, 100, [("hmm", .8, .95)], key="C",
                                 line_text=["Copper sparks fly"], line_spans={0: (.25, 1.25)})
        self.assertEqual(result["lines"][0]["lyrics"], "* Copper * sparks fly *")
        self.assertIn("w: * Copper * sparks fly *\n", abc)
        self.assertEqual(result["lines"][0]["lyric_fit"], "onset_order")
        self.check_result(abc, result)

    def test_line_timing_does_not_put_a_whole_line_under_one_note(self):
        words = [("Copper", 0, .5), ("sparks", .5, 1)]
        abc, result = transcribe(track((60, 1)), 100, words, key="C", merge_words=True,
                                 line_text=["Copper sparks"], line_spans={0: (0, 1)})
        line = result["lines"][0]
        self.assertEqual(line["lyric_fit"], "unaligned")
        self.assertIn("\n% [0:00≈] Lyrics could not be aligned to notes: Copper sparks\n" + line["music"] + "\n", abc)
        self.assertNotIn("\nw:", abc)

    def test_full_line_uses_a_shared_note_and_cuts_at_the_lyric_boundary(self):
        f0 = track((60, 1.25), (62, .75), (None, 1))
        words = [("Copper", 0, .25), ("sparks", .3, 1.4),
                 ("Clouds", 1.5, 1.75), ("turn", 2.25, 2.75)]
        abc, result = transcribe(f0, 100, words, key="C", merge_words=True,
                                 line_breaks=[1, 2], line_text=["Copper sparks", "Clouds", "turn"],
                                 line_spans={0: (0, 1), 1: (1, 2), 2: (2, 3)}, spread_timed_words=True)
        a, b, c = result["lines"]
        self.assertEqual((a["lyric_fit"], a["lyrics"]), ("timed", "Copper~sparks"))
        self.assertEqual(b["lyrics"], "* Clouds")
        self.assertEqual(c["unaligned_reason"], "no stable notes")
        self.assertIn("Lyrics could not be aligned to notes (no stable notes): turn", abc)
        self.assertEqual([n["end"] for n in result["notes"]], [1., 1.25, 2.])

    def test_full_line_tries_distinct_onsets_after_a_crowded_timed_fit(self):
        words = [("Copper", 0, .25), ("sparks", .25, .5), ("fly", .5, .75)]
        f0 = track((60, .75), *[(p, .25) for p in (62, 64, 65, 67)])
        abc, result = transcribe(f0, 100, words, merge_words=True, spread_timed_words=True)
        self.assertEqual(result["lines"][0]["lyric_fit"], "onset_order")
        self.assertIn("\nw: Copper * sparks * fly\n", abc)
        self.check_result(abc, result)

        abc, result = transcribe(track((60, .75)), 100, words, merge_words=True, spread_timed_words=True)
        self.assertIn("Lyrics could not be aligned to notes (too few stable note onsets): Copper sparks fly", abc)
        self.assertNotIn("\nw:", abc)

    def test_full_line_uses_measured_placement_only_for_matching_words(self):
        f0 = track(*[(p, .25) for p in (60, 62, 64, 65, 67)])
        for first, fit, lyrics in [("copper", "timed", "* Copper _ * sparks"),
                                   ("silver", "onset_order", "Copper * * * sparks")]:
            with self.subTest(first=first):
                words = [(first, .25, .75), ("sparks", 1, 1.25)]
                abc, result = transcribe(f0, 100, words, merge_words=True, spread_timed_words=True,
                                         line_text=["Copper sparks"], word_text={0: "Copper"})
                self.assertEqual(result["lines"][0]["lyric_fit"], fit)
                self.assertIn(f"\nw: {lyrics}\n", abc)
                self.check_result(abc, result)

    def test_held_detuned_note_at_ears_fps(self):
        f0 = hz(np.full(125, 57.18))
        abc, result = transcribe(f0, 62.5, [("zora", 0, 2)])
        self.assertEqual(len(result["notes"]), 1)
        n = result["notes"][0]
        self.assertEqual((n["start"], n["end"], n["midi"]), (0, 2, 57))
        self.assertAlmostEqual(n["cents_off"], 18)
        self.assertIn("=A,2", abc)
        self.check_result(abc, result)

    def test_slide_between_stable_notes(self):
        f0 = np.concatenate([track((60, 0.5)), hz(np.linspace(60, 64, 40)), track((64, 0.5))])
        abc, result = transcribe(f0, 100, [("vori", 0, 1.4)])
        self.assertEqual([n["midi"] for n in result["notes"]], [60, 64])
        self.assertIn("unstable_pitch", self.reasons(result))
        self.assertIn(' z ', abc)
        self.assertNotIn('"^untranscribed"', abc)
        self.assertEqual(result["lines"][0]["lyrics"], "vori _")
        self.check_result(abc, result)

    def test_slow_slide_not_staircase(self):
        abc, result = transcribe(hz(np.linspace(60, 64, 300)), 100, [("savi", 0, 3)])
        self.assertEqual(result["notes"], [])
        self.assertIn("pitch_drift", self.reasons(result))
        self.assertIn("savi", abc)
        self.check_result(abc, result)

    def test_melisma(self):
        abc, result = transcribe(track((60, 0.5), (62, 0.5), (64, 0.5)), 100, [("vella", 0, 1.5)])
        self.assertEqual(result["lines"][0]["lyrics"], "vella _ _")
        self.assertEqual(result["words"][0]["note_indices"], [0, 1, 2])
        self.check_result(abc, result)

    def test_unvoiced_word_does_not_shift_following_word(self):
        abc, result = transcribe(track((60, 0.5), (None, 0.5), (64, 0.5)), 100,
                                 [("dax", 0, 0.5), ("hesh", 0.5, 1), ("nori", 1, 1.5)])
        self.assertEqual(result["lines"][0]["lyrics"], "dax nori")
        self.assertEqual(abc.count('1 words without a stable note: hesh'), 1)
        self.assertNotIn('unaligned words:', abc)
        self.assertEqual(result["words"][1]["status"], "no_stable_note")
        self.assertEqual(result["notes"][1]["word_index"], 2)
        self.check_result(abc, result)

    def test_octave_blip_declined_without_repair(self):
        abc, result = transcribe(track((60, 0.5), (72, 0.04), (60, 0.5)), 100, [("zelli", 0, 1.04)])
        self.assertEqual([n["midi"] for n in result["notes"]], [60, 60])
        rejected = [d for d in result["declined"] if d["reason"] == "implausible_leap"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["frames"], 4)
        self.assertAlmostEqual(rejected[0]["end"] - rejected[0]["start"], 0.04)
        self.assertAlmostEqual(result["notes"][0]["end"], 0.5)
        self.assertAlmostEqual(result["notes"][1]["start"], 0.54)
        self.check_result(abc, result)

    def test_initial_octave_blip_and_sustained_leap(self):
        for segments, pitches in [(((72, 0.04), (60, 0.5)), [60]),
                                  (((60, 0.5), (72, 0.5), (74, 0.5), (60, 0.5)), [60, 60])]:
            with self.subTest(segments=segments):
                abc, result = transcribe(track(*segments), 100)
                self.assertEqual([n["midi"] for n in result["notes"]], pitches)
                self.assertIn("implausible_leap", self.reasons(result))
                self.check_result(abc, result)

    def test_weak_low_note_between_octave_and_twelfth_neighbours_is_omitted(self):
        f0 = track((59, .9), (None, .58), (47, .14), (None, .16), (66, .45))
        probabilities = np.full(len(f0), .95)
        probabilities[148:162] = .08
        words = [("dax", 0, .9), ("veni", 1.48, 1.62), ("nori", 1.78, 2.23)]
        abc, result = transcribe(f0, 100, words, voiced_prob=probabilities, line_breaks=[1.2],
                                 **PROFILES["middle"])
        self.assertEqual([n["midi"] for n in result["notes"]], [59, 66])
        self.assertEqual(result["words"][2]["note_indices"], [1])
        self.assertEqual(result["words"][1]["status"], "no_stable_note")
        omitted = result["low_note_omissions"]
        self.assertEqual(len(omitted), 1)
        self.assertEqual(omitted[0]["midi"], 47)
        self.assertEqual(omitted[0]["neighbour_midi"], [59, 66])
        self.assertIn("% 1 low note left out: likely tracker octave error", abc)
        self.assertEqual([d["frames"] for d in result["declined"]
                          if d["reason"] == "likely_tracker_octave_error"], [14])
        self.check_result(abc, result)

    def test_confident_octave_leaps_survive_including_short_low_note(self):
        for pitches in ([60, 72, 60], [72, 60, 72], [59, 47, 66]):
            with self.subTest(pitches=pitches):
                f0 = track(*[(p, .2) for p in pitches])
                abc, result = transcribe(f0, 100, voiced_prob=np.full(len(f0), .95), **PROFILES["middle"])
                self.assertEqual([n["midi"] for n in result["notes"]], pitches)
                self.assertEqual(result["low_note_omissions"], [])
                self.check_result(abc, result)

    def test_isolated_octave_errors_are_omitted_in_either_direction(self):
        for pitch, field, register in ((73, "high_note_omissions", "high"), (49, "low_note_omissions", "low")):
            with self.subTest(register=register):
                f0 = track((61, .4), (None, .07), (pitch, .16), (None, .08), (63, .4))
                abc, result = transcribe(f0, 100, voiced_prob=np.full(len(f0), .95), **PROFILES["middle"])
                self.assertEqual([n["midi"] for n in result["notes"]], [61, 63])
                self.assertEqual([n["midi"] for n in result[field]], [pitch])
                self.assertIn(f"% 1 {register} note left out: likely tracker octave error", abc)
                self.check_result(abc, result)

    def test_held_octave_and_stepwise_arrival_are_kept(self):
        for before, held in ((61, .32), (71, .16)):
            with self.subTest(before=before, held=held):
                f0 = track((61, .4), (before, .12), (None, .07), (73, held), (None, .08), (62, .4))
                abc, result = transcribe(f0, 100, voiced_prob=np.full(len(f0), .95), **PROFILES["middle"])
                self.assertIn(73, [n["midi"] for n in result["notes"]])
                self.assertEqual(result["high_note_omissions"], [])
                self.check_result(abc, result)

    def test_short_leap_without_an_octave_witness_is_kept(self):
        f0 = track((62, .4), (None, .07), (72, .16), (None, .08), (62, .4))
        abc, result = transcribe(f0, 100, voiced_prob=np.full(len(f0), .95), **PROFILES["middle"])
        self.assertEqual([n["midi"] for n in result["notes"]], [62, 72, 62])
        self.assertEqual(result["high_note_omissions"], [])
        self.check_result(abc, result)

    def test_short_low_entrance_needs_octave_confirmation_and_reported_range(self):
        for witness in (55, 56):
            f0 = track((None, .25), (43, .10), (None, .04), (witness, .06),
                       (57, .18), (None, .09), (54, .19))
            options = dict(PROFILES["middle"], **dict(voiced_prob=np.full(len(f0), .95), range_low=52))
            abc, result = transcribe(f0, 100, **options)
            self.assertEqual([n["midi"] for n in result["low_note_omissions"]], [43] if witness == 55 else [])
            self.check_result(abc, result)

    def test_low_subharmonics_need_a_corroborating_return_to_the_same_register(self):
        for pitches, confidence, omitted in (
                ([67, 59, 48, 55, 67], [.69, .11, .16, .28, .23], [48, 55]),
                ([67, 59, 48, 55, 67], [.69, .11, .52, .58, .65], []),
                ([67, 59, 48, 55, 65], [.69, .11, .16, .28, .23], []),
                ([67, 59, 48, 56, 67], [.69, .11, .16, .28, .23], [])):
            with self.subTest(pitches=pitches, confidence=confidence):
                spans = list(zip(pitches, (.53, .1, .67, .21, .16), confidence))
                f0, probabilities = [], []
                for (pitch, seconds, prob), gap in zip(spans, (.1, .08, .56, .06, 0)):
                    f0.extend(track((pitch, seconds), (None, gap)))
                    probabilities.extend([prob] * round(seconds * 100) + [.01] * round(gap * 100))
                probabilities[-4:] = [.55] * 4
                abc, result = transcribe(f0, 100, voiced_prob=probabilities, **PROFILES["middle"])
                self.assertEqual([n["midi"] for n in result["low_note_omissions"]], omitted)
                self.assertEqual([n["midi"] for n in result["notes"]], [p for p in pitches if p not in omitted])
                self.assertEqual("likely tracker subharmonics" in abc, bool(omitted))
                self.check_result(abc, result)

    def test_quarter_frequency_filter_uses_confidence_and_comfortable_range(self):
        f0 = track((62, .3), (None, .1), (38, .36), (None, .1), (57, .3))
        for floor, prob, omitted in ((50, .1, [38]), (39, .1, []), (50, .55, [])):
            with self.subTest(floor=floor, prob=prob):
                p = np.full(len(f0), .8)
                p[40:76] = prob
                abc, result = transcribe(f0, 100, voiced_prob=p, range_low=floor, **PROFILES["middle"])
                self.assertEqual([n["midi"] for n in result["low_note_omissions"]], omitted)
                self.check_result(abc, result)

    def test_low_note_uses_nearest_confident_neighbours_across_weak_transition(self):
        f0 = track((59, .5), (None, .1), (47, .14), (None, .1), (55, .1), (None, .1), (66, .5))
        probabilities = np.full(len(f0), .95)
        probabilities[60:94] = .08
        abc, result = transcribe(f0, 100, voiced_prob=probabilities, **PROFILES["middle"])
        self.assertEqual([n["midi"] for n in result["notes"]], [59, 55, 66])
        self.assertEqual(len(result["low_note_omissions"]), 1)
        self.check_result(abc, result)

    def test_low_note_filter_requires_both_neighbours_and_measured_weakness(self):
        for gap, low_pitch, duration, low_prob, neighbour_prob in (
                (.1, 47, .14, None, .95), (.1, 47, .14, .5, .95),
                (.1, 47, .14, .1, .3), (.8, 47, .14, .1, .95),
                (.1, 47, .4, .1, .95), (.1, 50, .14, .1, .95)):
            with self.subTest(gap=gap, low_pitch=low_pitch, duration=duration,
                              low_prob=low_prob, neighbour_prob=neighbour_prob):
                f0 = track((59, .5), (None, gap), (low_pitch, duration), (None, gap), (66, .5))
                probabilities = np.full(len(f0), neighbour_prob)
                a = round((.5 + gap) * 100)
                probabilities[a:a + round(duration * 100)] = np.nan if low_prob is None else low_prob
                abc, result = transcribe(f0, 100, voiced_prob=probabilities, **PROFILES["middle"])
                self.assertIn(low_pitch, [n["midi"] for n in result["notes"]])
                self.assertEqual(result["low_note_omissions"], [])
                self.check_result(abc, result)

    def test_voicing_probabilities_must_match_frames_and_be_valid(self):
        for probabilities in ([.1], [[.1, .2]], [.1, 1.1], [-.1, .1], [np.inf, .1]):
            with self.subTest(probabilities=probabilities), self.assertRaisesRegex(ValueError, "voiced_prob"):
                transcribe([220, 220], 100, voiced_prob=probabilities)

    def test_quantised_variable_beat_grid(self):
        abc, result = transcribe(track((60, 0.5), (62, 0.6), (64, 0.7)), 100,
                                 [("ka", 0, 0.5), ("ve", 0.5, 1.1), ("zo", 1.1, 1.8)],
                                 beats=[0, 0.5, 1.1, 1.8], tempo_bpm=120, subdivisions=4)
        self.assertEqual(result["rhythm"]["mode"], "quantised")
        self.assertEqual(result["rhythm"]["subdivisions_per_beat"], 2)
        self.assertEqual([(n["tick_start"], n["tick_end"]) for n in result["notes"]], [(0, 2), (2, 4), (4, 6)])
        self.assertIn("L:1/8", abc)
        self.assertIn("Q:1/4=120", abc)
        self.assertNotIn("M:", abc)
        self.assertNotIn("|", result["lines"][0]["music"])
        self.check_result(abc, result)

    def test_proportional_line_calibration_does_not_accumulate(self):
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        abc, result = transcribe(track(*[(p, 0.21) for p in pitches]), 100,
                                 unit_seconds=0.1, tempo_bpm=120)
        self.assertEqual(result["rhythm"]["mode"], "proportional")
        self.assertEqual(sum(n["tick_end"] - n["tick_start"] for n in result["notes"]), 16)
        self.assertAlmostEqual(result["notes"][-1]["written_end"], len(pitches) * .21)
        self.assertIn("NOT metrical", abc)
        self.assertNotIn("Q:", abc)
        self.assertNotIn("M:", abc)
        self.assertNotIn("K:", abc)
        self.assertNotIn("|", result["lines"][0]["music"])
        self.check_result(abc, result)

    def test_only_supplied_downbeats_make_bars_and_tied_lyrics(self):
        abc, result = transcribe(track((60, 2)), 100, [("zor", 0, 2)],
                                 beats=[0, 0.5, 1, 1.5, 2], downbeats=[1], meter="2/4", key="A minor")
        self.assertIn("M:2/4", abc)
        self.assertIn("K:Am", abc)
        self.assertEqual(result["lines"][0]["music"], "C8- | C8")
        self.assertEqual(result["lines"][0]["lyrics"], "zor _")
        self.assertEqual(len(result["notes"]), 1)
        self.check_result(abc, result)

    def test_explicit_dotted_beat_unit_for_compound_metre(self):
        abc, result = transcribe(track((60, 1)), 100, beats=[0, 0.5, 1], downbeats=[1],
                                 meter="6/8", beat_unit="3/8", tempo_bpm=120)
        from fractions import Fraction
        ticks = result["lines"][0]["tick_end"] - result["lines"][0]["tick_start"]
        self.assertEqual(ticks * Fraction(result["rhythm"]["L"]), Fraction(6, 8))
        self.assertIn("Q:3/8=120", abc)
        self.assertEqual(result["rhythm"]["beat_unit"], "3/8")
        self.check_result(abc, result)

    def test_wordless_note_and_rests_do_not_consume_words(self):
        abc, result = transcribe(track((60, 0.5), (None, 0.5), (62, 0.5), (64, 0.5)), 100,
                                 [("kavi", 1, 2)])
        self.assertEqual(result["lines"][0]["lyrics"], "* kavi _")
        self.check_result(abc, result)

    def test_line_boundaries_and_same_pitch_different_words(self):
        abc, result = transcribe(track((60, 2)), 100,
                                 [("zef", 0, 0.5), ("tavi", 0.5, 1), ("mori", 1, 2)], line_breaks=[1])
        self.assertEqual(len(result["lines"]), 2)
        self.assertEqual([line["lyrics"] for line in result["lines"]], ["zef tavi", "mori"])
        self.assertEqual(abc.count("\nw:"), 2)
        self.assertEqual(len(result["notes"]), 3)
        self.check_result(abc, result)

    def test_key_signature_and_natural_cancellations(self):
        abc, result = transcribe(track((61, 0.5), (60, 0.5), (65, 0.5), (72, 0.5)), 100, key="D major")
        self.assertEqual(result["lines"][0]["music"], "C2 =C2 =F2 =c2")
        self.check_result(abc, result)
        abc, result = transcribe(track((36, 0.5)), 100)
        self.assertIn("=C,,2", abc)
        self.check_result(abc, result)

    def test_collapsed_notes_and_grid_coverage_are_declined(self):
        abc, result = transcribe(track((60, 0.2), (62, 0.2), (64, 0.2)), 100,
                                 [("zavi", 0, 0.2), ("ka", 0.2, 0.4), ("nom", 0.4, 0.6)],
                                 beats=[0.1, 0.6], subdivisions=1)
        self.assertIn("outside_beat_grid", self.reasons(result, "note"))
        self.assertIn("duration_collapsed_on_grid", self.reasons(result, "note"))
        self.assertTrue(any(w["status"] == "no_renderable_note" for w in result["words"]))
        self.check_result(abc, result)

    def test_invalid_f0_overlap_and_out_of_track_word(self):
        abc, result = transcribe([np.nan, 0, -2, np.inf] + [float(hz(60))] * 50, 100,
                                 [("pavi", 0.04, 0.4), ("veni", 0.1, 0.54), ("zom", 2, 3)])
        self.assertIn("invalid_pitch", self.reasons(result))
        self.assertIn("ambiguous_word_overlap", self.reasons(result))
        self.assertEqual(result["words"][2]["status"], "no_stable_note")
        self.assertIn("zom", abc)
        self.check_result(abc, result)

    def test_empty_and_all_unvoiced(self):
        for f0, words in [([], []), ([], [("zavi", 0, 1)]), ([np.nan] * 50, [("pesh", 0, 0.5)])]:
            with self.subTest(f0=f0):
                abc, result = transcribe(f0, 100, words)
                self.assertEqual(result["notes"], [])
                self.check_result(abc, result)

    def test_lyric_operators_are_literal(self):
        abc, result = transcribe(track((60, 0.5)), 100, [(r"ze-*_%|~\&\"$", 0, 0.5)])
        self.assertIn(r"ze\-\u002a\u005f\u0025\u007c\u007e", result["lines"][0]["lyrics"])
        self.check_result(abc, result)

    def test_fractional_frame_line_cuts_do_not_duplicate_time(self):
        abc, result = transcribe(track((60, 1)), 100, line_breaks=[0.505], unit_seconds=0.005)
        self.assertEqual(result["settings"]["frame_line_breaks"], [0.5])
        self.assertEqual(result["lines"][0]["end"], 0.505)
        self.check_result(abc, result)

    def test_random_line_cuts_preserve_duration(self):
        rng = np.random.default_rng(381)
        for _ in range(100):
            fps = float(rng.choice([62.5, 100, 125]))
            pitches = rng.choice([60, 62, 64, 65], size=10)
            f0 = np.repeat(hz(pitches), rng.integers(1, 45, size=10))
            duration = len(f0) / fps
            cuts = sorted(rng.uniform(0, duration, size=4))
            kwargs = {"beats": np.linspace(0, duration, 6), "downbeats": [0, duration]} if rng.random() < 0.5 else {}
            abc, result = transcribe(f0, fps, line_breaks=cuts, min_note_s=0.04, unit_seconds=0.005, **kwargs)
            self.check_result(abc, result)

    def test_unused_grid_and_unmatched_downbeat_are_reported(self):
        abc, result = transcribe(track((60, 1)), 100, beats=[0, 0.5, 1], downbeats=[0.3])
        self.assertIn("downbeat_not_on_supplied_beat_grid", self.reasons(result))
        self.assertEqual(result["rhythm"]["bar_ticks"], [])
        self.check_result(abc, result)
        abc, result = transcribe(track((60, 1)), 100, mode="proportional", beats=[0, 1])
        self.assertIn("beats_unused_in_proportional_mode", self.reasons(result))
        self.check_result(abc, result)

    def test_short_and_out_of_tolerance_notes_are_not_promoted(self):
        for f0 in [track((60, 0.05)), track((60.4, 0.5))]:
            abc, result = transcribe(f0, 100)
            self.assertEqual(result["notes"], [])
            self.check_result(abc, result)

    def test_validation(self):
        bad = [{"fps": 0}, {"fps": np.nan}, {"f0": [[440]]}, {"beats": [0]},
               {"beats": [0, 0.5, 0.4]}, {"downbeats": [np.nan]}, {"key": "Z minor"},
               {"key": "C\nK:G"}, {"meter": "4/4"}, {"subdivisions": 1.5},
               {"subdivisions": True}, {"beat_unit": "0/4"}, {"tolerance_cents": 50}, {"unit_seconds": 0},
               {"mode": "imagined"}, {"line_breaks": [0]}, {"line_breaks": [1]},
               {"words": [("ze", 0.4, 0.2)]}, {"words": [("ze\nw: bad", 0, 0.5)]},
               {"words": [("ze", 0, 0.5)], "line_breaks": [0.25]},
               {"pitch_rule": "imagined"}, {"spread_cents": 0}, {"centre_window_s": 0},
               {"merge_words": "yes"}, {"directional_drift_cents": 0}, {"max_excursion_cents": 0}]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                inputs = {"f0": track((60, 0.5)), "fps": 100}
                inputs.update(kwargs)
                transcribe(**inputs)

    def test_json_cli_and_cached_word_schema(self):
        # tempfile stays in the workspace, as required by the house rules.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source, words, abc, data = [root / name for name in ("input.json", "words.json", "out.abc", "out.json")]
            source.write_text(json.dumps({"f0": track((60, 0.5)).tolist(), "fps": 100}))
            words.write_text(json.dumps({"v": 5, "words": [["zavi", 0, 0.5]]}))
            main(["--input", str(source), "--words", str(words), "--output", str(abc), "--json-output", str(data)])
            result = json.loads(data.read_text())
            self.assertIn("w: zavi", abc.read_text())
            self.check_result(abc.read_text(), result)

    def test_audio_cli_calls_only_existing_front_end_with_fakes(self):
        import sys
        import types
        fake = types.SimpleNamespace(SR=16000, FPS=62.5)
        calls = []
        fake.load_audio = lambda path, sr: calls.append((path, sr)) or np.zeros(100)
        fake._front_end = lambda audio, **kwargs: calls.append(kwargs) or {"f0": hz(np.full(32, 60))}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            words = Path(directory) / "words.json"
            words.write_text(json.dumps([["zavi", 0, 0.512]]))
            with patch.dict(sys.modules, {"raven_ears_plus": fake}), contextlib.redirect_stdout(io.StringIO()) as output:
                main(["--vocals", "invented-stem.wav", "--words", str(words)])
        self.assertEqual(calls, [("invented-stem.wav", 16000), {"fmin": 65.0, "fmax": 1400.0}])
        self.assertIn("w: zavi", output.getvalue())

    def test_centre_accepts_vibrato_measuring_raw_spread(self):
        t = np.arange(100) / 100
        for profile, amplitude in [("middle", .45), ("loose", .65)]:
            f0 = hz(60 + amplitude * np.sin(2 * np.pi * 6 * t))
            abc, r = transcribe(f0, 100, [("vemi", 0, 1)], **PROFILES[profile])
            self.assertEqual([n["midi"] for n in r["notes"]], [60])
            n = r["notes"][0]
            self.assertAlmostEqual(n["max_deviation_cents"], 100 * amplitude * np.max(np.abs(np.sin(2 * np.pi * 6 * t))))
            self.assertAlmostEqual(n["spread_cents"], np.percentile(100 * amplitude * np.sin(2 * np.pi * 6 * t), 90) -
                                   np.percentile(100 * amplitude * np.sin(2 * np.pi * 6 * t), 10))
            self.check_result(abc, r)
            self.assertEqual(transcribe(f0, 100)[1]["notes"], [])

    def test_scooped_onset_retains_plateau_without_flattening_slide(self):
        f0 = hz(np.r_[np.linspace(60, 64, 12), np.full(60, 64)])
        abc, r = transcribe(f0, 100, [("loma", 0, .72)], **PROFILES["middle"])
        self.assertEqual([n["midi"] for n in r["notes"]], [64])
        self.assertGreaterEqual(r["notes"][0]["start"], .09)
        self.assertLessEqual(r["notes"][0]["start"], .12)
        self.assertTrue(any(d["scope"] == "pitch" and d["start"] == 0 for d in r["declined"]))
        self.check_result(abc, r)

    def test_fast_syllables_are_retained_and_rendered(self):
        f0 = track((60, .08), (62, .08), (64, .08))
        words = [("za", 0, .08), ("vi", .08, .16), ("nu", .16, .24)]
        self.assertEqual(transcribe(f0, 100, words)[1]["notes"], [])
        abc, r = transcribe(f0, 100, words, **PROFILES["middle"])
        self.assertEqual([n["midi"] for n in r["notes"]], [60, 62, 64])
        self.assertTrue(all(w["status"] == "aligned" for w in r["words"]))
        self.check_result(abc, r)

    def test_word_boundary_merge_shares_one_note_without_attacks(self):
        words = [("za", 0, .05), ("nu", .05, .1), ("vi", .1, .2)]
        abc, r = transcribe(track((60, .2)), 100, words, **PROFILES["middle"])
        self.assertEqual(len(r["notes"]), 1)
        self.assertEqual(r["notes"][0]["word_indices"], [0, 1, 2])
        self.assertEqual(r["notes"][0]["word_overlap_s"], [.05, .05, .1])
        self.assertEqual(r["lines"][0]["lyrics"], "za~nu~vi")
        self.assertTrue(all(w["note_indices"] == [0] for w in r["words"]))
        self.check_result(abc, r)

    def test_merge_stops_at_lyric_line_and_keeps_melisma_slots(self):
        abc, r = transcribe(track((60, .2), (62, .2), (62, .2)), 100,
                            [("za", 0, .3), ("nu", .3, .4), ("vi", .4, .6)],
                            line_breaks=[.4], **PROFILES["middle"])
        self.assertEqual([line["lyrics"] for line in r["lines"]], ["za nu", "vi"])
        self.assertEqual([n["word_indices"] for n in r["notes"]], [[0], [0, 1], [2]])
        self.check_result(abc, r)

    def test_centre_does_not_bridge_unvoiced_overlap_or_octave_flags(self):
        for f0, words, reason in [
                (track((60, .3), (None, .02), (60, .3)), [], "unvoiced"),
                (track((60, .3), (72, .02), (60, .3)), [], "implausible_leap"),
                (track((60, .62)), [("za", 0, .32), ("vi", .3, .62)], "ambiguous_word_overlap")]:
            abc, r = transcribe(f0, 100, words, **PROFILES["middle"])
            self.assertEqual(len(r["notes"]), 2)
            self.assertAlmostEqual(r["notes"][0]["end"], .3)
            self.assertAlmostEqual(r["notes"][1]["start"], .32)
            self.assertIn(reason, self.reasons(r, "pitch"))
            self.check_result(abc, r)

    def test_centre_rejects_excess_spread_and_detuned_centre(self):
        for f0 in [hz(60 + .45 * np.sin(2 * np.pi * 6 * np.arange(100) / 100)), track((60.45, 1))]:
            abc, r = transcribe(f0, 100, pitch_rule="centre", spread_cents=20)
            self.assertEqual(r["notes"], [])
            self.assertIn("unstable_pitch", self.reasons(r))
            self.check_result(abc, r)

    def test_centre_directional_slide_does_not_become_scale(self):
        for profile in ("middle", "loose"):
            abc, r = transcribe(hz(np.linspace(60, 64, 300)), 100, **PROFILES[profile])
            self.assertEqual(r["notes"], [])
            self.assertIn("pitch_drift", self.reasons(r))
            self.check_result(abc, r)

    def test_short_large_excursion_cannot_hide_in_percentile_tails(self):
        f0 = track((60, .5), (63, .02), (60, .5))
        abc, r = transcribe(f0, 100, **PROFILES["middle"])
        self.assertEqual(len(r["notes"]), 2)
        self.assertEqual([n["midi"] for n in r["notes"]], [60, 60])
        self.assertAlmostEqual(r["notes"][0]["end"], .5)
        self.assertAlmostEqual(r["notes"][1]["start"], .52)
        self.assertIn("unstable_pitch", self.reasons(r))
        self.check_result(abc, r)

    def test_lyric_helper_repeated_lines_and_word_safe_rounding(self):
        words = [("Zavi", .02, .31), ("nolu", .31, .81), ("tavi", 1, 1.5),
                 ("zavi", 2, 2.5), ("nolu", 2.5, 3)]
        cuts, rows, summary = lyric_line_boundaries(["zavi", "nolu tavi", "zavi nolu"], words, 4)
        self.assertEqual(cuts, [.31, 2])
        self.assertEqual(summary["approx"], 0)
        self.assertEqual(len(rows), 3)
        abc, r = transcribe(track((60, 4)), 100, words, line_breaks=cuts, **PROFILES["middle"])
        self.assertEqual(abc.count("\nw:"), 3)
        self.assertEqual(len(re.findall(r"^% \[\d+:\d\d\]$", abc, re.M)), 3)
        self.check_result(abc, r)

    def test_lyric_helper_discloses_missing_line(self):
        cuts, rows, summary = lyric_line_boundaries(
            ["zavi nolu", "vemi loma", "tavi pesh"],
            [("zavi", 0, .5), ("nolu", .5, 1), ("tavi", 4, 4.5), ("pesh", 4.5, 5)], 6)
        self.assertTrue(rows[1]["approx"])
        self.assertEqual(summary["approx"], 1)
        self.assertLess(cuts[0], cuts[1])

    def test_cli_profile_lines_and_f0_cache_with_fakes(self):
        import sys
        import types
        fake = types.SimpleNamespace(SR=16000, FPS=100)
        fake.load_audio = lambda *args: np.zeros(100)
        fake._front_end = lambda *args, **kwargs: {
            "f0": track((60, .2), (None, .1), (62, .2)),
            "voiced_prob": np.r_[np.full(20, .95), np.full(10, np.nan), np.full(20, .95)]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            words, lines, cache, output = [root / name for name in ("words.json", "lines.txt", "f0.json", "out.json")]
            words.write_text(json.dumps([["zavi", 0, .2], ["nolu", .3, .5]]))
            lines.write_text("zavi\nnolu\n")
            with patch.dict(sys.modules, {"raven_ears_plus": fake}), contextlib.redirect_stdout(io.StringIO()) as captured:
                main(["--vocals", "invented.wav", "--words", str(words), "--lyric-lines", str(lines),
                      "--save-f0", str(cache), "--profile", "middle", "--key", "C minor", "--tempo-bpm", "117",
                      "--json-output", str(output)])
            cached = json.loads(cache.read_text())
            self.assertIsNone(cached["f0"][20])
            self.assertIsNone(cached["voiced_prob"][20])
            self.assertEqual(cached["voiced_prob"][:20], [.95] * 20)
            r = json.loads(output.read_text())
            self.assertEqual(r["settings"]["pitch_rule"], "centre")
            self.assertEqual(r["lyric_alignment"]["summary"]["total"], 2)
            self.assertIn("K:Cm", captured.getvalue())
            self.assertEqual(r["rhythm"]["fallback_reason"], "poor grid fit")
            self.assertLessEqual(len(captured.getvalue().split("I:abc-version")[0].splitlines()), 4)
            self.check_result(captured.getvalue(), r)

    def test_flat_and_sharp_keys_preserve_measured_pitches(self):
        for key, pitches, expected in [
                ("C minor", [60, 63, 67, 70], "C2 E2 G2 B2"),
                ("E major", [64, 66, 68, 71], "E2 F2 G2 B2"),
                ("C# major", [60, 61, 65, 66], "B,2 C2 E2 F2"),
                ("Cb major", [59, 60, 64, 65], "C2 =C2 F2 =F2")]:
            with self.subTest(key=key):
                abc, r = transcribe(track(*[(p, .5) for p in pitches]), 100, key=key)
                self.assertEqual(r["lines"][0]["music"], expected)
                events = decode_abc(abc, r["rhythm"], r["lines"])
                self.assertEqual([e["midi"] for e in events], pitches)
                self.assertIn("caller supplied, not measured here", abc)
                self.check_result(abc, r)

    def test_outside_key_accidental_and_restore_across_line(self):
        abc, r = transcribe(track((63, .5), (64, .5), (63, .5)), 100,
                            key="C minor", line_breaks=[1])
        self.assertEqual([l["music"] for l in r["lines"]], ["E2 =E2", "_E2"])
        self.assertEqual([e["midi"] for e in decode_abc(abc, r["rhythm"], r["lines"])], [63, 64, 63])
        self.check_result(abc, r)

    def test_chromatic_degrees_and_local_spelling_keep_the_sounding_pitch(self):
        f0 = track((66, .5), (61, .5), (65, .5), (66, .5))
        from keyfuse import spelling_context
        abc, result = transcribe(f0, 100, key="G major", spelling_at=spelling_context("G", [(1.5, 2, "Db")]))
        self.assertEqual(result["lines"][0]["music"], "F2 ^C2 =F2 _G2")
        self.assertEqual([e["midi"] for e in decode_abc(abc, result["rhythm"], result["lines"])],
                         [66, 61, 65, 66])
        self.check_result(abc, result)

    def test_tempo_only_recovers_known_sixteenth_rhythm(self):
        f0 = track((60, .25), (62, .125), (64, .125), (65, .5), fps=200)
        abc, r = transcribe(f0, 200, tempo_bpm=120, min_note_s=.08)
        self.assertEqual(r["rhythm"]["mode"], "quantised")
        self.assertIn("L:1/16", abc)
        self.assertIn("Q:1/4=120", abc)
        self.assertEqual([(n["tick_start"], n["tick_end"]) for n in r["notes"]], [(0, 2), (2, 3), (3, 4), (4, 8)])
        self.assertAlmostEqual(r["rhythm"]["fit"]["p90_ms"], 0)
        self.assertAlmostEqual(r["self_check"]["summary"]["median_pitch_error_cents"], 0)
        self.assertEqual(r["self_check"]["summary"]["voiced_coverage_share"], 1)
        self.check_result(abc, r)

    def test_poor_grid_fit_falls_back_explicitly(self):
        f0 = track(*zip([60, 62, 64, 65, 67, 65, 64, 62], [.19, .23, .31, .41, .28, .34, .27, .29]))
        abc, r = transcribe(f0, 100, tempo_bpm=120)
        self.assertEqual(r["rhythm"]["mode"], "proportional")
        self.assertIn("poor grid fit", abc)
        self.assertEqual(len(r["rhythm"]["candidate_fits"]), 2)
        self.assertNotIn("Q:", abc)
        self.assertTrue(all(n["tick_end"] - n["tick_start"] in (1, 2, 4) for n in r["notes"]))
        self.check_result(abc, r)

    def test_hook_indices_and_time_range_select_whole_lyric_lines(self):
        f0 = track((60, .5), (62, .5), (64, .5), (65, .5))
        for selection, expected in [({"line_indices": [1, 3]}, [1, 3]),
                                    ({"time_range": [.6, 1.4]}, [1, 2])]:
            abc, r = transcribe(f0, 100, line_breaks=[.5, 1, 1.5], **selection)
            self.assertEqual([line["index"] for line in r["lines"]], expected)
            self.assertEqual(abc.count("\nw:"), 2)
            self.assertEqual([n["line"] for n in r["notes"] if n["rendered"]], expected)
            self.assertEqual([row["line"] for row in r["self_check"]["lines"]], expected)
            self.check_result(abc, r)

    def test_round_trip_detects_changed_abc_and_tracks_missing_voice(self):
        f0 = track((60, .5), (60.4, .5), (60, .5))
        abc, r = transcribe(f0, 100, key="C major", line_breaks=[1])
        self.assertAlmostEqual(r["self_check"]["summary"]["median_pitch_error_cents"], 0)
        self.assertEqual(r["self_check"]["lines"][0]["voiced_coverage_share"], .5)
        altered = abc.replace("\nC2\n", "\nD2\n")
        checked = self_check(altered, f0, 100, r["rhythm"], r["lines"])
        self.assertAlmostEqual(checked["summary"]["median_pitch_error_cents"], 200)

    def test_long_rest_is_capped_and_line_padding_trimmed(self):
        f0 = track((60, .5), (None, 10), (62, .5))
        abc, r = transcribe(f0, 100)
        self.assertEqual(r["lines"][0]["music"], "=C2 z4 =D2")
        self.check_result(abc, r)
        abc, r = transcribe(f0, 100, line_breaks=[5.5])
        self.assertNotIn("z", " ".join(l["music"] for l in r["lines"]))
        self.assertEqual(r["lines"][0]["trimmed_trailing_s"], 5)
        self.assertEqual(r["lines"][1]["trimmed_leading_s"], 5)
        self.check_result(abc, r)

    def test_downbeats_without_metre_do_not_make_bars(self):
        abc, r = transcribe(track((60, 1)), 100, beats=[0, .5, 1], downbeats=[.5])
        self.assertEqual(r["rhythm"]["bar_ticks"], [])
        self.assertNotIn("|", abc)
        self.assertIn("metre_not_supplied", self.reasons(r))

    def test_inconsistent_downbeats_do_not_make_bars(self):
        abc, r = transcribe(track((60, 2)), 100, beats=[0, .5, 1, 1.5, 2],
                            downbeats=[0, .5, 2], meter="4/4")
        self.assertNotIn("M:", abc)
        self.assertNotIn("|", abc)
        self.assertIn("downbeats_inconsistent_with_metre", self.reasons(r))

    def test_selection_validation(self):
        for selection in [{"line_indices": []}, {"line_indices": [1]}, {"line_indices": [True]},
                          {"time_range": [0, 2]}, {"time_range": [.5, .5]},
                          {"line_indices": [0], "time_range": [0, .5]}]:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                transcribe(track((60, 1)), 100, **selection)

    def test_cli_hook_and_melody_override_input_selection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory) / "input.json"
            source.write_text(json.dumps({"f0": track((60, .5), (62, .5)).tolist(), "fps": 100,
                                          "line_breaks": [.5], "line_indices": [0]}))
            for option, count in [(["--line-indices", "1"], 1), (["--melody"], 2)]:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main(["--input", str(source)] + option)
                self.assertEqual(output.getvalue().count("\nw:"), count)
                if count == 1:
                    self.assertIn("=D2", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
