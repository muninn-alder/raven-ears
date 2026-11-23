"""Saved meter policy checks with invented evidence; python3 test_meter_trust.py."""
import copy
import unittest

import meter_trust as M


def windows(labels, confidence=0.95):
    return [[i * 15.0, label, confidence] for i, label in enumerate(labels)]


def changing(confidence=0.95):
    return {"verdict": "4/4 (simple)", "confidence": 0.8,
            "beats_per_bar": 4, "bar_consistency": 0.98,
            "windows": windows([4] * 4 + [3] * 4, confidence),
            "window_bars": [{"beats_per_bar": n, "bar_consistency": 1.0} for n in [4] * 4 + [3] * 4],
            "changes": [{"from": 4, "to": 3, "lo_s": 45, "hi_s": 60}],
            "change_chain": ["4/4", "3/4"]}


def section(start, end, name, confidence, bpb, bar_s, **extra):
    return {"start": start, "end": end, "label": name,
            "data": dict(verdict=name, confidence=confidence, beats_per_bar=bpb,
                         bar_consistency=0.98, bar_s=bar_s, **extra)}


class MeterTrustTests(unittest.TestCase):
    @staticmethod
    def tied_cycle():
        # Measured 13 Nov orchestral row: broad recurrence, a lone five-pulse
        # template, and inconsistent tracker bars. Labels/identities invented.
        return dict(engine="beat_this", verdict="5/4 (3+2)", confidence=1.0,
                    beats_per_bar=2, bar_consistency=0.25, tracker_bpm=157.9,
                    cycle_pulses=5, cycle_source="accent autocorrelation", cycle_strength=0.68,
                    cycle_acf={"2": .52, "3": .54, "4": .54, "5": .66, "6": .65,
                               "7": .68, "8": .63, "9": .54, "10": .59, "11": .56, "12": .56},
                    grouping="3+2", grouping_confidence=1.0,
                    grouping_scores=[{"grouping": "2+3", "r": .98}])

    def test_tied_cycles_cannot_borrow_certainty_from_a_single_grouping(self):
        d = self.tied_cycle()
        before = copy.deepcopy(d)
        self.assertEqual(M.cycle_support(d)["confidence"], 0)
        self.assertIsNone(M.one(d))
        text = "METER: 5/4 (3+2) · cycle 5 pulses · grouping confidence 1.00\n  measured detail\n"
        out = M.meter_text(text, d)
        self.assertIn("meter unclear: 5, 6, 7 and 8 pulses all fit about equally", out)
        self.assertIn("cycle support 0.00", out)
        self.assertIn("grouping 3+2 assumes 5 pulses", out)
        self.assertNotIn("1.00", out)
        self.assertNotIn("5/4", out)
        self.assertEqual(M.meter_text(out, d), out)
        tempo = M.tempo_text("TEMPO: 157 BPM · meter 5/4, see METER", d)
        self.assertIn("meter unclear", tempo)
        self.assertNotIn("5/4", tempo)
        self.assertEqual(d, before)

    def test_exact_tie_and_flat_acf_have_no_winner(self):
        d = self.tied_cycle()
        for acf in ({"5": .66, "7": .66}, {"5": .66, "7": .60},
                    {str(k): .66 for k in range(2, 13)}):
            d["cycle_acf"] = acf
            self.assertEqual(M.cycle_support(d)["confidence"], 0)
            self.assertIn("meter unclear", M.assess(d)[0])
            self.assertIsNone(M.one(d))

    def test_period_multiples_do_not_demote_the_real_odd_controls(self):
        # From Eden (five/ten) and Tolerate It (ten) measured ACF shapes.
        for k, acf, name in (
                (5, {"2": -.21, "3": -.22, "4": .24, "5": .51, "6": .12,
                     "7": -.25, "8": -.02, "9": .12, "10": .49, "11": .03, "12": -.13}, "5/4"),
                (10, {"2": -.03, "3": .19, "4": .18, "5": -.05, "6": .12,
                      "7": .11, "8": -.06, "9": .09, "10": .58, "11": .07, "12": -.06}, "10/8")):
            d = dict(self.tied_cycle(), cycle_pulses=k, cycle_acf=acf, verdict=name)
            self.assertIsNone(M.cycle_unclear(d))
            self.assertEqual(M.one(d), name)
        # Agreement with steady bars is independent evidence; a strong
        # conflicting accent cycle must not overrule that silently.
        d = dict(d, beats_per_bar=4, bar_consistency=.98)
        self.assertLessEqual(M.cycle_support(d)["confidence"], .03)
        self.assertIsNone(M.one(d))
        # Solsbury Hill: an unreliable one-pulse downbeat tracker must not
        # erase the clear seven-pulse accent cycle, nor count as agreement.
        d = dict(self.tied_cycle(), beats_per_bar=1, bar_consistency=.63,
                 cycle_pulses=7, verdict="7/4", confidence=.58,
                 cycle_acf={"2": -.12, "3": .27, "4": .27, "5": -.10, "6": .08,
                            "7": .33, "8": .06, "9": -.10, "10": .07, "11": .24, "12": -.05})
        self.assertEqual(M.one(d), "7/4")
        self.assertIsNone(M.tracker_unclear(d))
        self.assertFalse(M.cycle_support(d)["tracker_agrees"])

    def test_tied_saved_sections_and_all_card_views_withhold_the_name(self):
        import listen
        import slim
        d = self.tied_cycle()
        sections = {"sections": [dict(start=0, end=309.1, label="5/4", data=d)]}
        raw = "METER: 0:00–5:09 5/4\n  [0:00–5:09] 5/4 · grouping confidence 1.00\n"
        out = M.meter_text(raw, sections)
        self.assertNotIn("5/4", out)
        self.assertNotIn("1.00", out)
        self.assertEqual(M.meter_text(out, sections), out)
        song = {"cards": {"music": {"text": "MUSIC: example.mp3 · 5:09\nTEMPO: 157 BPM · meter 5/4, see METER", "json": {}},
                          "singer": {"text": raw, "json": {"meter": sections}},
                          "harmony": {"text": "", "json": {}}},
                "manifest": {"passes": {n: {"status": "ran"} for n in ("music", "singer", "harmony")}}}
        for text in (listen.full_text(song), slim.render_first_listen(
                {n: c["text"] for n, c in song["cards"].items()},
                {n: c["json"] for n, c in song["cards"].items()})):
            self.assertIn("meter unclear", text)
            self.assertNotIn("5/4", text)
            self.assertNotIn("1.00", text)
        # The older eight-pulse naming rule must not bypass the cycle tie.
        eight = dict(d, cycle_pulses=8, grouping="3+3+2", verdict="8/8")
        multi = {"sections": [dict(start=0, end=150, label="8/8", data=eight),
                              dict(start=150, end=309.1, label="8/8", data=eight)]}
        self.assertIsNone(M.accent_cycle_ambiguity(eight))
        self.assertIsNone(M.assess(multi)[0])
        out = M.meter_text("METER: 8/8 then 8/8\n  [0:00–2:30] 8/8 · grouping confidence 1.00\n"
                           "  [2:30–5:09] 8/8 · grouping confidence 1.00", multi)
        self.assertIn("meter unclear", out.splitlines()[0])
        self.assertNotIn("1.00", out)
        changed = dict(d, changes=[dict(from_=5, to=7)], change_chain=["5/4", "7/4"])
        out = M.meter_text("METER: 5/4 · grouping confidence 1.00", changed)
        self.assertNotIn("1.00", out)

    def test_fresh_meter_and_windows_reject_the_measured_tie(self):
        import numpy as np
        import singer
        from unittest.mock import patch
        d = self.tied_cycle()
        b = np.arange(563) * .38
        # No invented sound or calibration: mock the saved detector evidence.
        env = np.zeros(19000)
        pre = (b, b[::2], "beat_this", env, 86.13, {"mix": env})
        acf = {int(k): v for k, v in d["cycle_acf"].items()}
        with patch.object(singer, "cycle_from_acf", return_value=(acf, 5, .68)), \
             patch.object(singer, "fold_profile", return_value=np.array([-.11, .2, -.06, .31, -.1])):
            lines, data = singer.meter_block("unused", "unused", 309.1, _pre=pre)
            self.assertEqual(data["confidence"], 0)
            self.assertIn("meter unclear", lines[0])
            self.assertNotIn("1.00", lines[0])
            self.assertEqual(singer.window_cycle(np.zeros(len(b)), b, .38), (None, 0, None))

    def test_meter_evidence_survives_real_worker_and_song_save(self):
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        from unittest.mock import patch
        import listen
        # Run the actual singer main and listen worker in a child process;
        # replace only the audio front end with the measured evidence above.
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", listen.HERE)) as directory:
            root = Path(directory)
            audio = root / "example.mp3"
            audio.touch()
            stems = root / "stems"
            stems.mkdir()
            home = root / "ears"
            (home / "songs").mkdir(parents=True)
            acf = {int(k): v for k, v in self.tied_cycle()["cycle_acf"].items()}
            bootstrap = f"""
import sys
import numpy as np
import soundfile as sf
import singer
import listen
from types import SimpleNamespace
b = np.arange(563) * .38
singer.get_stems = lambda *args: ({str(stems)!r}, True)
sf.info = lambda *args: SimpleNamespace(duration=309.1)
singer.beats_downbeats = lambda *args: (b, b[::2], 'beat_this')
singer._env = lambda *args: (np.zeros(19000), 86.13)
singer.cycle_from_acf = lambda *args: ({acf!r}, 5, .68)
singer.fold_profile = lambda *args: np.array([-.11, .2, -.06, .31, -.1])
singer.segment_bars = lambda *args: [(0, 309.1)]
raise SystemExit(listen.main(sys.argv[1:]))
"""
            run = subprocess.run
            def child(command, **kwargs):
                return run([sys.executable, "-c", bootstrap, *command[2:]], **kwargs)
            args = listen.parser().parse_args([str(audio), "--no-singer", "--no-instruments", "--no-stems-timeline"])
            with patch.dict(os.environ, {"RAVEN_EARS_HOME": str(home), "RAVEN_NO_STOPS": "1"}), \
                 patch.object(listen.subprocess, "run", side_effect=child):
                result = listen.run_pass("singer", args)
            self.assertEqual(result["status"], "ran", result["reason"])
            song = {"cards": {"singer": result}}
            path = home / "song.json"
            listen.save(path, song)
            saved = json.loads(path.read_text())["cards"]["singer"]
            self.assertEqual(saved["json"]["meter"]["cycle_acf"], self.tied_cycle()["cycle_acf"])
            self.assertEqual(saved["json"]["meter"]["confidence"], 0)
            self.assertIn("meter unclear", saved["text"])
            self.assertNotIn("1.00", saved["text"])

    def test_ten_pulse_cycle_is_not_the_eight_pulse_fix(self):
        # Tolerate It, 12 Nov 2026 card: Aaron Dessner wrote it in 10/8 (METER-TEST-SET row 1).
        # 3+3+2+2 does not split into two bars of five, so the 8/8 rule must never rename it.
        d = {"verdict": "10/8 (3+3+2+2)", "cycle_pulses": 10, "grouping": "3+3+2+2",
             "grouping_confidence": 0.6, "beats_per_bar": 2, "tracker_bpm": 150.0,
             "pulse_bpm": 150.0, "confidence": 0.6}
        self.assertIsNone(M.accent_cycle_ambiguity(d))
        self.assertIsNone(M.accent_cycle_ambiguity(dict(d, beats_per_bar=4)))
        text = "METER: 10/8 (3+3+2+2) · cycle 10 pulses · grouping confidence 0.60\n"
        self.assertEqual(M.meter_text(text, d), text)

    def test_eight_pulse_template_does_not_establish_note_value(self):
        # Numerical shapes follow saved quarter-rate, eight-position profiles.
        # No audio, lyrics or identifying song data belongs in these fixtures.
        import numpy as np
        import singer
        from unittest.mock import patch
        shape = np.array([-0.7, 0.0, -0.4, -0.3, -0.1, -0.3, 0.0, -0.8])
        a = np.tile(shape, 75) + np.random.default_rng(0).normal(0, 0.32, 600)
        b = np.arange(len(a)) * 0.32
        d = b[::4]
        env = np.zeros(17000)
        pre = (b, d, "beat_this", env, 86.13, {"mix": env})
        with patch.object(singer, "accent_series", return_value=(a, ["drums", "bass", "mix"])):
            lines, data = singer.meter_block("unused", "unused", 192, _pre=pre)
        self.assertEqual(data["cycle_pulses"], 8)
        self.assertGreater(data["grouping_confidence"], 0.9)
        self.assertIsNone(singer._meter_name(8, (3, 2, 3), 187.5))
        self.assertIsNone(singer._meter_name(8, (2, 2, 2, 2), 150))
        self.assertIsNone(M.one(data))
        self.assertIn("4/4, accents repeat every 2 bars", lines[0])
        self.assertIn("8/8 only if each pulse is an eighth", lines[0])
        self.assertIn("accent cycle ≈ 2.56s", lines[1])
        self.assertNotIn("  bar ≈", "\n".join(lines))

    def test_saved_eight_cycles_keep_scale_weakness_and_capture(self):
        def eight(pulse, grouping, confidence):
            return dict(engine="beat_this", verdict="8/8 (" + grouping + ")", confidence=0.9,
                        tracker_bpm=pulse, pulse_bpm=pulse, beat_s=round(60/pulse, 3),
                        beats_per_bar=4, bar_consistency=0.6, bar_s=1.62,
                        cycle_source="accent autocorrelation", cycle_pulses=8, cycle_strength=0.5,
                        grouping=grouping, grouping_confidence=confidence)
        d = eight(187.5, "3+2+3", 1.0)
        captured = copy.deepcopy(d)
        text = "METER: 8/8 (3+2+3) · grouping confidence 1.00\n  bar ≈ 2.56s (cycle × 0.32s pulse)\n  measured detail\n"
        out = M.meter_text(text, d)
        self.assertIn("4/4, accents repeat every 2 bars at 187.5/min", out)
        self.assertIn("8/8 only if each pulse is an eighth", out)
        self.assertIn("  measured detail\n", out)
        self.assertEqual(M.meter_text(out, d), out)
        self.assertEqual(d, captured)
        self.assertIn("meter 4/4, accents repeat", M.tempo_text("TEMPO: 95.7 BPM · meter 8/8, see METER", d))
        # Without the tracker's 4-beat bars (or off a quarter-rate pulse) the scale stays open.
        for k, v in (("beats_per_bar", 3), ("tracker_bpm", 260.0)):
            other = dict(d, **{k: v}, pulse_bpm=d["pulse_bpm"] if k != "tracker_bpm" else v)
            self.assertIn("uncertain (4/4 over two bars / 8/8)", M.accent_cycle_ambiguity(other))
        import listen
        import slim
        song = {"cards": {"music": {"text": "MUSIC: example.mp3 · 3:12\n  TEMPO: 95.7 BPM · meter 8/8, see METER", "json": {}},
                          "singer": {"text": text, "json": {"meter": d}},
                          "harmony": {"text": "", "json": {}}},
                "manifest": {"passes": {n: {"status": "ran"} for n in ("music", "singer", "harmony")}}}
        before = copy.deepcopy(song)
        full = listen.full_text(song)
        feel = slim.render_first_listen({n: c["text"] for n, c in song["cards"].items()},
                                       {n: c["json"] for n, c in song["cards"].items()})
        for rendered in (full, feel):
            self.assertIn("4/4, accents repeat every 2 bars", rendered)
            self.assertIn("187.5/min", rendered)
            self.assertNotIn("meter 8/8,", rendered)
        self.assertEqual(song, before)
        weak = eight(150, "3+2+3", 0.11)
        sections = dict(sections=[dict(start=0, end=154.2, data=eight(150, "2+3+3", 1)),
                                  dict(start=154.2, end=220.1, data=eight(142.9, "2+2+2+2", 0.02))],
                        whole_song_data=weak)
        text = ("METER: 8/8 then 4/4\n  [0:00–2:34] 8/8 (2+3+3)\n"
                "    bar ≈ 3.20s (cycle × 0.40s pulse)\n  [2:34–3:40] 8 pulses per bar, grouping weak")
        out = M.meter_text(text, sections)
        self.assertIn("best fit 3+2+3 (grouping weak)", out.splitlines()[0])
        self.assertIn("[0:00–2:34] 4/4, accents repeat", out)
        self.assertIn("best fit 2+3+3 (uncertain)", out)
        self.assertIn("best fit 2+2+2+2 (grouping weak)", out)
        self.assertNotIn("    bar ≈", out)
        self.assertEqual(M.meter_text(out, sections), out)
        mixed = dict(sections=[sections["sections"][0], section(154.2, 900, "4/4", 0.8, 4, 1.62)])
        self.assertEqual(M.assess(mixed), ("4/4", True))
        out = M.meter_text(text, mixed)
        self.assertIn("METER: 4/4; meter changes unconfirmed", out)
        self.assertEqual(M.meter_text(out, mixed), out)

    def test_eighth_meter_remains_a_candidate_without_a_truth_override(self):
        # Identical pulse/accents could come from a notated additive 8/8 bar.
        # Retain that reading as a candidate; never force this evidence to 4/4.
        d = dict(verdict="8/8 (3+3+2)", confidence=1, cycle_pulses=8,
                 grouping="3+3+2", grouping_confidence=1, pulse_bpm=300)
        claim = M.assess(d)[0]
        self.assertIn("4/4 over two bars / 8/8", claim)
        self.assertIn("3+3+2", claim)
        self.assertIsNone(M.one(d))
        control = dict(verdict="10/8 (3+3+2+2)", confidence=0.6, cycle_pulses=10,
                       grouping="3+3+2+2", grouping_confidence=0.6, pulse_bpm=150)
        self.assertEqual(M.assess(control), ("10/8", False))
        self.assertEqual(M.meter_text("METER: " + control["verdict"], control), "METER: " + control["verdict"])

    def test_compound_needs_both_contrast_and_consistent_votes(self):
        for beats, name in ((2, "6/8"), (3, "9/8"), (4, "12/8")):
            d = dict(verdict=name + " (compound, each beat splits in 3)", confidence=0.8,
                     beats_per_bar=beats, bar_consistency=0.99, split_contrast=0.5, split_votes_ternary=0.95)
            self.assertEqual(M.assess(d), (name, False))
            for field, weak in (("split_contrast", 0.3), ("split_votes_ternary", 0.7),
                                ("split_votes_ternary", None), ("split_contrast", float("nan"))):
                weak_data = dict(d, **{field: weak})
                claim = M.assess(weak_data)[0]
                self.assertIn("subdivision ambiguous", claim)
                self.assertIn(f"{beats} beats per bar", claim)

    def test_saved_subdivision_is_hedged_without_relabelling_raw_evidence(self):
        d = dict(verdict="12/8 (compound, each beat splits in 3)", confidence=0.45,
                 beats_per_bar=4, split2=0.3, split3=0.55, split_contrast=0.3, split_votes_ternary=0.85)
        original = copy.deepcopy(d)
        text = "METER: " + d["verdict"] + " · confidence moderate (0.45)\n  between-beat onsets: measured detail"
        corrected = M.meter_text(text, d)
        self.assertIn("METER: 4 beats per bar; subdivision ambiguous", corrected)
        self.assertIn("unconfirmed subdivision candidate: " + text.splitlines()[0][7:], corrected)
        self.assertIn("  between-beat onsets: measured detail", corrected)
        self.assertEqual(M.meter_text(corrected, d), corrected)
        self.assertIn("meter 4 beats per bar; subdivision ambiguous", M.tempo_text("TEMPO: 120 BPM · meter 12/8, see METER", d))
        self.assertIn("subdivision ambiguous", M.assess({"sections": [dict(start=0, end=120, data=d)]})[0])
        self.assertEqual(d, original)

    def test_grouped_subdivisions_and_existing_ambiguous_verdicts_are_unchanged(self):
        grouped = dict(verdict="6/8 (compound, beats group in 3)", confidence=0.9,
                       beats_per_bar=6, group_scores={"2": 0.05, "3": 0.7})
        self.assertEqual(M.assess(grouped), ("6/8", False))
        ambiguous = dict(verdict="4 beats per bar; split 2 vs 3 is ambiguous (candidates 4/4 / 12/8)", confidence=0.03,
                         beats_per_bar=4, split_contrast=-0.02, split_votes_ternary=0.42)
        text = "METER: " + ambiguous["verdict"]
        self.assertEqual(M.meter_text(text, ambiguous), text)
        self.assertEqual(M.assess(ambiguous), (None, False))

    def test_weak_change_is_a_full_card_candidate_only(self):
        d = changing()
        d["windows"][4:6] = [[60, 3, 0.42], [75, 3, 0.50]]
        d["windows"] = d["windows"][:6] + [[90, 4, 1.0]]
        original = copy.deepcopy(d)
        self.assertEqual(M.assess(d), ("4/4", True))
        text = M.meter_text("METER: 4/4, then 3/4\n  measured detail", d)
        self.assertIn("METER: 4/4 (simple); meter changes unconfirmed", text)
        self.assertIn("≈1:00–1:30: 3 pulses (weak, confidence 0.42–0.50)", text)
        self.assertIn("unconfirmed candidates: 4/4, then 3/4", text)
        self.assertIn("  measured detail", text)
        self.assertEqual(M.meter_text(text, d), text)
        self.assertEqual(d, original)

    def test_strong_four_to_three_survives_local_bar_counts(self):
        d = changing()
        self.assertEqual(M.assess(d), ("4/4 then 3/4", False))
        self.assertEqual(M.meter_text("METER: 4/4 then 3/4", d), "METER: 4/4 then 3/4")

    def test_two_overlapping_windows_are_not_a_change(self):
        d = changing()
        d["windows"] = windows([4] * 4 + [3] * 2)
        self.assertEqual(M.assess(d), ("4/4", True))

    def test_weak_base_side_cannot_establish_a_change(self):
        d = changing()
        for w in d["windows"][:4]:
            w[2] = 0.5
        self.assertEqual(M.assess(d), ("4/4", True))

    def test_steady_four_beat_bars_veto_even_strong_triple_accents(self):
        d = changing()
        del d["window_bars"]  # older saved bundles have only the whole-song count
        self.assertEqual(M.assess(d), ("4/4", True))

    def test_half_pulse_and_weak_intro_are_demoted(self):
        d = {"sections": [
            section(0, 29, "8/8 (3+2+3)", 0.62, 4, 1.38,
                    cycle_source="accent autocorrelation", cycle_strength=0.34,
                    cycle_pulses=8, grouping_confidence=0.92),
            section(29, 50, "2/4", 0.70, 2, 1.38),
            section(50, 173, "4/4", 0.48, 4, 1.38)]}
        self.assertEqual(M.assess(d), ("4/4", True))
        # Confidence/duration alone must not promote a half-pulse reading.
        d["sections"] = [section(0, 60, "4/4", 0.95, 4, 2),
                         section(60, 120, "2/4", 0.95, 2, 2)]
        self.assertEqual(M.assess(d), (None, True))

    def test_strong_section_change_is_kept(self):
        d = {"sections": [section(0, 60, "4/4", 0.95, 4, 2),
                          section(60, 120, "3/4", 0.95, 3, 1.5)]}
        self.assertEqual(M.assess(d), ("4/4 then 3/4", False))

    def test_section_label_must_match_the_counted_bars(self):
        d = {"sections": [section(0, 60, "4/4", 0.95, 4, 2),
                          section(60, 120, "3/4", 0.95, 4, 1.5)]}
        self.assertEqual(M.assess(d), (None, True))

    def test_five_four_survives_bad_downbeats_and_weak_double_cycle(self):
        def five(start, end):
            s = section(start, end, "5/4 (2+3)", 1.0, 2, 0.56,
                        cycle_source="accent autocorrelation", cycle_pulses=5,
                        cycle_strength=0.57, grouping="2+3", grouping_confidence=1.0)
            s["data"]["bar_consistency"] = 0.32
            return s
        d = {"sections": [five(0, 257),
                          section(257, 387, "10 pulses per bar, grouping weak", 0.3, 1, 0.3, cycle_pulses=10),
                          five(387, 445)]}
        self.assertEqual(M.assess(d), ("5/4 (2+3)", False))
        original = "METER: 0:00–4:17 5/4 (2+3) · weak 10 · 5/4 again"
        self.assertEqual(M.meter_text(original, d), original)
        self.assertIn("meter 5/4 (2+3)", M.tempo_text("TEMPO: 108 BPM", d))

    def test_saved_views_agree_and_leave_bundle_untouched(self):
        import listen
        d = changing(0.45)
        song = {"cards": {"music": {"text": "MUSIC: invented.mp3 · 2:00\n  TEMPO: 120 BPM · meter 4/4 then 3/4, see METER", "json": {}},
                          "singer": {"text": "METER: 4/4 then 3/4", "json": {"meter": d}},
                          "harmony": {"text": "", "json": {}}},
                "manifest": {"passes": {n: {"status": "ran"} for n in ("music", "singer", "harmony")}}}
        before = copy.deepcopy(song)
        full = listen.full_text(song)
        self.assertIn("meter 4/4, see METER", full)
        self.assertIn("meter changes unconfirmed", full)
        import slim
        feel = slim.render_first_listen({n: c["text"] for n, c in song["cards"].items()},
                                       {n: c["json"] for n, c in song["cards"].items()})
        tempo = next(l for l in feel.splitlines() if l.startswith("TEMPO:"))
        self.assertIn("meter 4/4", tempo)
        self.assertNotIn("then 3/4", tempo)
        self.assertEqual(song, before)


def run_tests(check):
    for name in unittest.defaultTestLoader.getTestCaseNames(MeterTrustTests):
        result = unittest.TestResult()
        MeterTrustTests(name).run(result)
        check("meter trust: " + name, result.wasSuccessful(),
              "\n".join(trace for _, trace in result.failures + result.errors))


if __name__ == "__main__":
    unittest.main()
