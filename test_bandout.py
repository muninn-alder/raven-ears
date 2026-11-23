"""Band presence: synthetic states only; no audio, models or services."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import harmony as H
import singer


class BandOutTests(unittest.TestCase):
    def states(self, a=20, b=28):
        states = {s: np.ones(120, bool) for s in singer.SRC}
        for s in ("bass", "other"):
            states[s][int(a * 2):int(b * 2)] = False
        return states

    def candidates(self, states=None, rms=None, present=None, **kwargs):
        return H.band_out_candidates(self.states() if states is None else states,
            dict.fromkeys(singer.SRC, True) if present is None else present, .5, 60,
            np.full(int(60 * H.FPS) + 1, -10.) if rms is None else rms,
            **dict(downbeats=np.arange(0, 61, 4), **kwargs))

    def test_two_bars_with_voice_or_drums(self):
        for silent in (None, "drums", "vocals"):
            states = self.states()
            if silent:
                states[silent][:] = False
            accepted = [c for c in self.candidates(states) if c["accepted"]]
            self.assertEqual([(c["t0"], c["t1"], c["bars"]) for c in accepted], [(20, 28, 2)])
            line = H.band_out_line(accepted[0])
            self.assertLessEqual(len("   0:20  " + line), 118)
            if silent == "vocals":
                self.assertNotIn("voice alone", line)

    def test_rejections_are_retained_with_reasons(self):
        for a, b, reason in ((20, 24, "under two bars"), (0, 8, "overlaps first/last five seconds"),
                             (52, 60, "overlaps first/last five seconds")):
            with self.subTest(a=a, b=b):
                rows = self.candidates(self.states(a, b))
                self.assertEqual(len(rows), 1)
                self.assertFalse(rows[0]["accepted"])
                self.assertIn(reason, rows[0]["rejection_reasons"])
        rms = np.full(int(60 * H.FPS) + 1, -10.)
        rms[int(20 * H.FPS):int(28 * H.FPS)] = -23
        row = self.candidates(rms=rms)[0]
        self.assertFalse(row["accepted"])
        self.assertIn("quiet relative to song median", row["rejection_reasons"])

    def test_absent_stem_does_not_count_as_out(self):
        for stem in ("bass", "other"):
            states = self.states()
            del states[stem]
            rows = self.candidates(states, present={s: s != stem for s in singer.SRC})
            self.assertFalse(any(c["accepted"] for c in rows))
            self.assertIn("harmonic stem absent or never on", rows[0]["rejection_reasons"])
            states[stem] = np.zeros(120, bool)
            self.assertFalse(any(c["accepted"] for c in self.candidates(states)))

    def test_bass_or_other_on_ends_the_stretch(self):
        for stem in ("bass", "other"):
            states = self.states(20, 36)
            states[stem][56:] = True
            rows = [c for c in self.candidates(states) if c["accepted"]]
            self.assertEqual([(c["t0"], c["t1"]) for c in rows], [(20, 28)])
            states[stem][:] = True
            self.assertEqual(self.candidates(states), [])

    def test_silence_does_not_erase_the_active_portion(self):
        states = self.states(20, 40)
        states["vocals"][56:80] = False
        states["drums"][56:80] = False
        rows = self.candidates(states)
        self.assertEqual([(c["t0"], c["t1"]) for c in rows if c["accepted"]], [(20, 28)])
        self.assertIn("neither voice nor drums on", rows[1]["rejection_reasons"])

    def test_original_singer_output_is_byte_identical(self):
        # Golden output captured before the shared-state refactor, not recalculated expectations.
        curves = {s: np.full(180, -10.) for s in singer.SRC}
        for s in ("bass", "other"):
            curves[s][40:60] = -90
        with patch.object(singer, "stem_curves", return_value=(curves, .5)):
            lines, events, ref, present = singer.stems_timeline("unused", 90)
        golden = ("  drums  " + "█" * 60 + "\n"
                  "  bass   " + "█" * 13 + "▁" * 7 + "█" * 40 + "\n"
                  "  other  " + "█" * 13 + "▁" * 7 + "█" * 40 + "\n"
                  "  vocals " + "█" * 60 + "\n"
                  "         0:00" + " " * 51 + "1:30")
        self.assertEqual("\n".join(lines).encode(), golden.encode())
        self.assertEqual(events, [dict(t=0., text="opens with drums + bass + other + vocals"),
                                 dict(t=20., text="bass + other drop out"),
                                 dict(t=30., text="bass + other enter")])
        self.assertEqual(ref, dict.fromkeys(singer.SRC, -10.))
        self.assertEqual(present, dict.fromkeys(singer.SRC, True))

    def card(self, events=None):
        r = {k: [] for k in ("secs", "moves", "pedals", "br", "dips", "drops", "swells", "con", "tag", "csurp", "cads", "lines")}
        r.update(path="neutral.wav", dur=60, bpm=60, glob=(0, "major"), kinfo={},
                 key_verdict=dict(key="C major", line="C major"), spans=[(0, (0, "major"))],
                 notes=[], evs=events or [], tc=dict(step=2, T=[]), words=None)
        return r

    def test_json_keeps_labels_and_excludes_only_band_out_time_from_share(self):
        c = self.candidates()[0]
        original = [dict(t0=0, t1=40, root=0, q="maj"), dict(t0=40, t1=60, root=7, q="maj")]
        marked = H.mark_band_out(original, [c])
        self.assertEqual(sum(e["t1"] - e["t0"] for e in marked), 60)
        self.assertEqual([(e["t0"], e["t1"]) for e in marked if e["band_out"]], [(20, 28)])
        r = self.card(marked)
        r["band_presence"] = dict(status="checked", candidates=[c])
        with patch.dict("os.environ", RAVEN_NO_INSTRUMENTS="1"):
            text = "\n".join(H.block_chords(r))
            self.assertIn("most time on: I 62%, V 38%", text)
            timeline = H.build_timeline(r, cap=1)
            self.assertEqual(sum(txt == H.band_out_line(c) for _, _, txt in timeline), 1)
            with patch("ache.analyze", return_value={}):
                data = H.to_json(r)
        self.assertEqual(len(data["chords"]), 4)
        self.assertEqual(data["chords"][1]["chord"], "C")
        self.assertTrue(data["chords"][1]["band_out"])
        self.assertEqual(data["band_presence"]["candidates"], [c])

    def test_no_cached_stems_skips_without_demucs_and_prints_notes(self):
        r = self.card()
        with patch.object(singer, "file_hash", return_value="uncached"), \
             patch.object(H.os.path, "isfile", return_value=False), \
             patch.object(singer, "get_stems", side_effect=AssertionError("no separation")), \
             patch.object(singer, "stem_states", side_effect=AssertionError("no stems")):
            r["band_presence"] = H.band_presence(r["path"], 60, np.zeros(10), None, 60, r["notes"])
        self.assertEqual(r["band_presence"]["status"], "skipped")
        self.assertEqual(H.band_out_stretches(r), [])
        with ExitStack() as stack:
            for name in ("block_sound", "block_tension", "block_resolves", "block_lyric_timeline",
                         "block_contrast", "block_surprise", "block_chills"):
                stack.enter_context(patch.object(H, name, return_value=[]))
            stack.enter_context(patch("ache.block", return_value=[]))
            stack.enter_context(patch.dict("os.environ", RAVEN_NO_INSTRUMENTS="1"))
            text = H.report(r)
        self.assertIn("NOTES:", text)
        self.assertIn("band-presence check skipped: no cached stems available", text)
        self.assertNotIn("no harmonic instruments", text)

    def test_release_cannot_claim_full_band_across_a_band_out_stretch(self):
        evs = H.mark_band_out([dict(t0=0, t1=60, root=0, q="maj")], [self.candidates()[0]])
        b = dict(t0=18, t1=32, kind="release", comp={}, **{"from": .5, "to": .3})
        text = H.describe_build(b, evs, [(0, (0, "major"))], {},
                                {"rms_db": np.full(int(60 * H.FPS), -10.)})
        self.assertIn("harmonic instruments out for part of this span", text)
        self.assertNotIn("full band still there", text)


def run_checks(check, tmp=None):
    for test in unittest.defaultTestLoader.loadTestsFromTestCase(BandOutTests):
        result = unittest.TestResult()
        test.run(result)
        check("band presence: " + test._testMethodName.removeprefix("test_").replace("_", " "),
              result.wasSuccessful(), "\n".join(message for _, message in result.errors + result.failures))


if __name__ == "__main__":
    unittest.main()
