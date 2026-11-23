"""Short opening-region policy; invented chords only. python3 test_key_opening.py."""
import copy
import json
from pathlib import Path
import unittest

import key_opening as K


def region(cadence=False):
    opening = [(0, 4, "E"), (4, 5, "B7"), (5, 7, "E")] if cadence else [(0, 7, "E")]
    return {"home_key": "C major", "key_verdict": {"key": "C major", "state": "clear"},
            "chords": [dict(t0=a, t1=b, chord=c) for a, b, c in opening + [(7, 11, "Cmaj7"), (11, 15, "Am7"), (15, 21, "F")]]}


TEXT = "HARMONY: invented.mp3 · 2:00\nCHORDS\n  home: C major · tonic clear; ~120 BPM beat-tracker pulse\n  ends on I (C); opens on III\n"


class ShortIntroTests(unittest.TestCase):
    def test_sustained_chord_is_named_as_a_chord_not_a_key(self):
        data = region()
        before = copy.deepcopy(data)
        intro = "intro on E major chord (0:00–0:07), then C major"
        self.assertEqual(K.short_intro(data), intro)
        full = K.intro_text(TEXT, data)
        self.assertIn("ends on I (C); " + intro, full)
        self.assertEqual(K.intro_text(full, data), full)
        self.assertEqual(data, before)

    def test_local_dominant_tonic_supports_a_key_region(self):
        self.assertEqual(K.short_intro(region(True)), "intro in E major (0:00–0:07), then C major")
        data = region(True)
        for event, name in zip(data["chords"], ("G#m", "D#7", "G#m")):
            event["chord"] = name
        self.assertEqual(K.short_intro(data), "intro in G# minor (0:00–0:07), then C major")

    def test_diatonic_dominant_unclear_and_missing_evidence_do_not_name_a_region(self):
        for chord in ("C", "Am", "G", "G7", "F", "Dm", "unknown", None):
            data = region()
            data["chords"][0]["chord"] = chord
            self.assertIsNone(K.short_intro(data), chord)
        data = region()
        data["key_verdict"]["state"] = "unclear"
        self.assertIsNone(K.short_intro(data))
        self.assertIsNone(K.short_intro({}))
        self.assertIsNone(K.short_intro(None))

    def test_short_opening_and_settled_home_are_required(self):
        for end in (1, 20, 30):
            data = region()
            for event in data["chords"]:
                event["t0"] *= end / 7
                event["t1"] *= end / 7
            self.assertIsNone(K.short_intro(data), end)
        data = region()
        data["chords"][1]["chord"] = "G"
        self.assertIsNone(K.short_intro(data))
        data = region()
        data["chords"][2]["chord"] = "F#"
        self.assertIsNone(K.short_intro(data))
        data = region()
        data["chords"][0]["t0"] = 2
        self.assertIsNone(K.short_intro(data))
        data = region(True)
        data["chords"][1]["chord"] = None
        self.assertIsNone(K.short_intro(data))
        data = region(True)
        data["chords"][1]["t0"] += 0.5
        self.assertIsNone(K.short_intro(data))

    def test_malformed_optional_evidence_is_absent(self):
        for field, value in (("chords", [None]), ("chords", "not events"), ("key_verdict", []),
                             ("key_verdict", {"key": 5}), ("chords", [{"chord": "E", "t0": None, "t1": 5}]),
                             ("chords", [{"chord": "E", "t0": 0, "t1": float("inf")}])):
            self.assertIsNone(K.short_intro(dict(region(), **{field: value})))

    def test_failed_evidence_and_passes_cannot_lend_an_intro(self):
        import listen
        import slim
        data = region()
        data["stage_status"] = [{"stage": "chords", "status": "failed"}]
        self.assertIsNone(K.short_intro(data))
        data.pop("stage_status")
        self.assertEqual(K.intro_text(TEXT + "NOTES: chords: failed: synthetic failure\n", data),
                         TEXT + "NOTES: chords: failed: synthetic failure\n")
        song = {"cards": {"music": {"text": "", "json": {}}, "singer": {"text": "", "json": {}},
                          "harmony": {"text": TEXT, "json": data}},
                "manifest": {"passes": {n: {"status": "ran", "reason": ""} for n in ("music", "singer", "harmony")}}}
        before = copy.deepcopy(song)
        self.assertIn("intro on E major chord", listen.full_text(song))
        def feel():
            return slim.render_first_listen({n: c["text"] for n, c in song["cards"].items()},
                                            {n: c["json"] for n, c in song["cards"].items()}, song["manifest"]["passes"])
        self.assertIn("intro on E major chord", feel())
        self.assertEqual(song, before)
        song["manifest"]["passes"]["harmony"]["status"] = "failed"
        self.assertNotIn("intro on", feel())
        self.assertNotIn("intro on", listen.full_text(song))

    def test_existing_uhuru_fixture_is_unchanged(self):
        import listen
        song = json.loads((Path(__file__).parent / "fixtures/uhuru-title-song.json").read_text())
        card = song["cards"]["harmony"]
        self.assertIsNone(K.short_intro(card["json"], card["text"]))
        self.assertEqual(listen.pass_text(song, "harmony"), card["text"])


def run_tests(check):
    for name in unittest.defaultTestLoader.getTestCaseNames(ShortIntroTests):
        result = unittest.TestResult()
        ShortIntroTests(name).run(result)
        check("short intro: " + name, result.wasSuccessful(),
              "\n".join(trace for _, trace in result.failures + result.errors))


if __name__ == "__main__":
    unittest.main()
