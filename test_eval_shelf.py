"""Neutral text/JSON fixtures only. Also called by the main --fast runner."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import eval_shelf as E


class ShelfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ears_shelf_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.music = self.root / "music"
        self.cache = self.root / "cache"
        self.music.mkdir()

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def truth(self, dimension, value):
        return {"value": value, "source": "neutral.md:2", "status": "ear note", "evidence": "synthetic truth"}

    def score(self, dimension, truth, prediction, **kwargs):
        return E.score_dimension(dimension, self.truth(dimension, truth), {"value": prediction, "raw": "fixture", **kwargs})

    def fixture(self):
        self.write(self.music / "EAR-KEY-lantern.md", """# Lantern — listener's ear
- KEY: D♭ major (five flats). A relative minor is not right.
- METER: 4/4, with a 12/8 feel
- TEMPO: 80 BPM; 160 is the doubled pulse
INSTRUMENTS: piano, electric guitar
an event we cannot parse
""")
        self.write(self.music / "lyrics/lantern.txt", "# official words\n[Verse]\nWe walk home\n")
        self.write(self.music / "credits/lantern.txt", "# credits\nA Player: cello\n")
        self.write(self.cache / "raven/cards/Band - Lantern.txt", """MUSIC: Band - Lantern.mp3
  KEY: C# major · tonic fairly clear
  TEMPO: 160 BPM · felt beat likely ~80
  LYRICS: 1 lines (whisper large-v3, rough times)
    [0:00] We walked home (unsure: Whisper doubtful)
INSTRUMENTS (tagger):
  overall:
    piano            probably (score 0.7, 90% of the song)
    electric guitar  probably (score 0.6, 60% of the song)
    drums (type unclear) present, not nameable (energy only, 10%)
CREDITS:
  credited but unheard: cello
METER: 4/4 · confidence moderate
CHORDS
  home: C# major; ~160 BPM beat-tracker pulse
""")

    def test_six_dimensions_and_read_only(self):
        self.fixture()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        snapshot = E.evaluate(self.music, self.cache)
        self.assertEqual(len(snapshot["songs"]), 1)
        song = snapshot["songs"][0]
        for dim in ("key", "mode", "meter", "tempo", "instruments"):
            self.assertEqual(song["dimensions"][dim]["score"], 1.0, dim)
        self.assertAlmostEqual(song["dimensions"]["lyrics"]["wer"], 1 / 3)
        self.assertEqual(song["credits"], ["cello"])
        self.assertTrue(song["skipped"])
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.cache / "raven/eval").exists())

    def test_narrative_timeline_piano_is_not_a_negative(self):
        path = self.write(self.music / "EAR-KEY-example.md", """# Example (Band) — listener
Credits: synth, cello
Listener's timeline:
- 0:01 piano comes in
- 0:03 guitar strings come on
- 0:04 synth likely

Ears missed organ
""")
        song = E.parse_ear_key(path, [])
        self.assertEqual(song["truth"]["instruments"]["value"], ["guitar", "piano"])
        self.assertNotIn("key", song["truth"])

    def test_unknown_and_conflicting_truth_remain_visible(self):
        self.write(self.music / "EAR-KEY-unknown.md", "# Unknown\nKEY: likely C major or A minor\nMETER: unclear\n")
        self.write(self.music / "EAR-KEY-a.md", "# Same\nKEY: C major\n")
        self.write(self.music / "EAR-KEY-b.md", "# Same\nKEY: D major\n")
        snapshot = E.evaluate(self.music, self.cache)
        self.assertEqual(len(snapshot["songs"]), 2)
        by_id = {s["id"]: s for s in snapshot["songs"]}
        self.assertEqual(by_id["same"]["dimensions"]["key"]["status"], "unscorable: conflicting truth")
        self.assertEqual(by_id["unknown"]["dimensions"]["key"]["status"], "unscorable: no truth")
        self.assertIn("unscorable: no truth", E.scoreboard(snapshot))

    def test_ambiguous_truth_does_not_become_a_definite_call(self):
        path = self.write(self.music / "EAR-KEY-ambiguous.md", """# Ambiguous
METER: 4/4 or 12/8
TEMPO: 80 or 160 BPM
INSTRUMENTS: maybe piano, not cello
""")
        song = E.parse_ear_key(path, [])
        self.assertEqual(song["truth"], {})
        self.assertEqual(len(song["skipped"]), 3)

    def test_meter_table_status_and_notation(self):
        path = self.write(self.music / "METER-TEST-SET.md", """| # | Song | Meter | Status |
| 1 | Odd Song (Band, 2000) | 7/4 verses | VERIFIED |
| 2 | Five Song (Band, 2000) | quintuple meter | SECONDARY |
| 3 | Contested (Band, 2000) | 3/4 or 4/4 | UNVERIFIED |
| 4 | Decimal (Band, 2000) | 6.5/8 | SECONDARY |
""")
        songs = E.parse_meter_table(path, [])
        self.assertEqual(len(songs), 4)
        truth = songs[0]["truth"]["meter"]
        self.assertEqual(truth["status"], "VERIFIED")
        self.assertEqual(E.score_dimension("meter", truth, {"value": ["7/4", "4/4"]})["score"], 1)
        self.assertEqual(songs[1]["truth"]["meter"]["value"]["pulses"], 5)
        self.assertFalse(songs[2]["truth"])
        self.assertFalse(songs[3]["truth"])
        self.assertEqual(E.meters("6.5/8"), [])

    def test_key_relations_enharmonics_and_modes(self):
        self.assertEqual(E.key_value("D♭ major"), E.key_value("C# major"))
        ref = E.key_value("C major")
        for name, score, relation in (("C major", 1, "same"), ("G major", .5, "fifth"),
                                      ("A minor", .3, "relative"), ("C minor", .2, "parallel"),
                                      ("D minor", 0, "other")):
            row = self.score("key", ref, E.key_value(name))
            self.assertEqual((row["score"], row["relation"]), (score, relation))
        self.assertEqual(self.score("mode", "major", E.key_value("A minor"))["score"], 0)

    def test_unclear_key_never_uses_working_key(self):
        parsed = E.parse_analysis({"key_verdict": {"key": None, "working": "C major", "line": "tonic unclear"}})
        self.assertIsNone(parsed["key"]["value"])
        card = E.parse_card("  KEY: C major or A minor, tonic unclear\n")
        self.assertIsNone(card["key"]["value"])
        contradiction = E.parse_card("  KEY: C major\n  home: G major; tracker\n")
        self.assertIsNone(contradiction["key"]["value"])

    def test_json_uses_fused_music_claim(self):
        parsed = E.parse_analysis({"key": {"key": "A minor"},
                                   "key_claim": {"key": "C major", "line": "C major · tonic clear"}})
        self.assertEqual(parsed["key"]["value"], [0, "major"])
        parsed = E.parse_analysis({"key": {"key": "A minor"}, "key_claim": {"key": None}})
        self.assertIsNone(parsed["key"]["value"])

    def test_tempo_half_double_and_tolerance(self):
        for bpm, felt, expected in ((81, None, 1), (160, None, .5), (40, None, .5),
                                    (160, 80, 1), (100, None, 0)):
            self.assertEqual(self.score("tempo", 80, bpm, felt=felt)["score"], expected)

    def test_meter_changes_are_inventory_only(self):
        truth = {"signatures": ["7/4", "4/4"]}
        self.assertEqual(self.score("meter", truth, ["7/4"])["score"], .5)
        row = self.score("meter", truth, ["4/4", "7/4"])
        self.assertEqual(row["score"], 1)
        self.assertIn("timing and grouping not scored", row["metric"])

    def test_instruments_do_not_invent_subtypes_or_credits(self):
        data = {"families": {"guitar (type unclear)": {}, "keys (organ or piano?)": {},
                              "drums (type unclear)": {"energy_only": True}, "cello": {"credited_unheard": True},
                              "banjo/mandolin/ukulele": {}, "electric guitar (probably)": {}}}
        parsed = E.parse_instruments(data)["instruments"]["value"]
        self.assertEqual(parsed, ["electric guitar", "guitar"])
        self.assertTrue(E.instrument_matches("guitar", ["electric guitar"]))
        self.assertFalse(E.instrument_matches("electric guitar", ["guitar"]))
        self.assertFalse(E.instrument_matches("piano", ["keyboards"]))
        self.assertIsNone(E.parse_instruments({"error": "model failed"})["instruments"]["value"])

    def test_lyrics_word_edits_and_normalization(self):
        self.assertEqual(E.word_error("We're HOME!", "we’re home")["errors"], 0)
        self.assertEqual(E.word_error("a b c", "a x c d")["errors"], 2)
        self.assertEqual(E.word_error("a b c", "a c")["errors"], 1)
        self.assertEqual(E.word_error("a", "a b c")["wer"], 2)
        self.assertIsNone(E.word_error("", "anything"))

    def test_failed_transcription_is_not_silence(self):
        failed = E.parse_words({"words": [], "notes": ["Whisper could not run"]})
        empty = E.parse_words({"words": [], "notes": []})
        self.assertIsNone(failed["value"])
        self.assertEqual(empty["value"], "")
        truth = self.truth("lyrics", {"text": "we walk"})
        self.assertIsNone(E.score_dimension("lyrics", truth, failed)["score"])
        self.assertEqual(E.score_dimension("lyrics", truth, empty)["wer"], 1)

    def test_malformed_backend_values_abstain(self):
        self.assertIsNone(E.parse_words({"notes": None, "words": []})["value"])
        self.assertIsNone(E.parse_instruments({"families": {"piano": None}})["instruments"]["value"])
        for claim in (42, "", "5 beats?", "4/4 or 12/8"):
            self.assertIsNone(E.parse_analysis({"meter_claim": claim})["meter"]["value"])
        self.assertIsNone(E.parse_analysis({"key_verdict": {"key": None, "line": None}})["key"]["value"])

    def test_placed_or_truncated_lyrics_not_transcription(self):
        for suffix in ("official placed lyrics", "head and tail; 2 lines omitted", "aligned words"):
            self.assertNotIn("lyrics", E.parse_card(f"  LYRICS: {suffix}\n    [0:01] We walk\n"))
        self.assertIsNone(E.parse_words({"words": [4]})["value"])

    def test_matching_is_unique_and_recording_specific(self):
        paths = [Path("Band - Lantern.mp3"), Path("Other - Lantern.mp3")]
        found, reason = E.find_match(["Lantern"], paths)
        self.assertIsNone(found)
        self.assertIn("ambiguous", reason)
        self.assertIsNone(E.find_match(["Lantern"], [Path("Band - Lantern (Live).mp3")])[0])
        self.assertIsNone(E.find_match(["Lamp"], [Path("Lamplight.mp3")])[0])
        self.assertEqual(E.find_match(["Lantern"], paths[:1])[0], paths[0])

    def test_card_text_precedes_json_and_dotted_titles_survive(self):
        self.write(self.music / "EAR-KEY-lamp.md", "# Lamp 1.2\nKEY: C major\n")
        self.write(self.cache / "raven/cards/Lamp 1.2.txt", "  KEY: C major\n")
        self.write(self.cache / "raven/cards/Lamp 1.2.json", json.dumps({"key_verdict": {"key": "G major"}}))
        dims = E.evaluate(self.music, self.cache)["songs"][0]["dimensions"]
        self.assertEqual(dims["key"]["score"], 1)
        self.assertTrue(dims["key"]["analysis_source"].endswith("Lamp 1.2.txt"))

    def test_hashed_caches_and_latest_variant(self):
        audio = self.write(self.music / "Band - Lantern.mp3", "fake bytes, never decoded")
        self.write(self.music / "lyrics/lantern.txt", "We walk home")
        sha = E.file_sha(audio)
        self.write(self.cache / f"raven-stems/{sha[:16]}-a0/meter.json", json.dumps({"claim": "5/4"}))
        self.write(self.cache / f"raven-stems/{sha[:16]}-a0/instruments.json", json.dumps({"families": {"piano": {}}}))
        first = self.write(self.cache / f"raven-lyrics/{sha}.json", json.dumps({"words": [["wrong", 0, 1]]}))
        latest = self.write(self.cache / f"raven-lyrics/{sha}-names.json", json.dumps({"words": [["We walk home", 0, 1]]}))
        # DrvFs rounds sub-100 ns times to a tie; give the variants distinct seconds.
        os.utime(first, (1_700_000_000, 1_700_000_000))
        os.utime(latest, (1_700_000_010, 1_700_000_010))
        snapshot = E.evaluate(self.music, self.cache)
        row = snapshot["songs"][0]["dimensions"]["lyrics"]
        self.assertEqual(row["score"], 1)
        self.assertEqual(row["analysis_source"], str(latest))
        self.assertTrue(any("2 lyric cache variants" in s for s in snapshot["issues"]))
        self.assertNotIn("We walk home", json.dumps(snapshot))

    def test_bad_cache_does_not_hide_other_songs(self):
        self.fixture()
        self.write(self.music / "EAR-KEY-broken.md", "# Broken\nKEY: C major\n")
        self.write(self.cache / "raven/cards/Broken.json", "not json")
        snapshot = E.evaluate(self.music, self.cache)
        self.assertEqual(len(snapshot["songs"]), 2)
        self.assertTrue(any("malformed" in s for s in snapshot["issues"]))

    def test_snapshot_compare_pairs_truth_and_reports_both_directions(self):
        self.fixture()
        old = E.evaluate(self.music, self.cache)
        new = copy.deepcopy(old)
        dims = new["songs"][0]["dimensions"]
        dims["lyrics"]["score"] = 1
        dims["key"]["score"] = .3
        text = E.compare_snapshots(new, old)
        self.assertIn("better", text)
        self.assertIn("worse", text)
        dims["key"]["truth_id"] = "changed"
        self.assertIn("truth changed; scores not compared", E.compare_snapshots(new, old))
        new["policy"] = "different"
        self.assertIn("policy differs", E.compare_snapshots(new, old))

    def test_cli_save_compare_and_footer(self):
        self.fixture()
        argv = ["--music-dir", str(self.music), "--cache-dir", str(self.cache)]
        with patch("socket.socket", side_effect=AssertionError("network forbidden")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(E.main(argv + ["--save", "--compare"]), 0)
            self.assertIn("no previous snapshot", output.getvalue())
            self.assertTrue(output.getvalue().endswith("and now, you\n"))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(E.main(argv + ["--compare", "--save"]), 0)
            self.assertIn("paired rows", output.getvalue())
        saved = list((self.cache / "raven/eval").glob("*.json"))
        self.assertEqual(len(saved), 2)
        self.assertTrue(all(json.loads(p.read_text())["closing"] == "and now, you" for p in saved))
        self.write(self.cache / "raven/eval/eval-zzzz.json", "broken")
        self.write(self.cache / "raven/eval/eval-zzz.json", json.dumps({"schema": E.SCHEMA, "songs": [None]}))
        issues = []
        self.assertIsNotNone(E.latest_snapshot(self.cache / "raven/eval", issues))
        self.assertTrue(issues)

    def test_cli_missing_shelf_and_help_end_with_listener(self):
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(E.main(["--music-dir", str(self.root / "missing")]), 2)
        self.assertTrue(output.getvalue().endswith("and now, you\n"))
        with contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit):
            E.main(["--help"])
        self.assertTrue(output.getvalue().endswith("and now, you\n"))


def run_checks(check):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ShelfTests)
    result = unittest.TestResult()
    tests = list(suite)
    suite.run(result)
    failures = {str(test): detail for test, detail in result.failures + result.errors}
    for test in tests:
        check("eval shelf: " + test._testMethodName.removeprefix("test_"),
              str(test) not in failures, failures.get(str(test), ""))


if __name__ == "__main__":
    unittest.main()
