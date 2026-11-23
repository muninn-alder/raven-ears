"""Invented text, fake inference, real speech CLI and listen worker/save paths."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import credits
import jev_more as M
import jev_reads as J
import listen as L
import lyric_align
import music
import raven_ears_plus as R
from test_cpu_profile import card_fixture


def fake_http(req, timeout):
    payload = json.loads(req.data)
    state = payload["state"]
    values = {}
    for name, question in payload["questions"].items():
        if question["type"] == "score":
            values[name] = dict(score={"I lost hope. And feel down": .76, "I feel glad": 3.6}.get(state["line"], 2))
            continue
        p = .04
        if name in ("fine", "joking"):
            p = .94
        elif name.startswith("self_"):
            p = .93
        elif name == "same":
            p = .92 if "lanturn" in state["heard"] else .06
        elif name.startswith("split_"):
            candidate = state["candidates"][int(name.split("_")[1])]
            p = .95 if candidate == dict(artist="Invented Singer", title="Lantern") else .02
        values[name] = dict(noul=p)
    return io.BytesIO(json.dumps(dict(answers=values)).encode())


def lyric_rows():
    # Real saved music rows have t/end/text/approx/differs/matched/tokens/misheard.
    return [dict(t=4.0, end=7.5, text="Carry the lantern", approx=False, differs=True, matched=2, tokens=3,
                 misheard=dict(heard="Carry the lanturn", wrote="Carry the lantern",
                               heard_core="lanturn", wrote_core="lantern")),
            dict(t=12.0, end=15.5, text="Bring our lost friends home", approx=False, differs=True, matched=3, tokens=5,
                 misheard=dict(heard="Leave the empty room", wrote="Bring our lost friends home",
                               heard_core="Leave the empty room", wrote_core="Bring our lost friends home"))]


class MoreJevTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, TYPESAFE_API_KEY="invented-key", RAVEN_EARS_OFFLINE="0",
                              RAVEN_EARS_HOME=str(self.root / "ears"))
        self.env.start()
        self.addCleanup(self.env.stop)
        http = patch.object(J.request, "urlopen", side_effect=fake_http)
        self.http = http.start()
        self.addCleanup(http.stop)
        credits._JEV_NAMES.clear()

    def cli(self, path, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(R.main([str(path), "--quiet", *args]), 0)
        return out.getvalue()

    def test_real_speech_cli_saves_all_readings_and_never_learns_a_word_identity(self):
        words = [["I'm", 1.68, 2.1], ["fine.", 2.1, 2.8], ["Hey", 12.26, 12.7],
                 ["it's", 12.7, 13.1], ["Aster.", 13.1, 13.6]]
        wordfile = self.root / "words.json"
        wordfile.write_text(json.dumps(words))
        with card_fixture() as (path, *_), patch("voice_book.identify", return_value=dict(status="unknown", ranked=[])):
            data = json.loads(self.cli(path, "--words", str(wordfile), "--json"))
            self.assertEqual(len(data["jev_speech"]), 2)
            self.assertEqual(data["jev_speech"][1]["candidates"], ["Aster"])
            self.assertIn("answer", data["jev_speech"][0]["jev"])
            output = self.cli(path, "--words", str(wordfile))
            self.assertIn("WHO (from the words): speaker says they're Aster (Jev 0.93)", output)
            self.assertIn("JOKING? 0:01", output)
            self.assertLess(output.index("JOKING? 0:01"), output.index("JOKING? 0:12"))
            self.assertNotIn("WORDS vs VOICE:", output)
            self.assertNotIn("vs_usual", data)
        self.assertFalse(R.baseline_path("Aster").exists())
        for call in self.http.call_args_list:
            state = json.loads(call.args[0].data)["state"]
            self.assertEqual(set(state), {"line", "before", "after"})
        baseline = R.baseline_path("Birch")
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(''.join(json.dumps(dict(pitch_hz=210., wpm=130.)) + '\n' for _ in range(12)))
        before = baseline.read_bytes()
        with card_fixture() as (path, *_), patch("voice_book.identify", side_effect=AssertionError("explicit wins")):
            output = self.cli(path, "--words", str(wordfile), "--speaker", "Birch", "--no-baseline-update")
        self.assertNotIn("WORDS vs VOICE:", output)
        self.assertIn("--speaker says Birch", output)
        self.assertEqual(baseline.read_bytes(), before)
        self.assertFalse(R.baseline_path("Aster").exists())

    def test_speech_no_key_and_offline_are_byte_identical_to_default(self):
        expected = Path("fixtures/speech-default.card.txt").read_bytes()
        off = b"Jev off (TYPESAFE_API_KEY not set)"
        for env, message in (({"TYPESAFE_API_KEY": ""}, off),
                             ({"RAVEN_EARS_OFFLINE": "1"}, b"Jev off (offline mode)")):
            with patch.dict(os.environ, env), card_fixture() as (path, *_):
                self.assertEqual(self.cli(path, '--no-picture').encode(), expected.replace(off, message))
                self.assertNotIn("jev_speech", json.loads(self.cli(path, "--json")))
        self.http.assert_not_called()

    def test_low_probability_joking_and_stale_wording_stay_silent(self):
        rows = M.read_speech([dict(t=1, text="I'm fine.")], [])
        with patch.dict(M.SPEECH, joking=dict(M.SPEECH["joking"], instructions="A different question")):
            self.assertEqual(M.speech_lines(rows), [])
        rows[0]["jev"]["joking"] = dict(probability=.5, outcome="not settled")
        self.assertEqual(M.speech_lines(rows), [])

    def test_voice_match_conflict_keeps_both_witnesses_and_ambiguous_claims_abstain(self):
        rows = M.read_speech([dict(t=1, text="Hey it's Aster.")], [])
        lines = M.speech_lines(rows, speaker="Birch", voice=dict(status="matched"))
        self.assertIn("voice book says Birch", lines[-1])
        self.assertFalse(any("WHO" in line for line in M.speech_lines(rows, speaker="Aster")))
        more = M.read_speech([dict(t=1, text="It's Aster. Love, Birch")], [])
        self.assertFalse(any("WHO" in line for line in M.speech_lines(more)))

    def test_who_rejects_non_names_in_real_failure_sentence_shapes(self):
        # Same introduction/ordinary-predicate shapes as the live failures;
        # words are invented. The fake model votes yes on every candidate.
        texts = ["Hey Aster, I'm on earbuds today.",
                 "Thanks for catching that before I stopped stumbling. I'm committing the patch shortly.",
                 "I'm the raven of thought.", "I'm On earbuds.", "I'm Committing the patch.",
                 "I'm Working late.", "This is Fine."]
        rows = M.read_speech([dict(t=i, text=t) for i, t in enumerate(texts)], [])
        self.assertTrue(all(not row["candidates"] for row in rows))
        self.assertFalse(any("WHO" in line for line in M.speech_lines(rows)))
        for call in self.http.call_args_list:
            self.assertFalse(any(q.startswith("self_") for q in json.loads(call.args[0].data)["questions"]))

    def test_who_allows_local_names_and_unfamiliar_capitalised_names_only(self):
        import voice_book
        names = Path(os.environ["RAVEN_EARS_HOME"]) / "names.txt"
        names.parent.mkdir(parents=True)
        names.write_text("# Recognition hints\nmarigold # local name\n\n")
        # Use the real enrolled-book reader with a valid invented print.
        voice_book.home().mkdir()
        voice_book.voice_path("aster").write_text(json.dumps(dict(
            embedding=[1.] + [0.] * 255, count=1, seconds=12.)))
        text = "I'm aster. My name is marigold. I'm Zinnia. I'm unlisted."
        self.assertEqual(M.self_candidates(text), ["aster", "marigold", "Zinnia"])
        self.assertEqual(M.self_candidates("Love, Zinnia. It's Aster. This is Birch."), ["Zinnia", "Aster", "Birch"])
        rows = M.read_speech([dict(t=1, text="It's aster.")], [])
        self.assertTrue(any("they're aster" in line for line in M.speech_lines(rows)))
        for call in self.http.call_args_list:
            payload = json.loads(call.args[0].data)
            self.assertNotIn("marigold", json.dumps(payload))
            self.assertNotIn("embedding", json.dumps(payload))
            self.assertEqual(set(payload["state"]), {"line", "before", "after"})

    def test_words_vs_voice_uses_tune_spans_and_sends_only_text(self):
        words = [("I", 1, 1.2), ("lost", 1.2, 1.6), ("hope.", 1.6, 2),
                 ("And", 2.25, 2.5), ("feel", 2.6, 3.5), ("down", 2.75, 3),
                 ("Check", 4, 4.6), ("the", 4.6, 5.1), ("meter", 5.3, 6.2),
                 ("I", 12, 12.6), ("feel", 12.6, 13.3), ("glad", 13.3, 14)]
        affect = dict(ok=True, device="cpu", n_windows=3,
                      whole=dict(arousal=.56, dominance=.5, valence=.6),
                      centers=np.array([2.5, 5., 13.5]), P=np.array([[.65, .5, .68], [.5, .5, .5], [.52, .5, .7]]))
        model = dict(status="ran", model="omine", spans=[[1.5, 2.1], [3.5, 4.]])
        with card_fixture() as (path, _, af, ev, _), patch("laugh_models.selected", return_value="omine"), \
             patch("laugh_models.run_detector", return_value=model):
            af.side_effect = lambda *_: deepcopy(affect)
            ev.side_effect = lambda *_: dict(events=[dict(label="Laughter", name="Laughter", start=1.4,
                                                         end=1.8, score=.5)], device="cpu", breath={})
            fe = R._front_end.return_value
            mask = (fe["times"] >= 1) & (fe["times"] < 3.5)
            fe["f0"][mask] *= 2 ** (np.linspace(4, 8, mask.sum()) / 12)
            fe["rms_db"][mask] = -16.
            result = R.analyze(path, words=words, identify_voice=False, update_baseline=False)
        data = result["data"]
        rows = data["jev_speech"]
        self.assertEqual([(r["t"], r["end"]) for r in rows], [(1, 3.5), (4, 6.2), (12, 14)])
        self.assertEqual([(r["t"], r["end"]) for r in rows],
                         [(r["start"], r["end"]) for r in data["tune"]["phrases"]])
        self.assertCountEqual([json.loads(call.args[0].data)["state"] for call in self.http.call_args_list], [
            dict(line="I lost hope. And feel down", before="", after="Check the meter I feel glad"),
            dict(line="Check the meter", before="I lost hope. And feel down", after="I feel glad"),
            dict(line="I feel glad", before="I lost hope. And feel down Check the meter", after=""),
        ])
        self.assertIn("Jev: 3 speech requests sent/attempted to TypeSafe; text only, never audio", result["notes"])
        self.assertAlmostEqual(rows[0]["jev"]["stated_feeling"], -.62)
        self.assertGreater(rows[0]["voice"]["pitch_st"], 5.9)
        self.assertAlmostEqual(rows[0]["voice"]["wpm"], 144)
        self.assertEqual(rows[0]["voice"]["loud_db"], -16.)
        self.assertEqual(rows[0]["voice"]["affect"], dict(arousal=.65, valence=.68))
        self.assertEqual(rows[0]["voice"]["ending"], data["tune"]["phrases"][0]["ending"])
        self.assertEqual(rows[0]["voice"]["laugh_sources"], ["CNN14", "omine"])
        self.assertEqual(rows[1]["voice"]["laugh_sources"], [])
        self.assertIsNone(rows[0]["voice"]["usual_pitch_st"])
        line, = [line for line in result["lines"] if line.startswith(M.WORDS_VOICE)]
        self.assertIn("0:01 words -0.62 · voice +6 st", line)
        self.assertIn("aro 0.65 / val 0.68", line)
        self.assertIn("laughing (CNN14, omine)", line)
        self.assertNotIn("lost hope", line)
        self.assertNotRegex(line, "[–—]")

    def test_words_vs_voice_baseline_offsets_use_phrase_pitch(self):
        words = [("I", 1, 1.5), ("lost", 1.5, 1.8), ("hope.", 1.8, 2.2),
                 ("And", 2.2, 2.5), ("feel", 2.5, 2.9), ("down", 2.9, 3.3)]
        baseline = R.baseline_path("Birch")
        baseline.parent.mkdir(parents=True)
        baseline.write_text(json.dumps(dict(pitch_hz=200., range_st=7.)) + "\n")
        before = baseline.read_bytes()
        with card_fixture() as (path, _, af, *_):
            affect = af.side_effect()
            affect.update(centers=np.array([2.5]), P=np.array([[.6, .5, .7]]))
            af.side_effect = lambda *_: deepcopy(affect)
            result = R.analyze(path, words=words, speaker="Birch", update_baseline=False)
        line, = [line for line in result["lines"] if line.startswith(M.WORDS_VOICE)]
        self.assertIn("+0 st (-3 vs usual), range 0 st (note 0, usual 7)", line)
        self.assertEqual(baseline.read_bytes(), before)

    def test_words_vs_voice_reads_past_the_old_phrase_cap(self):
        notes = []
        rows = M.read_speech([dict(t=i, end=i + .4, text=f"Signal {i}") for i in range(65)], notes)
        self.assertEqual(len(rows), self.http.call_count)
        self.assertEqual(len(rows), 65)
        self.assertIn("65 speech requests sent/attempted", notes[0])

    def comparison(self, t, stated, valence=None, **voice):
        fields = dict(pitch_st=0., range_st=4., note_range_st=8., usual_pitch_st=None, usual_range_st=None,
                      wpm=150., loud_db=-24., loud_vs_note_db=0., ending="→", laugh_sources=[],
                      affect=None if valence is None else dict(arousal=.5, valence=valence))
        return dict(t=t, jev=dict(status="ran", stated_feeling=stated), voice=fields | voice)

    def test_words_vs_voice_caps_contrasts_and_counts_majority(self):
        rows = [self.comparison(t, .8, val) for t, val in enumerate((.39, .21, .13, .28))]
        rows += [self.comparison(t, -.6, .36) for t in range(4, 9)]
        lines = M.words_vs_voice_lines(rows)
        self.assertEqual(len(lines), 4)
        self.assertEqual([line.split(" words ", 1)[0].split()[-1] for line in lines[:3]], ["0:01", "0:02", "0:03"])
        self.assertEqual(lines[-1], "    Jev and vocal valence point the same way in 5/9 phrases")
        self.assertEqual(M.words_vs_voice_lines(rows[4:5] + [self.comparison(9, .8, .54)]), [])

    def test_animated_laugh_can_flag_negative_words_without_affect(self):
        negative = dict(t=22, stated=-.62, pitch_st=6, wpm=240.)
        calm = [self.comparison(t, 0.) for t in (1, 4, 9)]
        with_laugh = self.comparison(**negative, laugh_sources=["omine"])
        line, = M.words_vs_voice_lines(calm + [with_laugh])
        self.assertIn("0:22 words -0.62", line)
        self.assertIn("240 wpm", line)
        self.assertIn("laughing (omine)", line)
        self.assertNotIn("val ", line)
        self.assertEqual(M.words_vs_voice_lines(calm + [self.comparison(**negative)]), [])
        self.assertEqual(M.words_vs_voice_lines(calm + [self.comparison(22, -.62, wpm=240.,
                                                                        laugh_sources=["omine"])]), [])
        self.assertEqual(M.words_vs_voice_lines(calm + [self.comparison(22, .62, wpm=240., pitch_st=6,
                                                                        laugh_sources=["omine"])]), [])

    def test_phrase_affect_needs_enough_window_overlap(self):
        affect = dict(ok=True, centers=np.array([1.5, 3., 4.5]),
                      P=np.array([[.9, .5, .8], [.4, .5, .3], [.1, .5, .1]]))
        self.assertIsNone(R._phrase_affect(affect, 1.1, 1.8, 6))
        scores = R._phrase_affect(affect, .75, 3.5, 6)
        self.assertEqual(scores, dict(arousal=.665, valence=.565))
        self.assertIsNone(R._phrase_affect(affect, 3.5, 4.3, 6))

    def test_words_vs_voice_partly_failed_batch_has_one_block_line(self):
        def flaky(req, timeout):
            if json.loads(req.data)["state"]["line"] == "Check the signal.":
                raise TimeoutError("invented transcript must not enter notes")
            return fake_http(req, timeout)
        with card_fixture() as (path, *_), patch.object(J.request, "urlopen", side_effect=flaky):
            result = R.analyze(path, update_baseline=False)
        self.assertEqual([line for line in result["lines"] if line.startswith(M.WORDS_VOICE)],
                         [M.WORDS_VOICE + "Jev unavailable (1/2 phrases)"])
        self.assertIn("2 speech requests sent/attempted", "\n".join(result["notes"]))
        self.assertNotIn("invented transcript", "\n".join(result["notes"]))
        self.assertEqual(M.words_vs_voice_lines(None), [M.WORDS_VOICE + "Jev skipped (no transcript)"])

    def test_filename_drops_trailing_notes_and_keeps_recording_versions(self):
        for note in ("edit 4m09, not album", "not album", "cut at 4m09", "copy 2", "clip", "upload 3"):
            with self.subTest(note=note):
                candidates = M.filename_candidates("Invented Singer - Lantern (" + note + ").mp3")
                self.assertIn(dict(artist="Invented Singer", title="Lantern"), candidates)
                self.assertFalse(any(note in c[k] for c in candidates for k in ("artist", "title")))
        for version in ("Taylor's Version", "2011 Remastered", "Remastered", "Live", "Acoustic", "Remix", "Radio Edit"):
            with self.subTest(version=version):
                title = "Lantern (" + version + ")"
                candidates = M.filename_candidates("Invented Singer - " + title + " (copy 2) (Lyric Video).mp3")
                self.assertIn(dict(artist="Invented Singer", title=title), candidates)
        with patch.object(credits, "file_tags", return_value=(None, None)):
            result = credits.identify("Invented Singer - Lantern (edit 4m09, not album).mp3", for_lookup=True)
        self.assertEqual(result, ("Invented Singer", "Lantern", "file name by Jev"))

    def test_misheard_labels_keep_rows_and_uncertain_or_failed_answers_add_nothing(self):
        rows = lyric_rows()
        before = deepcopy(rows)
        M.add_misheard(rows, [])
        text = '\n'.join(lyric_align.misheard_block(rows, M.stamp))
        self.assertIn("plausible mishearing", text)
        self.assertIn("possible changed words; text only", text)
        self.assertEqual(len(lyric_align.misheard_lines(rows)), 2)
        for old, row in zip(before, rows):
            reading = row["misheard"].pop("jev")
            self.assertEqual(old, row)
            reading["same"]["probability"] = .5
            self.assertEqual(M.misheard_label(reading), "")
        with patch.object(J.request, "urlopen", side_effect=TimeoutError):
            M.add_misheard(rows, [])
        self.assertNotIn("Jev", '\n'.join(lyric_align.misheard_block(rows, M.stamp)))
        rows[0]["approx"] = True
        self.http.reset_mock()
        M.add_misheard(rows, [])
        self.assertEqual(self.http.call_count, 1)

    def test_filename_respects_flags_tags_offline_cache_and_never_sends_a_path(self):
        filename = self.root / "Invented Singer - Lantern (Official Lyric Video) [HD].mp3"
        with patch.object(credits, "file_tags", return_value=(None, None)):
            self.assertEqual(credits.identify(str(filename))[2], "file name")
            self.http.assert_not_called()
            self.assertEqual(credits.identify(str(filename), for_lookup=True), ("Invented Singer", "Lantern", "file name by Jev"))
            count = self.http.call_count
            self.assertEqual(credits.identify(str(filename), for_lookup=True)[1], "Lantern")
            self.assertEqual(self.http.call_count, count)
            self.assertEqual(credits.identify(str(filename), "Given Artist", "Given Title", for_lookup=True),
                             ("Given Artist", "Given Title", "given"))
            self.assertEqual(credits.identify(str(filename), "Given Artist", for_lookup=True)[0], "Given Artist")
            with patch.dict(os.environ, RAVEN_EARS_OFFLINE="1"):
                self.assertEqual(credits.identify(str(filename), for_lookup=True)[2], "file name")
                self.assertEqual(self.http.call_count, count)
        with patch.object(credits, "file_tags", return_value=("Tag Artist", "Tag Title")):
            self.assertEqual(credits.identify(str(filename), for_lookup=True), ("Tag Artist", "Tag Title", "tags"))
            self.assertEqual(self.http.call_count, count)
        for call in self.http.call_args_list:
            state = json.loads(call.args[0].data)["state"]
            self.assertEqual(state["filename"], filename.name)
            self.assertNotIn(str(self.root), json.dumps(state))

    def test_retired_meaning_hook_does_not_print_from_old_saved_melody(self):
        import listen_melody as LM
        song = L.load(Path("fixtures/uhuru-title-song.json"))
        song["melody"] = LM.build(song)
        before = {hook: LM.render(song, hook=hook) for hook in (False, True)}
        for field, prefix in (("text", "% "), ("hook_text", "")):
            target = song["melody"] if field == "text" else song["melody"]["json"]
            target[field] = prefix + 'HOOK (Jev meaning): retired trial line\n' + target[field]
        for hook in (False, True):
            self.assertEqual(LM.render(song, hook=hook), before[hook])
        self.http.assert_not_called()

    def test_failed_speech_reading_leaves_acoustics_and_one_honest_stage(self):
        with card_fixture() as (path, *_), patch.object(J.request, "urlopen", side_effect=TimeoutError):
            data = json.loads(self.cli(path, "--json"))
        self.assertIn("pitch", data)
        self.assertTrue(all(r["jev"]["status"] == "unavailable" for r in data["jev_speech"]))
        stages = [r for r in data["stage_status"] if "jev" in r["stage"].lower()]
        self.assertEqual([(r["stage"], r["status"]) for r in stages], [("Jev speech", "failed")])

    def music_worker_song(self):
        import harmony
        import lrclib
        audio = self.root / "Invented Singer - Lantern (Official Lyric Video) [HD].mp3"
        audio.write_bytes(b"not decoded; fake front end")
        lyrics = self.root / "lyrics.txt"
        lyrics.write_text('\n'.join(r["text"] for r in lyric_rows()))
        def resolve(path, *args):
            credits.identify(path, for_lookup=True)
            return dict(lyrics=lyric_rows(), label="LYRICS: writer's text, 2 of 2 lines heard",
                        lookup=dict(status="file"), words=None)
        def subprocess_bridge(argv, **kwargs):
            # run_pass and worker are real; replace only process creation so fake
            # inference/HTTP can run inside the worker's real CLI/capture path.
            self.assertEqual(argv[2], "--_worker")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = L.worker(argv[3], argv[4], argv[5:])
            return subprocess.CompletedProcess(argv, code, out.getvalue(), err.getvalue())
        with ExitStack() as stack:
            for name, value in (("analyze_key", None), ("analyze_tempo", None), ("analyze_sections", []),
                                ("analyze_sound", None), ("analyze_vocals", None), ("analyze_vocal_quality", None),
                                ("meter_info_for", (None, None)), ("key_claim_for", None)):
                stack.enter_context(patch.object(music, name, return_value=value))
            stack.enter_context(patch.object(R, "load_audio", return_value=np.ones(6 * music.SR)))
            stack.enter_context(patch.object(harmony, "music_features", return_value=dict(
                chroma=None, rms_db=np.array([-30., -20., -30.]), oenv=None, mfcc=None, hp_energy=None, cent=None)))
            stack.enter_context(patch.object(lrclib, "for_music", side_effect=resolve))
            stack.enter_context(patch.object(credits, "file_tags", return_value=(None, None)))
            stack.enter_context(patch("doctored_audio.inspect", return_value=dict(status="ran")))
            stack.enter_context(patch.object(L, "ears_version", return_value="invented-version"))
            stack.enter_context(patch('listening_sheet.song_sheet', return_value=Path('sheet.png')))
            stack.enter_context(patch.object(L.subprocess, "run", side_effect=subprocess_bridge))
            runner = stack.enter_context(patch.object(L, "run_pass", wraps=L.run_pass))
            args = L.parser().parse_args([str(audio), "--only", "music", "--view", "music", "--lyrics", str(lyrics)])
            first, _ = L.execute(args)
            runner.assert_called_once()
            path = L.store_home() / L.file_hash(audio) / "song.json"
            song = L.load(path)
            data = song["cards"]["music"]["json"]
            self.assertEqual(data['listening_sheet']['rms_db'], [-30., -20., -30.])
            self.assertIn("answer", data["lyrics"][0]["misheard"]["jev"])
            self.assertNotIn("meaning_hook", data)
            self.assertTrue(all(not any(q.startswith("hook_") for q in json.loads(c.args[0].data)["questions"])
                                for c in self.http.call_args_list))
            self.assertEqual(data["filename_reading"]["artist"], "Invented Singer")
            self.assertIn(M.FILENAME_NOTE, first)
            self.assertIn("plausible mishearing", first)
            before = path.read_bytes()
            self.http.reset_mock()
            with patch.dict(os.environ, TYPESAFE_API_KEY="", RAVEN_EARS_OFFLINE="1"):
                again, _ = L.execute(args)
            self.assertEqual(first, again)
            self.assertEqual(path.read_bytes(), before)
            self.http.assert_not_called()
            self.assertEqual(runner.call_count, 1)
            return data

    def test_real_music_worker_run_pass_scrub_save_and_replay(self):
        self.music_worker_song()

    def test_path_like_text_survives_real_worker_capture_in_every_new_payload(self):
        original = lyric_rows
        def rows_with_slashes():
            rows = original()
            rows[0]["misheard"]["wrote"] += " /invented/path"
            return rows
        with patch(__name__ + ".lyric_rows", side_effect=rows_with_slashes):
            data = self.music_worker_song()
        self.assertIn("/invented/path", data["lyrics"][0]["misheard"]["jev"]["payload"]["state"]["wrote"])


def run_tests(check):
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(MoreJevTests).run(result)
    problems = {test.id().rsplit(".", 1)[-1]: detail for test, detail in result.failures + result.errors}
    for name in unittest.defaultTestLoader.getTestCaseNames(MoreJevTests):
        check("more Jev: " + name, name not in problems, problems.get(name, ""))


if __name__ == "__main__":
    unittest.main()
