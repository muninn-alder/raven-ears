"""Fake passes and read-only saved cards; no audio decoding, models or services."""
from contextlib import redirect_stdout, redirect_stderr, nullcontext
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import listen as L
import listen_melody as M


class ListenTests(unittest.TestCase):
    def test_language_and_profile_changes_notice_without_relistening(self):
        with patch.dict(os.environ, {'RAVEN_EARS_LANG': 'en', 'RAVEN_EARS_PROFILE': ''}):
            self.stdout()
            original = self.path().read_bytes()
            self.calls.clear()
            with patch.dict(os.environ, {'RAVEN_EARS_LANG': 'fr'}):
                self.assertIn('Saved card options differ: environment', self.stdout())
            with patch.dict(os.environ, {'RAVEN_EARS_PROFILE': 'cpu'}):
                self.assertIn('Saved card options differ: environment', self.stdout())
            with patch.dict(os.environ, {'RAVEN_LYRICS_WHISPER': 'large-v3-turbo'}):
                self.assertIn('Saved card options differ: environment', self.stdout())
            self.assertEqual(self.calls, [])
            self.assertEqual(self.path().read_bytes(), original)
            with patch.dict(os.environ, {'RAVEN_EARS_PROFILE': 'cpu', 'RAVEN_EARS_LANG': 'fr'}):
                self.stdout('--fresh')
            saved = self.song()['manifest']['options']['environment']
            self.assertEqual(saved['RAVEN_EARS_PROFILE'], 'cpu')
            self.assertEqual(saved['RAVEN_EARS_LANG'], 'fr')
            self.assertEqual(self.calls, list(L.PASSES))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".listen-tests-", dir=os.environ.get("TMPDIR", L.HERE))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "Invented - Lantern.mp3"
        self.audio.write_bytes(b"invented bytes, deliberately not audio")
        self.env = patch.dict(os.environ, {"RAVEN_EARS_HOME": str(self.root / "ears"),
                                          "RAVEN_EARS_PICTURE": "",
                                          "RAVEN_EARS_NO_BASELINE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ("RAVEN_EARS_PROFILE", "CUDA_VISIBLE_DEVICES", "RAVEN_EARS_OFFLINE",
                     "WHISPER_MODEL", "RAVEN_EARS_DEVICE"):
            os.environ.pop(name, None)
        self.version = patch.object(L, "ears_version", return_value="test-version-1")
        self.version.start()
        self.addCleanup(self.version.stop)
        self.calls = []
        self.runner = patch.object(L, "run_pass", side_effect=self.fake_pass)
        self.passes = self.runner.start()
        self.addCleanup(self.runner.stop)
        self.quality = patch('doctored_audio.inspect', return_value={
            'status': 'ran', 'sample_rate': 32000, 'bass_share': .2})
        self.quality_probe = self.quality.start()
        self.addCleanup(self.quality.stop)
        sheet = patch('listening_sheet.song_sheet', return_value=Path('invented-sheet.png'))
        self.draw = sheet.start()
        self.addCleanup(sheet.stop)

    def test_picture_default_and_quick_keep_sheet_without_relistening(self):
        for flags in ((), ('--quick',)):
            with self.subTest(flags=flags):
                text, _ = self.call('--fresh', *flags)
                self.assertIn('PICTURE:', text)
                self.assertEqual(self.draw.call_count, 1)
                music_args = next(call.args[1] for call in self.passes.call_args_list if call.args[0] == 'music')
                self.assertIn('--_picture-data', L.pass_args('music', music_args))
                before = self.path().read_bytes()
                self.calls.clear()
                for controls in ((), ('--no-picture',), ('--picture',)):
                    self.call(*flags, *controls)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.path().read_bytes(), before)
                self.assertEqual(self.draw.call_count, 3)
                self.draw.reset_mock()
                self.passes.reset_mock()

    def test_picture_env_and_flags(self):
        for env, flags, enabled in (
                ('', (), True), ('0', (), False), ('1', (), True),
                ('', ('--no-picture',), False), ('0', ('--picture',), True),
                ('1', ('--no-picture',), False), ('0', ('--by-ear', 'marks.txt'), True),
                ('1', ('--by-ear', 'marks.txt', '--no-picture'), False)):
            with self.subTest(env=env, flags=flags), patch.dict(os.environ, {'RAVEN_EARS_PICTURE': env}):
                self.draw.reset_mock()
                text, _ = self.call(*flags)
                self.assertEqual(self.draw.called, enabled)
                self.assertEqual('PICTURE:' in text, enabled)

    def test_missing_default_picture_preserves_card_and_skips_capture(self):
        with patch.dict(sys.modules, {'matplotlib': None}):
            plain, _ = self.call('--no-picture')
            self.draw.reset_mock()
            before = self.path().read_bytes()
            text, _ = self.call()
            self.assertEqual(text, plain + 'pip install -r requirements-picture.txt to draw the listening sheet\n')
            self.assertEqual(self.path().read_bytes(), before)
            self.draw.assert_not_called()
            with patch.object(L, 'run_pass', wraps=self.fake_pass) as passes:
                self.call('--fresh')
            self.assertTrue(all(not call.args[1].picture for call in passes.call_args_list))
            self.draw.side_effect = ImportError('missing matplotlib', name='matplotlib')
            explicit, _ = self.call('--picture')
            self.assertTrue(explicit.endswith('pip install matplotlib to draw the listening sheet\n'))

    def test_picture_default_leaves_json_and_management_alone(self):
        self.call()
        self.draw.reset_mock()
        with patch.dict(sys.modules, {'matplotlib': None}):
            self.assertEqual(json.loads(self.call('--json')[0])['schema'], 1)
            self.call('--history')
            self.call('--delete')
            L.execute(L.parser().parse_args(['--delete-all']))
        self.draw.assert_not_called()

    def fake_pass(self, name, args):
        self.calls.append(name)
        texts = {"music": "MUSIC: Lantern  1:00\nTEMPO: 80 bpm\nLYRICS:\n  [0:04] An invented lantern\n",
                 "singer": "SINGER.PY Lantern 1:00\nSINGER:\n  an invented soft voice\n",
                 "harmony": "HARMONY: Lantern 1:00\nTENSION\n  an invented gentle rise\n"}
        return {"status": "ran", "reason": "", "text": texts[name],
                "json": {"duration": 60, "stage_status": []}, "diagnostics": ""}

    def call(self, *flags):
        return L.execute(L.parser().parse_args([str(self.audio), *flags]))

    def path(self):
        return L.store_home() / L.file_hash(self.audio) / "song.json"

    def song(self):
        return L.load(self.path())

    def test_audio_quality_is_saved_once_and_replayed_without_decoding(self):
        self.quality_probe.return_value = dict(status='ran', sample_rate=32000, bass_share=.93206)
        first, _ = self.call()
        self.assertIn('very bass-heavy: 93.2%', first)
        self.assertEqual(self.song()['audio_quality'], self.quality_probe.return_value)
        before = self.path().read_bytes()
        self.quality_probe.side_effect = AssertionError('saved views never decode')
        replay, _ = self.call()
        self.assertIn('very bass-heavy: 93.2%', replay)
        self.assertEqual(self.path().read_bytes(), before)
        self.assertEqual(self.quality_probe.call_count, 1)

    def notation_pass(self, name, args):
        result = self.fake_pass(name, args)
        if name == "singer":
            result["json"].update(melody_input={
                "f0": [p for _ in range(4) for p in
                       (261.625565, 293.664768, 329.627557, 293.664768) for _ in range(25)],
                "fps": 100, "words": [[word, i, i + .9] for i, word in enumerate(
                    ("Lantern", "glowing", "Lantern", "home"))],
                "lyric_lines": ["Lantern", "glowing", "Lantern", "home"],
                "lyrics_label": "LYRICS: invented timed words"},
                singer={"data": {"peak": {"ok": True, "start": 2, "end": 2.8}}})
        if name == "music":
            result["json"].update(key_claim={"key": "C major", "state": "clear", "line": "C major"},
                                  tempo={"bpm": 112, "regularity": .83})
        return result

    def test_hook_is_saved_and_full_melody_is_derived_without_engines(self):
        import melody_abc
        with patch.object(L, "run_pass", side_effect=self.notation_pass), \
             patch.object(melody_abc, "transcribe", wraps=melody_abc.transcribe) as transcribe:
            first, _ = self.call()
            self.assertEqual(transcribe.call_count, 1)
            self.assertEqual(transcribe.call_args.kwargs["key"], "C major")
            self.assertEqual(transcribe.call_args.kwargs["pitch_rule"], "centre")
            self.assertEqual(transcribe.call_args.kwargs["mode"], "proportional")
            self.assertEqual(len(transcribe.call_args.kwargs["line_breaks"]), 3)
            self.assertEqual(first.count("\nw: "), 2)
            song = self.song()
            self.assertEqual(song["melody"]["json"]["hook"]["line_indices"], [0, 1])
            self.assertIn("most-repeated lyric line", first)
            before = self.path().read_bytes()
            self.calls.clear()
            with patch.dict(sys.modules, {"singer": None, "music": None, "harmony": None, "torch": None}):
                whole, _ = self.call("--view", "melody")
                again, _ = self.call()
            self.assertEqual(first, again)
            self.assertEqual(whole.count("\nw: "), 4)
            self.assertEqual(whole.count("rhythm:"), 1)
            self.assertIn("rhythm: not notated; the card measures 112 BPM, steady 0.83", whole)
            self.assertEqual(self.path().read_bytes(), before)
            self.assertEqual(self.calls, [])
            self.assertEqual(transcribe.call_count, 2)

    def test_saved_note_labels_and_notation_refresh_in_card_and_sheet_views(self):
        import listening_sheet

        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call()
        song = self.song()
        music = song["cards"]["music"]["json"]
        music["key_claim"].update(key="G major", state="clear")
        harmony = song["cards"]["harmony"]
        harmony["json"].update(key_verdict={"key": "G major", "state": "clear"},
                               key_change_scan={"status": "ran", "claims": [
                                   {"from_key": "G major", "to": "Db major", "move": "tritone",
                                    "range": [25, 50], "destination_support": [30, 50]}]})
        harmony["text"] += ("\nCHILL CANDIDATES (candidates, not verdicts)\n"
                            "  appoggiaturas (vocal stem; treat as hints):\n"
                            "    0:08  G held 0.3 s over Dmaj7, then down onto G♭\n"
                            "    0:35  G held 0.3 s over Dmaj7, then down onto F♯\n")
        singer = song["cards"]["singer"]
        singer["text"] += "  highest held notes: G♭4 at 0:09 (0.6s) · F♯4 at 0:36 (0.6s)\n"
        song["melody"]["json"]["notation"].pop("pitch_revision")
        L.save(self.path(), song)
        before = self.path().read_bytes()
        self.calls.clear()

        feeling, _ = self.call()
        full, _ = self.call("--view", "full")
        abc, _ = self.call("--view", "melody")
        self.assertIn("appoggiatura: G→F♯ over Dmaj7 (0.3 s)", feeling)
        self.assertIn("highest held note: F♯4", feeling)
        self.assertIn("highest held note: G♭4", feeling)
        self.assertIn("G held 0.3 s over Dmaj7, then down onto F♯", full)
        self.assertIn("G held 0.3 s over Dmaj7, then down onto G♭", full)
        self.assertIn("\nK:G\n", full)
        self.assertIn("\nK:G\n", abc)
        sheet = listening_sheet.song_evidence(song, feeling, L.full_text(song))
        labels = {t: label for t, _, label, _ in sheet["events"] if "held 0.3 s" in label}
        self.assertTrue(labels[8].endswith("F♯"), labels)
        self.assertTrue(labels[35].endswith("G♭"), labels)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.path().read_bytes(), before)

    def test_old_raw_writer_sheet_is_cleaned_in_every_saved_view(self):
        import lyric_align as LA
        import slim
        def raw_sheet(name, args):
            card = self.notation_pass(name, args)
            raw = ["Harbour", "(for the paper moon)", "[Verse 1]", "Lantern", "glowing", "[Chorus]", "Lantern", "home"]
            words = [[word, i, i + .9] for i, word in enumerate(("Lantern", "glowing", "Lantern", "home"))]
            rows, summary = LA.align_lyrics(raw, words, 4, headers=False)
            label = LA.summary_line(summary)
            text = label + "\n" + "\n".join(f"  [{slim.clock(r['t'])}{'≈' if r['approx'] else ''}] {r['text']}" for r in rows) + "\n"
            card["text"] += text
            source = {"source": "file", "status": "file"}
            if name == "singer":
                card["json"]["melody_input"].update(lyric_lines=raw, lyrics_label=label)
                card["json"]["lyrics"] = dict(rows=rows, summary=summary, label=label, lookup=source)
            elif name == "music":
                card["json"].update(lyrics=rows, lyrics_align=label, lyrics_source=source)
            else:
                card["json"].update(lyrics_aligned=dict(lines=rows, summary=summary, label=label), lyrics_source=source)
                card["text"] += ('  overall: the words sit near neutral, out of 8 lines\n'
                                 f'ACHE {slim.clock(rows[5]["t"])} repeated line under ≈ "[Chorus]" (lean)\n')
            return card
        with patch.object(L, "run_pass", side_effect=raw_sheet):
            self.call("--title", "Harbour")
        before = self.path().read_bytes()
        song = self.song()
        song["melody"] = M.build(song)
        self.assertEqual(song["melody"]["status"], "ran", song["melody"].get("reason"))
        self.assertEqual(song["melody"]["json"]["lyric_alignment"]["summary"]["total"], 4)
        self.assertEqual(song["melody"]["json"]["lyric_alignment"]["rows"][0]["text"], "Lantern")
        for view in ("first", "full", "music", "singer", "harmony", "melody"):
            out, _ = L.render(song, view)
            self.assertNotIn("[Verse 1]", out, view)
            self.assertNotIn("[Chorus]", out, view)
            self.assertNotIn("(for the paper moon)", out, view)
            self.assertNotRegex(out, r"\[\d+:\d\d≈\] Harbour")
            self.assertNotIn("of 8 lines", out, view)
            if view == "harmony":
                self.assertIn("word/music aggregate unavailable", out)
        self.assertEqual(self.path().read_bytes(), before)

    def test_melody_failures_missing_inputs_and_explicit_retry(self):
        with patch.object(L, "run_pass", side_effect=self.notation_pass), \
             patch("melody_abc.transcribe", side_effect=ValueError("invented failure")):
            first, _ = self.call()
        self.assertIn("melody: failed: ValueError: invented failure", first)
        self.calls.clear()
        self.call("--view", "melody", "--run-missing")
        self.assertEqual(self.song()["melody"]["status"], "ran")
        self.assertEqual(self.calls, [])
        song = self.song()
        song["cards"]["singer"]["json"]["melody_input"]["words"] = None
        self.assertEqual(M.build(song)["reason"], "timed words unavailable")
        song["cards"]["singer"]["json"].pop("melody_input")
        self.assertIn("vocal-stem pitch track unavailable", M.build(song)["reason"])

    def test_melody_unclear_key_keeps_music_top_candidate(self):
        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call()
        song = self.song()
        song["cards"]["music"]["json"]["key_claim"] = {
            "key": None, "working": "A minor", "state": "tonic", "line": "tonic unclear"}
        result = M.build(song)
        self.assertEqual(result["status"], "ran")
        self.assertEqual(result["json"]["settings"]["key"], "A minor")
        self.assertIn("K:Am", result["text"])
        self.assertIn("(key unclear)", result["text"])

    def test_hook_discloses_low_confidence_note_omission(self):
        from test_melody_abc import track
        import numpy as np
        f0 = track((59, .5), (None, .1), (47, .14), (None, .1), (66, .5),
                   (None, .1), (66, .5), (None, .1), (66, .5))
        probabilities = np.full(len(f0), .95)
        probabilities[60:74] = .08
        singer = self.notation_pass("singer", []).copy()
        singer["json"]["melody_input"] = {
            "f0": f0.tolist(), "fps": 100, "voiced_prob": probabilities.tolist(),
            "words": [["Lantern glows", 0, len(f0) / 100]]}
        song = {"manifest": {"passes": {"singer": {"status": "ran"}, "music": {"status": "ran"}},
                             "options": {"title": "Lantern"}},
                "cards": {"singer": singer, "music": self.notation_pass("music", [])}}
        result = M.build(song)
        self.assertEqual(result["status"], "ran", result["reason"])
        comment = "% 1 low note left out: likely tracker octave error"
        self.assertIn(comment, result["text"])
        self.assertIn(comment, result["json"]["hook_text"])
        self.assertNotIn(47, [n["midi"] for n in result["json"]["notes"]])

    def test_hook_uses_peak_without_repeats_and_never_invents_a_peak(self):
        rows = [{"start": i, "end": i + 1} for i in range(4)]
        texts = ["first", "second", "third", "last"]
        indices, rule = M.hook_indices(texts, rows, {"ok": True, "start": 1.2, "end": 1.8})
        self.assertEqual(indices, [1, 2])
        self.assertIn("most intense moment", rule)
        self.assertEqual(M.hook_indices(texts, rows, {})[0], [])

    def test_word_phrases_split_at_pauses_including_the_threshold(self):
        words = [("one", 1, 2), ("two", 2.34, 3), ("three", 3.35, 4),
                 ("four", 4.5, 5)]
        self.assertEqual(M.phrase_word_boundaries(words, 6), [3.175, 4.25])

    def test_word_phrases_force_the_largest_safe_gap_before_seven_seconds(self):
        words = [("one", 0, 2), ("two", 2.3, 4), ("three", 4.4, 6.4),
                 ("four", 6.5, 8.5), ("five", 8.6, 10)]
        # Disable natural pause splitting to isolate the forced choice: the
        # largest gap is before 'three', not at the last possible boundary.
        self.assertEqual(M.phrase_word_boundaries(words, 11, pause_s=.5), [4.2])

    def test_word_phrases_never_cut_an_overlapping_or_long_word(self):
        words = [("held", 0, 8), ("overlap", 1, 2), ("tail", 7.9, 8.5),
                 ("after", 8.6, 10), ("last", 10.4, 11)]
        cuts = M.phrase_word_boundaries(words, 12)
        self.assertEqual(cuts, [8.55, 10.2])
        self.assertFalse(any(start < cut < end for _, start, end in words for cut in cuts))
        self.assertEqual(M.phrase_word_boundaries([("held", 0, 10)], 11), [])
        self.assertEqual(M.phrase_word_boundaries([], 11), [])

    def test_peak_hook_contains_the_moment_and_bounds_the_actual_window(self):
        rows = [{"start": 22, "end": 28}, {"start": 112, "end": 118},
                {"start": 118, "end": 123}]
        peak = {"ok": True, "start": 113, "end": 115}
        indices, rule = M.hook_indices(["early", "at peak", "later"], rows, peak)
        self.assertEqual(indices, [1])
        self.assertEqual(rule, "line at the singer's most intense moment")
        # The second row fits only when the entire displayed window fits.
        rows[2]["end"] = 120
        self.assertEqual(M.hook_indices(["early", "at peak", "later"], rows, peak)[0], [1, 2])
        rows[1]["end"] = 121
        self.assertEqual(M.hook_indices(["early", "at peak", "later"], rows, peak)[0], [])
        peak.update(start=80, end=84)
        self.assertEqual(M.hook_indices(["early", "at peak", "later"], rows, peak)[0], [])

    def test_whisper_row_repetition_compares_normalized_words(self):
        texts = ["We don't drift!", "Next phrase", "WE DON’T DRIFT.", "Ending"]
        rows = [{"start": i * 3, "end": i * 3 + 2} for i in range(4)]
        indices, rule = M.hook_indices(texts, rows, {})
        self.assertEqual(indices, [0, 1])
        self.assertIn("most-repeated", rule)

    def test_one_word_of_a_longer_title_is_not_the_title(self):
        texts = ["I am your demon, never leaving", "Will never fade away", "A thing of beauty"]
        rows = [{"start": 4.0 * i, "end": 4.0 * i + 3, "note_indices": list(range(5)), "lyrics": t}
                for i, t in enumerate(texts)]
        title = M.words_of("Never Fade Away")
        indices, rule = M.hook_indices(texts, rows, {}, title)
        self.assertEqual(indices[0], 1)
        self.assertIn('"never fade away"', rule)
        # The title line too sparse to print: the lone "never" row must not stand in for it.
        rows[1]["note_indices"] = [0, 1]
        indices, rule = M.hook_indices(texts, rows, {}, title, min_notes=4)
        self.assertEqual(indices, [])
        self.assertNotIn("line with the title", rule)

    def test_hook_counts_sung_lines_instead_of_repeated_section_headings(self):
        texts = ["[Chorus]", "Lantern glowing", "Bring it home", "[Chorus]",
                 "Lantern glowing", "Bring it home"]
        rows = [{"start": i, "end": i + 1} for i in range(len(texts))]
        self.assertEqual(M.hook_indices(texts, rows, {})[0], [1, 2])

    def test_hook_prefers_the_title_line_over_a_tie_of_repeats(self):
        texts = ["[Chorus]", "Lantern glowing", "Bring it home", "Is Embers: it's bright", "Walk on",
                 "[Chorus]", "Lantern glowing", "Bring it home", "Is Embers: it's bright", "Walk on"]
        rows = [{"start": i, "end": i + 1} for i in range(len(texts))]
        indices, rule = M.hook_indices(texts, rows, {}, ["embers"])
        self.assertEqual(indices, [3, 4])
        self.assertEqual(rule, 'line with the title "embers" and the line after it')
        # No title, or a title that is not in the lyrics: today's rule, unchanged.
        for title in (None, [], ["comet"]):
            indices, rule = M.hook_indices(texts, rows, {}, title)
            self.assertEqual(indices, [1, 2])
            self.assertIn("most-repeated lyric line", rule)
        # A row that is only the title is a heading, not a sung line.
        heads = ["EMBERS", "Lantern glowing", "Is Embers: it's bright", "Walk on"]
        self.assertEqual(M.hook_indices(heads, rows[:4], {}, ["embers"])[0], [2, 3])
        # A short common word is not enough to prove a title.
        self.assertEqual(M.hook_indices(["we go on", "so we sing"], rows[:2], {}, ["we"])[0], [])

    def test_title_words_from_metadata_or_filename(self):
        self.assertEqual(M.title_words({"title": "Don't Embers"}), ["dont", "embers"])
        self.assertEqual(M.title_words({"metadata_filename": "Invented Band - Embers (Official Lyric Video).mp3",
                                        "artist": "Invented Band"}), ["embers"])
        self.assertEqual(M.title_words({"metadata_filename": "invented-embers-glow-taylors-version-official-video.mp3"}),
                         ["invented", "embers", "glow"])
        self.assertEqual(M.title_words({"metadata_filename": None}), [])

    def test_unplaceable_lyric_lines_fall_back_to_word_phrases(self):
        # I Miss You, 10 Nov 2026: repeated short lines failed the whole melody pass.
        import listen_melody as M, melody_abc as A
        fixture = L.HERE / "fixtures" / "uhuru-title-song.json"
        song = L.validate_song(json.loads(fixture.read_text(encoding="utf-8")))
        with patch.object(A, "lyric_line_boundaries",
                          side_effect=ValueError("lyric lines cannot be placed at distinct word-safe boundaries")):
            song["melody"] = M.build(song)
        self.assertEqual(song["melody"]["status"], "ran", song["melody"].get("reason"))
        self.assertIn("word phrases", song["melody"]["json"]["header"])
        self.assertIn("cannot be placed", song["melody"]["json"]["lyric_lines_unplaced"])
        self.assertEqual(song["melody"]["json"]["settings"]["line_breaks"],
                         M.phrase_word_boundaries(
                             song["cards"]["singer"]["json"]["melody_input"]["words"],
                             song["melody"]["json"]["settings"]["duration_s"]))

    def test_real_shaped_bundle_without_lyrics_has_short_word_safe_rows(self):
        song = L.validate_song(json.loads((L.HERE / "fixtures/uhuru-title-song.json").read_text()))
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture["lyric_lines"] = []
        result = M.build(song)
        self.assertEqual(result["status"], "ran", result["reason"])
        data = result["json"]
        self.assertGreater(len(data["lines"]), 10)
        cuts = data["settings"]["line_breaks"]
        self.assertFalse(any(a < cut < b for _, a, b in capture["words"] for cut in cuts))
        for row in data["lines"]:
            words = [w for w in data["words"] if row["start"] <= w["start"] < row["end"]]
            if words:
                self.assertLessEqual(max(w["end"] for w in words) - words[0]["start"], 7 + 1e-9)
        self.assertIn('HOOK: line with the title "uhuru"', data["hook_text"])
        indices = data["hook"]["line_indices"]
        self.assertLessEqual(len(indices), 2)
        notes = [data["notes"][j] for i in indices for j in data["lines"][i]["note_indices"]]
        self.assertLessEqual(max(n["end"] for n in notes) - min(n["start"] for n in notes), 8)

    def test_whisper_segment_text_is_not_treated_as_placed_lyrics(self):
        song, _ = self.long_song()
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture["lyrics_label"] = "LYRICS: lookup off; words are Whisper's guesses (rough times)"
        with patch("melody_abc.lyric_line_boundaries", side_effect=AssertionError("not supplied lyrics")):
            result = M.build(song)
        self.assertEqual(result["status"], "ran", result["reason"])
        self.assertNotIn("lyric_alignment", result["json"])
        self.assertGreater(len(result["json"]["lines"]), 1)

    def test_no_hook_is_one_honest_line_and_peak_uses_displayed_bounds(self):
        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call()
        song = self.song()
        song["manifest"]["options"]["title"] = "Unknown title"
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture.update(lyric_lines=[], words=[["brief", 2, 3]], f0=[None] * 400)
        peak = song["cards"]["singer"]["json"]["singer"]["data"]["peak"]
        peak.update(start=.5, end=1)
        result = M.build(song)
        self.assertEqual(result["status"], "ran", result["reason"])
        self.assertEqual(result["json"]["hook_text"],
                         "no hook: no sung lyric words paired with rendered notes\n")
        peak.update(start=2.2, end=2.8)
        self.assertEqual(M.build(song)["json"]["hook_text"],
                         "no hook: no sung lyric words paired with rendered notes\n")

    def test_built_peak_hook_prints_the_peak_phrase_instead_of_the_early_row(self):
        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call()
        song = self.song()
        song["manifest"]["options"]["title"] = "Unknown title"
        capture = song["cards"]["singer"]["json"]["melody_input"]
        f0 = [None] * 12000
        words = [["early", 22, 23], ["peak", 112, 114], ["after", 115, 117]]
        for _, start, end in words:
            f0[start * 100:end * 100] = [(261.625565, 293.664768)[(i // 40) % 2]
                                        for i in range((end - start) * 100)]
        capture.update(lyric_lines=[], words=words, f0=f0)
        song["cards"]["singer"]["json"]["singer"]["data"]["peak"].update(start=113, end=114)
        result = M.build(song)
        self.assertEqual(result["status"], "ran", result["reason"])
        self.assertEqual(result["json"]["hook"]["line_indices"], [1, 2])
        hook = result["json"]["hook_text"]
        self.assertIn("most intense moment", hook)
        self.assertIn("% [1:52]", hook)
        self.assertIn("w: peak", hook)
        self.assertNotIn("early", hook)

    def test_zero_length_word_without_lyric_lines_does_not_crash(self):
        import listen_melody as M
        fixture = L.HERE / "fixtures" / "uhuru-title-song.json"
        song = L.validate_song(json.loads(fixture.read_text(encoding="utf-8")))
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture["lyric_lines"] = []
        t = capture["words"][5][1]
        capture["words"].insert(6, ["blip", t, t])
        song["melody"] = M.build(song)
        self.assertEqual(song["melody"]["status"], "ran", song["melody"].get("reason"))

    def test_rhythm_line_quotes_the_tempo_the_card_leads_with(self):
        # I Miss You, 10 Nov 2026: the rhythm line said 143.6 while TEMPO said 111.1.
        import listen_melody as M
        music = {"tempo": {"bpm": 143.6, "regularity": 0.78}, "meter_claim": "4/4"}
        meter = {"engine": "beat_this", "tracker_bpm": 111.1, "bar_consistency": 0.95, "beats_per_bar": 4,
                 "verdict": "4/4", "confidence": 0.9}
        line = M.rhythm_line(music, meter)
        self.assertIn("111.1 BPM", line)
        self.assertNotIn("143.6", line)
        self.assertIn("4/4", line)
        # 10 Nov 2026: steady beat_this leads even when the trackers agree.
        self.assertIn("144 BPM", M.rhythm_line(music, dict(meter, tracker_bpm=144.0)))
        self.assertIn("143.6 BPM", M.rhythm_line(music, None))

    def test_zero_length_whisper_word_does_not_fail_the_melody(self):
        # Feeling This, 10 Nov 2026: five words with start == end failed the whole pass.
        import listen_melody as M
        fixture = L.HERE / "fixtures" / "uhuru-title-song.json"
        song = L.validate_song(json.loads(fixture.read_text(encoding="utf-8")))
        words = song["cards"]["singer"]["json"]["melody_input"]["words"]
        t = words[5][1]
        words.insert(6, ["blip", t, t])
        song["melody"] = M.build(song)
        self.assertEqual(song["melody"]["status"], "ran", song["melody"].get("reason"))
        self.assertEqual(song["melody"]["json"]["words_dropped_zero_length"], 1)

    def test_title_hook_on_a_real_shaped_bundle(self):
        # Committed fixture: the structure of a real listen (repeated chorus lines tie), every lyric
        # word invented except the title word.
        fixture = L.HERE / "fixtures" / "uhuru-title-song.json"
        song = L.validate_song(json.loads(fixture.read_text(encoding="utf-8")))
        texts = song["cards"]["singer"]["json"]["melody_input"]["lyric_lines"]
        self.assertEqual(M.hook_indices(texts, [{}] * len(texts), {}, [])[1],
                         "most-repeated lyric line and the line after it (ties: first in song)")
        song["melody"] = M.build(song)
        self.assertEqual(song["melody"]["status"], "ran")
        hook = song["melody"]["json"]["hook_text"]
        self.assertIn('HOOK: line with the title "uhuru" and the line after it', hook)
        rows = [line for line in hook.splitlines() if line.startswith("% [")]
        self.assertEqual(len(rows), 2)
        self.assertIn("Uhuru", rows[0])
        alignment = song["melody"]["json"]["lyric_alignment"]["rows"]
        self.assertEqual(rows[0].split("] ")[1], alignment[song["melody"]["json"]["hook"]["line_indices"][0]]["text"])

    def test_legacy_melody_view_and_default_remain_read_only(self):
        self.call()
        song = self.song()
        song.pop("melody")
        L.save(self.path(), song)
        before = self.path().read_bytes()
        self.calls.clear()
        whole, _ = self.call("--view", "melody", "--run-missing")
        first, _ = self.call()
        self.assertEqual(whole, M.LEGACY + 'PICTURE: invented-sheet.png\n')
        self.assertIn(M.LEGACY.strip(), first)
        self.assertEqual(self.path().read_bytes(), before)
        self.assertEqual(self.calls, [])

    def test_hook_accidentals_do_not_depend_on_omitted_rows(self):
        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call()
        song = self.song()
        inputs = song["cards"]["singer"]["json"]["melody_input"]
        inputs["f0"] = [p for base in (277.182631, 277.182631, 261.625565, 277.182631)
                        for p in (base, 293.664768, base, 293.664768) for _ in range(25)]
        inputs["lyric_lines"] = ["opening", "Lantern", "home", "Lantern"]
        inputs["words"] = [[word, i, i + .9] for i, word in enumerate(inputs["lyric_lines"])]
        result = M.build(song)
        self.assertEqual(result["status"], "ran")
        self.assertEqual(result["json"]["hook"]["line_indices"], [1, 2])
        self.assertIn("^C", result["json"]["hook_text"])
        self.assertIn("=C", result["json"]["hook_text"])

    def long_song(self, label="LYRICS: invented timed words", heard=None):
        """Four lyric lines, the long one twice; one word per 0.3 s, two notes per word."""
        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call()
        song = self.song()
        long_line = "Lanterns glow above invented harbours while distant bells ring over silver water tonight softly and slowly while every lantern remembers"
        lines = [long_line, "Bring it home", long_line, "Bring it home"]
        words = [w for line in lines for w in line.split()]
        timed = heard or words
        pitches = [261.625565, 293.664768, 329.627557]
        inputs = song["cards"]["singer"]["json"]["melody_input"]
        inputs.update(lyric_lines=lines, lyrics_label=label,
                      words=[[w, i * .3, i * .3 + .28] for i, w in enumerate(timed)],
                      f0=[pitches[int(t / 15) % 3] for t in range(int(len(timed) * .3 * 100))])
        return song, lines

    def test_hook_is_pitches_and_words_only_and_fits_the_card_width(self):
        from stage_status import NOTES_WIDTH
        song, lines = self.long_song()
        result = M.build(song)
        self.assertEqual(result["status"], "ran")
        hook = result["json"]["hook_text"].splitlines()
        self.assertEqual(result["json"]["hook"]["line_indices"], [0, 1])
        self.assertTrue(all(len(row) <= NOTES_WIDTH for row in hook), max(hook, key=len))
        self.assertFalse(any(row.startswith(("L:", "% lengths")) for row in hook))
        self.assertIn("K:C", hook)
        rows = [row for row in hook if not row.startswith(("%", "w:", "K:", "MELODY", "HOOK", "rhythm", "  "))]
        self.assertTrue(rows)
        for row in rows:
            self.assertRegex(row, r"^[=^_A-Ga-g,' ]+$")  # no digits, no z, no bars
        # The long line is split into matching note/w: pairs, nothing dropped.
        pairs = [(hook[i - 1], row) for i, row in enumerate(hook) if row.startswith("w: ")]
        self.assertGreater(len(pairs), 2)
        for notes, sung in pairs:
            self.assertEqual(len(notes.split()), len(sung[3:].split()))
        sung = " ".join(w[3:] for _, w in pairs).replace("~", " ").replace("_ ", "")
        for word in lines[0].split() + lines[1].split():
            self.assertIn(word, sung)

    def test_whole_view_prints_the_writers_words_when_supplied_and_whispers_otherwise(self):
        heard = [w.replace("harbours", "harbors").replace("tonight", "to night") for w in
                 " ".join(["Lanterns glow above invented harbours while distant bells ring over silver water tonight softly and slowly while every lantern remembers",
                           "Bring it home"] * 2).split()]
        heard = " ".join(heard).split()
        for label in ("LYRICS: writer's text, 4 of 4 lines heard",
                      "LYRICS: LRCLIB synced lyrics; supplied times (≈)",
                      "LYRICS: LRCLIB had no synced lyrics; plain lyrics aligned to the vocals"):
            with self.subTest(label=label):
                song, lines = self.long_song(label, heard)
                result = M.build(song)
                self.assertEqual(result["status"], "ran", result["reason"])
                whole = result["text"]
                self.assertIn("% [0:00] " + lines[0], whole)
                self.assertIn("harbours", whole)
                self.assertNotIn("harbors", whole)
                self.assertNotIn("to~night", whole)
                self.assertIn("harbours", result["json"]["hook_text"])
        song, lines = self.long_song("LYRICS: LRCLIB lookup off; words are Whisper's guesses (rough times)", heard)
        whole = M.build(song)["text"]
        self.assertIn("harbors", whole)
        self.assertNotIn("harbours", whole)
        self.assertNotIn("% [0:00] ", whole)

    def test_saved_line_only_lyrics_render_in_both_views_without_a_new_listen(self):
        import melody_abc
        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call("--title", "Kites")
        song = self.song()
        singer = song["cards"]["singer"]["json"]
        texts = ["Silver kites rise", "Soft clouds cross green hills", "Night wind turns"]
        rows = [dict(t=t, end=end, text=text, approx=True, timing="LRCLIB supplied")
                for t, end, text in zip((.25, 1, 2), (1, 2, 3), texts)]
        singer["lyrics"] = dict(rows=rows, label="LYRICS: LRCLIB synced lyrics; supplied times (≈)")
        singer["melody_input"].update(lyric_lines=texts, lyrics_label=singer["lyrics"]["label"],
                                      words=[["hmm", .8, .95], ["Night", 2.25, 2.5],
                                             ["wind", 2.5, 2.75], ["turns", 2.75, 3]])
        transcribe = melody_abc.transcribe

        def old_transcribe(*args, **kwargs):
            kwargs.pop("line_spans", None)
            return transcribe(*args, **kwargs)

        with patch.object(melody_abc, "transcribe", side_effect=old_transcribe):
            song["melody"] = M.build(song)
        old = song["melody"]["json"]
        self.assertIn("Silver~kites~rise", old["lines"][0]["lyrics"])
        del old["notation"]["line_lyrics"]
        L.save(self.path(), song)
        before = self.path().read_bytes()
        self.calls.clear()

        first, _ = self.call("--title", "Kites")
        whole, _ = self.call("--title", "Kites", "--view", "melody")
        self.assertEqual(self.calls, [])
        self.assertEqual(before, self.path().read_bytes())
        self.assertIn("\nw: * Silver kites rise\n", first)
        self.assertIn("Lyrics could not be aligned to notes: " + texts[1], first)
        self.assertNotIn("Silver~", first)
        self.assertIn("\nw: Silver kites rise\n", whole)
        self.assertIn("Lyrics could not be aligned to notes (too few stable note onsets): " + texts[1], whole)
        self.assertNotIn("\nw:", first.split(texts[1], 1)[1].split("\n\n", 1)[0])
        self.assertIn("\nw: * Night wind turns", whole)

        singer["melody_input"]["words"] = []
        result = M.build(song)
        self.assertEqual(result["status"], "ran", result["reason"])
        self.assertEqual(result["json"]["lines"][0]["lyrics"], "* Silver kites rise")
        self.assertEqual(result["json"]["lines"][1]["lyric_fit"], "unaligned")
        self.assertTrue(result["json"]["lines"][1]["music"])
        self.assertEqual(result["json"]["lines"][1]["lyrics"], "")

    def test_rebuilt_hook_omits_low_dip_but_full_melody_keeps_it(self):
        from test_melody_abc import track

        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call("--title", "Lantern")
        song = self.song()
        song["melody"]["json"]["notation"]["pitch_revision"] = M.PITCH_REVISION - 1
        singer = song["cards"]["singer"]["json"]
        line = "The paper lantern by the window glows all night"
        singer["lyrics"] = {"rows": [dict(t=.25, end=4., text=line, approx=True,
                                           timing="LRCLIB supplied")]}
        f0 = track(*[(pitch, .5) for pitch in (64, 65, 67, 48, 65, 67, 65, 64)])
        singer["melody_input"].update(f0=f0.tolist(), voiced_prob=[.95] * len(f0),
                                      lyric_lines=[line], words=[])

        first, _ = L.render(song, "first-listen")
        self.assertIn(f"% [0:00≈] Lyrics could not be aligned to notes: {line}\n"
                      "% 1 low dip left out\nE2 F2 G2 F2 G2 F2 E2\n", first)
        whole, _ = L.render(song, "melody")
        self.assertIn("\nE F2 G2 C,2 F2 G2 F2 E2\n", whole)
        self.assertNotIn("low dip left out", whole)

    def test_sparse_hooks_fall_through_title_repetition_then_intensity(self):
        texts = ["Is Embers glowing", "Walk on", "Walk on", "Peak phrase"]
        rows = [dict(start=i, end=i + 1, note_indices=list(range(n)))
                for i, n in enumerate((3, 4, 4, 4))]
        peak = dict(ok=True, start=3.2, end=3.8)
        indices, rule = M.hook_indices(texts, rows, peak, ["embers"], min_notes=4)
        self.assertEqual(indices, [1, 2])
        self.assertIn("most-repeated", rule)
        for i in (1, 2):
            rows[i]["note_indices"] = [0, 1, 2]
        indices, rule = M.hook_indices(texts, rows, peak, ["embers"], min_notes=4)
        self.assertEqual(indices, [3])
        self.assertIn("most intense moment", rule)
        rows[3]["note_indices"] = [0, 1, 2]
        indices, rule = M.hook_indices(texts, rows, peak, ["embers"], min_notes=4)
        self.assertEqual(indices, [])
        self.assertIn("at least 4 rendered notes", rule)

    def test_hook_keeps_first_qualifying_pair_despite_missing_words(self):
        texts = ["Is Embers glowing", "Walk on", "Is Embers glowing", "Walk on"]
        rows = [dict(start=i, end=i + 1, note_indices=list(range(4)),
                     lyrics="Embers" if i % 2 == 0 else "on") for i in range(4)]
        self.assertEqual(M.hook_indices(texts, rows, {}, ["embers"], min_notes=4)[0], [0, 1])
        rows[1]["note_indices"] = [0, 1, 2]
        indices, rule = M.hook_indices(texts, rows, {}, ["embers"], min_notes=4)
        self.assertEqual(indices, [0])
        self.assertIn("next line has fewer than 4 rendered notes", rule)

    def test_hook_uses_first_aligned_repetition_and_keeps_original_fallback(self):
        texts = ["Embers carry the light through that long winter", "Rain rolls in",
                 "Embers carry the light through that long summer", "Dawn rolls in",
                 "Embers carry the light through that long winter"]
        rows = [dict(start=i, end=i + 1, note_indices=list(range(7)), lyrics=text)
                for i, text in enumerate(texts)]
        rows[0].update(lyric_fit="unaligned", lyrics="")
        rows[2]["lyric_fit"] = "onset_order"
        self.assertEqual(M.hook_indices(texts, rows, {}, ["embers"], min_notes=4)[0], [2, 3])
        self.assertEqual(M.hook_indices(texts, rows, {}, min_notes=4)[0], [4])
        for i in (2, 4):
            rows[i].update(lyric_fit="unaligned", lyrics="")
        self.assertEqual(M.hook_indices(texts, rows, {}, ["embers"], min_notes=4)[0], [0, 1])

    def test_parenthetical_title_response_falls_back_to_repeated_chorus(self):
        texts = ["Air grows cold (I'm feeling embers)", "Come inside (I'm feeling embers)",
                 "Carry the lantern home", "Leave the candle here",
                 "Carry the lantern home", "Leave the candle here"]
        rows = [dict(start=i, end=i + 1, note_indices=list(range(4))) for i in range(6)]
        indices, rule = M.hook_indices(texts, rows, {}, ["feeling", "embers"], min_notes=4)
        self.assertEqual(indices, [2, 3])
        self.assertIn("most-repeated", rule)

    def test_hook_never_selects_empty_or_wordless_rows(self):
        texts = ["", "Is Embers glowing", "Carry it home", "Carry it home"]
        rows = [dict(start=i, end=i + 1, note_indices=list(range(8)), lyrics=lyric)
                for i, lyric in enumerate(("", "* _ * _", "Carry~it home", "Carry~it home"))]
        indices, _ = M.hook_indices(texts, rows, {}, ["embers"], min_notes=4)
        self.assertEqual(indices, [2, 3])
        rows[3]["lyrics"] = "* _ * _"
        indices, rule = M.hook_indices(texts, rows, {}, ["carry"], min_notes=4)
        self.assertEqual(indices, [2])
        self.assertIn("next line has no rendered lyric words", rule)

    def test_hook_declines_notes_with_no_supplied_lyric_words_under_them(self):
        song, _ = self.long_song("LYRICS: LRCLIB synced lyrics; supplied times (≈)")
        for name, saved in (("fake", song), ("real-shaped", L.load(L.HERE / "fixtures/uhuru-title-song.json"))):
            with self.subTest(name=name):
                words = saved["cards"]["singer"]["json"]["melody_input"]["words"]
                with patch.object(M, "writer_word_text", return_value={i: "" for i in range(len(words))}):
                    result = M.build(saved)
                self.assertEqual(result["status"], "ran", result["reason"])
                self.assertGreaterEqual(max(len(line["note_indices"]) for line in result["json"]["lines"]), 4)
                self.assertTrue(result["json"]["hook_text"].startswith("no hook: "))
                self.assertIn("no sung lyric words", result["json"]["hook_text"])

    def test_synced_real_shaped_song_retries_sparse_title_then_declines_empty_pitch(self):
        # Saved-shaped CPU inputs, using the existing invented lyric fixture.
        song = L.load(L.HERE / "fixtures/uhuru-title-song.json")
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture["lyrics_label"] = "LYRICS: LRCLIB synced lyrics; supplied times (≈)"
        before = M.build(song)
        self.assertEqual(before["status"], "ran", before["reason"])
        first = before["json"]["hook"]["line_indices"][0]
        cuts = [0] + before["json"]["settings"]["line_breaks"] + [len(capture["f0"]) / capture["fps"]]
        a, b = [int(t * capture["fps"]) for t in cuts[first:first + 2]]
        capture["f0"][a:b] = [None] * (b - a)
        after = M.build(song)
        self.assertEqual(after["status"], "ran", after["reason"])
        indices = after["json"]["hook"]["line_indices"]
        self.assertTrue(indices)
        self.assertGreater(indices[0], first)
        self.assertIn('line with the title "uhuru"', after["json"]["hook_text"])
        self.assertTrue(all(len(after["json"]["lines"][i]["note_indices"]) >= 4 for i in indices))
        capture["f0"] = [None] * len(capture["f0"])
        declined = M.build(song)
        self.assertEqual(declined["status"], "ran", declined["reason"])
        self.assertTrue(declined["json"]["hook_text"].startswith("no hook: "))
        self.assertEqual(len(declined["json"]["hook_text"].splitlines()), 1)

    def test_real_shaped_hook_keeps_first_pair_and_discloses_unpaired_words(self):
        song = L.load(L.HERE / "fixtures/uhuru-title-song.json")
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture["lyrics_label"] = "LYRICS: LRCLIB synced lyrics; supplied times (≈)"
        before = M.build(song)["json"]
        first, second = before["hook"]["line_indices"]
        # Remove the measured pitches beneath one heard word, keeping the
        # remaining contour. Missing words must not force a later chorus.
        cut = before["settings"]["line_breaks"][first - 1]
        word = next(w for w in capture["words"] if w[1] >= cut)
        a, b = int(word[1] * capture["fps"]), int(word[2] * capture["fps"])
        capture["f0"][a:b] = [None] * (b - a)
        after = M.build(song)["json"]
        self.assertEqual(after["hook"]["line_indices"], [first, second])
        self.assertIn("words without a rendered note:", after["hook_text"])
        cuts = before["settings"]["line_breaks"]
        a, b = [int(t * capture["fps"]) for t in cuts[second - 1:second + 1]]
        capture["f0"][a:b] = [None] * (b - a)
        single = M.build(song)["json"]
        self.assertEqual(single["hook"]["line_indices"], [first])
        self.assertIn("next line has fewer than 4 rendered notes", single["hook_text"])

    def test_real_shaped_parenthetical_title_uses_repetition_then_peak(self):
        song = L.load(L.HERE / "fixtures/uhuru-title-song.json")
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture["lyric_lines"] = [text.replace("Uhuru", "(Uhuru)") for text in capture["lyric_lines"]]
        repeated = M.build(song)["json"]
        self.assertIn("most-repeated", repeated["hook"]["rule"])
        # Silence repeated rows so the same saved input has to reach intensity.
        texts = [r["text"] for r in repeated["lyric_alignment"]["rows"]]
        normalized = [" ".join(M.words_of(text)) for text in texts]
        cuts = [0] + repeated["settings"]["line_breaks"] + [len(capture["f0"]) / capture["fps"]]
        for i, text in enumerate(normalized):
            if normalized.count(text) > 1:
                a, b = [int(t * capture["fps"]) for t in cuts[i:i + 2]]
                capture["f0"][a:b] = [None] * (b - a)
        unique = next(i for i, text in enumerate(normalized)
                      if normalized.count(text) == 1 and len(repeated["lines"][i]["note_indices"]) >= 4)
        row = repeated["lyric_alignment"]["rows"][unique]
        song["cards"]["singer"]["json"]["singer"]["data"]["peak"] = dict(
            ok=True, start=row["t"] + .5, end=row["t"] + 1)
        intense = M.build(song)["json"]
        self.assertEqual(intense["hook"]["line_indices"][0], unique)
        self.assertIn("most intense moment", intense["hook"]["rule"])

    def test_real_shaped_whisper_hook_comments_always_contain_words(self):
        song = L.load(L.HERE / "fixtures/uhuru-title-song.json")
        capture = song["cards"]["singer"]["json"]["melody_input"]
        capture["lyrics_label"] = "LYRICS: words are Whisper's guesses (rough times)"
        result = M.build(song)
        self.assertEqual(result["status"], "ran", result["reason"])
        hook = result["json"]["hook_text"]
        comments = [r for r in hook.splitlines() if r.startswith("% [")]
        self.assertTrue(comments, hook)
        for comment in comments:
            self.assertTrue(M.words_of(comment.partition("]")[2]), comment)
        for i in result["json"]["hook"]["line_indices"]:
            self.assertTrue(M.words_of(result["json"]["lines"][i]["lyrics"]))

    def test_writer_word_text_pairs_inside_each_line(self):
        cuts = [2.0]
        words = [["Red", 0, 1], ["bem", 1, 2], ["extra", 2.1, 2.5], ["go", 2.5, 3], ["on", 3, 4]]
        texts = ["Red beam", "go on again"]
        self.assertEqual(M.writer_word_text(texts, words, cuts),
                         {0: "Red", 1: "beam", 2: "", 3: "go", 4: "on again"})

    def test_harmony_retry_does_not_recompute_saved_melody(self):
        with patch.object(L, "run_pass", side_effect=self.notation_pass):
            self.call()
        song = self.song()
        song["manifest"]["passes"]["harmony"].update(status="failed", reason="invented failure")
        L.save(self.path(), song)
        with patch.object(M, "build", side_effect=AssertionError("melody already saved")):
            self.call("--view", "harmony", "--run-missing")

    def test_singer_capture_reuses_one_front_end_and_serializes_unvoiced_frames(self):
        import singer
        calls = []
        def front_end(*args, **kwargs):
            calls.append("front end")
            return {"f0": [float("nan"), 261.625565], "voiced_prob": [float("nan"), .95]}
        def analyze(*args, **kwargs):
            fake._front_end([])
            return {"data": {}, "timings": {}}
        fake = SimpleNamespace(_front_end=front_end, analyze_affect=lambda: None,
                               analyze=analyze, FPS=62.5, release_models=lambda: None)
        captured = {}
        with patch.dict(sys.modules, {"raven_ears_plus": fake}), patch.object(singer, "_free_gpu"):
            singer.singer_block("invented-vocals.wav", 1, [("Lantern", 0, .1)],
                                capture_pitch=captured.update)
        self.assertEqual(calls, ["front end"])
        self.assertEqual(captured, {"f0": [None, 261.625565], "fps": 62.5, "voiced_prob": [None, .95]})
        json.dumps(captured, allow_nan=False)

    def test_real_saved_melodies_render_from_copied_bundles_without_models(self):
        # The committed real-shaped fixture, saved with its derived melody into a temporary home.
        song = L.validate_song(json.loads((L.HERE / "fixtures" / "uhuru-title-song.json").read_text(encoding="utf-8")))
        song["melody"] = M.build(song)
        L.save(L.store_home() / song["manifest"]["file_sha256"] / "song.json", song)
        bundles = list(L.store_home().glob("*/song.json"))
        self.assertTrue(bundles)
        for path in bundles:
            with self.subTest(bundle=path.parent.name):
                before = path.read_bytes(), path.stat().st_mtime_ns
                with patch.dict(sys.modules, {"singer": None, "music": None, "harmony": None, "torch": None}), \
                     patch.object(L.subprocess, "run", side_effect=AssertionError("no models")):
                    song = L.load(path)
                    first, _ = L.render(song, "first-listen")
                    whole, _ = L.render(song, "melody")
                self.assertEqual(song["melody"]["status"], "ran")
                self.assertIn(song["melody"]["json"]["rhythm_line"], whole)
                self.assertEqual(whole.count("rhythm:"), 1)
                self.assertEqual(first.count("\nw: "), 2)
                self.assertGreater(whole.count("\nw: "), 2)
                self.assertIn("THE SONG", first)
                self.assertIn("[1:04] da Uhuru:", first)
                self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))

    def test_first_listen_saves_text_json_manifest_and_second_calls_none(self):
        first, notices = self.call("--view", "full")
        self.assertEqual(self.calls, ["singer", "music", "harmony"])
        self.assertIn("Listening now", " ".join(notices))
        song = self.song()
        manifest = song["manifest"]
        self.assertEqual(manifest["duration_s"], 60)
        self.assertEqual(manifest["file_sha256"], L.file_hash(self.audio))
        self.assertEqual(manifest["ears_version"], "test-version-1")
        self.assertTrue(manifest["heard_at"])
        self.assertTrue(all(row["status"] == "ran" for row in manifest["passes"].values()))
        self.assertTrue(all(card["json"] for card in song["cards"].values()))
        before = self.path().read_bytes()
        self.calls.clear()
        second, notices = self.call("--view", "full")
        self.assertEqual(self.calls, [])
        self.assertEqual(first, second)
        self.assertIn("from the saved card, heard", " ".join(notices))
        self.assertEqual(self.path().read_bytes(), before)

    def test_saved_feeling_header_names_the_displayed_words(self):
        self.call()
        fallback = "LRCLIB had no matching lyrics; words are Whisper's guesses (rough times)"
        guessed = "THE SONG (no published lyrics found: words are Whisper's guesses, often wrong)"
        published = "THE SONG (lyrics from LRCLIB, times ≈)"
        supplied = "THE SONG (lyrics supplied, times ≈)"
        for status, label, expected in (
            ("nomatch", fallback, guessed),
            ("failed", "LRCLIB synced lyrics; supplied times (≈)", guessed),
            ("file", fallback, supplied),
            ("synced", fallback, published),
            ("plain", fallback, published),
            (None, fallback, guessed),
            (None, "writer's text, 1 of 1 lines heard", supplied),
            (None, "LRCLIB synced lyrics; supplied times (≈)", published),
            (None, "LRCLIB had no synced lyrics; plain lyrics aligned to the vocals", published),
            ("nomatch", "LRCLIB had no matching lyrics; no Whisper words available", "THE SONG"),
        ):
            with self.subTest(status=status, label=label):
                song = self.song()
                card = song["cards"]["music"]
                card["text"] = f"MUSIC: Lantern 1:00\nLYRICS: {label}\n"
                card["json"] = dict(duration=60, stage_status=[])
                if expected != "THE SONG":
                    card["text"] += "  [0:04] An invented lantern\n"
                    card["json"]["lyrics"] = [dict(t=4.0, end=5.0, text="An invented lantern", approx=False)]
                if status:
                    card["json"]["lyrics_source"] = dict(source="file" if status == "file" else "LRCLIB",
                                                         status=status)
                L.save(self.path(), song)
                original = self.path().read_bytes()
                self.calls.clear()
                text, _ = self.call()
                self.assertEqual(next(line for line in text.splitlines() if line.startswith("THE SONG")), expected)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.path().read_bytes(), original)

    def test_fresh_calls_all_again_and_replaces(self):
        self.call("--view", "full")
        _, notices = self.call("--fresh", "--view", "full")
        self.assertEqual(self.calls, list(L.PASSES) * 2)
        self.assertNotIn("from the saved card", " ".join(notices))

    def test_content_hash_reuses_same_bytes_at_another_path(self):
        self.call("--view", "full")
        moved = self.root / "other" / self.audio.name
        moved.parent.mkdir()
        moved.write_bytes(self.audio.read_bytes())
        self.calls.clear()
        _, notices = L.execute(L.parser().parse_args([str(moved), "--view", "singer"]))
        self.assertEqual(self.calls, [])
        self.assertIn("from the saved card", " ".join(notices))

    def test_changed_audio_makes_another_song(self):
        self.call("--view", "full")
        self.audio.write_bytes(b"different invented bytes")
        self.call("--view", "full")
        self.assertEqual(self.calls, list(L.PASSES) * 2)
        self.assertEqual(len(list(L.store_home().glob("*/song.json"))), 2)

    def test_supplied_lyrics_refresh_saved_song_and_identical_content_reuses_it(self):
        lyrics = self.root / "words.txt"
        lyrics.write_text("An invented lantern\n")
        self.call("--view", "full")
        self.calls.clear()
        notice = "re-listening: --lyrics differs from the saved card"
        def relisten(name, args):
            self.assertEqual(err.getvalue().splitlines(), [notice])
            self.assertEqual(args.lyrics, str(lyrics))
            return self.fake_pass(name, args)
        with patch.object(L, "home_context", return_value=nullcontext()), \
             patch.object(L, "run_pass", side_effect=relisten), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(L.main([str(self.audio), "--lyrics", str(lyrics), "--view", "full"]), 0)
        self.assertEqual(err.getvalue().splitlines().count(notice), 1)
        self.assertEqual(self.calls, list(L.PASSES))
        self.assertEqual(self.song()["manifest"]["options"]["lyrics_sha256"], L.file_hash(lyrics))
        self.assertTrue(all(row["listen_count"] == 2 for row in self.song()["manifest"]["passes"].values()))
        before = self.path().read_bytes()
        copy = self.root / "copy.txt"
        copy.write_bytes(lyrics.read_bytes())
        self.calls.clear()
        self.call("--lyrics", str(copy), "--view", "full")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.path().read_bytes(), before)
        lyrics.write_text("Another invented lantern\n")
        _, notices = self.call("--lyrics", str(lyrics), "--view", "full", "--run-missing")
        self.assertIn(notice, notices)
        self.assertEqual(self.calls, list(L.PASSES))
        self.assertEqual(self.song()["manifest"]["options"]["lyrics_sha256"], L.file_hash(lyrics))

    def test_changed_explicit_analysis_inputs_are_forwarded_on_refresh(self):
        prompt = self.root / "prompt.txt"
        prompt.write_text("Invented lantern\n")
        self.call()
        for flag, value, passes in (
            ("--artist", "Invented Artist", L.PASSES),
            ("--title", "Lantern", L.PASSES),
            ("--album", "Invented Album", L.PASSES),
            ("--names", str(prompt), ("music", "harmony")),
            ("--no-lyrics-lookup", None, L.PASSES),
        ):
            with self.subTest(flag=flag):
                self.calls.clear()
                L.run_pass.reset_mock()
                _, notices = self.call(flag, *([value] if value else []))
                self.assertIn(f"re-listening: {flag} differs from the saved card", notices)
                self.assertEqual(self.calls, list(L.PASSES))
                for call in L.run_pass.call_args_list:
                    name, args = call.args
                    self.assertEqual(flag in L.pass_args(name, args), name in passes)
                    if value:
                        self.assertEqual(getattr(args, flag[2:]), value)

    def test_read_only_and_partial_modes_refuse_a_conflicting_refresh(self):
        self.call()
        self.calls.clear()
        original = self.path().read_bytes()
        for flag in ("--only", "--lore"):
            with self.subTest(flag=flag), self.assertRaisesRegex(
                    ValueError, rf"--title differs from the saved card.*--fresh without {flag}"):
                self.call("--title", "Lantern", flag, *(["music"] if flag == "--only" else []))
        self.assertEqual(self.calls, [])
        self.assertEqual(original, self.path().read_bytes())

    def test_different_version_is_explicit_and_never_triggers_work(self):
        self.call("--only", "music", "--view", "music")
        before = self.path().read_bytes()
        self.calls.clear()
        with patch.object(L, "ears_version", return_value="test-version-2"):
            text, notices = self.call("--view", "singer", "--run-missing")
            self.assertIn("version differs", " ".join(notices))
            self.assertIn("--fresh", " ".join(notices))
            self.assertIn("not run", text)
            self.assertEqual(self.calls, [])
            self.assertEqual(self.path().read_bytes(), before)
            self.call("--fresh", "--view", "full")
        self.assertEqual(self.song()["manifest"]["ears_version"], "test-version-2")

    def test_failed_pass_is_saved_shown_and_only_explicitly_retried(self):
        def fail(name, args):
            if name == "singer":
                self.calls.append(name)
                raise OSError("invented missing backend")
            return self.fake_pass(name, args)
        with patch.object(L, "run_pass", side_effect=fail):
            text, _ = self.call("--view", "full")
        self.assertEqual(self.calls, list(L.PASSES))
        self.assertIn("singer: failed: OSError: invented missing backend", text)
        self.assertEqual(self.song()["manifest"]["passes"]["singer"]["status"], "failed")
        self.calls.clear()
        text, _ = self.call("--view", "singer")
        self.assertIn("invented missing backend", text)
        self.assertEqual(self.calls, [])
        text, notices = self.call("--view", "singer", "--run-missing")
        self.assertEqual(self.calls, ["singer"])
        self.assertIn("invented soft voice", text)
        self.assertIn("other cards remain saved", " ".join(notices))

    def test_missing_pass_does_no_work_until_run_missing(self):
        self.call("--only", "music", "--view", "music")
        self.calls.clear()
        text, _ = self.call("--view", "harmony")
        self.assertIn("harmony: not run: not selected", text)
        self.assertEqual(self.calls, [])
        self.call("--view", "harmony", "--run-missing")
        self.assertEqual(self.calls, ["harmony"])
        self.assertEqual(self.song()["manifest"]["passes"]["singer"]["status"], "not run")
        self.calls.clear()
        self.call("--view", "full", "--run-missing")
        self.assertEqual(self.calls, ["singer"])

    def test_partial_failed_stdout_and_json_are_retained(self):
        def fail(name, args):
            return {"status": "failed", "reason": "exit 1: invented error", "text": "partial evidence\n",
                    "json": {"stage_status": [{"stage": name, "status": "failed"}]}, "diagnostics": "invented error"}
        with patch.object(L, "run_pass", side_effect=fail):
            self.call("--view", "full")
        text, _ = self.call("--view", "singer")
        self.assertTrue(text.startswith("singer: failed: exit 1"))
        self.assertIn("partial evidence", text)
        self.assertEqual(self.song()["cards"]["singer"]["diagnostics"], "invented error")
        self.assertIsNone(self.song()["manifest"]["duration_s"])
        self.assertTrue(self.song()["manifest"]["duration_reason"])

    def test_interrupted_pass_leaves_explicit_not_run_checkpoint(self):
        def interrupt(name, args):
            if name == "music":
                raise KeyboardInterrupt()
            return self.fake_pass(name, args)
        with patch.object(L, "run_pass", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.call("--view", "full")
        self.assertEqual(self.song()["manifest"]["passes"]["singer"]["status"], "ran")
        self.assertEqual(self.song()["manifest"]["passes"]["music"]["status"], "not run")

    def test_atomic_save_failure_preserves_old_bundle(self):
        self.call("--view", "full")
        before = self.song()
        self.calls.clear()
        replace = L.os.replace
        writes = []
        def fail_publication(source, destination):
            writes.append(destination)
            if len(writes) == 4:  # three counter checkpoints, then replacement publication
                raise OSError("interrupted final save")
            replace(source, destination)
        with patch.object(L.os, "replace", side_effect=fail_publication):
            with self.assertRaisesRegex(OSError, "interrupted final save"):
                self.call("--fresh", "--view", "full")
        self.assertEqual(self.song()["cards"], before["cards"])
        self.assertEqual(self.song()["manifest"]["heard_at"], before["manifest"]["heard_at"])
        self.assertTrue(all(row["listen_count"] == 2 for row in self.song()["manifest"]["passes"].values()))
        self.assertEqual(list(self.path().parent.glob(".song-*")), [])
        self.assertEqual(len(writes), 4)
        self.assertEqual(self.calls, list(L.PASSES))
        self.assertIn("An invented lantern", self.stdout())

    def test_corruption_is_reported_and_not_silently_recomputed(self):
        self.call("--view", "full")
        self.path().write_text("{broken")
        self.calls.clear()
        with self.assertRaises(ValueError):
            self.call("--view", "full")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.path().read_text(), "{broken")
        self.call("--fresh", "--view", "full")
        self.assertEqual(self.calls, list(L.PASSES))

    def test_delete_one_and_all_preserve_other_engine_caches(self):
        self.assertIn("An invented lantern", self.stdout())
        first = self.path()
        other_cache = self.root / "ears" / "lyrics-cache.json"
        other_cache.write_text("invented other cache")
        sibling = first.parent / "keep.txt"
        sibling.write_text("unrelated sibling")
        unrelated = L.store_home() / "not-a-bundle"
        unrelated.mkdir()
        (unrelated / "song.json").write_text('{"unrelated": true}')
        (unrelated / "keep.txt").write_text("unrelated document")
        invalid = L.store_home() / ("a" * 64)
        invalid.mkdir()
        (invalid / "song.json").write_text("{broken")
        outside = self.root / "outside-store"
        outside.mkdir()
        (outside / "song.json").write_bytes(first.read_bytes())
        linked = L.store_home() / ("b" * 64)
        linked.symlink_to(outside, target_is_directory=True)
        linked_file = L.store_home() / ("c" * 64)
        linked_file.mkdir()
        (linked_file / "song.json").symlink_to(first)
        self.audio.write_bytes(b"another song")
        self.call("--view", "full")
        second = self.path()
        self.calls.clear()
        self.assertIn("Deleted saved song", self.stdout("--delete"))
        self.assertTrue(first.exists())
        self.assertFalse(second.exists())
        text, _ = L.execute(L.parser().parse_args(["--delete-all"]))
        self.assertIn("Unrelated or unsafe entries left intact", text)
        self.assertFalse(first.exists())
        self.assertTrue(other_cache.exists())
        self.assertTrue(sibling.exists())
        self.assertTrue((unrelated / "song.json").exists())
        self.assertTrue((unrelated / "keep.txt").exists())
        self.assertTrue((invalid / "song.json").exists())
        self.assertTrue(linked.is_symlink())
        self.assertTrue((outside / "song.json").exists())
        self.assertTrue((linked_file / "song.json").is_symlink())
        self.assertEqual(self.calls, [])

    def test_delete_refuses_symlinks_malformed_mismatched_and_missing_bundles(self):
        self.assertIn("An invented lantern", self.stdout())
        path = self.path()
        before = path.read_bytes()
        sibling = path.parent / "keep.txt"
        sibling.write_text("unrelated")
        for unsafe in ("malformed", "mismatched", "file symlink", "directory symlink", "missing"):
            with self.subTest(unsafe=unsafe):
                path.write_bytes(before)
                if unsafe == "malformed":
                    path.write_text("{unrelated")
                elif unsafe == "mismatched":
                    song = self.song()
                    song["manifest"]["file_sha256"] = "d" * 64
                    L.save(path, song)
                elif unsafe == "file symlink":
                    target = self.root / "unrelated.json"
                    target.write_bytes(before)
                    path.unlink()
                    path.symlink_to(target)
                elif unsafe == "directory symlink":
                    actual = self.root / "actual-bundle"
                    path.parent.rename(actual)
                    path.parent.symlink_to(actual, target_is_directory=True)
                elif unsafe == "missing":
                    path.unlink()
                if unsafe == "missing":
                    self.stdout("--delete")
                else:
                    with self.assertRaises(ValueError):
                        self.call("--delete")
                self.assertTrue(sibling.exists())
                if unsafe == "file symlink":
                    self.assertEqual(target.read_bytes(), before)
                    path.unlink()
                if unsafe == "directory symlink":
                    self.assertEqual((actual / "song.json").read_bytes(), before)
                    path.parent.unlink()
                    actual.rename(path.parent)
        path.write_bytes(before)
        self.assertIn("An invented lantern", self.stdout())
        self.stdout("--delete")
        self.assertTrue(sibling.exists())
        self.assertFalse(path.exists())

    def stdout(self, *flags):
        # Exercise main's stdout while retaining this test's disposable home.
        with patch.object(L, "home_context", return_value=nullcontext()), \
             redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            self.assertEqual(L.main([str(self.audio), *flags]), 0)
        return out.getvalue()

    def test_join_is_in_process_read_only_and_receives_authoritative_status(self):
        self.call("--view", "full")
        before = self.path().read_bytes()
        with patch.object(L.subprocess, "run", side_effect=AssertionError("no renderer subprocess")), \
             patch.object(L.tempfile, "TemporaryDirectory", side_effect=AssertionError("no render scratch")):
            text = self.stdout()
        self.assertIn("FEELING CARD:", text)
        self.assertIn("An invented lantern", text)
        self.assertEqual(self.path().read_bytes(), before)

    def test_saved_stop_views_refresh_old_and_new_cards_without_touching_json(self):
        import stops
        # Invented rows following saved Take Five/SFF event shapes. Include
        # vocal qualifiers and a faded stop so rendering cannot lose either.
        event = dict(start_s=229.72, length_s=.89, drop_db=17.7, kind="band break",
                     fall_to_db=13.4, fall_s=.66, fall_db_per_s=15.7,
                     first_exit_stem="drums", band_start_s=229.91, drop_floor_limited=True,
                     voice_break=dict(drop_db=20, length_s=.4, start_s=229, return_s=229.4),
                     voice_after=dict(from_drop_db=3, to_drop_db=18, start_s=230, end_s=231))
        faded = dict(kind="faded stop", start_s=214.78, length_s=2.64, low_length_s=1.74,
                     return_s=217.42, drop_db=17.6, fade_start_s=212.93, fall_s=3.16, fade_ramp_r2=.882)
        result = dict(status="ran", events=[dict(event, start_s=34.19, kind="uncertain",
                                                reason="drums fell only 4.4 dB; sustained layer or bleed"),
                                            event, faded], recurrence=[])
        self.call("--view", "full")
        song = self.song()
        song["cards"]["singer"]["json"]["stops"] = result
        detail = "\n".join(stops.card_lines(stops.refresh(result), detail=True))
        full = "\n".join(stops.card_lines(stops.refresh(result)))
        for saved in (detail, full):
            song["cards"]["singer"]["text"] = "SINGER: invented\n" + saved + "\nMETER: invented\n"
            L.save(self.path(), song)
            before = self.path().read_bytes()
            self.calls.clear()
            for view, expected in (("full", full), ("singer", full), ("detail", detail)):
                text, _ = self.call("--view", view)
                block = re.search(r"(?m)^STOPS:[^\n]*(?:\n[ \t]+[^\n]*)*", text)[0]
                self.assertEqual(block, expected)
                self.assertIn("METER: invented", text)
                self.assertIn("voice breaks:", block)
                self.assertIn("voice then fades", block)
                self.assertIn("faded stop", block)
            raw, _ = self.call("--view", "detail", "--json")
            self.assertEqual(json.loads(raw)["cards"]["singer"]["json"]["stops"], result)
            self.assertEqual(self.path().read_bytes(), before)
            self.assertEqual(self.calls, [])

    def test_saved_stop_curves_refresh_every_view_without_mutating_evidence(self):
        import stops
        self.call("--view", "full")
        song = self.song()
        event = dict(start_s=9.03, length_s=1.95, drop_db=38.4, kind=stops.ENTRY_KIND,
                     fall_to_db=36, fall_s=.45, fall_db_per_s=73.3)
        # Mock derivation: this tests persistence/rendering, not audio thresholds.
        curves = {s: [1.234567891e-9, 2.345678912e-8] for s in stops.STEMS}
        result = dict(status="ran", events=[], recurrence=[], curves=stops._pack_curves(curves))
        song["cards"]["singer"].update(json={"stops": result},
                                       text="SINGER: invented\nSTOPS: none found\nMETER: preserved\n")
        L.save(self.path(), song)
        before = self.path().read_bytes()
        with patch.object(stops, "detect", return_value=[event]) as detect, \
             patch.object(L.subprocess, "run", side_effect=AssertionError("no models")):
            for view in ("full", "detail", "singer", "summary"):
                text, _ = self.call("--view", view, "--no-picture")
                self.assertIn("band cuts, vocal stem enters: 1", text)
                self.assertIn("0:09.03", text)
                self.assertIn("singing/pitch not checked", text)
            self.assertEqual(detect.call_count, 6)
            for call in detect.call_args_list:
                for stem in stops.STEMS:
                    self.assertEqual(call.args[0][stem].tolist(), curves[stem])
        self.assertEqual(self.path().read_bytes(), before)

    def test_saved_instrumental_views_keep_bundle_bytes_and_raw_measurements(self):
        import instrumental_voice as IV
        import copy
        import compare
        import slim
        self.call('--view', 'full')
        song = self.song()
        source = dict(source='LRCLIB', status='instrumental')
        label = 'LYRICS: instrumental (LRCLIB); database designation, not a measurement of silence'
        song['cards']['music'].update(
            text=('MUSIC: invented.wav · 1:00\n'
                  '  VOCALS: yes, voice present ~18% of the track (stem holds 11% of the energy) '
                  '[method: vocal-stem energy] · 0:23–0:28, 0:58–0:59\n'
                  '  ' + label + '\n'
                  '  NOTE:     0:00–1:00: voice present, words unavailable (transcription did not run or failed)\n'),
            json=dict(duration=60, lyrics_source=source))
        song['cards']['singer'].update(
            text=('SINGER.PY invented.wav 1:00\n' + label + '\nSINGER:\n'
                  '  pitch: median C4 · comfortable range G3–G4\n'
                  '  highest held notes: D4 at 0:29 (0.9s)\n'
                  '  phrase endings: 6 lift, 3 fall, 7 hold\n'
                  '  pitch arc: C4 → D4 · the voice lifts ~3 semitones\n'
                  '  overall: arousal 0.40 · dominance 0.39 · valence 0.41 (0-1) '
                  '(speech-trained model; read as texture, not a verdict)\n'
                  'STOPS: 1 · band cuts, voice holds: 1 · full silence: 0 · band break: 0\n'
                  '  0:41.00 (0.8 s) band cuts 16 dB, voice holds\n'),
            json=dict(duration_s=60, lyrics=dict(lookup=source, label=label),
                      singer=dict(data=dict(affect=dict(arousal=.4, dominance=.39, valence=.41)))))
        song['cards']['harmony'].update(
            text=('HARMONY: invented.wav · 1:00\nTIMELINE\n'
                  '   0:00  opens quiet intro, -36 dB\n'
                  '   0:00  0:00–0:23: no voice (instrumental)\n'
                  '   0:23  0:23–0:28: voice present, words unavailable (transcription did not run or failed)\n'
                  '   0:28  0:28–0:58: no voice (instrumental)\n'),
            json=dict(duration=60, lyrics_source=source))
        L.save(self.path(), song)
        before = self.path().read_bytes()
        frozen = copy.deepcopy(song)
        self.calls.clear()
        with patch.dict(sys.modules, {'singer': None, 'music': None, 'harmony': None, 'torch': None}):
            for view in ('full', 'detail', 'music', 'singer', 'harmony', 'summary', 'first-listen'):
                out, _ = self.call('--view', view)
                self.assertNotIn('did not run or failed', out, view)
                self.assertNotIn('the voice', out, view)
                self.assertNotRegex(out, r'arousal 0\.|dominance 0\.|valence 0\.')
                if view in ('full', 'detail', 'singer', 'summary'):
                    self.assertIn(IV.MELODY, out)
                    self.assertIn(IV.AROUSAL, out)
                    self.assertIn('median C4', out)
                    self.assertIn('band cuts, vocal stem holds: 1', out)
        full = L.render(song, 'full')[0]
        small = slim.render(full)
        first = slim.render_first_listen({n: c['text'] for n, c in song['cards'].items()},
                                        {n: c['json'] for n, c in song['cards'].items()})
        for out in (small, first):
            self.assertIn(IV.MELODY, out)
            self.assertIn(IV.AROUSAL, out)
            self.assertIn(IV.WORDS, out)
            self.assertNotIn('did not run or failed', out)
        comparison = compare.parse_card(full)
        self.assertIn('median C4', comparison.voice)
        self.assertIn(IV.AROUSAL, comparison.voice)
        self.assertIn('the melody lifts', comparison.voice_arc)
        self.assertEqual(song, frozen)
        self.assertEqual(self.path().read_bytes(), before)
        self.assertEqual(self.calls, [])

    def test_detail_run_missing_needs_all_passes(self):
        self.call("--only", "music", "--view", "music")
        self.calls.clear()
        self.call("--view", "detail", "--run-missing")
        self.assertEqual(self.calls, ["singer", "harmony"])

    def test_saved_cnn_status_on_full_and_feel_key_lines(self):
        import stage_status
        song = json.loads((L.HERE / 'fixtures' / 'first-listen-song.json').read_text())
        before = json.dumps(song, sort_keys=True)
        for view in ('full', 'summary', 'first-listen'):
            with patch.object(stage_status, 'cnn_key_text', side_effect=lambda text, analysis=None: text):
                original, _ = L.render(song, view)
            self.assertEqual(L.render(song, view)[0], original)
        self.assertEqual(json.dumps(song, sort_keys=True), before)
        music = song['cards']['music']['json']
        music['stage_status'].append(
            stage_status.outcome('CNN key voter', 'skipped', 'venv-madmom not installed'))
        voters = music['key_claim']['voters']
        voters.pop('cnn', None)
        for view in ('full', 'summary', 'first-listen'):
            text, _ = L.render(song, view)
            key_line = next(line for line in text.splitlines() if line.strip().startswith(('KEY:', 'HOME')))
            if view == 'first-listen':
                key_line = ' '.join(text.split('\nHOME', 1)[1].split('\nAnd now, you:', 1)[0].split())
            self.assertIn(f'CNN key voter: optional, not installed ({len(voters)} voters used)', key_line)
            self.assertEqual(key_line.count('CNN key voter:'), 1)

    def test_step1_real_summary_card_fixture_still_renders(self):
        # Step 1's guard (PR #6), kept beside the notation tests: a real listen
        # with every lyric word replaced.
        fixture = L.HERE / "fixtures" / "first-listen-song.json"
        sha = json.loads(fixture.read_text(encoding="utf-8"))["manifest"]["file_sha256"]
        home = Path(os.environ["RAVEN_EARS_HOME"])
        (L.store_home() / sha).mkdir(parents=True)
        shutil.copyfile(fixture, L.store_home() / sha / "song.json")
        original = (fixture.read_bytes(), fixture.stat().st_mtime_ns)
        bundles = list(L.store_home().glob("*/song.json"))
        self.assertEqual(len(bundles), 1)
        before = {p.relative_to(home): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in home.rglob("*") if p.is_file()}
        with patch("socket.create_connection", side_effect=AssertionError("no network")), \
             patch.object(L.subprocess, "run", side_effect=AssertionError("no subprocess")), \
             patch.dict(sys.modules, {"music": None, "singer": None, "harmony": None, "torch": None}):
            song = L.load(bundles[0])
            self.assertTrue(all(r["status"] == "ran" for r in song["manifest"]["passes"].values()))
            text, view = L.render(song, "summary")
        self.assertEqual(view, "summary")
        self.assertNotIn("skipped reported", text)
        headline, lyrics = text.split("LYRICS:", 1)
        self.assertFalse(any(line.startswith(name + ":") for name in L.PASSES for line in headline.splitlines()))
        self.assertGreaterEqual(len(re.findall(r"(?m)^\[\d+:\d+≈?\]", lyrics)), 15)
        self.assertTrue(all(len(line) <= 160 for line in headline.splitlines()))
        # Also check whole harmony/contrast selections, so long definitions
        # cannot hide in headline continuations again.
        selections = []
        for line in headline.splitlines():
            if line.startswith("  "):
                selections[-1] += " " + line.strip()
            else:
                selections.append(line)
        self.assertTrue(all(len(line) <= 160 for line in selections
                            if line.startswith(("HARMONY", "CONTRAST"))))
        self.assertIn("HARMONY (in C major): ends on V (G); opens on Imaj7", text)
        # Expected skips moved to the full card (feel-card notes, 10 Nov 2026).
        self.assertNotIn("breath: skipped", text)
        self.assertNotIn("VS USUAL: skipped", text)
        self.assertRegex(text, r"NOTES: \d+ more on the full card \(--view full\)")
        self.assertEqual(text.count("tension 0–1:"), 1)
        self.assertEqual(text.count("contrast words=model sentiment"), 1)
        self.assertTrue(text.splitlines()[-2].startswith("NOTES: tension 0–1:"))
        self.assertEqual(before, {p.relative_to(home): (p.read_bytes(), p.stat().st_mtime_ns)
                                  for p in home.rglob("*") if p.is_file()})
        self.assertEqual(original, (fixture.read_bytes(), fixture.stat().st_mtime_ns))

    def test_real_saved_store_default_view_is_read_only_and_keeps_lyrics(self):
        # Committed fixture: a real listen's structure with every lyric word invented (see
        # fixtures/README-first-listen-bundle.md). The melody record is derived once and saved,
        # then the default view reads it back with every engine blocked.
        fixture = L.HERE / "fixtures" / "uhuru-title-song.json"
        song = L.validate_song(json.loads(fixture.read_text(encoding="utf-8")))
        original = (fixture.read_bytes(), fixture.stat().st_mtime_ns)
        song["melody"] = M.build(song)
        path = L.store_home() / song["manifest"]["file_sha256"] / "song.json"
        L.save(path, song)
        home = Path(os.environ["RAVEN_EARS_HOME"])
        before = {p.relative_to(home): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in home.rglob("*") if p.is_file()}
        with patch("socket.create_connection", side_effect=AssertionError("no network")), \
             patch.object(L.subprocess, "run", side_effect=AssertionError("no subprocess")), \
             patch.dict(sys.modules, {"music": None, "singer": None, "harmony": None, "torch": None,
                                      "melody_abc": None}):
            song = L.load(path)
            self.assertTrue(all(r["status"] == "ran" for r in song["manifest"]["passes"].values()))
            text, view = L.render(song, L.parser().parse_args([]).view)
        self.assertEqual(view, "first-listen")
        self.assertNotIn("skipped reported", text)
        headline, lyrics = text.split("THE SONG", 1)
        self.assertFalse(any(line.startswith(name + ":") for name in L.PASSES for line in headline.splitlines()))
        self.assertGreaterEqual(len(re.findall(r"(?m)^  \[\d+:\d+≈?\]", lyrics)), 15)
        self.assertTrue(all(len(line) <= 118 for line in text.splitlines()))
        self.assertIn('HOOK: line with the title "uhuru" and the line after it', text)
        self.assertEqual(text.count("\nw: "), 2)
        self.assertIn("opens on i, ends on i (home)", " ".join(text.split()))
        self.assertEqual(before, {p.relative_to(home): (p.read_bytes(), p.stat().st_mtime_ns)
                                  for p in home.rglob("*") if p.is_file()})
        self.assertEqual(original, (fixture.read_bytes(), fixture.stat().st_mtime_ns))

    def test_join_failed_partial_skipped_and_not_run_are_stdout_without_headlines(self):
        def partial(name, args):
            result = self.fake_pass(name, args)
            result.update(status="failed", reason="invented final write failure",
                          text="SINGER.PY example.mp3 1:00\nSINGER:\n  pitch: comfortable range D3–D4\nNOTES: all stages ran\n",
                          json={"singer": {"lines": ["pitch: comfortable range D3–D4"]}})
            return result
        with patch.object(L, "run_pass", side_effect=partial):
            self.call("--only", "singer", "--view", "full")
        song = self.song()
        song["manifest"]["passes"]["music"].update(status="skipped", reason="invented explicit skip")
        song["cards"]["music"]["text"] = "KEY: C major\n"
        L.save(self.path(), song)
        text = self.stdout()
        for message in ("singer: failed: invented final write failure", "music: skipped: invented explicit skip",
                        "harmony: not run: not selected"):
            self.assertIn(message, text)
        self.assertNotIn("example", text.splitlines()[0])
        for claim in ("comfortable range", "KEY: C major", "all stages ran"):
            self.assertNotIn(claim, text)

    def test_interrupted_refresh_keeps_previous_join_after_completed_replacement_pass(self):
        old = self.stdout()
        before = self.song()
        def interrupt(name, args):
            if name == "music":
                raise KeyboardInterrupt()
            result = self.fake_pass(name, args)
            result["text"] = "SINGER.PY example.mp3 1:00\nSINGER:\n  pitch: replacement evidence\n"
            return result
        with patch.object(L, "run_pass", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.call("--fresh")
        after = self.song()
        self.assertEqual(after["cards"], before["cards"])
        for name in L.PASSES:
            self.assertEqual(after["manifest"]["passes"][name]["status"], "ran")
            self.assertEqual(after["manifest"]["passes"][name]["date"], before["manifest"]["passes"][name]["date"])
        self.assertEqual(after["manifest"]["passes"]["singer"]["listen_count"], 2)
        self.assertEqual(after["manifest"]["passes"]["music"]["listen_count"], 1)
        self.assertEqual(self.stdout(), old)
        self.assertNotIn("replacement evidence", self.stdout())

    def test_forwarded_options_match_each_pass_and_help_explains_scopes(self):
        names = self.root / "hints.txt"
        names.write_text("InventedHint")
        text = self.stdout("--names", str(names), "--no-lyrics", "--no-meter", "--no-cache", "--no-instruments")
        self.assertIn("FEELING CARD:", text)
        rows = self.song()["manifest"]["passes"]
        for name in L.PASSES:
            forwarded = rows[name]["forwarded_options"]
            self.assertEqual(forwarded["device"], "auto")
            self.assertEqual("names_sha256" in forwarded, name != "singer")
            self.assertEqual("no_lyrics" in forwarded, name != "singer")
            self.assertEqual("no_meter" in forwarded, name == "singer")
            self.assertEqual("no_cache" in forwarded, name == "harmony")
            self.assertEqual("no_instruments" in forwarded, name != "music")
        help_text = " ".join(L.parser().format_help().split())
        for message in ("singer may still resolve/transcribe lyrics", "music still measures meter",
                        "music still measures sound/vocals", "harmony only", "singer uses its default names"):
            self.assertIn(message, help_text)

    def test_changed_inputs_mid_pass_fail_in_joined_stdout(self):
        for changed in ("audio", "lyrics", "names", "credits"):
            with self.subTest(changed=changed):
                supplied = {}
                for name in ("lyrics", "names", "credits"):
                    supplied[name] = self.root / (name + ".txt")
                    supplied[name].write_text("invented original " + name)
                self.audio.write_bytes(b"invented original " + changed.encode())
                flags = [part for name, path in supplied.items() for part in ("--" + name, str(path))]
                def mutate(name, args):
                    result = self.fake_pass(name, args)
                    result["text"] = "SINGER.PY example.mp3 1:00\nSINGER:\n  pitch: must not headline\n"
                    (self.audio if changed == "audio" else supplied[changed]).write_bytes(b"changed during pass")
                    return result
                original_path = self.path()
                with patch.object(L, "run_pass", side_effect=mutate):
                    text = self.stdout("--only", "singer", *flags)
                self.assertIn("singer: failed: inputs changed during listen: " + changed, text)
                self.assertNotIn("must not headline", text)
                song = L.load(original_path)
                self.assertEqual(song["manifest"]["passes"]["singer"]["status"], "failed")
                self.assertIsNone(song["manifest"]["duration_s"])
                self.assertIn("must not headline", song["cards"]["singer"]["text"])

    def test_disappeared_input_fails_without_counting_passes_never_started(self):
        lyrics = self.root / "lyrics.txt"
        lyrics.write_text("invented words")
        def remove(name, args):
            result = self.fake_pass(name, args)
            lyrics.unlink()
            return result
        with patch.object(L, "run_pass", side_effect=remove):
            text = self.stdout("--lyrics", str(lyrics))
        self.assertIn("singer: failed: inputs could not be re-hashed after pass", text)
        self.assertIn("music: failed: FileNotFoundError", text)
        self.assertEqual(self.calls, ["singer"])
        rows = self.song()["manifest"]["passes"]
        self.assertEqual(rows["singer"]["listen_count"], 1)
        self.assertEqual(rows["music"]["listen_count"], 0)
        self.assertEqual(rows["music"]["forwarded_options"], {})
        self.assertIsNone(rows["music"]["last_listened"])
        self.assertNotIn("An invented lantern", text)

    def test_analysis_paths_are_basenames_before_save_and_join(self):
        def paths(name, args):
            result = self.fake_pass(name, args)
            result["text"] = ""
            result["json"].update(file=str(self.audio), stems_dir=str(self.root / "stems"),
                nested={"files": [str(self.root / "invented.wav"), "diagnostic at " + str(self.root / "weights.bin"),
                           'diagnostic at "' + str(self.root / "with space/weights.bin") + '"',
                           r"C:\invented\audio.wav", r"diagnostic at C:\invented\weights.bin"]})
            return result
        with patch.object(L, "run_pass", side_effect=paths):
            text = self.stdout()
        self.assertIn("Lantern", text)
        self.assertNotIn(str(self.root), text)
        for card in self.song()["cards"].values():
            self.assertEqual(card["json"]["file"], self.audio.name)
            self.assertEqual(card["json"]["stems_dir"], "stems")
            self.assertNotIn(str(self.root), json.dumps(card["json"]))
            self.assertEqual(card["json"]["nested"]["files"], ["invented.wav", "diagnostic at weights.bin",
                'diagnostic at "weights.bin"', "audio.wav", "diagnostic at weights.bin"])

    def test_packed_stops_curves_survive_the_path_scrubber(self):
        # 13 Nov 2026: the path regexes read base64 "+/abc/def" as a path and cut
        # it, so every fresh bundle's STOPS curves failed to decode after #50.
        import numpy as np
        import stops
        rng = np.random.default_rng(7)
        for _ in range(20):
            curves = {s: rng.random(400) for s in stops.STEMS}
            packed = stops._pack_curves(curves)
            if re.search(r"[+=]/", packed["data"]):
                break
        else:
            self.fail("no invented curve produced a path-like base64 run")
        saved = L.analysis_basenames({"stops": {"status": "ran", "curves": packed}})
        self.assertEqual(saved["stops"]["curves"], packed)
        out = stops._unpack_curves(saved["stops"]["curves"])
        for s in stops.STEMS:
            self.assertTrue(np.array_equal(out[s], curves[s]))
        # Anything that is not a well-formed packed curve is still scrubbed.
        odd = L.analysis_basenames({"curves": {"encoding": "other", "data": "/home/x/secret.wav"}})
        self.assertEqual(odd["curves"]["data"], "secret.wav")

    def test_one_record_counts_attempts_not_reads_and_deleted_with_bundle(self):
        self.stdout("--only", "music")
        rows = self.song()["manifest"]["passes"]
        first = rows["music"]["first_listened"]
        self.assertEqual(rows["music"]["listen_count"], 1)
        self.assertEqual(rows["singer"]["listen_count"], 0)
        history = self.stdout("--history")
        self.assertIn("music: first listened " + first, history)
        self.assertIn("singer: first listened never", history)
        self.stdout()
        self.assertEqual(self.stdout("--history"), history)
        self.stdout("--run-missing")
        def fail(name, args):
            result = self.fake_pass(name, args)
            result.update(status="failed", reason="invented second attempt failure")
            return result
        with patch.object(L, "run_pass", side_effect=fail):
            text = self.stdout("--fresh")
        self.assertIn("singer: failed: invented second attempt failure", text)
        rows = self.song()["manifest"]["passes"]
        for row in rows.values():
            self.assertEqual(row["listen_count"], 2)
            self.assertGreaterEqual(row["last_listened"], row["first_listened"])
        self.assertEqual(rows["music"]["first_listened"], first)
        self.assertIn("count 2; saved status failed", self.stdout("--history"))
        self.assertEqual([p.name for p in self.path().parent.iterdir()], ["song.json"])
        self.assertIn("Deleted saved song", self.stdout("--delete"))
        self.assertFalse(self.path().exists())

    def test_saved_version_and_options_warnings_survive_stdout_redirect(self):
        self.stdout("--no-meter")
        self.calls.clear()
        with patch.object(L, "ears_version", return_value="test-version-2"):
            text = self.stdout("--run-missing")
        self.assertIn("Saved card version differs", text)
        self.assertIn("Saved card options differ: no_meter", text)
        self.assertEqual(self.calls, [])

    def test_harmony_reference_and_uncertainty_survive_in_joined_stdout(self):
        import harmony
        result = {"evs": [{"root": 9, "q": "min", "t0": 0, "t1": 60}],
                  "spans": [], "secs": [], "dur": 60, "glob": (0, "major"), "kinfo": {},
                  "key_verdict": {"key": None, "working": "C major",
                                  "line": "unclear between C major and A minor"},
                  "bpm": 80, "moves": [], "pedals": []}
        with patch.object(harmony, "label", return_value="vi"):
            chords = "\n".join(harmony.block_chords(result))
        def formatted(name, args):
            card = self.fake_pass(name, args)
            card["text"] = chords
            return card
        with patch.object(L, "run_pass", side_effect=formatted):
            text = " ".join(self.stdout("--only", "harmony").split())
        self.assertIn("ends on vi (Am); opens on vi", text)
        self.assertIn("HOME in C major, key unclear · ends on vi (Am); opens on vi", text)
        song = self.song()
        song["cards"]["harmony"]["text"] = "CHORDS\n  ends on vi (Am); opens on vi\n"
        L.save(self.path(), song)
        self.assertNotIn("ends on", self.stdout())

    def test_legacy_history_stays_unknown_after_refresh(self):
        self.stdout()
        song = self.song()
        for row in song["manifest"]["passes"].values():
            for key in ("first_listened", "last_listened", "listen_count"):
                row.pop(key)
        L.save(self.path(), song)
        self.assertIn("unknown (older bundle)", self.stdout("--history"))
        self.assertIn("An invented lantern", self.stdout("--fresh"))
        self.assertTrue(all(row["listen_count"] is None for row in self.song()["manifest"]["passes"].values()))
        self.assertIn("count unknown (older bundle)", self.stdout("--history"))

    def test_setup_notes_stay_full_and_different_counts_name_the_pass(self):
        def notes(name, args):
            result = self.fake_pass(name, args)
            count = 8 if name == "music" else 56
            result["text"] += ("NOTE: optional part missing optional weights\n"
                "NOTE: LRCLIB: skipped: lookup disabled\nNOTE: device cpu\nNOTE: stems cache ready\n"
                "NOTE: sentiment: failed: ModuleNotFoundError: No module named 'invented_optional'\n"
                f"NOTE: lyrics: re-listened to {count} long gaps and recovered {count} words\n")
            return result
        with patch.object(L, "run_pass", side_effect=notes):
            text = self.stdout("--view", "summary")
        self.assertIn("[music pass]: lyrics: re-listened to 8", text)
        self.assertIn("singer pass", text)
        self.assertIn("re-listened to 56", text)
        for setup in ("missing optional", "lookup disabled", "device cpu", "stems cache ready"):
            self.assertNotIn(setup, text)
            self.assertIn(setup, self.stdout("--view", "full"))
        # A stage that FAILED is evidence, not setup, even when a module is missing.
        self.assertIn("sentiment: failed: ModuleNotFoundError", " ".join(text.split()))

    def test_saved_read_survives_a_vanished_lyrics_file(self):
        # Review of PR #6, 10 Nov 2026: a deleted --lyrics file blocked the saved read.
        lyrics = self.root / "words.txt"
        lyrics.write_text("invented words")
        with patch.object(L, "run_pass", side_effect=self.fake_pass):
            self.stdout("--lyrics", str(lyrics))
        lyrics.unlink()
        calls = list(self.calls)
        with patch.object(L, "run_pass", side_effect=self.fake_pass):
            text = self.stdout("--lyrics", str(lyrics))
        self.assertEqual(self.calls, calls)
        self.assertIn("FEELING CARD", text)
        self.assertIn("supplied file not found (--lyrics)", text)

    def test_manifest_tracks_inputs_without_storing_their_paths(self):
        credits = self.root / "credits.txt"
        names = self.root / "names.txt"
        credits.write_text("Invented Player: piano\n")
        names.write_text("Lantern\n")
        self.call("--view", "full", "--credits", str(credits), "--names", str(names),
                  "--artist", "Invented Artist", "--title", "Lantern", "--no-lyrics-lookup", "--device", "cpu")
        inputs = self.song()["manifest"]["options"]
        self.assertEqual(inputs["credits_sha256"], L.file_hash(credits))
        self.assertEqual(inputs["names_sha256"], L.file_hash(names))
        self.assertEqual(inputs["device"], "cpu")
        self.assertEqual(inputs["lookups"], {"credits": False, "lyrics": False})
        self.assertNotIn(str(self.root), json.dumps(inputs))
        credits.write_text("Invented Player: guitar\n")
        self.calls.clear()
        _, notices = self.call("--view", "full", "--credits", str(credits))
        self.assertIn("re-listening: --credits differs from the saved card", notices)
        self.assertEqual(self.calls, list(L.PASSES))
        self.assertEqual(self.song()["manifest"]["options"]["credits_sha256"], L.file_hash(credits))

    def test_json_stdout_remains_parseable_with_provenance_on_stderr(self):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            self.assertEqual(L.main([str(self.audio), "--json"]), 0)
        self.assertEqual(json.loads(out.getvalue())["schema"], 1)
        self.assertIn("Listening now", err.getvalue())

    def test_no_baseline_main_ignores_persistent_overrides_and_cleans_up(self):
        persistent = self.root / "must-not-be-written"
        seen = []
        def isolated_pass(name, args):
            root = Path(os.environ["RAVEN_EARS_HOME"]).parent
            seen.append(root)
            for key in ("HOME", "RAVEN_STEMS_CACHE", "RAVEN_LYRICS_CACHE", "RAVEN_HARMONY_CACHE",
                        "RAVEN_PANNS_CACHE", "HF_HOME", "TORCH_HOME", "TMPDIR", "XDG_CACHE_HOME"):
                destination = Path(os.environ[key]) / "invented-write"
                self.assertTrue(destination.is_relative_to(root))
                destination.write_text("invented")
            return self.fake_pass(name, args)
        with patch.dict(os.environ, {key: str(persistent) for key in
             ("RAVEN_EARS_HOME", "RAVEN_STEMS_CACHE", "HF_HOME", "TORCH_HOME")}), \
             patch.object(L, "run_pass", side_effect=isolated_pass), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(L.main([str(self.audio), "--view", "full"]), 0)
            self.assertEqual(os.environ["RAVEN_EARS_HOME"], str(persistent))
        self.assertFalse(persistent.exists())
        self.assertEqual(len(set(seen)), 1)
        self.assertFalse(seen[0].exists())

    def test_worker_captures_json_and_passes_args_once(self):
        analysis = self.root / "analysis.json"
        calls = []
        def fake_main(argv, capture):
            calls.append(argv)
            capture({"invented": "analysis"})
            print("invented original card")
        with patch.object(L.importlib, "import_module", return_value=SimpleNamespace(main=fake_main)), \
             redirect_stdout(io.StringIO()) as out:
            self.assertEqual(L.worker("singer", str(analysis), ["invented.mp3", "--no-meter"]), 0)
        self.assertEqual(calls, [["invented.mp3", "--no-meter"]])
        self.assertEqual(json.loads(analysis.read_text()), {"invented": "analysis"})
        self.assertEqual(out.getvalue(), "invented original card\n")

    def test_length_check_through_singer_worker_save_and_every_saved_view(self):
        import singer
        import lrclib
        import lyrics_shared
        import soundfile
        from test_song_length import record
        self.runner.stop()
        def child(command, **kwargs):
            with patch.dict(os.environ, kwargs['env'], clear=True), redirect_stdout(io.StringIO()) as out:
                code = L.worker(command[3], command[4], command[5:])
            return SimpleNamespace(returncode=code, stdout=out.getvalue(), stderr='')
        def service(endpoint, params):
            if endpoint == 'get':
                return record() if abs(params['duration'] - 172) <= 2 else None
            self.assertEqual(endpoint, 'search')
            return [record()]
        with patch.object(singer, 'get_stems', return_value=(str(self.root), True)), \
             patch.object(singer, 'vocal_presence', return_value=(.3, 40)), \
             patch.object(singer, 'singer_block', return_value=(['SINGER: invented'], {}, {})) as singer_block, \
             patch.object(lrclib, 'cached_activity', return_value=None), \
             patch.object(lyrics_shared, 'get_words', return_value=[('invented', 10., 11.)]) as whisper, \
             patch.dict(lrclib.CONFIG), patch.dict(os.environ, RAVEN_NO_STOPS='1'), \
             patch.object(L.subprocess, 'run', side_effect=child), \
             patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            # Header durations measured on the supplied video/studio copies.
            for duration in (191.712, 172.848):
                warned = duration > 180
                self.path().unlink(missing_ok=True)
                # Distinct temporary metadata caches, so the fake requests are observed.
                with self.subTest(duration=duration), \
                     patch.object(soundfile, 'info', return_value=SimpleNamespace(duration=duration)), \
                     patch.object(lrclib, 'cache_path', return_value=str(self.root / f'lrclib-{duration}.json')), \
                     patch.object(lrclib, '_get_json', side_effect=service) as request:
                    text, _ = self.call('--only', 'singer', '--no-instruments', '--no-meter',
                                        '--no-stems-timeline', '--no-affect',
                                        '--artist', 'Example Artist', '--title', 'Example Song')
                    self.assertFalse(singer_block.call_args.kwargs['affect'])
                    self.assertEqual(request.call_count, 2 if warned else 1)
                    song = self.song()
                    self.assertTrue(song['manifest']['options']['no_affect'])
                    self.assertEqual('length_check' in song, warned)
                    self.assertEqual(song['cards']['singer']['text'].count('⚠ LENGTH:'), int(warned))
                    if warned:
                        self.assertEqual(song['length_check']['studio_s'], 172)
                        self.assertEqual(song['length_check']['sources'], ['LRCLIB'])
                        self.assertEqual(song['cards']['singer']['json']['lyrics']['lookup']['status'], 'nomatch')
                        self.assertIn('NOTES: LRCLIB has clean lyrics at 2:52; not used', text)
                    before = self.path().read_bytes()
                    request.reset_mock()
                    with patch.object(L, 'run_pass', side_effect=AssertionError('saved views never listen')):
                        for view in L.VIEWS:
                            replay, _ = self.call('--view', view)
                            self.assertEqual(replay.count('⚠ LENGTH:'), int(warned))
                    request.assert_not_called()
                    self.assertEqual(self.path().read_bytes(), before)
            self.assertEqual(whisper.call_count, 2)

    def test_runner_keeps_nonzero_exit_diagnostics_and_partial_analysis(self):
        self.runner.stop()
        L.store_home().mkdir(parents=True)
        def subprocess_fake(command, **kwargs):
            self.assertEqual(command[2:4], ["--_worker", "singer"])
            Path(command[4]).write_text('{"duration_s": 60}')
            return SimpleNamespace(returncode=1, stdout="invented partial card", stderr="invented backend error")
        with patch.object(L.subprocess, "run", side_effect=subprocess_fake):
            result = L.run_pass("singer", L.parser().parse_args([str(self.audio)]))
        self.assertEqual(result["status"], "failed")
        self.assertIn("invented backend error", result["reason"])
        self.assertEqual(result["json"], {"duration_s": 60})
        self.assertEqual(result["text"], "invented partial card")

    def test_timing_sidecars_leave_runner_and_saved_analysis_identical(self):
        import pstats
        self.runner.stop()
        L.store_home().mkdir(parents=True)
        timing = self.root / "timing"
        commands = []
        def fake_main(argv, capture):
            self.assertEqual(argv, [str(self.audio.resolve())])
            capture({"duration_s": 60, "stage_status": []})
            print("invented original card")
        def subprocess_fake(command, **kwargs):
            commands.append(command)
            with patch.object(L.importlib, "import_module", return_value=SimpleNamespace(main=fake_main)), \
                 redirect_stdout(io.StringIO()) as out:
                code = L.worker(command[3], command[4], command[5:])
            return SimpleNamespace(returncode=code, stdout=out.getvalue(), stderr="")
        args = L.parser().parse_args([str(self.audio)])
        with patch.dict(os.environ), patch.object(L.subprocess, "run", side_effect=subprocess_fake):
            os.environ.pop("RAVEN_EARS_TIMING_DIR", None)
            plain = L.run_pass("singer", args)
            self.assertFalse(timing.exists())
            os.environ["RAVEN_EARS_TIMING_DIR"] = str(timing)
            timed = L.run_pass("singer", args)
        self.assertEqual(plain, timed)
        # Temporary analysis destinations differ; engine arguments must not.
        self.assertEqual(commands[0][:4], commands[1][:4])
        self.assertEqual(commands[0][5:], commands[1][5:])
        path = self.root / "song.json"
        L.save(path, {"cards": {"singer": plain}})
        original = path.read_bytes()
        L.save(path, {"cards": {"singer": timed}})
        self.assertEqual(path.read_bytes(), original)
        stats = pstats.Stats(str(timing / "singer.pstats")).stats
        self.assertTrue(any(key[2] == "fake_main" for key in stats))

    def test_failed_worker_still_writes_timing_sidecar(self):
        import pstats
        timing = self.root / "failed-timing"
        def broken_main(argv, capture):
            raise ValueError("invented stage failure")
        with patch.dict(os.environ, {"RAVEN_EARS_TIMING_DIR": str(timing)}), \
             patch.object(L.importlib, "import_module", return_value=SimpleNamespace(main=broken_main)):
            with self.assertRaisesRegex(ValueError, "invented stage failure"):
                L.worker("singer", str(self.root / "analysis.json"), ["invented.mp3"])
        stats = pstats.Stats(str(timing / "singer.pstats")).stats
        self.assertTrue(any(key[2] == "broken_main" for key in stats))

    def test_quick_worker_policy_survives_run_pass_and_song_save(self):
        self.runner.stop()
        environments, commands = [], []
        def fake_main(argv, capture):
            # Quick leaves the lyrics model and device policy unchanged.
            self.assertEqual(argv, [str(self.audio.resolve()), *(["--quick"] if "--quick" in argv else []),
                                    *(["--_picture-data"] if "--_picture-data" in argv else []),
                                    *(["--key-changes"] if "--key-changes" in argv else []),
                                    *(["--no-key-changes"] if "--no-key-changes" in argv else [])])
            capture({"duration_s": 60, "stage_status": [],
                     "instruments": {"six_stems": os.environ.get("RAVEN_NO_SIX_STEMS") != "1"}})
            print("invented original card")
        def subprocess_fake(command, **kwargs):
            commands.append(command[5:])
            environments.append(kwargs["env"])
            with patch.dict(os.environ, kwargs["env"], clear=True), \
                 patch.object(L.importlib, "import_module", return_value=SimpleNamespace(main=fake_main)), \
                 redirect_stdout(io.StringIO()) as out:
                code = L.worker(command[3], command[4], command[5:])
            return SimpleNamespace(returncode=code, stdout=out.getvalue(), stderr="")
        with patch.object(L.subprocess, "run", side_effect=subprocess_fake):
            self.call("--quick", "--device", "cpu")
        song = self.song()
        self.assertTrue(song["manifest"]["options"]["quick"])
        for name in L.PASSES:
            self.assertFalse(song["cards"][name]["json"]["instruments"]["six_stems"])
            forwarded = song["manifest"]["passes"][name]["forwarded_options"]
            self.assertTrue(forwarded["quick"])
        self.assertEqual(commands, [[str(self.audio.resolve()), "--quick"],
                                    [str(self.audio.resolve()), "--quick", "--_picture-data"],
                                    [str(self.audio.resolve()), "--quick", "--no-key-changes"]])
        self.assertTrue(all(env["RAVEN_NO_SIX_STEMS"] == "1" for env in environments))
        self.assertTrue(all(env["RAVEN_EARS_DEVICE"] == "cpu" for env in environments))
        self.assertTrue(all(env.get("RAVEN_LYRICS_WHISPER") == os.environ.get("RAVEN_LYRICS_WHISPER")
                            for env in environments))
        before = self.path().read_bytes()
        self.assertNotEqual(os.environ.get("RAVEN_NO_SIX_STEMS"), "1")
        with patch.object(L, "run_pass", side_effect=AssertionError("saved views must not listen")):
            for view in L.VIEWS:
                text, _ = self.call("--view", view, "--device", "cpu")
                self.assertIn("Quick listen. Skipped: the extra guitar/piano separation", text)
            _, notices = self.call("--run-missing", "--device", "cpu")
        self.assertIn("quick", " ".join(notices))
        self.assertEqual(self.path().read_bytes(), before)
        # A failed quick pass cannot be retried with full settings in one bundle.
        song["manifest"]["passes"]["harmony"].update(status="failed", reason="invented failure")
        L.save(self.path(), song)
        with patch.object(L, "run_pass", side_effect=AssertionError("mixed settings")):
            _, notices = self.call("--run-missing", "--device", "cpu")
        self.assertIn("Missing passes left unchanged", " ".join(notices))
        with patch.object(L.subprocess, "run", side_effect=subprocess_fake):
            self.call("--fresh", "--device", "cpu")
        self.assertNotIn("quick", self.song()["manifest"]["options"])
        self.assertTrue(all(card["json"]["instruments"]["six_stems"]
                            for card in self.song()["cards"].values()))

    def test_default_does_not_acquire_quick_options_or_worker_policy(self):
        self.runner.stop()
        def subprocess_fake(command, **kwargs):
            self.assertEqual(command[5:], [str(self.audio.resolve()), *(["--quick"] if quick else []),
                             *(["--_picture-data"] if command[3] == "music" else []),
                             *(["--no-key-changes" if quick else "--key-changes"] if command[3] == "harmony" else [])])
            expected_env = dict(os.environ, RAVEN_EARS_DEVICE="auto")
            if quick:
                expected_env["RAVEN_NO_SIX_STEMS"] = "1"
            self.assertEqual(kwargs["env"], expected_env)
            Path(command[4]).write_text('{"duration_s": 60, "stage_status": []}')
            return SimpleNamespace(returncode=0, stdout="invented original card\n", stderr="")
        quick = False
        with patch.object(L.subprocess, "run", side_effect=subprocess_fake):
            text, _ = self.call()
            self.assertNotIn("quick", self.song()["manifest"]["options"])
            self.assertNotIn("Quick listen.", text)
            quick = True
            _, notices = self.call("--quick")
        self.assertIn("re-listening: --quick differs from the saved card", notices)

    def test_quick_with_instruments_disabled_has_plain_skip_note(self):
        text, _ = self.call("--quick", "--no-instruments")
        self.assertIn("Quick listen. Skipped: instrument naming and performer matching", text)
        self.assertIn("key-change scan", text)
        self.assertNotIn("checking synced lyrics", text)
        self.assertNotIn("Instrument names use four stems", text)

    def test_worker_progress_bypasses_captured_diagnostics(self):
        import stage_status
        self.runner.stop()
        L.store_home().mkdir(parents=True)
        args = L.parser().parse_args([str(self.audio), "--quick"])
        args._progress_started = 100

        def engine(argv, capture):
            self.assertIn("--quick", argv)
            stage_status.progress("pitch and melody")
            print("invented card")
            print("invented diagnostic", file=sys.stderr)
            capture({"duration_s": 60})

        def child(command, **kwargs):
            argv = command[2:]
            fd, = kwargs["pass_fds"]
            self.assertEqual(argv[4], str(fd))
            argv[4] = str(os.dup(fd))
            with (redirect_stdout(io.StringIO()) as stdout,
                  redirect_stderr(io.StringIO()) as stderr,
                  patch.object(L.importlib, "import_module", return_value=SimpleNamespace(main=engine))):
                code = L.main(argv)
            return SimpleNamespace(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())

        with (tempfile.TemporaryFile(mode="w+") as progress, redirect_stderr(progress),
              patch.object(L.time, "monotonic", return_value=173),
              patch.object(L.subprocess, "run", side_effect=child)):
            result = L.run_pass("singer", args)
            progress.seek(0)
            self.assertEqual(progress.read(), "  [1:13] pitch and melody…\n")
        self.assertEqual(result["text"], "invented card\n")
        self.assertEqual(result["diagnostics"], "invented diagnostic\n")
        self.assertEqual(result["status"], "ran")

    def test_combined_quick_lyrics_through_singer_worker_and_saved_views(self):
        import singer
        import lrclib
        import lyrics_shared as LS
        import soundfile
        self.runner.stop()
        # Same lookup/row shape as the supplied Love Story bundle; invented words.
        synced = {'source': 'LRCLIB', 'status': 'synced', 'plain': [], 'issues': [],
                  'synced': [{'t': 17.04, 'text': 'An invented lantern lights the room'},
                             {'t': 24.0, 'text': 'Another lantern shines'}]}
        given = self.root / 'given.txt'
        given.write_text('An invented lantern lights the room\nAnother lantern shines\n')
        heard = [('An', 17., 18.), ('invented', 18., 19.), ('lantern', 19., 20.)]
        def block(path, duration, words, **kwargs):
            self.assertEqual(kwargs.get('words_skip_reason'),
                             'skipped in a quick listen' if words is None else None)
            return ['pitch: invented'], {}, {}
        def child(command, **kwargs):
            with patch.dict(os.environ, kwargs['env'], clear=True), redirect_stdout(io.StringIO()) as out:
                code = L.worker(command[3], command[4], command[5:])
            return SimpleNamespace(returncode=code, stdout=out.getvalue(), stderr='')
        cases = [('synced', {}, False, True), ('synced', {}, False, True),
                 ('plain', {}, False, False), ('nomatch', {}, False, False),
                 ('synced', {'paired': True}, False, False),
                 ('synced', {'issues': ['search incomplete']}, False, False),
                 ('synced', {}, True, False)]
        with patch.object(singer, 'get_stems', return_value=(str(self.root), True)), \
             patch.object(singer, 'vocal_presence', return_value=(.3, 40)), \
             patch.object(singer, 'singer_block', side_effect=block), \
             patch.object(soundfile, 'info', return_value=SimpleNamespace(duration=60)), \
             patch.object(lrclib, 'QUICK', False), patch.dict(lrclib.CONFIG), \
             patch.dict(os.environ, RAVEN_NO_STOPS='1'), \
             patch.object(L.subprocess, 'run', side_effect=child):
            for index, (status, extras, file_input, skipped) in enumerate(cases):
                self.path().unlink(missing_ok=True)
                lookup = dict(synced, status=status, **extras)
                if status == 'plain':
                    lookup['plain'] = [r['text'] for r in synced['synced']]
                activity = [] if index == 1 else [(16., 22.)]
                flags = ['--quick', '--only', 'singer', '--no-instruments',
                         '--no-meter', '--no-stems-timeline']
                if file_input:
                    flags += ['--lyrics', str(given)]
                with self.subTest(status=status, extras=extras, file=file_input, activity=activity), \
                     patch.object(lrclib, 'get', return_value=lookup), \
                     patch.object(lrclib, 'cached_activity', return_value=activity) as vocal_check, \
                     patch.object(LS, 'get_words', return_value=heard) as whisper:
                    text, _ = self.call(*flags)
                    song = self.song()
                    self.assertEqual(song['manifest']['passes']['singer']['status'], 'ran', text)
                    result = song['cards']['singer']['json']['lyrics']
                    self.assertEqual(result['quick_skipped'], skipped)
                    self.assertEqual(whisper.call_count, 0 if skipped else 1)
                    self.assertFalse(song['manifest']['options']['key_changes'])
                    if skipped:
                        vocal_check.assert_called_once()
                        self.assertNotIn('words', result)  # singer never stores supplied words as heard evidence
                        self.assertEqual([r['t'] for r in result['rows']], [17.04, 24.0])
                        self.assertTrue(result['rows'][1]['no_voice'])
                        self.assertTrue(all(r['approx'] and r['timing'] == 'LRCLIB supplied' for r in result['rows']))
                        self.assertEqual(song['melody']['reason'], 'skipped in a quick listen')
                        before = self.path().read_bytes()
                        for view in L.VIEWS:
                            saved, _ = self.call('--view', view)
                            self.assertIn('checking synced lyrics against the singing', ' '.join(saved.split()))
                            self.assertIn('key-change scan', saved)
                            self.assertNotIn('transcription did not run or failed', saved)
                            self.assertNotIn('no word times', saved)
                        self.assertEqual(self.path().read_bytes(), before)
                    else:
                        self.assertNotIn('checking synced lyrics against the singing', text)

    def test_old_saved_key_scan_setting_is_inherited_until_fresh_or_explicit(self):
        self.call()
        for missing_field in (False, True):
            song = self.song()
            song['manifest']['options']['key_changes'] = False
            if missing_field:
                del song['manifest']['options']['key_changes']
            song['manifest']['passes']['harmony'].update(status='failed', reason='invented retry')
            L.save(self.path(), song)
            original = self.path().read_bytes()
            _, notices = self.call()
            self.assertFalse(any('options differ' in n for n in notices), notices)
            self.assertEqual(self.path().read_bytes(), original)
            def retry(name, args):
                self.assertEqual(name, 'harmony')
                self.assertIn('--no-key-changes', L.pass_args(name, args))
                return self.fake_pass(name, args)
            with patch.object(L, 'run_pass', side_effect=retry):
                _, notices = self.call('--run-missing')
            self.assertFalse(any('options differ' in n for n in notices), notices)
            self.calls.clear()
            _, notices = self.call('--key-changes')
            self.assertIn('re-listening: --key-changes differs from the saved card', notices)
            self.assertEqual(self.calls, list(L.PASSES))
        self.call('--fresh')
        self.assertTrue(self.song()['manifest']['options']['key_changes'])
        self.call('--fresh', '--no-key-changes')
        self.assertFalse(self.song()['manifest']['options']['key_changes'])

    def test_four_stem_instrument_cache_cannot_replace_or_read_full_cache(self):
        import instruments as I
        directory = self.root / "stems"
        directory.mkdir()
        full = {"version": I.VERSION, "six_stems": True, "families": {}}
        quick = dict(full, six_stems=False)
        (directory / "instruments.json").write_text(json.dumps(full))
        (directory / "instruments-4s.json").write_text(json.dumps(quick))
        with patch.object(I, "_stem_dir", return_value=(str(directory), True)), \
             patch.dict(os.environ):
            os.environ.pop("RAVEN_NO_SIX_STEMS", None)
            self.assertEqual(I.analyse_raw("invented.mp3"), full)
            os.environ["RAVEN_NO_SIX_STEMS"] = "1"
            self.assertEqual(I.analyse_raw("invented.mp3"), quick)
            (directory / "instruments-4s.json").unlink()
            with patch("raven_ears_plus._torch", side_effect=RuntimeError("fresh naming required")):
                with self.assertRaisesRegex(RuntimeError, "fresh naming required"):
                    I.analyse_raw("invented.mp3")
            self.assertEqual(json.loads((directory / "instruments.json").read_text()), full)

    def test_language_note_survives_worker_runner_and_song_save(self):
        import lyrics_shared as LS
        import lrclib
        self.runner.stop()
        L.store_home().mkdir(parents=True)
        text = ['Je suis dans une maison. Vous et nous, les amis, mes livres et mon papier sont pour vous.']
        def subprocess_fake(command, **kwargs):
            self.assertEqual(command[2:4], ['--_worker', 'singer'])
            notes = []
            with patch.object(LS, 'transcribe', return_value=[('papier', 10., 10.5)]), \
                 patch.object(LS, 'CACHE', str(self.root / 'lyrics')), \
                 patch.object(LS, '_model_attempts', return_value=[('cpu', 'int8')]):
                result = lrclib.resolve(str(self.audio), notes, 60., lyric_lines=text, title='Invented')
            data = dict(lyrics_notes=notes, lyrics=result)
            Path(command[4]).write_text(json.dumps(data))
            return SimpleNamespace(returncode=0, stdout='\n'.join(notes), stderr='')
        with patch.dict(os.environ), patch.object(L.subprocess, 'run', side_effect=subprocess_fake):
            os.environ.pop('RAVEN_EARS_LANG', None)
            card = L.run_pass('singer', L.parser().parse_args([str(self.audio)]))
        path = self.root / 'song.json'
        L.save(path, {'cards': {'singer': card}})
        saved = json.loads(path.read_text())['cards']['singer']
        self.assertEqual(saved['status'], 'ran')
        self.assertIn('lyrics: language fr (lyrics file, dominant whole-song text cues', saved['text'])
        self.assertEqual(saved['json']['lyrics_notes'], card['json']['lyrics_notes'])
        self.assertIn('language fr', saved['json']['lyrics_notes'][0])

    def test_checkpoint_error_survives_run_pass_scrubbing_and_song_save(self):
        import singer
        import stage_status as ST
        from unittest.mock import Mock
        from urllib.error import HTTPError
        self.runner.stop()
        L.store_home().mkdir(parents=True)
        checkpoint_home = self.root / 'torch'
        torch = SimpleNamespace(hub=SimpleNamespace(get_dir=lambda: str(checkpoint_home)))
        error = HTTPError(singer.BEAT_CHECKPOINT_URL, 500, 'Internal Server Error', {}, None)
        tracker = Mock(side_effect=AssertionError('download failed; no inference'))
        def subprocess_fake(command, **kwargs):
            @ST.engine
            def measure():
                try:
                    singer.beats_downbeats(str(self.audio), use_cache=False)
                except RuntimeError:
                    pass
                return {}
            with patch.dict(os.environ, {'RAVEN_EARS_OFFLINE': '0', 'RAVEN_EARS_DEVICE': 'cpu'}), \
                 patch.dict(sys.modules, {'torch': torch, 'madmom': None,
                     'beat_this.inference': SimpleNamespace(Audio2Beats=tracker)}), \
                 patch('urllib.request.urlopen', side_effect=error), redirect_stderr(io.StringIO()):
                data = measure()
            tracker.assert_not_called()
            Path(command[4]).write_text(json.dumps(data))
            return SimpleNamespace(returncode=0, stdout='\n'.join(ST.add_notes(['METER: unavailable'], data['stage_status'])), stderr='')
        with patch.object(L.subprocess, 'run', side_effect=subprocess_fake):
            card = L.run_pass('singer', L.parser().parse_args([str(self.audio)]))
        path = self.root / 'song.json'
        L.save(path, {'cards': {'singer': card}})
        saved = json.loads(path.read_text())['cards']['singer']
        for text in (saved['text'], ST.summary(saved['json']['stage_status'])):
            self.assertIn('checkpoint download failed', text)
            self.assertIn('HTTP Error 500', text)
            self.assertIn(singer.BEAT_CHECKPOINT_URL, text)
            for _, url in singer.BEAT_CHECKPOINT_MIRRORS:
                self.assertIn(url, text)
            self.assertIn('madmom', text)

    def test_invalid_analysis_remains_an_explicit_failure_with_original_text(self):
        self.runner.stop()
        L.store_home().mkdir(parents=True)
        for contents in ('{broken', '{"value": NaN}', '[]'):
            def subprocess_fake(command, **kwargs):
                Path(command[4]).write_text(contents)
                return SimpleNamespace(returncode=0, stdout="invented unchanged card", stderr="")
            with patch.object(L.subprocess, "run", side_effect=subprocess_fake):
                result = L.run_pass("singer", L.parser().parse_args([str(self.audio)]))
            self.assertEqual(result["status"], "failed")
            self.assertIn("invalid JSON analysis", result["reason"])
            self.assertEqual(result["text"], "invented unchanged card")
            self.assertIsNone(result["json"])

    def test_no_cache_and_pass_specific_options_route_only_to_accepting_passes(self):
        args = L.parser().parse_args([str(self.audio), "--names", "invented-names.txt", "--no-cache",
            "--no-lyrics", "--no-instruments", "--no-meter", "--no-stems-timeline"])
        singer, music, harmony = [L.pass_args(name, args) for name in L.PASSES]
        self.assertNotIn("--names", singer)
        self.assertNotIn("--no-lyrics", singer)
        self.assertIn("--no-meter", singer)
        self.assertNotIn("--no-meter", music)
        self.assertNotIn("--no-instruments", music)
        self.assertNotIn("--no-cache", music)
        self.assertIn("--no-cache", harmony)
        self.assertIn("--names", harmony)

    def test_default_and_overridden_store_roots(self):
        with patch.dict(os.environ, {"HOME": str(self.root)}, clear=True):
            self.assertEqual(L.store_home(), self.root / ".cache/raven/cards/songs")
            with patch.dict(os.environ, {"RAVEN_CARDS_DIR": str(self.root / "cards")}):
                self.assertEqual(L.store_home(), self.root / "cards/songs")
            with patch.dict(os.environ, {"RAVEN_EARS_HOME": str(self.root / "ears")}):
                self.assertEqual(L.store_home(), self.root / "ears/songs")

    def test_version_falls_back_without_git_and_detects_source_edits(self):
        self.version.stop()
        sources = self.root / "sources"
        sources.mkdir()
        script = sources / "music.py"
        script.write_text("version one")
        with patch.object(L, "HERE", sources), patch.object(L.subprocess, "run", side_effect=OSError("no git")):
            first = L.ears_version()
            script.write_text("version two")
            second = L.ears_version()
        self.assertTrue(first.startswith(L.VERSION + ":"))
        self.assertNotEqual(first, second)

    def test_music_capture_keeps_the_standalone_short_card_unchanged(self):
        import music
        fake_audio = SimpleNamespace(load_audio=lambda *_: [0] * music.SR)
        captured, outputs = [], []
        with patch.dict(sys.modules, {"raven_ears_plus": fake_audio, "librosa": SimpleNamespace()}), \
             patch.dict("credits.CONFIG"), patch.dict("credits._MEMO"), patch.dict("lrclib.CONFIG"):
            for capture in (None, captured.append):
                with redirect_stdout(io.StringIO()) as out:
                    music.main([str(self.audio), "--no-credits", "--no-lyrics-lookup"], capture=capture)
                outputs.append(out.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(captured[0]["duration"], 1)
        self.assertIn("stage_status", captured[0])

    def test_singer_capture_keeps_the_standalone_text_unchanged(self):
        import singer
        captured, outputs = [], []
        with patch.object(singer, "get_stems", return_value=(str(self.root), True)), \
             patch.object(singer.time, "time", return_value=1), \
             patch.dict(sys.modules, {"soundfile": SimpleNamespace(info=lambda _: SimpleNamespace(duration=60))}), \
             patch.dict("credits.CONFIG"), patch.dict("credits._MEMO"), patch.dict("lrclib.CONFIG"):
            for capture in (None, captured.append):
                with redirect_stdout(io.StringIO()) as out:
                    singer.main([str(self.audio), "--no-singer", "--no-meter", "--no-stems-timeline",
                                 "--no-instruments", "--no-credits"], capture=capture)
                outputs.append(out.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(captured[0]["duration_s"], 60)
        self.assertIn("stage_status", captured[0])

    def test_harmony_capture_keeps_the_standalone_text_unchanged_and_runs_once(self):
        import harmony
        captured, outputs = [], []
        with patch.object(harmony, "run", return_value={}) as run, \
             patch.object(harmony, "flag_contrast"), \
             patch.object(harmony, "report", return_value="invented harmony card"), \
             patch.object(harmony, "to_json", return_value={"duration": 60}) as to_json, \
             patch.dict("credits.CONFIG"), patch.dict("credits._MEMO"), patch.dict("lrclib.CONFIG"):
            for capture in (None, captured.append):
                with redirect_stdout(io.StringIO()) as out:
                    harmony.main([str(self.audio), "--no-credits", "--no-lyrics-lookup"], capture=capture)
                outputs.append(out.getvalue())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(to_json.call_count, 1)
        self.assertEqual(captured, [{"duration": 60}])


def run_tests(check):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ListenTests)
    result = unittest.TestResult()
    suite.run(result)
    problems = {str(test): text for test, text in result.failures + result.errors}
    check("listen once: invented cards, reuse, refresh, failures and isolation",
          result.wasSuccessful(), "\n".join(problems.values()))


if __name__ == "__main__":
    unittest.main()
