"""Listening sheets from public saved-card shapes; no inference or downloads."""
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import conversation_sheet as C
import listen as L
import listening_sheet as S
import raven_ears_plus as R
from test_cpu_profile import card_fixture

ROOT = Path(__file__).resolve().parent


class PictureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", ROOT))
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        environment = patch.dict(os.environ, {"RAVEN_EARS_HOME": str(self.root / "home"),
                                               "RAVEN_EARS_PICTURE": "",
                                               "RAVEN_EARS_NO_BASELINE": "1",
                                               "RAVEN_STEMS_CACHE": str(self.root / "stems")})
        environment.start()
        self.addCleanup(environment.stop)
        self.output = self.root / "sheet.png"
        self.song = json.loads((ROOT / "fixtures/uhuru-title-song.json").read_text())
        self.decided = L.render(self.song, "summary")[0]
        self.full = L.full_text(self.song)
        self.spectrum = (np.full((128, 100), -40.), np.geomspace(30, 11025, 129), 209.)

    def assert_png(self, output):
        from PIL import Image
        with Image.open(output) as image:
            self.assertEqual(image.width, 2400)
            self.assertGreaterEqual(image.height, 1200)
            self.assertGreater(np.asarray(image).std(), 15)

    def seed_baseline(self, speaker, pitches):
        path = R.baseline_path(speaker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(dict(pitch_hz=pitch, range_st=6., ts=i, dur_s=12.)) + "\n"
                                for i, pitch in enumerate(pitches)))
        return path

    def test_whisper_prompt_names_stay_out_of_card_json_and_sheet(self):
        from matplotlib.text import Text
        import lyrics_shared as words
        import singer

        names = self.root / "names.txt"
        names.write_text("Zyxwvut\nZyxwvut\nVan Example\n", encoding="utf-8")
        prompts = []

        class Whisper:
            def transcribe(self, samples, initial_prompt=None, **kwargs):
                prompts.append(initial_prompt)
                return iter([SimpleNamespace(start=0., end=1., text="Zyxwvut, Zyxwvut",
                                             avg_logprob=-.7, no_speech_prob=.1, words=[])]), None

        notes = []
        with patch.object(words, "NAMES_FILE", names), patch.object(words, "EXTRA_NAMES", []), \
             patch.object(words, "_load_model", return_value=(Whisper(), "cpu", "int8")), \
             patch.object(R, "release_models"), patch.object(R, "_preload_cuda12"), \
             patch.object(R, "load_audio", return_value=np.ones(R.SR)), \
             patch.object(singer, "get_stems", return_value=(self.root, True)), \
             patch.object(singer, "load_mono", return_value=(np.ones(R.SR), R.SR)):
            words.transcribe(self.root / "song.wav", notes)
        self.assertEqual(prompts, ["Zyxwvut, Van Example."])
        self.assertIn("lyrics: names list fed to Whisper (2 names)", notes)
        self.assertNotIn("Zyxwvut", str(notes))

        # Old text cards wrapped the preview while JSON kept it on one line.
        legacy = "lyrics: names list fed to Whisper (" + "Van Example, " * 6 + "Zyxwvut)"
        from stage_status import wrap_note
        card = self.song["cards"]["music"]
        card["text"] += "\n".join("  NOTE: " + n for n in notes) + "\n"
        card["text"] += "\n".join(wrap_note("  NOTE: " + legacy + " [shared cache]")) + "\n"
        card["json"]["notes"] = notes + [legacy]
        song_file = ROOT / "fixtures/neutral-voiced.wav"
        sha = L.file_hash(song_file)
        self.song["manifest"]["file_sha256"] = sha
        bundle = L.store_home() / sha / "song.json"
        bundle.parent.mkdir(parents=True)
        bundle.write_text(json.dumps(self.song, ensure_ascii=False), encoding="utf-8")
        serialized, _ = L.execute(L.parser().parse_args([str(song_file), "--json"]))
        self.assertNotIn("Zyxwvut", serialized)
        self.assertIn("(2 names)", serialized)
        self.assertIn("(count unavailable)", serialized)
        for view in L.VIEWS:
            self.assertNotIn("Zyxwvut", L.render(self.song, view)[0], view)

        decided = L.render(self.song, "first-listen")[0]
        full = L.render(self.song, "full")[0]
        self.assertIn("(2 names)", full)
        self.assertIn("(count unavailable) [shared cache]", full)
        import modes
        self.assertNotIn("Zyxwvut", modes.render(card["text"]))
        save = S._save

        def inspect(figure, output):
            figure.canvas.draw()
            labels = [artist.get_text() for artist in figure.findobj(match=Text)]
            self.assertTrue(any("Lyrics" in label for label in labels))
            self.assertNotIn("Zyxwvut", "\n".join(labels))
            return save(figure, output)

        with patch.object(S, "_spectrum", return_value=self.spectrum), patch.object(S, "_save", side_effect=inspect) as checked:
            S.song_sheet(song_file, self.song, decided, full, self.output)
            checked.assert_called_once()
        self.assert_png(self.output)

    def test_song_fixture_lanes_and_determinism(self):
        import matplotlib
        from matplotlib.figure import Figure
        snapshots = []
        finish = S._finish
        def inspect(figure, axes, *args):
            snapshots.append(([ax.get_ylabel() for ax in axes],
                              [text.get_text() for ax in axes for text in ax.texts]))
            return finish(figure, axes, *args)
        with patch.object(S, "_spectrum", return_value=self.spectrum), patch.object(S, "_finish", side_effect=inspect):
            S.song_sheet(ROOT / "fixtures/neutral-voiced.wav", self.song, self.decided, self.full, self.output)
            first = self.output.read_bytes()
            with matplotlib.rc_context({"savefig.bbox": "tight", "font.family": "monospace", "axes.facecolor": "black"}):
                S.song_sheet(ROOT / "fixtures/neutral-voiced.wav", self.song, self.decided, self.full, self.output)
        self.assertEqual(first, self.output.read_bytes())
        self.assert_png(self.output)
        self.assertEqual(snapshots[0][0], ["Full mix (Hz)", "Sounds\nalike",
                                         "Card events", "Vocal stem\n(Hz)", "Lyrics"])
        self.assertTrue(any(text.startswith("≈ ") for text in snapshots[0][1]))
        self.assertTrue(any(label.isdigit() for label in snapshots[0][1]))
        self.assertEqual(Figure().axes, [])

    def test_contrast_strip_matches_card_disagreement_on_saved_readings(self):
        from matplotlib.colors import to_hex
        lyrics = S.song_evidence(self.song, self.decided, self.full)["lyrics"]
        by_time = {r.t: r for r in lyrics}
        harmony = self.song["cards"]["harmony"]["json"]
        harmony["contrast"] = [
            dict(t=t + .12, text=by_time[t].text, words=w, music=m, lead="activity", flag=None)
            for t, (w, m) in zip(sorted(by_time), ((.4, -.4), (-.4, .4), (.8, .6), (.39, -.65)))]
        first, second, *_ = harmony["contrast"]
        start = first["t"]
        harmony["lyrics_aligned"] = dict(lines=[dict(t=start, end=start + 1.6, text=first["text"])])
        harmony["contrast"].append(dict(t=start, text="Different words", words=.9, music=-.9))
        finish = S._finish

        def inspect(figure, axes, *args):
            self.assertEqual([ax.get_ylabel() for ax in axes[-4:]],
                             ["Card events", "Contrast", "Vocal stem\n(Hz)", "Lyrics"])
            panel = axes[-3]
            zero, words, lift = panel.lines  # the start-only disagreement is now a 1 s band, not a line
            self.assertEqual(list(zero.get_ydata()), [0, 0])
            lo, hi = panel.get_ylim()
            self.assertEqual(lo, -hi)
            np.testing.assert_allclose(words.get_ydata()[::3], [.4, -.4, .8, .39])
            np.testing.assert_allclose(lift.get_ydata()[::3], [-.4, .4, .6, -.65])
            self.assertTrue(np.isnan(words.get_ydata()[2::3]).all())
            self.assertEqual(lift.get_markevery(), [3, 6, 9])  # Older rows have starts only.
            spans = [p.get_path().transformed(p.get_patch_transform()).get_extents() for p in panel.patches]
            for p in panel.patches:
                self.assertEqual(to_hex(p.get_facecolor()), S.DISAGREEMENT.lower())
            # The timed row keeps its true 1.6 s span; a start-only row gets the 1 s minimum band.
            self.assertTrue(any(abs(sp.x0 - start) < 1e-6 and abs(sp.x1 - (start + 1.6)) < 1e-6 for sp in spans))
            self.assertTrue(all(sp.x1 - sp.x0 >= S.CONTRAST_MIN_BAND_S - 1e-6 for sp in spans))
            self.assertTrue(any(abs((sp.x0 + sp.x1) / 2 - second["t"]) < 1e-6 for sp in spans))
            self.assertEqual([t.get_text() for t in panel.legend_.texts],
                             ["words sentiment", "song-relative music lift", "words and music disagree"])
            return finish(figure, axes, *args)

        decided = L.render(self.song, "first-listen")[0]
        self.assertEqual(decided.count("they disagree here"), 2)
        with patch.object(S, "_spectrum", return_value=self.spectrum), patch.object(S, "_finish", side_effect=inspect):
            S.song_sheet(Path("unused"), self.song, decided, self.full, self.output)
        self.assert_png(self.output)
        with patch.dict(harmony, {"stage_status": [
                dict(stage="word / music contrast", status="failed", kind="audio")]}):
            self.assertEqual(S.song_evidence(self.song, self.decided, self.full)["contrast"], [])
            self.assertNotIn("they disagree here", L.render(self.song, "first-listen")[0])
        with patch.dict(harmony, {"lyrics_source": dict(source="LRCLIB", status="instrumental")}):
            self.assertEqual(S.song_evidence(self.song, self.decided, self.full)["contrast"], [])
            self.assertNotIn("they disagree here", L.render(self.song, "first-listen")[0])

    def test_summary_song_cards_keep_original_bytes(self):
        # Digests measured with HEAD's listen.py before this change, using the
        # shipped sanitised real bundles. No renderer or audio decode is needed.
        for name, digest in (("first-listen-song", "b50c5bdde7e959ed732fecfa5caeafe37dda80eac7ed2ba68967c102de034ae5"),
                             ("uhuru-title-song", "419c5c0389c711251f5e40ac1108b824807f6c0f7ea5fcb365deb70ff95fcacf")):
            song = json.loads((ROOT / "fixtures" / (name + ".json")).read_text())
            with patch.object(S, "song_sheet", side_effect=AssertionError("picture is off")):
                card = L.render(song, "summary")[0]
            self.assertEqual(hashlib.sha256(card.encode()).hexdigest(), digest)

    def test_phrase_starts_use_saved_words_and_inherit_marks(self):
        import slim
        # Real Roommates row/word shapes and times; the restart used to be
        # hidden at the end of a 14-word card line rounded to [0:30].
        text = "that I could catch you in the right light I only want the beginning,"
        times = [29.84, 30.12, 30.4, 30.62, 31.08, 31.34, 31.66,
                 31.84, 32.34, 34.18, 34.74, 35.04, 35.26, 35.4, 35.66]
        words = [[w, a, b] for w, a, b in zip(text.split(), times, times[1:])]
        words[8][2] = 33.36
        row = slim.Lyric(30, text + " (unsure: Whisper doubtful)", True)
        before = copy.deepcopy(words)
        phrases = S._lyric_phrases([row], words)
        self.assertEqual([p.t for p in phrases], [29.84, 34.18])
        self.assertTrue(phrases[1].text.startswith("I only want the beginning,"))
        self.assertTrue(all(p.approx and "(unsure: Whisper doubtful)" in p.text for p in phrases))
        self.assertEqual(words, before)
        lines = S._lyric_phrases([row], words, line_starts=True)
        self.assertEqual([p.t for p in lines], [29.84, 34.18])
        self.assertTrue(lines[1].text.startswith("I only want the beginning,"))
        # The capitalised restart also works when Whisper spans the pause.
        words[8][2] = 34.18
        self.assertEqual([p.t for p in S._lyric_phrases([row], words)], [29.84, 34.18])
        # A line restart needs a measured pause, not just capitalisation.
        self.assertEqual(len(S._lyric_phrases([row], words, line_starts=True)), 1)
        # A wrapped line can have only two words before the same restart.
        short = slim.Lyric(31.84, "right light I only want the beginning,")
        self.assertEqual([p.t for p in S._lyric_phrases([short], words[7:])], [31.84, 34.18])
        words[8][2] = 33.88
        self.assertEqual([p.t for p in S._lyric_phrases([short], words[7:], line_starts=True)], [31.84, 34.18])
        words[8][2] = 33.881
        self.assertEqual(len(S._lyric_phrases([short], words[7:], line_starts=True)), 1)

    def test_phrase_punctuation_pauses_and_safe_fallback(self):
        import slim
        row = slim.Lyric(10, "hold me, stay here we wait")
        words = [["hold", 10, 10.2], ["me,", 10.2, 10.4], ["stay", 10.4, 10.6],
                 ["here", 10.6, 10.8], ["we", 11.1, 11.3], ["wait", 11.3, 11.5]]
        self.assertEqual([p.t for p in S._lyric_phrases([row], words)], [10, 10.4, 11.1])
        self.assertEqual([p.t for p in S._lyric_phrases([row], words, line_starts=True)], [10])
        for other in (None, [], [["different", 10, 11]],
                      [["hold", float("nan"), 11]], [["hold", 11, 10]], [["hold", 10]]):
            with self.subTest(words=other):
                self.assertEqual(S._lyric_phrases([row], other), [row])
        # Matching words on a different chorus cannot retime this card row.
        self.assertEqual(S._lyric_phrases([row], [[w, a + 60, b + 60] for w, a, b in words]), [row])
        self.song["cards"]["singer"]["json"]["stage_status"] = [{"stage": "words", "status": "failed"}]
        evidence = S.song_evidence(self.song, self.decided, self.full)
        self.assertEqual(evidence["phrases"], evidence["lyrics"])
        self.assertEqual(evidence["lyric_lines"], evidence["lyrics"])

    def test_micro_phrases_merge_for_labels_keep_real_black_dog_ticks(self):
        import slim
        # Saved Black Dog word boundaries, including the leading capitalised
        # title words that formerly each consumed their own label slot.
        row = slim.Lyric(45, "In The Black Dog", True)
        words = [["in", 44.38, 44.94], ["the", 44.94, 45.04],
                 ["black", 45.04, 45.46], ["dog", 45.46, 45.94]]
        phrases = S._lyric_phrases([row], words)
        before = copy.deepcopy(phrases)
        merged = S._merge_micro_phrases(phrases)
        self.assertEqual(merged, [slim.Lyric(44.38, "In The Black Dog", True, 45.94)])
        self.assertEqual(S._lyric_phrases([row], words, line_starts=True), merged)
        self.assertEqual(S._merge_micro_phrases([slim.Lyric(1, "In"), slim.Lyric(1.6, "The")]),
                         [slim.Lyric(1, "In"), slim.Lyric(1.6, "The")])
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8), dpi=150)
            Canvas(figure)
            ax = figure.subplots()
            S._phrase_labels(ax, phrases, 50, lines=merged)
            self.assertEqual([t.get_text() for t in ax.texts], ["In The Black Dog"])
            self.assertEqual([line.get_xdata()[0] for line in ax.lines], [44.38, 44.94, 45.04, 45.46])
        self.assertEqual(phrases, before)

    def test_line_labels_phrase_ticks_and_minimum_two_whole_words(self):
        import slim
        row = slim.Lyric(34, "I only want the beginning, I don't want the end", True)
        words = [["I", 34.22, 34.78], ["only", 34.78, 35.1], ["want", 35.1, 35.32],
                 ["the", 35.32, 35.44], ["beginning,", 35.44, 35.74], ["I", 35.88, 36.08],
                 ["don't", 36.08, 36.26], ["want", 36.26, 36.36], ["the", 36.36, 36.52], ["end", 36.52, 36.88]]
        phrases = S._lyric_phrases([row], words)
        lines = S._lyric_phrases([row], words, line_starts=True)
        self.assertEqual([p.t for p in phrases], [34.22, 35.88])
        self.assertEqual([p.t for p in lines], [34.22])
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8), dpi=150)
            Canvas(figure)
            ax = figure.subplots()
            S._phrase_labels(ax, phrases, 50, lines=lines)
            self.assertEqual(len(ax.texts), 1)
            self.assertTrue(ax.texts[0].get_text().startswith("I only want the beginning"))
            self.assertEqual([line.get_xdata()[0] for line in ax.lines], [34.22, 35.88])
            artist = figure.text(0, 0, "", fontsize=11)
            renderer = figure.canvas.get_renderer()
            self.assertEqual(S._fit_text(artist, "I only want", 30, renderer, min_words=2), "")
            self.assertEqual(S._fit_text(artist, "Only", 300, renderer, min_words=2), "")
            self.assertEqual(S._fit_text(artist, "Only ...", 300, renderer, min_words=2), "")
            artist.set_text("I only…")
            width = artist.get_window_extent(renderer).width + 1
            self.assertEqual(S._fit_text(artist, "I only want", width, renderer, min_words=2), "I only…")
            # Real neighbouring card lines and full studio duration: an empty
            # lane alone would miss the former three-row chorus regression.
            neighbours = [slim.Lyric(t, text, True) for t, text in (
                (25, "But life is lifing and pressure is pressuring me"),
                (28.92, "Oh, I wish that I could catch you in the right light"),
                (36.88, 'Want the part where you say, "Goddamn"'),
                (39, "Back of the dive bar, giving you head"),
                (41, "Then sneak home late, wake up your roommates"),
                (44, "I want the highlights, ten out of ten (Ah-ah)"),
                (47, "The butterflies from holding your hand (Ah-ah)"),
                (49, "Before we swept us under the bed"),
                (51, "And we became practically roommates"))]
            ax.clear()
            S._phrase_labels(ax, sorted(phrases + neighbours, key=lambda r: r.t), 172.8,
                             lines=sorted(lines + neighbours, key=lambda r: r.t))
            chorus = next(t for t in ax.texts if t.get_position()[0] == 34.22)
            self.assertTrue(chorus.get_text().startswith("I only want the beginning"))

    def test_approx_hoisted_only_when_all_lyrics_share_it(self):
        import slim
        context, Figure, Canvas = S._plotting()
        with context:
            for approx, title, first in ((True, "Lyrics ≈", "Keep these words"),
                                         (False, "Lyrics", "≈ Keep these words")):
                figure = Figure(figsize=(16, 8), dpi=150)
                Canvas(figure)
                ax = figure.subplots()
                rows = [slim.Lyric(1, "Keep these words", True),
                        slim.Lyric(8, "Keep this qualifier (unsure: supplied analysis)", approx)]
                S._phrase_labels(ax, rows, 20)
                self.assertEqual(ax.get_ylabel(), title)
                self.assertEqual(ax.texts[0].get_text(), first)
                self.assertTrue(ax.texts[1].get_text().startswith("(unsure) Keep"))
                self.assertFalse(ax.texts[1].get_text().startswith("≈"))

    def test_key_never_includes_lyrics_even_if_an_old_caller_supplies_overflow(self):
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8), dpi=150)
            Canvas(figure)
            figure._sheet_key = S._numbered_evidence([(1, 1, "entrance", False)], [], [], 20)
            figure._sheet_lyric_overflow = ["Lyric 0:44.38: ≈ In"]
            blocks, _ = S._bottom_key(figure)
            text = "\n".join(artist.get_text() for artist, _ in blocks)
            self.assertIn("entrance", text)
            self.assertNotIn("Lyric", text)

    def test_phrase_labels_preserve_starts_without_overlap_at_1600(self):
        import slim
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(1600 / 150, 8), dpi=150)
            Canvas(figure)
            ax = figure.subplots()
            phrases = [slim.Lyric(t, f"I only want opening {i} (unsure: supplied analysis)", True)
                       for i, t in enumerate((0, .03, .06, .09, .12, .15, .18, 8, 9.8, 9.9))]
            lanes = S._phrase_labels(ax, phrases, 10)
            self.assertGreaterEqual(lanes, 3)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            labels = [t for t in ax.texts if t.get_text()]
            boxes = [t.get_window_extent(renderer) for t in labels]
            self.assertTrue(all(t.get_text().startswith("(unsure) I only") for t in labels))
            self.assertTrue(all(len(t.get_text().removeprefix("(unsure) ").split()) >= 2 for t in labels))
            self.assertFalse(hasattr(figure, "_sheet_lyric_overflow"))
            ticks = [line.get_xdata()[0] for line in ax.lines if line.get_marker() == "|"]
            self.assertEqual(ticks, [p.t for p in phrases])
            for i, box in enumerate(boxes):
                self.assertGreaterEqual(box.x0, ax.bbox.x0 - .01)
                self.assertLessEqual(box.x1, ax.bbox.x1 + .01)
                for other in boxes[i + 1:]:
                    self.assertFalse(box.overlaps(other))

    def test_repeated_lines_collapse_but_keep_every_tick_and_uncertainty(self):
        import slim
        phrases = [slim.Lyric(1, "Eeh-eeh-eeh-eeh, eeh-eeh-eeh", True, 3),
                   slim.Lyric(4, "Eeh eeh eeh eeh, eeh eeh eeh (unsure: quiet)", False, 7),
                   slim.Lyric(9, "We could just kiss", False),
                   slim.Lyric(12, "We would just kiss", False),
                   slim.Lyric(15, "Eeh-eeh-eeh-eeh, eeh-eeh-eeh", True)]
        before = copy.deepcopy(phrases)
        groups = S._collapsed_phrases(phrases, 20)
        self.assertEqual([len(g["rows"]) for g in groups], [2, 1, 1, 1])
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8), dpi=150)
            Canvas(figure)
            ax = figure.subplots()
            S._phrase_labels(ax, phrases, 20)
            self.assertIn("≈ (unsure) Eeh", ax.texts[0].get_text())
            self.assertTrue(ax.texts[0].get_text().endswith(" ×2"))
            ticks = [line.get_xdata()[0] for line in ax.lines if len(set(line.get_xdata())) == 1]
            self.assertEqual(ticks, [1, 4, 9, 12, 15])
            self.assertTrue(any(list(line.get_xdata()) == [1, 7] for line in ax.lines))
        self.assertEqual(phrases, before)

    def test_long_label_shortening_keeps_start_and_marks(self):
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8), dpi=150)
            Canvas(figure)
            artist = figure.text(0, 0, "", fontsize=11)
            renderer = figure.canvas.get_renderer()
            value = S._fit_text(artist, "We should just kiss like real people do", 330, renderer, "≈ (unsure) ")
            self.assertTrue(value.startswith("≈ (unsure) We should just"))
            self.assertTrue(value.endswith("…"))
            artist.set_text(value)
            self.assertLessEqual(artist.get_window_extent(renderer).width, 330)

    def test_repeated_lyrics_go_grey_after_grouping_and_truncation(self):
        import slim
        line = "I only want the beginning and all the little things that we used to do together"
        phrases = [slim.Lyric(1, line + " (oh-oh)"), slim.Lyric(3, line + " (Oh oh)!"),
                   slim.Lyric(22, line.replace("together", "apart")),
                   slim.Lyric(48, line + " (when we met)"),
                   slim.Lyric(75, line.upper()), slim.Lyric(78, line + "."),
                   slim.Lyric(89, line + " (yeah)")]
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8), dpi=150)
            Canvas(figure)
            ax = figure.subplots()
            self.assertEqual(S._phrase_labels(ax, phrases, 100), 3)
            new_group, different, sung_aside, repeat_group, repeat_line = ax.texts
            self.assertTrue(new_group.get_text().endswith(" ×2"))
            self.assertEqual([t.get_color() for t in (new_group, different, sung_aside)], [S.INK] * 3)
            self.assertTrue(repeat_group.get_text().endswith("… ×2"))
            self.assertTrue(repeat_line.get_text().endswith("…"))
            self.assertEqual([t.get_color() for t in (repeat_group, repeat_line)], [S.REPEATED_LYRIC] * 2)
            self.assertEqual([t.get_position()[0] for t in ax.texts], [1, 22, 48, 75, 89])
            ticks = [r.get_xdata()[0] for r in ax.lines if len(set(r.get_xdata())) == 1]
            self.assertEqual(ticks, [r.t for r in phrases])

    def test_numbering_is_in_time_order_across_lanes_and_keeps_source(self):
        rows = S._numbered_evidence([(10, 11, "candidate: timbre shifts sharply (+4 dB)", True),
                                     (2, 2, "drop-out: the level falls 7 dB", False)],
                                    [(5, 5.2, "vocal-stem silence", False)],
                                    [(3, 3, "candidate: bass arrives", False)], 20)
        self.assertEqual([r["number"] for r in rows], [1, 2, 3, 4])
        self.assertEqual([r["start"] for r in rows], [2, 3, 5, 10])
        self.assertEqual(rows[1]["kind"], "by ear")
        self.assertEqual(rows[1]["label"], "candidate: bass arrives")
        self.assertEqual(rows[0]["label"], "drop-out: level falls 7 dB")
        self.assertTrue(rows[-1]["likely"])
        self.assertEqual(rows[-1]["label"], "timbre shifts sharply (+4 dB)")
        self.assertEqual(S._key_clock(5.2), "0:05.2")

    def test_dense_numbered_markers_and_bottom_key_do_not_overlap(self):
        rows = S._numbered_evidence([(1 + i % 2 * .01, 1.1, "candidate: " + "wide label " * 12, bool(i % 2))
                                     for i in range(12)], [], [(1, 1, "entrance", False)], 10)
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8), dpi=150)
            Canvas(figure)
            ax = figure.subplots()
            lanes = S._event_markers(ax, rows, 10)
            self.assertEqual(lanes, len(rows))
            figure._sheet_key = rows
            S._header(figure, "Dense", "KEY: unchanged", "Dashed or light = likely", plot_height=4)
            figure.canvas.draw()
            labels = list(ax.texts) + list(figure.texts)
            boxes = [t.get_window_extent() for t in labels if t.get_text()]
            for i, box in enumerate(boxes):
                self.assertGreaterEqual(box.x0, 0)
                self.assertLessEqual(box.x1, figure.bbox.width)
                for other in boxes[i + 1:]:
                    self.assertFalse(box.overlaps(other))
            key = "\n".join(t.get_text() for t in figure.texts[3:])
            self.assertIn("by ear: entrance", key)
            self.assertTrue(all(len(line) <= 60 for line in key.splitlines()))
            self.assertEqual(ax.lines[0].get_linestyle(), "-")
            self.assertTrue(any(line.get_linestyle() == "--" and line.get_alpha() < 1 for line in ax.lines))

    def test_delete_removes_only_owned_picture_sidecars(self):
        home = self.root / "songs"
        path = home / self.song["manifest"]["file_sha256"] / "song.json"
        L.save(path, self.song)
        for name in ("listening-sheet.png", "listening-spectrum.npz", "other.png"):
            path.with_name(name).write_bytes(b"fixture")
        self.assertTrue(L.delete_bundle(home, path))
        self.assertEqual([p.name for p in path.parent.iterdir()], ["other.png"])

    def test_old_bundle_never_invents_pitch_or_frame_rms(self):
        self.song = json.loads((ROOT / "fixtures/first-listen-song.json").read_text())
        decided = L.render(self.song, "summary")[0]
        evidence = S.song_evidence(self.song, decided, L.full_text(self.song))
        self.assertEqual(evidence["pitch"], {})
        self.assertEqual(evidence["curve"], {})
        self.assertEqual(len(evidence["sections"]), 5)
        self.assertGreater(len(evidence["lyrics"]), 10)
        with patch.object(S, "_spectrum", return_value=self.spectrum):
            S.song_sheet(ROOT / "fixtures/neutral-voiced.wav", self.song, decided, L.full_text(self.song), self.output)
        self.assert_png(self.output)

    def test_only_card_decided_keys_ranges_and_qualifiers(self):
        self.song["cards"]["harmony"]["json"]["key_changes"] = [{"t": 1, "key": "F major"}]
        decided = self.decided + "KEY CHANGES: C minor → D minor, somewhere 1:00–1:20 (up a whole tone)\n"
        full = self.full + ("\nTIMELINE\n  1:25 drop-out: level falls 8 dB\n"
                            "  1:30 unclear drop\nCHILL CANDIDATES\n  1:40 likely to land: full band\n"
                            "LYRICS:\n  [1:50] We drop between the lines\n")
        evidence = S.song_evidence(self.song, decided, full)
        self.assertEqual(evidence["keys"][0][:2], (60, 80))
        self.assertNotIn("F major", str(evidence["keys"]))
        self.assertIn((85, 85, "drop-out: level falls 8 dB", False), evidence["events"])
        self.assertIn((100, 100, "likely to land: full band", True), evidence["events"])
        self.assertFalse(any(a in (90, 110) for a, *_ in evidence["events"]))
        evidence = S.song_evidence(self.song, decided.replace("KEY CHANGES:", "KEY CHANGES: unclear"), full)
        self.assertEqual(evidence["keys"], [])
        failed = copy.deepcopy(self.song)
        failed["manifest"]["passes"]["singer"]["status"] = "failed"
        self.assertEqual(S.song_evidence(failed, decided, full)["pitch"], {})

    def test_pitch_holes_and_likely_styles(self):
        context, Figure, Canvas = S._plotting()
        with context:
            figure = Figure()
            Canvas(figure)
            ax = figure.subplots()
            S._pitch(ax, [0, 1, 2, 3], [100, None, 0, 200])
            values = ax.lines[0].get_ydata()
            self.assertTrue(np.isnan(values[1:3]).all())
            self.assertEqual(ax.get_yscale(), "log")
            figure.clear()
            ax = figure.subplots()
            S._labels(ax, [(1, 3, "likely to land", True)], 10)
            self.assertEqual(ax.lines[0].get_linestyle(), "--")
            self.assertLess(ax.lines[0].get_alpha(), 1)

    def test_by_ear_validation_and_independent_ticks(self):
        marks = self.root / "marks.txt"
        marks.write_text("\ufeff0:30.25 human return\n\n0:12 human cut\n", encoding="utf-8")
        rows = S.by_ear_rows(marks, 209)
        self.assertEqual([r[0] for r in rows], [30.25, 12])
        self.song["cards"]["singer"]["json"]["stops"] = {"status": "ran", "events": [
            dict(start_s=34.19, length_s=.76, drop_db=17.7, kind="band cuts, voice holds")]}
        before = copy.deepcopy(self.song)
        finish = S._finish
        def inspect(figure, axes, *args):
            events = next(ax for ax in axes if ax.get_ylabel() == "Card events")
            ticks = [line for line in events.lines if line.get_marker() == "v"]
            self.assertEqual([line.get_xdata()[0] for line in ticks], [12, 30.25])
            self.assertIn("human cut", [r["label"] for r in figure._sheet_key if r["kind"] == "by ear"])
            self.assertIn("section PEAK", [r["label"] for r in figure._sheet_key])
            self.assertTrue(all(t.get_text().isdigit() for t in events.texts))
            stem_ax = next(ax for ax in axes if ax.get_ylabel() == "Cached\nstems")
            np.testing.assert_allclose(stem_ax.lines[0].get_ydata(), [.6, 0])
            self.assertTrue(any(ax.get_ylabel() == "Gaps / cuts" for ax in axes))
            figure.canvas.draw()
            for ax in axes:
                self.assertGreaterEqual(ax.yaxis.label.get_window_extent().x0, 0,
                                        "all-lanes labels must fit inside the PNG")
            return finish(figure, axes, *args)
        stems = [("drums", np.array([.5, 1.5]), np.array([-20, -120]))]
        with patch.object(S, "_spectrum", return_value=self.spectrum), \
             patch.object(S, "_cached_stem_energy", return_value=stems), \
             patch.object(S, "_finish", side_effect=inspect):
            S.song_sheet(Path("unused"), self.song, self.decided, self.full, self.output, by_ear=marks)
            first = self.output.read_bytes()
            S.song_sheet(Path("unused"), self.song, self.decided, self.full, self.output, by_ear=marks)
        self.assertEqual(first, self.output.read_bytes())
        self.assertEqual(self.song, before)
        for bad in ("0:60 impossible", "-1:00 negative", "0:03", "3:30 beyond end", "nan bad"):
            marks.write_text(bad)
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "by ear line 1"):
                S.by_ear_rows(marks, 209)
        # A failed draw preserves the earlier sheet, as well as the text card.
        with patch.object(S, "_spectrum", return_value=self.spectrum):
            self.assertIn("by ear line 1", S.attempt(S.song_sheet, Path("unused"), self.song,
                          self.decided, self.full, self.output, by_ear=marks))
        self.assertEqual(first, self.output.read_bytes())

    def test_gap_projection_uses_real_clock_column_and_preserves_kind(self):
        # TIMELINE follows the shipped saved card: leading clock, then prose.
        full = ("HARMONY: fixture\nTIMELINE\n"
                "   0:12  near-silence for 0.7 s, 65 dB under the song's own level; then it comes back\n"
                "   0:20  sudden dip for 0.5 s, 15 dB under the song's own level\n"
                "   0:30  drop-out: the level falls 9 dB and stays down; the band thins\n"
                "LYRICS\n   0:40  drop-out: words, not a measurement\n")
        self.assertEqual(S._song_gaps(self.song, full),
                         [(12, 12.7, "near-silence", False),
                          (30, 30, "drop-out (end not measured)", False)])
        # Saved event fields follow test_stops' measured Take Five row shapes.
        known = dict(start_s=34.19, length_s=.76, drop_db=17.7, kind="band cuts, voice holds")
        result = dict(status="ran", events=[known, dict(known, start_s=50.12, kind="uncertain")])
        self.song["cards"]["singer"]["json"]["stops"] = result
        self.song["cards"]["singer"]["json"]["singer"] = {"data": {"gaps": {"silent": [[60, 61]]}}}
        before = copy.deepcopy(self.song)
        gaps = S._song_gaps(self.song, full)
        self.assertEqual(gaps[:2], [(34.19, 34.19 + .76, "band cuts, voice holds", False),
                                   (60, 61, "vocal-stem silence", False)])
        self.assertEqual(self.song, before)
        with patch("stops.refresh", return_value=dict(status="ran", events=[dict(known, start_s=75)])) as refresh:
            self.assertEqual(S._song_gaps(self.song, "")[0][:2], (75, 75.76))
        refresh.assert_called_once_with(result)
        self.song["cards"]["singer"]["json"]["singer"] = "instrumental"
        self.assertFalse(any(label == "vocal-stem silence" for _, _, label, _ in S._song_gaps(self.song, "")))
        self.song["manifest"]["passes"]["singer"]["status"] = "failed"
        self.song["manifest"]["passes"]["harmony"]["status"] = "failed"
        self.assertEqual(S._song_gaps(self.song, full), [])

    def test_repeat_lane_matches_nonadjacent_material_not_level_or_sustain(self):
        x = np.arange(128) / 128 * 2 * np.pi
        motifs = [np.sin(i * x) * 10 - 35 for i in range(1, 6)]
        # Real spectrum shape: frequency rows, equal-time dB columns. These
        # motifs are synthetic; identical copies vary only in overall level.
        parts = [motifs[0], motifs[1], motifs[2], motifs[0] - 5,
                 motifs[1] + 3, motifs[3], np.full(128, -90), motifs[4]]
        power = np.repeat(np.column_stack(parts), 10, axis=1)
        before = power.copy()
        edges, groups, strength = S._repeat_groups(power, 64)
        np.testing.assert_allclose(edges, np.arange(9) * 8)
        self.assertGreaterEqual(groups[0], 0)
        self.assertEqual(groups[0], groups[3])
        self.assertEqual(groups[1], groups[4])
        self.assertNotEqual(groups[0], groups[1])
        np.testing.assert_array_equal(groups[[2, 5, 6, 7]], -1)
        np.testing.assert_allclose(strength[[0, 1, 3, 4]], 1)
        np.testing.assert_array_equal(power, before)
        for stationary in (np.full((128, 80), -40.),
                           np.column_stack([motifs[0]] * 40 + [motifs[1]] * 40),
                           np.full((128, 80), -90.), np.zeros((128, 1))):
            self.assertTrue((S._repeat_groups(stationary, 64)[1] == -1).all())
        self.assertLessEqual(len(S._repeat_groups(np.tile(power, 100), 10000)[1]), 480)

    def test_repeat_groups_do_not_chain_weak_matches_and_render_pale(self):
        x = np.arange(128) / 128 * 2 * np.pi
        a, b = np.sin(x), np.cos(x)
        shapes = [np.cos(angle) * a + np.sin(angle) * b for angle in (0, np.deg2rad(25), np.pi / 3)]
        parts = [item for shape in shapes for item in (shape, -shape)] * 2
        power = np.repeat(np.column_stack(parts) * 10 - 35, 10, axis=1)
        edges, groups, strength = S._repeat_groups(power, 96)
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[4])  # A~B, B~C does not mean A~C.
        self.assertAlmostEqual(strength[0], np.cos(np.deg2rad(25)))
        finish = S._finish
        def inspect(figure, axes, *args):
            lane = axes[1]
            self.assertEqual(len(lane.images), 0, "no squashed matrix")
            self.assertEqual(len(lane.patches), len(groups))
            np.testing.assert_allclose(lane.patches[0].get_facecolor(), lane.patches[6].get_facecolor())
            self.assertLess(lane.patches[0].get_alpha(), .95)
            self.assertIn("pale = a little alike", args[-1])
            self.assertIn("pale <0.92", figure._sheet_notes)
            return finish(figure, axes, *args)
        with patch.object(S, "_spectrum", return_value=(power, self.spectrum[1], 96)), \
             patch.object(S, "_finish", side_effect=inspect):
            S.song_sheet(Path("unused"), self.song, self.decided, self.full, self.output)
            first = self.output.read_bytes()
            S.song_sheet(Path("unused"), self.song, self.decided, self.full, self.output)
        self.assertEqual(first, self.output.read_bytes())

    def test_long_header_grows_without_overlap_and_keeps_full_decisions(self):
        from matplotlib.figure import Figure
        long_tempo = "152 BPM (librosa estimate) · steady 1.00 · " + "beat tracker heard 76.9 BPM (bars steady 0.58) · " * 45
        decided = self.decided + "\nTEMPO: " + long_tempo + "\nMETER: 4/4 (candidate)\n"
        self.song["manifest"]["options"].update(artist="Long title", title="W" * 350)
        savefig = Figure.savefig
        def inspect(figure, *args, **kwargs):
            figure.canvas.draw()
            blocks = [t.get_window_extent() for t in figure.texts]
            self.assertGreater(blocks[0].y0, blocks[1].y1)
            self.assertGreater(blocks[1].y0, blocks[2].y1)
            self.assertGreater(blocks[2].y0, figure.axes[0].bbox.y1)
            for bounds in blocks:
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, figure.bbox.width)
            self.assertGreater(figure.get_figheight(), S.HEIGHT / 150)
            quoted = figure.texts[1].get_text().replace("\n", " ")
            self.assertIn(long_tempo.strip(), quoted)
            self.assertIn("METER: 4/4 (candidate)", quoted)
            return savefig(figure, *args, **kwargs)
        with patch.object(S, "_spectrum", return_value=self.spectrum), \
             patch.object(Figure, "savefig", inspect):
            S.song_sheet(Path("unused"), self.song, decided, self.full, self.output)
        from PIL import Image
        with Image.open(self.output) as image:
            self.assertEqual(image.width, 2400)
            self.assertGreater(image.height, 1200)

    def test_short_section_labels_stay_inside_their_own_boundaries(self):
        from matplotlib.figure import Figure
        # Same start/end/label/dB row shape as the real Black Dog sections.
        sections = [dict(start=a, end=b, label=label, db=-24, peak=False) for a, b, label in
                    ((0, 2, "quiet intro"), (2, 2.05, "mid"), (2.05, 3, "PEAK loud"),
                     (3, 7, "loud"), (7, 100, "quiet outro"))]
        context, _, Canvas = S._plotting()
        with context:
            figure = Figure(figsize=(16, 8))
            Canvas(figure)
            ax = S._axes(figure, [1, 1])[0]
            ax.set_xlim(0, 100)
            S._section_labels(ax, sections, 100)
            figure.canvas.draw()
            boxes = []
            for text, section in zip(ax.texts, sections):
                if not text.get_text():
                    continue
                box = text.get_window_extent()
                lo, hi = ax.transData.transform([(section["start"], 0), (section["end"], 0)])[:, 0]
                self.assertGreater(box.x0, lo)
                self.assertLess(box.x1, hi)
                self.assertFalse(any(box.overlaps(other) for other in boxes))
                self.assertAlmostEqual(text.get_bbox_patch().get_alpha(), .65)
                self.assertEqual(text.get_bbox_patch().get_facecolor()[:3], (0., 0., 0.))
                boxes.append(box)
            self.assertEqual(ax.texts[1].get_text(), "")
            self.assertEqual(ax.texts[-1].get_text(), "quiet outro")
            self.assertEqual(len(ax.lines), len(sections), "boundaries remain even without a label")

    def test_cached_stems_read_only_matching_wavs_and_preserve_stereo_energy(self):
        import soundfile as sf
        digest = self.song["manifest"]["file_sha256"]
        root = Path(os.environ["RAVEN_STEMS_CACHE"])
        directory = root / (digest[:16] + "-a0")
        self.assertEqual(S._cached_stem_energy(digest, 2), [])
        self.assertFalse(root.exists())
        directory.mkdir(parents=True)
        signal = np.full(16000, .1)
        signal[8000:] = 0
        sf.write(directory / "drums.wav", np.column_stack((signal, -signal)), 8000, subtype="FLOAT")
        sf.write(directory / "bass.wav", np.zeros(8000), 8000)  # mismatched length
        (directory / "other.wav").write_bytes(b"unreadable")
        six = root / (digest[:16] + "-a0-6s")
        six.mkdir()
        sf.write(six / "guitar.wav", signal, 8000, subtype="FLOAT")
        before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        with patch.object(S.subprocess, "Popen", side_effect=AssertionError("no decode or separation process")):
            curves = S._cached_stem_energy(digest, 2)
        self.assertEqual([name for name, *_ in curves], ["drums", "guitar (6s)"])
        np.testing.assert_allclose(curves[0][2][:10], -20, atol=1e-5)
        np.testing.assert_allclose(curves[0][2][10:], -120)
        self.assertEqual(curves[0][1][0], .05)
        self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
        sf.write(six / "vocals.wav", signal, 8000, subtype="FLOAT")
        self.assertEqual([name for name, *_ in S._cached_stem_energy(digest, 2)],
                         ["drums", "vocals (6s)", "guitar (6s)"])
        self.assertEqual(S._cached_stem_energy("0" * 64, 2), [])
        self.assertEqual(S._cached_stem_energy("../not-a-hash", 2), [])

    def test_by_ear_implies_picture_but_json_management_and_serve_reject_it(self):
        for flag in ("--json", "--history", "--delete", "--delete-all"):
            args = L.parser().parse_args(["unused.wav", "--by-ear", "unused.txt", flag])
            with patch.object(L, "run_pass", side_effect=AssertionError("no analysis")), \
                 self.assertRaisesRegex(ValueError, "song text card"):
                L.execute(args)
        for flag in ("--json", "--serve", "--setup", "--voices"):
            with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), \
                 patch.object(R, "analyze", side_effect=AssertionError("no analysis")), \
                 self.assertRaises(SystemExit) as result:
                R.main(["unused.wav", "--by-ear", "unused.txt", flag])
            self.assertEqual(result.exception.code, 2)

    def test_saved_stops_refresh_from_curves_and_render_gap_spans(self):
        import stops
        from test_stops import Stops
        directory = self.root / "synthetic-stems"
        directory.mkdir()
        fixture = Stops()
        fixture.tmp, fixture.sr = str(directory), 8000
        fixture.stems(voice="falls")
        measured = stops.analyse(directory)
        self.assertEqual(measured["status"], "ran")
        self.assertEqual(measured["events"][0]["kind"], "full silence")
        measured["events"] = []  # Card and picture must refresh from saved energy.
        self.song["cards"]["singer"]["json"]["stops"] = measured
        before = copy.deepcopy(self.song)
        gaps = S._song_gaps(self.song, self.full)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0][0], 3, delta=.03)
        self.assertAlmostEqual(gaps[0][1], 4, delta=.03)
        finish = S._finish
        def inspect(figure, axes, *args):
            ax = next(ax for ax in axes if ax.get_ylabel() == "Gaps / cuts")
            self.assertEqual(len(ax.patches), 1)
            self.assertEqual(ax.patches[0].get_hatch(), "////")
            np.testing.assert_allclose(ax.patches[0].get_bbox().intervalx, gaps[0][:2])
            self.assertEqual([t.get_text() for t in ax.texts], ["1"])
            self.assertIn("full silence", [r["label"] for r in figure._sheet_key])
            return finish(figure, axes, *args)
        with patch.object(S, "_spectrum", return_value=self.spectrum), patch.object(S, "_finish", side_effect=inspect):
            S.song_sheet(Path("unused"), self.song, self.decided, self.full, self.output)
        self.assertEqual(self.song, before)

    def test_spectrum_decodes_neutral_fixture_once_and_invalidates(self):
        cache = self.root / "spectrum.npz"
        actual = S._spectrum(ROOT / "fixtures/neutral-voiced.wav", cache, "fixture-hash")
        self.assertEqual(actual[0].shape[0], 128)
        self.assertLessEqual(actual[0].shape[1], 2400)
        self.assertAlmostEqual(actual[2], 32, places=1)
        with patch.object(S.subprocess, "Popen", side_effect=AssertionError("cache should avoid decoding")):
            cached = S._spectrum(Path("not-needed.wav"), cache, "fixture-hash")
        np.testing.assert_array_equal(actual[0], cached[0])
        with patch.object(S.subprocess, "Popen", side_effect=RuntimeError("cache invalidated")):
            with self.assertRaisesRegex(RuntimeError, "invalidated"):
                S._spectrum(Path("changed.wav"), cache, "new-hash")

    def test_song_cli_saved_view_does_no_pass_and_default_bytes_stay_equal(self):
        audio = self.root / "fixture.wav"
        audio.write_bytes(b"fixture audio identity; spectrum is supplied below")
        song = copy.deepcopy(self.song)
        song["manifest"]["file_sha256"] = L.file_hash(audio)
        with patch.dict(os.environ, {"RAVEN_EARS_HOME": str(self.root / "home")}):
            path = L.store_home() / L.file_hash(audio) / "song.json"
            L.save(path, song)
            before = path.read_bytes()
            args = L.parser().parse_args([str(audio), "--no-picture"])
            with patch.object(L, "run_pass", side_effect=AssertionError("saved view must not listen")), \
                 patch.object(S, "_spectrum", return_value=self.spectrum):
                plain = L.execute(args)[0]
                with patch.object(S, "song_sheet", side_effect=AssertionError("off means no renderer")):
                    self.assertEqual(L.execute(args)[0].encode(), plain.encode())
                args.picture = True
                pictured = L.execute(args)[0]
                footer = f"PICTURE: {path.with_name('listening-sheet.png')}\n"
                import feeling_card
                feeling_footer = feeling_card.picture_line(footer.rstrip('\n')) + '\n'
                self.assertTrue(pictured.endswith(feeling_footer))
                self.assertEqual(pictured.removesuffix(feeling_footer).encode(), plain.encode())
                self.assertTrue(all(len(row) <= 118 for row in pictured.splitlines()))
                self.assertEqual(pictured.count("PICTURE:"), 1)
                self.assertEqual(path.read_bytes(), before)
                for view in ("music", "melody", "full"):
                    args.view = view
                    self.assertIn(footer, L.execute(args)[0])
                marks = self.root / "by-ear.txt"
                marks.write_text("0:32 human entrance\n")
                args.view, args.picture, args.by_ear = "first-listen", None, str(marks)
                annotated = L.execute(args)[0]
                self.assertEqual(annotated.removesuffix(feeling_footer).encode(), plain.encode())
                self.assertEqual(path.read_bytes(), before)
                marks.write_text("invalid annotation")
                failed = L.execute(args)[0]
                self.assertTrue(failed.startswith(plain))
                self.assertIn("PICTURE unavailable: by ear line 1", failed)
                with patch.dict(sys.modules, {"matplotlib": None}):
                    missing = L.execute(args)[0]
                self.assertTrue(missing.endswith("pip install matplotlib to draw the listening sheet\n"))

    def test_frame_rms_survives_worker_scrubber_and_saved_bundle(self):
        # Exercise run_pass -> worker capture -> analysis_basenames -> save,
        # including the opt-in flag that asks music to retain its existing RMS.
        audio = self.root / "fixture.wav"
        audio.write_bytes(b"worker fixture")
        args = L.parser().parse_args([str(audio), "--picture"])
        curve = {"step_s": 512 / 22050, "rms_db": [-28.123, -19.25, -31.]}
        def subprocess_worker(argv, **kwargs):
            with redirect_stdout(io.StringIO()) as output:
                L.worker(argv[3], argv[4], argv[5:])
            return subprocess.CompletedProcess(argv, 0, stdout=output.getvalue(), stderr="")
        import music
        import harmony
        # Run music.main itself; replace only listening/lookup work. The new
        # field must actually be attached at the real capture boundary.
        with patch.dict(os.environ, {"RAVEN_EARS_HOME": str(self.root)}), ExitStack() as stack:
            L.store_home().mkdir()
            stack.enter_context(patch.object(R, "load_audio", return_value=np.zeros(6 * music.SR)))
            stack.enter_context(patch.object(harmony, "music_features", return_value={
                "chroma": [], "rms_db": np.array(curve["rms_db"]), "mfcc": [], "hp_energy": [], "cent": [], "oenv": []}))
            for method in ("analyze_key", "analyze_tempo", "analyze_sections", "analyze_sound", "analyze_vocals",
                           "analyze_vocal_quality", "key_claim_for"):
                stack.enter_context(patch.object(music, method, return_value=None))
            stack.enter_context(patch.object(music, "meter_info_for", return_value=(None, None)))
            stack.enter_context(patch.object(music, "format_card", return_value="MUSIC: fixture\n"))
            stack.enter_context(patch("credits.save_filename_reading"))
            args.no_lyrics = args.no_credits = args.no_lyrics_lookup = True
            with patch.object(L.subprocess, "run", side_effect=subprocess_worker):
                captured = L.run_pass("music", args)
                args.picture = False
                ordinary = L.run_pass("music", args)
                self.assertNotIn("listening_sheet", ordinary["json"])
                self.assertEqual(ordinary["text"], captured["text"])
        self.assertEqual(captured["status"], "ran")
        self.song["cards"]["music"] = captured
        L.save(self.root / "song.json", self.song)
        saved = L.load(self.root / "song.json")
        self.assertEqual(saved["cards"]["music"]["json"]["listening_sheet"], curve)
        args.picture = False
        self.assertNotIn("--_picture-data", L.pass_args("music", args))

    def test_speech_fixture_retains_existing_frames_only_on_request(self):
        with card_fixture() as (audio, *_):
            ordinary = R.analyze(audio, update_baseline=False)
            pictured = R.analyze(audio, update_baseline=False, capture_picture=True)
        expected = (ROOT / "fixtures/speech-default.card.txt").read_bytes()
        self.assertEqual(("\n".join(ordinary["lines"]) + "\n").encode(), expected)
        self.assertEqual(ordinary["lines"], pictured["lines"])
        self.assertNotIn("picture_data", ordinary)
        self.assertNotIn("picture_data", pictured["data"])
        measured = pictured["picture_data"]
        self.assertEqual(set(measured), {"times", "rms_db", "f0"})
        self.assertEqual(len(measured["times"]), len(measured["f0"]))
        finish = S._finish
        def inspect(figure, axes, *args):
            self.assertEqual([ax.get_ylabel() for ax in axes],
                             ["Energy (dB)", "Pitch (Hz)", "Card events", "Words"])
            labels = [text.get_text() for text in axes[2].texts]
            self.assertIn("WORDLESS", labels)
            self.assertIn("pause", labels)
            self.assertIn("Snort", labels)
            return finish(figure, axes, *args)
        with patch.object(S, "_finish", side_effect=inspect):
            S.speech_sheet(pictured["data"], measured, self.output)
        self.assert_png(self.output)
        output = io.StringIO()
        with patch.object(R, "analyze", return_value=pictured), redirect_stdout(output):
            R.main([str(audio), "--quiet", "--picture-out", str(self.output)])
        self.assertEqual(output.getvalue(), expected.decode() + f"PICTURE: {self.output}\n")

    def test_speech_picture_uses_the_card_pitch_threshold_without_teaching_the_baseline(self):
        baseline = self.seed_baseline("voice-a", [500.] + [None] * 55 + [100., 110., 120., 130., 140.])
        original = baseline.read_bytes()
        save = S._save

        def inspect(figure, output):
            pitch = figure.axes[1]
            band, = pitch.patches
            np.testing.assert_allclose(band.get_bbox().intervaly, [90.348, 149.652])
            self.assertLess(band.get_zorder(), pitch.lines[0].get_zorder())
            self.assertLess(band.get_alpha(), .2)
            self.assertLess(pitch.get_ylim()[0], 90.348)
            self.assertIn("pale band: usual median pitch (from 5 notes)", figure.texts[2].get_text())
            return save(figure, output)

        stdout = io.StringIO()
        with card_fixture() as (audio, *_), patch.object(S, "_save", side_effect=inspect), redirect_stdout(stdout):
            R.main([str(audio), "--speaker", "voice-a", "--quiet", "--picture-out", str(self.output)])
        usual = next(line for line in stdout.getvalue().splitlines() if "VS USUAL (n=60)" in line)
        self.assertIn("pitch ↑164 Hz (usual 120)", usual)
        self.assertEqual(baseline.read_bytes(), original)
        self.assert_png(self.output)

    def test_speech_without_enough_baseline_draws_a_reason_and_never_creates_one(self):
        cases = [(None, "no baseline"), ([120.] * 9, "too few baseline notes (9/10)"),
                 ([120.] * 4 + [None] * 6, "fewer than 5 baseline notes with measured pitch")]
        for i, (pitches, reason) in enumerate(cases):
            speaker = f"voice-{i}"
            baseline = R.baseline_path(speaker) if pitches is None else self.seed_baseline(speaker, pitches)
            original = baseline.read_bytes() if pitches is not None else None
            with self.subTest(reason=reason), card_fixture() as (audio, *_):
                result = R.analyze(audio, speaker=speaker, capture_picture=True)
                def inspect(_figure, axes, _duration, output, _title, _subtitle, footer):
                    self.assertFalse(axes[1].patches)
                    self.assertTrue(footer.endswith(f"Usual pitch band unavailable · {reason}"))
                    return output
                with patch.object(S, "_finish", side_effect=inspect):
                    S.speech_sheet(result["data"], result["picture_data"], self.output)
                if original is None:
                    self.assertFalse(baseline.exists())
                else:
                    self.assertEqual(baseline.read_bytes(), original)

    def test_speech_by_ear_and_silence_bands_do_not_relabel_word_pauses(self):
        with card_fixture() as (audio, *_):
            result = R.analyze(audio, update_baseline=False, capture_picture=True)
        data = copy.deepcopy(result["data"])
        data["gaps"]["silent"] = [[20.25, 21.5]]
        marks = self.root / "speech-ear.txt"
        marks.write_text("0:20.5 human silence\n")
        finish = S._finish
        def inspect(figure, axes, *args):
            for ax in axes[:2]:
                self.assertEqual(len(ax.patches), 1)
                np.testing.assert_allclose(ax.patches[0].get_bbox().intervalx, [20.25, 21.5])
            labels = [t.get_text() for t in axes[2].texts]
            self.assertIn("silence", labels)
            self.assertIn("pause", labels)
            self.assertIn("human silence", labels)
            return finish(figure, axes, *args)
        with patch.object(S, "_finish", side_effect=inspect):
            S.speech_sheet(data, result["picture_data"], self.output, by_ear=marks)
        output = io.StringIO()
        with patch.object(R, "analyze", return_value=result) as analyze, redirect_stdout(output):
            R.main([str(audio), "--quiet", "--by-ear", str(marks), "--picture-out", str(self.output)])
        self.assertTrue(analyze.call_args.kwargs["capture_picture"])
        self.assertEqual(output.getvalue(), (ROOT / "fixtures/speech-default.card.txt").read_text() +
                         f"PICTURE: {self.output}\n")

    def test_conversation_analysis_reuses_decode_and_card_laugh_spans(self):
        baseline = self.seed_baseline("voice-a", [180.] * 10)
        original = baseline.read_bytes()
        with card_fixture() as (audio, whisper, _, _, laughs), \
             patch.dict(os.environ, {"RAVEN_EARS_LAUGH_MODEL": "omine"}), \
             patch("laugh_models.run_detector", return_value=dict(
                 model="omine", status="ran", spans=[[3., 3.5], [4., 4.2]])) as omine:
            clips = C.analyze_clips([audio], ["voice-a"])
            R.load_audio.assert_called_once()
        whisper.assert_not_called()
        omine.assert_called_once_with("omine", str(audio))
        laughs.assert_called_once()
        clip, = clips
        self.assertEqual((clip.speaker, clip.laugh_source, clip.laughs),
                         ("voice-a", "omine", [[3., 4.2]]))
        self.assertEqual(clip.words, [])
        self.assertEqual(clip.measured["usual_pitch"], dict(low_hz=164., high_hz=196., notes=10))
        self.assertEqual(baseline.read_bytes(), original)
        edges, frequencies, power = clip.measured["spectrum"]
        self.assertEqual(edges[-1], 60.)
        self.assertEqual(power.shape, (len(frequencies) - 1, len(edges) - 1))

    def test_conversation_cli_uses_labels_then_confident_voice_and_keeps_words_private(self):
        words = [("transcript-marker", 1.68, 2.1)]
        match = dict(status="matched", ranked=[("voice-b", .87)])
        no_match = dict(status="unknown", ranked=[("voice-b", .76)])
        laugh = dict(start=3., end=4., label="laugh", peak=.3, total=.4)
        results = [dict(data=dict(duration_s=12., words=words, voice=voice, laughs={"laughs": [laugh]},
                                  laugh_model=dict(model="omine", status="failed")), lines=["transcript-marker"])
                   for voice in (match, no_match, match)]
        paths = [self.root / f"clip-{i}.webm" for i in range(3)]
        for path in paths:
            path.touch()
        stdout = io.StringIO()
        with patch.object(R, "analyze", side_effect=results) as analyze, \
             patch.object(C, "conversation_sheet", return_value=self.output) as draw, redirect_stdout(stdout):
            C.main([*map(str, paths), "--speakers", "-", "-", "voice-a", "--output", str(self.output)])
        self.assertEqual(stdout.getvalue(), f"{self.output}\n")
        self.assertEqual([call.args[0] for call in analyze.call_args_list], list(map(str, paths)))
        self.assertEqual([call.kwargs["speaker"] for call in analyze.call_args_list], ["", "", "voice-a"])
        self.assertEqual([call.kwargs["identify_voice"] for call in analyze.call_args_list], [True, True, False])
        self.assertTrue(all(call.kwargs["capture_picture"] and not call.kwargs["update_baseline"] and
                            not call.kwargs["use_whisper"] for call in analyze.call_args_list))
        clips = draw.call_args.args[0]
        self.assertEqual([clip.speaker for clip in clips], ["voice-b", "unknown", "voice-a"])
        self.assertTrue(all(clip.words == [] for clip in clips))
        self.assertEqual((clips[0].laugh_source, clips[0].laughs), ("CNN14", [(3., 4.)]))

    def test_conversation_cli_one_speaker_label_applies_to_every_clip(self):
        results = [dict(data=dict(duration_s=5., words=[], voice=None, laughs={"laughs": []}), lines=[])
                   for _ in range(3)]
        paths = [self.root / f"solo-{i}.webm" for i in range(3)]
        for path in paths:
            path.touch()
        with patch.object(R, "analyze", side_effect=results) as analyze, \
             patch.object(C, "conversation_sheet", return_value=self.output), redirect_stdout(io.StringIO()):
            C.main([*map(str, paths), "--speakers", "voice-a", "--output", str(self.output)])
        self.assertEqual([call.kwargs["speaker"] for call in analyze.call_args_list], ["voice-a"] * 3)

    def test_conversation_timestamps_keep_overlaps_and_only_shorten_long_empty_gaps(self):
        clips = [C.Clip("voice-a", duration, {}, [], "", []) for duration in (10., 2., 5., 2.)]
        stamps = C._timestamps(["2026-01-01T02:00:00+02:00", "2026-01-01T00:00:04Z",
                                "2026-01-01T00:00:12Z", "2026-01-01T00:04:17Z"])
        self.assertEqual(C._timeline(clips, stamps), ([0., 4., 12., 22.], [(17., 22., 240.)], 24.))
        self.assertEqual(C._timeline(clips[:2], []), ([0., 10.5], [(10., 10.5, .5)], 12.5))
        for times in (["2026-01-01"], ["2026-01-01T00:00:00", "2026-01-01T00:00:01Z"],
                      ["2026-01-01T00:01:00Z", "2026-01-01T00:00:00Z"]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                C._timestamps(times)

    def test_conversation_renders_speaker_lanes_hatched_gap_and_optional_words(self):
        from matplotlib.text import Text

        measured = dict(times=[0., 1., 2.], rms_db=[-40., -22., -30.], f0=[161., np.nan, 191.],
                        spectrum=(np.array([0., 1., 10.]), np.array([50., 1000., 8000.]),
                                  np.array([[-35., -21.], [-66., -41.]])))
        words = [("transcript-marker", 1., 2.)]
        with_band = dict(measured, usual_pitch=dict(low_hz=164., high_hz=196., notes=10))
        without_band = dict(measured, usual_pitch=dict(reason="no baseline"))
        clips = [C.Clip("voice-a", 10., with_band, [(3., 4.2)], "omine", words),
                 C.Clip("voice-b", 10., without_band, [(3., 4.2)], "CNN14", []),
                 C.Clip("voice-a", 10., measured, [], "omine", [])]
        stamps = C._timestamps(["2026-01-01T00:00:00Z", "2026-01-01T00:04:10Z", "2026-01-01T00:04:22Z"])
        save = S._save

        def inspect(figure, output):
            labels, spec_a, pitch_a, spec_b, pitch_b, *_ = figure.axes
            self.assertEqual([spec_a.get_ylabel(), spec_b.get_ylabel()],
                             ["voice-a\nSpectrum (Hz)", "voice-b\nSpectrum (Hz)"])
            self.assertEqual([len(spec_a.collections), len(spec_b.collections)], [2, 1])
            self.assertEqual([line.get_color() for line in (pitch_a.lines[0], pitch_b.lines[0])], list(C.COLOURS[:2]))
            np.testing.assert_allclose(pitch_b.lines[0].get_xdata(), [15., 16., 17.])
            self.assertTrue(np.isnan(pitch_b.lines[0].get_ydata()[1]))
            self.assertEqual(pitch_a.lines[1].get_linestyle(), "-")
            self.assertEqual(pitch_b.lines[1].get_linestyle(), "--")
            np.testing.assert_allclose(pitch_b.patches[0].get_bbox().intervalx, [18., 19.2])
            band, = [patch for patch in pitch_a.patches if patch.get_zorder() == 0]
            np.testing.assert_allclose(band.get_bbox().intervaly, [164., 196.])
            self.assertFalse(any(patch.get_zorder() == 0 for patch in pitch_b.patches))
            footer = figure.texts[2].get_text()
            self.assertIn("voice-a · pale band: usual median pitch (from 10 notes)", footer)
            self.assertIn("voice-b · Usual pitch band unavailable · no baseline", footer)
            rms = [line for ax in figure.axes if ax.get_ylabel() == "RMS (dB)" for line in ax.lines]
            self.assertEqual(len(rms), len(clips))
            self.assertTrue(all(0 < line.get_alpha() < 1 and line.get_linewidth() < .85 for line in rms))
            self.assertEqual(labels.patches[0].get_hatch(), "////")
            np.testing.assert_allclose(labels.patches[0].get_bbox().intervalx, [10., 15.])
            self.assertEqual(labels.xaxis.get_major_formatter()(15.), "4:10")
            start_labels = [text.get_text() for text in labels.texts]
            self.assertIn("… 4 min …", start_labels)
            self.assertIn("00:04:10", start_labels)
            all_text = [text.get_text() for text in figure.findobj(match=Text)]
            self.assertEqual("transcript-marker" in all_text, include_words)
            return save(figure, output)

        for include_words in (False, True):
            with self.subTest(include_words=include_words), patch.object(S, "_save", side_effect=inspect):
                C.conversation_sheet(clips, self.output, stamps=stamps, include_words=include_words)
                self.assert_png(self.output)

    def test_twenty_conversation_clips_keep_ticks_and_skip_colliding_label_boxes(self):
        clips = [C.Clip(f"voice-{i % 2}", 2., {}, [], "", []) for i in range(20)]
        stamps = C._timestamps([f"2026-01-01T23:{25 + i // 5}:{i % 5:02d}" for i in range(20)])

        def inspect(figure, output):
            figure.canvas.draw()
            timeline = figure.axes[0]
            self.assertEqual(len(timeline.lines), 20)
            starts = [artist for artist in timeline.texts if artist.get_text().startswith("23:")]
            self.assertGreater(len(starts), 0)
            self.assertLess(len(starts), 20)
            self.assertTrue(any(artist.get_text().startswith("…") for artist in timeline.texts))
            boxes = [artist.get_bbox_patch().get_window_extent() for artist in timeline.texts]
            for i, box in enumerate(boxes):
                self.assertTrue(all(not box.overlaps(other) for other in boxes[i + 1:]))
            return output

        with patch.object(S, "_save", side_effect=inspect):
            C.conversation_sheet(clips, self.output, stamps=stamps)

    def test_missing_matplotlib_prints_one_clear_line_after_card(self):
        with patch.dict(sys.modules, {"matplotlib": None}):
            self.assertEqual(S.attempt(S.song_sheet, None, None, None, None, self.output),
                             "pip install matplotlib to draw the listening sheet")
            self.assertEqual(S.attempt(S.speech_sheet, {}, {}, self.output),
                             "pip install matplotlib to draw the listening sheet")
        with card_fixture() as (audio, *_), patch.dict(sys.modules, {"matplotlib": None}):
            output = io.StringIO()
            with redirect_stdout(output):
                R.main([str(audio), "--quiet", "--picture", "--no-baseline-update"])
            self.assertTrue(output.getvalue().endswith("and now, you\npip install matplotlib to draw the listening sheet\n"))


class PictureDefaultTests(unittest.TestCase):
    def test_automatic_speech_sheet_keeps_baseline_policy_and_explicit_sheet_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory, card_fixture() as (audio, *_), \
             patch.object(S, 'speech_sheet', return_value=Path('sheet.png')):
            for env, flags, learns in (
                    ('', [], True), ('1', [], True), ('0', [], True),
                    ('', ['--no-picture'], True), ('', ['--picture'], False),
                    ('', ['--picture-out', 'sheet.png'], False),
                    ('', ['--by-ear', 'marks.txt'], False),
                    ('', ['--no-baseline-update'], False)):
                with self.subTest(env=env, flags=flags), \
                     patch.dict(os.environ, {'RAVEN_EARS_HOME': directory, 'RAVEN_EARS_NO_BASELINE': '0',
                                             'RAVEN_EARS_PICTURE': env}), redirect_stdout(io.StringIO()):
                    baseline = R.baseline_path('fixture')
                    baseline.unlink(missing_ok=True)
                    self.assertEqual(R.main([str(audio), '--speaker', 'fixture', '--quiet', *flags]), 0)
                    self.assertEqual(baseline.exists(), learns)
            with patch.dict(os.environ, {'RAVEN_EARS_HOME': directory, 'RAVEN_EARS_NO_BASELINE': '1',
                                         'RAVEN_EARS_PICTURE': ''}), redirect_stdout(io.StringIO()):
                self.assertEqual(R.main([str(audio), '--speaker', 'fixture', '--quiet']), 0)
                self.assertFalse(R.baseline_path('fixture').exists())

    def test_speech_default_env_flags_and_fast(self):
        result = dict(lines=['PITCH: invented', 'and now, you'], data={}, picture_data={}, notes=[], timings={})
        for env, flags, enabled in (
                ('', [], True), ('0', [], False), ('1', [], True),
                ('', ['--no-picture'], False), ('0', ['--picture'], True),
                ('1', ['--no-picture'], False), ('', ['--fast'], True),
                ('0', ['--picture-out', 'sheet.png'], True),
                ('0', ['--by-ear', 'marks.txt'], True),
                ('1', ['--picture-out', 'sheet.png', '--no-picture'], False)):
            with self.subTest(env=env, flags=flags), \
                 patch.dict(os.environ, {'RAVEN_EARS_PICTURE': env}), \
                 patch.object(R, 'analyze', return_value=result) as analyze, \
                 patch.object(S, 'speech_sheet', return_value=Path('sheet.png')) as draw, \
                 redirect_stdout(io.StringIO()) as out:
                self.assertEqual(R.main(['invented.wav', '--quiet', *flags]), 0)
                self.assertEqual(analyze.call_args.kwargs.get('capture_picture', False), enabled)
                self.assertEqual(draw.called, enabled)
                self.assertEqual(out.getvalue(), 'PITCH: invented\nand now, you\n' +
                                 ('PICTURE: sheet.png\n' if enabled else ''))

    def test_speech_missing_default_matplotlib_keeps_real_analysis_card(self):
        # Synthetic PCM with fake models, using the ordinary speech CLI all the way through.
        with card_fixture() as (audio, *_), \
             patch.dict(os.environ, {'RAVEN_EARS_PICTURE': ''}), \
             patch.dict(sys.modules, {'matplotlib': None}), \
             patch.object(R, 'analyze', wraps=R.analyze) as analyze, \
             patch.object(S, 'speech_sheet', side_effect=AssertionError('no drawing')):
            with redirect_stdout(io.StringIO()) as plain:
                R.main([str(audio), '--quiet', '--no-picture'])
            with redirect_stdout(io.StringIO()) as default:
                R.main([str(audio), '--quiet'])
            self.assertEqual(default.getvalue(), plain.getvalue() +
                             'pip install -r requirements-picture.txt to draw the listening sheet\n')
            self.assertEqual(analyze.call_args_list[0], analyze.call_args_list[1])

    def test_speech_json_and_server_do_not_acquire_sheets(self):
        result = dict(lines=['invented'], data={}, notes=[], timings={})
        with patch.dict(os.environ, {'RAVEN_EARS_PICTURE': '1'}), \
             patch.dict(sys.modules, {'matplotlib': None}), \
             patch.object(R, 'analyze', return_value=result) as analyze, \
             patch.object(R, 'serve', return_value=0) as serve, \
             patch.object(S, 'speech_sheet', side_effect=AssertionError('no drawing')):
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(R.main(['invented.wav', '--quiet', '--json']), 0)
            self.assertIsInstance(json.loads(out.getvalue()), dict)
            self.assertNotIn('capture_picture', analyze.call_args.kwargs)
            self.assertEqual(R.main(['--serve']), 0)
            self.assertNotIn('capture_picture', serve.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
