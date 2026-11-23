"""Metadata-only version checks; real saved-card shape, fake service responses."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import credits
import listen
import lrclib
import song_length as S


def row(source, duration):
    return {"source": source, "duration_s": duration}


def record(**changes):
    # LRCLIB's existing response shape (test_lrclib.record), invented text.
    return dict({"id": 1, "artistName": "Example Artist", "trackName": "Example Song",
                 "albumName": "Example Album", "duration": 172, "instrumental": False,
                 "plainLyrics": "An invented lantern", "syncedLyrics": "[00:10.00] An invented lantern"}, **changes)


class LengthTests(unittest.TestCase):
    def test_consensus_outliers_duplicates_and_ties(self):
        rows = [row("LRCLIB", 172), row("Discogs", 173), row("MusicBrainz", 300)]
        self.assertEqual(S.consensus(rows), {"studio_s": 172.5, "sources": ["LRCLIB", "Discogs"]})
        self.assertEqual(S.consensus(rows + [row("MusicBrainz", 300)] * 20), S.consensus(rows))
        for ambiguous in ([row("LRCLIB", 172), row("Discogs", 192)],
                          [row("LRCLIB", 172), row("LRCLIB", 192)], []):
            self.assertIsNone(S.consensus(ambiguous))
        self.assertEqual(S.consensus([row("Discogs", 172)]), {"studio_s": 172, "sources": ["Discogs"]})
        self.assertIsNone(S.consensus([row("Discogs", v) for v in (None, 0, -2, True, "bad", float("nan"), float("inf"))]))

    def test_tolerance_is_exclusive_and_scales_with_studio_length(self):
        for studio in (60, 172, 600):
            rows = [row("LRCLIB", studio)]
            for sign in (-1, 1):
                self.assertIsNone(S.finding(studio + sign * S.tolerance(studio), rows))
                found = S.finding(studio + sign * (S.tolerance(studio) + .01), rows)
                self.assertIsNotNone(found)
                self.assertEqual(found["sources"], ["LRCLIB"])

    def test_search_keeps_duration_without_using_other_version_lyrics(self):
        with patch.object(lrclib, "_get_json", side_effect=[None, [record()]]) as request:
            answer = lrclib._lookup("Example Artist", "Example Song", 192, None)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(answer["status"], "nomatch")
        self.assertEqual(answer["synced"], [])
        self.assertEqual(answer["plain"], [])
        found = S.finding(192, answer["studio_durations"], [answer])
        self.assertTrue(found["studio_lyrics_available"])
        self.assertIn("NOTES: LRCLIB has clean lyrics at 2:52; not used", S.annotate("MUSIC: invented", found))
        self.assertNotIn("studio_lyrics_available", S.finding(152, answer["studio_durations"], [answer]))

    def test_search_rejects_identity_album_versions_and_bad_lyrics(self):
        records = [record(artistName="Other"), record(trackName="Other"),
                   record(trackName="Example Song (Live)"), record(albumName="Live at Home"),
                   record(duration=None)]
        with patch.object(lrclib, "_get_json", side_effect=[None, records]):
            self.assertNotIn("studio_durations", lrclib._lookup("Example Artist", "Example Song", 192, None))
        with patch.object(lrclib, "_get_json", side_effect=[None, [record(albumName="Other Album")]]):
            self.assertNotIn("studio_durations", lrclib._lookup("Example Artist", "Example Song", 192, "Example Album"))
        with patch.object(lrclib, "_get_json", side_effect=[None, [record(plainLyrics=[], syncedLyrics=None)]]):
            answer = lrclib._lookup("Example Artist", "Example Song", 192, None)
        self.assertEqual(answer["studio_durations"], [row("LRCLIB", 172)])
        self.assertNotIn("clean_lyric_durations", answer)

    def test_exact_success_keeps_existing_one_request(self):
        with patch.object(lrclib, "_get_json", return_value=record()) as request:
            answer = lrclib._lookup("Example Artist", "Example Song", 172, None)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(answer["studio_durations"], [row("LRCLIB", 172)])
        self.assertNotIn("exact_failed", answer)

    def test_rejected_studio_lyrics_never_enter_whisper_context(self):
        import lyrics_shared
        with patch.object(lrclib, "_get_json", side_effect=[None, [record()]]), \
             patch.object(lyrics_shared, "lyric_context", wraps=lyrics_shared.lyric_context) as context:
            result = lrclib.resolve("unused.mp3", [], 192, artist="Example Artist", title="Example Song",
                                    use_cache=False, word_loader=lambda *_: [('other', 8., 9.)])
        self.assertEqual(context.call_args.args[0], [])
        self.assertIsNone(result["rows"])
        self.assertEqual(result["words"], [('other', 8., 9.)])

    def test_negative_cache_keeps_duration_finding_without_new_requests(self):
        import tempfile
        with tempfile.TemporaryDirectory() as cache, \
             patch.object(lrclib, "_get_json", side_effect=[None, [record()]]) as request:
            answer = lrclib.get("unused.mp3", artist="Example Artist", title="Example Song", duration=192, cache_dir=cache)
            request.reset_mock()
            replay = lrclib.get("unused.mp3", artist="Example Artist", title="Example Song", duration=192, cache_dir=cache)
        request.assert_not_called()
        self.assertTrue(replay["cached"])
        self.assertEqual(S.finding(192, answer["studio_durations"], [answer]),
                         S.finding(192, replay["studio_durations"], [replay]))

    def test_capture_reads_only_same_audio_memo_and_never_fetches(self):
        memo = {("copy.mp3", 192): {"studio_durations": [row("Discogs", 172)]},
                "other.mp3": {"studio_durations": [row("MusicBrainz", 300)]}}
        with patch.dict(credits._MEMO, memo, clear=True), \
             patch.object(credits, "for_song", side_effect=AssertionError("no requests")):
            data = {"duration": 192}
            S.capture("copy.mp3", data)
        self.assertEqual(data["length_check"]["sources"], ["Discogs"])

    def test_musicbrainz_duration_uses_existing_recording_search_shape(self):
        captured = json.loads((Path(__file__).parent / "fixtures/musicbrainz-performers.json").read_text())["Take Five"]["search"]
        artist = captured["artist-credit"][0]["name"]
        for changes, expected in (({}, 324), ({"title": "Take Five (Live)"}, None),
                                  ({"disambiguation": "live recording"}, None), ({"length": None}, None)):
            with self.subTest(changes=changes), \
                 patch.object(credits, "search_recordings", return_value=[dict(captured, **changes)]) as search, \
                 patch.object(credits, "recording_relations", return_value=[]), \
                 patch.object(credits, "release_track_relations", return_value=[]):
                answer = credits.lookup_musicbrainz(artist, "Take Five")
            self.assertEqual(answer.get("studio_duration_s"), expected)
            search.assert_called_once_with(artist, "Take Five")

    def test_all_saved_views_preserve_bytes_without_a_warning(self):
        song = listen.load(Path(__file__).parent / "fixtures/first-listen-song.json")
        for view in listen.VIEWS:
            before = listen.render(song, view)[0]
            known = copy.deepcopy(song)
            known["cards"]["music"]["json"]["studio_durations"] = [row("LRCLIB", song["manifest"]["duration_s"])]
            S.update_song(known)
            self.assertNotIn("length_check", known)
            self.assertEqual(listen.render(known, view)[0], before)
            unknown = copy.deepcopy(song)
            S.update_song(unknown)
            self.assertEqual(listen.render(unknown, view)[0], before)
        self.assertEqual(S.annotate("MUSIC: invented\n", None), "MUSIC: invented\n")

    def test_real_saved_card_with_injected_length_shows_one_warning_everywhere(self):
        song = listen.load(Path(__file__).parent / "fixtures/first-listen-song.json")
        data = song["cards"]["music"]["json"]
        data["studio_durations"] = [row("LRCLIB", 131.5), row("Discogs", 132)]
        S.update_song(song)
        for card in song["cards"].values():
            card["text"] = S.annotate(card["text"], song["length_check"])
        with patch("urllib.request.urlopen", side_effect=AssertionError("no requests")):
            for view in listen.VIEWS:
                text = listen.render(song, view)[0]
                if view == "first-listen":
                    self.assertGreater(text.index(S.WARNING), text.index('And now, you:'))
                else:
                    self.assertTrue(text.startswith(S.WARNING))
                self.assertEqual(text.count(S.WARNING), 1)
                self.assertIn("(LRCLIB, Discogs)", text)
                if view in ("music", "full", "first-listen"):
                    self.assertIn("THE SONG" if view == "first-listen" else "LYRICS:", text)


def run_tests(check):
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(LengthTests).run(result)
    failures = {test._testMethodName: detail for test, detail in result.failures + result.errors}
    for name in unittest.defaultTestLoader.getTestCaseNames(LengthTests):
        check("song length: " + name.removeprefix("test_"), name not in failures, failures.get(name, ""))


if __name__ == "__main__":
    unittest.main()
