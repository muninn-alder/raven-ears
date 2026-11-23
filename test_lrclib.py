"""LRCLIB checks: invented lyrics, mocked HTTP, temporary caches, no audio models."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse

import lrclib as L


TEXT = ["Carry a lantern", "Follow the river", "Open the window"]
WORDS = [(word, start + i * 0.4, start + i * 0.4 + 0.3)
         for line, start in zip(TEXT, (10, 20, 30)) for i, word in enumerate(line.split())]


def record(**kwargs):
    return dict({"id": 1, "artistName": "Example Artist", "trackName": "Example Song", "albumName": "Example Album",
                 "duration": 60, "instrumental": False, "plainLyrics": "\n".join(TEXT),
                 "syncedLyrics": "\n".join(f"[00:{start:02d}.00] {line}" for start, line in zip((10, 20, 30), TEXT))}, **kwargs)


class LyricsTests(unittest.TestCase):
    def test_published_text_selects_language_only_while_loading_words(self):
        import lyrics_shared as LS
        text = 'Je suis dans une maison. Vous et nous, les amis, mes livres et mon papier sont pour vous.'
        for synced in ('', '[00:10.00]' + text):
            answer = L._answer(record(plainLyrics=text, syncedLyrics=synced))
            def loader(path, notes):
                self.assertEqual(LS.effective_language(), 'fr')
                self.assertIn('LRCLIB', LS.language_choice()[1])
                return [('papier', 10., 10.5)]
            with self.subTest(status=answer['status']), patch.dict(os.environ), \
                 patch.object(L, 'get', return_value=answer):
                os.environ.pop('RAVEN_EARS_LANG', None)
                result = L.resolve('unused.mp3', [], 60., word_loader=loader)
                self.assertEqual(len(result['rows']), 1)
                self.assertIn('no lyric-text', LS.language_choice()[1])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        self.responses = []
        self.old_config = dict(L.CONFIG)
        self.addCleanup(lambda: L.CONFIG.update(self.old_config))
        L.CONFIG.update(off=False, artist="Example Artist", title="Example Song", album=None)
        self.enterContext(patch.dict(os.environ, {"RAVEN_NO_LYRICS_LOOKUP": ""}))
        self.enterContext(patch.object(L.urllib.request, "urlopen", side_effect=self.http))
        self.enterContext(patch.object(L.time, "sleep"))
        self.real_activity = L.cached_activity
        self.enterContext(patch.object(L, "cached_activity", return_value=None))

    def http(self, req, timeout):
        self.calls.append((req, timeout))
        if not self.responses:
            raise AssertionError("unexpected HTTP request: " + req.full_url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, bytes):
            return io.BytesIO(response)
        return io.BytesIO(json.dumps(response).encode())

    def lookup(self, **kwargs):
        return L.get("unused.mp3", artist="Example Artist", title="Example Song", duration=60,
                     cache_dir=self.tmp.name, **kwargs)

    def resolve(self, **kwargs):
        return L.resolve("unused.mp3", [], 60, word_loader=lambda *_: WORDS, cache_dir=self.tmp.name, **kwargs)

    def test_exact_match_metadata_timeout_and_user_agent(self):
        self.responses = [record()]
        result = self.lookup(album="Example Album")
        self.assertEqual(result["status"], "synced")
        self.assertEqual(result["source"], "LRCLIB")
        req, timeout = self.calls[0]
        self.assertEqual(timeout, 6)
        self.assertEqual(req.get_header("User-agent"), L.credits.UA)
        self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlsplit(req.full_url).query),
                         {"artist_name": ["Example Artist"], "track_name": ["Example Song"],
                          "duration": ["60.0"], "album_name": ["Example Album"]})

    def test_search_after_404_and_duration_tolerance(self):
        self.responses = [urllib.error.HTTPError("url", 404, "not found", {}, None),
                          [record(id=2, duration=62.001), record(id=3, duration=58), record(id=4, duration=62)]]
        result = self.lookup()
        self.assertEqual(result["id"], 3)
        self.assertEqual(result["endpoint"], "search")
        self.assertEqual(len(self.calls), 2)

    def test_search_rejects_wrong_artist_title_version_and_missing_duration(self):
        self.responses = [None, [record(artistName="Another Artist"), record(trackName="Another Song"),
                                 record(trackName="Example Song (Live)"), record(duration=None)]]
        self.assertEqual(self.lookup()["status"], "nomatch")

    def test_album_can_narrow_search(self):
        self.responses = [record(albumName="Other Album"), [record(albumName="Other Album")]]
        self.assertEqual(self.lookup(album="Example Album")["status"], "nomatch")

    def test_search_prefers_synced_over_exact_plain(self):
        self.responses = [record(syncedLyrics=None), [record(id=2)]]
        result = self.lookup()
        self.assertEqual((result["status"], result["id"]), ("synced", 2))

    def test_interleaved_translation_loses_to_a_clean_synced_text(self):
        # 11 Nov 2026: the exact endpoint returned an upload that put a translated line
        # under every sung line, same timestamp, plus credit lines at 0:00.
        sung = [f"Sung line {i}" for i in range(6)]
        paired = "[00:00.00] Credits: Someone\n" + "\n".join(
            f"[00:{10 + 4 * i:02d}.00] {line}\n[00:{10 + 4 * i:02d}.00] Translated {i}" for i, line in enumerate(sung))
        clean = "\n".join(f"[00:{10 + 4 * i:02d}.00] {line}" for i, line in enumerate(sung))
        self.responses = [record(id=7, syncedLyrics=paired), [record(id=7, syncedLyrics=paired), record(id=8, syncedLyrics=clean)]]
        result = self.lookup()
        self.assertEqual((result["status"], result["id"], result["paired"]), ("synced", 8, False))
        self.assertFalse(result.get("issues"))

    def test_only_a_paired_text_is_kept_but_disclosed(self):
        paired = "\n".join(f"[00:{10 + 4 * i:02d}.00] Sung {i}\n[00:{10 + 4 * i:02d}.00] Translated {i}" for i in range(6))
        self.responses = [record(id=7, syncedLyrics=paired), [record(id=7, syncedLyrics=paired)],
                          [record(id=7, syncedLyrics=paired)]]
        result = self.lookup()
        self.assertEqual(len(self.calls), 3)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.calls[2][0].full_url).query)
        self.assertEqual((q["artist_name"], q["track_name"]), (["Example Artist"], ["Example Song"]))
        self.assertTrue(result["paired"])
        self.assertTrue(any("interleaved translation" in issue for issue in result["issues"]))
        self.assertEqual(result["status"], "synced")

    def test_field_search_finds_the_clean_text_the_free_search_missed(self):
        paired = "\n".join(f"[00:{10 + 4 * i:02d}.00] Sung {i}\n[00:{10 + 4 * i:02d}.00] Translated {i}" for i in range(6))
        self.responses = [record(id=7, syncedLyrics=paired), [record(id=7, syncedLyrics=paired)], [record(id=9)]]
        result = self.lookup()
        self.assertEqual((result["id"], result["paired"], result["endpoint"]), (9, False, "search (fields)"))

    def test_a_song_repeating_its_chorus_is_not_paired(self):
        lines = "\n".join(f"[00:{10 + 3 * i:02d}.00] Same chorus line" for i in range(12))
        self.assertFalse(L._paired(L.parse_lrc(lines)))

    def test_cached_paired_answer_is_looked_up_again(self):
        rows = [{"t": float(t), "text": s} for t in range(10, 40, 4) for s in ("Sung", "Translated")]
        self.assertFalse(L._cache_ok({"source": "LRCLIB", "status": "synced", "synced": rows}))

    def test_plain_text_falls_back_to_existing_aligner(self):
        self.responses = [record(syncedLyrics=None), []]
        result = self.resolve()
        self.assertEqual([r["t"] for r in result["rows"]], [10, 20, 30])
        self.assertTrue(all(not r["approx"] for r in result["rows"]))
        self.assertIn("LRCLIB had no synced lyrics; plain lyrics aligned to the vocals", result["label"])
        self.assertEqual(result["words"], WORDS)

    def test_lrc_multiple_tags_metadata_and_milliseconds(self):
        rows = L.parse_lrc("[ar:Example]\n[00:02.25][01:03.125] Repeated line\n[00:10.00]\n"
                           "[00:70.00] invalid\nplain text\n[00:01] Opening line")
        self.assertEqual(rows, [{"t": 1, "text": "Opening line"}, {"t": 2.25, "text": "Repeated line"},
                                {"t": 63.125, "text": "Repeated line"}])

    def test_nonzero_offset_tags_prefer_alignment_without_assuming_convention(self):
        self.responses = [record(syncedLyrics="[offset:500]\n" + record()["syncedLyrics"]), []]
        result = self.resolve()
        self.assertEqual(result["lookup"]["status"], "plain")
        self.assertIn("synced lyrics were unusable", result["label"])
        self.assertEqual([r["t"] for r in result["rows"]], [10, 20, 30])

    def test_one_retry_for_timeout_then_success(self):
        self.responses = [TimeoutError(), record()]
        self.assertEqual(self.lookup()["status"], "synced")
        self.assertEqual(len(self.calls), 2)

    def test_http_503_and_connection_errors_retry_once(self):
        self.responses = [urllib.error.HTTPError("url", 503, "busy", {}, None), record()]
        self.assertEqual(self.lookup(use_cache=False)["status"], "synced")
        self.responses = [urllib.error.URLError("connection refused"), record()]
        self.assertEqual(self.lookup(use_cache=False)["status"], "synced")
        self.assertEqual(len(self.calls), 4)

    def test_both_requests_timing_out_never_cache_no_lyrics(self):
        self.responses = [TimeoutError() for _ in range(4)]
        result = self.resolve()
        self.assertEqual(result["lookup"]["status"], "failed")
        self.assertIn("LRCLIB get timed out", result["label"])
        self.assertIn("LRCLIB search timed out", result["label"])
        self.assertIn("words are Whisper's guesses", result["label"])
        self.assertIn("have the words? --lyrics FILE", result["label"])
        self.assertNotIn("no lyrics", result["label"])
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_partial_failure_with_empty_search_is_not_negative_cached(self):
        self.responses = [TimeoutError(), TimeoutError(), []]
        self.assertEqual(self.lookup()["status"], "failed")
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_successful_fallback_discloses_primary_failure(self):
        self.responses = [TimeoutError(), TimeoutError(), [record()]]
        result = self.resolve()
        self.assertEqual(result["lookup"]["endpoint"], "search")
        self.assertIn("LRCLIB get timed out", result["label"])
        self.assertEqual(result["lookup"]["status"], "synced")

    def test_plain_survives_failed_synced_search_without_claiming_absence(self):
        self.responses = [record(syncedLyrics=None), TimeoutError(), TimeoutError()]
        result = self.resolve()
        self.assertEqual(result["lookup"]["status"], "plain")
        self.assertIn("synced lookup incomplete", result["label"])
        self.assertNotIn("had no synced", result["label"])

    def test_invalid_json_and_schema_are_failure_not_absence(self):
        for response in (b"{", ["not an object"], {"error": "backend broke"}, record(plainLyrics=12),
                         record(instrumental="false"), record(syncedLyrics="broken LRC", plainLyrics=None)):
            self.responses = [response, []]
            self.assertEqual(self.lookup(use_cache=False)["status"], "failed")

    def test_success_cache_is_source_recorded_and_duration_specific(self):
        self.responses = [record()]
        first, second = self.lookup(), self.lookup()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(second["cached"])
        cp = L.cache_path("Example Artist", "Example Song", 60, cache_dir=self.tmp.name)
        saved = json.loads(Path(cp).read_text())
        self.assertEqual(saved["source"], "LRCLIB")
        self.assertEqual(saved["endpoint"], "get")
        self.assertNotEqual(cp, L.cache_path("Example Artist", "Example Song", 65, cache_dir=self.tmp.name))
        self.assertNotEqual(cp, L.cache_path("Example Artist", "Example Song (Live)", 60, cache_dir=self.tmp.name))
        self.assertEqual(first["synced"], second["synced"])

    def test_no_cache_does_not_read_or_write(self):
        self.responses = [record(), record(id=2)]
        self.assertEqual(self.lookup(use_cache=False)["id"], 1)
        self.assertEqual(self.lookup(use_cache=False)["id"], 2)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_successful_negative_expires_and_corrupt_cache_is_ignored(self):
        self.responses = [None, []]
        self.assertEqual(self.lookup()["status"], "nomatch")
        self.assertTrue(self.lookup()["cached"])
        cp = Path(L.cache_path("Example Artist", "Example Song", 60, cache_dir=self.tmp.name))
        cached = json.loads(cp.read_text())
        cached["when"] = 0
        cp.write_text(json.dumps(cached))
        self.responses = [record()]
        self.assertEqual(self.lookup()["status"], "synced")
        cp.write_text(json.dumps({"source": "LRCLIB", "status": "synced", "synced": [{"t": "wrong type"}]}))
        self.responses = [record()]
        self.assertEqual(self.lookup()["status"], "synced")
        cp.write_text("broken json")
        self.responses = [record()]
        self.assertEqual(self.lookup()["status"], "synced")

    def test_unusable_lrc_is_disclosed_before_plain_alignment(self):
        self.responses = [record(syncedLyrics="broken LRC"), []]
        result = self.resolve()
        self.assertIn("synced lyrics were unusable", result["label"])
        self.assertEqual(result["lookup"]["status"], "plain")

    def test_file_lyrics_bypass_network_even_when_empty(self):
        for lines in (TEXT, []):
            result = self.resolve(lyric_lines=lines)
            self.assertEqual(result["lookup"]["source"], "file")
            self.assertEqual([r["text"] for r in result["rows"]], lines)
        self.assertEqual(self.calls, [])

    def test_off_and_environment_switch_bypass_cache_and_http(self):
        self.assertEqual(self.lookup(off=True)["status"], "off")
        with patch.dict(os.environ, {"RAVEN_NO_LYRICS_LOOKUP": "1"}):
            result = self.resolve()
        self.assertIn("lookup off", result["label"])
        self.assertIn("Whisper's guesses", result["label"])
        self.assertEqual(self.calls, [])

    def test_missing_identity_is_disclosed_without_http(self):
        with patch.object(L.credits, "identify", return_value=(None, None, "partial")):
            result = self.resolve()
        self.assertIn("no artist/title", result["label"])
        self.assertEqual(self.calls, [])

    def test_instrumental_is_database_designation_without_whisper_or_silence(self):
        self.responses = [record(instrumental=True, plainLyrics=None, syncedLyrics=None)]
        with patch("lyrics_shared.get_words", side_effect=AssertionError("must not transcribe")):
            result = L.resolve("unused.mp3", [], 60, cache_dir=self.tmp.name)
        self.assertIn("instrumental (LRCLIB)", result["label"])
        self.assertIn("not a measurement of silence", result["label"])
        self.assertIsNone(result["words"])

    def test_synced_times_remain_supplied_and_approximate(self):
        self.responses = [record()]
        result = self.resolve(activity=[(9, 12), (19, 22), (29, 32)])
        self.assertTrue(all(r["approx"] for r in result["rows"]))
        self.assertEqual([r["t"] for r in result["rows"]], [10, 20, 30])
        self.assertIn("3 lines compared with Whisper", result["label"])
        self.assertIn("supplied times (≈)", result["label"])
        self.assertEqual(result["words"], WORDS)

    def test_synced_row_in_measured_quiet_is_marked_without_retiming(self):
        self.responses = [record()]
        result = self.resolve(activity=[(9, 12), (29, 32)])
        self.assertEqual([r["t"] for r in result["rows"]], [10, 20, 30])
        # The text stays clean (it feeds CONTRAST, Jev, hooks and repeat counts); the mark is a flag.
        self.assertEqual([r["text"] for r in result["rows"]], TEXT[:3])
        self.assertEqual([bool(r.get("no_voice")) for r in result["rows"]], [False, True, False])
        self.assertEqual(result["lookup"]["synced"][1]["text"], TEXT[1])
        self.assertIn("1 synced line falls where no voice was measured", result["label"])
        self.assertIn("[0:20≈] " + TEXT[1] + " (no voice measured here)", "\n".join(L.card_lines(result)))

    def test_synced_rows_without_activity_are_untouched(self):
        self.responses = [record()]
        result = self.resolve()
        self.assertEqual([r["text"] for r in result["rows"]], TEXT)
        self.assertNotIn("no voice", result["label"])

    def test_quiet_threshold_and_voice_boundaries(self):
        synced = [{"t": t} for t in (10, 12, 14.99, 15, 20)]
        aligned = [{"approx": True} for _ in synced]
        # [12, 15) is measured quiet, but both stamps touch a vocal
        # boundary within the timing allowance; [19, 21) is too short to flag.
        check = L.sanity_check(synced, aligned, 30, [(0, 12), (15, 19), (21, 30)])
        self.assertEqual(check["no_voice"], [])
        self.assertEqual(L.sanity_check(synced, aligned, 30, None)["no_voice"], [])
        self.assertEqual(L.sanity_check(synced, aligned, 30, [])["no_voice"], list(range(5)))
        # Without a duration, an unbounded quiet tail cannot be measured.
        self.assertEqual(L.sanity_check([{"t": 20}], [{"approx": True}], activity=[(0, 12)])["no_voice"], [])

    def test_synced_voice_window_and_start_slack_keep_a_quiet_line_flag(self):
        synced = [dict(t=10, end=14), dict(t=20, end=20.2),
                  dict(t=30, end=30.2), dict(t=40, end=43)]
        aligned = [dict(approx=True)] * len(synced)
        activity = [(13, 14), (18, 19), (31.5, 33), (43.1, 45)]
        result = L.sanity_check(synced, aligned, 50, activity)
        self.assertEqual(result["no_voice"], [3])
        self.assertFalse(result["disagrees"])

    def test_wrong_version_times_prefer_alignment_to_heard_words(self):
        self.responses = [record(syncedLyrics="\n".join(f"[00:{t:02d}] {line}" for t, line in zip((15, 25, 35), TEXT)))]
        result = self.resolve()
        self.assertEqual([r["t"] for r in result["rows"]], [10, 20, 30])
        self.assertIn("likely a different version", result["label"])
        self.assertIn("preferred alignment", result["label"])
        self.assertTrue(result["lookup"]["timing_check"]["disagrees"])

    def test_vocal_activity_can_challenge_lrc_when_whisper_has_no_words(self):
        self.responses = [record()]
        with patch("lyrics_shared.get_words", return_value=None):
            result = L.resolve("unused.mp3", [], 60, activity=[(40, 50)], cache_dir=self.tmp.name)
        self.assertIn("vocal-stem activity", result["label"])
        self.assertIn("lines placed (≈), not heard", result["label"])
        self.assertTrue(all(r["approx"] for r in result["rows"]))

    def test_no_timing_evidence_is_unverified_not_silence(self):
        self.responses = [record()]
        notes = []
        def failed(*args):
            raise RuntimeError("model failed")
        result = L.resolve("unused.mp3", notes, 60, word_loader=failed, cache_dir=self.tmp.name)
        self.assertIn("audio timing unverified", result["label"])
        self.assertIn("not evidence of silence", notes[0])
        self.assertTrue(all(r["approx"] for r in result["rows"]))

    def test_out_of_range_timestamps_realign_and_quiet_activity_is_evidence(self):
        synced = [{"t": -1, "text": TEXT[0]}, {"t": 61, "text": TEXT[1]}]
        aligned = [{"approx": True}, {"approx": True}]
        check = L.sanity_check(synced, aligned, 60, [])
        self.assertTrue(check["disagrees"])
        self.assertEqual(len(check["reasons"]), 2)

    def test_one_stray_word_match_does_not_reject_a_whole_song(self):
        synced = [{"t": 10}, {"t": 20}, {"t": 30}]
        aligned = [{"t": 40, "approx": False}, {"approx": True}, {"approx": True}]
        self.assertFalse(L.sanity_check(synced, aligned, 60)["disagrees"])

    def test_cached_stem_activity_reads_synthetic_audio_without_creating_stems(self):
        import singer
        import soundfile as sf
        import numpy as np
        stem_dir = Path(self.tmp.name) / "hash"
        stem_dir.mkdir()
        sr = 8000
        signal = np.zeros(sr * 6)
        signal[2 * sr:4 * sr] = 0.1 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)
        sf.write(stem_dir / "vocals.wav", signal, sr)
        with patch.object(singer, "CACHE", self.tmp.name), patch.object(singer, "file_hash", return_value="hash"), \
                patch.object(singer, "get_stems", side_effect=AssertionError("no separation allowed")):
            spans = self.real_activity("unused.mp3")
        self.assertTrue(spans)
        self.assertTrue(all(2 <= start < end <= 4 for start, end in spans))

    def test_broken_whisper_and_failed_lookup_are_both_visible(self):
        self.responses = [TimeoutError() for _ in range(4)]
        notes = []
        def failed(*args):
            raise RuntimeError("model failed")
        result = L.resolve("unused.mp3", notes, 60, word_loader=failed, cache_dir=self.tmp.name)
        self.assertIn("LRCLIB get timed out", result["label"])
        self.assertIn("no Whisper words available; not evidence of silence", result["label"])
        self.assertIn("Whisper unavailable", notes[0])

    def test_music_fallback_preserves_whisper_doubt(self):
        import music
        import lyrics_shared
        with patch.object(music, "get_lyrics", return_value=WORDS), patch.object(lyrics_shared, "UNSURE", [(9, 12, "Whisper doubtful")]):
            result = L.for_music("unused.mp3", [], 60, off=True, cache_dir=self.tmp.name)
        self.assertIn("unsure: Whisper doubtful", result["lyrics"][0]["text"])
        self.assertIn("Whisper's guesses", result["label"])

    def test_cli_reuses_credits_identity_and_records_off_switch(self):
        ap = argparse.ArgumentParser()
        L.credits.add_args(ap)
        L.add_args(ap)
        args = ap.parse_args(["--artist", "Example Artist", "--title", "Example Song", "--no-lyrics-lookup"])
        L.configure_from_args(args)
        self.assertEqual(L.CONFIG["artist"], "Example Artist")
        self.assertEqual(self.resolve()["lookup"]["status"], "off")
        self.assertEqual(self.calls, [])

    def test_music_card_shows_source_even_without_lines_and_keeps_invitation(self):
        import music
        self.responses = [record(instrumental=True)]
        result = L.for_music("unused.mp3", [], 60, cache_dir=self.tmp.name)
        card = music.format_card({"file": "example.mp3", "duration": 60, "lyrics": result["lyrics"],
                                  "lyrics_align": result["label"]})
        self.assertIn("instrumental (LRCLIB)", card)
        self.assertTrue(card.endswith("and now, you"))

    def test_harmony_timeline_and_json_keep_lrclib_provenance(self):
        import harmony as H
        import numpy as np
        self.responses = [record()]
        result = self.resolve()
        fake = {"align": {"rows": result["rows"], "summary": result["summary"], "label": result["label"]},
                "con": [], "evs": [], "spans": [], "tc": {"step": 2, "T": np.array([0.4] * 31)}}
        with patch.object(H, "build_timeline", return_value=[]):
            block = "\n".join(H.block_lyric_timeline(fake))
        self.assertIn("LRCLIB synced lyrics", block)
        self.assertIn("[0:10≈]", block)
        # to_json's music evidence is deliberately empty; the source should survive intact.
        fake.update(path="example.mp3", dur=60, key_verdict={"key": None}, br=[], cads=[], notes=[],
                    lyrics_source=result["lookup"])
        with patch.object(H, "build_timeline", return_value=[]), patch("ache.analyze", return_value={}):
            data = H.to_json(fake)
        self.assertEqual(data["lyrics_source"]["source"], "LRCLIB")
        self.assertEqual(data["lyrics_aligned"]["label"], result["label"])
        self.assertEqual(data["listener"], "and now, you")

    def test_harmony_voice_events_come_from_whisper_not_supplied_lines(self):
        import harmony as H
        fake = {key: [] for key in ("secs", "spans", "evs", "moves", "pedals", "br", "dips", "drops", "swells", "con", "tag", "csurp")}
        fake.update(tc={}, dur=60, kinfo={}, words=[], align={"rows": []},
                    lines=[{"t": 5, "end": 8, "text": "These words were supplied", "approx": True}])
        with patch.dict(os.environ, {"RAVEN_NO_INSTRUMENTS": "1"}):
            timeline = H.build_timeline(fake)
            self.assertNotIn("voice enters:", str(timeline))
            self.assertIn("voice presence unverified", str(timeline))
            fake["words"] = WORDS
            timeline = H.build_timeline(fake)
        self.assertEqual(timeline[0][0], 10)
        self.assertIn("Carry a lantern", timeline[0][2])
        self.assertNotIn("These words were supplied", str(timeline))

    def test_raw_writer_sheet_headers_are_not_counted_or_placed(self):
        raw = ["Example Song", "(for the paper moon)", "[Verse 1]", TEXT[0],
               "[Chorus]", TEXT[1], "(keep the fire)", TEXT[2]]
        result = self.resolve(lyric_lines=raw)
        self.assertEqual([r["text"] for r in result["rows"]], [TEXT[0], TEXT[1], "(keep the fire)", TEXT[2]])
        self.assertEqual(result["summary"]["total"], 4)
        self.assertIn("3 of 4 lines heard", result["label"])
        self.assertNotIn("[Chorus]", "\n".join(L.card_lines(result)))

    def test_heard_titles_parentheses_and_brackets_remain_sung_lines(self):
        raw = ["Example Song", "(for the paper moon)", "[Chorus]", TEXT[0]]
        words = [(word, i, i + .5) for i, word in enumerate(" ".join(raw).split())]
        result = L.resolve("unused.mp3", [], 60, raw, word_loader=lambda *_: words)
        self.assertEqual([r["text"] for r in result["rows"]], raw)
        self.assertTrue(all(not r["approx"] for r in result["rows"]))

    def test_filename_title_match_does_not_drop_a_title_fragment(self):
        import lyric_align as LA
        rows, _ = LA.align_lyrics(["Paper Moon", "moon", TEXT[0]], WORDS, 60,
                                  filename="Example - Paper Moon.mp3")
        self.assertEqual([r["text"] for r in rows], ["moon", TEXT[0]])
        rows, _ = LA.align_lyrics(["Lantern", TEXT[0]], WORDS, 60, filename="example-lantern.mp3")
        self.assertNotIn("Lantern", [r["text"] for r in rows])

    def test_saved_header_filter_keeps_the_title_as_a_later_lyric(self):
        import lyric_align as LA
        rows = [dict(text=text, t=t, approx=approx) for text, t, approx in
                (("Paper Moon", 0, True), (TEXT[0], 10, False), ("Paper Moon", 20, True))]
        data = dict(lyrics=rows, lyrics_align="LYRICS: writer's text, 1 of 3 lines heard, 2 placed between neighbours (≈)",
                    lyrics_source={"status": "file"})
        text = "LYRICS:\n  [0:00≈] Paper Moon\n  [0:10] Carry a lantern\n  [0:20≈] Paper Moon\n"
        cleaned, data = LA.clean_writer_card(text, data, title="Paper Moon")
        self.assertNotIn("[0:00≈] Paper Moon", cleaned)
        self.assertIn("[0:20≈] Paper Moon", cleaned)
        self.assertEqual(len(data["lyrics"]), 2)

    def test_singer_cli_shows_lookup_designation_and_listener(self):
        import singer
        audio = Path(self.tmp.name) / "example.wav"
        audio.touch()
        self.responses = [record(instrumental=True)]
        output = io.StringIO()
        cp = str(Path(self.tmp.name) / "lookup.json")
        with patch.object(singer, "get_stems", return_value=(self.tmp.name, True)), \
                patch.object(singer, "vocal_presence", return_value=(0.001, 0)), \
                patch("soundfile.info") as info, patch.object(L, "cache_path", return_value=cp), redirect_stdout(output):
            info.return_value.duration = 60
            rc = singer.main([str(audio), "--json", "--no-meter", "--no-stems-timeline", "--no-instruments",
                              "--artist", "Example Artist", "--title", "Example Song"])
        self.assertEqual(rc, 0)
        data = json.loads(output.getvalue())
        self.assertIn("instrumental (LRCLIB)", data["lyrics"]["label"])
        self.assertEqual(data["listener"], "and now, you")


def run_tests(check):
    class Result(unittest.TestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            check("LRCLIB: " + test._testMethodName.removeprefix("test_"), True)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            check("LRCLIB: " + test._testMethodName, False, self._exc_info_to_string(err, test))

        addError = addFailure

    unittest.defaultTestLoader.loadTestsFromTestCase(LyricsTests).run(Result())


if __name__ == "__main__":
    unittest.main()
