"""Optional song services: official API-shaped fakes, no accounts or network."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.parse

import credits
import listen as L
import song_extras as S

RUN_PASS = L.run_pass


def response(data):
    return io.BytesIO(json.dumps(data).encode())


def fingerprint_answer(score=.96, title="Lantern"):
    return dict(status="ok", results=[dict(id="acoustid-track-id", score=score, recordings=[dict(
        id="38035858-f990-4fbb-b3b2-f2f8b958eeba", title=title,
        artists=[dict(id="artist-id", name="Invented Singer")])])])


def linked_recordings_answer(titles, score=.984):
    # Shape reported by the live lookup: one result with several MB recordings.
    return dict(status="ok", results=[dict(id="acoustid-track-id", score=score, recordings=[dict(
        id=f"00000000-0000-0000-0000-{i:012d}", title=title,
        artists=[dict(id="artist-id", name="Taylor Swift")]) for i, title in enumerate(titles)])])


def tags_answer():
    return dict(toptags={"@attr": dict(artist="Invented Singer", track="Lantern"), "tag": [
        dict(name=name, count=str(count), url="https://www.last.fm/tag/example") for name, count in
        (("melancholy", 100), ("breakup", 70), ("piano", 40), ("Invented Singer", 100),
         ("seen live", 90), ("seenlive", 90), ("2016", 90), ("90s", 90), ("favorites", 80),
         ("my favorite songs", 70), ("Lantern", 90), ("Melancholy", 30), ("obscure", 19))]})


class SongExtrasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", L.HERE))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "mystery.mp3"
        self.audio.write_bytes(b"fake audio; never decoded")
        self.fixture = L.load(L.HERE / "fixtures/first-listen-song.json")
        env = dict(ACOUSTID_API_KEY="", LASTFM_API_KEY="", GENIUS_ACCESS_TOKEN="", TYPESAFE_API_KEY="",
                   RAVEN_EARS_OFFLINE="0", RAVEN_SONG_EXTRAS_WORKER="", RAVEN_EARS_NO_BASELINE="0",
                   RAVEN_EARS_HOME=str(self.root / "ears"))
        self.enterContext(patch.dict(os.environ, env))
        self.enterContext(patch("urllib.request.urlopen", side_effect=AssertionError("no network")))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("no network")))
        self.enterContext(patch.object(credits, "file_tags", return_value=(None, None)))
        self.enterContext(patch.object(S.time, "sleep"))
        self.enterContext(patch.object(L, "ears_version", return_value="test-version"))
        self.enterContext(patch("doctored_audio.inspect", return_value=dict(status="ran")))
        self.passes = self.enterContext(patch.object(L, "run_pass", side_effect=self.fake_pass))

    def fake_pass(self, name, args):
        card = deepcopy(self.fixture["cards"][name])
        return dict(status="ran", reason="", text=card["text"], json=card["json"], diagnostics="")

    def args(self, *flags):
        return L.parser().parse_args([str(self.audio), *flags])

    def saved_path(self):
        return L.store_home() / L.file_hash(self.audio) / "song.json"

    def test_no_keys_leaves_every_saved_view_byte_identical(self):
        with patch.object(S, "_request", side_effect=AssertionError("disabled")), \
             patch.object(S, "_fingerprint", side_effect=AssertionError("disabled")), \
             patch.object(credits, "identify", side_effect=AssertionError("disabled")):
            self.assertEqual(S.prepare(self.args(), self.audio), {})
            L.execute(self.args())
        song = L.load(self.saved_path())
        self.assertNotIn("song_extras", song)
        self.assertNotIn("song_extras", song["manifest"]["options"])
        for view in L.VIEWS:
            self.assertEqual(L.render(song, view), L._render(song, view))
        self.assertEqual(S.decorate("exact card\n", {}), "exact card\n")
        self.assertEqual(L.options(self.args()), L.options(self.args("--no-acoustid", "--no-lastfm")))

    def test_acoustid_runs_only_without_any_metadata_and_feeds_passes(self):
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake-acoustid"), \
             patch.object(S, "_fingerprint", return_value=(180., "AQAB-realistic-fingerprint")) as fp:
            fetch = Mock(side_effect=lambda *a, **k: response(fingerprint_answer()))
            for flags in (("--artist", "Given"), ("--title", "Given"), ("--no-acoustid",)):
                self.assertEqual(S.prepare(self.args(*flags), self.audio, fetch=fetch), {})
            with patch.object(credits, "file_tags", return_value=("Tag artist", None)):
                self.assertEqual(S.prepare(self.args(), self.audio, fetch=fetch), {})
            named = self.root / "Invented Singer - Lantern.mp3"
            self.assertEqual(S.prepare(self.args(), named, fetch=fetch), {})
            fp.assert_not_called()
            fetch.assert_not_called()
            with patch.object(S.urllib.request, "urlopen", fetch):
                text, _ = L.execute(self.args())
            self.assertIn("identified by fingerprint (AcoustID 0.96, 1 of 1 linked recordings agree)", text)
            self.assertEqual({(c.args[1].artist, c.args[1].title) for c in self.passes.call_args_list},
                             {("Invented Singer", "Lantern")})
            identity = L.load(self.saved_path())["song_extras"]["identification"]
            self.assertEqual(identity["recording_id"], "38035858-f990-4fbb-b3b2-f2f8b958eeba")
            params = urllib.parse.parse_qs(fetch.call_args.args[0].data.decode())
            self.assertEqual(set(params), {"client", "duration", "fingerprint", "meta", "format"})
            self.assertEqual(params["meta"], ["recordings"])
            self.assertNotIn("mystery", str(params))
            before = self.saved_path().read_bytes()
            with patch.object(S, "prepare", side_effect=AssertionError("saved reads do no lookups")):
                self.assertIn("AcoustID 0.96", L.execute(self.args())[0])
            self.assertEqual(before, self.saved_path().read_bytes())

    def test_acoustid_score_floor_ambiguity_missing_binary_and_positive_cache(self):
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"), \
             patch.object(S, "_fingerprint", return_value=(180., "fingerprint")):
            for score in (.899, float("nan"), 1.1):
                self.assertIsNone(S.identify(self.audio, fetch=lambda *a, **k: response(fingerprint_answer(score))))
            ambiguous = fingerprint_answer()
            ambiguous["results"].extend(fingerprint_answer(.95, "Other Song")["results"])
            self.assertIsNone(S.identify(self.audio, fetch=lambda *a, **k: response(ambiguous)))
            fetch = Mock(side_effect=lambda *a, **k: response(fingerprint_answer(.9)))
            self.assertEqual(S.identify(self.audio, fetch=fetch)["score"], .9)
            self.assertEqual(S.identify(self.audio, fetch=fetch)["score"], .9)
            fetch.assert_called_once()
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"), patch.object(S.shutil, "which", return_value=None), \
             redirect_stderr(io.StringIO()) as err:
            self.assertIsNone(S.identify(self.audio))
        self.assertEqual(err.getvalue().splitlines(), ["AcoustID: fpcalc missing; fingerprint identification skipped"])

    def test_acoustid_linked_recording_majority_and_abstentions(self):
        cases = ((4, 1, True), (2, 1, False), (3, 3, False), (3, 0, True),
                 (1, 0, True), (3, 1, True), (4, 2, True))
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"):
            for yes, no, accepted in cases:
                with self.subTest(yes=yes, no=no), patch.object(S, "_fingerprint",
                        return_value=(180., f"fingerprint-{yes}-{no}")):
                    data = linked_recordings_answer(["The Black Dog"] * yes + ["The Bolter"] * no)
                    result = S.identify(self.audio, fetch=lambda *a, **k: response(data))
                    if accepted:
                        self.assertEqual((result["artist"], result["title"], result["score"]),
                                         ("Taylor Swift", "The Black Dog", .984))
                        self.assertEqual((result["agreeing_recordings"], result["total_recordings"]),
                                         (yes, yes + no))
                    else:
                        self.assertIsNone(result)
            # A large lead alone is insufficient when the rest split their votes.
            data = linked_recordings_answer(["The Black Dog"] * 4 + ["The Bolter", "Other", "Another"])
            with patch.object(S, "_fingerprint", return_value=(180., "split-dissent")):
                self.assertIsNone(S.identify(self.audio, fetch=lambda *a, **k: response(data)))

    def test_acoustid_votes_deduplicate_ids_normalise_pairs_and_filter_scores(self):
        data = linked_recordings_answer(["The Black Dog", "THE BLACK DOG!", "the black dog", "The Black Dog", "The Bolter"])
        records = data["results"][0]["recordings"]
        records[1]["artists"][0]["name"] = "TAYLOR SWIFT"
        records[2]["artists"][0]["name"] = "Taylor   Swift"
        records.extend(deepcopy(records[:4]))
        data["results"].insert(0, dict(score=.95, recordings=deepcopy(records)))
        data["results"].append(dict(score=.999, recordings=[deepcopy(records[4])]))
        data["results"].extend(linked_recordings_answer(["Other"] * 10, score=.899)["results"])
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"), \
             patch.object(S, "_fingerprint", return_value=(180., "deduplicated")):
            result = S.identify(self.audio, fetch=lambda *a, **k: response(data))
        self.assertEqual((result["agreeing_recordings"], result["total_recordings"]), (4, 5))
        self.assertEqual(result["score"], .984)
        # Repeated links must not inflate a two-to-one split into an acceptance.
        data = linked_recordings_answer(["The Black Dog"] * 2 + ["The Bolter"])
        data["results"].extend(deepcopy(data["results"]) * 3)
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"), \
             patch.object(S, "_fingerprint", return_value=(180., "repeated-two-to-one")):
            self.assertIsNone(S.identify(self.audio, fetch=lambda *a, **k: response(data)))

    def test_acoustid_empty_or_unidentified_recordings_abstain(self):
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"):
            for records in ([], [dict(title="The Black Dog", artists=[dict(name="Taylor Swift")])]):
                with self.subTest(records=records), patch.object(S, "_fingerprint", return_value=(180., "empty")):
                    data = dict(status="ok", results=[dict(score=.984, recordings=records)])
                    self.assertIsNone(S.identify(self.audio, fetch=lambda *a, **k: response(data)))

    def test_acoustid_majority_saved_card_and_cache_preserve_agreement(self):
        data = linked_recordings_answer(["The Black Dog"] * 4 + ["The Bolter"])
        fetch = Mock(side_effect=lambda *a, **k: response(data))
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"), \
             patch.object(S, "_fingerprint", return_value=(180., "black-dog")), \
             patch.object(S.urllib.request, "urlopen", fetch):
            text, _ = L.execute(self.args())
            cached = S.identify(self.audio)
        fetch.assert_called_once()
        song = L.load(self.saved_path())
        self.assertEqual(song["song_extras"]["identification"], cached)
        line = "identified by fingerprint (AcoustID 0.98, 4 of 5 linked recordings agree)"
        self.assertIn(line, text)
        self.assertEqual({(c.args[1].artist, c.args[1].title) for c in self.passes.call_args_list},
                         {("Taylor Swift", "The Black Dog")})
        with patch.object(S, "prepare", side_effect=AssertionError("saved reads do no lookups")):
            for view in L.VIEWS:
                self.assertEqual(L.render(song, view)[0].count(line), 1)
        legacy = {k: v for k, v in cached.items() if k not in ("agreeing_recordings", "total_recordings")}
        self.assertIn("identified by fingerprint (AcoustID 0.98)", S.decorate("card", {"identification": legacy}))

    def test_tags_filter_floor_and_cache_keep_measurements_unchanged(self):
        self.assertEqual(S.filter_tags(tags_answer(), "Invented Singer", "Lantern"), ["melancholy", "breakup", "piano"])
        baseline, _ = L.execute(self.args())
        original_cards = L.load(self.saved_path())["cards"]
        fetch = Mock(side_effect=lambda *a, **k: response(tags_answer()))
        with patch.dict(os.environ, LASTFM_API_KEY="fake-lastfm"), patch.object(S.urllib.request, "urlopen", fetch):
            text, _ = L.execute(self.args("--fresh", "--artist", "Invented Singer", "--title", "Lantern"))
            self.assertEqual(S.listener_tags("Invented Singer", "Lantern"), ["melancholy", "breakup", "piano"])
        self.assertEqual(fetch.call_count, 1)
        song = L.load(self.saved_path())
        self.assertEqual(song["cards"], original_cards)
        evidence = song['song_extras']['lastfm_tag_evidence']
        self.assertEqual(evidence['top_count'], 100)
        self.assertEqual([r['count'] for r in evidence['counts']], [100, 70, 40])
        self.assertEqual(text, S.decorate(baseline, {"lastfm_tags": ["melancholy", "breakup", "piano"]}))
        for view in ("first-listen", "full"):
            self.assertEqual(L.render(song, view)[0].count("LISTENERS SAY"), 1)

    def test_relative_tag_support_and_legacy_count_absence(self):
        tags = dict(toptags=dict(tag=[dict(name=name, count=count) for name, count in
            [('genre', 100), ('style', 40), ('personal label', 39), ('another label', 20), ('tiny', 19)]]))
        evidence = S.tag_evidence(tags, 'Artist', 'Title')
        self.assertEqual(evidence['tags'], ['genre', 'style'])
        self.assertEqual(evidence['top_count'], 100)
        self.assertEqual([r['count'] for r in evidence['counts']], [100, 40])
        self.assertEqual(S.feeling_extras(dict(lastfm_tags=['genre', 'personal label']))['lastfm_tags'], ['genre', 'personal label'])  # legacy: kept as saved
        extras = dict(lastfm_tags=['genre', 'style', 'personal label'], lastfm_tag_evidence=evidence)
        self.assertEqual(S.feeling_extras(extras)['lastfm_tags'], ['genre', 'style'])
        self.assertEqual(extras['lastfm_tags'], ['genre', 'style', 'personal label'])
        # Old reduced caches are never trusted as counted evidence.
        with patch.object(S, '_cached', return_value=evidence) as cached, patch.dict(os.environ, LASTFM_API_KEY='fake'):
            self.assertEqual(S.listener_tags('Artist', 'Title'), ['genre', 'style'])
        self.assertEqual(cached.call_args.args[1], ['Artist', 'Title', 'relative-v1'])

    def test_fingerprint_binary_protocol_and_metadata_match_guards(self):
        with patch.object(S.shutil, "which", return_value="fpcalc"), \
             patch.object(S.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(
                 dict(duration=180.4, fingerprint="AQAB-fingerprint")))) as run:
            self.assertEqual(S._fingerprint(self.audio), (180.4, "AQAB-fingerprint"))
        self.assertEqual(run.call_args.args[0], ["fpcalc", "-json", str(self.audio.resolve())])
        self.assertTrue(run.call_args.kwargs["check"])
        wrong = tags_answer()
        wrong["toptags"]["@attr"]["artist"] = "Other artist"
        with patch.dict(os.environ, LASTFM_API_KEY="fake"):
            self.assertIsNone(S.listener_tags("Invented Singer", "Lantern",
                fetch=lambda *a, **k: response(wrong)))
        tags = dict(toptags=dict(tag=[dict(name="soft", count=20), dict(name="rare", count=19),
            dict(name="best of 2020", count=100)] + [dict(name=f"tag {i}", count=90-i) for i in range(7)]))
        self.assertEqual(len(S.filter_tags(tags, "Artist", "Title")), S.MAX_TAGS)
        self.assertEqual(S.filter_tags(dict(toptags=dict(tag=dict(name="soft", count="20"))), "A", "T"), ["soft"])

    def test_json_rate_limit_date_retry_after_and_genius_rejection(self):
        from email.utils import formatdate
        for service, fetch in (("Last.fm", Mock(side_effect=lambda *a, **k: response(dict(error=29, message="secret")))),
                ("Genius", Mock(side_effect=urllib.error.HTTPError("https://example.invalid", 429, "secret",
                    {"Retry-After": formatdate(S.time.time() + 300, usegmt=True)}, None)))):
            with redirect_stderr(io.StringIO()) as err:
                S._attempt(service, lambda: S._request(service, S.GENIUS, {}, fetch=fetch))
                S._attempt(service, lambda: S._request(service, S.GENIUS, {}, fetch=fetch))
            self.assertEqual(len(err.getvalue().splitlines()), 2)
            self.assertNotIn("secret", err.getvalue())
            fetch.assert_called_once()
            self.assertGreater(float((S._home() / service / ".rate").read_text()), S.time.time() + 50)
        # A 401 during lore prints exactly one safe diagnostic, no second failure.
        (S._home() / "Genius" / ".rate").unlink()
        song = deepcopy(self.fixture)
        song["manifest"]["options"].update(artist="Invented Singer", title="Lantern")
        with patch.dict(os.environ, GENIUS_ACCESS_TOKEN="fake"), redirect_stderr(io.StringIO()) as err:
            text = S.lore(song, self.audio, fetch=Mock(side_effect=urllib.error.HTTPError(
                "https://example.invalid/?token=secret", 401, "secret", {}, None)))
        self.assertEqual(text, "")
        self.assertEqual(err.getvalue(), "Genius: HTTP 401; skipped\n")

    def test_lore_rejects_wrong_song_and_uses_the_saved_filename_after_rename(self):
        song = deepcopy(self.fixture)
        song["manifest"]["options"].update(artist=None, title=None, metadata_filename="Invented Singer - Lantern.mp3")
        wrong = dict(meta=dict(status=200), response=dict(hits=[dict(type="song", result=dict(
            id=1, title="Lantern", primary_artist=dict(name="Other singer")))]))
        with patch.dict(os.environ, GENIUS_ACCESS_TOKEN="fake-genius"), redirect_stderr(io.StringIO()) as err:
            self.assertEqual(S.lore(song, self.audio, fetch=lambda *a, **k: response(wrong)), "")
            self.assertIn("no matching song", err.getvalue())
            self.assertIn(S.LORE_SEPARATOR, S.lore(song, "Different - Wrong.mp3", fetch=self.genius_fetch))

    def test_offline_and_off_switches_beat_caches_and_launch_nothing(self):
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake", LASTFM_API_KEY="fake", GENIUS_ACCESS_TOKEN="fake"), \
             patch.object(S, "_cached", side_effect=AssertionError("disabled")), \
             patch.object(S, "_fingerprint", side_effect=AssertionError("disabled")):
            self.assertEqual(S.prepare(self.args("--no-acoustid", "--no-lastfm"), self.audio), {})
            with patch.dict(os.environ, RAVEN_EARS_OFFLINE="1"):
                self.assertEqual(S.prepare(self.args(), self.audio), {})
                self.assertIsNone(S.identify(self.audio))
                self.assertIsNone(S.listener_tags("Artist", "Title"))
                self.assertIn("offline", S.lore(self.fixture, self.audio))
                with self.assertRaisesRegex(RuntimeError, "RAVEN_EARS_OFFLINE"):
                    S._request("Genius", S.GENIUS, {})

    def test_http_rejections_are_single_safe_line_and_429_is_not_retried(self):
        for service, code in ((s, c) for s in S.KEYS for c in (401, 403, 429)):
            with self.subTest(service=service, code=code):
                fetch = Mock(side_effect=urllib.error.HTTPError("https://example.invalid/?key=secret", code,
                    "secret must not print", {"Retry-After": "120"}, None))
                with redirect_stderr(io.StringIO()) as err:
                    self.assertIsNone(S._attempt(service, lambda: S._request(service, S.GENIUS, {}, fetch=fetch)))
                self.assertEqual(len(err.getvalue().splitlines()), 1)
                self.assertIn(f"HTTP {code}", err.getvalue())
                self.assertNotIn("secret", err.getvalue())
                fetch.assert_called_once()
                if code == 429:
                    with redirect_stderr(io.StringIO()):
                        S._attempt(service, lambda: S._request(service, S.GENIUS, {}, fetch=fetch))
                    fetch.assert_called_once()

    def test_lore_refuses_without_saved_song_and_without_running_passes(self):
        with patch.dict(os.environ, GENIUS_ACCESS_TOKEN="fake"), \
             patch.object(S, "lore", side_effect=AssertionError("must listen first")), \
             redirect_stdout(io.StringIO()) as out:
            self.assertEqual(L.main([str(self.audio), "--lore"]), 0)
        self.assertEqual(out.getvalue(), "listen first; the lore comes after\n")
        self.passes.assert_not_called()
        self.assertFalse(self.saved_path().exists())
        for flag in ("--fresh", "--run-missing", "--json", "--history", "--delete"):
            with self.assertRaisesRegex(ValueError, "only reads an existing card"):
                L.execute(self.args("--lore", flag))

    def test_genius_key_alone_does_not_request_or_change_a_card(self):
        baseline, _ = L.execute(self.args())
        with patch.dict(os.environ, GENIUS_ACCESS_TOKEN="fake-genius"), \
             patch.object(S, "lore", side_effect=AssertionError("opt in only")):
            self.assertEqual(L.execute(self.args("--fresh"))[0], baseline)
        self.assertNotIn("song_extras", L.load(self.saved_path()))

    def genius_fetch(self, req, timeout):
        url = urllib.parse.urlsplit(req.full_url)
        query = urllib.parse.parse_qs(url.query)
        self.assertEqual(req.get_header("Authorization"), "Bearer fake-genius")
        self.assertEqual(url.netloc, "api.genius.com")
        self.assertEqual(query["text_format"], ["plain"])
        song = dict(id=123, title="Lantern", primary_artist=dict(name="Invented Singer"),
                    description=dict(plain="A community account of the song. " * 80))
        if url.path == "/search":
            data = dict(hits=[dict(type="song", result=dict(song, title="Other Song")), dict(type="song", result=song)])
        elif url.path == "/songs/123":
            data = dict(song=song)
        else:
            self.assertEqual(url.path, "/referents")
            self.assertEqual(query["song_id"], ["123"])
            # Real Genius shape: classification is a review state, annotatable names the song.
            data = dict(referents=[dict(classification="unreviewed" if i == 0 else "accepted",
                                        annotatable=dict(type="Song", id=123),
                                        fragment="Carry the lantern " * 30,
                annotations=[dict(id=i, votes_total=i, body=dict(plain="An interpretation. " * 100),
                                  authors=[dict(user=dict(name="A contributor"))])]) for i in range(6)])
        return response(dict(meta=dict(status=200), response=data))

    def test_lore_follows_both_card_views_and_never_changes_saved_text(self):
        L.execute(self.args("--artist", "Invented Singer", "--title", "Lantern"))
        before = self.saved_path().read_bytes()
        fetch = Mock(side_effect=self.genius_fetch)
        with patch.dict(os.environ, GENIUS_ACCESS_TOKEN="fake-genius"), patch.object(S.urllib.request, "urlopen", fetch):
            for view in ("first-listen", "full"):
                normal, _ = L.execute(self.args("--view", view))
                out = io.StringIO()
                def checked_fetch(*args, **kwargs):
                    self.assertEqual(out.getvalue(), normal)
                    return self.genius_fetch(*args, **kwargs)
                fetch.side_effect = checked_fetch
                with redirect_stdout(out), redirect_stderr(io.StringIO()):
                    self.assertEqual(L.main([str(self.audio), "--lore", "--view", view]), 0)
                self.assertTrue(out.getvalue().startswith(normal + S.LORE_SEPARATOR))
                lore = out.getvalue()[len(normal):]
                self.assertIn("Genius annotation 5", lore)
                self.assertNotIn("Genius annotation 2", lore)
                self.assertLess(lore.index("annotation 5"), lore.index("annotation 4"))
                self.assertNotIn("Description", normal)
                self.assertLess(len(lore), 3100)
        self.assertEqual(fetch.call_count, 3)  # second view uses reduced positive cache
        self.assertEqual(self.saved_path().read_bytes(), before)
        caches = list((S._home() / "Genius").glob("*.json"))
        self.assertEqual(len(caches), 1)
        cached = caches[0].read_text()
        self.assertNotIn("fake-genius", cached)
        self.assertNotIn("lyrics", cached)

    def test_speech_cli_with_all_three_keys_matches_original_card_and_calls_none(self):
        import raven_ears_plus as speech
        from test_cpu_profile import card_fixture
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake", LASTFM_API_KEY="fake", GENIUS_ACCESS_TOKEN="fake"), \
             patch.object(S, "prepare", side_effect=AssertionError("music only")), \
             patch.object(S, "identify", side_effect=AssertionError("music only")), \
             patch.object(S, "listener_tags", side_effect=AssertionError("music only")), \
             patch.object(S, "lore", side_effect=AssertionError("music only")), \
             card_fixture() as (path, *_), patch("voice_book.identify", return_value=None), \
             redirect_stdout(io.StringIO()) as out:
            self.assertEqual(speech.main([str(path), "--quiet", "--no-baseline-update", "--no-picture"]), 0)
        self.assertEqual(out.getvalue().encode(), (L.HERE / "fixtures/speech-default.card.txt").read_bytes())

    def test_worker_capture_scrub_save_keeps_optional_evidence(self):
        # The actual run_pass -> worker -> capture -> privacy scrub -> song.json
        # route, with only the model-facing CLI replaced by a tiny fake.
        def fake_main(argv, capture):
            self.assertIn("--artist", argv)
            capture(dict(duration=180, song_extras={"lastfm_tags": ["piano"]}))
            print("MUSIC: saved test card\nKEY: C major")
        def bridge(argv, **kwargs):
            out = io.StringIO()
            with patch.dict(os.environ, kwargs["env"], clear=True), redirect_stdout(out):
                code = L.worker(argv[3], argv[4], argv[5:])
            return subprocess.CompletedProcess(argv, code, out.getvalue(), "")
        extras = dict(identification=dict(artist="Invented Singer", title="Lantern", score=.96,
                                         recording_id="recording-id", agreeing_recordings=4,
                                         total_recordings=5), lastfm_tags=["piano"])
        with patch.dict(os.environ, ACOUSTID_API_KEY="fake"), patch.object(S, "prepare", return_value=extras), \
             patch.object(L, "run_pass", wraps=RUN_PASS), patch.object(L.subprocess, "run", side_effect=bridge), \
             patch.object(L.importlib, "import_module", return_value=SimpleNamespace(main=fake_main)):
            text, _ = L.execute(self.args("--only", "music", "--view", "music"))
        song = L.load(self.saved_path())
        self.assertEqual(song["song_extras"], extras)
        self.assertEqual(song["cards"]["music"]["json"]["song_extras"], {"lastfm_tags": ["piano"]})
        self.assertIn("AcoustID 0.96, 4 of 5 linked recordings agree", text)
        self.assertIn("LISTENERS SAY", text)


def run_tests(check):
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(SongExtrasTests).run(result)
    problems = {test.id().rsplit(".", 1)[-1]: detail for test, detail in result.failures + result.errors}
    for name in unittest.defaultTestLoader.getTestCaseNames(SongExtrasTests):
        check("optional song services: " + name, name not in problems, problems.get(name, ""))


class GeniusAnnotationShapeTests(unittest.TestCase):
    """16 Nov 2026: Genius's classification is a review state; the old filter dropped every annotation."""

    def test_reviewed_first_unreviewed_labelled_other_pages_skipped(self):
        song = dict(id=7, title="Lantern", primary_artist=dict(name="Invented Singer"), description=dict(plain="About it."))
        def note(i, votes):
            return dict(id=i, votes_total=votes, body=dict(plain=f"Reading {i}."), authors=[])
        referents = [dict(classification="unreviewed", annotatable=dict(type="Song"), fragment="dig", annotations=[note(1, 99)]),
                     dict(classification="accepted", annotatable=dict(type="Song"), fragment="bury", annotations=[note(2, 3)]),
                     dict(classification="accepted", annotatable=dict(type="Album"), fragment="elsewhere", annotations=[note(3, 500)])]
        def fake(service, url, params, token=None, fetch=None):
            if url.endswith("/search"):
                return dict(meta=dict(status=200), response=dict(hits=[dict(type="song", result=song)]))
            if url.endswith("/songs/7"):
                return dict(meta=dict(status=200), response=dict(song=song))
            return dict(meta=dict(status=200), response=dict(referents=referents))
        with patch.dict(os.environ, GENIUS_ACCESS_TOKEN="fake"), patch.object(S, "_request", fake):
            result = S._genius_lore("Invented Singer", "Lantern", None)
        self.assertEqual([a["id"] for a in result["annotations"]], [2, 1])
        self.assertEqual([a["reviewed"] for a in result["annotations"]], [True, False])


if __name__ == "__main__":
    unittest.main()
