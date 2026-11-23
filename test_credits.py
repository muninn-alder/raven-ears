"""Original-release regression checks using public metadata rows; no network or audio."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import credits as C
import stage_status as ST


ARTIST, TITLE = "The Dave Brubeck Quartet", "Take Five"


class OriginalReleaseTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((Path(__file__).parent / "fixtures/discogs-original.json").read_text())
        self.calls = []
        self.transport = C._discogs_json
        self.enterContext(patch.object(C, "_discogs_json", side_effect=self.api))
        self.enterContext(patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")))

    def api(self, url, token):
        self.calls.append(url)
        parsed = urlparse(url)
        if parsed.path == "/database/search":
            typ = parse_qs(parsed.query)["type"][0]
            return copy.deepcopy(self.data.get(typ + "_search", self.data["search"]))
        group, key = parsed.path.strip("/").split("/")
        return copy.deepcopy(self.data[group][key])

    def lookup(self, duration=None):
        result = C.lookup_discogs(ARTIST, TITLE, "fake-token", duration=duration)
        self.assertLessEqual(len(self.calls), C.DISCOGS_MAX_REQS)
        return result

    def only_album(self):
        self.data["search"]["results"] = [r for r in self.data["search"]["results"] if r["id"] == 34081]

    def alternate_album(self):
        # Mutations below keep the fetched row/track/credit shape.
        row = copy.deepcopy(self.data["search"]["results"][0])
        row.update(id=1, master_id=1, title=ARTIST + " - Alternate Album")
        self.data["search"]["results"].append(row)
        master = copy.deepcopy(self.data["masters"]["34081"])
        master.update(id=1, main_release=2, title="Alternate Album")
        self.data["masters"]["1"] = master
        release = copy.deepcopy(self.data["releases"]["12384502"])
        release.update(id=2, master_id=1, title="Alternate Album")
        self.data["releases"]["2"] = release
        return row, master, release

    def test_take_five_uses_original_quartet_not_concord(self):
        result = self.lookup()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["matched"], ARTIST + " - Time Out (1959)")
        self.assertEqual({e["who"]: e["words"] for e in result["entries"]}, {
            "Paul Desmond": ["Alto Saxophone"], "Eugene Wright": ["Bass"],
            "Joe Morello": ["Drums"], "Dave Brubeck": ["Piano"]})
        self.assertEqual(parse_qs(urlparse(self.calls[0]).query)["per_page"], ["100"])
        self.assertEqual(result["release_id"], self.data["masters"]["34081"]["main_release"])

    def test_empty_original_does_not_borrow_later_players(self):
        self.data["releases"]["12384502"]["extraartists"] = []
        result = self.lookup()
        self.assertEqual(result["status"], "noperf")
        self.assertIn("Time Out (1959)", result["matched"])
        self.assertEqual(result["entries"], [])

    def test_earlier_ep_beats_later_album(self):
        self.data["search"]["results"] = [r for r in self.data["search"]["results"] if r["id"] in (34081, 664785)]
        next(r for r in self.data["search"]["results"] if r["id"] == 34081)["year"] = "1960"
        self.data["masters"]["34081"]["year"] = 1960
        self.data["releases"]["12384502"]["year"] = 1960
        result = self.lookup()
        self.assertIn("Experiments In Time", result["matched"])

    def test_album_year_tie_is_uncertain(self):
        self.only_album()
        self.alternate_album()
        result = self.lookup()
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("tie", result["detail"])
        self.assertEqual(result["entries"], [])

    def test_duration_can_resolve_same_year_album_tie(self):
        self.only_album()
        _, _, release = self.alternate_album()
        self.data["releases"]["12384502"]["tracklist"][0]["duration"] = "5:24"
        release["tracklist"][0]["duration"] = "8:50"
        result = self.lookup(duration=324)
        self.assertEqual(result["status"], "ok")
        self.assertIn("Time Out (1959)", result["matched"])

    def test_real_original_missing_length_is_neutral_with_known_audio_length(self):
        result = self.lookup(duration=326)
        self.assertEqual(result["status"], "ok")
        self.assertIn("track length not listed", result["match_note"])
        self.assertIn("Time Out (1959)", result["matched"])
        self.assertEqual(len(result["entries"]), 4)

    def test_missing_length_does_not_lose_same_year_tie_to_known_length(self):
        self.only_album()
        _, _, release = self.alternate_album()
        release["tracklist"][0]["duration"] = "5:26"
        result = self.lookup(duration=326)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("tie", result["detail"])
        self.assertEqual(result["entries"], [])

    def test_far_duration_is_uncertain(self):
        self.only_album()
        self.data["releases"]["12384502"]["tracklist"][0]["duration"] = "8:50"
        result = self.lookup(duration=324)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("differs by 206 s", result["detail"])
        self.assertEqual(result["studio_duration_s"], 530)

    def test_duration_survives_uncertain_credits_chain_and_cache(self):
        self.only_album()
        self.data["releases"]["12384502"]["tracklist"][0]["duration"] = "2:52"
        with patch.object(C, "lookup_musicbrainz", return_value={"found": False, "entries": []}), \
             patch.object(C, "discogs_token", return_value="fake-token"), tempfile.TemporaryDirectory() as directory:
            result = C.get("", ARTIST, TITLE, duration=192, cache_dir=directory)
            self.assertEqual(result["studio_durations"], [{"source": "Discogs", "duration_s": 172}])
            self.assertEqual(result["entries"], [])
            self.calls.clear()
            self.assertEqual(C.get("", ARTIST, TITLE, duration=192, cache_dir=directory), result)
            self.assertEqual(self.calls, [])

    def test_live_or_ambiguous_original_is_not_studio_duration_evidence(self):
        self.only_album()
        self.data["releases"]["12384502"]["tracklist"][0]["duration"] = "2:52"
        self.alternate_album()
        self.assertNotIn("studio_duration_s", self.lookup(192))

    def test_missing_year_and_reissue_main_release_are_uncertain(self):
        self.only_album()
        self.data["masters"]["34081"]["year"] = 0
        self.assertIn("year unknown", self.lookup()["detail"])
        self.calls.clear()
        self.data["masters"]["34081"]["year"] = 1959
        self.data["releases"]["12384502"]["year"] = 1997
        self.assertIn("main release year differs", self.lookup()["detail"])

    def test_live_only_is_uncertain_even_with_a_master(self):
        self.data["search"]["results"] = [r for r in self.data["search"]["results"] if r["id"] == 270654]
        result = self.lookup()
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("Concord On A Summer Night (1982)", result["matched"])

    def test_release_without_master_is_attributed_but_withheld(self):
        self.data["master_search"] = {"results": []}
        row = copy.deepcopy(self.data["search"]["results"][0])
        row.update(id=12384502, master_id=0, type="release")
        self.data["release_search"] = {"results": [row]}
        result = self.lookup()
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("no master", result["detail"])
        self.assertIn("Time Out (1959)", result["matched"])

    def test_release_search_resolves_master_instead_of_using_reissue(self):
        self.only_album()
        self.data["master_search"] = {"results": []}
        row = copy.deepcopy(self.data["search"]["results"][0])
        row.update(id=987, type="release", year="1997")
        self.data["release_search"] = {"results": [row]}
        self.assertEqual(self.lookup()["release_id"], 12384502)
        self.assertFalse(any(url.endswith("/releases/987") for url in self.calls))

    def test_pagination_and_detail_budget_never_make_a_certain_match(self):
        self.data["search"]["pagination"]["pages"] = 2
        self.assertIn("incomplete", self.lookup()["detail"])
        self.calls.clear()
        self.data["search"]["pagination"]["pages"] = 1
        self.only_album()
        for ident in (1, 3, 5):
            row, master, release = self.alternate_album()
            row["id"] = row["master_id"] = ident
            self.data["masters"][str(ident)] = dict(master, id=ident, main_release=ident+1)
            self.data["releases"][str(ident+1)] = dict(release, id=ident+1)
        self.assertIn("incomplete", self.lookup()["detail"])

    def test_uncertain_match_reaches_card_without_renaming_any_instrument(self):
        self.only_album()
        self.data["releases"]["12384502"]["year"] = 1997
        with patch.object(C, "lookup_musicbrainz", return_value={"found": False, "entries": []}), \
             patch.object(C, "discogs_token", return_value="fake-token"):
            value = C.get("", ARTIST, TITLE, use_cache=False)
        self.assertIn("Time Out (1997)", value["note"])
        self.assertIn("match uncertain", value["note"])
        self.assertIn("ears only", value["note"])
        self.assertIn("degraded", [r["status"] for r in value["stage_status"]])
        self.assertNotIn("credits: none found (Discogs)", ST.summary(value["stage_status"]))
        heard = {"families": {"keys (organ or piano?)": {"stem": "other"}}}
        rendered = C.apply(heard, value)
        self.assertEqual(rendered["families"], heard["families"])
        self.assertIn("match uncertain", rendered["credit_lines"][0])

    def test_old_policy_and_unknown_duration_cache_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            for old_policy in ("v2", "v3-original|None", "v3-original|326.0"):
                old_key = hashlib.sha1(("the dave brubeck quartet|take five|" + old_policy).encode()).hexdigest()[:16] + ".json"
                Path(directory, old_key).write_text(json.dumps({"entries": [{"who": "Wrong"}]}))
            with patch.object(C, "lookup_musicbrainz", return_value={"found": False, "entries": []}), \
                 patch.object(C, "discogs_token", return_value="fake-token"):
                first = C.get("", ARTIST, TITLE, cache_dir=directory)
                self.assertIn("Time Out (1959)", first["label"])
                self.calls.clear()
                C.get("", ARTIST, TITLE, cache_dir=directory)
                self.assertFalse(self.calls)
                known = C.get("", ARTIST, TITLE, cache_dir=directory, duration=326)
                self.assertTrue(self.calls)
                self.assertIn("track length not listed", known["label"])
                self.assertIsNone(known["note"])
                self.calls.clear()
                again = C.get("", ARTIST, TITLE, cache_dir=directory, duration=326)
                self.assertEqual(again["label"], known["label"])
                self.assertFalse(self.calls)

    def test_captured_generic_musicbrainz_performer_allows_original_discogs_on_card(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/musicbrainz-performers.json").read_text())[TITLE]
        for kind in ("Group", "Person"):
            with self.subTest(artist_type=kind):
                relations = copy.deepcopy(fixture["recording"]["relations"])
                relations[0]["artist"]["type"] = kind
                with patch.object(C, "search_recordings", return_value=[fixture["search"]]), \
                     patch.object(C, "recording_relations", return_value=relations), \
                     patch.object(C, "release_track_relations", return_value=[]), \
                     patch.object(C, "discogs_token", return_value="fake-token"):
                    result = C.get("", ARTIST, TITLE, duration=326, use_cache=False)
                self.assertEqual(result["source"], "discogs")
                self.assertEqual(result["studio_durations"], [{"source": "MusicBrainz", "duration_s": 324}])
                self.assertIn("Time Out (1959)", result["label"])
                self.assertIn("master main release; track length not listed", result["label"])
                self.assertIn("lists no performer credits", result["label"])
                self.assertNotIn("performer", [w for e in result["entries"] for w in e["words"]])
                self.assertEqual({e["who"] for e in result["entries"]},
                                 {"Paul Desmond", "Eugene Wright", "Joe Morello", "Dave Brubeck"})
                card = C.apply({"duration": 326, "families": {}}, result)
                self.assertIn(result["label"], card["credit_lines"][0])
                self.assertTrue(all("credited, not clearly heard" in line for line in card["credit_lines"][2:]))

    def test_generic_performer_with_only_modifiers_names_no_instrument(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/musicbrainz-performers.json").read_text())[TITLE]
        relations = copy.deepcopy(fixture["recording"]["relations"])
        relations[0]["attributes"] = ["additional", "guest"]
        self.assertEqual(C.entries_from_relations(relations), [])

    def test_captured_musicbrainz_instrument_and_vocal_roles_remain_identical(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/musicbrainz-performers.json").read_text())
        for title in ("Strawberry Fields Forever", "Twilight of the Thunder God", "Tolerate It"):
            with self.subTest(title=title):
                row = fixture[title]
                with patch.object(C, "search_recordings", return_value=[row["search"]]), \
                     patch.object(C, "recording_relations", return_value=row["recording"]["relations"]):
                    artist = row["search"]["artist-credit"][0]["name"]
                    result = C._from_lookup(C.chain(artist, title, duration=326), artist, title)
                self.assertEqual(result["source"], "musicbrainz")
                self.assertEqual(result["entries"], row["expected_entries"])
                self.assertEqual(result["label"], "MusicBrainz: " + artist + " - " + row["search"]["title"])
        self.assertFalse(self.calls)

    def test_duration_reaches_lookup_and_has_separate_memo(self):
        with patch.dict(C._MEMO, {}, clear=True), patch.object(C, "get", return_value={"entries": []}) as get:
            C.apply_for("example.wav", {"duration": 324})
            C.for_song("example.wav", duration=324)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(get.call_args.kwargs["duration"], 324)
            C.for_song("example.wav")
            self.assertEqual(get.call_count, 2)

    def test_musicbrainz_with_players_still_prevents_discogs_requests(self):
        entries = C.parse_credit_text("Example Player: piano")
        with patch.object(C, "lookup_musicbrainz", return_value={"entries": entries, "matched": "Example"}):
            result = C.chain(ARTIST, TITLE, duration=324)
        self.assertEqual(result["source"], "musicbrainz")
        self.assertFalse(self.calls)

    def test_transport_spaces_requests_and_keeps_token_out_of_url(self):
        now, started = [100.0], []
        def respond(request, timeout):
            started.append(now[0])
            self.assertNotIn("fake-token", request.full_url)
            self.assertEqual(request.get_header("Authorization"), "Discogs token=fake-token")
            body = io.BytesIO(b'{"results": []}')
            body.headers = {}
            return body
        def sleep(seconds):
            now[0] += seconds
        with patch.object(C.time, "time", side_effect=lambda: now[0]), \
             patch.object(C.time, "sleep", side_effect=sleep), patch.object(C, "_dc_last", [0.0]), \
             patch("urllib.request.urlopen", side_effect=respond):
            for _ in range(3):
                self.transport(C.DISCOGS + "database/search", "fake-token")
        self.assertTrue(all(b - a >= 1.0 for a, b in zip(started, started[1:])))


if __name__ == "__main__":
    unittest.main()
