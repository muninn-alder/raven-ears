"""Mocked HTTP and saved cards only. Run with python3 test_jev_reads.py."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import jev_reads as J
import stage_status as ST
import slim


def answer(score=2, masked=.1, longing=.1):
    return {"answers": {"stated_feeling": {"score": score, "legend": "saved legend",
                                           "probabilities": [0, 0, 1, 0, 0], "confidence": .8},
                        "masked": {"noul": masked}, "longing": {"noul": longing}}}


def response(score=2, masked=.1, longing=.1):
    return io.BytesIO(json.dumps(answer(score, masked, longing)).encode())


def lines(count=5):
    return [dict(t=i * 10, end=i * 10 + 8, text=f"Invented line {i}") for i in range(count)]


def contrasts(rows):
    return [dict(t=r["t"], text=r["text"], words=.1, music=.1, lead="energy", flag=None) for r in rows]


def harmony():
    if importlib.util.find_spec("numpy") is None:
        raise unittest.SkipTest("harmony integration needs the ears' NumPy environment")
    import harmony as H
    return H


class JevTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TYPESAFE_API_KEY": "invented-test-key",
                                          "RAVEN_EARS_OFFLINE": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_no_key_no_call_no_notes_no_mutation(self):
        rows = lines()
        con, notes = contrasts(rows), []
        before = deepcopy(con)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), patch.object(J.request, "urlopen") as http:
            self.assertIs(J.add_contrast(rows, con, notes=notes), con)
            self.assertEqual(con, before)
            self.assertEqual(notes, [])
            http.assert_not_called()

    def test_no_key_committed_fixture_views_byte_identical(self):
        harmony()  # Saved lyric rendering also needs the ears' NumPy dependency.
        import listen
        # Captured with the committed slim.py before Jev edits, not regenerated
        # by the implementation under test. Fixtures deliberately omit con JSON.
        # First-listen hashes include the explicit librosa estimate label;
        # removing only that label reproduces the previous hashes exactly.
        golden = {
            ("first-listen-song.json", "harmony"): "930b1e4a4ddca7327fad89620963658fb1fb7f0f62d24003e235339595d2bd52",
            ("first-listen-song.json", "summary"): "b50c5bdde7e959ed732fecfa5caeafe37dda80eac7ed2ba68967c102de034ae5",
            ("uhuru-title-song.json", "harmony"): "3e771683e0f7d87c4063c441c7d9ff65ccef93d50f6cc0da7a0c05cd0c2026a5",
            ("uhuru-title-song.json", "summary"): "419c5c0389c711251f5e40ac1108b824807f6c0f7ea5fcb365deb70ff95fcacf",
        }
        for fixture in ("first-listen-song.json", "uhuru-title-song.json"):
            song = json.loads((Path(__file__).parent / "fixtures" / fixture).read_text())
            before = deepcopy(song)
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), patch.object(J.request, "urlopen") as http:
                data = song["cards"]["harmony"]["json"] or {}
                data["contrast"] = J.add_contrast([], data.get("contrast", []))
                for view in ("harmony", "summary"):
                    self.assertEqual(listen.render(song, view), listen.render(before, view))
                    rendered = listen.render(song, view)[0]
                    self.assertEqual(hashlib.sha256(rendered.encode()).hexdigest(), golden[fixture, view])
                http.assert_not_called()

    def test_no_key_preserves_legacy_flag_text(self):
        H = harmony()
        con = contrasts(lines(1))
        con[0].update(words=-.8, music=.5)
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}), patch.object(J.request, "urlopen") as http:
            con = J.add_contrast(lines(1), con)
            text = "\n".join(H.block_contrast(dict(con=con)))
            self.assertIn("  0:00 'Invented line 0': words heavy (-0.80), music lifting (lift +0.5, energy leads)", text)
            self.assertNotIn("Jev", text)
            http.assert_not_called()

    def test_score_mapping(self):
        for score, mapped in ((0, -1), (1, -.5), (1.5, -.25), (2, 0), (2.5, .25), (4, 1)):
            with self.subTest(score=score):
                self.assertEqual(J.decode(answer(score)["answers"])["stated_feeling"], mapped)

    def test_noul_thresholds_and_boundaries(self):
        for p, outcome in ((.71, "yes"), (.7, "yes"), (.5, "not settled"), (.3, "no"), (.29, "no")):
            self.assertEqual(J.decode(answer(masked=p)["answers"])["masked"],
                             {"probability": p, "outcome": outcome})

    def test_longing_thresholds_and_boundaries(self):
        for p, outcome in ((1, "yes"), (.7, "yes"), (.69, "not settled"),
                           (.31, "not settled"), (.3, "no"), (0, "no")):
            self.assertEqual(J.decode(answer(longing=p)["answers"])["longing"],
                             {"probability": p, "outcome": outcome})

    def test_one_request_per_line_only_own_context_and_headers(self):
        rows = lines(7)
        def http(req, timeout):
            self.assertEqual(req.full_url, J.ENDPOINT)
            self.assertEqual(req.get_method(), "POST")
            self.assertEqual(req.get_header("Authorization"), "Bearer invented-test-key")
            self.assertEqual(req.get_header("Content-type"), "application/json")
            self.assertEqual(timeout, J.TIMEOUT)
            return response()
        with patch.object(J.request, "urlopen", side_effect=http) as http:
            readings = J.read_lines(rows, "Invented title")
        self.assertEqual(http.call_count, len(rows))
        sent = [json.loads(call.args[0].data) for call in http.call_args_list]
        self.assertEqual({p["state"]["line"] for p in sent}, {r["text"] for r in rows})
        for i, r in enumerate(readings):
            payload = r["payload"]
            self.assertEqual(payload["state"], dict(song_title="Invented title", line=rows[i]["text"],
                before=" ".join(x["text"] for x in rows[max(0, i - 2):i]),
                after=" ".join(x["text"] for x in rows[i + 1:i + 3])))
            self.assertEqual(payload["questions"], J.QUESTIONS)
            self.assertEqual(set(payload["questions"]), {"stated_feeling", "masked", "longing"})
            self.assertEqual(payload["model"], "jev-latest")
            self.assertIn(payload, sent)

    def test_failures_are_unavailable_with_count_reason_notes_and_status(self):
        for failure, reason in ((TimeoutError(), "timeout"),
                                (HTTPError(J.ENDPOINT, 429, "secret body", {}, None), "HTTP 429"),
                                (URLError(TimeoutError()), "timeout"),
                                (URLError("secret connection detail"), "connection error"),
                                (io.BytesIO(b"not json"), "malformed or missing answer"),
                                (io.BytesIO(b'{"answers": {}}'), "malformed or missing answer")):
            with self.subTest(reason=reason):
                notes = []
                effect = failure if isinstance(failure, Exception) else None
                @ST.engine
                def run():
                    with patch.object(J.request, "urlopen", side_effect=effect, return_value=failure):
                        return dict(readings=J.read_lines(lines(1), notes=notes))
                result = run()
                reading = result["readings"][0]
                self.assertEqual(reading["status"], "unavailable")
                self.assertNotIn("masked", reading)
                self.assertNotIn("stated_feeling", reading)
                self.assertIn("1/1 lines unavailable: " + reason, notes[1])
                self.assertEqual(result["stage_status"][0]["status"], "failed")
                printed = "\n".join(ST.add_notes(["  NOTE: " + n for n in notes], result["stage_status"]))
                self.assertIn("NOTES: Jev: failed:", printed)
                self.assertNotIn("secret", printed)

    def test_invalid_and_missing_values_never_become_answers(self):
        for value in (None, True, "2", -1, 5, float("nan"), float("inf")):
            for name, field in (("stated_feeling", "score"), ("masked", "noul"), ("longing", "noul")):
                raw = answer()
                raw["answers"][name][field] = value
                with patch.object(J, "_post", return_value=raw):
                    reading = J.read_lines(lines(1))[0]
                self.assertEqual(reading["status"], "unavailable")
                self.assertNotIn("masked", reading)
                json.dumps(reading, allow_nan=False)

    def test_missing_or_malformed_longing_makes_whole_new_reading_unavailable(self):
        for value in (None, {}, {"wrong": True}, {"noul": 1.01}, {"noul": "0.8"}):
            raw = answer()
            if value is None:
                del raw["answers"]["longing"]
            else:
                raw["answers"]["longing"] = value
            notes = []
            with patch.object(J, "_post", return_value=raw) as post:
                reading = J.read_lines(lines(1), notes=notes)[0]
            self.assertEqual(post.call_count, 1)
            self.assertEqual(reading["status"], "unavailable")
            self.assertEqual(reading["reason"], "malformed or missing answer")
            self.assertFalse(set(reading) & {"stated_feeling", "masked", "longing"})
            self.assertIn("1/1 lines unavailable", notes[1])
            self.assertEqual(reading["answer"], raw["answers"])

    def test_longing_alone_adds_one_summary_with_all_occurrences_and_first_example(self):
        H = harmony()
        rows = lines(4)
        rows[2]["text"] = rows[0]["text"]
        with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(longing=.8)):
            con = J.add_contrast(rows, contrasts(rows))
        full = "\n".join(H.block_contrast(dict(con=list(reversed(con)))))
        self.assertIn('Jev hears longing in 4 of 4 lines (first 0:00 “Invented line 0”)', full)
        self.assertEqual(full.count("Jev hears longing"), 1)
        self.assertFalse(slim.timed(slim.parse(full).blocks["contrast"]))
        self.assertTrue(all(J.display_reason(r) == "" for r in con))
        for p in (.1, .3, .5):
            for r in con:
                r["jev"]["longing"] = J.decode(answer(longing=p)["answers"])["longing"]
            self.assertNotIn("longing", "\n".join(H.block_contrast(dict(con=con))))

    def test_longing_summary_skips_unavailable_and_absent_and_limits_quote(self):
        con = contrasts(lines(4))
        for r in con:
            r["jev"] = dict(status="ran", **J.decode(answer(longing=.8)["answers"]))
        del con[0]["jev"]["longing"]
        con[1]["jev"]["status"] = "unavailable"
        con[2]["text"] = "Invented words " * 30
        summary = J.noul_summaries(con, slim.clock)[0]
        self.assertTrue(summary.startswith('Jev hears longing in 2 of 4 lines (first 0:20 “Invented words'))
        self.assertTrue(summary.endswith('…”)'))
        self.assertLessEqual(len(summary), 116)

    def test_runtime_noul_payload_summary_tag_defaults_and_request_count(self):
        H = harmony()
        question = dict(type="noul", instructions="Does `line` name a place?",
                        criteria={"true": "Names a place.", "false": "No place named."})
        for metadata in ({}, {"label": "places", "verb": "finds", "local_note": "never send"}):
            with patch.dict(J.QUESTIONS, {"mentions_a_place": dict(question, **metadata)}):
                for probability in (.8, .1, .5):
                    rows = lines(2)
                    raw = answer(3.82, longing=.8)
                    raw["answers"]["mentions_a_place"] = {"noul": probability}
                    def http(req, timeout):
                        payload = json.loads(req.data)
                        self.assertEqual(payload["questions"]["mentions_a_place"], question)
                        for sent in payload["questions"].values():
                            self.assertEqual(set(sent), {"type", "instructions", "criteria"})
                        return io.BytesIO(json.dumps(raw).encode())
                    with patch.object(J.request, "urlopen", side_effect=http) as requests:
                        con = contrasts(rows)
                        con[0]["words"] = -.24
                        con = J.add_contrast(rows, con)
                    self.assertEqual(requests.call_count, len(rows))
                    self.assertEqual(con[0]["jev"]["answer"], raw["answers"])
                    full = "\n".join(H.block_contrast(dict(con=con)))
                    label, verb = metadata.get("label", "mentions a place"), metadata.get("verb", "hears")
                    summary = f"Jev {verb} {label} in 2 of 2 lines"
                    for rendered in (full, slim.render(full),
                                     slim.render_first_listen({"harmony": full}),
                                     slim.render(analysis={"contrast": con})):
                        if probability == .8:
                            self.assertIn(summary, rendered)
                            self.assertIn(f"{label} (0.80)", rendered)
                            self.assertLess(rendered.index("Jev = "), rendered.index("Jev hears longing"))
                            self.assertLess(rendered.index("Jev hears longing"), rendered.index(summary))
                        else:
                            self.assertNotIn(summary, rendered)
                            self.assertNotIn(label + " (", rendered)
                    self.assertEqual(len(slim.timed(slim.parse(full).blocks["contrast"])), 1)
                    # Extras alone never select a timed row.
                    con[0]["words"] = .1
                    self.assertTrue(all(J.display_reason(r) == "" for r in con))
                    self.assertFalse(slim.timed(slim.parse("\n".join(H.block_contrast(dict(con=con)))).blocks["contrast"]))

    def test_extra_score_is_asked_saved_and_not_rendered(self):
        H = harmony()
        question = dict(type="score", instructions="Rate the weather in `line`.",
                        criteria=["Cold", "Mild", "Hot"], label="weather", verb="finds")
        raw = answer(3.82)
        raw["answers"]["weather_score"] = {"score": 2}
        with patch.dict(J.QUESTIONS, {"weather_score": question}), \
             patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(json.dumps(raw).encode())) as http:
            con = contrasts(lines(1))
            con[0]["words"] = -.24
            con = J.add_contrast(lines(1), con)
            self.assertEqual(http.call_count, 1)
            reading = con[0]["jev"]
            self.assertEqual(reading["weather_score"], 1)
            self.assertEqual(reading["answer"]["weather_score"], {"score": 2})
            self.assertEqual(set(reading["payload"]["questions"]["weather_score"]),
                             {"type", "instructions", "criteria"})
            full = "\n".join(H.block_contrast(dict(con=con)))
            self.assertIn("readings disagree", full)
            self.assertNotIn("weather", full)

    def test_removed_noul_and_new_unanswered_question_are_absent_in_saved_views(self):
        H = harmony()
        import listen
        question = dict(type="noul", instructions="Does `line` name a place?",
                        criteria={"true": "Place.", "false": "No place."}, label="places", verb="finds")
        con = contrasts(lines(1))
        con[0]["words"] = -.24
        con[0]["jev"] = dict(status="ran", **J.decode(answer(3.82, longing=.8)["answers"]))
        con[0]["jev"]["mentions_a_place"] = {"outcome": "yes", "probability": .8}
        with patch.dict(J.QUESTIONS, {"mentions_a_place": question}):
            full = "\n".join(H.block_contrast(dict(con=con)))
        self.assertIn("places (0.80)", full)
        song = json.loads((Path(__file__).parent / "fixtures/first-listen-song.json").read_text())
        song["cards"]["harmony"].update(text=full, json={"contrast": con})
        before = deepcopy(song)
        with patch.dict(J.QUESTIONS, {"new_unanswered": question}), \
             patch.object(J.request, "urlopen", side_effect=AssertionError("render called HTTP")) as http:
            for view in ("full", "harmony", "summary"):
                rendered, _ = listen.render(song, view)
                self.assertNotIn("places", rendered)
                self.assertIn("longing (0.80)", rendered)
            for rendered in (slim.render(full, analysis={"contrast": con}),
                             slim.render_first_listen({"harmony": full}, {"harmony": {"contrast": con}})):
                self.assertNotIn("places", rendered)
            self.assertEqual(song, before)
            http.assert_not_called()

    def test_longing_tag_on_each_existing_reason_and_compact_pick(self):
        H = harmony()
        for score, masked, words, music in ((3.82, .1, -.24, .1), (2, .85, .1, .1),
                                           (2, .1, -.8, .5), (3.82, .85, -.8, .5)):
            con = contrasts(lines(1))
            con[0].update(words=words, music=music)
            with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(score, masked, .77)):
                con = J.add_contrast(lines(1), con)
            full = "\n".join(H.block_contrast(dict(con=con)))
            timed = slim.timed(slim.parse(full).blocks["contrast"])
            self.assertEqual(len(timed), 1)
            self.assertIn("; longing (0.77)", timed[0])
            short = slim.render(full)
            first = slim.render_first_listen({"harmony": full})
            picked = next(r for r in first.splitlines() if r.startswith("CONTRAST:"))
            self.assertIn("; longing (0.77)", picked)
            self.assertIn("Jev hears longing in 1 of 1 lines", first)
            for rendered in (full, short, first):
                self.assertLessEqual(max(map(len, rendered.splitlines())), 118)

    def test_compact_longing_summary_uses_only_spare_budget(self):
        H = harmony()
        from test_slim import FirstListenTests
        cards = FirstListenTests().cards()
        with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(3.82, .1, .8)):
            con = J.add_contrast(lines(1), contrasts(lines(1)))
        con[0]["words"] = -.8
        full = "\n".join(H.block_contrast(dict(con=con)))
        summary = "  " + J.noul_summaries(con, slim.clock)[0] + "\n"
        cards["harmony"] = full
        # Fill the existing headline with real selections, at several lengths.
        summaries = set()
        for count in range(1, 26):
            cards["singer"] = 'SINGER:\n  pitch arc: ' + 'invented ' * (count * 12) + '\n'
            without = slim.render_first_listen(dict(cards, harmony=full.replace(summary, "")))
            with_summary = slim.render_first_listen(cards)
            kept = [r for r in with_summary.splitlines() if not r.startswith("Jev hears longing in ")]
            self.assertEqual(kept, without.splitlines())
            self.assertLessEqual(next(i for i, r in enumerate(with_summary.splitlines())
                                      if r.startswith("LYRICS:")), 25)
            summaries.add("Jev hears longing" in with_summary)
        self.assertEqual(summaries, {False, True})

    def test_multiple_noul_summaries_preserve_order_and_spare_budget(self):
        H = harmony()
        from test_slim import FirstListenTests
        cards = FirstListenTests().cards()
        question = dict(type="noul", instructions="Does `line` name a place?",
                        criteria={"true": "Place.", "false": "No place."}, label="places", verb="finds")
        con = contrasts(lines(1))
        con[0]["words"] = -.8
        con[0]["jev"] = dict(status="ran", **J.decode(answer(3.82, longing=.8)["answers"]))
        con[0]["jev"]["mentions_a_place"] = {"outcome": "yes", "probability": .8}
        with patch.dict(J.QUESTIONS, {"mentions_a_place": question}):
            full = "\n".join(H.block_contrast(dict(con=con)))
            without = "\n".join(r for r in full.splitlines() if not J.is_noul_summary(r.strip()))
            counts = set()
            for count in range(1, 26):
                cards["singer"] = 'SINGER:\n  pitch arc: ' + 'invented ' * (count * 12) + '\n'
                baseline = slim.render_first_listen(dict(cards, harmony=without))
                rendered = slim.render_first_listen(dict(cards, harmony=full))
                rows = rendered.splitlines()
                summaries = [r for r in rows if J.is_noul_summary(r)]
                self.assertEqual([r for r in rows if not J.is_noul_summary(r)], baseline.splitlines())
                self.assertEqual(summaries, J.noul_summaries(con, slim.clock)[:len(summaries)])
                self.assertLessEqual(next(i for i, r in enumerate(rows) if r.startswith("LYRICS:")), 25)
                self.assertLessEqual(max(map(len, rows)), 118)
                counts.add(len(summaries))
            self.assertEqual(counts, {0, 1, 2})

    def test_old_saved_bundle_without_longing_renders_without_calls(self):
        H = harmony()
        import listen
        song = json.loads((Path(__file__).parent / "fixtures/first-listen-song.json").read_text())
        with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(1, .85)):
            con = J.add_contrast(lines(1), contrasts(lines(1)))
        reading = con[0]["jev"]
        del reading["longing"], reading["answer"]["longing"], reading["payload"]["questions"]["longing"]
        card = song["cards"]["harmony"]
        card["json"]["contrast"] = con
        card["text"] = "\n".join(H.block_contrast(dict(con=con)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song.json"
            listen.save(path, song)
            saved = listen.load(path)
            with patch.object(J.request, "urlopen", side_effect=AssertionError("render called HTTP")):
                for view in ("harmony", "summary", "full"):
                    rendered, _ = listen.render(saved, view)
                    self.assertIn("Jev reads masking (0.85)", rendered)
                    self.assertNotIn("longing", rendered)
            self.assertNotIn("longing", saved["cards"]["harmony"]["json"]["contrast"][0]["jev"])

    def test_partial_failure_keeps_other_lines_and_marks_degraded(self):
        def http(req, timeout):
            if json.loads(req.data)["state"]["line"].endswith("1"):
                raise TimeoutError()
            return response(1, .85)
        notes = []
        @ST.engine
        def run():
            with patch.object(J.request, "urlopen", side_effect=http):
                return dict(readings=J.read_lines(lines(3), notes=notes))
        result = run()
        self.assertEqual([r["status"] for r in result["readings"]], ["ran", "unavailable", "ran"])
        self.assertEqual(result["stage_status"][0]["status"], "degraded")
        self.assertEqual(sum(n.startswith("Jev failed:") for n in notes), 1)
        self.assertIn("1/3", notes[1])

    def test_offline_skipped_with_note_and_zero_calls(self):
        notes = []
        @ST.engine
        def run():
            return dict(readings=J.read_lines(lines(2), notes=notes))
        with patch.dict(os.environ, {"RAVEN_EARS_OFFLINE": "1"}), patch.object(J.request, "urlopen") as http:
            result = run()
            http.assert_not_called()
        self.assertTrue(all(r["status"] == "skipped" for r in result["readings"]))
        self.assertEqual(notes, ["Jev skipped: offline mode (RAVEN_EARS_OFFLINE=1)"])
        self.assertEqual(result["stage_status"][0]["status"], "skipped")
        printed = "\n".join(ST.add_notes([], result["stage_status"]))
        self.assertIn("NOTES: Jev: skipped: offline mode", printed)

    def test_privacy_note_and_success_status(self):
        notes = []
        @ST.engine
        def run():
            with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response()):
                return dict(readings=J.read_lines(lines(3), notes=notes))
        result = run()
        self.assertEqual(len(notes), 1)
        self.assertIn("3 lyric lines", notes[0])
        self.assertIn("api.typesafe.ai", notes[0])
        self.assertIn("never audio", notes[0])
        self.assertEqual(result["stage_status"][0]["status"], "ran")

    def test_masked_and_sign_disagreement_show_beside_unchanged_measurements(self):
        H = harmony()
        rows = lines(3)
        con = contrasts(rows)
        con[1]["words"] = .9
        before = deepcopy(con)
        def http(req, timeout):
            i = int(json.loads(req.data)["state"]["line"][-1])
            return response(1 if i == 1 else 2, [.85, .1, .5][i])
        with patch.object(J.request, "urlopen", side_effect=http):
            con = J.add_contrast(rows, con)
        full = "\n".join(H.block_contrast(dict(con=con)))
        self.assertIn("Jev reads masking (0.85)", full)
        self.assertIn("words +0.90 (roberta) · Jev -0.50; readings disagree", full)
        self.assertNotIn("Invented line 2", full)
        self.assertTrue(all(c["flag"] is None for c in con))
        for old, new in zip(before, con):
            self.assertEqual(old, {k: v for k, v in new.items() if k != "jev"})
        self.assertLessEqual(max(map(len, full.splitlines())), 118)
        short = slim.render(full)
        self.assertIn("Jev reads masking (0.85)", short)
        self.assertIn("Jev = TypeSafe", short)
        first = slim.render_first_listen({"harmony": full}, {"harmony": {"contrast": con}})
        self.assertIn("CONTRAST:", first)
        self.assertIn("Jev reads masking (0.85)", first)
        self.assertIn("each line judged in context", first)
        self.assertLessEqual(max(map(len, first.splitlines())), 118)

    def test_not_settled_not_shown_and_disagreement_requires_opposite_sign(self):
        H = harmony()
        with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(4, .5)):
            con = J.add_contrast(lines(1), contrasts(lines(1)))
        full = "\n".join(H.block_contrast(dict(con=con)))
        self.assertNotIn("Invented line", full)
        self.assertEqual(con[0]["jev"]["masked"]["outcome"], "not settled")
        con[0]["words"] = -.6
        con[0]["jev"]["stated_feeling"] = 0
        self.assertEqual(J.display_reason(con[0]), "")
        con[0]["jev"]["stated_feeling"] = .01
        self.assertEqual(J.display_reason(con[0]), "readings disagree")

    def test_jev_available_when_roberta_failed(self):
        H = harmony()
        with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(1, .85)):
            con = J.add_contrast(lines(1), [])
        full = "\n".join(H.block_contrast(dict(con=con)))
        self.assertIn("roberta unavailable · Jev -0.50", full)
        self.assertIn("Jev reads masking", full)
        first = slim.render_first_listen({"harmony": full}, {"harmony": {
            "stage_status": [ST.outcome("sentiment", "failed", "invented failure")]}})
        self.assertIn("CONTRAST:", first)
        self.assertIn("Jev -0.50", first)

    def test_answers_payloads_saved_and_every_view_reuses_without_calls(self):
        H = harmony()
        import listen
        song = json.loads((Path(__file__).parent / "fixtures/first-listen-song.json").read_text())
        notes = []
        with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(1, .85, .8)) as http:
            con = J.add_contrast(lines(2), contrasts(lines(2)), "Invented title", notes)
            self.assertEqual(http.call_count, 2)
        card = song["cards"]["harmony"]
        card["json"].update(contrast=con, notes=notes)
        text = card["text"]
        a, b = text.index("CONTRAST ("), text.index("\nSURPRISE (")
        card["text"] = text[:a] + "\n".join(H.block_contrast(dict(con=con))) + "\n" + text[b:]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", Path(__file__).parent)) as tmp:
            path = Path(tmp) / "songs" / song["manifest"]["file_sha256"] / "song.json"
            listen.save(path, song)
            before = path.read_bytes()
            saved = listen.load(path)
            self.assertEqual(saved["cards"]["harmony"]["json"]["contrast"], con)
            self.assertEqual(con[0]["jev"]["answer"], answer(1, .85, .8)["answers"])
            with patch.object(J.request, "urlopen", side_effect=AssertionError("render called HTTP")), \
                 patch.dict(os.environ, {"TYPESAFE_API_KEY": "still-set", "RAVEN_EARS_OFFLINE": "1"}):
                for view in ("harmony", "summary", "full"):
                    rendered, _ = listen.render(saved, view)
                    self.assertIn("Jev reads masking", rendered)
                    self.assertIn("longing (0.80)", rendered)
            self.assertEqual(path.read_bytes(), before)

    def test_saved_payload_and_raw_answer_keep_path_like_lyric_text_verbatim(self):
        import listen
        song = json.loads((Path(__file__).parent / "fixtures/first-listen-song.json").read_text())
        rows = lines(1)
        rows[0]["text"] = "We walk /invented/lantern and Q:\\invented\\harbour"
        raw = answer(1, .85)
        raw["answers"]["stated_feeling"]["legend"] = "/invented/legend"
        with patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: io.BytesIO(json.dumps(raw).encode())) as http:
            con = J.add_contrast(rows, contrasts(rows), "/invented/title")
        sent = json.loads(http.call_args.args[0].data)
        song["cards"]["harmony"]["json"]["contrast"] = con
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", Path(__file__).parent)) as tmp:
            path = Path(tmp) / "song.json"
            listen.save(path, song)
            saved = listen.load(path)["cards"]["harmony"]["json"]["contrast"][0]["jev"]
            self.assertEqual(saved["payload"], sent)
            self.assertEqual(saved["answer"], raw["answers"])

    def test_existing_flag_keeps_lift_when_jev_unavailable_and_report_has_one_stage(self):
        H = harmony()
        notes = []
        con = contrasts(lines(1))
        con[0].update(words=-.8, music=.5)
        @ST.engine
        def run():
            with patch.object(J.request, "urlopen", side_effect=TimeoutError()):
                return dict(con=J.add_contrast(lines(1), con, notes=notes),
                            path="invented.mp3", dur=20, notes=notes)
        result = run()
        with ExitStack() as stack:
            for name in ("block_sound", "block_chords", "block_tension", "block_resolves", "block_timeline",
                         "block_lyric_timeline", "block_surprise", "block_chills"):
                stack.enter_context(patch.object(H, name, return_value=[]))
            stack.enter_context(patch("ache.block", return_value=[]))
            printed = H.report(result)
        self.assertIn("words -0.80 (roberta) · Jev unavailable; lift +0.5", printed)
        self.assertIn("words heavy (-0.80), music lifting (lift +0.5, energy leads)", printed)
        self.assertEqual(sum(line.startswith("NOTES:") for line in printed.splitlines()), 1)
        self.assertIn("Jev: failed: 1/1 lines unavailable: timeout", printed)
        self.assertNotIn("jev failed:", printed)

    def test_only_fresh_listen_calls_again_not_a_saved_view_or_key_change(self):
        H = harmony()
        import listen
        fixture = json.loads((Path(__file__).parent / "fixtures/first-listen-song.json").read_text())
        def fake_pass(name, args):
            card = deepcopy(fixture["cards"][name])
            if name == "harmony":
                notes = []
                con = J.add_contrast(lines(2), contrasts(lines(2)), "Invented title", notes)
                card["json"].update(contrast=con, notes=notes)
                text = card["text"]
                a, b = text.index("CONTRAST ("), text.index("\nSURPRISE (")
                card["text"] = text[:a] + "\n".join(H.block_contrast(dict(con=con))) + "\n" + text[b:]
            return dict(card, status="ran", reason="")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", Path(__file__).parent)) as tmp:
            root = Path(tmp)
            audio = root / "invented.mp3"
            audio.write_bytes(b"invented bytes, not an audio clip")
            def call(*flags):
                return listen.execute(listen.parser().parse_args([str(audio), "--view", "harmony", *flags]))
            with patch.dict(os.environ, {"RAVEN_EARS_HOME": str(root / "ears"), "RAVEN_EARS_NO_BASELINE": "0"}), \
                 patch.object(listen, "run_pass", side_effect=fake_pass) as engine, \
                 patch.object(listen, "ears_version", return_value="invented-version"), \
                 patch.object(J.request, "urlopen", side_effect=lambda *a, **kw: response(1, .85)) as http:
                first = call()[0]
                self.assertIn("Jev reads masking", first)
                self.assertEqual(http.call_count, 2)
                self.assertEqual(call()[0], first)
                self.assertEqual(engine.call_count, 3)
                with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
                    self.assertEqual(call()[0], first)  # Saved evidence survives removal of the key.
                self.assertEqual(http.call_count, 2)
                call("--fresh")
                self.assertEqual(http.call_count, 4)
                self.assertEqual(engine.call_count, 6)

    def test_old_answer_does_not_pose_as_answer_to_new_wording(self):
        q = dict(J.QUESTIONS["longing"])
        old_sent = J._sent_form(q)
        reading = {"status": "ran", "longing": {"probability": .9, "outcome": "yes"},
                   "payload": {"questions": {"longing": old_sent}}}
        self.assertEqual(J.saved_answer(reading, "longing", q)["outcome"], "yes")
        edited = dict(q, instructions="Does `line` mention the sea?")
        self.assertEqual(J.saved_answer(reading, "longing", edited), {})
        # Display-only keys are never sent, so changing them keeps the answer.
        self.assertEqual(J.saved_answer(reading, "longing", dict(q, label="pining"))["outcome"], "yes")
        # No saved payload (older bundle): trusted as before.
        self.assertEqual(J.saved_answer({"longing": {"outcome": "yes", "probability": .9}}, "longing", edited)["outcome"], "yes")


def run_tests(check):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(JevTests)
    result = unittest.TestResult()
    suite.run(result)
    problems = {str(test): text for test, text in result.failures + result.errors}
    for test in unittest.defaultTestLoader.getTestCaseNames(JevTests):
        key = f"{test} ({__name__}.JevTests.{test})"
        check("Jev: " + test, key not in problems, problems.get(key, ""))


if __name__ == "__main__":
    unittest.main()
