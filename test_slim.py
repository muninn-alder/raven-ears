"""Offline slim-card checks, also called by test_ears_plus.py --fast.

Only invented text is committed. Generated cached cards supply read-only
coverage; launchers/engine output are faked inside a temporary directory.
"""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import slim as S

ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "bin/raven-listen-slim"
CARD = '''# A small song  (2026-11-01 12:00)
=================== music
MUSIC: example.mp3 · 2:00
  KEY: unclear between C major and A minor
  KEY VOTES: spectral D minor (a vote, not the verdict)
  TEMPO: 96 BPM · steady 0.90 · beat strength 2.4x · meter 3/4, see METER
  SHAPE: quiet intro → loud 0:30 → PEAK loud 1:00 → quiet outro 1:30
    0:00–0:30 quiet -24 dB
    0:30–1:00 loud -14 dB
    1:00–1:30 loud -12 dB <- PEAK
    1:30–2:00 quiet -24 dB
  VOCALS: yes, voice present [method: vocal-stem energy] · 0:10–1:40
  VOCAL: breathier 1:30–2:00 [CPPS 8–12]
  LYRICS: supplied lines, some placed (≈)
    [0:10] A sample tray is waiting by the scale
    [0:30≈] Check the sample tray
    [1:00] Check the sample tray (unsure: thin evidence)
    [1:30] Another channel opens for the test
  NOTE: lyrics: one uncertain line: 'I failed to read the scale'
and now, you
=================== singer
SINGER.PY example.mp3 2:00 (stems cached in 0.0s)
SINGER:
  pitch: median C4 · comfortable range G3–G4 (12 semitones)
  most intense moment: 1:00–1:04 · 6 dB louder than the voice's median (speech-trained model; read as texture, not a verdict)
INSTRUMENTS (model estimates):
  overall:
    keys (organ or piano?) likely (score 0.6, 70% of the song) throughout
    bass guitar possibly (score 0.2, 50% of the song) 0:30–1:30
  who is playing ('+ voice' = the vocal stem has level):
    0:30–1:30 keys, bass + voice
CREDITS (local file):
  credited and heard:
    keyboard (Example Player): throughout
  credited, not clearly heard:
    flute (Example Player): credited, not clearly heard
ENTRANCES/EXITS:
  0:10 vocals enter
  0:30 bass guitar enters (possibly) [instrument tagger]
METER: 4 beats per bar; split ambiguous (4/4 / 12/8) · confidence low
and now, you
=================== harmony
HARMONY: example.mp3 · 2:00
CHORDS
  home: C major, working reference only
TENSION
  range 0.1–0.9, peak at 1:00
  BUILD 0:00–0:30 tension climbs 0.1→0.3
  BUILD 0:30–1:00 tension climbs 0.3→0.9 (working key, tonic unclear)
  RELEASE 1:00–1:30 tension eases 0.9→0.2
TIMELINE
  0:10 voice enters: 'A sample tray is waiting by the scale'
CONTRAST (words vs music)
  0:10 'A sample tray': words heavy (-0.4), music lifting (lift +0.3)
  0:30 'Check the sample tray': words heavy (-0.9), music lifting (lift +0.8)
  1:30 'Another channel': words warm (+0.6), music weighing down (lift -0.6)
SURPRISE (where the song breaks its own pattern)
  0:30 novelty peak (+8 dB across it): a section break (expected scaffolding)
  0:40 novelty peak (+1 dB across it): an unscheduled change
  1:10 vi after V (inside the key but off the path); usually V→I
  1:40 refrain check: 'Check the sample tray' repeats the same count
CHILL CANDIDATES (candidates, not verdicts)
  appoggiaturas (from a pyin pitch track of the full mix; approximate; treat as hints):
    0:20 C held 0.3 s over G, then down onto B
    0:50 C held 0.8 s over G, then down onto B
  new or unexpected harmony:
    1:10 deceptive V→vi (working reference, tonic unclear)
  sudden dynamic swells:
    0:30 +10 dB over about 2.5 s
    1:00 +14 dB over about 2.5 s
ACHE 0:50 borrowed iv (working reference, tonic unclear) (lean)
ACHE 1:10 deceptive V→vi (lean)
and now, you
=================== and now, you
  Old invitation.
'''


class SlimTests(unittest.TestCase):
    def test_word_gaps_do_not_bridge_a_measured_second_or_backtrack(self):
        for desc in ('voice present, words not heard (Whisper found none)',
                     'voice present, words unavailable (transcription did not run or failed)'):
            rows = [f'2:05  2:05–2:07: {desc}', f'2:08  2:08–2:11: {desc}']
            self.assertEqual(S._merge_word_gaps(rows), rows)
            self.assertEqual(S._merge_word_gaps(rows[::-1]), rows[::-1])
            rows[1] = f'2:07  2:07–2:11: {desc}'
            self.assertEqual(S._merge_word_gaps(rows), [f'2:05  2:05–2:11: {desc}'])

    def test_feel_merges_three_touching_word_gaps(self):
        description = "voice present, words not heard (Whisper found none)"
        rows = [f"{a}–{b}: {description}" for a, b in
                (("2:32", "2:34"), ("2:34", "2:36"), ("2:36", "2:39"))]
        cards = {"harmony": "HARMONY: invented.wav · 3:00\nTIMELINE\n  " + "\n  ".join(rows)}
        before = copy.deepcopy(cards)
        # Full voice evidence keeps legacy gap cleanup from changing the claims.
        cards["music"] = "MUSIC: invented.wav · 3:00\n  VOCALS: yes · 2:32–2:39\n"
        out = S.render_first_listen(cards)
        self.assertIn(f"WORDS 2:32–2:39: {description}", out)
        self.assertEqual(out.count("WORDS "), 1)
        self.assertEqual(cards["harmony"], before["harmony"])

    def test_word_gap_merge_keeps_barriers_and_lone_rows(self):
        gap = "voice present, words not heard"
        lone = f"2:32–2:34: {gap}"
        lyric = f"2:34–2:36: '{gap}'"
        following = f"2:36–2:39: {gap}"
        for rows in ([lone], [lone, lyric, following],
                     [lone, "2:34–2:36: no voice (instrumental)", following],
                     [lone, f"2:34–2:36: {gap} (Whisper found none)"],
                     [lone, f"2:36–2:39: {gap}"]):
            with self.subTest(rows=rows):
                self.assertEqual(S._merge_word_gaps(rows), rows)

    def test_word_gap_merge_on_the_real_timeline_row_shape(self):
        # Invented times in the shape Muninn's Vampires card printed (lead clock column, 11 Nov 2026).
        d = "voice present, words not heard (Whisper found none)"
        rows = [f"2:32  2:32–2:34: {d}", f"2:34  2:34–2:37: {d}", f"2:37  2:37–2:39: {d}",
                f"2:43  2:43–2:44: {d}", f"2:48  2:48–2:48: {d}"]
        self.assertEqual(S._merge_word_gaps(rows),
                         [f"2:32  2:32–2:39: {d}", f"2:43  2:43–2:44: {d}", f"2:48  2:48: {d}"])
        # A plain event between claims doesn't break the run; a lyric line or another claim does.
        between = [rows[0], rows[1], "2:37  acoustic guitar alone for 0:11", rows[2]]
        self.assertEqual(S._merge_word_gaps(between),
                         [f"2:32  2:32–2:39: {d}", "2:37  acoustic guitar alone for 0:11"])
        for barrier in ("2:37  voice: 'invented line'", "2:37  2:37–2:37: no voice (instrumental)"):
            with self.subTest(barrier=barrier):
                self.assertEqual(len(S._merge_word_gaps([rows[0], rows[1], barrier, rows[2]])), 3)
        lone = f"   2:36  2:36–2:39: {d}"
        self.assertEqual(S._merge_word_gaps([lone]), [lone])

    def test_order_length_and_invitation(self):
        out = S.render(CARD, "/cards/example.txt")
        labels = ["THE SHAPE", "ACHE", "CHILL CANDIDATES", "SURPRISE", "CONTRAST", "VOICE", "WORDS", "INSTRUMENTS", "DIDN'T HEAR", "full card:", S.INVITATION]
        self.assertEqual(sorted(out.index(label) for label in labels), [out.index(label) for label in labels])
        self.assertTrue(25 <= len(out.splitlines()) <= 40)
        self.assertTrue(out.endswith(S.INVITATION + "\n"))
        self.assertIn(f"full card: /cards/example.txt ({len(CARD.splitlines())} lines)", out)
        self.assertEqual(out.count(S.INVITATION), 1)

    def test_one_header_authority_and_qualifiers(self):
        header = S.render(CARD).splitlines()[0]
        self.assertIn("96 BPM · steady 0.90", header)
        self.assertIn("split ambiguous (4/4 / 12/8) · confidence low", header)
        self.assertIn("key unclear between C major and A minor", header)
        self.assertNotIn("3/4", header)
        self.assertNotIn("D minor", header)
        self.assertIn("confidence not on this card", S.header(S.parse("  KEY: C major")))
        arc = S.shape(S.parse("  SHAPE: quiet intro → loud 0:30\n"))
        self.assertIn("quiet intro (time not on this card)", "\n".join(arc))
        self.assertNotIn("0:00", "\n".join(arc))

    def test_voice_and_credits_are_evidence_not_feelings(self):
        out = S.render(CARD)
        self.assertIn("0:10 vocals enter", out)
        self.assertIn("comfortable range G3–G4", out)
        self.assertIn("breathier 1:30–2:00", out)
        self.assertIn("read as texture, not a verdict", out)
        self.assertIn("keys (organ or piano?) likely", out)
        self.assertIn("bass guitar possibly", out)
        self.assertIn("credited and heard: keyboard", out)
        self.assertIn("credited, not clearly heard: flute", out)
        self.assertNotIn("Example Player", out)
        self.assertIn("bass guitar enters (possibly)", out)

    def test_selection_is_explicit_and_deterministic(self):
        out = S.render(CARD)
        self.assertIn("BUILD 0:30–1:00", out)
        self.assertNotIn("BUILD 0:00–0:30", out)
        chills = out.split("CHILL CANDIDATES", 1)[1].split("SURPRISE", 1)[0]
        self.assertIn("0:50 C held 0.8 s", chills)
        self.assertIn("approximate full-mix pitch hint, voice unclear", chills)
        self.assertIn("1:00 +14 dB", chills)
        surprises = out.split("SURPRISE", 1)[1].split("CONTRAST", 1)[0]
        self.assertIn("1:10 vi after V", surprises)
        self.assertIn("unscheduled", surprises)
        self.assertNotIn("expected scaffolding", surprises)
        self.assertNotIn("refrain check", surprises)
        contrast = out.split("CONTRAST", 1)[1].split("VOICE", 1)[0]
        self.assertIn("0:30≈ 'Check", contrast)
        self.assertIn("1:30 'Another", contrast)
        self.assertNotIn("0:10 'A paper", contrast)
        self.assertEqual(out, S.render(CARD))

    def test_saved_chill_list_uses_the_printed_key_signature(self):
        text = ("HARMONY: Lantern · 1:00\nCHORDS\n  home: G major (1♯) · tonic clear\n"
                "CHILL CANDIDATES (candidates, not verdicts)\n"
                "  appoggiaturas (vocal stem; treat as hints):\n"
                "    0:50  G held 0.8 s over Dmaj7, then down onto G♭\n")
        self.assertIn("G held 0.8 s over Dmaj7, then down onto F♯", S.render(text))
        self.assertIn("G held 0.8 s over Dmaj7, then down onto G♭",
                      S.render(text.replace("G major (1♯) · tonic clear", "key unclear")))

    def test_hook_timing_and_lyric_uncertainty(self):
        out = S.render(CARD)
        self.assertIn("hook lean:", out)
        self.assertIn("2 exact line repeats on this card; some copies unsure; 0:30≈, 1:00", out)
        self.assertIn("first: [0:10]", out)
        self.assertIn("near tension peak 1:00: [1:00]", out)
        self.assertIn("(unsure: thin evidence)", out)
        self.assertIn("overlap unclear", out)
        self.assertIn("0:30≈ 'Check", out.split("CONTRAST", 1)[1].split("VOICE", 1)[0])
        self.assertNotIn("I failed to find", out)

    def test_no_inferred_lyric_overlap_across_instrumental_gap(self):
        card = "MUSIC: example.mp3 · 2:00\n  LYRICS: rough\n    [0:10] A short line\nTENSION\n  range 0.1–0.9, peak at 1:00\n"
        out = S.render(card)
        self.assertIn("at tension peak 1:00: no timed line on this card", out)
        self.assertNotIn("hook lean", out)

    def test_lyric_timeline_and_engine_invitations(self):
        card = CARD + "ignored after the final listener block\n"
        card = card.replace("TENSION\n", 'LYRIC TIMELINE\n  LYRICS: supplied text (≈ placed)\n  [0:30≈] “Check the sample tray” — chord IV, tension builds\nTENSION\n')
        parsed = S.parse(card)
        self.assertEqual(len(parsed.lyrics), 1)
        self.assertTrue(parsed.lyrics[0].approx)
        self.assertEqual(parsed.lyrics[0].text, "Check the sample tray")
        self.assertIn("most intense moment:", S.render(card))

    def test_duplicates_do_not_inflate_hook_counts(self):
        card = CARD.replace("    [0:30≈] Check the sample tray", "    [0:30≈] Check the sample tray\n    [0:30≈] Check the sample tray")
        self.assertIn("2 exact line repeats", S.render(card))

    def test_missing_blocks_instrumental_and_unreported_failures(self):
        for raw in ("", "# An empty card", "SINGER: instrumental (vocal stem energy low)\n"):
            out = S.render(raw)
            self.assertIn("ACHE: not on this card", out)
            self.assertIn("stage status not recorded (older card)", out)
            self.assertNotIn("ACHE none found", out)
            self.assertNotIn("silence", out)
            self.assertTrue(out.endswith(S.INVITATION + "\n"))
        self.assertIn("instrumental (vocal stem energy low)", S.render("SINGER: instrumental (vocal stem energy low)"))

    def test_reported_failures_never_become_none_or_silence(self):
        raw = """# Partial
INSTRUMENTS: failed (RuntimeError: backend down)
METER: failed (tracker unavailable)
  NOTE: lyrics: skipped by request
ACHE unclear: analysis failed (ValueError: bad shape)
  ACHE NOTE: pitch analysis unavailable
Traceback (most recent call last):
RuntimeError: singer backend down
"""
        out = S.render(raw)
        status = out.split("DIDN'T HEAR:", 1)[1]
        for phrase in ("INSTRUMENTS: failed", "METER: failed", "lyrics: skipped", "pitch analysis unavailable", "singer backend down"):
            self.assertIn(phrase, status)
        self.assertIn("ACHE unclear: analysis failed", out)
        self.assertNotIn("ACHE none found", out)
        self.assertNotIn("silence", out)
        done = S.render("ACHE none found\nand now, you\n")
        self.assertIn("ACHE none found", done)
        for block in ("SINGER", "SURPRISE", "CONTRAST", "CHILL CANDIDATES"):
            out = S.render(block + ": failed (backend unavailable)\n")
            self.assertIn(block + ": failed", out.split("DIDN'T HEAR:")[1])
            self.assertNotIn("no clear candidates reported", out)
        unavailable = S.render("INSTRUMENTS (model estimates):\n  unavailable: no result\n")
        self.assertIn("INSTRUMENTS: unavailable: no result", unavailable.split("DIDN'T HEAR:")[1])

    def test_harmony_only_uses_its_home_verdict(self):
        out = S.render("HARMONY: example.mp3 · 1:00\nCHORDS\n  home: unclear between C major and A minor. Numerals below use C major; ~90 BPM\n")
        self.assertIn("key unclear between C major and A minor", out.splitlines()[0])
        self.assertNotIn("Numerals", out.splitlines()[0])
        self.assertIn("harmonic references below are working only", out.splitlines()[0])

    def test_existing_analysis_ache_and_text_authority(self):
        data = json.loads((ROOT / "fixtures/ache-analysis.json").read_text())
        before = copy.deepcopy(data)
        data.update(file="example.mp3", duration=40)
        out = S.render("  KEY: C major (sure)\n", analysis=data)
        self.assertIn("key C major (sure)", out.splitlines()[0])
        self.assertIn("ACHE 0:10", out)
        self.assertIn("under ≈", out)
        self.assertIn("first: [0:10≈]", out)
        for k, v in before.items():
            self.assertEqual(data[k], v)
        self.assertIn("ACHE 0:50", S.render(CARD, analysis=data))
        self.assertNotIn("ACHE 0:10", S.render(CARD, analysis=data))
        stale = S.render("  KEY: D major (lean)\n", analysis=data)
        self.assertIn("key D major (lean)", stale.splitlines()[0])
        self.assertIn("supplemental key disagrees", stale)
        self.assertNotIn("borrowed iv", stale)

    def test_analysis_explicit_intervals_peak_and_builds(self):
        data = dict(file="example.mp3", duration=90, tempo={"bpm": 90, "regularity": .9},
                    key_verdict={"working": "C major", "key": None, "state": "tonic"},
                    sections=[dict(start=0, end=90, level="loud", peak=True)],
                    lyrics_aligned={"lines": [dict(t=9, end=13, text="Hold this bracket", approx=True)]},
                    tension={"step": 2, "values": [0.1] * 5 + [0.9, 0.1]},
                    builds_releases=[dict(kind="build", t0=0, t1=10, **{"from": .1, "to": .9})],
                    ache={"status": "ok", "moments": [], "notes": []})
        before = copy.deepcopy(data)
        out = S.render(analysis=data)
        self.assertIn("key C major (unclear)", out.splitlines()[0])
        self.assertIn("BUILD 0:00–0:10", out)
        self.assertIn("at tension peak 0:10: [0:09≈]", out)
        self.assertIn("ACHE none found", out)
        self.assertEqual(data, before)
        data["lyrics_aligned"]["lines"][0]["unsure"] = True
        self.assertIn("unsure: supplied analysis", S.render(analysis=data))

    def test_malformed_analysis_is_visible(self):
        for data in ([], {"ache": {"moments": None}}, {"sections": [{"start": 1}]}):
            out = S.render("# A partial card\n", analysis=data)
            self.assertIn("failed", out.split("DIDN'T HEAR:")[1])
            self.assertTrue(out.endswith(S.INVITATION + "\n"))

    def test_cli_stdin_missing_input_and_broken_optional_json(self):
        for raw in (CARD, CARD.replace("\n", "\r\n"), CARD.rstrip("\n")):
            proc = subprocess.run([sys.executable, str(ROOT / "slim.py"), "-", "--full-card", "/cards/example.txt"], input=raw.encode(), capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(f"({len(raw.splitlines())} lines)", proc.stdout.decode())
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = S.main(["/no/such/slim-source.txt"])
        self.assertEqual(code, 1)
        self.assertTrue(buf.getvalue().endswith(S.INVITATION + "\n"))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "card.txt"
            p.write_text(CARD)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = S.main([str(p), "--analysis", str(Path(directory) / "missing.json")])
            self.assertEqual(code, 0)
            self.assertIn("supplemental analysis failed", buf.getvalue())
            self.assertIn("THE SHAPE", buf.getvalue())

    def test_wrapper_saves_full_bytes_and_passes_options(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            raw = CARD.replace("\n", "\r\n").encode() + b"no final newline"
            source = d / "source.txt"
            saved = d / "saved full card.txt"
            script = d / "fake-listener"
            source.write_bytes(raw)
            script.write_text("#!" + sys.executable + "\n" + '''import os, pathlib, sys
d = pathlib.Path(__file__).parent
raw = (d / 'source.txt').read_bytes()
(d / 'saved full card.txt').write_bytes(raw)
(d / 'args.json').write_text(__import__('json').dumps(sys.argv[1:]))
sys.stdout.buffer.write(raw)
print('(card saved: ' + str(d / 'saved full card.txt') + ')', file=sys.stderr)
''')
            script.chmod(0o755)
            env = dict(os.environ, RAVEN_LISTEN=str(script))
            args = [sys.executable, str(WRAPPER), "neutral song", "--only", "singer", "--credits", "credits file.json"]
            proc = subprocess.run(args, env=env, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(saved.read_bytes(), raw)
            self.assertEqual(json.loads((d / "args.json").read_text()), args[2:])
            self.assertIn(f"full card: {saved}", proc.stdout.decode())
            self.assertTrue(proc.stdout.decode().endswith(S.INVITATION + "\n"))
            full = subprocess.run(args + ["--full"], env=env, capture_output=True)
            self.assertEqual(full.stdout, raw)
            stat = saved.stat()
            reopened = subprocess.run([sys.executable, str(WRAPPER), str(saved)], env=env, capture_output=True)
            self.assertEqual(reopened.returncode, 0, reopened.stderr)
            self.assertIn("THE SHAPE", reopened.stdout.decode())
            self.assertEqual(saved.read_bytes(), raw)
            self.assertEqual(saved.stat().st_mtime_ns, stat.st_mtime_ns)
            full = subprocess.run([sys.executable, str(WRAPPER), str(saved), "--full"], capture_output=True)
            self.assertEqual(full.stdout, raw)
            script.write_text("#!" + sys.executable + "\nimport sys\nprint('backend failed', file=sys.stderr)\nsys.exit(7)\n")
            failed = subprocess.run(args, env=env, capture_output=True)
            self.assertEqual(failed.returncode, 7)
            self.assertIn("no completed card", failed.stdout.decode())
            self.assertTrue(failed.stdout.decode().endswith(S.INVITATION + "\n"))

    def test_generated_cached_card_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neutral.txt"
            path.write_text(CARD, encoding="utf-8")
            raw = path.read_bytes()
            digest, stat = hashlib.sha256(raw).digest(), path.stat()
            out = S.render(raw.decode("utf-8"), path)
            self.assertTrue(25 <= len(out.splitlines()) <= 40)
            self.assertIn(f"full card: {path}", out)
            self.assertTrue(out.endswith(S.INVITATION + "\n"))
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)
            self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)


class FirstListenTests(unittest.TestCase):
    def cards(self):
        # The existing invented fixture has three separately headed passes.
        cards = dict(chunk.split('\n', 1) for chunk in CARD.split('=================== ')[1:])
        cards['harmony'] = cards['harmony'].replace('TENSION\n',
            'TENSION\n  curve (0–1, per 2 s; harmony-from-home 35%, chroma dissonance 20%, loudness 30%, activity 15%):\n')
        cards['harmony'] = cards['harmony'].replace('CONTRAST (words vs music)\n',
            "CONTRAST (words vs music)\n  words = local sentiment model (cardiffnlp roberta, line blended with its neighbours, -1..+1); music 'lift' = song-relative z of energy, activity, brightness, harmonic rhythm, minus half of tension. Mode (major/minor) is NOT used; it's a fact, not a mood.\n")
        return cards

    def bounded(self, out):
        lines = out.splitlines()
        first_lyrics = next(i for i, line in enumerate(lines) if line.startswith('LYRICS:'))
        self.assertLessEqual(first_lyrics, 25)
        self.assertTrue(all(len(line) <= 118 for line in lines))
        self.assertEqual(lines[-1], S.INVITATION)

    def test_stops_confident_counts_by_kind_and_first_two_only(self):
        raw = '''STOPS: 4 · band cuts, voice holds: 2 · full silence: 1 · band break: 1
  uncertain candidates: 1 (full list in JSON)
  0:03.00 (2 s) summed band falls 14 dB, uncertain
  0:10.25 (1.1 s ≈2 beats) band cuts 20 dB, voice holds
  0:25.50 (2.2 s) band cuts 30 dB, full silence
  0:40.00 (3 s) band cuts 25 dB, band break
  0:55.00 (4 s) band cuts 20 dB, voice holds
'''
        cards = self.cards()
        cards['singer'] += raw
        before = copy.deepcopy(cards)
        with mock.patch.dict(sys.modules, {'stops': None, 'singer': None, 'torch': None}):
            out = S.render_first_listen(cards)
        flat = ' '.join(out.split())
        self.assertIn('STOPS: 4 · band cuts, voice holds: 2 · full silence: 1 · band break: 1 · first '
                      '0:10.25 (1.1 s), 0:25.50 (2.2 s)', flat)
        for stamp in ('0:03.00', '0:40.00', '0:55.00'):
            self.assertNotIn(stamp, out)
        lines = out.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('STOPS:'))
        self.assertLessEqual(next(i for i in range(start + 1, len(lines)) if not lines[i].startswith('  ')) - start, 2)
        self.assertEqual(cards, before)
        self.bounded(out)

    def test_stops_json_fallback_and_text_wins(self):
        result = {'status': 'ran', 'events': [
            {'start_s': 3, 'length_s': 2, 'kind': 'uncertain'},
            {'start_s': 115.5, 'length_s': 1.05, 'kind': 'band cuts, voice holds'},
            {'start_s': 164.85, 'length_s': 5.24, 'kind': 'band cuts, voice holds'}]}
        analyses = {'singer': {'stops': result}}
        before = copy.deepcopy(analyses)
        out = S.render_first_listen({}, analyses)
        self.assertIn('STOPS: 2 · band cuts, voice holds: 2 · first 1:55.50 (1.05 s), 2:44.85 (5.24 s)', out)
        self.assertNotIn('0:03', out)
        self.assertEqual(analyses, before)
        self.bounded(out)
        out = S.render_first_listen({'singer': 'STOPS: none found (saved search limits)\n'}, analyses)
        self.assertIn('STOPS: none confident\n', out)
        self.assertNotIn('1:55', out)

    def test_stops_only_uncertain_and_ran_with_none(self):
        for events in ([], [{'start_s': 3, 'length_s': 2, 'kind': 'uncertain'}]):
            out = S.render_first_listen({}, {'singer': {'stops': {'status': 'ran', 'events': events}}})
            self.assertIn('STOPS: none confident\n', out)
            self.assertNotIn('0:03', out)
            self.bounded(out)
        for raw in ('STOPS: none found (0.3 to 8 s; accompaniment vs its own level)\n',
                    'STOPS: 0 confident · uncertain: 3 candidates not called (see --view detail / JSON)\n'
                    '  strongest band falls: 3:07.65 (9.7 dB, uncertain); 3:12.67 (9 dB, uncertain)\n',
                    'STOPS: 0 · band cuts, voice holds: 0 · full silence: 0 · band break: 0\n'
                    '  0:03.00 (2 s) summed band falls 14 dB, uncertain\n'):
            out = S.render_first_listen({'singer': raw})
            self.assertIn('STOPS: none confident\n', out)
            self.assertNotIn('0:03', out)
            self.assertNotIn('3:07.65', out)
            self.assertNotIn('3:12.67', out)
            self.bounded(out)

    def test_faded_stops_survive_text_and_json_summaries(self):
        event = dict(start_s=213.78, length_s=2.64, kind="faded stop")
        raw = ("STOPS: 1 · band cuts, voice holds: 0 · full silence: 0 · band break: 0 · faded stop: 1\n"
               "  3:33.78 (2.64 s) total stem energy falls 17.6 dB, faded stop\n")
        for cards, data in (({'singer': raw}, {}),
                            ({}, {'singer': {'stops': {'status': 'ran', 'events': [event]}}})):
            out = S.render_first_listen(cards, data)
            self.assertIn('STOPS: 1 · faded stop: 1 · first 3:33.78 (2.64 s)', out)
            self.assertNotIn('voice holds: 1', out)
            self.bounded(out)

    def test_repeated_cuts_and_entry_survive_text_and_json_summaries(self):
        event = dict(start_s=9.03, length_s=1.95, kind="band cuts, vocal stem enters")
        run = dict(count=7, interval_s=3.37, start_s=9.03, end_s=31.04)
        data = {'singer': {'stops': dict(status='ran', events=[event], recurrence=[run])}}
        raw = ('STOPS: 1 · band cuts, voice holds: 0 · full silence: 0 · band break: 0 · band cuts, vocal stem enters: 1\n'
               '  0:09.03 (1.95 s) band cuts 38.4 dB, vocal stem enters\n'
               '  repeated cuts: 7 cuts every ~3.37 s, 0:09.03–0:31.04\n'
               '    pulse not checked; cut spacing alone does not establish stop-time\n'
               '  vocal entry measured from stem energy; singing/pitch not checked\n')
        for cards in ({}, {'singer': raw}):
            out = S.render_first_listen(cards, data)
            self.assertIn('band cuts, vocal stem enters: 1', out)
            self.assertIn('first 0:09.03 (1.95 s)', out)
            self.assertIn('repeated cuts: 7 cuts every ~3.37 s', out)
            self.assertIn('pulse not checked', out)
            self.assertNotIn('stop-time:', out)
        out = S.render_first_listen({'singer': raw}, data)
        self.assertIn('singing/pitch not checked', out)

    def test_stops_absent_old_not_run_or_malformed_is_not_none(self):
        cases = [{}, {'stage_status': [{'stage': 'stops', 'status': 'not run'}]}]
        cases += [{'stops': value} for value in (None, [], {}, {'status': 'ran'},
                  {'status': 'not run', 'events': []}, {'status': 'ran', 'events': None},
                  {'status': 'ran', 'events': [None]}, {'status': 'ran', 'events': [{'kind': 'unknown'}]})]
        for data in cases:
            with self.subTest(data=data):
                out = S.render_first_listen(self.cards(), {'singer': data})
                self.assertNotIn('STOPS:', out)
                self.bounded(out)

    def test_stops_failed_or_skipped_stage_cannot_supply_headline(self):
        raw = ('STOPS: 1 · band cuts, voice holds: 1 · full silence: 0 · band break: 0\n'
               '  0:10.25 (1.1 s) band cuts 20 dB, voice holds\n')
        for status in ('failed', 'skipped', 'not run'):
            for stage in ('stops', 'stems'):
                for structured in (False, True):
                    cards = {'singer': raw}
                    analyses = {}
                    if structured:
                        analyses = {'singer': {'stage_status': [
                            {'stage': stage, 'status': status, 'reason': 'invented reason', 'kind': 'audio'}]}}
                    else:
                        cards['singer'] += f'NOTES: {stage}: {status}: invented reason\n'
                    out = S.render_first_listen(cards, analyses)
                    self.assertNotIn('STOPS:', out)
                    self.assertIn('invented reason', out)
                    self.bounded(out)
        out = S.render_first_listen({'singer': 'STOPS: failed: invented reason\n'})
        self.assertNotIn('none confident', out)
        self.assertIn('STOPS: failed: invented reason', out)
        self.assertGreater(out.index('STOPS: failed:'), out.index('LYRICS:'))
        out = S.render_first_listen({}, {'singer': {'stops': {'status': 'failed', 'reason': 'invented reason', 'events': []}}})
        self.assertIn('STOPS: failed: invented reason', out)
        self.assertNotIn('none confident', out)
        out = S.render_first_listen({'singer': raw}, pass_statuses={'singer': {'status': 'failed', 'reason': 'invented reason'}})
        self.assertNotIn('STOPS:', out)
        self.bounded(out)

    def test_all_passes_source_selection_full_words_and_purity(self):
        cards = self.cards()
        before = copy.deepcopy(cards)
        with mock.patch('socket.create_connection', side_effect=AssertionError('network forbidden')), \
             mock.patch('subprocess.run', side_effect=AssertionError('engines forbidden')), \
             mock.patch.dict(sys.modules, {'music': None, 'singer': None, 'harmony': None, 'torch': None}):
            out = S.render_first_listen(cards)
        self.bounded(out)
        self.assertEqual(cards, before)
        self.assertIn('PASSES: music card · singer card · harmony card', out)
        self.assertIn('comfortable range G3–G4', out)
        self.assertIn('speech-trained model; read as texture, not a verdict', ' '.join(out.split()))
        # The 0:30 lyric is placed (≈) on the music card; CONTRAST keeps that caveat.
        self.assertIn("0:30≈ 'Check the sample tray'", out)
        self.assertNotIn("0:10 'A sample tray': words heavy", out)
        self.assertIn('RELEASE 1:00–1:30 tension eases 0.9→0.2', out)
        self.assertIn('KEY: unclear between C major and A minor', out)
        for row in S.parse(cards['music']).lyrics:
            self.assertIn(f'[{S.clock(row.t)}{"≈" if row.approx else ""}] {row.text}', out)
        self.assertEqual(out.count('[0:30≈] Check the sample tray'), 1)

    def test_tempo_survives_a_pointer_to_the_singers_meter(self):
        # 10 Nov 2026: Give A Little Bit's feel card had no TEMPO at all, because the line said "see METER"
        # and METER lives on the singer card. Keep the tempo, drop the pointer to a block not on this card.
        out = S.render_first_listen(self.cards())
        self.assertIn('TEMPO: 96 BPM', out)
        self.assertNotIn('see METER', out)
        out = S.render_first_listen({'music': self.cards()['music']})
        self.assertIn('TEMPO: 96 BPM', out)

    def test_only_quick_card_missing_passes_are_explicit(self):
        out = S.render_first_listen({'music': self.cards()['music']})
        self.bounded(out)
        for name in ('singer', 'harmony'):
            self.assertIn(name + ': not supplied (not run or no saved card; unknown)', out)
        self.assertNotIn('most intense moment', out)
        self.assertNotIn('all stages ran', out)
        self.assertIn('Another channel opens for the test', out)

    def test_words_keep_duplicate_rows_order_and_timestamp_precision(self):
        raw = 'MUSIC: example.wav · 0:30\nLYRICS: invented words\n  [0:02.125≈] Repeat this invented word\n  [0:02.125≈] Repeat this invented word\n'
        out = S.render_first_listen({'music': raw})
        self.assertEqual(out.count('[0:02.125≈] Repeat this invented word'), 2)
        self.bounded(out)

    def test_skipped_lookup_keeps_words_but_failed_words_do_not_supply_lyrics(self):
        import stage_status as ST
        lyric = 'Invented calibration words'
        text = "LYRICS: Whisper's guesses\n  [0:03] " + lyric + '\n'
        note = "NOTES: words: Whisper's guesses (LRCLIB: skipped: lookup off)\n"
        for name, raw, analysis in (
                ('music', text + note, {}),
                ('music', note, {'lyrics': [{'t': 3, 'text': lyric}]}),
                ('singer', note, {'lyrics': {'rows': [{'t': 3, 'text': lyric}]}})):
            for state in ('ran', 'failed', 'skipped'):
                with self.subTest(source=name, text_rows=bool(raw == text + note), state=state):
                    data = dict(analysis, stage_status=[ST.outcome('words', state, kind='input'),
                                                       ST.outcome('LRCLIB', 'skipped', 'lookup off', kind='input')])
                    out = S.render_first_listen({name: raw}, {name: data},
                                               {name: {'status': 'ran', 'reason': ''}})
                    self.assertNotIn('skipped reported', out)
                    self.assertEqual('[0:03] ' + lyric in out, state == 'ran')
                    self.bounded(out)
            # Text-only legacy cards have the same nested input status.
            if raw == text + note:
                self.assertIn('[0:03] ' + lyric, S.render_first_listen({name: raw}))

    def test_misheard_line_only_with_the_writers_lyrics(self):
        raw = 'MUSIC: example.wav · 0:30\nLYRICS: invented words\n  [0:03] Pole pole my love\n'
        mis = {'heard': 'paul paul my love', 'wrote': 'Pole pole my love', 'heard_core': 'paul paul', 'wrote_core': 'Pole pole'}
        far = {'heard_core': 'a much longer wrong stretch', 'wrote_core': 'something else entirely', 'heard': 'x', 'wrote': 'y'}
        rows = [{'t': 3.0, 'text': 'Pole pole my love', 'approx': False, 'misheard': mis},
                {'t': 9.0, 'text': 'Another line', 'approx': False, 'misheard': far},
                {'t': 14.0, 'text': 'Third line', 'approx': False, 'misheard': None}]
        ran = {'music': {'status': 'ran', 'reason': ''}}
        out = S.render_first_listen({'music': raw}, {'music': {'lyrics': rows, 'lyrics_source': {'status': 'file'}}}, ran)
        # the example is the line whose words differ most, not the shortest
        self.assertIn('MISHEARD: 2 lines · e.g. 0:09 heard "a much longer wrong stretch" for "something else entirely"', out)
        self.assertLess(out.index('MISHEARD:'), out.index('LYRICS:'))
        self.bounded(out)
        # a database's lyrics are not the writer's words: no line
        out = S.render_first_listen({'music': raw}, {'music': {'lyrics': rows, 'lyrics_source': {'status': 'synced'}}}, ran)
        self.assertNotIn('MISHEARD', out)
        # supplied lyrics, nothing differs: no line
        clean = [dict(r, misheard=None) for r in rows]
        out = S.render_first_listen({'music': raw}, {'music': {'lyrics': clean, 'lyrics_source': {'status': 'file'}}}, ran)
        self.assertNotIn('MISHEARD', out)
        # an older saved card has no misheard field: one honest line, not silence
        old = [{k: v for k, v in r.items() if k != 'misheard'} for r in rows]
        out = S.render_first_listen({'music': raw}, {'music': {'lyrics': old, 'lyrics_source': {'status': 'file'}}}, ran)
        self.assertIn('MISHEARD: not in this saved card (older listen); --fresh to compute it', out)
        # harmony JSON and singer JSON carry the same rows
        out = S.render_first_listen({'harmony': 'HARMONY: example.wav · 0:30\n'}, {'harmony': {
            'lyrics_aligned': {'lines': rows}, 'lyrics_source': {'status': 'file'}}}, {'harmony': {'status': 'ran', 'reason': ''}})
        self.assertIn('MISHEARD: 2 lines', out)

    def test_failed_pass_keeps_quick_words_and_names_failure(self):
        cards = {'music': self.cards()['music'], 'singer': 'SINGER: unavailable: invented backend failed\n  pitch: comfortable range D3–D4\n'}
        out = S.render_first_listen(cards)
        self.bounded(out)
        self.assertNotIn('reported; details', out)
        self.assertIn('invented backend failed', out)
        self.assertNotIn('comfortable range', out)
        self.assertIn('[0:10] A sample tray is waiting by the scale', out)
        out = S.render_first_listen(cards, {'singer': []})
        self.assertIn('supplemental analysis failed (expected a JSON object)', out)
        self.assertIn('[0:10] A sample tray is waiting by the scale', out)

    def test_long_lines_omit_whole_selection_and_keep_notes_and_lyrics(self):
        cards = self.cards()
        cards['singer'] += '\nSINGER:\n  pitch arc: ' + 'up ' * 1500 + '(working estimate)\n'
        note = 'NOTE: invented diagnostic ' + 'detail ' * 200 + 'END_OF_NOTE'
        lyric = 'A very long invented line ' + 'word ' * 200 + 'END_OF_LYRIC'
        cards['music'] += '\n  ' + note + '\nLYRICS: invented words\n  [1:50] ' + lyric + '\n'
        out = S.render_first_listen(cards)
        self.bounded(out)
        self.assertIn('selections exceed the 25-line budget', out)
        self.assertNotIn('pitch arc:', out)
        self.assertIn(' '.join(lyric.split()), ' '.join(out.split()))
        self.assertIn(' '.join(note.removeprefix('NOTE: ').split()), ' '.join(out.split()))
        cards['music'] = cards['music'].replace('example.mp3', 'long' * 500)
        self.bounded(S.render_first_listen(cards))

    def test_json_supplement_and_original_status_details(self):
        import stage_status as ST
        analyses = {'singer': {'singer': {'lines': ['pitch: comfortable range D3–D4 (12 semitones)']},
                               'stage_status': [ST.outcome('meter', 'failed', 'invented decoding error')]},
                    'music': {'lyrics_align': "LYRICS: writer's text, one placed line",
                              'lyrics': [{'t': 3, 'text': 'Invented calibration words', 'approx': True}]}}
        before = copy.deepcopy(analyses)
        out = S.render_first_listen({}, analyses)
        self.bounded(out)
        self.assertEqual(analyses, before)
        self.assertIn('music JSON · singer JSON', out)
        self.assertIn('comfortable range D3–D4', out)
        self.assertNotIn('singer: failed reported', out)
        self.assertIn('invented decoding error', out)
        self.assertIn('[0:03≈] Invented calibration words', out)
        old = 'MUSIC: example.wav · 0:30\nNOTES: a recorded issue · +2 more\n'
        self.assertIn('2 more status entries omitted by source', S.render_first_listen({'music': old}))
        out = S.render_first_listen({'music': old}, {'music': {'stage_status': [ST.outcome('probe', 'failed', 'reason')]}})
        self.assertIn('probe: failed: reason', out)
        # These rows do not recover the old card's two omitted diagnostics.
        self.assertIn('2 more status entries omitted by source', out)

    def test_placed_contrast_and_end_chord_keep_source_qualifiers(self):
        import harmony
        cards = self.cards()
        # Use the real formatter: its uncertainty is in home, never in ends-on.
        result = {"evs": [{"root": 9, "q": "min", "t0": 0, "t1": 60}],
                  "spans": [], "secs": [], "dur": 60, "glob": (0, "major"),
                  "kinfo": {}, "key_verdict": {"key": None, "working": "C major",
                      "line": "unclear between C major and A minor"},
                  "bpm": 80, "moves": [], "pedals": []}
        with mock.patch.object(harmony, 'label', return_value='vi'):
            chords = '\n'.join(harmony.block_chords(result))
        self.assertIn('working reference only', chords)
        self.assertIn('ends on vi (Am); opens on vi', chords)
        cards['harmony'] = chords + "\nTENSION" + cards['harmony'].split('TENSION', 1)[1]
        cards['harmony'] += '\nLYRICS: supplied text\n  [0:30≈] Check the sample tray (unsure: thin evidence)\n'
        out = S.render_first_listen(cards)
        normalized = ' '.join(out.split())
        self.assertIn('ends on vi (Am); opens on vi', normalized)
        self.assertIn('HARMONY (in C major, key unclear): ends on vi (Am); opens on vi', normalized)
        self.assertNotIn('Numerals below', normalized)
        self.assertIn('0:30≈', out)
        self.assertIn('(lyric unsure on this card)', out)
        self.bounded(out)

    def test_harmony_without_reference_or_score_definitions_is_omitted(self):
        card = ("CHORDS\n  ends on vi (Am); opens on vi\nTENSION\n"
                "  RELEASE 0:10–0:20 tension eases 0.9→0.1\nCONTRAST (words vs music)\n"
                "  0:10 'Invented': words heavy (-0.9), music lifting (lift +0.8)\n")
        out = S.render_first_listen({'harmony': card})
        for claim in ('ends on', 'tension eases', 'words heavy'):
            self.assertNotIn(claim, out)
        self.bounded(out)

    def test_adverse_stage_blocks_plausible_partial_headline(self):
        for evidence in ({'stage_status': [{'stage': 'key', 'status': 'failed', 'reason': 'invented failure', 'kind': 'audio'}]}, {}):
            raw = 'KEY: C major\n'
            if not evidence:
                raw += 'NOTES: key: skipped: invented skip\n'
            out = S.render_first_listen({'music': raw}, {'music': evidence})
            self.assertNotIn('KEY: C major', out)
            self.assertIn('invented ' + ('failure' if evidence else 'skip'), out)
        self.bounded(out)

    def test_failed_pitch_stage_does_not_headline(self):
        # Review of PR #6, 10 Nov 2026: a failed pitch stage still headlined.
        out = S.render_first_listen({'singer': 'SINGER:\n  pitch: comfortable range C3–C4\n'},
            {'singer': {'stage_status': [{'stage': 'pitch', 'status': 'failed', 'kind': 'audio',
                                          'reason': 'invented failure'}]}})
        self.assertNotIn('SINGER pitch', out)

    def test_modulation_drops_home_key_reference_from_tension_turns(self):
        # Review of PR #6, 10 Nov 2026: after a key change the home key is not the reference.
        cards = self.cards()
        cards['harmony'] = cards['harmony'].replace('  home: C major, working reference only\n',
            '  home: C major, working reference only\n  key area changes at 0:30 → G major\n')
        self.assertIn('key area changes', cards['harmony'])
        out = S.render_first_listen(cards)
        self.assertNotIn('HARMONY (in C major) RELEASE', out)
        self.assertNotIn('HARMONY (in C major): ends on', out)

    def test_long_authoritative_failure_keeps_full_reason_with_bounded_headlines(self):
        reason = 'invented diagnostic ' * 200 + 'END_OF_REASON'
        out = S.render_first_listen({'singer': 'SINGER:\n  pitch: must not headline\n'},
            pass_statuses={'singer': {'status': 'failed', 'reason': reason}})
        self.bounded(out)
        self.assertIn('singer: failed: full reason after lyrics', out)
        self.assertIn(reason, ' '.join(out.split()))
        self.assertNotIn('must not headline', out)

    def test_wrapped_stage_failure_is_in_notes_and_blocks_headline(self):
        raw = 'KEY: C major\n  NOTE: key:\n            failed: invented wrapped failure\n'
        out = S.render_first_listen({'music': raw})
        self.assertNotIn('failed/unavailable reported', out)
        self.assertIn('invented wrapped failure', out)
        self.assertGreater(out.index('invented wrapped failure'), out.index('LYRICS:'))
        self.assertNotIn('KEY: C major', out)
        self.bounded(out)

    def test_identical_notes_fold_across_text_status_and_json(self):
        note = 'lyrics: re-listened to 8 long gaps and recovered 12 words'
        cards = {'music': '  NOTE:     ' + note.replace(' on ', '  on\t') + '\n',
                 'singer': 'NOTES: ' + note + '\n'}
        analyses = {'harmony': {'notes': [note + ' [shared cache]']}}
        before = copy.deepcopy((cards, analyses))
        out = S.render_first_listen(cards, analyses)
        self.assertEqual((cards, analyses), before)
        self.assertIn('NOTES [music pass·singer pass·harmony pass]: ' + note + '\n', out)
        self.assertEqual(out.count('re-listened to 8'), 1)
        self.assertNotIn('[shared cache]', out)
        self.bounded(out)

    def test_notes_with_different_numbers_stay_separate(self):
        cards = {'music': 'NOTE: lyrics: recovered 8 words\n',
                 'singer': 'NOTE: lyrics: recovered 56 words\n',
                 'harmony': 'NOTE: lyrics: recovered 8 words [shared cache]\n'}
        out = S.render_first_listen(cards)
        self.assertIn('NOTES [music pass·harmony pass]: lyrics: recovered 8 words\n', out)
        self.assertIn('NOTES [singer pass]: lyrics: recovered 56 words\n', out)
        self.assertEqual(out.count('recovered '), 2)
        self.bounded(out)

    def test_note_near_matches_keep_punctuation_and_case(self):
        cards = {'music': 'NOTE: probe: Ready.\n', 'singer': 'NOTE: probe: ready.\n',
                 'harmony': 'NOTE: probe: Ready\nNOTE: probe: Ready. [shared evidence] extra\n'}
        out = S.render_first_listen(cards)
        for label, text in (('music', 'Ready.'), ('singer', 'ready.'), ('harmony', 'Ready'),
                            ('harmony', 'Ready. [shared evidence] extra')):
            self.assertIn(f'NOTES [{label} pass]: probe: {text}\n', out)
        self.bounded(out)

    def test_failure_before_same_stage_success_and_stable_severity_order(self):
        import stage_status as ST
        analyses = {'music': {'stage_status': [ST.outcome('probe', reason='ready', kind='input'),
                                               ST.outcome('first', 'degraded', 'invented fallback')],
                              'notes': ['informational first']},
                    'singer': {'stage_status': [ST.outcome('probe', 'failed', 'invented reason'),
                                                ST.outcome('optional', 'skipped', 'disabled'),
                                                ST.outcome('second', 'degraded', 'invented retry')],
                               'notes': ['informational second']}}
        out = S.render_first_listen({}, analyses)
        block = out.split('LYRICS: no saved lyric rows supplied\n')[1]
        expected = ['NOTES [singer pass]: probe: failed: invented reason',
                    'NOTES [singer pass]: optional: skipped: disabled',
                    'NOTES [music pass]: first: degraded: invented fallback',
                    'NOTES [singer pass]: second: degraded: invented retry',
                    'NOTES [music pass]: probe: ready',
                    'NOTES [music pass]: informational first',
                    'NOTES [singer pass]: informational second', S.INVITATION]
        self.assertEqual(block.splitlines(), expected)
        self.bounded(out)

    def test_omitted_status_counts_are_one_line_with_each_pass(self):
        cards = {name: f'NOTES: all stages ran · +{count} more\n'
                 for name, count in (('music', 2), ('singer', 4), ('harmony', 3))}
        out = S.render_first_listen(cards)
        expected = ('NOTES: 9 more status entries omitted by source (music: 2; singer: 4; harmony: 3); '
                    'supply original JSON for details')
        self.assertIn(expected + '\n', out)
        self.assertEqual(out.count('omitted by source'), 1)
        self.bounded(out)

    def test_malformed_supplemental_json_is_absent_not_fatal(self):
        cards = {'music': 'MUSIC: example.wav · 0:30\nKEY: C major\n'}
        good = {'t': 3, 'text': 'invented words'}
        cases = [{'duration': 1e309}, {'duration': float('nan')}, {'duration': '30'}, {'file': 5},
                 {'key_claim': 'C major'}, {'key_verdict': ['C major']}, {'key_claim': {'line': 7}},
                 {'singer': {'lines': 5}}, {'notes': 5}, {'stage_status': 'oops'}, {'stage_status': [1, 'x']},
                 {'lyrics_aligned': 'oops'}, {'lyrics_aligned': {'lines': 5}}, {'lyrics': 5}]
        for row in ({'text': 'no time'}, {'t': None, 'text': 'x'}, {'t': 'x', 'text': 'x'},
                    {'t': float('nan'), 'text': 'x'}, {'t': 1e309, 'text': 'x'}, {'t': 2, 'text': 7}, 'text', 4):
            cases += [{'lyrics': [row, good]}, {'lyrics_aligned': {'lines': [row, good]}},
                      {'lyrics': {'rows': [row, good], 'label': 5}}]
        for name in ('music', 'singer', 'harmony'):
            for case in cases:
                with self.subTest(name=name, case=case):
                    out = S.render_first_listen(cards, {name: case})
                    self.assertTrue(out.endswith(S.INVITATION + "\n"))
                    self.assertNotIn('inf', out.split('\n', 1)[0])
        out = S.render_first_listen({}, {'singer': {'lyrics': [{'t': 1e309, 'text': 'bad'}, good]}})
        self.assertIn('[0:03] invented words', out)
        self.assertNotIn('bad', out)

    def test_folded_long_notes_wrap_without_losing_examples(self):
        import stage_status as ST
        note = "lyrics: 16 uncertain lines: " + '; '.join(f"{i}: 'invented example failed {i}'" for i in range(8))
        cards = {'music': '\n'.join(ST.wrap_note('  NOTE: ' + note)),
                 'singer': 'NOTE: ' + note + '\n',
                 'harmony': 'NOTE: ' + note + ' [shared cache]\nNOTES: informational last\n'}
        out = S.render_first_listen(cards)
        self.assertIn('NOTES [music pass·singer pass·harmony pass]: ' + note, ' '.join(out.split()))
        self.assertEqual(out.count('invented example failed 7'), 1)
        self.bounded(out)

    def feel_raw(self, *notes):
        return "MUSIC: example.wav · 0:30\nKEY: C major\nLYRICS: writer's text\n  [0:02] Invented calibration words\n" + \
            ''.join('  NOTE: ' + n + '\n' for n in notes)

    def test_expected_skips_move_to_one_count_line(self):
        out = S.render_first_listen({'music': self.feel_raw(
            'Whisper: skipped: word times supplied', 'lyrics file: supplied',
            "lyrics: names list fed to Whisper (Invented, Names.)",
            'lyrics: re-listened to 3 long gap(s) in short windows and recovered 8 words the one-pass transcript missed')})
        self.assertNotIn('word times supplied', out)
        self.assertNotIn('lyrics file', out)
        self.assertNotIn('names list', out)
        self.assertNotIn('re-listened', out)
        self.assertIn('NOTES: 4 more on the full card (--view full)', out)

    def test_failed_and_unknown_skips_stay_on_feel_card(self):
        out = S.render_first_listen({'music': self.feel_raw(
            'Whisper: skipped: word times supplied', 'probe: failed: invented reason',
            'tempo: degraded: invented fallback', 'key: skipped: invented other reason')})
        self.assertIn('probe: failed: invented reason', out)
        self.assertIn('tempo: degraded: invented fallback', out)
        self.assertIn('key: skipped: invented other reason', out)
        self.assertIn('NOTES: 1 more on the full card (--view full)', out)

    def test_unsure_lines_and_whisper_guess_words_stay(self):
        note = "lyrics: 2 line(s) kept but marked (unsure), quiet vocals or doubtful Whisper: 0:31 'invented a'; 0:36 'invented b'"
        raw = self.feel_raw(note, 'lyrics file: supplied').replace('writer\'s text', "Whisper's guesses")
        out = S.render_first_listen({'music': raw})
        self.assertIn(note, ' '.join(out.split()))
        self.assertIn("LYRICS: Whisper's guesses", out)
        self.assertEqual(out.count("'invented"), 2)

    def test_unsure_examples_capped_at_two_with_count(self):
        note = "lyrics: 4 line(s) kept but marked (unsure), quiet vocals or doubtful Whisper: 0:31 'invented a'; 0:36 'invented b'; 0:40 'invented c'; 0:44 'invented d'"
        out = S.render_first_listen({'music': self.feel_raw(note)})
        self.assertIn('Whisper was doubtful on 4 line(s)', out)
        self.assertIn('alignment check only', out)
        self.assertEqual(out.count("'invented"), 2)

    def test_nothing_to_move_means_no_count_line(self):
        out = S.render_first_listen({'music': self.feel_raw('probe: failed: invented reason')})
        self.assertNotIn('more on the full card', out)

    def test_one_pass_keeps_notes_order_and_lyric_evidence(self):
        raw = '''MUSIC: example.wav · 0:30
KEY: C major
LYRICS: writer's text, one heard
  [0:02] Invented calibration words
  NOTE: lyrics: invented diagnostic [shared cache]
  NOTE: probe: failed: invented reason
NOTES: informational entry · probe: degraded: invented fallback · +2 more
'''
        expected = '''FIRST LISTEN: example · 0:30
PASSES: music card
singer: not supplied (not run or no saved card; unknown)
harmony: not supplied (not run or no saved card; unknown)
KEY: C major
LYRICS: writer's text, one heard [music]
[0:02] Invented calibration words
music NOTES: informational entry
music NOTES: probe: degraded: invented fallback
music NOTES: 2 more status entries omitted by source; supply original JSON for details
music NOTES: NOTE: lyrics: invented diagnostic [shared cache]
music NOTES: NOTE: probe: failed: invented reason
and now, you
'''
        self.assertEqual(S.render_first_listen({'music': raw}), expected)
        self.bounded(expected)

    def test_notes_fix_wraps_and_exposes_every_entry_without_mutating_evidence(self):
        import stage_status as ST
        rows = [ST.outcome('words', reason="Whisper's guesses", kind='input'),
                ST.outcome('LRCLIB', 'skipped', 'lyrics file supplied', kind='input')]
        rows += [ST.outcome('stage%d' % i, 'failed', 'invented reason ' * 10) for i in range(4)]
        before = copy.deepcopy(rows)
        note = '  NOTE: lyrics: ' + 'invented detail ' * 30
        lines = ST.add_notes(["LYRICS: writer's text, one heard", note], rows)
        self.assertEqual(rows, before)
        self.assertTrue(all(len(line) <= 118 for line in lines))
        self.assertEqual(sum(line.startswith('NOTES:') for line in lines), 1)
        self.assertIn("words: writer's text", '\n'.join(lines))
        self.assertEqual(ST.source_notes('\n'.join(lines)), [' '.join(note.split())])
        details = ST.card_notes('\n'.join(lines))
        for i in range(4):
            self.assertIn('stage%d: failed: ' % i + ('invented reason ' * 10).strip(), details)
        self.assertTrue(any('LRCLIB: skipped: lyrics file supplied' in item for item in details))
        self.assertFalse(any('omitted by source' in item for item in details))
        combined = '\n'.join(lines) + '\nNOTES: another stage: failed: another reason\n'
        self.assertIn('another stage: failed: another reason', ST.card_notes(combined))

    def test_cli_saved_passes_and_unreadable_optional_pass(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            cards = self.cards()
            for name, text in cards.items():
                (root / name).write_text(text, encoding='utf-8')
            args = [str(root / 'music'), '--view', 'first-listen', '--singer-card', str(root / 'singer'),
                    '--harmony-card', str(root / 'harmony')]
            for missing in (False, True):
                if missing:
                    (root / 'singer').unlink()
                buf = io.StringIO()
                with redirect_stdout(buf):
                    self.assertEqual(S.main(args), 0)
                out = buf.getvalue()
                self.bounded(out)
                self.assertIn('[0:10] A sample tray is waiting by the scale', out)
                self.assertIn('singer card unavailable' if missing else 'comfortable range G3–G4', out)


def run_checks(check, tmp=None):
    # An accidental network call is a failed check, even with local cached data.
    with mock.patch("socket.create_connection", side_effect=AssertionError("network in slim tests")):
        for cls in (SlimTests, FirstListenTests):
            for name in unittest.defaultTestLoader.getTestCaseNames(cls):
                result = unittest.TestResult()
                cls(name).run(result)
                details = "\n".join(trace for _, trace in result.failures + result.errors)
                check("slim: " + name.removeprefix("test_").replace("_", " "), result.wasSuccessful(), details)


if __name__ == "__main__":
    unittest.main()
