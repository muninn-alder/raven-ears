"""Neutral text fixtures only; no models/network. No existing user cards are read.

    python test_relisten_compare.py
    python test_relisten_compare.py --fast   # existing fast suite, then this suite

run_checks(check) is an optional shared-runner hook.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import compare as C
import relisten as R


CARD = '''# Artist - Paper Boat (2026-11-03 19:00)
MUSIC: Artist - Paper Boat.mp3 · 2:00
  KEY: C major · tonic unclear
  TEMPO: 90 BPM · steady 0.90 · beat strength 2x
  SHAPE: quiet intro → PEAK loud 1:00 → quiet outro 1:50
    0:00–1:00 quiet -22 dB
    1:00–1:50 loud -12 dB <- PEAK
    1:50–2:00 quiet -24 dB
  VOCALS: yes, voice present ~80% (stem-energy estimate)
  VOCAL: breathier 1:50–2:00 (unclear)
  LYRICS: writer's text; one placed line (≈)
    [0:10≈] A neutral invented line (unsure: placed)
SINGER:
  pitch: median C4 · comfortable range G3–C5
  highest held notes: C5 at 1:20 (lean)
  pitch arc: G3 → C4 · the voice lifts
  most intense moment: 1:20–1:25 (speech-trained model; texture, not a verdict)
INSTRUMENTS (tagger estimate):
  overall:
    keys (organ or piano?) possibly throughout
    drum machine likely throughout
  who is playing ('+ voice' = stem has level):
    0:00–0:10 keys (possibly)
CREDITS (local names):
  credited and heard:
    piano (A. Player): 0:00–2:00
  credited, not clearly heard:
    flute (B. Player): credited, not clearly heard
ENTRANCES/EXITS:
  0:00 instruments at the open: keys (possibly)
METER: 4/4 lean
  grouping confidence low; also consistent with 12/8
CHORDS
  home: C major · tonic unclear; decoded on beats
TENSION
  range 0.1–0.7, peak at 1:10
TIMELINE
   0:00 opens quiet intro, -22 dB
CONTRAST (words vs music)
  overall: words lean heavy; music lifting (model estimate)
ACHE 1:20 ≈ "A neutral invented line"; unresolved V (lean)
=================== and now, you
  A previous invitation's prose.
'''


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.listens = self.root / 'listens'
        self.music = self.root / 'music'
        self.music.mkdir()
        self.song = 'Artist - Paper Boat'

    def write(self, filename, content, directory=None):
        directory = directory or self.listens / R.slug(self.song)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text(content, encoding='utf-8')
        return path

    def memory(self, **kwargs):
        return R.read_memory(self.song, self.listens, self.music, **kwargs)

    def test_first_time_and_invitation(self):
        out = R.prepend_memory('', self.memory())
        self.assertEqual(out, 'first time hearing this one\n\nand now, you\n')
        self.assertFalse(self.listens.exists())

    def test_json_history_dates_sort_and_notes(self):
        self.write('history.json', json.dumps({'listens': [
            {'date': '2026-11-01', 'note': 'A small lift.'},
            {'timestamp': '2026-11-03T20:00:00Z', 'notes': ['A late ache.', 'Likely the voice.']},
        ]}))
        out = R.memory_header(self.memory())
        self.assertIn('Previous listens: 2026-11-03, 2026-11-01', out)
        self.assertIn('LAST TIME (3 Nov): A late ache.', out)
        self.assertIn('LAST TIME (3 Nov): Likely the voice.', out)
        self.assertNotIn('first time', out)

    def test_per_note_dates_override_parent(self):
        self.write('entry.json', json.dumps({'date': '2026-11-01', 'notes': [
            {'date': '2026-11-03', 'text': 'A newer note.'}]}))
        self.assertIn('LAST TIME (3 Nov): A newer note.', R.memory_header(self.memory()))

    def test_markdown_and_filename_date(self):
        self.write('2026-11-01.md', 'The rhythm stayed with me.\n\n## 2026-11-03\nUnclear, but tender.\n')
        self.write('2026-11-02.json', json.dumps({'feeling': 'A little warmer.'}))
        out = R.memory_header(self.memory())
        self.assertIn('LAST TIME (3 Nov): Unclear, but tender.', out)
        self.assertIn('LAST TIME (2 Nov): A little warmer.', out)
        self.assertIn('LAST TIME (1 Nov): The rhythm stayed with me.', out)

    def test_undated_notes_never_use_mtime(self):
        self.write('notes.md', 'A feeling without a date.')
        out = R.memory_header(self.memory())
        self.assertIn('LAST TIME (date unknown)', out)
        self.assertNotIn('Previous listens:', out)

    def test_unknown_schema_does_not_become_a_note(self):
        self.write('state.json', '{"analysis": {"valence": 0.8}, "version": 7}')
        out = R.memory_header(self.memory())
        self.assertIn('previous listening unclear', out)
        self.assertNotIn('0.8', out)
        self.assertNotIn('first time', out)

    def test_broken_json_unicode_dates_and_empty_store(self):
        self.write('bad.json', '{')
        self.write('bad-date.json', '{"date":"2026-99-99", "note":"Uncertain."}')
        self.write('bad.md', 'placeholder').write_bytes(b'\xff')
        out = R.memory_header(self.memory())
        self.assertIn('unreadable (JSONDecodeError)', out)
        self.assertIn('unreadable (UnicodeDecodeError)', out)
        self.assertIn('date unknown', out)
        self.assertNotIn('2026-99-99', out)

    def test_sibling_session_hash_uses_saved_card_identity(self):
        directory = self.listens / 'artist---paper-boat-0123456789'
        self.write('state.json', json.dumps({'version': 1, 'card': '/cards/Artist - Paper Boat.txt',
                  'notes': [{'id': 1, 'view': 'lens words', 'text': 'A held breath.'}],
                  'closed': False}), directory)
        out = R.memory_header(self.memory())
        self.assertIn('LAST TIME (date unknown): A held breath.', out)
        self.assertNotIn('first time', out)

    def test_other_song_never_matches_a_prefix(self):
        self.write('state.json', json.dumps({'card': '/cards/Artist - Paper Boat (Live).txt',
                  'notes': ['Wrong version.']}), self.listens / 'artist-paper-boat-live-0123456789')
        self.assertEqual(R.memory_header(self.memory()), 'first time hearing this one')

    def test_broken_matching_hashed_store_is_visible(self):
        self.write('state.json', '{', self.listens / 'artist---paper-boat-0123456789')
        out = R.memory_header(self.memory())
        self.assertIn('unreadable', out)
        self.assertNotIn('first time', out)

    def test_explicit_store_slug_and_path_traversal(self):
        self.write('notes.md', 'The bridge stayed.', self.listens / 'custom-id')
        self.assertIn('The bridge stayed.', R.memory_header(self.memory(song_slug='custom-id')))
        with self.assertRaises(ValueError):
            self.memory(song_slug='../other')

    def test_shelf_exact_heading_and_no_unrelated_prose(self):
        self.write('SHELF.md', '# Vinyl shelf\n\n## Artist - Paper Boat\nA remembered sway.\n\n## Artist - Other Song\nPrivate unrelated note.\n', self.music)
        out = R.memory_header(self.memory())
        self.assertIn('SHELF (SHELF.md;', out)
        self.assertIn('A remembered sway.', out)
        self.assertNotIn('Private unrelated', out)
        self.assertNotIn('first time', out)
        self.assertNotIn('LAST TIME', out)

    def test_shelf_short_title_is_not_a_substring_search(self):
        self.song = 'Artist - All'
        self.write('notes.md', '# For all songs\nAll the notes live here.\n\n## Falling\nUnrelated.\n', self.music)
        self.assertFalse(self.memory().shelf)

    def test_answer_key_heading_attribution_and_preferred_feeling(self):
        self.write('EAR-KEY-paper-boat.md', '# Paper Boat (Artist) — Listener\'s ear, 3 Nov 2026\n\nKey: C major.\n\nFavourite moment: the band returns.\n', self.music)
        out = R.memory_header(self.memory())
        self.assertIn('Favourite moment: the band returns.', out)
        self.assertNotIn('Key: C major.', out)
        self.assertIn("not today's hearing", out)

    def test_shelf_answer_key_cannot_add_another_key_claim(self):
        self.write('EAR-KEY-paper-boat.md', '# Paper Boat\n\nKey / meter / tempo:\nKEY: G major; 12/8 feel.\n\nA remembered sway.\n', self.music)
        out = R.memory_header(self.memory())
        self.assertIn('A remembered sway.', out)
        self.assertNotIn('G major', out)

    def test_different_artist_heading_does_not_match(self):
        self.write('SHELF.md', '# Paper Boat — Someone Else\nTheir note.\n', self.music)
        self.write('EAR-KEY-paper-boat.md', '# Paper Boat (Someone Else)\nA different artist.\n', self.music)
        self.assertFalse(self.memory().shelf)

    def test_read_only_card_preserved(self):
        path = self.write('2026-11-03.md', 'The last chorus opened.')
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        card = 'KEY: C major\nAn existing note (unclear).\nand now, you\n'
        out = R.prepend_memory(card, self.memory())
        self.assertIn(card, out)
        self.assertEqual(out.count('and now, you'), 1)
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))

    def test_cli_card_and_errors_end_with_invitation(self):
        path = self.write('card.txt', 'KEY: C major\n', self.root)
        with redirect_stdout(io.StringIO()) as stream:
            code = R.main([self.song, '--card', str(path), '--listens-dir', str(self.listens), '--music-dir', str(self.music)])
        self.assertEqual(code, 0)
        self.assertTrue(stream.getvalue().endswith('and now, you\n'))
        with redirect_stdout(io.StringIO()) as stream:
            code = R.main([self.song, '--card', str(self.root / 'absent'), '--listens-dir', str(self.listens), '--music-dir', str(self.music)])
        self.assertEqual(code, 2)
        self.assertTrue(stream.getvalue().endswith('and now, you\n'))


class CompareTests(unittest.TestCase):
    def test_all_dimensions_and_no_new_key(self):
        a = C.parse_card(CARD)
        self.assertEqual(a.key, 'C major · tonic unclear')
        self.assertEqual(a.tempo, '90 BPM · steady 0.90 · beat strength 2x')
        out = C.compare_cards(a, a)
        for dimension in ('Tempo / feel', 'Key / mode', 'Opens', 'Shape (level arc)',
                          'Ache candidates / peaks', 'Voice', 'Instruments', 'Words vs music'):
            self.assertIn('| ' + dimension + ' |', out)
        self.assertTrue(out.endswith(C.INVITATION + '\n'))
        self.assertEqual(len(C.feel_differences(a, a)), 5)
        self.assertIn("The ears' read", out)

    def test_caveats_placed_words_and_unheard_credits_survive(self):
        a = C.parse_card(CARD)
        self.assertIn('organ or piano?', a.instruments)
        self.assertIn('possibly', a.instruments)
        self.assertIn('flute (B. Player): credited, not clearly heard', a.instruments)
        self.assertIn('≈', a.peaks)
        self.assertIn('(lean)', a.peaks)
        self.assertIn('source sentiment/music model read:', a.contrast)
        self.assertIn('grouping confidence low', a.meter)
        self.assertIn('also consistent with 12/8', a.meter)

    def test_distinct_peak_sources_not_a_measured_ache(self):
        card = C.parse_card(CARD)
        self.assertIn('level peak: 1:00', card.peaks)
        self.assertIn('voice most intense moment: 1:20', card.peaks)
        self.assertIn('tension proxy: range 0.1–0.7, peak at 1:10', card.peaks)
        self.assertIn('ache source (candidate, not a verdict)', card.peaks)
        self.assertNotIn('A previous invitation', C.compare_cards(card, card))

    def test_missing_and_backend_failure_never_become_silence(self):
        a = C.parse_card('NOTE: instrument backend failed\nKEY: unavailable\nACHE unclear: analysis failed\n')
        out = C.compare_cards(a, C.parse_card(''))
        self.assertIn('instrument backend failed', out)
        self.assertIn('analysis failed', out)
        self.assertIn('unavailable in this card', out)
        self.assertNotIn('silence', out)
        self.assertNotIn('major', out)

    def test_key_tempo_conflicts_and_home_authority(self):
        a = C.parse_card('KEY: C major · unclear\nKEY: G major\nTEMPO: 80 BPM\nTEMPO: 160 BPM\nCHORDS\n  home: A minor; model\n')
        self.assertTrue(a.key.startswith('unclear:'))
        self.assertTrue(a.tempo.startswith('unclear:'))
        self.assertIsNone(a.pulse)
        self.assertNotIn('A minor', a.key)
        self.assertIn('harmony home differs', a.warnings[-1])

    def test_qualified_tempo_not_used_as_certain_feel(self):
        for tempo in ('~90 BPM', '90 BPM likely', '90 BPM · unclear', '0 BPM', '90 BPM ≈'):
            self.assertIsNone(C.parse_card('TEMPO: ' + tempo).pulse)
        a = C.parse_card('TEMPO: 90 BPM')
        b = C.parse_card('TEMPO: 96 BPM')
        self.assertIn('B has the quicker tracked pulse', C.feel_differences(a, b)[0])
        b = C.parse_card('TEMPO: 92 BPM')
        self.assertIn('pulses sit close', C.feel_differences(a, b)[0])

    def test_missing_evidence_is_not_a_feel_difference(self):
        lines = C.feel_differences(C.parse_card(CARD), C.parse_card(''))
        self.assertTrue(all('unclear' in line for line in lines))

    def test_invitation_words_inside_lyrics_do_not_stop_reading(self):
        card = CARD.replace('A neutral invented line', 'and now, you are here')
        parsed = C.parse_card(card)
        self.assertNotEqual(parsed.voice, C.UNAVAILABLE)
        self.assertNotEqual(parsed.contrast, C.UNAVAILABLE)
        self.assertIn('≈', parsed.peaks)

    def test_each_pass_handoff_is_a_block_boundary(self):
        combined = CARD.replace('SINGER:\n', 'and now, you\n\nSINGER.PY example.mp3  2:00\nSINGER:\n')
        combined = combined.replace('CHORDS\n', 'and now, you\n\nHARMONY: example.mp3 · 2:00\nCHORDS\n')
        before, after = C.parse_card(CARD), C.parse_card(combined)
        for field in ('voice', 'instruments', 'contrast', 'peaks'):
            self.assertEqual(getattr(after, field), getattr(before, field), field)
        # Invitation prose must not inherit the block that preceded it.
        parsed = C.parse_card('INSTRUMENTS (tagger estimate):\n  keys possibly\nand now, you\n  stray invitation prose\n')
        self.assertNotIn('stray invitation', parsed.instruments)

    def test_identical_cards_do_not_force_feeling_differences(self):
        a = C.parse_card(CARD)
        differences = '\n'.join(C.feel_differences(a, a))
        self.assertIn('no clear difference', differences)
        self.assertNotIn('happier', differences)
        self.assertNotIn('sadder', differences)

    def test_feel_read_keeps_textual_uncertainty(self):
        a = C.parse_card('CONTRAST (words vs music)\n  overall: words lean heavy (unclear); music lifting (model estimate)\n')
        b = C.parse_card('CONTRAST (words vs music)\n  overall: words sit near neutral (-0.01 avg); separate arcs (r=+0.02)\n')
        words = C.feel_differences(a, b)[-1]
        self.assertIn('(unclear)', words)
        self.assertIn('(model estimate)', words)
        self.assertNotIn('(-0.01 avg)', words)

    def test_engine_only_card_and_unknown_blocks(self):
        a = C.parse_card('HARMONY: example\nCHORDS\n  home: G minor · unclear; model\nUNRECOGNISED\n  0:00 made up\n')
        self.assertEqual(a.key, 'G minor · unclear')
        self.assertEqual(a.opening, C.UNAVAILABLE)
        self.assertEqual(a.shape, C.UNAVAILABLE)

    def test_markdown_table_escapes_pipes(self):
        a = C.parse_card('KEY: C | G (unclear)')
        self.assertIn(r'C \| G (unclear)', C.compare_cards(a, a))

    def test_resolution_exact_identity_and_ambiguous_error(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            path = directory / 'Artist - Paper Boat.txt'
            path.write_text(CARD, encoding='utf-8')
            self.assertEqual(C.resolve_card('Artist - Paper Boat', directory), path)
            with self.assertRaises(ValueError):
                C.resolve_card('Paper Boat', directory)
            (directory / 'artist-paper-boat.txt').write_text('', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                C.resolve_card('Artist - Paper Boat', directory)

    def test_cli_success_error_and_no_writes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'example.txt'
            path.write_text(CARD, encoding='utf-8')
            before = (path.read_bytes(), path.stat().st_mtime_ns)
            with redirect_stdout(io.StringIO()) as stream:
                code = C.main([str(path), str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(stream.getvalue().endswith(C.INVITATION + '\n'))
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            with redirect_stdout(io.StringIO()) as stream:
                code = C.main(['missing-a', 'missing-b', '--cards-dir', root])
            self.assertEqual(code, 2)
            self.assertIn('comparison unavailable', stream.getvalue())
            self.assertTrue(stream.getvalue().endswith(C.INVITATION + '\n'))


def run_checks(check):
    """Adapter for the project's self-test callback; no private file access."""
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (MemoryTests, CompareTests))
    with patch.object(socket, 'socket', side_effect=AssertionError('network forbidden in tests')):
        result = unittest.TestResult()
        suite.run(result)
    failures = dict((test.id(), detail) for test, detail in result.failures + result.errors)
    for cls in (MemoryTests, CompareTests):
        for name in unittest.defaultTestLoader.getTestCaseNames(cls):
            identifier = cls(name).id()
            check('relisten/compare: ' + name, identifier not in failures, failures.get(identifier, ''))


def main():
    if '--fast' in sys.argv:
        # Exercise the unmodified shared runner using its own CLI arguments.
        import test_ears_plus
        with patch.object(sys, 'argv', ['test_ears_plus.py', '--fast']):
            if test_ears_plus.main():
                return 1
    outcomes = []

    def check(name, ok, detail=''):
        outcomes.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ('\n' + detail if detail else ''))

    run_checks(check)
    print(f'{sum(outcomes)} passed, {len(outcomes) - sum(outcomes)} failed')
    return 0 if all(outcomes) else 1


if __name__ == '__main__':
    sys.exit(main())
