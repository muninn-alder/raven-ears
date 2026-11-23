"""Synthetic text only. Also called by test_ears_plus.py --fast."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import modes as M


# Neutral invented words, names and evidence; no private lyrics or cached data.
CARD = '''# A small song
=================== music
MUSIC: example.mp3 · 3:20
  KEY: C major · tonic unclear
  TEMPO: 96 BPM · steady 0.90 · beat strength 2.4x · meter 4/4, see METER
  SHAPE: quiet intro → loud 0:30 → quiet 1:00 → loud 1:30 → quiet 2:00 → loud 2:30 → quiet outro 3:00
    0:00–0:30 quiet -24 dB
    0:30–1:00 loud -14 dB
    1:00–1:30 quiet -22 dB
    1:30–2:00 loud -14 dB
    2:00–2:30 quiet -22 dB
    2:30–3:00 loud -14 dB
    3:00–3:20 quiet -24 dB
  SOUND: warm (centroid 1900 Hz) · dynamic range 12 dB
  LYRICS: writer's text, one placed line (≈)
    [0:10] I place a sample tray beside the scale
    [0:20≈] A second probe is logging from the rack (unsure: thin evidence)
    [0:30] Check the counter over the interval
    [0:45] Follow the marker across the plot
    [1:00] Another batch is ready for the counter
    [1:15] An open channel sends the measured value
    [1:30] Check the counter over the interval
    [1:45] Follow the marker across the plot
    [2:00] A silver bracket holds the sensor cable
    [2:15] The clocks require a calibration check
    [2:30] Check the counter over the interval
    [2:45] Follow the marker across the plot
=================== singer
INSTRUMENTS (model estimate, not a verdict):
  overall:
    keys (organ or piano?) likely throughout
  who is playing ('+ voice' = the vocal stem has level):
    0:00–0:30 piano (possibly) + voice
    0:30–1:00 piano (possibly), bass guitar (likely) + voice
CREDITS (local file; names what the ears hear):
  credited, not clearly heard:
    flute (A. Player): credited, not clearly heard
ENTRANCES/EXITS:
  0:02 piano enters (likely)
  0:31 bass guitar enters (possibly)
METER: 4/4 lean, not certain
  model supports a four-beat grouping (estimate)
=================== harmony
HARMONY: example.mp3 · 3:20
CHORDS
  home: C major · tonic unclear
  0:00–0:30 [quiet]: I – IV
      no clear arrival on I
  0:30–1:00 [loud]: IV – V
  ends on V; opens on I
TIMELINE
  0:00 opens quiet intro, -24 dB
  0:03 0:03–1:00 tension climbs, FUTURE_RANGE
  0:10 voice enters: 'I place a sample tray'
  0:12 pedal on C until 1:30; FUTURE_PEDAL
  0:14 swell: the sound grows 6 dB
  0:15 borrowed chord (likely)
  1:00 new section: quiet
  2:00 last chorus: 'FUTURE_WORD' comes three times
  3:18 last chord: V
CONTRAST (words vs music)
  overall: the words lean heavy (-0.4 avg); music lifts
=================== and now, you
  Old invitation stays untouched in all.
'''


class ListeningModesTests(unittest.TestCase):
    def test_all_is_unchanged(self):
        for text in (CARD, CARD.replace('\n', '\r\n'), 'a card without final newline', ''):
            self.assertEqual(M.render(text), text)

    def test_all_cli_keeps_bytes_and_does_not_add_a_newline(self):
        raw = b'card\r\nno trailing newline'
        result = subprocess.run([sys.executable, str(Path(M.__file__)), '-', '--mode', 'all'],
                                input=raw, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, raw)

    def test_every_filtered_pass_ends_with_invitation(self):
        passes = [('first-listen', None), ('vibe', None)] + [('lens', x) for x in M.LENSES]
        for mode, lens in passes:
            out = M.render(CARD, mode, lens)
            self.assertEqual(out.splitlines()[-1], M.INVITATION)
            self.assertEqual(out.count(M.INVITATION), 1)

    def test_form_returns_and_late_new_bridge(self):
        sections = M.label_sections(M.parse_card(CARD))
        self.assertEqual([s.kind for s in sections], ['VERSE', 'CHORUS', 'VERSE', 'CHORUS', 'BRIDGE', 'CHORUS'])
        self.assertEqual([(s.start, s.end) for s in sections],
                         [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 200)])
        self.assertTrue(all('likely' in s.reason for s in sections))

    def test_first_listen_is_one_section_and_hides_future(self):
        out = M.render(CARD, 'first-listen')
        self.assertIn('FIRST LISTEN: section 1', out)
        self.assertIn('sample tray', out)
        self.assertIn('piano enters (likely)', out)
        self.assertIn('no clear arrival on I', out)
        for future in ('Check the sample', 'silver bracket', 'FUTURE_RANGE', 'ends on V', 'quiet outro', '3:18'):
            self.assertNotIn(future, out)
        self.assertIn('pedal on C until 0:30', out)
        self.assertIn('--next', out)
        self.assertNotIn('--section 2', out)

    def test_requested_section_only(self):
        out = M.render(CARD, 'first-listen', section=5)
        self.assertIn('FIRST LISTEN: section 5', out)
        self.assertIn('silver bracket', out)
        self.assertNotIn('sample tray', out)
        self.assertNotIn('Check the sample', out)
        final = M.render(CARD, 'first-listen', section=6)
        self.assertIn('FIRST LISTEN: section 6', final)
        self.assertIn('3:20', final)
        self.assertNotIn('--section 7', final)

    def test_no_refrain_means_form_unclear_not_an_invented_bridge(self):
        c = M.parse_card('MUSIC: example.mp3 · 2:00\n  SHAPE: changing\n    0:00–1:00 mid\n    1:00–2:00 loud\n')
        parts = M.label_sections(c)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(s.kind == 'VERSE' and 'unclear' in s.reason for s in parts))
        self.assertNotIn('silence', M.render(c.raw, 'first-listen').splitlines()[1])

    def test_short_repeated_line_and_changed_row_wrapping(self):
        text = '''MUSIC: example.mp3 · 2:00
  LYRICS: rough times
    [0:10] One small window shines
    [0:35] Other words before the return
    [1:00] One small window shines
'''
        self.assertEqual(sum(s.kind == 'CHORUS' for s in M.label_sections(M.parse_card(text))), 2)
        rows = [M.Event(10, 'Walk across the bridge and carry', 'words'),
                M.Event(20, 'all the light into the garden', 'words'),
                M.Event(40, 'Different words for an ordinary morning', 'words'),
                M.Event(70, 'Walk across the bridge', 'words'),
                M.Event(80, 'and carry all the light into the garden', 'words')]
        runs = M._returns(rows)
        self.assertEqual([a for a, b in runs], [0, 3])

    def test_chant_is_not_enough_to_call_a_chorus(self):
        rows = [M.Event(t, 'la la la la la la la la', 'words') for t in (10, 40, 70)]
        self.assertEqual(M._returns(rows), [])

    def test_words_lens_keeps_uncertainty_and_excludes_other_channels(self):
        out = M.render(CARD, 'lens', 'words')
        self.assertIn('[0:20≈]', out)
        self.assertIn('unsure: thin evidence', out)
        for unrelated in ('TEMPO:', 'KEY:', 'piano enters', 'IV – V', 'borrowed chord', 'FUTURE_WORD'):
            self.assertNotIn(unrelated, out)

    def test_groove_lens_keeps_meter_caveat_without_lyrics(self):
        out = M.render(CARD, 'lens', 'groove')
        self.assertIn('TEMPO: 96 BPM', out)
        self.assertIn('METER: 4/4 lean, not certain', out)
        self.assertIn('model supports a four-beat grouping (estimate)', out)
        self.assertIn('swell:', out)
        self.assertNotIn('sample tray', out)
        self.assertNotIn('piano enters', out)
        self.assertNotIn('borrowed chord', out)

    def test_harmony_lens_quotes_one_key_and_timed_chords(self):
        out = M.render(CARD, 'lens', 'harmony')
        self.assertIn('KEY: C major · tonic unclear', out)
        self.assertEqual(out.count('tonic unclear'), 1)
        self.assertIn('IV – V', out)
        self.assertIn('no clear arrival on I', out)
        self.assertNotIn('Check the sample', out)
        self.assertNotIn('piano enters', out)

    def test_instruments_lens_preserves_unheard_credits(self):
        out = M.render(CARD, 'lens', 'instruments')
        self.assertIn('keys (organ or piano?) likely', out)
        self.assertIn('flute (A. Player): credited, not clearly heard', out)
        self.assertIn('bass guitar enters (possibly)', out)
        self.assertNotIn('sample tray', out)
        self.assertNotIn('KEY:', out)

    def test_lens_events_are_chronological(self):
        c = M.parse_card(CARD)
        self.assertEqual([e.t for e in c.events], sorted(e.t for e in c.events))

    def test_fractional_times_and_key_changes_stay_on_the_timeline(self):
        text = 'MUSIC: example.mp3 · 1:00\n  KEY CHG: 0:12.5 → tonic unclear\n'
        out = M.render(text, 'lens', 'harmony')
        self.assertIn('[0:12.5] → tonic unclear', out)
        self.assertEqual(M.clock(59.9999), '1:00')
        self.assertEqual(M._channel('pedal on C in the bass'), 'harmony')

    def test_lyric_span_crossing_a_section_is_withheld(self):
        analysis = {'lyrics_aligned': {'lines': [
            {'t': 10, 'end': 40, 'text': 'A local onset with FUTURE_SPAN', 'approx': True}]}}
        out = M.render(CARD, 'first-listen', analysis=analysis)
        self.assertNotIn('FUTURE_SPAN', out)

    def test_last_onset_is_not_invented_as_song_duration(self):
        text = '  LYRICS: rough times\n    [0:10] One line with an unknown ending\n'
        with self.assertRaisesRegex(ValueError, 'no usable duration'):
            M.render(text, 'first-listen')

    def test_vibe_is_two_lines_plus_status_and_listener_and_labels_inference(self):
        out = M.render(CARD, 'vibe')
        self.assertEqual(len(out.splitlines()), 5)
        self.assertIn('96 BPM', out)
        self.assertIn('lyrics lean heavy (source model estimate', out)
        self.assertIn('bop lean: likely', out)
        self.assertNotIn('sample tray', out)
        self.assertNotIn('C major', out)

    def test_missing_evidence_is_never_silence(self):
        text = 'MUSIC: example.mp3 · 2:00\n  NOTE: LYRICS FAILED: no lyrics produced\n'
        for mode, lens in [('vibe', None), ('first-listen', None)] + [('lens', x) for x in M.LENSES]:
            out = M.render(text, mode, lens)
            self.assertIn('failure/skip reported', out)
            self.assertNotIn('instrumental', out)
        vibe = M.render(text, 'vibe')
        self.assertIn('tempo unavailable', vibe)
        self.assertIn('mood unclear', vibe)
        self.assertIn('bop unclear', vibe)

    def test_failure_note_cannot_spoil_later_lyrics(self):
        text = CARD.replace('  LYRICS:', "  NOTE: lyrics skipped at 2:00: 'SECRET_FUTURE'\n  LYRICS:")
        out = M.render(text, 'first-listen')
        self.assertIn('failure/skip', out)
        self.assertNotIn('SECRET_FUTURE', out)

    def test_lyric_timeline_splits_words_from_chords(self):
        text = '''HARMONY: example.mp3 · 1:00
LYRIC TIMELINE
  LYRICS: writer's text, one placed (≈)
  [0:10≈] "A quiet invented line" — chord I→IV, tension falls 0.7→0.2
  [0:30] ♪ guitar enters (likely)
'''
        out = M.render(text, 'lens', 'words')
        self.assertIn('[0:10≈] A quiet invented line', out)
        self.assertNotIn('chord', out)
        self.assertNotIn('guitar', out)
        self.assertIn('chord I→IV', M.render(text, 'lens', 'harmony'))
        self.assertIn('guitar enters (likely)', M.render(text, 'lens', 'instruments'))

    def test_existing_analysis_supplies_missing_evidence_not_new_key(self):
        analysis = {'duration': 80, 'key_verdict': {'line': 'A minor'},
                    'lyrics_aligned': {'lines': [{'t': 5, 'end': 10, 'text': 'An invented line', 'approx': True}]},
                    'timeline': [{'t': 20, 'text': 'guitar enters (likely)'}]}
        out = M.render('KEY: C major · unclear\n', 'lens', 'words', analysis=analysis)
        self.assertIn('[0:05–0:10≈] An invented line', out)
        self.assertNotIn('A minor', M.render('KEY: C major · unclear\n', 'lens', 'harmony', analysis=analysis))
        self.assertIn('guitar enters (likely)', M.render('', 'lens', 'instruments', analysis=analysis))

    def test_given_unmatched_lyrics_are_never_rendered_or_assigned_times(self):
        out = M.render(CARD, 'first-listen', lyrics='SECRET_UNHEARD\n\nMore unheard words here')
        self.assertNotIn('SECRET_UNHEARD', out)
        self.assertNotIn('More unheard', out)

    def test_given_stanzas_can_extend_refrain_to_block_end(self):
        c = M.parse_card(CARD)
        given = '''I place a sample tray beside the scale
A second probe is logging from the rack

Check the counter over the interval
Follow the marker across the plot

Another batch is ready for the counter
An open channel sends the measured value

Check the counter over the interval
Follow the marker across the plot

A silver bracket holds the sensor cable
The clocks require a calibration check

Check the counter over the interval
Follow the marker across the plot'''
        self.assertEqual([s.start for s in M.label_sections(c, given)], [0, 30, 60, 90, 120, 150])

    def test_invalid_requests_fail_clearly(self):
        for section in (0, -1, 7):
            with self.assertRaisesRegex(ValueError, '--section must be between'):
                M.render(CARD, 'first-listen', section=section)
        with self.assertRaisesRegex(ValueError, 'needs --lens'):
            M.render(CARD, 'lens')
        with self.assertRaisesRegex(ValueError, 'no usable duration'):
            M.render('Traceback: engine failed', 'first-listen')

    def test_cli_uses_existing_json_and_lyrics_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'card').write_text('MUSIC: example.mp3 · 1:00\n', encoding='utf-8')
            (root / 'analysis').write_text(json.dumps({'lyrics': [{'t': 5, 'text': 'Words from the card'}]}), encoding='utf-8')
            (root / 'lyrics').write_text('Unmatched supplied words', encoding='utf-8')
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = M.main([str(root / 'card'), '--mode', 'first-listen', '--analysis', str(root / 'analysis'),
                             '--lyrics', str(root / 'lyrics'), '--no-state'])
            self.assertEqual(rc, 0)
            self.assertIn('Words from the card', stdout.getvalue())
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})


def run_tests(check):
    """Small hook for the repository's own runner; no separate test framework."""
    class Result(unittest.TestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            check('modes: ' + test._testMethodName.removeprefix('test_'), True)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            check('modes: ' + test._testMethodName, False, self._exc_info_to_string(err, test))

        addError = addFailure

    unittest.defaultTestLoader.loadTestsFromTestCase(ListeningModesTests).run(Result())


if __name__ == '__main__':
    unittest.main()
