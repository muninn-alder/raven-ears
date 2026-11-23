"""Round two: neutral synthetic cards, temporary state, no audio or network."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import listening_sessions as S
import modes as M
import modes_view as V
import raven_listen_modes as W
from test_modes import CARD


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.card = self.root / 'a small song.txt'
        self.card.write_text(CARD, encoding='utf-8')
        self.home = self.root / 'listens'

    def call(self, *flags, card=True):
        args = ([str(self.card)] if card else []) + ['--state-home', str(self.home), *flags]
        return S.execute(S.parser().parse_args(args))

    def state(self):
        return S.load(S.session_path(self.home, self.card))

    def test_resume_replays_current_view_and_editable_notes(self):
        first = self.call('--mode', 'first-listen', '--note', 'a quiet door')
        replay = self.call('--resume')
        self.assertEqual(first, replay)
        self.assertIn('1. [section 1', replay)
        changed = self.call('--edit-note', '1', 'a bright door')
        self.assertIn('a bright door', changed)
        self.assertNotIn('a quiet door', changed)
        self.assertEqual(self.state()['notes'][0]['text'], 'a bright door')
        self.assertEqual(self.card.read_text(encoding='utf-8'), CARD)

    def test_next_needs_no_notes_and_preserves_notes_on_their_view(self):
        self.call('--mode', 'first-listen')
        next_view = self.call('--next', '--note', 'a lantern')
        self.assertIn('section 2 · 0:30–1:00', next_view)
        self.assertEqual(len(self.state()['notes']), 1)
        self.assertTrue(self.state()['notes'][0]['view'].startswith('section 2'))
        self.assertIn('a lantern', self.call('--next'))
        self.assertEqual(self.state()['section'], 3)

    def test_lens_session_resumes_and_advances(self):
        self.call('--mode', 'lens', '--lens', 'words', '--note', 'the words')
        self.assertIn('LENS: words', self.call('--resume', card=False))
        self.assertIn('LENS: groove', self.call('--next'))
        self.assertIn('LENS: harmony', self.call('--next'))
        self.assertIn('LENS: instruments', self.call('--next'))
        self.assertIn('Listen closed', self.call('--next'))
        self.assertIn('No open listens', self.call('--status', card=False))
        self.assertIn('the words', self.call('--resume'))

    def test_pause_continues_from_exact_stop_without_requiring_notes(self):
        out = self.call('--mode', 'first-listen', '--until', '0:17.5')
        self.assertIn('0:00–0:17.5', out)
        self.assertNotIn('A little rain', out)
        resumed = self.call('--resume')
        self.assertEqual(resumed, out)
        advanced = self.call('--next')
        self.assertIn('0:17.5–0:30', advanced)
        self.assertNotIn('[0:10]', advanced)
        self.assertIn('0:30–1:00', self.call('--next'))

    def test_from_until_can_cross_form_boundaries(self):
        out = self.call('--mode', 'first-listen', '--from', '0:45', '--until', '1:42')
        self.assertIn('0:45–1:42', out)
        self.assertNotIn('0:00', out)
        self.assertNotIn('2:00', out)
        self.assertIn('1:42–2:00', self.call('--next'))

    def test_single_open_listen_and_status_survive_a_process_reset(self):
        script = str(Path(M.__file__))
        args = [sys.executable, script, str(self.card), '--mode', 'first-listen', '--note', 'saved thought',
                '--state-home', str(self.home)]
        first = subprocess.run(args, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        replay = subprocess.run([sys.executable, script, '--resume', '--state-home', str(self.home)], capture_output=True)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(first.stdout, replay.stdout)
        status = self.call('--status', card=False)
        self.assertIn(str(self.card), status)
        self.assertIn('first-listen, section 1; 1 note(s)', status)

    def test_same_song_names_in_different_directories_do_not_collide(self):
        self.call('--mode', 'first-listen')
        other = self.root / 'other' / self.card.name
        other.parent.mkdir()
        other.write_text(CARD)
        S.execute(S.parser().parse_args([str(other), '--mode', 'lens', '--lens', 'groove', '--state-home', str(self.home)]))
        self.assertEqual(len(S.open_listens(self.home)), 2)
        with self.assertRaisesRegex(ValueError, 'supply a card'):
            self.call('--resume', card=False)

    def test_invalid_operations_leave_the_saved_state_intact(self):
        self.call('--mode', 'first-listen', '--note', 'keep me')
        path = S.session_path(self.home, self.card)
        before = path.read_bytes()
        for flags in (('--edit-note', '0', 'bad'), ('--edit-note', '2', 'bad'),
                      ('--edit-note', 'x', 'bad'), ('--until', '9:00'), ('--from', '0:20', '--until', '0:10'),
                      ('--section', '0'), ('--section', '999'), ('--mode', 'lens'),
                      ('--mode', 'all', '--note', 'bad'), ('--mode', 'lens', '--lens', 'words', '--until', '0:20')):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.call(*flags)
            self.assertEqual(path.read_bytes(), before)

    def test_interrupted_atomic_save_keeps_previous_state_and_cleans_temp(self):
        self.call('--mode', 'first-listen', '--note', 'keep me')
        path = S.session_path(self.home, self.card)
        before = path.read_bytes()
        with patch.object(S.os, 'replace', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                self.call('--note', 'not yet saved')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob('.state-*')), [])

    def test_corrupt_state_is_reported_and_not_replaced(self):
        self.call('--mode', 'first-listen')
        path = S.session_path(self.home, self.card)
        path.write_text('{ broken')
        with self.assertRaisesRegex(ValueError, 'original state was left intact'):
            self.call('--resume')
        self.assertEqual(path.read_text(), '{ broken')

    def test_changed_card_does_not_silently_reassign_notes(self):
        self.call('--mode', 'first-listen', '--note', 'for this card')
        self.card.write_text(CARD + '\nchanged')
        with self.assertRaisesRegex(ValueError, 'source card changed'):
            self.call('--resume')
        self.assertEqual(self.state()['notes'][0]['text'], 'for this card')

    def test_analysis_and_lyrics_remain_available_after_context_reset(self):
        analysis = self.root / 'analysis.json'
        analysis.write_text(json.dumps({'lyrics_aligned': {'lines': [
            {'t': 5, 'end': 10, 'text': 'An invented small line', 'approx': True}]}}))
        lyrics = self.root / 'lyrics.txt'
        lyrics.write_text('An invented small line')
        out = self.call('--mode', 'first-listen', '--analysis', str(analysis), '--lyrics', str(lyrics))
        analysis.unlink()
        lyrics.unlink()
        self.assertEqual(self.call('--resume'), out)
        self.assertIn('0:05–0:10≈', out)

    def test_environment_and_explicit_state_roots(self):
        args = S.parser().parse_args([])
        with patch.dict(os.environ, {'RAVEN_EARS_HOME': str(self.root)}, clear=True):
            self.assertEqual(S.state_home(args), self.root / 'listens')
            with patch.dict(os.environ, {'RAVEN_LISTENS_HOME': str(self.home)}):
                self.assertEqual(S.state_home(args), self.home)
                args.state_home = self.root / 'explicit'
                self.assertEqual(S.state_home(args), self.root / 'explicit')

    def test_all_and_stateless_views_write_nothing(self):
        self.assertEqual(self.call('--mode', 'all'), CARD)
        self.assertFalse(self.home.exists())
        self.call('--mode', 'first-listen', '--no-state')
        self.assertFalse(self.home.exists())
        self.assertIn('No open listens', self.call('--status', card=False))
        self.assertFalse(self.home.exists())

    def test_closed_first_listen_retains_notes(self):
        self.call('--mode', 'first-listen', '--section', '6', '--note', 'last thought')
        self.assertIn('Listen closed', self.call('--next'))
        self.assertIn('No open listens', self.call('--status', card=False))
        self.assertIn('new thought', self.call('--edit-note', '1', 'new thought'))


class LocalViewTests(unittest.TestCase):
    def test_tagged_duplicate_merges_origins_but_different_claims_do_not(self):
        card = '''MUSIC: example.mp3 · 1:00
=================== singer
ENTRANCES/EXITS:
  0:08 piano (credited) enters (likely) [instrument tagger]
=================== harmony
HARMONY: example.mp3 · 1:00
TIMELINE
  0:08 piano (credited) enters (likely)
  0:08 piano (credited) enters (possibly)
'''
        out = M.render(card, 'first-listen')
        self.assertEqual(out.count('piano (credited) enters (likely)'), 1)
        for source in ('instrument tagger', 'singer/entrances', 'harmony/timeline'):
            self.assertIn(source, out)
        self.assertIn('enters (possibly)', out)
        self.assertEqual(M.render(card, 'all'), card)

    def test_no_future_map_stats_or_form_explanations(self):
        out = M.render(CARD, 'first-listen')
        for spoiler in ('3:20', 'returning', 'between returning', 'late in the song', 'VERSE', 'CHORUS',
                        '--section 2', 'quiet outro', 'TEMPO:', 'KEY:', 'dynamic range', 'FUTURE_RANGE'):
            self.assertNotIn(spoiler, out)
        self.assertIn('no clear arrival on I', out)

    def test_spans_and_holds_are_clipped_and_aggregate_values_withheld(self):
        events = [M.Event(5, 'piano (possibly) + voice', 'instruments', 90),
                  M.Event(7, 'pedal on C until 1:30', 'harmony', 90),
                  M.Event(8, 'quiet -24 dB', 'groove', 90),
                  M.Event(9, 'I – IV', 'harmony', 90),
                  M.Event(10, 'climbs 0.2→0.9', 'harmony', 90),
                  M.Event(11, 'voice enters SECRET_WORDS', 'words', 90)]
        selected = V.local_events(events, 6, 20)
        self.assertEqual(selected[0].t, 6)
        self.assertEqual(selected[0].end, 20)
        self.assertIn('until 0:20', selected[1].text)
        rendered = '\n'.join(M._event_line(e) for e in selected)
        for hidden in ('1:30', 'SECRET_WORDS', '-24', '0.9', 'I – IV'):
            self.assertNotIn(hidden, rendered)

    def test_mid_line_pause_does_not_guess_a_word_boundary(self):
        out = M.render(CARD, 'first-listen', until=17)
        self.assertNotIn('I place a sample tray', out)
        self.assertIn('word timings are unavailable', out)

    def test_stateless_from_uses_the_containing_section(self):
        out = M.render(CARD, 'first-listen', start=47)
        self.assertIn('section 2 · 0:47–1:00', out)
        self.assertNotIn('[0:30]', out)

    def test_run_status_is_specific_without_diagnostic_quotes(self):
        card = M.parse_card('MUSIC: example.mp3 · 1:00\n  NOTE: lyrics FAILED at 0:50 SECRET\n')
        analysis = {'notes': ['credits skipped SECRET', 'no GPU so stage skipped'],
                    'ache': {'status': 'unavailable', 'notes': ['error SECRET']}}
        text = '\n'.join(V.run_status(card, analysis))
        for claim in ('lyrics failed/unavailable', 'credits lookup skipped', 'GPU stage skipped', 'ache stage failed/unavailable'):
            self.assertIn(claim, text)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('0:50', text)
        self.assertNotIn('all represented stages ran', text)

    def test_missing_engine_never_reads_as_silence(self):
        raw = '=================== music\nMUSIC: example.mp3 · 1:00\n=================== singer\nRuntimeError: backend broke\n'
        out = M.render(raw, 'vibe')
        self.assertIn('singer engine skipped or produced no card', out)
        self.assertIn('harmony engine skipped or produced no card', out)
        self.assertIn('missing detail is not silence', out)

    def test_whisper_words_and_unrecorded_credits_are_honest(self):
        raw = 'MUSIC: example.mp3 · 1:00\n  LYRICS: 1 line (whisper large-v3, rough times)\n    [0:05] A small invented line\n'
        for mode, lens in [('first-listen', None), ('vibe', None), ('lens', 'words')]:
            out = M.render(raw, mode, lens)
            self.assertIn("words are Whisper's guesses", out)
            self.assertIn('credits lookup not recorded', out)
            self.assertIn('stereo/space, reverb, production tricks, backing harmonies', out)
            self.assertNotIn('backing harmonies, meaning', out)
            self.assertEqual(out.splitlines()[-1], M.INVITATION)

    def test_success_is_limited_to_recorded_stages(self):
        raw = "MUSIC: example.mp3 · 1:00\n  LYRICS: writer's text\n    [0:05] A small invented line\nCREDITS (local file):\n  credited, not clearly heard: flute\n"
        self.assertIn('no failures/skips recorded for the represented stages', M.render(raw, 'vibe'))

    def test_lyrics_with_error_or_skip_words_are_not_backend_failures(self):
        raw = "MUSIC: example.mp3 · 1:00\n  LYRICS: writer's text\n    [0:05] We skipped an error in the sky\n"
        out = M.render(raw, 'first-listen')
        self.assertIn('We skipped an error in the sky', out)
        self.assertNotIn('failure/skip reported', out)
        self.assertNotIn('engine/stage failed', out)

    def test_credit_timeout_unreadable_file_and_empty_lookup_are_distinct(self):
        for note, expected in (('MusicBrainz timed out', 'credits lookup failed/unavailable'),
                               ('credits file unreadable', 'credits lookup failed/unavailable'),
                               ("MusicBrainz has no match for 'UNHEARD_TITLE'", 'credits: none found (MusicBrainz)')):
            raw = f'MUSIC: example.mp3 · 1:00\nCREDITS: {note}; ears only\n'
            out = M.render(raw, 'first-listen')
            self.assertIn(expected, out)
            self.assertNotIn('UNHEARD_TITLE', out)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cards = self.root / 'cards'
        self.cards.mkdir()
        self.home = self.root / 'listens'
        self.song = self.root / 'example.mp3'
        self.song.write_bytes(b'not real audio; never opened')
        self.card = self.cards / 'example.txt'
        self.card.write_text(CARD)
        self.env = patch.dict(os.environ, {'RAVEN_CARDS_HOME': str(self.cards), 'RAVEN_LISTEN_COMMAND': '/neutral/raven-listen'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def call(self, *flags):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            rc = W.main([*flags, '--state-home', str(self.home)])
        self.assertEqual(rc, 0)
        return stdout.getvalue()

    def test_cached_audio_and_direct_cards_need_no_subprocess(self):
        before = self.card.read_bytes()
        with patch.object(W.subprocess, 'run', side_effect=AssertionError('no engine/network')):
            out = self.call(str(self.song), '--mode', 'first-listen', '--note', 'hello')
            self.assertIn('FIRST LISTEN', out)
            self.assertIn('hello', self.call('--resume'))
            self.assertIn('LENS: groove', self.call(str(self.card), '--mode', 'lens', '--lens', 'groove'))
        self.assertEqual(self.card.read_bytes(), before)

    def test_all_audio_delegates_exactly_to_original(self):
        old = b'NOTE: lyrics: names list fed to Whisper (Zyxwvut)\r\n'
        result = subprocess.CompletedProcess([], 0, stdout=old, stderr=old)
        stdout, stderr = io.BytesIO(), io.BytesIO()
        with patch.object(W.subprocess, 'run', return_value=result) as run, \
             patch('sys.stdout') as out, patch('sys.stderr') as err:
            out.buffer, err.buffer = stdout, stderr
            self.assertEqual(W.main([str(self.song), '--mode', 'all', '--no-credits']), 0)
        self.assertEqual(run.call_args.args[0], ['/neutral/raven-listen', str(self.song), '--no-credits'])
        self.assertEqual(run.call_args.kwargs, {'capture_output': True})
        expected = old.replace(b'Zyxwvut', b'count unavailable')
        self.assertEqual(stdout.getvalue(), expected)
        self.assertEqual(stderr.getvalue(), expected)
        self.assertFalse(self.home.exists())

    def test_filtered_cache_miss_keeps_full_card_off_stdout(self):
        self.card.unlink()
        def generate(argv, **kwargs):
            self.card.write_text(CARD)
            return subprocess.CompletedProcess(argv, 0, stdout=CARD.encode(), stderr=b'')
        with patch.object(W.subprocess, 'run', side_effect=generate) as run:
            out = self.call(str(self.song), '--mode', 'first-listen', '--note', 'at the start')
        self.assertTrue(run.call_args.kwargs['capture_output'])
        self.assertNotIn('silver ladder', out)
        self.assertNotIn('Old invitation', out)
        self.assertIn('at the start', out)
        self.assertEqual(self.card.read_text(), CARD)

    def test_backend_failure_creates_no_session(self):
        self.card.unlink()
        result = subprocess.CompletedProcess([], 1, stdout=b'', stderr=b'failed')
        with patch.object(W.subprocess, 'run', return_value=result), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.call(str(self.song), '--mode', 'first-listen')
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.home.exists())

    def test_executable_wrapper_reads_card_with_exact_bytes(self):
        self.card.write_bytes(b'neutral\r\nno trailing newline')
        executable = Path(W.__file__).parent / 'bin/raven-listen-modes'
        result = subprocess.run([str(executable), str(self.card), '--mode', 'all'], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, self.card.read_bytes())


def run_tests(check):
    class Result(unittest.TestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            check('listening sessions: ' + test._testMethodName.removeprefix('test_'), True)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            check('listening sessions: ' + test._testMethodName, False, self._exc_info_to_string(err, test))

        addError = addFailure
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case)
                               for case in (SessionTests, LocalViewTests, WrapperTests))
    suite.run(Result())


if __name__ == '__main__':
    unittest.main()
