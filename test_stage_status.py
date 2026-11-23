"""Honest-card contracts: synthetic outcomes and mocked backend failures only."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import stage_status as S


class StatusTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ('RAVEN_EARS_PROFILE', 'CUDA_VISIBLE_DEVICES', 'RAVEN_EARS_OFFLINE',
                     'WHISPER_MODEL', 'RAVEN_EARS_DEVICE', 'RAVEN_NO_LYRICS_LOOKUP'):
            os.environ.pop(name, None)

    def test_states_and_no_detection(self):
        for state in S.STATES:
            self.assertEqual(S.outcome('probe', state)['status'], state)
        with self.assertRaises(ValueError):
            S.outcome('probe', 'silent')
        self.assertEqual(S.summary([S.outcome('events')]), 'all stages ran')
        self.assertNotIn('all stages ran', S.summary([]))

    def test_example_summary(self):
        rows = [S.outcome('words', reason="Whisper's guesses", kind='input'),
                S.outcome('LRCLIB', 'not-found', 'no match', kind='input'),
                S.outcome('lyrics file', 'not-found', 'no lyrics file', kind='input'),
                S.outcome('credits', 'not-found', 'no match', source='MusicBrainz', kind='input'),
                S.outcome('credits', 'not-found', 'no match', source='Discogs', kind='input'),
                S.outcome('chords')]
        self.assertEqual(S.summary(rows), "words: Whisper's guesses (LRCLIB: no match; no lyrics file) · credits: none found (MusicBrainz, Discogs) · all audio stages ran")

    def test_scope_retries_do_not_claim_success_or_leak(self):
        @S.stage('probe')
        def probe():
            S.mark('probe', 'degraded', 'CPU fallback')
            return {}
        @S.engine
        def run():
            probe()
            return {}
        first = run()
        self.assertEqual([r['status'] for r in first['stage_status']], ['degraded'])
        self.assertEqual(S.snapshot(), [])
        first['stage_status'].clear()
        self.assertEqual(run()['stage_status'][0]['status'], 'degraded')

    def test_stage_does_not_mutate_saved_results(self):
        original = {'status': 'ok', 'moments': [], 'notes': []}
        import ache
        before = copy.deepcopy(original)
        ache.analyze({'ache': original})
        self.assertEqual(original, before)

    def test_nested_failures_reach_parent(self):
        @S.stage('events')
        def broken():
            raise RuntimeError('weights missing')
        @S.engine
        def child():
            broken()
        @S.engine
        def parent():
            try:
                child()
            except RuntimeError:
                pass
            return {}
        self.assertEqual(parent()['stage_status'][0]['status'], 'failed')

    def test_fatal_cli_text_and_json(self):
        @S.engine
        @S.cli
        def command(argv):
            S.mark('decode')
            raise RuntimeError('backend unavailable')
        for argv in ([], ['--json']):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(command(argv), 1)
            text = buf.getvalue()
            self.assertIn('backend unavailable', text)
            if argv:
                data = json.loads(text)
                self.assertEqual(data['listener'], 'and now, you')
                self.assertEqual(data['stage_status'][-1]['status'], 'failed')
            else:
                self.assertTrue(text.endswith('and now, you\n'))

    def test_notes_only_change_is_new_line(self):
        lines = ['MUSIC: example.wav · 1:00', '  KEY: unclear', 'and now, you']
        out = S.add_notes(lines, [S.outcome('decode')])
        self.assertEqual(out, lines[:-1] + ['NOTES: all stages ran', 'and now, you'])
        self.assertEqual(lines[-1], 'and now, you')

    def test_lrclib_outcomes(self):
        import lrclib as L
        for state, want in [('nomatch', 'not-found'), ('empty', 'not-found'), ('failed', 'failed'),
                            ('synced', 'ran'), ('plain', 'ran'), ('instrumental', 'ran')]:
            with patch.object(L.credits, 'identify', return_value=('Example', 'Song', 'tags')), \
                 patch.object(L, '_lookup', return_value={'source': 'LRCLIB', 'status': state}):
                value = L.get('neutral.wav', use_cache=False)
            self.assertEqual(value['stage_status'][0]['status'], want)
        self.assertEqual(L.get('neutral.wav', off=True)['stage_status'][0]['status'], 'skipped')
        with patch.object(L.credits, 'identify', side_effect=RuntimeError('tags broken')):
            self.assertEqual(L.get('neutral.wav', use_cache=False)['stage_status'][0]['status'], 'failed')
        self.assertEqual(S.lookup_rows('LRCLIB', {'status': 'plain', 'issues': ['search timed out']})[0]['status'], 'degraded')

    def test_resolve_misses_and_whisper_failure(self):
        import lrclib as L
        # Use the actual decorated get; never touch HTTP or a real cache.
        with patch.dict(L.CONFIG, {'off': False}, clear=True), \
             patch.object(L.credits, 'identify', return_value=('Example', 'Song', 'tags')), \
             patch.object(L, '_lookup', return_value={'source': 'LRCLIB', 'status': 'nomatch'}):
            good = L.resolve('neutral.wav', [], 10, word_loader=lambda *_: [('hello', 1, 2)], use_cache=False)
            bad = L.resolve('neutral.wav', [], 10, word_loader=lambda *_: (_ for _ in ()).throw(RuntimeError('model broken')), use_cache=False)
        self.assertIn('LRCLIB: no match; no lyrics file', S.summary(good['stage_status']))
        self.assertIn('Whisper words unavailable', S.summary(bad['stage_status']))
        self.assertNotIn('all audio stages ran', S.summary(bad['stage_status']))

    def test_transcription_empty_is_not_broken_and_retries_degrade(self):
        self.assertEqual(S.word_rows([], [])[-1]['status'], 'ran')
        self.assertEqual(S.word_rows(None, [])[-1]['status'], 'failed')
        rows = S.word_rows([], ['LYRICS WARNING: CUDA FAILED to load', 'lyrics: whisper on cpu (int8), no VAD'])
        self.assertTrue(any(r['status'] == 'degraded' for r in rows))
        self.assertFalse(any(r['status'] == 'failed' for r in rows))

    def test_credits_sources_keep_misses_skips_and_errors(self):
        import credits as C
        for dc, want in [({'status': 'nomatch'}, 'not-found'), ({'status': 'failed', 'detail': 'timed out'}, 'failed'),
                         ({'status': 'skipped', 'detail': 'no token'}, 'skipped')]:
            answer = dict(entries=[], found=False, mb={'status': 'nomatch'}, dc=dc)
            with patch.object(C, 'identify', return_value=('Example', 'Song', 'tags')), patch.object(C, 'chain', return_value=answer):
                value = C.get('neutral.wav', use_cache=False)
            self.assertEqual([r['status'] for r in value['stage_status']], ['not-found', want])
        self.assertEqual(C.get('neutral.wav', off=True)['stage_status'][0]['status'], 'skipped')
        self.assertEqual(C.get('neutral.wav', credits_file='/nonexistent/neutral.txt')['stage_status'][0]['status'], 'failed')

    def test_credit_apply_failure_and_memo_status(self):
        import credits as C
        value = {'entries': [], 'stage_status': [S.outcome('credits', 'not-found', source='MusicBrainz', kind='input')]}
        @S.engine
        def run():
            C.for_song('neutral.wav')
            return {}
        with patch.dict(C._MEMO, {'neutral.wav': value}, clear=True):
            self.assertEqual(run()['stage_status'], value['stage_status'])
        @S.engine
        def broken():
            with patch.object(C, 'for_song', side_effect=RuntimeError('backend broken')):
                C.apply_for('neutral.wav', {})
            return {}
        self.assertEqual(broken()['stage_status'][0]['status'], 'failed')

    def test_instrument_missing_stems(self):
        import instruments as I
        with patch.object(I, 'analyse_raw', return_value={'error': 'no cached stems'}):
            value = I.analyse('neutral.wav')
        self.assertEqual(value['stage_status'][-1]['status'], 'skipped')
        self.assertIn('no cached stems', S.summary(value['stage_status']))

    def test_cnn_missing_broken_and_malformed(self):
        import keyfuse as K
        @S.engine
        def run():
            K.cnn_probs('neutral.wav')
            return {}
        for response, reason in [(RuntimeError('worker broken'), 'worker broken'),
                                 (SimpleNamespace(stdout='{"probs": [1]}'), 'expected 24')]:
            with patch.object(K, 'madmom_python', return_value='/nonexistent/python'), \
                 patch.object(K.subprocess, 'run', side_effect=response if isinstance(response, Exception) else None,
                              return_value=response):
                rows = run()['stage_status']
            self.assertIn(reason, S.summary(rows))
            self.assertTrue(any(r['status'] == 'failed' for r in rows))
        with patch.object(K, 'madmom_python', return_value=None):
            self.assertIn('venv-madmom not installed', S.summary(run()['stage_status']))

    def test_music_missing_key_and_meter(self):
        import music as M
        import singer
        import harmony
        @S.engine
        def run():
            with patch.object(singer, 'get_stems', side_effect=RuntimeError('stems broken')), \
                 patch.object(harmony, 'key_claim_for', side_effect=RuntimeError('key broken')):
                M.meter_info_for('neutral.wav')
                M.key_claim_for('neutral.wav')
            return {}
        rows = run()['stage_status']
        self.assertIn('stems broken', S.summary(rows))
        self.assertIn('key broken', S.summary(rows))
        self.assertFalse(any(r['status'] == 'ran' for r in rows))

    def _fake_speech_stack(self, stack):
        import raven_ears_plus as R
        if True:
            fake = dict(load_audio=np.ones(64000, dtype=np.float32), _torch=None, _front_end={},
                        analyze_pitch={'phrase_ends': []}, analyze_gaps={}, analyze_shape={}, analyze_peak={},
                        analyze_emphasis={}, analyze_quality={'ok': True}, _record={},
                        pitch_line='  PITCH: measured', shape_line='  SHAPE: measured',
                        quality_line='  QUALITY: measured', peak_line='  PEAK: measured',
                        gap_lines=[], words_lines=[], emphasis_line='  EMPHASIS: measured',
                        _source_sample_rate=44100, analyze_channel={'classification': None, 'rolloff_99_hz': 16000.0}, channel_line=None)
            stack.enter_context(patch.object(R.os.path, 'exists', return_value=True))
            for name, value in fake.items():
                stack.enter_context(patch.object(R, name, return_value=value))
            for name in ('release_models', '_note_vram'):
                stack.enter_context(patch.object(R, name))
            return R

    def test_missing_setup_puts_the_remedy_first_on_every_skip_line(self):
        import raven_ears_plus as R
        err = R.SetupNeeded('run --setup', 'PANNs checkpoint missing', detail='/x/Cnn14.pth')
        with ExitStack() as stack:
            self._fake_speech_stack(stack)
            stack.enter_context(patch.object(R, '_whisper_words', return_value=[('hello', .1, .4), ('there', .5, .9)]))
            stack.enter_context(patch.object(R, 'analyze_events', side_effect=err))
            stack.enter_context(patch('room_laugh.detect', side_effect=err))
            value = R.analyze('neutral.wav', update_baseline=False, affect=False)
        card = '\n'.join(value['lines'])
        self.assertIn('  LAUGHS:  skipped: run --setup (PANNs checkpoint missing)', value['lines'])
        self.assertIn('  EVENTS:  skipped: run --setup (PANNs checkpoint missing)', value['lines'])
        self.assertNotIn('FileNotFoundError', card)
        self.assertNotIn('Cnn14', card)
        self.assertTrue(all(len(l) <= 118 for l in value['lines']))
        notes = [l for l in value['lines'] if l.startswith('NOTES')]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].count('run --setup'), 1)  # said once, not once per stage

    def test_quick_singer_word_stages_are_deliberate_skips(self):
        import singer
        import raven_ears_plus as R
        with ExitStack() as stack:
            self._fake_speech_stack(stack)
            stack.enter_context(patch.object(singer, '_free_gpu'))
            whisper = stack.enter_context(patch.object(R, '_whisper_words', side_effect=AssertionError('no Whisper')))
            stack.enter_context(patch.object(R, 'analyze_events', return_value={}))
            stack.enter_context(patch.object(R, 'analyze_affect', return_value={'ok': False}))
            stack.enter_context(patch('room_laugh.detect', return_value={}))
            _, data, _ = singer.singer_block('neutral.wav', 2, words_skip_reason='skipped in a quick listen')
        whisper.assert_not_called()
        summary = S.summary(data['stage_status'])
        self.assertIn('pace / WORDLESS / emphasis: skipped: skipped in a quick listen', summary)
        self.assertIn('words: skipped: skipped in a quick listen', summary)
        self.assertNotIn('no word times', summary)
        self.assertNotIn('--no-whisper', summary)

    def test_real_missing_checkpoint_is_remedy_first(self):
        import raven_ears_plus as R
        with patch.object(R, 'PANNS_CKPT', Path('/nonexistent/Cnn14_mAP=0.431.pth')), patch.object(R, '_MODELS', {}):
            with self.assertRaises(FileNotFoundError) as caught:
                R._load_tagger('cpu')
        self.assertEqual(str(caught.exception), 'run --setup (PANNs checkpoint missing)')
        self.assertEqual(R._skip_why(caught.exception), 'run --setup (PANNs checkpoint missing)')
        self.assertTrue(R._skip_why(RuntimeError('boom')).startswith('RuntimeError: boom'))

    def test_notes_are_one_unique_line_within_the_budget(self):
        long_path = '/nonexistent/cache/raven-panns/Cnn14_mAP=0.431.pth'
        rows = [S.outcome('events', 'failed', 'FileNotFoundError: PANNs checkpoint missing at ' + long_path),
                S.outcome('events', 'skipped', 'PANNs checkpoint missing at ' + long_path[:40]),
                S.outcome('laughs', 'failed', 'FileNotFoundError: PANNs checkpoint missing at ' + long_path),
                S.outcome('Whisper', 'degraded', 'fell back to cuda int8_float16')]
        rows += [S.outcome('stage%d' % i, 'skipped', 'because reason number %d' % i) for i in range(5)]
        lines = S.add_notes(['  PITCH:   x', 'NOTES  raw digest; +4 more'], rows)
        notes = [l for l in lines if l.startswith('NOTES')]
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].startswith('NOTES: '))
        self.assertLessEqual(len(notes[0]), 118)
        self.assertRegex(notes[0], r'\+\d+ more$')
        self.assertEqual(notes[0].count('events:'), 1)
        self.assertRegex(notes[0], r'^NOTES: (events|laughs): failed')  # failures lead; they survive the cut
        self.assertNotIn('stage4', notes[0])           # the least important entries are the ones counted as +N more
        self.assertEqual(lines[-1], 'and now, you')
        short = S.add_notes([], [S.outcome('affect', 'skipped', '--no-affect')])
        self.assertEqual(short[0], 'NOTES: affect: skipped: --no-affect')

    def test_chosen_cpu_is_not_reported_as_a_fallback(self):
        note = 'words: own faster-whisper large-v3 on cpu (int8)'
        with patch.dict(os.environ, {'RAVEN_EARS_DEVICE': 'cpu'}):
            chosen = S.summary(S.word_rows([('hi', 0, .2)], [note]))
        with patch.dict(os.environ, {'RAVEN_EARS_DEVICE': 'auto'}):
            fallback = S.summary(S.word_rows([('hi', 0, .2)], [note]))
        self.assertNotIn('fell back', chosen)
        self.assertNotIn('degraded', chosen)
        self.assertIn('Whisper: degraded: fell back to CPU int8', fallback)

    def test_speech_failures_and_disabled_stages_on_card(self):
        with ExitStack() as stack:
            R = self._fake_speech_stack(stack)
            stack.enter_context(patch.object(R, '_whisper_words', return_value=None))
            stack.enter_context(patch.object(R, 'analyze_events', side_effect=FileNotFoundError('PANNs weights missing')))
            stack.enter_context(patch.object(R, 'analyze_affect', side_effect=RuntimeError('affect worker broken')))
            stack.enter_context(patch('room_laugh.detect', side_effect=RuntimeError('laugh worker broken')))
            value = R.analyze('neutral.wav', update_baseline=False)
            notes = value['lines'][-2]
            # The card carries ONE NOTES line within the line budget; what it cannot fit is counted
            # ("+N more"), and every reason stays whole in the structured stage_status.
            self.assertEqual(sum(l.startswith('NOTES') for l in value['lines']), 1)
            self.assertTrue(notes.startswith('NOTES: '))
            self.assertLessEqual(len(notes), 118)
            self.assertRegex(notes, r'failed|skipped')
            full = S.summary(value['stage_status'])
            for text in ('PANNs weights missing', 'affect worker broken', 'laugh worker broken', 'Whisper'):
                self.assertIn(text, full)
            off = R.analyze('neutral.wav', use_whisper=False, events=False, affect=False, laughs=False)
            for flag in ('--no-events', '--no-affect', '--no-laughs', '--no-whisper'):
                self.assertIn(flag, S.summary(off['stage_status']))
            self.assertEqual(sum(l.startswith('NOTES') for l in off['lines']), 1)
            self.assertLessEqual(len(off['lines'][-2]), 118)
            self.assertEqual(off['lines'][-1], 'and now, you')
            self.assertEqual(off['stage_status'], off['data']['stage_status'])

    def test_speech_cpu_retry_has_structured_degradation(self):
        import raven_ears_plus as R
        @S.engine
        def run():
            def worker(device):
                if device == 'cuda':
                    raise RuntimeError('GPU busy')
                return []
            with patch.object(R, '_pick_device', return_value='cuda'), patch.object(R, 'release_models'):
                R._with_fallback('events', worker, [])
            return {}
        self.assertIn('retried on CPU', S.summary(run()['stage_status']))

    def test_views_keep_precise_notes_and_listener(self):
        import music
        import slim
        import modes
        rows = [S.outcome('events', 'failed', 'PANNs weights missing'),
                S.outcome('LRCLIB', 'not-found', 'no match', kind='input')]
        raw = music.format_card({'file': 'neutral.wav', 'duration': 20, 'stage_status': rows})
        for view in [slim.render(raw)] + [modes.render(raw, mode=m, lens='words' if m == 'lens' else None)
                                        for m in ('vibe', 'lens', 'first-listen')]:
            self.assertIn('DIDN\'T HEAR:', view)
            self.assertIn('PANNs weights missing', view)
            self.assertIn('LRCLIB: no match', view)
            self.assertEqual(view.splitlines()[-1], 'and now, you')
        self.assertEqual(modes.render(raw, mode='all'), raw)
        self.assertNotIn('all stages ran', slim.render('NOTES: all stages ran\n' + raw).split("DIDN'T HEAR:")[1])

    def test_omission_counts_need_matching_structured_diagnostics(self):
        import slim
        import modes
        rows = [S.outcome(stage, 'failed', 'synthetic backend unavailable; ' + 'x' * 45)
                for stage in ('events', 'affect', 'quality')]
        # Older cards have only the bounded digest, without the new STATUS rows.
        raw = 'MUSIC: invented.wav · 1:00\n' + S.notes_line(S.summary_items(rows))
        self.assertIn('+2 more', raw)
        for analysis in (None, {'stage_status': rows[:1]}, {'stage_status': [S.outcome('decode')]}):
            self.assertIn('2 more status entries omitted', ' · '.join(S.card_notes(raw, analysis)))
            views = [slim.render(raw, analysis=analysis)] + [
                modes.render(raw, mode=mode, lens='words' if mode == 'lens' else None, analysis=analysis)
                for mode in ('vibe', 'lens', 'first-listen')]
            for view in views:
                self.assertIn('2 more status entries omitted', view)
        recovered = S.card_notes(raw, {'stage_status': rows})
        self.assertNotIn('omitted', ' · '.join(recovered))
        for stage in ('events', 'affect', 'quality'):
            self.assertIn(stage + ': failed', ' · '.join(recovered))
        current = '\n'.join(S.add_notes(['MUSIC: invented.wav · 1:00'], rows))
        for stage in ('events', 'affect', 'quality'):
            self.assertIn(stage + ': failed', slim.render(current))
        skips = [S.outcome(stage, 'skipped', 'disabled at user request; ' + 'x' * 45)
                 for stage in ('events', 'affect', 'quality')]
        raw = 'MUSIC: invented.wav · 1:00\n' + S.notes_line(S.summary_items(skips))
        self.assertIn('+2 more', raw)
        for analysis in (None, {'stage_status': skips[:1]}):
            view = slim.render(raw, analysis=analysis)
            self.assertIn('events: skipped', view)
            self.assertIn('2 more status entries omitted', view)

    def test_long_skip_diagnostics_fit_complete_card_lines(self):
        with ExitStack() as stack:
            R = self._fake_speech_stack(stack)
            for stage in ('events', 'affect', 'quality'):
                stack.enter_context(patch.object(R, 'analyze_' + stage, side_effect=RuntimeError('x' * 200)))
            stack.enter_context(patch('room_laugh.detect', side_effect=RuntimeError('x' * 200)))
            value = R.analyze('neutral.wav', words=[('paper', .1, .9)], use_whisper=False,
                              update_baseline=False)
        for line in value['lines']:
            self.assertLessEqual(len(line), R.MAX_LINE, line)
        self.assertIn('EVENTS:  skipped: RuntimeError:', '\n'.join(value['lines']))
        self.assertIn('events: failed', S.summary(value['stage_status']))

    def test_older_card_misses_are_named_without_inventing_stage_success(self):
        import slim
        import modes
        raw = "MUSIC: neutral.wav · 0:20\nLYRICS: LRCLIB had no matching lyrics; words are Whisper's guesses (rough times)\nCREDITS: MusicBrainz has no match for 'Example - Song'; Discogs has no match; ears only\n"
        for output in (slim.render(raw), modes.render(raw, mode='vibe')):
            self.assertIn('LRCLIB: no match', output)
            self.assertIn('credits: none found (MusicBrainz, Discogs)', output)
            self.assertIn('audio stage status not recorded (older card)', output)
            self.assertNotIn('all stages ran', output)

    def test_skipped_transcription_is_not_a_failed_whisper(self):
        import lrclib as L
        with patch.object(L, 'get', return_value={'status': 'nomatch'}), \
             patch.object(L, 'cached_activity', return_value=None):
            loader = unittest.mock.Mock(side_effect=AssertionError('must not transcribe'))
            value = L.resolve('neutral.wav', [], word_loader=loader, words_skip_reason='no vocal evidence')
        loader.assert_not_called()
        self.assertIn('words: skipped: no vocal evidence', S.summary(value['stage_status']))
        self.assertFalse(any(r['status'] == 'failed' for r in value['stage_status']))

    def test_harmony_lyric_rows_are_data_not_rendered_card_lines(self):
        import harmony as H
        import sys
        F = {'y_len': H.SR * 20, 'rms_db': np.array([-20., -10.])}
        lines = [{'t': 1, 'end': 2, 'text': 'a paper boat'}]
        mel = {'f0': [], 'vp': [], 'hop': H.HOP, 'source': 'full mix'}
        fake = dict(cache_path='/tmp/unused-honest-test', compute_features=F,
                    gather_key_inputs={'mel': mel, 'downbeats': [1, 2, 3, 4]},
                    resolve_home=({'changes': [{'at': 0, 'to': 'G major'}, {'at': 30, 'to': 'C major'}]},
                                  (0, 'major'), [], 90, [], False, {'key': None}),
                    sections_from_cache=[], key_spans=[], tension_curve={}, builds_releases=[],
                    find_dips=[], find_dropouts=[], find_swells=[], timbre_entrances=([], []),
                    notable_moves=[], find_pedals=[], appoggiaturas=[], self_sim_novelty=([], []),
                    chord_surprises=[], music_lift_frames=([], {}), find_cadences=[], phrase_ends=[],
                    contrast=[], repeated_phrase_anomaly=[], refrain_check={})
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'librosa': SimpleNamespace()}))
            for name, result in fake.items():
                stack.enter_context(patch.object(H, name, return_value=result))
            stack.enter_context(patch('lrclib.for_music', return_value={'words': [], 'lookup': {'status': 'file'},
                               'rows': lines, 'summary': {}, 'label': 'LYRICS: supplied'}))
            value = H.run('neutral.wav', use_cache=False)
        self.assertEqual(value['lines'], lines)
        self.assertNotIn('NOTES:', str(value['lines']))
        self.assertNotIn('changes', value['kinfo'])

    def test_wrap_note_bounds_extreme_indentation(self):
        for width in (0, 4, 40, 115, 300):
            lines = S.wrap_note(' ' * width + 'NOTE: invented diagnostic ' + 'word ' * 60)
            self.assertTrue(all(len(line) <= S.NOTES_WIDTH for line in lines), (width, lines))
            self.assertIn('NOTE: invented diagnostic', lines[0])

    def test_malformed_saved_ache_is_a_structured_failure(self):
        import ache
        @S.engine
        def run():
            ache.block({'ache': {'moments': None}})
            return {}
        self.assertIn('ache: failed', S.summary(run()['stage_status']))

    def test_ache_failures_and_more_ears_probes(self):
        import ache
        import more_ears
        value, failed = more_ears.run('/nonexistent/neutral.wav')
        self.assertTrue(failed)
        self.assertTrue(any(r['status'] == 'failed' for r in value['stage_status']))
        self.assertIn('NOTES:', more_ears.card(value))
        @S.engine
        def broken():
            ache.block({'evs': [{}]})
            return {}
        self.assertIn('ache: failed', S.summary(broken()['stage_status']))
        self.assertEqual(ache.analyze({'evs': []})['stage_status'][0]['status'], 'degraded')


def run_checks(check, tmp=None):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StatusTests)
    for test in suite:
        result = unittest.TestResult()
        test.run(result)
        check('honest card: ' + test._testMethodName.removeprefix('test_').replace('_', ' '),
              result.wasSuccessful(), '\n'.join(message for _, message in result.errors + result.failures))


if __name__ == '__main__':
    unittest.main()
