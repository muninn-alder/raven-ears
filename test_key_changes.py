"""Invented CNN rows in the saved fixed20 bench shape; no real audio/probabilities."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import key_changes as K


def row(start, key='C major', p=.96):
    probs = [(1 - p - .02) / 22] * 24
    index = K.KEYS.index(key)
    probs[index], probs[(index + 1) % 24] = p, .02
    return K.probability_row(start, start + 20, probs)


def run(start, key='C major', p=.96):
    return [row(start + i * 5, key, p) for i in range(3)]


def fixture():
    # Saved rows have leading crop bounds, top/second, p, margin and a 24-vector.
    return run(0) + [dict(row(15), p=.7)] + run(20, 'D major')


def engine_result():
    scan = K.analyze_rows(run(0) + run(20, 'A minor'))
    scan['claims'] = [dict(from_key='C major', to='Eb major', move='up a minor third',
                           at=72., range=[69., 75.], source_support=[62., 69.],
                           destination_support=[75., 90.], method='late signature')]
    return dict(path='invented.mp3', dur=90, key_verdict=dict(key='C major', line='C major', working='C major'),
                evs=[], spans=[(0, (0, 'major'))], tc=dict(step=1, T=[]), br=[], cads=[], con=[], notes=[],
                key_change_scan=scan)


def alternating(plan, p=.8):
    """Rows for a song whose sections lean different ways: plan = [(key, crops in a row), ...]."""
    rows, start = [], 0
    for key, n in plan:
        for _ in range(n):
            rows.append(row(start, key, p))
            start += 5
    return rows


class TonicAmbiguityTests(unittest.TestCase):
    LUCY = [('A major', 5), ('D major', 5), ('A major', 3), ('D major', 4), ('A major', 5), ('D major', 7)]

    def test_sections_that_prefer_a_fifth_apart_split_the_home(self):
        split = K.analyze_rows(alternating(self.LUCY))['section_tonics']
        self.assertTrue(split['split'])
        self.assertEqual(sorted(split['keys']), ['A major', 'D major'])
        # the two keys use the same notes (A mixolydian is D major's scale): this is not a modulation
        self.assertEqual(K.analyze_rows(alternating(self.LUCY))['claims'], [])

    def test_a_song_that_stays_or_travels_once_is_not_split(self):
        for plan in ([('C major', 12)], [('C major', 8), ('Eb major', 8)], [('C major', 8), ('G major', 8)],
                     [('C major', 6), ('A minor', 6), ('C major', 6)]):
            with self.subTest(plan=plan):
                self.assertFalse(K.analyze_rows(alternating(plan))['section_tonics']['split'])

    def test_unsure_crops_do_not_vote(self):
        self.assertFalse(K.analyze_rows(alternating(self.LUCY, p=.4))['section_tonics']['split'])

    def test_clear_wording_becomes_honest_and_nothing_else_changes(self):
        split = K.analyze_rows(alternating(self.LUCY))['section_tonics']
        said = K.tonic_ambiguity('D major (2\u266f) \u00b7 tonic clear', split)
        self.assertIn('tonic ambiguous between D and A', said)
        self.assertNotIn('tonic clear', said)
        self.assertEqual(K.tonic_ambiguity(said, split), said)
        self.assertEqual(K.tonic_ambiguity('D major \u00b7 tonic fairly clear', None), 'D major \u00b7 tonic fairly clear')
        self.assertEqual(K.tonic_ambiguity('D major \u00b7 tonic clear', dict(split=False)), 'D major \u00b7 tonic clear')

    def test_card_text_and_claim_both_drop_tonic_clear_for_a_split_song(self):
        import keyfuse
        result = K.analyze_rows(alternating(self.LUCY))
        text = 'MUSIC: Invented \u00b7 3:00\n  KEY: D major (2\u266f) \u00b7 tonic clear\n  TEMPO: 96\n  home: D major \u00b7 tonic clear; x\n'
        rendered = K.card_text(text, result)
        self.assertNotIn('tonic clear', rendered)
        self.assertEqual(rendered.count('tonic ambiguous between D and A'), 2)
        self.assertIn('TEMPO: 96', rendered)
        stable = K.analyze_rows(alternating([('C major', 12)]))
        self.assertEqual(K.card_text(text, stable), text)
        claim = dict(state='clear', key='D major', line='D major (2\u266f) \u00b7 tonic clear')
        guarded = keyfuse.apply_section_split(claim, result['section_tonics'])
        self.assertEqual(guarded['key'], 'D major')
        self.assertIn('tonic ambiguous between D and A', guarded['line'])
        self.assertIs(keyfuse.apply_section_split(claim, stable['section_tonics']), claim)


class KeyChangesTests(unittest.TestCase):
    def test_terminal_reprise_splits_a_crop_jump(self):
        reprise = run(80, 'Bb major', .8)
        rows = run(0, 'G major') + run(30, 'Bb major') + run(60, 'G major')
        final = run(115, 'B major')
        unsplit = K.analyze_rows(rows + reprise[:2] + final)
        split = K.analyze_rows(rows + reprise + final)
        self.assertEqual([(c['from_key'], c['to']) for c in split['claims']],
                         [('G major', 'Bb major'), ('Bb major', 'G major'),
                          ('G major', 'Bb major'), ('Bb major', 'B major')])
        self.assertEqual(split['claims'][:2], unsplit['claims'][:2])
        self.assertEqual(split['claims'][2]['range'], [70, 100])
        self.assertEqual(split['claims'][3]['range'], [90, 135])

    def test_harmony_confirms_a_short_terminal_signature_with_more_than_one_chord(self):
        import numpy as np
        import harmony as H

        def features(source, target):
            chroma = np.full((12, round(100 * H.FPS)), .1)
            edge = round(72 * H.FPS)
            for key, window in ((source, slice(0, edge)), (target, slice(edge, None))):
                chroma[list(K._scale(K.canonical(key))), window] = 1
            return {'chroma': chroma, 'rms_db': np.full(chroma.shape[1], -20)}

        def scan(source, target):
            crops = run(40, source) + run(70, target, .8)
            return K.analyze_rows(crops, include_late=True)

        chords = [dict(t0=a, t1=b, root=root, q=quality) for a, b, root, quality in (
            (40, 72, 0, 'maj'), (72, 80, 3, 'maj'), (80, 85, 10, 'maj'),
            (85, 90, 7, 'min'), (90, 100, 3, 'maj'))]
        home = (0, 'major')
        data = features('C major', 'Eb major')
        result, areas = H.late_key_areas(scan('C major', 'Eb major'), {}, data, chords, 100, home)
        self.assertEqual(K.lines(result),
                         ['KEY CHANGES: C major → Eb major, somewhere 1:09–1:15 (up a minor third)'])
        self.assertEqual(H.key_spans(areas, home), [(0., home), (72., (3, 'major'))])
        self.assertNotIn('late_runs', result)
        single_chord = chords[:1] + [dict(t0=72, t1=100, root=3, q='maj')]
        result, _ = H.late_key_areas(scan('C major', 'Eb major'), {}, data, single_chord, 100, home)
        self.assertEqual(K.lines(result), [])
        relative, _ = H.late_key_areas(scan('C major', 'A minor'), {}, features('C major', 'A minor'),
                                     chords, 100, home)
        self.assertEqual(K.lines(relative), [])

    def test_range_includes_whole_last_old_and_first_new_crop(self):
        result = K.analyze_rows(fixture())
        self.assertEqual(len(result['claims']), 1)
        claim = result['claims'][0]
        self.assertEqual((claim['from_key'], claim['to'], claim['move']), ('C major', 'D major', 'up a whole tone'))
        self.assertEqual(claim['range'], [10, 40])
        self.assertEqual(claim['source_support'], [0, 30])
        self.assertEqual(claim['destination_support'], [20, 50])
        self.assertNotIn('at', claim)
        self.assertNotIn('provisional_at', claim)

    def test_frozen_confidence_persistence_union_and_gaps(self):
        self.assertEqual(len(K.analyze_rows(run(0) + run(20, 'D major'))['claims']), 1)
        for broken in (run(20, 'D major')[:2], [row(t, 'D major') for t in (20, 25, 25.5)],
                       [row(t, 'D major') for t in (20, 25, 35)],
                       [row(20, 'D major'), row(25, 'E major'), row(30, 'D major')]):
            self.assertEqual(K.analyze_rows(run(0) + broken)['claims'], [])
        for field, value in (('p', .899), ('margin', .199)):
            broken = run(20, 'D major')
            broken[1][field] = value
            self.assertEqual(K.analyze_rows(run(0) + broken)['claims'], [])
        boundary = run(20, 'D major')
        for r in boundary:
            r.update(p=.90, margin=.20)
        self.assertEqual(len(K.analyze_rows(run(0) + boundary)['claims']), 1)

    def test_allowed_moves_are_exactly_the_frozen_set(self):
        for mode in ('major', 'minor'):
            labels = {K.canonical(k)[0]: k for k in K.KEYS if k.endswith(mode)}
            for source in range(12):
                for step in range(1, 12):
                    a, b = labels[source], labels[(source + step) % 12]
                    result = K.analyze_rows(run(0, a) + run(20, b))
                    self.assertEqual(bool(result['claims']), step in (1, 2, 3, 4, 8, 9), (a, b))
        for target in ('A minor', 'C minor', 'D minor'):
            result = K.analyze_rows(run(0) + run(20, target))
            self.assertEqual(result['claims'], [])
            self.assertEqual(len(result['withheld']), 1)

    def test_withheld_and_same_key_returns_update_the_current_support(self):
        result = K.analyze_rows(run(0) + run(20, 'A minor') + run(40, 'B minor'))
        self.assertEqual(result['claims'][0]['from_key'], 'A minor')
        result = K.analyze_rows(run(0) + run(60) + run(80, 'D major') + run(100))
        self.assertEqual(result['claims'][0]['range'], [70, 100])
        # Down a whole tone is withheld, even on a return to the initial key.
        self.assertEqual(len(result['claims']), 1)
        self.assertEqual(result['withheld'][0]['range'], [90, 120])

    def test_global_oracle_rows_never_enter_the_rule(self):
        rows = fixture() + [dict(row(60, 'E major'), kind='global')]
        self.assertEqual(K.analyze_rows(rows), K.analyze_rows(fixture()))
        for bad in (float('nan'), float('inf')):
            rows = fixture()
            rows[0]['p'] = bad
            with self.assertRaises(ValueError):
                K.analyze_rows(rows)

    def test_window_geometry_includes_only_full_crops_and_last_full_window(self):
        self.assertEqual(K.windows(19.9), [])
        self.assertEqual(K.windows(20), [(0, 20)])
        self.assertEqual(K.windows(31.25), [(0, 20), (5, 25), (10, 30), (11.25, 31.25)])
        for d in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                K.windows(d)

    def test_no_line_for_absence_failure_or_withheld_moves(self):
        for result in (None, {}, dict(status='failed', claims=[{}]), K.analyze_rows(run(0)),
                       K.analyze_rows(run(0) + run(20, 'A minor'))):
            self.assertEqual(K.lines(result), [])
            self.assertEqual(K.card_text('KEY: C major\n', result), 'KEY: C major\n')

    def test_display_preserves_key_note_once_and_rounds_outwards(self):
        result = K.analyze_rows(fixture())
        result['claims'][0]['range'] = [10.9, 40.1]
        import slim
        for indent in ('', '  '):
            with self.subTest(indent=indent):
                text = f'MUSIC: Invented · 1:00\n{indent}KEY: C major · tonic clear\n{indent}TEMPO: 120\n'
                rendered = K.card_text(text, result)
                self.assertIn('somewhere 0:10–0:41 (up a whole tone)', rendered)
                self.assertEqual(rendered.splitlines()[1], text.splitlines()[1])
                self.assertEqual(rendered.splitlines()[2], indent + K.lines(result)[0])
                self.assertEqual(rendered.splitlines()[3], indent + 'NOTE: ' + K.NOTE)
                self.assertEqual(rendered.count(K.NOTE), 1)
                self.assertEqual(K.card_text(rendered, result), rendered)
                legacy_slim = slim.render(rendered)
                self.assertIn(K.lines(result)[0], legacy_slim.splitlines())
                self.assertNotIn(K.NOTE, legacy_slim)

    def test_scan_cpu_caps_cache_and_failure(self):
        def model(command, **kwargs):
            self.assertEqual(command[:4], ['nice', '-n', '10', 'invented-python'])
            self.assertEqual(kwargs['env']['RAVEN_EARS_DEVICE'], 'cpu')
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
                self.assertLessEqual(int(kwargs['env'][name]), 4)
            start = float(command[command.index('--start') + 1])
            return SimpleNamespace(stdout=json.dumps(dict(probs=row(start, 'C major' if start < 15 else 'D major')['probs'])))
        with tempfile.TemporaryDirectory() as directory, \
             patch('keyfuse.madmom_python', return_value='invented-python') as py, \
             patch.object(K.subprocess, 'run', side_effect=model) as process:
            cp = str(Path(directory) / 'invented')
            result = K.scan('invented.wav', 45, cp)
            self.assertEqual(result['status'], 'ran', result)
            self.assertEqual(len(result['claims']), 1)
            self.assertEqual(process.call_count, 6)
            self.assertEqual(K.scan('invented.wav', 45, cp)['claims'], result['claims'])
            self.assertEqual(process.call_count, 6)
            py.return_value = None
            self.assertEqual(K.scan('invented.wav', 45)['status'], 'skipped')
            py.return_value = 'invented-python'
            process.side_effect = [SimpleNamespace(stdout=json.dumps(dict(probs=row(0)['probs']))),
                                   RuntimeError('invented failure')]
            result = K.scan('invented.wav', 45)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['claims'], [])
            with self.assertRaises(ValueError):
                K.probability_row(0, 20, [float('nan')] * 24)

    def test_slim_and_full_keep_all_claims_after_key_and_reject_failed_stage(self):
        import listen
        import slim
        result = K.analyze_rows(fixture())
        text = 'MUSIC: Invented · 1:00\n  KEY: C major · tonic clear\n  TEMPO: 120\n'
        cards = {n: dict(text='', json={}) for n in listen.PASSES}
        cards['music']['text'] = text
        cards['harmony']['json'] = dict(key_change_scan=result)
        statuses = {n: dict(status='ran', reason='') for n in listen.PASSES}
        song = dict(cards=cards, manifest=dict(passes=statuses))
        before = copy.deepcopy(song)
        full, _ = listen.render(song, 'full')
        slim_text = slim.render_first_listen({n: c['text'] for n, c in cards.items()},
                                             {n: c['json'] for n, c in cards.items()}, statuses)
        for rendered in (full, slim_text):
            self.assertIn(K.lines(result)[0], rendered)
            self.assertLess(rendered.index('KEY:'), rendered.index('KEY CHANGES:'))
            self.assertLess(rendered.index('KEY CHANGES:'), rendered.index('TEMPO:'))
        self.assertEqual(full.count(K.NOTE), 1)
        self.assertIn('  ' + K.lines(result)[0], full.splitlines())
        self.assertIn('  NOTE: ' + K.NOTE, full.splitlines())
        self.assertIn(K.lines(result)[0], slim_text.splitlines())
        self.assertNotIn(K.NOTE, slim_text)
        self.assertEqual(song, before)
        saved_texts = {n: c['text'] for n, c in cards.items()}
        saved_texts['harmony'] = K.card_text('HARMONY: Invented\n  KEY: C major\n', result)
        saved_slim = slim.render_first_listen(saved_texts,
                                              {n: c['json'] for n, c in cards.items()}, statuses)
        self.assertIn(K.lines(result)[0], saved_slim.splitlines())
        self.assertNotIn(K.NOTE, saved_slim)
        statuses['harmony']['status'] = 'failed'
        self.assertNotIn('KEY CHANGES:', listen.full_text(song))
        failed = slim.render_first_listen({n: c['text'] for n, c in cards.items()},
                                         {n: c['json'] for n, c in cards.items()}, statuses)
        self.assertNotIn('KEY CHANGES:', failed)

    def test_cli_default_and_opt_out_are_forwarded_only_to_harmony(self):
        import listen
        import harmony
        baseline = listen.parser().parse_args(['invented.wav'])
        args = listen.parser().parse_args(['invented.wav', '--key-changes'])
        for name in listen.PASSES:
            expected = [str(Path('invented.wav').resolve())]
            if name == 'harmony':
                expected += ['--key-changes']
            self.assertEqual(listen.pass_args(name, baseline), expected)
            self.assertEqual(listen.pass_args(name, args), expected)
            off = listen.parser().parse_args(["invented.wav", "--no-key-changes"])
            quick = listen.parser().parse_args(["invented.wav", "--quick", "--key-changes"])
            self.assertEqual(listen.pass_args(name, off),
                             [str(Path("invented.wav").resolve()), *(["--no-key-changes"] if name == "harmony" else [])])
            self.assertNotIn("--key-changes", listen.pass_args(name, quick))
        with patch.object(harmony, 'run', return_value={}) as engine, \
             patch.object(harmony, 'flag_contrast'), patch.object(harmony, 'report', return_value=''), \
             patch('credits.configure_from_args'), patch('lrclib.configure_from_args'), \
             patch('lyrics_shared.configure_for_run'), redirect_stdout(io.StringIO()):
            harmony.main(['invented.wav'])
            self.assertTrue(engine.call_args.kwargs['key_changes'])
            harmony.main(['invented.wav', '--no-key-changes'])
            self.assertFalse(engine.call_args.kwargs['key_changes'])
            harmony.main(['invented.wav', '--quick', '--key-changes'])
            self.assertFalse(engine.call_args.kwargs['key_changes'])
            harmony.main(['invented.wav', '--key-changes'])
            self.assertEqual(engine.call_args.kwargs['key_changes'], True)

    def test_worker_runner_save_scrubber_song_json_round_trip(self):
        import harmony
        import listen
        # Keep the real run_pass, worker, harmony.main, to_json, capture, save
        # and scrubber. Replace only subprocess transport and audio/model work.
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.dict(os.environ, RAVEN_EARS_HOME=str(root)))
            listen.store_home().mkdir()
            result = engine_result()
            def fake_run(*args, **kwargs):
                value = copy.deepcopy(result)
                if not kwargs['key_changes']:
                    value['key_change_scan'] = None
                return value
            engine = stack.enter_context(patch.object(harmony, 'run', side_effect=fake_run))
            stack.enter_context(patch.object(harmony, 'flag_contrast'))
            stack.enter_context(patch.object(harmony, 'build_timeline', return_value=[]))
            stack.enter_context(patch('ache.analyze', return_value=dict(status='ran', moments=[])))
            stack.enter_context(patch('ache.block', return_value=[]))
            for name in ('block_sound', 'block_chords', 'block_tension', 'block_resolves', 'block_timeline',
                         'block_lyric_timeline', 'block_contrast', 'block_surprise', 'block_chills'):
                stack.enter_context(patch.object(harmony, name, return_value=[]))
            def child(command, **kwargs):
                with redirect_stdout(io.StringIO()) as output:
                    code = listen.worker(command[3], command[4], command[5:])
                return SimpleNamespace(returncode=code, stdout=output.getvalue(), stderr='')
            stack.enter_context(patch.object(listen.subprocess, 'run', side_effect=child))
            for flags, enabled in [([], True), (['--no-key-changes'], False),
                                   (['--quick', '--key-changes'], False)]:
                with self.subTest(flags=flags):
                    args = listen.parser().parse_args(['invented.mp3', '--no-lyrics', '--no-credits', *flags])
                    card = listen.run_pass('harmony', args)
                    self.assertEqual(card['status'], 'ran', card['reason'])
                    self.assertEqual(engine.call_args.kwargs['key_changes'], enabled)
                    song = dict(cards=dict(harmony=card))
                    path = root / 'song.json'
                    listen.save(path, song)
                    saved = json.loads(path.read_text())['cards']['harmony']
                    if enabled:
                        expected = result['key_change_scan']
                        self.assertEqual(saved['json']['key_change_scan'], expected)
                        self.assertEqual(expected['withheld'][0]['move'], 'relative major/minor')
                        for line in K.lines(expected):
                            self.assertIn(line, saved['text'])
                        self.assertIn(K.LATE_NOTE, saved['text'])
                    else:
                        self.assertNotIn('key_change_scan', saved['json'])
                        self.assertNotIn('KEY CHANGES:', saved['text'])


def run_tests(check):
    for name in unittest.defaultTestLoader.getTestCaseNames(KeyChangesTests):
        result = unittest.TestResult()
        KeyChangesTests(name).run(result)
        check('key changes: ' + name, result.wasSuccessful(),
              '\n'.join(trace for _, trace in result.failures + result.errors))


if __name__ == '__main__':
    unittest.main()
