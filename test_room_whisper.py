"""Synthetic ROOM/WHISPER checks. No network, recordings or inference models.

run_checks joins --fast using supplied frame masks and mocked event probabilities.
Direct execution also checks the real pyin front end on resonated noise and pulses.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import raven_ears_plus as R
import room_whisper as W


def front(duration=4, voiced=False):
    n = int(duration * R.FPS) + 1
    return dict(times=np.arange(n) / R.FPS, active=np.ones(n, bool),
                voiced=np.full(n, voiced), f0=np.full(n, 140. if voiced else np.nan),
                rms_db=np.full(n, -20.), n=n, thr=-60.)


def context(scores, labels, centers=None, duration=8):
    p = np.asarray(scores, dtype=float)
    centers = np.arange(len(p)) * R.TAG_HOP + 1 if centers is None else centers
    return W.event_context(centers, p, labels, duration, R.TAG_HOP, R.EVENT_LABELS['Whispering'][1])


WORDS = [('paper', .2, .6), ('lantern', .65, 1.1), ('folds', 1.15, 1.55)]


class RoomWhisperTests(unittest.TestCase):
    def test_speech_events_do_not_report_the_measured_inhale_as_a_sneeze(self):
        # Selected CNN14 columns from the measured inhale, in AudioSet index order.
        columns = {0: 'Speech', 41: 'Breathing', 44: 'Gasp', 49: 'Sneeze', 50: 'Sniff'}
        labels = [f'unused-{i}' for i in range(527)]
        for index, label in columns.items():
            labels[index] = label
        centers = np.arange(23., 26.5, .5)
        probs = np.zeros((len(centers), len(labels)), np.float32)
        probs[:, list(columns)] = [
            [.056563, .000876, .002622, .000713, .001058],
            [.059139, .020886, .012249, .145341, .010729],
            [.426832, .025586, .015479, .419560, .005797],
            [.876861, .022267, .058332, .265133, .003575],
            [.799020, .018169, .020750, .397644, .005613],
            [.859100, .000465, .009134, .000463, .000727],
            [.840950, .000371, .006884, .000287, .000957],
        ]
        with patch.object(R, '_tag_windows', return_value=(centers, probs, labels)), \
                patch.object(R, '_pick_device', return_value='cpu'):
            result = R.analyze_events(np.zeros(0), 30., [], front(30))
            overlap = [e for e in result['events'] if e['start'] < 25.5 and e['end'] > 24.]
            self.assertEqual(overlap, [])

            # One window above the sneeze cutoff and a separately supported breath still fire.
            probs[2, 49] = .46
            probs[4, 41] = .31
            result = R.analyze_events(np.zeros(0), 30., [], front(30))
            self.assertEqual([(e['label'], e['start'], e['end']) for e in result['events']],
                             [('Sneeze', 23.75, 24.25), ('Breathing', 24.75, 25.25)])

    def test_room_two_windows_max_not_sum_and_exact_threshold(self):
        self.assertFalse(context([[.9], [.34]], ['Dog'])['room']['hits'])
        self.assertFalse(context([[.2, .2], [.2, .2]], ['Dog', 'Bark'])['room']['hits'])
        result = context([[.35, 0], [0, .35]], ['Dog', 'Bark'])['room']
        self.assertEqual(len(result['hits']), 1)
        self.assertEqual(result['hits'][0]['windows'], 2)
        self.assertEqual(result['hits'][0]['score'], .35)
        self.assertEqual(result['hits'][0]['labels'], ['Dog', 'Bark'])

    def test_room_throughout_is_strictly_over_sixty_percent(self):
        for count, throughout in ((3, False), (4, True)):
            room = context([[.4]] * count + [[0]] * (5 - count), ['Rain'])['room']
            self.assertEqual(room['hits'][0]['throughout'], throughout)
            self.assertIn('0:01-', W.room_line(room))
            self.assertNotIn('throughout', W.room_line(room))

    def test_room_strength_order_cap_and_disjoint_runs(self):
        labels = ['Dog', 'Rain', 'Television', 'Typing', 'Car']
        room = context([[.95, .8, .7, .6, .5]] * 2 + [[0] * 5], labels)['room']
        line = W.room_line(room)
        self.assertLess(line.index('dog'), line.index('rain'))
        self.assertIn('typing', line)
        self.assertNotIn('vehicle/car', line)
        disjoint = context([[.5], [0], [0], [.8]], ['Dog'], centers=[1, 2, 6, 7])['room']
        self.assertEqual(len(disjoint['hits'][0]['spans']), 2)
        self.assertNotIn('0:01–0:07', W.room_line(disjoint))

    def test_music_only_exception_quotes_wordless(self):
        ev = context([[.7], [.7]], ['Music'])
        self.assertFalse(W.room_result(ev, dict(wordless=[(2, 3)]))['hits'])
        self.assertTrue(W.room_result(ev, dict(wordless=[]))['hits'])
        ev = context([[.7, .5], [.7, .5]], ['Music', 'Dog'])
        self.assertEqual(len(W.room_result(ev, dict(wordless=[(2, 3)]))['hits']), 2)
        self.assertEqual(W.room_line(context([[0], [0]], ['Music'])['room']), '  ROOM:    nothing stands out')

    def test_room_dog_family_and_household_classes(self):
        for label in ('Dog', 'Bark', 'Howl', 'Yip', 'Growling', 'Whimper (dog)',
                      'Television', 'Music', 'Cat', 'Door', 'Dishes, pots, and pans',
                      'Water tap, faucet', 'Boiling'):
            with self.subTest(label=label):
                result = context([[.8], [.8], [0]], [label])['room']
                self.assertEqual(len(result['hits']), 1)
                self.assertIn('0:01-', W.room_line(result))
                self.assertNotIn('–', W.room_line(result))
        # AudioSet Speech cannot identify a second person.
        self.assertFalse(context([[.99], [.99]], ['Speech'])['room']['hits'])

    def test_engine_quiet_room_is_one_line(self):
        result, tag = self.analyze_fixture(probs=np.zeros((5, 2)))
        self.assertEqual(tag.call_count, 1)
        self.assertEqual([line for line in result['lines'] if line.startswith('  ROOM:')],
                         ['  ROOM:    nothing stands out'])

    def test_faint_dog_needs_a_clip_relative_score_peak_and_unvoiced_sound(self):
        fe = front(duration=5)
        centers = np.arange(.5, 5., .5)
        y = np.full(5 * R.SR_SED, 1e-4, np.float32)
        y[2 * R.SR_SED:int(2.2 * R.SR_SED)] = .002
        scores = np.full((len(centers), 2), .01)
        scores[3, 0], scores[4, 1] = .12, .13

        def run(audio, probabilities):
            return W.faint_dogs(centers, probabilities, ['Dog', 'Bark'], audio, fe, 5., R.SR_SED, R.TAG_HOP)

        trial = run(y, scores)
        self.assertEqual(trial['score_floor'], .01)
        self.assertEqual(trial['audio_floor_db'], -80.)
        self.assertEqual(trial['threshold'], .08)
        self.assertEqual(trial['spans'], [dict(start=1.75, end=2.75, time=2.5, score=.13)])
        for quiet in (np.zeros_like(y), np.full_like(y, .0004)):
            self.assertFalse(run(quiet, scores)['spans'])
        self.assertFalse(run(y, scores + .10)['spans'])
        fe['voiced'][:] = True
        self.assertFalse(run(y, scores)['spans'])

    def test_voice_reduction_and_overlay_keep_distinct_hits(self):
        fe = front(duration=5)
        fe['voiced'][(fe['times'] >= 1) & (fe['times'] < 2)] = True
        y = np.ones(5 * R.SR_SED, np.float32)
        reduced = W.voice_reduced(y, fe, R.SR_SED)
        self.assertAlmostEqual(reduced[int(1.5 * R.SR_SED)], .18)
        self.assertEqual(reduced[int(2.5 * R.SR_SED)], 1.)
        self.assertTrue(np.all(y == 1.))
        self.assertTrue(np.allclose(np.unique(reduced)[[0, -1]], [.18, 1.]))

        existing = context([[.6], [.7]], ['Dog'])
        def span(a, b, t):
            return dict(start=a, end=b, time=t, score=.2)
        raw = dict(method='raw', spans=[span(1., 1.5, 1.2), span(2., 2.5, 2.2)])
        extra = dict(method='voice-reduced', spans=[span(2.4, 2.8, 2.6), span(3.5, 4.5, 4.)])
        room = W.room_result(existing, {}, [raw, extra])
        loud, faint = room['hits']
        self.assertFalse(loud.get('faint'))
        self.assertEqual([s['method'] for s in faint['spans']], ['raw', 'voice-reduced'])
        self.assertIn('dog, faint (0:02, 0:04)', W.room_line(room))

    def test_frames_only_likely_and_no_words_decline(self):
        result = W.whisper_result(WORDS, front(), None, None, R.GAP_MIN, R.TAG_WIN)
        self.assertEqual(len(result['spans']), 1)
        self.assertEqual(result['spans'][0]['words'], 3)
        self.assertFalse(result['spans'][0]['confirmed'])
        self.assertIn('≈likely whispered; 0% voiced; AS unavailable', W.whisper_line(result))
        self.assertIsNone(W.whisper_line(W.whisper_result(None, front(), None, None, R.GAP_MIN, R.TAG_WIN)))
        self.assertFalse(W.whisper_result(WORDS, front(voiced=True), None, None, R.GAP_MIN, R.TAG_WIN)['spans'])

    def test_unworded_model_supported_frames_and_one_transcribed_word(self):
        events = context([[.83]] * 5, ['Whispering'])
        for words in (None, [], [('paper', .3, .5)]):
            result = W.whisper_result(words, front(), events, None, R.GAP_MIN, R.TAG_WIN, 4)
            self.assertEqual(len(result['spans']), 1)
            span = result['spans'][0]
            self.assertEqual((span['start'], span['end']), (0, 4))
            self.assertAlmostEqual(span['unvoiced_s'], 4)
            self.assertFalse(span['confirmed'])
            self.assertEqual(span['support_windows'], 5)
            line = W.whisper_line(result)
            self.assertIn('≈likely whispered; 4.0s unvoiced; Whispering 5 windows, peak 0.83', line)
            self.assertIn('1 word transcribed' if words else '0 words transcribed', line)
            self.assertLessEqual(len(line), R.MAX_LINE)
            if not words:
                self.assertEqual(W.whisper_tokens(result, words), [])
            json.dumps(result)

    def test_frame_and_word_candidates_merge_and_keep_inline_markers(self):
        events = context([[.83]] * 5, ['Whispering'])
        result = W.whisper_result(WORDS, front(), events, None, R.GAP_MIN, R.TAG_WIN, 4)
        self.assertEqual(len(result['spans']), 1)
        self.assertEqual(result['spans'][0]['indices'], [0, 1, 2])
        self.assertEqual(result['spans'][0]['end'], 4)
        self.assertIn('3 words transcribed', W.whisper_line(result))
        self.assertEqual(len(W.whisper_tokens(result, WORDS)), 2)
        # A word run can extend across the edge of a frame run; recheck the union.
        fe = front()
        fe['voiced'][0] = True
        merged = W.whisper_result([('paper', 0, .8)], fe, events, None, R.GAP_MIN, R.TAG_WIN, 4)
        self.assertEqual(len(merged['spans']), 1)
        self.assertEqual(merged['spans'][0]['start'], 0)
        self.assertAlmostEqual(merged['spans'][0]['unvoiced_s'], 4 - 1 / R.FPS)

    def test_frame_route_needs_local_count_share_and_time_support(self):
        for events in (None, context([[.9]] * 5, ['Music']),
                       context([[.39]] * 5, ['Whispering']),
                       context([[.9]], ['Whispering']),
                       context([[.9], [.9]], ['Whispering'], [6, 6.5]),
                       context([[.9], [.9], [0], [0], [0]], ['Whispering']),
                       context([[.9], [.9]], ['Whispering'], [1, 1.5], 10)):
            result = W.whisper_result(None, front(10), events, None, R.GAP_MIN, R.TAG_WIN, 10)
            self.assertIsNone(W.whisper_line(result))
        for events in (context([[.9]], ['Whispering']),
                       context([[.9], [.9], [0], [0], [0]], ['Whispering'])):
            self.assertFalse(W.whisper_result(None, front(), events, None, R.GAP_MIN, R.TAG_WIN, 4)['spans'])
        # No extrapolation across a distant active stretch, even with positive scores.
        fe = front(10)
        fe['active'][fe['times'] < 6] = False
        events = context([[.9]] * 5, ['Whispering'])
        self.assertIsNone(W.whisper_line(W.whisper_result(None, fe, events, None, R.GAP_MIN, R.TAG_WIN, 10)))

    def test_frame_duration_boundary_and_short_bridging_only(self):
        events = context([[.9]] * 5, ['Whispering'])
        for duration, expected in ((1.999, False), (2., True)):
            result = W.whisper_result(None, front(), events, None, R.GAP_MIN, R.TAG_WIN, duration)
            self.assertEqual(bool(result['spans']), expected)
        for mask in ('active', 'voiced'):
            # A one-frame break bridges (Fox's real whisper, 10 Nov 2026: a
            # whisper pauses between words and catches its voice now and then).
            fe = front()
            fe[mask][int(1.9 * R.FPS)] = mask == 'voiced'
            result = W.whisper_result(None, fe, events, None, R.GAP_MIN, R.TAG_WIN, 3.8)
            self.assertTrue(result['spans'])
            # A half-second break does not, so neither side reaches two seconds.
            fe = front()
            a = int(1.5 * R.FPS)
            fe[mask][a:a + int(0.5 * R.FPS)] = mask == 'voiced'
            result = W.whisper_result(None, fe, events, None, R.GAP_MIN, R.TAG_WIN, 3.4)
            self.assertFalse(result['spans'])
        fe = front(1.8)
        self.assertFalse(W.whisper_result(None, fe, events, None, R.GAP_MIN, R.TAG_WIN, 4)['spans'])

    def test_partly_voiced_whisper_is_half_whispered(self):
        # Whisper is a spectrum (Fox, 10 Nov 2026): partly voiced is its own word.
        events = context([[.9]] * 5, ['Whispering'])
        fe = front()
        fe['voiced'][::4] = True
        result = W.whisper_result(None, fe, events, None, R.GAP_MIN, R.TAG_WIN, 4)
        self.assertTrue(result['spans'])
        self.assertEqual(result['spans'][0]['label'], 'likely half-whispered (partly voiced)')
        self.assertIn('≈likely half-whispered', W.whisper_line(result))
        self.assertLessEqual(len(W.whisper_line(result)), R.MAX_LINE)

    def test_unworded_sparse_masks_are_diagnosed_and_switch_still_disables(self):
        @W.ST.engine
        def run(sparse=False):
            fe = front()
            if sparse:
                fe = {key: fe[key][::10] for key in ('times', 'active', 'voiced')}
            data = dict(events=context([[.9]] * 5, ['Whispering']), channel=None,
                        gaps=dict(wordless=None, unvoiced=[(0, 4)], unvoiced_s=4))
            notes = []
            W.enrich(data, fe, None, notes, R.GAP_MIN, R.TAG_WIN, 4)
            return dict(data=data, notes=notes)
        supported = run()
        self.assertTrue(supported['data']['whisper']['spans'])
        self.assertEqual(supported['data']['gaps']['unvoiced_s'], 0)
        sparse = run(True)
        self.assertIsNone(sparse['data']['whisper'])
        self.assertIn('incomplete frame coverage', ' '.join(sparse['notes']))
        self.assertTrue(any(row['stage'] == 'whisper' and row['status'] == 'failed'
                            for row in sparse['stage_status']))
        with patch.dict(os.environ, {'RAVEN_EARS_WHISPER': '0'}):
            disabled = run()
        self.assertIsNone(disabled['data']['whisper'])
        self.assertEqual(disabled['data']['gaps']['unvoiced_s'], 4)
        self.assertTrue(any(row['stage'] == 'whisper' and row['status'] == 'skipped'
                            for row in disabled['stage_status']))

    def test_separate_frame_stretches_stay_separate_and_narrowband_stays_likely(self):
        fe = front(8)
        fe['active'][(fe['times'] >= 3) & (fe['times'] < 5)] = False
        events = context([[.8]] * 13, ['Whispering'], duration=8)
        result = W.whisper_result(None, fe, events, {'classification': 'narrowband'},
                                  R.GAP_MIN, R.TAG_WIN, 8, masked='music')
        self.assertEqual(len(result['spans']), 2)
        self.assertLess(result['spans'][0]['end'], result['spans'][1]['start'])
        self.assertTrue(all(not span['confirmed'] for span in result['spans']))

    def test_merging_cannot_extend_model_support_across_unsupported_words(self):
        fe = front(10)
        # A long low-voicing word run contains one continuous frame candidate.
        fe['voiced'][int(3 * R.FPS)::int(R.FPS)] = True
        words = [('paper', i, i + .8) for i in range(10)]
        events = context([[.9]] * 5, ['Whispering'])
        result = W.whisper_result(words, fe, events, None, R.GAP_MIN, R.TAG_WIN, 10)
        self.assertEqual(len(result['spans']), 1)
        self.assertEqual(result['spans'][0]['end'], 9.8)
        self.assertFalse(result['spans'][0]['confirmed'])
        self.assertNotIn('unvoiced_s', result['spans'][0])
        self.assertLess(result['spans'][0]['support_time_share'], .6)

    def test_active_voiced_duration_and_gap_gates(self):
        for kind in ('silent', 'twenty_percent', 'short', 'pause', 'outside_audio', 'partly_outside'):
            fe, words = front(), WORDS
            if kind == 'silent':
                fe['active'][:] = False
            elif kind == 'twenty_percent':
                # Every fifth frame voiced => >=20% in these exact frame-aligned words.
                words = [('paper', 0, .8), ('lantern', .8, 1.6)]
                fe['voiced'][::5] = True
            elif kind == 'short':
                words = [('paper', .2, .7)]
            elif kind == 'outside_audio':
                words = [('paper', 5, 6)]
            elif kind == 'partly_outside':
                words = [('paper', 3.8, 5)]
            else:
                words = [('paper', .2, .6), ('lantern', 1.1, 1.5)]
            self.assertFalse(W.whisper_result(words, fe, None, None, R.GAP_MIN, R.TAG_WIN)['spans'], kind)
        result = W.whisper_result([('paper', 0, .6)], front(), None, None, R.GAP_MIN, R.TAG_WIN)
        self.assertEqual(len(result['spans']), 1)
        with self.assertRaisesRegex(ValueError, 'incomplete frame coverage'):
            W.whisper_result(WORDS, front(0), None, None, R.GAP_MIN, R.TAG_WIN)

    def test_confirmation_needs_local_score_and_narrowband_stays_likely(self):
        ev = context([[.9], [.9]], ['Whispering'])
        for channel, confirmed in ((None, True), ({'classification': 'narrowband'}, False),
                                   ({'classification': 'wideband-limited'}, True)):
            result = W.whisper_result(WORDS, front(1.8), ev, channel, R.GAP_MIN, R.TAG_WIN)
            self.assertEqual(result['spans'][0]['confirmed'], confirmed)
        far = context([[.9], [.9]], ['Whispering'], centers=[6, 7])
        result = W.whisper_result(WORDS, front(), far, None, R.GAP_MIN, R.TAG_WIN)
        self.assertFalse(result['spans'][0]['confirmed'])
        low = context([[.39], [.39]], ['Whispering'])
        self.assertFalse(W.whisper_result(WORDS, front(), low, None, R.GAP_MIN, R.TAG_WIN)['spans'][0]['confirmed'])
        voiced = W.whisper_result(WORDS, front(voiced=True), ev, None, R.GAP_MIN, R.TAG_WIN)
        self.assertFalse(voiced['spans'])

    def test_background_sound_withholds_unconfirmed_whisper(self):
        # Under music the pitch tracker can lose a normal voice: frame evidence alone is not a whisper.
        result = W.whisper_result(WORDS, front(), None, None, R.GAP_MIN, R.TAG_WIN, masked='music')
        self.assertEqual(result['spans'], [])
        self.assertEqual(result['withheld'], 1)
        self.assertIn('overlapping music', result['reason'])
        self.assertIsNone(W.whisper_line(result))
        # The model's own Whispering class still confirms one.
        ev = context([[.9], [.9]], ['Whispering'])
        heard = W.whisper_result(WORDS, front(1.8), ev, None, R.GAP_MIN, R.TAG_WIN, masked='music')
        self.assertTrue(heard['spans'][0]['confirmed'])

    def test_confirmation_requires_count_share_and_time_coverage(self):
        words = [('paper', .2 + i, .9 + i) for i in range(60)]
        centers = np.arange(1, 59.1, .5)
        for positive in (1, 2):
            scores = np.zeros((len(centers), 1))
            scores[:positive] = .9
            result = W.whisper_result(words, front(60), context(scores, ['Whispering'], centers, 60),
                                      None, R.GAP_MIN, R.TAG_WIN, duration=60)
            span = result['spans'][0]
            self.assertFalse(span['confirmed'])
            self.assertEqual(span['support_windows'], positive)
            self.assertEqual(span['total_windows'], len(centers))
            self.assertIn(f'AS support {positive}/{len(centers)}', W.whisper_line(result))
            self.assertIn('⟪≈LIKELY WHISPER⟫', str(W.whisper_tokens(result, words)))
        for centers, scores, confirmed in (([1], [[.9]], False),
                                            ([1, 1.5], [[.9], [.9]], True),
                                            ([1, 1.5, 2, 2.5], [[.9], [0], [0], [0]], False)):
            result = W.whisper_result(WORDS, front(1.8), context(scores, ['Whispering'], centers),
                                      None, R.GAP_MIN, R.TAG_WIN)
            self.assertEqual(result['spans'][0]['confirmed'], confirmed)
        # Two positive windows are all the supplied windows, but cover little of this run.
        sparse = W.whisper_result(words, front(60), context([[.9], [.9]], ['Whispering'], [1, 1.5], 60),
                                  None, R.GAP_MIN, R.TAG_WIN, duration=60)
        self.assertFalse(sparse['spans'][0]['confirmed'])
        for count in (2, 3):
            scores = [[.9]] * count + [[0]] * (5 - count)
            events = context(scores, ['Whispering'], [.5, .75, 1, 1.25, 1.5])
            result = W.whisper_result(WORDS, front(), events, None, R.GAP_MIN, R.TAG_WIN)
            self.assertEqual(result['spans'][0]['confirmed'], count == 3)

    def test_card_reports_measured_nonzero_voicing(self):
        fe = front(2)
        fe['voiced'][::8] = True
        result = W.whisper_result([('paper', 0, 1)], fe, None, None, R.GAP_MIN, R.TAG_WIN)
        self.assertAlmostEqual(result['spans'][0]['voiced_share'], 8 / 63)
        self.assertIn('13% voiced', W.whisper_line(result))
        self.assertNotIn('no pitch', W.whisper_line(result))

    def test_maskers_must_overlap_and_narrowband_keeps_model_support(self):
        @W.ST.engine
        def run(centers, scores, labels, narrow=False):
            data = dict(events=context(scores, labels, centers, 60),
                        gaps=dict(wordless=[], unvoiced=[]),
                        channel={'classification': 'narrowband'} if narrow else None)
            W.enrich(data, front(60), WORDS, [], R.GAP_MIN, R.TAG_WIN, 60)
            return data
        late = run([58, 58.5], [[.8], [.8]], ['Music'])
        self.assertTrue(late['whisper']['spans'])
        near = run([1, 1.5], [[.8], [.8]], ['Music'])
        self.assertEqual(near['whisper']['withheld'], 1)
        self.assertIn('overlapping music', near['whisper']['reason'])
        narrow = run([1, 1.5], [[.8, .9], [.8, .9]], ['Music', 'Whispering'], True)
        self.assertFalse(narrow['whisper']['spans'][0]['confirmed'])
        self.assertEqual(narrow['whisper']['withheld'], 0)
        # Separate early and late candidates are checked against their own times.
        events = context([[.8], [.8]], ['Music'], [58, 58.5], 60)
        words = WORDS + [('paper', 58, 58.8)]
        result = W.whisper_result(words, front(60), events, None, R.GAP_MIN, R.TAG_WIN, 60,
                                  masked=events['room']['hits'])
        self.assertEqual(len(result['spans']), 1)
        self.assertEqual(result['withheld'], 1)

    def test_invalid_probabilities_are_diagnosed_not_negative_or_confirmed(self):
        for invalid in (np.nan, np.inf, -np.inf, 1e308, -.1, 1.1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'probabilities'):
                context([[.9, invalid], [.9, invalid]], ['Dog', 'Bark'])
            events = dict(whisper_windows=[dict(center=1, score=invalid)], whisper_threshold=.4)
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                W.whisper_result(WORDS, front(), events, None, R.GAP_MIN, R.TAG_WIN)
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                W.whisper_result(None, front(), events, None, R.GAP_MIN, R.TAG_WIN)
        result, _tag = self.analyze_fixture(probs=np.full((5, 2), np.nan))
        self.assertIsNone(result['data']['room'])
        self.assertIn('unusable event probabilities', '\n'.join(result['lines']))
        self.assertTrue(any(row['stage'] == 'events' and row['status'] == 'failed'
                            for row in result['stage_status']))

    def test_incomplete_or_invalid_frames_are_diagnosed(self):
        for fe in (dict(times=[0], active=[True], voiced=[False]),
                   dict(times=[0, .5, 1], active=[True] * 3, voiced=[False] * 3),
                   dict(times=[0, .016], active=[np.nan, np.nan], voiced=[False] * 2)):
            with self.assertRaisesRegex(ValueError, 'frame'):
                W.whisper_result([('paper', 0, 1)], fe, None, None, R.GAP_MIN, R.TAG_WIN, 1)
        @W.ST.engine
        def run():
            data = dict(events=context([[.9], [.9]], ['Whispering']), channel=None,
                        gaps=dict(wordless=[], unvoiced=[]))
            notes = []
            W.enrich(data, front(0), WORDS, notes, R.GAP_MIN, R.TAG_WIN, 4)
            return dict(data=data, notes=notes)
        result = run()
        self.assertIsNone(result['data']['whisper'])
        self.assertIn('incomplete frame coverage', ' '.join(result['notes']))
        self.assertTrue(any(row['stage'] == 'whisper' and row['status'] == 'failed'
                            for row in result['stage_status']))

    def test_unvoiced_subtracts_all_spans(self):
        gaps = dict(unvoiced=[(0, 10)], unvoiced_s=10)
        W.exclude_unvoiced(gaps, dict(spans=[dict(start=a, end=a + 1) for a in (1, 3, 5, 7)]))
        self.assertEqual(gaps['unvoiced_s'], 6)
        self.assertEqual(gaps['unvoiced'], [(0, 1), (2, 3), (4, 5), (6, 7), (8, 10)])

    def test_inline_markers_preserve_peak_emphasis_events(self):
        result = W.whisper_result(WORDS, front(), None, None, R.GAP_MIN, R.TAG_WIN)
        tokens = R._word_tokens(WORDS, dict(ok=True, start=.2, end=1.15),
                                dict(events=[dict(label='Sigh', start=.7, end=.8)]),
                                dict(wordless=[]), {1}, result)
        text = ' '.join(t[2] for t in tokens)
        for marker in ('⟪PEAK⟫', '⟪/PEAK⟫', '⟪≈LIKELY WHISPER⟫', '⟪/WHISPER⟫', '*lantern*', '[sigh]'):
            self.assertIn(marker, text)
        self.assertLess(text.index('⟪≈LIKELY WHISPER⟫'), text.index('paper'))
        self.assertGreater(text.index('⟪/WHISPER⟫'), text.index('folds'))

    def test_card_limits_and_longest_then_time_order(self):
        spans = [dict(start=a, end=a + length, words=4, confirmed=True, voiced_share=0.,
                      support_windows=2, total_windows=2, support_time_share=1., model_available=True)
                 for a, length in
                 ((1, 1), (5, 4), (12, 3), (20, 2))]
        line = W.whisper_line(dict(spans=spans))
        self.assertNotIn('0:01 ', line)
        self.assertIn('0:05', line)
        self.assertIn('AS support 2/2, 100% time', line)
        for confirmed in (False, True):
            for span in spans:
                span['confirmed'] = confirmed
            self.assertLessEqual(len(W.whisper_line(dict(spans=spans))), R.MAX_LINE)
        ev = context(np.full((5, len(W.ROOM_CLASSES)), .8), [family[0] for _, family, _ in W.ROOM_CLASSES])
        self.assertLessEqual(len(W.room_line(ev['room'])), R.MAX_LINE)
        long = dict(spans=[dict(start=3600, end=7200, words=10000, confirmed=False,
                               unvoiced_s=3600., whisper_score=.83, support_windows=7200)])
        line = W.whisper_line(long)
        self.assertIn('Whispering 7200w peak 0.83', line)
        self.assertIn('10000 words transcribed', line)
        self.assertLessEqual(len(line), R.MAX_LINE)

    def analyze_fixture(self, events=True, narrow=False, fail=False, probs=None, words=WORDS, fe=None, laughs=False,
                        **options):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = stack.enter_context(tempfile.TemporaryDirectory())
        path = Path(directory) / 'invented.wav'
        path.touch()
        # Supplied frame masks are deliberate fixtures; real pyin is tested below.
        audio = np.full(4 * R.SR_SED, .1, np.float32)
        word_frames = front()
        word_frames['active'][word_frames['times'] >= 1.6] = False
        for name, value in (('_torch', None), ('load_audio', audio),
                            ('_front_end', word_frames if fe is None else fe),
                            ('_source_sample_rate', R.SR_SED), ('_pick_device', 'cpu'),
                            ('analyze_channel', dict(classification='narrowband' if narrow else None, rolloff_99_hz=3400)),
                            ('analyze_quality', dict(ok=False, reason='synthetic fixture'))):
            stack.enter_context(patch.object(R, name, return_value=value))
        for name in ('release_models', '_note_vram'):
            stack.enter_context(patch.object(R, name))
        if 'RAVEN_EARS_ROOM_DOGS' not in os.environ:
            # Only this key: a deferred patch.dict snapshots the WHOLE environment and restores it
            # at cleanup time, which re-applied the caller's own (already exited) env patches and
            # leaked RAVEN_EARS_LAUGH_MODEL / RAVEN_EARS_SEPARATE into later test modules.
            os.environ['RAVEN_EARS_ROOM_DOGS'] = 'off'
            self.addCleanup(os.environ.pop, 'RAVEN_EARS_ROOM_DOGS', None)
        tag = stack.enter_context(patch.object(R, '_tag_windows', return_value=(
            np.arange(1, 3.5, .5), np.full((5, 2), .8) if probs is None else probs, ['Dog', 'Whispering']),
            side_effect=RuntimeError('synthetic tagger unavailable') if fail else None))
        return R.analyze(path, words=words, events=events, affect=False, laughs=laughs,
                         use_whisper=False, update_baseline=False, **options), tag

    def test_separation_flag_reaches_single_clip_and_server(self):
        with patch.object(R, 'analyze', return_value=dict(lines=[], timings={}, notes=[])) as analyze:
            self.assertEqual(R.main(['invented.wav', '--separate', '--quiet', '--no-picture']), 0)
            self.assertIs(analyze.call_args.kwargs['separate'], True)
            self.assertEqual(R.main(['invented.wav', '--quiet', '--no-picture']), 0)
            self.assertIsNone(analyze.call_args.kwargs['separate'])
        with patch.object(R, 'serve', return_value=0) as serve:
            self.assertEqual(R.main(['--serve', '--separate']), 0)
            self.assertIs(serve.call_args.kwargs['separate'], True)

    def test_separation_routes_room_voice_and_laugh_worker_and_cleans_stems(self):
        import laugh_models
        import room_laugh

        vocals = np.full(4 * R.SR_SED, .05, np.float32)
        remainder = np.full_like(vocals, .02)
        scratch_paths = []

        def separate(path, samples, timings, scratch):
            self.assertEqual(samples, len(vocals))
            stem = Path(scratch) / 'vocals.wav'
            stem.touch()
            scratch_paths.append(stem)
            return vocals, remainder, str(stem), dict(status='ran', voice='vocals', room='remainder')

        def detect(name, path):
            self.assertEqual(name, 'omine')
            self.assertTrue(Path(path).is_file())
            self.assertEqual(Path(path), scratch_paths[0])
            return dict(model='omine', status='ran', spans=[])

        # Registered as a cleanup so it unwinds AFTER analyze_fixture's own environment
        # snapshot (which runs at cleanup time); a with-block here leaked the setting.
        env = patch.dict(os.environ, {'RAVEN_EARS_SEPARATE': '1', 'RAVEN_EARS_ROOM_DOGS': 'raw',
                                      'RAVEN_EARS_LAUGH_MODEL': 'omine'})
        env.start()
        self.addCleanup(env.stop)
        with patch.object(R, '_separated_audio', side_effect=separate), \
                patch.object(R, '_read_baseline') as baseline, \
                patch.object(R, 'analyze_faint_dogs', return_value=None) as dogs, \
                patch.object(room_laugh, 'detect', return_value=dict(laughs=[], snorts=[])) as cnn, \
                patch.object(laugh_models, 'run_detector', side_effect=detect) as omine:
            result, tag = self.analyze_fixture(fe=front(voiced=True), laughs=True, speaker='invented')
        self.assertIs(tag.call_args_list[0].args[0], vocals)
        self.assertIs(tag.call_args_list[1].args[0], remainder)
        self.assertIs(dogs.call_args.args[0], remainder)
        self.assertIs(cnn.call_args.kwargs['y'], vocals)
        baseline.assert_not_called()
        omine.assert_called_once()
        self.assertFalse(scratch_paths[0].exists())
        card = '\n'.join(result['lines'])
        self.assertIn('ROOM from remainder', card)
        self.assertIn('PITCH, TUNE, LAUGHS, QUALITY and other voice measures from vocals', card)
        self.assertIn('WORDS and PACE use supplied word times', card)
        self.assertIn('VS USUAL: skipped', card)
        self.assertEqual(result['data']['separation']['room'], 'remainder')

    def test_missing_separator_uses_mix_and_off_never_calls_it(self):
        env = patch.dict(os.environ, {'RAVEN_EARS_SEPARATE': '1'})
        env.start()
        self.addCleanup(env.stop)  # unwinds after analyze_fixture's cleanup-time env snapshot
        with patch.object(R, '_separated_audio', side_effect=ModuleNotFoundError('Demucs missing')) as split:
            failed, tag = self.analyze_fixture()
            off, _ = self.analyze_fixture(separate=False)
        split.assert_called_once()
        np.testing.assert_allclose(tag.call_args.args[0], .1)
        self.assertEqual(failed['data']['separation']['voice'], 'mix')
        self.assertEqual(failed['data']['separation']['room'], 'mix')
        self.assertIn('using original mix', '\n'.join(failed['lines']))
        self.assertNotIn('separation', off['data'])

    def test_speech_reuses_the_existing_stem_cache(self):
        import singer

        audio = np.full(R.SR_SED, .03, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            path = cache / 'invented.webm'
            path.touch()
            stem_dir = cache / singer.file_hash(path)
            stem_dir.mkdir()
            for source in singer.SRC:
                (stem_dir / (source + '.wav')).touch()
            (stem_dir / 'model.txt').write_text('htdemucs\n')
            with patch.object(singer, 'CACHE', directory), \
                    patch.object(singer, '_local_stem_model') as model, \
                    patch.object(singer, 'load_mono', return_value=(audio, R.SR_SED)):
                vocals, remainder, voice_file, info = R._separated_audio(path, len(audio), {}, directory)
        model.assert_not_called()
        np.testing.assert_array_equal(vocals, audio)
        np.testing.assert_allclose(remainder, 3 * audio)
        self.assertEqual(Path(voice_file).parent, stem_dir)
        self.assertEqual(info['model'], 'htdemucs')
        self.assertTrue(info['cached'])

    def test_engine_faint_pass_and_failures_use_the_speech_card(self):
        span = dict(start=2.5, end=3.5, time=3., score=.18)
        candidate = dict(method='raw', spans=[span])
        with patch.dict(os.environ, {'RAVEN_EARS_ROOM_DOGS': 'raw'}), \
                patch.object(R, 'analyze_faint_dogs', return_value=candidate):
            result, _ = self.analyze_fixture(probs=np.zeros((5, 2)))
        with patch.dict(os.environ, {'RAVEN_EARS_ROOM_DOGS': 'voice-reduced'}), \
                patch.object(R, 'analyze_faint_dogs', side_effect=[candidate, RuntimeError('synthetic failure')]) as dog:
            failed, _ = self.analyze_fixture(probs=np.zeros((5, 2)))
        self.assertIn('dog, faint (0:03)', '\n'.join(result['lines']))
        self.assertEqual(result['data']['room']['hits'][0]['spans'][0], dict(span, method='raw'))
        self.assertEqual([c.kwargs['reduced'] for c in dog.call_args_list], [False, True])
        self.assertIn('ROOM faint dogs (voice-reduced) skipped: RuntimeError: synthetic failure', failed['notes'])
        self.assertIn('dog, faint (0:03)', '\n'.join(failed['lines']))
        self.assertIsNotNone(failed['data']['events'])

    def test_faint_pass_scores_one_second_windows_on_the_selected_waveform(self):
        fe = front(duration=4, voiced=True)
        fe['voiced'][fe['times'] >= 2] = False
        y = np.full(4 * R.SR_SED, .1, np.float32)
        centers = np.arange(.5, 4., .5)
        with patch.object(R, '_pick_device', return_value='cpu'), \
                patch.object(R, '_tag_windows', return_value=(centers, np.zeros((7, 1)), ['Dog'])) as tag:
            self.assertEqual(R.analyze_faint_dogs(y, 4., fe, [])['method'], 'raw')
            self.assertEqual(R.analyze_faint_dogs(y, 4., fe, [], reduced=True)['method'], 'voice-reduced')
        raw, reduced = tag.call_args_list
        self.assertIs(raw.args[0], y)
        self.assertAlmostEqual(reduced.args[0][R.SR_SED], .018)
        self.assertAlmostEqual(reduced.args[0][3 * R.SR_SED], .1)
        self.assertEqual([c.kwargs['win'] for c in tag.call_args_list], [1., 1.])

    def test_engine_laugh_hint_uses_unranked_features_and_no_words_in_hint(self):
        words = [('paper', .1, .3), ('um', .4, 1.0), ('lantern', 1.1, 1.4), ('folds', 1.5, 1.8)]
        feats = [dict(pitch_st=0., loud_db=0., dur_ratio=1.) for _ in words]
        feats[1] = dict(pitch_st=12., loud_db=0., dur_ratio=2.)
        with patch('room_laugh.detect', return_value=dict(laughs=[], snorts=[])), \
                patch.object(R, 'emphasis_features', return_value=feats), \
                patch.dict(os.environ, {'RAVEN_EARS_LAUGH_MODEL': 'off'}):
            result, _ = self.analyze_fixture(words=words, laughs=True)
        line = next(line for line in result['lines'] if line.startswith('  LAUGHS:'))
        self.assertIn('laughing? big pitch jumps on short words at 0:00', line)
        self.assertNotIn('"um"', line)
        self.assertFalse(result['data']['emphasis']['words'])  # No loudness vote needed by the hint.
        self.assertEqual(set(result['data']['laugh_hint'][0]), {'start', 'end', 'pitch_st', 'dur_ratio'})

    def test_engine_optional_laugh_model_and_skip_status(self):
        for result in (dict(model='omine', status='ran', spans=[[1., 2.]], duration=4., seconds=.2),
                       dict(model='omine', status='failed', reason='unavailable', seconds=.1)):
            with self.subTest(status=result['status']), \
                    patch('room_laugh.detect', side_effect=RuntimeError('synthetic CNN14 failure')), \
                    patch('laugh_models.run_detector', return_value=result) as detector, \
                    patch.dict(os.environ, {'RAVEN_EARS_LAUGH_MODEL': 'omine'}):
                card, _ = self.analyze_fixture(laughs=True)
            detector.assert_called_once()
            self.assertEqual(card['data']['laugh_model'], result)
            if result['status'] == 'ran':
                line = next(line for line in card['lines'] if line.startswith('  LAUGHS:'))
                self.assertIn('CNN14 unavailable', line)
                self.assertIn('1 heard by omine (0:01-0:02)', line)
                self.assertNotIn('snorts none', line)
            else:
                self.assertIn('laugh model skipped: unavailable', card['notes'])

    def test_engine_no_laughs_does_not_start_optional_worker(self):
        with patch('laugh_models.run_detector') as detector, \
                patch.dict(os.environ, {'RAVEN_EARS_LAUGH_MODEL': 'gillick'}):
            self.analyze_fixture(laughs=False)
        detector.assert_not_called()

    def test_engine_reuses_tagger_and_emits_json_and_inline_output(self):
        result, tag = self.analyze_fixture()
        self.assertEqual(tag.call_count, 1)
        self.assertTrue(result['data']['room']['hits'])
        self.assertTrue(result['data']['whisper']['spans'][0]['confirmed'])
        text = '\n'.join(result['lines'])
        self.assertIn('ROOM:', text)
        self.assertIn('WHISPER:', text)
        self.assertIn('⟪WHISPER⟫', text)
        json.dumps(result['data'])
        self.assertTrue(all(len(line) <= R.MAX_LINE for line in result['lines'] if 'ROOM:' in line or 'WHISPER:' in line))

    def test_engine_unworded_single_word_and_merged_cards_reuse_one_tagger(self):
        for words in (None, [], [('paper', .3, .5)], WORDS):
            with self.subTest(words=words):
                result, tag = self.analyze_fixture(words=words, fe=front())
                self.assertEqual(tag.call_count, 1)
                lines = [line for line in result['lines'] if line.strip().startswith('WHISPER:')]
                self.assertEqual(len(lines), 1)
                self.assertIn('4.0s unvoiced; Whispering 5 windows, peak 0.80', lines[0])
                self.assertEqual(len(result['data']['whisper']['spans']), 1)
                json.dumps(result['data'])
        for enabled, failed in ((False, False), (True, True)):
            result, _tag = self.analyze_fixture(words=None, fe=front(), events=enabled, fail=failed)
            self.assertFalse(result['data']['whisper']['spans'])
            self.assertFalse(any(line.strip().startswith('WHISPER:') for line in result['lines']))

    def test_engine_event_skip_failure_and_narrowband(self):
        for enabled, failed in ((False, False), (True, True)):
            result, _tag = self.analyze_fixture(events=enabled, fail=failed)
            self.assertIsNone(result['data']['room'])
            self.assertFalse(result['data']['whisper']['spans'][0]['confirmed'])
            self.assertFalse(any(line.startswith('  ROOM:') for line in result['lines']))
            self.assertFalse(any(row['stage'] == 'room' for row in result['stage_status']))
            self.assertTrue(any(row['stage'] == 'events' and row['status'] != 'ran' for row in result['stage_status']))
        result, _tag = self.analyze_fixture(narrow=True)
        self.assertFalse(result['data']['whisper']['spans'][0]['confirmed'])
        self.assertIn('≈likely', '\n'.join(result['lines']))

    def test_feature_failure_and_off_switch_reach_notes(self):
        with patch.object(W, 'whisper_result', side_effect=ValueError('synthetic frames invalid')):
            result, _tag = self.analyze_fixture()
        self.assertIsNone(result['data']['whisper'])
        self.assertIn('synthetic frames invalid', '\n'.join(result['lines']))
        with patch.dict(os.environ, {'RAVEN_EARS_ROOM': '0', 'RAVEN_EARS_WHISPER': '0'}):
            result, _tag = self.analyze_fixture()
        self.assertIsNone(result['data']['room'])
        self.assertIsNone(result['data']['whisper'])
        self.assertTrue(any(row['stage'] == 'whisper' and row['status'] == 'skipped' for row in result['stage_status']))


class SyntheticFrontEndTests(unittest.TestCase):
    def test_resonated_noise_voiced_vowel_and_breath_without_words(self):
        from test_ears_plus import _resonate, vowel
        rng = np.random.default_rng(913)
        noise = _resonate(rng.normal(size=R.SR * 2), [(700, 110), (1200, 130), (2600, 200)])
        noise = .2 * noise / max(np.abs(noise))
        for audio, expected in ((noise, True), (vowel(2), False)):
            fe = R._front_end(audio.astype(np.float32))
            result = W.whisper_result(WORDS, fe, None, None, R.GAP_MIN, R.TAG_WIN)
            self.assertEqual(bool(result['spans']), expected)
            self.assertFalse(W.whisper_result(None, fe, None, None, R.GAP_MIN, R.TAG_WIN)['spans'])


def run_checks(check):
    for case in unittest.defaultTestLoader.loadTestsFromTestCase(RoomWhisperTests):
        result = unittest.TestResult()
        with patch.dict(os.environ, {'RAVEN_EARS_ROOM': '1', 'RAVEN_EARS_WHISPER': '1'}):
            case.run(result)
        check('ROOM/WHISPER: ' + case._testMethodName, result.wasSuccessful(),
              '\n'.join(detail for _, detail in result.failures + result.errors))


if __name__ == '__main__':
    unittest.main()
