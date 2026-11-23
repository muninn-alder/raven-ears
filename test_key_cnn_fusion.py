"""Measured key vectors and dissent policy; no audio, models, cache or network.

Run with python3 test_key_cnn_fusion.py. Frozen vectors are the old final
votes; replay vectors are measured after both chord-decoding passes. This test
does not itself decode chords. Fixture JSON contains only titles and vectors.
"""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import harmony as H
import keyfuse as K


# Frozen from the declared 25-song replay; tuple = working, state, key, dissent.
EXPECTED = {
    'tolerate it': (('A major', 'clear', 'A major', False), ('A major', 'clear', 'A major', False)),
    'Blue Nile': (('E major', 'clear', 'E major', False), ('E major', 'clear', 'E major', False)),
    'Strawberry Fields Forever': (('Bb major', 'clear', 'Bb major', False), ('Bb major', 'clear', 'Bb major', False)),
    'Rich Girl': (('F major', 'clear', 'F major', False), ('F major', 'clear', 'F major', False)),
    'Last Kiss': (('Bb major', 'clear', 'Bb major', False), ('Bb major', 'clear', 'Bb major', False)),
    'Take Five': (('Eb minor', 'lean', 'Eb minor', True), ('Eb minor', 'clear', 'Eb minor', False)),
    'Closer': (('C minor', 'split', None, True), ('C minor', 'lean', 'C minor', True)),
    'La Vie en rose': (('Ab major', 'lean', 'Ab major', True), ('Ab major', 'clear', 'Ab major', False)),
    'Thunder God': (('C# minor', 'clear', 'C# minor', False), ('C# minor', 'clear', 'C# minor', False)),
    'Counting Stars': (('C# minor', 'clear', 'C# minor', False), ('C# minor', 'clear', 'C# minor', False)),
    'Money': (('B minor', 'clear', 'B minor', False), ('B minor', 'clear', 'B minor', False)),
    'Diary of Jane': (('Bb minor', 'clear', 'Bb minor', False), ('Bb minor', 'clear', 'Bb minor', False)),
    'From Eden': (('C major', 'lean', 'C major', True), ('C major', 'lean', 'C major', True)),
    'Outshined': (('D minor', 'clear', 'D minor', False), ('D minor', 'clear', 'D minor', False)),
    'Fell On Black Days': (('E minor', 'clear', 'E minor', False), ('E minor', 'clear', 'E minor', False)),
    'Never Fade Away': (('E minor', 'clear', 'E minor', False), ('E minor', 'clear', 'E minor', False)),
    'Give a Little Bit': (('D major', 'clear', 'D major', False), ('D major', 'clear', 'D major', False)),
    'Solsbury Hill': (('B major', 'clear', 'B major', False), ('B major', 'clear', 'B major', False)),
    'Fade Into You': (('E major', 'clear', 'E major', False), ('E major', 'clear', 'E major', False)),
    'Homewrecker': (('Db major', 'clear', 'Db major', False), ('Db major', 'clear', 'Db major', False)),
    "Wouldn't It Be Nice": (('F major', 'clear', 'F major', False), ('F major', 'clear', 'F major', False)),
    'Father Figure (Taylor)': (('G major', 'lean', 'G major', True), ('G major', 'lean', 'G major', True)),
    'Father Figure (George Michael)': (('Bb major', 'clear', 'Bb major', False), ('Bb major', 'clear', 'Bb major', False)),
    'Getaway Car': (('C major', 'lean', 'C major', True), ('C major', 'clear', 'C major', False)),
    'Lucy': (('D major', 'clear', 'D major', False), ('D major', 'clear', 'D major', False)),
}
BASELINE_RIGHT = {
    'tolerate it': 'A major',
    'Blue Nile': 'E major',
    'Strawberry Fields Forever': 'Bb major',
    'Rich Girl': 'F major',
    'Last Kiss': 'Bb major',
    'Money': 'B minor',
    'Diary of Jane': 'Bb minor',
    'From Eden': 'C major',
    'Outshined': 'D minor',
    'Fell On Black Days': 'E minor',
    'Never Fade Away': 'E minor',
    'Give a Little Bit': 'D major',
    'Solsbury Hill': 'B major',
    'Homewrecker': 'Db major',
    "Wouldn't It Be Nice": 'F major',
}


def vote(key, gap=6.0):
    values = np.full(24, -gap)
    values[key] = 0.0
    return values


def rows():
    return json.loads(Path(__file__).with_name('test_key_cnn_fixtures.json').read_text())


def chord_events(*chords):
    events = []
    t = 0
    for root, quality, length in chords:
        events.append(dict(t0=t, t1=t + length, root=root, q=quality, conf=.9, bass=root))
        t += length
    return events


class MixolydianTests(unittest.TestCase):
    def test_fusion_keeps_votes_and_reads_numerals_from_modal_home(self):
        events = chord_events(*[(9, 'dom7', 6), (7, 'maj', 3), (2, 'maj', 2)] * 3,
                              (9, 'maj', 8))
        vote_result = K.verdict({'cnn': vote(2)})
        with (patch.object(H.M, 'analyze_key', return_value={'key': 'D major'}),
              patch('librosa.beat.beat_track', return_value=(120, [])),
              patch.object(H, 'decode_chords', return_value=events) as decode,
              patch.object(H, 'fused_verdict', return_value=vote_result)):
            _, home, decoded, _, _, _, verdict = H.resolve_home({'chroma': None, 'oenv': []}, 41, [])
        self.assertEqual(home, (9, 'mixolydian'))
        self.assertEqual(verdict['line'], "A mixolydian (D major's notes, ♭7 = G)")
        self.assertEqual(verdict['voters'], {'cnn': 'D major'})
        self.assertEqual(verdict['major_minor_vote']['top2'][0], 'D major')
        self.assertEqual([H.label(e, [(0, home)]) for e in decoded[:3]], ['I7', '♭VII', 'IV'])
        decode.assert_called_once()

    def test_repeated_secondary_dominant_stays_major(self):
        events = chord_events(*[(9, 'dom7', 10), (2, 'maj', 5), (7, 'maj', 3)] * 4,
                              (2, 'maj', 5))
        self.assertIsNone(H.mixolydian_home(events, (2, 'major')))

    def test_weak_dominant_ending_and_parallel_mixture_stay_major(self):
        major = chord_events((2, 'maj7', 3), *[(4, 'maj', 5), (9, 'maj', 4), (2, 'maj7', 3)] * 4,
                             (4, 'maj', 3))
        mixture = chord_events(*[(10, 'dom7', 8), (6, 'maj', 3), (8, 'maj', 3)] * 4, (10, 'maj', 10))
        self.assertIsNone(H.mixolydian_home(major, (9, 'major')))
        self.assertIsNone(H.mixolydian_home(mixture, (10, 'major')))

    def test_flat_seven_vamp_requires_independent_bass_home(self):
        events = chord_events((7, 'maj', 12), *[(2, 'maj', 6), (0, 'maj', 3), (7, 'dom7', 6)] * 5,
                              (2, 'maj', 6), (7, 'maj', 4))
        bass = dict(pc=np.tile([2, 7], 2000), rel=np.ones(4000))
        self.assertEqual(H.mixolydian_home(events, (2, 'major'), bass), (2, 'mixolydian'))
        self.assertIsNone(H.mixolydian_home(events, (2, 'major')))
        bass['pc'][-int(15 * H.FPS):] = 7
        self.assertIsNone(H.mixolydian_home(events, (2, 'major'), bass))

    def test_iv_estimate_keeps_modal_numerals_when_i7_holds_across_it(self):
        events = chord_events((2, 'maj', 20), (2, 'dom7', 15), (0, 'maj7', 5), (2, 'maj', 12))
        modal = (2, 'mixolydian')
        info = {'changes': [{'at': 30, 'to': 'G major'}]}
        revised = H.modal_key_areas(info, modal, events)
        self.assertEqual(revised['modal_leans'], info['changes'])
        self.assertEqual(H.key_spans(revised, modal), [(0, modal)])
        self.assertEqual(H.label(events[2], H.key_spans(revised, modal)), '♭VIImaj7')
        self.assertEqual(H.classify_cadence(events[2], events[3], modal), 'home')
        self.assertTrue(H.is_diatonic(2, 'dom7', modal))

    def test_iv_estimate_survives_tonicizing_cadence_or_band_reentry(self):
        modal = (2, 'mixolydian')
        info = {'changes': [{'at': 30, 'to': 'G major'}]}
        events = chord_events((2, 'maj', 20), (2, 'dom7', 15), (7, 'maj', 8), (0, 'maj', 5), (2, 'maj', 8))
        self.assertEqual(H.modal_key_areas(info, modal, events), info)
        info['changes'][0]['evidence'] = {'source': 'confirmed band re-entry'}
        events = chord_events((2, 'maj', 20), (2, 'dom7', 15), (0, 'maj', 5), (2, 'maj', 8))
        self.assertEqual(H.modal_key_areas(info, modal, events), info)

    def test_abc_uses_modal_signature_and_keeps_the_tonic(self):
        import eval_shelf
        import melody_abc as abc

        for mode, abc_key, parent in (('A mixolydian', 'AMix', 'D'), ('D mixolydian', 'DMix', 'G'),
                                      ('Db mixolydian', 'C#Mix', 'F#')):
            with self.subTest(mode=mode):
                self.assertEqual(K.spelling_key(mode), abc_key)
                self.assertEqual(abc._key(abc_key), abc_key)
                self.assertEqual(abc._signature(abc_key), abc._signature(parent))
        self.assertEqual(abc._pitch(60, 'DMix', {}), 'C')
        self.assertEqual(abc._pitch(61, 'DMix', {}), '^C')
        self.assertEqual(eval_shelf.key_value("A mixolydian (D major's notes, ♭7 = G)"), [9, 'mixolydian'])


class BluesRockTests(unittest.TestCase):
    def pitch_evidence(self, root, other_roots=()):
        n = int(82 * H.FPS)
        chroma = np.full((12, n), .1)
        for i, degree in enumerate((3, 4, 10)):
            chroma[(root + degree) % 12, i::3] = 1
        bass = dict(pc=np.resize([root, *other_roots], n), rel=np.ones(n))
        return chroma, bass

    def test_confirmed_home_requires_both_thirds_and_a_flat_seventh(self):
        events = chord_events(*[(7, 'maj', 4), (9, 'dom7', 3),
                                (9, 'min', 6), (9, 'maj7', 6)] * 4)
        chroma, bass = self.pitch_evidence(9)
        verdict = dict(K.verdict({'cnn': vote(21)}), top2=['A minor', 'A major'])
        self.assertEqual(H.blues_rock_home(events, (9, 'minor'), verdict, chroma, bass, []), (9, None))
        for missing in (0, 1, 7):
            without_pitch = chroma.copy()
            without_pitch[missing] = .1
            self.assertIsNone(H.blues_rock_home(events, (9, 'minor'), verdict, without_pitch, bass, []))
        # Each pitch alone in a different part of the track does not count as mixture.
        separated = np.full_like(chroma, .1)
        for pc, region in zip((0, 1, 7), np.array_split(np.arange(chroma.shape[1]), 3)):
            separated[pc, region] = 1
        self.assertFalse(H.blues_pitch_support(separated, bass, 9))
        change = [{'at': 35, 'to': 'C major'}]
        self.assertIsNone(H.blues_rock_home(events, (9, 'minor'), verdict, chroma, bass, change))

    def test_rival_dominant_needs_repeated_flat_seven_without_dominant_resolution(self):
        events = chord_events(*[(4, 'dom7', 4), (2, 'maj', 6), (9, 'maj', 5), (11, 'min', 6)] * 4)
        chroma, bass = self.pitch_evidence(4, (2, 9, 11))
        verdict = dict(K.verdict({'cnn': vote(16)}), key='A major', top2=['A major', 'E major'])
        self.assertEqual(H.blues_rock_home(events, (9, 'major'), verdict, chroma, bass, []), (4, 9))
        cadence = chord_events(*[(4, 'dom7', 4), (9, 'maj', 6), (2, 'maj', 5), (11, 'min', 6)] * 4)
        self.assertIsNone(H.blues_rock_home(cadence, (9, 'major'), verdict, chroma, bass, []))

    def test_cadences_wait_for_a_local_key_when_home_is_uncertain(self):
        events = chord_events((2, 'maj', 5), (9, 'maj', 5), (2, 'maj', 3), (7, 'maj', 7))
        spans = [(0, (9, 'major')), (10, (7, 'major'))]
        rows = H.find_cadences(events, spans, [(10, 'section'), (20, 'section')],
                               {'T': np.zeros(20), 'step': 2}, key_from=10)
        self.assertEqual([row['cadence'] for row in rows], ['unclear', 'authentic'])

    def test_abc_can_render_both_thirds_with_no_signature(self):
        import melody_abc as abc

        f0 = np.repeat([261.63, 277.18], 50)
        text, data = abc.transcribe(f0, 50, [], key='none')
        self.assertIn('K:none', text)
        self.assertIn('=C', text)
        self.assertIn('^C', text)
        self.assertEqual(data['notation']['key_provenance'], 'none')


class VoicingTests(unittest.TestCase):
    def test_power_chord_replaces_unsupported_voicing_and_keeps_audible_third(self):
        events = chord_events((4, 'maj7', 1), (2, 'maj7', 1), (9, 'dom7', 1))
        n = round(3 * H.FPS)
        edge1, edge2 = round(H.FPS), round(2 * H.FPS)
        chroma = np.full((12, n), .1)
        chroma[[4, 11], :edge1] = 1
        chroma[[2, 6, 9], edge1:edge2] = 1
        chroma[9, edge2:] = 1
        bass = np.repeat([4, 2, 9], [edge1, edge2 - edge1, n - edge2])
        F = dict(chroma=chroma, bass=chroma, rms_db=np.zeros(n),
                 bass_trk=dict(pc=bass, rel=np.ones(n)))

        voiced = H.power_chords(F, events)
        self.assertEqual([(e['root'], e['q']) for e in voiced if e['root'] is not None],
                         [(4, '5'), (2, 'maj')])
        self.assertEqual(H.chord_name(4, '5'), 'E5')
        self.assertEqual(H.roman(4, '5', (9, 'minor')), 'V5')
        self.assertEqual(H.chord_tones(4, '5'), {4, 11})
        self.assertEqual(voiced[-1]['t1'], 3)
        self.assertTrue(all(a['t1'] == b['t0'] for a, b in zip(voiced, voiced[1:])))
        self.assertEqual(events[0]['q'], 'maj7')

    def test_harmonic_minor_needs_recurrent_heard_leading_tone_on_major_v(self):
        events = chord_events(*[(9, 'min', 20), (4, 'maj', 2)] * 4)
        n = round(events[-1]['t1'] * H.FPS)
        chroma = np.full((12, n), .1)
        for e in events:
            a, b = round(e['t0'] * H.FPS), round(e['t1'] * H.FPS)
            chroma[[9, 0, 4] if e['root'] == 9 else [4, 8, 11], a:b] = 1
        self.assertEqual(H.harmonic_minor_note(events, chroma, 9), 'G♯')
        self.assertIsNone(H.harmonic_minor_note(events[:-2], chroma, 9))
        self.assertIsNone(H.harmonic_minor_note(
            [dict(e, q='min') for e in events], chroma, 9))
        chroma[7] = chroma[8]  # an adjacent bin at the same level cannot count as G-sharp
        self.assertIsNone(H.harmonic_minor_note(events, chroma, 9))


class CNNFusionTests(unittest.TestCase):
    def test_saved_d_sharp_minor_respells_to_e_flat_with_flats(self):
        saved = {'signature': '6♯', 'key': 'D# minor', 'working': 'D# minor',
                 'line': 'D# minor (6♯) · tonic clear'}
        claim = K.saved_key_claim(saved)
        self.assertEqual(claim['key'], 'Eb minor')
        self.assertEqual(claim['signature'], '6♭')
        self.assertEqual(claim['line'], 'Eb minor (6♭) · tonic clear')
        self.assertEqual(K.signature(K.key_index(6, 'major')), (6, '♯'))  # F# major keeps its sharps

    def test_key_spelling_agrees_with_all_signatures(self):
        import harmony
        import melody_abc

        for index in range(24):
            name = K.key_name(index)
            with self.subTest(key=name):
                self.assertEqual(K.key_index(*harmony.parse_key(name)), index)
                alterations = melody_abc._signature(melody_abc._key(name))
                count, accidental = K.signature(index)
                self.assertEqual(sum(alterations.values()), -count if accidental == '♭' else count)
        reading = K.verdict({'cnn': vote(1)})
        self.assertIn('Db major (5♭)', reading['line'])

    def test_note_spelling_follows_the_scale_and_functional_chromatic_degrees(self):
        for given, key, expected in (
                ('G♭4', 'G', 'F♯4'), ('F♯4', 'Db', 'G♭4'),
                ('D♭4', 'G', 'C♯4'), ('A♯4', 'C', 'B♭4'),
                ('C5', 'F#', 'B♯4'), ('B4', 'Cb', 'C♭5'),
                ('E♭4', None, 'E♭4')):
            with self.subTest(note=given, key=key):
                self.assertEqual(K.respell_note(given, key), expected)
        self.assertEqual(K.spelling_key('C# major'), 'Db')
        self.assertIsNone(K.claimed_spelling_key({'state': 'split', 'working': 'D major'}))

    def test_saved_bench_vectors(self):
        self.assertEqual(K.WEIGHTS['cnn'], 10.0)
        self.assertEqual(len(rows()), 25)
        for row in rows():
            for field, expected in zip(('voters', 'replay_voters'), EXPECTED[row['title']]):
                with self.subTest(song=row['title'], vectors=field):
                    voters = dict(row[field])
                    np.testing.assert_allclose(K.v_cnn(row['cnn_raw_madmom_order']), voters['cnn'])
                    result = K.verdict(voters)
                    actual = (result['working'], result['state'], result['key'], 'CNN-led' in result['line'])
                    self.assertEqual(actual, expected)
                    if row['title'] in BASELINE_RIGHT:
                        self.assertEqual(result['key'], BASELINE_RIGHT[row['title']])
                    if expected[-1]:
                        self.assertIsNotNone(result['votes_line'])
                        self.assertNotIn('tonic clear', result['line'])
                        self.assertNotIn('tonic fairly clear', result['line'])

    def test_fixture_contains_only_titles_and_numeric_vectors(self):
        for row in rows():
            self.assertEqual(set(row), {'title', 'cnn_raw_madmom_order', 'voters',
                                        'replay_voters'})
            self.assertIsInstance(row['title'], str)
            vectors = [row['cnn_raw_madmom_order']]
            for field in ('voters', 'replay_voters'):
                self.assertLessEqual(set(row[field]), set(K.WEIGHTS))
                vectors.extend(row[field].values())
            for vector in vectors:
                self.assertEqual(len(vector), 24)
                self.assertTrue(all(type(x) in (int, float) and np.isfinite(x) for x in vector))

    def test_missing_cnn_does_not_gain_influence_or_claim_leadership(self):
        for row in rows():
            with self.subTest(song=row['title']):
                voters = dict(row['voters'], cnn=None)
                result = K.verdict(voters)
                self.assertEqual(result, K.verdict(voters, dict(K.WEIGHTS, cnn=1.0)))
                self.assertNotIn('CNN-led', result['line'])

    def test_shared_dissent_caps_clear_without_changing_fusion(self):
        voters = {'cnn': vote(4), 'chords': vote(9), 'spectral': vote(9)}
        posterior, used = K.fuse(voters)
        result = K.verdict(voters)
        self.assertGreater(posterior[4], K.VERY_CLEAR_P)
        self.assertEqual(result['state'], 'lean')
        self.assertEqual(result['key'], 'E major')
        self.assertEqual(result['working'], 'E major')
        self.assertEqual(result['p'][0], round(float(posterior[4]), 2))
        self.assertEqual(result['line'], 'E major (4♯) · CNN-led; chords and spectral favor A major')
        self.assertEqual(result['votes_line'], 'cnn E · chords A · spectral A')
        self.assertEqual(used, ['cnn', 'chords', 'spectral'])

    def test_separate_dissent_names_each_actual_key_and_mode(self):
        result = K.verdict({'cnn': vote(9), 'chords': vote(21), 'spectral': vote(2)})
        self.assertEqual(result['state'], 'lean')
        self.assertEqual(result['key'], 'A major')
        self.assertEqual(result['line'],
                         'A major (3♯) · CNN-led; chords favor A minor; spectral favors D major')

    def test_both_dissenters_and_cnn_must_be_used(self):
        for name in ('cnn', 'chords', 'spectral'):
            for availability in ('missing', 'none', 'disabled'):
                with self.subTest(voter=name, availability=availability):
                    voters = {'cnn': vote(4), 'chords': vote(9), 'spectral': vote(9)}
                    weights = dict(K.WEIGHTS)
                    if availability == 'missing':
                        voters.pop(name)
                    elif availability == 'none':
                        voters[name] = None
                    else:
                        weights[name] = 0.0
                    self.assertNotIn('CNN-led', K.verdict(voters, weights)['line'])

    def test_one_dissenter_or_agreement_does_not_cap_clear(self):
        for chords, spectral in ((4, 9), (9, 4), (4, 4)):
            result = K.verdict({'cnn': vote(4), 'chords': vote(chords), 'spectral': vote(spectral)})
            self.assertEqual(result['state'], 'clear')
            self.assertNotIn('CNN-led', result['line'])
        # These are the two measured CNN misses: unanimous strongest voters do
        # not meet the declared rule, even though the musical home is wrong.
        for row in rows():
            if row['title'] in ('Fade Into You', 'Lucy'):
                result = K.verdict(row['replay_voters'])
                self.assertEqual(result['state'], 'clear')
                self.assertNotIn('CNN-led', result['line'])
                self.assertIsNone(result['votes_line'])

    def test_cnn_cannot_claim_to_lead_a_different_winner(self):
        result = K.verdict({'cnn': vote(4, .01), 'bass': vote(9),
                            'chords': vote(2, .01), 'spectral': vote(2, .01)})
        self.assertEqual(result['working'], 'A major')
        self.assertNotIn('CNN-led', result['line'])

    def test_tied_dissent_uses_existing_argmax_convention(self):
        tied = vote(0)
        tied[9] = 0.0
        result = K.verdict({'cnn': vote(4), 'chords': tied, 'spectral': tied})
        self.assertIn('chords and spectral favor C major', result['line'])
        self.assertEqual(result['voters']['chords'], 'C major')

    def test_already_unclear_is_not_promoted(self):
        row = next(r for r in rows() if r['title'] == 'Closer')
        result = K.verdict(row['voters'])
        self.assertEqual(result['state'], 'split')
        self.assertIsNone(result['key'])
        self.assertTrue(result['line'].startswith('unclear between C minor'))
        self.assertTrue(result['line'].endswith('CNN-led; chords and spectral favor F major'))
        # Relative-key uncertainty also stays uncertain, with the note added.
        cnn = vote(9, .1)
        cnn[18] = -.001
        result = K.verdict({'cnn': cnn, 'chords': vote(18, .001), 'spectral': vote(18, .001)})
        self.assertEqual(result['state'], 'tonic')
        self.assertIsNone(result['key'])
        self.assertIn('tonic unclear', result['line'])
        self.assertIn('CNN-led; chords and spectral favor F# minor', result['line'])

    def test_existing_lean_keeps_its_low_confidence_wording(self):
        result = K.verdict({'cnn': vote(9, .35), 'chords': vote(2, .001), 'spectral': vote(2, .001)})
        self.assertEqual(result['state'], 'lean')
        self.assertEqual(result['key'], 'A major')
        self.assertIn('low confidence, P ', result['line'])
        self.assertTrue(result['line'].endswith('CNN-led; chords and spectral favor D major'))

    def test_key_card_prints_the_attribution_and_votes(self):
        import music
        row = next(r for r in rows() if r['title'] == 'From Eden')
        result = K.verdict(row['replay_voters'])
        text = music.format_card({'file': 'example.mp3', 'duration': 60, 'key_claim': result})
        self.assertIn('KEY:      ' + result['line'], text)
        self.assertIn('KEY VOTES: ' + result['votes_line'], text)


if __name__ == '__main__':
    unittest.main()
