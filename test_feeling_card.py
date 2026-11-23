"""Offline weaving checks; optionally audit copied real bundles and old cards.

python3 test_feeling_card.py --bundles .limb/bundles --baseline .limb/legacy-summaries.json
"""
import argparse
import copy
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

import feeling_card as F
import listen as L
import slim
from stage_status import NOTES_WIDTH

BUNDLES = None
BASELINE = {}
FULL_BASELINE = None


def example():
    # Real LRPD row shapes and fractional clocks, with invented sung text.
    lyrics = [dict(t=25., end=29.3, text='A lantern in a quiet room', approx=True),
              dict(t=60.88, end=65.1, text='We will keep a lantern by the door', approx=True),
              dict(t=96.64, end=98., text='The shadows on the floor', approx=True),
              dict(t=153.3, end=157.7, text='We could keep a lantern by the door', approx=True)]
    lyric_text = 'LYRICS: supplied times (≈); line ends estimated\n' + '\n'.join(
        f"  [{slim.clock(r['t'])}≈] {r['text']}" for r in lyrics) + '\n'
    music = 'MUSIC: invented.wav · 3:20\nKEY: C major · tonic clear\nTEMPO: 90 BPM\nSHAPE: quiet intro → loud 1:05\nSOUND: dark (centroid 807 Hz) · dynamic range 15.9 dB\nVOCAL: clearer [CPPS 8.8, HNR 13 dB]\n' + lyric_text + '  NOTE: lyrics: invented provenance\n'
    singer = ('SINGER.PY invented.wav 3:20\n' + lyric_text +
              'SINGER:\n  most intense moment: 2:33–2:38 · arousal 0.69 (speech-trained model; read as texture, not a verdict)\n'
              'ENTRANCES/EXITS:\n  0:24 vocals enter\n  1:04 bass + drums enter\n  1:39 bass + drums drop out\n'
              '  2:33 vocals swells 18 dB\n  3:01 guitar alone for 0:12\n')
    harmony = ('HARMONY: invented.wav · 3:20\nCHORDS\n  home: C major; saved\n  ends on I (C); opens on I\n'
               "  notable moves:\n    plagal cadence IV→I (the 'amen' fall)  [1:39, 2:35]\n"
               'TENSION\n  curve (0–1, per 2 s; saved)\n    0:00 ▁▂▃▄\nTIMELINE\n'
               "   1:39  plagal cadence IV→I (the 'amen' fall)\n"
               'CHILL CANDIDATES (candidates, not verdicts)\n  appoggiaturas (pitch track hints):\n'
               '    2:33  F held 0.2 s over C, then up onto G\n')
    cadences = [dict(t=t, t_end=t+10, cadence='plagal', likely=True, strength=1.,
                     kind='line', **{'from': 'IV', 'to': 'I'}) for t in (98.66, 155.2)]
    cards = dict(music=dict(text=music, json=dict(lyrics=lyrics, lyrics_source=dict(status='synced'))),
                 singer=dict(text=singer, json={}),
                 harmony=dict(text=harmony, json=dict(cadences=cadences)))
    return dict(cards=cards, manifest=dict(options={}, passes={n: dict(status='ran', reason='') for n in L.PASSES}),
                melody=dict(status='ran', reason='', text='K:C\nC D\nw: lantern glow\n',
                            json=dict(hook_text='MELODY: approximate vocal ABC · K:C\nK:C\nC D\nw: lantern glow\n')),
                song_extras=dict(lastfm_tags=['gentle'], lastfm_tag_evidence=dict(
                    top_count=100, counts=[dict(name='gentle', count=100)])))


class FeelingTests(unittest.TestCase):
    def test_full_shows_lyrics_once_and_keeps_timed_harmony(self):
        song = example()
        harmony = song['cards']['harmony']
        harmony['text'] += ('LYRIC TIMELINE\n  LYRICS: supplied times (≈)\n'
                            '  [0:25≈] "A lantern in a quiet room" — chord V7\n')
        full = L.render(song, 'full')[0]
        self.assertEqual(full.count('[0:25≈] A lantern in a quiet room'), 1)
        self.assertNotIn('"A lantern in a quiet room"', full)
        self.assertIn('LYRICS: 4 lines, shown above in MUSIC', full)
        self.assertIn('[0:25≈] ♪ chord V7', full)
        singer = L.render(song, 'singer')[0]
        self.assertIn('[0:25≈] A lantern in a quiet room', singer)
        self.assertNotIn('shown above', singer)

        song['manifest']['passes']['music']['status'] = 'not run'
        song['cards']['music']['text'] = ''
        self.assertIn('[0:25≈] A lantern in a quiet room', L.render(song, 'full')[0])

    def test_timeline_credit_requires_a_performer_credit(self):
        song = example()
        singer = song['cards']['singer']
        singer['text'] += ('CREDITS: MusicBrainz unreachable; Discogs lists no performer credits; ears only\n')
        singer['json']['instruments'] = dict(families={'electric guitar (probably)': {}})
        harmony = song['cards']['harmony']
        event = 'electric guitar (credited) alone for 0:09'
        harmony['text'] = harmony['text'].replace('TIMELINE\n', f'TIMELINE\n   0:05  {event}\n')
        harmony['json']['timeline'] = [dict(t=5.0, text=event)]
        for view in ('full', 'harmony', 'first-listen'):
            out = L.render(song, view)[0]
            self.assertNotIn('(credited)', out)
            self.assertIn('electric guitar (probably) alone for 0:09', out)
        self.assertIn('electric guitar (probably)', L.saved_card_readings(song)['cards']['harmony']['json']['timeline'][0]['text'])

        singer['json']['instruments']['credits'] = dict(source='file', entries=[dict(classes=['piano'])])
        self.assertNotIn('(credited)', L.render(song, 'harmony')[0])
        singer['json']['instruments']['credits']['entries'][0]['classes'] = ['electric guitar']
        self.assertIn(event, L.render(song, 'harmony')[0])

    def test_fresh_harmony_timeline_uses_tagger_labels(self):
        import harmony

        result = {name: [] for name in ('secs', 'spans', 'evs', 'moves', 'pedals', 'br', 'dips',
                                        'drops', 'swells', 'con', 'tag', 'csurp', 'lines')}
        result.update(kinfo={}, tc={}, dur=60, path='example.wav')
        instruments = dict(duration=60, vocal_on=[], families={
            'electric guitar (probably)': dict(spans=[[5, 14]], share=.15, conf=.35)})
        with patch('instruments.analyse_raw', return_value=instruments), \
             patch('credits.apply_for', side_effect=AssertionError('timeline requested a separate lookup')):
            text = harmony.block_timeline(result)
        self.assertIn('   0:05  electric guitar enters (possibly)', text)

    def test_harmony_entry_checks_words_against_the_measured_stem(self):
        import harmony
        from lyrics_shared import vocal_onset, vocal_spans

        result = {name: [] for name in ('secs', 'spans', 'evs', 'moves', 'pedals', 'br', 'dips',
                                        'drops', 'swells', 'con', 'tag', 'csurp')}
        rows = [(0., 'opens with other'), (12., 'vocals enter'), (30., 'vocals drop out')]
        result.update(kinfo={}, tc={}, dur=60, path='example.wav',
                      band_presence=dict(vocal_onset=vocal_onset(rows), vocal_spans=vocal_spans(rows, 60)))
        for intro, start, end, expected in (
                ((0., .25), 11.36, 12.76, "   0:11  voice enters: 'The lantern'"),
                ((11.0, 11.2), 11.36, 12.76, "   0:11  voice enters: 'The lantern'"),
                ((0., .25), 16., 17., "   0:12  voice enters: 'The lantern' (words approximate here)")):
            with self.subTest(start=start), patch('instruments.analyse', return_value=None):
                result['words'] = [('Paper.', *intro), ('The', start, end), ('lantern', end, end + .4)]
                result['lines'] = harmony.lines_with_ends(result['words'])
                text = harmony.block_timeline(result)
            self.assertIn(expected, text)
            self.assertFalse(any('Paper.' in row for row in text))

        result['align'] = dict(rows=[dict(t=12.2, end=15., text='A lantern stays lit', approx=True)])
        with patch('instruments.analyse', return_value=None):
            for words in (result['words'], None):
                with self.subTest(words=words):
                    result['words'] = words
                    self.assertIn("   0:12  voice enters: 'A lantern stays lit' (time approximate)",
                                  harmony.block_timeline(result))

    def test_saved_harmony_entry_uses_the_same_stem_measurement(self):
        song = example()
        song['cards']['music']['json']['lyrics_source'] = dict(status='nomatch')
        singer = song['cards']['singer']
        singer['json'].update(stems=dict(events=[dict(t=0., text='opens with other'),
                                                 dict(t=24., text='vocals enter')]),
                              melody_input=dict(words=[['Paper.', 0., .25], ['Lantern', 26., 27.]]))
        harmony = song['cards']['harmony']
        old = "voice enters: 'Paper.'"
        harmony['json']['timeline'] = [dict(t=0., text=old), dict(t=5., text='swells')]
        harmony['text'] = harmony['text'].replace('TIMELINE\n', f'TIMELINE\n   0:00  {old}\n   0:05  swells\n')
        after = "voice enters: 'Lantern' (words approximate here)"
        self.assertIn('   0:05  swells\n   0:24  ' + after, L.render(song, 'full')[0])
        self.assertNotIn(old, L.render(song, 'harmony')[0])
        self.assertEqual(L.saved_card_readings(song)['cards']['harmony']['json']['timeline'],
                         [dict(t=5., text='swells'), dict(t=24., text=after)])
        self.assertIn('0:24', L.render(song, 'first-listen')[0])
        self.assertEqual(harmony['json']['timeline'][0], dict(t=0., text=old))

    def test_saved_placed_lyrics_refresh_event_quotes_in_full_and_sheet(self):
        import listening_sheet

        song = example()
        harmony = song['cards']['harmony']
        events = [(25.0, "voice enters: vocal stem active"),
                  (61.0, "voice: 'A ladder by the door'"),
                  (96.4, "line 'the meadows on the floor': drop in energy"),
                  (140., "voice: 'An old guess in the gap'")]
        harmony['json'].update(band_presence=dict(vocal_onset=24.5),
                               timeline=[dict(t=t, text=label) for t, label in events])
        harmony['text'] = harmony['text'].replace(
            'TIMELINE\n', 'TIMELINE\n' +
            ''.join(f'  {slim.clock(t):>5}  {label}\n' for t, label in events))
        harmony['text'] += 'ACHE 1:02  phrase ends away from home under "A ladder by the door" (lean)\n'
        full = L.full_text(song)
        self.assertIn("   0:25  voice enters: 'A lantern in a quiet room' (time approximate)", full)
        self.assertIn("   1:01  voice: 'We will keep a lantern by the door'", full)
        self.assertIn("   2:20  voice: words not placed here", full)
        self.assertIn('ACHE 1:02  phrase ends away from home under ≈ "We will keep a lantern by the door" (lean)', full)
        line = "line 'The shadows on the floor': drop in energy"
        self.assertIn(f"   1:36  {line}", full)
        data = L.saved_card_readings(song)['cards']['harmony']['json']
        self.assertEqual(data['timeline'][2], dict(t=96.4, text=line))
        decided, _ = L.render(song, 'first-listen')
        self.assertIn((96, 96, line, False), listening_sheet.song_evidence(song, decided, full)['events'])

    def test_one_instrument_hedge_in_fresh_and_saved_cards(self):
        import instruments

        label = 'electric guitar (probably)'
        result = dict(duration=242.0, n=240, hop=1.0, vocal_on=[[5.0, 60.0]], families={
            label: dict(stem='other', conf=.35, peak=.73, share=.59,
                        spans=[[5.0, 60.0], [70.0, 111.0], [125.0, 144.0], [202.0, 229.0]])})
        score = ' (score 0.35, 59% of the song)  0:05–1:00, 1:10–1:51, 2:05–2:24, 3:22–3:49'
        summary = '    electric guitar possibly' + score
        timeline = '    0:05–1:00  electric guitar (probably) + voice'
        self.assertIn(summary, instruments.block(result))
        self.assertIn(timeline, instruments.block(result))

        song = example()
        singer = song['cards']['singer']
        timeline = '    0:07–0:21  bass guitar, drums (type unclear), electric guitar (probably), strings + voice'
        enters, leaves = f'{label} enters (possibly)', f'{label} leaves (possibly)'
        singer['text'] += (
            'INSTRUMENTS (tagger estimate):\n  overall:\n'
            f'    {label} possibly{score}\n'
            "  who is playing ('+ voice' = the vocal stem has level):\n"
            f'{timeline}\n'
            'ENTRANCES/EXITS:\n'
            f'  0:05 {enters}  [instrument tagger]\n')
        singer['json']['instrument_events'] = [dict(t=5.0, text=enters)]
        harmony = song['cards']['harmony']
        harmony['text'] = harmony['text'].replace('TIMELINE\n', f'TIMELINE\n   2:24  {leaves}\n')
        harmony['json']['timeline'] = [dict(t=144.0, text=leaves)]
        full = L.render(song, 'full')[0]
        self.assertIn(summary, full)
        self.assertIn(timeline, full)
        self.assertIn('  0:05 electric guitar enters (possibly)  [instrument tagger]', full)
        self.assertIn('   2:24  electric guitar leaves (possibly)', full)
        self.assertIn(f'    {label} possibly{score}', singer['text'])
        reread = L.saved_card_readings(song)['cards']
        self.assertEqual(reread['singer']['json']['instrument_events'][0]['text'], 'electric guitar enters (possibly)')
        self.assertEqual(reread['harmony']['json']['timeline'][0]['text'], 'electric guitar leaves (possibly)')

        result['families'][label]['conf'] = .40
        self.assertIn((5.0, 'electric guitar enters (likely)'), instruments.events(result))
        result['families'][label]['conf'] = .55
        self.assertTrue(any(f'{label} (score 0.55,' in row for row in instruments.block(result)))
        self.assertIn((5.0, f'{label} enters'), instruments.events(result))

    def test_words_music_marker_requires_strong_opposite_readings_at_the_lyric(self):
        song = example()
        lyrics = song['cards']['music']['json']['lyrics']
        harmony = song['cards']['harmony']['json']
        harmony['contrast'] = [
            dict(t=row['t'], text=row['text'], words=words, music=music, lead='activity', flag=None)
            for row, (words, music) in zip(lyrics, ((.4, -.4), (-.51, .77), (.9, .7), (.39, -.8)))]
        harmony['contrast'] += [
            dict(t=lyrics[-1]['t'], text='Different words', words=.9, music=-.8),
            dict(t=lyrics[-1]['t'] - 3, text=lyrics[-1]['text'], words=.9, music=-.8)]
        text = F.logical_rows(L.render(song, 'first-listen')[0])
        marked = [r.lstrip() for r in text if 'they disagree here' in r]
        self.assertEqual(len(marked), 2)
        self.assertTrue(marked[0].startswith('[0:25≈] A lantern in a quiet room'))
        self.assertIn('← words bright, music low: they disagree here', marked[0])
        self.assertTrue(marked[1].startswith('[1:01≈] We will keep a lantern by the door'))
        self.assertIn('← words dark, music high: they disagree here', marked[1])
        for words, music in ((.8, .8), (-.8, -.8), (.8, -.39), (-.39, .8), (0., -.9)):
            self.assertEqual(F.contrast_marker(words, music), '')
        harmony['stage_status'] = [dict(stage='word / music contrast', status='failed', kind='audio')]
        self.assertNotIn('they disagree here', L.render(song, 'first-listen')[0])

    def test_tracker_notes_inside_silence_do_not_become_lean_ins(self):
        song = example()
        singer = song['cards']['singer']
        singer['json']['singer'] = dict(data=dict(gaps=dict(
            have_words=True, silent=[[152.8, 154.1]], stage_status=[])))
        singer['text'] = singer['text'].replace('SINGER:\n', 'SINGER:\n'
            '  highest held notes: C5 at 2:33 (0.5s)\n'
            '  pitch arc (median by section): A3 → C5 · the voice lifts ~15 semitones by 2:33\n')
        text = L.render(song, 'first-listen')[0]
        self.assertNotIn('appoggiatura:', text)
        self.assertNotIn('highest held note:', text)
        self.assertNotIn("voice's highest section", text)
        self.assertIn('appoggiaturas', L.render(song, 'full')[0])
        for t, allowed in ((151., True), (152., True), (153., False), (154., False), (155., True)):
            self.assertEqual(F._pitch_allowed(t, 24., [[152.8, 154.1]]), allowed)
        singer['json']['singer']['data']['gaps']['stage_status'] = [dict(stage='gaps', status='failed')]
        self.assertIn('appoggiatura:', L.render(song, 'first-listen')[0])

    def test_huge_stem_rises_are_priority_zero_entrances(self):
        for db, label in ((30, 'guitar steps up 30 dB, sustained'), (31, 'guitar enters'), (56, 'guitar enters')):
            song = example()
            song['cards']['singer']['text'] += f'  3:05 guitar steps up {db} dB in one go and stays up [guitar stem level]\n'
            text = L.render(song, 'first-listen')[0]
            self.assertIn('[%s]' % label, text)
            self.assertIn(f'guitar steps up {db} dB', L.render(song, 'full')[0])
        self.assertEqual(F._stem_entrance(
            'guitar arrives as 2 separate hits (2:37.5, 2:37.8) and holds from 2:40.0, 56 dB up [guitar stem level]'),
            'guitar enters')

    def test_placeholder_lyrics_are_not_sung_lines(self):
        for placeholder in ('(...)', '…', '[Instrumental]', '', '--- !!!'):
            with self.subTest(placeholder=placeholder):
                self.assertFalse(F._sung_line(placeholder + ' (no voice measured here)'))
                song = example()
                song['cards']['music']['text'] = song['cards']['music']['text'].replace(
                    '[0:25≈] A lantern in a quiet room', '[0:25≈] ' + placeholder + ' (no voice measured here)')
                self.assertNotIn('[0:25≈]', L.render(song, 'first-listen')[0])
        self.assertTrue(F._sung_line('Ooh, ah'))
        self.assertTrue(F._sung_line('123'))

    def test_meter_grouping_stays_in_full(self):
        jargon = '4/4, accents repeat every 2 bars at 157.9/min: best fit 2+2+2+2 (grouping weak); 8/8 only if each pulse is an eighth'
        self.assertEqual(F._meter('157.9 BPM', jargon, {}), '157.9 BPM in 4 (≈39 bars a minute)')
        self.assertIn('in 4', F._meter('100 BPM', '4 beats per bar, grouping weak', {}))
        self.assertNotIn('grouping', F._meter('100 BPM', '3/4, accents repeat', {}))
        self.assertIn('(uncertain)', F._meter('100 BPM', '4/4 (uncertain)', {}))
        self.assertIn('(ambiguous)', F._meter('100 BPM', '4 beats per bar (ambiguous)', {}))

    def test_opening_duplicate_and_qualified_enter(self):
        song = example()
        song['cards']['singer']['text'] += ('  0:00 instruments at the open: acoustic guitar\n'
            '  0:01 acoustic guitar (likely) enter\n  0:02 flute (possibly) enter\n'
            '  0:03 acoustic guitar (likely) enter\n')
        text = L.render(song, 'first-listen')[0]
        self.assertNotIn('0:01 ', text)
        self.assertIn('flute (possibly) enters', text)
        self.assertIn('acoustic guitar (likely) enters', text)
        self.assertIn('0:01 acoustic guitar (likely) enter', L.render(song, 'full')[0])
        song['cards']['singer']['text'] = song['cards']['singer']['text'].replace(
            'at the open: acoustic guitar', 'at the open: flute, acoustic guitar')
        self.assertNotIn('0:01 ', L.render(song, 'first-listen')[0])

    def test_hook_dips_title_cap_and_supplied_comment(self):
        import listen_melody as M
        # Saved word-phrase shape, including a leading unrelated lyric line.
        notes = [dict(start=t, end=t+.2, midi=m) for t, m in
                 ((0., 64), (1., 65), (2., 67), (3., 64), (4., 40))]
        data = dict(hook=dict(line_indices=[0]), notes=notes, lyric_lines_unplaced='failed',
                    lines=[dict(start=0., end=5., written_origin_s=0., note_indices=list(range(5)),
                                lyrics='old words lantern glow *')])
        hook = 'HOOK: line with the title "lantern"\nK:C\n% [0:00] Old words lead us to lantern glow\nE F G E E,,\nw: old words lantern glow *\n'
        lyrics = [slim.Lyric(0., 'Old words', True), slim.Lyric(2., 'Lantern glow', True)]
        out = M.feeling_hook(hook, data, lyrics, ['lantern'])
        self.assertIn('% [0:02≈] Lantern glow', out)
        self.assertIn('G E\nw: lantern glow', out)
        self.assertIn('% 1 low dip left out', out)
        self.assertNotIn('E,,', out)
        self.assertNotIn('old words', out)
        with patch.dict(sys.modules, {'melody_abc': None}):
            accidental = M.feeling_hook(hook.replace('E F G E E,,', '^G F G E E,,'), data, lyrics, ['lantern'])
        self.assertIn('^G E\nw: lantern glow', accidental)
        # An octave boundary is inclusive; adjacent low notes keep their register.
        data['notes'] = [dict(start=float(i), end=i+.2, midi=m) for i, m in enumerate((64, 52, 64, 64, 64))]
        self.assertIn('1 low dip', M.feeling_hook(hook, data, [], []))
        data['notes'][2]['midi'] = 52
        self.assertNotIn('low dip', M.feeling_hook(hook, data, [], []))

    def test_hum_along_uses_saved_lengths_without_moving_syllables(self):
        song = example()
        # The proportional saved hook stores note lengths as tick spans. Gaps
        # in the tick sequence are separate from the lengths of the notes.
        melody = song['melody']['json']
        melody.update(rhythm=dict(mode='proportional', L='1/8'), hook=dict(line_indices=[0]),
                      lines=[dict(start=25., end=27., note_indices=[0, 1, 2, 3],
                                  lyrics='a lantern in a')])
        melody['notes'] = [
            dict(start=t, end=t + .1 * length, midi=midi, tick_start=tick, tick_end=tick + length)
            for t, midi, tick, length in ((25., 61, 0, 1), (25.2, 64, 2, 2),
                                          (25.6, 65, 6, 4), (26.2, 63, 12, 1))]
        melody['hook_text'] = (
            'MELODY: approximate vocal ABC · K:C\nK:C\n% [0:25] A lantern in a\n^C E F _E\nw: a lantern in a\n')
        first = L.render(song, 'first-listen')[0]
        self.assertIn('unmarked = short, 2 = longer, 4 = longest', first)
        self.assertIn('^C E2 F4 _E\nw: a lantern in a', first)
        for view in ('summary', 'full', 'melody'):
            self.assertNotIn('Approximate lengths', L.render(song, view)[0])

        del melody['notes'][2]['tick_end']
        legacy = L.render(song, 'first-listen')[0]
        self.assertNotIn('Approximate lengths', legacy)
        self.assertIn('^C E F _E\nw: a lantern in a', legacy)

    def test_default_order_placements_width_and_retained_details(self):
        song = example()
        old = copy.deepcopy(song)
        text, used = L.render(song, L.parser().parse_args([]).view)
        self.assertEqual(used, 'first-listen')
        order = [F.READ, 'THE ARC', 'THE SONG', 'WHERE TO LEAN IN', '\nHUM ALONG:',
                 '\nHOME', F.INVITATION, 'LISTENERS SAY', 'NEXT LISTEN']
        self.assertEqual(sorted(text.index(x) for x in order), [text.index(x) for x in order])
        self.assertIn('[1:01≈] We will keep a lantern by the door  ← 1:04 bass + drums enter', text)
        could = text.split('[2:33≈]', 1)[1].split('3:01', 1)[0]
        self.assertIn('← 2:33 chill candidate · appoggiatura:', could)
        self.assertIn('1:39  [bass + drums drop out]', text)
        self.assertIn('3:01  [guitar alone for 0:12]', text)
        self.assertIn('    0:00 ▁▂▃▄', text)
        self.assertIn('HUM ALONG: follow the ups and downs', text)
        self.assertIn('90 BPM · opens on I, ends on I (home)', text)
        self.assertTrue(all(len(r) <= NOTES_WIDTH for r in text.splitlines()))
        for unwanted in ('SOUND:', 'CPPS', 'HNR', 'NOTES', 'WORDS ', 'opening 0:00',
                         'LYRICS:', 'MELODY:', 'rhythm:', '% pitches and words only', 'comes home'):
            self.assertNotIn(unwanted, text)
        full = L.render(song, 'full')[0]
        for kept in ('SOUND:', 'CPPS', 'HNR', '807 Hz', '8.8', 'HUM ALONG:', 'invented provenance'):
            self.assertIn(kept, full)
        self.assertEqual(song, old)

    def test_cadences_require_explicit_confidence_not_notable_move_prose(self):
        for flag in (True, None, False):
            song = example()
            for row in song['cards']['harmony']['json']['cadences']:
                if flag is None:
                    row.pop('likely')
                else:
                    row['likely'] = flag
            song['cards']['harmony']['text'] += ('TIMELINE\n'
                '   1:39  drop-out: the level falls 7 dB and stays down; the band thins\n')
            text = L.render(song, 'first-listen')[0]
            if flag is not None:
                self.assertIn('1:39  [bass + drums drop out]  ← 1:39 IV→I', text)
                self.assertEqual('IV→I (likely)' in text, flag)
            else:
                self.assertNotIn('← 1:39 IV→I', text)
            self.assertNotIn('amen', text)
            self.assertIn("plagal cadence IV→I", L.render(song, 'full')[0])
        song['cards']['harmony']['json']['stage_status'] = [dict(stage='cadences', status='failed')]
        self.assertNotIn('← 1:39 IV→I', L.render(song, 'first-listen')[0])

    def test_meter_numbers_and_ambiguity_never_become_a_waltz(self):
        for consistency in (.89, .90, .94):
            text = F._meter('142.9 BPM', '3/4', dict(bar_consistency=consistency))
            self.assertIn('142.9 BPM', text)
            self.assertIn('≈48 bars a minute', text)
            self.assertEqual('a waltz' in text, consistency >= .90)
        for count, alternatives in (('three', '3/4 / 9/8'), ('4', '4/4 / 12/8')):
            for consistency in (.89, .90, .94):
                ambiguous = F._meter('142.9 BPM', 'split 2 vs 3 is ambiguous '
                                     f'(candidates {alternatives}) · confidence low (0.24)',
                                     dict(bar_consistency=consistency))
                self.assertIn('in ' + count, ambiguous)
                self.assertEqual(' or ' in ambiguous, consistency < .90)
                self.assertNotIn('waltz', ambiguous)
                self.assertNotIn('confidence', ambiguous)
        unclear = F._meter('152 BPM', 'meter unclear; meter changes unconfirmed', {})
        self.assertEqual(unclear, '152 BPM · meter unclear; meter changes unconfirmed')

    def test_vocal_pitch_notes_wait_for_measured_onset_or_lyric_fallback(self):
        song = example()
        singer = song['cards']['singer']
        singer['text'] = singer['text'].replace('SINGER:\n', 'SINGER:\n'
            '  highest held notes: D4 at 0:00 (0.7s) · E4 at 0:24 (1.1s) · F4 at 0:25 (1.0s)\n'
            '  pitch arc (median by section): A3 → F4 · the voice lifts ~17 semitones by 0:00\n')
        harmony = song['cards']['harmony']
        harmony['text'] = harmony['text'].replace('    2:33  F held',
            '    0:00  A held 0.5 s over C, then up onto B\n    2:33  F held')
        for measured in (True, False):
            with self.subTest(measured=measured):
                sample = copy.deepcopy(song)
                if measured:
                    # JSON retains a fractional onset that the card rounded down.
                    sample['cards']['singer']['json']['stems'] = dict(events=[dict(t=24.5, text='vocals enter')])
                else:
                    sample['cards']['singer']['text'] = singer['text'].replace('  0:24 vocals enter\n', '')
                    # Measured continuing presence is not a measured onset.
                    sample['cards']['singer']['text'] += '  1:40 everything drops out but vocals\n'
                full = L.render(sample, 'full')[0]
                text = L.render(sample, 'first-listen')[0]
                self.assertNotIn('0:00 chill candidate', text)
                self.assertNotIn('highest held note: D4', text)
                self.assertNotIn("voice's highest section", text)
                self.assertEqual('highest held note: E4' in text, not measured)
                self.assertIn('highest held note: F4', text)
                self.assertIn('2:33 chill candidate', text.split('WHERE TO LEAN IN')[1])
                self.assertIn('0:00  A held', full)
                self.assertEqual(full, L.render(sample, 'full')[0])
        song['cards']['singer']['json']['stems'] = dict(events=[dict(t=0, text='opens with other + vocals')])
        text = L.render(song, 'first-listen')[0]
        self.assertIn('0:00 chill candidate', text)
        self.assertIn("voice's highest section", text)
        singer['text'] = re.sub(r'  pitch arc[^\n]*\n', '', singer['text'])
        text = L.render(song, 'first-listen')[0]
        self.assertIn('highest held note: D4', text)
        self.assertEqual(F._vocal_changes([(2, 'drums + vocals enter'), (8, 'vocals + drums drop out'),
                                          (10, 'vocals return')]), [(2, True), (8, False), (10, True)])
        # The gate belongs to the vocal tracker, not to chord-derived chills.
        song = example()
        song['cards']['harmony']['text'] += '  new or unexpected harmony:\n    0:00  C→G (likely)\n'
        self.assertIn('0:00  [chill candidate · C→G (likely)]', L.render(song, 'first-listen')[0])

    def test_no_voice_requires_whole_line_silence_and_onset_slack(self):
        mark = ' (no voice measured here)'
        for onset, keep in ((25., False), (26.5, False), (28., False), (30., True)):
            with self.subTest(onset=onset):
                song = example()
                song['cards']['music']['text'] = song['cards']['music']['text'].replace(
                    'A lantern in a quiet room\n', 'A lantern in a quiet room' + mark + '\n')
                song['cards']['singer']['json']['stems'] = dict(events=[
                    dict(t=0., text='opens with other'), dict(t=onset, text='vocals enter')])
                full = L.render(song, 'full')[0]
                text = L.render(song, 'first-listen')[0]
                line = next(r for r in text.splitlines() if '[0:25≈]' in r)
                self.assertEqual(mark in line, keep)
                self.assertIn(mark, full)
        # A short line stamped before the onset gets the same 1.5 s allowance.
        self.assertFalse(F._silent_line(25., 25.5, [(0., False), (26.5, True)], []))
        self.assertFalse(F._silent_line(25., 29., [], [(24., 26.)]))
        self.assertTrue(F._silent_line(25., 29., [], [(24., 27.), (27., 30.)]))
        self.assertFalse(F._silent_line(25., 29., [], []))
        # Without a saved end, the next line supplies the boundary.
        for saved_end in (False, True):
            song = example()
            if not saved_end:
                del song['cards']['music']['json']['lyrics'][0]['end']
            song['cards']['music']['text'] = song['cards']['music']['text'].replace(
                'A lantern in a quiet room\n', 'A lantern in a quiet room' + mark + '\n')
            song['cards']['singer']['json']['stems'] = dict(events=[
                dict(t=0., text='opens with other'), dict(t=40., text='vocals enter')])
            text = L.render(song, 'first-listen')[0]
            line = next(r for r in text.splitlines() if '[0:25≈]' in r)
            self.assertEqual(mark in line, saved_end)
        self.assertTrue(F._silent_line(153., 158., [(150., False)], []))
        # Neither unavailable evidence nor a failed stage can confirm silence.
        for stages in ([], [dict(stage='stems', status='failed')], [dict(stage='gaps', status='failed')]):
            song = example()
            song['cards']['music']['text'] = song['cards']['music']['text'].replace(
                'A lantern in a quiet room\n', 'A lantern in a quiet room' + mark + '\n')
            singer = song['cards']['singer']
            singer['text'] = singer['text'].replace('  0:24 vocals enter\n', '')
            singer['json']['stage_status'] = stages
            singer['json']['singer'] = dict(data=dict(gaps=dict(silent=[[24., 30.]],
                stage_status=[dict(stage='gaps', status='failed')])))
            self.assertNotIn(mark, L.render(song, 'first-listen')[0])

    def test_full_saved_voice_marks_use_precise_windows_in_each_pass(self):
        mark = ' (no voice measured here)'
        song = example()
        song['manifest']['duration_s'] = 200.
        song['cards']['singer']['json']['stems'] = dict(events=[
            dict(t=0., text='opens with other'), dict(t=28., text='vocals enter'),
            dict(t=30., text='vocals drop out')])
        for index in (0, 1):
            row = song['cards']['music']['json']['lyrics'][index]
            row['no_voice'] = True
            song['cards']['music']['text'] = song['cards']['music']['text'].replace(
                row['text'] + '\n', row['text'] + mark + '\n')
        full = L.render(song, 'full')[0]
        self.assertNotIn('A lantern in a quiet room' + mark, full)
        self.assertIn('We will keep a lantern by the door' + mark, full)
        self.assertTrue(song['cards']['music']['json']['lyrics'][0]['no_voice'])

    def test_concrete_solo_replaces_only_the_same_other_open_within_three_seconds(self):
        for start, stem, energy_only, opening, keep in (
                (2., 'other', False, 'other', False), (3., 'other', False, 'other', False),
                (3.1, 'other', False, 'other', True), (2., 'guitar', False, 'other', True),
                (2., 'other', True, 'other', True), (2., 'other', False, 'other + vocals', True)):
            with self.subTest(start=start, stem=stem, energy_only=energy_only, opening=opening):
                song = example()
                singer = song['cards']['singer']
                singer['text'] += f'  0:00 opens with {opening}\n'
                singer['json']['instruments'] = dict(families={'acoustic guitar': dict(
                    stem=stem, energy_only=energy_only, conf=.42, spans=[[start, 14.]])})
                song['cards']['harmony']['text'] += f'TIMELINE\n   0:{start:04.1f}  acoustic guitar alone for 0:14\n'
                text = L.render(song, 'first-listen')[0]
                self.assertEqual('opens with' in text, keep)
                self.assertIn('acoustic guitar alone', text)
                self.assertIn('opens with', L.render(song, 'full')[0])
        # A tag beginning right at 0:00 may already have named the opening.
        singer['text'] = singer['text'].replace('opens with other + vocals', 'opens with other')
        singer['json']['instruments']['families']['acoustic guitar']['spans'] = [[0., 14.]]
        self.assertNotIn('opens with', L.render(song, 'first-listen')[0])

    def test_phrase_end_denominator_names_judged_ends(self):
        for resolves, expected in (
                ('2 of 3 phrase ends likely land on I (2 authentic). 18 too blurry to call.',
                 '2 of 3 judged phrase ends likely land on I (2 V→I)'),
                ('10 of 10 phrase ends likely land on I (10 plagal).',
                 '10 of 10 judged phrase ends likely land on I, all by IV→I')):
            song = example()
            song['cards']['harmony']['text'] += 'RESOLVES: ' + resolves + '\n'
            text = slim.clean(L.render(song, 'first-listen')[0])
            self.assertIn(expected, text)
            self.assertNotIn('phrase ends: ', text)
            self.assertIn('RESOLVES: ' + resolves, L.render(song, 'full')[0])

    def test_texture_pitch_and_held_notes_keep_the_measured_section(self):
        song = example()
        song['cards']['music']['text'] = song['cards']['music']['text'].replace(
            'VOCAL: clearer [', 'VOCAL: clearer 0:24–1:05; breathier 1:30–2:00 [')
        song['cards']['singer']['text'] = song['cards']['singer']['text'].replace('SINGER:\n',
            'SINGER:\n  pitch arc (median by section): A3 → E3 → B3 → G3 → F4 → G3 → C3'
            ' · the voice lifts ~17 semitones by 1:54\n'
            '  highest held notes: B♭4 at 2:11 (0.6s, pushed hard)\n')
        text = L.render(song, 'first-listen')[0]
        self.assertIn('← 0:24 voice sounds clearer (relative)', text)
        self.assertIn('1:30  [voice sounds breathier (relative)]', text)
        self.assertIn("1:54  [the voice's highest section", text)
        self.assertIn('median F4, ~17 semitones above its lowest section', text)
        self.assertNotIn('highest of the song', text)
        self.assertIn('2:11  [highest held note: B♭4 (pushed hard)]', text)
        song['cards']['music']['json']['stage_status'] = [dict(stage='vocal quality', status='failed')]
        song['cards']['singer']['json']['stage_status'] = [dict(stage='pitch', status='failed')]
        text = L.render(song, 'first-listen')[0]
        self.assertNotIn('voice sounds clearer', text)
        self.assertNotIn('highest', text)

    def test_uncertain_vocal_texture_uses_a_completed_stem_measurement(self):
        song = example()
        music = song['cards']['music']
        music['text'] = music['text'].replace(
            'VOCAL: clearer [CPPS 8.8, HNR 13 dB]',
            'VOCAL: no clear contrast in breathiness between sections [CPPS 10.3–11.9, HNR 16 dB median]')
        music['json']['vocal_quality'] = dict(ok=True, engine='parselmouth')
        text = L.render(song, 'first-listen')[0]
        self.assertIn('VOICE TEXTURE   breathiness contrast inconclusive across measured sections (vocal stem)', text)
        self.assertNotIn('VOICE TEXTURE', L.render(song, 'summary')[0])
        music['json']['vocal_quality']['engine'] = 'numpy (approximate)'
        self.assertNotIn('VOICE TEXTURE', L.render(song, 'first-listen')[0])
        music['json']['vocal_quality']['engine'] = 'parselmouth'
        music['json']['stage_status'] = [dict(stage='vocal quality', status='skipped')]
        self.assertNotIn('VOICE TEXTURE', L.render(song, 'first-listen')[0])

    def test_spaces_in_gaps_are_not_attached_to_the_next_sung_line(self):
        song = example()
        singer = song['cards']['singer']
        # Same nested shape and times as Roommates' saved silence and stop.
        singer['json']['singer'] = dict(data=dict(gaps=dict(have_words=True,
            silent=[[139.264, 139.936]], stage_status=[dict(stage='gaps', status='ran')])))
        singer['json']['stops'] = dict(status='ran', events=[dict(start_s=139.9,
            kind='uncertain', voice_behavior='band cuts, vocal stem enters',
            stem_drop_db=dict(drums=40., bass=52.5, other=11.5))])
        text = L.render(song, 'first-listen')[0]
        self.assertIn('2:19  [a breath]', text)
        self.assertIn('2:20  [band drop-out candidate; vocal stem enters]', text)
        self.assertNotIn('voice alone', text)
        singer['text'] += '  2:19 everything drops out but vocals\n'
        self.assertIn('[band drops out, voice alone]', L.render(song, 'first-listen')[0])
        singer['json']['stage_status'] = [dict(stage='gaps', status='failed'), dict(stage='stops', status='failed')]
        text = L.render(song, 'first-listen')[0]
        self.assertNotIn('a breath', text)
        self.assertNotIn('band drop-out candidate', text)

    def test_space_priority_beats_swell_but_yields_to_entrance_and_chill(self):
        song = example()
        singer = song['cards']['singer']
        singer['json']['singer'] = dict(data=dict(gaps=dict(have_words=True, silent=[[154., 154.5]])))
        text = L.render(song, 'first-listen')[0].split('[2:33≈]', 1)[1].split('3:01', 1)[0]
        self.assertIn('a breath', text)
        self.assertIn('chill candidate', text)
        self.assertNotIn('vocals swell', text)
        singer['text'] += '  2:34 bass enter (possibly)\n'
        text = L.render(song, 'first-listen')[0].split('[2:33≈]', 1)[1].split('3:01', 1)[0]
        self.assertIn('bass enter', text)
        self.assertIn('chill candidate', text)
        self.assertNotIn('a breath', text)

    def test_priority_and_long_note_continuations(self):
        song = example()
        song['cards']['singer']['text'] += '  2:34 bass enter (possibly)\n'
        text = L.render(song, 'first-listen')[0]
        section = text.split('[2:33≈]', 1)[1].split('3:01', 1)[0]
        self.assertEqual(section.count('←'), 2)
        self.assertIn('bass enters (possibly)', section)
        self.assertIn('chill candidate', section)
        self.assertNotIn('swells', section)
        self.assertNotIn('most intense moment', section)
        self.assertRegex(section, r'\n    ← 2:33 chill candidate')
        self.assertTrue(all(len(row) <= NOTES_WIDTH for row in text.splitlines()))

    def test_qualifiers_survive_and_failed_passes_and_stages_supply_no_events(self):
        song = example()
        song['cards']['singer']['text'] = song['cards']['singer']['text'].replace(
            'arousal 0.69', 'two voices high + low? · arousal 0.69')
        text = L.render(song, 'first-listen')[0]
        self.assertIn('two voices high + low?', text)
        for name in ('singer', 'harmony'):
            failed = copy.deepcopy(song)
            failed['manifest']['passes'][name] = dict(status='failed', reason='invented failure')
            text = L.render(failed, 'first-listen')[0]
            self.assertIn(name + ': failed: invented failure', text)
            self.assertNotIn('bass + drums enter' if name == 'singer' else 'chill candidate', text)
        song['cards']['singer']['json']['stage_status'] = [dict(stage='stems', status='skipped')]
        song['cards']['harmony']['json']['stage_status'] = [dict(stage='chills', status='failed')]
        text = L.render(song, 'first-listen')[0]
        self.assertNotIn('bass + drums', text)
        self.assertNotIn('chill candidate', text)

    def test_stage_failures_and_key_change_ranges_keep_their_policy(self):
        song = example()
        song['cards']['harmony']['json']['stage_status'] = [dict(stage='appoggiaturas', status='failed')]
        self.assertNotIn('appoggiatura:', L.render(song, 'first-listen')[0])
        song['cards']['harmony']['json']['key_change_scan'] = dict(status='ran', claims=[
            dict(from_key='C major', to='G major', range=[60.2, 66.8], move='up a fifth')])
        text = L.render(song, 'first-listen')[0]
        self.assertIn('KEY CHANGES: C major → G major, somewhere 1:00–1:07', text)
        song['cards']['harmony']['json']['stage_status'].append(dict(stage='key changes', status='failed'))
        self.assertNotIn('KEY CHANGES:', L.render(song, 'first-listen')[0])

    def test_wrapped_event_qualifiers_and_wordless_peak_keep_the_caveat(self):
        song = example()
        song['cards']['singer']['text'] += '  1:59 vocals return\n    two voices? (likely)\n'
        text = L.render(song, 'first-listen')[0]
        self.assertIn('vocals return two voices? (likely)', text)
        song['cards']['harmony']['text'] += ('TIMELINE\n'
            '   2:33  2:33–2:38: wordless singing (lyrics: "La la")\n')
        text = L.render(song, 'first-listen')[0]
        lean = text.split('WHERE TO LEAN IN', 1)[1].split('HUM ALONG', 1)[0]
        self.assertIn('wordless most intense moment (speech-trained texture, not a verdict)', lean)

    def test_short_notes_keep_confidence_and_question_words(self):
        self.assertEqual(F._short_note('harp enters (likely)  [instrument tagger]'), 'harp enters (likely)')
        self.assertEqual(F._short_note('keys (piano or organ?) enters (possibly)  [instrument tagger]'),
                         'keys (piano or organ?) enters (possibly)')
        self.assertEqual(F._short_note('chill candidate · appoggiatura: F held 0.2 s over C, then up onto G'),
                         'chill candidate · appoggiatura: F→G over C (0.2 s)')
        self.assertEqual(F._short_note('two voices high + low?'), 'two voices high + low?')
        self.assertEqual(F._short_note('vocals swells 18 dB · drums thins 9 dB'),
                         'vocals swell 18 dB · drums thin 9 dB')
        self.assertEqual(F._short_note('chill candidate · timbre shift (-0 dB)'),
                         'chill candidate · timbre shift (0 dB)')
        self.assertEqual(F._short_note('level change -0.3 dB'), 'level change 0 dB')
        self.assertEqual(F._event_key('woodwinds (likely) enter'),
                         F._event_key('woodwinds enters (likely)  [instrument tagger]'))
        self.assertNotEqual(F._event_key('woodwinds enter (likely)'),
                            F._event_key('woodwinds enter (possibly)'))
        self.assertEqual(F._event_key('swell +9 dB over about 2.5 s'),
                         F._event_key('swell: the sound grows 9 dB in a couple of seconds'))

    def test_whisper_marks_and_instrumental_event_rows(self):
        song = example()
        song['cards']['music']['text'] = ("MUSIC: invented.wav · 3:20\nLYRICS: words are Whisper's guesses\n"
                                          '  [1:01≈] a lantern (unsure)\n')
        song['cards']['music']['json'] = {}
        text = L.render(song, 'first-listen')[0]
        self.assertIn('[1:01≈] a lantern (unsure)', text)
        for card in song['cards'].values():
            card['json']['lyrics_source'] = dict(source='LRCLIB', status='instrumental')
        text = L.render(song, 'first-listen')[0]
        self.assertIn('THE SONG', text)
        self.assertNotRegex(text.split('THE SONG')[1].split('HUM ALONG')[0], r'\[\d+:\d+≈?\]')
        self.assertIn('[bass + drums enter]', text)

    def test_changed_word_is_exact_and_abstains(self):
        lines = [slim.Lyric(i*20, f'We {word} keep a lantern by the door')
                 for i, word in enumerate(('will', 'would', 'could'))]
        self.assertIn('will → would → could', F._changed_word(lines))
        self.assertIn('in “We … keep a lantern by the door”', F._changed_word(lines))
        self.assertEqual(F._changed_word([slim.Lyric(0, 'Title'), *lines]), F._changed_word(lines))
        self.assertEqual(F._changed_word([lines[0], slim.Lyric(20, 'We could leave a lantern by the door')]), '')
        self.assertEqual(F._changed_word([lines[0], slim.Lyric(20, lines[1].text + ' (unsure)')]), '')
        self.assertEqual(F._changed_word([lines[0], lines[0]]), '')
        self.assertEqual(F._changed_word([slim.Lyric(0, 'Oh yes'), slim.Lyric(20, 'Oh no')]), '')

    def test_old_bundles_and_long_abc_stay_read_only(self):
        song = example()
        song['cards']['singer']['json'] = None
        song['cards']['harmony']['json'] = None
        song['melody']['json']['hook_text'] += ' '.join(['C']*100) + '\nw: ' + ' '.join(['glow']*100) + '\n'
        old = copy.deepcopy(song)
        with patch.object(L, 'run_pass', side_effect=AssertionError('no models')), \
             patch.object(L.subprocess, 'run', side_effect=AssertionError('no subprocess')), \
             patch('socket.create_connection', side_effect=AssertionError('no network')):
            text = L.render(song, 'first-listen')[0]
        self.assertTrue(all(len(r) <= NOTES_WIDTH for r in text.splitlines()))
        notes = text.split('K:C\n', 1)[1].split('\nHOME', 1)[0].splitlines()
        for i, row in enumerate(notes):
            if row.startswith('w:'):
                self.assertEqual(len(row[3:].split()), len(notes[i-1].split()))
        self.assertEqual(song, old)

    def test_picture_pointer_width_and_same_target(self):
        target = Path.home() / '.cache/raven/cards/songs' / ('a' * 64) / 'listening-sheet.png'
        pointer = F.picture_line('PICTURE: ' + str(target))
        self.assertTrue(all(len(r) <= NOTES_WIDTH for r in pointer.splitlines()))
        displayed = ''.join(pointer.removeprefix('PICTURE:').split())
        self.assertEqual(Path(displayed).expanduser().resolve(), target.resolve())
        self.assertEqual(F.picture_line('PICTURE: sheet.png'), 'PICTURE: sheet.png')

    def test_picture_keeps_the_same_decided_evidence(self):
        import listening_sheet
        song = example()
        full = L.full_text(song)
        for quick in (False, True):
            song['manifest']['options']['quick'] = quick
            self.assertEqual(listening_sheet.song_evidence(song, L.render(song, 'first-listen')[0], full),
                             listening_sheet.song_evidence(song, L.render(song, 'summary')[0], full))

    def test_committed_and_optional_real_bundles_conserve_every_legacy_line(self):
        paths = sorted((L.HERE / 'fixtures').glob('*song.json'))
        if BUNDLES:
            paths += sorted(BUNDLES.glob('*.json'))
        for path in paths:
            with self.subTest(bundle=path.name):
                song = json.loads(path.read_text())
                before = copy.deepcopy(song)
                saved = path.read_bytes(), path.stat().st_mtime_ns
                with patch.object(L, 'run_pass', side_effect=AssertionError('no model passes')), \
                     patch.object(L.subprocess, 'run', side_effect=AssertionError('no subprocess')), \
                     patch('urllib.request.urlopen', side_effect=AssertionError('no network')):
                    summary = L.render(song, 'summary')[0]
                    feeling = L.render(song, 'first-listen')[0]
                    full = L.render(song, 'full')[0]
                if path.stem in BASELINE:
                    self.assertEqual(summary, BASELINE[path.stem])
                if FULL_BASELINE and (FULL_BASELINE / (path.stem + '.full.txt')).exists():
                    self.assertEqual(full, (FULL_BASELINE / (path.stem + '.full.txt')).read_text())
                legacy = BASELINE.get(path.stem, summary)
                available = slim.clean(feeling + '\n' + full)
                for row in legacy.splitlines():
                    if row.strip():
                        self.assertIn(slim.clean(row), available, f'{path.name}: {row}')
                self.assertEqual(feeling, L.render(song, 'first-listen')[0])
                self.assertTrue(all(len(r) <= NOTES_WIDTH for r in feeling.splitlines()))
                self.assertEqual(song, before)
                self.assertEqual(saved, (path.read_bytes(), path.stat().st_mtime_ns))
                if path.stem == '81077acc':
                    self.assertIn('1:39 IV→I (likely)', feeling)
                    self.assertIn('2:35 IV→I (likely)', feeling)
                    self.assertIn('10 of 10 judged phrase ends likely land on I, all by IV→I',
                                  slim.clean(feeling))
                    self.assertIn('142.9 BPM in three (≈48 bars a minute)', feeling)
                    self.assertIn('words: longing (Jev .74) · chord: I, major', feeling)
                    self.assertIn('← 0:24 voice sounds clearer (relative)', feeling)
                    self.assertIn("← 1:54 the voice's highest section", feeling)
                    self.assertIn('~17 semitones above its lowest section', feeling)
                    self.assertIn('← 2:11 highest held note: B♭4 (pushed hard)', feeling)
                    self.assertNotIn('opens with', feeling)
                    self.assertIn('[acoustic guitar swells 10 dB]', feeling)
                    self.assertNotIn('-0 dB', feeling)
                    self.assertNotIn('a waltz', feeling)
                    self.assertNotIn('comes home', feeling)
                    self.assertNotIn('MELODY:', feeling)
                    self.assertNotIn('rhythm:', feeling)
                    self.assertNotIn('LYRICS:', feeling)
                    self.assertIn('LISTENERS SAY (Last.fm tags): soul, folk, haunting, ethereal', feeling)  # legacy row kept
                elif path.stem == '9556bf75':
                    self.assertIn('2:19  [a breath]', feeling)
                    self.assertIn('[band drop-out candidate; vocal stem enters]', feeling)
                    self.assertNotIn('voice alone', feeling)
                    self.assertLess(feeling.index('[band drop-out candidate'), feeling.index('[2:20≈]'))
                elif path.stem == '3b870372':
                    self.assertNotIn('0:00 chill candidate', feeling)
                    self.assertNotIn('no voice measured here', feeling)
                    self.assertNotIn('opens with', feeling)
                    self.assertIn('0:02  [acoustic guitar alone for 0:14]', feeling)
                    self.assertIn('2:00 chill candidate', feeling.split('WHERE TO LEAN IN')[1])
                    self.assertIn('120 BPM in 4 (≈30 bars a minute)', feeling)
                    self.assertIn('2 of 3 judged phrase ends likely land on I (2 V→I)', slim.clean(feeling))
                    self.assertIn('0:00  A held 0.5 s over B♭', full)
                    self.assertEqual(full.count('no voice measured here'), 9)
                    self.assertIn('candidates 4/4 / 12/8', full)
                elif path.stem == '3785af48':
                    self.assertIn('152 BPM · meter unclear; meter changes unconfirmed', feeling)
                    self.assertIn('sentiment -0.69 (roberta) · chord: I, major', feeling)
                    self.assertIn('0:36  [guitar enters]', feeling)
                    self.assertNotIn('steps up 32 dB', feeling)
                elif path.stem == 'a97d2cd8':
                    self.assertIn('← 2:38 guitar enters', feeling)
                    self.assertNotIn('+56 dB', feeling)
                    self.assertIn('guitar steps up 8 dB', feeling)
                    self.assertNotIn('(...)', feeling)
                    self.assertNotIn('no voice measured here', feeling)
                    self.assertIn('kakau', feeling)  # legacy names, no counts saved: shown as saved; a fresh listen applies the filter
                    self.assertIn('kakau', summary)
                    self.assertIn('157.9 BPM in 4', feeling)
                    self.assertNotIn('accents repeat', feeling)
                    self.assertIn('accents repeat', full)
                    hum = feeling.split('HUM ALONG:')[1].split('HOME')[0]
                    self.assertIn('% [1:08≈]', hum)
                    self.assertNotIn('headphones', hum)
                    self.assertNotIn('G,,', hum)
                    self.assertIn('% 1 low dip left out', hum)
                elif path.stem == '2eedf964':
                    self.assertIn('0:38  [a breath]', feeling)
                    self.assertNotIn('0:38 chill candidate', feeling)
                    self.assertNotIn('appoggiatura:', feeling)
                    self.assertIn('0:38  C held', full)
                    self.assertNotIn('0:01 ', feeling)
                    hum = feeling.split('HUM ALONG:')[1].split('HOME')[0]
                    self.assertIn('% 1 low dip left out', hum)
                    self.assertNotIn('A,,', hum)
                    comment = next(r for r in hum.splitlines() if r.startswith('% ['))
                    supplied = next(r for r in song['cards']['music']['json']['lyrics'] if r['t'] == 101.68)
                    self.assertEqual(comment.split('] ', 1)[1], supplied['text'])
                elif path.stem == 'dd486fa5':
                    line = next(r for r in feeling.splitlines() if 'woodwinds (likely) enters' in r)
                    self.assertEqual(line.count('woodwinds'), 1)
                elif path.stem == 'b279d540':
                    line = next(r for r in feeling.splitlines() if '2:52  [swell' in r)
                    self.assertEqual(line.count('swell'), 1)


def run_tests(check):
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(FeelingTests).run(result)
    check('feeling card: weaving, confidence, width, saved evidence and conservation', result.wasSuccessful(),
          '\n'.join(detail for test, detail in result.failures + result.errors))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundles', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--full-baseline', type=Path)
    args, rest = parser.parse_known_args()
    BUNDLES = args.bundles
    BASELINE = json.loads(args.baseline.read_text()) if args.baseline else {}
    FULL_BASELINE = args.full_baseline
    unittest.main(argv=[sys.argv[0], *rest])
