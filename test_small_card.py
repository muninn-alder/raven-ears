"""Four narrow saved-card regressions; invented evidence, no audio or services."""
import copy
import unittest
import os
from unittest.mock import patch

import lyrics_shared as LS
import meter_trust
import slim
import stage_status as ST


class SmallCard(unittest.TestCase):
    def test_lyric_source_stays_with_the_pass_that_supplied_the_words(self):
        missed = "LYRICS: LRCLIB had no matching lyrics; words are Whisper's guesses (rough times)"
        matched = "LYRICS: LRCLIB synced lyrics; supplied times (≈)"
        row = dict(t=4.0, end=5.0, text="An invented lantern", approx=False)
        cards = {"music": f"{missed}\n  [0:04] {row['text']}\n",
                 "singer": f"{matched}\n  [0:04] {row['text']}\n"}
        singer = {"lyrics": dict(rows=[row], label=matched, lookup=dict(source="LRCLIB", status="synced"))}
        out = slim.render_first_listen(cards, {"singer": singer})
        self.assertTrue(next(line for line in out.splitlines() if line.startswith("LYRICS:")).startswith(missed))
        self.assertIn("Whisper's guesses", next(line for line in slim.render(cards["music"]).splitlines()
                                               if line.startswith("WORDS (")))
        music = {"lyrics": [row], "lyrics_source": dict(source="LRCLIB", status="nomatch")}
        cards["music"] = cards["music"].replace(missed, matched)
        out = slim.render_first_listen(cards, {"music": music, "singer": singer})
        self.assertIn("LYRICS: words are Whisper's guesses (rough times) [music]", out)
        self.assertIn("lyrics from LRCLIB", slim.render("", analysis=singer))
        self.assertIn(row["text"], slim.render("", analysis=singer))

    def test_fallback_tempo_names_both_sources_in_fresh_and_saved_views(self):
        import listen
        import music
        import tempo_trust
        # Lover's real row shape: the tracker has a pulse, but only 76% of
        # its bars agree on the beat count. Invented tempo values differ.
        meter = dict(engine='beat_this', tracker_bpm=91.2, beats_per_bar=3,
                     bar_consistency=.76, bar_s=1.50, verdict='meter unclear', confidence=.39)
        result = dict(file='invented.mp3', duration=120,
                      tempo=dict(bpm=123, regularity=.81, beat_strength=3.87))
        old = 'MUSIC: invented.mp3 · 2:00\n  TEMPO:    123 BPM · steady 0.81 · beat strength 3.87x\n'
        song = {'cards': {'music': {'text': old, 'json': result},
                          'singer': {'text': 'METER: meter unclear', 'json': {'meter': meter}},
                          'harmony': {'text': '', 'json': {}}},
                'manifest': {'passes': {n: {'status': 'ran', 'reason': ''}
                                        for n in ('music', 'singer', 'harmony')}}}
        original = copy.deepcopy(song)
        full = listen.render(song, 'full')[0]
        feel = listen.render(song, 'summary')[0]
        for rendered in (music.format_card(result, meter), full, feel):
            line = next(l.strip() for l in rendered.splitlines() if l.strip().startswith('TEMPO:'))
            self.assertIn('123 BPM (librosa estimate)', line)
            self.assertIn('beat tracker heard 91.2 BPM (bars steady 0.76)', line)
        self.assertEqual(song, original)
        self.assertEqual(tempo_trust.tempo_text(full, 123, meter), full)
        for stage in ('meter', 'beat tracker'):
            song['cards']['singer']['json']['stage_status'] = [ST.outcome(stage, 'failed', 'invented')]
            for view in ('full', 'summary'):
                text = listen.render(song, view)[0]
                self.assertIn('123 BPM (librosa estimate)', text)
                self.assertNotIn('beat tracker heard', text)

    def test_librosa_only_and_section_tracker_estimates(self):
        import music
        import tempo_trust as TT
        line = '123 BPM · steady 0.81'
        for meter in (None, {}, {'engine': 'beat_this', 'tracker_bpm': float('nan')},
                      {'engine': 'beat_this', 'tracker_bpm': -1}):
            self.assertEqual(TT.estimate_line(line, meter), '123 BPM (librosa estimate) · steady 0.81')
        data = {'engine': 'beat_this', 'sections': [
            {'data': {'tracker_bpm': bpm}} for bpm in (91.2, 96.8)]}
        self.assertIn('beat tracker heard 91.2–96.8 BPM across sections', TT.estimate_line(line, data))
        data['whole_song_data'] = {'engine': 'beat_this', 'tracker_bpm': 94, 'bar_consistency': .7}
        self.assertIn('beat tracker heard 94 BPM (bars steady 0.70)', TT.estimate_line(line, data))
        fresh = music.format_card(dict(file='invented.mp3', duration=120,
                                       tempo=dict(bpm=123, regularity=.81, beat_strength=3.87)))
        for text in (fresh, slim.render_first_listen({'music': 'TEMPO: ' + line})):
            self.assertIn('123 BPM (librosa estimate)', text)
            self.assertNotIn('beat tracker heard', text)

    def test_tempo_wording_through_music_worker_and_song_save(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import listen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / 'invented.mp3'
            audio.touch()
            (root / 'songs').mkdir()
            bootstrap = '''
import sys
import numpy as np
import harmony, music, listen, raven_ears_plus as E
E.load_audio = lambda *args: np.zeros(10 * music.SR)
harmony.music_features = lambda *args: dict(chroma=None, rms_db=None, oenv=None, mfcc=None, hp_energy=None, cent=None)
music.analyze_tempo = lambda *args, **kwargs: dict(bpm=123, regularity=.81, beat_strength=3.87)
for name in ('analyze_key', 'analyze_sections', 'analyze_sound', 'analyze_vocals', 'analyze_vocal_quality', 'key_claim_for'):
    setattr(music, name, lambda *args, **kwargs: None)
music.meter_info_for = lambda *args: (None, dict(engine='beat_this', tracker_bpm=91.2, beats_per_bar=3, bar_consistency=.76))
raise SystemExit(listen.main(sys.argv[1:]))
'''
            run = subprocess.run
            def child(command, **kwargs):
                return run([sys.executable, '-c', bootstrap, *command[2:]], **kwargs)
            args = listen.parser().parse_args([str(audio), '--no-lyrics', '--no-instruments'])
            with patch.dict(os.environ, {'RAVEN_EARS_HOME': directory}), \
                 patch.object(listen.subprocess, 'run', side_effect=child):
                card = listen.run_pass('music', args)
            self.assertEqual(card['status'], 'ran', card['reason'])
            path = root / 'song.json'
            listen.save(path, {'cards': {'music': card}})
            saved = json.loads(path.read_text())['cards']['music']
            self.assertIn('123 BPM (librosa estimate)', saved['text'])
            self.assertIn('beat tracker heard 91.2 BPM', saved['text'])
            self.assertEqual(saved['json']['tempo']['bpm'], 123)
            self.assertEqual(saved['json']['file'], 'invented.mp3')

    def test_feel_tempo_keeps_numbers_and_qualifiers_without_arithmetic(self):
        section = dict(engine='beat_this', tracker_bpm=174, beats_per_bar=4,
                       bar_consistency=.95, bar_s=1.38, verdict='4/4', confidence=.9)
        meter = dict(sections=[dict(start=0, end=88, data=section),
                               dict(start=88, end=100, data=dict(section, beats_per_bar=2))])
        import tempo_trust
        reading = tempo_trust.lead_reading(87, meter)
        full = tempo_trust.lead_line(reading)
        self.assertIn('4 beats × 60 / 1.38s; 4-beat sections cover 88%', full)
        out = slim.render_first_listen({'music': 'TEMPO: 87 BPM'},
                                      {'music': {'tempo': {'bpm': 87}}, 'singer': {'meter': meter}})
        self.assertIn('TEMPO: 173.9 BPM', out)
        self.assertIn('librosa heard 87', out)
        self.assertNotIn('× 60', out)
        self.assertNotIn('sections cover', out)
        self.assertIn('may be double time', slim._feel_tempo(full + ' · may be double time'))
        self.assertIn('× 60', full)

    def test_gap_omissions_count_once_and_keep_original_bounds(self):
        dropped = set()
        spans = ([(0, 11), (20, 21), (39, 50)],)
        parts = LS.gap_segments(10, 40, spans, omitted=dropped)
        self.assertEqual(dropped, {(10, 11), (39, 40)})
        self.assertEqual([(a, b) for a, b, _ in parts], [(11, 20), (20, 21), (21, 39)])
        old = ('TIMELINE\n  0:10  the words stop; wordless for 0:30, until 0:40\n')
        evidence = 'VOCALS: yes · 0:00–0:11, 0:39–0:50\n'
        fixed = LS.clean_gap_claims(old, evidence)
        note = '2 short voice fragments (<2 s) at a gap edge not shown'
        self.assertEqual(fixed.count(note), 1)
        self.assertEqual(LS.clean_gap_claims(fixed, evidence), fixed)
        self.assertIn(note, slim.render_first_listen({'music': evidence, 'harmony': old}))
        self.assertEqual(LS.gap_edge_note({(10, 11)}),
                         '1 short voice fragment (<2 s) at a gap edge not shown')
        self.assertNotIn('not shown', LS.clean_gap_claims(old, 'VOCALS: yes · 0:00–0:50'))

    def test_new_timeline_records_one_note_when_built_twice(self):
        import harmony as H
        fake = {key: [] for key in ("secs", "spans", "evs", "moves", "pedals", "br", "dips", "drops", "swells", "con", "tag", "csurp")}
        fake.update(tc={}, dur=40, kinfo={}, words=[("paper", 0, 10), ("moon", 39, 40)],
                    lines=[], vocal_activity=[(0, 11)])
        @ST.engine
        def run():
            H.build_timeline(fake)
            H.build_timeline(fake)
            return {}
        with patch.dict(os.environ, {"RAVEN_NO_INSTRUMENTS": "1"}):
            rows = run()['stage_status']
        self.assertEqual(len([r for r in rows if r['stage'] == 'gap edges']), 1)
        note = ST.summary(rows)
        self.assertEqual(note.count('1 short voice fragment (<2 s) at a gap edge not shown'), 1)

    def test_instrumental_word_absence_preserves_backend_errors_and_saved_rows(self):
        rows = [ST.outcome('Whisper', 'failed', 'words unavailable'),
                ST.outcome('words', 'failed', 'Whisper words unavailable', kind='input'),
                ST.outcome('Whisper', 'failed', 'RuntimeError: decoder broken')]
        original = copy.deepcopy(rows)
        singer = 'SINGER: instrumental (low stem energy)\n'
        text = '\n'.join(ST.add_notes([], rows))
        fixed = ST.instrumental_text(text, singer)
        items = ST.card_notes(fixed, {'stage_status': rows})
        items = ST.display_items(singer, items)
        self.assertIn('words: no words to hear (instrumental)', items)
        self.assertIn('Whisper: failed: RuntimeError: decoder broken', items)
        self.assertNotIn('words unavailable', ' '.join(items))
        self.assertEqual(rows, original)
        self.assertEqual(ST.instrumental_text(text), text)
        quote = '[0:10] "words: failed: Whisper words unavailable"'
        self.assertEqual(ST.instrumental_text(quote, singer), quote)
        self.assertEqual(ST.instrumental_text('  unavailable (no usable lyric lines, see NOTES)', singer),
                         '  no words to hear (instrumental)')
        failed = slim.render_first_listen({'music': text, 'singer': singer},
                                         pass_statuses={'singer': {'status': 'failed', 'reason': 'invented'}})
        self.assertIn('words unavailable', failed)
        out = slim.render_first_listen({'music': text, 'singer': singer}, {'music': {'stage_status': rows}})
        self.assertIn('LYRICS: no words to hear (instrumental)', out)
        self.assertNotIn('words unavailable', out)
        self.assertIn('decoder broken', out)

    def test_feel_meter_quotes_full_card_authority_and_stage_gates(self):
        meter = {'verdict': '5/4 (2+3)', 'confidence': .9}
        cards = {'singer': 'SINGER: instrumental\nMETER: 5/4 (2+3) · confidence high\n'}
        data = {'singer': {'meter': meter}}
        out = slim.render_first_listen(cards, data)
        self.assertIn('METER: ' + meter_trust.assess(meter)[0] + '\n', out)
        for failed in ('meter', 'beat tracker'):
            data['singer']['stage_status'] = [ST.outcome(failed, 'failed', 'invented error')]
            self.assertNotIn('\nMETER:', slim.render_first_listen(cards, data))
        data = {'singer': {'meter': dict(meter, confidence=.1)}}
        self.assertNotIn('\nMETER:', slim.render_first_listen(cards, data))
        self.assertNotIn('\nMETER:', slim.render_first_listen(cards))


if __name__ == '__main__':
    unittest.main()
