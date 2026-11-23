"""Regression checks for honest cards; no real audio, cache writes or network."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import stage_status as S


class HonestBugs(unittest.TestCase):
    def test_voice_marker_stays_outside_quoted_lyrics(self):
        from lyrics_shared import supplied_gap_text, supplied_lyrics
        from slim import parse

        text = ('LYRICS: LRCLIB synced lyrics\n'
                '  [0:09≈] "Tin bell" (no voice measured here)\n'
                '  [0:14≈] "Paper moon"\n')
        _, rows = supplied_lyrics(text=text)
        self.assertEqual([(r["text"], r["no_voice"]) for r in rows],
                         [('"Tin bell"', True), ('"Paper moon"', False)])
        self.assertEqual(parse(text).lyrics[0].quote(),
                         '[0:09≈] “"Tin bell"” (no voice measured here)')
        gap = '  0:09  0:09–0:19: voice present, words not heard (Whisper found none)\n'
        projected = supplied_gap_text(gap, rows)
        self.assertIn('("\'Paper moon\'")', projected)
        self.assertNotIn("Tin bell", projected)
        stale = ('  0:09  0:09–0:19: voice present; lyrics have words here Whisper didn\'t hear '
                 '(""Tin bell" (no voice measured here)")\n')
        self.assertEqual(supplied_gap_text(stale, rows), projected)
        self.assertEqual(supplied_gap_text(stale, rows[:1]), gap)

    def test_repeated_vocables_require_only_explicit_syllables(self):
        from lyrics_shared import is_vocable
        for text in ('Eeh-eeh-eeh-eeh, eeh-eeh-eeh', 'Ooh ooh', 'Ah, ah, la-la',
                     'Na na na', 'Mm-mm', 'Oh oh', 'Hey hey', 'Yeah yeah'):
            with self.subTest(text=text):
                self.assertTrue(is_vocable(text))
        for text in ('', 'Oh', 'Oh hey yeah', 'Oh my love', 'Hey hey you', 'La vie la', '123', 'na na 2'):
            with self.subTest(text=text):
                self.assertFalse(is_vocable(text))
        from lyrics_shared import supplied_lyrics
        _, rows = supplied_lyrics(text='LYRICS: writer\'s text\n  [0:10≈] La la — I know\n')
        self.assertFalse(is_vocable(rows[0]['text']))

    def test_peak_pitch_separates_registers_and_ignores_small_cluster(self):
        from instrumental_voice import window_pitch_text
        low, high = 98., 293.6648
        for values in ([low] * 60, [low] * 76 + [high] * 24):
            self.assertEqual(window_pitch_text(values), 'pitch there ~G2')
        for values in ([low] * 60 + [high] * 40, [low] * 75 + [high] * 25,
                       [high] * 75 + [low] * 25, list(np.linspace(75, 100, 60)) + list(np.linspace(300, 600, 40))):
            self.assertIn('splits across octaves:', window_pitch_text(values))
            self.assertIn('(layered vocals or two voices?)', window_pitch_text(values))
        self.assertEqual(window_pitch_text([low] * 60 + [high] * 40),
                         'pitch there splits across octaves: ~G2 and ~D4 (layered vocals or two voices?)')
        self.assertEqual(window_pitch_text([None, np.nan, 0, -1]), '')
        self.assertNotIn('splits', window_pitch_text(np.geomspace(75, 600, 100)))
        self.assertNotIn('splits', window_pitch_text([low] * 50 + [low * 2] * 50))

    def test_saved_peak_pitch_uses_saved_voiced_frames_without_models(self):
        import instrumental_voice as singer
        raw = "SINGER:\n  most intense moment: 0:01–0:06 · 9 dB louder than the voice's median, pitch there ~G2\n"
        data = dict(singer=dict(data=dict(peak=dict(ok=True, start=1, end=6))),
                    melody_input=dict(fps=20, f0=[None] * 20 + [98.] * 60 + [293.6648] * 40 + [None] * 20))
        refreshed = singer.saved_peak_pitch(raw, data)
        self.assertIn('~G2 and ~D4', refreshed)
        self.assertEqual(singer.saved_peak_pitch(refreshed, data), refreshed)
        self.assertEqual(singer.saved_peak_pitch(raw, {}), raw)
        with patch('raven_ears_plus.analyze', side_effect=AssertionError('no model pass')):
            import slim
            out = slim.render_first_listen({'singer': raw}, {'singer': data})
            self.assertIn('~G2 and ~D4', ' '.join(out.split()))

    def test_unsure_note_follows_resolved_display_source_and_saved_wrapping(self):
        import lrclib
        import stage_status as ST
        note = ('lyrics: 18 line(s) kept but marked (unsure), quiet vocals or doubtful Whisper: '
                "0:24 'I had a thought, dear'; 0:27 'However scary'; 0:35 'Another line'")
        def loader(path, notes):
            notes.append(note)
            return [('I', 24., 25.)]
        for status in ('file', 'synced', 'plain', 'not-found'):
            notes = []
            lookup = dict(source='LRCLIB', status=status, plain=['La la'], synced=[dict(t=24., text='La la')])
            with self.subTest(status=status), patch('lrclib.get', return_value=lookup), patch('credits.identify', return_value=('', '', '')):
                result = lrclib.resolve('fixture.mp3', notes, duration=60,
                                        lyric_lines=['La la'] if status == 'file' else None,
                                        word_loader=loader, activity=[(0, 60)], quick=False)
            if status == 'not-found':
                self.assertIn(note, notes)
            else:
                self.assertIn('Whisper was doubtful on 18 line(s)', '\n'.join(notes))
                self.assertNotIn('kept but marked', '\n'.join(notes))
                raw = result['label'] + '\n' + '\n'.join(ST.wrap_note('  NOTE: ' + note)) + '\n'
                refreshed = ST.display_text(raw)
                self.assertIn('alignment check only', refreshed)
                self.assertEqual(ST.display_text(refreshed), refreshed)
                self.assertEqual(len(ST.source_notes(refreshed)), 1)
                self.assertIn("0:35 'Another line'", ST.source_notes(refreshed)[0])

    def test_supplied_words_refresh_real_shaped_gap_rows_and_notes(self):
        import lyrics_shared as LS
        vocable = 'Eeh-eeh-eeh-eeh, eeh-eeh-eeh'
        old = 'voice present, words not heard (Whisper found none)'
        text = ('HARMONY: fixture.mp3 · 3:20\nTIMELINE\n'
                f'  1:58  1:58–2:17: {old}\n'
                f'  2:38  2:38–2:50: {old}\n'
                f'  2:50  2:50–3:00: {old}\n'
                f'  NOTE: 1:57–2:17: {old}\n'
                '  NOTE: 3:00–3:06: no voice (instrumental)\n')
        rows = [dict(t=t, text=vocable, approx=True) for t in (119., 129., 160., 170.)]
        for status in ('synced', 'plain', 'file'):
            data = dict(lyrics_aligned=dict(lines=rows), lyrics_source=dict(status=status))
            out = LS.clean_gap_claims(text, analysis=data)
            self.assertIn(f'1:58  1:58–2:17: wordless singing (lyrics: "{vocable}")', out)
            self.assertIn('NOTE: 1:57–2:17: wordless singing', out)
            self.assertIn('3:00–3:06: no voice (instrumental)', out)
            self.assertNotIn('Whisper found none', out)
            self.assertEqual(LS.clean_gap_claims(out, analysis=data), out)
        unchanged = LS.clean_gap_claims(text, analysis={})
        self.assertIn(old, unchanged)
        self.assertNotIn("wordless singing", unchanged)
        self.assertEqual(LS.clean_gap_claims(text, analysis=dict(lyrics=rows, lyrics_source=dict(status='not-found'))), unchanged)
        # Boundary policy is line starts in [start, end), not estimated ends.
        self.assertIsNone(LS.supplied_gap_description(118, 119, rows))
        words = [dict(t=119, text='I could not ask you where you came from')]
        mixed = LS.supplied_gap_description(118, 137, rows + words)
        self.assertIn("lyrics have words here Whisper didn't hear", mixed)
        self.assertIn('I could not ask', mixed)
        self.assertIsNone(LS.supplied_gap_description(180, 190, rows))

    def test_saved_hook_vocable_and_genuinely_missing_words(self):
        import listen_melody as M
        rows = [dict(start=i, end=i + 1, note_indices=list(range(4)),
                     lyrics='We should kiss like real people do' if i == 0 else '* _ * _') for i in range(2)]
        for following, phrase in (('Eeh-eeh-eeh-eeh, eeh-eeh-eeh', 'next line is wordless'),
                                  ('I could not ask you', 'next line has no rendered lyric words')):
            texts = ['We should kiss like real people do', following]
            indices, rule = M.hook_indices(texts, rows, {}, ['like', 'real', 'people', 'do'], min_notes=4)
            self.assertEqual(indices, [0])
            self.assertIn(phrase, rule)
            old_rule = 'HOOK: line with the title "like real people do"; next line has no rendered lyric words\n'
            song = dict(melody=dict(status='ran', text='', json=dict(hook_text=old_rule, lines=rows,
                        hook=dict(line_indices=[0]), lyric_alignment=dict(rows=[dict(text=t) for t in texts]))))
            rendered = M.render(song, hook=True)
            self.assertIn(phrase, rendered)
            self.assertEqual(song['melody']['json']['hook_text'], old_rule)

    @staticmethod
    def supplied_bundle_fixture():
        """Supplied card's line/row shapes; invented F0, not a saved Hozier listen."""
        vocable = 'Eeh-eeh-eeh-eeh, eeh-eeh-eeh'
        label = 'LYRICS: LRCLIB synced lyrics; supplied times (≈), 20 lines compared with Whisper; line ends estimated'
        rows = [dict(t=t, end=t + 5, text=vocable, approx=True) for t in (119., 129., 160., 170.)]
        lyrics = label + '\n' + '\n'.join(f'  [{int(r["t"]) // 60}:{int(r["t"]) % 60:02d}≈] {r["text"]}' for r in rows)
        old = 'voice present, words not heard (Whisper found none)'
        unsure = ("lyrics: 18 line(s) kept but marked (unsure), quiet vocals or doubtful Whisper: "
                  "0:24 'I had a thought, dear'; 0:27 'However scary'")
        source = dict(source='LRCLIB', status='synced')
        music = dict(text='MUSIC: fixture.mp3 · 3:20\n  KEY: G major (1♯) · tonic clear\n'
                     '  VOCALS: yes, voice present [method: vocal-stem energy] · 0:25–3:00\n' + lyrics +
                     '\n  NOTE: ' + unsure + f'\n  NOTE: 1:57–2:17: {old}\n',
                     json=dict(duration=200, lyrics=rows, lyrics_source=source, lyrics_align=label, notes=[unsure, f'1:57–2:17: {old}']))
        peak = 'most intense moment: 1:59–2:04 · 9 dB louder than the voice\'s median, pitch there ~G2'
        singer = dict(text='SINGER.PY fixture.mp3 3:20\n' + lyrics + '\nSINGER:\n  ' + peak + '\n',
                      json=dict(lyrics=dict(rows=rows, label=label, lookup=source),
                                singer=dict(lines=[peak], data=dict(peak=dict(ok=True, start=119., end=124.))),
                                melody_input=dict(fps=20, f0=[None] * 2380 + [98.] * 60 + [293.6648] * 40 + [None] * 1520)))
        timeline = '\n'.join(f'  {a}  {a}–{b}: {old}' for a, b in (('1:58', '2:17'), ('2:38', '2:50'), ('2:50', '3:00')))
        harmony = dict(text='HARMONY: fixture.mp3 · 3:20\nTIMELINE\n' + timeline + '\n' + lyrics + '\n',
                       json=dict(lyrics_aligned=dict(lines=rows, label=label), lyrics_source=source))
        hook_text = 'HOOK: line with the title "like real people do"; next line has no rendered lyric words\n'
        melody = dict(status='ran', text='', json=dict(hook_text=hook_text, lines=[{}, {}],
                      hook=dict(line_indices=[0]), lyric_alignment=dict(rows=[dict(text='We should just kiss like real people do'), dict(text=vocable)])))
        return dict(schema=1, manifest=dict(options={}, passes={name: dict(status='ran', reason='') for name in ('music', 'singer', 'harmony')}),
                    cards=dict(music=music, singer=singer, harmony=harmony), melody=melody)

    def test_supplied_lyrics_saved_bundle_views_without_engines(self):
        import listen
        import doctored_audio  # scipy's first import probes an already loaded torch module.
        import slim
        import compare
        song = self.supplied_bundle_fixture()
        original = copy.deepcopy(song)
        with patch.dict(sys.modules, {'singer': None, 'music': None, 'harmony': None, 'torch': None}), \
             patch('listen.run_pass', side_effect=AssertionError('saved view must not run a pass')), \
             patch('listen_melody.build', side_effect=AssertionError('saved view must not rebuild notes')):
            for view in ('first-listen', 'summary', 'full'):
                out, _ = listen.render(song, view)
                out = ' '.join(out.split())
                if view != 'first-listen':
                    self.assertIn('Whisper was doubtful on 18 line(s)', out)
                self.assertIn('wordless singing' if view == 'first-listen' else 'wordless singing (lyrics:', out)
                self.assertIn('~G2 and ~D4', out)
                self.assertNotIn('kept but marked', out)
                self.assertNotIn('Whisper found none', out)
                self.assertIn('[1:59≈] Eeh-eeh-eeh-eeh, eeh-eeh-eeh', out)
                if view == 'first-listen':
                    self.assertIn('next line is wordless', out)
                    self.assertNotIn('WORDS ', out)
                if view == 'summary':
                    self.assertIn('WORDS 1:58', out)
            out = slim.render(song['cards']['singer']['text'], analysis=song['cards']['singer']['json'])
            self.assertIn('~G2 and ~D4', ' '.join(out.split()))
        self.assertEqual(song, original)
        # Text-only compare keeps the new singer wording too.
        card = compare.parse_card('SINGER:\n  wordless singing (lyrics: "La la"): 1:58–2:17\n')
        self.assertIn('wordless singing', card.voice)

    def test_language_votes_exclude_translations_but_preserve_display_text(self):
        import lyrics_shared as LS
        english = 'I have the paper and you have the ink. We put it on the table with my pencil.'
        french = 'Je suis dans une maison. Vous et nous, les amis, mes livres et mon papier sont pour vous.'
        spanish = 'Yo tengo el papel y tú tienes el lápiz. Hay una mesa para los libros y ya hay más luz.'
        unknown = 'Ael nîr sulen: (' + english + ')'
        cases = [(unknown, 'en', 'uncertain'),
                 (unknown.replace('(', '[').replace(')', ']'), 'en', 'uncertain'),
                 ('Ael nîr (translation [aside] ' + english + ')', 'en', 'uncertain'),
                 ('Ael nîr (translation\n' + english + ')', 'en', 'uncertain'),
                 (french + ' (' + english * 4 + ')', 'fr', 'dominant'),
                 (spanish + ' [' + english * 4 + ']', 'es', 'dominant'),
                 ('Cantorem lumen (' + french * 4 + ')\n' + english, 'en', 'dominant')]
        with patch.dict(os.environ):
            os.environ.pop('RAVEN_EARS_LANG', None)
            for text, language, reason in cases:
                lines = text.splitlines()
                original = list(lines)
                with self.subTest(language=language, reason=reason), LS.lyric_context(lines, 'LRCLIB'):
                    self.assertEqual(LS.language_choice()[0], language)
                    self.assertIn(reason, LS.language_choice()[1])
                    self.assertEqual(LS._LYRIC_TEXT.get()[0], tuple(original))
                self.assertEqual(lines, original)

    def test_song_language_uses_whole_text_and_explicit_override(self):
        import lyrics_shared as LS
        examples = {
            'en': 'I have the paper and you have the ink. We will put it on the table with my pencil.',
            'fr': 'Je suis dans une maison. Vous et nous, les amis, mes livres et mon papier sont pour vous.',
            'es': 'Yo tengo el papel y tú tienes el lápiz. Hay una mesa para los libros y ya hay más luz.',
            'de': 'Ich und du sind hier. Wir haben ein Buch mit einer Karte. Sie ist nicht für dich und mich.',
        }
        with patch.dict(os.environ):
            os.environ.pop('RAVEN_EARS_LANG', None)
            for language, text in examples.items():
                with self.subTest(language=language), LS.lyric_context([text], 'LRCLIB'):
                    self.assertEqual(LS.effective_language(), language)
                    self.assertIn('LRCLIB', LS.language_choice()[1])
                    with patch.dict(os.environ, {'RAVEN_EARS_LANG': 'ja'}):
                        self.assertEqual(LS.language_choice(), ('ja', 'explicit RAVEN_EARS_LANG'))
            for foreign in ('confiteor deo omnipotenti', 'je suis ici', 'yo estoy aquí'):
                with LS.lyric_context([examples['en'], foreign]):
                    self.assertEqual(LS.effective_language(), 'en')
            for text in ('', 'bonjour', '君の声が聞こえる', 'je suis aquí you are là'):
                with LS.lyric_context([text]):
                    self.assertEqual(LS.effective_language(), 'en')
                    self.assertIn('keeping English default', LS.language_choice()[1])
            with self.assertRaises(RuntimeError), LS.lyric_context([examples['fr']]):
                raise RuntimeError('context must reset on failure')
            self.assertIn('no lyric-text', LS.language_choice()[1])

    def test_quiet_phantom_gate_on_real_segment_shapes(self):
        import lyrics_shared as LS
        import raven_ears_plus as R
        import singer
        # Hedwig turbo CPU's 38.60–39.26 gate/confidence shape. Other text is
        # invented; energy is mocked, never synthetic-audio calibration.
        for text, quiet, reference, kept in (
                ('Thank you.', True, [], False),
                ("We'll be right back.", True, [], False),
                ("We'll be right back.", False, [], True),
                ('Thanks for watching.', True, [], False),
                ('Thank you.', False, [], True),
                ('Thank you.', True, ['I sing thank you to the paper moon'], True),
                ('The paper moon.', True, [], True),
                ('You 愛', True, [], True),
                ('君の声が聞こえる', True, [], True)):
            sg = SimpleNamespace(start=38.60, end=39.26, text=text,
                                 avg_logprob=-0.5198378364245096,
                                 no_speech_prob=8.944110035935537e-11, compression_ratio=.5555555555555556,
                                 words=[SimpleNamespace(word=text, start=38.60, end=39.26)])
            model = SimpleNamespace(transcribe=lambda *a, **k: (iter([sg]) if len(a[0]) == 100 else iter([]), None))
            notes = []
            with self.subTest(text=text, quiet=quiet, reference=bool(reference)), \
                 LS.lyric_context(reference), patch.object(R, 'SR', 1), \
                 patch.object(R, 'load_audio', return_value=np.empty(100)), \
                 patch.object(R, 'release_models'), patch.object(R, '_preload_cuda12'), \
                 patch.object(singer, 'get_stems', return_value=('invented', {})), \
                 patch.object(singer, 'load_mono', return_value=(np.empty(100), 1)), \
                 patch.object(LS, '_level_fn', side_effect=[lambda *a: -10., lambda *a: -40. if quiet else -10.]), \
                 patch.object(LS, '_load_model', return_value=(model, 'cpu', 'int8')), \
                 patch.object(LS, 'whisper_name_kwargs', return_value={}):
                self.assertEqual(bool(LS.transcribe('invented.wav', notes)), kept)
                if not kept and text != 'Thanks for watching.':
                    self.assertTrue(any('known filler' in n for n in notes))

    def test_word_guard_removes_only_the_measurably_quiet_intro(self):
        import lyrics_shared as LS

        notes = []
        words = [('Paper.', 0., .25), ('The', 11.36, 12.76), ('lantern', 12.76, 13.4)]
        level = LS._level_fn(np.r_[np.zeros(12 * 100), np.full(6 * 100, .1)], 100)
        self.assertEqual(LS.drop_quiet_intro(words, level, 18., notes), words[1:])
        self.assertIn('dropped 1 introductory Whisper word', notes[0])
        self.assertEqual(LS.drop_quiet_intro(words, None, 18., []), words)
        self.assertEqual(LS.drop_quiet_intro(words, lambda *a: -40., 18., []), words)
        self.assertEqual(LS.drop_quiet_intro(words[1:], level, 18., []), words[1:])

    def test_lyric_cache_uses_inferred_language_and_reference_veto(self):
        import lyrics_shared as LS
        import raven_ears_plus as R
        french = ['Je suis dans une maison. Vous et nous, les amis, mes livres et mon papier sont pour vous.']
        with tempfile.TemporaryDirectory() as directory, patch.object(LS, 'CACHE', directory), \
             patch.object(LS, 'effective_names', return_value=[]), \
             patch.object(LS, '_model_attempts', return_value=[('cpu', 'int8')]), patch.dict(os.environ):
            os.environ.pop('RAVEN_EARS_LANG', None)
            audio = Path(directory) / 'invented.wav'
            audio.touch()
            def words(path, notes):
                notes.append(LS._transcription_note(*LS._model_attempts(R)[0]))
                return [('papier', 10., 10.5)]
            with patch.object(LS, 'transcribe', side_effect=words) as transcribe:
                with LS.lyric_context(french, 'LRCLIB'):
                    original = LS.cache_file(audio)
                    LS.get_words(audio)
                    notes = []
                    LS.get_words(audio, notes)
                    self.assertIn('language fr (LRCLIB', notes[0])
                    self.assertTrue(any('[shared cache]' in n for n in notes))
                with LS.lyric_context(french + ['Thank you.'], 'LRCLIB'):
                    self.assertNotEqual(LS.cache_file(audio), original)
                with patch.dict(os.environ, {'RAVEN_EARS_LANG': 'en'}), LS.lyric_context(french):
                    self.assertNotEqual(LS.cache_file(audio), original)
                    LS.get_words(audio)
                self.assertEqual(transcribe.call_count, 2)

    def test_lyric_cache_separates_languages_and_leaves_legacy_untouched(self):
        import lyrics_shared as LS
        with tempfile.TemporaryDirectory() as directory, patch.object(LS, 'CACHE', directory), \
             patch.object(LS, 'effective_names', return_value=[]):
            audio = Path(directory) / 'invented.wav'
            audio.write_bytes(b'invented, not audio')
            legacy = Path(directory) / (LS.file_sha256(audio) + '.json')
            legacy.write_text('legacy cache must remain untouched')
            def words(path, notes):
                import raven_ears_plus as R
                notes.append(LS._transcription_note(*LS._model_attempts(R)[0]))
                return [(LS.effective_language(), 10., 10.5)]
            with patch.object(LS, 'transcribe', side_effect=words) as transcribe:
                paths = []
                for language in ('en', 'fr', 'en', 'fr'):
                    with patch.dict(os.environ, {'RAVEN_EARS_LANG': language}):
                        paths.append(LS.cache_file(audio))
                        self.assertEqual(LS.get_words(audio), [(language, 10., 10.5)])
                self.assertEqual(transcribe.call_count, 2)
                self.assertNotEqual(paths[0], paths[1])
                self.assertEqual(paths[:2], paths[2:])
                with patch.dict(os.environ):
                    os.environ.pop('RAVEN_EARS_LANG', None)
                    self.assertEqual(LS.cache_file(audio), paths[0])
            self.assertEqual(legacy.read_text(), 'legacy cache must remain untouched')

    def test_saved_gaps_keep_word_availability_singletons_and_quiet(self):
        import lyrics_shared as LS
        evidence = '  VOCALS: yes [method: vocal-stem energy] · 0:20–0:28, 0:30–0:40\n'
        unavailable = 'voice present, words unavailable (transcription did not run or failed)'
        text = ('TIMELINE\n'
                f'   0:20  0:20–0:24: {unavailable}\n'
                f'   0:24  0:24–0:32: {unavailable}\n'
                f'   0:32  0:32: {unavailable}\n'
                f'  [0:32≈] "Paper moons (unsure)" — chord V7, 0:32: {unavailable}\n')
        data = {'stage_status': [S.outcome('words', 'failed', 'invented backend failure')]}
        out = LS.clean_gap_claims(text, evidence, data)
        self.assertIn(f'0:20  0:20–0:28: {unavailable}', out)
        self.assertIn('0:28  0:28–0:30: no voice (instrumental)', out)
        self.assertIn(f'0:30  0:30–0:32: {unavailable}', out)
        self.assertIn(f'0:32  0:32: {unavailable}', out)
        self.assertIn('[0:32≈] "Paper moons (unsure)" — chord V7\n', out)
        self.assertNotIn('Whisper found none', out)
        self.assertEqual(LS.clean_gap_claims(out, evidence, data), out)
        old = text.replace('words unavailable (transcription did not run or failed)',
                           'words not heard (Whisper found none)')
        self.assertNotIn('Whisper found none', LS.clean_gap_claims(old, evidence, data))
        summary = LS.clean_gap_claims('  voice with no words: 0:20–0:28\n', evidence, data)
        self.assertIn('voice present, words unavailable', summary)
        self.assertNotIn('Whisper found none', summary)
        unknown = LS.clean_gap_claims(old, evidence.rstrip() + ' +2 more\n', data)
        self.assertIn('voice presence unverified', unknown)
        self.assertNotIn('no voice (instrumental)', unknown)

    def test_lrclib_display_provenance_keeps_alignment_failures(self):
        label = 'LYRICS: LRCLIB synced lyrics; supplied times (≈), 37 lines compared with Whisper; line ends estimated'
        text = (label + '\n  [0:10≈] "Paper moons (unsure)"\n'
                "  STATUS: words: Whisper's guesses (no lyrics file)\n"
                'NOTES: Whisper: failed: invented backend failure · lyrics file: no lyrics file\n')
        out = S.display_text(text)
        self.assertIn('lyrics: LRCLIB synced (displayed) · Whisper: alignment check', out)
        self.assertIn('Whisper: failed: invented backend failure', out)
        self.assertIn('[0:10≈] "Paper moons (unsure)"', out)
        self.assertNotIn('no lyrics file', out)
        self.assertEqual(S.display_text(out), out)

    def test_instrumental_designation_requires_lrclib_resolution(self):
        import instrumental_voice as IV
        for source in (None, {}, dict(source='LRCLIB', status='not-found'),
                       dict(source='LRCLIB', status='unreachable'),
                       dict(source='file', status='instrumental')):
            for field in ('lyrics_source', 'lyrics'):
                data = {field: source if field == 'lyrics_source' else dict(lookup=source)}
                self.assertFalse(IV.designated(data))
        source = dict(source='LRCLIB', status='instrumental')
        for data in (dict(lyrics_source=source), dict(lyrics=dict(lookup=source))):
            self.assertTrue(IV.designated(data))
        cards = {'music': dict(json=dict(lyrics_source=source))}
        self.assertFalse(IV.saved_designation(cards, {'music': dict(status='failed')}))
        self.assertTrue(IV.saved_designation(cards, {'music': dict(status='ran')}))

    def test_instrumental_stem_prose_keeps_measurements_and_hides_speech_affect(self):
        import instrumental_voice as IV
        # Invented values in the supplied Take Five card's exact line shapes.
        text = ("SINGER:\n"
                "  pitch: median C4 (262 Hz) · comfortable range G3–G4 (12 semitones)\n"
                "  highest held notes: D4 at 0:29 (0.9s)\n"
                "  lowest held notes: G3 at 0:56\n"
                "  pitch arc (median by section): C4 → D4 · the voice lifts ~3 semitones by 0:58\n"
                "  phrase endings: 6 lift, 3 fall, 7 hold\n"
                "  arousal across the song: ▂▇█  (0:00 → 1:00; 0.35 to 0.47; · = no voice) "
                "(speech-trained model; read as texture, not a verdict)\n"
                "    by section: 0:00=0.38 0:27=0.45\n"
                "  most intense moment: 0:45–0:50 · 11 dB louder than the voice's median · arousal 0.62 vs 0.40 average "
                "(speech-trained model; read as texture, not a verdict), pitch there ~D4\n"
                "  overall: arousal 0.40 · dominance 0.39 · valence 0.41 (0-1) "
                "(speech-trained model; read as texture, not a verdict)\n"
                "INSTRUMENTS (model estimates):\n"
                "  who is playing ('+ voice' = the vocal stem has level):\n"
                "    0:14–0:31  keys + voice\n"
                "STOPS: 1 · band cuts, voice holds: 1\n")
        self.assertEqual(IV.stem_text(text, False), text)
        out = IV.stem_text(text, True)
        for measurement in ('median C4 (262 Hz)', 'G3–G4 (12 semitones)',
                            'D4 at 0:29 (0.9s)', 'G3 at 0:56', '6 lift, 3 fall, 7 hold',
                            '11 dB louder', 'pitch there ~D4'):
            self.assertIn(measurement, out)
        self.assertIn(IV.MELODY, out)
        self.assertEqual(out.count(IV.AROUSAL), 1)
        self.assertIn('0:14–0:31  keys + vocal stem', out)
        self.assertIn('band cuts, vocal stem holds: 1', out)
        self.assertNotRegex(out, r'arousal \d|dominance|valence|by section:|the voice|comfortable range')
        self.assertEqual(IV.stem_text(out, True), out)
        self.assertEqual(IV.stem_text('SINGER:', True).splitlines(), [IV.MELODY, '  ' + IV.AROUSAL])

    def test_instrumental_gap_summary_never_joins_measured_quiet(self):
        import instrumental_voice as IV
        # Include the leading clock column, singleton times, and real descriptions.
        text = ('TIMELINE\n'
                '   0:00  opens quiet intro, -36 dB\n'
                '   0:00  0:00–0:23: no voice (instrumental)\n'
                '   0:23  0:23–0:28: voice present, words unavailable (transcription did not run or failed)\n'
                '   0:28  0:28: no voice (instrumental)\n'
                '   0:28  0:28–0:31: voice present, words not heard (Whisper found none)\n'
                '   0:31  0:31–0:58: no voice (instrumental)\n'
                '   0:40  bass enters (possibly)\n')
        out = IV.stem_text(text, True)
        self.assertEqual(out.count(IV.WORDS), 1)
        self.assertIn('0:40  bass enters (possibly)', out)
        self.assertNotRegex(out, r'voice present|no voice|active|0:23–0:31|did not run or failed|Whisper found none')
        quote = '  [0:10≈] "words unavailable (transcription did not run or failed) + voice"\n'
        self.assertEqual(IV.stem_text(quote, True), quote)
        self.assertEqual(IV.stem_text(out, True), out)

    def test_fresh_music_instrumental_wording_is_only_a_presentation_change(self):
        import instrumental_voice as IV
        import music
        data = dict(file='invented.wav', duration=60,
                    vocals=dict(method='vocal stem', frac=.18, share=.11, spans=[(23, 28), (58, 59)]),
                    lyrics_source=dict(source='LRCLIB', status='instrumental'),
                    lyric_gaps=[dict(start=0, end=60, description=
                        'voice present, words unavailable (transcription did not run or failed)')])
        before = copy.deepcopy(data)
        out = music.format_card(data)
        self.assertIn('vocal stem active ~18%', out)
        self.assertIn('0:23–0:28, 0:58–0:59', out)
        self.assertIn(IV.ATTRIBUTION, out)
        self.assertIn(IV.WORDS, out)
        self.assertEqual(data, before)
        for status in ('not-found', 'unreachable', 'synced'):
            data['lyrics_source']['status'] = status
            out = music.format_card(data)
            self.assertIn('yes, voice present', out)
            self.assertIn('did not run or failed', out)

    def test_instrumental_slim_json_fallback_cannot_restore_singer_prose(self):
        import instrumental_voice as IV
        import slim
        data = dict(file='invented.wav', duration=60,
                    lyrics=dict(lookup=dict(source='LRCLIB', status='instrumental'), rows=None),
                    singer=dict(lines=['pitch: median C4 · comfortable range G3–G4',
                                       'pitch arc: C4 → D4 · the voice lifts ~3 semitones',
                                       'overall: arousal 0.40 · dominance 0.39 · valence 0.41 (0-1)']))
        before = copy.deepcopy(data)
        for text in (slim.render(analysis=data),
                     slim.render_first_listen({}, {'singer': data})):
            self.assertIn('median C4', text)
            self.assertIn(IV.MELODY, text)
            self.assertIn(IV.AROUSAL, text)
            self.assertNotIn('the voice', text)
            self.assertNotRegex(text, r'arousal 0\.|dominance 0\.|valence 0\.')
        self.assertEqual(data, before)

    def test_unicode_lyrics_are_not_credit_or_note_filler(self):
        import lyrics_shared as LS
        for line in ('君の声が聞こえる', '月亮照着我们', 'أسمع صوتك', 'Я слышу твой голос',
                     '♪ 君の声 ♫', '１２３', 'Hello'):
            with self.subTest(line=line):
                self.assertFalse(LS.is_credit(line))
        for line in ('♪ ♫', '♬ 🎵 🎶', '(c)', '(C)', '©', '© 2026', '___', '… — !', '',
                     'captions by Example', 'all rights reserved', 'please subscribe'):
            with self.subTest(line=line):
                self.assertTrue(LS.is_credit(line))

    def test_madmom_discovery_order_and_venv_guard(self):
        import keyfuse as K
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = root / 'code'
            prefix = root / 'install' / 'venv'
            override = root / 'override'
            beside_code = code / 'venv-madmom' / 'bin' / 'python'
            beside_venv = prefix.parent / 'venv-madmom' / 'bin' / 'python'
            for path in (override, beside_code, beside_venv):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.object(K, 'HERE', str(code)), patch.object(sys, 'prefix', str(prefix)), \
                    patch.object(sys, 'base_prefix', str(root / 'base')), \
                    patch.dict(os.environ, {'RAVEN_MADMOM_PY': str(override)}):
                self.assertEqual(K.madmom_python(), str(override))
                override.unlink()  # A broken override still falls through.
                self.assertEqual(K.madmom_python(), str(beside_code))
                os.environ.pop('RAVEN_MADMOM_PY')
                self.assertEqual(K.madmom_python(), str(beside_code))
                beside_code.unlink()
                self.assertEqual(K.madmom_python(), str(beside_venv))
                with patch.object(sys, 'base_prefix', str(prefix)):
                    self.assertIsNone(K.madmom_python())
                beside_venv.unlink()
                self.assertIsNone(K.madmom_python())
                @S.engine
                def run():
                    self.assertIsNone(K.cnn_probs('invented.wav'))
                    return {}
                rows = run()['stage_status']
                self.assertEqual(rows[0]['status'], 'skipped')
                self.assert_cnn_key_warning(rows, 'CNN key voter: optional, not installed')

    def assert_cnn_key_warning(self, rows, warning):
        import music
        import slim
        data = dict(file='invented.wav', duration=60,
                    key_claim={'line': 'C major · tonic fairly clear'}, stage_status=rows)
        text = music.format_card(data)
        for rendered in (text, slim.render_first_listen({'music': text}, {'music': data}),
                         slim.render_first_listen({'music': 'KEY: C major'}, {'music': data})):
            key_line = next(line for line in rendered.splitlines() if line.strip().startswith('KEY:'))
            self.assertIn(warning, key_line)
            self.assertEqual(key_line.count(warning), 1)
        self.assertEqual(S.cnn_key_text(text, data), text)

    def test_failed_cnn_warning_stays_on_key_line(self):
        self.assert_cnn_key_warning([S.outcome('CNN key voter', 'failed', 'TimeoutExpired: invented')],
                                    'CNN key voter not run (execution error)')

    def test_ran_cnn_leaves_key_card_identical(self):
        import music
        import slim
        data = dict(file='invented.wav', duration=60, key_claim={'line': 'C major · tonic fairly clear'},
                    stage_status=[S.outcome('CNN key voter')])
        with patch.object(S, 'cnn_key_text', side_effect=lambda text, analysis=None: text):
            original = music.format_card(data)
            original_feel = slim.render_first_listen({'music': original}, {'music': data})
        self.assertEqual(music.format_card(data), original)
        self.assertEqual(slim.render_first_listen({'music': original}, {'music': data}), original_feel)

    def test_sub_second_gap_claim_prints_one_time(self):
        import lyrics_shared as LS
        out = LS.gap_claims(160.0, 175.0, ([(168.1, 168.6)],))
        self.assertIn("2:48: voice present", out)
        self.assertNotIn("2:48–2:48", out)

    def test_vocal_span_at_248_prints_one_time_and_keeps_the_measurement(self):
        # Fresh main on a real song printed 'VOCALS … 2:36–2:39, 2:48–2:48, 3:00–3:18' (11 Nov 2026).
        import music
        spans = [(14., 160.), (168., 168.2), (180., 190.)]
        vocals = dict(frac=.5, kinds=[], spans=list(spans), method='vocal stem', share=.3)
        card = music.format_card(dict(file='invented.wav', duration=200, vocals=vocals))
        self.assertIn('2:48,', card)
        self.assertNotIn('2:48–2:48', card)
        self.assertEqual(vocals['spans'], spans)  # the data keeps the blip; only the print changes

    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        for name in ('RAVEN_EARS_PROFILE', 'CUDA_VISIBLE_DEVICES', 'RAVEN_EARS_OFFLINE',
                     'WHISPER_MODEL', 'RAVEN_EARS_DEVICE'):
            os.environ.pop(name, None)

    def home_landing_result(self):
        # Invented: a moved middle area, then the original tonic to the end.
        return dict(glob=(4, 'minor'), spans=[(0., (4, 'minor')), (40., (11, 'minor'))],
                    kinfo={'changes': [{'at': 40., 'to': 'B minor'}]}, dur=120., bpm=120,
                    key_verdict={'key': 'E minor', 'working': 'E minor', 'line': 'E minor'},
                    evs=[dict(t0=a, t1=b, root=root, q='min7')
                         for a, b, root in ((0., 40., 4), (40., 80., 11), (80., 120., 4))],
                    secs=[dict(start=a, end=a+40., level='mid') for a in (0., 40., 80.)],
                    moves=[], pedals=[])

    def test_home_landing_after_a_moved_key_area(self):
        import harmony as H
        text = '\n'.join(H.block_chords(self.home_landing_result()))
        self.assertIn('ends on Em7: home i7 (iv7 of the B minor area); opens on Em7: home i7', text)
        self.assertIn('most time on: i7 67%, v7 33%', text)
        self.assertIn('[mid outro]: iv7', text)
        self.assertNotIn('never lands', text.split('[mid outro]:')[1])
        self.assertIn('section numerals use the local key-area estimates above', text)

    def test_home_absence_can_coexist_with_local_tonic(self):
        import harmony as H
        text = '\n'.join(H.block_chords(self.home_landing_result()))
        self.assertEqual(text.count('never lands on home tonic i'), 1)
        self.assertIn('never lands on home tonic i; local tonic time: 100%', text)
        r = self.home_landing_result()
        r['evs'][1]['root'] = 2  # neither tonic
        text = '\n'.join(H.block_chords(r))
        self.assertIn('never lands on home tonic i; local tonic time: 0%', text)

    def test_timeline_ending_agrees_with_home_landing(self):
        import harmony as H
        r = self.home_landing_result()
        r.update(secs=[], kinfo={}, tc={}, lines=[], csurp=[], br=[], dips=[], drops=[],
                 swells=[], con=[], tag=[])
        with patch.dict(os.environ, {'RAVEN_NO_INSTRUMENTS': '1'}):
            text = str(H.build_timeline(r))
        self.assertIn('last chord: Em7: home i7 (iv7 of the B minor area)', text)

    def test_home_tally_neither_splits_chords_nor_merges_local_names(self):
        import harmony as H
        r = self.home_landing_result()
        r.update(glob=(9, 'major'), spans=[(0., (9, 'major')), (40., (2, 'major'))],
                 key_verdict={'key': 'A major', 'working': 'A major', 'line': 'A major'},
                 kinfo={'changes': [{'at': 40., 'to': 'D major'}]})
        for e, root in zip(r['evs'], (2, 2, 9)):
            e.update(root=root, q='maj7')
        text = '\n'.join(H.block_chords(r))
        self.assertIn('most time on: IVmaj7 67%, Imaj7 33%', text)
        r['evs'][0]['band_out'] = True
        self.assertIn('most time on: IVmaj7 50%, Imaj7 50%', '\n'.join(H.block_chords(r)))

    def test_unclear_home_remains_a_working_reference_after_key_change(self):
        import harmony as H
        r = self.home_landing_result()
        r['key_verdict'].update(key=None, line='unclear between E minor and B minor')
        text = '\n'.join(H.block_chords(r))
        self.assertIn('ends on Em7: E minor working reference i7 (iv7 of the B minor area)', text)
        self.assertIn('never lands on E minor working-reference tonic i', text)
        self.assertNotIn('home tonic', text)
        self.assertNotIn(': home i7', text)

    def test_local_tonic_time_clips_a_chord_crossing_the_area_boundary(self):
        import harmony as H
        r = self.home_landing_result()
        r['evs'] = [dict(t0=20., t1=120., root=11, q='min7')]
        text = '\n'.join(H.block_chords(r))
        self.assertIn('0:40–1:20 [mid]: i7\n      never lands on home tonic i; local tonic time: 100%', text)

    def test_section_wholly_in_home_area_keeps_the_plain_warning(self):
        # Invented: the first section sits in the home area and never touches its tonic.
        import harmony as H
        r = self.home_landing_result()
        r['evs'][0]['root'] = 9
        text = '\n'.join(H.block_chords(r))
        first = text.split('0:00–0:40')[1].split('\n  0:40')[0]
        self.assertIn('never lands on I in this section: it circles without coming home', first)
        self.assertNotIn('local tonic time', first)

    def test_small_home_share_is_not_claimed_to_be_zero(self):
        import harmony as H
        r = self.home_landing_result()
        r['evs'][1]['t1'] = 79.
        r['evs'].insert(2, dict(t0=79., t1=80., root=4, q='min7'))
        text = '\n'.join(H.block_chords(r))
        self.assertIn('less than 5% of this section on home tonic i', text)
        self.assertNotIn('never lands', text)

    def test_tension_stays_local_after_a_key_change(self):
        # Muninn, PR #29: once the music settles in a new key, that key is home for now;
        # a last-chorus lift must not read as tense all chorus. Only CHORDS judges against home.
        import harmony as H
        n = int(120 * H.FPS) + 1
        f = dict(chroma=np.ones((12, n)), rms_db=np.zeros(n), oenv=np.ones(n), y_len=120 * H.SR)
        r = self.home_landing_result()
        moved = H.tension_curve(f, r['evs'], r['spans'])
        self.assertEqual(moved['harm'][30], H.func_tension(11, 'min7', (11, 'minor')))  # Bm7 in the B minor area: local i7

    def test_single_key_area_chords_remain_byte_identical(self):
        import harmony as H
        r = self.home_landing_result()
        r.update(spans=[(0., (4, 'minor'))], kinfo={})
        self.assertEqual('\n'.join(H.block_chords(r)) + '\n',
            'CHORDS\n'
            '  home: E minor; ~120 BPM beat-tracker pulse; chords decoded on beat-synced chroma, triads + sevenths, Viterbi-smoothed\n'
            '  most time on: i7 67%, v7 33%\n'
            '  0:00–0:40 [mid intro]: i7\n'
            '  0:40–1:20 [mid]: v7\n'
            '      never lands on I in this section: it circles without coming home\n'
            '  1:20–2:00 [mid outro]: i7\n'
            '  ends on i7 (Em7); opens on i7\n')

    def test_transcript_gaps_do_not_claim_wordless_singing(self):
        import harmony as H
        import lyrics_shared as LS
        fake = {key: [] for key in ("secs", "spans", "evs", "moves", "pedals", "br", "dips", "drops", "swells", "con", "tag", "csurp")}
        fake.update(tc={}, dur=70, kinfo={}, words=[("lantern", 2, 5), ("river", 35, 40)],
                    lines=[dict(t=2, end=5, text="lantern"), dict(t=35, end=40, text="river")],
                    vocal_activity=[(4, 65)])
        with patch.dict(os.environ, {"RAVEN_NO_INSTRUMENTS": "1"}):
            timeline = str(H.build_timeline(fake))
        self.assertIn("0:05–0:35: voice present, words not heard (Whisper found none)", timeline)
        self.assertIn("0:40–1:05: voice present, words not heard", timeline)
        self.assertIn("1:05–1:10: no voice (instrumental)", timeline)
        self.assertNotIn("wordless", timeline)
        self.assertIn("unverified", LS.gap_description(5, 35, (None,)))
        self.assertIn("voice present", LS.gap_description(5, 35, ([], [(10, 12)])))
        self.assertIn("no voice (instrumental)", LS.gap_description(5, 35, ([], [])))
        self.assertNotIn("Whisper found none", LS.gap_description(5, 35, ([(10, 12)],), False))
        fake.update(words=None, lines=[], vocal_activity=None,
                    mel=dict(source="vocal stem", f0=[200.] * 70, hop=H.SR))
        with patch.dict(os.environ, {"RAVEN_NO_INSTRUMENTS": "1"}):
            timeline = str(H.build_timeline(fake))
        self.assertIn("words unavailable", timeline)
        self.assertIn("voice presence unverified", timeline)
        self.assertNotIn("voice present", timeline)
        self.assertNotIn("Whisper found none", timeline)

    def test_stem_pitch_cannot_outvote_voice_energy_in_a_transcript_gap(self):
        import harmony as H
        fake = {key: [] for key in ("secs", "spans", "evs", "moves", "pedals", "br", "dips", "drops", "swells", "con", "tag", "csurp")}
        fake.update(tc={}, dur=40, kinfo={}, words=[("paper", 0, 5), ("moon", 35, 40)],
                    lines=[], mel=dict(source="vocal stem", f0=[200.] * 40, hop=H.SR))
        for activity, expected in (([], "no voice (instrumental)"), (None, "voice presence unverified"),
                                   ([(5, 35)], "voice present, words not heard")):
            with self.subTest(activity=activity), patch.dict(os.environ, {"RAVEN_NO_INSTRUMENTS": "1"}):
                timeline = str(H.build_timeline(dict(fake, vocal_activity=activity)))
            self.assertIn(expected, timeline)
            if not activity:
                self.assertNotIn("voice present", timeline)

    def test_gap_edges_drop_short_voice_fragments_without_extending_quiet(self):
        import lyrics_shared as LS
        self.assertEqual(LS.gap_claims(10, 40, ([(0, 11), (39, 50)],)),
                         "0:11–0:39: no voice (instrumental)")
        self.assertEqual(LS.gap_claims(10, 40, ([(0, 50)],)),
                         "0:10–0:40: voice present, words not heard (Whisper found none)")
        self.assertEqual(LS.gap_description(10, 40, ([],)), "no voice (instrumental)")
        import slim
        claim = "0:10–0:40: voice present, words not heard (Whisper found none)"
        evidence = "MUSIC: invented.wav · 1:00\n  VOCALS: yes · 0:00–0:11, 0:39–0:50\n"
        old = f"TIMELINE\n  0:10  {claim}\n"
        full = LS.clean_gap_claims(old, evidence)
        self.assertIn("0:11  0:11–0:39: no voice (instrumental)", full)
        feel = slim.render_first_listen(dict(music=evidence, harmony=old))
        self.assertNotIn("no voice (instrumental)", feel)
        self.assertNotIn("words not heard", feel)

    def test_solo_shaped_saved_gaps_follow_energy_edges_in_every_annotation(self):
        import lyrics_shared as LS
        import slim
        claim = "2:27–3:00: voice present, words not heard (Whisper found none)"
        evidence = ("MUSIC: invented.wav · 3:17\n"
                    "  VOCALS: yes [method: vocal-stem energy] · 0:14–2:28, 2:56–3:06\n")
        old = (f"  NOTE: {claim}\nTIMELINE\n   2:27  {claim}\n"
               f'  [2:23≈] "Paper moons" — chord V7, {claim}\n')
        fixed = LS.clean_gap_claims(old, evidence)
        for expected in ("2:28–2:56: no voice (instrumental)", "2:56–3:00: voice present"):
            self.assertEqual(fixed.count(expected), 1)
        self.assertNotIn("2:27–2:28", fixed)
        self.assertIn('[2:23≈] "Paper moons" — chord V7\n', fixed)
        self.assertNotIn(claim, fixed)
        self.assertEqual(LS.clean_gap_claims(fixed, evidence), fixed)
        quote = f'  [2:23≈] "{claim}"\n'
        self.assertEqual(LS.clean_gap_claims(quote, evidence), quote)
        for card in (slim.render(evidence + old),
                     slim.render_first_listen(dict(music=evidence, harmony=old))):
            self.assertNotIn(claim, card)
            self.assertNotIn("no voice (instrumental)", card)
            self.assertIn("2:56–3:00: voice present, words not heard", card)
        # The previous saved rendering may already contain split claims.
        split = ("2:27–2:28: voice present, words not heard (Whisper found none); "
                 "2:28–2:56: no voice detected; words not heard (Whisper found none); "
                 "2:56–3:00: voice present, words not heard (Whisper found none)")
        self.assertEqual(LS.clean_gap_claims(old.replace(claim, split), evidence), fixed)
        incomplete = evidence.rstrip() + " +2 more\n"
        fixed = LS.clean_gap_claims(old, incomplete)
        self.assertNotIn("no voice (instrumental)", fixed)
        self.assertIn("voice presence unverified", fixed)
        quiet = "  VOCALS: instrumental [method: vocal-stem energy]\n"
        self.assertNotIn("voice present", LS.clean_gap_claims(old, quiet))
        # A masked transcript still reports voice when energy confirms it.
        confirmed = "  VOCALS: yes [method: vocal-stem energy] · 0:14–3:06\n"
        self.assertIn(claim, LS.clean_gap_claims(old, confirmed))
        self.assertNotIn('chord V7, 2:27', LS.clean_gap_claims(old, confirmed))

    def test_new_gap_events_stay_off_lyric_timeline_rows(self):
        import harmony as H
        fake = dict(align=dict(rows=[dict(t=143, end=180, text="Paper moons", approx=True)],
                               label="LYRICS: invented supplied words"), con=[])
        events = [(147, 8, "2:28–2:56: no voice (instrumental)"),
                  (148, 5, "new section: loud")]
        with patch.object(H, "build_timeline", return_value=events), \
             patch.object(H, "_line_chords", return_value="V7"), \
             patch.object(H, "_line_tension", return_value=""):
            text = "\n".join(H.block_lyric_timeline(fake))
        self.assertIn('"Paper moons" — chord V7, new section: loud', text)
        self.assertNotIn("no voice", text)

    def test_singer_reports_unheard_words_in_all_condensed_views(self):
        import singer
        import slim
        import raven_ears_plus as rep
        data = {"gaps": {"wordless": [(5, 18), (40, 65)]}}
        with patch.object(rep, 'analyze', return_value=dict(data=data, timings={})), \
             patch.object(rep, 'release_models'), patch.object(singer, '_free_gpu'):
            lines, _, _ = singer.singer_block('invented.wav', 70)
        text = "SINGER:\n" + "\n".join("  " + line for line in lines) + "\n"
        self.assertIn("voice present, words not heard (Whisper found none): 0:05–0:18, 0:40–1:05", text)
        self.assertNotIn("voice with no words:", text)
        self.assertIn("voice present, words not heard", slim.render(text))
        self.assertIn("voice present, words not heard", slim.render_first_listen({"singer": text}))

    def test_saved_gap_prose_uses_saved_voice_evidence(self):
        import lyrics_shared as LS
        import slim
        old = "TIMELINE\n   0:25  the words stop; wordless for 0:27, until 0:52\n"
        evidence = "  VOCALS: yes · 0:10–2:20\n"
        fixed = LS.clean_gap_claims(old, evidence)
        self.assertIn("0:25–0:52: voice present, words not heard (Whisper found none)", fixed)
        self.assertNotIn("wordless", fixed)
        self.assertIn("unverified", LS.clean_gap_claims(old))
        lyric = "  [0:05] the words stop; wordless for 0:27, until 0:52\n"
        self.assertEqual(LS.clean_gap_claims(lyric, evidence), lyric)
        annotated = '  [0:04] "lantern river" — chord I, the words stop; wordless for 0:27, until 0:52\n'
        self.assertEqual(LS.clean_gap_claims(annotated, evidence),
                         '  [0:04] "lantern river" — chord I\n')
        feel = slim.render_first_listen(dict(music="MUSIC: example.mp3 · 1:00\n" + evidence, harmony=old))
        self.assertIn("0:25–0:52: voice present, words not heard", feel)

    def test_sentiment_failure_does_not_erase_lyrics_from_contrast(self):
        import harmony as H
        lines = [dict(text='paper boat', t=0, end=1)]
        def unavailable(*args, **kwargs):
            raise RuntimeError('missing model')
        @S.engine
        def run():
            return dict(lines=lines, con=H.contrast(lines, [], {}, 1))
        with patch.object(H, '_SENT', None), \
             patch.object(H, 'offline', return_value=False), \
             patch.dict(sys.modules, {'transformers': SimpleNamespace(pipeline=unavailable)}):
            result = run()
        card = '\n'.join(H.block_contrast(result))
        self.assertIn('unavailable (sentiment stage failed or unavailable, see NOTES)', card)
        self.assertNotIn('no lyrics', card)
        self.assertIn('sentiment: failed: RuntimeError: missing model', S.summary(result['stage_status']))
        self.assertIn('no transcribed lyric lines', '\n'.join(H.block_contrast(dict(con=[], lines=[], words=[]))))
        self.assertNotIn('no lyrics', '\n'.join(H.block_contrast(dict(con=[], lines=[], words=None))))

    def test_sentiment_pipeline_does_not_duplicate_hub_kwargs(self):
        import harmony as H
        def pipeline(*args, model_kwargs=None, **kwargs):
            # Transformers supplies local_files_only in hub_kwargs itself.
            def config(*, local_files_only):
                pass
            config(local_files_only=False, **(model_kwargs or {}))
            return lambda texts, **kw: [[dict(label='positive', score=.8),
                                        dict(label='negative', score=.2)] for _ in texts]
        with patch.object(H, '_SENT', None), patch.object(H, 'offline', return_value=False), \
             patch.dict(sys.modules, {'transformers': SimpleNamespace(pipeline=pipeline)}):
            self.assertAlmostEqual(H.sentiment(['paper boat'])[0], .6)

    def test_credit_merge_accepts_cached_lists_and_fresh_tuples(self):
        import credits as C
        spans = [[8, 12], (0, 4), [2, 6], (20, 25)]
        before = copy.deepcopy(spans)
        self.assertEqual(C._merge(spans), [[0, 12], [20, 25]])
        self.assertEqual(spans, before)
        res = dict(duration=40, n=40, hop=1, other_on=[(8, 20)],
                   other_character={'label': 'keys (piano or organ?)'},
                   families={'piano': dict(stem='other', spans=[[0, 8]])})
        cr = dict(source='discogs', label='Discogs', entries=[dict(who='', classes=['piano'])])
        result = C.apply(res, cr)
        self.assertIn('piano: 0:00–0:20', '\n'.join(result['credit_lines']))
        self.assertNotIn('credit_lines', res)

    def test_credit_apply_crash_reaches_standalone_result_and_card(self):
        import credits as C
        import music
        with (patch.object(C, 'for_song', return_value={}),
              patch.object(C, 'apply', side_effect=TypeError('incompatible spans')),
              redirect_stderr(io.StringIO())):
            result = C.apply_for('neutral.wav', {'families': {}})
        self.assertIn('credits: failed: TypeError: incompatible spans', S.summary(result['stage_status']))
        card = music.format_card(dict(file='neutral.wav', duration=30, stage_status=result['stage_status']))
        self.assertIn('credits: failed', card)
        self.assertEqual(card.splitlines()[-1], 'and now, you')

    def test_credits_no_cache_never_writes_a_lookup(self):
        import credits as C
        with patch.object(C, 'identify', return_value=('Example', 'Song', 'flags')), \
             patch.object(C, 'chain', return_value=dict(entries=[], mb={'status': 'nomatch'}, dc={'status': 'nomatch'})), \
             patch.object(C.os, 'makedirs') as mkdir:
            C.get('neutral.wav', use_cache=False)
        mkdir.assert_not_called()

    def test_all_engine_clis_configure_identity_before_backend_work(self):
        import credits as C
        import lrclib as L
        import music
        import singer
        import harmony
        old_c, old_l = dict(C.CONFIG), dict(L.CONFIG)
        try:
            for module, boundary in ((music, 'load_audio'), (singer, 'get_stems'), (harmony, 'cache_path')):
                owner = __import__('raven_ears_plus') if module is music else module
                argv = [module.__file__, 'neutral.wav', '--artist', 'Example Artist', '--title', 'Example Song', '--no-credits']
                with patch.object(sys, 'argv', argv), \
                     patch.dict(sys.modules, {'librosa': SimpleNamespace()}), \
                     patch.object(owner, boundary, side_effect=RuntimeError('stop before audio')), \
                     patch.object(singer.os.path, 'exists', return_value=True), redirect_stdout(io.StringIO()):
                    module.main()
                self.assertEqual((L.CONFIG['artist'], L.CONFIG['title']), ('Example Artist', 'Example Song'))
                self.assertEqual((C.CONFIG['artist'], C.CONFIG['title']), ('Example Artist', 'Example Song'))
                with patch.object(C, 'identify', return_value=('Example Artist', 'Example Song', 'flags')) as identify, \
                     patch.object(L, '_lookup', return_value={'status': 'nomatch'}):
                    L.resolve('neutral.wav', [], word_loader=lambda *_: [], use_cache=False)
                identify.assert_called_once_with('neutral.wav', 'Example Artist', 'Example Song', for_lookup=True)
        finally:
            C.CONFIG.update(old_c)
            L.CONFIG.update(old_l)

    def test_opening_estimate_and_settlement_are_not_modulations(self):
        import harmony as H
        import key_opening as K
        original = {'key': 'A major', 'changes': [{'at': 0, 'to': 'E major'},
                    {'at': 30, 'to': 'A major'}, {'at': 90, 'to': 'B major'}]}
        before = copy.deepcopy(original)
        result = K.settled(original, (9, 'major'), H.parse_key)
        self.assertEqual(result['changes'], [{'at': 90, 'to': 'B major'}])
        self.assertEqual(K.settled(result, (9, 'major'), H.parse_key), result)
        self.assertEqual(original, before)
        self.assertEqual(H.key_spans(original, (9, 'major')), [(0., (9, 'major')), (90., (11, 'major'))])
        # An opening which never returns home cannot supply an established home.
        unsettled = dict(changes=[{'at': 10, 'to': 'E major'}, {'at': 60, 'to': 'B major'}])
        self.assertNotIn('changes', K.settled(unsettled, (9, 'major'), H.parse_key))
        normal = dict(changes=[{'at': 40, 'to': 'E major'}, {'at': 100, 'to': 'A major'}])
        self.assertEqual(K.settled(normal, (9, 'major'), H.parse_key), normal)

    def test_opening_v_description_requires_chord_evidence(self):
        import key_opening as K
        import music
        evs = [dict(t0=0, t1=8, root=4, q='maj', conf=.8),
               dict(t0=8, t1=12, root=9, q='maj', conf=.8)]
        self.assertEqual(K.dominant_opening(evs, (9, 'major'), music.NAMES),
                         'likely opens on V (E) before settling in A')
        for field, value in [('conf', .1), ('root', 3), ('q', 'min'), ('t0', 9)]:
            unclear = copy.deepcopy(evs)
            unclear[0][field] = value
            self.assertIsNone(K.dominant_opening(unclear, (9, 'major'), music.NAMES))
        self.assertIsNone(K.dominant_opening(evs[:1], (9, 'major'), music.NAMES))

    def dominant_chroma(self, tonic=7, minor=False, distinguishing=0.1):
        import music as M
        profile = M.KS_MIN if minor else M.KS_MAJ
        home = 4 if minor else 0
        local = np.roll(profile, tonic).copy()
        local[1 if minor else 6] = distinguishing
        if minor:
            local[3] = 5.0  # E harmonic minor's D# is not B minor's distinguishing C#.
        fps = M.SR / M.HOP
        return np.concatenate([np.repeat(np.roll(profile, home)[:, None], int(120 * fps), axis=1),
                               np.repeat(local[:, None], int(60 * fps), axis=1)], axis=1)

    def test_dominant_pull_is_auditable_and_its_return_is_not_a_new_move(self):
        import harmony as H
        import key_opening as K
        import music as M
        chroma = self.dominant_chroma()
        before = chroma.copy()
        raw = M.analyze_key(chroma, 180)
        change = raw['changes'][0]
        self.assertEqual(change['to'], 'G major')
        self.assertEqual(change['suppressed'], 'dominant pull')
        self.assertEqual(change['evidence']['pitch'], 'F#')
        self.assertGreaterEqual(len(change['evidence']['windows']), 3)
        # Preserve the raw return too, but do not print a return from a move we withheld.
        raw['changes'].append(dict(at=175., to='C major'))
        saved = copy.deepcopy(raw)
        settled = K.settled(raw, (0, 'major'), H.parse_key)
        self.assertNotIn('changes', settled)
        self.assertEqual(settled['raw_changes'], raw['changes'])
        self.assertEqual(H.key_spans(settled, (0, 'major')), [(0., (0, 'major'))])
        self.assertEqual(K.settled(settled, (0, 'major'), H.parse_key), settled)
        self.assertEqual(raw, saved)
        np.testing.assert_array_equal(chroma, before)

    def test_real_dominant_move_with_distinguishing_pitch_is_kept(self):
        import music as M
        raw = M.analyze_key(self.dominant_chroma(distinguishing=2.88), 180)
        self.assertEqual(raw['changes'][0]['to'], 'G major')
        self.assertNotIn('suppressed', raw['changes'][0])

    def test_harmonic_minor_leading_tone_does_not_prove_dominant_key(self):
        import music as M
        raw = M.analyze_key(self.dominant_chroma(tonic=11, minor=True), 180)
        self.assertEqual(raw['key'], 'E minor')
        self.assertEqual(raw['changes'][0]['to'], 'B minor')
        self.assertEqual(raw['changes'][0]['suppressed'], 'dominant pull')
        self.assertEqual(raw['changes'][0]['evidence']['pitch'], 'C#')

    def test_other_intervals_and_subdominant_are_untouched(self):
        import music as M
        for tonic in (2, 5):
            chroma = self.dominant_chroma(tonic=tonic)
            raw = M.analyze_key(chroma, 180)
            self.assertEqual(raw['changes'][0]['to'], M.NAMES[tonic] + ' major')
            self.assertNotIn('suppressed', raw['changes'][0])

    def test_one_supported_window_protects_a_dominant_move(self):
        import music as M
        chroma = self.dominant_chroma()
        fps = M.SR / M.HOP
        chroma[6, int(140 * fps):int(160 * fps)] = 4.0
        raw = M.analyze_key(chroma, 180)
        self.assertEqual(raw['changes'][0]['to'], 'G major')
        self.assertNotIn('suppressed', raw['changes'][0])

    def test_later_supported_dominant_run_in_stale_span_is_kept(self):
        import music as M
        fps = M.SR / M.HOP
        chroma = np.concatenate([self.dominant_chroma(),
                                 np.repeat(M.KS_MAJ[:, None], int(10 * fps), axis=1),
                                 np.repeat(np.roll(M.KS_MAJ, 7)[:, None], int(40 * fps), axis=1)], axis=1)
        raw = M.analyze_key(chroma, 230)
        self.assertEqual(len(raw['changes']), 1)  # the brief home return was not emitted
        self.assertEqual(raw['changes'][0]['to'], 'G major')
        self.assertNotIn('suppressed', raw['changes'][0])

    def test_dominant_evidence_is_rechecked_against_fused_home(self):
        import music as M
        chroma = self.dominant_chroma()
        raw = M.analyze_key(chroma, 180)
        saved = copy.deepcopy(raw)
        # G is no longer V if fusion establishes E minor as the working home.
        checked = M.dominant_pull(raw, chroma, 180, (4, 'minor'))
        self.assertNotIn('suppressed', checked['changes'][0])
        self.assertNotIn('evidence', checked['changes'][0])
        self.assertEqual(raw, saved)

    def test_missing_pitch_evidence_cannot_suppress_a_change(self):
        import music as M
        raw = dict(key='C major', changes=[dict(at=60., to='G major')])
        for value in (0., float('nan')):
            chroma = np.full((12, int(120 * M.SR / M.HOP)), value)
            self.assertEqual(M.dominant_pull(raw, chroma, 120, (0, 'major')), raw)

    def test_all_harmony_views_share_filtered_key_changes(self):
        import harmony as H
        import key_opening as K
        kinfo = K.settled({'changes': [{'at': 0, 'to': 'E major'}, {'at': 30, 'to': 'A major'},
                                     {'at': 60, 'to': 'E major', 'suppressed': 'dominant pull',
                                      'evidence': {'pitch': 'Eb'}}, {'at': 75, 'to': 'A major'},
                                     {'at': 90, 'to': 'B major'}]}, (9, 'major'), H.parse_key)
        result = dict(kinfo=kinfo, glob=(9, 'major'), spans=H.key_spans(kinfo, (9, 'major')),
                      key_verdict=dict(key='A major', line='A major'), bpm=100, evs=[], secs=[], dur=120,
                      tc=dict(T=np.zeros(60), step=2), nvl=[], csurp=[], dips=[], tag=[], refrain={},
                      moves=[], pedals=[], br=[], drops=[], swells=[], con=[], lines=[],
                      path='neutral.wav', appo=[], ents=[], F={}, cads=[], notes=[])
        with patch.dict(H.os.environ, {'RAVEN_NO_INSTRUMENTS': '1'}), patch.object(H, 'R_STABLE', []):
            texts = ['\n'.join(H.block_chords(result)), '\n'.join(H.block_surprise(result)),
                     '\n'.join(H.block_chills(result)), str(H.build_timeline(result))]
            with patch('ache.analyze', return_value={'status': 'unavailable'}):
                data = H.to_json(result)
            self.assertEqual(data['key_changes'], kinfo['raw_changes'])
            self.assertEqual(data['key_changes'][2]['suppressed'], 'dominant pull')
        for i, text in enumerate(texts):
            self.assertNotIn('0:00', text)
            self.assertNotIn('0:30', text)
            self.assertNotIn('1:00', text)
            self.assertNotIn('1:15', text)
            self.assertNotIn('E major', text)
            self.assertIn('90' if i == 3 else '1:30', text)
            self.assertIn('B major', text)

    def test_whisper_retry_is_degraded_after_notes_are_collected(self):
        import raven_ears_plus as R
        def model(*args, **kwargs):
            if kwargs['device'] == 'cuda':
                raise RuntimeError('GPU busy')
            return SimpleNamespace(transcribe=lambda *a, **k: ([], None))
        @S.engine
        def run():
            notes = []
            words = R._whisper_words(np.ones(16000), notes)
            S.collect(S.speech_rows(dict(words=words, words_status=None), notes))
            return {}
        with patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=model)}), \
             patch.object(R, '_preload_cuda12'), patch.object(R, '_pick_device', return_value='cuda'), \
             patch.object(R, '_torch', return_value=None), patch('lyrics_shared.whisper_name_kwargs', return_value={}):
            rows = run()['stage_status']
        self.assertTrue(any(r['status'] == 'degraded' and 'CPU int8' in r.get('reason', '') for r in rows))
        self.assertFalse(any(r['status'] == 'failed' for r in rows))

    def test_song_retry_notes_are_not_reclassified_as_final_failures(self):
        import lrclib as L
        notes = []
        def words(*_):
            notes.extend(['lyrics warning: CUDA FAILED to load', 'lyrics: whisper on cpu (int8), no VAD'])
            return [('hello', 1, 2)]
        @S.engine
        def run():
            with patch.object(L, 'get', return_value={'status': 'nomatch'}):
                L.resolve('neutral.wav', notes, 10, word_loader=words)
            S.collect(S.engine_notes(notes, 'music'))
            return {}
        rows = run()['stage_status']
        self.assertFalse(any(r['status'] == 'failed' for r in rows))
        self.assertIn('CPU int8', S.summary(rows))
        self.assertTrue(any(r['status'] == 'failed' for r in
                            S.engine_notes(['vocals: tagger unavailable (RuntimeError)'], 'music')))

    def test_missing_and_broken_harmony_inputs_remain_distinct(self):
        import harmony as H
        import singer
        @S.engine
        def run():
            with patch.object(H, 'bass_stem_path', return_value=None):
                self.assertIsNone(H.bass_track('neutral.wav', '/tmp/unused', use_cache=False))
            with patch.object(singer, 'beats_downbeats', side_effect=RuntimeError('tracker broken')):
                self.assertIsNone(H.tracked_downbeats('neutral.wav', '/tmp/unused', use_cache=False))
            return {}
        rows = run()['stage_status']
        self.assertIn(S.outcome('bass pitch', 'skipped', 'no cached bass stem'), rows)
        self.assertIn(S.outcome('downbeats', 'failed', 'RuntimeError: tracker broken'), rows)

    def test_refrain_audio_failure_and_missing_stem_are_recorded(self):
        import harmony as H
        words = [(w, i * 3 + j * .5, i * 3 + j * .5 + .4)
                 for i in range(4) for j, w in enumerate(('paper', 'boat'))]
        @S.engine
        def run(stem):
            with patch.object(H, 'R_BEST', [('paper boat', [[0], [9]], [1, 1])]), \
                 patch.object(H, 'vocal_stem_path', return_value=stem), \
                 patch.object(H, 'load_vocal_22k', side_effect=RuntimeError('stem decoder broken')):
                H.refrain_check('neutral.wav', words, 15)
            return {}
        self.assertIn('refrain audio check: skipped: no vocal stem cached', S.summary(run(None)['stage_status']))
        self.assertIn('refrain audio check: failed: template match failed (RuntimeError: stem decoder broken)',
                      S.summary(run('neutral-stem.wav')['stage_status']))

    def test_tagger_cpu_retries_and_six_stem_switch_are_recorded(self):
        import instruments as I
        import music
        import raven_ears_plus as R
        @S.engine
        def instrument():
            with patch.object(I, '_stem_dir', return_value=('/tmp/unused', True)), \
                 patch.object(I.os.path, 'exists', return_value=False), \
                 patch.object(R, '_torch', return_value=None), \
                 patch.object(R, '_pick_device', return_value='cuda'), \
                 patch.object(R, '_load_tagger', side_effect=[RuntimeError('GPU busy'), (None, [])]), \
                 patch.object(R, 'load_audio', side_effect=RuntimeError('stop before audio')), \
                 patch.dict(I.os.environ, {'RAVEN_NO_SIX_STEMS': '1'}):
                with self.assertRaises(RuntimeError):
                    I.analyse('neutral.wav', use_cache=False)
            return {}
        rows = instrument()['stage_status']
        self.assertIn('instrument tagger: degraded: RuntimeError: GPU busy; retried on CPU', S.summary(rows))
        self.assertIn('six-stem naming: skipped: RAVEN_NO_SIX_STEMS', S.summary(rows))
        @S.engine
        def vocal():
            with patch.object(R, 'load_audio', return_value=np.ones(32000)), \
                 patch.object(R, '_pick_device', return_value='cuda'), \
                 patch.object(R, '_tag_windows', side_effect=[RuntimeError('GPU busy'),
                              (np.array([1.]), np.array([[.8]]), ['Singing'])]):
                self.assertIsNotNone(music.analyze_vocals_tagger('neutral.wav', []))
            return {}
        self.assertIn('vocal tagger: degraded: RuntimeError: GPU busy; retried on CPU', S.summary(vocal()['stage_status']))

    def test_speech_eligibility_skips_and_missing_file_cli(self):
        import singer
        rows = S.speech_rows(dict(events={'breath': {'skipped': 'too few breaths'}},
                                 affect={'ok': False}, emphasis={'ok': False, 'reason': 'fewer than 4 words'}), [])
        text = S.summary(rows)
        for reason in ('too few breaths', 'no usable voiced windows', 'fewer than 4 words'):
            self.assertIn(reason, text)
        output = io.StringIO()
        with patch.object(singer.os.path, 'exists', return_value=False), redirect_stdout(output):
            self.assertEqual(singer.main(['nonexistent-neutral.wav']), 1)
        self.assertIn('failed', output.getvalue())
        self.assertTrue(output.getvalue().endswith('and now, you\n'))

    def test_whisper_import_failure_and_exhausted_retries_are_failed(self):
        import raven_ears_plus as R
        @S.engine
        def run():
            self.assertIsNone(R._whisper_words(np.ones(16000), []))
            return {}
        with patch.dict(sys.modules, {'faster_whisper': None}), patch.object(R, '_preload_cuda12'):
            self.assertIn('Whisper: failed: faster-whisper not importable', S.summary(run()['stage_status']))
        def broken(*a, **k):
            raise RuntimeError('model broken')
        with patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=broken)}), \
             patch.object(R, '_preload_cuda12'), patch.object(R, '_pick_device', return_value='cpu'), \
             patch.object(R, '_torch', return_value=None):
            rows = run()['stage_status']
        self.assertIn('Whisper: failed: all transcription attempts failed', S.summary(rows))
        self.assertFalse(any(r['status'] == 'skipped' for r in rows))

    def test_supplied_word_diagnostics_never_become_whisper_guesses(self):
        rows = S.speech_rows(dict(words=[('hello', 1, 2)], words_status=None),
                             ['words: 1 over-long word capped'], transcribed=False)
        self.assertFalse(any(r['stage'] in ('Whisper', 'words') for r in rows))

    def test_retry_exception_notes_remain_degraded_and_skipped_crashes_fail(self):
        import raven_ears_plus as R
        @S.engine
        def run():
            notes = []
            def worker(device):
                if device == 'cuda':
                    raise RuntimeError('GPU busy')
                return {}
            with patch.object(R, '_pick_device', return_value='cuda'), patch.object(R, 'release_models'):
                events, _ = R._with_fallback('events', worker, notes)
            S.collect(S.speech_rows(dict(events=events), notes))
            return {}
        rows = run()['stage_status']
        self.assertFalse(any(r['status'] == 'failed' for r in rows))
        crash = S.engine_notes(['vocal quality skipped (RuntimeError: worker broken)'], 'music')
        self.assertEqual(crash[0]['status'], 'failed')

    def test_new_and_old_views_keep_gaps_and_listener_role(self):
        import modes
        import slim
        raw = '=================== music\nMUSIC: neutral.wav · 0:30\nNOTES: all audio stages ran\n=================== singer\nNOTES: events: failed: missing weights\n'
        for text in (modes.render(raw, 'vibe'), slim.render(raw)):
            self.assertIn("DIDN'T HEAR:", text)
            self.assertIn('missing weights', text)
            self.assertNotIn('all audio stages ran', text.split("DIDN'T HEAR:")[1])
            self.assertNotIn('meaning', text)
            self.assertEqual(text.splitlines()[-1], 'and now, you')
        self.assertIn('singer engine skipped or produced no card', modes.render(raw, 'vibe'))


def run_checks(check, tmp=None):
    for test in unittest.defaultTestLoader.loadTestsFromTestCase(HonestBugs):
        result = unittest.TestResult()
        test.run(result)
        check('honest regression: ' + test._testMethodName.removeprefix('test_').replace('_', ' '),
              result.wasSuccessful(), '\n'.join(message for _, message in result.errors + result.failures))


class HumAlongTests(unittest.TestCase):
    """Fox, 16 Nov 2026: label the melody so a reader knows what the letters are for."""

    def test_hum_along_follows_header_once_and_fits(self):
        import listen_melody as LM
        from stage_status import NOTES_WIDTH
        self.assertLessEqual(len(LM.HUM_ALONG), NOTES_WIDTH)
        text = "MELODY: approximate vocal ABC · K:G · middle\nHOOK: x\nK:G\nE, B, G,\n"
        song = {"melody": {"status": "ran", "json": {"hook_text": text}, "text": text}, "cards": {}, "manifest": {}}
        out = LM.render(song, hook=True)
        rows = out.splitlines()
        self.assertEqual(rows[1], LM.HUM_ALONG)
        self.assertEqual(out.count("HUM ALONG"), 1)
        song["melody"]["json"]["hook_text"] = out
        self.assertEqual(LM.render(song, hook=True).count("HUM ALONG"), 1)


class RealProofRegressions(unittest.TestCase):
    """Found on the real LRPD listen, 16 Nov 2026, after the fixture passed."""

    def test_timeline_event_rows_are_not_lyrics(self):
        import lyrics_shared as LS
        card = ("LYRICS: LRCLIB synced lyrics; supplied times (≈)\n"
                "[2:40≈] Eeh-eeh-eeh-eeh, eeh-eeh-eeh\n"
                "NOTES: x\n"
                "TIMELINE\n"
                "[2:41] ♪ drop-out: the level falls 6 dB\n")
        name, rows = LS.supplied_lyrics(None, card)
        self.assertEqual(name, "LRCLIB")
        self.assertEqual([r["text"] for r in rows], ["Eeh-eeh-eeh-eeh, eeh-eeh-eeh"])

    def test_wordless_rows_merge_like_old_wording(self):
        import slim
        q = '"Eeh-eeh-eeh-eeh, eeh-eeh-eeh"'
        rows = [f"1:58  1:58–1:59: wordless singing (lyrics: {q})",
                f"1:59  1:59–2:09: wordless singing (lyrics: {q})",
                f"2:09  2:09–2:17: wordless singing (lyrics: {q})"]
        self.assertEqual(slim._merge_word_gaps(rows), [f"1:58  1:58–2:17: wordless singing (lyrics: {q})"])

    def test_line_stamped_just_before_span_counts(self):
        import lyrics_shared as LS
        rows = [dict(t=128.6, text="Eeh-eeh-eeh-eeh, eeh-eeh-eeh")]
        self.assertTrue(LS.supplied_gap_description(129.4, 137.0, rows).startswith("wordless singing"))
        self.assertIsNone(LS.supplied_gap_description(131.0, 137.0, rows))


if __name__ == '__main__':
    unittest.main()

