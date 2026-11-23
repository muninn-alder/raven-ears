"""Synthetic and mocked MORE EARS checks, included by the house's --fast runner.

No songs, personal names, model imports, GPU, network or cache writes.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import more_ears as M
import more_ears_audio as A


def run_checks(check):
    sr = 4000
    t = np.arange(12 * sr) / sr
    tone = 0.2 * np.sin(2 * np.pi * 200 * t)
    other = 0.2 * np.sin(2 * np.pi * 331 * t)
    mono = np.column_stack([tone, tone])
    wide = np.column_stack([tone, other])
    anti = np.column_stack([tone, -tone])
    s = A.space(mono, sr)
    check('more ears: dual mono has zero side energy and centered balance',
          s['status'] == 'ok' and s['side_share'] < 1e-10 and abs(s['balance']) < 1e-10)
    s = A.space(wide, sr)
    check('more ears: different L/R signals have side share ~0.5',
          abs(s['side_share'] - 0.5) < 0.01 and not s['antiphase'])
    s = A.space(anti, sr)
    check('more ears: phase-opposed stereo is active, flagged, never mistaken for silence',
          s['status'] == 'ok' and s['antiphase'] and s['side_share'] > 0.99
          and A.production(anti, sr)['status'] == 'ok')
    check('more ears: antiphase card explains mono cancellation',
          'may cancel in mono' in M.card({'space': s}))
    s = A.space(np.column_stack([tone, np.zeros(len(tone))]), sr)
    check('more ears: left-only content has left balance, undefined correlation',
          s['balance'] < -0.99 and s['frames'][0]['correlation'] is None)
    check('more ears: missing channel never becomes a flat-peak detection',
          A.production(np.column_stack([tone, np.zeros(len(tone))]), sr)['flat_peak_fraction'] == 0)
    check('more ears: a constant offset channel cannot count as flattened audio peaks',
          A.production(np.column_stack([tone, np.ones(len(tone))]), sr)['flat_peak_fraction'] == 0)
    check('more ears: right-heavy stem leans right',
          A.space(np.column_stack([tone * 0.1, tone]), sr)['balance'] > 0.9)
    s = A.space(tone, sr)
    check('more ears: one channel is identified without inventing width',
          s['channels'] == 1 and 'side_share' not in s and 'one-channel' in M.card({'space': s}))
    s = A.space(np.column_stack([tone, np.where(t < 6, tone, other)]), sr)
    check('more ears: held stereo opening at 6s is found once',
          len(s['changes']) == 1 and s['changes'][0]['t'] == 6 and s['changes'][0]['delta'] > 0.45)
    s = A.space(np.column_stack([tone, np.where(t < 6, other, tone)]), sr)
    check('more ears: narrowing is also heard',
          len(s['changes']) == 1 and s['changes'][0]['delta'] < -0.45)
    panning = np.column_stack([tone, np.where(t < 6, tone, 0)])
    s = A.space(panning, sr)
    check('more ears: center-to-left movement is balance change, not a stereo-width claim',
          len(s['changes']) == 1 and not s['changes'][0]['width_lean']
          and s['pan_changes'][0]['balance'] < -0.99
          and 'opens outward' not in M.card({'space': s}))
    transient = mono.copy()
    transient[6 * sr:7 * sr, 1] = other[6 * sr:7 * sr]
    check('more ears: one wide hit does not become a held opening', not A.space(transient, sr)['changes'])
    check('more ears: spatial metrics survive a change of overall gain',
          abs(A.space(wide * 0.01, sr)['side_share'] - 0.5) < 0.01)
    gap = wide.copy()
    gap[4 * sr:7 * sr] = 0
    s = A.space(gap, sr)
    check('more ears: quiet windows have no spatial metrics, quiet gaps make no opening event',
          'side_share' not in s['frames'][5] and not s['changes'])
    silence = A.space(np.zeros((sr * 4, 2)), sr)
    check('more ears: silence is no spatial evidence, not near-mono music',
          silence['status'] == 'no_evidence' and 'side_share' not in silence
          and 'near-mono' not in M.card({'space': silence}))
    check('more ears: a DC offset is not stereo space',
          A.space(np.ones((sr * 4, 2)), sr)['status'] == 'no_evidence')
    check('more ears: very short valid audio stays inconclusive',
          A.space(np.zeros(2), sr)['status'] == 'no_evidence')

    p = A.production(tone * np.where((t >= 5) & (t < 7), 0.1, 1), sr)
    check('more ears: held 20dB dip/return has the right times',
          len(p['dips']) == 1 and p['dips'][0]['t'] == 5 and p['dips'][0]['return_t'] == 7
          and abs(p['dips'][0]['db'] - 20) < 0.01)
    check('more ears: fade-out is not a dip and return',
          not A.production(tone * np.maximum(0, 1 - t / 10), sr)['dips'])
    hit = tone.copy()
    hit[int(5 * sr):int(5.1 * sr)] *= 0.01
    check('more ears: short notch is not a held drop', not A.production(hit, sr)['dips'])
    p = A.production(tone * (0.5 + 0.4 * np.sin(2 * np.pi * 2 * t)), sr)
    check('more ears: 2Hz envelope movement is measured, sidechain remains unclear',
          len(p['swelling']) == 1 and abs(p['swelling'][0]['hz'] - 2) < 0.1
          and 'sidechain unclear' in M.card({'production': p}))
    check('more ears: steady tone has no envelope swell', not A.production(tone, sr)['swelling'])
    rng = np.random.default_rng(92)
    irregular = np.repeat(rng.uniform(0.1, 1, 48), sr // 4) * tone
    check('more ears: irregular envelope is not regular swelling', not A.production(irregular, sr)['swelling'])
    p = A.production(np.clip(tone * 4, -0.3, 0.3), sr)
    check('more ears: normalized flat rails remain measurable, no distortion diagnosis',
          p['flat_peak_fraction'] > 0.1 and 'source or effect unclear' in M.card({'production': p}))
    check('more ears: unclipped sine does not produce repeated flat peaks',
          A.production(tone, sr)['flat_peak_fraction'] < 0.001)
    p = A.production(np.zeros(sr), sr)
    check('more ears: quiet production is inconclusive and creates no effect events',
          p['status'] == 'no_evidence' and not p['dips'] and not p['swelling'])

    # Known exponential amplitude fall: -60dB in 1.2s, -20dB in 0.4s.
    tail = tone * np.exp(-np.log(1000) * t / 1.2)
    d = A.tail_decay(tail, sr, [0])
    check('more ears: exponential tail slope gives 0.4s per 20dB, never claims RT60/room',
          d['status'] == 'ok' and abs(d['tails'][0]['seconds_per_20db'] - 0.4) < 0.03
          and 'room unclear' in M.card({'tails': d}) and 'RT60' not in M.card({'tails': d}))
    check('more ears: a dry cutoff, sustained tone and silent tail do not fit a decay',
          all(A.tail_decay(x, sr, [0])['status'] == 'no_evidence'
              for x in (tone * (t < 0.1), tone, np.zeros(len(t)))))
    check('more ears: no acceptable tail never becomes a dry/close assertion',
          'remains unclear' in M.card({'tails': A.tail_decay(tone, sr, [0])}))
    near_floor = tail * 1e-5
    check('more ears: inaudibly weak tail does not get a slope',
          A.tail_decay(near_floor, sr, [0])['status'] == 'no_evidence')

    pt = np.arange(0, 2, 0.01)
    pm = np.where(pt < 1, 60.0, 64.0)
    p = A.pitch_steps(pt, pm)
    check('more ears: abrupt stable note step is flagged as a candidate only',
          len(p['steps']) == 1 and abs(p['steps'][0]['t'] - 1) < 0.01
          and 'hard correction unclear' in M.card({'pitch': p}))
    check('more ears: continuous glissando and vibrato have no flat-step event',
          not A.pitch_steps(pt, 60 + pt * 2)['steps']
          and not A.pitch_steps(pt, 60 + 0.3 * np.sin(2 * np.pi * 5 * pt))['steps'])
    gap_pitch = pm.copy()
    gap_pitch[95:105] = np.nan
    coarse = A.pitch_steps([0, 0.2, 0.4], [60, 64, 64])
    check('more ears: coarse pitch refusal is named, not presented as a negative detection',
          'unavailable (pitch track is coarser than 50 ms)' in M.card({'pitch': coarse}))
    check('more ears: unvoiced gap cannot be an abrupt voiced transition',
          not A.pitch_steps(pt, gap_pitch)['steps'])
    check('more ears: coarse pitch track cannot imply hard correction',
          A.pitch_steps([0, 0.2, 0.4], [60, 64, 64])['status'] == 'no_evidence')
    sparse_t = np.r_[pt[:100], pt[100:] + 2]
    check('more ears: missing pitch timestamps do not create a step',
          not A.pitch_steps(sparse_t, pm)['steps'])

    w = M.word_patterns(['I carry a lantern.', 'You open the gate.', 'We carry a lantern.',
                         'I carry a lantern!', '  ', 'I CARRY A LANTERN.'])
    check('more ears: repetitions ignore case/punctuation but count text lines honestly',
          w['repeated'][0] == {'text': 'i carry a lantern', 'count': 3, 'lines': [1, 4, 5]})
    check('more ears: pronoun counts and lexical switch positions have no meaning claim',
          w['pronouns'] == {'I': 3, 'you': 1, 'we': 1}
          and w['switches'][0] == {'line': 2, 'before': 'I', 'after': 'you'}
          and 'meaning is yours' in M.card({'words': w, 'lyrics_state': 'supplied'}))
    w = M.word_patterns(["I'm here", 'You’re here', "We've arrived", 'yourself ourselves myself'])
    check('more ears: contractions and reflexive forms count as English tokens',
          w['pronouns'] == {'I': 2, 'you': 2, 'we': 2})
    check('more ears: tokens do not count substrings as pronouns',
          M.word_patterns(['ivy youthful welcome music'])['pronouns'] == {'I': 0, 'you': 0, 'we': 0})
    check('more ears: pronoun ties/gaps do not invent a narrator switch',
          not M.word_patterns(['I see you', 'we arrive', 'lanterns glow', 'you arrive'])['switches'])
    check('more ears: empty/punctuation-only words are inconclusive',
          M.word_patterns(['', '...'])['status'] == 'no_evidence')
    w = M.word_patterns(['you arrive', 'you arrive'], 'transcript')
    check('more ears: unverified transcript keeps its provenance on the card',
          'transcript guesses' in M.card({'words': w, 'lyrics_state': 'transcript'}))
    line = M.didnt_hear(credits='failed', skipped={'vocal separation': 'no GPU'})
    check('more ears: honest gaps include missing lyrics, failed credits and explicit GPU skip',
          all(bit in line for bit in ('no lyrics in hand', 'guesses', 'credits lookup failed', 'no GPU')))
    line = M.didnt_hear(lyrics='supplied', credits='available', heard=M.LIMITS.keys())
    check('more ears: implemented evidence shrinks limits, proxies cannot dismiss hard problems',
          'stereo width' not in line and 'word patterns' not in line
          and 'room size' in line and 'simultaneous voice count' in line and 'autotune' in line)
    check('more ears: capability iterators shrink limits as consistently as lists',
          M.didnt_hear(heard=iter(['space', 'words'])) == M.didnt_hear(heard=['space', 'words']))
    check('more ears: every rendered card ends with the listener, including failures',
          all(M.card(r).splitlines()[-1] == 'and now, you'
              for r in ({}, {'space': silence}, {'unavailable': {'backend': 'broken'}})))

    invalid = [lambda: A.space([], sr), lambda: A.space(np.zeros((3, 4)), sr),
               lambda: A.production([np.nan], sr), lambda: A.space([1, 2], 0),
               lambda: A.tail_decay(tone, sr, [13]), lambda: A.tail_decay(tone, sr, [-1]),
               lambda: A.tail_decay(tone, sr, [np.nan]), lambda: A.pitch_steps([0, 0], [60, 62]),
               lambda: A.pitch_steps([0, 1], [60]), lambda: A.pitch_steps([0, 1], [60, np.inf]),
               lambda: M.didnt_hear(lyrics='guessed'), lambda: M.word_patterns(['hi'], 'unknown')]
    raised = []
    for op in invalid:
        try:
            op()
            raised.append(False)
        except ValueError:
            raised.append(True)
    check('more ears: invalid arrays, sample rates, endpoints, tracks and provenance are rejected', all(raised))

    # Runner reads explicit files only. Mock decoding failures and metadata; no models.
    with tempfile.TemporaryDirectory(prefix='more_ears_test_') as root:
        root = Path(root)
        # The existing modes use only tiny formatting hooks. Exercise the real
        # renderers/CLIs with all audio/model backends mocked or disabled.
        import ears_handoff as H
        import music
        import harmony
        import singer
        import instruments
        import raven_ears_plus as speech
        original = ['a card', H.HANDOFF, '']
        check('more ears: terminal handoff is idempotent and does not mutate input',
              H.finish_lines(H.finish_lines(original)) == ['a card', H.HANDOFF]
              and original == ['a card', H.HANDOFF, ''])
        raw = {'probe': 1}
        structured = H.finish_json(raw)
        check('more ears: structured handoff preserves valid JSON and leaves input alone',
              json.loads(json.dumps(structured))['listener'] == H.HANDOFF
              and list(structured)[-1] == 'listener' and raw == {'probe': 1})
        check('more ears: MUSIC text renderer ends with the listener',
              music.format_card({'file': 'neutral.wav', 'duration': 1}).endswith(H.HANDOFF))
        blocks = ('block_sound', 'block_chords', 'block_tension', 'block_resolves', 'block_timeline',
                  'block_lyric_timeline', 'block_contrast', 'block_surprise', 'block_chills')
        with contextlib.ExitStack() as stack:
            for name in blocks:
                stack.enter_context(patch.object(harmony, name, return_value=['a measured line']))
            text = harmony.report({'path': 'neutral.wav', 'dur': 1, 'notes': []})
        check('more ears: HARMONY text renderer ends with the listener', text.endswith(H.HANDOFF))
        with patch.object(harmony, 'build_timeline', return_value=[]):
            data = harmony.to_json(dict(path='neutral.wav', dur=1, key_verdict={'key': None},
                                        evs=[], tc={'step': 1, 'T': []}, br=[], cads=[], con={}, notes=[]))
        check('more ears: HARMONY structured renderer includes the listener', data['listener'] == H.HANDOFF)
        audiofile = root / 'neutral.wav'
        audiofile.touch()
        with patch.object(speech, '_torch', return_value=None), patch.object(speech, 'load_audio', return_value=np.zeros(32000)):
            quiet = speech.analyze(str(audiofile), use_whisper=False, events=False, affect=False, laughs=False)
        check('more ears: SPEECH quiet-input early return includes the listener', quiet['lines'][-1] == H.HANDOFF)
        for structured_mode in (False, True):
            output = io.StringIO()
            with patch.object(speech, 'analyze', return_value=quiet), contextlib.redirect_stdout(output):
                status = speech.main([str(audiofile), '--quiet'] + (['--json'] if structured_mode else []))
            text = output.getvalue().strip()
            check(f'more ears: SPEECH CLI handoff (JSON={structured_mode})',
                  status == 0 and (json.loads(text)['listener'] == H.HANDOFF if structured_mode else text.endswith(H.HANDOFF)))
            output = io.StringIO()
            with patch.object(singer, 'get_stems', return_value=(str(root), True)), \
                 patch('soundfile.info', return_value=SimpleNamespace(duration=1)), \
                 patch('credits.configure_from_args'), patch.dict(os.environ), contextlib.redirect_stdout(output):
                status = singer.main([str(audiofile), '--no-singer', '--no-meter', '--no-instruments',
                                      '--no-stems-timeline'] + (['--json'] if structured_mode else []))
            text = output.getvalue().strip()
            check(f'more ears: SINGER CLI handoff (JSON={structured_mode})',
                  status == 0 and (json.loads(text)['listener'] == H.HANDOFF if structured_mode else text.endswith(H.HANDOFF)))

        for structured_mode in (False, True):
            output = io.StringIO()
            with patch.object(instruments, 'analyse', return_value={'duration': 1}), \
                 patch.object(instruments, 'block', return_value=['INSTRUMENTS: unclear']), \
                 patch.object(instruments, 'events', return_value=[(0, 'a measured change')]), \
                 patch('credits.configure_from_args'), patch.dict(os.environ), contextlib.redirect_stdout(output):
                status = instruments.main([str(audiofile)] + (['--json'] if structured_mode else []))
            text = output.getvalue().strip()
            check(f'more ears: standalone INSTRUMENTS hands off after its events (JSON={structured_mode})',
                  status == 0 and (json.loads(text)['listener'] == H.HANDOFF if structured_mode else text.endswith(H.HANDOFF)))

        # A small real WAV round-trip covers the decoder boundary, not just mocks.
        import soundfile as sf
        sf.write(audiofile, wide[:sr], sr, subtype='FLOAT')
        result, failed = M.run(audiofile)
        check('more ears: real synthetic WAV decoding reaches both probes',
              not failed and result['space']['channels'] == 2 and result['production']['status'] == 'ok')
        words = root / 'words.txt'
        words.write_text('We hold a lantern\nWe hold a lantern\n', encoding='utf-8')
        ends = root / 'ends.json'
        ends.write_text('[0]')
        track = root / 'pitch.json'
        track.write_text(json.dumps({'times': pt.tolist(), 'midi': pm.tolist()}))
        with patch('soundfile.read', side_effect=RuntimeError('decoder broken')):
            result, failed = M.run('missing.wav', lyrics=words)
        text = M.card(result)
        check('more ears: broken decoder is unavailable, not silence, words still work',
              failed and 'decoder broken' in text and 'space' not in result
              and 'words' in result and text.endswith('and now, you'))
        with patch('soundfile.read', return_value=(wide, sr)):
            result, failed = M.run('song.wav', stems=root, lyrics=words, pitch_track=track)
        check('more ears: successful local runner combines stereo, four stem balances, words and pitch',
              not failed and len(result['pan']) == 4
              and {'space', 'pan', 'production', 'words', 'pitch'} <= set(result['heard']))
        with patch('soundfile.read', return_value=(tail[:, None], sr)):
            result, failed = M.run(stems=root, vocal_ends=ends)
        check('more ears: runner accepts explicit tail ends without inventing stereo from mono',
              not failed and result['tails']['status'] == 'ok' and 'pan' not in result['heard'])
        ends.write_text('null')
        with patch('soundfile.read', return_value=(tail[:, None], sr)):
            result, failed = M.run(stems=root, vocal_ends=ends)
        check('more ears: malformed/null end JSON is reported as a failed probe',
              failed and 'vocal-tail probe' in result['unavailable'])
        result, failed = M.run(vocal_ends=ends)
        check('more ears: requested tail probe with no stem reports skipped work',
              failed and 'no readable vocal stem' in M.card(result))
        result, failed = M.run(lyrics=root / 'missing.txt')
        check('more ears: unreadable supplied words report failure rather than successful text-only analysis',
              failed and result['lyrics_state'] == 'failed' and 'lyrics input failed' in M.card(result))
        words.write_text('...\n')
        result, failed = M.run(lyrics=words)
        check('more ears: empty lexical input reports unavailable words',
              failed and 'empty or no word tokens' in M.card(result))
        track.write_text('{broken json')
        result, failed = M.run(pitch_track=track)
        check('more ears: corrupt pitch JSON cannot look like an uncorrected voice',
              failed and 'pitch-track probe' in result['unavailable'] and 'pitch' not in result)
        words.write_text('We hold a lantern\nWe hold a lantern\n')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = M.main(['--lyrics', str(words), '--words-source', 'transcript'])
        check('more ears: text-only CLI is usable and ends with the listener',
              status == 0 and output.getvalue().strip().endswith('and now, you')
              and 'transcript guesses' in output.getvalue())
        output = io.StringIO()
        with patch('soundfile.read', side_effect=RuntimeError('decoder broken')), contextlib.redirect_stdout(output):
            status = M.main(['missing.wav'])
        check('more ears: failed CLI still emits an honest card and nonzero status',
              status == 1 and 'decoder broken' in output.getvalue()
              and output.getvalue().strip().endswith('and now, you'))


if __name__ == '__main__':
    results = []

    def check(name, ok, detail=''):
        results.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    run_checks(check)
    print(f'{sum(results)} passed, {len(results) - sum(results)} failed')
    raise SystemExit(int(not all(results)))
