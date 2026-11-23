#!/usr/bin/env python3
"""Optional MORE EARS card, reading local audio/stems/text without writing caches.

    python more_ears.py SONG.wav --stems /existing/stem/dir --lyrics words.txt
    python more_ears.py --lyrics words.txt                 # text only

No separation, transcription, lookup or model loading happens here. A failed probe
stays unavailable, never becomes a report of silence. These cards can be read
beside the existing song engines; their analysis paths stay separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import more_ears_audio as A
from ears_handoff import finish_lines
import stage_status as ST  # HONEST CARD: probe outcomes.

# Measurable capabilities leave this list only when that probe supplied evidence.
# Claims requiring a different method remain here even when a proxy succeeds.
LIMITS = {
    'space': 'stereo width',
    'pan': 'stem left/right balance',
    'production': 'level dips, regular swelling and flat peaks',
    'words': 'word patterns',
    'tails': 'vocal-tail decay',
    'pitch': 'abrupt steps in an existing vocal pitch track',
    'room': 'room size and direct/reverberant ratio',
    'effects': 'autotune, saturation, tape, filter or sidechain attribution',
    'voices': 'simultaneous voice count, doubling and call-and-response',
}


def didnt_hear(*, lyrics='missing', credits='not_requested', skipped=None, heard=(), stage_status=None):
    """One honest line for THIS run. Callers supply actual state; no hardware guessing.

    heard contains LIMITS keys for probes with usable evidence. Standing limits
    with no implemented method cannot be dismissed by a caller's capability flag.
    """
    if lyrics not in ('missing', 'supplied', 'transcript', 'failed'):
        raise ValueError('unknown lyrics state')
    if credits not in ('not_requested', 'available', 'failed'):
        raise ValueError('unknown credits state')
    bits = [ST.summary(stage_status)] if stage_status is not None else []
    if lyrics == 'missing':
        bits.append('no lyrics in hand; any transcribed words would be guesses')
    elif lyrics == 'transcript':
        bits.append('words are transcript guesses, not verified lyrics')
    elif lyrics == 'failed':
        bits.append('lyrics input failed; words unavailable')
    if credits == 'failed':
        bits.append('credits lookup failed; names unavailable')
    elif credits == 'not_requested':
        bits.append('credits not looked up')
    bits.extend(f'{name} unavailable ({reason})' for name, reason in (skipped or {}).items())
    implemented = {'space', 'pan', 'production', 'words', 'tails', 'pitch'}
    measured = set(heard) & implemented
    remaining = [label for key, label in LIMITS.items() if key not in measured]
    if remaining:
        bits.append('still unclear or unmeasured: ' + ', '.join(remaining))
    return "DIDN'T HEAR: " + '; '.join(bits or ['no additional input gaps reported'])


def word_patterns(lines, source='supplied'):
    """English lexical counts only. Line positions count nonblank text lines, never song times."""
    if source not in ('supplied', 'transcript'):
        raise ValueError('words source must be supplied or transcript')
    cleaned = [line.strip() for line in lines if line.strip()]
    tokens = [re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", line.casefold().replace('’', "'"))
              for line in cleaned]
    normal = [' '.join(t) for t in tokens]
    counts = Counter(n for n, t in zip(normal, tokens) if len(t) >= 2)
    repeated = [dict(text=n, count=count, lines=[i + 1 for i, v in enumerate(normal) if v == n])
                for n, count in counts.most_common() if count >= 2]
    groups = {'I': {'i', 'me', 'my', 'mine', 'myself'},
              'you': {'you', 'your', 'yours', 'yourself', 'yourselves'},
              'we': {'we', 'us', 'our', 'ours', 'ourselves'}}
    per_line = []
    for row in tokens:
        roots = [token.split("'")[0] for token in row]
        per_line.append({name: sum(t in group for t in roots) for name, group in groups.items()})
    totals = {name: sum(row[name] for row in per_line) for name in groups}
    dominant = []
    for row in per_line:
        winners = [k for k, v in row.items() if v == max(row.values()) and v > 0]
        dominant.append(winners[0] if len(winners) == 1 else None)
    switches = [dict(line=i + 1, before=dominant[i - 1], after=dominant[i])
                for i in range(1, len(dominant))
                if dominant[i - 1] and dominant[i] and dominant[i - 1] != dominant[i]]
    return dict(status='ok' if any(tokens) else 'no_evidence', source=source, repeated=repeated,
                pronouns=totals, switches=switches, line_count=len(cleaned))


def _time(t):
    # Tenths preserve useful sub-second events and never round a minute to :60.
    deci = max(0, round(t * 10))
    return f'{deci // 600}:{(deci % 600) / 10:04.1f}'


def _position(balance):
    if balance < -0.15:
        return 'leans left'
    if balance > 0.15:
        return 'leans right'
    return 'balanced left/right'


def card(result):
    """Every optional text card, including a failed-input card, ends with the listener."""
    lines = ['MORE EARS (local prototypes; causes stay unclear)']
    s = result.get('space', {})
    if s.get('status') == 'ok':
        if s['channels'] == 1:
            lines.append('  SPACE: one-channel file; stereo space unavailable')
        else:
            label = 'near-mono' if s['side_share'] < 0.02 else 'stereo difference present'
            lines.append(f"  SPACE: {label}; median side share {s['side_share']:.2f} [one-second mid/side energy]")
            if s['antiphase']:
                lines.append('  SPACE: some L/R windows oppose in phase; side energy may cancel in mono, not simply sound wide')
            for ev in s['changes']:
                if ev['width_lean']:
                    action = 'opens outward (width lean)' if ev['delta'] > 0 else 'draws inward (width lean)'
                else:
                    action = 'left/right difference grows' if ev['delta'] > 0 else 'left/right difference shrinks'
                qualifier = ' (phase opposition here; apparent width unclear)' if ev['phase_opposed'] else ''
                lines.append(f"  SPACE: {_time(ev['t'])} {action}, side share {ev['before']:.2f} → {ev['after']:.2f}{qualifier} [mid/side energy, not a section label]")
    elif s.get('status') == 'no_evidence':
        lines.append('  SPACE: no usable level in this input; space unclear')
    for name, p in result.get('pan', {}).items():
        if p.get('status') == 'ok' and p['channels'] == 2:
            lines.append(f"  PAN: {name} stem {_position(p['balance'])} ({p['balance']:+.2f}) [stem energy balance; bleed can move it]")
            for ev in p['pan_changes']:
                lines.append(f"  PAN: {name} stem {_time(ev['t'])} {_position(ev['balance'])} [held stem energy balance]")
        elif p.get('status') == 'no_evidence':
            lines.append(f'  PAN: {name} no usable stem level; position unclear')
        elif p.get('channels') == 1:
            lines.append(f'  PAN: {name} one-channel stem; position unavailable')
    p = result.get('production', {})
    if p.get('status') == 'ok':
        for ev in p['dips']:
            lines.append(f"  LEVEL: {_time(ev['t'])} dips ~{ev['db']:.0f} dB, returns {_time(ev['return_t'])} [held RMS dip; musical drop unclear]")
        for ev in p['swelling']:
            lines.append(f"  SWELL: {_time(ev['t'])}–{_time(ev['end'])} regular level movement ~{ev['hz']:.1f}/s [envelope only; sidechain unclear]")
        if p['flat_peak_fraction'] >= 0.001:
            lines.append(f"  PEAKS: repeated flat peaks ({p['flat_peak_fraction']:.1%} of samples) [clipping-like shape; source or effect unclear]")
        if not p['dips'] and not p['swelling'] and p['flat_peak_fraction'] < 0.001:
            lines.append('  PRODUCTION: no clear dip, regular swell or flat-peak candidate under these probes; effects remain unclear')
    elif p.get('status') == 'no_evidence':
        lines.append('  PRODUCTION: no usable level in this input; effects unclear')
    tails = result.get('tails', {})
    for ev in tails.get('tails', []):
        lines.append(f"  TAIL: {_time(ev['t'])} vocal-stem level falls {ev['fall_db']:.0f} dB over {ev['observed_s']:.2f} s (~{ev['seconds_per_20db']:.2f} s per 20 dB) [supplied end; held voice/delay/bleed possible, room unclear]")
    if tails.get('status') == 'no_evidence':
        lines.append('  TAIL: no clean falling tail at supplied ends; dry/close versus reverberant remains unclear')
    pitch = result.get('pitch', {})
    if pitch.get('steps'):
        where = ', '.join(_time(ev['t']) for ev in pitch['steps'][:8])
        noun = 'step' if len(pitch['steps']) == 1 else 'steps'
        lines.append(f"  PITCH: {len(pitch['steps'])} abrupt {noun} between near-flat pitch runs at {where} [supplied track; hard correction unclear]")
    elif pitch.get('reason'):
        lines.append(f"  PITCH: unavailable ({pitch['reason']}); hard correction remains unclear")
    elif pitch:
        lines.append('  PITCH: no clear abrupt-step candidate; hard correction remains unclear')
    words = result.get('words', {})
    if words.get('status') == 'ok':
        origin = 'supplied text, not audio-verified' if words['source'] == 'supplied' else 'transcript guesses'
        lines.append(f'  WORDS: {origin}; English lexical patterns only, meaning is yours')
        for rep in words['repeated'][:3]:
            positions = ', '.join(str(n) for n in rep['lines'])
            lines.append(f"  REPEAT: {rep['text']!r} ×{rep['count']} (nonblank text lines {positions}) [hook candidate by text repetition only]")
        counts = ', '.join(f'{name}={n}' for name, n in words['pronouns'].items())
        lines.append(f'  WORDS: pronoun tokens {counts} [English forms; no speaker or addressee inferred]')
        for ev in words['switches'][:5]:
            lines.append(f"  WORDS: nonblank text line {ev['line']}, pronoun balance {ev['before']} → {ev['after']} [token counts, not a change of narrator]")
    lines.append('  ' + didnt_hear(lyrics=result.get('lyrics_state', 'missing'),
                                  credits=result.get('credits_state', 'not_requested'),
                                  skipped=result.get('unavailable'), heard=result.get('heard', ()),
                                  stage_status=result.get('stage_status')))
    return '\n'.join(ST.add_notes(lines, result.get('stage_status', [])))


@ST.engine
def run(path=None, *, stems=None, lyrics=None, words_source='supplied', vocal_ends=None, pitch_track=None):
    """Read-only runner. stems is an explicit existing directory, never a separation request.

    Optional data files: vocal_ends = JSON seconds; pitch_track = JSON
    {"times": [...], "midi": [...]} with null for unvoiced MIDI frames.
    Returns (result, failed). Probe errors are named on the card and give CLI exit 1.
    """
    result = dict(unavailable={}, heard=[], lyrics_state='missing')
    failed = False

    def probe(name, operation):
        nonlocal failed
        try:
            value = operation()
            ST.mark(name)
            return value
        except Exception as exc:
            ST.mark(name, 'failed', f'{type(exc).__name__}: {exc}')
            result['unavailable'][name] = f'{type(exc).__name__}: {exc}'
            failed = True
            return None

    def read_audio(filename):
        import soundfile as sf
        y, sr = sf.read(filename, dtype='float32', always_2d=True)
        A.audio_array(y, sr)
        return y, sr

    def attach(name, value):
        if value is not None:
            result[name] = value
            if value['status'] == 'ok' and (name != 'space' or value['channels'] == 2):
                result['heard'].append(name)

    if path is not None:
        audio = probe('audio input', lambda: read_audio(path))
        if audio is not None:
            attach('space', probe('stereo probe', lambda: A.space(*audio)))
            attach('production', probe('production probe', lambda: A.production(*audio)))
    else:
        result['unavailable']['audio probes'] = 'not requested'
    vocal = None
    if stems is not None:
        result['pan'] = {}
        for name in ('vocals', 'drums', 'bass', 'other'):
            audio = probe(f'{name} stem', lambda name=name: read_audio(Path(stems) / f'{name}.wav'))
            if audio is None:
                continue
            p = probe(f'{name} balance', lambda: A.space(*audio))
            if p is not None:
                result['pan'][name] = p
                if p['status'] == 'ok' and p['channels'] == 2 and 'pan' not in result['heard']:
                    result['heard'].append('pan')
            if name == 'vocals':
                vocal = audio
    if vocal_ends is not None:
        if vocal is None:
            result['unavailable']['vocal-tail probe'] = 'no readable vocal stem; --stems required'
            failed = True
        else:
            attach('tails', probe('vocal-tail probe', lambda: A.tail_decay(
                *vocal, json.loads(Path(vocal_ends).read_text()))))
    if pitch_track is not None:
        def read_pitch():
            track = json.loads(Path(pitch_track).read_text())
            return A.pitch_steps(track['times'], track['midi'])
        attach('pitch', probe('pitch-track probe', read_pitch))
    if lyrics is not None:
        words = probe('lyrics input', lambda: word_patterns(Path(lyrics).read_text(encoding='utf-8').splitlines(), words_source))
        attach('words', words)
        result['lyrics_state'] = words_source if words and words['status'] == 'ok' else 'failed'
        if words and words['status'] != 'ok':
            result['unavailable']['lyrics input'] = 'empty or no word tokens'
            failed = True
    for name, reason in result['unavailable'].items():
        ST.mark(name, 'skipped' if reason == 'not requested' else 'failed', reason)
    for name, supplied in (('stem balance', stems), ('vocal tails', vocal_ends), ('pitch steps', pitch_track), ('word patterns', lyrics)):
        if supplied is None:
            ST.mark(name, 'skipped', 'not supplied')
    result['stage_status'] = ST.snapshot()
    return result, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs='?')
    parser.add_argument('--stems', help='existing vocals/drums/bass/other.wav directory; read only')
    parser.add_argument('--lyrics', help='UTF-8 text, one line per supplied or transcribed line')
    parser.add_argument('--words-source', choices=('supplied', 'transcript'), default='supplied')
    parser.add_argument('--vocal-ends', help='JSON array of seconds after direct vocal excitation ends; requires --stems')
    parser.add_argument('--pitch-track', help='JSON {times: seconds, midi: continuous semitones or null}')
    args = parser.parse_args(argv)
    from network_policy import local_audio_path
    for audio in (args.file, args.stems):
        if audio:
            local_audio_path(audio, parser)
    if not any((args.file, args.stems, args.lyrics, args.vocal_ends, args.pitch_track)):
        parser.error('supply audio, existing stems, words or a pitch track')
    result, failed = run(args.file, stems=args.stems, lyrics=args.lyrics,
                         words_source=args.words_source, vocal_ends=args.vocal_ends, pitch_track=args.pitch_track)
    print(card(result))
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
