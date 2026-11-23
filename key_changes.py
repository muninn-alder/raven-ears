"""Opt-in CNN key trajectories. Crop support is a range, never a boundary."""
import json
from itertools import groupby
import math
import os
from pathlib import Path
import re
import subprocess
import time

VERSION = 'cnn-key-changes-1'
KEYS = ['A major', 'Bb major', 'B major', 'C major', 'Db major', 'D major',
        'Eb major', 'E major', 'F major', 'F# major', 'G major', 'Ab major',
        'A minor', 'Bb minor', 'B minor', 'C minor', 'C# minor', 'D minor',
        'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor']
NOTE = ('from 20-second slices; times are a range, not a moment; '
        'relative major/minor moves are not claimed')
LATE_NOTE = 'late ranges use changes in pitch-class energy alongside the crop votes'


def canonical(key):
    match = re.fullmatch(r'([A-G])([b#♭♯]?) (major|minor)', key)
    if not match:
        raise ValueError('invalid key label')
    return (({'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[match[1]]
             + {'': 0, '#': 1, '♯': 1, 'b': -1, '♭': -1}[match[2]]) % 12, match[3])


def move(a, b):
    pa, ma = canonical(a)
    pb, mb = canonical(b)
    up = (pb - pa) % 12
    if ma != mb and ((ma == 'minor' and up == 3) or (ma == 'major' and up == 9)):
        return 'relative major/minor', False
    names = {0: 'same tonic', 1: 'up a semitone', 2: 'up a whole tone', 3: 'up a minor third',
             4: 'up a major third', 5: 'up a fourth', 6: 'tritone', 7: 'up a fifth',
             8: 'down a major third', 9: 'down a minor third', 10: 'down a whole tone', 11: 'down a semitone'}
    return names[up] + (f', {ma} to {mb}' if ma != mb else ''), ma == mb and up in (1, 2, 3, 4, 8, 9)


def windows(duration):
    """Frozen bench geometry: full 20s crops at 5s hops, plus the last full crop."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('invalid duration')
    if duration < 20:
        return []
    starts = list(range(0, math.floor((duration - 20) / 5) * 5 + 1, 5))
    if duration - 20 > starts[-1]:
        starts.append(duration - 20)
    return [(s, s + 20) for s in starts]


def probability_row(start, end, probs):
    if (len(probs) != 24 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)
            or abs(sum(probs) - 1) > .001):
        raise ValueError('expected 24 normalized CNN probabilities')
    best, second = sorted(range(24), key=lambda i: -probs[i])[:2]
    return dict(kind='fixed20', start=start, end=end, top=KEYS[best], second=KEYS[second],
                p=probs[best], margin=probs[best] - probs[second], probs=probs)


def _event(source, target):
    name, _ = move(source['key'], target['key'])
    return dict(from_key=source['key'], to=target['key'], move=name,
                range=[source['last_start'], target['start'] + 20],
                source_support=[source['start'], source['end']],
                destination_support=[target['start'], target['end']])


def _crop_run(group):
    return dict(key=group[0]['top'], start=group[0]['start'], end=group[-1]['end'],
                last_start=group[-1]['start'], n=len(group), min_p=min(r['p'] for r in group))


def _late_runs(fixed):
    """Short crop runs can propose a key for independent harmonic checks."""
    if not fixed:
        return []
    stop = fixed[-1]['end']
    tail = [r for r in fixed if r['end'] >= stop - 120]

    def strong(row):
        return canonical(row['top']) if row['p'] >= .55 and row['margin'] >= .15 else None

    runs = []
    for key, group in groupby(tail, key=strong):
        group = list(group)
        if key and len(group) >= 3 and all(0 < b['start'] - a['start'] <= 5.01
                                          for a, b in zip(group, group[1:])):
            runs.append(_crop_run(group))
    # An ambiguous ending can alternate between the new signature and a
    # neighbour. Keep the terminal crop's repeated label as a proposal only.
    last = tail[-1]
    if (not runs or runs[-1]['end'] < stop - 8) and strong(last):
        group = [last]
        for row in reversed(tail[:-1]):
            if group[-1]['start'] - row['start'] > 20:
                break
            if strong(row) == strong(last):
                group.append(row)
        if len(group) >= 3:
            runs.append(_crop_run(group[::-1]))
    return runs


def _split_late_return(claims, fixed):
    """Recognize a short reprise of a previously established crop key."""
    if len(claims) < 3:
        return claims
    first, back, last = claims[-3:]
    old, middle, final = (canonical(k) for k in (last['from_key'], back['from_key'], last['to']))
    if (canonical(first['from_key']) != old or canonical(first['to']) != middle
            or canonical(back['to']) != old or len({old[1], middle[1], final[1]}) != 1
            or (middle[0] - old[0]) % 12 not in (1, 2, 3)
            or (final[0] - middle[0]) % 12 not in (1, 2, 3)
            or last['range'][1] < fixed[-1]['end'] - 50):
        return claims
    for run in _late_runs(fixed):
        if (canonical(run['key']) != middle or run['min_p'] < .70
                or not last['range'][0] < run['start']
                or not 0 <= last['destination_support'][0] - run['end'] <= 10):
            continue
        old_start, old_end = last['source_support']
        # A moderate old-key winner after the frozen support ended is still
        # evidence for the left edge of this crop envelope.
        preceding = next((r for r in reversed(fixed)
                          if old_end - 20 < r['start'] < run['start']
                          and canonical(r['top']) == old and r['p'] >= .55), None)
        new_start, new_end = last['destination_support']
        leading = next((r for r in fixed if run['last_start'] < r['start'] < new_start
                        and canonical(r['top']) == final and r['p'] >= .55 and r['margin'] >= .15), None)
        source = dict(key=last['from_key'], start=old_start, end=old_end, last_start=last['range'][0])
        if preceding:
            source.update(end=preceding['end'], last_start=preceding['start'])
        target = dict(key=last['to'], start=leading['start'] if leading else new_start, end=new_end)
        return claims[:-1] + [_event(source, run), _event(run, target)]
    return claims


def analyze_rows(rows, include_late=False):
    """Production reducer for both fresh crops and saved bench rows.

    The main runs retain the frozen relative-withholding rule. A later same-key
    run replaces the previous support span; it must not swallow an intervening
    departure. Short runs only extend the result with a supported late reprise
    or serve as proposals for the separate harmony check.
    """
    fixed = sorted((r for r in rows if r['kind'] == 'fixed20'), key=lambda r: r['start'])
    current, runs = [], []
    for row in fixed + [None]:
        if row is not None:
            canonical(row['top'])
            if (any(not math.isfinite(row[k]) for k in ('start', 'end', 'p', 'margin'))
                    or row['start'] < 0 or abs(row['end'] - row['start'] - 20) > .001
                    or not 0 <= row['p'] <= 1 or not 0 <= row['margin'] <= 1):
                raise ValueError('invalid CNN crop row')
        ok = row is not None and row['p'] >= .90 and row['margin'] >= .20
        continues = (ok and current and canonical(row['top']) == canonical(current[-1]['top'])
                     and 0 < row['start'] - current[-1]['start'] <= 5.01)
        if current and not continues:
            if len(current) >= 3 and current[-1]['end'] - current[0]['start'] >= 30:
                runs.append(_crop_run(current))
            current = []
        if ok:
            current.append(row)
    current = None
    claims, withheld = [], []
    for run in runs:
        if current and canonical(run['key']) != canonical(current['key']):
            _, allowed = move(current['key'], run['key'])
            # The old label need not occupy its entire crop. Starting at the
            # first ambiguous crop (or old crop END) would falsely exclude part
            # of that uncertainty. Likewise include the entire first new crop.
            # This is an evidence envelope, conditional on those crop labels;
            # a CNN cannot guarantee a true boundary or exclude unseen keys.
            event = _event(current, run)
            (claims if allowed else withheld).append(event)
        current = run
    result = dict(version=VERSION, status='ran', slice_seconds=20, hop_seconds=5,
                  windows=len(fixed), claims=_split_late_return(claims, fixed), withheld=withheld, states=runs)
    if include_late and not claims:
        result['late_runs'] = _late_runs(fixed)
    return result


def scan(path, duration, cache_base=None):
    """Sequential CPU-only crops; no model/audio work unless explicitly requested."""
    started = time.monotonic()
    try:
        schedule = windows(duration)
        cf = Path(cache_base + '.key-changes.json') if cache_base else None
        if cf and cf.is_file():
            saved = json.loads(cf.read_text())
            if (saved.get('version') == VERSION and
                    [(r['start'], r['end']) for r in saved['rows']] == schedule):
                return dict(analyze_rows(saved['rows'], include_late=True), cached=True,
                            elapsed_s=time.monotonic() - started)
        import keyfuse
        py = keyfuse.madmom_python() if schedule else None
        if schedule and not py:
            return dict(version=VERSION, status='skipped', reason='venv-madmom not installed', claims=[])
        env = dict(os.environ, RAVEN_EARS_DEVICE='cpu', CUDA_VISIBLE_DEVICES='')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            try:
                env[name] = str(max(1, min(4, int(env.get(name, '4')))))
            except ValueError:
                env[name] = '4'
        rows = []
        for start, end in schedule:
            result = subprocess.run(['nice', '-n', '10', py, str(Path(__file__).with_name('madmom_key.py')),
                                     str(path), '--start', str(start), '--end', str(end)],
                                    capture_output=True, text=True, timeout=300, check=True, env=env)
            probs = json.loads(result.stdout.strip().splitlines()[-1])['probs']
            rows.append(probability_row(start, end, probs))
        result = dict(analyze_rows(rows, include_late=True), cached=False, elapsed_s=time.monotonic() - started)
        if cf:
            cf.write_text(json.dumps(dict(version=VERSION, rows=rows)))
        return result
    except Exception as error:
        # No claims from a partial scan. Failure is not evidence of a stable key.
        return dict(version=VERSION, status='failed', reason=f'{type(error).__name__}: {str(error)[:120]}',
                    claims=[], elapsed_s=time.monotonic() - started)


def _scale(key):
    root, mode = key
    steps = (0, 2, 4, 5, 7, 9, 11) if mode == 'major' else (0, 2, 3, 5, 7, 8, 10)
    return {(root + step) % 12 for step in steps}


def _relative(key):
    root, mode = key
    return ((root + 3) % 12, 'major') if mode == 'minor' else ((root - 3) % 12, 'minor')


def _tonic_time(chords, key, start, end):
    root, mode = key
    qualities = ('maj', 'maj7') if mode == 'major' else ('min', 'min7')
    return sum(max(0., min(end, chord['t1']) - max(start, chord['t0']))
               for chord in chords if chord['root'] == root and chord['q'] in qualities)


def _other_scale_roots(chords, key, start, end):
    other = _scale(key) - {key[0]}
    return {c['root'] for c in chords if c['root'] in other
            and min(end, c['t1']) - max(start, c['t0']) >= 1}


def _signature_edge(prefix, fps, source, target, stop):
    """Find a local swing in the pitches that distinguish the two signatures."""
    before_key, after_key = canonical(source['key']), canonical(target['key'])
    old, new = _scale(before_key), _scale(after_key)
    old_only, new_only = sorted(old - new), sorted(new - old)
    # Demand several accidentals. A relative key, borrowed chord or dominant
    # pull can win a crop without showing a durable change in scale.
    if len(new_only) < 3 or (after_key[0] - before_key[0]) % 12 not in (1, 3, 4):
        return None

    def share(start, end):
        energy = prefix[:, round(end * fps)] - prefix[:, round(start * fps)]
        a, b = energy[new_only].sum(), energy[old_only].sum()
        return float((a - b) / max(a + b, 1e-9))

    first = max(10, target['start'] - 4)
    last = min(stop - 10, target['start'] + 10, (prefix.shape[1] - 1) / fps - 10)
    options = []
    for half_second in range(math.ceil(2 * first), math.floor(2 * last) + 1):
        at = half_second / 2
        before, after = share(at - 10, at), share(at, at + 10)
        if before <= -.05 and after >= .08:
            options.append((after - before, at))
    if not options:
        return None
    swing, at = max(options)
    return at, swing, share


def late_signature_changes(result, chroma, rms_db, chords, fps, duration, areas, home):
    """Cross-check ending crop candidates against continuous harmony evidence."""
    import numpy as np

    runs = result.get('late_runs') or []
    if len(runs) < 2 or runs[-1]['end'] < duration - 8 or home[1] not in ('major', 'minor'):
        return []
    n = min(chroma.shape[1], len(rms_db))
    if not n:
        return []
    audible = np.flatnonzero(rms_db[:n] >= np.median(rms_db[:n]) - 20)
    if not audible.size:
        return []
    end = min(duration, (audible[-1] + 1) / fps)
    prefix = np.pad(np.cumsum(chroma[:, :n], axis=1, dtype=float), ((0, 0), (1, 0)))
    changes = []
    established = areas[-1] if areas else None
    for source, target in reversed(list(zip(runs, runs[1:]))):
        if (not 0 < target['start'] - source['last_start'] <= 30
                or target['start'] < duration * 2 / 3):
            break
        edge = _signature_edge(prefix, fps, source, target, end)
        if edge is None:
            break
        at, swing, share = edge
        if (end - at < 14 or (established and at <= established['at'] + 15)
                or min(share(at, (at + end) / 2), share((at + end) / 2, end)) < .08):
            break
        a, b = canonical(source['key']), canonical(target['key'])
        relative_a, relative_b = _relative(a), _relative(b)
        tonic_before = _tonic_time(chords, a, at - 20, at)
        tonic_after = _tonic_time(chords, b, at, end)
        relative_before = _tonic_time(chords, relative_a, at - 20, at)
        relative_after = _tonic_time(chords, relative_b, at, end)
        from_key, to_key = source['key'], target['key']
        # Resolve a major/relative-minor crop ambiguity only if both sides
        # agree on the alternative mode and its decoded tonic is stronger.
        if relative_before >= tonic_before + 3 and relative_after >= tonic_after + 3:
            a, b = relative_a, relative_b
            from_key = next(k for k in KEYS if canonical(k) == a)
            to_key = next(k for k in KEYS if canonical(k) == b)
            tonic_before, tonic_after = relative_before, relative_after
        if (a[1] != b[1] or (changes and b != canonical(changes[-1][1]['from_key']))
                or tonic_after < 3 or (tonic_before < 3 and not established)
                or len(_other_scale_roots(chords, b, at, end)) < 2):
            break
        name, allowed = move(from_key, to_key)
        if not allowed:
            break
        # The best 10-second contrast gives an internal alignment for chords.
        # Keep a symmetric tolerance for the transition in the printed range.
        changes.append((swing, dict(from_key=from_key, to=to_key, move=name, at=at,
                                    range=[at - 3, at + 3], source_support=[at - 10, at - 3],
                                    destination_support=[at + 3, end], method='late signature')))
        end = at
    if len(changes) == 1 and changes[0][0] < .30:
        return []
    changes = [claim for _, claim in reversed(changes)]
    if changes:
        source = canonical(changes[0]['from_key'])
        # Anchor the sequence to the latest named area or the home signature.
        if established:
            if source != canonical(established['to']):
                return []
        elif _scale(source) != _scale(home):
            return []
    return changes


def clock_range(bounds):
    # Round outwards so display precision never shrinks the evidence range.
    a, b = math.floor(bounds[0]), math.ceil(bounds[1])
    return f'{a // 60}:{a % 60:02d}–{b // 60}:{b % 60:02d}'


def lines(result):
    if not isinstance(result, dict) or result.get('status') != 'ran':
        return []
    out = []
    for claim in result.get('claims', []):
        out.append(f"KEY CHANGES: {claim['from_key']} → {claim['to']}, somewhere "
                   f"{clock_range(claim['range'])} ({claim['move']})")
    return out


def card_text(text, result, note=True):
    additions = lines(result)
    if not additions:
        return text
    # Saved/full cards may already contain the harmony copy. Place it once,
    # immediately after the existing whole-song KEY, preserving that line.
    rows = [r for r in text.splitlines() if not r.strip().startswith('KEY CHANGES:')
            and r.strip() not in ('NOTE: ' + NOTE, 'NOTE: ' + LATE_NOTE)]
    at = next((i + 1 for i, r in enumerate(rows) if r.lstrip().startswith('KEY:')), 1)
    indent = rows[at - 1][:len(rows[at - 1]) - len(rows[at - 1].lstrip())] if rows else ''
    if note:
        additions.append('NOTE: ' + (LATE_NOTE if any(c.get('method') == 'late signature'
                                                       for c in result['claims']) else NOTE))
    rows[at:at] = [indent + line for line in additions]
    return '\n'.join(rows) + ('\n' if text.endswith('\n') else '')
