"""Small, model-free listening probes. Arrays are samples x channels, never channels x samples.

These measure changes in a recording, not the producer's intent. No files, models,
network or caches are touched here. Thresholds are prototype heuristics, not probabilities.
"""
from __future__ import annotations

import numpy as np


def audio_array(y, sr):
    if not np.isfinite(sr) or sr < 100 or int(sr) != sr:
        raise ValueError("sample rate must be an integer >= 100 Hz")
    x = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] not in (1, 2) or not len(x):
        raise ValueError("audio must have samples x 1 or 2 channels and at least one sample")
    if not np.isfinite(x).all():
        raise ValueError("audio contains non-finite samples")
    return x


def _db(power):
    return 10 * np.log10(np.maximum(power, 1e-15))


def _blocks(x, sr, seconds):
    size = max(1, round(sr * seconds))
    for start in range(0, len(x), size):
        z = x[start:start + size]
        if len(z) >= size / 2:
            yield start / sr, z


def space(y, sr):
    """One-second mid/side power, L/R correlation and energy balance; silence is unscored.

    side_share = E(side)/(E(mid)+E(side)), not a perceptual width score.
    balance = (E(R)-E(L))/(E(R)+E(L)), not a source's physical location.
    """
    x = audio_array(y, sr)
    frames = []
    for t, z in _blocks(x, sr, 1.0):
        z = z - z.mean(axis=0)
        level = float(_db(np.mean(z * z)))
        frame = dict(t=t, end=t + len(z) / sr, db=level)
        if x.shape[1] == 2:
            l, r = z.T
            el, er = np.mean(l * l), np.mean(r * r)
            mid, side = np.mean(((l + r) / 2) ** 2), np.mean(((l - r) / 2) ** 2)
            frame.update(side_share=float(side / max(mid + side, 1e-15)),
                         side_mid_db=float(_db(side) - _db(mid)),
                         balance=float((er - el) / max(er + el, 1e-15)),
                         correlation=(float(np.mean(l * r) / np.sqrt(el * er))
                                      if min(el, er) > 1e-15 else None))
        frames.append(frame)
    gate = max(-70.0, float(np.percentile([f['db'] for f in frames], 90)) - 35) if frames else -70
    for f in frames:
        f['active'] = f['db'] > gate
        if not f['active']:
            for k in ('side_share', 'side_mid_db', 'balance', 'correlation'):
                f.pop(k, None)
    active = [f for f in frames if f['active']]
    out = dict(status='ok' if active else 'no_evidence', channels=x.shape[1], frames=frames,
               changes=[], pan_changes=[])
    if not active or x.shape[1] != 2:
        return out
    out.update(side_share=float(np.median([f['side_share'] for f in active])),
               balance=float(np.median([f['balance'] for f in active])),
               antiphase=any(f.get('correlation') is not None and f['correlation'] < -0.3
                             for f in active))
    # A held change on both sides of a boundary; never bridge a quiet gap.
    for i in range(3, len(frames) - 2):
        before, after = frames[i - 3:i], frames[i:i + 3]
        if not all(f['active'] for f in before + after):
            continue
        a = np.array([f['side_share'] for f in before])
        b = np.array([f['side_share'] for f in after])
        delta = float(np.median(b) - np.median(a))
        if abs(delta) >= 0.15 and np.ptp(a) < 0.12 and np.ptp(b) < 0.12:
            if not out['changes'] or frames[i]['t'] - out['changes'][-1]['t'] >= 3:
                balanced = all(abs(f['balance']) < 0.15 for f in before + after)
                phase_ok = all(f.get('correlation') is None or f['correlation'] >= -0.3
                               for f in before + after)
                out['changes'].append(dict(t=frames[i]['t'], delta=delta,
                                           before=float(np.median(a)), after=float(np.median(b)),
                                           width_lean=balanced and phase_ok, phase_opposed=not phase_ok))
        a = np.array([f['balance'] for f in before])
        b = np.array([f['balance'] for f in after])
        if abs(np.median(b) - np.median(a)) >= 0.4 and np.ptp(a) < 0.15 and np.ptp(b) < 0.15:
            if not out['pan_changes'] or frames[i]['t'] - out['pan_changes'][-1]['t'] >= 3:
                out['pan_changes'].append(dict(t=frames[i]['t'], balance=float(np.median(b))))
    return out


def production(y, sr):
    """Level dips/returns, periodic RMS movement and repeated flat peaks.

    Stereo power is measured without mono summing, so antiphase never becomes silence.
    A level dip is not automatically an EDM drop; periodicity is not sidechain evidence.
    """
    x = audio_array(y, sr)
    frames = [(t, float(_db(np.mean((z - z.mean(axis=0)) ** 2))))
              for t, z in _blocks(x, sr, 0.05)]
    t = np.array([f[0] for f in frames])
    db = np.array([f[1] for f in frames])
    out = dict(status='ok' if np.any(db > -70) else 'no_evidence', dips=[], swelling=[],
               flat_peak_fraction=0.0)
    if out['status'] != 'ok':
        return out
    # Flat peaks at the file's own rails also survive normalisation. A square wave
    # can make exactly the same measurement, so no distortion diagnosis follows.
    peak = np.max(np.abs(x), axis=0)
    flat = np.zeros(x.shape, dtype=bool)
    if len(x) > 2:
        rail = (np.abs(x) >= peak * 0.995) & (peak > 1e-6) & (np.std(x, axis=0) > 1e-6)
        tol = np.maximum(peak * 1e-5, 1e-12)
        same = np.abs(np.diff(x, axis=0)) <= tol
        flat[1:-1] = rail[1:-1] & same[:-1] & same[1:]
    out['flat_peak_fraction'] = float(flat.sum() / x.size)
    # Compare three held one-second levels: strong -> low -> strong, all internal.
    secs = [(a, float(_db(np.mean((z - z.mean(axis=0)) ** 2)))) for a, z in _blocks(x, sr, 1)]
    for i in range(3, len(secs) - 1):
        before = np.array([v for _, v in secs[i - 3:i]])
        if np.ptp(before) > 4 or np.median(before) < -60:
            continue
        for j in range(i + 1, min(i + 13, len(secs) - 1)):
            low = np.array([v for _, v in secs[i:j]])
            after = np.array([v for _, v in secs[j:j + 2]])
            depth = float(min(np.median(before), np.min(after)) - np.max(low))
            if depth >= 8 and np.ptp(after) < 4:
                if not out['dips'] or secs[i][0] >= out['dips'][-1]['return_t'] + 3:
                    out['dips'].append(dict(t=secs[i][0], return_t=secs[j][0], db=depth))
                break
    # Local ten-second windows, >=4 cycles, 0.5--4 Hz. Remove slow drift.
    from scipy.signal import detrend, find_peaks
    for start in range(0, len(db) - 199, 200):
        v = db[start:start + 200]
        if np.min(v) < max(-70, np.percentile(db, 90) - 35):
            continue
        depth = float(np.percentile(v, 90) - np.percentile(v, 10))
        if depth < 3:
            continue
        v = detrend(v)
        ac = np.correlate(v, v, mode='full')[len(v) - 1:]
        if ac[0] < 1e-8:
            continue
        ac = ac / ac[0]
        peaks, _ = find_peaks(ac)
        peaks = [p for p in peaks if 5 <= p <= 40 and ac[p] >= 0.65]
        if peaks:
            lag = max(peaks, key=lambda p: ac[p])
            out['swelling'].append(dict(t=float(t[start]), end=float(t[start] + 10),
                                        hz=float(20 / lag), db=depth))
    return out


def tail_decay(y, sr, ends):
    """Decay after caller-supplied excitation ends, in seconds from the stem origin.

    This is a gated level-slope probe, NOT blind RT60 or a room classifier. A held
    note, delay, denoising or Demucs artifacts can pass. The caller supplies ends;
    this function cannot know whether the direct voice has actually stopped.
    """
    x = audio_array(y, sr)
    ends = np.asarray(ends, dtype=float)
    if ends.ndim != 1 or not np.isfinite(ends).all() or np.any(ends < 0) or np.any(ends >= len(x) / sr):
        raise ValueError('vocal ends must be finite seconds inside the vocal stem')
    out = []
    for end in sorted(set(ends.tolist())):
        z = x[round(end * sr):round((end + 2.0) * sr)]
        env = np.array([np.mean(v * v) for _, v in _blocks(z, sr, 0.025)])
        if len(env) < 12 or np.max(env) < 1e-7:
            continue
        # Stop before a new attack or noise floor. Fit only a contiguous -5..-25 dB fall.
        db = _db(env)
        rel = db - db[0]
        indices = np.flatnonzero(rel <= -5)
        if not len(indices):
            continue
        first = int(indices[0])
        last = first
        while last + 1 < len(db) and rel[last + 1] >= -25 and db[last + 1] <= db[last] + 2:
            last += 1
        if last - first < 7 or db[first] - db[last] < 15 or np.min(db[first:last + 1]) < -75:
            continue
        times = np.arange(first, last + 1) * 0.025
        values = db[first:last + 1]
        slope, intercept = np.polyfit(times, values, 1)
        fit = slope * times + intercept
        r2 = 1 - np.sum((values - fit) ** 2) / max(np.sum((values - values.mean()) ** 2), 1e-12)
        if slope < 0 and r2 >= 0.9:
            out.append(dict(t=float(end), fall_db=float(db[first] - db[last]),
                            observed_s=float(times[-1] - times[0]),
                            seconds_per_20db=float(-20 / slope), r2=float(r2)))
    return dict(status='ok' if out else 'no_evidence', tails=out)


def pitch_steps(times, midi):
    """Optional existing monophonic pitch track, NaN = unvoiced. Never an autotune detector.

    Count >=0.15 s near-flat runs and abrupt transitions between them. MIDI values
    are continuous semitones; no key, tuning reference or note correctness is assumed.
    """
    t, m = np.asarray(times, dtype=float), np.asarray(midi, dtype=float)
    if t.ndim != 1 or m.shape != t.shape or len(t) < 2 or not np.isfinite(t).all():
        raise ValueError('pitch track needs matching one-dimensional times and midi arrays')
    if np.any(np.diff(t) <= 0) or t[0] < 0 or np.isinf(m).any():
        raise ValueError('pitch times must increase; midi may use NaN, never infinity')
    dt = float(np.median(np.diff(t)))
    if dt > 0.05:
        return dict(status='no_evidence', reason='pitch track is coarser than 50 ms', steps=[])
    runs, start = [], None
    low = high = 0.0
    for i in range(len(m) + 1):
        if i < len(m) and np.isfinite(m[i]):
            if start is None:
                start = i
                low = high = m[i]
            elif t[i] - t[i - 1] <= 1.5 * dt and max(high, m[i]) - min(low, m[i]) <= 0.1:
                low, high = min(low, m[i]), max(high, m[i])
                continue
            else:
                if t[i - 1] - t[start] + dt >= 0.15:
                    runs.append((start, i - 1, float(np.median(m[start:i]))))
                start = i
                low = high = m[i]
        else:
            if start is not None and t[i - 1] - t[start] + dt >= 0.15:
                runs.append((start, i - 1, float(np.median(m[start:i]))))
            start = None
    steps = []
    for a, b in zip(runs, runs[1:]):
        if t[b[0]] - t[a[1]] <= 0.05 and abs(b[2] - a[2]) >= 0.8:
            steps.append(dict(t=float(t[b[0]]), semitones=float(b[2] - a[2])))
    return dict(status='ok' if steps else 'no_evidence', steps=steps)
