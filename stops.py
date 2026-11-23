#!/usr/bin/env python3
"""Short accompaniment cuts in four cached stems. CPU arithmetic only; no cache writes.

    python3 stops.py STEM_DIR [--tempo BPM] [--beats-per-bar N] [--json]

Times describe energy edges in separated stems, not score positions. A full silence
means both band and voice fell by the stated relative threshold, not digital zero.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
import re
import textwrap
import zlib

import numpy as np

MIN_STOP_S = 0.3
MIN_BAND_BREAK_S = 0.5           # unvoiced strum gaps need a longer low interval
MAX_STOP_S = 8.0
HOP_S = 0.01
REFERENCE_S = 1.0
REFERENCE_GUARD_S = 0.10
DROP_DB = 12.0
CANDIDATE_DB = 8.0                 # shallower cuts are reported as uncertain
EDGE_START_DB = 3.0
RETURN_DB = 6.0
MAX_FALL_S = 1.5                  # -3 dB to observed floor (capped at 36 dB down)
FALL_FLOOR_DB = 36.0              # deeper separator/numerical floors do not time a cue
FADE_RAMP_S = 0.75                # a sustained, steady dB ramp remains uncertain
FADE_RAMP_R2 = 0.95
CARD_UNCERTAIN_DB = 12.0
STAGGER_S = 1.5
MAX_DECAY_S = 3.0                 # retain slower tails as uncertain, never confirm fades
RETURN_HOLD_S = 0.10
TAIL_GRACE_S = 0.08
LOW_FRACTION = 0.80
STEM_DROP_DB = 10.0
VOICE_HOLD_DB = 6.0              # maximum fall from pre-cut voice; a rise still holds
VOICE_RISE_MAX_DB = 12.0         # ...up to a median rise this size (= DROP_DB). Beyond it the vocal stem
                                 # more likely took over band energy than a singer pushed into the gap
                                 # (Rich Girl 0:19.8: other -16.1 dB, vocals +15.9 dB, drums/bass flat; 11 Nov 2026)
BLEED_DB = 18.0                   # possible band leakage in vocals; never subtracted
ABSENT_DB = 28.0
VOICE_PRESENT_DB = 24.0
INACTIVE_VOICE_DB = 50.0         # Joplin's four vocal residuals are 64–67 dB below the piano (90th percentiles)
SCAN_START_S = 0.30
ENTRY_KIND = "band cuts, vocal stem enters"  # energy alone cannot establish sung pitch
VERSION = 3
EPS = 1e-20                      # numerical floor, not a musical activity threshold
STEMS = ("drums", "bass", "other", "vocals")
KINDS = ("band cuts, voice holds", "full silence", "band break")
FADED_KIND = "faded stop"
FADED_REFERENCE_S = 12.0         # same local context as harmony.find_dips
FADED_SMOOTH_S = 0.25
FADED_R2 = 0.80                  # whole descent, including the modulated tail


def _db(energy):
    return 10 * np.log10(np.maximum(energy, EPS))


def _curves(stem_dir):
    """10 ms non-overlapping mean-square frames, averaged across channels.

    Do not average channels before squaring: antiphase stereo is still sound.
    Stream one second at a time; no full-song sample arrays or audio copies.
    """
    import soundfile as sf
    curves, layout = {}, None
    for stem in STEMS:
        path = Path(stem_dir) / (stem + ".wav")
        if not path.is_file():
            raise ValueError("missing stem: " + path.name)
        try:
            with sf.SoundFile(path) as audio:
                shape = (audio.samplerate, audio.frames)
                if layout is not None and shape != layout:
                    raise ValueError("unaligned stem (sample rate or length): " + path.name)
                layout = shape
                hop = round(audio.samplerate * HOP_S)
                if hop < 1 or abs(hop / audio.samplerate - HOP_S) > 1e-6:
                    raise ValueError("sample rate cannot give a 10 ms hop: " + path.name)
                chunks = []
                for block in audio.blocks(blocksize=hop * 100, dtype="float64", always_2d=True):
                    n = len(block) // hop
                    if n:
                        chunks.append(np.mean(block[:n * hop].reshape(n, hop, -1) ** 2, axis=(1, 2)))
                curves[stem] = np.concatenate(chunks) if chunks else np.empty(0)
        except (RuntimeError, OSError) as exc:
            # Decoder errors can contain private absolute paths.
            raise ValueError("cannot decode " + path.name + " (" + type(exc).__name__ + ")") from exc
    if len(curves["vocals"]) < round((MIN_STOP_S + REFERENCE_S) / HOP_S):
        raise ValueError("stems too short for a recent reference and a stop")
    if any(not np.all(np.isfinite(c)) for c in curves.values()):
        raise ValueError("non-finite stem samples")
    return curves, layout[1] / layout[0]


def half_beats(length_s, tempo):
    """Round ties upward to the nearest half beat."""
    return math.floor(length_s * tempo / 60 * 2 + 0.5) / 2


def _voice(curves, refs, own, start, end, band_ref):
    vref = refs[3]
    floor = max(own[3] * 10 ** (-ABSENT_DB / 10), EPS)
    bleed = band_ref * 10 ** (-BLEED_DB / 10)
    voice = curves[3, start:end]
    during = float(np.median(voice))
    change = float(_db(during) - _db(vref))
    # All four Joplin recordings: the voice output is separator residue, not
    # unmeasured singing. This describes the stem, not an instrument identity.
    if (own[3] <= max(own[:3]) * 10 ** (-INACTIVE_VOICE_DB / 10)
            and max(vref, float(np.quantile(voice, LOW_FRACTION))) <= band_ref * 10 ** (-INACTIVE_VOICE_DB / 10)):
        return "band break", change, None
    # Muddy 0:09.03 onward: a locally quiet voice enters AFTER the band falls.
    # Require sustained activity against its own level and the pre-cut band,
    # and a rise independent of that band; never call this a pitch measurement.
    if vref <= band_ref * 10 ** (-VOICE_RISE_MAX_DB / 10) and during > max(floor, bleed) and change >= VOICE_RISE_MAX_DB:
        answering = ((voice > max(floor, bleed))
                     & (voice >= vref * 10 ** (VOICE_RISE_MAX_DB / 10)))
        edges = np.diff(np.r_[False, answering, False].astype(int))
        if any(b - a >= round(MIN_STOP_S / HOP_S) for a, b in
               zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))):
            return ENTRY_KIND, change, None
    # A consistently weak stem is ambiguous, even if it looks like band bleed.
    # Only local inactivity relative to the voice's own level supports a break.
    absent = vref <= floor and vref <= bleed
    if absent:
        if float(np.quantile(voice, LOW_FRACTION)) <= min(floor, bleed):
            return "band break", change, None
        return "uncertain", change, "no voice in the second before the band began to fall, so whether the voice holds is unmeasured"
    if vref < own[3] * 10 ** (-VOICE_PRESENT_DB / 10) or vref <= bleed:
        return "uncertain", change, "pre-cut voice too weak to distinguish singing from band bleed"
    delta = _db(voice) - _db(vref)
    if np.mean(delta >= -VOICE_HOLD_DB) >= LOW_FRACTION:
        if change >= VOICE_RISE_MAX_DB:
            return "uncertain", change, (f"vocal stem rises {change:.1f} dB as the band falls: "
                                         "band energy may have moved into the vocal stem")
        return KINDS[0], change, None
    if np.mean(delta <= -DROP_DB) >= LOW_FRACTION:
        return "full silence", change, None
    return "uncertain", change, "voice neither holds (no more than 6 dB down) nor falls 12 dB for 80% of the cut"


def _recurrence(events, tempo=None, beats_per_bar=None):
    """Local runs, including repeated positions separated by intervening cuts.

    Four measured cuts, never extrapolated positions. Prefer the smallest
    spacing that explains a run; meter is optional, not inferred.
    """
    rows = []
    eligible = [e for e in events if e["kind"] in (*KINDS, ENTRY_KIND)
                or e.get("recurrence_eligible_kind") or e.get("recurrence_context_kind")]
    used = set()
    for a in range(len(eligible)):
        if a in used:
            continue
        best = []
        best_spacing = float("inf")
        for b in range(a + 1, len(eligible)):
            if b in used:
                continue
            period = eligible[b]["start_s"] - eligible[a]["start_s"]
            if period > MAX_STOP_S:
                break
            run = [a, b]
            valid_run = run.copy()
            # Rifkin 2:13–2:33: measured cut gaps 2.51–3.26 s; piano
            # performances breathe more than Muddy's 3.29–3.41 s verse grid.
            instrumental = all(eligible[j].get("instrumental") for j in run)
            # Muddy 1977 Winter's first verse: gaps 2.98–3.44 s, median
            # 3.20 s, maximum deviation 7.5%. Clapton 1994 needs 3.36%.
            tolerance = .20 if instrumental else .075
            for j in range(b + 1, len(eligible)):
                gap = eligible[j]["start_s"] - eligible[run[-1]]["start_s"]
                if gap > period + max(.06, period * tolerance) * 2:
                    break
                trial = np.diff([eligible[q]["start_s"] for q in run] + [eligible[j]["start_s"]])
                # A prefix's median can fail although the complete run fits
                # (Muddy 1977). Keep it while some spacing fits both extremes;
                # only a run that passes the actual median check may be used.
                spacing = float(np.median(trial) if instrumental else (np.min(trial) + np.max(trial)) / 2)
                if (j not in used and (not instrumental or eligible[j].get("instrumental"))
                        and np.max(np.abs(trial - spacing)) <= max(.06, spacing * tolerance) + 1e-8):
                    run.append(j)
                    period = spacing
                    median = float(np.median(trial))
                    if np.max(np.abs(trial - median)) <= max(.06, median * tolerance) + 1e-8:
                        valid_run = run.copy()
            run = valid_run
            gaps = np.diff([eligible[j]["start_s"] for j in run])
            spacing = float(np.median(gaps))
            if (len(run) >= 4 and (spacing < best_spacing
                                  or (spacing == best_spacing and len(run) > len(best)))):
                best = run
                best_spacing = spacing
        if not best:
            continue
        selected = [eligible[j] for j in best]
        spacing = float(np.median(np.diff([e["start_s"] for e in selected])))
        if spacing > MAX_STOP_S:
            continue
        row = dict(interval_s=round(spacing, 2), intervals=len(best) - 1,
                   count=len(best), start_s=selected[0]["start_s"],
                   end_s=round(selected[-1]["start_s"] + selected[-1]["length_s"], 2),
                   event_starts_s=[e["start_s"] for e in selected])
        short = sum(bool(e.get("recurrence_context_kind")) for e in selected)
        if short:
            row["short_gaps"] = short
        kinds = {e.get("recurrence_eligible_kind", e.get("recurrence_context_kind", e["kind"])) for e in selected}
        row["kind"] = next(iter(kinds)) if len(kinds) == 1 else "mixed cuts"
        if tempo:
            beats = half_beats(spacing, tempo)
            if abs(spacing * tempo / 60 - beats) <= .15:
                row["beats_approx"] = beats
                if beats_per_bar and beats / beats_per_bar >= 1 and abs(beats / beats_per_bar - round(beats / beats_per_bar)) < .03:
                    row["bars_approx"] = round(beats / beats_per_bar)
        rows.append(row)
        used.update(best)
    return rows


def _voice_break(c, refs, own, start, end):
    """A sustained relative vocal low followed by a measured recovery.

    Check the reported cue, including the staggered approach. A settled-only
    hold fraction can erase a break when the singer returns before the band.
    This describes energy, not whether a pause was intentional or a breath.
    """
    ref = refs[3]
    if ref < own[3] * 10 ** (-VOICE_PRESENT_DB / 10) or ref <= refs[:3].sum() * 10 ** (-BLEED_DB / 10):
        return None
    low = c[3, start:end] <= ref * 10 ** (-DROP_DB / 10)
    edges = np.diff(np.r_[False, low, False].astype(int))
    hold = round(RETURN_HOLD_S / HOP_S)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a < round(MIN_STOP_S / HOP_S):
            continue
        # Recovery above the low threshold must persist, not just spike once.
        for j in range(b, len(low) - hold + 1):
            if not np.any(low[j:j + hold]):
                return dict(start_s=round((start + int(a)) * HOP_S, 2),
                            length_s=round((int(b) - int(a)) * HOP_S, 2),
                            return_s=round((start + j) * HOP_S, 2), drop_db=DROP_DB)
    return None


def _fall(band, ref, edge_from, start, end):
    """Time the descent to the low tenth percentile, at most 36 dB down.

    A short 3-to-12 dB crossing can be the first part of a long fade. Fit the
    whole descent, excluding the crossing frame (a final jump to digital zero
    must not conceal a preceding gain ramp). A long straight dB ramp is still
    ambiguous: an exponential ringing note has that shape too.
    """
    floor_drop = float(_db(ref) - _db(np.quantile(band[start:end], .10)))
    target = min(FALL_FLOOR_DB, max(CANDIDATE_DB, floor_drop))
    low = np.flatnonzero(band[edge_from:end] <= ref * 10 ** (-target / 10))
    to = edge_from + int(low[0]) if len(low) else end - 1
    fall = (to - edge_from) * HOP_S
    y = _db(band[edge_from:to])
    r2 = 0.0
    if len(y) >= 3 and np.var(y) > 0:
        x = np.arange(len(y)) * HOP_S
        slope, intercept = np.polyfit(x, y, 1)
        r2 = float(1 - np.var(y - (slope * x + intercept)) / np.var(y)) if slope < 0 else 0.0
    return fall, target, r2


def _first_exit(c, recent, own, start, settled, guard):
    """First corroborated stem exit in a staggered cut, at most 1.5 s earlier.

    Preserve the summed-band boundary separately. A stem must be locally active
    before leaving, then stay 6 dB down for 90% of the approach to the low band.
    This guards against timing an ordinary inter-hit gap as an exit.
    """
    for j in range(max(guard, start - round(STAGGER_S / HOP_S)), start):
        for s in range(3):
            ref = recent[s, j - guard]
            active = (own[s] > max(own[:3]) * 10 ** (-ABSENT_DB / 10)
                      and ref > own[s] * 10 ** (-ABSENT_DB / 10))
            threshold = ref * 10 ** (-RETURN_DB / 10)
            if active and c[s, j - 1] > threshold >= c[s, j] and np.mean(
                    c[s, j:settled + 1] <= ref * 10 ** (-RETURN_DB / 10)) >= .90:
                return j, STEMS[s]
    return start, None


def _voice_after(c, ref, start, end):
    """Describe a measured vocal fall near the return, without extending the cut."""
    a = max(start, end - round(1 / HOP_S))
    b = min(c.shape[1], end + round(4 / HOP_S))
    voice = c[3]
    hold = round(.30 / HOP_S)
    low = voice <= ref * 10 ** (-24 / 10)
    for j in range(end, b - hold):
        if np.all(low[j:j + hold]):
            high = np.flatnonzero(voice[a:j] >= ref * 10 ** (-VOICE_HOLD_DB / 10))
            if len(high):
                first = a + int(high[-1]) + 1
                if .30 <= (j - first) * HOP_S <= 3:
                    return dict(start_s=round(first * HOP_S, 2), end_s=round(j * HOP_S, 2),
                                from_drop_db=VOICE_HOLD_DB, to_drop_db=24)
            break
    return None


def _faded_stops(c, own, events):
    """Separate fade-and-return evidence; never promote a sudden candidate.

    Like harmony.find_dips, use a 250 ms dB envelope and 12 s rolling median.
    Additionally require a slow fitted descent, two active band stems with
    falls beyond possible vocal bleed, and a sustained, musically active return.
    A single ringing stem, vocal bleed, or a terminal fade is insufficient.
    These are energy fades, not inferred edits or intentional fake endings.
    """
    from scipy.ndimage import median_filter, uniform_filter1d
    smooth = round(FADED_SMOOTH_S / HOP_S)
    window = round(FADED_REFERENCE_S / HOP_S)
    db = uniform_filter1d(_db(c.sum(axis=0)), smooth)
    ref = median_filter(db, size=window + 1, mode="nearest")
    low = db < ref - DROP_DB
    edges = np.diff(np.r_[False, low, False].astype(int))
    hold = round(RETURN_HOLD_S / HOP_S)
    present = np.any((c[:3] > own[:3, None] * 10 ** (-ABSENT_DB / 10))
                     & (own[:3, None] > max(own[:3]) * 10 ** (-ABSENT_DB / 10)), axis=0)
    found = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a, b = int(a), int(b)
        length = (b - a) * HOP_S
        if not MIN_STOP_S <= length <= MAX_STOP_S:
            continue
        baseline = float(np.mean(ref[a:b]))
        trough = a + int(np.argmin(db[a:b]))
        pre_from = max(0, a - window)
        pre = np.flatnonzero(db[pre_from:a] >= baseline - EDGE_START_DB)
        if not len(pre):
            continue
        start = pre_from + int(pre[-1])
        if start < round(REFERENCE_S / HOP_S):
            continue
        fall = (trough - start) * HOP_S
        if fall <= MAX_FALL_S:
            continue
        y = db[start:trough + 1]
        x = np.arange(len(y)) * HOP_S
        slope, intercept = np.polyfit(x, y, 1)
        r2 = float(1 - np.var(y - (slope * x + intercept)) / np.var(y)) if np.var(y) else 0
        if slope >= 0 or r2 < FADED_R2:
            continue
        # Freeze the wide-context level for the recovery; do not let a fade
        # lower its own return threshold into separator residue.
        end = b
        limit = min(c.shape[1] - hold + 1, a + round(MAX_STOP_S / HOP_S) + 1)
        while end < limit and not np.all((db[end:end + hold] >= baseline - RETURN_DB)
                                        & present[end:end + hold]):
            end += 1
        if end >= limit:
            continue
        if any(e["start_s"] < end * HOP_S and e["start_s"] + e["length_s"] > start * HOP_S
               for e in events + found):
            continue
        before = np.median(c[:, max(0, start - round(REFERENCE_S / HOP_S)):start], axis=1)
        during = np.median(c[:, a:b], axis=1)
        active = ((own[:3] > max(own[:3]) * 10 ** (-ABSENT_DB / 10))
                  & (before[:3] > own[:3] * 10 ** (-ABSENT_DB / 10)))
        drops = _db(before[:3]) - _db(during[:3])
        independent = drops + _db(during[3]) - _db(before[3])
        beyond_bleed = ((before[:3] > before[3] * 10 ** (-BLEED_DB / 10))
                        | (independent >= STEM_DROP_DB))
        if np.sum(active & (drops >= DROP_DB) & beyond_bleed) < 2:
            continue
        found.append(dict(kind=FADED_KIND, start_s=round(a * HOP_S, 2),
                          length_s=round((end - a) * HOP_S, 2),
                          low_length_s=round(length, 2), return_s=round(end * HOP_S, 2),
                          drop_db=round(baseline - float(db[trough]), 1),
                          fade_start_s=round(start * HOP_S, 2), fall_s=round(fall, 2),
                          fade_ramp_r2=round(r2, 3), reference_db=round(baseline, 1),
                          stem_drop_db={s: round(float(d), 1) for s, d in zip(STEMS[:3], drops)},
                          reference_s=FADED_REFERENCE_S, smooth_s=FADED_SMOOTH_S,
                          min_ramp_r2=FADED_R2))
    return found


def detect(curves, tempo=None, beats_per_bar=None):
    """Analyze aligned 10 ms energy curves (also useful for diagnostics).

    Each stem's reference is its median energy in the preceding second, ending
    100 ms before a 3 dB departure. Freeze all four references through the cut,
    including the approach to the 8 dB candidate threshold: a sustained tail
    must not continually lower its own reference. Long or steady decays remain
    uncertain unless a struck decay recurs regularly. Short measured cuts are
    masked from the reference. The first corroborated stem exit can precede
    the summed cut;
    depth uses the settled summed cut. Also check the whole reported interval
    for a continuing stem or a vocal interruption during a staggered approach.
    Without voice before or during the cut, require a 0.5 s summed-band low
    rather than the 0.3 s minimum used for cuts involving an active voice.
    Sum the three *energies*, never waveforms or dB. Each locally active stem
    must corroborate the summed drop; a dominating drum cannot hide sustained keys.
    No guessed bleed subtraction: residuals above threshold remain uncertain.
    Additional, non-overlapping slow dips use the separate faded-stop rules.
    """
    from scipy.ndimage import median_filter
    c = np.asarray([curves[s] for s in STEMS], dtype=float)
    if c.ndim != 2 or not c.shape[1] or not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("invalid aligned energy curves")
    n = c.shape[1]
    own = np.percentile(c, 90, axis=1)
    # 30 ms median rejects one-frame glitches without smearing an edge into a fade.
    c = median_filter(c, size=(1, 3), mode="nearest")
    k, guard = round(REFERENCE_S / HOP_S), round(REFERENCE_GUARD_S / HOP_S)
    # Odd window, entirely in the past; shift by the guard below.
    recent = median_filter(c, size=(1, k + 1), origin=(0, k // 2), mode="nearest")
    # Blind-spot warning (13 Nov 2026): clamped opening history removes the head blind spot; onset-freeze alone
    # cannot stop dense cuts owning the median. Exclude measured lows, not
    # guessed beat positions. Leave the median unchanged everywhere else.
    for j in range(min(k, n)):
        recent[:, j] = np.median(c[:, :j + 1], axis=1)
    masked = np.zeros(n, dtype=bool)
    band = c[:3].sum(axis=0)
    events = []
    i = round(SCAN_START_S / HOP_S)
    while i < n:
        refs = recent[:, i - guard]
        lo, hi = max(0, i - guard - k), i - guard + 1
        available = ~masked[lo:hi]
        if np.mean(~available) > .5 and np.any(available):
            unmasked = np.median(c[:, lo:hi][:, available], axis=1)
            if refs[:3].sum() <= unmasked[:3].sum() * 10 ** (-CANDIDATE_DB / 10):
                refs = unmasked
        ref = float(refs[:3].sum())
        if ref <= EPS or band[i] > ref * 10 ** (-EDGE_START_DB / 10):
            i += 1
            continue
        active = ((own[:3] > max(own[:3]) * 10 ** (-ABSENT_DB / 10))
                  & (refs[:3] > own[:3] * 10 ** (-ABSENT_DB / 10)))
        # An already absent band falling into padding, then recovering only to
        # separator residue, cannot establish a stop (12 Nov 2026).
        if not np.any(active):
            i += 1
            continue
        # Arm at -3 dB, but do not open an event until -8 dB. Disarm on a
        # sustained recovery, or after the bounded decay search. This also
        # exposes a slow tail that the sliding median would otherwise absorb.
        seed, trigger = i, i
        hold = round(RETURN_HOLD_S / HOP_S)
        decay_end = min(n, seed + round(MAX_DECAY_S / HOP_S))
        while trigger < decay_end:
            if band[trigger] <= ref * 10 ** (-CANDIDATE_DB / 10):
                break
            if trigger > seed and trigger + hold <= n and np.all(
                    band[trigger:trigger + hold] >= ref * 10 ** (-EDGE_START_DB / 10)):
                break
            trigger += 1
        if trigger == decay_end or band[trigger] > ref * 10 ** (-CANDIDATE_DB / 10):
            i += 1
            continue
        i = trigger
        # Return must persist for 100 ms, so a leaking transient is not a return.
        high = band >= ref * 10 ** (-RETURN_DB / 10)
        end = i
        search_end = min(n, i + round((MAX_STOP_S + MAX_DECAY_S) / HOP_S) + hold)
        while end < search_end and not (end + hold <= n and np.all(high[end:end + hold])):
            end += 1
        if end == search_end:
            # No qualifying return: no stop. Reacquire without skipping an
            # entire eight seconds; a later, quieter section can have cuts.
            i += k + guard
            continue
        pre_from = max(0, seed - k)
        pre = np.flatnonzero(band[pre_from:i] >= ref * 10 ** (-EDGE_START_DB / 10))
        if not len(pre):
            i = end + hold
            continue
        edge_from = pre_from + int(pre[-1])
        start = edge_from + 1
        while start < i and band[start] > ref * 10 ** (-RETURN_DB / 10):
            start += 1
        length = (end - start) * HOP_S
        low = np.flatnonzero(band[i:end] <= ref * 10 ** (-DROP_DB / 10))
        edge_to = i + int(low[0]) if len(low) else i
        fall, fall_target, ramp_r2 = _fall(band, ref, edge_from, start, end)
        if not (MIN_STOP_S - 1e-8 <= length <= MAX_STOP_S + 1e-8):
            i = end + hold
            continue
        # Judge voice/depth after -12 dB, separately from timing the full fall.
        settled = min(end - 1, max(edge_to, start + round(TAIL_GRACE_S / HOP_S)))
        plateau = c[:, settled:end]
        drops = _db(refs[:3]) - _db(np.median(plateau[:3], axis=1))
        drop = float(_db(ref) - _db(np.median(band[settled:end])))
        band_low_fraction = float(np.mean(band[settled:end] <= ref * 10 ** (-DROP_DB / 10)))
        # Fixed-gain vocal leakage falls with the voice. An active stem falling
        # independently by 10 dB for 80% of the cut is evidence beyond loudness.
        relative_drop = (_db(refs[:3, None]) - _db(plateau[:3])
                         + _db(plateau[3]) - _db(refs[3]))
        independent = np.mean(relative_drop >= STEM_DROP_DB, axis=1)
        cue, exit_stem = _first_exit(c, recent, own, start, settled, guard)
        # Timing a staggered exit must not enlarge the permitted stop duration.
        if (end - cue) * HOP_S > MAX_STOP_S + 1e-8:
            cue, exit_stem = start, None
        cue_refs = recent[:, cue - guard] if cue < start else refs
        context = c[:, min(end - 1, cue + round(TAIL_GRACE_S / HOP_S)):end]
        context_delta = _db(context[:3]) - _db(cue_refs[:3, None])
        cue_active = ((own[:3] > max(own[:3]) * 10 ** (-ABSENT_DB / 10))
                      & (cue_refs[:3] > own[:3] * 10 ** (-ABSENT_DB / 10)))
        held_fraction = np.mean(context_delta >= -RETURN_DB, axis=1)
        cut_fraction = np.mean(context_delta <= -STEM_DROP_DB, axis=1)
        holding = [STEMS[s] for s in range(3) if cue_active[s] and held_fraction[s] >= LOW_FRACTION]
        cutting = [STEMS[s] for s in range(3) if cue_active[s] and cut_fraction[s] >= LOW_FRACTION]
        voice_break = _voice_break(c, cue_refs, own, cue, end)
        reasons = []
        if fall > MAX_FALL_S + 1e-8:
            reasons.append(f"slow edge / sustained tail: fall to floor takes {fall:.2f} s, exceeds {MAX_FALL_S:g} s; fade not ruled out")
        if fall >= FADE_RAMP_S and ramp_r2 >= FADE_RAMP_R2:
            reasons.append(f"steady fade / sustained tail: {fall:.2f} s dB ramp (R² {ramp_r2:.3f}); fade not ruled out")
        tail_reasons = len(reasons)
        if ref <= refs[3] * 10 ** (-BLEED_DB / 10) and not np.any(active & (independent >= LOW_FRACTION)):
            reasons.append("weak accompaniment without an independent stem fall vs voice; could be vocal bleed alone")
        if band_low_fraction < LOW_FRACTION:
            reasons.append("accompaniment not 12 dB down for 80% of settled cut (tail, bleed or shallow dip)")
        for s in range(3):
            if active[s] and drops[s] < STEM_DROP_DB:
                reasons.append(f"{STEMS[s]} fell only {drops[s]:.1f} dB; sustained layer or bleed")
        if holding and cutting:
            reasons.append("partial cut: " + ", ".join(cutting) + " fall at least 10 dB while "
                           + ", ".join(holding) + " continue no more than 6 dB down for 80% of reported interval")
        voice_kind, voice_change, reason = _voice(c, refs, own, settled, end, ref)
        if voice_break:
            voice_kind, reason = "breaks then returns", "voice breaks then returns during reported interval"
        # 10 Nov 2026: Fleetwood's unvoiced intro strums made six false stops
        # (0.34–0.45 s). Require half a second without voice; a staggered stem
        # exit must not lengthen a short summed-band dip into a band break.
        short_break = voice_kind == "band break" and (end - start) * HOP_S < MIN_BAND_BREAK_S - 1e-8
        instrumental = (voice_kind == "band break"
                        and own[3] <= max(own[:3]) * 10 ** (-INACTIVE_VOICE_DB / 10))
        if short_break and not instrumental:
            i = end + hold
            continue
        if short_break:
            reasons.append("unvoiced gap shorter than 0.5 s; not an individual band break")
        if reason:
            reasons.append(reason)
        kind = "uncertain" if reasons else voice_kind
        voice_delta = _db(c[3, settled:end]) - _db(refs[3])
        length = (end - cue) * HOP_S
        event = dict(start_s=round(cue * HOP_S, 2), length_s=round(length, 2),
                     band_start_s=round(start * HOP_S, 2), first_exit_stem=exit_stem,
                     drop_db=round(drop, 1), kind=kind, fall_s=round(fall, 2),
                     fall_to_db=round(fall_target, 1), fade_ramp_r2=round(ramp_r2, 3),
                     fall_db_per_s=round((fall_target - EDGE_START_DB) / max(fall, HOP_S), 1),
                     drop_floor_limited=bool(np.median(band[settled:end]) <= EPS),
                     voice_change_db=round(voice_change, 1),
                     voice_behavior=voice_kind,
                     voice_hold_fraction=round(float(np.mean(voice_delta >= -VOICE_HOLD_DB)), 3),
                     band_low_fraction=round(band_low_fraction, 3),
                     stem_independent_fraction={s: round(float(f), 3) for s, f in zip(STEMS[:3], independent)},
                     stem_drop_db={s: round(float(d), 1) for s, d in zip(STEMS[:3], drops)},
                     reference_db={s: round(float(_db(r)), 1) for s, r in zip(STEMS, refs)})
        if instrumental:
            event["instrumental"] = True
        if holding and cutting:
            event["arrangement"] = "partial cut"
            event["continuing_stems"] = holding
            event["cutting_stems"] = cutting
            event["cue_reference_db"] = {s: round(float(_db(r)), 1) for s, r in zip(STEMS, cue_refs)}
            event["cue_stem_hold_fraction"] = {s: round(float(f), 3) for s, f in zip(STEMS[:3], held_fraction)}
            event["cue_stem_cut_fraction"] = {s: round(float(f), 3) for s, f in zip(STEMS[:3], cut_fraction)}
        if voice_break:
            event["voice_break"] = voice_break
        if reasons:
            event["reason"] = "; ".join(reasons)
        if kind == KINDS[0]:
            after = _voice_after(c, refs[3], start, end)
            if after:
                event["voice_after"] = after
        # Rifkin 2:13.79 / 2:22.34: a ringing single layer follows a measured
        # attack, then returns sharply. A straight dB tail alone cannot tell
        # decay from fading. Only a regular run may resolve this ambiguity;
        # depth, duration, voice and continuing-layer checks still have vetoes.
        single_layer = np.sum(own[:3] > max(own[:3]) * 10 ** (-ABSENT_DB / 10)) == 1
        history = _db(band[max(0, edge_from - k):edge_from + 1])
        attack_hop = round(RETURN_HOLD_S / HOP_S)
        struck = len(history) > attack_hop and np.any(history[attack_hop:] - history[:-attack_hop] >= RETURN_DB)
        if (tail_reasons and len(reasons) == tail_reasons and single_layer and struck
                and voice_kind == "band break" and fall <= MAX_FALL_S):
            event["recurrence_eligible_kind"] = voice_kind
        if (short_break and len(reasons) == 1 and event.get("instrumental")):
            event["recurrence_context_kind"] = voice_kind
        # Only a cut bounded within the reference window has an earlier high
        # to retain. Masking a longer break's recovery can create new cues in
        # the returning band (Muddy 0:14–0:26); reacquire normally after those.
        # Father Figure 3:25 / 4:59 (13 Nov 2026): an ambiguous candidate
        # must not raise the next reference and merge or erase a measured cut.
        # Instrumental short gaps / struck tails retain their separate vetoes.
        qualified = not reasons or event.get("recurrence_eligible_kind") or event.get("recurrence_context_kind")
        if qualified and band_low_fraction >= LOW_FRACTION and (end - start) * HOP_S <= REFERENCE_S:
            masked[start:end] = True
        if tempo:
            event["beats_approx"] = half_beats(length, tempo)
        events.append(event)
        i = end + hold
    for run in _recurrence(events, tempo, beats_per_bar):
        for event in events:
            if event["start_s"] in run["event_starts_s"] and event.get("recurrence_eligible_kind"):
                event["kind"] = event["recurrence_eligible_kind"]
                event["decay_evidence"] = event.pop("reason")
                event["confirmed_by_recurrence"] = True
    faded = _faded_stops(c, own, events)
    return sorted(events + faded, key=lambda e: e["start_s"]) if faded else events


def analyse(stem_dir, tempo=None, beats_per_bar=None):
    """Failures are data as well as visible card lines; never 'none found'."""
    settings = dict(min_s=MIN_STOP_S, min_band_break_s=MIN_BAND_BREAK_S,
                    max_s=MAX_STOP_S, hop_s=HOP_S,
                    reference_s=REFERENCE_S, reference_guard_s=REFERENCE_GUARD_S,
                    drop_db=DROP_DB, candidate_db=CANDIDATE_DB, max_fall_s=MAX_FALL_S,
                    max_decay_s=MAX_DECAY_S,
                    fall_floor_db=FALL_FLOOR_DB, fade_ramp_s=FADE_RAMP_S, fade_ramp_r2=FADE_RAMP_R2,
                    card_uncertain_db=CARD_UNCERTAIN_DB,
                    stagger_s=STAGGER_S,
                    edge_start_db=EDGE_START_DB, return_db=RETURN_DB, return_hold_s=RETURN_HOLD_S,
                    tail_grace_s=TAIL_GRACE_S, low_fraction=LOW_FRACTION,
                    stem_drop_db=STEM_DROP_DB, voice_hold_db=VOICE_HOLD_DB,
                    bleed_db=BLEED_DB, absent_db=ABSENT_DB, voice_present_db=VOICE_PRESENT_DB)
    settings.update(inactive_voice_db=INACTIVE_VOICE_DB, scan_start_s=SCAN_START_S,
                    reference_mask="measured lows bounded within reference window")
    try:
        if tempo is not None and (not math.isfinite(tempo) or tempo <= 0):
            raise ValueError("tempo must be finite and positive")
        if beats_per_bar is not None and (not math.isfinite(beats_per_bar) or beats_per_bar <= 0):
            raise ValueError("beats per bar must be finite and positive")
        curves, duration = _curves(stem_dir)
        events = detect(curves, tempo, beats_per_bar)
        recurrence = _recurrence(events, tempo, beats_per_bar)
        return dict(status="ran", version=VERSION, settings=settings, duration_s=round(duration, 2),
                    tempo_bpm=tempo, beats_per_bar=beats_per_bar, events=events, recurrence=recurrence,
                    curves=_pack_curves(curves))
    except Exception as exc:
        # Our validation errors name only stems, not private directories.
        if isinstance(exc, ImportError):
            reason = "missing required module: " + str(exc.name)
        else:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return dict(status="failed", reason=reason, settings=settings, events=[])


def _pack_curves(curves):
    """Lossless float64, stem-major in STEMS order; portable inside JSON.

    Float32 moves Rifkin's measured fall at 2:30.20 (13 Nov 2026). Keep
    every input bit so future rules can also re-derive threshold decisions.
    """
    c = np.asarray([curves[s] for s in STEMS], dtype="<f8")
    return dict(encoding="f64le-zlib-base64", stems=list(STEMS), hop_s=HOP_S,
                frames=c.shape[1], data=base64.b64encode(zlib.compress(c.tobytes())).decode("ascii"))


def _unpack_curves(saved):
    # Accept the full-precision lists saved by the first curve-bearing version.
    if "encoding" not in saved:
        return saved
    if (saved["encoding"] != "f64le-zlib-base64" or saved["stems"] != list(STEMS)
            or saved["hop_s"] != HOP_S or type(saved["frames"]) is not int or saved["frames"] <= 0):
        raise ValueError("invalid saved curve encoding")
    raw = zlib.decompress(base64.b64decode(saved["data"], validate=True))
    c = np.frombuffer(raw, dtype="<f8").reshape(len(STEMS), saved["frames"])
    return dict(zip(STEMS, c))


def refresh(result):
    """Derive current STOPS from saved energy, without changing the bundle."""
    if result.get("status") != "ran":
        return result
    if "curves" not in result:
        return dict(result, update_notice="re-listen to update STOPS (saved bundle has no energy curves)")
    try:
        events = detect(_unpack_curves(result["curves"]), result.get("tempo_bpm"), result.get("beats_per_bar"))
        return dict(result, version=VERSION, events=events,
                    recurrence=_recurrence(events, result.get("tempo_bpm"), result.get("beats_per_bar")))
    except (ValueError, KeyError, TypeError, zlib.error):
        return dict(status="failed", events=[], reason="invalid saved STOPS curves; re-listen to update STOPS")


def saved_text(text, result, *, detail=False):
    return re.sub(r"(?m)^STOPS:[^\n]*(?:\n[ \t]+[^\n]*)*",
                  lambda _: "\n".join(card_lines(refresh(result), detail=detail)), text)


def _time(seconds):
    ticks = round(seconds * 100)
    return f"{ticks // 6000}:{(ticks % 6000) / 100:05.2f}"


def card_lines(result, *, detail=False):
    """Detail preserves legacy diagnostics; shallower candidates stay JSON-only."""
    from stage_status import NOTES_WIDTH
    if result["status"] == "failed":
        lines = ["STOPS: failed: " + result["reason"]]
    else:
        events = [e for e in result["events"] if e["kind"] != "uncertain"
                  or (detail and e["drop_db"] >= CARD_UNCERTAIN_DB)]
        known = sum(e["kind"] != "uncertain" for e in events)
        candidates = [e for e in result["events"] if e["kind"] == "uncertain"]
        if not detail and not known and candidates:
            lines = [f"STOPS: 0 confident · uncertain: {len(candidates)} candidates not called (see --view detail / JSON)"]
            strongest = sorted(candidates, key=lambda e: e["drop_db"], reverse=True)[:2]
            lines.append("  strongest band falls: " + "; ".join(
                f"{_time(e['start_s'])} ({e['drop_db']:g} dB, uncertain)" for e in strongest))
        elif not events:
            lines = [f"STOPS: none found ({MIN_STOP_S:g} to {MAX_STOP_S:g} s; accompaniment vs its own level)"]
        else:
            counts = [f"{k}: {sum(e['kind'] == k for e in events)}" for k in KINDS]
            if any(e["kind"] == FADED_KIND for e in events):
                counts.append(f"{FADED_KIND}: {sum(e['kind'] == FADED_KIND for e in events)}")
            if any(e["kind"] == ENTRY_KIND for e in events):
                counts.append(f"{ENTRY_KIND}: {sum(e['kind'] == ENTRY_KIND for e in events)}")
            lines = [f"STOPS: {known} · " + " · ".join(counts)]
            uncertain = len(events) - known
            if uncertain:
                lines.append(f"  uncertain candidates: {uncertain} (band fall ≥{CARD_UNCERTAIN_DB:g} dB; full list in JSON)")
        hidden = len(result["events"]) - len(events) if detail else 0
        if hidden:
            lines.append(f"  {hidden} shallower uncertain candidates in JSON only (band fall <{CARD_UNCERTAIN_DB:g} dB)")
        for e in events:
            if e["kind"] == FADED_KIND:
                lines.append(f"  {_time(e['start_s'])} ({e['length_s']:g} s) total stem energy falls {e['drop_db']:g} dB, {FADED_KIND}")
                lines.append(f"    low for {e['low_length_s']:g} s; returns {_time(e['return_s'])}; "
                             f"fade from {_time(e['fade_start_s'])}, {e['fall_s']:g} s descent (R² {e['fade_ramp_r2']:.3f})")
                continue
            beats = f" ≈{e['beats_approx']:g} beats" if "beats_approx" in e else ""
            kind = "voice holds" if e["kind"] == KINDS[0] else e["kind"]
            if kind == ENTRY_KIND:
                kind = "vocal stem enters"
            limit = "≥" if e.get("drop_floor_limited") else ""
            band = "summed band falls" if e["kind"] == "uncertain" else "band cuts"
            lines.append(f"  {_time(e['start_s'])} ({e['length_s']:g} s{beats}) {band} {limit}{e['drop_db']:g} dB, {kind}")
            if "voice_break" in e:
                v = e["voice_break"]
                lines.append(f"    voice breaks: ≥{v['drop_db']:g} dB down for {v['length_s']:g} s at {_time(v['start_s'])}; returns {_time(v['return_s'])}")
            target = e["fall_to_db"]
            lines.append(f"    fall {EDGE_START_DB:g}→{target:g} dB in {e['fall_s']:.2f} s ({e['fall_db_per_s']:g} dB/s)")
            if e.get("first_exit_stem"):
                lines.append(f"    first exit: {e['first_exit_stem']}; summed band crosses −{RETURN_DB:g} dB at {_time(e['band_start_s'])}")
            if e.get("voice_after"):
                v = e["voice_after"]
                lines.append(f"    voice then fades {v['from_drop_db']:g}→{v['to_drop_db']:g} dB below pre-cut: {_time(v['start_s'])}→{_time(v['end_s'])}")
            if e.get("reason"):
                lines.append("    uncertain: " + e["reason"])
            if e.get("confirmed_by_recurrence"):
                lines.append("    ringing decay after a measured attack; regular cuts corroborate the stop")
        if not detail and known and candidates:
            lines.append(f"  uncertain: {len(candidates)} candidates not called (see --view detail / JSON)")
        for row in result.get("recurrence", []):
            if "count" in row:
                # Refused's scored ad-lib intro also has regular energy cuts
                # (13 Nov 2026). Spacing alone does not establish a pulse.
                lines.append(f"  repeated cuts: {row['count']} cuts every ~{row['interval_s']:g} s, "
                             f"{_time(row['start_s'])}–{_time(row['end_s'])}")
                lines.append("    pulse not checked; cut spacing alone does not establish stop-time")
                if row.get("short_gaps"):
                    lines.append(f"    includes {row['short_gaps']} short gaps, not called as individual band breaks")
            if "beats_approx" in row:
                count, unit = (row["bars_approx"], "bars") if "bars_approx" in row else (row["beats_approx"], "beats")
                lines.append(f"  {row['kind']}: every {count:g} {unit} ≈ ({row['intervals']} measured intervals)")
        if any(e["kind"] == ENTRY_KIND for e in events):
            lines.append("  vocal entry measured from stem energy; singing/pitch not checked")
        if result.get("version") == VERSION:
            lines.append(f"  opening 0:00–{_time(SCAN_START_S)} not checked (short reference history)")
        if result.get("update_notice"):
            lines.append("  " + result["update_notice"])
    return [part for line in lines for part in textwrap.wrap(line, width=NOTES_WIDTH, subsequent_indent="    ",
                                                            replace_whitespace=False, drop_whitespace=True)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stem_dir")
    ap.add_argument("--tempo", type=float)
    ap.add_argument("--beats-per-bar", type=float, help="explicit meter for recurrence in bars; never assumes 4/4")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--view", choices=("full", "detail"), default="full",
                    help="detail retains the candidate diagnostics; JSON includes all candidates")
    a = ap.parse_args(argv)
    from network_policy import local_audio_path
    local_audio_path(a.stem_dir, ap)
    result = analyse(a.stem_dir, a.tempo, a.beats_per_bar)
    print(json.dumps(result, indent=2, ensure_ascii=False) if a.json else
          "\n".join(card_lines(result, detail=a.view == "detail")))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
