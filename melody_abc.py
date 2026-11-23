#!/usr/bin/env python3
"""Conservative F0 -> ABC 2.1 fragment; no audio/model imports on the plain-value path.

transcribe(f0, fps, words, ...) returns (abc_fragment, JSON_safe_measurements).
Frame i occupies [i/fps, (i+1)/fps); word ownership uses the frame midpoint.
line_breaks are interior times, in seconds, between words, not inferred phrases.
Beats default to written quarter notes (a convention, not an inferred metre).
Pass the output to a renderer as an *ABC fragment*, not a complete ABC file.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

from keyfuse import MAJOR_KEYS, MINOR_KEYS, MIXOLYDIAN_KEYS, NATURALS, letter_signature as _signature, spell_pitch


def _runs(values):
    """Half-open runs of equal values, including tuple-valued labels."""
    if not len(values):
        return
    start, previous = 0, values[0]
    for i in range(1, len(values)):
        if values[i] != previous:
            yield start, i, previous
            start, previous = i, values[i]
    yield start, len(values), previous


def _positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _times(name, values):
    out = [float(x) for x in values]
    if any(not math.isfinite(x) or x < 0 for x in out):
        raise ValueError(f"{name} must contain finite, nonnegative times")
    if any(b <= a for a, b in zip(out, out[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return out


def _key(value):
    if value is None:
        return None
    if value == "none":
        return value
    match = re.fullmatch(r"([A-Ga-g])([#b]?)\s*(major|minor|mixolydian|maj|min|Mix|m)?",
                         value.strip().replace("♯", "#").replace("♭", "b"))
    if not match:
        raise ValueError("key must be a major, minor or mixolydian key, e.g. 'A minor' or 'Bb'")
    root, accidental, mode = match.groups()
    name = root.upper() + accidental + ("m" if mode in ("minor", "min", "m") else "")
    if mode in ("mixolydian", "Mix"):
        if name not in MIXOLYDIAN_KEYS:
            raise ValueError("key must have a conventional signature (at most seven sharps/flats)")
        name += "Mix"
    elif name not in MAJOR_KEYS and name not in MINOR_KEYS:
        raise ValueError("key must have a conventional signature (at most seven sharps/flats)")
    return name


def _pitch(midi, key=None, accidentals=None, spelling_key=None):
    # Without a supplied key, preserve round 2's explicit measured-pitch spelling.
    names = ("=C", "^C", "=D", "^D", "=E", "=F", "^F", "=G", "^G", "=A", "^A", "=B")
    if key in (None, "none"):
        token = names[midi % 12]
        octave = midi // 12 - 1
    else:
        signature = _signature(key)
        letter, alteration = spell_pitch(midi % 12, spelling_key or key)
        octave = (midi - NATURALS[letter] - alteration) // 12 - 1
        state_key = (letter, octave)
        current = accidentals.get(state_key, signature[letter])
        token = ("=" if alteration == 0 else "^" if alteration == 1 else "_") if current != alteration else ""
        token += letter
        accidentals[state_key] = alteration
    if octave >= 5:
        return token.lower() + "'" * (octave - 5)
    return token + "," * (4 - octave)


def _lyric(text):
    # Fixed-width Unicode escapes keep ABC lyric operators literal. No guessed
    # syllabification: whitespace joins words under one note; hyphens stay text.
    return "".join("~" if c.isspace() else "\\-" if c == "-" else
                   f"\\u{ord(c):04x}" if c in "_*~|%\\&\"$" else c for c in text)


def _length(ticks):
    return "" if ticks == 1 else str(ticks)


def low_note_comments(omissions, line):
    low = [n for n in omissions if n["line"] == line]
    subharmonics = sum(n["reason"] == "likely_tracker_subharmonic" for n in low)
    return [f"% {count} low note{'s' if count != 1 else ''} left out: likely tracker {reason}"
            for count, reason in ((len(low) - subharmonics, "octave error"),
                                  (subharmonics, "subharmonics")) if count]


def _subharmonic_clusters(notes, probabilities, voiced_prob, target, fps):
    """Yield weak half/third-frequency alternatives bracketed by one upper pitch."""
    if voiced_prob is None:
        return
    strong = target[(target >= 0) & (voiced_prob >= .5)]
    if not len(strong):
        return
    centre = np.median(strong)
    peaks = [np.max(voiced_prob[round(n["start"] * fps):round(n["end"] * fps)]) for n in notes]
    for i, first in enumerate(notes):
        pitch = first["midi"]
        if (probabilities[i] is None or probabilities[i] < .4 or
                first["end"] - first["start"] < .12 or abs(pitch - centre) > 4):
            continue
        candidates = []
        previous = first
        for j in range(i + 1, len(notes)):
            note, prob = notes[j], probabilities[j]
            if (note["start"] - first["end"] > 2 or note["start"] - previous["end"] > .75
                    or prob is None):
                break
            interval = pitch - note["midi"]
            if interval == 0 and prob >= .2 and peaks[j] >= .4:
                # An immediate switch from half-frequency back to the upper
                # note corroborates the earlier third-frequency alternative.
                if ({pitch - notes[k]["midi"] for k in candidates} == {12, 19}
                        and pitch - previous["midi"] == 12
                        and note["start"] - previous["end"] <= .12):
                    yield i, j, candidates
                break
            if prob >= .4 or note["line"] != first["line"]:
                break
            if interval >= 10:
                if (interval not in (12, 19) or prob >= (.2 if interval == 19 else .3)
                        or note["end"] - note["start"] > .8):
                    break
                candidates.append(j)
            previous = note


def _low_subharmonics(notes, probabilities, voiced_prob, target, fps, range_low):
    """Find weak, out-of-range lows corroborated by nearby stronger harmonics."""
    if voiced_prob is None or range_low is None:
        return
    reach = round(.75 * fps)
    needed = math.ceil(.08 * fps)
    for i, (note, prob) in enumerate(zip(notes, probabilities)):
        pitch = note["midi"]
        if (pitch > range_low - 3 or prob is None or prob >= .2
                or note["end"] - note["start"] > .55):
            continue
        a, b = round(note["start"] * fps), round(note["end"] * fps)
        peak = np.max(voiced_prob[a:b])
        if peak >= .5:
            continue

        def upper(start, end):
            window = target[start:end][voiced_prob[start:end] >= .4]
            candidates = [pitch + step for step in (12, 19, 24)]
            best = max(candidates, key=lambda p: np.count_nonzero(window == p))
            return best, np.count_nonzero(window == best)

        left, left_count = upper(max(0, a - reach), a)
        right, right_count = upper(b, b + reach)
        before = left if left_count >= needed else None
        after = right if right_count >= needed else None
        if before is not None and after is not None and abs(before - after) <= 7:
            yield i, before, after
        elif ((before is None) != (after is None) and pitch <= range_low - 5 and
              prob < .15 and peak < .35 and max(left_count, right_count) >= math.ceil(.096 * fps)):
            yield i, before, after


def _octave_outliers(notes, target, fps, short_s, range_low):
    """A short excursion needs nearby pitches in one consistent register."""
    for i, note in enumerate(notes):
        pitch = note["midi"]
        start, end = note["start"], note["end"]
        if end - start > short_s + 1e-9:
            continue
        a, b = round(start * fps), round(end * fps)
        # A continuous, confidently tracked vocal leap can be real even when
        # it is brief. Require breaks on both sides of an isolated excursion.
        edge = max(1, round(.18 * fps))
        if not (np.any(target[max(0, a - edge):a] < 0) and np.any(target[b:b + edge] < 0)):
            continue
        # A note split by a lyric boundary can be part of a held pitch.
        if np.count_nonzero(target[max(0, a - 1):b + 1] == pitch) / fps > short_s + 1e-9:
            continue
        left = notes[i - 1] if i and start - notes[i - 1]["end"] <= .75 else None
        right = notes[i + 1] if i + 1 < len(notes) and notes[i + 1]["start"] - end <= .75 else None
        approach = target[max(0, a - round(.25 * fps)):a]
        if any(p >= 0 and abs(p - pitch) <= 2 and (v - u) / fps >= .08
               for u, v, p in _runs(approach)):
            continue
        if left and right:
            neighbours = (left["midi"], right["midi"])
            if abs(neighbours[0] - neighbours[1]) <= 4:
                for shift in (-12, 12):
                    distances = [abs(pitch + shift - p) for p in neighbours]
                    if min(distances) <= 1 and max(distances) <= 4:
                        yield i, neighbours
                        break
        elif (not left and right and i + 2 < len(notes) and range_low is not None
              and pitch <= range_low - 5 and end - start <= .12):
            # At a quiet entrance, look for the missing octave in the raw
            # track and two subsequent notes in that register.
            after = notes[i + 2]
            neighbours = (right["midi"], after["midi"])
            if (after["start"] - right["end"] <= .5
                    and all(abs(pitch + 12 - p) <= 3 for p in neighbours)
                    and any(p == pitch + 12 and (v - u) / fps >= .05
                            for u, v, p in _runs(target[b:b + round(.25 * fps)]))):
                yield i, neighbours


UNALIGNED_LYRICS = "Lyrics could not be aligned to notes: "


def _line_words(text):
    words = []
    prefix = ""
    for token in text.split():
        if any(c.isalnum() for c in token):
            words.append(prefix + token)
            prefix = ""
        elif words:
            words[-1] += token
        else:
            prefix += token
    return words


def _word_key(word):
    return "".join(c for c in word.casefold() if c.isalnum())


def _has_word_coverage(words, notes, word_rows, shown_word):
    """A complete Whisper reading must reach rendered notes in more than one slot."""
    heard = set()
    slots = 0
    for note in notes:
        fresh = set(note["word_indices"]) - heard
        if fresh:
            heard.update(fresh)
            slots += 1
    if not words or slots < min(2, len(words)):
        return False

    written = [_word_key(w) for w in words]
    indices = sorted(heard)
    return (written == [_word_key(word_rows[i]["word"]) for i in indices]
            == [_word_key(shown_word(i)) for i in indices])


def _timed_lyric_labels(words, timed, notes):
    """Use raw word intervals only when they support every printed word."""
    if ([_word_key(w) for w in words] != [_word_key(w["word"]) for w in timed]
            or any(a["end"] > b["start"] for a, b in zip(timed, timed[1:]))):
        return None
    remaining = {w["index"]: _lyric(label) for w, label in zip(timed, words)}
    labels = []
    for note in notes:
        indices = note["word_indices"]
        fresh = [remaining.pop(i) for i in indices if i in remaining]
        # Two independently timed words can share a held pitch. Decline a
        # crowded fit and let the caller try distinct onsets instead.
        if len(fresh) > 2:
            return None
        labels.append("~".join(fresh) if fresh else "_" if indices else "*")
    return labels if not remaining else None


def _words_by_onset(words, span, notes, fps):
    """Spread words over the row's onsets, one per note."""
    start, end = span
    # Use the same half-frame convention as line cuts.
    inside = [i for i, note in enumerate(notes) if start <= note["start"] + 0.5 / fps < end]
    if not words or len(inside) < len(words):
        return None
    labels = ["*"] * len(notes)
    step = (len(inside) - 1) / max(1, len(words) - 1)
    for i, word in enumerate(words):
        labels[inside[round(i * step)]] = _lyric(word)
    return labels


PROFILES = {
    # Strict preserves round 1's pitch rules for comparison.
    "strict": {},
    "middle": {"pitch_rule": "centre", "min_note_s": 0.08,
               "tolerance_cents": 40.0, "max_drift_cents": 45.0,
               "spread_cents": 100.0, "centre_window_s": 0.112,
               "merge_words": True, "unit_seconds": 0.04},
    "loose": {"pitch_rule": "centre", "min_note_s": 0.048,
              "tolerance_cents": 49.0, "max_drift_cents": 70.0,
              "spread_cents": 150.0, "centre_window_s": 0.144,
              "max_excursion_cents": 150.0, "merge_words": True, "unit_seconds": 0.024},
}


def lyric_line_boundaries(lines, words, duration, *, alignment=None, keep_starts=False):
    """Ordered text alignment via the existing CPU-only lyric helper.

    Snap inferred starts out of word interiors for word-safe notation.
    With keep_starts, lyric starts own the frame cuts, even inside a word.
    Keep its approximate/matched flags: text similarity is not an acoustic check.
    Returned cuts may be passed to transcribe; rows disclose placement evidence.
    """
    from lyric_align import align_lyrics

    rows, summary = align_lyrics(lines, words, duration) if alignment is None else alignment
    if not rows:
        return [], [], summary
    occupied = []
    if not keep_starts:
        # Union word interiors so no chosen boundary cuts an ambiguous overlap.
        for _, start, end in words:
            if occupied and start < occupied[-1][1]:
                occupied[-1][1] = max(occupied[-1][1], end)
            else:
                occupied.append([start, end])
    cuts = []
    for row in rows[1:]:
        t = row["t"]
        for start, end in occupied:
            if start < t < end:
                t = start if t - start <= end - t else end
                break
        if not 0 < t < duration or (cuts and t <= cuts[-1]):
            kind = "starts" if keep_starts else "word-safe boundaries"
            raise ValueError(f"lyric lines cannot be placed at distinct {kind}")
        cuts.append(t)
    rows = [dict(row, boundary_s=boundary) for row, boundary in zip(rows, [0.0] + cuts)]
    return cuts, rows, summary


def _grid_fit(notes, beats, subdivisions):
    endpoints = np.array([n[k] for n in notes for k in ("start", "end")
                          if beats[0] <= n[k] <= beats[-1]])
    if not len(endpoints):
        return {"subdivisions_per_beat": subdivisions, "median_ms": None, "p90_ms": None,
                "endpoint_count": 0, "collapsed_notes": 0}
    ticks = np.arange(len(beats)) * subdivisions
    snapped = np.floor(np.interp(endpoints, beats, ticks) + 0.5)
    errors = np.abs(np.interp(snapped, ticks, beats) - endpoints) * 1000
    pairs = [(math.floor(float(np.interp(n["start"], beats, ticks)) + 0.5),
              math.floor(float(np.interp(n["end"], beats, ticks)) + 0.5)) for n in notes
             if beats[0] <= n["start"] <= n["end"] <= beats[-1]]
    return {"subdivisions_per_beat": subdivisions, "median_ms": float(np.median(errors)),
            "p90_ms": float(np.percentile(errors, 90)), "endpoint_count": len(endpoints),
            "collapsed_notes": sum(b <= a for a, b in pairs)}


def decode_abc(abc, rhythm, lines):
    """Decode only our emitted ABC subset, independently of detected note records.

    ABC has relative time. Line anchors and (for proportional output) one linear
    seconds-per-tick calibration per line are necessary side information. No
    per-note source times/pitches are used. This is not a general ABC parser.
    """
    key, music = None, []
    for row in abc.splitlines():
        if row.startswith("K:"):
            key = _key(row[2:])
        elif row and not row.startswith("%") and not re.match(r"^[A-Za-z]:", row):
            music.append(row)
    if len(music) != sum(bool(line["music"]) for line in lines):
        raise ValueError("ABC music lines do not match timing anchors")
    signature = _signature(None if key == "none" else key)
    state, events = {}, []
    token = re.compile(r"([=^_]?)([A-Ga-gz])([,']*)(\d*)(-?)|\|")
    for row, line in zip(music, (line for line in lines if line["music"])):
        cursor = 0
        compact = row.replace(" ", "")
        if "".join(m.group() for m in token.finditer(compact)) != compact:
            raise ValueError("unsupported ABC token in round-trip check")

        def seconds(offset):
            if rhythm["mode"] == "quantised":
                ticks = np.arange(len(rhythm["beats"])) * rhythm["subdivisions_per_beat"]
                return float(np.interp(line["tick_start"] + offset, ticks, rhythm["beats"]))
            return line["written_origin_s"] + offset * line["seconds_per_tick"]

        for match in token.finditer(compact):
            if match.group() == "|":
                state.clear()
                continue
            accidental, letter, marks, length, _ = match.groups()
            length = int(length or 1)
            if letter != "z":
                octave = (5 if letter.islower() else 4) + marks.count("'") - marks.count(",")
                state_key = (letter.upper(), octave)
                if accidental:
                    state[state_key] = {"=": 0, "^": 1, "_": -1}[accidental]
                alteration = state.get(state_key, signature[letter.upper()])
                natural = NATURALS[letter.upper()]
                events.append({"start": seconds(cursor), "end": seconds(cursor + length),
                               "midi": 12 * (octave + 1) + natural + alteration,
                               "line": line["index"]})
            cursor += length
    return events


def self_check(abc, f0, fps, rhythm, lines):
    """Absolute cents error on covered voiced frames; coverage denominator is all
    finite positive F0 in each original lyric slice, including omitted pitches.
    This tests agreement with the tracker, not correctness against the singer.
    """
    f0 = np.asarray(f0, dtype=float)
    midpoint = (np.arange(len(f0)) + 0.5) / fps
    voiced = np.isfinite(f0) & (f0 > 0)
    events = decode_abc(abc, rhythm, lines)
    rows, all_errors, total_voiced, total_covered = [], [], 0, 0
    for line in lines:
        decoded = np.full(len(f0), np.nan)
        for event in events:
            if event["line"] == line["index"]:
                decoded[(midpoint >= event["start"]) & (midpoint < event["end"])] = event["midi"]
        mask = voiced & (midpoint >= line["frame_start"]) & (midpoint < line["frame_end"])
        covered = mask & np.isfinite(decoded)
        errors = np.abs(100 * (69 + 12 * np.log2(f0[covered] / 440.0) - decoded[covered]))
        count, retained = int(np.sum(mask)), int(np.sum(covered))
        rows.append({"line": line["index"], "median_pitch_error_cents": float(np.median(errors)) if retained else None,
                     "voiced_seconds": count / fps, "covered_voiced_seconds": retained / fps,
                     "voiced_coverage_share": retained / count if count else None})
        all_errors.extend(errors.tolist())
        total_voiced += count
        total_covered += retained
    return {"reference": "cached F0, not independently verified melody",
            "pitch_error": "absolute cents against parsed MIDI; no tuning correction or octave folding",
            "coverage": "voiced F0 frames overlapped by notes decoded from this lyric line / all voiced frames in its original slice",
            "timing": "supplied/derived grid" if rhythm["mode"] == "quantised" else
                      "line-local linear calibration from first to last retained endpoint; no per-note warp",
            "lines": rows, "summary": {
                "median_pitch_error_cents": float(np.median(all_errors)) if all_errors else None,
                "voiced_coverage_share": total_covered / total_voiced if total_voiced else None,
                "voiced_seconds": total_voiced / fps, "covered_voiced_seconds": total_covered / fps}}


def transcribe(f0, fps, words=(), *, beats=None, downbeats=None, tempo_bpm=None,
               key=None, meter=None, line_breaks=(), mode="auto", subdivisions=4,
               beat_unit="1/4", line_indices=None, time_range=None,
               unit_seconds=0.125, tolerance_cents=35.0, min_note_s=0.12,
               max_drift_cents=20.0, max_leap_st=9.0, initial_blip_s=0.25,
               pitch_rule="strict", spread_cents=100.0, centre_window_s=0.112,
               merge_words=False, directional_drift_cents=20.0, max_excursion_cents=100.0,
               word_text=None, line_text=None, line_spans=None, voiced_prob=None, spelling_at=None,
               spread_timed_words=False, range_low=None):
    """Measure stable semitone-band runs and render an approximate ABC fragment.

    Strict: every accepted frame is within tolerance_cents of its written equal-tempered
    MIDI note; first/last-third median drift must be <= max_drift_cents. Runs are
    split at word/line boundaries and each must last >= min_note_s. A boundary
    split does NOT measure a new articulation. No smoothing or gap bridging.

    Centre: segment a short running median, then test the UNSMOOTHED median,
    p90-p10 spread and endpoint drift. No unvoiced gaps or octave flags bridged.
    merge_words allows one pitch to carry several words joined with ABC '~';
    those words share a measured pitch, not separately measured attacks.

    Abrupt leaps >= max_leap_st within a voiced island decline the destination
    until it returns near the last trusted pitch. A brief initial outlier may
    also be declined using the next longer run. These are suspicion heuristics,
    not corrected pitches; real leaps can be lost and persistent errors missed.
    Optional frame-aligned pyin voiced_prob protects confident (median >= .5)
    runs from that heuristic. A short (<= initial_blip_s), weak (median < .2)
    low note between the nearest better-supported neighbours (median >= .4),
    each 12 or 19 semitones higher (+/-1), is omitted across gaps <= .75s.
    A weak cluster containing both half- and third-frequency alternatives to
    the same nearby, well-supported register pitch may also be omitted.
    No pitch is corrected. Missing probabilities keep the legacy rules;
    pitch stability is not confidence, nor voicing probability pitch accuracy.

    Auto with tempo: try eighths/sixteenths; require median <=25ms, p90 <=50ms
    and no collapsed notes. Without beats, use tempo plus an F0 onset proxy.
    Explicit quantised mode honours the caller's grid/subdivisions without gating.
    Proportional: small 1/2/4 length set, line-local time calibration, not metre.
    line_indices are zero-based; time_range selects whole intersecting lyric slices.
    Bad input raises ValueError; measured omissions are in result['declined'].
    word_text ({word index: text}) and line_text (one string per lyric line) only
    change what is printed under notes and in line comments, never the measurements.
    line_spans ({line index: (start, end)}) marks approximate lyric rows. Use
    complete timed words where rendered, then try note onsets in the span.
    spread_timed_words allows line starts inside words, bounds notation to
    line_spans, and uses supported word timings or approximate onset order.
    spelling_at(t) can choose a local enharmonic spelling. The ABC key and
    accidental state still determine which accidentals must be written.
    """
    fps = _positive("fps", fps)
    tolerance_cents = _positive("tolerance_cents", tolerance_cents)
    if tolerance_cents >= 50:
        raise ValueError("tolerance_cents must be below 50 (nonoverlapping pitch bands)")
    min_note_s = _positive("min_note_s", min_note_s)
    max_drift_cents = _positive("max_drift_cents", max_drift_cents)
    max_leap_st = _positive("max_leap_st", max_leap_st)
    initial_blip_s = _positive("initial_blip_s", initial_blip_s)
    spread_cents = _positive("spread_cents", spread_cents)
    centre_window_s = _positive("centre_window_s", centre_window_s)
    directional_drift_cents = _positive("directional_drift_cents", directional_drift_cents)
    max_excursion_cents = _positive("max_excursion_cents", max_excursion_cents)
    if pitch_rule not in ("strict", "centre"):
        raise ValueError("pitch_rule must be strict or centre")
    if not isinstance(merge_words, bool):
        raise ValueError("merge_words must be boolean")
    unit_seconds = _positive("unit_seconds", unit_seconds)
    if isinstance(subdivisions, bool) or int(subdivisions) != subdivisions or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    subdivisions = int(subdivisions)
    if not isinstance(beat_unit, str) or not re.fullmatch(r"[1-9]\d*/[1-9]\d*", beat_unit):
        raise ValueError("beat_unit must be a positive fraction, e.g. '1/4' or '3/8'")
    beat_length = Fraction(beat_unit)
    beat_text = f"{beat_length.numerator}/{beat_length.denominator}"
    if tempo_bpm is not None:
        tempo_bpm = _positive("tempo_bpm", tempo_bpm)
    abc_key = _key(key)
    if meter is not None and not re.fullmatch(r"[1-9]\d*/[1-9]\d*", meter):
        raise ValueError("meter must be an explicit fraction, e.g. '3/4'")
    f0 = np.asarray(f0, dtype=float)
    if f0.ndim != 1:
        raise ValueError("f0 must be a one-dimensional sequence")
    if voiced_prob is not None:
        voiced_prob = np.asarray(voiced_prob, dtype=float)
        if voiced_prob.shape != f0.shape or np.any(np.isinf(voiced_prob)) or np.any(
                np.isfinite(voiced_prob) & ((voiced_prob < 0) | (voiced_prob > 1))):
            raise ValueError("voiced_prob must match f0 and contain probabilities in [0, 1] or null")

    def confidence(a, b):
        values = voiced_prob[a:b] if voiced_prob is not None else np.array([])
        return float(np.median(values)) if len(values) and np.all(np.isfinite(values)) else None

    duration = len(f0) / fps
    beats = _times("beats", () if beats is None else beats)
    downbeats = _times("downbeats", () if downbeats is None else downbeats)
    cuts = _times("line_breaks", line_breaks)
    if any(t <= 0 or t >= duration for t in cuts):
        raise ValueError("line_breaks must lie strictly inside the pitch track")
    if mode not in ("auto", "quantised", "proportional"):
        raise ValueError("mode must be auto, quantised, or proportional")
    requested_mode = mode
    mode = ("quantised" if beats else "proportional") if mode == "auto" else mode
    if mode == "quantised" and len(beats) < 2:
        raise ValueError("quantised mode needs at least two beat times")
    if mode == "proportional" and meter is not None:
        if requested_mode != "auto" or tempo_bpm is None:
            raise ValueError("a metre cannot be rendered in proportional mode")
    if line_indices is not None and time_range is not None:
        raise ValueError("choose line_indices or time_range, not both")
    selected_lines = set(range(len(cuts) + 1))
    if line_indices is not None:
        indices = list(line_indices)
        if not indices or any(isinstance(i, bool) or not isinstance(i, int) or i not in selected_lines for i in indices):
            raise ValueError("line_indices must be nonempty zero-based lyric line indices")
        selected_lines = set(indices)
    if time_range is not None:
        if len(time_range) != 2:
            raise ValueError("time_range must contain start and end seconds")
        a, b = map(float, time_range)
        if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= duration:
            raise ValueError("time_range must be nonempty and within the track")
        edges = [0.0] + cuts + [duration]
        selected_lines = {i for i, (start, end) in enumerate(zip(edges, edges[1:])) if start < b and end > a}

    word_rows = []
    for i, (text, start, end) in enumerate(words):
        start, end = float(start), float(end)
        if not isinstance(text, str) or not text.strip() or any(ord(c) < 32 for c in text):
            raise ValueError("words must contain nonempty text without control characters")
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("word intervals must be finite, nonnegative, and nonempty")
        if word_rows and start < word_rows[-1]["start"]:
            raise ValueError("words must be ordered by start time")
        if not spread_timed_words and any(start < t < end for t in cuts):
            raise ValueError("line_breaks must fall between words")
        word_rows.append({"index": i, "word": text.strip(), "start": start, "end": end,
                          "note_indices": []})

    def shown_word(wi):
        return word_rows[wi]["word"] if word_text is None else word_text.get(wi, word_rows[wi]["word"])

    declined = []
    midpoint = (np.arange(len(f0)) + 0.5) / fps
    # Ownership is sampled at midpoints, so a requested line cut can differ from
    # its actual frame edge by half a frame. Use that SAME edge for rendering:
    # rounding the requested cut separately could duplicate time across lines.
    frame_cuts = [int(np.searchsorted(midpoint, t, side="left")) / fps for t in cuts]
    line_id = np.searchsorted(cuts, midpoint, side="right")
    owner = np.full(len(f0), -1, dtype=int)
    for w in word_rows:
        mask = (midpoint >= w["start"]) & (midpoint < w["end"])
        if spread_timed_words:
            mask &= line_id == np.searchsorted(cuts, w["start"], side="right")
        owner[mask] = np.where(owner[mask] == -1, w["index"], -2)
    valid = np.isfinite(f0) & (f0 > 0)
    semitones = np.full(len(f0), np.nan)
    semitones[valid] = 69 + 12 * (np.log2(f0[valid]) - math.log2(440.0))
    valid &= (semitones >= -0.5) & (semitones < 127.5)
    target = np.full(len(f0), -1, dtype=int)
    target[valid] = np.floor(semitones[valid] + 0.5).astype(int)
    reason = np.full(len(f0), "unstable_pitch", dtype=object)
    reason[np.isnan(f0)] = "unvoiced"
    reason[~valid & ~np.isnan(f0)] = "invalid_pitch"

    in_window = np.ones(len(f0), dtype=bool)
    if spread_timed_words and line_spans:
        for i, (start, end) in line_spans.items():
            in_window[(line_id == i) & ((midpoint < start) | (midpoint >= end))] = False
        reason[valid & ~in_window] = "outside_lyric_window"

    # Check raw runs BEFORE duration filtering, so even a one-frame octave blip
    # is accounted for rather than hidden by a median filter.
    raw = list(_runs(np.where(in_window, target, -1)))
    reference = None
    for k, (a, b, pitch) in enumerate(raw):
        if pitch < 0:
            reference = None
            continue
        initial = (reference is None and (b - a) / fps <= initial_blip_s and
                   k + 1 < len(raw) and raw[k + 1][2] >= 0 and
                   raw[k + 1][1] - raw[k + 1][0] > b - a and
                   abs(raw[k + 1][2] - pitch) >= max_leap_st)
        prob = confidence(a, b)
        if (prob is None or prob < 0.5) and (
                initial or (reference is not None and abs(pitch - reference) >= max_leap_st)):
            reason[a:b] = "implausible_leap"
        else:
            reference = pitch

    cents = 100 * (semitones - target)
    eligible = valid & in_window
    if pitch_rule == "strict":
        eligible &= np.abs(cents) <= tolerance_cents + 1e-9
    eligible &= reason != "implausible_leap"
    if not spread_timed_words:
        reason[eligible & (owner == -2)] = "ambiguous_word_overlap"
        eligible &= owner != -2
    segment_target = target.copy()
    window = max(1, round(centre_window_s * fps) // 2 * 2 + 1)
    if pitch_rule == "centre":
        # Smooth only segmentation labels. Never replace the values measured
        # below, and never hide an octave-suspect or unvoiced frame in a window.
        islands = [(int(line_id[i]),) if eligible[i] else None for i in range(len(f0))]
        for a, b, label in _runs(islands):
            if label is None:
                continue
            half = window // 2
            padded = np.pad(semitones[a:b], (half, half), mode="edge")
            centre = np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
            segment_target[a:b] = np.floor(centre + 0.5).astype(int)
        # Percentiles alone can hide a short, multi-semitone excursion. Leave
        # those measured frames as gaps instead of flattening them into a note.
        eligible &= np.abs(100 * (semitones - segment_target)) <= max_excursion_cents + 1e-9
    labels = [(int(segment_target[i]), -1 if merge_words else int(owner[i]), int(line_id[i])) if eligible[i] else None
              for i in range(len(f0))]
    notes = []
    for a, b, label in _runs(labels):
        if label is None:
            continue
        reason[a:b] = "too_short"
        if (b - a) / fps + 1e-12 < min_note_s:
            continue
        pitch, wi, li = label
        measured = 100 * (semitones[a:b] - pitch)
        edge = max(1, (b - a) // 3)
        drift = float(abs(np.median(measured[-edge:]) - np.median(measured[:edge])))
        spread = float(np.percentile(measured, 90) - np.percentile(measured, 10))
        variation = float(np.sum(np.abs(np.diff(measured))))
        directionality = float(abs(measured[-1] - measured[0]) / variation) if variation > 1e-9 else 0.0
        if pitch_rule == "centre":
            reason[a:b] = "unstable_pitch"
            if abs(np.median(measured)) > tolerance_cents + 1e-9 or spread > spread_cents + 1e-9:
                continue
        reason[a:b] = "pitch_drift"
        if (drift > max_drift_cents + 1e-9 or
                (pitch_rule == "centre" and directionality >= 0.8 and drift > directional_drift_cents + 1e-9)):
            continue
        word_indices = sorted(set(int(i) for i in owner[a:b] if i >= 0))
        if merge_words:
            wi = word_indices[0] if word_indices else -1
        notes.append({"start": a / fps, "end": b / fps, "midi": pitch,
                      "cents_off": float(np.median(measured)),
                      "max_deviation_cents": float(np.max(np.abs(measured))),
                      "spread_cents": spread,
                      "directionality": directionality,
                      "drift_cents": drift, "frames": b - a, "line": li,
                      "word_index": wi if wi >= 0 else None,
                      "word": word_rows[wi]["word"] if wi >= 0 else None,
                      "word_indices": word_indices,
                      "word_overlap_s": [float(np.sum(owner[a:b] == i) / fps) for i in word_indices],
                      "rendered": False})
        reason[a:b] = "accepted"
    low_notes = {}
    probabilities = [confidence(round(n["start"] * fps), round(n["end"] * fps)) for n in notes]
    anchors = [i for i, prob in enumerate(probabilities) if prob is not None and prob >= 0.4]
    for i, note in enumerate(notes):
        prob = probabilities[i]
        if note["end"] - note["start"] > initial_blip_s + 1e-9 or prob is None or prob >= 0.2:
            continue
        before = next((j for j in reversed(anchors) if j < i), None)
        after = next((j for j in anchors if j > i), None)
        if before is None or after is None:
            continue
        previous, following = notes[before], notes[after]
        neighbours = (previous, following)
        if note["start"] - previous["end"] > 0.75 or following["start"] - note["end"] > 0.75:
            continue
        if not all(any(abs(n["midi"] - note["midi"] - jump) <= 1 for jump in (12, 19)) for n in neighbours):
            continue
        low_notes[i] = (before, after)
    clustered = set()
    for before, after, cluster in _subharmonic_clusters(notes, probabilities, voiced_prob, target, fps):
        for i in cluster:
            low_notes[i] = (before, after)
        clustered.update(cluster)
    extra_lows = {i: (before, after) for i, before, after in
                  _low_subharmonics(notes, probabilities, voiced_prob, target, fps, range_low)
                  if i not in low_notes}
    octave_outliers = dict(_octave_outliers(notes, target, fps, initial_blip_s, range_low))
    low_note_omissions, high_note_omissions = [], []
    for i in sorted(low_notes.keys() | extra_lows.keys() | octave_outliers.keys()):
        note = notes[i]
        a, b = round(note["start"] * fps), round(note["end"] * fps)
        why = ("likely_tracker_subharmonic" if i in clustered or i in extra_lows
               else "likely_tracker_octave_error")
        reason[a:b] = why
        if i in low_notes:
            before, after = low_notes[i]
            neighbour_midi = [notes[j]["midi"] for j in (before, after)]
            neighbour_prob = [probabilities[j] for j in (before, after)]
        elif i in extra_lows:
            neighbour_midi = list(extra_lows[i])
            neighbour_prob = None
        else:
            neighbour_midi = list(octave_outliers[i])
            neighbour_prob = None
        high = i in octave_outliers and note["midi"] > max(octave_outliers[i])
        omissions = high_note_omissions if high else low_note_omissions
        omissions.append({"start": note["start"], "end": note["end"], "line": note["line"],
                          "midi": note["midi"], "reason": why, "median_voiced_prob": probabilities[i],
                          "neighbour_midi": neighbour_midi, "neighbour_voiced_prob": neighbour_prob})
    notes = [n for n in notes if reason[round(n["start"] * fps)] == "accepted"]
    for idx, note in enumerate(notes):
        for wi in note["word_indices"]:
            word_rows[wi]["note_indices"].append(idx)
    for a, b, why in _runs(reason):
        if why != "accepted":
            declined.append({"scope": "pitch", "start": a / fps, "end": b / fps,
                             "frames": b - a, "reason": why})

    supplied_beats = list(beats)
    grid_source = "caller-supplied beat times" if beats else "none"
    anchor = None
    if not beats and tempo_bpm is not None and requested_mode != "proportional":
        # F0 has no intensity: a substantial retained pitch centre is only an
        # onset proxy, never a measured beat or an energy-defined strong onset.
        for note in notes:
            if note["end"] - note["start"] >= 0.16:
                anchor = note["start"]
                break
        if anchor is not None:
            period = 60 / tempo_bpm
            beats = (anchor + np.arange(math.floor(-anchor / period),
                                       math.ceil((duration - anchor) / period) + 1) * period).tolist()
            grid_source = "tempo + first retained >=160ms pitch-centre onset (F0 proxy, not energy)"
    # Selection changes layout, never the inferred rhythm of the chosen tune.
    fit_notes = [n for n in notes if n["line"] in selected_lines]
    candidates = [_grid_fit(notes, beats, s) for s in (2, 4)] if beats else []
    selected_fits = [_grid_fit(fit_notes, beats, s) for s in (2, 4)] if beats else []
    fit = None
    fallback = None
    if requested_mode == "auto" and tempo_bpm is not None:
        usable = [c for c in candidates if c["median_ms"] is not None and c["median_ms"] <= 25 and
                  c["p90_ms"] <= 50 and c["collapsed_notes"] == 0]
        if usable:
            # Use the coarsest passing grid; extra resolution must earn its place.
            fit = usable[0]
            subdivisions = fit["subdivisions_per_beat"]
            mode = "quantised"
        else:
            fit = min(candidates, key=lambda c: c["p90_ms"] if c["p90_ms"] is not None else math.inf) if candidates else None
            mode = "proportional"
            fallback = "poor grid fit" if fit and fit["median_ms"] is not None else "no usable onset/grid"
    elif mode == "quantised":
        fit = _grid_fit(notes, beats, subdivisions)

    if mode == "quantised":
        def position(t):
            return float(np.interp(t, beats, np.arange(len(beats)) * subdivisions))

        def seconds(tick):
            return float(np.interp(tick, np.arange(len(beats)) * subdivisions, beats))

        low, high = max(0.0, beats[0]), min(duration, beats[-1])
        unit = beat_length / subdivisions
    else:
        def position(t):
            return t / unit_seconds

        def seconds(tick):
            return tick * unit_seconds

        low, high = 0.0, duration
        unit = Fraction(1, 8)
        if beats:
            declined.append({"scope": "rhythm", "reason": "beats_unused_in_proportional_mode"})
        if tempo_bpm is not None:
            declined.append({"scope": "rhythm", "reason": "tempo_unused_in_proportional_mode"})

    # A proportional sketch has no common performance tempo. Choose its short
    # duration from measured notes, then calibrate each line with ONE linear map.
    # Long inter-line gaps are trimmed; internal gaps are capped at four units.
    short_seconds = (float(np.median([n["end"] - n["start"] for n in notes])) / 2
                     if notes else unit_seconds)

    def readable_length(seconds):
        return min((1, 2, 4), key=lambda n: abs(seconds - n * short_seconds))

    def tick(t):
        return int(math.floor(position(t) + 0.5))

    unit_text = f"{unit.numerator}/{unit.denominator}"
    bars = set()
    measure_ticks = float(Fraction(meter) / unit) if meter is not None else None
    matched_downbeats = [tick(t) for t in downbeats if mode == "quantised" and low <= t <= high
                         and min(abs(t - b) for b in beats) <= 1e-6]
    consistent_bars = measure_ticks is not None and all(
        abs((b - a) / measure_ticks - round((b - a) / measure_ticks)) <= 1e-6
        for a, b in zip(matched_downbeats, matched_downbeats[1:]))
    for t in downbeats:
        on_grid = mode == "quantised" and low <= t <= high and min(abs(t - b) for b in beats) <= 1e-6
        if consistent_bars and on_grid:
            bars.add(tick(t))
        else:
            why = ("downbeats_unused_in_proportional_mode" if mode != "quantised" else
                   "downbeat_not_on_supplied_beat_grid" if not on_grid else
                   "metre_not_supplied" if meter is None else "downbeats_inconsistent_with_metre")
            declined.append({"scope": "downbeat", "start": t,
                             "reason": why})
    for i, note in enumerate(notes):
        why = None
        note["selected"] = note["line"] in selected_lines
        if not note["selected"]:
            continue
        if note["start"] < low - 1e-9 or note["end"] > high + 1e-9:
            why = "outside_beat_grid"
        elif mode == "quantised" and tick(note["end"]) <= tick(note["start"]):
            why = "duration_collapsed_on_grid"
        if why:
            declined.append({"scope": "note", "note_index": i, "start": note["start"],
                             "end": note["end"], "reason": why})
        else:
            a, b = tick(note["start"]), tick(note["end"])
            note.update(rendered=True, tick_start=a, tick_end=b,
                        written_start=seconds(a), written_end=seconds(b),
                        start_error_s=seconds(a) - note["start"],
                        end_error_s=seconds(b) - note["end"])
    for w in word_rows:
        w["status"] = ("outside_selection" if int(np.searchsorted(cuts, w["start"], side="right")) not in selected_lines else
                       "aligned" if any(notes[i]["rendered"] for i in w["note_indices"]) else
                       "no_renderable_note" if w["note_indices"] else "no_stable_note")
        if w["status"] not in ("aligned", "outside_selection"):
            declined.append({"scope": "word", "word_index": w["index"], "word": w["word"],
                             "start": w["start"], "end": w["end"], "reason": w["status"]})

    pitch_description = (f"every frame ±{tolerance_cents:g}c" if pitch_rule == "strict" else
                         f"median ±{tolerance_cents:g}c; p90-p10<={spread_cents:g}c; raw max ±{max_excursion_cents:g}c; "
                         f"median win {1000 * window / fps:g}ms; directional(.8)<={directional_drift_cents:g}c")
    fit_text = (f"grid median {fit['median_ms']:.1f}ms / p90 {fit['p90_ms']:.1f}ms (whole track)"
                if fit and fit["median_ms"] is not None else "grid fit unmeasured (no usable grid)")
    bar_rule = "bars at supplied downbeats" if bars else "no bars (no trusted metre/downbeats)"
    rhythm_description = (f"quantised L:{unit_text}; {subdivisions}/beat; {fit_text}" if mode == "quantised" else
                          f"proportional 1/2/4, NOT metrical; line-scaled, gaps simplified; "
                          f"{fallback or 'no tempo/grid requested'}; {fit_text}")
    if requested_mode == "auto" and tempo_bpm is not None:
        rhythm_description += "; gate 25/50ms, zero collapsed notes"
    provenance = (f"key {abc_key or 'none'}; tempo {tempo_bpm:g} BPM; caller supplied, not measured here"
                  if tempo_bpm is not None else
                  f"key {abc_key}; caller supplied, not measured here; no tempo"
                  if abc_key not in (None, "none") else "no key or tempo supplied")
    grid_header = ("supplied beats" if supplied_beats else "tempo + first >=160ms F0-centre onset"
                   if anchor is not None else "none")
    header = ["% ABC 2.1 fragment; approximate transcription from a pitch track; rests may omit voice",
              f"% pitch: {pitch_description}; min {1000 * min_note_s:g}ms; drift<={max_drift_cents:g}c",
              f"% rhythm: {rhythm_description}; {bar_rule}",
              f"% provenance: {provenance}; grid {grid_header}"]
    header.extend(["I:abc-version 2.1", "I:abc-creator melody_abc.py", f"L:{unit_text}"])
    if meter is not None and bars:
        header.append(f"M:{meter}")
    if tempo_bpm is not None and mode == "quantised":
        header.append(f"Q:{beat_text}={tempo_bpm:g}")
    if abc_key is not None:
        header.append(f"K:{abc_key}")

    lines, gaps, accidental_state = [], [], {}
    timed_by_line = {}
    if spread_timed_words:
        for w in word_rows:
            li = int(np.searchsorted(cuts, w["start"], side="right"))
            timed_by_line.setdefault(li, []).append(w)
    omitted_short_gaps = []
    edges = [0.0] + cuts + [duration]
    frame_edges = [0.0] + frame_cuts + [duration]
    for li, (start, end) in enumerate(zip(edges, edges[1:])):
        if li not in selected_lines:
            continue
        selected = [(i, n) for i, n in enumerate(notes) if n["line"] == li and n["rendered"]]
        text = line_text[li] if line_text and li < len(line_text) else ""
        span = (line_spans or {}).get(li)
        lyric_fit, labels, unaligned_reason = None, None, None
        if spread_timed_words:
            timed = timed_by_line.get(li, [])
            row_text = text or " ".join(filter(None, (shown_word(w["index"]) for w in timed)))
            line_words = _line_words(row_text)
            if line_words:
                rendered = [n for _, n in selected]
                labels = _timed_lyric_labels(line_words, timed, rendered)
                if labels is not None:
                    lyric_fit = "timed"
                else:
                    labels = _words_by_onset(line_words, span or (start, end), rendered, fps)
                    lyric_fit = "onset_order" if labels is not None else "unaligned"
                    if labels is None:
                        unaligned_reason = "too few stable note onsets" if rendered else "no stable notes"
                        text = row_text
        elif span:
            line_words = _line_words(text)
            rendered = [n for _, n in selected]
            if not _has_word_coverage(line_words, rendered, word_rows, shown_word):
                labels = _words_by_onset(line_words, span, rendered, fps)
                lyric_fit = "onset_order" if labels is not None else "unaligned"
        music, lyrics, sung_words = [], [], set()
        frame_start, frame_end = frame_edges[li:li + 2]
        if spread_timed_words and span:
            start, end = span
            frame_start, frame_end = np.searchsorted(midpoint, (start, end)) / fps
        line_low, line_high = max(frame_start, low), min(frame_end, high)
        # Trim padding between lyric lines; retain all original slice bounds in JSON.
        if selected:
            line_low, line_high = selected[0][1]["start"], selected[-1][1]["end"]
        elif mode == "proportional":
            line_high = line_low
        cursor = tick(line_low) if line_low <= line_high else 0
        stop = tick(line_high) if line_low <= line_high else 0
        seconds_per_tick = None
        if mode == "proportional":
            cursor = 0
            source_cursor = line_low
            for _, note in selected:
                gap = note["start"] - source_cursor
                if gap > 1e-9:
                    if gap < short_seconds / 2:
                        omitted_short_gaps.append({"line": li, "start": source_cursor, "end": note["start"]})
                    else:
                        cursor += readable_length(gap)
                note["tick_start"] = cursor
                cursor += readable_length(note["end"] - note["start"])
                note["tick_end"] = cursor
                source_cursor = note["end"]
            stop, cursor = cursor, 0
            seconds_per_tick = (line_high - line_low) / stop if stop else short_seconds
            for _, note in selected:
                a, b = note["tick_start"], note["tick_end"]
                note.update(written_start=line_low + a * seconds_per_tick,
                            written_end=line_low + b * seconds_per_tick,
                            start_error_s=line_low + a * seconds_per_tick - note["start"],
                            end_error_s=line_low + b * seconds_per_tick - note["end"])
        line_tick_start = cursor

        def emit(a, b, note=None, label=None):
            boundaries = [a] + sorted(t for t in bars if a < t < b) + [b]
            for j, (x, y) in enumerate(zip(boundaries, boundaries[1:])):
                if y <= x:
                    continue
                if note is None:
                    music.append('z' + _length(y - x))
                    gaps.append({"line": li, "tick_start": x, "tick_end": y})
                else:
                    local_key = spelling_at(note["start"]) if spelling_at else None
                    music.append(_pitch(note["midi"], abc_key, accidental_state, local_key) + _length(y - x) +
                                 ("-" if j < len(boundaries) - 2 else ""))
                    if lyric_fit:
                        if label is not None:
                            lyrics.append(label if j == 0 else "_" if label != "*" else "*")
                    else:
                        indices = note["word_indices"]
                        fresh = [wi for wi in indices if wi not in sung_words]
                        shown = [_lyric(t) for t in (shown_word(wi) for wi in fresh) if t]
                        lyrics.append("*" if not indices else "~".join(shown) if fresh and shown else
                                      "*" if fresh else "_")
                        sung_words.update(indices)
                if y in bars and y > cursor:
                    music.append("|")
                    accidental_state.clear()

        for k, (_, n) in enumerate(selected):
            if n["tick_start"] > cursor:
                emit(cursor, n["tick_start"])
            emit(n["tick_start"], n["tick_end"], n, labels[k] if labels else None)
            cursor = n["tick_end"]
        if stop > cursor:
            emit(cursor, stop)
        missing = [w["index"] for w in word_rows
                   if int(np.searchsorted(cuts, w["start"], side="right")) == li and w["status"] != "aligned"]
        stamp = int(span[0] if span and (spread_timed_words or lyric_fit) else line_low)
        approx = "≈" if lyric_fit in ("onset_order", "unaligned") or (spread_timed_words and span) else ""
        comment = f"% [{stamp // 60}:{stamp % 60:02d}{approx}]"
        if unaligned_reason:
            label = UNALIGNED_LYRICS.removesuffix(": ")
            comment += f" {label} ({unaligned_reason}):"
        if text:
            comment += " " + (UNALIGNED_LYRICS if lyric_fit == "unaligned" and not unaligned_reason else "") + text
        omissions = low_note_comments(low_note_omissions, li)
        high = sum(n["line"] == li for n in high_note_omissions)
        high_comments = [f"% {high} high note{'s' if high != 1 else ''} left out: likely tracker octave error"] if high else []
        for omission in omissions + high_comments:
            comment += "\n" + omission
        if missing and not lyric_fit and not spread_timed_words:
            kind = "stable" if all(word_rows[i]["status"] == "no_stable_note" for i in missing) else "rendered"
            comment += f"\n% {len(missing)} words without a {kind} note: " + ", ".join(filter(None, map(shown_word, missing)))
        # A comment plus empty w: is preferable to inventing a duration for a
        # zero-length or fully out-of-grid line. Its absence is explicit in JSON.
        if not music:
            comment += "; no renderable duration"
        header.extend([comment, " ".join(music) if music else "% no music"])
        if lyric_fit != "unaligned":
            header.append("w: " + " ".join(lyrics))
        lines.append({"index": li, "start": start, "end": end,
                      "frame_start": frame_start, "frame_end": frame_end,
                      "tick_start": line_tick_start if line_low <= line_high else None,
                      "tick_end": stop if line_low <= line_high else None,
                      "written_origin_s": line_low, "seconds_per_tick": seconds_per_tick,
                      "trimmed_leading_s": max(0, line_low - frame_start),
                      "trimmed_trailing_s": max(0, frame_end - line_high),
                      "note_indices": [i for i, _ in selected],
                      "unaligned_word_indices": missing, "music": " ".join(music),
                      "lyrics": " ".join(lyrics)})
        if omissions:
            lines[-1]["low_note_comments"] = omissions
        if high_comments:
            lines[-1]["high_note_comments"] = high_comments
        if lyric_fit:
            lines[-1]["lyric_fit"] = lyric_fit
        if unaligned_reason:
            lines[-1]["unaligned_reason"] = unaligned_reason

    settings = {"fps": fps, "duration_s": duration, "tolerance_cents": tolerance_cents,
                "min_note_s": min_note_s, "max_drift_cents": max_drift_cents,
                "max_leap_st": max_leap_st, "initial_blip_s": initial_blip_s,
                "pitch_rule": pitch_rule, "spread_cents": spread_cents,
                "centre_window_s": centre_window_s, "centre_window_frames": window,
                "directional_drift_cents": directional_drift_cents,
                "max_excursion_cents": max_excursion_cents,
                "merge_words": merge_words,
                "requested_unit_seconds": unit_seconds,
                "key": key, "meter": meter, "line_breaks": cuts, "frame_line_breaks": frame_cuts}
    rhythm = {"mode": mode, "requested_mode": requested_mode,
              "subdivisions_per_beat": subdivisions if mode == "quantised" else None,
              "beat_unit": beat_text if mode == "quantised" else None,
              "unit_seconds": None, "proportional_short_target_s": short_seconds if mode == "proportional" else None,
              "L": unit_text, "beats": beats, "downbeats": downbeats,
              "supplied_beats": supplied_beats, "grid_source": grid_source, "anchor_s": anchor,
              "fit": fit, "candidate_fits": candidates, "selected_candidate_fits": selected_fits,
              "fallback_reason": fallback,
              "fit_threshold": {"median_ms_max": 25, "p90_ms_max": 50, "collapsed_notes_max": 0},
              "bar_decision": bar_rule,
              "omitted_short_gaps": omitted_short_gaps,
              "short_gap_omission_threshold_s": short_seconds / 2 if mode == "proportional" else None,
              "tempo_bpm": tempo_bpm, "bar_ticks": sorted(bars),
              "coverage_start_s": low, "coverage_end_s": high if high >= low else None}
    abc = "\n".join(header) + "\n"
    return abc, {"format": "abc-2.1-fragment", "settings": settings,
            "rhythm": rhythm, "notes": notes, "words": word_rows, "declined": declined,
            "low_note_omissions": low_note_omissions,
            "high_note_omissions": high_note_omissions,
            "gaps": gaps, "lines": lines,
            "selection": {"line_indices": sorted(selected_lines), "time_range": time_range,
                          "rule": "whole lyric slices; zero-based indices; ranges intersect slices"},
            "notation": {"key_provenance": "caller supplied, not measured here" if key not in (None, "none") else "none",
                         "rests": "omitted pitch or unvoiced; inter-line padding trimmed; proportional gaps <half a short unit omitted, others capped at 4",
                         "accidentals": "key-aware; octave-scoped accidentals persist until bar/cancellation"
                         if key not in (None, "none") else "explicit on every note",
                         "attacks": "not measured; same-pitch words may share a note",
                         "leap_rule": "decline unconfirmed abrupt leaps; omit short isolated octave excursions and corroborated low subharmonics; no pitch correction"},
            "self_check": self_check(abc, f0, fps, rhythm, lines)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSON object with f0 (null = unvoiced), fps, words, and optional transcribe kwargs")
    source.add_argument("--vocals", help="existing vocals stem; CPU pyin via the ears' front end; no separation or ASR")
    parser.add_argument("--words", help="JSON word triples, or an existing shared lyrics cache object with 'words'")
    parser.add_argument("--key")
    parser.add_argument("--tempo-bpm", type=float, help="caller claim; try a grid anchored to a retained F0 onset if beats absent")
    parser.add_argument("--lyric-lines", help="writer's text; ordered alignment to supplied word timings")
    parser.add_argument("--profile", choices=PROFILES, help="strict (round 1 pitch rules), middle, or loose")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--line-indices", "--hook-lines", nargs="+", type=int,
                           help="emit chosen zero-based lyric lines (caller chooses the hook)")
    selection.add_argument("--time-range", "--hook-time", nargs=2, type=float, metavar=("START", "END"),
                           help="emit whole lyric slices intersecting this seconds range")
    selection.add_argument("--melody", action="store_true", help="emit the whole song, overriding input JSON selection")
    parser.add_argument("--save-f0", help="save audio front-end F0/fps for reuse with --input")
    parser.add_argument("--output", help="write ABC fragment here instead of stdout")
    parser.add_argument("--json-output", help="write measurements and declines here")
    parser.add_argument("--fmin", type=float, default=65.0)
    parser.add_argument("--fmax", type=float, default=1400.0)
    args = parser.parse_args(argv)
    from network_policy import local_audio_path
    if args.vocals:
        local_audio_path(args.vocals, parser)
    if args.vocals:
        if not args.words:
            parser.error("--vocals requires --words (cached timings; no transcription is run)")
        if not 0 < args.fmin < args.fmax:
            parser.error("require 0 < fmin < fmax")
        import raven_ears_plus as ears
        audio = ears.load_audio(args.vocals, ears.SR)
        track = ears._front_end(audio, fmin=args.fmin, fmax=args.fmax)
        inputs = {"f0": track["f0"], "fps": ears.FPS}
        if "voiced_prob" in track:
            inputs["voiced_prob"] = track["voiced_prob"]
        if args.save_f0:
            cached = {name: [float(x) if np.isfinite(x) else None for x in inputs[name]]
                      for name in ("f0", "voiced_prob") if name in inputs}
            cached["fps"] = ears.FPS
            Path(args.save_f0).write_text(json.dumps(cached, allow_nan=False) + "\n", encoding="utf-8")
    else:
        if args.save_f0:
            parser.error("--save-f0 requires --vocals")
        inputs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.words:
        data = json.loads(Path(args.words).read_text(encoding="utf-8"))
        inputs["words"] = data["words"] if isinstance(data, dict) else data
    if args.key is not None:
        inputs["key"] = args.key
    if args.tempo_bpm is not None:
        inputs["tempo_bpm"] = args.tempo_bpm
    if args.profile:
        inputs.update(PROFILES[args.profile])
    if args.melody or args.line_indices is not None or args.time_range is not None:
        inputs.pop("line_indices", None)
        inputs.pop("time_range", None)
    if args.line_indices is not None:
        inputs["line_indices"] = args.line_indices
    if args.time_range is not None:
        inputs["time_range"] = args.time_range
    placements = None
    if args.lyric_lines:
        from lyric_align import read_lyrics_file
        cuts, placements, summary = lyric_line_boundaries(
            read_lyrics_file(args.lyric_lines), inputs.get("words", ()), len(inputs["f0"]) / inputs["fps"])
        inputs["line_breaks"] = cuts
    abc, result = transcribe(**inputs)
    if placements is not None:
        result["lyric_alignment"] = {"summary": summary, "rows": placements}
        for line in result["lines"]:
            line["lyric_placement"] = placements[line["index"]]
    if args.output:
        Path(args.output).write_text(abc, encoding="utf-8")
    else:
        print(abc, end="")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
