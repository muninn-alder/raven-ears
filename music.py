#!/usr/bin/env python
"""Ears+ MUSIC mode (24 Oct 2026). `music.py <file> [--json] [--no-lyrics]`

Audio analysis is local (librosa, Demucs vocal stem for VOCALS, the PANNs tagger only as a fallback, and the
shared faster-whisper large-v3 in lyrics_shared.py). LRCLIB receives song metadata unless --no-lyrics-lookup
is set. Music mode never touches a speaker baseline (it has no baseline code at all).

Card lines: KEY · TEMPO · SHAPE (sections, with PEAK) · SOUND · LYRICS.
Every number is measured; a field that can't be measured is left out, not printed as "unknown".
"""
from __future__ import annotations

import ears_profile as EP
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SR = 22050
HOP = 512
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Krumhansl-Kessler profiles
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


import stage_status as ST  # HONEST CARD: run-local outcomes.

def _t(s):
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def _log(m):
    print(f"[music] {m}", file=sys.stderr)


# --------------------------------------------------------------------------- key
def _key_scores(chroma_vec):
    """24 correlations (12 major then 12 minor)."""
    out = []
    for prof in (KS_MAJ, KS_MIN):
        for k in range(12):
            out.append(np.corrcoef(chroma_vec, np.roll(prof, k))[0, 1])
    return np.array(out)


def _kname(i):
    return f"{NAMES[i % 12]} {'major' if i < 12 else 'minor'}"


def _area(i):
    """Key area = the major key it shares a signature with (so C major and A minor are one area)."""
    return i if i < 12 else (i - 12 + 3) % 12


DOMINANT_PITCH_RATIO = 0.80


def dominant_pull(kinfo, chroma, dur, home):
    """Withhold unsupported same-mode V/v departures, retaining the raw estimate.

    Chroma leaks into adjacent semitones. Require the new signature note to be
    a trough, <=80% of the stronger neighbour, in EVERY winning window before
    withholding a move. One supported window protects it. This measures missing
    support, not literal note absence. In minor the new note is raised 6, not
    harmonic minor's raised 7. Other intervals and mode changes are untouched.
    Recheck after home fusion: a spectral home's annotation is provisional.
    """
    result = dict(kinfo)
    changes = [dict(c) for c in kinfo.get("changes", [])]
    for c in changes:
        if c.get("suppressed") == "dominant pull":
            c.pop("suppressed")
            c.pop("evidence", None)
    if not changes:
        return result
    result["changes"] = changes
    candidates = []
    for index, c in enumerate(changes):
        tonic, mode = c["to"].split()
        target = NAMES.index(tonic)
        if mode == home[1] and (target - home[0]) % 12 == 7:
            end = changes[index + 1]["at"] if index + 1 < len(changes) else dur
            candidates.append((c, target + (12 if mode == "minor" else 0), end))
    if not candidates:
        return result
    fps = SR / HOP
    W, step = int(20 * fps), int(10 * fps)
    starts = list(range(0, chroma.shape[1] - W + 1, step))
    vectors = [chroma[:, a:a + W].mean(axis=1) for a in starts]
    areas = [_area(int(np.argmax(_key_scores(v)))) for v in vectors]
    pc = (home[0] + (6 if home[1] == "major" else 9)) % 12
    for c, target, end in candidates:
        if not starts:
            continue
        i = min(range(len(starts)), key=lambda j: abs(starts[j] / fps - c["at"]))
        j = i
        while j < len(starts) and areas[j] == _area(target):
            j += 1
        if j - i < 3 or abs(starts[i] / fps - c["at"]) > 0.1:
            continue
        # A later supported run in the same stored span must protect the move
        # too: the raw detector may never have emitted the intervening return.
        indices = [n for n in range(i, len(starts)) if starts[n] / fps < end and areas[n] == _area(target)]
        evidence = []
        for n in indices:
            a, v = starts[n], vectors[n]
            if not np.isfinite(v).all() or v.sum() <= 0:
                break
            left, right = float(v[(pc - 1) % 12]), float(v[(pc + 1) % 12])
            ratio = float(v[pc]) / max(left, right, 1e-9)
            if ratio > DOMINANT_PITCH_RATIO or v[pc] >= min(left, right):
                break
            evidence.append({"at": round(a / fps, 3), "energy": round(float(v[pc]), 6),
                             "left": round(left, 6), "right": round(right, 6), "ratio": round(ratio, 6)})
        else:
            c.update(suppressed="dominant pull", evidence={
                "from": f"{NAMES[home[0]]} {home[1]}", "pitch": NAMES[pc],
                "end": round(min(dur, (starts[indices[-1]] + W) / fps), 3),
                "threshold": DOMINANT_PITCH_RATIO, "windows": evidence})
    return result


def reentry_changes(kinfo, chroma, dur, home, stops):
    """Demote a bounded signature lean when a later band return confirms the move.

    Require an interrupted early run with a consistently unsupported alteration,
    a return to the home signature before a confirmed stop, then three target
    windows supporting EVERY alteration after re-entry. Preserve the raw lean.
    A supported early modulation keeps its time. This does not choose a tonic.
    """
    if not kinfo.get("changes") or not stops or stops.get("status") != "ran":
        return kinfo
    fps = SR / HOP
    width, hop = int(20 * fps), int(10 * fps)
    starts = list(range(0, chroma.shape[1] - width + 1, hop))
    vectors = [chroma[:, a:a + width].mean(axis=1) for a in starts]
    if not vectors or not np.isfinite(vectors).all():
        return kinfo
    keys = [int(np.argmax(_key_scores(v))) for v in vectors]
    areas = [_area(k) for k in keys]
    home_area = (home[0] + {"major": 0, "minor": 3, "mixolydian": 5}[home[1]]) % 12
    scale = {0, 2, 4, 5, 7, 9, 11}
    home_notes = {(home_area + n) % 12 for n in scale}
    changes = [dict(c) for c in kinfo["changes"]]
    additions = []
    for ci, change in enumerate(changes):
        # Leave opening establishment and the existing dominant veto alone.
        if change.get("suppressed") or change["at"] < 25:
            continue
        name, mode = change["to"].split()
        target = NAMES.index(name) + (12 if mode == "minor" else 0)
        area = _area(target)
        target_notes = {(area + n) % 12 for n in scale}
        new, old = target_notes - home_notes, home_notes - target_notes
        pairs = [(n, o) for n in sorted(new) for o in sorted(old) if (n - o) % 12 in (1, 11)]
        if not new or len(pairs) != len(new) or len({o for n, o in pairs}) != len(old):
            continue
        i = min(range(len(starts)), key=lambda j: abs(starts[j] / fps - change["at"]))
        j = i
        while j < len(areas) and areas[j] == area:
            j += 1
        if abs(starts[i] / fps - change["at"]) > .1 or j - i < 3:
            continue
        if not any(all(vectors[w][n] <= vectors[w][o] for w in range(i, j)) for n, o in pairs):
            continue
        end = changes[ci + 1]["at"] if ci + 1 < len(changes) else dur
        for event in stops.get("events", []):
            if event.get("kind") not in ("band cuts, voice holds", "full silence", "band break") or event.get("arrangement") == "partial cut":
                continue
            cut = event["start_s"]
            at = round(cut + event["length_s"], 2)
            if cut < 10 or at >= end or at + 40 > dur:
                continue
            # Two contrary estimates can interrupt a run without satisfying
            # the original detector's three-window return rule.
            later = next((w for w in range(j + 2, len(areas) - 2)
                          if starts[w] / fps <= at <= (starts[w] + width) / fps
                          and areas[w:w + 3] == [area] * 3
                          and areas[w - 1] != area), None)
            if later is None:
                continue
            before = chroma[:, int((cut - 10) * fps):int(cut * fps)].mean(axis=1)
            if not np.isfinite(before).all() or _area(int(np.argmax(_key_scores(before)))) != home_area:
                continue
            after = [chroma[:, int((at + t) * fps):int((at + t + 20) * fps)].mean(axis=1) for t in (0, 10, 20)]
            if not np.isfinite(after).all() or any(int(np.argmax(_key_scores(v))) != target for v in after):
                continue
            if not all(v[n] > v[o] for v in after for n, o in pairs):
                continue
            evidence = dict(source="confirmed band re-entry", at=at,
                            early_window_span=[round(starts[i] / fps, 3), round((starts[j - 1] + width) / fps, 3)],
                            home_window=[round(cut - 10, 2), cut],
                            alterations=[dict(new=NAMES[n], old=NAMES[o],
                                              early_ratios=[round(float(vectors[w][n] / max(vectors[w][o], 1e-9)), 6) for w in range(i, j)],
                                              after_ratios=[round(float(v[n] / max(v[o], 1e-9)), 6) for v in after]) for n, o in pairs])
            change.update(suppressed="bounded key lean", evidence=evidence)
            additions.append(dict(at=at, to=change["to"], evidence=evidence))
            break
    if not additions:
        return kinfo
    result = dict(kinfo)
    result["changes"] = sorted(changes + additions, key=lambda c: c["at"])
    return result


@ST.stage('spectral key')
def analyze_key(chroma, dur):
    fps = SR / HOP
    g = chroma.mean(axis=1)
    sc = _key_scores(g)
    order = np.argsort(sc)[::-1]
    best, second = int(order[0]), int(order[1])
    # runner-up that isn't the relative/same-area key, for an honest margin
    alt = next((int(j) for j in order[1:] if _area(int(j)) != _area(best)), second)
    res = {"key": _kname(best), "r": round(float(sc[best]), 2),
           "margin": round(float(sc[best] - sc[alt]), 2)}
    if _area(second) == _area(best):
        res["also"] = _kname(second)          # relative major/minor ambiguity
    # windowed estimate: 20 s windows, 10 s hop; a change must hold >= 3 windows (~30 s)
    changes = []
    if dur >= 75:
        W, H = int(20 * fps), int(10 * fps)
        wins = []
        for a in range(0, chroma.shape[1] - W + 1, H):
            s = _key_scores(chroma[:, a:a + W].mean(axis=1))
            wins.append((a / fps, int(np.argmax(s))))
        areas = [_area(k) for _, k in wins]
        cur = _area(best)
        i = 0
        while i < len(wins):
            if areas[i] != cur:
                j = i
                while j < len(wins) and areas[j] == areas[i]:
                    j += 1
                if j - i >= 3:
                    ks = [k for _, k in wins[i:j]]
                    kk = max(set(ks), key=ks.count)
                    changes.append({"at": round(wins[i][0], 1), "to": _kname(kk)})
                    cur = areas[i]
                i = j
            else:
                i += 1
    if changes:
        res["changes"] = changes
    return dominant_pull(res, chroma, dur, (best % 12, "major" if best < 12 else "minor"))


# --------------------------------------------------------------------------- tempo
@ST.stage('tempo')
def analyze_tempo(y_perc, y, oenv=None):
    import librosa
    if oenv is None:
        oenv = librosa.onset.onset_strength(y=y_perc, sr=SR, hop_length=HOP)
    if oenv.max() <= 0:
        return None
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0 or len(beats) < 8:
        return None
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=HOP)
    ibi = np.diff(bt)
    cv = float(np.std(ibi) / np.mean(ibi))
    # regularity: 1 = metronome. Use robust spread (MAD) so one skipped beat doesn't wreck it.
    mad = float(np.median(np.abs(ibi - np.median(ibi))) / np.median(ibi))
    reg = float(np.clip(1 - 4 * mad, 0, 1))
    strength = float(np.mean(oenv[beats]) / (np.mean(oenv) + 1e-9))
    res = {"bpm": round(bpm, 1), "regularity": round(reg, 2), "beat_strength": round(strength, 2),
           "beats": len(beats), "_beats": beats}
    # half/double-time hint: if the tempo is outside 70-170 the beat tracker may be on a harmonic
    if bpm < 70 or bpm > 170:
        res["note"] = f"could also feel like {bpm * 2:.0f} or {bpm / 2:.0f}"
    # time signature: only claim 3/4 vs 4/4 when beat-strength periodicity is clearly one or the other
    try:
        bs = oenv[beats]
        bs = bs - bs.mean()
        ac = {p: float(np.corrcoef(bs[:-p], bs[p:])[0, 1]) for p in (3, 4) if len(bs) > 4 * p}
        if len(ac) == 2 and abs(ac[3] - ac[4]) > 0.25:
            res["meter"] = "3/4" if ac[3] > ac[4] else "4/4"
    except Exception:
        pass
    return res


# --------------------------------------------------------------------------- sections
@ST.stage('sections')
def analyze_sections(y, chroma, rms_db, dur, mfcc=None):
    import librosa
    fps = SR / HOP
    n = rms_db.shape[0]
    k = int(np.clip(round(dur / 28), 3, 10))
    if mfcc is None:
        mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    # work at ~2 frames/sec: median-pool features so the clustering sees structure, not notes
    step = int(fps / 2)
    nb = n // step
    if nb < 6:
        return None
    def pool(m):
        return np.stack([np.median(m[:, i * step:(i + 1) * step], axis=1) for i in range(nb)], axis=1)
    def z(m):
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)
    feats = np.vstack([z(pool(chroma)), z(pool(mfcc)[1:])])
    feats = librosa.segment.path_enhance(feats, 5) if False else feats
    # temporal smoothing keeps segments contiguous
    from scipy.ndimage import uniform_filter1d
    feats = uniform_filter1d(feats, size=5, axis=1)
    k = min(k, nb - 1)
    bounds = librosa.segment.agglomerative(feats, k)          # frame indices of starts (in pooled frames)
    bounds = sorted(set(int(b) for b in bounds) | {0})
    times = [b * step / fps for b in bounds] + [dur]
    # merge anything shorter than 8 s into its neighbour
    segs = [[times[i], times[i + 1]] for i in range(len(times) - 1)]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (a, b) in enumerate(segs):
            if b - a < 8:
                if i == 0:
                    segs[1][0] = a
                else:
                    segs[i - 1][1] = b
                segs.pop(i)
                changed = True
                break
    out = []
    for a, b in segs:
        fa, fb = int(a * fps), max(int(b * fps), int(a * fps) + 1)
        seg = rms_db[fa:fb]
        # How loud the section PLAYS: the median of its frames that carry sound. A plain mean in dB lets a fade tail or
        # digital silence (-120 dB) drag a full-band section down to "quiet" (Fearless 3:13-4:05: 38 s at -18 dB plus
        # 12 s of nothing read as "quiet outro -29.8 dB").
        live = seg[seg > rms_db.max() - 60] if len(seg) else seg
        out.append({"start": round(a, 1), "end": round(b, 1), "db": float(np.median(live if len(live) else seg))})
    dbs = np.array([s["db"] for s in out])
    lo, hi = dbs.min(), dbs.max()
    span = hi - lo
    for i, s in enumerate(out):
        rel = (s["db"] - lo) / span if span > 1.0 else 0.5
        s["rel"] = round(float(rel), 2)
        s["level"] = "quiet" if rel < 0.33 else ("mid" if rel < 0.72 else "loud")
        s["db"] = round(s["db"], 1)
    if span > 1.0:
        out[int(np.argmax(dbs))]["peak"] = True
    return out


# --------------------------------------------------------------------------- sound
@ST.stage('sound')
def analyze_sound(y, rms_db, hp_energy=None, cent=None):
    import librosa
    if cent is None:
        cent = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP)[0]
    m = len(cent)
    act = rms_db[:m] > (rms_db.max() - 40)
    c = float(np.median(cent[act])) if act.any() else float(np.median(cent))
    tone = "dark" if c < 1400 else ("warm" if c < 2200 else ("balanced" if c < 3200 else "bright"))
    # dynamic range: loud (95th pct) minus quiet (10th pct) of short-term loudness, ignoring digital silence
    r = rms_db[rms_db > rms_db.max() - 60]
    dr = float(np.percentile(r, 95) - np.percentile(r, 10)) if len(r) > 20 else None
    # harmonic / percussive
    if hp_energy is None:
        H, P = librosa.decompose.hpss(librosa.stft(y, hop_length=HOP))
        hp_energy = float(np.sum(np.abs(H) ** 2)), float(np.sum(np.abs(P) ** 2))
    eh, ep = hp_energy
    hfrac = eh / (eh + ep + 1e-12)
    res = {"centroid_hz": round(c), "tone": tone, "harmonic_frac": round(hfrac, 2)}
    if dr is not None:
        res["dyn_range_db"] = round(dr, 1)
    peak = float(np.max(np.abs(y)))
    res["peak_dbfs"] = round(20 * np.log10(peak + 1e-9), 1)
    return res


def _span(a, b):
    """'2:48' rather than '2:48–2:48' when a span's ends print as the same second (11 Nov 2026)."""
    return _t(a) if _t(a) == _t(b) else f"{_t(a)}–{_t(b)}"


def analyze_vocals_tagger(path, notes):
    """FALLBACK ONLY. PANNs AudioSet tags: fraction of the track where singing/speech is likely present.
    An ESTIMATE from a tagger, not a separation."""
    try:
        import raven_ears_plus as E
        y32 = E.load_audio(path, E.SR_SED)
        dev = E._pick_device(need_mb=1500)
        try:
            centers, probs, labels = E._tag_windows(y32, dev)
        except Exception as e:
            ST.mark("vocal tagger", "degraded", f"{type(e).__name__}: {str(e)[:60]}; retried on CPU")
            dev = "cpu"
            centers, probs, labels = E._tag_windows(y32, "cpu")
        ix = {l: i for i, l in enumerate(labels)}
        sing = [ix[l] for l in ("Singing", "Male singing", "Female singing", "Child singing", "Choir", "Rapping")
                if l in ix]
        speech = ix.get("Speech")
        if not sing:
            return None
        ps = probs[:, sing].max(axis=1)
        pv = np.maximum(ps, probs[:, speech]) if speech is not None else ps
        frac = float((pv >= 0.12).mean())
        kinds = []
        for l in ("Male singing", "Female singing", "Choir", "Rapping"):
            if l in ix and (probs[:, ix[l]] >= 0.12).mean() > 0.1:
                kinds.append(l.lower())
        spans = []
        on = pv >= 0.12
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                spans.append((float(centers[i]) - 1.0, float(centers[j - 1]) + 1.0))
                i = j
            else:
                i += 1
        # merge gaps < 6 s
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] < 6:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        return {"frac": round(frac, 2), "kinds": kinds, "spans": [(max(0, round(a, 1)), round(b, 1)) for a, b in merged],
                "device": dev, "method": "tagger"}
    except Exception as e:
        notes.append(f"vocals: tagger unavailable ({type(e).__name__}: {str(e)[:60]})")
        return None


def analyze_vocals_stem(path, notes):
    """Vocal-stem energy (24 Oct 2026; replaces the PANNs tagger, which called Tolerate It 'probably instrumental').
    Uses the Demucs stems cached by singer.py (~/.cache/raven-stems/<hash>/); if they are not there yet it
    separates them now (~30 s on the GPU) so singer.py reuses them. Returns None if stems cannot be had."""
    try:
        import singer
        sd, cached = singer.get_stems(path, {})
        share, act_s = singer.vocal_presence(sd)
        y, sr = singer.load_mono(os.path.join(sd, "vocals.wav"))
        hop = sr // 2
        n = len(y) // hop
        rms = np.sqrt((y[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
        db = 20 * np.log10(rms + 1e-9)
        ref = float(np.percentile(db, 90))
        # active = within 22 dB of the voice's own loud level AND above an absolute floor (so bleed in an
        # instrumental never counts as singing)
        on = db > max(ref - 22.0, -42.0)
        # close pauses < 1.5 s (breaths), drop blips < 1 s
        i = 0
        runs = []
        while i < len(on):
            if on[i]:
                j = i
                while j < len(on) and on[j]:
                    j += 1
                runs.append([i, j])
                i = j
            else:
                i += 1
        merged = []
        for a, b in runs:
            if merged and (a - merged[-1][1]) * 0.5 < 1.5:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        merged = [m for m in merged if (m[1] - m[0]) * 0.5 >= 1.0]
        act = sum((b - a) * 0.5 for a, b in merged)
        dur = len(y) / sr
        spans = []
        for a, b in merged:
            a, b = a * 0.5, b * 0.5
            if spans and a - spans[-1][1] < 6:
                spans[-1][1] = b
            else:
                spans.append([a, b])
        instrumental = share < 0.02 or act < 8
        return {"frac": round(act / max(dur, 1e-9), 2), "kinds": [], "share": round(share, 3),
                "spans": [] if instrumental else [(round(a, 1), round(b, 1)) for a, b in spans],
                "instrumental": instrumental, "method": "vocal stem", "active_s": round(act, 1),
                "stems_cached": cached}
    except Exception as e:
        notes.append(f"vocals: stem method unavailable ({type(e).__name__}: {str(e)[:70]}); fell back to the tagger")
        return None


@ST.stage('vocal quality')
def analyze_vocal_quality(path, secs, voc, notes):
    """VOCAL line: breathiness per section of the sung voice, measured on the cached Demucs vocal stem with the same
    code as the speech QUALITY line (raven_ears_plus.analyze_quality). Nothing for instrumentals or no sections."""
    if not secs or not voc or voc.get("instrumental") or not voc.get("spans") or voc.get("method") != "vocal stem":
        return None
    try:
        import singer
        import raven_ears_plus as E
        sd, _ = singer.get_stems(path, {})
        y16, _ = singer.load_mono(os.path.join(sd, "vocals.wav"), E.SR)
        bounds = [secs[0]["start"]] + [s["end"] for s in secs]
        return E.sung_quality(y16, voc["spans"], bounds, notes)
    except Exception as e:
        notes.append(f"vocal quality skipped ({type(e).__name__}: {str(e)[:70]})")
        return None


@ST.stage('vocals')
def analyze_vocals(path, notes):
    """Vocal-stem energy first; the PANNs tagger only if stems cannot be had."""
    v = analyze_vocals_stem(path, notes)
    if v is not None:
        try:
            import raven_ears_plus as E
            E.release_models()
            import singer
            singer._free_gpu()
        except Exception:
            pass
        return v
    return analyze_vocals_tagger(path, notes)


def get_lyrics(path, notes):
    """Shared, cached lyrics (lyrics_shared.py): Whisper large-v3, no VAD, gap re-listen, end-of-track
    hallucinations dropped, computed once per file and reused by singer.py and harmony.py."""
    from lyrics_shared import get_words
    return get_words(path, notes)


@ST.stage('meter')
def meter_info_for(path):
    """(claim, data): the meter the METER block commits to for this file (claim None if it does not commit or cannot run)
    and the METER numbers behind it. Same cached answer singer.py prints, so the TEMPO line cannot contradict it."""
    try:
        import singer
        import soundfile as sf
        sd, _ = singer.get_stems(path, {})
        dur = sf.info(os.path.join(sd, "vocals.wav")).duration
        data = singer.meter_sections(path, sd, dur)[1]
        return singer.meter_claim(data), data
    except Exception as e:
        ST.mark("meter", "failed", f"{type(e).__name__}: {str(e)[:60]}")
        _log(f"meter claim unavailable ({type(e).__name__}: {str(e)[:60]})")
        return None, None


def meter_claim_for(path):
    return meter_info_for(path)[0]


@ST.stage('key')
def key_claim_for(path, features=None):
    """The ONE key claim for the card, decided in harmony.py (chord evidence against the spectral reading). None if it cannot run."""
    try:
        import harmony
        return harmony.key_claim_for(path, features=features)
    except Exception as e:
        ST.mark("key", "failed", f"{type(e).__name__}: {str(e)[:60]}")
        _log(f"key claim unavailable ({type(e).__name__}: {str(e)[:60]})")
        return None


HALF_TIME_BPM = 140.0       # a tracked pulse above this, sitting in thirds, may be the double of the felt beat
HALF_TIME_LEAN = 0.40       # share of beats whose between-beat onsets lean to thirds


def half_time_note(bpm, meter_data, claim=None):
    """When the tracker's pulse is fast and the beats split in thirds (12/8, shuffle), the felt beat is probably half of it.
    Says so beside the number instead of changing the number. None otherwise, and none when the METER block committed to an
    odd meter (5/4, 7/8, 10/8...): thirds there are the grouping, not a 12/8 feel."""
    if not meter_data or bpm <= HALF_TIME_BPM:
        return None
    if claim:
        try:
            if any(int(c.split("/")[0]) not in (2, 3, 4, 6, 9, 12) for c in claim.split(" then ")):
                return None
        except ValueError:
            return None
    parts = [x.get("data", x) for x in (meter_data.get("sections") or [])] or [meter_data]
    lean = max([p.get("split_votes_ternary") or 0.0 for p in parts] or [0.0])
    if lean < HALF_TIME_LEAN:
        return None
    return f"~{bpm:g} is the tracked pulse; felt beat likely ~{bpm / 2:.0f} ({round(lean * 100)}% of beats lean to thirds, a 12/8-type feel)"


def lyric_lines(words, gap=1.2, maxw=14):
    lines, cur = [], []
    for w, s, e in words:
        if cur and (s - cur[-1][2] > gap or len(cur) >= maxw):
            lines.append(cur)
            cur = []
        cur.append((w, s, e))
    if cur:
        lines.append(cur)
    try:
        from lyrics_shared import UNSURE
    except Exception:
        UNSURE = []

    def doubt(l):
        for x in UNSURE:
            if x[0] - 0.3 <= l[0][1] <= x[1] + 0.3:
                return f" (unsure: {x[2]})" if len(x) > 2 else " (unsure)"
        return ""
    return [{"t": round(l[0][1], 1), "text": " ".join(w for w, _, _ in l) + doubt(l)} for l in lines]


# --------------------------------------------------------------------------- card
def shape_phrase(secs):
    if not secs:
        return ""
    parts = []
    for i, s in enumerate(secs):
        lab = s["level"]
        if i == 0 and lab != "loud":
            lab += " intro"
        elif i == len(secs) - 1 and len(secs) > 2:
            lab += " outro" if lab != "loud" else " finish"
        if s.get("peak"):
            lab = "PEAK " + lab
        parts.append(f"{lab} {_t(s['start'])}" if i else lab)
    return " → ".join(parts)


def format_card(r, meter_data=None):
    L = []
    L.append(f"MUSIC: {r['file']} · {_t(r['duration'])}")
    k = r.get("key")
    kc = r.get("key_claim")
    if kc:
        # one key per card: harmony.py decides (chords against the spectral reading); every other line quotes it
        L.append("  KEY:      " + kc["line"])
        if kc.get("votes_line"):
            L.append("  KEY VOTES: " + kc["votes_line"])
    elif k:
        bits = [k["key"], f"r={k['r']} (margin {k['margin']})"]
        if k.get("also"):
            bits.append(f"or {k['also']}")
        L.append("  KEY:      " + " · ".join(bits))
        for c in k.get("changes", []):
            if not c.get("suppressed"):
                L.append(f"  KEY CHG:  {_t(c['at'])} → {c['to']}")
    t = r.get("tempo")
    if t and t.get("lead"):
        # Steady beat_this bars lead; name librosa only when it differs substantially.
        L.append("  TEMPO:    " + t["lead"])
    elif t:
        bits = [f"{t['bpm']:g} BPM", f"steady {t['regularity']:.2f}", f"beat strength {t['beat_strength']:.2f}x"]
        # one meter per card: the METER block's answer (singer.meter_sections), never this tracker's own 3-vs-4 guess
        if r.get("meter_claim"):
            bits.append(f"meter {r['meter_claim']}, see METER")
        if t.get("half_time"):
            bits.append(t["half_time"])
        elif t.get("note"):
            bits.append(t["note"])
        import tempo_trust
        L.append("  TEMPO:    " + tempo_trust.estimate_line(" · ".join(bits), meter_data))
    secs = r.get("sections")
    if secs:
        L.append("  SHAPE:    " + shape_phrase(secs))
        for s in secs:
            tag = "  <- PEAK" if s.get("peak") else ""
            L.append(f"    {_t(s['start'])}–{_t(s['end'])}  {s['level']:<5} {s['db']:>6.1f} dB{tag}")
    s = r.get("sound")
    if s:
        bits = [f"{s['tone']} (centroid {s['centroid_hz']} Hz)"]
        if "dyn_range_db" in s:
            bits.append(f"dynamic range {s['dyn_range_db']} dB")
        bits.append(f"{round(s['harmonic_frac'] * 100)}% harmonic / {round((1 - s['harmonic_frac']) * 100)}% percussive")
        bits.append(f"peak {s['peak_dbfs']} dBFS")
        L.append("  SOUND:    " + " · ".join(bits))
    v = r.get("vocals")
    if v:
        if v.get("method") == "vocal stem":
            if v.get("instrumental"):
                L.append(f"  VOCALS:   instrumental (the vocal stem holds {v['share'] * 100:.1f}% of the energy, "
                         f"{v['active_s']:.0f} s above voice level) [method: vocal-stem energy]")
            else:
                sp = ", ".join(_span(a, b) for a, b in v["spans"][:4])
                more = f" +{len(v['spans']) - 4} more" if len(v["spans"]) > 4 else ""
                L.append(f"  VOCALS:   yes, voice present ~{round(v['frac'] * 100)}% of the track "
                         f"(stem holds {v['share'] * 100:.0f}% of the energy) [method: vocal-stem energy] · {sp}{more}")
        elif v["frac"] >= 0.15:
            sp = ", ".join(_span(a, b) for a, b in v["spans"][:4])
            more = f" +{len(v['spans']) - 4} more" if len(v["spans"]) > 4 else ""
            kind = f" ({', '.join(v['kinds'])})" if v["kinds"] else ""
            L.append(f"  VOCALS:   probably yes{kind}, ~{round(v['frac'] * 100)}% of the track [method: tagger FALLBACK, rough] · {sp}{more}")
        else:
            L.append(f"  VOCALS:   probably instrumental (~{round(v['frac'] * 100)}% vocal-like) [method: tagger FALLBACK, rough]")
    try:
        import raven_ears_plus as E
        vq = E.sung_quality_line(r.get("vocal_quality"))
    except Exception:
        vq = None
    if vq:
        L.append(vq)
    lyr = r.get("lyrics")
    if lyr:
        if r.get("lyrics_align"):
            L.append("  " + r["lyrics_align"])
        else:
            model = next((n.split()[2] for n in r.get("notes", [])
                          if n.startswith("lyrics: whisper ")), "large-v3")
            L.append(f"  LYRICS:   {len(lyr)} lines (whisper {model}, rough times)")
        for l in lyr:
            L.append(f"    [{_t(l['t'])}{'≈' if l.get('approx') else ''}] {l['text']}"
                         + (" (no voice measured here)" if l.get("no_voice") else ""))
        # MISHEARD: only against the writer's own words (--lyrics), never a database's text.
        if (r.get("lyrics_source") or {}).get("status") == "file":
            import lyric_align
            L += ["  " + x for x in lyric_align.misheard_block(lyr, _t, width=lyric_align.MAX_LINE - 2)]
    elif r.get("lyrics_align"):             # LRCLIB hook: failures/instrumental are visible without lyric rows.
        L.append("  " + r["lyrics_align"])
    for gap in r.get("lyric_gaps", []):
        L.append(f"  NOTE:     {_t(gap['start'])}–{_t(gap['end'])}: {gap['description']}")
    for n in r.get("notes", []):
        L.append(f"  NOTE:     {n}")
    from ears_handoff import finish_lines  # MORE EARS: leave the feeling to the listener.
    import instrumental_voice
    text = ST.cnn_key_text("\n".join(ST.add_notes(L, r.get("stage_status", ST.from_notes(r.get("notes", []), "music")))), r)
    import song_length
    return song_length.annotate(instrumental_voice.stem_text(text, instrumental_voice.designated(r)), r.get("length_check"))


@ST.engine
@ST.cli
def main(argv=None, capture=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--_picture-data", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-lyrics", action="store_true")
    ap.add_argument("--lyrics", metavar="FILE", help="the song's real words, one sung line per line; timed against the audio")
    ap.add_argument("--names", metavar="FILE", help="names/rare words for Whisper, one per line (default RAVEN_EARS_HOME/names.txt)")
    import lrclib                          # LRCLIB hook: metadata lookup, independently switchable.
    import credits                        # HONEST CARD: same metadata flags as the other engines.
    credits.add_args(ap)
    lrclib.add_args(ap)
    import song_extras
    song_extras.add_args(ap)
    a = ap.parse_args(argv)
    from network_policy import local_audio_path
    local_audio_path(a.file, ap)
    extras = song_extras.prepare(a, a.file)
    lrclib.configure_from_args(a)
    credits.configure_from_args(a)
    lyric_given = None
    if a.lyrics:
        import lyric_align
        lyric_given = lyric_align.read_lyrics_file(a.lyrics)
    import lyrics_shared
    lyrics_shared.configure_for_run(a.names, lyric_given)
    import librosa
    import raven_ears_plus as E
    notes = []
    y = E.load_audio(a.file, SR)
    dur = len(y) / SR
    if dur < 5:
        ST.mark("music analysis", "skipped", "audio shorter than 5 s")
        if capture is not None:
            from ears_handoff import finish_json
            capture(finish_json({"duration": dur, "stage_status": ST.snapshot()}))
        if a.json:
            from ears_handoff import finish_json
            print(json.dumps(finish_json({"stage_status": ST.snapshot()})))
        else:
            print("\n".join(ST.add_notes(["audio too short for music analysis (<5 s)"], ST.snapshot())))
        return
    _log(f"{dur:.0f}s decoded")
    import harmony
    ST.progress("music features and tempo")
    F = harmony.music_features(a.file, y)
    chroma, rms_db = F["chroma"], F["rms_db"]  # raw RMS; harmony clips its own copy later
    r = {"file": os.path.basename(a.file), "duration": round(dur, 1)}
    r["key"] = analyze_key(chroma, dur)
    t = analyze_tempo(None, y, oenv=F["oenv"])
    if t:
        t.pop("_beats", None)
        r["tempo"] = t
    r["sections"] = analyze_sections(y, chroma, rms_db, dur, mfcc=F["mfcc"])
    r["sound"] = analyze_sound(y, rms_db, F["hp_energy"], F["cent"])
    ST.progress("sound and vocals")
    r["vocals"] = analyze_vocals(a.file, notes)
    r["vocal_quality"] = analyze_vocal_quality(a.file, r.get("sections"), r.get("vocals"), notes)
    r["meter_claim"], md = meter_info_for(a.file)
    import tempo_trust
    reading = tempo_trust.lead_reading((r.get("tempo") or {}).get("bpm"), md, r.get("meter_claim"))
    if reading:
        r.setdefault("tempo", {})["lead"] = tempo_trust.lead_line(reading)   # bpm stays librosa's own number
    elif r.get("tempo"):
        hn = half_time_note(r["tempo"]["bpm"], md, r.get("meter_claim"))
        if hn:
            r["tempo"]["half_time"] = hn
    ST.progress("home key")
    r["key_claim"] = key_claim_for(a.file, F)
    if not a.no_lyrics:
        ST.progress("lyrics on the music card")
        # LRCLIB hook: file → synced → plain/aligned → Whisper, with provenance even on failure.
        lyr = lrclib.for_music(a.file, notes, dur, lyric_given)
        r.update(lyrics=lyr["lyrics"], lyrics_align=lyr["label"], lyrics_source=lyr["lookup"])
        import jev_more
        if (r.get("lyrics_source") or {}).get("status") == "file":
            jev_more.add_misheard(r.get("lyrics"), notes)
        activity = (r.get("vocals") or {}).get("spans")
        r["lyric_gaps"] = [dict(start=a, end=b, description=lyrics_shared.gap_description(
            a, b, (activity,), words_available=lyr["words"] is not None))
            for a, b in ([] if lyr.get("quick_skipped") else lyrics_shared.word_gaps(lyr["words"], dur))]
    else:
        ST.mark("words", "skipped", "--no-lyrics", kind="input")
        ST.mark("LRCLIB", "skipped", "--no-lyrics", kind="input")
    credits.save_filename_reading(a.file, r, notes)
    import song_length
    song_length.capture(a.file, r)
    ST.collect(ST.engine_notes([n for n in notes if not n.startswith("Jev")], "music"))
    r["stage_status"] = ST.snapshot()
    r = {k: v for k, v in r.items() if v is not None}
    if notes:
        r["notes"] = notes
    r["listener"] = "and now, you"  # Listener hook for the JSON card.
    if extras:
        r["song_extras"] = extras
    if capture is not None:
        from ears_handoff import finish_json
        captured = dict(r)
        if a._picture_data:
            captured["listening_sheet"] = {"step_s": HOP / SR, "rms_db": rms_db.tolist()}
        capture(finish_json(captured))
    if a.json:
        from ears_handoff import finish_json  # MORE EARS: handoff inside valid JSON.
        print(json.dumps(finish_json(r), indent=2, default=float))
    else:
        print(song_extras.decorate(format_card(r, md), extras))


if __name__ == "__main__":
    sys.exit(main())
