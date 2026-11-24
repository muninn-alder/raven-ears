#!/usr/bin/env python
"""Music Ears: HARMONY (24 Oct 2026). `harmony.py <file> [--json] [--no-lyrics] [--no-cache]`

Hears a song as a SITUATION, not a verdict: chords as roman numerals against the home key, a tension
curve with its builds and releases, one chronological timeline of stage directions, words-vs-music
contrast per lyric line, the places where the song breaks its own pattern, and the Sloboda chill
candidates. Models run locally (librosa, scikit-style numpy, the same Whisper large-v3 as music.py, a small
cardiffnlp sentiment model on CPU). Credits and LRCLIB can receive song metadata (independent off switches).
Reuses music.py for key, tempo, sections, sound and lyrics.

Rules: no verdict words; mode is a FACT, never a mood term; a spectral word
never stands in for a feeling (BODY 150-500 Hz is reported apart from BRIGHTNESS).
"""
from __future__ import annotations

import ears_profile as EP
from network_policy import offline

import argparse
import hashlib
import json
import os
import pickle
import sys
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import music as M  # noqa: E402  (reuse: key, tempo, sections, sound, lyrics)

SR, HOP = M.SR, M.HOP
FPS = SR / HOP
_t = M._t
PCN = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
KEYN = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
KEYN.update({"Db": 1, "D#": 3, "G#": 8})


import stage_status as ST  # HONEST CARD: run-local outcomes.
import key_opening
import jev_reads
from jev_reads import flag_contrast

def log(m):
    print(f"[harmony] {m}", file=sys.stderr)


def tt(s):
    """m:ss.s style when it matters"""
    return _t(s)


# --------------------------------------------------------------------------- features (cached)
def cache_path(path):
    st = os.stat(path)
    h = hashlib.md5(f"a0|{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:12]
    d = os.path.expanduser(os.environ.get("RAVEN_HARMONY_CACHE", "~/.cache/harmony"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{os.path.basename(path)}-{h}")


@ST.stage('features')
def compute_features(path, shared=None):
    return shared if shared is not None else _compute_features(path)


def music_features(path, y):
    """Music needs the same raw pack. Defer feature status/persistence to its key stage, as before."""
    cp = cache_path(path) + ".feat.pkl"
    if os.path.exists(cp):
        F = pickle.load(open(cp, "rb"))
        if "hp_energy" in F:
            return F
    F = _compute_features(path, y)
    F["stage_status"] = [r for r in ST.snapshot() if r["stage"] == "decode"]
    return F


def _compute_features(path, y=None):
    import librosa
    import raven_ears_plus as E
    if y is None:
        y = E.load_audio(path, SR)
    log(f"{len(y) / SR:.0f}s decoded; HPSS")
    S = librosa.stft(y, hop_length=HOP)
    A = np.abs(S)
    H, P = librosa.decompose.hpss(S)
    hp_energy = float(np.sum(np.abs(H) ** 2)), float(np.sum(np.abs(P) ** 2))
    yh = librosa.istft(H, hop_length=HOP, length=len(y))
    yp = librosa.istft(P, hop_length=HOP, length=len(y))
    del S, H, P
    log("chroma / bass / timbre")
    chroma = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=HOP)
    bass = librosa.feature.chroma_cqt(y=yh, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C1"),
                                      n_octaves=3, bins_per_octave=36)
    rms = librosa.feature.rms(S=A, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    cent = librosa.feature.spectral_centroid(S=A, sr=SR)[0]
    oenv = librosa.onset.onset_strength(y=yp, sr=SR, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR)
    pw = A ** 2
    tot = pw.sum() + 1e-12
    body = float(pw[(freqs >= 150) & (freqs < 500)].sum() / tot)
    top = float(pw[freqs >= 4000].sum() / tot)
    return {"y_len": len(y), "chroma": chroma, "bass": bass, "rms_db": rms_db, "mfcc": mfcc, "cent": cent,
            "oenv": oenv, "body": body, "top": top, "y": y, "hp_energy": hp_energy}


def vocal_stem_path(path):
    """The cached Demucs vocals stem for this file (singer.py / music.py make it), or None."""
    try:
        import singer
        p = os.path.join(singer.CACHE, singer.file_hash(path), "vocals.wav")
        return p if os.path.exists(p) else None
    except Exception:
        return None


BAND_OUT_MIN_BARS = 2.0
BAND_OUT_EDGE_S = 5.0
BAND_OUT_ACTIVITY_DB = -12.0    # mean mix dB relative to the song's median frame dB


def band_out_candidates(states, present, hop, dur, rms_db, downbeats=None, bpm=0):
    """Audit every bass/other-off run, including ineligible or quiet runs.

    States come only from singer.stem_states. Split at voice/drum silence so
    a silent tail cannot erase an otherwise eligible active stretch.
    """
    n = min((len(st) for st in states.values()), default=0)
    off_state = np.zeros(n, bool)
    off = ~states.get("bass", off_state) & ~states.get("other", off_state)
    voice, drums = states.get("vocals", off_state), states.get("drums", off_state)
    active = voice | drums
    db = np.asarray(downbeats if downbeats is not None else [], dtype=float)
    db = np.unique(db[np.isfinite(db)])
    bar_s = 240.0 / bpm if bpm > 0 else None
    source = "tracked/estimated downbeat intervals" if len(db) >= 2 else "beat pulse times assumed four beats"

    def position(t):
        i = int(np.clip(np.searchsorted(db, t, side="right") - 1, 0, len(db) - 2))
        return i + (t - db[i]) / (db[i + 1] - db[i])

    # Retain silent portions too, with a rejection reason.
    cuts = np.flatnonzero(np.r_[True, (off[1:] != off[:-1]) | (active[1:] != active[:-1]), True]) if n else []
    rows = []
    median = float(np.median(rms_db))
    for i, j in zip(cuts[:-1], cuts[1:]):
        if not off[i]:
            continue
        a, b = float(i * hop), min(float(j * hop), dur)
        bars = position(b) - position(a) if len(db) >= 2 else ((b - a) / bar_s if bar_s else None)
        seg = rms_db[int(a * FPS):max(int(b * FPS), int(a * FPS) + 1)]
        level = float(np.mean(seg) - median) if len(seg) else None
        reasons = []
        if not all(present.get(s, False) and np.any(states.get(s, [])) for s in ("bass", "other")):
            reasons.append("harmonic stem absent or never on")
        if not active[i]:
            reasons.append("neither voice nor drums on")
        if bars is None or bars < BAND_OUT_MIN_BARS:
            reasons.append("bar duration unavailable" if bars is None else "under two bars")
        if a < BAND_OUT_EDGE_S or b > dur - BAND_OUT_EDGE_S:
            reasons.append("overlaps first/last five seconds")
        if level is None or level < BAND_OUT_ACTIVITY_DB:
            reasons.append("mix level unavailable" if level is None else "quiet relative to song median")
        support = [name + ("" if np.all(st[i:j]) else " at times")
                   for name, st in (("drums", drums), ("voice", voice)) if np.any(st[i:j])]
        rows.append(dict(t0=a, t1=b, bars=None if bars is None else float(bars), bars_source=source,
                         level_vs_song_median_db=level, support=support,
                         voice_throughout=bool(np.all(voice[i:j])), accepted=not reasons,
                         rejection_reasons=reasons))
    return rows


@ST.stage('band presence')
def band_presence(path, dur, rms_db, downbeats, bpm, notes):
    """Read singer's four-stem cache across pass processes. Never run Demucs."""
    import singer
    from lyrics_shared import vocal_onset
    try:
        directory = os.path.join(singer.CACHE, singer.file_hash(path))
    except OSError:
        directory = None
    if directory is None or not all(os.path.isfile(os.path.join(directory, s + ".wav")) for s in singer.SRC):
        reason = "band-presence check skipped: no cached stems available"
        notes.append(reason)
        ST.mark("band presence", "skipped", reason)
        return dict(status="skipped", reason=reason, candidates=[])
    try:
        _, hop, ref, present, states = singer.stem_states(directory)
        rows = band_out_candidates(states, present, hop, dur, rms_db, downbeats, bpm)
        transitions = singer.stem_transitions(states, hop)
        vocal_spans = [(a * hop, min(b * hop, dur)) for a, b in singer._runs(states.get("vocals", ()))]
        return dict(status="checked", references=ref, present=present, candidates=rows,
                    vocal_onset=vocal_onset(transitions), vocal_spans=vocal_spans,
                    thresholds=dict(min_bars=BAND_OUT_MIN_BARS, edge_s=BAND_OUT_EDGE_S,
                                    activity_db=BAND_OUT_ACTIVITY_DB))
    except Exception as e:
        reason = f"band-presence check skipped: cached stems unreadable ({type(e).__name__})"
        notes.append(reason)
        ST.mark("band presence", "failed", reason)
        return dict(status="skipped", reason=reason, candidates=[])


def band_out_stretches(R):
    return [c for c in R.get("band_presence", {}).get("candidates", []) if c["accepted"]]


def band_out_line(c):
    support = "drums or voice" if any("at times" in s for s in c["support"]) else " + ".join(c["support"])
    reading = "chords from voice alone" if c["voice_throughout"] else "chord labels ungrounded"
    return (f"no harmonic instruments {tt(c['t0'])}–{tt(c['t1'])} (~{c['bars']:.0f} bars): "
            f"bass + other out; {support}; {reading}")


def mark_band_out(events, stretches):
    """Keep chord labels in JSON; split at band boundaries to exclude only affected time."""
    out = []
    for e in events:
        cuts = sorted({e["t0"], e["t1"]} | {t for c in stretches for t in (c["t0"], c["t1"])
                                                    if e["t0"] < t < e["t1"]})
        for a, b in zip(cuts[:-1], cuts[1:]):
            band_out = any(c["t0"] <= a and b <= c["t1"] for c in stretches)
            out.append(dict(e, t0=a, t1=b, band_out=band_out))
    return out


def load_vocal_22k(vp):
    import singer
    y, sr = singer.load_mono(vp, SR)
    return y.astype(np.float32)


def compute_melody(y, source="full mix"):
    import librosa
    log(f"melody (pyin on the {source}" + (", approximate)" if source == "full mix" else ")"))
    hop = 1024
    fmin = librosa.note_to_hz("C3") if source == "full mix" else librosa.note_to_hz("G2")
    f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=librosa.note_to_hz("C6"), sr=SR,
                              frame_length=2048, hop_length=hop)
    return {"f0": f0, "vp": vp, "hop": hop, "source": source}


# --------------------------------------------------------------------------- chords
QUAL = {  # name: (intervals, weights, suffix-class)
    "maj": ([0, 4, 7], [1.0, 0.85, 0.85]),
    "min": ([0, 3, 7], [1.0, 0.85, 0.85]),
    "dom7": ([0, 4, 7, 10], [1.0, 0.85, 0.85, 0.5]),
    "maj7": ([0, 4, 7, 11], [1.0, 0.85, 0.85, 0.5]),
    "min7": ([0, 3, 7, 10], [1.0, 0.85, 0.85, 0.5]),
    "dim": ([0, 3, 6], [1.0, 0.8, 0.8]),
    "5": ([0, 7], [1.0, 0.85]),
}
QNAMES = [q for q in QUAL if q != "5"]  # fifth-only readings need direct pitch evidence
STATES = [(r, q) for q in QNAMES for r in range(12)]


def cls(q):
    return {"maj": "M", "dom7": "M", "maj7": "M", "min": "m", "min7": "m", "dim": "d", "5": "5"}[q]


def chord_tones(root, q):
    return {(root + i) % 12 for i in QUAL[q][0]}


def chord_name(root, q):
    s = PCN[root]
    return s + {"maj": "", "min": "m", "dom7": "7", "maj7": "maj7", "min7": "m7", "dim": "°", "5": "5"}[q]


def parse_key(name):
    t, mode = name.split()
    return KEYN[t], mode


def key_label(k):
    if k[1] in ("mixolydian", "blues-rock"):
        from keyfuse import spelling_key
        return spelling_key(f"{M.NAMES[k[0]]} mixolydian").removesuffix("Mix") + f" {k[1]}"
    return f"{M.NAMES[k[0]]} {k[1]}"          # one spelling for keys everywhere (F# minor, not G♭ minor)


DIA = {"major": {(0, "M"), (2, "m"), (4, "m"), (5, "M"), (7, "M"), (9, "m"), (11, "d")},
       "minor": {(0, "m"), (2, "d"), (3, "M"), (5, "m"), (7, "m"), (7, "M"), (8, "M"), (10, "M")}}
FUNC = {"major": {0: 0.0, 9: 0.25, 5: 0.3, 4: 0.4, 2: 0.45, 7: 0.7, 11: 0.8},
        "minor": {0: 0.0, 3: 0.2, 8: 0.3, 10: 0.35, 5: 0.4, 7: 0.7, 2: 0.8},
        "mixolydian": {0: 0.0, 9: 0.25, 5: 0.3, 10: 0.35, 2: 0.45, 7: 0.7, 4: 0.8},
        "blues-rock": {0: 0.0, 9: 0.25, 5: 0.3, 10: 0.35, 3: 0.35, 2: 0.45, 7: 0.7}}
MIXOLYDIAN_SCALE = {0, 2, 4, 5, 7, 9, 10}
BLUES_PITCHES = MIXOLYDIAN_SCALE | {3}
ROM = [("", "I"), ("♭", "II"), ("", "II"), ("♭", "III"), ("", "III"), ("", "IV"), ("♯", "IV"), ("", "V"),
       ("♭", "VI"), ("", "VI"), ("♭", "VII"), ("", "VII")]


def roman(root, q, key):
    iv = (root - key[0]) % 12
    acc, num = ROM[iv]
    c = cls(q)
    if c in ("m", "d"):
        num = num.lower()
    suf = {"dom7": "7", "maj7": "maj7", "min7": "7", "dim": "°", "5": "5"}.get(q, "")
    return f"{acc}{num}{suf}"


def is_diatonic(root, q, key):
    if key[1] == "mixolydian":
        return all((tone - key[0]) % 12 in MIXOLYDIAN_SCALE for tone in chord_tones(root, q))
    if key[1] == "blues-rock":
        return ((root == key[0] and q == "maj7") or
                all((tone - key[0]) % 12 in BLUES_PITCHES for tone in chord_tones(root, q)))
    if q == "5":
        scale = {0, 2, 4, 5, 7, 9, 11} if key[1] == "major" else {0, 2, 3, 5, 7, 8, 10}
        return all((tone - key[0]) % 12 in scale for tone in chord_tones(root, q))
    return ((root - key[0]) % 12, cls(q)) in DIA[key[1]]


def func_tension(root, q, key):
    iv = (root - key[0]) % 12
    base = FUNC[key[1]].get(iv, 0.8) if is_diatonic(root, q, key) else 0.8
    if q == "dom7":
        base += 0.1
    return min(base, 1.0)


def key_spans(kinfo, glob):
    """[(t_start, (tonic, mode))] from music.py's key + changes."""
    sp = [(0.0, glob)]
    for c in key_opening.settled(kinfo, glob, parse_key).get("changes", []):
        sp.append((float(c["at"]), parse_key(c["to"])))
    return sp


def key_at(spans, t):
    k = spans[0][1]
    for s, kk in spans:
        if t >= s:
            k = kk
    return k


# --------------------------------------------------------------------------- bass stem (root evidence for the chords, tonic evidence for the key)
def bass_stem_path(path):
    """The cached Demucs bass stem (six-stem folder first, then the four-stem one), or None. Never separates anything itself."""
    try:
        import singer
        for d in (singer.stems6_dir(path), os.path.join(singer.CACHE, singer.file_hash(path))):
            p = os.path.join(d, "bass.wav")
            if os.path.exists(p):
                return p
    except Exception:
        pass
    return None


@ST.stage('bass pitch')
def bass_track(path, cp, use_cache=True):
    """Per-frame bass pitch class from the bass stem (pyin, 33-250 Hz): {"pc": int8[-1 = none], "rel": level / loud-bass level}.
    Same frame grid as the features (SR, HOP). Cached; None when there is no bass stem."""
    bp = cp + ".bassf0.pkl"
    if use_cache and os.path.exists(bp):
        return pickle.load(open(bp, "rb"))
    sp = bass_stem_path(path)
    if not sp:
        ST.mark("bass pitch", "skipped", "no cached bass stem")
        return None
    import librosa
    import singer
    y, _ = singer.load_mono(sp, SR)
    y = y.astype(np.float32)
    log("bass stem pitch (pyin)")
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("B3"), sr=SR,
                            frame_length=4096, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    pc = np.full(n, -1, dtype=np.int8)
    ok = ~np.isnan(f0[:n])
    pc[ok] = (np.round(12 * np.log2(f0[:n][ok] / 440.0) + 69).astype(int) % 12).astype(np.int8)
    loud = float(np.percentile(rms[:n], 95)) or 1.0
    out = {"pc": pc, "rel": (rms[:n] / loud).astype(np.float32)}
    if use_cache:
        pickle.dump(out, open(bp, "wb"))
    return out


BASS_MIN_REL = 0.1       # a bass frame this much quieter than the stem's loud part is leakage, not a note
BASS_CONF = 0.45         # share of a chord unit's voiced bass frames on one pitch class for the bass to count as "sitting" on it
BASS_PEN_OUT, BASS_PEN_INV = 0.50, 0.10   # score taken off a chord that lacks the bass note / has it as an inversion (root elsewhere)


def unit_bass(bt, units):
    """For each chord unit (a, b): (dominant bass pc or -1, confidence 0..1). Confidence = (voiced and loud share of the
    unit) x (share of those frames on the dominant pc); below BASS_CONF-ish it is not trusted by the caller."""
    out = []
    pc, rel = bt["pc"], bt["rel"]
    for a, b in units:
        fa, fb = int(a * FPS), max(int(b * FPS), int(a * FPS) + 1)
        p, r = pc[fa:fb], rel[fa:fb]
        ok = (p >= 0) & (r >= BASS_MIN_REL)
        if len(p) == 0 or ok.sum() < 3:
            out.append((-1, 0.0))
            continue
        cnt = np.bincount(p[ok], minlength=12)
        dom = int(cnt.argmax())
        out.append((dom, float(ok.mean() * cnt[dom] / ok.sum())))
    return out


def loop_period(U, lo=3, hi=40):
    """Length (in units) of the repeating chord loop, from how similar unit i is to unit i+L; None when nothing repeats."""
    n = len(U)
    if n < 3 * lo:
        return None
    S = U @ U.T
    hi = min(hi, n // 3)
    if hi < lo:
        return None
    m = {L: float(np.mean(np.diag(S, L))) for L in range(lo, hi + 1)}
    best = max(m.values())
    if best < 0.8:
        return None
    # the shortest lag that is almost as good (a loop of 4 also repeats at 8, 12 ...)
    return next(L for L in range(lo, hi + 1) if m[L] >= best - 0.01)


def loop_vote(U, logp, period, reps=3, gate=0.92):
    """Majority vote across repeats of a recurring loop: each unit's chord log-posterior is averaged with those of the
    units one, two, three ... loops away (both directions) whose chroma is nearly the same. One misheard repeat is
    outvoted; a consistently misheard one is not."""
    n = len(U)
    S = U @ U.T
    out = logp.copy()
    for i in range(n):
        js = [i + k * period * s for k in range(1, reps + 1) for s in (-1, 1) if 0 <= i + k * period * s < n]
        js = [j for j in js if S[i, j] >= gate]
        if js:
            out[i] = (logp[i] + logp[js].sum(axis=0)) / (1 + len(js))
    return out


@ST.stage('chords')
def decode_chords(F, beats_fr, glob_key, dur, bass_pen=True, loop=True):
    import librosa
    chroma, bass, rms_db = F["chroma"], F["bass"], F["rms_db"]
    n = chroma.shape[1]
    # units of >= ~0.6 s, aligned to beats when we have them
    if beats_fr is not None and len(beats_fr) > 8:
        bt = [0.0] + [b / FPS for b in beats_fr]
    else:
        bt = list(np.arange(0, dur, 0.5))
    edges = [0.0]
    for b in bt[1:]:
        if b - edges[-1] >= 0.6:
            edges.append(b)
    if dur - edges[-1] < 0.4:
        edges[-1] = dur
    elif edges[-1] < dur:
        edges.append(dur)
    tpl = np.zeros((len(STATES), 12))
    for i, (r, q) in enumerate(STATES):
        iv, w = QUAL[q]
        for a, b in zip(iv, w):
            tpl[i, (r + a) % 12] = b
    tn = tpl / np.linalg.norm(tpl, axis=1, keepdims=True)
    gate = rms_db.max() - 45
    units, U, B, ok = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        fa, fb = int(a * FPS), max(int(b * FPS), int(a * FPS) + 1)
        c = chroma[:, fa:fb].mean(axis=1)
        bs = bass[:, fa:fb].mean(axis=1)
        # bass chroma is 36 bins/oct squashed to 12 by librosa, so fine
        units.append((a, b))
        U.append(c / (np.linalg.norm(c) + 1e-9))
        B.append(bs / (bs.max() + 1e-9))
        ok.append(rms_db[fa:fb].mean() > gate)
    U, B = np.array(U), np.array(B)
    sc = U @ tn.T
    # bass bonus on the root, small prior toward diatonic chords, small cost on sevenths
    root_ix = np.array([r for r, _ in STATES])
    sc = sc + 0.10 * B[:, root_ix]
    dia = np.array([1.0 if is_diatonic(r, q, glob_key) else 0.0 for r, q in STATES])
    sev = np.array([1.0 if q in ("dom7", "maj7", "min7") else 0.0 for r, q in STATES])
    dm = np.array([1.0 if q == "dim" else 0.0 for r, q in STATES])
    sc = sc + 0.03 * dia - 0.05 * sev - 0.06 * dm
    # the bass STEM (not the chroma's low bins) as a root constraint: where it confidently sits on one pitch class, a chord
    # without that note is penalised, and one that has it but not as its root (an inversion) a little
    bt = F.get("bass_trk") if bass_pen else None
    if bt is not None:
        ub = unit_bass(bt, units)
        pen = np.zeros((12, len(STATES)))
        for b in range(12):
            for i, (r, q) in enumerate(STATES):
                pen[b, i] = 0.0 if r == b else (BASS_PEN_INV if b in chord_tones(r, q) else BASS_PEN_OUT)
        for i, (b, c) in enumerate(ub):
            if b >= 0 and c >= BASS_CONF:
                sc[i] = sc[i] - min(1.0, c / 0.6) * pen[b]
    lp = (sc - sc.max(axis=1, keepdims=True)) * 14
    lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
    if loop:
        per = loop_period(U)
        if per:
            lp = loop_vote(U, lp, per)
            lp = lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))
    p = np.exp(lp)
    nS = len(STATES)
    trans = np.full((nS, nS), 0.25 / (nS - 1))
    np.fill_diagonal(trans, 0.75)
    path = librosa.sequence.viterbi(p.T, trans)
    ev = []
    for (a, b), s, good in zip(units, path, ok):
        if not good:
            s = -1
        if ev and ev[-1]["s"] == s:
            ev[-1]["t1"] = b
        else:
            ev.append({"t0": a, "t1": b, "s": int(s)})
    # absorb flickers < 0.9 s into the neighbour with the longer duration
    changed = True
    while changed and len(ev) > 2:
        changed = False
        for i, e in enumerate(ev):
            if e["t1"] - e["t0"] < 0.9 and e["s"] != -1:
                left = ev[i - 1] if i > 0 else None
                right = ev[i + 1] if i + 1 < len(ev) else None
                tgt = left if (left and (not right or (left["t1"] - left["t0"]) >= (right["t1"] - right["t0"]))) else right
                if tgt is None:
                    continue
                tgt["t0"] = min(tgt["t0"], e["t0"])
                tgt["t1"] = max(tgt["t1"], e["t1"])
                ev.pop(i)
                changed = True
                break
        # merge neighbours that became identical
        k = 1
        while k < len(ev):
            if ev[k]["s"] == ev[k - 1]["s"]:
                ev[k - 1]["t1"] = ev[k]["t1"]
                ev.pop(k)
            else:
                k += 1
    out = []
    for e in ev:
        if e["s"] < 0:
            out.append({"t0": e["t0"], "t1": e["t1"], "root": None, "q": None})
            continue
        r, q = STATES[e["s"]]
        # how sure the chord model was: mean posterior of the chosen chord over the units it covers
        inside = [i for i, (a, b) in enumerate(units) if a >= e["t0"] - 1e-6 and b <= e["t1"] + 1e-6]
        conf = float(np.mean([p[i, e["s"]] for i in inside])) if inside else 0.0
        fa, fb = int(e["t0"] * FPS), max(int(e["t1"] * FPS), int(e["t0"] * FPS) + 1)
        b = bass[:, fa:fb].mean(axis=1)
        out.append({"t0": e["t0"], "t1": e["t1"], "root": r, "q": q, "bass": int(np.argmax(b)),
                    "bass_share": float(b.max() / (b.sum() + 1e-9)), "conf": conf})
    return out


def refine_key(events, key):
    """music.py's krumhansl pick can land on the relative major/minor. Decide by what the chords do:
    time on the tonic chord (+0.6 x the dominant, +bonus for the final chord)."""
    tonic, mode = key
    rel = ((tonic + 9) % 12, "minor") if mode == "major" else ((tonic + 3) % 12, "major")
    def score(k):
        s = 0.0
        for e in events:
            if e["root"] is None:
                continue
            d = e["t1"] - e["t0"]
            iv = (e["root"] - k[0]) % 12
            if iv == 0 and cls(e["q"]) == ("M" if k[1] == "major" else "m"):
                s += d
            elif iv == 7:
                s += 0.6 * d
        last = [e for e in events if e["root"] is not None]
        if last and (last[-1]["root"] - k[0]) % 12 == 0:
            s += 8
        return s
    a, b = score(key), score(rel)
    return (rel, True) if b > a * 1.25 else (key, False)


def key_by_chords(events, key, cands, margin=0.08):
    """Re-score the home key by what the decoded chords do. For the spectral pick and each rival candidate key
    (relative keys share a diatonic set, so they are left to refine_key), measure the share of decoded chord-time
    that is diatonic to it; the rest is 'borrowed'. A rival wins only if it leaves clearly less time borrowed
    (>= margin of the chord-time) than the incumbent. Returns (key, swapped, table)."""
    tot = sum(e["t1"] - e["t0"] for e in events if e["root"] is not None) or 1.0

    def borrowed(k):
        return sum(e["t1"] - e["t0"] for e in events if e["root"] is not None and not is_diatonic(e["root"], e["q"], k)) / tot
    base = borrowed(key)
    table = [(key, base)]
    best, bb = key, base
    for k in cands:
        if k == key:
            continue
        b = borrowed(k)
        table.append((k, b))
        if b < bb - 1e-9:
            best, bb = k, b
    if best != key and base - bb >= margin:
        return best, True, table
    return key, False, table


KEY_STRONG_SHARE, KEY_STRONG_LEAD = 0.85, 0.15   # chord-time diatonic share and lead over the next key area that make the chords decisive
KEY_TIE = 0.03                                     # chord-fit shares closer than this are a tie
KEY_TONIC_RATIO = 1.25                             # tonic-chord evidence of one of a relative pair must beat the other by this


def area_shares(events):
    """share of decoded chord-time that sits inside each key area's notes (named by its major key; relative keys share notes)."""
    tot = sum(e["t1"] - e["t0"] for e in events if e["root"] is not None) or 1.0
    return {t: sum(e["t1"] - e["t0"] for e in events if e["root"] is not None and is_diatonic(e["root"], e["q"], (t, "major"))) / tot
            for t in range(12)}


def tonic_scores(events, key):
    """time on the tonic chord (+0.6 x the dominant, +bonus for the final chord) for `key`."""
    s = 0.0
    for e in events:
        if e["root"] is None:
            continue
        d = e["t1"] - e["t0"]
        iv = (e["root"] - key[0]) % 12
        if iv == 0 and cls(e["q"]) == ("M" if key[1] == "major" else "m"):
            s += d
        elif iv == 7:
            s += 0.6 * d
    last = [e for e in events if e["root"] is not None]
    if last and (last[-1]["root"] - key[0]) % 12 == 0:
        s += 8
    return s


def area_name(t):
    return f"{M.NAMES[t]} major / {M.NAMES[(t + 9) % 12]} minor"


def key_verdict(events, glob, spec):
    """ONE key claim for the whole card. Weighs what the decoded chords say (diatonic share per key area) against the
    spectral reading `spec` (music.analyze_key output) and returns {"state", "key", "line", ...}:
      state 'chords'   the chords decide (>= KEY_STRONG_SHARE of chord-time fits one area, clearly ahead): that key is the claim.
      state 'agree'    chords not decisive but the spectral reading is in the same key area: the working key is the claim.
      state 'spectral' chords not decisive, spectral reading is confident (margin >= 0.10): the spectral key is the claim.
      state 'split'    otherwise: no claim (key None); the line says so in one sentence.
    `glob` is the working home (what numerals are read against), always returned as "working"."""
    sh = area_shares(events)
    rank = sorted(sh.items(), key=lambda kv: -kv[1])
    (a1, s1), (a2, s2) = rank[0], rank[1]
    spec_key = parse_key(spec["key"])
    spec_area = spec_key[0] if spec_key[1] == "major" else (spec_key[0] + 3) % 12
    if a1 != spec_area and sh[spec_area] >= s1 - KEY_TIE:        # the spectral area fits the chords as well as the best: a tie goes to it
        a1, s1 = spec_area, sh[spec_area]
        a2, s2 = max(((t, v) for t, v in rank if t != a1), key=lambda kv: kv[1])
    glob_area = glob[0] if glob[1] == "major" else (glob[0] + 3) % 12
    maj, mnr = (a1, "major"), ((a1 + 9) % 12, "minor")
    sM, sm = tonic_scores(events, maj), tonic_scores(events, mnr)
    pick = mnr if sm > sM * KEY_TONIC_RATIO else maj
    if glob_area == a1:
        pick = glob
    pair_unsure = max(sM, sm) < KEY_TONIC_RATIO * max(min(sM, sm), 1e-9)
    out = {"working": key_label(glob), "shares": {area_name(t): round(v, 2) for t, v in rank[:3]}, "spectral": spec["key"],
           "spectral_margin": spec.get("margin")}
    strong = s1 >= KEY_STRONG_SHARE and s1 - s2 >= KEY_STRONG_LEAD
    tonic_note = (f"; tonic unsure, {key_label(maj)} or {key_label(mnr)} (same notes)" if pair_unsure else "")
    if strong:
        out.update(state="chords", key=key_label(pick),
                   line=f"{key_label(pick)} (from the chords: {round(100 * s1)}% of the chord-time fits it, next best {area_name(a2)} {round(100 * s2)}%{tonic_note})")
    elif spec_area == a1:
        out.update(state="agree", key=key_label(pick),
                   line=f"{key_label(pick)} (chords and spectral reading agree on the key area; chords fit {round(100 * s1)}%{tonic_note})")
    elif spec.get("margin", 0) >= 0.10:
        out.update(state="spectral", key=key_label(spec_key),
                   line=f"{key_label(spec_key)} (spectral r={spec['r']}, margin {spec['margin']}; the chords fit {area_name(a1)} {round(100 * s1)}% and do not decide)")
    else:
        sp_share = sh[spec_area]
        if s1 - s2 < KEY_STRONG_LEAD:
            why = f"chords fit {area_name(a1)} {round(100 * s1)}% and {area_name(a2)} {round(100 * s2)}% about equally"
        else:
            why = f"chords favour {area_name(a1)} {round(100 * s1)}% but the spectral reading says {spec['key']} (margin {spec.get('margin')}, chords fit it {round(100 * sp_share)}%)"
        out.update(state="split", key=None, line=f"key unclear: {why}; the song never settles on a tonic")
    return out


def label(e, spans):
    k = key_at(spans, (e["t0"] + e["t1"]) / 2)
    return roman(e["root"], e["q"], k)


def section_progression(evs, a, b, spans):
    seq = []
    for e in evs:
        if e["root"] is None:
            continue
        t0, t1 = max(e["t0"], a), min(e["t1"], b)
        if t1 - t0 <= 0.3:
            continue
        lab = label(e, spans)
        if seq and seq[-1] == lab:
            continue
        seq.append(lab)
    if not seq:
        return None
    # smallest period whose repeats match >= 80%
    best = None
    for p in range(1, 9):
        if len(seq) < 2 * p:
            continue
        m = sum(seq[i] == seq[i + p] for i in range(len(seq) - p)) / (len(seq) - p)
        if m >= 0.8:
            best = p
            break
    if best:
        cyc = []
        for ph in range(best):
            cyc.append(Counter(seq[ph::best]).most_common(1)[0][0])
        # drop a cycle that is really a shorter one repeated
        reps = len(seq) / best
        return {"cycle": cyc, "reps": round(reps, 1) if reps % 1 else int(reps)}
    return {"seq": seq[:14], "more": max(0, len(seq) - 14)}


def key_distance_words(frm, to):
    up = (to[0] - frm[0]) % 12
    names = {1: "a semitone", 2: "a whole tone", 3: "a minor third", 4: "a major third", 5: "a fourth",
             6: "a tritone", 7: "a fifth", 8: "a minor sixth", 9: "a major sixth", 10: "a minor seventh",
             11: "a major seventh"}
    if up == 0:
        return "to its parallel minor/major" if frm[1] != to[1] else "sideways"
    if up in (5, 7, 1, 2, 3, 4) and up <= 7:
        return f"up {names[up]}"
    return f"down {names[12 - up]}"


@ST.stage('harmonic moves')
def notable_moves(evs, spans, dur):
    out = []
    ch = [e for e in evs if e["root"] is not None]
    for i, e in enumerate(ch):
        k = key_at(spans, e["t0"])
        iv = (e["root"] - k[0]) % 12
        c = cls(e["q"])
        rn = roman(e["root"], e["q"], k)
        prev = ch[i - 1] if i else None
        pk = key_at(spans, prev["t0"]) if prev else k
        if prev is not None:
            piv = (prev["root"] - pk[0]) % 12
            pc = cls(prev["q"])
            if k == pk:
                if k[1] == "major" and piv == 7 and pc == "M" and iv == 9 and c == "m":
                    out.append((e["t0"], "deceptive", f"deceptive cadence V→vi: the dominant is denied its tonic"))
                if k[1] == "minor" and piv == 7 and iv == 8 and c == "M":
                    out.append((e["t0"], "deceptive", "deceptive cadence V→♭VI"))
                if piv == 5 and iv == 0 and pc == "M" and c == "M":
                    out.append((e["t0"], "plagal", "plagal cadence IV→I (the 'amen' fall)"))
                if piv == 5 and iv == 0 and pc == "m" and c == "M" and k[1] == "major":
                    out.append((e["t0"], "plagal", "minor plagal iv→I: the borrowed minor subdominant softens into home"))
                if piv == 7 and iv == 0 and pc == "M" and c in ("M", "m"):
                    out.append((e["t0"], "authentic", "authentic cadence V→I"))
        if k[1] == "major":
            if iv == 5 and c == "m":
                out.append((e["t0"], "borrowed", f"borrowed {rn}: minor iv inside a major key"))
            if iv == 10 and c == "M":
                out.append((e["t0"], "borrowed", f"borrowed {rn}: ♭VII, the mixolydian flat seventh"))
            if iv == 8 and c == "M":
                out.append((e["t0"], "borrowed", f"borrowed {rn}: ♭VI from the parallel minor"))
            if iv == 3 and c == "M":
                out.append((e["t0"], "borrowed", f"borrowed {rn}: ♭III from the parallel minor"))
            if iv == 0 and c == "m":
                out.append((e["t0"], "borrowed", f"{rn}: the tonic turns minor"))
            if iv == 2 and c == "M" and e["q"] != "dom7":
                out.append((e["t0"], "borrowed", f"{rn}: major II (a lifted second degree)"))
        elif k[1] == "minor":
            if iv == 0 and c == "M":
                out.append((e["t0"], "borrowed", f"picardy: tonic lands major ({rn})"))
    # tonic never reached in long stretches
    return out


@ST.stage('pedals')
def find_pedals(evs, min_dur=6.0):
    res = []
    ch = [e for e in evs if e["root"] is not None and e.get("bass") is not None]
    i = 0
    while i < len(ch):
        j = i
        while j + 1 < len(ch) and ch[j + 1]["bass"] == ch[i]["bass"]:
            j += 1
        distinct = {(c["root"], c["q"]) for c in ch[i:j + 1]}
        if j > i and len(distinct) >= 2 and ch[j]["t1"] - ch[i]["t0"] >= min_dur:
            res.append((ch[i]["t0"], ch[j]["t1"], ch[i]["bass"], len(distinct)))
        i = j + 1
    return res


# --------------------------------------------------------------------------- tension
@ST.stage('tension')
def tension_curve(F, evs, spans, step=2.0):
    chroma, rms_db, oenv = F["chroma"], F["rms_db"], F["oenv"]
    dur = F["y_len"] / SR
    nb = int(dur // step)
    harm = np.zeros(nb); diss = np.zeros(nb); loud = np.zeros(nb); dens = np.zeros(nb)
    lo, hi = np.percentile(rms_db, 5), np.percentile(rms_db, 97)
    for i in range(nb):
        a, b = i * step, (i + 1) * step
        fa, fb = int(a * FPS), int(b * FPS)
        c = chroma[:, fa:fb].mean(axis=1)
        c = c / (c.sum() + 1e-9)
        ent = float(-(c * np.log(c + 1e-9)).sum() / np.log(12))
        mid = (a + b) / 2
        e = next((x for x in evs if x["t0"] <= mid < x["t1"] and x["root"] is not None), None)
        if e is not None:
            k = key_at(spans, mid)
            harm[i] = func_tension(e["root"], e["q"], k)
            ct = chord_tones(e["root"], e["q"])
            outside = float(sum(c[p] for p in range(12) if p not in ct))
        else:
            harm[i] = harm[i - 1] if i else 0.3
            outside = 0.3
        diss[i] = 0.5 * outside + 0.5 * ent
        loud[i] = np.clip((rms_db[fa:fb].mean() - lo) / (hi - lo + 1e-6), 0, 1)
        dens[i] = oenv[fa:fb].mean()
    def nz(x):
        l, h = np.percentile(x, 5), np.percentile(x, 95)
        return np.clip((x - l) / (h - l + 1e-9), 0, 1)
    diss_n, dens_n = nz(diss), nz(dens)
    T = 0.35 * harm + 0.20 * diss_n + 0.30 * loud + 0.15 * dens_n
    from scipy.ndimage import gaussian_filter1d
    Ts = gaussian_filter1d(T, 1.5)
    return {"step": step, "T": Ts, "raw": T, "harm": harm, "diss": diss_n, "loud": loud, "dens": dens_n}


def b_is_intro(T):
    return False


@ST.stage('builds / releases')
def builds_releases(tc, evs, spans, F):
    from scipy.signal import find_peaks
    T, step = tc["T"], tc["step"]
    pk, _ = find_peaks(T, prominence=0.12, distance=3)
    vl, _ = find_peaks(-T, prominence=0.12, distance=3)
    ext = sorted([(int(i), "p") for i in pk] + [(int(i), "v") for i in vl])
    # enforce alternation, keep the more extreme of runs
    alt = []
    for i, k in ext:
        if alt and alt[-1][1] == k:
            if (k == "p" and T[i] > T[alt[-1][0]]) or (k == "v" and T[i] < T[alt[-1][0]]):
                alt[-1] = (i, k)
        else:
            alt.append((i, k))
    if alt and alt[0][1] == "p" and alt[0][0] > 0:
        alt.insert(0, (0, "v"))
    if alt and alt[-1][1] == "v" and alt[-1][0] < len(T) - 2:
        alt.append((len(T) - 1, "p"))
    res = []
    for (i, ki), (j, kj) in zip(alt[:-1], alt[1:]):
        d = T[j] - T[i]
        if abs(d) < 0.18 or (j - i) * step < 8 or (i == 0 and ki == "v" and b_is_intro(T)):
            continue
        a, b = i * step, j * step
        comp = {n: float(tc[n][j] - tc[n][i]) for n in ("harm", "diss", "loud", "dens")}
        res.append({"kind": "build" if d > 0 else "release", "t0": a, "t1": b, "from": float(T[i]),
                    "to": float(T[j]), "comp": comp})
    return res


def chord_at(evs, t):
    return next((x for x in evs if x["t0"] <= t < x["t1"] and x["root"] is not None), None)


def describe_build(b, evs, spans, tc, F, working=False):
    comp = b["comp"]
    reference = "the working key" if working else "home"
    names = {"harm": f"harmony pulls from {reference}", "diss": "denser, less settled chroma",
             "loud": "level climbs", "dens": "more activity"}
    sign = 1 if b["kind"] == "build" else -1
    top = sorted(comp, key=lambda n: -sign * comp[n])[:2]
    why = " and ".join(names[n] if sign > 0 else names[n].replace("climbs", "falls").replace("pulls from", "settles toward").replace("denser, less settled chroma", "chroma clears").replace("more activity", "less activity") for n in top if sign * comp[n] > 0.03)
    e1 = chord_at(evs, max(b["t1"] - 1.0, 0))
    k = key_at(spans, b["t1"])
    if b["kind"] == "build":
        on = label(e1, spans) if e1 else "?"
        tail = ""
        if not working and e1 is not None and (e1["root"] - k[0]) % 12 != 0:
            tail = f", unresolved on {on}"
        return (f"{tt(b['t0'])}–{tt(b['t1'])} tension climbs {b['from']:.2f}→{b['to']:.2f}"
                f"{' (' + why + ')' if why else ''}{tail}")
    nxt = chord_at(evs, b["t1"] + 1.0)
    where = f" to {label(nxt, spans)}" if nxt else ""
    # band stays? compare loudness before/after
    fa = int(max(b["t0"] - 2, 0) * FPS); fb = int(b["t0"] * FPS) + 1
    ga = int(b["t1"] * FPS); gb = int((b["t1"] + 3) * FPS)
    ldiff = F["rms_db"][ga:gb].mean() - F["rms_db"][fa:fb].mean() if gb > ga and fb > fa else 0
    body = "with the full band still there" if ldiff > -2 else ("the band thinning" if ldiff > -6 else "most of it falling away")
    if any(e.get("band_out") and e["t0"] < b["t1"] and e["t1"] > b["t0"] for e in evs):
        body = "with harmonic instruments out for part of this span"
    return (f"{tt(b['t0'])}–{tt(b['t1'])} tension eases {b['from']:.2f}→{b['to']:.2f}, settling{where}, {body}"
            f"{' (' + why + ')' if why else ''}")


# --------------------------------------------------------------------------- dips / silences / swells / timbre novelty
@ST.stage('level dips')
def find_dips(F, dur):
    rms_db = F["rms_db"]
    from scipy.ndimage import uniform_filter1d, median_filter
    sm = uniform_filter1d(rms_db, int(0.25 * FPS))
    # the song's own level: rolling median over 12 s
    ref = median_filter(sm, size=int(12 * FPS), mode="nearest")
    low = sm < ref - 12
    low[: int(1.5 * FPS)] = False
    low[-int(1.5 * FPS):] = False
    res = []
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if (j - i) / FPS >= 0.3:
                a, b = i / FPS, j / FPS
                depth = float(ref[i:j].mean() - sm[i:j].min())
                abs_floor = float(sm[i:j].min())
                digital = abs_floor < rms_db.max() - 60
                after = sm[min(j + int(1.0 * FPS), len(sm) - 1)] - ref[j - 1]
                res.append({"t0": a, "t1": b, "depth": depth, "silent": bool(digital), "returns": bool(after > -6)})
            i = j
        else:
            i += 1
    return res


@ST.stage('dropouts')
def find_dropouts(F):
    """Sustained level falls of >= 6 dB within ~2 s that stay down >= 4 s (the band thins out)."""
    rms_db = F["rms_db"]
    from scipy.ndimage import uniform_filter1d
    sm = uniform_filter1d(rms_db, int(1.0 * FPS))
    step = int(0.5 * FPS)
    res = []
    last = -99
    w, f = int(3 * FPS), int(4 * FPS)
    for i in range(w, len(sm) - f, step):
        before = sm[i - w:i].mean()
        after = sm[i:i + f].mean()
        if before - after >= 6 and i / FPS - last > 8:
            res.append({"t": i / FPS, "drop": float(before - after)})
            last = i / FPS
    return res


@ST.stage('swells')
def find_swells(F):
    """Rises >= 7 dB over ~2.5 s (the sound grows suddenly), as chill candidates."""
    rms_db = F["rms_db"]
    from scipy.ndimage import uniform_filter1d
    sm = uniform_filter1d(rms_db, int(1.0 * FPS))
    w = int(2.5 * FPS)
    res, last = [], -99
    for i in range(w, len(sm), int(0.25 * FPS)):
        d = sm[i] - sm[i - w]
        if d >= 9 and i / FPS - last > 6 and i / FPS > 6:
            # find the start of the climb: local min before
            seg = sm[i - w:i + 1]
            t0 = (i - w + int(np.argmin(seg))) / FPS
            last = i / FPS
            if t0 >= 5:
                res.append({"t": t0, "rise": float(d)})
    return res


@ST.stage('timbre entrances')
def timbre_entrances(F, dur):
    """Candidate new voice/instrument: a jump in timbre (MFCC) between the 3 s before and 3 s after
    that is NOT just a level change. A proxy only; the stem limb handles real entrances."""
    from scipy.signal import find_peaks
    mf = F["mfcc"][1:]
    z = (mf - mf.mean(axis=1, keepdims=True)) / (mf.std(axis=1, keepdims=True) + 1e-6)
    sec = int(FPS)
    nb = z.shape[1] // sec
    P = np.stack([z[:, i * sec:(i + 1) * sec].mean(axis=1) for i in range(nb)], axis=1)
    W = 3
    nov = np.zeros(nb)
    for i in range(W, nb - W):
        nov[i] = np.linalg.norm(P[:, i - W:i].mean(axis=1) - P[:, i:i + W].mean(axis=1))
    nov[:6] = 0
    pk, _ = find_peaks(nov, height=nov.mean() + 1.3 * nov.std(), distance=8)
    rms = F["rms_db"]
    res = []
    for i in pk:
        a = rms[int((i - 3) * FPS):int(i * FPS)].mean()
        b = rms[int(i * FPS):int((i + 3) * FPS)].mean()
        res.append({"t": float(i), "nov": float(nov[i]), "dlevel": float(b - a)})
    return res, nov


# --------------------------------------------------------------------------- melody -> appoggiaturas
@ST.stage('appoggiaturas')
def appoggiaturas(mel, evs):
    f0, vp, hop = mel["f0"], mel["vp"], mel["hop"]
    import librosa
    midi = np.where(np.isfinite(f0), np.round(librosa.hz_to_midi(np.where(np.isfinite(f0), f0, 1.0))), np.nan)
    fr = SR / hop
    segs = []
    i = 0
    while i < len(midi):
        if np.isnan(midi[i]):
            i += 1
            continue
        j = i
        while j + 1 < len(midi) and midi[j + 1] == midi[i]:
            j += 1
        segs.append((i / fr, (j + 1) / fr, int(midi[i])))
        i = j + 1
    res = []
    for a, nxt in zip(segs[:-1], segs[1:]):
        d = a[1] - a[0]
        if d < 0.2 or d > 2.0 or nxt[0] - a[1] > 0.15 or abs(nxt[2] - a[2]) > 2 or nxt[2] == a[2]:
            continue
        e1 = chord_at(evs, (a[0] + a[1]) / 2)
        e2 = chord_at(evs, nxt[0] + 0.05)
        if not e1 or not e2:
            continue
        nct = (a[2] % 12) not in chord_tones(e1["root"], e1["q"])
        resolves = (nxt[2] % 12) in chord_tones(e2["root"], e2["q"])
        on_change = a[0] - e1["t0"] < 0.35
        # stepwise downward resolution (the classic sigh) scores higher
        if nct and resolves and a[0] - e1["t0"] < 1.0:
            score = d * (1.5 if on_change else 1.0) * (1.2 if nxt[2] < a[2] else 1.0)
            res.append({"t": a[0], "dur": d, "note": PCN[a[2] % 12], "to": PCN[nxt[2] % 12],
                        "chord": chord_name(e1["root"], e1["q"]), "score": score, "down": nxt[2] < a[2]})
    res.sort(key=lambda r: -r["score"])
    return res


# --------------------------------------------------------------------------- sentiment
_SENT = None
_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentNotCached(RuntimeError):
    remedy_first = True


def _cached_sentiment():
    from huggingface_hub.utils import LocalEntryNotFoundError

    remedy = "sentiment weights or tokenizer not cached; run --setup music while online"
    try:
        return setup_sentiment(check=True)
    except (LocalEntryNotFoundError, FileNotFoundError) as exc:
        raise SentimentNotCached(remedy) from exc


def setup_sentiment(check=False):
    from pathlib import Path
    from huggingface_hub import snapshot_download

    weights = ("model.safetensors", "pytorch_model.bin")
    tokens = ("vocab.json", "merges.txt")
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", *tokens, *weights)
    options = {"local_files_only": check}
    if not check:
        options["allow_patterns"] = files
    path = Path(snapshot_download(_SENTIMENT_MODEL, **options))

    def present(name):
        file = path / name
        return file.is_file() and file.stat().st_size > 0

    tokenizer = present("tokenizer.json") or all(map(present, tokens))
    if not present("config.json") or not any(map(present, weights)) or not tokenizer:
        raise FileNotFoundError("sentiment config, weights or tokenizer missing or empty")
    return str(path)


@ST.stage('sentiment')
def sentiment(texts):
    global _SENT
    if _SENT is None:
        from model_compat import check_transformers_torch
        check_transformers_torch()
        from transformers import pipeline
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        model = _cached_sentiment() if offline() else _SENTIMENT_MODEL
        _SENT = pipeline("text-classification", model=model,
                         tokenizer=model, top_k=None, device=-1)
    out = []
    for r in _SENT(texts, batch_size=16, truncation=True):
        d = {x["label"].lower(): x["score"] for x in r}
        out.append(float(d.get("positive", 0) - d.get("negative", 0)))
    return out


# gap re-listen and the hallucination filter now live in lyrics_shared.py (one transcription per file,
# cached by sha256 in ~/.cache/raven-lyrics, shared with music.py and singer.py).


def lines_with_ends(words, gap=0.7, maxw=10):
    lines, cur = [], []
    for w, s, e in words:
        if cur and (s - cur[-1][2] > gap or len(cur) >= maxw):
            lines.append(cur)
            cur = []
        cur.append((w, s, e))
    if cur:
        lines.append(cur)
    return [{"t": l[0][1], "end": l[-1][2], "text": " ".join(w for w, _, _ in l)} for l in lines]


def voice_entry(lines, words, spans, onset=None):
    """Place the entry using stem timing and quote the first supported words."""
    from lyrics_shared import LINE_SLACK

    if spans is None:
        if lines:
            return lines[0]["t"], f"first timed lyric (voice onset unclear): '{lines[0]['text'][:60]}'"
        return None
    if onset is None:
        onset = min((a for a, _ in spans), default=None)
    if onset is None:
        return None

    def overlaps(start, end):
        return any(a <= start < b or a < end and start < b for a, b in spans)

    if words:
        # A single Whisper line can straddle the onset. Select by word before
        # regrouping so the quote cannot bring an earlier silent word with it.
        first = next((i for i, (_, start, end) in enumerate(words) if overlaps(start, end)), len(words))
        lines = lines_with_ends(words[first:])
    line = next((row for row in lines if overlaps(row["t"], row["end"])), None)
    if not line:
        return onset, "voice enters: vocal stem active"
    quote = f"'{line['text'][:60]}'"
    if abs(line["t"] - onset) > LINE_SLACK:
        return onset, f"voice enters: {quote} (words approximate here)"
    return line["t"], f"voice enters: {quote}"


R_STABLE = []
R_BEST = []
STOP = set("i i'm you your we he she it they a an the to of and or but in on at is are was be am don't dont wanna gonna can't cant just so oh yeah ooh my me with for that this".split()) - {"be"}


def repeated_phrase_anomaly(words):
    """Find a short phrase that comes in bursts (a chorus tag). If the LAST burst has a different count
    than the earlier ones, report it. e.g. 'be better' x3, x3, x4."""
    import re
    toks = [(re.sub(r"[^a-z']", "", w.lower()), s) for w, s, _ in words]
    toks = [(w, s) for w, s in toks if w]
    best = []
    for n in (2, 3):
        occ = {}
        for i in range(len(toks) - n + 1):
            g = " ".join(w for w, _ in toks[i:i + n])
            occ.setdefault(g, []).append(toks[i][1])
        for g, ts in occ.items():
            if len(ts) < 6 or len(set(g.split())) < 2 or all(t in STOP for t in g.split()):
                continue
            # bursts: occurrences closer than 6 s apart belong together
            bursts, cur = [], [ts[0]]
            for t in ts[1:]:
                if t - cur[-1] < 6:
                    cur.append(t)
                else:
                    bursts.append(cur); cur = [t]
            bursts.append(cur)
            counts = [len(b) for b in bursts]
            if len(bursts) >= 3 and max(counts) >= 2:
                best.append((g, bursts, counts))
    out = []
    R_STABLE.clear()
    R_BEST[:] = sorted(best, key=lambda x: -sum(x[2]))[:1]
    for g, bursts, counts in sorted(best, key=lambda x: -sum(x[2]))[:1]:
        if len(set(counts)) == 1:
            R_STABLE.append({"phrase": g, "counts": counts, "t": bursts[-1][0]})
    for g, bursts, counts in best:
        prev = counts[:-1]
        mode = Counter(prev).most_common(1)[0][0]
        if counts[-1] != mode and mode >= 2:
            out.append({"t": bursts[-1][0], "phrase": g, "counts": counts, "was": mode, "now": counts[-1]})
    out.sort(key=lambda r: -abs(r["now"] - r["was"]))
    return out[:2]


@ST.stage('refrain check')
def refrain_check(path, words, dur):
    """REFRAIN COUNT, second opinion from the audio (24 Oct 2026). The text count (repeated_phrase_anomaly) only
    knows what Whisper transcribed. Here the phrase it found is turned into a TEMPLATE (MFCCs of each transcribed
    repeat, taken from the isolated vocal stem), the template is slid over the whole stem, and every place the
    voice sings something that matches it counts, transcribed or not. Bursts (matches < 6 s apart) are then
    compared with the text bursts. Needs the cached vocal stem; without it only the text count exists.
    Honest limits: it finds repeats that SOUND like the transcribed ones (same words, similar delivery); a repeat
    sung on a different melody, or buried under a doubled vocal, can fall under the threshold."""
    out = {"phrase": None}
    if not words or not R_BEST:
        return out
    import re
    g, bursts, counts = R_BEST[0]
    out.update(phrase=g, text_counts=counts, text_t=[b[0] for b in bursts])
    vp = vocal_stem_path(path)
    if not vp:
        out["stem"] = "no vocal stem cached"
        ST.mark("refrain audio check", "skipped", out["stem"])
        return out
    try:
        import librosa
        from scipy.ndimage import uniform_filter1d
        from scipy.signal import fftconvolve, find_peaks
        toks = [(re.sub(r"[^a-z']", "", w.lower()), s_, e_) for w, s_, e_ in words]
        toks = [t for t in toks if t[0]]
        gw = g.split()
        n_ = len(gw)
        occ = [(toks[i][1], toks[i + n_ - 1][2]) for i in range(len(toks) - n_ + 1)
               if [t[0] for t in toks[i:i + n_]] == gw]
        if len(occ) < 4:
            out["stem"] = "too few transcribed repeats to build a template"
            return out
        y = load_vocal_22k(vp)
        hop = 256
        fps = SR / hop
        M_ = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, n_fft=1024, hop_length=hop, n_mels=64)[1:]
        M_ = (M_ - M_.mean(1, keepdims=True)) / (M_.std(1, keepdims=True) + 1e-6)
        M_ = uniform_filter1d(M_, 3, axis=1)
        Mn = M_ / (np.linalg.norm(M_, axis=0) + 1e-9)
        if float(np.median([e - s_ for s_, e in occ])) < 0.7:
            out["stem"] = "phrase too short (<0.7 s) for a reliable template match"
            return out
        n = max(int(np.median([e - s_ for s_, e in occ]) * fps), 8)
        pick = occ if len(occ) <= 14 else [occ[i] for i in np.linspace(0, len(occ) - 1, 14).astype(int)]
        curves = []
        for s_, _e in pick:
            a = int(s_ * fps)
            if a + n > Mn.shape[1]:
                continue
            T = Mn[:, a:a + n]
            cc = sum(fftconvolve(Mn[d], T[d][::-1], mode="valid") for d in range(T.shape[0])) / n
            c = np.zeros(Mn.shape[1])
            c[:len(cc)] = cc
            curves.append(c)
        C = np.median(curves, axis=0)
        ref = float(np.median([C[min(int(s_ * fps), len(C) - 1)] for s_, _e in occ]))
        thr = max(0.3, 0.75 * ref)
        pk, _ = find_peaks(C, height=thr, distance=max(int(0.5 * fps), int(0.6 * n)))
        t = pk / fps
        sb, cur = [], []
        for x in t:
            if cur and x - cur[-1] >= 6:
                sb.append(cur)
                cur = []
            cur.append(float(x))
        if cur:
            sb.append(cur)
        # a lone hit that is not near a transcribed burst is a similar-sounding stretch, not a refrain
        sb = [b for b in sb if len(b) >= 2 or any(abs(tb[0] - b[0]) < 3.0 for tb in bursts)]
        out.update(stem="vocal stem template match", stem_counts=[len(b) for b in sb], stem_t=[b[0] for b in sb],
                   thr=round(thr, 2), ref=round(ref, 2))
        al = []
        for b in sb:
            m = [i for i, tb in enumerate(bursts) if abs(tb[0] - b[0]) < 3.0]
            al.append((b[0], counts[m[0]] if m else 0, len(b)))
        out["aligned"] = al
    except Exception as e:
        out["stem"] = f"template match failed ({type(e).__name__}: {str(e)[:60]})"
        ST.mark("refrain audio check", "failed", out["stem"])
    return out


# --------------------------------------------------------------------------- novelty / surprise
@ST.stage('novelty')
def self_sim_novelty(F, dur):
    from scipy.signal import find_peaks
    chroma = F["chroma"]; mf = F["mfcc"][1:]; rms = F["rms_db"]
    sec = int(FPS)
    nb = chroma.shape[1] // sec
    def pool(m):
        return np.stack([m[:, i * sec:(i + 1) * sec].mean(axis=1) for i in range(nb)], axis=1)
    def z(m):
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)
    X = np.vstack([z(pool(chroma)), z(pool(mf)), z(np.array([[rms[i * sec:(i + 1) * sec].mean() for i in range(nb)]]))])
    Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = Xn.T @ Xn
    L = 8
    g = np.exp(-0.5 * (np.arange(-L, L)[:, None] ** 2 + np.arange(-L, L)[None, :] ** 2) / (L / 2) ** 2)
    sgn = np.sign(np.arange(-L, L) + 0.5)
    K = g * np.outer(sgn, sgn)
    nov = np.zeros(nb)
    P = np.pad(S, L, mode="edge")
    for i in range(nb):
        nov[i] = (P[i:i + 2 * L, i:i + 2 * L] * K).sum()
    nov = np.maximum(nov, 0)
    nov[:6] = 0
    pk, _ = find_peaks(nov, height=nov.mean() + 1.2 * nov.std(), distance=10)
    return [(float(i), float(nov[i])) for i in pk], nov


@ST.stage('chord surprises')
def chord_surprises(evs, spans):
    ch = [e for e in evs if e["root"] is not None]
    labs = [label(e, spans) for e in ch]
    if len(labs) < 12:
        return []
    bi = Counter(zip(labs[:-1], labs[1:]))
    uni = Counter(labs[:-1])
    res = []
    for i in range(1, len(labs)):
        a, b = labs[i - 1], labs[i]
        if a == b:
            continue
        n = bi[(a, b)]
        others = Counter({c: v for (x, c), v in bi.items() if x == a})
        if not others:
            continue
        exp, expn = others.most_common(1)[0]
        dur = ch[i]["t1"] - ch[i]["t0"]
        if n == 1 and uni[a] >= 3 and expn >= 3 and exp != b and dur >= 1.0 and 1 < i < len(labs) - 2:
            k = key_at(spans, ch[i]["t0"])
            nd = not is_diatonic(ch[i]["root"], ch[i]["q"], k)
            res.append({"t": ch[i]["t0"], "from": a, "got": b, "expected": exp, "n_expected": expn,
                        "chromatic": nd, "dur": dur, "score": (2 if nd else 1) + dur / 10})
    res.sort(key=lambda r: -r["score"])
    return res


# --------------------------------------------------------------------------- cadences
CONF_SKIP, CONF_LIKELY = 0.25, 0.5   # chord-model confidence: under the first we don't call it, under the second we say "likely"


@ST.stage('downbeats')
def tracked_downbeats(path, cp, use_cache=True):
    """Bar starts from the real downbeat tracker (singer.py's Beat This!, madmom fallback), so a 7/4 or 10/8
    song gets its own bars instead of an assumed four. Cached beside the melody; None if the tracker fails."""
    # singer owns the success-only cache, backend identity and stage provenance.
    # Ignore old .downbeats.pkl files, including cached failures and fallbacks.
    try:
        import singer
        _, d, _ = singer.beats_downbeats(path, use_cache=use_cache)
        return [float(x) for x in d] if len(d) >= 4 else None
    except Exception as e:
        ST.mark("downbeats", "failed", f"{type(e).__name__}: {str(e)[:80]}")
        return None


def find_downbeats(beats, oenv, meter=4):
    """Bar starts (seconds), or None when no beat phase is clearly stronger. Beats are tracker frames; the
    phase is whichever 1-in-`meter` beat carries the most onset energy. The meter is assumed, not detected."""
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 4 * meter:
        return None
    oe = np.asarray(oenv)[np.clip(beats, 0, len(oenv) - 1)]
    share = np.array([oe[p::meter].mean() for p in range(meter)])
    if share.max() < 1.08 * np.mean(share):
        return None
    return [float(b) / FPS for b in beats[int(np.argmax(share))::meter]]


def classify_cadence(prev, last, key):
    """What the move prev -> last is, in key: authentic / plagal / deceptive / half / home (tonic by some other
    route), or None. Scale degrees only; prev and last are decoded chord events (root, q)."""
    ivp, ivl = (prev["root"] - key[0]) % 12, (last["root"] - key[0]) % 12
    cp, cl = cls(prev["q"]), cls(last["q"])
    minor = key[1] == "minor"
    if ivl == 0 and cl in ("M", "m"):      # a picardy third or a minor tonic in a major song is still home
        if ivp == 7 and cp == "M" or ivp == 11 and cp == "d":
            return "authentic"
        return "plagal" if ivp == 5 else "home"
    if ivp == 7 and cp == "M" and ((not minor and ivl == 9 and cl == "m") or (minor and ivl == 8 and cl == "M")):
        return "deceptive"
    if ivl == 7 and cl == "M" and ivp != 7:
        return "half"
    return None


@ST.stage('cadences')
def find_cadences(evs, spans, ends, tc, downbeats=None, key_from=0.0):
    """One verdict per phrase end. `ends` is [(t_end, kind)], kind "section" or "line". Reads the last two decoded
    chords at each end. strength (0-1): 0.4, +0.3 if the arrival is on a bar start, +0.3 if the bass is on the
    chord's root. before/after: tension (0-1), its peak in the 6 s up to the arrival and its low in the 6 s after."""
    ch = [e for e in evs if e["root"] is not None]
    T, step = tc["T"], tc["step"]
    res = []
    for t_end, kind in ends:
        i = next((j for j, e in enumerate(ch) if e["t0"] <= t_end - 0.1 < e["t1"]), None)
        if i is None or i == 0:
            continue
        prev, last = ch[i - 1], ch[i]
        conf = min(prev.get("conf", 1.0), last.get("conf", 1.0))
        if conf < CONF_SKIP or last["t0"] < key_from:
            res.append({"t_end": t_end, "kind": kind, "cadence": "unclear", "t": last["t0"], "strength": 0.0, "likely": True})
            continue
        key = key_at(spans, last["t0"])
        on_bar = bool(downbeats) and min(abs(last["t0"] - d) for d in downbeats) < 0.15
        strength = 0.4 + 0.3 * on_bar + 0.3 * (last.get("bass") == last["root"])
        k = int(last["t0"] // step)
        before = float(T[max(0, k - 3):k].max()) if 0 < k <= len(T) else None
        after = float(T[k:k + 3].min()) if k < len(T) else None
        res.append({"t_end": t_end, "kind": kind, "cadence": classify_cadence(prev, last, key) or "open",
                    "t": last["t0"], "strength": strength, "likely": conf < CONF_LIKELY or strength < 0.5,
                    "from": roman(prev["root"], prev["q"], key), "to": roman(last["root"], last["q"], key),
                    "before": before, "after": after})
    return res


def phrase_ends(secs, lines, dur, min_gap=1.0):
    """Section ends, plus lyric-line ends followed by a real pause (a line cut at the word limit is not a phrase
    end). A line end within 2 s of another end collapses into it."""
    ends = [(float(s["end"]), "section") for s in (secs or [])]
    for a, b in zip(lines or [], (lines or [])[1:] + [None]):
        if b is None or b["t"] - a["end"] >= min_gap:
            ends.append((float(a["end"]), "line"))
    ends.sort()
    out = []
    for t, k in ends:
        if out and t - out[-1][0] < 2.0:
            if k == "section" and out[-1][1] == "line":
                out[-1] = (t, k)
            continue
        out.append((min(t, dur), k))
    return out


# --------------------------------------------------------------------------- contrast
def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


@ST.stage('musical lift')
def music_lift_frames(F, tc, evs, dur):
    """Per-second 'lift' of the music: energy, activity, brightness (as a composite), harmonic rhythm,
    minus tension. NOT major/minor. A heavy/held vs lifted/driving axis, in song-relative z-scores."""
    n = int(dur)
    sec = int(FPS)
    loud = np.array([F["rms_db"][i * sec:(i + 1) * sec].mean() for i in range(n)])
    dens = np.array([F["oenv"][i * sec:(i + 1) * sec].mean() for i in range(n)])
    br = np.array([np.median(F["cent"][i * sec:(i + 1) * sec]) for i in range(n)])
    hr = np.zeros(n)
    for e in evs:
        i = int(e["t0"])
        if 0 <= i < n:
            hr[i] += 1
    from scipy.ndimage import uniform_filter1d
    hr = uniform_filter1d(hr, 8, mode="nearest")
    ten = np.interp(np.arange(n) + 0.5, (np.arange(len(tc["T"])) + 0.5) * tc["step"], tc["T"])
    comps = {"energy": zs(loud), "activity": zs(dens), "brightness": zs(br), "harmonic_rhythm": zs(hr), "tension": zs(ten)}
    lift = (comps["energy"] + comps["activity"] + comps["brightness"] + comps["harmonic_rhythm"]) / 4 - 0.5 * comps["tension"]
    return lift, comps


@ST.stage('word / music contrast')
def contrast(lines, lift, comps, dur):
    texts = [l["text"] for l in lines]
    ctx = []
    for i in range(len(lines)):
        ctx.append(" ".join(texts[max(0, i - 1):i + 2]))
    try:
        own = sentiment(texts)
        around = sentiment(ctx)
    except SentimentNotCached as exc:
        ST.mark("word / music contrast", "skipped", str(exc))
        return []
    except Exception as exc:
        ST.mark("word / music contrast", "failed", f"sentiment unavailable ({type(exc).__name__})")
        return []
    res = []
    for i, l in enumerate(lines):
        v = 0.6 * own[i] + 0.4 * around[i]
        a, b = int(l["t"]), max(int(min(l["end"], dur)), int(l["t"]) + 1)
        seg = slice(a, min(b + 1, len(lift)))
        mz = float(np.mean(lift[seg])) if len(lift[seg]) else 0.0
        which = max(comps, key=lambda n: abs(np.mean(comps[n][seg])) if len(comps[n][seg]) else 0)
        res.append({"t": l["t"], "text": l["text"], "words": round(v, 2), "music": round(mz, 2), "lead": which})
    return res


# --------------------------------------------------------------------------- main assembly
def vocalless_spans(lines, dur, min_gap=18):
    spans = []
    if not lines:
        return spans
    marks = [0.0] + [x for l in lines for x in (l["t"], l["end"])] + [dur]
    prev_end = 0.0
    for l in lines:
        if l["t"] - prev_end >= min_gap and prev_end > 0:
            spans.append((prev_end, l["t"]))
        prev_end = l["end"]
    if dur - prev_end >= min_gap:
        spans.append((prev_end, dur))
    if lines[0]["t"] >= min_gap:
        spans.insert(0, (0.0, lines[0]["t"]))
    return spans


def resolve_home_chords_only(F, dur, notes):
    """The chords-only home key (the older path, kept as the fallback when the fusion cannot run). Decided in ONE place. Returns (kinfo, glob, evs, bpm, beats, swapped, verdict). `glob` is the working
    home the roman numerals are read against; `verdict` (key_verdict) is the claim every card line quotes."""
    import librosa
    kinfo = M.analyze_key(F["chroma"], dur)
    spec0 = dict(kinfo)                       # the spectral reading, kept: the verdict weighs it against the chords
    # tempo needs the percussive signal; recompute cheaply from the cached onset envelope
    tempo, beats = librosa.beat.beat_track(onset_envelope=F["oenv"], sr=SR, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    glob = parse_key(kinfo["key"])
    # chords
    evs = decode_chords(F, beats, glob, dur)
    glob2, swapped = refine_key(evs, glob)
    if swapped:
        log(f"home key: spectral pick was {key_label(glob)}; the chords sit on {key_label(glob2)} (relative key), using that")
        glob = glob2
        kinfo = dict(kinfo)
        kinfo["key"] = f"{M.NAMES[glob[0]]} {glob[1]}"
    # key by chords: the spectral pick (+ last chord) can land a fifth away when a song starts/ends on the V.
    # Rescore the top spectral candidates by how much of the decoded chord-time is diatonic to each.
    try:
        sc_ = M._key_scores(F["chroma"].mean(axis=1))
        order_ = [int(j) for j in np.argsort(sc_)[::-1][:5]]
        area0 = (glob[0] if glob[1] == "major" else (glob[0] + 3) % 12)
        cands = []
        for j in order_:
            kk = (j % 12, "major" if j < 12 else "minor")
            if M._area(j) == area0:
                continue          # same key area = relative pair, decided by refine_key
            cands.append(kk)
        cands = cands[:3]
        kbest, kswap, ktab = key_by_chords(evs, glob, cands)
        R_KT = " ".join(f"{key_label(k)} {100 * (1 - b):.0f}%" for k, b in ktab)
        log("key by chords (diatonic share): " + R_KT)
        if kswap:
            log(f"home key: spectral pick was {key_label(glob)}; the decoded chords are diatonic to {key_label(kbest)} "
                         f"({R_KT}), using that")
            glob = kbest
            kinfo = dict(kinfo)
            kinfo["key"] = f"{M.NAMES[glob[0]]} {glob[1]}"
            kinfo.pop("changes", None)
            evs = decode_chords(F, beats, glob, dur)
            g3, sw3 = refine_key(evs, glob)
            if sw3:
                glob = g3
                kinfo["key"] = f"{M.NAMES[glob[0]]} {glob[1]}"
    except Exception as e_:
        ST.mark("key by chords", "failed", f"{type(e_).__name__}: {e_}")
        log(f"key-by-chords skipped ({type(e_).__name__}: {e_})")
    verdict = key_verdict(evs, glob, spec0)
    kinfo = M.dominant_pull(kinfo, F["chroma"], dur, glob)
    return kinfo, glob, evs, bpm, beats, swapped, verdict


def gather_key_inputs(path, cp, F, use_cache=True, y=None):
    """What the key voters need beyond the chroma: the bass stem's pitch track (attached to F), the vocal melody, the
    downbeats and the madmom CNN's 24 probabilities. Each part that is missing or fails is simply left out."""
    import keyfuse
    inp = {}
    try:
        F["bass_trk"] = bass_track(path, cp, use_cache)
    except Exception as e:
        ST.mark("bass pitch", "failed", f"{type(e).__name__}: {str(e)[:60]}")
        log(f"bass stem pitch skipped ({type(e).__name__}: {str(e)[:60]})")
        F["bass_trk"] = None
    try:
        inp["mel"] = load_melody(path, cp, y, use_cache)
    except Exception as e:
        ST.mark("melody", "failed", f"{type(e).__name__}: {str(e)[:60]}")
        log(f"melody skipped ({type(e).__name__}: {str(e)[:60]})")
    try:
        inp["downbeats"] = tracked_downbeats(path, cp, use_cache)
    except Exception:
        inp["downbeats"] = None
    inp["cnn"] = keyfuse.cnn_probs(path, cp if use_cache else None)
    if inp["cnn"] is None:
        log("madmom CNN key voter not available (no ./venv-madmom); fusing without it")
    return inp


def fused_verdict(F, spec0, evs, inputs, dur):
    """The ONE key claim from all voters (keyfuse.py). Returns the verdict dict, with the per-voter log-probabilities under "_L"."""
    import keyfuse as KF
    inputs = inputs or {}
    voters = {"spectral": KF.v_spectral(M._key_scores(F["chroma"].mean(axis=1)))}
    if inputs.get("cnn") is not None:
        voters["cnn"] = KF.v_cnn(inputs["cnn"])
    voters["chords"] = KF.v_chords(evs)[0]
    bt = F.get("bass_trk")
    if bt is not None:
        try:
            ends = [t for t, _ in phrase_ends(sections_from_cache(F, dur), [], dur)]
        except Exception:
            ends = []
        voters["bass"] = KF.v_bass(bt["pc"], bt["rel"], FPS, inputs.get("downbeats"), ends, dur)[0]
    mel = inputs.get("mel")
    if mel is not None:
        voters["melody"] = KF.v_melody(mel["f0"], mel["hop"] / SR)[0]
    voters["cadence"] = KF.v_cadence(evs)[0]
    v = KF.verdict(voters)
    v["spectral"] = KF.key_name(KEYN_INDEX(parse_key(spec0["key"])))
    v["_L"] = voters
    return v


def mixolydian_home(events, home, bass=None):
    """Check chord function after the 24-key vote, using the same decoded chords.

    A borrowed flat VII or a secondary dominant alone is common in major.
    Require a recurring flat VII, a substantially leading major tonic, and
    repeated returns to it. Keep the bar higher when only the mode changes.
    """
    tonic, mode = home
    chords = [e for e in events if e["root"] is not None and not e.get("band_out")]
    if not chords:
        return None
    major_time, root_time = Counter(), Counter()
    for chord in chords:
        length = chord["t1"] - chord["t0"]
        root_time[chord["root"]] += length
        if cls(chord["q"]) == "M":
            major_time[chord["root"]] += length
    total = sum(root_time.values())
    candidates = (tonic, (tonic + 7) % 12) if mode == "major" else (tonic,)
    for root in candidates:
        flat7, fourth = (root + 10) % 12, (root + 5) % 12
        rival = max((s for r, s in root_time.items() if r != root), default=0)
        if major_time[root] < .28 * total or major_time[flat7] < .08 * total:
            continue
        on_tonic = lambda e: e["root"] == root and cls(e["q"]) == "M"
        opens = on_tonic(chords[0]) and chords[0]["t0"] <= 5
        closes = on_tonic(chords[-1])
        flat_events = [e for e in chords if e["root"] == flat7 and cls(e["q"]) == "M"]
        tonic7 = [e for e in chords if on_tonic(e) and e["q"] == "dom7"]
        i7 = len(tonic7) >= 2 and sum(e["t1"] - e["t0"] for e in tonic7) >= .05 * total
        pairs = [(a, b) for a, b in zip(chords, chords[1:]) if abs(a["t1"] - b["t0"]) <= .1]
        returns = sum(a["root"] in (flat7, fourth) and cls(a["q"]) == "M" and on_tonic(b)
                      for a, b in pairs)
        parent_resolves = sum(a in tonic7 and b["root"] == fourth and cls(b["q"]) == "M" for a, b in pairs)
        if len(flat_events) < 3 or returns < 3 or parent_resolves >= 2:
            continue
        # In a loop the IV chord can take as much time as I. Require a
        # repeated flat VII–IV7–I turn and check where the bass comes to rest.
        if root == tonic and mode == "major" and bass is not None and major_time[(root + 8) % 12] < .05 * total:
            turns = sum(a in flat_events and b["root"] == fourth and b["q"] == "dom7" and on_tonic(c)
                        and abs(a["t1"] - b["t0"]) <= .1 and abs(b["t1"] - c["t0"]) <= .1
                        for a, b, c in zip(chords, chords[1:], chords[2:]))
            if turns >= 3 and modal_bass_home(bass, root):
                return root, "mixolydian"
        if major_time[root] < 1.25 * rival:
            continue
        if root == tonic:
            # A substantial flat VI alongside flat VII points to mixture.
            if opens and closes and i7 and major_time[(root + 8) % 12] < .05 * total:
                return root, "mixolydian"
            continue
        if ((opens or closes) and i7 or
                opens and major_time[root] >= 1.5 * root_time[fourth]
                and sum(a in flat_events and on_tonic(b) for a, b in pairs) >= 3):
            return root, "mixolydian"
    return None


def modal_bass_home(bass, root):
    from keyfuse import bass_histogram

    # An ending played above the bass need not be the implied harmonic home.
    tail = slice(-int(15 * FPS), None)
    for region in (slice(None), tail):
        hist, cover = bass_histogram(bass["pc"][region], bass["rel"][region], FPS)
        if hist is None or cover < .08 or hist[root] < .4 * hist.sum():
            return False
    return True


def chroma_peaks(chroma):
    neighbours = np.maximum(np.roll(chroma, 1, axis=0), np.roll(chroma, -1, axis=0))
    return (chroma > 1.2 * neighbours) & (chroma >= .35 * chroma.max(axis=0))


def power_chords(F, events):
    """Refine an unresolved blues riff with sustained fifth-only evidence."""
    bass = F.get("bass_trk")
    if bass is None or not events:
        return events
    n = min(F["chroma"].shape[1], F["bass"].shape[1], len(F["rms_db"]), len(bass["pc"]), len(bass["rel"]))
    if not n:
        return events
    chroma, low = F["chroma"][:, :n], F["bass"][:, :n]
    rms = F["rms_db"][:n] - F["rms_db"].max()
    pc, rel = bass["pc"][:n], bass["rel"][:n]
    root = pc.clip(min=0)
    frame = np.arange(n)
    peak = chroma.max(axis=0)
    fifth = chroma[(root + 7) % 12, frame]
    thirds = np.maximum(chroma[(root + 3) % 12, frame], chroma[(root + 4) % 12, frame])
    peaks = chroma_peaks(chroma)
    other_peaks = peaks.copy()
    other_peaks[root, frame] = False
    other_peaks[(root + 7) % 12, frame] = False
    other = np.where(other_peaks, chroma, 0).max(axis=0)
    support = (fifth >= .35 * peak) & (thirds <= .45 * peak) & (other <= .70 * peak)

    # At a fade a quiet stem can corroborate an existing root when both mix
    # chromas agree. Only the reliable bass track can introduce a new root.
    decoded = np.full(n, -1)
    for e in events:
        if e["root"] is not None:
            decoded[round(e["t0"] * FPS):round(e["t1"] * FPS)] = e["root"]
    reliable = (pc >= 0) & (rel >= BASS_MIN_REL)
    agrees = (pc == decoded) & (pc == low.argmax(axis=0)) & (pc == chroma.argmax(axis=0))
    anchor = (reliable | (agrees & (rel >= .01))) & (rms > -45) & (chroma[root, frame] >= .8 * peak)
    roots = np.where(anchor, pc, -1)
    edges = np.r_[0, np.flatnonzero(roots[1:] != roots[:-1]) + 1, n]
    replacements = []
    for a, b in zip(edges, edges[1:]):
        r = int(roots[a])
        if r < 0 or b - a < round(.18 * FPS):
            continue
        # A different note as strong as the fifth may be a real chord tone
        # above this bass, even when neither third of the bass pitch is heard.
        share = float(support[a:b].mean())
        if share >= .5 and other[a:b].mean() <= 1.15 * fifth[a:b].mean():
            end = min(b / FPS, events[-1]["t1"])
            if end > a / FPS:
                replacements.append(dict(t0=a / FPS, t1=end, root=r, q="5", bass=r, conf=share))
    if not replacements:
        return events

    # Overlay measured runs on the beat-synced events, preserving their
    # original times outside the supported replacements.
    out = []
    remaining = iter(events)
    event = next(remaining, None)
    for replacement in replacements:
        a, b = replacement["t0"], replacement["t1"]
        while event is not None and event["t1"] <= a:
            out.append(event)
            event = next(remaining, None)
        if event is not None and event["t0"] < a:
            out.append(dict(event, t1=a))
        out.append(replacement)
        while event is not None and event["t1"] <= b:
            event = next(remaining, None)
        if event is not None and event["t0"] < b:
            event = dict(event, t0=b)
    if event is not None:
        out.append(event)
    out.extend(remaining)

    # The leftover beat-synced spans can include rests, a count-in or a solo
    # note. A third or seventh has to show up while the chord root is audible.
    for i, e in enumerate(out):
        r, q = e["root"], e["q"]
        if r is None or q == "5":
            continue
        a, b = round(e["t0"] * FPS), round(e["t1"] * FPS)
        s = slice(a, b)
        audible = (chroma[r, s] >= .45 * peak[s]) & (rms[s] > -35)
        needed = max(3, .10 * (b - a))
        tones = QUAL[q][0]
        if np.count_nonzero(audible & peaks[(r + tones[1]) % 12, s]) < needed:
            out[i] = dict(t0=e["t0"], t1=e["t1"], root=None, q=None)
        elif len(tones) == 4 and np.count_nonzero(audible & peaks[(r + tones[3]) % 12, s]) < needed:
            out[i] = dict(e, q="maj" if cls(q) == "M" else "min")
    return out


def blues_pitch_support(chroma, bass, root):
    """Require both thirds and a flat seventh over the same tracked bass pitch."""
    n = min(chroma.shape[1], len(bass["pc"]))
    if not n:
        return False
    chroma = chroma[:, :n]
    usable = (bass["pc"][:n] >= 0) & (bass["rel"][:n] >= .10)
    on_root = usable & (bass["pc"][:n] == root)
    if usable.mean() < .30 or on_root.sum() < .12 * usable.sum():
        return False
    flat7, raised7 = (root + 10) % 12, (root + 11) % 12
    if chroma[flat7].mean() < 1.05 * chroma[raised7].mean():
        return False

    # A pitch has to stand above both neighbours to count. Broad chroma
    # across adjacent bins is insufficient evidence for two different notes.
    peaks = chroma_peaks(chroma)[[(root + degree) % 12 for degree in (3, 4, 10, 11)]] & on_root
    counts = np.count_nonzero(peaks, axis=1)
    if (any(counts[:3] < .06 * on_root.sum()) or counts[2] < 2 * counts[3]):
        return False
    # All three peaks must recur in the same stretches of the recording.
    frames = max(1, int(20 * FPS))
    n_blocks = (n + frames - 1) // frames
    by_block = [np.bincount(np.flatnonzero(peak) // frames, minlength=n_blocks) for peak in peaks[:3]]
    return np.count_nonzero(np.min(by_block, axis=0) >= 5) >= 3


def blues_rock_home(events, home, verdict, chroma, bass, changes):
    """Return (blues pitch reference, competing home or None) on strong evidence."""
    if bass is None or not verdict.get("key") or len(verdict.get("top2", ())) < 2:
        return None
    tonic, mode = home
    runner_up = parse_key(verdict["top2"][1])
    cnn = verdict["voters"].get("cnn")
    if not changes and runner_up[0] == tonic:
        candidate = tonic
    elif (mode == "major" and runner_up == ((tonic + 7) % 12, "major")
          and cnn and parse_key(cnn) == (runner_up[0], "minor")):
        candidate = runner_up[0]
    else:
        return None

    chords = [e for e in events if e["root"] is not None and not e.get("band_out")]
    times, counts = Counter(), Counter()
    for e in chords:
        times[e["root"], e["q"]] += e["t1"] - e["t0"]
        counts[e["root"], e["q"]] += 1
    total = sum(times.values())
    seventh = (candidate + 10) % 12
    major = ("maj", "maj7", "dom7")
    if (sum(counts[seventh, q] for q in major) < 3 or
            sum(times[seventh, q] for q in major) < .08 * total or
            counts[candidate, "dom7"] < 3 or not blues_pitch_support(chroma, bass, candidate)):
        return None

    if candidate == tonic:
        # Sustained major and minor tonic chords distinguish this from a
        # Mixolydian passage with an occasional chromatic note in the melody.
        if (sum(times[tonic, q] for q in major) >= .20 * total and
                sum(times[tonic, q] for q in ("min", "min7")) >= .20 * total and
                modal_bass_home(bass, tonic)):
            return tonic, None
    elif (times[candidate, "dom7"] >= .05 * total
          and not any(modal_bass_home(bass, root) for root in (tonic, candidate))):
        # The rival's seventh chord could still be V7 of the working key.
        # Require a recurring move to its own flat VII and no such V7–I pattern.
        arrivals = Counter(b["root"] for a, b in zip(chords, chords[1:])
                           if a["root"] == candidate and a["q"] == "dom7" and cls(b["q"]) == "M"
                           and abs(a["t1"] - b["t0"]) <= .1)
        if arrivals[seventh] >= 3 and arrivals[tonic] < 2:
            return candidate, tonic
    return None


def harmonic_minor_note(events, chroma, tonic):
    """Name a raised seventh when recurring major V has direct pitch support."""
    import keyfuse as KF

    dominant = (tonic + 7) % 12
    lead, flat7 = (tonic + 11) % 12, (tonic + 10) % 12
    peaks = chroma_peaks(chroma)
    major_time, v_time = 0., 0.
    lead_frames, flat_frames, v_frames = 0, 0, 0
    blocks = set()
    for e in events:
        if e["root"] != dominant or e.get("band_out"):
            continue
        seconds = e["t1"] - e["t0"]
        start, end = round(e["t0"] * FPS), round(e["t1"] * FPS)
        lead_hits = int(peaks[lead, start:end].sum())
        flat_frames += int(peaks[flat7, start:end].sum())
        v_frames += peaks[lead, start:end].size
        v_time += seconds
        if e["q"] in ("maj", "dom7"):
            major_time += seconds
            lead_frames += lead_hits
            if lead_hits >= max(5, .12 * (end - start)):
                blocks.add(int(e["t0"] // 20))
    if (len(blocks) < 4 or major_time < .80 * v_time or lead_frames < .12 * v_frames
            or lead_frames < 2 * flat_frames):
        return None
    name = KF.key_name(tonic + 12)
    letters = "CDEFGAB"
    letter = letters[letters.index(name[0]) - 1]
    alteration = (lead - KF.NATURALS[letter] + 6) % 12 - 6
    return letter + "♯" * alteration + "♭" * -alteration


def modal_key_areas(kinfo, home, events):
    """Withhold a same-signature IV estimate when I7 crosses its boundary.

    An I7 that resolves to IV, or a confirmed band re-entry, keeps the original
    change. A withheld estimate never starts a new numeral or cadence span.
    """
    if home[1] != "mixolydian" or not kinfo.get("changes"):
        return kinfo
    changes, leans = [], []
    current = home
    fourth = ((home[0] + 5) % 12, "major")
    flat7 = (home[0] + 10) % 12
    chords = [e for e in events if e["root"] is not None]
    for i, c in enumerate(kinfo["changes"]):
        target = parse_key(c["to"])
        at = c["at"]
        end = kinfo["changes"][i + 1]["at"] if i + 1 < len(kinfo["changes"]) else float("inf")
        if (current == home and target == fourth and at >= 25 and not c.get("suppressed")
                and (c.get("evidence") or {}).get("source") != "confirmed band re-entry"
                and chords and chords[0]["root"] == chords[-1]["root"] == home[0]
                and any(e["root"] == flat7 and cls(e["q"]) == "M" and at < e["t1"] and e["t0"] < end
                        for e in chords)):
            tonic7 = [(a, b) for a, b in zip(chords, chords[1:]) if a["root"] == home[0]
                      and a["q"] == "dom7" and a["t1"] >= at - 5 and a["t0"] < end]
            if (any(a["t0"] <= at < a["t1"] for a, _ in tonic7)
                    and not any(b["root"] == fourth[0] and cls(b["q"]) == "M"
                                and abs(a["t1"] - b["t0"]) <= .1 for a, b in tonic7)):
                leans.append(c)
                continue
        changes.append(c)
        if not c.get("suppressed"):
            current = target
    return dict(kinfo, changes=changes, modal_leans=leans) if leans else kinfo


@ST.stage('key fusion')
def resolve_home(F, dur, notes, inputs=None):
    """The home key, decided in ONE place. Returns (kinfo, glob, evs, bpm, beats, swapped, verdict). `glob` is the working
    home the roman numerals are read against; `verdict` (keyfuse.verdict) is the claim every card line quotes: the key
    signature and the tonic as two parts, from spectral / CNN / chords / bass / melody / cadence voters."""
    import librosa
    kinfo = M.analyze_key(F["chroma"], dur)
    spec0 = dict(kinfo)
    tempo, beats = librosa.beat.beat_track(onset_envelope=F["oenv"], sr=SR, hop_length=HOP)
    bpm = float(np.atleast_1d(tempo)[0])
    glob = parse_key(kinfo["key"])
    try:
        evs = decode_chords(F, beats, glob, dur)
        verdict = fused_verdict(F, spec0, evs, inputs, dur)
        top = parse_key(verdict["working"])
        if top != glob:
            # the chord decoder leans on the home key (a diatonic prior): decode again under the fused key and vote again
            log(f"home key: spectral pick was {key_label(glob)}; the fused reading is {key_label(top)}, decoding the chords again under it")
            glob = top
            evs = decode_chords(F, beats, glob, dur)
            verdict = fused_verdict(F, spec0, evs, inputs, dur)
            glob = parse_key(verdict["working"])
        log("key voters: " + verdict["votes"] + f" -> {verdict['line']}")
    except Exception as e:
        ST.mark("key fusion", "degraded", f"{type(e).__name__}; using the chords-only reading")
        log(f"key fusion failed ({type(e).__name__}: {str(e)[:80]}); using the chords-only reading")
        return resolve_home_chords_only(F, dur, notes)
    verdict.pop("_L", None)
    kinfo = dict(kinfo)
    swapped = glob != parse_key(spec0["key"])
    if swapped:
        kinfo["key"] = key_label(glob)
        if M._area(KEYN_INDEX(glob)) != M._area(KEYN_INDEX(parse_key(spec0["key"]))):
            kinfo.pop("changes", None)
    kinfo = M.dominant_pull(kinfo, F["chroma"], dur, glob)
    modal = mixolydian_home(evs, glob, F.get("bass_trk")) if verdict.get("key") else None
    blues = None if modal else blues_rock_home(
        evs, glob, verdict, F["chroma"], F.get("bass_trk"), kinfo.get("changes"))
    if modal:
        import keyfuse as KF
        glob = modal
        name = key_label(glob)
        parent = (glob[0] + 5) % 12
        letter, alteration = KF.spell_pitch((glob[0] + 10) % 12, KF.key_short(parent))
        flat7 = letter + {-1: "♭", 0: "", 1: "♯"}[alteration]
        # These scores describe the original 24-key vote, with no modal option.
        verdict["major_minor_vote"] = {field: verdict.pop(field) for field in ("p", "top2", "sig_p", "relation")}
        verdict.update(state="clear", key=name, working=name, signature=KF.sig_text(parent),
                       line=f"{name} ({KF.key_name(parent)}'s notes, ♭7 = {flat7})",
                       votes_line=verdict.get("votes"))
        log(f"chord returns support {name}; reading numerals against that home")
        kinfo["key"] = name
        swapped = True
    elif blues:
        import keyfuse as KF
        root, other = blues
        name = key_label((root, "blues-rock"))
        tonic = name.split()[0]
        # Keep the original vote, including its signature, as evidence.
        verdict["major_minor_vote"] = {field: verdict.pop(field)
                                      for field in ("p", "top2", "sig_p", "signature", "relation")}
        verdict.update(blues_rock=True, signature=None, votes_line=verdict.get("votes"))
        if other is None:
            spelling = KF.spelling_key(f"{tonic} mixolydian")
            third_letter = "CDEFGAB"[("CDEFGAB".index(tonic[0]) + 2) % 7]
            flat3, third, flat7 = (
                KF.respell_note(M.NAMES[(root + step) % 12], spelling, letter)
                for step, letter in ((3, third_letter), (4, third_letter), (10, None)))
            verdict.update(state="clear", key=name, working=name,
                           line=f"{name} (♭7 = {flat7}, both thirds {flat3} and {third})")
            glob = root, "blues-rock"
            kinfo["key"] = name
            swapped = True
        else:
            alternative = verdict["working"].split()[0]
            verdict.update(state="split", key=None,
                           line=f"{tonic} or {alternative} (home uncertain) · blues-rock harmony")
            # Use the fifth-only evidence for names and time in this riff.
            # The fused tonic and mode keep their original votes.
            evs = power_chords(F, evs)
        log(verdict["line"])
    elif glob[1] == "minor" and verdict["state"] == "clear" and not spec0.get("changes"):
        lead = harmonic_minor_note(evs, F["chroma"], glob[0])
        if lead is not None:
            verdict["line"] = verdict["line"].replace(
                f" ({verdict['signature']})", f" (harmonic minor colour, major V with {lead})", 1)
    return kinfo, glob, evs, bpm, beats, swapped, verdict


def KEYN_INDEX(k):
    return k[0] % 12 + (0 if k[1] == "major" else 12)


@ST.stage('melody')
def load_melody(path, cp, y=None, use_cache=True):
    """The vocal melody track (pyin on the vocal stem when cached, else the full mix), cached beside the features."""
    vp_ = vocal_stem_path(path)
    msrc = "vocal stem" if vp_ else "full mix"
    mp = cp + (".melvox.pkl" if vp_ else ".mel.pkl")
    if use_cache and os.path.exists(mp):
        return pickle.load(open(mp, "rb"))
    if vp_:
        ym = load_vocal_22k(vp_)
    else:
        if y is None:
            import raven_ears_plus as E
            y = E.load_audio(path, SR)
        ym = y
    mel = compute_melody(ym, msrc)
    if use_cache:
        pickle.dump(mel, open(mp, "wb"))
    return mel


def key_claim_for(path, use_cache=True, features=None):
    """The key claim for this file (the verdict dict), from the cached features. music.py quotes it so the card says one thing."""
    cp = cache_path(path)
    F = None
    if features is not None:
        F = compute_features(path, shared=features)
        if use_cache:
            pickle.dump({k: v for k, v in F.items() if k != "y"}, open(cp + ".feat.pkl", "wb"))
    elif use_cache and os.path.exists(cp + ".feat.pkl"):
        ST.mark("features")
        F = pickle.load(open(cp + ".feat.pkl", "rb"))
    if F is None:
        F = compute_features(path)
        if use_cache:
            pickle.dump({k: v for k, v in F.items() if k != "y"}, open(cp + ".feat.pkl", "wb"))
    F = dict(F)  # leave the caller's raw RMS and waveform available
    y = F.pop("y", None)
    F["rms_db"] = np.maximum(F["rms_db"], F["rms_db"].max() - 60)
    inputs = gather_key_inputs(path, cp, F, use_cache, y)
    return resolve_home(F, F["y_len"] / SR, [], inputs)[-1]


def reentry_key_changes(path, kinfo, F, dur, home):
    """Refine a key boundary from existing four-stem energy; never separate audio."""
    if "chroma" not in F or not any(not c.get("suppressed") for c in kinfo.get("changes", [])):
        return kinfo
    vocal = vocal_stem_path(path)
    if not vocal:
        return kinfo
    import stops
    directory = os.path.dirname(vocal)
    if not all(os.path.isfile(os.path.join(directory, s + ".wav")) for s in stops.STEMS):
        return kinfo
    result = stops.analyse(directory)
    if result["status"] == "failed":
        ST.mark("key boundary", "failed", result["reason"])
    return M.reentry_changes(kinfo, F["chroma"], dur, home, result)


def late_key_areas(scan, kinfo, F, evs, dur, home):
    if scan is None or "late_runs" not in scan:
        return scan, kinfo
    import key_changes as KC

    claims = KC.late_signature_changes(scan, F["chroma"], F["rms_db"], evs, FPS, dur,
                                       kinfo.get("changes", []), home)
    scan = dict(scan)
    scan.pop("late_runs")
    if claims:
        scan["claims"] = claims
        areas = [dict(at=c["at"], from_key=c["from_key"], to=c["to"], range=c["range"]) for c in claims]
        kinfo = dict(kinfo, changes=kinfo.get("changes", []) + areas)
    return scan, kinfo


@ST.engine
def run(path, want_lyrics=True, use_cache=True, lyric_lines=None, key_changes=False):
    import librosa
    cp = cache_path(path)
    F = None
    if use_cache and os.path.exists(cp + ".feat.pkl"):
        ST.mark("features")
        F = pickle.load(open(cp + ".feat.pkl", "rb"))
        log("features from cache")
    if F is None:
        F = compute_features(path)
        if use_cache:
            pickle.dump({k: v for k, v in F.items() if k != "y"}, open(cp + ".feat.pkl", "wb"))
    y = F.pop("y", None)
    mix_rms_db = F["rms_db"]       # band activity uses the original round-1 level measure
    F["rms_db"] = np.maximum(F["rms_db"], F["rms_db"].max() - 60)
    dur = F["y_len"] / SR
    notes = []
    ST.progress("harmony and key")
    inputs = gather_key_inputs(path, cp, F, use_cache, y)
    kinfo, glob, evs, bpm, beats, swapped, verdict = resolve_home(F, dur, notes, inputs)
    change_scan = None
    if key_changes:
        ST.progress("key changes")
        import key_changes as KC
        change_scan = KC.scan(path, dur, cp if use_cache else None)
        ST.mark('key changes', change_scan['status'], change_scan.get('reason', ''))
        import keyfuse as KF
        verdict = KF.apply_section_split(verdict, (change_scan or {}).get('section_tonics'))
    kinfo = reentry_key_changes(path, kinfo, F, dur, glob)
    kinfo = modal_key_areas(kinfo, glob, evs)
    # HONEST CARD: establish home before reporting a departure from it.
    kinfo = key_opening.settled(kinfo, glob, parse_key)
    change_scan, kinfo = late_key_areas(change_scan, kinfo, F, evs, dur, glob)
    secs = sections_from_cache(F, dur)
    spans = key_spans(kinfo, glob)
    uncertain_blues = verdict.get("blues_rock") and not verdict.get("key")
    key_from = (spans[1][0] if len(spans) > 1 else dur) if uncertain_blues else 0.0
    # words
    words, lines, align = None, [], None
    lyrics_source = None
    if want_lyrics:
        ST.progress("aligning words with the harmony")
        # LRCLIB hook: external lines stay separate from the audio's word evidence.
        import lrclib
        lyr = lrclib.for_music(path, notes, dur, lyric_lines, use_cache=use_cache)
        words, lyrics_source = lyr["words"], lyr["lookup"]
        if lyr["rows"] is not None:
            lines = lyr["rows"]
            align = {"rows": lines, "summary": lyr["summary"], "label": lyr["label"]}
        elif words:
            lines = lines_with_ends(words)
        if not align:
            notes.append(lyr["label"])
    else:
        ST.mark("words", "skipped", "--no-lyrics", kind="input")
        ST.mark("LRCLIB", "skipped", "--no-lyrics", kind="input")
    # tension
    tc = tension_curve(F, evs, spans)
    br = builds_releases(tc, evs, spans, F)
    dips = find_dips(F, dur)
    drops = find_dropouts(F)
    swells = find_swells(F)
    ents, tnov = timbre_entrances(F, dur)
    moves = [m for m in notable_moves(evs, spans, dur) if m[0] >= key_from]
    pedals = find_pedals(evs)
    # melody (loaded for the key voters already; cached)
    mel = inputs.get("mel") or {"f0": [], "vp": [], "hop": HOP, "source": "unavailable"}
    if inputs.get("mel") is None:
        ST.mark("appoggiaturas", "skipped", "melody unavailable")
    appo = appoggiaturas(mel, evs) if inputs.get("mel") is not None else []
    nvl, nov = self_sim_novelty(F, dur)
    csurp = chord_surprises(evs, spans)
    lift, comps = music_lift_frames(F, tc, evs, dur)
    downbeats = inputs.get("downbeats")
    if downbeats is None:
        ST.mark("cadence downbeats", "degraded", "tracker unavailable; estimated phase with assumed four-beat bars")
        downbeats = find_downbeats(beats, F["oenv"])
    presence = band_presence(path, dur, mix_rms_db, downbeats, bpm, notes)
    cads = find_cadences(evs, spans, phrase_ends(secs, lines, dur), tc, downbeats, key_from)
    con = contrast(lines, lift, comps, dur) if lines else []
    if lines and os.environ.get("TYPESAFE_API_KEY"):
        import credits
        import jev_reads
        title = credits.identify(path, credits.CONFIG.get("artist"), credits.CONFIG.get("title"))[1]
        con = jev_reads.add_contrast(lines, con, title or "", notes)
    if not lines:
        ST.mark("word / music contrast", "skipped", "no lyric lines")
    tag = repeated_phrase_anomaly(words) if words else []
    refrain = refrain_check(path, words, dur) if words else {"phrase": None}
    if not words:
        ST.mark("refrain check", "skipped", "skipped in a quick listen" if
                (lyrics_source or {}).get("quick_skipped") else "no heard words")
    evs = mark_band_out(evs, band_out_stretches(dict(band_presence=presence)))
    return dict(F=F, dur=dur, kinfo=kinfo, glob=glob, bpm=bpm, secs=secs, evs=evs, spans=spans, lines=lines,
                band_presence=presence,
                words=words, tc=tc, br=br, dips=dips, drops=drops, swells=swells, ents=ents, moves=moves,
                pedals=pedals, appo=appo, nvl=nvl, csurp=csurp, con=con, tag=tag, notes=notes, lift=lift,
                comps=comps, cads=cads, swapped=swapped, key_verdict=verdict, path=path, align=align, mel_source=mel.get('source', 'full mix'), refrain=refrain,
                mel=mel, vocal_activity=lrclib.cached_activity(path) if want_lyrics else None,
                lyrics_source=lyrics_source, key_change_scan=change_scan)  # ACHE/LRCLIB hooks: existing evidence + word provenance.


class _FakeY:  # not used
    pass


def sections_from_cache(F, dur):
    """Reuse the same MFCCs without changing librosa's process-wide function."""
    return M.analyze_sections(None, F["chroma"], F["rms_db"], dur, mfcc=F["mfcc"])


# --------------------------------------------------------------------------- report
def sec_label(s, i, n):
    base = s["level"]
    if i == 0 and base != "loud":
        base += " intro"
    elif i == n - 1 and n > 2:
        base += " outro" if base != "loud" else " finish"      # same words as music.shape_phrase: a full-band last section is a finish
    return base


def endpoint(e, R):
    """The same ending/opening claim in CHORDS and the timeline."""
    home = roman(e["root"], e["q"], R["glob"])
    kv = R["key_verdict"]
    reference = "home" if kv["key"] else f"{kv['working']} working reference"
    text = f"{chord_name(e['root'], e['q'])}: {reference} {home}"
    local = key_at(R["spans"], (e["t0"] + e["t1"]) / 2)
    if local != R["glob"]:
        text += f" ({label(e, R['spans'])} of the {key_label(local)} area)"
    return text


def block_chords(R):
    evs, spans, secs, dur = R["evs"], R["spans"], R["secs"], R["dur"]
    L = ["CHORDS"]
    k0 = R["glob"]
    kinfo = R["kinfo"]
    kv = R["key_verdict"]
    uncertain_blues = kv.get("blues_rock") and not kv["key"]
    moved = any(k != k0 for _, k in spans)
    home_spans = [(0.0, k0)]
    chord_method = "chords decoded on beat-synced chroma, triads + sevenths, Viterbi-smoothed"
    has_power = any(e["q"] == "5" for e in evs)
    if has_power:
        chord_method += " with power chords where the root and fifth stand out"
    if kv["key"]:
        L.append(f"  home: {kv['line']}; ~{R['bpm']:.0f} BPM beat-tracker pulse; {chord_method}")
    else:
        L.append(f"  home: {kv['line']}. Numerals below are read against {kv['working']} as a working reference only; ~{R['bpm']:.0f} BPM "
                 f"beat-tracker pulse; {chord_method}")
    for c in kinfo.get("changes", []):
        timing = "somewhere" if "range" in c else "at"
        L.append(f"  key area changes {timing} {key_change_time(c)} → {key_change_label(c)}")
    for c in kinfo.get("modal_leans", []):
        L.append(f"  {tt(c['at'])} leans on the ♭VII (mixolydian)")
    if moved:
        if uncertain_blues:
            L.append("  section numerals use the working reference and the local key-area estimates above")
        else:
            reference = "home" if kv["key"] else f"the {kv['working']} working reference"
            L.append(f"  most time on and endpoints use {reference}; section numerals use the local key-area estimates above")
    opening = key_opening.dominant_opening(evs, k0, M.NAMES) if kv["key"] else None
    if opening:
        L.append("  " + opening)
    tot = Counter()
    for e in evs:
        if e["root"] is not None and not e.get("band_out"):
            name = chord_name(e["root"], e["q"]) if uncertain_blues else label(e, home_spans)
            tot[name] += e["t1"] - e["t0"]
    s = sum(tot.values())
    L.append("  most time on: " + ", ".join(f"{k} {100 * v / (s or 1):.0f}%" for k, v in tot.most_common(6)))
    if has_power and dur:
        L.append(f"  chord shares cover {100 * s / dur:.0f}% of the recording with clear enough voicings")
    for c in band_out_stretches(R):
        L.append("  " + band_out_line(c))
    if band_out_stretches(R):
        L.append("  most time on excludes these stretches; chord labels there have no band harmony behind them")
    if secs:
        for i, sc in enumerate(secs):
            pr = section_progression(evs, sc["start"], sc["end"], spans)
            head = f"  {tt(sc['start'])}–{tt(sc['end'])} [{sec_label(sc, i, len(secs))}]: "
            if not pr:
                L.append(head + "no clear chords")
                continue
            if "cycle" in pr:
                L.append(head + " – ".join(pr["cycle"]) + f"  (×{pr['reps']})")
            else:
                L.append(head + " – ".join(pr["seq"]) + (" …" if pr["more"] else ""))
            # tonic share
            seg = [e for e in evs if e["root"] is not None and e["t1"] > sc["start"] and e["t0"] < sc["end"]]
            ton = sum(min(e["t1"], sc["end"]) - max(e["t0"], sc["start"]) for e in seg
                      if e["root"] == k0[0])
            tl = sc["end"] - sc["start"]
            if tl >= 15 and ton / tl < 0.05 and not uncertain_blues:
                at_home = all(k == k0 for j, (a, k) in enumerate(spans)
                              if a < sc["end"] and (spans[j + 1][0] if j + 1 < len(spans) else dur) > sc["start"])
                if not moved or (at_home and kv["key"]):
                    L.append("      never lands on I in this section: it circles without coming home")
                else:
                    tonic = "i" if k0[1] == "minor" else "I"
                    reference = "home" if kv["key"] else f"{kv['working']} working-reference"
                    claim = "never lands on" if ton == 0 else "less than 5% of this section on"
                    # Clip at area boundaries: a sustained chord can cross a key estimate.
                    local = sum(max(0.0, min(e["t1"], sc["end"], spans[j + 1][0] if j + 1 < len(spans) else dur)
                                            - max(e["t0"], sc["start"], a))
                                for e in seg for j, (a, k) in enumerate(spans) if e["root"] == k[0])
                    L.append(f"      {claim} {reference} tonic {tonic}; local tonic time: {100 * local / tl:.0f}% of this section (key areas above)")
    ch = [e for e in evs if e["root"] is not None]
    if ch:
        if uncertain_blues:
            L.append(f"  ends on {chord_name(ch[-1]['root'], ch[-1]['q'])}; opens on {chord_name(ch[0]['root'], ch[0]['q'])}")
        elif moved:
            L.append(f"  ends on {endpoint(ch[-1], R)}; opens on {endpoint(ch[0], R)}")
        else:
            L.append(f"  ends on {label(ch[-1], spans)} ({chord_name(ch[-1]['root'], ch[-1]['q'])}); opens on {label(ch[0], spans)}")
    # notable moves, grouped
    grp = {}
    for t, kind, txt in R["moves"]:
        grp.setdefault((kind, txt), []).append(t)
    if grp:
        L.append("  notable moves:")
        for (kind, txt), ts in sorted(grp.items(), key=lambda kv: kv[1][0]):
            ex = ", ".join(tt(x) for x in ts[:5]) + (f" +{len(ts) - 5} more" if len(ts) > 5 else "")
            L.append(f"    {txt}  [{ex}]")
    for a, b, bp, nd in R["pedals"]:
        L.append(f"    pedal point on {PCN[bp]} in the bass {tt(a)}–{tt(b)} while {nd} different chords move above it")
    return L


def block_tension(R):
    tc = R["tc"]
    L = ["TENSION"]
    T = tc["T"]
    kv = R.get("key_verdict") or {}
    working = kv.get("blues_rock") and not kv.get("key")
    reference = "harmony from working key" if working else "harmony-from-home"
    L.append(f"  curve (0–1, per {tc['step']:.0f} s; {reference} 35%, chroma dissonance 20%, loudness 30%, activity 15%):")
    blocks = " ▁▂▃▄▅▆▇█"
    spark = "".join(blocks[int(np.clip(v, 0, 0.999) * 9)] for v in T)
    # print in lines of 30 bins (60 s)
    for i in range(0, len(spark), 30):
        L.append(f"    {tt(i * tc['step'])} {spark[i:i + 30]}")
    L.append(f"  range {T.min():.2f}–{T.max():.2f}, peak at {tt(int(np.argmax(T)) * tc['step'])}")
    if not R["br"]:
        L.append("  no large builds or releases")
    for b in R["br"]:
        L.append(("  BUILD   " if b["kind"] == "build" else "  RELEASE ") +
                 describe_build(b, R["evs"], R["spans"], tc, R["F"], working))
    return L


def block_resolves(R):
    """RESOLVES: do the phrase ends come home. Two lines at most; nothing when there is no phrase end to read."""
    cads, secs = R["cads"], R["secs"] or []
    kv = R.get("key_verdict") or {}
    uncertain_blues = kv.get("blues_rock") and not kv.get("key")
    called = [c for c in cads if c["cadence"] != "unclear"]
    if not called:
        if uncertain_blues:
            return ["RESOLVES: tonic arrivals unclear while home is uncertain."]
        return []
    home = [c for c in called if c["cadence"] in ("authentic", "plagal", "home")]
    kinds = Counter(c["cadence"] for c in home)
    parts = [f"{kinds[k]} {'other' if k == 'home' else k}" for k in ("authentic", "plagal", "home") if kinds[k]]
    arrival = "a local tonic" if uncertain_blues else "I"
    s = f"{len(home)} of {len(called)} phrase ends {'likely ' if any(c['likely'] for c in home) else ''}land on {arrival}"
    s += f" ({', '.join(parts)})." if parts else "."
    # the section ends that do not arrive home are the ones worth naming
    for c in called:
        if c["kind"] != "section" or c in home:
            continue
        i = next((j for j, x in enumerate(secs) if abs(x["end"] - c["t_end"]) < 0.5), None)
        who = f"The {sec_label(secs[i], i, len(secs))} section ending {tt(c['t_end'])}" if i is not None else f"The end at {tt(c['t_end'])}"
        lk = "likely " if c["likely"] else ""
        if c["cadence"] == "half":
            s += f" {who} is {lk}a half cadence on {c['to']} and hangs."
        elif c["cadence"] == "deceptive":
            s += f" {who} is {lk}deceptive ({c['from']} to {c['to']})."
        else:
            s += f" {who} stops on {c['to']}, no cadence I can name."
    blurry = len(cads) - len(called)
    if blurry:
        s += f" {blurry} too blurry to call."
    rel = [c for c in home if c["before"] is not None and c["after"] is not None and c["before"] - c["after"] >= 0.15]
    if rel:
        b = max(rel, key=lambda c: c["before"] - c["after"])
        s += f" Biggest release: {tt(b['t'])}, tension {b['before']:.2f} to {b['after']:.2f}."
    return ["RESOLVES: " + s]


def key_change_text(R, c, i):
    frm = parse_key(c["from_key"]) if "from_key" in c else R["spans"][i][1]
    to = parse_key(c["to"])
    t = c["at"]
    tc = R["tc"]
    k = int(t // tc["step"])
    before = float(np.mean(tc["T"][max(0, k - 4):k])) if k > 0 else 0
    after = float(np.mean(tc["T"][k:k + 4]))
    mv = "tension drops, it opens" if after < before - 0.05 else ("tension tightens" if after > before + 0.05 else "tension holds")
    return f"modulates {key_distance_words(frm, to)} to {key_change_label(c)}; {mv}"


def key_change_label(c):
    return c["to"] if "range" in c else key_label(parse_key(c["to"]))


def key_change_time(c):
    if "range" in c:
        from key_changes import clock_range
        return clock_range(c["range"])
    return tt(c["at"])


def build_timeline(R, cap=40):
    import lyrics_shared as LS
    ev = []  # (t, priority, text)
    band_events = [(c["t0"], 10, band_out_line(c)) for c in band_out_stretches(R)]
    secs, spans, evs, tc = R["secs"], R["spans"], R["evs"], R["tc"]
    kv = R.get("key_verdict") or {}
    working = kv.get("blues_rock") and not kv.get("key")
    # LRCLIB hook: supplied text/times cannot establish voice entrances or wordless gaps.
    lines = lines_with_ends(R.get("words") or []) if R.get("align") else R["lines"]
    placed = (R.get("align") or {}).get("rows")
    dur = R["dur"]
    if secs:
        for i, s in enumerate(secs):
            lab = sec_label(s, i, len(secs))
            txt = f"{'PEAK section' if s.get('peak') else 'new section'}: {lab}, {s['db']:.0f} dB"
            if i == 0:
                txt = f"opens {lab}, {s['db']:.0f} dB"
            ev.append((s["start"], 5 if i else 8, txt))
    for i, c in enumerate(R["kinfo"].get("changes", [])):
        if "range" not in c:
            ev.append((c["at"], 10, key_change_text(R, c, i)))
    for c in R["kinfo"].get("modal_leans", []):
        ev.append((c["at"], 10, "leans on the ♭VII (mixolydian)"))
    seen = Counter()
    for t, kind, txt in R["moves"]:
        seen[txt] += 1
        if kind == "authentic":
            continue
        if seen[txt] == 1:
            ev.append((t, 6, txt))
    for a, b, bp, nd in R["pedals"]:
        ev.append((a, 6, f"pedal on {PCN[bp]} in the bass until {tt(b)}; the chords move, the floor doesn't"))
    for b in R["br"]:
        txt = describe_build(b, evs, spans, tc, R["F"], working)
        ev.append((b["t0"], 7, txt))
    for d in R["dips"]:
        kind = "near-silence" if d["silent"] else "sudden dip"
        txt = f"{kind} for {d['t1'] - d['t0']:.1f} s, {d['depth']:.0f} dB under the song's own level; " + ("then it comes back" if d["returns"] else "it does not come back")
        ev.append((d["t0"], 9, txt))
    for d in R["drops"]:
        ev.append((d["t"], 6, f"drop-out: the level falls {d['drop']:.0f} dB and stays down; the band thins"))
    for s in R["swells"]:
        ev.append((s["t"], 6, f"swell: the sound grows {s['rise']:.0f} dB in a couple of seconds"))
    for c in R["con"]:
        if c.get("flag"):
            txt = f"line '{c['text'][:60]}': {c['flag']}"
            ev.append((c["t"], 7, LS.event_lyric_quote(txt, c["t"], placed)))
    # The singer's measured entrance and the gap detector never depend on Whisper.
    presence = R.get("band_presence") or {}
    onset = presence.get("vocal_onset")
    spans_at_entry = presence.get("vocal_spans") if onset is not None else R.get("vocal_activity")
    if spans_at_entry is None:
        spans_at_entry = presence.get("vocal_spans")
    if lines or placed:
        entry = voice_entry(lines or (placed if spans_at_entry is None else []),
                            R.get("words"), spans_at_entry, onset)
        if entry:
            t, entry_text = entry
            when = onset if onset is not None else min((a for a, _ in spans_at_entry or []), default=t)
            ev.append((t, 7, LS.event_lyric_quote(entry_text, when, placed)))

    # Lyrics near subsequent section starts.
    if lines:
        starts = [(s["start"]) for s in secs] if secs else []
        for s in starts[1:]:
            nxt = next((l for l in lines if l["t"] >= s - 0.5), None)
            if nxt and nxt["t"] - s < 12:
                txt = LS.event_lyric_quote(f"voice: '{nxt['text'][:60]}'", nxt["t"], placed)
                ev.append((nxt["t"], 4, txt))
    omitted = set()
    # A transcript gap is not a wordless passage. Check independent voice evidence.
    # Vocal-stem pitch can be leaked lead guitar (10 Nov 2026); it cannot establish voice.
    mel = R.get("mel") or {}
    if (not (R.get("lyrics_source") or {}).get("quick_skipped") and
            (R.get("words") is not None or lines or R.get("vocal_activity") is not None or mel.get("source") == "vocal stem")):
        for a, b in LS.word_gaps(R.get("words") if "words" in R else
                                [(l["text"], l["t"], l["end"]) for l in lines], dur):
            for start, end, description in LS.gap_segments(a, b, (R.get("vocal_activity"),),
                    words_available=R.get("words") is not None or "words" not in R, omitted=omitted):
                ev.append((start, 8, f"{LS.span_text(start, end, tt)}: {description}"))
    if omitted:
        ST.mark("gap edges", reason=LS.gap_edge_note(omitted), kind="input")
    for tg in R["tag"]:
        txt = f"last chorus: '{tg['phrase']}' comes {tg['now']} times where it came {tg['was']} before"
        ev.append((tg["t"], 8, LS.event_lyric_quote(txt, tg["t"], placed)))
    ch = [e for e in evs if e["root"] is not None]
    if ch:
        last = chord_name(ch[-1]["root"], ch[-1]["q"])
        if working:
            ending = last
        else:
            ending = endpoint(ch[-1], R) if len({k for _, k in spans}) > 1 else f"{label(ch[-1], spans)} ({last})"
        ev.append((ch[-1]["t0"], 4, f"last chord: {ending}"))
    # chords surprises
    for c in R["csurp"][:3]:
        ev.append((c["t"], 6, f"unexpected {c['got']} after {c['from']} (this song usually goes {c['from']}→{c['expected']})"))
    # instrument entrances/exits from instruments.py (cached next to the Demucs stems; skipped without stems)
    if not os.environ.get("RAVEN_NO_INSTRUMENTS"):
        try:
            import instruments
            # The singer pass owns credit-backed names and their source block.
            inst = instruments.analyse(R["path"], apply_credits=False)
            ST.collect(inst.get("stage_status", []) if inst else [])
            ST.mark("instruments", "ran" if inst else "skipped", "" if inst else "no stems available")
            for t, txt in instruments.events(inst):
                if t > 0:
                    ev.append((t, 6 if "[guitar stem level]" in txt else 7, txt))     # a stem-level step ranks above a section line (5) so the cap keeps it
        except Exception as e:
            ST.mark("instruments", "failed", f"{type(e).__name__}: {str(e)[:80]}")
    else:
        ST.mark("instruments", "skipped", "RAVEN_NO_INSTRUMENTS")
        ST.mark("credits", "skipped", "instrument naming disabled", kind="input")
    ev.sort(key=lambda x: x[0])
    # dedupe near-equal text/time and cap
    # merge items within 1.5 s with identical leading words
    # cap: None keeps everything (the lyric timeline wants the fuller list)
    if cap and len(ev) > cap:
        keep = sorted(ev, key=lambda x: (-x[1], x[0]))[:cap]
        ev = sorted(keep, key=lambda x: x[0])
    # Presence notices survive the ordinary timeline cap, once per stretch.
    return sorted(ev + band_events, key=lambda x: x[0])


def block_timeline(R):
    L = ["TIMELINE"]
    for t, p, txt in build_timeline(R):
        L.append(f"  {tt(t):>5}  {txt}")
    return L


def _line_chords(R, a, b):
    """Chord numerals sounding in [a, b], consecutive repeats collapsed, at most 4 shown."""
    seq = []
    for e in R["evs"]:
        if e["root"] is None or min(e["t1"], b) - max(e["t0"], a) < 0.5:
            continue
        lab = label(e, R["spans"])
        if not seq or seq[-1] != lab:
            seq.append(lab)
    if len(seq) > 4:
        seq = seq[:2] + ["…"] + seq[-1:]
    return "→".join(seq)


def _line_tension(R, a, b):
    """'tension falls 0.71→0.22 (the release)' when it moves 0.15+ over the line, else ''."""
    tc = R["tc"]
    T, step = tc["T"], tc["step"]
    if not len(T):
        return ""
    ia = min(int(a // step), len(T) - 1)
    ib = min(int(max(b, a + step) // step), len(T) - 1)
    v0, v1 = float(T[ia]), float(T[ib])
    if abs(v1 - v0) < 0.15:
        return ""
    txt = f"tension {'rises' if v1 > v0 else 'falls'} {v0:.2f}→{v1:.2f}"
    if v1 < v0 and any(c.get("cadence") in ("authentic", "plagal", "home") and a - 0.5 <= c["t"] <= b + 1.0
                       and c.get("before") is not None and c.get("after") is not None and c["before"] - c["after"] >= 0.15
                       for c in R["cads"]):
        txt += " (the release)"
    return txt


def block_lyric_timeline(R):
    """With --lyrics: the writer's lines, each at its time (≈ when placed rather than heard), beside what the music
    is doing there. Lyric-derived events and transcript gaps stay on the main timeline."""
    al = R.get("align")
    if not al:
        return []
    import lyric_align as LA
    rows = al["rows"]
    L = ["LYRIC TIMELINE", "  " + (al.get("label") or LA.summary_line(al["summary"]))]  # LRCLIB provenance.
    skip = ("voice enters:", "first timed lyric (voice onset unclear):", "voice:", "line '")
    import re as _re
    evs = [(t, p, txt) for t, p, txt in build_timeline(R, cap=None) if not txt.startswith(skip)
           and not _re.match(r"\d+:\d\d–\d+:\d\d:", txt)  # transcript gaps already have a timeline row
           and not (_re.match(r"sudden dip for 0\.\d s", txt))]         # a sub-second dip is a blip, not a feeling
    # a contrast flag reads as music-vs-words colour; keep it, minus the line quote
    for c in R["con"]:
        if c.get("flag"):
            evs.append((c["t"], 7, c["flag"]))
    evs.sort(key=lambda x: x[0])
    own = [[] for _ in rows]       # events that ride on a line
    free = []                      # events in an instrumental stretch
    for t, p, txt in evs:
        k = next((i for i, r in enumerate(rows) if r["t"] <= t < r["end"] + 0.05), None)
        if k is None:
            nxt = next((i for i, r in enumerate(rows) if r["t"] > t), None)
            prev_end = rows[nxt - 1]["end"] if nxt else 0.0
            if nxt is not None and rows[nxt]["t"] - t < 4.0 and (nxt == 0 or rows[nxt]["t"] - prev_end < 4.0 or rows[nxt]["t"] - t < 2.0):
                k = nxt            # just before the voice: rides on the line it leads into
        if k is None:
            free.append((t, p, txt))
        else:
            own[k].append((p, t, txt))
    out = []
    for t, p, txt in free:
        out.append((t, f"  [{tt(t)}] ♪ {txt}"))
    for r, ev in zip(rows, own):
        bits = []
        ch = _line_chords(R, r["t"], r["end"])
        if ch:
            bits.append(f"chord {ch}")
        tn = _line_tension(R, r["t"], r["end"])
        if tn:
            bits.append(tn)
        bits += [txt for p, t, txt in sorted(sorted(ev, key=lambda x: -x[0])[:2], key=lambda x: x[1])]
        mark = "≈" if r["approx"] else ""
        quiet = " (no voice measured here)" if r.get("no_voice") else ""
        out.append((r["t"], f"  [{tt(r['t'])}{mark}] {LA.quote_line(r['text'])}{quiet}" + (" — " + ", ".join(bits) if bits else "")))
    out.sort(key=lambda x: x[0])
    return L + [x[1] for x in out]


def block_contrast(R):
    return jev_reads.block_contrast(R, tt)


def block_surprise(R):
    from lyrics_shared import event_lyric_quote

    L = ["SURPRISE (where the song breaks its own pattern)"]
    placed = (R.get("align") or {}).get("rows")
    items = []
    secs = R["secs"] or []
    bounds = [s["start"] for s in secs[1:]]
    for t, nv in sorted(R["nvl"], key=lambda x: -x[1])[:6]:
        nearb = [b for b in bounds if abs(b - t) <= 3]
        rms = R["F"]["rms_db"]
        a = rms[int(max(t - 6, 0) * FPS):int(t * FPS)].mean(); b = rms[int(t * FPS):int((t + 6) * FPS)].mean()
        dl = b - a
        e0 = chord_at(R["evs"], max(t - 1, 0)); e1 = chord_at(R["evs"], t + 1)
        harm = ""
        if e0 and e1:
            harm = f", chords {label(e0, R['spans'])}→{label(e1, R['spans'])}"
        kind = "a section break (expected scaffolding)" if nearb else "an unscheduled change, not on a section line"
        items.append((t, 5 + (0 if nearb else 3), f"{tt(t)} novelty peak, the song's texture (chroma+timbre+level) reshuffles ({dl:+.0f} dB across it{harm}): {kind}"))
    for c in R["csurp"][:4]:
        extra = "outside the key" if c["chromatic"] else "inside the key but off the path"
        items.append((c["t"], 7, f"{tt(c['t'])} {c['got']} after {c['from']} ({extra}); this song has gone {c['from']}→{c['expected']} {c['n_expected']} times and takes this turn once"))
    for d in R["dips"]:
        if d["returns"]:
            items.append((d["t0"], 8, f"{tt(d['t0'])} a stop: the sound falls away for {d['t1'] - d['t0']:.1f} s ({d['depth']:.0f} dB under its own level) and then resumes"))
    for k, c in enumerate(R["kinfo"].get("changes", [])):
        items.append((c["at"], 9, f"{key_change_time(c)} the key itself moves ({key_change_text(R, c, k)}); the home the song taught is left"))
    rf = R.get("refrain") or {}
    stem_sc = rf.get("stem_counts") or []
    stem_txt = ""
    if stem_sc:
        stem_txt = (f"; the vocal-stem template match (audio, whether or not Whisper wrote it down) finds "
                    f"{stem_sc} per burst")
    for tg in R["tag"]:
        items.append((tg["t"], 9, f"{tt(tg['t'])} the last chorus breaks the count: '{tg['phrase']}' arrives {tg['now']} times where earlier choruses gave {tg['was']} (counts per chorus: {tg['counts']}){stem_txt}"))
    stem_flagged = False
    if len(stem_sc) >= 3:
        prev = stem_sc[:-1]
        mode = Counter(prev).most_common(1)[0][0]
        if stem_sc[-1] != mode and mode >= 2 and not R["tag"]:
            items.append((rf["stem_t"][-1], 9, f"{tt(rf['stem_t'][-1])} the last chorus breaks the count in the AUDIO: the vocal stem has '{rf['phrase']}' "
                          f"{stem_sc[-1]} times where earlier choruses have {mode} (stem counts {stem_sc}; Whisper wrote {rf['text_counts']})"))
            stem_flagged = True
        for tb, nt, ns in rf.get("aligned", []):
            if ns > nt >= 1 and not stem_flagged:
                items.append((tb, 8, f"{tt(tb)} '{rf['phrase']}': the voice sings it {ns} times here, the transcript only wrote {nt} (sung but blurred in the words)"))
    for st in R_STABLE:
        agree = ""
        if stem_sc:
            same = all(c == st["counts"][0] for c in stem_sc)
            agree = (f" The vocal stem agrees ({stem_sc[0]} per burst in all {len(stem_sc)} bursts, template match at {tt(rf['stem_t'][-1])} too)."
                     if same else f" The vocal stem counts {stem_sc}, which does not line up with the transcript, so the audio count is not reliable for this phrase.")
        elif rf.get("stem"):
            agree = f" (audio check unavailable: {rf['stem']})"
        items.append((st["t"], 8, f"{tt(st['t'])} refrain check: '{st['phrase']}' comes in bursts of {st['counts'][0]} every time ({len(st['counts'])} bursts) as Whisper hears it, including the last chorus.{agree}"))
    # final section differs? compare last section vs the loudest earlier similar one by chord cycle
    if secs and len(secs) >= 3:
        last = secs[-1]
        prl = section_progression(R["evs"], last["start"], last["end"], R["spans"])
        if prl and "cycle" in prl and not any(s for s in []):
            pass
    items.sort(key=lambda x: x[0])
    top = sorted(items, key=lambda x: -x[1])[:10]
    for t, p, txt in sorted(top, key=lambda x: x[0]):
        stamp, _, description = txt.partition(" ")
        L.append(f"  {stamp} {event_lyric_quote(description, t, placed)}")
    if not top:
        L.append("  the song keeps its own pattern throughout")
    return L


def block_chills(R):
    L = ["CHILL CANDIDATES (Sloboda 1991; candidates, not verdicts)"]
    ap = R["appo"]
    if ap:
        L.append("  appoggiaturas (melody leans on a non-chord tone, then steps onto a chord tone; "
                 f"from a pyin pitch track of the {R['mel_source']}{', cleaner than the mix' if R['mel_source'] == 'vocal stem' else ', approximate'}; treat as hints):")
        for a in sorted(ap[:5], key=lambda r: r["t"]):
            L.append(f"    {tt(a['t'])}  {a['note']} held {a['dur']:.1f} s over {a['chord']}, then {'down' if a['down'] else 'up'} onto {a['to']}")
    else:
        L.append("  appoggiaturas: melody unavailable; not checked" if R.get("mel_source") == "unavailable" else
                 "  appoggiaturas: none found in the pitch track (or the melody track was too unsteady to say)")
    cs = [c for c in R["csurp"] if c["chromatic"]] + [c for c in R["csurp"] if not c["chromatic"]]
    mv = [(t, txt) for t, k, txt in R["moves"] if k in ("deceptive",) or (k == "borrowed" and txt.startswith("borrowed"))]
    first = {}
    for t, txt in mv:
        first.setdefault(txt, t)
    L.append("  new or unexpected harmony:")
    n = 0
    for c in cs[:3]:
        L.append(f"    {tt(c['t'])}  {c['got']} after {c['from']} ({'chromatic' if c['chromatic'] else 'diatonic detour'})")
        n += 1
    for txt, t in sorted(first.items(), key=lambda kv: kv[1])[:4]:
        L.append(f"    {tt(t)}  {txt}")
        n += 1
    for k, c in enumerate(R["kinfo"].get("changes", [])):
        L.append(f"    {key_change_time(c)}  {key_change_text(R, c, k)}")
        n += 1
    if not n:
        L.append("    none standing out")
    L.append("  sudden dynamic swells:")
    if R["swells"]:
        for s in R["swells"][:5]:
            L.append(f"    {tt(s['t'])}  +{s['rise']:.0f} dB over about 2.5 s")
    else:
        L.append("    none (no jump of 9 dB or more in 2.5 s)")
    L.append("  new voice or instrument entering (timbre-novelty proxy; the stem limb has the true entrances):")
    es = [e for e in R["ents"] if e["dlevel"] > -2][:5]
    if es:
        for e in es:
            L.append(f"    {tt(e['t'])}  timbre shifts sharply ({e['dlevel']:+.0f} dB level change)")
    else:
        L.append("    no clear candidates")
    return L


def block_sound(R):
    F = R["F"]
    cent = float(np.median(F["cent"][F["rms_db"][:len(F["cent"])] > F["rms_db"].max() - 40]))
    body, top = F["body"] * 100, F["top"] * 100
    L = ["SOUND (two separate things, kept apart on purpose)"]
    L.append(f"  low-mid BODY (150–500 Hz): {body:.0f}% of the energy · TOP (above 4 kHz): {top:.1f}% · centroid {cent:.0f} Hz")
    return L


@ST.engine
def report(R):
    out = [f"HARMONY: {os.path.basename(R['path'])} · {tt(R['dur'])}"]
    for n in R["notes"]:
        out.append(f"  NOTE: {n}")
    out.append("")
    for blk in (block_sound, block_chords, block_tension, block_resolves, block_timeline, block_lyric_timeline, block_contrast, block_surprise, block_chills):
        out += blk(R)
        out.append("")
    # ACHE hook: one self-contained block from existing evidence.
    import ache
    out += ache.block(R)
    from ears_handoff import finish_lines  # MORE EARS: leave the feeling to the listener.
    # Jev already marks exact outcomes; reparsing its notes would invent a second stage.
    notes = [n for n in R.get("notes", []) if not n.startswith("Jev")]
    rows = ST.merge(R.get("stage_status", []), ST.snapshot(), ST.engine_notes(notes, "harmony"))
    import instrumental_voice
    text = instrumental_voice.stem_text("\n".join(ST.add_notes(out, rows)).rstrip(),
                                       instrumental_voice.designated(R))
    import key_changes
    import keyfuse
    import song_length
    at = keyfuse.spelling_context(keyfuse.claimed_spelling_key(R.get("key_verdict")),
                                 keyfuse.spelling_spans(R.get("key_change_scan")))
    text = keyfuse.harmony_note_labels(text, at)
    return song_length.annotate(key_changes.card_text(text, R.get('key_change_scan')), R.get("length_check"))


@ST.engine
def to_json(R):
    d = {"file": os.path.basename(R["path"]), "duration": round(R["dur"], 1), "home_key": R["key_verdict"]["key"] or "unclear", "key_verdict": R["key_verdict"],
         "chords": [{"t0": round(e["t0"], 2), "t1": round(e["t1"], 2),
                     "chord": chord_name(e["root"], e["q"]) if e["root"] is not None else None,
                     "roman": label(e, R["spans"]) if e["root"] is not None else None,
                     "band_out": bool(e.get("band_out"))} for e in R["evs"]],
         "band_presence": R.get("band_presence", {"status": "unavailable", "candidates": []}),
         "tension": {"step": R["tc"]["step"], "values": [round(float(x), 3) for x in R["tc"]["T"]]},
         "builds_releases": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in b.items() if k != "comp"} for b in R["br"]],
         "timeline": [{"t": round(t, 1), "text": txt} for t, p, txt in build_timeline(R)],
         "cadences": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()} for c in R["cads"]],
         "contrast": R["con"], "notes": R["notes"]}
    if R.get("filename_reading") is not None:
        d["filename_reading"] = R["filename_reading"]
    if R.get("song_extras"):
        d["song_extras"] = R["song_extras"]
    for field in ("studio_durations", "length_check"):
        if R.get(field):
            d[field] = R[field]
    if R.get("kinfo", {}).get("raw_changes"):
        d["key_changes"] = R["kinfo"]["raw_changes"]
    if R.get("kinfo", {}).get("modal_leans"):
        d["modal_leans"] = R["kinfo"]["modal_leans"]
    if R.get('key_change_scan') is not None:
        d['key_change_scan'] = R['key_change_scan']
    if R.get("align"):
        d["lyrics_aligned"] = {"summary": R["align"]["summary"], "lines": R["align"]["rows"]}
        d["lyrics_aligned"]["label"] = R["align"].get("label")  # LRCLIB provenance.
    d["lyrics_source"] = R.get("lyrics_source")
    # ACHE hook: save the evidence and ranking, without exporting a raw pitch track.
    import ache
    try:
        d["ache"] = ache.analyze(R)
    except Exception as e:
        ST.mark("ache", "failed", f"{type(e).__name__}: {str(e)[:120]}")
        d["ache"] = {"status": "unavailable", "moments": [], "notes": [f"analysis failed ({type(e).__name__}: {e})"]}
    from ears_handoff import finish_json  # MORE EARS: handoff inside valid JSON.
    d["stage_status"] = ST.merge(R.get("stage_status", []), ST.snapshot(), d["ache"].get("stage_status", []))
    return finish_json(d)


@ST.engine
@ST.cli
def main(argv=None, capture=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-lyrics", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    keys = ap.add_mutually_exclusive_group()
    keys.add_argument("--key-changes", action="store_true", default=True,
                      help="scan key changes with sequential CPU CNN slices (default on; off with --quick)")
    keys.add_argument("--no-key-changes", dest="key_changes", action="store_false", help="skip the key-change scan")
    ap.add_argument("--no-instruments", action="store_true", help="leave instrument names off the timeline")
    ap.add_argument("--lyrics", metavar="FILE", help="the song's real words, one sung line per line; timed against the audio")
    ap.add_argument("--names", metavar="FILE", help="names/rare words for Whisper, one per line (default RAVEN_EARS_HOME/names.txt)")
    import credits
    credits.add_args(ap)
    import lrclib                          # LRCLIB hook: reuse --artist/--title from credits.
    lrclib.add_args(ap)
    import song_extras
    song_extras.add_args(ap)
    a = ap.parse_args(argv)
    from network_policy import local_audio_path
    local_audio_path(a.file, ap)
    extras = song_extras.prepare(a, a.file)
    credits.configure_from_args(a)
    lrclib.configure_from_args(a)
    if a.no_instruments:
        os.environ["RAVEN_NO_INSTRUMENTS"] = "1"
    import lyrics_shared as LS
    lyric_lines = None
    if a.lyrics:
        import lyric_align
        lyric_lines = lyric_align.read_lyrics_file(a.lyrics)
    LS.configure_for_run(a.names, lyric_lines)
    scan_options = dict(key_changes=a.key_changes and not a.quick)
    R = run(a.file, want_lyrics=not a.no_lyrics, use_cache=not a.no_cache, lyric_lines=lyric_lines, **scan_options)
    credits.save_filename_reading(a.file, R, R.get("notes", []))
    import song_length
    song_length.capture(a.file, R, R.get("dur"))
    flag_contrast(R)
    if extras:
        R["song_extras"] = extras
    if a.json:
        print(json.dumps(to_json(R), indent=2, default=float, ensure_ascii=False))
    else:
        print(song_extras.decorate(report(R), extras))
    if capture is not None:
        capture(to_json(R))


if __name__ == "__main__":
    sys.exit(main())
