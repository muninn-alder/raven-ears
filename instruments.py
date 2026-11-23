#!/usr/bin/env python
"""instruments.py: music ears, item 3 (1 Nov 2026). `instruments.py <file> [--json] [--no-cache]`

WHAT is playing, and when it comes in. Reuses the PANNs CNN14 AudioSet tagger that raven_ears_plus already loads
(nothing new installed, local only) and the cached Demucs stems (singer.py makes them, ~/.cache/raven-stems/<hash>/).
The tagger is run on each stem separately, because a full mix smears everything together:
  other.wav   -> the melodic instruments (guitars, strings, piano, organ, synth, brass, woodwinds, harp, bells, choir)
  drums.wav   -> kit / percussion
  bass.wav    -> bass guitar / double bass / synth bass
  vocals.wav  -> only used for "+ voice" (energy), not tagged.
With the six-stem model (htdemucs_6s, cached in <hash>-6s) guitar.wav and piano.wav are tagged too, as extra evidence beside the
four-stem 'other' (which still holds everything melodic, so nothing the four-stem run heard is lost); the six-stem 'other'
(guitar and piano taken out) is what the synth/pad shape test listens to. Drums, bass and vocals come from the four-stem model.
If the six-stem run fails the four-stem path runs as before. The piano stem has bleed/artifacts (demucs' own README says so),
so a piano that is weak or quieter than the rest is named "piano (weak, may be bleed)".
AudioSet has many near-duplicate classes, so labels are collapsed into families (see FAMILIES). A family counts as
PLAYING in a window only if its tag score clears a threshold AND the stem has real level there (so Demucs bleed in a
silent stem is not named). Entrances/exits are debounced like singer.py's. Low confidence is reported as such,
and a family that never clears the bar is simply not named; a stem that is loud but unnamed is called "unclear".
A tagger gives an ESTIMATE from a 527-class vocabulary; it names the nearest thing it knows, so read "likely" as likely.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SR = 32000                  # PANNs rate
WIN, HOP = 3.0, 1.0         # seconds; entrance times are good to about a second
VERSION = 6

# family -> (stem it is looked for in, [(AudioSet label, weight)], threshold). Score = max(weight * prob).
# Generic labels (Guitar, Plucked string instrument) carry less weight than the specific ones.
FAMILIES = {
    # one "guitar" family for presence (generic and specific labels together, so it does not flicker); it is
    # named afterwards from which specific label dominates while it is on (see GUITAR_NAMES)
    "guitar": ("other", [("Guitar", 1.0), ("Plucked string instrument", 0.9), ("Acoustic guitar", 1.0), ("Strum", 0.9),
                         ("Electric guitar", 1.0), ("Steel guitar, slide guitar", 1.0)], 0.30),
    "strings": ("other", [("String section", 1.0), ("Orchestra", 0.9), ("Violin, fiddle", 1.0), ("Cello", 1.0),
                          ("Bowed string instrument", 0.9), ("Double bass", 0.7), ("Pizzicato", 0.7)], 0.30),
    "piano": ("other", [("Piano", 1.0), ("Electric piano", 0.9), ("Keyboard (musical)", 0.6)], 0.30),
    "organ": ("other", [("Organ", 1.0), ("Hammond organ", 1.0), ("Electronic organ", 1.0)], 0.35),
    "harpsichord": ("other", [("Harpsichord", 1.0)], 0.40),
    "synth/pad": ("other", [("Synthesizer", 1.0), ("Sampler", 0.7), ("Ambient music", 0.5), ("New-age music", 0.5)], 0.35),
    "harp": ("other", [("Harp", 1.0)], 0.35),
    "banjo/mandolin/ukulele": ("other", [("Banjo", 1.0), ("Mandolin", 1.0), ("Ukulele", 1.0), ("Sitar", 1.0), ("Zither", 0.8)], 0.40),
    "brass": ("other", [("Brass instrument", 1.0), ("French horn", 1.0), ("Trumpet", 1.0), ("Trombone", 1.0)], 0.40),
    "woodwinds": ("other", [("Flute", 1.0), ("Clarinet", 1.0), ("Oboe", 1.0), ("Saxophone", 1.0),
                            ("Wind instrument, woodwind instrument", 0.9), ("Harmonica", 0.8), ("Accordion", 0.8)], 0.40),
    "bells/mallets": ("other", [("Glockenspiel", 1.0), ("Vibraphone", 1.0), ("Marimba, xylophone", 1.0), ("Bell", 0.8),
                                ("Music box", 1.0), ("Chime", 0.8), ("Steelpan", 1.0)], 0.40),
    "choir": ("other", [("Choir", 1.0), ("Chant", 0.6), ("Mantra", 0.6)], 0.40),
    "drum kit": ("drums", [("Drum kit", 1.0), ("Snare drum", 0.9), ("Bass drum", 0.9), ("Hi-hat", 0.9), ("Cymbal", 0.9),
                           ("Drum", 0.8), ("Tom-tom", 0.9)], 0.30),
    "percussion (hand/small)": ("drums", [("Tambourine", 1.0), ("Maraca", 1.0), ("Shaker", 1.0), ("Bongo", 1.0), ("Conga", 1.0),
                                          ("Tabla", 1.0), ("Clapping", 0.8), ("Finger snapping", 0.8), ("Percussion", 0.6)], 0.40),
    "drum machine": ("drums", [("Drum machine", 1.0), ("Electronic drum", 1.0)], 0.35),
    "bass guitar": ("bass", [("Bass guitar", 1.0)], 0.30),
    "double bass": ("bass", [("Double bass", 1.0), ("Cello", 0.8)], 0.35),
    "bass (synth-like)": ("bass", [("Synthesizer", 1.0), ("Sampler", 0.7)], 0.35),
}
# keyboard family: AudioSet classes grouped by what they are; when the leader is not clearly ahead the card says "keys (piano or organ?)"
KEYS_CLASSES = {"piano": ["Piano", "Electric piano", "Keyboard (musical)"],
                "organ": ["Organ", "Electronic organ", "Hammond organ"],
                "synth": ["Synthesizer"],
                "harpsichord": ["Harpsichord"]}
KEYS_FAMS = ("piano", "organ", "harpsichord")      # FAMILIES entries that merge into one keys family when it is unclear who is who
KEYS_LEAD = 1.5                                    # the top keyboard class must beat the runner-up by this factor to be named alone
KEYS_MIN_RIVAL = 0.12                              # a runner-up under this mean score is no rival
PIANO_CARRIES_DB = 6.0                             # piano stem within this of the other stem (or louder) = it carries the keys
PIANO_WEAK_DB = 15.0                               # piano stem this far under the other stem = bleed-like
PIANO_WEAK_CONF = 0.35
MELODIC = ("other", "guitar", "piano")
EXTRA_STEMS = {"guitar": "guitar", "piano": "piano"}   # six-stem: family -> the extra stem it is ALSO looked for in (best score wins)
ORGAN_SOLO_SCORE = 0.50                            # 'organ' alone needs this mean score; the tagger calls sustained piano chords and pads organ
EMPTY_STEM_DB = 25.0                               # a 6s guitar/piano stem this far under the loudest melodic stem is an empty residue
# timbre rules for the bass stem and for the unnamed 'other' stem; first guesses, set from 4 songs' bass stems (real bass guitars read
# attack 4.1-4.2, synth-like basses 1.9-2.1) and from drum/guitar/piano stems as negatives (see README)
BASS_SMOOTH_ATTACK, BASS_SMOOTH_SWING, BASS_FUND = 3.0, 8.0, 0.8
BASS_PLUCK_ATTACK, BASS_PLUCK_SWING = 4.0, 6.0
PAD_PERC, PAD_ATTACK, PAD_SWING = 0.20, 5.0, 9.0       # shape gate: not percussive, no steep attacks, no wide swings
PAD_HINT = 0.07                                          # ...and the tagger's own faint Synthesizer/pad score over those windows
GUITAR_SPECIFIC = {"acoustic guitar": [("Acoustic guitar", 1.0), ("Strum", 0.9)], "electric guitar": [("Electric guitar", 1.0)],
                   "slide/steel guitar": [("Steel guitar, slide guitar", 1.0)]}
# families that are the same player seen two ways: only the winner is kept per window
EXCLUSIVE = [("bass guitar", "double bass", "bass (synth-like)"),
             ("drum kit", "drum machine")]
STEM_DB_GATE = 24.0          # a window more than this far under the stem's own 90th-percentile level is bleed
MIN_ON_S = 5.0               # an instrument must hold for this long to count
CONF_EVENT = 0.35            # mean score while on, needed to name an entrance/exit on the timeline


def _log(m):
    print(f"[instruments] {m}", file=sys.stderr, flush=True)


def tm(s):
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def _stem_dir(path):
    import singer
    d = os.path.join(singer.CACHE, singer.file_hash(path))
    ok = os.path.isdir(d) and all(os.path.exists(os.path.join(d, f"{s}.wav")) for s in singer.SRC)
    return d, ok


def _window_rms(y):
    W, H = int(WIN * SR), int(HOP * SR)
    if len(y) < W:
        y = np.pad(y, (0, W - len(y)))
    starts = list(range(0, len(y) - W + 1, H))
    rms = np.array([20 * np.log10(np.sqrt(np.mean(y[s:s + W] ** 2)) + 1e-7) for s in starts])
    return y, starts, W, rms


def _tag(y, model, device, torch):
    """probs [nW, 527] and window rms dB, for 3 s windows every 1 s."""
    y, starts, W, rms = _window_rms(y)
    out = []
    for i in range(0, len(starts), 16):
        x = np.stack([y[s:s + W] for s in starts[i:i + 16]]).astype(np.float32)
        with torch.no_grad():
            out.append(model(torch.from_numpy(x).to(device))["clipwise_output"].float().cpu().numpy())
    return np.concatenate(out, 0), rms


def _runs(mask, hop):
    r, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            r.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    return r


def _debounce(score, thr, hop, min_s):
    """hysteresis: on above thr, off below 0.6*thr; close gaps up to 8 s (a held note or a breath is not an exit); drop runs shorter than min_s."""
    on = np.zeros(len(score), bool)
    state = False
    for i, v in enumerate(score):
        state = (v >= thr) if not state else (v >= 0.6 * thr)
        on[i] = state
    k = int(8.0 / hop)
    for a, b in _runs(~on, 1):
        if a > 0 and b < len(on) and b - a <= k:
            on[a:b] = True
    for a, b in _runs(on, 1):
        if (b - a) * hop < min_s:
            on[a:b] = False
    return on


def _name_guitar(p, ix, on, g, isolated=False):
    """Which kind of guitar-like sound: the specific label that leads while the family is on, else say it is unclear."""
    m = on & g
    mean = lambda labs: float(max(w * p[m][:, ix[l]].mean() for l, w in labs if l in ix))
    gen = mean([("Guitar", 1.0), ("Plucked string instrument", 0.9)])
    spec = {k: mean(v) for k, v in GUITAR_SPECIFIC.items()}
    best = max(spec, key=spec.get)
    if spec[best] >= 0.25 * gen and spec[best] >= 0.08:
        return best + " (probably)" if spec[best] < 0.5 * gen else best
    return "guitar (type unclear)" if isolated else "plucked strings (guitar-like, type unclear)"   # isolated = its own guitar stem


# ---------------------------------------------------------------------------- keyboards, bass, pad: judgement helpers
def keys_class_means(probs_by_stem, ix, on):
    """mean tag score per keyboard class (piano / organ / synth / harpsichord) over the windows in `on`, best stem per class."""
    out = {}
    for cls, labs in KEYS_CLASSES.items():
        best = 0.0
        for p in probs_by_stem:
            if len(on) and on.any():
                best = max(best, max(float(p[:len(on)][on][:, ix[l]].mean()) for l in labs if l in ix))
        out[cls] = round(best, 3)
    return out


def keys_verdict(means):
    """(name or None, leaders). Names the top keyboard class alone only when it is clearly ahead of the runner-up;
    otherwise names the family honestly: 'keys (piano or organ?)'."""
    rank = sorted(means.items(), key=lambda kv: -kv[1])
    (a, va), (b, vb) = rank[0], rank[1]
    if va <= 0:
        return None, (a, b)
    if a == "organ" and va < ORGAN_SOLO_SCORE and means.get("piano", 0) > 0:
        b = "piano"                               # a modest 'organ' score is how sustained piano chords and pads often tag
        return f"keys ({a} or {b}?)", (a, b)
    if vb < KEYS_MIN_RIVAL or va >= KEYS_LEAD * vb:
        return a, (a, b)
    return f"keys ({a} or {b}?)", (a, b)


def piano_carries(rms_piano, rms_other, gate_piano, piano_score, min_score=0.15):
    """Windows where the piano stem carries the keys: it has level, it is not far under the other stem, and the
    tagger hears a keyboard in it. Used to overrule an 'organ' tag on the other stem."""
    return gate_piano & (rms_piano >= rms_other - PIANO_CARRIES_DB) & (piano_score >= min_score)


def piano_is_weak(conf, rms_piano, rms_other, on):
    """True when the piano stem is likely bleed: low tagger score, or far quieter than the other stem while it is 'on'."""
    if conf < PIANO_WEAK_CONF:
        return True
    if on.any():
        return float(np.median(rms_piano[on]) - np.median(rms_other[on])) < -PIANO_WEAK_DB
    return False


SURGE_DB, SURGE_FRAME_S, SURGE_SPAN_S = 8.0, 0.25, 6.0     # a guitar-stem step of >= 8 dB that holds for 6 s


def guitar_surges(y, sr, ref_db, p_electric=None):
    """Where the isolated guitar stem steps up in level in one go and STAYS there (a band-return strum, a guitar entering under a
    song it was already quietly in). Level only: median of 0.25 s frames over the next 6 s vs the 6 s before, so one loud hit or a
    dropout cannot fake it; the stem must also reach within 20 dB of its own loud level (not an empty residue). `p_electric` is the
    tagger's per-window (3 s, every 1 s) Electric guitar score on that stem, quoted as a local reading, never a verdict.
    -> [{"t", "db", "elec_before", "elec_after"}]"""
    f = max(1, int(sr * SURGE_FRAME_S))
    n = len(y) // f
    if n < 4 * int(SURGE_SPAN_S / SURGE_FRAME_S):
        return []
    db = 20 * np.log10(np.sqrt(np.mean(y[:n * f].reshape(n, f) ** 2, axis=1)) + 1e-7)
    k = int(SURGE_SPAN_S / SURGE_FRAME_S)
    jump = np.full(n, -99.0)
    for i in range(k, n - k):
        jump[i] = np.median(db[i:i + k]) - np.median(db[i - k:i])
    out, i = [], k
    while i < n - k:
        if jump[i] >= SURGE_DB and np.median(db[i:i + k]) > ref_db - 20.0:
            j = i
            while j + 1 < n - k and jump[j + 1] >= SURGE_DB:
                j += 1
            peak = i + int(np.argmax(jump[i:j + 1]))
            before = float(np.median(db[peak - k:peak]))
            on = next((x for x in range(max(0, i - 8), min(n, peak + k)) if db[x] >= before + 6.0), peak)   # the first hit
            t = on * SURGE_FRAME_S
            rec = {"t": round(t, 1), "db": round(float(jump[peak]), 1)}
            # how it arrives: separate hits (>= 8 dB over the frame before) until the level first HOLDS (1.5 s above 60% of the step)
            hold = before + 0.6 * float(jump[peak])
            w = int(1.5 / SURGE_FRAME_S)
            sus = next((x for x in range(on, min(n - w, on + 24)) if (db[x:x + w] >= hold).all()), None)
            if sus is not None:
                hits = [round(x * SURGE_FRAME_S, 2) for x in range(max(1, on), sus) if db[x] >= db[x - 1] + 8.0 and db[x] >= before + 6.0]
                rec["sustained_from"], rec["hits"] = round(sus * SURGE_FRAME_S, 2), hits
            if p_electric is not None and len(p_electric) > 12:
                ti = int(t)
                bf = p_electric[max(0, ti - 8):max(1, ti - 2)]
                af = p_electric[max(0, ti - 1):ti + 2]
                if len(bf) and len(af):
                    rec["elec_before"], rec["elec_after"] = round(float(np.mean(bf)), 2), round(float(np.max(af)), 2)
            if not out or t - out[-1]["t"] > 8.0:
                out.append(rec)
            i = j + k
        else:
            i += 1
    return out


def _segments(y, mask, hop=HOP):
    """the audio of the windows in `mask` (central `hop` seconds of each 3 s window), concatenated; None if nothing."""
    parts = []
    off = int((WIN - hop) / 2 * SR)
    for i in np.where(mask)[0]:
        a = int(i * hop * SR) + off
        if a + int(hop * SR) <= len(y):
            parts.append(y[a:a + int(hop * SR)])
    return parts


def bass_features(y, mask):
    """envelope + spectrum numbers for the bass stem over the windows in `mask`:
    attack (p95 dB rise within 20 ms: a pluck/pick is steep), swing (typical dB spread inside 1 s: a held synth is flat),
    fund (share of 30-2000 Hz energy that sits under 130 Hz: a sub/sine-like bass is almost all fundamental)."""
    parts = _segments(y, mask)
    if len(parts) < 5:
        return None
    fr = int(0.01 * SR)
    att, swing, spec = [], [], np.zeros(fr * 8 // 2 + 1)
    for seg in parts:
        n = len(seg) // fr
        e = 20 * np.log10(np.sqrt((seg[:n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-7)
        att.append(np.percentile(np.maximum(e[2:] - e[:-2], 0), 95))
        swing.append(np.percentile(e, 90) - np.percentile(e, 10))
    allx = np.concatenate(parts)
    F = np.abs(np.fft.rfft(allx[:len(allx) // 8192 * 8192].reshape(-1, 8192) * np.hanning(8192), axis=1)).mean(0) ** 2
    fq = np.fft.rfftfreq(8192, 1 / SR)
    lo = F[(fq >= 30) & (fq < 130)].sum()
    tot = F[(fq >= 30) & (fq < 2000)].sum() + 1e-12
    return {"attack": round(float(np.median(att)), 1), "swing": round(float(np.median(swing)), 1), "fund": round(float(lo / tot), 2)}


def bass_kind(feat, synth_lean, bass_lean):
    """'bass (synth-like)', 'bass guitar (plucked)' or None (stay with 'bass (type unclear)').
    Synth-like: a smooth envelope with little pluck transient and a strong fundamental, and the tagger does not clearly
    prefer Bass guitar / Double bass. Plucked: a steep attack and a wide swing and the tagger leans bass, or the shape is
    clearly plucked on its own. Anything in between stays unclear."""
    if not feat:
        return None
    smooth = feat["attack"] <= BASS_SMOOTH_ATTACK and feat["swing"] <= BASS_SMOOTH_SWING
    plucked = feat["attack"] >= BASS_PLUCK_ATTACK and feat["swing"] >= BASS_PLUCK_SWING
    tag_synth = synth_lean >= 0.10 and synth_lean > bass_lean
    tag_bass = bass_lean >= 0.10 and bass_lean > synth_lean
    if smooth and feat["fund"] >= BASS_FUND and not tag_bass:
        return "bass (synth-like)"
    if plucked and not tag_synth:
        return "bass guitar (plucked)"
    return None


def pad_features(y, mask):
    """sustained-ness numbers for the 'other' stem over the windows in `mask`: perc (share of the energy HPSS calls
    percussive), swing (typical dB spread inside 1 s), attack (p95 dB rise within 20 ms). Envelope numbers alone cannot tell a
    pad from a sustained guitar, so pad_kind also wants the tagger's faint synth/pad hint."""
    import librosa
    parts = _segments(y, mask)
    if len(parts) < 5:
        return None
    fr = int(0.01 * SR)
    att, swing, perc = [], [], []
    for seg in parts:
        n = len(seg) // fr
        e = 20 * np.log10(np.sqrt((seg[:n * fr].reshape(n, fr) ** 2).mean(1)) + 1e-7)
        att.append(np.percentile(np.maximum(e[2:] - e[:-2], 0), 95))
        swing.append(np.percentile(e, 90) - np.percentile(e, 10))
        S = np.abs(librosa.stft(seg, n_fft=1024, hop_length=256))
        H, P = librosa.decompose.hpss(S)
        perc.append(float((P ** 2).sum() / ((H ** 2).sum() + (P ** 2).sum() + 1e-12)))
    return {"attack": round(float(np.median(att)), 1), "swing": round(float(np.median(swing)), 1),
            "perc": round(float(np.median(perc)), 2)}


def pad_kind(feat, hint=0.0):
    """'synth/pad-like layer (unconfirmed)' when the unnamed 'other' sound is sustained, smooth and not percussive AND the
    tagger's faint synth/pad score (`hint`) agrees; None when the evidence is thin (the card then keeps 'unclear')."""
    if not feat or hint < PAD_HINT:
        return None
    ok = feat["perc"] <= PAD_PERC and feat["attack"] <= PAD_ATTACK and feat["swing"] <= PAD_SWING
    return "synth/pad-like layer (unconfirmed)" if ok else None


import stage_status as ST  # HONEST CARD: optional naming and lookup outcomes.


@ST.stage("instruments", classify=lambda value: [ST.outcome("instruments",
    "skipped" if value.get("error") else "ran", value.get("error", ""))])
def analyse(path, use_cache=True, *, apply_credits=True):
    """analyse_raw plus the credits helping hand (credits.py; on by default, `--no-credits` / RAVEN_NO_CREDITS=1 turn it off).
    The cached result is always the ears' own; credited names are applied after, never stored."""
    res = analyse_raw(path, use_cache)
    if apply_credits:
        if res and "families" in res and not os.environ.get("RAVEN_NO_CREDITS"):
            import credits
            res = credits.apply_for(path, res)
        else:
            ST.mark("credits", "skipped", "instrument naming unavailable or credits disabled", kind="input")
    if res and res.get("six_note"):
        ST.mark("six-stem naming", "degraded", res["six_note"])
    return res


def analyse_raw(path, use_cache=True):
    """Returns a dict (JSON-able) or {"error": ...}. Cached next to the stems."""
    d, ok = _stem_dir(path)
    if not ok:
        return {"error": "no cached Demucs stems (run singer.py first)"}
    # Four-stem naming must neither reuse nor replace the extra-stem evidence.
    no_six = bool(os.environ.get("RAVEN_NO_SIX_STEMS"))
    cp = os.path.join(d, "instruments-4s.json" if no_six else "instruments.json")
    if use_cache and os.path.exists(cp):
        try:
            c = json.load(open(cp))
            if c.get("version") == VERSION:
                if no_six:
                    ST.mark("six-stem naming", "skipped", "RAVEN_NO_SIX_STEMS")
                return c
        except Exception:
            pass
    import raven_ears_plus as E
    import singer
    torch = E._torch()
    dev = E._pick_device(need_mb=1500)
    try:
        model, labels = E._load_tagger(dev)
    except Exception as e:
        ST.mark("instrument tagger", "degraded", f"{type(e).__name__}: {str(e)[:60]}; retried on CPU")
        dev = "cpu"
        model, labels = E._load_tagger("cpu")
    ix = {l: i for i, l in enumerate(labels)}
    t0 = time.time()
    # six-stem model for guitar / piano / a cleaner 'other'; drums, bass, vocals stay on the four-stem (ft) files
    d6, six, six_note, six_s = None, False, None, 0.0
    if not os.environ.get("RAVEN_NO_SIX_STEMS"):
        try:
            t6 = time.time()
            d6, cached6 = singer.get_stems(path, {}, six=True)
            six, six_s = True, (0.0 if cached6 else round(time.time() - t6, 1))
        except Exception as e:
            six_note = f"six-stem model unavailable ({type(e).__name__}: {str(e)[:60]}); four-stem naming"
            _log(six_note)
    else:
        ST.mark("six-stem naming", "skipped", "RAVEN_NO_SIX_STEMS")
    where = {s: (d6 if s in ("guitar", "piano") else d) for s in ("other", "guitar", "piano", "drums", "bass", "vocals")}
    stems = [s for s in ("other", "guitar", "piano", "drums", "bass", "vocals") if six or s in ("other", "drums", "bass", "vocals")]
    probs, rmss, wav = {}, {}, {}
    for s in stems:
        y = E.load_audio(os.path.join(where[s], f"{s}.wav"), SR)
        if s in ("bass", "guitar"):
            wav[s] = y
        if s == "vocals":
            rmss[s] = _window_rms(y)[3]
        else:
            probs[s], rmss[s] = _tag(y, model, dev, torch)
    wav["pad"] = E.load_audio(os.path.join(d6 if six else d, "other.wav"), SR)      # the six-stem 'other' has no guitar/piano in it
    n = min(len(v) for v in rmss.values())
    dur = float(len(y) / SR)
    gate = {}
    refs = {s: float(np.percentile(rms[:n], 90)) for s, rms in rmss.items()}
    mel_ref = max(refs[s] for s in MELODIC if s in refs)
    for s, rms in rmss.items():
        gate[s] = (rms[:n] > refs[s] - STEM_DB_GATE) & (refs[s] > -60)
        if s in ("guitar", "piano") and refs[s] < mel_ref - EMPTY_STEM_DB:
            gate[s] = np.zeros(n, bool)           # the six-stem model put (almost) nothing there: an empty residue, not an instrument
    # score per family per window
    S = {}
    for fam, (stem, members, thr) in FAMILIES.items():
        sc = np.zeros(n)
        for lab, w in members:
            if lab in ix:
                sc = np.maximum(sc, w * probs[stem][:n, ix[lab]])
        S[fam] = np.where(gate[stem], sc, 0.0)
        if six and fam in EXTRA_STEMS:             # the isolated stem is a second opinion; the better score wins
            ex = EXTRA_STEMS[fam]
            sx = np.zeros(n)
            for lab, w in members:
                if lab in ix:
                    sx = np.maximum(sx, w * probs[ex][:n, ix[lab]])
            S[fam] = np.maximum(S[fam], np.where(gate[ex], sx, 0.0))
    if six:                                       # the piano stem carries the keys: an 'organ' tag on the other stem is overruled
        pk = np.zeros(n)
        for lab in KEYS_CLASSES["piano"]:
            if lab in ix:
                pk = np.maximum(pk, probs["piano"][:n, ix[lab]])
        carries = piano_carries(rmss["piano"][:n], rmss["other"][:n], gate["piano"], pk)
        S["organ"] = np.where(carries, 0.0, S["organ"])
    for grp in EXCLUSIVE:                       # one winner per window inside a group of near-synonyms
        win = np.stack([S[g] for g in grp]).argmax(0)
        for k, g in enumerate(grp):
            S[g] = np.where(win == k, S[g], 0.0)
    fams = {}
    on_of = {}
    for fam, (stem, members, thr) in FAMILIES.items():
        sc = S[fam]
        on = _debounce(sc, thr, HOP, MIN_ON_S)
        on_of[fam] = on
        if stem in ("drums", "bass"):             # a soft stretch of the same stem belongs to the named kit/bass
            strict = rmss[stem][:n] > float(np.percentile(rmss[stem][:n], 90)) - 20.0
            for a_, b_ in _runs(strict, 1):
                if on[a_:b_].any():
                    on[a_:b_] = True
        if on.sum() * HOP < MIN_ON_S:
            continue
        if fam == "guitar":
            pg = np.maximum(probs["other"][:n], probs["guitar"][:n]) if six else probs["other"][:n]
            fam = _name_guitar(pg, ix, on, gate["other"][:n] | (gate["guitar"][:n] if six else False), isolated=six)
        fams[fam] = {"stem": stem, "spans": [(round(a, 1), round(b, 1)) for a, b in _runs(on, HOP)],
                     "conf": round(float(sc[on].mean()), 2), "peak": round(float(sc.max()), 2),
                     "share": round(float(on.mean()), 2)}
    # keyboards: when the leading class is not clearly ahead of the runner-up, name the family, not a guess
    present = [f for f in KEYS_FAMS if f in fams]
    keys_info = None
    if present:
        kon = np.zeros(n, bool)
        for f in present:
            kon |= on_of[f]
        kp = [probs["piano"], probs["other"]] if six else [probs["other"]]
        means = keys_class_means(kp, ix, kon)
        kname, leaders = keys_verdict(means)
        keys_info = {"means": means, "verdict": kname}
        if kname and kname.startswith("keys (") and leaders[0] in KEYS_CLASSES and leaders[1] in KEYS_CLASSES:
            top = max(present, key=lambda f: fams[f]["conf"])
            merged = {"stem": fams[top]["stem"], "spans": [(round(a, 1), round(b, 1)) for a, b in _runs(kon, HOP)],
                      "conf": max(fams[f]["conf"] for f in present), "peak": max(fams[f]["peak"] for f in present),
                      "share": round(float(kon.mean()), 2)}
            for f in present:
                fams.pop(f)
            fams[kname] = merged
    if "piano" in fams and six and piano_is_weak(fams["piano"]["conf"], rmss["piano"][:n], rmss["other"][:n], on_of["piano"]):
        fams["piano (weak, may be bleed)"] = fams.pop("piano")
    # drums/bass sound the tagger cannot name: say that they play, with no type (stem energy only)
    bass_timbre = None
    for s_, nm in (("drums", "drums (type unclear)"), ("bass", "bass (type unclear)")):
        rms = rmss[s_][:n]
        strict = rms > float(np.percentile(rms, 90)) - 20.0
        for f_, v_ in fams.items():
            if v_["stem"] == s_:
                for a, b in v_["spans"]:
                    strict[int(a / HOP):int(b / HOP)] = False
        on = _debounce(strict.astype(float), 0.5, HOP, MIN_ON_S) & strict
        if on.sum() * HOP >= 8:
            fams[nm] = {"stem": s_, "spans": [(round(a, 1), round(b, 1)) for a, b in _runs(on, HOP)], "conf": 0.0,
                        "peak": 0.0, "share": round(float(on.mean()), 2), "energy_only": True}
            if s_ == "bass":                      # no tag cleared: the shape of the sound can still say synth-like or plucked
                feat = bass_features(wav["bass"], on)
                lean = lambda labs: max([float(probs["bass"][:n][on][:, ix[l]].mean()) for l in labs if l in ix] or [0.0])
                kind = bass_kind(feat, lean(["Synthesizer", "Sampler"]), lean(["Bass guitar", "Double bass"]))
                bass_timbre = {"feat": feat, "kind": kind}
                if kind:
                    fams[kind] = fams.pop(nm)
                    fams[kind].pop("energy_only", None)
                    fams[kind]["timbre_only"] = True
    # stems with real level where nothing was named
    unclear = {}
    for s in ("other", "drums", "bass"):
        named = np.zeros(n, bool)
        for f in fams.values():
            if f["stem"] == s:
                for a, b in f["spans"]:
                    named[int(a / HOP):int(b / HOP)] = True
        loud = gate[s]
        unclear[s] = round(float((loud & ~named).sum() / max(1, loud.sum())), 2) if loud.sum() else 0.0
    # weak hints for the stems we could not name: best family by mean score over their unnamed windows
    hints = {}
    other_char = None
    for s in ("other", "drums", "bass"):
        un = gate[s].copy()
        for f in fams.values():
            if f["stem"] == s:
                for a, b in f["spans"]:
                    un[int(a / HOP):int(b / HOP)] = False
        if s == "other" and un.sum() * HOP >= 15:       # sustained, smooth, not percussive? then say "synth/pad-like", unconfirmed
            feat = pad_features(wav["pad"], un)
            hint = float(S["synth/pad"][un].mean())
            other_char = {"feat": feat, "hint": round(hint, 2), "label": pad_kind(feat, hint)}
        if un.sum() * HOP >= 15:
            cand = sorted(((float(S[g][un].mean()), g) for g, (st, _m, _t) in FAMILIES.items() if st == s), reverse=True)
            hints[s] = [(g, round(v, 2)) for v, g in cand[:2] if v >= 0.07]
    surges = []
    if six and "guitar" in wav and not gate["guitar"][:n].sum() == 0:
        try:
            pe = probs["guitar"][:, ix["Electric guitar"]] if "Electric guitar" in ix else None
            surges = guitar_surges(wav["guitar"], SR, refs["guitar"], pe)
        except Exception as e:
            ST.mark("guitar surge check", "failed", type(e).__name__)
            _log(f"guitar surge check skipped ({type(e).__name__})")
    res = {"version": VERSION, "duration": round(dur, 1), "guitar_surges": surges, "hop": HOP, "families": fams, "unclear": unclear, "hints": hints,
           "other_on": [(round(a, 1), round(b, 1)) for a, b in _runs(gate["other"][:n], HOP)],
           "vocal_on": [(round(a, 1), round(b, 1)) for a, b in _runs(gate["vocals"][:n], HOP)],
           "n": int(n), "tag_seconds": round(time.time() - t0, 1), "device": dev,
           "six_stems": six, "six_seconds": six_s, "six_note": six_note, "keys": keys_info, "other_character": other_char,
           "bass_timbre": bass_timbre, "stem_ref_db": {k: round(v, 1) for k, v in refs.items()}}
    try:
        json.dump(res, open(cp, "w"))
    except Exception:
        pass
    singer._free_gpu()
    return res


def _conf_word(c):
    return "clearly" if c >= 0.55 else ("likely" if c >= 0.40 else "possibly")


_DOUBLE_HEDGE = re.compile(
    r"^([ \t]*(?:\d+:\d{2}[ \t]+)?(?:" + "|".join(map(re.escape, GUITAR_SPECIFIC)) + r")) \(probably\)"
    r"(?:( +)(clearly|likely|possibly)( +)(?=\(score )"
    r"|( (?:enters|leaves)) \((clearly|likely|possibly)\)(?= *$| +\[instrument tagger\]))", re.M)


def single_hedge(text):
    """Keep one qualifier on a scored instrument row or a timed instrument event."""
    def replace(match):
        name, padding, score_word, score_gap, action, event_word = match.groups()
        # The family can be clear even when the guitar type is uncertain.
        if (score_word or event_word) == "clearly":
            return name + " (probably)" + (action or " ")
        if action:
            return f"{name}{action} ({event_word})"
        return name + padding + score_word + score_gap

    return _DOUBLE_HEDGE.sub(replace, text)


def _in(spans, t):
    return any(a <= t < b for a, b in spans)


def events(res):
    """[(t, text)] for the timelines: opens-with, entrances, exits, 'alone'. Only families with conf >= CONF_EVENT."""
    if not res or "families" not in res:
        return []
    dur, ev = res["duration"], []
    F = {k: dict(v) for k, v in res["families"].items() if v["conf"] >= CONF_EVENT and not v.get("energy_only")}
    for v in F.values():                          # on the timeline a pause under 15 s is not a leaving
        m = []
        for a, b in v["spans"]:
            if m and a - m[-1][1] < 15.0:
                m[-1][1] = b
            else:
                m.append([a, b])
        v["spans"] = m
    first = [k for k, v in F.items() if v["spans"][0][0] < 2.0]
    if first:
        ev.append((0.0, "instruments at the open: " + ", ".join(sorted(first))))
    for fam, v in F.items():
        if v["share"] >= 0.85:
            continue                              # on throughout: said in INSTRUMENTS, not an event
        for a, b in v["spans"]:
            if a >= 2.0:
                ev.append((a, single_hedge(f"{fam} enters ({_conf_word(v['conf'])})")))
            if b <= dur - 3.0:
                ev.append((b, single_hedge(f"{fam} leaves ({_conf_word(v['conf'])})")))
    for g in res.get("guitar_surges") or []:       # the guitar stem steps up and stays: a return of the band's guitar, said from level alone
        if any(("guitar" in x or "plucked" in x) and "enters" in x and abs(t0 - g["t"]) < 3.0 for t0, x in ev):
            continue
        txt = f"guitar steps up {g['db']:.0f} dB in one go and stays up [guitar stem level]"
        hits, sus = g.get("hits") or [], g.get("sustained_from")
        if len(hits) >= 2 and sus is not None and sus - g["t"] >= 1.0:
            ts_ = lambda x: f"{int(x) // 60}:{x - 60 * (int(x) // 60):04.1f}"
            txt = (f"guitar arrives as {len(hits)} separate hits ({', '.join(ts_(h) for h in hits)}) and holds from {ts_(sus)}, "
                   f"{g['db']:.0f} dB up [guitar stem level]")
        eb, ea = g.get("elec_before"), g.get("elec_after")
        if eb is not None and ea is not None and ea >= 0.25 and ea >= 3 * max(eb, 0.03):
            txt += f"; the tagger's electric-guitar score here is {ea:.2f} against {eb:.2f} before (a local reading, type unconfirmed)"
        ev.append((g["t"], txt))
    # alone: exactly one named family on for >= 8 s, no vocal
    run_fam, run_start = None, 0.0
    for t in list(np.arange(0, dur, HOP)) + [dur + 1]:
        cur = [k for k, v in F.items() if _in(v["spans"], t)] if t <= dur else []
        one = cur[0] if len(cur) == 1 and not _in(res.get("vocal_on", []), t) else None
        if one != run_fam:
            if run_fam and t - run_start >= 8.0 and run_start >= 2.0:
                ev.append((run_start, f"{run_fam} alone for {tm(t - run_start)}"))
            run_fam, run_start = one, t
    ev.sort(key=lambda x: x[0])
    return [(round(float(t), 1), x) for t, x in ev]


def block(res):
    """Readable INSTRUMENTS card (list of lines)."""
    L = ["INSTRUMENTS (PANNs AudioSet tagger run on each Demucs stem; an estimate, it names the nearest of 527 classes):"]
    if not res or "families" not in res:
        L.append("  unavailable: " + (res or {}).get("error", "no result"))
        return L
    fams = res["families"]
    if not fams:
        L.append("  unclear: no instrument family cleared the confidence bar")
    L.append("  overall:")
    wid = max([len(k) for k in fams] + [10])
    for fam, v in sorted(fams.items(), key=lambda kv: (kv[1]["stem"] not in MELODIC, -kv[1]["share"])):
        if v["share"] >= 0.85:
            where = "throughout"
        else:
            where = ", ".join(f"{tm(a)}–{tm(b)}" for a, b in v["spans"][:5]) + (f" +{len(v['spans']) - 5} more" if len(v["spans"]) > 5 else "")
        if v.get("credited_from"):
            L.append(f"    {fam:<{wid}} the ears heard \"{v['credited_from']}\" ({round(v['share'] * 100)}% of the song)  {where}")
            continue
        if v.get("energy_only"):
            L.append(f"    {fam:<{wid}} present, not nameable (energy only, {round(v['share'] * 100)}% of the song)  {where}")
            continue
        if v.get("timbre_only"):
            L.append(f"    {fam:<{wid}} from the sound's shape, no tag cleared ({round(v['share'] * 100)}% of the song)  {where}")
            continue
        line = f"    {fam:<{wid}} {_conf_word(v['conf']):<8} (score {v['conf']:.2f}, {round(v['share'] * 100)}% of the song)  {where}"
        L.append(single_hedge(line))
    for s, f in res.get("unclear", {}).items():
        named = any(v["stem"] == s for v in fams.values())
        if (not named and f > 0) or f >= 0.3:
            och = (res.get("other_character") or {}) if s == "other" else {}
            oc = och.get("label")
            L.append(f"    {s} stem: {round(f * 100)}% of its sound did not match a confident name: "
                     + (f"{oc} (the ears heard: {och['credited_from']})" if oc and och.get("credited_from")
                        else f"{oc} (sustained, smooth, not percussive)" if oc else "unclear")
                     + (" (faint hints: " + ", ".join(f"{g} {v:.2f}" for g, v in res.get("hints", {}).get(s, [])) + ")" if res.get("hints", {}).get(s) else ""))
    # who is playing: windows with the same set merged
    n, hop = res["n"], res["hop"]
    sets = [tuple(sorted(k for k, v in fams.items() if _in(v["spans"], i * hop + 0.01))) for i in range(n)]
    segs, a = [], 0
    for i in range(1, n + 1):
        if i == n or sets[i] != sets[a]:
            segs.append([a * hop, i * hop, sets[a]])
            a = i
    merged = []
    for s in segs:                                # fold segments under 6 s into the previous one
        if merged and (s[1] - s[0] < 6.0):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    out2 = []
    for s in merged:
        if out2 and out2[-1][2] == s[2]:
            out2[-1][1] = s[1]
        else:
            out2.append(s)
    L.append("  who is playing ('+ voice' = the vocal stem has level):")
    for a, b, st in out2:
        voice = " + voice" if _in(res.get("vocal_on", []), (a + b) / 2) else ""
        names = list(st)
        if _in(res.get("other_on", []), (a + b) / 2) and not any(fams[k]["stem"] == "other" for k in st):
            oc = (res.get("other_character") or {}).get("label")
            names.append(oc or "a melodic layer I cannot name (unclear)")
        L.append(f"    {tm(a)}–{tm(b)}  {', '.join(names) if names else 'silence/unclear'}{voice}")
    L += res.get("credit_lines", [])
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description="instruments.py: which instruments, and when they enter")
    ap.add_argument("audio")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    import credits
    credits.add_args(ap)
    import song_extras
    song_extras.add_args(ap)
    a = ap.parse_args(argv)
    from network_policy import local_audio_path
    local_audio_path(a.audio, ap)
    extras = song_extras.prepare(a, a.audio)
    credits.configure_from_args(a)
    path = os.path.abspath(a.audio)
    os.environ.setdefault("RAVEN_EARS_DEVICE", "auto")
    res = analyse(path, use_cache=not a.no_cache)
    if extras:
        res["song_extras"] = extras
    from ears_handoff import HANDOFF, finish_json  # MORE EARS: standalone listener handoff.
    if a.json:
        print(json.dumps(finish_json(res), indent=1, default=float))
    else:
        print("\n".join(block(res)))
        ev = events(res)
        if ev:
            print("ENTRANCES/EXITS (instruments):")
            for t, x in ev:
                print(f"  {tm(t)} {x}")
        extra_text = song_extras.decorate("", extras)
        if extra_text:
            print(extra_text, end="")
        print(HANDOFF)
    return 0


if __name__ == "__main__":
    sys.exit(main())
