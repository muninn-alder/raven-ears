#!/usr/bin/env python
"""singer.py: music ears, items 1+2 (24 Oct 2026).  `singer.py <file> [--json] [--keep-stems DIR]`

  STEMS   Demucs htdemucs_ft, local GPU, cached by file hash in ~/.cache/raven-stems/<hash>/
  SINGER  Ears+ (raven_ears_plus.analyze) run on the isolated vocals stem. speaker=None and
          update_baseline=False are hard-wired: this module never updates a speaker baseline.
  STEMS OVER TIME   energy curve per stem + entrances/exits (thresholds relative to each stem's own level)
  METER   Beat This! (CPJKU 2024) beats+downbeats; bar length from the downbeats, then onset accents to
          tell simple from compound (3/4 vs 6/8, 4/4 vs 12/8). Odd and additive meters (24 Oct, later): the bar CYCLE
          comes from autocorrelation of a pulse-synchronous accent series (drums+bass stems, full mix), then the
          GROUPING inside the cycle (3+3+2+2 vs 2+2+2+2+2...) from accent strength at the group starts folded over
          all bars. Says both readings when the grouping evidence is weak. Only claims what the evidence supports.

Audio analysis is local; credits and LRCLIB can receive song metadata (independent off switches).
Models run one at a time (8 GB card).
"""
from __future__ import annotations

import ears_profile as EP
from network_policy import offline, require_online, local_audio_path

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np

from instrumental_voice import window_pitch_text

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CACHE = os.path.expanduser(os.environ.get("RAVEN_STEMS_CACHE", "~/.cache/raven-stems"))
SRC = ["drums", "bass", "other", "vocals"]
SRC6 = SRC + ["guitar", "piano"]          # htdemucs_6s; its own cache folder (<hash>-6s) so the 4-stem cache is untouched
SR_ST = 44100
BARS = "▁▂▃▄▅▆▇█"
NOTE_FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]


MOOD_CAVEAT = "(speech-trained model; read as texture, not a verdict)"


import stage_status as ST  # HONEST CARD: run-local outcomes.

def _log(m):
    print(f"[singer] {m}", file=sys.stderr, flush=True)


def tm(s):
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def note(hz):
    m = int(round(69 + 12 * np.log2(hz / 440.0)))
    return f"{NOTE_FLAT[m % 12]}{m // 12 - 1}"


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------- 1. stems
def file_hash(path):
    path = local_audio_path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16] + "-a0"


def stems6_dir(path):
    return os.path.join(CACHE, file_hash(path) + "-6s")


def _local_stem_model(name, torch):
    """Use the same Demucs model formats without allowing a hub request."""
    from demucs.pretrained import REMOTE_ROOT
    from demucs.repo import BagOnlyRepo, LocalRepo, ModelLoadingError

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    if checkpoint_dir.is_dir():
        try:
            return BagOnlyRepo(REMOTE_ROOT, LocalRepo(checkpoint_dir)).get_model(name)
        except ModelLoadingError:
            pass
    try:
        from demucs.hf import DEFAULT_NAMESPACE, hf_repo_name, load_safetensors_model
        from demucs.apply import BagOfModels
        from huggingface_hub import try_to_load_from_cache
        import yaml
    except ImportError:
        raise FileNotFoundError(f"local Demucs weights unavailable for {name}") from None
    repo = f"{DEFAULT_NAMESPACE}/{hf_repo_name(name)}"

    def cached(filename):
        path = try_to_load_from_cache(repo, filename)
        if not isinstance(path, str):
            raise FileNotFoundError(f"local Demucs weights unavailable for {name}")
        return path

    with open(cached(name + ".yaml")) as f:
        bag = yaml.safe_load(f)
    files = [cached(sig + ".safetensors") for sig in bag["models"]]
    return BagOfModels([load_safetensors_model(path) for path in files],
                       bag.get("weights"), bag.get("segment"))


@ST.stage('stems')
def get_stems(path, T, six=False, *, local_output=None):
    """Return (dir, cached). Upgrade cached htdemucs to htdemucs_ft when online on the default profile.
    The opt-in CPU profile uses htdemucs for fresh four-stem output and marks it degraded.
    six=True: htdemucs_6s (adds guitar + piano; its README: guitar is OK, piano has bleed/artifacts), cached in <hash>-6s.
    local_output: use cached weights only and keep newly separated audio in this private scratch directory."""
    d = stems6_dir(path) if six else os.path.join(CACHE, file_hash(path))
    if all(os.path.exists(os.path.join(d, f"{s}.wav")) for s in (SRC6 if six else SRC)):
        model_note = os.path.join(d, "model.txt")
        upgrade = False
        if not six and os.path.exists(model_note):
            with open(model_note) as f:
                if f.read().strip() == "htdemucs":
                    upgrade = local_output is None and not EP.cpu() and not offline()
                    if EP.cpu():
                        ST.mark("stems", "degraded", "CPU profile chose htdemucs instead of htdemucs_ft; card quality difference unmeasured")
        if not upgrade:
            return d, True
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    if local_output is not None:
        d = os.path.join(local_output, "stems")
    if local_output is not None or offline():
        get_model = partial(_local_stem_model, torch=torch)
    else:
        from demucs.pretrained import get_model
    dev = "cpu" if EP.cpu() or os.environ.get("RAVEN_EARS_DEVICE", "auto").lower() == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    name = "htdemucs_6s" if six else "htdemucs" if EP.cpu() else "htdemucs_ft"
    try:
        model = get_model(name)
    except Exception as e:
        if six or name == "htdemucs":
            raise                      # fall back to the plain model if ft weights are missing
        ST.mark("stems", "degraded", "htdemucs_ft unavailable; using htdemucs")
        _log(f"htdemucs_ft unavailable ({e}); using htdemucs")
        model = get_model("htdemucs")
        name = "htdemucs"
    if not six and EP.cpu() and name == "htdemucs":
        ST.mark("stems", "degraded", "CPU profile chose htdemucs instead of htdemucs_ft; card quality difference unmeasured")
    os.makedirs(d, exist_ok=True)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-protocol_whitelist", "file,pipe",
                          "-i", path, "-map", "0:a:0",
                          "-f", "f32le", "-ac", "2", "-ar", str(SR_ST), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()
    model.to(dev).eval()
    w = torch.from_numpy(x)[None]
    ref = w.mean(1, keepdim=True)
    mu, sd = ref.mean(), ref.std() + 1e-8
    t = time.time()
    with torch.no_grad():
        out = apply_model(model, (w - mu) / sd, device=dev, split=True, overlap=0.25, progress=False)[0]
    out = out * sd + mu
    T["demucs"] = round(time.time() - t, 1)
    tmp = d + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for src, s in zip(model.sources, out):
        sf.write(os.path.join(tmp, f"{src}.wav"), s.T.clamp(-1, 1).cpu().numpy(), SR_ST, subtype="PCM_16")
    open(os.path.join(tmp, "model.txt"), "w").write(name + "\n")
    for f in os.listdir(tmp):
        os.replace(os.path.join(tmp, f), os.path.join(d, f))
    os.rmdir(tmp)
    del model, out, w
    _free_gpu()
    return d, False


def load_mono(path, sr=None):
    import soundfile as sf
    y, r = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr and r != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), int(r))
        y = resample_poly(y, int(sr) // g, int(r) // g).astype(np.float32)
        r = sr
    return y, r


# ---------------------------------------------------------------------------- 2. singer
def vocal_presence(stem_dir):
    """Is there a singer at all? Share of the mix's energy in the vocals stem + seconds of real vocal level."""
    tot, voc = 0.0, 0.0
    act = None
    for s in SRC:
        y, sr = load_mono(os.path.join(stem_dir, f"{s}.wav"))
        e = float((y.astype(np.float64) ** 2).sum())
        tot += e
        if s == "vocals":
            voc = e
            hop = sr
            n = len(y) // hop
            rms = np.sqrt((y[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
            db = 20 * np.log10(rms + 1e-9)
            act = float((db > -38).sum())
    share = voc / max(tot, 1e-12)
    return share, (act or 0.0)


def _runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


@ST.stage('singer')
def singer_block(vocals_wav, dur_total, words=None, capture_pitch=None, words_skip_reason=None, *, affect=True):
    """Run Ears+ on the vocals stem (never a baseline) and distil a SINGER block. Returns (lines, data)."""
    import raven_ears_plus as rep
    cap = {}
    o_fe, o_af = rep._front_end, rep.analyze_affect

    def fe_cap(y, *a, **k):
        cap["fe"] = o_fe(y, *a, **k)
        return cap["fe"]

    def af_cap(*a, **k):
        r = o_af(*a, **k)
        cap["af"] = r
        return r

    rep._front_end, rep.analyze_affect = fe_cap, af_cap
    try:
        # words come from the shared lyrics cache (lyrics_shared.py: one Whisper large-v3 pass per file, gap-recovered,
        # hallucination-filtered); Ears+ then does NOT run its own VAD-on Whisper over the stem.
        skip = {"words_skip_reason": words_skip_reason} if words_skip_reason else {}
        r = rep.analyze(vocals_wav, words=words or None, use_whisper=False, speaker=None, update_baseline=False, separate=False,
                        identify_voice=False, affect=affect, **skip)   # NEVER a baseline
    finally:
        rep._front_end, rep.analyze_affect = o_fe, o_af
    rep.release_models()
    _free_gpu()
    d, fe, af = r["data"], cap.get("fe"), cap.get("af")
    if capture_pitch is not None and fe is not None:
        # One call: a callback that replaces rather than updates must not lose f0.
        captured = {"f0": [float(x) if np.isfinite(x) else None for x in fe["f0"]], "fps": rep.FPS}
        if "voiced_prob" in fe:
            captured["voiced_prob"] = [float(x) if np.isfinite(x) else None for x in fe["voiced_prob"]]
        capture_pitch(captured)
    L = []

    # --- pitch: median/range, held high and low notes, where the voice climbs
    p = d.get("pitch") or {}
    if p.get("ok") and fe is not None:
        f0, voiced, times, rms = fe["f0"], fe["voiced"], fe["times"], fe["rms_db"]
        med = p["median_hz"]
        L.append(f"pitch: median {note(med)} ({med:.0f} Hz) · comfortable range {note(p['lo_hz'])}–{note(p['hi_hz'])} "
                 f"({p['range_st']:.0f} semitones)")
        fps = 1.0 / (times[1] - times[0])
        # held notes: contiguous voiced runs >= 0.4 s (gap <= 0.1 s bridged), pitch = run median
        v = voiced.copy()
        gap = int(0.1 * fps)
        idx = np.where(v)[0]
        runs = []
        if len(idx):
            s0 = prev = idx[0]
            for i in idx[1:]:
                if i - prev > gap:
                    runs.append((s0, prev))
                    s0 = i
                prev = i
            runs.append((s0, prev))
        held = []
        for a, b in runs:
            if (b - a) / fps >= 0.4 and p["lo_hz"] / 1.2 <= np.nanmedian(f0[a:b + 1]) <= p["hi_hz"] * 1.2:   # drop octave glitches
                seg = f0[a:b + 1]
                held.append((float(np.nanmedian(seg)), times[a], (b - a) / fps, float(np.nanmean(rms[a:b + 1]))))
        if held:
            loud_ref = np.percentile([h[3] for h in held], 70)

            def pick(key, rev, k=3, sep=6.0):
                out = []
                for h in sorted(held, key=key, reverse=rev):
                    if all(abs(h[1] - o[1]) > sep for o in out):
                        out.append(h)
                    if len(out) == k:
                        break
                return sorted(out, key=lambda h: h[1])
            hi = pick(lambda h: h[0], True)
            lo = pick(lambda h: h[0], False, 2)
            L.append("highest held notes: " + " · ".join(
                f"{note(h[0])} at {tm(h[1])} ({h[2]:.1f}s{', pushed hard' if h[3] >= loud_ref else ''})" for h in hi))
            L.append("lowest held notes: " + " · ".join(f"{note(h[0])} at {tm(h[1])}" for h in lo))
            # climb: median pitch by ~25 s windows; report the biggest rise and the window medians
            tt = np.array([h[1] for h in held]); ff = np.array([h[0] for h in held])
            nseg = max(4, min(8, int(dur_total // 25)))
            edges = np.linspace(tt.min(), tt.max() + 1e-3, nseg + 1)
            segm = []
            for k in range(nseg):
                s = (tt >= edges[k]) & (tt < edges[k + 1])
                segm.append(float(np.median(ff[s])) if s.sum() >= 3 else np.nan)
            ok = [x for x in segm if np.isfinite(x)]
            if len(ok) >= 3:
                st = [12 * np.log2(x / ok[0]) for x in ok]
                kmax = int(np.nanargmax(segm))
                rise = 12 * np.log2(segm[kmax] / np.nanmin(segm))
                arc = " → ".join(note(x) if np.isfinite(x) else "·" for x in segm)
                L.append(f"pitch arc (median by section): {arc}"
                         + (f" · the voice lifts ~{rise:.0f} semitones by {tm(edges[kmax])}" if rise >= 3 else " · stays in one register"))
        ends = p.get("phrase_ends") or []
        if ends:
            L.append(f"phrase endings: {ends.count('up')} lift, {ends.count('down')} fall, {ends.count('level')} hold")
    elif not p.get("ok"):
        L.append("pitch: too little voiced sound to measure")

    # --- voice events: not printed on songs (12 Nov 2026, Fox + Muninn). The speech event model
    # on a sung vocal stem says Snort up to x16 a song and Singing once or twice (Thunder God:
    # Grunt x7, Groan x12, Burp x1). It stays in the JSON and on voice-note cards, where it hears.
    br = (d.get("events") or {}).get("breath") or {}
    if br.get("per_min"):
        L.append(f"breathing: about {br['per_min']:.0f}/min")

    # --- wordless stretches
    g = d.get("gaps") or {}
    ww = [(a, b) for a, b in (g.get("wordless") or []) if b - a >= 2.0]
    # merge neighbours within 3 s
    merged = []
    for a, b in ww:
        if (merged and a - merged[-1][1] <= 3.0 and
                not any(s < a and e > merged[-1][1] for _, s, e in words or [])):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged = [m for m in merged if m[1] - m[0] >= 6.0]
    if merged:
        L.append("voice present, words not heard (Whisper found none): " +
                 ", ".join(f"{tm(a)}–{tm(b)}" for a, b in merged[:5]))

    # --- arousal curve + the most intense moment
    if af and af.get("ok"):
        C, P = np.asarray(af["centers"]), np.asarray(af["P"])[:, 0]
        act = fe["active"] if fe is not None else None
        nseg = max(6, min(12, int(dur_total // 25)))
        edges = np.linspace(0, dur_total, nseg + 1)
        vals = []
        for k in range(nseg):
            s = (C >= edges[k]) & (C < edges[k + 1])
            if act is not None:
                a0, a1 = int(edges[k] * rep.FPS), int(edges[k + 1] * rep.FPS)
                if act[a0:a1].mean() < 0.15:
                    vals.append(np.nan)
                    continue
            vals.append(float(P[s].mean()) if s.any() else np.nan)
        ok = np.array([x for x in vals if np.isfinite(x)])
        if len(ok) >= 3:
            lo_, hi_ = ok.min(), ok.max()
            bar = "".join(BARS[int(round((x - lo_) / max(hi_ - lo_, 1e-6) * 7))] if np.isfinite(x) else "·" for x in vals)
            L.append(f"arousal across the song: {bar}  (0:00 → {tm(dur_total)}; {lo_:.2f} to {hi_:.2f}; · = no voice) " + MOOD_CAVEAT)
            L.append("  by section: " + " ".join(f"{tm(edges[k])}={vals[k]:.2f}" if np.isfinite(vals[k]) else f"{tm(edges[k])}=·"
                                                 for k in range(nseg)))
    pk = d.get("peak") or {}
    if pk.get("ok"):
        extra = []
        if fe is not None:
            s = (fe["times"] >= pk["start"]) & (fe["times"] <= pk["end"]) & fe["voiced"]
            if s.sum() > 10:
                extra.append(window_pitch_text(fe['f0'][s]))
        arousal = (f" · arousal {pk['arousal']:.2f} vs {pk['note_arousal']:.2f} average " + MOOD_CAVEAT
                   if pk.get("arousal") is not None and pk.get("note_arousal") is not None else "")
        L.append(f"most intense moment: {tm(pk['start'])}–{tm(pk['end'])} · {pk['loud_db'] - pk['note_loud_db']:.0f} dB louder "
                 f"than the voice's median" + arousal
                 + ("".join(", " + x for x in extra)))
    if af and af.get("ok") and af.get("whole"):
        w = af["whole"]
        L.append(f"overall: arousal {w['arousal']:.2f} · dominance {w['dominance']:.2f} · valence {w['valence']:.2f} (0-1) " + MOOD_CAVEAT)
    return L, {k: v for k, v in d.items() if k != "words"}, r["timings"]


# ---------------------------------------------------------------------------- 3. stems over time
def smooth_med(x, k):
    from scipy.ndimage import median_filter
    return median_filter(x, size=k, mode="nearest")


def stem_curves(stem_dir, hop_s=0.5, win_s=1.0):
    out = {}
    for s in SRC:
        y, sr = load_mono(os.path.join(stem_dir, f"{s}.wav"))
        hop, win = int(hop_s * sr), int(win_s * sr)
        n = max(0, (len(y) - win) // hop + 1)
        c = np.empty(n)
        cs = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
        for i in range(n):
            c[i] = (cs[i * hop + win] - cs[i * hop]) / win
        out[s] = 10 * np.log10(c + 1e-10)
    return out, hop_s


def hysteresis(db, ref, on_off, min_s, hop_s):
    """Binary on/off with two thresholds, then drop any state shorter than min_s (debounce)."""
    on_t, off_t = ref - on_off[0], ref - on_off[1]
    st = np.zeros(len(db), bool)
    cur = False
    for i, v in enumerate(db):
        if not cur and v > on_t:
            cur = True
        elif cur and v < off_t:
            cur = False
        st[i] = cur
    m = int(min_s / hop_s)
    changed = True
    while changed:                       # remove the shortest interior run until all >= min_s
        changed = False
        runs = []
        i = 0
        while i < len(st):
            j = i
            while j < len(st) and st[j] == st[i]:
                j += 1
            runs.append([i, j, st[i]])
            i = j
        short = [r for r in runs[1:-1] if r[1] - r[0] < m] if len(runs) > 2 else []
        if short:
            r = min(short, key=lambda r: r[1] - r[0])
            st[r[0]:r[1]] = ~r[2]
            changed = True
    return st


def stem_states(stem_dir, min_s=4.0):
    """Shared stem levels, presence and debounced states for the singer and harmony."""
    curves, hop = stem_curves(stem_dir)
    n = min(len(c) for c in curves.values())
    # per-stem own level: 90th percentile of its dB over the frames that have any sound
    ref, present, states = {}, {}, {}
    for s, c in curves.items():
        c = smooth_med(c[:n], 3)
        curves[s] = c
        ref[s] = float(np.percentile(c, 90))
    top = max(ref.values())
    for s in SRC:
        present[s] = ref[s] > top - 28.0 and ref[s] > -55.0     # a stem 28 dB under the loudest is just bleed
        if present[s]:
            states[s] = hysteresis(curves[s], ref[s], (22.0, 28.0), min_s, hop)
    return curves, hop, ref, present, states


def stem_transitions(states, hop):
    """Group the measured stem transitions for the cards."""
    trans = []
    for s, st in states.items():
        d_ = np.diff(st.astype(int))
        for i in np.where(d_ != 0)[0]:
            trans.append((float((i + 1) * hop), s, "enter" if d_[i] > 0 else "exit"))
    trans.sort()
    start_on = [s for s, st in states.items() if st[0]]
    # group transitions within 2 s of each other
    groups, cur = [], []
    for t in trans:
        if cur and t[0] - cur[0][0] > 2.0:
            groups.append(cur)
            cur = []
        cur.append(t)
    if cur:
        groups.append(cur)
    on = set(start_on)
    txt = []
    if start_on:
        txt.append((0.0, "opens with " + " + ".join(start_on)))
    for g in groups:
        t = g[0][0]
        ent = [s for _t, s, k in g if k == "enter"]
        ext = [s for _t, s, k in g if k == "exit"]
        for s in ent:
            on.add(s)
        for s in ext:
            on.discard(s)
        bits = []
        if ent:
            if ent == ["vocals"]:
                bits.append("vocals " + ("return" if "vocals" in start_on or any(
                    k == "exit" and s == "vocals" for tt, s, k in trans if tt < t) else "enter"))
            else:
                bits.append(" + ".join(ent) + " enter")
        if ext:
            if not on:
                bits.append("everything drops out")
            elif len(on) == 1 and len(ext) >= 2:
                bits.append(f"everything drops out but {next(iter(on))}")
            else:
                bits.append(" + ".join(ext) + " drop out")
        txt.append((t, " · ".join(bits)))
    return txt


@ST.stage('stems timeline')
def stems_timeline(stem_dir, dur_total, min_s=4.0):
    curves, hop, ref, present, states = stem_states(stem_dir, min_s)
    n = min(len(c) for c in curves.values())
    lines, ev = [], []
    # sparklines: each stem's energy relative to its own level
    W = 60
    for s in SRC:
        if not present[s]:
            lines.append(f"  {s:<6} (absent or only bleed)")
            continue
        c = curves[s]
        seg = np.array_split(c, W)
        m = np.array([x.mean() for x in seg])
        z = np.clip((m - (ref[s] - 30)) / 30.0, 0, 1)
        lines.append(f"  {s:<6} {''.join(BARS[int(round(v * 7))] for v in z)}")
    lines.append(f"  {'':<6} 0:00{' ' * (W - 9)}{tm(n * hop)}")
    txt = stem_transitions(states, hop)
    # swells / thinnings: a sustained step in a stem's own level (>= 6 dB, 4 s before vs 4 s after)
    k = int(4.0 / hop)
    steps = []
    for s, c in curves.items():
        if not present[s]:
            continue
        for i in range(k, len(c) - k):
            m1, m0 = float(np.mean(c[i:i + k])), float(np.mean(c[i - k:i]))
            d_ = m1 - m0
            if min(m0, m1) < ref[s] - 25.0:          # one side is basically silence: that's an entrance/exit, not a swell
                continue
            if abs(d_) >= 6.0 and states[s][max(0, i - k)] and states[s][min(len(c) - 1, i + k - 1)]:
                steps.append([i * hop, s, d_])
    steps.sort(key=lambda x: -abs(x[2]))
    kept = []
    for t, s, d_ in steps:
        if all(not (s == s2 and abs(t - t2) < 10) for t2, s2, _d in kept):
            kept.append((t, s, d_))
    for t, s, d_ in sorted(kept[:8]):
        txt.append((t, f"{s} {'swells' if d_ > 0 else 'thins'} {abs(d_):.0f} dB"))
    txt.sort(key=lambda x: x[0])
    ev = [{"t": round(t, 1), "text": x} for t, x in txt]
    # tail: last stem activity vs. end of song
    return lines, ev, {s: round(ref[s], 1) for s in SRC}, present


# ---------------------------------------------------------------------------- 4. meter
BEAT_CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
BEAT_CHECKPOINT_MIRRORS = (
    ("Hugging Face mirror (oulianov/beat_this)", "https://huggingface.co/oulianov/beat_this/resolve/main/final0.ckpt"),
    ("Hugging Face mirror (teamup-tech/Beat-This-weights)", "https://huggingface.co/teamup-tech/Beat-This-weights/resolve/main/final0.ckpt"),
)
BEAT_CHECKPOINT_SHA256 = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
BEAT_CHECKPOINT_SIZE = 81058141


def _beat_checkpoint_expected():
    expected_hash = os.environ.get("RAVEN_BEAT_THIS_SHA256", BEAT_CHECKPOINT_SHA256).lower()
    expected_size = int(os.environ.get("RAVEN_BEAT_THIS_SIZE", BEAT_CHECKPOINT_SIZE))
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash) or expected_size <= 0:
        raise ValueError("RAVEN_BEAT_THIS_SHA256 must be 64 hex digits; RAVEN_BEAT_THIS_SIZE must be positive")
    return expected_hash, expected_size


def _verify_beat_checkpoint(path):
    expected_hash, expected_size = _beat_checkpoint_expected()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    actual_hash = digest.hexdigest()
    if size != expected_size or actual_hash != expected_hash:
        raise ValueError(
            f"Beat This! checkpoint verification failed: size {size} (expected {expected_size}); "
            f"sha256 {actual_hash} (expected {expected_hash}). "
            "For a verified upstream update, set RAVEN_BEAT_THIS_SHA256 and, if the size changed, RAVEN_BEAT_THIS_SIZE")


def setup_beat_checkpoint(check=False):
    """Verify the Torch cache, or fetch official then mirrors without exposing partial files."""
    import torch
    from pathlib import Path
    import tempfile
    import urllib.request
    _beat_checkpoint_expected()  # Reject malformed overrides before touching the cache or network.
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "beat_this-final0.ckpt"
    if checkpoint.is_file():
        try:
            _verify_beat_checkpoint(checkpoint)
            return checkpoint
        except ValueError as exc:
            if check:
                raise
            _log(str(exc))
    elif check:
        raise FileNotFoundError("Beat This! checkpoint missing or empty; run --setup music")
    require_online("Beat tracker weights missing or invalid in Torch cache; cached downbeats remain available")
    checkpoint.unlink(missing_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for source, url in (("official server", BEAT_CHECKPOINT_URL), *BEAT_CHECKPOINT_MIRRORS):
        tmp = None
        _log(f"get Beat This! checkpoint from {source}: {url}")
        try:
            with tempfile.NamedTemporaryFile(dir=checkpoint.parent, prefix=checkpoint.name + ".", suffix=".part", delete=False) as f:
                tmp = Path(f.name)
                with urllib.request.urlopen(url, timeout=60) as response:
                    shutil.copyfileobj(response, f)
            _verify_beat_checkpoint(tmp)
            tmp.replace(checkpoint)
            _log(f"Beat This! checkpoint from {source}, sha256 verified")
            return checkpoint
        except Exception as exc:
            error = f"{url}: {type(exc).__name__}: {exc}"
            errors.append(error)
            _log(error)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    raise RuntimeError("Beat This! checkpoint download failed: " + "; ".join(errors))


@ST.stage('beat tracker')
def beats_downbeats(path, use_cache=True):
    """Share the raw tracker result with harmony; meter's derived cache is separate."""
    local_audio_path(path)
    import pickle
    cp = identity = None
    if use_cache:
        try:
            import harmony
            import torch
            from importlib.metadata import version
            checkpoint = os.path.join(torch.hub.get_dir(), "checkpoints", "beat_this-final0.ckpt")
            st = os.stat(checkpoint) if os.path.isfile(checkpoint) else None
            dev = "cpu" if os.environ.get("RAVEN_EARS_DEVICE", "auto").lower() == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
            identity = (_src_hash(), offline(), dev, torch.__version__, version("beat_this"),
                        checkpoint, (st.st_mtime_ns, st.st_size) if st else None)
            cp = harmony.cache_path(path) + ".beats.pkl"
            if os.path.exists(cp):
                c = pickle.load(open(cp, "rb"))
                if c["identity"] == identity:
                    ST.collect(c["stage_status"])
                    return c["data"]
        except Exception:
            pass  # Cache failure must not prevent a fresh measurement.
    c = _tracked_beats(path)
    # A transient Beat This! failure should be retried, not frozen as a cached fallback.
    if cp and c["data"][2] == "beat_this":
        try:
            pickle.dump(dict(c, identity=identity), open(cp, "wb"))
        except OSError:
            pass
    return c["data"]


@ST.engine
def _tracked_beats(path):
    return {"data": _beats_downbeats(path)}


def _beats_downbeats(path):
    try:
        import torch
        from beat_this.inference import Audio2Beats
        dev = "cpu" if os.environ.get("RAVEN_EARS_DEVICE", "auto").lower() == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
        if offline():
            checkpoint = os.path.join(torch.hub.get_dir(), "checkpoints", "beat_this-final0.ckpt")
            if not os.path.isfile(checkpoint):
                require_online("Beat tracker weights missing from Torch cache; cached downbeats remain available")
            checkpoint = setup_beat_checkpoint(check=True)
            # Beat This! forwards this to torch.load. An open file cannot be
            # mistaken for a shortname/URL, even if the cache is later removed.
            with open(checkpoint, "rb") as weights:
                f = Audio2Beats(checkpoint_path=weights, device=dev)
        else:
            checkpoint = setup_beat_checkpoint()
            with open(checkpoint, "rb") as weights:
                f = Audio2Beats(checkpoint_path=weights, device=dev)
        from raven_ears_plus import load_audio
        b, d = f(load_audio(path, 22050), 22050)
        del f
        _free_gpu()
        return np.asarray(b), np.asarray(d), "beat_this"
    except Exception as e:
        if offline():
            raise  # No unchecked backend fallback after a local-only failure.
        beat_error = str(e)
        ST.mark("beat tracker", "degraded", f"Beat This! failed: {beat_error}; trying madmom")
        _log(f"beat_this failed ({type(e).__name__}: {e}); trying madmom")
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
        from raven_ears_plus import load_audio
        from madmom.audio.signal import Signal
        act = RNNDownBeatProcessor()(Signal(load_audio(path, 44100), sample_rate=44100))
        r = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4, 6], fps=100)(act)
    except Exception as exc:
        reason = f"Beat This! failed: {beat_error}; madmom fallback failed: {exc}"
        ST.mark("beat tracker", "failed", reason)
        raise RuntimeError(reason) from exc
    return r[:, 0], r[r[:, 1] == 1, 0], "madmom"


def _env(path, *, stem=False):
    import librosa
    if stem:
        y, sr = load_mono(path, 22050)
    else:
        from raven_ears_plus import load_audio
        sr = 22050
        y = load_audio(path, sr)
    hop = 256
    # percussive + broadband flux; log-compressed so quiet passages count
    e = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, lag=1, max_size=3)
    e = np.asarray(e, float)
    return e / (np.percentile(e, 95) + 1e-9), sr / hop


def _at(env, fps, t, w):
    a, b = int(max(0, (t - w) * fps)), int((t + w) * fps) + 1
    return float(env[a:b].max()) if b > a and a < len(env) else 0.0



def _compositions(K, parts=(2, 3)):
    out = []

    def rec(rem, cur):
        if rem == 0:
            out.append(tuple(cur))
            return
        for q in parts:
            if q <= rem:
                rec(rem - q, cur + [q])
    rec(K, [])
    return out


def _canon(c):
    return min(tuple(c[i:] + c[:i]) for i in range(len(c)))


def accent_series(b, p, envs_by_name):
    """Pulse-synchronous accent series: z-scored onset strength at every tracked pulse, drums + bass (weight 1)
    and the full mix (weight 0.5), summed. These are the 'percussive' and 'full mix' views in one series."""
    tot = np.zeros(len(b))
    used = []
    for nm, wgt in (("drums", 1.0), ("bass", 1.0), ("mix", 0.5)):
        if nm not in envs_by_name:
            continue
        e, fps = envs_by_name[nm]
        a = np.array([_at(e, fps, t, min(0.04, p / 8)) for t in b])
        tot += wgt * (a - a.mean()) / (a.std() + 1e-9)
        used.append(nm)
    return tot, used


def cycle_from_acf(a, b, p, kmax=16):
    """Autocorrelation of the accent series at k pulses (k = 1..kmax), only over pulse pairs that really are
    k*p apart in time (so a tracker dropout does not misalign the lags). Returns (S dict, K, Smax)
    where K is the SMALLEST local-peak lag >= 2 within 85% of the best lag in 2..12: a cycle candidate, not certainty."""
    from meter_trust import CYCLE_NEAR_PEAK
    S = {}
    for k in range(1, kmax + 1):
        dt = b[k:] - b[:-k]
        ok = np.abs(dt - k * p) < 0.15 * k * p
        if ok.sum() > 20:
            S[k] = float(np.corrcoef(a[:-k][ok], a[k:][ok])[0, 1])
    rng = [k for k in range(2, 13) if k in S]
    if not rng:
        return S, None, 0.0
    smax = max(S[k] for k in rng)
    cands = [k for k in rng if S[k] >= CYCLE_NEAR_PEAK * smax and S[k] >= S.get(k - 1, -9) and S[k] >= S.get(k + 1, -9)]
    K = cands[0] if cands else None

    def support(k):
        """A real k-cycle repeats at 2k and 3k too. Returns mean(S[2k], S[3k]) / S[k] over the lags we measured
        (None if neither was measured)."""
        m = [S[j] for j in (2 * k, 3 * k) if j in S]
        return (float(np.mean(m)) / max(S[k], 1e-9)) if m else None
    if K is not None and len(cands) > 1:
        sup = {k: support(k) for k in cands}
        if sup[K] is not None and sup[K] < 0.5:
            # the smallest peak does not recur at its own multiples (a 3 inside a 7 looks like this):
            # prefer a rival that does, if it is clearly better supported
            rival = max((k for k in cands if sup[k] is not None), key=lambda k: sup[k], default=None)
            if rival is not None and sup[rival] > sup[K] + 0.15:
                K = rival
    return S, K, smax


def fold_profile(a, b, p, K):
    """Mean accent at each of K positions in the cycle. Tracker dropouts split the beats into runs; each run is
    folded on its own and the runs are then rotated to line up with the longest one before averaging."""
    gaps = np.where(np.diff(b) > 1.5 * p)[0] + 1
    profs = []
    for sg in np.split(np.arange(len(b)), gaps):
        if len(sg) < 3 * K:
            continue
        P, C = np.zeros(K), np.zeros(K)
        for j, i in enumerate(sg):
            P[j % K] += a[i]
            C[j % K] += 1
        profs.append((len(sg), P / np.maximum(C, 1)))
    if not profs:
        return None
    ref = max(profs, key=lambda x: x[0])[1]
    tot, w = np.zeros(K), 0
    for n, P in profs:
        sh = max(range(K), key=lambda s: float(np.dot(np.roll(P, -s), ref)))
        tot += n * np.roll(P, -sh)
        w += n
    return tot / w


def rank_groupings(A):
    """Every way of cutting K pulses into groups of 2 and 3, at every rotation, scored by Pearson r between the
    accent profile and a start-of-group template. One entry per CYCLIC pattern (3+3+2+2 and 2+2+3+3 are the same
    cycle, differing only in where the bar line falls); the rotation chosen puts the strongest accent first."""
    K = len(A)
    best = {}
    for c in _compositions(K):
        starts = np.cumsum((0,) + c[:-1])
        for r in range(K):
            tpl = np.zeros(K)
            tpl[(starts + r) % K] = 1
            if tpl.std() == 0:
                continue
            sc = float(np.corrcoef(A, tpl)[0, 1])
            key = _canon(list(c))
            if key not in best or sc > best[key][0] + 1e-9:
                best[key] = (sc, c, r)
    out = sorted(best.values(), key=lambda x: -x[0])
    return out


QUARTER_PULSE = (70.0, 230.0)   # pulse rates (per min) at which a tracked pulse is read as a quarter note
EIGHTH_CYCLE = 8                # long-cycle naming heuristic; eight pulses alone do not establish /8


def _meter_name(K, c, pulse_bpm):
    """Name a K-pulse bar with grouping c. Eight pulses can span two bars of 4/4;
    neither an even nor an uneven accent template establishes their note value.
    Outside that ambiguity, retain the existing rate and long-cycle heuristics:
    quarter-rate short cycles use /4, fast or long uneven cycles use /8, and
    long even cycles use eighth pairs. A slow short cycle has no denominator."""
    if K == 8:
        return None
    if all(g == 2 for g in c) and K >= EIGHTH_CYCLE:
        return f"{K // 2}/4"
    if K >= EIGHTH_CYCLE:
        return f"{K}/8"
    if pulse_bpm > QUARTER_PULSE[1]:
        return f"{K}/8"
    if pulse_bpm < QUARTER_PULSE[0]:
        return None
    return f"{K}/4"


def meter_block(path, stem_dir, dur_total, _pre=None, _win=None):
    """_pre = (b, d, engine, env, fps, envs) computed once for the whole song; _win = (t0, t1) restricts the
    analysis to one section (the envelopes stay whole, everything indexes by absolute time)."""
    if _pre is None:
        b, d, engine = beats_downbeats(path)
        env = fps = envs = None
    else:
        b, d, engine, env, fps, envs = _pre
    if _win is not None:
        b = b[(b >= _win[0]) & (b < _win[1])]
        d = d[(d >= _win[0]) & (d < _win[1])]
    L, data = [], {"engine": engine}
    if len(b) < 16 or len(d) < 3:
        return ["meter: not enough beats found to say anything"], data
    # beats per bar = median number of beats between consecutive downbeats
    idx = np.searchsorted(b, d - 1e-3)
    counts = np.diff(idx)
    counts = counts[(counts >= 1) & (counts <= 16)]
    if len(counts) == 0:
        return ["meter: not enough bars found to say anything"], data
    bpb = int(np.median(counts))
    frac = float((counts == bpb).mean())
    ibi = float(np.median(np.diff(b)))
    bar_s = float(np.median(np.diff(d)))
    data.update(beats=len(b), downbeats=len(d), beats_per_bar=bpb, bar_consistency=round(frac, 2),
                beat_s=round(ibi, 3), bar_s=round(bar_s, 3), tracker_bpm=round(60 / ibi, 1))
    if env is None:
        env, fps = _env(path)
        # also an envelope from bass + drums only: accents land there, less melodic clutter
        envs = {"mix": env}
        for s in ("drums", "bass"):
            try:
                envs[s], _ = _env(os.path.join(stem_dir, f"{s}.wav"), stem=True)
            except Exception:
                pass

    def subdiv(e):
        """Per tracked-beat interval: onset at 1/2 vs at 1/3 and 2/3 of the way to the next beat."""
        votes, s2s, s3s = [], [], []
        for i in range(len(b) - 1):
            T = b[i + 1] - b[i]
            if T > 2.0 or T < 0.2:
                continue
            w = min(0.04, T / 14)
            h = _at(e, fps, b[i] + T / 2, w)
            t3 = 0.5 * (_at(e, fps, b[i] + T / 3, w) + _at(e, fps, b[i] + 2 * T / 3, w))
            s2s.append(h); s3s.append(t3)
            if max(h, t3) > 0.15:
                votes.append(1 if t3 > h else -1)
        s2, s3 = float(np.mean(s2s)), float(np.mean(s3s))
        return s2, s3, (float(np.mean(np.array(votes) > 0)) if votes else 0.5), len(votes)
    s2, s3, v3, nv = subdiv(env)
    contrast = (s3 - s2) / max(s3 + s2, 1e-9)
    data.update(split2=round(s2, 3), split3=round(s3, 3), split_contrast=round(contrast, 2), split_votes_ternary=round(v3, 2))

    # accent profile per beat position in the bar (mean onset at the tracked beats, by position)
    pos = np.full(len(b), -1)
    for k in range(len(d)):
        i0 = idx[k]
        i1 = idx[k + 1] if k + 1 < len(d) else len(b)
        for j, i in enumerate(range(i0, min(i1, i0 + bpb))):
            if j < bpb:
                pos[i] = j
    prof = {}
    for nm, e in envs.items():
        w = min(0.04, ibi / 8)
        vals = np.array([_at(e, fps, t, w) for t in b])
        prof[nm] = np.array([vals[pos == j].mean() if (pos == j).any() else 0 for j in range(bpb)])
    A = prof["mix"] + (prof.get("drums", 0) * 0.5) + (prof.get("bass", 0) * 0.5)
    A = A / max(A.max(), 1e-9)
    data["accent_by_beat"] = [round(float(x), 2) for x in A]

    def group_score(g):
        if bpb % g:
            return None
        m = np.zeros(bpb, bool); m[::g] = True
        return float(A[m].mean() - A[~m].mean()) if (~m).any() else None

    verdict, conf, why = None, 0.0, []
    pulse_bpm = None

    # ---- bar CYCLE from autocorrelation of the pulse-synchronous accent series (odd / additive meters).
    # The tracker's downbeat count is only trusted when the accent cycle agrees (a multiple of the accent period
    # that is itself periodic). Tolerate It: the tracker said 2, the accents repeat every 10 pulses.
    Kc, new_path, cyc_src = bpb, False, "tracker"
    S_acf, A_prof = {}, None
    try:
        env_in = {"mix": (env, fps)}
        for _nm in ("drums", "bass"):
            if _nm in envs:
                env_in[_nm] = (envs[_nm], fps)
        a_ser, used_env = accent_series(b, ibi, env_in)
        S_acf, K_acf, smax = cycle_from_acf(a_ser, b, ibi)
        data["cycle_acf"] = {str(k): round(v, 2) for k, v in S_acf.items() if 2 <= k <= 12}
        data["cycle_envelopes"] = used_env
        if K_acf is not None and smax >= 0.15:
            tracker_ok = (bpb % K_acf == 0 or K_acf % bpb == 0) and S_acf.get(bpb, -1) >= 0.6 * smax
            if not tracker_ok:
                Kc, cyc_src = K_acf, "accent autocorrelation"
            data["cycle_pulses"] = Kc
            data["cycle_source"] = cyc_src
            data["cycle_strength"] = round(smax, 2)
            if Kc >= 5 and not (Kc == bpb and Kc in (6, 9, 12)):
                A_prof = fold_profile(a_ser, b, ibi, Kc)
                new_path = A_prof is not None
    except Exception as e:
        ST.mark("meter cycle", "degraded", f"{type(e).__name__}; using the tracker's bar")
        _log(f"cycle analysis failed ({type(e).__name__}: {e}); using the tracker's bar")
    if new_path:
        rk = rank_groupings(A_prof)
        top = rk[0]
        even = next((x for x in rk if all(g == 2 for g in x[1])), None)
        pulse_bpm = 60 / ibi
        # put the strongest accent of the winning cycle at the bar line
        c0 = list(top[1])
        starts = (np.cumsum([0] + c0[:-1]) + top[2]) % Kc
        j0 = int(starts[int(np.argmax(A_prof[starts]))])
        rot = [i for i in range(len(c0)) if int(starts[i]) == j0][0]
        c_show = tuple(c0[rot:] + c0[:rot])
        gname = "+".join(map(str, c_show))
        gname = _show_group(Kc, gname)
        nm_top = _meter_name(Kc, c_show, pulse_bpm)
        rivals = [x for x in rk[1:] if top[0] - x[0] < 0.12]
        margin = top[0] - (rk[1][0] if len(rk) > 1 else 0.0)
        gconf = float(np.clip(margin / 0.3, 0, 1) * np.clip(top[0] / 0.45, 0, 1))
        from meter_trust import cycle_support
        cyc_conf = (cycle_support(data) or {"confidence": 0.0})["confidence"]
        data["grouping"] = gname
        data["grouping_scores"] = [{"grouping": "+".join(map(str, x[1])), "r": round(x[0], 2)} for x in rk[:5]]
        data["accent_profile"] = [round(float(x), 2) for x in A_prof]
        data["grouping_confidence"] = round(gconf, 2)
        also = []
        if even is not None and top[1] != even[1] and not all(g == 2 for g in c_show):
            if top[0] - even[0] < 0.30:
                also.append(f"{_meter_name(Kc, even[1], pulse_bpm) or f'{Kc} pulses'} (same bar length, even grouping 2+2+...; fit {even[0]:.2f} vs {top[0]:.2f})")
        for x in rivals:
            if even is not None and x[1] == even[1]:
                continue
            xs = np.cumsum((0,) + x[1][:-1])
            also.append(f"{'+'.join(map(str, x[1]))} (fit {x[0]:.2f})")
        unit = "beats" if Kc < EIGHTH_CYCLE else "pulses"
        if Kc == 8:
            from meter_trust import accent_cycle_ambiguity
            verdict = accent_cycle_ambiguity(data)
        elif gconf >= 0.35 and nm_top:
            verdict = f"{nm_top} ({gname})"
        elif gconf >= 0.35:
            verdict = f"{Kc} {unit} per bar (grouped {gname}; pulse too slow to name the note value)"
        else:
            verdict = f"{Kc} {unit} per bar, grouping weak (best fit {gname}{', ' + nm_top if nm_top else ''})"
        why.append(f"cycle candidate {Kc} pulses (from {cyc_src}: {S_acf.get(Kc, 0):.2f} at lag {Kc}; lags 2..12: "
                   + " ".join(f"{k}:{v:.2f}" for k, v in S_acf.items() if 2 <= k <= 12) + ")")
        bar_note = "; the accent cycle does not establish a bar line" if Kc == 8 else ", which the accents do not support"
        why.append("accent by pulse in the cycle (z): " + " ".join(f"{x:+.1f}" for x in A_prof) + f" · tracker counted {bpb}/bar ({frac * 100:.0f}% of its bars){bar_note}" if cyc_src != "tracker" else
                   "accent by pulse in the cycle (z): " + " ".join(f"{x:+.1f}" for x in A_prof))
        if also:
            why.append("also consistent with: " + "; ".join(also))
        conf = min(cyc_conf, gconf)
        data["also_consistent"] = also
    elif bpb in (2, 3, 4):
        # tracker is on the main pulse; does each pulse split in 2 or 3?
        from meter_trust import compound_split
        comp = compound_split(data)
        simp = contrast < -0.2
        name = {2: ("2/4", "6/8"), 3: ("3/4", "9/8"), 4: ("4/4", "12/8")}[bpb]
        if comp:
            verdict = f"{name[1]} (compound, each beat splits in 3)"
        elif simp:
            verdict = f"{name[0]} (simple, each beat splits in 2)"
        else:
            verdict = f"{bpb} beats per bar; split 2 vs 3 is ambiguous (candidates {name[0]} / {name[1]})"
        conf = min(1.0, abs(contrast) * 1.5) * frac
        why.append(f"between-beat onsets: halfway {s2:.2f} vs thirds {s3:.2f}; {v3 * 100:.0f}% of beats lean to thirds")
        pulse_bpm = 60 / ibi
        if comp and ibi < 0.45:
            why.append("fast tracked beat with a ternary feel could also be a shuffled/swung 4/4")
    elif bpb in (6, 9, 12):
        # tracker is running at the subdivision. Which beats group together?
        cand = {}
        for g in (2, 3, 4):
            sc = group_score(g)
            if sc is not None and not (g == 4 and bpb != 12):
                cand[g] = sc
        data["group_scores"] = {str(k): round(v, 3) for k, v in cand.items()}
        # name each grouping as the meter it implies
        names = {(6, 2): "3/4", (6, 3): "6/8", (9, 3): "9/8", (12, 3): "12/8", (12, 4): "4/4 (counted in sixteenths)", (12, 2): "6/4"}
        if cand:
            best = max(cand, key=cand.get)
            srt = sorted(cand.values(), reverse=True)
            margin = srt[0] - (srt[1] if len(srt) > 1 else 0)
            nm = names.get((bpb, best), f"{bpb}/8")
            comp = best == 3
            if srt[0] < 0.08 or (len(srt) > 1 and margin < 0.05):
                verdict = f"{bpb} tracked beats per bar; grouping unclear (candidates " + " / ".join(
                    names.get((bpb, g), "?") for g in cand) + ")"
                conf = 0.2
            else:
                verdict = f"{nm} ({'compound, beats group in 3' if comp else 'simple, beats group in ' + str(best)})"
                conf = min(1.0, (margin * 3 + srt[0]) / 1.0) * frac
            why.append(f"tracker is counting {bpb} per bar at {60 / ibi:.0f}/min; accent by beat {data['accent_by_beat']}")
            why.append("grouping accent contrast: " + ", ".join(f"in {g}s {v:+.2f}" for g, v in cand.items()))
            pulse_bpm = 60 / (ibi * best)
    else:
        verdict = f"{bpb} beats per bar (unusual); no further claim"
        conf = 0.2 * frac
    if pulse_bpm:
        data["pulse_bpm"] = round(pulse_bpm, 1)
    if Kc == 7 and len(counts) >= 4:
        pr = counts[:-1] + counts[1:]
        alt = [(counts[i], counts[i + 1]) in ((4, 3), (3, 4)) for i in range(len(counts) - 1)]
        data["tracker_43_pairs"] = int(sum(alt))
        if sum(alt) >= 3:
            why.append(f"tracker bars also pair up as 4+3 / 3+4 in {int(sum(alt))} of {len(alt)} neighbouring bar pairs")
    data["verdict"] = verdict
    data["confidence"] = round(conf, 2)
    tag = "high" if conf >= 0.6 else "moderate" if conf >= 0.35 else "low"
    if new_path:
        fit_label = "template confidence" if Kc == 8 else "grouping confidence"
        L.append(f"METER: {verdict} · cycle {Kc} pulses · {fit_label} {data['grouping_confidence']:.2f}"
                 + ("".join(f" · also consistent with {x}" for x in data.get('also_consistent', [])[:2])))
        span_label = "accent cycle" if Kc == 8 else "bar"
        L.append(f"  {span_label} ≈ {Kc * ibi:.2f}s (cycle × {ibi:.2f}s pulse); pulse ≈ {pulse_bpm:.0f}/min; {len(d)} tracker bars, "
                 f"{frac * 100:.0f}% with {bpb} beats  [{engine}]")
    else:
        L.append(f"METER: {verdict} · confidence {tag} ({conf:.2f})")
        L.append(f"  {len(d)} bars tracked, {frac * 100:.0f}% of bars have {bpb} beats; bar ≈ {bar_s:.2f}s"
                 + (f"; pulse ≈ {pulse_bpm:.0f}/min" if pulse_bpm else "") + f"  [{engine}]")
    for w_ in why:
        L.append("  " + w_)
    # does the bar structure change?  (downbeat spacing in thirds)
    bars_t = np.diff(d)
    if len(bars_t) > 12:
        thirds = np.array_split(bars_t, 3)
        meds = [float(np.median(x)) for x in thirds]
        if max(meds) / min(meds) > 1.15:
            L.append("  bar length changes across the song: " + " → ".join(f"{m:.2f}s" for m in meds))
    from meter_trust import meter_text
    return meter_text("\n".join(L), data).splitlines(), data


def _fmt_t(t):
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


def _show_group(K, gname):
    """7-pulse cycles read as 4+3 / 3+4 (the 2+2+3 refinement is the same cycle)."""
    return {"2+2+3": "4+3", "3+2+2": "3+4"}.get(gname, gname) if K == 7 else gname


def _section_label(data):
    """Compact 'meter (grouping)' for one section, from meter_block's data dict."""
    from meter_trust import accent_cycle_ambiguity, cycle_unclear, tracker_unclear
    unclear = cycle_unclear(data) or tracker_unclear(data)
    if unclear:
        return unclear
    if accent_cycle_ambiguity(data):
        return accent_cycle_ambiguity(data)
    v = data.get("verdict") or "?"
    nm = v.split(" ")[0]
    K = data.get("cycle_pulses")
    g = data.get("grouping")
    if g and data.get("grouping_confidence", 0) >= 0.35:
        return f"{nm} ({_show_group(K, g)})"
    if g:
        return f"{nm} (grouping weak, best {_show_group(K, g)})"
    return nm


def segment_bars(b, d, dur_total, min_run=8, min_seg_s=20.0):
    """Cut points from runs of >= min_run consecutive tracker bars with the SAME beat count. The edge of a long
    steady run is where a section's meter starts or stops (4/4 bridge: every bar exactly 4). Short pieces are
    absorbed into a neighbour so each section is long enough to test. Returns [(t0, t1), ...]."""
    idx = np.searchsorted(b, d - 1e-3)
    counts = np.diff(idx)
    t_bar = d[:-1]
    cuts = []
    i = 0
    while i < len(counts):
        j = i
        while j + 1 < len(counts) and counts[j + 1] == counts[i]:
            j += 1
        if j - i + 1 >= min_run:
            cuts += [float(t_bar[i]), float(d[j + 1]) if j + 1 < len(d) else float(dur_total)]
        i = j + 1
    edges = [0.0] + sorted(set(round(c, 3) for c in cuts if 0 < c < dur_total)) + [float(dur_total)]
    # absorb pieces shorter than min_seg_s into the shorter-merged neighbour
    changed = True
    while changed and len(edges) > 2:
        changed = False
        for k in range(len(edges) - 1):
            if edges[k + 1] - edges[k] < min_seg_s:
                del edges[k + 1 if k + 1 < len(edges) - 1 else k]
                changed = True
                break
    return list(zip(edges[:-1], edges[1:]))


METER_FIT_FLOOR = 0.70   # share of tracker bars that must have the claimed beat count before a section gets its own meter
FADE_FIT_FLOOR = 0.90    # same, for a short tail at the end of the song (fade-outs make the tracker wobble)
FADE_TAIL_S = 45.0


def meter_claim(data):
    """The meter the METER line commits to, as a short string ('7/4', '7/4 then 4/4'), or None when it does not
    commit (low confidence, ambiguous, no denominator). music.py's TEMPO line quotes this instead of guessing."""
    from meter_trust import assess
    return assess(data)[0]


# ---------------------------------------------------------------------------- 4b. meter CHANGES inside a song
CHANGE_CONF_FLOOR = 0.35   # same floor meter_claim uses: a window below it is not evidence of any meter
CHANGE_MAX_GAPS = 2        # a window whose beat grid has more dropouts than this (gap > 1.5 pulses) is undecidable
CHANGE_MIN_RUN = 2         # a changed meter must hold for this many consecutive windows to count


def window_plan(n_beats, ibi):
    """Window and hop in PULSES, chosen by tempo: about 30 s of music (8-16 bars at normal tempi), never under 56
    or over 96 pulses so even a 7-pulse bar gets 8+ bars. Hop is half a window, so each stretch is seen twice."""
    n = int(np.clip(round(30.0 / max(ibi, 1e-3)), 56, 96))
    return n, n // 2


def window_cycle(a, b, p):
    """Which bar cycle does this stretch of the accent series support? Returns (label, conf, grouping).
    label = pulses per bar (even cycles 2/4/8 all read as 4); confidence uses
    cycle separation and conditional grouping. (None, 0, None) if undecidable."""
    S, K, smax = cycle_from_acf(a, b, p)
    if K is None or smax < 0.15:
        return None, 0.0, None
    from meter_trust import cycle_support
    support = cycle_support({"cycle_pulses": K, "cycle_acf": S})
    if len(support["candidates"]) > 1:
        return None, 0.0, None
    conf = support["confidence"]
    grp = None
    if K >= 5:
        A = fold_profile(a, b, p, K)
        if A is None:
            return None, 0.0, None
        rk = rank_groupings(A)
        top = rk[0]
        margin = top[0] - (rk[1][0] if len(rk) > 1 else 0.0)
        gconf = float(np.clip(margin / 0.3, 0, 1) * np.clip(top[0] / 0.45, 0, 1))
        conf = min(conf, gconf)
        grp = top[1]
        if K == 8 and all(g == 2 for g in grp):
            K = 4
    elif K in (2, 4):
        K = 4
    return int(K), conf, grp


def window_labels(a, b, p, n=None, hop=None):
    """Slide a window of n pulses (hop apart) over the cached beats + accent series; one label per window.
    No beat tracking here: this only slices what meter_block already computed."""
    if n is None:
        n, hop = window_plan(len(b), p)
    out = []
    i = 0
    while i + n <= len(b):
        bw = b[i:i + n]
        if int((np.diff(bw) > 1.5 * p).sum()) > CHANGE_MAX_GAPS:
            lab, conf = None, 0.0   # the tracker lost the beat here: not a steady grid, so no evidence either way
        else:
            lab, conf, grp = window_cycle(a[i:i + n], bw, p)
        out.append({"t0": float(b[i]), "t1": float(b[i + n - 1]), "label": lab, "conf": round(conf, 2)})
        i += hop
    return out


def same_cycle(x, y):
    """The same meter read at a multiple of its bar (5 and 10, 4 and 8 or 12, 7 and 14) is not a change."""
    lo, hi = min(x, y), max(x, y)
    return lo >= 3 and hi % lo == 0


def detect_meter_changes(wins, floor=CHANGE_CONF_FLOOR, min_run=CHANGE_MIN_RUN):
    """Turn per-window labels into meter segments and the seams between them. A segment is a run of >= min_run
    CONSECUTIVE windows with the same label, each over the confidence floor. A lone window that disagrees (or a
    low-confidence one) is noise: it neither starts a change nor ends one. Two runs of the same meter either side
    of noise are one segment, so 7 -> (noise) -> 7 is no change, while 7 -> 4 -> 7 is two. A seam is reported as
    a range: from the centre of the last window of one meter to the centre of the first window of the next.
    Returns (segments, changes); changes is [] when nothing survives."""
    runs, cur = [], None
    for k, w in enumerate(wins):
        ok = w["label"] is not None and w["conf"] >= floor
        if ok and cur is not None and same_cycle(cur["label"], w["label"]) and cur["k1"] == k - 1:
            cur["k1"] = k
            cur["label"] = min(cur["label"], w["label"])
        else:
            if cur is not None:
                runs.append(cur)
            cur = {"label": w["label"], "k0": k, "k1": k} if ok else None
    if cur is not None:
        runs.append(cur)
    segs = []
    for r in runs:
        if r["k1"] - r["k0"] + 1 < min_run:
            continue
        if segs and same_cycle(segs[-1]["label"], r["label"]):
            segs[-1]["k1"] = r["k1"]
            segs[-1]["label"] = min(segs[-1]["label"], r["label"])
        else:
            segs.append(dict(r, kfirst=r["k1"]))
    centre = lambda w: 0.5 * (w["t0"] + w["t1"])
    changes = []
    for s0, s1 in zip(segs, segs[1:]):
        changes.append({"from": s0["label"], "to": s1["label"],
                        "lo": centre(wins[s0["k1"]]), "hi": centre(wins[s1["k0"]])})
    if changes:
        changes[-1]["tail"] = stretch_tail(wins, segs[-1], floor)
    return segs, changes


def stretch_tail(wins, seg, floor=CHANGE_CONF_FLOOR):
    """How the FINAL stretch ends. Its solid part is its first unbroken run of windows (seg['kfirst']); later
    fragments of the same label across noise do not make it run on. 'to the end' is only earned when nothing but
    windows that agree follow it. Otherwise blur = (t0, t1) of the first block of undecidable windows after it
    (None if none) and lone = (label, t0) of the first confident window that disagrees."""
    blur, lone, open_ = None, None, True
    for w in wins[seg["kfirst"] + 1:]:
        if w["label"] is None or w["conf"] < floor:
            if blur is None:
                blur = (w["t0"], w["t1"])
            elif open_:
                blur = (blur[0], w["t1"])
            continue
        open_ = False   # a decided window ends the first blur block
        if lone is None and w["label"] != seg["label"] and not same_cycle(seg["label"], w["label"]):
            lone = (w["label"], w["t0"])
    return {"to_end": blur is None and lone is None, "blur": blur, "lone": lone}


def _fmt_seam(lo, hi):
    a, b = _fmt_t(lo), _fmt_t(hi)
    return f"≈{a}" if a == b or hi - lo < 4 else f"≈{a}–{b}"


def label_name_for(label, pulse_bpm):
    if label == 4 and QUARTER_PULSE[0] <= pulse_bpm <= QUARTER_PULSE[1]:
        return "4/4"
    return _meter_name(label, (3,), pulse_bpm) or f"{label} beats per bar"


def tail_phrase(tail, nm):
    """'the end' for a stretch that runs to the track's end, else an honest 'end isn't clear (...)'."""
    if not tail or tail["to_end"]:
        return "the end"
    why = []
    if tail["blur"]:
        why.append(f"beats blur {_fmt_seam(*tail['blur'])}")
    if tail["lone"]:
        why.append(f"a {nm(tail['lone'][0])} window at {_fmt_seam(tail['lone'][1], tail['lone'][1])}")
    return "its end isn't clear (" + "; ".join(why) + ")"


def meter_change_text(changes, base_text, pulse_bpm, max_stretches=6):
    """The one-line phrase, e.g. 'METER: 7/4 (2+2+3), with a 4/4 stretch ≈4:20–4:35 to ≈5:55–6:10, then 7/4 again.'"""
    if not changes:
        return None
    labels = [changes[0]["from"]] + [c["to"] for c in changes]
    nm = lambda l: label_name_for(l, pulse_bpm)
    A = labels[0]
    others = set(labels) - {A}
    seam = lambda c: _fmt_seam(c["lo"], c["hi"])
    if len(others) == 1:
        B = others.pop()
        stretches = []
        for i, c in enumerate(changes):
            if c["to"] == B:
                if i + 1 < len(changes):
                    end = seam(changes[i + 1])
                else:
                    end = tail_phrase(c.get("tail"), nm)
                stretches.append((seam(c), end))
        shown = stretches[:max_stretches]
        parts = "; ".join(f"from {s}; {e}" if e.startswith("its end") else f"{s} to {e}" for s, e in shown)
        more = f" (+{len(stretches) - len(shown)} more)" if len(stretches) > len(shown) else ""
        noun = f"a {nm(B)} stretch" if len(stretches) == 1 else f"{nm(B)} stretches"
        tail = f", then {nm(A)} again" if labels[-1] == A else ""
        return f"METER: {base_text}, with {noun} {parts}{more}{tail}."
    chain = ", then ".join(f"{nm(c['to'])} from {seam(c)}" for c in changes[:max_stretches])
    tp = tail_phrase(changes[-1].get("tail"), nm)
    return f"METER: {base_text}, then {chain}" + (f"; the last one: {tp}." if tp.startswith("its end") else ".")


def meter_changes(b, ibi, a_ser):
    """Windowed change detector over a cached beat grid + accent series. Returns (wins, segs, changes)."""
    wins = window_labels(a_ser, b, ibi)
    segs, changes = detect_meter_changes(wins)
    return wins, segs, changes


def _window_changes(pre, whole_L, whole_D):
    """Run the windowed change detector on the cached beats/envelopes. Returns (lines, data) when a meter change
    survives the rules in detect_meter_changes, else None (the caller keeps the one-meter answer)."""
    b, d, engine, env, fps, envs = pre
    ibi = float(np.median(np.diff(b)))
    env_in = {"mix": (env, fps)}
    for nm_ in ("drums", "bass"):
        if nm_ in envs:
            env_in[nm_] = (envs[nm_], fps)
    a_ser, _ = accent_series(b, ibi, env_in)
    wins, segs, changes = meter_changes(b, ibi, a_ser)
    # Keep evidence even without a change, for saved-card policy checks. Raw
    # candidate detection stays separate from the stronger claim threshold.
    whole_D["windows"] = [[round(w["t0"], 1), w["label"], w["conf"]] for w in wins]
    idx = np.searchsorted(b, d - 1e-3)
    counts = np.diff(idx)
    whole_D["window_bars"] = []
    for w in wins:
        local = counts[(d[:-1] >= w["t0"]) & (d[1:] <= w["t1"])]
        bpb = int(np.median(local)) if len(local) else 0
        whole_D["window_bars"].append({"beats_per_bar": bpb,
                                       "bar_consistency": round(float((local == bpb).mean()), 2) if len(local) else 0.0})
    if os.environ.get("RAVEN_METER_DEBUG"):
        for w in wins:
            _log(f"meter window {_fmt_t(w['t0'])}-{_fmt_t(w['t1'])}: {w['label']} conf {w['conf']}")
    if not changes:
        return None
    pulse_bpm = 60 / ibi
    base = segs[0]["label"]
    verdict = whole_D.get("verdict") or ""
    base_name = label_name_for(base, pulse_bpm)
    whole_ok = whole_D.get("confidence", 0) >= CHANGE_CONF_FLOOR and verdict.split(" ")[0] == base_name
    text = meter_change_text(changes, verdict if whole_ok else base_name, pulse_bpm)
    L = [text]
    if whole_ok and len(whole_L) > 1:
        L.append(whole_L[1])
    chain = [label_name_for(changes[0]["from"], pulse_bpm)] + [label_name_for(c["to"], pulse_bpm) for c in changes]
    data = dict(whole_D)
    data["changes"] = [{"from": c["from"], "to": c["to"], "lo_s": round(c["lo"], 1), "hi_s": round(c["hi"], 1)} for c in changes]
    data["change_chain"] = chain
    data["windows"] = [[round(w["t0"], 1), w["label"], w["conf"]] for w in wins]
    return L, data


def _src_hash():
    import hashlib
    import meter_trust
    return hashlib.sha1(open(os.path.abspath(__file__), "rb").read()
                        + open(meter_trust.__file__, "rb").read()).hexdigest()[:12]


@ST.stage('meter')
def meter_sections(path, stem_dir, dur_total):
    """Cached wrapper (stem_dir/meter.json, keyed on this file's source hash) so music.py can read the same answer
    the METER line prints without running the tracker twice."""
    cp = os.path.join(stem_dir, "meter.json")
    try:
        c = json.load(open(cp))
        if c.get("src") == _src_hash():
            ST.collect(c.get("stage_status", []))
            if c["D"].get("engine") == "madmom":
                ST.mark("beat tracker", "degraded", "cached madmom fallback")
            return c["L"], c["D"]
    except Exception:
        pass
    before = ST.snapshot()
    L, D = _meter_sections(path, stem_dir, dur_total)
    from meter_trust import meter_text
    L = meter_text("\n".join(L), D).splitlines()
    try:
        json.dump({"src": _src_hash(), "L": L, "D": D, "claim": meter_claim(D),
                   "stage_status": [r for r in ST.snapshot() if r not in before]}, open(cp, "w"),
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    except Exception:
        pass
    return L, D


def _meter_sections(path, stem_dir, dur_total):
    """One meter per SECTION. Beats/bars/envelopes computed once; the cycle/grouping test (meter_block) runs per
    segment; adjacent segments that give the same answer are merged back. One answer = the old single METER block."""
    b, d, engine = beats_downbeats(path)
    if len(b) < 16 or len(d) < 3:
        return ["meter: not enough beats found to say anything"], {"engine": engine}
    env, fps = _env(path)
    envs = {"mix": env}
    for s_ in ("drums", "bass"):
        try:
            envs[s_], _ = _env(os.path.join(stem_dir, f"{s_}.wav"), stem=True)
        except Exception:
            pass
    pre = (b, d, engine, env, fps, envs)
    whole_L, whole_D = meter_block(path, stem_dir, dur_total, _pre=pre)
    try:
        wc = _window_changes(pre, whole_L, whole_D)
    except Exception as e:
        ST.mark("meter changes", "degraded", f"{type(e).__name__}; keeping the one-meter answer")
        _log(f"windowed meter change check failed ({type(e).__name__}: {e}); keeping the one-meter answer")
        wc = None
    if wc:
        return wc
    segs = segment_bars(b, d, dur_total)
    if len(segs) <= 1:
        return whole_L, whole_D
    res = []
    for (t0, t1) in segs:
        L, D = meter_block(path, stem_dir, dur_total, _pre=pre, _win=(t0, t1))
        v = D.get("verdict") or ""
        ok = bool(v) and D.get("confidence", 0) >= 0.3 and not any(w in v for w in ("unusual", "ambiguous", "unclear", "uncertain"))
        # confidence floor: a meter claim needs most tracker bars to agree with it; fades and short tails need more
        fit = D.get("bar_consistency", 1.0)
        tail = t1 >= dur_total - 1.0 and (t1 - t0) < FADE_TAIL_S
        floor = FADE_FIT_FLOOR if tail else METER_FIT_FLOOR
        weak = ok and fit < floor
        if weak:
            ok = False
        res.append({"t0": t0, "t1": t1, "name": v.split(" ")[0] if ok else None,
                    "weak_tail": bool(weak and tail), "fit": fit})
        if os.environ.get("RAVEN_METER_DEBUG"):
            _log(f"meter segment {_fmt_t(t0)}-{_fmt_t(t1)}: {v or '-'} conf {D.get('confidence')} fit {fit} -> {res[-1]['name']}")
    fade_unclear = (res[-1]["t0"], res[-1]["t1"], res[-1]["fit"]) if res[-1]["weak_tail"] else None
    fade_line = (f"  outro: meter unclear (fade) {_fmt_t(fade_unclear[0])}\u2013{_fmt_t(fade_unclear[1])}: the beat tracker wobbles "
                 f"there (only {fade_unclear[2]:.0%} of its bars fit one meter), so no change is claimed") if fade_unclear else None
    if all(r["name"] is None for r in res):
        return whole_L + ([fade_line] if fade_line else []), whole_D
    # a section the test could not decide takes the longer decided neighbour's meter
    for i, r in enumerate(res):
        if r["name"] is None:
            nb = [x for x in (res[i - 1] if i else None, res[i + 1] if i + 1 < len(res) else None)
                  if x is not None and x["name"] is not None]
            if not nb:
                nb = [x for x in res if x["name"]]
            r["name"] = max(nb, key=lambda x: x["t1"] - x["t0"])["name"]
            r["inherited"] = True
    merged = []
    for r in res:
        if merged and merged[-1]["name"] == r["name"]:
            merged[-1]["t1"] = r["t1"]
        else:
            merged.append(dict(r))
    merged[0]["t0"], merged[-1]["t1"] = 0.0, float(dur_total)
    for _pass in range(3):   # re-test each merged section as a whole; sections that now agree on the meter merge
        for m in merged:
            m["L"], m["D"] = meter_block(path, stem_dir, dur_total, _pre=pre, _win=(m["t0"], m["t1"]))
            m["label"] = _section_label(m["D"]) if m["D"].get("verdict") else m["name"]
            m["name"] = m["label"].split(" ")[0]
        mm = []
        for m in merged:
            if mm and mm[-1]["name"] == m["name"]:
                mm[-1]["t1"] = m["t1"]
            else:
                mm.append(m)
        if len(mm) == len(merged):
            break
        merged = mm
    if len(merged) == 1:
        if merged[0]["name"] == (whole_D.get("verdict") or "").split(" ")[0]:
            return whole_L + ([fade_line] if fade_line else []), whole_D
        return merged[0]["L"] + ([fade_line] if fade_line else []), merged[0]["D"]
    line = "METER: " + " \u00b7 ".join(f"{_fmt_t(m['t0'])}\u2013{_fmt_t(m['t1'])} {m['label']}" for m in merged)
    L = [line]
    if fade_line:
        L.append(fade_line)
    for m in merged:
        L.append(f"  [{_fmt_t(m['t0'])}\u2013{_fmt_t(m['t1'])}] " + m["L"][0].replace("METER: ", "", 1))
        L += ["  " + x for x in m["L"][1:3]]
    data = {"engine": engine, "sections": [
        {"start": round(m["t0"], 1), "end": round(m["t1"], 1), "label": m["label"], "data": m["D"]} for m in merged],
        "whole_song_verdict": whole_D.get("verdict"), "whole_song_data": whole_D}
    return L, data


# ---------------------------------------------------------------------------- main
@ST.engine
@ST.cli
def main(argv=None, capture=None):
    ap = argparse.ArgumentParser(description="singer.py: stems, the singer, stems over time, meter")
    ap.add_argument("audio")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--keep-stems", metavar="DIR", help="also copy the four stems into DIR")
    ap.add_argument("--no-singer", action="store_true")
    ap.add_argument("--no-affect", action="store_true", help="skip audEERING affect analysis (also RAVEN_EARS_NO_AFFECT=1)")
    ap.add_argument("--no-meter", action="store_true")
    ap.add_argument("--no-stems-timeline", action="store_true")
    ap.add_argument("--no-stops", action="store_true", help="skip short accompaniment cuts (also: RAVEN_NO_STOPS=1)")
    ap.add_argument("--tempo", type=float, help="supplied BPM for STOPS lengths in beats")
    ap.add_argument("--beats-per-bar", type=float, help="supplied meter for STOPS recurrence in bars")
    ap.add_argument("--no-instruments", action="store_true", help="skip instrument naming (also: RAVEN_NO_INSTRUMENTS=1)")
    import credits
    credits.add_args(ap)
    import lrclib                          # LRCLIB hook: lyric text never replaces heard word times.
    lrclib.add_args(ap)
    ap.add_argument("--lyrics", metavar="FILE", help="given lyrics, one sung line per line; replaces lookup")
    import song_extras
    song_extras.add_args(ap)
    a = ap.parse_args(argv)
    local_audio_path(a.audio, ap)
    extras = song_extras.prepare(a, a.audio)
    credits.configure_from_args(a)
    lrclib.configure_from_args(a)
    path = os.path.abspath(a.audio)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such file: {path}")
    os.environ.setdefault("RAVEN_EARS_DEVICE", "auto")
    T = {}
    t0 = time.time()
    ST.progress("separating stems")
    sd, cached = get_stems(path, T)
    T["stems_total"] = round(time.time() - t0, 1)
    T["stems_cached"] = cached
    if a.keep_stems:
        os.makedirs(a.keep_stems, exist_ok=True)
        for s in SRC:
            shutil.copy(os.path.join(sd, f"{s}.wav"), os.path.join(a.keep_stems, f"{s}.wav"))
    import soundfile as sf
    dur = sf.info(os.path.join(sd, "vocals.wav")).duration
    out, J = [], {"file": path, "duration_s": round(dur, 1), "stems_dir": sd}
    out.append(f"SINGER.PY  {os.path.basename(path)}  {tm(dur)}  (stems {'cached' if cached else 'separated'} in {T['stems_total']}s)")

    if not a.no_singer:
        ST.progress("words and lyrics")
        t = time.time()
        share, act_s = vocal_presence(sd)
        J["vocal_share"] = round(share, 4)
        J["vocal_active_s"] = act_s
        # LRCLIB hook: show the source even when there are no lyric rows.
        import lyric_align
        given = lyric_align.read_lyrics_file(a.lyrics) if a.lyrics else None
        lnotes = []
        lyr = lrclib.resolve(path, lnotes, dur, given,
                            words_skip_reason="too little vocal-stem energy for transcription" if share < 0.02 or act_s < 8 else None)
        J["lyrics"] = {k: v for k, v in lyr.items() if k != "words"}
        J["lyrics_notes"] = lnotes
        out += lrclib.card_lines(lyr)
        out += ["  NOTE: " + note for note in lnotes]
        if share < 0.02 or act_s < 8:
            ST.mark("singer", "skipped", "too little vocal-stem energy")
            out.append("SINGER: instrumental (vocals stem holds %.1f%% of the energy, %.0f s of real level)" % (share * 100, act_s))
            J["singer"] = "instrumental"
        else:
            ST.progress("pitch and melody")
            pitch = {}
            kwargs = {"capture_pitch": pitch.update} if capture is not None else {}
            if lyr.get("quick_skipped"):
                kwargs["words_skip_reason"] = "skipped in a quick listen"
            lines, data, tim = singer_block(os.path.join(sd, "vocals.wav"), dur, lyr["words"], affect=not a.no_affect, **kwargs)
            if capture is not None:
                # Only listen-once capture needs the already measured track. No
                # stem paths or model calls are needed to derive notation later.
                from music import lyric_lines
                rows = lyr["rows"] or lyric_lines(lyr["words"] or [])
                J["melody_input"] = dict(pitch, words=lyr["words"],
                                         lyric_lines=[row["text"] for row in rows],
                                         lyrics_label=lyr["label"])
            out.append("SINGER:")
            out += ["  " + x for x in lines]
            J["singer"] = {"lines": lines, "data": data}
            T["singer"] = round(time.time() - t, 1)

    inst_res, inst_ev = None, []
    if not a.no_instruments and not os.environ.get("RAVEN_NO_INSTRUMENTS"):
        ST.progress("instruments")
        t = time.time()
        try:
            import instruments
            inst_res = instruments.analyse(path)
            inst_ev = instruments.events(inst_res)
            out += instruments.block(inst_res)
            J["instruments"] = inst_res
            J["instrument_events"] = [{"t": t_, "text": x} for t_, x in inst_ev]
        except Exception as e:
            ST.mark("instruments", "failed", f"{type(e).__name__}: {str(e)[:80]}")
            out.append(f"INSTRUMENTS: failed ({type(e).__name__}: {str(e)[:80]})")
        T["instruments"] = round(time.time() - t, 1)

    if not a.no_stems_timeline:
        t = time.time()
        sl, ev, refs, present = stems_timeline(sd, dur)
        out.append("STEMS OVER TIME (each stem vs its own level; on/off debounced at 4 s). Shorter cuts: see STOPS.")
        out += sl
        out.append("ENTRANCES/EXITS:")
        allev = sorted([(e["t"], e["text"]) for e in ev] + [(t_, x + "  [instrument tagger]") for t_, x in inst_ev],
                       key=lambda z: z[0])
        out += [f"  {tm(t_)} {x}" for t_, x in allev]
        J["stems"] = {"own_level_db": refs, "present": present, "events": ev}
        T["timeline"] = round(time.time() - t, 1)

    if not a.no_stops and os.environ.get("RAVEN_NO_STOPS") != "1":
        t = time.time()
        import stops
        result = stops.analyse(sd, a.tempo, a.beats_per_bar)
        out += stops.card_lines(result)
        J["stops"] = result
        ST.mark("stops", result["status"], result.get("reason", ""))
        T["stops"] = round(time.time() - t, 1)

    if not a.no_meter:
        ST.progress("meter and downbeats")
        t = time.time()
        try:
            ml, md = meter_sections(path, sd, dur)
            out += ml
            J["meter"] = md
        except Exception as e:
            ST.mark("meter", "failed", f"{type(e).__name__}: {str(e)[:80]}")
            out.append(f"METER: failed ({type(e).__name__}: {e})")
        T["meter"] = round(time.time() - t, 1)
    for name, disabled, flag in (("singer", a.no_singer, "--no-singer"),
                                 ("meter", a.no_meter, "--no-meter"),
                                 ("stems timeline", a.no_stems_timeline, "--no-stems-timeline"),
                                 ("stops", a.no_stops or os.environ.get("RAVEN_NO_STOPS") == "1", "--no-stops / RAVEN_NO_STOPS=1"),
                                 ("instruments", a.no_instruments or bool(os.environ.get("RAVEN_NO_INSTRUMENTS")), "instrument naming disabled")):
        if disabled:
            ST.mark(name, "skipped", flag)
    if a.no_instruments or os.environ.get("RAVEN_NO_INSTRUMENTS"):
        ST.mark("credits", "skipped", "instrument naming disabled", kind="input")
    if a.no_singer:
        ST.mark("words", "skipped", "--no-singer", kind="input")
        ST.mark("LRCLIB", "skipped", "--no-singer", kind="input")
    filename_notes = []
    credits.save_filename_reading(path, J, filename_notes)
    import song_length
    song_length.capture(path, J)
    if filename_notes:
        J["filename_notes"] = filename_notes
        out += ["  NOTE: " + note for note in filename_notes]
    J["stage_status"] = ST.snapshot()
    J["timings_s"] = T
    if extras:
        J["song_extras"] = extras
    out.append("timings: " + ", ".join(f"{k} {v}" for k, v in T.items()))
    from ears_handoff import finish_lines, finish_json  # MORE EARS: listener handoff.
    if capture is not None:
        capture(finish_json(dict(J)))
    if a.json:
        print(json.dumps(finish_json(J), default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=1, ensure_ascii=False))
    else:
        import instrumental_voice
        text = instrumental_voice.stem_text("\n".join(ST.add_notes(out, J["stage_status"])),
                                          instrumental_voice.designated(J))
        print(song_extras.decorate(song_length.annotate(text, J.get("length_check")), extras))
    return 0


if __name__ == "__main__":
    sys.exit(main())
