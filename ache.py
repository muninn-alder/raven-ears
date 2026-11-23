#!/usr/bin/env python3
"""Possible ache points, from existing analysis only. No audio, models or cache writes.

    python ache.py analysis.json [--json]

`analyze(R)` reads harmony.run's dict (including `mel`), or harmony --json output.
New harmony JSON carries the computed `ache` result, so it can be rendered again
without storing the pitch track. Older JSON can supply chords/cadences/lyrics;
missing confidence is never filled in as certainty. See ACHE.md for thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys

import numpy as np

NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KEY_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]  # music.py's key spelling
PC = {n: i for i, n in enumerate(NAMES)}
PC.update({"C#": 1, "D#": 3, "F#": 6, "G#": 8, "A#": 10, "Cb": 11, "B#": 0})
QUALITY = {"maj": "", "min": "m", "dom7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim", "5": "5"}
RANK = {"unclear": 0, "lean": 1, "sure": 2}


def _time(t):
    s = max(0, round(t))
    return f"{s // 60}:{s % 60:02d}"


def _clean(text):
    return " ".join(str(text).split())


def _key(name):
    parts = str(name).replace("♭", "b").replace("♯", "#").split()
    if len(parts) == 2 and parts[0] in PC and parts[1] in ("major", "minor", "mixolydian", "blues-rock"):
        return PC[parts[0]], parts[1]
    return None


def _chord(e):
    return NAMES[e["root"]] + QUALITY[e["q"]]


def _certainty(e):
    conf = e.get("conf")
    if conf is None:
        return "unclear"
    return "sure" if conf >= 0.5 else "lean" if conf >= 0.25 else "unclear"


def _weaker(*values):
    return min(values, key=RANK.get)


def _events(R):
    if "evs" in R:
        return sorted(R["evs"] or [], key=lambda e: e["t0"])
    out = []
    for e in R.get("chords", []):
        m = re.fullmatch(r"([A-G](?:#|b|♭|♯)?)(maj7|m7|7|m|°|dim|5)?", e.get("chord") or "")
        if not m:
            out.append(dict(e, root=None, q=None))
            continue
        root = PC.get(m[1].replace("♭", "b").replace("♯", "#"))
        q = {None: "maj", "m": "min", "7": "dom7", "maj7": "maj7", "m7": "min7", "°": "dim", "dim": "dim", "5": "5"}[m[2]]
        out.append(dict(e, root=root, q=q))
    return sorted(out, key=lambda e: e["t0"])


def _lyrics(R):
    rows = (R.get("align") or {}).get("rows")
    if rows is None:
        rows = (R.get("lyrics_aligned") or {}).get("lines")
    return sorted(rows if rows is not None else R.get("lines", []), key=lambda l: l["t"])


def _high_notes(mel):
    """Top sustained pitches in this track, never an anatomical vocal range."""
    f0 = np.asarray(mel.get("f0", []), float)
    vp = np.asarray(mel.get("vp", []), float)
    if f0.ndim != 1 or f0.shape != vp.shape:
        raise ValueError("melody needs matching f0 and vp arrays")
    dt = float(mel.get("hop", 1024)) / float(mel.get("sr", 22050))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid melody frame interval")
    valid = np.isfinite(f0) & (f0 > 0) & (vp >= 0.8)
    if valid.sum() * dt < 4:
        return []
    pitch = np.zeros(len(f0))
    pitch[valid] = 69 + 12 * np.log2(f0[valid] / 440)
    low, high = np.percentile(pitch[valid], [10, 95])
    if high - low < 3:             # a monotone is not reaching upward
        return []
    held = valid & (pitch >= high - 1)
    edges = np.diff(np.r_[False, held, False].astype(int))
    out = []
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        if (b - a) * dt >= 0.8 and np.ptp(pitch[a:b]) <= 2:
            out.append((a * dt, b * dt))
    return out


import stage_status as ST  # HONEST CARD: keep missing evidence distinct from a completed scan.


@ST.stage("ache", classify=lambda value: [ST.outcome("ache",
    "skipped" if value["status"] != "ok" else "degraded" if value.get("notes") else "ran",
    "; ".join(value.get("notes", [])))])
def analyze(R):
    """Return {status, moments, notes}; scores rank evidence, not human feelings.

    Each moment exposes its evidence, weights and confidence. A supplied empty
    timeline means no candidates; absent/broken analysis means unavailable.
    """
    if not isinstance(R, dict):
        raise ValueError("analysis must be a JSON object")
    if "ache" in R:                 # already computed by harmony --json
        return R["ache"]
    notes = []
    if not any(R.get(k) is not None for k in ("evs", "chords", "mel", "dips", "cads", "cadences", "stem_events", "stems", "kinfo", "swells")):
        return {"status": "unavailable", "moments": [], "notes": ["analysis unavailable"]}
    evs, rows = _events(R), _lyrics(R)
    if ("evs" in R and R["evs"] is None) or ("chords" in R and R["chords"] is None):
        notes.append("chord analysis unavailable")
    elif evs and not any(e.get("root") is not None for e in evs):
        notes.append("chord evidence unclear; no harmonic candidates")
    verdict = R.get("key_verdict") or {}
    key_name = verdict.get("key") or verdict.get("working")
    home = _key(key_name)
    key_conf = "sure" if verdict.get("key") and verdict.get("state") in ("clear", "chords", "agree", "spectral") else "unclear"
    reference = "" if key_conf == "sure" else "; working reference, tonic unclear"
    spans = R.get("spans") or []
    changes = (R.get("kinfo") or {}).get("changes", [])

    def key_at(t):
        k = home
        for start, kk in spans:
            if 0 < start <= t:
                k = tuple(kk)
        if not spans:
            for c in changes:
                if c["at"] <= t:
                    k = _key(c["to"]) or k
        return k

    def key_quality(t):
        # Local spectral changes are estimates, even when the global key is clear.
        changed = any(c["at"] <= t for c in changes) or any(0 < start <= t for start, _ in spans)
        return _weaker(key_conf, "lean") if changed else key_conf

    evidence = []

    def add(t, end, kind, text, weight=2, confidence="lean"):
        if math.isfinite(t) and math.isfinite(end) and 0 <= t <= end:
            evidence.append(dict(t=float(t), end=float(end), kind=kind, text=text,
                                 weight=weight, confidence=confidence))

    for i, e in enumerate(evs):
        if e.get("root") is None or e.get("q") not in QUALITY or e["t1"] - e["t0"] < 0.4:
            continue
        t, end = e["t0"], e["t1"]
        k = key_at(t)
        if home and k and k[1] == "major" and e["q"] != "5":
            degree = (e["root"] - k[0]) % 12
            kind = "m" if e["q"] in ("min", "min7") else "M" if e["q"] in ("maj", "maj7", "dom7") else "d"
            roman = {(5, "m"): "iv", (8, "M"): "bVI", (10, "M"): "bVII"}.get((degree, kind))
            if roman:
                name = key_name if k == home else f"{KEY_NAMES[k[0]]} {k[1]}"
                add(t, end, "harmony", f"borrowed {roman} ({_chord(e)} in {name}{reference})",
                    confidence=_weaker(_certainty(e), key_quality(t)))
        # Use adjacent events, including unknowns: never bridge an unheard gap.
        prev = evs[i - 1] if i else None
        if home and k and prev and prev.get("root") is not None and prev.get("q") in ("maj", "dom7"):
            if (key_at(prev["t0"]) == k and prev["t1"] - prev["t0"] >= 0.4 and abs(prev["t1"] - t) < 0.15
                    and (prev["root"] - k[0]) % 12 == 7):
                target = (e["root"] - k[0]) % 12
                deceptive = (k[1] == "major" and target == 9 and e["q"] in ("min", "min7")) or (k[1] == "minor" and target == 8 and e["q"] in ("maj", "maj7"))
                if deceptive:
                    rn = "vi" if k[1] == "major" else "bVI"
                    add(t, end, "harmony", f"deceptive V→{rn} ({_chord(prev)}→{_chord(e)}{reference})",
                        confidence=_weaker(_certainty(prev), _certainty(e), key_quality(t)))

    for c in R.get("cads", R.get("cadences", [])):
        if c.get("cadence") in ("half", "open") and home and c.get("to"):
            text = "phrase ends on V and hangs" if c["cadence"] == "half" else f"phrase ends away from home ({c['to']})"
            add(c["t"], c.get("t_end", c["t"]), "harmony", text + reference,
                confidence=_weaker("lean", key_quality(c["t"])))

    for c in changes:
        add(c["at"], c["at"], "key", f"key area change to {_clean(c['to'])} (windowed estimate)")

    mel = R.get("mel")
    if mel is not None:
        try:
            for a, b in _high_notes(mel):
                source = mel.get("source", R.get("mel_source", "full mix"))
                voice = "voice near its highest sustained pitch in this track" if source == "vocal stem" else "full-mix melody near its highest sustained pitch (voice unclear)"
                add(a, b, "pitch", f"{voice}, held {b - a:.1f} s", confidence="lean")
        except (TypeError, ValueError) as e:
            notes.append(f"pitch analysis unavailable: {e}")
    else:
        notes.append("sustained vocal pitch not supplied")

    for d in R.get("dips", []):
        a, b = d["t0"], d["t1"]
        if b - a < 0.5:
            continue
        # A relative 12 dB dip is not silence. Only the backend's absolute-floor
        # evidence permits 'near-silence'; both remain measured dynamic signals.
        hush = "near-silence" if d.get("silent") else f"level dip ({d['depth']:.0f} dB)"
        add(a, a, "dynamics", f"drop to {hush} for {b - a:.1f} s", confidence="sure")
        if d.get("returns"):
            add(b, b, "dynamics", f"sound returns after {b - a:.1f} s {hush}", confidence="sure")
        for s in R.get("swells", []):
            if 0 <= s["t"] - b <= 3:
                add(s["t"], s["t"], "dynamics", f"swell +{s['rise']:.0f} dB after {b - a:.1f} s {hush}", weight=3, confidence="sure")

    # Optional singer JSON: a solo voice requires stem evidence, never a mix dip.
    stem_events = R.get("stem_events", (R.get("stems") or {}).get("events", []))
    for e in stem_events:
        if e.get("text") == "everything drops out but vocals":
            add(e["t"], e["t"], "texture", "band drops out to voice alone (stem activity)")

    def chord_at(t):
        return next((e for e in evs if e["t0"] <= t < e["t1"] and e.get("root") is not None and e.get("q") in QUALITY), None)

    # Compare the same relative position in exact repeated lines, not just their
    # starts: a changed chord may arrive halfway through a refrain.
    seen = {}
    for row in rows:
        text = _clean(row.get("text", ""))
        token = " ".join(re.findall(r"\w+", text.casefold()))
        end = row.get("end", row["t"])
        if token and token in seen and end > row["t"]:
            old = seen[token]
            old_end = old.get("end", old["t"])
            if old_end > old["t"]:
                scale = (end - row["t"]) / (old_end - old["t"])
                times = {row["t"]}
                for e in evs:
                    if row["t"] < e["t0"] < end:
                        times.add(e["t0"])
                    if old["t"] < e["t0"] < old_end:
                        times.add(row["t"] + (e["t0"] - old["t"]) * scale)
                for t in sorted(times):
                    before = chord_at(old["t"] + (t - row["t"]) / scale + 1e-6)
                    now = chord_at(t + 1e-6)
                    if (before and now and min(before["t1"] - before["t0"], now["t1"] - now["t0"]) >= 0.4
                            and (before["root"], before["q"]) != (now["root"], now["q"])):
                        approx = row.get("approx") or old.get("approx")
                        qualifier = "≈ placed lyric timing" if approx else "rough lyric timing"
                        add(t, t, "repetition", f"repeated line, chord {_chord(before)}→{_chord(now)} ({qualifier})",
                            confidence=_weaker("lean", _certainty(before), _certainty(now)))
                        break
        if token:
            seen[token] = row

    groups = []
    for anchor in evidence:
        # Fixed two-second neighbourhood, with actual overlaps for held notes.
        # Never chain nearby events until unrelated parts of a verse merge.
        nearby = [e for e in evidence if abs(e["t"] - anchor["t"]) <= 2 or
                  (e["kind"] == "pitch" and e["t"] <= anchor["t"] < e["end"])]
        families = {}
        for e in sorted(nearby, key=lambda e: (-e["weight"], -RANK[e["confidence"]], abs(e["t"] - anchor["t"]), e["t"])):
            families.setdefault(e["kind"], e)
        chosen = list(families.values())
        score = sum(e["weight"] for e in chosen)
        conf = _weaker(*(e["confidence"] for e in chosen))
        if conf == "sure" and len(chosen) < 2:
            conf = "lean"
        groups.append(dict(t=anchor["t"], score=score, confidence=conf, evidence=chosen))
    selected = []
    for g in sorted(groups, key=lambda g: (-g["score"], -RANK[g["confidence"]], g["t"])):
        if g["score"] >= 2 and all(abs(g["t"] - s["t"]) >= 8 for s in selected):
            selected.append(g)
        if len(selected) == 3:
            break
    for g in selected:
        row = next((r for r in rows if r["t"] <= g["t"] < r.get("end", r["t"])), None)
        if row:
            quote = _clean(row["text"]).replace('"', "'")
            g["lyric"] = ("≈ " if row.get("approx") else "") + '"' + quote[:100] + ('…' if len(quote) > 100 else '') + '"'
            if row.get("approx") or row.get("unsure") or "unsure:" in quote.lower():
                g["confidence"] = _weaker(g["confidence"], "lean")
    if evs and not home:
        notes.append("key verdict unavailable; no key-relative claims")
    notes.extend(_clean(n) for n in R.get("notes", []) if any(w in str(n).lower() for w in ("fail", "unavailable", "skipped")))
    return {"status": "ok", "moments": sorted(selected, key=lambda g: g["t"]), "notes": list(dict.fromkeys(notes))}


def render(result):
    lines = []
    for g in result["moments"]:
        text = ", ".join(e["text"] + (f" at {_time(e['t'])}" if abs(e["t"] - g["t"]) > 0.25 else "")
                         for e in g["evidence"])
        if g.get("lyric"):
            text += " under " + g["lyric"]
        lines.append(f"ACHE {_time(g['t'])}  {text} ({g['confidence']})")
    if not lines:
        lines.append("ACHE none found" if result["status"] == "ok" else "ACHE unclear: analysis unavailable")
    lines.extend("  ACHE NOTE: " + _clean(n) for n in result.get("notes", []))
    return lines


def block(R):
    """A broken ache backend must remain visible on the card."""
    try:
        return render(analyze(R))
    except Exception as e:
        ST.mark("ache", "failed", f"{type(e).__name__}: {_clean(e)[:120]}")
        return [f"ACHE unclear: analysis failed ({type(e).__name__}: {_clean(e)[:120]})"]


@ST.engine
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("analysis", help="existing analysis JSON; never an audio file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        with open(args.analysis, encoding="utf-8") as f:
            result = analyze(json.load(f))
        lines = render(result)       # validate saved results inside the failure boundary
        code = 0
    except Exception as e:
        ST.mark("ache", "failed", f"{type(e).__name__}: {_clean(e)[:120]}")
        result = {"status": "unavailable", "moments": [], "notes": [f"analysis failed ({type(e).__name__}: {_clean(e)[:120]})"]}
        lines = render(result)
        code = 1
    result["stage_status"] = ST.merge(result.get("stage_status", []), ST.snapshot())
    if args.json:
        print(json.dumps(dict(result, listener="and now, you"), ensure_ascii=False, indent=2))
    else:
        print("\n".join(ST.add_notes(lines, result["stage_status"])))
    return code


if __name__ == "__main__":
    sys.exit(main())
