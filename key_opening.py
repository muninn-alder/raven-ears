"""An opening key estimate cannot be a departure from an established home."""
import math
import re


def _chord(name):
    if not isinstance(name, str):
        return None
    match = re.fullmatch(r"([A-G])([#b♭♯]?)(maj7|m7|m|7|°|5)?", name or "")
    if not match:
        return None
    note, accidental, quality = match.groups()
    root = ({"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[note]
            + {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}[accidental]) % 12
    return root, quality or ""


def _fits(chord, key):
    root, mode = key
    if mode == "minor" and (chord[0] - root) % 12 == 7 and chord[1] in ("", "7"):
        return True  # harmonic-minor dominant
    scale = (0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10)
    tones = {"": (0, 4, 7), "maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10),
             "m": (0, 3, 7), "m7": (0, 3, 7, 10), "°": (0, 3, 6), "5": (0, 7)}[chord[1]]
    return all((chord[0] + tone - root) % 12 in scale for tone in tones)


def short_intro(data, text=""):
    """A short foreign opening, inferred from saved chords; never a modulation.

    10 Nov 2026: a long decoded triad alone is not proof of a tonicised key.
    Name that chord literally; require a local V–I/i before calling it a key.
    Do not invent beat counts from a whole-song BPM (intro tempo can differ).
    """
    from stage_status import card_notes, source_notes
    if not isinstance(data, dict):
        return None
    stages = ("chords", "key fusion", "key by chords", "key")
    if any(str(row.get("stage", "")).lower() in stages and row.get("status") in ("failed", "skipped", "not run")
           for row in data.get("stage_status") or [] if isinstance(row, dict)):
        return None
    if any(re.match(r"(?:NOTE:\s*)?(?:chords|key fusion|key by chords|key):\s*(?:failed|skipped|unavailable|disabled|not run)\b", row, re.I)
           for row in card_notes(text, data) + source_notes(text)):
        return None
    verdict = data.get("key_verdict")
    if not isinstance(verdict, dict):
        return None
    home = re.fullmatch(r"([A-G][#b♭♯]?) (major|minor)", str(verdict.get("key") or ""))
    if not home or verdict.get("state") != "clear":
        return None
    home_key = (_chord(home[1])[0], home[2])
    events = data.get("chords") or []
    if not isinstance(events, list) or not events:
        return None
    if any(not isinstance(e, dict) or any(type(e.get(k)) not in (int, float) or not math.isfinite(e[k])
                                         for k in ("t0", "t1")) for e in events):
        return None
    if (not 0 <= events[0]["t0"] <= 0.5 or any(e["t1"] <= e["t0"] for e in events)
            or any(a["t1"] > b["t0"] + 0.1 for a, b in zip(events, events[1:]))):
        return None
    chords = [_chord(e.get("chord")) for e in events]
    first = chords[0]
    if (first is None or first[1] not in ("", "maj7", "m", "m7")
            or _fits(first, home_key) or (first[0] - home_key[0]) % 12 in (0, 7)):
        return None  # tonic, diatonic openings and opening V do not establish another key
    tonic_quality = ("", "maj7") if home[2] == "major" else ("m", "m7")
    landing = next((i for i, c in enumerate(chords) if c and c[0] == home_key[0] and c[1] in tonic_quality), None)
    if landing is None:
        return None
    start, end = events[0]["t0"], events[landing]["t0"]
    if not 3 <= end - start < 20 or end >= 20 or events[landing]["t1"] - end < 2:
        return None
    if any(c is None for c in chords[:landing]) or any(
            abs(a["t1"] - b["t0"]) > 0.1 for a, b in zip(events[:landing], events[1:landing + 1])):
        return None
    intro_key = (first[0], "minor" if first[1].startswith("m") and first[1] != "maj7" else "major")
    # The established home must also fit the following phrase, not just one hit.
    tail = [(min(e["t1"], end + 12) - max(e["t0"], end), c)
            for e, c in zip(events[landing:], chords[landing:]) if e["t0"] < end + 12 and e["t1"] > end]
    if sum(t for t, c in tail if c and _fits(c, home_key)) < 10:
        return None
    lengths = [e["t1"] - e["t0"] for e in events[:landing]]
    if (sum(t for t, c in zip(lengths, chords) if _fits(c, intro_key)) < 0.9 * (end - start)
            or sum(t for t, c in zip(lengths, chords) if c[0] == first[0]) < 0.5 * (end - start)):
        return None
    cadence = any(a[0] == (first[0] + 7) % 12 and a[1] in ("", "7")
                  and b[0] == first[0] and b[1] in (("", "maj7") if intro_key[1] == "major" else ("m", "m7"))
                  and events[i + 1]["t1"] - events[i + 1]["t0"] >= 1
                  for i, (a, b) in enumerate(zip(chords[:landing], chords[1:landing])))
    if not cadence and any(c != first for c in chords[:landing]):
        return None
    name = re.match(r"[A-G][#b♭♯]?", events[0]["chord"])[0] + " " + intro_key[1]
    def clock(t):
        t = round(t)
        return f"{t // 60}:{t % 60:02d}"
    return (f"intro {'in' if cadence else 'on'} {name}{'' if cadence else ' chord'} "
            f"({clock(start)}–{clock(end)}), then {home[0]}")


def intro_text(text, data):
    intro = short_intro(data, text)
    if intro:
        return re.sub(r"(?m)^(  ends on [^\n;]+); opens on [^\n]+$", lambda m: m[1] + "; " + intro, text)
    return text


def settled(kinfo, home, parse_key, opening_seconds=25.0):
    """Keep later moves, but omit the opening estimate and its return home.

    The windowed key detector can start away from the whole-song tonic. Until
    that opening returns home, a later estimate is still part of establishing
    home. This is a conservative time gate, not a new tonic detector.
    """
    result = dict(kinfo)
    changes = []
    establishing = False
    current = home
    suppressed = False
    for change in sorted(kinfo.get('changes', []), key=lambda c: c['at']):
        target = parse_key(change['to'])
        if change['at'] < opening_seconds:
            establishing = target != home
        elif establishing:
            establishing = target != home
        elif change.get('suppressed'):
            suppressed = True
            continue  # a withheld departure cannot create a subsequent return
        elif target != current:
            changes.append(dict(change))
        current = target
    if suppressed:
        result['raw_changes'] = [dict(c) for c in kinfo['changes']]
    result.pop('changes', None)
    if changes:
        result['changes'] = changes
    return result


def dominant_opening(events, home, names):
    """Name an opening V only when the decoded chords support V then tonic."""
    chords = [e for e in events if e.get('root') is not None]
    if not chords:
        return None
    first = chords[0]
    if (first['t0'] > 5 or (first['root'] - home[0]) % 12 != 7
            or first['q'] not in ('maj', 'dom7') or first.get('conf', 0) < 0.5):
        return None
    tonic_quality = ('maj', 'maj7') if home[1] == 'major' else ('min', 'min7')
    tonic = next((e for e in chords[1:] if e['root'] == home[0] and e['q'] in tonic_quality
                  and e.get('conf', 0) >= 0.5 and e['t1'] - e['t0'] >= 2), None)
    if tonic is None:
        return None
    return f"likely opens on V ({names[first['root']]}) before settling in {names[home[0]]}"
