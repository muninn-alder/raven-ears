#!/usr/bin/env python3
"""A short listening card from saved text and optional existing analysis.

    python slim.py card.txt [--analysis analysis.json]
    python slim.py - --full-card /path/to/saved/card.txt

No audio, models, network or cache writes. Text wins over supplemental JSON.
Missing evidence stays missing; the listener supplies the feeling.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sys
import textwrap

INVITATION = "and now, you"
MISSING = "not on this card"
TIME = r"\d+:\d{2}(?:\.\d+)?"
FAILURE = re.compile(r"\b(?:failed|unavailable|skipped|disabled|traceback|cannot run|could not load)\b|\b\w*Error:", re.I)
NO_VOICE_MARK = " (no voice measured here)"


def clean(text):
    return " ".join(str(text).split())


def seconds(stamp):
    m, s = stamp.split(":")
    return int(m) * 60 + float(s)


def clock(t):
    s = max(0, round(float(t)))
    return f"{s // 60}:{s % 60:02d}"


def finite(x):
    """A real, finite number from supplemental JSON, else None (never a bool or string)."""
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return float(x) if ok else None


def token(text):
    text = re.sub(r"\(unsure[^)]*\)", "", text, flags=re.I)
    return " ".join(re.findall(r"\w+", text.casefold().replace("’", "'")))


def split_voice_mark(text):
    """Separate a printed lyric annotation, including the old quoted form."""
    if text == NO_VOICE_MARK.strip():
        return "", True
    words = text.removesuffix(NO_VOICE_MARK)
    if words != text:
        return words, True
    if text.endswith((NO_VOICE_MARK + '"', NO_VOICE_MARK + '”')):
        return text[:-len(NO_VOICE_MARK) - 1] + text[-1], True
    return text, False


@dataclass
class Lyric:
    t: float
    text: str
    approx: bool = False
    end: float | None = None
    no_voice: bool = False

    def quote(self):
        mark = NO_VOICE_MARK if self.no_voice else ""
        return f'[{clock(self.t)}{"≈" if self.approx else ""}] “{self.text}”{mark}'


@dataclass
class Card:
    title: str = MISSING
    length: str = MISSING
    facts: dict = field(default_factory=dict)
    blocks: dict = field(default_factory=lambda: defaultdict(list))
    lyrics: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def parse(text):
    """Read only known blocks. In particular, a quoted lyric is never a failure."""
    card = Card()
    block, aligned = "", []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("==================="):
            if "and now, you" in s:
                break
            block = ""
            continue
        if s == "and now, you":
            block = ""
            continue
        if s.startswith("# ") and card.title == MISSING:
            card.title = re.sub(r"\s+\(\d{4}-\d\d-\d\d.*\)$", "", s[2:])
        m = re.match(rf"(?:MUSIC:|HARMONY:)\s*(.*?)\s*·\s*({TIME})$", s)
        singer = re.match(rf"SINGER\.PY\s+(.*?)\s+({TIME})(?:\s|$)", s)
        if m or singer:
            title, length = (m or singer).groups()
            if card.title == MISSING:
                card.title = re.sub(r"\.(mp3|wav|flac|m4a|ogg)$", "", title, flags=re.I)
            if card.length == MISSING:
                card.length = length
            block = ""
            continue
        if s.startswith('KEY CHANGES:'):
            card.blocks['key changes'].append(s)
            block = ''
            continue
        m = re.match(r"(KEY|TEMPO|SHAPE|VOCAL|VOCALS|SOUND):\s*(.*)", s)
        if m:
            card.facts.setdefault(m[1], m[2])
            if FAILURE.search(m[2]):
                card.warnings.append(s)
            block = "shape" if m[1] == "SHAPE" else ""
            continue
        lyric = re.match(rf"\[({TIME})(≈)?\]\s*(.*)", s)
        if lyric and block in ("lyrics", "lyric timeline"):
            t, approx, body = lyric.groups()
            if not body.startswith("♪"):
                if block == "lyric timeline":
                    body = body.partition(" — ")[0]
                body, no_voice = split_voice_mark(body)
                if block == "lyric timeline" and body.startswith(('“', '"')) and body.endswith(('”', '"')):
                    body = body[1:-1]
                row = Lyric(seconds(t), body, bool(approx), no_voice=no_voice)
                (aligned if block == "lyric timeline" else card.lyrics).append(row)
            continue
        if s.startswith("LYRICS:"):
            if block != "lyric timeline":
                block = "lyrics"
        elif s.startswith("ACHE"):
            card.blocks["ache"].append(s)
            block = "ache"
        elif s.startswith("METER:"):
            card.facts.setdefault("METER", s.removeprefix("METER:").strip())
            block = "meter"
        elif s.startswith("VOCAL-STEM MELODY:"):
            block = "singer"
            card.blocks[block].append(s)
        elif s.startswith("SINGER:"):
            block = "singer"
            if s != "SINGER:":
                card.blocks[block].append(s.removeprefix("SINGER:").strip())
        elif s.startswith("STOPS:"):
            block = "stops"
            card.blocks[block].append(s)
        elif s.startswith("INSTRUMENTS"):
            block = "instruments"
            if s.startswith("INSTRUMENTS:"):
                card.blocks[block].append(s.removeprefix("INSTRUMENTS:").strip())
        elif s.startswith("CREDITS"):
            block = "credits"
        elif s in ("CHORDS", "TENSION", "TIMELINE", "LYRIC TIMELINE", "ENTRANCES/EXITS:"):
            block = s.lower().rstrip(":")
        elif s.startswith(("CONTRAST (", "CONTRAST:")):
            block = "contrast"
            if s.startswith("CONTRAST:"):
                card.blocks[block].append(s)
        elif s.startswith(("SURPRISE (", "SURPRISE:")):
            block = "surprise"
            if s.startswith("SURPRISE:"):
                card.blocks[block].append(s)
        elif s.startswith("CHILL CANDIDATES"):
            block = "chills"
            if s.startswith("CHILL CANDIDATES:"):
                card.blocks[block].append(s)
        elif s and not line[0].isspace():
            block = ""
        elif s and block:
            card.blocks[block].append(s)
        # Failure prose is kept, but lyrics and model sentiment quotes cannot
        # diagnose a backend. No absence is translated into silence.
        probe = s.split("'", 1)[0] if s.startswith("NOTE:") and "lyrics:" in s.lower() else s
        if FAILURE.search(probe) and (s.startswith(("NOTE:", "ACHE", "INSTRUMENTS:", "METER:", "SINGER:", "STOPS:", "SURPRISE:", "CONTRAST:", "CHILL CANDIDATES:", "["))
                                  or re.match(r"(?:Traceback|\w*Error:|unavailable:|failed:|skipped:|disabled:)", s, re.I)):
            card.warnings.append((block.upper() + ": " if block and probe.startswith(("unavailable:", "failed:", "skipped:", "disabled:")) else "") + probe)
    if aligned:
        card.lyrics = aligned
    if "KEY" not in card.facts:
        home = next((s[5:].strip() for s in card.blocks["chords"] if s.startswith("home:")), "")
        if home:
            card.facts["KEY"] = home.split(";", 1)[0].split(". Numerals", 1)[0]
    # Duplicate engine reports must not inflate the refrain count.
    card.lyrics = sorted({(r.t, r.text, r.approx): r for r in card.lyrics}.values(), key=lambda r: r.t)
    return card


def supplement(card, data):
    """Fill absent blocks from music/harmony/singer JSON, without deciding facts."""
    if not isinstance(data, dict):
        raise ValueError("analysis must be a JSON object")
    import instrumental_voice
    instrumental = instrumental_voice.designated(data)
    verdict = data.get("key_claim") or data.get("key_verdict") or {}
    source_key = card.facts.get("KEY", "")
    supplied_key = verdict.get("line") or verdict.get("key") or verdict.get("working") or ""
    key_names = lambda text: set(re.findall(r"[A-G][#b♭♯]? (?:major|minor|mixolydian|blues-rock)", text))
    if source_key and supplied_key:
        uncertain = re.search(r"unclear|between|working|\blean\b", source_key, re.I)
        supplied_clear = bool(verdict.get("key")) and verdict.get("state") in ("clear", "chords", "agree", "spectral")
        if key_names(source_key) != key_names(supplied_key) or (uncertain and supplied_clear):
            raise ValueError("supplemental key disagrees with card; use analysis from this run")
    if card.title == MISSING and data.get("file"):
        card.title = Path(data["file"]).stem
    dur = data.get("duration", data.get("dur", data.get("duration_s")))
    if card.length == MISSING and dur is not None:
        card.length = clock(dur)
    if verdict.get("line"):
        card.facts.setdefault("KEY", verdict["line"])
    elif verdict.get("key") or verdict.get("working"):
        confidence = "sure" if verdict.get("key") and verdict.get("state") in ("clear", "chords", "agree", "spectral") else "unclear"
        card.facts.setdefault("KEY", f"{verdict.get('key') or verdict['working']} ({confidence})")
    if not card.blocks['key changes']:
        import key_changes
        card.blocks['key changes'] = key_changes.lines(data.get('key_change_scan'))
    tempo = data.get("tempo")
    if isinstance(tempo, dict) and tempo.get("bpm") is not None:
        steady = f"steady {tempo['regularity']:.2f}" if tempo.get("regularity") is not None else "steadiness " + MISSING
        card.facts.setdefault("TEMPO", f"{tempo['bpm']:g} BPM · {steady}")
    if data.get("meter_claim"):
        card.facts.setdefault("METER", str(data["meter_claim"]) + " (confidence not on this card)")
    secs = data.get("sections", data.get("secs", [])) or []
    if not card.blocks["shape"] and "SHAPE" not in card.facts:
        for sec in secs:
            card.blocks["shape"].append(f"{clock(sec['start'])}–{clock(sec['end'])} {sec['level']}" + (" <- PEAK" if sec.get("peak") else ""))
    if not card.lyrics:
        aligned = data.get("lyrics_aligned") or data.get("align") or {}
        rows = aligned.get("lines", aligned.get("rows"))
        if rows is None:
            rows = data.get("lyrics", data.get("lines", []))
        if isinstance(rows, dict):
            rows = rows.get("rows")
        for r in rows or []:
            words, no_voice = split_voice_mark(clean(r["text"]))
            if r.get("unsure") and "unsure" not in words.lower():
                words += " (unsure: supplied analysis)"
            card.lyrics.append(Lyric(float(r["t"]), words, bool(r.get("approx")), r.get("end"),
                                     bool(r.get("no_voice") or no_voice)))
        card.lyrics.sort(key=lambda r: r.t)
    if not card.blocks["timeline"]:
        card.blocks["timeline"] = [f"{clock(r['t'])} {clean(r['text'])}" for r in data.get("timeline", [])]
    if not card.blocks["tension"]:
        for b in data.get("builds_releases", data.get("br", [])):
            card.blocks["tension"].append(f"{b['kind'].upper()} {clock(b['t0'])}–{clock(b['t1'])} tension "
                                          f"{'climbs' if b['kind'] == 'build' else 'eases'} {b['from']:.2f}→{b['to']:.2f}")
        tc = data.get("tension") or {}
        values = tc.get("values") or []
        if values:
            peak = max(range(len(values)), key=values.__getitem__)
            card.blocks["tension"].append(f"tension peak at {clock(peak * tc['step'])}")
    if not card.blocks["contrast"]:
        for c in data.get("contrast", []):
            if "jev" in c:
                import jev_reads
                if c.get("flag") or jev_reads.display_reason(c):
                    card.blocks["contrast"].append(jev_reads.contrast_line(c, clock(c['t'])))
            elif c.get("flag"):
                card.blocks["contrast"].append(f"{clock(c['t'])} '{clean(c['text'])}': {c['flag']}")
        if any((c.get("jev") or {}).get("status") == "ran" for c in data.get("contrast", [])):
            import jev_reads
            card.blocks["contrast"].append(jev_reads.EXPLANATION)
            card.blocks["contrast"].extend(jev_reads.noul_summaries(data["contrast"], clock))
    singer = data.get("singer")
    if not card.blocks["singer"]:
        if isinstance(singer, dict):
            card.blocks["singer"] = list(singer.get("lines", []))
        elif isinstance(singer, str):
            card.blocks["singer"] = [singer]
    if not card.blocks["entrances/exits"]:
        events = (data.get("stems") or {}).get("events", []) + data.get("instrument_events", [])
        card.blocks["entrances/exits"] = [f"{clock(e['t'])} {clean(e['text'])}" for e in sorted(events, key=lambda e: e["t"])]
    if not card.blocks["ache"]:
        import ache
        card.blocks["ache"] = ache.block(data)
    import lyrics_shared
    evidence = "VOCALS: " + card.facts.get("VOCALS", "")
    for block in ("singer", "timeline"):
        card.blocks[block] = lyrics_shared.clean_gap_claims("\n".join(card.blocks[block]), evidence, data).splitlines()
    if instrumental:
        for block in ("singer", "timeline"):
            rows = card.blocks[block]
            if block == "singer" and rows and not any(s.startswith("VOCAL-STEM MELODY:") for s in rows):
                rows = ["SINGER:"] + ["  " + s for s in rows]
            card.blocks[block] = [s.strip() for s in instrumental_voice.stem_text("\n".join(rows), True).splitlines()]
    card.warnings.extend(clean(n) for n in data.get("notes", []) if FAILURE.search(str(n)))


def timed(rows):
    return [s for s in rows if re.match(TIME + r"\s", s)]


def strongest(rows, count, score):
    return sorted(enumerate(rows), key=lambda item: (-score(item[1]), item[0]))[:count]


def delta(text):
    m = re.search(r"(\d+\.\d+)→(\d+\.\d+)", text)
    return abs(float(m[2]) - float(m[1])) if m else 0


def confidence(text):
    return text if re.search(r"clear|sure|lean|likely|confidence|ambiguous|moderate|low|high", text, re.I) else text + " (confidence not on this card)"


def _feel_tempo(text):
    """Keep the reading and qualifiers; arithmetic stays on the full card."""
    return " · ".join(part for part in text.split(" · ")
                      if not re.match(r"\d+ beats × 60 / .*; \d+-beat sections cover \d+%$", part))


def header(card):
    tempo = _feel_tempo(card.facts.get("TEMPO", "tempo " + MISSING))
    # Meter has one authority. Keep half-time and uncertainty notes from TEMPO.
    tempo = " · ".join(p.strip() for p in tempo.split(" · ") if not p.startswith(("beat strength", "meter ")))
    meter = card.facts.get("METER")
    if not meter:
        m = re.search(r"meter (.*?)(?:, see METER|$)", card.facts.get("TEMPO", ""))
        meter = m[1] + " (confidence not on this card)" if m else MISSING
    key = confidence(card.facts["KEY"]) if card.facts.get("KEY") else MISSING
    if re.search(r"unclear|between|working|\blean\b", key, re.I):
        key += " (harmonic references below are working only)"
    return f"{card.title} · {card.length} · {tempo} · meter {meter} · key {key}"


def shape(card):
    arc = card.facts.get("SHAPE")
    if arc:
        parts = arc.split(" → ")
        if not re.search(TIME, parts[0]):
            first = re.match(TIME, next(iter(card.blocks["shape"]), ""))
            parts[0] = first[0] + " " + parts[0] if first else parts[0] + " (time not on this card)"
    else:
        parts = [re.sub(r"\s+-?\d+(?:\.\d+)? dB.*", "", s) for s in card.blocks["shape"]]
    out = ["THE SHAPE"]
    out += ["  " + " → ".join(parts[i:i + 4]) for i in range(0, len(parts), 4)] if parts else ["  " + MISSING]
    tension = card.blocks["tension"]
    for kind in ("BUILD", "RELEASE"):
        rows = [s for s in tension if s.startswith(kind)]
        out.append("  " + (max(rows, key=delta) if rows else kind.lower() + ": " + MISSING))
    return out


def chills(card):
    group, rows = "", []
    for s in card.blocks["chills"]:
        if re.match(TIME + r"\s", s):
            # Retain source qualifiers: timbre novelty is not an instrument ID.
            rows.append((group, s))
        elif s.endswith(":"):
            group = s[:-1]
    def strength(row):
        group, s = row
        m = re.search(r"held ([\d.]+) s|\+([\d.]+) dB", s)
        return (2 if "chromatic" in s or "deceptive" in s else 1,
                float(next(x for x in m.groups() if x)) if m else 0)

    chosen = []
    groups = list(dict.fromkeys(g for g, _ in rows))
    for group in groups:
        chosen.append(max((row for row in rows if row[0] == group), key=strength))
        if len(chosen) == 3:
            break
    for row in sorted(rows, key=strength, reverse=True):
        if len(chosen) == 3:
            break
        if row not in chosen:
            chosen.append(row)
    out = []
    for g, s in chosen:
        if g.startswith("appoggiaturas"):
            g = "leaning note; vocal-stem pitch hint" if "vocal stem" in g else "leaning note; approximate full-mix pitch hint, voice unclear"
        elif g.startswith("new voice"):
            g = "timbre-novelty proxy; instrument identity unclear"
        out.append(f"{s} ({g})" if g else s)
    failed = next((s for s in card.blocks["chills"] if FAILURE.search(s)), "")
    return out or [failed or (MISSING if not card.blocks["chills"] else "no clear candidates reported (or pitch too unsteady to say)")]


def surprise_score(s):
    if "refrain check:" in s:
        return -1  # a verified repetition is not a surprise
    if "key itself moves" in s or "breaks the count" in s:
        return 5
    if "a stop:" in s:
        return 4
    if "after" in s and "novelty peak" not in s:
        return 3
    change = re.search(r"\(([+-]?\d+) dB", s)
    return (2 if "unscheduled" in s else 1) + min(abs(int(change[1])) / 100, 0.9) if change else 1


def contrast_score(s):
    jev = re.search(r"Jev ([+-]?[\d.]+)", s)
    if "(roberta)" in s or jev:
        words = re.search(r"words ([+-]?[\d.]+) \(roberta\)", s)
        masked = re.search(r"Jev reads masking \(([\d.]+)\)", s)
        lift = re.search(r"lift ([+-]?[\d.]+)", s)
        return max(float(masked[1]) + 2 if masked else 0,
                   abs(float(words[1]) - float(jev[1])) if words and jev else 0,
                   abs(float(lift[1]) - float(words[1])) if words and lift else 0)
    w = re.search(r"words \w+ \(([+-]?[\d.]+)\)", s)
    m = re.search(r"lift ([+-]?[\d.]+)", s)
    return abs(float(w[1]) - float(m[1])) if w and m else 0


def fit_contrast(text, width):
    """Reserve compact prefixes/qualifiers without separating a Jev value from its label."""
    if " · Jev " not in text or len(text) <= width:
        return text
    start, end = text.find("“"), text.rfind("” — ")
    if start < 0 or end <= start:
        return text
    room = width - (len(text) - (end - start - 1))
    if room < 1:
        # Three findings can leave no room even for an ellipsis after CONTRAST:.
        text = text[:end] + text[end:].replace("Jev reads masking", "masking", 1)
        room = width - (len(text) - (end - start - 1))
    if room < 1:
        return text
    quote = text[start + 1:end]
    return text[:start + 1] + quote[:room - 1].rstrip() + "…" + text[end:]


def lyric_qualifiers(card, text):
    """Carry placed timing and doubtful words onto lyric-derived summaries."""
    stamp = re.match(TIME, text)
    if not stamp:
        return text
    t = seconds(stamp[0])
    rows = [r for r in card.lyrics if abs(r.t - t) < 0.6 or (r.end is not None and r.t <= t < r.end)]
    if any(r.approx for r in rows) and not text[stamp.end():].startswith("≈"):
        text = text[:stamp.end()] + "≈" + text[stamp.end():]
    if any("unsure" in r.text.lower() for r in rows) and "unsure" not in text.lower():
        text += " (lyric unsure on this card)"
    return text


def voice(card):
    rows = card.blocks["singer"]
    pitch = next((s for s in rows if s.startswith("pitch:")), "range: " + MISSING)
    entrances = card.blocks["entrances/exits"]
    entry = next((s for s in entrances if re.search(r"\bvocals (?:enter|return)\b", s)), "")
    if not entry:
        measured = next((s for s in card.blocks["timeline"] if
                         "voice enters:" in s and s.endswith((" (words approximate here)", " (time approximate)",
                                                              ": vocal stem active"))), "")
        lyric = re.compile(rf"^({TIME})\s+(?:voice enters:|first timed lyric \(voice onset unclear\):)")
        timed_lyric = next((m for s in card.blocks["timeline"] if (m := lyric.match(s))), None)
        if measured:
            entry = measured.split("voice enters:", 1)[0] + "vocals enter (vocal stem)"
        elif timed_lyric:
            entry = lyric_qualifiers(card, timed_lyric[1] + " first timed lyric (voice onset unclear)")
        else:
            entry = "entry: " + MISSING
    instrumental = next((s for s in rows + [card.facts.get("VOCALS", "")] if "instrumental" in s), "")
    if instrumental:
        entry = instrumental
    stem = next((s for s in rows if s.startswith("VOCAL-STEM MELODY:")), "")
    if stem:
        entry = stem
    texture = card.facts.get("VOCAL", "texture: " + MISSING)
    moment = next((s for s in rows if s.startswith("most intense moment:")), "biggest vocal moment: " + MISSING)
    gaps = [s for s in rows + card.blocks["timeline"]
            if any(desc in s for desc in ("voice present, words ", "wordless singing (lyrics:",
                                         "voice present; lyrics have words here"))]
    arousal = [s for s in rows if s.startswith("AROUSAL:")] if stem else []
    return [entry + " · " + pitch, texture, moment] + arousal + gaps


def peak(card):
    for s in card.blocks["tension"]:
        m = re.search(r"peak at (" + TIME + ")", s)
        if m:
            return seconds(m[1]), "tension peak"
    for s in card.blocks["shape"]:
        if "PEAK" in s:
            m = re.match(TIME, s)
            if m:
                return seconds(m[0]), "loudest section start"
    return None, "biggest musical moment"


def words(card):
    rows = card.lyrics
    groups = defaultdict(list)
    for r in rows:
        if token(r.text):
            groups[token(r.text)].append(r)
    repeated = [g for g in groups.values() if len(g) > 1]
    out = []
    if repeated:
        g = max(repeated, key=lambda g: (len(g), len(token(g[0].text))))
        times = ", ".join(clock(r.t) + ("≈" if r.approx else "") for r in g[:3])
        if len(g) > 3:
            times += f", +{len(g) - 3} more"
        caution = "; some copies unsure" if any("unsure" in r.text.lower() for r in g) else ""
        out.append(f'hook lean: “{g[0].text}” ({len(g)} exact line repeats on this card{caution}; {times})')
    else:
        out.append("hook: no repeated whole line on this card" if rows else "hook: " + MISSING)
    out.append("first: " + (rows[0].quote() if rows else MISSING))
    t, name = peak(card)
    if t is None:
        out.append(name + ": " + MISSING)
    else:
        covering = [r for r in rows if r.end is not None and r.t <= t < r.end]
        nearby = min(rows, key=lambda r: abs(r.t - t)) if rows else None
        if covering:
            out.append(f"at {name} {clock(t)}: " + covering[0].quote())
        elif nearby and abs(nearby.t - t) <= 6:
            out.append(f"near {name} {clock(t)}: " + nearby.quote() + " (nearby line start; overlap unclear)")
        else:
            out.append(f"at {name} {clock(t)}: no timed line on this card")
    return out


def instruments(card):
    pool, heard, unheard, group = [], [], [], ""
    for s in card.blocks["instruments"]:
        if s == "overall:":
            group = "overall"
        elif s.startswith("who is playing"):
            group = ""
        elif group == "overall":
            # Strip numeric bookkeeping, keeping the tagger's uncertainty and
            # the distinction between a credited name and the broader sound.
            pool.append(clean(re.split(r"\s+\((?:score|\d+%|energy)", s)[0]))
        elif FAILURE.search(s):
            pool.append(s)
    for s in card.blocks["credits"]:
        if s in ("credited and heard:", "credited, not clearly heard:", "heard, not credited:"):
            group = s.rstrip(":")
        elif group == "credited, not clearly heard":
            unheard.append(s.split(" (")[0].split(":")[0])
        elif group == "credited and heard":
            heard.append(s.split(" (")[0].split(":")[0])
    entrances = [s for s in card.blocks["entrances/exits"] if "[instrument tagger]" in s and re.search(r"enters|arrives|steps up", s)]
    # Prefer a measured guitar arrival to a generic early entrance. Otherwise
    # retain source order rather than manufacture an importance score.
    entrance = next((s for s in entrances if "arrives" in s or "steps up" in s), next(iter(entrances), ""))
    out = ["heard pool (estimates): " + ("; ".join(pool) if pool else MISSING)]
    credit = []
    if heard:
        credit.append("credited and heard: " + ", ".join(heard))
    if unheard:
        credit.append("credited, not clearly heard: " + ", ".join(unheard))
    if credit:
        out.append("; ".join(credit))
    out.append("notable entrance: " + (entrance or MISSING))
    return out


def render(text="", full_card=None, analysis=None):
    """Return slim text; never change the source text or any file."""
    import lyric_align
    import lyrics_shared
    from stage_status import names_note_text
    text = names_note_text(text)
    text, analysis = lyric_align.clean_writer_card(text, analysis)
    text = lyrics_shared.clean_gap_claims(text, analysis=analysis)
    import instrumental_voice
    text = instrumental_voice.saved_peak_pitch(text, analysis)
    instrumental = (instrumental_voice.designated(analysis) if analysis is not None
                    else instrumental_voice.text_designation(text))
    text = instrumental_voice.stem_text(text, instrumental)
    import jev_reads
    text = jev_reads.refresh_contrast(text, analysis)
    import keyfuse
    data = analysis if isinstance(analysis, dict) else {}
    claim = data.get("key_verdict") or data.get("key_claim")
    key = keyfuse.claimed_spelling_key(claim) if claim else keyfuse.printed_spelling_key(text)
    at = keyfuse.spelling_context(key, keyfuse.spelling_spans(data.get("key_change_scan")))
    text = keyfuse.harmony_note_labels(keyfuse.singer_note_labels(text, at), at)
    card = parse(text)
    if analysis is not None:
        try:
            supplement(card, analysis)
        except Exception as e:
            card.warnings.append(f"supplemental analysis failed ({type(e).__name__}: {clean(e)})")
    out = [header(card), *card.blocks['key changes'], ""] + shape(card)
    ache = card.blocks["ache"]
    out += [s for s in ache if s.startswith("ACHE ")][:2] or ["ACHE: " + MISSING]
    out += ["CHILL CANDIDATES (hints, not feelings)"] + ["  " + s for s in chills(card)]
    surprises = [s for s in timed(card.blocks["surprise"]) if surprise_score(s) > 0]
    picks = [s for _, s in strongest(surprises, 2, surprise_score)]
    fallback = next((s for s in card.blocks["surprise"] if "keeps its own pattern" in s or FAILURE.search(s)), MISSING)
    out += ["SURPRISE"] + ["  " + s for s in picks or [fallback]]
    contrasts = timed(card.blocks["contrast"])
    picks = [fit_contrast(lyric_qualifiers(card, s), 116) for _, s in strongest(contrasts, 2, contrast_score)]
    fallback = next((s for s in card.blocks["contrast"] if s.startswith(("no lyrics", "no transcribed lyric", "no line", "overall:")) or FAILURE.search(s)), MISSING)
    out += ["CONTRAST (words vs music; model hints)"] + ["  " + s for s in picks or [fallback]]
    out += ["  " + s for s in card.blocks["contrast"] if s.startswith("Jev = ")]
    out += ["  " + s for s in card.blocks["contrast"] if jev_reads.is_noul_summary(s)]
    out += ["VOCAL-STEM MELODY" if instrumental else "VOICE"] + ["  " + s for s in voice(card)]
    origin = lyrics_shared.displayed_lyric_source(analysis, text, has_words=bool(card.lyrics))
    source_note = {"file": " · supplied lyrics", "synced": " · lyrics from LRCLIB",
                   "plain": " · lyrics from LRCLIB", "whisper": " · Whisper's guesses"}.get(origin, "")
    out += ([instrumental_voice.WORDS] if instrumental else
            ["WORDS (rough line times; ≈ placed, not heard)" + source_note] + ["  " + s for s in words(card)])
    out += ["INSTRUMENTS"] + ["  " + s for s in instruments(card)]
    warnings = list(dict.fromkeys(card.warnings + [s for s in ache if FAILURE.search(s)]))
    # Summarise reported failures, while keeping every failed stage visible.
    missing = [label for label, present in (("music", any(k in card.facts for k in ("TEMPO", "SHAPE", "SOUND", "VOCAL", "VOCALS"))), ("singer", bool(card.blocks["singer"])),
               ("harmony", bool(card.blocks["tension"] or card.blocks["timeline"] or card.blocks["chords"]))) if not present]
    from stage_status import card_notes
    precise = card_notes(text, analysis)
    status = " · ".join(dict.fromkeys(precise + warnings)) if precise or warnings else "stage status not recorded (older card)"
    if missing:
        status += "; " + ", ".join(missing) + ": " + MISSING
    out += ["DIDN'T HEAR: " + status, f"full card: {clean(full_card) if full_card is not None else 'path not supplied'} ({len(text.splitlines())} lines)", INVITATION]
    return "\n".join(out) + "\n"


def _setup_note(line):
    """Setup belongs on the full card; heard evidence stays on this view."""
    text = clean(line).removeprefix("NOTE:").strip()
    # A terminal cache tag describes reuse, not the content of a lyric note.
    text = re.sub(r"\s*\[shared cache\]$", "", text)
    probe = text.split("'", 1)[0] if text.startswith("lyrics:") else text
    # A stage that failed is heard evidence for this view, even when the cause
    # is a missing module (review of PR #6, 10 Nov 2026). Skips stay setup.
    if re.search(r"^\s*[\w /-]+:\s*failed\b", probe, re.I):
        return False
    return bool(re.search(
        r"\b(?:install|installation|setup|download|device|cuda|cpu|float32|float16|cache|cached)\b"
        r"|missing optional|optional .*(?:missing|unavailable|not installed)"
        r"|not importable|not installed|No module named|ModuleNotFoundError|ImportError"
        r"|(?:weights|checkpoint|dependency|package).*(?:missing|unavailable)"
        r"|missing.*(?:weights|checkpoint|dependency|package)"
        r"|(?:LRCLIB|credits|lookup|MusicBrainz|Discogs):? .*(?:skipped|disabled|not run)"
        r"|(?:skipped|disabled).*(?:lookup|LRCLIB|credits|MusicBrainz|Discogs)", probe, re.I))


def _feel_card_moved(line):
    """Expected, harmless notes that live on the full card only (feel card, 10 Nov 2026).

    A whitelist, not a pattern for "skipped": an unknown skip, anything failed,
    degraded or not run, unsure lyric lines and Whisper-guess words all stay.
    """
    text = clean(line).removeprefix("NOTE:").strip()
    text = re.sub(r"\s*\[shared cache\]$", "", text).rstrip()
    import key_changes
    if text in (key_changes.NOTE, key_changes.LATE_NOTE) or re.fullmatch(r"lyrics: names list fed to Whisper \(.*\)\.?", text, re.I):
        return True
    probe = text.split("'", 1)[0] if text.startswith("lyrics:") else text
    # FAILURE also matches "skipped", which is exactly what the whitelist below allows.
    if re.search(r"\b(?:failed|unavailable|disabled|traceback|cannot run|could not load)\b|\b\w*Error:", probe, re.I) \
            or re.search(r"\bnot run\b|\bnot importable\b|\bdegraded\b|fell back|fallback|unsure|guess", text, re.I):
        return False
    return bool(re.fullmatch(
        r"[\w /-]+: skipped: (?:word times supplied|no speaker given|the event model does not hear breathing here)\.?"
        r"|lyrics file: supplied\.?"
        r"|lyrics: re-listened to \d+ long gap\(s\) in short windows and recovered \d+ words the one-pass transcript missed\.?",
        text, re.I))


def _two_examples(line):
    """Feel card keeps the (unsure) count and at most two examples; the full card has them all."""
    text = clean(line)
    if not re.match(r"(?:NOTE:\s*)?lyrics: (?:\d+ line\(s\) kept but marked \(unsure\)|Whisper was doubtful on \d+ line\(s\))", text):
        return line
    separator = "): " if "lyrics: Whisper was doubtful" in text else " Whisper: "
    head, sep, tail = text.partition(separator)
    examples = re.split(r"; (?=\d+:\d\d )", tail)
    return line if not sep or len(examples) <= 2 else head + sep + "; ".join(examples[:2])


def _fold_first_listen_notes(notes):
    """Fold exact normalized text only; never reconcile stage outcomes across passes."""
    folded, omitted = {}, {}
    for name, details in notes.items():
        for line in details:
            text = clean(line).removeprefix("NOTE:").strip()
            text = re.sub(r"\s*\[shared cache\]$", "", text).rstrip()
            count = re.fullmatch(r"(\d+) more status entries omitted by source; supply original JSON for details", text)
            if count:
                omitted[name] = omitted.get(name, 0) + int(count[1])
                folded.setdefault(None, [])  # keep the first omission's position
            else:
                passes = folded.setdefault(text, [])
                if name not in passes:
                    passes.append(name)

    def rank(text):
        # Quoted transcript examples are words, not backend status reports.
        probe = text.split("'", 1)[0] if text.startswith("lyrics:") else text
        if FAILURE.search(probe) or re.search(r"\bnot run\b|\bnot importable\b", probe, re.I):
            return 0
        if re.search(r"\bdegraded\b|fell back|fallback|retried|approximat|fusing without|using the chords-only", probe, re.I):
            return 1
        return 2

    entries = []
    for text, passes in folded.items():
        if text is None:
            counts = "; ".join(f"{name}: {count}" for name, count in omitted.items())
            entries.append((2, f"NOTES: {sum(omitted.values())} more status entries omitted by source ({counts}); "
                            "supply original JSON for details"))
        else:
            entries.append((rank(text), f"NOTES [{'·'.join(name + ' pass' for name in passes)}]: {text}"))
    return [line for _, line in sorted(entries, key=lambda entry: entry[0])]


def _feel_stops(card, data):
    """Summarise saved classifications only; uncertain events never supply times."""
    kinds = ("band cuts, voice holds", "full silence", "band break", "faded stop", "band cuts, vocal stem enters")
    rows = [s.replace("vocal stem holds", "voice holds") for s in card.blocks["stops"]]
    counts, examples = {}, []
    if rows:
        if any(FAILURE.search(row) for row in rows):
            return ""
        if rows[0].startswith(("STOPS: none found", "STOPS: 0 confident")):
            counts = dict.fromkeys(kinds, 0)
        else:
            for kind in kinds:
                match = re.search(re.escape(kind) + r": (\d+)", " ".join(rows))
                if match:
                    counts[kind] = int(match[1])
                elif kind in ("faded stop", "band cuts, vocal stem enters"):
                    counts[kind] = 0
        for row in rows:
            match = re.fullmatch(rf"({TIME}) \(([\d.]+) s[^)]*\) "
                                 r"(?:band cuts .*?, (?:voice holds|full silence|band break|vocal stem enters)|"
                                 r"total stem energy falls .*?, faded stop)", row)
            if match:
                examples.append(f"{match[1]} ({match[2]} s)")
    else:
        result = data.get("stops")
        if not isinstance(result, dict) or result.get("status") != "ran" or not isinstance(result.get("events"), list):
            return ""
        events = result["events"]
        # Malformed evidence is absence, not an invented zero.
        if any(not isinstance(e, dict) or e.get("kind") not in (*kinds, "uncertain") for e in events):
            return ""
        counts = {kind: sum(e["kind"] == kind for e in events) for kind in kinds}
        for event in events:
            t, length = finite(event.get("start_s")), finite(event.get("length_s"))
            if event["kind"] in kinds and t is not None and t >= 0 and length is not None and length > 0:
                ticks = round(t * 100)
                examples.append(f"{ticks // 6000}:{(ticks % 6000) / 100:05.2f} ({length:g} s)")
    if len(counts) != len(kinds):
        return ""
    total = sum(counts.values())
    line = (f"STOPS: {total} · " + " · ".join(f"{kind}: {counts[kind]}" for kind in kinds if counts[kind])
            if total else "STOPS: none confident")
    if examples:
        line += " · first " + ", ".join(examples[:2])
    patterns = [row for row in rows if row.startswith(("stop-time:", "repeated cuts:", "pulse not checked;", "includes "))]
    if not rows:
        for r in result.get("recurrence", []):
            if "count" in r:
                patterns += [f"repeated cuts: {r['count']} cuts every ~{r['interval_s']:g} s, "
                             f"{clock(r['start_s'])}–{clock(r['end_s'])}",
                             "pulse not checked; cut spacing alone does not establish stop-time"]
                if r.get("short_gaps"):
                    patterns.append(f"includes {r['short_gaps']} short gaps, not called as individual band breaks")
    notices = [row for row in rows if row.startswith(("re-listen to update STOPS", "opening 0:00", "vocal entry measured"))]
    if patterns or notices:
        line += "\n" + "\n".join("  " + row for row in patterns + notices)
    return line


def _merge_word_gaps(rows, max_gap=0.0):
    """Join same-description gap claims only across continuous printed spans.

    Real timeline rows carry a leading clock column ('2:32  2:32–2:34: …') and a sub-second span
    prints as one time ('2:48: …'). Other gap claims ('no voice …') and quoted lyric rows are
    barriers; plain timeline events ('acoustic guitar alone for 0:11') sit between claims
    without breaking a run (Muninn's Vampires card, 11 Nov 2026)."""
    desc = (r"voice present, words (?:not heard(?: \(Whisper found none\))?|unavailable \(transcription did not run or failed\))"
            # 16 Nov 2026: the supplied-lyrics wordings merge the same way (LRPD printed six WORDS rows)
            r'|wordless singing \(lyrics: "[^"]*"\)'
            r'|voice present; lyrics have words here Whisper didn\'t hear \("[^"]*"\)')
    pattern = rf"\s*(?:({TIME})\s+)?({TIME})(?:–({TIME}))?: ({desc})"
    other_claim = rf"\s*(?:{TIME}\s+)?{TIME}(?:–{TIME})?: "
    out, previous = [], None          # previous = (index in out, lead, start, end, description)
    for row in rows:
        m = re.fullmatch(pattern, row)
        if not m:
            if "'" in row or re.match(other_claim, row):
                previous = None       # a lyric line or a different claim ends the run
            out.append(row)
            continue
        lead, start, end, description = m[1], m[2], m[3] or m[2], m[4]
        merged = bool(previous and description == previous[4]
                      and 0 <= seconds(start) - seconds(previous[3]) <= max_gap)
        if merged:
            idx, lead, start = previous[0], previous[1], previous[2]
            if seconds(previous[3]) > seconds(end):
                end = previous[3]
        if merged or start == end:     # rebuild only what changes; untouched rows stay byte-identical
            span = start if start == end else f"{start}–{end}"
            row = f"{lead}  {span}: {description}" if lead else f"{span}: {description}"
        if merged:
            out[idx] = row
        else:
            idx = len(out)
            out.append(row)
        previous = (idx, lead, start, end, description)
    return out


def render_first_listen(cards, analyses=None, pass_statuses=None):
    """Whole-song view of saved passes, at most 25 physical lines before lyrics.

    `cards` and optional `analyses` map music/singer/harmony to saved text/JSON.
    `pass_statuses`, when supplied, is authoritative; adverse passes supply no claims.
    Text wins. Saved JSON strings, lyric rows and stop events supplement it;
    no engine is imported. Long selections are omitted whole, never stripped of
    caveats. Evidence/status notes follow the complete lyrics; setup stays on
    the full card.
    """
    from stage_status import card_notes, display_items, source_notes, NOTES_WIDTH, instrumental_text, names_note_text
    analyses = analyses or {}
    pass_statuses = pass_statuses or {}
    import meter_trust
    cards = {name: names_note_text(text) if text else text for name, text in cards.items()}
    if pass_statuses.get("harmony", {}).get("status", "ran") == "ran":
        import key_opening
        cards["harmony"] = key_opening.intro_text(cards.get("harmony") or "", analyses.get("harmony"))
    singer_data = analyses.get("singer")
    meter = singer_data.get("meter") if isinstance(singer_data, dict) else None
    if isinstance(meter, dict) and pass_statuses.get("singer", {}).get("status", "ran") == "ran":
        cards["music"] = meter_trust.tempo_text(cards.get("music") or "", meter)
        cards["singer"] = meter_trust.meter_text(cards.get("singer") or "", meter)
    parsed, raw, data, status, status_details = {}, {}, {}, [], []
    import lyrics_shared
    evidence = "\n".join(t or "" for n, t in cards.items()
                         if pass_statuses.get(n, {}).get("status", "ran") == "ran")
    import instrumental_voice
    import keyfuse
    music_key, harmony_key, spans = keyfuse.card_spelling_keys(analyses, pass_statuses)
    vocal_key = keyfuse.spelling_context(music_key or harmony_key, spans)
    chord_key = keyfuse.spelling_context(harmony_key or music_key, spans)
    instrumental = any(instrumental_voice.designated(d) for n, d in analyses.items()
                       if pass_statuses.get(n, {}).get("status", "ran") == "ran")
    if not analyses:
        instrumental = instrumental_voice.text_designation(evidence)
    for name in ("music", "singer", "harmony"):
        raw[name] = cards.get(name) or ""
        data[name] = analyses.get(name)
        if data[name] is None:
            data[name] = {}
        if not isinstance(data[name], dict):
            data[name] = {"notes": ["supplemental analysis failed (expected a JSON object)"]}
        import lyric_align
        raw[name], data[name] = lyric_align.clean_writer_card(raw[name], data[name])
        raw[name] = instrumental_text(lyrics_shared.clean_gap_claims(raw[name], evidence, data[name]), evidence)
        if name == "singer":
            raw[name] = instrumental_voice.saved_peak_pitch(raw[name], data[name])
            raw[name] = keyfuse.singer_note_labels(raw[name], vocal_key)
        if name == "harmony":
            raw[name] = keyfuse.harmony_note_labels(raw[name], chord_key)
        raw[name] = instrumental_voice.stem_text(raw[name], instrumental)
        import jev_reads
        raw[name] = jev_reads.refresh_contrast(raw[name], data[name])
        parsed[name] = parse(raw[name])
        c, d = parsed[name], data[name]
        stops = d.get("stops")
        if name == "singer" and not c.blocks["stops"] and isinstance(stops, dict) and stops.get("status") == "failed":
            c.warnings.append("STOPS: failed: " + clean(stops.get("reason", "reason not recorded")))
        # singer --json already stores the exact SINGER block strings.
        singer = d.get("singer")
        if not c.blocks["singer"] and isinstance(singer, dict) and isinstance(singer.get("lines"), list):
            c.blocks["singer"] = lyrics_shared.clean_gap_claims(
                "\n".join(x for x in singer["lines"] if isinstance(x, str)), evidence, d).splitlines()
            voice = instrumental_voice.saved_peak_pitch("\n".join(c.blocks["singer"]), d)
            c.blocks["singer"] = keyfuse.singer_note_labels(voice, vocal_key).splitlines()
            if instrumental:
                rows = "SINGER:\n" + "\n".join("  " + s for s in c.blocks["singer"])
                c.blocks["singer"] = [s.strip() for s in instrumental_voice.stem_text(rows, True).splitlines()]
        for key, field in (("KEY", "key_claim"), ("KEY", "key_verdict")):
            claim = d.get(field)
            if isinstance(claim, dict) and isinstance(claim.get("line"), str) and claim["line"]:
                c.facts.setdefault(key, claim["line"])
        if c.title == MISSING and isinstance(d.get("file"), str) and d["file"]:
            c.title = Path(d["file"]).stem
        if c.length == MISSING and finite(d.get("duration")) is not None:
            c.length = clock(d["duration"])
        present = bool(raw[name].strip() or d)
        record = pass_statuses.get(name)
        if record and record["status"] != "ran":
            line = f"{name}: {record['status']}: {record['reason']}"
            if len(textwrap.wrap(line, width=NOTES_WIDTH)) > 3:
                status.append(f"{name}: {record['status']}: full reason after lyrics")
                status_details.append(line)
            else:
                status.append(line)
        elif not present and not record:
            status.append(f"{name}: not supplied (not run or no saved card; unknown)")
    m, s, h = (parsed[n] for n in ("music", "singer", "harmony"))
    title = next((parsed[n] for n in raw if parsed[n].title != MISSING
                  and pass_statuses.get(n, {}).get("status", "ran") == "ran"), parse(""))
    sources = [n + (" card" if raw[n].strip() else " JSON") for n in raw if raw[n].strip() or data[n]]
    out = [f"FIRST LISTEN: {title.title} · {title.length}",
           "PASSES: " + (" · ".join(sources) if sources else "none supplied")] + status

    def wrapped(text):
        return textwrap.wrap(text, width=NOTES_WIDTH, subsequent_indent="  ", break_on_hyphens=False)

    def eligible(name, stages=()):
        record = pass_statuses.get(name)
        if record and record["status"] != "ran":
            return False
        # Test explicit stage outcomes as well as legacy NOTES. A failed stage
        # cannot lend a plausible-looking partial line to the headline.
        details = card_notes(raw[name], data[name]) + source_notes(raw[name]) + parsed[name].warnings
        for row in data[name].get("stage_status") if isinstance(data[name].get("stage_status"), list) else []:
            if (isinstance(row, dict) and row.get("status") in ("failed", "skipped", "not run")
                    and isinstance(row.get("stage"), str) and row["stage"].lower() in stages):
                return False
        # Read the selected stage's outcome, not a nested input's status:
        # "words: Whisper's guesses (LRCLIB: skipped: ...)" still has words.
        return not any(re.match(r"(?:NOTE:\s*)?" + re.escape(stage) +
                                r":\s*(?:failed|skipped|unavailable|disabled|not run)\b", row, re.I)
                       for stage in stages for row in details)

    # Every numeric harmonic selection travels with its source definition and
    # reference, or is omitted whole. Music's key cannot stand in for harmony's.
    picks, definitions = [], []
    for label, stages in (("KEY", ("key", "spectral key")), ("TEMPO", ("tempo",)),
                          ("SHAPE", ("sections",)), ("SOUND", ("sound",))):
        if label == "TEMPO":
            import tempo_trust
            sdata = tempo_trust.saved_meter({"json": data["singer"], "text": raw["singer"]}, pass_statuses.get("singer"))
        if label == "TEMPO" and eligible("music") and eligible("singer", ("meter", "beat tracker")):
            # Saved cards carry librosa's number; steady beat_this bars lead,
            # including when the librosa tempo stage failed or left no line.
            tdata = data["music"].get("tempo")
            lib = tdata.get("bpm") if isinstance(tdata, dict) else None
            reading = tempo_trust.lead_reading(lib, sdata)
            if reading:
                picks.append("TEMPO: " + _feel_tempo(tempo_trust.lead_line(reading)))
                continue
        if label in m.facts and eligible("music", stages):
            body = _feel_tempo(tempo_trust.estimate_line(m.facts[label], sdata)) if label == "TEMPO" else m.facts[label]
            if label == "KEY":
                from stage_status import cnn_key_text
                body = cnn_key_text("KEY: " + body + "\n" + raw["music"], data["music"]).splitlines()[0][5:]
            if label == "SOUND":
                body = " · ".join(body.split(" · ")[:2])
            if label == "TEMPO" and "see METER" in body:
                # 10 Nov 2026: this used to drop the whole TEMPO line, because METER lives on the singer card,
                # not here (Give A Little Bit's feel card had no tempo at all). Keep the tempo; keep the meter
                # words only while the singer's meter stage stands, and drop the pointer to a block not on this card.
                if "METER" in m.facts and eligible("music", ("meter",)):
                    body += "; METER: " + m.facts["METER"]
                elif eligible("singer", ("meter", "beat tracker")):
                    body = re.sub(r",\s*see METER\b", "", body)
                else:
                    body = body.split(" · meter ")[0]
            if label == 'KEY' and eligible('harmony', ('key changes',)):
                import key_changes
                body = key_changes.tonic_ambiguity(
                    body, (data['harmony'].get('key_change_scan') or {}).get('section_tonics'))
            if not FAILURE.search(body):
                picks.append(label + ": " + body)
                if label == 'KEY' and eligible('harmony', ('key changes',)):
                    import key_changes
                    picks += key_changes.lines(data['harmony'].get('key_change_scan'))
    if eligible("singer", ("meter", "beat tracker")):
        # Quote the same policy used to correct the full-card heading.
        claim = meter_trust.assess(meter)[0]
        # Only when it adds something: TEMPO often already carries the meter (Huginn's #27 review).
        sig = re.match(r"\s*(\d+/\d+)", claim or "")
        tempo_pick = next((p for p in picks if p.startswith("TEMPO:")), "")
        if claim and claim not in tempo_pick and not (sig and sig[1] in tempo_pick):
            picks.append("METER: " + claim)
    if eligible("singer", ("singer", "stems", "pitch")):
        if instrumental:
            picks.append(instrumental_voice.MELODY)
        for prefix in ("pitch:", "pitch arc", "most intense moment:"):
            row = next((r for r in s.blocks["singer"] if r.startswith(prefix)), "")
            if row and not FAILURE.search(row):
                picks.append(("VOCAL-STEM MELODY " if instrumental else "SINGER ") + row)
        if instrumental:
            picks.append(instrumental_voice.AROUSAL)
        for row in s.blocks["singer"]:
            if row.startswith(("voice present, words not heard", "wordless singing (lyrics:",
                               "voice present; lyrics have words here")):
                picks.append("SINGER " + row)
    if eligible("harmony", ("words",)):
        picks += ["WORDS " + row for row in _merge_word_gaps(h.blocks["timeline"])
                  if any(desc in row for desc in ("voice present, words not heard", "wordless singing (lyrics:",
                                                  "voice present; lyrics have words here"))]
    if eligible("singer", ("stops", "stems")):
        line = _feel_stops(s, data["singer"])
        if line:
            picks.extend(instrumental_voice.stem_text(line, instrumental).splitlines())
    home = next((r.removeprefix("home:").strip().split("; ~", 1)[0]
                 for r in h.blocks["chords"] if r.startswith("home:")), "")
    key_pattern = r"[A-G][#b♭♯]? (?:major|minor|mixolydian|blues-rock)"
    working = re.search(r"read against (" + key_pattern + r")", home)
    keys = re.findall(key_pattern, home)
    modal = keys and keys[0].endswith(" mixolydian")
    key = working[1] if working else keys[0] if modal or len(set(keys)) == 1 else ""
    unclear = re.search(r"unclear|between|working|uncertain|ambiguous|\blean\b", home, re.I)
    reference = "in " + key + (", key unclear" if unclear else "") if key else ""
    claim = data["harmony"].get("key_verdict")
    uncertain_blues = isinstance(claim, dict) and claim.get("blues_rock") and not claim.get("key")
    changes = any(r.startswith("key area changes") for r in h.blocks["chords"])
    ending = next((r for r in h.blocks["chords"] if r.startswith("ends on ")), "")
    if ending and (reference or uncertain_blues) and not changes and eligible("harmony", ("chords", "key fusion", "key by chords")):
        picks.append("HARMONY (" + ("chords" if uncertain_blues else reference) + "): " + ending)
    turns = [r for r in h.blocks["tension"] if r.startswith(("BUILD ", "RELEASE "))]
    tension_definition = next((r for r in h.blocks["tension"] if r.startswith("curve (")), "")
    if turns and tension_definition and reference and not (changes or uncertain_blues) and eligible("harmony", ("tension", "builds / releases", "chords", "key fusion")):
        picks.append("HARMONY (" + reference + ") " + max(turns, key=delta))
        definitions.append("tension 0–1: harmony-from-home/dissonance/loudness/activity")
    contrasts = timed(h.blocks["contrast"])
    contrast_definition = next((r for r in h.blocks["contrast"] if r.startswith("words = ")), "")
    jev_definition = next((r for r in h.blocks["contrast"] if r.startswith("Jev = ")), "")
    jev_contrasts = [r for r in contrasts if re.search(r"Jev [+-][\d.]+", r)]
    old_eligible = eligible("harmony", ("sentiment", "musical lift", "word / music contrast"))
    available = contrasts if old_eligible else jev_contrasts
    if available and (contrast_definition or jev_definition) and eligible("harmony"):
        row = lyric_qualifiers(h, max(available, key=contrast_score))
        stamp = re.match(TIME, row)
        if stamp and not row[stamp.end():].startswith("≈"):
            t = seconds(stamp[0])
            placed = [r.t for card in (m, s, h) for r in card.lyrics if r.approx]
            for d in data.values():
                aligned = d.get("lyrics_aligned") if isinstance(d.get("lyrics_aligned"), dict) else {}
                placed += [float(r["t"]) for r in aligned.get("lines") or []
                           if isinstance(r, dict) and r.get("approx") and finite(r.get("t")) is not None]
            if any(abs(x - t) < 0.6 for x in placed):
                row = row[:stamp.end()] + "≈" + row[stamp.end():]
        picks.append("CONTRAST: " + fit_contrast(row, NOTES_WIDTH - len("CONTRAST: ")))
        definitions.append("contrast words=model sentiment, lift=song-relative")
        if jev_definition:
            picks.append(jev_definition)
    for label, stages in (("VOCALS", ("vocals",)), ("VOCAL", ("vocal quality",))):
        if label in m.facts and eligible("music", stages) and not FAILURE.search(m.facts[label]):
            picks.append(label + ": " + m.facts[label])

    # MISHEARD: one line, only against the writer's own words, from saved rows (JSON), never from a re-run.
    for name in raw:
        d = data[name]
        src = d.get("lyrics_source") if isinstance(d.get("lyrics_source"), dict) else None
        rows = d.get("lyrics")
        aligned = d.get("lyrics_aligned") if isinstance(d.get("lyrics_aligned"), dict) else {}
        if isinstance(rows, dict):                      # singer JSON keeps its rows and lookup together
            src, rows = rows.get("lookup"), rows.get("rows")
        elif aligned:
            rows = aligned.get("lines")
        rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
        if rows and isinstance(src, dict) and src.get("status") == "file" and eligible(name, ("words",)):
            import lyric_align
            if not any("misheard" in r for r in rows):
                picks.append("MISHEARD: not in this saved card (older listen); --fresh to compute it")
            else:
                line = lyric_align.feel_line([r for r in rows if isinstance(r.get("t"), (int, float))])
                if line:
                    picks.append(line)
            break

    if len(wrapped(out[0])) > 2:
        out[0] = "FIRST LISTEN: title exceeds view budget; see supplied cards."
    out = [part for line in out for part in wrapped(line)]
    omitted = 0
    for pick in picks:
        lines = wrapped(pick)
        # Reserve a line for the explicit omission notice, even if none is needed.
        if len(out) + len(lines) <= 24:
            out += lines
        else:
            omitted += 1
    # Optional Noul summaries use only spare headline space, after every existing pick.
    at = next((i + 1 for i, line in enumerate(out) if line.startswith("Jev = ")),
              next((i + 1 for i, line in enumerate(out) if line.startswith("CONTRAST:")), len(out)))
    for summary in filter(jev_reads.is_noul_summary, h.blocks["contrast"]):
        lines = wrapped(summary)
        if eligible("harmony") and len(out) + len(lines) <= 24:
            out[at:at] = lines
            at += len(lines)
    if omitted:
        out.append(f"VIEW: {omitted} selections exceed the 25-line budget; see supplied full cards.")

    # Keep one pass's words and provenance together. Never merge alignments from
    # different passes: the quick card is preferred, including its placed times.
    lyric_source = next((n for n in raw if parsed[n].lyrics and eligible(n, ("words",))), None)
    label = ""
    lyrics = []
    if lyric_source:
        label = next((line.strip() for line in raw[lyric_source].splitlines()
                      if line.strip().startswith("LYRICS:")), "LYRICS: provenance not recorded")
        lyrics = [f"[{clock(r.t)}{'≈' if r.approx else ''}] {r.text}"
                  + (NO_VOICE_MARK if r.no_voice else "") for r in parsed[lyric_source].lyrics]
        # Plain lyric blocks keep their original order, duplicate rows and time
        # precision. The generic slim parser deliberately deduplicates them.
        plain, in_lyrics, timeline = [], False, False
        for line in raw[lyric_source].splitlines():
            body = line.strip()
            if body == "LYRIC TIMELINE":
                timeline = True
            if body.startswith("LYRICS:"):
                in_lyrics = not timeline
            elif in_lyrics and re.match(rf"\[{TIME}(?:≈)?\]", body):
                plain.append(body)
            elif body and (not line[0].isspace() or body.startswith("NOTE:")):
                in_lyrics = False
        if plain:
            lyrics = plain
    else:
        for name, d in data.items():
            if not eligible(name, ("words",)):
                continue
            aligned = d.get("lyrics_aligned") if isinstance(d.get("lyrics_aligned"), dict) else {}
            source = d.get("lyrics")
            if isinstance(source, dict):  # singer JSON's LRCLIB result
                rows = source.get("rows")
                candidate_label = source.get("label")
            else:
                rows = aligned.get("lines") or source
                candidate_label = aligned.get("label") or d.get("lyrics_align")
            # Rows without a finite time or with non-text words are skipped, never fatal.
            rows = [r for r in rows if isinstance(r, dict) and finite(r.get("t")) is not None
                    and isinstance(r.get("text"), str)] if isinstance(rows, list) else []
            if rows:
                lyric_source = name
                label = candidate_label if isinstance(candidate_label, str) and candidate_label else \
                    "LYRICS: provenance not recorded in JSON"
                lyrics = [f"[{clock(r['t'])}{'≈' if r.get('approx') else ''}] {r['text']}" for r in rows]
                break
    if instrumental:
        label = next((line.strip().removeprefix("NOTE: ") for line in evidence.splitlines()
                      if re.match(r"\s*(?:NOTE:\s*)?LYRICS: instrumental \(LRCLIB\)", line)),
                     "LYRICS: instrumental (LRCLIB); database designation, not a measurement of silence")
    elif lyric_source:
        origin = lyrics_shared.displayed_lyric_source(data[lyric_source], label, has_words=bool(lyrics))
        if origin != lyrics_shared.displayed_lyric_source(text=label):
            label = {
                "file": "LYRICS: writer's text from the supplied file",
                "synced": "LYRICS: LRCLIB synced lyrics",
                "plain": "LYRICS: LRCLIB plain lyrics",
                "whisper": "LYRICS: words are Whisper's guesses (rough times)",
            }.get(origin, label)
    out += wrapped(instrumental_text(label or "LYRICS: no saved lyric rows supplied", evidence) +
                   (f" [{lyric_source}]" if lyric_source else ""))
    if instrumental:
        out += wrapped(instrumental_voice.WORDS)
    for line in lyrics:
        out += wrapped(line)
    for line in status_details:
        out += wrapped(line)
    # Detail has no line-count cap. Do not turn missing/failed stages into silence
    # or silently discard long notes and legacy '+N more' markers.
    notes, moved = {}, set()
    for name in raw:
        if not eligible(name):
            continue
        if not raw[name].strip() and not data[name]:
            continue
        details = card_notes(raw[name], data[name])
        details += source_notes(raw[name])
        details += [str(n) for n in (data[name].get("notes") if isinstance(data[name].get("notes"), list) else [])]
        details += parsed[name].warnings
        details = [names_note_text(line) for line in details]
        details = display_items(raw[name] + "\n" + evidence, list(dict.fromkeys(details)), data[name])
        if not details:
            details = ["stage status not recorded (older card)"]
        details = [line for line in details if not _setup_note(line)]
        # Expected, harmless notes move to the full card; count each distinct text once.
        moved.update(re.sub(r"\s*\[shared cache\]$", "", clean(line).removeprefix("NOTE:").strip()).rstrip()
                     for line in details if _feel_card_moved(line))
        details = [_two_examples(line) for line in details if not _feel_card_moved(line)]
        notes[name] = details
    if len(notes) == 1:
        # Keep the round-1 one-pass view, including its note spelling and order.
        for name, details in notes.items():
            for line in details:
                out += wrapped(f"{name} NOTES: {line}")
    else:
        for line in _fold_first_listen_notes(notes):
            out += wrapped(line)
    if moved:
        out += wrapped(f"NOTES: {len(moved)} more on the full card (--view full)")
    if definitions:
        out += wrapped("NOTES: " + "; ".join(definitions))
    return "\n".join(out + [INVITATION]) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", nargs="?", help="saved full card, or - for stdin; omit for analysis alone")
    ap.add_argument("--full-card", help="path to the unchanged full card (required for a stdin pointer)")
    ap.add_argument("--analysis", help="optional saved analysis JSON; missing text blocks only")
    ap.add_argument("--view", choices=("slim", "first-listen"), default="slim")
    for name in ("singer", "harmony"):
        ap.add_argument("--" + name + "-card", help="saved pass text for --view first-listen")
        ap.add_argument("--" + name + "-analysis", help="saved pass JSON for --view first-listen")
    args = ap.parse_args(argv)
    if not args.card and not args.analysis and not (args.view == "first-listen" and any(
            getattr(args, name + suffix) for name in ("singer", "harmony") for suffix in ("_card", "_analysis"))):
        ap.error("supply a full card or --analysis")
    try:
        text = sys.stdin.read() if args.card == "-" else Path(args.card).read_bytes().decode("utf-8") if args.card else ""
        data = None
        if args.analysis:
            try:
                data = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                data = {"notes": [f"supplemental analysis failed ({type(e).__name__}: {clean(e)})"]}
        if args.view == "first-listen":
            cards, analyses = {"music": text}, {"music": data}
            for name in ("singer", "harmony"):
                path = getattr(args, name + "_card")
                try:
                    cards[name] = Path(path).read_text(encoding="utf-8") if path else ""
                except OSError as e:
                    cards[name] = f"NOTE: {name} card unavailable ({type(e).__name__})"
                path = getattr(args, name + "_analysis")
                if path:
                    try:
                        analyses[name] = json.loads(Path(path).read_text(encoding="utf-8"))
                    except (OSError, ValueError) as e:
                        analyses[name] = {"notes": [f"supplemental analysis failed ({type(e).__name__}: {clean(e)})"]}
            sys.stdout.write(render_first_listen(cards, analyses))
        else:
            if any(getattr(args, name + suffix) for name in ("singer", "harmony") for suffix in ("_card", "_analysis")):
                ap.error("--singer/--harmony inputs require --view first-listen")
            sys.stdout.write(render(text, args.full_card or (args.card if args.card != "-" else None), data))
        return 0
    except (OSError, ValueError) as e:
        print(f"DIDN'T HEAR: slim card unavailable ({type(e).__name__}: {clean(e)})")
        print(INVITATION)
        return 1


if __name__ == "__main__":
    sys.exit(main())
