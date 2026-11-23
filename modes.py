#!/usr/bin/env python3
"""Listening passes over an existing card. No models or network.

    python modes.py card.txt --mode first-listen --section 1
    python modes.py card.txt --mode lens --lens groove
    python modes.py card.txt --mode vibe

Optional --analysis accepts existing music.py/harmony.py --json output. Plain
--lyrics supplies stanza boundaries for form detection only, never heard words.
See LISTENING_MODES.md for optional resumable sessions and the wrapper.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import re
import sys
import unicodedata

INVITATION = "and now, you"
LENSES = ("words", "groove", "harmony", "instruments")
TIME = r"\d+:\d{2}(?:\.\d+)?"
STAMP = re.compile(TIME)
ROW = re.compile(rf"^\s*\[?({TIME})(≈)?(?:[–−-]({TIME}))?\]?\s+(.*)$")
FAILURE = re.compile(r"failed|unavailable|unreachable|traceback|error|skipped|could not|cannot run|"
                     r"disabled|not run|unreadable|timed out|no (?:cuda|gpu)", re.I)


def seconds(value):
    m, s = value.split(":")
    return int(m) * 60 + float(s)


def clock(t):
    t = round(max(0, t), 3)
    whole = int(t)
    fraction = f"{t - whole:.3f}"[1:].rstrip("0").rstrip(".")
    return f"{whole // 60}:{whole % 60:02d}{fraction}"


def tokens(text):
    text = re.sub(r"\s*\(unsure[^)]*\)", "", text)
    text = unicodedata.normalize("NFKD", text.casefold().replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[^\W_]+", text.replace("'", ""), re.UNICODE)


@dataclass
class Event:
    t: float
    text: str
    channel: str
    end: float | None = None
    approx: bool = False
    source: str = ""


@dataclass
class Card:
    raw: str
    duration: float = 0
    events: list[Event] = field(default_factory=list)
    lyrics: list[Event] = field(default_factory=list)
    context: dict = field(default_factory=lambda: {k: [] for k in LENSES})
    warnings: list[str] = field(default_factory=list)
    boundaries: list[float] = field(default_factory=list)
    tempo: str = ""
    sound: str = ""
    mood: str = ""


@dataclass
class Section:
    start: float
    end: float
    kind: str
    reason: str


def _channel(text):
    """Unknown/mixed timeline prose is left out of single-channel passes."""
    low = text.lower()
    if "words not heard" in low or low.startswith(("voice:", "voice enters:", "line '", "last chorus:", "the words stop", "words ")):
        return "words"
    if re.search(r"chord|cadence|tonic|modulat|pedal|tension|borrowed|^unexpected |^i:", low):
        return "harmony"
    if re.search(r"guitar|piano|organ|synth|drum|bass|instrument|vocals? (?:enter|leave|swell)", low):
        return "instruments"
    if re.search(r"beat|pulse|meter|groove|swing|shuffle|section|opens |swell|dip|drop-out|level|wordless", low):
        return "groove"
    return ""


def _add_event(card, t, text, channel, end=None, approx=False, source=""):
    if channel and math.isfinite(t) and t >= 0:
        # An embedded interval (or 'until') matters to first-listen's spoiler guard.
        times = [seconds(x) for x in STAMP.findall(text)]
        limit = max([t] + times + ([end] if end is not None else []))
        card.events.append(Event(t, text, channel, limit if limit > t else None, approx, source))


def parse_card(text, analysis=None):
    """Read the current combined shell card, or any individual engine's card.

    Preserve source wording and caveats. Unknown blocks are deliberately ignored;
    they cannot safely be assigned to a lens just because they contain a time.
    """
    card = Card(text)
    block = ""
    engine = "card"
    lyric_rows = []
    continuation = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("==================="):
            engine = s.lstrip("= ").strip()
            block = ""
            if "and now, you" in s:
                break
            continue
        if s.startswith(("MUSIC:", "HARMONY:", "SINGER.PY")):
            times = STAMP.findall(s)
            if times:
                card.duration = max(card.duration, seconds(times[-1]))
            block = ""
            continue
        if FAILURE.search(s) and not ROW.match(line) and block not in ("lyrics", "lyric timeline"):
            # Notes can quote later lyrics. First-listen reports their status without
            # the quoted payload; full source notes remain available in --mode all.
            card.warnings.append(s)
        elif s.startswith(("NOTE:", "CREDITS:")) and FAILURE.search(s):
            card.warnings.append(s)
        if s.startswith("NOTE:"):
            continue
        if s.startswith("KEY:"):
            card.context["harmony"].append(s)
            continue
        if s.startswith("KEY CHG:"):
            stamp = STAMP.search(s)
            if stamp:
                _add_event(card, seconds(stamp[0]), s[stamp.end():].strip(), "harmony")
            continue
        if s.startswith("TEMPO:"):
            card.tempo = s.split(":", 1)[1].strip()
            card.context["groove"].append(s)
            continue
        if s.startswith("SOUND:"):
            card.sound = s.split(":", 1)[1].strip()
            continue
        if s.startswith("SHAPE:"):
            block = "shape"
            continue
        if s.startswith("LYRICS:"):
            if block != "lyric timeline":
                block = "lyrics"
            if s not in card.context["words"]:
                card.context["words"].append(s)
            continue
        if s.startswith("METER:"):
            block = "meter"
            card.context["groove"].append(s)
            continue
        if s == "CHORDS":
            block = "chords"
        elif s == "TIMELINE":
            block = "timeline"
        elif s == "LYRIC TIMELINE":
            block = "lyric timeline"
        elif s.startswith("INSTRUMENTS ("):
            block = "instruments"
            card.context["instruments"].append(s)
        elif s.startswith("CREDITS ("):
            block = "credits"
        elif s.startswith("who is playing"):
            block = "players"
        elif s == "ENTRANCES/EXITS:":
            block = "entrances"
        elif s.startswith("CONTRAST ("):
            block = "contrast"
        elif s.startswith("TENSION"):
            block = "tension"
        elif s and not line[0].isspace():
            block = ""

        row = ROW.match(line)
        if row:
            t, approx, end, body = row.groups()
            t, end = seconds(t), seconds(end) if end else None
            if block in ("lyrics", "lyric timeline") and not body.startswith("♪"):
                # In LYRIC TIMELINE the music after the em dash has its own channel.
                words, sep, detail = body.partition(" — ") if block == "lyric timeline" else (body, "", "")
                words = words.strip('“”"')
                lyric_rows.append(Event(t, words, "words", end, bool(approx), block))
                if sep:
                    # Split only the unambiguous chord field; mixed prose stays out
                    # of lenses, but remains available in the original card.
                    chord = re.match(r"chord [^,]+", detail)
                    if chord:
                        _add_event(card, t, chord[0], "harmony", source=block)
            else:
                channel = {"shape": "groove", "meter": "groove", "chords": "harmony",
                           "players": "instruments", "entrances": "instruments",
                           "tension": "harmony"}.get(block)
                if block == "tension" and not body.startswith("tension"):
                    channel = ""  # global sparkline chunks are not local events
                if block in ("timeline", "lyric timeline"):
                    body = body.removeprefix("♪ ")
                    channel = _channel(body)
                if block == "entrances" and body.startswith("vocals"):
                    channel = ""  # singer texture is outside these four lenses
                if channel:
                    _add_event(card, t, body, channel, end, source=f"{engine}/{block}")
                if block == "shape":
                    card.boundaries.append(t)
            continuation = card.events[-1] if card.events else None
        elif block == "chords" and line.startswith("      ") and continuation:
            continuation.text += "; " + s
        elif block == "chords" and s.startswith("home:") and not card.context["harmony"]:
            card.context["harmony"].append(s)
        elif block == "instruments" and s and s != "overall:" and not s.startswith("INSTRUMENTS ("):
            card.context["instruments"].append(s)
        elif block == "credits" and s:
            card.context["instruments"].append(s)
        elif block == "contrast" and s.startswith("overall:"):
            card.mood = s.removeprefix("overall: ")
        elif block == "meter" and s:
            card.context["groove"].append(s)

    # Prefer the writer's already-aligned text when both engine cards supply it.
    aligned = [e for e in lyric_rows if e.source == "lyric timeline"]
    card.lyrics = sorted(aligned or lyric_rows, key=lambda e: e.t)
    _analysis(card, analysis or {})
    from modes_view import dedupe
    card.events = dedupe(card.events)
    if not card.duration:
        # A final onset is not the song's ending. Only interval ends can supply
        # a missing duration; otherwise first-listen asks for existing analysis.
        ends = [e.end for e in card.events + card.lyrics if e.end is not None]
        card.duration = max(ends, default=0)
    return card


def _analysis(card, data):
    """Use existing JSON only for missing timing/lyrics, never re-decide facts."""
    dur = data.get("duration", data.get("dur", 0))
    if not card.duration and dur:
        card.duration = float(dur)
    rows = (data.get("lyrics_aligned") or {}).get("lines")
    if rows is None and not card.lyrics:
        rows = data.get("lyrics", data.get("lines", []))
    if rows:
        card.lyrics = [Event(float(r["t"]), r["text"], "words", r.get("end"),
                             bool(r.get("approx")), "analysis") for r in rows]
        card.lyrics.sort(key=lambda e: e.t)
    for s in data.get("sections", data.get("secs", [])):
        card.boundaries.append(float(s["start"]))
    if not any(e.source.rsplit("/", 1)[-1] in ("timeline", "lyric timeline") for e in card.events):
        for row in data.get("timeline", []):
            _add_event(card, float(row["t"]), row["text"], _channel(row["text"]), source="analysis")
    card.warnings += [str(n) for n in data.get("notes", []) if FAILURE.search(str(n))]


def _returns(rows):
    """Find returning lyric runs even when Whisper wraps them onto different rows.

    Six-token seeds grow into exact repeated runs. Returns must be 20 seconds
    apart (or 18% of a longer lyric span). A four-word line can be a refrain.
    All tokens retain their source row's time; no word timing is invented.
    """
    stream, owner = [], []
    for i, row in enumerate(rows):
        ts = tokens(row.text)
        stream.extend(ts)
        owner.extend([i] * len(ts))
    seeds = defaultdict(list)
    for i in range(len(stream) - 5):
        seeds[tuple(stream[i:i + 6])].append(i)
    candidates = []
    separation = max(20, (rows[-1].t - rows[0].t) * 0.18) if rows else 20
    starts = {}
    for i, row in enumerate(owner):
        starts.setdefault(row, i)
    for positions in seeds.values():
        # Avoid quadratic work on chant-like transcripts. Such evidence is weak
        # for verse/chorus anyway; do not interpret every repeated syllable.
        if len(positions) > 32:
            continue
        for a, b in zip(positions, positions[1:]):
            if rows[owner[b]].t - rows[owner[a]].t < separation:
                continue
            if a and b and stream[a - 1] == stream[b - 1]:
                continue  # another seed covers the beginning of this run
            n = 6
            while a + n < b and b + n < len(stream) and stream[a + n] == stream[b + n]:
                n += 1
            if len(set(stream[a:a + n])) < 4:
                continue
            candidates.append((n, [(a, a + n - 1), (b, b + n - 1)]))
    lines = defaultdict(list)
    for i, row in enumerate(rows):
        tok = tuple(tokens(row.text))
        if len(tok) >= 4 and len(set(tok)) >= 4:
            lines[tok].append(i)
    for tok, indices in lines.items():
        if len(indices) > 1 and rows[indices[-1]].t - rows[indices[0]].t >= separation:
            candidates.append((len(tok), [(starts[i], starts[i] + len(tok) - 1) for i in indices]))
    if not candidates:
        return []
    # One refrain family, not every returning pronoun or pre-chorus.
    _, best = max(candidates, key=lambda x: x[0] * (1 + math.log(len(x[1]))))
    first, last = best[0]
    anchor = stream[first:last + 1]
    length = len(anchor)
    seed = anchor[:min(6, length)]
    # Gather other matching runs, including third/fourth choruses. Keep the
    # token anchor, which may start halfway through a wrapped transcript row.
    for i in range(len(stream) - len(seed) + 1):
        if stream[i:i + len(seed)] == seed:
            j = i + len(seed)
            while j < len(stream) and j - i < length and stream[j] == anchor[j - i]:
                j += 1
            if not any(abs(rows[owner[i]].t - rows[owner[a]].t) < separation for a, b in best):
                best.append((i, j - 1))
    # Other returning phrases close to the same anchors can complete a chorus
    # after a changed word. Require both copies to line up; a lone chant cannot.
    windows = [(owner[a], owner[b]) for a, b in best]
    for _, pairs in sorted(candidates, reverse=True):
        matching = []
        for a, b in pairs:
            a, b = owner[a], owner[b]
            k = next((k for k, (x, y) in enumerate(windows)
                      if rows[x].t - 8 <= rows[a].t <= rows[y].t + 12), None)
            if k is not None:
                matching.append((k, a, b))
        if len({k for k, a, b in matching}) >= 2:
            for k, a, b in matching:
                windows[k] = (min(a, windows[k][0]), max(b, windows[k][1]))
        elif matching:
            # A changed opening may hide an earlier chorus from the chosen seed.
            # Another phrase shared with a known copy can recover that return.
            for a, b in pairs:
                a, b = owner[a], owner[b]
                if all(abs(rows[a].t - rows[x].t) >= separation for x, y in windows):
                    windows.append((a, b))
    return sorted(set(windows))


def _stanza_bounds(lyrics, rows):
    """Map given stanza starts to existing row times; unmatched text has no time."""
    stanzas = [tokens(s) for s in re.split(r"\n\s*\n", lyrics.strip()) if tokens(s)]
    stream, owner = [], []
    for row in rows:
        ts = tokens(row.text)
        stream.extend(ts)
        owner.extend([row.t] * len(ts))
    bounds, cursor = [], 0
    for stanza in stanzas:
        seed = stanza[:min(6, len(stanza))]
        if len(seed) < 4:
            continue
        for i in range(cursor, len(stream) - len(seed) + 1):
            if stream[i:i + len(seed)] == seed:
                bounds.append(owner[i])
                cursor = i + len(seed)
                break
    return bounds


def label_sections(card, lyrics=None):
    """A modest form lean: returning words suggest chorus, late new words bridge.

    Boundaries are row/stanza times, not newly measured musical downbeats.
    With no refrain, use the card's acoustic spans and leave the form unclear.
    """
    rows, dur = card.lyrics, card.duration
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError("card has no usable duration or timed evidence; supply --analysis JSON")
    returns = _returns(rows)
    if not returns:
        bounds = sorted({0.0, dur, *(t for t in card.boundaries if 0 < t < dur)})
        return [Section(a, b, "VERSE", "form unclear; no returning lyric block established")
                for a, b in zip(bounds, bounds[1:])]
    stanza = _stanza_bounds(lyrics, rows) if lyrics else []
    windows = []
    for a, b in returns:
        start = rows[a].t
        end = rows[b + 1].t if b + 1 < len(rows) else dur
        # Given stanzas may extend the detected refrain to its whole block.
        if stanza:
            start = max([t for t in stanza if t <= start] or [start])
            end = min([t for t in stanza if t > rows[b].t] or [end])
        if windows and start <= windows[-1][0]:
            windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
        else:
            if windows:
                windows[-1] = (windows[-1][0], min(start, windows[-1][1]))
            windows.append((start, end))
    sections, cursor = [], 0.0
    for start, end in windows:
        if start > cursor:
            late = cursor >= dur * 0.55 and bool(sections) and start - cursor >= 8
            text = " ".join(r.text for r in rows if cursor <= r.t < start)
            # A bridge needs actual new lyrical evidence, not a backend gap.
            current = tokens(text)
            earlier = tokens(" ".join(r.text for r in rows if r.t < cursor))
            phrases = {tuple(earlier[i:i + 4]) for i in range(len(earlier) - 3)}
            shared = sum(tuple(current[i:i + 4]) in phrases for i in range(len(current) - 3))
            novel = len(current) >= 6 and shared / max(1, len(current) - 3) < 0.2
            kind = "BRIDGE" if late and novel else "VERSE"
            reason = "likely; new lyric block late in the song" if kind == "BRIDGE" else "likely; between returning lyric blocks"
            sections.append(Section(cursor, start, kind, reason))
        sections.append(Section(start, end, "CHORUS", "likely; returning lyric block (a refrain may also be a verse)"))
        cursor = end
    # Intro is already in the opening verse; the tail belongs to the last chorus.
    sections[-1].end = dur
    return sections


def _event_line(e):
    span = f"–{clock(e.end)}" if e.end is not None else ""
    return f"  [{clock(e.t)}{span}{'≈' if e.approx else ''}] {e.text}"


def render(card, mode="all", lens=None, section=1, analysis=None, lyrics=None, start=None, until=None):
    """Return text. The 'all' view only removes legacy private prompt previews."""
    from stage_status import names_note_text
    card = names_note_text(card)
    if mode == "all":
        return card
    if mode not in ("first-listen", "lens", "vibe"):
        raise ValueError(f"unknown mode: {mode}")
    c = parse_card(card, analysis)
    from modes_view import local_events, run_status
    if (start is not None or until is not None) and mode != "first-listen":
        raise ValueError("--from/--until require --mode first-listen")
    if mode == "vibe":
        tempo = c.tempo or "tempo unavailable in this card"
        # 'Dark/warm' describes spectrum, not mood. Major/minor does not decide it.
        colour = c.sound.split(" · ")[0] if c.sound else "tone unavailable"
        mood = "mood unclear"
        m = re.search(r"words lean (heavy|warm)", c.mood)
        if m:
            mood = f"lyrics lean {m[1]} (source model estimate; musical mood unclear)"
        regularity = re.search(r"steady ([\d.]+)", c.tempo)
        strength = re.search(r"beat strength ([\d.]+)x", c.tempo)
        bop = "bop unclear"
        if regularity and strength and float(regularity[1]) >= 0.75 and float(strength[1]) >= 1.5:
            bop = "bop lean: likely (steady, pronounced pulse); your call"
        out = [f"tempo: {tempo}", f"sound: {colour}; {mood}; {bop}"]
    elif mode == "lens":
        if lens not in LENSES:
            raise ValueError("--mode lens needs --lens words|groove|harmony|instruments")
        out = [f"LENS: {lens}"] + c.context[lens]
        ev = c.lyrics if lens == "words" else [e for e in c.events if e.channel == lens]
        out += [_event_line(e) for e in ev]
        if not ev:
            out.append("No timed evidence for this lens in the card; that does not mean nothing was heard.")
    else:
        parts = label_sections(c, lyrics)
        if section < 1 or section > len(parts):
            raise ValueError(f"--section must be between 1 and {len(parts)}")
        if start is not None:
            section = next((i + 1 for i, p in enumerate(parts) if p.start <= start < p.end), section)
        part = parts[section - 1]
        a = part.start if start is None else start
        b = part.end if until is None else until
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= c.duration):
            raise ValueError("listening range must satisfy 0 <= from < until <= available duration")
        # Seven and Sunny's seven-ears (MIT) inspired this local-only view.
        # Form reasons depend on later returns, so they stay out of this pass.
        out = [f"FIRST LISTEN: section {section} · {clock(a)}–{clock(b)}",
               "  Instrument labels are source estimates; ≈ marks placed words, not heard timing."]
        selected = local_events(c.lyrics + c.events, a, b, c.duration)
        out += [_event_line(e) for e in sorted(selected, key=lambda e: e.t)]
        if not selected:
            out.append("No local detail available for this section; this is not a claim of silence.")
        out.append("Pause here. Continue with --next when ready; notes are optional.")
    return "\n".join(out + run_status(c, analysis) + [INVITATION]) + "\n"


def main(argv=None):
    from listening_sessions import main as session_main
    return session_main(argv)


if __name__ == "__main__":
    sys.exit(main())
