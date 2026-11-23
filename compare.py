#!/usr/bin/env python3
"""Compare existing listening cards, without running ears or deciding feelings.

    python compare.py 'Artist - Song A' 'Artist - Song B'
    python compare.py /path/to/a.txt /path/to/b.txt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys

from relisten import _read, slug, song_name

INVITATION = "and now, you: which one, and why?"
UNAVAILABLE = "unavailable in this card"
TIME = r"\d+:\d{2}(?:\.\d+)?"
ROW = re.compile(rf"^\s*\[?({TIME})(≈)?(?:[–−-]({TIME}))?\]?\s+(.*)$")
FAILURE = re.compile(r"failed|unavailable|unreachable|traceback|\berror\b|skipped|could not|cannot run|timed out", re.I)


def _seconds(stamp):
    minutes, seconds = stamp.split(":")
    return int(minutes) * 60 + float(seconds)


def _unique(items):
    return list(dict.fromkeys(x for x in items if x))


def _join(items):
    return "; ".join(_unique(items)) or UNAVAILABLE


def _claim(items, name):
    values = _unique(items)
    if len(values) > 1:
        return f"unclear: source {name} claims disagree: " + " / ".join(values)
    return values[0] if values else UNAVAILABLE


@dataclass
class Card:
    title: str
    tempo: str = UNAVAILABLE
    key: str = UNAVAILABLE
    meter: str = UNAVAILABLE
    meter_summary: str = UNAVAILABLE
    opening: str = UNAVAILABLE
    shape: str = UNAVAILABLE
    peaks: str = UNAVAILABLE
    voice: str = UNAVAILABLE
    instruments: str = UNAVAILABLE
    contrast: str = UNAVAILABLE
    pulse: float | None = None
    voice_arc: str = ""
    warnings: list[str] = field(default_factory=list)


def parse_card(text, title="untitled card"):
    """Quote selected source fields; do not recalculate key, meter or tempo.

    Unknown fields are omitted. Caveats, ≈ and model attribution survive.
    Acoustic, vocal and tension peaks stay separate; none proves an ache.
    """
    card = Card(title)
    tempos, keys, homes, meters = [], [], [], []
    shapes, spans, openings, peaks, voices, instruments, contrasts = [], [], [], [], [], [], []
    block = ""
    instrument_method = ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("and now, you") or (s.startswith("===") and "and now, you" in s.lower()):
            block = ""
            continue
        if s.startswith("# "):
            card.title = re.sub(r"\s+\(\d{4}-\d{2}-\d{2}.*\)\s*$", "", s[2:]).strip()
            continue
        if s.startswith("===="):
            block = ""
            continue
        if FAILURE.search(s) and (s.startswith(("NOTE:", "WARNING:", "ERROR:", "[")) or
                                  not line[0].isspace() or
                                  re.match(r"(?:KEY|TEMPO|METER|VOCAL|VOCALS):", s)):
            card.warnings.append(s)
        label, colon, value = s.partition(":")
        value = value.strip()
        if colon and label in ("KEY", "TEMPO", "METER", "SHAPE", "VOCAL", "VOCALS"):
            if label == "KEY":
                keys.append(value)
            elif label == "TEMPO":
                tempos.append(value)
            elif label == "METER":
                meters.append(value)
                block = "meter"
            elif label == "SHAPE":
                shapes.append(value)
                block = "shape"
            else:
                voices.append(s)
                block = ""
            continue
        if s.startswith("ACHE"):
            peaks.append("ache source (candidate, not a verdict): " + s)
            block = "ache"
            continue
        if s.startswith("VOCAL-STEM MELODY:"):
            block = "singer"
            voices.append(s)
        elif block == "singer" and s.startswith("AROUSAL:"):
            voices.append(s)
        elif s == "SINGER:":
            block = "singer"
        elif s.startswith("INSTRUMENTS"):
            block = "instruments"
            instrument_method = s.rstrip(":")
        elif s.startswith("who is playing"):
            block = "players"
        elif s.startswith("CREDITS"):
            block = "credits"
        elif s == "credited, not clearly heard:":
            block = "unheard credits"
        elif s in ("credited and heard:", "heard, not credited:"):
            block = "credits"
        elif s == "ENTRANCES/EXITS:":
            block = "entrances"
        elif s.startswith("CONTRAST"):
            block = "contrast"
        elif s.startswith("CHILL CANDIDATES"):
            block = "chill"
        elif s.startswith("TENSION"):
            block = "tension"
        elif s in ("CHORDS", "TIMELINE", "LYRIC TIMELINE"):
            block = s.lower()
        elif not line[0].isspace() or re.match(r"[A-Z][A-Z /]+:", s):
            block = ""

        row = ROW.match(line)
        if block == "shape" and row:
            spans.append(s)
            if "PEAK" in s:
                peaks.append("level peak: " + s)
        elif block == "singer":
            if s.startswith(("pitch:", "pitch arc", "highest held notes:", "voice with no words:",
                             "voice present, words not heard", "wordless singing (lyrics:",
                             "voice present; lyrics have words here")):
                voices.append(s)
            if s.startswith("pitch arc"):
                card.voice_arc = s
            if s.startswith("most intense moment:"):
                prefix = "vocal stem " if any(v.startswith("VOCAL-STEM MELODY:") for v in voices) else "voice "
                peaks.append(prefix + s)
        elif block == "instruments" and s != "overall:" and not s.startswith("INSTRUMENTS"):
            instruments.append(s)
        elif block == "unheard credits" and s != "credited, not clearly heard:":
            instruments.append(s)
        elif block == "players" and row and _seconds(row[1]) == 0:
            openings.append("tagger estimate at open: " + s)
        elif block == "entrances" and row and "instruments at the open" in s:
            openings.append(s)
        elif block == "timeline" and row and row[4].startswith("opens "):
            openings.insert(0, s)
        elif block == "tension" and s.startswith("range ") and "peak at" in s:
            peaks.append("tension proxy: " + s)
        elif block == "ache" and not s.startswith("ACHE"):
            peaks.append("ache candidate (not a verdict): " + s)
        elif block == "contrast" and s.startswith("overall:"):
            contrasts.append(s.removeprefix("overall:").strip())
        elif block == "chords" and s.startswith("home:"):
            homes.append(s.removeprefix("home:").split(";", 1)[0].strip())
        elif block == "meter" and line[0].isspace() and not s.startswith("METER:"):
            # Keep the ambiguity/confidence that may live below the headline.
            if meters and re.search(r"confidence|also consistent|ambiguous", s) and not s.startswith("cycle "):
                meters[-1] += "; " + s.split("; bar", 1)[0]
    card.tempo = _claim(tempos, "tempo")
    card.key = _claim(keys or homes, "key")
    if keys and homes and any(h not in keys for h in homes):
        card.warnings.append("harmony home differs from KEY; compare quotes KEY as the source decision")
    card.meter = _claim(meters, "meter")
    card.meter_summary = card.meter.split("; [", 1)[0]
    card.shape = _join(shapes or spans)
    if not openings and shapes:
        openings = [shapes[0].split("→", 1)[0].strip()]
    card.opening = _join(openings)
    card.peaks = _join(peaks)
    card.voice = _join(voices)
    if instruments:
        card.instruments = (instrument_method + ": " if instrument_method else "") + _join(instruments)
    if contrasts:
        card.contrast = "source sentiment/music model read: " + _join(contrasts)
    # Only an explicit BPM figure, with no uncertainty, supports a numeric lean.
    number = re.match(r"(\d+(?:\.\d+)?)\s+BPM\b", card.tempo)
    if number and not re.search(r"unclear|likely|lean|unsure|failed|unavailable|ambiguous|approx|≈|~", card.tempo, re.I):
        pulse = float(number[1])
        card.pulse = pulse if 0 < pulse < 1000 else None
    card.warnings = _unique(card.warnings)
    return card


def _cell(text):
    return text.replace("|", "\\|").replace("\n", " ")


def feel_differences(a, b):
    """Five evidence-led reads, including explicit limits when evidence is thin."""
    if a.pulse is not None and b.pulse is not None:
        delta = b.pulse - a.pulse
        if abs(delta) < 5:
            pulse = f"The tracked pulses sit close ({a.pulse:g} / {b.pulse:g} BPM); grouping may matter more to the sway."
        else:
            faster = "B" if delta > 0 else "A"
            pulse = f"{faster} has the quicker tracked pulse ({a.pulse:g} / {b.pulse:g} BPM), a possible stronger forward pull."
        if a.meter_summary != b.meter_summary:
            pulse += f" A: {a.meter_summary}; B: {b.meter_summary}."
        pulse += " Felt pace remains a lean."
    else:
        pulse = "A difference in felt pace is unclear; the tempo evidence is missing or qualified."
    if UNAVAILABLE in (a.opening, b.opening):
        opening = "A difference in how they open is unclear; opening evidence is missing."
    elif a.opening == b.opening:
        opening = "The opening evidence gives no clear difference: " + a.opening + "."
    else:
        opening = f"The first foothold differs: A opens with [{a.opening}]; B with [{b.opening}]."
    if UNAVAILABLE in (a.shape, b.shape):
        shape = "A difference in the rise and retreat is unclear; arc evidence is missing."
    elif a.shape == b.shape:
        shape = "The recorded arcs give no clear difference: " + a.shape + "."
    else:
        shape = f"The rise and retreat differ: A [{a.shape}]; B [{b.shape}]. These are level arcs; where either aches is yours to say."
    if a.voice_arc and b.voice_arc and a.voice_arc != b.voice_arc:
        a_arc, b_arc = a.voice_arc.rsplit(" · ", 1)[-1], b.voice_arc.rsplit(" · ", 1)[-1]
        subject = "melody" if "VOCAL-STEM MELODY:" in a.voice + b.voice else "voice"
        voice = f"The {subject} travels differently: A [{a_arc}]; B [{b_arc}]. That may change how a phrase lands."
    else:
        if "VOCAL-STEM MELODY:" in a.voice + b.voice:
            voice = "A difference in the melody's journey is unclear from these cards; the stem and instrument evidence above keeps its source caveats."
        else:
            voice = "A difference in the voice's journey is unclear from these cards; the voice and instrument evidence above keeps its source caveats."
    if UNAVAILABLE in (a.contrast, b.contrast):
        words = "A difference in how words meet music is unclear; contrast evidence is missing."
    elif a.contrast == b.contrast:
        words = "The words/music evidence gives no clear difference: " + a.contrast + "."
    else:
        def plain(contrast):
            text = contrast.removeprefix("source sentiment/music model read: ")
            text = re.sub(r"\s*\((?:[+-]?\d+(?:\.\d+)?(?: avg)?|r=[+-]?\d+(?:\.\d+)?)\)", "", text)
            return text.split(";", 2)[0] + ("; " + text.split(";", 2)[1].strip() if ";" in text else "")
        words = f"The words meet the sound differently: A [{plain(a.contrast)}]; B [{plain(b.contrast)}]. This is the source model's lean, not anyone's feeling."
    return [pulse, opening, shape, voice, words]


def compare_cards(a, b):
    lines = [f"A: {a.title}", f"B: {b.title}", "", "| Listen for | A | B |", "| --- | --- | --- |"]
    for label, left, right in (
        ("Tempo / feel", a.tempo + "; meter: " + a.meter, b.tempo + "; meter: " + b.meter),
        ("Key / mode", a.key, b.key),
        ("Opens", a.opening, b.opening),
        ("Shape (level arc)", a.shape, b.shape),
        ("Ache candidates / peaks", a.peaks, b.peaks),
        ("Voice / vocal-stem melody" if "VOCAL-STEM MELODY:" in a.voice + b.voice else "Voice", a.voice, b.voice),
        ("Instruments", a.instruments, b.instruments),
        ("Words vs music", a.contrast, b.contrast),
    ):
        lines.append(f"| {label} | {_cell(left)} | {_cell(right)} |")
    lines.extend(["", "Peaks describe level, voice or tension; an ache stays a candidate, never a measured feeling."])
    for side, card in (("A", a), ("B", b)):
        lines.extend(f"{side} source warning: {warning}" for warning in card.warnings)
    lines.extend(["", "The ears' read — leans, not a verdict:", ""])
    lines.extend("- " + line for line in feel_differences(a, b))
    lines.extend(["", INVITATION])
    return "\n".join(lines) + "\n"


def resolve_card(value, cards_dir=None):
    path = Path(value).expanduser()
    if path.is_file():
        return path
    cards_dir = Path(cards_dir) if cards_dir is not None else Path.home() / ".cache/raven/cards"
    # Exact normalised identity only. A title alone never guesses an artist.
    matches = [p for p in cards_dir.glob("*.txt") if slug(p.stem) == slug(song_name(value))]
    if len(matches) != 1:
        reason = "ambiguous" if matches else "not found"
        raise ValueError(f"card {reason}: {value}; give an existing card path or full Artist - Title")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--cards-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        paths = [resolve_card(value, args.cards_dir) for value in (args.a, args.b)]
        cards = [parse_card(_read(path), path.stem) for path in paths]
        print(compare_cards(*cards), end="")
    except (OSError, ValueError) as exc:
        print(f"comparison unavailable: {exc}\n{INVITATION}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
