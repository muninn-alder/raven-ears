"""Small, source-faithful view helpers for modes.py; no state or engine work."""
from dataclasses import replace
import re

# Sources of identical events are combined, never uncertainty/credit wording.
TAG = re.compile(r"\s*\[([^\[\]]+)\]\s*$")
STANDING = ("Still beyond these ears: stereo/space, reverb, production tricks, "
            "backing harmonies.")


def dedupe(events):
    groups = {}
    for event in events:
        body, tags = event.text.rstrip(), []
        while (match := TAG.search(body)):
            tags.insert(0, match[1])
            body = body[:match.start()].rstrip()
        key = (event.t, event.end, event.channel, event.approx, body)
        sources = tags + ([event.source] if event.source else [])
        if key not in groups:
            groups[key] = (replace(event, text=body), [])
        for source in sources:
            if source not in groups[key][1]:
                groups[key][1].append(source)
    result = []
    for event, sources in groups.values():
        # Single events keep source tags too; duplicate rows name each origin.
        suffix = f" [{'; '.join(sources)}]" if sources else ""
        result.append(replace(event, text=event.text + suffix))
    return sorted(result, key=lambda e: e.t)


def local_events(events, start, end, duration=None):
    """Clip known spans; omit prose whose later evidence cannot be separated.

    Borrowed from Seven and Sunny's seven-ears (MIT): no future evidence inside
    a first-listen view. Row-level lyrics cannot honestly be split into words.
    """
    import modes as M
    selected = []
    word_times = sorted({e.t for e in events if e.channel == "words"})
    word_limits = dict(zip(word_times, word_times[1:] + [duration]))
    for event in events:
        if event.t >= end or (event.t < start and (event.end is None or event.end <= start)):
            continue
        body = event.text
        if re.search(r"last chorus|last chord|to the end|this song|whole.song|throughout|"
                     r"overall|final |outro|peak|times where|earlier choruses|returning|"
                     r"wordless for|finishes|ends on", body, re.I):
            continue
        word_end = event.end if event.end is not None else word_limits.get(event.t)
        if event.channel == "words" and word_end is not None and word_end > end:
            body = "Words cross this pause; text withheld because word timings are unavailable."
        else:
            # Clip explicit intervals and holds. Other future references are unsafe.
            def span(match):
                a, b = M.seconds(match[1]), M.seconds(match[2])
                if a >= end or b <= start:
                    return match[0]
                return f"{M.clock(max(start, a))}–{M.clock(min(end, b))}"
            body = re.sub(rf"({M.TIME})[–−-]({M.TIME})", span, body)
            body = re.sub(rf"\buntil ({M.TIME})", lambda m: "until " + M.clock(min(end, M.seconds(m[1]))), body)
            if any(M.seconds(t) > end for t in M.STAMP.findall(body)):
                continue
            # Aggregate numeric comparisons are calculated over the original span;
            # retain static labels/chord sequences only when the span is complete.
            if event.end is not None and (event.end > end or event.t < start) and re.search(
                    r"→|\bdB\b|%|confidence|climbs|falls|rises|[IViv]+\s*[–-]\s*[IViv]+", body):
                continue
        selected.append(replace(event, t=max(start, event.t), text=body,
                                end=min(end, event.end) if event.end is not None else None))
    return sorted(selected, key=lambda e: e.t)


def run_status(card, analysis=None):
    """Report only recorded outcomes. Never infer success from missing warnings."""
    import modes as M
    from stage_status import card_notes
    precise = card_notes(card.raw, analysis)
    data = analysis or {}
    raw = card.raw
    notes = list(card.warnings)

    def diagnostic(value, path=""):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'stage_status':
                    continue
                location = f"{path} {key}".strip()
                if key in ("status", "error", "errors", "note", "notes", "lyrics_notes") and item:
                    for entry in item if isinstance(item, list) else [item]:
                        if not isinstance(entry, (dict, list)):
                            notes.append(f"{location}: {entry}")
                if isinstance(item, (dict, list)):
                    diagnostic(item, location)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    diagnostic(item, path)
    diagnostic(data)
    issues = []
    for note in notes:
        if note.startswith('NOTES:') or not M.FAILURE.search(note):
            continue
        # Classify the stage, then discard arbitrary diagnostics (may quote lyrics).
        lower = note.lower()
        stage = next((name for pattern, name in (
            (r"credits|discogs|musicbrainz", "credits lookup"),
            (r"lyrics|whisper|words", "lyrics"), (r"cuda|gpu", "GPU stage"),
            (r"instrument|tagger", "instrument stage"), (r"madmom|cnn|key", "key backend"),
            (r"demucs|stems?", "stem stage"), (r"vocal|singer", "voice stage"),
            (r"harmony", "harmony engine"), (r"music", "music engine"),
            (r"ache", "ache stage")) if re.search(pattern, lower)), "engine/stage")
        outcome = "skipped" if re.search(r"skip|disabled|not run", lower) else "failed/unavailable"
        issues.append(f"{stage} {outcome}")
    combined = bool(re.search(r"^=+ (?:music|singer|harmony)\s*$", raw, re.M))
    if combined:
        for engine, header in (("music", r"^MUSIC:"), ("singer", r"^SINGER\.PY"), ("harmony", r"^HARMONY:")):
            if not re.search(header, raw, re.M):
                issues.append(f"{engine} engine skipped or produced no card")
    if precise:
        return ["DIDN'T HEAR: " + " · ".join(dict.fromkeys(precise + issues)), STANDING]
    from lyrics_shared import displayed_lyric_source
    origin = displayed_lyric_source(data, raw, has_words=bool(card.lyrics))
    if card.lyrics and origin not in ("file", "synced", "plain"):
        issues.append("no writer's lyrics recorded; words are Whisper's guesses" if origin == "whisper"
                      else "no writer's lyrics recorded; word provenance unclear")
    elif not card.lyrics:
        issues.append("no timed words in this card; lyric stage status unclear" if not any("lyrics " in x for x in issues) else "")
    if not re.search(r"^CREDITS \(", raw, re.M) and not any("credits lookup" in x for x in issues):
        credits = data.get("credits")
        if re.search(r"^\s*CREDITS:.*(?:no match|no performer|no usable|no artist/title|could not identify)", raw, re.M | re.I):
            issues.append("credits lookup supplied no performer credits")
        elif not credits:
            issues.append("credits lookup not recorded (skipped or unavailable)")
    issues = list(dict.fromkeys(x for x in issues if x))
    if issues:
        line = "DIDN'T HEAR: " + "; ".join(issues) + "."
        if any(M.FAILURE.search(n) for n in notes):
            line += " Backend failure/skip reported; missing detail is not silence."
    else:
        line = "DIDN'T HEAR: no failures/skips recorded for the represented stages; audio stage status not recorded (older card)."
    return [line, STANDING]
