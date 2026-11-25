"""Shared lyrics for the music ears (24 Oct 2026).  `from lyrics_shared import get_words`

ONE transcription per file, used by music.py, singer.py and harmony.py:
  * cache: audio hash, names, language, model and device/precision (RAVEN_LYRICS_CACHE moves it)
  * Whisper large-v3 by default (RAVEN_LYRICS_WHISPER overrides), no VAD (the speech VAD
    throws away sung words), one model load for the main pass AND the gap re-listen
  * gap re-listen: Whisper sometimes skips 30 s of sung text in a one-pass transcript.
    Every lyric gap >= 9 s is re-heard in 25 s windows and kept if it clears the usual gates
  * end-of-track hallucinations ("Thank you.", "Thanks for watching", "Subtitles by ...") are dropped when they
    sit in the last ~15 s of the track or over near-silence, in the main pass and in the gap windows alike

Returns [(word, start_s, end_s), ...] or None. Local only.
"""
from __future__ import annotations

from network_policy import offline, require_online
from slim import NO_VOICE_MARK, split_voice_mark
from stage_status import names_note_text, whisper_names_note

import gc
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.environ.get("RAVEN_LYRICS_CACHE", os.path.expanduser("~/.cache/raven-lyrics"))
VERSION = 8  # First-audio-stream policy; old words may have used automatically selected stems.
TAIL_S = 15.0           # a hallucination phrase this close to the end is dropped
SILENCE_DB = 35.0       # ...or anywhere the region is this far under the song's loud level
VOCAL_GONE_DB = 30.0    # a segment this far under the vocals stem's own loud level has no singer in it
UNSURE = []             # [(start_s, end_s)] doubtful-but-sung spans from the last get_words() in this process

# ---- names list (hotwords / initial_prompt) --------------------------------------------------------------------
NAMES_FILE = None       # set by configure(); None -> RAVEN_EARS_HOME/names.txt if it exists
EXTRA_NAMES = []        # per-run additions (e.g. proper nouns pulled from a --lyrics file)
MAX_NAMES = 30          # Whisper's prompt window is ~224 tokens; a long list would crowd out the audio's own context


def is_vocable(text):
    """A repeated line of explicit singing syllables, not a language detector."""
    syllables = re.findall(r"[^\W\d_]+", text.casefold())
    allowed = {"eeh", "ooh", "ah", "la", "na", "mm", "oh", "hey", "yeah"}
    return (len(syllables) >= 2 and all(s in allowed for s in syllables)
            and len(set(syllables)) < len(syllables)
            and not re.search(r"[\d_]", text))


def lyric_excerpt(text, limit=38):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip(" ,-") + "…"


def lyric_lookup(analysis):
    data = analysis if isinstance(analysis, dict) else {}
    source = data.get("lyrics_source")
    if not isinstance(source, dict) and isinstance(data.get("lyrics"), dict):
        source = data["lyrics"].get("lookup")
    return source if isinstance(source, dict) else {}


def lyric_labels(text):
    return re.findall(r"(?m)^[ \t]*(?:NOTE: )?(LYRICS:[^\n]*)$", text)


def displayed_lyric_source(analysis=None, text="", *, has_words=False):
    """Classify the displayed words using resolver status or an explicit old label."""
    status = lyric_lookup(analysis).get("status")
    if status in ("file", "synced", "plain", "instrumental"):
        return status
    if status in ("nomatch", "empty", "off", "skipped", "failed"):
        return "whisper" if has_words else None
    labels = lyric_labels(text)
    patterns = (
        ("whisper", r"(?:words are )?Whisper['’]s guesses\b|"
                    r".+; words are Whisper['’]s guesses\b|\d+ lines? \(whisper [^,]+, rough times\)"),
        ("file", r"writer['’]s text\b"),
        ("synced", r"LRCLIB synced (?:lyrics\b|times disagree with this audio\b)"),
        ("plain", r"LRCLIB (?:(?:had no synced lyrics|synced lookup incomplete|synced lyrics were unusable); )?plain lyrics\b"),
        ("instrumental", r"instrumental \(LRCLIB\)"),
    )
    for source, pattern in patterns:
        if labels and all(re.match(r"LYRICS:\s*(?:" + pattern + ")", label, re.I) for label in labels):
            return source
    return None


def supplied_lyrics(analysis=None, text=""):
    """Read the resolver's displayed source and rows; never infer it from ≈.

    Text-only cards may use their explicit displayed-lyrics label. Structured
    lookup status, when present, wins over that legacy fallback.
    """
    data = analysis if isinstance(analysis, dict) else {}
    if isinstance(data.get("lyrics"), dict):
        rows = data["lyrics"].get("rows")
    elif isinstance(data.get("lyrics_aligned"), dict):
        rows = data["lyrics_aligned"].get("lines")
    else:
        rows = data.get("lyrics")
    status = displayed_lyric_source(data, text)
    name = {"file": "the supplied file", "synced": "LRCLIB", "plain": "LRCLIB"}.get(status)
    if not name:
        return None, []
    if not isinstance(rows, list):
        # Only quoted lyric blocks/timelines, never transcript gap prose.
        # 16 Nov 2026 real proof: the full card's TIMELINE also has "[m:ss]"
        # rows ("♪ drop-out: …"), so read only the LYRICS block's own rows.
        rows, block = [], []
        inside = False
        for line in text.splitlines():
            if re.match(r"\s*LYRICS:", line):
                inside = True
                continue
            if inside and re.match(r"\s*\[\d+:\d\d", line):
                block.append(line)
            elif inside and not line.startswith("  "):
                inside = False
        for match in re.finditer(r'(?m)^\s*\[(\d+):(\d\d)(?:≈)?\]\s*(.*)$', "\n".join(block)):
            body = match[3]
            if body.lstrip().startswith("♪"):
                continue
            body, no_voice = split_voice_mark(body)
            rows.append(dict(t=int(match[1]) * 60 + int(match[2]), text=body, no_voice=no_voice))
    kept = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("text"), str) and isinstance(row.get("t"), (int, float)) and np.isfinite(row["t"]):
            words, no_voice = split_voice_mark(row["text"])
            kept.append(dict(row, text=words, no_voice=True) if no_voice else row)
    return name, kept


def alignment_note(text, source):
    if not source:
        return text
    return re.sub(r"lyrics: (\d+) line\(s\) kept but marked \(unsure\), quiet vocals or doubtful\s+Whisper:",
                  lambda m: f"lyrics: Whisper was doubtful on {m[1]} line(s) "
                  f"(alignment check only; the displayed lyrics are from {source}):", text)


LINE_SLACK = 1.5


def voice_overlaps(start, end, spans):
    """A supplied line can start slightly early or late."""
    return any(a < end and b > start or a <= start + LINE_SLACK and b >= start - LINE_SLACK
               for a, b in spans)


def lyric_line_at(rows, t):
    """Pick the placed line at an event, allowing for rounded card clocks."""
    if not rows:
        return None
    near = [r for r in rows if r["text"].strip() and abs(r["t"] - t) <= 0.5]
    if near:
        return min(near, key=lambda r: abs(r["t"] - t))
    active = [r for r in rows if r["text"].strip() and r["t"] <= t < (r.get("end") or r["t"])]
    if active:
        return max(active, key=lambda r: r["t"])
    return min((r for r in rows if r["text"].strip() and 0 <= r["t"] - t <= LINE_SLACK),
               key=lambda r: r["t"], default=None)


_ENTRANCE_QUOTE = re.compile(
    r"(voice enters:|first timed lyric \(voice onset unclear\):) (?:'(.*)'|vocal stem active)"
    r"( \((?:words approximate here|time approximate)\))?$")
_LINE_QUOTE = re.compile(r"(voice:|line) '(.*)'(: .*)?$")
_REFRAIN_QUOTE = re.compile(
    r"(last chorus: |the last chorus breaks the count: |"
    r"the last chorus breaks the count in the AUDIO: the vocal stem has |refrain check: |)"
    r"'(.+)'( (?:comes|arrives) \d.*| \d+ times where.*|: the voice sings it .*)$")
_LANDING_QUOTE = re.compile(r'(.+) under (?:≈ )?"[^"]*"( \((?:sure|lean|unclear)\))$')


def event_lyric_quote(label, t, rows):
    """Use placed words for generated event quotes. Keep the event's timing."""
    if not rows:
        return label
    entry = _ENTRANCE_QUOTE.fullmatch(label)
    line = _LINE_QUOTE.fullmatch(label)
    refrain = _REFRAIN_QUOTE.fullmatch(label)
    landing = _LANDING_QUOTE.fullmatch(label)
    if not (entry or line or refrain or landing):
        return label
    row = lyric_line_at(rows, t)
    words = row["text"].strip() if row else ""
    quote = f"'{words[:60]}'"
    if entry:
        if not words:
            return "voice enters: vocal stem active" if entry[1] == "voice enters:" else entry[1][:-1]
        hedge = " (time approximate)" if entry[3] or entry[2] is None else ""
        return f"{entry[1]} {quote}{hedge}"
    if line:
        suffix = line[3] or ""
        if not words:
            return ("voice: words not placed here" if line[1] == "voice:" else "line" + suffix)
        return f"{line[1]} {quote}{suffix}"
    if landing:
        excerpt = words.replace('"', "'")
        excerpt = excerpt[:100] + "…" if len(excerpt) > 100 else excerpt
        mark = "≈ " if row and row.get("approx") else ""
        return landing[1] + (f' under {mark}"{excerpt}"' if excerpt else "") + landing[2]
    # A refrain count measures the detected phrase. Quote the corresponding
    # words only when the placed line contains that phrase.
    match = re.search(r"(?<!\w)" + re.escape(refrain[2]) + r"(?!\w)", words, re.I)
    phrase = f"'{match[0]}'" if match else "a refrain phrase"
    return refrain[1] + phrase + refrain[3]


def supplied_gap_description(start, end, rows):
    """Defer to displayed line starts inside the span, without claiming ASR heard them."""
    # A line stamped a moment before the span still sings into it (card spans
    # and lyric stamps are rounded separately), so allow LINE_SLACK seconds.
    inside = [r["text"] for r in rows if start - LINE_SLACK <= r["t"] < end
              and r["text"].strip() and not r.get("no_voice")]
    if not inside:
        return None
    from lyric_align import quote_line

    if all(is_vocable(line) for line in inside):
        return f'wordless singing (lyrics: {quote_line(lyric_excerpt(inside[0]))})'
    words = next(line for line in inside if not is_vocable(line))
    return f"voice present; lyrics have words here Whisper didn't hear ({quote_line(lyric_excerpt(words))})"


def supplied_gap_text(text, rows):
    """Refresh generated claims only; lyric quotes and measured quiet stay intact."""
    if not rows:
        return text
    stamp = r"\d+:\d\d"
    old = "voice present, words not heard (Whisper found none)"
    leaked = (r'voice present; lyrics have words here Whisper didn\'t hear \("[^\n]*?'
              + re.escape(NO_VOICE_MARK) + r'"?\)')
    seconds = lambda t: sum(int(v) * unit for v, unit in zip(t.split(":"), (60, 1)))
    def replace(match):
        a, b = match[1], match[2]
        description = supplied_gap_description(seconds(a), seconds(b or a), rows)
        return f"{a}{'–' + b if b else ''}: {description or old}"
    out = []
    for line in text.splitlines(keepends=True):
        if re.match(r"\s*\[", line):
            out.append(line)
            continue
        summary = re.fullmatch(rf"(\s*){re.escape(old)}: ((?:{stamp}–{stamp})(?:, {stamp}–{stamp})*)\s*", line)
        if summary:
            parts = []
            for a, b in re.findall(rf"({stamp})–({stamp})", summary[2]):
                description = supplied_gap_description(seconds(a), seconds(b), rows) or old
                parts.append(f"{summary[1]}{description}: {a}–{b}")
            line = "\n".join(parts) + ("\n" if line.endswith("\n") else "")
        else:
            # Match only a whole generated span description, not lyric words.
            line = re.sub(rf"({stamp})(?:–({stamp}))?: (?:{re.escape(old)}|{leaked})(?=; |\n|$)", replace, line)
        out.append(line)
    return "".join(out)


def vocal_changes(rows):
    """Read measured stem transitions, including grouped entrances/exits."""
    changes = []
    for t, label in rows:
        state = None
        for part in label.split(' · '):
            if part.startswith('opens with '):
                state = 'vocals' in part.removeprefix('opens with ').split(' + ')
            elif part.startswith('everything drops out'):
                state = part == 'everything drops out but vocals'
            else:
                match = re.match(r'(.+?) (enter|return|drop out)\b', part)
                if match and 'vocals' in match[1].split(' + '):
                    state = match[2] != 'drop out'
        if state is not None:
            changes.append((t, state))
    return sorted(changes)


def vocal_onset(rows):
    """First measured entrance, including voice already active at the open."""
    openings = [(t, part) for t, label in rows for part in label.split(' · ')
                if re.search(r'^opens with | (?:enter|return)\b', part)]
    return next((t for t, on in vocal_changes(openings) if on), None)


def vocal_spans(rows, duration):
    changes = vocal_changes(rows)
    return [(t, changes[i + 1][0] if i + 1 < len(changes) else duration)
            for i, (t, on) in enumerate(changes) if on]


def word_gaps(words, duration, min_gap=18.0):
    """Transcript gaps, never evidence of absent words or voice in the audio."""
    end, gaps = 0.0, []
    for _, start, stop in sorted(words or [], key=lambda w: w[1]):
        if start - end >= min_gap:
            gaps.append((end, start))
        end = max(end, stop)
    if duration - end >= min_gap:
        gaps.append((end, duration))
    return gaps


def gap_description(start, end, detectors=(), words_available=True):
    """Each detector is measured voice spans, [] for quiet, or None if unavailable.

    Any positive wins. Missing measurements cannot establish absence.
    """
    found = any(any(b > start and a < end for a, b in spans) for spans in detectors if spans is not None)
    words = "words not heard (Whisper found none)" if words_available else "words unavailable (transcription did not run or failed)"
    if found:
        return "voice present, " + words
    if detectors and all(spans is not None for spans in detectors):
        return "no voice (instrumental)"
    return words + "; voice presence unverified"


def gap_segments(start, end, detectors=(), words_available=True, *, omitted=None):
    """Scope each voice claim to its measured spans, including partial gap overlap."""
    spans = [(max(start, a), min(end, b)) for detector in detectors if detector is not None
             for a, b in detector if b > start and a < end]
    cuts = sorted({start, end, *(t for span in spans for t in span)})
    parts = []
    for a, b in zip(cuts, cuts[1:]):
        description = gap_description(a, b, detectors, words_available)
        if parts and parts[-1][2] == description:
            parts[-1] = (parts[-1][0], b, description)
        else:
            parts.append((a, b, description))
    # A detector edge can nick the transcript gap. Omit that fragment rather
    # than extending an instrumental claim beyond its measured bounds.
    kept = []
    for i, (a, b, description) in enumerate(parts):
        if (len(parts) > 1 and i in (0, len(parts) - 1) and b - a < 2
                and description.startswith("voice present")):
            if omitted is not None:
                omitted.add((a, b))
        else:
            kept.append((a, b, description))
    return kept


def gap_edge_note(omitted):
    n = len(omitted)
    return f"{n} short voice fragment{'s' if n != 1 else ''} (<2 s) at a gap edge not shown"


def span_text(a, b, clock):
    """'2:48' rather than '2:48–2:48' when both ends print as the same second."""
    return clock(a) if clock(a) == clock(b) else f"{clock(a)}–{clock(b)}"


def gap_claims(start, end, detectors=(), words_available=True, *, omitted=None):
    clock = lambda t: f"{int(t) // 60}:{int(t) % 60:02d}"
    return "; ".join(f"{span_text(a, b, clock)}: {description}"
                     for a, b, description in gap_segments(start, end, detectors, words_available, omitted=omitted))


def clean_gap_claims(text, evidence="", analysis=None):
    """Correct legacy saved prose from its printed measurements, without a listen."""
    stamp = r"\d+:\d\d"
    words = (r"words (?:not heard \(Whisper found none\)|"
             r"unavailable \(transcription did not run or failed\))")
    description = (rf"voice present, {words}|"
                   r"no voice detected; words not heard \(Whisper found none\)|"
                   r"no voice \(instrumental\)|"
                   rf"{words}; voice presence unverified")
    claim = rf"({stamp})(?:–({stamp}))?: (?:{description})"
    pattern = rf"{claim}(?:; {claim})*"
    stages = analysis.get("stage_status", []) if isinstance(analysis, dict) else []
    stages = [r for r in stages if isinstance(r, dict) and r.get("stage") == "words"
              and isinstance(r.get("status"), str)] if isinstance(stages, list) else []
    available = all(r.get("status") == "ran" for r in stages) if stages else None
    def claims(a, b, original):
        heard = available if available is not None else "words unavailable" not in original
        return gap_claims(a, b, detectors, heard, omitted=omitted)
    seconds = lambda t: sum(int(v) * unit for v, unit in zip(t.split(":"), (60, 1)))
    spans = []
    measured = False
    energy = []
    energy_measured = False
    energy_incomplete = False
    for line in (text + "\n" + evidence).splitlines():
        body = line.strip()
        if body.startswith("VOCALS:"):
            found = re.findall(rf"({stamp})–({stamp})", body)
            energy += [(seconds(a), seconds(b)) for a, b in found]
            energy_measured |= bool(found) or body.startswith("VOCALS: instrumental")
            energy_incomplete |= bool(re.search(r"\+\d+ more", body))
        if body.startswith(("VOCALS:", "voice with no words:", "voice present, words not heard")):
            found = re.findall(rf"({stamp})–({stamp})", body)
            spans += [(seconds(a), seconds(b)) for a, b in found]
            measured |= bool(found)
    if energy_measured:
        spans, measured = energy, True
    detectors = (spans if measured else None,)
    if energy_measured and energy_incomplete:
        detectors += (None,)
    timeline_claims = set()
    for source in (text + "\n" + evidence).splitlines():
        match = re.match(rf"\s*{stamp}\s+(.+)$", source)
        if match and re.fullmatch(pattern, match[1]):
            timeline_claims.add(match[1])
    out, omitted, previous = [], set(), None
    for line in text.splitlines():
        note = re.match(r"\s*NOTE:\s*(.+)$", line)
        if note and note[1] in timeline_claims:
            continue
        label = "words unavailable (transcription did not run or failed)" if available is False else "words not heard (Whisper found none)"
        line = re.sub(r"^(\s*)voice with no words:", r"\1voice present, " + label + ":", line)
        match = re.match(rf"(\s*({stamp})\s+)the words stop; wordless for ({stamp})(?:, until ({stamp})| to the end)$", line)
        if match:
            prefix, start, length, end = match.groups()
            a = seconds(start)
            b = seconds(end) if end else a + seconds(length)
            line = prefix + claims(a, b, line)
        elif '" — ' in line and re.match(rf"\s*\[{stamp}≈?\]", line):
            # Transcript gaps belong on the timeline, never on a lyric row.
            # Keep the quoted words and other musical annotations untouched.
            quote, tail = line.split('" — ', 1)
            tail = re.sub(rf"(?:, )?the words stop; wordless for {stamp}, until {stamp}", "", tail)
            tail = re.sub(rf"(?:, )?{stamp}(?:–{stamp})?: (?:{description})(?:; )?", "", tail).strip(" ,;")
            line = quote + ('" — ' + tail if tail else '"')
        if energy_measured or available is False:
            # Only generated gap annotations, never quoted lyric text. Repair
            # saved pitch-based claims as well as the older "wordless" prose.
            def replace_voice(match):
                bounds = re.findall(claim, match[0])
                a, b = seconds(bounds[0][0]), seconds(bounds[-1][1] or bounds[-1][0])
                if energy_measured and b > a:
                    return claims(a, b, match[0])
                return match[0].replace("words not heard (Whisper found none)",
                                       "words unavailable (transcription did not run or failed)") if available is False else match[0]
            if '" — ' in line:
                quote, tail = line.split('" — ', 1)
                line = quote + '" — ' + re.sub(pattern, replace_voice, tail)
            elif not re.match(rf"\s*\[{stamp}≈?\]", line):
                line = re.sub(pattern, replace_voice, line)
        timeline = re.match(rf"(\s*){stamp}\s+({pattern})$", line)
        if timeline:
            # Each detector segment gets its own row and actual start time;
            # condensed cards can then select only voice-present gaps.
            for part in re.split(rf"; (?={stamp}(?:–|: ))", timeline[2]):
                a, b, desc = re.fullmatch(rf"({stamp})(?:–({stamp}))?: (.+)", part).groups()
                b = b or a
                if previous and previous[2] == a and previous[3] == desc and a != b:
                    idx, a, _, _, indent = previous
                    out[idx] = f"{indent}{a}  {a}–{b}: {desc}"
                else:
                    idx, indent = len(out), timeline[1]
                    out.append(f"{indent}{a}  {part}")
                previous = (idx, a, b, desc, indent) if a != b else None
            continue
        if not re.match(rf"\s*{stamp}\s+[^'\"]+$", line):
            previous = None
        out.append(line)
    if omitted:
        at = len(out) - 1 if out and out[-1] == "and now, you" else len(out)
        out.insert(at, "NOTES: " + gap_edge_note(omitted))
    result = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    _, rows = supplied_lyrics(analysis, text)
    return supplied_gap_text(result, rows)


def ears_home():
    return os.environ.get("RAVEN_EARS_HOME", os.path.expanduser("~/.local/share/raven-ears"))


def default_names_path():
    return os.path.join(ears_home(), "names.txt")


def load_names(path=None):
    """One name per line; blank lines and #-comments ignored. Missing file -> []."""
    path = path or default_names_path()
    out = []
    try:
        for ln in open(path, encoding="utf-8"):
            ln = ln.split(" #")[0].strip()
            if ln and not ln.startswith("#") and ln not in out:
                out.append(ln)
    except OSError:
        pass
    return out


def configure(names_file=None, extra_names=()):
    """Set once per process (CLI): which names file to read, and extra names for this run only."""
    global NAMES_FILE
    NAMES_FILE = names_file
    EXTRA_NAMES[:] = [n for n in extra_names if n]


def configure_for_run(names_file=None, lyric_lines=None):
    """CLI helper: names file (default RAVEN_EARS_HOME/names.txt) plus, when lyrics are given, the rare words in them."""
    configure(names_file, rare_words(lyric_lines) if lyric_lines else ())


def effective_names():
    seen, out = set(), []
    for n in load_names(NAMES_FILE) + list(EXTRA_NAMES):
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out[:MAX_NAMES]


def names_tag(names=None):
    """Short hash of the names; part of the cache key so a changed list re-transcribes."""
    names = effective_names() if names is None else names
    return hashlib.sha1((names_mode("prompt") + "\n" + "\n".join(names)).encode("utf-8")).hexdigest()[:10] if names else ""


def names_mode(default="prompt"):
    """'hotwords' or 'prompt'. RAVEN_EARS_NAMES_MODE overrides. Both speech and sung words default to 'prompt':
    on nine real voice notes (24 Nov 2026), hotwords sometimes made large-v3 drop all case and punctuation and invent
    an ending, and the prompt didn't. Prompting is a hint, not evidence that a word was heard."""
    return os.environ.get("RAVEN_EARS_NAMES_MODE", default)


def prompt_token_budget(model):
    """How many initial_prompt tokens Whisper really keeps: faster-whisper takes the LAST max_length // 2 - 1
    (223 of 448) and silently drops the head."""
    try:
        return int(getattr(model, "max_length", 448)) // 2 - 1
    except (TypeError, ValueError):
        return 223


def _token_counter(model):
    """Count tokens the way faster-whisper does (encode(" " + text)), using the loaded model's own tokenizer.
    None when the model carries no tokenizer (then nothing can be measured, and nothing is trimmed)."""
    tok = getattr(model, "hf_tokenizer", None)
    if tok is None or not hasattr(tok, "encode"):
        return None
    try:
        tok.encode(" x", add_special_tokens=False)
    except Exception:
        return None
    return lambda text: len(tok.encode(" " + text.strip(), add_special_tokens=False).ids)


def fit_prompt(model, names, style="", budget=None):
    """Assemble an initial_prompt of names (first) then `style` (last, because the tail survives) inside Whisper's
    token budget. Over budget, names are dropped from the end of the list first; only if the style alone is over
    is the style cut (at a word boundary, from its end). Returns (prompt, note_or_None).
    With no tokenizer on the model the prompt is assembled untrimmed."""
    names = list(names or [])
    budget = prompt_token_budget(model) if budget is None else budget
    join = lambda ns: ", ".join(ns) + "." if ns else ""
    build = lambda ns, st: " ".join(x for x in (join(ns), st) if x)
    count = _token_counter(model)
    if count is None or count(build(names, style)) <= budget:
        return build(names, style), None
    note = None
    words = style.split()
    if style and count(style) > budget:
        lo, hi = 0, len(words)                     # most words that fit on their own
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if count(" ".join(words[:mid])) <= budget else (lo, mid - 1)
        over = count(style) - budget
        style = " ".join(words[:lo])
        note = f"style prompt trimmed, {over} tokens over (Whisper keeps {budget} prompt tokens)"
        return build([], style), note
    lo, hi = 0, len(names)                          # most names that fit beside the style
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if count(build(names[:mid], style)) <= budget else (lo, mid - 1)
    what = "to keep the verbatim style" if style else "to fit"
    note = (f"names prompt trimmed: {len(names) - lo} of {len(names)} names dropped {what} "
            f"(Whisper keeps {budget} prompt tokens)")
    return build(names[:lo], style), note


def whisper_name_kwargs(model, names=None, mode=None, notes=None):
    """kwargs for model.transcribe: `hotwords` if asked for and this faster-whisper has it, else a short
    `initial_prompt` (comma-separated names), trimmed to Whisper's token budget (said in `notes`, never silent)."""
    names = effective_names() if names is None else names
    if not names:
        return {}
    import inspect
    mode = mode or names_mode()
    try:
        params = inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):
        params = {}
    if mode == "hotwords" and "hotwords" in params:
        return {"hotwords": ", ".join(names)}
    if "initial_prompt" in params:
        prompt, note = fit_prompt(model, names)
        if note and notes is not None and note not in notes:
            notes.append(note)
        return {"initial_prompt": prompt}
    return {}


def names_echo(text, names=None, doubtful=False):
    """True when a segment is just the names list read back (a known hotwords/prompt hallucination over quiet or
    empty audio): nearly every word is a name or a near-miss of one (an approximate spelling of a configured token). A single word counts only
    when Whisper itself was doubtful about the segment; two or more always."""
    names = effective_names() if names is None else names
    if not names:
        return False
    import difflib
    known = sorted({x for n in names for x in re.findall(r"\w+", n.lower().replace("’", "").replace("'", ""))})
    toks = re.findall(r"\w+", text.lower().replace("’", "").replace("'", ""))
    if not toks or (len(toks) < 2 and not doubtful):
        return False
    hit = sum(1 for t in toks if t in known or difflib.get_close_matches(t, known, n=1, cutoff=0.8))
    return hit >= 0.8 * len(toks)


_SMALL = set("i a an the and or but of to in on at is are was were be am my me you your we he she it they them his her our "
             "their this that with for from as so not no oh yeah".split())


def rare_words(lines, limit=20):
    """Likely proper nouns / rare words in the writer's lyrics: capitalised words that are not the first word of a
    sentence, and any word with a non-ASCII letter (voilà, naïve). Common small words are skipped."""
    found = []
    for ln in lines:
        toks = re.findall(r"[^\W\d_][\w'’-]*", ln, flags=re.UNICODE)
        for i, w in enumerate(toks):
            w = w.strip("'’-")
            if len(w) < 3 or "'" in w or "’" in w or w.lower() in _SMALL:      # contractions are never names
                continue
            nonascii = any(ord(c) > 127 and c.isalpha() for c in w)
            mid_cap = i > 0 and w[0].isupper() and not w.isupper()
            if (nonascii or mid_cap) and w not in found:
                found.append(w)
    return found[:limit]


# Credits and filler Whisper invents over wordless music: dropped wherever they sit (a song does not sing them).
_CREDIT_ANY = ("transcript", "transcription by", "emily beynon", "captions by", "caption by", "closed caption",
               "copyright", "all rights reserved", "www.", ".com", "http", "subtitle", "amara.org", "for watching",
               "thank you for listening", "please subscribe", "like and subscribe")


def is_credit(text):
    """Invented credit/filler lines, and lines of nothing but music notes or (c)."""
    t = text.strip()
    if "©" in t or t.lower() == "(c)" or not any(c.isalnum() for c in t):
        return True
    low = t.lower()
    return any(h in low for h in _CREDIT_ANY)


_HALL_EXACT = re.compile(
    r"^(thank you( so much| very much)?( for (watching|listening|your time))?|thanks( so much)?( for (watching|listening))?|"
    r"subtitles?( by.*)?|subtitles? by the amara\.org community|amara\.org.*|translated by.*|"
    r"please (like|subscribe).*|(like and )?subscribe.*|see you (next time|in the next (video|one))|"
    r"bye( bye)?|goodbye|you|\.+|music|applause)$")
_HALL_ANY = ("thank you for watching", "thanks for watching", "amara.org", "subtitles by", "subscribe to",
             "thank you for listening")


def _norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w.' ]", " ", t.casefold().replace("’", "'"))).strip(" .")


def is_hallucination(text):
    n = _norm(text)
    if not n:
        return not any(c.isalnum() for c in text)
    return bool(_HALL_EXACT.match(n)) or any(h in n for h in _HALL_ANY)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_LYRIC_TEXT = ContextVar("lyric_text", default=((), "published lyrics"))
# Conservative text hints, not a general language detector. Shared words (la,
# de, que, etc.) cannot distinguish these languages and deliberately cast no vote.
_LANG_WORDS = {
    "en": "the and you your my with this that was were not but have for from they i it is of to in on a me he she we no",
    "fr": "je tu il elle nous vous ils elles est suis dans des les une mon mes qui sur pas pour le la de du et en un que au ce ne se",
    "es": "yo tú él ella nosotros ellos las los una del sus por como pero sin más muy hay ya y el al para se en un de con que no es su me mi te",
    "de": "ich du er wir ihr sie und der die das ist nicht mein mit ein eine dich mich zu den von im auf es in",
    "it": "io lui lei noi voi loro gli della delle sono sei mio miei nella perché anche più e il la le un una di che non con per si mi ti tu se",
    "pt": "eu você ele ela nós vocês eles elas não uma meu minha seus suas pelo pela com o a os as de que e um em para por se no na",
    "nl": "ik jij hij zij wij jullie het een niet mijn zijn van voor met maar ook als de en je is op te dat die in",
}
_LANG_WORDS = {lang: set(words.split()) for lang, words in _LANG_WORDS.items()}


@contextmanager
def lyric_context(lines=(), source="published lyrics"):
    """Run-local text evidence, shared by injectable two-argument word loaders."""
    token = _LYRIC_TEXT.set((tuple(lines or ()), source))
    try:
        yield
    finally:
        _LYRIC_TEXT.reset(token)


def _text_tokens(text):
    return re.findall(r"[^\W\d_]+", text.casefold(), re.UNICODE)


def language_choice():
    """Whole-song evidence: short foreign phrases cannot select a language.

    Require eight votes, four distinct cues, 15% token coverage and 75% of
    all language votes. Unknown/mixed/short text keeps the old English default.
    These conservative policy floors are not calibrated accuracy claims.
    """
    if "RAVEN_EARS_LANG" in os.environ:
        return os.environ["RAVEN_EARS_LANG"], "explicit RAVEN_EARS_LANG"
    lines, source = _LYRIC_TEXT.get()
    # Translations and stage directions are display text, not language evidence.
    text = " ".join(lines)
    while re.search(r"\([^()]*\)|\[[^\[\]]*\]", text):
        text = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", text)
    tokens = _text_tokens(text)
    # A cue shared by two dictionaries is ambiguous, even if added later.
    cues = {lang: words - set().union(*(v for k, v in _LANG_WORDS.items() if k != lang))
            for lang, words in _LANG_WORDS.items()}
    votes = {lang: [t for t in tokens if t in words] for lang, words in cues.items()}
    lang = max(votes, key=lambda k: len(votes[k]))
    count, total = len(votes[lang]), sum(len(v) for v in votes.values())
    if (count >= 8 and len(set(votes[lang])) >= 4 and count >= 0.15 * len(tokens)
            and count >= 0.75 * total):
        return lang, f"{source}, dominant whole-song text cues ({count} of {total} language votes)"
    return "en", (f"{source} language uncertain; keeping English default" if tokens
                  else "no lyric-text language evidence; keeping English default")


def effective_language():
    return language_choice()[0]


def quiet_phantom(text, quiet):
    """Only known filler over a measured quiet stem; supplied lyrics veto it."""
    # The showdown's programme sign-off is absent from the old phrase list.
    # Add it only here; a loud sung sign-off must not acquire a new tail gate.
    if not quiet or not (is_hallucination(text) or _norm(text) == "we'll be right back"):
        return False
    phrase = _text_tokens(text)
    reference = _text_tokens(" ".join(_LYRIC_TEXT.get()[0]))
    return not phrase or not any(reference[i:i + len(phrase)] == phrase
                                 for i in range(len(reference) - len(phrase) + 1))


def effective_model():
    return os.environ.get("RAVEN_LYRICS_WHISPER", "large-v3")


def _model_attempts(E):
    if E._pick_device(need_mb=4500) == "cuda":
        return [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]
    if E._pick_device(need_mb=1800) == "cuda":
        return [("cuda", "int8_float16"), ("cpu", "int8")]
    return [("cpu", "int8")]


def _transcription_note(device, compute_type):
    return f"lyrics: whisper {effective_model()} on {device} ({compute_type}), no VAD"


def cache_file(path):
    import raven_ears_plus as E
    os.makedirs(CACHE, exist_ok=True)
    tag = names_tag()
    language = hashlib.sha256(effective_language().encode("utf-8")).hexdigest()[:12]
    lines = _LYRIC_TEXT.get()[0]
    text = "-text-" + hashlib.sha256(" ".join(_text_tokens(" ".join(lines))).encode()).hexdigest()[:12] if lines else ""
    device, compute_type = _model_attempts(E)[0]
    model = hashlib.sha256(effective_model().encode("utf-8")).hexdigest()[:12]
    return os.path.join(CACHE, file_sha256(path) + ("-" + tag if tag else "") + "-lang-" + language
                        + text + f"-model-{model}-{device}-{compute_type}.json")


def _level_fn(y, sr):
    """dB of any time span relative to the song's loud level (95th percentile of 0.25 s frames)."""
    hop = int(0.25 * sr)
    n = len(y) // hop
    rms = np.sqrt((y[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    ref = float(np.percentile(db, 95)) if n else 0.0

    def rel(a, b):
        i, j = int(a * sr / hop), max(int(b * sr / hop), int(a * sr / hop) + 1)
        seg = db[i:j]
        return float(seg.mean() - ref) if len(seg) else -99.0
    return rel


def drop_quiet_intro(words, vrel, dur, notes):
    """Remove an isolated leading guess over deep stem silence.

    Leave quiet words alone unless later timed words have vocal support and
    there is still a quiet margin after the guess. The segment gate protects
    soft singing elsewhere, including words with weak or missing evidence.
    """
    if vrel is None or not words:
        return words
    first = next((i for i, (_, a, b) in enumerate(words)
                  if vrel(max(0.0, a - 0.5), min(dur, b + 0.5)) >= -VOCAL_GONE_DB), len(words))
    if first == len(words):
        return words
    cut = 0
    for _, a, b in words[:first]:
        stop = b + LINE_SLACK
        if stop > words[first][1] or any(vrel(t, t + 0.5) >= -VOCAL_GONE_DB for t in np.arange(0.0, stop, 0.5)):
            break
        cut += 1
    if cut:
        notes.append(f"lyrics: dropped {cut} introductory Whisper word(s) over a quiet vocal stem")
    return words[cut:]


def _cached_intro(words, path, notes):
    """Apply the word guard to old cache hits without re-running separation."""
    if len(words) < 2:
        return words
    try:
        import singer
        stem = os.path.join(singer.CACHE, singer.file_hash(path), "vocals.wav")
        if not os.path.isfile(stem):
            return words
        y, sr = singer.load_mono(stem)
        return drop_quiet_intro(words, _level_fn(y, sr), len(y) / sr, notes)
    except (OSError, RuntimeError, ValueError):
        return words


def drop_tail_hallucination(words, dur):
    """Word-level backstop: strip a trailing 'thank you ...' run inside the last TAIL_S seconds."""
    if not words:
        return words
    out = list(words)
    while out:
        tail = [w for w in out if w[1] >= dur - TAIL_S]
        if not tail:
            break
        txt = " ".join(w[0] for w in tail)
        if is_hallucination(txt):
            out = [w for w in out if w[1] < dur - TAIL_S]
            continue
        # trailing 1-4 words that form a hallucination phrase
        cut = None
        for n in (4, 3, 2, 1):
            if len(out) >= n and out[-n][1] >= dur - TAIL_S and is_hallucination(" ".join(w[0] for w in out[-n:])):
                cut = n
        if cut:
            out = out[:-cut]
            continue
        break
    return out


def _load_model(E, notes):
    from faster_whisper import WhisperModel
    size = effective_model()
    for d, ct in _model_attempts(E):
        try:
            return WhisperModel(size, device=d, compute_type=ct, local_files_only=offline()), d, ct
        except Exception as e:
            notes.append(f"LYRICS WARNING: {size} on {d} FAILED to load ({type(e).__name__}: {str(e)[:70]})")
    return None, None, None


def transcribe(path, notes):
    import raven_ears_plus as E
    E.release_models()
    E._preload_cuda12()
    y16 = E.load_audio(path, E.SR)
    sr = E.SR
    dur = len(y16) / sr
    rel = _level_fn(y16, sr)
    lang = effective_language()
    model, d, ct = _load_model(E, notes)
    if model is None:
        notes.append(f"LYRICS FAILED: {effective_model()} could not run; NO LYRICS PRODUCED, this is NOT a wordless song (no smaller-model fallback by design)")
        return None
    names = effective_names()
    dropped = []
    vrel = None
    try:                                        # the Demucs vocals stem (cached by singer.py) says where anyone is singing
        import singer
        sd, _ = singer.get_stems(path, {})
        yv, srv = singer.load_mono(os.path.join(sd, "vocals.wav"), sr)
        vrel = _level_fn(yv, srv)
        del yv
        E.release_models()
    except Exception as e:
        notes.append(f"LYRICS WARNING: vocals stem unavailable ({type(e).__name__}: {str(e)[:60]}); only the credit blocklist guards against invented lines")
    reasons = {"credit": [], "no vocals": [], "quiet filler": [], "unsure": []}
    unsure = []

    def keep_seg(sg, off=0.0):
        """Segment gate: invented credits, no singer in the vocals stem, then the old tail/silence rule."""
        a, b = sg.start + off, sg.end + off
        if names_echo(sg.text, names, doubtful=sg.avg_logprob < -0.6 or sg.no_speech_prob > 0.3):
            reasons["credit"].append((a, "names prompt echoed"))
            return False
        if is_credit(sg.text):
            reasons["credit"].append((a, sg.text.strip()))
            return False
        doubtful = sg.avg_logprob < -1.0 or sg.no_speech_prob > 0.5
        if vrel is not None and b > a:
            quiet = vrel(max(0.0, a - 0.5), min(b + 0.5, dur)) < -VOCAL_GONE_DB   # padded: stem timing slop at line edges
            if quiet_phantom(sg.text, quiet):
                reasons["quiet filler"].append((a, sg.text.strip()))
                return False
            if quiet and (sg.avg_logprob < -0.8 or sg.no_speech_prob > 0.4):
                reasons["no vocals"].append((a, sg.text.strip()))                    # stem quiet AND Whisper unconfident
                return False
            if quiet:                                                                # soft singing: keep, say so
                unsure.append((a, b, "vocals quiet here"))
                reasons["unsure"].append((a, sg.text.strip()))
            elif doubtful:
                unsure.append((a, b, "Whisper doubtful"))
                reasons["unsure"].append((a, sg.text.strip()))
        if is_hallucination(sg.text) and (a >= dur - TAIL_S or rel(a, min(b, dur)) < -SILENCE_DB):
            dropped.append((round(a, 1), sg.text.strip()))
            return False
        return True

    nk = {}
    try:
        nk = whisper_name_kwargs(model, names, mode=names_mode("prompt"), notes=notes)
        if nk:
            notes.append(whisper_names_note(len(names)))
        segs, _ = model.transcribe(y16, language=lang, word_timestamps=True, vad_filter=False,
                                   condition_on_previous_text=False, no_speech_threshold=0.6, **nk)
        words = [(w.word.strip(), float(w.start), float(w.end))
                 for sg in segs if sg.no_speech_prob < 0.8 and keep_seg(sg) for w in (sg.words or []) if w.word.strip()]
        words = drop_tail_hallucination(sorted(words, key=lambda w: w[1]), dur)
        notes.append(_transcription_note(d, ct))

        # ---- gap re-listen (same model, short windows)
        gaps, prev = [], 0.0
        for w, a, b in words:
            if a - prev >= 9.0:
                gaps.append((prev, a))
            prev = max(prev, b)
        if dur - prev >= 9.0:
            gaps.append((prev, dur))
        new = []
        for a, b in gaps:
            t = a
            while t < b - 1.0:
                u = min(b, t + 25.0)
                lo, hi = max(0.0, t - 1.0), min(dur, u + 1.0)
                sg_iter, _ = model.transcribe(y16[int(lo * sr):int(hi * sr)], language=lang, word_timestamps=True,
                                              vad_filter=False, condition_on_previous_text=False, no_speech_threshold=0.9, **nk)
                for sg in sg_iter:
                    if sg.no_speech_prob >= 0.6 or sg.avg_logprob < -1.0 or sg.compression_ratio > 2.4:
                        continue
                    low = sg.text.lower()
                    if any(h in low for h in ("thank you", "thanks for", "watching", "subtitles", "amara", "subscribe")):
                        continue
                    if not keep_seg(sg, lo):
                        continue
                    for w in sg.words or []:
                        ws = float(w.start) + lo
                        if t - 0.2 <= ws <= u and w.word.strip():
                            new.append((w.word.strip(), ws, float(w.end) + lo))
                t = u
        clean, run = [], 0                     # drop stuttering hallucination runs
        for i, x in enumerate(new):
            run = run + 1 if i and x[0].lower() == new[i - 1][0].lower() else 1
            if run <= 6:
                clean.append(x)
        if clean:
            notes.append(f"lyrics: re-listened to {len(gaps)} long gap(s) in short windows and recovered {len(clean)} words the one-pass transcript missed")
        words = drop_tail_hallucination(sorted(words + clean, key=lambda w: w[1]), dur)
        def _t(x):
            return f"{int(x // 60)}:{int(x % 60):02d}"
        for k, lab in (("credit", "invented credit/filler line(s)"),
                       ("quiet filler", "known filler line(s) over a measured quiet vocals stem"),
                       ("no vocals", "line(s) over a quiet vocals stem that Whisper was also unsure of")):
            if reasons[k]:
                notes.append(f"lyrics: dropped {len(reasons[k])} {lab} (Whisper hallucination): "
                             + "; ".join(f"{_t(t)} '{x}'" for t, x in reasons[k][:4]))
        if unsure:
            notes.append(f"lyrics: {len(unsure)} line(s) kept but marked (unsure), quiet vocals or doubtful Whisper: "
                         + "; ".join(f"{_t(t)} '{x}'" for t, x in reasons["unsure"][:4]))
        UNSURE[:] = unsure
        if dropped:
            notes.append("lyrics: dropped end-of-track/silence hallucination: " + "; ".join(f"{t}s '{x}'" for t, x in dropped[:4]))
        return drop_quiet_intro(words, vrel, dur, notes) or None
    except Exception as e:
        detail = "" if nk else f": {str(e)[:70]}"
        notes.append(f"LYRICS FAILED: transcription failed, NO LYRICS PRODUCED ({type(e).__name__}{detail})")
        return None
    finally:
        del model
        gc.collect()
        E.release_models()


def get_words(path, notes=None, use_cache=True):
    """Cached by audio bytes, names, language, model and intended device/precision."""
    import raven_ears_plus as E
    notes = notes if notes is not None else []
    language, why = language_choice()
    notes.append(f"lyrics: language {language} ({why})")
    cf = cache_file(path)
    expected = _transcription_note(*_model_attempts(E)[0])
    if use_cache and os.path.exists(cf):
        try:
            j = json.load(open(cf))
            if j.get("v") == VERSION:
                UNSURE[:] = [tuple(x) for x in j.get("unsure", [])]
                for n in j.get("notes", []):
                    notes.append(names_note_text(n) + " [shared cache]")
                return _cached_intro([tuple(w) for w in j["words"]], path, notes) or None
        except Exception:
            pass
    local = []
    words = transcribe(path, local)
    notes.extend(local)
    ok = words is not None or not any("could not run" in n or "failed" in n.lower() for n in local)
    # A fallback must not masquerade as the requested GPU/precision on the next run.
    if use_cache and ok and expected in local:  # never cache a failed run as "no lyrics"
        tmp = cf + ".tmp"
        json.dump({"v": VERSION, "words": [list(w) for w in (words or [])], "notes": local,
                   "unsure": [list(x) for x in UNSURE]}, open(tmp, "w"))
        os.replace(tmp, cf)
    return words
