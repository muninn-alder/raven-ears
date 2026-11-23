"""Derive saved notation from listen-once captures; render without audio engines."""
from bisect import bisect_right
from collections import Counter
import difflib
import math
import re
import textwrap

from lyrics_shared import is_vocable, lyric_excerpt


LEGACY = "melody: not in this saved song; --fresh to add\n"
LENGTHS = "% lengths: proportional display only; note attacks and rhythm are not notated"
FULL_GUIDE = (LENGTHS + "\n% unmarked = short, 2 = longer, 4 = longest · z = a gap in the pitch track\n"
              "% w: * leaves a note without a new word, _ continues a word, ~ joins words on one note\n"
              "% ≈ marks approximate line timing or words spread across note onsets")
# Four rendered notes per row give a short contour; one to three can be stray
# stem pitches rather than a usable hook (synced lyrics, 10 Nov 2026).
MIN_HOOK_NOTES = 4


def rhythm_line(music, singer_meter=None):
    tempo = music.get("tempo") or {}
    # Quote the same tempo the card's TEMPO line leads with: when the beat
    # tracker out-votes librosa (tempo_trust, #10), say its number, not librosa's
    # (I Miss You, 10 Nov 2026: 143.6 here vs 111.1 on TEMPO).
    import tempo_trust
    reading = tempo_trust.lead_reading(tempo.get("bpm"), singer_meter)
    if reading:
        return (f"rhythm: not notated; the card measures {reading['bpm']:g} BPM, {reading['meter']}, "
                f"bars steady {reading['consistency']:.2f} (beat tracker)"
                + (" · " + reading["source"] if reading["source"] else ""))
    if tempo.get("bpm") is None:
        return "rhythm: not notated; the music card has no tempo measurement"
    text = f"rhythm: not notated; the card measures {tempo['bpm']:g} BPM"
    if tempo.get("regularity") is not None:
        text += f", steady {tempo['regularity']:.2f}"
    return text


# Words that decorate an audio filename but are not part of a song's title.
NOISE = {"official", "lyric", "lyrics", "video", "audio", "music", "visualizer", "hd", "hq",
         "remastered", "remaster", "version", "taylors", "live", "explicit", "mv", "ft", "feat",
         "featuring", "full", "song", "karaoke", "instrumental"}
# A single short or common word is too weak to prove a title match.
COMMON = {"this", "that", "with", "have", "from", "your", "what", "when", "will", "love", "they",
          "there", "just", "like", "been", "were", "them", "then", "than", "into", "over", "more"}


def words_of(text):
    return re.findall(r"\w+", re.sub(r"['\u2019]", "", text.casefold().replace("_", " ")))


def title_words(options):
    """Title for the hook: metadata title if present, else the audio filename stem
    with artist and noise words and punctuation stripped. Empty list when unknown."""
    options = options or {}
    if options.get("title"):
        return words_of(options["title"])
    name = options.get("metadata_filename")
    if not name:
        return []
    stem = re.sub(r"[\[(][^\])]*[\])]", " ", name.rsplit(".", 1)[0])
    skip = NOISE | set(words_of(options.get("artist") or ""))
    return [w for w in words_of(stem) if w not in skip and not w.isdigit()]


def title_run(row, title):
    """Longest run of the title's words, in title order, inside one lyric row; else []."""
    padded = " " + " ".join(title) + " "
    best = []
    for i in range(len(row)):
        for j in range(i + len(best) + 1, len(row) + 1):
            if " " + " ".join(row[i:j]) + " " in padded:
                best = row[i:j]
    if len(best) == 1 and (len(best[0]) < 4 or best[0] in COMMON):
        return []
    return best


def phrase_word_boundaries(words, duration, pause_s=0.35, max_s=7.0):
    """Pause cuts, then largest-gap cuts to bound phrases; never split a word.

    Overlapping words form one indivisible group. A word/group longer than the
    limit stays whole; hook selection can decline it rather than truncate it.
    """
    groups = []
    for _, start, end in words:
        start, end = float(start), float(end)
        if groups and start < groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    pauses = [i for i in range(1, len(groups))
              if groups[i][0] - groups[i - 1][1] >= pause_s - 1e-9]
    splits = set(pauses)
    for lo, hi in zip([0] + pauses, pauses + [len(groups)]):
        while hi > lo + 1 and groups[hi - 1][1] - groups[lo][0] > max_s + 1e-9:
            candidates = [i for i in range(lo + 1, hi)
                          if groups[i - 1][1] - groups[lo][0] <= max_s + 1e-9]
            # Ties go to the latest gap to avoid tiny rows on continuous words.
            cut = max(candidates, key=lambda i: (
                round(groups[i][0] - groups[i - 1][1], 9), i)) if candidates else lo + 1
            splits.add(cut)
            lo = cut
    return [t for i in sorted(splits)
            if 0 < (t := (groups[i - 1][1] + groups[i][0]) / 2) < duration]


def hook_indices(texts, lines, peak, title=None, *, max_span=None, min_notes=None):
    # Writer files can contain [Chorus]/[Verse] labels. Repetition of a heading
    # is not repetition of a sung line (seen on the real Uhuru check).
    normalized = ["" if re.fullmatch(r"\s*\[[^]]+\]\s*", text) else
                  " ".join(words_of(text)) for text in texts]
    eligible = [i for i, text in enumerate(normalized)
                if text and (words_of(lines[i].get("lyrics", text)) or
                             (lines[i].get("lyric_fit") == "unaligned" and lines[i]["note_indices"]))]
    if not eligible:
        return [], "no sung lyric words paired with rendered notes"
    counts = Counter(normalized[i] for i in eligible)
    repeated = [i for i in eligible if counts[normalized[i]] > 1]
    # Title preference: the row with the longest run of the song's title words
    # (ties favour rendered word timing, then first in song), then the row after it.
    # A row that is only the title may be a writer's heading, so it does not count.
    matches = []
    for i in eligible if (title and texts) else []:
        # A repeated backing response in parentheses does not make every verse
        # a title hook (Feeling This); let the repeated chorus compete instead.
        row = words_of(re.sub(r"\([^)]*\)", "", texts[i]))
        run = title_run(row, title)
        if run and len(run) < len(row):
            matches.append((len(run), i, run))
    # When some row sings several title words, a row with only one of them is not the
    # title: "never" in "I am your demon, never leaving" is not SAMURAI's "Never Fade
    # Away" (11 Nov 2026). A title known only from a noisy file name ("kael-uhuru")
    # never matches more than one word, so its single word still counts.
    if any(n > 1 for n, _, _ in matches):
        matches = [m for m in matches if m[0] > 1]
    title_runs = {i: run for _, i, run in matches}
    candidates = [(i, f'line with the title "{" ".join(run)}" and the line after it', max_span)
                  for _, i, run in sorted(matches, key=lambda m: (
                      -m[0], not lines[m[1]].get("has_word_timing", True), m[1]))]
    if repeated:
        rule = "most-repeated lyric line and the line after it (ties: first in song)"
        candidates += [(i, rule, max_span) for i in sorted(repeated, key=lambda i: (-counts[normalized[i]], i))]
    peak_reason = "no repeated lyric line or measured intense moment available"
    if peak.get("ok") and eligible:
        moment = (peak["start"] + peak["end"]) / 2
        containing = [i for i in eligible if lines[i]["start"] <= moment <= lines[i]["end"]]
        peak_reason = "no sung row contains the measured intense moment"
        if containing:
            rule = "line at the singer's most intense moment and the line after it"
            candidates.append((containing[0], rule, 8.0))
    sparse = False
    for first, rule, span in candidates:
        related = [i for i in eligible if normalized[i] == normalized[first] or (
            first in title_runs and title_runs.get(i) == title_runs[first]
            and difflib.SequenceMatcher(None, normalized[first].split(),
                                        normalized[i].split(), autojunk=False).ratio() >= .85)]
        if len(related) > 1:
            # Use the first singable repetition, including small variants of a
            # title refrain. Otherwise keep the original candidate and caveat.
            aligned = next((i for i in related if lines[i].get("lyric_fit") != "unaligned"
                            and (min_notes is None or len(lines[i]["note_indices"]) >= min_notes)
                            and (span is None or lines[i]["end"] - lines[i]["start"] <= span)), first)
            if aligned != first:
                rule = rule.replace("ties: first in song", "first aligned occurrence")
                first = aligned
        if span is not None and lines[first]["end"] - lines[first]["start"] > span:
            continue
        if min_notes is not None and len(lines[first]["note_indices"]) < min_notes:
            sparse = True
            continue
        chosen = [i for i, text in enumerate(normalized) if i >= first and text][:2]
        omission = ""
        if len(chosen) == 2:
            if is_vocable(texts[chosen[1]]) and (chosen[1] not in eligible or
                    (min_notes is not None and len(lines[chosen[1]]["note_indices"]) < min_notes)):
                omission = f'; next line is wordless ("{lyric_excerpt(texts[chosen[1]])}")'
                chosen = [first]
            elif min_notes is not None and len(lines[chosen[1]]["note_indices"]) < min_notes:
                omission = f"; next line has fewer than {min_notes} rendered notes"
                chosen = [first]
            elif chosen[1] not in eligible:
                omission = "; next line has no rendered lyric words"
                chosen = [first]
            elif span is not None and lines[chosen[1]]["end"] - lines[first]["start"] > span:
                chosen = [first]
        if len(chosen) == 1:
            rule = rule.replace(" and the line after it", "")
        return chosen, rule + omission
    if sparse:
        return [], f"no title, repetition or intense moment candidate has at least {min_notes} rendered notes per line"
    return [], "selected sung row exceeds 8 s" if candidates else peak_reason


def writer_word_text(texts, words, cuts):
    """{word index: text} showing supplied lyrics where Whisper's words were timed.

    Words are paired inside each lyric line (difflib on letters only). A Whisper
    word the writer's line lacks shows nothing; writer words with no Whisper word
    ride on a neighbouring heard word, so the line is whole and every note keeps
    the timing Whisper measured."""
    def norm(word):
        return "".join(re.findall(r"\w+", word.casefold()))

    heard = {}
    for i, (_, start, _) in enumerate(words):
        heard.setdefault(bisect_right(cuts, start), []).append(i)
    labels = {}
    for line, indices in heard.items():
        if line >= len(texts):
            continue
        said = []
        for token in texts[line].split():
            if norm(token) or not said:
                said.append(token)
            else:
                said[-1] += " " + token
        a, b = [norm(words[i][0]) for i in indices], [norm(t) for t in said]
        slots = {i: [] for i in indices}
        carry = []
        for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            mine, theirs = indices[a0:a1], said[b0:b1]
            if tag == "insert":
                (slots[indices[a0 - 1]] if a0 else carry).extend(said[b0:b1])
            elif mine:
                # equal or replace: pair in order; extra writer words ride on the last one
                for k, i in enumerate(mine):
                    slots[i].extend(theirs[k:] if k == len(mine) - 1 else theirs[k:k + 1])
            if carry and mine:
                slots[mine[0]][:0] = carry
                carry = []
        if carry and indices:
            slots[indices[0]][:0] = carry
        labels.update({i: " ".join(parts) for i, parts in slots.items()})
    return labels


def wrap_comment(text, width):
    return textwrap.wrap(text, width=width, subsequent_indent="% ", break_long_words=False,
                         break_on_hyphens=False) or [text]


def split_rows(notes, syllables, width):
    """Wrap pitches and keep any w: row paired with the corresponding pitches."""
    if not syllables:
        return textwrap.wrap(" ".join(notes), width=width, break_long_words=False)
    if len(notes) != len(syllables):
        return [" ".join(notes), "w: " + " ".join(syllables)]
    rows, a, b = [], [], []
    for note, syllable in zip(notes, syllables):
        if a and max(len(" ".join(a + [note])), 3 + len(" ".join(b + [syllable]))) > width:
            rows += [" ".join(a), "w: " + " ".join(b)]
            a, b = [], []
        a.append(note)
        b.append(syllable)
    return rows + [" ".join(a), "w: " + " ".join(b)]


def _full_line_spans(placements, duration, singer_card, singer_status):
    from feeling_card import stem_rows
    from lyrics_shared import vocal_spans

    stems = stem_rows(singer_card["text"], singer_card["json"], singer_status)
    voiced = vocal_spans(stems, duration) or []
    starts = [row["t"] for row in placements]
    spans = {}
    for i, (start, end) in enumerate(zip(starts, starts[1:] + [duration])):
        # A trailing stem dropout can shorten the row. The next lyric start
        # is a hard ceiling, regardless of a Whisper word's measured end.
        stop = next((b for a, b in voiced if a - .5 <= start < b), end)
        spans[i] = (start, min(end, stop))
    return spans


def build(song, *, full=False):
    """One CPU notation call, after singer and music; no separation, F0 or ASR."""
    def not_run(reason):
        return {"status": "not run", "reason": reason, "text": "", "json": None}

    passes = song["manifest"]["passes"]
    if passes["singer"]["status"] != "ran":
        return not_run("vocal stem/pitch and timed words unavailable: singer " + passes["singer"]["status"])
    singer_card = song["cards"]["singer"]
    singer = singer_card["json"] or {}
    lyric_card = singer.get("lyrics") or {}
    if lyric_card.get("quick_skipped"):
        return not_run("skipped in a quick listen")
    inputs = singer.get("melody_input") or {}
    if not inputs.get("f0"):
        return not_run("vocal-stem pitch track unavailable")
    # Whisper can time a word with zero length (start == end); the notation
    # refuses those, which failed a whole real song (Feeling This, 10 Nov 2026).
    # Drop them once, here, so every step below shares the same word indices.
    def timed(w):
        try:
            return math.isfinite(float(w[1])) and math.isfinite(float(w[2])) and 0 <= float(w[1]) < float(w[2])
        except (TypeError, ValueError, IndexError):
            return False
    captured_words = inputs.get("words") or []
    words = [w for w in captured_words if timed(w)]
    dropped = len(captured_words) - len(words)
    texts = inputs.get("lyric_lines") or []
    # Singer captures also store Whisper segments here when no lyrics exist.
    if "words are Whisper's guesses" in inputs.get("lyrics_label", ""):
        texts = []
    saved_rows = lyric_card.get("rows") or []
    alignment = None
    if (texts and texts == [row["text"] for row in saved_rows] and
            (not words or any(row.get("timing") == "LRCLIB supplied" for row in saved_rows))):
        alignment = saved_rows, lyric_card.get("summary") or {}
    if not words and alignment is None:
        return not_run("timed words unavailable" + (" (none with a usable duration)" if dropped else ""))
    if passes["music"]["status"] != "ran":
        return not_run("music card unavailable; cannot quote its key and rhythm measurements")
    import keyfuse
    music = song["cards"]["music"]["json"] or {}
    claim = music.get("key_claim") or {}
    key = claim.get("key") or claim.get("working") or (music.get("key") or {}).get("key")
    if not key:
        return not_run("music card has no key candidate")
    unclear = (claim.get("state") not in ("clear", "chords", "agree", "spectral") or bool(re.search(
        r"unclear|unsure|between|working|\blean\b", claim.get("line", ""), re.I)))
    duration = len(inputs["f0"]) / inputs["fps"]
    if texts and alignment is None:
        import lyric_align
        options = song["manifest"].get("options") or {}
        rows, summary = lyric_align.align_lyrics(texts, words, duration, title=options.get("title"),
                                                 filename=options.get("metadata_filename"))
        if len(rows) != len(texts) and inputs.get("lyrics_label", "").startswith("LYRICS: writer's text"):
            inputs = dict(inputs, lyrics_label=lyric_align.summary_line(summary))
        texts = [r["text"] for r in rows]
        if full:
            alignment = rows, summary
    try:
        import melody_abc as abc
        # Saved older cards may use the pre-signature spelling of the tonic.
        display_key = "none" if claim.get("blues_rock") else keyfuse.spelling_key(key)
        if display_key == "none":
            key = display_key
        elif display_key and display_key.endswith("Mix"):
            key = display_key.removesuffix("Mix") + " mixolydian"
        elif display_key:
            tonic = display_key.removesuffix("m")
            key = tonic + (" minor" if display_key.endswith("m") else " major")
        else:
            display_key = abc._key(key)
        home, _, spans = keyfuse.song_spelling_keys(song)
        spelling_at = keyfuse.spelling_context(home or display_key, spans)
        kwargs = dict(abc.PROFILES["middle"], key=key, mode="proportional", spelling_at=spelling_at)
        if "voiced_prob" in inputs:
            kwargs["voiced_prob"] = inputs["voiced_prob"]
        voice = singer.get("singer") or {}
        voice_data = voice.get("data", {}) if isinstance(voice, dict) else {}
        if full:
            kwargs["spread_timed_words"] = True
            low_hz = (voice_data.get("pitch") or {}).get("lo_hz")
            if low_hz and low_hz > 0:
                kwargs["range_low"] = 69 + 12 * math.log2(low_hz / 440)
        placements, summary, unplaced = None, None, ""
        if texts:
            try:
                cuts, placements, summary = abc.lyric_line_boundaries(
                    texts, words, duration, alignment=alignment, keep_starts=full)
            except ValueError as error:
                # Many short repeated lines a second or two apart can defeat
                # placement (I Miss You, 10 Nov 2026). Keep the tune and disclose
                # the failed placement; use word phrases as for absent lyrics.
                unplaced, texts = str(error), []
        if texts:
            kwargs["line_breaks"] = cuts
            kwargs["line_text"] = texts
            if full:
                kwargs["line_spans"] = _full_line_spans(placements, duration, singer_card, passes["singer"])
                placements = [dict(row, end=kwargs["line_spans"][i][1]) for i, row in enumerate(placements)]
            else:
                kwargs["word_text"] = writer_word_text(texts, words, cuts)
                kwargs["line_spans"] = {i: (row["t"], row["end"])
                                        for i, row in enumerate(placements) if row["approx"]}
        else:
            kwargs["line_breaks"] = phrase_word_boundaries(words, duration)
        text, data = abc.transcribe(inputs["f0"], inputs["fps"], words, **kwargs)
        data["notation"]["line_lyrics"] = (
            "matching word intervals on rendered notes, otherwise approximate onset order" if full else
            "approximate rows use note onsets when word timing is unavailable")
        data["notation"]["pitch_revision"] = PITCH_REVISION
        data["notation"]["local_spelling"] = spans
        if placements is not None:
            data["lyric_alignment"] = {"rows": placements, "summary": summary}
            for line, placement in zip(data["lines"], placements):
                line["lyric_placement"] = placement
        if dropped:
            data["words_dropped_zero_length"] = dropped
        if unplaced:
            data["lyric_lines_unplaced"] = unplaced
        data["profile"] = "middle"
        data["key_unclear"] = unclear
        data["lyrics_label"] = inputs.get("lyrics_label", "lyric provenance unavailable")
        rhythm = rhythm_line(music, singer.get("meter"))
        data["rhythm_line"] = rhythm
        header = f"MELODY: approximate vocal ABC · K:{display_key} · middle"
        if unclear:
            header += " (key unclear)"
        if unplaced:
            header += " · word phrases (lyric lines could not be placed)"
        data["header"] = header
        text = "\n".join((FULL_GUIDE if full else LENGTHS) if row.startswith("% rhythm:") else row
                         for row in text.splitlines()) + "\n"
        melody_text = header + "\n" + rhythm + "\n% " + data["lyrics_label"] + "\n" + text
        if full:
            return {"status": "ran", "reason": "", "text": melody_text, "json": data}
        peak = voice_data.get("peak") or {}
        # Slice bounds include intros, interludes and outros. Hook windows use
        # the words AND rendered notes actually shown, not that silent padding.
        hook_lines, heard = [], [[] for _ in data["lines"]]
        shown = kwargs.get("word_text", {})
        for word in data["words"]:
            heard[bisect_right(kwargs["line_breaks"], word["start"])].append(word)
        for line, row in zip(data["lines"], heard):
            notes = [data["notes"][j] for j in line["note_indices"]]
            events = row + notes
            has_word_timing = not line.get("lyric_fit") or any(
                shown.get(wi) for note in notes for wi in note["word_indices"])
            hook_lines.append(dict(line, has_word_timing=has_word_timing,
                                   start=min((e["start"] for e in events), default=line["start"]),
                                   end=max((e["end"] for e in events), default=line["end"])))
        hook_texts = texts or [" ".join(w["word"] for w in row) for row in heard]
        indices, rule = hook_indices(hook_texts, hook_lines, peak,
                                       title_words(song["manifest"].get("options")),
                                       max_span=None if texts else 8.0, min_notes=MIN_HOOK_NOTES)
        data["hook"] = {"line_indices": indices, "rule": rule}
        # Re-spell only the selected notes: accidentals from earlier full-song
        # rows must not silently change a standalone hook's pitches. Pitches and
        # words only: no lengths or rests. Every line stays within the card width.
        from stage_status import NOTES_WIDTH
        state = {}
        hook = [header] + textwrap.wrap("HOOK: " + rule, width=NOTES_WIDTH, subsequent_indent="  ") + [
            rhythm, "% pitches and words only; lengths and rests are in --view melody",
            f"K:{display_key}"]
        for i in indices:
            line = data["lines"][i]
            placement = line.get("lyric_placement") or {}
            approx = "≈" if placement.get("approx") else ""
            stamp = int(placement["t"] if approx else line["written_origin_s"])
            unaligned = line.get("lyric_fit") == "unaligned"
            label = abc.UNALIGNED_LYRICS if unaligned else ""
            hook += wrap_comment(f"% [{stamp // 60}:{stamp % 60:02d}{approx}] {label}{hook_texts[i]}", NOTES_WIDTH)
            for omission in line.get("low_note_comments", []) + line.get("high_note_comments", []):
                hook += wrap_comment(omission, NOTES_WIDTH)
            notes = iter(data["notes"][j] for j in line["note_indices"])
            def spell(note):
                return abc._pitch(note["midi"], display_key, state, spelling_at(note["start"]))
            pitches = [spell(next(notes)) for _ in re.finditer(r"[=^_]?[A-Ga-g][,']*", line["music"])]
            if pitches:
                hook += split_rows(pitches, line["lyrics"].split(), NOTES_WIDTH)
            else:
                hook += ["% no stable notes", "w: " + line["lyrics"]]
            missing = [w for w in (shown.get(j, data["words"][j]["word"])
                                   for j in line["unaligned_word_indices"]) if w]
            if missing and not line.get("lyric_fit"):
                hook += wrap_comment("% words without a rendered note: " + ", ".join(missing), NOTES_WIDTH)
        data["hook_text"] = "\n".join(hook) + "\n" if indices else "no hook: " + rule + "\n"
        return {"status": "ran", "reason": "", "text": melody_text, "json": data}
    except Exception as error:
        return {"status": "failed", "reason": f"{type(error).__name__}: {error}", "text": "", "json": None}



HUM_ALONG = ("HUM ALONG: follow the ups and downs under the words "
             "(comma = lower, lowercase = higher, ^ sharp, _ flat)")
PITCH_REVISION = 2


def needs_lyric_refresh(song):
    """Old derived notation can be replaced in memory using the saved captures."""
    melody = song.get("melody") or {}
    data = melody.get("json") or {}
    if not melody or "line_lyrics" in (data.get("notation") or {}):
        return False
    singer = song["cards"].get("singer", {}).get("json") or {}
    inputs = singer.get("melody_input") or {}
    lyrics = singer.get("lyrics") or {}
    if not inputs.get("f0") or not inputs.get("lyric_lines") or lyrics.get("quick_skipped"):
        return False
    rows = (data.get("lyric_alignment") or {}).get("rows") or []
    return any(row.get("approx") for row in rows) or any(
        row.get("approx") for row in lyrics.get("rows") or [])


def needs_pitch_refresh(song):
    melody = song.get("melody") or {}
    data = melody.get("json") or {}
    if melody.get("status") != "ran" or data.get("format") != "abc-2.1-fragment":
        return False
    inputs = (song["cards"]["singer"].get("json") or {}).get("melody_input") or {}
    if not inputs.get("f0"):
        return False
    from keyfuse import song_spelling_keys
    notation = data.get("notation") or {}
    return notation.get("pitch_revision") != PITCH_REVISION or notation.get("local_spelling") != song_spelling_keys(song)[2]


def feeling_hook(text, data, lyrics, title=None):
    """Trim saved hook presentation only; keep measured notes and full ABC intact."""
    from statistics import median
    from stage_status import NOTES_WIDTH

    indices = (data.get('hook') or {}).get('line_indices') or []
    lines, notes = data.get('lines') or [], data.get('notes') or []
    if not indices or not notes or not all(0 <= i < len(lines) for i in indices):
        return text
    selected = [notes[j] for i in indices for j in lines[i]['note_indices']]
    if not selected:
        return text
    hook_rows = text.splitlines()
    pitch_row = r"[=^_]?[A-Ga-g][,']*(?: [=^_]?[A-Ga-g][,']*)*"
    tokens = [token for row in hook_rows if re.fullmatch(pitch_row, row) for token in row.split()]
    if len(tokens) != len(selected):
        return text
    saved_pitches = iter(tokens)
    old_state, new_state = {}, {}
    def retained_pitch(token, keep):
        prefix = token[0] if token[0] in '=^_' else ''
        base = token[len(prefix):]
        if prefix:
            old_state[base] = prefix
        desired = old_state.get(base, '')
        if keep:
            if not prefix and desired != new_state.get(base, ''):
                token = desired + base
            new_state[base] = desired
        return token
    centre = median(n['midi'] for n in selected)
    low = [n['midi'] <= centre - 12 for n in selected]
    # Isolated dips only: a run in a lower register is not a lone tracker dip.
    dips = {id(n) for k, n in enumerate(selected) if low[k]
            and (k == 0 or not low[k - 1]) and (k + 1 == len(low) or not low[k + 1])}
    start, end = min(n['start'] for n in selected), max(n['end'] for n in selected)
    comments = re.findall(r'(?m)^% \[([^]]+)\] ([^\n]*(?:\n% (?!\[|\d+ (?:low|high))[^\n]*)*)', text)
    spans = [(r, lyrics[i + 1].t if i + 1 < len(lyrics) else end)
             for i, r in enumerate(lyrics) if r.t < end and
             (lyrics[i + 1].t if i + 1 < len(lyrics) else end) > start]
    titled = [(r, stop) for r, stop in spans if title and title_run(words_of(r.text), title)]
    def supplied(heard, candidates):
        return max(candidates, key=lambda pair: difflib.SequenceMatcher(
            None, words_of(heard), words_of(pair[0].text), autojunk=False).ratio())[0] if candidates else None
    heard = comments[0][1].replace('\n% ', ' ') if comments else ''
    anchor = supplied(heard, titled)
    # Old word-phrase hooks can straddle several supplied lines. Start at the
    # title's line, retaining at most its successor within the saved window.
    leading = words_of(heard)[:2]
    def has_leading(row):
        words = words_of(row.text)
        return len(leading) == 2 and any(words[j:j + 2] == leading for j in range(len(words) - 1))
    capped = bool(anchor and data.get('lyric_lines_unplaced') and not has_leading(anchor)
                  and any(r.t < anchor.t and has_leading(r) for r, _ in spans))
    if capped:
        first = lyrics.index(anchor)
        start = max(start, anchor.t - .25)
        if first + 2 < len(lyrics):
            end = min(end, lyrics[first + 2].t)

    key_match = re.search(r'(?m)^K:(\S+)', text)
    if not key_match:
        return text
    output = text[:key_match.end()].splitlines()
    lengths = {}
    if (data.get('rhythm') or {}).get('mode') == 'proportional':
        for n in selected:
            a, b = n.get('tick_start'), n.get('tick_end')
            if type(a) is not int or type(b) is not int or b - a not in (1, 2, 4):
                lengths.clear()
                break
            lengths[id(n)] = '' if b - a == 1 else str(b - a)
    length_guide_at = len(output)
    longer = False
    for position, i in enumerate(indices):
        line = lines[i]
        lyric_fit = line.get('lyric_fit')
        syllables = line['lyrics'].split()
        if lyric_fit == 'unaligned':
            syllables = [None] * len(line['note_indices'])
        if len(syllables) != len(line['note_indices']):
            return text  # Unknown legacy pairing: never shift words under notes.
        pairs = []
        for j, syllable in zip(line['note_indices'], syllables):
            n = notes[j]
            inside = start <= n['start'] < end
            if lyric_fit == 'onset_order' and syllable != '*':
                # Keep supplied words attached to their rendered notes.
                dips.discard(id(n))
            pitch = retained_pitch(next(saved_pitches), inside and id(n) not in dips)
            if inside:
                length = lengths.get(id(n), '')
                if length and id(n) not in dips and not longer:
                    output.insert(length_guide_at, '% Approximate lengths · unmarked = short, 2 = longer, 4 = longest')
                    longer = True
                pairs.append((n, syllable, pitch + length))
        if not pairs:
            continue
        stamp, heard = comments[position] if position < len(comments) else ('', '')
        heard = heard.replace('\n% ', ' ')
        candidates = [(r, stop) for r, stop in spans
                      if r.t < line['end'] and stop > line['start']]
        chosen = anchor if capped and position == 0 else supplied(heard, candidates)
        comment = chosen.text if chosen and not lyric_fit else heard
        if start > line['start'] + .5 and chosen:
            clock = int(round(chosen.t))
            stamp = f'{clock // 60}:{clock % 60:02d}' + ('≈' if chosen.approx else '')
        output += wrap_comment(f'% [{stamp}] {comment}', NOTES_WIDTH)
        omissions = line.get('low_note_comments')
        if omissions is None:
            count = sum(n.get('line') == i for n in data.get('low_note_omissions') or [])
            omissions = [f"% {count} low note{'s' if count != 1 else ''} left out: likely tracker octave error"] if count else []
        for omission in omissions + line.get('high_note_comments', []):
            output += wrap_comment(omission, NOTES_WIDTH)
        count = sum(id(n) in dips for n, _, _ in pairs)
        if count:
            output.append(f"% {count} low dip{'s' if count != 1 else ''} left out")
        kept = [(pitch, s) for n, s, pitch in pairs if id(n) not in dips]
        if kept:
            pitches, words = zip(*kept)
            output += split_rows(list(pitches), [] if lyric_fit == 'unaligned' else list(words), NOTES_WIDTH)
    return '\n'.join(output) + '\n'


def render(song, hook=False):
    melody = song.get("melody")
    if melody is None:
        return LEGACY
    if melody["status"] != "ran":
        return f"melody: {melody['status']}: {melody['reason']}\n"
    text = melody["json"]["hook_text"] if hook else melody["text"]
    if hook and "; next line has no rendered lyric words" in text:
        data = melody["json"]
        indices = (data.get("hook") or {}).get("line_indices") or []
        rows = (data.get("lyric_alignment") or {}).get("rows") or []
        texts = [row["text"] for row in rows]
        if not texts:
            inputs = (song.get("cards", {}).get("singer", {}).get("json") or {}).get("melody_input") or {}
            if "words are Whisper's guesses" not in inputs.get("lyrics_label", ""):
                texts = inputs.get("lyric_lines") or []
        if len(indices) == 1 and len(texts) == len(data.get("lines") or []) and indices[0] + 1 < len(texts):
            following = texts[indices[0] + 1]
            if is_vocable(following):
                text = text.replace("; next line has no rendered lyric words",
                                    f'; next line is wordless ("{lyric_excerpt(following)}")')
    import tempo_trust
    cards = song.get("cards") or {}
    meter = tempo_trust.saved_meter(cards.get("singer") or {}, song.get("manifest", {}).get("passes", {}).get("singer"))
    music = cards.get("music", {}).get("json") or {}
    # Ignore the retired text witness in bundles saved during its trial.
    text = re.sub(r"(?m)^(?:% )?HOOK \(Jev meaning\):[^\n]*\n?", "", text)
    # Fox, 16 Nov 2026: a mind seeing ABC thinks "what am I supposed to do with
    # this". One line under the header turns the data into an invitation. It is
    # added at render time, so saved bundles get it without re-listening.
    if HUM_ALONG not in text:
        text = re.sub(r"(?m)^(MELODY: approximate vocal ABC[^\n]*\n)",
                      lambda m: m.group(1) + HUM_ALONG + "\n", text, count=1)
    return re.sub(r"(?m)^rhythm:[^\n]*", lambda _: rhythm_line(music, meter), text)
