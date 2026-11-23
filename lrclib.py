"""LRCLIB words for the song ears. Metadata leaves the house; audio never does.

Lookup and supplied line times stay separate from Whisper's heard word times.
Failures are reported, never cached as an empty answer. No network at import.
"""
from __future__ import annotations

from network_policy import offline, require_online

import hashlib
import json
import math
import os
import re
import socket
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import credits
from slim import NO_VOICE_MARK

API = "https://lrclib.net/api/"
UA = credits.UA
TIMEOUT_S = 6.0
NEG_TTL_S = credits.NEG_TTL_S
CONFIG = {"off": False, "artist": None, "title": None, "album": None}
QUICK = False


class LookupFailed(Exception):
    pass


import stage_status as ST  # HONEST CARD: run-local outcomes.

def _get_json(endpoint, params):
    require_online("LRCLIB lookup")
    req = urllib.request.Request(API + endpoint + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404 and endpoint == "get":
                return None
            retry = e.code == 429 or 500 <= e.code < 600
            why = f"HTTP {e.code}"
        except (TimeoutError, socket.timeout):
            retry, why = True, "timed out"
        except urllib.error.URLError as e:
            retry = True
            why = "timed out" if isinstance(e.reason, (TimeoutError, socket.timeout)) else "unreachable"
        except (ValueError, UnicodeError):
            retry, why = False, "returned invalid JSON"
        except OSError:
            retry, why = True, "unreachable"
        if not retry or attempt:
            raise LookupFailed(why)
        time.sleep(0.5)


def _norm(value):
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def _duration(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def _matches(rec, artist, title, duration, album):
    if not isinstance(rec, dict):
        return False
    if _norm(rec.get("artistName")) != _norm(artist) or _norm(rec.get("trackName")) != _norm(title):
        return False
    # Keep version words. A studio song is not automatically its live performance.
    if album and _norm(rec.get("albumName")) != _norm(album):
        return False
    rd = _duration(rec.get("duration"))
    return duration is None or (rd is not None and abs(rd - duration) <= 2.0)


def parse_lrc(text):
    """LRC -> [{t, text}]. Multiple tags and hundredths/milliseconds supported.

    Blank timed lines mark pauses, not lyrics. Metadata is not sung text.
    Nonzero offset tags fall back to alignment rather than assuming an offset convention.
    """
    if not isinstance(text, str):
        return []
    offsets = re.findall(r"\[offset:\s*([+-]?\d+)\s*\]", text, re.I)
    if offsets and any(int(offset) != 0 for offset in offsets):
        return []
    stamp = re.compile(r"\[(\d+):([0-5]\d(?:\.\d+)?)\]")
    rows = []
    for line in text.splitlines():
        tags = list(stamp.finditer(line))
        if not tags or line[:tags[0].start()].strip():
            continue
        lyric = line[tags[-1].end():].strip()
        if not lyric:
            continue
        for tag in tags:
            rows.append({"t": round(int(tag[1]) * 60 + float(tag[2]), 3), "text": lyric})
    return sorted(rows, key=lambda row: row["t"])


PAIRED_SHARE = 0.3   # share of synced rows that share their timestamp with another row
PAIRED_MIN_ROWS = 8


def _paired(rows):
    """True when the synced text pairs most lines with a second line at the same time:
    LRCLIB uploads that interleave a translation (11 Nov 2026: SAMURAI 'Never Fade
    Away' had a Vietnamese line under every English one, plus credit lines at 0:00).
    A song's own repeats land at different times, so they don't count."""
    if not isinstance(rows, list) or len(rows) < PAIRED_MIN_ROWS:
        return False
    times = {}
    for r in rows:
        if isinstance(r, dict) and isinstance(r.get("t"), (int, float)):
            times[r["t"]] = times.get(r["t"], 0) + 1
    shared = sum(n for n in times.values() if n > 1)
    return shared / len(rows) >= PAIRED_SHARE


def _answer(rec):
    if any(rec.get(key) is not None and not isinstance(rec[key], str) for key in ("plainLyrics", "syncedLyrics")):
        raise LookupFailed("returned invalid lyric fields")
    if rec.get("instrumental") is not None and not isinstance(rec["instrumental"], bool):
        raise LookupFailed("returned an invalid instrumental field")
    synced = parse_lrc(rec.get("syncedLyrics"))
    plain = rec.get("plainLyrics")
    plain = [line.strip() for line in plain.splitlines() if line.strip()] if isinstance(plain, str) else []
    status = "instrumental" if rec.get("instrumental") is True else ("synced" if synced else ("plain" if plain else "empty"))
    if status == "empty" and rec.get("syncedLyrics"):
        raise LookupFailed("returned unusable synced lyrics")
    return {"source": "LRCLIB", "status": status, "synced": synced, "plain": plain,
            "synced_unusable": bool(rec.get("syncedLyrics")) and not synced, "paired": _paired(synced),
            "id": rec.get("id"), "matched": {k: rec.get(k) for k in ("artistName", "trackName", "albumName", "duration")}}


def cache_path(artist, title, duration=None, album=None, cache_dir=None):
    directory = cache_dir or os.path.join(credits.ears_home(), "lrclib-cache")
    key = json.dumps([_norm(artist), _norm(title), _duration(duration), _norm(album), 1], ensure_ascii=False)
    return os.path.join(directory, hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")


def _cache_ok(answer):
    if not isinstance(answer, dict) or answer.get("source") != "LRCLIB":
        return False
    status = answer.get("status")
    if status == "synced":
        rows = answer.get("synced")
        if _paired(rows):
            return False
        return bool(rows) and isinstance(rows, list) and all(
            isinstance(r, dict) and isinstance(r.get("text"), str) and r["text"].strip()
            and isinstance(r.get("t"), (int, float)) and math.isfinite(r["t"]) for r in rows)
    if status == "plain":
        lines = answer.get("plain")
        return bool(lines) and isinstance(lines, list) and all(isinstance(line, str) and line.strip() for line in lines)
    if status == "instrumental":
        return True
    return (status in ("empty", "nomatch") and not answer.get("issues")
            and time.time() - (_duration(answer.get("when")) or 0) < NEG_TTL_S)


def _lookup(artist, title, duration, album):
    import song_length
    lengths, clean_lengths = set(), set()
    exact_failed = True

    def remember(rec, search=False):
        if (not _matches(rec, artist, title, None, album)
                or song_length.alternate(rec.get("trackName"))
                or song_length.alternate(rec.get("albumName"))):
            return
        seconds = _duration(rec.get("duration"))
        if seconds is not None:
            lengths.add(seconds)
            if search:
                try:
                    answer = _answer(rec)
                    if answer["status"] in ("synced", "plain") and not answer.get("paired"):
                        clean_lengths.add(seconds)
                except LookupFailed:
                    pass

    def finish(answer):
        if lengths:
            answer["studio_durations"] = [{"source": "LRCLIB", "duration_s": d} for d in sorted(lengths)]
        if exact_failed and clean_lengths:
            answer.update(exact_failed=True, clean_lyric_durations=sorted(clean_lengths))
        return answer

    params = {"artist_name": artist, "track_name": title}
    if duration is not None:
        params["duration"] = duration
    if album:
        params["album_name"] = album
    issues, candidates = [], []
    try:
        rec = _get_json("get", params)
        if rec is not None and (not isinstance(rec, dict) or not rec.get("artistName") or not rec.get("trackName")):
            raise LookupFailed("returned an invalid get response")
        remember(rec)
        if _matches(rec, artist, title, duration, album):
            exact_failed = False
            answer = _answer(rec)
            answer["endpoint"] = "get"
            if answer["status"] in ("synced", "instrumental") and not answer.get("paired"):
                return finish(answer)
            candidates.append(answer)
    except LookupFailed as e:
        issues.append("LRCLIB get " + str(e))
    # Search can supply synced words when the exact endpoint has only plain text.
    try:
        recs = _get_json("search", {"q": artist + " " + title})
        if not isinstance(recs, list) or any(not isinstance(rec, dict) for rec in recs):
            raise LookupFailed("returned an invalid search response")
        for rec in recs:
            remember(rec, search=True)
            if _matches(rec, artist, title, duration, album):
                answer = _answer(rec)
                answer["endpoint"] = "search"
                candidates.append(answer)
    except LookupFailed as e:
        issues.append("LRCLIB search " + str(e))
    # The free-text search can list only the translated uploads of a song while the
    # field search lists the clean ones (SAMURAI 'Never Fade Away', 11 Nov 2026). Ask
    # the field search only in that case, so every other lookup keeps its two requests.
    if (any(c.get("paired") for c in candidates)
            and not any(c["status"] == "synced" and not c.get("paired") for c in candidates)):
        try:
            fields = {"artist_name": artist, "track_name": title}
            if album:
                fields["album_name"] = album
            recs = _get_json("search", fields)
            if not isinstance(recs, list) or any(not isinstance(rec, dict) for rec in recs):
                raise LookupFailed("returned an invalid search response")
            for rec in recs:
                remember(rec, search=True)
                if _matches(rec, artist, title, duration, album):
                    answer = _answer(rec)
                    answer["endpoint"] = "search (fields)"
                    candidates.append(answer)
        except LookupFailed as e:
            issues.append("LRCLIB field search " + str(e))
    rank = {"synced": 3, "instrumental": 2, "plain": 1, "empty": 0}
    candidates.sort(key=lambda c: (-rank[c["status"]], bool(c.get("paired")), abs((_duration(c["matched"].get("duration")) or 0) - (duration or 0))))
    answer = candidates[0] if candidates else {"source": "LRCLIB", "status": "nomatch", "plain": [], "synced": []}
    if answer.get("paired"):
        issues.append("LRCLIB's only synced text pairs each line with a second line at the same time "
                      "(likely an interleaved translation); printed as supplied")
    if issues:
        answer["issues"] = issues
        if answer["status"] in ("empty", "nomatch"):
            answer["status"] = "failed"
    return finish(answer)


@ST.stage("LRCLIB", kind="input", classify=lambda value: ST.lookup_rows("LRCLIB", value))
def get(path, artist=None, title=None, duration=None, album=None, off=False, cache_dir=None, use_cache=True):
    """Metadata lookup, never raises for a broken backend. `use_cache=False` also prevents writes."""
    if off or offline() or os.environ.get("RAVEN_NO_LYRICS_LOOKUP", "").lower() in ("1", "true", "yes", "on"):
        return {"source": None, "status": "off", "note": "LRCLIB lookup off"}
    try:
        artist, title, _ = credits.identify(path, artist, title, for_lookup=True)
        if not artist or not title:
            return {"source": None, "status": "skipped", "note": "LRCLIB skipped: no artist/title (try --artist/--title)"}
        title = credits.clean_title(title)
        duration = _duration(duration)
        cp = cache_path(artist, title, duration, album, cache_dir)
        if use_cache:
            try:
                with open(cp, encoding="utf-8") as f:
                    cached = json.load(f)
                if _cache_ok(cached):
                    return dict(cached, cached=True)
            except (OSError, ValueError, AttributeError):
                pass
        answer = _lookup(artist, title, duration, album)
        answer["when"] = time.time()
        # A partial failure may accompany real lyrics, but never a negative cache entry.
        if use_cache and (answer["status"] in ("synced", "plain", "instrumental") or not answer.get("issues")):
            tmp = None
            try:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=os.path.dirname(cp), delete=False) as f:
                    tmp = f.name
                    json.dump(answer, f, ensure_ascii=False)
                os.replace(tmp, cp)
            except OSError:
                pass
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        return answer
    except Exception as e:
        return {"source": "LRCLIB", "status": "failed", "note": "LRCLIB lookup failed (" + type(e).__name__ + ")"}


def cached_activity(path):
    """Read an existing vocal stem only; never separate audio or create a cache."""
    try:
        import singer
        import soundfile as sf
        import numpy as np
        vp = os.path.join(singer.CACHE, singer.file_hash(path), "vocals.wav")
        if not os.path.isfile(vp):
            return None
        y, sr = sf.read(vp, always_2d=True)
        y = y.mean(axis=1)
        hop = max(1, sr // 2)
        n = len(y) // hop
        if not n:
            return None
        rms = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1))
        db = 20 * np.log10(rms + 1e-12)
        on = db > max(float(np.percentile(db, 90)) - 22.0, -42.0)
        return [(i * hop / sr, (i + 1) * hop / sr) for i in np.flatnonzero(on)]
    except Exception:
        return None


def sanity_check(synced, aligned, duration=None, activity=None):
    """Broad timing check, not verification of every word. None activity means unavailable; [] means measured quiet."""
    reasons, checks, no_voice = [], [], []
    if any(row["t"] < 0 or (duration and row["t"] >= duration) for row in synced):
        reasons.append("line starts outside this recording")
    pairs = [(a, b) for a, b in zip(synced, aligned) if not b["approx"]]
    if pairs:
        checks.append(f"{len(pairs)} lines compared with Whisper")
        if len(pairs) >= min(2, len(synced)) and sum(abs(a["t"] - b["t"]) > 3.0 for a, b in pairs) > len(pairs) / 2:
            reasons.append("line starts disagree with Whisper by more than 3 s on most matched lines")
    if activity is not None and synced:
        from lyrics_shared import gap_segments, voice_overlaps
        checks.append("vocal-stem activity checked")
        end = duration if duration is not None else max((b for a, b in activity), default=0)
        quiet = [(a, b) for a, b, description in gap_segments(0, end, (activity,))
                 if b - a >= 3.0 and description == "no voice (instrumental)"]
        misses = 0
        for i, row in enumerate(synced):
            stop = synced[i + 1]["t"] if i + 1 < len(synced) else end
            if voice_overlaps(row["t"], row.get("end", stop), activity):
                continue
            misses += 1
            if any(a <= row["t"] < b for a, b in quiet):
                no_voice.append(i)
        if misses > len(synced) / 2:
            reasons.append("most lines fall away from vocal-stem activity")
    return {"disagrees": bool(reasons), "reasons": reasons, "checks": checks, "no_voice": no_voice}


def _alignment_label(prefix, summary, words):
    if not words:
        return prefix + "; word timing unavailable or no words detected; lines placed (≈), not heard"
    return (prefix + f"; {summary['heard']} of {summary['total']} lines matched to Whisper, "
            f"{summary['approx']} placed (≈), {summary['differs']} where Whisper heard different words")


@ST.engine
def resolve(path, notes, duration=None, lyric_lines=None, word_loader=None, activity=None, words_skip_reason=None,
            quick=None, **lookup_kwargs):
    """Return {words, rows, summary, label, lookup}. `words` always remains audio evidence.

    Whisper is still needed for the singer/harmony evidence and for a timing check.
    External text never becomes invented word timestamps.
    """
    import lyric_align as LA
    if word_loader is None:
        from lyrics_shared import get_words
        word_loader = get_words
    lookup = {"source": "file", "status": "file"} if lyric_lines is not None else get(
        path, duration=duration, **dict(CONFIG, **lookup_kwargs))
    quick_skipped = bool((QUICK if quick is None else quick) and not words_skip_reason
                         and lyric_lines is None and lookup["status"] == "synced"
                         and not lookup.get("paired") and not lookup.get("issues"))
    if quick_skipped:
        words_skip_reason = "skipped in a quick listen"
        lookup = dict(lookup, quick_skipped=True)
        ST.mark("Whisper", "skipped", words_skip_reason, kind="input")
        ST.mark("unsure-line marks / gap recovery", "skipped", words_skip_reason, kind="input")
    if lyric_lines is not None:
        ST.mark("LRCLIB", "skipped", "lyrics file supplied", kind="input")
    ST.mark("lyrics file", "ran" if lyric_lines is not None else "not-found",
            "supplied" if lyric_lines is not None else "no lyrics file", kind="input")
    before_words = len(notes)
    words = None
    if lookup["status"] != "instrumental" and not words_skip_reason:
        try:
            from lyrics_shared import lyric_context
            text = lyric_lines if lyric_lines is not None else (
                [r["text"] for r in lookup.get("synced", [])] if lookup["status"] == "synced"
                else lookup.get("plain", []))
            with lyric_context(text, "lyrics file" if lyric_lines is not None else "LRCLIB"):
                words = word_loader(path, notes)
        except Exception as e:
            ST.mark("Whisper", "failed", f"{type(e).__name__}: {str(e)[:120]}")
            notes.append("LYRICS FAILED: Whisper unavailable (" + type(e).__name__ + "); not evidence of silence")
    if words_skip_reason:
        ST.mark("words", "skipped", words_skip_reason, kind="input")
    else:
        ST.collect(ST.word_rows(words, notes[before_words:], skipped=lookup["status"] == "instrumental"))
    issues = "; ".join(lookup.get("issues", []))
    status = lookup["status"]
    rows, summary = None, None
    if status == "instrumental":
        label = "LYRICS: instrumental (LRCLIB); database designation, not a measurement of silence"
        if issues:
            label += "; " + issues
    elif status in ("file", "synced", "plain"):
        text = lyric_lines if status == "file" else ([r["text"] for r in lookup["synced"]] if status == "synced" else lookup["plain"])
        title = lookup_kwargs.get("title") or CONFIG.get("title")
        if status == "file" and not title:
            _, title, _ = credits.identify(path)
        rows, summary = LA.align_lyrics(text, words, duration, title=title, filename=path,
                                       headers=status != "synced")
        if status == "file":
            label = LA.summary_line(summary)
            if not words:
                label += "; word timing unavailable or no words detected; ≈ is placement, not hearing"
        elif status == "plain":
            notice = "LRCLIB synced lookup incomplete" if issues else "LRCLIB had no synced lyrics"
            if lookup.get("synced_unusable"):
                notice = "LRCLIB synced lyrics were unusable"
            action = "plain lyrics aligned to the vocals" if words else "plain lyrics; alignment unavailable"
            prefix = "LYRICS: " + notice + "; " + action
            label = _alignment_label(prefix, summary, words)
        else:
            if activity is None:
                activity = cached_activity(path)
            windows = []
            for i, (row, supplied) in enumerate(zip(rows, lookup["synced"])):
                start, old_end = supplied["t"], row["end"]
                nxt = lookup["synced"][i + 1]["t"] if i + 1 < len(lookup["synced"]) else (duration or start + 5)
                estimate = old_end if 0 < old_end - start < 10 else start + max(0.4, row["tokens"] * 0.4)
                windows.append(dict(supplied, end=round(max(start, min(nxt, estimate)), 3)))
            check = sanity_check(windows, rows, duration, activity)
            lookup = dict(lookup, timing_check=check)
            if check["disagrees"] and not quick_skipped:
                label = _alignment_label("LYRICS: LRCLIB synced times disagree with this audio (" + "; ".join(check["reasons"])
                                         + "); likely a different version; " + ("preferred alignment to the vocals" if words
                                                                                else "alignment unavailable"), summary, words)
            else:
                # These are supplied times, even when a broad audio check agrees.
                for i, (row, supplied) in enumerate(zip(rows, windows)):
                    row.update(t=supplied["t"], end=supplied["end"], approx=True, timing="LRCLIB supplied")
                    if i in check["no_voice"]:
                        row["no_voice"] = True     # a flag, not text: rows feed CONTRAST, Jev, hooks and repeats
                summary = dict(summary, heard=0, approx=len(rows))
                checked = "; ".join(check["checks"]) if check["checks"] else "audio timing unverified"
                label = "LYRICS: LRCLIB synced lyrics; supplied times (≈), " + checked + "; line ends estimated"
                if quick_skipped:
                    label = ("LYRICS: LRCLIB synced lyrics; supplied times (≈); quick listen: "
                             "not checked against the singing; " + checked + "; line ends estimated")
                    if check["disagrees"]:
                        label += "; timing warning: " + "; ".join(check["reasons"])
                if check["no_voice"]:
                    n = len(check["no_voice"])
                    label += f"; {n} synced line{'s' if n != 1 else ''} fall{'s' if n == 1 else ''} where no voice was measured"
        if issues:
            label += "; " + issues
    else:
        # Fox, 11 Nov 2026: misheard lyrics are half the fun, but say how to fix them.
        why = lookup.get("note") or issues or ("LRCLIB had no lyrics" if status == "empty" else "LRCLIB had no matching lyrics")
        label = "LYRICS: " + why + ("; words are Whisper's guesses (rough times); have the words? --lyrics FILE" if words else "; no Whisper words available; not evidence of silence")
    # This branch selected the displayed text; Whisper's unsure marks refer
    # only to its own transcript when supplied lyrics are displayed instead.
    if status in ("file", "synced", "plain"):
        from lyrics_shared import alignment_note
        source = "the supplied file" if status == "file" else "LRCLIB"
        notes[:] = [alignment_note(note, source) for note in notes]
    return {"words": words, "rows": rows, "summary": summary, "label": label, "lookup": lookup,
            "quick_skipped": quick_skipped}


def add_args(ap):
    ap.add_argument("--quick", action="store_true", help="skip Whisper checking for clean synced LRCLIB lyrics")
    ap.add_argument("--no-lyrics-lookup", action="store_true", help="no LRCLIB requests (also RAVEN_NO_LYRICS_LOOKUP=1)")
    for name in ("artist", "title", "album"):
        if "--" + name not in ap._option_string_actions:
            ap.add_argument("--" + name, help=name + " for lyrics lookup (artist/title default to tags or file name)")


def configure_from_args(args):
    global QUICK
    QUICK = bool(getattr(args, "quick", False))
    CONFIG.update(off=bool(getattr(args, "no_lyrics_lookup", False)), artist=getattr(args, "artist", None),
                  title=getattr(args, "title", None), album=getattr(args, "album", None))


def for_music(path, notes, duration, lyric_lines=None, **kwargs):
    """Small music.py hook; keep its Whisper loader injectable for existing callers."""
    import music
    result = resolve(path, notes, duration, lyric_lines, word_loader=music.get_lyrics, **kwargs)
    result["lyrics"] = result["rows"] if result["rows"] is not None else music.lyric_lines(result["words"] or [])
    return result


def saved_voice_marks(text, data, name, activity, duration):
    """Refresh saved flags when the bundle itself contains positive stem evidence."""
    if NO_VOICE_MARK not in text and not activity:
        return text, data
    from lyric_align import quote_line
    from lyrics_shared import voice_overlaps
    from slim import seconds, split_voice_mark

    container = {"music": None, "singer": "lyrics", "harmony": "lyrics_aligned"}[name]
    field = "lines" if name == "harmony" else "rows" if container else "lyrics"
    block = ((data or {}).get(container) or {}) if container else (data or {})
    if not isinstance(block, dict):
        return text, data
    rows = block.get(field) or []
    if not isinstance(rows, list):
        rows = []
    windows = {}
    revised = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "t" not in row or "text" not in row:
            revised.append(row)
            continue
        start = row["t"]
        end = row.get("end")
        if end is None:
            following = rows[i + 1] if i + 1 < len(rows) else {}
            end = following.get("t", duration or start + 5) if isinstance(following, dict) else start + 5
        words, embedded = split_voice_mark(row["text"])
        voiced = voice_overlaps(start, end, activity)
        windows[credits.tm(start), quote_line(words) if name == "harmony" else words] = voiced
        if embedded or (row.get("no_voice") and voiced):
            row = dict(row, text=words)
            row.pop("no_voice", None)
            if not voiced:
                row["no_voice"] = True
        revised.append(row)

    marked = removed = 0
    def line(match):
        nonlocal marked, removed
        prefix, stamp, body = match.groups()
        if name == "harmony":
            body, separator, annotation = body.partition(" — ")
        else:
            separator = annotation = ""
        words, no_voice = split_voice_mark(body)
        if not no_voice:
            return match[0]
        marked += 1
        start = seconds(stamp)
        voiced = windows.get((stamp, words), voice_overlaps(start, start, activity))
        removed += voiced
        return prefix + words + ("" if voiced else NO_VOICE_MARK) + separator + annotation

    text = re.sub(r"(?m)^( +\[(\d+:\d\d)≈?\] +)([^\n]+)$", line, text)
    count = marked - removed
    status = r"; \d+ synced lines? falls? where no voice was measured"
    label = f"; {count} synced line{'s' if count != 1 else ''} fall{'s' if count == 1 else ''} where no voice was measured" if count else ""
    if removed:
        text = re.sub(status, label, text)
    if revised != rows:
        block = dict(block, **{field: revised})
        label_key = "label" if container else "lyrics_align"
        if removed and isinstance(block.get(label_key), str):
            block[label_key] = re.sub(status, label, block[label_key])
        data = dict(data, **{container: block}) if container else block
    return text, data


def no_voice_mark(row):
    """Printed after a synced line that lands inside a measured no-voice stretch; never part of the text."""
    return NO_VOICE_MARK if row.get("no_voice") else ""


def card_lines(result):
    """The singer's small lyric block; the music/harmony renderers keep their own timelines."""
    lines = [result["label"]]
    for row in result["rows"] or []:
        lines.append(f"  [{credits.tm(row['t'])}{'≈' if row.get('approx') else ''}] {row['text']}{no_voice_mark(row)}")
    return lines
