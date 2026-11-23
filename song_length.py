"""Version warning from metadata already fetched by the song ears. No I/O."""
import math
import re
import statistics
import unicodedata

SOURCES = ("LRCLIB", "MusicBrainz", "Discogs")
WARNING = "⚠ LENGTH:"
LYRIC_NOTE = "NOTES: LRCLIB has clean lyrics at "


def seconds(value):
    try:
        value = float(value) if not isinstance(value, bool) else 0
        return value if math.isfinite(value) and value > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def tolerance(duration):
    # Five seconds covers rounding, encoder padding and small fades; three percent
    # allows proportionally more slack on long tracks. The boundary is exclusive.
    return max(5.0, duration * 0.03)


def norm(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value or "").casefold()))


def alternate(value):
    return bool(re.search(r"\b(live|remix|acoustic|demo|instrumental|karaoke|cover|radio edit|"
                          r"version|session|mix|extended|edit)\b", value or "", re.I))


def consensus(rows):
    """One vote per service, regardless of uploads or passes. Ties abstain.

    Choose a unique largest agreeing cluster; then take the median of the
    services' medians. Two disagreeing services cannot establish a studio length.
    A single service suffices only when its returned lengths agree.
    """
    values = sorted({(s, d) for r in rows if isinstance(r, dict)
                     if (s := r.get("source")) in SOURCES
                     if (d := seconds(r.get("duration_s"))) is not None}, key=lambda r: r[1])
    clusters = set()
    for i in range(len(values)):
        largest = None
        for j in range(i + 1, len(values) + 1):
            group = values[i:j]
            if group[-1][1] - group[0][1] <= tolerance(statistics.median(d for s, d in group)):
                largest = frozenset(group)
        clusters.add(largest)
    if not clusters:
        return None
    # Drop subclusters rather than letting a service's near-duplicate lengths tie.
    clusters = [c for c in clusters if not any(c < other for other in clusters)]
    support = lambda c: len({s for s, d in c})
    best = max(map(support, clusters))
    winners = [c for c in clusters if support(c) == best]
    if len(winners) != 1:
        return None
    group = winners[0]
    sources = [s for s in SOURCES if any(source == s for source, d in group)]
    duration = statistics.median(statistics.median(d for source, d in group if source == s) for s in sources)
    return {"studio_s": round(duration, 3), "sources": sources}


def lookup(data):
    lyrics = data.get("lyrics_source") or data.get("lyrics") or {}
    return lyrics.get("lookup", lyrics) if isinstance(lyrics, dict) else {}


def evidence(data):
    lyr = lookup(data)
    rows = list(data.get("studio_durations") or []) + list(lyr.get("studio_durations") or [])
    matched = lyr.get("matched") or {}
    if (lyr.get("source") == "LRCLIB" and seconds(matched.get("duration"))
            and not alternate(matched.get("trackName")) and not alternate(matched.get("albumName"))):
        rows.append({"source": "LRCLIB", "duration_s": matched["duration"]})
    return rows


def finding(duration, rows, lookups=()):
    duration = seconds(duration)
    values = {(r.get("source"), d) for r in rows
              if isinstance(r, dict) and r.get("source") in SOURCES
              if (d := seconds(r.get("duration_s"))) is not None}
    rows = [{"source": s, "duration_s": d} for s, d in sorted(values)]
    result = consensus(rows)
    values = [r["duration_s"] for r in rows]
    conflict = bool(values and max(values) - min(values) > tolerance(statistics.median(values)))
    if not duration:
        return None
    if not result or abs(duration - result["studio_s"]) <= tolerance(result["studio_s"]):
        return {"status": "conflict", "copy_s": duration, "durations": rows} if conflict else None
    result["copy_s"] = duration
    if conflict:
        result["durations"] = rows
    if duration > result["studio_s"] and any(
            lyr.get("exact_failed") and any(abs(d - result["studio_s"]) <= 2 for v in lyr.get("clean_lyric_durations", [])
                                           if (d := seconds(v)) is not None) for lyr in lookups):
        result["studio_lyrics_available"] = True
    return result


def capture(path, data, duration=None):
    """Keep only already-fetched credits; never call credits.for_song here."""
    import credits
    rows = credits.duration_evidence(path)
    if rows:
        data["studio_durations"] = rows
    result = finding(duration or data.get("duration_s", data.get("duration")), evidence(data), [lookup(data)])
    if result:
        data["length_check"] = result


def update_song(song):
    cards = [card.get("json") or {} for name, card in song["cards"].items()
             if song["manifest"]["passes"][name]["status"] == "ran"]
    result = finding(song["manifest"]["duration_s"], [r for data in cards for r in evidence(data)],
                     [lookup(data) for data in cards])
    song.pop("length_check", None)
    if result:
        song["length_check"] = result


def strip(text):
    return "".join(line for line in text.splitlines(keepends=True)
                   if not line.startswith((WARNING, LYRIC_NOTE)))


def clock(value):
    value = int(value + 0.5)
    return f"{value // 60}:{value % 60:02d}"


def annotate(text, result):
    if not result:
        return text
    warning = ""
    if result.get("durations"):
        lengths = "; ".join(f"{r['source']} {clock(r['duration_s'])}" for r in result["durations"])
        warning = f"{WARNING} studio lengths conflict ({lengths}); this copy is {clock(result['copy_s'])}.\n"
        if result.get("status") == "conflict":
            return warning + strip(text)
    warning += (f"{WARNING} this copy is {clock(result['copy_s'])}; the studio track is "
               f"{clock(result['studio_s'])} ({', '.join(result['sources'])}). "
               "It may be a video, live or extended version, so the lyrics, sections and ending may not match.\n")
    if result.get("studio_lyrics_available"):
        warning += (f"{LYRIC_NOTE}{clock(result['studio_s'])}; not used because this longer copy's "
                    "duration does not match; the lyrics would be misplaced.\n")
    return warning + strip(text)
