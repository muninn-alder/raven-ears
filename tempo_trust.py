"""One tempo policy for full, feel and melody cards, including saved captures.

10 Nov 2026: steady beat_this bars lead even at half/double pulse or
when librosa agrees. Librosa is the fallback, not a veto on the beat tracker.
"""
import math
import re

AGREE_TOL = 0.04            # name librosa's other reading only beyond 4%
MIN_BAR_CONSISTENCY = 0.80  # share of bars at the median beat count
BAR_SPREAD_TOL = 0.15      # same bar-duration tolerance as meter_trust
DOMINANT_SHARE = 0.70


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _steady(data):
    if not isinstance(data, dict) or data.get("engine") != "beat_this":
        return False
    bpm, cons, bpb = _num(data.get("tracker_bpm")), _num(data.get("bar_consistency")), data.get("beats_per_bar")
    return (bpm is not None and bpm > 0 and cons is not None and MIN_BAR_CONSISTENCY <= cons <= 1
            and type(bpb) is int and bpb >= 2)


def _select(data):
    if not isinstance(data, dict):
        return None
    # Newer saves retain the whole-song measurement: do not cherry-pick sections
    # around an unsteady whole-song grid.
    if isinstance(data.get("whole_song_data"), dict):
        whole = data["whole_song_data"]
        return (whole, "whole-song tracker") if _steady(whole) else None
    sections = data.get("sections") or []
    if not sections:
        return (data, "") if _steady(data) else None
    if len(sections) == 1:
        d = sections[0].get("data", sections[0])
        return (d, "") if _steady(d) else None
    rows = []
    for section in sections:
        d = section.get("data", section)
        start, end = _num(section.get("start")), _num(section.get("end"))
        bar = _num(d.get("bar_s"))
        if not _steady(d) or start is None or end is None or end <= start or bar is None or bar <= 0:
            return None
        rows.append((d, end - start, bar))
    bars = [bar for _, _, bar in rows]
    if max(bars) / min(bars) > 1 + BAR_SPREAD_TOL:
        return None  # one tempo cannot honestly describe these different bar lengths
    total = sum(duration for _, duration, _ in rows)
    shares = {}
    for d, duration, _ in rows:
        bpb = d["beats_per_bar"]
        shares[bpb] = shares.get(bpb, 0) + duration
    bpb = max(shares, key=shares.get)
    share = shares[bpb] / total
    if share < DOMINANT_SHARE:
        return None
    # Duration-weighted median bar length, at the dominant beat count. Never
    # average BPMs: a short 2/4 half-pulse section must not drag down Feeling This.
    elapsed = 0
    for _, duration, bar in sorted(rows, key=lambda row: row[2]):
        elapsed += duration
        if elapsed >= total / 2:
            break
    d = {"tracker_bpm": round(60 * bpb / bar, 1), "beats_per_bar": bpb,
         "bar_consistency": min(d["bar_consistency"] for d, _, _ in rows)}
    return d, f"{bpb} beats × 60 / {bar:g}s; {bpb}-beat sections cover {share:.0%}"


def lead_reading(librosa_bpm, meter_data, claim=None):
    """None (librosa fallback), or the steady beat_this reading and its provenance."""
    selected = _select(meter_data)
    if selected is None:
        return None
    d, source = selected
    if claim is None:
        from meter_trust import assess
        claim = assess(meter_data)[0]
    return {"bpm": d["tracker_bpm"], "bpb": d["beats_per_bar"], "consistency": d["bar_consistency"],
            "librosa": _num(librosa_bpm), "meter": claim or f"{d['beats_per_bar']} beats per bar", "source": source}


def lead_line(reading):
    """TEMPO body, without a felt-beat hedge; preserve a differing librosa reading."""
    line = f"{reading['bpm']:g} BPM · {reading['meter']} · bars steady {reading['consistency']:.2f} (beat tracker)"
    if reading["source"]:
        line += " · " + reading["source"]
    lib = reading["librosa"]
    if lib and lib > 0 and abs(reading["bpm"] / lib - 1) > AGREE_TOL:
        line += f" · librosa heard {lib:g}"
    return line


def saved_meter(singer_card, pass_status=None):
    """An explicitly failed/skipped meter stage cannot lend its partial JSON."""
    from stage_status import card_notes, source_notes
    if (pass_status or {}).get("status", "ran") != "ran":
        return None
    data, text = singer_card.get("json") or {}, singer_card.get("text") or ""
    for row in data.get("stage_status") or []:
        if (isinstance(row, dict) and row.get("stage", "").lower() in ("meter", "beat tracker")
                and row.get("status") in ("failed", "skipped", "not run")):
            return None
    if any(re.match(r"(?:NOTE:\s*)?(?:meter|beat tracker):\s*(?:failed|skipped|unavailable|disabled|not run)\b", row, re.I)
           for row in card_notes(text, data) + source_notes(text)):
        return None
    return data.get("meter")


def estimate_line(line, meter_data=None):
    """Label the librosa fallback; an unsteady tracker is evidence, not a lead."""
    if "(beat tracker)" in line:
        return line
    line = re.sub(r"^(\d+(?:\.\d+)? BPM)(?! \(librosa estimate\))", r"\1 (librosa estimate)", line)
    # Re-rendering must not duplicate the comparison, or retain a stale one.
    line = line.split(" · beat tracker heard ")[0]
    if not isinstance(meter_data, dict):
        return line
    whole = meter_data.get("whole_song_data")
    rows = [whole] if isinstance(whole, dict) else [s.get("data", s) for s in meter_data.get("sections") or []]
    rows = rows or [meter_data]
    rates = sorted({_num(d.get("tracker_bpm")) for d in rows
                    if d.get("engine", meter_data.get("engine")) in ("beat_this", "madmom")
                    and (_num(d.get("tracker_bpm")) or 0) > 0})
    if rates:
        rate = f"{rates[0]:g}" if len(rates) == 1 else f"{rates[0]:g}–{rates[-1]:g}"
        line += f" · beat tracker heard {rate} BPM"
        if len(rows) > 1:
            line += " across sections"
        elif (cons := _num(rows[0].get("bar_consistency"))) is not None:
            line += f" (bars steady {cons:.2f})"
    return line


def tempo_text(text, librosa_bpm, meter_data, claim=None):
    """Correct just the saved TEMPO heading; leave captured evidence untouched."""
    reading = lead_reading(librosa_bpm, meter_data, claim)
    if reading is None:
        return re.sub(r"(?m)^([ \t]*TEMPO:[ \t]*)([^\n]*)",
                      lambda m: m[1] + estimate_line(m[2], meter_data), text)
    corrected, count = re.subn(r"(?m)^([ \t]*TEMPO:[ \t]*)[^\n]*", lambda m: m[1] + lead_line(reading), text)
    if count:
        return corrected
    line = "  TEMPO:    " + lead_line(reading) + "\n"
    # A failed librosa stage may have left no heading to replace.
    header = re.search(r"(?m)^MUSIC:[^\n]*(?:\n|$)", text)
    if header:
        return text[:header.end()].rstrip("\n") + "\n" + line + text[header.end():]
    return line + text
