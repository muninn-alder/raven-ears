"""Claim policy over saved meter evidence; no audio engines or cache writes.

Accent cycles and tracker sections remain candidates until both sides support a
change. The raw JSON and full-card detail remain available when a claim is demoted.
"""
import math
import re

CHANGE_CONFIDENCE = 0.70
CHANGE_WINDOWS = 3          # overlapping pairs can describe just one accent event
SECTION_SECONDS = 40.0
DOMINANT_SHARE = 0.70
BAR_FIT = 0.70
STEADY_BARS = 0.90
COMPOUND_CONTRAST = 0.40
COMPOUND_VOTES = 0.90
CYCLE_NEAR_PEAK = 0.85     # same candidate band as singer.cycle_from_acf


def cycle_support(data):
    """Conditional cycle evidence, separate from the grouping template fit.

    A cycle and its multiples/divisors can describe the same repeated accents.
    Competing lengths outside that family must lose clearly before we name one.
    These are evidence scores, not calibrated probabilities.
    """
    k = data.get("cycle_pulses")
    acf = {int(n): v for n, v in (data.get("cycle_acf") or {}).items()
           if str(n).isdigit() and 2 <= int(n) <= 12
           and type(v) in (int, float) and math.isfinite(v)}
    if not k or k not in acf:
        return None
    related = lambda n: n % k == 0 or k % n == 0
    rivals = [n for n in acf if not related(n)]
    height = acf[k]
    rival = max((acf[n] for n in rivals), default=0.0)
    # Retain the existing 0.5 recurrence scale; spend confidence on separation
    # too. Hedwig (13 Nov 2026): 0.66 at five is beaten by 0.68 at seven.
    separation = max(0.0, min(1.0, (height - rival) / max(height * (1 - CYCLE_NEAR_PEAK), 1e-9)))
    confidence = max(0.0, min(1.0, height / 0.5)) * separation
    bpb = data.get("beats_per_bar")
    agrees = bool(bpb and bpb >= 2 and (k % bpb == 0 or bpb % k == 0))
    if bpb and bpb >= 2 and not agrees and data.get("bar_consistency", 0) >= STEADY_BARS:
        confidence = min(confidence, 1 - data["bar_consistency"])
    peak = max(acf.values())
    candidates = sorted({k} | {n for n in rivals if acf[n] >= CYCLE_NEAR_PEAK * peak})
    if len(candidates) > 1:
        confidence = 0.0   # no winner inside the detector's own candidate band
    return dict(confidence=confidence, candidates=candidates, tracker_agrees=agrees)


def cycle_unclear(data):
    # Only the additive/grouping path names a meter from this cycle. Simple and
    # compound tracker bars have their own evidence (e.g. Counting Stars).
    support = cycle_support(data) if data.get("grouping") else None
    if not support:
        return None
    candidates = support["candidates"]
    if len(candidates) > 1:
        pulses = ", ".join(map(str, candidates[:-1])) + " and " + str(candidates[-1])
        return f"meter unclear: {pulses} pulses all fit about equally"
    if support["confidence"] < 0.35:
        return "meter unclear: accent cycle weak or contradicted by steady tracker bars"
    return None


def tracker_unclear(data):
    bpb, k = data.get("beats_per_bar"), data.get("cycle_pulses")
    if bpb == 1 and not data.get("grouping"):
        return "meter unclear: tracker counted one pulse per bar; no reliable bar length"
    if (not data.get("grouping") and bpb and k and k % bpb and bpb % k
            and data.get("bar_consistency", 0) < BAR_FIT):
        return f"meter unclear: accents suggest {k} pulses; tracker counted {bpb} beats per bar inconsistently"
    return None


def compound_split(data):
    # 10 Nov 2026: bouncy straight eighths cleared the old 0.20 contrast gate.
    # Bar regularity establishes the beat count, not triplet subdivision. Require
    # both a wide onset contrast and near-unanimous beat votes to name compound.
    contrast, votes = data.get("split_contrast"), data.get("split_votes_ternary")
    return (type(contrast) in (int, float) and COMPOUND_CONTRAST <= contrast <= 1
            and type(votes) in (int, float) and COMPOUND_VOTES <= votes <= 1)


def subdivision_ambiguity(data):
    bpb = data.get("beats_per_bar")
    names = {2: ("2/4", "6/8"), 3: ("3/4", "9/8"), 4: ("4/4", "12/8")}
    if (bpb in names and (data.get("verdict") or "").startswith(names[bpb][1] + " (compound, each beat splits in 3)")
            and not compound_split(data)):
        simple, compound = names[bpb]
        return f"{bpb} beats per bar; subdivision ambiguous (candidates {simple} / {compound})"
    return None


def accent_cycle_ambiguity(data):
    # 12 Nov 2026: eight quarter-rate pulses were named as eight eighths.
    # A template fit measures accents, not the pulse's note value or a bar line.
    if data.get("cycle_pulses") != 8 or not data.get("grouping") or cycle_unclear(data):
        return None
    pulse = data.get("tracker_bpm") or data.get("pulse_bpm")
    rate = f" at {pulse:g}/min" if type(pulse) in (int, float) and pulse > 0 else ""
    strength = "grouping weak" if data.get("grouping_confidence", 0) < 0.35 else "uncertain"
    # The beat tracker's own bars are the 4/4 evidence; a quarter-rate pulse (singer.QUARTER_PULSE)
    # makes eight pulses two of those bars. Fox's ear, Thunder God and The Old Religion (12 Nov 2026).
    if (data.get("beats_per_bar") == 4 and type(pulse) in (int, float) and 70.0 <= pulse <= 230.0):
        return (f"4/4, accents repeat every 2 bars{rate}: best fit {data['grouping']} ({strength}); "
                "8/8 only if each pulse is an eighth")
    return (f"uncertain (4/4 over two bars / 8/8); accent cycle 8 pulses{rate}, "
            f"best fit {data['grouping']} ({strength}); pulse note value unconfirmed")


def one(data):
    verdict = data.get("verdict") or ""
    support = cycle_support(data) if data.get("grouping") else None
    confidence = min(data.get("confidence", 0), support["confidence"]) if support else data.get("confidence", 0)
    if (cycle_unclear(data) or tracker_unclear(data) or accent_cycle_ambiguity(data) or subdivision_ambiguity(data) or confidence < 0.35 or not re.match(r"^\d+/\d+\b", verdict)
            or any(w in verdict for w in ("unusual", "ambiguous", "unclear", "weak", "per bar"))):
        return None
    return verdict.split()[0]


def same_cycle(a, b):
    return bool(a and b and min(a, b) >= 3 and max(a, b) % min(a, b) == 0)


def _bar_agrees(label, data):
    bpb = data.get("beats_per_bar")
    # Strong odd cycles can overrule an unreliable downbeat tracker (Take Five).
    return (data.get("bar_consistency", 0) < STEADY_BARS or not bpb
            or label == bpb or same_cycle(label, bpb))


def _window_change_ok(data):
    runs = []
    for i, w in enumerate(data.get("windows", [])):
        label, conf = w[1:3]
        local = data.get("window_bars") or []
        bars = local[i] if i < len(local) else data
        agrees = _bar_agrees(label, bars)
        if i < len(local):
            agrees = (bars.get("bar_consistency", 0) >= BAR_FIT
                      and (label == bars.get("beats_per_bar") or same_cycle(label, bars.get("beats_per_bar"))))
            if label and label >= 5 and bars.get("bar_consistency", 0) < BAR_FIT:
                agrees = True   # downbeats cannot resolve this odd/additive cycle
        ok = label is not None and conf >= CHANGE_CONFIDENCE and agrees
        if ok and runs and runs[-1]["end"] == i - 1 and same_cycle(runs[-1]["label"], label):
            runs[-1]["end"] = i
        elif ok:
            runs.append({"start": i, "end": i, "label": label})
    solid = [r["label"] for r in runs if r["end"] - r["start"] + 1 >= CHANGE_WINDOWS]
    chain = [solid[0]] if solid else []
    for label in solid[1:]:
        if not same_cycle(chain[-1], label):
            chain.append(label)
    candidates = data.get("changes") or []
    expected = [candidates[0]["from"]] + [c["to"] for c in candidates] if candidates else []
    return len(chain) > 1 and len(chain) == len(expected) and all(same_cycle(a, b) for a, b in zip(chain, expected))


def _section_ok(section):
    d = section["data"]
    name = one(d)
    if not name or d.get("confidence", 0) < CHANGE_CONFIDENCE or section["end"] - section["start"] < SECTION_SECONDS:
        return False
    if d.get("cycle_source") == "accent autocorrelation":
        return (d.get("cycle_strength", 0) >= 0.50
                and d.get("grouping_confidence", 0) >= CHANGE_CONFIDENCE
                and _bar_agrees(d.get("cycle_pulses"), d))
    numerator, denominator = map(int, name.split("/"))
    beats = numerator // 3 if denominator == 8 and numerator % 3 == 0 else numerator
    counted = d.get("beats_per_bar")
    return (d.get("bar_consistency", 0) >= BAR_FIT
            and (counted == beats or same_cycle(counted, beats)))


def assess(data):
    """(claim, demoted): short commitment and whether the old change heading needs a hedge."""
    if not isinstance(data, dict):
        return None, False
    sections = data.get("sections") or []
    unclear = cycle_unclear(data) or tracker_unclear(data)
    if unclear and not sections:
        return unclear, bool(data.get("changes") or data.get("change_chain"))
    cycle = accent_cycle_ambiguity(data)
    if cycle:
        return cycle, bool(data.get("changes") or data.get("change_chain"))
    if sections and all(accent_cycle_ambiguity(s["data"]) for s in sections):
        whole = data.get("whole_song_data") or {}
        return accent_cycle_ambiguity(whole) or accent_cycle_ambiguity(sections[0]["data"]), len(sections) > 1
    if sections:
        names = [one(s["data"]) for s in sections]
        if len(sections) == 1:
            d = sections[0]["data"]
            return cycle_unclear(d) or tracker_unclear(d) or subdivision_ambiguity(d) or names[0], False
        # Old sections lacking timing/bar evidence cannot establish a change.
        total = sum(max(0, s.get("end", 0) - s.get("start", 0)) for s in sections)
        if not total:
            return None, True
        all_solid = all(_section_ok(s) for s in sections)
        # A half/double pulse with the same bar duration is not a time change.
        bars_differ = all(a["data"].get("bar_s") and b["data"].get("bar_s")
                          and abs(a["data"]["bar_s"] / b["data"]["bar_s"] - 1) > 0.15
                          for a, b in zip(sections, sections[1:]))
        if all_solid and bars_differ:
            chain = [names[0]] + [n for i, n in enumerate(names[1:]) if n != names[i]]
            return " then ".join(chain), False
        shares = {}
        for s, name in zip(sections, names):
            if name:
                shares[name] = shares.get(name, 0) + s["end"] - s["start"]
        base = max(shares, key=shares.get) if shares else None
        if base and shares[base] / total >= DOMINANT_SHARE:
            winner = next(s["data"] for s, name in zip(sections, names) if name == base)
            cycle = winner.get("cycle_pulses")
            compatible = cycle and all(same_cycle(cycle, s["data"].get("cycle_pulses")) for s in sections)
            if compatible and winner.get("grouping_confidence", 0) >= CHANGE_CONFIDENCE:
                return base + " (" + winner["grouping"] + ")", False
            return base, True
        return None, True
    if data.get("change_chain") or data.get("changes"):
        if _window_change_ok(data):
            return " then ".join(data["change_chain"]), False
        return subdivision_ambiguity(data) or one(data), True
    return subdivision_ambiguity(data) or one(data), False


def _clock(t):
    t = round(t)
    return f"{t // 60}:{t % 60:02d}"


def _meter_text(text, data):
    """Re-evaluate claims from saved evidence without changing the capture."""
    single = data.get("sections") or []
    rows = [s["data"] for s in single] if single else [data]
    if any(accent_cycle_ambiguity(d) for d in rows):
        # Saved section blocks have a leading clock range, unlike the main line.
        section_index = 0
        active = data if not single else None
        lines = []
        for line in text.splitlines():
            if re.match(r"^\s+\[\d+:\d+–\d+:\d+\] ", line) and section_index < len(single):
                active = rows[section_index]
                section_index += 1
                if accent_cycle_ambiguity(active):
                    line = re.sub(r"^(\s+\[[^]]+\]) .*", lambda m: m[1] + " " + accent_cycle_ambiguity(active), line)
            if active and accent_cycle_ambiguity(active):
                line = re.sub(r"\bbar ≈ (.*?\(cycle ×)", r"accent cycle ≈ \1", line)
                line = line.replace(", which the accents do not support", "; the accent cycle does not establish a bar line")
            lines.append(line)
        text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        claim, demoted = assess(data)
        if accent_cycle_ambiguity(data) or all(accent_cycle_ambiguity(d) for d in rows):
            heading = "METER: " + claim + ("; meter changes unconfirmed" if demoted else "")
            return re.sub(r"(?m)^METER: .*$", lambda _: heading, text, count=1)
    ambiguity = subdivision_ambiguity(single[0]["data"] if len(single) == 1 else data)
    if ambiguity:
        heading = "METER: " + ambiguity
        match = re.search(r"(?m)^METER: (.*)$", text)
        if match and not match[0].startswith(heading):
            text = text[:match.start()] + heading + "\n  unconfirmed subdivision candidate: " + match[1] + text[match.end():]
    claim, demoted = assess(data)
    if not demoted or not re.search(r"(?m)^METER:", text) or "; meter changes unconfirmed" in text:
        return text
    old = re.search(r"(?m)^METER: (.*)$", text)[1]
    base = (ambiguity or data.get("verdict")) if claim and not data.get("sections") else claim
    heading = "METER: " + (base or "meter unclear") + "; meter changes unconfirmed"
    notes = []
    sections = data.get("sections") or []
    if sections:
        bars = [s["data"].get("bar_s", 0) for s in sections]
        if min(bars) > 0 and max(bars) / min(bars) <= 1.15:
            edges = ", ".join("≈" + _clock(s["start"]) for s in sections[1:])
            notes.append(f"  accent/tracker pattern boundaries {edges}; tracker bar stays ≈{bars[0]:.2f}s")
    wins = data.get("windows") or []
    bpb = data.get("beats_per_bar")
    i = 0
    while i < len(wins):
        j = i + 1
        label = wins[i][1]
        if label is None or not bpb or same_cycle(label, bpb):
            i = j
            continue
        while j < len(wins) and wins[j][1] == label:
            j += 1
        if j - i < 2:
            i = j
            continue
        end = wins[j][0] if j < len(wins) else wins[j - 1][0]
        strength = [w[2] for w in wins[i:j]]
        weak = "weak, " if max(strength) < CHANGE_CONFIDENCE else ""
        notes.append(f"  accent-cycle candidate ≈{_clock(wins[i][0])}–{_clock(end)}: {label} pulses "
                     f"({weak}confidence {min(strength):.2f}–{max(strength):.2f}); "
                     f"tracker bars {data.get('bar_consistency', 0):.0%} at {bpb} beats")
        i = j
    notes.append("  unconfirmed candidates: " + old)
    return re.sub(r"(?m)^METER: .*$", lambda _: "\n".join([heading] + notes), text, count=1)


def _confidence_line(line, data):
    unclear = cycle_unclear(data) or tracker_unclear(data)
    if unclear:
        # Both the main heading and saved section rows can contain a verdict.
        line = re.sub(r"^(METER: |\s+\[[^]]+\] ).*", lambda m: m[1] + unclear, line)
    support = cycle_support(data) if data.get("grouping") else None
    if support:
        conf = min(data.get("confidence", 0), data.get("grouping_confidence", 0), support["confidence"])
        tag = "high" if conf >= 0.6 else "moderate" if conf >= 0.35 else "low"
        fit = f"confidence {tag} ({conf:.2f}) · cycle support {support['confidence']:.2f}"
        # Template discrimination is conditional on K; five pulses have only
        # one cyclic 2+3 template. Never present that as meter confidence.
        fit += f" · tracker {data.get('beats_per_bar', '?')}/bar"
        fit += f" ({data.get('bar_consistency', 0):.0%} consistent)"
        fit += f" · grouping {data['grouping']} assumes {data['cycle_pulses']} pulses"
        pattern = r"(?:grouping|template) confidence \d+(?:\.\d+)?"
        if re.search(pattern, line):
            line = re.sub(pattern, lambda _: fit, line)
        elif unclear or " · cycle support " not in line:
            line += " · " + fit
    return line


def meter_text(text, data):
    """Apply saved claim policy, then qualify conditional grouping scores."""
    text = _meter_text(text, data)
    sections = data.get("sections") or []
    rows = [s["data"] for s in sections]
    lines, section_index, active = [], 0, None
    for line in text.splitlines():
        if line.startswith("METER: "):
            active = rows[0] if len(rows) == 1 else data
            line = _confidence_line(line, active)
        elif re.match(r"^\s+\[\d+:\d+–\d+:\d+\] ", line) and section_index < len(rows):
            active = rows[section_index]
            line = _confidence_line(line, active)
            section_index += 1
        elif active and cycle_support(active) and active.get("grouping"):
            k = active["cycle_pulses"]
            strength = (active.get("cycle_acf") or {}).get(str(k))
            if strength is not None:
                # Older cards printed the maximum ACF at the chosen lag, even
                # when another lag won. Keep the saved JSON; fix the display.
                line = re.sub(r"(from [^:]+: )[-\d.]+( at lag " + str(k) + r";)",
                              lambda m: m[1] + f"{strength:.2f}" + m[2], line)
            line = re.sub(r"^(\s*)cycle (\d+ pulses \(from )", r"\1cycle candidate \2", line)
            if cycle_unclear(active):
                line = re.sub(r"^(\s*)bar ≈", r"\1candidate cycle ≈", line)
                line = re.sub(r"(?:grouping|template) confidence \d+(?:\.\d+)?",
                              "conditional grouping fit; cycle unconfirmed", line)
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def tempo_text(text, data):
    claim, demoted = assess(data)
    if (not demoted and not data.get("sections") and not subdivision_ambiguity(data)
            and not accent_cycle_ambiguity(data) and not cycle_unclear(data) and not tracker_unclear(data)):
        return text
    def replace(match):
        line = re.sub(r" · meter .*?(?:, see METER|$)", "", match[0])
        label = claim if (claim or "").startswith("meter ") else "meter " + (claim or "")
        return line + (f" · {label}, see METER" if claim else "")
    return re.sub(r"(?m)^\s*TEMPO:[^\n]*", replace, text)
