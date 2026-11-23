#!/usr/bin/env python3
"""Read the answer shelf and existing ears outputs. Never runs an audio backend.

    python eval_shelf.py [--compare] [--save]

Only --save writes anything, in raven/eval under --cache-dir. Input notes are
user-supplied references, not independently verified ground truth.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


DIMENSIONS = ("key", "mode", "meter", "tempo", "instruments", "lyrics")
SCHEMA = 1
POLICY = "shelf-1"
NOTES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
KEY_RE = re.compile(r"\b([A-G])([#b♯♭]?)\s+(major|minor|mixolydian|blues-rock)\b", re.I)
METER_RE = re.compile(r"(?<![\d.])(\d+)/(2|4|8|16|32)\b")
UNCLEAR = re.compile(r"unclear|ambiguous|candidates|not enough|unavailable|failed|could not|no clear|working reference|\bor\b|\bvs\b", re.I)
INSTRUMENTS = {
    "electric upright bass": ("electric upright bass",),
    "electric guitar": ("electric guitar",), "acoustic guitar": ("acoustic guitar",),
    "bass guitar": ("bass guitar",), "synth bass": ("synth bass", "synthesizer bass"),
    "drum machine": ("drum programming", "drum machine", "programmed drums"),
    "drums": ("drum kit", "drums"), "piano": ("piano",),
    "organ": ("hammond organ", "organ"), "synth": ("synthesizer", "synthesiser", "synth"),
    "guitar": ("guitar strings", "guitar"), "bass": ("bass",),
    "voice": ("vocals", "voice", "vocal"), "violin": ("violins", "violin"),
    "strings": ("strings",), "cello": ("cello",), "clarinet": ("clarinet",),
    "saxophone": ("saxophone", "sax"), "banjo": ("banjo",),
    "mandolin": ("mandolin",), "ukulele": ("ukulele",), "percussion": ("percussion",),
    "keyboards": ("keyboards", "keyboard"), "omnichord": ("omnichord",),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def norm(value):
    value = unicodedata.normalize("NFKD", str(value)).casefold()
    value = re.sub(r"\([^)]*(?:lyric video|official video|remaster)[^)]*\)", " ", value)
    return " ".join(re.findall(r"[^\W_]+", value, re.UNICODE))


def contains(short, long):
    return bool(short) and f" {short} " in f" {long} "


def instruments_in(text):
    """Longest phrase first: 'electric guitar' doesn't also become 'guitar'."""
    text = text.lower()
    found = set()
    phrases = sorted(((alias, name) for name, aliases in INSTRUMENTS.items() for alias in aliases),
                     key=lambda x: -len(x[0]))
    for phrase, name in phrases:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text):
            found.add(name)
            text = re.sub(pattern, " ", text)
    return sorted(found)


def named_instruments(label):
    # A named family with unknown subtype still earns its broad name.
    label = re.sub(r"\(type unclear\)", "", label, flags=re.I)
    return [] if re.search(r"unclear|\?|/| or ", label) else instruments_in(label)


def key_value(text):
    m = KEY_RE.search(str(text))
    if not m:
        return None
    note, accidental, mode = m.groups()
    pc = (NOTES[note.upper()] + (1 if accidental in ("#", "♯") else -1 if accidental else 0)) % 12
    return [pc, mode.lower()]


def meters(text):
    return list(dict.fromkeys(f"{a}/{b}" for a, b in METER_RE.findall(str(text))))


def meter_estimate(claim):
    value = meters(claim) if isinstance(claim, str) and not UNCLEAR.search(claim) else None
    return {"value": value or None, "raw": claim or "cached meter abstained"}


def read_text(path, issues):
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as e:
        issues.append(f"{path}: unreadable ({type(e).__name__})")
        return None


def read_json(path, issues):
    text = read_text(path, issues)
    if text is None:
        return None
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except (ValueError, TypeError):
        issues.append(f"{path}: malformed JSON object; skipped")
        return None


def new_song(title, alias=None):
    return {"id": norm(title), "title": title, "aliases": [title] + ([alias] if alias else []),
            "truth": {}, "sources": [], "skipped": []}


def put_truth(song, dimension, value, source, evidence, status="ear note"):
    item = {"value": value, "source": str(source), "evidence": evidence.strip(), "status": status}
    old = song["truth"].get(dimension)
    if old and old.get("value") != value:
        song["truth"][dimension] = {"conflict": [old, item]}
        song["skipped"].append(f"conflicting {dimension} truth; unscorable")
    elif not old:
        song["truth"][dimension] = item


def parse_ear_key(path, issues):
    text = read_text(path, issues)
    if text is None:
        return new_song(path.stem, path.stem.removeprefix("EAR-KEY-"))
    heading = next((s.lstrip("# ") for s in text.splitlines() if s.startswith("# ")), "")
    title = re.split(r"\s+[—–]\s+", heading)[0] or path.stem.removeprefix("EAR-KEY-")
    # A final artist parenthesis is metadata; recording/version markers stay part of the identity.
    title = re.sub(r"\s+\((?![^)]*(?:Version|Live|Remix|Acoustic|Instrumental|Cover|Demo|Edit))[^)]*\)$", "", title, flags=re.I)
    song = new_song(title, path.stem.removeprefix("EAR-KEY-"))
    song["sources"].append(str(path))
    heard = set()
    timeline = False
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if not line:
                timeline = False
            continue
        source = f"{path}:{number}"
        m = re.match(r"(?:[-*]\s*)?(KEY|MODE|METER|TEMPO|INSTRUMENTS|HEARD)\s*:\s*(.+)",
                     line.replace("**", ""), re.I)
        if m:
            label, value = m[1].lower(), m[2]
            parsed = None
            if label == "key":
                # Only the initial assertion; later prose can describe a wrong relative key.
                assertion = re.split(r"[.;]", value)[0]
                parsed = key_value(value) if KEY_RE.match(value) and not re.search(
                    r"unclear|ambiguous|likely|lean|\bor\b", assertion, re.I) else None
                if parsed:
                    put_truth(song, "key", parsed, source, line)
                    put_truth(song, "mode", parsed[1], source, line)
            elif label == "mode":
                q = re.match(r"(major|minor|mixolydian|blues-rock)\b", value, re.I)
                parsed = q[0].lower() if q else None
                if parsed:
                    put_truth(song, "mode", parsed, source, line)
            elif label == "meter":
                q = re.match(r"(\d+/(?:2|4|8|16|32))\b", value)
                if UNCLEAR.search(value.split(",")[0]):
                    q = None
                parsed = {"signatures": [q[1]]} if q else None
                if parsed:
                    put_truth(song, "meter", parsed, source, line)
                    if len(meters(value)) > 1:
                        song["skipped"].append(f"{source}: meter feel/alternatives retained as context, not extra meters")
            elif label == "tempo":
                q = re.match(r"(?:~|≈)?\s*(\d+(?:\.\d+)?)\s*(?:BPM)?\b", value, re.I)
                if re.search(r"unclear|likely|maybe|\bor\b|\bvs\b", re.split(r"[;,]", value)[0], re.I):
                    q = None
                parsed = float(q[1]) if q and float(q[1]) > 0 else None
                if parsed:
                    put_truth(song, "tempo", parsed, source, line)
            else:
                parsed = instruments_in(value)
                if parsed and not re.search(r"unclear|likely|maybe|\bno\b|\bnot\b|\bor\b|\?|/", value, re.I):
                    heard.update(parsed)
                else:
                    parsed = None
            if not parsed:
                song["skipped"].append(f"{source}: ambiguous {label}: {value}")
            continue
        if re.search(r"timeline\s*:", line, re.I) and not line.startswith("Ears"):
            timeline = True
            continue
        if timeline and line.startswith("- "):
            positive = not re.search(r"\b(?:likely|unclear|maybe|missed)\b|not heard|\bno \w+", line, re.I)
            if positive:
                heard.update(instruments_in(line))
            song["skipped"].append(f"{source}: timing/event detail not scored")
        elif re.match(r"(?:[^:]+['’]s (?:favourite moment|detail)|Heard)\b", line, re.I):
            # Positive instrument names in the listener's own quoted observation only.
            if not re.search(r"\b(?:likely|unclear|maybe|no|not)\b", line, re.I):
                heard.update(instruments_in(line))
            song["skipped"].append(f"{source}: quoted event timing not scored")
        elif line.startswith("Credits"):
            song["skipped"].append(f"{source}: credits are not heard-instrument truth")
        else:
            song["skipped"].append(f"{source}: narrative/continuation not parsed")
    if heard:
        put_truth(song, "instruments", sorted(heard), path, "positive listener observations only; partial inventory")
    return song


def parse_meter_table(path, issues):
    text = read_text(path, issues)
    if text is None:
        return []
    songs = []
    for number, line in enumerate(text.splitlines(), 1):
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 4 or not cells[0].isdigit():
            continue
        label, assertion, status = cells[1:4]
        title = re.sub(r"\s*\([^)]*\)\s*$", "", label)
        song = new_song(title)
        song["sources"].append(f"{path}:{number}")
        if "UNVERIFIED" in status or re.search(r"no creator|contested|6\.5", assertion, re.I):
            song["skipped"].append(f"{path}:{number}: {status}; no settled meter truth: {assertion}")
        else:
            # Parenthetical attributions are not part of the assertion.
            value = meters(re.sub(r"\([^)]*\)", "", assertion))
            pulse = 5 if "quintuple" in assertion else 6 if '"in six"' in assertion else None
            if value or pulse:
                put_truth(song, "meter", {"signatures": value, "pulses": pulse,
                                         "partial": bool(re.search(r"verses|riff|solo|chorus", assertion))},
                          f"{path}:{number}", assertion, status)
            else:
                song["skipped"].append(f"{path}:{number}: meter assertion not parsed: {assertion}")
        songs.append(song)
    if "## Per-song notes" in text:
        issues.append(f"{path}: per-song prose and contested appendix skipped; table assertions only")
    return songs


def find_match(aliases, paths):
    """Exact identity first, then unique whole-token containment; never choose a fuzzy tie."""
    aliases = {norm(a) for a in aliases if a}
    exact = [p for p in paths if norm(p.stem) in aliases]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, "ambiguous match: " + ", ".join(str(p) for p in exact)
    candidates = [p for p in paths if any(contains(a, norm(p.stem)) or contains(norm(p.stem), a)
                                         for a in aliases)]
    # Do not silently match an original recording to a live/remix/version recording.
    def recording(s):
        return {w for w in ("live", "remix", "version", "acoustic", "instrumental", "cover", "demo", "edit")
                if w in norm(s).split()}
    candidates = [p for p in candidates if any(recording(a) == recording(p.stem) for a in aliases)]
    if len(candidates) == 1:
        return candidates[0], None
    return None, ("ambiguous match: " + ", ".join(str(p) for p in candidates)) if candidates else None


def load_shelf(music_dir, issues):
    songs = [parse_ear_key(p, issues) for p in sorted(music_dir.rglob("EAR-KEY-*.md"))]
    for path in sorted(music_dir.rglob("METER-TEST-SET.md")):
        songs.extend(parse_meter_table(path, issues))
    for folder, dimension in (("lyrics", "lyrics"), ("credits", "credits")):
        for path in sorted((music_dir / folder).glob("*.txt")):
            existing, reason = find_match([path.stem], [Path(s["title"]) for s in songs])
            song = next((s for s in songs if existing and s["title"] == str(existing)), None)
            if song is None:
                song = new_song(path.stem)
                songs.append(song)
            song["sources"].append(str(path))
            if reason:
                song["skipped"].append(reason)
            text = read_text(path, issues)
            if text is None:
                continue
            if dimension == "lyrics":
                clean = lyric_text(text)
                if words(clean):
                    # Keep words in memory for scoring; snapshots store only a digest and count.
                    put_truth(song, "lyrics", {"text": clean}, path, "local lyrics/ reference (official as supplied)", "supplied lyrics")
                else:
                    song["skipped"].append(f"{path}: empty lyrics reference")
            else:
                song["credits"] = instruments_in("\n".join(l.split(":", 1)[-1] for l in text.splitlines() if not l.startswith("#")))
                song["skipped"].append(f"{path}: credited inventory shown separately; no claim that every credit was heard")
    merged = {}
    for song in songs:
        if song["id"] not in merged:
            merged[song["id"]] = song
            continue
        prior = merged[song["id"]]
        prior["aliases"].extend(song["aliases"])
        prior["sources"].extend(song["sources"])
        prior["skipped"].extend(song["skipped"])
        for dim, truth in song["truth"].items():
            if "conflict" in truth:
                prior["truth"][dim] = truth
            else:
                put_truth(prior, dim, truth["value"], truth["source"], truth["evidence"], truth["status"])
    return sorted(merged.values(), key=lambda s: s["id"])


def lyric_text(text):
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") or re.fullmatch(r"\s*\[[^]]*\]\s*", line):
            continue
        line = re.sub(r"^\s*\[\d+:\d+(?:\.\d+)?\]\s*", "", line)
        lines.append(line)
    return "\n".join(lines)


def words(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text, re.UNICODE)


def word_error(ref, hyp):
    """Levenshtein WER with linear memory; insertions can make WER greater than 1."""
    a, b = words(ref), words(hyp)
    if not a:
        return None
    prev = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        curr = [i]
        for j, other in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (token != other)))
        prev = curr
    return {"errors": prev[-1], "reference_words": len(a), "hypothesis_words": len(b),
            "wer": prev[-1] / len(a), "word_accuracy": 1 - prev[-1] / len(a)}


def parse_card(text):
    result = {}
    key_lines = re.findall(r"^\s*KEY:\s*(.+)$", text, re.M)
    key_lines += re.findall(r"^\s*home:\s*(.+?)(?:;|$)", text, re.M)
    if key_lines:
        values = [key_value(s) if not UNCLEAR.search(s) else None for s in key_lines]
        result["key"] = {"value": values[0], "raw": key_lines[0]}
        if any(v != values[0] for v in values):
            result["key"] = {"value": None, "raw": "contradictory key lines: " + " / ".join(key_lines)}
    mt = re.search(r"^\s*METER:\s*(.+)$", text, re.M)
    if mt:
        result["meter"] = meter_estimate(mt[1])
    tp = re.search(r"^\s*TEMPO:\s*(\d+(?:\.\d+)?)\s*BPM(.*)$", text, re.M)
    if tp:
        felt = re.search(r"felt beat(?: likely)?\s*[~≈]?\s*(\d+(?:\.\d+)?)", tp[2])
        result["tempo"] = {"value": float(tp[1]), "felt": float(felt[1]) if felt else None, "raw": tp[0].strip()}
    # Only the tagger's overall block: credits, timelines and energy-only stems do not become names heard.
    block = re.search(r"^INSTRUMENTS[^\n]*\n(.*?)(?=^\S|\Z)", text, re.M | re.S)
    if block:
        labels, raw = [], []
        for line in block[1].splitlines():
            if "who is playing" in line:
                break
            if re.search(r"energy only|not nameable|credited.but.unheard|not heard|no tag cleared", line, re.I):
                continue
            if re.search(r"\(score |the ears heard", line):
                label = re.split(r"\s{2,}|\s+the ears heard", line.strip())[0]
                raw.append(label)
                labels.extend(named_instruments(label))
        result["instruments"] = {"value": sorted(set(labels)), "raw": raw}
        if re.search(r"unavailable|failed", block[1], re.I):
            result["instruments"] = {"value": None, "raw": "instrument backend unavailable"}
    vocal = re.search(r"^\s*VOCALS:\s*yes, voice present", text, re.M)
    if vocal:
        inventory = result.setdefault("instruments", {"value": [], "raw": [], "partial": True})
        if inventory["value"] is not None:
            inventory["value"] = sorted(set(inventory["value"]) | {"voice"})
            inventory["raw"].append("voice present (vocal-stem energy)")
    # MUSIC lyrics are raw Whisper. Never score placed/aligned official words as transcription.
    block = re.search(r"^\s*LYRICS:[^\n]*\n((?:[ \t]+[^\n]*\n?)*)", text, re.M)
    if block:
        lines = re.findall(r"^\s*\[\d+:\d+\]\s*(.+)$", block[1], re.M)
        if lines and not re.search(r"omitted|placed|aligned|≈|official", block[0], re.I):
            clean = " ".join(re.sub(r"\s*\(unsure:.*?\)", "", s) for s in lines)
            result["lyrics"] = {"value": clean, "raw": "card MUSIC Whisper lines"}
    return result


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_analysis(song, audio_files, cards, cache_dir, issues):
    result, used = {}, {}
    audio, reason = find_match(song["aliases"], audio_files)
    if reason:
        issues.append(f"{song['title']}: {reason}")
    aliases = song["aliases"] + ([audio.stem] if audio else [])
    # Group .txt/.json siblings as one identity; the text card is the public claim.
    identities = {}
    for path in cards:
        if path.stem not in identities or path.suffix == ".txt":
            identities[path.stem] = path
    card, reason = find_match(aliases, list(identities.values()))
    if reason:
        issues.append(f"{song['title']}: {reason}")
    if card:
        if card.suffix == ".txt":
            text = read_text(card, issues)
            if text is not None:
                result.update(parse_card(text))
                used.update({k: str(card) for k in result})
        else:
            data = read_json(card, issues)
            if data:
                result.update(parse_analysis(data))
                used.update({k: str(card) for k in result})
    if audio:
        try:
            sha = file_sha(audio)
        except OSError as e:
            issues.append(f"{audio}: cannot hash ({type(e).__name__})")
            return result, used
        stem = cache_dir / "raven-stems" / (sha[:16] + "-a0")
        for dimension, name in (("meter", "meter.json"), ("instruments", "instruments.json")):
            p = stem / name
            if (dimension in result and not result[dimension].get("partial")) or not p.exists():
                continue
            data = read_json(p, issues)
            if data is not None:
                if dimension == "meter":
                    claim = data.get("claim")
                    result[dimension] = meter_estimate(claim)
                else:
                    parsed = parse_instruments(data)
                    prior = result.get("instruments", {})
                    if parsed["instruments"]["value"] is not None:
                        parsed["instruments"]["value"] = sorted(set(parsed["instruments"]["value"]) | set(prior.get("value") or []))
                    result.update(parsed)
                used[dimension] = str(p)
        # Evaluate the latest saved transcription, not a claim about today's names configuration.
        variants = sorted((cache_dir / "raven-lyrics").glob(sha + "*.json"))
        if variants:
            p = max(variants, key=lambda p: (p.stat().st_mtime_ns, p.name))
            data = read_json(p, issues)
            if data is not None:
                parsed = parse_words(data)
                parsed["raw"] += f"; latest saved variant of {len(variants)}"
                result["lyrics"] = parsed
                used["lyrics"] = str(p)
            if len(variants) > 1:
                issues.append(f"{song['title']}: {len(variants)} lyric cache variants; selected latest saved {p.name}; names hints may differ")
        # JSON exports only. A CNN vector isn't the fused key, and feature pickles aren't verdicts.
        st = audio.stat()
        tag = hashlib.md5(f"{audio.absolute()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:12]
        export = cache_dir / "harmony" / f"{audio.name}-{tag}.json"
        if export.exists():
            data = read_json(export, issues)
            if data:
                for dim, value in parse_analysis(data).items():
                    if dim not in result:
                        result[dim], used[dim] = value, str(export)
    return result, used


def parse_words(data):
    supplied_notes = data.get("notes", [])
    if not isinstance(supplied_notes, list):
        return {"value": None, "raw": "malformed lyric cache notes"}
    notes = " ".join(str(x) for x in supplied_notes)
    entries = data.get("words")
    if re.search(r"failed|could not run|unavailable", notes, re.I) or not isinstance(entries, list):
        return {"value": None, "raw": "lyrics backend unavailable or malformed words"}
    tokens = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3 and isinstance(entry[0], str):
            tokens.append(entry[0])
        elif isinstance(entry, dict) and isinstance(entry.get("word"), str):
            tokens.append(entry["word"])
        else:
            return {"value": None, "raw": "malformed cached word"}
    return {"value": " ".join(tokens), "raw": "raw cached Whisper words"}


def parse_instruments(data):
    if data.get("error") or not isinstance(data.get("families"), dict):
        return {"instruments": {"value": None, "raw": "instrument backend unavailable or malformed families"}}
    labels, raw = set(), []
    for label, info in data["families"].items():
        if not isinstance(info, dict):
            return {"instruments": {"value": None, "raw": "malformed instrument family"}}
        if info.get("energy_only") or info.get("timbre_only") or info.get("credited_unheard"):
            continue
        raw.append(label)
        labels.update(named_instruments(label))
    return {"instruments": {"value": sorted(labels), "raw": raw}}


def parse_analysis(data):
    result = {}
    kv = data.get("key_verdict", data.get("key_claim"))
    if isinstance(kv, dict):
        raw = str(kv.get("line", ""))
        value = key_value(kv.get("key")) if kv.get("key") and not UNCLEAR.search(raw) else None
        result["key"] = {"value": value, "raw": raw or "cached key abstained"}
    elif isinstance(data.get("key"), dict):
        k = data["key"]
        result["key"] = {"value": key_value(k.get("key")) if not k.get("also") else None, "raw": "legacy spectral JSON key (not fused)"}
    if "meter_claim" in data:
        claim = data["meter_claim"]
        result["meter"] = meter_estimate(claim)
    if isinstance(data.get("tempo"), dict):
        value = data["tempo"].get("bpm")
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            result["tempo"] = {"value": value, "raw": "music JSON tempo"}
    if "words" in data:
        result["lyrics"] = parse_words(data)
    if "families" in data or "error" in data:
        result.update(parse_instruments(data))
    return result


def instrument_matches(truth, heard):
    if truth in heard:
        return True
    broad = {"guitar": {"electric guitar", "acoustic guitar"},
             "bass": {"bass guitar", "synth bass", "electric upright bass"},
             "drums": {"drum machine"}, "strings": {"violin", "cello"},
             "keyboards": {"piano", "organ", "synth"}}
    return bool(broad.get(truth, set()) & set(heard))


def score_dimension(dimension, truth, estimate):
    row = {"status": "unscorable: no truth", "score": None}
    if not truth or "conflict" in truth:
        if truth:
            row["status"] = "unscorable: conflicting truth"
        return row
    value = truth["value"]
    public_truth = value if dimension != "lyrics" else {"sha256": digest(words(value["text"])), "words": len(words(value["text"]))}
    row.update(truth=public_truth, truth_status=truth["status"], truth_source=truth["source"],
               truth_evidence=truth["evidence"], truth_id=digest([POLICY, public_truth]))
    if estimate is None:
        row["status"] = "unscorable: no cached analysis"
        return row
    prediction = estimate.get("value")
    row["call"] = estimate.get("raw")
    if prediction is None:
        row["status"] = "unscorable: abstained/unavailable (see call)"
        return row
    row["status"] = "scored"
    if dimension == "key":
        a, b = prediction, value
        relation = "other"
        if a == b:
            relation = "same"
        elif a[1] == b[1] and (a[0] - b[0]) % 12 in (5, 7):
            relation = "fifth"
        elif a[0] == b[0] and a[1] != b[1]:
            relation = "parallel"
        elif a[1] != b[1] and ((a[0] if a[1] == "major" else b[0]) -
                                     (a[0] if a[1] == "minor" else b[0])) % 12 == 3:
            relation = "relative"
        row.update(score={"same": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2, "other": 0.0}[relation], relation=relation)
    elif dimension == "mode":
        row["score"] = float(prediction[1] == value)
    elif dimension == "meter":
        if value.get("pulses"):
            row["score"] = float(any(int(m.split("/")[0]) == value["pulses"] for m in prediction))
            row["metric"] = "pulse-count family only; notation not settled"
        else:
            a, b = set(prediction), set(value["signatures"])
            partial = value.get("partial", False)
            row["score"] = len(a & b) / (len(b) if partial else len(a | b)) if b else 0.0
            row["metric"] = ("partial meter inventory coverage" if partial else "meter inventory intersection/union") + "; timing and grouping not scored"
    elif dimension == "tempo":
        def near(x, y):
            return abs(x - y) / y <= 0.03
        direct = near(prediction, value) or (estimate.get("felt") is not None and near(estimate["felt"], value))
        octave = near(prediction * 2, value) or near(prediction / 2, value)
        row.update(score=1.0 if direct else 0.5 if octave else 0.0,
                   metric="3% tolerance; named felt beat earns full credit; unqualified half/double earns 0.5")
    elif dimension == "instruments":
        matched = [name for name in value if instrument_matches(name, prediction)]
        row.update(score=len(matched) / len(value), matched=matched,
                   missed=[name for name in value if name not in matched],
                   metric="coverage of partial heard inventory; extra names are not judged false positives")
    elif dimension == "lyrics":
        stats = word_error(value["text"], prediction)
        row.update(stats)
        row["score"] = stats["word_accuracy"]
        row["call"] = str(estimate.get("raw", "raw Whisper transcription")) + " (text omitted from snapshot)"
        row["hypothesis_sha256"] = digest(words(prediction))
    return row


def evaluate(music_dir, cache_dir):
    issues = []
    songs = load_shelf(music_dir, issues)
    audio_files = sorted(p for p in music_dir.rglob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a", ".ogg"))
    cards_dir = cache_dir / "raven" / "cards"
    cards = sorted(p for p in cards_dir.glob("*") if p.suffix in (".txt", ".json"))
    rows = []
    for song in songs:
        estimates, sources = cached_analysis(song, audio_files, cards, cache_dir, issues)
        dims = {}
        for dimension in DIMENSIONS:
            estimate = estimates.get("key" if dimension == "mode" else dimension)
            dims[dimension] = score_dimension(dimension, song["truth"].get(dimension), estimate)
            dims[dimension]["analysis_source"] = sources.get("key" if dimension == "mode" else dimension)
        rows.append({"id": song["id"], "title": song["title"], "dimensions": dims,
                     "sources": song["sources"], "skipped": song["skipped"], "credits": song.get("credits", [])})
    return {"schema": SCHEMA, "policy": POLICY, "created_at": datetime.now(timezone.utc).isoformat(),
            "songs": rows, "issues": issues, "closing": "and now, you"}


def scoreboard(snapshot):
    lines = ["Ears answer shelf — existing calls only; source labels are not independently verified."]
    if not snapshot["songs"]:
        lines.append("No parseable answer keys or supplied lyrics/credits found.")
    for song in snapshot["songs"]:
        lines.append(f"\n{song['title']}")
        for dim, row in song["dimensions"].items():
            if row["score"] is None:
                text = row["status"]
            else:
                text = f"{row['score']:.3f} [{row['truth_status']}]"
                if row.get("relation"):
                    text += f" ({row['relation']})"
                if dim == "lyrics":
                    text += f" · WER {row['wer']:.3f}, {row['errors']}/{row['reference_words']} edits"
                if dim == "instruments":
                    text += " · heard-name coverage; missed: " + (", ".join(row["missed"]) or "none")
            lines.append(f"  {dim:<12} {text}")
            if row.get("truth_id"):
                lines.append(f"    truth: {row['truth_source']} [{row['truth_status']}]")
            if row.get("call") is not None and row["score"] is None:
                lines.append(f"    call: {row['call']}")
            if row["score"] is not None:
                lines.append(f"    call: {row.get('call')}; source: {row.get('analysis_source')}")
                if row.get("metric"):
                    lines.append(f"    metric: {row['metric']}")
        if song["credits"]:
            lines.append("  credited (not heard truth): " + ", ".join(song["credits"]))
        for skipped in song["skipped"]:
            lines.append("  skipped: " + skipped)
    lines.append("\nPer dimension (means over scored rows; missing truth and abstentions stay visible above):")
    for dim in DIMENSIONS:
        rows = [s["dimensions"][dim] for s in snapshot["songs"]]
        scores = [r["score"] for r in rows if r["score"] is not None]
        mean = f"{sum(scores) / len(scores):.3f}" if scores else "unscorable"
        lines.append(f"  {dim:<12} {mean}; {len(scores)} scored, {len(rows) - len(scores)} unscorable")
    for issue in snapshot["issues"]:
        lines.append("skipped: " + issue)
    return "\n".join(lines)


def compare_snapshots(current, previous):
    if previous.get("schema") != SCHEMA or previous.get("policy") != current["policy"]:
        return "Comparison unavailable: snapshot schema/scoring policy differs."
    old = {s["id"]: s for s in previous["songs"]}
    lines = [f"Comparison with {previous.get('created_at', 'undated snapshot')}:"]
    totals = {d: [] for d in DIMENSIONS}
    for song in current["songs"]:
        before = old.pop(song["id"], None)
        for dim, row in song["dimensions"].items():
            prior = before["dimensions"].get(dim) if before else None
            prefix = f"  {song['title']} / {dim}: "
            if prior is None:
                lines.append(prefix + "new row (no prior score)")
            elif row.get("truth_id") != prior.get("truth_id"):
                lines.append(prefix + "truth changed; scores not compared")
            elif row["score"] is None or prior["score"] is None:
                if row["status"] != prior["status"]:
                    lines.append(prefix + f"{prior['status']} → {row['status']}")
            else:
                delta = row["score"] - prior["score"]
                totals[dim].append(delta)
                if delta:
                    lines.append(prefix + f"{'better' if delta > 0 else 'worse'} {delta:+.3f} ({prior['score']:.3f} → {row['score']:.3f})")
    for song in old.values():
        lines.append(f"  {song['title']}: removed from shelf")
    for dim, deltas in totals.items():
        lines.append(f"  {dim}: mean change {sum(deltas) / len(deltas):+.3f} on {len(deltas)} paired rows" if deltas
                     else f"  {dim}: no comparable scored rows")
    return "\n".join(lines)


def latest_snapshot(directory, issues):
    for path in sorted(directory.glob("eval-*.json"), reverse=True):
        value = read_json(path, issues)
        valid = value and value.get("schema") == SCHEMA and isinstance(value.get("songs"), list)
        if valid:
            for song in value["songs"]:
                if (not isinstance(song, dict) or not isinstance(song.get("id"), str)
                        or not isinstance(song.get("title"), str) or not isinstance(song.get("dimensions"), dict)):
                    valid = False
                    break
                for dim in DIMENSIONS:
                    row = song["dimensions"].get(dim)
                    if not isinstance(row, dict) or not isinstance(row.get("status"), str) or "score" not in row:
                        valid = False
                        break
                    score = row["score"]
                    if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score)):
                        valid = False
                        break
        if valid:
            return value
        if value:
            issues.append(f"{path}: unsupported snapshot; skipped")
    return None


def save_snapshot(snapshot, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"eval-{stamp}.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--music-dir", type=Path, default=Path("example-music"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache", help="cache parent (contains raven/, raven-stems/, raven-lyrics/, harmony/)")
    parser.add_argument("--save", action="store_true", help="write a dated JSON under CACHE/raven/eval/")
    parser.add_argument("--compare", action="store_true", help="compare with last saved snapshot, before saving this run")
    try:
        args = parser.parse_args(argv)
        if not args.music_dir.is_dir():
            print(f"Shelf unavailable: {args.music_dir} is not a directory", file=sys.stderr)
            return 2
        snapshot = evaluate(args.music_dir, args.cache_dir)
        directory = args.cache_dir / "raven" / "eval"
        previous = latest_snapshot(directory, snapshot["issues"]) if args.compare else None
        print(scoreboard(snapshot))
        if args.compare:
            print("\n" + (compare_snapshots(snapshot, previous) if previous else "Comparison unavailable: no previous snapshot."))
        if args.save:
            print(f"\nSaved: {save_snapshot(snapshot, directory)}")
        return 0
    except (OSError, UnicodeError, ValueError, TypeError) as e:
        print(f"Evaluation unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    finally:
        print("and now, you")


if __name__ == "__main__":
    sys.exit(main())
