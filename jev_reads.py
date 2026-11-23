"""Optional TypeSafe text readings. Shared transport and lyric questions live here.

Only analysis calls read_lines; rendering consumes its saved results. No audio,
credentials or local paths belong in a payload or a diagnostic.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
import re
import socket
import textwrap
from urllib import error, request

import numpy as np

from network_policy import offline, require_online
import stage_status as ST

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
TIMEOUT = 20
MAX_WORKERS = 8
NOUL_YES = 0.7
NOUL_NO = 0.3
DISAGREEMENT = 0.6
EXPLANATION = "Jev = TypeSafe text model; each line judged in context; stated feeling -1..+1."

QUESTIONS = {
    "stated_feeling": {
        "type": "score",
        "instructions": "What feeling does the singer state outright in `line`, read in the context of `before` and `after`?",
        "criteria": [
            "The singer says they are hurting, despairing, furious or heartbroken.",
            "The singer says they are a little down, tired or bitter.",
            "The line states no feeling, or the feeling is mixed.",
            "The singer says they are content, hopeful or relieved.",
            "The singer says they are overjoyed, in love or elated.",
        ],
    },
    "masked": {
        "type": "noul",
        "instructions": "In `line`, does the singer claim to be fine, calm, unbothered or over it, while `before` and `after` show that they are not?",
        "criteria": {
            "true": "The singer says they're fine or calm, and the surrounding lines show hurt, longing or anger underneath.",
            "false": "The singer's stated feeling matches the surrounding lines, or the line makes no claim about how they feel.",
        },
    },
    "longing": {
        "type": "noul",
        "instructions": "In `line`, does the singer say they want someone or something they do not have yet, or no longer have?",
        "criteria": {
            "true": "The singer wants a person, a life or a moment that is out of reach right now.",
            "false": "The singer describes what they already have, or the line states no want.",
        },
    },
}


def enabled():
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def active():
    """New optional readings leave even notes/status untouched when offline."""
    return enabled() and not offline()


def read_tasks(tasks, notes, label):
    """Use the same transport, decoder and failure policy for other text questions.

    Call only during analysis. Callers save these rows (including wording/raw
    answers) under `jev`; saved rendering must never call this function.
    """
    if not active() or not tasks:
        return None
    payload_list = [dict(state=state, model=MODEL,
                         questions={n: _sent_form(q) for n, q in questions.items()})
                    for state, questions in tasks]
    key = os.environ["TYPESAFE_API_KEY"]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = list(pool.map(lambda p: _read(p, key), payload_list))
    notes.append(f"Jev: {len(rows)} {label} requests sent/attempted to TypeSafe; text only, never audio")
    failed = [r for r in rows if r["status"] != "ran"]
    reason = ""
    if failed:
        reason = f"{len(failed)}/{len(rows)} unavailable: " + "; ".join(dict.fromkeys(r["reason"] for r in failed))
        notes.append("Jev failed: " + reason)
    ST.mark("Jev " + label, "failed" if len(failed) == len(rows) else "degraded" if failed else "ran",
            reason, kind="input")
    return rows


def payloads(lines, song_title=""):
    texts = [line["text"] for line in lines]
    questions = {name: _sent_form(question) for name, question in QUESTIONS.items()}
    return [{"state": {"song_title": song_title or "",
                       "before": " ".join(texts[max(0, i - 2):i]),
                       "line": text, "after": " ".join(texts[i + 1:i + 3])},
             "model": MODEL, "questions": deepcopy(questions)} for i, text in enumerate(texts)]


def _number(value, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= high):
        raise ValueError("answer outside question scale")
    return float(value)


def decode(answers, questions=None):
    """Preserve raw answers separately; a missing/invalid answer fails the whole line."""
    values = {}
    for name, question in (QUESTIONS if questions is None else questions).items():
        answer = answers[name]
        kind = question["type"]
        if kind == "score":
            high = len(question["criteria"]) - 1
            values[name] = (_number(answer["score"], high) - high / 2) / (high / 2)
        elif kind == "noul":
            p = _number(answer["noul"], 1)
            values[name] = {"probability": p, "outcome":
                            "yes" if p >= NOUL_YES else "no" if p <= NOUL_NO else "not settled"}
        else:
            raise ValueError("unsupported question type")
    return values


def _post(payload, key):
    require_online("Jev text reading")
    req = request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with request.urlopen(req, timeout=TIMEOUT) as response:
        return json.load(response)


def _read(payload, key):
    row = {"payload": payload, "status": "unavailable"}
    try:
        response = _post(payload, key)
        json.dumps(response["answers"], allow_nan=False)
        row["answer"] = response["answers"]
        row.update(decode(row["answer"], payload["questions"]), status="ran")
    except error.HTTPError as exc:
        row["reason"] = f"HTTP {exc.code}"
    except (TimeoutError, socket.timeout):
        row["reason"] = "timeout"
    except error.URLError as exc:
        row["reason"] = "timeout" if isinstance(exc.reason, TimeoutError) else "connection error"
    except (ValueError, KeyError, TypeError, IndexError):
        row["reason"] = "malformed or missing answer"
    except Exception as exc:
        # Do not echo remote bodies, credentials, lyric text or local paths.
        row["reason"] = type(exc).__name__
    return row


def read_lines(lines, song_title="", notes=None):
    """One bounded request per line; outcomes are marked on the calling thread."""
    if not enabled() or not lines:
        return None
    notes = notes if notes is not None else []
    payload_list = payloads(lines, song_title)
    try:
        require_online("Jev lyric reading")
    except RuntimeError:
        reason = "offline mode (RAVEN_EARS_OFFLINE=1)"
        notes.append("Jev skipped: " + reason)
        ST.mark("Jev", "skipped", reason, kind="input")
        return [dict(status="skipped", reason=reason, payload=p) for p in payload_list]
    key = os.environ["TYPESAFE_API_KEY"]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = list(pool.map(lambda p: _read(p, key), payload_list))
    # Count attempts honestly: a transport error cannot confirm receipt by TypeSafe.
    notes.append(f"Jev: {len(rows)} lyric lines sent/attempted to TypeSafe (api.typesafe.ai); text and title only, never audio")
    failed = [r for r in rows if r["status"] != "ran"]
    if failed:
        reasons = "; ".join(dict.fromkeys(r["reason"] for r in failed))
        reason = f"{len(failed)}/{len(rows)} lines unavailable: {reasons}"
        notes.append("Jev failed: " + reason)
        ST.mark("Jev", "degraded" if len(failed) < len(rows) else "failed", reason, kind="input")
    else:
        ST.mark("Jev", kind="input")
    return rows


def add_contrast(lines, contrast, song_title="", notes=None):
    readings = read_lines(lines, song_title, notes)
    if readings is None:
        return contrast
    if not contrast:
        # Jev still has a reading when the independent roberta backend failed.
        contrast = [dict(t=l["t"], text=l["text"], words=None, music=None, lead=None) for l in lines]
    for row, reading in zip(contrast, readings):
        row["jev"] = reading
    return contrast


def display_reason(row):
    reading = row.get("jev") or {}
    if reading.get("status") != "ran":
        return ""
    reasons = []
    masked = reading["masked"]
    if masked["outcome"] == "yes":
        reasons.append(f"Jev reads masking ({masked['probability']:.2f})")
    words, stated = row.get("words"), reading["stated_feeling"]
    if words is not None and words * stated < 0 and abs(words - stated) > DISAGREEMENT:
        reasons.append("readings disagree")
    if reasons or row.get("flag"):
        for name, question in extra_nouls():
            value = saved_answer(reading, name, question)
            if value.get("outcome") == "yes":
                reasons.append(f"{noul_label(name, question)} ({value['probability']:.2f})")
    return "; ".join(reasons)


def _sent_form(question):
    return {k: v for k, v in question.items() if k in ("type", "instructions", "criteria")}


def saved_answer(reading, name, question):
    """A saved answer counts only if it answered TODAY's wording of the question.

    Editing a question's instructions or criteria under the same name must not
    make an old answer pose as an answer to the new question; it stays absent
    until a fresh listen asks again. Readings saved without payloads are trusted."""
    value = reading.get(name) or {}
    sent = ((reading.get("payload") or {}).get("questions") or {}).get(name)
    if value and sent is not None and sent != _sent_form(question):
        return {}
    return value


def extra_nouls():
    return ((name, q) for name, q in QUESTIONS.items()
            if name not in ("stated_feeling", "masked") and q["type"] == "noul")


def noul_label(name, question):
    return question.get("label") or name.replace("_", " ")


def summary_prefix(name, question):
    return f"Jev {question.get('verb') or 'hears'} {noul_label(name, question)} in "


def is_noul_summary(text):
    return any(text.startswith(summary_prefix(name, q)) for name, q in extra_nouls())


def noul_summaries(rows, stamp, width=116):
    """Count occurrences in question order; absent/removed answers do not qualify."""
    summaries = []
    for name, question in extra_nouls():
        yes = [r for r in rows if (r.get("jev") or {}).get("status") == "ran"
               and saved_answer(r["jev"], name, question).get("outcome") == "yes"]
        if not yes:
            continue
        first = min(yes, key=lambda r: r["t"])
        pre = summary_prefix(name, question) + f"{len(yes)} of {len(rows)} lines (first {stamp(first['t'])} “"
        text = " ".join(first["text"].split())
        room = max(1, width - len(pre) - len("”)"))
        if len(text) > room:
            text = text[:room - 1].rstrip() + "…"
        summaries.append(pre + text + "”)")
    return summaries


def flag_contrast(R):
    con = R["con"]
    for c in con:
        w, m = c["words"], c["music"]
        c["flag"] = None
        if w is None or m is None:
            continue
        if w <= -0.3 and m >= 0.3:
            c["flag"] = f"words heavy ({w:+.2f}), music lifting (lift {m:+.1f}, {c['lead']} leads)"
        elif w >= 0.3 and m <= -0.3:
            c["flag"] = f"words warm ({w:+.2f}), music weighing down (lift {m:+.1f}, {c['lead']} leads)"
        elif w <= -0.6 and m >= 0.2:
            c["flag"] = f"words heavy ({w:+.2f}), music not following (lift {m:+.1f})"
        elif w >= 0.6 and m <= -0.2:
            c["flag"] = f"words warm ({w:+.2f}), music not following (lift {m:+.1f})"
    return con


def block_contrast(R, stamp):
    L = ["CONTRAST (words vs music)"]
    con = flag_contrast(R)
    if not con:
        if R.get("lines") or R.get("words"):
            L.append("  unavailable (sentiment stage failed or unavailable, see NOTES)")
        elif R.get("words") is None:
            L.append("  unavailable (no usable lyric lines, see NOTES)")
        else:
            L.append("  no transcribed lyric lines, so no contrast to measure")
        return L
    L.append("  words = local sentiment model (cardiffnlp roberta, line blended with its neighbours, -1..+1); "
             "music 'lift' = song-relative z of energy, activity, brightness, harmonic rhythm, minus half of tension. "
             "Mode (major/minor) is NOT used; it's a fact, not a mood.")
    jev = any("jev" in c for c in con)
    if jev:
        L = [part for line in L for part in textwrap.wrap(line, width=118, subsequent_indent="    ", break_on_hyphens=False)]
        if any(c["jev"]["status"] != "skipped" for c in con if "jev" in c):
            L.append("  " + EXPLANATION)
        L.extend("  " + summary for summary in noul_summaries(con, stamp))
    fl = [c for c in con if c["flag"]]
    # collapse repeated identical lines (chorus repeats) to first + count
    seen = {}
    shown = [c for c in con if c["flag"] or (jev and display_reason(c))]
    for c in shown:
        # A chorus repeat collapses to its first time + "(also …)" even when Jev's value
        # moves by a hundredth between repeats; only a different reason splits it.
        key = c["text"].lower() if not jev else (c["text"].lower(), c["flag"], display_reason(c))
        seen.setdefault(key, []).append(c)
    if not shown and any(c["words"] is not None for c in con):
        L.append("  no line where words and music strongly disagree")
    for key, cs in sorted(seen.items(), key=lambda kv: kv[1][0]["t"]):
        c = cs[0]
        more = f"  (also {', '.join(stamp(x['t']) for x in cs[1:4])})" if len(cs) > 1 else ""
        if jev:
            L.append("  " + contrast_line(c, stamp(c['t'])))
            if c["flag"]:
                L += textwrap.wrap(c["flag"] + more, width=118, initial_indent="    ", subsequent_indent="    ", break_on_hyphens=False)
            elif more:
                L += textwrap.wrap(more.strip(), width=118, initial_indent="    ", subsequent_indent="    ", break_on_hyphens=False)
        else:
            L.append(f"  {stamp(c['t'])} '{c['text']}': {c['flag']}{more}")
    measured = [c for c in con if c["words"] is not None and c["music"] is not None]
    if not measured:
        L.append("  words vs music unavailable (sentiment stage failed or unavailable, see NOTES)")
        return L
    w = np.array([c["words"] for c in measured]); m = np.array([c["music"] for c in measured])
    r = float(np.corrcoef(w, m)[0, 1]) if len(measured) > 3 and w.std() > 0 and m.std() > 0 else 0.0
    mw = float(w.mean())
    wtxt = "lean heavy" if mw < -0.15 else ("lean warm" if mw > 0.15 else "sit near neutral")
    nf = len(fl)
    if r < -0.2:
        rel = "pull against each other: when the words go down the music goes up"
    elif r > 0.3:
        rel = "move together: the music rises and falls with what is being said"
    else:
        rel = "mostly run on separate tracks: the music's arc follows its own structure, not the sentences"
    summary = "  overall: " + ("roberta vs music — " if jev else "")
    L.append(f"{summary}the words {wtxt} ({mw:+.2f} avg); words and music {rel} (r={r:+.2f}); "
             f"{len(set(c['text'].lower() for c in fl))} distinct lines disagree strongly, out of {len(con)} lines")
    return [part for line in L for part in textwrap.wrap(line, width=118, subsequent_indent="    ", break_on_hyphens=False)] if jev else L


def refresh_contrast(text, data):
    """Render saved Jev answers with today's questions, without changing the bundle."""
    if not isinstance(data, dict):
        return text
    rows = data.get("contrast") or []
    if not any("jev" in r for r in rows):
        return text
    block = re.search(r"(?m)^CONTRAST[^\n]*\n(?:[ \t][^\n]*\n|\n)*", text + "\n")
    if not block:
        return text
    from slim import clock
    rendered = "\n".join(block_contrast({"con": deepcopy(rows)}, clock)) + "\n\n"
    return text[:block.start()] + rendered + text[block.end():]


def display_values(row):
    reading = row.get("jev") or {}
    words = row.get("words")
    values = f"words {words:+.2f} (roberta)" if words is not None else "roberta unavailable"
    return values + (f" · Jev {reading['stated_feeling']:+.2f}" if reading.get("status") == "ran"
                     else " · Jev " + reading.get("status", "unavailable"))


def contrast_line(row, stamp, width=116):
    """Keep values and reason together; shorten only the quoted lyric."""
    tail = display_values(row)
    if row.get("flag") and row.get("music") is not None:
        tail += f"; lift {row['music']:+.1f}"
    reason = display_reason(row)
    if reason:
        tail += "; " + reason
    pre = stamp + " “"
    room = max(1, width - len(pre) - len("” — ") - len(tail))
    text = " ".join(row["text"].split())
    if len(text) > room:
        text = text[:room - 1].rstrip() + "…"
    return pre + text + "” — " + tail
