"""--lyrics FILE: lay the WRITER's words on the audio (Nov 2026).

Whisper's word times are the ears; the given lyric lines are the text. Both are normalised (accents folded, case and
punctuation dropped) and aligned with Needleman-Wunsch over tokens, with a fuzzy token match so "remember" meets
"remembers". Lines with enough matched tokens take their time from those tokens ("heard"). Lines Whisper never got
(whispered, another language, missed) are placed between their heard neighbours and flagged approx (the card shows ≈):
honest, never presented as heard. Repeated choruses land on the right occurrence because the alignment is ordered.

The card prints the writer's text. Where Whisper heard something clearly different, the lyric rows carry its version
too (row["misheard"]) and misheard_block() prints heard vs wrote, one row per line.
Local only, no models: this module is plain numpy + difflib.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np

MATCH_SIM = 0.75        # token similarity at or above which two tokens count as the same word
_GAP, _MISS = -0.5, -0.6
_SMALL = set("i a an the and or but of to in on at is are was were be am my me you your we he she it they them his her "
             "our their this that with for from as so not no oh yeah la na".split())


# What does NOT count as a mishearing (the rule, in one place). Case, punctuation and accents are gone before we compare.
# A substituted word pair is ignored when it is: a colloquial spelling of the same word (gonna/going, cos/because),
# the same word with a contraction/plural tail on one side (he/he'd, run/runs: too short for the fuzzy match), or two vocables
# (oh/ooh/ah/mm/la/na/yeah). Everything else Whisper put where the writer wrote a different word is a mishearing.
# Words Whisper dropped or added are not mishearings, and neither is a line that was only placed (≈).
_SAME = {"gonna": "going", "wanna": "want", "gotta": "got", "cause": "because", "cos": "because", "cuz": "because",
         "coz": "because", "til": "until", "till": "until", "ok": "okay", "tho": "though", "thru": "through",
         "ya": "you", "em": "them"}
_VOCABLE = set("oh ooh oo ah aah aa mm mmm hm hmm uh um la na yeah yea ye hey ha ho woah whoa".split())
CONTEXT_WORDS = 2       # words kept either side of a differing span in the MISHEARD block
MAX_LINE = 118


def _trivial(a, b):
    if _SAME.get(a, a) == _SAME.get(b, b):
        return True
    if a in _VOCABLE and b in _VOCABLE:
        return True
    lo, hi = sorted((a, b), key=len)
    # "he'd" is folded to "hed": a bare "he" (or "run"/"runs", "it"/"its") is the same word, not a mishearing
    return len(lo) >= 2 and hi.startswith(lo) and hi[len(lo):] in ("s", "d", "ll", "ve", "re", "m", "es")


def read_lyrics_file(path):
    """Plain text, one sung line per line; blank lines ignored."""
    with open(path, encoding="utf-8-sig") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _fold(t):
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c)).lower().replace("’", "").replace("'", "")


def norm_tokens(text):
    return [x for x in re.split(r"[^a-z0-9]+", _fold(text)) if x]


_MEMO = {}


def _sim(a, b):
    if a == b:
        return 1.0
    if len(a) < 4 or len(b) < 4:
        return 0.0
    k = (a, b)
    v = _MEMO.get(k)
    if v is None:
        v = difflib.SequenceMatcher(None, a, b).ratio() if (a[0] == b[0] or abs(len(a) - len(b)) <= 2) else 0.0
        _MEMO[k] = v
    return v


def _nw(A, B):
    """Global alignment of token lists A (lyrics) and B (heard). Returns [(i, j, sim)] for paired positions; a pair
    under MATCH_SIM is a substitution: the ear heard *something* there, just not this word."""
    n, m = len(A), len(B)
    S = np.zeros((n + 1, m + 1))
    S[:, 0] = np.arange(n + 1) * _GAP
    S[0, :] = np.arange(m + 1) * _GAP
    P = np.zeros((n + 1, m + 1), dtype=np.int8)      # 0 diagonal, 1 lyric token unheard, 2 heard token extra
    P[1:, 0], P[0, 1:] = 1, 2
    for i in range(1, n + 1):
        ai, row, prev = A[i - 1], S[i], S[i - 1]
        for j in range(1, m + 1):
            sm = _sim(ai, B[j - 1])
            d = prev[j - 1] + ((1.0 + sm) if sm >= MATCH_SIM else _MISS)
            u = prev[j] + _GAP
            lf = row[j - 1] + _GAP
            if d >= u and d >= lf:
                row[j], P[i, j] = d, 0
            elif u >= lf:
                row[j], P[i, j] = u, 1
            else:
                row[j], P[i, j] = lf, 2
    i, j, out = n, m, []
    while i > 0 or j > 0:
        p = P[i, j]
        if p == 0:
            out.append((i - 1, j - 1, _sim(A[i - 1], B[j - 1])))
            i, j = i - 1, j - 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def _edge(t):
    return t.strip(" .,;:!?\"“”‘’()[]—–-…")


def _misheard(line, pairs, m, toks, B, bt, bw, hw):
    """None, or {heard, wrote, heard_core, wrote_core} for a HEARD line where Whisper put a different word than the
    writer's. pairs: [(token index in line, heard token index, sim)]; m: the line's matched tokens (for its time span).
    Only substituted pairs count, only where the heard word sits inside the line's own matched span (a shaky alignment
    is never read as a mishearing), and never the trivial kind (_trivial)."""
    words = line.split()
    tokw = [k for k, w in enumerate(words) for _ in norm_tokens(w)]       # token index in line -> word index in line
    if not m or len(tokw) != len(toks):
        return None
    t0, t1 = min(x[1] for x in m) - 2.0, max(x[2] for x in m) + 2.0
    inside = [(i, j, sm) for i, j, sm in pairs if t0 <= bt[j][0] <= t1]
    subs = [(i, j) for i, j, sm in inside if sm < MATCH_SIM and not _trivial(toks[i], B[j])]
    if not subs:
        return None
    w0, w1 = min(tokw[i] for i, _ in subs), max(tokw[i] for i, _ in subs)
    c0, c1 = max(0, w0 - CONTEXT_WORDS), min(len(words) - 1, w1 + CONTEXT_WORDS)
    h0, h1 = min(bw[j] for _, j in subs), max(bw[j] for _, j in subs)
    ctx = [bw[j] for i, j, _ in inside if c0 <= tokw[i] <= c1]
    a, b = min(ctx + [h0]), max(ctx + [h1])
    lead, tail = ("… " if c0 > 0 else ""), (" …" if c1 < len(words) - 1 else "")
    return {"heard": lead + " ".join(hw[a:b + 1]) + tail, "wrote": lead + " ".join(words[c0:c1 + 1]) + tail,
            "heard_core": _edge(" ".join(hw[h0:h1 + 1])), "wrote_core": _edge(" ".join(words[w0:w1 + 1]))}


def misheard_lines(rows):
    """Rows (this module's, or saved JSON) with a mishearing, in time order."""
    return sorted((r for r in rows or [] if isinstance(r, dict) and r.get("misheard")), key=lambda r: r["t"])


def _wrap_row(stamp, heard, wrote, width):
    import textwrap
    text = f'heard "{heard}"   wrote "{wrote}"'
    pad = " " * (len(stamp) + 1)
    return textwrap.wrap(text, width=width, initial_indent=stamp + "  ", subsequent_indent=pad + "   ",
                         break_long_words=True, break_on_hyphens=False)


def misheard_block(rows, clock, width=MAX_LINE):
    """The MISHEARD block as card lines (empty when nothing differs). `clock` formats seconds as m:ss."""
    bad = misheard_lines(rows)
    if not bad:
        return []
    out = ["MISHEARD (Whisper vs the writer's words)"]
    for r in bad:
        out += _wrap_row("  " + clock(r["t"]), r["misheard"]["heard"].replace('"', "'"),
                         r["misheard"]["wrote"].replace('"', "'"), width)
        if r["misheard"].get("jev"):
            import jev_more
            label = jev_more.misheard_label(r["misheard"]["jev"])
            if label:
                out.append("    " + label)
    unheard = sum(1 for r in rows if isinstance(r, dict) and r.get("approx"))
    if unheard:
        out.append(f"  +{unheard} line{'s' if unheard != 1 else ''} Whisper didn't hear (placed ≈, not counted as mishearings)")
    return out


def _distance(a, b):
    """Levenshtein distance over the letters only (case and punctuation ignored)."""
    a, b = "".join(norm_tokens(a)), "".join(norm_tokens(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def feel_line(rows, width=MAX_LINE):
    """One line for the feel card, or None. Example: the line whose heard and written words differ most (letters), so it
    leads with the funniest, not the tiniest; ties go to the shorter. Only examples that fit `width` unless none do."""
    bad = misheard_lines(rows)
    if not bad:
        return None

    def line(r):
        c, (m, s) = r["misheard"], divmod(int(round(r["t"])), 60)
        q = lambda x: x.replace(chr(34), chr(39))
        return (f"MISHEARD: {len(bad)} line{'s' if len(bad) != 1 else ''} · e.g. {m}:{s:02d} heard "
                f"\"{q(c['heard_core'])}\" for \"{q(c['wrote_core'])}\"")
    fits = [r for r in bad if len(line(r)) <= width] or bad
    best = max(fits, key=lambda r: (_distance(r["misheard"]["heard_core"], r["misheard"]["wrote_core"]),
                                    -len(r["misheard"]["heard_core"]) - len(r["misheard"]["wrote_core"])))
    return line(best)


def header_indices(rows, title=None, filename=None):
    """Unheard sheet furniture; a matched sung line always wins over its shape."""
    import os
    title = norm_tokens(title or "")
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    filenames = [norm_tokens(stem)]
    if "-" in stem:
        filenames.append(norm_tokens(stem.split(" - " if " - " in stem else "-", 1)[1]))
    first_heard = next((i for i, r in enumerate(rows) if not r["approx"]), len(rows))
    headers = []
    for i, row in enumerate(rows):
        text = row["text"].strip()
        if not row["approx"]:
            continue
        tokens = norm_tokens(text)
        is_title = bool(tokens) and (tokens == title or tokens in filenames)
        if (text.startswith("[") and text.endswith("]") or
                i < first_heard and (is_title or text.startswith("(") and text.endswith(")"))):
            headers.append(i)
    return headers


def align_lyrics(lines, words, dur=None, title=None, filename=None, headers=True):
    """lines: [str], the writer's lines. words: [(word, start, end)] from Whisper.
    Returns (rows, summary). row: {t, end, text, approx, differs, matched, tokens}.
    summary: {total, heard, approx, differs}."""
    L = [norm_tokens(x) for x in lines]
    A, owner, first = [], [], []
    for li, tk in enumerate(L):
        first.append(len(A))
        for x in tk:
            A.append(x)
            owner.append(li)
    B, bt, bw, hw = [], [], [], []
    for w, s, e in words or []:
        hw.append(str(w).strip())
        pcs = norm_tokens(w)
        e = min(e, s + max(1.0, 0.3 + 0.1 * len(w)))     # Whisper stretches a word over a following hum or silence
        for k, pc in enumerate(pcs):
            d = (e - s) / len(pcs)
            B.append(pc)
            bt.append((s + k * d, s + (k + 1) * d))
            bw.append(len(hw) - 1)
    pairs = _nw(A, B) if A and B else []
    per = [{"m": [], "sub": 0, "pairs": []} for _ in lines]
    for i, j, sm in pairs:
        per[owner[i]]["pairs"].append((i - first[owner[i]], j, sm))
        if sm >= MATCH_SIM:
            per[owner[i]]["m"].append((i, bt[j][0], bt[j][1]))
        else:
            per[owner[i]]["sub"] += 1
    rows = []
    for li, ln in enumerate(lines):
        n = len(L[li])
        m = per[li]["m"]
        if len(m) > 1:                          # drop a stray match far from the rest of the line
            cl, cur = [], [m[0]]
            for x in m[1:]:
                if x[1] - cur[-1][2] > 5.0:
                    cl.append(cur)
                    cur = []
                cur.append(x)
            cl.append(cur)
            m = max(cl, key=len)
        frac = len(m) / n if n else 0.0
        content = [x for x in m if A[x[0]] not in _SMALL]
        heard = bool(m) and frac >= 0.4 and (n > 3 or frac >= 0.5) and (bool(content) or frac >= 0.6)
        rows.append({"text": ln, "tokens": n, "matched": len(m), "approx": not heard,
                     "differs": n > 0 and per[li]["sub"] >= max(1, 0.4 * n),
                     "misheard": _misheard(ln, per[li]["pairs"], m, A[first[li]:first[li] + n], B, bt, bw, hw)
                     if heard else None,
                     "_m": m if heard else [], "t": None, "end": None})
    if headers:
        omitted = set(header_indices(rows, title, filename))
        if omitted:
            # Re-align without furniture so it cannot steal words or placement time.
            return align_lyrics([ln for i, ln in enumerate(lines) if i not in omitted], words, dur,
                                headers=False)
    sp = [(max(x[2] for x in r["_m"]) - min(x[1] for x in r["_m"])) / r["matched"]
          for r in rows if not r["approx"] and r["matched"] > 1]
    spt = float(np.clip(np.median(sp), 0.2, 0.8)) if sp else 0.4      # seconds per token, from this song's own heard lines
    last_end = 0.0
    for li, r in enumerate(rows):
        if r["approx"]:
            continue
        m = r["_m"]
        t0, t1 = min(x[1] for x in m), max(x[2] for x in m)
        lead = min(x[0] for x in m) - first[li]
        trail = first[li] + r["tokens"] - 1 - max(x[0] for x in m)
        r["t"] = max(min(last_end, t0), t0 - min(1.5, lead * spt))
        r["end"] = t1 + min(1.5, trail * spt)
        last_end = max(last_end, r["end"])
    i = 0
    while i < len(rows):                        # place the unheard runs between their heard neighbours
        if not rows[i]["approx"]:
            i += 1
            continue
        j = i
        while j < len(rows) and rows[j]["approx"]:
            j += 1
        run = rows[i:j]
        need = [max(1, r["tokens"]) * spt for r in run]
        tot = sum(need)
        a = rows[i - 1]["end"] if i > 0 else None
        b = rows[j]["t"] if j < len(rows) else None
        if a is None and b is None:
            a, b = 0.0, (dur or tot)
        elif a is None:
            a = max(0.0, b - tot - 0.3)
        elif b is None:
            b = a + tot + 0.3
            if dur:
                b = min(dur, b)
        span = max(b - a, 0.2)
        if span >= tot * 1.8:                   # roomy gap: do not stretch, centre the run in it
            cur, scale = a + (span - tot) / 2, 1.0
        else:
            cur, scale = a, span / tot
        for r, nd in zip(run, need):
            r["t"], r["end"] = cur, cur + nd * scale
            cur += nd * scale
        i = j
    out = [{"t": round(r["t"], 1), "end": round(max(r["end"], r["t"]), 1), "text": r["text"], "approx": r["approx"],
            "differs": r["differs"], "matched": r["matched"], "tokens": r["tokens"], "misheard": r["misheard"]}
           for r in rows]
    # One place decides this fact: the summary's "differs" count is the MISHEARD rule (_misheard), not the older per-row
    # flag (>=40% of a line's tokens substituted), which stays on the rows and is not used for any decision.
    summ = {"total": len(out), "approx": sum(r["approx"] for r in out), "differs": sum(bool(r["misheard"]) for r in out)}
    summ["heard"] = summ["total"] - summ["approx"]
    return out, summ


def summary_line(s):
    txt = f"LYRICS: writer's text, {s['heard']} of {s['total']} lines heard"
    if s["approx"]:
        txt += f", {s['approx']} placed between neighbours (≈)"
    if s["differs"]:
        txt += f", {s['differs']} where Whisper heard different words (writer's kept)"
    return txt


def clean_writer_card(text, data, title=None, filename=None):
    """Read old saved writer cards without presenting sheet headers as sung rows.

    Keep saved timing/evidence for the remaining rows. No audio or store writes.
    """
    if not isinstance(data, dict):
        return text, data
    source = data.get("lyrics_source") or {}
    if not source and isinstance(data.get("lyrics"), dict):
        source = data["lyrics"].get("lookup") or {}
    if source.get("status") != "file":
        return text, data
    container, field, label_field = data, "lyrics", "lyrics_align"
    if isinstance(data.get("lyrics"), dict):
        container, field, label_field = data["lyrics"], "rows", "label"
    elif isinstance(data.get("lyrics_aligned"), dict):
        container, field, label_field = data["lyrics_aligned"], "lines", "label"
    rows = container.get(field) or []
    omitted = set(header_indices(rows, title, filename or data.get("file")))
    if not omitted:
        return text, data
    kept = [r for i, r in enumerate(rows) if i not in omitted]
    summary = dict(total=len(kept), heard=sum(not r["approx"] for r in kept),
                   approx=sum(r["approx"] for r in kept), differs=sum(bool(r.get("misheard")) for r in kept))
    old_label = container.get(label_field, "")
    old_differs = re.search(r", (\d+) where Whisper heard different words", old_label)
    summary["differs"] = (container.get("summary") or {}).get("differs",
                          int(old_differs[1]) if old_differs else summary["differs"])
    # Preserve saved mishearing claims and timing caveats; older rows may not
    # contain the word pairs needed to recompute those claims.
    new_label = re.sub(r"\d+ of \d+ lines heard", f"{summary['heard']} of {summary['total']} lines heard", old_label)
    new_label = re.sub(r", \d+ placed between neighbours \(≈\)",
                       f", {summary['approx']} placed between neighbours (≈)" if summary["approx"] else "", new_label)
    if old_label:
        text = text.replace(old_label, new_label)
    clock = lambda t: f"{max(0, int(round(t))) // 60}:{max(0, int(round(t))) % 60:02d}"
    headers = {(clock(rows[i]["t"]), rows[i]["text"]) for i in omitted}
    output = []
    aggregate_note = "word/music aggregate unavailable: saved analysis included lyric-sheet headers"
    removed_aggregate = False
    for line in text.splitlines():
        match = re.match(r"\s*\[(\d+:\d\d)≈\]\s*(.*)", line)
        if match and any(match[1] == t and (match[2] == h or match[2].startswith('"' + h + '"')) for t, h in headers):
            continue
        if any(line.startswith("ACHE " + t + " ") and 'under ≈ "' + h + '" (' in line for t, h in headers):
            continue
        if line.strip().startswith("overall: the words ") and f"out of {len(rows)} lines" in line:
            output.append("  NOTE: " + aggregate_note)
            removed_aggregate = True
            continue
        output.append(line)
    data = dict(data)
    changed = dict(container, **{field: kept, label_field: new_label})
    if "summary" in changed:
        changed["summary"] = summary
    if isinstance(data.get("lyrics"), dict):
        data["lyrics"] = changed
    elif isinstance(data.get("lyrics_aligned"), dict):
        data["lyrics_aligned"] = changed
    else:
        data.update(changed)
    if isinstance(data.get("ache"), dict):
        data["ache"] = dict(data["ache"], moments=[m for m in data["ache"].get("moments", [])
                            if (clock(m["t"]), m.get("lyric")) not in {(t, '≈ "' + h + '"') for t, h in headers}])
    if isinstance(data.get("contrast"), list):
        data["contrast"] = [r for r in data["contrast"] if (clock(r["t"]), r["text"]) not in headers]
    if removed_aggregate:
        data["notes"] = list(data.get("notes") or []) + [aggregate_note]
    return "\n".join(output) + ("\n" if text.endswith("\n") else ""), data


def quote_line(text, maxc=90):
    t = text.replace("“", "'").replace("”", "'").replace('"', "'")
    return '"' + (t if len(t) <= maxc else t[:maxc - 1].rstrip() + "…") + '"'
