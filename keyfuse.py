"""
keyfuse.py: ONE key claim for the card, from several independent voters.

Chroma and chord-fit statistics say WHICH NOTES are in play (the key signature) far better than WHICH ONE IS HOME (the
tonic): A major and F# minor share all seven notes. So the claim has two parts, and the card says both:
    KEY: A major (3♯) · tonic fairly clear
    KEY: 3♯ (A major or F# minor, same notes; tonic unclear)

Voters (each gives a log-probability over the 24 keys; index = tonic pc, 0..11 major then 12..23 minor, like music.py):
  spectral  the chroma template reading (music.py), softened
  cnn       madmom's CNN key model, run in its OWN venv (venv-madmom) as a subprocess, JSON out. Optional.
  chords    how likely the decoded chords are under each key (degree frequencies: I, IV, V, vi ... in major; i, VI, VII ... in minor)
  bass      the bass stem's pitch classes, time-weighted, downbeats and phrase/song ends counted extra
  melody    long vocal notes that end a phrase (resting notes)
  cadence   V->I and IV->I landings in the decoded chords; low weight, and flat when the song never lands (that is fine)
They are summed with fixed weights (WEIGHTS, hand-set; the shelf is far too small to fit them) and soft-maxed.

Licence note: the madmom CNN key model files are CC BY-NC-SA 4.0 (the code is BSD). Personal, non-commercial use only.
Local only: the subprocess reads the audio file on this machine and prints 24 numbers.
"""
import json
import os
import re
import subprocess
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_SIGNATURE_SPELLINGS = {"C# major": "Db major", "D# minor": "Eb minor", "Ab minor": "G# minor"}  # Eb minor (6 flats) over D# minor (6 sharps): the charts say Eb (Take Five, Fox 22 Nov)
MAJOR_KEYS = "Cb Gb Db Ab Eb Bb F C G D A E B F# C#".split()
MINOR_KEYS = "Abm Ebm Bbm Fm Cm Gm Dm Am Em Bm F#m C#m G#m D#m A#m".split()
MIXOLYDIAN_KEYS = MAJOR_KEYS[1:] + ["G#"]
NATURALS = dict(zip("CDEFGAB", (0, 2, 4, 5, 7, 9, 11)))
_ALTERATIONS = {"": 0, "#": 1, "♯": 1, "b": -1, "♭": -1}
_PITCH_NAME = r"[A-G][#b♯♭]?"
WEIGHTS = {"cnn": 10.0, "spectral": 0.5, "chords": 0.7, "bass": 0.3, "cadence": 0.2, "melody": 0.15}
CLEAR_P, CLEAR_MARGIN = 0.55, 0.25          # a "confident" tonic needs this much posterior and lead
VERY_CLEAR_P = 0.85
SPAN = 6.0                                    # no voter may put a key more than this many nats below its own favourite


def key_name(i):
    old = f"{NAMES[i % 12]} {'major' if i < 12 else 'minor'}"
    return _SIGNATURE_SPELLINGS.get(old, old)


def key_short(i):
    return _short_name(key_name(i))


def _short_name(name):
    tonic, mode = name.split()
    return tonic + ("m" if mode == "minor" else "")


def saved_key_claim(claim):
    """Respell a saved fused verdict without changing its pitch classes or votes."""
    if not isinstance(claim, dict) or "signature" not in claim:
        return claim
    updated = claim.copy()
    for field in ("working", "key", "spectral"):
        if claim.get(field) in _SIGNATURE_SPELLINGS:
            updated[field] = _SIGNATURE_SPELLINGS[claim[field]]
    if updated.get("working") != claim.get("working") and updated.get("working"):
        tonic, mode = updated["working"].split()
        updated["signature"] = sig_text(key_index(NATURALS[tonic[0]] + _ALTERATIONS[tonic[1:]], mode))
    if claim.get("line"):
        for old, new in _SIGNATURE_SPELLINGS.items():
            if old in updated["line"]:
                tonic, mode = new.split()
                index = key_index(NATURALS[tonic[0]] + _ALTERATIONS[tonic[1:]], mode)
                updated["line"] = re.sub(rf"{re.escape(old)}( \(\d[♯♭]\))?",
                                         lambda m: new + (f" ({sig_text(index)})" if m[1] else ""),
                                         updated["line"])
    if "top2" in claim:
        updated["top2"] = [_SIGNATURE_SPELLINGS.get(key, key) for key in claim["top2"]]
    if "voters" in claim:
        updated["voters"] = {voter: _SIGNATURE_SPELLINGS.get(key, key) for voter, key in claim["voters"].items()}
        if updated["voters"] != claim["voters"]:
            votes = " · ".join(f"{voter} {_short_name(key)}" for voter, key in updated["voters"].items())
            updated["votes"] = votes
            if claim.get("votes_line"):
                updated["votes_line"] = votes
    return claim if claim == updated else updated


def key_index(tonic, mode):
    return tonic % 12 + (0 if mode == "major" else 12)


def area(i):
    """The major key whose seven notes this key uses (C major and A minor are one area)."""
    return i if i < 12 else (i - 12 + 3) % 12


def signature(i):
    """Key signature of key i as (n, '♯'|'♭'|'')."""
    n = (7 * area(i)) % 12                    # sharps, going round the circle of fifths
    if n == 6:                                # six either way: follow the key's own spelling
        return (6, "♭" if "b" in key_name(i).split()[0] else "♯")
    if n < 6:
        return (n, "♯" if n else "")
    return (12 - n, "♭")


def sig_text(i):
    n, s = signature(i)
    return f"{n}{s}" if n else "no ♯/♭"


@lru_cache(maxsize=None)
def letter_signature(key):
    """Letter alterations for an ABC key."""
    if key and key.endswith("Mix"):
        fifths = MIXOLYDIAN_KEYS.index(key.removesuffix("Mix")) - 7
    else:
        fifths = (MINOR_KEYS if key and key.endswith("m") else MAJOR_KEYS).index(key or "C") - 7
    signature = dict.fromkeys(NATURALS, 0)
    for letter in ("FCGDAEB" if fifths > 0 else "BEADGCF")[:abs(fifths)]:
        signature[letter] = 1 if fifths > 0 else -1
    return signature


@lru_cache(maxsize=None)
def spell_pitch(pc, key, chord_letter=None):
    """Return (letter, alteration) in a conventional ABC key."""
    sig = letter_signature(key)
    letters = list(NATURALS)
    root = key[0]
    interval = (pc - NATURALS[root] - sig[root]) % 12
    # These chromatic degrees have common functional spellings. Other ties
    # prefer the signature's direction after allowing a natural cancellation.
    degree = ({6: 3, 9: 5, 11: 6} if key.endswith("m") else {6: 3, 10: 6}).get(interval)
    degree_letter = letters[(letters.index(root) + degree) % 7] if degree is not None else None
    direction = -1 if -1 in sig.values() else 1
    choices = []
    for letter, natural in NATURALS.items():
        for alteration in (-1, 0, 1):
            if (natural + alteration) % 12 == pc:
                cost = (alteration != sig[letter], letter != chord_letter, letter != degree_letter,
                        alteration != 0, alteration != direction)
                choices.append((cost, letter, alteration))
    _, letter, alteration = min(choices)
    return letter, alteration


def spelling_key(name):
    """Use the same economical enharmonic key as the fused KEY line."""
    match = re.fullmatch(r"([A-G])([#b♯♭]?) (major|minor|mixolydian)", name or "")
    if not match:
        return None
    pc = NATURALS[match[1]] + _ALTERATIONS[match[2]]
    if match[3] == "mixolydian":
        letter, alteration = spell_pitch(pc % 12, key_short((pc + 5) % 12))
        return letter + {-1: "b", 0: "", 1: "#"}[alteration] + "Mix"
    return key_short(key_index(pc, match[3]))


def claimed_spelling_key(claim):
    if not isinstance(claim, dict):
        return None
    if claim.get("state") in ("lean", "split"):
        return None
    if claim.get("state") == "tonic":
        # Relative keys share a signature. Spell chromatic degrees against
        # their relative major without choosing a tonic for the card.
        key = spelling_key(claim.get("working"))
        if key:
            return MAJOR_KEYS[MINOR_KEYS.index(key)] if key.endswith("m") else key
    return spelling_key(claim.get("key"))


def printed_spelling_key(text):
    for match in re.finditer(rf"(?m)^[ \t]*(?:KEY:|home:)[ \t]+"
                             rf"({_PITCH_NAME} (?:major|minor|mixolydian))([^\n]*)", text):
        if not re.search(r"unclear|unsure|working|low confidence|\blean\b", match[2], re.I):
            return spelling_key(match[1])
    return None


def spelling_spans(scan):
    """Use a reported destination's crop support for timed labels."""
    spans = []
    if (scan or {}).get("status") == "ran":
        for claim in scan.get("claims") or []:
            bounds = claim.get("destination_support")
            key = spelling_key(claim.get("to"))
            if key and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                spans.append([*bounds, key])
    return spans


def spelling_context(home, spans=()):
    def at(t=None):
        for start, end, key in reversed(spans):
            if t is not None and start <= t < end:
                return key
        return home
    return at


def card_spelling_keys(analyses, passes):
    def data(name):
        value = analyses.get(name)
        return value if isinstance(value, dict) and passes.get(name, {}).get("status", "ran") == "ran" else {}

    music, harmony = data("music"), data("harmony")
    music_claim = music.get("key_claim") or music.get("key")
    harmony_claim = harmony.get("key_verdict")
    return (claimed_spelling_key(music_claim), claimed_spelling_key(harmony_claim),
            spelling_spans(harmony.get("key_change_scan")))


def song_spelling_keys(song):
    return card_spelling_keys({name: card.get("json") for name, card in song["cards"].items()},
                              song["manifest"]["passes"])


_NOTE_NAME = _PITCH_NAME + r"-?\d+"
_CLOCK = r"\d+:\d\d"
_APPOGGIATURA = re.compile(
    rf"^(\s*({_CLOCK})\s+)({_PITCH_NAME})( held [\d.]+ s over )"
    rf"({_PITCH_NAME})(maj7|m7|7|m|°|5)?(, then (?:down|up) onto )({_PITCH_NAME})$")
_CHORD_STEPS = {
    "": {4: 2, 7: 4}, "m": {3: 2, 7: 4}, "°": {3: 2, 6: 4}, "5": {7: 4},
    "7": {4: 2, 7: 4, 10: 6}, "maj7": {4: 2, 7: 4, 11: 6}, "m7": {3: 2, 7: 4, 10: 6},
}


def _pitch_number(name):
    return NATURALS[name[0]] + _ALTERATIONS[name[1:]]


def respell_note(name, key, chord_letter=None):
    if key is None:
        return name
    pitch, octave = re.fullmatch(rf"({_PITCH_NAME})(-?\d+)?", name).groups()
    number = _pitch_number(pitch)
    letter, alt = spell_pitch(number % 12, key, chord_letter)
    result = letter + ("♯" if alt > 0 else "♭" if alt < 0 else "")
    if octave is not None:
        result += str(int(octave) + (number - NATURALS[letter] - alt) // 12)
    return result


def _seconds(clock):
    minute, second = map(int, clock.split(":"))
    return 60 * minute + second


def harmony_note_labels(text, at):
    """Refresh measured note labels in fresh and saved harmony prose."""
    def row(line):
        match = _APPOGGIATURA.fullmatch(line)
        if match:
            prefix, clock, held, over, root, quality, then, landed = match.groups()
            key = at(_seconds(clock))
            if key is None:
                return line
            quality = quality or ""
            chord = respell_note(root, key)
            interval = (_pitch_number(landed) - _pitch_number(root)) % 12
            step = _CHORD_STEPS[quality].get(interval)
            letters = list(NATURALS)
            letter = letters[(letters.index(chord[0]) + step) % 7] if step is not None else None
            return (prefix + respell_note(held, key) + over + chord + quality + then +
                    respell_note(landed, key, letter))
        pedal = re.match(rf"^(\s*(?:({_CLOCK})\s+)?pedal (?:point )?on )"
                         rf"({_PITCH_NAME})( in the bass (?:({_CLOCK})–|until ))", line)
        if pedal:
            key = at(_seconds(pedal[2] or pedal[5])) if (pedal[2] or pedal[5]) else at(None)
            return pedal[1] + respell_note(pedal[3], key) + line[pedal.end(3):]
        return line
    return "\n".join(row(line) for line in text.split("\n"))


def singer_note_labels(text, at):
    """Refresh only the SINGER pitch fields, leaving lyric text alone."""
    def replace(line):
        body = line.lstrip()
        if body.startswith(("highest held notes:", "lowest held notes:")):
            return re.sub(rf"({_NOTE_NAME})(?= at ({_CLOCK}))",
                          lambda m: respell_note(m[1], at(_seconds(m[2]))), line)
        if body.startswith(("pitch: median ", "pitch arc (median by section):")):
            return re.sub(_NOTE_NAME, lambda m: respell_note(m[0], at(None)), line)
        if body.startswith("most intense moment:") and "pitch there" in body:
            time = re.search(_CLOCK, body)
            key = at(_seconds(time[0])) if time else at()
            head, tail = line.split("pitch there", 1)
            return head + "pitch there" + re.sub(_NOTE_NAME, lambda m: respell_note(m[0], key), tail)
        return line
    return "\n".join(replace(line) for line in text.split("\n"))


def relation(a, b):
    """MIREX relation of estimate a to reference b: same / fifth / relative / parallel / other."""
    if a == b:
        return "same"
    ta, tb, ma, mb = a % 12, b % 12, a // 12, b // 12
    if ma == mb and (ta - tb) % 12 in (5, 7):
        return "fifth"
    if ma != mb:
        if ta == tb:
            return "parallel"
        maj, mnr = (ta, tb) if ma == 0 else (tb, ta)
        if (maj - mnr) % 12 == 3:
            return "relative"
    return "other"


def _norm(L, span=SPAN):
    """log-probabilities from any score vector: normalised, and floored `span` nats under the best."""
    L = np.asarray(L, float)
    L = L - (L.max() + np.log(np.exp(L - L.max()).sum()))
    return np.maximum(L, L.max() - span)


# --------------------------------------------------------------------------- voters
def v_spectral(scores):
    """`scores` = the 24 chroma-template correlations from music._key_scores."""
    return _norm(np.asarray(scores, float) / 0.06)


def madmom_to_ours(probs):
    """madmom orders its 24 outputs A, Bb, B, C ... G# (major) then A minor ... G# minor."""
    p = np.asarray(probs, float).reshape(-1)
    out = np.zeros(24)
    for j in range(12):
        out[(9 + j) % 12] = p[j]
        out[12 + (9 + j) % 12] = p[12 + j]
    return out


def v_cnn(probs):
    p = madmom_to_ours(probs)
    return _norm(0.5 * np.log(p + 0.005))


# chord degree frequencies: (semitones above the tonic, class) -> weight. Class M major/dom, m minor, d diminished.
_MAJ_DEG = {(0, "M"): .22, (7, "M"): .15, (5, "M"): .13, (9, "m"): .12, (2, "m"): .07, (4, "m"): .05, (11, "d"): .01,
            (2, "M"): .015, (10, "M"): .02, (3, "M"): .01, (8, "M"): .012, (5, "m"): .012, (7, "m"): .01, (4, "M"): .02,
            (9, "M"): .01, (0, "m"): .005}
_MIN_DEG = {(0, "m"): .22, (8, "M"): .12, (3, "M"): .11, (10, "M"): .11, (7, "M"): .08, (7, "m"): .05, (5, "m"): .10,
            (2, "d"): .01, (0, "M"): .01, (5, "M"): .02, (2, "m"): .01, (11, "d"): .015, (1, "M"): .01, (9, "d"): .005}
FLOOR = 0.003


def _deg_table(deg):
    t = np.full((12, 3), FLOOR)
    for (iv, c), w in deg.items():
        t[iv, "Mmd".index(c)] = w
    return np.log(t / t.sum())


_LOG_DEG = {"major": _deg_table(_MAJ_DEG), "minor": _deg_table(_MIN_DEG)}


def _cls(q):
    return {"maj": "M", "dom7": "M", "maj7": "M", "min": "m", "min7": "m", "dim": "d"}[q]


def v_chords(events, max_hold=6.0):
    """Likelihood of the decoded chord sequence under each key. A held chord counts for at most `max_hold` seconds; the
    evidence is scaled by how many DIFFERENT chords there are (a 3-chord loop is little evidence however long it plays).
    Returns (L, info) or (None, info) when there are no chords."""
    ch = [e for e in events if e.get("root") is not None]
    tot = sum(min(e["t1"] - e["t0"], max_hold) for e in ch)
    if tot <= 0:
        return None, {}
    distinct = len({(e["root"], _cls(e["q"])) for e in ch})
    L = np.zeros(24)
    for k in range(24):
        tonic, mode = k % 12, "major" if k < 12 else "minor"
        tab = _LOG_DEG[mode]
        s = 0.0
        for e in ch:
            s += min(e["t1"] - e["t0"], max_hold) * tab[(e["root"] - tonic) % 12, "Mmd".index(_cls(e["q"]))]
        L[k] = s / tot
    n_eff = min(20.0, 5.0 * distinct)
    return _norm(L * n_eff), {"distinct": distinct}


_MAJ_PROF = np.full(12, .01)
_MAJ_PROF[[0, 7, 5, 9, 2, 4, 11]] = [.30, .15, .10, .08, .06, .08, .03]
_MIN_PROF = np.full(12, .01)
_MIN_PROF[[0, 7, 5, 3, 8, 10, 2]] = [.30, .13, .10, .10, .07, .07, .04]
_MAJ_PROF, _MIN_PROF = np.log(_MAJ_PROF / _MAJ_PROF.sum()), np.log(_MIN_PROF / _MIN_PROF.sum())


def _hist_vote(hist, n_eff):
    h = np.asarray(hist, float)
    if h.sum() <= 0:
        return None
    h = h / h.sum()
    L = np.array([sum(h[(t + d) % 12] * (_MAJ_PROF if k < 12 else _MIN_PROF)[d] for d in range(12))
                  for k in range(24) for t in [k % 12]])
    return _norm(L * n_eff)


def bass_histogram(pc, rel, fps, downbeats=None, ends=(), dur=None, min_rel=0.1):
    """Duration-weighted bass pitch-class histogram from per-frame bass pcs (-1 = none) and a relative level.
    Frames in the first 0.4 s of a bar count x1.5; the bass at each phrase end and at the song's end is added again.
    Returns (hist12, cover) where cover is the share of the song with a usable bass."""
    pc = np.asarray(pc)
    ok = (pc >= 0) & (np.asarray(rel) >= min_rel)
    if not ok.any():
        return None, 0.0
    w = ok.astype(float)
    if downbeats:
        for d in downbeats:
            a = int(d * fps)
            w[a:a + int(0.4 * fps)] *= 1.5
    hist = np.bincount(pc[ok], weights=w[ok], minlength=12)[:12]
    base = hist.sum()
    extra = np.zeros(12)
    last = None
    for t in list(ends) + ([dur] if dur else []):
        b = int(max(0, t - 1.5) * fps), int(t * fps)
        seg = np.where(ok[b[0]:b[1]], pc[b[0]:b[1]], -1)
        seg = seg[seg >= 0]
        if len(seg) >= 3:
            m = int(np.bincount(seg, minlength=12).argmax())
            extra[m] += 0.04 * base * (2.0 if t == dur else 1.0)
            last = m
    return hist + extra, float(ok.mean())


def v_bass(pc, rel, fps, downbeats=None, ends=(), dur=None):
    hist, cover = bass_histogram(pc, rel, fps, downbeats, ends, dur)
    if hist is None or cover < 0.08:
        return None, {"cover": round(cover, 2)}
    L = _hist_vote(hist, 24.0 * min(1.0, cover / 0.5))
    return L, {"cover": round(cover, 2), "hist": [round(float(x), 3) for x in hist / hist.sum()]}


def resting_notes(f0, hop_s, min_len=0.5, gap=0.3):
    """Pitch classes of long vocal notes that end a phrase: a held run of one semitone >= min_len seconds, followed by
    no voiced frame for `gap` seconds (or the end of the track). Returns a list of (pc, seconds)."""
    f0 = np.asarray(f0, float)
    ok = ~np.isnan(f0)
    midi = np.full(len(f0), -1)
    midi[ok] = np.round(12 * np.log2(f0[ok] / 440.0) + 69).astype(int)
    out, i, n = [], 0, len(f0)
    g = max(1, int(gap / hop_s))
    while i < n:
        if midi[i] < 0:
            i += 1
            continue
        j = i
        while j < n and midi[j] == midi[i]:
            j += 1
        dur = (j - i) * hop_s
        if dur >= min_len and not ok[j:j + g].any():
            out.append((int(midi[i]) % 12, dur))
        i = j
    return out


def v_melody(f0, hop_s):
    notes = resting_notes(f0, hop_s)
    if len(notes) < 4:
        return None, {"n": len(notes)}
    hist = np.zeros(12)
    for pc, d in notes:
        hist[pc] += min(d, 2.0)
    return _hist_vote(hist, min(24.0, float(len(notes)))), {"n": len(notes)}


def v_cadence(events):
    """V->I (dominant to the tonic triad of matching mode) counts 1, IV->I 0.4, per key; flat when none."""
    ch = [e for e in events if e.get("root") is not None]
    ev = np.zeros(24)
    for a, b in zip(ch[:-1], ch[1:]):
        if a["root"] == b["root"]:
            continue
        ca, cb = _cls(a["q"]), _cls(b["q"])
        for k in range(24):
            t = k % 12
            want = "M" if k < 12 else "m"
            if cb != want or (b["root"] - t) % 12 != 0:
                continue
            iv = (a["root"] - t) % 12
            if iv == 7 and ca == "M":
                ev[k] += 1.0
            elif iv == 5 and ca in ("M", "m"):
                ev[k] += 0.4
    if ev.max() < 1.0:
        return None, {"landings": 0}
    return _norm(np.log1p(ev) * 2.0), {"landings": round(float(ev.max()), 1)}


# --------------------------------------------------------------------------- madmom subprocess
def madmom_python():
    """Find madmom via override, beside the code, then beside the running venv."""
    candidates = [os.environ.get("RAVEN_MADMOM_PY"), os.path.join(HERE, "venv-madmom", "bin", "python")]
    if sys.prefix != sys.base_prefix:
        candidates.append(os.path.join(os.path.dirname(sys.prefix), "venv-madmom", "bin", "python"))
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


import stage_status as ST  # HONEST CARD: a missing voter is not a vote for silence.


@ST.stage("CNN key voter")
def cnn_probs(path, cache_base=None, timeout=300):
    """24 madmom key probabilities for the whole mix, or None when madmom is not installed / fails. Cached as JSON."""
    cf = cache_base + ".cnnkey.json" if cache_base else None
    if cf and os.path.exists(cf):
        try:
            return json.load(open(cf))["probs"]
        except Exception:
            pass
    py = madmom_python()
    if not py:
        ST.mark("CNN key voter", "skipped", "venv-madmom not installed")
        return None
    try:
        r = subprocess.run([py, os.path.join(HERE, "madmom_key.py"), path], capture_output=True, text=True, timeout=timeout)
        probs = json.loads(r.stdout.strip().splitlines()[-1])["probs"]
        if len(probs) != 24:
            ST.mark("CNN key voter", "failed", "expected 24 probabilities")
            return None
    except Exception as e:
        ST.mark("CNN key voter", "failed", f"{type(e).__name__}: {str(e)[:60]}")
        print(f"[keyfuse] madmom key unavailable ({type(e).__name__}: {str(e)[:60]})", file=sys.stderr)
        return None
    if cf:
        json.dump({"probs": probs}, open(cf, "w"))
    return probs


# --------------------------------------------------------------------------- fusion
def fuse(voters, weights=None):
    """voters: {name: log-prob vector or None}. Returns the posterior P over the 24 keys and the list of voters used."""
    w = dict(WEIGHTS if weights is None else weights)
    used = [n for n, L in voters.items() if L is not None and w.get(n, 0) > 0]
    S = np.zeros(24)
    for n in used:
        S += w[n] * np.asarray(voters[n])
    S = S / max(1.0, len(used) / 2.0)      # the voters are not independent (they all hear the same song): temper the sum
    P = np.exp(S - S.max())
    return P / P.sum(), used


def mirex_score(est, ref):
    return {"same": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2}.get(relation(est, ref), 0.0)


def verdict(voters, weights=None):
    """The ONE key claim: {state, key, line, votes, ...}. state:
      clear     one key, P >= CLEAR_P and a lead of CLEAR_MARGIN: `KEY: A major (3♯) · tonic fairly clear`
      tonic     the notes are known (the top two keys share a signature) but not which is home
      split     the top two keys use different notes and are close: say both
      lean      none of the above: the top key with its odds, flagged as low confidence"""
    P, used = fuse(voters, weights)
    order = np.argsort(P)[::-1]
    k1, k2 = int(order[0]), int(order[1])
    p1, p2 = float(P[k1]), float(P[k2])
    margin = p1 - p2
    sig_p = {}
    for k in range(24):
        sig_p[area(k)] = sig_p.get(area(k), 0.0) + float(P[k])
    a1 = area(k1)
    tops = {n: int(np.argmax(voters[n])) for n in used}
    dissent = [n for n in used if area(tops[n]) != a1 or tops[n] != k1]
    votes = " · ".join(f"{n} {key_short(tops[n])}" for n in used)
    out = {"working": key_name(k1), "p": [round(p1, 2), round(p2, 2)], "top2": [key_name(k1), key_name(k2)],
           "signature": sig_text(k1), "sig_p": round(sig_p[a1], 2), "voters": {n: key_name(tops[n]) for n in used},
           "dissent": dissent, "votes": votes, "relation": relation(k2, k1)}
    sh = area(k1) == area(k2)
    if p1 >= CLEAR_P and margin >= CLEAR_MARGIN:
        how = "tonic clear" if p1 >= VERY_CLEAR_P else "tonic fairly clear"
        out.update(state="clear", key=key_name(k1), line=f"{key_name(k1)} ({sig_text(k1)}) · {how}")
    elif sh:
        rel_ = " or ".join(key_name(k) for k in sorted((k1, k2), key=lambda k: k >= 12))
        out.update(state="tonic", key=None, line=f"{sig_text(k1)} ({rel_}, same notes; tonic unclear)")
    elif sig_p[a1] >= 0.7 and margin < CLEAR_MARGIN:
        # the notes are known (the rest of the posterior is the relative key / near neighbours), the home is not
        rel_k = next((int(k) for k in order[:4] if area(int(k)) == a1 and int(k) != k1), None)
        if rel_k is not None:
            rel_ = " or ".join(key_name(k) for k in sorted((k1, rel_k), key=lambda k: k >= 12))
            out.update(state="tonic", key=None, line=f"{sig_text(k1)} ({rel_}, same notes; tonic unclear)")
        else:
            out.update(state="lean", key=key_name(k1), line=f"{key_name(k1)} ({sig_text(k1)}) · tonic unclear")
    elif margin < CLEAR_MARGIN and p2 >= 0.15:
        out.update(state="split", key=None,
                   line=f"unclear between {key_name(k1)} ({sig_text(k1)}) {p1:.2f} and {key_name(k2)} ({sig_text(k2)}) {p2:.2f}")
    else:
        out.update(state="lean", key=key_name(k1), line=f"{key_name(k1)} ({sig_text(k1)}) · low confidence, P {p1:.2f}")
    # CNN leadership is not independent agreement: name both heavyweight dissenters.
    if tops.get("cnn") == k1 and all(n in tops and tops[n] != k1 for n in ("chords", "spectral")):
        if tops["chords"] == tops["spectral"]:
            note = f"chords and spectral favor {key_name(tops['chords'])}"
        else:
            note = f"chords favor {key_name(tops['chords'])}; spectral favors {key_name(tops['spectral'])}"
        if out["state"] == "clear":
            out.update(state="lean", line=f"{key_name(k1)} ({sig_text(k1)})")
        out["line"] += f" · CNN-led; {note}"
    # one short evidence line, only when it earns its place: the call is not clear, or a heavyweight voter dissents
    heavy = [n for n in dissent if WEIGHTS.get(n, 0) >= 0.5]
    out["votes_line"] = f"{votes}" if (out["state"] != "clear" or heavy) and len(used) > 1 else None
    return out
