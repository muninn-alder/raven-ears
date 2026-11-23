#!/usr/bin/env python
"""credits.py: who played what (1 Nov 2026), a helping hand for instruments.py. Stays ON by default; `--no-credits` turns it off.

Sources, one shape. `--credits FILE` (plain text, see credits.example.txt), else an automatic lookup: MusicBrainz first (free JSON web
service, no key, one request a second), then Discogs (built from liner notes; needs a free personal token in DISCOGS_TOKEN) when
MusicBrainz times out, errors, or has no performer credits. Answers are cached on disk in the ears home so a song is asked about once.
A failed lookup is never allowed to read as "no credits": the card says which source answered and which failed and why.

The hard rule: credits can NAME what the ears hear, never INVENT what they don't. They only break a tie where the tagger was
unsure ("keys (organ or piano?)", "synth/pad-like layer (unconfirmed)", "guitar (type unclear)") on a stem that has real level,
and the new name is always printed `(credited)`. A credited instrument the ears never found stays "credited, not clearly heard".
"""
from __future__ import annotations

from network_policy import offline, require_online

import argparse
import copy
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import socket
import time
import unicodedata
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MB = "https://musicbrainz.org/ws/2/"
UA = "raven-ears-plus/1.0 (song performer credits lookup)"
TIMEOUT_S = 6.0                 # per MusicBrainz request: fail fast and fall through to Discogs
MB_BUDGET_S = 15.0              # all MusicBrainz requests for one song together
DISCOGS = "https://api.discogs.com/"
DISCOGS_TIMEOUT_S = 6.0
DISCOGS_MIN_GAP_S = 1.05        # 60 a minute authenticated
DISCOGS_MAX_REQS = 8            # per song
MIN_GAP_S = 1.1                 # MusicBrainz asks for at most one request a second
MIN_TITLE_SIM, MIN_ARTIST_SIM = 0.80, 0.60
NEG_TTL_S = 7 * 86400           # "nothing found" is asked again after a week; a found song is kept
MAX_TRIES = 3                   # candidate recordings fetched when the best one carries no performer credits
CONFIG = {"file": None, "off": False, "artist": None, "title": None}
_MEMO = {}
_JEV_NAMES = {}
_last_req = [0.0]
_mb_deadline = [None]
_dc_last = [0.0]

# credit word -> the tagger's vocabulary; first match wins, so the specific rows sit above the general ones
CREDIT_WORDS = [
    (r"\b(?:drums?\W*(?:program\w*|machines?)|program\w*\W+drums?|drum\W+loops?|beats?|808)\b", "drum machine"),
    (r"\b(?:drums?|drumming|percussion|kit)\b", "drums"),
    (r"\b(?:english\W+horn|flutes?|clarinets?|oboes?|(?:contra)?bassoons?|sax(?:ophones?)?|woodwinds?|harmonicas?|accordions?)\b", "woodwinds"),
    (r"\b(?:horns?|trumpets?|trombones?|brass|tubas?)\b", "brass"),
    (r"\b(?:vocals?|voices?|sing|singing|choirs?|chorus)\b", "vocals"),
    (r"\b(?:synth\w*\W+bass|bass\W+synth(?:esizers?)?)\b", "synth bass"),
    (r"\b(?:double\W+bass|upright\W+bass|contrabass)\b", "double bass"),
    (r"\bbass\b", "bass guitar"),
    (r"\b(?:banjos?|mandolins?|ukuleles?|sitars?)\b", "banjo/mandolin/ukulele"),
    (r"\belectric\W+guitars?\b", "electric guitar"),
    (r"\b(?:acoustic|classical|nylon(?:[\s-]+strings?)?|(?:12|twelve)[\s-]+strings?)[\s-]+guitars?\b", "acoustic guitar"),
    (r"\bguitars?\b", "guitar"),
    (r"\b(?:pianos?|rhodes|wurlitzer)\b", "piano"),
    (r"\b(?:organs?|hammond)\b", "organ"),
    (r"\bharpsichords?\b", "harpsichord"),
    (r"\b(?:synth(?:s|esi[sz]ers?)?|moog|modular|mellotron|sampl(?:es?|ers?|ing)|electronics?|omnichord)\b", "synth"),
    (r"\b(?:keyboards?|keys)\b", "keyboard"),
    (r"\bharps?\b", "harp"),
    (r"\b(?:violins?|violas?|cellos?|strings?|orchestra)\b", "strings"),
]
# These name work around an instrument, not a performance on it. Programming remains a credit.
SUPPORT_ROLES = r"\b(?:technicians?|techs?|tuning|tuners?|production|processing|licensing)\b"
# a credited class also stands behind these (the tagger's names are coarser or finer than the credit's)
COVERS = {"keyboard": {"piano", "organ", "harpsichord", "synth", "keyboard"},
          "guitar": {"guitar", "electric guitar", "acoustic guitar"},
          "electric guitar": {"guitar"}, "acoustic guitar": {"guitar"},
          "drums": {"drum machine", "percussion"}, "drum machine": {"drums"}, "percussion": {"drums"},
          "bass guitar": {"bass"}, "double bass": {"bass"}, "synth bass": {"bass"}}
MELODIC_CLASSES = ("synth", "piano", "organ", "harpsichord", "keyboard", "electric guitar", "acoustic guitar", "guitar",
                   "strings", "brass", "woodwinds", "harp", "banjo/mandolin/ukulele")
BASE_CLASSES = {"piano": "piano", "organ": "organ", "harpsichord": "harpsichord", "synth": "synth", "synthesizer": "synth",
                "synth/pad-like layer": "synth", "keyboard": "keyboard", "electric guitar": "electric guitar",
                "acoustic guitar": "acoustic guitar", "slide/steel guitar": "guitar", "guitar": "guitar", "plucked strings": "guitar",
                "bass guitar": "bass guitar", "double bass": "double bass", "synth bass": "synth bass", "bass": "bass",
                "drum kit": "drums", "drums": "drums", "drum machine": "drum machine", "percussion": "percussion",
                "strings": "strings", "brass": "brass", "woodwinds": "woodwinds", "harp": "harp", "choir": "choir",
                "banjo/mandolin/ukulele": "banjo/mandolin/ukulele", "bells/mallets": "bells/mallets"}


import stage_status as ST  # HONEST CARD: run-local outcomes.

def _log(m):
    print(m, file=sys.stderr)


def tm(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def ears_home():
    return os.environ.get("RAVEN_EARS_HOME", os.path.expanduser("~/.local/share/raven-ears"))


# ---------------------------------------------------------------------------- words -> classes
def map_word(word):
    """one credit word ('drum programming', 'Synthesizer') -> a class in the tagger's vocabulary, or None."""
    w = word.lower().strip()
    if re.search(SUPPORT_ROLES, w):
        return None
    for rx, cls in CREDIT_WORDS:
        if re.search(rx, w):
            return cls
    return None


def split_words(s):
    return [x.strip() for x in re.split(r"[,;/&]|\band\b", s) if x.strip()]


def parse_credit_text(text):
    """`Name: a, b, c`, `Name - a, b` or just `a, b` per line; # comments and blanks ignored.
    Returns entries [{"who": name|None, "words": [...], "classes": [...]}]."""
    out = []
    for ln in text.splitlines():
        ln = ln.split(" #")[0].strip()
        if not ln or ln.startswith("#"):
            continue
        who, rest = None, ln
        if ":" in ln:
            who, rest = [x.strip() for x in ln.split(":", 1)]
        else:
            m = re.match(r"^(.+?)\s+[-–—]\s+(.+)$", ln)
            if m and map_word(m.group(1)) is None:
                who, rest = m.group(1).strip(), m.group(2).strip()
        words = split_words(rest)
        classes = []
        for w in words:
            c = map_word(w)
            if c and c not in classes:
                classes.append(c)
        if words:
            out.append({"who": who or None, "words": words, "classes": classes})
    return out


def read_credits_file(path):
    with open(path, encoding="utf-8") as f:
        return parse_credit_text(f.read())


def entries_from_relations(rels):
    """MusicBrainz relations -> entries; a bare 'performer' names no role."""
    by = {}
    for r in rels or []:
        typ = (r.get("type") or "").lower()
        if typ not in ("instrument", "vocal", "performer", "programming"):
            continue
        nm = ((r.get("artist") or {}).get("name")) or None
        if not nm:
            continue
        attrs = [a for a in (r.get("attributes") or []) if a and a.lower() not in MODIFIERS]
        if typ == "performer" and not attrs:
            continue
        if typ == "vocal":
            words = attrs or ["vocals"]
            words = [w if "vocal" in w.lower() else w + " vocals" for w in words]
        elif typ == "programming":
            words = [a + " programming" for a in attrs] or ["programming"]
        else:
            words = attrs or [typ]
        e = by.setdefault(nm, {"who": nm, "words": [], "classes": []})
        for w in words:
            if w not in e["words"]:
                e["words"].append(w)
            c = map_word(w)
            if c and c not in e["classes"]:
                e["classes"].append(c)
    return list(by.values())


# ---------------------------------------------------------------------------- identify the recording
def _tags_ffprobe(path):
    from network_policy import local_audio_path
    path = local_audio_path(path)
    try:
        o = subprocess.run(["ffprobe", "-v", "quiet", "-protocol_whitelist", "file,pipe", "-print_format", "json", "-show_format", path],
                           capture_output=True, text=True, timeout=20).stdout
        tags = json.loads(o).get("format", {}).get("tags", {})
        t = {k.lower(): v for k, v in tags.items()}
        return t.get("artist") or t.get("album_artist"), t.get("title")
    except Exception:
        return None, None


def file_tags(path):
    try:
        import mutagen
        f = mutagen.File(path, easy=True)
        if f is not None and f.tags:
            g = lambda k: (f.tags.get(k) or [None])[0]
            return g("artist") or g("albumartist"), g("title")
    except Exception:
        pass
    return _tags_ffprobe(path)


def name_from_file(path):
    """'Artist - Title.mp3' -> (artist, title); no ' - ' -> (None, None)."""
    base = os.path.splitext(os.path.basename(path))[0]
    if " - " in base:
        a, t = base.split(" - ", 1)
        return a.strip(), t.strip()
    return None, None


def identify(path, artist=None, title=None, *, for_lookup=False):
    """(artist, title, where) from --artist/--title, then the file's tags, then 'Artist - Title' in the name."""
    ta, tt = (None, None)
    if not (artist and title):
        ta, tt = file_tags(path)
    fa, ft = (None, None)
    if not ((artist or ta) and (title or tt)):
        fa, ft = name_from_file(path)
    a, t = artist or ta or fa, title or tt or ft
    where = "given" if (artist and title) else ("tags" if (ta and tt) else ("file name" if fa else "partial"))
    if for_lookup and not ((artist or ta) and (title or tt)):
        import jev_more
        if jev_more.J.active():
            key = os.path.basename(path)
            if key not in _JEV_NAMES:
                notes = []
                reading = jev_more.read_filename(key, notes)
                _JEV_NAMES[key] = (reading, notes)
            reading, _ = _JEV_NAMES[key]
            if reading and reading.get("artist") and reading.get("title"):
                a, t = artist or ta or reading["artist"], title or tt or reading["title"]
                reading["filled"] = {k: v for k, v, given in (("artist", a, artist or ta), ("title", t, title or tt)) if not given}
                where = "file name by Jev"
    return a, t, where


def save_filename_reading(path, data, notes):
    """Copy analysis evidence/notes, never make a request while printing."""
    import jev_more
    if not jev_more.J.active():
        return
    reading, diagnostics = _JEV_NAMES.get(os.path.basename(path), (None, []))
    if reading is not None:
        data["filename_reading"] = reading
    for note in diagnostics:
        if note not in notes:
            notes.append(note)
    if reading and reading.get("artist"):
        fields = " / ".join(f"{key}: {value}" for key, value in reading.get("filled", {}).items())
        note = jev_more.short(fields, 55) + " " + jev_more.FILENAME_NOTE
        if note not in notes:
            notes.append(note)


NOISE_WORDS = (r"official\s+(?:lyrics?\s+video|lyrics?|music\s+video|video|audio|visuali[sz]er|hd\s+video)", r"lyrics?\s+video", r"lyrics?",
               r"music\s+video", r"video", r"audio", r"visuali[sz]er", r"hd", r"4k", r"hq", r"full\s+song",
               r"(?:\d{4}\s+)?(?:digital\s+)?re-?master(?:ed)?(?:\s+\d{4})?")


def clean_title(title):
    """Video/upload noise off a title before searching: '(Lyric Video)', '[Official Audio]', 'HD', '4K', '(2011 Remastered)' ...
    (case-insensitive, round or square brackets, or a bare 'HD'/'4K' at the end). Version markers that name a different
    recording -- (Example's Version), (Live), (Acoustic), (Remix) -- are kept."""
    t = title or ""
    noise = "|".join(NOISE_WORDS)
    t = re.sub(r"\s*[\(\[]\s*(?:" + noise + r")\s*[\)\]]", " ", t, flags=re.I)
    t = re.sub(r"(?:\s+|\s*[-|]\s*)(?:official\s+)?(?:hd|4k|hq)\s*$", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -|")
    return t or (title or "").strip()


def drop_version(title):
    """'Example Track (Example's Version)' -> 'Example Track'; None when there is no '(... Version)' marker to drop."""
    d = re.sub(r"\s*[\(\[][^\)\]]*\bversion\b[^\)\]]*[\)\]]", " ", title or "", flags=re.I)
    d = re.sub(r"\s+", " ", d).strip()
    return d if d and d != (title or "").strip() else None


def _norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[\(\[].*?[\)\]]", " ", s)
    s = re.sub(r"\b(feat|ft|featuring)\b.*$", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _sim(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        r = max(r, 0.9)
    return r


MODIFIERS = ("additional", "guest", "solo", "task", "minor", "uncredited")      # relationship attributes that say how, not what
VERSION_WORDS = ("live", "remix", "acoustic", "demo", "instrumental", "karaoke", "cover", "radio edit", "version", "session", "mix")


def score_candidate(rec, artist, title):
    """0..1 or None. Artist + title similarity; a live/remix/demo/etc. version is marked down unless the wanted name says so."""
    names = [c.get("name") or (c.get("artist") or {}).get("name") or "" for c in rec.get("artist-credit", []) if isinstance(c, dict)]
    joined = " ".join(names)
    a_sim = max([_sim(artist, n) for n in names] + [_sim(artist, joined)])
    t_sim = _sim(title, rec.get("title", ""))
    if t_sim < MIN_TITLE_SIM or a_sim < MIN_ARTIST_SIM:
        return None
    s = 0.5 * t_sim + 0.35 * a_sim + 0.15 * min(1.0, float(rec.get("score", 0)) / 100.0)
    blob = (rec.get("title", "") + " " + (rec.get("disambiguation") or "")).lower()
    wanted = title.lower()
    for w in VERSION_WORDS:
        if re.search(r"\b" + w + r"\b", blob) and not re.search(r"\b" + w + r"\b", wanted):
            s -= 0.25
    rels = rec.get("releases") or []
    if rels:
        if all(any(x.lower() in ("live", "compilation", "remix", "dj-mix", "demo") for x in
                   (r.get("release-group", {}).get("secondary-types") or [])) for r in rels):
            s -= 0.10
        if any(r.get("status") == "Official" and r.get("release-group", {}).get("primary-type") == "Album" for r in rels):
            s += 0.05
    return s


def pick_candidates(recs, artist, title, n=MAX_TRIES):
    """best matches first, each as (score, rec); same-song duplicates close to the top are kept (one may carry the credits)."""
    sc = [(score_candidate(r, artist, title), r) for r in recs or []]
    sc = sorted([x for x in sc if x[0] is not None], key=lambda x: -x[0])
    if not sc:
        return []
    return [x for x in sc[:n] if x[0] >= sc[0][0] - 0.12]


# ---------------------------------------------------------------------------- the web (the one place that touches the network)
class LookupFailed(Exception):
    pass


def _why(e):
    """a short honest reason for a failed request (never contains a URL, so never a token); None means a timeout."""
    if isinstance(e, (socket.timeout, TimeoutError)) or "timed out" in str(e).lower():
        return None
    return f"{type(e).__name__}: {str(e)[:60]}"


def _get_json(url):
    require_online("MusicBrainz lookup")
    """GET url as JSON, politely: descriptive User-Agent, under one request a second, short timeout. Raises LookupFailed."""
    if _mb_deadline[0] is not None and time.time() > _mb_deadline[0]:
        raise LookupFailed(f"timed out (over {MB_BUDGET_S:g} s in all)")
    wait = MIN_GAP_S - (time.time() - _last_req[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in (0, 1):
        try:
            _last_req[0] = time.time()
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt == 0 and getattr(e, "code", None) in (429, 503):      # MusicBrainz's "slow down": wait once, ask again
                time.sleep(4.0)
                continue
            raise LookupFailed(_why(e) or f"timed out after {TIMEOUT_S:g} s")


def _q(**kw):
    return urllib.parse.urlencode(kw)


def search_recordings(artist, title):
    lq = lambda s: s.replace('"', " ")
    q = f'recording:"{lq(title)}" AND artist:"{lq(artist)}"'
    return _get_json(MB + "recording?" + _q(query=q, fmt="json", limit=10)).get("recordings", [])


def recording_relations(mbid):
    return _get_json(MB + f"recording/{mbid}?" + _q(inc="artist-rels+work-rels", fmt="json")).get("relations", [])


def release_track_relations(release_id, mbid):
    """the same recording's relations as listed on one release (some credits only live there)."""
    d = _get_json(MB + f"release/{release_id}?" + _q(inc="recordings+artist-rels+recording-level-rels", fmt="json"))
    for m in d.get("media", []):
        for t in m.get("tracks", []):
            if (t.get("recording") or {}).get("id") == mbid:
                return t["recording"].get("relations", [])
    return []


def _release_order(rec):
    rs = rec.get("releases") or []
    good = lambda r: (r.get("status") == "Official", r.get("release-group", {}).get("primary-type") == "Album",
                      not r.get("release-group", {}).get("secondary-types"))
    return sorted(rs, key=good, reverse=True)[:2]


def lookup_musicbrainz(artist, title):
    """-> {"found": bool, "matched": 'Artist - Title', "mbid": id, "entries": [...]}. Raises LookupFailed on network trouble."""
    _mb_deadline[0] = time.time() + MB_BUDGET_S
    cands = pick_candidates(search_recordings(artist, title), artist, title)
    if not cands:
        return {"found": False, "entries": []}
    first = None
    for sc, rec in cands:
        names = ", ".join(c.get("name", "") for c in rec.get("artist-credit", []) if isinstance(c, dict))
        res = {"found": True, "matched": f"{names} - {rec.get('title', '')}", "mbid": rec["id"], "entries": []}
        import song_length
        length = _duration(rec.get("length"))
        if (length and song_length.norm(title) == song_length.norm(rec.get("title"))
                and song_length.norm(artist) == song_length.norm(names)
                and not song_length.alternate(rec.get("title"))
                and not song_length.alternate(rec.get("disambiguation"))
                and not all("Live" in (r.get("release-group", {}).get("secondary-types") or [])
                            for r in rec.get("releases") or [{}])):
            res["studio_duration_s"] = length / 1000.0
        first = first or res
        ent = entries_from_relations(recording_relations(rec["id"]))
        if not ent:
            for rel in _release_order(rec):
                ent = entries_from_relations(release_track_relations(rel["id"], rec["id"]))
                if ent:
                    break
        if ent:
            res["entries"] = ent
            return res
    return first


# ---------------------------------------------------------------------------- Discogs (second source)
# Roles that name work around the music, not a performance on it (a "Vocal Engineer" is not a singer).
DISCOGS_NONPERF = (r"\b(?:producer|produced|engineer\w*|mix(?:ed|ing)?|master(?:ed|ing)?|record(?:ed|ing)|written|writer|lyrics|composed|"
                   r"composer|arrang\w*|conduct\w*|design\w*|photograph\w*|art\s+direction|edit(?:ed|ing)?|a&r|management|copyright|"
                   r"phonographic|published|liner|coordinat\w*|executive|direction|translated|remix\w*|technician|featuring|credits?)\b")
DISCOGS_INSTRUMENTS = (r"\b(?:bagpipes?|glockenspiel|vibraphone|marimba|xylophone|timpani|tambourine|bongos?|congas?|cowbell|celesta|"
                       r"melodica|recorder|theremin|dulcimer|lute|lyre|kalimba|didgeridoo|steel\s+drums?|pedal\s+steel|"
                       r"lap\s+steel|dobro|piccolo|ocarina|whistle|programm\w*|vocal\w*|rapping|bells?|chimes|handclaps?|"
                       r"snaps?|stomps?|tapes?|loops?|turntables?|scratch\w*)\b")
DISCOGS_BRACKET_MODS = ("backing", "lead", "acoustic", "electric", "additional", "background", "classical", "nylon", "12-string", "bass")


def discogs_token():
    """The Discogs token or None: env DISCOGS_TOKEN, else the file named by env RAVEN_DISCOGS_ENV (a line `DISCOGS_TOKEN=...`).
    Never printed, logged, cached or put in a URL (it travels in a header)."""
    t = (os.environ.get("DISCOGS_TOKEN") or "").strip()
    if t:
        return t
    path = os.environ.get("RAVEN_DISCOGS_ENV")
    if path:
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln.startswith("export "):
                        ln = ln[7:].strip()
                    if ln.startswith("DISCOGS_TOKEN="):
                        v = ln.split("=", 1)[1].strip().strip("'\"")
                        if v:
                            return v
        except OSError:
            pass
    return None


def _discogs_json(url, token):
    require_online("Discogs lookup")
    """GET one Discogs URL as JSON. Descriptive User-Agent, token in the Authorization header, ~1 request a second, 6 s timeout,
    one retry on 429 after a backoff, and a pause when X-Discogs-Ratelimit-Remaining runs low. Raises LookupFailed (no token in it)."""
    wait = DISCOGS_MIN_GAP_S - (time.time() - _dc_last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json",
                                               "Authorization": f"Discogs token={token}"})
    for attempt in (0, 1):
        try:
            _dc_last[0] = time.time()
            with urllib.request.urlopen(req, timeout=DISCOGS_TIMEOUT_S) as r:
                rem = r.headers.get("X-Discogs-Ratelimit-Remaining")
                body = json.loads(r.read().decode("utf-8"))
            try:
                if rem is not None and int(rem) <= 1:
                    time.sleep(5.0)
            except ValueError:
                pass
            return body
        except Exception as e:
            code = getattr(e, "code", None)
            if attempt == 0 and code == 429:
                time.sleep(6.0)
                continue
            if code in (401, 403):
                raise LookupFailed(f"token rejected (HTTP {code})")
            if code == 429:
                raise LookupFailed("rate limit hit twice")
            msg = _why(e) or f"timed out after {DISCOGS_TIMEOUT_S:g} s"
            raise LookupFailed(msg.replace(token, "***") if token else msg)


def _strip_n(name):
    """Discogs disambiguates people with '(2)' after the name."""
    return re.sub(r"\s*\(\d+\)\s*$", "", name or "").strip()


def discogs_words(role):
    """A Discogs role string ('Electric Guitar, Bass [Fretless]', 'Vocals [Backing]') -> credit words; production/mix/arranging roles -> []."""
    out = []
    depth, cur, parts = 0, "", []
    for ch in role or "":
        depth += (ch == "[") - (ch == "]")
        if ch == "," and depth <= 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    for part in parts:
        part = part.strip()
        m = re.match(r"^(.*?)\s*\[(.*)\]\s*$", part)
        base, br = (m.group(1).strip(), m.group(2).strip()) if m else (part, "")
        if not base or re.search(DISCOGS_NONPERF, base.lower()):
            continue
        word = f"{br} {base}" if br and br.lower() in DISCOGS_BRACKET_MODS else base
        if re.search(r"\bprogrammed\b", word.lower()):
            word = "programming"
        if map_word(word) is not None or re.search(DISCOGS_INSTRUMENTS, word.lower()):
            if word not in out:
                out.append(word)
    return out


def _track_set(spec, positions):
    """Discogs album-level `tracks` ('1, 3-5', 'A1 to A3') -> the set of tracklist positions it names."""
    hit = set()
    for part in re.split(r"\s*,\s*", spec or ""):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\S+?)\s*(?:-|to|–)\s*(\S+)$", part, flags=re.I)
        if m and m.group(1) in positions and m.group(2) in positions:
            a, b = positions.index(m.group(1)), positions.index(m.group(2))
            hit.update(positions[min(a, b):max(a, b) + 1])
        else:
            hit.add(part)
    return hit


def _duration(value):
    try:
        value = float(value)
        return round(value, 1) if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        return None


def _track_duration(track):
    parts = (track.get("duration") or "").split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        return None
    return _duration(sum(int(p) * 60 ** i for i, p in enumerate(reversed(parts))))


def _discogs_track(release, title, duration=None):
    tracks = [t for t in release.get("tracklist") or [] if t.get("type_", "track") == "track"]
    def rank(t):
        seconds = _track_duration(t)
        return (_sim(title, t.get("title", "")),
                -abs(seconds - duration) if seconds and duration else -float("inf"))
    best = max(tracks, key=rank, default=None)
    return best if best and _sim(title, best.get("title", "")) >= MIN_TITLE_SIM else None


def entries_from_discogs(release, title, duration=None):
    """A Discogs release dict + the song title -> (entries, position or None). Per-track extraartists beat album-level ones; an
    album-level credit counts when its `tracks` names this position, or (no `tracks` given) for a person with no per-track credit."""
    tl = [t for t in release.get("tracklist") or [] if t.get("type_", "track") == "track"]
    best = _discogs_track(release, title, duration)
    if best is None:
        return [], None
    positions = [t.get("position", "") for t in tl]
    pos = best.get("position", "")
    by = {}

    def add(a, rank):
        nm = _strip_n(a.get("name") or a.get("anv") or "")
        words = discogs_words(a.get("role"))
        if not nm or not words:
            return
        e = by.get(nm)
        if e is None or e["rank"] > rank:
            e = by[nm] = {"who": nm, "words": [], "classes": [], "rank": rank}
        if e["rank"] == rank:
            for w in words:
                if w not in e["words"]:
                    e["words"].append(w)
                c = map_word(w)
                if c and c not in e["classes"]:
                    e["classes"].append(c)

    for a in best.get("extraartists") or []:
        add(a, 0)                                        # per-track: wins
    for a in release.get("extraartists") or []:
        tk = (a.get("tracks") or "").strip()
        if tk:
            if pos in _track_set(tk, positions):
                add(a, 1)
        else:
            add(a, 2)
    return [{k: v for k, v in e.items() if k != "rank"} for e in by.values()], pos


def _release_rank(r, title=""):
    formats = list(r.get("format") or [])
    for f in r.get("formats") or []:
        formats.extend(f.get("descriptions") or [])
    fmt = " ".join(formats).lower()
    # Release notes too: "Recorded at the Concord Pavilion ... Jazz Festival" is a live album whose
    # title and formats never say so (Take Five, 12 Nov 2026).
    blob = (r.get("title", "") + " " + fmt + " " + str(r.get("notes") or "")[:400]).lower()
    risky = bool(re.search(r"\b(?:compilation|greatest hits|best of|essential|collection|concert|festival|unofficial)\b", blob))
    for word in VERSION_WORDS:
        if re.search(r"\b" + word + r"\b", blob) and not re.search(r"\b" + word + r"\b", title.lower()):
            risky = True
    try:
        yr = int(r.get("year") or 9999)
        if yr <= 0:
            yr = 9999
    except (TypeError, ValueError):
        yr = 9999
    return (risky, yr, 0 if "album" in fmt else 1 if "single" in fmt else 2)


def lookup_discogs(artist, title, token, duration=None):
    """Search up to 100 masters, resolve the best three to main_release, then rank
    verified tracks: avoid live/compilation/version releases, avoid far durations
    (> max(15 s, 5%)), earliest master year, album before single/EP on that year,
    then close duration (within max(5 s, 2%)); missing lengths are neutral and
    labelled as not listed. Credits never affect rank.
    A release search is a fallback when masters yield no matching track. No master,
    missing year, tied rank, risky release, incomplete search, or an unconfirmed
    listed duration/version gives 'uncertain' with attribution and no usable credits.
    A main release newer than its master is also uncertain. At most eight requests;
    this is evidence for an original, not proof that a master is the original take.
    """
    duration = _duration(duration)
    used = [0]

    def get(path, **kw):
        if used[0] >= DISCOGS_MAX_REQS:
            raise LookupFailed("Discogs candidate request budget exhausted")
        used[0] += 1
        return _discogs_json(DISCOGS + path + (("?" + _q(**kw)) if kw else ""), token)

    candidates, incomplete = [], False
    for typ in ("master", "release"):
        if used[0] >= DISCOGS_MAX_REQS - 1:
            break
        response = get("database/search", type=typ, artist=artist, track=title, per_page=100)
        incomplete |= (response.get("pagination") or {}).get("pages", 1) > 1
        good = [r for r in response.get("results", [])
                if _sim(artist, (r.get("title") or "").split(" - ", 1)[0]) >= MIN_ARTIST_SIM]
        good.sort(key=lambda r: _release_rank(r, title))
        seen = set()
        checked = []
        for r in good:
            cost = 2 if typ == "master" or r.get("master_id") else 1
            if len(checked) >= 3 or used[0] + cost > DISCOGS_MAX_REQS:
                break
            checked.append(r)
            master = None
            rid = r.get("id")
            mid = rid if typ == "master" else r.get("master_id")
            if mid:
                master = get(f"masters/{mid}")
                rid = master.get("main_release")
                if not rid:
                    continue
            if not rid or rid in seen:
                continue
            seen.add(rid)
            rel = get(f"releases/{rid}")
            track = _discogs_track(rel, title, duration)
            if track is None:
                continue
            names = ", ".join(_strip_n(a.get("name", "")) for a in rel.get("artists") or [])
            matched = f"{names or artist} - {rel.get('title', '')}" + (f" ({rel.get('year')})" if rel.get("year") else "")
            original = dict(r, **(master or rel))
            risk, year, kind = _release_rank(original, title)
            risk |= _release_rank(rel, title)[0] or _release_rank(track, title)[0]
            seconds = _track_duration(track)
            if not seconds and master:
                seconds = _track_duration(_discogs_track(master, title, duration) or {})
            delta = abs(seconds - duration) if seconds and duration else None
            far = delta is not None and delta > max(15, duration * 0.05)
            close = delta is not None and delta <= max(5, duration * 0.02)
            rank = (risk, far, year, kind, 0 if close or delta is None else 1)
            reasons = []
            if not master:
                reasons.append("no master establishes the original release")
            if year == 9999:
                reasons.append("original year unknown")
            if risk:
                reasons.append("live, compilation or alternate-version release")
            if master and _release_rank(rel, title)[1] != year:
                reasons.append("main release year differs from the master year")
            if delta is not None and not close:
                reasons.append(f"track duration differs by {delta:.0f} s")
            blob = (rel.get("title", "") + " " + track.get("title", "")).lower()
            if any(re.search(r"\b" + w + r"\b", title.lower()) and
                   not re.search(r"\b" + w + r"\b", blob) for w in VERSION_WORDS):
                reasons.append("requested recording version not confirmed")
            if names and _sim(artist, names) < MIN_ARTIST_SIM:
                reasons.append("release artist differs from requested artist")
            result = {"matched": matched,
                      "release_id": rid, "master_id": mid, "year": rel.get("year")}
            import song_length
            if (seconds and song_length.norm(track.get("title")) == song_length.norm(title)
                    and not song_length.alternate(track.get("title"))):
                result["studio_duration_s"] = seconds
            if seconds is None:
                result["match_note"] = "master main release; track length not listed" if master else "track length not listed"
            candidates.append((rank, result, reasons, rel))
        if candidates:
            # Search rows beyond the detail budget can still leave the year tied.
            best = min(c[0] for c in candidates)
            if any(_release_rank(r, title) <= (best[0], best[2], best[3]) for r in good[len(checked):]):
                incomplete = True
            break
    if not candidates:
        return {"status": "uncertain" if incomplete or used[0] >= DISCOGS_MAX_REQS - 1 else "nomatch",
                "detail": "no original verified in the candidate search", "entries": []}
    candidates.sort(key=lambda c: c[0])
    rank, result, reasons, release = candidates[0]
    if any(other[0] == rank for other in candidates[1:]):
        reasons.append("multiple releases tie for the original")
    if incomplete:
        reasons.append("candidate search incomplete")
    # A length mismatch with an otherwise verified original is the evidence we
    # need. Other match doubts must not become a claim about the studio track.
    if any(not reason.startswith("track duration differs by ") for reason in reasons):
        result.pop("studio_duration_s", None)
    if reasons:
        result.update(status="uncertain", entries=[], detail="; ".join(reasons))
    else:
        entries, _ = entries_from_discogs(release, title, duration)
        result.update(status="ok" if entries else "noperf", entries=entries)
    return result


# ---------------------------------------------------------------------------- cache + the top-level get()
def cache_path(artist, title, cache_dir=None, duration=None):
    d = cache_dir or os.path.join(ears_home(), "credits-cache")
    k = lambda x: re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKD", x or "").lower()).strip()      # brackets KEPT: "Song (Live)" is not "Song"
    return os.path.join(d, hashlib.sha1(f"{k(artist)}|{k(title)}|v4-original|{_duration(duration)}".encode()).hexdigest()[:16] + ".json")


def chain(artist, title, duration=None):
    """MusicBrainz, then Discogs when MusicBrainz failed or had no performer credits. Never raises.
    -> {"mb": {...}, "dc": {...}, "entries", "source", "matched", "found"}; each status is ok / uncertain / nomatch / noperf / failed / skipped
    (failed and skipped carry a `detail`)."""
    c = {"entries": [], "source": None, "matched": None, "found": False}
    try:
        r = lookup_musicbrainz(artist, title)
        if r.get("entries"):
            mb = {"status": "ok", "matched": r.get("matched")}
            c.update(entries=r["entries"], source="musicbrainz", matched=r.get("matched"), found=True)
        elif r.get("found"):
            mb = {"status": "noperf", "matched": r.get("matched")}
            c.update(matched=r.get("matched"), found=True)
        else:
            mb = {"status": "nomatch"}
        if r.get("studio_duration_s"):
            mb["studio_duration_s"] = r["studio_duration_s"]
    except LookupFailed as e:
        mb = {"status": "failed", "detail": str(e)}
    except Exception as e:
        mb = {"status": "failed", "detail": type(e).__name__}
    c["mb"] = mb
    dc = {"status": "notried"}
    if not c["entries"]:
        tok = discogs_token()
        if not tok:
            dc = {"status": "skipped", "detail": "no DISCOGS_TOKEN set; its search needs one"}
        else:
            try:
                r = lookup_discogs(artist, title, tok, duration=duration)
                dc = {k: v for k, v in r.items() if k != "entries"}
                if r["entries"]:
                    c.update(entries=r["entries"], source="discogs", matched=r.get("matched"), found=True)
            except LookupFailed as e:
                dc = {"status": "failed", "detail": str(e)}
            except Exception as e:
                dc = {"status": "failed", "detail": type(e).__name__}
    c["dc"] = dc
    return c


def _failed(c):
    return "failed" in ((c.get("mb") or {}).get("status"), (c.get("dc") or {}).get("status"))


def _cache_ok(c):
    """A cached answer is kept when it has credits. A 'nothing found' is kept a week, unless Discogs was never really asked
    (no token then) and a token has appeared since."""
    if c.get("entries"):
        return True
    if time.time() - c.get("when", 0) >= NEG_TTL_S:
        return False
    if (c.get("dc") or {}).get("status") in ("nomatch", "noperf", "uncertain"):
        return True
    return not discogs_token()


@ST.stage("credits", kind="input", classify=lambda value: ST.lookup_rows("credits", value))
def get(path, artist=None, title=None, credits_file=None, off=False, cache_dir=None, use_cache=True, duration=None):
    """The credits for one song. Never raises. -> {"source": 'file'|'musicbrainz'|'discogs'|None, "label": str, "entries": [...], "note": str|None}"""
    if off or offline() or os.environ.get("RAVEN_NO_CREDITS"):
        return {"source": None, "entries": [], "note": None, "off": True}
    if credits_file:
        try:
            ent = read_credits_file(credits_file)
        except OSError as e:
            return {"source": None, "entries": [], "note": f"credits file unreadable ({e.strerror or e}); ears only"}
        return {"source": "file", "label": "credits file", "entries": ent,
                "note": None if ent else "credits file has no usable lines; ears only"}
    try:
        a, t, where = identify(path, artist, title, for_lookup=True)
    except Exception as e:
        return {"source": None, "entries": [], "note": f"could not identify the song ({type(e).__name__}); ears only"}
    if not (a and t):
        return {"source": None, "entries": [],
                "note": "no artist/title to look up (no tags, no 'Artist - Title' file name; try --artist/--title); ears only"}
    tq = clean_title(t)                                     # the title as searched; the card still names the song by its file name
    duration = _duration(duration)
    cp = cache_path(a, tq, cache_dir, duration)
    if use_cache and os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as handle:
                c = json.load(handle)
            if _cache_ok(c):
                return _from_lookup(c, a, t, tq)
        except Exception:
            pass
    c = chain(a, tq, duration=duration)
    alt = drop_version(tq)
    if not c["entries"] and alt and not _failed(c):         # "(Example's Version)" found nothing: ask once more without it, and say so
        c2 = chain(a, alt, duration=duration)
        if c2["entries"] or (c2["found"] and not c["found"]):
            c = c2
            c["dropped_version"] = True
    c["when"] = time.time()
    if use_cache and (c["entries"] or not _failed(c)):       # a failed lookup is never cached as "nothing found"
        try:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, "w", encoding="utf-8") as handle:
                json.dump(c, handle)
        except OSError:
            pass
    return _from_lookup(c, a, t, tq)


def _from_lookup(c, a, t, tq=None):
    tq = tq or t
    cleaned = f" (searched as '{tq}', video/upload words removed)" if tq != t else ""
    dv = " [matched without the '(... Version)' marker, so these may be the original recording's credits, not this version's]" if c.get("dropped_version") else ""
    mb = c.get("mb") or {"status": "ok" if c.get("entries") else ("noperf" if c.get("found") else "nomatch"), "matched": c.get("matched")}
    dc = c.get("dc") or {"status": "notried"}

    statuses = ST.credit_sources(dict(c, mb=mb, dc=dc))
    lengths = [{"source": name, "duration_s": row["studio_duration_s"]}
               for name, row in (("MusicBrainz", mb), ("Discogs", dc)) if row.get("studio_duration_s")]
    evidence = {"studio_durations": lengths} if lengths else {}

    def failed_msg(name, detail):
        return f"{name} {detail}" if str(detail).startswith("timed out") else f"{name} unreachable ({detail})"

    def mb_msg():
        s = mb["status"]
        if s == "failed":
            return failed_msg("MusicBrainz", mb.get("detail"))
        if s == "nomatch":
            return f"MusicBrainz has no match for '{a} - {t}'{cleaned}"
        return f"MusicBrainz has '{mb.get('matched')}' but lists no performer credits{dv}"

    def dc_msg():
        s = dc["status"]
        if s == "failed":
            return failed_msg("Discogs", dc.get("detail"))
        if s == "skipped":
            return f"Discogs skipped ({dc.get('detail')})"
        if s == "nomatch":
            return "Discogs has no match"
        if s == "uncertain":
            release = f" for '{dc['matched']}'" if dc.get("matched") else ""
            return f"Discogs match uncertain{release} ({dc.get('detail')})"
        if s == "noperf":
            return f"Discogs has '{dc.get('matched')}' but lists no performer credits{dv}"
        return None

    if not c.get("entries"):
        why = [m for m in (mb_msg(), dc_msg()) if m]
        return {"source": None, "entries": [], "note": "; ".join(why) + "; ears only", "stage_status": statuses, **evidence}
    if c.get("source") == "discogs":
        extra = "" if mb["status"] == "ok" else f"; {mb_msg()}"
        match_note = f" ({dc['match_note']})" if dc.get("match_note") else ""
        return {"source": "discogs", "label": f"Discogs: {c.get('matched')}{match_note}{dv}{extra}", "entries": c["entries"], "note": None, "stage_status": statuses, **evidence}
    return {"source": "musicbrainz", "label": f"MusicBrainz: {c.get('matched')}{dv}", "entries": c["entries"], "note": None, "stage_status": statuses, **evidence}


# ---------------------------------------------------------------------------- the tie-break
def label_classes(label):
    """a card label ('keys (organ or piano?)', 'synth (credited)', 'bass (synth-like)') -> the set of classes it names."""
    l = label.lower().strip()
    m = re.match(r"keys \((\w+) or (\w+)\?\)", l)
    if m:
        return {m.group(1), m.group(2)}
    m = re.match(r"(\w+) or (\w+) \(both credited\)", l)
    if m:
        return {m.group(1), m.group(2)}
    if "synth-like" in l:
        return {"synth bass"}
    base = re.sub(r"\s*\(.*$", "", l)
    c = BASE_CLASSES.get(base)
    return {c} if c else set()


def covers(credited, cls):
    return cls == credited or cls in COVERS.get(credited, ())


def resolve(label, credited):
    """The tie-break for ONE unsure label: the credited class that wins, as '<class> (credited)', or None (leave it alone).
    Only labels the tagger was unsure about are touched; a confident name is never overruled."""
    l = label.lower()
    if "(credited)" in l or "both credited" in l:
        return None
    cr = set(credited)
    m = re.match(r"keys \((\w+) or (\w+)\?\)", l)
    if m:
        a, b = m.group(1), m.group(2)
        hit = [c for c in (a, b) if c in cr]
        if len(hit) == 1:
            return f"{hit[0]} (credited)"
        if len(hit) == 2:
            return f"{a} or {b} (both credited)"
        return "keyboard (credited)" if "keyboard" in cr else None
    if l.startswith("synth/pad-like"):
        return "synth (credited)" if "synth" in cr else ("keyboard (credited)" if "keyboard" in cr else None)
    if l.startswith("piano (weak"):
        return "piano (credited)" if "piano" in cr else None
    if "guitar" in l or l.startswith("plucked strings"):
        if "(probably)" in l:
            cls = next(iter(label_classes(l.replace(" (probably)", ""))), None)
            return f"{cls} (credited)" if cls in cr else None
        if l.startswith(("guitar (type unclear)", "plucked strings")):
            hit = [c for c in ("electric guitar", "acoustic guitar") if c in cr]
            if len(hit) == 1:
                return f"{hit[0]} (credited)"
            return "guitar (credited)" if not hit and "guitar" in cr else None
        return None
    if l.startswith("bass (synth-like)"):
        return "synth bass (credited)" if "synth bass" in cr else None
    if l.startswith(("bass (type unclear)", "bass guitar (plucked)")):
        hit = [c for c in ("bass guitar", "double bass", "synth bass") if c in cr]
        return f"{hit[0]} (credited)" if len(hit) == 1 else None
    if l.startswith("drums (type unclear)"):
        hit = [c for c in ("drum machine", "drums") if c in cr]
        return f"{hit[0]} (credited)" if len(hit) == 1 else None
    return None


def resolve_unnamed(credited):
    """A melodic stem with real level and no name at all: named only when the credits leave exactly one melodic candidate."""
    hit = [c for c in MELODIC_CLASSES if c in set(credited)]
    return f"{hit[0]} (credited)" if len(hit) == 1 else None


_CREDITED_LABEL = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(set(BASE_CLASSES.values()), key=len, reverse=True)) +
    r")(?: or (\w+) \(both credited\)| \(credited\))")


def saved_timeline_text(text, instruments):
    """Older harmony passes could apply a separate lookup with no printed source."""
    if "(credited)" not in text and "(both credited)" not in text:
        return text
    credits = instruments.get("credits") or {}
    entries = credits.get("entries", []) if credits.get("source") else []
    supported = {c for entry in entries for c in entry["classes"]}
    labels = instruments.get("families") or {}

    def replace(match):
        classes = {c for c in match.groups() if c}
        if classes <= supported:
            return match[0]
        label = next((name for name in labels if resolve(name, classes) == match[0]), None)
        return label or match[0].replace("both credited", "possibly").replace("(credited)", "(possibly)")

    return _CREDITED_LABEL.sub(replace, text).replace("(possibly) enters (possibly)", "enters (possibly)").replace(
        "(possibly) leaves (possibly)", "leaves (possibly)")


# ---------------------------------------------------------------------------- apply to an instruments.analyse() result
def _merge(spans, gap=8.0):
    m = []
    # HONEST CARD: JSON cache spans are lists; fresh spans can be tuples.
    for a, b in sorted((a, b) for a, b in spans):
        if m and a - m[-1][1] < gap:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    return m


def _where(spans, dur):
    m = _merge(spans)
    if sum(b - a for a, b in m) >= 0.85 * dur:
        return "throughout"
    s = ", ".join(f"{tm(a)}–{tm(b)}" for a, b in m[:4])
    return s + (f" +{len(m) - 4} more" if len(m) > 4 else "")


def _unnamed_spans(res):
    """spans where the 'other' stem has level but no family claims it (the unnamed melodic layer)."""
    n, hop = res["n"], res["hop"]
    un = [False] * n
    for a, b in res.get("other_on", []):
        for i in range(int(a / hop), min(n, int(b / hop))):
            un[i] = True
    for v in res["families"].values():
        if v["stem"] == "other":
            for a, b in v["spans"]:
                for i in range(int(a / hop), min(n, int(b / hop))):
                    un[i] = False
    out, a = [], None
    for i in range(n + 1):
        if i < n and un[i] and a is None:
            a = i
        elif (i == n or not un[i]) and a is not None:
            out.append((a * hop, i * hop))
            a = None
    return [(a, b) for a, b in out if b - a >= 4.0]


def apply(res, cr):
    """A copy of an instruments result with credited names where the tagger was unsure, plus res['credit_lines'] (the CREDITS block).
    Nothing is added that the ears did not already hear on a stem with level."""
    if not res or "families" not in res or cr.get("off"):
        return res
    res = copy.deepcopy(res)
    ents = cr.get("entries") or []
    if not ents:
        if cr.get("note"):
            res["credit_lines"] = ["CREDITS: " + cr["note"]]
        return res
    people = {}
    for e in ents:
        for c in e["classes"]:
            people.setdefault(c, [])
            if e["who"] and e["who"] not in people[c]:
                people[c].append(e["who"])
    credited = list(people)
    # 1. the tie-break, only on families that exist (so they already have level)
    fams = {}
    for lab, v in res["families"].items():
        new = resolve(lab, credited) if v.get("spans") else None
        if new and new not in fams and new not in res["families"]:
            v = dict(v, credited_from=lab)
            v.pop("energy_only", None)
            v.pop("timbre_only", None)
            lab = new
        fams[lab] = v
    res["families"] = fams
    oc = res.get("other_character")
    un_spans = _unnamed_spans(res) if oc else []
    if oc and un_spans:
        new = resolve(oc["label"], credited) if oc.get("label") else resolve_unnamed(credited)
        if new:
            oc["credited_from"] = oc.get("label") or "unnamed melodic layer"
            oc["label"] = new
    # 2. the block: credited and heard / credited, not clearly heard / heard, not credited
    dur = res["duration"]
    items = [(lab, v["spans"]) for lab, v in fams.items()]
    if oc and oc.get("label") and un_spans:
        items.append((oc["label"], un_spans))
    vocal = res.get("vocal_on", [])
    heard, missing, used = [], [], set()
    for cls in credited:
        who = f" ({', '.join(people[cls])})" if people[cls] else ""
        if cls == "vocals":
            if sum(b - a for a, b in vocal) >= 8:
                heard.append(f"vocals{who}: {_where(vocal, dur)}")
            else:
                missing.append(f"vocals{who}: credited, not clearly heard")
            continue
        hit = [(i, lab, sp) for i, (lab, sp) in enumerate(items) if any(covers(cls, x) for x in label_classes(lab))]
        if hit:
            used.update(i for i, _, _ in hit)
            heard.append(f"{cls}{who}: {_where([s for _, _, sp in hit for s in sp], dur)}")
        else:
            extra = ""
            if cls in MELODIC_CLASSES and res.get("unclear", {}).get("other", 0) >= 0.3:
                extra = " (some melodic sound stayed unnamed; it could be this)"
            missing.append(f"{cls}{who}: credited, not clearly heard{extra}")
    extra_h = []
    if any(c != "vocals" for c in credited):
        for i, (lab, sp) in enumerate(items):
            if i in used or not label_classes(lab):
                continue
            note = ""
            if "synth bass" in label_classes(lab) and any(c in credited for c in ("synth", "keyboard")):
                note = "; possibly synth bass"
            extra_h.append(f"{lab}: not credited{note}")
    L = [f"CREDITS ({cr.get('label', 'credits')}; names what the ears hear, never adds what they don't):"]
    for title_, rows in (("credited and heard:", heard), ("credited, not clearly heard:", missing), ("heard, not credited:", extra_h)):
        if rows:
            L.append("  " + title_)
            L += [f"    {x}" for x in rows]
    if len(L) == 1:
        L[0] = L[0][:-1] + " nothing in the list matches anything the ears picked out."
    res["credit_lines"] = L
    res["credits"] = {"source": cr.get("source"), "label": cr.get("label"), "entries": ents}
    return res


# ---------------------------------------------------------------------------- CLI plumbing shared by singer / harmony / instruments
def add_args(ap):
    ap.add_argument("--credits", metavar="FILE", help="who played what, plain text (see credits.example.txt); replaces the MusicBrainz/Discogs lookup")
    ap.add_argument("--no-credits", action="store_true", help="no credits at all (when a database match is wrong for this recording)")
    ap.add_argument("--artist", help="artist for the MusicBrainz lookup (default: file tags, then 'Artist - Title' in the file name)")
    ap.add_argument("--title", help="title for the MusicBrainz lookup")


def configure_from_args(a):
    CONFIG.update(file=getattr(a, "credits", None), off=bool(getattr(a, "no_credits", False)),
                  artist=getattr(a, "artist", None), title=getattr(a, "title", None))
    _MEMO.clear()
    _JEV_NAMES.clear()


def for_song(path, duration=None):
    """credits for this process's configured run, looked up once per song. RAVEN_NO_CREDITS=1 switches it off."""
    if offline():
        return get(path, off=True)
    duration = _duration(duration)
    key = (path, duration) if duration is not None else path
    if key not in _MEMO:
        off = CONFIG["off"] or bool(os.environ.get("RAVEN_NO_CREDITS"))
        _MEMO[key] = get(path, CONFIG["artist"], CONFIG["title"], CONFIG["file"], off=off, duration=duration)
    ST.collect(_MEMO[key].get("stage_status", []))
    return _MEMO[key]


def duration_evidence(path):
    """Read only this run's memo; do not look up credits just to check length."""
    rows = []
    for key, value in _MEMO.items():
        cached_path = key[0] if isinstance(key, tuple) else key
        if os.path.abspath(cached_path) == os.path.abspath(path):
            rows.extend(value.get("studio_durations") or [])
    return rows


def apply_for(path, res):
    try:
        cr = for_song(path, duration=(res or {}).get("duration"))
        result = apply(res, cr)
        if result is not None:
            result["stage_status"] = ST.merge(result.get("stage_status", []), cr.get("stage_status", []))
        return result
    except Exception as e:
        ST.mark("credits", "failed", f"{type(e).__name__}: {str(e)[:60]}", kind="input")
        _log(f"credits: failed ({type(e).__name__}: {str(e)[:60]})")
        if res is not None:
            res = dict(res, stage_status=ST.merge(res.get("stage_status", []),
                [ST.outcome("credits", "failed", f"{type(e).__name__}: {str(e)[:60]}", kind="input")]))
        return res


@ST.engine
@ST.cli
def main(argv=None):
    ap = argparse.ArgumentParser(description="credits.py: show the credits that would be used for a song")
    ap.add_argument("audio", nargs="?", default="")
    ap.add_argument("--duration-seconds", type=float, help="known audio duration for the Discogs recording match")
    add_args(ap)
    a = ap.parse_args(argv)
    from network_policy import local_audio_path
    local_audio_path(a.audio, ap)
    c = get(a.audio, a.artist, a.title, a.credits, off=a.no_credits, duration=a.duration_seconds)
    from ears_handoff import finish_json
    print(json.dumps(finish_json(c), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
