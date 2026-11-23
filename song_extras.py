"""Optional song-only metadata. Never imported by the speech engine.

Call prepare only at a music CLI boundary. Listener tags are display evidence,
not detector input; Genius is called separately, after printing a saved card.
"""
import fcntl
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from network_policy import offline, require_online

UA = "raven-ears/1.0 (optional song metadata)"
ACOUSTID = "https://api.acoustid.org/v2/lookup"
LASTFM = "https://ws.audioscrobbler.com/2.0/"
GENIUS = "https://api.genius.com/"
# A conservative acceptance floor, not a measured probability of correctness.
MIN_SCORE = .90
TAG_FLOOR = 20
TAG_SHARE = .40
MAX_TAGS = 5
LORE_SEPARATOR = "— after the listening: the lore (Genius) —"
KEYS = {"AcoustID": "ACOUSTID_API_KEY", "Last.fm": "LASTFM_API_KEY", "Genius": "GENIUS_ACCESS_TOKEN"}


class LookupFailed(Exception):
    pass


def _short(value, limit):
    text = " ".join(str(value or "").split())
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _norm(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value or "").casefold()))


def _home():
    return Path(os.environ.get("RAVEN_EARS_HOME", Path.home() / ".local/share/raven-ears")) / "song-extras"


def _cached(service, identity, lookup, ttl=7 * 86400):
    """Cache only useful, already reduced answers; never credentials or requests."""
    directory = _home() / service
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = directory / (key + ".json")
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if ttl is None or time.time() - row["at"] < ttl:
            return row["answer"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    answer = lookup()
    if answer:
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    json.dump({"at": time.time(), "answer": answer}, handle, ensure_ascii=False, allow_nan=False)
                    handle.close()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError:
            pass  # A read-only cache must not lose a successful answer.
    return answer


def _request(service, url, params, *, token=None, post=False, fetch=None):
    require_online(service + " lookup")
    encoded = urllib.parse.urlencode(params)
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if post:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url if post else url + "?" + encoded,
                                 data=encoded.encode() if post else None, headers=headers)
    # Shared across processes, including sequential listen workers. Never retry
    # a rejected request; persist Retry-After so the next command also waits.
    directory = _home() / service
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".rate").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        try:
            next_at = float(lock.read() or 0)
        except ValueError:
            next_at = 0
        delay = next_at - time.time()
        gap = .4 if service == "AcoustID" else 1.05
        if delay > gap + .1:
            raise LookupFailed("rate limited; try later")
        if delay > 0:
            time.sleep(delay)
        next_at = time.time() + gap
        try:
            with (fetch or urllib.request.urlopen)(req, timeout=8) as response:
                result = json.loads(response.read(2_000_000).decode("utf-8"))
            if not isinstance(result, dict):
                raise LookupFailed("invalid response")
            if service == "Last.fm" and str(result.get("error")) == "29":
                next_at = time.time() + 60
                raise LookupFailed("API rate limited; skipped")
            return result
        except urllib.error.HTTPError as error:
            if error.code == 429:
                try:
                    retry_after = error.headers.get("Retry-After", "60")
                    try:
                        retry = float(retry_after)
                    except ValueError:
                        retry = parsedate_to_datetime(retry_after).timestamp() - time.time()
                except (ValueError, TypeError, AttributeError, OverflowError):
                    retry = 60
                next_at = time.time() + max(60, retry if math.isfinite(retry) else 60)
            raise LookupFailed(f"HTTP {error.code}; skipped") from None
        except (OSError, ValueError, UnicodeError):
            # Exception strings can contain request URLs (and API keys).
            raise LookupFailed("unavailable or invalid response; skipped") from None
        finally:
            lock.seek(0)
            lock.truncate()
            lock.write(str(next_at))


def _enabled(service, off=False):
    return not off and not offline() and bool(os.environ.get(KEYS[service], "").strip())


def _attempt(service, action):
    try:
        return action()
    except (LookupFailed, OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        reason = str(error) if isinstance(error, LookupFailed) else "unavailable or invalid response; skipped"
        print(f"{service}: {reason}", file=sys.stderr)
        return None


def _fingerprint(path):
    binary = shutil.which("fpcalc")
    if not binary:
        raise LookupFailed("fpcalc missing; fingerprint identification skipped")
    try:
        result = subprocess.run([binary, "-json", str(Path(path).resolve())],
                                capture_output=True, text=True, timeout=120, check=True)
        data = json.loads(result.stdout)
        duration = float(data["duration"])
        fingerprint = data["fingerprint"]
        if not math.isfinite(duration) or duration <= 0 or not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("invalid fingerprint")
        return duration, fingerprint
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError):
        raise LookupFailed("fpcalc failed; fingerprint identification skipped") from None


def identify(path, *, off=False, fetch=None):
    if not _enabled("AcoustID", off):
        return None

    def lookup():
        duration, fingerprint = _fingerprint(path)
        def remote():
            data = _request("AcoustID", ACOUSTID, dict(client=os.environ["ACOUSTID_API_KEY"],
                duration=int(duration), fingerprint=fingerprint, meta="recordings", format="json"), post=True, fetch=fetch)
            if data.get("status") != "ok":
                raise LookupFailed("API rejected lookup; skipped")
            candidates = {}
            for hit in data.get("results", []):
                score = float(hit.get("score", 0))
                if not math.isfinite(score) or not MIN_SCORE <= score <= 1:
                    continue
                for recording in hit.get("recordings", []):
                    artist = ", ".join(a["name"].strip() for a in recording.get("artists", []) if a.get("name", "").strip())
                    title = (recording.get("title") or "").strip()
                    recording_id = recording.get("id")
                    if artist and title and recording_id:
                        candidate = dict(artist=artist, title=title, score=score, recording_id=recording_id)
                        previous = candidates.get(recording_id)
                        if previous:
                            if (_norm(previous["artist"]), _norm(previous["title"])) != (_norm(artist), _norm(title)):
                                return None  # Conflicting metadata for one ID cannot cast a reliable vote.
                            if previous["score"] >= score:
                                continue
                        candidates[recording_id] = candidate
            groups = {}
            for candidate in candidates.values():
                pair = (_norm(candidate["artist"]), _norm(candidate["title"]))
                groups.setdefault(pair, []).append(candidate)
            if not groups:
                return None
            ranked = sorted(groups.values(), key=len, reverse=True)
            agreeing, total = len(ranked[0]), len(candidates)
            # Keep unanimous matches, including a single linked recording.
            # Community disagreements need both a two-thirds share and a two-vote lead.
            if len(ranked) > 1 and (3 * agreeing < 2 * total or agreeing - len(ranked[1]) < 2):
                return None
            return dict(max(ranked[0], key=lambda c: c["score"]),
                        agreeing_recordings=agreeing, total_recordings=total)
        return _cached("AcoustID", [duration, fingerprint], remote, ttl=None)
    return _attempt("AcoustID", lookup)


def filter_tags(data, artist, title):
    return tag_evidence(data, artist, title)["tags"]


def tag_evidence(data, artist, title):
    tags = (data.get("toptags") or {}).get("tag", [])
    if isinstance(tags, dict):
        tags = [tags]
    kept = {}
    top = 0
    for tag in tags:
        name = _short(tag.get("name"), 48)
        norm = _norm(name)
        try:
            weight = float(tag.get("count", 0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight):
            continue
        top = max(top, weight)
        junk = (not norm or norm in (_norm(artist), _norm(title))
                or re.search(r"\b(?:seen\s*live|favou?rites?|favou?rite|favs?|loved|love|playlist|my|scrobbles?)\b", norm)
                or re.search(r"\b(?:19|20)\d{2}\b", norm)
                or re.fullmatch(r"\d{2,4}s|\d+", norm)
                or "http" in norm or len(name) > 40)
        if not junk and math.isfinite(weight) and weight >= TAG_FLOOR:
            if norm not in kept or weight > kept[norm][0]:
                kept[norm] = (weight, name)
    ranked = [(weight, name) for weight, name in sorted(kept.values(), key=lambda row: -row[0])
              if weight >= TAG_SHARE * top][:MAX_TAGS]
    return dict(tags=[name for weight, name in ranked], top_count=top,
                counts=[dict(name=name, count=weight) for weight, name in ranked])


def feeling_extras(extras):
    """Apply the relative-support rule when the counts were saved.

    Legacy bundles kept only the reduced names (no counts), so the rule can't
    be checked there; Huginn, 16 Nov 2026: keep their row as it was rather
    than silently drop "soul, folk, haunting, ethereal" from four of six cards.
    A fresh listen saves the counts and gets the stricter filter.
    """
    extras = dict(extras or {})
    evidence = extras.get("lastfm_tag_evidence") or {}
    top = evidence.get("top_count")
    if not isinstance(top, (int, float)) or not math.isfinite(top) or top <= 0:
        return extras
    supported = {r.get("name") for r in evidence.get("counts") or []
                 if isinstance(r.get("count"), (int, float)) and math.isfinite(r["count"])
                 and r["count"] >= max(TAG_FLOOR, TAG_SHARE * top)}
    extras["lastfm_tags"] = [name for name in extras.get("lastfm_tags") or [] if name in supported][:MAX_TAGS]
    return extras


def listener_tags(artist, title, *, off=False, fetch=None):
    evidence = listener_tag_evidence(artist, title, off=off, fetch=fetch)
    return evidence["tags"] if evidence else None


def listener_tag_evidence(artist, title, *, off=False, fetch=None):
    if not _enabled("Last.fm", off) or not artist or not title:
        return None
    def lookup():
        data = _request("Last.fm", LASTFM, dict(method="track.getTopTags", artist=artist, track=title,
            api_key=os.environ["LASTFM_API_KEY"], format="json", autocorrect=0), fetch=fetch)
        if data.get("error"):
            raise LookupFailed("API rate limited; skipped" if str(data["error"]) == "29" else "API rejected lookup; skipped")
        attributes = (data.get("toptags") or {}).get("@attr") or {}
        if any(attributes.get(field) and _norm(attributes[field]) != _norm(expected)
               for field, expected in (("artist", artist), ("track", title))):
            return None
        return tag_evidence(data, artist, title)
    # Old caches contain names only, so their relative support is unknowable.
    return _attempt("Last.fm", lambda: _cached("Last.fm", [artist, title, "relative-v1"], lookup))


def add_args(ap):
    ap.add_argument("--no-acoustid", action="store_true", help="disable optional fingerprint identification")
    ap.add_argument("--no-lastfm", action="store_true", help="disable optional listeners' tags")


def prepare(args, path, *, fetch=None):
    """Song CLI only. Explicit/partial flags, tags and split filenames all win."""
    result = {}
    if os.environ.get("RAVEN_SONG_EXTRAS_WORKER") == "1":
        return result
    want_id = _enabled("AcoustID", getattr(args, "no_acoustid", False))
    want_tags = _enabled("Last.fm", getattr(args, "no_lastfm", False))
    if not (want_id or want_tags):
        return result
    import credits
    artist, title, _ = credits.identify(path, args.artist, args.title)
    if want_id and not artist and not title:
        identification = identify(path, fetch=fetch)
        if identification:
            result["identification"] = identification
            artist, title = identification["artist"], identification["title"]
            args.artist, args.title = artist, title
    if want_tags:
        evidence = listener_tag_evidence(artist, title, fetch=fetch)
        if evidence and evidence["tags"]:
            result["lastfm_tags"] = evidence["tags"]
            result["lastfm_tag_evidence"] = evidence
    return result


def decorate(text, extras):
    lines = []
    identity = (extras or {}).get("identification")
    if identity:
        agreement = ""
        if "agreeing_recordings" in identity and "total_recordings" in identity:
            agreement = f", {identity['agreeing_recordings']} of {identity['total_recordings']} linked recordings agree"
        lines.append(f"{_short(identity['artist'], 100)} — {_short(identity['title'], 100)}: "
                     f"identified by fingerprint (AcoustID {identity['score']:.2f}{agreement})")
    tags = (extras or {}).get("lastfm_tags")
    if tags:
        lines.append("LISTENERS SAY (Last.fm tags): " + ", ".join(tags))
    if not lines:
        return text
    # Listener descriptions follow the feeling invitation, before NEXT LISTEN.
    invitation = text.find("\nAnd now, you:")
    if invitation >= 0:
        from stage_status import NOTES_WIDTH
        lines = [textwrap.fill(line, width=NOTES_WIDTH, subsequent_indent="  ") for line in lines]
        end = text.find("\n", invitation + 1)
        while end >= 0 and text[end + 1:end + 2] == ' ':
            end = text.find("\n", end + 1)
        end = len(text) if end < 0 else end + 1
        return text[:end].rstrip('\n') + "\n" + "\n".join(lines) + "\n" + text[end:]
    # Keep the listener invitation last, as in the existing card format.
    suffix = "and now, you\n"
    if text.endswith(suffix):
        return text[:-len(suffix)] + "\n".join(lines) + "\n" + suffix
    return text + "\n" + "\n".join(lines) + "\n"


def _genius_lore(artist, title, fetch):
    token = os.environ["GENIUS_ACCESS_TOKEN"]
    def request(endpoint, **params):
        data = _request("Genius", GENIUS + endpoint, dict(text_format="plain", **params), token=token, fetch=fetch)
        if int((data.get("meta") or {}).get("status", 200)) != 200:
            raise LookupFailed("API rejected lookup; skipped")
        return data.get("response") or {}
    hits = request("search", q=artist + " " + title).get("hits", [])
    matches = [h["result"] for h in hits if h.get("type") == "song"
               and _norm(h.get("result", {}).get("title")) == _norm(title)
               and _norm((h.get("result", {}).get("primary_artist") or {}).get("name")) == _norm(artist)]
    if not matches:
        raise LookupFailed("no matching song; skipped")
    song_id = int(matches[0]["id"])
    song = request(f"songs/{song_id}").get("song") or {}
    if (_norm(song.get("title")) != _norm(title)
            or _norm((song.get("primary_artist") or {}).get("name")) != _norm(artist)):
        raise LookupFailed("song metadata differs; skipped")
    description = _short((song.get("description") or {}).get("plain"), 1000)
    referents = request("referents", song_id=song_id, per_page=50, page=1).get("referents", [])
    annotations = []
    for referent in referents:
        # Only brief fragments of this song's own lyrics, never full lyrics.
        # 16 Nov 2026: Genius's `classification` is a review state
        # (verified/accepted/unreviewed/...), not a kind, so the old
        # `== "lyrics"` test silently dropped every annotation (LRPD had 9).
        kind = (referent.get("annotatable") or {}).get("type")
        if kind not in (None, "Song"):
            continue
        state = referent.get("classification")
        for annotation in referent.get("annotations", []):
            body = _short((annotation.get("body") or {}).get("plain"), 500)
            if not body:
                continue
            authors = [a.get("user", {}).get("name") for a in annotation.get("authors", [])]
            annotations.append(dict(fragment=_short(referent.get("fragment"), 80), text=body,
                by=", ".join(_short(a, 60) for a in authors[:3] if a) or "Genius contributors",
                votes=int(annotation.get("votes_total", 0)), id=int(annotation["id"]),
                reviewed=state in ("verified", "accepted")))
    # Reviewed annotations first, then by votes; an unreviewed one is labelled as such.
    annotations.sort(key=lambda a: (not a["reviewed"], -a["votes"]))
    if not description and not annotations:
        raise LookupFailed("no lore available; skipped")
    return dict(song_id=song_id, description=description, annotations=annotations[:3])


def lore(song, path, *, fetch=None):
    """Called only AFTER the CLI has printed/flushed the saved listening card."""
    if not _enabled("Genius"):
        return "Genius: offline; skipped\n" if offline() else "Genius: set GENIUS_ACCESS_TOKEN; skipped\n"
    identity = (song.get("song_extras") or {}).get("identification") or {}
    options = song["manifest"].get("options") or {}
    artist, title = identity.get("artist") or options.get("artist"), identity.get("title") or options.get("title")
    if not (artist and title):
        import credits
        tag_artist, tag_title = credits.file_tags(path)
        file_artist, file_title = credits.name_from_file(options.get("metadata_filename") or "")
        artist, title = artist or tag_artist or file_artist, title or tag_title or file_title
    if not (artist and title):
        return "Genius: artist/title unavailable; skipped\n"
    result = _attempt("Genius", lambda: _cached("Genius", [artist, title, "annotations-v2"], lambda: _genius_lore(artist, title, fetch)))
    if not result:
        return ""
    lines = [LORE_SEPARATOR, f"Genius: https://genius.com/songs/{result['song_id']}"]
    if result["description"]:
        lines.append("Description (Genius contributors): " + result["description"])
    for row in result["annotations"]:
        review = "" if row.get("reviewed", True) else ", unreviewed"
        lines.append(f"“{row['fragment']}” — {row['text']} ({row['by']}, Genius annotation {row['id']}{review})")
    return "\n".join(lines) + "\n"
