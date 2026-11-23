#!/usr/bin/env python3
"""Bring back written memories, read-only. See RELISTEN_COMPARE.md for formats."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import unicodedata

INVITATION = "and now, you"
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][0-9:.]+(?:Z|[+-]\d{2}:?\d{2})?)?")
NOTE_KEYS = ("note", "notes", "feeling", "feelings", "reflection", "response", "felt", "text")
DATE_KEYS = ("listened_at", "timestamp", "date", "created_at", "at")


def slug(song):
    """Punctuation/case insensitive, with version words retained."""
    text = unicodedata.normalize("NFKD", str(song).casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "-".join(re.findall(r"[^\W_]+", text, re.UNICODE))


def song_name(song):
    name = Path(str(song)).name
    if Path(name).suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".txt"):
        name = Path(name).stem
    return name


def _short(text, limit=240):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _date(value):
    if not isinstance(value, str):
        return ""
    match = DATE.search(value)
    if match:
        try:
            datetime.fromisoformat(match[0].replace("Z", "+00:00"))
            return match[0]
        except ValueError:
            pass
    return ""


@dataclass
class Memory:
    listens: list[tuple[str, str]] = field(default_factory=list)
    shelf: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    found: bool = False


def _json_records(value, fallback="", depth=0):
    if depth > 12:
        raise ValueError("memory JSON nested too deeply")
    if isinstance(value, list):
        return [row for item in value for row in
                ([(fallback, item.strip())] if isinstance(item, str) and item.strip()
                 else [] if item is None or item == ""
                 else _json_records(item, fallback, depth + 1))]
    if not isinstance(value, dict):
        raise ValueError("memory JSON needs an object or list")
    date = next((_date(value[k]) for k in DATE_KEYS if _date(value.get(k))), fallback)
    rows = []
    for key in NOTE_KEYS:
        note = value.get(key)
        if isinstance(note, str) and note.strip():
            rows.append((date, note.strip()))
        elif isinstance(note, (list, dict)):
            rows.extend(_json_records(note, date, depth + 1))
    for key in ("listens", "entries", "history"):
        if key in value:
            rows.extend(_json_records(value[key], date, depth + 1))
    return rows or ([(date, "")] if date else [])


def _markdown_records(text, fallback=""):
    rows, date, paragraph = [], fallback, []

    def flush():
        if paragraph:
            rows.append((date, " ".join(paragraph)))
            paragraph.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            date = _date(line) or date
        elif not line.strip():
            flush()
        else:
            paragraph.append(line.strip())
    flush()
    return rows or ([(date, "")] if date else [])


def _read(path):
    # Bound accidental dumps; no writes, mtimes as dates, or lossy decode.
    with path.open("r", encoding="utf-8") as stream:
        text = stream.read(1_000_001)
    if len(text) > 1_000_000:
        raise ValueError("file exceeds 1 MB")
    return text


def _title(song):
    return re.split(r"\s+[–—-]\s+", song, maxsplit=1)[-1]


def _heading_name(heading):
    heading = re.sub(r"^#+\s*", "", heading).strip()
    # Strip dated annotations, while retaining an explicitly different artist.
    parts = re.split(r"\s+[—–]\s+", heading, maxsplit=1)
    if len(parts) == 2 and re.search(r"\b(?:ear|notes?|listen)\b|\b\d{4}\b", parts[1], re.I):
        return parts[0].strip()
    return heading


def _shelf_sections(text, song, file_match=False):
    identities = {slug(song), slug(_title(song))}
    lines = text.splitlines()
    headings = [line for line in lines if line.startswith("#")]
    if file_match and (not headings or slug(_heading_name(headings[0])) in
                       ("notes", "shelf", "answer-key", "memories")):
        return ["\n".join(lines)]
    found = []
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        heading = _heading_name(line)
        # "Song (Artist)" is a common answer-key heading. Require the full
        # identity when the artist is supplied, except for a title-only heading.
        artist = re.split(r"\s+[–—-]\s+", song, maxsplit=1)[0] if _title(song) != song else ""
        heading = re.sub(rf"\s*\({re.escape(artist)}\)\s*$", "", heading, flags=re.I) if artist else heading
        if slug(heading) not in identities:
            continue
        level = len(line) - len(line.lstrip("#"))
        end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("#") and
                    len(lines[j]) - len(lines[j].lstrip("#")) <= level), len(lines))
        found.append("\n".join(lines[i + 1:end]))
    return found


def read_memory(song, listens_dir=None, music_dir=None, song_slug=None):
    """Read exact slug directories and song-specific shelf sections defensively.

    Use song_slug when the store's naming differs. No fuzzy title/lyric search.
    Missing dates stay unknown; shelf annotations never become heard facts.
    """
    memory = Memory()
    song = song_name(song)
    listens_dir = Path(listens_dir) if listens_dir is not None else Path.home() / ".cache/raven/listens"
    music_dir = Path(music_dir) if music_dir is not None else Path("example-music")
    wanted = song_slug or slug(song)
    if not wanted or Path(wanted).name != wanted or wanted in (".", ".."):
        raise ValueError("song slug must be a single directory name")
    try:
        directories = []
        for path in sorted(listens_dir.iterdir()):
            if not path.is_dir():
                continue
            matches = path.name == wanted or (song_slug is None and slug(path.name) == wanted)
            if not matches and song_slug is None:
                # listening_sessions uses an 80-character slug plus a path hash.
                # Verify the saved card identity; never guess by a title prefix.
                state_path = path / "state.json"
                try:
                    state = json.loads(_read(state_path))
                    saved_card = state.get("card") if isinstance(state, dict) else None
                    matches = isinstance(saved_card, str) and slug(song_name(saved_card)) == wanted
                except (OSError, ValueError, RecursionError):
                    if re.fullmatch(re.escape(wanted) + r"-[a-f0-9]{10}", slug(path.name)):
                        memory.warnings.append(f"prior note {path.name}/state.json unreadable")
            if matches:
                directories.append(path)
    except FileNotFoundError:
        directories = []
    except OSError as exc:
        directories = []
        memory.warnings.append(f"listen store unavailable ({type(exc).__name__})")
    for directory in directories:
        memory.found = True
        try:
            paths = sorted(directory.rglob("*"))
        except OSError as exc:
            memory.warnings.append(f"listen directory unavailable ({type(exc).__name__})")
            continue
        for path in paths:
            if path.suffix.lower() not in (".json", ".md", ".markdown") or not path.is_file():
                continue
            try:
                text = _read(path)
                fallback = _date(path.stem) or _date(path.parent.name)
                rows = _json_records(json.loads(text), fallback) if path.suffix.lower() == ".json" else _markdown_records(text, fallback)
                memory.listens.extend(rows)
            except (OSError, ValueError, RecursionError) as exc:
                memory.warnings.append(f"prior note {path.name} unreadable ({type(exc).__name__})")
    try:
        shelf_files = sorted(p for p in music_dir.rglob("*") if p.is_file() and
                             (p.suffix.lower() in (".md", ".markdown") or
                              re.search(r"shelf|notes", p.name, re.I)))
    except FileNotFoundError:
        shelf_files = []
    except OSError as exc:
        shelf_files = []
        memory.warnings.append(f"shelf unavailable ({type(exc).__name__})")
    identities = {slug(song), slug(_title(song))}
    for path in shelf_files:
        try:
            file_id = re.sub(r"^(?:ear-key|shelf|notes)-", "", slug(path.stem))
            for section in _shelf_sections(_read(path), song, file_id in identities):
                paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
                if paragraphs:
                    # An old answer key must not supply a second key/tempo/meter
                    # claim above today's card. Keep the reference if no note fits.
                    notes = [p for p in paragraphs if not re.search(r"\b(?:KEY|TEMPO|METER)\s*:", p, re.I)
                             and not re.match(r"(?:credits|key\s*/)", p, re.I)]
                    note = next((p for p in notes if re.search(r"\bfeel(?:s|ing)?\b|\bfelt\b|favou?rite|why it matters", p, re.I)),
                                notes[0] if notes else "reference found; no personal note identified")
                    memory.shelf.append((path.name, _short(note)))
        except (OSError, ValueError) as exc:
            memory.warnings.append(f"shelf file {path.name} unreadable ({type(exc).__name__})")
    memory.listens = sorted(dict.fromkeys(memory.listens), key=lambda row: row[0], reverse=True)
    return memory


def memory_header(memory, limit=3):
    """A small attributed reminder, not a replacement for today's listening."""
    lines = []
    dates = sorted({d[:10] for d, _ in memory.listens if d}, reverse=True)
    if dates:
        lines.append("Previous listens: " + ", ".join(dates[:8]) + (" …" if len(dates) > 8 else ""))
    for date, note in memory.listens[:limit]:
        label = f"{int(date[8:10])} {datetime.fromisoformat(date[:10]).strftime('%b')}" if date else "date unknown"
        lines.append(f"LAST TIME ({label}): {_short(note) if note else 'no written note found'}")
    for source, note in memory.shelf[:2]:
        lines.append(f"SHELF ({source}; written note, not today's hearing): {note}")
    if not memory.listens:
        if memory.found or memory.warnings:
            lines.insert(0, "previous listening unclear; no readable dated listen found")
        elif memory.shelf:
            lines.insert(0, "no previous listen recorded here; shelf note found")
        else:
            lines.insert(0, "first time hearing this one")
    lines.extend("MEMORY: " + warning for warning in memory.warnings)
    return "\n".join(lines)


def prepend_memory(card, memory):
    result = memory_header(memory) + ("\n\n" + card.rstrip() if card.strip() else "")
    # Preserve the original invitation and ensure every output ends with it.
    return result + ("\n" if result.splitlines()[-1] == INVITATION else "\n\n" + INVITATION + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("song", help="song name or audio path (audio is never opened)")
    parser.add_argument("--card", help="existing text card, or - for stdin")
    parser.add_argument("--listens-dir", type=Path)
    parser.add_argument("--music-dir", type=Path)
    parser.add_argument("--slug", help="exact directory name used by the listen store")
    args = parser.parse_args(argv)
    try:
        memory = read_memory(args.song, args.listens_dir, args.music_dir, args.slug)
        card = (sys.stdin.read() if args.card == "-" else _read(Path(args.card))) if args.card else ""
        print(prepend_memory(card, memory), end="")
    except (OSError, ValueError) as exc:
        print(f"relisten unavailable: {exc}\n{INVITATION}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
