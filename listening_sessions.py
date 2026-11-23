"""Editable, atomic listening state; inspired by Seven and Sunny's seven-ears (MIT).

A resume replays the current view. Advancing is explicit and never needs a note.
Only this CLI layer writes state; modes.render remains a pure text operation.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import modes as M


def parser():
    ap = argparse.ArgumentParser(description="Listening choices over an existing UTF-8 card.")
    ap.add_argument("card", nargs="?", help="existing card, or - for stateless stdin")
    ap.add_argument("--mode", choices=("all", "first-listen", "lens", "vibe"))
    ap.add_argument("--lens", choices=M.LENSES)
    ap.add_argument("--section", type=int, help="1-based first-listen section")
    ap.add_argument("--from", dest="start", type=parse_time, metavar="M:SS")
    ap.add_argument("--until", type=parse_time, metavar="M:SS")
    ap.add_argument("--analysis", help="existing engine JSON; never invokes an engine")
    ap.add_argument("--lyrics", help="plain lyrics, used only for stanza matching")
    ap.add_argument("--resume", action="store_true", help="replay the saved view and notes")
    ap.add_argument("--next", action="store_true", help="advance a saved view without requiring notes")
    ap.add_argument("--note", action="append", default=[], help="append an editable note to this view")
    ap.add_argument("--edit-note", nargs=2, metavar=("N", "TEXT"), help="replace note N (1-based across this listen)")
    ap.add_argument("--status", action="store_true", help="list open listens")
    ap.add_argument("--finish", action="store_true", help="close a listen, retaining editable notes")
    ap.add_argument("--state-home", type=Path, help="listens directory (also RAVEN_LISTENS_HOME)")
    ap.add_argument("--no-state", action="store_true", help="render without saving a session")
    return ap


def parse_time(value):
    if not re.fullmatch(r"\d+:[0-5]\d(?:\.\d+)?", value):
        raise argparse.ArgumentTypeError("time must be M:SS (seconds 00–59)")
    return M.seconds(value)


def state_home(args):
    if args.state_home is not None:
        return args.state_home.expanduser()
    if os.environ.get("RAVEN_LISTENS_HOME"):
        return Path(os.environ["RAVEN_LISTENS_HOME"]).expanduser()
    if os.environ.get("RAVEN_EARS_HOME"):
        return Path(os.environ["RAVEN_EARS_HOME"]).expanduser() / "listens"
    return Path.home() / ".cache/raven/listens"


def session_path(home, card):
    path = Path(card).resolve()
    slug = re.sub(r"[^\w-]+", "-", path.stem.casefold()).strip("-")[:80] or "song"
    # Same titles in different directories must not overwrite each other's notes.
    suffix = hashlib.sha256(str(path).encode()).hexdigest()[:10]
    return home / f"{slug}-{suffix}" / "state.json"


def load(path):
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state.get("version") != 1 or state.get("mode") not in ("first-listen", "lens")
                or not isinstance(state["notes"], list) or not isinstance(state["card"], str)
                or not isinstance(state["section"], int) or not isinstance(state["closed"], bool)
                or not isinstance(state["fingerprint"], str)
                or (state["mode"] == "lens" and state["lens"] not in M.LENSES)):
            raise ValueError("invalid session fields")
        for index, note in enumerate(state["notes"], 1):
            if note["id"] != index or not isinstance(note["text"], str) or not isinstance(note["view"], str):
                raise ValueError("invalid note fields")
        return state
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"cannot read listen {path}: {error}; original state was left intact") from error


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a complete, durable write: interrupted writes keep the old view.
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".state-", delete=False) as handle:
            name = handle.name
            json.dump(state, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


@contextmanager
def locked(home):
    home.mkdir(parents=True, exist_ok=True)
    with (home / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def open_listens(home):
    if not home.exists():
        return []
    return [(path, load(path)) for path in sorted(home.glob("*/state.json"))]


def status(home):
    rows = []
    for _, state in open_listens(home):
        if not state["closed"]:
            view = state["lens"] if state["mode"] == "lens" else f"section {state['section']}"
            rows.append(f"{state['card']} — {state['mode']}, {view}; {len(state['notes'])} note(s)")
    return "\n".join((rows or ["No open listens."]) + [M.INVITATION]) + "\n"


def notes_text(state):
    if not state["notes"]:
        return ""
    return "Your notes so far (editable):\n" + "\n".join(
        f"  {n['id']}. [{n['view']}] {n['text']}" for n in state["notes"]) + "\n"


def validate(args):
    if args.status:
        if (args.card or args.mode or args.lens or args.section is not None or args.start is not None
                or args.until is not None or args.resume or args.next or args.note or args.edit_note
                or args.finish or args.analysis or args.lyrics or args.no_state):
            raise ValueError("--status only accepts --state-home")
        return
    if args.no_state and (args.resume or args.next or args.note or args.edit_note or args.finish):
        raise ValueError("session controls cannot be combined with --no-state")
    if args.resume and args.next:
        raise ValueError("choose --resume to replay or --next to advance")
    if args.mode and args.mode != "first-listen" and (args.section is not None or args.start is not None or args.until is not None):
        raise ValueError("--section/--from/--until require --mode first-listen")
    if args.lens and args.mode not in (None, "lens"):
        raise ValueError("--lens requires --mode lens")
    if args.section is not None and args.section < 1:
        raise ValueError("--section must be a positive integer")
    if args.until is not None and (args.start or 0) >= args.until:
        raise ValueError("--until must be after --from (or 0:00)")
    if args.mode in ("all", "vibe") and (args.resume or args.next or args.note or args.edit_note or args.finish):
        raise ValueError("sessions require first-listen or lens")


def read_card(path):
    if path == "-":
        return sys.stdin.buffer.read().decode("utf-8")
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def execute(args):
    validate(args)
    home = state_home(args)
    if args.status:
        return status(home)
    controls = bool(args.resume or args.next or args.note or args.edit_note or args.finish)
    if not args.card:
        if not controls:
            raise ValueError("supply a card, --resume, or --status")
        choices = [(p, s) for p, s in open_listens(home) if not s["closed"]]
        if len(choices) != 1:
            raise ValueError("supply a card to choose a listen; use --status to see open listens")
        args.card = choices[0][1]["card"]
    mode = args.mode or ("lens" if args.lens else "first-listen" if any(
        x is not None for x in (args.section, args.start, args.until)) else "all")
    if args.card == "-" and controls:
        raise ValueError("session controls need a saved card path, not stdin")
    persistent = args.card != "-" and not args.no_state and (controls or mode in ("first-listen", "lens"))
    if persistent:
        with locked(home):
            return session(args, home, mode, controls)
    text = read_card(args.card)
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8")) if args.analysis else None
    lyrics = Path(args.lyrics).read_text(encoding="utf-8") if args.lyrics else None
    return M.render(text, mode, args.lens, args.section if args.section is not None else 1,
                    analysis, lyrics, args.start, args.until)


def session(args, home, mode, controls):
    path = session_path(home, args.card)
    text = read_card(args.card)
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    state = load(path) if path.exists() else None
    if state is None and (args.resume or args.next or args.edit_note or args.finish or
                          (controls and mode not in ("first-listen", "lens"))):
        raise ValueError("no saved listen for this card; open it with --mode first-listen or --mode lens first")
    if state and state["fingerprint"] != fingerprint:
        raise ValueError(f"source card changed; move {path} aside to start a new listen (notes retained there)")
    if state is None:
        state = dict(version=1, card=str(Path(args.card).resolve()), fingerprint=fingerprint,
                     mode=mode, lens=args.lens, section=1, start=None, until=None,
                     analysis=None, lyrics=None, notes=[], closed=False)
    if args.analysis:
        state["analysis"] = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    if args.lyrics:
        state["lyrics"] = Path(args.lyrics).read_text(encoding="utf-8")
    if args.mode or args.lens or not controls:
        state["mode"] = mode
        state["closed"] = False
    if args.lens:
        state["lens"] = args.lens
    if state["mode"] == "lens":
        if args.start is not None or args.until is not None or args.section is not None:
            raise ValueError("--section/--from/--until require first-listen")
        if state["lens"] not in M.LENSES:
            raise ValueError("--mode lens needs --lens words|groove|harmony|instruments")
    elif args.lens:
        raise ValueError("--lens requires --mode lens")
    if args.section is not None:
        state.update(section=args.section, start=None, until=None)
    if args.next:
        if state["closed"]:
            raise ValueError("this listen is closed; reopen a view explicitly")
        if state["mode"] == "lens":
            index = M.LENSES.index(state["lens"]) + 1
            if index == len(M.LENSES):
                state["closed"] = True
            else:
                state["lens"] = M.LENSES[index]
        else:
            card = M.parse_card(text, state["analysis"])
            parts = M.label_sections(card, state["lyrics"])
            section = state["section"]
            if section < 1 or section > len(parts):
                raise ValueError("saved section is out of range")
            end = state["until"] if state["until"] is not None else parts[section - 1].end
            if end >= card.duration:
                state["closed"] = True
            else:
                index = next(i for i, part in enumerate(parts) if part.start <= end < part.end)
                state.update(section=index + 1, start=end, until=None)
    if args.start is not None:
        state["start"] = args.start
        # A new start defaults to the containing section's end.
        card = M.parse_card(text, state["analysis"])
        parts = M.label_sections(card, state["lyrics"])
        index = next((i for i, part in enumerate(parts) if part.start <= args.start < part.end), None)
        if index is None:
            raise ValueError("--from is outside available timed evidence")
        state.update(section=index + 1, until=None)
    if args.until is not None:
        state["until"] = args.until
    # Render/validate before any write, so bad ranges/edits leave the last view intact.
    output = M.render(text, state["mode"], state["lens"], state["section"], state["analysis"],
                      state["lyrics"], state["start"], state["until"])
    if state["mode"] == "lens":
        view = f"lens {state['lens']}"
    else:
        view = output.splitlines()[0].removeprefix("FIRST LISTEN: ")
    for note in args.note:
        state["notes"].append(dict(id=len(state["notes"]) + 1, view=view, text=note))
    if args.edit_note:
        try:
            index = int(args.edit_note[0])
        except ValueError as error:
            raise ValueError("--edit-note N must be a 1-based note number") from error
        if index < 1 or index > len(state["notes"]):
            raise ValueError("--edit-note N must name an existing note")
        state["notes"][index - 1]["text"] = args.edit_note[1]
    if args.finish:
        state["closed"] = True
    save(path, state)
    tail = notes_text(state)
    if state["closed"]:
        tail += "Listen closed; notes remain editable.\n"
    return output.removesuffix(M.INVITATION + "\n") + tail + M.INVITATION + "\n"


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        sys.stdout.write(execute(args))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        ap.exit(2, f"modes: {error}\n{M.INVITATION}\n")
    return 0
