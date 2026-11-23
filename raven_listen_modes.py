"""Repo-local wrapper for raven-listen; installed scripts are never edited here."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

from network_policy import offline, local_audio_path
from stage_status import names_note_bytes
import listening_sessions as S
from modes import INVITATION


def main(argv=None):
    ap = S.parser()
    ap.description = "Choose a listening pass over a saved card or a raven-listen song."
    for flag in ("--names", "--credits", "--artist", "--title"):
        ap.add_argument(flag)
    ap.add_argument("--no-credits", action="store_true")
    ap.add_argument("--only", choices=("music", "singer", "harmony"))
    args = ap.parse_args(argv)
    if args.card:
        local_audio_path(args.card, ap)
    try:
        S.validate(args)
        engine_args = []
        for name in ("lyrics", "names", "credits", "artist", "title", "only"):
            if getattr(args, name):
                engine_args += ["--" + name, getattr(args, name)]
        if args.no_credits:
            engine_args.append("--no-credits")
        explicit_view = bool(args.mode or args.lens or args.section is not None
                             or args.start is not None or args.until is not None)
        control_only = bool(args.resume or args.next or args.edit_note or args.finish
                            or (args.note and not explicit_view))
        source = Path(args.card).expanduser() if args.card and args.card != "-" else None
        direct_card = source is not None and source.suffix.lower() in (".txt", ".md", ".card")
        if args.status or args.card is None or args.card == "-" or direct_card:
            if direct_card:
                args.card = str(source)
            sys.stdout.write(S.execute(args))
            return 0
        if control_only:
            # Resume/notes over an audio name use its saved full card, with no lookup.
            song = source
        else:
            song = source if source.is_file() else None
        effective_mode = args.mode or ("lens" if args.lens else "first-listen" if any(
            x is not None for x in (args.section, args.start, args.until)) else "all")
        if offline() and not control_only:
            raise ValueError("external launcher disabled by RAVEN_EARS_OFFLINE=1; pass a saved card")
        runner = os.environ.get("RAVEN_LISTEN_COMMAND") or shutil.which("raven-listen")
        if effective_mode == "all" and not control_only:
            if not runner:
                raise ValueError("raven-listen is not installed; pass a saved card instead")
            result = subprocess.run([runner, args.card, *engine_args], capture_output=True)
            sys.stderr.buffer.write(names_note_bytes(result.stderr))
            sys.stdout.buffer.write(names_note_bytes(result.stdout))
            return result.returncode
        if song is None:
            fetch = shutil.which("raven-song")
            if not fetch:
                raise ValueError("raven-song is not installed; pass a saved card or audio path")
            result = subprocess.run([fetch, args.card], capture_output=True, text=True)
            if result.returncode or not result.stdout.strip():
                raise ValueError("raven-song failed to resolve the song")
            song = Path(result.stdout.strip().splitlines()[-1]).expanduser()
            if not song.is_file():
                raise ValueError("raven-song did not return an existing audio file")
        cards = Path(os.environ.get("RAVEN_CARDS_HOME", str(Path.home() / ".cache/raven/cards")))
        card = cards / (song.stem + ".txt")
        if not card.is_file():
            if control_only:
                raise ValueError("saved card is missing; pass its path or use --status")
            if not runner:
                raise ValueError("no cached card and raven-listen is not installed")
            # Suppress the full card on filtered passes. The original keeps its cache.
            result = subprocess.run([runner, str(song), *engine_args], capture_output=True)
            if result.returncode:
                raise ValueError("raven-listen failed; no listening view was saved")
            if not card.is_file():
                raise ValueError("raven-listen produced no saved card at the expected path")
        args.card = str(card)
        sys.stdout.write(S.execute(args))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        ap.exit(2, f"raven-listen-modes: {error}\n{INVITATION}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
