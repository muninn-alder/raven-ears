#!/usr/bin/env python3
"""Listen once, then read saved song cards. Card text and JSON include lyrics.

    python listen.py song.mp3 [--fresh] [--view singer|melody]
    python listen.py song.mp3 --delete
    python listen.py --delete-all

RAVEN_EARS_HOME/songs, otherwise ~/.cache/raven/cards/songs, holds local bundles.
--delete only removes these bundles, not stems, lyrics or other engine caches.
Melody is derived once from captured vocal F0/words and the music card's key.
The default shows two lyric slices; --view melody reads the whole saved ABC.
--view full summarises uncertain stops; --view detail retains their diagnostics.
"""
import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import time

HERE = Path(__file__).resolve().parent
VERSION = "listen-once-1"
PASSES = ("singer", "music", "harmony")
VIEWS = ("full", "detail", "music", "singer", "harmony", "melody", "first-listen", "summary")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--view", choices=VIEWS, default="first-listen")
    ap.add_argument("--fresh", action="store_true", help="replace the saved song by running every pass again")
    ap.add_argument("--quick", action="store_true",
                    help="four stems; skip lyric checking for clean synced LRCLIB lyrics, and the key-change scan")
    ap.add_argument("--run-missing", action="store_true", help="retry failed/not-run passes needed by this view")
    ap.add_argument("--only", choices=PASSES, help="on a new listen, leave other passes explicitly not run")
    ap.add_argument("--delete", action="store_true", help="delete this song's saved bundle")
    ap.add_argument("--delete-all", action="store_true", help="delete all saved song bundles")
    ap.add_argument("--json", action="store_true", help="print the saved bundle, including manifest and all analyses")
    from listening_sheet import add_picture_args
    add_picture_args(ap, help="draw a local listening sheet beside the saved song")
    ap.add_argument("--by-ear", metavar="FILE", help="picture-only m:ss labels marked by a human (implies --picture)")
    ap.add_argument("--history", action="store_true", help="show per-pass listening dates/counts from the bundle; do no work")
    ap.add_argument("--lore", action="store_true", help="read Genius lore after an existing saved card")
    import song_extras
    song_extras.add_args(ap)
    scopes = {"names": "music and harmony only; singer uses its default names",
              "no-lyrics": "music and harmony only; singer may still resolve/transcribe lyrics",
              "keep-stems": "singer only", "no-singer": "singer only",
              "no-meter": "singer only; music still measures meter",
              "no-affect": "skip audEERING in the singer pass (also RAVEN_EARS_NO_AFFECT=1)",
              "no-stems-timeline": "singer only", "no-cache": "harmony only",
              "no-instruments": "singer and harmony only; music still measures sound/vocals"}
    for name in ("lyrics", "names", "credits", "artist", "title", "album", "keep-stems"):
        ap.add_argument("--" + name, help=scopes.get(name, "forwarded to singer, music and harmony"))
    for name in ("no-lyrics", "no-lyrics-lookup", "no-credits", "no-cache", "no-instruments",
                 "no-singer", "no-meter", "no-stems-timeline", "no-affect"):
        ap.add_argument("--" + name, action="store_true",
                        help=scopes.get(name, "forwarded to singer, music and harmony"))
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), help="all three passes via RAVEN_EARS_DEVICE")
    keys = ap.add_mutually_exclusive_group()
    keys.add_argument("--key-changes", action="store_true", default=None,
                      help="harmony key-change scan (default on for fresh normal listens; off with --quick)")
    keys.add_argument("--no-key-changes", dest="key_changes", action="store_false",
                      help="skip the harmony key-change scan")
    return ap


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ears_version():
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git = VERSION
    # A commit alone misses the uncommitted changes limbs are asked to leave.
    h = hashlib.sha256()
    sources = sorted(HERE.glob("*.py")) + sorted(HERE.glob("requirements*.txt"))
    for path in sources:
        if not path.name.startswith("test_"):
            h.update(path.name.encode() + b"\0" + path.read_bytes())
    return git + ":" + h.hexdigest()


def store_home():
    if os.environ.get("RAVEN_EARS_HOME"):
        return Path(os.environ["RAVEN_EARS_HOME"]).expanduser() / "songs"
    cards = os.environ.get("RAVEN_CARDS_HOME") or os.environ.get("RAVEN_CARDS_DIR")
    return Path(cards).expanduser() / "songs" if cards else Path.home() / ".cache/raven/cards/songs"


@contextmanager
def home_context():
    """No-baseline CLI runs cannot write through persistent home/cache overrides."""
    if os.environ.get("RAVEN_EARS_NO_BASELINE") != "1":
        yield
        return
    before = dict(os.environ)
    bytecode = sys.dont_write_bytecode
    # Keep even temporary writes in this checkout, as the house rules require.
    with tempfile.TemporaryDirectory(prefix=".listen-home-", dir=HERE) as directory:
        root = Path(directory)
        homes = {"HOME": "home", "RAVEN_EARS_HOME": "ears", "XDG_CACHE_HOME": "cache",
                 "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data", "XDG_STATE_HOME": "state",
                 "RAVEN_STEMS_CACHE": "stems", "RAVEN_LYRICS_CACHE": "lyrics",
                 "RAVEN_HARMONY_CACHE": "harmony", "RAVEN_PANNS_CACHE": "panns",
                 "RAVEN_CARDS_HOME": "cards", "RAVEN_CARDS_DIR": "cards",
                 "HF_HOME": "huggingface", "TORCH_HOME": "torch", "TMPDIR": "tmp",
                 "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
                 "HF_HUB_CACHE": "huggingface/hub", "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
                 "TRANSFORMERS_CACHE": "huggingface/transformers", "TORCH_EXTENSIONS_DIR": "extensions",
                 "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton", "JOBLIB_TEMP_FOLDER": "tmp",
                 "TMP": "tmp", "TEMP": "tmp"}
        try:
            for key, suffix in homes.items():
                path = root / suffix
                path.mkdir(parents=True, exist_ok=True)
                os.environ[key] = str(path)
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            sys.dont_write_bytecode = True
            yield
        finally:
            os.environ.clear()
            os.environ.update(before)
            sys.dont_write_bytecode = bytecode


def options(args):
    result = {name: getattr(args, name) for name in
              ("artist", "title", "album", "no_lyrics", "no_lyrics_lookup", "no_credits",
               "no_cache", "no_instruments", "no_singer", "no_meter", "no_stems_timeline", "no_affect",
               "key_changes")}
    if args.quick:
        result["quick"] = True
    result["key_changes"] = not args.quick and args.key_changes is not False
    for name in ("lyrics", "credits", "names"):
        path = getattr(args, name)
        if name == "names" and not path:
            default = Path(os.environ.get("RAVEN_EARS_HOME", Path.home() / ".local/share/raven-ears")) / "names.txt"
            path = default if default.is_file() else None
        result[name + "_sha256"] = (file_hash(Path(path).expanduser())
                                    if path and Path(path).expanduser().exists() else None)
    result["device"] = args.device or os.environ.get("RAVEN_EARS_DEVICE", "auto")
    # Untagged files can use their basename for automatic metadata lookups.
    result["metadata_filename"] = Path(args.audio).name if not args.artist or not args.title else None
    # Record output-affecting environment, never tokens or cache directory names.
    result["environment"] = {name: os.environ.get(name, "") for name in
                             ("RAVEN_EARS_OFFLINE", "RAVEN_NO_CREDITS", "RAVEN_NO_LYRICS_LOOKUP",
                              "RAVEN_NO_INSTRUMENTS", "RAVEN_NO_SIX_STEMS", "RAVEN_EARS_LANG",
                              "RAVEN_EARS_NAMES_MODE", "RAVEN_EARS_WHISPER", "WHISPER_MODEL", "RAVEN_EARS_PROFILE",
                              "RAVEN_LYRICS_WHISPER", "RAVEN_EARS_NO_AFFECT")}
    result["lookups"] = {
        "credits": not (args.no_credits or args.credits or os.environ.get("RAVEN_NO_CREDITS")
                        or os.environ.get("RAVEN_EARS_OFFLINE", "").lower() in ("1", "true", "yes", "on")),
        "lyrics": not (args.no_lyrics_lookup or args.lyrics
                       or os.environ.get("RAVEN_NO_LYRICS_LOOKUP", "").lower() in ("1", "true", "yes", "on")
                       or os.environ.get("RAVEN_EARS_OFFLINE", "").lower() in ("1", "true", "yes", "on"))}
    import song_extras
    if any(os.environ.get(song_extras.KEYS[s]) for s in ("AcoustID", "Last.fm")):
        result["song_extras"] = {"acoustid": song_extras._enabled("AcoustID", args.no_acoustid),
                                 "lastfm": song_extras._enabled("Last.fm", args.no_lastfm)}
    return result


def pass_args(name, args):
    values = ["credits", "artist", "title", "album", "lyrics"]
    switches = ["no_credits", "no_lyrics_lookup"]
    if name != "singer":
        values += ["names"]
        switches += ["no_lyrics"]
    if name == "singer":
        values += ["keep_stems"]
        switches += ["no_singer", "no_meter", "no_stems_timeline", "no_affect"]
    if name != "music":
        switches += ["no_instruments"]
    if name == "harmony":
        switches += ["no_cache"]
    if args.quick:
        switches += ["quick"]
    flags = []
    for key in values:
        value = getattr(args, key)
        if value:
            flags += ["--" + key.replace("_", "-"), value]
    flags += ["--" + key.replace("_", "-") for key in switches if getattr(args, key)]
    if name == "music" and args.picture:
        flags += ["--_picture-data"]
    if name == "harmony":
        flags += ["--key-changes" if not args.quick and args.key_changes is not False else "--no-key-changes"]
    return [str(Path(args.audio).expanduser().resolve()), *flags]


def forwarded_options(name, args, requested):
    """Only explicit CLI options actually sent to this pass, plus its device env."""
    argv = iter(pass_args(name, args)[1:])
    result = {}
    for flag in argv:
        key = flag[2:].replace("-", "_")
        if key == "_picture_data":
            continue
        if key in ("key_changes", "no_key_changes"):
            result["key_changes"] = key == "key_changes"
            continue
        value = getattr(args, key)
        if not isinstance(value, bool):
            next(argv)
        if key in ("lyrics", "names", "credits"):
            key, value = key + "_sha256", requested[key + "_sha256"]
        elif key == "keep_stems":
            value = Path(value).name
        result[key] = value
    result["device"] = requested["device"]
    return result


def input_hashes(args, strict=False):
    # A supplied file that has since gone is recorded as None, so a saved read
    # still works and reports changed options (review of PR #6, 10 Nov 2026).
    # During a listen (strict) a vanished file is an error, as before.
    def hashed(name):
        path = Path(getattr(args, name)).expanduser()
        if not strict and name != "audio" and not path.exists():
            return None
        return file_hash(path)
    return {name: hashed(name) for name in ("audio", "lyrics", "names", "credits") if getattr(args, name)}


def changed_analysis_flags(args, requested, saved):
    """Compare explicit overrides only. An ordinary read keeps the saved settings."""
    flags = []
    for key, value in requested.items():
        name = key.removesuffix("_sha256")
        explicit = args.key_changes is not None if key == "key_changes" else getattr(args, name, None)
        if explicit and value != saved.get(key, False if key in ("key_changes", "quick") else None):
            flags.append("--" + ("no-key-changes" if key == "key_changes" and not args.key_changes
                                 else name.replace("_", "-")))
    for flag, service in (("no_acoustid", "acoustid"), ("no_lastfm", "lastfm")):
        if getattr(args, flag) and saved.get("song_extras", {}).get(service):
            flags.append("--" + flag.replace("_", "-"))
    return flags


def analysis_basenames(value):
    """Captured analysis is portable; raw cards and diagnostics remain evidence."""
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if key == "jev" and isinstance(child, dict):
                # These contain only API text state and model output.
                # Path-like text must survive exactly for a repeatable request.
                out[key] = {k: v if k in ("payload", "answer") else analysis_basenames(v)
                            for k, v in child.items()}
            elif (key == "curves" and isinstance(child, dict)
                  and child.get("encoding") == "f64le-zlib-base64"
                  and re.fullmatch(r"[A-Za-z0-9+/=]*", str(child.get("data", "")))):
                # Packed STOPS energy is base64, never a path. The path regexes
                # below read "+/abc/def" as a path and cut it to "def", which
                # corrupted every fresh bundle's curves (#50, 13 Nov 2026).
                out[key] = dict(child)
            else:
                out[key] = analysis_basenames(child)
        return out
    if isinstance(value, list):
        return [analysis_basenames(child) for child in value]
    if isinstance(value, str):
        if "\n" not in value and Path(value).is_absolute():
            return Path(value).name
        if PureWindowsPath(value).is_absolute():
            return PureWindowsPath(value).name
        value = re.sub(r"([\"'])(/[^\n\"']+|[A-Za-z]:[\\/][^\n\"']+)\1",
                       lambda match: match[1] + (PureWindowsPath(match[2]).name
                           if PureWindowsPath(match[2]).drive else Path(match[2]).name) + match[1], value)
        value = re.sub(r"(?<![\w:/])[A-Za-z]:[\\/][^\s\"'<>]+",
                       lambda match: PureWindowsPath(match[0]).name, value)
        return re.sub(r"(?<![\w:/])/(?:[^\s/\"'<>]+/)*[^\s/\"'<>]+",
                      lambda match: Path(match[0]).name, value)
    return value


def worker(name, analysis_path, argv):
    """Internal subprocess entry; engines are imported only here."""
    def capture(data):
        Path(analysis_path).write_text(json.dumps(data, ensure_ascii=False,
            default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")
    # Opt-in measurement sidecars; never add profiling data to a saved card.
    timing_dir = os.environ.get("RAVEN_EARS_TIMING_DIR")
    if timing_dir:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            return importlib.import_module(name).main(argv, capture=capture) or 0
        finally:
            profiler.disable()
            directory = Path(timing_dir)
            directory.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(directory / (name + ".pstats")))
    return importlib.import_module(name).main(argv, capture=capture) or 0


def print_progress(started, label, stream):
    elapsed = int(time.monotonic() - started)
    print(f"  [{elapsed // 60}:{elapsed % 60:02d}] {label}…", file=stream, flush=True)


def run_pass(name, args):
    with tempfile.TemporaryDirectory(prefix=".listen-pass-", dir=store_home()) as directory:
        analysis = Path(directory) / "analysis.json"
        env = dict(os.environ, RAVEN_EARS_DEVICE=args.device or os.environ.get("RAVEN_EARS_DEVICE", "auto"))
        if any(os.environ.get(key) for key in ("ACOUSTID_API_KEY", "LASTFM_API_KEY")):
            env["RAVEN_SONG_EXTRAS_WORKER"] = "1"
        if args.quick:
            env["RAVEN_NO_SIX_STEMS"] = "1"
        command = [sys.executable, str(HERE / "listen.py"), "--_worker", name, str(analysis)]
        started = getattr(args, "_progress_started", None)
        # Keep worker stderr captured as diagnostics. A separate fd carries only progress.
        channel = os.fdopen(os.dup(sys.stderr.fileno()), "w") if started is not None else nullcontext()
        with channel as stream:
            progress_options = {}
            if stream is not None:
                fd = stream.fileno()
                command += ["--_progress", str(fd), str(started)]
                progress_options["pass_fds"] = (fd,)
            result = subprocess.run([*command, *pass_args(name, args)],
                                    capture_output=True, text=True, env=env, **progress_options)
        data, json_error = None, ""
        try:
            data = json.loads(analysis.read_text(encoding="utf-8")) if analysis.exists() else None
            if data is not None and not isinstance(data, dict):
                raise ValueError("analysis is not an object")
            json.dumps(data, allow_nan=False)
        except ValueError as error:
            data, json_error = None, f"invalid JSON analysis: {error}"
        reason = ""
        if result.returncode:
            detail = (result.stderr.strip() or result.stdout.strip())[-2000:]
            reason = f"exit {result.returncode}: {detail or 'no diagnostic returned'}"
        elif data is None:
            reason = json_error or "pass returned no JSON analysis"
        return {"status": "failed" if reason else "ran", "reason": reason,
                "text": result.stdout, "json": data, "diagnostics": result.stderr}


def save(path, song):
    for card in song["cards"].values():
        card["json"] = analysis_basenames(card["json"])
    if "melody" in song:
        song["melody"] = analysis_basenames(song["melody"])
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".song-", delete=False) as handle:
            name = handle.name
            json.dump(private_notes(song), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def load(path):
    try:
        return private_notes(validate_song(json.loads(path.read_text(encoding="utf-8"))))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"cannot read saved song: {error}; original left intact; use --fresh to replace it") from error


def validate_song(song):
    if song.get("schema") != 1 or not isinstance(song.get("manifest"), dict):
        raise ValueError("invalid saved song; use --fresh to replace it")
    manifest = song["manifest"]
    for key in ("file_sha256", "duration_s", "ears_version", "options", "heard_at", "passes"):
        if key not in manifest:
            raise ValueError("incomplete manifest; use --fresh to replace it")
    for name in PASSES:
        row = manifest["passes"][name]
        card = song["cards"][name]
        if (row["status"] not in ("ran", "failed", "skipped", "not run") or not isinstance(row["reason"], str)
                or not isinstance(card["text"], str) or not isinstance(card["json"], (dict, type(None)))):
            raise ValueError("invalid pass record; use --fresh to replace it")
    datetime.fromisoformat(manifest["heard_at"])
    if "melody" in song:
        melody = song["melody"]
        if (melody["status"] not in ("ran", "failed", "not run")
                or not isinstance(melody["reason"], str) or not isinstance(melody["text"], str)
                or not isinstance(melody["json"], (dict, type(None)))
                or (melody["status"] == "ran" and not isinstance(melody["json"].get("hook_text"), str))):
            raise ValueError("invalid melody record")
    if not isinstance(manifest["options"], dict) or not isinstance(manifest["ears_version"], str):
        raise ValueError("invalid version/options")
    return song


def delete_bundle(home, path):
    """Remove only a validated regular bundle in a direct content-hash directory."""
    if (path.parent.parent != home or not re.fullmatch(r"[0-9a-f]{64}", path.parent.name)
            or path.parent.is_symlink() or path.is_symlink()):
        raise ValueError("refusing deletion: not a regular bundle in a hash directory")
    if not path.exists():
        return False
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("refusing deletion: bundle is not a regular file")
    song = load(path)
    if song["manifest"]["file_sha256"] != path.parent.name:
        raise ValueError("refusing deletion: bundle hash does not match directory")
    path.unlink()
    for name in ("listening-sheet.png", "listening-spectrum.npz"):
        sidecar = path.with_name(name)
        if sidecar.is_file() and not sidecar.is_symlink():
            sidecar.unlink()
    if not any(path.parent.iterdir()):
        path.parent.rmdir()
    return True


def history_text(song):
    lines = ["LISTENING RECORD (completed pass attempts, including failed attempts):"]
    for name in PASSES:
        row = song["manifest"]["passes"][name]
        count = row.get("listen_count")
        first = row.get("first_listened") or "unknown (older bundle)"
        last = row.get("last_listened", row.get("date")) or "unknown"
        if row["status"] == "not run" and not count:
            first = last = "never"
            count = 0
        lines.append(f"{name}: first listened {first}; last listened {last}; "
                     f"count {count if count is not None else 'unknown (older bundle)'}; "
                     f"saved status {row['status']}" + (f": {row['reason']}" if row['reason'] else ""))
    return "\n".join(lines) + "\n"


def pass_text(song, name, *, detail=False):
    row = song["manifest"]["passes"][name]
    text = song["cards"][name]["text"]
    if name == "singer" and row["status"] == "ran":
        import instrumental_voice
        import keyfuse
        text = instrumental_voice.saved_peak_pitch(text, song["cards"][name]["json"])
        music_key, harmony_key, spans = keyfuse.song_spelling_keys(song)
        text = keyfuse.singer_note_labels(text, keyfuse.spelling_context(music_key or harmony_key, spans))
        result = (song["cards"][name]["json"] or {}).get("stops")
        if isinstance(result, dict):
            import stops
            text = stops.saved_text(text, result, detail=detail)
    # Derived claims can be corrected from saved evidence without re-listening
    # or changing the captured cards/JSON on disk.
    meter = (song["cards"].get("singer", {}).get("json") or {}).get("meter")
    if isinstance(meter, dict) and song["manifest"]["passes"]["singer"]["status"] == "ran":
        import meter_trust
        if name == "music":
            text = meter_trust.tempo_text(text, meter)
        elif name == "singer":
            text = meter_trust.meter_text(text, meter)
    if name == "music" and row["status"] == "ran":
        import tempo_trust
        trusted = tempo_trust.saved_meter(song["cards"].get("singer", {}), song["manifest"]["passes"].get("singer"))
        tempo = (song["cards"]["music"].get("json") or {}).get("tempo") or {}
        text = tempo_trust.tempo_text(text, tempo.get("bpm"), trusted)
    if name == "harmony" and row["status"] == "ran":
        import key_opening
        text = key_opening.intro_text(text, song["cards"][name]["json"])
    if row["status"] == "ran":
        import instrumental_voice
        text = instrumental_voice.stem_text(text, instrumental_voice.saved_designation(
            song["cards"], song["manifest"]["passes"]))
    if row["status"] != "ran":
        return f"{name}: {row['status']}: {row['reason']}\n" + text
    return text


def private_notes(value):
    """Drop old prompt previews at the saved-bundle boundary."""
    from stage_status import names_note_text

    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, list):
            return [clean(value) for value in item]
        return names_note_text(item) if isinstance(item, str) else item

    return clean(value)


_ENTRY_ROW = re.compile(r"(?m)^  +(\d+:\d\d)  voice enters: [^\n]*\n?")
_TIMELINE_ROW = re.compile(r"  +(\d+:\d\d)  ")


def saved_voice_entry(text, data, singer, singer_status):
    """Repair a legacy timeline from the measured stem and timed words on file."""
    if "vocal_onset" in ((data or {}).get("band_presence") or {}):
        return text, data
    header = re.search(r"(?m)^TIMELINE\n((?:  [^\n]*\n?)*)", text)
    match = _ENTRY_ROW.search(header[1]) if header else None
    if not match:
        return text, data
    from feeling_card import stem_rows
    from lyrics_shared import LINE_SLACK, vocal_onset, vocal_spans
    from slim import seconds

    singer_data = singer.get("json") or {}
    rows = stem_rows(singer["text"], singer_data, singer_status)
    onset = vocal_onset(rows)
    if onset is None:
        return text, data
    timeline = (data or {}).get("timeline") or []
    old_t = next((r["t"] for r in timeline if r["text"].startswith("voice enters:")), seconds(match[1]))
    words = (singer_data.get("melody_input") or {}).get("words")
    first_silent = words and words[0][1] < onset and words[0][2] <= onset and abs(words[0][1] - old_t) <= 0.5
    if abs(old_t - onset) <= LINE_SLACK and not first_silent:
        return text, data

    from harmony import tt, voice_entry
    spans = vocal_spans(rows, singer_data.get("duration_s") or (data or {}).get("duration") or float("inf"))
    t, label = voice_entry([], words, spans, onset)
    if timeline:
        kept = [r for r in timeline if not r["text"].startswith("voice enters:")]
        data = dict(data, timeline=sorted(kept + [{"t": round(t, 1), "text": label}], key=lambda r: r["t"]))
    block = _ENTRY_ROW.sub("", header[1]).splitlines(keepends=True)
    for i, line in enumerate(block):
        stamp = _TIMELINE_ROW.match(line)
        if stamp and seconds(stamp[1]) > t:
            break
    else:
        i = len(block)
    block.insert(i, f"  {tt(t):>5}  {label}\n")
    return text[:header.start(1)] + "".join(block) + text[header.end(1):], data


def saved_event_quotes(text, data, cards, passes):
    from feeling_card import stem_rows
    from lyric_align import clean_writer_card
    from lyrics_shared import event_lyric_quote, supplied_lyrics, vocal_onset
    from slim import seconds

    for name in ("music", "singer", "harmony"):
        if passes[name]["status"] == "ran":
            card = cards[name]
            source_text, source_data = (text, data) if name == "harmony" else (card["text"], card.get("json"))
            source_text, source_data = clean_writer_card(source_text, source_data)
            _, placed = supplied_lyrics(source_data, source_text)
            if placed:
                break
    else:
        return text, data

    onset = ((data or {}).get("band_presence") or {}).get("vocal_onset")
    if onset is None:
        singer = cards["singer"]
        onset = vocal_onset(stem_rows(singer["text"], singer.get("json") or {}, passes["singer"]))

    def quote(t, label):
        when = onset if onset is not None and label.startswith("voice enters:") else t
        return event_lyric_quote(label, when, placed)

    timeline = (data or {}).get("timeline") or []
    times = {}
    for row in timeline:
        times.setdefault(row["text"], []).append(row["t"])
    if timeline:
        data = dict(data, timeline=[dict(row, text=quote(row["t"], row["text"])) for row in timeline])

    def block(match):
        def row(line):
            prefix, stamp, label = line.groups()
            t = seconds(stamp)
            measured = min(times.get(label, (t,)), key=lambda when: abs(when - t))
            return prefix + quote(measured if abs(measured - t) <= 0.55 else t, label)
        return re.sub(r"(?m)^( +(\d+:\d\d) +)([^\n]+)$", row, match[0])

    text = re.sub(r"(?m)^(?:TIMELINE|SURPRISE \([^\n]+\))\n(?: +[^\n]*\n?)*", block, text)
    # Current ACHE already quotes the aligned harmony rows at full precision.
    if not ((data or {}).get("lyrics_aligned") or {}).get("lines"):
        text = re.sub(r"(?m)^(ACHE (\d+:\d\d) +)([^\n]+)$",
                      lambda m: m[1] + quote(seconds(m[2]), m[3]), text)
    return text, data


def saved_card_readings(song):
    """Remove old prompt previews and refresh the saved card's claims."""
    import credits
    import keyfuse
    import lrclib
    from feeling_card import stem_rows
    from instruments import single_hedge
    from lyrics_shared import vocal_spans

    song = private_notes(song)
    cards = song["cards"]
    passes = song["manifest"]["passes"]
    duration = song["manifest"].get("duration_s") or 0
    activity = []
    if passes["music"]["status"] == "ran":
        vocals = (cards["music"]["json"] or {}).get("vocals") or {}
        if vocals.get("method") == "vocal stem":
            activity.extend(vocals.get("spans") or [])
    if passes["harmony"]["status"] == "ran":
        presence = (cards["harmony"]["json"] or {}).get("band_presence") or {}
        activity.extend(presence.get("vocal_spans") or [])
    if duration:
        singer = cards["singer"]
        activity.extend(vocal_spans(stem_rows(singer["text"], singer["json"] or {}, passes["singer"]), duration))
    for name, card in cards.items():
        if passes[name]["status"] == "ran":
            text, data = lrclib.saved_voice_marks(card["text"], card.get("json"), name, activity, duration)
            cards[name] = dict(card, text=text, json=data)
    music_key, harmony_key, spans = keyfuse.song_spelling_keys(song)
    vocal_key = keyfuse.spelling_context(music_key or harmony_key, spans)
    chord_key = keyfuse.spelling_context(harmony_key or music_key, spans)
    singer_card = cards["singer"]
    singer_data = singer_card.get("json")
    singer = (singer_data or {}).get("instruments") or {}
    voice = (singer_data or {}).get("singer")
    if isinstance(voice, dict) and isinstance(voice.get("lines"), list):
        lines = [keyfuse.singer_note_labels(line, vocal_key) if isinstance(line, str) else line
                 for line in voice["lines"]]
        singer_data = dict(singer_data, singer=dict(voice, lines=lines))
    if singer_data and singer_data.get("instrument_events"):
        singer_data = dict(singer_data, instrument_events=[
            dict(row, text=single_hedge(row["text"])) for row in singer_data["instrument_events"]])
    cards["singer"] = dict(singer_card, text=keyfuse.singer_note_labels(single_hedge(singer_card["text"]), vocal_key),
                           json=singer_data)
    rebuild_melody = False
    for name, field in (("music", "key_claim"), ("harmony", "key_verdict")):
        card = cards[name]
        text, data = card["text"], card.get("json")
        claim = (data or {}).get(field)
        updated = keyfuse.saved_key_claim(claim)
        if updated is not claim:
            if claim.get("line"):
                text = text.replace(claim["line"], updated["line"])
                if card.get("diagnostics"):
                    card["diagnostics"] = card["diagnostics"].replace(claim["line"], updated["line"])
            if claim.get("votes_line"):
                text = text.replace(claim["votes_line"], updated["votes_line"])
            data = dict(data, **{field: updated})
            if name == "harmony" and data.get("home_key") == claim.get("key"):
                data["home_key"] = updated["key"]
            if name == "music" and claim.get("working") != updated.get("working"):
                rebuild_melody = True
                if (data.get("key") or {}).get("key") == claim.get("working"):
                    data["key"] = dict(data["key"], key=updated["working"])
        if name == "harmony":
            passes = song["manifest"]["passes"]
            if passes["harmony"]["status"] == "ran":
                text, data = saved_voice_entry(text, data, cards["singer"], passes["singer"])
                text, data = saved_event_quotes(text, data, cards, passes)
            text = single_hedge(credits.saved_timeline_text(text, singer))
            text = keyfuse.harmony_note_labels(text, chord_key)
            if data and data.get("timeline"):
                data = dict(data, timeline=[
                    dict(row, text=keyfuse.harmony_note_labels(
                        single_hedge(credits.saved_timeline_text(row["text"], singer)), lambda _: chord_key(row.get("t"))))
                    for row in data["timeline"]])
        cards[name] = dict(card, text=text, json=data)
    if rebuild_melody and (song.get("melody") or {}).get("status") == "ran":
        import listen_melody
        melody = listen_melody.build(song)
        if melody["status"] == "ran":
            song["melody"] = melody
    return song


_LYRIC_ROW = r"\[\d+:\d\d≈?\][ \t]+[^\n]+"
_LYRIC_BLOCK = re.compile(r"(?m)^([ \t]*)LYRICS:[^\n]*\n((?:[ \t]+" + _LYRIC_ROW + r"\n)+)")


def full_lyrics(music, singer, harmony):
    """Keep the primary lyric block and the harmony annotations at their times."""
    import lyric_align
    from slim import split_voice_mark

    primary = _LYRIC_BLOCK.search(music)
    duplicate = _LYRIC_BLOCK.search(singer)
    def words(block):
        return [row.partition("]")[2].strip() for row in block[2].splitlines()]
    if primary and duplicate and words(primary) == words(duplicate):
        singer = (singer[:duplicate.start()] + duplicate[1] +
                  f"LYRICS: {len(duplicate[2].splitlines())} lines, shown above in MUSIC\n" +
                  singer[duplicate.end():])
        duplicate = None
    # A harmony row adds chords and other timed evidence. Keep those annotations
    # when a plain lyric row above has already supplied its text.
    shown = set()
    for block in (primary, duplicate):
        if block:
            shown.update(lyric_align.quote_line(split_voice_mark(lyric)[0])
                         for lyric in words(block))
    timeline = re.search(r"(?m)^LYRIC TIMELINE\n(?:[ \t]+[^\n]*\n)+", harmony)
    if shown and timeline:
        def annotation(match):
            indent, stamp, lyric, suffix = match.groups()
            if lyric not in shown:
                return match[0]
            return f"{indent}{stamp}] ♪ {suffix}\n" if suffix else ""
        projected = re.sub(r'(?m)^([ \t]+)(\[\d+:\d\d≈?)\][ \t]+("[^"\n]*")'
                           r'(?: \(no voice measured here\))?(?: — ([^\n]+))?\n',
                           annotation, timeline[0])
        if projected != timeline[0]:
            projected = projected.replace("LYRIC TIMELINE\n", "LYRIC TIMELINE\n  Lyrics shown above.\n", 1)
            harmony = harmony[:timeline.start()] + projected + harmony[timeline.end():]
    return music, singer, harmony


def full_text(song, *, detail=False):
    return _full_text(saved_card_readings(song), detail=detail)


def _full_text(song, *, detail):
    cards = tuple(pass_text(song, name, detail=detail) for name in ("music", "singer", "harmony"))
    if not detail:
        cards = full_lyrics(*cards)
    text = "\n\n".join(card.rstrip() for card in cards) + "\n"
    import key_changes
    harmony = song['cards'].get('harmony', {})
    result = (harmony.get('json') or {}).get('key_change_scan')
    if song['manifest']['passes'].get('harmony', {}).get('status') == 'ran':
        text = key_changes.card_text(text, result)
    return text


def render(song, view, notices=()):
    import song_extras
    import song_length
    from stage_status import NOTES_WIDTH
    song = saved_card_readings(song)
    text, used = _render(song, view, notices)
    extras = song.get("song_extras")
    if view == "first-listen":
        extras = song_extras.feeling_extras(extras)
    text = song_extras.decorate(text, extras)
    if view == "first-listen":
        # Keep the recording-length warning at the end of this reading order.
        annotated = song_length.annotate(text, song.get("length_check"))
        warning = annotated[:-len(text)] if text and annotated.endswith(text) else ""
        warning = "\n".join(textwrap.fill(row, width=NOTES_WIDTH, subsequent_indent="  ")
                            for row in warning.splitlines()) + ("\n" if warning else "")
        return text + warning, used
    return song_length.annotate(text, song.get("length_check")), used


def _render(song, view, notices=()):
    import doctored_audio
    import listen_melody
    import lyric_align
    import lyrics_shared
    import stage_status
    import song_length
    options = song["manifest"].get("options") or {}
    quick_note = ""
    if options.get("quick"):
        disabled = options.get("no_instruments") or options.get("environment", {}).get("RAVEN_NO_INSTRUMENTS")
        losses = ["instrument naming and performer matching" if disabled else
                  "the extra guitar/piano separation (instrument names less certain; no isolated guitar swells)"]
        lyric_skips = set()
        for name, card in song["cards"].items():
            data = card.get("json") or {}
            source = data.get("lyrics" if name == "singer" else "lyrics_source")
            if isinstance(source, dict) and source.get("quick_skipped"):
                lyric_skips.add(name)
        if lyric_skips:
            missing = "no unsure-line marks or recovered words for those lyrics"
            if "harmony" in lyric_skips:
                missing += "; no refrain check"
            if "singer" in lyric_skips:
                missing += "; no wordless lines, word-based pace, emphasis or melody notation"
            losses.append("checking synced lyrics against the singing (" + missing + ")")
        if options.get("key_changes") is False:
            losses.append("the key-change scan")
        skipped = ", ".join(losses[:-1]) + ", and " + losses[-1] if len(losses) > 1 else losses[0]
        quick_note = textwrap.fill("Quick listen. Skipped: " + skipped + ".", width=100) + "\n"
    cards = {}
    changed, headers_changed = False, False
    evidence = "\n".join(c["text"] for n, c in song["cards"].items()
                         if song["manifest"]["passes"].get(n, {}).get("status", "ran") == "ran")
    for name, card in song["cards"].items():
        text, data = lyric_align.clean_writer_card(song_length.strip(card["text"]), card["json"],
                title=options.get("title"), filename=options.get("metadata_filename"))
        text = stage_status.instrumental_text(lyrics_shared.clean_gap_claims(text, evidence, data), evidence)
        text = stage_status.display_text(text, data)
        text = stage_status.cnn_key_text(text, data)
        if name == "harmony":
            import jev_reads
            text = jev_reads.refresh_contrast(text, data)
        cards[name] = dict(card, text=text, json=data)
        headers_changed |= data is not card["json"]
        changed |= data is not card["json"] or text != card["text"]
    if changed:
        song = dict(song, cards=cards)
    if view == "melody":
        inputs = (song["cards"]["singer"]["json"] or {}).get("melody_input") or {}
        if inputs.get("f0"):
            song = dict(song, melody=listen_melody.build(song, full=True))
        return quick_note + doctored_audio.annotate(listen_melody.render(song), song.get("audio_quality")), "melody"
    if view not in PASSES:
        melody_view = view not in ("full", "detail")
        if melody_view and headers_changed:
            song = dict(song, melody=listen_melody.build(song))
        elif listen_melody.needs_pitch_refresh(song) or (melody_view and listen_melody.needs_lyric_refresh(song)):
            melody = listen_melody.build(song)
            if melody["status"] == "ran":
                song = dict(song, melody=melody)
    if view in ("full", "detail"):
        import feeling_card
        full = _full_text(song, detail=view == "detail")
        full += feeling_card.retained_details(summary_text(song), full)
        return quick_note + doctored_audio.annotate(full, song.get("audio_quality")), view
    if view in PASSES:
        return quick_note + doctored_audio.annotate(pass_text(song, view), song.get("audio_quality")), view
    text = summary_text(song, notices)
    if view == "first-listen":
        import feeling_card
        text = feeling_card.render(text, {n: pass_text(song, n) for n in PASSES},
                                   {n: song["cards"][n]["json"] for n in PASSES},
                                   song["manifest"]["passes"], listen_melody.render(song, hook=True),
                                   (song.get("melody") or {}).get("json"),
                                   listen_melody.title_words(options))
    return quick_note + doctored_audio.annotate(text, song.get("audio_quality")), view


def summary_text(song, notices=()):
    """The pre-feeling-card first listen, including its saved hook."""
    import listen_melody
    import slim
    singer = song["cards"]["singer"]
    stops_result = (singer["json"] or {}).get("stops")
    texts = {name: song["cards"][name]["text"] for name in PASSES}
    if song["manifest"]["passes"]["singer"]["status"] == "ran" and isinstance(stops_result, dict):
        import stops
        texts["singer"] = stops.saved_text(texts["singer"], stops_result)
    text = slim.render_first_listen(
        texts,
        {name: song["cards"][name]["json"] for name in PASSES},
        song["manifest"]["passes"])
    # Keep ABC rows intact: wrapping them changes the notation/lyric pairing.
    melody = listen_melody.render(song, hook=True).rstrip()
    lyric_header = re.search(r"(?m)^LYRICS:", text)
    start = lyric_header.start() if lyric_header else text.rindex(slim.INVITATION)
    text = text[:start] + melody + "\n" + text[start:]
    provenance = [line for line in notices if line.startswith(("Saved card", "Missing passes"))]
    if provenance:
        from stage_status import NOTES_WIDTH
        lines = [part for line in provenance for part in textwrap.wrap(line, width=NOTES_WIDTH)]
        text = text.removesuffix(slim.INVITATION + "\n") + "\n".join(lines) + "\n" + slim.INVITATION + "\n"
    return text


def execute(args, *, on_relisten=None, progress_started=None):
    from network_policy import local_audio_path
    if args.audio:
        local_audio_path(args.audio)
    def progress(label):
        if progress_started is not None:
            print_progress(progress_started, label, sys.stderr)

    import listening_sheet
    picture, picture_hint = listening_sheet.prepare_picture(
        args, text_card=not (args.json or args.history or args.delete or args.delete_all),
        error="--picture needs a song text card; omit JSON, history and deletion controls")
    args = argparse.Namespace(**dict(vars(args), picture=picture))
    if args.lore and (args.fresh or args.run_missing or args.only or args.delete or args.delete_all
                      or args.json or args.history):
        raise ValueError("--lore only reads an existing card; omit other controls")
    if args.delete_all:
        if args.audio or args.delete or args.fresh or args.run_missing:
            raise ValueError("--delete-all takes no song or listen controls")
    elif not args.audio:
        raise ValueError("supply an audio file")
    if args.delete and (args.fresh or args.run_missing):
        raise ValueError("--delete cannot also run passes")
    if args.fresh and args.only:
        raise ValueError("--fresh runs every pass; omit --only")
    if args.history and (args.fresh or args.run_missing or args.only or args.delete or args.delete_all or args.json):
        raise ValueError("--history only reads an existing bundle; omit other controls")
    if os.environ.get("RAVEN_EARS_NO_BASELINE") == "1" and args.keep_stems:
        raise ValueError("--keep-stems is unavailable in a disposable no-baseline home")
    home = store_home()
    if home.is_symlink():
        raise ValueError("refusing symlink song store")
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (home / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.delete_all:
            deleted, refused = 0, 0
            for directory in home.iterdir():
                if directory.name == ".lock":
                    continue
                try:
                    if delete_bundle(home, directory / "song.json"):
                        deleted += 1
                    else:
                        refused += 1
                except (OSError, ValueError):
                    refused += 1
            return (f"Deleted {deleted} validated saved song bundles (other files and engine caches retained).\n"
                    f"Unrelated or unsafe entries left intact: {refused}.\n"), []
        audio_hash = file_hash(Path(args.audio).expanduser())
        path = home / audio_hash / "song.json"
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("refusing symlink song bundle")
        if args.lore and not path.exists():
            return "listen first; the lore comes after\n", []
        if args.delete:
            deleted = delete_bundle(home, path)
            return ("Deleted saved song (other engine caches retained).\n" if deleted else
                    "No saved bundle found (unrelated files retained).\n"), []
        if args.history:
            if not path.exists():
                raise ValueError("no saved bundle for this song")
            return history_text(load(path)), []
        current_options, version = options(args), ears_version()
        original_inputs = input_hashes(args)
        if original_inputs["audio"] != audio_hash or any(
                original_inputs.get(name) != current_options[name + "_sha256"]
                for name in ("lyrics", "names", "credits") if getattr(args, name)):
            raise ValueError("inputs changed before listening; try again with stable files")
        previous = None
        if path.exists():
            try:
                previous = load(path)
            except ValueError:
                if not args.fresh:
                    raise
        refresh = args.fresh
        song = None if refresh else previous
        gone = [name for name in ("lyrics", "names", "credits")
                if getattr(args, name) and not Path(getattr(args, name)).expanduser().exists()]
        if gone and not song:
            raise ValueError("supplied file not found: --" + ", --".join(gone))
        notices, changed = [], []
        if song:
            manifest = song["manifest"]
            if manifest["file_sha256"] != audio_hash:
                raise ValueError("saved file hash does not match its directory; use --fresh")
            missing = ["--" + name for name in gone]
            overrides = [flag for flag in changed_analysis_flags(args, current_options, manifest["options"])
                         if flag not in missing]
            if overrides:
                difference = ", ".join(overrides) + (" differs" if len(overrides) == 1 else " differ")
                if missing or args.only or args.lore:
                    action = ("supply the missing " + ", ".join(missing) + " file(s) and use --fresh"
                              if missing else "use --fresh without " + ("--only" if args.only else "--lore"))
                    raise ValueError(difference + " from the saved card; " + action)
                notice = "re-listening: " + difference + " from the saved card"
                if on_relisten is None:
                    notices.append(notice)
                else:
                    on_relisten(notice)
                refresh = True
                song = None
            elif missing:
                notices.append("Saved card: supplied file not found (" + ", ".join(missing)
                               + "); using saved inputs. Supply the file and use --fresh to refresh.")
        if song:
            # An implicit new default must not reinterpret an old saved listen,
            # including any passes retried with --run-missing.
            if args.key_changes is None and not args.quick:
                current_options["key_changes"] = manifest["options"].get("key_changes", False)
                args = argparse.Namespace(**dict(vars(args), key_changes=current_options["key_changes"]))
            if manifest["ears_version"] != version:
                notices.append("Saved card version differs from this checkout; use --fresh to refresh.")
            changed = [key for key in current_options if current_options[key] != manifest["options"].get(
                key, False if key == 'key_changes' else None)]
            if manifest["options"].get("quick") and not args.quick:
                changed.append("quick")
            if changed:
                notices.append("Saved card options differ: " + ", ".join(changed) + "; use --fresh to refresh.")
            date = datetime.fromisoformat(manifest["heard_at"])
            notices.append(f"from the saved card, heard {date.day} {date:%b}")
            pending = [name for name in PASSES if manifest["passes"][name]["status"] != "ran"]
            needed = PASSES if args.view in ("full", "detail", "first-listen", "summary") or args.json else (args.view,)
            if args.view == "melody":
                needed = ("singer", "music")
            todo = [name for name in pending if name in needed] if args.run_missing else []
            if todo and (gone or changed or manifest["ears_version"] != version):
                notices.append("Missing passes left unchanged; --fresh is required before running with different inputs or version.")
                todo = []
        else:
            now = datetime.now(timezone.utc).isoformat()
            song = {"schema": 1, "manifest": {"file_sha256": audio_hash, "duration_s": None,
                "duration_reason": "no pass supplied a duration", "ears_version": version,
                "options": current_options, "heard_at": now, "passes": {}}, "cards": {},
                "melody": {"status": "not run", "reason": "not reached", "text": "", "json": None}}
            for name in PASSES:
                reason = "not selected" if args.only and name != args.only else "not reached"
                song["manifest"]["passes"][name] = {"status": "not run", "reason": reason, "date": None,
                    "first_listened": None, "last_listened": None, "listen_count": 0, "forwarded_options": {}}
                song["cards"][name] = {"text": "", "json": None, "diagnostics": ""}
            todo = [args.only] if args.only else list(PASSES)
            progress("checking the audio")
            import song_extras
            lookup_args = argparse.Namespace(**vars(args))
            extras = song_extras.prepare(lookup_args, args.audio)
            if extras:
                song["song_extras"] = extras
            import doctored_audio
            song["audio_quality"] = doctored_audio.inspect(Path(args.audio).expanduser())
            if progress_started is None:
                notices.append("Listening now; saving the full song card.")
            if not refresh:
                save(path, song)
        for name in todo:
            before_inputs = None
            attempted = False
            try:
                before_inputs = input_hashes(args, strict=True)
                attempted = True
                if song.get("song_extras", {}).get("identification"):
                    identity = song["song_extras"]["identification"]
                    run_args = argparse.Namespace(**vars(args))
                    run_args.artist, run_args.title = identity["artist"], identity["title"]
                else:
                    run_args = args
                if progress_started is not None:
                    run_args = argparse.Namespace(**vars(run_args), _progress_started=progress_started)
                progress(name + " pass")
                result = run_pass(name, run_args)
            except (OSError, ValueError) as error:
                result = {"status": "failed", "reason": f"{type(error).__name__}: {error}",
                          "text": "", "json": None, "diagnostics": ""}
            try:
                after_inputs = input_hashes(args, strict=True)
                changed_inputs = [key for key in original_inputs if
                                  (before_inputs is not None and before_inputs.get(key) != original_inputs[key])
                                  or after_inputs.get(key) != original_inputs[key]]
                input_error = "inputs changed during listen: " + ", ".join(changed_inputs) if changed_inputs else ""
            except OSError:
                input_error = "inputs could not be re-hashed after pass"
            if input_error:
                result["status"] = "failed"
                result["reason"] = "; ".join(filter(None, (result["reason"], input_error)))
            old = (previous or song)["manifest"]["passes"][name]
            now = datetime.now(timezone.utc).isoformat()
            count = old.get("listen_count", 0 if old["status"] == "not run" else None)
            song["manifest"]["passes"][name] = {"status": result.pop("status"),
                "reason": result.pop("reason"), "date": now,
                "first_listened": now if attempted and count == 0 else old.get("first_listened"),
                "last_listened": now if attempted else old.get("last_listened", old.get("date")),
                "listen_count": count + int(attempted) if count is not None else None,
                "forwarded_options": forwarded_options(name, run_args, current_options) if attempted else {}}
            song["cards"][name] = result
            data = result["json"] or {}
            duration = data.get("duration_s", data.get("duration"))
            if (song["manifest"]["passes"][name]["status"] == "ran"
                    and isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0):
                song["manifest"]["duration_s"] = duration
                song["manifest"]["duration_reason"] = ""
            import song_length
            song_length.update_song(song)
            # New listens checkpoint outcomes. Refresh publishes only when every
            # replacement pass has finished. Only listening counters advance on
            # the previous bundle; its complete cards/statuses stay readable.
            if refresh and previous:
                for key in ("first_listened", "last_listened", "listen_count"):
                    old[key] = song["manifest"]["passes"][name][key]
                save(path, previous)
            if not refresh:
                save(path, song)
        if not args.lore and "melody" in song and (set(todo) & {"singer", "music"}
                or song["melody"]["reason"] == "not reached"
                or (args.run_missing and not changed and song["manifest"]["ears_version"] == version
                    and song["melody"]["status"] != "ran"
                    and args.view in ("melody", "first-listen", "summary", "full", "detail"))):
            # A derived part of the captured singer/music passes, stored apart
            # from their original cards. Saved views never call an audio engine.
            progress("writing melody notation")
            import listen_melody
            song["melody"] = listen_melody.build(song)
            save(path, song)
        if refresh:
            save(path, song)
        if todo and "from the saved card" in " ".join(notices):
            notices.append("Ran missing passes: " + ", ".join(todo) + "; other cards remain saved.")
        if args.keep_stems and "singer" not in todo:
            notices.append("--keep-stems only copies stems when singer runs; use --fresh to run it again.")
        if args.json:
            return json.dumps(saved_card_readings(song), ensure_ascii=False, indent=2) + "\n", notices
        text, used = render(song, args.view, notices)
        if args.picture:
            progress("drawing the picture")
            import listening_sheet
            decided = text if used == "first-listen" else render(song, "first-listen")[0]
            picture = listening_sheet.attempt(
                listening_sheet.song_sheet, Path(args.audio).expanduser(), song,
                decided, full_text(song), path.with_name("listening-sheet.png"), by_ear=args.by_ear)
            if used == "first-listen":
                import feeling_card
                picture = feeling_card.picture_line(picture)
            text = text.rstrip("\n") + "\n" + picture + "\n"
        elif picture_hint:
            text += picture_hint + "\n"
        notices.append("view: " + used)
        # A compact view must not hide a failed or deliberately omitted pass.
        if args.view in ("first-listen", "summary"):
            notices += [f"{name}: {row['status']}: {row['reason']}" for name, row in
                        song["manifest"]["passes"].items() if row["status"] != "ran"]
        return text, notices


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--_worker":
        if argv[3:4] == ["--_progress"]:
            import stage_status
            with os.fdopen(int(argv[4]), "w") as stream:
                started = float(argv[5])
                with stage_status.progress_to(lambda label: print_progress(started, label, stream)):
                    return worker(argv[1], argv[2], argv[6:])
        return worker(argv[1], argv[2], argv[3:])
    started = time.monotonic() if sys.stderr.isatty() else None
    ap = parser()
    args = ap.parse_args(argv)
    try:
        with home_context():
            text, notices = execute(args, progress_started=started,
                                    on_relisten=lambda line: print(line, file=sys.stderr, flush=True))
            for notice in notices:
                print(notice, file=sys.stderr)
            sys.stdout.write(text)
            if args.lore and text != "listen first; the lore comes after\n":
                # The reader gets the feeling before any lore request starts.
                sys.stdout.flush()
                import song_extras
                path = store_home() / file_hash(Path(args.audio).expanduser()) / "song.json"
                sys.stdout.write(song_extras.lore(load(path), args.audio))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"listen: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
