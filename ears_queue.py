#!/usr/bin/env python3
"""One song at a time, with a durable queue and honest failures.

    ears_queue.py add "artist - song" /path/to/song.mp3 songs.txt
    ears_queue.py list
    ears_queue.py remove ID
    ears_queue.py run --max 4 --until 07:00 --reuse-stages
    ears_queue.py cron

RAVEN_QUEUE_DIR moves state (default ~/.local/state/raven-ears/queue).
RAVEN_CARDS_DIR moves cards (default ~/.cache/raven/cards). Tests use both.
No audio models are imported here. Engines and raven-song are subprocesses.
"""
from __future__ import annotations

from network_policy import offline, local_audio_path

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

import stage_status as ST

HERE = Path(__file__).resolve().parent
ENGINES = ("music", "singer", "harmony")
HEADERS = {"music": "MUSIC:", "singer": "SINGER.PY", "harmony": "HARMONY:"}
INVITATION = "and now, you"
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".aiff"}
# Only diagnostic-shaped lines, never an ordinary lyric containing "failed".
BACKEND_PROBLEM = re.compile(
    r"^\s*(?:\[[\w -]+\]|NOTE:|[A-Z][A-Z ._+/-]*:|"
    r"(?:vocals|vocal quality|lyrics|words|key fusion|melody|bass stem pitch|meter claim|key claim|"
    r"Whisper|breath|credits|refrain check|word / music contrast|pace / WORDLESS / emphasis)\b)"
    r".*\b(?:failed|unavailable|unreachable|timed out|could not|not importable|skipped)\b", re.MULTILINE)

# Exact eligibility decisions, not a general exemption for skipped backends.
INTENTIONAL_SKIPS = {
    ("Whisper", "word times supplied"),
    ("VS USUAL", "no speaker given"),
    ("breath", "no breath-rate evidence"),
    ("breath", "the event model does not hear breathing here"),
    ("credits (Discogs)", "not needed"),
    ("credits (Discogs)", "no DISCOGS_TOKEN set; its search needs one"),
    ("Whisper", "LRCLIB instrumental designation"),
    ("words", "LRCLIB instrumental designation"),
}
INSTRUMENTAL_SKIPS = {
    ("Whisper", "--no-whisper"),
    ("words", "--no-Whisper, no word times given"),
    ("pace / WORDLESS / emphasis", "no word times"),
    ("word / music contrast", "no lyric lines"),
    ("refrain check", "no heard words"),
}


class QueueError(Exception):
    pass


class Interrupted(QueueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        # Make the rename durable as well as the file contents.
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@contextmanager
def locked(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink a flock file: another process may already have it open.
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise QueueError("another queue run is still listening (run.lock is held)") from None
        yield f


class Queue:
    def __init__(self, state_dir=None, cards_dir=None, ears_dir=None, python=None):
        state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
        self.state = Path(state_dir or os.environ.get("RAVEN_QUEUE_DIR", state_home / "raven-ears/queue")).expanduser().resolve()
        self.cards = Path(cards_dir or os.environ.get("RAVEN_CARDS_DIR", Path.home() / ".cache/raven/cards")).expanduser().resolve()
        self.ears = Path(ears_dir or HERE).resolve()
        self.python = str(python or sys.executable)

    def items(self):
        value = read_json(self.state / "queue.json", [])
        if not isinstance(value, list) or any(
                not isinstance(x, dict) or any(not isinstance(x.get(k), str) for k in ("id", "target", "status"))
                or not isinstance(x.get("attempts", 0), int) or x.get("attempts", 0) < 0 for x in value):
            raise QueueError("queue.json is malformed; refusing to replace it")
        return value

    def update(self, item_id, **fields):
        with locked(self.state / "state.lock"):
            items = self.items()
            for item in items:
                if item["id"] == item_id:
                    item.update(fields)
                    write_json(self.state / "queue.json", items)
                    return item
        raise QueueError(f"queue item disappeared: {item_id}")

    def add(self, targets):
        targets = expand_targets(targets)
        with locked(self.state / "state.lock"):
            items = self.items()
            for target in targets:
                if any(x["target"] == target for x in items):
                    continue
                items.append({"id": digest(target.encode())[:16], "target": target,
                              "status": "pending", "added_at": time.time(), "attempts": 0})
            write_json(self.state / "queue.json", items)
        return items

    def remove(self, item_id):
        # Do not remove an active item or let removal race a runner's snapshot.
        with locked(self.state / "run.lock", blocking=False):
            with locked(self.state / "state.lock"):
                items = self.items()
                matches = [x for x in items if x["id"].startswith(item_id)]
                if len(matches) != 1:
                    raise QueueError("give one unambiguous ID from list")
                write_json(self.state / "queue.json", [x for x in items if x != matches[0]])
                return matches[0]

    def log(self, **event):
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / "listens.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(at=time.time(), **event), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def version(self):
        """Hash actual source bytes, including uncommitted edits, and run settings.

        Conservative: every engine's checkpoint uses the same dependency stamp.
        Only the digest is stored; names, environment values and tokens are not.
        """
        sources = sorted(p for p in self.ears.glob("*.py") if not p.name.startswith("test_"))
        sources += sorted(self.ears.glob("requirements*.txt"))
        sources += [p for p in (self.ears / "pyproject.toml",) if p.exists()]
        if not all((self.ears / (e + ".py")).is_file() for e in ENGINES):
            raise QueueError(f"engine scripts missing in {self.ears}")
        h = hashlib.sha256()
        newest = 0
        for p in sources:
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
            newest = max(newest, p.stat().st_mtime_ns)
        home = Path(os.environ.get("RAVEN_EARS_HOME", Path.home() / ".local/share/raven-ears"))
        names = home / "names.txt"
        h.update(names.read_bytes() if names.is_file() else b"no names file")
        settings = {k: v for k, v in os.environ.items()
                    if (k.startswith("RAVEN_") and k not in {"RAVEN_QUEUE_DIR", "RAVEN_CARDS_DIR"})
                    or k in {"DISCOGS_TOKEN", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"}}
        h.update(json.dumps(settings, sort_keys=True).encode())
        h.update(str(self.ears).encode())
        h.update(self.python.encode())
        h.update(str(Path(self.python).stat().st_mtime_ns).encode())
        return {"sha256": h.hexdigest(), "newest_source_ns": newest}

    def command(self, args, lock):
        """Children inherit the run lock, including if the queue is SIGKILLed.

        For normal cancellation, terminate the whole child process group and
        wait before releasing the lock. No second GPU song can start early.
        """
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as out:
            child = subprocess.Popen(args, cwd=self.ears, stdout=out, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True, pass_fds=(lock.fileno(),))
            try:
                code = child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The engine may have exited just before the signal.
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            out.seek(0)
            return code, out.read()

    def resolve(self, item, lock):
        target = item["target"]
        if Path(target).is_file():
            return Path(target).resolve()
        if looks_like_path(target):
            raise QueueError(f"audio file is missing: {target}")
        # A remembered fetch lets a crash resume without another download.
        if item.get("audio") and Path(item["audio"]).is_file():
            return Path(item["audio"])
        if offline():
            raise QueueError("song resolution disabled by RAVEN_EARS_OFFLINE=1; supply a local audio path")
        command = os.environ.get("RAVEN_SONG_COMMAND", "raven-song")
        code, output = self.command([command, target], lock)
        lines = output.strip().splitlines()
        if code or not lines or not Path(lines[-1]).is_file():
            raise QueueError(f"raven-song exit {code}: {output.strip()[-2000:] or 'no audio path returned'}")
        return Path(lines[-1]).resolve()

    def fingerprint(self, audio):
        st = audio.stat()
        return {"path": str(audio), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                "sha256": file_digest(audio)}

    def fresh(self, card, stamp):
        try:
            meta = read_json(card.with_suffix(".queue.json"), {})
            return (meta.get("stamp") == stamp and meta.get("complete") is True
                    and card.stat().st_mtime_ns >= stamp["version"]["newest_source_ns"]
                    and meta.get("card_sha256") == file_digest(card))
        except (OSError, ValueError, TypeError, AttributeError):
            return False

    def card_for(self, audio):
        card = self.cards / (audio.stem + ".txt")
        try:
            meta = read_json(card.with_suffix(".queue.json"), {})
            owner = meta.get("stamp", {}).get("audio", {}).get("path")
        except (OSError, ValueError, TypeError, AttributeError):
            owner = None
        if owner and owner != str(audio):
            # raven-listen uses basenames. Keep that convention unless two
            # different source paths would overwrite each other's cards.
            return self.cards / (audio.stem + "-" + digest(str(audio).encode())[:12] + ".txt")
        return card

    def listen(self, item, audio, stamp, lock, reuse_stages=False):
        parts, stages, failures = [], [], []
        key = digest(json.dumps(stamp, sort_keys=True).encode())
        for engine in ENGINES:
            saved = self.state / "stages" / key / (engine + ".json")
            result = None
            if reuse_stages:
                try:
                    candidate = read_json(saved)
                    if (candidate and candidate.get("ok") and candidate.get("stamp") == stamp
                            and candidate.get("output_sha256") == digest(candidate.get("output", "").encode())
                            and stage_problem(engine, candidate.get("output", ""), 0,
                                              candidate.get("stage_status")) is None):
                        result = dict(candidate, reason=None)
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
            start = time.monotonic()
            reused = result is not None
            if not reused:
                try:
                    code, output = self.command([self.python, str(self.ears / (engine + ".py")), str(audio)], lock)
                except OSError as e:
                    code, output = -1, f"{type(e).__name__}: {e}"
                reason = stage_problem(engine, output, code)
                result = {"ok": reason is None, "output": output, "reason": reason, "stamp": stamp,
                          "output_sha256": digest(output.encode())}
                # Failed/degraded stages are never reused. Successful siblings can be.
                if reuse_stages and result["ok"]:
                    write_json(saved, result)
            elapsed = round(time.monotonic() - start, 3)
            stages.append({"engine": engine, "duration_s": elapsed, "reused": reused,
                           "ok": result["ok"], "reason": result["reason"]})
            parts.append(f"\n=================== {engine}\n")
            if reused:
                parts.append("QUEUE: reused earlier engine output for the same code, settings and audio.\n")
            parts.append(result["output"].rstrip() + "\n")
            if not result["ok"]:
                failures.append(f"{engine}: {result['reason']}")
                parts.append(f"QUEUE: analysis incomplete: {result['reason']}; missing detail is not silence.\n")
        return parts, stages, failures

    def run(self, maximum=None, until=None, reuse_stages=False, refresh=False):
        attempted, failed = 0, 0
        with locked(self.state / "run.lock", blocking=False) as lock:
            # Snapshot: additions during a run wait for the next run. Every item,
            # including done items, is checked again for changed source/audio.
            for item in self.items():
                if maximum is not None and attempted >= maximum:
                    break
                if until is not None and datetime.now() >= until:
                    print("Cutoff reached; remaining songs stay queued.")
                    break
                if item["status"] == "running":
                    self.log(id=item["id"], target=item["target"], status="interrupted", duration_s=None,
                             reason="previous runner ended before recording completion; retrying")
                start = time.monotonic()
                attempted += 1
                self.update(item["id"], status="running", started_at=time.time(),
                            attempts=item.get("attempts", 0) + 1)
                self.log(id=item["id"], target=item["target"], status="started")
                stages = []
                card = None
                try:
                    audio = self.resolve(item, lock)
                    self.update(item["id"], audio=str(audio))
                    stamp = {"version": self.version(), "audio": self.fingerprint(audio)}
                    card = self.card_for(audio)
                    if not refresh and self.fresh(card, stamp):
                        status, reason = "skipped", "card matches code, settings and audio stamp"
                        attempted -= 1  # --max bounds listens, not cache hits.
                    else:
                        parts, stages, failures = self.listen(item, audio, stamp, lock, reuse_stages and not refresh)
                        if stamp != {"version": self.version(), "audio": self.fingerprint(audio)}:
                            failures.append("code, settings or audio changed during listening; retry with stable inputs")
                        text = f"# {audio.stem}  ({datetime.now():%Y-%m-%d %H:%M})\n" + "".join(parts)
                        if failures:
                            text += "\nQUEUE: incomplete listen; " + "; ".join(failures) + "\n"
                        text += "\n" + INVITATION + "\n"
                        status, reason = ("failed", "; ".join(failures)) if failures else ("success", "all three engines completed")
                        if failures:
                            # A failed retry cannot destroy an earlier complete card.
                            card = card.with_suffix(".failed.txt")
                            atomic_text(card, text)
                        else:
                            atomic_text(card, text)
                            write_json(card.with_suffix(".queue.json"),
                                       {"complete": True, "stamp": stamp, "card_sha256": digest(text.encode())})
                except (KeyboardInterrupt, Interrupted) as e:
                    reason = str(e) or "listening interrupted by keyboard"
                    self.finish(item, "interrupted", reason, start, stages, card)
                    raise Interrupted(reason) from None
                except (OSError, ValueError, QueueError) as e:
                    status, reason = "failed", f"{type(e).__name__}: {e}"
                self.finish(item, status, reason, start, stages, card)
                if status == "failed":
                    failed += 1
        return 1 if failed else 0

    def finish(self, item, status, reason, start, stages, card):
        duration = round(time.monotonic() - start, 3)
        self.update(item["id"], status="done" if status in {"success", "skipped"} else status,
                    last_status=status, reason=reason, duration_s=duration, card=str(card) if card else None)
        self.log(id=item["id"], target=item["target"], status=status, reason=reason,
                 duration_s=duration, stages=stages, card=str(card) if card else None)
        print(f"{item['id']} {status} ({duration:.3f}s): {item['target']} — {reason}", flush=True)


def stage_problem(engine, output, code, stage_status=None):
    if code:
        return f"exit {code}: {output.strip()[-1500:] or 'no diagnostic output'}"
    if not any(line.startswith(HEADERS[engine]) for line in output.splitlines()):
        return f"no {HEADERS[engine]} card returned: {output.strip()[-1500:] or 'empty output'}"
    problems = []
    if stage_status is not None:
        if not isinstance(stage_status, list) or any(
                not isinstance(r, dict) or not isinstance(r.get("stage"), str)
                or r.get("status") not in ST.STATES for r in stage_status):
            return "invalid structured stage status"
        # A saved caller may have the original rows; never infer their outcome
        # from words such as "failed" inside a successful stage's reason.
        instrumental = any(r.get("stage") == "words" and r.get("status") == "skipped"
                           and r.get("reason") == "LRCLIB instrumental designation" for r in stage_status)
        for row in stage_status:
            label = row["stage"]
            if row.get("source") and row["source"] != label:
                label += " (" + row["source"] + ")"
            pair = (label, row.get("reason", ""))
            # Degraded work finished and says so on the card (e.g. the CPU profile's
            # htdemucs stems); listening again would only degrade the same way.
            if row["status"] in {"ran", "not-found", "degraded"}:
                continue
            if row["status"] == "skipped" and (pair in INTENTIONAL_SKIPS or
                                                instrumental and pair in INSTRUMENTAL_SKIPS):
                continue
            problems.append(ST.describe(row))
    # Engine subprocesses return text. STATUS/NOTES retain the recorded state;
    # older diagnostics still need the conservative fallback below. Keep each
    # pass separate so an instrumental skip cannot excuse another pass's work.
    for block in re.split(r"(?m)^and now, you\s*$", output):
        notes = ST.card_notes(block)
        instrumental = any(n in {"words: skipped: LRCLIB instrumental designation",
                                 "Whisper: skipped: LRCLIB instrumental designation"} for n in notes)
        for note in notes:
            state = re.fullmatch(r"(.+?): (failed|degraded|skipped)(?:: (.*))?", note)
            if state:
                label, status, reason = state.groups()
                if status == "degraded":
                    continue
                pair = (label, reason or "")
                if status == "skipped" and (pair in INTENTIONAL_SKIPS or
                                            instrumental and pair in INSTRUMENTAL_SKIPS):
                    continue
                problems.append(note)
            elif "status entries omitted by source" in note:
                problems.append(note)
            elif BACKEND_PROBLEM.match("NOTES: " + note):
                problems.append(note)
        legacy = []
        for line in block.splitlines():
            if line.strip().startswith(("STATUS:", "NOTES:")):
                continue
            diagnostic = re.sub(r"^(?:NOTE:|\[[\w -]+\])\s*", "", line.strip())
            skip = re.fullmatch(r"(.+?): skipped: (.*)", diagnostic)
            if skip and ((skip[1], skip[2]) in INTENTIONAL_SKIPS or
                         instrumental and (skip[1], skip[2]) in INSTRUMENTAL_SKIPS):
                continue
            # These exact legacy credit and word-availability explanations also
            # occur in the supplied completed instrumental card.
            line = line.replace("; Discogs skipped (no DISCOGS_TOKEN set; its search needs one); ears only",
                                "; ears only")
            if instrumental:
                line = line.replace("voice present, words unavailable (transcription did not run or failed)",
                                    "voice present, words not transcribed (LRCLIB instrumental designation)")
            match = BACKEND_PROBLEM.match(line)
            if match:
                legacy.append(match[0])
        problems.extend(legacy)
    if problems:
        return "backend reported missing/failed analysis: " + "; ".join(problems)[:2000]
    if "Traceback (most recent call last):" in output:
        return "backend traceback despite exit 0: " + output.strip()[-1500:]
    return None


def looks_like_path(value):
    if value.startswith(("http://", "https://")):
        return False
    return value.startswith(("/", "./", "../", "~")) or Path(value).suffix.lower() in AUDIO_SUFFIXES


def expand_targets(targets):
    out = []

    def add(value, base):
        value = value.strip()
        local_audio_path(value)
        if not value or value.startswith("-") or "\n" in value or "\r" in value:
            raise QueueError("a song must be nonempty text or a path, never an option")
        p = Path(value).expanduser()
        p = p if p.is_absolute() else base / p
        if p.is_file() or looks_like_path(value):
            if not p.is_file():
                raise QueueError(f"no such audio file: {p}")
            value = str(p.resolve())
        out.append(value)

    for target in targets:
        local_audio_path(target)
        path = Path(target).expanduser()
        if path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip() and not line.lstrip().startswith("#"):
                    add(line, path.resolve().parent)
        else:
            add(target, Path.cwd())
    return out


def cutoff(value, now=None):
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise argparse.ArgumentTypeError("use local time HH:MM (00:00–23:59)")
    hour, minute = map(int, value.split(":"))
    return (now or datetime.now()).replace(hour=hour, minute=minute, second=0, microsecond=0)


def cron_line(queue):
    # Explicit interpreter, worktree and PATH: cron does not inherit an interactive shell.
    args = [queue.python, str(HERE / "ears_queue.py"), "run"]
    env = {"PATH": str(Path.home() / ".local/bin") + ":/usr/local/bin:/usr/bin:/bin",
           "RAVEN_QUEUE_DIR": str(queue.state), "RAVEN_CARDS_DIR": str(queue.cards)}
    command = "env " + " ".join(k + "=" + shlex.quote(v) for k, v in env.items()) + " " + shlex.join(args)
    command += " >> " + shlex.quote(str(queue.state / "cron.log")) + " 2>&1"
    # mkdir is needed before the shell opens the redirect on the first run.
    command = "mkdir -p " + shlex.quote(str(queue.state)) + " && " + command
    return "45 3 * * * " + command.replace("%", r"\%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    add = sub.add_parser("add", help="song query, audio path, or a .txt list (one song per line)")
    add.add_argument("targets", nargs="+")
    sub.add_parser("list", help="show IDs, status and last honest reason")
    remove = sub.add_parser("remove", help="remove an ID or unique ID prefix; keeps cards")
    remove.add_argument("id")
    run = sub.add_parser("run", help="one GPU song at a time; retry failures on the next run")
    run.add_argument("--max", type=int, dest="maximum", help="at most N attempted listens, excluding cache hits")
    run.add_argument("--until", type=cutoff, help="today's local HH:MM; finish current song, then stop")
    run.add_argument("--reuse-stages", action="store_true", help="reuse successful engine checkpoints with identical inputs")
    run.add_argument("--refresh", action="store_true", help="bypass card and stage reuse (e.g. after model/credits changes)")
    sub.add_parser("cron", help="print the exact 03:45 crontab line; install nothing")
    a = ap.parse_args(argv)
    if a.action == "run" and a.maximum is not None and a.maximum < 0:
        ap.error("--max must be zero or greater")
    queue = Queue()
    if a.action == "add":
        items = queue.add(a.targets)
        print(f"{len(items)} songs in the queue.")
    elif a.action == "list":
        for item in queue.items():
            print(f"{item['id']}  {item['status']:11}  {item['target']}")
            if item.get("reason"):
                print("  " + item["reason"])
    elif a.action == "remove":
        print("Removed " + queue.remove(a.id)["target"])
    elif a.action == "cron":
        print(cron_line(queue))
    else:
        return queue.run(a.maximum, a.until, a.reuse_stages, a.refresh)
    return 0


def cli(argv=None):
    def interrupted(signum, frame):
        raise Interrupted(f"listening interrupted by signal {signum}")

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        return main(argv)
    except Interrupted as e:
        print(f"ears_queue: {e}; unfinished songs remain queued", file=sys.stderr)
        return 130
    except (OSError, ValueError, QueueError) as e:
        print(f"ears_queue: {e}", file=sys.stderr)
        return 1
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        print(INVITATION, flush=True)


if __name__ == "__main__":
    sys.exit(cli())
