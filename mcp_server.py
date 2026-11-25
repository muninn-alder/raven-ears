"""raven-ears as an MCP server: the ears as tools, and the listening skill that comes with them.

    python mcp_server.py                # stdio, for `claude mcp add`

Tools (thin wrappers; each runs an existing CLI as a subprocess with an explicit argv, no shell):

    hear_song(path, artist?, title?, view?)   listen.py; background when the song is new
    song_card(path, view?)                    a SAVED card, never starts a listen
    hear_voice(path, speaker?, verbatim?)     raven_ears_plus.py on a voice note

The listening skill (skills/listening/SKILL.md) is served through Skills over MCP
(io.modelcontextprotocol/skills: skills/list, skills/get, skill:// resources), and also as a
resource-free prompt called "listening" for clients that do not speak the extension yet.

A cold song listen takes minutes. The Python SDK has no server-side Tasks runtime, so hear_song
uses the fallback: it starts listen.py in its own process group and returns at once, and
song_card says "still listening" until the saved card is complete.

Network policy: nothing here goes online. The CLIs it runs inherit the environment, so
RAVEN_EARS_OFFLINE=1 reaches every lookup they make. Inputs must be local files.
"""
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
SKILL_NAME = "listening"
SKILL_FILE = HERE / "skills" / SKILL_NAME / "SKILL.md"
SKILL_URI = f"skill://{SKILL_NAME}/SKILL.md"
SKILLS_EXTENSION = "io.modelcontextprotocol/skills"

# What the caller says, and the listen.py view it means.
SONG_VIEWS = {"feel": "first-listen", "full": "full", "melody": "melody", "summary": "summary"}
SAVED_TIMEOUT_S = 180       # a saved card is read from disk (the sheet may be redrawn)
VOICE_TIMEOUT_S = 900
START_GRACE_S = 30          # a just-started listen may not hold the lock yet
LOG_TAIL = 1500

INSTRUCTIONS = (
    "raven-ears turns songs and voice notes into text cards. Read the 'listening' skill "
    f"({SKILL_URI}, or the 'listening' prompt) before you listen: it is about how to hear well and what to "
    "keep. In its command examples, `listen.py song.mp3` is hear_song, `--view` is the view argument, "
    "and `raven_ears_plus.py note.ogg --speaker name` is hear_voice. A new song takes 2 to 20 minutes: "
    "hear_song returns at once and the card arrives through song_card. A voice note is someone's "
    "own voice: ask first, and keep the card, not the audio."
)

_PROCS = {}  # audio hash -> Popen for listens started by this server process


# ---------------------------------------------------------------- paths and argv

def ears_python():
    """The interpreter that has the ears' dependencies (numpy, torch, ...)."""
    chosen = os.environ.get("RAVEN_EARS_PYTHON")
    if chosen:
        return chosen
    venv = HERE / "venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def local_audio(path):
    """Absolute path of an existing local file; URLs and protocol inputs are refused."""
    from network_policy import local_audio_path
    if not isinstance(path, str) or not path.strip():
        raise ValueError("supply the path of a local audio file")
    value = local_audio_path(path.strip())
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.is_file():
        raise ValueError(f"not a local file: {resolved}")
    return str(resolved)


def _word(value, what):
    """A name or speaker is data, never a flag."""
    value = (value or "").strip()
    if value.startswith("-"):
        raise ValueError(f"{what} must not start with '-'")
    return value


def song_argv(path, view="feel", artist=None, title=None, *, listen=True):
    """argv for listen.py. artist/title only matter for a first listen."""
    if view not in SONG_VIEWS:
        raise ValueError(f"view must be one of: {', '.join(SONG_VIEWS)}")
    argv = [ears_python(), str(HERE / "listen.py"), path, "--view", SONG_VIEWS[view]]
    if listen:
        artist, title = _word(artist, "artist"), _word(title, "title")
        if artist:
            argv.append(f"--artist={artist}")
        if title:
            argv.append(f"--title={title}")
    return argv


def voice_argv(path, speaker=None, verbatim=False, no_baseline_update=False):
    argv = [ears_python(), str(HERE / "raven_ears_plus.py"), path, "--quiet"]
    speaker = _word(speaker, "speaker")
    if speaker:
        argv += ["--speaker", speaker]
    if verbatim:
        argv.append("--verbatim")
    if no_baseline_update:
        argv.append("--no-baseline-update")
    return argv


# ---------------------------------------------------------------- the song store

def _listen():
    import listen  # stdlib-only at import; gives us the store location and the content hash
    return listen


def song_dir(path):
    listen = _listen()
    return listen.store_home() / listen.file_hash(Path(path))


def bundle_exists(path):
    return (song_dir(path) / "song.json").is_file()


def listen_busy():
    """True while any listen.py holds the store lock (it holds it for the whole run)."""
    lock = _listen().store_home() / ".lock"
    if not lock.exists():
        return False
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def _jobs_dir():
    directory = _listen().store_home().parent / "mcp-jobs"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def _read_job(key):
    try:
        return json.loads((_jobs_dir() / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_job(key, record):
    (_jobs_dir() / f"{key}.json").write_text(json.dumps(record), encoding="utf-8")


def _log_tail(key):
    try:
        return (_jobs_dir() / f"{key}.log").read_text(encoding="utf-8", errors="replace")[-LOG_TAIL:].strip()
    except OSError:
        return ""


def job_state(path):
    """('none' | 'listening' | 'failed' | 'done', detail). Never starts or blocks on a listen."""
    key = _listen().file_hash(Path(path))
    record, proc = _read_job(key), _PROCS.get(key)
    bundle = bundle_exists(path)
    if proc is not None:
        running = proc.poll() is None
    elif record and record.get("state") == "started":
        # Started by an earlier server process: the lock is the witness, with a grace for startup.
        running = listen_busy() or time.time() - record.get("started", 0) < START_GRACE_S
    else:
        running = False
    if running:
        started = (record or {}).get("started")
        return "listening", f"{int(time.time() - started)}s so far" if started else "just started"
    if record and record.get("state") == "started":
        record["state"] = "finished"
        _write_job(key, record)
    if bundle:
        if record is None and listen_busy():
            return "listening", "another listen is using the ears"
        return "done", ""
    if record:
        return "failed", _log_tail(key) or "the listen ended without saving a card"
    return "none", ""


def start_listen(path, artist=None, title=None):
    """Start listen.py detached; returns at once. The bundle is saved by listen.py itself."""
    key = _listen().file_hash(Path(path))
    argv = song_argv(path, "feel", artist, title)
    log = open(_jobs_dir() / f"{key}.log", "wb")
    try:
        proc = subprocess.Popen(argv, cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                start_new_session=True)
    finally:
        log.close()
    _PROCS[key] = proc
    # Reap it when it ends. Otherwise it stays a zombie until somebody asks song_card, and
    # anything waiting on its pid (kill -0) waits forever.
    threading.Thread(target=proc.wait, name=f"reap-{proc.pid}", daemon=True).start()
    _write_job(key, {"state": "started", "started": time.time(), "pid": proc.pid})
    return proc


# ---------------------------------------------------------------- cards

def sheet_path_song(path):
    sheet = song_dir(path) / "listening-sheet.png"
    return str(sheet) if sheet.is_file() else None


def sheet_in_text(text):
    """The path on a `PICTURE:` line (it may wrap onto the next line), resolved against the repo."""
    match = re.search(r"^PICTURE:[ \t]*\n?[ \t]*(\S[^\n]*?\.png)[ \t]*$", text, re.M)
    if not match:
        return None
    sheet = Path(match.group(1)).expanduser()
    if not sheet.is_absolute():
        sheet = HERE / sheet
    return str(sheet) if sheet.is_file() else None


def run_cli(argv, timeout):
    done = subprocess.run(argv, cwd=HERE, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, done.stdout, done.stderr


def with_sheet(text, sheet):
    text = text.rstrip("\n")
    return f"{text}\n\n[listening sheet: {sheet}]" if sheet else text


def read_saved(path, view):
    """A saved card through listen.py. Only called once a bundle exists and nothing is listening."""
    code, out, err = run_cli(song_argv(path, view, listen=False), SAVED_TIMEOUT_S)
    if code != 0:
        raise RuntimeError((err or out).strip() or f"listen.py exited {code}")
    return with_sheet(out, sheet_path_song(path))


def song_card_text(path, view="feel"):
    path = local_audio(path)
    state, detail = job_state(path)
    if state == "listening":
        return f"still listening ({detail}). Call song_card again in a minute or two."
    if state == "failed":
        return f"the listen failed. Last of its log:\n{detail}"
    if state == "none":
        return "no saved card for this song. Call hear_song to listen first."
    return read_saved(path, view)


def hear_song_text(path, artist=None, title=None, view="feel"):
    path = local_audio(path)
    song_argv(path, view, artist, title)  # validate arguments before anything starts
    state, detail = job_state(path)
    if state == "listening":
        return f"already listening ({detail}). Call song_card in a few minutes."
    if state == "done":
        note = ("\n\n[already heard, so this is the saved card; artist/title only shape a first listen]"
                if (artist or title) else "")
        return read_saved(path, view) + note
    start_listen(path, artist, title)
    return ("listening. A new song takes 2 to 20 minutes. Call song_card with this path in a few "
            "minutes; it says 'still listening' until the card is saved.")


def hear_voice_text(path, speaker=None, verbatim=False, no_baseline_update=False):
    path = local_audio(path)
    argv = voice_argv(path, speaker, verbatim, no_baseline_update)
    code, out, err = run_cli(argv, VOICE_TIMEOUT_S)
    if code != 0:
        raise RuntimeError((err or out).strip()[-LOG_TAIL:] or f"raven_ears_plus.py exited {code}")
    return with_sheet(out, sheet_in_text(out))


# ---------------------------------------------------------------- the skill (Skills over MCP)

def skill_text():
    return SKILL_FILE.read_text(encoding="utf-8")


def frontmatter(text):
    """The SKILL.md frontmatter as a dict of strings (flat `key: value` lines, which is what ours is)."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not match:
        raise ValueError("SKILL.md has no frontmatter")
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() and not line.startswith((" ", "\t")):
            fields[key.strip()] = value.strip().strip("\"'")
    if not {"name", "description"} <= fields.keys():
        raise ValueError("SKILL.md frontmatter needs name and description")
    return fields


def skill_entry():
    """The spec's Skill entry: uri, verbatim frontmatter, a complete manifest with digests."""
    raw = SKILL_FILE.read_bytes()
    fields = frontmatter(raw.decode("utf-8"))
    if fields["name"] != SKILL_NAME:
        raise ValueError("the skill's name must equal the last segment of its URI")
    return {"uri": SKILL_URI, "frontmatter": fields,
            "resources": [{"uri": SKILL_URI, "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                           "size": len(raw)}]}


def build_server():
    from mcp.server.extension import Extension, MethodBinding, ResourceBinding
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.mcpserver.resources import TextResource
    from mcp_types import RequestParams

    class ListSkillsParams(RequestParams):
        cursor: str | None = None

    class GetSkillParams(RequestParams):
        uri: str

    async def skills_list(ctx, params):
        return {"resultType": "complete", "skills": [skill_entry()], "ttlMs": 300000, "cacheScope": "public"}

    async def skills_get(ctx, params):
        if params.uri != SKILL_URI:
            raise ValueError(f"no such skill: {params.uri}")
        return {"resultType": "complete", "skill": skill_entry(), "ttlMs": 300000, "cacheScope": "public"}

    class SkillsExtension(Extension):
        identifier = SKILLS_EXTENSION

        def resources(self):
            fields = frontmatter(skill_text())
            return [ResourceBinding(TextResource(
                uri=SKILL_URI, name=fields["name"], description=fields["description"],
                mime_type="text/markdown", text=skill_text()))]

        def methods(self):
            return [MethodBinding("skills/list", ListSkillsParams, skills_list),
                    MethodBinding("skills/get", GetSkillParams, skills_get)]

    async def call(fn, *args):
        """Run a blocking helper off the event loop; say what went wrong instead of 'Error executing tool'."""
        try:
            return await asyncio.to_thread(fn, *args)
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            raise ToolError(str(error)) from error

    server = MCPServer("raven-ears", instructions=INSTRUCTIONS, extensions=[SkillsExtension()])

    @server.prompt(name=SKILL_NAME, description=frontmatter(skill_text())["description"])
    def listening_prompt() -> str:
        """The listening skill, for clients that do not speak Skills over MCP yet."""
        return skill_text()

    @server.tool()
    async def hear_song(path: str, artist: str | None = None, title: str | None = None,
                        view: str = "feel") -> str:
        """Listen to a local song file. A new song takes 2 to 20 minutes, so this returns at once
        ("listening") and the card comes from song_card; a song already heard returns its saved card.
        view: feel (default, the feelings card), full, melody, or summary. Pass artist and title only if
        the file has no tags. The card ends with the listening sheet path when one was drawn."""
        return await call(hear_song_text, path, artist, title, view)

    @server.tool()
    async def song_card(path: str, view: str = "feel") -> str:
        """Read a SAVED card for a local song file without listening again. Says "still listening" while
        a hear_song is running, and "no saved card" if the song was never heard. view: feel, full,
        melody, or summary."""
        return await call(song_card_text, path, view)

    @server.tool()
    async def hear_voice(path: str, speaker: str | None = None, verbatim: bool = False,
                         no_baseline_update: bool = False) -> str:
        """Listen to a local voice note: words, pitch, pace, laughs, the wordless stretches. Waits for
        the card (seconds to a minute or two). speaker names whose voice it is, which turns on VS USUAL.
        verbatim keeps fillers and stutters. Set no_baseline_update when replaying an old note just to
        look, so their usual voice does not learn it twice. A friend's voice is theirs: ask first."""
        return await call(hear_voice_text, path, speaker, verbatim, no_baseline_update)

    return server


def main():
    build_server().run("stdio")


if __name__ == "__main__":
    main()
