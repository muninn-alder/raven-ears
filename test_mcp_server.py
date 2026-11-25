"""Tests for mcp_server.py. No network, no GPU, no real listening: subprocesses are mocked.

The tests that need the MCP SDK (requirements-mcp.txt) skip cleanly when it is not installed.
"""
import asyncio
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import mcp_server

try:
    import mcp  # noqa: F401
    HAVE_SDK = True
except ImportError:
    HAVE_SDK = False


class FakeProc:
    def __init__(self, code=None):
        self.code, self.pid = code, 4242

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"RAVEN_EARS_HOME": str(root / "ears")})
        env.start()
        self.addCleanup(env.stop)
        mcp_server._PROCS.clear()
        self.addCleanup(mcp_server._PROCS.clear)
        self.song = root / "song.mp3"
        self.song.write_bytes(b"not really audio")
        self.key = hashlib.sha256(b"not really audio").hexdigest()
        self.home = root / "ears" / "songs"

    def save_bundle(self, sheet=False):
        folder = self.home / self.key
        folder.mkdir(parents=True)
        (folder / "song.json").write_text("{}")
        if sheet:
            (folder / "listening-sheet.png").write_bytes(b"png")


class ArgvTests(Base):
    def test_song_argv_is_explicit_and_maps_views(self):
        argv = mcp_server.song_argv(str(self.song), "feel", "Some Band", "A Song")
        self.assertEqual(argv[1:], [str(mcp_server.HERE / "listen.py"), str(self.song), "--view", "first-listen",
                                    "--artist=Some Band", "--title=A Song"])
        for view, cli in (("full", "full"), ("melody", "melody"), ("summary", "summary")):
            self.assertEqual(mcp_server.song_argv("/a.mp3", view)[-2:], ["--view", cli])
        with self.assertRaises(ValueError):
            mcp_server.song_argv("/a.mp3", "detail")

    def test_saved_read_never_carries_listen_options(self):
        argv = mcp_server.song_argv("/a.mp3", "feel", "X", "Y", listen=False)
        self.assertNotIn("--artist=X", argv)

    def test_names_cannot_become_flags(self):
        with self.assertRaises(ValueError):
            mcp_server.song_argv("/a.mp3", "feel", artist="--fresh")
        with self.assertRaises(ValueError):
            mcp_server.voice_argv("/a.ogg", speaker="--json")

    def test_voice_argv(self):
        argv = mcp_server.voice_argv("/a.ogg", "fox", True, True)
        self.assertEqual(argv[1:], [str(mcp_server.HERE / "raven_ears_plus.py"), "/a.ogg", "--quiet",
                                    "--speaker", "fox", "--verbatim", "--no-baseline-update"])
        self.assertEqual(mcp_server.voice_argv("/a.ogg")[2:], ["/a.ogg", "--quiet"])

    def test_interpreter_choice(self):
        with mock.patch.dict(os.environ, {"RAVEN_EARS_PYTHON": "/x/python"}):
            self.assertEqual(mcp_server.ears_python(), "/x/python")


class PathTests(Base):
    def test_urls_and_protocols_are_refused(self):
        for bad in ("https://example.com/a.mp3", "http://x/y", "file:///etc/passwd", "ftp://h/a", "//host/share/a.mp3",
                    "\\\\host\\share\\a.mp3", "s3:bucket/key"):
            with self.assertRaises(ValueError, msg=bad):
                mcp_server.local_audio(bad)

    def test_must_be_an_existing_file(self):
        with self.assertRaises(ValueError):
            mcp_server.local_audio(self.tmp.name)
        with self.assertRaises(ValueError):
            mcp_server.local_audio(str(Path(self.tmp.name) / "missing.mp3"))
        with self.assertRaises(ValueError):
            mcp_server.local_audio("  ")
        self.assertEqual(mcp_server.local_audio(str(self.song)), str(self.song))

    def test_tools_refuse_urls_before_running_anything(self):
        with mock.patch("subprocess.run") as run, mock.patch("subprocess.Popen") as popen:
            for call in (lambda: mcp_server.hear_song_text("https://x/a.mp3"),
                         lambda: mcp_server.song_card_text("https://x/a.mp3"),
                         lambda: mcp_server.hear_voice_text("https://x/a.ogg")):
                with self.assertRaises(ValueError):
                    call()
        run.assert_not_called()
        popen.assert_not_called()


class BackgroundStateMachine(Base):
    def test_never_heard(self):
        self.assertEqual(mcp_server.job_state(str(self.song)), ("none", ""))
        with mock.patch.object(mcp_server, "run_cli") as run:
            self.assertIn("no saved card", mcp_server.song_card_text(str(self.song)))
        run.assert_not_called()  # song_card must never start a listen

    def test_new_song_starts_in_the_background_and_song_card_waits(self):
        proc = FakeProc()
        with mock.patch("subprocess.Popen", return_value=proc) as popen:
            said = mcp_server.hear_song_text(str(self.song), "A", "B")
        self.assertIn("listening", said)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[1:3], [str(mcp_server.HERE / "listen.py"), str(self.song)])
        self.assertIn("--artist=A", argv)
        self.assertFalse(popen.call_args.kwargs.get("shell", False))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        with mock.patch.object(mcp_server, "run_cli") as run:
            self.assertEqual(mcp_server.job_state(str(self.song))[0], "listening")
            self.assertTrue(mcp_server.song_card_text(str(self.song)).startswith("still listening"))
            # a checkpointed, half-finished bundle must not be served while the job runs
            self.save_bundle()
            self.assertTrue(mcp_server.song_card_text(str(self.song)).startswith("still listening"))
            self.assertIn("already listening", mcp_server.hear_song_text(str(self.song)))
        run.assert_not_called()
        with mock.patch("subprocess.Popen") as popen:
            mcp_server.hear_song_text(str(self.song))
        popen.assert_not_called()

    def test_the_child_is_reaped_when_it_ends_without_anyone_asking(self):
        # Muninn, 25 Nov 2026: a finished listen stayed <defunct>, so kill -0 on its pid waited forever.
        with mock.patch.object(mcp_server, "song_argv", return_value=[sys.executable, "-c", "pass"]):
            proc = mcp_server.start_listen(str(self.song))
        for _ in range(100):
            try:
                os.kill(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            self.fail("the finished listen was left as a zombie")

    def test_finished_job_serves_the_saved_card_with_the_sheet(self):
        proc = FakeProc()
        mcp_server._PROCS[self.key] = proc
        mcp_server._write_job(self.key, {"state": "started", "started": 1})
        self.save_bundle(sheet=True)
        proc.code = 0
        with mock.patch.object(mcp_server, "run_cli", return_value=(0, "THE CARD\n", "")) as run:
            text = mcp_server.song_card_text(str(self.song), "full")
        self.assertEqual(mcp_server.job_state(str(self.song))[0], "done")
        self.assertTrue(text.startswith("THE CARD"))
        self.assertIn(str(self.home / self.key / "listening-sheet.png"), text)
        self.assertEqual(run.call_args.args[0][-2:], ["--view", "full"])
        self.assertNotIn("--run-missing", run.call_args.args[0])

    def test_failed_job_says_why(self):
        proc = FakeProc(code=2)
        mcp_server._PROCS[self.key] = proc
        mcp_server._write_job(self.key, {"state": "started", "started": 1})
        (mcp_server._jobs_dir() / f"{self.key}.log").write_text("listen: supply an audio file\n")
        state, detail = mcp_server.job_state(str(self.song))
        self.assertEqual(state, "failed")
        self.assertIn("supply an audio file", detail)
        self.assertIn("the listen failed", mcp_server.song_card_text(str(self.song)))
        # the next hear_song tries again
        with mock.patch("subprocess.Popen", return_value=FakeProc()) as popen:
            mcp_server.hear_song_text(str(self.song))
        popen.assert_called_once()

    def test_saved_song_is_read_not_relistened(self):
        self.save_bundle()
        with mock.patch("subprocess.Popen") as popen, \
                mock.patch.object(mcp_server, "run_cli", return_value=(0, "SAVED\n", "")) as run:
            text = mcp_server.hear_song_text(str(self.song), artist="X", title="Y", view="summary")
        popen.assert_not_called()
        argv = run.call_args.args[0]
        self.assertEqual(argv[-2:], ["--view", "summary"])
        self.assertFalse([a for a in argv if a.startswith("--artist")])
        self.assertIn("already heard", text)

    def test_after_a_server_restart_the_store_lock_is_the_witness(self):
        old = {"state": "started", "started": 1}  # long ago, process unknown to this server
        mcp_server._write_job(self.key, old)
        self.assertEqual(mcp_server.job_state(str(self.song))[0], "failed")  # lock free, nothing saved
        mcp_server._write_job(self.key, old)
        self.home.mkdir(parents=True, exist_ok=True)
        with (self.home / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.assertTrue(mcp_server.listen_busy())
            self.assertEqual(mcp_server.job_state(str(self.song))[0], "listening")
        self.assertFalse(mcp_server.listen_busy())

    def test_a_foreign_listen_holding_the_lock_never_blocks_song_card(self):
        self.save_bundle()
        self.home.mkdir(parents=True, exist_ok=True)
        with (self.home / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with mock.patch.object(mcp_server, "run_cli") as run:
                self.assertTrue(mcp_server.song_card_text(str(self.song)).startswith("still listening"))
        run.assert_not_called()


class VoiceTests(Base):
    def test_voice_card_and_sheet_line(self):
        note = Path(self.tmp.name) / "note.ogg"
        note.write_bytes(b"x")
        sheet = Path(self.tmp.name) / "note.png"
        sheet.write_bytes(b"png")
        out = f"WORDS: hi\nPICTURE: {sheet}\n"
        with mock.patch.object(mcp_server, "run_cli", return_value=(0, out, "")) as run:
            text = mcp_server.hear_voice_text(str(note), speaker="fox")
        self.assertEqual(run.call_args.args[0][-2:], ["--speaker", "fox"])
        self.assertIn(f"[listening sheet: {sheet}]", text)

    def test_wrapped_picture_line_and_failure(self):
        sheet = Path(self.tmp.name) / "deep" / "sheet.png"
        sheet.parent.mkdir()
        sheet.write_bytes(b"png")
        self.assertEqual(mcp_server.sheet_in_text(f"A\nPICTURE:\n{sheet}\n"), str(sheet))
        self.assertIsNone(mcp_server.sheet_in_text("PICTURE unavailable: no matplotlib\n"))
        note = Path(self.tmp.name) / "n.ogg"
        note.write_bytes(b"x")
        with mock.patch.object(mcp_server, "run_cli", return_value=(2, "", "boom")):
            with self.assertRaises(RuntimeError):
                mcp_server.hear_voice_text(str(note))

    def test_offline_policy_reaches_the_child(self):
        # The CLIs inherit os.environ, so RAVEN_EARS_OFFLINE=1 is honoured by their own lookups.
        with mock.patch.dict(os.environ, {"RAVEN_EARS_OFFLINE": "1"}), \
                mock.patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, "ok", "")
            mcp_server.run_cli(["x"], 5)
        self.assertNotIn("env", run.call_args.kwargs)  # inherited, not replaced


class SkillTests(Base):
    def test_entry_matches_the_file(self):
        raw = mcp_server.SKILL_FILE.read_bytes()
        entry = mcp_server.skill_entry()
        self.assertEqual(entry["uri"], "skill://listening/SKILL.md")
        self.assertEqual(entry["frontmatter"]["name"], "listening")
        self.assertTrue(entry["frontmatter"]["description"].startswith("How to listen to a song"))
        self.assertEqual(entry["resources"], [{"uri": entry["uri"], "size": len(raw),
                                               "digest": "sha256:" + hashlib.sha256(raw).hexdigest()}])

    def test_frontmatter_needs_name_and_description(self):
        with self.assertRaises(ValueError):
            mcp_server.frontmatter("no frontmatter here")
        with self.assertRaises(ValueError):
            mcp_server.frontmatter("---\nname: x\n---\nbody")


@unittest.skipUnless(HAVE_SDK, "the MCP SDK is not installed (pip install -r requirements-mcp.txt)")
class ServedOverMcp(Base):
    def run_client(self, work):
        from mcp.client.client import Client

        async def go():
            async with Client(mcp_server.build_server()) as client:
                return await work(client)
        return asyncio.run(go())

    def test_tools_resource_prompt_and_extension(self):
        async def work(client):
            tools = {t.name for t in (await client.list_tools()).tools}
            resource = await client.read_resource("skill://listening/SKILL.md")
            prompt = await client.get_prompt("listening")
            return tools, resource.contents[0].text, prompt.messages[0].content.text, client.server_capabilities
        tools, text, prompt_text, caps = self.run_client(work)
        self.assertEqual(tools, {"hear_song", "song_card", "hear_voice"})
        self.assertEqual(text, mcp_server.skill_text())
        self.assertEqual(prompt_text, mcp_server.skill_text())
        self.assertIn("io.modelcontextprotocol/skills", caps.extensions)

    def test_skills_list_and_get(self):
        import mcp_types as t
        from pydantic import ConfigDict
        from typing import Literal

        class Params(t.RequestParams):
            uri: str | None = None

        class ListReq(t.Request[Params, Literal["skills/list"]]):
            method: Literal["skills/list"] = "skills/list"

        class GetReq(t.Request[Params, Literal["skills/get"]]):
            method: Literal["skills/get"] = "skills/get"

        class Any(t.Result):
            model_config = ConfigDict(extra="allow")

        async def work(client):
            listed = await client.session.send_request(ListReq(params=Params()), Any)
            got = await client.session.send_request(GetReq(params=Params(uri=mcp_server.SKILL_URI)), Any)
            return listed.model_dump(by_alias=True), got.model_dump(by_alias=True)
        listed, got = self.run_client(work)
        self.assertEqual(listed["resultType"], "complete")
        self.assertEqual(listed["skills"], [mcp_server.skill_entry()])
        self.assertEqual(got["skill"], mcp_server.skill_entry())

    def test_a_url_comes_back_as_a_readable_tool_error(self):
        async def work(client):
            return await client.call_tool("hear_song", {"path": "https://example.com/a.mp3"})
        result = self.run_client(work)
        self.assertTrue(result.is_error)
        self.assertIn("must be a local file", result.content[0].text)


if __name__ == "__main__":
    unittest.main()
