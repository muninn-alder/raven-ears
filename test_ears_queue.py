"""Queue checks: temporary files, fake subprocesses; no models or network."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import ears_queue as Q


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ears_queue_test_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.engines = self.root / "engines"
        self.engines.mkdir()
        for engine in Q.ENGINES:
            (self.engines / (engine + ".py")).write_text("# fake engine\n")
        self.audio = self.root / "neutral song.mp3"
        self.audio.write_bytes(b"not audio: no decoder will be called")
        self.queue = Q.Queue(self.root / "state", self.root / "cards", self.engines)
        self.calls = []
        self.output = io.StringIO()
        self.stdout = redirect_stdout(self.output)
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.command_patch = patch.object(self.queue, "command", side_effect=self.fake_command)
        self.mock_command = self.command_patch.start()
        self.addCleanup(self.command_patch.stop)

    def fake_command(self, args, lock):
        self.calls.append(args)
        engine = Path(args[1]).stem
        return 0, Q.HEADERS[engine] + " neutral example\nand now, you\n"

    def enqueue(self, *targets):
        return self.queue.add(targets or [str(self.audio)])

    def events(self):
        return [json.loads(x) for x in (self.queue.state / "listens.jsonl").read_text().splitlines()]

    def test_add_text_list_relative_paths_comments_and_deduplication(self):
        songs = self.root / "songs.txt"
        songs.write_text("\ufeff# examples only\n\nneutral song.mp3\nExample Artist - Example Song\nneutral song.mp3\n")
        items = self.enqueue(str(songs))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["target"], str(self.audio))
        self.assertEqual(items[1]["target"], "Example Artist - Example Song")
        self.assertEqual(len(self.enqueue(str(self.audio))), 2)

    def test_bad_input_does_not_partially_add(self):
        self.enqueue()
        for bad in ["--bad-option", "", "one\ntwo", str(self.root / "missing.mp3")]:
            with self.assertRaises(Q.QueueError):
                self.enqueue("another song", bad)
        self.assertEqual(len(self.queue.items()), 1)
        self.assertFalse(self.calls)

    def test_remove_unique_prefix_keeps_cards(self):
        item = self.enqueue()[0]
        self.queue.run()
        self.queue.remove(item["id"][:8])
        self.assertEqual(self.queue.items(), [])
        self.assertTrue((self.queue.cards / "neutral song.txt").is_file())
        with self.assertRaises(Q.QueueError):
            self.queue.remove("no-such-id")

    def test_full_listen_runs_three_engines_sequentially_and_ends_with_listener(self):
        self.enqueue()
        self.assertEqual(self.queue.run(), 0)
        self.assertEqual([Path(x[1]).stem for x in self.calls], list(Q.ENGINES))
        card = self.queue.cards / "neutral song.txt"
        self.assertEqual(card.read_text().splitlines()[-1], "and now, you")
        self.assertTrue(card.with_suffix(".queue.json").is_file())
        event = self.events()[-1]
        self.assertEqual(event["status"], "success")
        self.assertEqual(len(event["stages"]), 3)
        self.assertTrue(all(s["ok"] and s["duration_s"] >= 0 for s in event["stages"]))
        self.assertEqual(self.queue.items()[0]["status"], "done")

    def test_fresh_card_skips_all_engines(self):
        self.enqueue()
        self.queue.run()
        self.calls.clear()
        self.queue.run()
        self.assertFalse(self.calls)
        self.assertEqual(self.events()[-1]["status"], "skipped")

    def test_source_edit_invalidates_done_card_without_git_commit(self):
        self.enqueue()
        self.queue.run()
        (self.engines / "shared_dependency.py").write_text("# new implementation\n")
        self.calls.clear()
        self.queue.run()
        self.assertEqual(len(self.calls), 3)

    def test_audio_edit_and_card_tampering_invalidate_stamp(self):
        self.enqueue()
        self.queue.run()
        self.audio.write_bytes(b"new neutral bytes")
        self.calls.clear()
        self.queue.run()
        self.assertEqual(len(self.calls), 3)
        card = self.queue.cards / "neutral song.txt"
        card.write_text("a partial overwrite\n")
        self.calls.clear()
        self.queue.run()
        self.assertEqual(len(self.calls), 3)

    def test_old_unstamped_card_is_not_assumed_complete(self):
        self.enqueue()
        card = self.queue.cards / "neutral song.txt"
        Q.atomic_text(card, "MUSIC: only one engine\nand now, you\n")
        self.queue.run()
        self.assertEqual(len(self.calls), 3)

    def test_card_must_be_newer_than_code(self):
        self.enqueue()
        self.queue.run()
        card = self.queue.cards / "neutral song.txt"
        os.utime(card, (1, 1))
        self.calls.clear()
        self.queue.run()
        self.assertEqual(len(self.calls), 3)

    def test_changed_setting_invalidates_stage_and_card(self):
        self.enqueue()
        self.queue.run(reuse_stages=True)
        self.calls.clear()
        with patch.dict(os.environ, {"RAVEN_NO_CREDITS": "1"}):
            self.queue.run(reuse_stages=True)
        self.assertEqual(len(self.calls), 3)

    def test_failed_engine_is_logged_retryable_and_does_not_replace_good_card(self):
        self.enqueue()
        self.queue.run()
        card = self.queue.cards / "neutral song.txt"
        original = card.read_bytes()
        self.audio.write_bytes(b"changed input")

        def fail(args, lock):
            if Path(args[1]).stem == "singer":
                return 7, "GPU backend could not load weights\n"
            return self.fake_command(args, lock)

        self.mock_command.side_effect = fail
        self.assertEqual(self.queue.run(), 1)
        self.assertEqual(card.read_bytes(), original)
        failed_card = card.with_suffix(".failed.txt").read_text()
        self.assertIn("missing detail is not silence", failed_card)
        self.assertTrue(failed_card.endswith("and now, you\n"))
        self.assertIn("exit 7", self.queue.items()[0]["reason"])
        self.assertEqual(self.events()[-1]["status"], "failed")
        self.assertEqual(self.events()[-1]["card"], str(card.with_suffix(".failed.txt")))
        self.mock_command.side_effect = self.fake_command
        self.assertEqual(self.queue.run(), 0)
        self.assertEqual(self.queue.items()[0]["status"], "done")

    def test_exit_zero_backend_failure_and_empty_output_are_not_success(self):
        for output in ["", "audio too short for music analysis (<5 s)",
                       "MUSIC: example\n  NOTE: lyrics: failed (missing model)",
                       "MUSIC: example\n[music] meter claim unavailable (missing backend)",
                       "MUSIC: example\n  vocal quality skipped (ImportError)",
                       "MUSIC: example\nTraceback (most recent call last):\nError"]:
            self.assertIsNotNone(Q.stage_problem("music", output, 0), output)
        self.assertIsNone(Q.stage_problem("music", "MUSIC: example\n [0:10] the light failed me\n  KEY: unclear\n", 0))

    def test_reuse_successful_stages_after_failure_only_with_flag(self):
        self.enqueue()

        def fail(args, lock):
            result = self.fake_command(args, lock)
            return (2, "missing singer weights") if Path(args[1]).stem == "singer" else result

        self.mock_command.side_effect = fail
        self.assertEqual(self.queue.run(reuse_stages=True), 1)
        self.mock_command.side_effect = self.fake_command
        self.calls.clear()
        self.assertEqual(self.queue.run(reuse_stages=True), 0)
        self.assertEqual([Path(x[1]).stem for x in self.calls], ["singer"])
        self.assertEqual([s["reused"] for s in self.events()[-1]["stages"]], [True, False, True])
        self.assertIn("QUEUE: reused earlier engine output", (self.queue.cards / "neutral song.txt").read_text())
        # Without the flag, a stale card still runs every engine.
        (self.queue.cards / "neutral song.txt").unlink()
        self.calls.clear()
        self.queue.run()
        self.assertEqual(len(self.calls), 3)

    def test_intentional_song_skips_complete_and_reuse(self):
        # STATUS line shapes/reasons copied from the three saved studio cards.
        common = ("  STATUS: breath: skipped: the event model does not hear breathing here\n"
                  "  STATUS: Whisper: skipped: word times supplied\n"
                  "  STATUS: VS USUAL: skipped: no speaker given\n"
                  "  STATUS: credits (Discogs): skipped: not needed\n"
                  "NOTES: breath: skipped: the event model does not hear breathing here · +3 more\n")
        instrumental = ("  STATUS: Whisper: skipped: LRCLIB instrumental designation\n"
                        "  STATUS: words: skipped: LRCLIB instrumental designation\n"
                        "  STATUS: Whisper: skipped: --no-whisper\n"
                        "  STATUS: words: skipped: --no-Whisper, no word times given\n"
                        "  STATUS: pace / WORDLESS / emphasis: skipped: no word times\n"
                        "  STATUS: word / music contrast: skipped: no lyric lines\n"
                        "  STATUS: refrain check: skipped: no heard words\n"
                        "NOTES: Whisper: skipped: LRCLIB instrumental designation · +6 more\n")
        for notes in (common, instrumental):
            with self.subTest(notes=notes):
                for engine in Q.ENGINES:
                    self.assertIsNone(Q.stage_problem(engine, Q.HEADERS[engine] + " example\n" + notes, 0))
        self.enqueue()
        def recorded(args, lock):
            code, text = self.fake_command(args, lock)
            return code, text.replace("and now, you", common + "and now, you")
        self.mock_command.side_effect = recorded
        self.assertEqual(self.queue.run(reuse_stages=True), 0)
        self.calls.clear()
        self.assertEqual(self.queue.run(reuse_stages=True), 0)
        self.assertFalse(self.calls)
        (self.queue.cards / "neutral song.txt").unlink()
        self.assertEqual(self.queue.run(reuse_stages=True), 0)
        self.assertFalse(self.calls)

    def test_stage_states_and_required_missing_work_still_fail(self):
        for note in ("Whisper: skipped: missing model", "Whisper: skipped: --no-whisper",
                     "words: skipped: --no-Whisper, no word times given",
                     "meter: skipped: --no-meter", "AFFECT: failed: missing weights",
                     "AFFECT unavailable (ImportError)"):
            with self.subTest(note=note):
                self.assertIsNotNone(Q.stage_problem("music", "MUSIC: example\nNOTES: " + note, 0))
        split = ("MUSIC: example\nNOTES: words: skipped: LRCLIB instrumental designation\n"
                 "and now, you\nSINGER.PY example\nNOTES: Whisper: skipped: --no-whisper\n")
        self.assertIsNotNone(Q.stage_problem("singer", split, 0))
        self.assertIsNotNone(Q.stage_problem("music", "MUSIC: example\nNOTES: all stages ran · +1 more", 0))
        for status, reason, ok in (("ran", "earlier attempt failed; final scan complete", True),
                                   ("skipped", "word times supplied", True),
                                   ("failed", "word times supplied", False),
                                   ("degraded", "retried on CPU", True),
                                   ("skipped", "missing model", False)):
            rows = [dict(stage="Whisper", status=status, kind="audio", reason=reason)]
            self.assertEqual(Q.stage_problem("music", "MUSIC: example\n", 0, rows) is None, ok)
        for invalid in ({}, [None], [dict(stage="Whisper", status="unknown")]):
            self.assertIsNotNone(Q.stage_problem("music", "MUSIC: example\n", 0, invalid))

    def test_degraded_work_is_finished_not_retried(self):
        # CPU-profile stems are always htdemucs; a retry would degrade identically.
        for note in ("stems: degraded: CPU profile: htdemucs instead of htdemucs_ft; card quality difference unmeasured",
                     "Whisper: degraded: fell back to CPU int8"):
            with self.subTest(note=note):
                self.assertIsNone(Q.stage_problem("singer", "SINGER.PY example\nNOTES: " + note, 0))
                self.assertIsNotNone(Q.stage_problem("singer", "SINGER.PY example\nNOTES: " + note
                                                     + " · AFFECT: failed: missing weights", 0))

    def test_legacy_skips_are_exact_and_failures_cannot_hide_beside_them(self):
        for note in ("NOTE: Whisper: skipped: word times supplied",
                     "breath: skipped: no breath-rate evidence",
                     "CREDITS: Example; Discogs skipped (no DISCOGS_TOKEN set; its search needs one); ears only"):
            self.assertIsNone(Q.stage_problem("music", "MUSIC: example\n" + note, 0))
            self.assertIsNotNone(Q.stage_problem("music", "MUSIC: example\n" + note + "\nAFFECT: unavailable", 0))
        self.assertIsNotNone(Q.stage_problem("music", "MUSIC: example\nWhisper: skipped: missing model", 0))
        self.assertIsNotNone(Q.stage_problem("music", "MUSIC: example\nNOTE: Whisper: skipped: word times supplied; failed model", 0))
        for failure in ("unreachable (URLError)", "timed out after 6 s"):
            credits = ("CREDITS: MusicBrainz " + failure +
                       "; Discogs skipped (no DISCOGS_TOKEN set; its search needs one); ears only")
            self.assertIsNotNone(Q.stage_problem("music", "MUSIC: example\n" + credits, 0))

    def test_tampered_stage_checkpoint_is_recomputed(self):
        self.enqueue()
        self.queue.run(reuse_stages=True)
        (self.queue.cards / "neutral song.txt").unlink()
        saved = next((self.queue.state / "stages").glob("*/music.json"))
        data = json.loads(saved.read_text())
        data["output"] = "MUSIC: altered text\n"
        saved.write_text(json.dumps(data))
        self.calls.clear()
        self.queue.run(reuse_stages=True)
        self.assertEqual([Path(x[1]).stem for x in self.calls], ["music"])

    def test_refresh_bypasses_card_and_stage_reuse(self):
        self.enqueue()
        self.queue.run(reuse_stages=True)
        self.calls.clear()
        self.queue.run(reuse_stages=True, refresh=True)
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(all(not s["reused"] for s in self.events()[-1]["stages"]))

    def test_crash_status_is_recovered_and_logged(self):
        item = self.enqueue()[0]
        self.queue.update(item["id"], status="running", attempts=1)
        self.queue.run()
        recovered = [x for x in self.events() if x["status"] == "interrupted"]
        self.assertEqual(len(recovered), 1)
        self.assertIsNone(recovered[0]["duration_s"])
        self.assertEqual(self.queue.items()[0]["attempts"], 2)

    def test_max_excludes_fresh_cards_and_bounds_attempts(self):
        self.enqueue()
        self.queue.run()
        second = self.root / "second.mp3"
        third = self.root / "third.mp3"
        second.write_bytes(b"second")
        third.write_bytes(b"third")
        self.enqueue(str(second), str(third))
        self.calls.clear()
        self.queue.run(maximum=1)
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(all(x[-1] == str(second) for x in self.calls))
        self.assertEqual(self.queue.items()[2]["status"], "pending")
        self.calls.clear()
        self.queue.run(maximum=0)
        self.assertFalse(self.calls)

    def test_until_is_today_local_soft_cutoff(self):
        now = datetime(2026, 11, 4, 3, 45)
        self.assertEqual(Q.cutoff("07:00", now), datetime(2026, 11, 4, 7, 0))
        for invalid in ["24:00", "4:15", "07:60", "tomorrow"]:
            with self.assertRaises(Exception):
                Q.cutoff(invalid, now)
        self.enqueue()
        self.queue.run(until=datetime.now() - timedelta(seconds=1))
        self.assertFalse(self.calls)
        self.assertEqual(self.queue.items()[0]["status"], "pending")

    def test_cutoff_between_songs_finishes_current_song(self):
        self.enqueue(str(self.audio), "another song")
        before = datetime.now()

        class Clock:
            @staticmethod
            def now():
                return before if len(self.calls) < 3 else before + timedelta(hours=2)

        with patch.object(Q, "datetime", Clock):
            self.queue.run(until=before + timedelta(hours=1))
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.queue.items()[1]["status"], "pending")

    def test_failed_fetch_is_logged_and_does_not_stop_next_song(self):
        self.enqueue("example search", str(self.audio))

        def fetch(args, lock):
            if len(args) == 2:
                return 9, "fetch backend unavailable"
            return self.fake_command(args, lock)

        self.mock_command.side_effect = fetch
        self.assertEqual(self.queue.run(), 1)
        self.assertIn("raven-song exit 9", self.queue.items()[0]["reason"])
        self.assertEqual(self.queue.items()[1]["status"], "done")

    def test_fetch_once_and_resume_from_remembered_audio(self):
        self.enqueue("example search")

        def fetch(args, lock):
            if len(args) == 2:
                self.calls.append(args)
                return 0, "shelf match\n" + str(self.audio) + "\n"
            return self.fake_command(args, lock)

        self.mock_command.side_effect = fetch
        self.queue.run()
        self.assertEqual(len(self.calls), 4)
        self.calls.clear()
        self.queue.run()
        self.assertFalse(self.calls)

    def test_missing_local_path_is_never_fetched(self):
        self.enqueue()
        self.audio.unlink()
        self.queue.run()
        self.assertFalse(self.calls)
        self.assertIn("audio file is missing", self.queue.items()[0]["reason"])

    def test_same_basename_different_inputs_keep_separate_cards(self):
        self.enqueue()
        self.queue.run()
        original = (self.queue.cards / "neutral song.txt").read_bytes()
        second = self.root / "other" / self.audio.name
        second.parent.mkdir()
        second.write_bytes(b"a different recording")
        self.enqueue(str(second))
        self.queue.run()
        self.assertEqual((self.queue.cards / "neutral song.txt").read_bytes(), original)
        self.assertEqual(len(list(self.queue.cards.glob("*.txt"))), 2)
        self.calls.clear()
        self.queue.run()
        self.assertFalse(self.calls)

    def test_code_change_during_listen_is_not_stamped_successful(self):
        self.enqueue()

        def changed(args, lock):
            (self.engines / "changed.py").write_text("# changed while listening\n")
            return self.fake_command(args, lock)

        self.mock_command.side_effect = changed
        self.assertEqual(self.queue.run(), 1)
        self.assertFalse((self.queue.cards / "neutral song.queue.json").exists())
        self.assertIn("changed during listening", self.queue.items()[0]["reason"])

    def test_add_during_listen_survives_but_waits_for_next_run(self):
        self.enqueue()

        def adding(args, lock):
            self.enqueue("added while listening")
            return self.fake_command(args, lock)

        self.mock_command.side_effect = adding
        self.queue.run()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.queue.items()[1]["status"], "pending")

    def test_second_run_and_removal_refuse_held_lock_but_add_is_allowed(self):
        item = self.enqueue()[0]
        with Q.locked(self.queue.state / "run.lock"):
            with self.assertRaisesRegex(Q.QueueError, "another queue run"):
                self.queue.run()
            with self.assertRaises(Q.QueueError):
                self.queue.remove(item["id"])
            self.enqueue("added safely")
        self.assertFalse(self.calls)

    def test_interrupt_logs_retryable_reason(self):
        self.enqueue()
        self.mock_command.side_effect = Q.Interrupted("signal 15")
        with self.assertRaises(Q.Interrupted):
            self.queue.run()
        self.assertEqual(self.queue.items()[0]["status"], "interrupted")
        self.assertEqual(self.events()[-1]["status"], "interrupted")

    def test_corrupt_queue_refuses_overwrite(self):
        self.enqueue()
        state_file = self.queue.state / "queue.json"
        for text in ["broken json", "{}", '[{"target": "incomplete"}]']:
            state_file.write_text(text)
            with self.assertRaises((Q.QueueError, ValueError)):
                self.enqueue("another song")
            self.assertEqual(state_file.read_text(), text)

    def test_invalid_sidecar_is_safe_cache_miss(self):
        self.enqueue()
        self.queue.run()
        (self.queue.cards / "neutral song.queue.json").write_text("{broken")
        self.calls.clear()
        self.queue.run()
        self.assertEqual(len(self.calls), 3)

    def test_cli_env_overrides_and_listener_on_every_command(self):
        env = {"RAVEN_QUEUE_DIR": str(self.queue.state), "RAVEN_CARDS_DIR": str(self.queue.cards)}
        with patch.dict(os.environ, env):
            instance = Q.Queue()
            self.assertEqual(instance.state, self.queue.state)
            self.assertEqual(instance.cards, self.queue.cards)
            for args in [["add", "example search"], ["list"], ["cron"], ["remove", "unknown"]]:
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = Q.cli(args)
                self.assertEqual(stdout.getvalue().splitlines()[-1], "and now, you")
                self.assertEqual(result, 1 if args[0] == "remove" else 0)
            for args in [["--help"], ["run", "--max", "-1"], ["run", "--until", "99:10"]]:
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(SystemExit):
                    Q.cli(args)
                self.assertTrue(stdout.getvalue().endswith("and now, you\n"))

    def test_cron_is_quoted_self_contained_and_does_not_install_or_write(self):
        self.queue.state = self.root / "state with ' quote and %"
        line = Q.cron_line(self.queue)
        self.assertTrue(line.startswith("45 3 * * * mkdir -p "))
        self.assertIn("RAVEN_QUEUE_DIR=", line)
        self.assertIn("RAVEN_CARDS_DIR=", line)
        self.assertIn("\\%", line)
        self.assertIn(" >> ", line)
        self.assertFalse(self.queue.state.exists())

    def test_real_subprocess_failure_stdout_and_stderr_are_preserved(self):
        self.command_patch.stop()
        with Q.locked(self.queue.state / "run.lock") as lock:
            code, output = self.queue.command([sys.executable, "-c",
                "import sys; print('neutral stdout'); print('neutral stderr', file=sys.stderr); sys.exit(4)"], lock)
        self.assertEqual(code, 4)
        self.assertIn("neutral stdout", output)
        self.assertIn("neutral stderr", output)

    def test_sigterm_waits_for_child_and_records_interrupted_status(self):
        self.command_patch.stop()
        self.enqueue()
        marker = self.root / "engine-ready"
        for engine in Q.ENGINES:
            (self.engines / (engine + ".py")).write_text(
                "from pathlib import Path\nimport os, time\n"
                f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n")
        parent_code = (
            "import signal, sys; import ears_queue as Q\n"
            "def stop(s, f): raise Q.Interrupted('signal 15')\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "q = Q.Queue(*sys.argv[1:4])\n"
            "try: q.run()\n"
            "except Q.Interrupted: sys.exit(130)\n")
        parent = subprocess.Popen([sys.executable, "-c", parent_code, str(self.queue.state),
                                   str(self.queue.cards), str(self.engines)], cwd=Q.HERE,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists(), "fake engine did not start")
            parent.terminate()
            self.assertEqual(parent.wait(timeout=5), 130)
            self.assertEqual(self.queue.items()[0]["status"], "interrupted")
            self.assertEqual(self.events()[-1]["status"], "interrupted")
            with Q.locked(self.queue.state / "run.lock", blocking=False):
                pass
            with self.assertRaises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if marker.exists():
                try:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            parent.stderr.close()

    def test_child_keeps_lock_after_parent_crash(self):
        self.command_patch.stop()
        marker = self.root / "child-ready"
        release = self.root / "release-child"
        child_script = self.root / "child.py"
        child_script.write_text(
            "from pathlib import Path\nimport sys, time\n"
            "Path(sys.argv[1]).write_text(str(__import__('os').getpid()))\n"
            "while not Path(sys.argv[2]).exists(): time.sleep(0.01)\n")
        parent_code = (
            "import sys; from pathlib import Path; import ears_queue as Q\n"
            "q = Q.Queue(*sys.argv[1:4])\n"
            "with Q.locked(q.state / 'run.lock') as lock:\n"
            " q.command([sys.executable, sys.argv[4], sys.argv[5], sys.argv[6]], lock)\n")
        parent = subprocess.Popen([sys.executable, "-c", parent_code, str(self.queue.state),
                                   str(self.queue.cards), str(self.engines), str(child_script), str(marker), str(release)],
                                  cwd=Q.HERE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists(), "fake child did not start")
            parent.kill()
            parent.wait(timeout=5)
            with self.assertRaises(Q.QueueError):
                with Q.locked(self.queue.state / "run.lock", blocking=False):
                    pass
        finally:
            release.touch()
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if marker.exists():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        with Q.locked(self.queue.state / "run.lock", blocking=False):
                            break
                    except Q.QueueError:
                        time.sleep(0.01)
                else:
                    os.kill(int(marker.read_text()), signal.SIGKILL)
                    self.fail("fake child did not release the run lock")
            parent.stderr.close()


def run_tests(check):
    """Small shared-runner hook; everything stays in temporary directories."""
    class Result(unittest.TestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            check("queue: " + test._testMethodName.removeprefix("test_"), True)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            check("queue: " + test._testMethodName, False, self._exc_info_to_string(err, test))

        addError = addFailure

    unittest.defaultTestLoader.loadTestsFromTestCase(QueueTests).run(Result())


if __name__ == "__main__":
    unittest.main()
