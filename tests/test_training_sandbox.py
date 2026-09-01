import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sandbox
from sandbox import (
    NUMERIC_THREAD_ENV,
    SANDBOX_PROFILE,
    build_training_sandbox_profile,
    run_training_sandbox,
    training_sandbox_environment,
    validate_training_sandbox_paths,
)


LEGACY_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{sandbox}"))
(allow file-write* (literal "/dev/null"))
(deny file-read* (literal "{evaluator}"))
"""


class TrainingSandboxTests(unittest.TestCase):
    def _roots(self, root):
        roots = {}
        for name in ("source", "input", "output", "tmp", "runtime"):
            path = root / name
            path.mkdir()
            roots[name] = path.resolve()
        return roots

    def _run(self, roots, command, **overrides):
        options = {
            "source_dir": roots["source"],
            "input_dir": roots["input"],
            "output_dir": roots["output"],
            "tmp_dir": roots["tmp"],
            "runtime_roots": [roots["runtime"]],
            "timeout_s": 3,
            "cpu_seconds": 3,
            "memory_bytes": 256 * 1024 * 1024,
            "file_size_bytes": 1024 * 1024,
            "externally_isolated": True,
        }
        options.update(overrides)
        with patch.object(sandbox.sys, "platform", "linux"):
            return run_training_sandbox(command, **options)

    def test_legacy_profile_is_unchanged(self):
        self.assertEqual(SANDBOX_PROFILE, LEGACY_PROFILE)

    def test_profile_is_default_deny_with_narrow_read_and_write_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            profile = build_training_sandbox_profile(
                roots["source"], roots["input"], roots["output"],
                roots["tmp"], [roots["runtime"]],
            )

        self.assertIn("(deny default)", profile)
        self.assertIn("(deny network*)", profile)
        self.assertNotIn("(allow default)", profile)
        read_clause, remainder = profile.split("(allow process-exec", 1)
        _exec_clause, write_clause = remainder.split("(allow file-write*", 1)
        for label in ("source", "input", "output", "tmp", "runtime"):
            self.assertIn(str(roots[label]), read_clause)
        self.assertIn(str(roots["output"]), write_clause)
        self.assertIn(str(roots["tmp"]), write_clause)
        self.assertNotIn(str(roots["source"]), write_clause)
        self.assertNotIn(str(roots["input"]), write_clause)
        self.assertNotIn(str(roots["runtime"]), write_clause)

    def test_paths_are_canonical_and_must_not_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = self._roots(root)
            paths = validate_training_sandbox_paths(
                roots["source"], roots["input"], roots["output"],
                roots["tmp"], [roots["runtime"]],
            )
            self.assertTrue(paths.source_dir.is_absolute())
            self.assertEqual(paths.runtime_roots, (roots["runtime"],))

            nested = roots["source"] / "nested"
            nested.mkdir()
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                validate_training_sandbox_paths(
                    roots["source"], roots["input"], nested,
                    roots["tmp"], [roots["runtime"]],
                )

    def test_symlink_and_broad_allowlist_roots_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = self._roots(root)
            linked_source = root / "linked-source"
            try:
                linked_source.symlink_to(roots["source"], target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                validate_training_sandbox_paths(
                    linked_source, roots["input"], roots["output"],
                    roots["tmp"], [roots["runtime"]],
                )
            nested_link = roots["source"] / "host-link"
            nested_link.symlink_to(Path.home(), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "entry host-link"):
                validate_training_sandbox_paths(
                    roots["source"], roots["input"], roots["output"],
                    roots["tmp"], [roots["runtime"]],
                )
            nested_link.unlink()
            hardlink_source = roots["source"] / "hardlink-source"
            hardlink_alias = roots["source"] / "hardlink-alias"
            hardlink_source.write_text("not sealed")
            os.link(hardlink_source, hardlink_alias)
            with self.assertRaisesRegex(ValueError, "exactly one hard link"):
                validate_training_sandbox_paths(
                    roots["source"], roots["input"], roots["output"],
                    roots["tmp"], [roots["runtime"]],
                )
            hardlink_alias.unlink()
            hardlink_source.unlink()
            with self.assertRaisesRegex(ValueError, "overly broad"):
                validate_training_sandbox_paths(
                    Path.home(), roots["input"], roots["output"],
                    roots["tmp"], [roots["runtime"]],
                )
            for broad in (Path("/private"), Path("/Volumes")):
                if not broad.is_dir():
                    continue
                with self.subTest(broad=str(broad)), self.assertRaisesRegex(
                    ValueError, "overly broad",
                ):
                    validate_training_sandbox_paths(
                        roots["source"], roots["input"], roots["output"],
                        roots["tmp"], [broad],
                    )

    def test_environment_is_fixed_and_does_not_inherit_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            env = training_sandbox_environment(
                roots["tmp"], [roots["runtime"]],
            )
            self.assertEqual(env["HOME"], str(roots["tmp"]))
            self.assertEqual(env["TMPDIR"], str(roots["tmp"]))
            self.assertEqual(env["PYTHONHASHSEED"], "0")
            self.assertTrue(all(env[key] == "1" for key in NUMERIC_THREAD_ENV))
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)

            destination = roots["output"] / "env.json"
            program = (
                "import json,os,sys; "
                "json.dump(dict(os.environ),open(sys.argv[1],'w'))"
            )
            with patch.dict(
                os.environ, {"AWS_SECRET_ACCESS_KEY": "must-not-leak"},
                clear=False,
            ):
                result = self._run(
                    roots,
                    [sys.executable, "-c", program, str(destination)],
                )
            self.assertEqual(result["status"], "ok", result["log_tail"])
            candidate_env = json.loads(destination.read_text())
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", candidate_env)
            self.assertEqual(candidate_env["HOME"], str(roots["tmp"]))
            self.assertEqual(candidate_env["PYTHONHASHSEED"], "0")
            self.assertEqual(int(candidate_env["OMP_NUM_THREADS"]), 1)
            self.assertGreater(result["peak_memory_bytes"], 0)
            self.assertEqual(result["output_entries"], 1)
            self.assertEqual(result["output_bytes"], destination.stat().st_size)

    def test_candidate_cannot_shadow_limit_wrapper_imports_before_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = self._roots(root)
            outside = root / "outside.txt"
            (roots["source"] / "resource.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(outside)!r}).write_text('escaped')\n"
            )
            result = self._run(
                roots,
                [sys.executable, "-c", "print('candidate ran')"],
            )
            self.assertEqual(result["status"], "ok", result["log_tail"])
            self.assertFalse(outside.exists())

    def test_non_macos_fails_closed_even_with_legacy_escape_hatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            with (
                patch.object(sandbox.sys, "platform", "linux"),
                patch.dict(
                    os.environ, {"OPENHYRA_ALLOW_UNSANDBOXED": "1"},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "fails closed"):
                    run_training_sandbox(
                        [sys.executable, "-c", "pass"],
                        source_dir=roots["source"],
                        input_dir=roots["input"],
                        output_dir=roots["output"],
                        tmp_dir=roots["tmp"],
                        runtime_roots=[roots["runtime"]],
                    )

    def test_each_run_requires_fresh_writable_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            (roots["output"] / "previous-instance.bin").write_bytes(b"old")
            with patch.object(sandbox.sys, "platform", "linux"):
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    run_training_sandbox(
                        [sys.executable, "-c", "pass"],
                        source_dir=roots["source"],
                        input_dir=roots["input"],
                        output_dir=roots["output"],
                        tmp_dir=roots["tmp"],
                        runtime_roots=[roots["runtime"]],
                        externally_isolated=True,
                    )

    def test_process_starts_in_its_own_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            destination = roots["output"] / "process.json"
            program = (
                "import json,os,sys; "
                "json.dump({'pid':os.getpid(),'pgrp':os.getpgrp()},"
                "open(sys.argv[1],'w'))"
            )
            result = self._run(
                roots,
                [sys.executable, "-c", program, str(destination)],
            )
            self.assertEqual(result["status"], "ok", result["log_tail"])
            process = json.loads(destination.read_text())
            self.assertEqual(process["pid"], process["pgrp"])

    def test_timeout_kills_the_entire_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            marker = roots["output"] / "late-child.txt"
            child = (
                "import pathlib,sys,time; time.sleep(0.6); "
                "pathlib.Path(sys.argv[1]).write_text('escaped')"
            )
            parent = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]]); "
                "time.sleep(10)"
            )
            started = time.monotonic()
            result = self._run(
                roots,
                [sys.executable, "-c", parent, str(marker)],
                timeout_s=0.2,
                cpu_seconds=2,
            )
            self.assertEqual(result["status"], "timeout")
            self.assertLess(time.monotonic() - started, 2)
            time.sleep(0.7)
            self.assertFalse(marker.exists())

    def test_normal_completion_also_kills_leftover_descendants(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            marker = roots["output"] / "late-child.txt"
            child = (
                "import pathlib,sys,time; time.sleep(0.5); "
                "pathlib.Path(sys.argv[1]).write_text('escaped')"
            )
            parent = (
                "import subprocess,sys; "
                f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[1]])"
            )
            result = self._run(
                roots,
                [sys.executable, "-c", parent, str(marker)],
            )
            self.assertEqual(result["status"], "ok", result["log_tail"])
            time.sleep(0.6)
            self.assertFalse(marker.exists())

    def test_file_size_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            destination = roots["output"] / "too-large.bin"
            program = (
                "import pathlib,sys; "
                "pathlib.Path(sys.argv[1]).write_bytes(b'x'*8192)"
            )
            result = self._run(
                roots,
                [sys.executable, "-c", program, str(destination)],
                file_size_bytes=1024,
            )
            self.assertEqual(result["status"], "crash")
            self.assertLessEqual(destination.stat().st_size, 1024)

    def test_aggregate_output_entry_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            program = (
                "import pathlib,sys,time; root=pathlib.Path(sys.argv[1]); "
                "[(root / ('f%03d' % i)).write_bytes(b'') for i in range(40)]; "
                "time.sleep(2)"
            )
            result = self._run(
                roots,
                [sys.executable, "-c", program, str(roots["output"])],
                max_output_entries=8,
            )
            self.assertEqual(result["status"], "resource_limit")
            self.assertIn("aggregate limit", result["log_tail"])

    def test_aggregate_output_byte_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self._roots(Path(temporary).resolve())
            destination = roots["output"] / "aggregate.bin"
            program = (
                "import pathlib,sys,time; "
                "pathlib.Path(sys.argv[1]).write_bytes(b'x'*8192)"
            )
            result = self._run(
                roots,
                [sys.executable, "-c", program, str(destination)],
                file_size_bytes=16384,
                max_total_output_bytes=4096,
            )
            self.assertEqual(result["status"], "resource_limit")
            self.assertIn("aggregate limit", result["log_tail"])

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS Seatbelt")
    def test_seatbelt_enforces_read_and_write_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = self._roots(root)
            allowed = roots["input"] / "allowed.txt"
            copied = roots["output"] / "copied.txt"
            frozen = roots["source"] / "frozen.txt"
            secret = root / "host-secret.txt"
            allowed.write_text("allowed")
            frozen.write_text("frozen")
            secret.write_text("secret")
            program = (
                'set -e; /bin/cat "$1" > "$2"; '
                'if /bin/cat "$3" >/dev/null 2>&1; then exit 41; fi; '
                'if (printf bad > "$4") 2>/dev/null; then exit 42; fi'
            )
            result = run_training_sandbox(
                [
                    "/bin/sh", "-c", program, "sandbox-test",
                    str(allowed), str(copied), str(secret), str(frozen),
                ],
                source_dir=roots["source"],
                input_dir=roots["input"],
                output_dir=roots["output"],
                tmp_dir=roots["tmp"],
                runtime_roots=[
                    sys.prefix, "/bin", "/usr/lib", "/System/Library",
                    "/Library/Apple",
                ],
                timeout_s=3,
                cpu_seconds=3,
                memory_bytes=256 * 1024 * 1024,
                file_size_bytes=1024 * 1024,
            )
            self.assertEqual(result["status"], "ok", result["log_tail"])
            self.assertEqual(copied.read_text(), "allowed")
            self.assertEqual(frozen.read_text(), "frozen")


if __name__ == "__main__":
    unittest.main()
