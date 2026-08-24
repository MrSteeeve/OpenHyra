import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from external_formal_runner import (
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    build_external_formal_runner,
    inspect_runner_executable,
)

ATTESTATION = {
    "environment_sha256": "1" * 64,
    "lean_binary_sha256": "2" * 64,
    "toolchain": "leanprover/lean4:v4.26.0",
    "mathlib_revision": "3" * 40,
    "mathlib_tree_sha256": "4" * 64,
}


def _write_runner(path, response_body, *, bind_request=True):
    binding_line = (
        "response['request_sha256'] = "
        "hashlib.sha256(request_bytes).hexdigest()\n"
        if bind_request else
        ""
    )
    source = (
        f"#!{sys.executable}\n"
        "import base64, hashlib, json, sys\n"
        "request_bytes = sys.stdin.buffer.read()\n"
        "request = json.loads(request_bytes)\n"
        f"assert request['schema'] == {REQUEST_SCHEMA!r}\n"
        "assert request['isolation'] == {\n"
        "    'audit_inputs_read_only': bool(request['audit_argv']),\n"
        "    'fresh_scratch_required': True,\n"
        "    'network_allowed': False,\n"
        "    'separate_audit_process_required': bool(request['audit_argv']),\n"
        "    'trusted_audit_rematerialization_required': "
        "bool(request['audit_argv']),\n"
        "    'workspace_writable': False,\n"
        "}\n"
        "entry = request['files'][0]\n"
        "content = base64.b64decode(entry['content_base64'])\n"
        "assert hashlib.sha256(content).hexdigest() == entry['sha256']\n"
        f"response = {response_body!r}\n"
        f"{binding_line}"
        "print(json.dumps(response, sort_keys=True))\n"
    )
    path.write_text(source)
    path.chmod(0o700)


class ExternalFormalRunnerTests(unittest.TestCase):
    def test_round_trip_uses_current_strict_protocol(self):
        response = {
            "schema": RESPONSE_SCHEMA,
            "returncode": 0,
            "stdout": "compiled",
            "stderr": "",
            "audit_returncode": 0,
            "audit_stdout": "audited",
            "audit_stderr": "",
            "timed_out": False,
            "output_complete": True,
            "attestation": ATTESTATION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "runner"
            _write_runner(executable, response)
            runner, identity = build_external_formal_runner(executable)
            executable_sha256 = hashlib.sha256(
                executable.read_bytes()
            ).hexdigest()
            result = runner(SimpleNamespace(
                phase="compile_then_audit",
                argv=(
                    "lake", "env", "lean", "-o",
                    "OpenHyraCandidate.olean", "OpenHyraCandidate.lean",
                ),
                audit_argv=(
                    "lake", "env", "lean", "OpenHyraVerification.lean",
                ),
                files={
                    "OpenHyraCandidate.lean": b"theorem t : True := by trivial",
                    "OpenHyraVerification.lean": b"#check t",
                },
                cwd=".",
                timeout_s=2,
                max_output_bytes=4_096,
                network_allowed=False,
                workspace_writable=False,
            ))

        self.assertEqual(identity["protocol"], REQUEST_SCHEMA)
        self.assertNotRegex(identity["protocol"], r"(?:v1|v2)$")
        self.assertEqual(
            identity["sha256"],
            executable_sha256,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "compiled")
        self.assertEqual(result["audit_stdout"], "audited")
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["output_complete"])
        self.assertEqual(result["attestation"], ATTESTATION)

    def test_runner_identity_is_rechecked_before_each_call(self):
        response = {
            "schema": RESPONSE_SCHEMA,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "audit_returncode": 0,
            "audit_stdout": "",
            "audit_stderr": "",
            "timed_out": False,
            "output_complete": True,
            "attestation": ATTESTATION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "runner"
            _write_runner(executable, response)
            runner, _identity = build_external_formal_runner(executable)
            executable.write_text(
                f"#!{sys.executable}\nprint('changed')\n"
            )
            executable.chmod(0o700)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                runner(SimpleNamespace(
                    phase="compile_and_audit",
                    argv=("lean", "Proof.lean"),
                    files={"Proof.lean": b"#check Nat"},
                    cwd=".",
                    timeout_s=1,
                    max_output_bytes=1_024,
                    network_allowed=False,
                    workspace_writable=False,
                ))

    def test_relative_symlink_and_group_writable_runners_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            inspect_runner_executable("runner")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "runner"
            _write_runner(executable, {
                "schema": RESPONSE_SCHEMA,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "audit_returncode": 0,
                "audit_stdout": "",
                "audit_stderr": "",
                "timed_out": False,
                "output_complete": True,
                "attestation": ATTESTATION,
            })
            link = root / "runner-link"
            link.symlink_to(executable)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                inspect_runner_executable(link)
            executable.chmod(0o720)
            with self.assertRaisesRegex(ValueError, "group or other"):
                inspect_runner_executable(executable)

    def test_response_schema_is_fail_closed(self):
        response = {
            "schema": "openhyra-formal-runner-response-v1",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "audit_returncode": 0,
            "audit_stdout": "",
            "audit_stderr": "",
            "timed_out": False,
            "output_complete": True,
            "attestation": ATTESTATION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "runner"
            _write_runner(executable, response)
            runner, _identity = build_external_formal_runner(executable)
            with self.assertRaisesRegex(ValueError, "response schema"):
                runner(SimpleNamespace(
                    phase="compile_and_audit",
                    argv=("lean", "Proof.lean"),
                    files={"Proof.lean": b"#check Nat"},
                    cwd=".",
                    timeout_s=1,
                    max_output_bytes=1_024,
                    network_allowed=False,
                    workspace_writable=False,
                ))

    def test_response_missing_runtime_attestation_is_rejected(self):
        response = {
            "schema": RESPONSE_SCHEMA,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "audit_returncode": 0,
            "audit_stdout": "",
            "audit_stderr": "",
            "timed_out": False,
            "output_complete": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "runner"
            _write_runner(executable, response)
            runner, _identity = build_external_formal_runner(executable)
            with self.assertRaisesRegex(
                ValueError, "missing field.*attestation"
            ):
                runner(SimpleNamespace(
                    phase="compile_and_audit",
                    argv=("lean", "Proof.lean"),
                    files={"Proof.lean": b"#check Nat"},
                    cwd=".",
                    timeout_s=1,
                    max_output_bytes=1_024,
                    network_allowed=False,
                    workspace_writable=False,
                    expected_toolchain="leanprover/lean4:v4.26.0",
                    expected_mathlib_revision="3" * 40,
                ))

    def test_response_bound_to_a_different_request_is_rejected(self):
        response = {
            "schema": RESPONSE_SCHEMA,
            "request_sha256": "0" * 64,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "audit_returncode": 0,
            "audit_stdout": "",
            "audit_stderr": "",
            "timed_out": False,
            "output_complete": True,
            "attestation": ATTESTATION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "runner"
            _write_runner(
                executable, response, bind_request=False
            )
            runner, _identity = build_external_formal_runner(executable)
            with self.assertRaisesRegex(
                ValueError, "different request"
            ):
                runner(SimpleNamespace(
                    phase="probe_lean",
                    argv=("lean", "--version"),
                    audit_argv=(),
                    files={"OpenHyraCandidate.lean": b"#check Nat"},
                    cwd=".",
                    timeout_s=1,
                    max_output_bytes=1_024,
                    network_allowed=False,
                    workspace_writable=False,
                ))


if __name__ == "__main__":
    unittest.main()
