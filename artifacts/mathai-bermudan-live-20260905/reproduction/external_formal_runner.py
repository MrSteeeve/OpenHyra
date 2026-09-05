"""Client for an explicitly trusted, out-of-process formal proof runner.

The harness never executes candidate Lean code directly.  When the operator
supplies ``--formal-runner``, this module sends one bounded JSON request to that
executable.  The executable is responsible for materializing the supplied
files in a fresh isolated environment, denying network/workspace access,
applying the requested limits, and returning one strict JSON response.  For
``compile_then_audit`` it must run ``argv`` and ``audit_argv`` as separate
processes, discard candidate stdout from the audit channel, rematerialize the
trusted audit source after compilation, and expose compile outputs read-only
to the audit process.

The executable itself is treated as trusted configuration: it must be an
absolute, non-symlink, non-group/world-writable regular file.  Its SHA-256 is
frozen in run provenance and rechecked before every invocation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REQUEST_SCHEMA = "openhyra-formal-runner-request"
RESPONSE_SCHEMA = "openhyra-formal-runner-response"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RUNNER_BYTES = 4 * 1024 * 1024
TRANSPORT_GRACE_SECONDS = 10
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
TOOLCHAIN_RE = re.compile(r"[A-Za-z0-9_./:+-]{1,128}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_exact_executable(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open formal runner: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("formal runner must be a regular file")
        if info.st_size <= 0 or info.st_size > MAX_RUNNER_BYTES:
            raise ValueError(
                f"formal runner size must be in 1..{MAX_RUNNER_BYTES} bytes"
            )
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("formal runner changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("formal runner changed while it was read")
        return b"".join(chunks), info
    finally:
        os.close(descriptor)


def inspect_runner_executable(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    """Validate and hash one operator-supplied runner executable."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("--formal-runner must be an absolute path")
    try:
        link_info = os.lstat(candidate)
    except OSError as exc:
        raise ValueError(f"formal runner is unavailable: {exc}") from exc
    if stat.S_ISLNK(link_info.st_mode):
        raise ValueError("formal runner must not be a symlink")
    data, info = _read_exact_executable(candidate)
    if (
        link_info.st_dev != info.st_dev
        or link_info.st_ino != info.st_ino
        or link_info.st_size != info.st_size
    ):
        raise ValueError("formal runner changed while it was inspected")
    if not info.st_mode & 0o111:
        raise ValueError("formal runner must be executable")
    if info.st_mode & 0o022:
        raise ValueError(
            "formal runner must not be writable by group or other users"
        )
    allowed_owners = {os.getuid()}
    if os.getuid() != 0:
        allowed_owners.add(0)
    if info.st_uid not in allowed_owners:
        raise ValueError(
            "formal runner must be owned by the current user or root"
        )
    return {
        "protocol": REQUEST_SCHEMA,
        "path": str(candidate),
        "sha256": _sha256(data),
        "size_bytes": len(data),
    }


def _strict_json_object(data: bytes) -> Mapping[str, Any]:
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"formal runner response repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        decoded = data.decode("utf-8")
        payload = json.loads(decoded, object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "formal runner response must be one UTF-8 JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("formal runner response must be an object")
    required = {
        "schema",
        "request_sha256",
        "returncode",
        "stdout",
        "stderr",
        "audit_returncode",
        "audit_stdout",
        "audit_stderr",
        "timed_out",
        "output_complete",
        "attestation",
    }
    unknown = sorted(set(payload) - required)
    missing = sorted(required - set(payload))
    if unknown:
        raise ValueError(
            "formal runner response has unknown field(s): "
            + ", ".join(unknown)
        )
    if missing:
        raise ValueError(
            "formal runner response is missing field(s): "
            + ", ".join(missing)
        )
    if payload["schema"] != RESPONSE_SCHEMA:
        raise ValueError(
            f"formal runner response schema must be {RESPONSE_SCHEMA!r}"
        )
    if (
        not isinstance(payload["request_sha256"], str)
        or not SHA256_RE.fullmatch(payload["request_sha256"])
    ):
        raise ValueError(
            "formal runner response request_sha256 must be a SHA-256"
        )
    for field in ("returncode", "audit_returncode"):
        value = payload[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError(
                f"formal runner response {field} must be an integer or null"
            )
    for field in ("stdout", "stderr", "audit_stdout", "audit_stderr"):
        if not isinstance(payload[field], str):
            raise ValueError(
                f"formal runner response {field} must be text"
            )
    for field in ("timed_out", "output_complete"):
        if not isinstance(payload[field], bool):
            raise ValueError(
                f"formal runner response {field} must be a boolean"
            )
    if not payload["timed_out"] and payload["returncode"] is None:
        raise ValueError(
            "formal runner response returncode is required unless timed out"
        )
    attestation = payload["attestation"]
    if not isinstance(attestation, dict):
        raise ValueError(
            "formal runner response attestation must be an object"
        )
    attestation_fields = {
        "environment_sha256",
        "lean_binary_sha256",
        "toolchain",
        "mathlib_revision",
        "mathlib_tree_sha256",
    }
    if set(attestation) != attestation_fields:
        raise ValueError(
            "formal runner response attestation fields do not match "
            "the current protocol"
        )
    for field in (
        "environment_sha256",
        "lean_binary_sha256",
        "mathlib_tree_sha256",
    ):
        if (
            not isinstance(attestation[field], str)
            or not SHA256_RE.fullmatch(attestation[field])
        ):
            raise ValueError(
                f"formal runner attestation {field} must be a SHA-256"
            )
    if (
        not isinstance(attestation["toolchain"], str)
        or not TOOLCHAIN_RE.fullmatch(attestation["toolchain"])
        or ".." in attestation["toolchain"]
    ):
        raise ValueError(
            "formal runner attestation toolchain has invalid syntax"
        )
    if (
        not isinstance(attestation["mathlib_revision"], str)
        or not GIT_REVISION_RE.fullmatch(
            attestation["mathlib_revision"]
        )
    ):
        raise ValueError(
            "formal runner attestation mathlib_revision must be a full "
            "commit hash"
        )
    return payload


def _bounded_file_bytes(stream, *, limit: int, label: str) -> bytes:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size > limit:
        raise ValueError(f"{label} exceeds the {limit}-byte transport limit")
    stream.seek(0)
    return stream.read()


def _validate_request_file_path(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("formal runner file path must be non-empty text")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError("formal runner file path must be normalized and relative")
    return path


class ExternalFormalRunner:
    """Callable adapter implementing the formalization module's runner API."""

    def __init__(self, executable: str | os.PathLike[str]):
        self.identity = dict(inspect_runner_executable(executable))
        self.executable = self.identity["path"]
        self._executable_bytes, _info = _read_exact_executable(
            Path(self.executable)
        )
        if _sha256(self._executable_bytes) != self.identity["sha256"]:
            raise ValueError(
                "formal runner changed while its identity was frozen"
            )

    def _assert_unchanged(self) -> None:
        current = inspect_runner_executable(self.executable)
        if current != self.identity:
            raise RuntimeError(
                "formal runner identity changed after run initialization"
            )

    def __call__(self, request) -> Mapping[str, Any]:
        self._assert_unchanged()
        files = []
        total_file_bytes = 0
        for path, raw in sorted(request.files.items()):
            normalized_path = _validate_request_file_path(path)
            if not isinstance(raw, bytes):
                raise ValueError(
                    f"formal runner file {normalized_path!r} must be bytes"
                )
            total_file_bytes += len(raw)
            files.append({
                "path": normalized_path,
                "size_bytes": len(raw),
                "sha256": _sha256(raw),
                "content_base64": base64.b64encode(raw).decode("ascii"),
            })
        payload = {
            "schema": REQUEST_SCHEMA,
            "phase": request.phase,
            "argv": list(request.argv),
            "audit_argv": list(
                getattr(request, "audit_argv", ()) or ()
            ),
            "cwd": request.cwd,
            "files": files,
            "limits": {
                "timeout_s": request.timeout_s,
                "max_output_bytes": request.max_output_bytes,
                "total_file_bytes": total_file_bytes,
            },
            "isolation": {
                "network_allowed": request.network_allowed,
                "workspace_writable": request.workspace_writable,
                "fresh_scratch_required": True,
                "separate_audit_process_required": bool(
                    getattr(request, "audit_argv", ())
                ),
                "trusted_audit_rematerialization_required": bool(
                    getattr(request, "audit_argv", ())
                ),
                "audit_inputs_read_only": bool(
                    getattr(request, "audit_argv", ())
                ),
            },
            "expected_environment": {
                "toolchain": getattr(
                    request, "expected_toolchain", None
                ),
                "mathlib_revision": getattr(
                    request, "expected_mathlib_revision", None
                ),
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError(
                f"formal runner request exceeds {MAX_REQUEST_BYTES} bytes"
            )

        response_limit = max(
            64 * 1024,
            2 * int(request.max_output_bytes) + 64 * 1024,
        )
        with (
            tempfile.TemporaryDirectory(prefix="openhyra-formal-client-") as home,
            tempfile.TemporaryFile() as stdout_stream,
            tempfile.TemporaryFile() as stderr_stream,
        ):
            sealed_executable = Path(home) / "formal-runner"
            sealed_executable.write_bytes(self._executable_bytes)
            sealed_executable.chmod(0o500)
            environment = {
                "HOME": home,
                "TMPDIR": home,
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "LANG": "C",
                "LC_ALL": "C",
            }
            process = subprocess.Popen(
                [str(sealed_executable)],
                stdin=subprocess.PIPE,
                stdout=stdout_stream,
                stderr=stderr_stream,
                env=environment,
                start_new_session=True,
                close_fds=True,
            )
            try:
                process.communicate(
                    encoded,
                    timeout=request.timeout_s + TRANSPORT_GRACE_SECONDS,
                )
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                return {
                    "returncode": None,
                    "stdout": "",
                    "stderr": "external formal runner transport timed out",
                    "audit_returncode": None,
                    "audit_stdout": "",
                    "audit_stderr": "",
                    "timed_out": True,
                    "output_complete": False,
                    "attestation": None,
                }
            stderr = _bounded_file_bytes(
                stderr_stream,
                limit=response_limit,
                label="formal runner stderr",
            ).decode("utf-8", errors="replace")
            if process.returncode != 0:
                raise RuntimeError(
                    "formal runner process failed: " + stderr[:4_000]
                )
            response_bytes = _bounded_file_bytes(
                stdout_stream,
                limit=response_limit,
                label="formal runner response",
            )

        response = _strict_json_object(response_bytes)
        request_sha256 = _sha256(encoded)
        if response["request_sha256"] != request_sha256:
            raise ValueError(
                "formal runner response is bound to a different request"
            )
        return {
            "returncode": response["returncode"],
            "stdout": response["stdout"],
            "stderr": response["stderr"],
            "audit_returncode": response["audit_returncode"],
            "audit_stdout": response["audit_stdout"],
            "audit_stderr": response["audit_stderr"],
            "timed_out": response["timed_out"],
            "output_complete": response["output_complete"],
            "attestation": response["attestation"],
        }


def build_external_formal_runner(
    executable: str | os.PathLike[str],
) -> tuple[ExternalFormalRunner, Mapping[str, Any]]:
    runner = ExternalFormalRunner(executable)
    return runner, dict(runner.identity)


__all__ = [
    "ExternalFormalRunner",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "build_external_formal_runner",
    "inspect_runner_executable",
]
