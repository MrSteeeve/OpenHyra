"""Fail-closed validation primitives for optional Lean 4 artifacts.

This module deliberately does not execute subprocesses or write files.  It
validates and seals the bytes of a submitted artifact, then asks a
caller-supplied *trusted, isolated* runner to probe and invoke Lean.  A runner
must materialize ``RunnerRequest.files`` in a fresh scratch directory, deny
network access, impose the requested limits, and discard the directory after
each request.

The candidate controls neither the command nor the theorem names used for the
final audit.  A manifest is accepted only when its raw SHA-256 is supplied by a
trusted caller and every listed file matches its declared size and SHA-256.
Successful compilation alone is insufficient: a generated audit module checks
the expected declarations and, by default, rejects non-standard axioms.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple


SCHEMA = "openhyra-lean4-artifact"
TARGET = "lean4"
MANIFEST_NAME = "formalization.json"
RESERVED_AUDIT_FILE = "OpenHyraVerification.lean"
RESERVED_CANDIDATE_FILE = "OpenHyraCandidate.lean"
TRUSTED_SPEC_FILE = "OpenHyraSumDiff/Spec.lean"
REQUEST_SCHEMA = "openhyra-lean4-request"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
IDENTIFIER_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*"
)
MODULE_PART_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
TOOLCHAIN_RE = re.compile(r"[A-Za-z0-9_./:+-]{1,128}")
GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
LEAN_VERSION_RE = re.compile(
    r"\bversion\s+([0-9]+\.[0-9]+\.[0-9]+(?:-(?:rc|beta)[0-9]+)?)\b",
    re.IGNORECASE,
)
PINNED_VERSION_RE = re.compile(
    r":v([0-9]+\.[0-9]+\.[0-9]+(?:-(?:rc|beta)[0-9]+)?)$",
    re.IGNORECASE,
)
FORBIDDEN_SOURCE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_'])(sorry|admit|sorryAx)(?![A-Za-z0-9_'])"
)
FORBIDDEN_INLINE_META_RE = re.compile(
    r"(?<![A-Za-z0-9_'])("
    r"axiom|opaque|theorem|def|abbrev|instance|"
    r"initialize|builtin_initialize|elab|macro|syntax|"
    r"namespace|section|end|import|attribute|set_option|"
    r"unsafe|extern|run_tac"
    r")(?![A-Za-z0-9_'])"
)
FORBIDDEN_DIAGNOSTIC_RE = re.compile(
    r"(declaration\s+uses\s+['\"]?sorry|"
    r"(?<![A-Za-z0-9_'])sorryAx(?![A-Za-z0-9_']))",
    re.IGNORECASE,
)

DEFAULT_ALLOWED_AXIOMS = (
    "Classical.choice",
    "Quot.sound",
    "propext",
)

FORMAL_CLAIM_TYPES = {
    "universal_upper_bound": "UniversalUpperBoundAt",
    "approximating_family": "ApproximatingAt",
    "supremum_eq": "SupremumAt",
    "nonattainment": "NonattainedAt",
}
MAX_PROOFS = 16
MAX_PROOF_TERM_CHARS = 400_000


@dataclass(frozen=True)
class VerificationLimits:
    """Trusted resource limits for artifact ingestion and runner calls."""

    max_manifest_bytes: int = 64 * 1024
    max_file_bytes: int = 512 * 1024
    max_total_bytes: int = 4 * 1024 * 1024
    max_files: int = 64
    max_path_chars: int = 240
    max_path_depth: int = 12
    max_output_bytes: int = 128 * 1024
    probe_timeout_s: int = 20
    compile_timeout_s: int = 300
    audit_timeout_s: int = 120

    def validate(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"limits.{name} must be a positive integer")


@dataclass(frozen=True)
class RunnerRequest:
    """One command to run inside caller-owned isolation.

    ``files`` contains only hash-validated bytes.  The runner must not expose
    the source workspace; it should copy these bytes to an empty scratch root
    and run ``argv`` with ``cwd`` relative to that root.  When ``audit_argv``
    is non-empty, it must be executed in a separate process after compilation;
    the trusted audit source must be rematerialized, compile outputs must be
    read-only, and candidate stdout must never enter ``audit_stdout``.
    """

    phase: str
    argv: Tuple[str, ...]
    files: Mapping[str, bytes]
    cwd: str
    timeout_s: int
    max_output_bytes: int
    audit_argv: Tuple[str, ...] = ()
    network_allowed: bool = False
    workspace_writable: bool = False
    expected_toolchain: Optional[str] = None
    expected_mathlib_revision: Optional[str] = None


@dataclass(frozen=True)
class RunnerResult:
    """Normalized result returned by an isolated runner."""

    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    audit_returncode: Optional[int] = None
    audit_stdout: str = ""
    audit_stderr: str = ""
    timed_out: bool = False
    output_complete: Optional[bool] = None
    attestation: Optional[Mapping[str, str]] = None


@dataclass(frozen=True)
class ValidatedArtifact:
    """Hash-validated artifact bytes and normalized manifest metadata."""

    driver: str
    toolchain: str
    entrypoint: str
    expected_theorems: Tuple[str, ...]
    files: Mapping[str, bytes]
    manifest_sha256: str
    artifact_sha256: str


class ArtifactRejected(ValueError):
    """An untrusted artifact failed validation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


Runner = Callable[[RunnerRequest], Any]


def _strict_object(
    value: Any,
    *,
    required: Iterable[str],
    allowed: Iterable[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactRejected("invalid_schema", f"{path} must be an object")
    required_set = set(required)
    allowed_set = set(allowed)
    unknown = sorted(set(value) - allowed_set)
    missing = sorted(required_set - set(value))
    if unknown:
        raise ArtifactRejected(
            "invalid_schema",
            f"{path} has unknown field(s): {', '.join(unknown)}",
        )
    if missing:
        raise ArtifactRejected(
            "invalid_schema",
            f"{path} is missing field(s): {', '.join(missing)}",
        )
    return value


def _strict_positive_int(value: Any, *, path: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactRejected("invalid_schema", f"{path} must be an integer")
    if value <= 0 or value > maximum:
        raise ArtifactRejected(
            "invalid_schema", f"{path} must be in the range 1..{maximum}",
        )
    return value


def _validate_sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ArtifactRejected(
            "invalid_hash", f"{path} must be a lowercase SHA-256 digest",
        )
    return value


def _validate_relative_path(
    value: Any,
    *,
    path: str,
    limits: VerificationLimits,
) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactRejected("invalid_path", f"{path} must be a path string")
    if len(value) > limits.max_path_chars:
        raise ArtifactRejected("invalid_path", f"{path} is too long")
    if "\x00" in value or "\\" in value:
        raise ArtifactRejected(
            "invalid_path", f"{path} contains a forbidden character",
        )
    pure = PurePosixPath(value)
    parts = pure.parts
    if (
        pure.is_absolute()
        or not parts
        or len(parts) > limits.max_path_depth
        or any(part in {"", ".", ".."} for part in parts)
        or any(part.startswith(".") for part in parts)
    ):
        raise ArtifactRejected(
            "invalid_path", f"{path} must be a normalized relative path",
        )
    normalized = pure.as_posix()
    if normalized != value:
        raise ArtifactRejected(
            "invalid_path", f"{path} must be normalized POSIX syntax",
        )
    if normalized in {RESERVED_AUDIT_FILE, RESERVED_CANDIDATE_FILE}:
        raise ArtifactRejected(
            "reserved_path", f"{path} is reserved by the trusted verifier",
        )
    return normalized


def _json_without_duplicate_keys(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactRejected(
            "invalid_manifest", "formalization.json must be valid UTF-8",
        ) from exc

    def pairs_hook(pairs: Sequence[Tuple[str, Any]]) -> Mapping[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactRejected(
                    "invalid_manifest",
                    f"formalization.json contains duplicate key {key!r}",
                )
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=pairs_hook)
    except ArtifactRejected:
        raise
    except (ValueError, RecursionError) as exc:
        raise ArtifactRejected(
            "invalid_manifest", "formalization.json is not valid bounded JSON",
        ) from exc


def _open_root(root: Path) -> int:
    try:
        info = os.lstat(root)
    except (FileNotFoundError, OSError) as exc:
        raise ArtifactRejected(
            "artifact_missing", "formalization artifact root is unavailable",
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ArtifactRejected(
            "unsafe_artifact_root",
            "formalization artifact root must be a real directory",
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(root, flags)
    except OSError as exc:
        raise ArtifactRejected(
            "unsafe_artifact_root",
            f"could not safely open formalization artifact root: {exc}",
        ) from exc


def _read_relative_regular_file(
    root_fd: int,
    relative: str,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read a bounded regular file via no-follow directory descriptors."""

    parts = PurePosixPath(relative).parts
    current_fd = os.dup(root_fd)
    try:
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        for component in parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                raise ArtifactRejected(
                    "unsafe_path",
                    f"could not safely traverse {label}: {exc}",
                ) from exc
            os.close(current_fd)
            current_fd = next_fd

        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NONBLOCK", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
        except OSError as exc:
            raise ArtifactRejected(
                "unsafe_file", f"could not safely open {label}: {exc}",
            ) from exc
        try:
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactRejected(
                    "unsafe_file", f"{label} must be a regular file",
                )
            if info.st_nlink != 1:
                raise ArtifactRejected(
                    "unsafe_file", f"{label} must have exactly one hard link",
                )
            if info.st_size > max_bytes:
                raise ArtifactRejected(
                    "file_too_large",
                    f"{label} exceeds the {max_bytes}-byte limit",
                )
            chunks = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(file_fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise ArtifactRejected(
                    "file_too_large",
                    f"{label} exceeds the {max_bytes}-byte limit",
                )
            return data
        finally:
            os.close(file_fd)
    finally:
        os.close(current_fd)


def _validate_theorem_names(
    names: Any,
    *,
    path: str,
    require_nonempty: bool = True,
) -> Tuple[str, ...]:
    if not isinstance(names, (list, tuple)):
        raise ArtifactRejected("invalid_schema", f"{path} must be a list")
    if require_nonempty and not names:
        raise ArtifactRejected(
            "missing_theorem", f"{path} must contain at least one theorem",
        )
    if len(names) > 32:
        raise ArtifactRejected(
            "invalid_schema", f"{path} exceeds 32 theorem names",
        )
    result = []
    for index, name in enumerate(names):
        if (
            not isinstance(name, str)
            or len(name) > 160
            or not IDENTIFIER_RE.fullmatch(name)
        ):
            raise ArtifactRejected(
                "invalid_theorem",
                f"{path}[{index}] is not a safe Lean identifier",
            )
        result.append(name)
    if len(result) != len(set(result)):
        raise ArtifactRejected(
            "invalid_theorem", f"{path} contains duplicate theorem names",
        )
    return tuple(result)


def _entrypoint_module(entrypoint: str) -> str:
    if not entrypoint.endswith(".lean"):
        raise ArtifactRejected(
            "invalid_entrypoint", "entrypoint must have a .lean suffix",
        )
    parts = PurePosixPath(entrypoint[:-5]).parts
    if not parts or any(not MODULE_PART_RE.fullmatch(part) for part in parts):
        raise ArtifactRejected(
            "invalid_entrypoint",
            "entrypoint must map to a safe Lean module name",
        )
    return ".".join(parts)


def _strip_lean_comments_and_strings(source: str) -> str:
    """Replace Lean comments and strings while preserving token boundaries."""

    output = []
    index = 0
    block_depth = 0
    in_line_comment = False
    in_string = False
    escaped = False
    while index < len(source):
        current = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""

        if in_line_comment:
            if current == "\n":
                in_line_comment = False
                output.append("\n")
            else:
                output.append(" ")
            index += 1
            continue

        if block_depth:
            if current == "/" and following == "-":
                block_depth += 1
                output.extend((" ", " "))
                index += 2
            elif current == "-" and following == "/":
                block_depth -= 1
                output.extend((" ", " "))
                index += 2
            else:
                output.append("\n" if current == "\n" else " ")
                index += 1
            continue

        if in_string:
            output.append("\n" if current == "\n" else " ")
            if escaped:
                escaped = False
            elif current == "\\":
                escaped = True
            elif current == '"':
                in_string = False
            index += 1
            continue

        if current == "-" and following == "-":
            in_line_comment = True
            output.extend((" ", " "))
            index += 2
        elif current == "/" and following == "-":
            block_depth = 1
            output.extend((" ", " "))
            index += 2
        elif current == '"':
            in_string = True
            output.append(" ")
            index += 1
        else:
            output.append(current)
            index += 1
    return "".join(output)


def _validate_inline_term_boundary(term: str, *, path: str) -> None:
    if any(marker in term for marker in ("--", "/-", "-/")):
        raise ArtifactRejected(
            "forbidden_inline_syntax",
            f"{path}.term must not contain Lean comments",
        )
    stripped = _strip_lean_comments_and_strings(term)
    match = FORBIDDEN_INLINE_META_RE.search(stripped)
    if match:
        raise ArtifactRejected(
            "forbidden_inline_syntax",
            f"{path}.term contains forbidden token {match.group(1)!r}",
        )
    pairs = {")": "(", "]": "[", "}": "{"}
    opening = set(pairs.values())
    stack = []
    for character in stripped:
        if character in opening:
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                raise ArtifactRejected(
                    "forbidden_inline_syntax",
                    f"{path}.term crosses its trusted wrapper boundary",
                )
    if stack:
        raise ArtifactRejected(
            "forbidden_inline_syntax",
            f"{path}.term has unbalanced delimiters",
        )


def _check_forbidden_source(files: Mapping[str, bytes]) -> None:
    for name, data in files.items():
        if not name.endswith(".lean"):
            continue
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactRejected(
                "invalid_source", f"{name} must be valid UTF-8",
            ) from exc
        match = FORBIDDEN_SOURCE_TOKEN_RE.search(
            _strip_lean_comments_and_strings(source),
        )
        if match:
            raise ArtifactRejected(
                "forbidden_proof_hole",
                f"{name} contains forbidden token {match.group(1)!r}",
            )


def _canonical_artifact_hash(
    manifest_sha256: str,
    entries: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "manifest_sha256": manifest_sha256,
        "files": sorted(
            (
                {
                    "path": entry["path"],
                    "size_bytes": entry["size_bytes"],
                    "sha256": entry["sha256"],
                }
                for entry in entries
            ),
            key=lambda entry: entry["path"],
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_lean_artifact(
    artifact_root: os.PathLike[str] | str,
    *,
    expected_manifest_sha256: str,
    expected_theorems: Sequence[str],
    limits: VerificationLimits = VerificationLimits(),
    expected_toolchain: Optional[str] = None,
) -> ValidatedArtifact:
    """Read and hash-validate one submitted artifact without executing it.

    ``expected_manifest_sha256`` and ``expected_theorems`` are trusted inputs,
    not candidate-selected values.  The raw manifest digest binds all file
    digests; the file digests then bind every byte handed to the runner.
    """

    limits.validate()
    trusted_manifest_hash = _validate_sha256(
        expected_manifest_sha256, path="expected_manifest_sha256",
    )
    trusted_theorems = _validate_theorem_names(
        tuple(expected_theorems), path="expected_theorems",
    )
    if expected_toolchain is not None:
        if (
            not isinstance(expected_toolchain, str)
            or not TOOLCHAIN_RE.fullmatch(expected_toolchain)
            or ".." in expected_toolchain
        ):
            raise ValueError("expected_toolchain has invalid syntax")

    root = Path(artifact_root)
    root_fd = _open_root(root)
    try:
        manifest_data = _read_relative_regular_file(
            root_fd,
            MANIFEST_NAME,
            max_bytes=limits.max_manifest_bytes,
            label=MANIFEST_NAME,
        )
        actual_manifest_hash = hashlib.sha256(manifest_data).hexdigest()
        if actual_manifest_hash != trusted_manifest_hash:
            raise ArtifactRejected(
                "manifest_hash_mismatch",
                "formalization.json does not match its trusted SHA-256",
            )
        manifest = _json_without_duplicate_keys(manifest_data)
        _strict_object(
            manifest,
            required={
                "schema", "target", "driver", "toolchain", "entrypoint",
                "expected_theorems", "files",
            },
            allowed={
                "schema", "target", "driver", "toolchain", "entrypoint",
                "expected_theorems", "files",
            },
            path="manifest",
        )
        if manifest["schema"] != SCHEMA:
            raise ArtifactRejected(
                "unsupported_schema", f"manifest.schema must be {SCHEMA!r}",
            )
        if manifest["target"] != TARGET:
            raise ArtifactRejected(
                "unsupported_target", "manifest.target must be 'lean4'",
            )
        driver = manifest["driver"]
        if driver not in {"lean", "lake"}:
            raise ArtifactRejected(
                "unsupported_driver",
                "manifest.driver must be either 'lean' or 'lake'",
            )
        toolchain = manifest["toolchain"]
        if (
            not isinstance(toolchain, str)
            or not TOOLCHAIN_RE.fullmatch(toolchain)
            or ".." in toolchain
        ):
            raise ArtifactRejected(
                "invalid_toolchain", "manifest.toolchain has invalid syntax",
            )
        if expected_toolchain is not None and toolchain != expected_toolchain:
            raise ArtifactRejected(
                "toolchain_mismatch",
                "manifest.toolchain does not match the trusted toolchain",
            )
        entrypoint = _validate_relative_path(
            manifest["entrypoint"],
            path="manifest.entrypoint",
            limits=limits,
        )
        _entrypoint_module(entrypoint)
        manifest_theorems = _validate_theorem_names(
            manifest["expected_theorems"],
            path="manifest.expected_theorems",
        )
        if manifest_theorems != trusted_theorems:
            raise ArtifactRejected(
                "theorem_mismatch",
                "manifest.expected_theorems does not match the trusted target",
            )

        entries = manifest["files"]
        if not isinstance(entries, list) or not entries:
            raise ArtifactRejected(
                "invalid_schema", "manifest.files must be a non-empty list",
            )
        if len(entries) > limits.max_files:
            raise ArtifactRejected(
                "too_many_files",
                f"manifest.files exceeds the {limits.max_files}-file limit",
            )

        normalized_entries = []
        seen_paths = set()
        declared_total = 0
        allowed_special = {
            "lakefile.lean",
            "lakefile.toml",
            "lean-toolchain",
            "lake-manifest.json",
        }
        for index, raw_entry in enumerate(entries):
            path = f"manifest.files[{index}]"
            _strict_object(
                raw_entry,
                required={"path", "size_bytes", "sha256"},
                allowed={"path", "size_bytes", "sha256"},
                path=path,
            )
            name = _validate_relative_path(
                raw_entry["path"], path=f"{path}.path", limits=limits,
            )
            if name in seen_paths:
                raise ArtifactRejected(
                    "duplicate_path", f"manifest.files repeats {name!r}",
                )
            seen_paths.add(name)
            if not name.endswith(".lean") and name not in allowed_special:
                raise ArtifactRejected(
                    "unsupported_file",
                    f"{name!r} is not an allowlisted Lean artifact file",
                )
            if name.endswith(".lean"):
                lean_parts = list(PurePosixPath(name).parts)
                lean_parts[-1] = lean_parts[-1][:-5]
                if any(
                    not MODULE_PART_RE.fullmatch(part)
                    for part in lean_parts
                ):
                    raise ArtifactRejected(
                        "invalid_path",
                        f"{name!r} does not map to a safe Lean module path",
                    )
            size = _strict_positive_int(
                raw_entry["size_bytes"],
                path=f"{path}.size_bytes",
                maximum=limits.max_file_bytes,
            )
            digest = _validate_sha256(
                raw_entry["sha256"], path=f"{path}.sha256",
            )
            declared_total += size
            if declared_total > limits.max_total_bytes:
                raise ArtifactRejected(
                    "artifact_too_large",
                    f"artifact exceeds the {limits.max_total_bytes}-byte limit",
                )
            normalized_entries.append({
                "path": name,
                "size_bytes": size,
                "sha256": digest,
            })

        if entrypoint not in seen_paths:
            raise ArtifactRejected(
                "missing_entrypoint",
                "manifest.entrypoint is not present in manifest.files",
            )
        if "lean-toolchain" not in seen_paths:
            raise ArtifactRejected(
                "missing_toolchain_file",
                "manifest.files must include lean-toolchain",
            )
        if driver == "lake" and not (
            {"lakefile.lean", "lakefile.toml"} & seen_paths
        ):
            raise ArtifactRejected(
                "missing_lakefile",
                "a lake artifact must include lakefile.lean or lakefile.toml",
            )

        files = {}
        actual_total = 0
        for entry in normalized_entries:
            data = _read_relative_regular_file(
                root_fd,
                entry["path"],
                max_bytes=limits.max_file_bytes,
                label=entry["path"],
            )
            if len(data) != entry["size_bytes"]:
                raise ArtifactRejected(
                    "size_mismatch",
                    f"{entry['path']} does not match its declared size",
                )
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ArtifactRejected(
                    "file_hash_mismatch",
                    f"{entry['path']} does not match its declared SHA-256",
                )
            actual_total += len(data)
            if actual_total > limits.max_total_bytes:
                raise ArtifactRejected(
                    "artifact_too_large",
                    f"artifact exceeds the {limits.max_total_bytes}-byte limit",
                )
            files[entry["path"]] = data
    finally:
        os.close(root_fd)

    try:
        toolchain_file = files["lean-toolchain"].decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ArtifactRejected(
            "invalid_toolchain", "lean-toolchain must contain ASCII text",
        ) from exc
    if toolchain_file != toolchain:
        raise ArtifactRejected(
            "toolchain_mismatch",
            "lean-toolchain content does not match manifest.toolchain",
        )
    _check_forbidden_source(files)

    return ValidatedArtifact(
        driver=driver,
        toolchain=toolchain,
        entrypoint=entrypoint,
        expected_theorems=manifest_theorems,
        files=MappingProxyType(dict(files)),
        manifest_sha256=actual_manifest_hash,
        artifact_sha256=_canonical_artifact_hash(
            actual_manifest_hash, normalized_entries,
        ),
    )


def _bounded_runner_text(
    value: Any,
    *,
    field: str,
    max_bytes: int,
) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        raw = value
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
        raw = text.encode("utf-8")
    else:
        raise ValueError(f"runner {field} must be text or bytes")
    if len(raw) > max_bytes:
        raise ValueError(f"runner {field} exceeds the output limit")
    return text


def _normalize_runtime_attestation(
    raw: Any,
) -> Optional[Mapping[str, str]]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("runner attestation must be an object")
    required = {
        "environment_sha256",
        "lean_binary_sha256",
        "toolchain",
        "mathlib_revision",
        "mathlib_tree_sha256",
    }
    unknown = sorted(set(raw) - required)
    missing = sorted(required - set(raw))
    if unknown:
        raise ValueError(
            "runner attestation has unknown field(s): "
            + ", ".join(str(item) for item in unknown)
        )
    if missing:
        raise ValueError(
            "runner attestation is missing field(s): "
            + ", ".join(missing)
        )
    normalized = {}
    for field in (
        "environment_sha256",
        "lean_binary_sha256",
        "mathlib_tree_sha256",
    ):
        value = raw[field]
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise ValueError(
                f"runner attestation.{field} must be a lowercase SHA-256"
            )
        normalized[field] = value
    toolchain = raw["toolchain"]
    if (
        not isinstance(toolchain, str)
        or not TOOLCHAIN_RE.fullmatch(toolchain)
        or ".." in toolchain
    ):
        raise ValueError(
            "runner attestation.toolchain has invalid syntax"
        )
    revision = raw["mathlib_revision"]
    if (
        not isinstance(revision, str)
        or not GIT_REVISION_RE.fullmatch(revision)
    ):
        raise ValueError(
            "runner attestation.mathlib_revision must be a full commit hash"
        )
    normalized["toolchain"] = toolchain
    normalized["mathlib_revision"] = revision
    return MappingProxyType(normalized)


def _normalize_runner_result(
    raw: Any,
    *,
    max_output_bytes: int,
) -> RunnerResult:
    if isinstance(raw, RunnerResult):
        returncode = raw.returncode
        stdout = raw.stdout
        stderr = raw.stderr
        audit_returncode = raw.audit_returncode
        audit_stdout = raw.audit_stdout
        audit_stderr = raw.audit_stderr
        timed_out = raw.timed_out
        output_complete = raw.output_complete
        attestation = raw.attestation
    elif isinstance(raw, Mapping):
        allowed = {
            "returncode", "exit_code", "stdout", "stderr", "timed_out",
            "audit_returncode", "audit_stdout", "audit_stderr",
            "output_complete", "attestation",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "runner result has unknown field(s): "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        if "returncode" in raw and "exit_code" in raw:
            raise ValueError(
                "runner result must not contain both returncode and exit_code",
            )
        returncode = raw.get("returncode", raw.get("exit_code"))
        stdout = raw.get("stdout", "")
        stderr = raw.get("stderr", "")
        audit_returncode = raw.get("audit_returncode")
        audit_stdout = raw.get("audit_stdout", "")
        audit_stderr = raw.get("audit_stderr", "")
        timed_out = raw.get("timed_out", False)
        output_complete = raw.get("output_complete")
        attestation = raw.get("attestation")
    else:
        returncode = getattr(raw, "returncode", None)
        stdout = getattr(raw, "stdout", "")
        stderr = getattr(raw, "stderr", "")
        audit_returncode = getattr(raw, "audit_returncode", None)
        audit_stdout = getattr(raw, "audit_stdout", "")
        audit_stderr = getattr(raw, "audit_stderr", "")
        timed_out = getattr(raw, "timed_out", False)
        output_complete = getattr(raw, "output_complete", None)
        attestation = getattr(raw, "attestation", None)

    if not isinstance(timed_out, bool):
        raise ValueError("runner timed_out must be a boolean")
    if output_complete is not None and not isinstance(
        output_complete, bool
    ):
        raise ValueError("runner output_complete must be a boolean or null")
    if returncode is not None and (
        isinstance(returncode, bool) or not isinstance(returncode, int)
    ):
        raise ValueError("runner returncode must be an integer or null")
    if audit_returncode is not None and (
        isinstance(audit_returncode, bool)
        or not isinstance(audit_returncode, int)
    ):
        raise ValueError(
            "runner audit_returncode must be an integer or null"
        )
    if not timed_out and returncode is None:
        raise ValueError("runner result is missing a return code")

    stdout_text = _bounded_runner_text(
        stdout, field="stdout", max_bytes=max_output_bytes,
    )
    stderr_text = _bounded_runner_text(
        stderr, field="stderr", max_bytes=max_output_bytes,
    )
    audit_stdout_text = _bounded_runner_text(
        audit_stdout, field="audit_stdout", max_bytes=max_output_bytes,
    )
    audit_stderr_text = _bounded_runner_text(
        audit_stderr, field="audit_stderr", max_bytes=max_output_bytes,
    )
    if (
        len(stdout_text.encode("utf-8"))
        + len(stderr_text.encode("utf-8"))
        + len(audit_stdout_text.encode("utf-8"))
        + len(audit_stderr_text.encode("utf-8"))
        > max_output_bytes
    ):
        raise ValueError("combined runner output exceeds the output limit")
    return RunnerResult(
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        audit_returncode=audit_returncode,
        audit_stdout=audit_stdout_text,
        audit_stderr=audit_stderr_text,
        timed_out=timed_out,
        output_complete=output_complete,
        attestation=_normalize_runtime_attestation(attestation),
    )


def _diagnostic_text(result: RunnerResult, limit: int = 4_000) -> str:
    combined = "\n".join(
        part for part in (result.stdout.strip(), result.stderr.strip()) if part
    )
    if len(combined) <= limit:
        return combined
    return combined[:limit] + " ...[truncated]"


def _run(
    runner: Runner,
    request: RunnerRequest,
) -> RunnerResult:
    raw = runner(request)
    return _normalize_runner_result(
        raw, max_output_bytes=request.max_output_bytes,
    )


def _base_result(status: str, reason: str, **extra: Any) -> Mapping[str, Any]:
    return {
        "target": TARGET,
        "status": status,
        "reason": reason,
        **extra,
    }


def _runner_failure(
    phase: str,
    reason: str,
    detail: str,
    artifact: ValidatedArtifact,
    *,
    status: str,
    toolchain: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    return _base_result(
        status,
        reason,
        manifest_sha256=artifact.manifest_sha256,
        artifact_sha256=artifact.artifact_sha256,
        expected_theorems=list(artifact.expected_theorems),
        toolchain=dict(toolchain or {}),
        failure={"phase": phase, "detail": detail[:4_000]},
    )


def _make_audit_source(
    artifact: ValidatedArtifact,
    *,
    audit_axioms: bool,
) -> bytes:
    module_name = _entrypoint_module(artifact.entrypoint)
    lines = [f"import {module_name}", ""]
    for theorem in artifact.expected_theorems:
        lines.append(f"#check {theorem}")
        if audit_axioms:
            lines.append(f"#print axioms {theorem}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _parse_axioms(
    output: str,
    theorem_names: Sequence[str],
) -> Mapping[str, Tuple[str, ...]]:
    parsed = {}
    for theorem in theorem_names:
        escaped = re.escape(theorem)
        with_axioms = re.findall(
            rf"(?m)^\s*['\"]?{escaped}['\"]?\s+"
            rf"depends on axioms:\s*\[([^\]]*)\]\s*$",
            output,
        )
        without_axioms = re.findall(
            rf"(?m)^\s*['\"]?{escaped}['\"]?\s+"
            rf"does not depend on any axioms\s*$",
            output,
        )
        if len(with_axioms) + len(without_axioms) != 1:
            raise ValueError(
                f"expected exactly one axiom audit record for {theorem}"
            )
        if with_axioms:
            names = tuple(
                item.strip()
                for item in with_axioms[0].split(",")
                if item.strip()
            )
            if any(not IDENTIFIER_RE.fullmatch(name) for name in names):
                raise ValueError(
                    f"could not safely parse axiom names for {theorem}",
                )
            parsed[theorem] = names
        else:
            parsed[theorem] = ()
    return parsed


def verify_lean_artifact(
    artifact_root: Optional[os.PathLike[str] | str],
    *,
    expected_manifest_sha256: Optional[str],
    expected_theorems: Sequence[str],
    runner: Runner,
    limits: VerificationLimits = VerificationLimits(),
    expected_toolchain: Optional[str] = None,
    allowed_axioms: Optional[Iterable[str]] = DEFAULT_ALLOWED_AXIOMS,
    include_axiom_audit_text: bool = False,
) -> Mapping[str, Any]:
    """Validate and verify an optional Lean 4 artifact.

    Statuses have intentionally narrow meanings:

    * ``not_submitted``: ``artifact_root`` is ``None``; the runner is untouched.
    * ``unavailable``: a valid artifact exists, but the isolated runner or
      requested Lean/Lake toolchain could not be used.
    * ``rejected``: artifact validation, compilation, theorem checking, timeout,
      forbidden-hole, or axiom policy failed.
    * ``verified``: hashes, compilation, expected declarations, and the enabled
      axiom policy all passed inside the caller's isolation boundary.

    Set ``allowed_axioms=None`` to skip ``#print axioms``.  This weakens the
    result and is recorded as ``not_requested``.  The default permits only
    Lean's standard classical/propositional axioms and still rejects
    ``sorryAx`` diagnostics.
    """

    if artifact_root is None:
        return _base_result("not_submitted", "no_artifact")
    if expected_manifest_sha256 is None:
        return _base_result(
            "rejected",
            "missing_trusted_manifest_hash",
            failure={
                "phase": "validation",
                "detail": (
                    "a submitted artifact requires a trusted manifest SHA-256"
                ),
            },
        )
    if not callable(runner):
        raise ValueError("runner must be callable")
    limits.validate()
    trusted_theorems = _validate_theorem_names(
        tuple(expected_theorems), path="expected_theorems",
    )

    if allowed_axioms is None:
        normalized_allowed_axioms = None
    else:
        normalized_allowed_axioms = set()
        for axiom in allowed_axioms:
            if not isinstance(axiom, str) or not IDENTIFIER_RE.fullmatch(axiom):
                raise ValueError(f"invalid allowed axiom name: {axiom!r}")
            normalized_allowed_axioms.add(axiom)

    try:
        artifact = validate_lean_artifact(
            artifact_root,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_theorems=trusted_theorems,
            limits=limits,
            expected_toolchain=expected_toolchain,
        )
    except ArtifactRejected as exc:
        return _base_result(
            "rejected",
            exc.code,
            failure={"phase": "validation", "detail": str(exc)[:4_000]},
        )
    except OSError as exc:
        return _base_result(
            "rejected",
            "artifact_read_error",
            failure={"phase": "validation", "detail": str(exc)[:4_000]},
        )

    sealed_files = MappingProxyType(dict(artifact.files))
    common = {
        "files": sealed_files,
        "cwd": ".",
        "max_output_bytes": limits.max_output_bytes,
        "network_allowed": False,
        "workspace_writable": False,
    }
    toolchain_evidence = {
        "requested": artifact.toolchain,
        "lean": {"available": False},
        "lake": {
            "required": artifact.driver == "lake",
            "available": artifact.driver != "lake",
        },
    }

    try:
        lean_probe = _run(
            runner,
            RunnerRequest(
                phase="probe_lean",
                argv=("lean", "--version"),
                timeout_s=limits.probe_timeout_s,
                **common,
            ),
        )
    except Exception as exc:
        return _runner_failure(
            "probe_lean",
            "runner_unavailable",
            str(exc),
            artifact,
            status="unavailable",
            toolchain=toolchain_evidence,
        )
    if lean_probe.timed_out:
        return _runner_failure(
            "probe_lean",
            "lean_probe_timeout",
            _diagnostic_text(lean_probe),
            artifact,
            status="unavailable",
            toolchain=toolchain_evidence,
        )
    if lean_probe.returncode != 0:
        return _runner_failure(
            "probe_lean",
            "lean_unavailable",
            _diagnostic_text(lean_probe),
            artifact,
            status="unavailable",
            toolchain=toolchain_evidence,
        )
    lean_version_text = _diagnostic_text(lean_probe, 1_000)
    toolchain_evidence["lean"] = {
        "available": True,
        "version": lean_version_text,
    }
    pinned_match = PINNED_VERSION_RE.search(artifact.toolchain)
    reported_match = LEAN_VERSION_RE.search(lean_version_text)
    if pinned_match and (
        not reported_match or reported_match.group(1) != pinned_match.group(1)
    ):
        return _runner_failure(
            "probe_lean",
            "lean_version_mismatch",
            (
                f"requested {pinned_match.group(1)!r}, "
                f"reported {reported_match.group(1)!r}"
                if reported_match
                else "runner did not report a parseable Lean version"
            ),
            artifact,
            status="unavailable",
            toolchain=toolchain_evidence,
        )

    if artifact.driver == "lake":
        try:
            lake_probe = _run(
                runner,
                RunnerRequest(
                    phase="probe_lake",
                    argv=("lake", "--version"),
                    timeout_s=limits.probe_timeout_s,
                    **common,
                ),
            )
        except Exception as exc:
            return _runner_failure(
                "probe_lake",
                "runner_unavailable",
                str(exc),
                artifact,
                status="unavailable",
                toolchain=toolchain_evidence,
            )
        if lake_probe.timed_out or lake_probe.returncode != 0:
            return _runner_failure(
                "probe_lake",
                (
                    "lake_probe_timeout"
                    if lake_probe.timed_out else
                    "lake_unavailable"
                ),
                _diagnostic_text(lake_probe),
                artifact,
                status="unavailable",
                toolchain=toolchain_evidence,
            )
        toolchain_evidence["lake"] = {
            "required": True,
            "available": True,
            "version": _diagnostic_text(lake_probe, 1_000),
        }

    command_prefix = ("lean",) if artifact.driver == "lean" else (
        "lake", "env", "lean",
    )
    try:
        compile_result = _run(
            runner,
            RunnerRequest(
                phase="compile",
                argv=(*command_prefix, artifact.entrypoint),
                timeout_s=limits.compile_timeout_s,
                **common,
            ),
        )
    except Exception as exc:
        return _runner_failure(
            "compile",
            "runner_unavailable",
            str(exc),
            artifact,
            status="unavailable",
            toolchain=toolchain_evidence,
        )
    if compile_result.timed_out:
        return _runner_failure(
            "compile",
            "compile_timeout",
            _diagnostic_text(compile_result),
            artifact,
            status="rejected",
            toolchain=toolchain_evidence,
        )
    if compile_result.returncode != 0:
        return _runner_failure(
            "compile",
            "compile_failed",
            _diagnostic_text(compile_result),
            artifact,
            status="rejected",
            toolchain=toolchain_evidence,
        )
    compile_output = "\n".join(
        (compile_result.stdout, compile_result.stderr),
    )
    if FORBIDDEN_DIAGNOSTIC_RE.search(compile_output):
        return _runner_failure(
            "compile",
            "forbidden_proof_hole",
            _diagnostic_text(compile_result),
            artifact,
            status="rejected",
            toolchain=toolchain_evidence,
        )

    audit_axioms = normalized_allowed_axioms is not None
    audit_files = dict(sealed_files)
    audit_files[RESERVED_AUDIT_FILE] = _make_audit_source(
        artifact, audit_axioms=audit_axioms,
    )
    audit_common = dict(common)
    audit_common["files"] = MappingProxyType(audit_files)
    try:
        audit_result = _run(
            runner,
            RunnerRequest(
                phase="audit",
                argv=(*command_prefix, RESERVED_AUDIT_FILE),
                timeout_s=limits.audit_timeout_s,
                **audit_common,
            ),
        )
    except Exception as exc:
        return _runner_failure(
            "audit",
            "runner_unavailable",
            str(exc),
            artifact,
            status="unavailable",
            toolchain=toolchain_evidence,
        )
    if audit_result.timed_out:
        return _runner_failure(
            "audit",
            "audit_timeout",
            _diagnostic_text(audit_result),
            artifact,
            status="rejected",
            toolchain=toolchain_evidence,
        )
    if audit_result.returncode != 0:
        return _runner_failure(
            "audit",
            "theorem_check_failed",
            _diagnostic_text(audit_result),
            artifact,
            status="rejected",
            toolchain=toolchain_evidence,
        )
    audit_output = "\n".join((audit_result.stdout, audit_result.stderr))
    if FORBIDDEN_DIAGNOSTIC_RE.search(audit_output):
        return _runner_failure(
            "audit",
            "forbidden_proof_hole",
            _diagnostic_text(audit_result),
            artifact,
            status="rejected",
            toolchain=toolchain_evidence,
        )

    axiom_evidence: Mapping[str, Any]
    if normalized_allowed_axioms is None:
        axiom_evidence = {"status": "not_requested"}
    else:
        try:
            parsed_axioms = _parse_axioms(
                audit_output, artifact.expected_theorems,
            )
        except ValueError as exc:
            return _runner_failure(
                "audit",
                "axiom_audit_unparseable",
                str(exc),
                artifact,
                status="rejected",
                toolchain=toolchain_evidence,
            )
        unexpected = {
            theorem: sorted(set(axioms) - normalized_allowed_axioms)
            for theorem, axioms in parsed_axioms.items()
            if set(axioms) - normalized_allowed_axioms
        }
        if unexpected:
            return _runner_failure(
                "audit",
                "unexpected_axioms",
                json.dumps(
                    unexpected,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                artifact,
                status="rejected",
                toolchain=toolchain_evidence,
            )
        axiom_evidence = {
            "status": "checked",
            "allowed": sorted(normalized_allowed_axioms),
            "observed": {
                theorem: list(axioms)
                for theorem, axioms in parsed_axioms.items()
            },
        }
        if include_axiom_audit_text:
            axiom_evidence = {
                **axiom_evidence,
                "text": _diagnostic_text(audit_result),
            }

    return _base_result(
        "verified",
        "all_checks_passed",
        manifest_sha256=artifact.manifest_sha256,
        artifact_sha256=artifact.artifact_sha256,
        expected_theorems=list(artifact.expected_theorems),
        toolchain=toolchain_evidence,
        checks={
            "schema": "checked",
            "paths_sizes_hashes": "checked",
            "forbidden_proof_holes": "checked",
            "compile": "checked",
            "theorem_declarations": "checked",
            "axioms": axiom_evidence,
        },
    )


def _normalize_rational(value: Any, *, path: str) -> Mapping[str, int]:
    """Normalize one candidate-discovered rational without float coercion."""
    _strict_object(
        value,
        required={"numerator", "denominator"},
        allowed={"numerator", "denominator"},
        path=path,
    )
    numerator = value["numerator"]
    denominator = value["denominator"]
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        raise ArtifactRejected(
            "invalid_claim_target", f"{path}.numerator must be an integer",
        )
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise ArtifactRejected(
            "invalid_claim_target",
            f"{path}.denominator must be a positive integer",
        )
    if abs(numerator) > 1_000_000 or denominator > 1_000_000:
        raise ArtifactRejected(
            "invalid_claim_target", f"{path} exceeds the rational bound",
        )
    common = math.gcd(abs(numerator), denominator)
    return {
        "numerator": numerator // common,
        "denominator": denominator // common,
    }


def _claim_type_expression(claim: Mapping[str, Any]) -> str:
    template = claim.get("template")
    declaration = FORMAL_CLAIM_TYPES.get(template)
    if declaration is None:
        raise ArtifactRejected(
            "unsupported_formal_claim",
            f"claim {claim.get('id')!r} has no trusted formal target",
        )
    target = _normalize_rational(
        claim.get("target"), path=f"claim {claim.get('id')}.target",
    )
    numerator = target["numerator"]
    denominator = target["denominator"]
    rational = f"(({numerator} : ℝ) / ({denominator} : ℝ))"
    return f"OpenHyraSumDiff.{declaration} {rational}"


def validate_formalization_request(
    raw: Optional[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    *,
    allow_sealed_hashes: bool = False,
) -> Tuple[Optional[Mapping[str, Any]], Mapping[str, str]]:
    """Validate inline proof terms against trusted claim templates.

    The candidate supplies only proof terms and claim ids.  It cannot choose
    imports, declaration names, theorem types, commands, or verdicts.
    """
    if raw is None:
        return None, {}
    _strict_object(
        raw,
        required={"schema", "target", "proofs"},
        allowed=(
            {"schema", "target", "proofs", "request_sha256"}
            if allow_sealed_hashes else
            {"schema", "target", "proofs"}
        ),
        path="formalization",
    )
    if raw["schema"] != REQUEST_SCHEMA:
        raise ArtifactRejected(
            "unsupported_schema",
            f"formalization.schema must be {REQUEST_SCHEMA!r}",
        )
    if raw["target"] != TARGET:
        raise ArtifactRejected(
            "unsupported_target", "formalization.target must be 'lean4'",
        )
    raw_proofs = raw["proofs"]
    if not isinstance(raw_proofs, list) or not raw_proofs:
        raise ArtifactRejected(
            "invalid_schema", "formalization.proofs must be a non-empty list",
        )
    if len(raw_proofs) > MAX_PROOFS:
        raise ArtifactRejected(
            "invalid_schema",
            f"formalization.proofs exceeds {MAX_PROOFS} entries",
        )

    claim_by_id = {
        claim.get("id"): claim
        for claim in claims
        if isinstance(claim.get("id"), str)
    }
    normalized_proofs = []
    theorem_types = {}
    seen = set()
    for index, raw_proof in enumerate(raw_proofs):
        path = f"formalization.proofs[{index}]"
        _strict_object(
            raw_proof,
            required={"claim_id", "term"},
            allowed=(
                {"claim_id", "term", "proof_sha256"}
                if allow_sealed_hashes else
                {"claim_id", "term"}
            ),
            path=path,
        )
        claim_id = raw_proof["claim_id"]
        if not isinstance(claim_id, str) or claim_id not in claim_by_id:
            raise ArtifactRejected(
                "unknown_claim", f"{path}.claim_id is not a known claim",
            )
        if claim_id in seen:
            raise ArtifactRejected(
                "duplicate_claim", f"formalization repeats claim {claim_id!r}",
            )
        seen.add(claim_id)
        term = raw_proof["term"]
        if not isinstance(term, str) or not term.strip():
            raise ArtifactRejected(
                "invalid_proof", f"{path}.term must be non-empty text",
            )
        term = term.strip()
        if len(term) > MAX_PROOF_TERM_CHARS:
            raise ArtifactRejected(
                "invalid_proof",
                f"{path}.term exceeds {MAX_PROOF_TERM_CHARS} characters",
            )
        _validate_inline_term_boundary(term, path=path)
        stripped = _strip_lean_comments_and_strings(term)
        match = FORBIDDEN_SOURCE_TOKEN_RE.search(stripped)
        if match:
            raise ArtifactRejected(
                "forbidden_proof_hole",
                f"{path}.term contains forbidden token {match.group(1)!r}",
            )
        theorem_types[claim_id] = _claim_type_expression(
            claim_by_id[claim_id],
        )
        proof_sha256 = hashlib.sha256(term.encode("utf-8")).hexdigest()
        supplied_proof_sha256 = raw_proof.get("proof_sha256")
        if (
            allow_sealed_hashes
            and supplied_proof_sha256 is not None
            and supplied_proof_sha256 != proof_sha256
        ):
            raise ArtifactRejected(
                "sealed_hash_mismatch",
                f"{path}.proof_sha256 does not match the proof term",
            )
        normalized_proofs.append({
            "claim_id": claim_id,
            "term": term,
            "proof_sha256": proof_sha256,
        })
    normalized_proofs.sort(key=lambda item: item["claim_id"])
    normalized = {
        "schema": REQUEST_SCHEMA,
        "target": TARGET,
        "proofs": normalized_proofs,
    }
    request_sha256 = hashlib.sha256(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    supplied_request_sha256 = raw.get("request_sha256")
    if (
        allow_sealed_hashes
        and supplied_request_sha256 is not None
        and supplied_request_sha256 != request_sha256
    ):
        raise ArtifactRejected(
            "sealed_hash_mismatch",
            "formalization.request_sha256 does not match the request",
        )
    normalized["request_sha256"] = request_sha256
    return MappingProxyType(normalized), MappingProxyType(theorem_types)


def build_formalization_wrapper(
    normalized: Mapping[str, Any],
    theorem_types: Mapping[str, str],
    *,
    spec_module: str = "OpenHyraSumDiff.Spec",
    trusted_spec_source: Optional[bytes] = None,
) -> Tuple[bytes, Mapping[str, str]]:
    """Generate declarations whose types are owned by the trusted task."""
    if not isinstance(spec_module, str) or not IDENTIFIER_RE.fullmatch(
        spec_module
    ):
        raise ValueError("spec_module must be a safe Lean module name")
    theorem_names = {}
    if trusted_spec_source is None:
        lines = [f"import {spec_module}", ""]
    else:
        if not isinstance(trusted_spec_source, bytes):
            raise ValueError("trusted_spec_source must be bytes")
        try:
            spec_text = trusted_spec_source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "trusted_spec_source must be valid UTF-8"
            ) from exc
        lines = [spec_text.rstrip(), ""]
    lines.extend([
        "set_option autoImplicit false",
        "namespace OpenHyraCandidate",
        "",
    ])
    for index, proof in enumerate(normalized["proofs"]):
        claim_id = proof["claim_id"]
        theorem_name = f"OpenHyraCandidate.claim_{index:02d}"
        local_name = theorem_name.rsplit(".", 1)[-1]
        theorem_names[claim_id] = theorem_name
        lines.extend([
            f"theorem {local_name} : {theorem_types[claim_id]} :=",
            f"  ({proof['term']})",
            "",
        ])
    lines.append("end OpenHyraCandidate")
    lines.append("")
    return "\n".join(lines).encode("utf-8"), MappingProxyType(theorem_names)


def build_formalization_audit(
    theorem_names: Mapping[str, str],
) -> bytes:
    """Build a trusted audit module run after candidate compilation."""
    lines = [
        "import OpenHyraCandidate",
        "",
        "set_option autoImplicit false",
        "",
    ]
    for theorem_name in theorem_names.values():
        if (
            not isinstance(theorem_name, str)
            or not IDENTIFIER_RE.fullmatch(theorem_name)
        ):
            raise ValueError("theorem name has invalid syntax")
        lines.append(f"#check {theorem_name}")
        lines.append(f"#print axioms {theorem_name}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def verify_formalization_request(
    raw: Optional[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    *,
    runner: Optional[Runner],
    trusted_files: Mapping[str, bytes],
    command_prefix: Sequence[str] = ("lake", "env", "lean"),
    toolchain: Optional[str] = None,
    mathlib_revision: Optional[str] = None,
    allowed_axioms: Iterable[str] = DEFAULT_ALLOWED_AXIOMS,
    limits: VerificationLimits = VerificationLimits(),
) -> Mapping[str, Any]:
    """Compile typed inline proofs through a caller-owned isolated runner."""
    try:
        normalized, theorem_types = validate_formalization_request(
            raw,
            claims,
            allow_sealed_hashes=True,
        )
    except ArtifactRejected as exc:
        return _base_result(
            "rejected",
            exc.code,
            failure={"phase": "validation", "detail": str(exc)[:4_000]},
        )
    if normalized is None:
        return _base_result("not_submitted", "no_formalization_request")
    base = {
        "request_sha256": normalized["request_sha256"],
        "proofs": [
            {
                "claim_id": item["claim_id"],
                "proof_sha256": item["proof_sha256"],
            }
            for item in normalized["proofs"]
        ],
    }
    if runner is None:
        return _base_result(
            "unavailable",
            "isolated_runner_not_configured",
            **base,
        )
    if (
        not isinstance(command_prefix, (list, tuple))
        or not command_prefix
        or any(not isinstance(item, str) or not item for item in command_prefix)
    ):
        raise ValueError("command_prefix must be non-empty trusted argv")
    if toolchain is not None and (
        not isinstance(toolchain, str)
        or not TOOLCHAIN_RE.fullmatch(toolchain)
        or ".." in toolchain
    ):
        raise ValueError("toolchain has invalid syntax")
    if mathlib_revision is not None and (
        not isinstance(mathlib_revision, str)
        or not GIT_REVISION_RE.fullmatch(mathlib_revision)
    ):
        raise ValueError(
            "mathlib_revision must be a full lowercase commit hash"
        )
    limits.validate()
    allowed = set()
    for axiom in allowed_axioms:
        if not isinstance(axiom, str) or not IDENTIFIER_RE.fullmatch(axiom):
            raise ValueError(f"invalid allowed axiom name: {axiom!r}")
        allowed.add(axiom)

    wrapper, theorem_names = build_formalization_wrapper(
        normalized,
        theorem_types,
        trusted_spec_source=trusted_files.get(TRUSTED_SPEC_FILE),
    )
    audit = build_formalization_audit(theorem_names)
    files = dict(trusted_files)
    for reserved in (RESERVED_CANDIDATE_FILE, RESERVED_AUDIT_FILE):
        if reserved in files:
            raise ValueError(f"trusted_files reserves {reserved}")
    files[RESERVED_CANDIDATE_FILE] = wrapper
    files[RESERVED_AUDIT_FILE] = audit
    if sum(len(data) for data in files.values()) > limits.max_total_bytes:
        return _base_result(
            "rejected", "formalization_input_too_large", **base,
        )
    common_request = {
        "files": MappingProxyType(files),
        "cwd": ".",
        "max_output_bytes": limits.max_output_bytes,
        "network_allowed": False,
        "workspace_writable": False,
        "expected_toolchain": toolchain,
        "expected_mathlib_revision": mathlib_revision,
    }
    toolchain_evidence = {
        "requested": toolchain,
        "lean": {"available": False},
    }
    try:
        probe = _run(
            runner,
            RunnerRequest(
                phase="probe_lean",
                argv=(*tuple(command_prefix), "--version"),
                timeout_s=limits.probe_timeout_s,
                **common_request,
            ),
        )
    except Exception as exc:
        return _base_result(
            "unavailable",
            "runner_unavailable",
            toolchain=toolchain_evidence,
            failure={
                "phase": "probe_lean",
                "detail": str(exc)[:4_000],
            },
            **base,
        )
    if probe.timed_out:
        return _base_result(
            "unavailable",
            "lean_probe_timeout",
            toolchain=toolchain_evidence,
            failure={
                "phase": "probe_lean",
                "detail": _diagnostic_text(probe),
            },
            **base,
        )
    if probe.returncode != 0:
        return _base_result(
            "unavailable",
            "lean_unavailable",
            toolchain=toolchain_evidence,
            failure={
                "phase": "probe_lean",
                "detail": _diagnostic_text(probe),
            },
            **base,
        )
    version_text = _diagnostic_text(probe, 1_000)
    toolchain_evidence["lean"] = {
        "available": True,
        "version": version_text,
    }
    if toolchain is not None:
        pinned = PINNED_VERSION_RE.search(toolchain)
        reported = LEAN_VERSION_RE.search(version_text)
        if pinned and (
            reported is None or reported.group(1) != pinned.group(1)
        ):
            return _base_result(
                "unavailable",
                "lean_version_mismatch",
                toolchain=toolchain_evidence,
                failure={
                    "phase": "probe_lean",
                    "detail": (
                        f"requested {pinned.group(1)!r}, "
                        + (
                            f"reported {reported.group(1)!r}"
                            if reported is not None else
                            "runner did not report a parseable Lean version"
                        )
                    ),
                },
                **base,
            )
    runtime_attestation = probe.attestation
    if mathlib_revision is not None:
        if probe.output_complete is not True:
            return _base_result(
                "unavailable",
                "runner_output_completeness_unattested",
                toolchain=toolchain_evidence,
                failure={
                    "phase": "probe_lean",
                    "detail": (
                        "the runner did not attest that its output was "
                        "complete and untruncated"
                    ),
                },
                **base,
            )
        if runtime_attestation is None:
            return _base_result(
                "unavailable",
                "runtime_attestation_missing",
                toolchain=toolchain_evidence,
                failure={
                    "phase": "probe_lean",
                    "detail": (
                        "the runner did not attest its Lean/Mathlib "
                        "execution environment"
                    ),
                },
                **base,
            )
        if (
            runtime_attestation["toolchain"] != toolchain
            or runtime_attestation["mathlib_revision"] != mathlib_revision
        ):
            return _base_result(
                "unavailable",
                "runtime_attestation_mismatch",
                toolchain=toolchain_evidence,
                runtime_attestation=dict(runtime_attestation),
                failure={
                    "phase": "probe_lean",
                    "detail": (
                        "the attested toolchain or Mathlib revision does "
                        "not match trusted task configuration"
                    ),
                },
                **base,
            )
    request = RunnerRequest(
        phase="compile_then_audit",
        argv=(
            *tuple(command_prefix),
            "-o",
            "OpenHyraCandidate.olean",
            RESERVED_CANDIDATE_FILE,
        ),
        audit_argv=(*tuple(command_prefix), RESERVED_AUDIT_FILE),
        timeout_s=limits.compile_timeout_s + limits.audit_timeout_s,
        **common_request,
    )
    try:
        result = _run(runner, request)
    except Exception as exc:
        return _base_result(
            "unavailable",
            "runner_unavailable",
            failure={
                "phase": "compile_then_audit",
                "detail": str(exc)[:4_000],
            },
            toolchain=toolchain_evidence,
            **base,
        )
    if result.timed_out:
        return _base_result(
            "rejected",
            "formalization_timeout",
            failure={
                "phase": "compile_then_audit",
                "detail": _diagnostic_text(result),
            },
            toolchain=toolchain_evidence,
            **base,
        )
    if mathlib_revision is not None and result.output_complete is not True:
        return _base_result(
            "unavailable",
            "runner_output_completeness_unattested",
            failure={
                "phase": "compile_then_audit",
                "detail": (
                    "the runner did not attest that its output was "
                    "complete and untruncated"
                ),
            },
            toolchain=toolchain_evidence,
            **base,
        )
    if mathlib_revision is not None and result.attestation != runtime_attestation:
        return _base_result(
            "unavailable",
            "runtime_attestation_changed",
            failure={
                "phase": "compile_then_audit",
                "detail": (
                    "the runner environment changed between its Lean probe "
                    "and proof compilation"
                ),
            },
            toolchain=toolchain_evidence,
            runtime_attestation=(
                dict(result.attestation)
                if result.attestation is not None else None
            ),
            **base,
        )
    if result.returncode != 0:
        return _base_result(
            "rejected",
            "formalization_failed",
            failure={
                "phase": "candidate_compile",
                "detail": _diagnostic_text(result),
            },
            toolchain=toolchain_evidence,
            **base,
        )
    if result.audit_returncode is None:
        return _base_result(
            "unavailable",
            "separate_axiom_audit_not_run",
            failure={
                "phase": "axiom_audit",
                "detail": (
                    "the isolated runner did not return a separate audit "
                    "process result"
                ),
            },
            toolchain=toolchain_evidence,
            **base,
        )
    if result.audit_returncode != 0:
        return _base_result(
            "rejected",
            "axiom_audit_failed",
            failure={
                "phase": "axiom_audit",
                "detail": "\n".join((
                    result.audit_stdout, result.audit_stderr,
                ))[:4_000],
            },
            toolchain=toolchain_evidence,
            **base,
        )
    compile_output = "\n".join((result.stdout, result.stderr))
    audit_output = "\n".join((
        result.audit_stdout, result.audit_stderr,
    ))
    if FORBIDDEN_DIAGNOSTIC_RE.search(
        compile_output + "\n" + audit_output
    ):
        return _base_result(
            "rejected",
            "forbidden_proof_hole",
            failure={
                "phase": "compile_then_audit",
                "detail": (
                    compile_output + "\n" + audit_output
                )[:4_000],
            },
            toolchain=toolchain_evidence,
            **base,
        )
    try:
        parsed = _parse_axioms(
            audit_output, tuple(theorem_names.values())
        )
    except ValueError as exc:
        return _base_result(
            "rejected",
            "axiom_audit_unparseable",
            failure={"phase": "axiom_audit", "detail": str(exc)},
            toolchain=toolchain_evidence,
            **base,
        )
    unexpected = {
        claim_id: sorted(set(parsed[theorem_name]) - allowed)
        for claim_id, theorem_name in theorem_names.items()
        if set(parsed[theorem_name]) - allowed
    }
    if unexpected:
        return _base_result(
            "rejected",
            "unexpected_axioms",
            unexpected_axioms=unexpected,
            toolchain=toolchain_evidence,
            **base,
        )
    verified_claims = sorted(theorem_names)
    return _base_result(
        "verified",
        "all_checks_passed",
        verified_claim_ids=verified_claims,
        theorem_names=dict(theorem_names),
        theorem_types=dict(theorem_types),
        wrapper_sha256=hashlib.sha256(wrapper).hexdigest(),
        audit_sha256=hashlib.sha256(audit).hexdigest(),
        toolchain=toolchain_evidence,
        runtime_attestation=(
            dict(runtime_attestation)
            if runtime_attestation is not None else None
        ),
        axioms={
            claim_id: list(parsed[theorem_names[claim_id]])
            for claim_id in verified_claims
        },
        **base,
    )


__all__ = [
    "ArtifactRejected",
    "DEFAULT_ALLOWED_AXIOMS",
    "MANIFEST_NAME",
    "RESERVED_AUDIT_FILE",
    "RESERVED_CANDIDATE_FILE",
    "RunnerRequest",
    "RunnerResult",
    "SCHEMA",
    "REQUEST_SCHEMA",
    "TARGET",
    "TRUSTED_SPEC_FILE",
    "ValidatedArtifact",
    "VerificationLimits",
    "validate_lean_artifact",
    "validate_formalization_request",
    "build_formalization_wrapper",
    "build_formalization_audit",
    "verify_lean_artifact",
    "verify_formalization_request",
]
