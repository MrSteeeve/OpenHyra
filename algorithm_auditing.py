"""Safe freezing and provenance helpers for per-instance algorithm audits.

This module deliberately does not execute candidate code.  It establishes the
immutable boundary that a private audit runner can consume later:

* an algorithm bundle is exactly ``train.py``, ``manifest.json``, and an
  optional ``features.json``;
* distinct Top-K bundles are copied from Experience Bank records before a
  private seed is drawn; and
* every per-instance/repeat policy artifact can be bound back to the frozen
  algorithm, its inputs, runtime, and training seed.

The implementation is independent of the evaluator and harness.  It reuses the
sandbox's no-follow, single-hard-link reader so callers do not acquire a second,
weaker untrusted-file path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sandbox import read_regular_file


ALGORITHM_BUNDLE_SCHEMA = "openhyra-candidate-algorithm-bundle.v1"
ALGORITHM_FREEZE_MANIFEST_SCHEMA = "openhyra-algorithm-audit-freeze.v1"
PER_CELL_POLICY_PROVENANCE_SCHEMA = (
    "openhyra-per-cell-policy-provenance.v1"
)

REQUIRED_ALGORITHM_FILES = frozenset({"train.py", "manifest.json"})
OPTIONAL_ALGORITHM_FILES = frozenset({"features.json"})
ALLOWED_ALGORITHM_FILES = REQUIRED_ALGORITHM_FILES | OPTIONAL_ALGORITHM_FILES
DEFAULT_MAX_BUNDLE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_SAFE_INTEGER = (1 << 63) - 1
HEX_DIGITS = frozenset("0123456789abcdef")
SAFE_RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}")
VALID_STAGES = frozenset({"search", "audit"})


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("provenance must be canonical-JSON serializable") from exc


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS)
    )


def _require_sha256(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_safe_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not SAFE_LABEL.fullmatch(value):
        raise ValueError(f"{field} must be bounded safe text")
    return value


def _require_exact_fields(payload: Mapping[str, Any], expected: set[str],
                          label: str) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{label} fields must be exactly: {', '.join(sorted(expected))}"
        )


class _RecordMapping(Mapping[str, Any]):
    """Small mapping adapter for immutable records used by integrations."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class AlgorithmFileRecord(_RecordMapping):
    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class RecordProvenance(_RecordMapping):
    record_id: str
    record_sha256: str
    record_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_sha256": self.record_sha256,
            "record_path": self.record_path,
        }


@dataclass(frozen=True)
class SourceProvenance(_RecordMapping):
    source_path: str
    declared_algorithm_bundle_sha256: str | None
    total_bytes: int
    files: tuple[AlgorithmFileRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "declared_algorithm_bundle_sha256": (
                self.declared_algorithm_bundle_sha256
            ),
            "total_bytes": self.total_bytes,
            "files": [item.to_dict() for item in self.files],
        }


@dataclass(frozen=True)
class CandidateAlgorithmBundle(_RecordMapping):
    schema: str
    algorithm_bundle_sha256: str
    record_provenance: RecordProvenance
    source_provenance: SourceProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "algorithm_bundle_sha256": self.algorithm_bundle_sha256,
            "record_provenance": self.record_provenance.to_dict(),
            "source_provenance": self.source_provenance.to_dict(),
        }


@dataclass(frozen=True)
class FrozenAlgorithmCandidate(_RecordMapping):
    rank: int
    record_id: str
    search_score: float
    algorithm_bundle_sha256: str
    frozen_bundle: str
    record_provenance: RecordProvenance
    source_provenance: SourceProvenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "record_id": self.record_id,
            "search_score": self.search_score,
            "algorithm_bundle_sha256": self.algorithm_bundle_sha256,
            "frozen_bundle": self.frozen_bundle,
            "record_provenance": self.record_provenance.to_dict(),
            "source_provenance": self.source_provenance.to_dict(),
        }


@dataclass(frozen=True)
class AlgorithmFreezeManifest(_RecordMapping):
    schema: str
    frozen_at: str
    run_manifest_sha256: str
    direction: str
    requested_top_k: int
    eb_snapshot_version: int
    eb_snapshot_sha256: str
    bundle_subdir: str
    candidate_count: int
    candidates: tuple[FrozenAlgorithmCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "frozen_at": self.frozen_at,
            "run_manifest_sha256": self.run_manifest_sha256,
            "direction": self.direction,
            "requested_top_k": self.requested_top_k,
            "eb_snapshot_version": self.eb_snapshot_version,
            "eb_snapshot_sha256": self.eb_snapshot_sha256,
            "bundle_subdir": self.bundle_subdir,
            "candidate_count": self.candidate_count,
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @property
    def sha256(self) -> str:
        """Digest of the canonical logical manifest, not pretty-print bytes."""
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class PerCellPolicyProvenance(_RecordMapping):
    schema: str
    stage: str
    suite: str
    instance: str
    repeat: int
    freeze_manifest_sha256: str
    run_manifest_sha256: str
    evaluation_request_sha256: str
    trusted_runner_sha256: str
    algorithm_bundle_sha256: str
    instance_sha256: str
    training_input_sha256: str
    train_seed: int
    runtime_sha256: str
    policy_artifact_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "suite": self.suite,
            "instance": self.instance,
            "repeat": self.repeat,
            "freeze_manifest_sha256": self.freeze_manifest_sha256,
            "run_manifest_sha256": self.run_manifest_sha256,
            "evaluation_request_sha256": self.evaluation_request_sha256,
            "trusted_runner_sha256": self.trusted_runner_sha256,
            "algorithm_bundle_sha256": self.algorithm_bundle_sha256,
            "instance_sha256": self.instance_sha256,
            "training_input_sha256": self.training_input_sha256,
            "train_seed": self.train_seed,
            "runtime_sha256": self.runtime_sha256,
            "policy_artifact_sha256": self.policy_artifact_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class _BundleSnapshot:
    bundle: CandidateAlgorithmBundle
    files: tuple[tuple[str, bytes], ...]


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _validate_candidate_json(data: bytes, name: str) -> None:
    """Require a JSON object without taking ownership of policy semantics."""
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"candidate {name} must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"candidate {name} must contain one JSON object")


def _reject_symlink_components(raw: Path, lexical_root: Path) -> None:
    """Reject symlinks below a trusted lexical root, including parent links."""
    try:
        relative = raw.relative_to(lexical_root)
    except ValueError as exc:
        raise ValueError("candidate source path escapes the expected EB root") from exc
    current = lexical_root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise ValueError("candidate source directory not found") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("candidate source path must not traverse symbolic links")


def _validate_source_path(source_dir: os.PathLike[str] | str,
                          expected_source_root: os.PathLike[str] | str | None
                          ) -> tuple[Path, Path]:
    raw = Path(source_dir)
    if not raw.is_absolute() or ".." in raw.parts:
        raise ValueError("candidate source path must be absolute and contain no '..'")
    try:
        info = os.lstat(raw)
    except FileNotFoundError as exc:
        raise ValueError("candidate source directory not found") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("candidate source must be a real directory")
    resolved = raw.resolve(strict=True)
    if expected_source_root is not None:
        root_raw = Path(expected_source_root)
        if not root_raw.is_absolute() or ".." in root_raw.parts:
            raise ValueError("expected source root must be an absolute safe path")
        try:
            root = root_raw.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("candidate source path escapes the expected EB root") from exc
        if raw.is_relative_to(root_raw):
            _reject_symlink_components(raw, root_raw)
        else:
            # macOS commonly canonicalizes /var to /private/var.  A provenance
            # path already stored in canonical form is still beneath the same
            # trusted root and must not be mistaken for an escape.
            _reject_symlink_components(resolved, root)
    return raw, resolved


def _source_signature(source_dir: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    """Return a stable, flat directory signature and reject unsafe entries."""
    entries: list[tuple[str, int, int, int, int]] = []
    try:
        scanned = list(os.scandir(source_dir))
    except OSError as exc:
        raise ValueError(f"could not scan candidate source: {exc}") from exc
    for entry in scanned:
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"could not inspect candidate entry {entry.name}") from exc
        if entry.name not in ALLOWED_ALGORITHM_FILES:
            raise ValueError(f"undeclared candidate bundle entry: {entry.name}")
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"candidate bundle entry {entry.name} must be a regular file"
            )
        entries.append(
            (entry.name, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        )
    names = {item[0] for item in entries}
    missing = REQUIRED_ALGORITHM_FILES - names
    if missing:
        raise ValueError(
            "candidate algorithm bundle is missing: " + ", ".join(sorted(missing))
        )
    return tuple(sorted(entries))


def _declared_algorithm_hash(record: Mapping[str, Any]) -> str | None:
    declared: list[str] = []
    for container in (
        record,
        record.get("metrics", {}),
        record.get("metadata", {}),
    ):
        if not isinstance(container, Mapping):
            raise ValueError("record metrics and metadata must be objects")
        if "algorithm_bundle_sha256" not in container:
            continue
        value = container["algorithm_bundle_sha256"]
        declared.append(_require_sha256(value, "algorithm_bundle_sha256"))
    if len(set(declared)) > 1:
        raise ValueError("record has conflicting algorithm bundle digests")
    return declared[0] if declared else None


def _bundle_digest(files: tuple[AlgorithmFileRecord, ...]) -> str:
    return _sha256_json({
        "schema": ALGORITHM_BUNDLE_SCHEMA,
        "files": [item.to_dict() for item in files],
    })


def _validate_bundle_subdir(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("bundle_subdir must be explicit non-empty text")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("bundle_subdir must be a safe relative POSIX path")
    canonical = path.as_posix()
    if canonical in {"", "/"}:
        raise ValueError("bundle_subdir must be a safe relative POSIX path")
    return canonical


def _record_run_manifest_sha256(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("record metadata must be an object")
    return _require_sha256(
        metadata.get("run_manifest_sha256"),
        "record.metadata.run_manifest_sha256",
    )


def _read_bundle_snapshot(
        record: Mapping[str, Any],
        source_dir: os.PathLike[str] | str | None = None,
        *,
        expected_source_root: os.PathLike[str] | str | None = None,
        bundle_subdir: str = ".",
        expected_run_manifest_sha256: str | None = None,
        max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
        require_declared_bundle_hash: bool = True) -> _BundleSnapshot:
    if not isinstance(record, Mapping):
        raise ValueError("Experience Bank record must be an object")
    record_id = record.get("id")
    if not isinstance(record_id, str) or not SAFE_RECORD_ID.fullmatch(record_id):
        raise ValueError("Experience Bank record id is unsafe")
    if (
        isinstance(max_bundle_bytes, bool)
        or not isinstance(max_bundle_bytes, int)
        or max_bundle_bytes < 1
    ):
        raise ValueError("max_bundle_bytes must be a positive integer")

    recorded_path = record.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError("Experience Bank record has no source path")
    candidate_slot = recorded_path if source_dir is None else source_dir
    raw_slot, resolved_slot = _validate_source_path(
        candidate_slot, expected_source_root,
    )
    recorded_raw = Path(recorded_path)
    if (
        not recorded_raw.is_absolute()
        or ".." in recorded_raw.parts
        or recorded_raw.resolve(strict=True) != resolved_slot
    ):
        raise ValueError("record path does not match the candidate source directory")

    canonical_subdir = _validate_bundle_subdir(bundle_subdir)
    raw_source = raw_slot / Path(canonical_subdir)
    raw_source, resolved_source = _validate_source_path(
        raw_source, resolved_slot,
    )
    if expected_run_manifest_sha256 is not None:
        expected_run = _require_sha256(
            expected_run_manifest_sha256, "expected_run_manifest_sha256",
        )
        if _record_run_manifest_sha256(record) != expected_run:
            raise ValueError("record run manifest provenance mismatch")

    before = _source_signature(raw_source)
    total = 0
    byte_items: list[tuple[str, bytes]] = []
    file_records: list[AlgorithmFileRecord] = []
    for name, _dev, _ino, size, _mtime in before:
        remaining = max_bundle_bytes - total
        if size > remaining:
            raise ValueError(
                f"candidate algorithm bundle exceeds {max_bundle_bytes} bytes"
            )
        data = read_regular_file(
            raw_source / name,
            remaining,
            label=f"candidate algorithm file {name}",
        )
        total += len(data)
        byte_items.append((name, data))
        file_records.append(AlgorithmFileRecord(
            path=name,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        ))
    after = _source_signature(raw_source)
    if before != after:
        raise ValueError("candidate algorithm bundle changed while it was read")
    files = tuple(sorted(file_records, key=lambda item: item.path))
    bytes_by_name = tuple(sorted(byte_items))
    byte_map = dict(bytes_by_name)
    _validate_candidate_json(byte_map["manifest.json"], "manifest.json")
    if "features.json" in byte_map:
        _validate_candidate_json(byte_map["features.json"], "features.json")
    algorithm_hash = _bundle_digest(files)
    declared_hash = _declared_algorithm_hash(record)
    if require_declared_bundle_hash and declared_hash is None:
        raise ValueError("record lacks declared algorithm_bundle_sha256 provenance")
    if declared_hash is not None and declared_hash != algorithm_hash:
        raise ValueError("record algorithm bundle provenance mismatch")
    record_payload = dict(record)
    record_hash = _sha256_json(record_payload)
    record_provenance = RecordProvenance(
        record_id=record_id,
        record_sha256=record_hash,
        record_path=str(resolved_slot),
    )
    source_provenance = SourceProvenance(
        source_path=str(resolved_source),
        declared_algorithm_bundle_sha256=declared_hash,
        total_bytes=total,
        files=files,
    )
    return _BundleSnapshot(
        bundle=CandidateAlgorithmBundle(
            schema=ALGORITHM_BUNDLE_SCHEMA,
            algorithm_bundle_sha256=algorithm_hash,
            record_provenance=record_provenance,
            source_provenance=source_provenance,
        ),
        files=bytes_by_name,
    )


def read_candidate_algorithm_bundle(
        record: Mapping[str, Any],
        source_dir: os.PathLike[str] | str | None = None,
        *,
        expected_source_root: os.PathLike[str] | str | None = None,
        bundle_subdir: str = ".",
        expected_run_manifest_sha256: str | None = None,
        max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
        require_declared_bundle_hash: bool = True) -> CandidateAlgorithmBundle:
    """Safely inspect one EB algorithm bundle without executing or copying it."""
    return _read_bundle_snapshot(
        record,
        source_dir,
        expected_source_root=expected_source_root,
        bundle_subdir=bundle_subdir,
        expected_run_manifest_sha256=expected_run_manifest_sha256,
        max_bundle_bytes=max_bundle_bytes,
        require_declared_bundle_hash=require_declared_bundle_hash,
    ).bundle


def _valid_scored_records(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    valid: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("status") != "ok":
            continue
        score = record.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            continue
        valid.append(record)
    return valid


def _ordered_records(records: Iterable[Mapping[str, Any]], direction: str
                     ) -> list[Mapping[str, Any]]:
    if direction not in {"min", "max"}:
        raise ValueError("search direction must be min or max")
    ordered = sorted(
        _valid_scored_records(records),
        key=lambda record: (
            -float(record["score"])
            if direction == "max" else float(record["score"]),
            str(record.get("id", "")),
        ),
    )
    seen_ids: set[str] = set()
    for record in ordered:
        record_id = record.get("id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate Experience Bank record id: {record_id}")
        seen_ids.add(record_id)
    return ordered


def compute_eb_snapshot_sha256(
        records: Iterable[Mapping[str, Any]], version: int) -> str:
    """Hash one ordered Experience Bank snapshot with its append version."""
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError("eb_snapshot_version must be a non-negative integer")
    materialized = list(records)
    if len(materialized) != version:
        raise ValueError("EB snapshot version must equal its record count")
    return _sha256_json({
        "domain": "openhyra-eb-snapshot.v1",
        "version": version,
        "records": [dict(record) for record in materialized],
    })


def _select_top_k_declared(
        records: list[Mapping[str, Any]],
        *,
        direction: str,
        top_k: int,
        run_manifest_sha256: str) -> list[Mapping[str, Any]]:
    """Select by immutable EB declarations before touching candidate sources."""
    selected: list[Mapping[str, Any]] = []
    seen_hashes: set[str] = set()
    for record in _ordered_records(records, direction):
        if _record_run_manifest_sha256(record) != run_manifest_sha256:
            raise ValueError("record run manifest provenance mismatch")
        declared = _declared_algorithm_hash(record)
        if declared is None:
            raise ValueError("record lacks declared algorithm_bundle_sha256 provenance")
        if declared in seen_hashes:
            continue
        seen_hashes.add(declared)
        selected.append(record)
        if len(selected) == top_k:
            break
    if not selected:
        raise ValueError("private audit has no successful algorithm candidates")
    return selected


def _validate_snapshot_run_provenance(
        records: list[Mapping[str, Any]], run_manifest_sha256: str) -> None:
    seen_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("every EB snapshot entry must be an object")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not SAFE_RECORD_ID.fullmatch(record_id):
            raise ValueError("every EB snapshot record must have a safe id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate Experience Bank record id: {record_id}")
        seen_ids.add(record_id)
        if _record_run_manifest_sha256(record) != run_manifest_sha256:
            raise ValueError("record run manifest provenance mismatch")


def _validate_destination_path(
        destination: os.PathLike[str] | str,
        expected_destination_root: os.PathLike[str] | str) -> Path:
    destination_path = Path(destination)
    root_raw = Path(expected_destination_root)
    if (
        not destination_path.is_absolute()
        or ".." in destination_path.parts
        or not root_raw.is_absolute()
        or ".." in root_raw.parts
    ):
        raise ValueError("destination and expected_destination_root must be safe absolute paths")
    try:
        root_info = os.lstat(root_raw)
    except FileNotFoundError as exc:
        raise ValueError("expected destination root does not exist") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("expected destination root must be a real directory")
    try:
        relative = destination_path.relative_to(root_raw)
    except ValueError as exc:
        raise ValueError("algorithm freeze destination escapes its trusted root") from exc
    if not relative.parts:
        raise ValueError("algorithm freeze destination must be below its trusted root")
    current = root_raw
    for part in relative.parts[:-1]:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise ValueError("algorithm freeze destination parent must already exist") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("algorithm freeze destination ancestors must be real directories")
    resolved_root = root_raw.resolve(strict=True)
    try:
        destination_path.parent.resolve(strict=True).relative_to(resolved_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("algorithm freeze destination escapes its trusted root") from exc
    return destination_path


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_relative_bundle_path(record_id: str) -> str:
    path = PurePosixPath("candidates") / record_id
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("frozen candidate path is unsafe")
    return path.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.write_bytes(encoded)
    path.chmod(0o400)


def freeze_top_k_algorithm_bundles(
        records: Iterable[Mapping[str, Any]] | Any,
        destination: os.PathLike[str] | str,
        *,
        direction: str,
        top_k: int,
        run_manifest_sha256: str,
        eb_snapshot_version: int,
        eb_snapshot_sha256: str,
        expected_source_root: os.PathLike[str] | str,
        expected_destination_root: os.PathLike[str] | str,
        bundle_subdir: str,
        max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
        frozen_at: str | None = None) -> AlgorithmFreezeManifest:
    """Freeze distinct search winners and write the manifest last.

    ``records`` may be an iterable of EB records or an ExperienceBank-like
    object exposing ``records()``.  Algorithm hashes, rather than trained policy
    hashes, are the deduplication key.  The caller may safely generate its
    private seed only after this function returns successfully.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("audit top_k must be a positive integer")
    run_manifest_sha256 = _require_sha256(
        run_manifest_sha256, "run_manifest_sha256",
    )
    eb_snapshot_sha256 = _require_sha256(
        eb_snapshot_sha256, "eb_snapshot_sha256",
    )
    canonical_subdir = _validate_bundle_subdir(bundle_subdir)
    timestamp = frozen_at or _timestamp()
    if not isinstance(timestamp, str) or not timestamp or len(timestamp) > 128:
        raise ValueError("frozen_at must be bounded non-empty text")
    if hasattr(records, "snapshot") and callable(records.snapshot):
        observed_version, records = records.snapshot()
        if observed_version != eb_snapshot_version:
            raise ValueError("EB changed after the requested snapshot")
    elif hasattr(records, "records") and callable(records.records):
        records = records.records()
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Iterable):
        raise ValueError("records must be an iterable of Experience Bank records")

    materialized = list(records)
    observed_snapshot_sha256 = compute_eb_snapshot_sha256(
        materialized, eb_snapshot_version,
    )
    if observed_snapshot_sha256 != eb_snapshot_sha256:
        raise ValueError("EB snapshot digest mismatch")
    _validate_snapshot_run_provenance(materialized, run_manifest_sha256)
    selected_records = _select_top_k_declared(
        materialized,
        direction=direction,
        top_k=top_k,
        run_manifest_sha256=run_manifest_sha256,
    )
    selected: list[tuple[Mapping[str, Any], _BundleSnapshot]] = []
    for record in selected_records:
        snapshot = _read_bundle_snapshot(
            record,
            expected_source_root=expected_source_root,
            bundle_subdir=canonical_subdir,
            expected_run_manifest_sha256=run_manifest_sha256,
            max_bundle_bytes=max_bundle_bytes,
            require_declared_bundle_hash=True,
        )
        selected.append((record, snapshot))

    destination_path = _validate_destination_path(
        destination, expected_destination_root,
    )
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(
            "algorithm freeze destination already exists; refusing to reuse it"
        )
    created = False
    try:
        destination_path.mkdir(mode=0o700)
        created = True
        candidates_root = destination_path / "candidates"
        candidates_root.mkdir(mode=0o700)
        frozen_candidates: list[FrozenAlgorithmCandidate] = []
        for rank, (record, snapshot) in enumerate(selected, start=1):
            record_id = str(record["id"])
            relative = _safe_relative_bundle_path(record_id)
            frozen_dir = destination_path / Path(relative)
            frozen_dir.mkdir(mode=0o700)
            for name, data in snapshot.files:
                target = frozen_dir / name
                target.write_bytes(data)
                target.chmod(0o400)
            frozen_dir.chmod(0o500)
            frozen_candidates.append(FrozenAlgorithmCandidate(
                rank=rank,
                record_id=record_id,
                search_score=float(record["score"]),
                algorithm_bundle_sha256=(
                    snapshot.bundle.algorithm_bundle_sha256
                ),
                frozen_bundle=relative,
                record_provenance=snapshot.bundle.record_provenance,
                source_provenance=snapshot.bundle.source_provenance,
            ))
        candidates_root.chmod(0o500)
        manifest = AlgorithmFreezeManifest(
            schema=ALGORITHM_FREEZE_MANIFEST_SCHEMA,
            frozen_at=timestamp,
            run_manifest_sha256=run_manifest_sha256,
            direction=direction,
            requested_top_k=top_k,
            eb_snapshot_version=eb_snapshot_version,
            eb_snapshot_sha256=eb_snapshot_sha256,
            bundle_subdir=canonical_subdir,
            candidate_count=len(frozen_candidates),
            candidates=tuple(frozen_candidates),
        )
        # This file is the completion marker.  It is deliberately written only
        # after every candidate byte has been copied and sealed read-only.
        _write_json(destination_path / "manifest.json", manifest.to_dict())
        destination_path.chmod(0o500)
        # Re-open the completed freeze through the same hostile-input path that
        # later audit orchestration will use.  Return only after it is verifiable.
        return verify_frozen_algorithm_bundles(
            destination_path,
            records=materialized,
            expected_manifest_sha256=manifest.sha256,
            expected_run_manifest_sha256=run_manifest_sha256,
            expected_destination_root=expected_destination_root,
            max_bundle_bytes=max_bundle_bytes,
        )
    except Exception:
        if created:
            destination_path.chmod(0o700)
            for path in destination_path.rglob("*"):
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(destination_path, ignore_errors=True)
        raise


def _parse_file_record(payload: object) -> AlgorithmFileRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("algorithm file record must be an object")
    _require_exact_fields(payload, {"path", "size_bytes", "sha256"},
                          "algorithm file record")
    path = payload["path"]
    if path not in ALLOWED_ALGORITHM_FILES:
        raise ValueError("algorithm file path is not declared by the bundle schema")
    size = payload["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("algorithm file size_bytes must be non-negative")
    return AlgorithmFileRecord(
        path=path,
        size_bytes=size,
        sha256=_require_sha256(payload["sha256"], "algorithm file sha256"),
    )


def _parse_record_provenance(payload: object) -> RecordProvenance:
    if not isinstance(payload, Mapping):
        raise ValueError("record provenance must be an object")
    _require_exact_fields(
        payload, {"record_id", "record_sha256", "record_path"},
        "record provenance",
    )
    record_id = payload["record_id"]
    if not isinstance(record_id, str) or not SAFE_RECORD_ID.fullmatch(record_id):
        raise ValueError("record provenance id is unsafe")
    record_path = payload["record_path"]
    if not isinstance(record_path, str) or not Path(record_path).is_absolute():
        raise ValueError("record provenance path must be absolute")
    return RecordProvenance(
        record_id=record_id,
        record_sha256=_require_sha256(
            payload["record_sha256"], "record provenance sha256",
        ),
        record_path=record_path,
    )


def _parse_source_provenance(payload: object) -> SourceProvenance:
    if not isinstance(payload, Mapping):
        raise ValueError("source provenance must be an object")
    _require_exact_fields(payload, {
        "source_path", "declared_algorithm_bundle_sha256", "total_bytes", "files",
    }, "source provenance")
    source_path = payload["source_path"]
    if not isinstance(source_path, str) or not Path(source_path).is_absolute():
        raise ValueError("source provenance path must be absolute")
    declared = payload["declared_algorithm_bundle_sha256"]
    declared = _require_sha256(declared, "declared algorithm bundle sha256")
    total = payload["total_bytes"]
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("source provenance total_bytes must be non-negative")
    raw_files = payload["files"]
    if not isinstance(raw_files, list):
        raise ValueError("source provenance files must be an array")
    files = tuple(_parse_file_record(item) for item in raw_files)
    paths = tuple(item.path for item in files)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("source provenance files must be unique and sorted")
    if not REQUIRED_ALGORITHM_FILES.issubset(paths):
        raise ValueError("source provenance omits a required algorithm file")
    if sum(item.size_bytes for item in files) != total:
        raise ValueError("source provenance total_bytes does not match its files")
    return SourceProvenance(source_path, declared, total, files)


def _parse_frozen_candidate(payload: object) -> FrozenAlgorithmCandidate:
    if not isinstance(payload, Mapping):
        raise ValueError("frozen algorithm candidate must be an object")
    _require_exact_fields(payload, {
        "rank", "record_id", "search_score", "algorithm_bundle_sha256",
        "frozen_bundle", "record_provenance", "source_provenance",
    }, "frozen algorithm candidate")
    rank = payload["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("candidate rank must be positive")
    record_id = payload["record_id"]
    if not isinstance(record_id, str) or not SAFE_RECORD_ID.fullmatch(record_id):
        raise ValueError("frozen candidate record id is unsafe")
    score = payload["search_score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise ValueError("candidate search_score must be finite")
    frozen_bundle = payload["frozen_bundle"]
    expected_path = _safe_relative_bundle_path(record_id)
    if frozen_bundle != expected_path:
        raise ValueError("frozen candidate bundle path is not canonical")
    record_provenance = _parse_record_provenance(payload["record_provenance"])
    if record_provenance.record_id != record_id:
        raise ValueError("candidate and record provenance ids differ")
    source_provenance = _parse_source_provenance(payload["source_provenance"])
    bundle_hash = _require_sha256(
        payload["algorithm_bundle_sha256"], "algorithm_bundle_sha256",
    )
    if _bundle_digest(source_provenance.files) != bundle_hash:
        raise ValueError("candidate bundle digest does not match its file manifest")
    if (
        source_provenance.declared_algorithm_bundle_sha256 is not None
        and source_provenance.declared_algorithm_bundle_sha256 != bundle_hash
    ):
        raise ValueError("declared candidate digest differs from frozen digest")
    return FrozenAlgorithmCandidate(
        rank=rank,
        record_id=record_id,
        search_score=float(score),
        algorithm_bundle_sha256=bundle_hash,
        frozen_bundle=frozen_bundle,
        record_provenance=record_provenance,
        source_provenance=source_provenance,
    )


def validate_freeze_manifest(payload: Mapping[str, Any] | AlgorithmFreezeManifest
                             ) -> AlgorithmFreezeManifest:
    """Validate the exact v1 freeze-manifest schema and ordering invariants."""
    if isinstance(payload, AlgorithmFreezeManifest):
        payload = payload.to_dict()
    if not isinstance(payload, Mapping):
        raise ValueError("algorithm freeze manifest must be an object")
    _require_exact_fields(payload, {
        "schema", "frozen_at", "run_manifest_sha256", "direction",
        "requested_top_k", "eb_snapshot_version", "eb_snapshot_sha256",
        "bundle_subdir", "candidate_count", "candidates",
    }, "algorithm freeze manifest")
    if payload["schema"] != ALGORITHM_FREEZE_MANIFEST_SCHEMA:
        raise ValueError("unsupported algorithm freeze manifest schema")
    frozen_at = payload["frozen_at"]
    if not isinstance(frozen_at, str) or not frozen_at or len(frozen_at) > 128:
        raise ValueError("freeze manifest frozen_at must be bounded text")
    run_hash = _require_sha256(
        payload["run_manifest_sha256"], "run_manifest_sha256",
    )
    direction = payload["direction"]
    if direction not in {"min", "max"}:
        raise ValueError("freeze manifest direction must be min or max")
    requested_top_k = payload["requested_top_k"]
    if (
        isinstance(requested_top_k, bool)
        or not isinstance(requested_top_k, int)
        or requested_top_k < 1
    ):
        raise ValueError("freeze manifest requested_top_k must be positive")
    snapshot_version = payload["eb_snapshot_version"]
    if (
        isinstance(snapshot_version, bool)
        or not isinstance(snapshot_version, int)
        or snapshot_version < 0
    ):
        raise ValueError("freeze manifest EB snapshot version must be non-negative")
    snapshot_hash = _require_sha256(
        payload["eb_snapshot_sha256"], "eb_snapshot_sha256",
    )
    bundle_subdir = _validate_bundle_subdir(payload["bundle_subdir"])
    count = payload["candidate_count"]
    raw_candidates = payload["candidates"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("freeze manifest candidate_count must be positive")
    if count > requested_top_k:
        raise ValueError("freeze manifest candidate_count exceeds requested_top_k")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != count:
        raise ValueError("freeze manifest candidate_count mismatch")
    candidates = tuple(_parse_frozen_candidate(item) for item in raw_candidates)
    if tuple(item.rank for item in candidates) != tuple(range(1, count + 1)):
        raise ValueError("freeze manifest ranks must be contiguous and ordered")
    ids = [item.record_id for item in candidates]
    hashes = [item.algorithm_bundle_sha256 for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("freeze manifest contains duplicate record ids")
    if len(hashes) != len(set(hashes)):
        raise ValueError("freeze manifest contains duplicate algorithms")
    return AlgorithmFreezeManifest(
        schema=ALGORITHM_FREEZE_MANIFEST_SCHEMA,
        frozen_at=frozen_at,
        run_manifest_sha256=run_hash,
        direction=direction,
        requested_top_k=requested_top_k,
        eb_snapshot_version=snapshot_version,
        eb_snapshot_sha256=snapshot_hash,
        bundle_subdir=bundle_subdir,
        candidate_count=count,
        candidates=candidates,
    )


def _load_strict_json_file(path: Path, max_bytes: int, label: str) -> Mapping[str, Any]:
    data = read_regular_file(path, max_bytes, label=label)
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def verify_frozen_algorithm_bundles(
        destination: os.PathLike[str] | str,
        *,
        records: Iterable[Mapping[str, Any]] | Any,
        expected_manifest_sha256: str,
        expected_run_manifest_sha256: str,
        expected_destination_root: os.PathLike[str] | str,
        max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
        max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
        expected_source_root: os.PathLike[str] | str | None = None,
        verify_source_unchanged: bool = False) -> AlgorithmFreezeManifest:
    """Re-verify a freeze and optionally detect source changes since freezing.

    All selection and run anchors are external to the freeze directory, so an
    attacker cannot make rewritten candidate bytes self-consistent merely by
    replacing the local manifest.
    """
    root = _validate_destination_path(destination, expected_destination_root)
    try:
        root_info = os.lstat(root)
    except FileNotFoundError as exc:
        raise ValueError("algorithm freeze directory not found") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("algorithm freeze must be a real directory")
    payload = _load_strict_json_file(
        root / "manifest.json", max_manifest_bytes, "algorithm freeze manifest",
    )
    manifest = validate_freeze_manifest(payload)
    expected = _require_sha256(
        expected_manifest_sha256, "expected_manifest_sha256",
    )
    expected_run = _require_sha256(
        expected_run_manifest_sha256, "expected_run_manifest_sha256",
    )
    if manifest.sha256 != expected:
        raise ValueError("algorithm freeze manifest digest mismatch")
    if manifest.run_manifest_sha256 != expected_run:
        raise ValueError("algorithm freeze run manifest digest mismatch")

    if hasattr(records, "snapshot") and callable(records.snapshot):
        observed_version, records = records.snapshot()
        if observed_version != manifest.eb_snapshot_version:
            raise ValueError("EB changed after the frozen snapshot")
    elif hasattr(records, "records") and callable(records.records):
        records = records.records()
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Iterable):
        raise ValueError("records must be an iterable of Experience Bank records")
    materialized = list(records)
    if compute_eb_snapshot_sha256(
            materialized, manifest.eb_snapshot_version,
    ) != manifest.eb_snapshot_sha256:
        raise ValueError("EB snapshot digest mismatch")
    _validate_snapshot_run_provenance(materialized, expected_run)
    expected_records = _select_top_k_declared(
        materialized,
        direction=manifest.direction,
        top_k=manifest.requested_top_k,
        run_manifest_sha256=expected_run,
    )
    expected_selection = [
        (
            rank,
            str(record["id"]),
            float(record["score"]),
            _declared_algorithm_hash(record),
            _sha256_json(dict(record)),
        )
        for rank, record in enumerate(expected_records, start=1)
    ]
    actual_selection = [
        (
            item.rank,
            item.record_id,
            item.search_score,
            item.algorithm_bundle_sha256,
            item.record_provenance.record_sha256,
        )
        for item in manifest.candidates
    ]
    if actual_selection != expected_selection:
        raise ValueError("algorithm freeze does not match the EB Top-K selection")

    root_names = {entry.name for entry in os.scandir(root)}
    if root_names != {"manifest.json", "candidates"}:
        raise ValueError("algorithm freeze contains undeclared root entries")
    candidates_root = root / "candidates"
    candidates_info = os.lstat(candidates_root)
    if stat.S_ISLNK(candidates_info.st_mode) or not stat.S_ISDIR(candidates_info.st_mode):
        raise ValueError("frozen candidates path must be a real directory")
    expected_ids = {item.record_id for item in manifest.candidates}
    actual_ids = {entry.name for entry in os.scandir(candidates_root)}
    if actual_ids != expected_ids:
        raise ValueError("frozen candidates differ from the freeze manifest")

    for candidate in manifest.candidates:
        bundle_dir = root / Path(candidate.frozen_bundle)
        bundle_info = os.lstat(bundle_dir)
        if stat.S_ISLNK(bundle_info.st_mode) or not stat.S_ISDIR(bundle_info.st_mode):
            raise ValueError(
                f"{candidate.record_id} frozen bundle must be a real directory"
            )
        snapshot_signature = _source_signature(bundle_dir)
        expected_names = {item.path for item in candidate.source_provenance.files}
        actual_names = {item[0] for item in snapshot_signature}
        if actual_names != expected_names:
            raise ValueError(f"{candidate.record_id} frozen files changed")
        total = 0
        observed_files: list[AlgorithmFileRecord] = []
        for name, _dev, _ino, size, _mtime in snapshot_signature:
            remaining = max_bundle_bytes - total
            if size > remaining:
                raise ValueError("frozen algorithm bundle exceeds its byte limit")
            data = read_regular_file(
                bundle_dir / name,
                remaining,
                label=f"frozen algorithm file {candidate.record_id}/{name}",
            )
            total += len(data)
            observed_files.append(AlgorithmFileRecord(
                name, len(data), hashlib.sha256(data).hexdigest(),
            ))
        observed = tuple(sorted(observed_files, key=lambda item: item.path))
        if snapshot_signature != _source_signature(bundle_dir):
            raise ValueError(
                f"{candidate.record_id} changed while its freeze was verified"
            )
        if observed != candidate.source_provenance.files:
            raise ValueError(f"{candidate.record_id} changed after algorithm freeze")
        if _bundle_digest(observed) != candidate.algorithm_bundle_sha256:
            raise ValueError(f"{candidate.record_id} bundle digest mismatch")
        if verify_source_unchanged:
            if expected_source_root is None:
                raise ValueError(
                    "expected_source_root is required to re-verify EB sources"
                )
            source_record = next(
                record for record in expected_records
                if record["id"] == candidate.record_id
            )
            current = _read_bundle_snapshot(
                source_record,
                expected_source_root=expected_source_root,
                bundle_subdir=manifest.bundle_subdir,
                expected_run_manifest_sha256=expected_run,
                max_bundle_bytes=max_bundle_bytes,
                require_declared_bundle_hash=True,
            ).bundle
            if current.algorithm_bundle_sha256 != candidate.algorithm_bundle_sha256:
                raise ValueError(f"{candidate.record_id} source changed after freeze")
    return manifest


def build_per_cell_provenance(
        *,
        freeze_manifest_sha256: str,
        run_manifest_sha256: str,
        evaluation_request_sha256: str,
        trusted_runner_sha256: str,
        algorithm_bundle_sha256: str,
        instance_sha256: str,
        training_input_sha256: str,
        train_seed: int,
        runtime_sha256: str,
        policy_artifact_sha256: str,
        stage: str,
        suite: str,
        instance: str,
        repeat: int) -> PerCellPolicyProvenance:
    """Build the immutable lineage record for one instance/repeat policy."""
    if stage not in VALID_STAGES:
        raise ValueError("per-cell stage must be search or audit")
    suite = _require_safe_text(suite, "suite")
    instance = _require_safe_text(instance, "instance")
    if (
        isinstance(repeat, bool)
        or not isinstance(repeat, int)
        or repeat < 0
        or repeat > MAX_SAFE_INTEGER
    ):
        raise ValueError("repeat must be a non-negative 63-bit integer")
    if (
        isinstance(train_seed, bool)
        or not isinstance(train_seed, int)
        or not 0 <= train_seed <= MAX_SAFE_INTEGER
    ):
        raise ValueError("train_seed must be a non-negative 63-bit integer")
    return PerCellPolicyProvenance(
        schema=PER_CELL_POLICY_PROVENANCE_SCHEMA,
        stage=stage,
        suite=suite,
        instance=instance,
        repeat=repeat,
        freeze_manifest_sha256=_require_sha256(
            freeze_manifest_sha256, "freeze_manifest_sha256",
        ),
        run_manifest_sha256=_require_sha256(
            run_manifest_sha256, "run_manifest_sha256",
        ),
        evaluation_request_sha256=_require_sha256(
            evaluation_request_sha256, "evaluation_request_sha256",
        ),
        trusted_runner_sha256=_require_sha256(
            trusted_runner_sha256, "trusted_runner_sha256",
        ),
        algorithm_bundle_sha256=_require_sha256(
            algorithm_bundle_sha256, "algorithm_bundle_sha256",
        ),
        instance_sha256=_require_sha256(instance_sha256, "instance_sha256"),
        training_input_sha256=_require_sha256(
            training_input_sha256, "training_input_sha256",
        ),
        train_seed=train_seed,
        runtime_sha256=_require_sha256(runtime_sha256, "runtime_sha256"),
        policy_artifact_sha256=_require_sha256(
            policy_artifact_sha256, "policy_artifact_sha256",
        ),
    )


def validate_per_cell_provenance(
        payload: Mapping[str, Any] | PerCellPolicyProvenance,
        *,
        expected_stage: str,
        expected_suite: str,
        expected_instance: str,
        expected_repeat: int,
        expected_train_seed: int,
        expected_freeze_manifest_sha256: str,
        expected_run_manifest_sha256: str,
        expected_evaluation_request_sha256: str,
        expected_trusted_runner_sha256: str,
        expected_algorithm_bundle_sha256: str,
        expected_instance_sha256: str,
        expected_training_input_sha256: str,
        expected_runtime_sha256: str,
        expected_policy_artifact_sha256: str,
        ) -> PerCellPolicyProvenance:
    """Validate the exact per-cell schema against mandatory external anchors."""
    if isinstance(payload, PerCellPolicyProvenance):
        payload = payload.to_dict()
    if not isinstance(payload, Mapping):
        raise ValueError("per-cell policy provenance must be an object")
    _require_exact_fields(payload, {
        "schema", "stage", "suite", "instance", "repeat",
        "freeze_manifest_sha256", "run_manifest_sha256",
        "evaluation_request_sha256", "trusted_runner_sha256",
        "algorithm_bundle_sha256", "instance_sha256",
        "training_input_sha256", "train_seed", "runtime_sha256",
        "policy_artifact_sha256",
    }, "per-cell policy provenance")
    if payload["schema"] != PER_CELL_POLICY_PROVENANCE_SCHEMA:
        raise ValueError("unsupported per-cell policy provenance schema")
    record = build_per_cell_provenance(
        freeze_manifest_sha256=payload["freeze_manifest_sha256"],
        run_manifest_sha256=payload["run_manifest_sha256"],
        evaluation_request_sha256=payload["evaluation_request_sha256"],
        trusted_runner_sha256=payload["trusted_runner_sha256"],
        algorithm_bundle_sha256=payload["algorithm_bundle_sha256"],
        instance_sha256=payload["instance_sha256"],
        training_input_sha256=payload["training_input_sha256"],
        train_seed=payload["train_seed"],
        runtime_sha256=payload["runtime_sha256"],
        policy_artifact_sha256=payload["policy_artifact_sha256"],
        stage=payload["stage"],
        suite=payload["suite"],
        instance=payload["instance"],
        repeat=payload["repeat"],
    )
    if expected_stage not in VALID_STAGES:
        raise ValueError("expected_stage must be search or audit")
    expected_suite = _require_safe_text(expected_suite, "expected_suite")
    expected_instance = _require_safe_text(expected_instance, "expected_instance")
    if (
        isinstance(expected_repeat, bool)
        or not isinstance(expected_repeat, int)
        or not 0 <= expected_repeat <= MAX_SAFE_INTEGER
    ):
        raise ValueError("expected_repeat must be a non-negative 63-bit integer")
    if (
        isinstance(expected_train_seed, bool)
        or not isinstance(expected_train_seed, int)
        or not 0 <= expected_train_seed <= MAX_SAFE_INTEGER
    ):
        raise ValueError("expected_train_seed must be a non-negative 63-bit integer")
    cell_coordinates = {
        "stage": (record.stage, expected_stage),
        "suite": (record.suite, expected_suite),
        "instance": (record.instance, expected_instance),
        "repeat": (record.repeat, expected_repeat),
        "train_seed": (record.train_seed, expected_train_seed),
    }
    for label, (observed, expected_value) in cell_coordinates.items():
        if observed != expected_value:
            raise ValueError(f"per-cell {label} binding mismatch")
    anchors = {
        "freeze manifest": (
            record.freeze_manifest_sha256, expected_freeze_manifest_sha256,
        ),
        "run manifest": (record.run_manifest_sha256, expected_run_manifest_sha256),
        "evaluation request": (
            record.evaluation_request_sha256, expected_evaluation_request_sha256,
        ),
        "trusted runner": (
            record.trusted_runner_sha256, expected_trusted_runner_sha256,
        ),
        "algorithm bundle": (
            record.algorithm_bundle_sha256, expected_algorithm_bundle_sha256,
        ),
        "instance": (record.instance_sha256, expected_instance_sha256),
        "training input": (
            record.training_input_sha256, expected_training_input_sha256,
        ),
        "runtime": (record.runtime_sha256, expected_runtime_sha256),
        "policy artifact": (
            record.policy_artifact_sha256, expected_policy_artifact_sha256,
        ),
    }
    for label, (observed, expected_value) in anchors.items():
        expected_digest = _require_sha256(
            expected_value, f"expected_{label.replace(' ', '_')}_sha256",
        )
        if observed != expected_digest:
            raise ValueError(f"per-cell {label} provenance mismatch")
    return record


# Concise integration aliases.  The explicit names above remain the canonical
# API; these make call sites read naturally in audit orchestration code.
freeze_candidate_algorithm_bundles = freeze_top_k_algorithm_bundles
verify_algorithm_freeze = verify_frozen_algorithm_bundles
build_per_cell_policy_provenance = build_per_cell_provenance
validate_per_cell_policy_provenance = validate_per_cell_provenance
