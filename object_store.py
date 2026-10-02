from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class ObjectStore:
    """Content-addressed, immutable storage for Experience Bank objects."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("digest must be 64 lowercase hexadecimal characters")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or filename.startswith(".tmp-")
            or Path(filename).name != filename
        ):
            raise ValueError("filename must be a non-empty basename")

    def _object_dir(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / digest

    def _object_files(self, digest: str) -> list[Path]:
        directory = self._object_dir(digest)
        if not directory.is_dir():
            return []
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and not path.name.startswith(".tmp-")
        )

    def put_bytes(self, data: bytes, filename: str) -> str:
        digest = hashlib.sha256(data).hexdigest()
        self._validate_filename(filename)
        if self.exists(digest):
            return digest

        directory = self._object_dir(digest)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / filename
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=".tmp-", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.rename(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return digest

    def put_file(self, source_path: Path, filename: str | None = None) -> str:
        source = Path(source_path)
        stored_name = source.name if filename is None else filename
        return self.put_bytes(source.read_bytes(), stored_name)

    def put_json(self, obj: dict, filename: str) -> str:
        data = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return self.put_bytes(data, filename)

    def get_path(self, digest: str, filename: str | None = None) -> Path | None:
        self._validate_digest(digest)
        if filename is None:
            directory = self._object_dir(digest)
            return directory if self._object_files(digest) else None

        self._validate_filename(filename)
        path = self._object_dir(digest) / filename
        return path if path.is_file() else None

    def exists(self, digest: str) -> bool:
        self._validate_digest(digest)
        return bool(self._object_files(digest))

    def verify(self, digest: str) -> bool:
        self._validate_digest(digest)
        files = self._object_files(digest)
        return bool(files) and all(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest
            for path in files
        )

    def list_objects(self) -> list[str]:
        base = self.root / "sha256"
        if not base.is_dir():
            return []

        digests = []
        for shard in base.iterdir():
            if not shard.is_dir():
                continue
            for directory in shard.iterdir():
                digest = directory.name
                try:
                    self._validate_digest(digest)
                except ValueError:
                    continue
                if digest[:2] == shard.name and self._object_files(digest):
                    digests.append(digest)
        return sorted(digests)

    def delete(self, digest: str) -> None:
        self._validate_digest(digest)
        raise RuntimeError("objects are immutable and cannot be deleted")
