import hashlib

import pytest

import object_store
from object_store import ObjectStore


def test_put_bytes_and_get(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    data = b"OpenHyra experience"

    digest = store.put_bytes(data, "experience.bin")
    path = store.get_path(digest, "experience.bin")

    assert digest == hashlib.sha256(data).hexdigest()
    assert path is not None
    assert path.read_bytes() == data
    assert store.get_path(digest) == path.parent


def test_put_file(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"candidate result")
    store = ObjectStore(tmp_path / "objects")

    digest = store.put_file(source)
    stored = store.get_path(digest, source.name)

    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert stored is not None
    assert stored.read_bytes() == source.read_bytes()


def test_put_json(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    digest = store.put_json({"z": "\u7814\u7a76", "a": 1}, "record.json")

    path = store.get_path(digest, "record.json")
    assert path is not None
    assert path.read_bytes() == '{"a":1,"z":"\u7814\u7a76"}'.encode()


def test_dedup(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    first = store.put_bytes(b"same content", "first.bin")
    second = store.put_bytes(b"same content", "second.bin")

    assert first == second
    object_directory = store.get_path(first)
    assert object_directory is not None
    assert [path.name for path in object_directory.iterdir()] == ["first.bin"]


def test_exists_and_verify(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    digest = store.put_bytes(b"trusted", "value.bin")
    missing = "0" * 64

    assert store.exists(digest)
    assert store.verify(digest)
    assert not store.exists(missing)
    assert not store.verify(missing)

    path = store.get_path(digest, "value.bin")
    assert path is not None
    path.write_bytes(b"corrupted")
    assert not store.verify(digest)


def test_delete_raises(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    with pytest.raises(RuntimeError, match="immutable"):
        store.delete("a" * 64)


@pytest.mark.parametrize("digest", ["abc", "g" * 64, "A" * 64, "0" * 63])
@pytest.mark.parametrize("method", ["get_path", "exists", "verify", "delete"])
def test_invalid_digest(tmp_path, digest, method):
    store = ObjectStore(tmp_path / "objects")
    with pytest.raises(ValueError):
        getattr(store, method)(digest)


def test_list_objects(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    expected = {
        store.put_bytes(b"one", "one.bin"),
        store.put_bytes(b"two", "two.bin"),
        store.put_bytes(b"three", "three.bin"),
    }

    assert set(store.list_objects()) == expected


def test_atomic_write(tmp_path, monkeypatch):
    store = ObjectStore(tmp_path / "objects")

    def interrupted_rename(source, destination):
        raise OSError("interrupted")

    monkeypatch.setattr(object_store.os, "rename", interrupted_rename)
    with pytest.raises(OSError, match="interrupted"):
        store.put_bytes(b"never committed", "value.bin")

    assert not any(path.is_file() for path in store.root.rglob("*"))
    assert store.list_objects() == []
