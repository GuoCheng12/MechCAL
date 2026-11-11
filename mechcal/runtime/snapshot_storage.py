"""Lossless, content-addressed bulk arrays outside frequently written snapshots.

Only the on-disk representation changes. Runtime records and JSON pointers use
the complete values. Missing/corrupt payloads fail closed; they are never empty
observations. Small and historical self-contained JSON files remain unchanged.
"""

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from mechcal.runtime.progress import atomic_json

STORAGE_KEY = "_mechcal_storage"
ARRAY_BYTES = 64 * 1024
ARRAY_ITEMS = 128
_ANY = TypeAdapter(object)


def _pointer(path, key):
    return path + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _blob_path(root, key):
    if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise ValueError("Invalid snapshot payload digest")
    root = Path(root).resolve()
    path = root / "_payloads" / (key + ".json")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Snapshot payload escapes its run directory")
    return path


class SnapshotStorage:
    def __init__(self, root, case_id):
        self.root = Path(root)
        self.case_id = case_id
        self._verified = {}
        self._batch = None

    @contextmanager
    def batch(self):
        """Share packing only during one immutable state publication.

        Never retain object-identity caches between operations: accepted edits can
        change nested values. The caller must not mutate state inside this scope.
        """
        if self._batch is not None:
            raise RuntimeError("Snapshot publication batches cannot be nested")
        self._batch = {"arrays": {}, "roots": {}}
        try:
            yield
        finally:
            self._batch = None

    def _write_array(self, value):
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          default=lambda x: _ANY.dump_python(x, mode="json")).encode()
        if len(data) < ARRAY_BYTES:
            return None
        key = hashlib.sha256(data).hexdigest()
        path = _blob_path(self.root, key)
        if path.exists():
            stat = path.stat()
            stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if self._verified.get(key) != stamp and path.read_bytes() != data:
                raise ValueError("Existing snapshot payload conflicts with its digest")
        else:
            # Publish the complete array before any checkpoint can reference it.
            # Preserve the bytes whose digest is recorded in the manifest.
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".payload-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                Path(temporary).replace(path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        stat = path.stat()
        self._verified[key] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return {"sha256": key, "bytes": len(data)}

    def pack(self, payload):
        roots = self._batch["roots"] if self._batch is not None else {}
        if id(payload) in roots:
            return roots[id(payload)][1]
        arrays = {}
        seen = self._batch["arrays"] if self._batch is not None else {}

        def visit(value, path):
            if isinstance(value, BaseModel):
                # Avoid model_dump copying every bulk array before it is packed.
                nested = {name for name in type(value).model_fields
                          if isinstance(getattr(value, name), (BaseModel, dict, list))}
                result = value.model_dump(mode="json", exclude=nested)
                result.update({name: visit(getattr(value, name), _pointer(path, name)) for name in nested})
                return result
            if isinstance(value, list):
                if len(value) >= ARRAY_ITEMS:
                    key = id(value)
                    if key not in seen:
                        seen[key] = (value, self._write_array(value))
                    entry = seen[key][1]
                    if entry is not None:
                        arrays[path] = entry
                        return None
                return [visit(child, _pointer(path, i)) for i, child in enumerate(value)]
            if isinstance(value, dict):
                return {key: visit(child, _pointer(path, key)) for key, child in value.items()}
            return _ANY.dump_python(value, mode="json")

        result = visit(payload, "")
        if not isinstance(result, dict) or STORAGE_KEY in result:
            raise ValueError("Snapshot must be an unpacked JSON object")
        if arrays:
            if result.get("case_id") != self.case_id:
                raise ValueError("Snapshot payload belongs to another case")
            result[STORAGE_KEY] = {"version": 1, "case_id": self.case_id, "arrays": arrays}
        roots[id(payload)] = (payload, result)
        return result

    def write(self, path, payload):
        atomic_json(Path(path), self.pack(payload))


def read_json(path, *, blob_root=None):
    """Read old JSON or restore exact arrays; no model or filesystem search."""
    path = Path(path)
    return decode_json(path.read_bytes(), blob_root=blob_root if blob_root is not None else path.parent)


def decode_json(data, *, blob_root):
    """Hydrate the exact bytes used by a caller's provenance checksum."""
    value = json.loads(data)
    if not isinstance(value, dict) or STORAGE_KEY not in value:
        return value
    storage = value.pop(STORAGE_KEY)
    if (not isinstance(storage, dict) or storage.get("version") != 1
            or storage.get("case_id") != value.get("case_id")
            or not isinstance(storage.get("arrays"), dict)):
        raise ValueError("Invalid snapshot storage version or case ownership")
    root, loaded = Path(blob_root), {}
    pointers = storage["arrays"]
    if not all(isinstance(p, str) for p in pointers):
        raise ValueError("Invalid snapshot array path")
    for pointer, entry in pointers.items():
        if not isinstance(pointer, str) or not pointer.startswith("/") or not isinstance(entry, dict):
            raise ValueError("Invalid snapshot array reference")
        parts = pointer[1:].split("/")
        if (any(re.search(r"~(?![01])", part) for part in parts)
                or any("/" + "/".join(parts[:i]) in pointers for i in range(1, len(parts)))):
            raise ValueError("Overlapping or invalid snapshot array paths")
        key = entry.get("sha256")
        if key not in loaded:
            data = _blob_path(root, key).read_bytes()
            if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != key:
                raise ValueError("Snapshot array checksum or size mismatch")
            array = json.loads(data)
            if not isinstance(array, list):
                raise ValueError("Snapshot payload is not an array")
            loaded[key] = (array, len(data))
        array, size = loaded[key]
        if size != entry.get("bytes"):
            raise ValueError("Conflicting snapshot array size")
        parent = value
        parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")]
        def key_for(container, part):
            if not isinstance(container, list):
                return part
            if not part.isdigit() or str(int(part)) != part:
                raise ValueError("Noncanonical snapshot array index")
            return int(part)
        try:
            for part in parts[:-1]:
                parent = parent[key_for(parent, part)]
            leaf = key_for(parent, parts[-1])
            if parent[leaf] is not None:
                raise ValueError("Snapshot array placeholder conflicts with stored content")
            parent[leaf] = array
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Snapshot array path is unavailable") from exc
    return value


def read_case_run(path):
    from mechcal.schemas import CaseRun
    return CaseRun.model_validate(read_json(path))
