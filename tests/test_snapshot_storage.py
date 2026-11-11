import hashlib
import json
from copy import deepcopy

import pytest

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import put, record_index
from mechcal.runtime.command_loop import save
from mechcal.runtime.persistence import RunStore
from mechcal.runtime.snapshot_storage import STORAGE_KEY, read_case_run, read_json
from mechcal.schemas import CaseRun
from test_command_protocol import case


def large_case():
    c = case()
    rows = [{"orbital": i, "coefficient": i / 100003, "note": "literal source"} for i in range(4000)]
    return put(c, results={"result:large": {"status": "success", "structured_results": {"a/b~c": rows},
                                         "other_representation": deepcopy(rows)}})


def test_large_checkpoint_lossless_deduplicated_and_legacy_reads(tmp_path):
    c = large_case()
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    path = store.write_json("command_checkpoint.json", c)
    packed = json.loads(path.read_text())
    assert packed["runtime"]["command_protocol"]["phase"] == "choose"
    assert len(packed[STORAGE_KEY]["arrays"]) == 2
    assert len(list((store.root / "_payloads").glob("*.json"))) == 1
    assert path.stat().st_size < len(c.model_dump_json()) / 10
    assert read_case_run(path) == c
    assert digest(read_json(path)) == digest(c.model_dump(mode="json"))
    with pytest.raises(ValueError):
        CaseRun.model_validate(packed)  # Cannot silently treat a compact snapshot as evidence.
    legacy = store.root / "legacy.json"
    legacy.write_text(c.model_dump_json())
    assert read_case_run(legacy) == c


def test_save_and_resume_preserve_old_pointers_and_do_not_rewrite_arrays(tmp_path):
    store = RunStore(tmp_path, case().case_id)
    store.initialize()
    c = save(large_case(), store)
    blobs = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
             for p in (store.root / "_payloads").glob("*.json")}
    c = save(put(c, note="A prose-only update"), store)
    assert store.read_case_run("command_checkpoint.json") == c
    assert store.read_case_run() == c
    assert blobs == {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in blobs}
    restored = store.read_case_run()
    index = record_index(restored)
    assert index.available["result:large"]["structured_results"]["a/b~c"][3999]["orbital"] == 3999


def test_publication_reuses_one_snapshot_but_not_the_next_mutation(tmp_path, monkeypatch):
    store = RunStore(tmp_path, case().case_id)
    store.initialize()
    c = large_case()
    calls, original = [], store.snapshots._write_array
    monkeypatch.setattr(store.snapshots, "_write_array", lambda value: (calls.append(id(value)), original(value))[1])
    with store.snapshots.batch():
        store.write_json("command_checkpoint.json", c)
        store.write_case_run(c)
    assert len(calls) == 2  # Two source representations, not twice per projection.
    c.runtime["command_protocol"]["results"]["result:large"]["structured_results"]["a/b~c"][0]["orbital"] = -1
    with store.snapshots.batch():
        store.write_json("command_checkpoint.json", c)
    assert len(calls) == 4
    assert store.read_case_run("command_checkpoint.json") == c


def test_publication_failure_discards_identity_caches(tmp_path):
    store = RunStore(tmp_path, case().case_id)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        with store.snapshots.batch():
            store.snapshots.pack(large_case())
            raise RuntimeError("fixture interruption")
    assert store.snapshots._batch is None


def test_null_storage_manifest_is_corruption_not_legacy_json(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"case_id": "CASE", STORAGE_KEY: None}))
    with pytest.raises(ValueError, match="storage version"):
        read_json(path)


def test_foreign_case_cannot_publish_compact_snapshot(tmp_path):
    store = RunStore(tmp_path, "OTHER")
    store.initialize()
    with pytest.raises(ValueError, match="another case"):
        store.write_json("command_checkpoint.json", large_case())
    assert not (store.root / "command_checkpoint.json").exists()


@pytest.mark.parametrize("damage", ["missing", "corrupt", "owner", "version", "traversal", "size", "placeholder"])
def test_invalid_snapshot_is_never_empty_evidence(tmp_path, damage):
    c = large_case()
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    path = store.write_json("command_checkpoint.json", c)
    packed = json.loads(path.read_text())
    metadata = packed[STORAGE_KEY]
    entry = next(iter(metadata["arrays"].values()))
    blob = store.root / "_payloads" / (entry["sha256"] + ".json")
    if damage == "missing":
        blob.unlink()
    elif damage == "corrupt":
        blob.write_bytes(b"[]")
    elif damage == "owner":
        packed["case_id"] = "different-case"
    elif damage == "version":
        metadata["version"] = 99
    elif damage == "traversal":
        entry["sha256"] = "../../other-case"
    elif damage == "size":
        entry["bytes"] += 1
    else:
        packed["runtime"]["command_protocol"]["results"]["result:large"]["other_representation"] = []
    path.write_text(json.dumps(packed))
    with pytest.raises((ValueError, FileNotFoundError)):
        read_case_run(path)


def test_interrupted_publish_leaves_last_checkpoint_usable(tmp_path, monkeypatch):
    import mechcal.runtime.snapshot_storage as module
    store = RunStore(tmp_path, case().case_id)
    store.initialize()
    old = large_case()
    path = store.write_json("command_checkpoint.json", old)
    before = path.read_bytes()
    new = old.model_copy(deep=True)
    new.runtime["command_protocol"]["results"]["result:large"]["other_representation"][0]["coefficient"] = 9.5
    monkeypatch.setattr(module, "atomic_json", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.write_json("command_checkpoint.json", new)
    assert path.read_bytes() == before
    assert read_case_run(path) == old


def test_payload_symlink_cannot_escape_run(tmp_path):
    c = large_case()
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "_payloads").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        store.write_json("command_checkpoint.json", c)


def test_artifact_and_evidence_projections_use_same_reader(tmp_path):
    c = large_case()
    store = RunStore(tmp_path, c.case_id)
    store.initialize()
    body = {"case_id": c.case_id, "rows": c.runtime["command_protocol"]["results"]["result:large"]["other_representation"]}
    for name in ("artifacts/manifest.json", "evidence_units.json"):
        path = store.write_json(name, body)
        assert read_json(path, blob_root=store.root) == body


def test_snapshot_serialization_equals_standard_model_dump_for_small_case(tmp_path):
    c = case()
    store = RunStore(tmp_path, c.case_id)
    path = store.write_json("command_checkpoint.json", c)
    assert json.loads(path.read_text()) == c.model_dump(mode="json")
