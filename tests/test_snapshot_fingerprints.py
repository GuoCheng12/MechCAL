import pytest

from mechcal.runtime import command_operations as ops
from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_review_workflow import binding
from test_command_protocol import case


def test_source_digest_computed_once_per_snapshot_not_per_issue(monkeypatch):
    c = ops.put(case(), results={"R1": {"status": "success", "values": list(range(4000))}})
    index = ops.record_index(c)
    calls = []
    original = ops.digest
    monkeypatch.setattr(ops, "digest", lambda value: (calls.append(value), original(value))[1])
    for _ in range(20):
        assert binding("input:smiles", ["R1"], index.available, index=index) == {
            "input:smiles": digest(index.available["input:smiles"]), "R1": digest(index.available["R1"])}
    assert len(calls) == 2
    updated = ops.put(c, results={"R1": {"status": "success", "values": [99]}})
    fresh = ops.record_index(updated)
    assert fresh.version("R1") != index.version("R1")
    assert len(calls) == 3
    with pytest.raises(RuntimeError, match="snapshot"):
        ops.record_index(updated, index)
    with pytest.raises(RuntimeError, match="own"):
        binding("R1", [], fresh.available, index=index)


def test_case_local_memo_does_not_survive_state_update():
    c = case()
    old = ops.record_index(c)
    assert old.memo("test", lambda: 1) == old.memo("test", lambda: 2) == 1
    assert ops.record_index(ops.put(c, revision=1)).memo("test", lambda: 2) == 2
