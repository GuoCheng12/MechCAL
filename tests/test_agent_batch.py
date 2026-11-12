"""Batch runner: finished cases are skipped, interrupted ones are set aside and rerun."""

import json
import subprocess

from mechcal.agent import batch


def test_batch_skips_finished_and_reruns_incomplete(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = __import__("pathlib").Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True)
        (out / "result.json").write_text(json.dumps({"status": "submitted"}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(batch.subprocess, "run", fake_run)
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test")
    (tmp_path / "0047").mkdir()
    (tmp_path / "0047" / "result.json").write_text("{}")          # finished before
    (tmp_path / "0020").mkdir()                                     # interrupted: no result.json
    rows = batch.main(["--cases", "0047,0020,68", "--out", str(tmp_path), "--parallel", "2", "--", "--max-computations", "5"])
    states = {r["case"]: r["state"] for r in rows}
    assert states == {"0047": "skipped (already finished)", "0020": "finished", "0068": "finished"}
    assert len(calls) == 2 and all(c[-2:] == ["--max-computations", "5"] for c in calls)
    assert any(p.name.startswith("0020-incomplete-") for p in tmp_path.iterdir())
