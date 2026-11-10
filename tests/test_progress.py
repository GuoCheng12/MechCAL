import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from rich.console import Console

from mechcal.cli.monitor import demo
from mechcal.runtime.progress import (
    ExperimentProgress, CaseProgress, ProgressSnapshot, read_snapshot, render_progress, write_activity,
)


def rendered(snapshot, width=120, now=None):
    stream = io.StringIO()
    Console(file=stream, width=width, color_system=None).print(
        render_progress(snapshot, width=width, now=now)
    )
    return stream.getvalue()


def test_lifecycle_and_metrics(tmp_path):
    with ExperimentProgress(tmp_path, ["a", "b"], method="test", mode="off") as progress:
        progress.start("a")
        progress.update("a", "judging")
        assert read_snapshot(progress.path).cases[0].stage == "judging"
        progress.finish("a", scores={"recall_at_3": 0.666666, "es_ndcg_at_3": None})
        progress.start("b")
        progress.finish("b", status="cached")
    snapshot = read_snapshot(progress.path)
    assert snapshot.state == "completed"
    assert snapshot.cases[0].scores == {"recall_at_3": 0.666666}
    view = rendered(snapshot)
    assert "Case" in view and "Status" in view and "Stage / activity" in view
    assert "0.667" in view and "Finished 2/2" in view
    assert "Success 2" in view and "cached" in view
    assert not list(tmp_path.glob("*.tmp"))
    assert not progress._thread.is_alive()


def test_failure_does_not_swallow_exception_or_store_secret(tmp_path):
    with pytest.raises(ValueError):
        with ExperimentProgress(tmp_path, ["a", "b"], method="test", mode="off") as progress:
            progress.start("a")
            raise ValueError("secret-token-and-prompt")
    snapshot = read_snapshot(progress.path)
    assert snapshot.state == "failed"
    assert [row.status for row in snapshot.cases] == ["failed", "not_run"]
    assert "secret-token" not in progress.path.read_text()


def test_keyboard_interrupt(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with ExperimentProgress(tmp_path, ["a"], method="test", mode="off") as progress:
            with progress.case("a"):
                raise KeyboardInterrupt()
    assert read_snapshot(progress.path).cases[0].status == "interrupted"


def test_parallel_cases_and_atomic_reads(tmp_path):
    ids = [str(index) for index in range(10)]
    with ExperimentProgress(tmp_path, ids, method="parallel", mode="off") as progress:
        def run(case_id):
            with progress.case(case_id):
                progress.update(case_id, "judging")
                assert len(read_snapshot(progress.path).cases) == 10
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run, ids))
    assert all(row.status == "done" for row in read_snapshot(progress.path).cases)


def test_one_parallel_failure_does_not_mark_successes_failed(tmp_path):
    with ExperimentProgress(tmp_path, ["a", "b"], method="test", mode="off") as progress:
        with pytest.raises(RuntimeError):
            with progress.case("a"):
                raise RuntimeError("not persisted")
        with progress.case("b"):
            pass
    snapshot = read_snapshot(progress.path)
    assert snapshot.state == "completed_with_errors"
    assert [row.status for row in snapshot.cases] == ["failed", "done"]


def test_mechcal_activity_and_judge_phase(tmp_path):
    with ExperimentProgress(tmp_path, ["a"], method="MechCAL", max_rounds=30, mode="off") as progress:
        run = tmp_path / "mas_runs" / "a"
        progress.start("a", run_dir=run)
        write_activity(run, "Microscopic", "R003", "torsion scan")
        view = rendered(progress.snapshot)
        assert "Microscopic" in view and "R003/30" in view
        assert "Finished 0/1" in view
        progress.update("a", "judging")
        view = rendered(progress.snapshot)
        assert "judging" in view and "Microscopic" not in view
        progress.finish("a")
        assert "R003/30" in rendered(progress.snapshot)


def test_stale_or_corrupt_activity_not_reused(tmp_path):
    run = tmp_path / "run"
    write_activity(run, "Old activity", "R030")
    with ExperimentProgress(tmp_path, ["a"], method="test", mode="off") as progress:
        progress.start("a", run_dir=run)
        assert "Old activity" not in rendered(progress.snapshot)
        (run / "activity.json").write_text("{")
        assert "inference" in rendered(progress.snapshot)
        progress.finish("a")


def test_stale_heartbeat_is_unknown_not_success_or_failure():
    snapshot = ProgressSnapshot(method="test", started_at=100, updated_at=110,
        cases=[CaseProgress(case_id="a", status="running", started_at=100)])
    view = rendered(snapshot, now=200)
    assert "No heartbeat" in view and "unknown" in view
    assert "00:10" in view and "Success 0" in view and "Failed 0" in view


@pytest.mark.parametrize("width", [60, 80, 120])
def test_terminal_width_and_literal_text(width):
    snapshot = ProgressSnapshot(method="[bold]literal", cases=[
        CaseProgress(case_id="AIE_DDX_MECH_V4_0037", status="running",
                     stage="Microscopic", detail="Long route detail " * 5),
    ])
    view = rendered(snapshot, width)
    assert "0037" in view and "[bold]literal" in view
    assert "\x1b" not in view
    assert all(len(line) <= width for line in view.splitlines())


def test_ten_case_demo_has_no_fake_metrics(tmp_path):
    demo(tmp_path, mode="off", delay=0)
    snapshot = read_snapshot(tmp_path / "progress.json")
    assert len(snapshot.cases) == 10
    assert snapshot.method.startswith("DEMO ONLY")
    assert all(not row.scores for row in snapshot.cases)
    assert sum(row.status == "failed" for row in snapshot.cases) == 1


def test_display_write_failure_does_not_fail_experiment(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("no space")
    monkeypatch.setattr("mechcal.runtime.progress.atomic_json", fail)
    with ExperimentProgress(tmp_path, ["a"], method="test", mode="off") as progress:
        with progress.case("a"):
            write_activity(tmp_path, "Planner")
    assert progress.snapshot.state == "completed" and progress.persistence_error == "OSError"


def test_non_tty_never_emits_live_control_codes(tmp_path):
    stream = io.StringIO()
    console = Console(file=stream)
    with ExperimentProgress(tmp_path, ["a"], method="test", console=console) as progress:
        with progress.case("a"):
            pass
    assert "\x1b" not in stream.getvalue()
    assert progress._live is None


@pytest.mark.parametrize("workers", [1, 2])
def test_direct_cli_generate_only_progress(tmp_path, monkeypatch, workers):
    from mechcal.cli import evaluate_direct
    from mechcal.runtime.llm import OpenAICompatibleSettings

    config = OpenAICompatibleSettings(base_url="https://offline.test", model="test", api_key="not-used")
    monkeypatch.setattr(evaluate_direct.OpenAICompatibleSettings, "from_env", lambda: config)
    monkeypatch.setattr(evaluate_direct, "_load_cases", lambda _: [
        {"case_id": "a"}, {"case_id": "b"},
    ])
    monkeypatch.setattr(evaluate_direct, "_make_subject", lambda **_: SimpleNamespace(subject_id="test"))
    monkeypatch.setattr(evaluate_direct, "_run_or_load_raw", lambda *args, **kwargs: {"raw_json": {
        "mechanism_predictions": [{"label": "RIM_RIR_RIV", "rank": 1}],
    }})
    monkeypatch.setattr("sys.argv", ["mechcal-direct", "--case-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"), "--generate-only", "--progress", "off",
        "--workers", str(workers)])
    evaluate_direct.main()
    snapshot = read_snapshot(tmp_path / "out/progress.json")
    assert snapshot.state == "completed" and snapshot.judge == "disabled"
    assert all(row.status == "done" for row in snapshot.cases)


def test_mechcal_cli_failure_is_not_shown_as_success(tmp_path, monkeypatch):
    from mechcal.cli import evaluate
    from mechcal.runtime.llm import OpenAICompatibleSettings

    config = OpenAICompatibleSettings(base_url="https://offline.test", model="test", api_key="not-used")
    monkeypatch.setattr(evaluate.OpenAICompatibleSettings, "from_env", lambda: config)
    monkeypatch.setattr(evaluate, "_load_cases", lambda _: [{"case_id": "a", "reference_mechanisms": []}])
    monkeypatch.setattr(evaluate, "LLMMechanismEvidenceSupportJudge", lambda **_: SimpleNamespace(
        model="no-api-test", rubric={"mechanism_evidence_rubric_version": "v2"},
    ))
    monkeypatch.setattr(evaluate, "_run_or_load_mas", lambda *args, **kwargs: SimpleNamespace(
        status="failed", runtime={}, round_ids=[],
    ))
    monkeypatch.setattr(evaluate, "_prediction_payload", lambda _: [])
    monkeypatch.setattr("sys.argv", ["mechcal-evaluate", "--case-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "out"), "--progress", "off"])
    evaluate.main()
    snapshot = read_snapshot(tmp_path / "out/progress.json")
    assert snapshot.state == "completed_with_errors"
    assert snapshot.cases[0].status == "failed" and not snapshot.cases[0].scores
