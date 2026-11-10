"""Display-only experiment state. No prompts, credentials or scientific inputs."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.live import Live
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from mechcal.runtime.retry import ComponentResponseRejected


Status = Literal["waiting", "running", "done", "cached", "failed", "interrupted", "not_run"]
TERMINAL = {"done", "cached", "failed", "interrupted", "not_run"}
COLORS = {"done": "green", "cached": "cyan", "failed": "red", "running": "yellow",
          "waiting": "dim", "interrupted": "magenta", "not_run": "dim"}


class CaseProgress(BaseModel):
    case_id: str
    status: Status = "waiting"
    stage: str = "queued"
    detail: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    stage_started_at: float | None = None
    run_dir: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)


class ProgressSnapshot(BaseModel):
    schema_version: int = 1
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    method: str
    model: str = ""
    judge: str = ""
    pid: int = Field(default_factory=os.getpid)
    host: str = Field(default_factory=socket.gethostname)
    started_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    state: str = "running"
    max_rounds: int | None = None
    cases: list[CaseProgress]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_snapshot(path: Path) -> ProgressSnapshot:
    return ProgressSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def write_activity(root: Path, stage: str, round_id: str | None = None, detail: str = "") -> None:
    # Observability failure must not change a scientific run's control flow.
    try:
        atomic_json(root / "activity.json", {
            "stage": stage, "round_id": round_id, "detail": detail, "updated_at": time.time(),
        })
    except OSError:
        pass


class ExperimentProgress:
    def __init__(
        self, output_dir: Path, case_ids: list[str], *, method: str,
        model: str = "", judge: str = "", max_rounds: int | None = None,
        mode: str = "auto", console: Console | None = None,
    ) -> None:
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Progress case IDs must be unique")
        self.path = output_dir.resolve() / "progress.json"
        self.snapshot = ProgressSnapshot(method=method, model=model, judge=judge,
            max_rounds=max_rounds, cases=[CaseProgress(case_id=value) for value in case_ids])
        self.rows = {row.case_id: row for row in self.snapshot.cases}
        self.console = console or Console(stderr=True)
        self.show = mode != "off" and self.console.is_terminal
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self.persistence_error: str | None = None

    def _save(self) -> None:
        self.snapshot.updated_at = time.time()
        try:
            atomic_json(self.path, self.snapshot.model_dump(mode="json"))
        except OSError as exc:
            self.persistence_error = type(exc).__name__

    def __enter__(self):
        self._save()
        if self.show:
            self._live = Live(console=self.console, get_renderable=self.render, refresh_per_second=2)
            self._live.start()
        self._thread = threading.Thread(target=self._heartbeat, name="mechcal-progress", daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(2):
            with self._lock:
                self._save()

    def __exit__(self, exc_type, exc, traceback):
        interrupted = exc_type is KeyboardInterrupt or isinstance(exc, ComponentResponseRejected)
        self._stop.set()
        if self._thread:
            self._thread.join()
        with self._lock:
            for row in self.snapshot.cases:
                if row.status == "running":
                    self.finish(row.case_id, status="interrupted" if interrupted else "failed",
                                detail=exc_type.__name__ if exc_type else "Incomplete case")
                elif row.status == "waiting":
                    self.finish(row.case_id, status="not_run", detail="Batch ended before this case")
            self.snapshot.state = (
                "interrupted" if interrupted else
                "failed" if exc_type else
                "completed_with_errors" if any(r.status not in {"done", "cached"} for r in self.rows.values())
                else "completed"
            )
            self.snapshot.ended_at = time.time()
            self._save()
        if self._live:
            self._live.stop()
        if self.persistence_error:
            self.console.print(Text(f"Progress checkpoint unavailable: {self.persistence_error}"))
        return False

    @contextmanager
    def case(self, case_id: str):
        self.start(case_id)
        try:
            yield
        except BaseException as exc:
            self.finish(case_id, status="interrupted" if isinstance(exc, (KeyboardInterrupt, ComponentResponseRejected)) else "failed",
                        detail=type(exc).__name__)
            raise
        else:
            if self.rows[case_id].status == "running":
                self.finish(case_id)

    def start(self, case_id: str, *, run_dir: Path | None = None) -> None:
        with self._lock:
            row = self.rows[case_id]
            row.status = "running"
            row.started_at = row.stage_started_at = time.time()
            row.ended_at = None
            row.stage = "inference"
            row.run_dir = str(run_dir.resolve()) if run_dir else None
            self._save()

    def update(self, case_id: str, stage: str, *, detail: str = "") -> None:
        with self._lock:
            row = self.rows[case_id]
            row.stage = stage
            row.detail = detail
            row.stage_started_at = time.time()
            self._save()

    def finish(self, case_id: str, *, status: Status = "done", detail: str = "", scores: dict | None = None) -> None:
        if status not in TERMINAL:
            raise ValueError("finish requires a terminal status")
        with self._lock:
            row = self.rows[case_id]
            row.status = status
            row.stage = status
            row.detail = detail
            row.ended_at = time.time()
            row.scores = {key: float(value) for key, value in (scores or {}).items()
                          if key in {"recall_at_3", "es_ndcg_at_3", "support_at_top1"}
                          and isinstance(value, (int, float))}
            self._save()

    def render(self):
        with self._lock:
            snapshot = self.snapshot.model_copy(deep=True)
        return render_progress(snapshot, width=self.console.width)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def _activity(row: CaseProgress) -> dict:
    if not row.run_dir:
        return {}
    try:
        value = json.loads((Path(row.run_dir) / "activity.json").read_text())
        if isinstance(value, dict) and value.get("updated_at", 0) >= (row.started_at or 0):
            return value
    except (OSError, ValueError, TypeError):
        pass
    return {}


def render_progress(snapshot: ProgressSnapshot, *, width: int = 100, now: float | None = None):
    now = time.time() if now is None else now
    stale = snapshot.state == "running" and now - snapshot.updated_at > 15
    end = snapshot.ended_at or (snapshot.updated_at if stale else now)
    rows = snapshot.cases
    succeeded = sum(row.status in {"done", "cached"} for row in rows)
    failed = sum(row.status == "failed" for row in rows)
    finished = sum(row.status in TERMINAL for row in rows)
    title = Text(f"MechCAL experiments | {snapshot.method} | {snapshot.state}", style="bold")
    subtitle = Text(f"Model: {snapshot.model or '-'} | Judge: {snapshot.judge or '-'}")
    totals = Text(f"Finished {finished}/{len(rows)}  Success {succeeded}  Failed {failed}  "
                  f"Elapsed {_duration(end - snapshot.started_at)}")
    table = Table(expand=True, box=None, padding=(0, 1))
    table.add_column("Case", no_wrap=True, max_width=25)
    table.add_column("Status", no_wrap=True)
    table.add_column("Stage / activity", overflow="fold", ratio=2)
    table.add_column("Round", no_wrap=True)
    table.add_column("Elapsed", no_wrap=True, justify="right")
    if width >= 105:
        table.add_column("In stage", no_wrap=True, justify="right")
        table.add_column("R@3", justify="right")
        table.add_column("ES@3", justify="right")
    for row in rows:
        activity = _activity(row)
        show_activity = row.status == "running" and row.stage == "inference"
        stage = str(activity.get("stage") or row.stage) if show_activity else row.stage
        detail = str(activity.get("detail") or row.detail) if show_activity else row.detail
        if detail:
            stage += f": {detail}"
        round_id = activity.get("round_id") or "-"
        if round_id != "-" and snapshot.max_rounds:
            round_id = f"{round_id}/{snapshot.max_rounds}"
        case_end = row.ended_at or end
        elapsed = case_end - row.started_at if row.started_at else None
        stage_start = (activity.get("updated_at") if show_activity else None) or row.stage_started_at
        stage_elapsed = case_end - stage_start if row.status == "running" and stage_start else None
        case_id = row.case_id.removeprefix("AIE_DDX_MECH_V4_")
        cells = [Text(case_id), Text(row.status, style=COLORS[row.status]), Text(stage),
                 Text(str(round_id)), Text(_duration(elapsed))]
        if width >= 105:
            cells.extend([Text(_duration(stage_elapsed)),
                Text(f"{row.scores['recall_at_3']:.3f}" if "recall_at_3" in row.scores else "-"),
                Text(f"{row.scores['es_ndcg_at_3']:.3f}" if "es_ndcg_at_3" in row.scores else "-")])
        table.add_row(*cells)
    footer = Text(
        "No heartbeat for >15s. Last saved state; process health is unknown."
        if stale else "Round is an upper limit, not a completion percentage. Missing scores remain '-'.",
        style="yellow" if stale else "dim",
    )
    return Group(title, subtitle, totals,
                 ProgressBar(total=max(1, len(rows)), completed=finished), Text(""), table, footer)
