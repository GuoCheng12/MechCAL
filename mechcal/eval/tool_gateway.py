"""One recorded tool gateway shared by every evaluated method (release 0.3.0 evaluation).

Agents (ReAct baseline, the redesigned MechCAL, external agents via an adapter) call tools only
through this gateway. Each call is executed by the existing MechCAL tool adapters,
persisted in the MechCAL run layout (``rounds/Rnnn/execution_*_tool_result.json``) and
registered as an evaluator artifact, so all methods are graded on identical records.
Observations returned to the agent list numeric outputs with their JSON pointers so
that citations can be copied rather than guessed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mechcal.eval.submission import EvaluatorArtifact, ArtifactStore, resolve_pointer

MAX_OBSERVATION_VALUES = 120
MAX_STRING_CHARS = 300
# Bookkeeping leaves that are not scientific observations.
NOISE_SEGMENTS = ("/timing/", "/tool_args/", "/runtime/", "/provenance/")


SKIP_KEYS = {"mulliken_charges", "excited_to_excited_transition_dipoles", "transition_conventions",
             "section_presence", "requested_parameters", "effective_parameters", "method_provenance",
             "tool_args", "timing", "runtime", "provenance", "file_paths", "coordinates", "xyz",
             "positions", "geometry_xyz", "atom_map", "step_records", "raw_text", "stdout", "stderr"}
MAX_ROWS = 12
MAX_INNER = 3


MIN_CALL_SECONDS = 60


def _canonical(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def _token(key) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def observable_fields(document: Any, prefix: str = "", omitted: dict | None = None,
                      depth: int = 0) -> tuple[list[tuple[str, float]], list[tuple[str, str]], dict]:
    """Scientific leaves for an agent observation: numbers and short labels with exact pointers.

    Bulky or bookkeeping branches are skipped and reported in ``omitted`` (pointer -> size) so
    the agent can fetch them with an exact read. Long lists keep their first rows; nested
    lists inside rows keep their first few items. Pointers always resolve in the artifact.
    """
    omitted = {} if omitted is None else omitted
    numbers: list[tuple[str, float]] = []
    labels: list[tuple[str, str]] = []
    if isinstance(document, dict):
        for key, value in document.items():
            pointer = f"{prefix}/{_token(key)}"
            if key in SKIP_KEYS or str(key).endswith(("_path", "_paths", "_sha256")):
                if isinstance(value, (list, dict)) and value:
                    omitted[pointer] = len(value)
                continue
            n, lab, _ = observable_fields(value, pointer, omitted, depth + 1)
            numbers += n
            labels += lab
    elif isinstance(document, list):
        limit = MAX_ROWS if depth <= 4 and all(isinstance(x, dict) for x in document) else MAX_INNER
        for i, value in enumerate(document[:limit]):
            n, lab, _ = observable_fields(value, f"{prefix}/{i}", omitted, depth + 1)
            numbers += n
            labels += lab
        if len(document) > limit:
            omitted[prefix] = len(document)
    elif isinstance(document, bool):
        labels.append((prefix, str(document).lower()))
    elif isinstance(document, (int, float)):
        if document == document and abs(document) != float("inf"):
            numbers.append((prefix, float(document)))
    elif isinstance(document, str) and 0 < len(document) <= 80 and "/" not in document[:1]:
        labels.append((prefix, document))
    return numbers, labels, omitted


def numeric_leaves(document: Any, prefix: str = "") -> list[tuple[str, float]]:
    """All finite numeric leaves as (JSON pointer, value), in document order."""
    out: list[tuple[str, float]] = []
    if isinstance(document, dict):
        for key, value in document.items():
            token = str(key).replace("~", "~0").replace("/", "~1")
            out += numeric_leaves(value, f"{prefix}/{token}")
    elif isinstance(document, list):
        for i, value in enumerate(document):
            out += numeric_leaves(value, f"{prefix}/{i}")
    elif isinstance(document, (int, float)) and not isinstance(document, bool):
        if document == document and abs(document) != float("inf"):
            out.append((prefix, float(document)))
    return out


@dataclass
class CallRecord:
    call_index: int
    capability_id: str
    tool_args: dict
    status: str
    artifact_id: str | None
    wall_seconds: float
    error: str | None = None


@dataclass
class ToolGateway:
    case_id: str
    smiles: str
    run_dir: Path
    enable_amesp: bool = False
    allowed: tuple[str, ...] | None = None
    max_calls: int | None = None
    max_wall_seconds: float | None = None     # total tool wall time for the run; long calls are clamped to it
    calls: list[CallRecord] = field(default_factory=list)
    _observations: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        from mechcal.capabilities import default_capability_registry
        from mechcal.schemas import ArtifactManifest, CaseInput, CaseRun, EvidenceLedger
        from mechcal.tools.local_macro import LocalMacroTool
        from mechcal.tools.local_micro import LocalMicroscopicTool
        from mechcal.tools.local_structure import LocalStructureTool

        self.run_dir = Path(self.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.registry = default_capability_registry()
        self.store = ArtifactStore(case_id=self.case_id)
        self._macro = LocalMacroTool()
        self._micro = LocalMicroscopicTool(enable_amesp=self.enable_amesp)
        case_input = CaseInput(case_id=self.case_id, smiles=self.smiles, user_query="Rank mechanisms.")
        artifact, failure = LocalStructureTool().prepare(case_input, round_id="R000", workspace=self.run_dir)
        self.case_run = CaseRun(case_id=self.case_id, input=case_input, status="running",
                                evidence_ledger=EvidenceLedger(case_id=self.case_id),
                                artifact_manifest=ArtifactManifest(case_id=self.case_id).with_records([artifact]))
        self.prepared_artifact_id = artifact.artifact_id
        self.preparation_failure = failure.kind

    # ---- catalog ---------------------------------------------------------------------------
    def tool_ids(self) -> list[str]:
        ids = [c.capability_id for c in self.registry.snapshot().capabilities]
        return [i for i in ids if self.allowed is None or i in self.allowed]

    def describe(self, capability_ids: list[str] | None = None) -> list[dict]:
        ids = capability_ids or self.tool_ids()
        unknown = [i for i in ids if i not in self.tool_ids()]
        if unknown:
            raise KeyError(f"Unknown or disallowed tools: {unknown}")
        return self.registry.describe_tools(ids, self.case_run.artifact_manifest, amesp_enabled=self.enable_amesp)

    def catalog(self) -> list[dict]:
        """Compact cards: id, description and readiness; full argument schemas via ``describe``."""
        cards = []
        for cap in self.registry.snapshot().capabilities:
            if cap.capability_id not in self.tool_ids():
                continue
            status = self.registry.readiness(cap.capability_id, self.case_run.artifact_manifest,
                                             amesp_enabled=self.enable_amesp)["status"]
            cards.append({"tool": cap.capability_id, "description": cap.description, "status": status})
        return cards

    # ---- execution -------------------------------------------------------------------------
    def call(self, capability_id: str, tool_args: dict | None = None) -> dict:
        from mechcal.schemas import DispatchRequest
        from mechcal.schemas.execution import AgentExecutionPlan

        tool_args = dict(tool_args or {})
        index = len(self.calls) + 1
        executed = sum(c.status != "rejected" for c in self.calls)
        if self.max_calls is not None and executed >= self.max_calls:
            self.calls.append(CallRecord(index, capability_id, tool_args, "rejected", None, 0.0, "budget exhausted"))
            return {"status": "rejected", "error": f"Tool-call budget of {self.max_calls} is exhausted."}
        if capability_id not in self.tool_ids():
            self.calls.append(CallRecord(index, capability_id, tool_args, "rejected", None, 0.0, "unknown tool"))
            return {"status": "rejected", "error": f"Unknown or disallowed tool: {capability_id}"}
        same = next((c for c in self.calls if c.artifact_id and c.capability_id == capability_id
                     and _canonical(c.tool_args) == _canonical(tool_args)), None)
        if same is not None:
            self.calls.append(CallRecord(index, capability_id, tool_args, "rejected", None, 0.0, f"duplicate of {same.artifact_id}"))
            previous = dict(self._observations.get(same.artifact_id, {}))
            previous.update(duplicate_of=same.artifact_id,
                            note=(f"Identical request already executed as {same.artifact_id} (status {same.status}); "
                                  "not re-run. Its result is shown again. Change the method or arguments for a new result."))
            return previous
        note = ""
        if self.max_wall_seconds is not None:
            left = self.wall_seconds_left()
            if left < MIN_CALL_SECONDS:
                self.calls.append(CallRecord(index, capability_id, tool_args, "rejected", None, 0.0, "wall budget exhausted"))
                return {"status": "rejected", "error": f"Tool wall-time budget of {self.max_wall_seconds:.0f} s is exhausted."}
            requested = tool_args.get("max_wall_seconds")
            if isinstance(requested, (int, float)) and requested > left:
                tool_args["max_wall_seconds"] = int(left)
                note = f"max_wall_seconds lowered from {requested} to {int(left)} (remaining tool wall-time budget)."
        cap = self.registry.require(capability_id)
        round_id, dispatch_id = f"R{index:03d}", f"gw:{index:03d}:{capability_id}"
        request = DispatchRequest(
            dispatch_id=dispatch_id, round_id=round_id, agent_name=cap.owner_agent,
            capability_id=capability_id, task=f"Gateway call {index}", objective="Recorded tool call.",
            evidence_goal_family=cap.evidence_family, route=cap.route,
            input_refs=self.case_run.artifact_manifest.artifact_ids(), tool_args=tool_args)
        plan = AgentExecutionPlan(
            plan_id=f"{dispatch_id}:plan", round_id=round_id, agent_name=cap.owner_agent,
            dispatch_id=dispatch_id, capability_id=capability_id, selected_route=cap.route,
            tool_steps=[f"execute {cap.route}"], tool_args=tool_args,
            tool_arg_rationale="Supplied by the calling agent.", parameter_adjustment_hint="No planner state; arguments come from the calling agent.",
            artifact_refs=self.case_run.artifact_manifest.artifact_ids(), planning_mode="llm")
        started = time.monotonic()
        try:
            tool = self._macro if cap.owner_agent == "macro" else self._micro
            result = tool.run(request, self.case_run, execution_plan=plan)
        except Exception as exc:  # noqa: BLE001 - recorded and returned to the agent, never hidden
            elapsed = time.monotonic() - started
            self.calls.append(CallRecord(index, capability_id, tool_args, "error", None, elapsed, str(exc)))
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:MAX_STRING_CHARS]}
        elapsed = time.monotonic() - started
        alias = self._persist(result, index)
        if result.artifact_updates:
            self.case_run = self.case_run.touch(
                artifact_manifest=self.case_run.artifact_manifest.with_records(list(result.artifact_updates)))
        self.calls.append(CallRecord(index, capability_id, tool_args, result.status, alias, elapsed))
        obs = self.observation(result, alias)
        self._observations[alias] = dict(obs)
        if note:
            obs["note"] = note
        return obs

    def _persist(self, result, index: int) -> str:
        path = self.run_dir / "rounds" / result.round_id / f"execution_{result.dispatch_id.replace(':', '_')}_tool_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(), encoding="utf-8")
        status = {"success": "success", "partial": "partial"}.get(result.status, "failed")
        outputs = {"structured_results": result.structured_results,
                   "evidence_metrics": {e.evidence_id: e.metrics for e in result.evidence_units}}
        provenance = {"source_file": str(path.relative_to(self.run_dir)), "round_id": result.round_id,
                      "tool_result_id": result.tool_result_id}
        alias = f"T{index:03d}"
        for artifact_id in (result.tool_result_id, alias):
            self.store.add(EvaluatorArtifact(
                artifact_id=artifact_id, case_id=self.case_id, tool=result.capability_id,
                status=status, molecule_smiles=self.smiles, outputs=outputs, provenance=provenance))
        return alias

    def observation(self, result, alias: str) -> dict:
        artifact = self.store.artifacts[alias]
        root = artifact.outputs.get("structured_results", {})
        focus = "/structured_results/parsed_observables" if isinstance(root.get("parsed_observables"), dict) else "/structured_results"
        numbers, labels, omitted = observable_fields(resolve_pointer(artifact.outputs, focus), focus)
        numbers = [(p, v) for p, v in numbers if not any(seg in p + "/" for seg in NOISE_SEGMENTS)]
        return {
            "artifact_id": alias, "tool": result.capability_id, "status": result.status,
            "summary": result.summary[:MAX_STRING_CHARS],
            "failure": None if result.status == "success" else result.failure.message[:MAX_STRING_CHARS],
            "values": {p: v for p, v in numbers[:MAX_OBSERVATION_VALUES]},
            "labels": {p: v for p, v in labels[:MAX_OBSERVATION_VALUES]},
            "values_truncated": max(0, len(numbers) - MAX_OBSERVATION_VALUES),
            "omitted_for_read": dict(list(omitted.items())[:20]),
            "reusable_artifacts": [a.artifact_id for a in result.artifact_updates],
            "limitations": list(result.structured_results.get("limitations", []))[:5]
            if isinstance(result.structured_results.get("limitations"), list) else [],
        }

    def read(self, artifact_id: str, pointer: str = "") -> dict:
        """Exact lookup into a recorded artifact (no computation, no new evidence)."""
        artifact = self.store.artifacts.get(artifact_id)
        if artifact is None:
            return {"status": "rejected", "error": f"Unknown artifact {artifact_id}"}
        try:
            value = resolve_pointer(artifact.outputs, pointer)
        except KeyError:
            return {"status": "rejected", "error": f"Pointer {pointer!r} not found in {artifact_id}"}
        text = json.dumps(value, ensure_ascii=False, default=str)
        return {"status": "ok", "artifact_id": artifact_id, "pointer": pointer,
                "value": value if len(text) <= 4000 else text[:4000] + " …[truncated]"}

    def wall_seconds_used(self) -> float:
        return sum(c.wall_seconds for c in self.calls)

    def wall_seconds_left(self) -> float:
        return float("inf") if self.max_wall_seconds is None else max(0.0, self.max_wall_seconds - self.wall_seconds_used())

    def call_log(self) -> list[dict]:
        """One line per executed call, so every role sees what was already run and how it ended."""
        rows = []
        for c in self.calls:
            if c.artifact_id is None:
                continue
            art = self.store.artifacts.get(c.artifact_id)
            failure = ""
            if c.status != "success" and art is not None:
                failure = str(resolve_pointer(art.outputs, "/structured_results").get("failure_code", "")) if isinstance(
                    art.outputs.get("structured_results"), dict) else ""
            rows.append({"artifact_id": c.artifact_id, "tool": c.capability_id, "status": c.status,
                         "wall_seconds": round(c.wall_seconds), "args": {k: v for k, v in c.tool_args.items()
                                                                        if not isinstance(v, (list, dict)) or len(str(v)) < 80},
                         **({"failure_code": failure} if failure else {})})
        return rows

    def executed_calls(self) -> int:
        return sum(c.status != "rejected" for c in self.calls)

    def cost(self) -> dict:
        return {"tool_calls": self.executed_calls(), "rejected_calls": len(self.calls) - self.executed_calls(),
                "tool_wall_seconds": round(sum(c.wall_seconds for c in self.calls), 3),
                "by_status": {s: sum(c.status == s for c in self.calls) for s in {c.status for c in self.calls}}}
