"""One recorded tool gateway shared by every evaluated method (evaluation v2).

Agents (ReAct baseline, MechCAL v2, external agents via an adapter) call tools only
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

from mechcal.eval.evidence_v4 import ArtifactRecordV4, ArtifactStore, resolve_pointer

MAX_OBSERVATION_VALUES = 80
MAX_STRING_CHARS = 300
# Bookkeeping leaves that are not scientific observations.
NOISE_SEGMENTS = ("/timing/", "/tool_args/", "/runtime/", "/provenance/")


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
    calls: list[CallRecord] = field(default_factory=list)

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
        return self.observation(result, alias)

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
            self.store.add(ArtifactRecordV4(
                artifact_id=artifact_id, case_id=self.case_id, tool=result.capability_id,
                status=status, molecule_smiles=self.smiles, outputs=outputs, provenance=provenance))
        return alias

    def observation(self, result, alias: str) -> dict:
        artifact = self.store.artifacts[alias]
        values = [(p, v) for p, v in numeric_leaves(artifact.outputs)
                  if not any(seg in p + "/" for seg in NOISE_SEGMENTS)]
        return {
            "artifact_id": alias, "tool": result.capability_id, "status": result.status,
            "summary": result.summary[:MAX_STRING_CHARS],
            "failure": None if result.status == "success" else result.failure.message[:MAX_STRING_CHARS],
            "values": {p: v for p, v in values[:MAX_OBSERVATION_VALUES]},
            "values_truncated": max(0, len(values) - MAX_OBSERVATION_VALUES),
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

    def executed_calls(self) -> int:
        return sum(c.status != "rejected" for c in self.calls)

    def cost(self) -> dict:
        return {"tool_calls": self.executed_calls(), "rejected_calls": len(self.calls) - self.executed_calls(),
                "tool_wall_seconds": round(sum(c.wall_seconds for c in self.calls), 3),
                "by_status": {s: sum(c.status == s for c in self.calls) for s in {c.status for c in self.calls}}}
