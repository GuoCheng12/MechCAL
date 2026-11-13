"""Evaluation subjects that emit the grounded submission format natively: Direct LLM and ReAct over the gateway.

No heuristic fallback answers: if a subject never produces a valid submission within its
budget, that outcome is recorded as such and scores as an empty answer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from mechcal.eval.submission import ArtifactStore, Submission
from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.tool_gateway import ToolGateway
from mechcal.mechanism_pool import MECHANISM_POOL, mechanism_definitions

INSTRUCTIONS = "submission_instructions.md"
SUBMISSION_PROTOCOL_VERSION = "0.3.0"


def model_submission_schema() -> dict:
    """JSON schema of what the model writes; the harness adds case_id."""
    schema = Submission.model_json_schema()
    schema["properties"].pop("case_id")
    schema["required"] = ["mechanism_predictions"]
    defs = schema["$defs"]
    defs["Prediction"]["properties"]["label"]["enum"] = list(MECHANISM_POOL)
    schema["properties"]["mechanism_predictions"].update(minItems=3, maxItems=3)
    return schema


def parse_model_submission(data: Any, case_id: str) -> Submission:
    if not isinstance(data, dict) or "mechanism_predictions" not in data:
        raise ValueError("Return an object with a mechanism_predictions list")
    try:
        submission = Submission.model_validate({"case_id": case_id, "mechanism_predictions": data["mechanism_predictions"]})
    except ValidationError as exc:
        raise ValueError(_compact_errors(exc)) from exc
    if len(submission.mechanism_predictions) != 3:
        raise ValueError("Submit exactly three predictions")
    return submission


def _compact_errors(exc: ValidationError, limit: int = 8) -> str:
    parts = [f"{'/'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:limit]]
    return "; ".join(parts)


@dataclass
class UsageLedger:
    calls: int = 0
    by_role: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def add(self, role: str, usage: dict | None) -> None:
        self.calls += 1
        bucket = self.by_role[role]
        bucket["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = (usage or {}).get(key)
            if isinstance(value, (int, float)):
                bucket[key] += int(value)
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        if isinstance(cached, (int, float)):
            bucket["cached_prompt_tokens"] += int(cached)

    def as_dict(self) -> dict:
        return {"model_calls": self.calls, "by_role": {r: dict(v) for r, v in self.by_role.items()}}


def _task_payload(smiles: str) -> dict:
    return {"verified_smiles": smiles, "mechanism_pool": mechanism_definitions()}


@dataclass
class SubjectResult:
    case_id: str
    subject: str
    status: str
    submission: Submission | None
    store: ArtifactStore
    usage: dict
    tool_cost: dict
    transcript: list[dict]
    error: str | None = None

    def to_json(self) -> dict:
        return {"case_id": self.case_id, "subject": self.subject, "status": self.status, "error": self.error,
                "protocol": SUBMISSION_PROTOCOL_VERSION,
                "submission": self.submission.model_dump() if self.submission else None,
                "usage": self.usage, "tool_cost": self.tool_cost, "transcript": self.transcript}


class DirectSubject:
    """Single-response closed-book baseline; one repair if the format is invalid."""

    name = "direct"

    def __init__(self, client, *, max_repairs: int = 1) -> None:
        self.client, self.max_repairs = client, max_repairs
        self.system = load_eval_prompt(INSTRUCTIONS) + "\n\nNo tools are available in this setting; use structural and prior claims only."

    def _catalog(self, gateway: ToolGateway, smiles: str) -> list[dict]:
        """The same capability cards as the MechCAL agent (same tools, text, applicability and cost)."""
        from mechcal.agent import cards as cardlib
        from mechcal.agent.costs import CostModel
        from mechcal.agent.facts import structural_facts
        from mechcal.agent.supervision import ARG_CLASS
        entries = cardlib.tool_entries(gateway)
        self._entries = {e["tool"]: e for e in entries}
        facts = structural_facts(smiles)
        kind = next(a.kind for a in gateway.case_run.artifact_manifest.artifacts if a.artifact_id == gateway.prepared_artifact_id)
        costs = CostModel(int(facts["summary"]["atoms_with_h"]), smiles=smiles)
        cost = lambda tool: {"cost": costs.estimate(tool, {k: v["default"] for k, v in self._entries[tool]["args"].items()
                                                           if "default" in v})}
        return cardlib.catalog(entries, cardlib.molecule_profile(facts, kind), ARG_CLASS,
                               {a.kind for a in gateway.case_run.artifact_manifest.artifacts}, cost=cost)

    def run(self, case_id: str, smiles: str) -> SubjectResult:
        usage, transcript, feedback = UsageLedger(), [], None
        for _ in range(self.max_repairs + 1):
            result = self.client.complete_json_result(system_prompt=self.system, payload=_task_payload(smiles),
                                                      schema_feedback=feedback)
            usage.add("direct", result.usage)
            transcript.append({"response": result.data, "parse_error": result.parse_error})
            try:
                submission = parse_model_submission(result.data, case_id)
                return SubjectResult(case_id, self.name, "submitted", submission, ArtifactStore(case_id=case_id),
                                     usage.as_dict(), {"tool_calls": 0, "tool_wall_seconds": 0.0}, transcript)
            except ValueError as exc:
                feedback = str(exc)
        return SubjectResult(case_id, self.name, "no_valid_submission", None, ArtifactStore(case_id=case_id),
                             usage.as_dict(), {"tool_calls": 0, "tool_wall_seconds": 0.0}, transcript, feedback)


REACT_PROTOCOL = """

You may use tools before submitting. Reply with exactly one JSON action per turn:
- {"action": "describe_tools", "tools": ["<tool id>", ...]}  -> full argument specifications
- {"action": "call_tool", "tool": "<tool id>", "args": {...}}  -> runs a calculation
- {"action": "read_artifact", "artifact_id": "T001", "pointer": "/..."}  -> exact value lookup
- {"action": "submit", "mechanism_predictions": [...]}  -> final answer, ends the episode
The prepared input structure has artifact_id given in the payload; many tools take it as
source_artifact_id. Tool results list numeric values by JSON pointer: cite those exactly.
When must_submit_now is true, submit immediately."""


class ReactSubject:
    """Transcript-only ReAct agent over the recorded gateway. Budgets are explicit and reported."""

    name = "react"

    def __init__(self, client, gateway_factory: Callable[[str, str], ToolGateway], *,
                 max_model_calls: int = 20, max_tool_calls: int = 8, cost_log_path: str | None = None) -> None:
        self.client, self.gateway_factory = client, gateway_factory
        self.max_model_calls, self.max_tool_calls = max_model_calls, max_tool_calls
        self.cost_log_path = cost_log_path          # same recorded timings as the agent (read only)
        self.system = load_eval_prompt(INSTRUCTIONS) + REACT_PROTOCOL

    def _catalog(self, gateway: ToolGateway, smiles: str) -> list[dict]:
        """The same capability cards as the MechCAL agent (same tools, text, applicability and cost)."""
        from mechcal.agent import cards as cardlib
        from mechcal.agent.costs import CostModel
        from mechcal.agent.facts import structural_facts
        from mechcal.agent.supervision import ARG_CLASS
        entries = cardlib.tool_entries(gateway)
        self._entries = {e["tool"]: e for e in entries}
        facts = structural_facts(smiles)
        kind = next(a.kind for a in gateway.case_run.artifact_manifest.artifacts if a.artifact_id == gateway.prepared_artifact_id)
        costs = CostModel(int(facts["summary"]["atoms_with_h"]), self.cost_log_path, smiles=smiles)
        cost = lambda tool: {"cost": costs.estimate(tool, {k: v["default"] for k, v in self._entries[tool]["args"].items()
                                                           if "default" in v})}
        return cardlib.catalog(entries, cardlib.molecule_profile(facts, kind), ARG_CLASS,
                               {a.kind for a in gateway.case_run.artifact_manifest.artifacts}, cost=cost)

    def run(self, case_id: str, smiles: str) -> SubjectResult:
        gateway = self.gateway_factory(case_id, smiles)
        gateway.max_calls = self.max_tool_calls
        usage, history, error = UsageLedger(), [], None
        base = {**_task_payload(smiles), "prepared_structure_artifact_id": gateway.prepared_artifact_id,
                "tool_catalog": self._catalog(gateway, smiles), "tool_call_budget": self.max_tool_calls}
        for step in range(1, self.max_model_calls + 1):
            payload = {**base, "history": history,
                       "tool_calls_used": gateway.executed_calls(),
                       "must_submit_now": step == self.max_model_calls or gateway.executed_calls() >= self.max_tool_calls}
            result = self.client.complete_json_result(system_prompt=self.system, payload=gateway.public(payload))
            usage.add("react", result.usage)
            action = result.data if isinstance(result.data, dict) else {}
            kind = action.get("action")
            if kind == "submit":
                try:
                    submission = parse_model_submission(action, case_id)
                    return SubjectResult(case_id, self.name, "submitted", submission, gateway.store,
                                         usage.as_dict(), gateway.cost(), history)
                except ValueError as exc:
                    error = str(exc)
                    history.append({"action": action, "observation": {"status": "rejected", "error": error}})
                    continue
            if kind == "call_tool":
                observation = gateway.call(str(action.get("tool", "")), action.get("args") or {})
            elif kind == "describe_tools":
                wanted = [str(t) for t in action.get("tools") or []]
                unknown = [t for t in wanted if t not in self._entries]
                observation = ({"status": "rejected", "error": f"Unknown or unavailable tools: {unknown}"} if unknown else
                               {"status": "ok", "tools": {t: self._entries[t]["args"] for t in wanted}})
            elif kind == "read_artifact":
                observation = gateway.read(str(action.get("artifact_id", "")), str(action.get("pointer", "")))
            else:
                observation = {"status": "rejected", "error": result.parse_error or "Unknown action; use one of the four actions."}
            history.append({"action": action, "observation": observation})
        return SubjectResult(case_id, self.name, "no_valid_submission", None, gateway.store, usage.as_dict(),
                             gateway.cost(), history, error or "Model-call budget exhausted without a valid submission.")


def save_result(result: SubjectResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts.json").write_text(result.store.model_dump_json(), encoding="utf-8")
    path = out_dir / "subject_result.json"
    path.write_text(json.dumps(result.to_json(), ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    return path
