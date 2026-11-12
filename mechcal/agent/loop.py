"""MechCAL agent loop (release 0.3.0): Planner over a hypothesis board, optional Worker and Critic.

Every tool call goes through the recorded gateway; every model call is logged with its
payload size and usage; the final answer is a projection of the board, filtered by the
deterministic grounding check when ``self_check`` is on.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mechcal.agent.board import Board, ClaimIn, Status
from mechcal.agent.facts import structural_facts
from mechcal.agent.recipes import RECIPE_ARGS, RECIPES, RecipeContext, run_recipe
from mechcal.agent import cards as cardlib
from mechcal.agent import supervision as sv
from mechcal.agent.costs import CostModel, expected_seconds
from mechcal.eval.grounding import ELIGIBLE, check_submission
from mechcal.eval.grounded_subjects import UsageLedger
from mechcal.eval.tool_gateway import ToolGateway
from mechcal.mechanism_pool import MECHANISM_POOL, mechanism_definitions
from mechcal.runtime.prompts import load_prompt

AGENT_RELEASE = "0.3.0"
# The tools an agent may see are exactly those with a capability card (mechcal/agent/cards/tool_cards.json).


@dataclass
class AgentConfig:
    max_planner_calls: int = 16
    max_computations: int = 12
    max_tool_wall_seconds: float = 3600.0    # total tool wall time per run; long requests are clamped to what is left
    critic: bool = True
    critic_every: int = 3
    worker: bool = True                      # professor-student mode: Planner delegates, Worker executes
    worker_max_calls: int = 6                # Worker model calls per delegate/revise session
    worker_max_computations: int = 4         # computations per delegate/revise session
    cost_check_fraction: float = 0.5         # delegation returns a cost check if it may use more of the time left
    cost_log_path: str = ""                  # cross-run cost log (JSON lines); empty: seed table + this run only
    api_retries: int = 3                     # extra attempts after a failed model call (network, timeout, server)
    api_backoff_seconds: float = 5.0         # wait before retry k: backoff * 3**k (5, 15, 45 s)
    max_internal_errors: int = 3             # unexpected exceptions tolerated per run before stopping with a result
    self_check: bool = True
    iterate: bool = True                     # False: plan once, run, submit (ablation)
    recipes: bool = True                     # False: single tools only (ablation)


# ---- model-facing schemas ------------------------------------------------------------------------

class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _label(value: str) -> str:
    if value not in MECHANISM_POOL:
        raise ValueError(f"Unknown family label {value!r}; use one of {list(MECHANISM_POOL)}")
    return value


KINDS = {"structural", "computed", "prior"}
ACTIONS = {"recipe", "tool", "delegate", "revise", "read", "submit"}


def _norm_family_item(item):
    if not isinstance(item, dict):
        return item
    item = dict(item)
    if "label" not in item:
        for alias in ("family", "mechanism", "mechanism_family"):
            if alias in item:
                item["label"] = item.pop(alias)
                break
    return item


def _norm_claim(item):
    item = _norm_family_item(item)
    if not isinstance(item, dict):
        return item
    if "kind" not in item and item.get("type") in KINDS:
        item["kind"] = item.pop("type")
    if "statement" not in item:
        for alias in ("text", "claim", "content", "description"):
            if isinstance(item.get(alias), str):
                item["statement"] = item.pop(alias)
                break
    for q in item.get("quantities") or []:
        if isinstance(q, dict) and isinstance(q.get("source"), dict):
            src = q.pop("source")
            q.setdefault("artifact_id", src.get("artifact_id"))
            q.setdefault("pointer", src.get("pointer"))
            if src.get("unit"):
                q.setdefault("source_unit", src.get("unit"))
    return item


def _norm_limitations(value):
    if isinstance(value, dict):
        out = []
        for label, text in value.items():
            for t in (text if isinstance(text, list) else [text]):
                out.append({"label": label, "text": str(t)})
        return out
    if isinstance(value, list):
        return [_norm_family_item(v) for v in value if isinstance(v, dict)]
    return []


def _norm_action(action, allowed=ACTIONS):
    if not isinstance(action, dict):
        return action
    a = dict(action)
    if "type" not in a:
        if a.get("kind") in allowed:
            a["type"] = a.pop("kind")
        elif a.get("action") in allowed:
            a["type"] = a.pop("action")
        else:
            for key in allowed:
                if key in a and isinstance(a[key], (str, dict)) and key != "read":
                    a["type"] = key
                    break
    kind = a.get("type")
    if kind in {"recipe", "tool"} and not a.get("name") and isinstance(a.get(kind), str):
        a["name"] = a.pop(kind)
    if kind == "delegate" and not a.get("task") and isinstance(a.get("delegate"), str):
        a["task"] = a.pop("delegate")
    if kind == "delegate" and not a.get("recipe") and a.get("name") in RECIPES:
        a["recipe"] = a.pop("name")
    if kind == "delegate" and isinstance(a.get("tools"), str):
        a["tools"] = [a["tools"]]
    if "arguments" in a and "args" not in a:
        a["args"] = a.pop("arguments")
    return a


def _tolerant(fn, data):
    """Alias normalisation must never raise: on odd input, validation reports the problem instead."""
    try:
        return fn(data)
    except Exception:  # noqa: BLE001
        return data


def normalize_planner(data):
    if not isinstance(data, dict):
        return data
    d = dict(data)
    d["updates"] = [_norm_family_item(u) for u in d.get("updates") or []]
    d["claims"] = [_norm_claim(c) for c in d.get("claims") or []]
    d["limitations"] = _norm_limitations(d.get("limitations"))
    d["action"] = _norm_action(d.get("action") or {})
    d["adopt"] = [{"finding_id": x} if isinstance(x, str) else x for x in d.get("adopt") or []]
    return d


class FamilyUpdate(_M):
    label: str
    plausibility: float | None = Field(default=None, ge=0, le=1)
    status: Status | None = None
    reason: str | None = None
    next_test: str | None = None
    @field_validator("label")
    @classmethod
    def _known(cls, v):
        return _label(v)


class Limitation(_M):
    label: str
    text: str
    @field_validator("label")
    @classmethod
    def _known(cls, v):
        return _label(v)


class ProposalResponse(_M):
    proposal_id: str
    decision: Literal["accept", "adapt", "decline"]
    reason: str


class Action(_M):
    type: Literal["recipe", "tool", "delegate", "revise", "read", "submit"]
    name: str | None = None
    args: dict = Field(default_factory=dict)
    separates: list[str] = Field(default_factory=list)
    expectation: str | dict[str, str] = ""
    task: str = ""
    artifact_id: str | None = None
    pointer: str | None = None
    # delegate (professor-student): what to run and the scientific settings the student must keep
    recipe: str | None = None
    tools: list[str] = Field(default_factory=list)
    fixed_args: dict = Field(default_factory=dict)
    suggested_args: dict = Field(default_factory=dict)   # delegate: technical starting values the Worker may change
    confirm_cost: bool = False                # delegate: go ahead after a cost check
    wall_budget_seconds: float | None = None  # delegate: tool time this task may use (student works within it)
    # revise: continue an existing task with its attempt record
    task_id: str | None = None
    instruction: str = ""
    args_hint: dict = Field(default_factory=dict)
    approve: bool = False

    @model_validator(mode="after")
    def _contrast(self):
        for f in self.separates:
            _label(f)
        if self.type in {"recipe", "tool", "delegate"}:
            if len(set(self.separates)) < 2:
                raise ValueError("action.separates must name at least two different families this computation distinguishes")
            text = " ".join(self.expectation.values()) if isinstance(self.expectation, dict) else self.expectation
            if not text.strip():
                raise ValueError("action.expectation must state the expected outcome under each separated family")
        if self.type in {"recipe", "tool"} and not self.name:
            raise ValueError("action.name is required for recipe and tool actions")
        if self.type == "delegate" and not self.task.strip():
            raise ValueError("action.task is required for delegate")
        if self.type == "delegate" and not (self.recipe or self.tools):
            raise ValueError("delegate must name the recipe or the tools the Worker may use")
        if self.type == "delegate" and self.recipe and self.recipe not in RECIPES:
            raise ValueError(f"Unknown recipe {self.recipe!r}; choose one of {sorted(RECIPES)}")
        if self.type == "revise" and not (self.task_id and self.instruction.strip()):
            raise ValueError("revise needs task_id and instruction")
        if self.type == "read" and not self.artifact_id:
            raise ValueError("action.artifact_id is required for read")
        return self


class Adoption(_M):
    finding_id: str
    label: str | None = None            # family the claim is attached to (defaults to the finding's label)
    @field_validator("label")
    @classmethod
    def _known(cls, v):
        return None if v is None else _label(v)


class PlannerStep(_M):
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        return _tolerant(normalize_planner, data)

    reading: str = ""
    adopt: list[Adoption] = Field(default_factory=list)
    updates: list[FamilyUpdate] = Field(default_factory=list)
    claims: list[ClaimIn] = Field(default_factory=list)
    drop_claims: list[str] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    proposal_responses: list[ProposalResponse] = Field(default_factory=list)
    action: Action

    @field_validator("claims")
    @classmethod
    def _claim_labels(cls, v):
        for c in v:
            _label(c.label)
        return v


class Finding(_M):
    """A claim drafted by the Worker; the Planner decides whether it becomes board evidence."""
    label: str | None = None
    statement: str
    kind: Literal["structural", "computed", "prior"]
    atom_ids: list[int] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    quantities: list[dict] = Field(default_factory=list)


class ProposedChange(_M):
    kind: Literal["recipe", "tool"] = "tool"
    name: str = ""
    args: dict = Field(default_factory=dict)
    reason: str = ""


class WorkerAction(_M):
    type: Literal["recipe", "tool", "read", "report"]
    name: str | None = None
    args: dict = Field(default_factory=dict)
    reason: str = ""                       # why this attempt / what was changed and why
    artifact_id: str | None = None
    pointer: str | None = None
    # report
    status: str | None = None
    summary: str = ""
    claims: list[Finding] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)      # the student's own concerns, beyond system warnings
    question_for_planner: str = ""
    proposed_change: ProposedChange | None = None


class WorkerStep(_M):
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        return _tolerant(cls._norm, data)

    @staticmethod
    def _norm(data):
        if isinstance(data, dict):
            data = dict(data)
            raw = dict(data.get("action") or {})
            if raw.get("type") == "finish" or raw.get("action") == "finish":
                raw["type"] = "report"
            a = _norm_action(raw, allowed={"recipe", "tool", "read", "report"})
            if isinstance(a, dict):
                if "findings" in a and "claims" not in a:
                    a["claims"] = a.pop("findings")
                a["claims"] = [_norm_claim(c) for c in a.get("claims") or []]
                a["warnings"] = [w if isinstance(w, str) else json.dumps(w) for w in a.get("warnings") or []]
            data["action"] = a
        return data

    thought: str = ""
    action: WorkerAction


class CriticRank(_M):
    label: str
    plausibility: float = Field(ge=0, le=1)
    basis: str = ""
    @field_validator("label")
    @classmethod
    def _known(cls, v):
        return _label(v)


class CriticTest(_M):
    label: str
    separates: list[str] = Field(default_factory=list)
    recipe_or_tool: str = ""
    why: str = ""


class CriticOutput(_M):
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        def norm(d):
            if isinstance(d, dict):
                d = {**d, "ranking": [_norm_family_item(r) for r in d.get("ranking") or []],
                     "tests": [_norm_family_item(t) for t in d.get("tests") or []]}
            return d
        return _tolerant(norm, data)

    ranking: list[CriticRank]
    tests: list[CriticTest] = Field(default_factory=list)


# ---- runner ---------------------------------------------------------------------------------------

@dataclass
class RunLog:
    path: Path
    events: list = field(default_factory=list)

    def write(self, event: dict) -> None:
        event = {"t": round(time.time(), 3), **event}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


class AgentRun:
    def __init__(self, client, *, case_id: str, smiles: str, run_dir: Path, config: AgentConfig | None = None,
                 enable_amesp: bool = True, worker_client=None, critic_client=None) -> None:
        self.client, self.cfg = client, config or AgentConfig()
        self.worker_client, self.critic_client = worker_client or client, critic_client or client
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.case_id, self.smiles = case_id, smiles
        self.gateway = ToolGateway(case_id=case_id, smiles=smiles, run_dir=self.run_dir / "tools", enable_amesp=enable_amesp,
                                   max_wall_seconds=self.cfg.max_tool_wall_seconds)
        self.recipe_ctx = RecipeContext(self.gateway)
        self.facts = structural_facts(smiles)
        self.board = Board.initial(case_id, smiles)
        self.usage = UsageLedger()
        self.log = RunLog(self.run_dir / "events.jsonl")
        self.history: list[str] = []
        self.latest: Any = {"note": "No calculations yet. Prepared structure: " + self.gateway.prepared_artifact_id}
        self.proposals: list[dict] = []
        self.critic_runs = 0
        self.last_critic_at = 0
        self.stop_reason = ""
        self.tasks: dict[str, sv.Task] = {}
        self.findings: dict[str, dict] = {}
        self.protocol = {"planner_rejected": 0, "approvals": 0, "status_corrections": 0}
        self.tool_cards = self._tool_cards()
        self.card_by_tool = {c["tool"]: c for c in self.tool_cards}
        prepared_kind = next(a.kind for a in self.gateway.case_run.artifact_manifest.artifacts
                             if a.artifact_id == self.gateway.prepared_artifact_id)
        self.profile = cardlib.molecule_profile(self.facts, prepared_kind)
        self.costs = CostModel(int(self.facts["summary"]["atoms_with_h"]), self.cfg.cost_log_path or None, smiles=smiles)
        self.recipe_ctx.estimate = lambda tool, args: expected_seconds(self._tool_cost(tool, args))
        self._cost_seen = 0
        self.gateway.allowed = tuple(card["tool"] for card in self.tool_cards)
        # professor prompt when the Worker is on; the single-agent Planner prompt otherwise
        self.planner_prompt = load_prompt("agent_professor.md" if self.cfg.worker else "agent_planner.md")
        self.worker_prompt = load_prompt("agent_worker.md")
        self.critic_prompt = load_prompt("agent_critic.md")

    # -- catalog
    def _tool_cards(self) -> list[dict]:
        return cardlib.tool_entries(self.gateway)

    def _artifacts(self) -> list[dict]:
        """Reusable inputs produced so far (geometries, bundles), for exact source_artifact_id choice."""
        return [{"artifact_id": a.artifact_id, "kind": a.kind, "status": a.status}
                for a in self.gateway.case_run.artifact_manifest.artifacts]

    def _recipe_cards(self) -> list[dict]:
        if not self.cfg.recipes:
            return []
        specs = {c["tool"]: c["args"] for c in self.tool_cards}
        out = []
        for name, (_, desc) in RECIPES.items():
            args = {arg: specs.get(tool, {}).get(targ, {}) for arg, (tool, targ) in RECIPE_ARGS.get(name, {}).items()}
            out.append({**cardlib.recipe_view(name, desc, args, self.profile), "cost": self._recipe_cost(name)})
        return out

    # -- model calls
    response_mode = "json_schema"

    def _format(self, role: str, model_cls) -> dict:
        if self.response_mode == "json_schema":
            return {"type": "json_schema", "json_schema": {"name": role, "schema": model_cls.model_json_schema(), "strict": False}}
        return {"type": "json_object"}

    def _ask(self, client, role: str, system: str, payload: dict, model_cls, *, repairs: int = 1):
        feedback = None
        for attempt in range(repairs + 1):
            body = json.dumps(payload, ensure_ascii=False, default=str)
            started = time.monotonic()
            kwargs = dict(system_prompt=system, payload=payload, schema_feedback=feedback, preserve_invalid_response=True)
            result = self._call_model(client, role, kwargs, model_cls)
            if result is None:                               # the API failed after retries: an invalid answer
                self.usage.add(role, {})
                continue
            self.usage.add(role, result.usage)
            if getattr(result, "parse_error", None):        # malformed JSON: an invalid answer, repaired like any other
                self.protocol["malformed_json"] = self.protocol.get("malformed_json", 0) + 1
                feedback = (f"Your reply was not one valid JSON object ({str(result.parse_error)[:300]}). "
                            "Return the complete object again as plain JSON, nothing before or after it.")
                self.log.write({"role": role, "attempt": attempt + 1, "payload_bytes": len(body), "usage": result.usage,
                                "valid": False, "error": feedback})
                continue
            event = {"role": role, "attempt": attempt + 1, "payload_bytes": len(body), "usage": result.usage,
                     "seconds": round(time.monotonic() - started, 2), "response": result.data,
                     "parse_error": result.parse_error}
            try:
                parsed = model_cls.model_validate(result.data)
                self.log.write({**event, "valid": True})
                return parsed
            except ValidationError as exc:
                feedback = ("Your JSON did not match the required format: "
                            + "; ".join(f"{'/'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8])
                            + ". Return the complete object again with exactly the field names of the example.")
                self.log.write({**event, "valid": False, "error": feedback})
        return None

    def _call_model(self, client, role: str, kwargs: dict, model_cls):
        """One model call with bounded retries; None if the provider keeps failing (never raises)."""
        for k in range(self.cfg.api_retries + 1):
            try:
                return client.complete_json_result(**kwargs, response_format=self._format(role, model_cls))
            except Exception as exc:  # noqa: BLE001 - network, timeout, server or format rejection
                text = f"{type(exc).__name__}: {exc}"
                if self.response_mode == "json_schema" and re.search(r"response_format|json_schema", text, re.I):
                    self.log.write({"role": role, "event": "json_schema_rejected", "error": text[:300]})
                    self.response_mode = "json_object"       # provider does not take the schema format; fall back
                    continue
                self.protocol["api_errors"] = self.protocol.get("api_errors", 0) + 1
                self.log.write({"role": role, "event": "api_error", "try": k + 1, "error": text[:300]})
                if k < self.cfg.api_retries:
                    time.sleep(self.cfg.api_backoff_seconds * 3 ** k)
        self.protocol["api_failures"] = self.protocol.get("api_failures", 0) + 1
        self.log.write({"role": role, "event": "api_failure", "error": "model call failed after retries"})
        return None

    # -- planner
    def _computations(self) -> int:
        return self.gateway.executed_calls()

    def _wall_left(self) -> int:
        return int(self.gateway.wall_seconds_left())

    def _out_of_tools(self) -> bool:
        return self._computations() >= self.cfg.max_computations or self._wall_left() < 60

    def _planner_payload(self, must_submit: bool) -> dict:
        flagged = {p["label"] for p in self.proposals if p.get("label")}
        return {
            "task": "Rank the three most plausible mechanism families and support each with checkable evidence.",
            "verified_smiles": self.smiles, "mechanism_pool": mechanism_definitions(),
            "structural_facts": self.facts, "board": self.board.view(detailed=flagged),
            "latest": self.latest, "history": self.history[-12:],
            "proposals": self.proposals, "reusable_artifacts": self._artifacts(),
            "tool_log": self.gateway.call_log(),
            "recipes": self._recipe_cards(),
            "tools": self._professor_catalog() if self.cfg.worker else self.tool_cards,
            **({"tasks": [t.summary() for t in self.tasks.values()],
                "planner_direct_tools": sorted(sv.PLANNER_DIRECT_TOOLS)} if self.cfg.worker else {}),
            "budget": {"planner_calls_left": self.cfg.max_planner_calls - self.usage.by_role["planner"]["calls"],
                       "computations_used": self._computations(), "computations_max": self.cfg.max_computations,
                       "tool_wall_seconds_left": self._wall_left(),
                       "worker_available": self.cfg.worker, "must_submit_now": must_submit},
        }

    def _sync_costs(self) -> None:
        """Feed this run's executed calls into the cost model."""
        for c in self.gateway.calls[self._cost_seen:]:
            if c.artifact_id:
                self.costs.add(c.capability_id, c.tool_args, c.wall_seconds, c.status)
        self._cost_seen = len(self.gateway.calls)

    def _defaults(self, tool: str) -> dict:
        return {k: v["default"] for k, v in self.card_by_tool.get(tool, {"args": {}})["args"].items() if "default" in v}

    def _tool_cost(self, tool: str, args: dict | None = None) -> dict:
        self._sync_costs()
        return self.costs.estimate(tool, {**self._defaults(tool), **(args or {})})

    def _needs(self) -> tuple[bool, bool]:
        kinds = {a.artifact_id: a.kind for a in self.gateway.case_run.artifact_manifest.artifacts}
        has_geometry = kinds.get(self.gateway.prepared_artifact_id) == "prepared_structure" or any(
            k == "prepared_structure" and a.startswith("geometry:") for a, k in kinds.items())
        return not has_geometry, "amesp_baseline_bundle" not in kinds.values()

    def _recipe_cost(self, recipe: str, args: dict | None = None) -> dict:
        self._sync_costs()
        need_geometry, need_bundle = self._needs()
        return self.costs.recipe_estimate(recipe, args, need_geometry=need_geometry, need_bundle=need_bundle)

    def _professor_catalog(self) -> list[dict]:
        """Capability cards for the Planner (the same list the baselines see), with expected cost."""
        kinds_present = {a.kind for a in self.gateway.case_run.artifact_manifest.artifacts}
        return cardlib.catalog(self.tool_cards, self.profile, sv.ARG_CLASS, kinds_present, cost=self._cost_fields)

    def _cost_fields(self, tool: str) -> dict:
        fields = {"cost": self._tool_cost(tool)}
        methods = sorted({r.key for r in self.costs.records if r.tool == tool and r.key})
        if methods:
            fields["cost_by_method"] = {m: self.costs.estimate(tool, dict(zip(("functional", "grid_level"),
                                        (m.split("/")[0], int(m.split("grid")[1]))))) for m in methods}
        return fields

    def _apply(self, step: PlannerStep, n: int) -> None:
        for a in step.adopt:                         # Worker findings the Planner accepts as board evidence
            f = self.findings.get(a.finding_id)
            label = a.label or (f or {}).get("label")
            if f is None or label not in MECHANISM_POOL:
                self.history.append(f"step {n}: adopt {a.finding_id} rejected (unknown finding or no family label)")
                continue
            try:
                claim = ClaimIn(**{**{k: f[k] for k in ("statement", "kind", "atom_ids", "fact_ids", "quantities")},
                                   "label": label}).to_claim(self.facts)
                cid = self.board.add_claim(label, claim, step=n, origin="worker")
                f["adopted_as"] = cid
            except (ValueError, TypeError, KeyError) as exc:
                self.history.append(f"step {n}: adopt {a.finding_id} rejected ({exc})")
        for u in step.updates:
            fam = self.board.families[u.label]
            for key in ("plausibility", "status", "reason", "next_test"):
                if getattr(u, key) is not None:
                    setattr(fam, key, getattr(u, key))
        for c in step.claims:
            try:
                self.board.add_claim(c.label, c.to_claim(self.facts), step=n)
            except (ValueError, TypeError, KeyError) as exc:
                self.history.append(f"step {n}: claim rejected ({exc})")
        self.board.drop_claims(step.drop_claims)
        fresh: dict[str, list[str]] = {}
        for lim in step.limitations:                 # a family's limitations are restated, not accumulated
            fresh.setdefault(lim.label, []).append(lim.text)
        self.board.limitations.update(fresh)
        answered = {r.proposal_id: r for r in step.proposal_responses}
        for p in self.proposals:
            if p["proposal_id"] in answered:
                r = answered[p["proposal_id"]]
                self.log.write({"role": "proposal_response", "proposal": p, "decision": r.decision, "reason": r.reason})
        self.proposals = [p for p in self.proposals if p["proposal_id"] not in answered]

    def _execute(self, action: Action, n: int) -> str:
        if self.cfg.worker and (action.type == "recipe"
                                or (action.type == "tool" and action.name not in sv.PLANNER_DIRECT_TOOLS)):
            self.protocol["planner_rejected"] += 1
            self.latest = {"status": "rejected", "error": (
                "With the Worker on, the Planner runs only single-step, parameter-free tools "
                f"({sorted(sv.PLANNER_DIRECT_TOOLS)}) and read. Delegate experiments that need settings or several "
                "steps: give the recipe or tools and the scientific settings in fixed_args.")}
            return f"{action.type} {action.name} rejected (delegate it)"
        if action.type == "revise":
            if not self.cfg.worker:
                self.latest = {"status": "rejected", "error": "Worker is disabled in this configuration."}
                return "revise rejected (disabled)"
            report = self._revise(action, n)
            self.latest = report
            return f"revise {action.task_id} -> {report['status']}"
        if action.type == "recipe":
            if not self.cfg.recipes:
                self.latest = {"status": "rejected", "error": "Recipes are disabled in this configuration; use tools."}
                return "recipe rejected (disabled)"
            res = run_recipe(self.recipe_ctx, action.name or "", action.args)
            self.latest = res.to_json()
            return f"recipe {action.name} -> {res.status}"
        if action.type == "tool":
            obs = self.gateway.call(action.name or "", action.args)
            self.latest = obs
            return f"tool {action.name} -> {obs.get('status')}"
        if action.type == "read":
            self.latest = self.gateway.read(action.artifact_id or "", action.pointer or "")
            return f"read {action.artifact_id}{action.pointer or ''} -> {self.latest.get('status')}"
        if action.type == "delegate":
            if not self.cfg.worker:
                self.latest = {"status": "rejected", "error": "Worker is disabled in this configuration."}
                return "delegate rejected (disabled)"
            report = self._delegate(action, n)
            self.latest = report
            return f"delegate {report.get('task_id', '')} -> {report['status']} ({len(report.get('claims', []))} findings)"
        return "submit"

    # -- professor-student
    def _delegate(self, action: Action, n: int) -> dict:
        tid = f"W{len(self.tasks) + 1}"
        unknown = [t for t in action.tools if t not in self.card_by_tool]
        if unknown:
            return {"task_id": tid, "status": "rejected", "error": f"Unknown tools {unknown}; use ids from tools."}
        fixed, suggested, notes = sv.split_settings(self.gateway.resolve_args(action.fixed_args), action.suggested_args)
        problems = sv.validate_fixed_args({**suggested, **fixed}, tools=list(action.tools), recipe=action.recipe,
                                          recipe_args=RECIPE_ARGS)
        if problems:
            return {"task_id": tid, "status": "rejected", "error": "fixed_args: " + "; ".join(problems)}
        task = sv.Task(task_id=tid, objective=action.task, recipe=action.recipe, tools=list(action.tools),
                       fixed_args=fixed, separates=list(action.separates),
                       expectation=action.expectation, created_step=n)
        source = task.fixed_args.get("source_artifact_id")
        if source:
            if source not in {a.artifact_id for a in self.gateway.case_run.artifact_manifest.artifacts}:
                return {"task_id": tid, "status": "rejected", "error": f"fixed source_artifact_id {source} does not exist."}
            task.geometry = self._geometry_of(source)
        task.pinned.update({k: v for k, v in task.fixed_args.items() if sv.ARG_CLASS.get(k) == "scientific"})
        task.wall_budget = action.wall_budget_seconds
        task.suggested, task.notes = suggested, notes
        estimate = (self._recipe_cost(task.recipe, task.fixed_args) if task.recipe
                    else self.costs.combine([self._tool_cost(t, task.fixed_args) for t in task.tools]))
        expected, left = expected_seconds(estimate), self.gateway.wall_seconds_left()
        limit = min(left, task.wall_budget or float("inf"))
        if expected is not None and not action.confirm_cost and (
                expected > self.cfg.cost_check_fraction * left or expected > limit):
            self.protocol["cost_checks"] = self.protocol.get("cost_checks", 0) + 1
            check = {"status": "cost_check", "task_id": None, "estimate": estimate,
                     "tool_wall_seconds_left": round(left), "task_wall_budget": task.wall_budget,
                     "message": (f"This experiment is expected to take {'at least ' if estimate['status'] == 'lower_bound' else ''}"
                                 f"~{expected:.0f} s of the {left:.0f} s tool time left"
                                 + (f" (task budget {task.wall_budget:.0f} s)" if task.wall_budget else "")
                                 + ". Delegate again with confirm_cost=true to run it, or choose cheaper settings.")}
            self.log.write({"role": "cost_check", "objective": action.task, "estimate": estimate, "left": round(left)})
            return check
        if action.confirm_cost:
            self.protocol["cost_confirmed"] = self.protocol.get("cost_confirmed", 0) + 1
        task.estimate = estimate
        self.tasks[tid] = task
        self.log.write({"role": "task_created", "task_id": tid, "objective": action.task, "recipe": action.recipe,
                        "tools": action.tools, "fixed_args": action.fixed_args})
        return self._student(task, n)

    def _revise(self, action: Action, n: int) -> dict:
        task = self.tasks.get(action.task_id or "")
        if task is None:
            return {"task_id": action.task_id, "status": "rejected", "error": f"Unknown task {action.task_id}."}
        if len(task.instructions) >= sv.MAX_REVISIONS:
            return {"task_id": task.task_id, "status": "rejected",
                    "error": f"{task.task_id} already had {sv.MAX_REVISIONS} revisions; delegate a new task instead."}
        approved = None
        if action.approve and task.pending:
            approved = dict(task.pending)
            self._approve(task, approved)
        for key, value in self.gateway.resolve_args(action.args_hint).items():   # professor sets settings directly
            if key == "source_artifact_id":
                task.fixed_args[key] = value
                task.geometry = self._geometry_of(value)
                continue
            if sv.ARG_CLASS.get(key) == "scientific" or key in task.fixed_args:
                task.fixed_args[key] = value
            if sv.ARG_CLASS.get(key) == "scientific":
                task.pinned[key] = value
        task.instructions.append({"step": n, "instruction": action.instruction, "args_hint": action.args_hint,
                                  "approved_change": approved, "at_attempt": len(task.attempts)})
        task.pending = None
        task.retry_base = len(task.attempts)                 # a revision opens a fresh retry allowance
        self.log.write({"role": "task_revised", "task_id": task.task_id, "instruction": action.instruction,
                        "args_hint": action.args_hint, "approved": approved})
        return self._student(task, n)

    def _approve(self, task: sv.Task, change: dict) -> None:
        self.protocol["approvals"] += 1
        name, kind = change.get("name") or "", change.get("kind") or "tool"
        if name and not task.route_allowed(kind, name):
            task.approved.append({kind: name})
        args = change.get("args") or {}
        view = sv.recipe_tool_args(name, args, RECIPE_ARGS) if kind == "recipe" else args
        for key, value in {**args, **view}.items():
            if key in task.fixed_args:                   # an approved value replaces a fixed one, technical or not
                task.fixed_args[key] = value
            if sv.ARG_CLASS.get(key) == "scientific":
                task.pinned[key] = value
        source = self.gateway.resolve((change.get("args") or {}).get("source_artifact_id"))
        if source:
            task.geometry = self._geometry_of(source)

    def _geometry_of(self, artifact_id: str) -> str | None:
        kinds = {a.artifact_id: a.kind for a in self.gateway.case_run.artifact_manifest.artifacts}
        for aid in [artifact_id, *self.gateway.ancestors(artifact_id)]:
            if kinds.get(aid) in sv.GEOMETRY_KINDS:
                return aid
        return None

    def _student_cards(self, task: sv.Task) -> dict:
        def mark(card):
            return {**card, "args": {k: {**v, "class": sv.ARG_CLASS.get(k, "technical")} for k, v in card["args"].items()},
                    "applies_to_this_molecule": cardlib.applicability(card["tool"], self.profile),
                    "cost": self._tool_cost(card["tool"], task.fixed_args)}
        tools = [mark(self.card_by_tool[t]) for t in sorted(set(task.tools) | sv.PREREQUISITE_TOOLS
                                                              | sv.RECIPE_TOOLS.get(task.recipe or "", set())
                                                              | {a["tool"] for a in task.approved if a.get("tool")})
                 if t in self.card_by_tool]
        recipes = [r for r in self._recipe_cards() if r["recipe"] == task.recipe
                   or any(a.get("recipe") == r["recipe"] for a in task.approved)]
        return {"recipes": recipes, "tools": tools}

    def _student_payload(self, task: sv.Task, recent: list, comps_left: int, call: int) -> dict:
        retries = {}
        for a in task.attempts[task.retry_base:]:
            key = f"{a.kind} {a.name}"
            fails = [x for x in task.route_attempts(a.kind, a.name)
                     if x.outcome != "success" or any(w["student_fixable"] for w in x.warnings)]
            retries[key] = max(0, sv.MAX_RETRIES + 1 - len(fails)) if fails else sv.MAX_RETRIES + 1
        artifacts = [{**r, "from_task_geometry": bool(task.geometry) and self.gateway.derived_from(r["artifact_id"], task.geometry)}
                     for r in self._artifacts()]
        must = call == self.cfg.worker_max_calls or comps_left <= 0 or self._wall_left() < 60
        return {
            "assignment": {"task_id": task.task_id, "objective": task.objective, "recipe": task.recipe, "tools": task.tools,
                           "fixed_args": task.fixed_args, "separates": task.separates, "expectation": task.expectation,
                           "planner_instructions": task.instructions},
            "settings_in_force": {"locked": task.fixed_args, "scientific_in_force": task.pinned,
                                  "suggested_technical": task.suggested, "geometry": task.geometry},
            "attempts": [a.to_json() for a in task.attempts],
            "interceptions": task.interceptions[-3:],
            "recent_results": recent[-3:],
            **self._student_cards(task),
            "reusable_artifacts": artifacts,
            "structural_facts": self.facts, "family_labels": list(MECHANISM_POOL),
            "budget": {"computations_left": comps_left, "tool_wall_seconds_left": self._wall_left(),
                       **({"task_wall_seconds_left": round(task.wall_budget - sum(a.wall_seconds for a in task.attempts))}
                          if task.wall_budget else {}),
                       "attempts_left_per_route": retries, "model_calls_left": self.cfg.worker_max_calls - call,
                       "must_report_now": must},
        }

    def _attempt_from(self, task: sv.Task, kind: str, name: str, args: dict, result: dict, reason: str,
                      seconds: float) -> sv.Attempt:
        if kind == "recipe":
            steps = result.get("observations") or []
            failing = next((o for o in steps if o.get("status") not in {"success", "partial"}), None)
            outcome = "success" if result.get("status") == "success" else (failing or {}).get("status", result.get("status"))
            info = failing or {}
            warnings = [w for o in steps for w in o.get("warnings", [])]
            ids = [o["artifact_id"] for o in steps if o.get("artifact_id")]
            if result.get("status") == "rejected":
                info = {"failure_class": "system", "failure_code": "invalid_args", "failure": result.get("note")}
                outcome = "rejected"
        else:
            outcome, info, warnings = result.get("status", "error"), result, result.get("warnings", [])
            ids = [result["artifact_id"]] if result.get("artifact_id") else []
            if outcome in {"error", "rejected"}:
                info = {"failure_class": result.get("failure_class", "system"),
                        "failure_code": result.get("failure_code", outcome), "failure": result.get("error")}
        return sv.Attempt(n=len(task.attempts) + 1, kind=kind, name=name, args=args, outcome=outcome, artifact_ids=ids,
                          failure_class=info.get("failure_class"), failure_code=info.get("failure_code"),
                          failure=(info.get("failure") or info.get("error") or None) if outcome != "success" else None,
                          warnings=warnings, changed=sv.changed_from_previous(task, kind, name, args), reason=reason,
                          wall_seconds=round(seconds, 1))

    def _student(self, task: sv.Task, n: int) -> dict:
        if task.wall_budget:
            used = sum(a.wall_seconds for a in task.attempts)
            if task.wall_budget - used < 60:
                return self._report(task, {"status": "blocked_technical",
                                           "summary": f"The task's tool-time budget ({task.wall_budget:.0f} s) is used."}, n)
            saved = self.gateway.max_wall_seconds
            cap = self.gateway.wall_seconds_used() + task.wall_budget - used
            self.gateway.max_wall_seconds = cap if saved is None else min(saved, cap)
            try:
                return self._safe_session(task, n)
            finally:
                self.gateway.max_wall_seconds = saved
        return self._safe_session(task, n)

    def _safe_session(self, task: sv.Task, n: int) -> dict:
        try:
            return self._student_session(task, n)
        except Exception as exc:  # noqa: BLE001 - report the task as blocked instead of losing the run
            self._internal_error(exc, f"worker session {task.task_id}")
            try:
                return self._report(task, {"status": "blocked_technical",
                                           "summary": f"Internal error in the Worker session ({type(exc).__name__}); "
                                                      "the attempts recorded so far are listed."}, n)
            except Exception:  # noqa: BLE001
                return {"task_id": task.task_id, "status": "blocked_technical", "summary": "Internal error."}

    def _student_session(self, task: sv.Task, n: int) -> dict:
        recent: list = []
        session_intercepts: dict = {}
        start = self._computations()
        for call in range(1, self.cfg.worker_max_calls + 1):
            comps_left = min(self.cfg.worker_max_computations - (self._computations() - start),
                             self.cfg.max_computations - self._computations())
            step = self._ask(self.worker_client, "worker", self.worker_prompt,
                             self._student_payload(task, recent, comps_left, call), WorkerStep)
            if step is None:
                return self._report(task, {"status": "blocked_technical", "summary": "Worker output was invalid twice."}, n)
            a = step.action
            if a.type == "report":
                return self._report(task, a.model_dump(), n)
            if a.type == "read":
                recent.append(self.gateway.read(a.artifact_id or "", a.pointer or ""))
                continue
            if comps_left <= 0 or self._wall_left() < 60:
                recent.append({"status": "rejected", "error": "Computation budget for this session is used; report now."})
                continue
            kind, name = a.type, a.name or ""
            card = self.card_by_tool.get(name, {"args": {}})
            if kind == "recipe":
                accepted = set(RECIPE_ARGS.get(name, {})) | {"source_artifact_id"}
                defaults = {}
            else:
                accepted = set(card["args"])
                defaults = {k: v["default"] for k, v in card["args"].items() if "default" in v}
            block, args = sv.check_call(task, kind, name, self.gateway.resolve_args(a.args), gateway=self.gateway, defaults=defaults,
                                        accepted=accepted, recipe_args=RECIPE_ARGS)
            if block:
                entry = {"kind": kind, "name": name, "args": a.args, **block}
                task.interceptions.append(entry)
                if block.get("needs_approval"):
                    task.pending = {"kind": kind, "name": name, "args": a.args, "reason": a.reason}
                self.log.write({"role": "interception", "task_id": task.task_id, **entry})
                if session_intercepts.get((kind, name, json.dumps(a.args, sort_keys=True, default=str))):
                    # the same refused call again: end the session and hand the question to the Planner
                    return self._report(task, {"status": "needs_approval" if block.get("needs_approval") else "blocked_technical",
                                               "summary": f"The Worker repeated a refused call ({block['code']}): {block['message']}",
                                               "proposed_change": task.pending if block.get("needs_approval") else None}, n)
                session_intercepts[(kind, name, json.dumps(a.args, sort_keys=True, default=str))] = True
                recent.append({"status": "intercepted", **block})
                continue
            started = time.monotonic()
            if kind == "recipe":
                res = run_recipe(self.recipe_ctx, name, args)
                result, used = res.to_json(), res.geometry
            else:
                result = self.gateway.call(name, args)
                used = self._geometry_of(args["source_artifact_id"]) if args.get("source_artifact_id") else None
            attempt = self._attempt_from(task, kind, name, args, result, a.reason, time.monotonic() - started)
            task.attempts.append(attempt)
            if attempt.outcome in {"success", "partial"}:
                sv.pin_after(task, kind, name, args, defaults=defaults, recipe_args=RECIPE_ARGS, geometry_used=used)
            self.log.write({"role": "attempt", "task_id": task.task_id, **attempt.to_json()})
            recent.append(result if kind == "tool" else {"recipe": name, "status": result["status"],
                                                          "observations": result["observations"][-2:]})
        final = self._ask(self.worker_client, "worker", self.worker_prompt,      # one report-only turn
                          {**self._student_payload(task, recent, 0, self.cfg.worker_max_calls),
                           "note": "Model-call limit reached: return a report action now; any other action is ignored."},
                          WorkerStep)
        if final is not None and final.action.type == "report":
            return self._report(task, final.action.model_dump(), n)
        ran = any(a.outcome in {"success", "partial"} for a in task.attempts[task.retry_base:])
        return self._report(task, {"status": "done_with_warnings" if ran else "blocked_technical",
                                   "summary": "The Worker did not write a report; status set from the recorded attempts.",
                                   "warnings": ["No report from the Worker: results are listed in attempts, no findings drafted."]}, n)

    def _report(self, task: sv.Task, raw: dict, n: int) -> dict:
        findings = []
        for i, c in enumerate(raw.get("claims") or [], start=1):
            fid = f"{task.task_id}.f{len(task.reports) + 1}.{i}"
            f = {"finding_id": fid, **{k: c.get(k) for k in ("label", "statement", "kind", "atom_ids", "fact_ids", "quantities")}}
            f["label"] = f["label"] if f["label"] in MECHANISM_POOL else None
            self.findings[fid] = f
            findings.append(f)
        report = {"task_id": task.task_id, "status": raw.get("status"), "summary": raw.get("summary", ""),
                  "claims": findings, "warnings": [{"code": "student_concern", "message": w} for w in raw.get("warnings") or []],
                  "question_for_planner": raw.get("question_for_planner", ""),
                  "proposed_change": raw.get("proposed_change") or task.pending}
        report, corrections = sv.validate_report(task, report)
        if report["status"] == "needs_approval":
            task.pending = report["proposed_change"] or task.pending
        report["attempts"] = [{"n": a.n, "route": f"{a.kind} {a.name}", "args": a.args, "outcome": a.outcome,
                               "artifact_ids": a.artifact_ids, "changed": a.changed, "reason": a.reason,
                               **({"failure": (a.failure or "")[:200], "failure_class": a.failure_class} if a.outcome != "success" else {}),
                               **({"warnings": [w["code"] for w in a.warnings]} if a.warnings else {})}
                              for a in task.attempts]
        report["artifacts"] = [aid for a in task.attempts for aid in a.artifact_ids]
        report["interceptions"] = [{"code": i["code"], "name": i["name"]} for i in task.interceptions]
        if task.notes:
            report["system_notes"] = list(task.notes)
        if corrections:
            report["system_corrections"] = corrections
            self.protocol["status_corrections"] += len(corrections)
        task.reports.append(report)
        self.log.write({"role": "worker_report", "step": n, **{k: v for k, v in report.items() if k != "attempts"}})
        return report

    # -- critic
    def _evidence_digest(self) -> list[dict]:
        digest = []
        for aid, art in self.gateway.store.artifacts.items():
            if not aid.startswith("T"):
                continue
            from mechcal.eval.tool_gateway import by_depth, observable_fields
            root = art.outputs.get("structured_results", {})
            focus = root.get("parsed_observables") if isinstance(root.get("parsed_observables"), dict) else root
            numbers, labels, _ = observable_fields(focus, "")
            numbers = by_depth(numbers)
            digest.append({"artifact_id": aid, "tool": art.tool, "status": art.status,
                           "values": dict(numbers[:30]), "labels": dict(labels[:10])})
        return digest

    def _critic(self, n: int, *, before_submit: bool) -> None:
        try:
            self._critic_inner(n, before_submit=before_submit)
        except Exception as exc:  # noqa: BLE001 - a failed review is skipped, not fatal
            self._internal_error(exc, "critic")

    def _critic_inner(self, n: int, *, before_submit: bool) -> None:
        payload = {"verified_smiles": self.smiles, "mechanism_pool": mechanism_definitions(),
                   "structural_facts": self.facts, "calculations": self._evidence_digest(),
                   "available_recipes": [r["recipe"] for r in self._recipe_cards()]}
        out = self._ask(self.critic_client, "critic", self.critic_prompt, payload, CriticOutput)
        self.critic_runs += 1
        self.last_critic_at = self._computations()
        proposals = []
        if out is not None:
            critic_rank = [r.label for r in sorted(out.ranking, key=lambda r: -r.plausibility)]
            board_top = self.board.ranking()[:3]
            basis = {r.label: r.basis for r in out.ranking}
            tests = {t.label: t for t in out.tests}
            for label in critic_rank[:3]:
                if label not in board_top:
                    t = tests.get(label)
                    proposals.append({"label": label, "kind": "under_weighted",
                                      "text": f"Independent fact-based view ranks {label} in its top 3: {basis.get(label, '')}",
                                      "suggested_test": t.model_dump() if t else None})
            for label in board_top:
                if label in critic_rank[5:]:
                    proposals.append({"label": label, "kind": "possibly_over_weighted",
                                      "text": f"Independent view ranks {label} low: {basis.get(label, '')}"})
        check = check_submission(self.board.to_submission(), self.gateway.store, self.smiles)
        for pred in check.predictions:
            for c in pred.claims:
                if c.outcome not in ELIGIBLE | {"prior"}:
                    claim = self.board.claims_for(pred.label)[c.claim_index]
                    proposals.append({"label": pred.label, "kind": "claim_check",
                                      "text": f"Claim {claim.claim_id} fails the grounding check ({c.outcome}); fix or drop it."})
        for i, p in enumerate(proposals[:5], start=1):
            p["proposal_id"] = f"P{self.critic_runs}.{i}"
        self.proposals = proposals[:5]
        self.log.write({"role": "critic_summary", "before_submit": before_submit, "proposals": self.proposals})

    # -- main
    def run(self) -> dict:
        started = time.monotonic()
        self.log.write({"role": "start", "release": AGENT_RELEASE, "config": asdict(self.cfg),
                        "case_id": self.case_id, "smiles": self.smiles, "facts": len(self.facts["facts"])})
        submitted, pre_submit_critic_done = False, False
        n = 0
        while True:
            n += 1
            try:
                calls_used = self.usage.by_role["planner"]["calls"]
                must_submit = (calls_used >= self.cfg.max_planner_calls - 1 or self._out_of_tools()
                               or (not self.cfg.iterate and n >= 2))
                if calls_used >= self.cfg.max_planner_calls + 2:
                    self.stop_reason = "planner_call_budget_exhausted"
                    break
                step = self._ask(self.client, "planner", self.planner_prompt, self._planner_payload(must_submit), PlannerStep)
                if step is None:
                    self.history.append(f"step {n}: invalid planner output after repair")
                    continue
                missing = {p["proposal_id"] for p in self.proposals} - {r.proposal_id for r in step.proposal_responses}
                self._apply(step, n)
                if missing:
                    self.history.append(f"step {n}: proposals {sorted(missing)} were not answered; answer them")
                action = step.action
                if action.type == "submit" or must_submit:
                    if self.cfg.critic and not pre_submit_critic_done and not must_submit:
                        pre_submit_critic_done = True
                        self._critic(n, before_submit=True)
                        if self.proposals:
                            self.history.append(f"step {n}: submit requested; Critic raised proposals to answer first")
                            self.latest = {"note": "Pre-submission Critic check. Answer the proposals, then submit or test."}
                            continue
                    submitted = True
                    self.stop_reason = "planner_submitted" if action.type == "submit" else "budget"
                    self.history.append(f"step {n}: submit")
                    break
                if self._out_of_tools() and action.type in {"recipe", "tool", "delegate", "revise"}:
                    self.latest = {"status": "rejected", "error": "Computation budget exhausted; submit now."}
                    self.history.append(f"step {n}: {action.type} rejected (budget)")
                    continue
                summary = self._execute(action, n)
                sep = ",".join(action.separates)
                self.history.append(f"step {n}: {summary}" + (f" [separates {sep}]" if sep else ""))
                self.log.write({"role": "action", "step": n, "action": action.model_dump(), "summary": summary,
                                "board_top3": self.board.ranking()[:3]})
                if (self.cfg.critic and self.cfg.iterate
                        and self._computations() - self.last_critic_at >= self.cfg.critic_every):
                    self._critic(n, before_submit=False)
            except Exception as exc:  # noqa: BLE001 - a bug must not lose the run
                if self._internal_error(exc, f"planner step {n}"):
                    self.stop_reason = "internal_error"
                    break

        try:
            return self._finish(submitted, time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 - still leave a record of the run
            self._internal_error(exc, "finish")
            minimal = {"case_id": self.case_id, "release": AGENT_RELEASE, "status": "stopped", "stop_reason": "finish_failed",
                       "error": f"{type(exc).__name__}: {exc}", "usage": self.usage.as_dict(), "tool_cost": self.gateway.cost(),
                       "board": self.board.model_dump()}
            (self.run_dir / "result.json").write_text(json.dumps(minimal, ensure_ascii=False, indent=1, default=str))
            return minimal

    def _internal_error(self, exc: Exception, where: str) -> bool:
        """Record an unexpected exception; True when the run should stop."""
        self.protocol["internal_errors"] = self.protocol.get("internal_errors", 0) + 1
        self.log.write({"role": "internal_error", "where": where, "error": f"{type(exc).__name__}: {exc}"[:500],
                        "traceback": traceback.format_exc()[-3000:]})
        self.history.append(f"{where}: internal error ({type(exc).__name__}); the step was skipped")
        self.latest = {"status": "error", "error": "An internal error interrupted the last step; it was skipped."}
        return self.protocol["internal_errors"] >= self.cfg.max_internal_errors

    def _protocol_metrics(self) -> dict:
        tasks = list(self.tasks.values())
        repaired = hit = 0
        for t in tasks:          # self-repair: a route's system failure later fixed with no revision in between
            bounds = [0, *[i["at_attempt"] for i in t.instructions], len(t.attempts)]
            for lo, hi in zip(bounds, bounds[1:]):
                segment = t.attempts[lo:hi]
                for route in {(a.kind, a.name) for a in segment}:
                    runs = [a for a in segment if (a.kind, a.name) == route]
                    first = next((i for i, a in enumerate(runs) if a.outcome != "success" and a.failure_class == "system"), None)
                    if first is None:
                        continue
                    hit += 1
                    repaired += any(a.outcome == "success" for a in runs[first + 1:])
        codes: dict[str, int] = {}
        for t in tasks:
            for i in t.interceptions:
                codes[i["code"]] = codes.get(i["code"], 0) + 1
        return {"delegations": len(tasks), "revisions": sum(len(t.instructions) for t in tasks),
                "attempts": sum(len(t.attempts) for t in tasks), "interceptions": codes,
                "system_failure_episodes": hit, "self_repaired": repaired,
                "report_statuses": [r["status"] for t in tasks for r in t.reports],
                "findings": len(self.findings), "findings_adopted": sum(1 for f in self.findings.values() if f.get("adopted_as")),
                **self.protocol}

    def _finish(self, submitted: bool, seconds: float) -> dict:
        self._sync_costs()
        self.costs.append_log([r for r in self.costs.records if r.note == "this run"])
        raw = self.board.to_submission()
        check = check_submission(raw, self.gateway.store, self.smiles)
        eligible = None
        if self.cfg.self_check:
            eligible = set()
            for pred in check.predictions:
                claims = self.board.claims_for(pred.label)
                for c in pred.claims:
                    if c.outcome in ELIGIBLE | {"prior"}:
                        eligible.add(claims[c.claim_index].claim_id)
        submission = self.board.to_submission(eligible=eligible)
        result = {
            "case_id": self.case_id, "release": AGENT_RELEASE, "status": "submitted" if submitted else "stopped",
            "stop_reason": self.stop_reason, "seconds": round(seconds, 1), "config": asdict(self.cfg),
            "usage": self.usage.as_dict(), "tool_cost": self.gateway.cost(), "critic_runs": self.critic_runs,
            "self_check_removed": [c.claim_id for c in self.board.claims if eligible is not None and c.claim_id not in eligible
                                   and c.label in self.board.ranking()[:3]],
            "protocol": self._protocol_metrics(),
            "submission": submission.model_dump(), "board": self.board.model_dump(),
            "raw_grounding": check.model_dump(),
        }
        (self.run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
        (self.run_dir / "artifacts.json").write_text(self.gateway.store.model_dump_json())
        self.log.write({"role": "finish", "status": result["status"], "stop_reason": self.stop_reason,
                        "usage": result["usage"], "tool_cost": result["tool_cost"]})
        return result
