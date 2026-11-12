"""MechCAL agent loop (release 0.3.0): Planner over a hypothesis board, optional Worker and Critic.

Every tool call goes through the recorded gateway; every model call is logged with its
payload size and usage; the final answer is a projection of the board, filtered by the
deterministic grounding check when ``self_check`` is on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mechcal.agent.board import Board, ClaimIn, Status
from mechcal.agent.facts import structural_facts
from mechcal.agent.recipes import RECIPE_ARGS, RECIPES, RecipeContext, run_recipe
from mechcal.eval.grounding import ELIGIBLE, check_submission
from mechcal.eval.grounded_subjects import UsageLedger
from mechcal.eval.tool_gateway import ToolGateway
from mechcal.mechanism_pool import MECHANISM_POOL, mechanism_definitions
from mechcal.runtime.prompts import load_prompt

AGENT_RELEASE = "0.3.0"
EXCLUDED_TOOLS = {
    "microscopic.list_artifact_bundles", "microscopic.list_artifact_bundle_members",
    "microscopic.inspect_raw_artifact_bundle", "microscopic.parse_snapshot_outputs",
    "microscopic.extract_torsion_candidates_from_bundle", "microscopic.unsupported_excited_state_relaxation",
    "microscopic.run_excited_state_relaxation", "microscopic.scan_excited_state_coordinate",
    "microscopic.estimate_soc_amesp", "microscopic.list_available_conformers",
}


@dataclass
class AgentConfig:
    max_planner_calls: int = 16
    max_computations: int = 12
    max_tool_wall_seconds: float = 3600.0    # total tool wall time per run; long requests are clamped to what is left
    critic: bool = True
    critic_every: int = 3
    worker: bool = True
    worker_max_calls: int = 6
    worker_max_computations: int = 4
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
ACTIONS = {"recipe", "tool", "delegate", "read", "submit"}


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
    if "arguments" in a and "args" not in a:
        a["args"] = a.pop("arguments")
    return a


def normalize_planner(data):
    if not isinstance(data, dict):
        return data
    d = dict(data)
    d["updates"] = [_norm_family_item(u) for u in d.get("updates") or []]
    d["claims"] = [_norm_claim(c) for c in d.get("claims") or []]
    d["limitations"] = _norm_limitations(d.get("limitations"))
    d["action"] = _norm_action(d.get("action") or {})
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
    type: Literal["recipe", "tool", "delegate", "read", "submit"]
    name: str | None = None
    args: dict = Field(default_factory=dict)
    separates: list[str] = Field(default_factory=list)
    expectation: str | dict[str, str] = ""
    task: str = ""
    artifact_id: str | None = None
    pointer: str | None = None

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
        if self.type == "read" and not self.artifact_id:
            raise ValueError("action.artifact_id is required for read")
        return self


class PlannerStep(_M):
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        return normalize_planner(data)

    reading: str = ""
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


class WorkerAction(_M):
    type: Literal["recipe", "tool", "read", "finish"]
    name: str | None = None
    args: dict = Field(default_factory=dict)
    artifact_id: str | None = None
    pointer: str | None = None
    report: str = ""
    claims: list[ClaimIn] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class WorkerStep(_M):
    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            a = _norm_action(data.get("action") or {}, allowed={"recipe", "tool", "read", "finish"})
            if isinstance(a, dict):
                a["claims"] = [_norm_claim(c) for c in a.get("claims") or []]
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
        if isinstance(data, dict):
            data = {**data, "ranking": [_norm_family_item(r) for r in data.get("ranking") or []],
                    "tests": [_norm_family_item(t) for t in data.get("tests") or []]}
        return data

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
        self.tool_cards = self._tool_cards()
        self.gateway.allowed = tuple(card["tool"] for card in self.tool_cards)
        self.planner_prompt = load_prompt("agent_planner.md")
        self.worker_prompt = load_prompt("agent_worker.md")
        self.critic_prompt = load_prompt("agent_critic.md")

    # -- catalog
    def _tool_cards(self) -> list[dict]:
        ids = [i for i in self.gateway.tool_ids() if i.startswith("microscopic.") and i not in EXCLUDED_TOOLS]
        cards = []
        for d in self.gateway.describe(ids):
            props = (d.get("tool_arg_schema") or {}).get("properties") or {}
            args = {}
            for k, v in props.items():
                spec = {key: v[key] for key in ("type", "default", "minimum", "maximum", "enum", "minItems", "maxItems")
                        if key in v and v[key] is not None}
                items = v.get("items") or next((x.get("items") for x in v.get("anyOf", []) if x.get("items")), None)
                if isinstance(items, dict) and items.get("enum"):
                    spec["items_enum"] = items["enum"]
                if not spec.get("type") and v.get("anyOf"):
                    spec["type"] = "|".join(str(x.get("type")) for x in v["anyOf"] if x.get("type"))
                args[k] = spec
            cards.append({"tool": d["capability_id"], "status": d.get("status"),
                          "description": (d.get("description") or "")[:240], "args": args})
        return cards

    def _artifacts(self) -> list[dict]:
        """Reusable inputs produced so far (geometries, bundles), for exact source_artifact_id choice."""
        return [{"artifact_id": a.artifact_id, "kind": a.kind, "status": a.status}
                for a in self.gateway.case_run.artifact_manifest.artifacts]

    def _recipe_cards(self) -> list[dict]:
        if not self.cfg.recipes:
            return []
        specs = {c["tool"]: c["args"] for c in self.tool_cards}
        cards = []
        for name, (_, desc) in RECIPES.items():
            args = {arg: specs.get(tool, {}).get(targ, {}) for arg, (tool, targ) in RECIPE_ARGS.get(name, {}).items()}
            cards.append({"recipe": name, "description": desc, **({"args": args} if args else {})})
        return cards

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
            try:
                result = client.complete_json_result(system_prompt=system, payload=payload, schema_feedback=feedback,
                                                     response_format=self._format(role, model_cls))
            except Exception as exc:  # noqa: BLE001 - provider rejected the schema format; fall back once
                if self.response_mode != "json_schema":
                    raise
                self.log.write({"role": role, "event": "json_schema_rejected", "error": str(exc)[:300]})
                self.response_mode = "json_object"
                result = client.complete_json_result(system_prompt=system, payload=payload, schema_feedback=feedback,
                                                     response_format=self._format(role, model_cls))
            self.usage.add(role, result.usage)
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
            "recipes": self._recipe_cards(), "tools": self.tool_cards,
            "budget": {"planner_calls_left": self.cfg.max_planner_calls - self.usage.by_role["planner"]["calls"],
                       "computations_used": self._computations(), "computations_max": self.cfg.max_computations,
                       "tool_wall_seconds_left": self._wall_left(),
                       "worker_available": self.cfg.worker, "must_submit_now": must_submit},
        }

    def _apply(self, step: PlannerStep, n: int) -> None:
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
            report = self._worker(action, n)
            self.latest = report
            return f"delegate -> {report['status']} ({len(report.get('claims_added', []))} claims)"
        return "submit"

    # -- worker
    def _worker(self, action: Action, n: int) -> dict:
        observations: list = []
        start_comp = self._computations()
        for call in range(1, self.cfg.worker_max_calls + 1):
            budget_left = min(self.cfg.worker_max_computations - (self._computations() - start_comp),
                              self.cfg.max_computations - self._computations())
            payload = {"task": action.task, "separates": action.separates, "verified_smiles": self.smiles,
                       "structural_facts": self.facts, "recipes": self._recipe_cards(), "tools": self.tool_cards,
                       "family_labels": list(MECHANISM_POOL),
                       "reusable_artifacts": self._artifacts(), "tool_log": self.gateway.call_log(),
                       "results_so_far": observations[-4:],
                       "computations_left": budget_left, "tool_wall_seconds_left": self._wall_left(),
                       "must_finish_now": call == self.cfg.worker_max_calls or budget_left <= 0 or self._wall_left() < 60}
            step = self._ask(self.worker_client, "worker", self.worker_prompt, payload, WorkerStep)
            if step is None:
                return {"status": "invalid_worker_output", "observations": observations[-4:]}
            a = step.action
            if a.type == "finish":
                added, findings = [], []
                for c in a.claims:
                    if c.label not in MECHANISM_POOL:
                        findings.append({"label_given": c.label, "statement": c.statement, "kind": c.kind,
                                         "quantities": c.quantities})
                        continue
                    try:
                        added.append(self.board.add_claim(c.label, c.to_claim(self.facts), step=n, origin="worker"))
                    except (ValueError, TypeError, KeyError) as exc:
                        a.unresolved.append(f"claim not recorded: {exc}")
                return {"status": "finished", "report": a.report, "unresolved": a.unresolved,
                        "claims_added": added, "findings_without_family": findings, "observations": observations[-3:]}
            if (budget_left <= 0 or self._wall_left() < 60) and a.type in {"recipe", "tool"}:
                observations.append({"status": "rejected", "error": "Worker computation budget exhausted; finish now."})
                continue
            if a.type == "recipe" and self.cfg.recipes:
                observations.append(run_recipe(self.recipe_ctx, a.name or "", a.args).to_json())
            elif a.type == "tool":
                observations.append(self.gateway.call(a.name or "", a.args))
            elif a.type == "read":
                observations.append(self.gateway.read(a.artifact_id or "", a.pointer or ""))
            else:
                observations.append({"status": "rejected", "error": "Unknown or disabled action."})
        return {"status": "budget_exhausted", "observations": observations[-3:]}

    # -- critic
    def _evidence_digest(self) -> list[dict]:
        digest = []
        for aid, art in self.gateway.store.artifacts.items():
            if not aid.startswith("T"):
                continue
            from mechcal.eval.tool_gateway import observable_fields
            root = art.outputs.get("structured_results", {})
            focus = root.get("parsed_observables") if isinstance(root.get("parsed_observables"), dict) else root
            numbers, labels, _ = observable_fields(focus, "")
            digest.append({"artifact_id": aid, "tool": art.tool, "status": art.status,
                           "values": dict(numbers[:30]), "labels": dict(labels[:10])})
        return digest

    def _critic(self, n: int, *, before_submit: bool) -> None:
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
            if self._out_of_tools() and action.type in {"recipe", "tool", "delegate"}:
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
        return self._finish(submitted, time.monotonic() - started)

    def _finish(self, submitted: bool, seconds: float) -> dict:
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
            "submission": submission.model_dump(), "board": self.board.model_dump(),
            "raw_grounding": check.model_dump(),
        }
        (self.run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
        (self.run_dir / "artifacts.json").write_text(self.gateway.store.model_dump_json())
        self.log.write({"role": "finish", "status": result["status"], "stop_reason": self.stop_reason,
                        "usage": result["usage"], "tool_cost": result["tool_cost"]})
        return result
