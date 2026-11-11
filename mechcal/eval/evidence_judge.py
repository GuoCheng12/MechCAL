"""Layer B: evidence-rubric sufficiency judge over Layer-A-checked claims (release 0.3.0 evaluation).

The judge receives the input graph, each claim with its evaluator grounding outcome and,
for computed claims, the cited artifact values. Its answer is validated in code:
conditions must cite claims of an accepted kind that are Layer-A eligible and that the
judge itself marked consistent; the fabrication gate is then applied to the score.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mechcal.eval.submission import ArtifactStore, Prediction, resolve_pointer
from mechcal.eval.grounding import ELIGIBLE, PredictionCheck, check_prediction, gate_support
from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.zero_shot_v3 import structure_facts

RUBRIC_PATH = Path(__file__).parent / "rubrics/mechanism_evidence_rubric.json"
FREEZE_PATH = Path(__file__).parent / "rubrics/mechanism_evidence_rubric.freeze.json"
PROMPT_NAME = "mechanism_evidence_grounded_judge.md"
PROMPT_PATH = Path(__file__).parent / "prompts" / PROMPT_NAME
MAX_CONTEXT_CHARS = 1500
WARRANT = "warrant"


def load_rubric() -> dict:
    return json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))


def judge_rubric_view(rubric: dict, label: str) -> dict:
    """Rubric sections the judge sees. Evaluator coverage notes (tool names) are excluded."""
    lab = copy.deepcopy(rubric["labels"][label])
    lab.pop("ceilings", None)
    return {
        "mechanism_evidence_rubric_version": rubric["mechanism_evidence_rubric_version"],
        "evidence_kinds": rubric["evidence_kinds"],
        "global_rules": rubric["global_rules"],
        "decision_policy": rubric["decision_policy"],
        "support_score_scale": rubric["support_score_scale"],
        "mechanism_rubric": lab,
    }


def claim_id(index: int) -> str:
    return f"C{index + 1}"


def _context(outputs: Any, pointer: str) -> Any:
    parent = pointer.rsplit("/", 1)[0] if "/" in pointer.lstrip("/") else ""
    try:
        value = resolve_pointer(outputs, parent)
    except KeyError:
        return None
    text = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(text) <= MAX_CONTEXT_CHARS else text[:MAX_CONTEXT_CHARS] + " …[truncated]"


def claim_payload(prediction: Prediction, check: PredictionCheck, store: ArtifactStore) -> list[dict]:
    claims = []
    for i, (claim, result) in enumerate(zip(prediction.evidence, check.claims)):
        entry = {"claim_id": claim_id(i), "kind": claim.kind, "statement": claim.statement,
                 "grounding_outcome": result.outcome, "atom_ids": claim.atom_ids}
        if claim.kind == "computed":
            cited = []
            for q in claim.quantities:
                item = {"name": q.name, "claimed_value": q.value, "unit": q.unit}
                if q.source and (art := store.artifacts.get(q.source.artifact_id)):
                    item.update(artifact_id=art.artifact_id, artifact_status=art.status,
                                pointer=q.source.pointer, artifact_context=_context(art.outputs, q.source.pointer))
                    try:
                        item["artifact_value"] = resolve_pointer(art.outputs, q.source.pointer)
                    except KeyError:
                        item["artifact_value"] = None
                cited.append(item)
            entry["cited_values"] = cited
        claims.append(entry)
    return claims


def judge_payload(prediction: Prediction, check: PredictionCheck, store: ArtifactStore,
                  smiles: str, rubric: dict) -> dict:
    return {
        **judge_rubric_view(rubric, prediction.label),
        "submitted_prediction": {"label": prediction.label, "limitations": prediction.limitations},
        "claims": claim_payload(prediction, check, store),
        "structural_source": structure_facts(smiles),
    }


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClaimVerification(_M):
    claim_id: str
    outcome: Literal["consistent", "contradicted", "beyond_source"]
    explanation: str = Field(min_length=1)


class Condition(_M):
    condition_index: int = Field(ge=1)
    satisfied: bool
    claim_ids: list[str]
    explanation: str = Field(min_length=1)


class Judgement(_M):
    evidence_support_score: Literal[0.0, 0.5, 1.0]
    verifications: list[ClaimVerification]
    selected_route_kind: Literal["strong_support", "partial_support", "none"]
    selected_route_id: str | None
    conditions: list[Condition]
    overstatement: str
    rationale: str = Field(min_length=1)


class JudgeResponse(_M):
    judgement: Judgement


def response_schema(label: str, rubric: dict) -> dict:
    """One schema alternative per route plus zero, so route/score/conditions agree structurally."""
    schema = JudgeResponse.model_json_schema()
    template = schema["$defs"].pop("Judgement")
    variants = []
    for kind, score in (("strong_support", 1.0), ("partial_support", 0.5)):
        for route in rubric["labels"][label][kind]["any_of"]:
            v = copy.deepcopy(template)
            p = v["properties"]
            p["evidence_support_score"] = {"type": "number", "enum": [score]}
            p["selected_route_kind"] = {"type": "string", "enum": [kind]}
            p["selected_route_id"] = {"type": "string", "enum": [route["id"]]}
            p["conditions"].update(minItems=len(route["all_of"]), maxItems=len(route["all_of"]))
            variants.append(v)
    zero = copy.deepcopy(template)
    zero["properties"].update(evidence_support_score={"type": "number", "enum": [0.0]},
                              selected_route_kind={"type": "string", "enum": ["none"]},
                              selected_route_id={"type": "null"})
    zero["properties"]["conditions"].update(maxItems=0)
    variants.append(zero)
    schema["properties"]["judgement"] = {"anyOf": variants}
    return schema


def validate_judgement(raw: dict, prediction: Prediction, check: PredictionCheck, rubric: dict) -> tuple[Judgement, list[int]]:
    """Validate the judge's answer against the rubric and Layer A. Returns (judgement, used claim indices)."""
    j = JudgeResponse.model_validate(raw).judgement
    ids = {claim_id(i): i for i in range(len(prediction.evidence))}
    verdict = {}
    for v in j.verifications:
        if v.claim_id not in ids:
            raise ValueError(f"Verification cites unknown claim {v.claim_id}")
        verdict[v.claim_id] = v.outcome
    if j.evidence_support_score == 0.0:
        if j.selected_route_kind != "none" or j.selected_route_id is not None or j.conditions:
            raise ValueError("A zero score must not select a route or list conditions")
        return j, []
    kind = "strong_support" if j.evidence_support_score == 1.0 else "partial_support"
    if j.selected_route_kind != kind:
        raise ValueError("Selected route kind disagrees with the score")
    route = next((r for r in rubric["labels"][prediction.label][kind]["any_of"] if r["id"] == j.selected_route_id), None)
    if route is None:
        raise ValueError("Selected route does not exist for this mechanism")
    if [c.condition_index for c in j.conditions] != list(range(1, len(route["all_of"]) + 1)):
        raise ValueError(f"Assess every condition of route {route['id']} in order")
    outcome = {claim_id(c.claim_index): c.outcome for c in check.claims}
    used: set[int] = set()
    for cond, spec in zip(j.conditions, route["all_of"]):
        if not cond.satisfied:
            raise ValueError("A positive score requires every condition of the selected route")
        if not cond.claim_ids or any(cid not in ids for cid in cond.claim_ids):
            raise ValueError(f"Condition {cond.condition_index} must cite existing claims")
        if any(verdict.get(cid) != "consistent" for cid in cond.claim_ids):
            raise ValueError(f"Condition {cond.condition_index} cites a claim not verified as consistent")
        accepts = set(spec["accepts"])
        findings = [cid for cid in cond.claim_ids
                    if prediction.evidence[ids[cid]].kind in accepts and outcome[cid] in ELIGIBLE]
        if accepts - {WARRANT}:
            if not findings and WARRANT not in accepts:
                raise ValueError(f"Condition {cond.condition_index} needs an eligible {sorted(accepts)} claim")
            used.update(ids[cid] for cid in findings)
        for cid in cond.claim_ids:
            if outcome[cid] not in ELIGIBLE | {"prior"}:
                raise ValueError(f"Condition {cond.condition_index} relies on {cid} ({outcome[cid]})")
    if not used:
        raise ValueError("A positive score must rest on at least one eligible finding")
    return j, sorted(used)


def final_score(judgement: Judgement, check: PredictionCheck, used: list[int]) -> float:
    return gate_support(float(judgement.evidence_support_score), check, used_claims=used)


def freeze_manifest() -> dict:
    """Content hashes of everything that defines the evidence score; recorded before system runs."""
    schema_sample = {label: response_schema(label, load_rubric()) for label in sorted(load_rubric()["labels"])}

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    return {
        "rubric_sha256": sha(RUBRIC_PATH.read_bytes()),
        "prompt_sha256": sha(PROMPT_PATH.read_bytes()),
        "schema_sha256": sha(json.dumps(schema_sample, sort_keys=True).encode()),
        "grounding_module_sha256": sha((Path(__file__).parent / "grounding.py").read_bytes()),
    }


class EvidenceJudge:
    """Callable judge: (prediction, store, smiles) -> scored record. ``client`` must expose
    ``complete_json(system, user_payload, schema) -> dict`` (an adapter over the runtime client)."""

    def __init__(self, client, *, max_repairs: int = 1) -> None:
        self.client = client
        self.rubric = load_rubric()
        self.prompt = load_eval_prompt(PROMPT_NAME)
        self.max_repairs = max_repairs

    def judge(self, prediction: Prediction, store: ArtifactStore, smiles: str) -> dict:
        check = check_prediction(prediction, store, smiles)
        if not prediction.evidence:
            return {"score": 0.0, "status": "empty_evidence", "grounding": check.model_dump()}
        payload = judge_payload(prediction, check, store, smiles, self.rubric)
        schema = response_schema(prediction.label, self.rubric)
        feedback = None
        for attempt in range(self.max_repairs + 1):
            raw = self.client.complete_json(self.prompt, {**payload, **({"schema_feedback": feedback} if feedback else {})}, schema)
            try:
                judgement, used = validate_judgement(raw, prediction, check, self.rubric)
            except ValueError as exc:
                feedback = str(exc)
                continue
            return {"score": final_score(judgement, check, used), "judge_score": judgement.evidence_support_score,
                    "used_claims": used, "status": "scored", "attempts": attempt + 1,
                    "judgement": judgement.model_dump(), "grounding": check.model_dump()}
        return {"score": None, "status": "invalid_judgement", "error": feedback, "grounding": check.model_dump()}
