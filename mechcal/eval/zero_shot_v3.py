"""Scoped V3 pilot: source-check zero-shot findings against a parsed SMILES graph."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mechcal.eval.prompts import load_eval_prompt
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.llm import OpenAIJsonClient

PROMPT_NAME = "mechanism_evidence_support_judge_v3_zero_shot.md"
PROMPT_VERSION = "2026-10-09-v3-route-contract-v2"
SEMANTIC_SCOPE_PROMPT_VERSION = "2026-10-10-v3-semantic-scope-v1-candidate"
RUBRIC_PATH = Path(__file__).parent / "rubrics/mechanism_evidence_rubric_v3.json"


def compose_v3_judge_prompt(base_prompt: str, *, semantic_scope: bool = False) -> str:
    """Opt-in candidate policy; failed calibration must not change formal scores."""
    if not semantic_scope:
        return base_prompt
    return base_prompt.rstrip() + "\n\n" + load_eval_prompt("evidence_interpretation_policy.md")


class EvidenceSubmission(BaseModel):
    """Evaluation input; an explicitly empty evidence list is not a parse failure."""

    model_config = ConfigDict(extra="forbid")
    label: str
    rank: int = Field(ge=1, le=3)
    confidence: float = Field(ge=0, le=1)
    claim_status: str
    support_strength: str
    evidence: list[str]
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_submitted_text(self):
        if not all(text.strip() for text in self.evidence + self.limitations):
            raise ValueError("Evidence and limitations must contain nonempty text")
        return self


class Prediction(EvidenceSubmission):
    evidence: list[str] = Field(min_length=1)


class Predictions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mechanism_predictions: list[Prediction] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def check_ranking(self):
        labels = [p.label for p in self.mechanism_predictions]
        if len(set(labels)) != 3 or any(label not in MECHANISM_POOL for label in labels):
            raise ValueError("Require three distinct labels from the fixed mechanism pool")
        if [p.rank for p in self.mechanism_predictions] != [1, 2, 3]:
            raise ValueError("Predictions must be ordered with ranks 1, 2, 3")
        if any(
            not all(text.strip() for text in p.evidence + p.limitations)
            for p in self.mechanism_predictions
        ):
            raise ValueError("Evidence and limitations must contain nonempty text")
        return self


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    submitted_quote: str = Field(min_length=1)
    outcome: Literal["consistent", "contradicted", "beyond_source", "unavailable"]
    atom_ids: list[int]
    explanation: str = Field(min_length=1)


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition_index: int = Field(ge=1)
    satisfied: bool
    verification_indices: list[int]
    explanation: str = Field(min_length=1)


class Judgement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["scored", "pending"]
    evidence_support_score: Literal[0.0, 0.5, 1.0] | None
    verifications: list[Verification]
    selected_route_kind: Literal["strong_support", "partial_support", "none"]
    selected_route_id: str | None
    conditions: list[Condition]
    overstatement: str
    rationale: str = Field(min_length=1)


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judgement: Judgement


def judge_response_schema(label: str, rubric: dict) -> dict:
    """Constrain route/score combinations without changing the rubric or saved judgement."""
    schema = JudgeResponse.model_json_schema()
    template = schema["$defs"].pop("Judgement")
    # Model-scored submissions still require actual quoted findings. Only the
    # local empty-submission rule may produce a judgement without verifications.
    template["properties"]["verifications"]["minItems"] = 1
    condition_props = schema["$defs"]["Condition"]["properties"]
    condition_props["satisfied"] = {"type": "boolean", "enum": [True]}
    condition_props["verification_indices"]["items"]["minimum"] = 1
    condition_props["verification_indices"]["minItems"] = 1
    variants = []
    for kind, score in (("strong_support", 1.0), ("partial_support", 0.5)):
        for route in rubric["labels"][label][kind]["any_of"]:
            variant = copy.deepcopy(template)
            props = variant["properties"]
            props["status"] = {"type": "string", "enum": ["scored"]}
            props["evidence_support_score"] = {"type": "number", "enum": [score]}
            props["selected_route_kind"] = {"type": "string", "enum": [kind]}
            props["selected_route_id"] = {"type": "string", "enum": [route["id"]]}
            props["conditions"].update(
                minItems=len(route["all_of"]), maxItems=len(route["all_of"])
            )
            variants.append(variant)
    for status, score in (("scored", 0.0), ("pending", None)):
        variant = copy.deepcopy(template)
        props = variant["properties"]
        props["status"] = {"type": "string", "enum": [status]}
        props["evidence_support_score"] = (
            {"type": "null"} if score is None else {"type": "number", "enum": [score]}
        )
        props["selected_route_kind"] = {"type": "string", "enum": ["none"]}
        props["selected_route_id"] = {"type": "null"}
        props["conditions"].update(maxItems=0)
        variants.append(variant)
    # Structured Outputs requires an object root; the alternatives are nested.
    schema["properties"]["judgement"] = {"anyOf": variants}
    return schema


def structure_facts(smiles: str) -> dict:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Input SMILES could not be parsed by RDKit")
    canonical = Chem.MolToSmiles(mol, canonical=True)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return {
        "source": "evaluator_parsed_verified_smiles",
        "rdkit_version": rdBase.rdkitVersion,
        "smiles": smiles,
        "canonical_smiles": canonical,
        "mapped_smiles": Chem.MolToSmiles(mol, canonical=False),
        "atoms": [
            {
                "id": a.GetIdx() + 1,
                "element": a.GetSymbol(),
                "formal_charge": a.GetFormalCharge(),
                "aromatic": a.GetIsAromatic(),
                "hydrogens": a.GetTotalNumHs(),
            }
            for a in mol.GetAtoms()
        ],
        "bonds": [
            {"a": b.GetBeginAtomIdx() + 1, "b": b.GetEndAtomIdx() + 1, "type": str(b.GetBondType())}
            for b in mol.GetBonds()
        ],
        "rings": [[i + 1 for i in ring] for ring in mol.GetRingInfo().AtomRings()],
        "rotatable_bond_count": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "scope": (
            "Molecular connectivity only. No geometry, excited states, packing or measurements."
        ),
    }


def judge_payload(prediction: EvidenceSubmission, facts: dict, rubric: dict) -> dict:
    return {
        "mechanism_evidence_rubric_version": "v3",
        "global_rules": rubric["global_rules"],
        "decision_policy": rubric["decision_policy"],
        "verification_policy": rubric["verification_policy"],
        "support_score_scale": rubric["support_score_scale"],
        "mechanism_rubric": rubric["labels"][prediction.label],
        "submitted_prediction": {
            "label": prediction.label,
            "evidence": prediction.evidence,
            "limitations": prediction.limitations,
        },
        "structural_source": facts,
        "source_availability": {
            "subject_scientific_tool_calls": 0,
            "tool_records": [],
            "experimental_observations": [],
        },
    }


def validate_judgement(
    raw: dict, prediction: EvidenceSubmission, facts: dict, rubric: dict,
) -> Judgement:
    judgement = Judgement.model_validate(raw)
    if not prediction.evidence:
        if (
            judgement.status != "scored"
            or judgement.evidence_support_score != 0.0
            or judgement.verifications
            or judgement.selected_route_kind != "none"
            or judgement.selected_route_id is not None
            or judgement.conditions
        ):
            raise ValueError(
                "Empty submitted evidence requires zero support and no findings or route"
            )
        return judgement
    if not judgement.verifications:
        raise ValueError("Nonempty evidence requires at least one verification")
    atom_ids = {a["id"] for a in facts["atoms"]}
    for finding in judgement.verifications:
        if not any(finding.submitted_quote in text for text in prediction.evidence):
            raise ValueError("Verification quote is not from the submitted evidence")
        if not set(finding.atom_ids).issubset(atom_ids):
            raise ValueError("Verification refers to an atom outside the input graph")
    score = judgement.evidence_support_score
    if judgement.status == "pending":
        if score is not None or not any(
            v.outcome == "unavailable" for v in judgement.verifications
        ):
            raise ValueError("Pending requires an unavailable essential finding and a null score")
    elif score is None:
        raise ValueError("A scored judgement must have an explicit numeric score")
    if score in {0.5, 1.0}:
        kind = "strong_support" if score == 1 else "partial_support"
        if judgement.selected_route_kind != kind:
            raise ValueError("Selected route kind disagrees with the score")
        routes = rubric["labels"][prediction.label][kind]["any_of"]
        route = next((r for r in routes if r["id"] == judgement.selected_route_id), None)
        if route is None:
            raise ValueError("Selected route ID does not exist for this mechanism")
        if [c.condition_index for c in judgement.conditions] != list(
            range(1, len(route["all_of"]) + 1)
        ):
            raise ValueError(
                "Every condition of the selected route must be assessed in order. "
                f"For route {route['id']}, fill {len(route['all_of'])} condition records "
                f"for this checklist: {json.dumps(route['all_of'])}. "
                "A positive score cannot use an empty conditions list. If a required "
                "condition is not supported, reconsider the route and numeric score."
            )
        if not all(c.satisfied for c in judgement.conditions):
            raise ValueError("A positive score requires every selected route condition to hold")
        if any(not c.verification_indices for c in judgement.conditions):
            raise ValueError("Every positive route condition must cite a submitted finding")
        used = [i for c in judgement.conditions for i in c.verification_indices]
        if not used or any(i < 1 or i > len(judgement.verifications) for i in used):
            raise ValueError("Positive support must point to actual submitted findings")
        if any(judgement.verifications[i - 1].outcome != "consistent" for i in used):
            raise ValueError("A positive route cannot rely on an ineligible finding")
    elif (
        judgement.selected_route_kind != "none"
        or judgement.selected_route_id is not None
        or judgement.conditions
    ):
        raise ValueError("Zero or pending must not select a completed positive route")
    return judgement


def empty_evidence_judgement(prediction: EvidenceSubmission) -> Judgement | None:
    """Apply the rubric's no-submitted-findings case without an API call."""
    if prediction.label not in MECHANISM_POOL:
        raise ValueError("Unknown mechanism label")
    if prediction.evidence:
        return None
    return Judgement(
        status="scored",
        evidence_support_score=0.0,
        verifications=[],
        selected_route_kind="none",
        selected_route_id=None,
        conditions=[],
        overstatement="not_assessed",
        rationale=(
            "The submitted evidence list is explicitly empty. No submitted finding can "
            "satisfy a strong- or partial-support route. This is a deterministic "
            "empty-submission assessment, not an LLM judgement or a failed-call fallback. "
            "It does not establish that the predicted mechanism is absent."
        ),
    )


class ZeroShotV3Judge:
    def __init__(self, client: OpenAIJsonClient, *, semantic_scope: bool = False) -> None:
        self.client = client
        self.rubric = json.loads(RUBRIC_PATH.read_text())
        self.prompt = compose_v3_judge_prompt(load_eval_prompt(PROMPT_NAME), semantic_scope=semantic_scope)

    def payload(self, prediction: EvidenceSubmission, facts: dict) -> dict:
        return judge_payload(prediction, facts, self.rubric)

    def response_schema(self, prediction: EvidenceSubmission) -> dict:
        return judge_response_schema(prediction.label, self.rubric)

    def validate_response(
        self, raw: dict, prediction: EvidenceSubmission, facts: dict,
    ) -> Judgement:
        response = JudgeResponse.model_validate(raw)
        return validate_judgement(response.judgement.model_dump(), prediction, facts, self.rubric)
