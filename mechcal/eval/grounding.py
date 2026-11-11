"""Layer A: deterministic grounding of submitted evidence claims (no LLM).

Each claim receives exactly one outcome. Computed claims must cite a successful (or
partial, with the cited value present) artifact computed for the same molecule, and
every cited value must match the artifact within tolerance. Numbers written in the
statement must agree with the declared quantities, so editing the prose alone is caught.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from mechcal.eval.submission import (
    ArtifactStore, EvidenceClaim, Prediction, Quantity, StructuralAssertion, Submission,
    resolve_pointer,
)

GROUNDING_VERSION = "0.3.0"
REL_TOL = 0.02

FABRICATION = frozenset({"missing_artifact", "wrong_molecule", "unresolved_pointer",
                         "value_mismatch", "statement_mismatch"})
INVALID_SOURCE = frozenset({"failed_artifact"})
ELIGIBLE = frozenset({"grounded", "structural_ok", "structural_unchecked"})
# Order used to pick the single outcome of a claim when several problems exist.
SEVERITY = ("missing_artifact", "wrong_molecule", "unresolved_pointer", "failed_artifact",
            "value_mismatch", "statement_mismatch", "structural_bad", "unsourced")

# Numbers that are not part of an identifier such as S1, T2, 6-31G, def2-SVP, C21 or B3LYP.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_\-.])[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_]|-[A-Za-z0-9])")

_ENERGY_IN_EV = {"ev": 1.0, "hartree": 27.211386245988, "eh": 27.211386245988,
                 "kcal/mol": 0.0433641043, "kj/mol": 0.0103642688, "cm-1": 1.23984198e-4,
                 "mev": 1e-3}
_LENGTH_IN_A = {"a": 1.0, "å": 1.0, "angstrom": 1.0, "nm_len": 10.0, "bohr": 0.529177210903, "pm": 0.01}
_HC_EV_NM = 1239.841984


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_index: int
    kind: str
    outcome: str
    problems: list[str] = Field(default_factory=list)
    undeclared_numbers: list[str] = Field(default_factory=list)


class PredictionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    rank: int
    claims: list[ClaimCheck]
    outcome_counts: dict[str, int]
    eligible_claims: list[int]
    fabricated: bool
    invalid_source: bool


class SubmissionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grounding_version: str = GROUNDING_VERSION
    case_id: str
    predictions: list[PredictionCheck]
    computed_claims: int
    grounded_computed_claims: int
    fabricated_claims: int
    unsourced_claims: int
    fabrication_rate: float | None
    grounded_rate: float | None


def _norm_unit(unit: str | None) -> str:
    return (unit or "").strip().lower().replace("⁻¹", "-1").replace("^-1", "-1").replace(" ", "")


def convert(value: float, from_unit: str | None, to_unit: str | None) -> float:
    """Convert between energy units, wavelength (nm) and energy, or length units."""
    a, b = _norm_unit(from_unit), _norm_unit(to_unit)
    if a == b:
        return value
    if a in _ENERGY_IN_EV and b in _ENERGY_IN_EV:
        return value * _ENERGY_IN_EV[a] / _ENERGY_IN_EV[b]
    if a == "nm" and b in _ENERGY_IN_EV:
        return _HC_EV_NM / value / _ENERGY_IN_EV[b]
    if a in _ENERGY_IN_EV and b == "nm":
        return _HC_EV_NM / (value * _ENERGY_IN_EV[a])
    la, lb = ("nm_len" if a == "nm" else a), ("nm_len" if b == "nm" else b)
    if la in _LENGTH_IN_A and lb in _LENGTH_IN_A:
        return value * _LENGTH_IN_A[la] / _LENGTH_IN_A[lb]
    raise ValueError(f"Incompatible units: {from_unit!r} -> {to_unit!r}")


def _decimals(text: str) -> int:
    mantissa = text.lower().split("e")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def _tolerance(reference: float, written: str) -> float:
    return max(REL_TOL * abs(reference), 0.5 * 10 ** (-_decimals(written)), 1e-12)


def _written(value: float) -> str:
    return format(value, ".12g")


def close(claimed: float, actual: float, written: str | None = None) -> bool:
    written = written or _written(claimed)
    return math.isfinite(actual) and abs(claimed - actual) <= _tolerance(actual, written)


def statement_numbers(statement: str) -> list[str]:
    return [m.group(0).replace("−", "-") for m in _NUMBER.finditer(statement)]


@lru_cache(maxsize=4096)
def _canonical(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=512)
def _mol(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Case SMILES could not be parsed")
    return mol


def _check_quantity(q: Quantity, store: ArtifactStore, case_smiles: str,
                    written: str | None = None) -> str | None:
    if q.source is None:
        return "unsourced"
    artifact = store.artifacts.get(q.source.artifact_id)
    if artifact is None:
        return "missing_artifact"
    if _canonical(artifact.molecule_smiles) != _canonical(case_smiles):
        return "wrong_molecule"
    if artifact.status == "failed":
        return "failed_artifact"
    try:
        raw = resolve_pointer(artifact.outputs, q.source.pointer)
    except KeyError:
        return "unresolved_pointer"
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return "unresolved_pointer"
    try:
        actual = convert(float(raw), q.source.unit or q.unit, q.unit)
    except (ValueError, ZeroDivisionError):
        return "value_mismatch"
    return None if close(q.value, actual, written) else "value_mismatch"


def _statement_matches(claim: EvidenceClaim) -> tuple[list[str | None], list[str]]:
    """Match each declared quantity to a number written in the statement.

    Returns the written form per quantity (None when absent) and the unmatched numbers.
    The written form carries the precision the claim actually asserts (JSON drops "0.40").
    """
    written = statement_numbers(claim.statement)
    forms: list[str | None] = []
    used: set[int] = set()
    for q in claim.quantities:
        match = next((i for i, w in enumerate(written)
                      if i not in used and close(q.value, float(w), w)), None)
        forms.append(None if match is None else written[match])
        if match is not None:
            used.add(match)
    return forms, [w for i, w in enumerate(written) if i not in used]


def check_assertion(assertion: StructuralAssertion, smiles: str) -> bool:
    from rdkit import Chem

    mol = _mol(smiles)
    n = mol.GetNumAtoms()
    if any(not 1 <= a <= n for a in assertion.atoms):
        return False
    atoms = [mol.GetAtomWithIdx(a - 1) for a in assertion.atoms]
    t = assertion.type
    if t == "element":
        return assertion.element is not None and all(x.GetSymbol() == assertion.element for x in atoms)
    if t == "aromatic":
        return all(x.GetIsAromatic() for x in atoms)
    if t == "in_ring":
        return all(x.IsInRing() for x in atoms)
    if len(assertion.atoms) < 2:
        return False
    i, j = assertion.atoms[0] - 1, assertion.atoms[-1] - 1
    if t in {"bonded", "rotatable_bond"}:
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None or len(assertion.atoms) != 2:
            return False
        if t == "bonded":
            return True
        pattern = Chem.MolFromSmarts("[!$([NH])&!$([NH2])&!D1&!$(*#*)]-&!@[!$([NH])&!$([NH2])&!D1&!$(*#*)]")
        return any({a, b} == {i, j} for a, b in mol.GetSubstructMatches(pattern))
    if t == "conjugated_path":
        path = Chem.GetShortestPath(mol, i, j)
        return len(path) >= 2 and all(
            mol.GetBondBetweenAtoms(a, b).GetIsConjugated() for a, b in zip(path, path[1:]))
    return False


def check_claim(index: int, claim: EvidenceClaim, store: ArtifactStore, case_smiles: str) -> ClaimCheck:
    problems: list[str] = []
    undeclared: list[str] = []
    n_atoms = _mol(case_smiles).GetNumAtoms()
    if claim.kind == "computed":
        if not claim.quantities:
            problems.append("unsourced")
        forms, undeclared = _statement_matches(claim)
        problems += [p for q, w in zip(claim.quantities, forms)
                     if (p := _check_quantity(q, store, case_smiles, w))]
        problems += ["statement_mismatch" for w in forms if w is None]
        outcome = next((s for s in SEVERITY if s in problems), "grounded")
    elif claim.kind == "structural":
        if claim.quantities:
            problems.append("unsourced")
        bad_ids = any(not 1 <= a <= n_atoms for a in claim.atom_ids)
        bad_assertion = any(not check_assertion(a, case_smiles) for a in claim.assertions)
        if bad_ids or bad_assertion:
            problems.append("structural_bad")
        undeclared = statement_numbers(claim.statement) if claim.quantities else []
        if "structural_bad" in problems:
            outcome = "structural_bad"
        elif "unsourced" in problems:
            outcome = "unsourced"
        else:
            outcome = "structural_ok" if claim.assertions else "structural_unchecked"
    else:
        outcome = "unsourced" if claim.quantities else "prior"
        if claim.quantities:
            problems.append("unsourced")
    return ClaimCheck(claim_index=index, kind=claim.kind, outcome=outcome,
                      problems=sorted(set(problems)), undeclared_numbers=undeclared)


def check_prediction(prediction: Prediction, store: ArtifactStore, case_smiles: str) -> PredictionCheck:
    claims = [check_claim(i, c, store, case_smiles) for i, c in enumerate(prediction.evidence)]
    counts = Counter(c.outcome for c in claims)
    return PredictionCheck(
        label=prediction.label, rank=prediction.rank, claims=claims, outcome_counts=dict(counts),
        eligible_claims=[c.claim_index for c in claims if c.outcome in ELIGIBLE],
        fabricated=any(c.outcome in FABRICATION for c in claims),
        invalid_source=any(c.outcome in INVALID_SOURCE for c in claims),
    )


def check_submission(submission: Submission, store: ArtifactStore, case_smiles: str) -> SubmissionCheck:
    if store.case_id != submission.case_id:
        raise ValueError("Artifact store and submission belong to different cases")
    preds = [check_prediction(p, store, case_smiles) for p in submission.mechanism_predictions]
    computed = [c for p in preds for c in p.claims if c.kind == "computed"]
    fabricated = sum(c.outcome in FABRICATION for c in computed)
    grounded = sum(c.outcome == "grounded" for c in computed)
    unsourced = sum(c.outcome == "unsourced" for p in preds for c in p.claims)
    return SubmissionCheck(
        case_id=submission.case_id, predictions=preds, computed_claims=len(computed),
        grounded_computed_claims=grounded, fabricated_claims=fabricated, unsourced_claims=unsourced,
        fabrication_rate=fabricated / len(computed) if computed else None,
        grounded_rate=grounded / len(computed) if computed else None,
    )


def gate_support(score: float, check: PredictionCheck, used_claims: Iterable[int] | None = None) -> float:
    """Apply the fabrication cap to a Layer B score.

    With ``used_claims`` (the claim indices the judge's selected route relied on), only those
    claims count; without it, any fabricated or invalid-source claim in the prediction caps it.
    A route may never rest on a claim that is not Layer-A eligible.
    """
    by_index = {c.claim_index: c for c in check.claims}
    if used_claims is None:
        bad = check.fabricated or check.invalid_source
    else:
        used = list(used_claims)
        bad = any(i not in by_index or by_index[i].outcome not in ELIGIBLE for i in used)
    return 0.0 if bad else score
