"""Hypothesis board: the single source of what the agent currently believes about 11 families."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mechcal.eval.submission import (
    EvidenceClaim, Prediction, Quantity, QuantitySource, StructuralAssertion, Submission,
)
from mechcal.mechanism_pool import MECHANISM_POOL

Status = Literal["untested", "supported", "weakened", "mixed", "not_testable"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoardClaim(_M):
    claim_id: str
    label: str
    claim: EvidenceClaim
    origin: Literal["planner", "worker"] = "planner"
    step: int
    # the Planner's judgement: would this observation appear for most molecules (no), or does it set
    # this molecule apart (yes / partly)?
    discriminates: Literal["yes", "partly", "no"] | None = None
    discriminates_why: str = ""


class FamilyState(_M):
    label: str
    plausibility: float = Field(ge=0, le=1)
    status: Status = "untested"
    reason: str = ""
    next_test: str = ""
    prior: float | None = Field(default=None, ge=0, le=1)   # from structure and chemistry knowledge, before evidence
    prior_basis: str = ""


class Board(_M):
    case_id: str
    smiles: str
    families: dict[str, FamilyState]
    claims: list[BoardClaim] = Field(default_factory=list)
    limitations: dict[str, list[str]] = Field(default_factory=dict)
    issued_claims: int = 0          # monotonic: ids are never reused after a claim is dropped

    @classmethod
    def initial(cls, case_id: str, smiles: str) -> "Board":
        return cls(case_id=case_id, smiles=smiles,
                   families={f: FamilyState(label=f, plausibility=0.1) for f in MECHANISM_POOL})

    def ranking(self) -> list[str]:
        return sorted(self.families, key=lambda f: (-self.families[f].plausibility, MECHANISM_POOL.index(f)))

    def add_claim(self, label: str, claim: EvidenceClaim, *, step: int, origin: str = "planner",
                  discriminates: str | None = None, discriminates_why: str = "") -> str:
        self.issued_claims += 1
        cid = f"K{self.issued_claims}"
        self.claims.append(BoardClaim(claim_id=cid, label=label, claim=claim, origin=origin, step=step,
                                      discriminates=discriminates, discriminates_why=discriminates_why))
        return cid

    def generic_only(self, label: str) -> bool:
        """True when every computed claim of a family is marked as not discriminating (and there is one)."""
        computed = [c for c in self.claims_for(label) if c.claim.kind == "computed"]
        return bool(computed) and all(c.discriminates == "no" for c in computed)

    def drop_claims(self, claim_ids: list[str]) -> None:
        self.claims = [c for c in self.claims if c.claim_id not in set(claim_ids)]

    def claims_for(self, label: str) -> list[BoardClaim]:
        return [c for c in self.claims if c.label == label]

    def view(self, *, detailed: set[str] | None = None, top: int = 5) -> dict:
        """Compact view: all 11 families in one line each; claims only for top/flagged families."""
        ranking = self.ranking()
        show = set(ranking[:top]) | (detailed or set())
        rows = []
        for rank, label in enumerate(ranking, start=1):
            f = self.families[label]
            row = {"rank": rank, "label": label, "prior": f.prior, "plausibility": round(f.plausibility, 3),
                   "status": f.status, "n_claims": len(self.claims_for(label))}
            if label in show:
                row.update(reason=f.reason, prior_basis=f.prior_basis, next_test=f.next_test,
                           claims=[{"claim_id": c.claim_id, "kind": c.claim.kind, "statement": c.claim.statement,
                                    **({"discriminates": c.discriminates} if c.discriminates else {})}
                                   for c in self.claims_for(label)])
            rows.append(row)
        return {"families": rows}

    def to_submission(self, *, eligible: set[str] | None = None) -> Submission:
        """Project the top-3 into the grounded submission. ``eligible`` optionally filters claim ids."""
        preds = []
        for rank, label in enumerate(self.ranking()[:3], start=1):
            claims = [c.claim for c in self.claims_for(label) if eligible is None or c.claim_id in eligible]
            preds.append(Prediction(label=label, rank=rank, evidence=claims,
                                    limitations=self.limitations.get(label) or ["Isolated-molecule calculations only."]))
        return Submission(case_id=self.case_id, mechanism_predictions=preds)


class ClaimIn(BaseModel):
    """Flat claim shape the models write (quantities carry their source inline)."""

    model_config = ConfigDict(extra="ignore")
    label: str
    statement: str
    kind: Literal["structural", "computed", "prior"]
    atom_ids: list[int] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    quantities: list[dict] = Field(default_factory=list)
    discriminates: Literal["yes", "partly", "no"] | None = None
    discriminates_why: str = ""

    def to_claim(self, facts: dict) -> EvidenceClaim:
        by_id = {f["fact_id"]: f for f in facts.get("facts", [])}
        assertions = [StructuralAssertion(**by_id[f]["assertion"]) for f in self.fact_ids
                      if f in by_id and "assertion" in by_id[f]]
        atoms = list(dict.fromkeys(self.atom_ids + [a for f in self.fact_ids if f in by_id for a in by_id[f]["atom_ids"]]))
        quantities = []
        for q in self.quantities:
            source = None
            if q.get("artifact_id"):
                source = QuantitySource(artifact_id=str(q["artifact_id"]), pointer=str(q.get("pointer", "")),
                                        unit=q.get("source_unit") or None)
            quantities.append(Quantity(name=str(q.get("name") or "value"), value=float(q["value"]),
                                       unit=str(q.get("unit") or ""), source=source))
        return EvidenceClaim(statement=self.statement, kind=self.kind, atom_ids=atoms if self.kind == "structural" else [],
                             assertions=assertions if self.kind == "structural" else [], quantities=quantities)
