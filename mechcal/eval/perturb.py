"""Metric-validity perturbations for evidence scoring (release 0.3.0 evaluation).

Each operator edits one prediction of a real submission and states the expected effect
on that prediction's evidence score. Running a scorer (text-only judge, the 0.2.0 route judge, the grounded judge ...) over
the original and perturbed submissions measures whether the score reacts to falsified
evidence (detection) and stays put under meaning-preserving edits (false alarms).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

from mechcal.eval.submission import (
    ArtifactStore, EvidenceClaim, Quantity, QuantitySource, Submission, resolve_pointer,
)
from mechcal.eval.grounding import close, statement_numbers

Expected = Literal["decrease", "no_increase", "no_change"]
Scorer = Callable[[Submission, ArtifactStore, str], Sequence[float]]

_FLIPS = [
    ("bright", "dark"), ("increases", "decreases"), ("increase", "decrease"), ("higher", "lower"),
    ("high", "low"), ("strong", "weak"), ("large", "small"), ("localized", "delocalized"),
    ("allowed", "forbidden"), ("planar", "twisted"), ("restricted", "unrestricted"),
    ("supports", "argues against"), ("consistent with", "inconsistent with"),
    ("significant", "negligible"), ("separated", "overlapping"), ("below", "above"),
]


@dataclass
class Perturbation:
    operator: str
    rank: int
    expected: Expected
    submission: Submission
    store: ArtifactStore
    note: str = ""


@dataclass
class PerturbationResult:
    case_id: str
    operator: str
    rank: int
    expected: Expected
    base: float
    perturbed: float
    note: str = ""
    delta: float = field(init=False)

    def __post_init__(self):
        self.delta = self.perturbed - self.base

    @property
    def applicable(self) -> bool:
        # A score already at 0 cannot reveal a decrease; such items are reported, not counted.
        return not (self.expected == "decrease" and self.base <= 0)

    @property
    def passed(self) -> bool:
        if self.expected == "decrease":
            return self.delta < 0
        if self.expected == "no_increase":
            return self.delta <= 0
        return self.delta == 0


def _copy(submission: Submission, store: ArtifactStore):
    return submission.model_copy(deep=True), store.model_copy(deep=True)


def _prediction(submission: Submission, rank: int):
    return next(p for p in submission.mechanism_predictions if p.rank == rank)


def _format_like(value: float, written: str) -> str:
    decimals = len(written.split(".")[1]) if "." in written else 0
    return f"{value:.{decimals}f}" if decimals else str(int(round(value)))


def _replace_number(statement: str, old_value: float, new_value: float) -> str | None:
    for written in statement_numbers(statement):
        if close(old_value, float(written), written):
            return statement.replace(written, _format_like(new_value, written), 1)
    return None


def _sourced(submission: Submission, rank: int):
    pred = _prediction(submission, rank)
    for ci, claim in enumerate(pred.evidence):
        for qi, q in enumerate(claim.quantities):
            if claim.kind == "computed" and q.source is not None:
                yield ci, qi, claim, q


def tamper_value(submission, store, rank, *, factor: float = 10.0, statement_only: bool = False):
    sub, st = _copy(submission, store)
    for ci, qi, claim, q in _sourced(sub, rank):
        new = q.value * factor if q.value else 1.0
        text = _replace_number(claim.statement, q.value, new)
        if text is None:
            continue
        claim.statement = text
        if not statement_only:
            q.value = new
        mode = "statement_only" if statement_only else "consistent"
        return Perturbation("tamper_value" + ("_text" if statement_only else ""), rank, "decrease", sub, st,
                            f"claim {ci} quantity {q.name} x{factor} ({mode})")
    return None


def swap_molecule(submission, store, rank, donor: ArtifactStore):
    """Cite another molecule's artifacts from the same tool, with text matching the donor values."""
    sub, st = _copy(submission, store)
    swapped = 0
    for ci, qi, claim, q in _sourced(sub, rank):
        own = store.artifacts.get(q.source.artifact_id)
        if own is None:
            continue
        for art in donor.artifacts.values():
            if art.tool != own.tool or art.status == "failed":
                continue
            try:
                value = float(resolve_pointer(art.outputs, q.source.pointer))
            except (KeyError, TypeError, ValueError):
                continue
            text = _replace_number(claim.statement, q.value, value)
            if text is None:
                continue
            new_id = f"swap:{donor.case_id}:{art.artifact_id}"
            st.add(art.model_copy(update={"artifact_id": new_id}))
            claim.statement, q.value = text, value
            q.source = QuantitySource(artifact_id=new_id, pointer=q.source.pointer, unit=q.source.unit)
            swapped += 1
            break
    if not swapped:
        return None
    return Perturbation("swap_molecule", rank, "decrease", sub, st, f"{swapped} quantities from {donor.case_id}")


def flip_conclusion(submission, store, rank):
    sub, st = _copy(submission, store)
    flipped = 0
    for claim in _prediction(sub, rank).evidence:
        text = claim.statement
        for a, b in _FLIPS:
            pattern = re.compile(rf"\b({re.escape(a)}|{re.escape(b)})\b", re.IGNORECASE)
            new, n = pattern.subn(lambda m: _swap_case(m.group(0), a, b), text)
            if n:
                text, flipped = new, flipped + n
                break
        claim.statement = text
    if not flipped:
        return None
    return Perturbation("flip_conclusion", rank, "decrease", sub, st, f"{flipped} direction words inverted")


def _swap_case(word: str, a: str, b: str) -> str:
    target = b if word.lower() == a else a
    return target.capitalize() if word[:1].isupper() else target


def strip_artifacts(submission, store, rank):
    sub, st = _copy(submission, store)
    stripped = 0
    for _, _, _, q in list(_sourced(sub, rank)):
        q.source = None
        stripped += 1
    if not stripped:
        return None
    return Perturbation("strip_artifacts", rank, "decrease", sub, st, f"{stripped} sources removed")


def inject_fabrication(submission, store, rank, *, fake_citation: bool = True):
    sub, st = _copy(submission, store)
    source = QuantitySource(artifact_id="fabricated:A999", pointer="/oscillator_strength") if fake_citation else None
    _prediction(sub, rank).evidence.append(EvidenceClaim(
        statement="The lowest singlet excited state has an oscillator strength of 0.85, "
                  "confirming a strongly emissive state for this mechanism.",
        kind="computed",
        quantities=[Quantity(name="oscillator_strength", value=0.85, unit="", source=source)],
    ))
    name = "inject_fake_citation" if fake_citation else "inject_unsourced"
    return Perturbation(name, rank, "no_increase", sub, st, "plausible fabricated claim appended")


def _default_paraphrase(statement: str) -> str:
    body = statement[0].lower() + statement[1:] if statement[:2] != statement[:2].upper() else statement
    return "Our analysis indicates that " + body


def paraphrase(submission, store, rank, *, rewrite: Callable[[str], str] | None = None):
    """Meaning-preserving control. Pass an LLM rewrite for realistic paraphrases."""
    sub, st = _copy(submission, store)
    rewrite = rewrite or _default_paraphrase
    pred = _prediction(sub, rank)
    if not pred.evidence:
        return None
    for claim in pred.evidence:
        new = rewrite(claim.statement)
        if sorted(statement_numbers(new)) != sorted(statement_numbers(claim.statement)):
            raise ValueError("Paraphrase changed the written numbers; not a control")
        claim.statement = new
    return Perturbation("paraphrase", rank, "no_change", sub, st)


def reorder(submission, store, rank):
    sub, st = _copy(submission, store)
    pred = _prediction(sub, rank)
    if len(pred.evidence) < 2:
        return None
    pred.evidence.reverse()
    return Perturbation("reorder", rank, "no_change", sub, st)


def default_operators(donor: ArtifactStore | None = None):
    ops = [tamper_value, lambda s, st, r: tamper_value(s, st, r, statement_only=True),
           flip_conclusion, strip_artifacts, inject_fabrication,
           lambda s, st, r: inject_fabrication(s, st, r, fake_citation=False), paraphrase, reorder]
    if donor is not None:
        ops.insert(2, lambda s, st, r: swap_molecule(s, st, r, donor))
    return ops


def run_suite(
    items: Sequence[tuple[Submission, ArtifactStore, str, ArtifactStore | None]],
    scorer: Scorer, *, ranks: Sequence[int] = (1,),
) -> list[PerturbationResult]:
    """``items``: (submission, store, case SMILES, donor store from another case or None)."""
    results = []
    for submission, store, smiles, donor in items:
        base = list(scorer(submission, store, smiles))
        for rank in ranks:
            if rank > len(submission.mechanism_predictions):
                continue
            for op in default_operators(donor):
                p = op(submission, store, rank)
                if p is None:
                    continue
                after = list(scorer(p.submission, p.store, smiles))
                results.append(PerturbationResult(submission.case_id, p.operator, rank, p.expected,
                                                  base[rank - 1], after[rank - 1], p.note))
    return results


def summarize_suite(results: Sequence[PerturbationResult]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[r.operator].append(r)
    out = {}
    for op, rows in sorted(groups.items()):
        applicable = [r for r in rows if r.applicable]
        out[op] = {
            "expected": rows[0].expected, "n": len(rows), "n_applicable": len(applicable),
            "pass_rate": sum(r.passed for r in applicable) / len(applicable) if applicable else None,
            "mean_delta": sum(r.delta for r in applicable) / len(applicable) if applicable else None,
        }
    return out
