"""Freeze a proposed judgment before asking its author for rank-bound reasons."""

import hashlib
import json
from copy import deepcopy

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.boundary_decisions import required_pairs, validate_ranking
from mechcal.runtime.evidence_comparison import argument_content
from mechcal.runtime.hypothesis_tests import validate_leading
from mechcal.runtime.result_contract import operational_evidence_refs
from mechcal.runtime.retry import RequestValidationError
from mechcal.schemas.boundary_decisions import BoundaryComparison
from mechcal.schemas.hypothesis_tests import LeadingComparison
from mechcal.schemas.ranking_explanations import RankingExplanations


def _memory_brief(record: dict, record_id: str, fields: set[str], path: str = "") -> dict:
    brief = {key: deepcopy(value) for key, value in record.items() if key in fields}
    brief["memory_record_id"] = record_id
    if path:
        brief["memory_path"] = path
    return brief


def comparison_enabled(case):
    return case.runtime.get("comparative_evidence") and (
        case.runtime.get("hypothesis_tests") or case.runtime.get("evidence_decision_contracts"))


def retain_comparisons(case, update):
    """Reuse only while the scientific inputs and required comparison roles agree."""
    ranking = update.ranking_update
    if case.runtime.get("scoped_evidence_updates"):
        if not ranking:
            return False
        draft = RankingExplanationDraft(case, update)
        if draft.leading_required or draft.pairs:
            return False
        candidate = draft.apply(RankingExplanations(
            action="explain_ranking", draft_id=draft.draft_id, leading=None, boundaries=[]))
        try:
            validate_leading(case, candidate)
            validate_ranking(case, candidate)
        except RequestValidationError:
            return False
        ranking.leading_comparison = candidate.ranking_update.leading_comparison
        ranking.boundary_comparisons = candidate.ranking_update.boundary_comparisons
        update.portfolio._scoped_comparison_receipt = deepcopy(candidate.portfolio._scoped_comparison_receipt)
        return True
    if not ranking or ranking.basis != "unchanged" or update.dispositions:
        return False
    previous = case.runtime.get("question_loop", {}).get("assimilation", {}).get("ranking_update")
    if not previous or case.portfolio != update.portfolio:
        return False
    if previous.get("candidate_bases", []) != [b.model_dump(mode="json") for b in ranking.candidate_bases]:
        return False
    before = {a.support_id: argument_content(a) for a in case.mechanism_support_arguments}
    after = {a.support_id: argument_content(a) for a in update.mechanism_support_arguments}
    if before != after:
        return False
    candidate = update.model_copy(deep=True)
    candidate.ranking_update.leading_comparison = (
        LeadingComparison.model_validate(previous["leading_comparison"])
        if previous.get("leading_comparison") else None)
    candidate.ranking_update.boundary_comparisons = [
        BoundaryComparison.model_validate(c) for c in previous.get("boundary_comparisons", [])]
    try:
        validate_leading(case, candidate)
        validate_ranking(case, candidate)
    except RequestValidationError:
        return False  # Historical explanations no longer cover the required roles.
    ranking.leading_comparison = candidate.ranking_update.leading_comparison
    ranking.boundary_comparisons = candidate.ranking_update.boundary_comparisons
    return True


class RankingExplanationDraft:
    def __init__(self, case, update):
        self.case = case
        self.update = update.model_copy(deep=True)
        self.leading_required = bool(case.runtime.get("hypothesis_tests"))
        self.required_pairs = dict(
            (f"cutoff:{i + 1}", pair) for i, pair in enumerate(
                required_pairs(case.portfolio, self.update.portfolio)
                if case.runtime.get("evidence_decision_contracts") else []))
        self.ranked = self.update.portfolio.sorted_hypotheses()
        self._comparison_context = None
        self.retained_comparisons = {}
        self.pairs = self.required_pairs.copy()
        if case.runtime.get("scoped_evidence_updates"):
            from mechcal.runtime.scoped_updates import comparison_records, comparison_reusable

            records = comparison_records(case)
            leading = records.get("leading")
            if (self.leading_required and leading
                    and leading["leader"] == self.ranked[0].name == self.update.portfolio.current
                    and comparison_reusable(case, self.update, "leading")):
                self.retained_comparisons["leading"] = deepcopy(leading)
                self.leading_required = False
            for target_id, (included, excluded) in self.required_pairs.items():
                key = f"boundary:{included}|{excluded}"
                record = records.get(key)
                if (record and (record["included_label"], record["excluded_label"]) == (included, excluded)
                        and comparison_reusable(case, self.update, key)):
                    self.retained_comparisons[key] = deepcopy(record)
                    del self.pairs[target_id]
        self.scientific_refs = ({e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"})
        self.scientific_refs -= operational_evidence_refs(case)
        # Include scientific content, not only scores: same-order reinterpretations
        # and another case/revision must never share an explanation receipt.
        draft_state = {
            "case_id": case.case_id, "update": self.update.model_dump(mode="json"),
            "leading_required": self.leading_required, "pairs": self.pairs,
            "scientific_refs": sorted(self.scientific_refs),
        }
        if case.runtime.get("scoped_evidence_updates"):
            from mechcal.runtime.scoped_updates import _comparison_context

            self._comparison_context = _comparison_context(case)
            draft_state.update(comparison_context=self._comparison_context, required_pairs=self.required_pairs,
                               retained_comparisons=self.retained_comparisons)
        self.draft_id = hashlib.sha256(json.dumps(draft_state, sort_keys=True).encode()).hexdigest()

    def payload(self):
        schema = RankingExplanations.model_json_schema()
        schema["properties"]["draft_id"] = {"const": self.draft_id}
        schema["properties"]["leading"] = (
            {"$ref": "#/$defs/LeaderExplanation"} if self.leading_required else {"type": "null"})
        schema["$defs"]["LeaderExplanation"]["properties"]["challenger"]["enum"] = [
            label for label in MECHANISM_POOL if label != self.ranked[0].name]
        schema["properties"]["boundaries"].update(minItems=len(self.pairs), maxItems=len(self.pairs))
        if self.pairs:
            schema["$defs"]["BoundaryExplanation"]["properties"]["target_id"]["enum"] = list(self.pairs)
        for name in ("LeaderExplanation", "BoundaryExplanation"):
            schema["$defs"][name]["properties"]["evidence_refs"]["items"]["enum"] = sorted(self.scientific_refs)
        previous = self.case.runtime.get("question_loop", {}).get("last_comparison") or {}
        previous_ranking = previous.get("ranking_update") or {}
        payload = {
            "stage": "planner_explain_ranking", "case_id": self.case.case_id,
            "revision": self.update.revision - 1, "draft_id": self.draft_id,
            "smiles": self.case.input.smiles,
            "proposed_ranking": [dict(rank=i + 1, **h.model_dump(mode="json"))
                                 for i, h in enumerate(self.ranked)],
            "leading_target": {"leader": self.ranked[0].name} if self.leading_required else None,
            "boundary_targets": [{"target_id": key, "included_label": a, "excluded_label": b}
                                 for key, (a, b) in self.pairs.items()],
            "rationale": self.update.ranking_update.model_dump(
                mode="json", exclude={"leading_comparison", "boundary_comparisons"}),
            "support_arguments": [a.model_dump(mode="json") for a in self.update.mechanism_support_arguments],
            "new_result_assessments": [d.model_dump(mode="json") for d in self.update.dispositions],
            "previous_comparison": {
                "revision": previous.get("revision"),
                "leading_comparison": previous_ranking.get("leading_comparison"),
                "boundary_comparisons": previous_ranking.get("boundary_comparisons", []),
            },
            "observations": [{"evidence_id": e.evidence_id, "claim": e.claim, "summary": e.summary,
                              "limits": e.limits, "basis": e.basis}
                             for e in self.case.evidence_ledger.items if e.evidence_id in self.scientific_refs],
            "scientific_evidence": False, "output_contract": schema,
        }
        return self._memory_payload(payload, previous) if self.case.runtime.get("planner_case_memory") else payload

    def _memory_payload(self, payload: dict, previous: dict) -> dict:
        """Project only context; proposed judgments and explanation contracts stay frozen."""
        payload = deepcopy(payload)
        targets = {label for pair in self.pairs.values() for label in pair}
        old_leading = (previous.get("ranking_update") or {}).get("leading_comparison")
        if self.leading_required:
            targets.update(h.name for h in self.ranked[:2])
            if old_leading:
                targets.add(old_leading["challenger"])
            if self.update.ranking_update.leading_comparison:
                targets.add(self.update.ranking_update.leading_comparison.challenger)

        # Only exactly equivalent public records may be deferred to memory.
        old_hypotheses = {h.name: h.model_dump(mode="json") for h in self.case.portfolio.hypotheses}
        for i, hypothesis in enumerate(payload["proposed_ranking"]):
            if (hypothesis["name"] not in targets and
                    {k: v for k, v in hypothesis.items() if k != "rank"} == old_hypotheses.get(hypothesis["name"])):
                payload["proposed_ranking"][i] = _memory_brief(
                    hypothesis, "hypothesis:" + hypothesis["name"], {
                        "rank", "name", "confidence", "differential_priority", "evidence_support",
                        "claim_status", "status", "evidence_refs",
                    })
        bases = payload["rationale"].pop("candidate_bases")
        old_bases = (previous.get("ranking_update") or {}).get("candidate_bases", [])
        payload["candidate_bases"] = [
            _memory_brief(basis, "judgment:last_comparison", {"label", "positive_refs", "counter_refs"},
                          f"/ranking_update/candidate_bases/{old_bases.index(basis)}")
            if basis["label"] not in targets and basis in old_bases else basis
            for basis in bases
        ]
        old_arguments = {a.support_id: a.model_dump(mode="json") for a in self.case.mechanism_support_arguments}
        payload["support_arguments"] = [
            _memory_brief(argument, "argument:" + argument["support_id"], {
                "support_id", "target_label", "support_level", "observation_refs",
            }) if argument["target_label"] not in targets and argument == old_arguments.get(argument["support_id"])
            else argument for argument in payload["support_arguments"]
        ]

        # Deduplicate exact prose only, never similar scientific interpretations.
        prose_paths = {}
        for i, basis in enumerate(payload["candidate_bases"]):
            for field in ("prior_rationale", "supported_observation", "inference_gap"):
                if basis.get(field):
                    prose_paths.setdefault(basis[field], f"/candidate_bases/{i}/{field}")
        for i, hypothesis in enumerate(payload["proposed_ranking"]):
            rationale = hypothesis.get("rationale")
            if rationale in prose_paths:
                hypothesis["rationale_context_path"] = prose_paths[rationale]
                del hypothesis["rationale"]
            elif rationale:
                prose_paths[rationale] = f"/proposed_ranking/{i}/rationale"
        explanation = payload["rationale"]["explanation"]
        if explanation in prose_paths:
            payload["rationale"]["explanation_context_path"] = prose_paths[explanation]
            del payload["rationale"]["explanation"]

        if previous:
            comparison = payload["previous_comparison"]
            comparison["memory_record_id"] = "judgment:last_comparison"
            comparison["leading_comparison"] = (
                _memory_brief(old_leading, "judgment:last_comparison", {
                    "leader", "challenger", "preference_status", "evidence_refs",
                }, "/ranking_update/leading_comparison") if self.leading_required and old_leading else None)
            pairs = {frozenset(pair) for pair in self.pairs.values()}
            comparison["boundary_comparisons"] = [
                _memory_brief(c, "judgment:last_comparison", {
                    "included_label", "excluded_label", "outcome", "basis", "evidence_refs",
                }, f"/ranking_update/boundary_comparisons/{i}")
                for i, c in enumerate(comparison["boundary_comparisons"])
                if frozenset((c["included_label"], c["excluded_label"])) in pairs]

        contexts = {e.evidence_id: e.context for e in self.case.evidence_ledger.items}
        for observation in payload["observations"]:
            observation.update(context=contexts[observation["evidence_id"]],
                               memory_record_id=observation["evidence_id"])
        payload["mechanism_pool"] = list(MECHANISM_POOL)
        payload["memory_policy"] = {
            "candidates": "Every pool label remains eligible as challenger, including unranked labels. "
                          "Compact is not excluded; proposed_ranking is the complete proposed order.",
            "records": "memory_record_id opens the complete case record with read_case_records; "
                       "memory_path selects the exact nested record. Follow next_start for fragments. "
                       "Deferred current judgments exactly match memory; new or changed judgments stay inline.",
            "duplicates": "Fields ending in _context_path are exact JSON-pointer aliases within this payload.",
            "sources": "Observations retain their claims, summaries, limits, basis and context. "
                       "Read memory_record_id for the complete EvidenceUnit, including exact numbers and conditions. "
                       "This frozen explanation does not repeat assimilation executions. Judgments are not evidence.",
        }
        return payload

    def apply(self, response: RankingExplanations):
        issues = []

        def check(ok, path, message):
            if not ok:
                issues.append({"path": path, "code": "frozen_ranking_explanation", "message": message})

        check(response.draft_id == self.draft_id, ["draft_id"], "Use the exact supplied draft_id.")
        check((response.leading is not None) == self.leading_required, ["leading"],
              "Explain the supplied leader when leading_target is present; otherwise return null.")
        ids = [b.target_id for b in response.boundaries]
        check(len(ids) == len(set(ids)) and set(ids) == self.pairs.keys(), ["boundaries"],
              f"Explain each fixed boundary target exactly once: {list(self.pairs)}.")
        if response.leading:
            check(response.leading.challenger in set(MECHANISM_POOL) - {self.ranked[0].name}, ["leading", "challenger"],
                  "Choose a different mechanism from the shared pool.")
        for index, explanation in enumerate([response.leading, *response.boundaries]):
            if explanation is None:
                continue
            check(set(explanation.evidence_refs) <= self.scientific_refs, ["evidence_refs"],
                  f"Use only current-case scientific references: {sorted(self.scientific_refs)}.")
            needs_ref = (explanation.preference_status == "evidence_favors_leader" if index == 0
                         else explanation.basis in {"observations", "mixed"})
            check(not needs_ref or bool(explanation.evidence_refs), ["evidence_refs"],
                  "An observation-based explanation requires a scientific reference; uncertainty is allowed.")
        if issues:
            raise RequestValidationError(issues)
        result = self.update.model_copy(deep=True)
        retained_leading = self.retained_comparisons.get("leading")
        if response.leading:
            result.ranking_update.leading_comparison = LeadingComparison(
                leader=self.ranked[0].name, **response.leading.model_dump())
        else:
            result.ranking_update.leading_comparison = (
                LeadingComparison.model_validate(deepcopy(retained_leading)) if retained_leading else None)
        by_id = {b.target_id: b for b in response.boundaries}
        result.ranking_update.boundary_comparisons = [
            BoundaryComparison(included_label=a, excluded_label=b,
                               **by_id[key].model_dump(exclude={"target_id"}))
            if key in self.pairs else BoundaryComparison.model_validate(
                deepcopy(self.retained_comparisons[f"boundary:{a}|{b}"]))
            for key, (a, b) in self.required_pairs.items()]
        if self._comparison_context is not None:
            from mechcal.runtime.scoped_updates import _comparison_entry

            ranking = result.ranking_update.model_dump(mode="json")
            records = {f"boundary:{row['included_label']}|{row['excluded_label']}": row
                       for row in ranking["boundary_comparisons"]}
            if ranking["leading_comparison"]:
                records["leading"] = ranking["leading_comparison"]
            # Runtime-only proof survives apply_assimilation's portfolio transfer,
            # but is never part of model-authored or serialized scientific data.
            result.portfolio._scoped_comparison_receipt = {
                "revision": result.revision,
                "comparisons": {key: _comparison_entry(
                    self._comparison_context, result.portfolio, result.mechanism_support_arguments, ranking, row)
                    for key, row in records.items()},
            }
        return result
