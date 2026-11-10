"""Versioned dependency checks for reusing judgments after reviewed new evidence."""

import hashlib
import json
from copy import deepcopy

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.questions import state
from mechcal.runtime.retry import ModelContractError


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def observation_versions(case):
    sources = {}
    artifact_ids = {}
    for result in [*state(case).get("results", []), *state(case).get("pending_results", [])]:
        # The review is attached after the receipt is issued; it is not a source fact.
        facts = {key: value for key, value in result.items() if key != "explanation_review"}
        for ref in result.get("evidence_refs", []):
            sources.setdefault(ref, {}).setdefault(result["result_id"], set()).add(digest(facts))
            artifact_ids.setdefault(ref, set()).update(result.get("input_refs", []))
            if result.get("resolved_source_artifact_id"):
                artifact_ids[ref].add(result["resolved_source_artifact_id"])
    artifacts = {a.artifact_id: a.model_dump(mode="json") for a in case.artifact_manifest.artifacts}
    return {e.evidence_id: digest({
        "observation": e.model_dump(mode="json"),
        "results": {key: sorted(values) for key, values in sources.get(e.evidence_id, {}).items()},
        "artifacts": {ref: artifacts.get(ref) for ref in
                      set(e.artifact_refs) | artifact_ids.get(e.evidence_id, set())},
    }) for e in case.evidence_ledger.items}


def comparison_records(case):
    ranking = (state(case).get("assimilation") or {}).get("ranking_update") or {}
    records = {}
    if ranking.get("leading_comparison"):
        records["leading"] = ranking["leading_comparison"]
    for row in ranking.get("boundary_comparisons", []):
        records[f"boundary:{row['included_label']}|{row['excluded_label']}"] = row
    return records


def impact_context(case):
    pending = state(case).get("pending_results", [])
    arguments = [a.model_dump(mode="json") for a in case.mechanism_support_arguments]
    comparisons = comparison_records(case)
    snapshot = {"case_id": case.case_id, "smiles": case.input.smiles,
                "results": pending, "arguments": arguments, "comparisons": comparisons,
                "observations": observation_versions(case)}
    return {
        "snapshot_id": digest(snapshot), "result_ids": [r["result_id"] for r in pending],
        "mechanism_pool": list(MECHANISM_POOL),
        "claims": [{key: row[key] for key in
                    ("support_id", "target_label", "finding", "warrant", "boundary", "observation_refs")}
                   for row in arguments],
        "comparisons": comparisons,
        "notice": "Check new observations against every supplied claim, including uncited contradictions "
                  "and alternatives outside current leaders. This only identifies possible impact, "
                  "not acceptance, scientific scores, or a compulsory change of ranking.",
    }


def validate_impact_review(review, case):
    if not case.runtime.get("scoped_evidence_updates"):
        return
    if review.status != "reviewed":
        if review.impact_review is not None:
            raise ModelContractError("Unavailable result review cannot authorize scoped reuse.")
        return
    checked = review.impact_review
    if checked is None:
        raise ModelContractError("Provide impact_review; missing is not no impact.")
    context = impact_context(case)
    support_ids = {a.support_id for a in case.mechanism_support_arguments}
    expected = {"result_ids": set(context["result_ids"]), "checked_support_ids": support_ids}
    permitted = {"affected_support_ids": support_ids, "affected_labels": set(MECHANISM_POOL),
                 "affected_comparison_ids": set(context["comparisons"])}
    if checked.snapshot_id != context["snapshot_id"]:
        raise ModelContractError("Impact review snapshot differs from the supplied batch/judgment.")
    for key, values in {**expected, **permitted}.items():
        actual = getattr(checked, key)
        if len(actual) != len(set(actual)) or not set(actual) <= values or (key in expected and set(actual) != values):
            raise ModelContractError(f"Impact review {key} must use supplied IDs with complete coverage where required.")


def impact_receipt(case, review):
    """Runtime supplies provenance; the reviewer cannot choose observation hashes."""
    from mechcal.runtime.evidence_comparison import argument_content

    if review.status != "reviewed" or review.impact_review is None:
        return None
    context = impact_context(case)
    return {
        "version": 2, "case_id": case.case_id, "smiles": case.input.smiles,
        "impact_prompt": digest(load_prompt("result_impact.md")),
        "review": review.impact_review.model_dump(mode="json"),
        "observations": observation_versions(case),
        "batch_evidence_ids": sorted({ref for r in state(case).get("pending_results", [])
                                      for ref in r.get("evidence_refs", [])}),
        "arguments": {a.support_id: digest(argument_content(a)) for a in case.mechanism_support_arguments},
        "comparisons": {key: digest(row) for key, row in context["comparisons"].items()},
    }


def current_receipt(case):
    receipt = state(case).get("evidence_impact_receipt")
    if not receipt or receipt.get("version") != 2 or receipt.get("case_id") != case.case_id:
        return None
    if receipt.get("smiles") != case.input.smiles or receipt.get("observations") != observation_versions(case):
        return None
    if receipt.get("impact_prompt") != digest(load_prompt("result_impact.md")):
        return None
    return receipt


def scoped_audit_basis(case, argument):
    from mechcal.runtime.evidence_comparison import argument_content
    refs = argument.observation_refs
    versions = observation_versions(case)
    return digest({
        "version": 3, "case_id": case.case_id, "smiles": case.input.smiles,
        "argument": argument_content(argument),
        "sources": {ref: versions.get(ref, "input:smiles" if ref == "input:smiles" else "missing") for ref in refs},
        "audit_prompt": load_prompt("photophysics_arbiter.md"),
        "impact_prompt": load_prompt("result_impact.md"),
    })


def new_batch_allows_reuse(case, cached, *, support_id=None, comparison_id=None, labels=()):
    current = observation_versions(case)
    old = cached.get("observations")
    if old is None:
        return False
    changed = {key for key in set(old) | set(current) if old.get(key) != current.get(key)}
    if not changed:
        return True
    receipt = current_receipt(case)
    if not receipt or not changed <= set(receipt["batch_evidence_ids"]):
        return False
    review = receipt["review"]
    if review["global_review"] or set(labels) & set(review["affected_labels"]):
        return False
    if support_id:
        from mechcal.runtime.evidence_comparison import argument_content

        argument = next((a for a in case.mechanism_support_arguments if a.support_id == support_id), None)
        if (support_id not in review["checked_support_ids"] or support_id in review["affected_support_ids"]
                or argument is None
                or receipt.get("arguments", {}).get(support_id) != digest(argument_content(argument))):
            return False
    if comparison_id:
        if (comparison_id in review["affected_comparison_ids"]
                or receipt.get("comparisons", {}).get(comparison_id)
                != digest(comparison_records(case).get(comparison_id))):
            return False
    # Planner and Worker can expand the scope even if the reviewer missed it.
    batch_ids = set(review["result_ids"])
    for row in state(case).get("assimilation", {}).get("dispositions", []):
        if row["result_id"] in batch_ids:
            if set(labels) & {e["label"] for e in row.get("mechanism_effects", [])
                              if e.get("effect") != "non_discriminating"}:
                return False
    for result in state(case).get("results", []):
        if result["result_id"] in batch_ids:
            implicated = {i["label"] for f in (result.get("worker_explanation") or {}).get("findings", [])
                          for i in f.get("mechanism_implications", [])}
            if set(labels) & implicated:
                return False
    return True


def _comparison_context(case):
    from mechcal.schemas.boundary_decisions import BoundaryComparison
    from mechcal.schemas.hypothesis_tests import LeadingComparison
    from mechcal.schemas.ranking_explanations import RankingExplanations

    return {
        "version": 3, "case_id": case.case_id, "smiles": case.input.smiles,
        "contract": digest({
            "pool": list(MECHANISM_POOL),
            "explanation_prompt": load_prompt("planner_explain_ranking.md"),
            "impact_prompt": load_prompt("result_impact.md"),
            "schemas": [model.model_json_schema() for model in
                        (RankingExplanations, LeadingComparison, BoundaryComparison)],
        }),
        "observations": observation_versions(case),
    }


def _comparison_labels(row):
    return ([row["leader"], row["challenger"]] if "leader" in row
            else [row["included_label"], row["excluded_label"]])


def _comparison_entry(context, portfolio, arguments, ranking, row):
    from mechcal.runtime.evidence_comparison import argument_content

    labels = set(_comparison_labels(row))
    hypotheses = {h.name: h.model_dump(mode="json") for h in portfolio.hypotheses if h.name in labels}
    bases = {b["label"]: b for b in ranking.get("candidate_bases", []) if b["label"] in labels}
    supports = {a.support_id: argument_content(a) for a in arguments if a.target_label in labels}
    refs = set(row.get("evidence_refs", []))
    refs.update(ref for h in hypotheses.values() for ref in h["evidence_refs"])
    refs.update(ref for b in bases.values() for field in ("positive_refs", "counter_refs") for ref in b[field])
    refs.update(ref for a in supports.values() for ref in a["observation_refs"])
    ranked = [h.name for h in portfolio.sorted_hypotheses()]
    roles = {label: ("leader", "runner_up", "cutoff", "nearest_excluded")[i] if i < 4 else "excluded"
             for i, label in enumerate(ranked)}
    return {
        **deepcopy(context), "comparison": digest(row),
        "inputs": digest({
            "membership": sorted(ranked),
            "pair_roles": {label: roles.get(label) for label in labels},
            "leading_roles": [portfolio.current, portfolio.runner_up, *ranked[:2]] if "leader" in row else None,
            "hypotheses": hypotheses, "bases": bases, "arguments": supports,
        }),
        "sources": {ref: context["observations"].get(ref, context["smiles"] if ref == "input:smiles" else None)
                    for ref in refs},
    }


def _same_comparison_basis(cached, current):
    return (not cached.get("invalidated", False)
            and all(cached.get(key) == current[key] for key in current if key != "observations"))


def comparison_reusable(case, update, key):
    """No forced rank or mechanism choice; only exact accepted comparisons qualify."""
    prior = comparison_records(case).get(key)
    accepted = state(case).get("comparison_dependency_cache", {}).get(key)
    if prior is None or not accepted or update.ranking_update is None:
        return False
    labels = _comparison_labels(prior)
    context = _comparison_context(case)
    previous = state(case).get("assimilation", {}).get("ranking_update") or {}
    before = _comparison_entry(context, case.portfolio, case.mechanism_support_arguments, previous, prior)
    after = _comparison_entry(context, update.portfolio, update.mechanism_support_arguments,
                              update.ranking_update.model_dump(mode="json"), prior)
    if not _same_comparison_basis(accepted, before) or not _same_comparison_basis(before, after):
        return False
    if any(set(labels) & {e.label for e in d.mechanism_effects if e.effect != "non_discriminating"}
           for d in update.dispositions):
        return False
    return new_batch_allows_reuse(case, accepted, comparison_id=key, labels=labels)


def remember_comparisons(case):
    """Advance dependencies only for a validated explanation or authorized reuse."""
    previous = state(case).get("comparison_dependency_cache", {})
    result = deepcopy(previous)
    context = _comparison_context(case)
    ranking = (state(case).get("assimilation") or {}).get("ranking_update") or {}
    revision = state(case).get("revision", 0)
    receipt = getattr(case.portfolio, "_scoped_comparison_receipt", {})
    explained = receipt.get("comparisons", {}) if receipt.get("revision") == revision else {}
    effects = {effect["label"] for disposition in state(case).get("assimilation", {}).get("dispositions", [])
               for effect in disposition.get("mechanism_effects", []) if effect.get("effect") != "non_discriminating"}
    for key, row in comparison_records(case).items():
        current = _comparison_entry(context, case.portfolio, case.mechanism_support_arguments, ranking, row)
        cached = previous.get(key)
        authorized = (explained.get(key) == current
                      and receipt["revision"] > (cached or {}).get("receipt_revision", -1))
        unreviewed_effects = (effects.intersection(_comparison_labels(row))
                             and (cached or {}).get("receipt_revision", -1) < revision)
        reusable = (cached is not None and not unreviewed_effects
                    and _same_comparison_basis(cached, current)
                    and new_batch_allows_reuse(case, cached, comparison_id=key, labels=_comparison_labels(row)))
        if authorized or reusable or not previous:
            result[key] = {**current, "receipt_revision": (
                revision if authorized or cached is None else cached.get("receipt_revision", -1))}
        elif cached is not None:
            result[key] = {**deepcopy(cached), "invalidated": True}
    return result
