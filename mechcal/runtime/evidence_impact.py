"""Deliver Worker suggestions with the old judgment; never choose scientific changes."""

from mechcal.runtime.retry import RequestValidationError


def impact_targets(case):
    state = case.runtime.get("question_loop", {})
    hypotheses = {h.name: h for h in case.portfolio.hypotheses}
    ranking = state.get("assimilation", {}).get("ranking_update") or {}
    bases = {item["label"]: item for item in ranking.get("candidate_bases", [])}
    targets = []
    for result in state.get("pending_results", []):
        suggestions = {}
        for finding in (result.get("worker_explanation") or {}).get("findings", []):
            for item in finding.get("mechanism_implications", []):
                suggestions.setdefault(item["label"], []).append({
                    "finding_id": finding["finding_id"], "reason": item["reason"],
                    "evidence_refs": list(finding["evidence_refs"]),
                })
        for label, findings in suggestions.items():
            hypothesis = hypotheses.get(label)
            targets.append({
                "result_id": result["result_id"], "label": label,
                "worker_suggestions": findings,
                "current_hypothesis": hypothesis.model_dump(mode="json") if hypothesis else None,
                "current_candidate_basis": bases.get(label),
                "current_support_ids": [a.support_id for a in case.mechanism_support_arguments
                                        if a.target_label == label],
            })
    return targets


def impact_context(case):
    return {
        "targets": impact_targets(case),
        "notice": "These are Worker interpretations of existing observations, not new evidence. "
        "For each (result_id, label) respond in that disposition's mechanism_effects.judgment_response. "
        "Read the finding and its fidelity review; you may disagree. Update stale factual claims, "
        "candidate bases and arguments explicitly when justified. No score or rank change is required.",
    }


def validate_impact_responses(case, update):
    """Check response coverage only; agreement and scientific change are not enforced."""
    targets = impact_targets(case)
    if not targets:
        return
    dispositions = {d.result_id: (i, d) for i, d in enumerate(update.dispositions)}
    issues = []
    for target in targets:
        pair = dispositions.get(target["result_id"])
        if pair is None:
            path = ["dispositions"]
            message = "Provide the pending result disposition before responding to its implications."
        else:
            index, disposition = pair
            matches = [(i, e) for i, e in enumerate(disposition.mechanism_effects)
                       if e.label == target["label"]]
            path = ["dispositions", index, "mechanism_effects"]
            if len(matches) != 1:
                message = "Address this Worker-flagged label once in mechanism_effects; disagreement is allowed."
            elif not (matches[0][1].judgment_response or "").strip():
                path += [matches[0][0], "judgment_response"]
                message = "Explain the effect on the current judgment, including justified retention or disagreement."
            else:
                continue
        issues.append({
            "path": path, "repair_paths": [path], "code": "evidence_impact_response",
            "result_id": target["result_id"], "label": target["label"], "message": message,
        })
    if issues:
        raise RequestValidationError(issues)
