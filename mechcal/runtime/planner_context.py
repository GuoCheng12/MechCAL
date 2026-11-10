"""Working context projections. Full records stay in case-local memory."""

from copy import deepcopy

from mechcal.runtime.observation_context import project_observations


def referenced_ids(value):
    if isinstance(value, dict):
        refs = set()
        for key, item in value.items():
            if key.endswith(("_refs", "_ref")):
                refs.update(x for x in (item if isinstance(item, list) else [item]) if isinstance(x, str))
            refs.update(referenced_ids(item))
        return refs
    if isinstance(value, list):
        return set().union(*(referenced_ids(item) for item in value)) if value else set()
    return set()


def select(record, fields):
    return {key: value for key, value in record.items() if key in fields}


def question_brief(question):
    return select(question, {
        "question_id", "question", "mechanism_id", "related_labels", "status", "answer_state",
        "execution_state", "decision_role", "evidence_refs", "result_ids", "acceptable_capability_ids",
    })


def scoped_questions(payload: dict) -> list[dict]:
    """Use explicit question state and links, not inferred scientific relevance."""
    questions = payload.get("questions", [])
    if (payload["stage"] == "planner_final_revision" or payload.get("revision") == 0
            or payload.get("argument_review_pending")):
        return deepcopy(questions)
    pending = payload.get("pending_results", [])
    relevant_ids = {qid for row in pending for qid in row.get("question_ids", [])}
    if pending:
        relevant_ids.update(qid for row in (payload.get("result_review") or {}).get("interpretations", [])
                            for qid in row.get("question_ids", []))
    stop_review = payload.get("stop_review_feedback") or {}
    routes = set(stop_review.get("capability_ids", []))
    refs = referenced_ids(stop_review)
    result = []
    for question in questions:
        if (question.get("answer_state") not in {"answered", "deferred"}
                or (question.get("decision_role") == "material" and question.get("answer_state") != "answered")
                or question["question_id"] in relevant_ids
                or routes.intersection(question.get("acceptable_capability_ids", []))
                or refs.intersection(question.get("evidence_refs", []))):
            result.append(deepcopy(question))
        else:
            result.append({
                **deepcopy(question_brief(question)),
                **deepcopy(select(question, {"assessment", "missing_or_unresolved"})),
                "memory_record_id": question["question_id"],
            })
    return result


def scope_assimilation_capabilities(result: dict) -> None:
    """Registry names and status convey breadth without dispatch contracts."""
    if "capability_catalog" in result:
        catalog = result.pop("capability_catalog")
        by_status: dict[str, list[str]] = {}
        for row in catalog:
            by_status.setdefault(row["status"], []).append(row["capability_id"])
        result["capability_overview"] = {
            "total": len(catalog),
            "status_counts": {status: len(ids) for status, ids in by_status.items()},
            "capability_ids_by_status": by_status,
        }
    for key in ("capability_cards", "capability_catalog_shared", "capability_catalog_format",
                "ability_evidence_guide", "artifact_available_capability_ids", "tool_discovery",
                "tool_loading_mode"):
        result.pop(key, None)


def result_impact_scope(payload: dict) -> tuple[set[str], set[str]]:
    """Review scope widens visible context; absent review never certifies no impact."""
    pending = payload.get("pending_results", [])
    if (not payload.get("scoped_evidence_updates") or payload["stage"] != "planner_assimilate"
            or not pending):
        return set(), set()
    arguments = payload.get("previous_support_arguments", [])
    all_ids = {a["support_id"] for a in arguments}
    review = payload.get("result_review") or {}
    impact = review.get("impact_review") or {}
    batch_matched = (review.get("status") == "reviewed" and impact.get("snapshot_id")
                     and set(impact.get("result_ids", [])) == {r["result_id"] for r in pending}
                     and set(impact.get("checked_support_ids", [])) == all_ids)
    if not batch_matched:
        return set(), set()
    if impact.get("global_review"):
        return all_ids, {h["name"] for h in payload.get("portfolio", {}).get("hypotheses", [])}
    labels = set(impact.get("affected_labels", []))
    ids = set(impact.get("affected_support_ids", []))
    ids.update(a["support_id"] for a in arguments if a["target_label"] in labels)
    labels.update(a["target_label"] for a in arguments if a["support_id"] in ids)
    return ids, labels


def comparison_brief(context):
    """The active comparison, not the initial guess or full argument-edit history."""
    result = deepcopy(context)
    if not result.get("enabled"):
        return result
    initial = result.get("initial_judgment")
    if initial:
        result["initial_judgment"] = {"memory_record_id": "judgment:initial", "revision": initial.get("revision")}
    if result.get("latest_update"):
        result.pop("latest_update")
        result["read_only_audit"] = {"memory_record_id": "judgment:last_comparison"}
    return result


def decision_working_set(result, *, action, affected, affected_labels=()):
    """Keep all candidates visible; retrieve verbose, unchanged judgments by ID."""
    bases = result.get("boundary_decision_context", {}).get("current_candidate_bases", [])
    covered = {b["label"] for b in bases}
    targets = {t.get("label") for t in result.get("evidence_impact_context", {}).get("targets", [])}
    targets.update(affected_labels)
    targets.update(a.get("target_label") for a in result.get("previous_support_arguments", [])
                   if a.get("support_id") in affected)
    portfolio = result.get("portfolio", {})
    hypothesis_fields = {"name", "confidence", "differential_priority", "evidence_support",
                         "claim_status", "status", "evidence_refs"}
    if result.get("scoped_evidence_updates"):
        hypothesis_fields.add("validation_needed")
    portfolio["hypotheses"] = [
        h if h.get("name") not in covered or (not action and h.get("name") in targets) else {
            **select(h, hypothesis_fields),
            "memory_record_id": f"hypothesis:{h['name']}",
        } for h in portfolio.get("hypotheses", [])
    ]
    if not result.get("scoped_evidence_updates"):
        result["questions"] = [
            select(q, {"question_id", "question", "mechanism_id", "related_labels", "observable",
                         "expected_basis", "answer_state", "decision_role", "assessment", "evidence_refs",
                         "result_ids", "acceptable_capability_ids"})
            for q in result.get("questions", [])]
    tests = result.get("hypothesis_test_context", {})
    tests["recent_tests"] = [{
        **select(t, {"revision", "round", "result_id", "memory_record_id"}),
        "tested_claim": t.get("tested_claim"),
        "assessment": select(t.get("assessment", {}), {
            "basis_effect", "comparison_effect", "evidence_refs", "operational_refs",
        }),
    } for t in tests.get("recent_tests", [])]
    # Current corrections stay verbatim. Historical reviews are retrievable once,
    # not repeated alongside accepted questions and candidate bases.
    if action and not result.get("argument_review_pending"):
        for key, record_id in (("result_review", "review:result"), ("argument_review", "review:argument")):
            if result.get(key):
                result[key] = {"memory_record_id": record_id}
    feedback = result.get("coverage_feedback")
    if feedback:
        result["coverage_feedback"] = {
            **select(feedback, {"coverage_summary", "decision_challenges", "recommended_next_routes"}),
            "memory_record_id": "review:coverage",
        }
    result["memory_policy"]["judgments"] = (
        "All candidate bases remain visible. Brief hypotheses/questions/tests omit old prose, "
        "not scientific records. Read memory_record_id (question_id for questions) "
        "before reinterpreting omitted details."
    )


def build_context(payload):
    from mechcal.capabilities.discovery import compact_catalog_payload

    result = deepcopy(payload)
    stage = payload["stage"]
    action = stage == "planner_question_act"
    final = stage == "planner_final_revision"
    result["context_version"] = "case_working_set_v3"
    scoped = payload.get("scoped_evidence_updates", False)
    initial = scoped and stage == "planner_assimilate" and payload.get("revision") == 0
    result["questions"] = scoped_questions(payload) if scoped else [
        question_brief(q) if q.get("answer_state") in {"answered", "deferred"} else q
        for q in result.get("questions", [])
    ]
    # Conditions, failures and delivery checks prevent redundant work.
    result["execution_history"] = [
        {**select(r, {
            "result_id", "capability_id", "status", "evidence_refs", "artifact_ids",
            "question_ids", "request_signature", "calculation_identity", "round_id",
            "requested_parameters", "resolved_source_artifact_id", "failure", "delivery",
        }), "memory_record_id": r["result_id"]} for r in result.get("execution_history", [])
    ]
    result.pop("previous_worker_explanations", None)
    result["comparison_context"] = comparison_brief(result.get("comparison_context", {}))
    tests = result.get("hypothesis_test_context", {})
    if tests.get("enabled"):
        tests["recent_tests"] = [{
            **select(t, {"revision", "round", "result_id", "assessment"}),
            "tested_claim": t.get("plan", {}).get("hypothesis_test"),
            "memory_record_id": f"test:{t['revision']}:{t['result_id']}",
        } for t in tests.get("recent_tests", [])]
    candidates = result.get("candidate_context", {})
    if candidates.get("history"):
        candidates["history"] = {
            "count": len(candidates["history"]), "memory_record_id": "history:candidate_responses",
        }
    for target in result.get("evidence_impact_context", {}).get("targets", []):
        for key, path, rows in (
            ("current_hypothesis", "/portfolio/hypotheses", result.get("portfolio", {}).get("hypotheses", [])),
            ("current_candidate_basis", "/boundary_decision_context/current_candidate_bases",
             result.get("boundary_decision_context", {}).get("current_candidate_bases", [])),
        ):
            # Replace only exact duplicates, not similar scientific statements.
            value = target.get(key)
            if value is not None and value in rows:
                target.pop(key)
                target[key + "_context_path"] = f"{path}/{rows.index(value)}"

    pending = result.get("pending_results", [])
    new_refs = {ref for r in pending for ref in r.get("evidence_refs", [])}
    review = result.get("argument_review") or {}
    reviewed = set(review.get("support_ids", [])) if result.get("argument_review_pending") else set()
    if result.get("argument_review_pending") and review.get("routing", {}).get("mode") != "local_audit":
        reviewed.update(a["support_id"] for a in payload.get("previous_support_arguments", []))
    affected = set(reviewed)
    affected.update(sid for t in result.get("evidence_impact_context", {}).get("targets", [])
                    for sid in t.get("current_support_ids", []))
    impact_ids, impact_labels = result_impact_scope(payload)
    affected.update(impact_ids)
    arguments = payload.get("previous_support_arguments", [])
    incremental = payload.get("incremental_update", {}).get("enabled", False)
    result["previous_support_arguments"] = [
        deepcopy(a) if initial or final or (not action and (not incremental or a["support_id"] in affected)) else {
            **select(a, {"support_id", "target_label", "support_level", "finding", "boundary", "observation_refs"}),
            "memory_record_id": f"argument:{a['support_id']}",
        } for a in arguments
    ]
    # An old citation is a retrieval handle, not a reason to resend all observations.
    detailed_refs = set(new_refs) if not action else set()
    if not action:
        detailed_refs |= referenced_ids([a for a in arguments if a["support_id"] in reviewed or final
                                        or a["support_id"] in impact_ids])
        detailed_refs |= referenced_ids([
            h for h in payload.get("portfolio", {}).get("hypotheses", []) if h["name"] in impact_labels])
        detailed_refs |= referenced_ids([
            b for b in payload.get("boundary_decision_context", {}).get("current_candidate_bases", [])
            if b["label"] in impact_labels])
    if final:
        detailed_refs |= referenced_ids(result.get("portfolio"))
        detailed_refs |= referenced_ids(result.get("final_internal_audit"))
    result["observations"] = [e for e in result.get("observations", []) if e["evidence_id"] in detailed_refs]
    if scoped:
        # Relevance pins the source and its boundaries, not every historical array.
        result["observations"] = project_observations(result["observations"], fresh_ids=new_refs)
    result["source_checks"] = [c for c in result.get("source_checks", []) if c["evidence_id"] in detailed_refs]
    visible_refs = {e["evidence_id"] for e in result["observations"]}
    # Supply new EvidenceUnits once. Unique raw fields remain exactly retrievable.
    for row in pending:
        refs = set(row.get("evidence_refs", []))
        if refs and refs <= visible_refs:
            row.pop("structured_results", None)
            row.pop("evidence_units", None)
            row["raw_result_record_id"] = row["result_id"]
    if not result.get("argument_review_pending") and not final and review:
        result["argument_review"] = {"memory_record_id": "review:argument", "pending": False}
    if not pending and result.get("result_review") and not final:
        result["result_review"] = {"memory_record_id": "review:result", "pending": False}
    if action:
        for key in ("pending_results", "disposition_requirements", "evidence_impact_context"):
            result.pop(key, None)
        result["unassimilated_result_ids"] = [r["result_id"] for r in pending]
    result["memory_policy"] = {
        "scope": "This run/case only. Full records are retrievable by ID; omitted is not absent.",
        "source": "Observations are facts within their conditions. Judgments, reviews and failures are not evidence.",
        "working_set": "Current judgment and all evidence index entries remain visible. Old details are read on demand.",
        "new_results": "observations supplies new EvidenceUnits; raw_result_record_id opens the complete execution.",
        "reuse": "Keep unchanged judgments. Read original conditions and limits before revising or transferring a finding.",
    }
    if not initial and not final and (incremental or action):
        decision_working_set(result, action=action, affected=affected, affected_labels=impact_labels)
    if scoped and stage == "planner_assimilate" and "capability_catalog" in result:
        scope_assimilation_capabilities(result)
    elif "capability_catalog" in result:
        result.update(compact_catalog_payload(result["capability_catalog"]))
    if scoped:
        result["context_version"] = "case_scoped_working_set_v1"
        result["memory_policy"]["stage_scope"] = (
            "Assimilation capability_overview lists registry names/status, not parameter contracts or evidence. "
            "Action retains the complete catalog. All questions and gaps remain visible for stop accounting; "
            "read brief question memory_record_id before changing omitted details. "
            "Batch-matched review impact pins affected old arguments and source summaries, not bulk metrics. "
            "metrics_memory marks exact fields held in case memory. Missing review is unknown, "
            "not an unaffected judgment or reuse authorization; other old sources remain readable by ID. "
            "Scope alone does not establish tested impact or scientific reuse."
        )
    return result
