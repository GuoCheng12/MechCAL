"""Stage-specific public input views. Omitted fields stay in a versioned context record."""

from copy import deepcopy

from mechcal.runtime.observation_context import project_observations
from mechcal.runtime.memory_views import lookup_memory
from mechcal.runtime.planner_context import select
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas.memory import ContextReadPlan


LOOKUP_STAGES = {"planner_assimilate", "planner_question_act", "planner_final_revision"}
ITERATIVE_STAGES = {"planner_assimilate", "planner_question_act"}

# Transport fields, not a scientific relevance filter. Everything else is readable
# from the immutable context snapshot. Repair/final stages keep their own contracts.
DECISION_FIELDS = {
    "stage", "case_id", "revision", "round_id", "smiles", "task", "mechanism_pool",
    "scoped_evidence_updates", "decision_value_checks", "decision_economy", "comparative_evidence",
    "public_input_source", "run_limits", "portfolio", "questions", "previous_assessment",
    "previous_support_arguments", "support_arguments", "candidate_context",
    "hypothesis_test_context", "boundary_decision_context", "pending_results",
    "evidence_impact_context", "disposition_requirements", "operational_only_evidence_refs",
    "result_review", "argument_review", "argument_review_pending", "coverage_feedback",
    "stop_review_feedback", "evidence_index", "observations", "source_checks",
    "execution_history", "artifacts", "output_contract", "incremental_update",
    "assimilation_context", "unassimilated_result_ids", "memory_access", "retrieved_memory",
    "capability_cards", "tool_discovery", "schema_feedback",
    "predecision_review", "predecision_comparison",
}


def current_state(payload):
    """Reuse the accepted Planner assessment verbatim; never generate a summary."""
    assessment = payload.get("previous_assessment")
    result = {"base_revision": payload.get("revision"), "assessment": deepcopy(assessment),
              "kind": "accepted_judgment_not_evidence" if assessment else "no_saved_assessment"}
    if assessment is not None:
        result["read_path"] = "/previous_assessment"
    return result


def brief(row, fields, path):
    result = deepcopy(select(row, fields))
    omitted = sorted(set(row) - set(result) - {"schema_version", "record_type"})
    if omitted:
        result["read_path"] = path
    return result


def rows(values, fields, path):
    return [brief(row, fields, f"{path}/{i}") for i, row in enumerate(values)]


def directory(payload):
    """Index every capability and its status; definitions are explicit lookups."""
    catalog = payload.get("capability_catalog")
    if catalog is None:
        return deepcopy(payload.get("capability_overview", {}))
    return [select(r, {"capability_id", "status", "missing_inputs", "missing_inputs_ref", "reason", "reason_ref"})
            for r in catalog]


def lookup_context(payload, context_id, *, focus_ids=()):
    result = select(payload, {"case_id", "revision", "smiles", "task", "mechanism_pool",
                              "memory_access", "retrieved_memory", "tool_discovery", "schema_feedback"})
    result = deepcopy(result)
    result["retrieved_memory"], result["memory_read_receipts"] = lookup_memory(result.get("retrieved_memory", []))
    result.update(stage="planner_context_lookup", decision_stage=payload["stage"],
                  context_record_id=context_id, output_contract=ContextReadPlan.model_json_schema())
    result["available_fields"] = [k for k in payload if k not in {
        "retrieved_memory", "memory_access", "tool_discovery",
    } and not (k == "capability_cards" and "tool_discovery" in payload)]
    hypotheses = payload.get("portfolio", {}).get("hypotheses", payload.get("proposed_ranking", []))
    result["candidates"] = [select(h, {"name", "status", "evidence_refs"}) for h in hypotheses]
    result["questions"] = rows(payload.get("questions", []), {
        "question_id", "question", "mechanism_id", "answer_state", "decision_role",
        "execution_state", "status",
    }, "/questions")
    result["current_working_state"] = current_state(payload)
    # New work must be discoverable before deciding which historical details to read.
    result["attention"] = deepcopy(select(payload, {
        "assimilation_context", "argument_review_pending", "unassimilated_result_ids",
        "run_limits",
    }))
    result["attention"]["pending_candidate_count"] = len((payload.get("candidate_context") or {}).get("pending", []))
    result["attention"]["feedback_paths"] = ["/" + k for k in (
        "result_review", "argument_review", "coverage_feedback", "stop_review_feedback", "predecision_review",
    ) if payload.get(k)]
    result["pending_results"] = rows(payload.get("pending_results", []), {
        "result_id", "question_ids", "status", "capability_id", "evidence_refs", "failure", "observation_receipt",
    }, "/pending_results")
    result["evidence_index"] = deepcopy(payload.get("evidence_index", []))
    result["capability_catalog"] = directory(payload)
    result["catalog_notice"] = "ID/status index only. Purposes, inputs and outputs: read /capability_catalog in context_record_id or describe_tools."
    for field in ("capability_catalog_shared", "capability_catalog_format"):
        if field in payload:
            result[field] = deepcopy(payload[field])
    result["loaded_tool_ids"] = payload.get("tool_discovery", {}).get("loaded_ids", [
        c["capability_id"] for c in payload.get("capability_cards", [])])
    result["focus_capability_ids"] = list(focus_ids)
    # Query errors/corrections are operational facts. Never hide them in the lookup phase.
    result["tool_query_feedback"] = deepcopy(payload.get("tool_discovery", {}).get("last_query"))
    return result


def decision_context(payload, context_id, *, focus_ids=()):
    iterative = payload["stage"] in ITERATIVE_STAGES
    result = deepcopy(select(payload, DECISION_FIELDS) if iterative else payload)
    result["context_version"] = "planner_delta_v1"
    result["context_record_id"] = context_id
    result["memory_policy"] = (
        "Current-case public state only. read_path is an exact JSON pointer in context_record_id. "
        "Omitted fields are retrievable, not absent. Retained findings/boundaries are accepted Planner "
        "accounts of cited observations, not new measurements. Reuse visible information within its "
        "conditions; read sources when details are missing, challenged or need reinterpretation. "
        "Facts, interpretations, failures and directory entries remain distinct. "
        "Keep unchanged records by omission; do not rewrite the full investigation. "
        "Decide directly when sufficient. Otherwise request prepare_context with focused reads or tool descriptions."
    )
    result.pop("memory_continuation", None)
    # Protocol instructions already stated in the stage prompt, not molecular facts.
    for field in ("incremental_update", "hypothesis_test_context", "boundary_decision_context",
                  "evidence_impact_context", "memory_access"):
        if isinstance(result.get(field), dict):
            result[field].pop("notice", None)
    if isinstance(result.get("incremental_update"), dict):
        result["incremental_update"].pop("contract", None)
    if isinstance(result.get("public_input_source"), dict):
        result["public_input_source"] = select(result["public_input_source"], {"evidence_id", "scope"})
    moved = {key: "/" + key for key in payload if key not in result}
    for key in ("previous_assessment", "comparison_context", "capability_catalog",
                "capability_overview", "capability_catalog_shared", "capability_catalog_format",
                "ability_evidence_guide", "hypothesis_write_contract", "question_write_contract"):
        if key in result:
            moved[key] = "/" + key
            result.pop(key)
    result["context_paths"] = moved
    if iterative:
        result["current_working_state"] = current_state(payload)
    result["capability_ids"] = (
        [r["capability_id"] for r in payload.get("capability_catalog", [])]
        or payload.get("capability_overview", {}).get("capability_ids_by_status", {})
    )
    if "capability_cards" in result:
        result["capability_cards"] = [compact_tool_card(c) for c in result["capability_cards"] if c["capability_id"] in focus_ids]
    for field in ("portfolio",):
        if field in result:
            result[field]["hypotheses"] = rows(payload[field].get("hypotheses", []), {
                "name", "confidence", "differential_priority", "claim_status", "status", "evidence_refs",
            }, f"/{field}/hypotheses")
    if "proposed_ranking" in result:
        result["proposed_ranking"] = rows(payload["proposed_ranking"], {
            "rank", "name", "confidence", "differential_priority", "evidence_support", "claim_status",
            "status", "evidence_refs", "validation_needed",
        }, "/proposed_ranking")
    result["questions"] = rows(payload.get("questions", []), {
        "question_id", "question", "mechanism_id", "related_labels", "status", "answer_state",
        "execution_state", "decision_role", "evidence_refs", "result_ids",
    } | ({"missing_or_unresolved"} if not iterative else set()), "/questions")
    for field in ("previous_support_arguments", "support_arguments"):
        if field in result:
            fields = {"support_id", "target_label", "support_level", "observation_refs", "finding", "boundary"}
            if not iterative:
                fields.add("warrant")
            result[field] = rows(payload[field], fields, "/" + field)
    boundary = result.get("boundary_decision_context", {})
    if boundary.get("current_candidate_bases"):
        boundary["current_candidate_bases"] = rows(boundary["current_candidate_bases"], {
            "label", "positive_refs", "counter_refs",
        } | ({"supported_observation", "inference_gap"} if not iterative else set()),
            "/boundary_decision_context/current_candidate_bases")
    if boundary.get("current_boundary_comparisons"):
        boundary["current_boundary_comparisons"] = rows(boundary["current_boundary_comparisons"], {
            "included_label", "excluded_label", "outcome", "basis", "evidence_refs",
        } | ({"remaining_difference"} if not iterative else set()),
            "/boundary_decision_context/current_boundary_comparisons")
    tests = result.get("hypothesis_test_context", {})
    if tests.get("recent_tests"):
        tests["recent_tests"] = rows(tests["recent_tests"], {
            "result_id", "revision", "round", "memory_record_id",
        } | ({"assessment"} if not iterative else set()), "/hypothesis_test_context/recent_tests")
    if tests.get("current_comparison"):
        tests["current_comparison"] = brief(tests["current_comparison"], {
            "leader", "challenger", "preference_status", "evidence_refs",
        } | ({"leading_hypothesis", "alternative_hypothesis"} if not iterative else set()),
            "/hypothesis_test_context/current_comparison")
    if "execution_history" in result:
        result["execution_history"] = rows(result["execution_history"], {
            "result_id", "question_ids", "capability_id", "status", "failure", "round_id", "observation_receipt",
        } | ({"delivery", "evidence_refs", "artifact_ids"} if not iterative else set()), "/execution_history")
    if "observations" in result:
        result["observations"] = project_observations(result["observations"])
    for i, item in enumerate(result.get("pending_results", [])):
        original = payload["pending_results"][i]
        item.clear()
        item.update(brief(original, {
            "result_id", "dispatch_id", "question_ids", "capability_id", "status", "failure", "delivery",
            "evidence_refs", "artifact_ids", "investigation_plan", "worker_explanation", "explanation_review",
            "observation_receipt", "required_observables",
        }, f"/pending_results/{i}"))
        if isinstance(item.get("investigation_plan"), dict):
            item["investigation_plan"] = brief(item["investigation_plan"], {
                "dispatch_id", "purpose", "candidate_labels", "decision_question", "possible_outcomes",
                "hypothesis_test",
            }, f"/pending_results/{i}/investigation_plan")
        if isinstance(item.get("worker_explanation"), dict):
            explanation = item["worker_explanation"]
            # Complete individual findings retain quantities, interpretations and limits.
            # The overall prose summary is available on demand, not sent a second time.
            if explanation.get("findings"):
                item["worker_explanation"] = brief(explanation, {
                    "question_ids", "findings", "missing_observations",
                }, f"/pending_results/{i}/worker_explanation")
    # Reviewed finding IDs/status stay visible. Positive fidelity commentary is not
    # an additional observation; corrections and unresolved reviews remain verbatim.
    def audits(value, path):
        if isinstance(value, dict):
            for key, child in list(value.items()):
                child_path = f"{path}/{key}"
                if key == "finding_audits" and isinstance(child, list):
                    value[key] = [brief(a, {"finding_id", "status", "correction"}, f"{child_path}/{i}")
                                  if a.get("status") == "faithful" and not a.get("correction") else a
                                  for i, a in enumerate(child)]
                else:
                    audits(child, child_path)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                audits(child, f"{path}/{i}")
    audits(result.get("result_review"), "/result_review")
    for i, item in enumerate(result.get("pending_results", [])):
        audits(item.get("explanation_review"), f"/pending_results/{i}/explanation_review")
    return result


def compact_tool_card(card):
    """Retain executable contracts and scientific limits, omit empty catalog metadata."""
    def schema(value):
        if isinstance(value, dict):
            maps = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
            return {k: {name: schema(node) for name, node in v.items()} if k in maps else schema(v)
                    for k, v in value.items() if k not in {"title", "$schema"}}
        if isinstance(value, list):
            return [schema(v) for v in value]
        return value
    fields = {"capability_id", "owner_agent", "evidence_family", "route", "description",
              "required_artifact_kinds", "source_artifact_kinds", "outputs", "tool_arg_schema", "question_loop_arg_schema",
              "requires_amesp", "backend_enabled", "disabled_reason", "allowed_argument_values",
              "usage_hints", "limitations", "execution_dependencies", "delivery_contract",
              "status", "missing_inputs", "reason"}
    return {k: schema(v) if k.endswith("schema") else deepcopy(v) for k, v in card.items()
            if k in fields and v not in (None, [], {})}


def delta_prompt(stage):
    name = "planner_lookup_delta.md" if stage == "planner_context_lookup" else (
        "planner_act_delta.md" if stage == "planner_question_act" else
        "planner_judgment_delta.md" if stage == "planner_assimilate" else None
    )
    return load_prompt(name) if name else None
