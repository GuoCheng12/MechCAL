"""Offline stage-scoping checks; fixtures make no scientific impact claims."""

from collections import Counter
from copy import deepcopy

import pytest

from mechcal.agents.planner import PlannerAgent
from mechcal.agents.planner_references import PlannerReferences
from mechcal.capabilities.discovery import ToolDiscoverySession, case_catalog
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.case_memory import CaseMemoryStore, serialized
from mechcal.runtime.planner_context import build_context
from mechcal.runtime.planner_memory import PlannerMemorySession
from mechcal.runtime.questions import public_payload, replace_state, state, validate_action
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.memory import MemoryQuery
from mechcal.schemas.question_loop import QuestionAction
from test_case_memory import Client, REGISTRY, fixture
from test_context_efficiency import expanded_context
from test_question_loop import action


STAGES = ("planner_assimilate", "planner_question_act", "planner_final_revision")


def scoped_context(stage="planner_assimilate"):
    payload = expanded_context(stage)
    payload["scoped_evidence_updates"] = True
    template = payload["questions"][0]
    payload["questions"] = [
        {**deepcopy(template), "question_id": qid, "answer_state": answer,
         "decision_role": role, "acceptable_capability_ids": [], "evidence_refs": [],
         "result_ids": [], "expected_basis": "proxy"}
        for qid, answer, role in (
            ("open", "open", "limitation"), ("partial", "partial", "material"),
            ("answered", "answered", "limitation"), ("deferred", "deferred", "limitation"),
            ("material", "deferred", "material"), ("pending", "answered", "limitation"),
            ("reviewed", "answered", "limitation"), ("stop_route", "deferred", "limitation"),
            ("stop_ref", "answered", "limitation"),
        )
    ]
    payload["pending_results"][0]["question_ids"] = ["pending"]
    payload["result_review"] = {"interpretations": [{"question_ids": ["reviewed"]}]}
    payload["stop_review_feedback"] = {"capability_ids": ["macro.fixture"], "evidence_refs": ["counter"]}
    payload["questions"][-2]["acceptable_capability_ids"] = ["macro.fixture"]
    payload["questions"][-1]["evidence_refs"] = ["counter"]
    return payload


@pytest.mark.parametrize("stage", STAGES)
def test_absent_and_disabled_flag_keep_legacy_projection_identical(stage):
    payload = scoped_context(stage)
    payload.pop("scoped_evidence_updates")
    legacy = build_context(payload)
    payload["scoped_evidence_updates"] = False
    disabled = build_context(payload)
    assert disabled.pop("scoped_evidence_updates") is False
    assert disabled == legacy
    assert disabled["context_version"] == "case_working_set_v3"


def test_assimilation_only_receives_registry_breadth_and_status(tmp_path):
    case, _ = fixture(tmp_path, scoped_evidence_updates=True, incremental_judgment=True)
    payload = scoped_context()
    catalog = case_catalog(REGISTRY, case)
    payload.update(capability_catalog=catalog, capability_cards=REGISTRY.cards(),
                   ability_evidence_guide={"unused": "guide"}, tool_discovery={"loaded_ids": []},
                   tool_loading_mode="on_demand", artifact_available_capability_ids=["unused"],
                   capability_catalog_shared={"1": "unused"}, capability_catalog_format="unused")
    original = deepcopy(payload)
    projected = build_context(payload)
    overview = projected["capability_overview"]
    assert overview["total"] == len(catalog)
    assert overview["status_counts"] == dict(Counter(row["status"] for row in catalog))
    assert {cid: status for status, ids in overview["capability_ids_by_status"].items() for cid in ids} == {
        row["capability_id"]: row["status"] for row in catalog
    }
    for key in ("capability_catalog", "capability_cards", "ability_evidence_guide", "tool_discovery",
                "tool_loading_mode", "artifact_available_capability_ids", "capability_catalog_shared",
                "capability_catalog_format"):
        assert key not in projected
    assert "tool_arg_schema" not in serialized(overview)
    assert "description" not in serialized(overview)
    assert payload == original


@pytest.mark.parametrize("stage", ("planner_question_act", "planner_final_revision"))
def test_non_assimilation_catalog_and_cards_remain_complete(tmp_path, stage):
    case, _ = fixture(tmp_path, scoped_evidence_updates=True, incremental_judgment=True)
    payload = scoped_context(stage)
    payload.update(capability_catalog=case_catalog(REGISTRY, case), capability_cards=REGISTRY.cards())
    projected = build_context(payload)
    restored = deepcopy(projected["capability_catalog"])
    for row in restored:
        for field in ("reason", "missing_inputs"):
            if field + "_ref" in row:
                row[field] = projected["capability_catalog_shared"][row.pop(field + "_ref")]
    assert restored == payload["capability_catalog"]
    assert projected["capability_cards"] == payload["capability_cards"]
    assert "capability_overview" not in projected


@pytest.mark.parametrize("stage", ("planner_assimilate", "planner_question_act"))
def test_questions_keep_open_and_explicitly_relevant_details_with_all_gaps(stage):
    payload = scoped_context(stage)
    original = deepcopy(payload)
    projected = build_context(payload)
    assert [q["question_id"] for q in projected["questions"]] == [q["question_id"] for q in payload["questions"]]
    for before, after in zip(payload["questions"], projected["questions"], strict=True):
        if before["question_id"] in {"answered", "deferred"}:
            assert "observable" not in after and "expected_basis" not in after
            assert after["memory_record_id"] == before["question_id"]
            for key in ("assessment", "missing_or_unresolved", "answer_state", "decision_role",
                        "acceptable_capability_ids", "evidence_refs", "result_ids"):
                assert after[key] == before[key]
        else:
            assert after == before
    assert payload == original


@pytest.mark.parametrize("stage", STAGES)
def test_no_label_score_gap_or_boundary_record_is_lost(stage):
    payload = scoped_context(stage)
    payload["mechanism_pool"] = list(MECHANISM_POOL)
    hypothesis = payload["portfolio"]["hypotheses"][0]
    basis = payload["boundary_decision_context"]["current_candidate_bases"][0]
    payload["portfolio"]["hypotheses"] = [
        {**deepcopy(hypothesis), "name": label, "confidence": index / len(MECHANISM_POOL),
         "differential_priority": 1 - index / len(MECHANISM_POOL), "validation_needed": [label + " gap"]}
        for index, label in enumerate(MECHANISM_POOL)
    ]
    payload["boundary_decision_context"]["current_candidate_bases"] = [
        {**deepcopy(basis), "label": label, "inference_gap": label + " inference gap"}
        for label in MECHANISM_POOL
    ]
    projected = build_context(payload)
    assert projected["mechanism_pool"] == payload["mechanism_pool"]
    assert projected["boundary_decision_context"] == payload["boundary_decision_context"]
    assert projected["evidence_index"] == payload["evidence_index"]
    for before, after in zip(payload["portfolio"]["hypotheses"], projected["portfolio"]["hypotheses"], strict=True):
        for key in ("name", "confidence", "differential_priority", "validation_needed", "evidence_refs"):
            assert after[key] == before[key]


def test_initial_and_final_questions_keep_complete_details():
    for stage, revision in (("planner_assimilate", 0), ("planner_final_revision", 2)):
        payload = scoped_context(stage)
        payload["revision"] = revision
        projected = build_context(payload)
        assert projected["questions"] == payload["questions"]
        assert projected["portfolio"] == payload["portfolio"]
        assert projected["previous_support_arguments"] == payload["previous_support_arguments"]


@pytest.mark.parametrize("stage", ("planner_assimilate", "planner_final_revision"))
def test_active_correction_and_exact_source_pins_are_not_weakened(stage):
    payload = scoped_context(stage)
    payload["argument_review_pending"] = True
    payload["final_internal_audit"] = {"evidence_refs": ["counter"], "reason": "Exact counter-source needed"}
    payload["previous_final_decision"] = {"unresolved_gaps": ["Do not erase this limitation"]}
    original = deepcopy(payload)
    projected = build_context(payload)
    for key in ("questions", "argument_review", "result_review", "previous_support_arguments",
                "final_internal_audit", "previous_final_decision"):
        assert projected[key] == payload[key]
    refs = {"old", "new", "counter"} if stage == "planner_final_revision" else {"old", "new"}
    assert projected["observations"] == [e for e in payload["observations"] if e["evidence_id"] in refs]
    assert projected["source_checks"] == [e for e in payload["source_checks"] if e["evidence_id"] in refs]
    for key in ("investigation_plan", "worker_explanation"):
        assert projected["pending_results"][0][key] == payload["pending_results"][0][key]
    assert payload == original


def test_missing_impact_review_does_not_claim_unaffected():
    payload = scoped_context()
    projected = build_context(payload)
    assert projected["observations"] == [payload["observations"][1]]
    assert projected["previous_support_arguments"][0]["memory_record_id"] == "argument:S1"
    assert "impact_receipt" not in projected and "evidence_impact_context" not in projected
    assert "capability_overview" not in projected
    assert "Scope alone does not establish tested impact" in projected["memory_policy"]["stage_scope"]


def test_compacted_records_remain_exactly_retrievable_and_stop_contract_keeps_all_questions(tmp_path):
    case, store = fixture(tmp_path, scoped_evidence_updates=True, incremental_judgment=True)
    questions = deepcopy(state(case)["questions"])
    questions["Q2"] = {**deepcopy(questions["Q1"]), "question_id": "Q2", "answer_state": "answered",
                       "assessment": "Exact accepted answer", "missing_or_unresolved": ["Residual limitation"]}
    questions["Q3"] = {**deepcopy(questions["Q2"]), "question_id": "Q3", "answer_state": "deferred"}
    execution = {"result_id": "X1", "capability_id": "macro.fixture", "status": "partial",
                 "question_ids": ["Q2"], "evidence_refs": [],
                 "structured_results": {"conditions": {"state": 3, "unit": "eV"}, "raw": [1, 2, 3]}}
    case = replace_state(case, questions=questions, results=[execution])
    original = case.model_dump(mode="json")
    payload = public_payload(case, REGISTRY, stage="planner_question_act")
    assert payload["scoped_evidence_updates"] is True
    payload["output_contract"] = QuestionAction.model_json_schema()
    projected = build_context(payload)
    index = CaseMemoryStore(store, case)
    public_questions = {q["question_id"]: q for q in payload["questions"]}
    for row in projected["questions"]:
        rid = row.get("memory_record_id", row["question_id"])
        exact = index.read(MemoryQuery(action="read_case_records", record_ids=[rid]))["records"][0]["record"]
        assert exact == public_questions[row["question_id"]]
    exact = index.read(MemoryQuery(action="read_case_records", record_ids=["X1"]))["records"][0]["record"]
    assert exact == execution
    assert projected["output_contract"] == payload["output_contract"]
    stopping = action(case.case_id, state(case)["revision"], questions=projected["questions"])
    validate_action(stopping, case, REGISTRY)
    stopping.question_dispositions.pop("Q3")
    with pytest.raises(ValueError, match="every retained question"):
        validate_action(stopping, case, REGISTRY)
    assert case.model_dump(mode="json") == original


def test_scoped_session_does_not_add_calls_queries_or_drop_action_discovery(tmp_path):
    case, store = fixture(tmp_path, scoped_evidence_updates=True, incremental_judgment=True)
    original = case.model_dump(mode="json")
    for stage in ("planner_assimilate", "planner_question_act"):
        payload = public_payload(case, REGISTRY, stage=stage)
        assert payload["scoped_evidence_updates"] is True
        memory = PlannerMemorySession(case, store, REGISTRY, stage, payload)
        client = Client([{"done": True}])
        discovery = ToolDiscoverySession(REGISTRY, case, "R001") if stage == "planner_question_act" else None
        assert memory.complete(client, PlannerReferences(case), payload=memory.payload,
                               system_prompt="fixture", discovery=discovery) == {"done": True}
        assert len(client.calls) == 1 and memory.used == 0 and not memory.loaded
        wire = client.calls[0]["payload"]
        if discovery:
            assert {row["capability_id"] for row in wire["capability_catalog"]} == {
                row["capability_id"] for row in case_catalog(REGISTRY, case)
            }
        else:
            assert "capability_catalog" not in wire and "capability_cards" not in wire
    assert case.model_dump(mode="json") == original


def test_memory_disabled_remains_disabled_even_with_scoped_flag(tmp_path):
    case, _ = fixture(tmp_path, scoped_evidence_updates=True, incremental_judgment=True,
                      planner_case_memory=False)
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    assert PlannerAgent()._memory_session(case, None, payload) is None
    assert payload["capability_catalog"] == case_catalog(REGISTRY, case)


def reviewed_impact_context():
    payload = scoped_context()
    argument = payload["previous_support_arguments"][0]
    hypothesis = payload["portfolio"]["hypotheses"][0]
    basis = payload["boundary_decision_context"]["current_candidate_bases"][0]
    for label in ("B", "C", "D"):
        ref = "source:" + label
        payload["portfolio"]["hypotheses"].append({
            **deepcopy(hypothesis), "name": label, "evidence_refs": [ref],
        })
        payload["boundary_decision_context"]["current_candidate_bases"].append({
            **deepcopy(basis), "label": label, "positive_refs": [ref],
        })
        payload["observations"].append({"evidence_id": ref, "metrics": {"energy": 2.3, "unit": "eV"},
                                        "context": label + " exact conditions", "limits": ["Synthetic fixture"]})
        payload["source_checks"].append({"evidence_id": ref, "executions": [{"status": "partial"}]})
        payload["evidence_index"].append({"evidence_id": ref, "claim": label + " source"})
        if label != "D":
            payload["previous_support_arguments"].append({
                **deepcopy(argument), "support_id": "S:" + label, "target_label": label,
                "observation_refs": [ref], "warrant": label + " exact warrant",
            })
    payload["mechanism_pool"] = ["A", "B", "C", "D"]
    payload["evidence_impact_context"] = {"targets": [{"label": "A", "current_support_ids": ["S1"]}]}
    payload["result_review"].update(status="reviewed", impact_review={
        "snapshot_id": "runtime-validated-fixture", "result_ids": ["X2"],
        "checked_support_ids": [a["support_id"] for a in payload["previous_support_arguments"]],
        "affected_support_ids": [], "affected_labels": [], "affected_comparison_ids": [],
        "global_review": False, "reason": "Synthetic impact routing, not scientific evidence",
    })
    return payload


@pytest.mark.parametrize("field,value", [("affected_support_ids", ["S:B"]), ("affected_labels", ["B"])])
def test_reviewer_scope_adds_uncited_targets_to_worker_scope_and_pins_exact_sources(field, value):
    payload = reviewed_impact_context()
    payload["result_review"]["impact_review"][field] = value
    original = deepcopy(payload)
    projected = build_context(payload)
    for before, after in zip(payload["previous_support_arguments"], projected["previous_support_arguments"], strict=True):
        if before["target_label"] in {"A", "B"}:
            assert after == before
        else:
            assert "warrant" not in after and after["memory_record_id"] == "argument:S:C"
    assert projected["portfolio"]["hypotheses"][:2] == payload["portfolio"]["hypotheses"][:2]
    for field in ("observations", "source_checks"):
        assert projected[field] == [row for row in payload[field] if row["evidence_id"] in {"new", "source:B"}]
    for field in ("mechanism_pool", "boundary_decision_context", "evidence_index", "result_review"):
        assert projected[field] == payload[field]
    assert len(projected["portfolio"]["hypotheses"]) == 4
    legacy = build_context({**payload, "scoped_evidence_updates": False})
    assert projected["pending_results"] == legacy["pending_results"]
    assert payload == original


def test_review_label_without_an_argument_keeps_hypothesis_basis_and_source_details():
    payload = reviewed_impact_context()
    payload["portfolio"]["hypotheses"][-1]["evidence_refs"] = []
    payload["result_review"]["impact_review"]["affected_labels"] = ["D"]
    projected = build_context(payload)
    assert projected["portfolio"]["hypotheses"][-1] == payload["portfolio"]["hypotheses"][-1]
    assert projected["observations"][-1] == payload["observations"][-1]
    assert projected["boundary_decision_context"] == payload["boundary_decision_context"]


@pytest.mark.parametrize("review_state", ["global", "missing", "unavailable", "stale_batch", "incomplete_coverage"])
def test_only_explicit_global_impact_pins_all_old_sources(review_state):
    payload = reviewed_impact_context()
    review = payload["result_review"]
    if review_state == "global":
        review["impact_review"]["global_review"] = True
    elif review_state == "missing":
        review.pop("impact_review")
    elif review_state == "unavailable":
        review.update(status="unavailable", impact_review=None)
    elif review_state == "stale_batch":
        review["impact_review"]["result_ids"] = ["old-batch"]
    else:
        review["impact_review"]["checked_support_ids"] = ["S1"]
    projected = build_context(payload)
    if review_state == "global":
        assert projected["previous_support_arguments"] == payload["previous_support_arguments"]
        assert projected["portfolio"] == payload["portfolio"]
        for field in ("observations", "source_checks"):
            assert projected[field] == [row for row in payload[field] if row["evidence_id"] != "counter"]
    else:
        legacy = build_context({**payload, "scoped_evidence_updates": False})
        for field in ("observations", "source_checks", "previous_support_arguments", "pending_results"):
            assert projected[field] == legacy[field]
        assert "Missing review is unknown" in projected["memory_policy"]["stage_scope"]
    assert projected["evidence_index"] == payload["evidence_index"]


@pytest.mark.parametrize("stage", ["planner_question_act", "planner_assimilate"])
def test_past_impact_review_does_not_expand_action_or_a_later_empty_batch(stage):
    payload = reviewed_impact_context()
    payload["stage"] = stage
    payload["result_review"]["impact_review"]["global_review"] = True
    if stage == "planner_assimilate":
        payload["pending_results"] = []
        payload["evidence_impact_context"] = {"targets": []}
    projected = build_context(payload)
    assert projected["observations"] == []
    assert all("warrant" not in row for row in projected["previous_support_arguments"])
    assert projected["result_review"]["memory_record_id"] == "review:result"
    assert projected["evidence_index"] == payload["evidence_index"]


def test_scoped_impact_pinning_does_not_drop_unrepresented_raw_results():
    payload = reviewed_impact_context()
    payload["result_review"]["impact_review"]["affected_labels"] = ["B"]
    payload["pending_results"][0]["evidence_refs"] = ["unrepresented-source"]
    original = deepcopy(payload)
    assert build_context(payload)["pending_results"] == original["pending_results"]
    assert payload == original


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stage", STAGES)
def test_emitted_session_requests_stay_scoped_after_discovery_queries_and_retries(tmp_path, enabled, stage):
    case, store = fixture(tmp_path, scoped_evidence_updates=enabled, incremental_judgment=True,
                          planner_strict_outputs=True)
    payload = public_payload(case, REGISTRY, stage=stage)
    payload["output_contract"] = {
        "type": "object", "properties": {"done": {"type": "boolean"}},
        "required": ["done"], "additionalProperties": False,
    }
    memory = PlannerMemorySession(case, store, REGISTRY, stage, payload)
    discovery = ToolDiscoverySession(REGISTRY, case, "R001")
    capability_id = case_catalog(REGISTRY, case)[0]["capability_id"]
    discovery.query({"action": "describe_tools", "capability_ids": [capability_id]})
    before = deepcopy(memory.payload)
    contract = PlannerOutputContract(memory.augment(memory.payload))
    query = MemoryQuery(action="read_case_records", record_ids=["Q1"]).model_dump(mode="json")
    client = Client([contract.encode(query, kind="memory_query"), contract.encode({"done": True}),
                     contract.encode({"done": True})])
    references = PlannerReferences(case)
    for feedback in (None, "Synthetic format retry"):
        assert memory.complete(client, references, payload=memory.payload, system_prompt="fixture",
                               discovery=discovery, schema_feedback=feedback) == {"done": True}
    assert len(client.calls) == 3 and memory.used == 1
    for call in client.calls:
        emitted = call["payload"]
        if enabled and stage == "planner_assimilate":
            for key in ("capability_catalog", "capability_cards", "tool_discovery", "tool_loading_mode",
                        "capability_catalog_format", "capability_catalog_shared"):
                assert key not in emitted
            assert emitted["capability_overview"] == before["capability_overview"]
        else:
            assert {row["capability_id"] for row in emitted["capability_catalog"]} == {
                row["capability_id"] for row in case_catalog(REGISTRY, case)
            }
            assert [row["capability_id"] for row in emitted["capability_cards"]] == [capability_id]
    assert client.calls[-1]["payload"]["retrieved_memory"][0]["records"][0]["record"]["question_id"] == "Q1"
    assert memory.payload == before


def test_final_emission_removes_late_cards_without_overwriting_existing_overview(tmp_path):
    case, store = fixture(tmp_path, scoped_evidence_updates=True, incremental_judgment=True)
    payload = public_payload(case, REGISTRY, stage="planner_assimilate")
    memory = PlannerMemorySession(case, store, REGISTRY, payload["stage"], payload)
    memory.payload["capability_cards"] = REGISTRY.cards()
    before = deepcopy(memory.payload)
    client = Client([{"done": True}])
    assert memory.complete(client, PlannerReferences(case), payload=memory.payload,
                           system_prompt="fixture") == {"done": True}
    emitted = client.calls[0]["payload"]
    assert "capability_cards" not in emitted and "capability_catalog" not in emitted
    assert emitted["capability_overview"] == before["capability_overview"]
    assert memory.payload == before


def test_answered_material_question_is_brief_unless_explicitly_linked_to_current_work():
    payload = scoped_context("planner_question_act")
    question = next(q for q in payload["questions"] if q["question_id"] == "answered")
    question["decision_role"] = "material"
    projected = build_context(payload)
    brief = next(q for q in projected["questions"] if q["question_id"] == "answered")
    assert brief["memory_record_id"] == "answered" and "observable" not in brief
    assert brief["assessment"] == question["assessment"]
    assert brief["missing_or_unresolved"] == question["missing_or_unresolved"]
    question["acceptable_capability_ids"] = payload["stop_review_feedback"]["capability_ids"]
    assert next(q for q in build_context(payload)["questions"] if q["question_id"] == "answered") == question
