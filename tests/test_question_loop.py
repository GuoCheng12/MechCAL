"""Scripted control-flow/fault tests. These are not model performance results."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.capabilities import default_capability_registry
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.gates import DeterministicGate
from mechcal.runtime.questions import (
    apply_assimilation, decision_from_action, dependency_inputs, public_payload, replace_state, request_signature,
    state, validate_action, validate_assimilation, validate_review,
)
from mechcal.schemas import DispatchRequest
from mechcal.schemas.question_loop import QuestionAction, QuestionAssimilation, ResultReview
from test_runtime import _case_run_for_gate, _evidence_unit, _fake_llm_deps


REGISTRY = default_capability_registry()
TOOL = "macro.screen_rotor_torsion_topology"
NOTE = "Offline control-flow fixture, not a scientific inference."


def question(qid="Q1"):
    return dict(question_id=qid, mechanism_id="RIM_RIR_RIV", question=NOTE,
                observable="rotor topology", expected_basis="proxy",
                related_labels=["RIM_RIR_RIV", "ICT_TICT_CT"], assessment=NOTE)


def assimilation(payload):
    return dict(
        case_id=payload["case_id"], revision=payload["revision"] + 1,
        questions=payload.get("questions") or [question()],
        portfolio={"current": "RIM_RIR_RIV", "runner_up": "ICT_TICT_CT", "hypotheses": [
            {"name": label, "differential_priority": p, "rationale": NOTE}
            for label, p in [("RIM_RIR_RIV", .6), ("ICT_TICT_CT", .3), ("ESIPT_PT", .1)]
        ]},
        dispositions=[dict(result_id=r["result_id"], question_ids=r["question_ids"],
                           outcome="no_result" if r["status"] in {"failed", "unsupported", "precondition_missing"}
                           else "non_discriminating", interpretation=NOTE,
                           reviewer_response="No approval inferred from unavailable review. " + NOTE,
                           evidence_refs=r["evidence_refs"])
                      for r in payload.get("pending_results", [])],
        decision_assessment={"current_best": NOTE, "weakened_or_unresolved": NOTE,
                             "next_useful_step": NOTE, "evidence_refs": []},
        coverage_response=NOTE,
        audit_responses={sid: NOTE for sid in (payload.get("argument_review") or {}).get("support_ids", [])}
        if payload.get("argument_review_pending") else {},
    )


def request(round_id="R001", *, cid=TOOL, questions=None, args=None):
    spec = REGISTRY.require(cid)
    return DispatchRequest(dispatch_id=round_id + ":test", round_id=round_id,
                           agent_name=spec.owner_agent, capability_id=cid, route=spec.route,
                           evidence_goal_family=spec.evidence_family, task=NOTE, objective=NOTE,
                           question_ids=questions or ["Q1"], expected_observation=NOTE,
                           tool_args=args or {})


def action(case_id, revision, *, dispatch=False, round_id="R001", questions=None, at_limit=False):
    return QuestionAction(
        case_id=case_id, revision=revision, action="dispatch" if dispatch else "finalize",
        dispatch_requests=[request(round_id)] if dispatch else [],
        stopping_assessment={"judgment_state": "unresolved", "finish_basis": None
                             if dispatch or at_limit else "no_useful_available_step", "reason": NOTE},
        question_dispositions={q["question_id"]: NOTE for q in questions or [question()]},
        ranking_justification=NOTE, ranking_uncertainty_material=True, unresolved_gaps=[NOTE],
        final_answer_draft=None if dispatch else NOTE, rationale=NOTE, reviewer_response=NOTE,
    )


def initial_case():
    return replace_state(_case_run_for_gate(), phase="assimilate", revision=0,
                         questions={}, results=[], pending_results=[])


def ready_case():
    case = initial_case()
    update = QuestionAssimilation.model_validate(assimilation(dict(case_id=case.case_id, revision=0)))
    return apply_assimilation(case, update)


def test_action_cannot_skip_assimilation_or_mutate_ranking():
    c = initial_case()
    a = action(c.case_id, 1)
    with pytest.raises(ValueError, match="Assimilate"):
        validate_action(a, c, REGISTRY)
    c = ready_case()
    d = decision_from_action(a, c, "R001")
    assert DeterministicGate().apply(d, c).status == "finalize"
    d = d.model_copy(update={"portfolio": d.portfolio.model_copy(update={"current": "ICT_TICT_CT"})})
    assert DeterministicGate().apply(d, c).status == "retry_policy"
    with pytest.raises(ValueError):
        QuestionAction.model_validate({**a.model_dump(), "portfolio": {}})


@pytest.mark.parametrize("change", [dict(case_id="another_case"), dict(revision=0)])
def test_stale_or_cross_case_actions_rejected(change):
    c = ready_case()
    a = action(c.case_id, 1).model_copy(update=change)
    with pytest.raises(ValueError):
        validate_action(a, c, REGISTRY)


@pytest.mark.parametrize("change", [dict(question_dispositions={}), dict(ranking_justification=""),
                                  dict(unresolved_gaps=[])])
def test_finalization_must_account_for_open_questions(change):
    c = ready_case()
    with pytest.raises(ValueError):
        validate_action(action(c.case_id, 1).model_copy(update=change), c, REGISTRY)


def test_question_persistence_and_reopening():
    c = ready_case()
    c = replace_state(c, phase="assimilate")
    data = assimilation(dict(case_id=c.case_id, revision=1, questions=list(state(c)["questions"].values())))
    data["questions"][0]["answer_state"] = "deferred"
    c = apply_assimilation(c, QuestionAssimilation.model_validate(data))
    c = replace_state(c, phase="assimilate")
    data = assimilation(dict(case_id=c.case_id, revision=2, questions=list(state(c)["questions"].values())))
    data["questions"][0]["answer_state"] = "open"
    assert apply_assimilation(c, QuestionAssimilation.model_validate(data)).portfolio == c.portfolio
    data["questions"] = [question("Q2")]
    with pytest.raises(ValueError, match="delete"):
        validate_assimilation(QuestionAssimilation.model_validate(data), c)


def pending_case(status="failed"):
    c = ready_case()
    r = dict(result_id="X1", question_ids=["Q1"], status=status, evidence_refs=[],
             capability_id=TOOL, structured_results={"missing_deliverables": ["target"]})
    return replace_state(c, phase="assimilate", results=[r], pending_results=[r])


@pytest.mark.parametrize("status", ["success", "partial", "failed", "unsupported", "precondition_missing"])
def test_all_results_need_disposition_and_missing_is_not_negative(status):
    c = pending_case(status)
    data = assimilation(dict(case_id=c.case_id, revision=1, pending_results=state(c)["pending_results"]))
    good = QuestionAssimilation.model_validate(data)
    updated = apply_assimilation(c, good)
    assert updated.portfolio == c.portfolio
    assert state(updated)["questions"]["Q1"]["answer_state"] == "open"
    assert len(updated.evidence_ledger.items) == len(c.evidence_ledger.items)
    data["dispositions"] = []
    with pytest.raises(ValueError, match="Every pending"):
        validate_assimilation(QuestionAssimilation.model_validate(data), c)


def test_failure_cannot_be_declared_informative_and_review_cannot_rank():
    c = pending_case()
    data = assimilation(dict(case_id=c.case_id, revision=1, pending_results=state(c)["pending_results"]))
    data["dispositions"][0]["outcome"] = "informative"
    with pytest.raises(ValueError, match="Failed execution"):
        validate_assimilation(QuestionAssimilation.model_validate(data), c)
    with pytest.raises(ValueError):
        ResultReview.model_validate({"status": "reviewed", "ranking": ["RIM_RIR_RIV"]})
    review = ResultReview(interpretations=[dict(result_id="X1", question_ids=["Q1"],
                                              delivered=NOTE, missing=NOTE, boundary=NOTE)])
    validate_review(review, c)
    review.interpretations[0].evidence_refs = ["another_case:observation"]
    with pytest.raises(ValueError, match="cite evidence"):
        validate_review(review, c)


def test_duplicate_and_shared_dependency_checks_allow_explicit_retry():
    c = ready_case()
    a = action(c.case_id, 1, dispatch=True)
    r = dict(result_id="X1", status="success", signature=request_signature(a.dispatch_requests[0], REGISTRY))
    c = replace_state(c, results=[r])
    with pytest.raises(ValueError, match="retry_reason"):
        validate_action(a, c, REGISTRY)
    a.dispatch_requests[0].retry_reason = "Deliberate replication fixture."
    validate_action(a, c, REGISTRY)
    local = REGISTRY.require("microscopic.run_targeted_localized_orbital_analysis")
    natural = REGISTRY.require("microscopic.run_targeted_natural_orbital_analysis")
    assert local.metadata["execution_dependencies"] == natural.metadata["execution_dependencies"]
    c = replace_state(c, results=[dict(result_id="X2", status="failed",
                                     dependency_inputs=dependency_inputs(request(cid=natural.capability_id)),
                                     failed_dependencies=local.metadata["execution_dependencies"])])
    a.dispatch_requests = [request(cid=natural.capability_id)]
    validate_action(a, c, REGISTRY)  # Shared failed substeps alone do not prove identical work.


def test_changed_parameters_and_input_are_distinct_work():
    cid = "microscopic.run_torsion_brightness_coupling_scan"
    a = request(cid=cid, args={"angles": [0, 90]})
    b = request(cid=cid, args={"angles": [0, 180]})
    assert request_signature(a, REGISTRY) != request_signature(b, REGISTRY)
    b = a.model_copy(update={"input_refs": ["other_geometry"]})
    assert request_signature(a, REGISTRY) != request_signature(b, REGISTRY)


class QuestionClient:
    def __init__(self, rounds=1, discover=False):
        self.calls = []
        self.rounds = rounds
        self.discover = discover

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        self.calls.append(deepcopy(payload))
        assert schema_feedback is None, schema_feedback
        if payload["stage"] == "planner_assimilate":
            assert "dispatch_requests" not in payload["output_contract"]["properties"]
            return assimilation(payload)
        if payload["stage"] == "planner_final_revision":
            update = assimilation(payload)
            final = action(payload["case_id"], update["revision"], questions=payload["questions"],
                           at_limit=not payload["run_limits"]["scientific_tools_allowed"])
            return dict(update=update, final_action=final.model_dump(mode="json"),
                        audit_responses={a["support_id"]: NOTE for a in payload["previous_support_arguments"]})
        assert payload["stage"] == "planner_question_act"
        tools = payload.get("tool_discovery", {})
        if self.discover and tools.get("remaining_batches") and not tools.get("loaded_ids"):
            return dict(action="describe_tools", capability_ids=[TOOL])
        n = int(payload["round_id"][1:])
        a = action(payload["case_id"], payload["revision"],
                   dispatch=n <= self.rounds and payload["run_limits"]["scientific_tools_allowed"],
                   round_id=payload["round_id"], questions=payload["questions"],
                   at_limit=not payload["run_limits"]["scientific_tools_allowed"])
        if a.dispatch_requests and n > 1:
            a.dispatch_requests[0].retry_reason = "Offline repeated-work fixture."
        return a.model_dump(mode="json")


def loop(tmp_path, *, rounds=1, unlimited=False, discover=False):
    config = OrchestratorConfig(run_base_dir=tmp_path, decision_mode="question_loop_v1",
                                max_exploration_rounds=None if unlimited else rounds,
                                unlimited_exploration=unlimited, enable_support_auditor=False,
                                enable_agenda_coverage_reviewer=False,
                                planner_tool_loading="on_demand" if discover else "eager")
    deps = _fake_llm_deps(config)
    client = QuestionClient(rounds, discover)
    deps.planner.llm_client = client
    deps.planner.tool_loading_mode = config.planner_tool_loading
    return MechCALOrchestrator(config, deps), client


def test_end_to_end_round_accounting_discovery_and_saved_updates(tmp_path):
    o, client = loop(tmp_path, discover=True)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="fixture")
    assert result.status == "finalized"
    assert result.round_ids == ["R001", "R002"]
    assert result.runtime["worker_call_counts"]["macro"] == 1
    assert result.runtime["termination"]["reason"] == "exploration_limit_reached"
    assert state(result)["phase"] == "done"
    updates = [p for p in client.calls if p["stage"] == "planner_assimilate"]
    assert len(updates) == 3  # initial, result, one bounded stop-review response
    assert updates[1]["pending_results"]
    assert len(state(result)["results"]) == 1
    assert result.differential_mechanism_portfolio is None  # no second ranking agent
    assert (tmp_path / "fixture/questions/revision_0002.json").exists()
    recorded = json.loads((tmp_path / "fixture/rounds/R001/round.json").read_text())
    assert len(recorded["agent_reports"]) == 1
    assert recorded["agent_reports"][0]["evidence_units"]
    before = len(client.calls)
    resumed = o.run(smiles="CCO", user_query=NOTE, case_id="fixture", resume=True)
    assert resumed == result
    assert len(client.calls) == before
    with pytest.raises(FileExistsError):
        o.run(smiles="CCO", user_query=NOTE, case_id="fixture")
    saved_input = (tmp_path / "fixture/input.json").read_bytes()
    saved_run = (tmp_path / "fixture/case_run.json").read_bytes()
    with pytest.raises(ValueError, match="differs"):
        o.run(smiles="CCC", user_query=NOTE, case_id="fixture", resume=True)
    assert (tmp_path / "fixture/input.json").read_bytes() == saved_input
    assert (tmp_path / "fixture/case_run.json").read_bytes() == saved_run


def test_uncapped_loop_passes_eight_rounds_without_forced_stop(tmp_path):
    o, _ = loop(tmp_path, rounds=9, unlimited=True)
    r = o.run(smiles="CCO", user_query=NOTE, case_id="uncapped")
    assert r.round_ids[-1] == "R010"
    assert r.runtime["termination"]["reason"] == "no_useful_available_step"
    assert not r.runtime["termination"]["limit_reached"]


def test_resume_reconciles_completed_tool_without_reexecution(tmp_path, monkeypatch):
    import mechcal.runtime.question_loop as module

    o, _ = loop(tmp_path)
    original = module.checkpoint
    def crash_after_tool(case, store):
        if state(case).get("completed_dispatch_ids"):
            raise KeyboardInterrupt("simulated interruption after tool history persisted")
        return original(case, store)
    monkeypatch.setattr(module, "checkpoint", crash_after_tool)
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="resume")
    path = tmp_path / "resume/history/tool_events.jsonl"
    before = path.read_text()
    monkeypatch.setattr(module, "checkpoint", original)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="resume", resume=True)
    assert result.status == "finalized"
    assert path.read_text() == before
    assert result.runtime["worker_call_counts"]["macro"] == 1


def test_unknown_inflight_never_replayed(tmp_path, monkeypatch):
    o, _ = loop(tmp_path)
    monkeypatch.setattr(o, "_dispatch_all", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        o.run(smiles="CCO", user_query=NOTE, case_id="unknown")
    with pytest.raises(RuntimeError, match="unknown completion"):
        o.run(smiles="CCO", user_query=NOTE, case_id="unknown", resume=True)


def test_old_results_readable_and_question_mode_requires_explicit_limit(tmp_path):
    from mechcal.schemas import CaseRun

    c = _case_run_for_gate()
    assert CaseRun.model_validate_json(c.model_dump_json()) == c
    with pytest.raises(ValueError, match="explicit exploration"):
        OrchestratorConfig(run_base_dir=tmp_path, decision_mode="question_loop_v1")
    assert json.loads(c.model_dump_json())["runtime"].get("question_loop") is None


def test_multiple_results_multiple_questions_and_duplicate_result_rejected():
    c = pending_case("partial")
    r1 = state(c)["pending_results"][0]
    r1["question_ids"] = ["Q1", "Q2"]
    r2 = {**r1, "result_id": "X2", "question_ids": ["Q2"]}
    c = replace_state(c, results=[r1, r2], pending_results=[r1, r2])
    data = assimilation(dict(case_id=c.case_id, revision=1,
                             questions=[question(), question("Q2")], pending_results=[r1, r2]))
    validate_assimilation(QuestionAssimilation.model_validate(data), c)
    data["dispositions"][1] = data["dispositions"][0]
    with pytest.raises(ValueError, match="Every pending"):
        validate_assimilation(QuestionAssimilation.model_validate(data), c)


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("existing_question", [False, True])
def test_same_case_result_reuse_preserves_execution_provenance(pending, existing_question):
    c = pending_case("success")
    if not pending:
        c = replace_state(c, pending_results=[])
    if existing_question:
        c = replace_state(c, questions={**state(c)["questions"], "Q2": question("Q2")})
    before = c.model_dump(mode="json")
    data = assimilation(dict(case_id=c.case_id, revision=1,
                             pending_results=state(c)["pending_results"],
                             questions=[question(), {**question("Q2"), "result_ids": ["X1"],
                                 "assessment": "The same rotor observation also informs this broader question."}]))
    updated = apply_assimilation(c, QuestionAssimilation.model_validate(data))
    assert state(updated)["questions"]["Q2"]["result_ids"] == ["X1"]
    assert state(updated)["results"] == before["runtime"]["question_loop"]["results"]
    assert state(updated)["results"][0]["question_ids"] == ["Q1"]
    assert updated.evidence_ledger == c.evidence_ledger
    assert updated.portfolio == c.portfolio
    assert c.model_dump(mode="json") == before


def test_result_reuse_cannot_reassign_disposition_or_review_links():
    c = pending_case("success")
    data = assimilation(dict(case_id=c.case_id, revision=1,
                             pending_results=state(c)["pending_results"],
                             questions=[question(), {**question("Q2"), "result_ids": ["X1"]}]))
    data["dispositions"][0]["question_ids"] = ["Q2"]
    with pytest.raises(ValueError, match="Disposition X1 must preserve request-question links") as err:
        validate_assimilation(QuestionAssimilation.model_validate(data), c)
    assert "Q1" in str(err.value) and "Q2" in str(err.value)
    review = ResultReview(interpretations=[dict(result_id="X1", question_ids=["Q2"],
                                              delivered=NOTE, missing=NOTE, boundary=NOTE)])
    with pytest.raises(ValueError, match="original request"):
        validate_review(review, c)


@pytest.mark.parametrize("field,foreign_id", [("result_ids", "foreign:execution"),
                                              ("evidence_refs", "foreign:observation")])
def test_result_reuse_rejects_unknown_and_foreign_case_references(field, foreign_id):
    c = pending_case("success")
    foreign = replace_state(initial_case().touch(case_id="foreign_case"),
                            results=[dict(result_id=foreign_id, question_ids=["Q2"],
                                          status="success", evidence_refs=[])])
    assert foreign_id not in {r["result_id"] for r in state(c)["results"]}
    assert foreign_id in {r["result_id"] for r in state(foreign)["results"]}
    data = assimilation(dict(case_id=c.case_id, revision=1,
                             pending_results=state(c)["pending_results"],
                             questions=[question(), {**question("Q2"), field: [foreign_id]}]))
    with pytest.raises(ValueError) as err:
        validate_assimilation(QuestionAssimilation.model_validate(data), c)
    message = str(err.value)
    assert all(value in message for value in ["Q2", c.case_id, field, foreign_id])


def test_failed_result_reuse_preserves_no_result_and_creates_no_support():
    c = pending_case("unsupported")
    data = assimilation(dict(case_id=c.case_id, revision=1,
                             pending_results=state(c)["pending_results"],
                             questions=[question(), {**question("Q2"), "result_ids": ["X1"],
                                 "execution_state": "blocked", "assessment": "This route is unavailable, not counterevidence."}]))
    updated = apply_assimilation(c, QuestionAssimilation.model_validate(data))
    assert state(updated)["assimilation"]["dispositions"][0]["outcome"] == "no_result"
    assert state(updated)["results"][0]["status"] == "unsupported"
    assert updated.mechanism_support_arguments == []
    assert updated.evidence_ledger == c.evidence_ledger
    data["dispositions"][0]["outcome"] = "informative"
    with pytest.raises(ValueError, match="Failed execution"):
        validate_assimilation(QuestionAssimilation.model_validate(data), c)


def test_reused_result_observations_remain_in_planner_context():
    c = pending_case("success")
    ev = _evidence_unit(evidence_id="obs:X1", round_id="R001", source_report_id="report:X1",
                        agent_name="macro", family="torsion_sensitivity", summary=NOTE)
    other = ev.model_copy(update={"evidence_id": "obs:unrelated"})
    c = c.touch(evidence_ledger=c.evidence_ledger.model_copy(update={"items": [ev, other]}))
    r = {**state(c)["results"][0], "evidence_refs": [ev.evidence_id]}
    c = replace_state(c, results=[r], pending_results=[])
    data = assimilation(dict(case_id=c.case_id, revision=1,
                             questions=[question(), {**question("Q2"), "result_ids": ["X1"]}]))
    updated = apply_assimilation(c, QuestionAssimilation.model_validate(data))
    payload = public_payload(updated, REGISTRY, stage="planner_assimilate")
    assert [e["evidence_id"] for e in payload["observations"]] == [ev.evidence_id]
    assert payload["execution_history"][0]["question_ids"] == ["Q1"]
    assert len(payload["evidence_index"]) == 2


@pytest.mark.parametrize("exhaust", [False, True])
def test_rejected_assimilation_saved_without_mutating_checkpoint(tmp_path, exhaust):
    from mechcal.runtime.retry import SchemaRetryError

    o, _ = loop(tmp_path, rounds=0, unlimited=True)
    class Client(QuestionClient):
        def __init__(self):
            super().__init__(rounds=0)
            self.attempts = 0
            self.rejected = []
            self.feedback = []

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_assimilate" and payload["revision"] == 0:
                self.attempts += 1
                self.feedback.append(schema_feedback)
                data = assimilation(payload)
                if self.attempts == 1:
                    data["portfolio"]["hypotheses"][0]["differential_priority"] = 2
                elif self.attempts == 2 or exhaust:
                    data["questions"][0]["result_ids"] = ["foreign:execution"]
                else:
                    return data
                self.rejected.append(deepcopy(data))
                return data
            return super().complete_json(payload=payload, system_prompt=system_prompt,
                                         schema_feedback=schema_feedback)

    client = Client()
    o.deps.planner.llm_client = client
    if exhaust:
        with pytest.raises(SchemaRetryError):
            o.run(smiles="CCO", user_query=NOTE, case_id="invalid-assimilation")
        saved = json.loads((tmp_path / "invalid-assimilation/question_checkpoint.json").read_text())
        assert saved["runtime"]["question_loop"]["revision"] == 0
        assert saved["runtime"]["question_loop"]["phase"] == "assimilate"
        assert saved["runtime"]["question_loop"]["questions"] == {}
    else:
        result = o.run(smiles="CCO", user_query=NOTE, case_id="invalid-assimilation")
        assert result.status == "finalized"
        assert state(result)["revision"] == 2  # initial, stop-review response; retries do not advance
        assert result.runtime["worker_call_counts"]["macro"] == 0
        saved = result.model_dump(mode="json")
    events = [json.loads(line) for line in (
        tmp_path / "invalid-assimilation/history/planner_assimilation_failures.jsonl"
    ).read_text().splitlines()]
    assert len(events) == (3 if exhaust else 2)
    assert [e["response"] for e in events] == client.rejected
    assert [e["attempt"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["validation_stage"] == "schema"
    assert events[1]["validation_stage"] == "validator"
    assert "foreign:execution" in events[1]["error"]
    assert json.loads(client.feedback[1])["error"] == events[0]["error"]
    assert json.loads(client.feedback[2])["error"] == events[1]["error"]
    assert json.loads(client.feedback[1])["rejected_response"] == events[0]["response"]
    assert client.attempts == 3
    assert all(e["case_id"] == "invalid-assimilation" and e["round_id"] == "R001"
               and e["revision"] == 1 and e["max_attempts"] == 3 for e in events)
    accepted = {k: saved[k] for k in ["portfolio", "evidence_ledger", "mechanism_support_arguments"]}
    accepted["questions"] = saved["runtime"]["question_loop"]["questions"]
    assert "foreign:execution" not in json.dumps(accepted)
    assert "planner_assimilation_failures" not in json.dumps(saved)


def test_auditor_unavailable_is_visible_and_not_evidence(tmp_path, monkeypatch):
    o, client = loop(tmp_path)
    o.config = replace(o.config, enable_support_auditor=True, continue_on_arbiter_failure=True)
    def unavailable(_, **kwargs):
        raise TimeoutError("offline injected timeout")
    monkeypatch.setattr(o.deps.photophysics_arbiter, "review_results", unavailable)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="auditor-failure")
    seen = [p for p in client.calls if p["stage"] == "planner_assimilate" and p["pending_results"]]
    assert seen[0]["result_review"]["status"] == "unavailable"
    assert seen[0]["pending_results"][0]["structured_results"]
    assert state(result)["result_review"]["interpretations"] == []
    assert all("review" not in e.evidence_id for e in result.evidence_ledger.items)


def test_successful_result_review_uses_only_execution_sources(tmp_path):
    from mechcal.agents.photophysics_arbiter import PhotophysicsArbiterAgent
    c = pending_case("success")
    class Client:
        def complete_json(self, **kwargs):
            self.payload = kwargs["payload"]
            return dict(status="reviewed", interpretations=[dict(
                result_id="X1", question_ids=["Q1"], delivered=NOTE, missing=NOTE, boundary=NOTE)])
    client = Client()
    review = PhotophysicsArbiterAgent(llm_client=client).review_results(c)
    assert review.status == "reviewed"
    assert "portfolio" not in client.payload
    assert client.payload["pending_results"][0]["structured_results"]["missing_deliverables"]


def test_review_challenge_requires_response_but_cannot_force_tools(tmp_path, monkeypatch):
    from mechcal.schemas.decisions import StopReview
    o, client = loop(tmp_path, rounds=0, unlimited=True)
    o.config = replace(o.config, enable_agenda_coverage_reviewer=True)
    monkeypatch.setattr(o.deps.mechanism_agenda, "review_stop", lambda *a, **kw: StopReview(
        rationale=NOTE, challenge="Offline challenge: reconsider the answer."))
    result = o.run(smiles="CCO", user_query=NOTE, case_id="reviewed")
    assert result.status == "finalized"
    assert result.runtime["worker_call_counts"]["macro"] == 0
    assert len([p for p in client.calls if p["stage"] == "planner_assimilate"]) == 2
    assert result.runtime["termination"]["reviewer_response"]


def test_schema_retries_do_not_reset_discovery_allowance(tmp_path):
    o, _ = loop(tmp_path, discover=True)
    class Client(QuestionClient):
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_question_act" and payload["round_id"] == "R001":
                remaining = payload["tool_discovery"]["remaining_batches"]
                if remaining:
                    return dict(action="describe_tools", capability_ids=[TOOL])
                if schema_feedback is None:
                    return {"bad": "force bounded format retry"}
                assert remaining == 0
            return super().complete_json(payload=payload, system_prompt=system_prompt)
    client = Client()
    o.deps.planner.llm_client = client
    result = o.run(smiles="CCO", user_query=NOTE, case_id="query-retries")
    assert result.status == "finalized"
    events = [json.loads(l) for l in (tmp_path / "query-retries/history/tool_discovery.jsonl").read_text().splitlines()]
    queries = [e for e in events if e["kind"] == "tool_description_query" and e["round_id"] == "R001"]
    assert len(queries) == 2


def test_unloaded_tool_is_rejected_before_execution(tmp_path):
    o, client = loop(tmp_path, discover=True)
    client.discover = False
    # Deliberately refuse format feedback to demonstrate failure before any tool.
    with pytest.raises(AssertionError, match="Load current details"):
        o.run(smiles="CCO", user_query=NOTE, case_id="unloaded")
    rows = [json.loads(line) for line in (tmp_path / "unloaded/history/tool_events.jsonl").read_text().splitlines()]
    assert all(row["event"] == "prepare_structure" for row in rows)


def test_action_cannot_smuggle_an_unlinked_request():
    c = ready_case()
    a = action(c.case_id, 1, dispatch=True)
    d = decision_from_action(a, c, "R001")
    d.dispatch_requests = [request(questions=["not-a-question"])]
    assert DeterministicGate().apply(d, c).status == "retry_policy"


def test_same_round_same_route_results_remain_distinct(tmp_path):
    o, _ = loop(tmp_path)
    c = ready_case()
    requests = [request(args={"focus_tags": [tag]}) for tag in ["rotors", "flexibility"]]
    requests[1].dispatch_id = "R001:other"
    traces = [o.deps.workers["macro"].run_with_trace(r, c) for r in requests]
    assert traces[0][0].tool_result_id != traces[1][0].tool_result_id
    assert traces[0][1].report_id != traces[1][1].report_id
    combined = o._apply_reports(c, [t[1] for t in traces])
    assert len(combined.evidence_ledger.items) == 2
    assert all(t[0].raw_results["original_tool_result_id"] == "R001:macro" for t in traces)


def test_failed_dependency_does_not_block_changed_input():
    c = ready_case()
    cid = "microscopic.run_targeted_natural_orbital_analysis"
    r = request(cid=cid)
    c = replace_state(c, results=[dict(result_id="X", status="failed",
                                     dependency_inputs=dependency_inputs(r),
                                     failed_dependencies=["amesp.s0_orbital_singlepoint"])])
    a = action(c.case_id, 1, dispatch=True)
    a.dispatch_requests = [r.model_copy(update={"input_refs": ["different:geometry"]})]
    validate_action(a, c, REGISTRY)


def test_forged_review_reference_does_not_become_evidence():
    c = initial_case()
    data = assimilation(dict(case_id=c.case_id, revision=0))
    data["portfolio"]["hypotheses"][0]["evidence_refs"] = ["review:recommendation"]
    with pytest.raises(ValueError, match="existing observations"):
        validate_assimilation(QuestionAssimilation.model_validate(data), c)


def test_exported_contract_exposes_actual_priority_and_reference_constraints():
    contract = QuestionAssimilation.model_json_schema()["$defs"]
    priority = contract["HypothesisEntry"]["properties"]["differential_priority"]
    number = next(item for item in priority["anyOf"] if item["type"] == "number")
    assert number["minimum"] == 0 and number["maximum"] == 1
    refs = contract["MechanismSupportArgument"]["properties"]["observation_refs"]
    assert refs["minItems"] == 1


def test_initial_hypotheses_need_no_fabricated_support():
    c = initial_case()
    data = assimilation(dict(case_id=c.case_id, revision=0))
    update = QuestionAssimilation.model_validate(data)
    validate_assimilation(update, c)
    assert update.mechanism_support_arguments == []
    data["portfolio"]["hypotheses"][1]["differential_priority"] = 2
    with pytest.raises(ValueError):
        QuestionAssimilation.model_validate(data)


@pytest.mark.parametrize("damaged_history", [False, True])
def test_discovery_allowance_survives_interrupted_act(tmp_path, damaged_history):
    o, _ = loop(tmp_path, discover=True)
    class InterruptedClient(QuestionClient):
        broken = True
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_question_act" and self.broken:
                if payload["tool_discovery"]["remaining_batches"]:
                    return dict(action="describe_tools", capability_ids=[TOOL])
                raise TimeoutError("interrupted after description queries")
            if payload["stage"] == "planner_question_act" and payload["round_id"] == "R001":
                assert payload["tool_discovery"]["remaining_batches"] == 0
            return super().complete_json(payload=payload, system_prompt=system_prompt, schema_feedback=schema_feedback)
    client = InterruptedClient()
    o.deps.planner.llm_client = client
    with pytest.raises(TimeoutError):
        o.run(smiles="CCO", user_query=NOTE, case_id="query-resume")
    client.broken = False
    if damaged_history:
        history = tmp_path / "query-resume/history/tool_discovery.jsonl"
        with history.open("a") as handle:
            handle.write('{"kind":')
        with pytest.raises(json.JSONDecodeError):
            o.run(smiles="CCO", user_query=NOTE, case_id="query-resume", resume=True)
        return
    result = o.run(smiles="CCO", user_query=NOTE, case_id="query-resume", resume=True)
    assert result.status == "finalized"
