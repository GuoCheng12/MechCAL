"""Memory isolation and scheduling, not scientific performance claims."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest
from jsonschema import Draft202012Validator

from mechcal.runtime import command_exploration_memory as facts, command_exploration_review as er
from mechcal.runtime import command_followup as f, command_hypotheses as h, command_investigation as inv
from mechcal.runtime.command_loop import _apply, _describe_tools, _hypothesis_followup, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, protocol, put, record_index
from mechcal.runtime.command_reads import query_results
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import Finalize, ReadEvidence, ReportExploration, UpdateHypothesesWithInvestigation, computation_model
from test_command_followup import observed, question, enabled
from test_command_protocol import REGISTRY, TOOL, store, LABELS
from test_experiment_options import suggestion
from test_investigation_policy import select, choice
from test_native_commands import NativeFake, reply
from test_exploration_batches import batch_reply
from mechcal.agents.command_tools import parse_tool_turn, tool_contracts


def prepared(tmp_path):
    o = enabled(tmp_path)
    o.config = replace(o.config, command_review_workflow=True, command_computational_review=True,
        command_investigation_policy=True, command_reviewer_exploration=True,
        command_compact_context=True, enable_agenda_coverage_reviewer=True)
    c = observed()
    protocol(c)["results"]["result:new"]["capability_id"] = TOOL
    return er.enable(inv.enable(c)), o


def finish_reply(*, experiments=None, text="PRIVATE_NOTE: compare computable alternatives.", call_id="finish"):
    report = ReportExploration(kind="report_exploration", text=text, experiments=experiments or [])
    return reply("report_exploration", report.model_dump(mode="json", exclude={"kind"}), call_id=call_id)


def turn(c, o, s):
    return er.run_turn(o, c, s, catalog=REGISTRY.compact_catalog(c.artifact_manifest, amesp_enabled=False),
                       describe_tools=lambda command: _describe_tools(o, c, command))


def poison(c):
    marker = "HIDDEN_PLANNER_OPINION_CANARY"
    c = c.model_copy(deep=True)
    c.input.user_query = marker
    c.input.metadata = {"hidden_reference": marker}
    return put(c, note=marker, ranking={LABELS[0]: {"confidence": .99, "rationale": marker}},
        arguments={"private:argument": {"text": marker}},
        explanations={"result:new": {"status": "available", "text": marker}},
        review={"text": marker}, review_memory={"reviewer": {"text": marker}},
        query_receipts={"bad": {"not_a_receipt": marker}}, numeric_checks={"bad": {"note": marker}},
        read_handoffs={"result:new": {"note": marker}}, latest_exchange={"text": marker},
        actions={"action:one": {"dispatch_id": "action:one", "capability_id": TOOL,
                  "task": marker, "expected_new_information": marker, "tool_args": {"seed": 7}}})


def test_projection_whitelist_preserves_facts_not_agent_memory(tmp_path):
    c, _ = prepared(tmp_path)
    c = poison(c)
    p = facts.project(c)
    wire = json.dumps(p.model_dump(mode="json"))
    assert "HIDDEN_PLANNER_OPINION_CANARY" not in wire
    assert protocol(p)["results"]["result:new"]["structured_results"] == {"test": 1}
    assert protocol(p)["actions"]["action:one"]["tool_args"] == {"seed": 7}
    assert protocol(p)["query_receipts"] == {} and protocol(p)["numeric_checks"] == {}
    assert not p.portfolio.hypotheses and not p.evidence_ledger.items
    assert "private:argument" not in record_index(p).available


def test_raw_unit_interpretations_are_not_shared(tmp_path):
    c, _ = prepared(tmp_path)
    result = deepcopy(protocol(c)["results"]["result:new"])
    result["evidence_units"] = [{"evidence_id": "unit:one", "source_report_id": "result:new",
        "metrics": {"energy": 1.2}, "claim": "CANARY", "summary": "CANARY", "support": "CANARY",
        "hypothesis_refs": ["CANARY"], "claim_refs": ["CANARY"], "context": "CANARY"}]
    c = put(c, results={"result:new": result})
    projected = facts.project(c)
    assert "CANARY" not in json.dumps(projected.model_dump(mode="json"))
    assert record_index(projected).available["unit:one"]["metrics"] == {"energy": 1.2}


def test_input_invariant_to_other_agents_judgment_changes(tmp_path):
    c, o = prepared(tmp_path)
    c = poison(c)
    s = store(tmp_path, c)
    client = NativeFake([finish_reply(), finish_reply()])
    o.deps.exploration_agent.llm_client = client
    turn(er.start(c, "first_observations"), o, s)
    changed = put(c, note="OTHER_OPINION", ranking={}, arguments={}, explanations={},
                  review_memory={}, query_receipts={}, numeric_checks={}, read_handoffs={}, latest_exchange={})
    turn(er.start(changed, "first_observations"), o, s)
    a, b = client.requests
    for key in ("system_prompt", "payload", "tools", "exchange", "outcome"):
        assert a[key] == b[key]
    assert "HIDDEN_PLANNER_OPINION_CANARY" not in json.dumps(a["payload"])


@pytest.mark.parametrize("ref", ["private:argument", "followup:state", "context:planner", "explanation:result:new", "foreign:case"])
def test_forbidden_memory_reads_are_denied_even_if_id_is_known(tmp_path, ref):
    c, _ = prepared(tmp_path)
    p = facts.project(poison(c))
    outcome, _ = query_results(p, ReadEvidence(kind="read_evidence", records=[{"ref": ref}]),
                              allowed_refs=record_index(p).available)
    assert outcome["status"] == "rejected"
    assert outcome["issues"][0]["code"] == "out_of_scope"


def test_private_reads_and_notes_survive_resume_without_touching_global_caches(tmp_path):
    c, o = prepared(tmp_path)
    c = poison(c)
    client = NativeFake([reply("read_evidence", {"records": [{"ref": "result:new", "paths": ["/structured_results/test"]}]}),
                        finish_reply(experiments=[suggestion()])])
    o.deps.exploration_agent.llm_client = client
    s = store(tmp_path, c)
    c = er.start(c, "first_observations")
    before = deepcopy(protocol(c))
    c = turn(c, o, s)
    assert er.state(c).query_receipts
    assert protocol(c)["query_receipts"] == before["query_receipts"]
    c = CaseRun.model_validate_json(c.model_dump_json())
    c = turn(c, o, s)
    assert er.state(c).note.startswith("PRIVATE_NOTE")
    assert er.state(c).session is None
    assert protocol(c)["explanations"] == before["explanations"]
    assert protocol(c)["results"] == before["results"]
    assert er.due(c) is None
    qid = next(iter(er.state(c).proposals))
    assert f.state(c).questions[qid].origin == "exploration_reviewer"
    assert f.state(c).questions[qid].trials == []
    for key in ("rounds", "revision", "ranking", "arguments", "exploration"):
        assert protocol(c)[key] == before[key]


def test_planner_sees_proposals_but_not_private_note(tmp_path):
    c, o = prepared(tmp_path)
    o.deps.exploration_agent.llm_client = NativeFake([finish_reply(experiments=[suggestion()])])
    c = turn(er.start(c, "first_observations"), o, store(tmp_path, c))
    material = json.dumps(planner_payload(o, c, native=True))
    assert "PRIVATE_NOTE" not in material and "Can a controlled change" in material
    assert facts.MEMORY_REF not in record_index(c).available


def test_private_note_cannot_be_cited_as_experiment_source(tmp_path):
    c, o = prepared(tmp_path)
    client = NativeFake([finish_reply(experiments=[suggestion(source_refs=[facts.MEMORY_REF])], call_id="bad"),
                        finish_reply(experiments=[suggestion()], call_id="repair")])
    o.deps.exploration_agent.llm_client = client
    s = store(tmp_path, c)
    c = turn(er.start(c, "first_observations"), o, s)
    assert er.state(c).session["corrections"] == 1 and not er.state(c).proposals
    c = turn(c, o, s)
    assert len(er.state(c).proposals) == 1


def test_proposals_and_private_memory_are_not_overwritten_by_planner_edits(tmp_path):
    c, o = prepared(tmp_path)
    o.deps.exploration_agent.llm_client = NativeFake([finish_reply(experiments=[suggestion()])])
    c = turn(er.start(c, "first_observations"), o, store(tmp_path, c))
    qid = next(iter(er.state(c).proposals))
    original = er.state(c).proposals[qid]
    edited = f.state(c).questions[qid].value.model_copy(update={"question": "PLANNER_CHANGED", "status": "deferred"})
    c, _ = h.update(c, UpdateHypothesesWithInvestigation(kind="update_hypotheses", note="Edit only shared question.", questions=[edited]))
    assert er.state(c).proposals[qid] == original
    view = facts.project(c, er.state(c).model_dump(mode="json"))
    assert "PLANNER_CHANGED" not in json.dumps(view.model_dump(mode="json"))
    assert er.due(c) is None


def mark_seen(c):
    m = er.state(c)
    m.seen_versions = facts.versions(facts.project(c))
    m.attempts.append({"status": "available"})
    return er.save_state(c, m)


def test_three_operational_triggers_and_no_per_tool_or_per_round_review(tmp_path):
    c, _ = prepared(tmp_path)
    assert er.due(c) == "first_observations"
    c = mark_seen(c)
    assert er.due(c) is None
    c = put(c, results={**protocol(c)["results"], "result:next": {"status": "failed", "structured_results": {"timeout": True}}})
    assert er.due(c) is None  # A tool result alone is not an experiment boundary.
    c, _ = h.update(c, UpdateHypothesesWithInvestigation(kind="update_hypotheses", note="Route blocked.",
        questions=[question(status="deferred")]))
    c = f.record_trial(c, f.question_snapshot(c, "compare"), capability_id=TOOL, purpose="Private goal.", result_id="result:next")
    assert er.due(c) == "question_boundary"
    c = mark_seen(c)
    c = select(c, decision="conclude")
    assert er.due(c) is None  # Repeated stopping does not repeat the episode.
    c = put(c, results={**protocol(c)["results"], "result:third": {"status": "success"}})
    assert er.due(c) == "before_conclusion"


def test_boundary_does_not_require_a_successful_computation(tmp_path):
    c, _ = prepared(tmp_path)
    c = put(c, results={"failed": {"status": "failed"}})
    assert er.due(c) is None
    c = select(c, decision="conclude")
    assert er.due(c) == "before_conclusion"


def test_unchanged_basis_failure_is_recorded_once_not_a_clearance(tmp_path):
    c, o = prepared(tmp_path)
    client = NativeFake([finish_reply(experiments=[suggestion(source_refs=["bad"])], call_id=str(i)) for i in range(3)])
    o.deps.exploration_agent.llm_client = client
    s = store(tmp_path, c)
    c = er.start(c, "first_observations")
    for _ in range(3):
        c = turn(c, o, s)
    assert er.state(c).attempts[-1]["status"] == "unavailable"
    assert not er.state(c).proposals and er.due(c) is None
    assert protocol(c)["results"] and protocol(c)["phase"] == "choose"


def test_session_case_and_fact_versions_checked(tmp_path):
    c, o = prepared(tmp_path)
    c = er.start(c, "first_observations")
    for edit in ({"case_id": "foreign"}, {"mode": "audit"}):
        bad = c.model_copy(deep=True)
        protocol(bad)["reviewer_exploration"]["session"].update(edit)
        with pytest.raises(RuntimeError, match="identity"):
            turn(bad, o, store(tmp_path, c))
    changed = put(c, results={"result:new": {"status": "failed"}})
    with pytest.raises(RuntimeError, match="sources changed"):
        turn(changed, o, store(tmp_path, c))
    bad = c.model_copy(deep=True)
    protocol(bad)["reviewer_exploration"]["case_id"] = "foreign"
    with pytest.raises(RuntimeError, match="ownership"):
        er.state(bad)


def test_native_exploration_has_no_execution_authority_and_valid_schema(tmp_path):
    c, o = prepared(tmp_path)
    client = NativeFake([finish_reply()])
    o.deps.exploration_agent.llm_client = client
    turn(er.start(c, "first_observations"), o, store(tmp_path, c))
    request = client.requests[0]
    assert request["exchange"] is None and request["outcome"] is None
    assert {t["name"] for t in request["tools"]} == {
        "read_evidence", "reduce_evidence", "compare_evidence", "describe_tools", "report_exploration"}
    for t in request["tools"]:
        Draft202012Validator.check_schema(t["parameters"])
    assert "experiments" in ReportExploration.model_json_schema()["properties"]


def test_config_is_opt_in_and_old_mode_remains_compatible(tmp_path):
    c, o = prepared(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        replace(o.config, command_experiment_options=True, enable_support_auditor=True)
    with pytest.raises(ValueError, match="requires command_investigation_policy"):
        replace(o.config, command_investigation_policy=False)
    assert er.state(observed()) is None and er.due(observed()) is None
    assert not replace(o.config, command_reviewer_exploration=False).command_reviewer_exploration


def test_finalization_yields_for_exploration_without_executing_or_rejecting(tmp_path):
    c, o = prepared(tmp_path)
    c = select(c, decision="conclude")
    out = _apply(o, c, Finalize(kind="finalize", answer="Do not deliver yet."), store(tmp_path, c))
    assert protocol(out)["phase"] == "exploration_review"
    assert protocol(out)["outcome"]["status"] == "exploration_requested"
    assert not out.final_answer and protocol(out)["rounds"] == protocol(c)["rounds"]


def test_combined_choice_and_compute_resumes_with_complete_receipts(tmp_path):
    c, o = prepared(tmp_path)
    update = UpdateHypothesesWithInvestigation(kind="update_hypotheses", note="Select next experiment.", investigation=choice())
    compute = computation_model(REGISTRY.require(TOOL), with_question=True)(
        kind="compute", capability_id=TOOL, arguments={}, purpose="Test a comparison.", question_id=None)
    wire = batch_reply(reply("update_hypotheses", update.model_dump(mode="json", exclude={"kind"}), call_id="u"),
        reply("compute_" + TOOL.replace(".", "_"), compute.model_dump(mode="json", exclude={"kind", "capability_id"}), call_id="c"))
    tools, models = tool_contracts(REGISTRY, [TOOL], hypothesis_tracking=True, review_workflow=True, investigation_policy=True)
    parsed = parse_tool_turn(wire, tools, models)
    s = store(tmp_path, c)
    c = save(_apply(o, put(c, latest_exchange=wire), parsed, s), s)
    c = save(_hypothesis_followup(o, c, s), s)
    assert protocol(c)["phase"] == "exploration_review" and not protocol(c).get("ticket")
    c = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    o.deps.exploration_agent.llm_client = NativeFake([finish_reply()])
    c = save(turn(c, o, s), s)
    assert protocol(c)["outcome"]["status"] == "hypothesis_step_completed"
    assert set(protocol(c)["outcome"]["call_results"]) == {"u", "c"}
    c = _apply(o, c, compute, s)
    assert protocol(c)["phase"] == "execute"


def test_new_options_require_planner_reconsideration_not_automatic_execution(tmp_path):
    c, o = prepared(tmp_path)
    c = select(c)
    compute = computation_model(REGISTRY.require(TOOL), with_question=True)(
        kind="compute", capability_id=TOOL, arguments={}, purpose="Test.", question_id=None)
    s = store(tmp_path, c)
    c = _apply(o, c, compute, s)
    o.deps.exploration_agent.llm_client = NativeFake([finish_reply(experiments=[suggestion()])])
    c = turn(c, o, s)
    assert protocol(c)["phase"] == "choose" and not protocol(c).get("ticket")
    with pytest.raises(OperationRejected, match="No current investigate"):
        _apply(o, c, compute, s)
    qid = next(iter(er.state(c).proposals))
    c = select(c, question_id=qid, considered_question_ids=[qid])
    assert protocol(_apply(o, c, compute, s))["phase"] == "execute"


def test_queries_and_corrections_share_a_bounded_private_episode(tmp_path):
    c, o = prepared(tmp_path)
    client = NativeFake([
        reply("describe_tools", {"capability_ids": [TOOL]}, call_id="details"),
        reply("read_evidence", {"records": [{"ref": "result:new", "paths": ["/status"]}]}, call_id="read"),
        finish_reply(experiments=[suggestion(source_refs=["bad"])], call_id="bad"),
        finish_reply(),
    ])
    o.deps.exploration_agent.llm_client = client
    s = store(tmp_path, c)
    before = deepcopy(protocol(c))
    c = er.start(c, "first_observations")
    for _ in range(4):
        c = turn(c, o, s)
    assert er.state(c).attempts[-1]["status"] == "available"
    assert {t["name"] for t in client.requests[-1]["tools"]} == {"report_exploration"}
    for key in ("loaded_tools", "visible_tools", "rounds", "results", "ranking"):
        assert protocol(c)[key] == before[key]


def test_audit_scope_cannot_access_exploration_private_memory(tmp_path):
    c, o = prepared(tmp_path)
    o.deps.exploration_agent.llm_client = NativeFake([finish_reply(experiments=[suggestion()])])
    c = turn(er.start(c, "first_observations"), o, store(tmp_path, c))
    index = record_index(c)
    assert "PRIVATE_NOTE" not in json.dumps(index.available)
    outcome, _ = query_results(c, ReadEvidence(kind="read_evidence", records=[{"ref": facts.MEMORY_REF}]),
                              allowed_refs=index.available)
    assert outcome["status"] == "rejected"


def test_complete_loop_schedules_exploration_before_planner_and_preserves_resume(tmp_path):
    from mechcal.runtime.command_loop import run_command_loop
    from mechcal.runtime.question_loop import configuration_fingerprint
    c, o = prepared(tmp_path)
    s = store(tmp_path, c)

    class Client:
        settings = None
        def __init__(self):
            self.roles = []

        def complete_tool_turn(self, **kw):
            role = kw.get("role", "planner")
            self.roles.append(role)
            if role == "exploration_reviewer":
                return finish_reply(experiments=[suggestion()])
            assert role == "planner"
            assert "PRIVATE_NOTE" not in json.dumps(kw["payload"])
            assert "Can a controlled change" in json.dumps(kw["payload"])
            (s.root / "STOP_REQUESTED").touch()
            data = UpdateHypothesesWithInvestigation(kind="update_hypotheses", note="Decide independently.",
                investigation=choice()).model_dump(mode="json", exclude={"kind"})
            return reply("update_hypotheses", data, call_id=f"planner{len(self.roles)}")

    client = Client()
    o.deps.planner.llm_client = client
    o.deps.exploration_agent.llm_client = client
    c = put(c, fingerprint=configuration_fingerprint(o, c))
    save(c, s)
    out = run_command_loop(o, c, s, resume=True)
    assert client.roles == ["exploration_reviewer", "planner"]
    assert out.status == "paused" and protocol(out)["rounds"] == 0
    assert len(er.state(out).attempts) == 1
    assert er.due(out) is None
    (s.root / "STOP_REQUESTED").unlink()
    run_command_loop(o, out, s, resume=True)
    assert client.roles == ["exploration_reviewer", "planner", "planner"]


def test_calculation_failure_remains_failure_and_never_becomes_evidence(tmp_path):
    from mechcal.runtime.command_operations import reference_issues
    c, _ = prepared(tmp_path)
    c = put(c, results={"failed": {"tool_result_id": "failed", "status": "failed", "failure": "Timed out."}})
    projected = facts.project(c)
    assert protocol(projected)["results"]["failed"]["status"] == "failed"
    assert reference_issues(projected, ["failed"], ["refs"], scientific=True)[0]["code"] == "failed_evidence"


def test_original_proposals_receive_only_execution_receipts(tmp_path):
    c, o = prepared(tmp_path)
    o.deps.exploration_agent.llm_client = NativeFake([finish_reply(experiments=[suggestion()])])
    c = turn(er.start(c, "first_observations"), o, store(tmp_path, c))
    qid = next(iter(er.state(c).proposals))
    c = f.record_trial(c, f.question_snapshot(c, qid), capability_id=TOOL,
                       purpose="PLANNER_PRIVATE_PURPOSE", result_id="result:new")
    receipts = facts.trial_receipts(c, [qid])
    assert receipts[qid] == [{"result_id": "result:new", "capability_id": TOOL, "status": "success"}]
    assert "PLANNER_PRIVATE" not in json.dumps(receipts)
