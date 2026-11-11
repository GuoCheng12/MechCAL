"""Question hierarchy records scientific scope, not automatic scientific decisions."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mechcal.agents.command_tools import parse_tool_turn, tool_contracts
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime import command_followup as f, command_hypotheses as h
from mechcal.runtime.command_comparison import question_evidence_view, question_source_refs
from mechcal.runtime.command_impact_review import review_material, validate_suggestions
from mechcal.runtime.command_loop import _comment, _catalog
from mechcal.runtime.command_operations import OperationRejected, protocol, put
from mechcal.runtime.command_review_workflow import comparison_options
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import UpdateHypothesesWithWorkflow
from test_command_context_efficiency import expand
from test_command_followup import question, report
from test_command_protocol import REGISTRY
from test_computational_review import fixture, report as final_report
from test_native_commands import NativeFake, reply
from test_reviewer_execution_context import execution_case, OTHER, RESULT, UNIT


def apply(c, *questions):
    return h.update(f.enable(c), UpdateHypothesesWithWorkflow(
        kind="update_hypotheses", note="Offline scope fixture.", questions=list(questions)))[0]


def parent(**kw):
    return question(id="mechanism", scope="mechanism", relation="complementary", **kw)


def child(**kw):
    return question(**{"id": "test", "scope": "test", "parent_id": "mechanism", **kw})


def linked():
    c = apply(execution_case(), parent(), child(contrast={"varied": "state_root",
        "held_fixed": "input geometry", "readout": "state properties", "baseline_refs": [OTHER]}))
    return f.record_trial(c, f.question_snapshot(c, "test"), capability_id="microscopic.fixture",
        purpose="Offline failed trial", dispatch_id="command:failed-attempt", result_id=RESULT)


def test_closed_child_does_not_close_parent_or_update_ranking():
    c = linked()
    before = deepcopy(protocol(c))
    after = apply(c, child(status="resolved", assessment="distinguished"))
    rows = f.question_rows(f.state(after))
    assert [r["id"] for r in rows] == ["mechanism"]
    assert rows[0]["status"] == "open"
    assert rows[0]["subquestions"][0]["assessment"] == "distinguished"
    assert rows[0]["subquestions"][0]["result_ids"] == [RESULT]
    for field in ("rounds", "results", "actions", "ranking", "arguments"):
        assert protocol(after).get(field) == before.get(field)


def test_closed_parent_with_open_descendant_stays_visible_without_rewriting_judgment():
    c = apply(execution_case(), parent(status="resolved", assessment="distinguished"), child())
    rows = f.question_rows(f.state(c))
    assert len(rows) == 2 and rows[0]["unresolved_descendants"]
    assert rows[0]["assessment"] == "distinguished"
    assert f.state(c).questions["mechanism"].value.status == "resolved"


def test_forward_links_and_nested_visibility():
    c = apply(execution_case(), child(parent_id="middle"),
        question(id="middle", parent_id="mechanism", status="resolved", assessment="distinguished"),
        parent(status="resolved", assessment="distinguished"))
    assert {q["id"] for q in f.question_rows(f.state(c))} == {"test", "middle", "mechanism"}


@pytest.mark.parametrize("questions", [
    [child(parent_id="foreign")], [parent(parent_id="mechanism")],
    [parent(parent_id="test"), child()],
    [child(contrast={"baseline_refs": ["foreign:result"]})],
])
def test_invalid_graph_or_baseline_is_atomic_local_error(questions):
    c = linked()
    before = c.model_dump()
    with pytest.raises(OperationRejected):
        apply(c, *questions)
    assert c.model_dump() == before


def test_null_keeps_metadata_and_empty_detaches_across_resume():
    c = linked()
    c = apply(c, question(id="test", status="deferred"))
    restored = CaseRun.model_validate_json(c.model_dump_json())
    value = f.state(restored).questions["test"].value
    assert value.parent_id == "mechanism" and value.scope == "test"
    assert value.contrast.baseline_refs == [OTHER]
    after = apply(restored, question(id="test", parent_id="", contrast={}))
    value = f.state(after).questions["test"].value
    assert value.parent_id is None and value.contrast.baseline_refs == []
    assert f.state(after).questions["test"].trials == f.state(restored).questions["test"].trials


def test_legacy_questions_remain_unknown_and_do_not_add_source_context():
    c = apply(execution_case(), question())
    assert f.state(c).questions["compare"].value.scope is None
    assert question_evidence_view(c)["sources"] == []
    assert f.question_rows(f.state(c))[0]["id"] == "compare"


def test_case_isolation_and_corrupted_saved_graph_are_not_silently_repaired():
    c = linked()
    other = apply(execution_case(), question())
    assert set(f.state(other).questions) == {"compare"}
    broken = f.state(c)
    broken.questions["mechanism"].value.parent_id = "test"
    with pytest.raises(RuntimeError, match="cycle"):
        f.question_rows(broken)
    protocol(c)["evidence_followup"]["case_id"] = "other-case"
    with pytest.raises(RuntimeError, match="ownership"):
        f.state(c)


def test_observed_conditions_and_shared_sources_are_not_inferred_from_plan():
    c = linked()
    results = deepcopy(protocol(c)["results"])
    results[OTHER]["structured_results"]["parsed_observables"].update(
        method="actual-method", computed_geometry_xyz="optimized.xyz")
    c = put(c, results=results)
    before = c.model_dump()
    view = question_evidence_view(c, result_ids=[RESULT])
    assert {q["id"] for q in view["questions"]} == {"test", "mechanism"}
    assert len(view["sources"]) == 2
    sources = {row["result_id"]: row for row in view["sources"]}
    assert sources[OTHER]["observed_conditions"] == {
        "method": "actual-method", "computed_geometry_xyz": "optimized.xyz"}
    assert sources[OTHER]["effective_parameters"]["values"] == {"state_root": 3}
    assert sources[RESULT]["effective_parameters"] is None
    assert sources[RESULT]["status"] == "failed"
    assert sources[RESULT]["observed_conditions"] == {}
    assert question_source_refs(view) == {OTHER, RESULT}
    assert c.model_dump() == before


def test_impact_can_read_baseline_but_cannot_report_it_as_new_evidence():
    c = linked()
    material = review_material(c, [RESULT])
    assert OTHER in question_source_refs(material["question_evidence"])
    assert OTHER not in material["association_source_ids"]
    with pytest.raises(OperationRejected, match="new group"):
        validate_suggestions(c, [RESULT], report(OTHER))
    assert UNIT in material["association_source_ids"]


@pytest.mark.parametrize("count", [2, 100])
def test_nested_method_provenance_is_visible_or_explicitly_deferred(count):
    c = linked()
    results = deepcopy(protocol(c)["results"])
    structured = results[OTHER]["structured_results"]
    structured["parsed_observables"]["molecular_context"] = {"charge": 0, "multiplicity": 1}
    structured["method_provenance"] = [{"step_id": f"step_{i}", "source": "submitted_input",
        "executed_method_label": "fixture-method", "input_sha256": str(i),
        "input_blocks": [{"name": "large_detail", "lines": ["x" * 10000]}]} for i in range(count)]
    c = put(c, results=results)
    source = next(s for s in question_evidence_view(c)["sources"] if s["result_id"] == OTHER)
    assert source["observed_conditions"]["molecular_context"] == {"charge": 0, "multiplicity": 1}
    methods = source["method_steps"]
    assert methods["count"] == count
    assert methods["path"] == "/structured_results/method_provenance"
    if count == 2:
        assert methods["rows"][0]["executed_method_label"] == "fixture-method"
        assert methods["rows"][0]["source"] == "submitted_input"
        assert methods["rows"][1]["path"].endswith("/1")
        assert "input_blocks" not in methods["rows"][0]
    else:
        assert methods["rows"] is None
    assert protocol(c)["results"][OTHER]["structured_results"] == structured


def test_final_review_sees_outside_top3_question_and_can_read_exact_sources(tmp_path):
    c, o, s = fixture(tmp_path)
    outside = next(label for label in MECHANISM_POOL if label not in protocol(c)["ranking"])
    source = apply(linked(), parent(labels=[outside]), child(labels=[outside]))
    c = put(c, results=protocol(source)["results"], actions=protocol(source)["actions"],
            evidence_followup=protocol(source)["evidence_followup"])
    fake = NativeFake([reply("read_evidence", {"records": [{"ref": OTHER,
        "paths": ["/structured_results/parsed_observables"]}]}),
        reply("record_review", final_report().model_dump(exclude={"kind"}), call_id="complete")])
    o.deps.scientific_reviewer.llm_client = fake
    queried = _comment(o, c, s)
    payload = expand(fake.requests[0]["payload"])["sections"]
    assert len(payload["computational_review"]["question_evidence"]["sources"]) == 2
    assert any(outside in q["labels"] for q in payload["computational_review"]["comparisons"])
    assert protocol(queried)["reviewer_session"]["outcome"]["status"] == "read"
    after = _comment(o, queried, s)
    assert protocol(after)["results"] == protocol(c)["results"]
    assert protocol(after)["ranking"] == protocol(c)["ranking"]
    assert len(fake.requests) == 2  # One optional read, the existing review; no extra agent.
    assert comparison_options(c, _catalog(o, c))[0]["scope"] == "mechanism"


@pytest.mark.parametrize("strict", [False, True])
def test_wire_schema_and_parser_agree_on_small_objects(strict):
    command = UpdateHypothesesWithWorkflow(kind="update_hypotheses", note="Offline schema fixture.",
        questions=[parent(), child(contrast={"varied": "torsion", "baseline_refs": ["input:smiles"]})])
    tools, models = tool_contracts(REGISTRY, [], hypothesis_tracking=True, evidence_followup=True,
                                   review_workflow=True, strict_schema=strict)
    schema = next(t["parameters"] for t in tools if t["name"] == "update_hypotheses")
    Draft202012Validator.check_schema(schema)
    data = command.model_dump(mode="json", exclude={"kind"})
    assert not list(Draft202012Validator(schema).iter_errors(data))
    parsed = parse_tool_turn(reply("update_hypotheses", data), tools, models)
    assert parsed.questions[1].contrast.varied == "torsion"
    assert parsed.questions[0].scope == "mechanism"
