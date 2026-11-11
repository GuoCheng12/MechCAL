from copy import deepcopy

import pytest

from mechcal.runtime import execution_inventory as inv, command_review_workflow as workflow
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.runtime.command_review import review_projection
from mechcal.runtime.command_task_context import prepare_planner, acknowledge_planner
from mechcal.runtime.command_compact import expand_text
from test_command_protocol import case
from test_command_context_efficiency import review_material, with_review


TOOL = "microscopic.search_dimer_contacts"


def result(status="success", count=4, tool=TOOL):
    return {"capability_id": tool, "status": status, "round_id": "R011",
            "structured_results": {"parsed_observables": {"candidates": [{"status": "success"}] * count}},
            "failure": {"kind": "none"}}


def prepared():
    return workflow.enable(put(case(), results={"old": result("partial", 1)}))


def test_recovery_preserves_history_and_does_not_sum_unique_poses():
    c = prepared()
    c = put(c, results={**protocol(c)["results"], "new": result()})
    before = c.model_dump(mode="json")
    row = inv.groups(c)[TOOL]
    assert row["statuses"] == {"partial": 1, "success": 1}
    assert row["recent"][0]["returned_members"]["candidates"]["returned"] == 1
    assert row["recent"][1]["returned_members"]["candidates"]["returned"] == 4
    assert c.model_dump(mode="json") == before


def test_failure_after_success_cannot_erase_success():
    c = put(prepared(), results={"good": result(), "bad": result("failed", 0)})
    assert inv.groups(c)[TOOL]["statuses"] == {"success": 1, "failed": 1}


def test_partial_execution_does_not_imply_each_returned_geometry_failed():
    row = result("partial", 1)
    row["structured_results"]["parsed_observables"]["candidates"][0]["geometry_converged"] = True
    counts = inv.result_fact("partial", row)["returned_members"]["candidates"]
    assert counts["statuses"] == {"success": 1}
    assert counts["geometry_converged_counts"] == {"true": 1}


def test_repeated_failures_keep_traceable_earlier_success():
    c = put(prepared(), results={"good": result(), **{f"bad{i}": result("failed", 0) for i in range(4)}})
    row = inv.groups(c)[TOOL]
    assert row["earlier_usable_result"]["result_id"] == "good"
    assert len(row["recent"]) == 3


def test_inventory_is_bounded_and_omissions_explicit():
    c = put(prepared(), results={str(i): result() for i in range(20)})
    row = inv.groups(c)[TOOL]
    assert row["attempts"] == 20 and len(row["recent"]) == 3 and row["omitted_attempts"] == 17


@pytest.mark.parametrize("same_route", [True, False])
def test_new_related_result_invalidates_unchanged_argument_not_unrelated_result(same_route):
    c = prepared()
    payload = review_material()
    payload["arguments"][0].update(refs=["old"], text="Only one modeled geometry was returned. " * 30)
    c, receipt, _ = with_review(c, payload)
    c = put(c, results={**protocol(c)["results"], "new": result(tool=TOOL if same_route else "other")})
    material, _, _ = review_projection(c, payload, "reviewer", index=record_index(c))
    field = "arguments:" + payload["arguments"][0]["id"]
    if same_route:
        assert material["sections"][field] == payload["arguments"][0]
        assert field in material["changed_sections"]
        assert any(field in row["sections"] and TOOL in row["capability_ids"]
                   for row in material["execution_rechecks"]["groups"])
    else:
        assert material["sections"][field]["previous_review_ref"] == receipt["ref"]
        assert not any(field in row["sections"] for row in material.get("execution_rechecks", {}).get("groups", []))


def test_final_limitations_reconsidered_on_new_result_without_rewriting_them():
    c = prepared()
    payload = review_material()
    payload["assembled_answer"] = {"answer": "Only one geometry. " * 100,
                                  "limitations": ["No alternative arrangements examined. " * 30]}
    c, _, _ = with_review(c, payload)
    c = put(c, results={**protocol(c)["results"], "new": result()})
    material, receipt, _ = review_projection(c, payload, "final_reviewer", index=record_index(c))
    assert material["sections"]["assembled_answer:limitations"] == payload["assembled_answer"]["limitations"]
    assert "assembled_answer:limitations" in material["changed_sections"]
    assert receipt["sections"]["assembled_answer:answer"] == payload["assembled_answer"]["answer"]


def test_scoped_issue_binding_changes_only_with_related_executions():
    c = prepared()
    # Use one owned index per binding, as in production.
    i = record_index(c)
    before = workflow.binding("input:smiles", ["old"], i.available, index=i)
    other = put(c, results={**protocol(c)["results"], "unrelated": result(tool="other")})
    i = record_index(other)
    assert workflow.binding("input:smiles", ["old"], i.available, index=i) == before
    c = put(c, results={**protocol(c)["results"], "new": result()})
    i = record_index(c)
    assert workflow.binding("input:smiles", ["old"], i.available, index=i) != before


def test_planner_receives_delta_not_repeated_inventory_and_full_is_readable():
    c = prepared()
    payload = {"stage": "planner_command", "execution_inventory": inv.view(c)}
    c, first = prepare_planner(c, payload)
    assert TOOL in expand_text(first)["execution_inventory"]["by_capability"]
    c = acknowledge_planner(c)
    c, second = prepare_planner(c, payload)
    assert expand_text(second)["execution_inventory"]["by_capability"] == {}
    assert protocol(c)["planner_context"]["fields"]["execution_inventory"] == payload["execution_inventory"]
    c = put(c, results={**protocol(c)["results"], "new": result()})
    c, third = prepare_planner(c, {**payload, "execution_inventory": inv.view(c)})
    assert expand_text(third)["execution_inventory"]["by_capability"][TOOL]["attempts"] == 2


def test_seen_inventory_is_not_reused_across_cases():
    c = prepared()
    c = put(c, planner_inventory_seen={"case_id": "foreign", "versions": {TOOL: inv.groups(c)[TOOL]["version"]}})
    _, payload = prepare_planner(c, {"stage": "planner_command", "execution_inventory": inv.view(c)})
    assert TOOL in expand_text(payload)["execution_inventory"]["by_capability"]


def test_interrupted_dispatch_does_not_drop_unsent_inventory():
    c = prepared()
    payload = {"stage": "planner_command", "execution_inventory": inv.view(c)}
    c, _ = prepare_planner(c, payload)
    c, retry = prepare_planner(c, payload)
    assert TOOL in expand_text(retry)["execution_inventory"]["by_capability"]
    c = acknowledge_planner(c)
    _, next_turn = prepare_planner(c, payload)
    assert expand_text(next_turn)["execution_inventory"]["by_capability"] == {}


def test_untried_disabled_and_failed_are_distinct():
    c = put(prepared(), results={"bad": result("failed", 0)})
    catalog = [{"capability_id": TOOL, "status": "ready"},
               {"capability_id": "motion", "status": "ready"},
               {"capability_id": "disabled", "status": "disabled"},
               {"capability_id": "needs_structure", "status": "missing_input", "missing_inputs": ["prepared_structure"]}]
    options = inv.opportunities(c, catalog)
    assert [r["capability_id"] for r in options] == ["motion", "needs_structure"]
    assert inv.route_history(c, TOOL)["statuses"] == {"failed": 1}
    assert inv.route_history(c, "motion")["attempts"] == 0


def test_claims_and_ranking_not_changed_by_status_summary():
    c = prepared()
    before = deepcopy(c.model_dump(mode="json"))
    inv.view(c)
    inv.opportunities(c, [{"capability_id": "untried", "status": "ready"}])
    assert c.model_dump(mode="json") == before
