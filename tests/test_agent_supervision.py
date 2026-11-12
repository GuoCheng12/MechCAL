"""Professor-student protocol (release 0.3.0), offline: scripted Planner/Worker replies, real gateway."""

import json

import pytest

from mechcal.agent import supervision as sv
from mechcal.agent.loop import AgentConfig, AgentRun
from mechcal.eval.tool_checks import classify_failure, result_warnings
from test_agent_loop import SMILES, ByRole, contrast

DIH = "microscopic.list_rotatable_dihedrals"
TORSION = "microscopic.run_torsion_brightness_coupling_scan"
BUNDLE = "microscopic.run_baseline_bundle"


def worker_tool(name, args, reason="try"):
    return {"thought": reason, "action": {"type": "tool", "name": name, "args": args, "reason": reason}}


def worker_report(status, **extra):
    return {"thought": "report", "action": {"type": "report", "status": status, "summary": "s", **extra}}


def submit(p):
    return {"proposal_responses": [{"proposal_id": x["proposal_id"], "decision": "decline", "reason": "t"}
                                   for x in p["proposals"]], "action": {"type": "submit"}}


def run_agent(tmp_path, planner, worker_replies, **cfg):
    client = ByRole(planner, worker=worker_replies)
    config = AgentConfig(critic=False, max_planner_calls=8, **cfg)
    run = AgentRun(client, case_id="CASE", smiles=SMILES, run_dir=tmp_path, config=config, enable_amesp=False)
    return run, run.run(), client


def latest_reports(client):
    return [p["payload"]["latest"] for p in client.payloads["planner"] if isinstance(p["payload"]["latest"], dict)
            and "task_id" in p["payload"]["latest"]]


def test_system_error_two_retries_then_blocked_technical(tmp_path):
    planner = [{"action": contrast(type="delegate", task="List dihedrals.", tools=[DIH])}, submit]
    worker = [worker_tool(DIH, {"source_artifact_id": "nope-1"}),
              worker_tool(DIH, {"source_artifact_id": "nope-2"}, "other id"),
              worker_tool(DIH, {"source_artifact_id": "nope-3"}, "third id"),
              worker_tool(DIH, {"source_artifact_id": "nope-4"}, "fourth id"),      # beyond the limit: intercepted
              worker_report("done")]                                                   # corrected by the system
    run, result, client = run_agent(tmp_path, planner, worker)
    report = latest_reports(client)[0]
    assert report["status"] == "blocked_technical" and report["system_corrections"]
    assert [a["outcome"] for a in report["attempts"]] == ["precondition_missing"] * 3
    assert report["attempts"][1]["changed"] == {"source_artifact_id": {"from": "nope-1", "to": "nope-2"}}
    assert report["attempts"][1]["reason"] == "other id"
    assert {"code": "retry_limit", "name": DIH} in report["interceptions"]
    assert result["tool_cost"]["tool_calls"] == 3                      # the intercepted call was not executed
    assert result["protocol"]["system_failure_episodes"] == 1 and result["protocol"]["self_repaired"] == 0


def test_identical_retry_is_intercepted_and_self_repair_counts(tmp_path):
    planner = [{"action": contrast(type="delegate", task="List dihedrals.", tools=[DIH])}, submit]
    worker = [worker_tool(DIH, {"source_artifact_id": "nope"}),
              worker_tool(DIH, {"source_artifact_id": "nope"}),                     # no change: intercepted
              worker_tool(DIH, {"source_artifact_id": "structure:prepared"}, "use the prepared structure"),
              worker_report("done")]
    run, result, client = run_agent(tmp_path, planner, worker)
    report = latest_reports(client)[0]
    assert report["status"] == "done" and {"code": "no_change", "name": DIH} in report["interceptions"]
    assert result["protocol"]["self_repaired"] == 1


def test_not_applicable_failure_is_not_retried(tmp_path):
    planner = [{"action": contrast(type="delegate", task="Excited states.", tools=[BUNDLE])}, submit]
    worker = [worker_tool(BUNDLE, {"source_artifact_id": "structure:prepared"}),
              worker_tool(BUNDLE, {"source_artifact_id": "structure:prepared", "s1_nstates": 4}, "fewer states"),
              worker_report("blocked_technical")]
    run, result, client = run_agent(tmp_path, planner, worker)
    report = latest_reports(client)[0]
    assert report["attempts"][0]["failure_class"] == "not_applicable"           # Amesp disabled in this run
    assert {"code": "not_applicable", "name": BUNDLE} in report["interceptions"]
    assert len(report["attempts"]) == 1 and result["tool_cost"]["tool_calls"] == 1


def test_scientific_change_needs_approval_then_runs_after_revise(tmp_path):
    def revise(p):
        report = p["latest"]
        assert report["status"] == "needs_approval" and report["proposed_change"]["args"]["dihedral_id"] == "dih:002"
        return {"action": {"type": "revise", "task_id": "W1", "instruction": "Approved: scan dih:002.", "approve": True}}

    planner = [{"action": contrast(type="delegate", task="Scan dih:001.", tools=[TORSION],
                                   fixed_args={"dihedral_id": "dih:001"})}, revise, submit]

    def after_revise(payload):
        assert payload["attempts"] == [] and payload["assignment"]["planner_instructions"][0]["approved_change"]
        return worker_tool(TORSION, {"source_artifact_id": "structure:prepared", "dihedral_id": "dih:002"}, "approved")

    worker = [worker_tool(TORSION, {"source_artifact_id": "structure:prepared", "dihedral_id": "dih:002"}, "better rotor"),
              worker_report("needs_approval", question_for_planner="Scan dih:002 instead?",
                            proposed_change={"kind": "tool", "name": TORSION, "args": {"dihedral_id": "dih:002"},
                                             "reason": "dih:002 is the aryl rotor"}),
              after_revise, worker_report("blocked_technical")]
    run, result, client = run_agent(tmp_path, planner, worker)
    first, second = latest_reports(client)
    assert first["interceptions"][0]["code"] == "fixed_setting_changed"
    assert result["protocol"]["approvals"] == 1 and result["protocol"]["revisions"] == 1
    assert second["attempts"][0]["args"]["dihedral_id"] == "dih:002"            # executed only after approval
    assert result["tool_cost"]["tool_calls"] == 1


def test_revise_sees_earlier_attempts(tmp_path):
    def revise(p):
        return {"action": {"type": "revise", "task_id": "W1", "instruction": "Use structure:prepared.",
                           "args_hint": {"source_artifact_id": "structure:prepared"}}}

    planner = [{"action": contrast(type="delegate", task="List dihedrals.", tools=[DIH])}, revise, submit]
    seen = {}

    def second_session(payload):
        seen["attempts"] = payload["attempts"]
        seen["instructions"] = payload["assignment"]["planner_instructions"]
        return worker_tool(DIH, {"source_artifact_id": "structure:prepared"}, "as instructed")

    worker = [worker_tool(DIH, {"source_artifact_id": "nope"}), worker_report("blocked_technical"),
              second_session, worker_report("done")]
    run, result, client = run_agent(tmp_path, planner, worker)
    assert seen["attempts"][0]["args"] == {"source_artifact_id": "nope"}
    assert seen["instructions"][0]["instruction"] == "Use structure:prepared."
    assert latest_reports(client)[-1]["status"] == "done"


def test_warnings_reach_observation_and_report(tmp_path, monkeypatch):
    import mechcal.eval.tool_gateway as gw
    monkeypatch.setattr(gw, "result_warnings",
                        lambda tool, status, po: [{"code": "fewer_states_than_requested", "message": "3 of 6", "student_fixable": True}])
    planner = [{"action": contrast(type="delegate", task="List dihedrals.", tools=[DIH])}, submit]
    seen = {}

    def report(payload):
        seen["recent"] = payload["recent_results"][-1]
        return worker_report("done", warnings=["S1/S2 near-degenerate"])

    worker = [worker_tool(DIH, {"source_artifact_id": "structure:prepared"}), report]
    run, result, client = run_agent(tmp_path, planner, worker)
    rep = latest_reports(client)[0]
    assert seen["recent"]["warnings"][0]["code"] == "fewer_states_than_requested"
    assert rep["status"] == "done_with_warnings" and rep["system_corrections"]
    assert {w["code"] for w in rep["warnings"]} == {"fewer_states_than_requested", "student_concern"}


def test_planner_cannot_run_parameterised_tools_or_recipes_with_worker_on(tmp_path):
    planner = [{"action": contrast(type="tool", name=TORSION, args={"source_artifact_id": "structure:prepared"})},
               {"action": contrast(type="recipe", name="excited_states")},
               {"action": contrast(type="tool", name=DIH, args={"source_artifact_id": "structure:prepared"})}, submit]
    run, result, client = run_agent(tmp_path, planner, [])
    assert result["protocol"]["planner_rejected"] == 2
    assert result["tool_cost"]["tool_calls"] == 1                      # only the whitelisted listing ran
    _, result_a, _ = run_agent(tmp_path / "a", [{"action": contrast(type="tool", name=TORSION,
                                                                   args={"source_artifact_id": "structure:prepared"})}, submit],
                               [], worker=False)
    assert result_a["protocol"]["planner_rejected"] == 0 and result_a["tool_cost"]["tool_calls"] == 1


def test_adopted_finding_becomes_board_evidence(tmp_path):
    planner = [{"action": contrast(type="delegate", task="List dihedrals.", tools=[DIH])},
               lambda p: {"adopt": [{"finding_id": p["latest"]["claims"][0]["finding_id"], "label": "RIM_RIR_RIV"}],
                          "action": {"type": "submit"}}]
    worker = [worker_tool(DIH, {"source_artifact_id": "structure:prepared"}),
              worker_report("done", claims=[{"label": "not-a-family", "kind": "prior", "statement": "Rotors exist."}])]
    run, result, client = run_agent(tmp_path, planner, worker)
    claims = [c for c in result["board"]["claims"] if c["label"] == "RIM_RIR_RIV"]
    assert claims and claims[0]["origin"] == "worker" and result["protocol"]["findings_adopted"] == 1


# ---- deterministic rules ------------------------------------------------------------------------

class FakeGateway:
    def __init__(self, parents, left=3600.0):
        self.parents, self.left = parents, left

    def derived_from(self, aid, root):
        while aid:
            if aid == root:
                return True
            aid = self.parents.get(aid)
        return False

    def wall_seconds_left(self):
        return self.left


def task(**kw):
    return sv.Task(task_id="W1", objective="x", recipe=kw.pop("recipe", None), tools=kw.pop("tools", [TORSION]),
                   fixed_args=kw.pop("fixed_args", {}), separates=[], expectation="", created_step=1, **kw)


def check(t, name, args, gateway=None, defaults=None, accepted=None, kind="tool"):
    return sv.check_call(t, kind, name, args, gateway=gateway or FakeGateway({}), defaults=defaults or {},
                         accepted=accepted or set(args) | set(t.fixed_args), recipe_args={})


def test_geometry_lineage_allows_derived_artifacts_only():
    gw = FakeGateway({"bundle:1": "geometry:a", "geometry:a": "structure:initial", "geometry:b": "structure:initial"})
    t = task(tools=["microscopic.analyze_hole_electron_from_bundle"], geometry="geometry:a")
    assert check(t, "microscopic.analyze_hole_electron_from_bundle", {"source_artifact_id": "bundle:1"}, gw)[0] is None
    block, _ = check(t, "microscopic.analyze_hole_electron_from_bundle", {"source_artifact_id": "geometry:b"}, gw)
    assert block["code"] == "geometry_change" and block["needs_approval"]
    block, _ = check(t, "microscopic.analyze_hole_electron_from_bundle", {"source_artifact_id": "structure:initial"}, gw)
    assert block["code"] == "geometry_change"                          # going back to the raw seed is a change too


def test_fixed_settings_are_injected_and_target_state_must_stay_visible():
    t = task(fixed_args={"state_root": 3, "solvents": ["water"]})
    block, args = check(t, BUNDLE, {"s1_nstates": 6}, accepted={"s1_nstates", "state_root", "solvents"})
    assert block is None and args["state_root"] == 3 and args["solvents"] == ["water"]
    block, _ = check(t, BUNDLE, {"s1_nstates": 2}, accepted={"s1_nstates", "state_root"})
    assert block["code"] == "target_state_dropped"


def test_recipe_cannot_silently_ignore_a_fixed_setting():
    t = task(recipe="spin_states", tools=[], fixed_args={"functional": "pbe0"})
    block, _ = sv.check_call(t, "recipe", "spin_states", {}, gateway=FakeGateway({}), defaults={},
                             accepted={"nstates_per_spin", "source_artifact_id"}, recipe_args={})
    assert block["code"] == "recipe_cannot_honor"


def test_pinned_scientific_setting_and_route_change_need_approval():
    t = task(pinned={"functional": "pbe"}, tools=["microscopic.compare_spin_states_pyscf"])
    block, _ = check(t, "microscopic.compare_spin_states_pyscf", {"functional": "pbe0"})
    assert block["code"] == "scientific_change"
    assert check(t, "microscopic.compare_spin_states_pyscf", {"functional": "pbe", "grid_level": 2})[0] is None
    block, _ = check(t, "microscopic.search_dimer_contacts", {})
    assert block["code"] == "route_change" and block["needs_approval"]


def test_timeout_retry_must_fit_in_the_wall_time_left():
    t = task(tools=["microscopic.compare_spin_states_pyscf"])
    t.attempts.append(sv.Attempt(n=1, kind="tool", name="microscopic.compare_spin_states_pyscf",
                                 args={"max_wall_seconds": 900}, outcome="failed", failure_class="system",
                                 failure_code="timeout"))
    block, _ = check(t, "microscopic.compare_spin_states_pyscf", {"max_wall_seconds": 1500}, FakeGateway({}, left=600))
    assert block["code"] == "cannot_fit"
    assert check(t, "microscopic.compare_spin_states_pyscf", {"max_wall_seconds": 1500}, FakeGateway({}, left=2000))[0] is None
    assert check(t, "microscopic.compare_spin_states_pyscf", {"max_wall_seconds": 900, "grid_level": 0},
                 FakeGateway({}, left=600))[0] is None                 # another technical change is allowed


def test_every_tool_argument_is_classified(tmp_path, monkeypatch):
    qc = tmp_path / "qc.json"
    qc.write_text(json.dumps({"enabled": True}))
    monkeypatch.setenv("MECHCAL_QC_CONFIG", str(qc))
    run = AgentRun(ByRole([]), case_id="CASE", smiles=SMILES, run_dir=tmp_path / "run", enable_amesp=False)
    names = {a for c in run.tool_cards for a in c["args"]}
    assert names - set(sv.ARG_CLASS) == set()
    assert {"functional", "basis", "solvents", "state_root", "dihedral_id"} <= {k for k, v in sv.ARG_CLASS.items() if v == "scientific"}


def test_failure_classes_and_result_warnings():
    assert classify_failure("precondition_missing", "precondition_missing",
                            "Neither MMFF94 nor UFF parameters are available for this molecule.")["retryable"] is False
    assert classify_failure("precondition_missing", "precondition_missing",
                            "Central bond must be a discovered non-ring rotatable bond.")["failure_class"] == "system"
    assert classify_failure("failed", "runtime_failed", "Whole-route wall time exceeded")["failure_code"] == "timeout"
    assert classify_failure("unsupported", "unsupported", "Amesp is disabled by the run environment.")["retryable"] is False
    codes = lambda tool, po: {w["code"]: w["student_fixable"] for w in result_warnings(tool, "success", po)}
    assert codes("b", {"requested_parameters": {"s1_nstates": 6}, "state_count": 4}) == {"fewer_states_than_requested": True}
    assert codes("v", {"imaginary_mode_indices_below_minus20": [0]}) == {"imaginary_modes": False}
    assert codes("p", {"stability_check_status": "completed", "scf_internal_stable": True, "scf_external_stable": True,
                       "states": [{"state": "S1", "excitation_ev": 2.0}, {"state": "T1", "excitation_ev": 2.1}]}) \
        == {"inverted_singlet_triplet": False}
    assert codes("t", {"route_records": [{"final_energy_hartree": -1.0}, {"final_energy_hartree": -0.9}]}) \
        == {"scan_energy_jump": True}
    assert codes("ok", {"requested_parameters": {"s1_nstates": 6}, "state_count": 6, "optimization_converged": True}) == {}


def test_call_aliases_resolve_to_the_artifact_the_call_produced(tmp_path):
    from mechcal.eval.tool_gateway import ToolGateway
    g = ToolGateway(case_id="CASE", smiles=SMILES, run_dir=tmp_path)
    g.produced["T007"] = ["geometry:abc"]
    g.parents["bundle:1"] = "geometry:abc"
    assert g.resolve_args({"source_artifact_id": "T007", "x": "T007"}) == {"source_artifact_id": "geometry:abc", "x": "T007"}
    assert g.derived_from("bundle:1", "T007") and g.resolve("structure:prepared") == "structure:prepared"


def test_session_without_report_gets_a_final_report_turn(tmp_path):
    planner = [{"action": contrast(type="delegate", task="List dihedrals.", tools=[DIH])}, submit]
    read = {"thought": "look", "action": {"type": "read", "artifact_id": "T001", "pointer": ""}}
    worker = [worker_tool(DIH, {"source_artifact_id": "structure:prepared"})] + [read] * 5 + [worker_report("done")]
    run, result, client = run_agent(tmp_path, planner, worker)
    assert latest_reports(client)[0]["status"] == "done"
    assert "note" in client.payloads["worker"][-1]["payload"]


def test_observation_keeps_shallow_values_when_capped():
    from mechcal.eval.tool_gateway import by_depth
    items = [("/a/deep/x/y/z", 1), ("/a/0/e", 2), ("/a/deep/x/y/w", 3), ("/a/1/e", 4)]
    assert by_depth(items)[:2] == [("/a/0/e", 2), ("/a/1/e", 4)]


# ---- cost visibility ----------------------------------------------------------------------------

def test_cost_estimates_scale_and_mark_lower_bounds(tmp_path):
    from mechcal.agent.costs import CostModel
    m = CostModel(56)
    pbe = m.estimate("microscopic.compare_spin_states_pyscf", {"functional": "pbe", "grid_level": 1})
    assert pbe["status"] == "estimate" and pbe["seconds"] == 923 and "56 atoms" in pbe["basis"]
    pbe0 = m.estimate("microscopic.compare_spin_states_pyscf", {"functional": "pbe0", "grid_level": 1})
    assert pbe0["status"] == "lower_bound" and pbe0["at_least_seconds"] == 1800
    assert CostModel(28).estimate("microscopic.compare_spin_states_pyscf", {"functional": "pbe"})["seconds"] == round(923 / 8)
    assert m.estimate("microscopic.run_charge_population_panel")["status"] == "unknown"
    log = tmp_path / "cost.jsonl"
    m2 = CostModel(30, log)
    m2.append_log([m2.add("microscopic.run_charge_population_panel", {}, 12.0, "success")])
    assert CostModel(30, log).estimate("microscopic.run_charge_population_panel")["seconds"] == 12


def test_expensive_delegation_returns_a_cost_check_first(tmp_path):
    def confirm(p):
        assert p["latest"]["status"] == "cost_check" and p["latest"]["estimate"]["status"] == "estimate"
        return {"action": contrast(type="delegate", task="Excited states.", recipe="excited_states", confirm_cost=True)}

    planner = [{"action": contrast(type="delegate", task="Excited states.", recipe="excited_states")}, confirm, submit]
    worker = [worker_report("blocked_technical")]
    run, result, client = run_agent(tmp_path, planner, worker, cost_check_fraction=0.0001)
    assert result["protocol"]["cost_checks"] == 1 and result["protocol"]["cost_confirmed"] == 1
    assert result["protocol"]["delegations"] == 1
    cards = {c["recipe"]: c for c in client.payloads["planner"][0]["payload"]["recipes"]}
    assert cards["spin_states"]["cost"]["status"] in {"estimate", "lower_bound", "unknown"}


def test_task_wall_budget_caps_the_gateway_during_the_session(tmp_path):
    seen = {}
    holder = {}

    def look(payload):
        seen["cap"] = holder["run"].gateway.max_wall_seconds
        return worker_report("blocked_technical")

    planner = [{"action": contrast(type="delegate", task="List.", tools=[DIH], wall_budget_seconds=120)},
               {"action": contrast(type="delegate", task="List.", tools=[DIH], wall_budget_seconds=30)}, submit]
    client = ByRole(planner, worker=[look])
    run = AgentRun(client, case_id="CASE", smiles=SMILES, run_dir=tmp_path, enable_amesp=False,
                   config=AgentConfig(critic=False, max_planner_calls=6))
    holder["run"] = run
    result = run.run()
    assert seen["cap"] == pytest.approx(120, abs=1) and run.gateway.max_wall_seconds == 3600
    assert latest_reports(client)[1]["status"] == "blocked_technical"      # 30 s budget: no Worker call at all
    assert len(client.payloads["worker"]) == 1
