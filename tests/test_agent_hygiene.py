"""No case identifier or local path reaches a model; priors stay fixed once results exist; ReAct sees the same costs."""

import json

from mechcal.agent.loop import AgentConfig, AgentRun, PlannerStep
from mechcal.eval.grounded_subjects import ReactSubject
from mechcal.eval.tool_gateway import REDACTED_CASE, ToolGateway, redact
from test_agent_loop import SMILES, ByRole, contrast

CASE = "AIE_DDX_MECH_V4_0047"          # real ids name the benchmark ("AIE"): a hint no subject may see
DIH = "microscopic.list_rotatable_dihedrals"


def _seen(payloads) -> str:
    return json.dumps(payloads, default=str)


def test_redact_replaces_ids_and_paths_in_values_only():
    value = {"/a/b": f"/tmp/x/{CASE}_R001.out", "n": 1, "rows": [CASE, "keep"], CASE: "key kept"}
    out, changed = redact(value, [("/tmp/x", "<run_dir>"), (CASE, REDACTED_CASE)])
    assert out == {"/a/b": f"<run_dir>/{REDACTED_CASE}_R001.out", "n": 1, "rows": [REDACTED_CASE, "keep"], CASE: "key kept"}
    assert changed == 2


def test_gateway_outputs_never_show_the_case_id_or_run_dir(tmp_path):
    g = ToolGateway(case_id=CASE, smiles=SMILES, run_dir=tmp_path / "tools")
    raw = g._read(g.prepared_artifact_id, "")
    assert CASE in json.dumps(raw)                       # the structure file names the case ...
    shown = [g.read(g.prepared_artifact_id, ""), g.call(DIH, {}), g.call_log(), g.describe([DIH])]
    text = _seen(shown)
    assert CASE not in text and str(tmp_path) not in text   # ... but no agent sees it
    assert shown[1]["status"] == "success" and shown[0]["value"]["xyz_angstrom"]
    assert g.store.artifacts["T001"].case_id == CASE          # the evaluator side keeps the real id


def test_agent_payloads_carry_no_case_id_and_priors_stay_fixed(tmp_path):
    priors = [{"label": "RIM_RIR_RIV", "prior": 0.3, "prior_basis": "two aryl rotors", "plausibility": 0.3},
              {"label": "ICT_TICT_CT", "prior": 0.5, "prior_basis": "amine donor, nitrile acceptor", "plausibility": 0.5}]
    planner = [
        {"updates": priors, "action": {"type": "read", "artifact_id": "structure:prepared", "pointer": ""}},
        {"updates": [{"label": "RIM_RIR_RIV", "prior": 0.35}],          # no result yet: a prior may still change
         "action": contrast(type="tool", name=DIH, args={"source_artifact_id": "structure:prepared"})},
        {"updates": [{"label": "RIM_RIR_RIV", "prior": 0.9, "plausibility": 0.6, "reason": "rotors listed"}],
         "action": {"type": "submit"}},
    ]
    client = ByRole(planner)
    cfg = AgentConfig(max_planner_calls=6, max_computations=3, critic=False, worker=False)
    run = AgentRun(client, case_id=CASE, smiles=SMILES, run_dir=tmp_path, config=cfg, enable_amesp=False)
    result = run.run()
    assert result["status"] == "submitted"
    text = _seen(client.payloads)
    assert CASE not in text and str(tmp_path) not in text
    rim = run.board.families["RIM_RIR_RIV"]
    assert rim.prior == 0.35 and rim.plausibility == 0.6                 # after a result: prior kept, plausibility moved
    assert result["protocol"]["prior_changes_refused"] == 1
    assert any("prior of RIM_RIR_RIV kept at 0.35" in h for h in run.history)


def test_model_call_guard_redacts_whatever_slips_through(tmp_path):
    client = ByRole([{"action": {"type": "submit"}}])
    run = AgentRun(client, case_id=CASE, smiles=SMILES, run_dir=tmp_path, enable_amesp=False,
                   config=AgentConfig(critic=False, worker=False))
    run._call_model(client, "planner", {"system_prompt": "planner", "payload": {"history": [f"saw {CASE}"]}}, PlannerStep)
    assert client.payloads["planner"][0]["payload"] == {"history": [f"saw {REDACTED_CASE}"]}
    assert run.protocol["identifiers_redacted"] == 1


def test_react_sees_the_agents_cost_log_and_no_case_id(tmp_path):
    log = tmp_path / "cost.jsonl"
    log.write_text(json.dumps({"tool": DIH, "atoms": 22, "seconds": 3.0, "finished": True, "note": "logged run"}) + "\n")
    seen = {}

    def submit(payload):
        seen["catalog"] = {c["tool"]: c for c in payload["tool_catalog"]}
        mp = [{"label": lab, "rank": i + 1, "evidence": [{"statement": "Donor and acceptor.", "kind": "structural"}],
               "limitations": ["topology only"]} for i, lab in enumerate(["ICT_TICT_CT", "RIM_RIR_RIV", "PET_ET"])]
        return {"action": "submit", "mechanism_predictions": mp}

    class Client:
        def __init__(self):
            self.payloads, self.replies = [], [{"action": "read", "artifact_id": "structure:prepared", "pointer": ""}, submit]

        def complete_json_result(self, *, system_prompt, payload, **_):
            self.payloads.append(payload)
            item = self.replies.pop(0)
            return type("R", (), {"data": item(payload) if callable(item) else item, "parse_error": None,
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})()

    client = Client()
    factory = lambda case_id, smiles: ToolGateway(case_id=case_id, smiles=smiles, run_dir=tmp_path / "tools")
    result = ReactSubject(client, factory, max_model_calls=3, max_tool_calls=2, cost_log_path=str(log)).run(CASE, SMILES)
    assert result.status == "submitted"
    assert "logged run" in json.dumps(seen["catalog"][DIH])
    text = _seen(client.payloads)
    assert CASE not in text and str(tmp_path) not in text
