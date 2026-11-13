"""Independent priors: tool-free samples averaged and fixed; the Planner sets them only when elicitation fails."""

import json

from mechcal.agent.loop import AgentConfig, AgentRun
from mechcal.mechanism_pool import MECHANISM_POOL
from test_agent_loop import SMILES, ByRole, contrast

DIH = "microscopic.list_rotatable_dihedrals"


def sample(**override):
    return {"priors": [{"label": lab, "prior": override.get(lab, 0.1), "basis": f"basis {override.get(lab, 0.1)}"}
                       for lab in MECHANISM_POOL]}


def test_priors_are_averaged_fixed_and_recorded(tmp_path):
    priors = [sample(ICT_TICT_CT=0.6, RIM_RIR_RIV=0.3), sample(ICT_TICT_CT=0.8, RIM_RIR_RIV=0.5),
              sample(ICT_TICT_CT=0.7, RIM_RIR_RIV=0.1)]
    planner = [
        lambda p: {"updates": [{"label": "RIM_RIR_RIV", "prior": 0.9, "plausibility": 0.35}],      # prior refused
                   "action": contrast(type="tool", name=DIH, args={"source_artifact_id": "structure:prepared"})},
        {"updates": [{"label": "ICT_TICT_CT", "plausibility": 0.75, "reason": "D-A path"}], "action": {"type": "submit"}},
    ]
    client = ByRole(planner, prior=priors)
    cfg = AgentConfig(max_planner_calls=5, max_computations=2, critic=False, worker=False, prior_samples=3)
    run = AgentRun(client, case_id="CASE", smiles=SMILES, run_dir=tmp_path, config=cfg, enable_amesp=False)
    result = run.run()
    ict, rim = run.board.families["ICT_TICT_CT"], run.board.families["RIM_RIR_RIV"]
    assert ict.prior == 0.7 and ict.prior_basis == "basis 0.7"                 # mean; basis of the closest sample
    assert rim.prior == 0.3 and rim.plausibility == 0.35                       # Planner moved plausibility, not prior
    assert result["protocol"]["prior_changes_refused"] == 1
    assert len(result["prior_samples"]) == 3 and result["usage"]["by_role"]["prior"]["calls"] == 3
    first = client.payloads["planner"][0]["payload"]["board"]["families"]
    assert next(f for f in first if f["label"] == "ICT_TICT_CT")["prior"] == 0.7   # the Planner starts from them
    prior_payload = client.payloads["prior"][0]["payload"]
    assert set(prior_payload) == {"verified_smiles", "mechanism_pool", "structural_facts"}   # no tools, no board
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    summary = next(e for e in events if e["role"] == "prior_summary")
    assert summary["samples"] == 3 and summary["spread"]["RIM_RIR_RIV"] == 0.4


def test_failed_elicitation_leaves_priors_to_the_planner(tmp_path):
    incomplete = {"priors": [{"label": "ICT_TICT_CT", "prior": 0.5}]}
    planner = [{"updates": [{"label": lab, "prior": 0.2, "prior_basis": "x"} for lab in MECHANISM_POOL],
                "action": {"type": "submit"}}]
    client = ByRole(planner, prior=[incomplete] * 2)
    cfg = AgentConfig(max_planner_calls=3, critic=False, worker=False, prior_samples=2)
    run = AgentRun(client, case_id="CASE", smiles=SMILES, run_dir=tmp_path, config=cfg, enable_amesp=False)
    result = run.run()
    assert result["prior_samples"] == [] and not run.priors_fixed
    assert all(f.prior == 0.2 for f in run.board.families.values())
    assert any("prior elicitation failed" in h for h in run.history)


def test_independent_priors_are_off_by_default_and_can_be_switched_on(tmp_path, monkeypatch):
    from mechcal.agent import cli
    assert cli.RUN_DEFAULTS["prior_samples"] == 0 and AgentConfig().prior_samples == 0
    seen = {}
    monkeypatch.setattr(cli, "backbone_settings", lambda: (_ for _ in ()).throw(SystemExit("stop after parsing")))
    monkeypatch.setattr(cli, "AgentConfig", lambda **kw: seen.update(kw) or AgentConfig(**kw))
    import pytest
    with pytest.raises(SystemExit, match="stop after parsing"):
        cli.main(["--case-id", "AIE_DDX_MECH_V4_0047", "--out", str(tmp_path / "x"), "--prior-samples", "3"])
    assert seen["prior_samples"] == 3
