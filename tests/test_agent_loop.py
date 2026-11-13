"""MechCAL agent loop (release 0.3.0) offline: scripted model replies, real RDKit-backed tool."""

import json
from dataclasses import dataclass, field

import pytest

from mechcal.agent.board import Board, ClaimIn
from mechcal.agent.facts import structural_facts
from mechcal.agent.loop import AgentConfig, AgentRun, PlannerStep
from mechcal.eval.grounding import check_submission

SMILES = "CN(C)c1ccc(cc1)C#N"
DIH_TOOL = "microscopic.list_rotatable_dihedrals"


@dataclass
class Reply:
    data: dict
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100})
    parse_error: str | None = None


class ByRole:
    """Scripted client: picks the next reply for the role inferred from the system prompt."""

    def __init__(self, planner, critic=(), worker=(), prior=()):
        self.queues = {"planner": list(planner), "critic": list(critic), "worker": list(worker), "prior": list(prior)}
        self.payloads = {"planner": [], "critic": [], "worker": [], "prior": []}

    def complete_json_result(self, *, system_prompt, payload, schema_feedback=None, **_):
        role = ("critic" if "You are the Critic" in system_prompt else "worker" if "You are the Worker" in system_prompt
                else "prior" if "You are the prior elicitor" in system_prompt else "planner")
        self.payloads[role].append({"payload": payload, "feedback": schema_feedback})
        item = self.queues[role].pop(0)
        return Reply(item(payload) if callable(item) else item)


def contrast(**action):
    return {"separates": ["RIM_RIR_RIV", "ICT_TICT_CT"], "expectation": "rotors favour RIM; D-A path favours ICT", **action}


def test_facts_and_claim_conversion():
    facts = structural_facts(SMILES)
    rotor = next(f for f in facts["facts"] if f["kind"] == "conjugated_rotor")
    claim = ClaimIn(label="RIM_RIR_RIV", statement="The NMe2-aryl bond rotates.", kind="structural",
                    fact_ids=[rotor["fact_id"]]).to_claim(facts)
    assert claim.assertions and claim.atom_ids == rotor["atom_ids"]
    board = Board.initial("C", SMILES)
    board.families["ICT_TICT_CT"].plausibility = 0.8
    board.add_claim("RIM_RIR_RIV", claim, step=1)
    sub = board.to_submission()
    assert sub.mechanism_predictions[0].label == "ICT_TICT_CT" and len(sub.mechanism_predictions) == 3
    assert check_submission(sub, __import__("mechcal.eval.submission", fromlist=["ArtifactStore"]).ArtifactStore(case_id="C"),
                            SMILES).predictions


def test_claim_ids_are_not_reused_after_drop():
    board = Board.initial("C", SMILES)
    claim = ClaimIn(label="PET_ET", statement="x", kind="prior").to_claim({"facts": []})
    first = board.add_claim("PET_ET", claim, step=1)
    second = board.add_claim("PET_ET", claim, step=1)
    board.drop_claims([first])
    third = board.add_claim("PET_ET", claim, step=2)
    assert len({first, second, third}) == 3 and [c.claim_id for c in board.claims] == [second, third]


def test_recipe_cards_carry_tool_argument_limits(tmp_path):
    from mechcal.agent.recipes import RECIPE_ARGS, RECIPES
    run = AgentRun(ByRole([]), case_id="CASE", smiles=SMILES, run_dir=tmp_path, enable_amesp=False)
    tools = {c["tool"]: c["args"] for c in run.tool_cards}
    for recipe, mapping in RECIPE_ARGS.items():
        assert recipe in RECIPES
        for tool, arg in mapping.values():
            if tool in tools:                       # external-QC tools register only with MECHCAL_QC_CONFIG
                assert arg in tools[tool], (recipe, tool, arg)
    cards = {c["recipe"]: c for c in run._recipe_cards()}
    assert "acetonitrile" in cards["solvent_polarity"]["args"]["solvents"]["items_enum"]
    assert "maximum" in cards["excited_states"]["args"]["nstates"]


def test_action_requires_contrast():
    with pytest.raises(Exception):
        PlannerStep.model_validate({"action": {"type": "tool", "name": DIH_TOOL, "separates": ["RIM_RIR_RIV"], "expectation": "x"}})


def test_full_loop_offline(tmp_path):
    facts = structural_facts(SMILES)
    rotor = next(f for f in facts["facts"] if f["kind"] == "conjugated_rotor")["fact_id"]
    da = next(f for f in facts["facts"] if f["kind"] == "donor_acceptor_path")["fact_id"]

    def step2(payload):
        latest = payload["latest"]
        pointer = next(p for p in latest["values"] if p.endswith("/atom_indices/0"))
        value = latest["values"][pointer]
        return {"reading": "Dihedral inventory returned.",
                "updates": [{"label": "RIM_RIR_RIV", "plausibility": 0.6, "status": "mixed"}],
                "claims": [
                    {"label": "RIM_RIR_RIV", "kind": "computed", "statement": f"The first dihedral starts at atom index {value:g}.",
                     "quantities": [{"name": "first_atom", "value": value, "artifact_id": latest["artifact_id"], "pointer": pointer}]},
                    {"label": "ICT_TICT_CT", "kind": "computed", "statement": "S1 oscillator strength is 0.91.",
                     "quantities": [{"name": "f", "value": 0.91, "artifact_id": "T999", "pointer": "/x"}]}],
                "action": contrast(type="delegate", task="List dihedrals again and report.", tools=[DIH_TOOL])}

    planner = [
        {"oops": True},                                   # invalid -> repaired
        {"updates": [{"label": "ICT_TICT_CT", "plausibility": 0.7, "reason": "D-A path"}],
         "claims": [{"label": "ICT_TICT_CT", "kind": "structural", "statement": "Amine donor conjugated to nitrile.", "fact_ids": [da]},
                    {"label": "RIM_RIR_RIV", "kind": "structural", "statement": "Rotor next to the ring.", "fact_ids": [rotor]}],
         "action": contrast(type="tool", name=DIH_TOOL, args={"source_artifact_id": "structure:prepared"})},
        step2,
        lambda p: {"adopt": [f["finding_id"] for f in p["latest"]["claims"]],      # professor adopts the finding
                   "proposal_responses": [{"proposal_id": x["proposal_id"], "decision": "decline", "reason": "test"}
                                          for x in p["proposals"]], "action": {"type": "submit"}},
        lambda p: {"proposal_responses": [{"proposal_id": x["proposal_id"], "decision": "decline", "reason": "test"}
                                          for x in p["proposals"]], "action": {"type": "submit"}},
    ]
    critic = [{"ranking": [{"label": "PET_ET", "plausibility": 0.9, "basis": "amine donor"},
                           {"label": "ICT_TICT_CT", "plausibility": 0.8}, {"label": "RIM_RIR_RIV", "plausibility": 0.5}],
               "tests": [{"label": "PET_ET", "separates": ["PET_ET", "ICT_TICT_CT"], "recipe_or_tool": "redox_levels"}]}] * 3
    worker = [{"thought": "run", "action": {"type": "tool", "name": DIH_TOOL, "args": {"source_artifact_id": "structure:prepared"}}},
              lambda p: {"thought": "done", "action": {"type": "report", "status": "done", "summary": "listed", "claims": [
                  {"label": "RIM_RIR_RIV", "kind": "prior", "statement": "Rotors can dissipate energy."}]}}]
    client = ByRole(planner, critic, worker)
    cfg = AgentConfig(max_planner_calls=8, max_computations=5, critic_every=2, worker_max_calls=3)
    run = AgentRun(client, case_id="CASE", smiles=SMILES, run_dir=tmp_path, config=cfg, enable_amesp=False)
    result = run.run()

    assert result["status"] == "submitted" and result["stop_reason"] == "planner_submitted"
    assert result["tool_cost"]["tool_calls"] == 1                       # worker repeat is not re-run
    assert result["tool_cost"]["rejected_calls"] == 1
    assert client.payloads["planner"][1]["feedback"]                    # repair feedback after invalid output
    assert client.payloads["critic"] and "board" not in client.payloads["critic"][0]["payload"]  # blind critic
    labels = [p["label"] for p in result["submission"]["mechanism_predictions"]]
    assert labels[0] == "ICT_TICT_CT"
    ict = result["submission"]["mechanism_predictions"][0]["evidence"]
    assert all("0.91" not in c["statement"] for c in ict)              # fabricated claim removed by self-check
    assert result["self_check_removed"]
    rim = next(p for p in result["submission"]["mechanism_predictions"] if p["label"] == "RIM_RIR_RIV")
    assert any(c["kind"] == "computed" for c in rim["evidence"])       # grounded claim kept
    assert any(c["kind"] == "prior" for c in rim["evidence"])          # worker claim recorded
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e["role"] == "proposal_response" for e in events)
    assert (tmp_path / "result.json").exists() and (tmp_path / "tools/rounds").is_dir()
    usage = result["usage"]["by_role"]
    assert usage["planner"]["calls"] >= 4 and usage["critic"]["calls"] >= 1 and usage["worker"]["calls"] == 2


def test_budget_forces_submission(tmp_path):
    args = iter([{}, {"source_artifact_id": "structure:prepared"}] * 5)   # distinct requests (not duplicates)
    always_tool = lambda p: {"action": contrast(type="tool", name=DIH_TOOL, args=next(args))}
    client = ByRole([always_tool] * 10)
    cfg = AgentConfig(max_planner_calls=10, max_computations=2, critic=False, worker=False)
    result = AgentRun(client, case_id="CASE", smiles=SMILES, run_dir=tmp_path, config=cfg, enable_amesp=False).run()
    assert result["status"] == "submitted" and result["stop_reason"] == "budget"
    assert result["tool_cost"]["tool_calls"] == 2


def test_planner_aliases_seen_in_the_first_0003_smoke_are_normalized():
    raw = {
        "claims": [{"family": "ICT_TICT_CT", "type": "structural", "text": "D-A path.", "id": "x"}],
        "limitations": {"ICT_TICT_CT": "vertical only", "RIM_RIR_RIV": ["topology only", "no scan"]},
        "updates": [{"family": "RIM_RIR_RIV", "plausibility": 0.3}],
        "action": {"kind": "tool", "tool": DIH_TOOL, "args": {}, "separates": ["RIM_RIR_RIV", "ICT_TICT_CT"],
                   "expectation": {"RIM_RIR_RIV": "rotors", "ICT_TICT_CT": "path"}},
    }
    step = PlannerStep.model_validate(raw)
    assert step.claims[0].label == "ICT_TICT_CT" and step.claims[0].kind == "structural"
    assert step.claims[0].statement == "D-A path." and len(step.limitations) == 3
    assert step.action.type == "tool" and step.action.name == DIH_TOOL
    recipe = PlannerStep.model_validate({"action": {"type": "recipe", "recipe": "excited_states",
                                                    "separates": ["ICT_TICT_CT", "PET_ET"], "expectation": "x"}})
    assert recipe.action.name == "excited_states"
    nested = PlannerStep.model_validate({"claims": [{"label": "PET_ET", "kind": "computed", "statement": "IP is 8.3 eV.",
        "quantities": [{"name": "ip", "value": 8.3, "unit": "eV", "source": {"artifact_id": "T001", "pointer": "/a"}}]}],
        "action": {"type": "submit"}})
    assert nested.claims[0].to_claim({"facts": []}).quantities[0].source.artifact_id == "T001"
