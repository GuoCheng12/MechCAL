"""Chaos test of the agent loop: garbage model replies and injected API failures must never end a run.

Each run gets random replies (wrong types, unknown tools/families, malformed JSON) and random client
exceptions (network, timeout, server). The run must finish, write result.json, and hit no internal error.
"""

import json
import random
from pathlib import Path

import pytest

from mechcal.agent.loop import AgentConfig, AgentRun
from mechcal.mechanism_pool import MECHANISM_POOL
from test_agent_loop import Reply

SMILES = ["CN(C)c1ccc(cc1)C#N", "Oc1ccccc1-c1nc2ccccc2s1", "CCCCCCCCCCOc1ccc([N+]#[C-]->[Au]c2c(F)c(F)c(F)c(F)c2F)cc1"]
TOOLS = ["microscopic.list_rotatable_dihedrals", "microscopic.run_torsion_brightness_coupling_scan",
         "microscopic.compare_spin_states_pyscf", "microscopic.analyze_hole_electron_from_bundle", "nope.tool"]
RECIPES = ["excited_states", "spin_states", "torsion_response", "charge_transfer", "bogus"]
FAM = list(MECHANISM_POOL) + ["NOT_A_FAMILY"]


def junk(r, depth=0):
    c = r.random()
    if depth > 2 or c < 0.25:
        return r.choice([None, 0, -1, 3.7, "x", "", "1200", "abc", True, [], {}, [1, 2], "T001", "structure:prepared"])
    if c < 0.6:
        return {r.choice(["a", "args", "value", "pointer", "label", "type"]): junk(r, depth + 1) for _ in range(r.randint(0, 3))}
    return [junk(r, depth + 1) for _ in range(r.randint(0, 3))]


def args(r):
    a = {}
    for k in r.sample(["source_artifact_id", "dihedral_id", "dihedral_atoms", "angles", "s1_nstates", "max_wall_seconds",
                       "functional", "state_root", "fragments", "fragment_atom_indices", "nstates", "solvents"], r.randint(0, 4)):
        a[k] = r.choice([junk(r), "structure:prepared", "T001", 3, "pbe0", [0, 1, 2, 3], [[0, 1], [2]], "1200", -5, 99999])
    return a


def planner_reply(r, payload):
    kind = r.choice(["tool", "recipe", "delegate", "revise", "read", "submit", "garbage", "nothing"])
    if kind == "garbage":
        return junk(r)
    fams = r.sample(FAM, 2)
    action = {"type": kind, "name": r.choice(TOOLS + RECIPES), "args": args(r), "separates": fams,
              "expectation": r.choice(["x", {fams[0]: "y"}, junk(r)]), "task": r.choice(["do it", "", junk(r)]),
              "recipe": r.choice(RECIPES + [None]), "tools": r.choice([[r.choice(TOOLS)], junk(r), []]),
              "fixed_args": r.choice([args(r), junk(r)]), "task_id": r.choice(["W1", "W2", "W9", junk(r)]),
              "instruction": r.choice(["go", ""]), "approve": r.choice([True, False, "yes"]),
              "artifact_id": r.choice(["T001", "structure:prepared", "nope", junk(r)]), "pointer": r.choice(["", "/x", junk(r)]),
              "confirm_cost": r.choice([True, False]), "wall_budget_seconds": r.choice([None, 30, 500, "abc", -1])}
    claims = [{"label": r.choice(FAM), "kind": r.choice(["computed", "structural", "prior", "weird"]),
               "statement": r.choice(["S1 is 3.2 eV.", "x", junk(r)]), "fact_ids": r.choice([["F1"], junk(r)]),
               "quantities": r.choice([[{"name": "e", "value": r.choice([3.2, "abc", None]), "artifact_id": "T001", "pointer": "/x"}], junk(r)])}
              for _ in range(r.randint(0, 2))]
    adopt = [r.choice(["W1.f1.1", {"finding_id": "W1.f1.1", "label": r.choice(FAM)}, junk(r)])] if r.random() < 0.3 else []
    return {"reading": "r", "updates": [{"label": r.choice(FAM), "plausibility": r.choice([0.5, 2.0, "x"])}],
            "claims": claims, "adopt": adopt, "drop_claims": r.choice([["K1"], junk(r)]),
            "limitations": r.choice([[{"label": r.choice(FAM), "text": "t"}], junk(r)]),
            "proposal_responses": [{"proposal_id": p.get("proposal_id"), "decision": r.choice(["accept", "bad"]), "reason": "r"}
                                   for p in payload.get("proposals", [])], "action": action}


def worker_reply(r, payload):
    kind = r.choice(["tool", "recipe", "read", "report", "report", "finish", "garbage"])
    if kind == "garbage":
        return junk(r)
    return {"thought": "t", "action": {"type": kind, "name": r.choice(TOOLS + RECIPES), "args": args(r), "reason": "r",
                                       "artifact_id": r.choice(["T001", "structure:prepared", junk(r)]),
                                       "pointer": r.choice(["", "/kind", junk(r)]),
                                       "status": r.choice(["done", "needs_approval", "blocked_technical", "weird", None]),
                                       "claims": [{"label": r.choice(FAM + [None]), "kind": "computed", "statement": "x",
                                                   "quantities": r.choice([[], junk(r)])}],
                                       "warnings": r.choice([["w"], junk(r)]),
                                       "proposed_change": r.choice([None, {"kind": "tool", "name": r.choice(TOOLS), "args": args(r)}, junk(r)])}}


class Chaos:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def complete_json_result(self, *, system_prompt, payload, **kw):
        c = self.r.random()
        if c < 0.05:
            raise ConnectionError("injected network failure")
        if c < 0.08:
            raise TimeoutError("injected timeout")
        if c < 0.10:
            raise RuntimeError("injected HTTP 500")
        if c < 0.18:
            rep = Reply({}); rep.parse_error = "ValueError: injected malformed JSON"; return rep
        if "You are the Critic" in system_prompt:
            data = self.r.choice([{"ranking": [{"label": self.r.choice(FAM), "plausibility": self.r.random()}]}, junk(self.r)])
        elif "You are the Worker" in system_prompt:
            data = worker_reply(self.r, payload)
        else:
            data = planner_reply(self.r, payload)
        return Reply(data, usage=self.r.choice([{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, {}, None]))




@pytest.mark.parametrize("seed", range(12))
def test_agent_run_survives_chaos(tmp_path, seed):
    cfg = AgentConfig(max_planner_calls=8, max_computations=5, critic_every=2, worker_max_calls=3, api_backoff_seconds=0)
    result = AgentRun(Chaos(seed), case_id="FUZZ", smiles=SMILES[seed % 3], run_dir=tmp_path, config=cfg,
                      enable_amesp=False).run()
    assert (tmp_path / "result.json").exists() and result["status"] in {"submitted", "stopped"}
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert not [e for e in events if e.get("role") == "internal_error"], "an exception reached the safety net"


def test_api_failures_are_retried_then_treated_as_invalid(tmp_path):
    class Down:
        calls = 0

        def complete_json_result(self, **_):
            Down.calls += 1
            raise ConnectionError("down")

    cfg = AgentConfig(max_planner_calls=3, critic=False, api_retries=2, api_backoff_seconds=0)
    result = AgentRun(Down(), case_id="C", smiles=SMILES[0], run_dir=tmp_path, config=cfg, enable_amesp=False).run()
    assert result["stop_reason"] == "planner_call_budget_exhausted" and (tmp_path / "result.json").exists()
    assert result["protocol"]["api_failures"] >= 1 and Down.calls == 3 * result["protocol"]["api_failures"]
