"""Independent clients, memory ownership and launch wiring; no external calls."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from mechcal.agents.reviewers import ExplorationAgent, ScientificReviewerAgent
from mechcal.cli import launch
from mechcal.runtime.orchestrator import default_deps, mcp_deps
from mechcal.runtime.command_loop import _comment
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.runtime.question_loop import configuration_fingerprint
from mechcal.schemas import CaseRun
from mechcal.tools.mcp_adapter import AieMcpToolSpecs
from test_command_protocol import REGISTRY, case, orchestrator, store
from test_commentary_delivery import prose
from test_native_commands import NativeFake, reply
from test_responses_api import settings
from test_reviewer_exploration import er, finish_reply, poison, prepared, turn


@pytest.mark.parametrize("mcp", [False, True])
def test_factories_create_independent_clients_with_equal_settings(tmp_path, monkeypatch, mcp):
    cfg = orchestrator(tmp_path).config
    monkeypatch.setattr(AieMcpToolSpecs, "discover_capability_registry", lambda *_: REGISTRY)
    deps = mcp_deps(cfg, object()) if mcp else default_deps(cfg)
    assert isinstance(deps.scientific_reviewer, ScientificReviewerAgent)
    assert isinstance(deps.exploration_agent, ExplorationAgent)
    clients = [deps.scientific_reviewer.llm_client, deps.exploration_agent.llm_client,
               deps.photophysics_arbiter.llm_client]
    assert len({id(c) for c in clients}) == 3
    assert clients[0].settings == clients[1].settings == clients[2].settings


def test_cli_configures_both_agents_with_the_requested_provider(tmp_path, monkeypatch):
    cfg = orchestrator(tmp_path).config
    monkeypatch.setattr(launch, "mcp_deps", lambda runtime, _: default_deps(runtime))
    requested = replace(settings(reasoning_effort="high"), model="offline-model")
    o = launch.create_orchestrator(cfg, requested)
    clients = [o.deps.planner.llm_client, o.deps.scientific_reviewer.llm_client,
               o.deps.exploration_agent.llm_client]
    assert len({id(c) for c in clients}) == 3
    assert all(c.settings == requested for c in clients)


@pytest.mark.parametrize("name", ["scientific_reviewer", "exploration_agent"])
def test_fingerprint_tracks_each_agent_and_rejects_missing_dependencies(tmp_path, name):
    o, c = orchestrator(tmp_path), case()
    before = configuration_fingerprint(o, c)
    agent = getattr(o.deps, name)
    agent.llm_client.settings = replace(agent.llm_client.settings, model="different-offline-model")
    assert configuration_fingerprint(o, c) != before
    setattr(o.deps, name, None)
    with pytest.raises(ValueError, match="explicit"):
        configuration_fingerprint(o, c)


@pytest.mark.parametrize("exploring", [False, True])
@pytest.mark.parametrize("mismatch", ["agent", "case", "mode"])
def test_session_identity_rejected_before_model_call(tmp_path, exploring, mismatch):
    c = case()
    client = NativeFake([])
    agent = ExplorationAgent(client) if exploring else ScientificReviewerAgent(client)
    state = {"queries": 0, "case_id": c.case_id, "agent_id": agent.agent_id}
    if exploring:
        state["mode"] = "exploration"
    if mismatch == "agent":
        state["agent_id"] = "scientific_reviewer" if exploring else "exploration_agent"
    elif mismatch == "case":
        state["case_id"] = "foreign-case"
    else:
        state.pop("agent_id")  # Legacy checkpoints still have a scope marker.
        state["mode"] = "scientific" if exploring else "exploration"
    with pytest.raises(RuntimeError, match="session"):
        agent.review(c, {}, store(tmp_path, c), state=state,
                     **({} if exploring else {"role": "reviewer"}))
    assert not client.requests


def test_scientific_agent_rejects_exploration_task(tmp_path):
    c, client = case(), NativeFake([])
    with pytest.raises(ValueError, match="exploration"):
        ScientificReviewerAgent(client).review(c, {}, store(tmp_path, c),
            state={"queries": 0}, role="exploration_reviewer")
    assert not client.requests


def test_exploration_rejects_foreign_memory_before_projection(tmp_path):
    c, o = prepared(tmp_path)
    with pytest.raises(RuntimeError, match="different case"):
        o.deps.exploration_agent.fact_view(c, {"case_id": "foreign-case", "note": "private"})


def test_independent_calls_private_memory_and_checkpoint_continuation(tmp_path):
    c, o = prepared(tmp_path)
    s = store(tmp_path, c)
    scientific = NativeFake([prose("Scientific feedback, not exploration memory.")])
    exploring = NativeFake([
        reply("read_evidence", {"records": [{"ref": "result:new", "paths": ["/structured_results/test"]}]}),
        finish_reply(),
    ])
    o.deps.scientific_reviewer.llm_client = scientific
    o.deps.exploration_agent.llm_client = exploring
    c = turn(er.start(poison(c), "first_observations"), o, s)
    assert len(exploring.requests) == 1 and not scientific.requests
    session = er.state(c).session
    assert session["agent_id"] == "exploration_agent" and session["case_id"] == c.case_id
    c = CaseRun.model_validate_json(c.model_dump_json())
    c = turn(c, o, s)
    private = deepcopy(protocol(c)["reviewer_exploration"])
    # Coverage is a subtask of Scientific Reviewer, never the exploration client.
    c = put(c, phase="review", review_role="coverage_reviewer", ranking={}, arguments={}, review_changes=[],
            read_handoffs={}, query_receipts={}, numeric_checks={}, latest_exchange=None)
    c = _comment(o, c, s)
    assert len(scientific.requests) == 1 and len(exploring.requests) == 2
    assert protocol(c)["reviewer_exploration"] == private
    assert "PRIVATE_NOTE" not in json.dumps(scientific.requests[0]["payload"])
    assert "HIDDEN_PLANNER_OPINION_CANARY" not in json.dumps(exploring.requests[0]["payload"])
    events = [json.loads(line) for line in (s.root / "history/review_agent_calls.jsonl").read_text().splitlines()]
    assert [e["agent_id"] for e in events] == ["exploration_agent", "exploration_agent", "scientific_reviewer"]


def test_legacy_scientific_session_is_bound_without_mutating_caller(tmp_path):
    c = case()
    index = record_index(c)
    client = NativeFake([reply(), prose("The structure is available.")])
    agent = ScientificReviewerAgent(client)
    state = {"queries": 0}
    kwargs = {"scope": {"input:smiles": index.available["input:smiles"]},
              "role": "coverage_reviewer", "prompt_name": "command_reviewer_text.md"}
    _, pending = agent.review(c, {}, store(tmp_path, c), state=state, **kwargs)
    assert state == {"queries": 0}
    assert pending["agent_id"] == agent.agent_id and pending["case_id"] == c.case_id
    feedback, pending = agent.review(c, {}, store(tmp_path, c), state=pending, **kwargs)
    assert feedback["status"] == "available" and pending is None
