"""Resumed agents and execution must see the same current calculation limits."""

from copy import deepcopy
from dataclasses import replace

import pytest

from mechcal.runtime.command_operations import protocol, put
from mechcal.runtime.question_loop import configuration_fingerprint
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import envelope_model
from test_command_handoff import unchanged_science
from test_command_protocol import REGISTRY, TOOL, ScriptClient, command, orchestrator, wire


@pytest.mark.parametrize("phase", ["choose", "review"])
@pytest.mark.parametrize("previous_limit", [1, 3, None])
@pytest.mark.parametrize("overrides,allowed,maximum", [
    ({"max_exploration_rounds": 3}, True, 4),
    ({"max_exploration_rounds": 1}, False, 2),
    ({"max_exploration_rounds": None, "unlimited_exploration": True}, True, None),
])
def test_explicit_resume_refreshes_limits_without_replaying_science(
        tmp_path, phase, previous_limit, overrides, allowed, maximum):
    o = orchestrator(tmp_path)
    o.config = replace(o.config, max_exploration_rounds=previous_limit,
                       unlimited_exploration=previous_limit is None)
    empty = envelope_model(REGISTRY, [])
    loaded = envelope_model(REGISTRY, [TOOL])
    o.deps.planner.llm_client = ScriptClient([
        wire(empty, {"command": {"kind": "describe_tools", "capability_ids": [TOOL]}}),
        wire(loaded, {"command": command().model_dump(mode="json")}),
        ConnectionError("pause after one calculation"),
    ])
    with pytest.raises(ConnectionError):
        o.run(smiles="CCO", user_query="Offline limit fixture", case_id="resume-limits")
    path = tmp_path / "resume-limits/command_checkpoint.json"
    saved = CaseRun.model_validate_json(path.read_text())
    assert protocol(saved)["rounds"] == 1
    assert saved.runtime["run_limits"]["max_exploration_rounds"] == previous_limit
    o.config = replace(o.config, **{"unlimited_exploration": False, **overrides})
    client = ScriptClient([ConnectionError("inspect refreshed input")])
    o.deps.planner.llm_client = client
    o.deps.scientific_reviewer.llm_client = client
    # This is the explicit configuration-fingerprint migration used by the
    # copy-on-resume launcher; it must not alter any accepted scientific state.
    saved = put(saved, phase=phase, review_role="reviewer", review_changes=[])
    saved.runtime["command_protocol"]["fingerprint"] = configuration_fingerprint(o, saved)
    path.write_text(saved.model_dump_json())
    before = deepcopy(saved)
    with pytest.raises(ConnectionError, match="inspect refreshed input"):
        o.run(smiles="CCO", user_query="Offline limit fixture", case_id="resume-limits", resume=True)
    after = CaseRun.model_validate_json(path.read_text())
    limits = after.runtime["run_limits"]
    assert limits["scientific_tools_allowed"] is allowed
    assert limits["exploration_rounds_used"] == 1
    assert limits["max_rounds"] == maximum
    assert limits["max_exploration_rounds"] == o.config.max_exploration_rounds
    assert limits["unlimited_exploration"] == o.config.unlimited_exploration
    payload = client.calls[0]["payload"]
    if phase == "review":
        assert payload["run_limits"] == limits
    else:
        assert payload["tools_allowed"] is allowed
    unchanged_science(before, after)
