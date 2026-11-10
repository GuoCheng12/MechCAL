"""Discovery control-flow fixtures do not stand in for scientific evaluations."""

import json

import pytest

from mechcal.agents.mechanism_agenda import _coverage_payload, _sanitize_coverage
from mechcal.agents.planner import PlannerAgent
from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.capabilities.discovery import ToolDiscoverySession, case_catalog
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.gates import DeterministicGate
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient, _response_action
from mechcal.runtime.persistence import RunStore
from mechcal.schemas import ArtifactRecord, PlannerDecision
from mechcal.tools import StdioMcpTransport
from mechcal.tools.mcp_adapter import AieMcpToolSpecs
from test_planner_owned_loop import normalize, reply
from test_runtime import _case_run_for_gate, _fake_llm_deps

MACRO = "macro.screen_rotor_torsion_topology"
BASELINE = "microscopic.run_baseline_bundle"
RECOVER = "microscopic.optimize_ground_state_geometry"
RELAX = "microscopic.run_excited_state_relaxation"
SCAN = "microscopic.run_relaxed_coordinate_scan"


def case(*, prepared=False, enabled=True):
    result = _case_run_for_gate(with_prepared_structure=prepared)
    return result.touch(runtime={
        **result.runtime, "planner_tool_loading": "on_demand",
        "tool_environment": {"amesp_enabled": enabled},
    })


def query(*ids):
    return {"action": "describe_tools", "capability_ids": list(ids)}


def dispatch(tool, **args):
    result = PlannerDecision.model_validate(normalize(reply("dispatch", selected_capability_ids=[tool])))
    return result.dispatch_requests[0].model_copy(update={"tool_args": args})


def session(current=None, registry=None):
    return ToolDiscoverySession(registry or default_capability_registry(), current or case(), "R001")


def test_full_catalog_includes_missing_input_and_live_recovery_status():
    registry = default_capability_registry()
    current = case()
    catalog = {c["capability_id"]: c for c in case_catalog(registry, current)}
    assert len(catalog) == 53
    assert catalog[BASELINE]["status"] == "missing_input"
    assert catalog[BASELINE]["missing_inputs"] == ["prepared_structure"]
    assert all("tool_arg_schema" not in row for row in catalog.values())
    assert registry.availability_violations(dispatch(BASELINE), current.artifact_manifest, amesp_enabled=True)
    initial = ArtifactRecord(artifact_id="initial", kind="initial_structure", created_by_agent="structure",
                             created_by_round="R000")
    current = current.touch(artifact_manifest=current.artifact_manifest.with_records([initial]))
    states = {c["capability_id"]: c["status"] for c in case_catalog(registry, current)}
    assert states[RECOVER] == "ready"
    assert states[BASELINE] == "missing_input"
    prepared = initial.model_copy(update={"artifact_id": "prepared", "kind": "prepared_structure"})
    current = current.touch(artifact_manifest=current.artifact_manifest.with_records([prepared]))
    assert {c["capability_id"]: c["status"] for c in case_catalog(registry, current)}[BASELINE] == "ready"


def test_queries_are_read_only_and_allow_missing_input_descriptions(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata lookup must not execute chemistry")

    monkeypatch.setattr("mechcal.tools.LocalMacroTool.run", forbidden)
    monkeypatch.setattr("mechcal.tools.LocalMicroscopicTool.run", forbidden)
    current = case()
    before = current.model_dump_json()
    lookup = session(current)
    lookup.query(query(BASELINE, MACRO))
    assert current.model_dump_json() == before
    assert list(lookup.state.loaded_tools) == [BASELINE, MACRO]
    assert lookup.augment_payload({})["capability_cards"][0]["status"] == "missing_input"
    lookup.query(query(MACRO))
    assert len(lookup.state.loaded_tools) == 2
    assert lookup.batches_used == 2
    with pytest.raises(ValueError, match="exhausted"):
        lookup.query(query(MACRO))


@pytest.mark.parametrize("invalid", [query("not_a_tool"), query(), query(MACRO, MACRO),
                                    query(*[f"tool{i}" for i in range(7)])])
def test_invalid_queries_consume_allowance_and_load_nothing(invalid):
    lookup = session()
    with pytest.raises(ValueError):
        lookup.query(invalid)
    assert lookup.batches_used == 1
    assert lookup.state.loaded_tools == {}
    assert lookup.last_query["error"]


def test_cache_is_case_scoped_and_registry_versioned():
    current = case()
    lookup = session(current)
    lookup.query(query(MACRO))
    saved = lookup.state.model_dump(mode="json")
    current = current.touch(runtime={**current.runtime, "tool_discovery": saved})
    restored = ToolDiscoverySession(lookup.registry, current, "R002")
    assert restored.state == lookup.state
    assert restored.batches_used == 0
    assert session(current.touch(case_id="another-case")).state.loaded_tools == {}
    changed = [spec.model_copy(update={"description": "Changed definition."}) if spec.capability_id == MACRO
               else spec for spec in lookup.registry.snapshot().capabilities]
    assert session(current, CapabilityRegistry(changed)).state.loaded_tools == {}


def test_decision_revision_owns_query_budget_not_scientific_round():
    current = case()
    current.runtime["question_loop"] = {"revision": 1}
    lookup = session(current)
    lookup.query(query(MACRO))
    lookup.query(query(BASELINE))
    current.runtime.update(tool_discovery=lookup.state.model_dump(mode="json"),
                           tool_discovery_budget=lookup.budget_snapshot())
    # Retries and checkpoint restoration of the same decision cannot reset limits.
    assert session(current).batches_used == 2
    with pytest.raises(ValueError, match="exhausted"):
        session(current).query(query(RECOVER))
    # Reviewer feedback can produce a new accepted decision within R001.
    current.runtime["question_loop"]["revision"] = 2
    revised = session(current)
    assert revised.batches_used == 0
    assert set(revised.state.loaded_tools) == {MACRO, BASELINE}
    revised.query(query(RECOVER))
    assert revised.budget_snapshot()["decision_revision"] == 2
    assert revised.batches_used == 1
    assert session(current.touch(case_id="separate-case")).batches_used == 0


def test_legacy_budget_without_revision_is_retained_conservatively():
    current = case()
    current.runtime["question_loop"] = {"revision": 2}
    current.runtime["tool_discovery_budget"] = {
        "case_id": current.case_id, "round_id": "R001", "batches_used": 2,
    }
    assert session(current).batches_used == 2
    assert ToolDiscoverySession(default_capability_registry(), current, "R002").batches_used == 0


def test_loading_cannot_bypass_amesp_s1_or_ablation_restrictions():
    lookup = session(case(prepared=True))
    lookup.query(query(RELAX, SCAN))
    with pytest.raises(ValueError, match="disabled"):
        lookup.validate_requests([dispatch(RELAX)])
    with pytest.raises(ValueError, match="disabled"):
        lookup.validate_requests([dispatch(SCAN, state_root=1)])
    lookup = session(case(prepared=True, enabled=False))
    lookup.query(query(BASELINE))
    with pytest.raises(ValueError, match="disabled"):
        lookup.validate_requests([dispatch(BASELINE)])
    lookup = session(case(prepared=True), default_capability_registry().with_amesp_backend(False))
    lookup.query(query(BASELINE))
    with pytest.raises(ValueError, match="disabled"):
        lookup.validate_requests([dispatch(BASELINE)])
    lookup = session(registry=default_capability_registry().for_owner_agents(("macro",)))
    with pytest.raises(ValueError, match="disallowed"):
        lookup.query(query(BASELINE))


class Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.payloads = []
        self.feedback = []

    def is_configured(self):
        return True

    def complete_json(self, *, payload, system_prompt, schema_feedback=None):
        self.payloads.append(payload)
        self.feedback.append(schema_feedback)
        return next(self.replies)


def planner(client):
    return PlannerAgent(llm_client=client, enable_llm_decisions=True, tool_loading_mode="on_demand")


def test_queries_need_no_ranking_and_schema_retry_does_not_reset_allowance():
    malformed = reply()
    del malformed["decision_assessment"]
    client = Replies([query(MACRO), query(BASELINE), malformed, query(RECOVER), reply()])
    events = []
    result = planner(client).plan_initial(case(), round_id="R001", discovery_event_sink=events.append)
    assert result.action == "finalize"
    assert [p["tool_discovery"]["remaining_batches"] for p in client.payloads] == [2, 1, 0, 0, 0]
    assert RECOVER not in result.raw_response["tool_discovery_state"]["loaded_tools"]
    assert client.feedback[-1] and "exhausted" in client.feedback[-1]
    assert sum(e["kind"] == "tool_description_query" for e in events) == 3
    assert len(client.payloads[0]["capability_catalog"]) == 53
    assert client.payloads[0]["capability_cards"] == []


def test_dispatch_requires_loaded_detail_and_still_checks_parameters():
    client = Replies([reply("dispatch", selected_capability_ids=[MACRO]), query(MACRO),
                      reply("dispatch", selected_capability_ids=[MACRO])])
    result = planner(client).plan_initial(case(), round_id="R001")
    assert result.action == "dispatch"
    assert "Load current details" in client.feedback[1]
    bad = dispatch(MACRO, imaginary_parameter=1).model_dump(mode="json")
    bad["round_id"] = "R001"
    client = Replies([query(MACRO), reply("dispatch", dispatch_requests=[bad]), reply()])
    result = planner(client).plan_initial(case(), round_id="R001")
    assert result.action == "finalize"
    assert "tool_args" in client.feedback[-1]


def test_independent_gate_rejects_unloaded_or_stale_details():
    current = case()
    lookup = session(current)
    result = PlannerDecision.model_validate(normalize(reply("dispatch", selected_capability_ids=[MACRO])))
    assert DeterministicGate().apply(result, current).status == "retry_policy"
    lookup.query(query(MACRO))
    current = current.touch(runtime={**current.runtime, "tool_discovery": lookup.state.model_dump(mode="json")})
    assert DeterministicGate().apply(result, current).status == "allow"


def test_reviewer_receives_same_directory_and_does_not_recommend_unready_tool():
    current, registry = case(), default_capability_registry()
    payload = _coverage_payload(current, registry, "R002", previous_planner_action="dispatch")
    assert payload["capability_cards"] == case_catalog(registry, current)
    coverage = _sanitize_coverage({"recommended_next_routes": [{
        "route": "run_baseline_bundle", "capability_ids": [BASELINE],
        "reason": "Offline test", "targets": ["RIM_RIR_RIV"],
    }]}, case_run=current, round_id="R002", capability_registry=registry)
    assert coverage.recommended_next_routes == []


def test_description_contains_actual_schema_defaults_bounds_and_limitations():
    registry = default_capability_registry()
    descriptions = registry.describe_tools(
        [SCAN, "macro.run_dimer_packing_proxy"], case(prepared=True).artifact_manifest,
        amesp_enabled=True,
    )
    scan, proxy = descriptions
    assert scan["allowed_argument_values"] == {"state_root": [0]}
    assert scan["tool_arg_schema"]["properties"]["max_wall_seconds"]["default"] == 300
    assert scan["tool_arg_schema"]["properties"]["max_wall_seconds"]["maximum"] == 1800
    assert "No dimer" in proxy["description"]
    assert scan["usage_hints"]["atom_indices"]
    conformers = registry.describe_tools(
        ["microscopic.run_conformer_bundle", "microscopic.run_conformer_state_probe"],
        case(prepared=True).artifact_manifest, amesp_enabled=True,
    )
    assert all("MMFF94 or UFF" in row["limitations"][0] for row in conformers)
    assert all("not a state count" in row["usage_hints"]["td_tout"] for row in conformers)


def test_query_usage_is_separate_and_failure_keeps_cache_and_logs(tmp_path, monkeypatch):
    from openai.types.responses import Response
    from test_responses_api import response_body

    replies = iter([Response.model_validate(response_body(
                        json.dumps(query(MACRO)), parallel_tool_calls=False, tool_choice="none", tools=[],
                    )),
                    TimeoutError("Offline simulated relay timeout")])

    def complete(*args, **kwargs):
        value = next(replies)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr("mechcal.runtime.llm.create_completion", complete)
    monkeypatch.setenv("MECHCAL_LLM_TELEMETRY_PATH", str(tmp_path / "calls.jsonl"))
    client = OpenAIJsonClient(OpenAICompatibleSettings(
        base_url="https://unused.test", model="offline-fixture", api_key="offline-fixture",
        max_retries=0,
    ))
    current = case()
    store = RunStore(tmp_path, current.case_id)
    store.initialize()
    before = current.model_dump_json()
    with pytest.raises(TimeoutError):
        planner(client).plan_initial(current, round_id="R001", discovery_event_sink=store.write_tool_discovery_event)
    events = [json.loads(line) for line in (store.root / "history/tool_discovery.jsonl").read_text().splitlines()]
    usage = next(e for e in events if e["kind"] == "tool_discovery_model_response")
    assert usage["action"] == "describe_tools" and usage["usage"]["total_tokens"] == 18
    assert events[-1]["kind"] == "tool_discovery_model_error"
    assert list(json.loads((store.root / "tool_discovery.json").read_text())["loaded_tools"]) == [MACRO]
    assert current.model_dump_json() == before
    assert not (store.root / "evidence_units.json").exists()
    telemetry = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert telemetry[0]["response_action"] == "describe_tools"
    assert telemetry[0]["payload_metadata"]["tool_loading_mode"] == "on_demand"
    assert telemetry[1]["usage"] is None


@pytest.mark.parametrize("wrapper", [None, "planner_decision", "decision", "response"])
def test_lookup_usage_classifies_accepted_response_wrappers(wrapper):
    response = query(MACRO)
    assert _response_action({wrapper: response} if wrapper else response) == "describe_tools"


def test_queries_do_not_add_rounds_actions_or_evidence_and_persist_across_rounds(tmp_path):
    config = OrchestratorConfig(
        run_base_dir=tmp_path, max_rounds=8, planner_tool_loading="on_demand",
        enable_agenda_coverage_reviewer=False, enable_support_auditor=False,
    )
    client = Replies([query(MACRO), reply("dispatch", selected_capability_ids=[MACRO]),
                      reply("dispatch", selected_capability_ids=[MACRO]), reply()])
    deps = _fake_llm_deps(config)
    deps.planner = planner(client)
    result = MechCALOrchestrator(config, deps).run(smiles="CCO", user_query="Assess", case_id="discovery")
    assert result.status == "finalized"
    assert result.round_ids == ["R001", "R002", "R003"]
    assert result.runtime["worker_call_counts"]["macro"] == 2
    assert "planner_tool_discovery.md" in result.runtime["ablation"]["prompt_files"]
    assert list(client.payloads[2]["tool_discovery"]["loaded_ids"]) == [MACRO]
    assert client.payloads[2]["tool_discovery"]["remaining_batches"] == 2
    assert all(e.capability_id == MACRO for e in result.evidence_ledger.items)
    persisted = json.loads((tmp_path / "discovery/tool_discovery.json").read_text())
    assert persisted == result.runtime["tool_discovery"]
    events = [json.loads(line) for line in (tmp_path / "discovery/history/tool_discovery.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "tool_description_query" for e in events) == 1


@pytest.mark.integration
def test_registry_matches_real_mcp_and_backend_state(monkeypatch):
    pytest.importorskip("fastmcp")
    monkeypatch.setenv("MECHCAL_ENABLE_AMESP", "0")
    remote = AieMcpToolSpecs().discover_capability_registry(StdioMcpTransport())
    local = default_capability_registry().with_amesp_backend(False)
    assert remote.cards() == local.cards()
    assert case_catalog(remote, case(prepared=True)) == case_catalog(local, case(prepared=True))
