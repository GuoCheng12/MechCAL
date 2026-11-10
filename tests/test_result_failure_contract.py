"""Offline failure contracts; scripted decisions are not scientific evidence."""

import json
import socket
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from mechcal.runtime.questions import apply_assimilation, replace_state, state, validate_assimilation
from mechcal.runtime.result_contract import constrain_disposition_schema, disposition_contract
from mechcal.runtime.retry import RequestValidationError, SchemaRetryError
from mechcal.runtime.structured_output import PlannerOutputContract
from mechcal.schemas.question_loop import QuestionAssimilation
from mechcal.schemas.failures import FailureReport
from mechcal.schemas.tool_results import ToolExecutionResult
from test_question_loop import (
    NOTE, TOOL, QuestionClient, assimilation, loop, pending_case, request,
)
from test_runtime import _evidence_unit


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline test cannot call external services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def case_with_units(status, *, observation=False, boundary_only=False):
    case = pending_case(status)
    units = [_evidence_unit(evidence_id="X1:status", source_report_id="X1", round_id="R001",
                            agent_name="macro", family="torsion_sensitivity", summary=NOTE,
                            status="failed", support="unresolved")]
    if observation:
        units.append(_evidence_unit(evidence_id="X1:observation", source_report_id="X1", round_id="R001",
                                    agent_name="macro", family="torsion_sensitivity", summary=NOTE,
                                    status="partial", metrics={"fixture_value": 1.25}))
    result = deepcopy(state(case)["pending_results"][0])
    result["evidence_refs"] = [e.evidence_id for e in units]
    result["structured_results"].update(boundary_only=boundary_only)
    case = case.touch(evidence_ledger=case.evidence_ledger.with_items(units))
    return replace_state(case, results=[result], pending_results=[result])


def update_for(case, outcome):
    data = assimilation({"case_id": case.case_id, "revision": state(case)["revision"],
                         "pending_results": state(case)["pending_results"]})
    data["dispositions"][0]["outcome"] = outcome
    return data


@pytest.mark.parametrize("status", ["failed", "unsupported", "precondition_missing", "partial"])
@pytest.mark.parametrize("outcome", ["no_result", "incomplete", "informative", "non_discriminating"])
def test_status_only_schema_and_validator_agree(status, outcome):
    case = case_with_units(status, boundary_only=status == "partial")
    contract = disposition_contract(state(case)["pending_results"][0], case)
    assert contract["allowed_outcomes"] == ["no_result"]
    schema = QuestionAssimilation.model_json_schema()
    constrain_disposition_schema(schema, [contract])
    data = update_for(case, outcome)
    errors = list(Draft202012Validator(schema).iter_errors(data))
    if outcome == "no_result":
        assert not errors
        api = PlannerOutputContract({"output_contract": schema})
        normalized = QuestionAssimilation.model_validate(data).model_dump(mode="json")
        assert api.decode(api.encode(normalized)) == normalized
        before = case.model_dump(mode="json")
        validate_assimilation(QuestionAssimilation.model_validate(data), case)
        assert case.model_dump(mode="json") == before
    else:
        assert errors
        with pytest.raises(RequestValidationError) as exc:
            validate_assimilation(QuestionAssimilation.model_validate(data), case)
        issue = exc.value.issues[0]
        assert issue["path"] == "dispositions[0].outcome"
        assert issue["result_id"] == "X1" and issue["actual"] == outcome
        assert issue["allowed"] == ["no_result"]
        assert data["dispositions"][0]["outcome"] == outcome  # no silent repair


@pytest.mark.parametrize("status", ["success", "partial", "failed"])
def test_surviving_subresults_allow_incomplete_and_keep_numbers(status):
    case = case_with_units(status, observation=True)
    contract = disposition_contract(state(case)["pending_results"][0], case)
    assert "incomplete" in contract["allowed_outcomes"]
    assert contract["observation_refs"] == ["X1:observation"]
    assert contract["operational_refs"] == ["X1:status"]
    schema = QuestionAssimilation.model_json_schema()
    constrain_disposition_schema(schema, [contract])
    data = update_for(case, "incomplete")
    Draft202012Validator(schema).validate(data)
    api = PlannerOutputContract({"output_contract": schema})
    normalized = QuestionAssimilation.model_validate(data).model_dump(mode="json")
    assert api.decode(api.encode(normalized)) == normalized
    updated = apply_assimilation(case, QuestionAssimilation.model_validate(data))
    assert updated.evidence_ledger == case.evidence_ledger
    assert updated.evidence_ledger.items[-1].metrics == {"fixture_value": 1.25}
    assert updated.portfolio == case.portfolio
    assert state(updated)["questions"]["Q1"]["answer_state"] == "open"


def test_mixed_result_schema_uses_each_result_not_one_global_enum():
    case = case_with_units("partial", observation=True)
    good = deepcopy(state(case)["pending_results"][0])
    failed = {**good, "result_id": "X2", "status": "failed", "evidence_refs": ["X1:status"]}
    case = replace_state(case, results=[good, failed], pending_results=[good, failed])
    schema = QuestionAssimilation.model_json_schema()
    constrain_disposition_schema(schema, [disposition_contract(r, case) for r in [good, failed]])
    data = assimilation({"case_id": case.case_id, "revision": state(case)["revision"],
                         "pending_results": [good, failed]})
    data["dispositions"][0]["outcome"] = "incomplete"
    Draft202012Validator(schema).validate(data)
    api = PlannerOutputContract({"output_contract": schema})
    normalized = QuestionAssimilation.model_validate(data).model_dump(mode="json")
    assert api.decode(api.encode(normalized)) == normalized
    validate_assimilation(QuestionAssimilation.model_validate(data), case)
    data["dispositions"][1]["outcome"] = "incomplete"
    assert list(Draft202012Validator(schema).iter_errors(data))


def test_failure_records_are_usable_for_recovery_not_support_arguments():
    case = case_with_units("partial", observation=True)
    data = update_for(case, "incomplete")
    data["questions"][0].update(result_ids=["X1"], evidence_refs=["X1:status"])
    argument = dict(support_id="S1", target_label="RIM_RIR_RIV", support_level="partial",
                    finding=NOTE, warrant=NOTE, boundary=NOTE, observation_refs=["X1:status"])
    data["mechanism_support_arguments"] = [argument]
    with pytest.raises(RequestValidationError, match="operational-only") as exc:
        validate_assimilation(QuestionAssimilation.model_validate(data), case)
    assert exc.value.issues[0]["invalid_refs"] == ["X1:status"]
    argument["observation_refs"] = ["X1:observation"]
    validate_assimilation(QuestionAssimilation.model_validate(data), case)


@pytest.mark.parametrize("bundle_status", ["available", "partial"])
@pytest.mark.parametrize("boundary_only", [True, False])
def test_real_microscopic_bundle_projection_preserves_boundary_scope(bundle_status, boundary_only):
    from mechcal.schemas import MicroscopicArtifactBundle
    from mechcal.tools.amesp_microscopic import MICROSCOPIC_ROUTE_PROFILES
    from mechcal.tools.local_micro import _artifact_from_bundle, _evidence_from_bundle

    case = pending_case("partial" if bundle_status == "partial" else "success")
    req = request(cid="microscopic.run_relaxed_coordinate_scan")
    profile = MICROSCOPIC_ROUTE_PROFILES[req.route]
    rows = [{"member_id": "failed-point", "status": "failed", "geometry_converged": False}]
    if not boundary_only:
        rows.append({"member_id": "retained-point", "status": "partial", "geometry_converged": True,
                     "final_energy_hartree": -10.25, "excited_states": []})
    bundle = MicroscopicArtifactBundle(
        bundle_id="R001:scan", capability_name=req.route, status=bundle_status,
        source_artifact_ids=["structure:prepared"], geometry_source="offline fixture",
        parsed_observables={"boundary_only": boundary_only, "route_records": rows,
                            "successful_member_count": 0, "available_descriptors": []},
    )
    artifact = _artifact_from_bundle(bundle, profile=profile, round_id="R001")
    unit = _evidence_from_bundle(bundle, profile=profile, request=req, artifact=artifact, case_run=case)
    assert unit.basis == ("checklist" if boundary_only else "computed")
    result = deepcopy(state(case)["pending_results"][0])
    result.update(evidence_refs=[unit.evidence_id],
                  structured_results={"parsed_observables": bundle.parsed_observables})
    case = case.touch(evidence_ledger=case.evidence_ledger.with_items([unit]))
    case = replace_state(case, results=[result], pending_results=[result])
    contract = disposition_contract(result, case)
    if boundary_only:
        assert contract["allowed_outcomes"] == ["no_result"]
        assert contract["observation_refs"] == [] and contract["operational_refs"] == [unit.evidence_id]
    else:
        assert "incomplete" in contract["allowed_outcomes"]
        assert contract["observation_refs"] == [unit.evidence_id]
    data = update_for(case, "no_result" if boundary_only else "incomplete")
    data["mechanism_support_arguments"] = [dict(
        support_id="S1", target_label="RIM_RIR_RIV", support_level="partial",
        finding=NOTE, warrant=NOTE, boundary=NOTE, observation_refs=[unit.evidence_id],
    )]
    if boundary_only:
        with pytest.raises(RequestValidationError, match="operational-only"):
            validate_assimilation(QuestionAssimilation.model_validate(data), case)
    else:
        updated = apply_assimilation(case, QuestionAssimilation.model_validate(data))
        assert updated.evidence_ledger == case.evidence_ledger
        assert state(updated)["results"][0]["structured_results"]["parsed_observables"]["route_records"] == rows


def test_mixed_units_and_checklists_are_not_blanket_discarded():
    case = case_with_units("partial", observation=True)
    units = case.evidence_ledger.items
    units[0].status = "partial"
    units[0].basis = "checklist"
    units[0].metrics = {"computed_observables": {"boundary_only": True}}
    units[1].basis = "checklist"  # A checklist alone is not a failure-status flag.
    contract = disposition_contract(state(case)["pending_results"][0], case)
    assert contract["observation_refs"] == ["X1:observation"]
    assert contract["operational_refs"] == ["X1:status"]
    assert "incomplete" in contract["allowed_outcomes"]


def inject_result(monkeypatch, worker, status, *, boundary_only=False):
    original = worker.execute_tool
    calls = []

    def run(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(value.tool_result_id)
        # Only the first request fails; the scripted Planner chooses the next one.
        if len(calls) > 1 or status == "success":
            return value
        data = value.model_dump(mode="json")
        data.update(status=status, failure=FailureReport(kind="runtime_failed", message=NOTE,
                                                        recoverable=True).model_dump(mode="json"))
        data["structured_results"]["boundary_only"] = boundary_only
        for unit in data["evidence_units"]:
            unit["status"] = "failed" if boundary_only else "partial"
        return ToolExecutionResult.model_validate(data)

    monkeypatch.setattr(worker, "execute_tool", run)
    return calls


@pytest.mark.parametrize("status", ["failed", "unsupported", "precondition_missing", "partial"])
def test_single_failed_tool_repairs_contract_then_planner_continues(tmp_path, monkeypatch, status):
    o, _ = loop(tmp_path, rounds=2, unlimited=True)
    calls = inject_result(monkeypatch, o.deps.workers["macro"], status, boundary_only=True)

    class Client(QuestionClient):
        repairs = 0
        rejected = None

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_assimilate" and payload["pending_results"]:
                data = assimilation(payload)
                contract = payload["disposition_requirements"][0]
                if contract["allowed_outcomes"] == ["no_result"]:
                    if self.rejected is None:
                        data["dispositions"][0]["outcome"] = "incomplete"
                        self.rejected = deepcopy(data)
                        return data
                    feedback = json.loads(schema_feedback)
                    assert feedback["rejected_response"] == self.rejected
                    assert feedback["issues"][0]["result_id"] == contract["result_id"]
                    assert feedback["issues"][0]["allowed"] == ["no_result"]
                    self.repairs += 1
                    data["dispositions"][0]["outcome"] = "no_result"
                return data
            return super().complete_json(payload=payload, system_prompt=system_prompt,
                                         schema_feedback=schema_feedback)

    client = Client(rounds=2)
    o.deps.planner.llm_client = client
    result = o.run(smiles="CCO", user_query=NOTE, case_id="recoverable-tool")
    assert result.status == "finalized" and client.repairs == 1
    assert len(calls) == 2
    assert [r["status"] for r in state(result)["results"]] == [status, "success"]
    assert result.runtime["worker_call_counts"]["macro"] == 2
    assert not result.mechanism_support_arguments


def test_exhausted_correction_preserves_pending_and_requires_explicit_resume(tmp_path, monkeypatch):
    o, _ = loop(tmp_path, rounds=1, unlimited=True)
    calls = inject_result(monkeypatch, o.deps.workers["macro"], "failed", boundary_only=True)

    class Client(QuestionClient):
        invalid_attempts = 0

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            if payload["stage"] == "planner_assimilate" and payload["pending_results"]:
                self.invalid_attempts += 1
                data = assimilation(payload)
                data["dispositions"][0]["outcome"] = "incomplete"
                return data
            return super().complete_json(payload=payload, system_prompt=system_prompt,
                                         schema_feedback=schema_feedback)

    client = Client()
    o.deps.planner.llm_client = client
    with pytest.raises(SchemaRetryError):
        o.run(smiles="CCO", user_query=NOTE, case_id="exhausted")
    root = tmp_path / "exhausted"
    saved = json.loads((root / "question_checkpoint.json").read_text())
    assert client.invalid_attempts == 3 and len(calls) == 1
    assert saved["runtime"]["question_loop"]["phase"] == "assimilate"
    assert saved["runtime"]["question_loop"]["pending_results"]
    assert saved["runtime"]["termination"]["scientific_completion"] is False
    assert not (root / "final.json").exists()
    o.deps.planner.llm_client = QuestionClient(rounds=1)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="exhausted", resume=True)
    assert result.status == "finalized" and len(calls) == 1
    assert state(result)["resume_generation"] == 1


@pytest.mark.parametrize("status", ["success", "partial"])
def test_report_failure_reuses_saved_subresults_without_calculating_again(tmp_path, monkeypatch, status):
    o, _ = loop(tmp_path, rounds=1, unlimited=True)
    worker = o.deps.workers["macro"]
    calls = inject_result(monkeypatch, worker, status)
    report = worker.report_result

    def fail(*args, **kwargs):
        raise TimeoutError("Offline report failure; calculation has already returned")

    monkeypatch.setattr(worker, "report_result", fail)
    with pytest.raises(TimeoutError):
        o.run(smiles="CCO", user_query=NOTE, case_id="report")
    checkpoint = next((tmp_path / "report/executions").glob("*.json"))
    saved = json.loads(checkpoint.read_text())
    assert saved["stage"] == "tool_returned" and saved["tool_result"]["status"] == status
    monkeypatch.setattr(worker, "report_result", report)
    result = o.run(smiles="CCO", user_query=NOTE, case_id="report", resume=True)
    assert result.status == "finalized" and len(calls) == 1
    assert json.loads(checkpoint.read_text())["tool_result"] == saved["tool_result"]


@pytest.mark.parametrize("kind", ["permission", "401", "403"])
def test_denied_access_is_not_converted_to_recoverable_tool_result(tmp_path, monkeypatch, kind):
    import httpx

    o, _ = loop(tmp_path)
    worker = o.deps.workers["macro"]
    case = pending_case()
    req = request(cid=TOOL)
    plan = worker.plan(req, case)
    if kind == "permission":
        error = PermissionError("Offline permission denied")
    else:
        response = httpx.Response(int(kind), request=httpx.Request("POST", "https://offline.invalid"))
        error = httpx.HTTPStatusError("Offline access denied", request=response.request, response=response)

    def deny(*args, **kwargs):
        raise error

    monkeypatch.setattr(worker.tool_client, "run", deny)
    with pytest.raises(type(error)):
        worker.execute_tool(req, case, execution_plan=plan)
