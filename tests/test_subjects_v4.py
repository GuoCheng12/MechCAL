"""Gateway and v4-native subjects, offline: scripted model replies, real RDKit tools."""

from dataclasses import dataclass, field

import pytest

from mechcal.eval.grounding import check_submission
from mechcal.eval.subjects_v4 import (
    DirectV4Subject, ReactV4Subject, model_submission_schema, parse_model_submission, save_result,
)
from mechcal.eval.tool_gateway import ToolGateway, numeric_leaves

SMILES = "CN(C)c1ccc(cc1)C#N"
DA_TOOL = "macro.screen_donor_acceptor_layout"


@dataclass
class Reply:
    data: dict
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    parse_error: str | None = None


class Scripted:
    def __init__(self, *replies):
        self.replies, self.payloads = list(replies), []

    def complete_json_result(self, *, system_prompt, payload, schema_feedback=None, **_):
        self.payloads.append({"payload": payload, "feedback": schema_feedback})
        item = self.replies.pop(0)
        return item(payload) if callable(item) else Reply(item)


def preds(first_evidence):
    return {"mechanism_predictions": [
        {"label": "ICT_TICT_CT", "rank": 1, "evidence": first_evidence, "limitations": ["no excited states"]},
        {"label": "RIM_RIR_RIV", "rank": 2, "evidence": [
            {"statement": "The NMe2-aryl bond is rotatable.", "kind": "structural", "atom_ids": [2, 4]}],
         "limitations": ["topology only"]},
        {"label": "PET_ET", "rank": 3, "evidence": [], "limitations": ["no energetics"]},
    ]}


STRUCTURAL = [{"statement": "Dimethylamino donor para to a nitrile acceptor.", "kind": "structural"}]


def test_numeric_leaves_escape_pointer_tokens():
    assert numeric_leaves({"a/b": [1, {"c~": 2.5, "d": "x", "e": True}]}) == [("/a~1b/0", 1.0), ("/a~1b/1/c~0", 2.5)]


def test_schema_and_parser():
    schema = model_submission_schema()
    assert "case_id" not in schema["properties"]
    assert schema["$defs"]["PredictionV4"]["properties"]["label"]["enum"][0] == "RIM_RIR_RIV"
    sub = parse_model_submission(preds(STRUCTURAL), "CASE")
    assert sub.case_id == "CASE" and len(sub.mechanism_predictions) == 3
    with pytest.raises(ValueError, match="kind"):
        parse_model_submission(preds([{"statement": "x", "kind": "vibes"}]), "CASE")
    two = preds(STRUCTURAL)
    two["mechanism_predictions"].pop()
    with pytest.raises(ValueError, match="three"):
        parse_model_submission(two, "CASE")


def test_direct_repairs_once_then_submits():
    bad = preds(STRUCTURAL)
    bad["mechanism_predictions"][1]["label"] = "ICT_TICT_CT"
    client = Scripted(bad, preds(STRUCTURAL))
    result = DirectV4Subject(client).run("CASE", SMILES)
    assert result.status == "submitted" and result.usage["model_calls"] == 2
    assert "distinct" in client.payloads[1]["feedback"]
    assert result.usage["by_role"]["direct"]["total_tokens"] == 240


def test_direct_records_failure_without_fallback_answer():
    result = DirectV4Subject(Scripted({"oops": 1}, {"oops": 2})).run("CASE", SMILES)
    assert result.status == "no_valid_submission" and result.submission is None


@pytest.fixture
def gateway_factory(tmp_path):
    return lambda case_id, smiles: ToolGateway(case_id=case_id, smiles=smiles, run_dir=tmp_path / case_id)


def test_gateway_records_calls_aliases_and_budget(tmp_path):
    g = ToolGateway(case_id="CASE", smiles=SMILES, run_dir=tmp_path, max_calls=2)
    obs = g.call(DA_TOOL)
    assert obs["status"] == "success" and obs["artifact_id"] == "T001"
    assert obs["values"] and not any("/timing/" in p for p in obs["values"])
    assert g.store.artifacts["T001"].provenance["tool_result_id"] in g.store.artifacts
    assert g.call("nope")["status"] == "rejected"      # unknown tool: rejected, not charged
    assert g.call(DA_TOOL)["status"] == "success"
    assert g.call(DA_TOOL)["status"] == "rejected"     # budget of 2 executed calls reached
    assert g.cost()["tool_calls"] == 2 and g.cost()["rejected_calls"] == 2
    pointer, value = next(iter(obs["values"].items()))
    assert g.read("T001", pointer)["value"] == value
    assert (tmp_path / "rounds/R001").is_dir()


def test_react_end_to_end_citation_is_grounded(gateway_factory, tmp_path):
    def submit_with_citation(payload):
        obs = payload["history"][-1]["observation"]
        pointer = "/structured_results/metrics/donor_acceptor_proxy"
        assert pointer in obs["values"]
        value = obs["values"][pointer]
        claim = {"statement": f"The donor-acceptor proxy is {value:g}.", "kind": "computed",
                 "quantities": [{"name": "da_proxy", "value": value, "unit": "",
                                 "source": {"artifact_id": obs["artifact_id"], "pointer": pointer}}]}
        return Reply({"action": "submit", **preds([claim, *STRUCTURAL])})

    client = Scripted({"action": "describe_tools", "tools": [DA_TOOL]},
                      {"action": "call_tool", "tool": DA_TOOL, "args": {}},
                      submit_with_citation)
    result = ReactV4Subject(client, gateway_factory, max_model_calls=5, max_tool_calls=3).run("CASE", SMILES)
    assert result.status == "submitted" and result.tool_cost["tool_calls"] == 1
    check = check_submission(result.submission, result.store, SMILES)
    assert check.predictions[0].claims[0].outcome == "grounded" and check.fabrication_rate == 0.0
    assert client.payloads[0]["payload"]["tool_catalog"] and not client.payloads[0]["payload"]["must_submit_now"]
    path = save_result(result, tmp_path / "out")
    assert path.exists() and (tmp_path / "out/artifacts_v4.json").exists()


def test_react_forces_submission_and_reports_exhaustion(gateway_factory):
    client = Scripted(*[{"action": "call_tool", "tool": DA_TOOL, "args": {}}] * 3)
    result = ReactV4Subject(client, gateway_factory, max_model_calls=3, max_tool_calls=1).run("CASE", SMILES)
    assert result.status == "no_valid_submission" and result.submission is None
    assert client.payloads[1]["payload"]["must_submit_now"] is True   # tool budget used up
    assert result.transcript[1]["observation"]["status"] == "rejected"
