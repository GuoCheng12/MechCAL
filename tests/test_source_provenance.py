"""Field ownership and dependence are source metadata, not inferred from prose."""

from copy import deepcopy

import pytest

from mechcal.runtime.command_memory import (
    REPORT_BYTES, json_bytes, result_index, result_report, source_provenance,
)
from mechcal.runtime.command_operations import OperationRejected, protocol, put, revise
from test_command_protocol import case, judgment
from test_reviewer_execution_context import OTHER, capture_review, execution_case


def test_field_ownership_is_scoped_complete_and_does_not_borrow_values():
    a = {"structured_results": {"metrics": {"rotatable_bond_count": 11}}}
    b = {"structured_results": {"metrics": {"mol_logp": 7.2315}}}
    assert source_provenance(a)["field_ownership"] == [{
        "path": "/structured_results/metrics", "names": ["rotatable_bond_count"], "complete": True}]
    assert source_provenance(b)["field_ownership"][0]["names"] == ["mol_logp"]
    assert source_provenance(a)["independence"] == "not_recorded"


def test_recorded_derivation_reaches_worker_planner_and_reviewer(tmp_path):
    c = execution_case()
    results = deepcopy(protocol(c)["results"])
    raw = results[OTHER]
    raw["structured_results"]["metrics"] = {"rotatable_bond_count": 11,
        "independent_observation": False, "derived_from": ["rotatable_bond_count"],
        "calculation_scope": "derived_single_molecule_descriptor_heuristic"}
    c = put(c, results=results)
    meta = source_provenance(raw)
    assert meta["independence"] == "declared_not_independent"
    assert result_report(OTHER, raw)["provenance"] == meta
    assert result_index(OTHER, raw)["provenance"] == meta
    payload = capture_review(tmp_path, c, refs=[OTHER])
    assert payload["execution_context"][0]["provenance"] == meta
    assert protocol(c)["results"] == results


def test_equal_values_do_not_prove_independence_or_dependency():
    raw = {"structured_results": {"metrics": {"x": 11}, "parsed_observables": {"x": 11}}}
    meta = source_provenance(raw)
    assert meta["independence"] == "not_recorded" and not meta["derivation"]
    raw["structured_results"]["metrics"]["independent_observation"] = True
    assert source_provenance(raw)["independence"] == "declared_independent_unverified"
    raw["structured_results"]["parsed_observables"]["independent_observation"] = False
    assert source_provenance(raw)["independence"] == "declared_not_independent"


def test_large_metadata_is_bounded_explicitly_partial_and_not_absence():
    raw = {"structured_results": {
        "metrics": {f"measurement_{i}": i for i in range(500)},
        "parsed_observables": {f"other_{i}": i for i in range(500)}}}
    for container in (raw["structured_results"], *raw["structured_results"].values()):
        container.update(derived_from=["source"] * 10000, calculation_scope="scope " * 2000,
                         independent_observation=False)
    before = deepcopy(raw)
    meta = source_provenance(raw)
    assert json_bytes(meta) <= 2048
    assert all(not f["complete"] for f in meta["field_ownership"])
    assert any(d.get("deferred") for d in meta["derivation"])
    assert json_bytes(result_report("r", raw)) <= REPORT_BYTES
    assert raw == before


def test_wrong_field_reference_is_local_and_correct_source_commits():
    raw = {"status": "success", "structured_results": {"metrics": {"rotatable_bond_count": 11}}}
    other = {"status": "success", "structured_results": {"metrics": {"mol_logp": 7.2315}}}
    c = put(case(), results={"rotor": raw, "packing": other})
    change = judgment().model_copy(deep=True)
    change.changes[0].refs = ["rotor", "packing"]
    from mechcal.schemas.commands import ReadRecord
    change.changes[0].citations = [ReadRecord(ref="rotor", paths=["/structured_results/metrics/mol_logp"])]
    before = c.model_dump()
    with pytest.raises(OperationRejected) as exc:
        revise(c, change)
    assert any(i["code"] == "unknown_path" for i in exc.value.issues)
    assert c.model_dump() == before
    change.changes[0].citations[0].ref = "packing"
    assert protocol(revise(c, change))["revision"] == 1


def test_structured_field_list_does_not_forbid_a_valid_raw_source_path():
    raw = {"status": "success", "structured_results": {"metrics": {"rotatable_bond_count": 11}},
           "raw_results": {"base_metrics": {"mol_logp": 7.2315}}}
    c = put(case(), results={"rotor": raw})
    change = judgment().model_copy(deep=True)
    change.changes[0].refs = ["rotor"]
    from mechcal.schemas.commands import ReadRecord
    change.changes[0].citations = [ReadRecord(ref="rotor", paths=["/raw_results/base_metrics/mol_logp"])]
    assert "mol_logp" not in source_provenance(raw)["field_ownership"][0]["names"]
    assert protocol(revise(c, change))["revision"] == 1
