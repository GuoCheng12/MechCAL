"""Review decisions/provenance changes, with source-addressable quantitative citations."""

from copy import deepcopy

import pytest

from mechcal.agents.observation_view import observation_view
from mechcal.runtime.command_loop import _apply, _comment
from mechcal.runtime.command_operations import OperationRejected, protocol, put, revise
from mechcal.runtime.command_review import review_basis, review_citations
from mechcal.schemas.commands import ReadRecord, ReviseJudgment
from test_command_protocol import ScriptClient, case, judgment, orchestrator, store


def coefficient_case():
    rows = [{"from_orbital": 87, "to_orbital": 88, "coefficient": -0.0023579}]
    rows += [{"from_orbital": 86, "to_orbital": 88 + i, "coefficient": -.6 + i * .01} for i in range(10)]
    return put(case(), results={"raw": {"status": "success", "state_index": 3, "components": rows}})


def cited_judgment():
    update = judgment().model_copy(deep=True)
    update.changes[0].refs = ["raw"]
    update.changes[0].citations = [ReadRecord(ref="raw", paths=["/components/0"])]
    return ReviseJudgment.model_validate(update.model_dump(mode="json"))


def test_summary_never_relabels_original_array_indices():
    raw = protocol(coefficient_case())["results"]["raw"]
    before = deepcopy(raw)
    view = observation_view(raw, preserve_paths=True)
    assert "components" not in view
    assert view["component_excerpt"][0]["source_index"] == 1
    assert view["component_excerpt"][0]["value"] == raw["components"][1]
    assert all(row["source_index"] != 0 for row in view["component_excerpt"])
    assert raw == before
    # Legacy/judge views are unchanged by this opt-in addressing mode.
    assert "components" in observation_view(raw)


def test_reviewer_gets_exact_omitted_row_not_another_coefficient(tmp_path):
    o, c = orchestrator(tmp_path, reviews=True), coefficient_case()
    s = store(tmp_path, c)
    c = _apply(o, c, cited_judgment(), s)
    client = ScriptClient(["The exact cited row is present; evidence still needs scientific interpretation."])
    o.deps.photophysics_arbiter.llm_client = client
    reviewed = _comment(o, c, s)
    payload = client.calls[0]["payload"]
    assert payload["citations"][0]["data"]["/components/0"]["coefficient"] == -.0023579
    assert "components" not in payload["observations"]["raw"]
    assert protocol(reviewed)["results"] == protocol(c)["results"]


def test_wording_changes_defer_to_explicit_or_final_review_without_semantic_guessing(tmp_path):
    o, c = orchestrator(tmp_path, reviews=True), case()
    s = store(tmp_path, c)
    c = _apply(o, c, judgment(), s)
    assert protocol(c)["phase"] == "review"
    o.deps.photophysics_arbiter.llm_client = ScriptClient(["Keep the claim bounded."])
    c = _comment(o, c, s)
    updated = judgment().model_copy(deep=True)
    updated.changes[0].text = "A clearer description of the same structural hypothesis."
    updated.changes[1].rationale = "Corrected wording, unchanged judgment."
    assert review_basis(revise(c, updated)) == review_basis(c)
    edited = _apply(o, c, updated, s)
    assert protocol(edited)["phase"] == "choose"
    assert protocol(edited)["review_trigger"] == "same_sources_return_to_planner"
    assert protocol(edited)["review"] == protocol(c)["review"]
    updated.changes[0].text = "A different interpretation requires a scientific check."
    updated.request_review = True
    assert protocol(_apply(o, edited, updated, s))["phase"] == "review"


@pytest.mark.parametrize("change", ["source", "confidence", "support", "rationale", "candidate"])
def test_unreviewed_legacy_judgments_still_get_review(tmp_path, change):
    o, c = orchestrator(tmp_path, reviews=True), coefficient_case()
    s = store(tmp_path, c)
    c = put(revise(c, judgment()), phase="choose")
    updated = judgment().model_copy(deep=True)
    if change == "source":
        updated.changes[0].refs.append("raw")
    elif change == "confidence":
        updated.changes[1].confidence = .8
    elif change == "support":
        updated.changes[1].support = "partial"
    elif change == "rationale":
        updated.changes[1].rationale = "A different explanation with the same numerical scores."
    else:
        updated = ReviseJudgment(kind="revise_judgment", changes=[{
            "kind": "candidate", "label": "PACKING_HOST_MATRIX_CONFINEMENT", "status": "considering",
            "note": "Structural possibility, not a packing calculation.", "refs": ["input:smiles"]}])
    assert protocol(_apply(o, c, updated, s))["phase"] == "review"


@pytest.mark.parametrize("fault", ["unknown", "path", "directory", "uncited", "large"])
def test_bad_citations_reject_locally_and_do_not_partially_commit(tmp_path, fault):
    c = coefficient_case()
    cmd = cited_judgment()
    citation = cmd.changes[0].citations[0]
    if fault == "unknown": citation.ref = "other-case"
    if fault == "path": citation.paths = ["/components/999"]
    if fault == "directory": citation.view = "directory"
    if fault == "uncited": citation.ref, citation.paths = "input:smiles", ["/smiles"]
    if fault == "large":
        c = put(c, results={"raw": {"status": "success", "large": "x" * 25000}})
        citation.paths = ["/large"]
    before = c.model_dump_json()
    with pytest.raises(OperationRejected):
        revise(c, cmd)
    assert c.model_dump_json() == before


def test_legacy_argument_without_citations_and_paged_selections():
    c = revise(case(), judgment())
    assert review_citations(c, list(protocol(c)["arguments"].values())) == []
    c = coefficient_case()
    cmd = cited_judgment()
    citation = cmd.changes[0].citations[0]
    citation.paths = ["/components"]
    from mechcal.schemas.commands import ReadPage
    citation.page = ReadPage(offset=1, limit=2)
    c = revise(c, cmd)
    rows = review_citations(c, list(protocol(c)["arguments"].values()))
    assert rows[0]["pages"]["/components"]["offset"] == 1
    assert rows[0]["data"]["/components"] == protocol(c)["results"]["raw"]["components"][1:3]
