"""Regression checks for the citation-copy failures observed in Cases 0040/0020."""

import copy

import pytest

from mechcal.agents.evidence_references import EvidenceReferences
from mechcal.agents.photophysics_arbiter import (
    DisabledPhotophysicsWebSearch,
    PhotophysicsArbiterAgent,
)
from mechcal.agents.planner import PlannerAgent
from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
from mechcal.runtime.orchestrator import mcp_deps
from mechcal.schemas import EvidenceLedger
from test_planner_owned_loop import reply
from test_runtime import FakeMcpTransport, _case_run_for_gate, _evidence_unit

DIHEDRAL = (
    "R002:dispatch:002:microscopic:e7d1fd5eb28c426682d42f788df49061:list_rotatable_dihedrals"
)
TORSION = (
    "R003:dispatch:002:microscopic:13b4d76b0f1648a98cd146f88d6bdfe8:run_torsion_brightness_coupling_scan"
)
MIXED = (
    "R002:dispatch:002:microscopic:e7d1fd5eb28c42698cd146f88d6bdfe8:list_rotatable_dihedrals"
)
FRONTIER = (
    "R003:frontier-partition:microscopic:dd74d60188b34bf0a27756160a349e2a:run_frontier_orbital_partition"
)
WRONG_FRONTIER = (
    "R003:frontier-partition:microscopic:dd74d60188b34bf0a277ec5c6b44b25:run_frontier_orbital_partition"
)


def citation_case(ids):
    case = _case_run_for_gate()
    return case.touch(
        current_round_id="R004",
        evidence_ledger=EvidenceLedger(case_id=case.case_id, items=[
            _evidence_unit(
                evidence_id=eid, round_id=f"R{i + 1:03d}", source_report_id=f"report:{i}",
                agent_name="microscopic", family="torsion_sensitivity",
                summary="Offline fixture observation, not a model result.",
            ) for i, eid in enumerate(ids)
        ]),
    )


def test_exact_roundtrip_and_no_input_mutation():
    refs = EvidenceReferences.from_ids([DIHEDRAL, TORSION])
    original = {"evidence_id": DIHEDRAL, "evidence_refs": [DIHEDRAL, TORSION],
                "rationale": f"Compare {DIHEDRAL} with {TORSION}.",
                "metrics": {"f": 0.0401, "failed": False}, "artifact_refs": ["artifact:E001"]}
    before = copy.deepcopy(original)
    encoded = refs.encode(original)
    assert encoded["evidence_refs"] == ["E001", "E002"]
    assert encoded["rationale"] == "Compare E001 with E002."
    assert refs.decode(encoded) == original == before
    assert refs.decode({"text": "E0010 artifact:E001 E001.extra"}) == {
        "text": "E0010 artifact:E001 E001.extra"
    }


@pytest.mark.parametrize("invalid", [MIXED, WRONG_FRONTIER, "E999", "E01"])
def test_no_fuzzy_repair_or_silent_reference_drop(invalid):
    refs = EvidenceReferences.from_ids([DIHEDRAL, TORSION, FRONTIER])
    for field in ("evidence_refs", "observation_refs", "evidence_refs_added"):
        with pytest.raises(ValueError, match="Unknown evidence reference"):
            refs.decode({"nested": [{field: [invalid]}]})


def test_aliases_are_unique_and_keep_existing_short_citations():
    refs = EvidenceReferences.from_ids([DIHEDRAL, "E001", "E1", TORSION, DIHEDRAL])
    assert refs.short_to_full == {
        "E002": DIHEDRAL, "E001": "E001", "E1": "E1", "E003": TORSION,
    }
    before = EvidenceReferences.from_ids([DIHEDRAL, TORSION])
    after = EvidenceReferences.from_ids([DIHEDRAL, TORSION, FRONTIER])
    assert all(after.short_to_full[k] == v for k, v in before.short_to_full.items())


def test_case0040_planner_repairs_only_after_explicit_corrected_model_response():
    class Client:
        def __init__(self):
            self.feedback = []

        def is_configured(self):
            return True

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            self.feedback.append(schema_feedback)
            assert [e["evidence_id"] for e in payload["evidence_index"]] == ["E001", "E002"]
            assert DIHEDRAL not in str(payload)
            response = reply()
            response["decision_assessment"]["evidence_refs"] = [MIXED] if not schema_feedback else ["E001"]
            response["portfolio"]["hypotheses"][0]["evidence_refs"] = ["E002"]
            return response

    client = Client()
    decision = PlannerAgent(llm_client=client, enable_llm_decisions=True).plan_initial(
        citation_case([DIHEDRAL, TORSION]), round_id="R004",
    )
    assert decision.action == "finalize"
    assert len(client.feedback) == 2
    assert "Available: E001, E002" in client.feedback[1]
    assert decision.decision_assessment.evidence_refs == [DIHEDRAL]
    assert decision.portfolio.hypotheses[0].evidence_refs == [TORSION]
    assert decision.raw_response["evidence_reference_map"] == {"E001": DIHEDRAL, "E002": TORSION}
    assert decision.raw_response["wire_model_response"]["decision_assessment"]["evidence_refs"] == ["E001"]


def test_case0020_auditor_recovers_on_second_attempt():
    class Client:
        def __init__(self):
            self.feedback = []

        def is_configured(self):
            return True

        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            self.feedback.append(schema_feedback)
            assert payload["compact_evidence_ids"] == ["E001"]
            assert FRONTIER not in str(payload)
            return {"coverage_axes": [{
                "axis_id": "charge", "label": "Charge localization", "status": "covered",
                "evidence_refs": [WRONG_FRONTIER] if not schema_feedback else ["E001"],
                "rationale": "Offline fixture of a bounded observation.",
            }]}

    client = Client()
    review = PhotophysicsArbiterAgent(
        llm_client=client, web_search=DisabledPhotophysicsWebSearch(), max_attempts=3,
    ).review(citation_case([FRONTIER]))
    assert len(client.feedback) == 2
    assert "Available: E001" in client.feedback[1]
    assert review.coverage_axes[0].evidence_refs == [FRONTIER]


def test_auditor_correction_budget_is_bounded_and_never_returns_fabricated_review():
    class Client:
        calls = 0

        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            self.calls += 1
            return {"source_evidence_refs": [WRONG_FRONTIER]}

    client = Client()
    with pytest.raises(RuntimeError, match="failed to produce a valid review"):
        PhotophysicsArbiterAgent(
            llm_client=client, web_search=DisabledPhotophysicsWebSearch(), max_attempts=3,
        ).review(citation_case([FRONTIER]))
    assert client.calls == 3


def test_network_failure_does_not_multiply_schema_retries():
    class Client:
        calls = 0

        def is_configured(self):
            return True

        def complete_json(self, **kwargs):
            self.calls += 1
            raise TimeoutError("Fixture transport failure after client retries")

    client = Client()
    with pytest.raises(RuntimeError, match="TimeoutError"):
        PhotophysicsArbiterAgent(
            llm_client=client, web_search=DisabledPhotophysicsWebSearch(), max_attempts=3,
        ).review(citation_case([FRONTIER]))
    assert client.calls == 1


def test_default_local_and_mcp_auditors_allow_three_correction_attempts(tmp_path):
    config = OrchestratorConfig(run_base_dir=tmp_path)
    assert MechCALOrchestrator(config).deps.photophysics_arbiter.max_attempts == 3
    assert mcp_deps(config, FakeMcpTransport()).photophysics_arbiter.max_attempts == 3
