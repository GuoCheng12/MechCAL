"""Information fidelity tests, not proof of any chemical mechanism."""

import pytest

from mechcal.agents.evidence_context import current_support_arguments
from mechcal.eval.tool_sources import build_tool_source_packet
from mechcal.runtime.output_assembler import OutputAssembler
from mechcal.runtime.questions import apply_assimilation, public_payload, replace_state, validate_action
from mechcal.runtime.gates import DeterministicGate
from mechcal.schemas import MechanismSupportArgument
from mechcal.schemas.question_loop import QuestionAssimilation
from mechcal.tools.amesp_transitions import parse_transitions
from mechcal.tools.amesp_microscopic import MICROSCOPIC_ROUTE_PROFILES, _parsed_observables
from test_question_loop import REGISTRY, NOTE, action, assimilation, initial_case, ready_case, request, loop
from test_runtime import _evidence_unit

# Short verbatim numeric excerpt from the saved Case 0003 Amesp output.
TRANSITIONS = """
 Ground to excited state transition electric dipole moments(Au):
   state-state      X         Y        Z         Dip       Osc
     0     1     0.0800   -0.0645    0.4789    0.2399    0.0170
     0     2    -0.0144    0.0105   -0.0751    0.0060    0.0005

 Excited to excited state transition electric dipole moments(Au):
   state-state        X            Y            Z          Osc
     1     2       0.0630      -0.3770       0.0943      0.0011

 State    1 : E =    2.8931 eV     428.553 nm      23334.33 cm-1
       87 -->   88      0.7009178
 E(TD) =    -89.660714483      <S**2>= 0.000     f=  0.0170
 State    2 : E =    3.1800 eV     389.886 nm      25648.50 cm-1
       87 -->   89     -0.7064908
 E(TD) =    -89.650170347      <S**2>= 0.000     f=  0.0005
"""


def test_actual_transition_values_keep_indices_signs_and_units():
    parsed = parse_transitions(TRANSITIONS)
    assert parsed["dominant_transitions"][1] == {
        "state_index": 2, "components": [{"from_orbital": 87, "to_orbital": 89, "coefficient": -.7064908}]}
    assert parsed["ground_to_excited_transition_dipoles"][0]["vector_au"] == [.08, -.0645, .4789]
    assert parsed["excited_to_excited_transition_dipoles"][0]["to_state"] == 2
    assert parsed["transition_conventions"]["orbital_index_base"] == 1
    assert "charge transfer" in parsed["transition_conventions"]["boundary"]
    scientific = _parsed_observables(
        profile=MICROSCOPIC_ROUTE_PROFILES["run_targeted_state_characterization"],
        s0_text="", s1_text=TRANSITIONS, excited_states=[{"state_index": 1}])
    assert scientific["available_descriptors"] == ["dominant_transitions", "state_ordering"]
    assert scientific["missing_descriptors"] == ["state_family_overlap"]


def test_headers_truncated_blocks_and_malformed_vectors_are_not_delivered():
    raw = "Ground to excited state transition electric dipole moments(Au):\n"
    raw += " state-state X Y Z Dip Osc\n0 1 nan .2 .3 .4 .5\n"
    raw += "State 1 : E = 3.0 eV\n 87 --> 88 0.7\n"
    parsed = parse_transitions(raw)
    assert parsed["dominant_transitions"] == []
    assert parsed["ground_to_excited_transition_dipoles"] == []


def test_fortran_exponents_are_preserved():
    parsed = parse_transitions(TRANSITIONS.replace("-0.7064908", "-7.064908D-01"))
    assert parsed["dominant_transitions"][1]["components"][0]["coefficient"] == -.7064908


def support(sid, ref="E1"):
    return MechanismSupportArgument(support_id=sid, target_label="RIM_RIR_RIV", support_level="weak",
                                    finding=f"{sid}: {NOTE}", warrant=f"{sid}: bounded interpretation",
                                    boundary=f"{sid}: not a mechanism proof", observation_refs=[ref])


def test_export_exact_arguments_without_inventing_extra_support():
    c = ready_case()
    units = [_evidence_unit(evidence_id=ref, round_id="R001", source_report_id="fixture",
                            agent_name="macro", family="torsion_sensitivity", summary=NOTE)
             for ref in ("E1", "E2")]
    c.portfolio.hypotheses[0].evidence_refs = ["E1", "E2"]
    c = c.touch(evidence_ledger=c.evidence_ledger.with_items(units),
                mechanism_support_arguments=[support("S1"), support("S2", "input:smiles")])
    before = c.model_dump_json()
    p = OutputAssembler().build_mechanism_predictions(c)[0]
    assert [(d.finding, d.warrant, d.boundary, d.evidence_refs) for d in p.evidence_details] == [
        (a.finding, a.warrant, a.boundary, a.observation_refs) for a in c.mechanism_support_arguments]
    assert p.evidence_refs == ["E1", "input:smiles"]
    assert len(current_support_arguments(c)) == 2
    assert c.model_dump_json() == before
    assert OutputAssembler().build_mechanism_predictions(c.touch(mechanism_support_arguments=[]))[0].evidence == []


def test_unknown_argument_sources_are_rejected_not_silently_dropped():
    c = ready_case().touch(mechanism_support_arguments=[support("S1", "foreign")])
    with pytest.raises(ValueError, match="unknown source"):
        OutputAssembler().build_mechanism_predictions(c)


def test_structural_argument_survives_without_a_tool_call(tmp_path):
    c = initial_case()
    payload = public_payload(c, REGISTRY, stage="planner_assimilate")
    data = assimilation(payload)
    data["mechanism_support_arguments"] = [support("S1", "input:smiles").model_dump()]
    c = apply_assimilation(c, QuestionAssimilation.model_validate(data))
    assert c.evidence_ledger.items == []
    assert OutputAssembler().build_mechanism_predictions(c)[0].evidence_refs == ["input:smiles"]
    packet = build_tool_source_packet(tmp_path, ["input:smiles", "foreign"])
    assert packet.unresolved_evidence_refs == ["foreign"]
    assert packet.tool_records == [] and packet.tool_action_count == 0
    from mechcal.runtime.questions import decision_from_action
    assert DeterministicGate().apply(decision_from_action(action(c.case_id, 1), c, "R001"), c).status == "finalize"


@pytest.mark.parametrize("cid,valid", [
    ("microscopic.run_frontier_orbital_partition", {"donor_atoms": [0], "acceptor_atoms": [1]}),
    ("microscopic.run_torsion_snapshots", {"dihedral_atoms": [0, 1, 2, 3], "angles": [-90, 90]}),
])
def test_omitted_selections_get_actionable_feedback_without_changing_state(cid, valid):
    c = ready_case()
    before = c.model_dump_json()
    a = action(c.case_id, 1, dispatch=True)
    a.dispatch_requests = [request(cid=cid)]
    with pytest.raises(ValueError, match="explicit measurement selection.*") as error:
        validate_action(a, c, REGISTRY)
    assert "list_rotatable_dihedrals" in str(error.value)
    assert c.model_dump_json() == before
    a.dispatch_requests[0].tool_args = valid
    validate_action(a, c, REGISTRY)
    assert c.model_dump_json() == before


@pytest.mark.parametrize("cid,valid", [
    ("microscopic.run_frontier_orbital_partition", {"donor_atoms": [0], "acceptor_atoms": [1]}),
    ("microscopic.run_torsion_snapshots", {"dihedral_id": "dih:001", "angles": [-90, 90]}),
])
def test_planner_schema_retry_repairs_selection_without_execution(tmp_path, cid, valid):
    o, _ = loop(tmp_path)
    c = ready_case()
    c = c.touch(runtime={**c.runtime, "tool_environment": {"amesp_enabled": True}})
    calls = []
    class Client:
        def is_configured(self):
            return True
        def complete_json(self, *, payload, system_prompt, schema_feedback=None):
            calls.append(schema_feedback)
            a = action(c.case_id, 1, dispatch=True)
            a.dispatch_requests = [request(cid=cid, args=valid if len(calls) > 1 else {})]
            return a.model_dump(mode="json")
    o.deps.planner.llm_client = Client()
    before = c.model_dump_json()
    decision = o.deps.planner.plan_initial(c, round_id="R001")
    assert len(calls) == 2 and "explicit measurement selection" in calls[1]
    assert decision.dispatch_requests[0].tool_args == valid
    assert c.model_dump_json() == before


def test_question_loop_card_exposes_the_same_required_fields():
    cards = {c["capability_id"]: c for c in REGISTRY.cards()}
    schema = cards["microscopic.run_frontier_orbital_partition"]["question_loop_arg_schema"]
    assert set(schema["required"]) == {"donor_atoms", "acceptor_atoms"}
    assert cards["microscopic.run_torsion_snapshots"]["question_loop_arg_schema"]["required"] == ["angles"]
