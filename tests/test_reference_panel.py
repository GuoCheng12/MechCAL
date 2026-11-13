"""Reference panel: descriptor extraction by tool, percentile context, and the professor's report field."""

import json

import pytest

from mechcal.agent import reference
from mechcal.agent import supervision as sv
from mechcal.agent.loop import AgentConfig, AgentRun
from test_agent_loop import SMILES, ByRole

M = "microscopic."
BUNDLE = {"mulliken_charge_count": 22, "excited_states": [{"excitation_energy_ev": 4.3, "oscillator_strength": 0.06},
                                                          {"excitation_energy_ev": 4.32, "oscillator_strength": 1.5}],
          "state_descriptors": [{"state_index": 1, "radiative_rate_estimate_per_s": 4.6e7, "gap_to_higher_state_ev": 0.02}],
          "bright_state": {"oscillator_strength": 1.5}}


def test_bundle_descriptors_separate_molecule_and_dimer():
    mono = reference.extract(M + "run_baseline_bundle", BUNDLE, natoms=22)
    assert mono == {"s1_excitation_ev": 4.3, "s1_oscillator_strength": 0.06, "s1_radiative_rate_per_s": 4.6e7,
                    "s2_minus_s1_ev": 0.02, "max_oscillator_strength": 1.5}
    dimer = reference.extract(M + "run_baseline_bundle", {**BUNDLE, "mulliken_charge_count": 44}, natoms=22, monomer_s1=4.5)
    assert set(dimer) == {"dimer_s1_oscillator_strength", "dimer_s2_minus_s1_ev", "dimer_s1_shift_ev"}
    assert dimer["dimer_s1_shift_ev"] == pytest.approx(-0.2)


def test_other_tools_and_missing_values():
    he = {"hole_electron_spatial_analysis": {"state_index": 1, "centroid_distance_angstrom": 5.1, "spatial_overlap_sr": 0.36,
                                             "t_index_angstrom": 3.3, "fragment_pair": {"net_transfer_first_to_second_percent": -48.7}}}
    assert reference.extract(M + "analyze_hole_electron_from_bundle", he, natoms=22)["he_centroid_distance_angstrom"] == 5.1
    assert reference.extract(M + "analyze_hole_electron_from_bundle", he, natoms=22, source_atoms=44) == {
        "dimer_net_transfer_abs_percent": 48.7}
    s2 = {"hole_electron_spatial_analysis": {**he["hole_electron_spatial_analysis"], "state_index": 2}}
    assert reference.extract(M + "analyze_hole_electron_from_bundle", s2, natoms=22) == {}   # panel holds S1 only
    torsion = {"route_summary": {"energy_range_hartree": 0.01, "oscillator_strength_range": 0.28}}
    assert reference.extract(M + "run_torsion_brightness_coupling_scan", torsion, natoms=22)["torsion_s0_energy_range_ev"] == \
        pytest.approx(0.2721, abs=1e-4)
    vib = {"all_frequencies_cm_minus1": [-30.0, 12.0, 40.0, 75.0]}
    assert reference.extract(M + "analyze_ground_state_vibrations", vib, natoms=22) == {
        "lowest_frequency_cm1": 12.0, "modes_below_50_cm1": 2.0}
    dimer = {"candidates": [{"interaction_energy_ev": -0.5}, {"interaction_energy_ev": -0.8}]}
    assert reference.extract(M + "search_dimer_contacts", dimer, natoms=22, heavy_atoms=8)["dimer_interaction_ev_per_heavy_atom"] == -0.1
    soc = {"couplings": [{"singlet": 0, "triplet": 1, "rss_cm_inverse": 9.0}, {"singlet": 1, "triplet": 2, "rss_cm_inverse": 0.4}]}
    assert reference.extract(M + "evaluate_spin_orbit_couplings", soc, natoms=22) == {"max_s1_triplet_soc_cm1": 0.4}
    assert reference.extract(M + "run_baseline_bundle", {"excited_states": []}, natoms=22) == {}
    assert reference.extract("macro.anything", {"x": 1}, natoms=22) == {}


def _panel(n=10):
    return {"molecules": [{"name": f"m{i}", "values": {"s1_oscillator_strength": i / 10}} for i in range(n)]}


def test_context_gives_percentile_median_and_iqr():
    rows = reference.context({"s1_oscillator_strength": 0.75, "unknown": 1.0}, _panel())
    assert len(rows) == 1
    r = rows[0]
    assert r["panel_percentile"] == 80 and r["panel_median"] == 0.45 and r["panel_n"] == 10
    assert r["panel_iqr"] == [pytest.approx(0.225), pytest.approx(0.675)]
    assert reference.context({"s1_oscillator_strength": 0.75}, _panel(5)) == []      # too few panel values
    assert reference.context({"s1_oscillator_strength": 0.75}, None) == [] or reference.load_panel() is not None


def test_report_carries_panel_context_for_its_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "load_panel", lambda *a, **k: _panel())
    run = AgentRun(ByRole([]), case_id="CASE", smiles=SMILES, run_dir=tmp_path, enable_amesp=False,
                   config=AgentConfig(critic=False))
    run.gateway.reference_values["T001"] = {"s1_oscillator_strength": 0.05}
    task = sv.Task(task_id="W1", objective="q", recipe="excited_states", tools=[], fixed_args={}, separates=[], expectation={},
                   created_step=1)
    task.attempts.append(sv.Attempt(n=1, kind="recipe", name="excited_states", args={}, outcome="success", artifact_ids=["T001"]))
    report = run._report(task, {"status": "done", "summary": "ok", "claims": []}, 1)
    assert report["reference_panel"]["T001"][0]["panel_percentile"] == 10
    json.dumps(report)


def test_every_role_gets_the_same_neutral_panel_description():
    """The panel is described, not interpreted: no prompt tells a role what counts as typical."""
    from pathlib import Path
    import re
    from mechcal.eval.grounded_subjects import REACT_PROTOCOL
    root = Path(__file__).resolve().parents[1] / "mechcal/prompts"
    texts = {name: (root / name).read_text() for name in ("agent_professor.md", "agent_planner.md", "agent_worker.md")}
    texts["react"] = REACT_PROTOCOL
    flat = {k: re.sub(r"\s+", " ", v) for k, v in texts.items()}
    core = ("percentile, median and interquartile range among 43 common organic chromophores (aromatics, dyes and "
            "other widely studied luminescent or absorbing molecules, chosen as common examples and not to resemble "
            "the molecules studied here; isolated molecules, no metal complexes), computed with the same tool and "
            "default settings. The panel has no mechanism labels.")
    for name, text in flat.items():
        assert core in text, name
        assert not re.search(r"interquartile range is typical|[Nn]ear the median|percentile exists, use it", text), name
