"""Capability cards: complete, consistent with the tools, family-neutral, and applicability from SMILES only."""

import json
import re

import pytest

from mechcal.agent import cards as cardlib
from mechcal.agent.facts import structural_facts
from mechcal.agent.loop import AgentRun
from mechcal.agent.recipes import RECIPES
from mechcal.mechanism_pool import MECHANISM_POOL
from test_agent_loop import ByRole

# Family labels and their distinctive abbreviations; plain physical terms ("triplet", "radiative") are allowed.
FAMILY_WORDS = set(MECHANISM_POOL) | {"ESIPT", "TICT", "ICT", "AIE", "SOKR", "RACI", "RIM", "RIR", "RIV", "anti-Kasha"}


@pytest.fixture()
def run(tmp_path, monkeypatch):
    qc = tmp_path / "qc.json"
    qc.write_text(json.dumps({"enabled": True}))
    monkeypatch.setenv("MECHCAL_QC_CONFIG", str(qc))
    return AgentRun(ByRole([]), case_id="C", smiles="CN(C)c1ccc(cc1)C#N", run_dir=tmp_path / "r", enable_amesp=True)


def test_every_visible_tool_has_a_card_matching_its_arguments(run):
    shown = {c["tool"]: c for c in run.tool_cards}
    assert set(shown) == set(cardlib.visible_tools())
    for tool, card in cardlib.load_cards()["tools"].items():
        args = set(shown[tool]["args"]) - {"source_artifact_id"}
        assert set(card["you_can_vary"]) == args, (tool, set(card["you_can_vary"]) ^ args)


def test_cards_never_name_a_mechanism_family():
    text = json.dumps(cardlib.load_cards())
    found = sorted(w for w in FAMILY_WORDS if re.search(rf"\b{re.escape(w)}\b", text))
    assert not found, found


def test_every_recipe_has_a_card(run):
    recipes = cardlib.load_cards()["recipes"]
    assert set(RECIPES) <= set(recipes)
    for name in RECIPES:
        assert all(t in cardlib.visible_tools() for g in recipes[name]["requires_any"] for t in g)


@pytest.mark.parametrize("smiles, prepared, tool, ok", [
    ("CCCCCCCCCCOc1ccc([N+]#[C-]->[Au]c2c(F)c(F)c(F)c(F)c2F)cc1", "initial_structure", "microscopic.analyze_hole_electron_from_bundle", True),
    ("CCCCCCCCCCOc1ccc([N+]#[C-]->[Au]c2c(F)c(F)c(F)c(F)c2F)cc1", "initial_structure", "microscopic.compare_singlet_triplet_states", False),
    ("CCCCCCCCCCOc1ccc([N+]#[C-]->[Au]c2c(F)c(F)c(F)c(F)c2F)cc1", "initial_structure", "microscopic.run_conformer_bundle", False),
    ("C1=C(c2cc[c-]([Fe+2][c-]3cccc3)c2)Cc2ccccc21", "initial_structure", "microscopic.analyze_hole_electron_from_bundle", False),
    ("C1=C(c2cc[c-]([Fe+2][c-]3cccc3)c2)Cc2ccccc21", "initial_structure", "microscopic.compare_spin_states_pyscf", True),
    ("CC[n+]1ccc(C=Cc2ccccc2)cc1", "prepared_structure", "microscopic.analyze_ground_state_vibrations", False),
    ("CC[n+]1ccc(C=Cc2ccccc2)cc1", "prepared_structure", "microscopic.compare_spin_states_pyscf", True),
    ("Oc1ccccc1-c1nc2ccccc2s1", "prepared_structure", "microscopic.scan_cis_excitation_coordinate", False),
    ("Oc1ccccc1-c1cn2ccccc2n1", "prepared_structure", "microscopic.scan_cis_excitation_coordinate", True),
])
def test_applicability_uses_only_what_the_smiles_implies(smiles, prepared, tool, ok):
    profile = cardlib.molecule_profile(structural_facts(smiles), prepared)
    verdict = cardlib.applicability(tool, profile)
    assert verdict["ok"] is ok and (ok or verdict["why"])


def test_planner_sees_full_cards_with_applicability_and_cost(run):
    catalog = {c["tool"]: c for c in run._professor_catalog()}
    card = catalog["microscopic.compare_contact_torsion_response"]
    assert card["computes"].startswith("The same rigid rotation") and "does_not_tell" in card
    assert card["you_can_vary"]["atom_indices"]["class"] == "scientific"
    assert card["applies_to_this_molecule"]["ok"] and "cost" in card
    assert catalog["microscopic.analyze_hole_electron_from_bundle"]["input_ready"].startswith("needs amesp_baseline_bundle")
