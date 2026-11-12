"""Recipes pass the settings they accept to the tools they run (offline, stub gateway)."""

from types import SimpleNamespace

from mechcal.agent.recipes import RecipeContext, run_recipe

M = "microscopic."


class StubGateway:
    def __init__(self, smiles, artifacts):
        self.smiles, self.calls, self.prepared_artifact_id = smiles, [], "structure:prepared"
        self.case_run = SimpleNamespace(artifact_manifest=SimpleNamespace(artifacts=[
            SimpleNamespace(artifact_id=a, kind=k, status="available", metadata=m) for a, k, m in artifacts]))

    def derived_from(self, aid, root):
        return aid == root or aid.startswith(root)

    def call(self, tool, args):
        self.calls.append((tool, dict(args)))
        out = {"status": "success", "reusable_artifacts": [], "labels": {}}
        if tool.endswith("run_baseline_bundle"):
            out["reusable_artifacts"] = [args["source_artifact_id"] + ":bundle"]
            self.case_run.artifact_manifest.artifacts.append(SimpleNamespace(
                artifact_id=args["source_artifact_id"] + ":bundle", kind="amesp_baseline_bundle", status="available", metadata={}))
        return out


def test_torsion_recipe_accepts_dihedral_atoms():
    gw = StubGateway("CN(C)c1ccc(cc1)C#N", [("structure:prepared", "prepared_structure", {})])
    assert run_recipe(RecipeContext(gw), "torsion_response", {"dihedral_atoms": [0, 1, 3, 4], "angles": [0, 90]}).status == "success"
    tool, args = gw.calls[-1]
    assert tool == M + "run_torsion_brightness_coupling_scan" and args["dihedral_atoms"] == [0, 1, 3, 4]
    assert "dihedral_id" not in args


def test_spin_recipe_uses_the_dft_route_when_a_functional_is_given():
    gw = StubGateway("CN(C)c1ccc(cc1)C#N", [("structure:prepared", "prepared_structure", {})])
    run_recipe(RecipeContext(gw), "spin_states", {})
    assert gw.calls[0][0] == M + "compare_singlet_triplet_states"                 # CHNO default: TD-DFTB
    gw2 = StubGateway("CN(C)c1ccc(cc1)C#N", [("structure:prepared", "prepared_structure", {})])
    run_recipe(RecipeContext(gw2), "spin_states", {"functional": "pbe0"})
    assert gw2.calls[0][0] == M + "compare_spin_states_pyscf" and gw2.calls[0][1]["functional"] == "pbe0"


def test_dimer_recipe_takes_custom_fragments_or_the_two_molecules():
    dimer = ("geometry:x:dimer_01", "prepared_structure", {"modeled_dimer": True, "fragment_atom_indices": [[0, 1], [2, 3]]})
    gw = StubGateway("CC", [("structure:prepared", "prepared_structure", {}), dimer])
    run_recipe(RecipeContext(gw), "dimer_excited_states", {"source_artifact_id": "geometry:x:dimer_01"})
    assert gw.calls[-1][1]["fragments"] == {"molecule_1": [0, 1], "molecule_2": [2, 3]}
    gw2 = StubGateway("CC", [("structure:prepared", "prepared_structure", {}), dimer])
    run_recipe(RecipeContext(gw2), "dimer_excited_states", {"source_artifact_id": "geometry:x:dimer_01",
                                                            "fragments": {"donor": [0], "acceptor": [3]}})
    assert gw2.calls[-1][1]["fragments"] == {"donor": [0], "acceptor": [3]}
