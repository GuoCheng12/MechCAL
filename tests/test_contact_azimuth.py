"""Coordinate integrity, not assertions about mechanism labels."""

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem

from mechcal.schemas.external_qc import ContactMotionArgs
from mechcal.tools.contact_motion import coordinate_preflight, evaluate, moving_side


def fixture_geometry():
    mol = Chem.RWMol()
    for symbol in ['C', 'C', 'Au', 'C', 'C', 'C', 'He']:
        mol.AddAtom(Chem.Atom(symbol))
    for i, j, kind in [(0, 4, Chem.BondType.SINGLE), (4, 1, Chem.BondType.SINGLE),
                       (1, 2, Chem.BondType.DATIVE), (2, 5, Chem.BondType.SINGLE),
                       (5, 3, Chem.BondType.SINGLE)]:
        mol.AddBond(i, j, kind)
    mol = mol.GetMol()
    Chem.GetSymmSSSR(mol)
    atoms = Atoms('CCAuCCCHe', positions=[[0, 1, 0], [0, 0, 0], [2, 0, 0],
        [3, 0, 1], [0, .1, 0], [3, 0, .01], [0, 0, 8]])
    args = ContactMotionArgs(source_artifact_id='a', coordinate_mode='fragment_azimuth',
        atom_indices=[0, 1, 2, 3], angles_degrees=[0., 60.])
    return mol, atoms, args


def test_dative_off_axis_coordinate_preserves_graph_and_partner():
    mol, atoms, args = fixture_geometry()
    moving, checks = coordinate_preflight(mol, atoms, args, list(range(6)))
    assert moving == [2, 3, 5]
    assert min(checks['reference_axis_sines']) > .05
    for angle in [-150., 0., 75.]:
        changed = atoms.copy()
        changed.set_dihedral(*args.atom_indices, angle, indices=moving)
        assert abs((changed.get_dihedral(*args.atom_indices)-angle+180)%360-180) < 1e-9
        assert np.array_equal(changed.positions[[0, 1, 4, 6]], atoms.positions[[0, 1, 4, 6]])
        for bond in mol.GetBonds():
            ids = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            assert changed.get_distance(*ids) == pytest.approx(atoms.get_distance(*ids), abs=1e-10)


def test_coordinate_is_rigid_frame_invariant():
    mol, atoms, args = fixture_geometry()
    rotated = atoms.copy()
    rotated.rotate(47, [1, 2, 3], center='COM')
    rotated.translate([5, 2, 1])
    moving, first = coordinate_preflight(mol, atoms, args, list(range(6)))
    _, second = coordinate_preflight(mol, rotated, args, list(range(6)))
    assert first['reference_lever_arms_angstrom'] == pytest.approx(second['reference_lever_arms_angstrom'])
    atoms.set_dihedral(*args.atom_indices, 45, indices=moving)
    rotated.set_dihedral(*args.atom_indices, 45, indices=moving)
    assert np.allclose(atoms.get_all_distances(), rotated.get_all_distances(), atol=1e-9)


@pytest.mark.parametrize('indices,mode', [([0, 1, 2, 3], 'bonded_dihedral'),
    ([0, 1, 2, 4], 'fragment_azimuth'), ([6, 1, 2, 3], 'fragment_azimuth'),
    ([0, 1, 2, 99], 'fragment_azimuth'), ([0, 1, 2, 2], 'fragment_azimuth'),
    ([0, 1, 2, 3], 'unknown')])
def test_invalid_graph_coordinates_rejected(indices, mode):
    mol, _, _ = fixture_geometry()
    with pytest.raises(ValueError):
        moving_side(mol, indices, list(range(6)), mode)


@pytest.mark.parametrize('smiles', ['C1CCCCC1', 'CC=CC', 'CC#CC'])
def test_azimuth_does_not_enable_ring_or_multiple_bond_rotation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    with pytest.raises(ValueError):
        moving_side(mol, [0, 1, 2, 3], list(range(mol.GetNumAtoms())), 'fragment_azimuth')


def test_nearly_collinear_anchor_rejected():
    mol, atoms, args = fixture_geometry()
    args = args.model_copy(update={'atom_indices': [4, 1, 2, 5]})
    with pytest.raises(ValueError, match='off-axis'):
        coordinate_preflight(mol, atoms, args, list(range(6)))


def test_legacy_default_still_bonded():
    args = ContactMotionArgs(source_artifact_id='a', atom_indices=[0, 1, 2, 3], angles_degrees=[0., 60.])
    assert args.coordinate_mode == 'bonded_dihedral'
    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    assert 3 in moving_side(mol, args.atom_indices, list(range(mol.GetNumAtoms())))


def test_invalid_coordinate_does_not_reach_energy_backend(tmp_path, monkeypatch):
    mol, atoms, args = fixture_geometry()
    path = tmp_path/'source.sdf'
    Chem.MolToMolFile(mol, str(path))
    args = args.model_copy(update={'atom_indices': [4, 1, 2, 5]})
    monkeypatch.setattr('mechcal.tools.contact_motion.contact_observables',
        lambda *a: pytest.fail('Invalid coordinate must not launch a calculation'))
    with pytest.raises(ValueError, match='off-axis'):
        evaluate(SimpleNamespace(workdir=tmp_path), atoms, args,
            SimpleNamespace(file_paths={'sdf': str(path)}), [list(range(6)), [6]])


def test_cost_and_usage_are_not_advertised_as_uniformly_cheap():
    from mechcal.capabilities.external_qc import external_capabilities
    from mechcal.tools.qc_process import QCConfig
    specs = {s.route: s for s in external_capabilities(QCConfig())}
    for route in ['estimate_soc_amesp', 'compare_spin_states_pyscf', 'sample_conformers_crest']:
        assert specs[route].cost == 'high'
        assert specs[route].metadata['tool_arg_hints']['compute_effort']
    assert specs['compare_contact_torsion_response'].cost == 'low'
    assert specs['search_dimer_contacts'].cost == 'medium'


def test_memory_catalog_preserves_expensive_route_warning():
    from mechcal.capabilities.registry import CapabilityRegistry
    from mechcal.capabilities.external_qc import external_capabilities
    from mechcal.capabilities.discovery import case_catalog
    from mechcal.schemas import ArtifactManifest
    from mechcal.tools.qc_process import QCConfig
    case = SimpleNamespace(runtime={'planner_case_memory': True}, artifact_manifest=ArtifactManifest(case_id='a'))
    rows = {r['capability_id']: r for r in case_catalog(CapabilityRegistry(external_capabilities(QCConfig())), case)}
    assert rows['microscopic.estimate_soc_amesp']['cost'] == 'high'
    assert rows['microscopic.sample_conformers_crest']['cost'] == 'high'
