from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmilesFeatures:
    length: int
    aromatic_atom_count: int
    hetero_atom_count: int
    branch_point_count: int
    double_bond_count: int
    ring_digit_count: int
    donor_acceptor_proxy: int
    flexibility_proxy: float
    conjugation_proxy: float


def extract_smiles_features(smiles: str) -> SmilesFeatures:
    return _rdkit_smiles_features(smiles) or _fallback_string_features(smiles)


def scaffold_proxy(smiles: str) -> str:
    features = extract_smiles_features(smiles)
    return (
        f"aro{features.aromatic_atom_count}-"
        f"het{features.hetero_atom_count}-"
        f"br{features.branch_point_count}-"
        f"db{features.double_bond_count}"
    )


def _fallback_string_features(smiles: str) -> SmilesFeatures:
    aromatic_atom_count = smiles.count("c")
    hetero_atom_count = sum(smiles.count(symbol) for symbol in "NOSPnos")
    branch_point_count = smiles.count("(")
    double_bond_count = smiles.count("=")
    ring_digit_count = sum(character.isdigit() for character in smiles)
    donor_acceptor_proxy = int(any(symbol in smiles for symbol in "NO")) + int("S" in smiles)
    flexibility_proxy = round(branch_point_count + max(len(smiles) / 18.0, 0.5), 3)
    conjugation_proxy = round(double_bond_count + aromatic_atom_count * 0.35, 3)
    return SmilesFeatures(
        length=len(smiles),
        aromatic_atom_count=aromatic_atom_count,
        hetero_atom_count=hetero_atom_count,
        branch_point_count=branch_point_count,
        double_bond_count=double_bond_count,
        ring_digit_count=ring_digit_count,
        donor_acceptor_proxy=donor_acceptor_proxy,
        flexibility_proxy=flexibility_proxy,
        conjugation_proxy=conjugation_proxy,
    )


def _rdkit_smiles_features(smiles: str) -> SmilesFeatures | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Lipinski
    except ModuleNotFoundError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    mol = canonical or mol
    aromatic_atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    hetero_atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    branch_point_count = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
    double_bond_count = sum(
        1
        for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 2.0 and not bond.GetIsAromatic()
    )
    donor_count = int(Lipinski.NumHDonors(mol))
    acceptor_count = int(Lipinski.NumHAcceptors(mol))
    aromatic_bond_count = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
    return SmilesFeatures(
        length=len(smiles),
        aromatic_atom_count=aromatic_atom_count,
        hetero_atom_count=hetero_atom_count,
        branch_point_count=branch_point_count,
        double_bond_count=double_bond_count,
        ring_digit_count=max(mol.GetRingInfo().NumRings() * 2, 0),
        donor_acceptor_proxy=int(bool(donor_count or acceptor_count))
        + int(bool(donor_count and acceptor_count)),
        flexibility_proxy=round(branch_point_count + max(mol.GetNumAtoms() / 18.0, 0.5), 3),
        conjugation_proxy=round(
            double_bond_count + aromatic_bond_count * 0.2 + aromatic_atom_count * 0.15,
            3,
        ),
    )
