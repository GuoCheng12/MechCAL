"""Warnings for SMILES patterns that were systematically wrong in the original benchmark records.

The SMILES verification against the source papers (2026-11-13) found two recurring errors in
raw_v2_119: every pyrene-containing case was written with a pyren-2-yl substituent (the papers
have pyren-1-yl), and every chrysene skeleton was wrong (pyrene, benz[a]anthracene or
benzo[a]pyrene had been written as chrysene or picene). These checks flag the patterns for a
person to compare with the source; they do not decide anything. A flagged structure can be right.
"""

from __future__ import annotations

from functools import lru_cache

PYREN_2_YL = "pyrene substituted at C2/C7 (pyren-2-yl); compare with the source, pyren-1-yl was meant in every earlier case"
CHRYSENE = "chrysene-type skeleton outside a pyrene system; compare with the source, it replaced pyrene or benz[a]anthracene before"


@lru_cache(maxsize=2)
def _patterns():
    from rdkit import Chem
    return (Chem.MolFromSmarts("c1cc2ccc3cccc4ccc(c1)c2c34"), Chem.MolFromSmarts("c1ccc2c(c1)ccc1c3ccccc3ccc21"))


def structure_warnings(smiles: str) -> list[str]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["SMILES does not parse"]
    pyrene, chrysene = _patterns()
    out = set()
    pyrene_matches = mol.GetSubstructMatches(pyrene)
    for match in pyrene_matches:
        core = set(match)
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            ring_nbrs = [n for n in atom.GetNeighbors() if n.GetIdx() in core]
            # C2/C7: two pyrene neighbours, neither of them a fusion atom (fusion atoms have three)
            if len(ring_nbrs) == 2 and all(sum(x.GetIdx() in core for x in n.GetNeighbors()) == 2 for n in ring_nbrs):
                if any(n.GetIdx() not in core and not mol.GetBondBetweenAtoms(idx, n.GetIdx()).IsInRing()
                       for n in atom.GetNeighbors()):
                    out.add(PYREN_2_YL)
    if not pyrene_matches and mol.HasSubstructMatch(chrysene):
        out.add(CHRYSENE)
    return sorted(out)
