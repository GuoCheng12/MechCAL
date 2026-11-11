"""Deterministic structural facts from the verified SMILES (RDKit, 1-based atom ids).

These replace the 0.2.0 "macro proxy" tools. They are facts about the input graph, given to
the agent once at the start; each fact carries atom ids and, where possible, a structural
assertion that the evaluator can re-check. They are not mechanism verdicts.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

METALS = {"Li", "Na", "K", "Mg", "Ca", "Al", "Zn", "Cu", "Ag", "Au", "Pt", "Pd", "Ir", "Ru", "Os", "Rh",
          "Re", "Fe", "Co", "Ni", "Mn", "Cr", "Ti", "Zr", "Hf", "Cd", "Hg", "Ga", "In", "Sn", "Pb", "Bi",
          "Eu", "Tb", "Gd", "Sm", "Dy", "Yb", "Nd", "Er", "Tm", "Ce", "La", "Pr", "Ho", "Lu"}
HEAVY_ATOMS = {"Br", "I", "Se", "Te", "S", "P", "Si", "Ge", "As", "Sb"} | METALS

DONORS = {
    "dialkyl/diaryl amine N": "[NX3;H0,H1,H2;!$(N-C=[O,S,N]);!$(N-S(=O)=O);!$([N+])]([#6])",
    "ether/alkoxy O on aromatic": "[OX2;H0]([#6])c",
    "carbazole/indole-type N": "[nX3;H0,H1]",
    "thioether S on aromatic": "[SX2;H0]([#6])c",
    "phenoxide/hydroxyl O on aromatic": "[OX2H1,OX1-]c",
}
ACCEPTORS = {
    "nitrile": "[CX2]#[NX1]",
    "isocyanide": "[N+]#[C-]",
    "carbonyl (ketone/aldehyde/ester/amide/imide)": "[CX3]=[OX1]",
    "nitro": "[N+](=O)[O-]",
    "dicyanovinyl": "C=C(C#N)C#N",
    "pyridine-type N": "[nX2]",
    "pyridinium/azinium N+": "[n+,N+;!$([N+](=O)[O-]);!$([N+]#[C-])]",
    "three-coordinate boron": "[BX3]",
    "sulfonyl": "S(=O)(=O)",
    "perfluoroaryl": "c(F)c(F)c(F)",
    "imine C=N": "[CX3]=[NX2]",
}
ESIPT_DONOR = "[OX2H1,NX3H1,NX3H2]"
ESIPT_ACCEPTOR = "[$([NX2]),$([nX2]),$([OX1]=[#6])]"


def _conj(atom) -> bool:
    return atom.GetIsAromatic() or any(b.GetIsConjugated() for b in atom.GetBonds())


def _ids(match) -> list[int]:
    return [i + 1 for i in match]


def _matches(mol, smarts: str) -> list[tuple[int, ...]]:
    patt = Chem.MolFromSmarts(smarts)
    return list(mol.GetSubstructMatches(patt)) if patt is not None else []


def _conjugated_components(mol) -> list[set[int]]:
    parent = list(range(mol.GetNumAtoms()))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    touched = set()
    for bond in mol.GetBonds():
        if bond.GetIsConjugated() or bond.GetIsAromatic():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            parent[find(a)] = find(b)
            touched.update((a, b))
    groups: dict[int, set[int]] = {}
    for i in touched:
        groups.setdefault(find(i), set()).add(i)
    return sorted(groups.values(), key=len, reverse=True)


def structural_facts(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES could not be parsed")
    facts: list[dict] = []

    def add(kind: str, text: str, atoms: list[int], assertion: dict | None = None):
        facts.append({"fact_id": f"F{len(facts) + 1}", "kind": kind, "text": text, "atom_ids": atoms,
                      **({"assertion": assertion} if assertion else {})})

    symbols = sorted({a.GetSymbol() for a in mol.GetAtoms()})
    charges = [(a.GetIdx() + 1, a.GetSymbol(), a.GetFormalCharge()) for a in mol.GetAtoms() if a.GetFormalCharge()]
    summary = {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "atoms_with_h": Chem.AddHs(mol).GetNumAtoms(),
        "net_charge": Chem.GetFormalCharge(mol),
        "elements": symbols,
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "mol_logp": round(Descriptors.MolLogP(mol), 2),
        "fragments": len(Chem.GetMolFrags(mol)),
    }
    for idx, sym, q in charges:
        add("formal_charge", f"Atom {idx} ({sym}) carries formal charge {q:+d}.", [idx])
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in METALS:
            nbrs = [n.GetIdx() + 1 for n in atom.GetNeighbors()]
            add("metal_center", f"Metal {atom.GetSymbol()} (atom {atom.GetIdx() + 1}) bonded to atoms {nbrs}.",
                [atom.GetIdx() + 1, *nbrs], {"type": "element", "atoms": [atom.GetIdx() + 1], "element": atom.GetSymbol()})
        elif atom.GetSymbol() in HEAVY_ATOMS - {"S", "P", "Si"}:
            add("heavy_atom", f"Heavy atom {atom.GetSymbol()} at atom {atom.GetIdx() + 1}.", [atom.GetIdx() + 1],
                {"type": "element", "atoms": [atom.GetIdx() + 1], "element": atom.GetSymbol()})

    components = _conjugated_components(mol)
    for n, comp in enumerate(components[:4], start=1):
        if len(comp) >= 6:
            add("conjugated_system", f"Conjugated system {n}: {len(comp)} atoms.", sorted(i + 1 for i in comp))

    donor_sites, acceptor_sites = [], []
    seen_donor: set[int] = set()
    for name, smarts in DONORS.items():
        for m in _matches(mol, smarts):
            atom = m[0]
            if atom in seen_donor:
                continue
            seen_donor.add(atom)
            if _conj(mol.GetAtomWithIdx(atom)) or any(_conj(nb) for nb in mol.GetAtomWithIdx(atom).GetNeighbors()):
                donor_sites.append((name, atom))
                add("donor_site", f"Electron-donor site: {name} at atom {atom + 1}.", [atom + 1])
    for name, smarts in ACCEPTORS.items():
        covered: set[int] = set()
        for m in _matches(mol, smarts):
            if covered & set(m):
                continue
            if name == "perfluoroaryl":  # report the whole ring once
                ring = next((r for r in mol.GetRingInfo().AtomRings() if m[0] in r), m)
                m = tuple(ring)
            covered.update(m)
            acceptor_sites.append((name, m[0]))
            add("acceptor_site", f"Electron-acceptor site: {name} at atoms {_ids(m)}.", _ids(m))
    linked: set[tuple[int, int]] = set()
    for dname, d in donor_sites:
        for aname, a in acceptor_sites:
            if d == a or (d, a) in linked:
                continue
            linked.add((d, a))
            path = Chem.GetShortestPath(mol, d, a)
            if len(path) < 3:
                continue
            conjugated = all((b := mol.GetBondBetweenAtoms(x, y)).GetIsConjugated() or b.GetIsAromatic()
                             for x, y in zip(path[1:-1], path[2:-1])) if len(path) > 3 else True
            if conjugated:
                add("donor_acceptor_path",
                    f"{dname} (atom {d + 1}) and {aname} (atom {a + 1}) are linked through a conjugated path of {len(path) - 1} bonds.",
                    [d + 1, a + 1])

    rot = Chem.MolFromSmarts("[!$([NH])&!$([NH2])&!D1&!$(*#*)]-&!@[!$([NH])&!$([NH2])&!D1&!$(*#*)]")
    rotors = []
    for a, b in mol.GetSubstructMatches(rot):
        aa, bb = mol.GetAtomWithIdx(a), mol.GetAtomWithIdx(b)
        if _conj(aa) or _conj(bb):
            rotors.append((a, b))
    for a, b in rotors[:12]:
        add("conjugated_rotor", f"Rotatable single bond {a + 1}-{b + 1} attached to the conjugated framework.",
            [a + 1, b + 1], {"type": "rotatable_bond", "atoms": [a + 1, b + 1]})
    if len(rotors) > 12:
        add("conjugated_rotor_count", f"{len(rotors)} rotatable bonds attach to the conjugated framework (first 12 listed).", [])

    for m in _matches(mol, "[CX3](c)(c)=[CX3]"):
        olefin = [m[0] + 1, m[3] + 1]
        aryls = sum(1 for x in olefin for nb in mol.GetAtomWithIdx(x - 1).GetNeighbors() if nb.GetIsAromatic())
        if aryls >= 3:
            add("aryl_olefin", f"C=C bond {olefin[0]}={olefin[1]} carries {aryls} aryl substituents (propeller-like rotors).",
                olefin, {"type": "bonded", "atoms": olefin})
            break

    seen = set()
    for d in _matches(mol, ESIPT_DONOR):
        for a in _matches(mol, ESIPT_ACCEPTOR):
            path = Chem.GetShortestPath(mol, d[0], a[0])
            if len(path) == 5 and (d[0], a[0]) not in seen:  # D-X-Y-Z-A: with the H, a six-membered H-bond ring
                seen.add((d[0], a[0]))
                add("intramolecular_hbond_motif",
                    f"H-bond donor atom {d[0] + 1} and acceptor atom {a[0] + 1} are 4 bonds apart (possible six-membered intramolecular H-bond ring).",
                    [d[0] + 1, a[0] + 1])

    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) >= 12:
            add("macrocycle", f"Macrocyclic ring of {len(ring)} atoms (possible cavity/host).", sorted(i + 1 for i in ring),
                {"type": "in_ring", "atoms": [ring[0] + 1]})
    for m in _matches(mol, "[CH2][CH2][CH2][CH2][CH2][CH2]"):
        add("long_alkyl_chain", f"Long alkyl chain starting near atom {m[0] + 1} (amphiphilic/packing-relevant).", _ids(m))
        break
    return {"source": "rdkit_deterministic", "rdkit_version": Chem.rdBase.rdkitVersion, "smiles": smiles,
            "mapped_smiles": _mapped(mol), "summary": summary, "facts": facts,
            "scope": "Graph facts only: no geometry, excited states, packing or measurements."}


def _mapped(mol) -> str:
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(copy, canonical=False)
