"""Second, automatic structure check: MolScribe reads each cropped source figure and the result is
compared with the case SMILES (and any proposed correction) by InChIKey.

Runs in its own environment (MolScribe needs torch 1.x), not in the project venv:
    /home/devbox/miniconda3/envs/mechcal-ocsr/bin/python analysis/smiles_verification/ocsr_check.py \
        --crops <crops.json> --checkpoint <swin_base_char_aux_1m680k.pth> --out <file.json>

crops.json: [{"case_id", "image", "case_smiles", "proposed_smiles" (optional)}]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from molscribe import MolScribe
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def inchikey(smiles: str | None) -> str | None:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
    try:
        return Chem.MolToInchiKey(mol) or None
    except Exception:  # noqa: BLE001
        return None


def formula(smiles: str | None) -> str | None:
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return CalcMolFormula(mol) if mol else None


def main_fragment(smiles: str | None) -> str | None:
    """Largest fragment, with wildcard atoms (labels such as 'H26' read as R groups) replaced by H.

    Figure crops often carry stray text (labels, counter-ions, neighbouring drawings) that the
    recogniser turns into extra fragments; the compound itself is the largest fragment.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in sorted((a for a in rw.GetAtoms() if a.GetAtomicNum() == 0), key=lambda a: -a.GetIdx()):
        rw.RemoveAtom(atom.GetIdx())
    try:
        clean = rw.GetMol()
        Chem.SanitizeMol(clean)
        frags = Chem.GetMolFrags(clean, asMols=True)
    except Exception:  # noqa: BLE001 - unreadable recognition
        return None
    big = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(big)


def compare(a: str | None, b: str | None) -> str:
    if not a or not b:
        return "no_key"
    if a == b:
        return "same_structure"
    if a[:14] == b[:14]:
        return "same_connectivity"
    return "different"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    torch.set_num_threads(16)
    model = MolScribe(str(a.checkpoint), device=torch.device("cpu"))
    rows = []
    for item in json.loads(a.crops.read_text()):
        pred = model.predict_image_file(item["image"], return_atoms_bonds=False, return_confidence=True)
        smi = pred.get("smiles")
        k = inchikey(smi)
        row = {"case_id": item["case_id"], "image": item["image"], "ocsr_smiles": smi,
               "ocsr_formula": formula(smi), "confidence": round(float(pred.get("confidence") or 0), 3),
               "vs_case": compare(k, inchikey(item["case_smiles"])), "case_formula": formula(item["case_smiles"])}
        main = main_fragment(smi)
        km = inchikey(main)
        row["ocsr_main_fragment"] = main
        row["main_vs_case"] = compare(km, inchikey(main_fragment(item["case_smiles"])))
        if item.get("proposed_smiles"):
            row["vs_proposed"] = compare(k, inchikey(item["proposed_smiles"]))
            row["main_vs_proposed"] = compare(km, inchikey(main_fragment(item["proposed_smiles"])))
            row["proposed_formula"] = formula(item["proposed_smiles"])
        rows.append(row)
        print(f"{row['case_id'][-4:]}  main_vs_case={row['main_vs_case']:<17} "
              f"main_vs_proposed={row.get('main_vs_proposed', '-'):<17} conf={row['confidence']:<5} {main}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
