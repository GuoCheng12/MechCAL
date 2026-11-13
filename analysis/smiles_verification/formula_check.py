"""Check a case's SMILES against the molecular formula and mass data printed in its source papers.

A formula or high-resolution mass match confirms composition only, not connectivity (isomers
share both). It is the first of several independent checks; see README.md in this folder.

Usage:
    python analysis/smiles_verification/formula_check.py --cases dev --text-dir <dir> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "benchmarks/photomechbench/raw_v2_119/cases"
SPLIT = ROOT / "benchmarks/photomechbench/splits/raw_v2_119_dev_test.json"

ELECTRON = 0.000548580
ADDUCTS = {"M": 0.0, "M+H": 1.007276, "M+Na": 22.989218, "M+K": 38.963158, "M-H": -1.007276,
           "M+•": -ELECTRON, "M-•": ELECTRON, "M+NH4": 18.033823}
FORMULA = re.compile(r"(?<![A-Za-z])C\s?\d{1,3}\s?H\s?\d{1,3}(?:\s?(?:Br|Cl|Na|Fe|Au|Pt|Ir|Cu|Zn|Si|Se|Sn|B|N|O|F|S|P|I|K)\s?\d{0,3})*")
MASS_LINE = re.compile(r"HRMS|HR-MS|MALDI|ESI|EI-MS|FAB|m/z|calcd|Calcd|found|Found|\[M", re.I)
NUMBER = re.compile(r"(?<![\d.])(\d{2,4}\.\d{1,5})(?![\d.])")


def hill(formula: str) -> str:
    """Normalise a formula string to Hill order without spaces or charge."""
    counts: dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula.replace(" ", "")):
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    order = ["C", "H"] + sorted(k for k in counts if k not in ("C", "H"))
    return "".join(f"{el}{counts[el] if counts[el] != 1 else ''}" for el in order if el in counts)


def smiles_identity(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    neutral = hill(re.sub(r"[+-]\d*$", "", formula))
    charge = Chem.GetFormalCharge(mol)
    mass = ExactMolWt(mol)                    # includes electron mass correction for charged species
    return {"formula": formula, "formula_hill": neutral, "charge": charge, "monoisotopic_mass": round(mass, 5),
            "adducts": {k: round(mass + v, 5) for k, v in ADDUCTS.items()} if charge == 0 else
                       {f"M{'+' if charge > 0 else '-'}{abs(charge)}(m/z)": round(mass / abs(charge), 5), "M": round(mass, 5)}}


def windows(text: str, name: str, width: int = 2500) -> list[str]:
    """Text windows around each occurrence of the compound designation (whole-word match)."""
    if not name:
        return []
    out = []
    for part in {p.strip() for p in re.split(r"\s*/\s*", name) if len(p.strip()) >= 2}:
        for m in re.finditer(rf"(?<![\w-]){re.escape(part)}(?![\w-])", text):
            out.append(text[max(0, m.start() - width): m.end() + width])
    return out


def scan(text: str) -> dict:
    formulas = {hill(f) for f in FORMULA.findall(text)}
    masses = []
    for line in text.splitlines():
        if MASS_LINE.search(line):
            masses.extend(float(x) for x in NUMBER.findall(line))
    return {"formulas": sorted(formulas), "masses": sorted(set(masses))}


def match_masses(identity: dict, masses: list[float]) -> list[dict]:
    hits = []
    for label, ref in identity["adducts"].items():
        for m in masses:
            ppm = (m - ref) / ref * 1e6
            decimals = len(str(m).split(".")[1])
            if decimals >= 3 and abs(ppm) <= 10:
                hits.append({"adduct": label, "reported": m, "calc_from_smiles": ref, "ppm": round(ppm, 1), "resolution": "high"})
            elif decimals <= 2 and abs(m - ref) <= 0.6:
                hits.append({"adduct": label, "reported": m, "calc_from_smiles": ref, "delta": round(m - ref, 2), "resolution": "nominal"})
    return hits


def check_case(case: dict, text_dir: Path) -> dict:
    h = case["hidden_reference"]
    smiles = case["public_input"]["molecule"]["structure"]["value"]
    ident = smiles_identity(smiles)
    texts = []
    for doc in h["source_documents"]:
        p = text_dir / (doc["doi"].replace("/", "_") + ".txt")
        texts.append(p.read_text(errors="ignore") if p.exists() else "")
    full = "\n".join(texts)
    near = "\n".join(w for t in texts for w in windows(t, h["target_name"]))
    near_scan, all_scan = scan(near), scan(full)
    target = ident["formula_hill"]
    near_masses = match_masses(ident, near_scan["masses"])
    all_masses = match_masses(ident, all_scan["masses"])
    if target in near_scan["formulas"] or any(m["resolution"] == "high" for m in near_masses):
        verdict = "composition_confirmed_near_name"
    elif target in all_scan["formulas"] or any(m["resolution"] == "high" for m in all_masses):
        verdict = "composition_found_elsewhere_in_paper"
    elif near_scan["formulas"]:
        verdict = "formula_conflict"               # the paper prints formulas near the name, none matches
    elif all_masses:
        verdict = "nominal_mass_only"
    else:
        verdict = "no_composition_data"
    return {"case_id": case["case_id"], "target_name": h["target_name"], "smiles": smiles, **ident,
            "name_found": bool(near), "verdict": verdict,
            "formulas_near_name": near_scan["formulas"][:12], "mass_hits_near_name": near_masses[:6],
            "mass_hits_anywhere": all_masses[:6],
            "formula_in_paper": target in all_scan["formulas"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="dev", help="dev, test, all, or comma-separated case ids")
    ap.add_argument("--text-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    split = json.loads(SPLIT.read_text())
    if a.cases == "dev":
        ids = [d["case_id"] for d in split["dev"]]
    elif a.cases == "test":
        ids = list(split["test"])
    elif a.cases == "all":
        ids = sorted(p.stem for p in CASES.glob("*.json"))
    else:
        ids = a.cases.split(",")
    rows = [check_case(json.loads((CASES / f"{i}.json").read_text()), a.text_dir) for i in ids]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for r in rows:
        print(f"{r['case_id'][-4:]}  {r['verdict']:<38} {r['formula_hill']:<16} name_found={r['name_found']}  "
              f"near={r['formulas_near_name'][:4]}")


if __name__ == "__main__":
    main()
