"""Match each case SMILES against all structures recognised in its source papers (paper_structures.py).

Per case:
  found_exact          the case structure (salt/charge parent) is among the recognised structures
  found_connectivity   same skeleton, stereo or charge differs
  label_disagrees      a structure carrying the case's designation label was recognised, and it differs
  not_found            no recognised structure matches; the nearest one is reported with its similarity
  no_structures        nothing usable was recognised in the papers

Only public input, the compound designation and the source documents of a case are read; reference
mechanisms are not.

Usage:
    python analysis/smiles_verification/match_structures.py --cases test --structures <dir> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "benchmarks/photomechbench/raw_v2_119/cases"
CORRECTED = ROOT / "benchmarks/photomechbench/raw_v2_119_corrected/cases"
SPLIT = ROOT / "benchmarks/photomechbench/splits/raw_v2_119_dev_test.json"
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def parent(smiles: str | None):
    if not smiles or "*" in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.ChargeParent(mol)
    except Exception:  # noqa: BLE001
        return None
    heavy = mol.GetNumHeavyAtoms()
    if heavy < 5 or heavy > 250 or mol.GetNumAtoms() == 0:
        return None
    if not any(a.GetIsAromatic() or a.GetSymbol() not in ("C",) for a in mol.GetAtoms()):
        return None                                   # saturated pure-carbon cages: recognition noise
    return mol


def norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("ꞏ", ""))


def designations(target: str) -> set[str]:
    parts = {p.strip() for p in re.split(r"\s*/\s*", target or "") if p.strip()}
    out = set()
    for p in parts:
        n = norm_label(re.sub(r"^(compound|complex|probe)\s+", "", p, flags=re.I))
        if n:
            out.add(n)
    return out


def load_structures(struct_dir: Path, stems: list[str]) -> tuple[list[dict], list[str]]:
    mols, missing = [], []
    for stem in stems:
        f = struct_dir / f"{stem}.json"
        if not f.exists():
            missing.append(stem)
            continue
        data = json.loads(f.read_text())
        for page in data.get("pages", []):
            for m in page["molecules"]:
                mol = parent(m.get("smiles"))
                if mol is None:
                    continue
                mols.append({"paper": stem, "page": page["page"], "smiles": Chem.MolToSmiles(mol),
                             "key": Chem.MolToInchiKey(mol), "fp": FPGEN.GetFingerprint(mol),
                             "labels": [norm_label(t) for t in m.get("labels", []) if t]})
    return mols, missing


def check_case(case: dict, struct_dir: Path) -> dict:
    h = case["hidden_reference"]
    smiles = case["public_input"]["molecule"]["structure"]["value"]
    stems = [d["doi"].replace("/", "_") for d in h["source_documents"]]
    mols, missing = load_structures(struct_dir, stems)
    row = {"case_id": case["case_id"], "target_name": h["target_name"], "smiles": smiles,
           "n_structures": len(mols), "papers_missing": missing}
    pm = parent(smiles)
    if pm is None:
        row["verdict"] = "case_smiles_unparsable"
        return row
    key, fp = Chem.MolToInchiKey(pm), FPGEN.GetFingerprint(pm)
    if not mols:
        row["verdict"] = "no_structures"
        return row
    exact = [m for m in mols if m["key"] == key]
    conn = [m for m in mols if m["key"][:14] == key[:14]]
    desig = designations(h["target_name"])
    labelled = [m for m in mols if desig & set(m["labels"])]
    for m in mols:
        m["similarity"] = round(DataStructs.TanimotoSimilarity(fp, m["fp"]), 3)
    nearest = max(mols, key=lambda m: m["similarity"])
    view = lambda m: {k: m[k] for k in ("paper", "page", "smiles", "labels", "similarity")}  # noqa: E731
    if exact:
        row["verdict"], row["match"] = "found_exact", view(exact[0])
    elif conn:
        row["verdict"], row["match"] = "found_connectivity", view(conn[0])
    elif labelled:
        best = max(labelled, key=lambda m: m["similarity"])
        row["verdict"], row["labelled_structure"] = "label_disagrees", view(best)
    else:
        row["verdict"] = "not_found"
    row["nearest"] = view(nearest)
    if labelled:
        row["labelled_structures"] = [view(m) for m in sorted(labelled, key=lambda m: -m["similarity"])[:3]]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="test", help="test, dev (corrected dev SMILES) or comma-separated ids")
    ap.add_argument("--structures", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    split = json.loads(SPLIT.read_text())
    if a.cases == "test":
        ids, src = list(split["test"]), RAW
    elif a.cases == "dev":
        ids, src = [d["case_id"] for d in split["dev"]], CORRECTED
    else:
        ids, src = a.cases.split(","), RAW
    rows = [check_case(json.loads((src / f"{i}.json").read_text()), a.structures) for i in ids]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    from collections import Counter
    print(Counter(r["verdict"] for r in rows))
    for r in rows:
        if r["verdict"] not in ("found_exact",):
            n = r.get("labelled_structure") or r.get("nearest") or {}
            print(f"{r['case_id'][-4:]}  {r['verdict']:<20} {r['target_name'][:28]:<28} sim={n.get('similarity', '-')}  {n.get('smiles', '')[:90]}")


if __name__ == "__main__":
    main()
