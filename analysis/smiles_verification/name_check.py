"""Check a case's SMILES against chemical names printed in its source papers.

For each compound designation (for example "TPC" or "complex 1") the script collects the text
just before "(designation)" in the paper, tries every trailing word span as a chemical name with
OPSIN (offline, deterministic), and looks up trivial names in PubChem. Each parsed structure is
compared with the case SMILES by InChIKey: the full key (same structure), the first block
(same connectivity, stereo or charge may differ), or different.

Usage:
    python analysis/smiles_verification/name_check.py --cases dev --text-dir <reading-order text dir> \
        --opsin <opsin jar> --out <file.json> [--pubchem]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "benchmarks/photomechbench/raw_v2_119/cases"
SPLIT = ROOT / "benchmarks/photomechbench/splits/raw_v2_119_dev_test.json"
MAX_TOKENS = 10


def key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        mol.UpdatePropertyCache(strict=False)
    try:
        return Chem.MolToInchiKey(mol) or None
    except Exception:  # noqa: BLE001 - some organometallics have no InChI
        return None


def compare(case_key: str | None, other: str | None) -> str:
    if not case_key or not other:
        return "no_key"
    if case_key == other:
        return "same_structure"
    if case_key[:14] == other[:14]:
        return "same_connectivity"
    return "different"


def designations(target: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*/\s*", target or "") if p.strip()]


def name_spans(text: str, designation: str) -> list[str]:
    """Word spans that end right before "(designation)" or "designation," style definitions."""
    spans: list[str] = []
    for m in re.finditer(rf"\(\s*{re.escape(designation)}\s*[,;)]", text):
        before = text[max(0, m.start() - 400): m.start()]
        before = re.sub(r"-\s*\n\s*", "-", before)          # line-break hyphenation inside names
        before = re.sub(r"\s+", " ", before).strip()
        words = before.split(" ")
        for k in range(1, min(MAX_TOKENS, len(words)) + 1):
            span = " ".join(words[-k:]).strip(" ,.;:")
            if len(span) >= 6:
                spans.append(span)
    return list(dict.fromkeys(spans))


def opsin(names: list[str], jar: Path) -> dict[str, str]:
    if not names:
        return {}
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "in.txt", Path(d) / "out.txt"
        inp.write_text("\n".join(n.replace("\n", " ") for n in names) + "\n")
        subprocess.run(["java", "-jar", str(jar), "-osmi", str(inp), str(out)], capture_output=True, timeout=600)
        lines = out.read_text().split("\n") if out.exists() else []
    return {n: s.strip() for n, s in zip(names, lines) if s.strip()}


def pubchem(name: str) -> dict | None:
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(name, safe="") + "/property/InChIKey,IsomericSMILES,IUPACName/JSON")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            props = json.load(r)["PropertyTable"]["Properties"]
    except Exception:  # noqa: BLE001 - not found or network error
        return None
    finally:
        time.sleep(0.25)                                    # PubChem asks for <= 5 requests/s
    return props[0] if props else None


def check_case(case: dict, text_dir: Path, jar: Path, use_pubchem: bool) -> dict:
    h = case["hidden_reference"]
    smiles = case["public_input"]["molecule"]["structure"]["value"]
    ck = key(smiles)
    text = "\n".join((text_dir / (d["doi"].replace("/", "_") + ".txt")).read_text(errors="ignore")
                     for d in h["source_documents"] if (text_dir / (d["doi"].replace("/", "_") + ".txt")).exists())
    desig = designations(h["target_name"])
    spans = [s for d in desig for s in name_spans(text, d)]
    parsed = opsin(desig + spans, jar)
    hits = []
    for name, smi in parsed.items():
        hits.append({"name": name, "source": "opsin", "smiles": smi, "match": compare(ck, key(smi))})
    if use_pubchem:
        for d in desig:
            if len(d) >= 4 and not re.fullmatch(r"(compound|complex)?\s*\d+[a-z]?", d, re.I):
                p = pubchem(d)
                if p:
                    hits.append({"name": d, "source": "pubchem", "smiles": p.get("IsomericSMILES") or p.get("SMILES"),
                                 "iupac": p.get("IUPACName"), "match": compare(ck, p.get("InChIKey"))})
    # Longest parsed span per source is the most specific reading of the definition.
    order = {"same_structure": 0, "same_connectivity": 1, "different": 2, "no_key": 3}
    hits.sort(key=lambda x: (order[x["match"]], -len(x["name"])))
    best = hits[0]["match"] if hits else "no_name_parsed"
    return {"case_id": case["case_id"], "target_name": h["target_name"], "smiles": smiles, "inchikey": ck,
            "definition_spans": len(spans), "verdict": best, "hits": hits[:6]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="dev")
    ap.add_argument("--text-dir", required=True, type=Path)
    ap.add_argument("--opsin", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pubchem", action="store_true")
    a = ap.parse_args()
    split = json.loads(SPLIT.read_text())
    ids = ([d["case_id"] for d in split["dev"]] if a.cases == "dev" else list(split["test"]) if a.cases == "test"
           else sorted(p.stem for p in CASES.glob("*.json")) if a.cases == "all" else a.cases.split(","))
    rows = [check_case(json.loads((CASES / f"{i}.json").read_text()), a.text_dir, a.opsin, a.pubchem) for i in ids]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    for r in rows:
        top = r["hits"][0] if r["hits"] else {}
        print(f"{r['case_id'][-4:]}  {r['verdict']:<18} spans={r['definition_spans']:<3} {top.get('source', ''):<8} "
              f"{top.get('name', '')[:70]}")


if __name__ == "__main__":
    main()
