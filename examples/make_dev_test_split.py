"""Write the development/test split of PhotoMechBench (raw_v2_119), checking its constraints.

The split is by paper cluster: every case of a paper is on the same side. Development cases are the
ones already used to build the agent, plus cases added for coverage of the mechanism pool, charge,
size and elements. Everything else is test and must not be run by the agent until the system is
frozen. Re-running this script must reproduce the committed file byte for byte.

    python examples/make_dev_test_split.py          # writes benchmarks/photomechbench/splits/raw_v2_119_dev_test.json
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechcal.eval.stats import load_reference_cases, paper_clusters  # noqa: E402

from mechcal.eval.dataset import DATASET as BENCH  # noqa: E402  (structure-corrected copy of raw_v2_119)
OUT = ROOT / "benchmarks/photomechbench/splits/raw_v2_119_dev_test.json"   # outside the frozen raw_v2_119 snapshot
P = "AIE_DDX_MECH_V4_"
USED = {  # run by the agent during development (2026-11-11/12)
    "0003": "smoke runs 1-7 of the agent", "0047": "dev1/dev2 runs", "0020": "dev1/dev2 runs", "0068": "dev1/dev2 runs",
    "0101": "dev1/dev2 runs", "0154": "dev1/dev2 runs", "0141": "dev1/dev2 runs",
}
ADDED = {  # chosen for coverage; single-paper clusters so no test case loses a sibling
    "0140": "HOST_GUEST primary (none in the used set); smallest such case",
    "0030": "HOST_GUEST + RIM primary; charged (+2), boron, 88 atoms",
    "0056": "RACI primary (none in the used set)",
    "0084": "RACI + RIM primary",
    "0009": "AGGREGATE + PACKING primary",
    "0019": "ESIPT + RACI primary (second ESIPT case)",
    "0083": "PET primary (second PET case); chlorine",
    "0104": "RIM primary (most frequent primary family); 72 atoms",
    "0152": "RADIATIVE + SOKR primary: a second true RADIATIVE case, to see whether reducing the RADIATIVE "
            "over-prediction hurts real ones (its paper also brings 0153)",
}


def build() -> dict:
    cases = load_reference_cases(BENCH / "cases")
    by_id = {c.case_id: c for c in cases}
    cluster = paper_clusters(cases)
    dev_ids = sorted(P + k for k in {**USED, **ADDED})
    dev_clusters = {cluster[c] for c in dev_ids}
    # every case of a development paper is development too
    dev_ids = sorted(c.case_id for c in cases if cluster[c.case_id] in dev_clusters)
    test_ids = sorted(c.case_id for c in cases if c.case_id not in set(dev_ids))
    assert set(dev_ids) | set(test_ids) == set(by_id) and not set(dev_ids) & set(test_ids)
    assert not {cluster[c] for c in dev_ids} & {cluster[c] for c in test_ids}, "a paper crosses the split"

    def coverage(ids):
        prim = collections.Counter(l for c in ids for l in by_id[c].primary_labels)
        anyl = collections.Counter(l for c in ids for l in by_id[c].labels)
        return {"primary": dict(sorted(prim.items())), "any": dict(sorted(anyl.items()))}

    def features(cid):
        m = Chem.MolFromSmiles(by_id[cid].smiles)
        return {"atoms_with_h": Chem.AddHs(m).GetNumAtoms(), "charge": Chem.GetFormalCharge(m),
                "elements": sorted({a.GetSymbol() for a in m.GetAtoms()})}

    role = lambda c: ("used_in_development" if c[len(P):] in USED else
                      "added_for_coverage" if c[len(P):] in ADDED else "same_paper_as_a_development_case")
    dev = [{"case_id": c, "role": role(c),
            "reason": USED.get(c[len(P):]) or ADDED.get(c[len(P):]) or "same paper as a development case",
            "primary_labels": sorted(by_id[c].primary_labels), **features(c)} for c in dev_ids]
    digest = hashlib.sha256(json.dumps({"dev": dev_ids, "test": test_ids}).encode()).hexdigest()
    return {
        "name": "photomechbench_raw_v2_119_dev_test", "created": "2026-11-12",
        "method": ("Split by paper cluster (shared DOI). Development = cases already used to build the agent plus "
                   "cases added for coverage of primary families, charge, size and elements; all other cases are test. "
                   "Test cases must not be run by the agent before the system (tools, cards, prompts, rubric) is frozen."),
        "counts": {"dev_cases": len(dev_ids), "test_cases": len(test_ids), "dev_papers": len(dev_clusters),
                   "test_papers": len({cluster[c] for c in test_ids})},
        "dev": dev, "test": test_ids,
        "coverage": {"dev": coverage(dev_ids), "test": coverage(test_ids)},
        "sha256": digest,
    }


if __name__ == "__main__":
    split = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(split["counts"]), "\ndev primary:", split["coverage"]["dev"]["primary"])
    print("test primary:", split["coverage"]["test"]["primary"], "\nsha256", split["sha256"])
