"""Result table of the per-molecule metrics with paper-unit bootstrap intervals and floors (evaluation side).

    python examples/molecule_table.py mechcal=outputs/agent/dev7a,outputs/agent/dev7b direct=outputs/baselines/direct_dev7

Each method is one or more directories of graded runs (grade.json written by examples/grade_agent_run.py;
several directories are repeats). Per molecule a method's value is its mean over its runs; the table
gives the mean over molecules with a 95 % interval from resampling paper units. Floors are computed on
the molecules the methods ran on: three random families, all seven, the three most frequent families of
the development set, structure kNN-5 (similarity-weighted label transfer from the other paper units
of rebuilt_302, its top three as the answer), and structure rules (STRUCTURE_RULES: a family is claimed
when the molecule's deterministic structural facts contain a named motif; no labels are used). Every pair of methods is compared on the molecules both
have (mean difference and its interval). Prints the tables and, with --out, writes them as JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from mechcal.agent.facts import structural_facts
from mechcal.eval import molecule_metrics as mm
from mechcal.eval.dataset import CASES_DIR, case_record, public_smiles
from mechcal.eval.dataset_analysis import knn_predictor, similarity_matrix
from mechcal.eval.splits import load_split
from mechcal.eval.stats import load_reference_cases
from mechcal.mechanism_pool import MECHANISM_POOL as POOL

# Textbook structure shortcuts, fixed on 2026-11-23 before any score was computed and not tuned afterwards.
# Each uses fact kinds of mechcal/agent/facts.py (the facts every method receives). The order is the answer
# order, most specific motif first. PET, AGGREGATE and RADIATIVE have no rule: no fact kind is a standard
# shortcut for them. A molecule on which no rule fires gets an empty answer.
STRUCTURE_RULES = (
    ("ESIPT_PT", {"intramolecular_hbond_motif"}),
    ("TRIPLET_ISC_RTP_TADF", {"heavy_atom", "metal_center"}),
    ("ICT_TICT_CT", {"donor_acceptor_path"}),
    ("RIM_RIR_RIV", {"conjugated_rotor", "aryl_olefin"}),
)


def structure_rule_answer(smiles: str) -> list[str]:
    kinds = {f["kind"] for f in structural_facts(smiles)["facts"]}
    return [family for family, motifs in STRUCTURE_RULES if kinds & motifs]


LABELS = {"es_rec": "ES-Rec", "top1": "Top-1", "evr": "EVR", "ep": "EP", "rec": "Rec", "bal": "BAL"}


def method_rows(dirs: list[Path]) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for d in dirs:
        for g in sorted(d.glob("*/grade.json")):
            grade = json.loads(g.read_text())
            rows.setdefault(grade["case_id"], []).append(grade["metrics"])
    return rows


def floors(case_ids: list[str], refs: dict[str, mm.Reference], split: dict) -> dict[str, dict[str, list[dict]]]:
    dev_refs = [mm.reference_of(case_record(d["case_id"])["hidden_reference"]) for d in split["dev"]]
    counts = Counter(f for r in dev_refs for f in r.supported)
    common = sorted(POOL, key=lambda f: (-counts[f], POOL.index(f)))[:3]
    rng = random.Random(0)
    cases = load_reference_cases(CASES_DIR)
    units, predict, by_id = split["units"], knn_predictor(similarity_matrix(cases), k=5), {c.case_id: c for c in cases}
    knn = {c: predict([t for t in cases if units[t.case_id] != units[c]], by_id[c]) for c in case_ids}
    return {
        "random 3 families": {c: [mm.molecule_metrics(rng.sample(list(POOL), 3), refs[c]) for _ in range(500)] for c in case_ids},
        "all 7 families": {c: [mm.molecule_metrics(list(POOL), refs[c])] for c in case_ids},
        "most frequent 3 (dev)": {c: [mm.molecule_metrics(common, refs[c])] for c in case_ids},
        "structure kNN-5": {c: [mm.molecule_metrics(knn[c], refs[c])] for c in case_ids},
        "structure rules": {c: [mm.molecule_metrics(structure_rule_answer(public_smiles(c)), refs[c])] for c in case_ids},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("methods", nargs="+", help="name=dir[,dir...]")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    split = load_split()
    methods = {}
    for spec in args.methods:
        name, dirs = spec.split("=", 1)
        methods[name] = mm.per_case(method_rows([Path(d) for d in dirs.split(",")]))
    case_ids = sorted({c for v in methods.values() for c in v})
    refs = {c: mm.reference_of(case_record(c)["hidden_reference"]) for c in case_ids}
    floor_values = {n: mm.per_case(r) for n, r in floors(case_ids, refs, split).items()}
    report = {"cases": len(case_ids), "methods": {}, "floors": {}, "pairs": {}}
    for kind, group in (("methods", methods), ("floors", floor_values)):
        for name, values in group.items():
            report[kind][name] = {"cases": len(values), "mean": mm.summarize(values),
                                  "ci95": mm.unit_bootstrap(values, split["units"], n=args.resamples)}
    for a, b in combinations(methods, 2):
        shared = sorted(set(methods[a]) & set(methods[b]))
        diff = {c: {m: methods[a][c][m] - methods[b][c][m] for m in mm.METRICS if m in methods[a][c] and m in methods[b][c]}
                for c in shared}
        report["pairs"][f"{a} - {b}"] = {"cases": len(shared), "mean": mm.summarize(diff),
                                         "ci95": mm.unit_bootstrap(diff, split["units"], n=args.resamples)}
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))

    def cell(d, m, signed=False):
        if d["mean"][m] is None:
            return f"{'-':^18s}"
        lo, hi = d["ci95"][m]
        return f"{d['mean'][m]:+.2f} [{lo:+.2f},{hi:+.2f}]" if signed else f"{d['mean'][m]:.2f} [{lo:.2f},{hi:.2f}]  "
    head = f"{'':26s} n   " + " ".join(f"{LABELS[m]:18s}" for m in mm.METRICS)
    print(f"{len(case_ids)} molecules; mean with 95% paper-unit interval (EVR: lower is better)\n{head}")
    for kind in ("methods", "floors"):
        for name, d in report[kind].items():
            print(f"{name:26s} {d['cases']:3d} " + " ".join(cell(d, m) for m in mm.METRICS))
    if report["pairs"]:
        print(f"\npaired differences on shared molecules\n{head}")
        for name, d in report["pairs"].items():
            print(f"{name:26s} {d['cases']:3d} " + " ".join(cell(d, m, signed=True) for m in mm.METRICS))


if __name__ == "__main__":
    main()
