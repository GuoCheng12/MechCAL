"""Offline PhotoMechBench dataset report (no model API calls).

Writes JSON and a short Markdown summary: label distribution, paper clusters,
near-duplicate pairs with label agreement, and SMILES-blind / structure-only floors.

    python examples/dataset_report.py --out outputs/dataset_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechcal.eval.dataset_analysis import (
    floors, label_distribution, near_duplicate_pairs, similarity_matrix, within_paper_agreement,
)
from mechcal.eval.stats import load_reference_cases

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "benchmarks/photomechbench/raw_v2_119/cases"


def fmt(ci: dict) -> str:
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    cases = load_reference_cases(args.cases)
    sims = similarity_matrix(cases)
    pairs = near_duplicate_pairs(cases, sims, threshold=args.threshold)
    report = {
        "n_cases": len(cases),
        "label_distribution": label_distribution(cases),
        "paper_clusters": within_paper_agreement(cases),
        "near_duplicates": {
            "threshold": args.threshold, "pairs": len(pairs),
            "cross_paper_pairs": sum(not p["same_paper_cluster"] for p in pairs),
            "pairs_with_different_primary": sum(not p["same_primary_set"] for p in pairs),
            "items": pairs,
        },
        "floors": floors(cases, sims, n_resamples=args.resamples),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset_report.json").write_text(json.dumps(report, indent=2))
    pc, nd = report["paper_clusters"], report["near_duplicates"]
    lines = [
        f"# PhotoMechBench dataset report ({len(cases)} cases)", "",
        f"- Paper clusters: {pc['clusters']}; {pc['cases_in_multi_case_clusters']} cases in "
        f"{pc['multi_case_clusters']} multi-case clusters (largest {pc['largest_cluster']}).",
        f"- Within-cluster pairs with identical label sets: {pc['within_cluster_identical_label_sets']}"
        f"/{pc['within_cluster_pairs']} (mean Jaccard {pc['within_cluster_mean_label_jaccard']:.3f}).",
        f"- Near-duplicate pairs (Tanimoto >= {nd['threshold']}): {nd['pairs']}; cross-paper "
        f"{nd['cross_paper_pairs']}; different primary set {nd['pairs_with_different_primary']}.",
        "", "## Floors (95% paper-clustered bootstrap CI)", "",
        "| Method | Top-1 Primary | Recall@3 | nDCG@3 | Macro R@3 |", "| --- | --- | --- | --- | --- |",
    ]
    for name, f in report["floors"].items():
        lines.append(f"| {name} | {fmt(f['top1_primary'])} | {fmt(f['recall_at_3'])} | "
                     f"{fmt(f['ndcg_at_3'])} | {f['macro_recall_at_3']:.3f} |")
    lines += ["", "## Near-duplicate pairs with different primary labels", "",
              "| A | B | Tanimoto | same paper | primary A | primary B |", "| --- | --- | --- | --- | --- | --- |"]
    for p in pairs:
        if not p["same_primary_set"]:
            lines.append(f"| {p['case_a'][-4:]} | {p['case_b'][-4:]} | {p['tanimoto']} | "
                         f"{p['same_paper_cluster']} | {', '.join(p['primary_a'])} | {', '.join(p['primary_b'])} |")
    (args.out / "dataset_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
