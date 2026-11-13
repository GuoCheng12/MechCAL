"""Build the reference panel: standard descriptors of common chromophores (no labels, no LLM).

    python examples/build_reference_panel.py --out outputs/reference_panel --parallel 6
    python examples/build_reference_panel.py --out outputs/reference_panel --install   # write mechcal/agent/reference_panel.json

The panel is a fixed list of widely studied organic chromophores (aromatics, donor-acceptor dyes,
xanthene, coumarin, BODIPY and cyanine dyes, carbonyl, azo, proton-transfer and heteroaromatic
compounds), chosen as common examples of luminescent and absorbing molecules, not by mechanism.
Molecules whose skeleton (InChIKey first block) appears anywhere in the benchmark (dev or test,
original or corrected SMILES) are left out. Each molecule runs the same recipes with default
settings through the recorded gateway; descriptor values are those the gateway reports at run
time (mechcal.agent.reference).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # this checkout's mechcal, not an installed copy
from mechcal.eval.dataset import CASES_DIR, RAW  # noqa: E402

PANEL = {
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc1ccccc12",
    "pyrene": "c1cc2ccc3cccc4ccc(c1)c2c34",
    "perylene": "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54",
    "tetracene": "c1ccc2cc3cc4ccccc4cc3cc2c1",
    "biphenyl": "c1ccc(cc1)-c1ccccc1",
    "p_terphenyl": "c1ccc(cc1)-c1ccc(cc1)-c1ccccc1",
    "fluorene": "c1ccc2c(c1)Cc1ccccc1-2",
    "trans_stilbene": "C(=C/c1ccccc1)\\c1ccccc1",
    "diphenylanthracene": "c1ccc(cc1)-c1c2ccccc2c(-c2ccccc2)c2ccccc12",
    "bis_phenylethynyl_anthracene": "C(#Cc1ccccc1)c1c2ccccc2c(C#Cc2ccccc2)c2ccccc12",
    "tetraphenylethylene": "c1ccc(cc1)C(=C(c1ccccc1)c1ccccc1)c1ccccc1",
    "carbazole": "c1ccc2c(c1)[nH]c1ccccc12",
    "triphenylamine": "c1ccc(cc1)N(c1ccccc1)c1ccccc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "dibenzofuran": "c1ccc2c(c1)oc1ccccc12",
    "quinoline": "c1ccc2ncccc2c1",
    "acridine": "c1ccc2nc3ccccc3cc2c1",
    "phenothiazine": "c1ccc2c(c1)Nc1ccccc1S2",
    "dmabn": "CN(C)c1ccc(cc1)C#N",
    "p_nitroaniline": "Nc1ccc(cc1)[N+](=O)[O-]",
    "dans": "CN(C)c1ccc(/C=C/c2ccc(cc2)[N+](=O)[O-])cc1",
    "prodan": "CCC(=O)c1ccc2cc(N(C)C)ccc2c1",
    "dcm": "CN(C)c1ccc(/C=C/c2cc(=C(C#N)C#N)cc(C)o2)cc1",
    "nile_red": "CCN(CC)c1ccc2nc3c(cc(=O)c4ccccc43)oc2c1",
    "coumarin_1": "CCN(CC)c1ccc2c(C)cc(=O)oc2c1",
    "umbelliferone": "O=c1ccc2ccc(O)cc2o1",
    "fluorescein": "OC(=O)c1ccccc1-c1c2ccc(=O)cc-2oc2cc(O)ccc12",
    "rhodamine_b": "CCN(CC)c1ccc2c(-c3ccccc3C(=O)O)c3ccc(=[N+](CC)CC)cc-3oc2c1",
    "bodipy_tetramethyl_phenyl": "Cc1cc(C)n2c1C(c1ccccc1)=C1C(C)=CC(C)=[N+]1[B-]2(F)F",
    "thiacarbocyanine": "CCN1/C(=C/C=C/c2sc3ccccc3[n+]2CC)Sc2ccccc21",
    "amino_naphthalimide": "CCCCN1C(=O)c2cccc3c(N)ccc(C1=O)c23",
    "perylene_diimide": "CN1C(=O)c2ccc3c4ccc5C(=O)N(C)C(=O)c6ccc(c7ccc(C1=O)c2c37)c4c56",
    "benzophenone": "O=C(c1ccccc1)c1ccccc1",
    "xanthone": "O=c1c2ccccc2oc2ccccc12",
    "thioxanthone": "O=c1c2ccccc2sc2ccccc12",
    "chalcone": "O=C(/C=C/c1ccccc1)c1ccccc1",
    "azobenzene": "c1ccc(cc1)/N=N/c1ccccc1",
    "hydroxyphenyl_benzothiazole": "Oc1ccccc1-c1nc2ccccc2s1",
    "hydroxyflavone": "O=c1c(O)c(-c2ccccc2)oc2ccccc12",
    "salicylaldehyde": "O=Cc1ccccc1O",
    "methyl_salicylate": "COC(=O)c1ccccc1O",
}

RECIPES = [("excited_states", {}), ("charge_transfer", {"state_root": 1}), ("torsion_response", {}),
           ("vibrations", {}), ("solvent_polarity", {}), ("dimer_excited_states", {}), ("spin_states", {})]


def benchmark_skeletons() -> set[str]:
    """Skeletons of every benchmark molecule, from the original and the structure-corrected SMILES."""
    from rdkit import Chem
    keys = set()
    for cases in (RAW / "cases", CASES_DIR):
        for path in cases.glob("*.json"):
            smi = json.loads(path.read_text())["public_input"]["molecule"]["structure"]["value"]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                keys.add(Chem.MolToInchiKey(mol).split("-")[0])
    return keys


def run_one(name: str, smiles: str, out: Path, config: Path) -> dict:
    """One molecule in this process: all recipes, then the descriptor values the gateway recorded."""
    from rdkit import Chem
    from mechcal.agent.environment import tool_environment
    from mechcal.agent.recipes import RecipeContext, run_recipe
    from mechcal.eval.tool_gateway import ToolGateway

    mol = Chem.MolFromSmiles(smiles)
    elements = {a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()}
    run_dir = out / name
    started = time.monotonic()
    recipes = {}
    with tool_environment(config, run_dir) as deployment:
        gw = ToolGateway(case_id=f"PANEL_{name}", smiles=smiles, run_dir=run_dir / "tools",
                         enable_amesp=bool(deployment.runtime.get("enable_amesp", True)), max_wall_seconds=5400)
        ctx = RecipeContext(gw)
        for recipe, args in RECIPES:
            if recipe == "spin_states" and (not elements <= {"C", "H", "N", "O"} or Chem.GetFormalCharge(mol) != 0):
                recipes[recipe] = "skipped (TD-DFTB route covers neutral CHNO only)"
                continue
            res = run_recipe(ctx, recipe, args)
            recipes[recipe] = res.status
    values: dict[str, float] = {}
    for alias, vals in gw.reference_values.items():
        for k, v in vals.items():
            values.setdefault(k, v)                 # first occurrence (default settings, S1, first pose)
    row = {"name": name, "smiles": smiles, "values": values, "recipes": recipes,
           "tool_calls": gw.cost()["tool_calls"], "minutes": round((time.monotonic() - started) / 60, 1)}
    (run_dir / "panel_row.json").write_text(json.dumps(row, indent=1))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--config", type=Path, default=ROOT / "mechcal.local.toml")
    ap.add_argument("--one", help="internal: run one molecule in this process")
    ap.add_argument("--only", help="comma-separated names (default: all)")
    ap.add_argument("--install", action="store_true", help="collect rows and write mechcal/agent/reference_panel.json")
    args = ap.parse_args()
    if args.one:
        print(json.dumps(run_one(args.one, PANEL[args.one], args.out, args.config)))
        return
    if args.install:
        rows = [json.loads(p.read_text()) for p in sorted(args.out.glob("*/panel_row.json"))]
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        panel = {"description": __doc__.split("\n\n")[2].replace("\n", " "), "built_with_commit": commit,
                 "built": time.strftime("%Y-%m-%d"), "molecules": [{k: r[k] for k in ("name", "smiles", "values")} for r in rows]}
        target = ROOT / "mechcal/agent/reference_panel.json"
        target.write_text(json.dumps(panel, indent=1))
        print(f"{len(rows)} molecules -> {target}")
        return
    skip = benchmark_skeletons()
    from rdkit import Chem
    names = [n for n in (args.only.split(",") if args.only else PANEL)
             if Chem.MolToInchiKey(Chem.MolFromSmiles(PANEL[n])).split("-")[0] not in skip]
    left_out = sorted(set(args.only.split(",") if args.only else PANEL) - set(names))
    print(f"{len(names)} molecules; left out (in the benchmark): {left_out}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    def launch(name):
        if (args.out / name / "panel_row.json").is_file():
            return name, "done before"
        with (args.out / f"{name}.log").open("w") as log:
            code = subprocess.run([sys.executable, __file__, "--one", name, "--out", str(args.out), "--config", str(args.config)],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        return name, "ok" if (args.out / name / "panel_row.json").is_file() else f"failed (exit {code})"

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for name, state in pool.map(launch, names):
            print(name, state, flush=True)


if __name__ == "__main__":
    main()
