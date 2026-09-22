"""Check local scientific dependencies without credentials or model API calls."""

from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from mechcal.schemas import CaseInput
from mechcal.tools.amesp_baseline import AmespBaselineRunner
from mechcal.tools.local_structure import LocalStructureTool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amesp", action="store_true", help="Run real aTB1/TDA calculations.")
    parser.add_argument("--mcp", action="store_true", help="Also verify MCP dependency imports.")
    args = parser.parse_args()
    packages = ["mechcal", "rdkit", "ase", "numpy", "scipy"]
    import ase
    import numpy
    import scipy
    from rdkit import Chem

    assert Chem.MolFromSmiles("C=O") is not None
    assert ase.Atoms("H2").get_chemical_formula() == "H2"
    assert numpy.isfinite(scipy.constants.Avogadro)
    if args.mcp:
        import fastmcp
        import mcp

        assert fastmcp.FastMCP is not None and mcp.ClientSession is not None
        packages.extend(["fastmcp", "mcp"])
    result = {"versions": {package: version(package) for package in packages}}
    with TemporaryDirectory(prefix="mechcal-tool-check-") as directory:
        artifact, failure = LocalStructureTool().prepare(
            CaseInput(case_id="tool-check", smiles="C=O", user_query="Installation check."),
            round_id="R001",
            workspace=Path(directory),
        )
        if failure.kind != "none" or artifact.status != "available":
            raise RuntimeError(f"RDKit structure preparation failed: {failure.model_dump()}")
        result["structure_preparation"] = "passed"
        if args.amesp:
            binary = os.environ.get("MECHCAL_AMESP_BIN", "")
            path = Path(binary).expanduser()
            if not binary or not path.is_file() or not os.access(path, os.X_OK):
                raise RuntimeError("Set MECHCAL_AMESP_BIN to an executable Amesp file.")
            runner = AmespBaselineRunner(amesp_bin=path)
            baseline = runner.run_baseline(
                prepared_artifact=artifact, case_id="tool-check", round_id="R001"
            )
            if baseline.final_energy_hartree is None or not baseline.excited_states:
                raise RuntimeError("Amesp output lacks a parsed energy or excited state.")
            result["amesp"] = {
                "ground_state": "passed",
                "excited_state": "passed",
                "parsed_state_count": len(baseline.excited_states),
            }
        else:
            result["amesp"] = "not requested"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
