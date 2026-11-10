"""CREST ensemble sampling with bounded work and reusable source-mapped members."""

from pathlib import Path
import numpy as np
from ase.io import read, write
from rdkit import Chem
from mechcal.tools.qc_process import QCError
from mechcal.tools.xtb_backend import single, number, NUMBER


def sample(job, atoms, args, source):
    directory = job.workdir/"crest"
    directory.mkdir()
    write(directory/"input.xyz", atoms)
    command = [job.config.crest_binary, "input.xyz", "--legacy", "--gfn2", "--"+args.search_level,
        "--T", str(job.config.threads), "--chrg", "0", "--uhf", "0",
        "--ewin", str(args.energy_window_kcal_mol), "--nozs", "-xnam", job.config.xtb_binary]
    if args.md_length_ps is not None:
        command.extend(["--mdlen", str(args.md_length_ps)])
    failure = None
    try:
        text = job.execute(command, directory)
        if "Version 3.0.2," not in text:
            raise QCError("unsupported", "This adapter was validated with CREST 3.0.2.")
        if "CREST terminated normally" not in text:
            raise QCError("not_converged", "CREST normal termination marker absent.")
    except QCError as exc:
        if exc.code == "unsupported":
            raise
        failure = exc
    ensemble = directory/"crest_conformers.xyz"
    if not ensemble.is_file():
        raise failure or QCError("parse_failed", "CREST did not write an ensemble.")
    # Only final ensemble members are reused, never intermediate MD frames.
    geometries = read(ensemble, index=":")
    mol = Chem.MolFromMolFile(source.file_paths["sdf"], removeHs=False)
    rows = []
    for index, geometry in enumerate(geometries[:args.max_members]):
        row = dict(member_id=f"crest:{index+1:02d}")
        try:
            if geometry.get_chemical_symbols() != atoms.get_chemical_symbols() or not np.isfinite(geometry.positions).all():
                raise QCError("parse_failed", "CREST changed the source atom order or produced invalid coordinates.")
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                ratio = geometry.get_distance(a, b)/atoms.get_distance(a, b)
                if not .6 < ratio < 1.6:
                    raise QCError("numerical_validation_failed", "Sample breaks or strongly distorts a source bond; topology cannot be reused.")
            # Independent single point checks the final member, not an MD energy comment.
            member = job.workdir/f"member_{index+1:02d}"
            output = single(job, geometry, member, ["--gfn", "2"])
            row.update(status="success", xyz=str(member/"input.xyz"),
                energy_hartree=number(output, rf"TOTAL ENERGY\s+({NUMBER})\s+Eh"),
                geometry_converged=None, geometry_optimization_status="crest_final_ensemble",
                electronic_status="converged")
        except QCError as exc:
            row.update(status="failed", failure_code=exc.code, reason=str(exc))
        rows.append(row)
    if failure:
        rows.append(dict(member_id="sampling", status="failed", failure_code=failure.code, reason=str(failure)))
    good = [r for r in rows if r["status"] == "success"]
    if good:
        minimum = min(r["energy_hartree"] for r in good)
        for row in good:
            row["relative_energy_kcal_mol"] = (row["energy_hartree"]-minimum)*627.509474
    return dict(method="CREST 3.0.2 legacy external GFN2-xTB gas-phase conformer search", points=rows,
        ensemble_file=str(ensemble), ensemble_size=len(geometries), returned_limit=args.max_members,
        sampling_settings=dict(search_level=args.search_level, md_length_ps=args.md_length_ps,
            md_length_mode="automatic" if args.md_length_ps is None else "explicit",
            max_members_limits="returned members only, not sampling effort"),
        constraints="none; atom freezing is not supported by this validated 3.0.2 legacy route", seed=None,
        interpretation=f"Returned {len(good)} source-mapped conformers with checked GFN2 single-point energies.",
        limitations=["Stochastic reduced search; this interface does not promise a reproducible random seed or exhaustive ensemble.",
            "CREST and xTB share an energy model; these are not independent physical evidence sources.",
            "A shorter requested MD length reduces sampling coverage; it is not an exhaustive conformer survey or a guaranteed runtime.",
            "Unconstrained search; no fixed atoms, crystal, population free energy or excited-state kinetics is computed.",
            "Bond-length integrity is checked; absolute stereochemistry and all possible rearrangements are not independently reidentified."])
