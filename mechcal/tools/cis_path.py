"""Small energy-only coordinate comparisons, not excited-state relaxation."""
import json
from pathlib import Path
import numpy as np
from ase.io import write
from rdkit import Chem
from mechcal.tools.contact_motion import moving_side
from mechcal.tools.qc_process import QCError


def scan(job, atoms, args, source):
    if max(args.atom_indices)>=len(atoms):
        raise QCError('invalid_args','Coordinate index is outside source atom map.')
    mol=Chem.MolFromMolFile(source.file_paths['sdf'],removeHs=False)
    if args.coordinate=='dihedral':
        moving=moving_side(mol,args.atom_indices,list(range(len(atoms))))
    else:
        a,b=args.atom_indices
        if atoms[b].symbol!='H' or not mol.GetBondBetweenAtoms(a,b):
            raise QCError('invalid_args','Bond scan moves a bonded H only; give donor then H.')
        moving=[b]
    geometries=[]
    for value in args.values:
        geometry=atoms.copy()
        if args.coordinate=='dihedral':
            geometry.set_dihedral(*args.atom_indices,value,indices=moving)
        else:
            a,b=args.atom_indices
            direction=atoms.positions[b]-atoms.positions[a]
            length=np.linalg.norm(direction)
            if length<1e-8:
                raise QCError('invalid_args','Coordinate endpoints coincide.')
            geometry.positions[b]=atoms.positions[a]+direction/length*value
        distances=geometry.get_all_distances(); np.fill_diagonal(distances,np.inf)
        if distances.min()<.55:
            raise QCError('invalid_args','Coordinate scan would create an atomic overlap.')
        geometries.append(geometry)
    request=dict(symbols=atoms.get_chemical_symbols(), coordinates_angstrom=[g.positions.tolist() for g in geometries],
        state_root=args.state_root,threads=job.config.threads)
    directory=job.workdir/'cis_path'; directory.mkdir()
    input_path,output=directory/'request.json',directory/'result.json'
    input_path.write_text(json.dumps(request,indent=2))
    failure=None
    try:
        job.execute([job.config.pyscf_python,str(Path(__file__).with_name('cis_path_worker.py').resolve()),str(input_path),str(output)],directory)
    except QCError as exc:
        failure=exc
    if not output.exists():
        raise failure or QCError('parse_failed','No completed path point.')
    raw=json.loads(output.read_text())
    if raw.get('request')!=request or raw.get('schema_version')!='cis_path_v1':
        raise QCError('parse_failed','Path output/request mismatch.')
    rows=raw['points']
    for index,row in enumerate(rows):
        if row['point_index']!=index or not np.isfinite(row['selected_total_energy_hartree']):
            raise QCError('parse_failed','Invalid path point identity or energy.')
        path=directory/f'geometry_{index}.xyz'; write(path,geometries[index])
        row.update(coordinate_value=args.values[index],xyz=str(path),geometry_converged=None,
            geometry_optimization_status='not_requested',electronic_status='converged')
    if failure:
        rows.append(dict(status='failed',failure_code=failure.code,reason=str(failure)))
    return dict(method='RHF/6-31G/CIS, fixed-coordinate path, three singlets',points=rows,
        coordinate=args.coordinate,atom_indices=args.atom_indices,moved_atom_indices=moving,
        coordinate_unit='angstrom' if args.coordinate=='bond' else 'degree',
        interpretation=f'{sum(p["status"]=="success" for p in rows)} fixed-coordinate points completed; inspect state-overlap ambiguity before comparing energies.',
        limitations=['Not an optimized excited-state path, activation barrier, conical intersection or rate.',
            'State continuity uses transition-density overlaps; ambiguous or missed states do not establish a continuous surface.',
            'CIS/small-basis qualitative CHNO scope. No solvent, SOC or dynamic populations.',
            'Proton-coordinate points retain the source atom map; source bonding labels describe the starting topology.'])
