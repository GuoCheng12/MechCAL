"""Fixed-geometry SOC from validated DFTB outputs and isolated PySOC."""

import hashlib
import json
from pathlib import Path

from ase.io import write

from mechcal.tools.dftb_soc_io import prepare
from mechcal.tools.dftb_spin import spin_parameters, parse_states
from mechcal.tools.pysoc_protocol import RawSOC, METHOD
from mechcal.tools.pysoc_source import SOURCE_HASHES, SOURCE_COMMIT
from mechcal.tools.qc_process import QCError


def export_overlap(job, directory):
    try:
        job.execute([job.config.dftb_binary], directory)
        return 'normal_exit'
    except QCError as exc:
        log = (directory/'process.log').read_text()
        if not (exc.code == 'process_failed' and job.records[-1].get('returncode') == 1
                and 'Hamilton/Overlap written, exiting program.' in log and 'ERROR!' not in log):
            raise
        return 'intentional_export_stop_returncode_1'


def generate(job, atoms):
    constants, provenance = spin_parameters(job.config, atoms.get_chemical_symbols())
    angular = '\n'.join(f'{e} = "{("s" if e == "H" else "p")}"' for e in sorted(constants))
    spin = '\n'.join(e+' = { '+' '.join(map(str,constants[e]))+' }' for e in sorted(constants))
    prefix = str(Path(job.config.parameter_dir).resolve())+'/'
    if any(c in prefix for c in ('"','\n','\r')):
        raise QCError('invalid_args','Unsupported HSD path delimiters')
    common = f'''Geometry = GenFormat {{
 <<< "input.gen"
}}
Driver = {{}}
Hamiltonian = DFTB {{
 SCC = Yes
 SCCTolerance = 1e-10
 MaxSCCIterations = 250
 Charge = 0
 MaxAngularMomentum = {{ {angular} }}
 SlaterKosterFiles = Type2FileNames {{
  Prefix = "{prefix}"
  Separator = "-"
  Suffix = ".skf"
 }}
 SpinConstants = {{ ShellResolvedSpin = Yes
 {spin}
 }}
}}
ExcitedState = Casida {{
 NrOfExcitations = 3
 Symmetry = Both
 Diagonaliser = Arpack {{}}
 WriteTransitions = Yes
 WriteSPTransitions = Yes
 WriteXplusY = Yes
 WriteTransitionDipole = Yes
}}
ParserOptions {{ ParserVersion = 14 }}
'''
    source=job.workdir/'soc_source'
    spectra,matrices=source/'spectra',source/'matrices'
    for directory in (spectra,matrices):
        directory.mkdir(parents=True)
        write(directory/'input.gen',atoms,format='gen')
        write(directory/'geometry.xyz',atoms,format='xyz')
    (spectra/'dftb_in.hsd').write_text(common+'''Analysis { WriteEigenvectors = Yes
 EigenvectorsAsText = Yes }
Options { WriteResultsTag = Yes }
''')
    output=job.execute([job.config.dftb_binary],spectra)
    if 'SCC is NOT converged' in output:
        raise QCError('not_converged','DFTB reference did not converge')
    states=parse_states((spectra/'EXC.DAT').read_text(),3)
    (matrices/'dftb_in.hsd').write_text(common+'Options { WriteHS = Yes }\n')
    export_status=export_overlap(job,matrices)
    required=[spectra/n for n in ('band.out','eigenvec.out','XplusY.DAT','SPX.DAT','EXC.DAT','geometry.xyz')]
    required.extend(matrices/n for n in ('oversqr.dat','geometry.xyz'))
    manifest=dict(status='generated_not_yet_validated',states=states,
        files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in required})
    (source/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    return source,states,provenance,export_status


def evaluate(job, atoms):
    source,states,parameters,export_status=generate(job,atoms)
    normalized=job.workdir/'soc_normalized'
    checks=prepare(source,normalized)
    request=dict(source=str(Path(job.config.pysoc_source).resolve()),inputs=str(normalized),
        output=str(job.workdir/'soc_kernel'),source_hashes=SOURCE_HASHES)
    request_path,result_path=job.workdir/'soc_request.json',job.workdir/'soc_result.json'
    request_path.write_text(json.dumps(request,indent=2))
    from mechcal.tools import pysoc_worker
    job.execute([job.config.pysoc_python,str(Path(pysoc_worker.__file__).resolve()),
        str(request_path),str(result_path)],job.workdir)
    raw=RawSOC.model_validate_json(result_path.read_text())
    if raw.canonical_input_sha256!=hashlib.sha256((normalized/'canonical.json').read_bytes()).hexdigest():
        raise QCError('parse_failed','SOC result is not for the requested canonical input')
    energies={r['state']:r['excitation_ev'] for r in states}
    pairs=[{**r.model_dump(),'singlet_excitation_ev':0.0 if r.singlet==0 else energies[f'S{r.singlet}'],
            'triplet_excitation_ev':energies[f'T{r.triplet}']} for r in raw.couplings]
    return dict(method=METHOD,software_version=raw.pysoc_version,states=states,couplings=pairs,
        coupling_unit=raw.unit,operator=raw.operator,scale=raw.scale,ci_threshold=raw.ci_threshold,
        atom_index_base=0,symbols_in_source_order=atoms.get_chemical_symbols(),coordinates_angstrom=atoms.positions.tolist(),
        input_checks=checks,parameters=parameters,matrix_export_status=export_status,
        provenance={'source_directory':str(source),'normalized_directory':str(normalized),
            'raw_result':str(result_path),'request':str(request_path),'kernel_directory':str(job.workdir/'soc_kernel'),
            'pysoc_source_commit':SOURCE_COMMIT,'pysoc_source_hashes':SOURCE_HASHES,
            'worker_sha256':hashlib.sha256(Path(pysoc_worker.__file__).read_bytes()).hexdigest()},
        interpretation='Computed spin-orbit coupling magnitudes between S0-S3 and T1-T3 at the unchanged source geometry. Read the state-specific coupling together with its energy separation; neither quantity alone is an ISC rate.',
        limitations=['Neutral closed-shell CHNO, fixed mio-1-1 and ZEFF approximation; no metal or heavy-atom extrapolation.',
            'Only three roots per spin manifold. Higher states and state-following are not covered.',
            'No vibrational overlap, environment, spin populations, ISC rate, quantum yield or mechanism verdict.',
            'This is a qualitative TD-DFTB-based calculation, not interchangeable with calibrated ab initio SOC.',
            'A modeled geometry is not an observed crystal or experimental validation.'])
