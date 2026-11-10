"""Explicit approximate conventional TDDFT/SOC route, separate from aTB."""
import hashlib
import os
from pathlib import Path
from mechcal.tools.amesp_baseline import _write_amesp_input
from mechcal.tools.amesp_soc import parse_spin_soc
from mechcal.tools.qc_process import QCError


def evaluate(job, atoms, args):
    directory=job.workdir/'soc'; directory.mkdir()
    aip,aop=directory/'soc.aip',directory/'soc.aop'
    keywords=['b3lyp',args.basis,'tda','soc']
    if args.integral_mode=='ricosx':
        keywords.append('ricosx')
    _write_amesp_input(aip_path=aip,keywords=keywords,charge=0,multiplicity=1,
        symbols=atoms.get_chemical_symbols(),coordinates=atoms.positions.tolist(),
        block_lines=[('scf',['maxcyc 150','conver 8']),('posthf',[f'nstates {args.nstates_per_spin}','tdspin all','tout 5'])],
        npara=job.config.threads,maxcore_mb=1000)
    binary=Path(job.config.amesp_soc_binary).resolve()
    # Amesp's manual requires an unlimited main stack and a 4 GiB OpenMP stack.
    # Apply this to this child process only, not to the server or other tools.
    limiter=Path('/usr/bin/prlimit')
    if not limiter.is_file():
        raise QCError('unsupported','Amesp SOC requires Linux prlimit for its documented process-local stack settings.')
    job.execute([str(limiter),'--stack=unlimited','--',str(binary),aip.name,aop.name],directory,
        environment={'PATH':str(binary.parent)+os.pathsep+os.environ.get('PATH',''),'KMP_STACKSIZE':'4g'})
    text=aop.read_text()
    if 'Normal termination of Amesp!' not in text:
        raise QCError('not_converged','Amesp did not finish normally; retain raw output.')
    try:
        raw=parse_spin_soc(text,nstates_per_spin=args.nstates_per_spin)
    except ValueError as exc:
        raise QCError('parse_failed',str(exc)) from exc
    raw.update(method=f'B3LYP/TDA/{args.basis}, {args.integral_mode} integrals, Amesp effective-charge SOC, gas phase',
        integral_mode=args.integral_mode,
        geometry_optimization_status='not_requested',coordinates_angstrom=atoms.positions.tolist(),
        symbols_in_source_order=atoms.get_chemical_symbols(),
        provenance={'input':str(aip),'output':str(aop),'input_sha256':hashlib.sha256(aip.read_bytes()).hexdigest(),
            'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()},
        runtime_stack={'main':'unlimited within child','openmp':'4g per thread (reservation, not measured RSS)'},
        interpretation=f'Computed {len(raw["singlet_triplet_couplings"])} approximate singlet-triplet SOC magnitudes with explicit state labels.',
        limitations=['Effective-charge spin-orbit approximation; not an all-electron relativistic SOC reference.',
            'Au uses the library def2 scalar ECP plus Amesp effective-charge SOC; absolute couplings need independent validation.',
            'Near-degenerate individual state couplings depend on the chosen state basis; compare subspace norms, not arbitrary ordinal pairs.',
            'RICOSX, when explicitly selected, is an integral approximation with Amesp-selected auxiliary basis, not a change to the declared functional.',
            'No ISC/RISC rates, triplet yields, kinetics or mechanism assignment.'])
    return raw
