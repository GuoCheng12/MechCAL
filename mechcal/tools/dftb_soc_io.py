"""Validated modern-DFTB to PySOC input representation."""

import hashlib
import json
from pathlib import Path
import re
import shutil

from ase.io import read
from ase.units import Hartree
import numpy as np


def finite(values):
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite scientific data')
    return data


def _lowest_roots(path: Path, keep: int = 3) -> str:
    """EXC.DAT with only the lowest ``keep`` roots of each spin (all of it when only those were solved)."""
    lines = path.read_text().splitlines()
    rows = [(i, l.split()) for i, l in enumerate(lines) if '->' in l.split()]
    kept = set()
    for spin in ('S', 'T'):
        manifold = sorted((float(f[0]), i) for i, f in rows if f[-1] == spin)
        kept.update(i for _, i in manifold[:keep])
    return '\n'.join(l for i, l in enumerate(lines) if '->' not in l.split() or i in kept) + '\n'


def prepare(source: Path, target: Path):
    spectra, matrices = source/'spectra', source/'matrices'
    manifest = json.loads((source/'input_manifest.json').read_text())
    if manifest['status'] != 'generated_not_yet_validated':
        raise ValueError('Cannot accept a failed generation')
    for name, digest in manifest['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Raw data changed: '+name)
    atoms = read(spectra/'geometry.xyz')
    other = read(matrices/'geometry.xyz')
    symbols = atoms.get_chemical_symbols()
    if symbols != other.get_chemical_symbols() or not np.array_equal(atoms.positions, other.positions):
        raise ValueError('Overlap and spectral geometries differ')
    if not set(symbols) <= {'C','H','N','O'}:
        raise ValueError('Only validated CHNO basis mapping is allowed')
    nelec = sum({'C':4,'H':1,'N':5,'O':6}[s] for s in symbols)
    nao = sum(1 if s=='H' else 4 for s in symbols)
    lines = [l.split() for l in (spectra/'band.out').read_text().splitlines() if l.strip()]
    if not lines or lines[0][:4] != ['KPT','1','SPIN','1'] or len(lines) != nao+1:
        raise ValueError('Unexpected orbital header/dimension')
    bands=[]
    for index, fields in enumerate(lines[1:],1):
        if len(fields)!=3 or int(fields[0])!=index:
            raise ValueError('Unknown band row or orbital index')
        bands.append([float(fields[1]),float(fields[2])])
    bands=finite(bands)
    nocc=nelec//2
    if nelec%2 or not np.array_equal(bands[:,1], [2.]*nocc+[0.]*(nao-nocc)):
        raise ValueError('Unexpected reference occupancy/valence electron count')
    if np.any(np.diff(bands[:,0]) < 0):
        raise ValueError('Unordered orbital energies')
    lines=[l.strip() for l in (matrices/'oversqr.dat').read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    if len(lines)<2 or lines[0].split()!=['T',str(nao),'1'] or lines[1]!='1':
        raise ValueError('Unknown overlap header')
    overlap=finite([l.split() for l in lines[2:]])
    if overlap.shape!=(nao,nao) or not np.allclose(overlap,overlap.T,atol=1e-12,rtol=0):
        raise ValueError('Invalid overlap matrix')
    smallest=float(np.linalg.eigvalsh(overlap)[0])
    if smallest<=0:
        raise ValueError('Nonpositive overlap')
    expected=[]
    basis=[]
    for atom,symbol in enumerate(symbols,1):
        expected.extend([(atom,symbol,'s')])
        basis.append(1)
        if symbol!='H':
            expected.extend((atom,symbol,label) for label in ('p_y','p_z','p_x'))
            basis.append(3)
    blocks=re.split(r'(?m)^Eigenvector:\s*(\d+)\s*\(up\)\s*$', (spectra/'eigenvec.out').read_text())
    if len(blocks)!=1+2*nao or blocks[0].strip()!='Coefficients and Mulliken populations of the atomic orbitals':
        raise ValueError('Unknown eigenvector layout')
    orbitals=[]
    for number,content in zip(blocks[1::2],blocks[2::2]):
        if int(number)!=len(orbitals)+1:
            raise ValueError('Nonsequential eigenvectors')
        entries=[]
        atom=symbol=None
        for line in content.splitlines():
            f=line.split()
            if not f:
                continue
            if len(f)==5:
                atom,symbol=int(f[0]),f[1]
                label,coefficient,population=f[2:]
            elif len(f)==3 and atom is not None:
                label,coefficient,population=f
            else:
                raise ValueError('Unknown eigenvector row')
            entries.append((atom,symbol,label,float(coefficient)))
            finite([population])
        if [e[:3] for e in entries]!=expected:
            raise ValueError('Atom or AO order differs from the source')
        orbitals.append([e[3] for e in entries])
    coefficients=finite(orbitals)
    gram_error=float(np.max(np.abs(coefficients@overlap@coefficients.T-np.eye(nao))))
    if gram_error>1e-4:
        raise ValueError(f'MO metric normalization failed: {gram_error}')
    transitions=[]
    for line in (spectra/'SPX.DAT').read_text().splitlines():
        f=line.split()
        if not f or f[0] in ('#',) or set(f[0])=={'='}:
            continue
        if len(f)!=6 or f[4]!='->' or int(f[0])!=len(transitions)+1:
            raise ValueError('Unknown single-particle transition row')
        finite([f[1],f[2]])
        transitions.append((int(f[3]),int(f[5])))
    product={(i,a) for i in range(1,nocc+1) for a in range(nocc+1,nao+1)}
    ndim=nocc*(nao-nocc)
    if len(transitions)!=ndim or set(transitions)!=product:
        raise ValueError('Incomplete/duplicate occupied-virtual transition map')
    state_vectors={}
    state_energies={}
    lines=[l.split() for l in (spectra/'XplusY.DAT').read_text().splitlines() if l.strip()]
    position=0
    for _ in range(2):
        # a block holds every solved root (3, or more when the solver needed more); roots 1-3 are kept
        if (position>=len(lines) or len(lines[position])!=2 or lines[position][0]!=str(ndim)
                or not lines[position][1].isdigit() or int(lines[position][1])<3):
            raise ValueError('Unknown RPA block dimensions')
        nroots=int(lines[position][1])
        position+=1
        block_spin=None
        for root in range(1,nroots+1):
            if position>=len(lines):
                raise ValueError('Missing RPA state header')
            header=lines[position]
            if len(header)!=3 or header[0]!=str(root) or header[1] not in ('S','T'):
                raise ValueError('Invalid RPA state header')
            spin=header[1]
            if block_spin is not None and spin!=block_spin:
                raise ValueError('Mixed spin block')
            block_spin=spin
            energy=float(header[2])*Hartree
            finite([energy])
            position+=1
            vector=[]
            while len(vector)<ndim:
                if position>=len(lines):
                    raise ValueError('Truncated RPA vector')
                vector.extend(float(v) for v in lines[position])
                position+=1
            if len(vector)!=ndim or (spin,root) in state_vectors or energy<=0:
                raise ValueError('Invalid RPA vector size/state/energy')
            if root<=3:
                state_vectors[(spin,root)]=finite(vector).tolist()
                state_energies[(spin,root)]=energy
    if position!=len(lines) or set(state_vectors)!={(s,i) for s in ('S','T') for i in range(1,4)}:
        raise ValueError('Unconsumed/duplicate RPA block')
    rows=manifest['states']
    if len(rows)!=6 or len({r['state'] for r in rows})!=6:
        raise ValueError('Incomplete state summary')
    for r in rows:
        if abs(state_energies[(r['spin'],int(r['state'][1:]))]-r['excitation_ev'])>0.0006:
            raise ValueError('RPA and EXC state energies disagree')
    target.mkdir(exist_ok=False)
    for name in ('geometry.xyz','SPX.DAT'):
        shutil.copyfile(spectra/name,target/name)
    (target/'EXC.DAT').write_text(_lowest_roots(spectra/'EXC.DAT'))
    shutil.copyfile(matrices/'oversqr.dat',target/'oversqr.dat')
    (target/'band.out').write_text('\n'.join(f'{e:.12g} {o:.12g}' for e,o in bands)+'\n')
    labels={'s':'s1','p_y':'p1','p_z':'p2','p_x':'p3'}
    text='Coefficients and Mulliken populations of the atomic orbitals\n\n'
    for index,values in enumerate(coefficients,1):
        text+=f'Eigenvector: {index} (up)\n'
        text+='\n'.join(f'{labels[label]} {v:.12g}' for (_,_,label),v in zip(expected,values))+'\n'
    (target/'eigenvec.out').write_text(text)
    text=''
    for spin in ('T','S'):
        text+=f'{ndim} 3\n'
        for root in range(1,4):
            text+=f'{root} {spin} {state_energies[(spin,root)]/Hartree:.12g}\n'
            text+=' '.join(f'{v:.12g}' for v in state_vectors[(spin,root)])+'\n'
    (target/'XplusY.DAT').write_text(text)
    ordered_pairs=sorted(transitions)
    indices=[transitions.index(pair) for pair in ordered_pairs]
    expected_ci=[state_vectors[(spin,root)][index] for spin in ('S','T') for root in range(1,4) for index in indices]
    canonical=dict(num_orbitals=nao,num_occupied_orbitals=nocc,num_virtual_orbitals=nao-nocc,
        MO_energies=bands[:,0].tolist(),AO_overlaps=overlap.ravel().tolist(),
        MOA_coefficients=coefficients.ravel().tolist(),CI_coefficients=expected_ci,ao_basis=basis,
        geometry=[[s,*p] for s,p in zip(symbols,atoms.positions.tolist())],
        singlet_states=[[i,next(r['excitation_ev'] for r in rows if r['state']==f'S{i}')] for i in range(1,4)],
        triplet_states=[[i,next(r['excitation_ev'] for r in rows if r['state']==f'T{i}')] for i in range(1,4)])
    (target/'canonical.json').write_text(json.dumps(canonical,indent=2))
    report=dict(status='validated_representation_only',nao=nao,occupied=nocc,transitions=ndim,
        overlap_min_eigenvalue=smallest,mo_metric_max_error=gram_error,
        raw_hashes=manifest['files'],normalization='band columns and AO labels; explicit T/S blocks; no coefficient modification',
        states=rows)
    (target/'adapter_report.json').write_text(json.dumps(report,indent=2))
    return report
