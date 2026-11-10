"""Fixed-coordinate CIS path with explicit transition-density state overlap."""
import json
from pathlib import Path
import sys
import numpy as np
from pyscf import gto, scf, lib, __version__


def run(request, output):
    if __version__ != '2.14.0':
        raise ValueError('Requires validated PySCF 2.14.0')
    lib.num_threads(request['threads'])
    previous=None
    continuity=True
    rows=[]
    for index, xyz in enumerate(request['coordinates_angstrom']):
        mol=gto.M(atom=list(zip(request['symbols'],xyz)),basis='6-31g',spin=0,charge=0,
            verbose=4,max_memory=1000,output=str(output.with_name(f'point_{index}.log')))
        mf=scf.RHF(mol)
        mf.conv_tol=1e-10; mf.max_cycle=150
        mf.kernel()
        if not mf.converged:
            raise ValueError(f'SCF not converged at point {index}')
        td=mf.TDA(); td.nstates=3; td.conv_tol=1e-6; td.max_cycle=150
        td.kernel()
        if len(td.e)!=3 or not all(td.converged) or min(td.e)<=0:
            raise ValueError(f'Incomplete/invalid CIS states at point {index}')
        vind,_=td.gen_vind(mf)
        vectors=np.asarray([xy[0].ravel() for xy in td.xy])
        vectors/=np.linalg.norm(vectors,axis=1)[:,None]
        residuals=np.linalg.norm(vind(vectors)-td.e[:,None]*vectors,axis=1)
        if max(residuals)>2e-5:
            raise ValueError(f'CIS residual check failed at point {index}')
        occupied=mf.mo_coeff[:,mf.mo_occ>0]; virtual=mf.mo_coeff[:,mf.mo_occ==0]
        densities=[occupied@v.reshape(occupied.shape[1],virtual.shape[1])@virtual.T for v in vectors]
        if previous is None:
            selected=request['state_root']-1; overlaps=None; confidence='initial_ordinal_root'
        else:
            old_mol,old_density=previous
            cross=gto.intor_cross('int1e_ovlp',old_mol,mol)
            overlaps=[float(abs(np.einsum('ij,ij',old_density,cross@d@cross.T))) for d in densities]
            order=np.argsort(overlaps)[::-1]; selected=int(order[0])
            confidence='continuous' if overlaps[selected]>=.7 and overlaps[selected]-overlaps[int(order[1])]>=.1 else 'ambiguous'
        continuity = continuity and confidence != 'ambiguous'
        strengths=td.oscillator_strength()
        row=dict(member_id=f'path:{index+1:02d}',status='success',point_index=index,
            ground_energy_hartree=float(mf.e_tot),states=[dict(state=f'S{i+1}',excitation_ev=float(e*27.211386245988),
                oscillator_strength=float(strengths[i]),residual=float(residuals[i])) for i,e in enumerate(td.e)],
            selected_state=f'S{selected+1}',selected_total_energy_hartree=float(mf.e_tot+td.e[selected]),
            state_tracking_status=confidence,path_continuity_valid=continuity,transition_density_overlaps=overlaps)
        rows.append(row)
        previous=(mol,densities[selected])
        output.write_text(json.dumps({'schema_version':'cis_path_v1','request':request,'points':rows},indent=2,allow_nan=False))


if __name__=='__main__':
    source,output=map(Path,sys.argv[1:])
    try:
        run(json.loads(source.read_text()),output)
    except Exception as exc:
        output.with_suffix('.failure.json').write_text(json.dumps({'reason':str(exc)}))
        raise
