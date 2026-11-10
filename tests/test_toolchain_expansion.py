"""Contract and physical bookkeeping tests, distinct from numerical acceptance."""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
from ase.io import read, write
from rdkit import Chem

from test_external_qc import case, config, dispatch
from mechcal.schemas.external_qc import BroadSpinArgs, ContactMotionArgs, ConformerSamplingArgs
from mechcal.tools.external_qc import execute_external, source_preflight
from mechcal.tools.contact_motion import moving_side
from mechcal.tools.qc_process import QCError


@pytest.mark.parametrize('model,fields', [
    (BroadSpinArgs, {'fragment_atom_indices':[[0],[0,1]]}),
    (BroadSpinArgs, {'nstates_per_spin':0}),
    (BroadSpinArgs, {'basis':'arbitrary'}),
    (ContactMotionArgs, {'atom_indices':[0,1,2,3], 'angles_degrees':[-180.,180.]}),
    (ContactMotionArgs, {'atom_indices':[0,1,2,3], 'angles_degrees':[float('nan'),0.]}),
    (ConformerSamplingArgs, {'frozen_atom_indices':[0,0]}),
    (ConformerSamplingArgs, {'max_members':1000}),
])
def test_arguments(model, fields):
    with pytest.raises(ValueError):
        model.model_validate({'source_artifact_id':'a', **fields})


def test_torsion_side_is_from_graph_not_atom_guess():
    mol=Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    moving=moving_side(mol,[0,1,2,3],list(range(mol.GetNumAtoms())))
    assert 3 in moving and 0 not in moving and 1 not in moving
    with pytest.raises(ValueError):
        moving_side(Chem.MolFromSmiles('C1CCCCC1'),[0,1,2,3],list(range(6)))
    with pytest.raises(ValueError):
        moving_side(mol,[0,1,2,3],[0,1,2])


def test_contact_fixed_geometry_is_not_failed_optimization(tmp_path):
    from ase import Atoms
    from mechcal.schemas.external_qc import ContactScanArgs
    from mechcal.tools import xtb_backend
    from unittest.mock import patch
    from types import SimpleNamespace
    job=SimpleNamespace(workdir=tmp_path,deadline=float('inf'))
    args=ContactScanArgs(source_artifact_id='a',atom_indices=[0,1],distances_angstrom=[2.,3.])
    with patch.object(xtb_backend,'contact_observables',return_value={}):
        rows=xtb_backend.scan_contacts(job,Atoms('HH',positions=[[0,0,0],[0,0,2]]),args,[[0],[1]])['points']
    assert all(r['status']=='success' and r['geometry_converged'] is None and r['geometry_optimization_status']=='not_requested' for r in rows)


def test_new_route_disabled_or_missing_does_not_run(config,case,monkeypatch):
    called=[]
    monkeypatch.setattr('mechcal.tools.pyscf_states.evaluate',lambda *a:called.append(a))
    result=execute_external(dispatch('compare_spin_states_pyscf'),case,config=config)
    assert result.status=='unsupported' and not called
    cfg=config.model_copy(update={'pyscf_python':sys.executable})
    result=execute_external(dispatch('compare_spin_states_pyscf',source_artifact_id='another-case'),case,config=cfg)
    assert result.status=='precondition_missing' and not called


def test_new_routes_reject_monomer_contact_and_invalid_partition(config,case):
    result=execute_external(dispatch('compare_contact_torsion_response',atom_indices=[0,1,2,3],angles_degrees=[0.,60.]),case,config=config)
    assert result.status=='precondition_missing'
    cfg=config.model_copy(update={'pyscf_python':sys.executable})
    result=execute_external(dispatch('compare_spin_states_pyscf',fragment_atom_indices=[[0]]),case,config=cfg)
    assert result.status=='failed' and result.failure.details['code']=='invalid_args'


def test_partial_manifold_does_not_disappear(config,case,monkeypatch):
    def partial(*a):
        return dict(method='Synthetic contract fixture',interpretation='One manifold only',limitations=['Not chemistry'],
            states=[{'state':'S1'}],points=[{'status':'success'},{'status':'failed','failure_code':'timeout','reason':'T timed out'}])
    monkeypatch.setattr('mechcal.tools.pyscf_states.evaluate',partial)
    result=execute_external(dispatch('compare_spin_states_pyscf'),case,config=config.model_copy(update={'pyscf_python':sys.executable}))
    assert result.status=='partial' and result.structured_results['parsed_observables']['states']
    assert result.failure.details['successful_results_preserved']


def test_dimer_source_mismatch_rejected_before_execution(config,case):
    source=case.artifact_manifest.artifacts[0]
    source.metadata.update(modeled_dimer=True,fragment_atom_indices=[[0,1],[2,3]])
    result=execute_external(dispatch('compare_contact_torsion_response',atom_indices=[0,1,2,3],angles_degrees=[0.,60.]),case,config=config)
    assert result.status=='precondition_missing'


def test_amesp_ablation_cannot_be_bypassed_by_external_route(config,case,monkeypatch):
    from mechcal.tools.local_micro import LocalMicroscopicTool
    cfg=config.model_copy(update={'amesp_soc_binary':sys.executable,'allow_amesp_soc':True})
    path=Path(__import__('os').environ['MECHCAL_QC_CONFIG'])
    path.write_text(cfg.model_dump_json())
    monkeypatch.setenv('MECHCAL_ENABLE_AMESP','true')
    result=LocalMicroscopicTool(enable_amesp=False).run(dispatch('estimate_soc_amesp'),case)
    assert result.status=='unsupported' and not result.artifact_updates


def test_missing_resource_measurement_is_not_zero_memory(config,tmp_path):
    from mechcal.tools.qc_process import QCJob
    job=QCJob(tmp_path/'job',config,10)
    job.records=[{'wall_seconds':1.,'timed_out':True}]
    assert job.timing()['peak_process_rss_kb'] is None
    assert job.timing()['rss_measured_process_count']==0


def test_soc_stack_settings_are_child_local(config,tmp_path,monkeypatch):
    from ase import Atoms
    from mechcal.schemas.external_qc import AmespSOCArgs
    from mechcal.tools import amesp_soc_backend
    from mechcal.tools.qc_process import QCJob
    import os
    before=dict(os.environ)
    job=QCJob(tmp_path/'job',config.model_copy(update={'amesp_soc_binary':sys.executable}),10)
    captured={}
    def execute(argv,directory,*,environment):
        captured.update(argv=argv,environment=environment)
        (directory/'soc.aop').write_text('Normal termination of Amesp!')
        return ''
    monkeypatch.setattr(job,'execute',execute)
    monkeypatch.setattr(amesp_soc_backend,'parse_spin_soc',lambda *a,**k: {'singlet_triplet_couplings':[]})
    amesp_soc_backend.evaluate(job,Atoms('HBr',positions=[[0,0,0],[0,0,1.41]]),AmespSOCArgs(source_artifact_id='a'))
    assert captured['argv'][:3]==['/usr/bin/prlimit','--stack=unlimited','--']
    assert captured['environment']['KMP_STACKSIZE']=='4g'
    assert dict(os.environ)==before


def test_states_survive_a_late_stability_timeout(config,tmp_path,monkeypatch):
    from ase import Atoms
    from mechcal.tools import pyscf_states
    from mechcal.tools.qc_process import QCJob
    job=QCJob(tmp_path/'job',config,10)
    def execute(argv,directory,**kwargs):
        request=json.loads((directory/'request.json').read_text())
        (directory/'result.json').write_text(json.dumps({
            'schema_version':'pyscf_spin_states_v1','request':request,'scf_converged':True,
            'stability_check_status':'not_completed','states':[
                {'state':'S1','excitation_ev':2.,'td_residual_norm':1e-8,
                 'hole_population_by_atom':[.5,.5],'electron_population_by_atom':[.5,.5]}]}))
        raise QCError('timeout','Stability check exceeded time allowance.')
    monkeypatch.setattr(job,'execute',execute)
    result=pyscf_states.evaluate(job,Atoms('H2',positions=[[0,0,0],[0,0,.74]]),BroadSpinArgs(source_artifact_id='a'),0)
    assert len(result['states'])==1
    assert result['points'][0]['status']=='success'
    assert any(p['status']=='failed' for p in result['points'])
    assert 'stability is not established' in result['interpretation']


@pytest.mark.parametrize('seconds,expected',[(None,330.),(900,930.),(1800,1830.),(1801,120.),('900',120.)])
def test_mcp_wait_uses_only_validated_computation_deadlines(seconds,expected,config):
    from mechcal.tools import StdioMcpTransport
    request=dispatch('compare_spin_states_pyscf')
    if seconds is not None:
        request.tool_args['max_wall_seconds']=seconds
    transport=StdioMcpTransport()
    args={'dispatch_request':request.model_dump(mode='json'),'execution_plan':None}
    assert transport._call_timeout('run_microscopic',args)==expected
    assert transport._call_timeout('list_capabilities',args)==120.


def test_mcp_deadline_does_not_override_explicit_transport_policy(config):
    from dataclasses import replace
    from mechcal.tools import StdioMcpTransport, StdioMcpServerConfig
    config=StdioMcpServerConfig.current_python_server()
    args={'dispatch_request':dispatch('compare_spin_states_pyscf',max_wall_seconds=900).model_dump(mode='json')}
    assert StdioMcpTransport(replace(config,honor_tool_deadlines=False))._call_timeout('run_microscopic',args)==120.
    assert StdioMcpTransport(replace(config,timeout_seconds=1900))._call_timeout('run_microscopic',args)==1900.
    args['dispatch_request']['capability_id']='microscopic.other'
    assert StdioMcpTransport(config)._call_timeout('run_microscopic',args)==120.
