"""Protocol fixtures below are synthetic, not physical validation calculations."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from ase.units import Bohr, Hartree
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mechcal.capabilities import default_capability_registry
from mechcal.schemas.external_qc import QCArgs
from mechcal.tools.external_qc import execute_external, source_preflight
from mechcal.tools.pyscf_backend import RawCISResult, RawCISContrastResult
from mechcal.tools.pyscf_worker import METHOD, SCHEMA, VERSION, validate_request
from mechcal.tools.qc_process import QCError, QCJob
from test_external_qc import case, config, dispatch

ROUTE = 'evaluate_cis_s1_gradient'


def fixture_result(request):
    return dict(schema_version=SCHEMA, status='success', version=VERSION, method=METHOD,
        units={'energy':'hartree','gradient':'hartree/bohr','coordinates':'angstrom'},
        request=request, scf_converged=True, td_converged=[True]*3, ground_energy_hartree=-10.0,
        excitations_hartree=[.1,.2,.3], s1_total_energy_hartree=-9.9,
        gradient_hartree_per_bohr=[[.01,0.0,-.02] for _ in request['symbols']],
        ground_gradient_hartree_per_bohr=[[.02,0.0,-.01] for _ in request['symbols']],
        ground_gradient_seconds=.1,
        oscillator_strengths=[.0,.1,.2], td_residual_norms=[1e-7]*3,
        dominant_transitions=[dict(occupied=0,virtual=0,largest_normalized_weight=.8)]*3,
        scf_seconds=.1,td_seconds=.2,gradient_seconds=.3,memory_setting_mb=1000,nao=10)


def configured(config):
    config = config.model_copy(update={'pyscf_python':sys.executable})
    (Path(config.parameter_dir).parent/'config.json').write_text(config.model_dump_json())
    return config


@pytest.fixture
def request_data():
    return dict(symbols=['H','H'], coordinates_angstrom=[[0.,0.,0.],[0.,0.,.75]], cpu_threads=4)


@pytest.mark.parametrize('replacement', [
    {'symbols':['H','B']}, {'coordinates_angstrom':[[0.,0.,float('nan')],[0.,0.,1.]]},
    {'coordinates_angstrom':[[0.,0.],[0.,0.,1.]]}, {'cpu_threads':True}, {'cpu_threads':0},
    {'cpu_threads':9}, {'state_root':2}, {'symbols':[]}, {'coordinates_angstrom':None},
])
def test_worker_request_contract(request_data, replacement):
    with pytest.raises(ValueError):
        validate_request({**request_data,**replacement})


@pytest.mark.parametrize('replacement', [
    {'version':'0'}, {'method':'B3LYP'}, {'status':'failed'}, {'schema_version':'other'},
    {'units':{'energy':'eV','gradient':'hartree/bohr','coordinates':'angstrom'}},
    {'scf_converged':False}, {'td_converged':[True,True,False]}, {'td_converged':[True,'yes',True]},
    {'excitations_hartree':[.2,.1,.3]}, {'excitations_hartree':[0.,.1,.2]},
    {'gradient_hartree_per_bohr':[[.1,.2,float('nan')]]*2},
    {'gradient_hartree_per_bohr':[[.1,.2,.3]]}, {'gradient_hartree_per_bohr':[[.1,.2]]*2},
    {'oscillator_strengths':[0.,-.1,0.]}, {'oscillator_strengths':[0.,float('inf'),0.]},
    {'td_residual_norms':[1e-6,2e-4,1e-6]}, {'s1_total_energy_hartree':-10.},
    {'gradient_seconds':-1.}, {'nao':0}, {'unknown_field':1},
    {'dominant_transitions':[dict(occupied=100,virtual=0,largest_normalized_weight=.5)]*3},
    {'dominant_transitions':[dict(occupied=0,virtual=100,largest_normalized_weight=.5)]*3},
])
def test_invalid_results_are_not_evidence(request_data, replacement):
    with pytest.raises(ValueError):
        RawCISContrastResult.model_validate({**fixture_result(request_data),**replacement})


def test_old_raw_results_remain_readable_but_not_accepted_as_new_contrasts(request_data):
    legacy=fixture_result(request_data)
    legacy['schema_version']='pyscf_cis_s1_v1'
    del legacy['ground_gradient_hartree_per_bohr'],legacy['ground_gradient_seconds']
    assert RawCISResult.model_validate(legacy).status=='success'
    with pytest.raises(ValueError):
        RawCISContrastResult.model_validate(legacy)


@pytest.mark.parametrize('value',[None,[],[[0.,0.,0.]],[[0.,0.,float('nan')]]*2])
def test_incomplete_ground_gradient_is_not_a_comparison(request_data,value):
    data=fixture_result(request_data)
    data['ground_gradient_hartree_per_bohr']=value
    with pytest.raises(ValueError):
        RawCISContrastResult.model_validate(data)


@pytest.mark.parametrize('values', [{'state_root':2},{'method':'dft'},{'basis':'sto-3g'},
    {'gradient':False},{'max_wall_seconds':601},{'max_wall_seconds':True}])
def test_scientific_method_cannot_be_silently_substituted(values):
    with pytest.raises(ValueError):
        QCArgs(source_artifact_id='prepared',**values)


def test_discovery_is_read_only_and_uses_execution_scope(config, case, monkeypatch):
    monkeypatch.setattr(QCJob,'execute',lambda *args:pytest.fail('No computation in discovery'))
    registry = default_capability_registry()
    assert registry.readiness('microscopic.'+ROUTE,case.artifact_manifest,amesp_enabled=False)['status']=='disabled'
    config = configured(config)
    registry = default_capability_registry()
    before = case.model_dump_json()
    assert registry.readiness('microscopic.'+ROUTE,case.artifact_manifest,amesp_enabled=False)['status']=='ready'
    assert registry.readiness('microscopic.'+ROUTE,case.artifact_manifest,amesp_enabled=False,
        source_artifact_id='missing')['status']=='missing_input'
    assert case.model_dump_json()==before
    assert execute_external(dispatch(ROUTE),case,config=config.model_copy(update={'enabled':False})).status=='unsupported'
    card=next(c for c in registry.cards() if c['route']==ROUTE)
    assert card['tool_arg_schema']==QCArgs.model_json_schema()
    assert registry.readiness('microscopic.scan_excited_state_coordinate',case.artifact_manifest,amesp_enabled=False)['status']=='disabled'


@pytest.mark.parametrize('smiles', ['CB','[NH4+]','[CH3]','CO.CO','CCCCCCCCCCCCCC'])
def test_scope_boundaries_match_catalog_and_execution(config, case, smiles):
    config=configured(config)
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol,randomSeed=19)==0
    source=case.artifact_manifest.artifacts[0]
    Chem.MolToXYZFile(mol,source.file_paths['xyz'])
    Chem.MolToMolFile(mol,source.file_paths['sdf'])
    with pytest.raises(ValueError):
        source_preflight(case.artifact_manifest,'prepared','pyscf')
    state=default_capability_registry().readiness('microscopic.'+ROUTE,case.artifact_manifest,amesp_enabled=False)
    assert state['status']=='missing_input'
    result=execute_external(dispatch(ROUTE),case,config=config)
    assert result.status=='precondition_missing' and not result.artifact_updates
    assert result.evidence_units[0].basis=='checklist'


def test_adapter_preserves_atoms_and_units(config,case,monkeypatch):
    config=configured(config)
    def stub(self,argv,directory):
        request=json.loads(Path(argv[-2]).read_text())
        Path(argv[-1]).write_text(json.dumps(fixture_result(request)))
        return 'Synthetic adapter fixture, not chemistry.'
    monkeypatch.setattr(QCJob,'execute',stub)
    before=case.model_dump_json()
    result=execute_external(dispatch(ROUTE),case,config=config)
    assert result.status=='success'
    obs=result.structured_results['parsed_observables']
    gradient=np.asarray(obs['s1_gradient_ev_per_angstrom'])
    assert gradient[0,0]==pytest.approx(.01*Hartree/Bohr)
    assert np.array_equal(np.asarray(obs['s1_forces_ev_per_angstrom']),-gradient)
    ground=np.asarray(obs['s0_gradient_ev_per_angstrom'])
    difference=np.asarray(obs['excitation_gradient_ev_per_angstrom'])
    assert np.array_equal(np.asarray(obs['s0_forces_ev_per_angstrom']),-ground)
    assert np.array_equal(difference,gradient-ground)
    assert np.array_equal(np.asarray(obs['s1_minus_s0_force_ev_per_angstrom']),-difference)
    assert obs['largest_force_changes'][0]['atom_index']==0
    assert difference[0,0]==pytest.approx(-.01*Hartree/Bohr)
    assert obs['states'][0]['excitation_ev']==pytest.approx(.1*Hartree)
    assert result.evidence_units[0].relation=='unknown' and result.evidence_units[0].support=='not_applicable'
    assert case.model_dump_json()==before
    assert len(obs['provenance']['worker_sha256'])==64


@pytest.mark.parametrize('kind',['missing','geometry','malformed','timeout','process_failed'])
def test_adapter_failures_do_not_become_zero_or_success(config,case,monkeypatch,kind):
    config=configured(config)
    def stub(self,argv,directory):
        if kind in {'timeout','process_failed'}:
            raise QCError(kind,'synthetic backend failure')
        if kind=='missing':
            return ''
        request=json.loads(Path(argv[-2]).read_text())
        data=fixture_result(request)
        if kind=='geometry':
            data=deepcopy(data)
            data['request']['coordinates_angstrom'][0][0]+=.01
        Path(argv[-1]).write_text('not JSON' if kind=='malformed' else json.dumps(data))
        return ''
    monkeypatch.setattr(QCJob,'execute',stub)
    result=execute_external(dispatch(ROUTE),case,config=config)
    assert result.status=='failed' and not result.artifact_updates
    assert result.structured_results['boundary_only'] and result.evidence_units[0].basis=='checklist'
