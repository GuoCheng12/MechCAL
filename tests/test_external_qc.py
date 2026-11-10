"""Protocol tests use synthetic observables only to test adapters, not chemistry."""
import json
import sys
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from mechcal.capabilities import default_capability_registry
from mechcal.schemas import ArtifactRecord, ArtifactManifest, CaseInput, CaseRun, DispatchRequest, EvidenceLedger
from mechcal.schemas.external_qc import ExcitedScanArgs
from mechcal.tools.external_qc import execute_external, source_preflight
from mechcal.tools.qc_process import QCConfig, QCError, QCJob


@pytest.fixture
def config(tmp_path, monkeypatch):
    params = tmp_path/'params'
    params.mkdir()
    config = QCConfig(enabled=True, xtb_binary=sys.executable, dftb_binary=sys.executable,
                      parameter_dir=str(params))
    path = tmp_path/'config.json'
    path.write_text(config.model_dump_json())
    monkeypatch.setenv('MECHCAL_QC_CONFIG', str(path))
    return config


@pytest.fixture
def case(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles('CO'))
    assert AllChem.EmbedMolecule(mol, randomSeed=19) == 0
    Chem.MolToXYZFile(mol, str(tmp_path/'a.xyz'))
    Chem.MolToMolFile(mol, str(tmp_path/'a.sdf'))
    artifact = ArtifactRecord(artifact_id='prepared', kind='prepared_structure',
        created_by_round='R000', created_by_agent='structure', metadata={'charge': 0, 'multiplicity': 1},
        file_paths={'xyz': str(tmp_path/'a.xyz'), 'sdf': str(tmp_path/'a.sdf')})
    return CaseRun(case_id='test', input=CaseInput(case_id='test', smiles='CO', user_query='test'),
        evidence_ledger=EvidenceLedger(case_id='test'),
        artifact_manifest=ArtifactManifest(case_id='test', artifacts=[artifact]))


def dispatch(route='estimate_vertical_ip_ea', **kwargs):
    spec = default_capability_registry().require('microscopic.'+route)
    return DispatchRequest(dispatch_id='D1', round_id='R001', agent_name='microscopic',
        capability_id=spec.capability_id, route=route, evidence_goal_family=spec.evidence_family,
        task='test', objective='test', tool_args={'source_artifact_id': 'prepared', **kwargs})


def test_opt_in_catalog_readiness_and_isolation(config, case, monkeypatch):
    registry = default_capability_registry()
    from mechcal.capabilities.external_qc import ROUTES
    assert len(registry.snapshot().capabilities) == 53 + len(ROUTES)
    before = case.model_dump_json()
    details = registry.describe_tools(['microscopic.evaluate_excited_state_gradient'],
                                      case.artifact_manifest, amesp_enabled=False)
    assert details[0]['status'] == 'ready'
    assert 'state_root' in details[0]['tool_arg_schema']['properties']
    assert case.model_dump_json() == before
    assert registry.readiness('microscopic.evaluate_excited_state_gradient',
        ArtifactManifest(case_id='empty'), amesp_enabled=True)['status'] == 'missing_input'
    assert registry.readiness('microscopic.scan_excited_state_coordinate',
        case.artifact_manifest, amesp_enabled=True)['status'] == 'disabled'
    assert registry.for_owner_agents(('macro',)).get('microscopic.estimate_vertical_ip_ea') is None
    monkeypatch.delenv('MECHCAL_QC_CONFIG')
    assert len(default_capability_registry().snapshot().capabilities) == 53


@pytest.mark.parametrize('args', [dict(values=[0.0, 360.0]), dict(atom_indices=[0, 1, 1, 2]),
                               dict(state_root=3, nstates=3), dict(max_wall_seconds=999),
                               dict(values=[float('nan'), 1.0]), dict(arbitrary_shell='x')])
def test_argument_limits(args):
    values = dict(source_artifact_id='a', coordinate='dihedral', atom_indices=[0, 1, 2, 3], values=[0., 15.])
    with pytest.raises(ValueError):
        ExcitedScanArgs.model_validate({**values, **args})


def test_missing_source_and_disabled_return_status_only(config, case):
    bad = dispatch(source_artifact_id='absent')
    result = execute_external(bad, case, config=config)
    assert result.status == 'precondition_missing'
    assert result.evidence_units[0].basis == 'checklist'
    assert not result.artifact_updates
    result = execute_external(dispatch(), case, config=config.model_copy(update={'enabled': False}))
    assert result.status == 'unsupported'
    result = execute_external(dispatch('scan_excited_state_coordinate'), case, config=config)
    assert result.status == 'unsupported'


def test_coordinate_disagreement(config, case):
    from ase.io import read, write
    source = case.artifact_manifest.artifacts[0]
    atoms = read(source.file_paths['xyz'])
    atoms.positions[0, 0] += 0.1
    write(source.file_paths['xyz'], atoms)
    with pytest.raises(ValueError, match='coordinates disagree'):
        source_preflight(case.artifact_manifest, 'prepared', 'xtb')


def test_discovery_checks_explicit_dimer_partner(config, case):
    req = dispatch('search_dimer_contacts', partner_artifact_id='absent')
    violations = default_capability_registry().availability_violations(
        req, case.artifact_manifest, amesp_enabled=False)
    assert violations and 'absent' in ' '.join(violations)
    result = execute_external(req, case, config=config)
    assert result.status == 'precondition_missing'


def test_empty_geometry_does_not_crash_discovery_or_execution(config, case):
    from pathlib import Path
    Path(case.artifact_manifest.artifacts[0].file_paths['xyz']).write_text('')
    state = default_capability_registry().readiness('microscopic.estimate_vertical_ip_ea',
                                                  case.artifact_manifest, amesp_enabled=False)
    assert state['status'] == 'missing_input'
    result = execute_external(dispatch(), case, config=config)
    assert result.status == 'precondition_missing'
    assert result.structured_results['boundary_only']


def test_success_contract_and_typed_backend_failure(config, case, monkeypatch):
    def stub(job, atoms):
        return {'method': 'synthetic protocol fixture', 'interpretation': 'Protocol test only.',
                'limitations': ['Not chemistry.'], 'value': 1.0}
    monkeypatch.setattr('mechcal.tools.xtb_backend.ip_ea', stub)
    result = execute_external(dispatch(), case, config=config)
    assert result.status == 'success'
    assert result.evidence_units[0].source_report_id == result.tool_result_id
    assert result.evidence_units[0].summary
    assert result.structured_results['timing']['gpu_used'] is False
    def fail(job, atoms):
        raise QCError('not_converged', 'synthetic failed SCC')
    monkeypatch.setattr('mechcal.tools.xtb_backend.ip_ea', fail)
    result = execute_external(dispatch(), case, config=config)
    assert result.status == 'failed'
    assert result.structured_results['failure_code'] == 'not_converged'
    assert result.evidence_units[0].status == 'failed'
    assert result.failure.recoverable
    assert result.failure.details['code'] == 'not_converged'
    assert result.failure.details['automatic_retry'] is False


@pytest.mark.parametrize('code,recoverable', [('not_converged', True), ('timeout', True), ('unsupported', False)])
def test_member_failure_reaches_worker_without_losing_reason(config, case, monkeypatch, code, recoverable):
    calls = []
    def failed_search(*args):
        calls.append(args)
        return {'method': 'synthetic contract fixture', 'interpretation': 'No usable candidates.',
                'limitations': ['Not chemistry.'], 'candidates': [], 'search_attempts': [
                    {'search_index': 1, 'status': 'failed', 'failure_code': code, 'reason': 'Specific member reason'}]}
    monkeypatch.setattr('mechcal.tools.xtb_backend.dimer', failed_search)
    result = execute_external(dispatch('search_dimer_contacts', partner_artifact_id='prepared'), case, config=config)
    assert result.status == 'failed'
    assert result.failure.recoverable is recoverable
    assert result.failure.details['code'] == code
    assert result.failure.message == 'Specific member reason'
    assert result.failure.details['member_failures'][0]['search_index'] == 1
    assert result.evidence_units[0].status == 'failed'
    assert len(calls) == 1


def test_partial_scan_keeps_successful_point(config, case, monkeypatch):
    def mixed(*args):
        return {'method': 'synthetic contract fixture', 'interpretation': 'One point succeeded.',
                'limitations': ['Not chemistry.'], 'points': [
                    {'coordinate_value': 0., 'converged': True, 'energy_ev': 1.25},
                    {'coordinate_value': 30., 'converged': False}]}
    monkeypatch.setattr('mechcal.tools.dftb_backend.scan', mixed)
    result = execute_external(dispatch('scan_excited_state_coordinate', coordinate='dihedral',
        atom_indices=[0, 1, 2, 3], values=[0., 30.]), case,
        config=config.model_copy(update={'allow_excited_scan': True}))
    assert result.status == 'partial'
    assert result.structured_results['parsed_observables']['points'][0]['energy_ev'] == 1.25
    assert result.failure.kind == 'partial_evidence' and result.failure.recoverable
    assert result.failure.details['code'] == 'not_converged'
    assert result.artifact_updates[0].status == 'partial'


def test_deadline_kills_process_group_and_keeps_log(config, tmp_path):
    import os
    import time
    job = QCJob(tmp_path/'job', config, .2)
    script = 'import subprocess,time; p=subprocess.Popen(["sleep","30"]); print(p.pid,flush=True); time.sleep(30)'
    with pytest.raises(QCError, match='process group killed'):
        job.execute([sys.executable, '-c', script], job.workdir/'step')
    assert job.records[0]['timed_out']
    assert (job.workdir/'process_records.json').exists()
    pid = int((job.workdir/'step/process.log').read_text().strip())
    # A killed orphan may briefly remain a zombie until its parent reaps it.
    stat = __import__('pathlib').Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().split()[2] == 'Z'


def test_process_error_is_not_zero_energy(config, tmp_path):
    job = QCJob(tmp_path/'job', config, 5)
    with pytest.raises(QCError, match='Exit 3'):
        job.execute([sys.executable, '-c', 'raise SystemExit(3)'], job.workdir/'step')
    assert job.records[0]['returncode'] == 3
    assert 'peak_rss_kb' in job.records[0]
