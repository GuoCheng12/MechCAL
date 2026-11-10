"""Spin-table parser and MCP adapter contracts; stubs are not chemistry results."""

from pathlib import Path

import pytest

from mechcal.capabilities import default_capability_registry
from mechcal.schemas.external_qc import SpinComparisonArgs
from mechcal.tools.dftb_spin import compare_states, parse_states, spin_parameters
from mechcal.tools.external_qc import execute_external
from mechcal.tools.qc_process import QCError, QCJob
from test_external_qc import case, config, dispatch

# Rows from the recorded Case0019 DFTB+ feasibility run.
TABLE = """
w [eV] Osc.Str. Transition Weight KS [eV] Sym.
3.325 0.00000000 39 -> 40 1.000 3.365 T
3.640 0.00000000 38 -> 40 0.997 3.756 T
3.886 0.00000000 37 -> 40 0.996 3.997 T
3.527 0.10162673 39 -> 40 0.998 3.365 S
3.973 0.07588114 38 -> 40 0.995 3.756 S
4.222 0.05903387 37 -> 40 0.990 3.997 S
"""
SPIN = "H:\n -0.0717\nC:\n -0.0306 -0.0251\n -0.0251 -0.0227\nO:\n -0.0352 -0.0296\n -0.0296 -0.0278\n"


def parameter_fixture(config):
    root = Path(config.parameter_dir)
    extras = root.parent / 'extras'
    extras.mkdir(exist_ok=True)
    (extras / 'spinw.txt').write_text(SPIN)
    for a in 'CHO':
        for b in 'CHO':
            (root / f'{a}-{b}.skf').write_text('synthetic parameter availability fixture, not for calculation')
    return config.model_copy(update={'parameter_revision': 'synthetic-test-revision'})


def test_state_labels_are_spin_resolved_not_line_numbers():
    states = parse_states(TABLE, 3)
    assert [s['state'] for s in states] == ['S1', 'S2', 'S3', 'T1', 'T2', 'T3']
    assert states[0]['excitation_ev'] - states[3]['excitation_ev'] == pytest.approx(.202)
    assert states[3]['oscillator_strength'] == 0


@pytest.mark.parametrize('text', [TABLE.replace(' T', ' S'), TABLE.replace(' T', ' X'),
    TABLE.replace('3.325', 'nan'), TABLE.replace('3.325', '-1'),
    TABLE.replace('0.10162673', '-1'), TABLE.replace('0.10162673', 'inf'),
    TABLE.replace('1.000', '1.500'), TABLE.replace('39 -> 40', '39 -> 38'),
    TABLE.replace('39 -> 40', '39 40'), TABLE.replace('3.365 T', '3.365 T extra'), '',
    '\n'.join(TABLE.splitlines()[:-1]), TABLE + TABLE.splitlines()[2], TABLE.replace('[eV]', '[Ha]')])
def test_incomplete_or_invalid_table_is_failure_not_zero(text):
    with pytest.raises(QCError) as error:
        parse_states(text, 3)
    assert error.value.code == 'parse_failed'


@pytest.mark.parametrize('arguments', [{'nstates_per_spin': 0}, {'nstates_per_spin': 1}, {'nstates_per_spin': 2}, {'nstates_per_spin': 7},
    {'nstates_per_spin': True}, {'nstates_per_spin': 3.0}, {'state_root': 1},
    {'max_wall_seconds': 601}])
def test_argument_contract(arguments):
    with pytest.raises(ValueError):
        SpinComparisonArgs(source_artifact_id='prepared', **arguments)


def test_parameter_provenance_and_matrix_validation(config):
    config = parameter_fixture(config)
    matrices, provenance = spin_parameters(config, ['C', 'H', 'O'])
    assert matrices['C'] == [-.0306, -.0251, -.0251, -.0227]
    assert len(provenance['spin_constants_sha256']) == 64
    assert len(provenance['slater_koster_sha256']) == 9
    path = Path(provenance['spin_constants_file'])
    for text in [SPIN.replace(' -0.0306 -0.0251', ' -0.0306'),
                 SPIN.replace('-0.0306', 'nan'), SPIN.replace('-0.0306 -0.0251', '-0.0306 -0.0100'),
                 SPIN + 'H:\n -0.0717\n', 'invalid header\n' + SPIN]:
        path.write_text(text)
        with pytest.raises(ValueError):
            spin_parameters(config, ['C', 'H', 'O'])
    path.write_text(SPIN)
    (Path(config.parameter_dir) / 'C-H.skf').unlink()
    with pytest.raises(ValueError, match='Cannot read'):
        spin_parameters(config, ['C', 'H', 'O'])


def test_readiness_matches_execution_without_computation(config, case, monkeypatch):
    request = dispatch('compare_singlet_triplet_states')
    before = case.model_dump_json()
    monkeypatch.setattr(QCJob, 'execute', lambda *args: pytest.fail('Preflight must not compute'))
    registry = default_capability_registry()
    assert registry.readiness(request.capability_id, case.artifact_manifest, amesp_enabled=False)['status'] == 'missing_input'
    result = execute_external(request, case, config=config)
    assert result.status == 'precondition_missing'
    assert result.evidence_units[0].basis == 'checklist'
    assert not result.artifact_updates and case.model_dump_json() == before
    ready = parameter_fixture(config)
    configfile = Path(config.parameter_dir).parent / 'config.json'
    configfile.write_text(ready.model_dump_json())
    registry = default_capability_registry()
    assert registry.readiness(request.capability_id, case.artifact_manifest, amesp_enabled=False)['status'] == 'ready'
    assert execute_external(request, case, config=ready.model_copy(update={'enabled': False})).status == 'unsupported'
    case.artifact_manifest.artifacts[0].metadata['charge'] = 1
    assert execute_external(request, case, config=ready).status == 'precondition_missing'


def test_registered_schema_is_same_as_execution(config):
    details = default_capability_registry().cards()
    card = next(c for c in details if c['route'] == 'compare_singlet_triplet_states')
    assert card['tool_arg_schema'] == SpinComparisonArgs.model_json_schema()
    assert card['tool_arg_schema']['properties']['nstates_per_spin']['maximum'] == 6


def test_adapter_state_result_and_missing_table(config, case, monkeypatch):
    config = parameter_fixture(config)
    def stub(self, argv, directory):
        assert 'Symmetry = Both' in (directory/'dftb_in.hsd').read_text()
        assert 'StateOfInterest' not in (directory/'dftb_in.hsd').read_text()
        (directory/'EXC.DAT').write_text(TABLE)
        return 'synthetic parser/adapter fixture'
    monkeypatch.setattr(QCJob, 'execute', stub)
    result = execute_external(dispatch('compare_singlet_triplet_states'), case, config=config)
    assert result.status == 'success'
    data = result.structured_results['parsed_observables']
    assert len(data['spin_gaps']) == 9
    assert data['s1_minus_t1_ev'] == pytest.approx(.202)
    assert result.evidence_units[0].relation == 'unknown'
    monkeypatch.setattr(QCJob, 'execute', lambda *args: 'no table')
    result = execute_external(dispatch('compare_singlet_triplet_states'), case, config=config)
    assert result.status == 'failed' and result.structured_results['failure_code'] == 'parse_failed'
    assert not result.artifact_updates


def test_nonconvergence_is_not_used(config, case, monkeypatch):
    config = parameter_fixture(config)
    monkeypatch.setattr(QCJob, 'execute', lambda *args: 'SCC is NOT converged')
    result = execute_external(dispatch('compare_singlet_triplet_states'), case, config=config)
    assert result.status == 'failed'
    assert result.structured_results['failure_code'] == 'not_converged'
