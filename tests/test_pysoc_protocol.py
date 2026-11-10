"""Synthetic contracts only; real chemistry is covered by the recorded MCP audit."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from mechcal.capabilities import default_capability_registry
from mechcal.schemas.external_qc import QCArgs
from mechcal.tools.dftb_soc import export_overlap
from mechcal.tools.external_qc import execute_external
from mechcal.tools.pysoc_protocol import RawSOC, METHOD
from mechcal.tools.pysoc_source import SOURCE_HASHES, source_reason
from mechcal.tools.qc_process import QCError, QCJob
from test_external_qc import config, case, dispatch

ROUTE='evaluate_spin_orbit_couplings'


def raw():
    return dict(schema_version='pysoc_zeff_v1',status='success',pysoc_version='2.3.0',
        method=METHOD,unit='cm^-1',operator='ZEFF',scale=1.0,ci_threshold=1e-5,
        canonical_input_sha256='0'*64,source_hashes=deepcopy(SOURCE_HASHES),
        couplings=[dict(singlet=s,triplet=t,rss_cm_inverse=5.,components_cm_inverse=[3.,4.,0.])
                   for s in range(4) for t in range(1,4)])


def configured(config,monkeypatch):
    config=config.model_copy(update={'pysoc_python':sys.executable,'pysoc_source':'synthetic'})
    monkeypatch.setattr('mechcal.tools.pysoc_source.source_reason',lambda source:None)
    monkeypatch.setattr('mechcal.tools.dftb_spin.spin_parameters',lambda *a:({},{}))
    (Path(config.parameter_dir).parent/'config.json').write_text(config.model_dump_json())
    return config


def test_valid_protocol():
    assert len(RawSOC.model_validate(raw()).couplings)==12


@pytest.mark.parametrize('change',[
    {'schema_version':'other'},{'status':'failed'},{'pysoc_version':'2.2.0'},
    {'method':'other'},{'unit':'eV'},{'operator':'ONE'},{'scale':2.0},
    {'ci_threshold':.1},{'canonical_input_sha256':'invalid'},{'source_hashes':{}},
    {'couplings':[]},{'extra':1},
])
def test_invalid_protocol(change):
    with pytest.raises(ValueError):
        RawSOC.model_validate({**raw(),**change})


@pytest.mark.parametrize('change',[
    {'singlet':4},{'triplet':0},{'singlet':'1'},
    {'rss_cm_inverse':4.0},{'rss_cm_inverse':float('nan')},
    {'components_cm_inverse':[3.,-4.,0.]},{'components_cm_inverse':[3.,4.]},
    {'components_cm_inverse':[float('inf'),0.,0.]},
])
def test_invalid_pairs(change):
    data=raw()
    data['couplings'][0].update(change)
    with pytest.raises(ValueError):
        RawSOC.model_validate(data)


def test_duplicate_pairs():
    data=raw()
    data['couplings'][1]=data['couplings'][0]
    with pytest.raises(ValueError,match='missing or duplicated'):
        RawSOC.model_validate(data)


@pytest.mark.parametrize('change',[{'operator':'ONE'},{'nstates':8},{'charge':1},{'scale':2.0},
                                  {'max_wall_seconds':601},{'max_wall_seconds':True}])
def test_method_not_overridden(change):
    with pytest.raises(ValueError):
        QCArgs(source_artifact_id='prepared',**change)


def test_unconfigured_and_source_tamper(tmp_path,config):
    assert config.backend_reason('pysoc') is not None
    assert source_reason('') is not None
    assert source_reason(str(tmp_path)) is not None
    (tmp_path/'pysoc').mkdir()
    (tmp_path/'pysoc/__init__.py').write_text('not the validated implementation')
    assert 'differs' in source_reason(str(tmp_path))


def test_discovery_execution_boundaries(config,case,monkeypatch):
    monkeypatch.setattr(QCJob,'execute',lambda *a:pytest.fail('Discovery must not calculate'))
    before=case.model_dump_json()
    registry=default_capability_registry()
    assert registry.readiness('microscopic.'+ROUTE,case.artifact_manifest,amesp_enabled=False)['status']=='disabled'
    config=configured(config,monkeypatch)
    registry=default_capability_registry()
    assert registry.readiness('microscopic.'+ROUTE,case.artifact_manifest,amesp_enabled=False)['status']=='ready'
    assert registry.readiness('microscopic.'+ROUTE,case.artifact_manifest,amesp_enabled=False,
                              source_artifact_id='absent')['status']=='missing_input'
    result=execute_external(dispatch(ROUTE,source_artifact_id='absent'),case,config=config)
    assert result.status=='precondition_missing' and not result.artifact_updates
    result=execute_external(dispatch(ROUTE),case,config=config.model_copy(update={'enabled':False}))
    assert result.status=='unsupported' and not result.artifact_updates
    assert case.model_dump_json()==before


@pytest.mark.parametrize('code',['parse_failed','timeout','not_converged','process_failed'])
def test_failures_are_not_scientific_evidence(code,config,case,monkeypatch):
    config=configured(config,monkeypatch)
    def fail(*args):
        raise QCError(code,'synthetic failure')
    monkeypatch.setattr('mechcal.tools.dftb_soc.evaluate',fail)
    result=execute_external(dispatch(ROUTE),case,config=config)
    assert result.status=='failed' and not result.artifact_updates
    assert result.structured_results['failure_code']==code
    assert all(e.basis=='checklist' for e in result.evidence_units)


@pytest.mark.parametrize('code,returncode,log,allowed',[
    ('process_failed',1,'Hamilton/Overlap written, exiting program.',True),
    ('process_failed',2,'Hamilton/Overlap written, exiting program.',False),
    ('timeout',1,'Hamilton/Overlap written, exiting program.',False),
    ('process_failed',1,'Hamilton/Overlap written, exiting program.\nERROR!',False),
    ('process_failed',1,'failed',False),
])
def test_matrix_exit_classification(tmp_path,code,returncode,log,allowed):
    (tmp_path/'process.log').write_text(log)
    def fail(*args):
        raise QCError(code,'synthetic exit')
    job=SimpleNamespace(execute=fail,config=SimpleNamespace(dftb_binary='synthetic'),records=[{'returncode':returncode}])
    if allowed:
        assert export_overlap(job,tmp_path)=='intentional_export_stop_returncode_1'
    else:
        with pytest.raises(QCError):
            export_overlap(job,tmp_path)
