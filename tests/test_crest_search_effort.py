import pytest
from ase import Atoms

from mechcal.schemas.external_qc import ConformerSamplingArgs
from mechcal.tools.crest_backend import sample
from mechcal.tools.qc_process import QCError


@pytest.mark.parametrize('length', [.49, 101., float('nan'), float('inf')])
def test_invalid_search_length(length):
    with pytest.raises(ValueError):
        ConformerSamplingArgs(source_artifact_id='a', md_length_ps=length)


@pytest.mark.parametrize('length', [None, 1., 5.])
def test_explicit_length_is_forwarded_without_changing_method(tmp_path, length):
    from types import SimpleNamespace
    commands = []
    def execute(command, directory):
        commands.append(command)
        raise QCError('timeout', 'synthetic stop before an ensemble exists')
    job = SimpleNamespace(workdir=tmp_path, config=SimpleNamespace(crest_binary='crest',
        xtb_binary='xtb', threads=4), execute=execute)
    args = ConformerSamplingArgs(source_artifact_id='a', md_length_ps=length, max_members=2)
    with pytest.raises(QCError, match='synthetic stop'):
        sample(job, Atoms('H2', positions=[[0, 0, 0], [0, 0, .74]]), args, None)
    command = commands[0]
    assert '--gfn2' in command and '--squick' in command
    if length is None:
        assert '--mdlen' not in command
    else:
        assert command[command.index('--mdlen')+1] == str(length)
