from pathlib import Path

import pytest

from mechcal.schemas.tool_arguments import RelaxationArgs, validate_tool_args
from mechcal.tools.amesp_baseline import AmespBaselineError, AmespStepResult
from mechcal.tools.amesp_relaxation import (
    converged_geometry,
    endpoint_states,
    optimization_history,
    run_relaxation,
)


# Minimal parser fixtures, not simulated scientific evidence.
STATES = """State 1 : E = 5.1297 eV
 E(TD) = -6.456280999 <S**2>= 0.000 f= 0.0000
State 2 : E = 9.0660 eV
 E(TD) = -6.311624395 <S**2>= 0.000 f= 0.5085
"""
GEOMETRY = """Geometry Optimization Converged after 7 Steps.
Final Geometry(angstroms):
2
C 0.0 0.0 0.0
O 1.3 0.0 0.0
Final Energy: -6.4715264219
"""


@pytest.mark.parametrize(
    "args",
    [
        {"state_root": 0},
        {"state_root": 4, "s1_nstates": 3},
        {"state_root": True},
        {"max_opt_cycles": 201},
        {"max_opt_cycles": 0},
        {"solvent": "water"},
        {"state_spin": "triplet"},
        {"method": "arbitrary"},
        {"state_root": None},
    ],
)
def test_relaxation_arguments_reject_unsupported_requests(args):
    with pytest.raises(ValueError):
        validate_tool_args("run_excited_state_relaxation", args)


def test_defaults_and_selected_root_are_explicit():
    args = validate_tool_args("run_excited_state_relaxation", {"state_root": 2})
    assert args == {"state_root": 2, "s1_nstates": 3, "td_tout": 3, "max_opt_cycles": 60}


def test_public_guide_distinguishes_relaxation_from_ci_search():
    from mechcal.capabilities import default_capability_registry
    from mechcal.capabilities.evidence_guide import compact_ability_evidence_guide

    rows = {
        row["capability_id"]: row
        for row in compact_ability_evidence_guide(default_capability_registry())
    }
    row = rows["microscopic.run_excited_state_relaxation"]
    assert row["evidence_tier"] == "boundary_only"
    assert "Suspended" in row["boundary"]
    assert "no character tracking" in row["boundary"]
    assert (
        "run_excited_state_relaxation"
        in rows["microscopic.unsupported_excited_state_relaxation"]["boundary"]
    )


def test_convergence_requires_marker_and_matching_geometry():
    coords, count = converged_geometry(GEOMETRY, ["C", "O"])
    assert coords[1][0] == 1.3 and count == 7
    for text, symbols in [
        (GEOMETRY.split("\n", 1)[1], ["C", "O"]),
        (GEOMETRY, ["O", "C"]),
        (GEOMETRY.replace("1.3", "nan"), ["C", "O"]),
        (GEOMETRY.replace("1.3", "broken"), ["C", "O"]),
    ]:
        with pytest.raises(AmespBaselineError):
            converged_geometry(text, symbols)


def test_endpoint_parser_does_not_merge_optimization_iterations():
    assert len(endpoint_states(STATES, 2)) == 2
    with pytest.raises(AmespBaselineError, match="one complete"):
        endpoint_states(STATES + STATES, 2)
    with pytest.raises(AmespBaselineError):
        endpoint_states(STATES.replace("f= 0.5085", ""), 2)
    with pytest.raises(AmespBaselineError, match="singlet"):
        endpoint_states(STATES.replace("<S**2>= 0.000", "<S**2>= 2.000"), 2)


def test_history_keeps_state_manifolds_separate():
    history = optimization_history("Geom Opt Step: 1\n" + STATES + "\nGeom Opt Step: 2\n" + STATES)
    assert [h["step"] for h in history] == [1, 2]
    assert [len(h["excited_states"]) for h in history] == [2, 2]


class FixtureRunner:
    def __init__(self, tmp_path, fail_stage=None):
        self.amesp_bin = tmp_path / "fake-binary"
        self.amesp_bin.write_bytes(b"test fixture only")
        self.fail_stage = fail_stage
        self.calls = []

    def _run_step(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["step_id"]
        workdir = kwargs["workdir"]
        aip, aop = workdir / (name + ".aip"), workdir / (name + ".aop")
        aip.write_text("test fixture only")
        if name == self.fail_stage:
            aop.write_text("Geom Opt Step: 1\n" + STATES)
            raise AmespBaselineError(
                "runtime_failed", "Fixture failure", file_paths={"aip": str(aip), "aop": str(aop)}
            )
        aop.write_text(GEOMETRY if name == "excited_optimization" else STATES)
        return AmespStepResult(name, aip, aop, workdir / "stdout", workdir / "stderr", 0, True, 0.0)


@pytest.mark.parametrize("failure", ["excited_optimization", "final_vertical"])
def test_partial_results_do_not_fabricate_a_final_comparison(tmp_path, failure):
    runner = FixtureRunner(tmp_path, failure)
    result = run_relaxation(
        runner,
        workdir=tmp_path,
        symbols=["C", "O"],
        coordinates=[[0, 0, 0], [1.2, 0, 0]],
        charge=0,
        multiplicity=1,
        args=RelaxationArgs(s1_nstates=2),
    )
    assert result.observables["comparison"] is None
    assert result.final_spectrum_text == ""
    assert "initial_state_manifold" in result.observables["available_descriptors"]
    assert result.missing
    assert result.observables["optimization_converged"] == (failure == "final_vertical")
    assert bool(result.file_paths.get("analysis_geometry_xyz")) == (failure == "final_vertical")


def test_initial_failure_has_no_computed_evidence(tmp_path):
    runner = FixtureRunner(tmp_path, "initial_vertical")
    with pytest.raises(AmespBaselineError) as error:
        run_relaxation(
            runner,
            workdir=tmp_path,
            symbols=["C", "O"],
            coordinates=[[0, 0, 0], [1.2, 0, 0]],
            charge=0,
            multiplicity=1,
            args=RelaxationArgs(s1_nstates=2),
        )
    assert Path(error.value.file_paths["initial_vertical_aop"]).is_file()
    assert len(runner.calls) == 1


def test_target_parameters_reach_every_step(tmp_path):
    runner = FixtureRunner(tmp_path)
    result = run_relaxation(
        runner,
        workdir=tmp_path,
        symbols=["C", "O"],
        coordinates=[[0, 0, 0], [1.2, 0, 0]],
        charge=2,
        multiplicity=1,
        args=RelaxationArgs(s1_nstates=2, state_root=2, max_opt_cycles=15),
    )
    assert not result.missing and result.observables["optimization_converged"]
    assert result.observables["state_tracking"] == "fixed_ordinal_root_not_character_tracked"
    for call in runner.calls:
        assert call["charge"] == 2 and call["multiplicity"] == 1
        assert "root 2" in dict(call["block_lines"])["posthf"]
    assert "maxcyc 15" in dict(runner.calls[1]["block_lines"])["opt"]


def test_worker_preserves_relaxation_args(tmp_path):
    from mechcal.agents.execution_planner import WorkerExecutionPlanner
    from mechcal.capabilities import default_capability_registry
    from tests.test_tool_parameters import case_with_structure, request

    case = case_with_structure(tmp_path, "C=O")
    req = request("run_excited_state_relaxation", {"state_root": 2, "max_opt_cycles": 20})
    plan = WorkerExecutionPlanner(
        capability_registry=default_capability_registry()
    ).deterministic_plan(agent_name="microscopic", request=req, case_run=case)
    assert plan.tool_args["state_root"] == 2
    assert plan.tool_args["max_opt_cycles"] == 20


def test_relaxation_unsupported_elements_fail_before_backend(tmp_path):
    from tests.test_tool_parameters import case_with_structure, request
    from mechcal.tools.local_micro import LocalMicroscopicTool

    case = case_with_structure(tmp_path, "CCl")
    result = LocalMicroscopicTool(enable_amesp=True).run(
        request("run_excited_state_relaxation", {}), case
    )
    assert result.status == "unsupported"
