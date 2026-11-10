import pytest

from mechcal.chem.validated_scopes import SCAN_SCOPE
from mechcal.tools.amesp_microscopic import (
    AmespMicroscopicError,
    microscopic_delivery_contract,
    microscopic_preflight,
)
from tests.test_tool_parameters import case_with_structure


@pytest.mark.parametrize("smiles", ["CCS", "CB(O)O", "CCF", "CCCl", "CCBr", "C[Si](C)(C)C"])
def test_neutral_extended_scan_scope(tmp_path, smiles):
    case = case_with_structure(tmp_path, smiles)
    microscopic_preflight("run_relaxed_coordinate_scan", case.artifact_manifest.artifacts[0])


@pytest.mark.parametrize("problem", ["metal", "spin", "charged_extension", "odd_electrons", "size"])
def test_unvalidated_scan_input_fails_before_calculation(tmp_path, problem):
    case = case_with_structure(tmp_path, "CCS" if problem == "charged_extension" else "CCO")
    source = case.artifact_manifest.artifacts[0]
    if problem == "spin":
        source.metadata["multiplicity"] = 3
    elif problem == "charged_extension":
        source.metadata["charge"] = 2
    elif problem == "odd_electrons":
        source.metadata["charge"] = 1
    else:
        from pathlib import Path
        path = Path(source.file_paths["xyz"])
        if problem == "metal":
            path.write_text("1\nunsupported metal\nFe 0 0 0\n")
        else:
            path.write_text("81\nsize limit\n" + "C 0 0 0\n" * 81)
    with pytest.raises(AmespMicroscopicError):
        microscopic_preflight("run_relaxed_coordinate_scan", source)


def test_scan_scope_exposed_in_delivery_contract():
    assert SCAN_SCOPE in microscopic_delivery_contract("run_relaxed_coordinate_scan")["backend_constraints"]
