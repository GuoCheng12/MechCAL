"""The development/test split is reproducible, keeps papers together, and test cases are locked."""

import importlib.util
import json
from pathlib import Path

import pytest

from mechcal.eval.splits import SPLIT_PATH, load_split, split_of

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location("make_split", ROOT / "examples/make_dev_test_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_split_is_reproducible_and_paper_disjoint():
    module = builder()
    rebuilt = module.build()
    committed = json.loads(SPLIT_PATH.read_text())
    assert rebuilt == committed, "the split file differs from what examples/make_dev_test_split.py builds"
    dev = {d["case_id"] for d in committed["dev"]}
    assert {module.P + k for k in module.USED} <= dev          # everything used in development stays dev
    assert len(dev) + len(committed["test"]) == 119 and not dev & set(committed["test"])


def test_every_family_has_two_primary_dev_cases():
    primary = load_split()["coverage"]["dev"]["primary"]
    assert len(primary) == 11 and min(primary.values()) >= 2


def test_held_out_cases_are_refused_without_explicit_permission(tmp_path, monkeypatch):
    from mechcal.agent import batch, cli
    test_case = load_split()["test"][0]
    assert split_of(test_case) == "test" and split_of("AIE_DDX_MECH_V4_0003") == "dev" and split_of("custom") is None
    with pytest.raises(SystemExit, match="held-out test case"):
        cli.main(["--case-id", test_case, "--out", str(tmp_path / "x")])
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test")
    with pytest.raises(SystemExit, match="held-out test case"):
        batch.main(["--cases", test_case[-4:], "--out", str(tmp_path / "b")])
