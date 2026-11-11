"""Offline integrity checks for the released reference snapshot, not scientific adjudication."""

import csv
import hashlib
import importlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from mechcal.mechanism_pool import MECHANISM_POOL


DATASET = Path(__file__).resolve().parents[1] / "benchmarks" / "photomechbench"
SNAPSHOT = DATASET / "raw_v2_119"
CASE_FILES = sorted((SNAPSHOT / "cases").glob("*.json"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator():
    schema = read_json(SNAPSHOT / "schema" / "case.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@pytest.mark.parametrize("path", CASE_FILES, ids=lambda path: path.stem)
def test_case_contract_and_reference_links(path, validator):
    case = read_json(path)
    validator.validate(case)
    assert case["case_id"] == path.stem
    assert case["provenance"]["annotation_method"] == "assistant_source_reconstruction"
    assert set(case["public_input"]) == {"molecule"}
    hidden = case["hidden_reference"]
    sources = {item["source_id"]: item for item in hidden["source_documents"]}
    evidence = {item["evidence_id"]: item for item in hidden["reference_evidence_units"]}
    decisions = {item["label"]: item for item in hidden["mechanism_decisions"]}
    references = {item["label"]: item for item in hidden["reference_mechanisms"]}
    assert len(sources) == len(hidden["source_documents"])
    assert len(evidence) == len(hidden["reference_evidence_units"])
    assert len(references) == len(hidden["reference_mechanisms"])
    assert len(decisions) == len(hidden["mechanism_decisions"]) == 11
    assert set(decisions) == set(MECHANISM_POOL)
    for item in evidence.values():
        assert item["source_id"] in sources
        assert 1 <= item["pdf_page"] <= sources[item["source_id"]]["page_count"]
    supported = set()
    for label, item in decisions.items():
        ids = item["evidence_ids"]
        assert len(ids) == len(set(ids))
        assert set(ids) <= evidence.keys()
        if item["status"] != "supported":
            assert item["role"] is None
            continue
        supported.add(label)
        assert item["role"] in {"primary", "co_primary", "secondary"}
        assert any(
            evidence[eid]["scope"] in {"target", "series_including_target"}
            and evidence[eid]["kind"] != "background"
            for eid in ids
        )
        assert references[label] == {
            "label": label,
            "role": item["role"],
            "gain": 1 if item["role"] == "secondary" else 2,
            "evidence_ids": ids,
            "rationale": item["rationale"],
        }
    assert set(references) == supported


def test_snapshot_membership_hashes_and_statistics():
    manifest = read_json(SNAPSHOT / "manifest.json")
    cases = [read_json(path) for path in CASE_FILES]
    assert len(cases) == manifest["case_count"] == 119
    assert [case["case_id"] for case in cases] == manifest["case_ids"]
    assert set(manifest["case_files_sha256"]) == {
        str(path.relative_to(SNAPSHOT)) for path in CASE_FILES
    }
    for path in CASE_FILES:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            manifest["case_files_sha256"][str(path.relative_to(SNAPSHOT))]
        )
    excluded = read_json(SNAPSHOT / "excluded_cases.json")
    assert [item["case_id"] for item in excluded] == manifest["excluded_case_ids"]
    assert manifest["excluded_case_ids"] == ["AIE_DDX_MECH_V4_0127"]
    assert not set(manifest["case_ids"]) & set(manifest["excluded_case_ids"])
    assert read_json(SNAPSHOT / "mechanism_pool.json") == {
        "mechanism_pool": list(MECHANISM_POOL)
    }
    assert manifest["schema_version"] == "photomechbench.raw.v2"
    assert manifest["version"] == read_json(DATASET / "manifest.json")["version"]
    hidden = [case["hidden_reference"] for case in cases]
    refs = [item for case in hidden for item in case["reference_mechanisms"]]
    assert len(refs) == manifest["reference_count"] == 401
    assert sum(len(case["reference_evidence_units"]) for case in hidden) == (
        manifest["evidence_count"]
    ) == 998
    assert len({item["doi"] for case in hidden for item in case["source_documents"]}) == (
        manifest["source_dois"]
    ) == 76
    assert dict(Counter(item["label"] for item in refs)) == manifest["label_counts"]
    assert dict(Counter(item["role"] for item in refs)) == manifest["role_counts"]


def test_csv_agrees_with_references_and_stays_input_only():
    with (DATASET / "raw_v2_119.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["case_id", "smiles"]
        rows = list(reader)
    assert rows == [
        {"case_id": case["case_id"],
         "smiles": case["public_input"]["molecule"]["structure"]["value"]}
        for case in map(read_json, CASE_FILES)
    ]


def test_complete_original_snapshot_checksums():
    inventory = read_json(DATASET / "raw_v2_119_checksums.json")
    files = {str(path.relative_to(SNAPSHOT)): path
             for path in SNAPSHOT.rglob("*") if path.is_file()}
    assert set(files) == set(inventory["files_sha256"])
    assert len(files) == inventory["file_count"] == 249
    assert sum(path.stat().st_size for path in files.values()) == (
        inventory["total_bytes"]
    ) == 2683619
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == inventory["files_sha256"][name]


def test_original_exports_agree_with_complete_cases():
    cases = [read_json(path) for path in CASE_FILES]
    for name, field in [("public_inputs.jsonl", "public_input"),
                        ("references.jsonl", "hidden_reference")]:
        rows = [json.loads(line) for line in (SNAPSHOT / name).read_text(
            encoding="utf-8").splitlines()]
        assert rows == [{"case_id": case["case_id"], field: case[field]} for case in cases]
    # The original CSV uses CRLF; the earlier input-only export uses LF.
    with (SNAPSHOT / "inputs.csv").open(encoding="utf-8", newline="") as original:
        with (DATASET / "raw_v2_119.csv").open(encoding="utf-8", newline="") as public:
            assert list(csv.DictReader(original)) == list(csv.DictReader(public))
    validation = read_json(SNAPSHOT / "validation.json")
    manifest = read_json(SNAPSHOT / "manifest.json")
    assert validation["valid"] is True and validation["errors"] == []
    for key in ["case_count", "unique_structures", "source_dois", "evidence_count",
                "reference_count", "role_counts", "label_counts", "case_ids"]:
        assert validation[key] == manifest[key]


def test_audit_and_distribution_scope():
    audits = sorted((SNAPSHOT / "audit").glob("*.json"))
    assert [path.stem for path in audits] == [path.stem for path in CASE_FILES]
    for path in audits:
        assert read_json(path)["case_id"] == path.stem
    for path in DATASET.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            assert path in {SNAPSHOT / "public_inputs.jsonl", SNAPSHOT / "references.jsonl"}
        else:
            assert path.suffix in {".md", ".csv", ".json"}, path
        assert not re.search(
            r"/home/|/tmp/|sk-[a-zA-Z0-9]{16,}|hf_[a-zA-Z0-9]{20,}|"
            r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
            path.read_text(encoding="utf-8"),
        ), path
    license_text = (DATASET / "LICENSE.md").read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: CC-BY-4.0" in license_text
    assert "does not relicense" in " ".join(license_text.split())


@pytest.mark.parametrize("module_name", [
    "evaluate", "evaluate_direct", "evaluate_tools", "evaluate_chemgraph",
    "evaluate_prior", "rescore",
])
def test_existing_evaluation_loaders(module_name):
    if module_name == "evaluate_chemgraph":
        pytest.importorskip("chemgraph", reason="Optional ChemGraph baseline is not installed")
    module = importlib.import_module(f"mechcal.cli.{module_name}")
    loaded = module._load_cases(SNAPSHOT / "cases")
    expected = [read_json(path) for path in CASE_FILES]
    assert len(loaded) == len(expected) == 119
    for actual, case in zip(loaded, expected, strict=True):
        assert actual["case_id"] == case["case_id"]
        assert actual["reference_mechanisms"] == case["hidden_reference"]["reference_mechanisms"]
        if "smiles" in actual:
            assert actual["smiles"] == case["public_input"]["molecule"]["structure"]["value"]
