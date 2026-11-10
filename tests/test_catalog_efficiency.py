"""Lossless catalog factoring changes context, not capability permissions."""

from copy import deepcopy
import json

import pytest

from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.capabilities.discovery import ToolDiscoverySession, case_catalog, compact_catalog_payload
from test_tool_discovery import BASELINE, MACRO, RELAX, SCAN, case, dispatch, query


def serialized_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def expanded_catalog(payload, catalog_key="capability_catalog"):
    shared = payload.get("capability_catalog_shared", {})
    return [{
        key.removesuffix("_ref") if key.endswith("_ref") else key:
        shared[value] if key.endswith("_ref") else value
        for key, value in row.items()
    } for row in payload[catalog_key]]


def memory_case(**kwargs):
    current = case(**kwargs)
    current.runtime["planner_case_memory"] = True
    return current


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_catalog_factors_losslessly_without_mutating_case_or_input(enabled, prepared):
    registry = default_capability_registry()
    current = memory_case(enabled=enabled, prepared=prepared)
    original_case = current.model_dump(mode="json")
    catalog = case_catalog(registry, current)
    source = {"capability_catalog": catalog, "artifact_available_capability_ids": ["stale"]}
    original_source = deepcopy(source)
    lookup = ToolDiscoverySession(registry, current, "R001")
    result = lookup.augment_payload(source)
    assert expanded_catalog(result) == catalog
    for before, after in zip(catalog, result["capability_catalog"], strict=True):
        for key in ("capability_id", "description", "required_artifact_kinds", "outputs", "status"):
            assert after.get(key) == before.get(key)
    assert "artifact_available_capability_ids" not in result
    assert result == lookup.augment_payload(source)
    assert source == original_source
    assert current.model_dump(mode="json") == original_case


def test_shared_values_keep_exact_restrictions_paths_and_loaded_cards():
    original = default_capability_registry()
    reason = "Disabled: required parameter file /local/qc/inputs/exact parameters.json is missing. " * 4
    registry = CapabilityRegistry([
        spec.model_copy(update={"disabled_reason": reason})
        for spec in original.snapshot().capabilities
    ])
    current = memory_case(prepared=True)
    lookup = ToolDiscoverySession(registry, current, "R001")
    lookup.query(query(MACRO, BASELINE))
    result = lookup.augment_payload({})
    assert reason in result["capability_catalog_shared"].values()
    assert all(row["status"] == "disabled" for row in result["capability_catalog"])
    assert expanded_catalog(result) == case_catalog(registry, current)
    assert result["capability_cards"] == registry.describe_tools(
        [MACRO, BASELINE], current.artifact_manifest, amesp_enabled=True,
    )
    with pytest.raises(ValueError, match="disabled"):
        lookup.validate_requests([dispatch(MACRO)])


def test_disabled_backend_catalog_has_measurable_total_payload_reduction():
    registry = default_capability_registry()
    current = memory_case(enabled=False)
    result = ToolDiscoverySession(registry, current, "R001").augment_payload({})
    before = {key: value for key, value in result.items()
              if key not in {"capability_catalog_shared", "capability_catalog_format"}}
    before["capability_catalog"] = [
        {key: value for key, value in row.items()
         if key != "molecular_input" and value not in (None, "", [], {})}
        for row in registry.compact_catalog(current.artifact_manifest, amesp_enabled=False)
    ]
    before["artifact_available_capability_ids"] = [
        row["capability_id"] for row in before["capability_catalog"] if row["status"] == "ready"
    ]
    assert serialized_size(result) < serialized_size(before) * 0.9


@pytest.mark.parametrize("enabled", [False, True])
def test_short_directory_omits_only_detail_metadata_and_preserves_readiness(enabled):
    registry = default_capability_registry()
    current = memory_case(prepared=True, enabled=enabled)
    registry_before = registry.snapshot().model_dump(mode="json")
    original = registry.compact_catalog(current.artifact_manifest, amesp_enabled=enabled)
    short = case_catalog(registry, current)
    excluded = {"molecular_input", "execution_dependencies", "unimplemented_structured_descriptors"}
    assert short == [{key: value for key, value in row.items()
                      if key not in excluded and value not in (None, "", [], {})} for row in original]
    assert all(row.keys() <= {
        "capability_id", "description", "outputs", "status", "required_artifact_kinds",
        "missing_inputs", "reason", "source_artifact_kinds", "cost",
    } for row in short)
    assert [(r["capability_id"], r.get("cost")) for r in short] == [
        (r["capability_id"], r.get("cost")) for r in original]
    assert serialized_size(short) < serialized_size(original)

    metadata_ids = [row["capability_id"] for row in original
                    if row["execution_dependencies"] or row["unimplemented_structured_descriptors"]]
    assert metadata_ids
    cards_before = registry.describe_tools(metadata_ids, current.artifact_manifest, amesp_enabled=enabled)
    lookup = ToolDiscoverySession(registry, current, "R001")
    lookup.augment_payload({})
    for index in range(0, len(metadata_ids), 6):
        # Each decision retains the existing two-batch limit.
        lookup = ToolDiscoverySession(registry, current, f"R{index + 2:03d}")
        ids = metadata_ids[index:index + 6]
        lookup.query(query(*ids))
        loaded = lookup.augment_payload({})["capability_cards"]
        assert loaded == cards_before[index:index + 6]
        for card in loaded:
            spec = registry.require(card["capability_id"])
            assert card["execution_dependencies"] == spec.metadata["execution_dependencies"]
            assert card["delivery_contract"] == spec.metadata["delivery_contract"]
            readiness = registry.readiness(
                spec.capability_id, current.artifact_manifest, amesp_enabled=enabled,
            )
            catalog_row = next(row for row in short if row["capability_id"] == spec.capability_id)
            assert catalog_row["status"] == readiness["status"]
            assert catalog_row.get("missing_inputs", []) == readiness["missing_inputs"]
            assert catalog_row.get("reason") == readiness["reason"] or not readiness["reason"]
    assert registry.snapshot().model_dump(mode="json") == registry_before


@pytest.mark.parametrize("catalog_key", ["capability_catalog", "capability_cards"])
def test_public_compactor_supports_planner_and_coverage_without_mutation(catalog_key):
    registry = default_capability_registry()
    current = memory_case(enabled=False)
    catalog = case_catalog(registry, current)
    original = deepcopy(catalog)
    result = compact_catalog_payload(catalog, catalog_key=catalog_key)
    assert expanded_catalog(result, catalog_key) == original
    assert catalog == original
    assert result == compact_catalog_payload(catalog, catalog_key=catalog_key)
    assert serialized_size(result) < serialized_size({catalog_key: catalog})
    assert "capability_catalog_shared" in result
    if catalog_key != "capability_catalog":
        assert "capability_catalog" not in result


@pytest.mark.parametrize("enabled", [False, True])
def test_memory_disabled_payload_is_unchanged(enabled):
    registry = default_capability_registry()
    current = case(enabled=enabled)
    lookup = ToolDiscoverySession(registry, current, "R001")
    lookup.query(query(MACRO, BASELINE))
    catalog = registry.compact_catalog(current.artifact_manifest, amesp_enabled=enabled)
    assert case_catalog(registry, current) == catalog
    assert lookup.augment_payload({"fixture": True}) == {
        "fixture": True,
        "tool_loading_mode": "on_demand",
        "capability_catalog": catalog,
        "capability_cards": registry.describe_tools(
            [MACRO, BASELINE], current.artifact_manifest, amesp_enabled=enabled,
        ),
        "ability_evidence_guide": None,
        "artifact_available_capability_ids": [
            row["capability_id"] for row in catalog if row["status"] == "ready"
        ],
        "tool_discovery": {
            "decision_revision": None, "remaining_batches": 1, "max_ids_per_batch": 6,
            "loaded_ids": [MACRO, BASELINE], "visible_ids": [MACRO, BASELINE],
            "last_query": lookup.last_query,
        },
    }


def test_small_catalog_does_not_grow_or_retain_stale_shared_values():
    registry = CapabilityRegistry([default_capability_registry().require(MACRO)])
    current = memory_case()
    lookup = ToolDiscoverySession(registry, current, "R001")
    result = lookup.augment_payload({
        "capability_catalog_shared": {"1": "stale"}, "capability_catalog_format": "stale",
    })
    assert result["capability_catalog"] == case_catalog(registry, current)
    assert "capability_catalog_shared" not in result
    assert "capability_catalog_format" not in result


def test_factoring_does_not_bypass_disabled_arguments_or_registry_ablation():
    registry = default_capability_registry()
    current = memory_case(prepared=True)
    lookup = ToolDiscoverySession(registry, current, "R001")
    lookup.query(query(RELAX, SCAN))
    lookup.augment_payload({})
    for request in (dispatch(RELAX), dispatch(SCAN, state_root=1)):
        with pytest.raises(ValueError, match="disabled"):
            lookup.validate_requests([request])
    restricted = registry.for_owner_agents(("macro",))
    lookup = ToolDiscoverySession(restricted, current, "R001")
    assert {row["capability_id"] for row in lookup.augment_payload({})["capability_catalog"]} == {
        spec.capability_id for spec in restricted.snapshot().capabilities
    }
    with pytest.raises(ValueError, match="disallowed"):
        lookup.query(query(BASELINE))


def test_backend_disabled_loaded_tool_is_still_rejected():
    registry = default_capability_registry().with_amesp_backend(False)
    current = memory_case(prepared=True)
    lookup = ToolDiscoverySession(registry, current, "R001")
    lookup.query(query(BASELINE))
    result = lookup.augment_payload({})
    assert next(row for row in result["capability_catalog"]
                if row["capability_id"] == BASELINE)["status"] == "disabled"
    with pytest.raises(ValueError, match="disabled"):
        lookup.validate_requests([dispatch(BASELINE)])
