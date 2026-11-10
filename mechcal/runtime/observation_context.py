"""Evidence input views; authoritative records and scientific judgments are untouched."""

from copy import deepcopy
import json

from mechcal.schemas.memory import StoredMetricFields


# Transport sizes only, not scientific relevance or evidence-strength thresholds.
INLINE_METRICS_BYTES = 2048
INLINE_VALUE_BYTES = 256
METRIC_CONTEXT_KEYS = frozenset({
    "method_provenance", "tool_args", "effective_parameters", "molecular_context",
    "modeled_systems", "limitations", "limits", "units", "unit", "conditions",
    "charge", "multiplicity", "state", "state_index", "transition_conventions",
})


def byte_size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def observation_card(observation: dict) -> dict:
    """Retain text, failure/boundary metadata and small exact values, not bulk arrays."""
    result = deepcopy(observation)
    metrics = observation.get("metrics", {})
    if (not observation.get("evidence_id") or observation.get("metrics_memory")
            or byte_size(metrics) <= INLINE_METRICS_BYTES):
        return result
    retained, omitted = {}, []
    for key, value in metrics.items():
        if key in METRIC_CONTEXT_KEYS or byte_size(value) <= INLINE_VALUE_BYTES:
            retained[key] = deepcopy(value)
        else:
            omitted.append("/metrics/" + key.replace("~", "~0").replace("/", "~1"))
    if not omitted:
        return result
    result["metrics"] = retained
    result["metrics_memory"] = StoredMetricFields(
        record_id=observation["evidence_id"], omitted_paths=omitted[:12],
        omitted_field_count=len(omitted),
    ).model_dump(mode="json")
    return result


def project_observations(observations: list[dict], *, fresh_ids=()) -> list[dict]:
    fresh = set(fresh_ids)
    return [deepcopy(row) if row.get("evidence_id") in fresh else observation_card(row)
            for row in observations]
