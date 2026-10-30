from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mechcal.capabilities.registry import CapabilityRegistry
from mechcal.schemas import CaseRun, DispatchRequest

DiscoveryEventSink = Callable[[dict[str, Any]], None]


class DescribeToolsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["describe_tools"]
    capability_ids: list[str] = Field(min_length=1, max_length=6)


class LoadedToolState(BaseModel):
    case_id: str
    loaded_tools: dict[str, str] = Field(default_factory=dict)


def amesp_enabled(case: CaseRun) -> bool:
    return case.runtime.get("tool_environment", {}).get("amesp_enabled", False) is True


def case_catalog(registry: CapabilityRegistry, case: CaseRun) -> list[dict[str, object]]:
    rows = registry.compact_catalog(case.artifact_manifest, amesp_enabled=amesp_enabled(case))
    if not case.runtime.get("planner_case_memory"):
        return rows
    directory_fields = {
        "capability_id", "description", "required_artifact_kinds", "outputs", "status",
        "missing_inputs", "reason",
    }
    return [{key: value for key, value in row.items()
             if key in directory_fields and value not in (None, "", [], {})} for row in rows]


def compact_catalog_payload(
    catalog: list[dict[str, object]], *, catalog_key: str = "capability_catalog",
) -> dict[str, Any]:
    """Return memory-mode payload fields with lossless shared readiness details.

    Pass case_catalog rows; Coverage can use catalog_key="capability_cards".
    Include the returned shared values and format notice alongside the rows.
    """
    def size(value: object) -> int:
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())

    rows = [dict(row) for row in catalog]
    shared: dict[str, object] = {}
    for field in ("reason", "missing_inputs"):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            if field in row:
                groups[json.dumps(row[field], ensure_ascii=False, sort_keys=True)].append(index)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            value = rows[indices[0]][field]
            ref = str(len(shared) + 1)
            replacement = {field + "_ref": ref}
            if len(indices) * size({field: value}) <= len(indices) * size(replacement) + size({ref: value}):
                continue
            shared[ref] = value
            for index in indices:
                rows[index].pop(field)
                rows[index].update(replacement)
    original = {catalog_key: catalog}
    if not shared:
        return original
    result = {
        catalog_key: rows,
        "capability_catalog_shared": shared,
        "capability_catalog_format": (
            "Resolve each <field>_ref as capability_catalog_shared[ref]. "
            "Status is explicit; disabled tools cannot run. describe_tools returns full details."
        ),
    }
    return result if size(result) < size(original) else original


class ToolDiscoverySession:
    """One decision's query allowance; loaded definitions belong to one case."""

    def __init__(
        self, registry: CapabilityRegistry, case: CaseRun, round_id: str,
        event_sink: DiscoveryEventSink | None = None,
    ) -> None:
        self.registry = registry
        self.case = case
        self.round_id = round_id
        self.decision_revision = case.runtime.get("question_loop", {}).get("revision")
        self.event_sink = event_sink
        budget = case.runtime.get("tool_discovery_budget", {})
        same_decision = (budget.get("case_id") == case.case_id and budget.get("round_id") == round_id
                         and budget.get("decision_revision") in (None, self.decision_revision))
        self.batches_used = (
            int(budget.get("batches_used", 0))
            if same_decision
            else 0
        )
        self.last_query: dict[str, Any] | None = None
        self.visible_ids: set[str] = set()
        raw = case.runtime.get("tool_discovery", {})
        saved = LoadedToolState.model_validate(raw) if raw else LoadedToolState(case_id=case.case_id)
        self.state = LoadedToolState(case_id=case.case_id)
        if saved.case_id == case.case_id:
            self.state.loaded_tools = {
                key: version for key, version in saved.loaded_tools.items()
                if registry.get(key) is not None and version == registry.definition_version(key)
            }
        if same_decision:
            # Legacy decisions displayed all loaded cards. New ones restore the
            # exact working set even when the description allowance is exhausted.
            self.visible_ids.update(key for key in budget.get("visible_ids", self.state.loaded_tools)
                                    if key in self.state.loaded_tools)

    def emit(self, event: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink({
                **event, "case_id": self.case.case_id, "round_id": self.round_id,
                "decision_revision": self.decision_revision,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "batches_used": self.batches_used,
                "visible_ids": sorted(self.visible_ids),
                "state": self.state.model_dump(mode="json"),
            })

    def budget_snapshot(self) -> dict[str, Any]:
        return {"case_id": self.case.case_id, "round_id": self.round_id,
                "decision_revision": self.decision_revision, "batches_used": self.batches_used,
                "visible_ids": sorted(self.visible_ids)}

    def query(self, response: dict[str, Any]) -> None:
        result: dict[str, Any] = {
            "kind": "tool_description_query", "requested_ids": response.get("capability_ids"),
            "returned_ids": [], "error": None,
        }
        try:
            if self.batches_used >= 2:
                raise ValueError("Tool description query allowance exhausted (2 batches per decision).")
            self.batches_used += 1
            request = DescribeToolsRequest.model_validate(response)
            if len(request.capability_ids) != len(set(request.capability_ids)):
                raise ValueError("Each description query must contain distinct capability IDs.")
            descriptions = self.registry.describe_tools(
                request.capability_ids, self.case.artifact_manifest,
                amesp_enabled=amesp_enabled(self.case),
            )
            self.state.loaded_tools.update({
                str(item["capability_id"]): str(item["definition_version"])
                for item in descriptions
            })
            result["returned_ids"] = list(request.capability_ids)
            self.visible_ids.update(request.capability_ids)
        except ValueError as exc:
            result["error"] = str(exc)
            raise
        finally:
            self.last_query = result
            self.emit(result)

    def augment_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        catalog = case_catalog(self.registry, self.case)
        memory_enabled = self.case.runtime.get("planner_case_memory")
        if memory_enabled:
            # Auto-load only details named by open questions, not every past tool.
            self.visible_ids.update(
                capability_id for question in payload.get("questions", [])
                if question.get("answer_state") not in {"answered", "deferred"}
                for capability_id in question.get("acceptable_capability_ids", [])
                if capability_id in self.state.loaded_tools
            )
            card_ids = [key for key in self.state.loaded_tools if key in self.visible_ids]
        else:
            card_ids = list(self.state.loaded_tools)
        result = {
            **payload,
            "tool_loading_mode": "on_demand",
            "capability_catalog": catalog,
            "capability_cards": self.registry.describe_tools(
                card_ids, self.case.artifact_manifest,
                amesp_enabled=amesp_enabled(self.case),
            ),
            "ability_evidence_guide": None,
            "artifact_available_capability_ids": [
                row["capability_id"] for row in catalog if row["status"] == "ready"
            ],
            "tool_discovery": {
                "decision_revision": self.decision_revision,
                "remaining_batches": max(0, 2 - self.batches_used),
                "max_ids_per_batch": 6,
                "loaded_ids": list(self.state.loaded_tools),
                "visible_ids": card_ids,
                "last_query": self.last_query,
            },
        }
        if memory_enabled:
            result.pop("artifact_available_capability_ids")
            result.pop("capability_catalog_shared", None)
            result.pop("capability_catalog_format", None)
            result.update(compact_catalog_payload(catalog))
        return result

    def validate_requests(self, requests: list[DispatchRequest]) -> None:
        violations = discovery_dispatch_violations(self.registry, self.case, self.state, requests)
        if violations:
            raise ValueError("Tool discovery policy: " + " ".join(violations))


def discovery_dispatch_violations(
    registry: CapabilityRegistry, case: CaseRun, state: LoadedToolState,
    requests: list[DispatchRequest],
) -> list[str]:
    violations = []
    for request in requests:
        spec = registry.get(request.capability_id)
        if (state.case_id != case.case_id or spec is None or
                state.loaded_tools.get(request.capability_id) != registry.definition_version(request.capability_id)):
            violations.append(f"Load current details for {request.capability_id} before dispatch.")
        violations.extend(registry.availability_violations(
            request, case.artifact_manifest, amesp_enabled=amesp_enabled(case),
        ))
    return violations
