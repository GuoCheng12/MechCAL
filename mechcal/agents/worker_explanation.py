"""Read-only, source-linked interpretation inside the existing Worker report call."""

from typing import Any
from pydantic import ValidationError

from mechcal.agents.observation_view import observation_view
from mechcal.capabilities.schemas import CapabilitySpec
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.question_changes import public_questions
from mechcal.runtime.field_repair import FieldRepairProducer
from mechcal.runtime.retry import ModelContractError, RequestValidationError, RejectedResponseError, complete_recorded_json
from mechcal.schemas import CaseRun, DispatchRequest, ToolExecutionResult
from mechcal.schemas.worker_explanation import WorkerExplanation


def explain_result(
    client: OpenAIJsonClient, request: DispatchRequest, case: CaseRun,
    tool_result: ToolExecutionResult, capability: CapabilitySpec, feedback: str | None,
    *, event_sink=None,
) -> WorkerExplanation:
    from mechcal.runtime.observation_delivery import delivery_receipt
    source = observation_view({
        "structured_results": tool_result.structured_results,
        "evidence_units": [e.model_dump(mode="json") for e in tool_result.evidence_units],
    })
    questions = public_questions(case)
    payload = {
        "stage": "worker_result_explanation",
        "agent_name": request.agent_name,
        "smiles": case.input.smiles,
        "mechanism_pool": list(MECHANISM_POOL),
        "dispatch_request": request.model_dump(mode="json"),
        "questions": [questions[q] for q in request.question_ids if q in questions],
        "capability": capability.model_dump(mode="json"),
        "status": tool_result.status,
        "observation_receipt": delivery_receipt(request, tool_result.structured_results),
        "failure": tool_result.failure.model_dump(mode="json"),
        "tool_execution_result": source,
        "source_catalog": source_catalog(source),
        "output_contract": WorkerExplanation.model_json_schema(),
    }
    from mechcal.runtime.scoped_contracts import worker_schema
    payload["output_contract"] = worker_schema(payload["output_contract"], request, tool_result)
    def full(feedback):
        return complete_recorded_json(client, event_sink=event_sink,
            system_prompt=load_prompt("worker_explanation.md"),
            compact_context=case.runtime.get("scoped_evidence_updates", False),
            payload=payload, schema_feedback=feedback)

    def repair(correction, feedback):
        return complete_recorded_json(client, event_sink=event_sink,
            system_prompt=load_prompt("worker_explanation_repair.md"),
            compact_context=case.runtime.get("scoped_evidence_updates", False),
            payload={**correction, "stage": "worker_explanation_repair",
                     "tool_execution_result": source, "source_catalog": payload["source_catalog"],
                     "observation_receipt": payload["observation_receipt"],
                     "question_ids": request.question_ids,
                     "mechanism_pool": list(MECHANISM_POOL),
                     "usable_evidence_refs": [e.evidence_id for e in tool_result.evidence_units
                                              if e.status in {"present", "partial"}]},
            schema_feedback=None)

    producer = FieldRepairProducer(full, repair, response_model=WorkerExplanation, event_sink=event_sink)
    # Only the active question loop supplies structured retry context.
    data = producer(feedback) if case.runtime.get("question_loop") else full(feedback)
    try:
        explanation = WorkerExplanation.model_validate(data)
        validate_explanation(explanation, request, tool_result, source)
    except (ValidationError, ModelContractError) as exc:
        rejected = RejectedResponseError(str(exc), data)
        rejected.issues = (exc.errors(include_url=False, include_context=False, include_input=False)
                           if isinstance(exc, ValidationError) else getattr(exc, "issues", []))
        raise rejected from exc
    return explanation


def validate_explanation(explanation, request, tool_result, source):
    issues = []

    def issue(message, path):
        issues.append({"message": message, "repair_paths": [path]})

    if set(explanation.question_ids) != set(request.question_ids):
        issue("Worker explanation must preserve the request's question_ids.", ["question_ids"])
    if tool_result.status not in {"success", "partial"} and explanation.findings:
        raise ModelContractError("Failed or unavailable computation cannot yield scientific findings.")
    evidence = {e.evidence_id: e for e in tool_result.evidence_units}
    for index, finding in enumerate(explanation.findings):
        labels = [item.label for item in finding.mechanism_implications]
        if len(labels) != len(set(labels)) or not set(labels) <= set(MECHANISM_POOL):
            issue("Use distinct mechanism-pool labels for this finding's implications.",
                  ["findings", index, "mechanism_implications"])
        for ref in finding.evidence_refs:
            if ref not in evidence or evidence[ref].status not in {"present", "partial"}:
                issue("Worker finding must cite usable evidence from this execution.",
                      ["findings", index, "evidence_refs"])
        for path in finding.source_paths:
            try:
                resolve_source_path(source, path)
            except ModelContractError as exc:
                issue(str(exc), ["findings", index, "source_paths"])
    if issues:
        raise RequestValidationError(issues)


def source_catalog(source):
    """Offer exact, coarse-grained pointers without duplicating observations."""
    entries = []

    def visit(value, path, depth):
        if value is None or value == {} or value == []:
            return
        if depth >= 2:
            entries.append({"path": path, "type": type(value).__name__})
        if depth >= 3:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + "/" + key.replace("~", "~0").replace("/", "~1"), depth + 1)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + f"/{index}", depth + 1)

    visit(source, "", 0)
    return entries


def resolve_source_path(source: dict[str, Any], path: str) -> Any:
    parts = path.split("/")
    if len(parts) < 3 or parts[0] or parts[1] not in {"structured_results", "evidence_units"}:
        raise ModelContractError("Source paths must point inside structured_results or evidence_units.")
    value = source
    try:
        for part in parts[1:]:
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not key.isdecimal() or (len(key) > 1 and key.startswith("0")):
                    raise ValueError("Invalid array index")
                value = value[int(key)]
            else:
                value = value[key]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ModelContractError(f"Worker explanation source path does not resolve: {path}") from exc
    if value is None or value == {} or value == []:
        raise ModelContractError(f"Worker explanation source path contains no observation: {path}")
    return value
