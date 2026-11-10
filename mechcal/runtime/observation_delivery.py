"""Verify requested data delivery without interpreting physical mechanisms."""

import math
import re

from pydantic import ValidationError

from mechcal.schemas.observation_delivery import ObservableRequirement


def pointer(value, path):
    for token in path.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list) and token.isdecimal():
            index = int(token)
            value = value[index] if index < len(value) else None
        elif isinstance(value, dict):
            value = value.get(token)
        else:
            return None
    return value


def requested_values(request):
    requirements = list(request.required_observables)
    if request.route == "extract_geometry_descriptors_from_bundle":
        from mechcal.schemas.tool_arguments import GeometryDescriptorArgs
        args = GeometryDescriptorArgs.model_validate(request.tool_args)
        for selected in args.measurements:
            measurement = selected.model_dump(mode="json")
            prefix = "/parsed_observables/geometry_measurements/" + measurement["measurement_id"]
            requirements.append(ObservableRequirement(
                path=prefix + "/value", value_type="number", unit_path=prefix + "/unit",
                unit="angstrom" if measurement["kind"] == "distance" else "degree",
            ))
    return list({(r.path, r.value_type, r.unit_path, r.unit): r for r in requirements}.values())


def delivery_receipt(request, structured_results):
    try:
        requirements = requested_values(request)
    except ValidationError as exc:
        return {"status": "invalid_request", "checks": [],
                "errors": exc.errors(include_url=False, include_context=False, include_input=False),
                "scope": "Invalid measurement parameters. No fulfillment or scientific judgment inferred."}
    checks = []
    for requirement in requirements:
        value = pointer(structured_results, requirement.path)
        valid = {
            "number": isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
            "string": isinstance(value, str) and bool(value.strip()),
            "array": isinstance(value, list) and bool(value),
            "object": isinstance(value, dict) and bool(value),
        }[requirement.value_type]
        unit_matches = requirement.unit_path is None or pointer(structured_results, requirement.unit_path) == requirement.unit
        checks.append({**requirement.model_dump(mode="json"), "delivered": valid and unit_matches,
                       "reason": "value and unit verified" if valid and unit_matches else
                       "value absent, empty, invalid or unit mismatch"})
    return {"status": "not_specified" if not checks else "delivered" if all(c["delivered"] for c in checks) else "unmet",
            "checks": checks, "scope": "Data presence/type/unit only. Execution success is separate; scientific resolution remains Planner judgment."}


def requirement_violations(spec, request):
    fields = spec.metadata.get("delivery_contract", {}).get("value_fields", [])
    if not fields:
        return []
    patterns = [re.escape(path).replace(re.escape("{measurement_id}"), r"[A-Za-z][A-Za-z0-9_]{0,47}") for path in fields]
    violations = [f"Requested value {r.path} is not a documented output of {spec.capability_id}. "
            f"Available value paths: {fields}. Select parameters or a different route explicitly; none will be substituted."
            for r in request.required_observables if not any(re.fullmatch(pattern, r.path) for pattern in patterns)]
    if request.route == "extract_geometry_descriptors_from_bundle":
        from mechcal.schemas.tool_arguments import GeometryDescriptorArgs
        selected = {m.measurement_id for m in GeometryDescriptorArgs.model_validate(request.tool_args).measurements}
        for requirement in request.required_observables:
            parts = requirement.path.split("/")
            if parts[1:3] == ["parsed_observables", "geometry_measurements"] and len(parts) >= 4 and parts[3] not in selected:
                violations.append(f"Measurement {parts[3]} needs explicit measurements parameters with source XYZ indices.")
    return violations


def validate_question_delivery(case, update):
    """Do not close a data prerequisite using only executions missing that data."""
    from mechcal.runtime.retry import RequestValidationError

    results = {r["result_id"]: r for r in case.runtime.get("question_loop", {}).get("results", [])}
    issues = []
    for index, question in enumerate(update.questions):
        linked = [results[rid] for rid in question.result_ids if rid in results]
        failed = [r for r in linked if r.get("observation_receipt", {}).get("status") == "unmet"
                  and (r.get("investigation_plan") or {}).get("purpose") == "preparation"]
        if question.answer_state != "answered" or not linked or len(failed) != len(linked):
            continue
        # Other observations or structural input may independently answer a question.
        missing_result_refs = {ref for r in failed for ref in r.get("evidence_refs", [])}
        if set(question.evidence_refs) - missing_result_refs:
            continue
        issues.append({"path": ["questions", index, "answer_state"], "code": "requested_observation_unmet",
                       "record_selector": {"question_id": question.question_id},
                       "message": "All cited executions failed the requested data contract. Keep this question partial/open/deferred, "
                       "or cite an independent observation that actually answers it. No mechanism is promoted or rejected."})
    if issues:
        raise RequestValidationError(issues)
