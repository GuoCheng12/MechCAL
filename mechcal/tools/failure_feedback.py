"""Operational recovery advice, separate from scientific observations."""

from mechcal.schemas.failures import FailureReport


_POLICIES = {
    "not_converged": (True, "Check which optimization or electronic step did not converge. "
                      "Revise its allowed iteration limit or initial geometry before a new request."),
    "numerical_validation_failed": (True, "Inspect the failed numerical checks and thresholds. "
                                    "A finer spatial grid may help if available; otherwise inspect the source calculation. "
                                    "Do not relax validation thresholds or treat rejected values as evidence."),
    "timeout": (True, "Inspect the completed steps and elapsed time. Request a larger allowed wall-time limit "
                "or a smaller job; do not repeat the unchanged request automatically."),
    "invalid_args": (True, "Correct the indicated parameters before executing the request."),
    "missing_input": (True, "Supply a valid registered input for this route or explicitly request its preparation."),
    "unsupported": (False, "This route is unavailable for this configuration or input. "
                    "Choose a supported route; repeating this request will not enable it."),
    "parse_failed": (False, "Inspect the saved output and parser contract. "
                     "No numerical-precision or convergence diagnosis has been established."),
}
_ALIASES = {"capability_unsupported": "unsupported", "precondition_missing": "missing_input",
            "tool_args_invalid": "invalid_args"}
_KINDS = {"unsupported": "capability_unsupported", "missing_input": "precondition_missing",
          "invalid_args": "tool_args_invalid"}


def tool_failure(code: str, message: str, *, details: dict | None = None) -> FailureReport:
    category = _ALIASES.get(code, code)
    recoverable, remedy = _POLICIES.get(category, (True,
        "Inspect the saved failure and execution logs before deciding whether to correct, retry or change route."))
    kind = "partial_evidence" if code.startswith("partial_") else _KINDS.get(category, "runtime_failed")
    return FailureReport(kind=kind, message=message,
        recoverable=recoverable, details={**(details or {}), "code": code, "remedy": remedy,
            "automatic_retry": False})


def member_failures(observables: dict, status: str) -> FailureReport:
    """Aggregate failed members without losing successful members or their provenance."""
    if status == "success":
        return FailureReport()
    rows = observables.get("search_attempts", observables.get("points", []))
    failures = []
    for row in rows:
        if row.get("status") in {"success", "duplicate"} or row.get("converged") is True:
            continue
        code = row.get("failure_code", "not_converged" if row.get("converged") is False else "runtime_failed")
        message = row.get("reason", row.get("failure_message", "Requested member did not produce a usable result."))
        report = tool_failure(code, message)
        failures.append({k: row[k] for k in ("member_id", "search_index", "coordinate_value", "status") if k in row} | {
            "code": code, "message": report.message, "recoverable": report.recoverable,
            "remedy": report.details["remedy"]})
    codes = list(dict.fromkeys(f["code"] for f in failures))
    code = codes[0] if len(codes) == 1 else "mixed_failures" if codes else "missing_results"
    message = "; ".join(dict.fromkeys(f["message"] for f in failures)) or "Requested results are incomplete."
    report = tool_failure(code, message, details={"member_failures": failures,
        "successful_results_preserved": status == "partial"})
    return report.model_copy(update={"kind": "partial_evidence" if status == "partial" else report.kind,
                                     "recoverable": any(f["recoverable"] for f in failures)})
