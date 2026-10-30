"""Identify explicit access/billing blockers, not ordinary transient rate limits."""


def is_review_service_failure(exc: BaseException) -> bool:
    """Only known service/output failures may follow a configured unavailable policy."""
    import httpx
    from openai import APIError
    from mechcal.runtime.retry import ComponentResponseRejected, SchemaRetryError

    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ComponentResponseRejected):
            return False
        if isinstance(exc, (SchemaRetryError, APIError, httpx.HTTPError, TimeoutError, ConnectionError)):
            return True
        exc = exc.__cause__
    return False


class ApiRunPaused(RuntimeError):
    def __init__(self, reason: str, *, partial: dict | None = None):
        super().__init__(f"API execution paused: {reason}")
        self.reason = reason
        self.partial = partial or {}


def exception_type_chain(exc: BaseException | None) -> list[str]:
    """Record diagnostic types, never request headers, credentials or response bodies."""
    names, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        names.append(type(exc).__name__)
        exc = exc.__cause__ or exc.__context__
    return names


def is_api_transport_error(exc: BaseException) -> bool:
    import httpx
    from openai import APIConnectionError

    return isinstance(exc, (APIConnectionError, httpx.NetworkError, httpx.TimeoutException,
                            httpx.RemoteProtocolError, TimeoutError, ConnectionError))


def api_interruption_reason(exc: BaseException) -> str | None:
    """Run-level classification after bounded retries, not the per-request retry policy."""
    if reason := api_pause_reason(exc):
        return reason
    return "api_transport_failure" if is_api_transport_error(exc) else None


def api_pause_reason(exc: BaseException) -> str | None:
    visited = set()
    while exc is not None and id(exc) not in visited:
        visited.add(id(exc))
        if isinstance(exc, ApiRunPaused):
            return exc.reason
        if isinstance(exc, PermissionError):
            return "local_permission"
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", getattr(response, "status_code", None))
        if status in {401, 403}:
            return "authentication_or_permission"
        body = getattr(exc, "body", None)
        if body is None and response is not None:
            try:
                body = response.json()
            except (ValueError, AttributeError):
                pass
        error = body.get("error", body) if isinstance(body, dict) else {}
        codes = [getattr(exc, "code", None)]
        if isinstance(error, dict):
            codes += [error.get("code"), error.get("type")]
        if any(isinstance(code, str) and code.lower() in {
            "insufficient_quota", "quota_exceeded", "billing_hard_limit_reached", "credit_balance_exhausted",
        } for code in codes):
            return "quota_exhausted"
        exc = exc.__cause__ or exc.__context__
    return None
