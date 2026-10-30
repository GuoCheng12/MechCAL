"""Durable independent computations. Partial execution never commits a judgment."""

from copy import deepcopy

from pydantic import ValidationError

from mechcal.runtime.command_operations import OperationRejected, prepare_computation, protocol, put, reject
from mechcal.schemas.commands import computation_model


def pending_experiments(case):
    return [{"experiment_id": eid, "status": row["status"],
             "command": row.get("command"), "original_call": row.get("original_call") if not row.get("command") else None,
             "issues": row.get("issues", [])}
            for eid, row in protocol(case).get("experiments", {}).items()
            if row["status"] in {"blocked", "queued"}]


def _validate(o, case, data):
    spec = o.deps.capability_registry.get(data.get("capability_id"))
    if spec is None:
        reject(["capability_id"], "unknown_tool", "The saved capability is no longer registered.",
               "Cancel this request and select an available capability.")
    try:
        command = computation_model(spec).model_validate(data)
    except ValidationError as exc:
        raise OperationRejected([dict(path=list(e["loc"]), code=e["type"], message=e["msg"],
                                      remedy="Cancel this request and submit corrected parameters.")
                                 for e in exc.errors(include_input=False, include_url=False)]) from exc
    if command.kind != "compute":
        raise RuntimeError("Non-computation stored in experiment queue")
    return prepare_computation(case, command, o.deps.capability_registry, amesp_enabled=o.config.enable_amesp)


def stage_batch(o, case, batch):
    p = protocol(case)
    if p.get("compute_batch"):
        raise RuntimeError("An unfinished batch cannot be overwritten")
    rows = deepcopy(p.get("experiments", {}))
    batch_id = (p.get("latest_exchange") or {}).get("request_id", f"batch:{p['turn']}")
    ids = []
    # Validate every member against the SAME input state, not outputs of siblings.
    for call in batch.calls:
        eid = batch_id + ":" + call.call_id
        if eid in rows:
            raise RuntimeError("Experiment identity already exists")
        row = {"command": call.command, "original_call": call.original_call, "reply_id": call.call_id,
               "status": "queued", "issues": call.issues, "attempts": []}
        if call.issues:
            row["status"] = "blocked"
        else:
            try:
                _validate(o, case, call.command)
            except OperationRejected as exc:
                row.update(status="blocked", issues=exc.issues)
        rows[eid] = row
        ids.append(eid)
    return put(case, experiments=rows, compute_batch={"ids": ids, "active": None},
               phase="batch_next", outcome_request_id=None)


def manage_experiments(o, case, command):
    p = protocol(case)
    rows = deepcopy(p.get("experiments", {}))
    ids = command.experiment_ids
    if len(set(ids)) != len(ids) or any(eid not in rows or rows[eid]["status"] != "blocked" for eid in ids):
        reject(["experiment_ids"], "pending_experiment", "Use distinct currently blocked experiment IDs.",
               "Read pending_experiments; completed calculations cannot be retried by this operation.")
    if command.action == "cancel":
        for eid in ids:
            rows[eid].update(status="cancelled", cancellation_reason=command.reason)
        return put(case, experiments=rows)
    exchange = p.get("latest_exchange")
    reply_ids = ([c["id"] for c in exchange["calls"]] if exchange else [f"control:{p['turn']}"])
    if len(reply_ids) != 1:
        raise RuntimeError("Retry must have one native control call")
    for eid in ids:
        row = rows[eid]
        if row.get("command") is None:
            reject(["experiment_ids"], "invalid_saved_parameters", "This saved request has no valid computation command.",
                   "Cancel it and submit a corrected computation.", record_id=eid)
    for eid in ids:
        row = rows[eid]
        row["attempts"].append({k: row.get(k) for k in ("status", "issues", "reply_id", "result_id")})
        row.update(status="queued", issues=[], reply_id=reply_ids[0])
        try:
            _validate(o, case, row["command"])
        except OperationRejected as exc:
            row.update(status="blocked", issues=exc.issues)
    return put(case, experiments=rows, compute_batch={"ids": ids, "active": None},
               phase="batch_next", outcome_request_id=None)


def retain_rejected_computations(case, exchange, issues):
    if not exchange or not exchange.get("complete"):
        return case
    p = protocol(case)
    rows = deepcopy(p.get("experiments", {}))
    pending = (p.get("pending_command") or {}).get("command")
    for call in exchange.get("calls", []):
        if not call["name"].startswith("compute_"):
            continue
        eid = exchange["request_id"] + ":" + call["id"]
        rows.setdefault(eid, {"command": pending if pending and pending.get("kind") == "compute" else None,
                             "original_call": call, "reply_id": call["id"], "status": "blocked",
                             "issues": issues, "attempts": []})
    return put(case, experiments=rows)


def finish_member(case):
    p = protocol(case)
    batch = deepcopy(p["compute_batch"])
    rows = deepcopy(p["experiments"])
    eid = batch["active"]
    rid = p["pending_result"]
    rows[eid].update(status=p["results"][rid]["status"], result_id=rid, executed=True)
    batch["active"] = None
    return put(case, experiments=rows, compute_batch=batch, phase="batch_next", outcome_request_id=None)


def next_member(o, case):
    p = protocol(case)
    batch = deepcopy(p["compute_batch"])
    rows = deepcopy(p["experiments"])
    for eid in batch["ids"]:
        row = rows[eid]
        if row["status"] != "queued":
            continue
        try:
            request, key = _validate(o, case, row["command"])
        except OperationRejected as exc:
            row.update(status="blocked", issues=exc.issues)
            continue
        if request is None:
            row.update(status="reused", result_id=p["computations"][key], executed=False)
            continue
        batch["active"] = eid
        return put(case, experiments=rows, compute_batch=batch, phase="execute",
                   ticket={"request": request.model_dump(mode="json"), "digest": key}, outcome_request_id=None)
    per_call = {}
    for eid in batch["ids"]:
        row = rows[eid]
        result = {"experiment_id": eid, "status": row["status"], "executed": row.get("executed", False),
                  "issues": row.get("issues", [])}
        if rid := row.get("result_id"):
            result.update(result_id=rid, result_report=p.get("result_reports", {}).get(rid),
                          worker_explanation=p.get("explanations", {}).get(rid))
        per_call.setdefault(row["reply_id"], {"experiments": []})["experiments"].append(result)
    exchange = p.get("latest_exchange")
    return put(case, experiments=rows, compute_batch=None, phase="choose", ticket=None,
               outcome={"status": "compute_batch_completed", "call_results": per_call},
               outcome_request_id=exchange["request_id"] if exchange else None)
