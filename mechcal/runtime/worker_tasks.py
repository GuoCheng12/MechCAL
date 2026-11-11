"""Durable bounded sub-tasks. Workers choose actions; the registry executes them.

Task exchanges are separate from the Planner exchange. Each real calculation
still uses the normal execution checkpoint, ledger and global round accounting.
"""

import hashlib
from pathlib import Path
import time
import uuid

from pydantic import TypeAdapter, ValidationError

from mechcal.agents import worker_task as agent
from mechcal.agents.command_tools import parse_call
from mechcal.agents.command_worker import worker_scope
from mechcal.runtime.command_context import scope_card, scoped_directory, record_table
from mechcal.runtime.command_memory import result_report, json_bytes, digest, READ_BYTES
from mechcal.runtime import command_read_handoff as handoff
from mechcal.runtime.worker_memory import receipt_handles, task_memory, remember_directories, directory_memory, progress_note
from mechcal.runtime.worker_inputs import input_paths, gap_report
from mechcal.runtime.command_operations import (
    OperationRejected, check_owner, check_refs, protocol, put, record_identity,
    record_index, prepare_computation, cached_computation, reject,
)
from mechcal.runtime.command_reads import query_results, merge_handles, READ_KINDS
from mechcal.runtime.command_review import review_execution_context
from mechcal.schemas.commands import tool_argument_model
from mechcal.schemas.worker_tasks import WorkerTaskState


# These are identity/physical-state conditions, not retry recipes. Values default
# to the registered tool default unless Planner binds a different value.
FIXED_CONDITIONS = {"charge", "multiplicity", "spin", "state_root", "method", "basis", "functional", "solvent"}
LOG_SUFFIXES = {".log", ".out", ".err", ".txt", ".stdout", ".stderr"}
READ_OPERATIONS = READ_KINDS | {"describe_tools", "read_worker_log"}


def state(case):
    task = WorkerTaskState.model_validate(protocol(case)["worker_task"])
    if task.case_id != case.case_id:
        raise RuntimeError("Worker task belongs to another case")
    return task


def retain(case, task, **updates):
    return put(case, worker_task=task.model_dump(mode="json"), **updates)


def validate_bindings(command, models):
    """Reject ambiguous or impossible locks before a Worker spends any calls."""
    occupied = set()
    for i, item in enumerate(command.fixed_parameters):
        path = ["fixed_parameters", i]
        if item.capability_id is not None and item.capability_id not in models:
            reject([*path, "capability_id"], "invalid_binding", "Binding names an unauthorized capability.",
                   "Select an ID from allowed_capability_ids.")
        targets = [cid for cid, model in models.items() if item.name in model.model_fields
                   and item.capability_id in (None, cid)]
        if not targets:
            reject([*path, "name"], "invalid_binding", "No targeted tool exposes this parameter: " + item.name,
                   "Use the parameter name and capability_id from the tool description.")
        if item.capability_id is None and len(targets) > 1 and item.name not in FIXED_CONDITIONS:
            reject([*path, "capability_id"], "ambiguous_binding",
                   f"{item.name} is shared by multiple tools; a global lock can prevent intermediate-output handoff.",
                   "Set capability_id to the tool whose input must stay fixed. Leave downstream source IDs "
                   "unbound so the Worker can use task-produced artifacts. For intentionally fixed inputs in "
                   "several tools, add one scoped binding per tool. Available tools: " + ", ".join(targets))
        for cid in targets:
            key = (cid, item.name)
            if key in occupied:
                reject(path, "conflicting_binding", f"More than one binding applies to {cid}.{item.name}.",
                       "Use one binding per tool parameter; do not override a task-wide condition.")
            field = models[cid].model_fields[item.name]
            try:
                TypeAdapter(field.rebuild_annotation()).validate_python(item.value, strict=True)
            except ValidationError:
                reject([*path, "value"], "invalid_binding_value", f"Value does not satisfy {cid}.{item.name}.",
                       "Use the registered parameter type and allowed range.")
            occupied.add(key)


def begin(o, case, command):
    check_owner(case)
    if not o.config.command_worker_tasks:
        reject(["kind"], "operation_disabled", "Worker delegation is disabled.", "Use an advertised operation.")
    if protocol(case).get("worker_task"):
        raise RuntimeError("Another Worker task is already active")
    allowed = case.runtime.get("ablation", {}).get("enabled_capability_owners", ["macro", "microscopic"])
    if command.owner not in allowed:
        reject(["owner"], "permission", "Worker is disabled by this run.", "Use an enabled Worker.")
    check_refs(case, command.input_refs, ["input_refs"])
    models = {}
    for cid in command.allowed_capability_ids:
        spec = o.deps.capability_registry.get(cid)
        if spec is None or spec.owner_agent != command.owner:
            reject(["allowed_capability_ids"], "permission", "Unknown capability or wrong Worker owner: " + cid,
                   "Use registered capabilities owned by the delegated Worker.")
        models[cid] = tool_argument_model(spec)
    validate_bindings(command, models)
    from mechcal.runtime.command_followup import question_snapshot
    question_snapshot(case, command.question_id)
    now = time.time()
    task = WorkerTaskState(case_id=case.case_id, task_id="worker-task:" + uuid.uuid4().hex,
                           contract=command, started_at=now, deadline=now + o.config.worker_task_seconds)
    available = scope(case, task)
    artifacts = set(case.artifact_manifest.artifact_ids())
    for i, item in enumerate(command.fixed_parameters):
        if item.name.endswith("artifact_id") and (item.value not in available or item.value not in artifacts):
            reject(["fixed_parameters", i, "value"], "binding_input_scope",
                   "A fixed input must be an existing, delegated artifact.",
                   "Include the chosen artifact in input_refs. Leave future product IDs unbound.", record_id=item.value)
    paths = task_input_paths(o, case, task, available=available)
    # A saved usable result can still be inspected and delivered without new computation.
    has_result = any(protocol(case).get("results", {}).get(ref, {}).get("status") in {"success", "partial"}
                     for ref in available)
    if not paths["input_ready"] and not has_result:
        report = gap_report(paths, case.artifact_manifest)
        reject(["input_refs"], "delegation_input_gap", "No authorized tool can start with the delegated inputs.",
               "Correct input_refs or allowed_capability_ids before starting a Worker.", details=report)
    return retain(case, task, phase="worker_task", pending_command=None)


def scope(case, task, index=None):
    index = record_index(case, index)
    p = protocol(case)
    refs = {"input:smiles", *task.contract.input_refs}
    # An explicitly delegated result grants its own input/output records, not
    # unrelated case history or arbitrary strings found inside an observation.
    for rid in [*task.contract.input_refs, *task.result_ids]:
        if rid in p.get("results", {}):
            result = p["results"][rid]
            request = p.get("actions", {}).get(result.get("dispatch_id"), {})
            refs.update(worker_scope(case, rid, request, index=index))
    return {ref: index.available[ref] for ref in sorted(refs) if ref in index.available}


def scoped_case(case, task, available):
    manifest = case.artifact_manifest.model_copy(update={
        "artifacts": [a for a in case.artifact_manifest.artifacts if a.artifact_id in available]})
    return put(case.touch(artifact_manifest=manifest), loaded_tools=task.loaded_tools)


def task_input_paths(o, case, task, *, available=None):
    available = scope(case, task) if available is None else available
    narrowed = scoped_case(case, task, available)
    return input_paths(o.deps.capability_registry, task.contract.allowed_capability_ids,
                       narrowed.artifact_manifest, amesp_enabled=o.config.enable_amesp)


def payload(o, case, task):
    index = record_index(case)
    available = scope(case, task, index)
    narrowed = scoped_case(case, task, available)
    directory, paths = scoped_directory([scope_card(record_identity(case, ref, index=index), value)
                                        for ref, value in available.items()])
    allowed = set(task.contract.allowed_capability_ids)
    catalog = [r for r in o.deps.capability_registry.compact_catalog(narrowed.artifact_manifest,
               amesp_enabled=o.config.enable_amesp) if r["capability_id"] in allowed]
    return {
        "task": task.contract.model_dump(exclude={"kind", "question_id"}),
        "smiles": case.input.smiles, "tool_catalog": catalog,
        "loaded_capability_ids": sorted(task.loaded_tools),
        "fixed_default_conditions": sorted(FIXED_CONDITIONS),
        "read_scope": record_table(directory), "read_path_sets": paths,
        "log_index": [{"record_id": ref, "file_key": key, "bytes": path.stat().st_size if path.is_file() else None}
                      for ref, files in log_files(case, task).items() for key, path in files.items()],
        "attempts": [{"result_id": rid, "capability_id": protocol(case)["results"][rid]["capability_id"],
                      "status": protocol(case)["results"][rid]["status"],
                      "artifact_ids": [a["artifact_id"] for a in protocol(case)["results"][rid].get("artifact_updates", [])
                                       if a["artifact_id"] in available]} for rid in task.result_ids],
        "execution_context": review_execution_context(case,
            {*task.contract.input_refs, *task.result_ids} & available.keys(), available),
        "working_memory": task_memory(task, available),
        "navigation_memory": directory_memory(task, available),
        "previous_work_note": progress_note(task),
        "input_access": task_input_paths(o, case, task, available=available),
        "last_input_error": task.input_error,
        "remaining": {"model_calls_including_this": o.config.worker_task_max_calls - task.model_calls + 1,
                      "computations": o.config.worker_task_max_computations - task.tool_attempts,
                      "seconds": max(0, round(task.deadline - time.time(), 1))},
        "latest_result": (result_report(task.result_ids[-1], protocol(case)["results"][task.result_ids[-1]])
                          if task.result_ids and not task.exchange else None),
    }


def log_files(case, task):
    """Tool-registered artifacts and process logs, including nested failed steps."""
    from mechcal.tools.qc_logs import registered_failure_logs
    available = scope(case, task)
    files = {a.artifact_id: {key: Path(path) for key, path in a.file_paths.items()
                          if Path(path).suffix.lower() in LOG_SUFFIXES}
             for a in case.artifact_manifest.artifacts if a.artifact_id in available}
    for ref, row in protocol(case).get("results", {}).items():
        workdir = row.get("failure", {}).get("details", {}).get("workdir")
        if ref not in available:
            continue
        if ref in available and workdir:
            root = Path(workdir)
            if not any(p.is_symlink() for p in (root, *root.parents)) and root.is_dir():
                files[ref] = {p.name: p for p in root.iterdir() if p.suffix.lower() in LOG_SUFFIXES and p.is_file()}
        registered = registered_failure_logs(row)
        if registered:
            files.setdefault(ref, {}).update(registered)
    return files


def read_log(case, task, command):
    files = log_files(case, task)
    path = files.get(command.record_id, {}).get(command.file_key)
    if path is None:
        reject(["record_id", "file_key"], "unknown_log", "No such registered file in this task.",
               "Choose an exact record_id and file_key from log_index.")
    if any(p.is_symlink() for p in (path, *path.parents)):
        reject(["file_key"], "unsafe_log", "Log path contains a symbolic link.", "Request a regular registered log file.")
    try:
        with path.open("rb") as handle:
            handle.seek(command.offset)
            data = handle.read(command.limit)
            total = path.stat().st_size
    except (FileNotFoundError, IsADirectoryError, PermissionError) as exc:
        reject(["file_key"], "log_unavailable", type(exc).__name__, "Use saved failure details or report the missing log.")
    return {"status": "read", "record_id": command.record_id, "file_key": command.file_key,
            "offset": command.offset, "next_offset": command.offset + len(data), "total_bytes": total,
            "chunk_sha256": hashlib.sha256(data).hexdigest(), "text": data.decode("utf-8", errors="replace")}


def finish(case, task, store, *, status, summary, refs, reason="worker_report", input_request=None):
    index = record_index(case)
    available = scope(case, task, index)
    p = protocol(case)
    source_ids = set()
    for ref in refs:
        if ref not in available:
            reject(["refs"], "unknown_reference", "Completion cites a record outside this task: " + ref,
                   "Use an exact ID in read_scope; existing results, evidence and artifacts are all accepted.", record_id=ref)
        if ref in p.get("results", {}):
            source_ids.add(ref)
        source_ids.update(r for r in index.artifact_producers.get(ref, []) if r in available)
        parent = available[ref].get("source_report_id")
        if parent in available and parent in p.get("results", {}):
            source_ids.add(parent)
    usable = [rid for rid in source_ids if p["results"][rid]["status"] in {"success", "partial"}]
    if status in {"complete", "partial"} and not usable:
        reject(["status"], "no_usable_result", "No successful or partial calculation supports this delivery.",
               "Report blocked with the actual failures, or obtain a usable result.")
    report = {"task_id": task.task_id, "case_id": case.case_id, "owner": task.contract.owner,
              "objective": task.contract.objective, "status": status, "summary": summary,
              "refs": refs, "source_result_ids": sorted(source_ids), "all_attempt_result_ids": task.result_ids,
              "usable_result_ids": [r for r in task.result_ids if p["results"][r]["status"] in {"success", "partial"}],
              "model_calls": task.model_calls, "tool_attempts": task.tool_attempts,
              "elapsed_seconds": round(time.time() - task.started_at, 3), "completion_reason": reason,
              "scientific_approval": False,
              "summary_role": "worker_interpretation",
              "source_observations": task_memory(task, available),
              "execution_context": review_execution_context(case, source_ids | set(task.result_ids), available)}
    if input_request is not None:
        report.update(input_request=input_request, summary_role="runtime_input_handoff")
    reports = {**p.get("worker_task_reports", {}), task.task_id: report}
    read_handoffs = dict(p.get("read_handoffs", {}))
    if task.result_ids:
        rid = task.result_ids[-1]
        previous, _ = handoff.retained(case, result_ids=[rid], index=index)
        delivered = [{"ref": row["ref"], "path": row["path"], "page": row.get("page"),
                      "bytes": json_bytes(row["data"]), "sha256": digest(row["data"]), "requested": True}
                     for row in report["source_observations"]["findings"]]
        read_handoffs[rid] = handoff.capture(case, rid, merge_handles(previous, delivered), index=index)
    # Existing explanation records keep Worker prose visible to later review.
    explanations = dict(p.get("explanations", {}))
    for rid in task.result_ids if input_request is None else []:
        explanations[rid] = {"status": "available", "text": summary, "refs": refs,
                             "task_id": task.task_id, "source_result_ids": sorted(source_ids | set(task.result_ids)),
                             "scientific_approval": False, "scope": "task_interpretation"}
    store.write_json("worker_tasks/" + task.task_id.split(":")[1] + ".json",
                     {"task": task.model_dump(mode="json"), "report": report})
    return put(case, worker_task=None, worker_task_reports=reports, explanations=explanations, read_handoffs=read_handoffs,
               phase="choose", ticket=None, pending_command=None, outcome={"status": "worker_task_delivered", **report},
               outcome_request_id=(p.get("latest_exchange") or {}).get("request_id"))


def exhausted(case, task, store, reason):
    usable = [r for r in task.result_ids if protocol(case)["results"][r]["status"] in {"success", "partial"}]
    return finish(case, task, store, status="partial" if usable else "blocked", refs=task.result_ids,
                  summary="Worker task stopped at its operational limit (" + reason + "). "
                          "All recorded results are retained; the delegated objective is not confirmed complete.", reason=reason)


def handoff_inputs(o, case, task, store, reason, *, issues=()):
    usable = [r for r in task.result_ids if protocol(case)["results"][r]["status"] in {"success", "partial"}]
    request = {**gap_report(task_input_paths(o, case, task), case.artifact_manifest), "issues": list(issues)}
    return finish(case, task, store, status="partial" if usable else "blocked", refs=task.result_ids,
                  reason=reason, input_request=request,
                  summary="Execution input handoff required. No new calculation ran. Existing results are retained; "
                          "Planner must correct the referenced input or change the delegated scope.")


def validate_computation(o, case, task, command):
    if command.capability_id not in task.contract.allowed_capability_ids:
        reject(["capability_id"], "permission", "Tool is outside the delegation.", "Return the required route to Planner.")
    if task.tool_attempts >= o.config.worker_task_max_computations or time.time() >= task.deadline:
        reject(["kind"], "task_limit", "No new computation can start within this task.", "Deliver partial or blocked results.")
    spec = o.deps.capability_registry.require(command.capability_id)
    args = command.arguments.model_dump(exclude_none=True)
    requested_seconds = args.get("max_wall_seconds")
    remaining_seconds = max(0, int(task.deadline - time.time()))
    if requested_seconds is not None and requested_seconds > remaining_seconds:
        reject(["arguments", "max_wall_seconds"], "task_time_remaining",
               f"Requested {requested_seconds}s exceeds this Worker's remaining {remaining_seconds}s. Nothing started.",
               "Choose an explicit shorter computation, or return partial/blocked and request a longer delegation. "
               "The runtime will not silently shorten a scientific calculation.")
    fields = tool_argument_model(spec).model_fields
    locked = {name: field.default for name, field in fields.items()
              if name in FIXED_CONDITIONS and not field.is_required()}
    locked.update({item.name: item.value for item in task.contract.fixed_parameters
                   if item.name in fields and item.capability_id in (None, command.capability_id)})
    for name, value in locked.items():
        if args.get(name) != value:
            reject(["arguments", name], "fixed_condition",
                   f"Task fixes {name} to {value!r} for {command.capability_id}.",
                   "Keep this condition or return a partial/blocked report requesting a revised delegation. "
                   "Do not alternate between a locked input and an input rejected by the tool.")
    available = scope(case, task)
    if command.retry_of and command.retry_of not in available:
        reject(["retry_of"], "out_of_scope", "Retry target is outside this task.", "Use an authorized failed result.")
    return prepare_computation(scoped_case(case, task, available), command, o.deps.capability_registry,
                               amesp_enabled=o.config.enable_amesp)


def apply(o, case, task, command, store, *, read_allowance=READ_BYTES):
    registry = o.deps.capability_registry
    if command.kind == "describe_tools":
        if set(command.capability_ids) - set(task.contract.allowed_capability_ids):
            reject(["capability_ids"], "permission", "Description request is outside the delegation.",
                   "Select an authorized capability or return the missing route to Planner.")
        from mechcal.runtime.command_loop import _describe_tools
        details = _describe_tools(o, scoped_case(case, task, scope(case, task)), command)
        if json_bytes(details) > read_allowance:
            reject(["capability_ids"], "response_too_large", "Tool descriptions exceed this read-turn allowance.",
                   "Describe fewer capabilities together.")
        task.loaded_tools.update({cid: registry.definition_version(cid) for cid in command.capability_ids})
        task.outcome = {"status": "described", "tools": details}
    elif command.kind in READ_KINDS:
        available = scope(case, task)
        task.outcome, handles = query_results(case, command, allowed_refs=available, allowance=read_allowance)
        task.read_handles = merge_handles(task.read_handles, handles)
        task.directory_reads = remember_directories(task.directory_reads, task.outcome, available)
    elif command.kind == "read_worker_log":
        if command.limit + 500 > read_allowance:
            reject(["limit"], "response_too_large", "Log page exceeds the remaining read-turn allowance.",
                   "Use a smaller limit or a later read turn.")
        task.outcome = read_log(case, task, command)
    elif command.kind == "complete_worker_task":
        return finish(case, task, store, status=command.status, summary=command.summary, refs=command.refs)
    elif command.kind == "compute":
        paths = task_input_paths(o, case, task)
        if command.capability_id in paths["unreachable"]:
            return handoff_inputs(o, case, task, store, "input_scope_gap")
        request, key = validate_computation(o, case, task, command)
        if request is None:
            rid = protocol(case)["computations"][key]
            if rid not in scope(case, task):
                reject(["arguments"], "cached_out_of_scope", "An existing result is outside this delegation.",
                       "Ask Planner to delegate that result; do not repeat its calculation.")
            task.outcome = cached_computation(case, rid)
            task.outcome["result_report"] = result_report(rid, protocol(case)["results"][rid])
            task.read_handles = merge_handles(task.read_handles, receipt_handles(task.outcome))
            task.result_ids = list(dict.fromkeys([*task.result_ids, rid]))
        else:
            from mechcal.runtime.command_followup import question_snapshot
            task.tool_attempts += 1
            task.pending_command = None
            task.outcome = {"status": "accepted", "executed": False, "dispatch_id": request.dispatch_id}
            return retain(case, task, phase="execute", ticket={"request": request.model_dump(mode="json"),
                "digest": key, "worker_task_id": task.task_id,
                "question": question_snapshot(case, task.contract.question_id)})
    else:
        reject(["kind"], "permission", "This operation is not a Worker operation.", "Use an advertised task operation.")
    task.pending_command = None
    return retain(case, task)


def read_batch(o, case, task, reply, tools, models, store):
    """Independent metadata reads share one bounded response; no science mutates."""
    outcomes = {}
    for call in reply["calls"]:
        candidate_task = task.model_copy(deep=True)
        try:
            command = parse_call(call, tools, models)
            candidate = apply(o, case, candidate_task, command, store,
                              read_allowance=max(0, READ_BYTES - json_bytes(outcomes) - 500))
            case, task = candidate, state(candidate)
            outcome = task.outcome
        except OperationRejected as exc:
            outcome = {"status": "rejected", "executed": False, "issues": exc.issues}
        outcomes[call["id"]] = outcome
    task.pending_command = None
    task.outcome = {"status": "read_batch", "call_results": outcomes}
    return retain(case, task)


def accept_result(case, result_id):
    task = state(case)
    if protocol(case)["ticket"].get("worker_task_id") != task.task_id:
        raise RuntimeError("Worker computation belongs to another task")
    task.result_ids = list(dict.fromkeys([*task.result_ids, result_id]))
    task.input_error = None
    task.outcome = protocol(case)["outcome"]
    task.read_handles = merge_handles(task.read_handles, receipt_handles(task.outcome))
    return retain(case, task, phase="worker_task", ticket=None)


def turn(o, case, store):
    from mechcal.runtime.command_loop import save
    task = state(case)
    client = o.deps.workers[task.contract.owner].report_writer.llm_client
    # Rehydrate definitions from the registry. No persisted schema becomes an
    # execution authority after a code/definition change.
    task.loaded_tools = {cid: version for cid, version in task.loaded_tools.items()
                         if cid in task.contract.allowed_capability_ids and
                         o.deps.capability_registry.definition_version(cid) == version}
    tools, models = agent.contracts(o.deps.capability_registry, task.loaded_tools, client)
    if task.pending_command is None:
        if time.time() >= task.deadline or task.model_calls >= o.config.worker_task_max_calls:
            return exhausted(case, task, store, "time_limit" if time.time() >= task.deadline else "model_call_limit")
        task.model_calls += 1
        case = save(retain(case, task), store)
        store.activity(task.contract.owner + " Worker task: diagnose / execute / deliver", case.current_round_id)
        reply = agent.ask(client, payload(o, case, task), tools, task, store)
        task.exchange = reply
        if reply["complete"] and reply.get("text", "").strip():
            task.progress_note = {"request_id": reply["request_id"], "text": reply["text"],
                                  "after_result_ids": list(task.result_ids)}
        # Persist the model's operation before interpreting or dispatching it.
        task.pending_command = reply
        case = save(retain(case, task), store)
    reply = task.pending_command
    command = None
    try:
        calls = reply["calls"]
        is_reads = 1 < len(calls) <= 4 and all(call["name"] in READ_OPERATIONS for call in calls)
        if not reply["complete"] or (len(calls) != 1 and not is_reads):
            reject(["calls"], "task_command_shape", "No partial response or mixed computation/control batch can run.",
                   "Submit up to four read-only operations, or one computation/completion alone.")
        if is_reads:
            candidate = read_batch(o, case, task, reply, tools, models, store)
        else:
            command = parse_call(calls[0], tools, models)
            candidate = apply(o, case, task, command, store)
    except OperationRejected as exc:
        task.pending_command = None
        task.outcome = {"status": "rejected", "executed": False, "issues": exc.issues}
        input_error = (command is not None and command.kind == "compute" and any(
            issue["code"] in {"missing_input", "unknown_artifact"}
            and issue["path"] and str(issue["path"][-1]).endswith("artifact_id") for issue in exc.issues))
        if input_error:
            scoped = scoped_case(case, task, scope(case, task))
            fingerprint = digest({"capability": o.deps.capability_registry.definition_version(command.capability_id),
                "arguments": command.arguments.model_dump(mode="json"),
                "artifacts": scoped.artifact_manifest.model_dump(mode="json")})
            repeated = task.input_error is not None and task.input_error["fingerprint"] == fingerprint
            task.input_error = {"fingerprint": fingerprint, "capability_id": command.capability_id,
                                "issues": exc.issues,
                                "remedy": "Select an exact visible artifact ID; unchanged input errors return to Planner."}
            candidate = (handoff_inputs(o, case, task, store, "repeated_input_error", issues=exc.issues)
                         if repeated else retain(case, task))
        else:
            candidate = retain(case, task)
    current = protocol(candidate).get("worker_task")
    store.append_jsonl("history/worker_tasks.jsonl", {"task_id": task.task_id, "request_id": reply["request_id"],
        "calls": reply["calls"], "phase": protocol(candidate)["phase"],
        "outcome": current.get("outcome") if current else protocol(candidate)["outcome"]})
    return candidate
