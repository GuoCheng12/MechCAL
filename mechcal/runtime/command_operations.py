"""Atomic command handlers. Scientific review never acts as a format gate."""

import hashlib
import json
from copy import deepcopy

from pydantic import ValidationError

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.command_memory import READ_BYTES, json_bytes, pointer_child, record_layout, result_report, selection_coverage
from mechcal.runtime.command_views import workflow_records
from mechcal.schemas import DispatchRequest, FinalAnswer, MechanismPrediction
from mechcal.schemas.commands import ReadEvidence, internal_arguments
from mechcal.schemas.evidence import HypothesisEntry, HypothesisPortfolio


class OperationRejected(Exception):
    def __init__(self, issues, *, exchange=None, judgment_draft=None):
        self.issues = issues
        self.exchange = exchange
        self.judgment_draft = judgment_draft
        super().__init__("; ".join(i["message"] for i in issues))


def reject(path, code, message, remedy, *, record_id=None, details=None):
    issue = dict(path=path, code=code, message=message, remedy=remedy, record_id=record_id)
    if details is not None:
        issue["details"] = details
    raise OperationRejected([issue])


def protocol(case):
    return case.runtime.get("command_protocol", {})


def put(case, **changes):
    return case.touch(runtime={**case.runtime, "command_protocol": {**protocol(case), **changes}})


def check_owner(case):
    if any(owner != case.case_id for owner in (
        case.input.case_id, case.evidence_ledger.case_id, case.artifact_manifest.case_id,
        protocol(case).get("case_id", case.case_id),
        (protocol(case).get("judgment_draft") or {}).get("case_id", case.case_id),
        (protocol(case).get("exploration") or {}).get("case_id", case.case_id),
        (protocol(case).get("evidence_followup") or {}).get("case_id", case.case_id),
        (protocol(case).get("review_workflow") or {}).get("case_id", case.case_id),
        (protocol(case).get("investigation_policy") or {}).get("case_id", case.case_id),
        (protocol(case).get("reviewer_exploration") or {}).get("case_id", case.case_id),
    )):
        raise RuntimeError("Case storage ownership mismatch; no command may execute.")


def observation_records(case):
    """Index source identities, including diagnostics excluded from the ledger."""
    check_owner(case)
    p = protocol(case)
    ledger = {e.evidence_id: e.model_dump(mode="json") for e in case.evidence_ledger.items}
    available = {
        "input:smiles": {"smiles": case.input.smiles},
        **ledger,
        **{a.artifact_id: a.model_dump(mode="json") for a in case.artifact_manifest.artifacts},
        **p.get("results", {}),
    }
    seen = set()
    for result_id, result in p.get("results", {}).items():
        if not isinstance(result, dict):
            continue
        if (result.get("tool_result_id", result_id) != result_id
                or result.get("case_id", case.case_id) != case.case_id):
            raise RuntimeError(f"Result identity or case ownership mismatch: {result_id}")
        for unit in result.get("evidence_units", []):
            ref = unit.get("evidence_id")
            # Older summary-only entries have no standalone identity.
            if ref is None:
                continue
            if (not isinstance(ref, str) or not ref.strip()
                    or unit.get("source_report_id") != result_id
                    or unit.get("case_id", case.case_id) != case.case_id):
                raise RuntimeError(f"Evidence identity or parent ownership mismatch: {ref}")
            if ref in seen:
                raise RuntimeError(f"Duplicate source evidence ID: {ref}")
            seen.add(ref)
            if ref in available:
                # The ledger may add derived claim links, but cannot replace raw facts.
                accepted = ledger.get(ref)
                if (accepted is None or available[ref] != accepted
                        or any(accepted.get(k) != v for k, v in unit.items() if k != "claim_refs")):
                    raise RuntimeError(f"Conflicting source record ID: {ref}")
            else:
                available[ref] = unit
    return available


def _all_records(case, available):
    p = protocol(case)
    derived = {
        **p.get("actions", {}),
        **p.get("experiments", {}),
        **p.get("arguments", {}),
        **{"report:" + rid: p.get("result_reports", {}).get(rid) or result_report(rid, value)
           for rid, value in p.get("results", {}).items() if isinstance(value, dict)},
        **{"explanation:" + rid: {"source_result_id": rid, **value}
           for rid, value in p.get("explanations", {}).items()},
        **({"draft:" + p["judgment_draft"]["draft_id"]: p["judgment_draft"]}
           if p.get("judgment_draft") else {}),
        **({p["final_draft"]["draft_id"]: p["final_draft"]} if p.get("final_draft") else {}),
        **p.get("review_receipts", {}),
        **({"followup:state": p["evidence_followup"]} if p.get("evidence_followup") is not None else {}),
        **workflow_records(case),
    }
    context = p.get("planner_context")
    if context is not None:
        if context.get("case_id") != case.case_id:
            raise RuntimeError("Planner context belongs to a different case")
        derived["context:planner"] = context
    context = p.get("review_context")
    if context is not None:
        from mechcal.runtime.command_review_context import context_ref
        if (context.get("case_id") != case.case_id or context.get("role") != "reviewer"
                or context.get("ref") != context_ref(context)):
            raise RuntimeError("Review context identity or case ownership mismatch")
        if context["ref"] in derived:
            raise RuntimeError("Review context shadows a derived record")
        derived[context["ref"]] = context
    for ref, receipt in p.get("review_receipts", {}).items():
        if receipt.get("case_id") != case.case_id or receipt.get("ref") != ref:
            raise RuntimeError(f"Review receipt identity or case ownership mismatch: {ref}")
    for group in ("actions", "experiments"):
        for ref, row in p.get(group, {}).items():
            if row.get("case_id", case.case_id) != case.case_id:
                raise RuntimeError(f"Operation case ownership mismatch: {ref}")
    groups = [p.get(name, {}) for name in ("actions", "experiments", "arguments")]
    if sum(map(len, groups)) != len(set().union(*groups)):
        raise RuntimeError("Operation and interpretation record IDs collide")
    conflicts = available.keys() & derived.keys()
    if conflicts:
        raise RuntimeError("Derived records shadow source IDs: " + ", ".join(sorted(conflicts)))
    return {**available, **derived}


class RecordIndex:
    """Ownership-checked view for one immutable case snapshot and one operation.

    Never persist or share this index across a case update. Raw records remain
    authoritative; this only avoids rebuilding their projections per reference.
    """

    def __init__(self, case):
        self.case = case
        self.observations = observation_records(case)
        self.available = _all_records(case, self.observations)
        self.artifact_producers = {}
        for rid, result in protocol(case).get("results", {}).items():
            if not isinstance(result, dict):
                continue
            for artifact in result.get("artifact_updates", []):
                ref = artifact.get("artifact_id")
                if ref:
                    self.artifact_producers.setdefault(ref, []).append(rid)

    def require(self, case):
        if self.case is not case:
            raise RuntimeError("Record index belongs to a different case snapshot")
        return self


def record_index(case, index=None):
    return RecordIndex(case) if index is None else index.require(case)


def records(case, *, index=None):
    return record_index(case, index).available


def citation_source_refs(case, refs):
    """Allow only the selected records and their explicitly registered parents.

    A parent supplies exact calculation fields for a cited evidence unit. This
    is not an alias: IDs and paths stay unchanged, and siblings are not added.
    Callers validate record ownership/identity before using this relation.
    """
    selected = list(dict.fromkeys(refs))
    for result_id, result in protocol(case).get("results", {}).items():
        if not isinstance(result, dict):
            continue
        if any(unit.get("evidence_id") in refs and unit.get("source_report_id") == result_id
               for unit in result.get("evidence_units", [])) and result_id not in selected:
            selected.append(result_id)
    return selected


def record_identity(case, ref, *, index=None):
    """The same identities are displayed, read and validated; no scientific verdict."""
    p = protocol(case)
    index = record_index(case, index)
    known = index.available
    row = known[ref]
    operation = ref in p.get("actions", {}) or ref in p.get("experiments", {})
    observation = ref in index.observations
    parent = p.get("results", {}).get(row.get("source_report_id"), {})
    failed = any(r.get("status") in {"failed", "unsupported", "precondition_missing", "missing"}
                 for r in (row, parent))
    return {"ref": ref, "record_kind": "operation" if operation else "observation" if observation else "interpretation",
            "workspace_read": (ref in workflow_records(case) or ref in p.get("arguments", {})
                               or ref == "context:planner"
                               or ref == (p.get("review_context") or {}).get("ref")
                               or ref == "draft:" + (p.get("judgment_draft") or {}).get("draft_id", "")
                               or ref == (p.get("final_draft") or {}).get("draft_id")),
            "citation_uses": (["support", "limitation"] if observation and not failed else
                              ["limitation"] if observation or operation else []),
            "citation_source_refs": citation_source_refs(case, [ref]) if observation or operation else [],
            **({"source_result_ref": ref} if ref in p.get("results", {}) else
               {"source_result_ref": row["source_report_id"]} if parent else {}),
            **({"result_ref": row["result_id"]} if operation and row.get("result_id") else {})}


def reference_issues(case, refs, path, *, scientific=False, scalar=False, allow_failed=False, index=None):
    index = record_index(case, index)
    known, observations = index.available, index.observations
    issues = []
    for i, ref in enumerate(refs):
        location = path if scalar else [*path, i]
        issue = dict(path=location, record_id=ref)
        if ref not in known:
            issues.append(dict(**issue, code="unknown_reference", message="Reference is not in this case.",
                remedy="Use an exact record ID from the index; do not append a JSON pointer to an ID."))
            continue
        item = known[ref]
        parent_id = item.get("source_report_id") if isinstance(item, dict) else None
        parent = protocol(case).get("results", {}).get(parent_id, {})
        operation = ref in protocol(case).get("actions", {}) or ref in protocol(case).get("experiments", {})
        if scientific and operation and allow_failed:
            continue
        if scientific and ref not in observations:
            identity = record_identity(case, ref, index=index)
            issues.append(dict(**issue, code="not_observation", message="An argument or action is not a raw observation.",
                remedy=("Use an operation only in limitation_refs. For support cite its saved result, if any."
                        if operation else "This is context only: omit it from both refs and limitation_refs. "
                        "For evidence cite the underlying observation; keep planning notes as text."),
                details=identity))
        elif scientific and not allow_failed and any(
            isinstance(row, dict) and row.get("status") in {"failed", "unsupported", "precondition_missing", "missing"}
            for row in (item, parent)
        ):
            issues.append(dict(**issue, code="failed_evidence", message="Failed computation is not scientific evidence.",
                remedy="For a limitation, move the ID from refs to limitation_refs. Otherwise cite a successful observation."))
    return issues


def check_refs(case, refs, path, *, scientific=False, scalar=False, allow_failed=False, index=None):
    issues = reference_issues(case, refs, path, scientific=scientific, scalar=scalar,
                             allow_failed=allow_failed, index=index)
    if issues:
        raise OperationRejected(issues)


def read_records(case, command, *, index=None):
    result = []
    layouts = []
    index = record_index(case, index)
    available = index.available
    for i, query in enumerate(command.records):
        check_refs(case, [query.ref], ["records", i, "ref"], scalar=True, index=index)
        row = available[query.ref]
        selected = {}
        pages = {}
        coverage = {}
        paths = query.paths or [""]
        if query.view == "data" and query.page and len(paths) != 1:
            reject(["records", i, "page"], "ambiguous_page", "A data page selects exactly one list.",
                   "Use one list path per paged record request.", record_id=query.ref)
        for j, path in enumerate(paths):
            value, parent = row, ""
            try:
                if path and not path.startswith("/"):
                    raise ValueError("JSON pointer must begin with /")
                for key in path.split("/")[1:] if path else []:
                    decoded = key.replace("~1", "/").replace("~0", "~")
                    if isinstance(value, list) and (not decoded.isdigit() or str(int(decoded)) != decoded):
                        raise ValueError("Array index must be a nonnegative integer without leading zeros")
                    value = value[int(decoded)] if isinstance(value, list) else value[decoded]
                    parent += "/" + key
            except (KeyError, IndexError, TypeError, ValueError):
                reject(["records", i, "paths", j], "unknown_path", "Record path does not exist.",
                       "Use an exact child path below, or view=directory to navigate without reading values.", record_id=query.ref,
                       details={"record_layout": record_layout(value, parent)})
            coverage[path] = selection_coverage(value, view=query.view,
                                                page=query.page.model_dump() if query.page else None)
            if query.view == "directory":
                selected[path] = record_layout(value, path, **(query.page.model_dump() if query.page else {}))
                layouts.append((query.ref, path, value))
                continue
            layouts.append((query.ref, path, value))
            if query.page:
                if not isinstance(value, list):
                    reject(["records", i, "page"], "not_a_list", "Data paging requires a list value.",
                           "Use view=directory for objects, or read an exact child field with page=null.", record_id=query.ref)
                start, size = query.page.offset, query.page.limit
                pages[path] = {"offset": start, "total_rows": len(value),
                               "next_offset": start + size if start + size < len(value) else None}
                value = value[start:start + size]
            selected[path] = value
        item = {"ref": query.ref, "data": selected if query.paths or query.view == "directory" else selected[""]}
        item["selection"] = query.model_dump(exclude={"ref"})
        item["coverage"] = coverage
        if query.view == "directory":
            item["view"] = "directory"
        if pages:
            item["pages"] = pages
        result.append(item)
    if json_bytes(result) > READ_BYTES:
        layouts.sort(key=lambda entry: json_bytes(entry[2]), reverse=True)
        reject(["records"], "response_too_large", "Selected records exceed the 24 KB read allowance.",
               "Select exact child paths, view=directory, or a smaller list page. No evidence was truncated or changed.",
               details={"record_layouts": [{"ref": ref, **record_layout(value, path)}
                                           for ref, path, value in layouts[:6]],
                        "omitted_layouts": max(0, len(layouts) - 6)})
    return result


def revise(case, command):
    check_owner(case)
    p = protocol(case)
    arguments = deepcopy(p.get("arguments", {}))
    ranking = deepcopy(p.get("ranking", {}))
    candidates = deepcopy(p.get("candidates", {}))
    seen = set()
    issues = []
    known = records(case)

    def issue(path, code, message, remedy, **details):
        issues.append(dict(path=path, code=code, message=message, remedy=remedy,
                           record_id=details.pop("record_id", None), **details))

    for i, change in enumerate(command.changes):
        key = (change.kind, change.label) if change.kind in {"ranking", "candidate"} else ("argument", change.id)
        if key in seen:
            issue(["changes", i], "duplicate_edit", "One transaction edits this record twice.",
                   "Combine the edits into one record.")
            continue
        seen.add(key)
        if change.kind == "withdraw_argument":
            if change.id not in arguments:
                issue(["changes", i, "id"], "unknown_argument", "Cannot withdraw an unknown argument.",
                       "Use an existing argument ID.", record_id=change.id)
                continue
            del arguments[change.id]
            continue
        if change.label not in MECHANISM_POOL:
            issue(["changes", i, "label"], "unknown_label", "Label is outside the mechanism pool.",
                   "Use one of the supplied canonical labels.")
            continue
        if change.kind in {"argument", "candidate"}:
            issues += reference_issues(case, change.refs, ["changes", i, "refs"], scientific=True)
            issues += reference_issues(case, change.limitation_refs, ["changes", i, "limitation_refs"],
                                       scientific=True, allow_failed=True)
        if change.kind == "candidate":
            candidates[change.label] = change.model_dump(mode="json")
        elif change.kind == "argument":
            if change.id in known and change.id not in p.get("arguments", {}):
                issue(["changes", i, "id"], "record_collision", "Argument ID already belongs to another record.",
                       "Choose a new argument ID; observation and action IDs are reserved.", record_id=change.id)
                continue
            allowed_sources = citation_source_refs(case, [*change.refs, *change.limitation_refs])
            for j, citation in enumerate(change.citations):
                if citation.ref not in allowed_sources or citation.view != "data":
                    issue(["changes", i, "citations", j], "citation_scope",
                          "Citation must select data from a cited record or its registered source result.",
                          "Use an allowed source with its own exact paths. Do not move paths to a child record.",
                          record_id=citation.ref, details={"allowed_source_refs": allowed_sources})
            if change.citations:
                try:
                    read_records(case, ReadEvidence(kind="read_evidence", records=change.citations))
                except OperationRejected as exc:
                    issues.extend({**item, "path": ["changes", i, "citations", *item["path"][1:]]}
                                  for item in exc.issues)
            arguments[change.id] = change.model_dump(mode="json")
        else:
            ranking[change.label] = change.model_dump(mode="json")
    # Validate the completed transaction, including unchanged records. No partial
    # acceptance when withdrawing an argument still referenced by another entry.
    for label, entry in ranking.items():
        for ref in entry["argument_ids"]:
            if ref not in arguments or arguments[ref]["label"] != label:
                index = next((i for i, c in enumerate(command.changes)
                              if getattr(c, "label", None) == label and c.kind == "ranking"), None)
                path = ["changes", index, "argument_ids"] if index is not None else ["changes"]
                issue(path, "argument_link", "Ranking references a missing or differently labelled argument.",
                      "Correct the ranking's argument_ids or the argument label in the same draft.", record_id=ref,
                      details={"expected_label": label, "actual_label": arguments.get(ref, {}).get("label"),
                               "matching_argument_ids": [aid for aid, a in arguments.items() if a["label"] == label]})
    if issues:
        raise OperationRejected(issues)
    if (arguments == p.get("arguments", {}) and ranking == p.get("ranking", {})
            and candidates == p.get("candidates", {})):
        return case
    hypotheses = [HypothesisEntry(
        name=label, confidence=r["confidence"], differential_priority=r["priority"],
        rationale=("\n\n".join(arguments[aid]["text"] for aid in r["argument_ids"])
                   if p.get("review_workflow") is not None else r["rationale"]),
        evidence_support=r["support"], claim_status=r["claim"],
        evidence_refs=list(dict.fromkeys(ref for aid in r["argument_ids"] for ref in arguments[aid]["refs"])),
        validation_needed=r["validation_needed"],
    ) for label, r in ranking.items()]
    ordered = HypothesisPortfolio(hypotheses=hypotheses).sorted_hypotheses()
    portfolio = HypothesisPortfolio(hypotheses=hypotheses,
        current=ordered[0].name if ordered else "unknown",
        runner_up=ordered[1].name if len(ordered) > 1 else None)
    return put(case.touch(portfolio=portfolio), arguments=arguments, ranking=ranking, candidates=candidates,
               revision=p.get("revision", 0) + 1, note=command.note)


def cached_computation(case, ref):
    p = protocol(case)
    result = p.get("results", {}).get(ref)
    if (result is None or result.get("tool_result_id", ref) != ref
            or result.get("case_id", case.case_id) != case.case_id):
        raise RuntimeError("Computation cache points to a missing or foreign result: " + ref)
    return {"status": "reused", "result_id": ref, "executed": False,
            "result_status": result["status"], "failure": result.get("failure"),
            "retry_of": ref if result["status"] in {"failed", "partial"} else None,
            "reason": "Saved result returned; no computation ran. To retry a failed or partial result, "
                      "set retry_of to its result ID. A retry in purpose text alone does not execute work."}


def prepare_computation(case, command, registry, *, amesp_enabled):
    check_owner(case)
    from mechcal.runtime.command_followup import question_snapshot
    question_snapshot(case, getattr(command, "question_id", None))
    p = protocol(case)
    spec = registry.require(command.capability_id)
    allowed = case.runtime.get("ablation", {}).get("enabled_capability_owners", ["macro", "microscopic"])
    if spec.owner_agent not in allowed:
        reject(["capability_id"], "permission", "This route is disabled by the ablation configuration.",
               "Choose an enabled route.")
    if p.get("loaded_tools", {}).get(spec.capability_id) != registry.definition_version(spec.capability_id):
        reject(["capability_id"], "description_required", "Tool definition is not loaded or has changed.",
               "Use describe_tools before execution.")
    try:
        args = internal_arguments(command, spec)
    except ValidationError as exc:
        raise OperationRejected([dict(path=["arguments", *e["loc"]], code=e["type"],
            message=e["msg"], remedy="Use the loaded tool's parameter contract.", record_id=None)
            for e in exc.errors(include_input=False, include_url=False)]) from exc
    except ValueError as exc:
        reject(["arguments"], "parameter_combination", str(exc), "Correct the indicated parameters.")
    artifact_ids = {a.artifact_id for a in case.artifact_manifest.artifacts}
    for key, value in args.items():
        if key.endswith("artifact_id") and value not in artifact_ids:
            reject(["arguments", key], "unknown_artifact", "Artifact does not belong to this case.",
                   "Use an artifact ID from this case's index.", record_id=value)
    if spec.input_artifact_kinds and "source_artifact_id" in type(command.arguments).model_fields:
        if not args.get("source_artifact_id"):
            reject(["arguments", "source_artifact_id"], "missing_input", "Select the calculation input explicitly.",
                   "Read the artifact index and provide source_artifact_id.")
    digest = hashlib.sha256(json.dumps({
        "case": case.case_id, "smiles": case.input.smiles,
        "tool": spec.capability_id, "definition": registry.definition_version(spec.capability_id),
        "arguments": args,
    }, sort_keys=True).encode()).hexdigest()
    base_digest = digest
    retry_of = command.retry_of
    if retry_of is not None:
        result = p.get("results", {}).get(retry_of)
        if result is None:
            reject(["retry_of"], "unknown_result", "Retry target is not a result in this case.",
                   "Use the failed or partial result ID, not a tool-call ID.", record_id=retry_of)
        if (result.get("tool_result_id", retry_of) != retry_of
                or result.get("case_id", case.case_id) != case.case_id):
            raise RuntimeError("Retry result identity or ownership mismatch: " + retry_of)
        if result["status"] not in {"failed", "partial"}:
            reject(["retry_of"], "not_retryable", "Only failed or partial calculations can be explicitly retried.",
                   "Reuse successful results; obtain missing inputs before submitting a new calculation.", record_id=retry_of)
        previous = p.get("actions", {}).get(result.get("dispatch_id"))
        if previous is None:
            raise RuntimeError("Retry result has no saved execution request: " + retry_of)
        if previous["capability_id"] != spec.capability_id or previous["tool_args"] != args:
            reject(["retry_of"], "retry_mismatch", "Retry must use the original tool and parameters.",
                   "For changed parameters submit a new computation with retry_of=null.", record_id=retry_of)
        # One explicit retry target identifies one attempt, including after resume.
        digest = hashlib.sha256(json.dumps({"base": base_digest, "retry_of": retry_of}, sort_keys=True).encode()).hexdigest()
    if digest in p.get("computations", {}):
        return None, digest
    if retry_of is not None and p.get("computations", {}).get(base_digest) != retry_of:
        reject(["retry_of"], "stale_retry", "A newer calculation or changed tool definition supersedes this retry target.",
               "Inspect the latest result before deciding to retry.", record_id=retry_of)
    if not case.runtime.get("run_limits", {}).get("scientific_tools_allowed", True):
        reject(["kind"], "execution_limit", "The configured scientific execution limit has been reached.",
               "Use saved evidence and finalize; do not claim scientific convergence.")
    rid = f"R{p.get('rounds', 0) + 1:03d}"
    request = DispatchRequest(
        dispatch_id="command:" + digest[:24], round_id=rid, agent_name=spec.owner_agent,
        capability_id=spec.capability_id, route=spec.route, evidence_goal_family=spec.evidence_family,
        task=command.purpose, objective=command.purpose, tool_args=args,
        input_refs=[v for k, v in args.items() if k.endswith("artifact_id")],
        constraints=({"command_retry": {"result_id": retry_of, "base_digest": base_digest}} if retry_of else {}),
        retry_reason=command.purpose if retry_of else "",
    )
    violations = registry.validate_dispatch(request, case.artifact_manifest)
    violations += registry.availability_violations(request, case.artifact_manifest, amesp_enabled=amesp_enabled)
    if violations:
        reject(["arguments"], "not_executable", "; ".join(violations),
               "Correct parameters or obtain the missing input; no computation ran.")
    return request, digest


def finish(case, command):
    check_owner(case)
    p = protocol(case)
    if p.get("judgment_draft"):
        reject(["kind"], "pending_judgment", "An uncommitted judgment draft is still open.",
               "Repair it, replace it explicitly with revise_judgment, or discard it before finalizing accepted records.")
    ordered = case.portfolio.sorted_hypotheses()
    if len(ordered) < 3:
        reject(["kind"], "incomplete_ranking", "Final delivery requires three ranked mechanisms.",
               "Submit ranking changes first, then finalize.")
    predictions = []
    for i, h in enumerate(ordered[:3], 1):
        entry = p["ranking"][h.name]
        arguments = [p["arguments"][aid] for aid in entry["argument_ids"]]
        limitation_only = [a["text"] for a in arguments if a.get("limitation_refs") and not a["refs"]]
        predictions.append(MechanismPrediction(
            label=h.name, rank=i, confidence=h.confidence,
            differential_priority=h.differential_priority,
            claim_status=h.claim_status, support_strength=h.evidence_support,
            evidence_refs=h.evidence_refs,
            evidence=[a["text"] for a in arguments if not a.get("limitation_refs") or a["refs"]],
            validation_needed=h.validation_needed, limitations=[*command.limitations, *limitation_only],
        ))
    answer = FinalAnswer(
        case_id=case.case_id, current_hypothesis=ordered[0].name,
        confidence=ordered[0].confidence, answer=command.answer,
        mechanism_predictions=predictions, limitations=command.limitations,
        evidence_refs=list(dict.fromkeys(r for h in ordered[:3] for r in h.evidence_refs)),
        artifact_refs=[a.artifact_id for a in case.artifact_manifest.artifacts],
        round_refs=case.round_ids,
    )
    return put(case.touch(status="finalized", final_answer=answer,
                          mechanism_predictions=predictions), phase="done")
