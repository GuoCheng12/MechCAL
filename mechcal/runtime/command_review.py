"""Attach scoped execution facts to reviewed sources, not inferred conditions."""

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.agents.observation_view import observation_view
from mechcal.runtime.command_context import source_card
from mechcal.runtime.command_memory import digest, json_bytes, pointer_child, result_report, source_provenance
from mechcal.runtime.command_operations import protocol, read_records, record_identity, record_index
from mechcal.runtime.command_views import candidate_view, enabled
from mechcal.schemas import CaseRun
from mechcal.schemas.commands import ReadEvidence
from mechcal.tools.qc_logs import failure_diagnostics


def candidate_overview(case: CaseRun) -> list[dict]:
    p = protocol(case)
    if p.get("evidence_followup") is not None:
        entries = (p.get("exploration") or {}).get("entries", {})
        return [candidate_view(case, {"label": label, "ranked": label in p.get("ranking", {}),
                 **entries.get(label, {"confidence": None, "note": "Unassessed.", "refs": [], "limitation_refs": []})})
                for label in MECHANISM_POOL]
    return [{"label": label, "ranked": label in p.get("ranking", {}),
             **p.get("candidates", {}).get(label, {
                 "status": "considering" if label in p.get("ranking", {}) else "unexamined",
                 "note": "No separate comparison note recorded.", "refs": [], "limitation_refs": []})}
            for label in MECHANISM_POOL]


def review_basis(case: CaseRun, *, index=None) -> str:
    """The available observations/requests, not judgments or review prose."""
    index = record_index(case, index)
    return index.memo("review_basis", lambda: digest({"case_id": case.case_id, "observations": index.observations,
                       "actions": protocol(case).get("actions", {})}))


def claim_labels(case: CaseRun) -> set[str]:
    p = protocol(case)
    return set(p.get("ranking", {})) | {a["label"] for a in p.get("arguments", {}).values()}


def interim_review_trigger(case: CaseRun, *, index=None) -> str | None:
    """Schedule review once per source basis and formal claim scope, not per edit.

    This is a scheduling receipt, never approval. Explicit review and the exact
    content-bound final review remain separate paths.
    """
    previous = protocol(case).get("interim_review")
    if not previous:
        return "initial_claims"
    if previous["basis"] != review_basis(case, index=index):
        return "new_observations"
    if claim_labels(case) - set(previous["labels"]):
        return "new_claim_scope"
    return None


def completed_interim_review(case: CaseRun, *, index=None) -> dict:
    basis = review_basis(case, index=index)
    previous = protocol(case).get("interim_review") or {}
    labels = claim_labels(case)
    if previous.get("basis") == basis:
        labels.update(previous["labels"])
    return {"basis": basis, "labels": sorted(labels), "revision": protocol(case)["revision"]}


def reference_contract(case: CaseRun, refs, *, index=None) -> dict:
    """Use the same record identities as the command validator, not a second list."""
    index = record_index(case, index)
    identities = [record_identity(case, ref, index=index) for ref in sorted(set(refs))]
    return {"support_refs": [r["ref"] for r in identities if "support" in r["citation_uses"]],
            "limitation_refs": [r["ref"] for r in identities if "limitation" in r["citation_uses"]],
            "context_only_refs": [r["ref"] for r in identities if not r["citation_uses"]]}


def review_citations(case: CaseRun, arguments: list[dict], *, index=None) -> list[dict]:
    """Resolve the exact Planner-selected fields; never regenerate numbers with an LLM."""
    index = record_index(case, index)
    rows, seen = [], set()
    for argument in arguments:
        selections = argument.get("citations", [])
        if not selections:
            continue
        for row in read_records(case, ReadEvidence(kind="read_evidence", records=selections), index=index):
            identity = digest({"ref": row["ref"], "selection": row["selection"]})
            if identity not in seen:
                rows.append({**row, "values_sha256": digest(row["data"])})
                seen.add(identity)
    return rows


def review_observations(case, refs, available):
    """Exact bounded views, with full records available via scoped read_evidence."""
    shown = {}
    results = protocol(case).get("results", {})
    for ref in sorted(refs):
        row = available[ref]
        if json_bytes(row) <= 2048:
            shown[ref] = observation_view(row, preserve_paths=True)
        elif ref in results:
            shown[ref] = result_report(ref, row)
        else:
            fields, deferred = {}, []
            for name, value in row.items():
                if json_bytes(value) <= 1024 and json_bytes({**fields, name: value}) <= 2048:
                    fields[name] = value
                else:
                    deferred.append({"path": pointer_child("", name), "bytes": json_bytes(value)})
            shown[ref] = {"fields": fields, "deferred": deferred,
                          "policy": "Selected exact fields; deferred values remain readable at the original ref/path."}
    return shown


def reviewer_scope(case, refs, available):
    """Cited records and their direct execution/input/output records, not all history."""
    scoped = set(refs)
    contexts = review_execution_context(case, refs, available)
    for interpretation in review_worker_interpretations(case, contexts):
        scoped.add(interpretation["full_text_ref"])
        scoped.update(interpretation["refs"])
        observations = interpretation.get("source_observations") or {}
        scoped.update(row["ref"] for key in ("findings", "other_reads") for row in observations.get(key, []))
    for context in review_execution_context(case, scoped & available.keys(), available):
        rid = context["result_id"]
        result = protocol(case)["results"][rid]
        scoped.add(rid)
        scoped.update(e["evidence_id"] for e in result.get("evidence_units", []) if e.get("evidence_id"))
        scoped.update(a["artifact_id"] for a in result.get("artifact_updates", []))
        request = context.get("request")
        if request:
            scoped.add(request["dispatch_id"])
            scoped.update(request["input_refs"])
    return {ref: available[ref] for ref in sorted(scoped) if ref in available}


def _parameters(ref: str, path: str, values: dict | None) -> dict | None:
    if values is None:
        return None
    shown, deferred = {}, []
    for key, value in values.items():
        if json_bytes(value) <= 2048 and json_bytes({**shown, key: value}) <= 4096:
            shown[key] = value
        else:
            deferred.append({"path": pointer_child(path, key), "bytes": json_bytes(value)})
    return {"source_ref": ref, "path": path, "values": shown, "deferred": deferred}


def geometry_records(result_id, result):
    """Read declared geometry lineage at the producers' actual storage locations."""
    keys = {"parent_artifact_id", "parent_member_id", "topology_artifact_id", "geometry_source",
            "geometry_validation", "atom_index_base", "atom_count", "heavy_atom_count"}
    containers = [("/structured_results", result.get("structured_results", {}))]
    for i, artifact in enumerate(result.get("artifact_updates", [])):
        metadata = artifact.get("metadata", {})
        path = f"/artifact_updates/{i}/metadata"
        containers.extend([(path, metadata), (path + "/bundle", metadata.get("bundle", {}))])
    return [_parameters(result_id, path, {key: values[key] for key in sorted(keys & values.keys())})
            for path, values in containers if isinstance(values, dict) and keys & values.keys()]


def review_execution_context(case: CaseRun, refs: set[str], available: dict) -> list[dict]:
    """Join only cited sources to exact stored results and dispatches.

    `available` is the ownership-checked records index. Input references identify
    inputs; they do not pull those inputs' contents or upstream execution history.
    """
    p = protocol(case)
    contexts = []
    for result_id, result in p.get("results", {}).items():
        if result_id not in available or not isinstance(result, dict):
            continue
        source_ids = {result_id}
        source_ids.update(ref for ref in refs if isinstance(available[ref], dict)
                          and available[ref].get("source_report_id") == result_id)
        source_ids.update(a["artifact_id"] for a in result.get("artifact_updates", []))
        sources = sorted(refs & source_ids)
        if not sources:
            continue
        dispatch_id = result.get("dispatch_id")
        action = p.get("actions", {}).get(dispatch_id)
        request = None
        if action is not None:
            if (action.get("dispatch_id", dispatch_id) != dispatch_id
                    or action.get("case_id", case.case_id) != case.case_id
                    or any(action[key] != result[key] for key in ("capability_id", "round_id", "agent_name")
                           if key in action and key in result)):
                raise RuntimeError(f"Execution request identity or ownership mismatch: {dispatch_id}")
            request = {"dispatch_id": dispatch_id, "input_refs": action.get("input_refs", []),
                       "parameters": _parameters(dispatch_id, "/tool_args", action.get("tool_args"))}
        computed = result.get("structured_results", {}).get("parsed_observables", {})
        geometries = geometry_records(result_id, result)
        contexts.append({
            "source_refs": sources, "result_id": result_id,
            "capability_id": result.get("capability_id"), "round_id": result.get("round_id"),
            "status": result.get("status"), "failure": result.get("failure"),
            "provenance": source_provenance(result),
            "request": request,
            "outputs": [{key: artifact[key] for key in ("artifact_id", "kind", "status") if key in artifact}
                        for artifact in result.get("artifact_updates", [])],
            "reported_geometry": geometries[0] if len(geometries) == 1 else None,
            "geometry_records": geometries,
            "diagnostics": failure_diagnostics(result),
            "effective_parameters": _parameters(result_id,
                "/structured_results/parsed_observables/effective_parameters",
                computed.get("effective_parameters")),
        })
    return contexts


def review_worker_interpretations(case: CaseRun, contexts: list[dict]) -> list[dict]:
    """One unverified interpretation per task, not independent prose per result."""
    explanations = protocol(case).get("explanations", {})
    rows, seen = [], set()
    for context in contexts:
        rid = context["result_id"]
        explanation = explanations.get(rid)
        if not explanation:
            continue
        identity = explanation.get("task_id") or rid
        if identity in seen:
            continue
        seen.add(identity)
        text = explanation.get("text", "")
        truncated = json_bytes(text) > 8192
        delivery = protocol(case).get("worker_task_reports", {}).get(explanation.get("task_id"), {})
        if delivery.get("case_id", case.case_id) != case.case_id:
            raise RuntimeError("Worker interpretation belongs to another case")
        rows.append({"source_result_ids": explanation.get("source_result_ids", [rid]),
                     "status": explanation.get("status"), "text_preview": text[:1600] if truncated else text,
                     "text_truncated": truncated, "full_text_ref": "explanation:" + rid,
                     "refs": explanation.get("refs", []), "scientific_approval": False,
                     "source_observations": delivery.get("source_observations"),
                     "scope": "task_interpretation" if explanation.get("task_id") else "result_interpretation"})
    return rows


def review_sections(payload, *, granular=False):
    """Partition a review by exact records, without estimating semantic similarity."""
    sections = {}
    for key, value in payload.items():
        if key == "observations":
            sections.update({f"observations:{ref}": row for ref, row in value.items()})
        elif key == "execution_inventory":
            sections.update({f"execution_inventory:{cid}": row for cid, row in value["by_capability"].items()})
            sections["execution_inventory_policy"] = value["policy"]
        elif key in {"arguments", "execution_context", "citations"}:
            for i, row in enumerate(value):
                identity = (row["id"] if key == "arguments" else row["result_id"] if key == "execution_context"
                            else digest({"ref": row["ref"], "selection": row["selection"]}))
                sections[f"{key}:{identity}"] = row
        elif key == "assembled_answer" and isinstance(value, dict):
            for field, item in value.items():
                if granular and field == "mechanism_predictions":
                    sections["assembled_answer:prediction_order"] = [row["label"] for row in item]
                    for row in item:
                        sections[f"assembled_answer:prediction:{row['label']}"] = row
                else:
                    sections[f"assembled_answer:{field}"] = item
        else:
            sections[key] = value
    return sections


def review_projection(case, payload, role, *, index, task_context=False):
    """Reuse exact prior sections only while the complete source basis is unchanged.

    Raw values stay in the case and the prior receipt. Reuse is a context routing
    decision, never scientific approval. New/changed observations invalidate it.
    """
    index.require(case)
    basis = review_basis(case, index=index)
    sections = review_sections(payload, granular=enabled(case))
    fingerprint = digest({"basis": basis, "sections": sections})
    receipt = {"ref": f"review:{role}:{fingerprint}", "case_id": case.case_id,
               "basis": basis, "sections": sections}
    contract = {"mechanism_definitions": payload.get("mechanism_definitions"),
                "comparison_version": payload.get("computational_review", {}).get("candidate_comparison", {}).get("version")}
    if any(value is not None for value in contract.values()):
        receipt["context_contract"] = digest(contract)
    if enabled(case) and role in {"reviewer", "final_reviewer"}:
        from mechcal.runtime.command_review_delta import project
        return project(case, payload, role, sections, receipt, index, task_context=task_context)
    previous = protocol(case).get("review_memory", {}).get(role)
    if previous and index.available.get(previous.get("ref")) != previous:
        raise RuntimeError("Review memory is not the registered case-local receipt")
    if previous and previous.get("context_contract") != receipt.get("context_contract"):
        previous = None
    reusable = previous and (enabled(case) or previous.get("basis") == basis) and previous.get("feedback", {}).get("status") == "available"
    shown, reused = {}, []
    for key, value in sections.items():
        if (reusable and key in previous["sections"] and previous["sections"][key] == value
                and json_bytes(value) > 512 and key.split(":", 1)[0] in
                {"arguments", "observations", "execution_context", "citations", "assembled_answer", "worker_interpretations"}):
            shown[key] = {"previous_review_ref": previous["ref"],
                          "path": pointer_child("/sections", key),
                          "meaning": "Exact section unchanged; prior feedback is not approval."}
            if isinstance(value, dict):
                shown[key].update({k: value[k] for k in ("id", "label", "ref", "status") if k in value})
            reused.append(key)
        elif key.startswith("observations:"):
            ref = key.split(":", 1)[1]
            shown[key] = source_card(ref, index.available[ref])
        elif key in {"changes", "previous_arguments"}:
            # Current claims are already supplied. Only previous wording is useful
            # for comparison; the old evidence body remains readable by source ID.
            shown[key] = ([{k: row[k] for k in ("kind", "id", "label", "text") if k in row}
                           for row in value] if key == "previous_arguments" else
                          [{k: row[k] for k in ("kind", "id", "label") if k in row} for row in value])
        else:
            shown[key] = value
    material = {"stage": payload["stage"], "role": role, "sections": shown,
                "review_contract": "Review current changed sections in context of the ranking and other mechanisms. "
                    "Unchanged sections were previously read, not approved. Keep unresolved prior objections. "
                    "If a change affects an unchanged claim, read its exact saved section. Source cards are previews; "
                    "citations and execution_context contain exact values/conditions. Read raw fields when needed.",
                "reused_sections": reused}
    if enabled(case):
        material["changed_sections"] = [key for key, value in sections.items()
            if not previous or previous.get("sections", {}).get(key) != value]
    if reused:
        material["previous_feedback"] = ({"ref": previous["ref"], "path": "/feedback",
            "meaning": "Prior prose remains readable; tracked current issues are supplied under review_workflow."}
            if enabled(case) else previous["feedback"])
    return material, receipt, previous["ref"] if reused else None
