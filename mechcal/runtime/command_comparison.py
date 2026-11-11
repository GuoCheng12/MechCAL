"""Shared-source comparison views, derived from saved records without ranking science."""

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.command_memory import digest, json_bytes
from mechcal.runtime.command_operations import protocol, record_index
from mechcal.runtime.command_review import review_execution_context


def interpretations(case):
    p = protocol(case)
    entries = (p.get("exploration") or {}).get("entries", {})
    arguments = p.get("arguments", {})
    rows = []
    for label in MECHANISM_POOL:
        entry = entries.get(label, p.get("candidates", {}).get(label, {}))
        args = [a for a in arguments.values() if a["label"] == label]
        rows.append({"label": label, "formally_ranked": label in p.get("ranking", {}),
                     "confidence": entry.get("confidence"), "explanation": entry.get("explanation"),
                     "assessment_ref": "judgment:assessments" if label in entries else None,
                     "argument_refs": [a["id"] for a in args],
                     "argument_versions": {a["id"]: digest(a) for a in args},
                     **{field: sorted({r for row in [entry, *args] for r in row.get(field, [])})
                        for field in ("refs", "limitation_refs")}})
    return rows


def comparison_view(case, *, index=None):
    """Group cited records by registered parent, never by matching numbers or prose.

    Artifact inputs establish dependency, not duplicate measurements. Independence
    is left unknown unless declared, and even declared independence is unverified.
    """
    index = record_index(case, index)
    candidates = interpretations(case)
    refs = {r for row in candidates for field in ("refs", "limitation_refs") for r in row[field]}
    contexts = review_execution_context(case, refs, index.available)
    producers = {}
    for row in contexts:
        for ref in row["source_refs"]:
            producers.setdefault(ref, set()).add(row["result_id"])
    owners = {ref: next(iter(ids)) for ref, ids in producers.items() if len(ids) == 1}
    grouped = {}
    for row in candidates:
        for field in ("refs", "limitation_refs"):
            for ref in row[field]:
                source = owners.get(ref, ref)
                item = grouped.setdefault(source, {"source_ref": source, "refs": set(),
                                                    "used_by": set(), "limitations_for": set()})
                item["refs"].add(ref)
                item["used_by" if field == "refs" else "limitations_for"].add(row["label"])
    by_source = {row["result_id"]: row for row in contexts}
    sources = []
    for source, item in sorted(grouped.items()):
        context = by_source.get(source)
        raw = index.available[source]
        entry = {k: sorted(v) if isinstance(v, set) else v for k, v in item.items()}
        entry.update(source_version=index.version(source), status=raw.get("status", "not_recorded"))
        if context:
            inputs = (context["request"] or {}).get("input_refs", [])
            entry.update(capability_id=context["capability_id"], report_ref="report:" + source,
                independence=context["provenance"]["independence"],
                derivation=context["provenance"]["derivation"], input_refs=inputs,
                input_sources=[{"input_ref": ref, "produced_by": index.artifact_producers.get(ref, [])}
                               for ref in inputs])
        else:
            entry["independence"] = "not_recorded"
            if source in producers:
                entry["produced_by"] = sorted(producers[source])
        sources.append(entry)
    return {"version": 1, "interpretations": candidates, "source_directory": sources,
            "policy": "Interpretations are model judgments. Source rows are recorded provenance, not new observations. "
                      "Read exact observations/citations for values. The same source may inform several mechanisms; "
                      "this does not create independent confirmations or give one label ownership of the evidence. "
                      "Shared input geometry alone does not make two calculations identical. Missing measurements "
                      "are unknown, not counterevidence. Compare each case-specific explanation with the public family meaning."}


def comparison_refs(view):
    """Same readable source union for shortlisted and other assessed candidates."""
    return {ref for row in view["source_directory"]
            for ref in [row["source_ref"], *row["refs"], *row.get("produced_by", [])]}


def method_steps(result):
    """Exact recorded labels; input provenance is not proof of convergence."""
    steps = result.get("structured_results", {}).get("method_provenance")
    if steps is None:
        return None
    path = "/structured_results/method_provenance"
    rows = [{"path": f"{path}/{i}", **{k: step[k] for k in
        ("step_id", "source", "requested_method_label", "executed_method_label",
         "recorded_method_label", "input_keywords", "input_sha256") if k in step}}
        for i, step in enumerate(steps)]
    return {"path": path, "count": len(steps), "rows": rows if json_bytes(rows) <= 2048 else None}


def question_evidence_view(case, *, result_ids=None, index=None):
    """Join explicit comparison designs to actual trials; never infer a fair control."""
    from mechcal.runtime.command_followup import state, question_rows
    current = state(case)
    if current is None:
        return {"questions": [], "sources": []}
    questions = question_rows(current, include_closed=True)
    refs_by_question = {}
    for q in questions:
        record = current.questions[q["id"]]
        refs = set(q.get("contrast", {}).get("baseline_refs", []) if q.get("contrast") else [])
        refs.update(t["result_id"] for t in record.trials if t.get("result_id"))
        refs_by_question[q["id"]] = refs
    # Parent views can inspect descendants' observations without inheriting conclusions.
    for q in questions:
        parent = q["parent_id"]
        while parent:
            refs_by_question[parent].update(refs_by_question[q["id"]])
            parent = current.questions[parent].value.parent_id
    selected = [q for q in questions if (q["scope"] or q["parent_id"] or q["contrast"])
                and (result_ids is None or refs_by_question[q["id"]].intersection(result_ids))]
    refs = set().union(*(refs_by_question[q["id"]] for q in selected))
    index = record_index(case, index)
    contexts = review_execution_context(case, refs, index.available)
    sources = []
    for context in contexts:
        result = protocol(case)["results"][context["result_id"]]
        observed = result.get("structured_results", {}).get("parsed_observables", {})
        fields = {k: observed[k] for k in ("method", "state", "environment", "charge", "multiplicity",
                  "computed_geometry_xyz", "geometry", "density_scope", "molecular_context")
                  if k in observed and json_bytes(observed[k]) <= 512}
        sources.append({**context, "observed_conditions": fields,
                        "method_steps": method_steps(result),
                        "conditions_path": "/structured_results/parsed_observables"})
    return {"questions": [{"id": q["id"], "source_refs": sorted(refs_by_question[q["id"]])} for q in selected],
            "sources": sources,
            "policy": "Shared sources are listed once. Requested controls are not verified controls. "
                      "Compare recorded methods, geometries and effective parameters before interpreting differences. "
                      "Absent conditions are unknown; input geometry does not imply unchanged computed geometry. "
                      "Method labels retain their recorded source; submitted-input provenance is not convergence. "
                      "Read method_steps.path for full details or deferred rows. "
                      "Parent questions may inspect child results without inheriting their answers. "
                      "No source counts or arithmetic automatically change confidence."}


def question_source_refs(view):
    return {ref for row in view["questions"] for ref in row["source_refs"]}
