"""Evidence-triggered proposals and explicit Planner receipts; no scientific scoring."""

from copy import deepcopy
import hashlib
import json

from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.result_contract import disposition_contract, operational_evidence_refs
from mechcal.runtime.retry import RequestValidationError


def _state(case):
    return case.runtime.get("question_loop", {}).get("candidate_expansion", {})


def _facts(result):
    # Allowlist the computational envelope. Never copy task, expected outcome,
    # investigation plan, Worker interpretation, review, or mechanism annotations.
    data = result.get("structured_results", {})
    return {"result_id": result["result_id"], "capability_id": result["capability_id"],
            "status": result["status"], "evidence_refs": result.get("evidence_refs", []),
            "computed": {k: data[k] for k in (
                "parsed_observables", "method_provenance", "limitations", "missing_deliverables",
                "molecular_context", "backend", "metrics") if k in data},
            "failure": result.get("failure", {})}


def _signature(result):
    # Exact duplicate detection only. Different conditions may be scientifically relevant.
    facts = _facts(result)
    return hashlib.sha256(json.dumps({"capability_id": facts["capability_id"],
        "parsed_observables": facts["computed"].get("parsed_observables", {}),
        "metrics": facts["computed"].get("metrics", {}),
        "source": result.get("resolved_source_artifact_id"),
        "parameters": result.get("requested_parameters", {}),
        "molecular_context": facts["computed"].get("molecular_context"),
        "methods": [{k: method[k] for k in ("step_id", "executed_method_label", "input_sha256")
                     if k in method} for method in facts["computed"].get("method_provenance", [])]},
        sort_keys=True).encode()).hexdigest()


def new_observations(case):
    seen = set(_state(case).get("seen_observations", []))
    fresh = []
    for result in case.runtime.get("question_loop", {}).get("pending_results", []):
        refs = disposition_contract(result, case)["observation_refs"]
        signature = _signature(result)
        preparation = (result.get("investigation_plan") or {}).get("purpose") == "preparation"
        structured = result.get("structured_results", {})
        has_values = bool(structured.get("parsed_observables") or structured.get("metrics"))
        # Legacy evidence families also call topology/dimer observations "geometry".
        # Use the recorded investigation purpose, not that overloaded category.
        if not preparation and has_values and refs and signature not in seen:
            fresh.append({"result_id": result["result_id"], "signature": signature,
                          "evidence_refs": refs})
            seen.add(signature)
    return fresh


def proposal_payload(case, fresh):
    from mechcal.agents.observation_view import observation_view

    mode = case.runtime.get("candidate_expansion_mode", "off")
    if mode not in {"reflection", "independent"}:
        raise ValueError("Candidate proposals require an enabled mode")
    s = case.runtime.get("question_loop", {})
    facts = [_facts(r) for r in s.get("results", [])]
    payload = {"stage": "candidate_expansion", "mode": mode,
               "smiles": case.input.smiles, "mechanism_pool": list(MECHANISM_POOL),
               "new_result_ids": [r["result_id"] for r in fresh],
               "new_observation_refs": [ref for r in fresh for ref in r["evidence_refs"]],
               "computations": observation_view(facts),
               "operational_only_evidence_refs": sorted(operational_evidence_refs(case)),
               "scope": "Computed observations and limitations, not experimental mechanism validation. "
                        "Failures are missing observations, not counterevidence."}
    if mode == "reflection":
        payload["current_judgment"] = {
            "portfolio": case.portfolio.model_dump(mode="json"),
            "questions": list(s.get("questions", {}).values()),
            "assessment": s.get("assimilation", {}).get("decision_assessment"),
            "pending_plans": [r.get("investigation_plan") for r in s.get("pending_results", [])],
        }
    return payload


def validate_proposals(proposal, payload):
    fresh_refs = set(payload["new_observation_refs"])
    forbidden = set(payload["operational_only_evidence_refs"])
    refs = {ref for r in payload["computations"] for ref in r["evidence_refs"]} - forbidden
    names = set()
    issues = []
    for index, item in enumerate(proposal.proposals):
        def issue(code, message, **details):
            issues.append({"path": f"proposals.{index}", "code": code,
                           "proposal_index": index, "repair_paths": [["proposals"], ["rationale"]],
                           "message": f"Proposal at zero-based index {index}: {message}", **details})

        if (not set(item.related_labels) <= set(MECHANISM_POOL)
                or len(set(item.related_labels)) != len(item.related_labels)):
            issue("candidate_labels", "Use distinct mechanism-pool labels for this explanation.")
        invalid = sorted(set(item.evidence_refs) - refs)
        if invalid:
            issue("candidate_unknown_source", "Citations must refer to delivered observations from this case.",
                  invalid_refs=invalid, available_observation_refs=sorted(refs))
        if not set(item.evidence_refs) & fresh_refs:
            issue("candidate_no_new_observation",
                  "This explanation cites no newly delivered observation. Reinterpret it using new data only "
                  "if scientifically justified, or explicitly withdraw it by omitting it from proposals. "
                  "Old observations may remain valid, but alone do not meet this new-candidate contract. "
                  "Do not add a new citation merely to pass validation.",
                  cited_refs=item.evidence_refs, required_new_observation_refs=sorted(fresh_refs))
        if item.hypothesis.casefold() in names:
            issue("candidate_duplicate", "Do not duplicate an explanation within the same batch.")
        names.add(item.hypothesis.casefold())
    if issues:
        raise RequestValidationError(issues)


def admit(case, fresh, proposal):
    from mechcal.runtime.questions import replace_state

    if not fresh:
        raise ValueError("No new observations to expand")
    s = deepcopy(_state(case))
    if s.get("pending"):
        raise ValueError("Handle existing candidates before generating another batch")
    if set(s.get("seen_observations", [])) & {r["signature"] for r in fresh}:
        raise ValueError("This evidence batch was already processed")
    validate_proposals(proposal, proposal_payload(case, fresh))
    batches = s.get("batches", [])
    batch_id = f"B{len(batches) + 1:03d}"
    candidates = [{"candidate_id": f"{batch_id}:C{i + 1:02d}", **p.model_dump(mode="json")}
                  for i, p in enumerate(proposal.proposals)]
    batch = {"batch_id": batch_id, "mode": case.runtime["candidate_expansion_mode"],
             "round_index": case.runtime["question_loop"]["round_index"],
             "previous_top3": [h.name for h in case.portfolio.sorted_hypotheses()[:3]],
             "previous_question_ids": list(case.runtime["question_loop"].get("questions", {})),
             "new_observations": deepcopy(fresh), "candidates": candidates,
             "rationale": proposal.rationale}
    s.update(batches=[*batches, batch], pending=candidates,
             seen_observations=[*s.get("seen_observations", []), *[r["signature"] for r in fresh]])
    return replace_state(case, phase="assimilate", candidate_expansion=s)


def candidate_context(case):
    if case.runtime.get("candidate_expansion_mode", "off") == "off":
        return {"enabled": False}
    s = _state(case)
    return {"enabled": True, "pending": deepcopy(s.get("pending", [])),
            "history": deepcopy(s.get("responses", [])),
            "notice": "Proposals are not evidence. Handle each pending candidate exactly once; "
                      "investigate/merge link actual questions. No forced tool call or ranking change."}


def validate_responses(case, update):
    pending = {p["candidate_id"]: p for p in _state(case).get("pending", [])}
    ids = [r.candidate_id for r in update.candidate_responses]
    issues = []
    repair_paths = [["candidate_responses"]]
    if len(ids) != len(set(ids)) or set(ids) != set(pending):
        issues.append("Return exactly one candidate_response for each pending candidate: " + str(sorted(pending)))
    questions = {q.question_id: q for q in update.questions}
    refs = {e.evidence_id for e in case.evidence_ledger.items} | {"input:smiles"}
    refs -= operational_evidence_refs(case)
    for reply in update.candidate_responses:
        if reply.candidate_id not in pending:
            continue
        qids = set(reply.question_ids)
        if len(qids) != len(reply.question_ids) or not qids <= questions.keys() or not set(reply.evidence_refs) <= refs:
            issues.append(f"{reply.candidate_id}: cite existing questions and observations, not proposal IDs.")
            if not qids <= questions.keys():
                repair_paths.append(["questions"])
            continue
        if reply.decision in {"investigate", "merge"}:
            labels = set(pending[reply.candidate_id]["related_labels"])
            if not qids or not any(labels & set(questions[q].related_labels) for q in qids):
                issues.append(f"{reply.candidate_id}: investigate/merge needs a question covering this explanation.")
                repair_paths.append(["questions"])
        if reply.decision == "investigate" and qids and all(
                questions[q].answer_state not in {"open", "partial"} for q in qids):
            issues.append(f"{reply.candidate_id}: investigate must link an open or partial question.")
            repair_paths.append(["questions"])
    if issues:
        raise RequestValidationError([{"path": "candidate_responses", "code": "candidate_receipt",
            "repair_paths": [list(p) for p in dict.fromkeys(tuple(p) for p in repair_paths)],
            "message": " ".join(issues)}])


def preserve_responses(case, update):
    if case.runtime.get("candidate_expansion_mode", "off") == "off":
        return {}
    s = deepcopy(_state(case))
    pending = {p["candidate_id"]: p for p in s.get("pending", [])}
    s["responses"] = [*s.get("responses", []), *[
        {"revision": update.revision, "candidate": pending[r.candidate_id],
         "response": r.model_dump(mode="json")} for r in update.candidate_responses]]
    s["pending"] = []
    return {"candidate_expansion": s}
