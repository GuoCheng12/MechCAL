"""Short request-local citations for Planner updates and their scoped repairs."""

import json
import re
from copy import deepcopy

from mechcal.agents.evidence_references import _rewrite
from mechcal.runtime.retry import RejectedResponseError, complete_recorded_json


class PlannerReferences:
    def __init__(self, case):
        self.strict_outputs = case.runtime.get("planner_strict_outputs", False)
        self.compact_context = case.runtime.get("scoped_evidence_updates", False)
        self.wire_responses = {}
        self.rejected_wire_contracts = {}
        evidence = [e.evidence_id for e in case.evidence_ledger.items]
        state = case.runtime.get("question_loop", {})
        results = [r["result_id"] for r in state.get("results", [])]
        results += [r["result_id"] for r in state.get("pending_results", [])]
        self.aliases = {}
        reserved = set(evidence + results)
        for prefix, records in (("E", evidence), ("X", results)):
            ordinal = 1
            for record in dict.fromkeys(records):
                # Legacy short IDs and input:smiles do not need another name.
                if not re.fullmatch(r"execution:[0-9a-f]{20}(?::E\d+)?", record):
                    continue
                while f"{prefix}{ordinal:03d}" in reserved:
                    ordinal += 1
                alias = f"{prefix}{ordinal:03d}"
                reserved.add(alias)
                self.aliases[alias] = record
        if case.runtime.get("planner_delta_context"):
            # Literal names in stored prose must not become citation aliases.
            reserved.update(re.findall(r"\bFILE\d+\b", case.model_dump_json()))
            self.aliases.update(artifact_aliases(
                [a.artifact_id for a in case.artifact_manifest.artifacts], reserved=set(self.aliases) | reserved,
            ))

    def encode(self, value):
        return _rewrite(value, {full: short for short, full in self.aliases.items()})

    def restore(self, value):
        # Unknown citations remain unchanged for the typed field validators.
        # Never fuzzy-match a typo to evidence or mutate persisted records.
        return _rewrite(value, self.aliases)

    def complete(self, client, *, payload, schema_feedback=None, **kwargs):
        feedback = schema_feedback
        if feedback:
            try:
                feedback = json.dumps(self.encode(json.loads(feedback)), ensure_ascii=False)
            except json.JSONDecodeError:
                feedback = self.encode(feedback)
        payload = self.encode(payload)
        contract = None
        if self.strict_outputs:
            from mechcal.runtime.structured_output import PlannerOutputContract
            from mechcal.runtime.prompts import load_prompt
            contract = PlannerOutputContract(payload)
            kwargs["response_format"] = contract.response_format()
            kwargs["system_prompt"] += "\n" + load_prompt("planner_structured_output.md")
            feedback = self.wire_feedback(feedback)
        try:
            response = complete_recorded_json(
                client, payload=payload, schema_feedback=feedback,
                compact_context=self.compact_context, **kwargs,
            )
            if contract is not None:
                decoded = self.decode_response(contract, response)
                if kwargs.get("event_sink"):
                    kwargs["event_sink"]({"kind": "planner_response_decoded",
                                          "contract": "planner_response_v1", "response": decoded})
                response = decoded
        except RejectedResponseError as exc:
            raise RejectedResponseError(str(exc), self.restore(exc.response), issues=exc.issues) from exc
        return self.restore(response)

    def decode_response(self, contract, response):
        from jsonschema import ValidationError
        try:
            decoded = contract.decode(response)
        except (ValueError, ValidationError) as exc:
            from mechcal.runtime.schema_diagnostics import format_issues, schema_issues
            issues = schema_issues(exc) if isinstance(exc, ValidationError) else []
            # Keep the exact wire schema for bounded edits to a rejected draft.
            # Invalid JSON or an ambiguous envelope still requires regeneration.
            envelope = response.get("response") if isinstance(response, dict) else None
            if (issues and isinstance(envelope, dict) and envelope.get("kind") == "decision"
                    and all(issue["path"][:2] == ["response", "data"] for issue in issues)):
                self.rejected_wire_contracts[json.dumps(self.restore(response), sort_keys=True)] = contract
            detail = format_issues(issues) if issues else str(exc)
            raise RejectedResponseError(
                f"Strict Planner response rejected: {detail}",
                json.dumps(self.restore(response), ensure_ascii=False), issues=issues,
            ) from exc
        self.wire_responses[json.dumps(decoded, sort_keys=True)] = deepcopy(response)
        return decoded

    def rejected_contract(self, response):
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return None
        return self.rejected_wire_contracts.get(json.dumps(response, sort_keys=True))

    def wire_feedback(self, feedback):
        if not feedback:
            return feedback
        try:
            context = json.loads(feedback)
        except json.JSONDecodeError:
            return feedback
        if not isinstance(context, dict):
            return feedback
        rejected = context.get("rejected_response")
        if isinstance(rejected, dict):
            wire = self.wire_responses.get(json.dumps(rejected, sort_keys=True))
            if wire is not None:
                context["rejected_response"] = wire
                context["response_representation"] = "planner_response_v1_wire"
                context["issue_paths_relative_to"] = "/response/data"
            elif "response" not in rejected:
                # A normalized internal snapshot may differ from the received data.
                # Never teach its shape as a replacement for the wire envelope.
                context.pop("rejected_response")
                context["response_representation"] = "issues_only"
        context["return_contract"] = "Return the complete planner_response_v1 response envelope from the supplied schema."
        return json.dumps(context, ensure_ascii=False)


def artifact_aliases(ids, *, reserved=()):
    """Transport-only artifact handles, using the same reversible citation convention."""
    reserved = set(reserved) | set(ids)
    aliases, ordinal = {}, 1
    for rid in dict.fromkeys(ids):
        if len(rid) < 32:
            continue
        while f"FILE{ordinal:03d}" in reserved:
            ordinal += 1
        alias = f"FILE{ordinal:03d}"
        reserved.add(alias)
        aliases[alias] = rid
    return aliases
