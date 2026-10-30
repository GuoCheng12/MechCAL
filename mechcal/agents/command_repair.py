"""Use one bounded draft-edit protocol for Planner actions and judgments."""

import json

from mechcal.runtime.judgment_draft import JudgmentDraft
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.retry import RejectedResponseError


class CommandRepairProducer:
    def __init__(self, *, complete, refs, model, payload, prompt, event_sink=None,
                 validation_history=None):
        self.complete = complete
        self.refs = refs
        self.model = model
        self.payload = payload
        self.prompt = prompt
        self.event_sink = event_sink
        self.validation_history = validation_history

    def __call__(self, feedback):
        draft = wire_contract = None
        payload, prompt = self.payload, self.prompt
        if feedback is not None:
            correction = json.loads(feedback)
            if self.validation_history is not None:
                correction["validation_history"] = self.validation_history
            wire_contract = self.refs.rejected_contract(correction.get("rejected_response"))
            if wire_contract is not None and isinstance(correction["rejected_response"], str):
                correction["rejected_response"] = json.loads(correction["rejected_response"])
            if isinstance(correction.get("rejected_response"), dict):
                draft = JudgmentDraft(payload["case_id"], payload["revision"], payload["stage"],
                    self.model, correction,
                    schema=self.refs.restore(wire_contract.schema) if wire_contract else None)
                payload = {"stage": "planner_judgment_repair", "case_id": payload["case_id"],
                           "revision": payload["revision"], **draft.payload()}
                command = draft.original.get("final_action") or draft.original
                requests = command.get("dispatch_requests") if isinstance(command, dict) else None
                payload["repair_capability_ids"] = list(dict.fromkeys(
                    request["capability_id"] for request in (requests if isinstance(requests, list) else [])
                    if isinstance(request, dict) and isinstance(request.get("capability_id"), str)))
                if draft.requires_context:
                    payload["scientific_context"] = self.payload
                    payload["candidate_command"] = draft.original
                prompt = load_prompt("planner_judgment_repair.md")
                self.emit({"kind": "judgment_draft", "draft_id": draft.draft_id,
                           "response": draft.original, "targets": payload["targets"]})
                correction.pop("rejected_response")
            feedback = json.dumps(correction)
        try:
            response = self.complete(payload=payload, system_prompt=prompt,
                                     schema_feedback=feedback, event_sink=self.event_sink)
        except RejectedResponseError as exc:
            if draft is None:
                raise
            raise draft.rejected_correction(exc, exc.response) from exc
        if draft is None:
            return response
        candidate = draft.apply(response)
        if wire_contract is not None:
            candidate = self.refs.restore(self.refs.decode_response(wire_contract, self.refs.encode(candidate)))
        self.emit({"kind": "draft_repair_applied", "draft_id": draft.draft_id,
                   "edits": response, "response": candidate})
        return candidate

    def emit(self, event):
        if self.event_sink is not None:
            self.event_sink({**event, "scientific_evidence": False})
