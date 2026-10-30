"""Self-contained Planner decisions with optional bounded context reads."""

import hashlib
import time

from pydantic import ValidationError

from mechcal.runtime.case_memory import serialized
from mechcal.runtime.delta_context import ITERATIVE_STAGES, LOOKUP_STAGES, decision_context, delta_prompt, lookup_context
from mechcal.runtime.retry import RejectedResponseError
from mechcal.schemas.memory import ContextNeeded, ContextReadPlan


QUERY_ACTIONS = {"search_case_records", "read_case_records", "find_compatible_artifacts", "describe_tools"}


def read_needs_review(response):
    """A directory, deferred field or fragment needs review, just like a read error."""
    if response.get("error"):
        return True
    for row in response.get("records", []):
        record = row.get("record", {})
        if row.get("error") or record.get("error") or record.get("field_errors"):
            return True
        if ("json_fragment" in row or record.get("view") == "outline"
                or record.get("deferred_paths") or record.get("context_not_loaded")):
            return True
        if record.get("view") in {"brief", "findings"} and record.get("next_offset") is not None:
            return True
        if record.get("view") != "brief" and any(isinstance(value, dict) and set(value) == {"error"}
                                                 for value in record.get("selected_fields", {}).values()):
            return True
    return False


def complete_delta(session, client, references, *, payload, system_prompt, schema_feedback=None,
                   event_sink=None, discovery=None):
    from mechcal.agents.planner import _unwrap_planner_response

    key = hashlib.sha256(serialized(payload).encode()).hexdigest()
    saved = session.delta_prepared.get(key)
    direct = payload["stage"] in ITERATIVE_STAGES
    preparing = (payload["stage"] in LOOKUP_STAGES and not direct and saved is None) or key in session.delta_lookup_pending
    focus = list(saved or [])
    if payload["stage"] == "planner_judgment_repair" and discovery is not None:
        # Repair has a new payload hash, not a new tool-discovery session.
        # Preserve the definitions needed by the rejected requests only.
        focus = list(dict.fromkeys([*focus, *[
            cid for cid in payload.get("repair_capability_ids", [])
            if cid in discovery.state.loaded_tools]]))
        discovery.visible_ids.update(focus)
    context_goal = session.delta_lookup_pending.get(key)
    if saved and (discovery is None or set(saved) - set(discovery.state.loaded_tools)):
        # Discovery invalidates stale definition versions. Re-prepare without
        # resetting either the memory or tool-description allowance.
        preparing = True
        context_goal = "Reload changed or unavailable focused tool definitions before deciding."
    # A source snapshot is not a conversation transcript. Register it once, without
    # query replies/budgets, so repeated reads retain their identity across retries.
    source = discovery.augment_payload(payload) if discovery is not None else dict(payload)
    for field in ("retrieved_memory", "memory_access", "tool_discovery"):
        source.pop(field, None)
    if discovery is not None:
        source.pop("capability_cards", None)  # Mutable descriptions come from the registry.
    context_id = session.index.register_context(source)
    while True:
        full = session.augment(payload)
        if discovery is not None:
            full = discovery.augment_payload(full)
        active = (lookup_context(full, context_id, focus_ids=focus) if preparing else
                  decision_context(full, context_id, focus_ids=focus))
        if preparing and context_goal:
            active["lookup_goal"] = context_goal
        active["context_access"] = {
            "lookup_active": preparing,
            "reopens_remaining": max(0, 2 - session.delta_reopens.get(key, 0)),
        }
        prompt = delta_prompt(active["stage"]) or system_prompt
        usage = None

        def record(event):
            nonlocal usage
            if event.get("usage") is not None:
                usage = event["usage"]
            if event_sink:
                event_sink({**event, "context_mode": "planner_delta_v1", "lookup_only": preparing})

        session.emit("delta_request_context", context_record_id=context_id, lookup_only=preparing,
                     decision_stage=payload["stage"],
                     original_context_bytes=len(serialized(full).encode()),
                     working_context_bytes=len(serialized(active).encode()),
                     field_bytes={k: len(serialized(v).encode()) for k, v in active.items()})
        session.store.activity("Planner context lookup" if preparing else "Planner " + payload["stage"],
                               f"R{session.case.runtime['question_loop'].get('round_index', 0):03d}",
                               f"Memory {session.used}/{session.max_queries}; delta context; awaiting response")
        started = time.monotonic()
        response = references.complete(client, payload=active, system_prompt=prompt,
                                       schema_feedback=schema_feedback, event_sink=record)
        unwrapped = _unwrap_planner_response(response)
        action = unwrapped.get("action")
        session.emit("memory_model_call", operation=action if action in QUERY_ACTIONS | {"prepare_context", "need_context"}
                     else "decision", lookup_only=preparing, usage=usage,
                     latency_seconds=time.monotonic() - started, usage_is_duplicate_of_request_log=True)
        if action == "need_context":
            if preparing:
                raise RejectedResponseError("Already in context lookup", response)
            if session.delta_reopens.get(key, 0) >= 2:
                raise RejectedResponseError("Context reopening allowance exhausted (2 per decision)", response)
            context_goal = ContextNeeded.model_validate(unwrapped).reason
            session.delta_reopens[key] = session.delta_reopens.get(key, 0) + 1
            session.delta_lookup_pending[key] = context_goal
            session.emit("delta_context_reopened", context_key=key, count=session.delta_reopens[key], reason=context_goal)
            preparing = True
        elif action == "prepare_context":
            if not preparing and not direct:
                raise RejectedResponseError("prepare_context is valid only during context lookup", response)
            try:
                ready = ContextReadPlan.model_validate(unwrapped)
                if (not preparing and not ready.memory_queries and not ready.describe_capability_ids
                        and ready.focus_capability_ids == focus):
                    raise ValueError("Empty context read makes no progress; submit a decision or specify missing information")
                if len(set(ready.focus_capability_ids)) != len(ready.focus_capability_ids):
                    raise ValueError("focus_capability_ids must be distinct")
                if len(ready.memory_queries) > session.max_queries - session.used:
                    raise ValueError("Read plan exceeds the remaining memory query allowance")
                if ready.describe_capability_ids:
                    if discovery is None:
                        raise ValueError("Tool descriptions are available in action selection only")
                    discovery.query({"action": "describe_tools", "capability_ids": ready.describe_capability_ids})
                followups = []
                for query in ready.memory_queries:
                    found = session.query(query.model_dump(mode="json"))
                    if read_needs_review(found):
                        followups.append(found)
                loaded = set(discovery.state.loaded_tools) if discovery else set()
                if set(ready.focus_capability_ids) - loaded:
                    raise ValueError("Read tool definitions before selecting focus_capability_ids")
            except (ValidationError, ValueError) as exc:
                raise RejectedResponseError(str(exc), response) from exc
            focus = ready.focus_capability_ids
            if followups and (not direct or preparing):
                # The Planner must see incomplete reads before accepting or narrowing them.
                # An empty later prepare_context can deliberately proceed with known gaps.
                session.emit("delta_context_read_followup", context_key=key,
                             read_count=len(followups), queries_remaining=session.max_queries - session.used)
                continue
            if followups:
                # Incomplete reads are visible in the next self-contained decision.
                # The Planner can continue reading or decide within known limits.
                session.emit("delta_context_read_followup", context_key=key,
                             read_count=len(followups), queries_remaining=session.max_queries - session.used)
            if discovery is not None:
                discovery.visible_ids.update(focus)
            session.delta_prepared[key] = focus
            session.delta_lookup_pending.pop(key, None)
            session.emit("delta_context_ready", context_key=key, focus_capability_ids=focus)
            preparing = False
        elif action in QUERY_ACTIONS:
            if action == "describe_tools":
                if discovery is None:
                    raise RejectedResponseError("Tool descriptions are available in action selection only", response)
                try:
                    discovery.query(unwrapped)
                except ValueError as exc:
                    raise RejectedResponseError(str(exc), response) from exc
                # Newly requested definitions enter the working set, not every historical tool.
                focus = list(dict.fromkeys([*focus, *unwrapped["capability_ids"]]))
            else:
                session.query(unwrapped)
            # A late scientific-stage lookup is allowed and charged in full. Subsequent
            # retrievals return to the small lookup contract, never silently drop a query.
            if payload["stage"] in LOOKUP_STAGES and not direct:
                preparing = True
        elif preparing:
            raise RejectedResponseError("Context lookup cannot submit a scientific decision", response)
        else:
            return response
