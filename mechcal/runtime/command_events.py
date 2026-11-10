"""Small runtime events shared by monitoring and acceptance statistics."""

import json
import time

from pydantic import BaseModel, Field

from mechcal.runtime.command_memory import digest


class CommandEvent(BaseModel):
    event_id: str
    timestamp: float = Field(default_factory=time.time)
    kind: str
    role: str = "runtime"
    status: str
    details: dict = Field(default_factory=dict)


def write_event(store, event):
    event = CommandEvent.model_validate(event).model_dump(mode="json")
    path = "events/" + digest(event["event_id"]) + ".json"
    existing = store.root / path
    if existing.exists():
        saved = json.loads(existing.read_text())
        if {k: v for k, v in saved.items() if k != "timestamp"} != {k: v for k, v in event.items() if k != "timestamp"}:
            raise RuntimeError("Runtime event conflicts with its persisted identity")
    else:
        store.write_json(path, event)


def track_state(case, previous):
    p = case.runtime["command_protocol"]
    old = (previous or {}).get("runtime", {}).get("command_protocol", {})
    now = time.time()
    clock = dict(old.get("clock", {}))
    elapsed = clock.get("elapsed_seconds", 0.0)
    if clock and previous.get("status") == "running":
        elapsed += max(0.0, now - clock["updated_at"])
    clock.update(started_at=clock.get("started_at", now), updated_at=now,
                 elapsed_seconds=elapsed, stopped_at=None if case.status == "running" else clock.get("stopped_at") or now)
    state = {"phase": p["phase"], "status": case.status, "computations": p.get("rounds", 0),
             "judgment_updates": p.get("revision", 0), "turn": p.get("turn", 0),
             "interruption": p.get("interruption")}
    last = old.get("state_event")
    sequence = old.get("event_sequence", 0)
    if last is None or last["details"] != state:
        sequence += 1
        last = CommandEvent(event_id=f"state:{sequence}", timestamp=now, kind="state",
                            status=case.status, details=state).model_dump(mode="json")
    return case.touch(runtime={**case.runtime, "command_protocol": {
        **p, "clock": clock, "event_sequence": sequence, "state_event": last}})


def stop_reason(root):
    path = root / "STOP_REQUESTED"
    if not path.exists() or not path.read_text().strip():
        return {"kind": "user_stop", "requires_explicit_resume": True}
    try:
        reason = json.loads(path.read_text())
    except ValueError:
        reason = None
    if isinstance(reason, dict) and reason.get("reason") in {"observed_tokens", "model_calls"}:
        return {"kind": "acceptance_budget", **reason, "requires_explicit_resume": True}
    return {"kind": "user_stop", "requires_explicit_resume": True}


def read_issue_count(outcome):
    """Count each call's issues once, not again on its individual record rows."""
    if "call_results" in outcome:
        return sum(read_issue_count(row) for row in outcome["call_results"].values())
    return len(outcome.get("issues", []))
