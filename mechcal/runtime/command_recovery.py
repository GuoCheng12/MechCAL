"""Bound model-mediated delivery, independently of scientific round limits."""

from pydantic import BaseModel, ConfigDict, Field

from mechcal.runtime.command_memory import digest
from mechcal.runtime.command_operations import check_owner, protocol, put


class DeliveryRecovery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    source_basis: str
    model_turns: int = Field(default=0, ge=0)
    max_model_turns: int = Field(default=12, ge=1)
    turn_starts: list[dict] = Field(default_factory=list)


class DeliveryRecoveryLimit(RuntimeError):
    pass


def state(case):
    check_owner(case)
    saved = protocol(case).get("delivery_recovery")
    if saved is None:
        return None
    value = DeliveryRecovery.model_validate(saved)
    if value.case_id != case.case_id:
        raise RuntimeError("Delivery recovery belongs to another case")
    return value


def begin(case, *, max_model_turns=12):
    """Only new immutable calculation records begin a new delivery episode."""
    current = state(case)
    basis = digest(list(protocol(case).get("results", {})))
    if current is None or current.source_basis != basis:
        current = DeliveryRecovery(case_id=case.case_id, source_basis=basis,
                                   max_model_turns=max_model_turns)
    return put(case, delivery_recovery=current.model_dump(mode="json"))


def before_model_turn(case, *, max_model_turns=12):
    p = protocol(case)
    if not p.get("final_draft") or p.get("delivery_rounds") != p.get("rounds"):
        return case
    current = state(begin(case, max_model_turns=max_model_turns))
    if current.model_turns >= current.max_model_turns:
        raise DeliveryRecoveryLimit(
            f"Delivery used {current.model_turns} model turns without submitting. "
            "Calculations, draft and review records are retained; inspect the unresolved operation before continuing.")
    current.model_turns += 1
    from mechcal.runtime.command_delivery import delivery_status
    snapshot = delivery_status(case)
    current.turn_starts.append({"turn": current.model_turns, "phase": p["phase"],
        "role": p.get("review_role", "reviewer") if p["phase"] == "review" else "planner",
        "preceding_operation": (p.get("outcome") or {}).get("status"),
        "draft_id": snapshot["draft_id"], "open_fact_issues": snapshot["open_fact_issues"],
        "exact_draft_reviewed": snapshot["exact_draft_reviewed"]})
    return put(case, delivery_recovery=current.model_dump(mode="json"))


def view(case, *, index=None):
    current = state(case)
    if current is None:
        return None
    from mechcal.runtime.command_delivery import delivery_status
    snapshot = delivery_status(case, index=index)
    previous = current.turn_starts[-1] if current.turn_starts else {}
    return {"model_turns_used": current.model_turns,
            "model_turns_remaining": max(0, current.max_model_turns - current.model_turns),
            "delivery_status": snapshot,
            "progress_since_last_call": {"draft_changed": previous.get("draft_id") != snapshot["draft_id"],
                "fact_issues_removed": sorted(set(previous.get("open_fact_issues", [])) - set(snapshot["open_fact_issues"])),
                "exact_draft_reviewed": snapshot["exact_draft_reviewed"]},
            "policy": "Delivery turns include Planner and Reviewer. Reads, new draft IDs and review retries "
                      "do not reset this allowance. Scientific calculations have a separate allowance."}
