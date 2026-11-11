"""Separate scientific review and exploration agents with case-owned sessions."""

from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar


def _session(state, case_id, agent_id, *, exploration=False):
    if state.get("agent_id", agent_id) != agent_id or state.get("case_id", case_id) != case_id:
        raise RuntimeError("Review session belongs to a different agent or case")
    # Older checkpoints have no agent_id. Their existing exploration marker
    # still prevents an exploration session from becoming a scientific review.
    if (state.get("mode") == "exploration") != exploration:
        raise RuntimeError("Review session has the wrong memory scope")
    return {**deepcopy(state), "agent_id": agent_id, "case_id": case_id}


def _record_call(store, agent_id, case_id, role):
    store.append_jsonl("history/review_agent_calls.jsonl", {
        "agent_id": agent_id, "case_id": case_id, "role": role,
    })


@dataclass
class ScientificReviewerAgent:
    llm_client: object
    agent_id: ClassVar[str] = "scientific_reviewer"
    roles: ClassVar[frozenset[str]] = frozenset({
        "coverage_reviewer", "reviewer", "final_reviewer", "impact_reviewer",
    })

    def review(self, case, payload, store, *, state, role, **kwargs):
        from mechcal.agents.command_worker import scoped_commentary_turn

        if role not in self.roles:
            raise ValueError("Scientific Reviewer cannot perform an exploration task")
        state = _session(state, case.case_id, self.agent_id)
        _record_call(store, self.agent_id, case.case_id, role)
        return scoped_commentary_turn(self.llm_client, case, payload, store,
                                      state=state, role=role, **kwargs)

    def review_text(self, case, payload, store, *, role):
        from mechcal.agents.command_client import ask_reviewer_text

        if role not in self.roles - {"impact_reviewer"}:
            raise ValueError("Unsupported plain-text scientific review task")
        _record_call(store, self.agent_id, case.case_id, role)
        return ask_reviewer_text(self.llm_client, payload, store, role=role)


@dataclass
class ExplorationAgent:
    llm_client: object
    agent_id: ClassVar[str] = "exploration_agent"

    def fact_view(self, case, memory):
        from mechcal.runtime.command_exploration_memory import project

        if memory.get("case_id") != case.case_id:
            raise RuntimeError("Exploration memory belongs to a different case")
        return project(case, memory)

    def review(self, case, payload, store, *, state, **kwargs):
        from mechcal.agents.command_worker import scoped_commentary_turn

        state = _session(state, case.case_id, self.agent_id, exploration=True)
        _record_call(store, self.agent_id, case.case_id, "exploration_reviewer")
        return scoped_commentary_turn(self.llm_client, case, payload, store,
            state=state, role="exploration_reviewer", prompt_name="reviewer_exploration.md", **kwargs)
