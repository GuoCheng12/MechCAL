# Independent Review Agents

The command workflow has two explicit agent components, not two modes on the
same reviewer component. They may use the same model and provider. Independence
means separate invocation, client instances, session ownership and memory scope;
it does not require a separate process or a different backbone.

| Component | Responsibility | Visible history | Published output |
| --- | --- | --- | --- |
| `ScientificReviewerAgent` | Check claims, evidence links and the consequences of new observations | Scoped claims, calculation facts, earlier review receipts and unresolved issues | Review feedback and optional experiments |
| `ExplorationAgent` | Propose useful competing explanations and computable comparisons | Calculator facts, exact execution parameters, its own notes, proposals and execution receipts | Optional experiment proposals, not private notes |

Planner retains scientific ranking and execution authority. Neither review
agent executes calculations or writes mechanism rankings. Coverage, impact,
incremental review and final review remain subtasks of Scientific Reviewer.
Exploration remains event-driven; this split adds no model calls or triggers.
It does not alter tools, rubric, judge, inference settings or scientific rules.

## Message and Memory Boundaries

```text
                         calculation records
                           /            \
              scoped claims + facts    facts-only projection
                         /                \
              Scientific Reviewer      Exploration Agent
               private review state    private research notes
                         \                /
                      public feedback/proposals
                                  |
                               Planner
```

Both agents reuse the same read-only retrieval implementation. The caller grants
the scope. Exploration receives a new, whitelisted fact view, never the Planner
case object as its read source. Planner judgments, Worker explanations and
Scientific Reviewer opinions are excluded. Sharing a transport helper is not
sharing a conversation.

Each request is assembled explicitly; the model has no implicit memory of an
earlier API request. Case checkpoints persist private state and retrieval
receipts. Pending review sessions are bound to `agent_id` and `case_id` before a
model call. Cross-agent or cross-case sessions are rejected. A legacy session
without `agent_id` can bind to its original scope, but an exploration-mode
session cannot bind to Scientific Reviewer.

For checkpoint and analysis compatibility, existing persisted names remain:

- Exploration: `reviewer_exploration`, `private:reviewer_exploration` and the
  historical request role `exploration_reviewer`.
- Scientific review: `reviewer_session`, `impact_session`, `review_memory` and
  versioned review receipts.

These names do not indicate a shared agent instance. Local
`history/review_agent_calls.jsonl` records the explicit component, case and
subtask for each invocation. Raw requests and model usage remain in their
existing logs. Agent instances themselves do not hold case history, so reusing
an orchestrator for another case cannot reuse a private in-memory conversation.

## Configuration and Custom Launchers

`default_deps` and `mcp_deps` construct both components with distinct clients and
the existing reviewer settings. The installed CLI assigns both the requested
native-command model settings, using separate clients. The command-workflow
configuration fingerprint includes both components' model settings, excluding
credentials. Legacy non-command workflows retain their existing dependencies.

Custom command launchers must explicitly configure
`deps.scientific_reviewer.llm_client` and `deps.exploration_agent.llm_client`.
Overriding `mechanism_agenda` or `photophysics_arbiter` no longer configures these
command agents. Missing explicit dependencies fail configuration validation;
the runtime does not silently borrow another component's client. Frozen builds
and running experiments are not modified by this change. Normal fingerprint
checks still apply to resuming runs across code changes.

## Validation Scope

Offline tests cover client/settings separation in local and MCP factories,
installed CLI wiring, configuration fingerprints, session ownership, legacy
session binding, checkpoint continuation and actual routing to separate clients.
Existing projection tests reject private-memory reads and verify that changing
Planner opinions does not change Exploration input. Published proposals remain
visible to Planner, while private notes do not.

No online scientific performance or token-saving claim follows from this split.
Input-size optimization is a separate change described in
[Task memory and message transmission](context-transmission.md).

On 2026-11-12, the 15 new independence/wiring tests and 155 existing targeted
tests passed. The full suite then passed 3,619 tests with four skips and one
existing dependency-deprecation warning in 365.99 seconds. `git diff --check`
passed. No online experiment, merge or push was performed for this split.
