from __future__ import annotations

import asyncio
import json

import httpx
import pytest

pytest.importorskip("chemgraph", reason="Install the optional pinned ChemGraph environment.")
pytest.importorskip("langchain_openai")

from chemgraph.graphs.multi_agent import construct_multi_agent_graph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from mechcal.baselines.chemgraph_mas.budget import (
    BudgetExceeded,
    BudgetLimits,
    SharedBudget,
)
from mechcal.baselines.chemgraph_mas.model import (
    BudgetedChatOpenAI,
    _chat_result_is_empty,
    _compact_tool_message_for_planner,
    _normalize_responses_content,
    _recover_post_tool_empty_response,
    _responses_empty_retry_kwargs,
    _responses_initial_request_kwargs,
)
from mechcal.baselines.chemgraph_mas.schemas import PhotoMechOutput
from mechcal.baselines.chemgraph_mas.subject import (
    ChemGraphMASSubject,
    _compact_aggregation_state,
)
from mechcal.baselines.chemgraph_mas.tools import ChemGraphToolHarness
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.llm import OpenAICompatibleSettings
from mechcal.cli.evaluate_chemgraph import _cache_within_budget


def test_budget_counts_roles_tools_and_tokens() -> None:
    budget = SharedBudget(
        BudgetLimits(
            max_model_calls=2,
            max_tool_calls=1,
            max_total_tokens=100,
            max_output_tokens_per_call=20,
        )
    )
    assert budget.reserve_model_call(role="planner", estimated_prompt_tokens=10) == 20
    budget.record_model_usage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    budget.reserve_tool_call(owner="macro", capability_id="macro.macro_structure_scan")
    with pytest.raises(BudgetExceeded):
        budget.reserve_tool_call(
            owner="macro",
            capability_id="macro.screen_rotor_rim_prior",
        )
    snapshot = budget.snapshot()
    assert snapshot["usage"]["model_calls_by_role"] == {"planner": 1}
    assert snapshot["usage"]["total_tokens"] == 15


def test_budget_rejects_actual_token_overrun() -> None:
    budget = SharedBudget(
        BudgetLimits(
            max_model_calls=2,
            max_tool_calls=1,
            max_total_tokens=100,
            max_output_tokens_per_call=20,
        )
    )
    budget.reserve_model_call(role="planner", estimated_prompt_tokens=10)
    with pytest.raises(BudgetExceeded):
        budget.record_model_usage(
            prompt_tokens=95,
            completion_tokens=10,
            total_tokens=105,
        )
    assert budget.snapshot()["usage"]["token_overrun"] == 5


def test_evaluation_cache_requires_actual_usage_within_budget() -> None:
    valid = {
        "budget": {
            "limits": {"max_total_tokens": 200_000},
            "usage": {"total_tokens": 199_999, "token_overrun": 0},
        }
    }
    assert _cache_within_budget(valid)

    overrun = json.loads(json.dumps(valid))
    overrun["budget"]["usage"]["total_tokens"] = 200_001
    overrun["budget"]["usage"]["token_overrun"] = 1
    assert not _cache_within_budget(overrun)


def test_output_schema_requires_three_distinct_ordered_pool_labels() -> None:
    predictions = []
    for rank, label in enumerate(MECHANISM_POOL[:3], start=1):
        predictions.append(
            {
                "label": label,
                "rank": rank,
                "confidence": 0.5,
                "claim_status": "candidate_requires_validation",
                "support_strength": "weak_or_proxy",
                "evidence": ["Finding: x. Warrant: y. Boundary: z."],
                "limitations": ["No experimental validation."],
            }
        )
    output = PhotoMechOutput.model_validate(
        {
            "final_summary": "Three bounded candidates.",
            "mechanism_predictions": predictions,
        }
    )
    assert [item.rank for item in output.mechanism_predictions] == [1, 2, 3]

    duplicate = json.loads(output.model_dump_json())
    duplicate["mechanism_predictions"][1]["label"] = duplicate[
        "mechanism_predictions"
    ][0]["label"]
    with pytest.raises(ValueError):
        PhotoMechOutput.model_validate(duplicate)


def test_tool_harness_exposes_only_two_local_routes(tmp_path) -> None:
    budget = SharedBudget(
        BudgetLimits(
            max_model_calls=5,
            max_tool_calls=3,
            max_total_tokens=1000,
            max_output_tokens_per_call=100,
        )
    )
    harness = ChemGraphToolHarness(
        case_id="test-case",
        public_case_id="Case 001",
        smiles="CCO",
        task="Rank mechanisms.",
        workspace=tmp_path,
        budget=budget,
        enable_amesp=False,
    )
    policy = harness.policy_snapshot()
    assert policy["exposed_langchain_tools"] == [
        "run_macro_analysis",
        "run_microscopic_analysis",
    ]
    assert policy["evidence_ledger_shared_with_agents"] is False
    assert policy["evidence_ledger_unit_count"] == 0
    assert "PubChem" in policy["forbidden_sources"]

    result = json.loads(
        harness.run_macro_analysis(
            "macro.macro_structure_scan",
            {},
        )
    )
    assert result["status"] == "success"
    assert result["capability_id"] == "macro.macro_structure_scan"
    assert harness.case_run.evidence_ledger.items == []


def test_upstream_graph_constructs_with_only_whitelisted_tools(tmp_path) -> None:
    budget = SharedBudget(
        BudgetLimits(
            max_model_calls=5,
            max_tool_calls=3,
            max_total_tokens=1000,
            max_output_tokens_per_call=100,
        )
    )
    harness = ChemGraphToolHarness(
        case_id="test-graph",
        public_case_id="Case 002",
        smiles="CCO",
        task="Rank mechanisms.",
        workspace=tmp_path,
        budget=budget,
        enable_amesp=False,
    )
    sync_http = httpx.Client(trust_env=False)
    async_http = httpx.AsyncClient(trust_env=False)
    try:
        llm = BudgetedChatOpenAI(
            shared_budget=budget,
            model="deepseek-v4-flash",
            api_key="offline-test",
            base_url="http://127.0.0.1:9/v1",
            temperature=0.0,
            max_tokens=100,
            max_retries=0,
            http_client=sync_http,
            http_async_client=async_http,
        )
        graph = construct_multi_agent_graph(
            llm,
            planner_prompt="[ROLE: PLANNER] offline construction test",
            executor_prompt="[ROLE: EXECUTOR] offline construction test",
            executor_tools=harness.langchain_tools(),
            structured_output=False,
            max_retries=0,
            max_task_retries=0,
            human_supervised=False,
        )
        node_names = set(graph.get_graph().nodes)
        assert {"Planner", "executor_subgraph"}.issubset(node_names)
        assert "ResponseAgent" not in node_names
    finally:
        sync_http.close()
        asyncio.run(async_http.aclose())


def test_responses_api_payload_omits_unsupported_sampling_controls(tmp_path) -> None:
    limits = BudgetLimits(
        max_model_calls=5,
        max_tool_calls=3,
        max_total_tokens=1000,
        max_output_tokens_per_call=100,
    )
    subject = ChemGraphMASSubject(
        settings=OpenAICompatibleSettings(
            base_url="https://provider.example/v1",
            model="gpt-5.5",
            api_key="offline-test",
            reasoning_effort="high",
            temperature=0.0,
            top_p=1.0,
        ),
        budget_limits=limits,
        run_base_dir=tmp_path,
        enable_amesp=False,
        use_responses_api=True,
    )
    llm, sync_http, async_http = subject._make_llm(SharedBudget(limits))
    try:
        payload = llm._get_request_payload(
            [HumanMessage(content="Return a bounded mechanism ranking.")]
        )
        assert "messages" not in payload
        assert isinstance(payload["input"], list)
        assert payload["reasoning"] == {"effort": "high"}
        assert payload["max_output_tokens"] == 100
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert payload["store"] is False
    finally:
        sync_http.close()
        asyncio.run(async_http.aclose())


def test_responses_content_blocks_are_normalized_for_chemgraph_parser() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "summary": []},
            {"type": "text", "text": '{"tasks":[],"next_step":"FINISH"}'},
        ],
        tool_calls=[
            {
                "name": "run_macro_analysis",
                "args": {"capability_id": "macro.macro_structure_scan"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    result = ChatResult(generations=[ChatGeneration(message=message)])

    _normalize_responses_content(result)

    assert result.generations[0].message.content == (
        '{"tasks":[],"next_step":"FINISH"}'
    )
    assert result.generations[0].message.tool_calls[0]["name"] == "run_macro_analysis"


def test_empty_responses_retry_uses_required_tools_or_json_mode() -> None:
    empty = ChatResult(generations=[ChatGeneration(message=AIMessage(content=[]))])
    assert _chat_result_is_empty(empty)
    assert _responses_empty_retry_kwargs({"tools": [{"type": "function"}]})[
        "tool_choice"
    ] == "required"
    assert _responses_empty_retry_kwargs({})["response_format"] == {
        "type": "json_object"
    }


def test_responses_initial_request_matches_chemgraph_role_contract() -> None:
    executor_messages = [
        HumanMessage(content="[ROLE: EXECUTOR] Run one assigned capability.")
    ]
    executor_kwargs = _responses_initial_request_kwargs(
        executor_messages,
        {"tools": [{"type": "function"}]},
    )
    assert executor_kwargs["tool_choice"] == "required"

    post_tool_kwargs = _responses_initial_request_kwargs(
        [
            *executor_messages,
            ToolMessage(content='{"status":"success"}', tool_call_id="call-1"),
        ],
        {"tools": [{"type": "function"}]},
    )
    assert "tool_choice" not in post_tool_kwargs

    planner_kwargs = _responses_initial_request_kwargs(
        [HumanMessage(content="[ROLE: PLANNER] Return a plan.")],
        {},
    )
    assert planner_kwargs["response_format"] == {"type": "json_object"}

    aggregator_kwargs = _responses_initial_request_kwargs(
        [HumanMessage(content="[ROLE: AGGREGATOR] Return the final ranking.")],
        {},
    )
    assert aggregator_kwargs["response_format"] == {"type": "json_object"}


def test_post_tool_empty_response_reuses_real_tool_result() -> None:
    empty = ChatResult(generations=[ChatGeneration(message=AIMessage(content=[]))])
    messages = [
        HumanMessage(content="Run the assigned capability."),
        ToolMessage(
            content='{"status":"success","summary":"bounded observation"}',
            tool_call_id="call-1",
        ),
    ]

    assert _recover_post_tool_empty_response(empty, messages)
    assert empty.generations[0].message.content == (
        '{"status":"success","summary":"bounded observation"}'
    )


def test_recovered_tool_result_keeps_metrics_and_drops_duplicate_payload() -> None:
    compact = json.loads(
        _compact_tool_message_for_planner(
            json.dumps(
                {
                    "status": "success",
                    "capability_id": "macro.screen_rotor_rim_prior",
                    "selected_route": "screen_rotor_rim_prior",
                    "summary": "Bounded rotor proxy.",
                    "structured_results": {
                        "selected_route": "screen_rotor_rim_prior",
                        "result_name": "screen_rotor_rim_prior_proxy",
                        "metrics": {"rotatable_bond_count": 5, "rim_prior": True},
                        "structural_prior": {"duplicate": "large payload"},
                    },
                    "evidence": [
                        {
                            "claim": "Rotor topology is present.",
                            "basis": "proxy",
                            "support": "not_applicable",
                            "status": "present",
                            "summary": "Bounded structural topology.",
                            "limits": ["Not direct photophysical evidence."],
                            "observable_tags": ["duplicate", "metadata"],
                        }
                    ],
                    "artifact_ids": ["not-needed-by-planner"],
                }
            )
        )
    )

    assert compact["structured_results"]["metrics"]["rotatable_bond_count"] == 5
    assert "structural_prior" not in compact["structured_results"]
    assert "observable_tags" not in compact["evidence"][0]
    assert "artifact_ids" not in compact


def test_aggregation_state_keeps_final_summary_and_compact_tool_results() -> None:
    state = {
        "messages": [
            HumanMessage(content="Full public query that should not be repeated."),
            AIMessage(content="Initial planner thought."),
            ToolMessage(
                content=json.dumps(
                    {
                        "status": "success",
                        "capability_id": "macro.screen_rotor_rim_prior",
                        "structured_results": {
                            "metrics": {"rim_prior": True},
                            "structural_prior": {"duplicate": "large"},
                        },
                        "artifact_ids": ["duplicate"],
                    }
                ),
                tool_call_id="call-1",
            ),
            AIMessage(content="Final Planner summary with bounded observations."),
        ],
        "executor_results": ["duplicate state field"],
    }

    compact = _compact_aggregation_state(state)

    assert compact["planner_summary"] == (
        "Final Planner summary with bounded observations."
    )
    assert compact["executor_results"][0]["structured_results"]["metrics"] == {
        "rim_prior": True
    }
    assert "structural_prior" not in compact["executor_results"][0][
        "structured_results"
    ]
    assert "artifact_ids" not in compact["executor_results"][0]
