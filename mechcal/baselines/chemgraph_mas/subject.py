from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import httpx
from chemgraph.graphs.multi_agent import construct_multi_agent_graph
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import ValidationError

from mechcal.baselines.chemgraph_mas.budget import BudgetLimits, SharedBudget
from mechcal.baselines.chemgraph_mas.model import (
    BudgetedChatOpenAI,
    _compact_tool_message_for_planner,
)
from mechcal.baselines.chemgraph_mas.schemas import PhotoMechOutput
from mechcal.baselines.chemgraph_mas.tools import ChemGraphToolHarness
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.llm import OpenAICompatibleSettings, _parse_json_object

PROMPT_VERSION = "2026-07-24"
UPSTREAM_VERSION = "v0.6.0"
UPSTREAM_COMMIT = "203e18c8529869531fa4addbcdb3f1395a831eaa"


class ChemGraphMASSubject:
    subject_id = "chemgraph_mas_adapted"
    subject_kind = "chemgraph_mas"

    def __init__(
        self,
        *,
        settings: OpenAICompatibleSettings,
        budget_limits: BudgetLimits,
        run_base_dir: Path,
        enable_amesp: bool = True,
        recursion_limit: int = 80,
        schema_retries: int = 1,
        use_responses_api: bool = False,
    ) -> None:
        missing = settings.missing_fields()
        if missing:
            raise RuntimeError("Missing LLM settings: " + ", ".join(missing))
        self.settings = settings
        self.budget_limits = budget_limits
        self.run_base_dir = run_base_dir.expanduser().resolve()
        self.enable_amesp = enable_amesp
        self.recursion_limit = max(10, recursion_limit)
        self.schema_retries = max(0, schema_retries)
        self.use_responses_api = use_responses_api

    @property
    def model(self) -> str:
        return self.settings.model

    def run(
        self,
        *,
        case_id: str,
        public_case_id: str,
        smiles: str,
        task: str,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.run_async(
                case_id=case_id,
                public_case_id=public_case_id,
                smiles=smiles,
                task=task,
            )
        )

    async def run_async(
        self,
        *,
        case_id: str,
        public_case_id: str,
        smiles: str,
        task: str,
    ) -> dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        workspace = self.run_base_dir / case_id / run_id
        budget = SharedBudget(self.budget_limits)
        harness = ChemGraphToolHarness(
            case_id=case_id,
            public_case_id=public_case_id,
            smiles=smiles,
            task=task,
            workspace=workspace,
            budget=budget,
            enable_amesp=self.enable_amesp,
        )
        llm, sync_http, async_http = self._make_llm(budget)
        try:
            planning_tool_ceiling = min(
                self.budget_limits.max_tool_calls,
                5,
                max(1, (self.budget_limits.max_model_calls - 14) // 2),
            )
            per_round_tool_ceiling = min(4, planning_tool_ceiling)
            planner_prompt = (
                _load_prompt("chemgraph_mas_planner.md")
                + "\n\nALLOWED ANALYSIS CATALOG\n"
                + harness.compact_catalog()
                + "\n\nSHARED CASE BUDGET\n"
                + f"- Model calls: {self.budget_limits.max_model_calls}\n"
                + f"- Tool calls: {self.budget_limits.max_tool_calls}\n"
                + f"- Total tokens: {self.budget_limits.max_total_tokens}\n"
                + (
                    "- Select only the analyses needed to distinguish the leading "
                    "mechanisms. Do not request the full catalog.\n"
                )
                + (
                    f"- Plan no more than {planning_tool_ceiling} capability "
                    "invocations across the entire case. This is a cumulative hard "
                    "ceiling, not a per-round allowance.\n"
                )
                + (
                    f"- Request no more than {per_round_tool_ceiling} capability "
                    "invocations in one dispatch. Count every capability ID already "
                    "requested before planning follow-up work.\n"
                )
                + (
                    "- Use at most two Executor rounds. After the second Executor "
                    "round, set next_step to FINISH so that retries and final "
                    "aggregation remain within the model-call limit."
                )
            )
            graph = construct_multi_agent_graph(
                llm,
                planner_prompt=planner_prompt,
                executor_prompt=_load_prompt("chemgraph_mas_executor.md"),
                executor_tools=harness.langchain_tools(),
                structured_output=False,
                max_retries=self.schema_retries,
                max_task_retries=1,
                human_supervised=False,
            )
            public_query = _public_query(
                public_case_id=public_case_id,
                smiles=smiles,
                task=task,
            )
            config = {
                "configurable": {"thread_id": f"{case_id}:{run_id}"},
                "recursion_limit": self.recursion_limit,
            }
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=public_query)]},
                config=config,
            )
            output = await self._aggregate(
                llm=llm,
                public_case_id=public_case_id,
                smiles=smiles,
                task=task,
                state=state,
            )
        finally:
            sync_http.close()
            await async_http.aclose()

        record = {
            "case_id": case_id,
            "public_case_id": public_case_id,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "upstream": {
                "repository": "https://github.com/argonne-lcf/ChemGraph",
                "version": UPSTREAM_VERSION,
                "commit": UPSTREAM_COMMIT,
            },
            "public_input": {
                "smiles": smiles,
                "mechanism_pool": list(MECHANISM_POOL),
                "task": task,
            },
            "raw_json": output.model_dump(mode="json"),
            "chemgraph_state": _serialize_value(state),
            "tool_policy": harness.policy_snapshot(),
            "budget": budget.snapshot(),
            "workflow_policy": {
                "api_protocol": (
                    "responses" if self.use_responses_api else "chat_completions"
                ),
                "planner_executor_replanning": True,
                "final_aggregator": True,
                "human_supervision": False,
                "cross_case_memory": False,
                "mechcal_evidence_ledger": False,
                "coverage_reviewer": False,
                "support_auditor": False,
                "claim_gate": False,
            },
            "run_dir": str(workspace),
        }
        _write_json(workspace / "raw_record.json", record)
        return record

    def _make_llm(
        self,
        budget: SharedBudget,
    ) -> tuple[BudgetedChatOpenAI, httpx.Client, httpx.AsyncClient]:
        sync_http = httpx.Client(
            timeout=self.settings.timeout_seconds,
            trust_env=self.settings.trust_env,
        )
        async_http = httpx.AsyncClient(
            timeout=self.settings.timeout_seconds,
            trust_env=self.settings.trust_env,
        )
        llm = BudgetedChatOpenAI(
            shared_budget=budget,
            model=self.settings.model,
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            temperature=None if self.use_responses_api else self.settings.temperature,
            top_p=None if self.use_responses_api else self.settings.top_p,
            max_tokens=min(
                self.settings.max_tokens or self.budget_limits.max_output_tokens_per_call,
                self.budget_limits.max_output_tokens_per_call,
            ),
            reasoning_effort=self.settings.reasoning_effort,
            use_responses_api=self.use_responses_api,
            store=False if self.use_responses_api else None,
            max_retries=0,
            timeout=self.settings.timeout_seconds,
            http_client=sync_http,
            http_async_client=async_http,
        )
        return llm, sync_http, async_http

    async def _aggregate(
        self,
        *,
        llm: BudgetedChatOpenAI,
        public_case_id: str,
        smiles: str,
        task: str,
        state: dict[str, Any],
    ) -> PhotoMechOutput:
        payload = {
            "case_id": public_case_id,
            "verified_smiles": smiles,
            "task": task,
            "mechanism_pool": list(MECHANISM_POOL),
            **_compact_aggregation_state(state),
        }
        messages: list[BaseMessage] = [
            SystemMessage(content=_load_prompt("chemgraph_mas_aggregator.md")),
            HumanMessage(
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
        ]
        last_error: Exception | None = None
        for attempt in range(self.schema_retries + 1):
            response = await llm.ainvoke(messages)
            raw_text = _message_text(response)
            try:
                return PhotoMechOutput.model_validate(_parse_json_object(raw_text))
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.schema_retries:
                    break
                messages.extend(
                    [
                        response,
                        HumanMessage(
                            content=(
                                "Return only a corrected JSON object matching the requested "
                                f"schema. Previous validation error: {exc}"
                            )
                        ),
                    ]
                )
        raise RuntimeError(f"Aggregator failed schema validation: {last_error}")


def _public_query(
    *,
    public_case_id: str,
    smiles: str,
    task: str,
) -> str:
    return (
        f"Case: {public_case_id}\n"
        f"Verified SMILES: {smiles}\n"
        f"Task: {task}\n"
        "Rank exactly three mechanisms from this fixed pool:\n"
        + "\n".join(f"- {label}" for label in MECHANISM_POOL)
        + "\nUse only the allowed local Macro and Microscopic analyses."
    )


def _compact_aggregation_state(state: dict[str, Any]) -> dict[str, Any]:
    messages = state.get("messages")
    if not isinstance(messages, list):
        messages = []

    planner_summary = ""
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        content = _message_text(message).strip()
        if content:
            planner_summary = content
            break

    executor_results: list[Any] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, str):
            compact = _compact_tool_message_for_planner(content)
            try:
                executor_results.append(json.loads(compact))
            except json.JSONDecodeError:
                executor_results.append(compact)
        else:
            executor_results.append(_serialize_value(content))

    if not executor_results:
        executor_results = _serialize_value(state.get("executor_results", []))
    return {
        "planner_summary": planner_summary,
        "executor_results": executor_results,
    }


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / name
    return path.read_text(encoding="utf-8").strip()


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") if isinstance(item, dict) else item)
            for item in content
        )
    return str(content)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return {
            "type": value.type,
            "content": value.content,
            "tool_calls": getattr(value, "tool_calls", []),
        }
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _serialize_value(value.model_dump(mode="json"))
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
