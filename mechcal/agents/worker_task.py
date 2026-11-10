"""Worker-owned operational choices through the existing native tool transport."""

from mechcal.agents.command_tools import NativeCommandClient, tool_contracts, strict_tool_schema
from mechcal.runtime.command_schema import command_parameters
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas.worker_tasks import CompleteWorkerTask, ReadWorkerLog


def contracts(registry, loaded, client):
    tools, models = tool_contracts(registry, loaded, strict_schema=strict_tool_schema(client))
    names = {"describe_tools", "read_evidence", "reduce_evidence", "compare_evidence"}
    names.update(name for name in models if name.startswith("compute_"))
    tools = [tool for tool in tools if tool["name"] in names]
    models = {name: spec for name, spec in models.items() if name in names}
    for name, model, description in (
        ("complete_worker_task", CompleteWorkerTask, "Return the delegated task's result, partial answer or blocker, with concise prose."),
        ("read_worker_log", ReadWorkerLog, "Read a bounded text-log page by record_id and an exact file_key from log_index. No arbitrary paths."),
    ):
        identity = {"kind": name}
        tools.append({"name": name, "description": description,
                      "parameters": command_parameters(model, identity, strict=strict_tool_schema(client))})
        models[name] = (model, identity)
    return tools, models


def ask(client, payload, tools, task, store):
    kwargs = dict(system_prompt=load_prompt("worker_task.md") + "\n" + load_prompt("numeric_evidence.md"), payload=payload,
        tools=tools, exchange=task.exchange, outcome=task.outcome, store=store,
        role=task.contract.owner + "_worker_task", allow_text=True)
    method = getattr(client, "complete_tool_turn", None)
    return method(**kwargs) if method else NativeCommandClient.complete_tool_turn(client, **kwargs)
