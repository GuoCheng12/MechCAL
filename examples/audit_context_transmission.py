"""Replay message construction without model calls or scientific calculations.

Run this same script against two source checkouts and the same checkpoint.
Byte measurements are not provider token usage or online quality results.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--responses", type=int, default=234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report path.")
    sys.path.insert(0, str(args.source.resolve()))
    from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
    from mechcal.runtime.command_compact import expand_text
    from mechcal.runtime.command_loop import planner_payload
    from mechcal.runtime.command_task_context import prepare_planner
    from mechcal.runtime.command_memory import json_bytes, digest, resolve_pointer
    from mechcal.runtime.command_operations import protocol, record_index
    from mechcal.runtime.snapshot_storage import read_case_run
    from mechcal.agents.command_tools import ask_tools
    from mechcal.runtime.llm import OpenAICompatibleSettings
    from mechcal.capabilities.registry import CapabilityRegistry
    from mechcal.capabilities.schemas import CapabilityCatalog
    try:
        from mechcal.runtime.command_review_wire import operation_wire
    except ImportError:
        operation_wire = None

    report = {"source": str(args.source.resolve()), "network_calls": 0,
              "measurement": "serialized JSON bytes, not API tokens; no online scientific claims"}
    with args.checkpoint.open("rb") as handle:
        report["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    case = read_case_run(args.checkpoint)
    before = digest(case.model_dump(mode="json"))
    cfg = OrchestratorConfig(run_base_dir=args.output.parent / "unused-runs",
        decision_mode="question_loop_v1", unlimited_exploration=True,
        enable_amesp=True, planner_command_protocol=True, command_native_tools=True,
        command_hypothesis_tracking=True, command_evidence_followup=True,
        command_review_workflow=True, command_computational_review=True,
        command_investigation_policy=True, command_reviewer_exploration=True,
        command_compact_context=True, command_task_context=True, command_worker_tasks=True)
    orchestrator = MechCALOrchestrator(cfg)
    catalog_file = args.checkpoint.parent / "capabilities.json"
    catalog = CapabilityCatalog.model_validate_json(catalog_file.read_text())
    orchestrator.deps.capability_registry = CapabilityRegistry(catalog.capabilities)
    report["catalog_sha256"] = hashlib.sha256(catalog_file.read_bytes()).hexdigest()
    original = planner_payload(orchestrator, case, native=True)
    prepared, view = prepare_planner(case, original)

    class Captured(Exception):
        pass

    class CaptureClient:
        settings = OpenAICompatibleSettings(base_url="https://api.deepseek.com", model="offline",
            api_key="offline-not-a-credential", api_protocol="chat_completions")
        request = None

        def complete_tool_turn(self, **kw):
            self.request = kw
            raise Captured()

    client = CaptureClient()
    try:
        ask_tools(client, orchestrator.deps.capability_registry, protocol(prepared).get("visible_tools", []),
                  view, None)
    except Captured:
        pass
    assert client.request is not None
    request = client.request
    transmitted = request["payload"]
    if operation_wire:
        transmitted, _ = operation_wire(transmitted, None, None)
    tools = [{"type": "function", "function": {**t, "strict": False}} for t in request["tools"]]
    messages = [{"role": "system", "content": request["system_prompt"]},
                {"role": "user", "content": json.dumps(transmitted, ensure_ascii=False, separators=(",", ":"))}]
    report["planner"] = {"payload_before_task_view": json_bytes(original), "payload_after_task_view": json_bytes(view),
        "request_bytes": json_bytes({"messages": messages, "tools": tools}), "tools_bytes": json_bytes(tools),
        "system_bytes": len(request["system_prompt"].encode()),
        "largest_fields": sorted([(k, json_bytes(v)) for k, v in expand_text(view).items()], key=lambda p: -p[1])[:12]}
    fields = protocol(prepared)["planner_context"]["fields"]
    report["planner"]["stored_context_bytes"] = json_bytes(fields)
    report["source_case_unchanged"] = before == digest(case.model_dump(mode="json"))
    assert report["source_case_unchanged"]
    assert record_index(case).observations == record_index(prepared).observations

    if args.requests:
        try:
            from mechcal.runtime.command_context_memory import project_scoped
        except ImportError:
            project_scoped = None
        incoming, completed = [], 0
        with args.requests.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "request" and row.get("role") != "planner":
                    incoming.append(row)
                elif row.get("event") == "response":
                    completed += 1
                    if completed >= args.responses:
                        break
        grouped = {}
        for row in incoming:
            role = row["role"]
            if role not in {"worker", "impact_reviewer", "exploration_reviewer", "reviewer", "final_reviewer"}:
                continue
            messages = row["messages"]
            pos = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), None)
            if pos is None:
                continue
            value = expand_text(json.loads(messages[pos]["content"]))
            reports = [value["result"]] if "result" in value else value.get("results", [])
            focus = {r["source_result_id"] for r in reports if r.get("source_result_id")}
            focus.update(r["ref"] for r in value.get("working_memory", {}).get("findings", []))
            view, context = (project_scoped(value, case_id=case.case_id, role=role, focus_refs=focus)
                             if project_scoped else (value, None))
            if context:
                for entry in view["task_memory"]["fields"]:
                    assert resolve_pointer(context, entry["path"]) == context["fields"][entry["field"]]
            changed = list(messages)
            if operation_wire:
                view, _ = operation_wire(view, None, None)
            if project_scoped:
                changed[pos] = {**messages[pos], "content": json.dumps(view, ensure_ascii=False, separators=(",", ":"))}
            group = grouped.setdefault(role, {"requests": 0, "original_bytes": 0, "projected_bytes": 0})
            group["requests"] += 1
            group["original_bytes"] += json_bytes({"messages": messages, "tools": row.get("tools", [])})
            group["projected_bytes"] += json_bytes({"messages": changed, "tools": row.get("tools", [])})
        report["historical_role_projection"] = grouped
        report["historical_projection_note"] = "Same recorded turns; includes system/history/tools, excludes additional queries."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
