"""Bounded metadata loop shared by Planner assimilation, action and final revision."""

import json
import time

from pydantic import ValidationError

from mechcal.capabilities.discovery import amesp_enabled
from mechcal.runtime.case_memory import CaseMemoryStore, serialized
from mechcal.runtime.memory_continuation import continuation_context
from mechcal.runtime.memory_views import retrieved_working_set
from mechcal.runtime.planner_context import build_context, scope_assimilation_capabilities
from mechcal.runtime.prompts import load_prompt
from mechcal.runtime.retry import RejectedResponseError
from mechcal.schemas.memory import MemoryQuery


MEMORY_ACTIONS = {"search_case_records", "read_case_records", "find_compatible_artifacts"}


class PlannerMemorySession:
    max_queries = 4

    def __init__(self, case, store, registry, stage, payload):
        self.case, self.store, self.registry = case, store, registry
        self.index = CaseMemoryStore(store, case)
        self.stage = stage
        self.identity = {
            "case_id": case.case_id, "stage": stage, "snapshot": self.index.snapshot,
            "revision": payload["revision"],
        }
        self.used = 0
        self.loaded = {}
        self.delta_prepared = {}
        self.delta_reopens = {}
        self.delta_lookup_pending = {}
        self.log_path = store.root / "history/memory_queries.jsonl"
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if all(event.get(k) == v for k, v in self.identity.items()) and event["kind"] == "memory_query":
                    self.used = max(self.used, event["queries_used"])
                    self.loaded.pop(serialized(event["request"]), None)
                    self.loaded[serialized(event["request"])] = event["response"]
                if all(event.get(k) == v for k, v in self.identity.items()) and event["kind"] == "delta_context_ready":
                    self.delta_prepared[event["context_key"]] = event["focus_capability_ids"]
                    self.delta_lookup_pending.pop(event["context_key"], None)
                if all(event.get(k) == v for k, v in self.identity.items()) and event["kind"] == "delta_context_reopened":
                    self.delta_reopens[event["context_key"]] = event["count"]
                    self.delta_lookup_pending[event["context_key"]] = event["reason"]
        if self.used > self.max_queries:
            raise ValueError("Invalid memory query allowance in checkpoint")
        self.payload = build_context(payload)
        self.emit("memory_context", legacy_bytes=len(serialized(payload).encode()),
                  projected_bytes=len(serialized(self.payload).encode()),
                  context_version=self.payload["context_version"],
                  field_bytes={key: len(serialized(value).encode()) for key, value in self.payload.items()})

    def emit(self, kind, **data):
        self.store.append_jsonl("history/memory_queries.jsonl", {
            **self.identity, "kind": kind, "timestamp": time.time(), **data,
        })

    def augment(self, payload):
        loaded = list(self.loaded.values())
        if self.case.runtime.get("scoped_evidence_updates"):
            loaded = retrieved_working_set(loaded)
        return {
            **payload,
            "memory_access": {
                "query_schema": MemoryQuery.model_json_schema(),
                "queries_remaining": self.max_queries - self.used,
                "max_queries": self.max_queries,
                "notice": "Queries share this allowance across format retries and resume. "
                          "No scientific round/action is consumed. Only current case records are accessible.",
            },
            "retrieved_memory": loaded,
        }

    def query(self, raw):
        if self.used >= self.max_queries:
            raise RejectedResponseError("Memory query allowance exhausted; use loaded records or submit a decision.", raw)
        self.used += 1
        started = time.monotonic()
        try:
            request = MemoryQuery.model_validate(raw)
            if request.action == "search_case_records":
                response = self.index.search(request)
            elif request.action == "read_case_records":
                response = self.index.read(request)
            else:
                response = self.compatible_artifacts(request)
            response = {"request": raw, **response}
        except (ValidationError, ValueError) as exc:
            # A typed query error is returned to the model, not silently converted to no matches.
            response = {"request": raw, "error": str(exc), "status": "query_rejected"}
        self.loaded.pop(serialized(raw), None)
        self.loaded[serialized(raw)] = response
        self.emit("memory_query", request=raw, response=response, queries_used=self.used,
                  latency_seconds=time.monotonic() - started)
        self.store.activity("Query case memory", f"R{self.case.runtime['question_loop'].get('round_index', 0):03d}",
                            str(raw.get("action")) + (": rejected" if response.get("error") else ": returned"))
        return response

    def compatible_artifacts(self, request):
        spec = self.registry.get(request.capability_id)
        if spec is None:
            raise ValueError(f"Unknown capability_id: {request.capability_id}")
        candidates = [a for a in self.case.artifact_manifest.artifacts if a.kind in spec.input_artifact_kinds]
        result = []
        for artifact in candidates:
            readiness = self.registry.readiness(
                spec.capability_id, self.case.artifact_manifest, amesp_enabled=amesp_enabled(self.case),
                source_artifact_id=artifact.artifact_id,
                tool_args={**request.tool_args, "source_artifact_id": artifact.artifact_id},
            )
            result.append({"artifact_id": artifact.artifact_id, "kind": artifact.kind, **readiness})
        end = request.offset + request.limit
        return {
            "artifacts": result[request.offset:end], "total": len(result),
            "next_offset": end if end < len(result) else None,
            "capability_status": self.registry.readiness(
                spec.capability_id, self.case.artifact_manifest, amesp_enabled=amesp_enabled(self.case),
                tool_args=request.tool_args,
            ),
            "notice": "Registry preflight only, not a completed calculation or guarantee of success. "
                      "Unspecified calculation parameters still require validation at dispatch.",
        }

    def complete(self, client, references, *, payload, system_prompt, schema_feedback=None,
                 event_sink=None, discovery=None):
        if self.case.runtime.get("planner_delta_context"):
            from mechcal.runtime.delta_session import complete_delta
            return complete_delta(self, client, references, payload=payload, system_prompt=system_prompt,
                                  schema_feedback=schema_feedback, event_sink=event_sink, discovery=discovery)
        if payload.get("stage") == "planner_question_act":
            system_prompt = load_prompt("planner_action_compact.md")
            if discovery is not None:
                system_prompt += "\n" + load_prompt("planner_tool_discovery.md")
        elif (payload.get("stage") == "planner_assimilate"
              and payload.get("incremental_update", {}).get("interface") == "judgment_proposal_v3"):
            system_prompt = load_prompt("planner_judgment_compact.md")
        while True:
            continuing = (self.used > 0 or schema_feedback is not None
                          or (discovery is not None and discovery.batches_used > 0))
            working = continuation_context(payload, retrieved_working_set(list(self.loaded.values()))) \
                if continuing else payload
            active = self.augment(working)
            if discovery is not None:
                active = discovery.augment_payload(active)
            if (self.case.runtime.get("scoped_evidence_updates")
                    and active.get("stage") == "planner_assimilate"):
                # Discovery may add its directory after the working-set projection.
                scope_assimilation_capabilities(active)
            if self.case.runtime.get("scoped_evidence_updates"):
                self.emit(
                    "memory_request_context", continuation=continuing,
                    original_context_bytes=len(serialized(payload).encode()),
                    working_context_bytes=len(serialized(working).encode()),
                    stored_retrieval_bytes=len(serialized(list(self.loaded.values())).encode()),
                    visible_retrieval_bytes=len(serialized(active["retrieved_memory"]).encode()),
                    field_bytes={k: len(serialized(v).encode()) for k, v in active.items()},
                )
            usage = None

            def record(event):
                nonlocal usage
                if event.get("usage") is not None:
                    usage = event["usage"]
                if event_sink:
                    event_sink(event)

            started = time.monotonic()
            self.store.activity(
                "Planner " + active.get("stage", self.stage).removeprefix("planner_"),
                f"R{self.case.runtime['question_loop']['round_index']:03d}",
                f"Memory queries used: {self.used}/{self.max_queries}; awaiting model response",
            )
            response = references.complete(
                client, payload=active, system_prompt=system_prompt + "\n" + load_prompt(
                    "planner_memory_focused.md" if self.case.runtime.get("scoped_evidence_updates")
                    else "planner_memory_compact.md"),
                schema_feedback=schema_feedback, event_sink=record,
            )
            from mechcal.agents.planner import _unwrap_planner_response
            unwrapped = _unwrap_planner_response(response)
            action = unwrapped.get("action")
            is_query = action in MEMORY_ACTIONS or action == "describe_tools"
            self.emit("memory_model_call", operation=action if is_query or action == "repair_judgment" else "decision",
                      usage=usage, latency_seconds=time.monotonic() - started,
                      usage_is_duplicate_of_request_log=True)
            if action in MEMORY_ACTIONS:
                self.query(unwrapped)
            elif action == "describe_tools" and discovery is not None:
                try:
                    discovery.query(unwrapped)
                except ValueError as exc:
                    raise RejectedResponseError(str(exc), response) from exc
            elif action == "describe_tools":
                raise RejectedResponseError("Tool-description queries are available during action selection only.", response)
            else:
                return response
