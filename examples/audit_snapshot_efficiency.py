"""Offline replay of persisted state. Never makes model or scientific tool calls.

Run with --source pointing at either checkout to compare the same checkpoint.
Output is always written to a new directory, never into the source experiment.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sys.path.insert(0, str(args.source.resolve()))
    from mechcal.runtime import MechCALOrchestrator, OrchestratorConfig
    from mechcal.runtime.command_loop import planner_payload, save
    from mechcal.runtime.command_operations import protocol, put, record_index
    from mechcal.runtime.command_review_delta import project
    from mechcal.runtime.command_memory import digest, json_bytes, resolve_pointer
    from mechcal.runtime.command_task_context import prepare_planner
    from mechcal.runtime.persistence import RunStore
    from mechcal.schemas import CaseRun
    try:
        from mechcal.runtime.snapshot_storage import read_case_run
    except ImportError:
        def read_case_run(path):
            return CaseRun.model_validate_json(path.read_text())

    with args.checkpoint.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    report = {"source": str(args.source.resolve()), "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": checksum,
              "network_calls": 0}

    def timed(name, function):
        start = time.perf_counter()
        value = function()
        report[name + "_seconds"] = time.perf_counter() - start
        print(name, round(report[name + "_seconds"], 3), flush=True)
        return value

    case = timed("load", lambda: read_case_run(args.checkpoint))
    cfg = OrchestratorConfig(run_base_dir=args.output / "runs", decision_mode="question_loop_v1",
        unlimited_exploration=True, enable_amesp=True, planner_command_protocol=True,
        command_native_tools=True, command_hypothesis_tracking=True, command_evidence_followup=True,
        command_review_workflow=True, command_computational_review=True, command_investigation_policy=True,
        command_reviewer_exploration=True, command_compact_context=True, command_task_context=True,
        command_worker_tasks=True, command_delivery_max_model_turns=30)
    orchestrator = MechCALOrchestrator(cfg)
    payload = timed("planner_payload", lambda: planner_payload(orchestrator, case, native=True))
    _, wire = timed("planner_task_view", lambda: prepare_planner(case, payload))
    report.update(planner_payload_bytes=json_bytes(payload), planner_payload_sha256=digest(payload),
                  planner_task_bytes=json_bytes(wire), planner_task_sha256=digest(wire))

    store = RunStore(args.output / "runs", case.case_id)
    store.initialize()
    saved = timed("save_first", lambda: save(case, store))
    blobs = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in store.root.glob("_payloads/*.json")}
    saved = timed("save_second", lambda: save(saved, store))
    report["arrays_rewritten_on_second_save"] = sum(
        p.name not in blobs or blobs[p.name] != (p.stat().st_size, p.stat().st_mtime_ns)
        for p in store.root.glob("_payloads/*.json"))
    restored = timed("restore", lambda: read_case_run(store.root / "command_checkpoint.json"))
    report["roundtrip_exact"] = saved.model_dump(mode="json") == restored.model_dump(mode="json")
    assert report["roundtrip_exact"]
    report.update(checkpoint_bytes=(store.root / "command_checkpoint.json").stat().st_size,
                  unique_array_bytes=sum(s[0] for s in blobs.values()), unique_array_count=len(blobs))
    # Same scientific source identities and every citation survive persistence.
    original, after = record_index(case), record_index(restored)
    report["observations_exact"] = original.observations == after.observations
    assert report["observations_exact"]

    # Replay consecutive historical final-review sections against the same source
    # snapshot, without regenerating their scientific judgments.
    receipts = [r for r in protocol(case).get("review_receipts", {}).values()
                if r["ref"].startswith("review:final_reviewer:") and r.get("feedback", {}).get("status") == "available"]
    comparisons = []
    for prior, current in zip(receipts, receipts[1:]):
        c = put(case, review_memory={"final_reviewer": prior})
        receipt = deepcopy(current)
        sections = receipt["sections"]
        material, _, _ = project(c, {"stage": "command_final_reviewer"}, "final_reviewer",
                                sections, receipt, record_index(c), task_context=True)
        shown = material["sections"].get("computational_review", {})
        checks = []
        for field in ("executions", "comparisons", "question_evidence", "candidate_comparison", "investigation"):
            value = shown.get(field)
            if isinstance(value, dict) and "previous_review_ref" in value:
                anchor = protocol(c)["review_receipts"][value["previous_review_ref"]]
                base = resolve_pointer(anchor, value["path"])
                if "changes" not in value:
                    checks.append(base == sections["computational_review"][field])
                else:
                    checks.append(digest(base) == value["base_sha256"] and
                                  digest(sections["computational_review"][field]) == value["current_sha256"])
        assert all(checks)
        comparisons.append({"prior": prior["ref"], "current": current["ref"],
                            "material_bytes": json_bytes(material), "background_bytes": json_bytes(shown),
                            "checked_anchors": len(checks)})
    report["review_replays"] = comparisons
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
