"""Publish a sanitized, offline audit of recorded calls; never run a model."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


SECRET = re.compile(
    r"(?<![\w-])(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
)
CREDENTIAL_VALUE = re.compile(
    r'("(?:api_key|access_token|refresh_token|password|client_secret)"\s*:\s*")[^"\n]+(")',
    re.IGNORECASE,
)
EXCLUDED = {".mo": "binary wavefunction, not needed for token analysis",
            ".lock": "process lock, not a historical result"}
TZ = timezone(timedelta(hours=8))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def sanitize(data, workspace, home):
    text = data.decode("utf-8")
    text = SECRET.sub("REDACTED_CREDENTIAL", text)
    text = CREDENTIAL_VALUE.sub(r"\1REDACTED_CREDENTIAL\2", text)
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]{16,}", "Bearer REDACTED_CREDENTIAL", text)
    # Replace paths, not model observations, scientific values, or usage counters.
    text = text.replace(str(workspace), "/SOURCE_WORKSPACE")
    text = text.replace(str(home), "/SOURCE_HOME")
    if "PRIVATE KEY-----" in text:
        raise ValueError("Private key material: refuse publication")
    return text.encode()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def usage_value(usage, primary, alternate=None):
    value = usage.get(primary)
    if value is None and alternate:
        value = usage.get(alternate)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"Invalid recorded counter: {primary}")
    return value


def telemetry(rows):
    """Match exact recorded attempts. Missing usage remains null, never zero."""
    requests = {}
    output = []
    seen = Counter()
    for line, row in enumerate(rows, 1):
        key = (row["id"], row.get("attempt", 1))
        if row["event"] == "request":
            if key in requests:
                raise ValueError("Duplicate request id/attempt")
            requests[key] = (line, row)
            continue
        if row["event"] != "response":
            raise ValueError(f"Unimplemented history event: {row['event']}")
        if key not in requests:
            raise ValueError("Response without a recorded request")
        request_line, req = requests.pop(key)
        usage = row.get("usage") or {}
        raw = row.get("raw_response") or {}
        tokens_in = usage_value(usage, "input_tokens", "prompt_tokens")
        tokens_out = usage_value(usage, "output_tokens", "completion_tokens")
        total = usage_value(usage, "total_tokens")
        if tokens_in is not None and tokens_out is not None:
            if total is not None and total != tokens_in + tokens_out:
                raise ValueError("Usage total disagrees with recorded input/output")
            if total is None:
                total = tokens_in + tokens_out
        cached = usage_value(usage, "cached_tokens")
        if cached is None:
            cached = usage_value(usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}, "cached_tokens")
        reasoning = usage_value(usage, "reasoning_tokens")
        if reasoning is None:
            reasoning = usage_value(usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}, "reasoning_tokens")
        if cached is not None and tokens_in is not None and cached > tokens_in:
            raise ValueError("Cached input exceeds total input")
        components = Counter()
        component_records = []
        for msg in req.get("messages", []):
            kind = msg.get("role") or msg.get("type") or "unknown"
            data = json_bytes(msg)
            sha = digest(data)
            components[kind] += len(data)
            component_records.append({"kind": kind, "bytes": len(data), "sha256": sha,
                                      "previous_identical_occurrences": seen[sha]})
            seen[sha] += 1
        tool_bytes = json_bytes(req.get("tools", []))
        tools_sha = digest(tool_bytes)
        tools_previous = seen[tools_sha]
        seen[tools_sha] += 1
        elapsed = row.get("elapsed_seconds")
        start = req.get("timestamp")
        end = start + elapsed if start is not None and elapsed is not None else None
        output.append({
            "event": "derived_api_call", "derived": True,
            "provenance": {"file": "case/history/command_requests.jsonl",
                           "request_line": request_line, "response_line": line,
                           "note": "Usage is provider-recorded, not a text-token estimate."},
            "sequence": len(output) + 1, "id": key[0], "attempt": key[1],
            "role": row["role"], "model": raw.get("model"),
            "protocol": req.get("protocol"), "transport": req.get("transport"),
            "started_at": start, "finished_at_estimated": end,
            "started_at_shanghai": datetime.fromtimestamp(start, TZ).isoformat() if start else None,
            "elapsed_seconds": elapsed, "usage_recorded": bool(usage),
            "input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": total,
            "cached_input_tokens": cached, "reasoning_output_tokens": reasoning,
            "raw_usage": usage,
            "response_status": raw.get("status"),
            "generated_commands": [item.get("name") for item in raw.get("output", [])
                                   if item.get("type") == "function_call"],
            "recorded_messages_bytes": req.get("messages_bytes"),
            "recorded_tools_bytes": req.get("tools_bytes"),
            "original_input_components_bytes": dict(components),
            "original_message_fingerprints": component_records,
            "original_tools_bytes": len(tool_bytes), "original_tools_sha256": tools_sha,
            "previous_identical_tools_occurrences": tools_previous,
        })
    if requests:
        raise ValueError("Unanswered requests require explicit reporting before publishing")
    return output


def totals(calls):
    result = {"calls": len(calls), "calls_with_usage": sum(c["usage_recorded"] for c in calls)}
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_output_tokens"):
        values = [c[key] for c in calls]
        result[key] = sum(values) if all(v is not None for v in values) else None
    result["summed_request_elapsed_seconds"] = sum(c["elapsed_seconds"] or 0 for c in calls)
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(source, target, workspace, home):
    if target.exists():
        raise ValueError("Use a new output directory; never overwrite audit records")
    histories = list(source.rglob("history/command_requests.jsonl"))
    if len(histories) != 1:
        raise ValueError("Expected one case history")
    history = histories[0]
    case = history.parent.parent
    if case.name != "AIE_DDX_MECH_V4_0003":
        raise ValueError("Publication is authorized only for Case 0003")
    source_rows = [json.loads(line) for line in history.read_text().splitlines()]
    calls = telemetry(source_rows)
    summary = json.loads((case.parent.parent / "summary.json").read_text())
    target.mkdir(parents=True)
    inventory = []
    standalone = {"command_checkpoint.json", "case_run.json", "history/command_requests.jsonl"}
    # Keep every text record in a single archive, including tool inputs/outputs,
    # operation receipts and memory. Only large binary wavefunctions and locks go.
    with (target / "records.tar.gz").open("wb") as fh:
        with gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in sorted(source.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("Unexpected symlink in source records")
                    if not path.is_file():
                        continue
                    rel = path.relative_to(source).as_posix()
                    original = path.read_bytes()
                    record = {"source_relative_path": rel, "source_bytes": len(original),
                              "source_sha256": digest(original)}
                    if path.suffix in EXCLUDED:
                        record.update(included=False, reason=EXCLUDED[path.suffix])
                        inventory.append(record)
                        continue
                    data = sanitize(original, workspace, home)
                    if "_payloads" in path.parts and data != original:
                        raise ValueError("Redaction would invalidate a content-addressed payload")
                    if path.suffix == ".json":
                        json.loads(data)
                    elif path.suffix == ".jsonl":
                        for line in data.splitlines():
                            json.loads(line)
                    info = tarfile.TarInfo(rel)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    archive.addfile(info, io.BytesIO(data))
                    record.update(included=True, published_bytes=len(data), published_sha256=digest(data),
                                  redacted=(data != original))
                    inventory.append(record)
                    public_path = None
                    if path.is_relative_to(case):
                        case_rel = path.relative_to(case).as_posix()
                        if case_rel in standalone or case_rel.startswith("_payloads/"):
                            public_path = target / "case" / case_rel
                    elif path.parent == case.parent.parent and path.name in {"manifest.json", "summary.json", "budget_stop.json"}:
                        public_path = target / "recorded" / path.name
                    if public_path is not None:
                        public_path.parent.mkdir(parents=True, exist_ok=True)
                        public_path.write_bytes(data)
    (target / "api_telemetry.jsonl").write_bytes(b"".join(json_bytes(row) + b"\n" for row in calls))
    by_role = defaultdict(list)
    for call in calls:
        by_role[call["role"]].append(call)
    result = {"source_run_name": source.name, "case_id": case.name,
              "started_at_shanghai": calls[0]["started_at_shanghai"],
              "status": summary["status"], "phase": summary.get("phase"),
              "finalized": summary["finalized"], "judge_called": summary["judge_called"],
              "rounds": summary["rounds"], "scientific_computations": summary["scientific_computations"],
              "wall_elapsed_seconds": summary["elapsed_seconds"],
              "totals": totals(calls), "by_role": {k: totals(v) for k, v in sorted(by_role.items())},
              "original_api_telemetry_present": False,
              "telemetry_source": "case/history/command_requests.jsonl",
              "resumable_archive": False,
              "archive_note": "Analysis archive. Source paths are anonymized; binary wavefunctions and locks are excluded.",
              "accounting_note": "Cached input is part of input. Reasoning output is part of output. Neither is additive.",
              "bytes_note": "Component bytes/fingerprints describe original messages, before path redaction. They are not token estimates.",
              "source_files": len(inventory), "included_files": sum(x["included"] for x in inventory),
              "excluded_files": sum(not x["included"] for x in inventory),
              "archive_bytes": (target / "records.tar.gz").stat().st_size,
              "archive_sha256": digest((target / "records.tar.gz").read_bytes())}
    for field in ("input_tokens", "output_tokens"):
        if result["totals"][field] != summary[field]:
            raise ValueError(f"Source summary and request history disagree: {field}")
    write_json(target / "token_summary.json", result)
    write_json(target / "source_inventory.json", inventory)
    write_csv(target / "calls.csv", calls, ["sequence", "role", "model", "id", "attempt", "started_at_shanghai",
              "elapsed_seconds", "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
              "reasoning_output_tokens", "recorded_messages_bytes", "recorded_tools_bytes"])
    write_csv(target / "by_agent.csv", [{"agent": k, **v} for k, v in result["by_role"].items()],
              ["agent", "calls", "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
               "reasoning_output_tokens", "summed_request_elapsed_seconds"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.source.resolve(), args.output.resolve(), args.workspace.resolve(), args.home.resolve())
    print(json.dumps({k: report[k] for k in ("source_run_name", "totals", "included_files", "excluded_files", "archive_bytes")}))
