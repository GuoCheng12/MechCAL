"""Bounded diagnostics from explicitly registered process logs, not directory search."""

import hashlib
from pathlib import Path

LOG_SUFFIXES = {".log", ".out", ".err", ".txt", ".stdout", ".stderr"}
DIAGNOSTIC_BYTES = 4096


def safe_log(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if any(p.is_symlink() for p in (path, *path.parents, root, *root.parents)):
        return None
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in LOG_SUFFIXES:
        return None
    return path if path.is_file() else None


def log_excerpt(path, root):
    """Keep byte offsets and hash so a tail cannot be mistaken for a complete log."""
    path = safe_log(path, root)
    if path is None:
        return None
    try:
        with path.open("rb") as handle:
            total = handle.seek(0, 2)
            offset = max(0, total - DIAGNOSTIC_BYTES)
            handle.seek(offset)
            data = handle.read(DIAGNOSTIC_BYTES)
    except OSError:
        return None
    return {"log": str(path), "offset": offset, "total_bytes": total,
            "truncated": offset > 0, "chunk_sha256": hashlib.sha256(data).hexdigest(),
            "text": data.decode("utf-8", errors="replace")}


def registered_failure_logs(result):
    root = result.get("raw_results", {}).get("workdir") or (result.get("failure") or {}).get("details", {}).get("workdir")
    if not root:
        return {}
    files = {}
    for step in result.get("raw_results", {}).get("steps", []):
        if step.get("log"):
            path = safe_log(step["log"], root)
            if path:
                files[str(path.relative_to(Path(root).resolve()))] = path
    return files


def failure_diagnostics(result):
    """Saved excerpts first; legacy results may read their registered logs only."""
    if result.get("status") not in {"failed", "partial", "unsupported", "precondition_missing"}:
        return []
    steps = result.get("raw_results", {}).get("steps", [])
    failed = [(i, s) for i, s in enumerate(steps) if s.get("returncode") or s.get("timed_out")]
    selected = failed[-2:] or list(enumerate(steps))[-1:]
    registered = registered_failure_logs(result)
    root = result.get("raw_results", {}).get("workdir") or (result.get("failure") or {}).get("details", {}).get("workdir")
    rows = []
    for i, step in selected:
        excerpt = step.get("diagnostic")
        if not excerpt and root and step.get("log"):
            path = safe_log(step["log"], root)
            if path and path in registered.values():
                excerpt = log_excerpt(path, root)
        if excerpt:
            rows.append({"step_path": f"/raw_results/steps/{i}", "returncode": step.get("returncode"),
                         "timed_out": step.get("timed_out", False), "excerpt": excerpt,
                         "omitted_failed_steps": max(0, len(failed) - len(selected)),
                         "saved_in_record": bool(step.get("diagnostic")), "scientific_evidence": False})
    return rows
