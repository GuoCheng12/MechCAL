"""Rebuildable search index over public case records, never over arbitrary files."""

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from mechcal.runtime.questions import state
from mechcal.runtime.question_changes import public_questions
from mechcal.runtime.memory_views import allocate_read_budgets, outline_record, render_record, selected_record
from mechcal.runtime.memory_brief import brief_record
from mechcal.runtime.memory_findings import findings_record
from mechcal.schemas.memory import MemoryQuery


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pointer(value, path):
    if not path.startswith("/"):
        raise ValueError("Selected fields must be JSON pointers starting with /")
    for part in path[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise ValueError(f"Field not found: {path}")
    return value


def record_rows(case, root):
    """Only explicit public sources enter memory; case metadata and judge files do not."""
    for item in case.evidence_ledger.items:
        yield item.evidence_id, "observation", item.model_dump(mode="json")
    s = state(case)
    for argument in case.mechanism_support_arguments:
        yield f"argument:{argument.support_id}", "judgment", argument.model_dump(mode="json")
    for hypothesis in case.portfolio.hypotheses:
        yield f"hypothesis:{hypothesis.name}", "judgment", hypothesis.model_dump(mode="json")
    for test in s.get("hypothesis_test_history", []):
        yield f"test:{test['revision']}:{test['result_id']}", "judgment", test
    responses = s.get("candidate_expansion", {}).get("responses", [])
    if responses:
        yield "history:candidate_responses", "judgment", {"responses": responses}
    for kind in ("argument", "result"):
        if s.get(f"{kind}_review"):
            yield f"review:{kind}", "judgment", s[f"{kind}_review"]
    if case.mechanism_agenda_coverage:
        yield "review:coverage", "judgment", case.mechanism_agenda_coverage.model_dump(mode="json")
    results = {r["result_id"]: r for r in [*s.get("results", []), *s.get("pending_results", [])]}
    for rid, result in results.items():
        yield rid, "execution", result
    for qid, question in public_questions(case).items():
        yield qid, "question", question
    for artifact in case.artifact_manifest.artifacts:
        yield artifact.artifact_id, "artifact", artifact.model_dump(mode="json")
    # Revision files are written only after accepted updates. Never index raw model responses.
    for path in sorted((root / "questions").glob("revision_*.json")):
        value = json.loads(path.read_text())
        if value.get("case_id") != case.case_id:
            raise ValueError("Memory revision belongs to another case")
        if value["revision"] > s.get("revision", 0):
            continue
        allowed = ("revision", "portfolio", "mechanism_support_arguments", "decision_assessment",
                   "ranking_update", "argument_changes", "dispositions")
        yield f"judgment:{value['revision']}", "judgment", {k: value[k] for k in allowed if k in value}
    if s.get("initial_judgment"):
        initial = s["initial_judgment"]
        yield "judgment:initial", "judgment", {
            k: initial[k] for k in ("revision", "portfolio", "mechanism_support_arguments",
                                    "decision_assessment") if k in initial
        }
    if s.get("last_comparison"):
        yield "judgment:last_comparison", "judgment", s["last_comparison"]


class CaseMemoryStore:
    """One index per run/case, with atomic rebuild and no dependency on stale cache data."""

    def __init__(self, store, case):
        if store.case_id != case.case_id:
            raise ValueError("Memory store/case mismatch")
        self.root = store.root.resolve()
        self.case_id = case.case_id
        self.focused_reads = case.runtime.get("scoped_evidence_updates", False)
        self.brief_auto = case.runtime.get("planner_delta_context", False)
        self.path = self.root / "memory" / "index.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [(rid, kind, serialized(body)) for rid, kind, body in record_rows(case, self.root)]
        self.snapshot = hashlib.sha256(serialized({
            "case_id": case.case_id, "root": str(self.root), "revision": state(case).get("revision", 0),
            "rows": rows,
        }).encode()).hexdigest()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            old = dict(db.execute("SELECT key,value FROM meta"))
            if old and (old.get("case_id") != self.case_id or old.get("root") != str(self.root)):
                raise ValueError("Memory index belongs to another run/case")
            db.execute("CREATE TABLE IF NOT EXISTS records "
                       "(id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS contexts "
                       "(id TEXT PRIMARY KEY, snapshot TEXT NOT NULL, body TEXT NOT NULL)")
            if old.get("snapshot") != self.snapshot:
                db.execute("DELETE FROM records")
                db.executemany("INSERT INTO records VALUES (?,?,?)", rows)
                db.execute("DELETE FROM meta")
                db.executemany("INSERT INTO meta VALUES (?,?)", [
                    ("case_id", self.case_id), ("root", str(self.root)), ("snapshot", self.snapshot),
                ])

    def register_context(self, payload):
        """Archive a trusted public Planner payload, never an arbitrary file or model response."""
        if payload.get("case_id") != self.case_id:
            raise ValueError("Context belongs to another case")
        body = serialized(payload)
        rid = "context:" + hashlib.sha256((self.snapshot + body).encode()).hexdigest()[:24]
        with closing(self._connect()) as check:
            check.execute("SELECT 1")
        with closing(sqlite3.connect(self.path)) as db, db:
            current = dict(db.execute("SELECT key,value FROM meta"))
            if current.get("snapshot") != self.snapshot:
                raise ValueError("Memory snapshot changed while registering context")
            db.execute("INSERT OR IGNORE INTO contexts VALUES (?,?,?)", (rid, self.snapshot, body))
        return rid

    def _connect(self):
        db = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
        current = dict(db.execute("SELECT key,value FROM meta"))
        if current.get("snapshot") != self.snapshot or current.get("case_id") != self.case_id:
            db.close()
            raise ValueError("Memory snapshot changed during this decision")
        return db

    def search(self, request: MemoryQuery):
        terms = re.findall(r"\w+", request.query.lower())
        with closing(self._connect()) as db:
            rows = list(db.execute("SELECT id,kind,body FROM records WHERE (? IS NULL OR kind=?)",
                                   (request.kind, request.kind)))
            if request.kind in (None, "context"):
                rows.extend(db.execute("SELECT id,'context',body FROM contexts WHERE snapshot=?", (self.snapshot,)))
        hits = []
        for rid, kind, body in rows:
            text = (rid + " " + body).lower()
            matches = sum(term in text for term in terms)
            if terms and not matches:
                continue
            data = json.loads(body)
            preview = next((data[k] for k in ("claim", "task", "finding", "question", "reason")
                            if isinstance(data.get(k), str)), body)
            hits.append((matches, rid, {
                "record_id": rid, "kind": kind, "status": data.get("status"),
                "capability_id": data.get("capability_id"),
                "preview": preview[:280], "preview_only": True,
            }))
        hits.sort(key=lambda row: (-row[0], row[1]))
        end = request.offset + request.limit
        return {"records": [item[2] for item in hits[request.offset:end]], "total": len(hits),
                "next_offset": end if end < len(hits) else None}

    def read(self, request: MemoryQuery):
        sources = []
        with closing(self._connect()) as db:
            for rid in request.record_ids:
                row = db.execute("SELECT kind,body FROM records WHERE id=?", (rid,)).fetchone()
                if row is None:
                    row = db.execute("SELECT 'context',body FROM contexts WHERE id=? AND snapshot=?",
                                     (rid, self.snapshot)).fetchone()
                sources.append(row)
        budgets = [None] * len(sources)
        if self.focused_reads and len(sources) > 1:
            demands = []
            for row in sources:
                if row is None:
                    demands.append(0)
                elif request.view == "brief":
                    preview = brief_record(json.loads(row[1]), request.paths, offset=request.offset, budget=12000)
                    demands.append(12000 if preview.get("next_offset") is not None else len(serialized(preview)) + 40)
                elif request.view == "findings":
                    preview = findings_record(json.loads(row[1]), offset=request.offset, budget=12000)
                    demands.append(12000 if preview.get("next_offset") is not None else len(serialized(preview)) + 40)
                elif request.view == "outline":
                    demands.append(len(serialized(outline_record(
                        json.loads(row[1]), request.paths, offset=request.offset, budget=12000,
                        shallow=row[0] == "context" and not request.paths,
                    ))))
                elif request.paths:
                    demands.append(len(serialized(selected_record(json.loads(row[1]), request.paths, focused=True))))
                else:
                    demands.append(len(row[1]))
            budgets = allocate_read_budgets(demands)
        results = [
            {"record_id": rid, "error": "unknown_record"} if source is None else
            render_record(rid, source[0], source[1], request, focused=self.focused_reads, budget=budget,
                          brief_auto=self.brief_auto)
            for rid, source, budget in zip(request.record_ids, sources, budgets, strict=True)
        ]
        return {"records": results}
