"""Bounded edits to an uncommitted command, never to the accepted case snapshot."""

import hashlib
import json
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Any, Literal

from mechcal.runtime.field_repair import apply_corrections, correction_contract
from mechcal.runtime.retry import RejectedResponseError

RECORD_KEYS = ("question_id", "support_id", "result_id", "name", "label")


class DraftEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str
    operation: Literal["replace", "remove"] = "replace"
    value: Any = None


class DraftEdits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["repair_judgment"]
    draft_id: str
    edits: list[DraftEdit] = Field(min_length=1)


def _get(data, path):
    for key in path:
        data = data[key]
    return data


def _normalize_path(data, path):
    normalized = []
    for key in path:
        if isinstance(data, list) and isinstance(key, str) and key.isdigit():
            key = int(key)
        normalized.append(key)
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            data = None
    return normalized


def _command_path(issue, draft):
    """Resolve validation locations against this draft, not sparse-array indexes."""
    if issue.get("loc"):
        path = list(issue["loc"])
        # Pydantic discriminated-union tags are schema branches, not JSON keys.
        for index in range(len(path) - 2):
            if (path[index] == "questions" and type(path[index + 1]) is int
                    and path[index + 2] in {"create", "update"}):
                del path[index + 2]
                break
        return path
    path = issue.get("snapshot_path", (issue.get("repair_paths") or [issue.get("path", [])])[0])
    if isinstance(path, str):
        path = [path]  # Only a literal root field; never parse exception prose.
    prefix = [issue["command_field"]] if issue.get("command_field") else []
    body = draft.get(prefix[0], {}) if prefix else draft
    if path == ["portfolio"]:
        path = ["hypotheses"]
    if path[:2] == ["portfolio", "hypotheses"]:
        path = path[1:]
    selectors = {}
    for record in issue.get("record_selectors", []):
        collection = record["collection_path"]
        if collection[:2] == ["portfolio", "hypotheses"]:
            collection = collection[1:]
        selectors[tuple(collection)] = record["selector"]
    if not selectors and issue.get("record_selector") and path:
        selectors[(path[0],)] = issue["record_selector"]
    resolved = []
    for position, part in enumerate(path):
        selector = selectors.get(tuple(path[:position]))
        if isinstance(body, list) and selector:
            matches = [i for i, row in enumerate(body)
                       if isinstance(row, dict) and all(row.get(k) == v for k, v in selector.items())]
            if len(matches) != 1:
                # An omitted record needs an explicit model-authored change.
                return prefix + resolved
            part = matches[0]
        resolved.append(part)
        try:
            body = body[int(part) if isinstance(body, list) else part]
        except (KeyError, IndexError, TypeError, ValueError):
            body = None
    return prefix + resolved


class JudgmentDraft:
    def __init__(self, case_id, revision, stage, model, feedback, *, schema=None):
        self.original = deepcopy(feedback["rejected_response"])
        self.model = model
        self.schema = deepcopy(schema if schema is not None else model.model_json_schema())
        self.feedback = feedback
        self.draft_id = hashlib.sha256(json.dumps({
            "case_id": case_id, "revision": revision, "stage": stage,
            "draft": self.original,
        }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        self.targets = {}
        self.requires_context = any(
            issue.get("requires_scientific_context",
                      ("loc" not in issue and issue.get("code") != "schema_enum")
                      or issue.get("type") == "missing")
            for issue in feedback.get("issues", [])
        )
        issues = feedback.get("issues", [])
        for issue in issues:
            paths = issue.get("repair_paths") if "snapshot_path" not in issue else None
            for allowed_path in paths or [None]:
                target_issue = {**issue, "repair_paths": [allowed_path]} if allowed_path is not None else issue
                self._issue_target(target_issue)
        if not self.targets:
            self.requires_context = True
            self._add([], "replace")
        # Wider authorized targets subsume their leaves. Model output cannot
        # contain overlapping edits or two writes to the same location.
        for key, target in list(self.targets.items()):
            if any(other["path"] != target["path"]
                   and target["path"][:len(other["path"])] == other["path"]
                   for other in self.targets.values()):
                del self.targets[key]

    def _issue_target(self, issue):
        path = _normalize_path(self.original, _command_path(issue, self.original))
        # A missing parent or an unscoped semantic error needs a wider
        # explicit correction, not a guessed leaf or a silent fallback.
        while path:
            try:
                parent = _get(self.original, path[:-1])
                if not isinstance(parent, dict | list):
                    raise KeyError(path[-1])
                if isinstance(parent, list):
                    _get(self.original, path)
                elif path[-1] not in parent and issue.get("type") != "missing":
                    raise KeyError(path[-1])
                break
            except (KeyError, IndexError, TypeError):
                path.pop()
        self._add(path, "remove" if issue.get("type") == "extra_forbidden" else "replace")

    def _add(self, path, operation):
        selector = None
        record_context = None
        record = self.original
        stable = []
        for part in path:
            try:
                record = record[part]
            except (KeyError, IndexError, TypeError):
                record = None
            if isinstance(part, int) and isinstance(record, dict):
                selector = {k: record[k] for k in RECORD_KEYS if k in record} or None
                record_context = record
                stable.append(selector or part)
            else:
                stable.append(part)
        target_id = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]
        self.targets[target_id] = {
            "path": path, "selector": selector, "operation": operation,
            "current_value": record, "field": path[-1] if path else "whole_command",
            "record_context": record_context,
        }

    def payload(self):
        schema = DraftEdits.model_json_schema()
        variants = []
        for target_id, target in self.targets.items():
            value_schema = {}
            if target["operation"] == "replace":
                if not target["path"]:
                    value_schema = deepcopy(self.schema)
                else:
                    contract = correction_contract(self.model, [target["path"]], schema=self.schema)
                    value_schema = contract["properties"]["repairs"]["items"]["anyOf"][0]["properties"]["value"]
                schema["$defs"].update(self.schema.get("$defs", {}))
            properties = {
                "target_id": {"const": target_id},
                "operation": {"const": target["operation"]},
            }
            if target["operation"] == "replace":
                properties["value"] = value_schema
            variants.append({"type": "object", "additionalProperties": False,
                             "required": list(properties), "properties": properties})
        schema["properties"]["draft_id"] = {"const": self.draft_id}
        schema["properties"]["edits"] = {"type": "array", "minItems": 1,
                                          "maxItems": len(self.targets), "items": {"anyOf": variants}}
        return {
            "draft_id": self.draft_id,
            "issues": self.feedback.get("issues", []), "error": self.feedback["error"],
            "targets": [{"target_id": key, **{k: v for k, v in target.items() if k != "path"}}
                        for key, target in self.targets.items()],
            "output_contract": schema,
            "scientific_evidence": False,
        }

    def apply(self, raw):
        try:
            response = DraftEdits.model_validate(raw)
            if response.draft_id != self.draft_id:
                raise ValueError("Stale or foreign draft_id; accepted case state has not changed.")
            ids = [edit.target_id for edit in response.edits]
            if len(ids) != len(set(ids)) or not set(ids) <= self.targets.keys():
                raise ValueError("Use distinct authorized target_id values only.")
            result = deepcopy(self.original)
            for edit in response.edits:
                target = self.targets[edit.target_id]
                if edit.operation != target["operation"]:
                    raise ValueError("The requested operation is not authorized for this target.")
                path = target["path"]
                if edit.operation == "remove":
                    if "value" in edit.model_fields_set:
                        raise ValueError("A remove edit must not supply a value.")
                    del _get(result, path[:-1])[path[-1]]
                else:
                    if "value" not in edit.model_fields_set:
                        raise ValueError("A replace edit must supply value.")
                    if path:
                        result = apply_corrections(result, {
                            "repairs": [{"path": path, "value": edit.value}],
                        }, [path])
                    else:
                        if not isinstance(edit.value, dict):
                            raise ValueError("A whole-command replacement must be an object.")
                        result = deepcopy(edit.value)
            return result
        except (ValidationError, ValueError) as exc:
            raise self.rejected_correction(exc, raw) from exc

    def rejected_correction(self, error, raw):
        failure = RejectedResponseError(str(error), self.original)
        failure.issues = self.feedback.get("issues", [])
        failure.repair_failure = {"submitted_correction": raw, "error": str(error)}
        return failure
