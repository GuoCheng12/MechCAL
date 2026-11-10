"""Request-local short citations; persisted evidence IDs are never renamed."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from mechcal.runtime.retry import ModelContractError

REFERENCE_FIELDS = {
    "evidence_refs", "observation_refs", "source_evidence_refs",
    "support_evidence_refs", "weakening_evidence_refs", "evidence_refs_added",
    "new_evidence_refs", "positive_evidence_refs", "negative_evidence_refs",
}


class EvidenceReferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    short_to_full: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_ids(cls, evidence_ids: list[str]) -> EvidenceReferences:
        ids = list(dict.fromkeys(evidence_ids))
        reserved = set(ids)
        mapping: dict[str, str] = {}
        ordinal = 1
        for evidence_id in ids:
            # Preserve existing short IDs in older runs and fixtures.
            if re.fullmatch(r"E\d+", evidence_id):
                mapping[evidence_id] = evidence_id
                continue
            while f"E{ordinal:03d}" in reserved or f"E{ordinal:03d}" in mapping:
                ordinal += 1
            mapping[f"E{ordinal:03d}"] = evidence_id
            ordinal += 1
        return cls(short_to_full=mapping)

    def encode(self, value: Any) -> Any:
        return _rewrite(value, {full: short for short, full in self.short_to_full.items()})

    def decode(self, value: Any) -> Any:
        # Validate before normalizers can silently discard unrecognized citations.
        known = set(self.short_to_full) | set(self.short_to_full.values())
        invalid = sorted(_references(value) - known)
        if invalid:
            raise ModelContractError(
                "Unknown evidence reference(s): " + ", ".join(invalid)
                + ". Copy exact supplied evidence IDs; do not reconstruct them. Available: "
                + (", ".join(self.short_to_full) or "none")
            )
        return _rewrite(value, self.short_to_full)


def _references(value: Any) -> set[str]:
    if isinstance(value, dict):
        refs: set[str] = set()
        for key, item in value.items():
            if key in REFERENCE_FIELDS:
                values = item if isinstance(item, list) else [item]
                refs.update(v for v in values if isinstance(v, str) and v)
            refs.update(_references(item))
        return refs
    if isinstance(value, list):
        return set().union(*(_references(item) for item in value)) if value else set()
    return set()


def _rewrite(value: Any, mapping: dict[str, str]) -> Any:
    if not mapping:
        return value
    # Token boundaries prevent E001 from changing E0010 or artifact:E001.
    pattern = re.compile(
        r"(?<![\w:.-])(?:" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        + r")(?![\w:-]|\.\w)"
    )

    def walk(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: walk(child) for key, child in item.items()}
        if isinstance(item, list):
            return [walk(child) for child in item]
        if isinstance(item, str):
            return pattern.sub(lambda match: mapping[match.group()], item)
        return item

    return walk(value)
