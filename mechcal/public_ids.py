from __future__ import annotations

from typing import Any

PUBLIC_CASE_ID_METADATA_KEY = "public_case_id"
DEFAULT_PUBLIC_CASE_ID = "current_smiles_case"


def anonymized_case_id(index: int) -> str:
    return f"CASE_{index:03d}"


def normalize_public_case_id(value: Any) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_PUBLIC_CASE_ID


def public_case_id_from_metadata(metadata: dict[str, Any] | None) -> str:
    metadata = metadata or {}
    return normalize_public_case_id(metadata.get(PUBLIC_CASE_ID_METADATA_KEY))
