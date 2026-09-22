from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from mechcal.public_ids import public_case_id_from_metadata
from mechcal.runtime.llm import OpenAIJsonClient
from mechcal.runtime.prompts import load_prompt
from mechcal.schemas import (
    CaseRun,
    EvidenceUnit,
    MechanismProgram,
    MechanismSupportArgument,
    PhotophysicsReview,
)

GENERIC_WEB_QUERIES = (
    "aggregation-induced emission photophysics RIM ESIPT ICT review",
    "AIE luminogen restriction of intramolecular motion TICT ESIPT mechanism review",
    "fluorescence aggregation caused quenching excimer pi stacking AIE review",
)
FORBIDDEN_QUERY_TOKENS = {
    "smiles",
    "case_id",
    "hidden_reference",
    "semantic_evidence_targets",
    "semantic_diagnosis_targets",
    "reference_evidence_units",
    "reference_diagnosis_units",
}
DEFAULT_ARBITER_EVIDENCE_LIMIT = 3
DEFAULT_ARBITER_SUPPORT_LIMIT = 4
COMPACT_REVIEW_CONTRACT = (
    "Return JSON with keys review_id, case_id, round_id, coverage_axes, "
    "hypothesis_cards, support_argument_audits, recommended_next_routes, overclaim_warnings, "
    "source_evidence_refs. coverage_axes items need axis_id, label, status "
    "(covered|missing|boundary_only|not_applicable), evidence_refs, rationale, "
    "recommended_routes. hypothesis_cards items need hypothesis_id, mechanism, "
    "status (computed_supported|proxy_supported|plausible|weakened|rejected|"
    "underdetermined), support_evidence_refs, weakening_evidence_refs, "
    "missing_or_unresolved, reasoning_summary, scope_limits, priority. "
    "support_argument_audits items need support_id, audit_status "
    "(accepted|boundary_added|downgraded|unsupported|invalid_ref|overclaim), "
    "optional recommended_support_level (strong|partial|weak|unsupported), "
    "audit_notes, and optional boundary_patch. "
    "Use only allowed_evidence_ids and available_capability_ids."
)


class PhotophysicsArbiterAgent:
    """LLM verifier for AIE photophysics mechanism reviews.

    The Arbiter reviews public SMILES-derived inputs, typed MAS evidence, and
    generic literature context. It does not dispatch tools and does not read
    private answer keys.
    """

    def __init__(
        self,
        *,
        llm_client: OpenAIJsonClient | None = None,
        web_search: GenericPhotophysicsWebSearch | None = None,
        max_attempts: int = 4,
    ) -> None:
        self.llm_client = llm_client or OpenAIJsonClient()
        self.web_search = web_search or GenericPhotophysicsWebSearch()
        self.max_attempts = max(1, max_attempts)

    def review(self, case_run: CaseRun) -> PhotophysicsReview:
        round_id = case_run.current_round_id or (
            case_run.round_ids[-1] if case_run.round_ids else "FINAL"
        )
        evidence_items = _select_evidence_for_review(case_run.evidence_ledger.items)
        compact_evidence_ids = [item.evidence_id for item in evidence_items]
        allowed_evidence_ids = [item.evidence_id for item in case_run.evidence_ledger.items]
        if not self.llm_client.is_configured():
            raise RuntimeError("PhotophysicsArbiterAgent requires a configured LLM client.")

        literature_context = _compact_literature_context(self.web_search.search(case_run))
        payload = {
            "public_case": {
                "case_label": "current_smiles_case",
                "smiles": case_run.input.smiles,
                "user_query": _clip(case_run.input.user_query, 160),
                "round_id": round_id,
            },
            "mechanism_program": _compact_mechanism_program(case_run.mechanism_program),
            "evidence_ledger": [
                _safe_evidence_payload(item)
                for item in evidence_items
            ],
            "evidence_selection_note": (
                "The runtime selected a compact, public, source-grounded evidence "
                "subset for this verifier call. Existing support arguments may "
                "cite any ID listed in allowed_evidence_ids, but new verifier "
                "claims should rely on the compact evidence shown here."
            ),
            "planner_state": {
                "current_hypothesis": case_run.portfolio.current,
                "hypotheses": [
                    {
                        "name": item.name,
                        "confidence": item.confidence,
                        "status": item.status,
                        "rationale": _clip(item.rationale, 100),
                    }
                    for item in case_run.portfolio.sorted_hypotheses()[:6]
                ],
            },
            "mechanism_support_arguments": [
                _compact_support_argument(item)
                for item in case_run.mechanism_support_arguments[
                    -DEFAULT_ARBITER_SUPPORT_LIMIT:
                ]
            ],
            "allowed_support_ids": [
                item.support_id
                for item in case_run.mechanism_support_arguments[
                    -DEFAULT_ARBITER_SUPPORT_LIMIT:
                ]
            ],
            "generic_literature_context": literature_context,
            "compact_evidence_ids": compact_evidence_ids,
            "allowed_evidence_ids": allowed_evidence_ids,
            "available_capability_ids": _available_capability_ids(),
            "output_contract": COMPACT_REVIEW_CONTRACT,
            "policy": {
                "case_specific_literature_search": "forbidden",
                "external_answer_access": "none",
                "final_decision_authority": "Planner only",
            },
        }
        last_error: str | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.llm_client.complete_json(
                    system_prompt=load_prompt("photophysics_arbiter.md"),
                    payload=payload,
                    schema_feedback=last_error,
                )
                review = PhotophysicsReview.model_validate(
                    _normalize_review_response(response)
                )
                return _sanitize_review(
                    review,
                    case_run=case_run,
                    round_id=round_id,
                    allowed_evidence_ids=allowed_evidence_ids,
                    compact_evidence_ids=compact_evidence_ids,
                    literature_context=literature_context,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"Attempt {attempt + 1} failed: {type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"PhotophysicsArbiterAgent failed to produce a valid review: {last_error}"
        )


class GenericPhotophysicsWebSearch:
    """Fixed-query generic mechanism search with case-specific query blocking."""

    def __init__(self, *, timeout_seconds: float = 8.0, max_results: int = 2) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    def search(self, case_run: CaseRun) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        for query in GENERIC_WEB_QUERIES:
            validate_generic_query(query, case_run)
            results.extend(self._search_query(query))
        deduped: dict[str, dict[str, str]] = {}
        for result in results:
            key = result.get("url") or result.get("title") or result.get("query")
            if key:
                deduped[key] = result
        return list(deduped.values())[:6]

    def _search_query(self, query: str) -> list[dict[str, str]]:
        url = "https://duckduckgo.com/html/?q=" + quote_plus(query)
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                html = response.read(120_000).decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            return [
                {
                    "query": query,
                    "title": "generic web search unavailable",
                    "url": "",
                    "snippet": f"{type(exc).__name__}: {exc}",
                }
            ]
        parsed = _parse_duckduckgo_results(html, query=query)
        return parsed[: self.max_results]


class DisabledPhotophysicsWebSearch:
    """No-web search provider for closed-book benchmark runs."""

    def search(self, case_run: CaseRun) -> list[dict[str, str]]:
        return [
            {
                "query": "disabled",
                "title": "generic web search disabled",
                "url": "",
                "snippet": (
                    "Closed-book benchmark mode: no web search or literature "
                    "retrieval was used."
                ),
            }
        ]


def validate_generic_query(query: str, case_run: CaseRun) -> None:
    normalized = query.lower()
    forbidden = set(FORBIDDEN_QUERY_TOKENS)
    forbidden.add(case_run.input.smiles.lower())
    forbidden.add(case_run.case_id.lower())
    for token in _smiles_fragments(case_run.input.smiles):
        forbidden.add(token)
    leaked = [
        token
        for token in forbidden
        if token and len(token) >= 6 and token in normalized
    ]
    if leaked:
        raise ValueError(
            "PhotophysicsArbiter generic web query contains case-specific token(s): "
            + ", ".join(sorted(set(leaked))[:3])
        )


def _parse_duckduckgo_results(html: str, *, query: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    blocks = re.split(r'<div class="result', html)
    for block in blocks[1:]:
        title_match = re.search(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.S,
        )
        if title_match is None:
            continue
        snippet_match = re.search(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            block,
            flags=re.S,
        )
        title = _strip_html(title_match.group(2))
        snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""
        results.append(
            {
                "query": query,
                "title": title,
                "url": unescape(title_match.group(1)),
                "snippet": snippet,
            }
        )
    return results


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"\s+", " ", unescape(text)).strip()
    return text[:500]


def _smiles_fragments(smiles: str) -> set[str]:
    fragments = set(re.findall(r"[A-Za-z0-9@+\\[\\]\\(\\)=#$-]{6,}", smiles.lower()))
    fragments.add(smiles.lower())
    return fragments


def _select_evidence_for_review(
    items: list[EvidenceUnit],
    *,
    limit: int = DEFAULT_ARBITER_EVIDENCE_LIMIT,
) -> list[EvidenceUnit]:
    if len(items) <= limit:
        return list(items)
    selected: list[EvidenceUnit] = []
    selected_ids: set[str] = set()
    seen_capabilities: set[str] = set()
    for item in reversed(items):
        capability_key = item.capability_id or item.agent_name or item.family
        if capability_key in seen_capabilities:
            continue
        selected.append(item)
        selected_ids.add(item.evidence_id)
        seen_capabilities.add(capability_key)
        if len(selected) >= limit // 2:
            break
    for item in reversed(items):
        if item.evidence_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.evidence_id)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: (item.round_id, item.evidence_id))


def _compact_mechanism_program(program: MechanismProgram | None) -> dict[str, Any] | None:
    if program is None:
        return None
    return {
        "candidate_mechanisms": [
            {
                "mechanism_id": item.mechanism_id,
                "label": item.label,
                "mechanism_family": item.mechanism_family,
            }
            for item in program.candidate_mechanisms[:5]
        ],
        "evidence_questions": [
            {
                "question_id": item.question_id,
                "mechanism_id": item.mechanism_id,
                "observable": item.observable,
                "status": item.status,
                "acceptable_capability_ids": list(item.acceptable_capability_ids[:3]),
            }
            for item in program.evidence_questions[:6]
        ],
        "computational_routes": [
            {
                "question_id": item.question_id,
                "capability_id": item.capability_id,
                "agent_name": item.agent_name,
                "route": _clip(item.route, 100),
            }
            for item in program.computational_routes[:8]
        ],
        "scope_boundaries": [
            {
                "boundary_id": item.boundary_id,
                "statement": _clip(item.statement, 120),
            }
            for item in program.scope_boundaries
        ],
    }


def _safe_evidence_payload(item: EvidenceUnit) -> dict[str, Any]:
    metrics = {
        key: _safe_metric_value(value)
        for key, value in item.metrics.items()
        if key not in {"smiles", "prepared_file_paths"}
    }
    return {
        "evidence_id": item.evidence_id,
        "agent_name": item.agent_name,
        "capability_id": item.capability_id,
        "claim": _clip(item.claim, 160),
        "basis": item.basis,
        "support": item.support,
        "family": item.family,
        "status": item.status,
        "observable_tags": list(item.observable_tags[:4]),
        "metrics": dict(list(metrics.items())[:8]),
    }


def _compact_support_argument(item: MechanismSupportArgument) -> dict[str, Any]:
    return {
        "support_id": item.support_id,
        "target_label": item.target_label,
        "support_level": item.support_level,
        "finding": _clip(item.finding, 160),
        "warrant": _clip(item.warrant, 180),
        "boundary": _clip(item.boundary, 160),
        "observation_refs": list(item.observation_refs[:4]),
        "audit_status": item.audit_status,
        "audit_notes": list(item.audit_notes[:3]),
    }


def _compact_literature_context(results: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "query": _clip(item.get("query"), 100),
            "title": _clip(item.get("title"), 120),
            "url": _clip(item.get("url"), 120),
            "snippet": _clip(item.get("snippet"), 160),
        }
        for item in results[:2]
    ]


def _safe_metric_value(value: Any) -> Any:
    if isinstance(value, str):
        return _clip(value, 80)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_metric_value(item) for item in value[:4]]
    if isinstance(value, dict):
        return {
            str(key): _safe_metric_value(item)
            for key, item in list(value.items())[:5]
        }
    return _clip(str(value), 80)


def _normalize_review_response(response: dict[str, Any]) -> dict[str, Any]:
    normalized = _strip_mapping_keys(response)
    normalized.setdefault("review_id", "pending_photophysics_review")
    normalized.setdefault("case_id", "current_smiles_case")
    normalized.setdefault("round_id", "current_round")
    normalized["coverage_axes"] = [
        _normalize_axis(item, index=index)
        for index, item in enumerate(_as_list(normalized.get("coverage_axes")), start=1)
        if isinstance(item, dict)
    ]
    normalized["hypothesis_cards"] = [
        _normalize_card(item)
        for item in _as_list(normalized.get("hypothesis_cards"))
        if isinstance(item, dict)
    ]
    normalized["support_argument_audits"] = [
        _normalize_support_argument_audit(item)
        for item in _as_list(normalized.get("support_argument_audits"))
        if isinstance(item, dict)
    ]
    normalized["recommended_next_routes"] = _as_text_list(
        normalized.get("recommended_next_routes")
    )
    normalized["overclaim_warnings"] = _as_text_list(
        normalized.get("overclaim_warnings")
    )
    normalized["source_evidence_refs"] = _as_text_list(
        normalized.get("source_evidence_refs")
    )
    allowed_top_level_keys = {
        "review_id",
        "case_id",
        "round_id",
        "coverage_axes",
        "hypothesis_cards",
        "support_argument_audits",
        "recommended_next_routes",
        "overclaim_warnings",
        "source_evidence_refs",
    }
    return {
        key: value
        for key, value in normalized.items()
        if key in allowed_top_level_keys
    }


def _normalize_support_argument_audit(item: dict[str, Any]) -> dict[str, Any]:
    audit = _strip_mapping_keys(item)
    audit["audit_notes"] = _as_text_list(audit.get("audit_notes"))
    audit["audit_status"] = _normalize_support_audit_status(
        audit.get("audit_status")
    )
    recommended = _normalize_support_level(audit.get("recommended_support_level"))
    if recommended is None:
        audit.pop("recommended_support_level", None)
    else:
        audit["recommended_support_level"] = recommended
    if audit.get("boundary_patch") is not None:
        audit["boundary_patch"] = _clip(str(audit.get("boundary_patch")), 400)
    return {
        key: audit[key]
        for key in (
            "support_id",
            "audit_status",
            "recommended_support_level",
            "audit_notes",
            "boundary_patch",
        )
        if key in audit
    }


def _normalize_support_audit_status(value: Any) -> str:
    text = str(value or "accepted").strip().lower()
    aliases = {
        "ok": "accepted",
        "valid": "accepted",
        "needs_boundary": "boundary_added",
        "boundary_needed": "boundary_added",
        "too_strong": "downgraded",
        "not_supported": "unsupported",
        "invalid_refs": "invalid_ref",
        "over_claim": "overclaim",
    }
    normalized = aliases.get(text, text)
    allowed = {
        "accepted",
        "boundary_added",
        "downgraded",
        "unsupported",
        "invalid_ref",
        "overclaim",
    }
    return normalized if normalized in allowed else "accepted"


def _normalize_support_level(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "proxy": "partial",
        "proxy_supported": "partial",
        "computed_supported": "strong",
        "plausible": "weak",
        "underdetermined": "weak",
        "not_supported": "unsupported",
    }
    normalized = aliases.get(text, text)
    allowed = {"strong", "partial", "weak", "unsupported"}
    return normalized if normalized in allowed else None


def _normalize_axis(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    axis = _strip_mapping_keys(item)
    if not str(axis.get("axis_id") or "").strip():
        axis["axis_id"] = f"AX{index:02d}"
    if not str(axis.get("label") or "").strip():
        axis["label"] = "Unlabeled photophysics coverage axis"
    if not str(axis.get("rationale") or "").strip():
        axis["rationale"] = (
            "LLM omitted rationale; keep this axis as a status-only verifier note."
        )
    axis["evidence_refs"] = _as_text_list(axis.get("evidence_refs"))
    axis["recommended_routes"] = _as_text_list(axis.get("recommended_routes"))
    axis["status"] = _normalize_axis_status(axis.get("status"))
    return {
        key: axis[key]
        for key in (
            "axis_id",
            "label",
            "status",
            "evidence_refs",
            "rationale",
            "recommended_routes",
        )
        if key in axis
    }


def _normalize_axis_status(value: Any) -> str:
    text = str(value or "missing").strip().lower()
    aliases = {
        "complete": "covered",
        "present": "covered",
        "satisfied": "covered",
        "unresolved": "missing",
        "underdetermined": "missing",
        "unknown": "missing",
        "needs_evidence": "missing",
        "requires_experiment": "boundary_only",
        "experiment_required": "boundary_only",
        "out_of_scope": "not_applicable",
    }
    normalized = aliases.get(text, text)
    allowed = {"covered", "missing", "boundary_only", "not_applicable"}
    return normalized if normalized in allowed else "missing"


def _normalize_card(item: dict[str, Any]) -> dict[str, Any]:
    card = _strip_mapping_keys(item)
    for field in (
        "support_evidence_refs",
        "weakening_evidence_refs",
        "missing_or_unresolved",
        "scope_limits",
    ):
        card[field] = _as_text_list(card.get(field))
    card["priority"] = _normalize_priority(card.get("priority", 0.0))
    card["status"] = _normalize_card_status(card.get("status"))
    if (
        card["status"]
        in {"computed_supported", "proxy_supported", "weakened", "rejected"}
        and not card["support_evidence_refs"]
        and not card["weakening_evidence_refs"]
    ):
        card["status"] = "underdetermined"
        if not card["missing_or_unresolved"]:
            card["missing_or_unresolved"] = [
                "Needs source-grounded runtime evidence before stronger assessment."
            ]
    if card["status"] == "underdetermined" and not card["missing_or_unresolved"]:
        card["missing_or_unresolved"] = [
            "Needs additional source-grounded runtime evidence."
        ]
    return {
        key: card[key]
        for key in (
            "hypothesis_id",
            "mechanism",
            "status",
            "support_evidence_refs",
            "weakening_evidence_refs",
            "missing_or_unresolved",
            "reasoning_summary",
            "scope_limits",
            "priority",
        )
        if key in card
    }


def _normalize_card_status(value: Any) -> str:
    normalized = str(value or "underdetermined").strip().lower()
    aliases = {
        "supported": "proxy_supported",
        "structure_supported": "proxy_supported",
        "computed": "computed_supported",
        "weakens": "weakened",
        "unsupported": "underdetermined",
        "unknown": "underdetermined",
        "uncertain": "underdetermined",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "computed_supported",
        "proxy_supported",
        "plausible",
        "weakened",
        "rejected",
        "underdetermined",
    }
    return normalized if normalized in allowed else "underdetermined"


def _strip_mapping_keys(value: dict[str, Any]) -> dict[str, Any]:
    stripped: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key).strip()
        if isinstance(item, dict):
            stripped[normalized_key] = _strip_mapping_keys(item)
        elif isinstance(item, list):
            stripped[normalized_key] = [
                _strip_mapping_keys(element) if isinstance(element, dict) else element
                for element in item
            ]
        else:
            stripped[normalized_key] = item
    return stripped


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_text_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in _as_list(value):
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _normalize_priority(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().lower()
    mapped = {
        "high": 0.85,
        "medium": 0.5,
        "moderate": 0.5,
        "low": 0.2,
    }
    if text in mapped:
        return mapped[text]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _clip(value: str | None, limit: int) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _available_capability_ids() -> list[str]:
    return [
        "macro.screen_esipt_structural_motif",
        "macro.screen_donor_acceptor_layout",
        "macro.screen_rotor_torsion_topology",
        "macro.screen_aggregation_prone_scaffold",
        "macro.screen_pi_stacking_prone_geometry",
        "macro.run_solid_state_emission_proxy",
        "macro.run_crystal_restriction_checklist",
        "microscopic.run_torsion_snapshots",
        "microscopic.run_conformer_bundle",
        "microscopic.run_targeted_localized_orbital_analysis",
        "microscopic.run_targeted_natural_orbital_analysis",
        "microscopic.unsupported_excited_state_relaxation",
    ]


def _sanitize_review(
    review: PhotophysicsReview,
    *,
    case_run: CaseRun,
    round_id: str,
    allowed_evidence_ids: list[str],
    compact_evidence_ids: list[str],
    literature_context: list[dict[str, str]],
) -> PhotophysicsReview:
    allowed = set(allowed_evidence_ids)
    axes = [
        axis.model_copy(
            update={
                "evidence_refs": [ref for ref in axis.evidence_refs if ref in allowed],
                "recommended_routes": [
                    route
                    for route in axis.recommended_routes
                    if route in _available_capability_ids()
                ],
            }
        )
        for axis in review.coverage_axes
    ]
    cards = [
        card.model_copy(
            update={
                "support_evidence_refs": [
                    ref for ref in card.support_evidence_refs if ref in allowed
                ],
                "weakening_evidence_refs": [
                    ref for ref in card.weakening_evidence_refs if ref in allowed
                ],
            }
        )
        for card in review.hypothesis_cards
    ]
    allowed_support_ids = {
        item.support_id for item in case_run.mechanism_support_arguments
    }
    support_argument_audits = [
        audit
        for audit in review.support_argument_audits
        if audit.support_id in allowed_support_ids
    ]
    warnings = [
        *review.overclaim_warnings,
        "SupportAuditor used only generic literature-search context; "
        "case-specific literature retrieval is forbidden.",
    ]
    literature_note = "Generic web context unavailable."
    if literature_context:
        literature_note = "Generic web context: " + "; ".join(
            str(item.get("title") or item.get("snippet") or item.get("query"))
            for item in literature_context[:3]
        )
    warnings.append(literature_note[:800])
    public_case_id = _public_case_id(case_run)
    return review.model_copy(
        update={
            "review_id": f"{public_case_id}:{round_id}:photophysics_review",
            "case_id": public_case_id,
            "round_id": round_id,
            "coverage_axes": axes,
            "hypothesis_cards": cards,
            "support_argument_audits": support_argument_audits,
            "recommended_next_routes": [
                route
                for route in review.recommended_next_routes
                if route in _available_capability_ids()
            ],
            "overclaim_warnings": list(dict.fromkeys(warnings)),
            "source_evidence_refs": compact_evidence_ids,
        }
    )


def _public_case_id(case_run: CaseRun) -> str:
    return public_case_id_from_metadata(case_run.input.metadata)
