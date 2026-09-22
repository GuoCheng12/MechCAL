from __future__ import annotations

import json
import math
import os
from pathlib import Path

from mechcal.eval.prompts import load_eval_prompt
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient

MECHANISM_EVIDENCE_RUBRIC_PATH = (
    Path(__file__).resolve().parent / "rubrics" / "mechanism_evidence_rubric_v2.json"
)
MECHANISM_EVIDENCE_SUPPORT_PROMPT = "mechanism_evidence_support_judge.md"
MECHANISM_EVIDENCE_SUPPORT_PROMPT_VERSION = "2026-06-05"
SUPPORT_SCORE_LEVELS = (0.0, 0.5, 1.0)
DEFAULT_MECHANISM_SUPPORT_JUDGE_MODEL = "deepseek-v4-flash"


class LLMMechanismEvidenceSupportJudge:
    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
        rubric: dict[str, object] | None = None,
    ) -> None:
        self.client = client or OpenAIJsonClient(_support_judge_settings(settings))
        self.rubric = rubric or load_mechanism_evidence_rubric()
        self.max_schema_attempts = 3

    @property
    def model(self) -> str:
        return self.client.settings.model

    def score(self, prediction: dict[str, object]) -> dict[str, object]:
        normalized = _normalize_prediction(prediction, index=1)
        label = str(normalized["label"])
        label_rubric = _rubric_labels(self.rubric).get(label)
        if label_rubric is None:
            return {
                "prediction_id": normalized["prediction_id"],
                "label": label,
                "evidence_support_score": 0.0,
                "rationale": "Predicted label is outside the benchmark mechanism pool.",
                "judge_model": self.model,
                "judge_prompt_version": MECHANISM_EVIDENCE_SUPPORT_PROMPT_VERSION,
            }
        payload = {
            "mechanism_evidence_rubric_version": self.rubric.get(
                "mechanism_evidence_rubric_version"
            ),
            "support_score_scale": self.rubric.get("support_score_scale"),
            "predicted_mechanism": _compact_support_judge_prediction(normalized),
            "mechanism_rubric": label_rubric,
        }
        raw = self._complete_support_json(payload)
        score = _validated_support_score(raw)
        return {
            "prediction_id": normalized["prediction_id"],
            "label": label,
            "evidence_support_score": score,
            "rationale": str(raw.get("rationale") or ""),
            "judge_model": self.model,
            "judge_prompt_version": MECHANISM_EVIDENCE_SUPPORT_PROMPT_VERSION,
            "raw_judge_response": raw,
        }

    def _complete_support_json(self, payload: dict[str, object]) -> dict[str, object]:
        feedback: str | None = None
        last_error: Exception | None = None
        for _ in range(self.max_schema_attempts):
            try:
                raw = self.client.complete_json(
                    system_prompt=load_eval_prompt(MECHANISM_EVIDENCE_SUPPORT_PROMPT),
                    payload=payload,
                    schema_feedback=feedback,
                )
                _validated_support_score(raw)
                return raw
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                feedback = (
                    "Return exactly one JSON object with keys "
                    "evidence_support_score and rationale. Do not include prose "
                    f"outside JSON. Previous error: {type(exc).__name__}: {exc}"
                )
        raise last_error or RuntimeError("Support judge failed without an exception.")


def _support_judge_settings(
    settings: OpenAICompatibleSettings | None,
) -> OpenAICompatibleSettings | None:
    if settings is None:
        return None
    judge_base_url = (
        os.environ.get("MECHCAL_SUPPORT_JUDGE_BASE_URL", "").strip()
        or settings.base_url
    )
    judge_api_key = (
        os.environ.get("MECHCAL_SUPPORT_JUDGE_API_KEY", "").strip()
        or settings.api_key
    )
    judge_model = os.environ.get(
        "MECHCAL_SUPPORT_JUDGE_MODEL",
        DEFAULT_MECHANISM_SUPPORT_JUDGE_MODEL,
    ).strip()
    if not judge_model:
        judge_model = settings.model
    return OpenAICompatibleSettings(
        base_url=judge_base_url,
        model=judge_model,
        api_key=judge_api_key,
        reasoning_effort=settings.reasoning_effort,
        timeout_seconds=float(
            os.environ.get(
                "MECHCAL_SUPPORT_JUDGE_TIMEOUT",
                str(max(settings.timeout_seconds, 90.0)),
            )
        ),
        max_retries=settings.max_retries,
        max_tokens=900 if settings.max_tokens is None else max(settings.max_tokens, 900),
        trust_env=settings.trust_env,
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=settings.seed,
    )


def _support_score_value(raw: dict[str, object]) -> object | None:
    for key in (
        "evidence_support_score",
        "support_score",
        "evidence_score",
        "score",
    ):
        value = raw.get(key)
        if value is not None:
            return value
    for key in ("result", "judgement", "judgment", "assessment"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            value = _support_score_value(nested)
            if value is not None:
                return value
    return None


def _validated_support_score(raw: dict[str, object]) -> float:
    score_value = _support_score_value(raw)
    if score_value is None:
        raise ValueError("Support judge JSON must include evidence_support_score.")
    score = _number(score_value)
    if score is None or not math.isfinite(score):
        raise ValueError("Support judge evidence_support_score must be numeric.")
    if score not in SUPPORT_SCORE_LEVELS:
        raise ValueError(
            "Support judge evidence_support_score must be one of 0.0, 0.5, or 1.0."
        )
    return score


def _compact_support_judge_prediction(
    normalized: dict[str, object],
) -> dict[str, object]:
    return {
        "prediction_id": normalized["prediction_id"],
        "label": normalized["label"],
        "confidence": normalized["confidence"],
        "claim_status": normalized["claim_status"],
        "support_strength": normalized["support_strength"],
        "evidence_support": normalized["evidence_support"],
        "evidence": _text_list(normalized.get("evidence"), limit=3, max_chars=700),
    }


def _text_list(value: object, *, limit: int, max_chars: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if max_chars is not None and len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        items.append(text)
        if len(items) >= limit:
            break
    return items


def load_mechanism_evidence_rubric(path: Path | None = None) -> dict[str, object]:
    rubric_path = path or MECHANISM_EVIDENCE_RUBRIC_PATH
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    labels = _rubric_labels(rubric)
    missing = sorted(set(MECHANISM_POOL) - set(labels))
    extra = sorted(set(labels) - set(MECHANISM_POOL))
    if missing or extra:
        raise ValueError(
            "Mechanism evidence rubric labels do not match the mechanism pool: "
            f"missing={missing}, extra={extra}"
        )
    return rubric


def judge_mechanism_prediction_support(
    predictions: list[dict[str, object]],
    *,
    judge: LLMMechanismEvidenceSupportJudge,
) -> dict[str, object]:
    normalized_predictions = _normalize_predictions(predictions)
    judgements = [judge.score(prediction) for prediction in normalized_predictions]
    return {
        "support_scores": {
            str(item["prediction_id"]): float(item["evidence_support_score"])
            for item in judgements
        },
        "judgements": judgements,
    }


def score_mechanism_ranking(
    predictions: list[dict[str, object]],
    reference_mechanisms: list[dict[str, object]],
    *,
    support_scores: dict[str, float] | None = None,
    k: int = 3,
) -> dict[str, object]:
    ranked_predictions = _normalize_predictions(predictions)
    reference_gains = _reference_gains(reference_mechanisms)
    primary_reference_labels = _primary_reference_labels(reference_mechanisms)
    top_k = ranked_predictions[:k]
    top_labels = [str(item["label"]) for item in top_k]
    reference_labels = set(reference_gains)

    top1_label = top_labels[0] if top_labels else None
    top1_primary = (
        1.0 if top1_label is not None and top1_label in primary_reference_labels else 0.0
    )
    top1_any = 1.0 if top1_label is not None and top1_label in reference_labels else 0.0
    recall_at_k = (
        len(set(top_labels) & reference_labels) / len(reference_labels)
        if reference_labels
        else 0.0
    )
    ndcg_at_k, dcg_at_k, idcg_at_k = _ndcg(
        ranked_predictions,
        reference_gains=reference_gains,
        support_scores=None,
        k=k,
    )

    if support_scores is None:
        es_ndcg_at_k = None
        support_at_top1 = None
        unsupported_strong_claim_rate_at_k = None
        claim_safety_at_k = None
        ucr_at_k = None
        top1_support_score = None
        credibility_gap_at_k = None
    else:
        es_ndcg_at_k, _, _ = _ndcg(
            ranked_predictions,
            reference_gains=reference_gains,
            support_scores=support_scores,
            k=k,
        )
        top1_support_score = _support_score(top_k[0], support_scores) if top_k else None
        support_at_top1 = (
            1.0 if top1_support_score is not None and top1_support_score >= 0.5 else 0.0
        )
        unsupported_strong_count = sum(
            1
            for item in top_k
            if _is_strong_mechanism_claim(item)
            and _support_score(item, support_scores) == 0.0
        )
        unsupported_strong_claim_rate_at_k = (
            unsupported_strong_count / len(top_k) if top_k else 0.0
        )
        claim_safety_at_k = 1.0 - unsupported_strong_claim_rate_at_k
        ucr_at_k = unsupported_strong_claim_rate_at_k
        credibility_gap_at_k = ndcg_at_k - es_ndcg_at_k

    return {
        "k": k,
        "top1_primary": top1_primary,
        "top1_any": top1_any,
        "top1_hit": top1_any,
        f"recall_at_{k}": recall_at_k,
        f"ndcg_at_{k}": ndcg_at_k,
        f"es_ndcg_at_{k}": es_ndcg_at_k,
        "support_at_top1": support_at_top1,
        f"unsupported_strong_claim_rate_at_{k}": unsupported_strong_claim_rate_at_k,
        f"claim_safety_at_{k}": claim_safety_at_k,
        f"ucr_at_{k}": ucr_at_k,
        f"credibility_gap_at_{k}": credibility_gap_at_k,
        "details": {
            "ranked_predictions": ranked_predictions,
            "reference_gains": reference_gains,
            "primary_reference_labels": sorted(primary_reference_labels),
            "top1_support_score": top1_support_score,
            f"dcg_at_{k}": dcg_at_k,
            f"idcg_at_{k}": idcg_at_k,
        },
    }


def _rubric_labels(rubric: dict[str, object]) -> dict[str, object]:
    labels = rubric.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("Mechanism evidence rubric must contain a labels object.")
    return labels


def _normalize_predictions(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    seen_labels: set[str] = set()
    for index, item in enumerate(predictions, start=1):
        prediction = _normalize_prediction(item, index=index)
        label = str(prediction["label"])
        if not label or label in seen_labels:
            continue
        seen_labels.add(label)
        normalized.append(prediction)
    return normalized


def _normalize_prediction(item: dict[str, object], *, index: int) -> dict[str, object]:
    label = _string(item.get("label") or item.get("mechanism") or item.get("mechanism_label"))
    evidence_items = _evidence_items(item)
    return {
        "prediction_id": _string(item.get("prediction_id") or item.get("id")) or f"P{index:03d}",
        "label": label or "",
        "confidence": _first_number(
            item.get("differential_priority"),
            item.get("plausibility_confidence"),
            item.get("confidence"),
        ),
        "differential_priority": _number(item.get("differential_priority")),
        "evidence": evidence_items,
        "claim_status": _string(item.get("claim_status")) or "likely",
        "support_strength": _string(item.get("support_strength")),
        "evidence_support": _string(item.get("evidence_support")),
    }


def _evidence_items(item: dict[str, object]) -> list[str]:
    evidence_details = item.get("evidence_details")
    if isinstance(evidence_details, list):
        detail_items = []
        for value in evidence_details:
            text = _evidence_detail_text(value)
            if text.strip():
                detail_items.append(text)
        if detail_items:
            return detail_items
    evidence_items: list[str] = []
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        evidence_items.extend(str(value) for value in evidence if str(value).strip())
    elif _string(evidence):
        evidence_items.append(_string(evidence) or "")
    return [value for value in evidence_items if value.strip()]


def _evidence_detail_text(value: object) -> str:
    if not isinstance(value, dict):
        return str(value)
    finding = _string(value.get("finding")) or ""
    warrant = _string(value.get("warrant")) or ""
    boundary = _string(value.get("boundary")) or ""
    return f"Finding: {finding} Warrant: {warrant} Boundary: {boundary}"


def _reference_gains(reference_mechanisms: list[dict[str, object]]) -> dict[str, float]:
    gains: dict[str, float] = {}
    for item in reference_mechanisms:
        label = _string(item.get("label"))
        if label is None:
            continue
        gain = _number(item.get("gain"))
        if gain is None:
            gain = 2.0 if _string(item.get("role")) == "primary" else 1.0
        gains[label] = max(gains.get(label, 0.0), gain)
    return gains


def _primary_reference_labels(reference_mechanisms: list[dict[str, object]]) -> set[str]:
    labels: set[str] = set()
    for item in reference_mechanisms:
        label = _string(item.get("label"))
        role = _string(item.get("role"))
        if label is not None and role in {"primary", "co_primary"}:
            labels.add(label)
    return labels


def _is_strong_mechanism_claim(item: dict[str, object]) -> bool:
    claim_status = _normalized_token(item.get("claim_status"))
    support_strength = _normalized_token(item.get("support_strength"))
    if claim_status in {
        "candidate_requires_validation",
        "underdetermined",
        "weak_or_proxy",
        "weak_candidate",
    }:
        return False
    if support_strength in {"weak_or_proxy", "weak", "proxy", "underdetermined"}:
        return False
    return claim_status in {
        "computed_supported",
        "proxy_supported",
        "supported",
        "supported_claim",
        "likely",
        "final",
        "primary",
    }


def _ndcg(
    predictions: list[dict[str, object]],
    *,
    reference_gains: dict[str, float],
    support_scores: dict[str, float] | None,
    k: int,
) -> tuple[float, float, float]:
    dcg = 0.0
    for rank, item in enumerate(predictions[:k], start=1):
        gain = reference_gains.get(str(item["label"]), 0.0)
        if support_scores is not None:
            gain *= _support_score(item, support_scores)
        dcg += gain / math.log2(rank + 1)
    ideal_gains = sorted(reference_gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return (dcg / idcg if idcg else 0.0, dcg, idcg)


def _support_score(item: dict[str, object], support_scores: dict[str, float]) -> float:
    prediction_id = str(item["prediction_id"])
    label = str(item["label"])
    raw = support_scores.get(prediction_id)
    if raw is None:
        raw = support_scores.get(label, 0.0)
    return _quantize_support_score(raw)


def _quantize_support_score(value: float) -> float:
    clamped = max(0.0, min(1.0, value))
    return min(SUPPORT_SCORE_LEVELS, key=lambda level: (abs(level - clamped), level))


def _number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _first_number(*values: object) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _string(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _normalized_token(value: object) -> str:
    return str(value or "").strip().lower()
