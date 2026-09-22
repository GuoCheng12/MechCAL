from __future__ import annotations

COMMON_PRIOR_AT3_LABELS = (
    "RIM_RIR_RIV",
    "PACKING_HOST_MATRIX_CONFINEMENT",
    "ICT_TICT_CT",
)


def common_prior_predictions(top_k: int = 3) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    for rank, label in enumerate(COMMON_PRIOR_AT3_LABELS[:top_k], start=1):
        predictions.append(
            {
                "prediction_id": f"P{rank:03d}",
                "rank": rank,
                "label": label,
                "confidence": _prior_confidence(rank),
                "claim_status": "candidate_requires_validation",
                "support_strength": "unsupported",
                "evidence": [
                    (
                        "Global common-prior baseline only; this is not "
                        "case-specific evidence."
                    )
                ],
                "limitations": [
                    (
                        "This baseline ignores the input structure and is intended "
                        "only as a prior-frequency comparator."
                    )
                ],
            }
        )
    return predictions


def common_prior_support_scores(predictions: list[dict[str, object]]) -> dict[str, float]:
    return {str(item["prediction_id"]): 0.0 for item in predictions}


def _prior_confidence(rank: int) -> float:
    return max(0.0, 1.0 - 0.2 * (rank - 1))
