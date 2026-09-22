"""Metric API example with synthetic labels and support scores, not research data."""

import json

from mechcal.eval.mechanism_ranking import score_mechanism_ranking


def main() -> None:
    predictions = [
        {"prediction_id": "P001", "rank": 1, "label": "RIM_RIR_RIV"},
        {"prediction_id": "P002", "rank": 2, "label": "ICT_TICT_CT"},
        {"prediction_id": "P003", "rank": 3, "label": "ESIPT_PT"},
    ]
    references = [
        {"label": "RIM_RIR_RIV", "role": "primary", "gain": 2},
        {"label": "ICT_TICT_CT", "role": "secondary", "gain": 1},
    ]
    result = score_mechanism_ranking(
        predictions, references, support_scores={"P001": 0.5, "P002": 1.0, "P003": 0.0}
    )
    print(json.dumps({"synthetic_example": True, "scores": result}, indent=2))


if __name__ == "__main__":
    main()
