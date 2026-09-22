from __future__ import annotations

import re
from typing import Literal

from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.schemas import AlignmentMetric, NormalizedPrediction
from mechcal.runtime.llm import OpenAICompatibleSettings, OpenAIJsonClient

JUDGE_PROMPT = "semantic_alignment_judge.md"
JUDGE_PROMPT_VERSION = "2026-06-01"
PAIR_MATCH_THRESHOLD = 0.5
PAIR_SCORE_LEVELS = (0.0, 0.5, 1.0)
MATCH_COUNT_PRIORITY = 1_000_000


class LLMSemanticAlignmentJudge:
    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
    ) -> None:
        self.client = client or OpenAIJsonClient(settings)

    @property
    def model(self) -> str:
        return self.client.settings.model

    def score(
        self,
        *,
        metric_name: Literal["EA", "DA"],
        prediction_payload: list[dict[str, object]],
        semantic_targets: list[dict[str, object]],
    ) -> AlignmentMetric:
        try:
            raw = self.client.complete_json(
                system_prompt=load_eval_prompt(JUDGE_PROMPT),
                payload={
                    "metric_name": metric_name,
                    "prediction_payload": prediction_payload,
                    "semantic_targets": semantic_targets,
                },
            )
            prediction_ids = _prediction_ids(prediction_payload)
            target_ids = _target_ids(semantic_targets)
            candidate_pairs = _candidate_pairs(
                raw,
                prediction_ids=prediction_ids,
                target_ids=target_ids,
            )
            candidate_pairs, removed_pairs = _direction_filtered_pairs(
                metric_name=metric_name,
                candidate_pairs=candidate_pairs,
                prediction_payload=prediction_payload,
                semantic_targets=semantic_targets,
                prediction_ids=prediction_ids,
                target_ids=target_ids,
            )
            candidate_pairs, strict_removed_pairs = strict_semantic_filtered_pairs(
                candidate_pairs=candidate_pairs,
                prediction_payload=prediction_payload,
                semantic_targets=semantic_targets,
                prediction_ids=prediction_ids,
                target_ids=target_ids,
            )
            matched_pairs = _one_to_one_matches(candidate_pairs)
            matched_pair_count = len(matched_pairs)
            precision, recall, f1 = _precision_recall_f1(
                matched_prediction_count=matched_pair_count,
                predicted_positive_count=len(prediction_payload),
                matched_target_count=matched_pair_count,
                reference_target_count=len(semantic_targets),
            )
            return AlignmentMetric(
                metric_name=metric_name,
                status="scored",
                score=_clamp01(f1),
                precision=_clamp01(precision),
                recall=_clamp01(recall),
                f1=_clamp01(f1),
                predicted_positive_count=len(prediction_payload),
                reference_target_count=len(semantic_targets),
                matched_prediction_count=matched_pair_count,
                matched_target_count=matched_pair_count,
                true_positives=matched_pair_count,
                false_positives=len(prediction_payload) - matched_pair_count,
                false_negatives=len(semantic_targets) - matched_pair_count,
                target_count=len(semantic_targets),
                rationale=str(raw.get("rationale") or ""),
                judge_model=self.model,
                judge_prompt_version=JUDGE_PROMPT_VERSION,
                details={
                    **_judge_details(raw),
                    "matching_method": "one_to_one_max_cardinality_then_score",
                    "match_threshold": PAIR_MATCH_THRESHOLD,
                    "candidate_pairs": candidate_pairs,
                    "direction_filter_removed_pairs": removed_pairs,
                    "strict_filter_removed_pairs": strict_removed_pairs,
                    "matched_pairs": matched_pairs,
                    "missed_targets": _missed_targets(semantic_targets, matched_pairs),
                    "unmatched_predictions": _unmatched_predictions(
                        prediction_payload, matched_pairs
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return AlignmentMetric(
                metric_name=metric_name,
                status="judge_failed",
                score=None,
                target_count=len(semantic_targets),
                rationale=str(exc),
                judge_model=self.model,
                judge_prompt_version=JUDGE_PROMPT_VERSION,
            )


def score_prediction(
    prediction: NormalizedPrediction,
    hidden_reference: dict[str, object] | None,
    *,
    judge: LLMSemanticAlignmentJudge | None = None,
) -> list[AlignmentMetric]:
    evidence_targets = _semantic_targets(hidden_reference, "semantic_evidence_targets")
    diagnosis_targets = _semantic_targets(hidden_reference, "semantic_diagnosis_targets")
    return [
        _score_one(
            metric_name="EA",
            prediction_payload=[
                _compact_evidence_payload(item.model_dump(mode="json"))
                for item in prediction.evidence_units
            ],
            targets=evidence_targets,
            judge=judge,
        ),
        _score_one(
            metric_name="DA",
            prediction_payload=[
                _compact_diagnosis_payload(item.model_dump(mode="json"))
                for item in prediction.diagnosis_units
            ],
            targets=diagnosis_targets,
            judge=judge,
        ),
    ]


def _compact_evidence_payload(item: dict[str, object]) -> dict[str, object]:
    return {
        "evidence_id": item.get("evidence_id"),
        "claim": item.get("claim"),
        "context": item.get("context"),
        "basis": item.get("basis"),
        "support": item.get("support"),
        "status": item.get("status"),
        "limits": item.get("limits") or [],
    }


def _compact_diagnosis_payload(item: dict[str, object]) -> dict[str, object]:
    return {
        "diagnosis_id": item.get("diagnosis_id"),
        "mechanism": item.get("mechanism"),
        "context": item.get("context"),
        "status": item.get("status"),
        "evidence_refs": item.get("evidence_refs") or [],
        "missing_or_unresolved": item.get("missing_or_unresolved") or [],
        "reasoning_summary": item.get("reasoning_summary"),
        "scope_limits": item.get("scope_limits") or [],
    }


def _score_one(
    *,
    metric_name: Literal["EA", "DA"],
    prediction_payload: list[dict[str, object]],
    targets: list[dict[str, object]],
    judge: LLMSemanticAlignmentJudge | None,
) -> AlignmentMetric:
    if not targets:
        return AlignmentMetric(
            metric_name=metric_name,
            status="reference_missing",
            score=None,
            target_count=0,
            rationale=(
                f"{metric_name} was not scored because no semantic "
                "targets were supplied in hidden_reference."
            ),
        )
    if judge is None:
        return AlignmentMetric(
            metric_name=metric_name,
            status="judge_failed",
            score=None,
            target_count=len(targets),
            rationale="No semantic alignment judge was provided.",
        )
    return judge.score(
        metric_name=metric_name,
        prediction_payload=prediction_payload,
        semantic_targets=targets,
    )


def _semantic_targets(
    hidden_reference: dict[str, object] | None,
    key: str,
) -> list[dict[str, object]]:
    if not hidden_reference:
        return []
    raw = hidden_reference.get(key)
    if isinstance(raw, list):
        return [
            item
            for item in raw
            if isinstance(item, dict) and _is_current_scoring_target(item)
        ]
    return []


def _is_current_scoring_target(item: dict[str, object]) -> bool:
    if any(key in item for key in ("access", "capabilities", "target", "rubric")):
        return False
    return _string(item.get("scoring_role")) in {"primary_score", "boundary_score"}


def _prediction_ids(prediction_payload: list[dict[str, object]]) -> list[str]:
    ids: list[str] = []
    for index, item in enumerate(prediction_payload, start=1):
        ids.append(
            _string(
                item.get("evidence_id")
                or item.get("diagnosis_id")
                or item.get("prediction_id")
                or item.get("id")
            )
            or f"P{index:03d}"
        )
    return ids


def _target_ids(semantic_targets: list[dict[str, object]]) -> list[str]:
    ids: list[str] = []
    for index, item in enumerate(semantic_targets, start=1):
        ids.append(
            _string(
                item.get("target_id")
                or item.get("evidence_id")
                or item.get("diagnosis_id")
                or item.get("id")
            )
            or f"T{index:03d}"
        )
    return ids


def _candidate_pairs(
    raw: dict[str, object],
    *,
    prediction_ids: list[str],
    target_ids: list[str],
) -> list[dict[str, object]]:
    raw_pairs = _raw_pair_list(raw)
    valid_prediction_ids = set(prediction_ids)
    valid_target_ids = set(target_ids)
    pairs_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        prediction_id = _string(
            item.get("prediction_id")
            or item.get("predicted_id")
            or item.get("prediction")
            or item.get("prediction_item_id")
            or item.get("source_id")
        )
        target_id = _string(
            item.get("target_id")
            or item.get("reference_id")
            or item.get("semantic_target_id")
            or item.get("target")
        )
        if prediction_id not in valid_prediction_ids or target_id not in valid_target_ids:
            continue
        score = _number(
            item.get("alignment_score")
            or item.get("score")
            or item.get("similarity")
            or item.get("match_score")
        )
        if score is None:
            continue
        violation = _bool(
            item.get("violation")
            or item.get("must_not_claim_violation")
            or item.get("violates_must_not_claim")
        )
        normalized = {
            "prediction_id": prediction_id,
            "target_id": target_id,
            "alignment_score": _quantize_pair_score(score),
            "rationale": str(item.get("rationale") or ""),
            "violation": violation,
        }
        key = (prediction_id, target_id)
        previous = pairs_by_key.get(key)
        if previous is None or _pair_sort_key(normalized) > _pair_sort_key(previous):
            pairs_by_key[key] = normalized
    return sorted(
        pairs_by_key.values(),
        key=lambda item: (
            prediction_ids.index(str(item["prediction_id"])),
            target_ids.index(str(item["target_id"])),
        ),
    )


def _raw_pair_list(raw: dict[str, object]) -> list[object]:
    for key in ("candidate_pairs", "pair_scores", "alignments", "matches"):
        value = raw.get(key)
        if isinstance(value, list):
            return value
    return []


def _one_to_one_matches(candidate_pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    eligible_pairs = [
        pair
        for pair in candidate_pairs
        if not _bool(pair.get("violation"))
        and (_number(pair.get("alignment_score")) or 0.0) >= PAIR_MATCH_THRESHOLD
    ]
    if not eligible_pairs:
        return []

    prediction_ids = list(dict.fromkeys(str(pair["prediction_id"]) for pair in eligible_pairs))
    target_ids = list(dict.fromkeys(str(pair["target_id"]) for pair in eligible_pairs))
    if min(len(prediction_ids), len(target_ids)) <= 22:
        return _one_to_one_matches_dp(
            eligible_pairs,
            prediction_ids=prediction_ids,
            target_ids=target_ids,
        )
    source = 0
    first_prediction = 1
    first_target = first_prediction + len(prediction_ids)
    sink = first_target + len(target_ids)
    graph: list[list[dict[str, object]]] = [[] for _ in range(sink + 1)]
    pair_edges: list[dict[str, object]] = []

    def add_edge(
        source_node: int,
        target_node: int,
        *,
        capacity: int,
        cost: int,
        pair: dict[str, object] | None = None,
    ) -> dict[str, object]:
        edge = {
            "to": target_node,
            "rev": len(graph[target_node]),
            "cap": capacity,
            "cost": cost,
            "pair": pair,
        }
        reverse = {
            "to": source_node,
            "rev": len(graph[source_node]),
            "cap": 0,
            "cost": -cost,
            "pair": None,
        }
        graph[source_node].append(edge)
        graph[target_node].append(reverse)
        return edge

    for index in range(len(prediction_ids)):
        add_edge(source, first_prediction + index, capacity=1, cost=0)
    for index in range(len(target_ids)):
        add_edge(first_target + index, sink, capacity=1, cost=0)
    for pair in eligible_pairs:
        score = _number(pair.get("alignment_score")) or 0.0
        cost = -(MATCH_COUNT_PRIORITY + int(round(_clamp01(score) * 1000)))
        edge = add_edge(
            first_prediction + prediction_ids.index(str(pair["prediction_id"])),
            first_target + target_ids.index(str(pair["target_id"])),
            capacity=1,
            cost=cost,
            pair=pair,
        )
        pair_edges.append(edge)

    _augment_negative_cost_paths(graph, source=source, sink=sink)
    matches = [edge["pair"] for edge in pair_edges if edge["cap"] == 0 and edge["pair"]]
    return sorted(
        [dict(pair) for pair in matches if isinstance(pair, dict)],
        key=lambda item: prediction_ids.index(str(item["prediction_id"])),
    )


def _direction_filtered_pairs(
    *,
    metric_name: Literal["EA", "DA"],
    candidate_pairs: list[dict[str, object]],
    prediction_payload: list[dict[str, object]],
    semantic_targets: list[dict[str, object]],
    prediction_ids: list[str],
    target_ids: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if metric_name != "DA":
        return candidate_pairs, []
    predictions_by_id = {
        prediction_id: item
        for prediction_id, item in zip(prediction_ids, prediction_payload, strict=True)
    }
    targets_by_id = {
        target_id: item for target_id, item in zip(target_ids, semantic_targets, strict=True)
    }
    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    for pair in candidate_pairs:
        prediction = predictions_by_id.get(str(pair["prediction_id"]), {})
        target = targets_by_id.get(str(pair["target_id"]), {})
        if _da_direction_compatible(prediction, target):
            kept.append(pair)
        else:
            removed.append(
                {
                    **pair,
                    "direction_filter_reason": (
                        "Diagnosis status is incompatible with the reference target status."
                    ),
                }
            )
    return kept, removed


def strict_semantic_filtered_pairs(
    *,
    candidate_pairs: list[dict[str, object]],
    prediction_payload: list[dict[str, object]],
    semantic_targets: list[dict[str, object]],
    prediction_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Remove broad matches that do not share the target's scientific anchors."""
    prediction_ids = prediction_ids or _prediction_ids(prediction_payload)
    target_ids = target_ids or _target_ids(semantic_targets)
    predictions_by_id = {
        prediction_id: item
        for prediction_id, item in zip(prediction_ids, prediction_payload, strict=True)
    }
    targets_by_id = {
        target_id: item for target_id, item in zip(target_ids, semantic_targets, strict=True)
    }
    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    for pair in candidate_pairs:
        prediction = predictions_by_id.get(str(pair.get("prediction_id")), {})
        target = targets_by_id.get(str(pair.get("target_id")), {})
        compatible, reasons = _strict_anchor_compatible(prediction, target)
        if compatible:
            kept.append(pair)
        else:
            removed.append({**pair, "strict_filter_reason": "; ".join(reasons)})
    return kept, removed


_STRICT_ANCHOR_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "lysosome",
        ("lysosome",),
        ("lysosome",),
    ),
    (
        "reaction_based_imaging",
        ("chemical reaction", "reaction based", "reaction-based"),
        ("chemical reaction", "reaction", "reactive"),
    ),
    (
        "tict_or_charge_transfer",
        (
            "tict",
            "charge transfer",
            "charge-transfer",
            "solvatochrom",
            "homo",
            "lumo",
            "donor acceptor",
            "donor-acceptor",
            "donor–acceptor",
        ),
        (
            "tict",
            "ict",
            "charge transfer",
            "charge",
            "ct",
            "donor",
            "acceptor",
            "solvatochrom",
            "homo",
            "lumo",
            "frontier",
        ),
    ),
    (
        "lipid_droplet_or_oil",
        ("lipid droplet", "ld", "oil mimicking", "oil-like", "cyan emission"),
        ("lipid droplet", "ld", "lipophil", "hydrophobic", "oil", "cyan", "droplet"),
    ),
    (
        "mitochondria",
        ("mitochond",),
        ("mitochond", "cationic"),
    ),
    (
        "cell_or_in_vivo_imaging",
        (
            "live cell",
            "live-cell",
            "cell staining",
            "colocalization",
            "in vivo",
            "c elegans",
            "long term cell",
            "long-term cell",
        ),
        ("cell", "in vivo", "imaging", "colocalization", "organelle"),
    ),
    (
        "dls_or_aggregation_environment",
        ("dynamic light scattering", "dls"),
        ("dynamic light scattering", "dls", "aggregate", "aggregation", "water fraction"),
    ),
    (
        "emission_environment",
        ("methanol", "water", "solvent", "emission maxima", "emission peak"),
        (
            "methanol",
            "water",
            "solvent",
            "polarity",
            "emission",
            "spectra",
            "spectrum",
            "photoluminescence",
            "pl",
        ),
    ),
    (
        "viscosity_or_motion_restriction",
        ("viscosity",),
        ("viscosity", "rim", "rir", "riv", "restriction", "motion", "rotation"),
    ),
    (
        "sulfur_cluster_or_thiophene",
        (
            "sulfur",
            "thiophene",
            "clusteroluminescence",
            "s s interaction",
            "sulfur sulfur",
        ),
        ("sulfur", "thiophene", "cluster", "s s", "heavy sulfur"),
    ),
    (
        "mof_or_coordination_framework",
        ("mof", "framework", "coordination"),
        ("mof", "framework", "coordination", "ligand"),
    ),
    (
        "gas_pressure_response",
        ("gas pressure", "pressure", "vacuum", "nitrogen", "argon", "co2", "gas specific"),
        ("gas", "pressure", "vacuum", "deformation", "framework", "adsorption"),
    ),
    (
        "lanthanide_center",
        ("lanthanide", "eu", "gd"),
        ("lanthanide", "eu", "gd", "antenna", "metal"),
    ),
    (
        "esipt_or_tautomer",
        ("esipt", "keto tautomer", "tautomer"),
        (
            "esipt",
            "keto",
            "tautomer",
            "proton transfer",
            "phenolic",
            "phenol",
            "carbonyl",
            "imine",
            "hydrogen bond",
        ),
    ),
    (
        "conical_intersection_or_nonadiabatic",
        ("conical", "nonadiabatic", "surface hopping", "ct-keto"),
        ("conical", "ci", "nonadiabatic", "tict", "ct", "charge transfer", "keto"),
    ),
    (
        "dpe_stilbene_or_photoisomerization",
        ("dpe", "stilbene", "photocyclization", "e z", "isomerization"),
        ("dpe", "stilbene", "photocyclization", "isomerization", "alkene", "torsion"),
    ),
    (
        "dimer_or_multiple_species",
        ("dimer", "h dimer", "monomer", "multiple emissive"),
        ("dimer", "monomer", "multiple", "species", "h dimer"),
    ),
    (
        "pi_stacking_or_acq",
        ("pi pi", "stacking", "acq", "aggregation caused quenching"),
        ("pi", "stacking", "acq", "quenching"),
    ),
)


def _strict_anchor_compatible(
    prediction: dict[str, object],
    target: dict[str, object],
) -> tuple[bool, list[str]]:
    prediction_text = _strict_prediction_text_blob(prediction)
    target_text = _strict_target_text_blob(target)
    reasons: list[str] = []

    for rule_name, target_terms, prediction_terms in _STRICT_ANCHOR_RULES:
        if _contains_any_anchor(target_text, target_terms) and not _contains_any_anchor(
            prediction_text,
            prediction_terms,
        ):
            reasons.append(f"missing target anchor: {rule_name}")

    if _is_ld_mito_contact_target(target_text) and not (
        _contains_any_anchor(prediction_text, ("mitochond",))
        and _contains_any_anchor(prediction_text, ("lipid droplet", "ld", "lipid", "droplet"))
        and _contains_any_anchor(
            prediction_text,
            ("contact", "interaction", "fission", "fusion", "ferroptosis"),
        )
    ):
        reasons.append("missing coupled LD/mitochondria/contact anchor")

    return not reasons, reasons


def _is_ld_mito_contact_target(text: str) -> bool:
    return _contains_any_anchor(
        text,
        (
            "ld mito",
            "ld mitochond",
            "lipid droplet mitochond",
            "mitochondrial fission",
            "ferroptosis",
            "organelle contact",
        ),
    )


def _strict_prediction_text_blob(item: dict[str, object]) -> str:
    fields = [
        item.get("claim"),
        item.get("basis"),
        item.get("support"),
        item.get("status"),
        item.get("context"),
        item.get("limits"),
        item.get("mechanism"),
        item.get("reasoning_summary"),
        item.get("missing_or_unresolved"),
        item.get("scope_limits"),
        item.get("alignment_guidance"),
        item.get("claim_text"),
        item.get("diagnosis_direction"),
        item.get("source_mechanism"),
        item.get("smiles_only_alignment_guidance"),
    ]
    return _normalize_anchor_text(" ".join(str(field) for field in fields if field))


def _strict_target_text_blob(item: dict[str, object]) -> str:
    fields = [
        item.get("claim_text"),
        item.get("alignment_guidance"),
        item.get("diagnosis_direction"),
        item.get("source_mechanism"),
        item.get("smiles_only_alignment_guidance"),
        item.get("target_kind"),
        item.get("access_class"),
        item.get("required_capabilities"),
        item.get("acceptable_response_level"),
    ]
    return _normalize_anchor_text(" ".join(str(field) for field in fields if field))


def _normalize_anchor_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("π", "pi")
    normalized = normalized.replace("–", "-").replace("—", "-").replace("_", " ")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9+\-.]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f" {normalized} "


def _contains_any_anchor(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_anchor(text, term) for term in terms)


def _contains_anchor(text: str, term: str) -> bool:
    normalized_term = _normalize_anchor_text(term).strip()
    if not normalized_term:
        return False
    if " " in normalized_term:
        return f" {normalized_term} " in text
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", text) is not None


def _da_direction_compatible(
    prediction: dict[str, object],
    target: dict[str, object],
) -> bool:
    target_class = _target_diagnosis_class(target)
    prediction_status = _string(prediction.get("status"))
    if target_class is None or prediction_status is None:
        return True
    supported_statuses = {"computed_supported", "proxy_supported", "plausible"}
    weakened_statuses = {"weakened", "rejected", "underdetermined"}
    underdetermined_statuses = {"underdetermined"}
    if target_class == "supported":
        return prediction_status in supported_statuses
    if target_class == "weakened_or_rejected":
        return prediction_status in weakened_statuses
    if target_class == "underdetermined_or_followup":
        return prediction_status in underdetermined_statuses
    return True


def _target_diagnosis_class(target: dict[str, object]) -> str | None:
    raw_values = [
        _string(target.get("diagnosis_direction")),
    ]
    for raw in raw_values:
        if raw is None:
            continue
        normalized = raw.strip().lower().replace("-", "_")
        if normalized in {"supported", "weakened_or_rejected"}:
            return normalized
        if normalized in {"underdetermined", "underdetermined_or_followup"}:
            return "underdetermined_or_followup"
    return None


def _one_to_one_matches_dp(
    eligible_pairs: list[dict[str, object]],
    *,
    prediction_ids: list[str],
    target_ids: list[str],
) -> list[dict[str, object]]:
    if len(target_ids) <= len(prediction_ids):
        pairs_by_prediction: list[list[tuple[int, dict[str, object]]]] = [
            [] for _ in prediction_ids
        ]
        for pair in eligible_pairs:
            pairs_by_prediction[prediction_ids.index(str(pair["prediction_id"]))].append(
                (target_ids.index(str(pair["target_id"])), pair)
            )
        states: dict[int, tuple[int, float, list[dict[str, object]]]] = {0: (0, 0.0, [])}
        for pairs in pairs_by_prediction:
            next_states = dict(states)
            for mask, (count, score, selected_pairs) in states.items():
                for target_index, pair in pairs:
                    target_bit = 1 << target_index
                    if mask & target_bit:
                        continue
                    candidate = (
                        count + 1,
                        score + (_number(pair.get("alignment_score")) or 0.0),
                        [*selected_pairs, pair],
                    )
                    next_mask = mask | target_bit
                    if _matching_state_key(candidate) > _matching_state_key(
                        next_states.get(next_mask)
                    ):
                        next_states[next_mask] = candidate
            states = next_states
        best = max(states.values(), key=_matching_state_key)
        return [dict(pair) for pair in best[2]]

    pairs_by_target: list[list[tuple[int, dict[str, object]]]] = [[] for _ in target_ids]
    for pair in eligible_pairs:
        pairs_by_target[target_ids.index(str(pair["target_id"]))].append(
            (prediction_ids.index(str(pair["prediction_id"])), pair)
        )
    states = {0: (0, 0.0, [])}
    for pairs in pairs_by_target:
        next_states = dict(states)
        for mask, (count, score, selected_pairs) in states.items():
            for prediction_index, pair in pairs:
                prediction_bit = 1 << prediction_index
                if mask & prediction_bit:
                    continue
                candidate = (
                    count + 1,
                    score + (_number(pair.get("alignment_score")) or 0.0),
                    [*selected_pairs, pair],
                )
                next_mask = mask | prediction_bit
                if _matching_state_key(candidate) > _matching_state_key(
                    next_states.get(next_mask)
                ):
                    next_states[next_mask] = candidate
        states = next_states
    best = max(states.values(), key=_matching_state_key)
    return sorted(
        [dict(pair) for pair in best[2]],
        key=lambda item: prediction_ids.index(str(item["prediction_id"])),
    )


def _matching_state_key(
    state: tuple[int, float, list[dict[str, object]]] | None,
) -> tuple[int, float]:
    if state is None:
        return (-1, -1.0)
    return (state[0], state[1])


def _augment_negative_cost_paths(
    graph: list[list[dict[str, object]]],
    *,
    source: int,
    sink: int,
) -> None:
    node_count = len(graph)
    while True:
        distances = [float("inf")] * node_count
        parents: list[tuple[int, int] | None] = [None] * node_count
        distances[source] = 0.0
        for _ in range(node_count - 1):
            updated = False
            for node, edges in enumerate(graph):
                if distances[node] == float("inf"):
                    continue
                for edge_index, edge in enumerate(edges):
                    if int(edge["cap"]) <= 0:
                        continue
                    next_node = int(edge["to"])
                    next_distance = distances[node] + int(edge["cost"])
                    if next_distance < distances[next_node]:
                        distances[next_node] = next_distance
                        parents[next_node] = (node, edge_index)
                        updated = True
            if not updated:
                break
        if distances[sink] >= 0 or parents[sink] is None:
            break
        node = sink
        while node != source:
            parent = parents[node]
            if parent is None:
                break
            previous_node, edge_index = parent
            edge = graph[previous_node][edge_index]
            reverse = graph[node][int(edge["rev"])]
            edge["cap"] = int(edge["cap"]) - 1
            reverse["cap"] = int(reverse["cap"]) + 1
            node = previous_node


def _missed_targets(
    semantic_targets: list[dict[str, object]],
    matched_pairs: list[dict[str, object]],
) -> list[str]:
    matched_target_ids = {str(pair["target_id"]) for pair in matched_pairs}
    return [
        _target_label(item, index)
        for index, item in enumerate(semantic_targets, start=1)
        if (_string(item.get("target_id")) or f"T{index:03d}") not in matched_target_ids
    ]


def _unmatched_predictions(
    prediction_payload: list[dict[str, object]],
    matched_pairs: list[dict[str, object]],
) -> list[str]:
    matched_prediction_ids = {str(pair["prediction_id"]) for pair in matched_pairs}
    prediction_ids = _prediction_ids(prediction_payload)
    return [
        _prediction_label(item, index)
        for index, item in enumerate(prediction_payload, start=1)
        if prediction_ids[index - 1] not in matched_prediction_ids
    ]


def _target_label(item: dict[str, object], index: int) -> str:
    target_id = _string(item.get("target_id")) or f"T{index:03d}"
    target = str(item.get("claim_text") or item.get("alignment_guidance") or "")
    return f"{target_id}: {target}" if target else target_id


def _prediction_label(item: dict[str, object], index: int) -> str:
    prediction_id = _prediction_ids([item])[0] if item else f"P{index:03d}"
    summary = str(item.get("claim") or item.get("mechanism") or "")
    return f"{prediction_id}: {summary}" if summary else prediction_id


def _pair_sort_key(pair: dict[str, object]) -> tuple[bool, float]:
    return (
        not _bool(pair.get("violation")),
        _number(pair.get("alignment_score")) or 0.0,
    )


def _judge_details(raw: dict[str, object]) -> dict[str, object]:
    excluded = {
        "candidate_pairs",
        "pair_scores",
        "alignments",
        "matches",
        "precision",
        "recall",
        "f1",
        "predicted_positive_count",
        "reference_target_count",
        "matched_prediction_count",
        "matched_target_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "rationale",
    }
    return {key: value for key, value in raw.items() if key not in excluded}


def _number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _string(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _integer(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _quantize_pair_score(value: float) -> float:
    clamped = _clamp01(value)
    return min(PAIR_SCORE_LEVELS, key=lambda level: (abs(level - clamped), level))


def _alignment_counts(
    raw: dict[str, object],
    *,
    prediction_count: int,
    target_count: int,
) -> dict[str, int]:
    predicted_positive_count = (
        _integer(raw.get("predicted_positive_count"))
        or _integer(raw.get("prediction_count"))
        or prediction_count
    )
    reference_target_count = (
        _integer(raw.get("reference_target_count"))
        or _integer(raw.get("target_count"))
        or target_count
    )
    matched_prediction_count = (
        _integer(raw.get("matched_prediction_count"))
        or _integer(raw.get("true_positives"))
        or 0
    )
    matched_target_count = (
        _integer(raw.get("matched_target_count"))
        or _integer(raw.get("covered_target_count"))
        or _integer(raw.get("true_positives"))
        or 0
    )
    predicted_positive_count = max(0, predicted_positive_count)
    reference_target_count = max(0, reference_target_count)
    return {
        "predicted_positive_count": predicted_positive_count,
        "reference_target_count": reference_target_count,
        "matched_prediction_count": min(max(0, matched_prediction_count), predicted_positive_count),
        "matched_target_count": min(max(0, matched_target_count), reference_target_count),
    }


def _precision_recall_f1(
    *,
    matched_prediction_count: int,
    predicted_positive_count: int,
    matched_target_count: int,
    reference_target_count: int,
) -> tuple[float, float, float]:
    precision = (
        matched_prediction_count / predicted_positive_count
        if predicted_positive_count
        else 0.0
    )
    recall = matched_target_count / reference_target_count if reference_target_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1
