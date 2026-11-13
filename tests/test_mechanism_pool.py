"""The 7-family pool used by the 0.3.0 agent, its baselines and the evaluation (author decision 2026-11-13)."""

import json

from mechcal.eval import evidence_judge as ej
from mechcal.eval.common_prior import COMMON_PRIOR_AT3_LABELS
from mechcal.eval.family_reduction import ENVIRONMENT, MERGED_INTO, OUT_OF_POOL
from mechcal.mechanism_pool import (ELEVEN_FAMILY_DEFINITIONS, ELEVEN_FAMILY_POOL, MECHANISM_DEFINITIONS,
                                    MECHANISM_POOL, eleven_family_definitions, mechanism_definitions)

REMOVED = {"HOST_GUEST_INTERACTION", "SOKR_ANTI_KASHA", "PACKING_HOST_MATRIX_CONFINEMENT", "RACI_CI_ACCESS",
           "TRIPLET_METAL_ENERGY_TRANSFER"}


def test_seven_family_pool_and_public_meanings():
    assert len(MECHANISM_POOL) == 7 and tuple(MECHANISM_DEFINITIONS) == MECHANISM_POOL
    assert not REMOVED & set(MECHANISM_POOL)
    data = mechanism_definitions()
    assert "packing, host or matrix" in data["definitions"]["RIM_RIR_RIV"]
    assert "conical intersection" in data["definitions"]["RIM_RIR_RIV"]
    assert "packing-dependent" in data["definitions"]["AGGREGATE_EXCITON_EXCIMER"]
    assert "metal" not in data["definitions"]["TRIPLET_ISC_RTP_TADF"].lower()
    assert not {"hidden_reference", "criteria", "score", "case_id"} & data.keys()


def test_eleven_family_pool_is_kept_for_the_frozen_records():
    assert len(ELEVEN_FAMILY_POOL) == 11 and tuple(ELEVEN_FAMILY_DEFINITIONS) == ELEVEN_FAMILY_POOL
    assert len(eleven_family_definitions()["definitions"]) == 11
    mapped = {MERGED_INTO.get(f, f) for f in ELEVEN_FAMILY_POOL} - ENVIRONMENT - OUT_OF_POOL
    assert mapped == set(MECHANISM_POOL)


def test_evidence_rubric_covers_the_seven_families_and_awaits_the_author():
    rubric = ej.load_rubric()
    assert tuple(rubric["mechanism_pool"]) == MECHANISM_POOL and tuple(rubric["labels"]) == MECHANISM_POOL
    assert rubric["status"] == "draft_pending_author"
    rim = {r["id"] for level in ("strong_support", "partial_support") for r in rubric["labels"]["RIM_RIR_RIV"][level]["any_of"]}
    assert {"crossing_access_and_restriction", "coordinate_linked_crossing_proxy", "environment_constraint_proxy"} <= rim
    triplet = [r["id"] for r in rubric["labels"]["TRIPLET_ISC_RTP_TADF"]["strong_support"]["any_of"]]
    assert "metal_sensitization" not in triplet
    assert json.loads(ej.FREEZE_PATH.read_text())["status"] == "draft_pending_author"


def test_common_prior_uses_pool_families():
    assert set(COMMON_PRIOR_AT3_LABELS) <= set(MECHANISM_POOL) and len(COMMON_PRIOR_AT3_LABELS) == 3
