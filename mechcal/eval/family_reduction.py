"""Map reference labels of the original 11-family records onto the 7-family pool (2026-11-13, author decision).

Mechanical rules only; no case is judged individually:

- RACI_CI_ACCESS is merged into RIM_RIR_RIV (role: the stronger of the two; gain: the larger;
  evidence ids: union).
- TRIPLET_METAL_ENERGY_TRANSFER is renamed TRIPLET_ISC_RTP_TADF.
- PACKING_HOST_MATRIX_CONFINEMENT is no longer a family. Its label is removed and kept as
  ``environment_context`` (the packing, host or matrix the source paper describes). Whether a given
  packing effect is a motion restriction (RIM) or a change of intermolecular coupling (AGGREGATE) is
  an annotation decision, so it is not guessed here; the rebuilt dataset annotates it directly.
- HOST_GUEST_INTERACTION and SOKR_ANTI_KASHA are outside the pool; their labels are removed and
  kept as ``out_of_pool_labels``.

The original labels and decisions are kept in ``hidden_reference.family_reduction`` so that every
mapping can be audited.
"""

from __future__ import annotations

import copy

from mechcal.mechanism_pool import ELEVEN_FAMILY_POOL, MECHANISM_POOL

MERGED_INTO = {"RACI_CI_ACCESS": "RIM_RIR_RIV", "TRIPLET_METAL_ENERGY_TRANSFER": "TRIPLET_ISC_RTP_TADF"}
ENVIRONMENT = {"PACKING_HOST_MATRIX_CONFINEMENT"}
OUT_OF_POOL = {"HOST_GUEST_INTERACTION", "SOKR_ANTI_KASHA"}
ROLE_RANK = {"primary": 3, "co_primary": 2, "secondary": 1, None: 0}
STATUS_RANK = {"supported": 3, "contradicted": 2, "not_established": 1}
PRIMARY_ROLES = {"primary", "co_primary"}

assert set(ELEVEN_FAMILY_POOL) - ENVIRONMENT - OUT_OF_POOL - set(MERGED_INTO) <= set(MECHANISM_POOL)


def _target(label: str) -> str | None:
    if label in ENVIRONMENT or label in OUT_OF_POOL:
        return None
    return MERGED_INTO.get(label, label)


def _merge_mechanism(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    out["role"] = max((a.get("role"), b.get("role")), key=lambda r: ROLE_RANK.get(r, 0))
    out["gain"] = max(a.get("gain", 0), b.get("gain", 0))
    out["evidence_ids"] = sorted(set(a.get("evidence_ids", [])) | set(b.get("evidence_ids", [])))
    out["rationale"] = " ".join(x for x in (a.get("rationale"), b.get("rationale")) if x)
    return out


def _merge_decision(a: dict, b: dict) -> dict:
    out = copy.deepcopy(max((a, b), key=lambda d: STATUS_RANK.get(d.get("status"), 0)))
    out["role"] = max((a.get("role"), b.get("role")), key=lambda r: ROLE_RANK.get(r, 0))
    if out["status"] != "supported":
        out["role"] = None
    out["evidence_ids"] = sorted(set(a.get("evidence_ids", [])) | set(b.get("evidence_ids", [])))
    out["rationale"] = " ".join(x for x in (a.get("rationale"), b.get("rationale")) if x)
    return out


def reduce_hidden_reference(hidden: dict) -> dict:
    """A copy of ``hidden`` whose labels use the 7-family pool, with the mapping recorded."""
    out = copy.deepcopy(hidden)
    mechanisms: dict[str, dict] = {}
    environment, outside = [], []
    for m in hidden["reference_mechanisms"]:
        target = _target(m["label"])
        if target is None:
            (environment if m["label"] in ENVIRONMENT else outside).append(copy.deepcopy(m))
            continue
        item = dict(copy.deepcopy(m), label=target)
        mechanisms[target] = _merge_mechanism(mechanisms[target], item) if target in mechanisms else item
    decisions: dict[str, dict] = {}
    for d in hidden["mechanism_decisions"]:
        target = _target(d["label"])
        if target is None:
            continue
        item = dict(copy.deepcopy(d), label=target)
        decisions[target] = _merge_decision(decisions[target], item) if target in decisions else item
    out["reference_mechanisms"] = [mechanisms[f] for f in MECHANISM_POOL if f in mechanisms]
    out["mechanism_decisions"] = [decisions[f] for f in MECHANISM_POOL if f in decisions]
    has_primary_before = any(m.get("role") in PRIMARY_ROLES for m in hidden["reference_mechanisms"])
    has_primary_after = any(m.get("role") in PRIMARY_ROLES for m in out["reference_mechanisms"])
    out["family_reduction"] = {
        "pool": list(MECHANISM_POOL),
        "rules": "mechcal/eval/family_reduction.py",
        "environment_context": environment,
        "out_of_pool_labels": outside,
        "primary_lost": has_primary_before and not has_primary_after,
        "no_labels_left": not out["reference_mechanisms"],
        "original_reference_mechanisms": copy.deepcopy(hidden["reference_mechanisms"]),
        "original_mechanism_decisions": copy.deepcopy(hidden["mechanism_decisions"]),
    }
    return out
