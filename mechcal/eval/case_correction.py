from __future__ import annotations

import json
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

JsonDict = dict[str, Any]

CASE_CORRECTION_POLICY_VERSION = "smiles-only-scoreability-v3"
CASE_CORRECTION_GENERATOR = "mechcal.eval.case_correction"
DEFAULT_MAX_EVIDENCE_TARGETS = 12
DEFAULT_MAX_DIAGNOSIS_TARGETS = 8

AccessClass = Literal[
    "structure_proxy",
    "low_cost_compute_proxy",
    "generic_literature",
    "wet_lab_required",
    "high_level_compute_required",
    "application_context_required",
    "source_paper_only",
]
DiagnosisAccess = Literal[
    "supported",
    "weakened_or_rejected",
    "underdetermined_or_followup",
]
ScoringRole = Literal[
    "primary_score",
    "boundary_score",
    "must_not_claim_only",
    "oracle_only",
]
TargetKind = Literal["evidence", "diagnosis", "boundary", "must_not_claim"]
AcceptableResponseLevel = Literal[
    "mechanism_family",
    "proxy_observable",
    "exact_measurement",
    "protocol_followup",
]

_CAPABILITY_CATEGORIES = {
    "smiles_structure": "structure",
    "donor_acceptor_proxy": "structure",
    "lipophilicity_proxy": "structure",
    "rotor_or_torsion_proxy": "structure",
    "frontier_or_charge_proxy": "structure",
    "solvent_pl_spectroscopy": "experiment",
    "qy_or_lifetime_measurement": "experiment",
    "dls_aggregation": "experiment",
    "viscosity_pl": "experiment",
    "bioimaging_colocalization": "experiment",
    "bioimaging_perturbation": "experiment",
    "pxrd_or_crystal_characterization": "experiment",
    "gas_pressure_photophysics": "experiment",
    "photochemistry_product_analysis": "experiment",
    "ultrafast_spectroscopy": "experiment",
    "td_dft_or_orbital_analysis": "compute",
    "excited_state_scan": "compute",
    "conical_intersection_search": "compute",
    "nonadiabatic_dynamics": "compute",
    "crystal_qmmm_or_packing_compute": "compute",
    "source_identity": "source",
    "literature_context": "source",
}

_NEGATIVE_ACCESS_TERMS = (
    "not accessible from smiles",
    "not inferable from smiles",
    "not available from public input",
    "not available from public_input",
    "requires literature",
    "requires article",
    "requires source",
    "hidden reference only",
    "hidden_reference_only",
    "hidden_reference_source_only",
    "hidden_reference_source_quote",
)

_CAPABILITY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "dls_aggregation",
        ("dynamic light scattering", "dls", "aggregate in water", "aggregates in water"),
    ),
    (
        "viscosity_pl",
        ("viscosity", "viscous", "viscosity-dependent", "restriction of motion"),
    ),
    (
        "bioimaging_perturbation",
        (
            "ferroptosis",
            "erastin",
            "ferrostatin",
            "cell stimulation",
            "mitochondrial damage",
            "fission",
            "fusion",
            "organelle dynamics",
            "mitochondrial dynamics",
            "ld-mito",
            "ld mitochond",
        ),
    ),
    (
        "bioimaging_colocalization",
        (
            "cell",
            "live-cell",
            "live cell",
            "colocalization",
            "microscopy",
            "lysosome",
            "mitochondrial imaging",
            "organelle imaging",
        ),
    ),
    (
        "pxrd_or_crystal_characterization",
        (
            "pxrd",
            "xrd",
            "diffraction",
            "crystal structure",
            "crystal packing",
            "single-crystal",
            "powder",
            "polymorph",
            "mof",
            "framework",
        ),
    ),
    (
        "gas_pressure_photophysics",
        ("gas pressure", "vacuum", "nitrogen", "co2", "argon", "air", "pressure response"),
    ),
    (
        "photochemistry_product_analysis",
        ("irradiation", "photocyclization", "photoisomerization", "gc-ms", "1h nmr"),
    ),
    (
        "qy_or_lifetime_measurement",
        ("quantum yield", "qy", "lifetime", "decay measurement", "fluorescence yield"),
    ),
    (
        "ultrafast_spectroscopy",
        ("ultrafast", "transient ir", "uv/ir", "time-resolved", "picosecond"),
    ),
    (
        "solvent_pl_spectroscopy",
        (
            "emission",
            "fluorescence",
            "photoluminescence",
            "spectra",
            "spectrum",
            "absorption",
            "emission peak",
            "emission maxima",
            "fluorescence peak",
            "absorption peak",
            "water fraction",
            "methanol",
            "thf",
        ),
    ),
    (
        "nonadiabatic_dynamics",
        (
            "nonadiabatic dynamics",
            "surface hopping",
            "surface-hopping",
            "trajectory",
            "tsh",
            "quantum dynamics",
        ),
    ),
    (
        "conical_intersection_search",
        ("conical intersection", " ci ", "ci region", "s1/s0", "surface crossing"),
    ),
    (
        "crystal_qmmm_or_packing_compute",
        ("qm/mm", "qmmm", "crystal calculation", "crystal model", "packing compute"),
    ),
    (
        "excited_state_scan",
        (
            "excited-state scan",
            "potential energy",
            "potential-energy",
            "barrierless",
            "s1 minimum",
            "ct intermediate",
        ),
    ),
    (
        "td_dft_or_orbital_analysis",
        ("dft", "td-dft", "homo", "lumo", "orbital", "vertical emission"),
    ),
    (
        "source_identity",
        (
            "source explicitly includes",
            "smiles-matched",
            "studied molecule",
            "compound 1",
            "cis-stilbene",
            "trans-stilbene",
            "requires literature",
            "requires article",
            "literature identification",
            "paper_statement",
        ),
    ),
    (
        "literature_context",
        (
            "source article",
            "source conclusion",
            "source limitation",
            "source review",
            "paper argues",
            "article frames",
            "paper lists",
            "authors explicitly",
            "hidden_reference",
        ),
    ),
    (
        "lipophilicity_proxy",
        ("lipophil", "hydrophobic", "logp", "c log p", "oil", "lipid droplet"),
    ),
    (
        "donor_acceptor_proxy",
        ("donor-acceptor", "donor acceptor", "d-a", "acceptor", "donor", "charge transfer"),
    ),
    (
        "rotor_or_torsion_proxy",
        ("rotor", "rotation", "torsion", "rim", "rir", "riv", "flapping", "motion"),
    ),
    (
        "frontier_or_charge_proxy",
        ("cationic", "frontier", "charge", "ct", "tict"),
    ),
    (
        "smiles_structure",
        (
            "smiles",
            "scaffold",
            "functional group",
            "substituent",
            "phenyl",
            "thiophene",
            "stilbene",
            "dpe",
            "ligand",
            "carboxylated",
            "molecular structure",
        ),
    ),
)

_SUPPORTED_ROLE_TERMS = ("supported", "primary", "final", "core")
_WEAKENED_ROLE_TERMS = ("weakened", "rejected", "deprioritized", "insufficient")
_UNDERDETERMINED_ROLE_TERMS = ("underdetermined", "follow", "necessary", "caveat")

_GLOBAL_MUST_NOT_CLAIM = (
    "Do not claim paper-derived experiments, spectra, quantum yields, microscopy, DLS, "
    "crystal structures, CI/TSH/nonadiabatic calculations, QM/MM, or exact source "
    "numeric results as produced by MAS unless they are explicitly present in the "
    "public input or generated by the run."
)


@dataclass(frozen=True)
class CorrectedCaseBatchResult:
    output_dir: Path
    case_count: int
    summary_path: Path
    audit_path: Path


def correct_case_document(
    document: JsonDict,
    *,
    max_evidence_targets: int = DEFAULT_MAX_EVIDENCE_TARGETS,
    max_diagnosis_targets: int = DEFAULT_MAX_DIAGNOSIS_TARGETS,
) -> JsonDict:
    """Add compact SMILES-only semantic targets without mutating reference facts."""

    corrected, _audit = _correct_case_document_with_audit(
        document,
        max_evidence_targets=max_evidence_targets,
        max_diagnosis_targets=max_diagnosis_targets,
    )
    return corrected


def build_semantic_evidence_targets(
    reference_units: list[JsonDict],
    *,
    max_targets: int = DEFAULT_MAX_EVIDENCE_TARGETS,
) -> list[JsonDict]:
    targets, _audit = _build_semantic_evidence_targets_with_audit(
        reference_units,
        max_targets=max_targets,
    )
    return targets


def build_semantic_diagnosis_targets(
    reference_units: list[JsonDict],
    *,
    max_targets: int = DEFAULT_MAX_DIAGNOSIS_TARGETS,
) -> list[JsonDict]:
    targets, _audit = _build_semantic_diagnosis_targets_with_audit(
        reference_units,
        source_evidence_units=[],
        max_targets=max_targets,
    )
    return targets


def correct_case_path(
    input_path: Path,
    output_dir: Path,
    *,
    max_evidence_targets: int = DEFAULT_MAX_EVIDENCE_TARGETS,
    max_diagnosis_targets: int = DEFAULT_MAX_DIAGNOSIS_TARGETS,
) -> CorrectedCaseBatchResult:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_documents = list(_iter_case_documents(input_path))
    summaries: list[JsonDict] = []
    audit_cases: list[JsonDict] = []
    for name, document in case_documents:
        corrected, audit = _correct_case_document_with_audit(
            document,
            max_evidence_targets=max_evidence_targets,
            max_diagnosis_targets=max_diagnosis_targets,
        )
        output_path = output_dir / name
        output_path.write_text(
            json.dumps(corrected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        hidden = _dict(corrected.get("hidden_reference"))
        summaries.append(
            {
                "file": name,
                "case_id": corrected.get("case_id"),
                "smiles": extract_public_smiles(corrected),
                "semantic_evidence_target_count": len(
                    _dict_list(hidden.get("semantic_evidence_targets"))
                ),
                "semantic_diagnosis_target_count": len(
                    _dict_list(hidden.get("semantic_diagnosis_targets"))
                ),
                "validation_status": audit["validation"]["status"],
                "validation_issue_count": audit["validation"]["issue_count"],
            }
        )
        audit_cases.append(audit)
    summary = {
        "policy_version": CASE_CORRECTION_POLICY_VERSION,
        "input_path": str(input_path.expanduser().resolve()),
        "output_dir": str(output_dir),
        "case_count": len(summaries),
        "cases": summaries,
    }
    summary_path = output_dir / "correction_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_path = output_dir / "correction_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "policy_version": CASE_CORRECTION_POLICY_VERSION,
                "generator": CASE_CORRECTION_GENERATOR,
                "case_count": len(audit_cases),
                "cases": audit_cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return CorrectedCaseBatchResult(
        output_dir=output_dir,
        case_count=len(summaries),
        summary_path=summary_path,
        audit_path=audit_path,
    )


def extract_public_smiles(document: JsonDict) -> str | None:
    public_input = _dict(document.get("public_input"))
    molecule = _dict(public_input.get("molecule"))
    structure = _dict(molecule.get("structure"))
    if _clean_string(structure.get("format")).lower() == "smiles":
        value = _clean_string(structure.get("value") or structure.get("smiles"))
        return value or None
    normalized = _dict(public_input.get("normalized_structure"))
    value = _clean_string(normalized.get("smiles"))
    return value or None


def _correct_case_document_with_audit(
    document: JsonDict,
    *,
    max_evidence_targets: int,
    max_diagnosis_targets: int,
) -> tuple[JsonDict, JsonDict]:
    corrected = deepcopy(document)
    public_input = _dict(corrected.get("public_input"))
    smiles = extract_public_smiles(corrected)
    if smiles:
        corrected["public_input"] = _public_input_with_normalized_smiles(public_input, smiles)

    hidden = _dict(corrected.get("hidden_reference"))
    source_evidence = _dict_list(hidden.get("reference_evidence_units"))
    source_diagnoses = _dict_list(hidden.get("reference_diagnosis_units"))
    evidence_targets, evidence_audit = _build_semantic_evidence_targets_with_audit(
        source_evidence,
        max_targets=max_evidence_targets,
    )
    diagnosis_targets, diagnosis_audit = _build_semantic_diagnosis_targets_with_audit(
        source_diagnoses,
        source_evidence_units=source_evidence,
        max_targets=max_diagnosis_targets,
    )
    validation = validate_semantic_targets(
        evidence_targets=evidence_targets,
        diagnosis_targets=diagnosis_targets,
        audit_items=[*evidence_audit, *diagnosis_audit],
    )
    hidden["semantic_evidence_targets"] = evidence_targets
    hidden["semantic_diagnosis_targets"] = diagnosis_targets
    hidden["semantic_target_policy"] = {
        "policy_version": CASE_CORRECTION_POLICY_VERSION,
        "generator": CASE_CORRECTION_GENERATOR,
        "public_input_scope": "smiles_only",
        "raw_reference_preserved": True,
        "target_schema": [
            "target_id",
            "source_ids",
            "target_kind",
            "access_class",
            "required_capabilities",
            "scoring_role",
            "acceptable_response_level",
            "claim_text",
            "alignment_guidance",
            "not_acceptable",
            "rubric_0_0",
            "rubric_0_5",
            "rubric_1_0",
            "scoreability_rationale",
        ],
        "must_not_claim": _GLOBAL_MUST_NOT_CLAIM,
        "metric_contract": (
            "EA and DA score only semantic targets with scoring_role primary_score "
            "or boundary_score. oracle_only and must_not_claim_only targets are "
            "retained for audit/policy but excluded from SMILES-only recall."
        ),
        "agent_reproducibility_contract": (
            "Dataset-making agents copy reference_* units verbatim, derive compact "
            "targets with capability-first deterministic rules, and retain the "
            "separate correction audit for label validation."
        ),
    }
    corrected["hidden_reference"] = hidden
    corrected["benchmark_correction"] = {
        "policy_version": CASE_CORRECTION_POLICY_VERSION,
        "generator": CASE_CORRECTION_GENERATOR,
        "input_structure_format": "smiles" if smiles else "unknown",
        "semantic_info_changed": False,
        "added_fields": [
            "public_input.normalized_structure",
            "hidden_reference.semantic_evidence_targets",
            "hidden_reference.semantic_diagnosis_targets",
            "hidden_reference.semantic_target_policy",
        ],
        "validation": {
            "status": validation["status"],
            "issue_count": validation["issue_count"],
        },
    }
    audit = {
        "case_id": corrected.get("case_id"),
        "smiles": extract_public_smiles(corrected),
        "items": [*evidence_audit, *diagnosis_audit],
        "validation": validation,
    }
    return corrected, audit


def _build_semantic_evidence_targets_with_audit(
    reference_units: list[JsonDict],
    *,
    max_targets: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    candidates = [
        _semantic_evidence_target(unit, index=index)
        for index, unit in enumerate(reference_units, start=1)
    ]
    selected = _select_diverse_targets(candidates, max_targets=max_targets)
    numbered = [
        _renumber_target(item, prefix="SE", index=index)
        for index, item in enumerate(selected, 1)
    ]
    return [_strip_private(item) for item in numbered], [_audit_item(item) for item in numbered]


def _build_semantic_diagnosis_targets_with_audit(
    reference_units: list[JsonDict],
    *,
    source_evidence_units: list[JsonDict],
    max_targets: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    evidence_by_id = {
        _clean_string(item.get("evidence_id")): item for item in source_evidence_units
    }
    candidates = [
        _semantic_diagnosis_target(unit, index=index, evidence_by_id=evidence_by_id)
        for index, unit in enumerate(reference_units, start=1)
    ]
    selected = _select_diagnosis_targets(candidates, max_targets=max_targets)
    numbered = [
        _renumber_target(item, prefix="SD", index=index)
        for index, item in enumerate(selected, 1)
    ]
    return [_strip_private(item) for item in numbered], [_audit_item(item) for item in numbered]


def _semantic_evidence_target(unit: JsonDict, *, index: int) -> JsonDict:
    source_id = _clean_string(unit.get("evidence_id")) or f"E{index:03d}"
    claim = _clean_string(unit.get("claim"))
    interpretation = _clean_string(unit.get("mechanistic_interpretation"))
    access_text = _clean_string(unit.get("evidence_accessibility"))
    links = [str(item) for item in unit.get("mechanism_links") or []]
    analysis = _analyze_evidence_capabilities(
        access=access_text,
        claim=claim,
        interpretation=interpretation,
        links=links,
    )
    access_class = _access_class_from_capabilities(analysis["capabilities"])
    scoring_role = _evidence_scoring_role(access_class)
    response_level = _evidence_response_level(access_class)
    target_kind = _target_kind_for_scoring_role(scoring_role, default="evidence")
    return {
        "target_id": source_id,
        "source_ids": [source_id],
        "target_kind": target_kind,
        "access_class": access_class,
        "required_capabilities": analysis["capabilities"],
        "scoring_role": scoring_role,
        "acceptable_response_level": response_level,
        "claim_text": _evidence_claim_text(
            claim=claim,
            access_class=access_class,
            capabilities=analysis["capabilities"],
            scoring_role=scoring_role,
        ),
        "alignment_guidance": _evidence_alignment_guidance(
            claim=claim,
            access_class=access_class,
            capabilities=analysis["capabilities"],
            scoring_role=scoring_role,
        ),
        "not_acceptable": _not_acceptable(access_class),
        "rubric_0_0": _evidence_rubric_0_0(access_class),
        "rubric_0_5": _evidence_rubric_0_5(access_class, analysis["capabilities"]),
        "rubric_1_0": _evidence_rubric_1_0(access_class, analysis["capabilities"]),
        "scoreability_rationale": _scoreability_rationale(
            access_class=access_class,
            scoring_role=scoring_role,
        ),
        "_selection": {
            "priority": _evidence_priority(access_class, analysis["capabilities"], claim),
            "mechanism_key": _mechanism_key(links, claim),
            "source_order": index,
        },
        "_audit": {
            "target_type": "evidence",
            "source_ids": [source_id],
            "source_text": claim,
            "reference_accessibility": access_text,
            "assigned_access_class": access_class,
            "assigned_scoring_role": scoring_role,
            "assigned_capabilities": analysis["capabilities"],
            "classification_reason": analysis["reason"],
        },
    }


def _semantic_diagnosis_target(
    unit: JsonDict,
    *,
    index: int,
    evidence_by_id: dict[str, JsonDict],
) -> JsonDict:
    source_id = _clean_string(unit.get("diagnosis_id")) or f"D{index:03d}"
    mechanism = _clean_string(unit.get("mechanism"))
    status = _clean_string(unit.get("reference_status"))
    role = _clean_string(unit.get("diagnosis_role"))
    role_class = _diagnosis_role_class(status, role)
    capabilities = _diagnosis_capabilities(unit, evidence_by_id)
    access_class = _access_class_from_capabilities(capabilities)
    scoring_role = _diagnosis_scoring_role(access_class, role_class)
    response_level = _diagnosis_response_level(scoring_role)
    target_kind = _target_kind_for_scoring_role(scoring_role, default="diagnosis")
    return {
        "target_id": source_id,
        "source_ids": [source_id],
        "target_kind": target_kind,
        "access_class": access_class,
        "required_capabilities": capabilities,
        "scoring_role": scoring_role,
        "acceptable_response_level": response_level,
        "diagnosis_direction": role_class,
        "claim_text": _diagnosis_claim_text(
            mechanism=mechanism,
            access_class=access_class,
            scoring_role=scoring_role,
        ),
        "alignment_guidance": _diagnosis_alignment_guidance(
            mechanism=mechanism,
            role_class=role_class,
            scoring_role=scoring_role,
            access_class=access_class,
        ),
        "not_acceptable": _not_acceptable(access_class),
        "rubric_0_0": _diagnosis_rubric_0_0(role_class),
        "rubric_0_5": _diagnosis_rubric_0_5(role_class, capabilities),
        "rubric_1_0": _diagnosis_rubric_1_0(role_class, capabilities),
        "scoreability_rationale": _scoreability_rationale(
            access_class=access_class,
            scoring_role=scoring_role,
        ),
        "_selection": {
            "priority": _diagnosis_priority(role_class, role, mechanism, scoring_role),
            "mechanism_key": _mechanism_key([], mechanism),
            "source_order": index,
        },
        "_audit": {
            "target_type": "diagnosis",
            "source_ids": [source_id],
            "source_text": mechanism,
            "reference_status": status,
            "diagnosis_role": role,
            "supporting_evidence_ids": unit.get("supporting_evidence_ids") or [],
            "assigned_access_class": access_class,
            "assigned_scoring_role": scoring_role,
            "assigned_diagnosis_direction": role_class,
            "assigned_capabilities": capabilities,
            "classification_reason": {
                "role_class_terms": [status, role],
                "supporting_evidence_ids": unit.get("supporting_evidence_ids") or [],
            },
        },
    }


def _diagnosis_capabilities(
    unit: JsonDict,
    evidence_by_id: dict[str, JsonDict],
) -> list[str]:
    capabilities: list[str] = []
    for evidence_id in unit.get("supporting_evidence_ids") or []:
        evidence = evidence_by_id.get(str(evidence_id))
        if not evidence:
            continue
        analysis = _analyze_evidence_capabilities(
            access=_clean_string(evidence.get("evidence_accessibility")),
            claim=_clean_string(evidence.get("claim")),
            interpretation=_clean_string(evidence.get("mechanistic_interpretation")),
            links=[str(item) for item in evidence.get("mechanism_links") or []],
        )
        capabilities.extend(analysis["capabilities"])
    if capabilities:
        return _unique_preserving_order(capabilities)[:6]
    fallback = _analyze_evidence_capabilities(
        access="",
        claim=_clean_string(unit.get("mechanism")),
        interpretation=_clean_string(unit.get("expert_conclusion")),
        links=[],
    )
    return fallback["capabilities"][:6] or ["literature_context"]


def _analyze_evidence_capabilities(
    *,
    access: str,
    claim: str,
    interpretation: str,
    links: list[str],
) -> JsonDict:
    fields = {
        "reference_accessibility": access,
        "claim": claim,
        "mechanistic_interpretation": interpretation,
        "mechanism_links": " ".join(links),
    }
    normalized_fields = {name: _normalize_text(value) for name, value in fields.items()}
    matched_capabilities: list[JsonDict] = []
    capabilities: list[str] = []
    for capability, terms in _CAPABILITY_RULES:
        matches = _matched_terms_by_field(normalized_fields, terms)
        if capability == "solvent_pl_spectroscopy":
            matches = _filter_generic_pl_matches(matches)
        if not matches:
            continue
        capabilities.append(capability)
        matched_capabilities.append(
            {
                "capability": capability,
                "category": _CAPABILITY_CATEGORIES[capability],
                "matched_terms": sorted({term for terms in matches.values() for term in terms}),
                "matched_fields": sorted(matches),
            }
        )
    negative_matches = _matched_terms_by_field(normalized_fields, _NEGATIVE_ACCESS_TERMS)
    capabilities = _unique_preserving_order(capabilities)
    if negative_matches:
        capabilities = [item for item in capabilities if item != "smiles_structure"]
        matched_capabilities = [
            item
            for item in matched_capabilities
            if item.get("capability") != "smiles_structure"
        ]
    if negative_matches and not _has_non_structure_capability(capabilities):
        capabilities.append("literature_context")
        matched_capabilities.append(
            {
                "capability": "literature_context",
                "category": "source",
                "matched_terms": sorted(
                    {term for terms in negative_matches.values() for term in terms}
                ),
                "matched_fields": sorted(negative_matches),
            }
        )
    if not capabilities:
        capabilities = ["literature_context"]
        matched_capabilities.append(
            {
                "capability": "literature_context",
                "category": "source",
                "matched_terms": [],
                "matched_fields": [],
            }
        )
    return {
        "capabilities": capabilities,
        "reason": {
            "matched_capabilities": matched_capabilities,
            "negative_access_terms": {
                field: terms for field, terms in sorted(negative_matches.items())
            },
        },
    }


def validate_semantic_targets(
    *,
    evidence_targets: list[JsonDict],
    diagnosis_targets: list[JsonDict],
    audit_items: list[JsonDict],
) -> JsonDict:
    issues: list[JsonDict] = []
    audit_by_id = {str(item.get("target_id")): item for item in audit_items}
    for target in evidence_targets:
        target_id = str(target.get("target_id"))
        issues.extend(_validate_v3_target_fields(target, metric_name="EA"))
        access_class = str(target.get("access_class") or "")
        capabilities = [str(item) for item in target.get("required_capabilities") or []]
        expected_access = _access_class_from_capabilities(capabilities)
        if access_class != expected_access:
            issues.append(
                _validation_issue(
                    target_id,
                    "access_class_capability_mismatch",
                    f"access_class={access_class} expected={expected_access}",
                )
            )
        audit = audit_by_id.get(target_id, {})
        negative_terms = _dict(_dict(audit.get("classification_reason")).get(
            "negative_access_terms"
        ))
        if access_class == "structure_proxy" and negative_terms:
            issues.append(
                _validation_issue(
                    target_id,
                    "negative_smiles_access_misclassified",
                    "negative SMILES/source accessibility terms cannot be pure structure_proxy",
                )
            )
        if access_class in {"wet_lab_required", "application_context_required"}:
            _append_boundary_rubric_issue_if_missing(
                issues,
                target,
                words=("measurement", "experiment", "follow-up", "spectroscopy", "imaging"),
            )
        if access_class == "high_level_compute_required":
            _append_boundary_rubric_issue_if_missing(
                issues,
                target,
                words=("compute", "calculation", "ci", "nonadiabatic", "excited-state"),
            )
    for target in diagnosis_targets:
        target_id = str(target.get("target_id"))
        issues.extend(_validate_v3_target_fields(target, metric_name="DA"))
        direction = str(target.get("diagnosis_direction") or "")
        if direction not in {
            "supported",
            "weakened_or_rejected",
            "underdetermined_or_followup",
        }:
            issues.append(
                _validation_issue(
                    target_id,
                    "diagnosis_direction_invalid",
                    "diagnosis_direction must be supported/weakened_or_rejected/underdetermined",
                )
            )
    return {
        "status": "passed" if not issues else "failed",
        "issue_count": len(issues),
        "issues": issues,
    }


_V3_TARGET_FIELDS = {
    "target_id",
    "source_ids",
    "target_kind",
    "access_class",
    "required_capabilities",
    "scoring_role",
    "acceptable_response_level",
    "claim_text",
    "alignment_guidance",
    "not_acceptable",
    "rubric_0_0",
    "rubric_0_5",
    "rubric_1_0",
    "scoreability_rationale",
}
_V3_DIAGNOSIS_EXTRA_FIELDS = {"diagnosis_direction"}
_LEGACY_TARGET_FIELDS = {"access", "capabilities", "target", "rubric"}


def _validate_v3_target_fields(
    target: JsonDict,
    *,
    metric_name: Literal["EA", "DA"],
) -> list[JsonDict]:
    target_id = str(target.get("target_id"))
    expected = set(_V3_TARGET_FIELDS)
    if metric_name == "DA":
        expected |= _V3_DIAGNOSIS_EXTRA_FIELDS
    issues: list[JsonDict] = []
    legacy = sorted(set(target) & _LEGACY_TARGET_FIELDS)
    if legacy:
        issues.append(
            _validation_issue(
                target_id,
                "legacy_target_fields_present",
                "v3 semantic targets must not include legacy fields: " + ", ".join(legacy),
            )
        )
    missing = sorted(expected - set(target))
    if missing:
        issues.append(
            _validation_issue(
                target_id,
                "v3_target_fields_missing",
                "missing fields: " + ", ".join(missing),
            )
        )
    role = str(target.get("scoring_role") or "")
    if role not in {"primary_score", "boundary_score", "must_not_claim_only", "oracle_only"}:
        issues.append(
            _validation_issue(
                target_id,
                "scoring_role_invalid",
                f"invalid scoring_role={role!r}",
            )
        )
    access_class = str(target.get("access_class") or "")
    if access_class not in {
        "structure_proxy",
        "low_cost_compute_proxy",
        "generic_literature",
        "wet_lab_required",
        "high_level_compute_required",
        "application_context_required",
        "source_paper_only",
    }:
        issues.append(
            _validation_issue(
                target_id,
                "access_class_invalid",
                f"invalid access_class={access_class!r}",
            )
        )
    return issues


def _append_boundary_rubric_issue_if_missing(
    issues: list[JsonDict],
    target: JsonDict,
    *,
    words: tuple[str, ...],
) -> None:
    rubric_text = _normalize_text(
        " ".join(
            str(target.get(key) or "")
            for key in ("rubric_0_5", "rubric_1_0", "alignment_guidance")
        )
    )
    if _contains_any(rubric_text, words):
        return
    issues.append(
        _validation_issue(
            str(target.get("target_id")),
            "boundary_rubric_missing_followup_language",
            "boundary target rubric must name the required follow-up family",
        )
    )


def _target_kind_for_scoring_role(
    scoring_role: ScoringRole,
    *,
    default: Literal["evidence", "diagnosis"],
) -> TargetKind:
    if scoring_role == "must_not_claim_only":
        return "must_not_claim"
    if scoring_role == "boundary_score":
        return "boundary"
    return default


def _evidence_scoring_role(access_class: AccessClass) -> ScoringRole:
    if access_class in {"structure_proxy", "low_cost_compute_proxy"}:
        return "primary_score"
    if access_class == "source_paper_only":
        return "oracle_only"
    return "boundary_score"


def _diagnosis_scoring_role(
    access_class: AccessClass,
    role_class: DiagnosisAccess,
) -> ScoringRole:
    if access_class in {"source_paper_only", "generic_literature"}:
        return "oracle_only"
    if access_class == "high_level_compute_required":
        return "boundary_score"
    if access_class in {"wet_lab_required", "application_context_required"}:
        return "boundary_score"
    if role_class == "underdetermined_or_followup":
        return "boundary_score"
    return "primary_score"


def _evidence_response_level(access_class: AccessClass) -> AcceptableResponseLevel:
    if access_class in {"structure_proxy", "low_cost_compute_proxy"}:
        return "proxy_observable"
    if access_class == "source_paper_only":
        return "exact_measurement"
    return "protocol_followup"


def _diagnosis_response_level(scoring_role: ScoringRole) -> AcceptableResponseLevel:
    if scoring_role == "primary_score":
        return "mechanism_family"
    if scoring_role == "oracle_only":
        return "exact_measurement"
    return "protocol_followup"


def _evidence_alignment_guidance(
    *,
    claim: str,
    access_class: AccessClass,
    capabilities: list[str],
    scoring_role: ScoringRole,
) -> str:
    if scoring_role == "primary_score":
        return (
            "Credit structure-derived or low-cost proxy evidence that names the same "
            f"observable or mechanism family without claiming hidden paper results: {claim}"
        )
    if scoring_role == "oracle_only":
        return (
            "This is a source-paper-specific fact. It is retained for audit/oracle "
            "settings and is excluded from SMILES-only recall."
        )
    capability_text = _capability_boundary_text(capabilities)
    if access_class == "high_level_compute_required":
        return (
            "Credit recognizing a high-level excited-state/nonadiabatic computation "
            f"boundary for {capability_text} before claiming the paper result."
        )
    return (
        "Credit recognizing the same observable family and naming the required wet-lab, "
        f"material, or application-context follow-up for {capability_text} before "
        "claiming the paper result."
    )


def _evidence_claim_text(
    *,
    claim: str,
    access_class: AccessClass,
    capabilities: list[str],
    scoring_role: ScoringRole,
) -> str:
    if scoring_role == "primary_score":
        return claim
    capability_text = _capability_boundary_text(capabilities)
    if scoring_role == "oracle_only":
        return f"Source-paper-only evidence target for {capability_text}."
    if access_class == "high_level_compute_required":
        return (
            "Boundary target: high-level excited-state or nonadiabatic computation "
            f"is required to assess {capability_text}."
        )
    if access_class == "application_context_required":
        return (
            "Boundary target: biological/imaging application evidence is required "
            f"to assess {capability_text}."
        )
    return (
        "Boundary target: wet-lab or material-characterization evidence is required "
        f"to assess {capability_text}."
    )


def _diagnosis_claim_text(
    *,
    mechanism: str,
    access_class: AccessClass,
    scoring_role: ScoringRole,
) -> str:
    if scoring_role == "primary_score":
        return mechanism
    if scoring_role == "oracle_only":
        return f"Source-paper-only diagnosis target: {mechanism}"
    if access_class == "high_level_compute_required":
        return f"High-level computation boundary for diagnosis: {mechanism}"
    return f"Experimental or application-evidence boundary for diagnosis: {mechanism}"


def _diagnosis_alignment_guidance(
    *,
    mechanism: str,
    role_class: DiagnosisAccess,
    scoring_role: ScoringRole,
    access_class: AccessClass,
) -> str:
    direction = {
        "supported": "support or prioritize",
        "weakened_or_rejected": "weaken, reject, or mark as insufficient",
        "underdetermined_or_followup": "mark as underdetermined or requiring follow-up",
    }[role_class]
    if scoring_role == "primary_score":
        return (
            f"Credit a mechanism-family diagnosis that correctly {direction}s this "
            f"hypothesis from available proxy evidence: {mechanism}"
        )
    if scoring_role == "oracle_only":
        return (
            "This diagnosis is source-paper-specific and excluded from SMILES-only "
            f"recall unless literature/oracle input is provided: {mechanism}"
        )
    if access_class == "high_level_compute_required":
        return (
            f"Credit identifying that this mechanistic question should be {direction} "
            "only after high-level excited-state/nonadiabatic computation, not from "
            f"SMILES alone: {mechanism}"
        )
    return (
        f"Credit identifying that this mechanism should be {direction} only with the "
        f"named experimental/application evidence, not from SMILES alone: {mechanism}"
    )


def _not_acceptable(access_class: AccessClass) -> list[str]:
    common = [
        (
            "claiming exact paper numbers, spectra, microscopy, DLS, crystal data, "
            "or CI results as newly observed"
        ),
        "matching only generic AIE wording without the target-specific mechanism or boundary",
    ]
    if access_class in {"structure_proxy", "low_cost_compute_proxy"}:
        return [
            *common,
            "treating a proxy-compatible motif as confirmed wet-lab behavior",
        ]
    if access_class == "high_level_compute_required":
        return [
            *common,
            "asserting CI/RACI/nonadiabatic pathway details without high-level computation",
        ]
    if access_class in {"wet_lab_required", "application_context_required"}:
        return [
            *common,
            (
                "asserting biological, imaging, aggregation, or photophysical "
                "measurements without experiment"
            ),
        ]
    return [
        *common,
        "using source-paper-only wording as if it was inferred from public SMILES",
    ]


def _evidence_rubric_0_0(access_class: AccessClass) -> str:
    if access_class in {"structure_proxy", "low_cost_compute_proxy"}:
        return (
            "No same mechanism/proxy observable, wrong causal direction, or "
            "overclaimed hidden result."
        )
    return (
        "No same boundary/observable family, generic follow-up only, or "
        "overclaimed unavailable result."
    )


def _evidence_rubric_0_5(access_class: AccessClass, capabilities: list[str]) -> str:
    capability_text = _capability_text(capabilities)
    if access_class in {"structure_proxy", "low_cost_compute_proxy"}:
        return (
            "Names the same mechanism family or proxy observable with explicit limits, "
            f"but misses a target-specific detail. Relevant capability: {capability_text}."
        )
    return (
        "Names the same mechanism family and a relevant follow-up boundary, but misses "
        f"the specific observable or environment. Relevant capability: {capability_text}."
    )


def _evidence_rubric_1_0(access_class: AccessClass, capabilities: list[str]) -> str:
    capability_text = _capability_text(capabilities)
    if access_class in {"structure_proxy", "low_cost_compute_proxy"}:
        return (
            "Provides target-specific structure/proxy evidence, correct causal direction, "
            f"and no hidden-result overclaim. Relevant capability: {capability_text}."
        )
    return (
        "Names the target-specific observable/mechanism and the exact required "
        f"experiment, material characterization, application assay, or computation. "
        f"Relevant capability: {capability_text}."
    )


def _diagnosis_rubric_0_0(role_class: DiagnosisAccess) -> str:
    return (
        f"Wrong mechanism, wrong diagnosis direction ({role_class}), generic status "
        "overlap, or overclaim."
    )


def _diagnosis_rubric_0_5(role_class: DiagnosisAccess, capabilities: list[str]) -> str:
    return (
        f"Same mechanism family with correct {role_class} direction, but missing a "
        "target-specific proxy/boundary detail. Relevant capability: "
        f"{_capability_text(capabilities)}."
    )


def _diagnosis_rubric_1_0(role_class: DiagnosisAccess, capabilities: list[str]) -> str:
    return (
        f"Same mechanism question with correct {role_class} direction, target-specific "
        f"proxy or boundary, and no unavailable-result overclaim. Relevant capability: "
        f"{_capability_text(capabilities)}."
    )


def _scoreability_rationale(
    *,
    access_class: AccessClass,
    scoring_role: ScoringRole,
) -> str:
    if scoring_role == "primary_score":
        return (
            f"{access_class} target is scoreable from SMILES or allowed low-cost proxy "
            "evidence because it asks for mechanism-family/proxy reasoning, not exact paper facts."
        )
    if scoring_role == "boundary_score":
        return (
            f"{access_class} target is scoreable as a boundary/follow-up judgment: the "
            "agent should name the required evidence before claiming the paper fact."
        )
    if scoring_role == "must_not_claim_only":
        return "Target is retained only as an overclaim constraint, not a recall objective."
    return (
        f"{access_class} target is source-paper/oracle information and is excluded from "
        "SMILES-only EA/DA recall."
    )


def _capability_text(capabilities: list[str]) -> str:
    return ", ".join(capabilities[:4]) if capabilities else "none"


def _capability_boundary_text(capabilities: list[str]) -> str:
    display_capabilities = [
        item
        for item in capabilities
        if _CAPABILITY_CATEGORIES.get(item) != "source"
    ] or capabilities
    labels = [
        _CAPABILITY_BOUNDARY_LABELS.get(item, item)
        for item in display_capabilities[:3]
    ]
    return ", ".join(labels) if labels else "the relevant mechanism/observable"


def _evidence_target_text(claim: str, access: str) -> str:
    prefix = {
        "structure_proxy": "Use SMILES-derived or low-cost proxy evidence to align with",
        "experiment_boundary": (
            "Identify the same mechanism or observable and state the required "
            "experiment/material measurement before claiming"
        ),
        "compute_boundary": (
            "Identify the same mechanism and state the required high-level "
            "calculation before claiming"
        ),
        "source_boundary": (
            "Identify the mechanism family or source-identity boundary without "
            "claiming the paper fact as newly observed"
        ),
        "mixed_boundary": (
            "Identify the same mechanism and state the required experiment and/or "
            "calculation before claiming"
        ),
    }.get(access, "Identify the same mechanism family with explicit limits")
    return f"{prefix}: {claim}"


def _diagnosis_target_text(mechanism: str, role_class: str) -> str:
    if role_class == "supported":
        return (
            "Support or prioritize this mechanism family when SMILES-derived "
            f"evidence justifies it, with explicit limits: {mechanism}"
        )
    if role_class == "weakened_or_rejected":
        return (
            "Weaken, reject, or mark this mechanism as insufficient when the "
            f"SMILES-derived differential diagnosis does not support it: {mechanism}"
        )
    return (
        "Mark this mechanism or follow-up requirement as underdetermined or necessary "
        f"rather than overclaiming it from SMILES alone: {mechanism}"
    )


def _evidence_rubric(capabilities: list[str], access: str) -> list[str]:
    rubric = [
        "same mechanism family or specific observable",
        "same causal direction or explicit boundary",
    ]
    for capability in capabilities:
        item = _CAPABILITY_RUBRIC.get(capability)
        if item:
            rubric.append(item)
    if access == "experiment_boundary":
        rubric.append("states the relevant measurement or experiment is required")
    elif access == "compute_boundary":
        rubric.append("states the relevant high-level calculation is required")
    elif access == "mixed_boundary":
        rubric.append("states the relevant experiment and calculation boundaries")
    elif access == "source_boundary":
        rubric.append("does not claim source-only or literature-only facts as observed")
    else:
        rubric.append("uses structure-derived or computed proxy evidence")
    rubric.append("does not claim source-specific results as produced by MAS")
    return _unique_preserving_order(rubric)


def _diagnosis_rubric(role_class: str, capabilities: list[str]) -> list[str]:
    if role_class == "supported":
        rubric = [
            "supports same mechanism family",
            "uses available SMILES-derived evidence or proxy",
            "states missing source-level evidence when needed",
        ]
    elif role_class == "weakened_or_rejected":
        rubric = [
            "weakens or rejects the same alternative",
            "marks the same single-mechanism explanation as incomplete",
            "does not overclaim hidden source facts",
        ]
    else:
        rubric = [
            "marks the same mechanism or follow-up as underdetermined",
            "lists the required follow-up instead of overclaiming",
            "does not overclaim unavailable experiments or calculations",
        ]
    if capabilities:
        rubric.append("uses or names relevant capability boundary: " + ", ".join(capabilities[:3]))
    return rubric


_CAPABILITY_RUBRIC = {
    "smiles_structure": "names the relevant scaffold, functional group, or structure proxy",
    "donor_acceptor_proxy": "names donor-acceptor or charge-transfer structural logic",
    "lipophilicity_proxy": "names lipophilicity, hydrophobicity, or partitioning proxy",
    "rotor_or_torsion_proxy": "names rotor, torsion, RIM/RIR/RIV, or motion restriction",
    "frontier_or_charge_proxy": "names frontier-orbital, charge, CT, or cationic proxy",
    "solvent_pl_spectroscopy": (
        "states solvent/PL spectra or peak positions require measurement"
    ),
    "qy_or_lifetime_measurement": "states QY or lifetime requires measurement",
    "dls_aggregation": "states DLS or equivalent aggregation measurement is required",
    "viscosity_pl": "states viscosity-dependent PL measurement is required",
    "bioimaging_colocalization": "states colocalization microscopy is required",
    "bioimaging_perturbation": "states perturbation bioimaging is required",
    "pxrd_or_crystal_characterization": "states PXRD/crystal/packing evidence is required",
    "gas_pressure_photophysics": "states gas-pressure photophysics measurement is required",
    "photochemistry_product_analysis": "states photochemistry product analysis is required",
    "ultrafast_spectroscopy": "states ultrafast/time-resolved spectroscopy is required",
    "td_dft_or_orbital_analysis": "states DFT/TD-DFT/orbital calculation is required",
    "excited_state_scan": "states excited-state potential-energy scan is required",
    "conical_intersection_search": "states CI search or equivalent high-level compute is required",
    "nonadiabatic_dynamics": "states nonadiabatic/surface-hopping dynamics is required",
    "crystal_qmmm_or_packing_compute": "states crystal/QM-MM/packing compute is required",
    "source_identity": "does not claim source identity without literature/source lookup",
    "literature_context": "does not claim literature-only context as newly observed",
}

_CAPABILITY_BOUNDARY_LABELS = {
    "smiles_structure": "SMILES-visible scaffold or functional-group evidence",
    "donor_acceptor_proxy": "donor-acceptor or charge-transfer structural logic",
    "lipophilicity_proxy": "lipophilicity or partitioning proxy",
    "rotor_or_torsion_proxy": "rotor/torsion or RIM/RIR/RIV proxy",
    "frontier_or_charge_proxy": "frontier-orbital, charge, CT, or cationic proxy",
    "solvent_pl_spectroscopy": "solvent or aggregation photoluminescence spectroscopy",
    "qy_or_lifetime_measurement": "quantum-yield or lifetime measurement",
    "dls_aggregation": "DLS or equivalent aggregation measurement",
    "viscosity_pl": "viscosity-dependent photoluminescence measurement",
    "bioimaging_colocalization": "bioimaging colocalization microscopy",
    "bioimaging_perturbation": "live-cell perturbation imaging",
    "pxrd_or_crystal_characterization": "PXRD, crystal, or packing characterization",
    "gas_pressure_photophysics": "gas-pressure photophysics measurement",
    "photochemistry_product_analysis": "photochemistry product analysis",
    "ultrafast_spectroscopy": "ultrafast or time-resolved spectroscopy",
    "td_dft_or_orbital_analysis": "DFT/TD-DFT or orbital analysis",
    "excited_state_scan": "excited-state potential-energy scan",
    "conical_intersection_search": "conical-intersection search",
    "nonadiabatic_dynamics": "nonadiabatic or surface-hopping dynamics",
    "crystal_qmmm_or_packing_compute": "crystal, QM/MM, or packing computation",
    "source_identity": "source identity or literature lookup",
    "literature_context": "source-paper or literature-only context",
}


def _access_class_from_capabilities(capabilities: list[str]) -> AccessClass:
    categories = _capability_categories(capabilities)
    if "compute" in categories:
        return "high_level_compute_required"
    if any(
        capability in capabilities
        for capability in ("bioimaging_colocalization", "bioimaging_perturbation")
    ):
        return "application_context_required"
    if "experiment" in categories:
        return "wet_lab_required"
    if "source" in categories:
        return "source_paper_only"
    return "structure_proxy"


def _capability_categories(capabilities: list[str]) -> set[str]:
    return {
        _CAPABILITY_CATEGORIES[item]
        for item in capabilities
        if item in _CAPABILITY_CATEGORIES
    }


def _has_non_structure_capability(capabilities: list[str]) -> bool:
    return bool(_capability_categories(capabilities) - {"structure"})


def _diagnosis_role_class(status: str, role: str) -> DiagnosisAccess:
    haystack = _normalize_text(f"{status} {role}")
    if _contains_any(haystack, _WEAKENED_ROLE_TERMS):
        return "weakened_or_rejected"
    if _contains_any(haystack, _UNDERDETERMINED_ROLE_TERMS):
        return "underdetermined_or_followup"
    if _contains_any(haystack, _SUPPORTED_ROLE_TERMS):
        return "supported"
    return "supported" if status == "supported" else "underdetermined_or_followup"


def _public_input_with_normalized_smiles(public_input: JsonDict, smiles: str) -> JsonDict:
    updated = deepcopy(public_input)
    updated["normalized_structure"] = {
        "format": "smiles",
        "smiles": smiles,
        "source_path": "public_input.molecule.structure.value",
        "normalization": "copied_without_semantic_change",
    }
    return updated


def _iter_case_documents(input_path: Path):
    path = input_path.expanduser().resolve()
    if path.is_dir():
        for file_path in sorted(path.glob("*.json")):
            if file_path.name in {"correction_summary.json", "correction_audit.json"}:
                continue
            yield file_path.name, json.loads(file_path.read_text(encoding="utf-8"))
        return
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(item for item in archive.namelist() if item.endswith(".json")):
                with archive.open(name) as handle:
                    yield Path(name).name, json.loads(handle.read().decode("utf-8"))
        return
    if path.suffix == ".json":
        yield path.name, json.loads(path.read_text(encoding="utf-8"))
        return
    raise ValueError(f"Unsupported case input path: {path}")


def _select_diverse_targets(candidates: list[JsonDict], *, max_targets: int) -> list[JsonDict]:
    if len(candidates) <= max_targets:
        return candidates
    ordered = sorted(
        candidates,
        key=lambda item: (
            _dict(item.get("_selection")).get("priority", 99),
            _dict(item.get("_selection")).get("source_order", 999),
        ),
    )
    selected: list[JsonDict] = []
    used_mechanisms: set[str] = set()
    for item in ordered:
        mechanism = str(_dict(item.get("_selection")).get("mechanism_key") or "")
        if mechanism in used_mechanisms:
            continue
        selected.append(item)
        used_mechanisms.add(mechanism)
        if len(selected) >= max_targets:
            break
    for item in ordered:
        if len(selected) >= max_targets:
            break
        if item not in selected:
            selected.append(item)
    return sorted(
        selected,
        key=lambda item: _dict(item.get("_selection")).get("source_order", 999),
    )


def _select_diagnosis_targets(candidates: list[JsonDict], *, max_targets: int) -> list[JsonDict]:
    if len(candidates) <= max_targets:
        return candidates
    selected: list[JsonDict] = []
    for role_class in ("supported", "weakened_or_rejected", "underdetermined_or_followup"):
        bucket = [item for item in candidates if item.get("diagnosis_direction") == role_class]
        bucket = sorted(
            bucket,
            key=lambda item: (
                _dict(item.get("_selection")).get("priority", 99),
                _dict(item.get("_selection")).get("source_order", 999),
            ),
        )
        if bucket:
            selected.append(bucket[0])
    ordered = sorted(
        candidates,
        key=lambda item: (
            _dict(item.get("_selection")).get("priority", 99),
            _dict(item.get("_selection")).get("source_order", 999),
        ),
    )
    for item in ordered:
        if len(selected) >= max_targets:
            break
        if item not in selected:
            selected.append(item)
    return sorted(
        selected,
        key=lambda item: _dict(item.get("_selection")).get("source_order", 999),
    )


def _renumber_target(item: JsonDict, *, prefix: str, index: int) -> JsonDict:
    updated = deepcopy(item)
    source_ids = updated.get("source_ids") or []
    source_id = str(source_ids[0]) if source_ids else str(updated.get("target_id") or index)
    target_id = f"{prefix}{index:03d}:{source_id}"
    updated["target_id"] = target_id
    if isinstance(updated.get("_audit"), dict):
        updated["_audit"]["target_id"] = target_id
    return updated


def _strip_private(item: JsonDict) -> JsonDict:
    cleaned = dict(item)
    cleaned.pop("_selection", None)
    cleaned.pop("_audit", None)
    return cleaned


def _audit_item(item: JsonDict) -> JsonDict:
    audit = dict(_dict(item.get("_audit")))
    audit["target_id"] = item.get("target_id")
    return audit


def _evidence_priority(access_class: str, capabilities: list[str], claim: str) -> int:
    priority = {
        "structure_proxy": 0,
        "low_cost_compute_proxy": 1,
        "high_level_compute_required": 2,
        "wet_lab_required": 3,
        "application_context_required": 4,
        "generic_literature": 5,
        "source_paper_only": 6,
    }.get(access_class, 7)
    lowered = claim.lower()
    if capabilities:
        priority -= 1
    if any(term in lowered for term in ("mechanism", "quenching", "emission", "aie")):
        priority -= 1
    return priority


def _diagnosis_priority(
    role_class: str,
    role: str,
    mechanism: str,
    scoring_role: str,
) -> int:
    priority = {
        "supported": 0,
        "weakened_or_rejected": 1,
        "underdetermined_or_followup": 2,
    }.get(role_class, 3)
    priority += {
        "primary_score": 0,
        "boundary_score": 1,
        "must_not_claim_only": 3,
        "oracle_only": 4,
    }.get(scoring_role, 5)
    lowered = f"{role} {mechanism}".lower()
    if any(term in lowered for term in ("primary", "final", "core")):
        priority -= 1
    return priority


def _mechanism_key(links: list[str], text: str) -> str:
    if links:
        return _normalize_token(links[0])
    tokens = [
        "esipt",
        "tict",
        "raci",
        "conical",
        "rim",
        "rir",
        "photocyclization",
        "aggregation",
        "crystal",
        "charge",
        "clusteroluminescence",
        "solvent",
        "viscosity",
    ]
    lowered = text.lower()
    for token in tokens:
        if token in lowered:
            return token
    words = [_normalize_token(item) for item in text.split()[:4]]
    return "-".join(item for item in words if item) or "unknown"


def _matched_terms_by_field(
    fields: dict[str, str],
    terms: tuple[str, ...],
) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for field, text in fields.items():
        field_matches = [term for term in terms if _term_in_text(term, text)]
        if field_matches:
            matches[field] = field_matches
    return matches


def _filter_generic_pl_matches(matches: dict[str, list[str]]) -> dict[str, list[str]]:
    generic_terms = {"emission", "fluorescence", "photoluminescence", "absorption"}
    filtered: dict[str, list[str]] = {}
    for field, terms in matches.items():
        kept = [
            term
            for term in terms
            if term not in generic_terms
            or field in {"claim", "reference_accessibility"}
        ]
        if kept:
            filtered[field] = kept
    return filtered


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(_term_in_text(needle, text) for needle in needles)


def _term_in_text(term: str, text: str) -> bool:
    normalized = _normalize_text(term)
    if not normalized:
        return False
    return f" {normalized} " in f" {text} "


def _normalize_text(value: str) -> str:
    return (
        str(value or "")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("–", " ")
        .replace("—", " ")
    )


def _normalize_token(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _unique_preserving_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _validation_issue(target_id: str, code: str, message: str) -> JsonDict:
    return {"target_id": target_id, "code": code, "message": message}


def _dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[JsonDict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _clean_string(value: Any) -> str:
    return str(value or "").strip()
