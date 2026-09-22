from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mechcal.eval.baseline_adapter import (
    BaselineCommonNarrativeAdapter,
    render_baseline_adapter_packet,
)
from mechcal.eval.case_correction import (
    CASE_CORRECTION_POLICY_VERSION,
    correct_case_document,
    correct_case_path,
)
from mechcal.eval.extractors import (
    FreeTextToSchemaExtractor,
    normalized_prediction_from_payload,
    payload_from_structure_llm_json,
)
from mechcal.eval.mas import (
    MASNarrativeToSchemaExtractor,
    build_mas_strict_source_text,
    narrative_prediction_from_case_run,
    prediction_from_case_run,
    strict_common_prediction_from_case_run,
)
from mechcal.eval.mechanism_baselines import _normalize_native_evidence
from mechcal.eval.metrics import LLMSemanticAlignmentJudge, score_prediction
from mechcal.eval.schemas import EvalCase, SubjectRawOutput
from mechcal.eval.subjects import (
    EvidenceFormattedReactToolLLMFullBaselineSubject,
    ReactToolLLMBaselineSubject,
    ReactToolLLMFullBaselineSubject,
)
from mechcal.runtime.llm import _response_metadata
from mechcal.schemas import (
    ArtifactManifest,
    CaseInput,
    CaseRun,
    ConclusionLedger,
    DiagnosisConclusion,
    EvidenceConclusion,
    EvidenceLedger,
    EvidenceUnit,
    FinalAnswer,
    MechanismPrediction,
    MechanismPredictionEvidence,
)


def test_normalize_native_evidence_accepts_list_of_records() -> None:
    evidence = _normalize_native_evidence(
        [
            {
                "Finding": "A rigid aromatic core is present.",
                "Warrant": "The core can restrict intramolecular motion.",
                "Boundary": "Packing is not available from SMILES.",
            }
        ]
    )

    assert evidence == [
        (
            "Finding: A rigid aromatic core is present. "
            "Warrant: The core can restrict intramolecular motion. "
            "Boundary: Packing is not available from SMILES."
        )
    ]


def test_response_metadata_preserves_search_flags() -> None:
    response = SimpleNamespace(
        metadata={"search_used": False, "reasoning_used": True}
    )

    assert _response_metadata(response) == {
        "search_used": False,
        "reasoning_used": True,
    }


class _FakeJsonClient:
    settings = SimpleNamespace(model="test-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, payload, schema_feedback
        return {
            "final_summary": "ESIPT is plausible from the structure-only baseline, but unverified.",
            "evidence_units": [
                {
                    "claim": (
                        "The structure contains a phenolic donor near a "
                        "heteroaromatic carbonyl/imine acceptor motif."
                    ),
                    "context": "structure-only baseline interpretation",
                    "basis": "proxy",
                    "support": "supports",
                    "summary": (
                        "The free-text baseline identified an ESIPT-like "
                        "donor/acceptor motif."
                    ),
                    "limits": ["No computation or experiment was run."],
                    "observable": "esipt_structural_motif",
                    "family": "geometry_precondition",
                    "relation": "supports",
                    "status": "present",
                    "observable_tags": ["esipt", "structural_prior"],
                }
            ],
            "diagnosis_units": [
                {
                    "mechanism": "ESIPT",
                    "context": "structure-only baseline interpretation",
                    "status": "proxy_supported",
                    "reasoning_summary": (
                        "The extracted baseline diagnosis links the visible "
                        "donor/acceptor motif to ESIPT plausibility."
                    ),
                    "missing_or_unresolved": ["Needs spectra or computation for confirmation."],
                    "scope_limits": ["No wet-lab evidence is available."],
                }
            ],
        }


class _FakeJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        assert payload["semantic_targets"]
        prediction_id = (
            payload["prediction_payload"][0].get("evidence_id")
            or payload["prediction_payload"][0].get("diagnosis_id")
        )
        first_target_id = payload["semantic_targets"][0]["target_id"]
        candidate_pairs = [
            {
                "prediction_id": prediction_id,
                "target_id": first_target_id,
                "alignment_score": 0.8,
                "rationale": "The predicted item aligns with the first target.",
                "violation": False,
            }
        ]
        if payload["metric_name"] == "EA":
            candidate_pairs.append(
                {
                    "prediction_id": prediction_id,
                    "target_id": payload["semantic_targets"][1]["target_id"],
                    "alignment_score": 0.7,
                    "rationale": "This broad prediction also overlaps another target.",
                    "violation": False,
                }
            )
        return {
            "rationale": "Candidate pair scores were returned for one-to-one matching.",
            "candidate_pairs": candidate_pairs,
            "violations": [],
        }


class _FakeWrongDirectionJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        prediction_id = (
            payload["prediction_payload"][0].get("evidence_id")
            or payload["prediction_payload"][0].get("diagnosis_id")
        )
        return {
            "rationale": "The judge proposed a wrong-direction diagnosis match.",
            "candidate_pairs": [
                {
                    "prediction_id": prediction_id,
                    "target_id": payload["semantic_targets"][0]["target_id"],
                    "alignment_score": 1.0,
                    "rationale": "Wrong-direction match that the deterministic filter removes.",
                    "violation": False,
                }
            ],
            "violations": [],
        }


class _FakeBroadAnchorJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        prediction_id = (
            payload["prediction_payload"][0].get("evidence_id")
            or payload["prediction_payload"][0].get("diagnosis_id")
        )
        return {
            "rationale": "The judge proposed a broad 0.5 semantic overlap.",
            "candidate_pairs": [
                {
                    "prediction_id": prediction_id,
                    "target_id": payload["semantic_targets"][0]["target_id"],
                    "alignment_score": 0.5,
                    "rationale": "Broad partial overlap.",
                    "violation": False,
                }
            ],
            "violations": [],
        }


class _FakeBaselineCommonAdapterClient:
    settings = SimpleNamespace(model="adapter-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        encoded_payload = json.dumps(payload, ensure_ascii=False)
        assert "hidden_reference" not in payload
        assert "SECRET_REFERENCE_TOKEN" not in encoded_payload
        assert "TICT is plausible from the donor-acceptor scaffold" in payload["raw_packet"]
        evidence = {
            "source_ref": "RAW_TEXT",
            "source_snippet": "TICT is plausible from the donor-acceptor scaffold",
            "claim": "The donor-acceptor scaffold makes TICT plausible.",
            "context": "baseline raw answer",
            "basis": "proxy",
            "support": "supports",
            "summary": "The raw answer links donor-acceptor structure to TICT.",
            "limits": ["No spectra or computation were provided."],
            "observable": "donor_acceptor_tict_proxy",
            "family": "charge_localization",
            "relation": "supports",
            "status": "present",
            "observable_tags": ["tict", "donor_acceptor"],
        }
        return {
            "final_summary": "TICT is plausible, but unverified.",
            "evidence_units": [evidence, dict(evidence)],
            "diagnosis_units": [
                {
                    "source_ref": "RAW_TEXT",
                    "source_snippet": "TICT is plausible",
                    "mechanism": "TICT",
                    "context": "baseline raw answer",
                    "status": "plausible",
                    "reasoning_summary": "The raw answer proposes TICT as plausible.",
                    "missing_or_unresolved": ["Needs spectra or computation."],
                    "scope_limits": ["Structure-only baseline."],
                }
            ],
        }


class _FakeBaselineBoundaryStatusAdapterClient:
    settings = SimpleNamespace(model="adapter-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        assert "RIM/RIR is proxy-supported" in payload["raw_packet"]
        assert "Wet-lab boundary is boundary-only" in payload["raw_packet"]
        return {
            "final_summary": "RIM is structurally supported; wet-lab work is boundary-only.",
            "evidence_units": [
                {
                    "source_ref": "RAW_TEXT",
                    "source_snippet": (
                        "RIM/RIR is proxy-supported; Wet-lab boundary is boundary-only"
                    ),
                    "claim": "The raw answer gives RIM/RIR a proxy-supported status.",
                    "context": "baseline final answer",
                    "basis": "proxy",
                    "support": "supports",
                    "summary": "RIM/RIR is structurally supported in the raw answer.",
                    "limits": ["No wet-lab restriction control."],
                    "observable": "rim_rir_proxy",
                    "family": "torsion_sensitivity",
                    "relation": "supports",
                    "status": "present",
                    "observable_tags": ["rim", "rir"],
                }
            ],
            "diagnosis_units": [
                {
                    "source_ref": "RAW_TEXT",
                    "source_snippet": "RIM/RIR is proxy-supported",
                    "mechanism": "RIM/RIR",
                    "context": "baseline final answer",
                    "status": "proxy-supported",
                    "reasoning_summary": "The raw answer uses proxy-supported status.",
                    "missing_or_unresolved": ["No wet-lab restriction control."],
                    "scope_limits": ["SMILES-only baseline."],
                },
                {
                    "source_ref": "RAW_TEXT",
                    "source_snippet": "Wet-lab boundary is boundary-only",
                    "mechanism": "wet-lab boundary",
                    "context": "baseline final answer",
                    "status": "boundary-only",
                    "reasoning_summary": "The raw answer marks wet-lab evidence as boundary-only.",
                    "missing_or_unresolved": ["Water-fraction PL and DLS are not observed."],
                    "scope_limits": ["Boundary statement only."],
                },
            ],
        }


class _FakeBaselineCommonRetryClient:
    settings = SimpleNamespace(model="adapter-model")

    def __init__(self) -> None:
        self.calls = 0
        self.feedback_seen: str | None = None

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, payload
        self.calls += 1
        if self.calls == 1:
            return {
                "final_summary": "TICT is plausible, but unverified.",
                "evidence_units": [
                    {
                        "source_ref": "RAW_TEXT",
                        "source_snippet": "not copied from the raw packet",
                        "claim": "The donor-acceptor scaffold makes TICT plausible.",
                        "context": "baseline raw answer",
                        "basis": "proxy",
                        "support": "supports",
                        "summary": "The raw answer links donor-acceptor structure to TICT.",
                        "limits": ["No spectra or computation were provided."],
                        "observable": "donor_acceptor_tict_proxy",
                        "family": "charge_localization",
                        "relation": "supports",
                        "status": "present",
                        "observable_tags": ["tict", "donor_acceptor"],
                    }
                ],
                "diagnosis_units": [],
            }
        self.feedback_seen = schema_feedback
        return {
            "final_summary": "TICT is plausible, but unverified.",
            "evidence_units": [
                {
                    "source_ref": "RAW_TEXT",
                    "source_snippet": "TICT is plausible",
                    "claim": "The donor-acceptor scaffold makes TICT plausible.",
                    "context": "baseline raw answer",
                    "basis": "proxy",
                    "support": "supports",
                    "summary": "The raw answer links donor-acceptor structure to TICT.",
                    "limits": ["No spectra or computation were provided."],
                    "observable": "donor_acceptor_tict_proxy",
                    "family": "charge_localization",
                    "relation": "supports",
                    "status": "present",
                    "observable_tags": ["tict", "donor_acceptor"],
                }
            ],
            "diagnosis_units": [],
        }


class _FakeMASNarrativeClient:
    settings = SimpleNamespace(model="judge-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        assert "torsion scan" in payload["raw_text"]
        return {
            "final_summary": "Computed torsion evidence supports a torsional quenching pathway.",
            "evidence_units": [
                {
                    "claim": (
                        "The MAS torsion scan shows oscillator-strength collapse at a "
                        "twisted geometry."
                    ),
                    "context": "MAS final narrative",
                    "basis": "computed",
                    "support": "supports",
                    "summary": "Computed torsion evidence supports torsional nonradiative decay.",
                    "limits": ["No paper reference computation was reproduced."],
                    "observable": "torsion_brightness_coupling",
                    "family": "torsion_sensitivity",
                    "relation": "supports",
                    "status": "present",
                    "observable_tags": ["torsion", "oscillator_strength"],
                }
            ],
            "diagnosis_units": [
                {
                    "mechanism": "Torsional nonradiative decay",
                    "context": "MAS final narrative",
                    "status": "computed_supported",
                    "reasoning_summary": (
                        "The final narrative links computed torsion sensitivity to a "
                        "quenching mechanism."
                    ),
                    "missing_or_unresolved": ["Crystal packing evidence is absent."],
                    "scope_limits": ["MAS evidence is bounded to its configured tools."],
                }
            ],
        }


class _FakeStrictMASCommonAdapterClient:
    settings = SimpleNamespace(model="adapter-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        raw_packet = payload["raw_packet"]
        assert "CONCLUSION_LEDGER" in raw_packet
        assert "raw_json" not in raw_packet.lower()
        assert "hidden_reference" not in json.dumps(payload, ensure_ascii=False)
        return {
            "final_summary": "MAS explicitly registered RIM/RIR proxy support.",
            "evidence_units": [
                {
                    "source_ref": "RAW_TEXT",
                    "source_snippet": (
                        "RIM/RIR is proxy-supported by accepted rotor evidence."
                    ),
                    "claim": "Accepted MAS evidence supports a RIM/RIR proxy.",
                    "context": "MAS conclusion ledger",
                    "basis": "proxy",
                    "support": "supports",
                    "summary": "The conclusion ledger registered RIM/RIR proxy support.",
                    "limits": ["No wet-lab PL claim."],
                    "observable": "rim_rir_proxy",
                    "family": "torsion_sensitivity",
                    "relation": "supports",
                    "status": "present",
                    "observable_tags": ["rim", "rir"],
                }
            ],
            "diagnosis_units": [
                {
                    "source_ref": "RAW_TEXT",
                    "source_snippet": "RIM/RIR proxy mechanism",
                    "mechanism": "RIM/RIR proxy mechanism",
                    "context": "MAS conclusion ledger",
                    "status": "proxy_supported",
                    "reasoning_summary": "The ledger cites accepted rotor evidence.",
                    "missing_or_unresolved": ["Needs aggregation PL."],
                    "scope_limits": ["No wet-lab claim."],
                }
            ],
        }


class _FakeReactJsonClient:
    settings = SimpleNamespace(model="react-model")

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        encoded_payload = json.dumps(payload, ensure_ascii=False)
        assert "SECRET_REFERENCE_TOKEN" not in encoded_payload
        self.calls += 1
        if payload.get("force_final"):
            return {
                "thought": "Stop after the available tool observation.",
                "action": "final",
                "tool_name": None,
                "tool_args": {},
                "final_answer": (
                    "The transcript contains a structure scan observation. It supports "
                    "only bounded, structure-derived AIE screening; wet-lab and "
                    "high-level excited-state claims remain unavailable."
                ),
            }
        return {
            "thought": "Collect a broad structure scan before finalizing.",
            "action": "tool",
            "tool_name": "macro.macro_structure_scan",
            "tool_args": {},
            "final_answer": None,
        }


class _FakeFailingReactJsonClient:
    settings = SimpleNamespace(model="react-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        encoded_payload = json.dumps(payload, ensure_ascii=False)
        assert "SECRET_REFERENCE_TOKEN" not in encoded_payload
        raise RuntimeError("forced ReAct decision failure")


def test_free_text_extractor_builds_normalized_prediction() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:zero_shot_llm:raw",
        case_id="demo",
        subject_id="zero_shot_llm",
        subject_kind="zero_shot_llm",
        model="test-model",
        prompt_version="test",
        raw_text="ESIPT is plausible but unverified.",
    )

    prediction = FreeTextToSchemaExtractor(client=_FakeJsonClient()).extract(case, raw)  # type: ignore[arg-type]

    assert prediction.extractor_id == "free_text_to_schema_v1"
    assert prediction.evidence_units[0].basis == "proxy"
    assert prediction.diagnosis_units[0].status == "proxy_supported"
    assert prediction.diagnosis_units[0].evidence_refs == [prediction.evidence_units[0].evidence_id]


def test_structure_llm_payload_normalizes_common_enum_drift() -> None:
    payload = {
        "final_summary": "The structure has plausible TICT/AIE motifs but remains unresolved.",
        "evidence_units": [
            {
                "claim": "A donor-acceptor motif is visible from the SMILES.",
                "context": "structure-only baseline interpretation",
                "basis": "structure",
                "support": "supported",
                "summary": "The motif is a structure-only proxy for CT/TICT behavior.",
                "limits": ["No wet-lab data are available."],
                "observable": "donor_acceptor_topology",
                "family": "charge_localization",
                "relation": "support",
                "status": "unresolved",
                "observable_tags": ["donor_acceptor", "tict"],
            }
        ],
        "diagnosis_units": [
            {
                "mechanism": "TICT-AIE",
                "context": "structure-only baseline interpretation",
                "status": "supported",
                "reasoning_summary": "The donor-acceptor topology makes TICT-AIE plausible.",
                "missing_or_unresolved": ["Needs spectra or computation for confirmation."],
                "scope_limits": ["SMILES-only baseline."],
            }
        ],
    }

    extracted = payload_from_structure_llm_json(payload)

    assert extracted.evidence_units[0].basis == "proxy"
    assert extracted.evidence_units[0].support == "supports"
    assert extracted.evidence_units[0].relation == "supports"
    assert extracted.evidence_units[0].status == "partial"
    assert extracted.diagnosis_units[0].status == "proxy_supported"


def test_score_prediction_requires_hidden_reference_targets() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:zero_shot_llm:raw",
        case_id="demo",
        subject_id="zero_shot_llm",
        subject_kind="zero_shot_llm",
        model="test-model",
        prompt_version="test",
        raw_text="ESIPT is plausible but unverified.",
    )
    prediction = FreeTextToSchemaExtractor(client=_FakeJsonClient()).extract(case, raw)  # type: ignore[arg-type]

    metrics = score_prediction(prediction, hidden_reference=None)

    assert [metric.metric_name for metric in metrics] == ["EA", "DA"]
    assert {metric.status for metric in metrics} == {"reference_missing"}
    assert all(metric.score is None for metric in metrics)


def test_score_prediction_uses_v3_targets_for_precision_recall_f1() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:zero_shot_llm:raw",
        case_id="demo",
        subject_id="zero_shot_llm",
        subject_kind="zero_shot_llm",
        model="test-model",
        prompt_version="test",
        raw_text="ESIPT is plausible but unverified.",
    )
    prediction = FreeTextToSchemaExtractor(client=_FakeJsonClient()).extract(case, raw)  # type: ignore[arg-type]

    metrics = score_prediction(
        prediction,
        hidden_reference={
            "semantic_evidence_targets": [
                {
                    "target_id": "SE1",
                    "source_ids": ["E1"],
                    "target_kind": "evidence",
                    "access_class": "structure_proxy",
                    "required_capabilities": ["smiles_structure"],
                    "scoring_role": "primary_score",
                    "acceptable_response_level": "proxy_observable",
                    "claim_text": "ESIPT is structurally plausible.",
                    "alignment_guidance": "Donor/acceptor motif supports ESIPT.",
                    "not_acceptable": [],
                    "rubric_0_0": "wrong mechanism",
                    "rubric_0_5": "same mechanism family",
                    "rubric_1_0": "target-specific proxy",
                    "scoreability_rationale": "scoreable from SMILES proxy",
                },
                {
                    "target_id": "SE2",
                    "source_ids": ["E2"],
                    "target_kind": "evidence",
                    "access_class": "structure_proxy",
                    "required_capabilities": ["smiles_structure"],
                    "scoring_role": "primary_score",
                    "acceptable_response_level": "proxy_observable",
                    "claim_text": "ESIPT is the dominant mechanism.",
                    "alignment_guidance": "A broader second target should not be co-matched.",
                    "not_acceptable": [],
                    "rubric_0_0": "wrong mechanism",
                    "rubric_0_5": "same mechanism family",
                    "rubric_1_0": "target-specific proxy",
                    "scoreability_rationale": "scoreable from SMILES proxy",
                }
            ],
            "semantic_diagnosis_targets": [
                {
                    "target_id": "SD1",
                    "source_ids": ["D1"],
                    "target_kind": "diagnosis",
                    "access_class": "structure_proxy",
                    "required_capabilities": ["smiles_structure"],
                    "scoring_role": "primary_score",
                    "acceptable_response_level": "mechanism_family",
                    "diagnosis_direction": "supported",
                    "claim_text": "ESIPT-supported AIE",
                    "alignment_guidance": "ESIPT remains relevant but not proven.",
                    "not_acceptable": [],
                    "rubric_0_0": "wrong mechanism",
                    "rubric_0_5": "same mechanism family",
                    "rubric_1_0": "target-specific diagnosis",
                    "scoreability_rationale": "scoreable from SMILES proxy",
                }
            ],
        },
        judge=LLMSemanticAlignmentJudge(client=_FakeJudgeClient()),  # type: ignore[arg-type]
    )

    assert [metric.status for metric in metrics] == ["scored", "scored"]
    assert metrics[0].precision == 1.0
    assert metrics[0].recall == 0.5
    assert round(metrics[0].f1 or 0.0, 3) == 0.667
    assert metrics[0].score == metrics[0].f1
    assert metrics[0].matched_prediction_count == 1
    assert metrics[0].matched_target_count == 1
    assert metrics[0].false_positives == 0
    assert metrics[0].false_negatives == 1
    assert len(metrics[0].details["candidate_pairs"]) == 2
    assert len(metrics[0].details["matched_pairs"]) == 1
    assert [
        item["alignment_score"] for item in metrics[0].details["candidate_pairs"]
    ] == [1.0, 0.5]


def test_score_prediction_rejects_legacy_reference_targets() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:zero_shot_llm:raw",
        case_id="demo",
        subject_id="zero_shot_llm",
        subject_kind="zero_shot_llm",
        model="test-model",
        prompt_version="test",
        raw_text="ESIPT is plausible but unverified.",
    )
    prediction = FreeTextToSchemaExtractor(client=_FakeJsonClient()).extract(case, raw)  # type: ignore[arg-type]

    metrics = score_prediction(
        prediction,
        hidden_reference={
            "semantic_evidence_targets": [
                {
                    "target_id": "SE1",
                    "access": "structure_proxy",
                    "capabilities": ["smiles_structure"],
                    "target": "Legacy target should not score.",
                    "rubric": ["legacy"],
                }
            ],
            "semantic_diagnosis_targets": [],
        },
        judge=LLMSemanticAlignmentJudge(client=_FakeJudgeClient()),  # type: ignore[arg-type]
    )

    assert metrics[0].status == "reference_missing"


def test_da_scoring_filters_wrong_direction_matches() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:zero_shot_llm:raw",
        case_id="demo",
        subject_id="zero_shot_llm",
        subject_kind="zero_shot_llm",
        model="test-model",
        prompt_version="test",
        raw_text="ESIPT is plausible but unverified.",
    )
    prediction = FreeTextToSchemaExtractor(client=_FakeJsonClient()).extract(case, raw)  # type: ignore[arg-type]

    metrics = score_prediction(
        prediction,
        hidden_reference={
            "semantic_diagnosis_targets": [
                {
                    "target_id": "SD1",
                    "source_ids": ["D1"],
                    "target_kind": "diagnosis",
                    "access_class": "structure_proxy",
                    "required_capabilities": ["smiles_structure"],
                    "scoring_role": "primary_score",
                    "acceptable_response_level": "mechanism_family",
                    "diagnosis_direction": "weakened_or_rejected",
                    "claim_text": "Weaken or reject ESIPT.",
                    "alignment_guidance": "ESIPT should be weakened.",
                    "not_acceptable": [],
                    "rubric_0_0": "wrong direction",
                    "rubric_0_5": "same mechanism",
                    "rubric_1_0": "same mechanism and direction",
                    "scoreability_rationale": "scoreable from differential diagnosis",
                }
            ],
        },
        judge=LLMSemanticAlignmentJudge(client=_FakeWrongDirectionJudgeClient()),  # type: ignore[arg-type]
    )

    da_metric = metrics[1]
    assert da_metric.status == "scored"
    assert da_metric.true_positives == 0
    assert da_metric.false_positives == 1
    assert da_metric.false_negatives == 1
    assert da_metric.details["candidate_pairs"] == []
    assert len(da_metric.details["direction_filter_removed_pairs"]) == 1


def test_scoring_filters_broad_cross_anchor_matches() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:structure_llm:raw",
        case_id="demo",
        subject_id="structure_llm",
        subject_kind="structure_llm",
        model="test-model",
        prompt_version="test",
        raw_json={},
    )
    prediction = normalized_prediction_from_payload(
        case=case,
        raw_output=raw,
        payload=payload_from_structure_llm_json(
            {
                "final_summary": "Sulfur cluster emission is plausible.",
                "evidence_units": [
                    {
                        "claim": (
                            "Thiophene sulfur through-space clusteroluminescence may "
                            "contribute to AIE."
                        ),
                        "context": "structure-only baseline interpretation",
                        "basis": "proxy",
                        "support": "supports",
                        "summary": "Sulfur and thiophene motifs suggest cluster emission.",
                        "relation": "supports",
                        "status": "present",
                        "limits": ["No wet-lab data are available."],
                    }
                ],
                "diagnosis_units": [],
            }
        ),
    )

    metrics = score_prediction(
        prediction,
        hidden_reference={
            "semantic_evidence_targets": [
                {
                    "target_id": "SE1",
                    "source_ids": ["E1"],
                    "target_kind": "evidence",
                    "access_class": "structure_proxy",
                    "required_capabilities": ["donor_acceptor_proxy"],
                    "scoring_role": "primary_score",
                    "acceptable_response_level": "proxy_observable",
                    "claim_text": (
                        "Donor-acceptor TICT-AIE with cationic mitochondrial "
                        "acceptor and lipophilic lipid-droplet donor."
                    ),
                    "alignment_guidance": "Donor-acceptor TICT-AIE target.",
                    "not_acceptable": [],
                    "rubric_0_0": "wrong anchor",
                    "rubric_0_5": "same mechanism family",
                    "rubric_1_0": "target-specific D-A/TICT proxy",
                    "scoreability_rationale": "scoreable from SMILES proxy",
                }
            ],
        },
        judge=LLMSemanticAlignmentJudge(client=_FakeBroadAnchorJudgeClient()),  # type: ignore[arg-type]
    )

    ea_metric = metrics[0]
    assert ea_metric.true_positives == 0
    assert ea_metric.details["candidate_pairs"] == []
    assert len(ea_metric.details["strict_filter_removed_pairs"]) == 1
    assert "tict_or_charge_transfer" in ea_metric.details["strict_filter_removed_pairs"][0][
        "strict_filter_reason"
    ]


def test_scoring_keeps_same_anchor_partial_matches() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:structure_llm:raw",
        case_id="demo",
        subject_id="structure_llm",
        subject_kind="structure_llm",
        model="test-model",
        prompt_version="test",
        raw_json={},
    )
    prediction = normalized_prediction_from_payload(
        case=case,
        raw_output=raw,
        payload=payload_from_structure_llm_json(
            {
                "final_summary": "RIM is plausible from rotor restriction.",
                "evidence_units": [
                    {
                        "claim": (
                            "Restricted intramolecular motion can explain viscosity-linked "
                            "fluorescence enhancement."
                        ),
                        "context": "structure-only baseline interpretation",
                        "basis": "proxy",
                        "support": "supports",
                        "summary": "RIM/RIR is the relevant structural prior.",
                        "relation": "supports",
                        "status": "present",
                        "limits": ["No viscosity experiment was run."],
                    }
                ],
                "diagnosis_units": [],
            }
        ),
    )

    metrics = score_prediction(
        prediction,
        hidden_reference={
            "semantic_evidence_targets": [
                {
                    "target_id": "SE1",
                    "source_ids": ["E1"],
                    "target_kind": "boundary",
                    "access_class": "wet_lab_required",
                    "required_capabilities": ["viscosity_pl"],
                    "scoring_role": "boundary_score",
                    "acceptable_response_level": "protocol_followup",
                    "claim_text": "Viscosity enhances fluorescence and supports RIM.",
                    "alignment_guidance": "Viscosity-dependent PL is required.",
                    "not_acceptable": [],
                    "rubric_0_0": "wrong anchor",
                    "rubric_0_5": "same mechanism with boundary",
                    "rubric_1_0": "names viscosity PL follow-up",
                    "scoreability_rationale": "scoreable as boundary",
                }
            ],
        },
        judge=LLMSemanticAlignmentJudge(client=_FakeBroadAnchorJudgeClient()),  # type: ignore[arg-type]
    )

    ea_metric = metrics[0]
    assert ea_metric.true_positives == 1
    assert len(ea_metric.details["candidate_pairs"]) == 1
    assert ea_metric.details["strict_filter_removed_pairs"] == []


def test_strict_anchor_ignores_global_not_acceptable_terms() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:structure_llm:raw",
        case_id="demo",
        subject_id="structure_llm",
        subject_kind="structure_llm",
        model="test-model",
        prompt_version="test",
        raw_json={},
    )
    prediction = normalized_prediction_from_payload(
        case=case,
        raw_output=raw,
        payload=payload_from_structure_llm_json(
            {
                "final_summary": "Solvent PL remains a wet-lab boundary.",
                "evidence_units": [
                    {
                        "claim": (
                            "Solvent and aggregation photoluminescence spectra are "
                            "required before claiming AIE behavior."
                        ),
                        "context": "structure-only baseline interpretation",
                        "basis": "checklist",
                        "support": "unresolved",
                        "summary": "No PL measurement was available.",
                        "relation": "neutral",
                        "status": "missing",
                        "limits": ["No wet-lab data are available."],
                    }
                ],
                "diagnosis_units": [],
            }
        ),
    )

    metrics = score_prediction(
        prediction,
        hidden_reference={
            "semantic_evidence_targets": [
                {
                    "target_id": "SE1",
                    "source_ids": ["E1"],
                    "target_kind": "boundary",
                    "access_class": "wet_lab_required",
                    "required_capabilities": ["solvent_pl_spectroscopy"],
                    "scoring_role": "boundary_score",
                    "acceptable_response_level": "protocol_followup",
                    "claim_text": (
                        "Boundary target: wet-lab evidence is required to assess "
                        "solvent or aggregation photoluminescence spectroscopy."
                    ),
                    "alignment_guidance": "Credit solvent or aggregation PL follow-up.",
                    "not_acceptable": [
                        "claiming DLS, conical-intersection, or surface-hopping results"
                    ],
                    "rubric_0_0": "No same boundary.",
                    "rubric_0_5": "Same boundary family.",
                    "rubric_1_0": "Exact PL boundary.",
                    "scoreability_rationale": "scoreable as boundary",
                }
            ],
        },
        judge=LLMSemanticAlignmentJudge(client=_FakeBroadAnchorJudgeClient()),  # type: ignore[arg-type]
    )

    ea_metric = metrics[0]
    assert ea_metric.true_positives == 1
    assert ea_metric.details["strict_filter_removed_pairs"] == []


def test_baseline_common_adapter_extracts_source_grounded_units_without_hard_cap() -> None:
    case = EvalCase(
        case_id="demo",
        smiles="C1=CC=CC=C1",
        hidden_reference={"secret": "SECRET_REFERENCE_TOKEN"},
    )
    raw = SubjectRawOutput(
        raw_output_id="demo:structure_llm:raw",
        case_id="demo",
        subject_id="structure_llm",
        subject_kind="structure_llm",
        model="test-model",
        prompt_version="test",
        raw_text=(
            "TICT is plausible from the donor-acceptor scaffold, but spectra "
            "and computation are missing."
        ),
    )

    prediction = BaselineCommonNarrativeAdapter(
        client=_FakeBaselineCommonAdapterClient(),  # type: ignore[arg-type]
    ).extract(case, raw)

    assert prediction.extractor_id == "baseline_common_narrative_to_schema_v1"
    assert prediction.extractor_model == "adapter-model"
    assert len(prediction.evidence_units) == 1
    assert len(prediction.diagnosis_units) == 1
    assert prediction.metadata["adapter_role"] == "baseline_common_extraction"
    audit = prediction.metadata["grounding_audit"]
    assert audit["removed_duplicate_evidence_count"] == 1
    assert audit["grounding"][0]["source_snippet_found"] is True


def test_baseline_common_adapter_normalizes_full_react_status_words() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:react_tool_llm_full:raw",
        case_id="demo",
        subject_id="react_tool_llm_full",
        subject_kind="react_tool_llm_full",
        model="test-model",
        prompt_version="test",
        raw_text=(
            "Mechanism diagnosis: RIM/RIR is proxy-supported from Observation 2. "
            "Wet-lab boundary is boundary-only from Observation 9."
        ),
    )

    prediction = BaselineCommonNarrativeAdapter(
        client=_FakeBaselineBoundaryStatusAdapterClient(),  # type: ignore[arg-type]
    ).extract(case, raw)

    assert [unit.status for unit in prediction.diagnosis_units] == [
        "proxy_supported",
        "underdetermined",
    ]
    assert all(
        item["source_snippet_found"]
        for item in prediction.metadata["grounding_audit"]["grounding"]
    )
    repairs = prediction.metadata["grounding_audit"]["snippet_repairs"]
    assert repairs[0]["new_source_snippet"] == "RIM/RIR is proxy-supported"


def test_react_tool_llm_subject_records_transcript_without_hidden_reference(tmp_path) -> None:
    case = EvalCase(
        case_id="react_demo",
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism.",
        hidden_reference={"secret": "SECRET_REFERENCE_TOKEN"},
    )
    subject = ReactToolLLMBaselineSubject(
        client=_FakeReactJsonClient(),  # type: ignore[arg-type]
        run_base_dir=tmp_path,
        max_steps=1,
        enable_amesp=False,
    )

    raw_output = subject.run(case)

    assert raw_output.subject_kind == "react_tool_llm"
    assert raw_output.subject_id == "react_tool_llm"
    assert "Thought 1:" in raw_output.raw_text
    assert "Action 1:" in raw_output.raw_text
    assert "Observation 1:" in raw_output.raw_text
    assert "Final Answer:" in raw_output.raw_text
    assert "SECRET_REFERENCE_TOKEN" not in raw_output.raw_text
    assert "SECRET_REFERENCE_TOKEN" not in json.dumps(raw_output.raw_json, ensure_ascii=False)
    assert raw_output.metadata["hidden_reference_access"] is False
    assert raw_output.raw_json is not None
    assert (
        raw_output.raw_json["react_policy"]["state_management"]
        == "transcript_only_no_mas_hypothesis_or_evidence_ledger"
    )


def test_react_tool_llm_full_uses_coverage_fallback_without_mas_state(tmp_path) -> None:
    case = EvalCase(
        case_id="react_full_demo",
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism.",
        hidden_reference={"secret": "SECRET_REFERENCE_TOKEN"},
    )
    subject = ReactToolLLMFullBaselineSubject(
        client=_FakeFailingReactJsonClient(),  # type: ignore[arg-type]
        run_base_dir=tmp_path,
        max_steps=3,
        enable_amesp=False,
    )

    raw_output = subject.run(case)

    assert raw_output.subject_kind == "react_tool_llm_full"
    assert raw_output.subject_id == "react_tool_llm_full"
    assert "Action 1: macro.macro_structure_scan" in raw_output.raw_text
    assert "Action Input 1:" in raw_output.raw_text
    assert "Action 2: macro.screen_donor_acceptor_architecture" in raw_output.raw_text
    assert "Action 3: macro.screen_rotor_rim_prior" in raw_output.raw_text
    assert "Full ReAct Final Answer." in raw_output.raw_text
    assert "Mechanism diagnosis:" in raw_output.raw_text
    assert "RIM/RIR is" in raw_output.raw_text
    assert "ESIPT is underdetermined:" in raw_output.raw_text
    assert "scorecard" not in raw_output.raw_text.lower()
    assert "diagnosis ledger" not in raw_output.raw_text.lower()
    assert "SECRET_REFERENCE_TOKEN" not in raw_output.raw_text
    assert raw_output.raw_json is not None
    assert (
        raw_output.raw_json["react_policy"]["state_management"]
        == "transcript_only_no_mas_hypothesis_or_evidence_ledger"
    )
    assert raw_output.raw_json["react_policy"]["finalization_timeout_seconds"] == 60.0


def test_evidence_formatted_react_tool_llm_full_fallback_uses_fwb_format(
    tmp_path,
) -> None:
    case = EvalCase(
        case_id="react_full_fwb_demo",
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism.",
        hidden_reference={"secret": "SECRET_REFERENCE_TOKEN"},
    )
    subject = EvidenceFormattedReactToolLLMFullBaselineSubject(
        client=_FakeFailingReactJsonClient(),  # type: ignore[arg-type]
        run_base_dir=tmp_path,
        max_steps=3,
        enable_amesp=False,
    )

    raw_output = subject.run(case)

    assert raw_output.subject_kind == "evidence_formatted_react_tool_llm_full"
    assert raw_output.subject_id == "evidence_formatted_react_tool_llm_full"
    assert "Mechanism diagnosis:" in raw_output.raw_text
    assert "Finding:" in raw_output.raw_text
    assert "Warrant:" in raw_output.raw_text
    assert "Boundary:" in raw_output.raw_text
    assert "SECRET_REFERENCE_TOKEN" not in raw_output.raw_text
    assert raw_output.raw_json is not None
    assert raw_output.raw_json["react_policy"]["state_management"] == (
        "transcript_only_no_mas_hypothesis_or_evidence_ledger"
    )


def test_baseline_common_adapter_packet_hides_react_policy_from_extraction() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:react_tool_llm:raw",
        case_id="demo",
        subject_id="react_tool_llm",
        subject_kind="react_tool_llm",
        model="test-model",
        prompt_version="test",
        raw_text="Observation 1: tool evidence supports an ESIPT structural motif.",
        raw_json={
            "react_policy": {"tool_ids": ["SECRET_TOOL_ID"]},
            "transcript": ["SECRET_DUPLICATED_TRANSCRIPT"],
            "final_answer": "ESIPT remains a structure-derived possibility.",
        },
    )

    packet = render_baseline_adapter_packet(case, raw)

    assert "SECRET_TOOL_ID" not in packet
    assert "SECRET_DUPLICATED_TRANSCRIPT" not in packet
    assert "react_policy" not in packet
    assert "tool_ids" not in packet
    assert "ESIPT remains a structure-derived possibility." not in packet

    raw_full = raw.model_copy(
        update={
            "raw_output_id": "demo:react_tool_llm_full:raw",
            "subject_id": "react_tool_llm_full",
            "subject_kind": "react_tool_llm_full",
        }
    )
    packet_full = render_baseline_adapter_packet(case, raw_full)
    assert "SECRET_TOOL_ID" not in packet_full
    assert "SECRET_DUPLICATED_TRANSCRIPT" not in packet_full
    assert "react_policy" not in packet_full
    assert "ESIPT remains a structure-derived possibility." not in packet_full


def test_baseline_common_adapter_retries_when_source_snippet_is_not_grounded() -> None:
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1")
    raw = SubjectRawOutput(
        raw_output_id="demo:structure_llm:raw",
        case_id="demo",
        subject_id="structure_llm",
        subject_kind="structure_llm",
        model="test-model",
        prompt_version="test",
        raw_text="TICT is plausible from the donor-acceptor scaffold.",
    )
    client = _FakeBaselineCommonRetryClient()

    prediction = BaselineCommonNarrativeAdapter(client=client).extract(case, raw)  # type: ignore[arg-type]

    assert client.calls == 2
    assert client.feedback_seen is not None
    assert "source_snippet must be a contiguous copied substring" in client.feedback_seen
    assert len(prediction.evidence_units) == 1
    audit = prediction.metadata["grounding_audit"]
    assert audit["grounding"][0]["source_snippet_found"] is True


def test_prediction_from_case_run_uses_mas_subject_kind() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        evidence_ledger=EvidenceLedger(case_id="demo"),
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    prediction = prediction_from_case_run(case_run, model="test-model")

    assert prediction.subject_kind == "mechcal"
    assert prediction.raw_output.raw_json is not None


def test_mas_eval_adapter_has_no_benchmark_aware_rule_templates() -> None:
    source = Path("mechcal/eval/mas.py").read_text(encoding="utf-8").lower()

    forbidden_fragments = [
        "mas_confidence_gated_synthesis_v1",
        "deterministic_confidence_gated_synthesis",
        "sulfur-rich",
        "clusteroluminescence",
        "esipt-only explanation",
        "crystal-state s/s",
        "_coverage_projection_prediction",
        "_synthetic_boundary_evidence",
    ]
    assert [item for item in forbidden_fragments if item in source] == []


def test_mas_narrative_adapter_extracts_conclusion_level_units() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        evidence_ledger=EvidenceLedger(case_id="demo"),
        artifact_manifest=ArtifactManifest(case_id="demo"),
        final_answer=FinalAnswer(
            case_id="demo",
            current_hypothesis="torsional decay",
            confidence=0.5,
            answer=(
                "The torsion scan shows oscillator-strength collapse at twisted "
                "geometries, supporting torsion-sensitive quenching."
            ),
        ),
    )

    prediction = narrative_prediction_from_case_run(
        case_run,
        model="test-model",
        extractor=MASNarrativeToSchemaExtractor(client=_FakeMASNarrativeClient()),  # type: ignore[arg-type]
    )

    assert prediction.extractor_id == "mas_narrative_to_schema_v1"
    assert prediction.evidence_units[0].basis == "computed"
    assert prediction.diagnosis_units[0].status == "computed_supported"
    assert prediction.diagnosis_units[0].evidence_refs == [prediction.evidence_units[0].evidence_id]


def test_mas_strict_common_adapter_uses_conclusion_ledger_text() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                EvidenceUnit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    claim="Accepted rotor evidence supports RIM/RIR.",
                    context="macro proxy",
                    basis="proxy",
                    support="supports",
                    family="torsion_sensitivity",
                    summary="Accepted rotor evidence supports RIM/RIR.",
                    relation="supports",
                    status="present",
                )
            ],
        ),
        artifact_manifest=ArtifactManifest(case_id="demo"),
        mechanism_predictions=[
            MechanismPrediction(
                label="RIM_RIR_RIV",
                rank=1,
                confidence=0.62,
                plausibility_confidence=0.62,
                claim_status="candidate_requires_validation",
                support_strength="weak_or_proxy",
                evidence_refs=["E1"],
                evidence_details=[
                    MechanismPredictionEvidence(
                        finding="Accepted rotor evidence supports RIM/RIR.",
                        warrant=(
                            "Rotor evidence weakly supports motion restriction as a "
                            "candidate mechanism."
                        ),
                        boundary="Needs aggregation PL.",
                        evidence_refs=["E1"],
                    )
                ],
                validation_needed=["Needs aggregation PL."],
                limitations=["No wet-lab claim."],
            )
        ],
        conclusion_ledger=ConclusionLedger(
            ledger_id="demo:R001:conclusion_ledger",
            case_id="demo",
            round_id="R001",
            evidence_conclusions=[
                EvidenceConclusion(
                    conclusion_id="EC001",
                    statement="RIM/RIR is proxy-supported by accepted rotor evidence.",
                    context="MAS conclusion ledger",
                    mechanism_family="RIM/RIR",
                    direction="supports",
                    basis="proxy",
                    source_evidence_refs=["E1"],
                    limits=["No wet-lab PL claim."],
                )
            ],
            diagnosis_conclusions=[
                DiagnosisConclusion(
                    conclusion_id="DC001",
                    mechanism="RIM/RIR proxy mechanism",
                    status="proxy_supported",
                    statement="RIM/RIR proxy mechanism is supported by accepted evidence.",
                    reasoning_summary="The ledger cites accepted rotor evidence.",
                    source_evidence_refs=["E1"],
                    missing_or_unresolved=["Needs aggregation PL."],
                    scope_limits=["No wet-lab claim."],
                )
            ],
        ),
    )
    transcript = build_mas_strict_source_text(case_run)
    case = EvalCase(case_id="demo", smiles="C1=CC=CC=C1", hidden_reference={"secret": "x"})

    prediction = strict_common_prediction_from_case_run(
        case_run,
        case=case,
        model="test-model",
        adapter=BaselineCommonNarrativeAdapter(  # type: ignore[arg-type]
            client=_FakeStrictMASCommonAdapterClient()
        ),
    )

    assert "CONCLUSION_LEDGER" in transcript
    assert "claim_status=candidate_requires_validation" in transcript
    assert "support_strength=weak_or_proxy" in transcript
    assert "validation_needed: Needs aggregation PL." in transcript
    assert "finding: Accepted rotor evidence supports RIM/RIR." in transcript
    assert "warrant: Rotor evidence weakly supports motion restriction" in transcript
    assert "boundary: Needs aggregation PL." in transcript
    assert prediction.extractor_id == "baseline_common_narrative_to_schema_v1"
    assert prediction.evidence_units[0].claim == "Accepted MAS evidence supports a RIM/RIR proxy."
    assert prediction.diagnosis_units[0].mechanism == "RIM/RIR proxy mechanism"


def test_case_correction_adds_mas_fair_targets_without_changing_reference() -> None:
    document = {
        "case_id": "case",
        "public_input": {
            "molecule": {"structure": {"format": "smiles", "value": "C1=CC=CC=C1"}},
            "task": "Assess.",
        },
        "hidden_reference": {
            "reference_evidence_units": [
                {
                    "evidence_id": "E1",
                    "claim": "Viscosity increases red emission.",
                    "mechanistic_interpretation": "Viscosity supports RIM.",
                    "mechanism_links": ["RIM", "AIE"],
                    "evidence_accessibility": (
                        "Not accessible from SMILES alone; requires viscosity-dependent "
                        "fluorescence experiments."
                    ),
                    "paper_quote": "source quote",
                },
                {
                    "evidence_id": "E2",
                    "claim": "Donor-acceptor structure supports TICT plausibility.",
                    "mechanistic_interpretation": "D-A topology is relevant to CT.",
                    "mechanism_links": ["TICT"],
                    "evidence_accessibility": (
                        "Partly inferable from SMILES but requires solvent spectra."
                    ),
                    "paper_quote": "source quote 2",
                },
            ],
            "reference_diagnosis_units": [
                {
                    "diagnosis_id": "D1",
                    "mechanism": "RIM-supported AIE",
                    "reference_status": "supported",
                    "diagnosis_role": "primary_supported_mechanism",
                    "expert_conclusion": "RIM is supported.",
                    "supporting_evidence_ids": ["E1"],
                },
                {
                    "diagnosis_id": "D2",
                    "mechanism": "TICT-only explanation",
                    "reference_status": "weakened_or_rejected",
                    "diagnosis_role": "weakened_or_rejected_mechanism",
                    "expert_conclusion": "TICT-only is insufficient.",
                    "supporting_evidence_ids": ["E2"],
                },
            ],
        },
    }

    original_reference = document["hidden_reference"]["reference_evidence_units"][0].copy()
    corrected = correct_case_document(document)
    hidden = corrected["hidden_reference"]

    assert corrected["benchmark_correction"]["semantic_info_changed"] is False
    assert hidden["semantic_target_policy"]["policy_version"] == CASE_CORRECTION_POLICY_VERSION
    assert hidden["reference_evidence_units"][0] == original_reference
    assert corrected["public_input"]["normalized_structure"]["smiles"] == "C1=CC=CC=C1"
    assert len(hidden["semantic_evidence_targets"]) == 2
    assert len(hidden["semantic_diagnosis_targets"]) == 2
    assert set(hidden["semantic_evidence_targets"][0]) == {
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
    assert hidden["semantic_evidence_targets"][0]["source_ids"] == ["E1"]
    assert hidden["semantic_evidence_targets"][0]["access_class"] == "wet_lab_required"
    assert "viscosity_pl" in hidden["semantic_evidence_targets"][0][
        "required_capabilities"
    ]
    assert hidden["semantic_evidence_targets"][0]["scoring_role"] == "boundary_score"
    assert hidden["semantic_diagnosis_targets"][0]["diagnosis_direction"] == "supported"
    assert (
        hidden["semantic_diagnosis_targets"][1]["diagnosis_direction"]
        == "weakened_or_rejected"
    )
    assert "must_not_claim" in hidden["semantic_target_policy"]
    assert corrected["benchmark_correction"]["validation"]["status"] == "passed"


def test_case_correction_uses_negative_access_before_smiles_terms() -> None:
    document = {
        "case_id": "case",
        "public_input": {
            "molecule": {"structure": {"format": "smiles", "value": "C1=CC=CC=C1"}},
            "task": "Assess.",
        },
        "hidden_reference": {
            "reference_evidence_units": [
                {
                    "evidence_id": "E1",
                    "claim": (
                        "LD-mitochondrion contact increases during ferroptosis and "
                        "is reversed by ferrostatin-1."
                    ),
                    "mechanistic_interpretation": (
                        "This is a live-cell perturbation imaging observation."
                    ),
                    "evidence_accessibility": (
                        "Not accessible from SMILES alone; requires ferroptosis "
                        "perturbation and imaging quantification."
                    ),
                }
            ],
            "reference_diagnosis_units": [],
        },
    }

    corrected = correct_case_document(document)
    target = corrected["hidden_reference"]["semantic_evidence_targets"][0]

    assert target["access_class"] == "application_context_required"
    assert "bioimaging_perturbation" in target["required_capabilities"]
    assert "smiles_structure" not in target["required_capabilities"]


def test_case_correction_distinguishes_bio_dynamics_from_nonadiabatic_dynamics() -> None:
    document = {
        "case_id": "case",
        "public_input": {
            "molecule": {"structure": {"format": "smiles", "value": "C1=CC=CC=C1"}},
            "task": "Assess.",
        },
        "hidden_reference": {
            "reference_evidence_units": [
                {
                    "evidence_id": "E1",
                    "claim": "Organelle dynamics show LD-mito fission/fusion contacts.",
                    "mechanistic_interpretation": "Requires live-cell imaging.",
                    "evidence_accessibility": "Requires microscopy.",
                },
                {
                    "evidence_id": "E2",
                    "claim": "Nonadiabatic dynamics support a CI-mediated decay route.",
                    "mechanistic_interpretation": "Requires surface hopping dynamics.",
                    "evidence_accessibility": "Requires high-level computation.",
                },
            ],
            "reference_diagnosis_units": [],
        },
    }

    corrected = correct_case_document(document)
    targets = corrected["hidden_reference"]["semantic_evidence_targets"]

    assert targets[0]["access_class"] == "application_context_required"
    assert "bioimaging_perturbation" in targets[0]["required_capabilities"]
    assert "nonadiabatic_dynamics" not in targets[0]["required_capabilities"]
    assert targets[1]["access_class"] == "high_level_compute_required"
    assert "nonadiabatic_dynamics" in targets[1]["required_capabilities"]


def test_case_correction_batch_writes_audit_report(tmp_path) -> None:  # noqa: ANN001
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    document = {
        "case_id": "case",
        "public_input": {
            "molecule": {"structure": {"format": "smiles", "value": "C1=CC=CC=C1"}},
            "task": "Assess.",
        },
        "hidden_reference": {
            "reference_evidence_units": [
                {
                    "evidence_id": "E1",
                    "claim": "PXRD shows subtle pressure-dependent peak shifts.",
                    "mechanistic_interpretation": "PXRD constrains the pressure mechanism.",
                    "evidence_accessibility": "requires in-situ PXRD.",
                }
            ],
            "reference_diagnosis_units": [],
        },
    }
    (input_dir / "case.json").write_text(json.dumps(document), encoding="utf-8")

    result = correct_case_path(input_dir, output_dir)

    assert result.audit_path.exists()
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    item = audit["cases"][0]["items"][0]
    assert item["assigned_access_class"] == "wet_lab_required"
    assert item["assigned_capabilities"] == ["pxrd_or_crystal_characterization"]
    assert audit["cases"][0]["validation"]["status"] == "passed"
