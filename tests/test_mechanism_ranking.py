from __future__ import annotations

import json
import math
import re
from types import SimpleNamespace

import pytest

from mechcal.eval.common_prior import (
    common_prior_predictions,
    common_prior_support_scores,
)
from mechcal.eval.mechanism_baselines import (
    StructureLLMMechanismBaseline,
    _mechanism_prediction_tagged_regex,
    _normalize_native_thinking_prediction_json,
    build_structure_llm_mechanism_payload,
    prediction_payload_from_structure_llm_mechanism_json,
)
from mechcal.eval.mechanism_ranking import (
    MECHANISM_POOL,
    LLMMechanismEvidenceSupportJudge,
    _support_judge_settings,
    judge_mechanism_prediction_support,
    load_mechanism_evidence_rubric,
    score_mechanism_ranking,
)
from mechcal.runtime.llm import JsonCompletionResult, OpenAICompatibleSettings
from mechcal.schemas import (
    ArtifactManifest,
    CaseInput,
    CaseRun,
    EvidenceLedger,
    EvidenceUnit,
    HypothesisEntry,
    HypothesisPortfolio,
    MechanismPrediction,
    MechanismSupportArgument,
)
from mechcal.cli.evaluate import (
    _assign_public_case_ids,
    _compact_score_text,
    _prediction_payload,
    _unscored_metrics,
)
from mechcal.cli.evaluate_direct import (
    _has_mechanism_predictions,
    _load_cached_eval,
    _make_subject,
    _run_or_load_raw,
)
from mechcal.cli.run_shards import (
    _load_public_case_id_map,
    _public_case_id_map,
)


class _FakeSupportJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        encoded_payload = json.dumps(payload, ensure_ascii=False)
        assert "reference_mechanisms" not in encoded_payload
        assert "hidden_reference" not in encoded_payload
        assert payload["predicted_mechanism"]["label"] == "RIM_RIR_RIV"
        assert "mechanism_rubric" in payload
        return {
            "evidence_support_score": 0.5,
            "rationale": "The supplied rotor evidence is relevant but incomplete.",
        }


class _FakeEvidenceDetailJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, schema_feedback
        evidence = payload["predicted_mechanism"]["evidence"]
        assert evidence == [
            (
                "Finding: Rotor scan shows a torsion-sensitive brightness proxy. "
                "Warrant: This weakly supports RIM because restricted torsion can "
                "reduce nonradiative motion. Boundary: Aggregate restriction is "
                "not directly measured."
            )
        ]
        assert "legacy duplicate text" not in json.dumps(evidence, ensure_ascii=False)
        return {
            "evidence_support_score": 0.5,
            "rationale": "Structured evidence detail was passed to the judge.",
        }


class _FakeStructureJsonClient:
    settings = SimpleNamespace(model="chem-r-8b")

    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def complete_json(self, **kwargs):  # noqa: ANN003, ANN201
        self.request = kwargs
        return {
            "final_summary": "Structure-only assessment.",
            "mechanism_predictions": [],
        }


class _FakeTaggedStructureJsonClient:
    settings = SimpleNamespace(model="chem-r-8b")

    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def complete_json_result(self, **kwargs):  # noqa: ANN003, ANN201
        self.request = kwargs
        return JsonCompletionResult(
            data={
                "mechanism_predictions": [
                    {
                        "label": label,
                        "rank": rank,
                        "confidence": 0.5,
                        "claim_status": "candidate_requires_validation",
                        "support_strength": "weak_or_proxy",
                        "evidence": [
                            "Finding: motif. Warrant: proxy. Boundary: SMILES only."
                        ],
                        "limitations": ["Needs validation."],
                    }
                    for rank, label in enumerate(MECHANISM_POOL[:3], start=1)
                ]
            },
            raw_text="<think>reasoning</think><answer>{}</answer>",
            finish_reason="stop",
            usage={"completion_tokens": 100},
        )


class _FakeNativeThinkingJsonClient:
    settings = SimpleNamespace(model="intern-s1-mini")

    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def complete_json_result(self, **kwargs):  # noqa: ANN003, ANN201
        self.request = kwargs
        return JsonCompletionResult(
            data={
                "mechanism_predictions": [
                    {
                        "label": label,
                        "rank": rank,
                        "confidence": 0.5,
                        "claim_status": "candidate_requires_validation",
                        "support_strength": "weak_or_proxy",
                        "evidence": [
                            "Finding: motif. Warrant: proxy. Boundary: SMILES only."
                        ],
                        "limitations": ["Needs validation."],
                    }
                    for rank, label in enumerate(MECHANISM_POOL[:3], start=1)
                ]
            },
            raw_text="reasoning</think>{}",
            reasoning_text="reasoning",
            finish_reason="stop",
            usage={"completion_tokens": 100},
        )


class _FakeEmptyThenValidSupportJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def __init__(self) -> None:
        self.calls = 0
        self.feedback_seen: list[str | None] = []

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, payload
        self.calls += 1
        self.feedback_seen.append(schema_feedback)
        if self.calls == 1:
            return {}
        return {
            "evidence_support_score": 0.5,
            "rationale": "Valid JSON after schema feedback.",
        }


class _FakeAlternateScoreKeyJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, payload, schema_feedback
        return {
            "result": {
                "support_score": "0.5",
                "rationale": "Alternate but unambiguous score key.",
            }
        }


class _FakeNonNumericThenValidSupportJudgeClient:
    settings = SimpleNamespace(model="judge-model")

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, *, system_prompt, payload, schema_feedback=None):  # noqa: ANN001
        del system_prompt, payload, schema_feedback
        self.calls += 1
        if self.calls == 1:
            return {
                "evidence_support_score": "unsupported",
                "rationale": "Invalid non-numeric score.",
            }
        return {
            "evidence_support_score": 0.5,
            "rationale": "Valid score after retry.",
        }


def test_mechanism_evidence_rubric_matches_pool() -> None:
    rubric = load_mechanism_evidence_rubric()

    assert rubric["mechanism_evidence_rubric_version"] == "v2"
    assert tuple(rubric["mechanism_pool"]) == MECHANISM_POOL
    assert set(rubric["labels"]) == set(MECHANISM_POOL)


def test_sharded_mas_runner_preserves_global_public_case_ids(tmp_path) -> None:
    global_case_ids = ["AIE_DDX_001", "AIE_DDX_002", "AIE_DDX_003"]
    public_case_id_map = _public_case_id_map(global_case_ids)
    map_path = tmp_path / "public_case_id_map.json"
    map_path.write_text(
        json.dumps({"public_case_id_map": public_case_id_map}, indent=2) + "\n",
        encoding="utf-8",
    )
    shard_cases = [{"case_id": "AIE_DDX_003"}]

    _assign_public_case_ids(shard_cases, map_path=map_path)

    assert public_case_id_map["AIE_DDX_003"] == "CASE_003"
    assert shard_cases == [{"case_id": "AIE_DDX_003", "public_case_id": "CASE_003"}]


def test_sharded_subset_can_reuse_larger_public_case_id_map(tmp_path) -> None:
    map_path = tmp_path / "public_case_id_map.json"
    map_path.write_text(
        json.dumps(
            {
                "public_case_id_map": {
                    "AIE_DDX_001": "CASE_001",
                    "AIE_DDX_002": "CASE_002",
                    "AIE_DDX_003": "CASE_003",
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_public_case_id_map(map_path)

    assert loaded["AIE_DDX_003"] == "CASE_003"


def test_structure_llm_mechanism_payload_is_public_only() -> None:
    payload = build_structure_llm_mechanism_payload(
        {
            "case_id": "AIE_DDX_SECRET_001",
            "public_case_id": "CASE_001",
            "smiles": "C1=CC=CC=C1",
            "task": "Predict mechanisms.",
            "reference_mechanisms": [{"label": "RACI_CI_ACCESS"}],
            "hidden_reference": {"secret": "SECRET_REFERENCE_TOKEN"},
        }
    )
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["case_id"] == "CASE_001"
    assert payload["mechanism_pool"] == list(MECHANISM_POOL)
    assert "AIE_DDX_SECRET_001" not in encoded
    assert "SECRET_REFERENCE_TOKEN" not in encoded
    assert "reference_mechanisms" not in encoded
    assert "hidden_reference" not in encoded


def test_structure_llm_mechanism_prediction_payload_normalizes_response() -> None:
    predictions = prediction_payload_from_structure_llm_mechanism_json(
        {
            "mechanism_predictions": [
                {
                    "label": "RIM_RIR_RIV",
                    "rank": 2,
                    "confidence": 1.5,
                    "evidence": ["Rotor proxy."],
                    "limitations": "Needs aggregation PL.",
                }
            ]
        }
    )

    assert predictions == [
        {
            "prediction_id": "P001",
            "rank": 2,
            "label": "RIM_RIR_RIV",
            "confidence": 1.0,
            "claim_status": "likely",
            "support_strength": "",
            "evidence_refs": [],
            "evidence": ["Rotor proxy."],
            "limitations": ["Needs aggregation PL."],
        }
    ]


def test_structure_llm_runner_accepts_explicit_subject_identity() -> None:
    settings = OpenAICompatibleSettings(
        base_url="http://127.0.0.1:8011/v1",
        model="chem-r-8b",
        api_key="local-test-key",
    )

    subject = _make_subject(
        settings=settings,
        evidence_format=True,
        subject_id="chem_r_8b_direct",
        subject_kind="structure_llm",
    )

    assert subject.subject_id == "chem_r_8b_direct"
    assert subject.subject_kind == "structure_llm"
    assert subject.model == "chem-r-8b"


def test_structure_llm_can_require_exactly_three_schema_valid_predictions() -> None:
    client = _FakeStructureJsonClient()
    subject = StructureLLMMechanismBaseline(
        client=client,  # type: ignore[arg-type]
        strict_json_schema=True,
    )

    subject.run(
        {
            "public_case_id": "CASE_001",
            "smiles": "CCO",
            "task": "Rank likely mechanisms.",
        }
    )

    assert client.request is not None
    payload = client.request["payload"]
    assert isinstance(payload, dict)
    assert payload["output_constraints"] == {
        "mechanism_prediction_count": 3,
        "evidence_items_per_prediction": 1,
        "limitation_items_per_prediction": 1,
        "concise_output": True,
        "include_final_summary": False,
    }
    assert client.request["omit_response_format"] is True
    extra_body = client.request["extra_body"]
    assert isinstance(extra_body, dict)
    guided_json = extra_body["guided_json"]
    predictions = guided_json["properties"]["mechanism_predictions"]
    assert len(predictions["prefixItems"]) == 3
    assert predictions["items"] is False
    assert [
        item["properties"]["rank"]["enum"] for item in predictions["prefixItems"]
    ] == [[1], [2], [3]]
    assert "final_summary" not in guided_json["properties"]


def test_structure_llm_uses_bounded_grammar_for_retry_output() -> None:
    client = _FakeStructureJsonClient()
    subject = StructureLLMMechanismBaseline(
        client=client,  # type: ignore[arg-type]
        strict_json_schema=True,
    )

    subject.run(
        {
            "public_case_id": "CASE_001",
            "smiles": "CCO",
            "task": "Rank likely mechanisms.",
            "_bounded_output": True,
        }
    )

    assert client.request is not None
    extra_body = client.request["extra_body"]
    guided_grammar = extra_body["guided_grammar"]
    assert guided_grammar.count("prediction1") == 2
    assert guided_grammar.count("prediction2") == 2
    assert guided_grammar.count("prediction3") == 2
    assert all(label in guided_grammar for label in MECHANISM_POOL)


def test_chem_r_native_mode_preserves_tags_and_raw_completion() -> None:
    client = _FakeTaggedStructureJsonClient()
    subject = StructureLLMMechanismBaseline(
        client=client,  # type: ignore[arg-type]
        native_think_answer=True,
    )

    result = subject.run(
        {
            "public_case_id": "CASE_001",
            "smiles": "CCO",
            "task": "Rank likely mechanisms.",
        }
    )

    assert client.request is not None
    assert client.request["omit_response_format"] is True
    assert client.request["answer_tag"] == "answer"
    regex = client.request["extra_body"]["guided_regex"]
    assert regex.startswith("<think>")
    assert "</think>" in regex
    assert "<answer>" in regex
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"completion_tokens": 100}
    assert result["raw_completion"].startswith("<think>")


def test_native_thinking_json_mode_preserves_reasoning_and_final_answer() -> None:
    client = _FakeNativeThinkingJsonClient()
    subject = StructureLLMMechanismBaseline(
        client=client,  # type: ignore[arg-type]
        native_thinking_json=True,
    )

    result = subject.run(
        {
            "public_case_id": "CASE_001",
            "smiles": "CCO",
            "task": "Rank likely mechanisms.",
        }
    )

    assert client.request is not None
    assert client.request["omit_response_format"] is True
    assert client.request["parse_after_tag"] == "think"
    assert client.request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": 50,
    }
    constraints = client.request["payload"]["output_constraints"]
    assert constraints["evidence_field_example"] == [
        "Finding: ... Warrant: ... Boundary: ..."
    ]
    assert constraints["limitations_field_example"] == ["One concise limitation."]
    assert result["raw_reasoning"] == "reasoning"
    assert result["finish_reason"] == "stop"
    assert result["format_adapter"] == "native_thinking_json_v1"


def test_native_thinking_json_adapter_only_normalizes_equivalent_fields() -> None:
    raw = {
        "mechanism_predictions": [
            {
                "label": "RIM_RIR_RIV",
                "rank": 1,
                "evidence": {
                    "Finding": "A rigid aromatic core is present.",
                    "Warrant": "Rigidity can restrict intramolecular motion.",
                    "Boundary": "Aggregate restriction is not observed.",
                },
                "limitation": "SMILES does not provide an aggregation state.",
            }
        ]
    }

    normalized = _normalize_native_thinking_prediction_json(raw)

    prediction = normalized["mechanism_predictions"][0]
    assert prediction["evidence"] == [
        "Finding: A rigid aromatic core is present. "
        "Warrant: Rigidity can restrict intramolecular motion. "
        "Boundary: Aggregate restriction is not observed."
    ]
    assert prediction["limitations"] == [
        "SMILES does not provide an aggregation state."
    ]
    assert raw["mechanism_predictions"][0]["evidence"] == {
        "Finding": "A rigid aromatic core is present.",
        "Warrant": "Rigidity can restrict intramolecular motion.",
        "Boundary": "Aggregate restriction is not observed.",
    }


def test_chem_r_tagged_regex_requires_three_ranked_predictions() -> None:
    regex = _mechanism_prediction_tagged_regex()

    assert all(label in regex for label in MECHANISM_POOL)
    assert "Finding: " in regex
    assert " Warrant: " in regex
    assert " Boundary: " in regex
    assert re.fullmatch(regex, _tagged_chem_r_response(MECHANISM_POOL[:3]))
    assert not re.fullmatch(
        regex,
        _tagged_chem_r_response([MECHANISM_POOL[0]] * 3),
    )


def _tagged_chem_r_response(labels: list[str]) -> str:
    rows = [
        {
            "label": label,
            "rank": rank,
            "confidence": 0.5,
            "claim_status": "candidate_requires_validation",
            "support_strength": "weak_or_proxy",
            "evidence": ["Finding: motif Warrant: proxy Boundary: SMILES only"],
            "limitations": ["Needs validation"],
        }
        for rank, label in enumerate(labels, start=1)
    ]
    answer = json.dumps(
        {"mechanism_predictions": rows},
        separators=(",", ":"),
    )
    return f"<think>reasoning</think><answer>{answer}</answer>"


def test_strict_raw_validation_rejects_duplicate_labels() -> None:
    duplicate = {
        "raw_json": {
            "mechanism_predictions": [
                {
                    "label": "RIM_RIR_RIV",
                    "rank": rank,
                    "evidence": ["Finding: x. Warrant: y. Boundary: z."],
                    "limitations": ["Needs validation."],
                }
                for rank in (1, 2, 3)
            ]
        }
    }

    assert not _has_mechanism_predictions(duplicate, min_count=3)


def test_strict_raw_validation_rejects_length_finish() -> None:
    raw_record = {
        "finish_reason": "length",
        "raw_json": {
            "mechanism_predictions": [
                {"label": "RIM_RIR_RIV", "rank": 1},
                {"label": "ICT_TICT_CT", "rank": 2},
                {"label": "ESIPT_PT", "rank": 3},
            ]
        },
    }

    assert not _has_mechanism_predictions(raw_record, min_count=3)


def test_structure_llm_require_cached_raw_never_calls_subject(tmp_path) -> None:
    subject = SimpleNamespace(subject_id="chem_r_8b_direct")

    with pytest.raises(RuntimeError, match="Required valid raw cache is missing"):
        _run_or_load_raw(
            subject,  # type: ignore[arg-type]
            {"case_id": "AIE_DDX_MECH_V4_0001"},
            raw_path=tmp_path / "missing.json",
            refresh=False,
            required_predictions=3,
            require_cached=True,
        )


def test_structure_llm_eval_resume_requires_matching_configuration(tmp_path) -> None:
    subject = SimpleNamespace(
        subject_id="chem_r_8b_direct",
        model="chem-r-8b",
        prompt_version="2026-07-25",
    )
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "case_id": "AIE_DDX_MECH_V4_0001",
                "subject_id": subject.subject_id,
                "model": subject.model,
                "prompt_version": subject.prompt_version,
                "mechanism_predictions": [{}, {}, {}],
                "support_judgements": [{"judge_model": "deepseek-v4-flash"}],
                "scores": {"es_ndcg_at_3": 0.5},
            }
        ),
        encoding="utf-8",
    )

    assert (
        _load_cached_eval(
            path,
            refresh=False,
            case_id="AIE_DDX_MECH_V4_0001",
            subject=subject,  # type: ignore[arg-type]
            judge_model="deepseek-v4-flash",
        )
        is not None
    )
    assert (
        _load_cached_eval(
            path,
            refresh=False,
            case_id="AIE_DDX_MECH_V4_0001",
            subject=subject,  # type: ignore[arg-type]
            judge_model="another-judge",
        )
        is None
    )


def test_support_judge_can_use_a_separate_provider(monkeypatch) -> None:
    subject_settings = OpenAICompatibleSettings(
        base_url="http://127.0.0.1:8011/v1",
        model="chem-r-8b",
        api_key="local-test-key",
    )
    monkeypatch.setenv(
        "MECHCAL_SUPPORT_JUDGE_BASE_URL",
        "https://judge.example/v1",
    )
    monkeypatch.setenv("MECHCAL_SUPPORT_JUDGE_API_KEY", "judge-test-key")
    monkeypatch.setenv("MECHCAL_SUPPORT_JUDGE_MODEL", "deepseek-v4-flash")

    judge_settings = _support_judge_settings(subject_settings)

    assert judge_settings is not None
    assert judge_settings.base_url == "https://judge.example/v1"
    assert judge_settings.api_key == "judge-test-key"
    assert judge_settings.model == "deepseek-v4-flash"


def test_structure_llm_mechanism_prediction_preserves_explicit_claim_boundary() -> None:
    predictions = prediction_payload_from_structure_llm_mechanism_json(
        {
            "mechanism_predictions": [
                {
                    "label": "RIM_RIR_RIV",
                    "claim_status": "candidate_requires_validation",
                    "support_strength": "weak_or_proxy",
                    "evidence": [
                        "Aryl torsions are only a SMILES-level proxy; aggregate "
                        "restriction is unverified."
                    ],
                }
            ]
        }
    )

    assert predictions[0]["claim_status"] == "candidate_requires_validation"
    assert predictions[0]["support_strength"] == "weak_or_proxy"


def test_mas_mechanism_runner_payload_preserves_structured_claim_fields() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="RACI_CI_ACCESS",
            hypotheses=[
                HypothesisEntry(
                    name="RACI_CI_ACCESS",
                    confidence=0.62,
                    status="plausible",
                    rationale="Planner kept a CI-access candidate.",
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                EvidenceUnit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_snapshots",
                    claim="Torsion-dependent brightness is visible.",
                    context="runtime evidence",
                    basis="computed",
                    support="supports",
                    summary="Torsion-dependent brightness is visible.",
                    family="torsion_sensitivity",
                    relation="supports",
                    status="present",
                )
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R001:support:001",
                target_label="RACI_CI_ACCESS",
                support_level="weak",
                finding="Torsion-dependent brightness is visible.",
                warrant=(
                    "This weakly supports a validation-needed CI-access candidate "
                    "because torsion may connect to a nonradiative pathway."
                ),
                boundary="Needs explicit S1/S0 conical-intersection search.",
                observation_refs=["E1"],
            )
        ],
        mechanism_predictions=[
            MechanismPrediction(
                label="RACI_CI_ACCESS",
                rank=1,
                confidence=0.62,
                evidence=["legacy text without structured details"],
            )
        ],
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    payload = _prediction_payload(case_run)

    assert payload[0]["label"] == "RACI_CI_ACCESS"
    assert payload[0]["plausibility_confidence"] == 0.62
    assert payload[0]["differential_priority"] == 0.62
    assert payload[0]["claim_status"] == "candidate_requires_validation"
    assert payload[0]["support_strength"] == "weak_or_proxy"
    assert payload[0]["evidence_support"] == "weak_or_proxy"
    assert payload[0]["validation_needed"] == [
        "Needs explicit S1/S0 conical-intersection search."
    ]
    detail = payload[0]["evidence_details"][0]  # type: ignore[index]
    assert detail["record_type"] == "MechanismPredictionEvidence"
    assert detail["finding"] == "Torsion-dependent brightness is visible."
    assert detail["warrant"] == (
        "This weakly supports a validation-needed CI-access candidate "
        "because torsion may connect to a nonradiative pathway."
    )
    assert detail["boundary"] == "Needs explicit S1/S0 conical-intersection search."
    assert detail["evidence_refs"] == ["E1"]


def test_score_mechanism_ranking_with_evidence_support_metrics() -> None:
    predictions = [
        {
            "prediction_id": "P1",
            "label": "ICT_TICT_CT",
            "confidence": 0.9,
            "evidence": ["The scaffold has a donor-acceptor motif."],
        },
        {
            "prediction_id": "P2",
            "label": "RIM_RIR_RIV",
            "confidence": 0.8,
            "evidence": ["AIE often involves RIM."],
        },
        {
            "prediction_id": "P3",
            "label": "ESIPT_PT",
            "confidence": 0.7,
            "evidence": ["Phenolic donor and imine/carbonyl acceptor support ESIPT."],
        },
    ]
    reference = [
        {"label": "ESIPT_PT", "role": "primary", "gain": 2},
        {"label": "RACI_CI_ACCESS", "role": "primary", "gain": 2},
        {"label": "ICT_TICT_CT", "role": "secondary", "gain": 1},
    ]

    scores = score_mechanism_ranking(
        predictions,
        reference,
        support_scores={"P1": 0.5, "P2": 0.0, "P3": 1.0},
    )

    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    dcg = 1 / math.log2(2) + 2 / math.log2(4)
    es_dcg = 0.5 / math.log2(2) + 2 / math.log2(4)
    assert scores["top1_primary"] == 0.0
    assert scores["top1_any"] == 1.0
    assert scores["top1_hit"] == 1.0
    assert scores["recall_at_3"] == 2 / 3
    assert scores["ndcg_at_3"] == dcg / idcg
    assert scores["es_ndcg_at_3"] == es_dcg / idcg
    assert scores["support_at_top1"] == 1.0
    assert scores["unsupported_strong_claim_rate_at_3"] == 1 / 3
    assert math.isclose(scores["claim_safety_at_3"], 2 / 3)
    assert scores["ucr_at_3"] == 1 / 3
    assert scores["credibility_gap_at_3"] == scores["ndcg_at_3"] - scores["es_ndcg_at_3"]


def test_unsupported_rate_ignores_validation_bound_candidates() -> None:
    predictions = [
        {
            "prediction_id": "P1",
            "label": "RACI_CI_ACCESS",
            "claim_status": "candidate_requires_validation",
            "support_strength": "weak_or_proxy",
            "evidence": ["Torsion is a proxy; CI validation remains missing."],
        },
        {
            "prediction_id": "P2",
            "label": "ICT_TICT_CT",
            "claim_status": "supported_claim",
            "evidence": ["D-A motif proves CT."],
        },
    ]
    reference = [{"label": "RACI_CI_ACCESS", "role": "primary", "gain": 2}]

    scores = score_mechanism_ranking(
        predictions,
        reference,
        support_scores={"P1": 0.0, "P2": 0.0},
        k=2,
    )

    assert scores["top1_primary"] == 1.0
    assert scores["top1_any"] == 1.0
    assert scores["unsupported_strong_claim_rate_at_2"] == 0.5
    assert scores["claim_safety_at_2"] == 0.5
    assert scores["ucr_at_2"] == 0.5


def test_support_judge_scores_without_reference_payload() -> None:
    judge = LLMMechanismEvidenceSupportJudge(client=_FakeSupportJudgeClient())  # type: ignore[arg-type]

    result = judge_mechanism_prediction_support(
        [
            {
                "prediction_id": "P1",
                "label": "RIM_RIR_RIV",
                "confidence": 0.7,
                "evidence": ["The molecule has rotatable aryl bonds."],
            }
        ],
        judge=judge,
    )

    assert result["support_scores"] == {"P1": 0.5}
    assert result["judgements"][0]["judge_model"] == "judge-model"


def test_support_judge_uses_timeout_and_minimum_token_settings() -> None:
    judge = LLMMechanismEvidenceSupportJudge(
        settings=OpenAICompatibleSettings(
            base_url="https://example.test/v1",
            model="subject-model",
            api_key="test-key",
            timeout_seconds=30.0,
            max_retries=0,
            max_tokens=4096,
        )
    )

    assert judge.client.settings.model == "deepseek-v4-flash"
    assert judge.client.settings.timeout_seconds == 90.0
    assert judge.client.settings.max_tokens == 4096

    low_cap_judge = LLMMechanismEvidenceSupportJudge(
        settings=OpenAICompatibleSettings(
            base_url="https://example.test/v1",
            model="judge-model",
            api_key="test-key",
            timeout_seconds=30.0,
            max_retries=0,
            max_tokens=256,
        )
    )
    assert low_cap_judge.client.settings.max_tokens == 900


def test_support_judge_retries_empty_json_response() -> None:
    client = _FakeEmptyThenValidSupportJudgeClient()
    judge = LLMMechanismEvidenceSupportJudge(client=client)  # type: ignore[arg-type]

    result = judge_mechanism_prediction_support(
        [
            {
                "prediction_id": "P1",
                "label": "ESIPT_PT",
                "confidence": 0.5,
                "evidence": ["O-H...N donor acceptor proxy with explicit boundary."],
            }
        ],
        judge=judge,
    )

    assert result["support_scores"] == {"P1": 0.5}
    assert client.calls == 2
    assert client.feedback_seen[1] is not None


def test_support_judge_accepts_common_score_key_aliases() -> None:
    judge = LLMMechanismEvidenceSupportJudge(
        client=_FakeAlternateScoreKeyJudgeClient(),  # type: ignore[arg-type]
    )

    result = judge_mechanism_prediction_support(
        [
            {
                "prediction_id": "P1",
                "label": "ICT_TICT_CT",
                "confidence": 0.5,
                "evidence": ["D-A proxy with explicit boundary."],
            }
        ],
        judge=judge,
    )

    assert result["support_scores"] == {"P1": 0.5}


def test_support_judge_retries_non_numeric_score_instead_of_defaulting_to_zero() -> None:
    client = _FakeNonNumericThenValidSupportJudgeClient()
    judge = LLMMechanismEvidenceSupportJudge(client=client)  # type: ignore[arg-type]

    result = judge_mechanism_prediction_support(
        [
            {
                "prediction_id": "P1",
                "label": "ICT_TICT_CT",
                "confidence": 0.5,
                "evidence": ["D-A proxy with explicit boundary."],
            }
        ],
        judge=judge,
    )

    assert result["support_scores"] == {"P1": 0.5}
    assert result["judgements"][0]["raw_judge_response"] == {
        "evidence_support_score": 0.5,
        "rationale": "Valid score after retry.",
    }
    assert client.calls == 2


def test_support_judge_prefers_structured_evidence_details() -> None:
    judge = LLMMechanismEvidenceSupportJudge(
        client=_FakeEvidenceDetailJudgeClient(),  # type: ignore[arg-type]
    )

    result = judge_mechanism_prediction_support(
        [
            {
                "prediction_id": "P1",
                "label": "RIM_RIR_RIV",
                "evidence_details": [
                    {
                        "finding": "Rotor scan shows a torsion-sensitive brightness proxy.",
                        "warrant": (
                            "This weakly supports RIM because restricted torsion can "
                            "reduce nonradiative motion."
                        ),
                        "boundary": "Aggregate restriction is not directly measured.",
                    }
                ],
                "evidence": ["legacy duplicate text"],
            }
        ],
        judge=judge,
    )

    assert result["support_scores"] == {"P1": 0.5}


def test_common_prior_is_validation_bound_comparator() -> None:
    predictions = common_prior_predictions()
    scores = score_mechanism_ranking(
        predictions,
        [{"label": "RIM_RIR_RIV", "role": "primary", "gain": 2}],
        support_scores=common_prior_support_scores(predictions),
    )

    assert [item["label"] for item in predictions] == [
        "RIM_RIR_RIV",
        "PACKING_HOST_MATRIX_CONFINEMENT",
        "ICT_TICT_CT",
    ]
    assert all(item["claim_status"] == "candidate_requires_validation" for item in predictions)
    assert scores["top1_primary"] == 1.0
    assert scores["es_ndcg_at_3"] == 0.0
    assert scores["unsupported_strong_claim_rate_at_3"] == 0.0
    assert scores["claim_safety_at_3"] == 1.0


def test_unscored_mas_row_metrics_print_without_crashing() -> None:
    scores = _unscored_metrics()

    assert scores["top1_primary"] is None
    assert scores["claim_safety_at_3"] is None
    assert _compact_score_text(scores) == "unscored"
