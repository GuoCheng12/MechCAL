from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import mechcal.agents.planner as planner_module
from mechcal.agents.conclusion_ledger import ConclusionLedgerAgent
from mechcal.agents.mechanism_agenda import MechanismAgendaAgent
from mechcal.agents.mechanism_critic import (
    MechanismCriticAgent,
    _critic_payload,
    _sanitize_portfolio_response,
)
from mechcal.agents.photophysics_arbiter import (
    GenericPhotophysicsWebSearch,
    PhotophysicsArbiterAgent,
    validate_generic_query,
)
from mechcal.agents.planner import PlannerAgent
from mechcal.agents.report_writer import WorkerReportWriter
from mechcal.benchmark import (
    build_diagnosis_alignment_payload,
    build_evidence_alignment_payload,
    run_benchmark,
)
from mechcal.capabilities import CapabilityRegistry, default_capability_registry
from mechcal.capabilities.evidence_guide import (
    ability_evidence_guide,
    compact_ability_evidence_guide,
)
from mechcal.compat import load_case_run, write_legacy_report
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.runtime import (
    MechCALOrchestrator,
    BaseAgentRuntime,
    OrchestratorConfig,
    OrchestratorDeps,
    OutputAssembler,
    RunStore,
    default_deps,
    mcp_deps,
)
from mechcal.runtime.context import build_planner_context
from mechcal.runtime.mechanism_program import build_diagnosis_units
from mechcal.runtime.orchestrator import (
    _apply_planner_priorities_to_differential_portfolio,
    _has_low_margin_portfolio,
    _rankings_stable_under_low_margin,
)
from mechcal.schemas import (
    AgendaCoverageItem,
    AgentExecutionPlan,
    AgentReport,
    ArtifactManifest,
    ArtifactRecord,
    CandidateMechanism,
    CaseInput,
    CaseRun,
    ComputationalRoute,
    ConclusionLedger,
    DecisionGateResult,
    DiagnosisConclusion,
    DiagnosisUnit,
    DifferentialMechanismPortfolio,
    DifferentialMechanismPortfolioRow,
    DispatchRequest,
    EvidenceConclusion,
    EvidenceLedger,
    EvidenceMechanismAttribution,
    EvidenceMechanismAttributionUpdate,
    EvidenceQuestion,
    EvidenceUnit,
    FinalAnswer,
    HypothesisEntry,
    HypothesisPortfolio,
    MechanismAgendaCoverage,
    MechanismProgram,
    MechanismSupportArgument,
    MicroscopicArtifactBundle,
    PhotophysicsReview,
    PlannerDecision,
    RecommendedAgendaRoute,
    ScopeBoundary,
    ToolExecutionResult,
)
from mechcal.schemas.failures import FailureReport
from mechcal.tools import LocalMicroscopicTool, StdioMcpTransport
from mechcal.tools.amesp_microscopic import (
    MICROSCOPIC_ROUTE_PROFILES,
    AmespMicroscopicError,
    MicroscopicRouteResult,
)


def _evidence_unit(
    *,
    evidence_id: str,
    round_id: str,
    source_report_id: str,
    agent_name: str,
    family: str,
    summary: str,
    capability_id: str | None = None,
    claim: str | None = None,
    context: str | None = None,
    basis: str = "computed",
    support: str = "supports",
    observable: str | None = None,
    status: str = "present",
    observable_tags: list[str] | None = None,
    metrics: dict[str, object] | None = None,
) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_id=evidence_id,
        round_id=round_id,
        source_report_id=source_report_id,
        agent_name=agent_name,
        capability_id=capability_id or f"{agent_name}.test",
        claim=claim or summary,
        context=context or "test evidence context",
        basis=basis,  # type: ignore[arg-type]
        support=support,  # type: ignore[arg-type]
        observable=observable or (observable_tags or [family])[0],
        family=family,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        observable_tags=observable_tags or [],
        summary=summary,
        metrics=metrics or {},
    )


def _fake_llm_deps(
    config: OrchestratorConfig,
    base_deps: OrchestratorDeps | None = None,
) -> OrchestratorDeps:
    base_deps = base_deps or MechCALOrchestrator(config).deps
    client = FakeJsonClient()
    return OrchestratorDeps(
        planner=PlannerAgent(
            capability_registry=base_deps.capability_registry,
            llm_client=client,  # type: ignore[arg-type]
            enable_llm_decisions=True,
        ),
        mechanism_agenda=MechanismAgendaAgent(
            capability_registry=base_deps.capability_registry,
            llm_client=client,  # type: ignore[arg-type]
        ),
        photophysics_arbiter=PhotophysicsArbiterAgent(
            llm_client=client,  # type: ignore[arg-type]
            web_search=FakeGenericSearch(),
        ),
        mechanism_critic=MechanismCriticAgent(
            llm_client=client,  # type: ignore[arg-type]
        ),
        conclusion_ledger=ConclusionLedgerAgent(
            llm_client=client,  # type: ignore[arg-type]
        ),
        output_assembler=OutputAssembler(),
        structure=base_deps.structure,
        workers=base_deps.workers,
        gate=base_deps.gate,
        capability_registry=base_deps.capability_registry,
    )


def test_orchestrator_runs_one_molecule_typed_flow(tmp_path) -> None:
    case_id = "benzene_v2"
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=4)
    result = MechCALOrchestrator(
        config,
        deps=_fake_llm_deps(config),
    ).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
    )

    run_dir = tmp_path / case_id
    families = set(result.evidence_ledger.covered_families())
    evidence_ids = {item.evidence_id for item in result.evidence_ledger.items}

    assert result.status == "finalized"
    assert result.runtime["orchestrator"] == "mechcal"
    assert result.runtime["langgraph"] is False
    assert result.runtime["ablation"]["mode"] == "full_mechcal"
    assert result.round_ids == ["R001", "R002", "R003", "R004"]
    assert result.final_answer is not None
    assert result.photophysics_review is not None
    assert result.mechanism_agenda_coverage is not None
    assert result.differential_mechanism_portfolio is not None
    assert result.conclusion_ledger is None
    assert result.mechanism_predictions
    assert result.final_answer.mechanism_predictions == result.mechanism_predictions
    assert result.final_answer.round_refs == result.round_ids
    assert result.claim_ledger is not None
    assert result.claim_ledger.assessments
    assert result.claim_ledger.coverage_debt_hypotheses()
    assert "geometry_precondition" in families
    assert "external_precedent" not in families
    assert result.operational_notes
    assert all(note.note_id not in evidence_ids for note in result.operational_notes)
    assert build_planner_context(
        result,
        round_index=len(result.round_ids),
        max_rounds=4,
    ).operational_notes
    assert (run_dir / "input.json").is_file()
    assert (run_dir / "case_run.json").is_file()
    assert (run_dir / "final.json").is_file()
    assert (run_dir / "photophysics_review.json").is_file()
    assert (run_dir / "mechanism_agenda_coverage.json").is_file()
    assert (run_dir / "differential_mechanism_portfolio.json").is_file()
    assert (run_dir / "mechanism_predictions.json").is_file()
    assert not (run_dir / "conclusion_ledger.json").exists()
    assert (run_dir / "capabilities.json").is_file()
    assert (run_dir / "claims" / "claim_ledger.json").is_file()
    assert (run_dir / "history" / "planner_messages.jsonl").is_file()
    assert (run_dir / "history" / "tool_events.jsonl").is_file()
    assert (run_dir / "rounds" / "R001" / "macro_execution_plan.json").is_file()
    assert (run_dir / "rounds" / "R001" / "microscopic_execution_plan.json").is_file()
    assert (run_dir / "rounds" / "R001" / "macro_tool_result.json").is_file()
    assert (run_dir / "rounds" / "R001" / "microscopic_tool_result.json").is_file()
    assert (run_dir / "rounds" / "R001" / "macro_report.json").is_file()
    assert (run_dir / "rounds" / "R001" / "microscopic_report.json").is_file()
    assert (run_dir / "rounds" / "R001" / "claim_ledger.json").is_file()
    assert (
        run_dir / "rounds" / "R002" / "macro_run_dimer_packing_proxy_report.json"
    ).is_file()
    assert (
        run_dir / "rounds" / "R002" / "macro_run_aggregate_contact_proxy_report.json"
    ).is_file()
    assert (
        run_dir
        / "rounds"
        / "R002"
        / "macro_run_crystal_restriction_checklist_report.json"
    ).is_file()
    assert (
        run_dir / "rounds" / "R002" / "microscopic_run_bright_dark_state_ordering_report.json"
    ).is_file()
    assert not (run_dir / "rounds" / "R002" / "verifier_execution_plan.json").exists()
    assert not (run_dir / "rounds" / "R002" / "verifier_tool_result.json").exists()
    assert not (run_dir / "rounds" / "R002" / "verifier_report.json").exists()

    persisted_final = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    assert persisted_final["round_refs"] == result.round_ids
    persisted_capabilities = json.loads(
        (run_dir / "capabilities.json").read_text(encoding="utf-8")
    )
    persisted_case = json.loads((run_dir / "case_run.json").read_text(encoding="utf-8"))
    assert persisted_case["operational_notes"]
    assert persisted_case["mechanism_agenda_coverage"]["agenda_items"]
    assert persisted_case["differential_mechanism_portfolio"]["rows"]
    assert persisted_case["claim_ledger"]["hypothesis_debts"]
    assert persisted_case["mechanism_predictions"]
    assert any(
        item["claim_refs"]
        for item in persisted_case["evidence_ledger"]["items"]
        if item["agent_name"] == "microscopic"
    )
    capability_ids = {
        item["capability_id"] for item in persisted_capabilities["capabilities"]
    }
    assert "microscopic.run_baseline_bundle" in capability_ids
    assert not any(item.startswith("verifier.") for item in capability_ids)


def test_mechanism_critic_payload_is_public_only() -> None:
    case_run = CaseRun(
        case_id="secret_internal_case",
        input=CaseInput(
            case_id="secret_internal_case",
            smiles="C1=CC=CC=C1",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "CASE_PUBLIC"},
        ),
        evidence_ledger=EvidenceLedger(
            case_id="secret_internal_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_rotor_torsion_topology",
                    summary="Rotor proxy was checked.",
                )
            ],
        ),
        artifact_manifest=ArtifactManifest(case_id="secret_internal_case"),
    )

    payload = _critic_payload(case_run, round_id="R001")
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["public_case"]["case_label"] == "current_smiles_case"  # type: ignore[index]
    coverage_rows = payload["mechanism_evidence_coverage_table"]
    assert isinstance(coverage_rows, list)
    assert coverage_rows[0]["route_hits"][0]["evidence_tier"] == "structural_trigger"
    assert "hidden_reference" not in encoded
    assert "reference_mechanisms" not in encoded
    assert "semantic_evidence_targets" not in encoded
    assert "semantic_diagnosis_targets" not in encoded
    assert "benchmark target" not in encoded.lower()
    assert "source paper answer" not in encoded.lower()


def test_mechanism_critic_sanitizer_preserves_evidence_tier_priority_floor() -> None:
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="critic_floor_case",
            smiles="Oc1ccccc1n2cccc2",
            user_query="Assess likely photophysical mechanisms.",
            metadata={"public_case_id": "PUBLIC_CRITIC_FLOOR"},
        ),
        evidence_ledger=EvidenceLedger(
            case_id="critic_floor_case",
            items=[
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    metrics={"esipt_motif_proxy": True},
                    summary="ESIPT structural motif proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_rim",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Rotor topology proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    summary="D-A layout proxy.",
                ),
            ],
        ),
    )
    rows = []
    for label in MECHANISM_POOL:
        rows.append(
            {
                "label": label,
                "trigger_status": "weak_trigger",
                "differential_priority": 0.9,
                "support_strength": "weak_proxy",
                "claim_status": "candidate_requires_validation",
                "positive_evidence_refs": [],
                "negative_evidence_refs": [],
                "missing_validation": [],
                "rationale": "No runtime evidence assigned.",
            }
        )
    for row in rows:
        if row["label"] == "ESIPT_PT":
            row["differential_priority"] = 0.9
            row["positive_evidence_refs"] = ["E_esipt"]
            row["rationale"] = "ESIPT motif proxy detected."
        if row["label"] == "RIM_RIR_RIV":
            row["differential_priority"] = 0.02
            row["positive_evidence_refs"] = ["E_rim"]
            row["rationale"] = "Rotor topology proxy detected."
        if row["label"] == "ICT_TICT_CT":
            row["differential_priority"] = 0.02
            row["positive_evidence_refs"] = ["E_da", "E_rim"]
            row["rationale"] = "D-A layout and torsion proxies detected."
        if row["label"] == "PET_ET":
            row["differential_priority"] = 0.9
            row["support_strength"] = "unsupported"
            row["positive_evidence_refs"] = ["E_esipt"]
    portfolio = _sanitize_portfolio_response(
        {
            "rows": rows,
            "evidence_attributions": [
                {
                    "evidence_id": "E_esipt",
                    "updates": [
                        {
                            "label": "ESIPT_PT",
                            "priority_effect": "increase",
                            "support_effect": "supports",
                            "warrant": "ESIPT-specific structural motif proxy.",
                            "boundary": "No PT barrier or spectral evidence.",
                        }
                    ],
                },
                {
                    "evidence_id": "E_rim",
                    "updates": [
                        {
                            "label": "RIM_RIR_RIV",
                            "priority_effect": "increase",
                            "support_effect": "supports",
                            "warrant": "Rotor topology proxy.",
                            "boundary": "No restriction evidence.",
                        }
                    ],
                },
                {
                    "evidence_id": "E_da",
                    "updates": [
                        {
                            "label": "ICT_TICT_CT",
                            "priority_effect": "increase",
                            "support_effect": "supports",
                            "warrant": "Donor-acceptor layout proxy.",
                            "boundary": "No CT excited-state descriptor.",
                        }
                    ],
                },
            ],
        },
        case_run=case_run,
        round_id="R001",
        allowed_evidence_ids={"E_esipt", "E_rim", "E_da"},
    )

    by_label = {row.label: row for row in portfolio.rows}
    assert by_label["ESIPT_PT"].differential_priority == pytest.approx(0.24)
    assert by_label["ICT_TICT_CT"].differential_priority == pytest.approx(0.215)
    assert by_label["RIM_RIR_RIV"].differential_priority == pytest.approx(0.12)
    assert by_label["PET_ET"].differential_priority == pytest.approx(0.05)


def test_parallel_summary_sanitizer_keeps_priority_separate_from_support() -> None:
    case_run = _case_run_for_gate().touch(current_round_id="R001")
    response = {
        "rows": [
            {
                "label": "RACI_CI_ACCESS",
                "trigger_status": "plausible",
                "priority": 72,
                "support_strength": "unsupported",
                "claim_status": "candidate_requires_validation",
                "positive_evidence_refs": [],
                "negative_evidence_refs": [],
                "missing_validation": ["CI search required."],
                "rationale": "Important candidate without direct CI evidence.",
            }
        ],
        "evidence_attributions": [],
        "policy_notes": [],
    }

    parallel = _sanitize_portfolio_response(
        response,
        case_run=case_run,
        round_id="R001",
        allowed_evidence_ids=set(),
        calibrate_evidence_tier_priority=False,
    )
    full_critic = _sanitize_portfolio_response(
        response,
        case_run=case_run,
        round_id="R001",
        allowed_evidence_ids=set(),
    )

    parallel_row = next(
        row for row in parallel.rows if row.label == "RACI_CI_ACCESS"
    )
    full_row = next(
        row for row in full_critic.rows if row.label == "RACI_CI_ACCESS"
    )
    assert parallel_row.differential_priority == pytest.approx(0.72)
    assert parallel_row.support_strength == "unsupported"
    assert full_row.differential_priority == pytest.approx(0.05)


def test_output_assembler_keeps_planner_ranking_and_uses_reviewed_evidence() -> None:
    evidence = _evidence_unit(
        evidence_id="E1",
        round_id="R001",
        source_report_id="R001:microscopic",
        agent_name="microscopic",
        family="torsion_sensitivity",
        capability_id="microscopic.run_torsion_snapshots",
        summary="Torsion brightness proxy is visible.",
        claim="Torsion snapshots show geometry-dependent brightness proxies.",
    )
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="RACI_CI_ACCESS",
            hypotheses=[
                HypothesisEntry(
                    name="RACI_CI_ACCESS",
                    confidence=0.9,
                    differential_priority=0.9,
                    rationale="Planner-owned final ranking.",
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E1"],
                    validation_needed=["Needs explicit S1/S0 crossing search."],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.4,
                    differential_priority=0.4,
                    rationale="Lower-priority Planner candidate.",
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(case_id="demo", items=[evidence]),
        differential_mechanism_portfolio=DifferentialMechanismPortfolio(
            portfolio_id="demo:R001:differential",
            case_id="demo",
            round_id="R001",
            rows=[
                DifferentialMechanismPortfolioRow(
                    label="RACI_CI_ACCESS",
                    trigger_status="triggered",
                    differential_priority=0.85,
                    support_strength="weak_proxy",
                    claim_status="candidate_requires_validation",
                    positive_evidence_refs=["E1"],
                    missing_validation=["Needs explicit S1/S0 crossing search."],
                    rationale="Torsion-coupled brightness makes CI-access a candidate.",
                ),
                DifferentialMechanismPortfolioRow(
                    label="ICT_TICT_CT",
                    trigger_status="weak_trigger",
                    differential_priority=0.4,
                    support_strength="unsupported",
                    claim_status="candidate_requires_validation",
                    rationale="No CT-specific runtime evidence was assigned.",
                ),
            ],
            evidence_attributions=[
                EvidenceMechanismAttribution(
                    evidence_id="E1",
                    updates=[
                        EvidenceMechanismAttributionUpdate(
                            label="RACI_CI_ACCESS",
                            priority_effect="increase",
                            support_effect="supports",
                            warrant=(
                                "Torsion-sensitive brightness is a proxy for a "
                                "motion-coupled nonradiative pathway."
                            ),
                            boundary="No explicit conical-intersection search was run.",
                        )
                    ],
                )
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R001:support:old",
                target_label="ICT_TICT_CT",
                support_level="strong",
                finding="Old support argument.",
                warrant="Old warrant.",
                boundary="Old boundary.",
                observation_refs=["E1"],
            )
        ],
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].label == "RACI_CI_ACCESS"
    assert predictions[0].differential_priority == pytest.approx(0.9)
    assert predictions[0].evidence_refs == ["E1"]
    assert predictions[0].support_strength == "weak_or_proxy"
    assert predictions[0].claim_status == "candidate_requires_validation"
    assert predictions[0].evidence_details[0].evidence_refs == ["E1"]
    assert "conical-intersection" in predictions[0].evidence_details[0].boundary
    assert predictions[1].label == "ICT_TICT_CT"


def test_output_assembler_expands_support_arguments_with_source_evidence_details() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="Oc1ccccn1", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="ESIPT_PT",
            hypotheses=[
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.29,
                    differential_priority=0.29,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_esipt"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    summary="ESIPT structural motif proxy was screened.",
                    claim=(
                        "ESIPT structural screen found donor and acceptor "
                        "motifs in the molecule."
                    ),
                    metrics={
                        "hbd_count": 1,
                        "hba_count": 2,
                        "proton_transfer_pair_proxy_count": 1,
                        "proton_transfer_pair_proxies": [
                            {
                                "pair_type": "O-H...N",
                                "donor_atom": "O1",
                                "acceptor_atom": "N7",
                                "topological_distance": 4,
                            }
                        ],
                    },
                )
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R002:support:001",
                target_label="ESIPT_PT",
                support_level="weak",
                finding="Initial verdict ranked ESIPT from runtime evidence.",
                warrant="A proton donor/acceptor motif is an ESIPT screening trigger.",
                boundary="No excited-state proton-transfer PES or spectra were run.",
                observation_refs=["E_esipt"],
            )
        ],
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].label == "ESIPT_PT"
    assert "O-H...N" in predictions[0].evidence_details[0].finding
    assert "hbd_count=1" in predictions[0].evidence_details[0].finding
    assert "Initial verdict ranked" not in predictions[0].evidence_details[0].finding


def test_output_assembler_exposes_state_proxy_metrics_for_judge() -> None:
    case_run = CaseRun(
        case_id="state_proxy_case",
        input=CaseInput(case_id="state_proxy_case", smiles="c1ccccc1", user_query="Assess."),
        artifact_manifest=ArtifactManifest(case_id="state_proxy_case"),
        portfolio=HypothesisPortfolio(
            current="RADIATIVE_RATE_STATE_BALANCE",
            hypotheses=[
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    confidence=0.3,
                    differential_priority=0.3,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_state"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="state_proxy_case",
            items=[
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_bright_dark_state_ordering",
                    family="state_ordering_brightness",
                    claim="Bright/dark state-ordering proxy was computed.",
                    summary="Bright/dark state-ordering proxy was computed.",
                    metrics={
                        "oscillator_strength": 0.137,
                        "bright_state_index": 2,
                        "state_count": 4,
                    },
                )
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    finding = predictions[0].evidence_details[0].finding
    assert "oscillator_strength=0.137" in finding
    assert "bright_state_index=2" in finding
    assert "state_count=4" in finding


def test_output_assembler_exposes_interaction_inventory_for_judge() -> None:
    case_run = CaseRun(
        case_id="interaction_proxy_case",
        input=CaseInput(
            case_id="interaction_proxy_case",
            smiles="O=C(N)c1ccncc1",
            user_query="Assess.",
        ),
        artifact_manifest=ArtifactManifest(case_id="interaction_proxy_case"),
        portfolio=HypothesisPortfolio(
            current="HOST_GUEST_INTERACTION",
            hypotheses=[
                HypothesisEntry(
                    name="HOST_GUEST_INTERACTION",
                    confidence=0.25,
                    differential_priority=0.25,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_interaction"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="interaction_proxy_case",
            items=[
                _evidence_unit(
                    evidence_id="E_interaction",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_polar_binding_site_prior",
                    family="geometry_precondition",
                    claim="Polar binding-site structural-prior observation was screened.",
                    summary="Polar binding-site structural-prior observation was screened.",
                    metrics={
                        "hba_count": 3,
                        "carbonyl_like_site_count": 1,
                        "polar_binding_site_proxy": True,
                        "host_guest_followup_trigger": True,
                        "pet_receptor_like_proxy": True,
                        "donor_atom_symbols": ["N2"],
                        "acceptor_atom_symbols": ["O0", "N2"],
                        "polar_atom_symbols": ["O0", "N2"],
                    },
                )
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    finding = predictions[0].evidence_details[0].finding
    assert "carbonyl_like_site_count=1" in finding
    assert "polar_atom_symbols=['O0', 'N2']" in finding
    assert "host_guest_followup_trigger=True" in finding


def test_output_assembler_exposes_packing_contact_metrics_for_judge() -> None:
    case_run = CaseRun(
        case_id="packing_proxy_case",
        input=CaseInput(case_id="packing_proxy_case", smiles="c1ccccc1", user_query="Assess."),
        artifact_manifest=ArtifactManifest(case_id="packing_proxy_case"),
        portfolio=HypothesisPortfolio(
            current="PACKING_HOST_MATRIX_CONFINEMENT",
            hypotheses=[
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.21,
                    differential_priority=0.21,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_pack"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="packing_proxy_case",
            items=[
                _evidence_unit(
                    evidence_id="E_pack",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.run_dimer_packing_proxy",
                    family="geometry_precondition",
                    claim="Packing/contact proxy route records bounded dimer descriptors.",
                    summary="Packing/contact proxy route records bounded dimer descriptors.",
                    metrics={
                        "dimer_contact_score_proxy": 0.60918,
                        "aromatic_fraction_proxy": 0.5,
                        "mol_logp": 2.0918,
                        "rotatable_bond_count": 2,
                    },
                )
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    finding = predictions[0].evidence_details[0].finding
    assert "dimer_contact_score_proxy=0.609" in finding
    assert "aromatic_fraction_proxy=0.500" in finding
    assert "mol_logp=2.092" in finding


def test_output_assembler_supplements_planner_label_with_screening_refs() -> None:
    case_run = CaseRun(
        case_id="screening_refs_case",
        input=CaseInput(case_id="screening_refs_case", smiles="c1ccccc1", user_query="Assess."),
        artifact_manifest=ArtifactManifest(case_id="screening_refs_case"),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.3,
                    differential_priority=0.3,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_solid"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="screening_refs_case",
            items=[
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro:rotor",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Rotor/torsion topology proxy was screened.",
                    metrics={"rotatable_bond_count": 5, "torsion_candidate_count": 5},
                ),
                _evidence_unit(
                    evidence_id="E_solid",
                    round_id="R002",
                    source_report_id="R002:macro:solid",
                    agent_name="macro",
                    capability_id="macro.run_solid_state_emission_proxy",
                    family="geometry_precondition",
                    summary="Solid-state emission proxy was screened.",
                    metrics={"rotor_burden_proxy": 5, "aggregation_prone_proxy": True},
                ),
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].evidence_refs == ["E_solid", "E_rotor"]
    findings = " ".join(item.finding for item in predictions[0].evidence_details)
    assert "rotatable_bond_count=5" in findings
    assert "torsion_candidate_count=5" in findings


def test_output_assembler_does_not_supplement_negative_solid_state_screen() -> None:
    case_run = CaseRun(
        case_id="negative_solid_proxy_case",
        input=CaseInput(
            case_id="negative_solid_proxy_case",
            smiles="CCN",
            user_query="Assess.",
        ),
        artifact_manifest=ArtifactManifest(case_id="negative_solid_proxy_case"),
        portfolio=HypothesisPortfolio(
            current="PACKING_HOST_MATRIX_CONFINEMENT",
            hypotheses=[
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.28,
                    differential_priority=0.28,
                    evidence_support="weak_or_proxy",
                    evidence_refs=[],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="negative_solid_proxy_case",
            items=[
                _evidence_unit(
                    evidence_id="E_solid_negative",
                    round_id="R001",
                    source_report_id="R001:macro:solid",
                    agent_name="macro",
                    capability_id="macro.run_solid_state_emission_proxy",
                    family="geometry_precondition",
                    summary="Solid-state emission proxy was screened.",
                    metrics={
                        "aggregation_prone_proxy": False,
                        "rotor_burden_proxy": 3,
                    },
                )
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].label == "PACKING_HOST_MATRIX_CONFINEMENT"
    assert predictions[0].evidence_refs == []
    assert predictions[0].evidence_details == []


def test_output_assembler_does_not_treat_generic_aggregation_as_exciton_evidence() -> None:
    case_run = CaseRun(
        case_id="generic_aggregation_case",
        input=CaseInput(
            case_id="generic_aggregation_case",
            smiles="c1ccccc1",
            user_query="Assess.",
        ),
        artifact_manifest=ArtifactManifest(case_id="generic_aggregation_case"),
        portfolio=HypothesisPortfolio(
            current="AGGREGATE_EXCITON_EXCIMER",
            hypotheses=[
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.22,
                    differential_priority=0.22,
                    evidence_support="weak_or_proxy",
                    evidence_refs=[],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="generic_aggregation_case",
            items=[
                _evidence_unit(
                    evidence_id="E_agg_generic",
                    round_id="R001",
                    source_report_id="R001:macro:aggregation",
                    agent_name="macro",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    family="geometry_precondition",
                    summary="Aggregation-prone scaffold proxy was screened.",
                    metrics={
                        "aggregation_prone_proxy": True,
                        "aromatic_ring_count": 4,
                        "mol_logp": 5.0,
                    },
                )
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].label == "AGGREGATE_EXCITON_EXCIMER"
    assert predictions[0].evidence_refs == []
    assert predictions[0].evidence_details == []


def test_output_assembler_orders_grounded_candidates_by_planner_priority() -> None:
    case_run = CaseRun(
        case_id="priority_order_case",
        input=CaseInput(case_id="priority_order_case", smiles="c1ccccc1", user_query="Assess."),
        artifact_manifest=ArtifactManifest(case_id="priority_order_case"),
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.2,
                    differential_priority=0.2,
                    evidence_refs=["E_rotor"],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.31,
                    differential_priority=0.31,
                    evidence_refs=["E_ct"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="priority_order_case",
            items=[
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro:rotor",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Rotor structural proxy was screened.",
                    metrics={"rotatable_bond_count": 2, "torsion_candidate_count": 2},
                ),
                _evidence_unit(
                    evidence_id="E_ct",
                    round_id="R001",
                    source_report_id="R001:macro:ct",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    summary="Donor/acceptor structural proxy was screened.",
                    metrics={
                        "donor_acceptor_proxy": 1,
                        "donor_acceptor_partition_proxy": 1.0,
                    },
                ),
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [item.label for item in predictions[:2]] == ["RIM_RIR_RIV", "ICT_TICT_CT"]
    assert predictions[1].differential_priority == pytest.approx(0.20)
    case_run.evidence_ledger.items[1].metrics["donor_atom_symbols"] = ["N1"]
    case_run.evidence_ledger.items[1].metrics["acceptor_atom_symbols"] = ["O1"]

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [item.label for item in predictions[:2]] == ["RIM_RIR_RIV", "ICT_TICT_CT"]
    assert predictions[1].differential_priority == pytest.approx(0.20)


def test_output_assembler_orders_ict_evidence_by_mechanism_specificity() -> None:
    case_run = CaseRun(
        case_id="ict_evidence_order_case",
        input=CaseInput(
            case_id="ict_evidence_order_case",
            smiles="N(c1ccccc1)c1ncccc1",
            user_query="Assess.",
        ),
        artifact_manifest=ArtifactManifest(case_id="ict_evidence_order_case"),
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.32,
                    differential_priority=0.32,
                    evidence_refs=["E_torsion", "E_ct", "E_da"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="ict_evidence_order_case",
            items=[
                _evidence_unit(
                    evidence_id="E_torsion",
                    round_id="R002",
                    source_report_id="R002:micro:torsion",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_brightness_coupling_scan",
                    family="torsion_sensitivity",
                    summary="Torsion-brightness proxy.",
                    metrics={"state_count": 0},
                ),
                _evidence_unit(
                    evidence_id="E_ct",
                    round_id="R002",
                    source_report_id="R002:micro:ct",
                    agent_name="microscopic",
                    capability_id="microscopic.extract_ct_descriptors_from_bundle",
                    family="charge_localization",
                    summary="CT descriptor artifact-backed proxy.",
                    metrics={
                        "state_count": 1,
                        "oscillator_strength": 0.18,
                        "missing_deliverable_count": 2,
                    },
                ),
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro:da",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    summary="Donor/acceptor structural proxy.",
                    metrics={
                        "donor_acceptor_proxy": 1,
                        "donor_acceptor_partition_proxy": 1.0,
                        "donor_atom_symbols": ["N1"],
                        "acceptor_atom_symbols": ["N7"],
                    },
                ),
            ],
        ),
    )

    prediction = OutputAssembler().build_mechanism_predictions(case_run)[0]

    assert prediction.evidence_details[0].evidence_refs == ["E_da"]
    assert prediction.evidence_details[1].evidence_refs == ["E_ct"]
    assert "donor_acceptor_proxy=1" in prediction.evidence_details[0].finding
    assert "state_count=1" in prediction.evidence_details[1].finding


def test_public_update_reducer_keeps_weak_ct_artifact_below_motion_proxy() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        round_ids=["R001", "R002"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.30,
                    differential_priority=0.30,
                    evidence_refs=["E_torsion"],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.24,
                    differential_priority=0.24,
                    evidence_refs=["E_da"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="ct_floor_case",
            items=[
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro:da",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    summary="Donor/acceptor structural proxy.",
                    metrics={
                        "donor_acceptor_proxy": 1,
                        "donor_acceptor_partition_proxy": 1.0,
                    },
                ),
                _evidence_unit(
                    evidence_id="E_torsion",
                    round_id="R002",
                    source_report_id="R002:micro:torsion",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_brightness_coupling_scan",
                    family="torsion_sensitivity",
                    summary="Torsion-brightness proxy.",
                    metrics={"state_count": 0},
                ),
                _evidence_unit(
                    evidence_id="E_ct",
                    round_id="R002",
                    source_report_id="R002:micro:ct",
                    agent_name="microscopic",
                    capability_id="microscopic.extract_ct_descriptors_from_bundle",
                    family="charge_localization",
                    summary="CT descriptor artifact-backed proxy.",
                    metrics={
                        "state_count": 1,
                        "missing_deliverable_count": 2,
                    },
                ),
            ],
        ),
    )
    payload = planner_module._compact_planner_payload(
        case_run=case_run,
        round_id="R003",
        stage="planner_update",
        context={},
        capability_registry=default_capability_registry(),
    )
    decision = planner_module._normalize_planner_llm_response(
        planner_module._public_update_evidence_response(
            case_run=case_run,
            round_id="R003",
            payload=payload,
        ),
        round_id="R003",
        existing_portfolio=case_run.portfolio.model_dump(mode="json"),
        previous_portfolio_hypotheses=[
            item.model_dump(mode="json")
            for item in planner_module._full_portfolio_hypotheses(case_run)
        ],
        allowed_evidence_ids=[item.evidence_id for item in case_run.evidence_ledger.items],
        capability_registry=default_capability_registry(),
        stage="planner_update",
        prompt_payload=payload,
    )

    hypotheses = {
        item["name"]: item for item in decision["portfolio"]["hypotheses"]
    }
    assert hypotheses["ICT_TICT_CT"]["differential_priority"] == pytest.approx(0.24)
    assert hypotheses["ICT_TICT_CT"]["differential_priority"] < hypotheses[
        "RIM_RIR_RIV"
    ]["differential_priority"]


def test_public_update_reducer_promotes_informative_ct_over_generic_motion_proxy() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        round_ids=["R001", "R002"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.30,
                    differential_priority=0.30,
                    evidence_refs=["E_torsion"],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.24,
                    differential_priority=0.24,
                    evidence_refs=["E_da"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="ct_floor_case",
            items=[
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro:da",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    summary="Donor/acceptor structural proxy.",
                    metrics={
                        "donor_acceptor_proxy": 1,
                        "donor_acceptor_partition_proxy": 1.0,
                    },
                ),
                _evidence_unit(
                    evidence_id="E_torsion",
                    round_id="R002",
                    source_report_id="R002:micro:torsion",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_brightness_coupling_scan",
                    family="torsion_sensitivity",
                    summary="Torsion-brightness proxy.",
                    metrics={"state_count": 0},
                ),
                _evidence_unit(
                    evidence_id="E_ct",
                    round_id="R002",
                    source_report_id="R002:micro:ct",
                    agent_name="microscopic",
                    capability_id="microscopic.extract_ct_descriptors_from_bundle",
                    family="charge_localization",
                    summary="CT descriptor artifact-backed proxy.",
                    metrics={
                        "state_count": 1,
                        "ct_descriptor_proxy": True,
                        "electron_hole_separation_proxy": True,
                        "missing_deliverable_count": 0,
                    },
                ),
            ],
        ),
    )
    payload = planner_module._compact_planner_payload(
        case_run=case_run,
        round_id="R003",
        stage="planner_update",
        context={},
        capability_registry=default_capability_registry(),
    )
    decision = planner_module._normalize_planner_llm_response(
        planner_module._public_update_evidence_response(
            case_run=case_run,
            round_id="R003",
            payload=payload,
        ),
        round_id="R003",
        existing_portfolio=case_run.portfolio.model_dump(mode="json"),
        previous_portfolio_hypotheses=[
            item.model_dump(mode="json")
            for item in planner_module._full_portfolio_hypotheses(case_run)
        ],
        allowed_evidence_ids=[item.evidence_id for item in case_run.evidence_ledger.items],
        capability_registry=default_capability_registry(),
        stage="planner_update",
        prompt_payload=payload,
    )

    hypotheses = {
        item["name"]: item for item in decision["portfolio"]["hypotheses"]
    }
    assert hypotheses["ICT_TICT_CT"]["differential_priority"] == pytest.approx(0.32)
    assert hypotheses["ICT_TICT_CT"]["differential_priority"] > hypotheses[
        "RIM_RIR_RIV"
    ]["differential_priority"]


def test_public_update_reducer_keeps_empty_torsion_scan_from_promoting_raci() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        round_ids=["R001", "R002"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.21,
                    differential_priority=0.21,
                    evidence_refs=["E_rotor"],
                ),
                HypothesisEntry(
                    name="RACI_CI_ACCESS",
                    confidence=0.16,
                    differential_priority=0.16,
                    evidence_refs=[],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="empty_torsion_case",
            items=[
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro:rotor",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    metrics={"rotatable_bond_count": 3, "torsion_candidate_count": 3},
                    summary="Rotor/torsion topology proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_torsion_empty",
                    round_id="R002",
                    source_report_id="R002:micro:torsion",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_brightness_coupling_scan",
                    family="torsion_sensitivity",
                    basis="computed",
                    metrics={"state_count": 0, "missing_deliverable_count": 1},
                    summary="Torsion scan produced no excited-state brightness descriptors.",
                ),
            ],
        ),
    )
    payload = planner_module._compact_planner_payload(
        case_run=case_run,
        round_id="R003",
        stage="planner_update",
        context={},
        capability_registry=default_capability_registry(),
    )
    decision = planner_module._normalize_planner_llm_response(
        planner_module._public_update_evidence_response(
            case_run=case_run,
            round_id="R003",
            payload=payload,
        ),
        round_id="R003",
        existing_portfolio=case_run.portfolio.model_dump(mode="json"),
        previous_portfolio_hypotheses=[
            item.model_dump(mode="json")
            for item in planner_module._full_portfolio_hypotheses(case_run)
        ],
        allowed_evidence_ids=[item.evidence_id for item in case_run.evidence_ledger.items],
        capability_registry=default_capability_registry(),
        stage="planner_update",
        prompt_payload=payload,
    )

    hypotheses = {
        item["name"]: item for item in decision["portfolio"]["hypotheses"]
    }
    assert hypotheses["RIM_RIR_RIV"]["differential_priority"] >= 0.21
    assert hypotheses["RACI_CI_ACCESS"]["differential_priority"] < hypotheses[
        "RIM_RIR_RIV"
    ]["differential_priority"]
    assert "E_torsion_empty" not in hypotheses["RACI_CI_ACCESS"].get("evidence_refs", [])


def test_public_update_reducer_allows_raci_with_torsion_brightness_signal() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        round_ids=["R001", "R002"],
        portfolio=HypothesisPortfolio(
            current="RACI_CI_ACCESS",
            hypotheses=[
                HypothesisEntry(
                    name="RACI_CI_ACCESS",
                    confidence=0.16,
                    differential_priority=0.16,
                    evidence_refs=[],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="raci_torsion_signal_case",
            items=[
                _evidence_unit(
                    evidence_id="E_torsion_signal",
                    round_id="R002",
                    source_report_id="R002:micro:torsion",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_brightness_coupling_scan",
                    family="torsion_sensitivity",
                    basis="computed",
                    metrics={
                        "state_count": 3,
                        "oscillator_strength_range": 0.12,
                        "missing_deliverable_count": 0,
                    },
                    summary="Torsion scan produced geometry-dependent brightness descriptors.",
                )
            ],
        ),
    )
    payload = planner_module._compact_planner_payload(
        case_run=case_run,
        round_id="R003",
        stage="planner_update",
        context={},
        capability_registry=default_capability_registry(),
    )
    decision = planner_module._normalize_planner_llm_response(
        planner_module._public_update_evidence_response(
            case_run=case_run,
            round_id="R003",
            payload=payload,
        ),
        round_id="R003",
        existing_portfolio=case_run.portfolio.model_dump(mode="json"),
        previous_portfolio_hypotheses=[
            item.model_dump(mode="json")
            for item in planner_module._full_portfolio_hypotheses(case_run)
        ],
        allowed_evidence_ids=[item.evidence_id for item in case_run.evidence_ledger.items],
        capability_registry=default_capability_registry(),
        stage="planner_update",
        prompt_payload=payload,
    )

    hypotheses = {
        item["name"]: item for item in decision["portfolio"]["hypotheses"]
    }
    assert hypotheses["RACI_CI_ACCESS"]["differential_priority"] >= 0.27
    assert "E_torsion_signal" in hypotheses["RACI_CI_ACCESS"].get("evidence_refs", [])


def test_public_update_reducer_does_not_let_weak_ct_artifact_overtake_esipt() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        round_ids=["R001", "R002"],
        portfolio=HypothesisPortfolio(
            current="ESIPT_PT",
            hypotheses=[
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.31,
                    differential_priority=0.31,
                    evidence_refs=["E_esipt"],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.24,
                    differential_priority=0.24,
                    evidence_refs=["E_da"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="weak_ct_esipt_case",
            items=[
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro:esipt",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    summary="ESIPT structural motif proxy.",
                    metrics={
                        "esipt_motif_proxy": True,
                        "proton_transfer_pair_proxy_count": 2,
                    },
                ),
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro:da",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    summary="Donor/acceptor structural proxy.",
                    metrics={
                        "donor_acceptor_proxy": 2,
                        "donor_acceptor_partition_proxy": 1.0,
                    },
                ),
                _evidence_unit(
                    evidence_id="E_ct",
                    round_id="R002",
                    source_report_id="R002:micro:ct",
                    agent_name="microscopic",
                    capability_id="microscopic.extract_ct_descriptors_from_bundle",
                    family="charge_localization",
                    summary="Weak CT descriptor artifact-backed proxy.",
                    metrics={
                        "state_count": 1,
                        "oscillator_strength": 0.07,
                        "missing_deliverable_count": 2,
                    },
                ),
            ],
        ),
    )
    payload = planner_module._compact_planner_payload(
        case_run=case_run,
        round_id="R003",
        stage="planner_update",
        context={},
        capability_registry=default_capability_registry(),
    )
    decision = planner_module._normalize_planner_llm_response(
        planner_module._public_update_evidence_response(
            case_run=case_run,
            round_id="R003",
            payload=payload,
        ),
        round_id="R003",
        existing_portfolio=case_run.portfolio.model_dump(mode="json"),
        previous_portfolio_hypotheses=[
            item.model_dump(mode="json")
            for item in planner_module._full_portfolio_hypotheses(case_run)
        ],
        allowed_evidence_ids=[item.evidence_id for item in case_run.evidence_ledger.items],
        capability_registry=default_capability_registry(),
        stage="planner_update",
        prompt_payload=payload,
    )

    hypotheses = {
        item["name"]: item for item in decision["portfolio"]["hypotheses"]
    }
    assert hypotheses["ESIPT_PT"]["differential_priority"] == pytest.approx(0.31)
    assert hypotheses["ICT_TICT_CT"]["differential_priority"] == pytest.approx(0.24)
    assert hypotheses["ICT_TICT_CT"]["differential_priority"] < hypotheses[
        "ESIPT_PT"
    ]["differential_priority"]


def test_output_assembler_keeps_planner_order_over_reviewed_rows() -> None:
    evidence = [
        _evidence_unit(
            evidence_id="E_rotor",
            round_id="R001",
            source_report_id="R001:macro:rotor",
            agent_name="macro",
            family="geometry_precondition",
            capability_id="macro.screen_rotor_torsion_topology",
            summary="Rotor and torsion topology proxy was screened.",
        ),
        _evidence_unit(
            evidence_id="E_esipt",
            round_id="R001",
            source_report_id="R001:macro:esipt",
            agent_name="macro",
            family="geometry_precondition",
            capability_id="macro.screen_esipt_structural_motif",
            summary="ESIPT structural motif proxy was screened.",
        ),
        _evidence_unit(
            evidence_id="E_agg",
            round_id="R001",
            source_report_id="R001:macro:aggregation",
            agent_name="macro",
            family="geometry_precondition",
            capability_id="macro.screen_aggregation_prone_scaffold",
            summary="Aggregation-prone scaffold proxy was screened.",
        ),
    ]
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(
            case_id="demo",
            smiles="N(c1ccccc1)c2ccc(C=C)cc2",
            user_query="Assess.",
        ),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.2,
                    differential_priority=0.2,
                ),
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.2,
                    differential_priority=0.2,
                ),
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.2,
                    differential_priority=0.2,
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(case_id="demo", items=evidence),
        differential_mechanism_portfolio=DifferentialMechanismPortfolio(
            portfolio_id="demo:R001:differential",
            case_id="demo",
            round_id="R001",
            rows=[
                DifferentialMechanismPortfolioRow(
                    label="AGGREGATE_EXCITON_EXCIMER",
                    trigger_status="weak_trigger",
                    differential_priority=0.2,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_agg"],
                    rationale="Aggregation proxy only.",
                ),
                DifferentialMechanismPortfolioRow(
                    label="RIM_RIR_RIV",
                    trigger_status="weak_trigger",
                    differential_priority=0.21,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_rotor", "E_agg"],
                    rationale="Rotor and aggregation proxies.",
                ),
                DifferentialMechanismPortfolioRow(
                    label="ESIPT_PT",
                    trigger_status="weak_trigger",
                    differential_priority=0.2,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_esipt"],
                    rationale="Mechanism-specific ESIPT motif proxy.",
                ),
            ],
        ),
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [item.label for item in predictions] == [
        "RIM_RIR_RIV",
        "ESIPT_PT",
        "AGGREGATE_EXCITON_EXCIMER",
    ]


def test_output_assembler_uses_planner_portfolio_not_reviewed_row_order() -> None:
    evidence = [
        _evidence_unit(
            evidence_id="E_rotor",
            round_id="R001",
            source_report_id="R001:macro:rotor",
            agent_name="macro",
            family="geometry_precondition",
            capability_id="macro.screen_rotor_torsion_topology",
            summary="Rotor and torsion topology proxy was screened.",
        ),
        _evidence_unit(
            evidence_id="E_da",
            round_id="R001",
            source_report_id="R001:macro:da",
            agent_name="macro",
            family="geometry_precondition",
            capability_id="macro.screen_donor_acceptor_layout",
            summary="Donor-acceptor layout proxy was screened.",
        ),
        _evidence_unit(
            evidence_id="E_state",
            round_id="R001",
            source_report_id="R001:microscopic",
            agent_name="microscopic",
            family="state_ordering_brightness",
            capability_id="microscopic.run_baseline_bundle",
            basis="computed",
            summary="Low-cost state-ordering proxy was screened.",
        ),
    ]
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="N(c1ccccc1)c2ccc(C=C)cc2", user_query="Assess."),
        portfolio=HypothesisPortfolio(current="RIM_RIR_RIV"),
        evidence_ledger=EvidenceLedger(case_id="demo", items=evidence),
        differential_mechanism_portfolio=DifferentialMechanismPortfolio(
            portfolio_id="demo:R001:differential",
            case_id="demo",
            round_id="R001",
            rows=[
                DifferentialMechanismPortfolioRow(
                    label="RIM_RIR_RIV",
                    trigger_status="weak_trigger",
                    differential_priority=0.22,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_rotor"],
                    rationale="Rotor proxy only.",
                ),
                DifferentialMechanismPortfolioRow(
                    label="ICT_TICT_CT",
                    trigger_status="weak_trigger",
                    differential_priority=0.2,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_da", "E_state"],
                    rationale="D-A plus microscopic state proxy.",
                ),
                DifferentialMechanismPortfolioRow(
                    label="PACKING_HOST_MATRIX_CONFINEMENT",
                    trigger_status="weak_trigger",
                    differential_priority=0.1,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_rotor", "E_da", "E_state"],
                    rationale="More evidence, but outside the low-margin group.",
                ),
            ],
        ),
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions == []


def test_output_assembler_keeps_low_margin_order_for_only_weak_structural_refs() -> None:
    evidence = [
        _evidence_unit(
            evidence_id="E_rotor",
            round_id="R001",
            source_report_id="R001:macro:rotor",
            agent_name="macro",
            family="geometry_precondition",
            capability_id="macro.screen_rotor_torsion_topology",
            summary="Rotor and torsion topology proxy was screened.",
        ),
        _evidence_unit(
            evidence_id="E_agg",
            round_id="R001",
            source_report_id="R001:macro:aggregation",
            agent_name="macro",
            family="geometry_precondition",
            capability_id="macro.screen_aggregation_prone_scaffold",
            summary="Aggregation-prone scaffold proxy was screened.",
        ),
    ]
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="N(c1ccccc1)c2ccc(C=C)cc2", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.22,
                    differential_priority=0.22,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_rotor"],
                ),
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.2,
                    differential_priority=0.2,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_rotor", "E_agg"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(case_id="demo", items=evidence),
        differential_mechanism_portfolio=DifferentialMechanismPortfolio(
            portfolio_id="demo:R001:differential",
            case_id="demo",
            round_id="R001",
            rows=[
                DifferentialMechanismPortfolioRow(
                    label="RIM_RIR_RIV",
                    trigger_status="weak_trigger",
                    differential_priority=0.22,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_rotor"],
                    rationale="Rotor proxy.",
                ),
                DifferentialMechanismPortfolioRow(
                    label="PACKING_HOST_MATRIX_CONFINEMENT",
                    trigger_status="weak_trigger",
                    differential_priority=0.2,
                    support_strength="weak_proxy",
                    positive_evidence_refs=["E_rotor", "E_agg"],
                    rationale="More weak structural proxies, but no stronger tier.",
                ),
            ],
        ),
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [item.label for item in predictions[:2]] == [
        "RIM_RIR_RIV",
        "PACKING_HOST_MATRIX_CONFINEMENT",
    ]


def test_reviewer_feedback_floor_is_not_erased_before_next_planner_update() -> None:
    case_run = CaseRun(
        case_id="reviewer_floor",
        input=CaseInput(
            case_id="reviewer_floor",
            smiles="Oc1ccc2ncccc2c1",
            user_query="Assess likely photophysical mechanisms.",
        ),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.1,
                    differential_priority=0.1,
                    rationale="Generic rotor prior.",
                ),
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.0,
                    differential_priority=0.0,
                    rationale="Planner has not absorbed the ESIPT screen yet.",
                ),
                HypothesisEntry(
                    name="PET_ET",
                    confidence=0.0,
                    differential_priority=0.0,
                    rationale="Unsupported reviewer row should not be floored.",
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(case_id="reviewer_floor", items=[]),
        artifact_manifest=ArtifactManifest(case_id="reviewer_floor"),
    )
    reviewer_portfolio = DifferentialMechanismPortfolio(
        portfolio_id="reviewer_floor:R001:differential",
        case_id="reviewer_floor",
        round_id="R001",
        rows=[
            DifferentialMechanismPortfolioRow(
                label="RIM_RIR_RIV",
                trigger_status="not_triggered",
                differential_priority=0.1,
                support_strength="unsupported",
                rationale="Generic broad candidate.",
            ),
            DifferentialMechanismPortfolioRow(
                label="ESIPT_PT",
                trigger_status="weak_trigger",
                differential_priority=0.2,
                support_strength="weak_proxy",
                positive_evidence_refs=["E_esipt"],
                missing_validation=["Need PT barrier or spectral evidence."],
                rationale="Public structural ESIPT motif proxy was detected.",
            ),
            DifferentialMechanismPortfolioRow(
                label="PET_ET",
                trigger_status="weak_trigger",
                differential_priority=0.2,
                support_strength="unsupported",
                positive_evidence_refs=["E_pet"],
                rationale="Unsupported rows cannot gain priority from the floor.",
            ),
        ],
    )

    merged = _apply_planner_priorities_to_differential_portfolio(
        case_run,
        reviewer_portfolio,
    )

    rows = {row.label: row for row in merged.rows}
    assert rows["ESIPT_PT"].differential_priority == pytest.approx(0.12)
    assert rows["PET_ET"].differential_priority == pytest.approx(0.0)
    assert rows["RIM_RIR_RIV"].differential_priority == pytest.approx(0.1)


def test_reviewer_feedback_can_rank_when_planner_portfolio_is_uninformative() -> None:
    case_run = CaseRun(
        case_id="uninformative_planner",
        input=CaseInput(
            case_id="uninformative_planner",
            smiles="Oc1ccc2ncccc2c1",
            user_query="Assess likely photophysical mechanisms.",
        ),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(name=label, confidence=0.0, differential_priority=0.0)
                for label in ("RIM_RIR_RIV", "ESIPT_PT", "ICT_TICT_CT")
            ],
        ),
        evidence_ledger=EvidenceLedger(case_id="uninformative_planner", items=[]),
        artifact_manifest=ArtifactManifest(case_id="uninformative_planner"),
    )
    reviewer_portfolio = DifferentialMechanismPortfolio(
        portfolio_id="uninformative_planner:R001:differential",
        case_id="uninformative_planner",
        round_id="R001",
        rows=[
            DifferentialMechanismPortfolioRow(
                label="RIM_RIR_RIV",
                trigger_status="weak_trigger",
                differential_priority=0.12,
                support_strength="weak_proxy",
                positive_evidence_refs=["E_rim"],
                rationale="Rotor proxy was detected.",
            ),
            DifferentialMechanismPortfolioRow(
                label="ESIPT_PT",
                trigger_status="triggered",
                differential_priority=0.24,
                support_strength="weak_proxy",
                positive_evidence_refs=["E_esipt"],
                rationale="ESIPT motif proxy was detected.",
            ),
            DifferentialMechanismPortfolioRow(
                label="ICT_TICT_CT",
                trigger_status="weak_trigger",
                differential_priority=0.18,
                support_strength="weak_proxy",
                positive_evidence_refs=["E_ict"],
                rationale="D-A proxy was detected.",
            ),
        ],
    )

    merged = _apply_planner_priorities_to_differential_portfolio(
        case_run,
        reviewer_portfolio,
    )

    rows = {row.label: row for row in merged.rows}
    assert rows["ESIPT_PT"].differential_priority == pytest.approx(0.24)
    assert rows["ICT_TICT_CT"].differential_priority == pytest.approx(0.18)
    assert rows["RIM_RIR_RIV"].differential_priority == pytest.approx(0.12)


def test_output_assembler_marks_low_margin_predictions() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.51,
                    differential_priority=0.51,
                    rationale="Top candidate.",
                ),
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.49,
                    differential_priority=0.49,
                    rationale="Close competing candidate.",
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(case_id="demo", items=[]),
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].low_margin_group == "LMG001"
    assert predictions[1].low_margin_group == "LMG001"
    assert predictions[0].margin_to_next == pytest.approx(0.02)
    assert "low-margin competing candidates" in predictions[0].ranking_stability_note
    assert predictions[0].support_strength == "unsupported"


def test_output_assembler_does_not_export_unsupported_no_ref_candidate_above_grounded_one() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="RACI_CI_ACCESS",
            hypotheses=[
                HypothesisEntry(
                    name="RACI_CI_ACCESS",
                    confidence=0.2,
                    differential_priority=0.2,
                    evidence_support="unsupported",
                    claim_status="underdetermined",
                    rationale="No explicit CI evidence was collected.",
                ),
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.19,
                    differential_priority=0.19,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    rationale="Rotor topology is a public proxy.",
                    evidence_refs=["E_rotor"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro:rotor",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_rotor_torsion_topology",
                    summary="Rotor and torsion topology proxy was screened.",
                )
            ],
        ),
        differential_mechanism_portfolio=None,
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R001:support:rim",
                target_label="RIM_RIR_RIV",
                support_level="weak",
                finding="Rotor topology was observed.",
                warrant="Rotor/torsion topology is relevant to RIM/RIR screening.",
                boundary="No direct condensed-phase restriction evidence was collected.",
                observation_refs=["E_rotor"],
            )
        ],
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].label == "RIM_RIR_RIV"
    assert predictions[0].evidence_refs == ["E_rotor"]
    assert predictions[1].label == "RACI_CI_ACCESS"
    assert predictions[1].support_strength == "unsupported"


def test_output_assembler_uses_rotor_flexibility_floor_for_rim() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="CC1=CC=CC=C1", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.24,
                    differential_priority=0.24,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    rationale="Weak donor-acceptor proxy.",
                    evidence_refs=["E_ict"],
                ),
                HypothesisEntry(
                    name="TRIPLET_METAL_ENERGY_TRANSFER",
                    confidence=0.23,
                    differential_priority=0.23,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    rationale="Heavy-atom or triplet proxy.",
                    evidence_refs=["E_triplet"],
                ),
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.22,
                    differential_priority=0.22,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    rationale="Packing proxy.",
                    evidence_refs=["E_pack"],
                ),
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.12,
                    differential_priority=0.12,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    rationale="RIM was underweighted by the previous portfolio.",
                    evidence_refs=["E_solid"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                _evidence_unit(
                    evidence_id="E_ict",
                    round_id="R001",
                    source_report_id="R001:macro:ict",
                    agent_name="macro",
                    family="charge_localization",
                    capability_id="macro.screen_donor_acceptor_layout",
                    metrics={"donor_acceptor_proxy": 1},
                    summary="Weak donor-acceptor proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_triplet",
                    round_id="R001",
                    source_report_id="R001:macro:triplet",
                    agent_name="macro",
                    family="charge_localization",
                    capability_id="macro.screen_metal_triplet_prior",
                    metrics={"sulfur_phosphorus_triplet_prior": True},
                    summary="Triplet proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_pack",
                    round_id="R001",
                    source_report_id="R001:macro:pack",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.run_dimer_packing_proxy",
                    metrics={"dimer_contact_score_proxy": 0.58},
                    summary="Dimer contact proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_solid",
                    round_id="R001",
                    source_report_id="R001:macro:solid",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.run_solid_state_emission_proxy",
                    metrics={
                        "aggregation_prone_proxy": True,
                        "rotor_burden_proxy": 1,
                    },
                    summary="Solid-state rotor burden proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro:rotor",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_rotor_torsion_topology",
                    metrics={
                        "rotatable_bond_count": 1,
                        "torsion_candidate_count": 1,
                        "branch_point_count": 12,
                        "flexibility_proxy": 13.6,
                    },
                    summary="Rotor topology shows one formal rotor in a branched scaffold.",
                ),
            ],
        ),
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert predictions[0].label == "RIM_RIR_RIV"
    assert predictions[0].differential_priority == pytest.approx(0.25)
    assert set(predictions[0].evidence_refs) >= {"E_solid", "E_rotor"}


def test_output_assembler_keeps_planner_portfolio_order_under_grounded_ties() -> None:
    case_run = CaseRun(
        case_id="demo",
        input=CaseInput(case_id="demo", smiles="C1=CC=CC=C1", user_query="Assess."),
        portfolio=HypothesisPortfolio(
            current="ESIPT_PT",
            runner_up="PET_ET",
            hypotheses=[
                HypothesisEntry(
                    name="HOST_GUEST_INTERACTION",
                    confidence=0.02,
                    differential_priority=0.02,
                    rationale="Same-priority candidate listed first by raw ordering.",
                    evidence_refs=["E_host"],
                ),
                HypothesisEntry(
                    name="PET_ET",
                    confidence=0.02,
                    differential_priority=0.02,
                    rationale="Planner runner-up under a low-margin tie.",
                    evidence_refs=["E_pet"],
                ),
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.02,
                    differential_priority=0.02,
                    rationale="Planner current hypothesis under a low-margin tie.",
                    evidence_refs=["E_esipt"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                _evidence_unit(
                    evidence_id="E_host",
                    round_id="R001",
                    source_report_id="R001:macro:host",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_polar_binding_site_prior",
                    summary="Host-guest proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_pet",
                    round_id="R001",
                    source_report_id="R001:macro:pet",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_polar_binding_site_prior",
                    summary="PET proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro:esipt",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_esipt_structural_motif",
                    summary="ESIPT proxy.",
                ),
            ],
        ),
        artifact_manifest=ArtifactManifest(case_id="demo"),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [item.label for item in predictions] == [
        "HOST_GUEST_INTERACTION",
        "PET_ET",
        "ESIPT_PT",
    ]


def test_orchestrator_agenda_review_trigger_uses_low_margin_only() -> None:
    zero_case = _case_run_for_gate().touch(
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            runner_up="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(name="RIM_RIR_RIV", confidence=0.0),
                HypothesisEntry(name="ICT_TICT_CT", confidence=0.0),
            ],
        )
    )
    assert not _has_low_margin_portfolio(zero_case)

    separated_case = zero_case.touch(
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            runner_up="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.7,
                    differential_priority=0.7,
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.4,
                    differential_priority=0.4,
                ),
            ],
        )
    )
    assert not _has_low_margin_portfolio(separated_case)

    low_margin_case = zero_case.touch(
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            runner_up="AGGREGATE_EXCITON_EXCIMER",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.51,
                    differential_priority=0.51,
                ),
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.49,
                    differential_priority=0.49,
                ),
            ],
        )
    )
    assert _has_low_margin_portfolio(low_margin_case)


def test_orchestrator_evidence_sufficiency_finalize_guard(tmp_path) -> None:
    orchestrator = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=tmp_path, max_rounds=30)
    )
    case_run = _case_run_for_gate().touch(
        round_ids=["R001", "R002"],
        photophysics_review=PhotophysicsReview(
            review_id="demo:R002:photophysics_review",
            case_id="demo",
            round_id="R002",
        ),
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R002",
                    source_report_id="R002:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_brightness_coupling_scan",
                    family="torsion_sensitivity",
                    summary="Torsion-brightness coupling proxy.",
                ),
                _evidence_unit(
                    evidence_id="E2",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Rotor structural proxy.",
                ),
                _evidence_unit(
                    evidence_id="E3",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    family="geometry_precondition",
                    summary="Aggregation structural proxy.",
                ),
            ],
        ),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.42,
                    differential_priority=0.42,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E1"],
                ),
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.30,
                    differential_priority=0.30,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E3"],
                ),
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R002:support:rim",
                target_label="RIM_RIR_RIV",
                support_level="weak",
                finding="Torsion-brightness coupling proxy.",
                warrant="The route screens motion-coupled brightness changes.",
                boundary="No condensed-phase restriction experiment was run.",
                observation_refs=["E1"],
            )
        ],
    )

    assert not orchestrator._should_finalize_on_evidence_sufficiency(case_run)
    covered_case = case_run.touch(
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                *case_run.evidence_ledger.items,
                _evidence_unit(
                    evidence_id="E4",
                    round_id="R003",
                    source_report_id="R003:macro",
                    agent_name="macro",
                    capability_id="macro.run_dimer_packing_proxy",
                    family="geometry_precondition",
                    summary="Dimer packing proxy follow-up.",
                ),
                _evidence_unit(
                    evidence_id="E5",
                    round_id="R003",
                    source_report_id="R003:macro",
                    agent_name="macro",
                    capability_id="macro.run_aggregate_contact_proxy",
                    family="geometry_precondition",
                    summary="Aggregate contact proxy follow-up.",
                ),
                _evidence_unit(
                    evidence_id="E6",
                    round_id="R003",
                    source_report_id="R003:macro",
                    agent_name="macro",
                    capability_id="macro.run_crystal_restriction_checklist",
                    family="geometry_precondition",
                    summary="Crystal restriction checklist follow-up.",
                ),
            ],
        )
    )
    assert orchestrator._should_finalize_on_evidence_sufficiency(covered_case)
    low_margin_case = case_run.touch(
        evidence_ledger=covered_case.evidence_ledger,
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.42,
                    differential_priority=0.42,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E1"],
                ),
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.39,
                    differential_priority=0.39,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E3"],
                ),
            ],
        )
    )
    assert orchestrator._should_finalize_on_evidence_sufficiency(low_margin_case)
    low_margin_without_top_support = low_margin_case.touch(mechanism_support_arguments=[])
    assert not orchestrator._should_finalize_on_evidence_sufficiency(
        low_margin_without_top_support
    )
    dispatch_gate = DecisionGateResult(
        gate_id="R002:gate",
        round_id="R002",
        status="allow",
        terminal=False,
        dispatch_requests=[
            DispatchRequest(
                dispatch_id="R002:dispatch:macro:screen_rotor_torsion_topology",
                round_id="R002",
                agent_name="macro",
                capability_id="macro.screen_rotor_torsion_topology",
                task="Collect rotor topology.",
                objective="Collect rotor topology.",
                evidence_goal_family="geometry_precondition",
                route="screen_rotor_torsion_topology",
            )
        ],
    )
    assert orchestrator._can_finalize_on_evidence_sufficiency_after_round(
        covered_case,
        store=RunStore(tmp_path, "evidence_sufficiency_guard"),
        gate_result=dispatch_gate,
        calibrated_after_dispatch=False,
    ) is False
    assert orchestrator._can_finalize_on_evidence_sufficiency_after_round(
        covered_case,
        store=RunStore(tmp_path, "evidence_sufficiency_guard"),
        gate_result=dispatch_gate,
        calibrated_after_dispatch=True,
    )


def test_orchestrator_evidence_sufficiency_allows_missing_review_after_verifier_failure(
    tmp_path,
) -> None:
    orchestrator = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=tmp_path, max_rounds=30)
    )
    case_run = _case_run_for_gate().touch(
        round_ids=["R001", "R002"],
        photophysics_review=None,
        runtime={
            "orchestrator": "mechcal",
            "langgraph": False,
            "verifier_failures": [
                {
                    "type": "RuntimeError",
                    "message": "APITimeoutError: Request timed out.",
                    "agent": "PhotophysicsArbiterAgent",
                    "policy": "continue_without_scientific_fallback",
                }
            ],
            "mechanism_critic_failures": [
                {
                    "type": "RuntimeError",
                    "message": "APITimeoutError: Request timed out.",
                    "agent": "MechanismCriticAgent",
                    "policy": (
                        "continue_with_previous_differential_portfolio_without_"
                        "scientific_fallback"
                    ),
                }
            ],
        },
        evidence_ledger=EvidenceLedger(
            case_id="demo",
            items=[
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Rotor proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_agg",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    family="geometry_precondition",
                    summary="Aggregation proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R002",
                    source_report_id="R002:micro",
                    agent_name="microscopic",
                    capability_id="microscopic.run_bright_dark_state_ordering",
                    family="state_ordering_brightness",
                    summary="State ordering proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_dimer",
                    round_id="R002",
                    source_report_id="R002:macro",
                    agent_name="macro",
                    capability_id="macro.run_dimer_packing_proxy",
                    family="geometry_precondition",
                    summary="Dimer packing proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_contact",
                    round_id="R002",
                    source_report_id="R002:macro",
                    agent_name="macro",
                    capability_id="macro.run_aggregate_contact_proxy",
                    family="geometry_precondition",
                    summary="Aggregate contact proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_checklist",
                    round_id="R002",
                    source_report_id="R002:macro",
                    agent_name="macro",
                    capability_id="macro.run_crystal_restriction_checklist",
                    family="geometry_precondition",
                    summary="Crystal restriction checklist.",
                ),
            ],
        ),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.42,
                    differential_priority=0.42,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E_state"],
                ),
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.30,
                    differential_priority=0.30,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E_agg"],
                ),
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R002:support:rim",
                target_label="RIM_RIR_RIV",
                support_level="weak",
                finding="Rotor proxy.",
                warrant="Rotor topology screens RIM/RIR/RIV.",
                boundary="No direct condensed-phase restriction evidence was collected.",
                observation_refs=["E_state"],
            )
        ],
    )

    assert orchestrator._should_finalize_on_evidence_sufficiency(case_run)


def test_convergence_finalization_uses_current_planner_state_without_recalibration(
    tmp_path,
) -> None:
    class FailingPlanner:
        def calibrate_final_portfolio(self, case_run: CaseRun, *, round_id: str):
            del case_run, round_id
            raise AssertionError("convergence finalization must not recalibrate")

    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=30)
    deps = _fake_llm_deps(config)
    deps.planner = FailingPlanner()  # type: ignore[assignment]
    orchestrator = MechCALOrchestrator(config, deps=deps)
    case_run = _case_run_for_gate().touch(
        case_id="converged_without_recalibration",
        input=CaseInput(
            case_id="converged_without_recalibration",
            smiles="C1=CC=CC=C1",
            user_query="Assess.",
        ),
        status="running",
        current_round_id="R006",
        round_ids=["R001", "R002", "R003", "R004", "R005", "R006"],
        photophysics_review=PhotophysicsReview(
            review_id="converged_without_recalibration:R006:photophysics_review",
            case_id="converged_without_recalibration",
            round_id="R006",
        ),
        evidence_ledger=EvidenceLedger(
            case_id="converged_without_recalibration",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_esipt_structural_motif",
                    summary="ESIPT structural proxy.",
                )
            ],
        ),
        portfolio=HypothesisPortfolio(
            current="ESIPT_PT",
            hypotheses=[
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.42,
                    differential_priority=0.42,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    evidence_refs=["E1"],
                    rationale="Planner-owned converged ranking.",
                ),
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.30,
                    differential_priority=0.30,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    rationale="Planner-owned runner-up.",
                ),
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R006:support:esipt",
                target_label="ESIPT_PT",
                support_level="weak",
                finding="ESIPT structural proxy.",
                warrant="The route screens a proton-transfer-compatible motif.",
                boundary="No excited-state proton-transfer computation was run.",
                observation_refs=["E1"],
            )
        ],
    )
    store = RunStore(tmp_path, "converged_without_recalibration")
    store.initialize()

    result = orchestrator._finish_converged_case(case_run, store)

    assert result.status == "finalized"
    assert result.mechanism_predictions[0].label == "ESIPT_PT"
    assert result.final_answer is not None
    assert result.final_answer.mechanism_predictions == result.mechanism_predictions
    assert (store.root / "mechanism_predictions.json").is_file()
    assert not (store.root / "history" / "planner_final_calibrations.jsonl").exists()


def test_low_margin_convergence_allows_stable_top1_and_top3_set() -> None:
    assert _rankings_stable_under_low_margin(
        [
            ("RADIATIVE_RATE_STATE_BALANCE", "PACKING_HOST_MATRIX_CONFINEMENT", "SOKR_ANTI_KASHA"),
            ("RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA", "PACKING_HOST_MATRIX_CONFINEMENT"),
            ("RADIATIVE_RATE_STATE_BALANCE", "PACKING_HOST_MATRIX_CONFINEMENT", "SOKR_ANTI_KASHA"),
        ]
    )
    assert not _rankings_stable_under_low_margin(
        [
            ("PACKING_HOST_MATRIX_CONFINEMENT", "RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA"),
            ("RADIATIVE_RATE_STATE_BALANCE", "SOKR_ANTI_KASHA", "PACKING_HOST_MATRIX_CONFINEMENT"),
            ("RADIATIVE_RATE_STATE_BALANCE", "PACKING_HOST_MATRIX_CONFINEMENT", "SOKR_ANTI_KASHA"),
        ]
    )


def test_default_deps_do_not_cap_scientific_finalization_timeouts(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MECHCAL_OPENAI_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MECHCAL_OPENAI_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("MECHCAL_OPENAI_TIMEOUT", "600")
    monkeypatch.setenv("MECHCAL_OPENAI_MAX_RETRIES", "0")

    deps = default_deps(OrchestratorConfig(run_base_dir=tmp_path, max_rounds=1))

    assert deps.photophysics_arbiter.llm_client.settings.timeout_seconds == 600.0
    assert deps.mechanism_critic.llm_client.settings.timeout_seconds == 600.0
    assert deps.conclusion_ledger.llm_client.settings.timeout_seconds == 600.0
    assert deps.photophysics_arbiter.llm_client.settings.max_retries == 0
    assert deps.conclusion_ledger.llm_client.settings.max_retries == 0


def test_runtime_failure_preserves_persisted_finalization_state(tmp_path) -> None:
    class PersistedReviewArbiter:
        def review(self, case_run: CaseRun) -> PhotophysicsReview:
            round_id = case_run.round_ids[-1] if case_run.round_ids else "R001"
            return PhotophysicsReview(
                review_id=f"{case_run.case_id}:{round_id}:photophysics_review",
                case_id=case_run.case_id,
                round_id=round_id,
                overclaim_warnings=["test review persisted before ledger failure"],
            )

    class FailingOutputAssembler:
        def build_mechanism_predictions(self, case_run: CaseRun) -> list[object]:
            raise RuntimeError("output assembly failed after review")

    case_id = "failed_after_review"
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=2)
    deps = _fake_llm_deps(config)
    deps.photophysics_arbiter = PersistedReviewArbiter()  # type: ignore[assignment]
    deps.output_assembler = FailingOutputAssembler()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="output assembly failed after review"):
        MechCALOrchestrator(config, deps=deps).run(
            smiles="C1=CC=CC=C1",
            user_query="Assess the likely AIE mechanism for this molecule.",
            case_id=case_id,
        )

    persisted = json.loads((tmp_path / case_id / "case_run.json").read_text())
    assert persisted["status"] == "failed"
    assert persisted["photophysics_review"] is not None
    assert persisted["runtime"]["failure"]["message"] == "output assembly failed after review"
    assert (tmp_path / case_id / "photophysics_review.json").is_file()


def test_orchestrator_can_continue_after_transient_arbiter_failure_without_fallback(
    tmp_path,
) -> None:
    case_id = "transient_arbiter_failure"
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=2,
        continue_on_arbiter_failure=True,
    )
    deps = _fake_llm_deps(config)
    deps.photophysics_arbiter = PhotophysicsArbiterAgent(
        llm_client=FailingArbiterLLM(),  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
        max_attempts=1,
    )

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_042",
    )

    assert result.status == "stopped"
    assert result.photophysics_review is None
    assert result.runtime["verifier_failures"]

    run_dir = tmp_path / case_id
    verifier_failures = run_dir / "history" / "verifier_failures.jsonl"
    runtime_failures = run_dir / "history" / "runtime_failures.jsonl"
    assert verifier_failures.is_file()
    assert not runtime_failures.exists()

    persisted = json.loads((run_dir / "case_run.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "stopped"
    assert persisted["runtime"]["verifier_failures"][0]["policy"] == (
        "continue_without_scientific_fallback"
    )
    assert not (run_dir / "photophysics_review.json").exists()


def test_orchestrator_skips_repeated_verifier_calls_after_evidence_sufficiency(
    tmp_path,
) -> None:
    class FinalizesAfterFollowupEvidenceClient(FakeJsonClient):
        def complete_json(self, **kwargs) -> dict[str, object]:  # type: ignore[no-untyped-def]
            system_prompt = kwargs["system_prompt"]
            payload = kwargs["payload"]
            if (
                "PlannerDecision" in system_prompt
                and isinstance(payload, dict)
                and payload.get("output_contract")
                and payload.get("round_id") == "R003"
            ):
                evidence = payload.get("new_round_evidence")
                evidence_refs = [
                    str(item.get("evidence_id"))
                    for item in evidence
                    if isinstance(item, dict) and item.get("evidence_id")
                ] if isinstance(evidence, list) else []
                followup_refs = [
                    ref
                    for ref in evidence_refs
                    if "run_bright_dark_state_ordering" in ref
                    or "run_torsion_brightness_coupling_scan" in ref
                    or "run_solid_state_emission_proxy" in ref
                ]
                assert followup_refs
                return {
                    "decision_id": "R003:planner_final_after_followup",
                    "round_id": "R003",
                    "portfolio": {
                        "current": "RIM_RIR_RIV",
                        "runner_up": "PACKING_HOST_MATRIX_CONFINEMENT",
                        "hypotheses": [
                            {
                                "name": "RIM_RIR_RIV",
                                "confidence": 0.42,
                                "differential_priority": 0.42,
                                "evidence_support": "weak_or_proxy",
                                "claim_status": "candidate_requires_validation",
                                "status": "pending",
                                "rationale": (
                                    "Planner absorbed follow-up route evidence before "
                                    "finalizing."
                                ),
                                "evidence_refs": followup_refs[:1],
                                "validation_needed": [
                                    "Aggregate-state validation remains missing."
                                ],
                            },
                            {
                                "name": "PACKING_HOST_MATRIX_CONFINEMENT",
                                "confidence": 0.30,
                                "differential_priority": 0.30,
                                "evidence_support": "weak_or_proxy",
                                "claim_status": "candidate_requires_validation",
                                "status": "pending",
                                "rationale": "Solid-state proxy remains a candidate.",
                                "evidence_refs": followup_refs[-1:],
                                "validation_needed": [
                                    "Packing or aggregate morphology validation."
                                ],
                            },
                        ],
                    },
                    "current_hypothesis": "RIM_RIR_RIV",
                    "runner_up_hypothesis": "PACKING_HOST_MATRIX_CONFINEMENT",
                    "confidence": 0.42,
                    "diagnosis": "Finalize only after Planner reads follow-up evidence.",
                    "action": "finalize",
                    "dispatch_requests": [],
                    "unresolved_gaps": [
                        "Aggregate-state measurement remains unavailable."
                    ],
                    "priority_deltas": [
                        {
                            "label": "RIM_RIR_RIV",
                            "previous_priority": 0.18,
                            "priority_delta": 0.24,
                            "new_priority": 0.42,
                            "delta_reason": (
                                "Follow-up route evidence was read by Planner before "
                                "finalization."
                            ),
                            "evidence_refs_added": followup_refs[:1],
                            "support_change": "upgraded",
                        }
                    ],
                    "mechanism_support_arguments": [
                        {
                            "support_id": "R003:support:rim",
                            "target_label": "RIM_RIR_RIV",
                            "support_level": "weak",
                            "finding": "Follow-up route evidence was absorbed.",
                            "warrant": (
                                "Discriminating route evidence is required before "
                                "early evidence-sufficiency finalization."
                            ),
                            "boundary": "This is proxy support, not wet-lab confirmation.",
                            "observation_refs": followup_refs[:1],
                        }
                    ],
                    "final_answer_draft": (
                        "Runtime evidence supports a bounded RIM/RIR proxy mechanism "
                        "after follow-up evidence was incorporated."
                    ),
                    "raw_response": {"prompt_name": "planner_decision.md"},
                }
            return super().complete_json(**kwargs)

    case_id = "repeated_verifier_failures"
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=5,
        continue_on_arbiter_failure=True,
        verifier_failure_stop_threshold=3,
    )
    deps = _fake_llm_deps(config)
    deps.planner = PlannerAgent(
        capability_registry=deps.capability_registry,
        llm_client=FinalizesAfterFollowupEvidenceClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )
    deps.photophysics_arbiter = PhotophysicsArbiterAgent(
        llm_client=FailingArbiterLLM(),  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
        max_attempts=1,
    )

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_043",
    )

    assert result.status == "finalized"
    assert result.round_ids == ["R001", "R002", "R003"]
    assert len(result.runtime["verifier_failures"]) == 2
    assert result.final_answer is not None
    assert result.mechanism_predictions

    run_dir = tmp_path / case_id
    assert (run_dir / "history" / "verifier_failures.jsonl").is_file()
    assert not (run_dir / "history" / "runtime_failures.jsonl").exists()
    assert (run_dir / "final.json").is_file()


class FailsAfterFirstCriticPortfolio:
    def __init__(self, base: MechanismCriticAgent) -> None:
        self.base = base
        self.calls = 0

    def review(self, case_run: CaseRun) -> DifferentialMechanismPortfolio:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("APITimeoutError: Request timed out.")
        return self.base.review(case_run)


class MalformedAfterFirstCriticPortfolio:
    def __init__(self, base: MechanismCriticAgent) -> None:
        self.base = base
        self.calls = 0

    def review(self, case_run: CaseRun) -> DifferentialMechanismPortfolio:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError(
                "MechanismCriticAgent failed to produce a valid portfolio: "
                "Attempt 1 failed: ValueError: MechanismCritic rows must cover "
                "mechanism_pool exactly; missing=['RIM_RIR_RIV'], extra=[]"
            )
        return self.base.review(case_run)


def test_orchestrator_reuses_previous_portfolio_after_transient_critic_failure(
    tmp_path,
) -> None:
    case_id = "transient_mechanism_critic_failure"
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=4)
    deps = _fake_llm_deps(config)
    critic = FailsAfterFirstCriticPortfolio(deps.mechanism_critic)
    deps.mechanism_critic = critic  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_044",
    )

    assert result.status == "finalized"
    assert result.differential_mechanism_portfolio is not None
    assert result.mechanism_predictions
    assert critic.calls >= 2

    run_dir = tmp_path / case_id
    assert (run_dir / "history" / "mechanism_critic_failures.jsonl").is_file()
    assert not (run_dir / "history" / "runtime_failures.jsonl").exists()

    persisted = json.loads((run_dir / "case_run.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "finalized"
    assert persisted["mechanism_predictions"]


def test_orchestrator_reuses_previous_portfolio_after_malformed_critic_output(
    tmp_path,
) -> None:
    case_id = "malformed_mechanism_critic_output"
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=4)
    deps = _fake_llm_deps(config)
    critic = MalformedAfterFirstCriticPortfolio(deps.mechanism_critic)
    deps.mechanism_critic = critic  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_046",
    )

    assert result.status == "finalized"
    assert result.differential_mechanism_portfolio is not None
    assert result.mechanism_predictions
    assert critic.calls >= 2

    run_dir = tmp_path / case_id
    assert (run_dir / "history" / "mechanism_critic_failures.jsonl").is_file()
    assert not (run_dir / "history" / "runtime_failures.jsonl").exists()


class FailsAfterInitialCoverage:
    def __init__(self, base: MechanismAgendaAgent) -> None:
        self.base = base
        self.calls: list[str] = []

    def review_coverage(
        self,
        case_run: CaseRun,
        *,
        round_id: str,
        previous_planner_action: str | None = None,
    ) -> MechanismAgendaCoverage:
        self.calls.append(round_id)
        if round_id != "R001":
            raise RuntimeError("APITimeoutError: Request timed out.")
        return self.base.review_coverage(
            case_run,
            round_id=round_id,
            previous_planner_action=previous_planner_action,
        )

    def propose(self, case_run: CaseRun, *, round_id: str) -> MechanismProgram:
        return self.base.propose(case_run, round_id=round_id)


def test_orchestrator_continues_after_transient_agenda_coverage_failure(
    tmp_path,
) -> None:
    case_id = "transient_agenda_coverage_failure"
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=3)
    deps = _fake_llm_deps(config)
    deps.mechanism_agenda = FailsAfterInitialCoverage(  # type: ignore[assignment]
        deps.mechanism_agenda
    )
    orchestrator = MechCALOrchestrator(config, deps=deps)
    store = RunStore(tmp_path, case_id)
    store.initialize()
    case_run = _case_run_for_gate().touch(
        case_id=case_id,
        input=CaseInput(
            case_id=case_id,
            smiles="C1=CC=CC=C1",
            user_query="Assess mechanism.",
        ),
        current_round_id="R002",
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id=f"{case_id}:R001:agenda_coverage_bootstrap",
            case_id=case_id,
            round_id="R001",
            coverage_summary="Initial coverage exists.",
            agenda_items=[
                AgendaCoverageItem(
                    label="RIM_RIR_RIV",
                    coverage_status="needs_follow_up",
                    why_relevant="Existing public coverage is incomplete.",
                    missing_evidence="Follow-up evidence is still missing.",
                    suggested_question="Collect bounded public evidence.",
                    suggested_route="macro.collect_structure_evidence",
                    urgency="medium",
                    boundary="No private answer key is available.",
                )
            ],
        ),
    )

    result = orchestrator._review_agenda_coverage(
        case_run,
        store,
        round_id="R002",
    )

    assert result.mechanism_agenda_coverage is not None
    assert result.runtime["agenda_coverage_failures"][0]["policy"] == (
        "continue_with_previous_agenda_coverage_without_scientific_fallback"
    )

    run_dir = tmp_path / case_id
    assert (run_dir / "history" / "agenda_coverage_failures.jsonl").is_file()
    assert not (run_dir / "history" / "runtime_failures.jsonl").exists()
    persisted = json.loads((run_dir / "case_run.json").read_text(encoding="utf-8"))
    assert persisted["runtime"]["agenda_coverage_failures"][0]["agent"] == (
        "MechanismAgendaAgent"
    )


class PlannerFailsOnRound:
    def __init__(self, base: PlannerAgent, *, fail_round_id: str) -> None:
        self.base = base
        self.fail_round_id = fail_round_id

    def plan_initial(self, case_run: CaseRun, *, round_id: str) -> PlannerDecision:
        if round_id == self.fail_round_id:
            raise RuntimeError("APITimeoutError: Request timed out.")
        return self.base.plan_initial(case_run, round_id=round_id)

    def plan_update(
        self,
        context,
        case_run: CaseRun,
        *,
        round_id: str,
    ) -> PlannerDecision:
        if round_id == self.fail_round_id:
            raise RuntimeError("APITimeoutError: Request timed out.")
        return self.base.plan_update(context, case_run, round_id=round_id)

    def build_final_answer(self, decision, case_run: CaseRun) -> FinalAnswer:
        return self.base.build_final_answer(decision, case_run)


def test_orchestrator_stops_after_transient_planner_update_without_fallback(
    tmp_path,
) -> None:
    case_id = "transient_planner_update_failure"
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=3,
        continue_on_planner_update_failure=True,
    )
    deps = _fake_llm_deps(config)
    deps.planner = PlannerFailsOnRound(  # type: ignore[assignment]
        deps.planner,
        fail_round_id="R002",
    )

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_007",
    )

    assert result.status == "stopped"
    assert result.round_ids == ["R001"]
    assert result.runtime["planner_failures"][0]["attempted_round_id"] == "R002"
    assert result.runtime["planner_failures"][0]["policy"] == (
        "stop_without_scientific_fallback"
    )
    assert result.final_answer is not None
    assert "no synthetic Planner decision" in result.final_answer.answer
    assert result.mechanism_predictions == []

    run_dir = tmp_path / case_id
    assert (run_dir / "history" / "planner_failures.jsonl").is_file()
    assert not (run_dir / "history" / "runtime_failures.jsonl").exists()
    assert not (run_dir / "rounds" / "R002" / "planner_decision.json").exists()
    assert (run_dir / "final.json").is_file()


def test_initial_planner_timeout_still_fails_without_fallback(tmp_path) -> None:
    case_id = "initial_planner_timeout"
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=3,
        continue_on_planner_update_failure=True,
    )
    deps = _fake_llm_deps(config)
    deps.planner = PlannerFailsOnRound(  # type: ignore[assignment]
        deps.planner,
        fail_round_id="R001",
    )

    with pytest.raises(RuntimeError, match="APITimeoutError"):
        MechCALOrchestrator(config, deps=deps).run(
            smiles="C1=CC=CC=C1",
            user_query="Assess the likely AIE mechanism for this molecule.",
            case_id=case_id,
            public_case_id="CASE_008",
        )

    run_dir = tmp_path / case_id
    persisted = json.loads((run_dir / "case_run.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert "planner_failures" not in persisted["runtime"]
    assert (run_dir / "history" / "runtime_failures.jsonl").is_file()
    assert not (run_dir / "history" / "planner_failures.jsonl").exists()


def test_stop_without_terminal_decision_does_not_repeat_final_arbiter_audit(
    tmp_path,
) -> None:
    class CountingArbiter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def review(self, case_run: CaseRun) -> PhotophysicsReview:
            round_id = case_run.current_round_id or "R001"
            self.calls.append(round_id)
            return PhotophysicsReview(
                review_id=f"CASE_001:{round_id}:photophysics_review",
                case_id="CASE_001",
                round_id=round_id,
                overclaim_warnings=["counting arbiter review"],
            )

    class FailingCalibrationPlanner:
        def __init__(self, base) -> None:
            self.base = base

        def plan_initial(self, case_run: CaseRun, *, round_id: str) -> PlannerDecision:
            return self.base.plan_initial(case_run, round_id=round_id)

        def plan_update(
            self,
            context,
            case_run: CaseRun,
            *,
            round_id: str,
        ) -> PlannerDecision:
            return self.base.plan_update(context, case_run, round_id=round_id)

        def calibrate_final_portfolio(self, case_run: CaseRun, *, round_id: str):
            del case_run, round_id
            raise AssertionError("max-round stop must not recalibrate")

        def build_final_answer(self, decision, case_run: CaseRun) -> FinalAnswer:
            return self.base.build_final_answer(decision, case_run)

    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=2)
    deps = _fake_llm_deps(config)
    arbiter = CountingArbiter()
    deps.photophysics_arbiter = arbiter  # type: ignore[assignment]
    deps.planner = FailingCalibrationPlanner(deps.planner)  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id="no_duplicate_final_audit",
        public_case_id="CASE_001",
    )

    assert result.status == "stopped"
    assert arbiter.calls == ["R001"]


def test_support_auditor_can_be_disabled_for_ablation(tmp_path) -> None:
    class CountingArbiter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def review(self, case_run: CaseRun) -> PhotophysicsReview:
            self.calls.append(case_run.current_round_id or "unknown")
            raise AssertionError("support auditor should be disabled")

    case_id = "support_auditor_disabled_ablation"
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=2,
        enable_support_auditor=False,
    )
    deps = _fake_llm_deps(config)
    arbiter = CountingArbiter()
    deps.photophysics_arbiter = arbiter  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_001",
    )

    assert result.status in {"finalized", "stopped"}
    assert arbiter.calls == []
    assert result.photophysics_review is None
    assert result.runtime["support_auditor"]["enabled"] is False
    assert result.runtime["support_auditor_skipped_rounds"]

    run_dir = tmp_path / case_id
    assert (run_dir / "history" / "support_auditor_skipped.jsonl").is_file()
    assert not (run_dir / "photophysics_review.json").exists()
    persisted = json.loads((run_dir / "case_run.json").read_text(encoding="utf-8"))
    assert persisted["runtime"]["support_auditor"]["enabled"] is False


def test_agenda_coverage_reviewer_can_be_disabled_for_ablation(tmp_path) -> None:
    class FailingAgendaReviewer:
        def review_coverage(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("agenda reviewer should be disabled")

    case_id = "agenda_coverage_reviewer_disabled_ablation"
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=2,
        enable_agenda_coverage_reviewer=False,
    )
    deps = _fake_llm_deps(config)
    deps.mechanism_agenda = FailingAgendaReviewer()  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_001",
    )

    assert result.status in {"finalized", "stopped"}
    assert result.mechanism_agenda_coverage is None
    assert result.runtime["agenda_coverage_reviewer"]["enabled"] is False
    run_dir = tmp_path / case_id
    assert not (run_dir / "mechanism_agenda_coverage.json").exists()
    persisted = json.loads((run_dir / "case_run.json").read_text(encoding="utf-8"))
    assert persisted["runtime"]["agenda_coverage_reviewer"]["enabled"] is False


def test_incremental_portfolio_can_be_disabled_for_ablation(tmp_path) -> None:
    class InspectingPlanner:
        def __init__(self, base) -> None:
            self.base = base
            self.update_views: list[CaseRun] = []

        def plan_initial(self, case_run: CaseRun, *, round_id: str) -> PlannerDecision:
            return self.base.plan_initial(case_run, round_id=round_id)

        def plan_update(
            self,
            context,
            case_run: CaseRun,
            *,
            round_id: str,
        ) -> PlannerDecision:
            self.update_views.append(case_run)
            return self.base.plan_update(context, case_run, round_id=round_id)

        def calibrate_final_portfolio(self, case_run: CaseRun, *, round_id: str):
            self.update_views.append(case_run)
            return self.base.calibrate_final_portfolio(case_run, round_id=round_id)

        def build_final_answer(self, decision, case_run: CaseRun) -> FinalAnswer:
            return self.base.build_final_answer(decision, case_run)

    case_id = "incremental_portfolio_disabled_ablation"
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=3,
        enable_incremental_mechanism_portfolio=False,
    )
    deps = _fake_llm_deps(config)
    planner = InspectingPlanner(deps.planner)
    deps.planner = planner  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id=case_id,
        public_case_id="CASE_001",
    )

    assert result.status in {"finalized", "stopped"}
    assert planner.update_views
    assert all(view.portfolio.current == "unknown" for view in planner.update_views)
    assert any(view.mechanism_support_arguments for view in planner.update_views)
    assert any(view.photophysics_review is not None for view in planner.update_views)
    assert result.runtime["incremental_mechanism_portfolio"]["enabled"] is False


def test_parallel_worker_summary_runs_each_worker_once_without_control_agents(
    tmp_path,
) -> None:
    class FailingPlanner:
        def plan_initial(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("parallel summary must not call Planner")

        def plan_update(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("parallel summary must not call Planner")

        def calibrate_final_portfolio(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("parallel summary must not call Planner")

    class FailingReviewer:
        def review_coverage(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("parallel summary must not call Coverage Reviewer")

    class FailingAuditor:
        def review(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("parallel summary must not call Support Auditor")

    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=30,
        ablation_mode="parallel_worker_summary",
    )
    deps = _fake_llm_deps(config)
    deps.planner = FailingPlanner()  # type: ignore[assignment]
    deps.mechanism_agenda = FailingReviewer()  # type: ignore[assignment]
    deps.photophysics_arbiter = FailingAuditor()  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="c1ccccc1C=Cc2ccccc2",
        user_query="Predict likely AIE mechanisms.",
        case_id="parallel_worker_summary",
        public_case_id="CASE_001",
    )

    assert result.status == "finalized"
    assert result.round_ids == ["R001"]
    assert result.runtime["worker_call_counts"] == {
        "macro": 1,
        "microscopic": 1,
    }
    assert result.runtime["termination"]["reason"] == (
        "parallel_worker_summary_complete"
    )
    assert result.mechanism_agenda_coverage is None
    assert result.photophysics_review is None
    assert result.mechanism_predictions
    run_dir = tmp_path / "parallel_worker_summary"
    assert not (run_dir / "history" / "planner_messages.jsonl").exists()
    assert not (run_dir / "mechanism_agenda_coverage.json").exists()
    assert not (run_dir / "photophysics_review.json").exists()
    synthesis_payloads = [
        payload
        for payload in deps.mechanism_critic.llm_client.payloads  # type: ignore[union-attr]
        if payload.get("task") == "parallel_worker_summary_one_shot_synthesis"
    ]
    assert len(synthesis_payloads) == 1
    payload_text = json.dumps(synthesis_payloads[0], ensure_ascii=False)
    assert "hidden_reference" not in payload_text
    assert "reference_mechanisms" not in payload_text
    assert "planner_state" not in synthesis_payloads[0]


def test_parallel_worker_summary_retries_invalid_synthesis_with_schema_feedback(
    tmp_path,
) -> None:
    class RetryThenSucceedClient:
        def __init__(self) -> None:
            self.calls = 0
            self.feedbacks: list[str | None] = []

        def is_configured(self) -> bool:
            return True

        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt
            self.calls += 1
            self.feedbacks.append(schema_feedback)
            if self.calls == 1:
                raise ValueError("LLM response contained an unterminated JSON object.")
            return {
                "rows": [
                    {
                        "label": "RIM_RIR_RIV",
                        "trigger_status": "weak_trigger",
                        "differential_priority": 0.4,
                        "support_strength": "weak_proxy",
                        "claim_status": "candidate_requires_validation",
                        "positive_evidence_refs": [],
                        "negative_evidence_refs": [],
                        "missing_validation": ["restriction evidence"],
                        "rationale": "Flexible structure provides only a motion proxy.",
                    }
                ],
                "evidence_attributions": [],
                "policy_notes": [],
            }

    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=30,
        ablation_mode="parallel_worker_summary",
    )
    deps = _fake_llm_deps(config)
    client = RetryThenSucceedClient()
    deps.mechanism_critic = MechanismCriticAgent(
        llm_client=client,  # type: ignore[arg-type]
        max_attempts=3,
    )

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="c1ccccc1C=Cc2ccccc2",
        user_query="Predict likely AIE mechanisms.",
        case_id="parallel_worker_summary_retry",
        public_case_id="CASE_001",
    )

    assert result.status == "finalized"
    assert client.calls == 2
    assert client.feedbacks[0] is None
    assert client.feedbacks[1] is not None
    assert "unterminated JSON object" in client.feedbacks[1]


def test_one_pass_mechcal_limits_each_selected_worker_to_one_call(tmp_path) -> None:
    class OnePassPlanner:
        def __init__(self, base, registry) -> None:  # noqa: ANN001
            self.base = base
            self.registry = registry
            self.calibration_calls = 0

        def plan_initial(self, case_run: CaseRun, *, round_id: str) -> PlannerDecision:
            capability_ids = [
                "macro.macro_structure_scan",
                "macro.screen_rotor_torsion_topology",
                "microscopic.run_baseline_bundle",
                "microscopic.run_bright_dark_state_ordering",
            ]
            dispatches = []
            for capability_id in capability_ids:
                capability = self.registry.require(capability_id)
                dispatches.append(
                    DispatchRequest(
                        dispatch_id=f"{round_id}:{capability_id}",
                        round_id=round_id,
                        agent_name=capability.owner_agent,
                        capability_id=capability.capability_id,
                        task="Collect one bounded observation.",
                        objective="Collect one bounded observation.",
                        evidence_goal_family=capability.evidence_family,
                        route=capability.route,
                    )
                )
            return PlannerDecision(
                decision_id=f"{round_id}:one_pass_initial",
                round_id=round_id,
                portfolio=case_run.portfolio,
                current_hypothesis="unknown",
                diagnosis="Select targeted first-round worker routes.",
                action="dispatch",
                dispatch_requests=dispatches,
                rationale="Initial targeted delegation only.",
            )

        def calibrate_final_portfolio(
            self,
            case_run: CaseRun,
            *,
            round_id: str,
        ) -> PlannerDecision:
            self.calibration_calls += 1
            refs = [item.evidence_id for item in case_run.evidence_ledger.items]
            portfolio = HypothesisPortfolio(
                current="RIM_RIR_RIV",
                runner_up="RADIATIVE_RATE_STATE_BALANCE",
                hypotheses=[
                    HypothesisEntry(
                        name="RIM_RIR_RIV",
                        confidence=0.55,
                        differential_priority=0.55,
                        evidence_support="weak_or_proxy",
                        claim_status="candidate_requires_validation",
                        evidence_refs=refs[:1],
                    ),
                    HypothesisEntry(
                        name="RADIATIVE_RATE_STATE_BALANCE",
                        confidence=0.45,
                        differential_priority=0.45,
                        evidence_support="weak_or_proxy",
                        claim_status="candidate_requires_validation",
                        evidence_refs=refs[1:2],
                    ),
                ],
            )
            return PlannerDecision(
                decision_id=f"{round_id}:one_pass_final",
                round_id=round_id,
                portfolio=portfolio,
                current_hypothesis=portfolio.current,
                runner_up_hypothesis=portfolio.runner_up,
                confidence=0.55,
                diagnosis="Single synthesis from first-round evidence.",
                action="finalize",
                final_answer_draft="One-pass evidence synthesis completed.",
                rationale="No feedback loop or revisit is permitted.",
            )

        def build_final_answer(
            self,
            decision: PlannerDecision,
            case_run: CaseRun,
        ) -> FinalAnswer:
            return self.base.build_final_answer(decision, case_run)

    class FailingReviewer:
        def review_coverage(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("one-pass mode must not call Coverage Reviewer")

    class FailingAuditor:
        def review(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("one-pass mode must not call Support Auditor")

    class FailingCritic:
        def review(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("one-pass mode must not call MechanismCritic")

    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=30,
        ablation_mode="one_pass_mechcal",
        enable_incremental_mechanism_portfolio=False,
    )
    deps = _fake_llm_deps(config)
    planner = OnePassPlanner(deps.planner, deps.capability_registry)
    deps.planner = planner  # type: ignore[assignment]
    deps.mechanism_agenda = FailingReviewer()  # type: ignore[assignment]
    deps.photophysics_arbiter = FailingAuditor()  # type: ignore[assignment]
    deps.mechanism_critic = FailingCritic()  # type: ignore[assignment]

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="c1ccccc1C=Cc2ccccc2",
        user_query="Predict likely AIE mechanisms.",
        case_id="one_pass_mechcal",
        public_case_id="CASE_001",
    )

    assert result.status == "finalized"
    assert result.round_ids == ["R001"]
    assert result.runtime["worker_call_counts"] == {
        "macro": 1,
        "microscopic": 1,
    }
    assert result.runtime["termination"]["reason"] == "one_pass_complete"
    assert planner.calibration_calls == 1
    assert result.mechanism_agenda_coverage is None
    assert result.photophysics_review is None
    assert result.mechanism_predictions


@pytest.mark.parametrize(
    ("mode", "enabled_owner", "disabled_owner"),
    [
        ("mechcal_wo_macro", "microscopic", "macro"),
        ("mechcal_wo_microscopic", "macro", "microscopic"),
    ],
)
def test_worker_ablation_removes_owner_from_registry_and_runtime(
    tmp_path,
    mode: str,
    enabled_owner: str,
    disabled_owner: str,
) -> None:
    config = OrchestratorConfig(
        run_base_dir=tmp_path,
        max_rounds=2,
        ablation_mode=mode,  # type: ignore[arg-type]
    )
    deps = _fake_llm_deps(config)

    result = MechCALOrchestrator(config, deps=deps).run(
        smiles="c1ccccc1C=Cc2ccccc2",
        user_query="Predict likely AIE mechanisms.",
        case_id=mode,
        public_case_id="CASE_001",
    )

    owners = {str(item["owner_agent"]) for item in deps.capability_registry.cards()}
    assert owners == {enabled_owner}
    assert set(deps.workers) == {enabled_owner}
    assert result.runtime["ablation"]["disabled_capability_owners"] == [
        disabled_owner
    ]
    assert result.evidence_ledger.items
    assert {
        item.agent_name for item in result.evidence_ledger.items
    } == {enabled_owner}
    raw_text = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    assert "hidden_reference" not in raw_text
    assert "reference_mechanisms" not in raw_text


def test_photophysics_arbiter_review_feeds_planner_diagnosis(tmp_path) -> None:
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=5)
    result = MechCALOrchestrator(
        config,
        deps=_fake_llm_deps(config),
    ).run(
        smiles="c1ccccc1C=Cc2ccccc2",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id="neutral_aromatic_alkene",
    )

    assert result.photophysics_review is not None
    assert result.photophysics_review.source_evidence_refs
    assert result.mechanism_program is not None
    assert result.mechanism_program.scope_boundaries
    diagnoses = build_diagnosis_units(result)
    assert diagnoses
    assert diagnoses[0].context == "Planner final synthesis"
    assert not any(item.agent_name == "planner" for item in result.evidence_ledger.items)
    assert (tmp_path / "neutral_aromatic_alkene" / "photophysics_review.json").is_file()

    persisted = json.loads(
        (tmp_path / "neutral_aromatic_alkene" / "case_run.json").read_text(
            encoding="utf-8"
        )
    )
    raw_text = json.dumps(persisted, ensure_ascii=False).lower()
    assert "dpe" not in raw_text
    assert "stilbene" not in raw_text


class FakeArbiterLLM:
    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        assert "SupportAuditor" in system_prompt
        payload_text = json.dumps(payload)
        assert "hidden_reference" not in payload_text
        assert "gate_case" not in payload_text
        assert payload["public_case"]["case_label"] == "current_smiles_case"  # type: ignore[index]
        assert "output_contract" in payload
        assert "output_schema" not in payload
        assert schema_feedback is None
        return {
            "review_id": "llm_supplied_id",
            "case_id": "llm_supplied_case",
            "round_id": "llm_supplied_round",
            "coverage_axes": [
                {
                    "axis_id": "rim_rotor",
                    "label": "RIM/torsion check",
                    "status": "covered",
                    "evidence_refs": ["E1", "not_allowed"],
                    "rationale": "Typed evidence supports torsion review.",
                    "recommended_routes": ["microscopic.run_torsion_snapshots"],
                }
            ],
            "hypothesis_cards": [
                {
                    "hypothesis_id": "rim_torsion",
                    "mechanism": "Restriction of intramolecular motion",
                    "status": "proxy_supported",
                    "support_evidence_refs": ["E1", "not_allowed"],
                    "weakening_evidence_refs": [],
                    "missing_or_unresolved": ["Needs aggregate PL."],
                    "reasoning_summary": "RIM is supported only at proxy level.",
                    "scope_limits": ["No wet-lab claim."],
                    "priority": 0.8,
                }
            ],
            "recommended_next_routes": [
                "microscopic.run_torsion_snapshots",
                "not.a.capability",
            ],
            "overclaim_warnings": ["Do not overclaim PL."],
            "source_evidence_refs": ["E1"],
        }


class FakeGenericSearch(GenericPhotophysicsWebSearch):
    def search(self, case_run: CaseRun) -> list[dict[str, str]]:
        return [
            {
                "query": "aggregation-induced emission photophysics review",
                "title": "Generic AIE review",
                "url": "https://example.test/review",
                "snippet": "Generic mechanism background.",
            }
        ]


class FailingArbiterLLM:
    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        raise TimeoutError("simulated arbiter timeout")


class SchemaDriftArbiterLLM:
    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        return {
            "review_id": "llm_supplied_id",
            "case_id": "llm_supplied_case",
            "round_id": "llm_supplied_round",
            "coverage_axes": {
                "axis_id": "rim_rotor",
                "label": "RIM/torsion check",
                "status": "underdetermined",
                "evidence_refs": "E1",
                "rationale": "Typed evidence supports torsion review.",
                "recommended_routes": "microscopic.run_torsion_snapshots",
                "and": "stray prose key from an LLM JSON response",
            },
            "hypothesis_cards": [
                {
                    "hypothesis_id ": "rim_torsion",
                    "mechanism": "Restriction of intramolecular motion",
                    "status": "proxy_supported",
                    "support_evidence_refs": "E1",
                    "weakening_evidence_refs": "",
                    "missing_or_unresolved": "Needs aggregate PL.",
                    "reasoning_summary": "RIM is supported only at proxy level.",
                    "scope_limits": "No wet-lab claim.",
                    "priority": "high",
                }
            ],
            "recommended_next_routes": "microscopic.run_torsion_snapshots",
            "overclaim_warnings": "Do not overclaim PL.",
            "source_evidence_refs": "E1",
            "overclaim_evidence_refs": [],
            "policy": {
                "case_specific_literature_search": "forbidden",
                "final_decision_authority": "Planner only",
            },
        }


class MissingAxisRationaleArbiterLLM:
    def __init__(self) -> None:
        self.calls = 0

    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        del system_prompt, payload, schema_feedback
        self.calls += 1
        return {
            "review_id": "llm_supplied_id",
            "case_id": "llm_supplied_case",
            "round_id": "llm_supplied_round",
            "coverage_axes": [
                {
                    "axis_id": "rim_rotor",
                    "label": "RIM/torsion check",
                    "status": "covered",
                    "evidence_refs": ["E1"],
                    "recommended_routes": [],
                }
            ],
            "hypothesis_cards": [],
            "recommended_next_routes": [],
            "overclaim_warnings": [],
            "source_evidence_refs": ["E1"],
        }


class RetryArbiterLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.feedbacks: list[str | None] = []

    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        del system_prompt, payload
        self.calls += 1
        self.feedbacks.append(schema_feedback)
        if self.calls == 1:
            return {
                "review_id": "retry_review",
                "case_id": "gate_case",
                "round_id": "R001",
                "coverage_axes": [
                    {
                        "axis_id": "esipt_axis",
                        "label": "ESIPT check",
                        "status": "covered",
                        "evidence_refs": ["E1"],
                        "rationale": "Evidence exists.",
                        "recommended_routes": [],
                    }
                ],
                "hypothesis_cards": [
                    {
                        "hypothesis_id": "",
                        "mechanism": "ESIPT",
                        "status": "rejected",
                        "support_evidence_refs": [],
                        "weakening_evidence_refs": [],
                        "missing_or_unresolved": [],
                        "reasoning_summary": "Invalid unsupported rejection.",
                        "scope_limits": [],
                        "priority": 0.1,
                    }
                ],
                "recommended_next_routes": [],
                "overclaim_warnings": [],
                "source_evidence_refs": ["E1"],
            }
        return {
            "review_id": "retry_review",
            "case_id": "gate_case",
            "round_id": "R001",
            "coverage_axes": [
                {
                    "axis_id": "esipt_axis",
                    "label": "ESIPT check",
                    "status": "missing",
                    "evidence_refs": ["E1"],
                    "rationale": "Evidence is insufficient for rejection.",
                    "recommended_routes": [],
                }
            ],
            "hypothesis_cards": [
                {
                    "hypothesis_id": "esipt_reject",
                    "mechanism": "ESIPT",
                    "status": "underdetermined",
                    "support_evidence_refs": [],
                    "weakening_evidence_refs": [],
                    "missing_or_unresolved": ["Need direct ESIPT motif evidence."],
                    "reasoning_summary": "No supplied evidence supports rejection.",
                    "scope_limits": [],
                    "priority": 0.1,
                }
            ],
            "recommended_next_routes": [],
            "overclaim_warnings": [],
            "source_evidence_refs": ["E1"],
        }


def test_photophysics_arbiter_llm_review_sanitizes_refs_and_routes() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        input=CaseInput(
            case_id="gate_case",
            smiles="C1=CC=CC=C1",
            user_query="Assess mechanism.",
            metadata={"public_case_id": "CASE_007"},
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    observable_tags=["macro_structure_scan_proxy"],
                    summary="Macro evidence.",
                )
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R001:support:001",
                target_label="RIM_RIR_RIV",
                support_level="partial",
                finding="Runtime evidence identifies rotatable aryl units.",
                warrant=(
                    "This partially supports RIM/RIR/RIV because rotatable "
                    "units can open intramolecular-motion loss channels."
                ),
                boundary=(
                    "This is proxy support; aggregate-state restriction was "
                    "not measured."
                ),
                observation_refs=["E1"],
            )
        ],
    )
    arbiter = PhotophysicsArbiterAgent(
        llm_client=FakeArbiterLLM(),  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
    )

    review = arbiter.review(case_run)

    assert review.case_id == "CASE_007"
    assert review.review_id == "CASE_007:R001:photophysics_review"
    assert review.round_id == "R001"
    assert review.coverage_axes[0].evidence_refs == ["E1"]
    assert review.hypothesis_cards[0].support_evidence_refs == ["E1"]
    assert review.recommended_next_routes == ["microscopic.run_torsion_snapshots"]
    assert any("Generic web context" in item for item in review.overclaim_warnings)


def test_photophysics_arbiter_allows_support_refs_outside_compact_subset() -> None:
    class CompactSubsetArbiterLLM:
        settings = type("Settings", (), {"model": "fake-arbiter"})()

        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def is_configured(self) -> bool:
            return True

        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt, schema_feedback
            self.payload = payload
            return {
                "review_id": "llm_supplied_id",
                "case_id": "llm_supplied_case",
                "round_id": "llm_supplied_round",
                "coverage_axes": [],
                "hypothesis_cards": [
                    {
                        "hypothesis_id": "rim_card",
                        "mechanism": "RIM_RIR_RIV",
                        "status": "proxy_supported",
                        "support_evidence_refs": ["E_old"],
                        "weakening_evidence_refs": [],
                        "missing_or_unresolved": [],
                        "reasoning_summary": "Older cited evidence remains ledger-grounded.",
                        "scope_limits": [],
                        "priority": 0.3,
                    }
                ],
                "support_argument_audits": [],
                "recommended_next_routes": [],
                "overclaim_warnings": [],
                "source_evidence_refs": ["E_new"],
            }

    evidence_items = [
        _evidence_unit(
            evidence_id=f"E{i}",
            round_id=f"R{i:03d}",
            source_report_id=f"R{i:03d}:macro",
            agent_name="macro",
            capability_id=f"macro.route_{i}",
            family="geometry_precondition",
            observable_tags=[f"route_{i}"],
            summary=f"Evidence {i}.",
        )
        for i in range(1, 8)
    ]
    evidence_items[0] = evidence_items[0].model_copy(update={"evidence_id": "E_old"})
    evidence_items[-1] = evidence_items[-1].model_copy(update={"evidence_id": "E_new"})
    client = CompactSubsetArbiterLLM()
    case_run = _case_run_for_gate().touch(
        current_round_id="R007",
        evidence_ledger=EvidenceLedger(case_id="gate_case", items=evidence_items),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R007:support:001",
                target_label="RIM_RIR_RIV",
                support_level="partial",
                finding="Older runtime evidence identifies a rotor proxy.",
                warrant="This can support RIM as a bounded proxy.",
                boundary="Direct aggregate restriction remains unmeasured.",
                observation_refs=["E_old"],
            )
        ],
    )
    arbiter = PhotophysicsArbiterAgent(
        llm_client=client,  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
    )

    review = arbiter.review(case_run)

    assert "E_old" in client.payload["allowed_evidence_ids"]  # type: ignore[index]
    assert "E_old" not in client.payload["compact_evidence_ids"]  # type: ignore[index]
    assert review.hypothesis_cards[0].support_evidence_refs == ["E_old"]
    assert "E_old" not in review.source_evidence_refs


def test_photophysics_arbiter_llm_failure_raises_without_scientific_fallback() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    observable_tags=["macro_structure_scan_proxy"],
                    summary="Macro evidence.",
                )
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R001:support:001",
                target_label="RIM_RIR_RIV",
                support_level="partial",
                finding="Runtime evidence identifies rotatable aryl units.",
                warrant=(
                    "This partially supports RIM/RIR/RIV because rotatable "
                    "units can open intramolecular-motion loss channels."
                ),
                boundary=(
                    "This is proxy support; aggregate-state restriction was "
                    "not measured."
                ),
                observation_refs=["E1"],
            )
        ],
    )
    arbiter = PhotophysicsArbiterAgent(
        llm_client=FailingArbiterLLM(),  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
    )

    with pytest.raises(RuntimeError, match="failed to produce a valid review"):
        arbiter.review(case_run)


def test_photophysics_arbiter_retries_invalid_grounding_contract() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    observable_tags=["macro_structure_scan_proxy"],
                    summary="Macro evidence.",
                )
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R001:support:001",
                target_label="RIM_RIR_RIV",
                support_level="partial",
                finding="Runtime evidence identifies rotatable aryl units.",
                warrant=(
                    "This partially supports RIM/RIR/RIV because rotatable "
                    "units can open intramolecular-motion loss channels."
                ),
                boundary=(
                    "This is proxy support; aggregate-state restriction was "
                    "not measured."
                ),
                observation_refs=["E1"],
            )
        ],
    )
    fake_client = RetryArbiterLLM()
    arbiter = PhotophysicsArbiterAgent(
        llm_client=fake_client,  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
        max_attempts=2,
    )

    review = arbiter.review(case_run)

    assert fake_client.calls == 2
    assert fake_client.feedbacks[0] is None
    assert "text fields must not be empty" in str(fake_client.feedbacks[1])
    assert review.hypothesis_cards[0].status == "underdetermined"


def test_photophysics_arbiter_normalizes_llm_schema_drift() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    observable_tags=["macro_structure_scan_proxy"],
                    summary="Macro evidence.",
                )
            ],
        ),
    )
    arbiter = PhotophysicsArbiterAgent(
        llm_client=SchemaDriftArbiterLLM(),  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
    )

    review = arbiter.review(case_run)

    assert review.coverage_axes[0].evidence_refs == ["E1"]
    assert review.coverage_axes[0].status == "missing"
    assert review.coverage_axes[0].recommended_routes == [
        "microscopic.run_torsion_snapshots"
    ]
    assert review.hypothesis_cards[0].hypothesis_id == "rim_torsion"
    assert review.hypothesis_cards[0].scope_limits == ["No wet-lab claim."]
    assert review.hypothesis_cards[0].priority == pytest.approx(0.85)


def test_photophysics_arbiter_fills_missing_axis_rationale_without_retry() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    observable_tags=["macro_structure_scan_proxy"],
                    summary="Macro evidence.",
                )
            ],
        ),
    )
    fake_client = MissingAxisRationaleArbiterLLM()
    arbiter = PhotophysicsArbiterAgent(
        llm_client=fake_client,  # type: ignore[arg-type]
        web_search=FakeGenericSearch(),
    )

    review = arbiter.review(case_run)

    assert fake_client.calls == 1
    assert review.coverage_axes[0].rationale.startswith("LLM omitted rationale")


class FakeConclusionLedgerLLM:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        assert "ConclusionLedgerAgent" in system_prompt
        payload_text = json.dumps(payload)
        assert "hidden_reference" not in payload_text
        assert "semantic_evidence_targets" not in payload_text
        assert "gate_case" not in payload_text
        assert schema_feedback is None
        self.payloads.append(payload)
        return {
            "evidence_conclusions": [
                {
                    "conclusion_id": "EC001",
                    "statement": "Macro evidence supports a rotor-restriction proxy.",
                    "context": "Planner-grounded conclusion ledger",
                    "mechanism_family": "RIM/RIR",
                    "direction": "supports",
                    "basis": "proxy",
                    "source_evidence_refs": ["E1", "not_allowed"],
                    "source_snippets": ["Macro evidence."],
                    "limits": ["No wet-lab PL claim."],
                    "confidence": "medium",
                }
            ],
            "diagnosis_conclusions": [
                {
                    "conclusion_id": "DC001",
                    "mechanism": "RIM/RIR proxy mechanism",
                    "status": "proxy_supported",
                    "statement": "RIM/RIR is proxy-supported by accepted evidence.",
                    "reasoning_summary": "Macro evidence supports a rotor-restriction proxy.",
                    "source_evidence_refs": ["E1", "not_allowed"],
                    "source_snippets": ["Macro evidence."],
                    "missing_or_unresolved": ["Needs aggregation PL."],
                    "scope_limits": ["No wet-lab claim."],
                    "confidence": "medium",
                }
            ],
            "pending_questions": [
                {
                    "question_id": "PQ001",
                    "question": "Aggregation-state PL remains unresolved.",
                    "needed_evidence": ["water-fraction PL"],
                    "source_evidence_refs": ["E1", "not_allowed"],
                }
            ],
            "policy_notes": ["LLM ledger used runtime evidence."],
        }


def test_conclusion_ledger_agent_sanitizes_refs_without_hidden_reference() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        input=CaseInput(
            case_id="gate_case",
            smiles="C1=CC=CC=C1",
            user_query="Assess mechanism.",
            metadata={"public_case_id": "CASE_009"},
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    summary="Macro evidence.",
                )
            ],
        ),
    )
    client = FakeConclusionLedgerLLM()
    agent = ConclusionLedgerAgent(llm_client=client)  # type: ignore[arg-type]

    ledger = agent.build(case_run)

    assert client.payloads
    assert ledger.case_id == "CASE_009"
    assert ledger.ledger_id == "CASE_009:R001:conclusion_ledger"
    assert ledger.evidence_conclusions[0].source_evidence_refs == ["E1"]
    assert ledger.evidence_conclusions[0].source_snippets == ["Macro evidence."]
    assert ledger.diagnosis_conclusions[0].source_evidence_refs == ["E1"]
    assert ledger.pending_questions[0].source_evidence_refs == ["E1"]


def test_conclusion_ledger_allows_negative_public_source_policy_note() -> None:
    class SourcePolicyNoteLLM(FakeConclusionLedgerLLM):
        def complete_json(self, **kwargs) -> dict[str, object]:  # type: ignore[no-untyped-def]
            response = super().complete_json(**kwargs)
            response["policy_notes"] = [
                "No source paper or paper-specific facts were used."
            ]
            return response

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    summary="Macro evidence.",
                )
            ],
        ),
    )

    ledger = ConclusionLedgerAgent(
        llm_client=SourcePolicyNoteLLM()  # type: ignore[arg-type]
    ).build(case_run)

    assert "No source paper or paper-specific facts were used." in ledger.policy_notes


def test_conclusion_ledger_rejects_private_reference_field_names() -> None:
    class PrivateReferenceFieldLLM(FakeConclusionLedgerLLM):
        def complete_json(self, **kwargs) -> dict[str, object]:  # type: ignore[no-untyped-def]
            response = super().complete_json(**kwargs)
            response["policy_notes"] = ["hidden_reference was inspected."]
            return response

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    summary="Macro evidence.",
                )
            ],
        ),
    )

    with pytest.raises(RuntimeError, match="forbidden private/source text"):
        ConclusionLedgerAgent(
            llm_client=PrivateReferenceFieldLLM()  # type: ignore[arg-type]
        ).build(case_run)


def test_conclusion_ledger_downgrades_unsupported_strong_diagnosis() -> None:
    class UngroundedDiagnosisLLM(FakeConclusionLedgerLLM):
        def complete_json(self, **kwargs) -> dict[str, object]:  # type: ignore[no-untyped-def]
            self.payloads.append(kwargs["payload"])
            return {
                "evidence_conclusions": [
                    {
                        "conclusion_id": "EC001",
                        "statement": "Macro evidence exists.",
                        "context": "Planner-grounded conclusion ledger",
                        "mechanism_family": None,
                        "direction": "unresolved",
                        "basis": "proxy",
                        "source_evidence_refs": ["E1"],
                        "source_snippets": ["Macro evidence."],
                        "limits": [],
                        "confidence": "medium",
                    }
                ],
                "diagnosis_conclusions": [
                    {
                        "conclusion_id": "DC001",
                        "mechanism": "ESIPT",
                        "status": "proxy_supported",
                        "statement": "Unsupported strong diagnosis.",
                        "reasoning_summary": "No evidence refs were supplied.",
                        "source_evidence_refs": [],
                        "source_snippets": [],
                        "missing_or_unresolved": [],
                        "scope_limits": [],
                        "confidence": "medium",
                    }
                ],
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    summary="Macro evidence.",
                )
            ],
        ),
    )
    ledger = ConclusionLedgerAgent(
        llm_client=UngroundedDiagnosisLLM()  # type: ignore[arg-type]
    ).build(case_run)

    assert ledger.diagnosis_conclusions[0].status == "underdetermined"
    assert ledger.diagnosis_conclusions[0].missing_or_unresolved


def test_build_diagnosis_units_uses_conclusion_ledger() -> None:
    case_run = _case_run_for_gate().touch(
        conclusion_ledger=ConclusionLedger(
            ledger_id="gate_case:R001:conclusion_ledger",
            case_id="gate_case",
            round_id="R001",
            evidence_conclusions=[
                EvidenceConclusion(
                    conclusion_id="EC001",
                    statement="Evidence supports a rotor-restriction proxy.",
                    context="test",
                    source_evidence_refs=["E1"],
                )
            ],
            diagnosis_conclusions=[
                DiagnosisConclusion(
                    conclusion_id="DC001",
                    mechanism="RIM/RIR proxy mechanism",
                    status="proxy_supported",
                    statement="RIM/RIR is proxy-supported.",
                    reasoning_summary="Accepted evidence supports the proxy.",
                    source_evidence_refs=["E1"],
                    missing_or_unresolved=["Needs aggregation PL."],
                    scope_limits=["No wet-lab claim."],
                )
            ],
        )
    )

    diagnoses = build_diagnosis_units(case_run)

    assert diagnoses[0].diagnosis_id == "DC001"
    assert diagnoses[0].context == "ConclusionLedgerAgent runtime record"
    assert diagnoses[0].evidence_refs == ["E1"]


def test_output_assembler_packages_planner_portfolio_without_new_reasoning() -> None:
    case_run = _case_run_for_gate().touch(
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.7,
                    status="plausible",
                    rationale="Planner ranked this from cited evidence.",
                    evidence_refs=["E1", "missing"],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.4,
                    status="pending",
                    rationale="Planner kept this as a lower-ranked possibility.",
                    evidence_refs=[],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Rotor evidence.",
                    claim="Runtime evidence identifies rotatable aryl units.",
                )
            ],
        ),
        mechanism_support_arguments=[
            MechanismSupportArgument(
                support_id="R001:support:001",
                target_label="RIM_RIR_RIV",
                support_level="partial",
                finding="Runtime evidence identifies rotatable aryl units.",
                warrant=(
                    "This partially supports RIM/RIR/RIV because rotatable "
                    "units can open intramolecular-motion loss channels."
                ),
                boundary=(
                    "This is proxy support; aggregate-state restriction was "
                    "not measured."
                ),
                observation_refs=["E1"],
            )
        ],
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [item.label for item in predictions] == ["RIM_RIR_RIV", "ICT_TICT_CT"]
    assert predictions[0].rank == 1
    assert predictions[0].evidence_refs == ["E1"]
    assert predictions[0].evidence == [
        (
            "Finding: Runtime evidence identifies rotatable aryl units. "
            "Warrant: This partially supports RIM/RIR/RIV because rotatable "
            "units can open intramolecular-motion loss channels. "
            "Boundary: This is proxy support; aggregate-state restriction was "
            "not measured."
        )
    ]
    assert predictions[0].evidence_details
    assert predictions[0].evidence_details[0].finding == (
        "Runtime evidence identifies rotatable aryl units."
    )
    assert predictions[0].evidence_details[0].warrant == (
        "This partially supports RIM/RIR/RIV because rotatable "
        "units can open intramolecular-motion loss channels."
    )
    assert predictions[0].evidence_details[0].boundary == (
        "This is proxy support; aggregate-state restriction was not measured."
    )
    assert predictions[0].evidence_details[0].evidence_refs == ["E1"]
    assert predictions[0].claim_status == "partially_supported_candidate"
    assert predictions[0].support_strength == "partial"
    assert predictions[0].validation_needed == [
        "This is proxy support; aggregate-state restriction was not measured."
    ]
    assert predictions[1].evidence_refs == []


def test_photophysics_arbiter_blocks_case_specific_web_queries() -> None:
    case_run = _case_run_for_gate()

    with pytest.raises(ValueError, match="case-specific"):
        validate_generic_query(case_run.input.smiles, case_run)


class UnconfiguredAgendaLLM:
    def is_configured(self) -> bool:
        return False


def test_mechanism_agenda_requires_configured_llm_without_scientific_fallback() -> None:
    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=UnconfiguredAgendaLLM(),  # type: ignore[arg-type]
    )
    with pytest.raises(RuntimeError, match="requires a configured LLM client"):
        agent.propose(_case_run_for_gate(), round_id="R001")


def test_mechanism_agenda_agent_does_not_call_programmatic_fallback() -> None:
    source = Path("mechcal/agents/mechanism_agenda.py").read_text(encoding="utf-8")

    assert "build_fallback_mechanism_agenda" not in source
    assert "extract_smiles_features" not in source
    assert '"molecule_features"' not in source


class CandidateOnlyAgendaLLM:
    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        del system_prompt, payload, schema_feedback
        return {
            "candidate_mechanisms": [
                {
                    "mechanism_id": "candidate_1",
                    "label": "Candidate mechanism",
                    "mechanism_family": "RIM_RIR_RIV",
                    "context": "Runtime evidence context.",
                    "rationale": "This is an agenda candidate, not a final conclusion.",
                }
            ]
        }


class RetryAgendaLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.feedbacks: list[str | None] = []

    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        del system_prompt, payload
        self.calls += 1
        self.feedbacks.append(schema_feedback)
        if self.calls == 1:
            return {"final_diagnosis": "forbidden final output"}
        return {
            "candidate_mechanisms": [
                {
                    "mechanism_id": "candidate_1",
                    "label": "Candidate mechanism",
                    "mechanism_family": "RIM_RIR_RIV",
                    "context": "Runtime evidence context.",
                    "rationale": "This is an agenda candidate, not a final conclusion.",
                }
            ],
            "evidence_questions": [
                {
                    "question_id": "Q001",
                    "mechanism_id": "candidate_1",
                    "question": "Which public evidence route can bound this candidate?",
                    "observable": "coverage_gap",
                    "expected_basis": "proxy",
                    "status": "unresolved",
                    "acceptable_capability_ids": [
                        "macro.macro_structure_scan",
                    ],
                }
            ],
        }


def test_mechanism_agenda_repairs_missing_questions_without_scientific_fallback() -> None:
    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=CandidateOnlyAgendaLLM(),  # type: ignore[arg-type]
    )

    program = agent.propose(_case_run_for_gate(), round_id="R001")

    assert len(program.candidate_mechanisms) == 1
    assert len(program.evidence_questions) == 1
    assert program.evidence_questions[0].mechanism_id == "candidate_1"
    assert program.evidence_questions[0].acceptable_capability_ids
    assert "final diagnosis" not in program.evidence_questions[0].question.lower()


def test_mechanism_agenda_retries_malformed_program_json_without_scientific_fallback() -> None:
    client = RetryAgendaLLM()
    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=client,  # type: ignore[arg-type]
    )

    program = agent.propose(_case_run_for_gate(), round_id="R001")

    assert client.calls == 2
    assert client.feedbacks[0] is None
    assert client.feedbacks[1] is not None
    assert "forbidden" in client.feedbacks[1].lower()
    assert program.evidence_questions[0].acceptable_capability_ids == [
        "macro.macro_structure_scan"
    ]


class CoverageAgendaLLM:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        del system_prompt, schema_feedback
        self.payloads.append(payload)
        return {
            "coverage_summary": "Public evidence coverage memo only.",
            "agenda_items": [
                {
                    "label": "RACI_CI_ACCESS",
                    "coverage_status": "under_screened",
                    "why_relevant": "Torsion-sensitive pathways are not yet bounded.",
                    "missing_evidence": "No CI-specific proxy has been collected.",
                    "suggested_question": "Does torsion evidence bound CI-access risk?",
                    "suggested_route": "microscopic.run_torsion_brightness_coupling_scan",
                    "urgency": "high",
                    "boundary": "This is coverage feedback, not final diagnosis.",
                }
            ],
            "screened_out": [
                {
                    "label": "TRIPLET_METAL_ENERGY_TRANSFER",
                    "reason": "No public metal motif is present.",
                }
            ],
            "low_margin_competitions": [
                {
                    "labels": ["RACI_CI_ACCESS", "AGGREGATE_EXCITON_EXCIMER"],
                    "reason": "Both remain proxy-level candidates.",
                    "suggested_disambiguating_route": "torsion versus aggregate route.",
                }
            ],
            "recommended_next_routes": [
                {
                    "route": "microscopic.run_torsion_brightness_coupling_scan",
                    "targets": ["RACI_CI_ACCESS"],
                    "reason": "This checks the under-screened RACI proxy.",
                    "capability_ids": ["microscopic.run_torsion_brightness_coupling_scan"],
                }
            ],
        }


def test_mechanism_agenda_initial_coverage_bootstraps_without_llm() -> None:
    client = CoverageAgendaLLM()
    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=client,  # type: ignore[arg-type]
    )
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="esipt_public_case",
            smiles="Oc1ccccc1n2cccc2",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "PUBLIC_CASE"},
        )
    )

    coverage = agent.review_coverage(case_run, round_id="R001")

    assert client.payloads == []
    payload_text = coverage.model_dump_json().lower()
    assert "hidden_reference" not in payload_text
    assert "reference_mechanisms" not in payload_text
    assert "mechanism_predictions" not in payload_text
    assert "top-3" not in payload_text
    assert "final_diagnosis" not in payload_text
    capability_ids = [
        capability_id
        for route in coverage.recommended_next_routes
        for capability_id in route.capability_ids
    ]
    assert "microscopic.run_baseline_bundle" in capability_ids
    assert "macro.screen_esipt_structural_motif" in capability_ids
    assert "macro.screen_polar_binding_site_prior" in capability_ids
    assert "macro.screen_donor_acceptor_layout" in capability_ids
    assert "macro.screen_rotor_torsion_topology" in capability_ids
    assert "macro.screen_aggregation_prone_scaffold" in capability_ids


def test_mechanism_agenda_coverage_payload_is_public_only() -> None:
    client = CoverageAgendaLLM()
    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=client,  # type: ignore[arg-type]
    )
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="secret_internal_case",
            smiles="C1=CC=CC=C1",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "PUBLIC_CASE"},
        )
    )

    coverage = agent.review_coverage(case_run, round_id="R002")

    assert coverage.agenda_items[0].label == "RACI_CI_ACCESS"
    guide = client.payloads[0]["ability_evidence_guide"]
    assert guide
    guide_by_capability = {item["capability_id"]: item for item in guide}
    assert "ICT_TICT_CT" in guide_by_capability["macro.screen_donor_acceptor_layout"][
        "screens"
    ]
    assert "RADIATIVE_RATE_STATE_BALANCE" in guide_by_capability[
        "microscopic.run_bright_dark_state_ordering"
    ]["screens"]
    payload_text = json.dumps(client.payloads[0], ensure_ascii=False).lower()
    assert "hidden_reference" not in payload_text
    assert "reference_mechanisms" not in payload_text
    assert "semantic_evidence_targets" not in payload_text
    assert "semantic_diagnosis_targets" not in payload_text
    assert "benchmark target" not in payload_text


def test_mechanism_agenda_first_coverage_keeps_low_cost_state_route() -> None:
    class ThreeRouteCoverageLLM(CoverageAgendaLLM):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt, payload, schema_feedback
            return {
                "coverage_summary": "Early coverage memo.",
                "agenda_items": [
                    {
                        "label": "RIM_RIR_RIV",
                        "coverage_status": "under_screened",
                        "why_relevant": "Rotor evidence is not yet screened.",
                        "missing_evidence": "Rotor topology is missing.",
                        "suggested_question": "Can rotors be bounded?",
                        "suggested_route": "macro.screen_rotor_torsion_topology",
                        "urgency": "high",
                        "boundary": "Coverage only, not diagnosis.",
                    }
                ],
                "screened_out": [],
                "low_margin_competitions": [],
                "recommended_next_routes": [
                    {
                        "route": "macro.macro_structure_scan",
                        "targets": ["RIM_RIR_RIV"],
                        "reason": "Macro scaffold screen.",
                        "capability_ids": ["macro.macro_structure_scan"],
                    },
                    {
                        "route": "macro.screen_rotor_torsion_topology",
                        "targets": ["RIM_RIR_RIV", "RACI_CI_ACCESS"],
                        "reason": "Rotor screen.",
                        "capability_ids": ["macro.screen_rotor_torsion_topology"],
                    },
                    {
                        "route": "macro.screen_donor_acceptor_architecture",
                        "targets": ["ICT_TICT_CT"],
                        "reason": "D-A screen.",
                        "capability_ids": ["macro.screen_donor_acceptor_architecture"],
                    },
                ],
            }

    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=ThreeRouteCoverageLLM(),  # type: ignore[arg-type]
    )
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="esipt_public_case",
            smiles="Oc1ccccc1n2cccc2",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "PUBLIC_CASE"},
        )
    )

    coverage = agent.review_coverage(case_run, round_id="R001")

    capability_ids = [
        capability_id
        for route in coverage.recommended_next_routes
        for capability_id in route.capability_ids
    ]
    assert "microscopic.run_baseline_bundle" in capability_ids
    assert "macro.screen_esipt_structural_motif" in capability_ids
    assert "macro.screen_polar_binding_site_prior" in capability_ids
    assert "macro.screen_donor_acceptor_layout" in capability_ids
    assert "macro.screen_rotor_torsion_topology" in capability_ids
    assert "macro.screen_aggregation_prone_scaffold" in capability_ids
    assert len(coverage.recommended_next_routes) == 6


def test_mechanism_agenda_first_coverage_adds_aggregation_when_no_esipt_trigger() -> None:
    class ThreeRouteCoverageLLM(CoverageAgendaLLM):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt, payload, schema_feedback
            return {
                "coverage_summary": "Early coverage memo.",
                "agenda_items": [
                    {
                        "label": "RIM_RIR_RIV",
                        "coverage_status": "under_screened",
                        "why_relevant": "Rotor evidence is not yet screened.",
                        "missing_evidence": "Rotor topology is missing.",
                        "suggested_question": "Can rotors be bounded?",
                        "suggested_route": "macro.screen_rotor_torsion_topology",
                        "urgency": "high",
                        "boundary": "Coverage only, not diagnosis.",
                    }
                ],
                "screened_out": [],
                "low_margin_competitions": [],
                "recommended_next_routes": [
                    {
                        "route": "macro.screen_donor_acceptor_architecture",
                        "targets": ["ICT_TICT_CT"],
                        "reason": "D-A screen.",
                        "capability_ids": ["macro.screen_donor_acceptor_architecture"],
                    },
                    {
                        "route": "macro.screen_rotor_torsion_topology",
                        "targets": ["RIM_RIR_RIV", "RACI_CI_ACCESS"],
                        "reason": "Rotor screen.",
                        "capability_ids": ["macro.screen_rotor_torsion_topology"],
                    },
                    {
                        "route": "macro.macro_structure_scan",
                        "targets": ["RIM_RIR_RIV"],
                        "reason": "Macro scaffold screen.",
                        "capability_ids": ["macro.macro_structure_scan"],
                    },
                ],
            }

    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=ThreeRouteCoverageLLM(),  # type: ignore[arg-type]
    )
    coverage = agent.review_coverage(_case_run_for_gate(), round_id="R001")

    capability_ids = [
        capability_id
        for route in coverage.recommended_next_routes
        for capability_id in route.capability_ids
    ]
    assert "microscopic.run_baseline_bundle" in capability_ids
    assert "macro.screen_aggregation_prone_scaffold" in capability_ids
    assert "macro.screen_donor_acceptor_layout" in capability_ids
    assert "macro.screen_rotor_torsion_topology" in capability_ids
    assert len(coverage.recommended_next_routes) == 4


def test_mechanism_agenda_first_coverage_adds_triplet_prior_when_triggered() -> None:
    class ThreeRouteCoverageLLM(CoverageAgendaLLM):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt, payload, schema_feedback
            return {
                "coverage_summary": "Early coverage memo.",
                "agenda_items": [],
                "screened_out": [],
                "low_margin_competitions": [],
                "recommended_next_routes": [
                    {
                        "route": "macro.screen_donor_acceptor_architecture",
                        "targets": ["ICT_TICT_CT"],
                        "reason": "D-A screen.",
                        "capability_ids": ["macro.screen_donor_acceptor_architecture"],
                    },
                    {
                        "route": "macro.screen_rotor_torsion_topology",
                        "targets": ["RIM_RIR_RIV"],
                        "reason": "Rotor screen.",
                        "capability_ids": ["macro.screen_rotor_torsion_topology"],
                    },
                ],
            }

    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=ThreeRouteCoverageLLM(),  # type: ignore[arg-type]
    )
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="triplet_public_case",
            smiles="Brc1ccccc1",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "PUBLIC_TRIPLET_CASE"},
        )
    )

    coverage = agent.review_coverage(case_run, round_id="R001")

    capability_ids = [
        capability_id
        for route in coverage.recommended_next_routes
        for capability_id in route.capability_ids
    ]
    assert "microscopic.run_baseline_bundle" in capability_ids
    assert "macro.screen_metal_triplet_prior" in capability_ids
    assert "macro.screen_aggregation_prone_scaffold" in capability_ids
    assert "macro.screen_donor_acceptor_layout" in capability_ids
    assert "macro.screen_rotor_torsion_topology" in capability_ids
    assert len(coverage.recommended_next_routes) == 5


def test_mechanism_agenda_first_coverage_adds_polar_binding_prior_when_triggered() -> None:
    class ThreeRouteCoverageLLM(CoverageAgendaLLM):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt, payload, schema_feedback
            return {
                "coverage_summary": "Early coverage memo.",
                "agenda_items": [],
                "screened_out": [],
                "low_margin_competitions": [],
                "recommended_next_routes": [
                    {
                        "route": "macro.screen_rotor_torsion_topology",
                        "targets": ["RIM_RIR_RIV"],
                        "reason": "Rotor screen.",
                        "capability_ids": ["macro.screen_rotor_torsion_topology"],
                    },
                    {
                        "route": "macro.screen_donor_acceptor_architecture",
                        "targets": ["ICT_TICT_CT"],
                        "reason": "D-A screen.",
                        "capability_ids": ["macro.screen_donor_acceptor_architecture"],
                    },
                ],
            }

    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=ThreeRouteCoverageLLM(),  # type: ignore[arg-type]
    )
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="polar_public_case",
            smiles="CC(=O)c1cccc(C(=O)C)c1",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "PUBLIC_POLAR_CASE"},
        )
    )

    coverage = agent.review_coverage(case_run, round_id="R001")

    capability_ids = [
        capability_id
        for route in coverage.recommended_next_routes
        for capability_id in route.capability_ids
    ]
    assert "microscopic.run_baseline_bundle" in capability_ids
    assert "macro.screen_polar_binding_site_prior" in capability_ids
    assert "macro.screen_donor_acceptor_layout" in capability_ids
    assert "macro.screen_rotor_torsion_topology" in capability_ids
    assert "macro.screen_aggregation_prone_scaffold" in capability_ids
    assert len(coverage.recommended_next_routes) == 5


def test_mechanism_agenda_followup_adds_packing_contact_routes() -> None:
    class EmptyCoverageLLM(CoverageAgendaLLM):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt, payload, schema_feedback
            return {
                "coverage_summary": "Follow-up coverage memo.",
                "agenda_items": [],
                "screened_out": [],
                "low_margin_competitions": [],
                "recommended_next_routes": [],
            }

    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=EmptyCoverageLLM(),  # type: ignore[arg-type]
    )
    case_run = _case_run_for_gate().touch(
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="R001:macro:aggregation",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    family="geometry_precondition",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    summary="Aggregation-prone scaffold was screened from structure.",
                    observable_tags=["aggregation_proxy"],
                )
            ],
        )
    )

    coverage = agent.review_coverage(case_run, round_id="R002")

    capability_ids = [
        capability_id
        for route in coverage.recommended_next_routes
        for capability_id in route.capability_ids
    ]
    assert capability_ids[:3] == [
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_crystal_restriction_checklist",
    ]
    payload_text = coverage.model_dump_json().lower()
    assert "hidden_reference" not in payload_text
    assert "reference_mechanisms" not in payload_text


def test_mechanism_agenda_coverage_rejects_top3_or_final_diagnosis() -> None:
    class BadCoverageLLM(CoverageAgendaLLM):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            del system_prompt, payload, schema_feedback
            return {
                "coverage_summary": "Top-3 final diagnosis says RACI wins.",
                "agenda_items": [
                    {
                        "label": "RACI_CI_ACCESS",
                        "coverage_status": "under_screened",
                        "why_relevant": "bad",
                        "missing_evidence": "bad",
                        "suggested_question": "bad",
                        "suggested_route": "bad",
                        "urgency": "high",
                        "boundary": "bad",
                    }
                ],
            }

    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=BadCoverageLLM(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="forbidden"):
        agent.review_coverage(_case_run_for_gate(), round_id="R002")


def test_planner_normalizer_runs_coverage_route_before_finalize() -> None:
    response = {
        "decision_id": "R003:planner_final",
        "round_id": "R003",
        "portfolio": {
            "current": "RIM_RIR_RIV",
            "runner_up": "PACKING_HOST_MATRIX_CONFINEMENT",
            "hypotheses": [
                {
                    "name": "RIM_RIR_RIV",
                    "confidence": 0.58,
                    "differential_priority": 0.58,
                    "status": "plausible",
                    "evidence_support": "partial",
                    "claim_status": "partially_supported_candidate",
                    "rationale": "Runtime evidence provides bounded rotor support.",
                    "evidence_refs": ["E1"],
                }
            ],
        },
        "current_hypothesis": "RIM_RIR_RIV",
        "confidence": 0.58,
        "diagnosis": "LLM attempted to finalize while coverage still has a route.",
        "action": "finalize",
        "dispatch_requests": [],
        "final_answer_draft": "Final draft should be deferred until coverage route runs.",
    }
    prompt_payload = {
        "agenda_coverage_memo": {
            "recommended_next_routes": [
                {
                    "capability_ids": ["macro.run_dimer_packing_proxy"],
                    "targets": ["PACKING_HOST_MATRIX_CONFINEMENT"],
                }
            ]
        },
        "new_round_evidence": [
            {
                "evidence_id": "E1",
                "capability_id": "macro.screen_aggregation_prone_scaffold",
            }
        ],
        "mechanism_evidence_coverage_table": [
            {
                "label": "PACKING_HOST_MATRIX_CONFINEMENT",
                "capability_ids": ["macro.screen_aggregation_prone_scaffold"],
            }
        ],
    }

    normalized = planner_module._normalize_planner_llm_response(
        response,
        round_id="R003",
        existing_portfolio={
            "current": "RIM_RIR_RIV",
            "hypotheses": [
                {
                    "name": "RIM_RIR_RIV",
                    "confidence": 0.58,
                    "status": "plausible",
                    "evidence_refs": ["E1"],
                }
            ],
        },
        previous_portfolio_hypotheses=[
            {
                "name": "RIM_RIR_RIV",
                "confidence": 0.58,
                "differential_priority": 0.58,
                "status": "plausible",
                "evidence_refs": ["E1"],
            }
        ],
        allowed_evidence_ids=["E1"],
        capability_registry=default_capability_registry(),
        stage="planner_update",
        prompt_payload=prompt_payload,
    )

    assert normalized["action"] == "dispatch"
    assert normalized["final_answer_draft"] is None
    assert normalized["dispatch_requests"][0]["capability_id"] == (
        "macro.run_dimer_packing_proxy"
    )


def test_planner_normalizer_resolves_high_urgency_agenda_suggested_route() -> None:
    response = {
        "portfolio": {
            "current": "ESIPT_PT",
            "hypotheses": [
                {
                    "name": "ESIPT_PT",
                    "confidence": 0.20,
                    "differential_priority": 0.20,
                }
            ],
        },
        "current_hypothesis": "ESIPT_PT",
        "diagnosis": "ESIPT remains under-screened.",
        "action": "finalize",
    }
    prompt_payload = {
        "agenda_coverage_memo": {
            "agenda_items": [
                {
                    "label": "ESIPT_PT",
                    "coverage_status": "under_screened",
                    "suggested_route": "screen_esipt_structural_motif",
                    "urgency": "high",
                }
            ],
            "recommended_next_routes": [],
        }
    }
    registry = default_capability_registry().for_owner_agents(("macro",))

    normalized = planner_module._normalize_planner_llm_response(
        response,
        round_id="R002",
        existing_portfolio={
            "current": "ESIPT_PT",
            "hypotheses": [
                {
                    "name": "ESIPT_PT",
                    "confidence": 0.20,
                    "differential_priority": 0.20,
                }
            ],
        },
        previous_portfolio_hypotheses=[],
        allowed_evidence_ids=[],
        capability_registry=registry,
        stage="planner_update",
        prompt_payload=prompt_payload,
    )

    assert normalized["action"] == "dispatch"
    assert normalized["final_answer_draft"] is None
    assert [item["capability_id"] for item in normalized["dispatch_requests"]] == [
        "macro.screen_esipt_structural_motif"
    ]


def test_planner_normalizer_prioritizes_high_urgency_route_under_dispatch_limit() -> None:
    response = {
        "portfolio": {
            "current": "ICT_TICT_CT",
            "hypotheses": [
                {
                    "name": "ICT_TICT_CT",
                    "confidence": 0.20,
                    "differential_priority": 0.20,
                }
            ],
        },
        "current_hypothesis": "ICT_TICT_CT",
        "diagnosis": "Continue bounded public-evidence collection.",
        "action": "dispatch",
        "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
    }
    prompt_payload = {
        "agenda_coverage_memo": {
            "agenda_items": [
                {
                    "label": "ICT_TICT_CT",
                    "coverage_status": "needs_follow_up",
                    "suggested_route": "macro.screen_donor_acceptor_architecture",
                    "urgency": "high",
                }
            ],
            "recommended_next_routes": [
                {
                    "route": capability_id,
                    "capability_ids": [capability_id],
                }
                for capability_id in (
                    "macro.run_dimer_packing_proxy",
                    "macro.run_aggregate_contact_proxy",
                    "macro.run_crystal_restriction_checklist",
                    "macro.screen_polar_binding_site_prior",
                    "macro.run_solid_state_emission_proxy",
                )
            ],
        }
    }
    registry = default_capability_registry().for_owner_agents(("macro",))

    normalized = planner_module._normalize_planner_llm_response(
        response,
        round_id="R003",
        existing_portfolio={
            "current": "ICT_TICT_CT",
            "hypotheses": [
                {
                    "name": "ICT_TICT_CT",
                    "confidence": 0.20,
                    "differential_priority": 0.20,
                }
            ],
        },
        previous_portfolio_hypotheses=[],
        allowed_evidence_ids=[],
        capability_registry=registry,
        stage="planner_update",
        prompt_payload=prompt_payload,
    )

    capability_ids = [
        item["capability_id"] for item in normalized["dispatch_requests"]
    ]
    assert capability_ids[0] == "macro.screen_donor_acceptor_architecture"
    assert len(capability_ids) <= planner_module.MAX_DISPATCHES_PER_ROUND


def test_first_evidence_ranking_does_not_depend_on_bootstrap_memo_id() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="First-round rotor topology evidence.",
                )
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R002:agenda_coverage",
            case_id="gate_case",
            round_id="R002",
            coverage_summary="Reviewer replaced the bootstrap memo after evidence collection.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="macro.run_dimer_packing_proxy",
                    targets=["PACKING_HOST_MATRIX_CONFINEMENT"],
                    reason="Continue bounded packing coverage.",
                    capability_ids=["macro.run_dimer_packing_proxy"],
                )
            ],
        ),
        runtime={
            "ablation": {"mode": "mechcal_wo_microscopic"},
            "incremental_mechanism_portfolio": {
                "enabled": False,
                "policy": "stateless_ranking_with_feedback_retained",
                "prior_planner_ranking_present": False,
            }
        },
    )

    assert planner_module._should_use_initial_evidence_ranking_payload(
        case_run=case_run,
        stage="planner_update",
        new_round_evidence=[{"evidence_id": "E1"}],
    )
    payload = planner_module._initial_evidence_ranking_payload(
        case_run=case_run,
        round_id="R002",
        context={"round_index": 2, "max_rounds": 30},
        capability_registry=default_capability_registry().for_owner_agents(("macro",)),
    )
    assert not any(
        capability_id.startswith("microscopic.")
        for capability_id in payload["dispatch_capability_ids_available"]
    )


def test_stateless_followup_payload_keeps_high_urgency_suggested_route() -> None:
    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E2",
                    round_id="R002",
                    source_report_id="R002:macro",
                    agent_name="macro",
                    capability_id="macro.run_dimer_packing_proxy",
                    family="geometry_precondition",
                    summary="Second-round packing proxy.",
                )
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R003:agenda_coverage",
            case_id="gate_case",
            round_id="R003",
            coverage_summary="ESIPT remains under-screened.",
            agenda_items=[
                AgendaCoverageItem(
                    label="ESIPT_PT",
                    coverage_status="under_screened",
                    why_relevant="The structural screen has not run.",
                    missing_evidence="No ESIPT motif screen.",
                    suggested_question="Should the ESIPT motif be screened?",
                    suggested_route="screen_esipt_structural_motif",
                    urgency="high",
                    boundary="Structural proxy only.",
                )
            ],
        ),
        runtime={
            "ablation": {"mode": "mechcal_wo_microscopic"},
            "incremental_mechanism_portfolio": {
                "enabled": False,
                "policy": "stateless_ranking_with_feedback_retained",
                "prior_planner_ranking_present": True,
            },
        },
    )
    registry = default_capability_registry().for_owner_agents(("macro",))

    payload = planner_module._compact_planner_payload(
        case_run=case_run,
        round_id="R003",
        stage="planner_update",
        context={"round_index": 3, "max_rounds": 30},
        capability_registry=registry,
    )

    assert payload["agenda_coverage_memo"]["agenda_items"][0][
        "suggested_route"
    ] == "screen_esipt_structural_motif"
    assert [
        request["capability_id"]
        for request in planner_module._dispatches_from_coverage_memo(
            payload_context=payload,
            round_id="R003",
            capability_registry=registry,
        )
    ] == ["macro.screen_esipt_structural_motif"]


def test_planner_normalizer_supplies_neutral_finalize_draft() -> None:
    response = {
        "portfolio": {
            "current": "RIM_RIR_RIV",
            "hypotheses": [
                {
                    "name": "RIM_RIR_RIV",
                    "confidence": 0.25,
                    "differential_priority": 0.25,
                }
            ],
        },
        "current_hypothesis": "RIM_RIR_RIV",
        "diagnosis": "No further available route is required.",
        "action": "finalize",
        "final_answer_draft": None,
    }

    normalized = planner_module._normalize_planner_llm_response(
        response,
        round_id="R003",
        existing_portfolio={
            "current": "RIM_RIR_RIV",
            "hypotheses": [
                {
                    "name": "RIM_RIR_RIV",
                    "confidence": 0.25,
                    "differential_priority": 0.25,
                }
            ],
        },
        previous_portfolio_hypotheses=[],
        allowed_evidence_ids=[],
        capability_registry=default_capability_registry().for_owner_agents(("macro",)),
        stage="planner_update",
        prompt_payload={},
    )

    decision = PlannerDecision.model_validate(normalized)
    assert decision.action == "finalize"
    assert decision.final_answer_draft


def test_planner_normalizer_finalizes_when_available_routes_are_exhausted() -> None:
    response = {
        "portfolio": {
            "current": "RIM_RIR_RIV",
            "hypotheses": [
                {
                    "name": "RIM_RIR_RIV",
                    "confidence": 0.25,
                    "differential_priority": 0.25,
                }
            ],
        },
        "current_hypothesis": "RIM_RIR_RIV",
        "diagnosis": "All capabilities available in this ablation were exhausted.",
        "action": "stop",
        "rationale": "Missing disabled-worker evidence remains unresolved.",
    }

    normalized = planner_module._normalize_planner_llm_response(
        response,
        round_id="R004",
        existing_portfolio={
            "current": "RIM_RIR_RIV",
            "hypotheses": [
                {
                    "name": "RIM_RIR_RIV",
                    "confidence": 0.25,
                    "differential_priority": 0.25,
                }
            ],
        },
        previous_portfolio_hypotheses=[],
        allowed_evidence_ids=[],
        capability_registry=default_capability_registry().for_owner_agents(("macro",)),
        stage="planner_update",
        prompt_payload={},
    )

    decision = PlannerDecision.model_validate(normalized)
    assert decision.action == "finalize"
    assert decision.final_answer_draft


def test_planner_policy_accepts_repeated_high_urgency_route_already_executed() -> None:
    registry = default_capability_registry().for_owner_agents(("macro",))
    case_run = _case_run_for_gate().touch(
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Rotor topology was already screened.",
                )
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R002:coverage",
            case_id="gate_case",
            round_id="R002",
            coverage_summary="Reviewer repeated an already executed route.",
            agenda_items=[
                AgendaCoverageItem(
                    label="RIM_RIR_RIV",
                    coverage_status="needs_follow_up",
                    why_relevant="Rotor topology is relevant.",
                    missing_evidence="Direct photophysical validation.",
                    suggested_question="Should rotor topology be screened again?",
                    suggested_route="macro.screen_rotor_torsion_topology",
                    urgency="high",
                    boundary="The route is a structural proxy.",
                )
            ],
        ),
    )
    decision = PlannerDecision(
        decision_id="R002:planner",
        round_id="R002",
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.25,
                    differential_priority=0.25,
                )
            ],
        ),
        current_hypothesis="RIM_RIR_RIV",
        confidence=0.25,
        diagnosis="Coverage is being extended with a new packing route.",
        action="dispatch",
        dispatch_requests=[
            DispatchRequest(
                dispatch_id="R002:dispatch:macro:packing",
                round_id="R002",
                agent_name="macro",
                capability_id="macro.run_dimer_packing_proxy",
                task="Collect a bounded packing proxy.",
                objective="Collect a bounded packing proxy.",
                evidence_goal_family="geometry_precondition",
                route="run_dimer_packing_proxy",
            )
        ],
    )

    assert not planner_module._ignored_high_urgency_under_screened_coverage(
        decision,
        case_run,
        capability_registry=registry,
    )


def test_planner_policy_does_not_require_disabled_high_urgency_worker_route() -> None:
    registry = default_capability_registry().for_owner_agents(("macro",))
    case_run = _case_run_for_gate().touch(
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R002:coverage",
            case_id="gate_case",
            round_id="R002",
            coverage_summary="Microscopic validation remains unavailable in this ablation.",
            agenda_items=[
                AgendaCoverageItem(
                    label="RACI_CI_ACCESS",
                    coverage_status="needs_follow_up",
                    why_relevant="A microscopic decay pathway remains unresolved.",
                    missing_evidence="Microscopic torsion-brightness evidence is unavailable.",
                    suggested_question="Can the missing microscopic route be validated later?",
                    suggested_route="microscopic.run_torsion_brightness_coupling_scan",
                    urgency="high",
                    boundary="The disabled route remains an evidence gap.",
                )
            ],
        ),
    )
    decision = PlannerDecision(
        decision_id="R002:planner",
        round_id="R002",
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.25,
                    differential_priority=0.25,
                )
            ],
        ),
        current_hypothesis="RIM_RIR_RIV",
        confidence=0.25,
        diagnosis="Collect the remaining available structural proxy.",
        action="dispatch",
        dispatch_requests=[
            DispatchRequest(
                dispatch_id="R002:dispatch:macro:packing",
                round_id="R002",
                agent_name="macro",
                capability_id="macro.run_dimer_packing_proxy",
                task="Collect a bounded packing proxy.",
                objective="Collect a bounded packing proxy.",
                evidence_goal_family="geometry_precondition",
                route="run_dimer_packing_proxy",
            )
        ],
    )

    assert not planner_module._ignored_high_urgency_under_screened_coverage(
        decision,
        case_run,
        capability_registry=registry,
    )


def _case_run_for_gate(*, with_prepared_structure: bool = True) -> CaseRun:
    artifacts = [
        ArtifactRecord(
            artifact_id="structure:prepared",
            kind="prepared_structure",
            created_by_round="R000",
            created_by_agent="structure",
            reusable_for=["macro", "microscopic"],
        )
    ] if with_prepared_structure else []
    return CaseRun(
        case_id="gate_case",
        input=CaseInput(
            case_id="gate_case",
            smiles="C1=CC=CC=C1",
            user_query="Assess mechanism.",
        ),
        status="running",
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[
                HypothesisEntry(name="unknown", confidence=0.2, status="pending")
            ],
        ),
        evidence_ledger=EvidenceLedger(case_id="gate_case"),
        artifact_manifest=ArtifactManifest(case_id="gate_case", artifacts=artifacts),
        runtime={"orchestrator": "mechcal"},
    )


def _planner_decision_for_dispatch(request: DispatchRequest) -> PlannerDecision:
    return PlannerDecision(
        decision_id="R001:planner",
        round_id="R001",
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[
                HypothesisEntry(name="unknown", confidence=0.2, status="pending")
            ],
        ),
        current_hypothesis="unknown",
        confidence=0.2,
        diagnosis="Dispatch test decision.",
        action="dispatch",
        dispatch_requests=[request],
    )


def _planner_finalize_decision(round_id: str = "R002") -> PlannerDecision:
    return PlannerDecision(
        decision_id=f"{round_id}:planner",
        round_id=round_id,
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[
                HypothesisEntry(name="unknown", confidence=0.2, status="pending")
            ],
        ),
        current_hypothesis="unknown",
        confidence=0.2,
        diagnosis="Finalize test decision.",
        action="finalize",
        final_answer_draft="Typed final draft.",
    )


def test_capability_registry_rejects_duplicate_ids() -> None:
    spec = default_capability_registry().snapshot().capabilities[0]

    with pytest.raises(ValueError, match="Duplicate capability_id"):
        CapabilityRegistry([spec, spec])


def test_capability_registry_exposes_migrated_microscopic_routes() -> None:
    capabilities = {
        item.capability_id: item for item in default_capability_registry().snapshot().capabilities
    }
    expected_routes = {
        "microscopic.list_rotatable_dihedrals": ["prepared_structure"],
        "microscopic.list_available_conformers": ["prepared_structure"],
        "microscopic.list_artifact_bundles": ["amesp_baseline_bundle"],
        "microscopic.list_artifact_bundle_members": ["amesp_baseline_bundle"],
        "microscopic.run_baseline_bundle": ["prepared_structure"],
        "microscopic.run_bright_dark_state_ordering": ["prepared_structure"],
        "microscopic.run_frontier_orbital_partition": ["prepared_structure"],
        "microscopic.run_charge_population_panel": ["prepared_structure"],
        "microscopic.run_solvation_polarity_proxy": ["prepared_structure"],
        "microscopic.run_conformer_bundle": ["prepared_structure"],
        "microscopic.run_conformer_state_probe": ["prepared_structure"],
        "microscopic.run_torsion_snapshots": ["prepared_structure"],
        "microscopic.run_torsion_brightness_coupling_scan": ["prepared_structure"],
        "microscopic.extract_ct_descriptors_from_bundle": ["amesp_baseline_bundle"],
        "microscopic.run_targeted_state_characterization": ["amesp_baseline_bundle"],
        "microscopic.run_targeted_charge_analysis": ["amesp_baseline_bundle"],
        "microscopic.run_targeted_density_population_analysis": ["amesp_baseline_bundle"],
        "microscopic.run_targeted_transition_dipole_analysis": ["amesp_baseline_bundle"],
        "microscopic.run_targeted_charge_redistribution_analysis": ["amesp_baseline_bundle"],
        "microscopic.run_ris_state_characterization": ["amesp_baseline_bundle"],
        "microscopic.run_targeted_localized_orbital_analysis": ["amesp_baseline_bundle"],
        "microscopic.run_targeted_natural_orbital_analysis": ["amesp_baseline_bundle"],
        "microscopic.parse_snapshot_outputs": ["amesp_baseline_bundle"],
        "microscopic.extract_torsion_candidates_from_bundle": ["amesp_baseline_bundle"],
        "microscopic.extract_geometry_descriptors_from_bundle": ["amesp_baseline_bundle"],
        "microscopic.inspect_raw_artifact_bundle": ["amesp_baseline_bundle"],
        "microscopic.unsupported_excited_state_relaxation": [],
    }

    assert {
        item.capability_id
        for item in capabilities.values()
        if item.owner_agent == "microscopic"
    } == set(expected_routes)
    for capability_id, required_kinds in expected_routes.items():
        capability = capabilities[capability_id]
        assert capability.owner_agent == "microscopic"
        assert capability.tool_binding.tool_name == "run_microscopic"
        assert capability.required_artifact_kinds == required_kinds


def test_capability_registry_exposes_migrated_macro_proxy_routes() -> None:
    capabilities = {
        item.capability_id: item for item in default_capability_registry().snapshot().capabilities
    }
    expected_routes = {
        "macro.macro_structure_scan",
        "macro.screen_donor_acceptor_layout",
        "macro.screen_rotor_torsion_topology",
        "macro.screen_planarity_compactness",
        "macro.screen_intramolecular_hbond_preorganization",
        "macro.screen_conformer_geometry_proxy",
        "macro.screen_neutral_aromatic_structure",
        "macro.screen_donor_acceptor_architecture",
        "macro.screen_cationic_targeting_prior",
        "macro.screen_lipophilicity_proxy",
        "macro.screen_polar_binding_site_prior",
        "macro.screen_metal_triplet_prior",
        "macro.screen_rotor_rim_prior",
        "macro.screen_esipt_structural_motif",
        "macro.screen_aggregation_prone_scaffold",
        "macro.screen_pi_stacking_prone_geometry",
        "macro.run_dimer_packing_proxy",
        "macro.run_aggregate_contact_proxy",
        "macro.run_crystal_restriction_checklist",
        "macro.run_solid_state_emission_proxy",
        "macro.run_water_fraction_aggregation_checklist",
    }

    assert {
        item.capability_id
        for item in capabilities.values()
        if item.owner_agent == "macro"
    } == expected_routes
    for capability_id in expected_routes:
        capability = capabilities[capability_id]
        assert capability.owner_agent == "macro"
        assert capability.evidence_family == "geometry_precondition"
        assert capability.required_artifact_kinds == []
        assert capability.tool_binding.tool_name == "run_macro"
        assert "evidence_level" not in capability.metadata
        assert capability.metadata["mechanism_directness"] == "not_direct_mechanism_evidence"


def test_claim_reducer_maps_evidence_to_claims_and_debt() -> None:
    from mechcal.runtime.claims import (
        annotate_evidence_units,
        reduce_claim_ledger,
    )

    evidence = [
        _evidence_unit(
            evidence_id="R001:microscopic:baseline",
            round_id="R001",
            source_report_id="R001:microscopic",
            agent_name="microscopic",
            family="state_ordering_brightness",
            observable_tags=["run_baseline_bundle"],
            summary="Baseline attempted.",
        ),
        _evidence_unit(
            evidence_id="R002:microscopic:torsion",
            round_id="R002",
            source_report_id="R002:microscopic",
            agent_name="microscopic",
            family="torsion_sensitivity",
            observable_tags=["run_torsion_snapshots"],
            summary="Torsion snapshots attempted.",
        ),
    ]

    annotated = annotate_evidence_units(evidence)
    ledger = reduce_claim_ledger(
        "claim_case",
        EvidenceLedger(case_id="claim_case", items=annotated),
    )
    assessments = {item.claim_id: item for item in ledger.assessments}

    assert "neutral_aromatic_le_signature_screened" in annotated[0].claim_refs
    assert "rim_torsion_sensitivity_screened" in annotated[1].claim_refs
    assert (
        assessments["neutral_aromatic_le_signature_screened"].coverage_status
        == "covered"
    )
    assert assessments["rim_torsion_sensitivity_screened"].coverage_status == "covered"
    assert "ICT/TICT" in ledger.coverage_debt_hypotheses()


def test_mechanism_program_and_diagnosis_units_validate_contracts() -> None:
    program = MechanismProgram(
        program_id="MP001",
        case_id="case",
        round_id="R001",
        candidate_mechanisms=[
            CandidateMechanism(
                mechanism_id="torsion_ct_quench",
                label="Torsional CT quenching",
                mechanism_family="torsional_nonradiative_decay",
                context="single-molecule solution proxy",
            )
        ],
        evidence_questions=[
            EvidenceQuestion(
                question_id="Q001",
                mechanism_id="torsion_ct_quench",
                question="Does torsion expose a CT-like nonradiative proxy?",
                observable="torsion_ct_descriptor",
                expected_basis="computed",
                status="computable",
                acceptable_capability_ids=["microscopic.run_torsion_snapshots"],
            )
        ],
        computational_routes=[
            ComputationalRoute(
                question_id="Q001",
                capability_id="microscopic.run_torsion_snapshots",
                agent_name="microscopic",
                route="run_torsion_snapshots",
                expected_basis="computed",
            )
        ],
        scope_boundaries=[
            ScopeBoundary(
                boundary_id="B001",
                statement="Conical intersections require high-level QM beyond this route.",
            )
        ],
    )

    reloaded = MechanismProgram.model_validate(program.model_dump(mode="json"))
    assert reloaded.computational_routes[0].question_id == "Q001"

    diagnosis = DiagnosisUnit(
        diagnosis_id="D001",
        mechanism="Torsional CT quenching",
        context="single-molecule solution proxy",
        status="proxy_supported",
        evidence_refs=["R001:microscopic:torsion"],
        reasoning_summary="Computed proxy evidence supports torsion-sensitive CT behavior.",
    )
    assert DiagnosisUnit.model_validate(diagnosis.model_dump(mode="json")).status == (
        "proxy_supported"
    )

    with pytest.raises(ValidationError):
        DiagnosisUnit(
            diagnosis_id="D002",
            mechanism="Unsupported claim",
            context="test",
            status="computed_supported",
            reasoning_summary="No evidence refs are present.",
        )


def test_semantic_judge_payloads_exclude_paper_trace() -> None:
    final = DiagnosisUnit(
        diagnosis_id="D001",
        mechanism="AIE proxy",
        context="structure-only",
        status="proxy_supported",
        evidence_refs=["R001:macro:evidence"],
        reasoning_summary="Proxy evidence is available.",
    )
    case_run = _case_run_for_gate().touch(
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="R001:macro:evidence",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    family="geometry_precondition",
                    summary="Macro proxy evidence.",
                )
            ],
        ),
        diagnosis_units=[final],
    )
    hidden_reference = {
        "paper_trace": {"raw_reference_evidence_units": [{"paper_quote": "hidden"}]},
        "semantic_evidence_targets": [{"target": "Structure proxy should be present."}],
        "semantic_diagnosis_targets": [{"target": "AIE should not be overclaimed."}],
    }

    evidence_payload = build_evidence_alignment_payload(case_run, hidden_reference)
    diagnosis_payload = build_diagnosis_alignment_payload(case_run, hidden_reference)

    assert "paper_trace" not in evidence_payload
    assert "paper_trace" not in diagnosis_payload
    assert evidence_payload["semantic_evidence_targets"] == [
        {"target": "Structure proxy should be present."}
    ]
    assert diagnosis_payload["semantic_diagnosis_targets"] == [
        {"target": "AIE should not be overclaimed."}
    ]

    with pytest.raises(ValidationError):
        DiagnosisUnit(
            diagnosis_id="D003",
            mechanism="Conical intersection access",
            context="high-level QM",
            status="underdetermined",
            reasoning_summary="Missing explicit unresolved needs.",
        )


def test_failed_tool_result_blocks_matching_claim() -> None:
    from mechcal.runtime.claims import annotate_evidence_units, reduce_claim_ledger

    request = DispatchRequest(
        dispatch_id="R002-micro-001",
        round_id="R002",
        agent_name="microscopic",
        capability_id="microscopic.run_targeted_localized_orbital_analysis",
        task="Run localized orbital follow-up.",
        objective="Cover ICT localized orbital claim.",
        evidence_goal_family="charge_localization",
        route="run_targeted_localized_orbital_analysis",
    )
    failure = FailureReport(
        kind="runtime_failed",
        message="tool runtime failed",
        recoverable=True,
    )
    result = WorkerReportWriter(
        capability_registry=default_capability_registry()
    ).failed_tool_result(request, failure, error_type="RuntimeError")
    evidence = annotate_evidence_units(result.evidence_units)
    ledger = reduce_claim_ledger(
        "claim_case",
        EvidenceLedger(case_id="claim_case", items=evidence),
    )
    assessments = {item.claim_id: item for item in ledger.assessments}

    assert result.status == "failed"
    assert result.evidence_units[0].status == "failed"
    assert "ict_localized_orbital_screened" in evidence[0].claim_refs
    assert (
        assessments["ict_localized_orbital_screened"].coverage_status
        == "blocked"
    )


def test_planner_requires_llm_without_claim_debt_scientific_fallback() -> None:
    from mechcal.runtime.claims import annotate_evidence_units, reduce_claim_ledger

    evidence = annotate_evidence_units(
        [
            _evidence_unit(
                evidence_id="R001:macro:structural_prior",
                round_id="R001",
                source_report_id="R001:macro",
                agent_name="macro",
                family="geometry_precondition",
                observable_tags=["macro_structure_scan_proxy", "structural_prior"],
                summary="Macro structural prior was attempted.",
            ),
            _evidence_unit(
                evidence_id="R001:microscopic:baseline",
                round_id="R001",
                source_report_id="R001:microscopic",
                agent_name="microscopic",
                family="state_ordering_brightness",
                observable_tags=["run_baseline_bundle"],
                summary="Microscopic baseline was attempted.",
            ),
        ]
    )
    baseline_artifact = ArtifactRecord(
        artifact_id="R001:amesp_baseline_bundle",
        kind="amesp_baseline_bundle",
        created_by_round="R001",
        created_by_agent="microscopic",
        reusable_for=["microscopic", "artifact_inspection"],
    )
    case_run = _case_run_for_gate().touch(
        evidence_ledger=EvidenceLedger(case_id="gate_case", items=evidence),
        claim_ledger=reduce_claim_ledger(
            "gate_case",
            EvidenceLedger(case_id="gate_case", items=evidence),
        ),
        artifact_manifest=_case_run_for_gate().artifact_manifest.with_records(
            [baseline_artifact]
        ),
    )

    with pytest.raises(RuntimeError, match="requires enable_llm_decisions=True"):
        PlannerAgent().plan_update(
            build_planner_context(case_run, round_index=2, max_rounds=20),
            case_run,
            round_id="R002",
        )


def test_gate_blocks_unknown_capability() -> None:
    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.unknown",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )

    result = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=Path("/tmp"))
    ).deps.gate.apply(
        _planner_decision_for_dispatch(request),
        _case_run_for_gate(),
    )

    assert result.status == "retry_policy"
    assert result.terminal is True
    assert any("Unknown capability_id" in item for item in result.violations)


def test_gate_blocks_wrong_capability_owner() -> None:
    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="microscopic.run_baseline_bundle",
        task="Collect microscopic baseline evidence.",
        objective="Collect microscopic baseline evidence.",
        evidence_goal_family="state_ordering_brightness",
        route="run_baseline_bundle",
    )

    result = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=Path("/tmp"))
    ).deps.gate.apply(
        _planner_decision_for_dispatch(request),
        _case_run_for_gate(),
    )

    assert result.status == "retry_policy"
    assert any("belongs to microscopic" in item for item in result.violations)


def test_gate_allows_macro_smiles_only_fallback_without_prepared_structure() -> None:
    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )

    result = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=Path("/tmp"))
    ).deps.gate.apply(
        _planner_decision_for_dispatch(request),
        _case_run_for_gate(with_prepared_structure=False),
    )

    assert result.status == "allow"
    assert not result.violations


def test_gate_blocks_microscopic_followup_without_baseline_bundle() -> None:
    request = DispatchRequest(
        dispatch_id="R002:dispatch:microscopic",
        round_id="R002",
        agent_name="microscopic",
        capability_id="microscopic.run_targeted_charge_analysis",
        task="Run targeted charge follow-up.",
        objective="Run targeted charge follow-up.",
        evidence_goal_family="charge_localization",
        route="run_targeted_charge_analysis",
    )

    result = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=Path("/tmp"))
    ).deps.gate.apply(
        _planner_decision_for_dispatch(request),
        _case_run_for_gate(),
    )

    assert result.status == "retry_policy"
    assert any("requires artifact kind amesp_baseline_bundle" in item for item in result.violations)


def test_gate_blocks_finalize_without_required_evidence_attempts() -> None:
    result = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=Path("/tmp"))
    ).deps.gate.apply(
        _planner_finalize_decision(),
        _case_run_for_gate(),
    )

    assert result.status == "retry_policy"
    assert result.terminal is True
    assert any("Macro structural_prior" in item for item in result.violations)
    assert any("Microscopic baseline" in item for item in result.violations)


def test_gate_allows_finalize_after_required_evidence_and_followup_attempts() -> None:
    evidence = [
        _evidence_unit(
            evidence_id="R001:macro:structural_prior",
            round_id="R001",
            source_report_id="R001:macro",
            agent_name="macro",
            family="geometry_precondition",
            observable_tags=["structural_prior"],
            summary="Macro structural prior was attempted.",
        ),
        _evidence_unit(
            evidence_id="R001:microscopic:baseline",
            round_id="R001",
            source_report_id="R001:microscopic",
            agent_name="microscopic",
            family="state_ordering_brightness",
            status="unsupported",
            observable_tags=["run_baseline_bundle"],
            summary="Microscopic baseline was attempted.",
        ),
        _evidence_unit(
            evidence_id="R002:microscopic:bright_dark",
            round_id="R002",
            source_report_id="R002:microscopic",
            agent_name="microscopic",
            capability_id="microscopic.run_bright_dark_state_ordering",
            family="state_ordering_brightness",
            observable_tags=["bright_dark_state_ordering"],
            summary="A discriminating follow-up route was attempted.",
        ),
    ]
    case_run = _case_run_for_gate().touch(
        evidence_ledger=EvidenceLedger(case_id="gate_case", items=evidence)
    )

    result = MechCALOrchestrator(
        OrchestratorConfig(run_base_dir=Path("/tmp"))
    ).deps.gate.apply(
        _planner_finalize_decision(round_id="R003"),
        case_run,
    )

    assert result.status == "finalize"
    assert result.terminal is True


class FakeMicroscopicRunner:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tool_args: list[dict[str, object]] = []

    def run(
        self,
        *,
        capability_name: str,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        tool_args: dict[str, object] | None = None,
    ) -> MicroscopicRouteResult:
        del case_id
        self.tool_args.append(tool_args or {})
        profile = MICROSCOPIC_ROUTE_PROFILES[capability_name]
        manifest_path = self.tmp_path / round_id / capability_name / "bundle_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        bundle = MicroscopicArtifactBundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            status="available",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source="test_geometry",
            file_paths={"manifest": str(manifest_path), "workdir": str(manifest_path.parent)},
            parsed_observables={"state_count": 1, "available_descriptors": ["ground_state_energy"]},
        )
        manifest_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return MicroscopicRouteResult(bundle=bundle, manifest_path=manifest_path)


class PartialFailureMicroscopicRunner:
    def run(
        self,
        *,
        capability_name: str,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        tool_args: dict[str, object] | None = None,
    ) -> MicroscopicRouteResult:
        del capability_name, source_artifact, case_id, round_id, tool_args
        raise AmespMicroscopicError(
            "partial_member_series_insufficient_successes",
            "Series produced fewer than two successful members.",
            details={"successful_member_count": 1, "attempted_member_count": 4},
        )


class PartialBaselineMicroscopicRunner:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def run(
        self,
        *,
        capability_name: str,
        source_artifact: ArtifactRecord,
        case_id: str,
        round_id: str,
        tool_args: dict[str, object] | None = None,
    ) -> MicroscopicRouteResult:
        del case_id, tool_args
        profile = MICROSCOPIC_ROUTE_PROFILES[capability_name]
        manifest_path = self.tmp_path / round_id / capability_name / "bundle_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        bundle = MicroscopicArtifactBundle(
            bundle_id=f"{round_id}:{profile.artifact_kind}",
            capability_name=capability_name,
            status="partial",
            source_artifact_ids=[source_artifact.artifact_id],
            geometry_source="prepared_geometry_fixed_point",
            file_paths={"manifest": str(manifest_path), "workdir": str(manifest_path.parent)},
            parsed_observables={
                "state_count": 0,
                "available_descriptors": ["ground_state_energy"],
                "missing_descriptors": ["state_ordering", "oscillator_strength"],
            },
            missing_deliverables=[
                "S1 vertical excitation",
                "S1-derived descriptor: state_ordering",
            ],
            limitations=[
                "S1 vertical excitation unavailable; retained only successful S0 observables."
            ],
        )
        manifest_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return MicroscopicRouteResult(bundle=bundle, manifest_path=manifest_path)


def test_local_microscopic_tool_returns_typed_artifact_for_migrated_route(tmp_path) -> None:
    source_bundle = MicroscopicArtifactBundle(
        bundle_id="R001:amesp_baseline_bundle",
        capability_name="run_baseline_bundle",
        status="available",
        source_artifact_ids=["structure:prepared"],
        geometry_source="test_geometry",
        file_paths={"manifest": str(tmp_path / "baseline_manifest.json")},
    )
    baseline_artifact = ArtifactRecord(
        artifact_id=source_bundle.bundle_id,
        kind="amesp_baseline_bundle",
        created_by_round="R001",
        created_by_agent="microscopic",
        reusable_for=["microscopic", "artifact_inspection"],
        metadata={"bundle": source_bundle.model_dump(mode="json")},
    )
    case_run = _case_run_for_gate().touch(
        artifact_manifest=_case_run_for_gate().artifact_manifest.with_records([baseline_artifact])
    )
    request = DispatchRequest(
        dispatch_id="R002:dispatch:microscopic",
        round_id="R002",
        agent_name="microscopic",
        capability_id="microscopic.run_targeted_charge_analysis",
        task="Run targeted charge follow-up.",
        objective="Run targeted charge follow-up.",
        evidence_goal_family="charge_localization",
        route="run_targeted_charge_analysis",
    )

    fake_runner = FakeMicroscopicRunner(tmp_path)
    execution_plan = AgentExecutionPlan(
        plan_id="R002:dispatch:microscopic:plan",
        round_id="R002",
        agent_name="microscopic",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_steps=["run_microscopic"],
        tool_args={"source_artifact_id": baseline_artifact.artifact_id, "s1_nstates": 3},
        tool_arg_rationale="Use the existing baseline bundle and request bounded state depth.",
        parameter_adjustment_hint="Repeat runs may vary s1_nstates within capability bounds.",
        artifact_refs=[baseline_artifact.artifact_id],
        planning_mode="llm",
    )

    result = LocalMicroscopicTool(
        enable_amesp=True,
        amesp_runner=fake_runner,  # type: ignore[arg-type]
    ).run(request, case_run, execution_plan=execution_plan)

    assert result.status == "success"
    assert result.artifact_updates[0].kind == "targeted_charge_analysis_bundle"
    assert result.artifact_updates[0].metadata["bundle"]["capability_name"] == request.route
    assert result.evidence_units[0].family == "charge_localization"
    assert result.structured_results["tool_args"] == {
        "source_artifact_id": baseline_artifact.artifact_id
    }
    assert fake_runner.tool_args == [{"source_artifact_id": baseline_artifact.artifact_id}]


def test_local_microscopic_tool_keeps_partial_s0_baseline_without_s1_overclaim(tmp_path) -> None:
    request = DispatchRequest(
        dispatch_id="R001:dispatch:microscopic",
        round_id="R001",
        agent_name="microscopic",
        capability_id="microscopic.run_baseline_bundle",
        task="Collect low-cost baseline evidence.",
        objective="Collect low-cost baseline evidence.",
        evidence_goal_family="state_ordering_brightness",
        route="run_baseline_bundle",
    )

    result = LocalMicroscopicTool(
        enable_amesp=True,
        amesp_runner=PartialBaselineMicroscopicRunner(tmp_path),  # type: ignore[arg-type]
    ).run(request, _case_run_for_gate())

    assert result.status == "success"
    assert result.artifact_updates[0].status == "partial"
    assert result.evidence_units[0].status == "partial"
    assert result.evidence_units[0].support == "unresolved"
    assert "S1 vertical excitation" in result.evidence_units[0].claim
    assert "were unavailable" in result.evidence_units[0].claim
    assert "oscillator-strength evidence were unavailable" in result.evidence_units[0].claim
    assert result.structured_results["missing_deliverables"] == [
        "S1 vertical excitation",
        "S1-derived descriptor: state_ordering",
    ]


def test_local_microscopic_tool_maps_partial_amesp_error_to_schema_kind() -> None:
    request = DispatchRequest(
        dispatch_id="R001:dispatch:microscopic",
        round_id="R001",
        agent_name="microscopic",
        capability_id="microscopic.run_torsion_brightness_coupling_scan",
        task="Run torsion brightness coupling.",
        objective="Run torsion brightness coupling.",
        evidence_goal_family="torsion_sensitivity",
        route="run_torsion_brightness_coupling_scan",
    )

    result = LocalMicroscopicTool(
        enable_amesp=True,
        amesp_runner=PartialFailureMicroscopicRunner(),  # type: ignore[arg-type]
    ).run(request, _case_run_for_gate())

    assert result.status == "partial"
    assert result.failure.kind == "partial_evidence"
    assert result.failure.details["code"] == (
        "partial_member_series_insufficient_successes"
    )
    assert result.evidence_units[0].status == "partial"


def test_local_microscopic_tool_passes_bounded_series_args(tmp_path) -> None:
    request = DispatchRequest(
        dispatch_id="R001:dispatch:microscopic",
        round_id="R001",
        agent_name="microscopic",
        capability_id="microscopic.run_conformer_bundle",
        task="Run conformer series.",
        objective="Run conformer series.",
        evidence_goal_family="conformer_sensitivity",
        route="run_conformer_bundle",
    )
    execution_plan = AgentExecutionPlan(
        plan_id="R001:dispatch:microscopic:plan",
        round_id="R001",
        agent_name="microscopic",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_steps=["run_microscopic"],
        tool_args={"max_members": 99, "s1_nstates": 4, "td_tout": 2},
        tool_arg_rationale="Exercise bounded series args before tool-side clamping.",
        parameter_adjustment_hint="Repeat runs may vary max_members, s1_nstates, or td_tout.",
        planning_mode="llm",
    )
    fake_runner = FakeMicroscopicRunner(tmp_path)

    result = LocalMicroscopicTool(
        enable_amesp=True,
        amesp_runner=fake_runner,  # type: ignore[arg-type]
    ).run(request, _case_run_for_gate(), execution_plan=execution_plan)

    assert result.status == "success"
    assert result.structured_results["tool_args"] == {
        "max_members": 6,
        "s1_nstates": 4,
        "td_tout": 2,
    }
    assert fake_runner.tool_args == [result.structured_results["tool_args"]]


@pytest.mark.parametrize(
    "route",
    [
        "screen_donor_acceptor_layout",
        "screen_rotor_torsion_topology",
        "screen_planarity_compactness",
        "screen_intramolecular_hbond_preorganization",
        "screen_conformer_geometry_proxy",
        "screen_neutral_aromatic_structure",
        "screen_donor_acceptor_architecture",
        "screen_cationic_targeting_prior",
        "screen_lipophilicity_proxy",
        "screen_metal_triplet_prior",
        "screen_rotor_rim_prior",
        "screen_esipt_structural_motif",
        "screen_aggregation_prone_scaffold",
        "screen_pi_stacking_prone_geometry",
        "run_dimer_packing_proxy",
        "run_aggregate_contact_proxy",
        "run_crystal_restriction_checklist",
        "run_solid_state_emission_proxy",
        "run_water_fraction_aggregation_checklist",
    ],
)
def test_local_macro_tool_returns_structural_prior_proxy_routes(route: str) -> None:
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id=f"R001:dispatch:macro:{route}",
        round_id="R001",
        agent_name="macro",
        capability_id=f"macro.{route}",
        task=f"Collect {route} structural prior.",
        objective=f"Collect {route} structural prior.",
        evidence_goal_family="geometry_precondition",
        route=route,
    )

    result = LocalMacroTool().run(request, _case_run_for_gate())

    assert result.status == "success"
    assert result.selected_route == route
    assert "evidence_level" not in result.structured_results
    assert result.evidence_units[0].basis in {"proxy", "checklist"}
    assert str(result.structured_results["result_name"]).endswith(("_proxy", "_checklist"))
    assert {
        "structural_prior",
        "material_level_proxy",
        "material_level_checklist",
    } & set(result.evidence_units[0].observable_tags)
    assert result.evidence_units[0].relation == "neutral"
    assert result.evidence_units[0].artifact_refs == ["structure:prepared"]
    if route == "screen_rotor_torsion_topology":
        assert "rotatable bonds" in result.evidence_units[0].claim
        assert "torsion candidates" in result.evidence_units[0].claim
    if route == "screen_aggregation_prone_scaffold":
        assert "aromatic atoms" in result.evidence_units[0].claim
        assert "aggregation-prone proxy" in result.evidence_units[0].claim


def test_local_macro_tool_metal_triplet_prior_is_bounded() -> None:
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro:screen_metal_triplet_prior",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.screen_metal_triplet_prior",
        task="Collect metal/triplet structural prior.",
        objective="Collect metal/triplet structural prior.",
        evidence_goal_family="geometry_precondition",
        route="screen_metal_triplet_prior",
    )
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="triplet_prior_case",
            smiles="CCBr",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "PUBLIC_TRIPLET_PRIOR"},
        )
    )

    result = LocalMacroTool().run(request, case_run)

    assert result.status == "success"
    evidence = result.evidence_units[0]
    metrics = result.structured_results["metrics"]
    assert metrics["heavy_atom_triplet_prior"] is True
    assert "Br" in metrics["heavy_atom_symbols"]
    assert evidence.basis == "proxy"
    assert evidence.relation == "neutral"
    assert "not phosphorescence" in evidence.claim


def test_local_macro_tool_polar_binding_site_prior_is_bounded() -> None:
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro:screen_polar_binding_site_prior",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.screen_polar_binding_site_prior",
        task="Collect polar interaction structural prior.",
        objective="Collect host-guest/PET structural prior.",
        evidence_goal_family="geometry_precondition",
        route="screen_polar_binding_site_prior",
    )
    case_run = _case_run_for_gate().touch(
        input=CaseInput(
            case_id="polar_prior_case",
            smiles="CC(=O)c1cccc(C(=O)C)c1",
            user_query="Assess mechanisms.",
            metadata={"public_case_id": "PUBLIC_POLAR_PRIOR"},
        )
    )

    result = LocalMacroTool().run(request, case_run)

    assert result.status == "success"
    evidence = result.evidence_units[0]
    metrics = result.structured_results["metrics"]
    assert metrics["polar_binding_site_proxy"] is True
    assert metrics["pet_receptor_like_proxy"] is True
    assert "host_guest_interaction_prior" in evidence.observable_tags
    assert evidence.basis == "proxy"
    assert "not guest uptake" in evidence.claim
    assert "binding" in evidence.claim


def test_local_macro_tool_uses_bounded_plan_args() -> None:
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.screen_rotor_torsion_topology",
        task="Collect rotor/torsion structural prior.",
        objective="Collect rotor/torsion structural prior.",
        evidence_goal_family="geometry_precondition",
        route="screen_rotor_torsion_topology",
    )
    execution_plan = AgentExecutionPlan(
        plan_id="R001:dispatch:macro:plan",
        round_id="R001",
        agent_name="macro",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_steps=["run_macro"],
        tool_args={"focus_tags": ["torsion", "rotor"], "ignored": True},
        tool_arg_rationale="Focus the macro proxy scan on rotor and torsion tags.",
        parameter_adjustment_hint="Repeat runs may vary focus_tags within the macro route.",
        planning_mode="llm",
    )

    result = LocalMacroTool().run(
        request,
        _case_run_for_gate(),
        execution_plan=execution_plan,
    )

    assert result.structured_results["tool_args"] == {"focus_tags": ["torsion", "rotor"]}
    assert result.structured_results["metrics"]["focus_tags"] == ["torsion", "rotor"]


class FakeJsonClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def is_configured(self) -> bool:
        return True

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, object],
        schema_feedback: str | None = None,
    ) -> dict[str, object]:
        del schema_feedback
        self.payloads.append(payload)
        if payload.get("task") == "agenda_coverage_review":
            return {
                "coverage_summary": (
                    "Coverage reviewer flags rotor and ESIPT routes as "
                    "mechanism-screening work, without ranking mechanisms."
                ),
                "agenda_items": [
                    {
                        "label": "RIM_RIR_RIV",
                        "coverage_status": "needs_follow_up",
                        "why_relevant": (
                            "Rotor evidence remains relevant to public runtime "
                            "screening."
                        ),
                        "missing_evidence": "Need bounded rotor or packing proxy.",
                        "suggested_question": (
                            "Does public runtime evidence bound rotor restriction?"
                        ),
                        "suggested_route": "macro.screen_rotor_torsion_topology",
                        "urgency": "medium",
                        "boundary": (
                            "Coverage feedback is not a final diagnosis or ranking."
                        ),
                    }
                ],
                "screened_out": [
                    {
                        "label": "TRIPLET_METAL_ENERGY_TRANSFER",
                        "reason": "No public metal or triplet motif is in scope.",
                    }
                ],
                "low_margin_competitions": [],
                "recommended_next_routes": [
                    {
                        "route": "macro.screen_rotor_torsion_topology",
                        "targets": ["RIM_RIR_RIV"],
                        "reason": "This route improves rotor coverage.",
                        "capability_ids": ["macro.screen_rotor_torsion_topology"],
                    }
                ],
            }
        if "MechanismAgenda" in system_prompt:
            capability_cards = payload["capability_cards"]
            assert isinstance(capability_cards, list)
            capability_ids = [
                str(item["capability_id"])
                for item in capability_cards
                if isinstance(item, dict)
            ]
            selected = [
                item
                for item in (
                    "macro.screen_rotor_torsion_topology",
                    "macro.screen_esipt_structural_motif",
                    "microscopic.run_baseline_bundle",
                )
                if item in capability_ids
            ]
            return {
                "candidate_mechanisms": [
                    {
                        "mechanism_id": "rim_proxy",
                        "label": "RIM/RIR proxy agenda",
                        "mechanism_family": "rim",
                        "context": "Agenda item derived from supplied SMILES/evidence.",
                        "rationale": "Runtime evidence should bound rotor restriction.",
                    }
                ],
                "evidence_questions": [
                    {
                        "question_id": "Q001",
                        "mechanism_id": "rim_proxy",
                        "question": "Does available evidence bound rotor restriction?",
                        "observable": "rotor_restriction_proxy",
                        "expected_basis": "proxy",
                        "status": "computable",
                        "acceptable_capability_ids": selected[:2],
                    }
                ],
                "computational_routes": [
                    {
                        "question_id": "Q001",
                        "capability_id": capability_id,
                        "expected_basis": "proxy",
                    }
                    for capability_id in selected[:2]
                ],
                "scope_boundaries": [
                    {
                        "boundary_id": "private_answer_key_absent",
                        "statement": (
                            "Final mechanism claims require Planner synthesis from "
                            "runtime evidence only."
                        ),
                    }
                ],
            }
        if "PhotophysicsArbiter" in system_prompt or "SupportAuditor" in system_prompt:
            allowed_ids = payload.get("allowed_evidence_ids")
            assert isinstance(allowed_ids, list)
            evidence_id = str(allowed_ids[0]) if allowed_ids else "E1"
            allowed_support_ids = payload.get("allowed_support_ids")
            assert isinstance(allowed_support_ids, list)
            support_id = str(allowed_support_ids[0]) if allowed_support_ids else ""
            return {
                "review_id": f"{payload['public_case']['round_id']}:review",  # type: ignore[index]
                "case_id": "runtime_case",
                "round_id": payload["public_case"]["round_id"],  # type: ignore[index]
                "coverage_axes": [
                    {
                        "axis_id": "rim_axis",
                        "label": "RIM/torsion evidence boundary",
                        "status": "covered",
                        "evidence_refs": [evidence_id],
                        "rationale": "Runtime evidence provides a proxy review axis.",
                        "recommended_routes": [],
                    }
                ],
                "hypothesis_cards": [
                    {
                        "hypothesis_id": "rim_proxy",
                        "mechanism": "RIM_RIR_RIV",
                        "status": "proxy_supported",
                        "support_evidence_refs": [evidence_id],
                        "weakening_evidence_refs": [],
                        "missing_or_unresolved": ["Needs aggregate-state measurement."],
                        "reasoning_summary": "Runtime evidence supports a bounded proxy.",
                        "scope_limits": ["No wet-lab claim."],
                        "priority": 0.8,
                    }
                ],
                "support_argument_audits": [
                    {
                        "support_id": support_id,
                        "audit_status": "accepted",
                        "recommended_support_level": "partial",
                        "audit_notes": ["Argument is bounded to proxy evidence."],
                        "boundary_patch": None,
                    }
                ] if support_id else [],
                "recommended_next_routes": [],
                "overclaim_warnings": ["Keep final claims bounded to runtime evidence."],
                "source_evidence_refs": [evidence_id],
            }
        if "MechanismCritic" in system_prompt:
            allowed_ids = payload.get("allowed_evidence_ids")
            assert isinstance(allowed_ids, list)
            payload_text = json.dumps(payload)
            assert "hidden_reference" not in payload_text
            evidence_id = str(allowed_ids[0]) if allowed_ids else ""
            rows = []
            for index, label in enumerate(MECHANISM_POOL):
                rows.append(
                    {
                        "label": label,
                        "trigger_status": "triggered"
                        if label == "RIM_RIR_RIV"
                        else "not_triggered",
                        "differential_priority": 0.8 if label == "RIM_RIR_RIV" else 0.0,
                        "support_strength": "partial"
                        if label == "RIM_RIR_RIV" and evidence_id
                        else "unsupported",
                        "claim_status": "partially_supported"
                        if label == "RIM_RIR_RIV" and evidence_id
                        else "candidate_requires_validation",
                        "positive_evidence_refs": [evidence_id]
                        if label == "RIM_RIR_RIV" and evidence_id
                        else [],
                        "negative_evidence_refs": [],
                        "missing_validation": ["Needs aggregate-state measurement."]
                        if index == 0
                        else [],
                        "rationale": "Fake critic portfolio row from runtime evidence.",
                    }
                )
            return {
                "portfolio_id": "fake_portfolio",
                "case_id": "runtime_case",
                "round_id": payload["public_case"]["round_id"],  # type: ignore[index]
                "rows": rows,
                "evidence_attributions": [
                    {
                        "evidence_id": evidence_id,
                        "updates": [
                            {
                                "label": "RIM_RIR_RIV",
                                "priority_effect": "increase",
                                "support_effect": "supports",
                                "warrant": "Runtime evidence partially supports RIM.",
                                "boundary": "Aggregate-state validation remains missing.",
                            }
                        ],
                    }
                ] if evidence_id else [],
                "policy_notes": ["Runtime evidence only."],
            }
        if "ConclusionLedgerAgent" in system_prompt:
            allowed_ids = payload.get("allowed_evidence_ids")
            assert isinstance(allowed_ids, list)
            evidence_id = str(allowed_ids[0]) if allowed_ids else "E1"
            return {
                "evidence_conclusions": [
                    {
                        "conclusion_id": "EC001",
                        "statement": "Runtime evidence supports a bounded RIM proxy.",
                        "context": "Conclusion ledger from accepted evidence.",
                        "mechanism_family": "RIM/RIR",
                        "direction": "supports",
                        "basis": "proxy",
                        "source_evidence_refs": [evidence_id],
                        "source_snippets": [],
                        "limits": ["No wet-lab PL claim."],
                        "confidence": "medium",
                    }
                ],
                "diagnosis_conclusions": [
                    {
                        "conclusion_id": "DC001",
                        "mechanism": "RIM/RIR proxy mechanism",
                        "status": "proxy_supported",
                        "statement": "RIM/RIR remains proxy-supported by runtime evidence.",
                        "reasoning_summary": "Accepted evidence supports a bounded proxy.",
                        "source_evidence_refs": [evidence_id],
                        "source_snippets": [],
                        "missing_or_unresolved": ["Needs aggregate-state measurement."],
                        "scope_limits": ["No wet-lab claim."],
                        "confidence": "medium",
                    }
                ],
                "pending_questions": [
                    {
                        "question_id": "PQ001",
                        "question": "Aggregate-state photophysics remains unresolved.",
                        "needed_evidence": ["Aggregate-state PL or related measurement."],
                        "source_evidence_refs": [evidence_id],
                    }
                ],
                "policy_notes": ["Runtime evidence only."],
            }
        if payload.get("stage") == "planner_initial_verdict":
            evidence = payload.get("evidence")
            evidence_refs = [
                str(item.get("id"))
                for item in evidence
                if isinstance(item, dict) and item.get("id")
            ] if isinstance(evidence, list) else []
            next_ids = payload.get("next_capability_ids_available")
            next_ids = next_ids if isinstance(next_ids, list) else []
            return {
                "ranked_candidates": [
                    {
                        "label": "RIM_RIR_RIV",
                        "priority": 0.55,
                        "support": "weak_or_proxy",
                        "status": "candidate_requires_validation",
                        "evidence_refs": evidence_refs[:2],
                        "reason": "Runtime evidence provides bounded rotor evidence.",
                    }
                ],
                "action": "dispatch",
                "next_capability_ids": [str(next_ids[0])] if next_ids else [],
            }
        if "Initial Evidence Ranking Planner" in system_prompt:
            round_id = str(payload["round_id"])
            evidence = payload.get("evidence")
            evidence_refs = [
                str(item.get("evidence_id"))
                for item in evidence
                if isinstance(item, dict) and item.get("evidence_id")
            ] if isinstance(evidence, list) else []
            return {
                "decision_id": f"{round_id}:planner_initial_evidence_ranking",
                "round_id": round_id,
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": None,
                    "hypotheses": [],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "runner_up_hypothesis": None,
                "confidence": 0.55,
                "diagnosis": (
                    "Fake compact initial ranking from runtime evidence and agenda "
                    "coverage."
                ),
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Need follow-up evidence."],
                "priority_deltas": [
                    {
                        "label": "RIM_RIR_RIV",
                        "previous_priority": 0.0,
                        "priority_delta": 0.55,
                        "new_priority": 0.55,
                        "delta_reason": "Runtime evidence provides bounded rotor evidence.",
                        "evidence_refs_added": evidence_refs[:2],
                        "support_change": "newly_supported",
                    }
                ],
                "mechanism_support_arguments": [
                    {
                        "support_id": f"{round_id}:support:001",
                        "target_label": "RIM_RIR_RIV",
                        "support_level": "partial",
                        "finding": "Runtime observations include bounded rotor evidence.",
                        "warrant": "Rotor observations screen motion-related mechanisms.",
                        "boundary": "No wet-lab or packing restriction evidence.",
                        "observation_refs": evidence_refs[:2],
                    }
                ] if evidence_refs else [],
                "final_answer_draft": None,
                "rationale": (
                    "Continue follow-up after compact initial ranking while "
                    "respecting agenda coverage."
                ),
                "raw_response": {},
            }
        if "PlannerDecision" in system_prompt and "output_contract" in payload:
            portfolio = payload["previous_mechanism_portfolio_state"]
            assert isinstance(portfolio, dict)
            capability_cards = payload["capability_cards"]
            assert isinstance(capability_cards, list)
            round_id = str(payload["round_id"])
            recent_evidence = payload.get("new_round_evidence")
            evidence_refs = [
                str(item.get("evidence_id"))
                for item in recent_evidence
                if isinstance(item, dict) and item.get("evidence_id")
            ] if isinstance(recent_evidence, list) else []
            support_arguments = [
                {
                    "support_id": f"{round_id}:support:001",
                    "target_label": "RIM_RIR_RIV",
                    "support_level": "partial",
                    "finding": "Runtime observations include bounded rotor evidence.",
                    "warrant": (
                        "Rotor observations partially support RIM/RIR/RIV because "
                        "intramolecular motion can contribute to nonradiative loss."
                    ),
                    "boundary": (
                        "This remains proxy support because aggregate-state "
                        "restriction was not measured."
                    ),
                    "observation_refs": evidence_refs[:2],
                }
            ] if evidence_refs else []
            if round_id == "R001":
                planner_portfolio = portfolio
                current_hypothesis = "unknown"
            else:
                planner_portfolio = {
                    "current": "RIM_RIR_RIV",
                    "runner_up": None,
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.55,
                            "status": "plausible",
                            "rationale": (
                                "Fake LLM portfolio grounded in runtime evidence."
                            ),
                            "evidence_refs": evidence_refs[:4],
                        }
                    ],
                }
                current_hypothesis = "RIM_RIR_RIV"
            route_by_round = {
                "R002": "macro.screen_rotor_torsion_topology",
                "R003": "macro.screen_esipt_structural_motif",
            }
            if round_id == "R001":
                capability_ids = [
                    "macro.macro_structure_scan",
                    "microscopic.run_baseline_bundle",
                ]
            else:
                capability_id = route_by_round.get(round_id)
                capability_ids = [capability_id] if capability_id else []
            if not capability_ids:
                return {
                    "decision_id": f"{round_id}:planner_final",
                    "round_id": round_id,
                    "portfolio": planner_portfolio,
                    "current_hypothesis": current_hypothesis,
                    "confidence": 0.5,
                    "diagnosis": "Fake LLM planner finalized from runtime evidence.",
                    "action": "finalize",
                    "dispatch_requests": [],
                    "unresolved_gaps": ["Aggregate-state measurement remains unavailable."],
                    "mechanism_support_arguments": support_arguments,
                    "final_answer_draft": (
                        "Runtime evidence supports a bounded RIM/RIR proxy mechanism "
                        "with unresolved aggregate-state confirmation."
                    ),
                    "raw_response": {"prompt_name": "planner_decision.md"},
                }
            selected_cards = [
                item
                for item in capability_cards
                if item["capability_id"] in set(capability_ids)
            ]
            return {
                "decision_id": f"{round_id}:planner_llm",
                "round_id": round_id,
                "portfolio": planner_portfolio,
                "current_hypothesis": current_hypothesis,
                "confidence": 0.2,
                "diagnosis": "Fake LLM planner selected one registry capability.",
                "action": "dispatch",
                "dispatch_requests": [
                    {
                        "dispatch_id": (
                            f"{round_id}:dispatch:{capability_card['owner_agent']}:"
                            f"{capability_card['route']}"
                        ),
                        "round_id": round_id,
                        "agent_name": capability_card["owner_agent"],
                        "capability_id": capability_card["capability_id"],
                        "task": "Collect macro evidence.",
                        "objective": "Collect macro evidence.",
                        "evidence_goal_family": capability_card["evidence_family"],
                        "route": capability_card["route"],
                    }
                    for capability_card in selected_cards
                ],
                "mechanism_support_arguments": support_arguments,
                "raw_response": {"prompt_name": "planner_decision.md"},
            }
        if "tool_execution_result" in payload:
            tool_result = payload["tool_execution_result"]
            assert isinstance(tool_result, dict)
            return {
                "planner_readable_report": (
                    f"{tool_result['agent_name']} LLM report summarized typed "
                    "tool observations."
                ),
                "policy_notes": ["Fake LLM report writer used ToolExecutionResult."],
            }
        request = payload["dispatch_request"]
        assert isinstance(request, dict)
        capability = payload["capability_card"]
        assert isinstance(capability, dict)
        return {
            "plan_id": f"{request['dispatch_id']}:plan",
            "round_id": request["round_id"],
            "agent_name": request["agent_name"],
            "dispatch_id": request["dispatch_id"],
            "capability_id": request["capability_id"],
            "selected_route": request["route"],
            "tool_steps": [capability["tool_name"]],
            "tool_args": {"focus_tags": ["topology_proxy"]},
            "tool_arg_rationale": "Focused macro proxy scan on topology descriptors.",
            "parameter_adjustment_hint": "Repeated use can vary focus_tags only.",
            "artifact_refs": [],
            "precondition_status": "satisfied",
            "planning_mode": "llm",
            "notes": ["Fake LLM selected the capability-bound worker tool."],
        }


def test_worker_execution_plan_can_use_llm_capability_card() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    fake_client = FakeJsonClient()
    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
        llm_client=fake_client,  # type: ignore[arg-type]
        enable_llm_planning=True,
    )

    plan = agent.plan(request, _case_run_for_gate())

    assert plan.planning_mode == "llm"
    assert plan.capability_id == "macro.macro_structure_scan"
    assert plan.artifact_refs == []
    assert plan.tool_args == {"focus_tags": ["topology_proxy"]}
    assert plan.tool_arg_rationale == "Focused macro proxy scan on topology descriptors."
    capability_card = fake_client.payloads[0]["capability_card"]
    assert capability_card
    assert capability_card["required_artifact_kinds"] == []


def test_worker_execution_plan_normalizes_dict_artifact_refs() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    class ArtifactRefDictClient(FakeJsonClient):
        def complete_json(self, **kwargs) -> dict[str, object]:  # type: ignore[no-untyped-def]
            result = super().complete_json(**kwargs)
            result["artifact_refs"] = [{"artifact_id": "structure:prepared"}]
            return result

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
        llm_client=ArtifactRefDictClient(),  # type: ignore[arg-type]
        enable_llm_planning=True,
    )

    plan = agent.plan(request, _case_run_for_gate())

    assert plan.artifact_refs == ["structure:prepared"]


def test_worker_execution_plan_allows_omitted_optional_route_args() -> None:
    from mechcal.agents import MicroscopicAgent
    from mechcal.tools.local_micro import LocalMicroscopicTool

    class NoOptionalArgsClient(FakeJsonClient):
        def complete_json(self, **kwargs) -> dict[str, object]:  # type: ignore[no-untyped-def]
            payload = kwargs["payload"]
            request = payload["dispatch_request"]
            capability = payload["capability_card"]
            assert isinstance(request, dict)
            assert isinstance(capability, dict)
            return {
                "plan_id": f"{request['dispatch_id']}:plan",
                "round_id": request["round_id"],
                "agent_name": request["agent_name"],
                "dispatch_id": request["dispatch_id"],
                "capability_id": request["capability_id"],
                "selected_route": request["route"],
                "tool_steps": [capability["tool_name"]],
                "tool_args": {},
                "tool_arg_rationale": "Use public tool defaults for optional depth args.",
                "parameter_adjustment_hint": (
                    "Repeated use can vary optional depth args if needed."
                ),
                "artifact_refs": [],
                "precondition_status": "satisfied",
                "planning_mode": "llm",
            }

    request = DispatchRequest(
        dispatch_id="R001:dispatch:microscopic",
        round_id="R001",
        agent_name="microscopic",
        capability_id="microscopic.run_baseline_bundle",
        task="Run baseline bundle.",
        objective="Run baseline bundle.",
        evidence_goal_family="state_ordering_brightness",
        route="run_baseline_bundle",
    )
    agent = MicroscopicAgent(
        LocalMicroscopicTool(),
        capability_registry=default_capability_registry(),
        llm_client=NoOptionalArgsClient(),  # type: ignore[arg-type]
        enable_llm_planning=True,
    )

    plan = agent.plan(request, _case_run_for_gate())

    assert plan.planning_mode == "llm"
    assert plan.tool_args == {}
    assert plan.artifact_refs == ["structure:prepared"]


def test_worker_execution_plan_requires_configured_llm_when_enabled() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
        enable_llm_planning=True,
    )

    with pytest.raises(RuntimeError, match="requires a configured LLM client"):
        agent.plan(request, _case_run_for_gate())


def test_execution_plan_requires_operational_planning_text() -> None:
    with pytest.raises(ValidationError):
        AgentExecutionPlan(
            plan_id="R001:dispatch:macro:plan",
            round_id="R001",
            agent_name="macro",
            dispatch_id="R001:dispatch:macro",
            capability_id="macro.macro_structure_scan",
            selected_route="macro_structure_scan",
            tool_steps=["run_macro"],
        )


def test_worker_rejects_control_fields_inside_tool_args() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )
    execution_plan = AgentExecutionPlan(
        plan_id="R001:dispatch:macro:plan",
        round_id="R001",
        agent_name="macro",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_steps=["run_macro"],
        tool_args={"route": "screen_rotor_torsion_topology"},
        tool_arg_rationale="Exercise validation of forbidden control fields in tool args.",
        parameter_adjustment_hint="Remove control fields before retrying this route.",
        planning_mode="llm",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
    )

    with pytest.raises(ValueError, match="tool_args contains control fields"):
        agent.run_with_trace(
            request,
            _case_run_for_gate(),
            execution_plan=execution_plan,
        )


def test_worker_rejects_source_artifact_id_outside_artifact_refs() -> None:
    from mechcal.agents import MicroscopicAgent
    from mechcal.tools.local_micro import LocalMicroscopicTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:microscopic",
        round_id="R001",
        agent_name="microscopic",
        capability_id="microscopic.run_solvation_polarity_proxy",
        task="Run solvation proxy.",
        objective="Run solvation proxy.",
        evidence_goal_family="state_ordering_brightness",
        route="run_solvation_polarity_proxy",
    )
    execution_plan = AgentExecutionPlan(
        plan_id="R001:dispatch:microscopic:plan",
        round_id="R001",
        agent_name="microscopic",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_steps=["run_microscopic"],
        tool_args={"source_artifact_id": "R001:amesp_baseline_bundle"},
        tool_arg_rationale="Exercise validation of source artifact selection.",
        parameter_adjustment_hint="Use an artifact id selected in artifact_refs.",
        artifact_refs=["structure:prepared"],
        planning_mode="llm",
    )
    agent = MicroscopicAgent(
        LocalMicroscopicTool(),
        capability_registry=default_capability_registry(),
    )

    with pytest.raises(ValueError, match="source_artifact_id"):
        agent.run_with_trace(
            request,
            _case_run_for_gate(),
            execution_plan=execution_plan,
        )


def test_worker_report_carries_operational_note_outside_evidence() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.screen_rotor_torsion_topology",
        task="Collect rotor/torsion structural prior.",
        objective="Collect rotor/torsion structural prior.",
        evidence_goal_family="geometry_precondition",
        route="screen_rotor_torsion_topology",
    )
    execution_plan = AgentExecutionPlan(
        plan_id="R001:dispatch:macro:plan",
        round_id="R001",
        agent_name="macro",
        dispatch_id=request.dispatch_id,
        capability_id=request.capability_id,
        selected_route=request.route,
        tool_steps=["run_macro"],
        tool_args={"focus_tags": ["torsion", "rotor"]},
        tool_arg_rationale="Focused route-local proxy scan on rotor tags.",
        parameter_adjustment_hint="Repeated use can vary focus_tags only.",
        planning_mode="llm",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
    )

    tool_result, report = agent.run_with_trace(
        request,
        _case_run_for_gate(),
        execution_plan=execution_plan,
    )

    assert report.operational_note is not None
    assert report.operational_note.tool_args == {"focus_tags": ["torsion", "rotor"]}
    assert report.operational_note.args_rationale == (
        "Focused route-local proxy scan on rotor tags."
    )
    assert report.operational_note.adjustment_hint == (
        "Repeated use can vary focus_tags only."
    )
    assert report.operational_note.note_id not in {
        item.evidence_id for item in tool_result.evidence_units
    }


def test_operational_note_rejects_scientific_confidence_language() -> None:
    from mechcal.schemas import OperationalNote

    with pytest.raises(ValidationError):
        OperationalNote(
            note_id="R001:macro:operational_note",
            source_report_id="R001:macro",
            round_id="R001",
            agent_name="macro",
            dispatch_id="R001:dispatch:macro",
            capability_id="macro.macro_structure_scan",
            selected_route="macro_structure_scan",
            args_rationale="This changes mechanism confidence.",
        )


def test_macro_llm_planning_flag_does_not_enable_other_workers(tmp_path) -> None:
    deps = MechCALOrchestrator(
        OrchestratorConfig(
            run_base_dir=tmp_path,
            enable_macro_llm_planning=True,
        )
    ).deps

    assert deps.workers["macro"].execution_planner.enable_llm is True
    assert deps.workers["microscopic"].execution_planner.enable_llm is False
    assert set(deps.workers) == {"macro", "microscopic"}


def test_microscopic_llm_planning_flag_does_not_enable_other_workers(tmp_path) -> None:
    deps = MechCALOrchestrator(
        OrchestratorConfig(
            run_base_dir=tmp_path,
            enable_microscopic_llm_planning=True,
        )
    ).deps

    assert deps.workers["macro"].execution_planner.enable_llm is False
    assert deps.workers["microscopic"].execution_planner.enable_llm is True
    assert set(deps.workers) == {"macro", "microscopic"}


def test_worker_accepts_native_tool_result() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
    )

    tool_result, report = agent.run_with_trace(request, _case_run_for_gate())

    assert isinstance(tool_result, ToolExecutionResult)
    assert tool_result.tool_result_id == report.report_id
    assert tool_result.evidence_units[0].source_report_id == report.report_id
    assert report.raw_results == tool_result.raw_results


def test_worker_report_can_use_llm_tool_result_boundary() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    fake_client = FakeJsonClient()
    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
        llm_client=fake_client,  # type: ignore[arg-type]
        enable_llm_reports=True,
    )

    tool_result, report = agent.run_with_trace(request, _case_run_for_gate())

    assert "tool_execution_result" in fake_client.payloads[0]
    payload_tool_result = fake_client.payloads[0]["tool_execution_result"]
    assert isinstance(payload_tool_result, dict)
    assert payload_tool_result["runtime_will_preserve_full_payloads"] is True
    assert "raw_results" not in payload_tool_result
    assert "structured_results" not in payload_tool_result
    assert "structured_results_summary" in payload_tool_result
    assert report.planner_readable_report == (
        "macro LLM report summarized typed tool observations."
    )
    assert report.policy_notes == ["Fake LLM report writer used ToolExecutionResult."]
    assert report.evidence_units == tool_result.evidence_units
    assert report.raw_results == tool_result.raw_results


def test_worker_report_requires_configured_llm_when_enabled() -> None:
    from mechcal.agents import MacroAgent
    from mechcal.tools.local_macro import LocalMacroTool

    request = DispatchRequest(
        dispatch_id="R001:dispatch:macro",
        round_id="R001",
        agent_name="macro",
        capability_id="macro.macro_structure_scan",
        task="Collect macro evidence.",
        objective="Collect macro evidence.",
        evidence_goal_family="geometry_precondition",
        route="macro_structure_scan",
    )
    agent = MacroAgent(
        LocalMacroTool(),
        capability_registry=default_capability_registry(),
        enable_llm_reports=True,
    )

    with pytest.raises(RuntimeError, match="requires a configured LLM client"):
        agent.run_with_trace(request, _case_run_for_gate())


def test_planner_initial_coverage_dispatch_does_not_call_llm() -> None:
    from mechcal.agents import PlannerAgent

    fake_client = FakeJsonClient()
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=fake_client,  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )
    case_run = _case_run_for_gate().touch(
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R001:agenda_coverage",
            case_id="gate_case",
            round_id="R001",
            coverage_summary="Initial route coverage only.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="microscopic.run_baseline_bundle",
                    targets=["RADIATIVE_RATE_STATE_BALANCE"],
                    reason="Collect low-cost state proxy before ranking.",
                    capability_ids=["microscopic.run_baseline_bundle"],
                )
            ],
        )
    )

    decision = planner.plan_initial(case_run, round_id="R001")

    assert decision.dispatch_requests[0].capability_id == "microscopic.run_baseline_bundle"
    assert decision.portfolio.current == "unknown"
    assert fake_client.payloads == []


def test_initial_coverage_uses_macro_routes_for_smiles_only_fallback() -> None:
    from mechcal.agents.mechanism_agenda import MechanismAgendaAgent

    case_run = _case_run_for_gate(with_prepared_structure=False).touch(
        artifact_manifest=ArtifactManifest(
            case_id="gate_case",
            artifacts=[
                ArtifactRecord(
                    artifact_id="structure:smiles_context",
                    kind="smiles_structure_context",
                    created_by_round="R000",
                    created_by_agent="structure",
                    status="partial",
                    reusable_for=["macro"],
                )
            ],
        )
    )
    agent = MechanismAgendaAgent(
        capability_registry=default_capability_registry(),
        llm_client=FakeJsonClient(),  # type: ignore[arg-type]
    )

    coverage = agent.review_coverage(case_run, round_id="R001")

    capability_ids = {
        capability_id
        for route in coverage.recommended_next_routes
        for capability_id in route.capability_ids
    }
    assert "macro.screen_donor_acceptor_layout" in capability_ids
    assert "macro.screen_rotor_torsion_topology" in capability_ids
    assert "macro.screen_aggregation_prone_scaffold" in capability_ids
    assert "microscopic.run_baseline_bundle" not in capability_ids


def test_metal_triplet_prior_does_not_screen_radiative_rate() -> None:
    guide_by_capability = {
        item["capability_id"]: item
        for item in compact_ability_evidence_guide(default_capability_registry())
    }

    screens = guide_by_capability["macro.screen_metal_triplet_prior"]["screens"]

    assert screens == ["TRIPLET_METAL_ENERGY_TRANSFER"]


def test_planner_llm_payload_includes_compact_public_capability_cards() -> None:
    from mechcal.agents.planner import _compact_planner_payload

    payload = _compact_planner_payload(
        case_run=_case_run_for_gate(),
        round_id="R002",
        stage="planner_update",
        context={"round_index": 2, "max_rounds": 30},
        capability_registry=default_capability_registry(),
    )

    assert payload["capability_cards"]
    assert payload["output_contract"]
    assert payload["mechanism_triage_contract"]
    assert payload["differential_ranking_guidance"]
    assert payload["incremental_update_contract"]
    guide = payload["ability_evidence_guide"]
    assert guide
    guide_by_capability = {item["capability_id"]: item for item in guide}
    assert "RACI_CI_ACCESS" in guide_by_capability[
        "microscopic.run_torsion_brightness_coupling_scan"
    ]["screens"]
    assert "RADIATIVE_RATE_STATE_BALANCE" in guide_by_capability[
        "microscopic.run_bright_dark_state_ordering"
    ]["screens"]
    previous_state = payload["previous_mechanism_portfolio_state"]
    assert isinstance(previous_state, dict)
    previous_hypotheses = previous_state["hypotheses"]
    assert isinstance(previous_hypotheses, list)
    if previous_hypotheses:
        previous_labels = {str(item["name"]) for item in previous_hypotheses}
        assert set(MECHANISM_POOL).issubset(previous_labels)
    else:
        assert set(MECHANISM_POOL).issubset(set(payload["mechanism_pool"]))
        assert previous_state["initial_state_note"]
    assert "planner_decision_schema" not in payload
    encoded_payload = json.dumps(payload, ensure_ascii=False)
    for forbidden in (
        "hidden_reference",
        "reference_mechanisms",
        "semantic_evidence_targets",
        "semantic_diagnosis_targets",
        "benchmark target",
    ):
        assert forbidden not in encoded_payload


def test_ability_evidence_guide_is_public_and_boundary_only() -> None:
    guide = ability_evidence_guide(default_capability_registry())

    assert guide
    by_capability = {item["capability_id"]: item for item in guide}
    assert by_capability["microscopic.run_baseline_bundle"][
        "suggested_claim_strength"
    ] == "weak_or_proxy"
    assert "RADIATIVE_RATE_STATE_BALANCE" in by_capability[
        "microscopic.run_baseline_bundle"
    ]["useful_for_screening"]
    assert "SOKR_ANTI_KASHA" not in by_capability[
        "microscopic.run_baseline_bundle"
    ]["useful_for_screening"]
    assert "SOKR_ANTI_KASHA" in by_capability[
        "microscopic.run_bright_dark_state_ordering"
    ]["useful_for_screening"]
    assert "HOST_GUEST_INTERACTION" in by_capability[
        "macro.screen_polar_binding_site_prior"
    ]["useful_for_screening"]
    assert "PET_ET" in by_capability[
        "macro.screen_polar_binding_site_prior"
    ]["useful_for_screening"]
    assert by_capability["macro.screen_polar_binding_site_prior"]["evidence_tier"] == (
        "structural_trigger"
    )
    assert by_capability["macro.screen_metal_triplet_prior"]["evidence_tier"] == (
        "mechanism_specific_structural_trigger"
    )
    assert by_capability["microscopic.run_baseline_bundle"]["evidence_tier"] == (
        "computed_direct_proxy"
    )
    assert "ICT_TICT_CT" not in by_capability[
        "microscopic.run_baseline_bundle"
    ]["useful_for_screening"]
    assert "RACI_CI_ACCESS" in by_capability[
        "microscopic.unsupported_excited_state_relaxation"
    ]["useful_for_screening"]
    assert "outside current scope" in by_capability[
        "microscopic.unsupported_excited_state_relaxation"
    ]["support_boundary"]
    encoded = json.dumps(guide, ensure_ascii=False).lower()
    for forbidden in (
        "hidden_reference",
        "reference_mechanisms",
        "semantic_evidence_targets",
        "semantic_diagnosis_targets",
        "benchmark target",
        "top-3",
        "final diagnosis",
    ):
        assert forbidden not in encoded


def test_compact_ability_evidence_guide_is_public_and_short() -> None:
    guide = compact_ability_evidence_guide(default_capability_registry())

    assert guide
    by_capability = {item["capability_id"]: item for item in guide}
    torsion = by_capability["microscopic.run_torsion_brightness_coupling_scan"]
    assert "RACI_CI_ACCESS" in torsion["screens"]
    assert torsion["support_ceiling"] == "weak_or_proxy"
    assert torsion["evidence_tier"] == "computed_direct_proxy"
    assert by_capability["microscopic.run_frontier_orbital_partition"][
        "evidence_tier"
    ] == "electronic_weak_trigger"
    assert by_capability["macro.run_solid_state_emission_proxy"]["evidence_tier"] == (
        "structural_trigger"
    )
    assert by_capability["microscopic.extract_ct_descriptors_from_bundle"][
        "evidence_tier"
    ] == "electronic_weak_trigger"
    assert by_capability["macro.screen_polar_binding_site_prior"]["evidence_tier"] == (
        "structural_trigger"
    )
    assert "observables" not in torsion
    assert len(json.dumps(guide, ensure_ascii=False)) < 12000
    encoded = json.dumps(guide, ensure_ascii=False).lower()
    for forbidden in (
        "hidden_reference",
        "reference_mechanisms",
        "semantic_evidence_targets",
        "semantic_diagnosis_targets",
        "benchmark target",
        "top-3",
        "final diagnosis",
    ):
        assert forbidden not in encoded


def test_planner_normalizes_llm_hypothesis_portfolio_schema_drift() -> None:
    class SchemaDriftPlannerClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R001:planner_llm",
                "round_id": "R001",
                "hypothesis_portfolio": {
                    "primary_hypothesis": "RIM/RIR proxy mechanism",
                    "secondary_hypothesis": "ESIPT",
                    "candidate_hypotheses": [
                        {
                            "mechanism": "RIM/RIR proxy mechanism",
                            "confidence": 0.62,
                            "status": "plausible",
                            "reasoning_summary": (
                                "Runtime evidence should next bound rotor restriction."
                            ),
                            "evidence_refs": [],
                        },
                        {
                            "mechanism": "ESIPT",
                            "confidence": 0.08,
                            "status": "pending",
                            "rationale": "Needs motif screening.",
                            "evidence_refs": [],
                        },
                    ],
                },
                "current_hypothesis": "unknown",
                "confidence": 0.62,
                "diagnosis": "LLM returned a drifted portfolio shape.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.macro_structure_scan"],
                "unresolved_gaps": ["Need runtime evidence."],
                "final_answer_draft": None,
                "rationale": "Collect bounded structural evidence.",
                "raw_response": {},
            }

    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=SchemaDriftPlannerClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_initial(_case_run_for_gate(), round_id="R001")

    assert decision.current_hypothesis == "RIM/RIR proxy mechanism"
    assert decision.portfolio.current == "RIM/RIR proxy mechanism"
    assert decision.runner_up_hypothesis == "ESIPT"
    assert decision.portfolio.hypotheses[0].name == "RIM/RIR proxy mechanism"
    assert decision.dispatch_requests[0].capability_id == "macro.macro_structure_scan"


def test_planner_caps_unsupported_initial_prior_without_evidence_refs() -> None:
    class UnsupportedHighPriorClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R001:planner_llm",
                "round_id": "R001",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": "ESIPT_PT",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.72,
                            "differential_priority": 0.72,
                            "evidence_support": "unsupported",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "High prior without evidence refs.",
                            "evidence_refs": [],
                        },
                        {
                            "name": "ESIPT_PT",
                            "confidence": 0.41,
                            "differential_priority": 0.41,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Supported candidate keeps priority.",
                            "evidence_refs": ["E1"],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.72,
                "diagnosis": "Initial unsupported prior should be bounded.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.macro_structure_scan"],
                "unresolved_gaps": ["Need runtime evidence."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Collect bounded evidence.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    summary="ESIPT motif proxy.",
                )
            ],
        )
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=UnsupportedHighPriorClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_initial(case_run, round_id="R001")

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RIM_RIR_RIV"].differential_priority == pytest.approx(0.2)
    assert by_label["ESIPT_PT"].differential_priority == pytest.approx(0.41)


def test_planner_initial_dispatch_uses_agenda_coverage_routes() -> None:
    class InitialPlannerClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R001:planner_llm",
                "round_id": "R001",
                "portfolio": {
                    "current": "ICT_TICT_CT",
                    "runner_up": None,
                    "hypotheses": [
                        {
                            "name": "ICT_TICT_CT",
                            "confidence": 0.5,
                            "differential_priority": 0.5,
                            "evidence_support": "unsupported",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "LLM selected D-A first.",
                            "evidence_refs": [],
                        }
                    ],
                },
                "current_hypothesis": "ICT_TICT_CT",
                "confidence": 0.5,
                "diagnosis": "Initial LLM route choice should be overridden by coverage memo.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_donor_acceptor_architecture"],
                "unresolved_gaps": ["Need balanced early coverage."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Collect D-A evidence.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="COV001",
            case_id="gate_case",
            round_id="R001",
            coverage_summary="Balanced public early coverage.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="microscopic.run_baseline_bundle",
                    targets=["RADIATIVE_RATE_STATE_BALANCE"],
                    reason="State proxy.",
                    capability_ids=["microscopic.run_baseline_bundle"],
                ),
                RecommendedAgendaRoute(
                    route="macro.screen_esipt_structural_motif",
                    targets=["ESIPT_PT"],
                    reason="ESIPT motif screen.",
                    capability_ids=["macro.screen_esipt_structural_motif"],
                ),
                RecommendedAgendaRoute(
                    route="macro.screen_aggregation_prone_scaffold",
                    targets=["PACKING_HOST_MATRIX_CONFINEMENT"],
                    reason="Aggregation/packing screen.",
                    capability_ids=["macro.screen_aggregation_prone_scaffold"],
                ),
                RecommendedAgendaRoute(
                    route="macro.screen_donor_acceptor_layout",
                    targets=["ICT_TICT_CT"],
                    reason="D-A screen.",
                    capability_ids=["macro.screen_donor_acceptor_layout"],
                ),
                RecommendedAgendaRoute(
                    route="macro.screen_rotor_torsion_topology",
                    targets=["RIM_RIR_RIV"],
                    reason="Rotor screen.",
                    capability_ids=["macro.screen_rotor_torsion_topology"],
                ),
            ],
        )
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=InitialPlannerClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_initial(case_run, round_id="R001")

    assert [request.capability_id for request in decision.dispatch_requests] == [
        "microscopic.run_baseline_bundle",
        "macro.screen_esipt_structural_motif",
        "macro.screen_aggregation_prone_scaffold",
        "macro.screen_donor_acceptor_layout",
        "macro.screen_rotor_torsion_topology",
    ]


def test_planner_update_prioritizes_first_coverage_route_for_dispatch() -> None:
    class UpdatePlannerClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R004:planner_llm",
                "round_id": "R004",
                "portfolio": {
                    "current": "ICT_TICT_CT",
                    "runner_up": "PACKING_HOST_MATRIX_CONFINEMENT",
                    "hypotheses": [
                        {
                            "name": "ICT_TICT_CT",
                            "confidence": 0.3,
                            "differential_priority": 0.3,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "D-A route remains under-screened.",
                            "evidence_refs": ["E1"],
                        },
                        {
                            "name": "PACKING_HOST_MATRIX_CONFINEMENT",
                            "confidence": 0.29,
                            "differential_priority": 0.29,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Packing route remains low-margin.",
                            "evidence_refs": ["E1"],
                        },
                    ],
                },
                "current_hypothesis": "ICT_TICT_CT",
                "confidence": 0.3,
                "diagnosis": "Planner selected CT, but coverage memo requires disambiguation.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_donor_acceptor_architecture"],
                "unresolved_gaps": ["Need low-margin coverage route."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Collect D-A evidence while respecting agenda coverage.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R003",
        round_ids=["R001", "R002", "R003"],
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    family="geometry_precondition",
                    summary="Aggregation proxy.",
                )
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="COV004",
            case_id="gate_case",
            round_id="R004",
            coverage_summary="Low-margin packing/ICT coverage.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="macro.run_solid_state_emission_proxy",
                    targets=[
                        "PACKING_HOST_MATRIX_CONFINEMENT",
                        "AGGREGATE_EXCITON_EXCIMER",
                    ],
                    reason="Disambiguate packing from aggregate/electronic proxies.",
                    capability_ids=["macro.run_solid_state_emission_proxy"],
                )
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=UpdatePlannerClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=4, max_rounds=30),
        case_run,
        round_id="R004",
    )

    assert [request.capability_id for request in decision.dispatch_requests[:2]] == [
        "macro.run_solid_state_emission_proxy",
        "macro.screen_donor_acceptor_architecture",
    ]


def test_planner_retries_placeholder_portfolio_when_context_exists() -> None:
    class PlaceholderThenValidPlannerClient(FakeJsonClient):
        def __init__(self) -> None:
            super().__init__()
            self.planner_calls = 0
            self.feedbacks: list[str | None] = []

        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            self.feedbacks.append(schema_feedback)
            self.planner_calls += 1
            base = {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "confidence": 0.4,
                "diagnosis": "Planner is using runtime evidence only.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Need torsion topology."],
                "final_answer_draft": None,
                "rationale": "Collect bounded follow-up evidence.",
                "raw_response": {},
            }
            if self.planner_calls == 1:
                return {
                    **base,
                    "portfolio": {
                        "current": "unknown",
                        "runner_up": None,
                        "hypotheses": [
                            {
                                "name": "unknown",
                                "confidence": 0.2,
                                "status": "pending",
                                "rationale": "No hypothesis supplied.",
                                "evidence_refs": [],
                            }
                        ],
                    },
                    "current_hypothesis": "unknown",
                }
            return {
                **base,
                "portfolio": {
                    "current": "RIM/RIR proxy mechanism",
                    "runner_up": None,
                    "hypotheses": [
                        {
                            "name": "RIM/RIR proxy mechanism",
                            "confidence": 0.45,
                            "status": "plausible",
                            "rationale": "Evidence context supports follow-up testing.",
                            "evidence_refs": ["E1"],
                        }
                    ],
                },
                "current_hypothesis": "RIM/RIR proxy mechanism",
            }

    case_run = _case_run_for_gate().touch(
        mechanism_program=MechanismProgram(
            program_id="MP001",
            case_id="runtime_case",
            round_id="R001",
            candidate_mechanisms=[
                CandidateMechanism(
                    mechanism_id="rim_proxy",
                    label="RIM/RIR proxy mechanism",
                    mechanism_family="rim",
                    context="agenda context from prior LLM step",
                )
            ],
        )
    )
    fake_client = PlaceholderThenValidPlannerClient()
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=fake_client,  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    assert fake_client.planner_calls == 2
    assert fake_client.feedbacks[0] is None
    assert "placeholder 'unknown' portfolio" in str(fake_client.feedbacks[1])
    assert decision.current_hypothesis == "RIM/RIR proxy mechanism"
    assert decision.dispatch_requests[0].capability_id == (
        "macro.screen_rotor_torsion_topology"
    )


def test_planner_incremental_update_preserves_previous_mechanism_rows() -> None:
    class PartialPortfolioUpdateClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RACI_CI_ACCESS",
                    "runner_up": "RIM_RIR_RIV",
                    "hypotheses": [
                        {
                            "name": "RACI_CI_ACCESS",
                            "confidence": 0.71,
                            "differential_priority": 0.71,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "plausible",
                            "rationale": "New torsion evidence raises RACI priority.",
                            "evidence_refs": ["E2"],
                            "validation_needed": ["CI search remains needed."],
                        }
                    ],
                },
                "current_hypothesis": "RACI_CI_ACCESS",
                "confidence": 0.71,
                "diagnosis": "Planner updated the previous portfolio incrementally.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Need motion-restriction follow-up."],
                "priority_deltas": [
                    {
                        "label": "RACI_CI_ACCESS",
                        "previous_priority": 0.0,
                        "priority_delta": 0.71,
                        "new_priority": 0.71,
                        "delta_reason": (
                            "New torsion-sensitive evidence E2 raises RACI as a "
                            "validation-needed candidate."
                        ),
                        "evidence_refs_added": ["E2"],
                        "support_change": "newly_supported",
                    }
                ],
                "rationale": "Collect discriminating follow-up evidence.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R002:agenda_coverage",
            case_id="gate_case",
            round_id="R002",
            coverage_summary="RACI remains under-screened.",
            agenda_items=[
                AgendaCoverageItem(
                    label="RACI_CI_ACCESS",
                    coverage_status="under_screened",
                    why_relevant="Torsion-sensitive pathway needs coverage.",
                    missing_evidence="No CI-specific proxy has been checked.",
                    suggested_question="Does torsion evidence bound CI access?",
                    suggested_route="macro.screen_rotor_torsion_topology",
                    urgency="high",
                    boundary="Coverage memo is advisory only.",
                )
            ],
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="macro.screen_rotor_torsion_topology",
                    targets=["RACI_CI_ACCESS"],
                    reason="This route covers the high-urgency item.",
                    capability_ids=["macro.screen_rotor_torsion_topology"],
                )
            ],
        ),
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            runner_up="PACKING_HOST_MATRIX_CONFINEMENT",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.5,
                    differential_priority=0.5,
                    evidence_support="partial",
                    claim_status="partially_supported_candidate",
                    status="plausible",
                    rationale="Existing rotor evidence.",
                    evidence_refs=["E1"],
                    validation_needed=["Packing restriction check."],
                ),
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.42,
                    differential_priority=0.42,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    status="pending",
                    rationale="Existing packing proxy.",
                    evidence_refs=["E1"],
                    validation_needed=["Crystal packing evidence."],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    summary="Existing rotor evidence.",
                ),
                _evidence_unit(
                    evidence_id="E2",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="torsion_sensitivity",
                    summary="New torsion-sensitive brightness proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=PartialPortfolioUpdateClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    labels = {item.name for item in decision.portfolio.hypotheses}
    assert set(MECHANISM_POOL).issubset(labels)
    rim = next(item for item in decision.portfolio.hypotheses if item.name == "RIM_RIR_RIV")
    assert rim.differential_priority == 0.5
    assert "E1" in rim.evidence_refs
    raci = next(
        item for item in decision.portfolio.hypotheses if item.name == "RACI_CI_ACCESS"
    )
    assert raci.differential_priority == 0.71
    assert raci.evidence_refs == ["E2"]
    assert decision.priority_deltas[0].label == "RACI_CI_ACCESS"
    assert decision.priority_deltas[0].new_priority == pytest.approx(0.71)
    assert decision.priority_deltas[0].evidence_refs_added == ["E2"]
    payload = planner.llm_client.payloads[0]  # type: ignore[union-attr]
    assert payload["new_round_evidence"]
    assert payload["incremental_update_contract"]
    assert payload["agenda_coverage_memo"]["agenda_items"][0]["label"] == "RACI_CI_ACCESS"


def test_planner_update_reducer_absorbs_followup_evidence_after_empty_json() -> None:
    class EmptyUpdateClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {}

    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        round_ids=["R001", "R002"],
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            runner_up="PET_ET",
            hypotheses=[
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.19,
                    differential_priority=0.19,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_da"],
                ),
                HypothesisEntry(
                    name="PET_ET",
                    confidence=0.18,
                    differential_priority=0.18,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_da"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    metrics={"donor_acceptor_proxy": 1},
                    summary="D-A structural proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_torsion",
                    round_id="R002",
                    source_report_id="R002:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_torsion_brightness_coupling_scan",
                    family="torsion_sensitivity",
                    basis="computed",
                    metrics={
                        "oscillator_strength": 0.03,
                        "state_count": 1,
                        "torsion_candidate_count": 5,
                    },
                    summary="Torsion-brightness coupling proxy was computed.",
                    claim="Torsion scan shows geometry-dependent brightness proxies.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=EmptyUpdateClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=3, max_rounds=30),
        case_run,
        round_id="R003",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RIM_RIR_RIV"].differential_priority >= 0.3
    assert by_label["RACI_CI_ACCESS"].differential_priority >= 0.27
    assert (
        by_label["RACI_CI_ACCESS"].differential_priority
        < by_label["RIM_RIR_RIV"].differential_priority
    )
    assert "E_torsion" in by_label["RACI_CI_ACCESS"].evidence_refs
    assert "E_torsion" in by_label["RIM_RIR_RIV"].evidence_refs
    assert any(delta.label == "RACI_CI_ACCESS" for delta in decision.priority_deltas)
    assert any(
        argument.target_label == "RACI_CI_ACCESS"
        and argument.observation_refs == ["E_torsion"]
        for argument in decision.mechanism_support_arguments
    )
    assert decision.raw_response["source"] == "public_update_evidence_reducer"


def test_planner_update_reducer_absorbs_high_flexibility_rotor_topology() -> None:
    class EmptyUpdateClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {}

    case_run = _case_run_for_gate().touch(
        current_round_id="R002",
        round_ids=["R001", "R002"],
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            runner_up="TRIPLET_METAL_ENERGY_TRANSFER",
            hypotheses=[
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.24,
                    differential_priority=0.24,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_ict"],
                ),
                HypothesisEntry(
                    name="TRIPLET_METAL_ENERGY_TRANSFER",
                    confidence=0.23,
                    differential_priority=0.23,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_triplet"],
                ),
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.12,
                    differential_priority=0.12,
                    evidence_support="weak_or_proxy",
                    evidence_refs=[],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_ict",
                    round_id="R001",
                    source_report_id="R001:macro:ict",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="charge_localization",
                    metrics={"donor_acceptor_proxy": 1},
                    summary="Weak donor-acceptor proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_triplet",
                    round_id="R001",
                    source_report_id="R001:macro:triplet",
                    agent_name="macro",
                    capability_id="macro.screen_metal_triplet_prior",
                    family="charge_localization",
                    metrics={"sulfur_phosphorus_triplet_prior": True},
                    summary="Triplet proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R002",
                    source_report_id="R002:macro:rotor",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    metrics={
                        "rotatable_bond_count": 1,
                        "torsion_candidate_count": 1,
                        "branch_point_count": 12,
                        "flexibility_proxy": 13.6,
                    },
                    summary="Rotor topology shows one formal rotor in a branched scaffold.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=EmptyUpdateClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=3, max_rounds=30),
        case_run,
        round_id="R003",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RIM_RIR_RIV"].differential_priority == pytest.approx(0.25)
    assert by_label["RIM_RIR_RIV"].differential_priority > by_label[
        "ICT_TICT_CT"
    ].differential_priority
    assert "E_rotor" in by_label["RIM_RIR_RIV"].evidence_refs
    assert any(delta.label == "RIM_RIR_RIV" for delta in decision.priority_deltas)


def test_planner_payload_includes_priority_hygiene_and_evidence_metrics() -> None:
    class InspectingPlannerClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RADIATIVE_RATE_STATE_BALANCE",
                    "runner_up": "ICT_TICT_CT",
                    "hypotheses": [
                        {
                            "name": "RADIATIVE_RATE_STATE_BALANCE",
                            "confidence": 0.42,
                            "differential_priority": 0.42,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "plausible",
                            "rationale": "State-balance candidate has computed proxy evidence.",
                            "evidence_refs": ["E_state"],
                        },
                        {
                            "name": "ICT_TICT_CT",
                            "confidence": 0.3,
                            "differential_priority": 0.3,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "CT has only a D-A structural trigger.",
                            "evidence_refs": ["E_da"],
                        },
                    ],
                },
                "current_hypothesis": "RADIATIVE_RATE_STATE_BALANCE",
                "confidence": 0.42,
                "diagnosis": "Planner calibrated from public runtime evidence.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.run_solid_state_emission_proxy"],
                "unresolved_gaps": ["Need material-level follow-up."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Collect bounded follow-up evidence.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        portfolio=HypothesisPortfolio(
            current="ICT_TICT_CT",
            runner_up="RADIATIVE_RATE_STATE_BALANCE",
            hypotheses=[
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.38,
                    differential_priority=0.38,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_da"],
                ),
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.34,
                    differential_priority=0.34,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_agg"],
                ),
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    confidence=0.2,
                    differential_priority=0.2,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_agg"],
                ),
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    confidence=0.31,
                    differential_priority=0.31,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_state"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    summary="State ordering and oscillator strength proxy.",
                    metrics={"oscillator_strength": 0.27, "state_count": 3},
                    observable_tags=["run_baseline_bundle"],
                ),
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_architecture",
                    family="geometry_precondition",
                    summary="D-A structural proxy.",
                    basis="proxy",
                    metrics={"donor_acceptor_proxy": 1},
                    observable_tags=["screen_donor_acceptor_architecture_proxy"],
                ),
                _evidence_unit(
                    evidence_id="E_agg",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.run_solid_state_emission_proxy",
                    family="geometry_precondition",
                    summary="Solid-state structural proxy.",
                    basis="proxy",
                    metrics={"aggregation_prone_proxy": True},
                    observable_tags=["run_solid_state_emission_proxy"],
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=InspectingPlannerClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    payload = planner.llm_client.payloads[0]  # type: ignore[union-attr]
    evidence_by_id = {
        item["evidence_id"]: item
        for item in payload["new_round_evidence"]
    }
    assert evidence_by_id["E_state"]["metrics"]["oscillator_strength"] == 0.27
    hygiene_by_label = {
        item["label"]: item for item in payload["priority_hygiene_summary"]
    }
    assert hygiene_by_label["RADIATIVE_RATE_STATE_BALANCE"]["best_evidence_tier"] == (
        "computed_direct_proxy"
    )
    assert hygiene_by_label["ICT_TICT_CT"]["best_evidence_tier"] == (
        "mechanism_specific_structural_trigger"
    )
    assert "prefer PACKING_HOST_MATRIX_CONFINEMENT" in hygiene_by_label[
        "AGGREGATE_EXCITON_EXCIMER"
    ]["calibration_note"]
    assert "more appropriate" in hygiene_by_label[
        "PACKING_HOST_MATRIX_CONFINEMENT"
    ]["calibration_note"]
    encoded = json.dumps(payload, ensure_ascii=False).lower()
    assert "hidden_reference" not in encoded
    assert "reference_mechanisms" not in encoded


def test_planner_applies_evidence_tier_priority_floor_without_support_upgrade() -> None:
    class LowTierCrowdingClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": "PACKING_HOST_MATRIX_CONFINEMENT",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.2,
                            "differential_priority": 0.2,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Generic rotor trigger.",
                            "evidence_refs": ["E_rim"],
                        },
                        {
                            "name": "ESIPT_PT",
                            "confidence": 0.1,
                            "differential_priority": 0.1,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "ESIPT motif trigger.",
                            "evidence_refs": ["E_esipt"],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.2,
                "diagnosis": "Planner returned a crowded weak-proxy ranking.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.2,
                    differential_priority=0.2,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_rim"],
                ),
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.1,
                    differential_priority=0.1,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_esipt"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_rim",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    metrics={"rotatable_bond_count": 2, "torsion_candidate_count": 2},
                    summary="Generic rotor topology proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    metrics={"esipt_motif_proxy": True},
                    summary="ESIPT structural motif proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=LowTierCrowdingClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["ESIPT_PT"].differential_priority == pytest.approx(0.23)
    assert by_label["ESIPT_PT"].evidence_support == "weak_or_proxy"
    esipt_delta = next(
        delta for delta in decision.priority_deltas if delta.label == "ESIPT_PT"
    )
    assert esipt_delta.support_change == "unchanged"
    assert "evidence-tier" in esipt_delta.delta_reason
    assert esipt_delta.evidence_refs_added == ["E_esipt"]


def test_planner_priority_floor_does_not_revive_screened_mechanism() -> None:
    class ScreenedMotifClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.2,
                            "differential_priority": 0.2,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Generic rotor trigger.",
                            "evidence_refs": ["E_rim"],
                        },
                        {
                            "name": "ESIPT_PT",
                            "confidence": 0.02,
                            "differential_priority": 0.02,
                            "evidence_support": "unsupported",
                            "claim_status": "underdetermined",
                            "status": "screened",
                            "rationale": "Coverage reviewer screened out ESIPT geometry.",
                            "evidence_refs": ["E_esipt"],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.2,
                "diagnosis": "Screened mechanism should not be revived by a motif floor.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_rim",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Generic rotor topology proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    metrics={"esipt_motif_proxy": True},
                    summary="ESIPT structural motif proxy.",
                ),
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R001:coverage",
            case_id="gate_case",
            round_id="R001",
            coverage_summary="ESIPT was screened out by geometry.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="macro.screen_rotor_torsion_topology",
                    targets=["RIM_RIR_RIV"],
                    reason="Continue rotor screening.",
                    capability_ids=["macro.screen_rotor_torsion_topology"],
                )
            ],
            screened_out=[
                {
                    "label": "ESIPT_PT",
                    "reason": "No intramolecular donor-acceptor hydrogen bond geometry.",
                }
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=ScreenedMotifClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["ESIPT_PT"].differential_priority == pytest.approx(0.02)
    esipt_deltas = [
        delta for delta in decision.priority_deltas if delta.label == "ESIPT_PT"
    ]
    assert all("evidence-tier" not in delta.delta_reason for delta in esipt_deltas)


def test_planner_priority_floor_can_pass_unresolved_high_priority_placeholders() -> None:
    class UnresolvedCrowdingClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": "ICT_TICT_CT",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.15,
                            "differential_priority": 0.15,
                            "evidence_support": "unsupported",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Broad structural placeholder.",
                            "evidence_refs": [],
                        },
                        {
                            "name": "ICT_TICT_CT",
                            "confidence": 0.15,
                            "differential_priority": 0.15,
                            "evidence_support": "unsupported",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Broad electronic placeholder.",
                            "evidence_refs": [],
                        },
                        {
                            "name": "ESIPT_PT",
                            "confidence": 0.12,
                            "differential_priority": 0.12,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "ESIPT motif trigger.",
                            "evidence_refs": ["E_esipt"],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.15,
                "diagnosis": "Planner left unresolved broad placeholders above ESIPT.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            runner_up="ICT_TICT_CT",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.15,
                    differential_priority=0.15,
                    evidence_support="unsupported",
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    confidence=0.15,
                    differential_priority=0.15,
                    evidence_support="unsupported",
                ),
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.05,
                    differential_priority=0.05,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_esipt"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    metrics={"esipt_motif_proxy": True},
                    summary="ESIPT structural motif proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=UnresolvedCrowdingClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["ESIPT_PT"].differential_priority == pytest.approx(0.23)
    assert decision.portfolio.current == "ESIPT_PT"
    esipt_delta = next(
        delta for delta in decision.priority_deltas if delta.label == "ESIPT_PT"
    )
    assert esipt_delta.evidence_refs_added == ["E_esipt"]


def test_planner_caps_weak_priority_without_runtime_evidence_refs() -> None:
    class UnsupportedHighScoreClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R001:planner_llm",
                "round_id": "R001",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": "ICT_TICT_CT",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.5,
                            "differential_priority": 0.5,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "High initial prior without cited runtime evidence.",
                            "evidence_refs": [],
                        },
                        {
                            "name": "ICT_TICT_CT",
                            "confidence": 0.45,
                            "differential_priority": 0.45,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "High initial prior without cited runtime evidence.",
                            "evidence_refs": [],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.5,
                "diagnosis": "Initial mechanism triage before evidence collection.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Need runtime evidence."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Collect evidence.",
                "raw_response": {},
            }

    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=UnsupportedHighScoreClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_initial(_case_run_for_gate(), round_id="R001")

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RIM_RIR_RIV"].differential_priority == pytest.approx(0.12)
    assert by_label["ICT_TICT_CT"].differential_priority == pytest.approx(0.12)


def test_planner_priority_floor_applies_computed_direct_proxy() -> None:
    class ComputedProxyCrowdingClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": "PACKING_HOST_MATRIX_CONFINEMENT",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.2,
                            "differential_priority": 0.2,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Generic aggregation proxy.",
                            "evidence_refs": ["E_agg"],
                        },
                        {
                            "name": "RADIATIVE_RATE_STATE_BALANCE",
                            "confidence": 0.15,
                            "differential_priority": 0.15,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Computed oscillator-strength proxy.",
                            "evidence_refs": ["E_state"],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.2,
                "diagnosis": "Planner kept a generic proxy above computed proxy evidence.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.2,
                    differential_priority=0.2,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_agg"],
                ),
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    confidence=0.0,
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_state"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_agg",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    family="geometry_precondition",
                    metrics={"aggregation_prone_proxy": True},
                    summary="Aggregation-prone scaffold proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_bright_dark_state_ordering",
                    family="state_ordering_brightness",
                    basis="computed",
                    metrics={"oscillator_strength": 0.27, "state_count": 3},
                    summary="Computed oscillator-strength proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=ComputedProxyCrowdingClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].differential_priority == pytest.approx(
        0.24
    )
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].evidence_support == "weak_or_proxy"
    assert decision.portfolio.current == "RADIATIVE_RATE_STATE_BALANCE"


def test_planner_priority_floor_keeps_bounded_near_dark_state_balance_candidate() -> None:
    class LowOscillatorClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RADIATIVE_RATE_STATE_BALANCE",
                    "hypotheses": [
                        {
                            "name": "RADIATIVE_RATE_STATE_BALANCE",
                            "confidence": 0.15,
                            "differential_priority": 0.15,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Very weak oscillator-strength proxy.",
                            "evidence_refs": ["E_state"],
                        }
                    ],
                },
                "current_hypothesis": "RADIATIVE_RATE_STATE_BALANCE",
                "confidence": 0.15,
                "diagnosis": "Computed proxy remains weak.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="RADIATIVE_RATE_STATE_BALANCE",
            hypotheses=[
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    confidence=0.0,
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_state"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    basis="computed",
                    metrics={"oscillator_strength": 0.0008, "state_count": 1},
                    summary="Near-dark oscillator-strength baseline proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=LowOscillatorClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].differential_priority == pytest.approx(
        0.18
    )
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].evidence_support == "weak_or_proxy"


def test_planner_promotes_strong_standalone_oscillator_as_weak_radiative_candidate() -> None:
    class StandaloneOscillatorClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RADIATIVE_RATE_STATE_BALANCE",
                    "hypotheses": [
                        {
                            "name": "RADIATIVE_RATE_STATE_BALANCE",
                            "confidence": 0.24,
                            "differential_priority": 0.24,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Standalone oscillator proxy.",
                            "evidence_refs": ["E_state"],
                        },
                        {
                            "name": "AGGREGATE_EXCITON_EXCIMER",
                            "confidence": 0.22,
                            "differential_priority": 0.22,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Aggregation-prone scaffold proxy.",
                            "evidence_refs": ["E_agg"],
                        },
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.21,
                            "differential_priority": 0.21,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Rotor proxy.",
                            "evidence_refs": ["E_rotor"],
                        },
                    ],
                },
                "current_hypothesis": "RADIATIVE_RATE_STATE_BALANCE",
                "confidence": 0.24,
                "diagnosis": "Standalone oscillator remains a weak candidate.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="RADIATIVE_RATE_STATE_BALANCE",
            hypotheses=[
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    confidence=0.0,
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_state"],
                ),
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.0,
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_agg"],
                ),
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.0,
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_rotor"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    basis="computed",
                    metrics={"oscillator_strength": 0.24, "state_count": 1},
                    summary="Standalone oscillator proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_agg",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    family="geometry_precondition",
                    metrics={"aggregation_prone_proxy": True, "aromatic_ring_count": 4},
                    summary="Aggregation-prone scaffold proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_rotor",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    metrics={"rotatable_bond_count": 6, "torsion_candidate_count": 6},
                    summary="Rotor proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=StandaloneOscillatorClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    labels = [item.name for item in decision.portfolio.hypotheses[:3]]
    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert labels[0] == "RADIATIVE_RATE_STATE_BALANCE"
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].differential_priority == pytest.approx(
        0.29
    )
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].evidence_support == "weak_or_proxy"


def test_esipt_context_keeps_baseline_oscillator_as_state_balance_candidate() -> None:
    class EsiptCoupledOscillatorClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "ESIPT_PT",
                    "hypotheses": [
                        {
                            "name": "ESIPT_PT",
                            "confidence": 0.28,
                            "differential_priority": 0.28,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "ESIPT motif proxy.",
                            "evidence_refs": ["E_esipt"],
                        },
                        {
                            "name": "RADIATIVE_RATE_STATE_BALANCE",
                            "confidence": 0.18,
                            "differential_priority": 0.18,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Baseline oscillator proxy in ESIPT context.",
                            "evidence_refs": ["E_state"],
                        },
                    ],
                },
                "current_hypothesis": "ESIPT_PT",
                "confidence": 0.28,
                "diagnosis": "ESIPT context should retain state-balance candidate.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="ESIPT_PT",
            hypotheses=[
                HypothesisEntry(
                    name="ESIPT_PT",
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_esipt"],
                ),
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_state"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    metrics={"esipt_motif_proxy": True},
                    summary="ESIPT motif proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    basis="computed",
                    metrics={"oscillator_strength": 0.0692, "state_count": 1},
                    summary="Baseline oscillator proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=EsiptCoupledOscillatorClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    labels = [item.name for item in decision.portfolio.hypotheses[:2]]
    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert labels == ["ESIPT_PT", "RADIATIVE_RATE_STATE_BALANCE"]
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].differential_priority == pytest.approx(
        0.24
    )


def test_planner_caps_generic_prior_below_specific_proxy_without_support_upgrade() -> None:
    class GenericPriorAboveSpecificProxyClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "AGGREGATE_EXCITON_EXCIMER",
                    "runner_up": "ESIPT_PT",
                    "hypotheses": [
                        {
                            "name": "AGGREGATE_EXCITON_EXCIMER",
                            "confidence": 0.3,
                            "differential_priority": 0.3,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Generic aromatic aggregation prior.",
                            "evidence_refs": ["E_agg"],
                        },
                        {
                            "name": "ESIPT_PT",
                            "confidence": 0.12,
                            "differential_priority": 0.12,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Mechanism-specific ESIPT motif proxy.",
                            "evidence_refs": ["E_esipt"],
                        },
                    ],
                },
                "current_hypothesis": "AGGREGATE_EXCITON_EXCIMER",
                "confidence": 0.3,
                "diagnosis": "Planner over-ranked a generic aggregation prior.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="AGGREGATE_EXCITON_EXCIMER",
            hypotheses=[
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.3,
                    differential_priority=0.3,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_agg"],
                ),
                HypothesisEntry(
                    name="ESIPT_PT",
                    confidence=0.0,
                    differential_priority=0.0,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_esipt"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_agg",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_aggregation_prone_scaffold",
                    family="geometry_precondition",
                    metrics={"aggregation_prone_proxy": True},
                    summary="Aggregation-prone scaffold proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    metrics={"esipt_motif_proxy": True},
                    summary="ESIPT structural motif proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=GenericPriorAboveSpecificProxyClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["AGGREGATE_EXCITON_EXCIMER"].differential_priority == pytest.approx(
        0.19
    )
    assert by_label["ESIPT_PT"].differential_priority == pytest.approx(0.23)
    assert by_label["ESIPT_PT"].evidence_support == "weak_or_proxy"
    assert decision.portfolio.current == "ESIPT_PT"
    aggregate_delta = next(
        delta
        for delta in decision.priority_deltas
        if delta.label == "AGGREGATE_EXCITON_EXCIMER"
    )
    assert aggregate_delta.support_change == "weakened"


def test_planner_caps_weak_polar_only_candidate_below_ct_artifact() -> None:
    class WeakPolarAboveCtClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "HOST_GUEST_INTERACTION",
                    "hypotheses": [
                        {
                            "name": "HOST_GUEST_INTERACTION",
                            "confidence": 0.25,
                            "differential_priority": 0.25,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Weak polar-site screening proxy.",
                            "evidence_refs": ["E_polar"],
                        },
                        {
                            "name": "ICT_TICT_CT",
                            "confidence": 0.24,
                            "differential_priority": 0.24,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "D-A plus CT artifact proxy.",
                            "evidence_refs": ["E_da", "E_ct"],
                        },
                    ],
                },
                "current_hypothesis": "HOST_GUEST_INTERACTION",
                "confidence": 0.25,
                "diagnosis": "Weak polar candidate should not dominate CT artifact.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="HOST_GUEST_INTERACTION",
            hypotheses=[
                HypothesisEntry(
                    name="HOST_GUEST_INTERACTION",
                    differential_priority=0.25,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_polar"],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    differential_priority=0.24,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_da", "E_ct"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_polar",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_polar_binding_site_prior",
                    family="geometry_precondition",
                    metrics={
                        "formal_charge": 0,
                        "hbd_count": 0,
                        "hba_count": 3,
                        "carbonyl_like_site_count": 3,
                        "polar_binding_site_proxy": True,
                        "host_guest_followup_trigger": True,
                        "pet_receptor_like_proxy": True,
                    },
                    summary="Weak neutral polar-site prior.",
                ),
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    metrics={"donor_acceptor_proxy": 1, "conjugation_proxy": 10.0},
                    summary="D-A structural prior.",
                ),
                _evidence_unit(
                    evidence_id="E_ct",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.extract_ct_descriptors_from_bundle",
                    family="charge_localization",
                    basis="computed",
                    metrics={"state_count": 1, "oscillator_strength": 0.015},
                    summary="CT descriptor artifact proxy.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=WeakPolarAboveCtClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["HOST_GUEST_INTERACTION"].differential_priority == pytest.approx(
        0.19
    )
    assert by_label["ICT_TICT_CT"].differential_priority == pytest.approx(0.24)
    assert decision.portfolio.current == "ICT_TICT_CT"


def test_output_assembler_caps_weak_polar_and_generic_packing_export_priority() -> None:
    case_run = _case_run_for_gate().touch(
        portfolio=HypothesisPortfolio(
            current="PACKING_HOST_MATRIX_CONFINEMENT",
            hypotheses=[
                HypothesisEntry(
                    name="PACKING_HOST_MATRIX_CONFINEMENT",
                    differential_priority=0.28,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_pack"],
                ),
                HypothesisEntry(
                    name="HOST_GUEST_INTERACTION",
                    differential_priority=0.25,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_polar"],
                ),
                HypothesisEntry(
                    name="ICT_TICT_CT",
                    differential_priority=0.24,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_ct"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_pack",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.run_solid_state_emission_proxy",
                    family="geometry_precondition",
                    metrics={"aggregation_prone_proxy": True, "rotor_burden_proxy": 1},
                    summary="Generic packing proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_polar",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_polar_binding_site_prior",
                    family="geometry_precondition",
                    metrics={
                        "formal_charge": 0,
                        "hbd_count": 0,
                        "hba_count": 3,
                        "carbonyl_like_site_count": 3,
                        "polar_binding_site_proxy": True,
                        "host_guest_followup_trigger": True,
                    },
                    summary="Weak polar-site prior.",
                ),
                _evidence_unit(
                    evidence_id="E_ct",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.extract_ct_descriptors_from_bundle",
                    family="charge_localization",
                    basis="computed",
                    metrics={"state_count": 1, "oscillator_strength": 0.015},
                    summary="CT descriptor artifact proxy.",
                ),
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [prediction.label for prediction in predictions] == [
        "ICT_TICT_CT",
        "PACKING_HOST_MATRIX_CONFINEMENT",
        "HOST_GUEST_INTERACTION",
    ]
    assert predictions[0].differential_priority == pytest.approx(0.24)
    assert predictions[1].differential_priority == pytest.approx(0.22)
    assert predictions[2].differential_priority == pytest.approx(0.19)


def test_output_assembler_caps_single_state_radiative_and_polar_pet_priority() -> None:
    case_run = _case_run_for_gate().touch(
        portfolio=HypothesisPortfolio(
            current="PET_ET",
            hypotheses=[
                HypothesisEntry(
                    name="PET_ET",
                    differential_priority=0.24,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_polar"],
                ),
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    differential_priority=0.24,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_state"],
                ),
                HypothesisEntry(
                    name="TRIPLET_METAL_ENERGY_TRANSFER",
                    differential_priority=0.23,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_triplet"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_polar",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_polar_binding_site_prior",
                    family="geometry_precondition",
                    metrics={
                        "formal_charge": 0,
                        "hbd_count": 0,
                        "hba_count": 3,
                        "carbonyl_like_site_count": 3,
                        "polar_binding_site_proxy": True,
                        "host_guest_followup_trigger": True,
                        "pet_receptor_like_proxy": True,
                    },
                    summary="Weak polar-site prior.",
                ),
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_bright_dark_state_ordering",
                    family="state_ordering_brightness",
                    basis="computed",
                    metrics={
                        "state_count": 1,
                        "oscillator_strength": 0.015,
                        "bright_state_index": 1,
                    },
                    summary="Single-state oscillator proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_triplet",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_metal_triplet_prior",
                    family="geometry_precondition",
                    metrics={"sulfur_phosphorus_triplet_prior": True},
                    summary="Sulfur triplet structural prior.",
                ),
            ],
        ),
    )

    predictions = OutputAssembler().build_mechanism_predictions(case_run)

    assert [prediction.label for prediction in predictions] == [
        "TRIPLET_METAL_ENERGY_TRANSFER",
        "PET_ET",
        "RADIATIVE_RATE_STATE_BALANCE",
    ]
    assert predictions[0].differential_priority == pytest.approx(0.23)
    assert predictions[1].differential_priority == pytest.approx(0.18)
    assert predictions[2].differential_priority == pytest.approx(0.17)


def test_initial_verdict_accepts_public_mechanism_aliases() -> None:
    class AliasInitialVerdictClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if payload.get("stage") != "planner_initial_verdict":
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            evidence = payload.get("evidence")
            evidence_refs = [
                str(item.get("id"))
                for item in evidence
                if isinstance(item, dict) and item.get("id")
            ] if isinstance(evidence, list) else []
            return {
                "mechanism_predictions": [
                    {
                        "mechanism": "ESIPT",
                        "priority": 0.4,
                        "support": "weak_or_proxy",
                        "status": "candidate_requires_validation",
                        "evidence_refs": evidence_refs[:1],
                        "reason": "ESIPT alias is grounded in runtime evidence.",
                    },
                    {
                        "mechanism": "state balance",
                        "priority": 0.35,
                        "support": "weak_or_proxy",
                        "status": "candidate_requires_validation",
                        "evidence_refs": evidence_refs[1:2],
                        "reason": "State-balance alias is grounded in runtime evidence.",
                    },
                ],
                "action": "finalize",
                "next_capability_ids": [],
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[HypothesisEntry(name=label) for label in MECHANISM_POOL],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_esipt",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_esipt_structural_motif",
                    family="geometry_precondition",
                    metrics={"esipt_motif_proxy": True},
                    summary="ESIPT structural motif proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    metrics={"oscillator_strength": 0.0692, "state_count": 1},
                    summary="Low-cost state-balance proxy.",
                ),
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R001:agenda_coverage_bootstrap",
            case_id="gate_case",
            round_id="R001",
            coverage_summary="Bootstrap coverage.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="macro.screen_esipt_structural_motif",
                    targets=["ESIPT_PT"],
                    reason="Bootstrap route already collected ESIPT proxy evidence.",
                    capability_ids=["macro.screen_esipt_structural_motif"],
                )
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=AliasInitialVerdictClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    labels = [item.name for item in decision.portfolio.hypotheses]
    assert labels[:2] == ["ESIPT_PT", "RADIATIVE_RATE_STATE_BALANCE"]
    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["ESIPT_PT"].differential_priority > 0.0
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].differential_priority > 0.0
    assert by_label["ESIPT_PT"].evidence_support == "weak_or_proxy"


def test_initial_verdict_plan_actions_are_converted_to_followup_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("MECHCAL_INITIAL_VERDICT_MODE", "llm")

    class NestedPlanInitialVerdictClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if payload.get("stage") not in {
                "planner_initial_verdict",
                "planner_initial_evidence_ranking",
            }:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            evidence_refs = [
                str(item.get("id") or item.get("evidence_id"))
                for item in payload.get("evidence", [])
                if isinstance(item, dict) and (item.get("id") or item.get("evidence_id"))
            ]
            return {
                "verdict": {
                    "differential_priority": [
                        {
                            "mechanism": "RADIATIVE_RATE_STATE_BALANCE",
                            "rationale": "Baseline bundle leaves state-balance under-screened.",
                            "evidence_refs": evidence_refs[:1],
                        },
                        {
                            "mechanism": "ICT_TICT_CT",
                            "rationale": "D-A proxy needs orbital partition follow-up.",
                            "evidence_refs": evidence_refs[1:2],
                        },
                    ]
                },
                "plan": {
                    "actions": [
                        "microscopic.run_bright_dark_state_ordering",
                        "microscopic.run_frontier_orbital_partition",
                    ]
                },
                "action": "finalize",
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[HypothesisEntry(name=label) for label in MECHANISM_POOL],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    metrics={"oscillator_strength": 0.012, "state_count": 1},
                    summary="Low-cost state-balance proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    metrics={"donor_acceptor_proxy": 1},
                    summary="D-A structural proxy.",
                ),
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R001:agenda_coverage_bootstrap",
            case_id="gate_case",
            round_id="R001",
            coverage_summary="Bootstrap coverage.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="microscopic.run_bright_dark_state_ordering",
                    targets=["RADIATIVE_RATE_STATE_BALANCE"],
                    reason="Collect state-balance follow-up.",
                    capability_ids=["microscopic.run_bright_dark_state_ordering"],
                )
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=NestedPlanInitialVerdictClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    assert decision.action == "dispatch"
    assert [request.capability_id for request in decision.dispatch_requests] == [
        "microscopic.run_bright_dark_state_ordering",
        "microscopic.run_frontier_orbital_partition",
    ]
    assert decision.final_answer_draft is None


def test_public_initial_evidence_reducer_dispatches_discriminating_followup(monkeypatch) -> None:
    monkeypatch.delenv("MECHCAL_INITIAL_VERDICT_MODE", raising=False)

    class InitialVerdictShouldNotBeCalled(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if payload.get("stage") == "planner_initial_verdict":
                raise AssertionError("Initial evidence reducer should avoid LLM call.")
            return super().complete_json(
                system_prompt=system_prompt,
                payload=payload,
                schema_feedback=schema_feedback,
            )

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[HypothesisEntry(name=label) for label in MECHANISM_POOL],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_state",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    metrics={"oscillator_strength": 0.021, "state_count": 1},
                    summary="Low-cost state-balance proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_da",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    metrics={"donor_acceptor_proxy": 1},
                    summary="D-A structural proxy.",
                ),
            ],
        ),
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R001:agenda_coverage_bootstrap",
            case_id="gate_case",
            round_id="R001",
            coverage_summary="Bootstrap coverage.",
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="microscopic.run_bright_dark_state_ordering",
                    targets=["RADIATIVE_RATE_STATE_BALANCE"],
                    reason="Collect state-balance follow-up.",
                    capability_ids=["microscopic.run_bright_dark_state_ordering"],
                )
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=InitialVerdictShouldNotBeCalled(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    assert decision.action == "dispatch"
    assert "microscopic.run_bright_dark_state_ordering" in {
        request.capability_id for request in decision.dispatch_requests
    }
    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].differential_priority > 0.0


def test_public_initial_evidence_reducer_keeps_interaction_and_triplet_candidates(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MECHCAL_INITIAL_VERDICT_MODE", raising=False)
    monkeypatch.setenv("MECHCAL_UPDATE_REDUCER_MODE", "reducer")

    class InitialVerdictShouldNotBeCalled(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if payload.get("stage") == "planner_initial_verdict":
                raise AssertionError("Initial evidence reducer should avoid LLM call.")
            return super().complete_json(
                system_prompt=system_prompt,
                payload=payload,
                schema_feedback=schema_feedback,
            )

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[HypothesisEntry(name=label) for label in MECHANISM_POOL],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_triplet",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_metal_triplet_prior",
                    family="geometry_precondition",
                    metrics={
                        "metal_or_lanthanide_prior": False,
                        "heavy_atom_triplet_prior": True,
                        "sulfur_phosphorus_triplet_prior": False,
                    },
                    summary="Heavy-atom triplet structural prior.",
                ),
                _evidence_unit(
                    evidence_id="E_interaction",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_polar_binding_site_prior",
                    family="geometry_precondition",
                    metrics={
                        "polar_binding_site_proxy": True,
                        "host_guest_followup_trigger": True,
                        "pet_receptor_like_proxy": True,
                    },
                    summary="Polar receptor structural prior.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=InitialVerdictShouldNotBeCalled(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["HOST_GUEST_INTERACTION"].differential_priority == pytest.approx(0.19)
    assert by_label["HOST_GUEST_INTERACTION"].evidence_support == "weak_or_proxy"
    assert by_label["HOST_GUEST_INTERACTION"].claim_status == "candidate_requires_validation"
    assert by_label["PET_ET"].differential_priority == pytest.approx(0.18)
    assert by_label["TRIPLET_METAL_ENERGY_TRANSFER"].differential_priority == pytest.approx(
        0.28
    )
    assert by_label["TRIPLET_METAL_ENERGY_TRANSFER"].evidence_support == "weak_or_proxy"


def test_public_update_reducer_ignores_negative_structural_screens(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MECHCAL_UPDATE_REDUCER_MODE", "reducer")

    class InitialVerdictShouldNotBeCalled(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if payload.get("stage") == "planner_initial_verdict":
                raise AssertionError("Initial evidence reducer should avoid LLM call.")
            return super().complete_json(
                system_prompt=system_prompt,
                payload=payload,
                schema_feedback=schema_feedback,
            )

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="unknown",
            hypotheses=[HypothesisEntry(name=label) for label in MECHANISM_POOL],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_da_negative",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_donor_acceptor_layout",
                    family="geometry_precondition",
                    metrics={
                        "donor_acceptor_proxy": 0,
                        "donor_acceptor_partition_proxy": 0.0,
                    },
                    summary="Donor/acceptor screen checked but not triggered.",
                ),
                _evidence_unit(
                    evidence_id="E_polar_negative",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_polar_binding_site_prior",
                    family="geometry_precondition",
                    metrics={
                        "polar_binding_site_proxy": False,
                        "host_guest_followup_trigger": False,
                        "pet_receptor_like_proxy": False,
                    },
                    summary="Polar interaction-site screen checked but not triggered.",
                ),
                _evidence_unit(
                    evidence_id="E_triplet_positive",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_metal_triplet_prior",
                    family="geometry_precondition",
                    metrics={"metal_or_lanthanide_prior": True},
                    summary="Metal/triplet structural prior.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=InitialVerdictShouldNotBeCalled(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["TRIPLET_METAL_ENERGY_TRANSFER"].differential_priority == pytest.approx(
        0.3
    )
    assert by_label["ICT_TICT_CT"].differential_priority == pytest.approx(0.0)
    assert by_label["PET_ET"].differential_priority == pytest.approx(0.0)
    assert by_label["HOST_GUEST_INTERACTION"].differential_priority == pytest.approx(0.0)


def test_initial_ranking_metrics_preserve_interaction_and_triplet_triggers() -> None:
    from mechcal.agents.planner import _initial_ranking_metrics

    metrics = _initial_ranking_metrics(
        {
            "metal_or_lanthanide_prior": False,
            "heavy_atom_triplet_prior": True,
            "sulfur_phosphorus_triplet_prior": True,
            "triplet_metal_followup_trigger": True,
            "polar_binding_site_proxy": True,
            "host_guest_followup_trigger": True,
            "pet_receptor_like_proxy": True,
            "structure_source": "prepared_structure",
        }
    )

    assert metrics["heavy_atom_triplet_prior"] is True
    assert metrics["sulfur_phosphorus_triplet_prior"] is True
    assert metrics["triplet_metal_followup_trigger"] is True
    assert metrics["host_guest_followup_trigger"] is True
    assert "structure_source" not in metrics


def test_failed_computed_route_does_not_trigger_priority_floor() -> None:
    class FailedComputedRouteClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": "RADIATIVE_RATE_STATE_BALANCE",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.2,
                            "differential_priority": 0.2,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Generic placeholder.",
                            "evidence_refs": [],
                        },
                        {
                            "name": "RADIATIVE_RATE_STATE_BALANCE",
                            "confidence": 0.17,
                            "differential_priority": 0.17,
                            "evidence_support": "unsupported",
                            "claim_status": "underdetermined",
                            "status": "pending",
                            "rationale": "Computed route failed.",
                            "evidence_refs": ["E_failed"],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.2,
                "diagnosis": "Planner should not promote failed computed evidence.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Computed route failed."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.2,
                    differential_priority=0.2,
                    evidence_support="weak_or_proxy",
                ),
                HypothesisEntry(
                    name="RADIATIVE_RATE_STATE_BALANCE",
                    confidence=0.0,
                    differential_priority=0.0,
                    evidence_support="unsupported",
                    evidence_refs=["E_failed"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_failed",
                    round_id="R001",
                    source_report_id="R001:microscopic",
                    agent_name="microscopic",
                    capability_id="microscopic.run_baseline_bundle",
                    family="state_ordering_brightness",
                    basis="checklist",
                    support="unresolved",
                    status="failed",
                    summary="Baseline bundle failed and is boundary-only status evidence.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=FailedComputedRouteClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["RADIATIVE_RATE_STATE_BALANCE"].differential_priority == pytest.approx(
        0.05
    )
    assert decision.portfolio.current == "RIM_RIR_RIV"


def test_planner_priority_floor_does_not_use_critic_positive_attribution_alone() -> None:
    class CriticAttributionClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.22,
                            "differential_priority": 0.22,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Generic rotor trigger.",
                            "evidence_refs": ["E_rim"],
                        },
                        {
                            "name": "HOST_GUEST_INTERACTION",
                            "confidence": 0.06,
                            "differential_priority": 0.06,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "pending",
                            "rationale": "Critic found a polar interaction prior.",
                            "evidence_refs": ["E_critic_only"],
                        },
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.22,
                "diagnosis": "Planner returned a generic weak-proxy ranking.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Evidence remains proxy-level."],
                "priority_deltas": [],
                "mechanism_support_arguments": [],
                "final_answer_draft": None,
                "rationale": "Continue bounded screening.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        round_ids=["R001"],
        portfolio=HypothesisPortfolio(
            current="RIM_RIR_RIV",
            hypotheses=[
                HypothesisEntry(
                    name="RIM_RIR_RIV",
                    confidence=0.22,
                    differential_priority=0.22,
                    evidence_support="weak_or_proxy",
                    evidence_refs=["E_rim"],
                ),
                    HypothesisEntry(
                        name="HOST_GUEST_INTERACTION",
                        confidence=0.06,
                        differential_priority=0.06,
                        evidence_support="weak_or_proxy",
                        evidence_refs=["E_critic_only"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E_rim",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Generic rotor topology proxy.",
                ),
                _evidence_unit(
                    evidence_id="E_critic_only",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Generic rotor topology used only by critic attribution.",
                ),
            ],
        ),
        differential_mechanism_portfolio=DifferentialMechanismPortfolio(
            portfolio_id="gate_case:R001:differential",
            case_id="gate_case",
            round_id="R001",
            rows=[
                DifferentialMechanismPortfolioRow(
                    label=label,
                    differential_priority=0.0,
                    support_strength="unsupported",
                    rationale="No reviewer priority in this unit test.",
                )
                for label in MECHANISM_POOL
            ],
            evidence_attributions=[
                EvidenceMechanismAttribution(
                    evidence_id="E_critic_only",
                    updates=[
                        EvidenceMechanismAttributionUpdate(
                            label="HOST_GUEST_INTERACTION",
                            priority_effect="increase",
                            support_effect="supports",
                            warrant="Polar interaction sites can screen host-guest hypotheses.",
                            boundary="No guest uptake or binding evidence exists.",
                        )
                    ],
                )
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=CriticAttributionClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    by_label = {item.name: item for item in decision.portfolio.hypotheses}
    assert by_label["HOST_GUEST_INTERACTION"].differential_priority == pytest.approx(0.06)
    assert by_label["HOST_GUEST_INTERACTION"].evidence_support == "weak_or_proxy"
    assert not any(
        delta.label == "HOST_GUEST_INTERACTION" for delta in decision.priority_deltas
    )
    payload = planner.llm_client.payloads[-1]  # type: ignore[union-attr]
    hygiene = {
        item["label"]: item
        for item in payload["priority_hygiene_summary"]
        if isinstance(item, dict)
    }
    assert hygiene["HOST_GUEST_INTERACTION"]["critic_positive_ref_count"] == 1


def test_planner_merges_high_urgency_coverage_route_when_ignored() -> None:
    class IgnoresCoverageClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "RIM_RIR_RIV",
                    "runner_up": "PACKING_HOST_MATRIX_CONFINEMENT",
                    "hypotheses": [
                        {
                            "name": "RIM_RIR_RIV",
                            "confidence": 0.5,
                            "differential_priority": 0.5,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "plausible",
                            "rationale": "Existing rotor proxy.",
                            "evidence_refs": ["E1"],
                        }
                    ],
                },
                "current_hypothesis": "RIM_RIR_RIV",
                "confidence": 0.5,
                "diagnosis": "Planner continues rotor follow-up.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Need more rotor context."],
                "priority_deltas": [],
                "rationale": "Collect rotor evidence.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        mechanism_agenda_coverage=MechanismAgendaCoverage(
            coverage_id="gate_case:R002:agenda_coverage",
            case_id="gate_case",
            round_id="R002",
            coverage_summary="RACI coverage is high urgency.",
            agenda_items=[
                AgendaCoverageItem(
                    label="RACI_CI_ACCESS",
                    coverage_status="under_screened",
                    why_relevant="Torsion-sensitive decay pathway is unresolved.",
                    missing_evidence="No microscopic torsion brightness scan.",
                    suggested_question="Does torsion evidence bound RACI?",
                    suggested_route="microscopic.run_torsion_brightness_coupling_scan",
                    urgency="high",
                    boundary="Coverage only; Planner owns ranking.",
                )
            ],
            recommended_next_routes=[
                RecommendedAgendaRoute(
                    route="microscopic.run_torsion_brightness_coupling_scan",
                    targets=["RACI_CI_ACCESS"],
                    reason="Disambiguates high-urgency RACI coverage.",
                    capability_ids=["microscopic.run_torsion_brightness_coupling_scan"],
                )
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.screen_rotor_torsion_topology",
                    family="geometry_precondition",
                    summary="Existing rotor evidence.",
                )
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=IgnoresCoverageClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    assert decision.action == "dispatch"
    assert [request.capability_id for request in decision.dispatch_requests[:2]] == [
        "microscopic.run_torsion_brightness_coupling_scan",
        "macro.screen_rotor_torsion_topology",
    ]
    assert decision.failure.kind == "none"


def test_planner_low_margin_carry_forward_preserves_previous_relative_order() -> None:
    class LowMarginReversalClient(FakeJsonClient):
        def complete_json(
            self,
            *,
            system_prompt: str,
            payload: dict[str, object],
            schema_feedback: str | None = None,
        ) -> dict[str, object]:
            if "PlannerDecision" not in system_prompt:
                return super().complete_json(
                    system_prompt=system_prompt,
                    payload=payload,
                    schema_feedback=schema_feedback,
                )
            self.payloads.append(payload)
            return {
                "decision_id": "R002:planner_llm",
                "round_id": "R002",
                "portfolio": {
                    "current": "AGGREGATE_EXCITON_EXCIMER",
                    "runner_up": "RACI_CI_ACCESS",
                    "hypotheses": [
                        {
                            "name": "AGGREGATE_EXCITON_EXCIMER",
                            "confidence": 0.51,
                            "differential_priority": 0.51,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "plausible",
                            "rationale": "Only a low-margin proxy shift was returned.",
                            "evidence_refs": ["E1"],
                        },
                        {
                            "name": "RACI_CI_ACCESS",
                            "confidence": 0.50,
                            "differential_priority": 0.50,
                            "evidence_support": "weak_or_proxy",
                            "claim_status": "candidate_requires_validation",
                            "status": "plausible",
                            "rationale": "Existing torsion-coupled candidate remains close.",
                            "evidence_refs": ["E1"],
                        },
                    ],
                },
                "current_hypothesis": "AGGREGATE_EXCITON_EXCIMER",
                "confidence": 0.51,
                "diagnosis": "Planner returned a low-margin reversal without new evidence.",
                "action": "dispatch",
                "selected_capability_ids": ["macro.screen_rotor_torsion_topology"],
                "unresolved_gaps": ["Need discriminating follow-up."],
                "priority_deltas": [
                    {
                        "label": "AGGREGATE_EXCITON_EXCIMER",
                        "previous_priority": 0.49,
                        "priority_delta": 0.02,
                        "new_priority": 0.51,
                        "delta_reason": "Small proxy-only adjustment.",
                        "evidence_refs_added": [],
                        "support_change": "unchanged",
                    },
                    {
                        "label": "RACI_CI_ACCESS",
                        "previous_priority": 0.50,
                        "priority_delta": 0.0,
                        "new_priority": 0.50,
                        "delta_reason": "No material change.",
                        "evidence_refs_added": [],
                        "support_change": "unchanged",
                    },
                ],
                "rationale": "Collect discriminating follow-up evidence.",
                "raw_response": {},
            }

    case_run = _case_run_for_gate().touch(
        current_round_id="R001",
        portfolio=HypothesisPortfolio(
            current="RACI_CI_ACCESS",
            runner_up="AGGREGATE_EXCITON_EXCIMER",
            hypotheses=[
                HypothesisEntry(
                    name="RACI_CI_ACCESS",
                    confidence=0.50,
                    differential_priority=0.50,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    status="plausible",
                    rationale="Existing torsion-coupled candidate.",
                    evidence_refs=["E1"],
                ),
                HypothesisEntry(
                    name="AGGREGATE_EXCITON_EXCIMER",
                    confidence=0.49,
                    differential_priority=0.49,
                    evidence_support="weak_or_proxy",
                    claim_status="candidate_requires_validation",
                    status="plausible",
                    rationale="Existing aggregate proxy candidate.",
                    evidence_refs=["E1"],
                ),
            ],
        ),
        evidence_ledger=EvidenceLedger(
            case_id="gate_case",
            items=[
                _evidence_unit(
                    evidence_id="E1",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    capability_id="macro.macro_structure_scan",
                    family="geometry_precondition",
                    summary="Existing shared proxy evidence.",
                ),
            ],
        ),
    )
    planner = PlannerAgent(
        capability_registry=default_capability_registry(),
        llm_client=LowMarginReversalClient(),  # type: ignore[arg-type]
        enable_llm_decisions=True,
    )

    decision = planner.plan_update(
        build_planner_context(case_run, round_index=2, max_rounds=30),
        case_run,
        round_id="R002",
    )

    ranked_labels = [item.name for item in decision.portfolio.sorted_hypotheses()[:2]]
    assert ranked_labels == ["RACI_CI_ACCESS", "AGGREGATE_EXCITON_EXCIMER"]
    assert decision.current_hypothesis == "RACI_CI_ACCESS"
    assert decision.raw_response["low_margin_competitions"]
    raci_delta = next(
        item for item in decision.priority_deltas if item.label == "RACI_CI_ACCESS"
    )
    assert raci_delta.ranking_stability_note is not None


def test_worker_report_rejects_mechanism_adjudication() -> None:
    with pytest.raises(ValidationError):
        AgentReport(
            report_id="R001:macro",
            round_id="R001",
            agent_name="macro",
            task_received="collect evidence",
            status="success",
            evidence_units=[
                _evidence_unit(
                    evidence_id="R001:macro:evidence",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    family="geometry_precondition",
                    summary="Macro returned bounded test evidence.",
                )
            ],
            planner_readable_report="We conclude the final mechanism is ESIPT.",
        )


def test_base_agent_runtime_retries_with_schema_feedback() -> None:
    runtime = BaseAgentRuntime(agent_name="macro", response_model=AgentReport)
    feedback_seen: list[str | None] = []

    def producer(feedback: str | None) -> dict[str, object]:
        feedback_seen.append(feedback)
        report_text_by_attempt = {
            1: "We conclude the final mechanism is ESIPT.",
            2: "Macro returned bounded observations only.",
        }
        return {
            "report_id": "R001:macro",
            "round_id": "R001",
            "agent_name": "macro",
            "task_received": "collect evidence",
            "status": "success",
            "evidence_units": [
                _evidence_unit(
                    evidence_id="R001:macro:evidence",
                    round_id="R001",
                    source_report_id="R001:macro",
                    agent_name="macro",
                    family="geometry_precondition",
                    summary="Macro returned bounded test evidence.",
                ).model_dump(mode="json")
            ],
            "planner_readable_report": report_text_by_attempt[len(feedback_seen)],
        }

    report = runtime.run_schema_retry(producer)

    assert report.status == "success"
    assert feedback_seen[0] is None
    assert "forbidden phrase" in str(feedback_seen[1])


class FakeMcpTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.arguments: list[tuple[str, str, dict[str, object]]] = []

    def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((server_name, tool_name))
        self.arguments.append((server_name, tool_name, arguments))
        handlers = {
            "list_capabilities": self._list_capabilities,
            "prepare_structure": self._prepare_structure,
            "run_macro": self._agent_report,
            "run_microscopic": self._agent_report,
        }
        return handlers[tool_name](arguments)

    def _list_capabilities(self, arguments: dict[str, object]) -> dict[str, object]:
        del arguments
        return {
            "catalog": default_capability_registry().snapshot().model_dump(mode="json")
        }

    def _prepare_structure(self, arguments: dict[str, object]) -> dict[str, object]:
        return {
            "artifact": {
                "artifact_id": "mcp:structure",
                "kind": "prepared_structure",
                "created_by_round": arguments["round_id"],
                "created_by_agent": "structure",
                "status": "available",
                "reusable_for": ["macro", "microscopic"],
            },
            "failure": FailureReport().model_dump(mode="json"),
        }

    def _agent_report(self, arguments: dict[str, object]) -> dict[str, object]:
        request = arguments["dispatch_request"]
        assert isinstance(request, dict)
        agent_name = str(request["agent_name"])
        round_id = str(request["round_id"])
        route = str(request["route"])
        result_id = f"{round_id}:{agent_name}:{route}"
        family_by_agent = {
            "macro": "geometry_precondition",
            "microscopic": "state_ordering_brightness",
        }
        status_by_agent = {
            "macro": "success",
            "microscopic": "partial",
        }
        tags_by_agent = {
            "macro": ["structural_prior"],
            "microscopic": [route],
        }
        failure_by_status = {
            "success": FailureReport(),
            "partial": FailureReport(
                kind="partial_evidence",
                message=f"{agent_name} MCP adapter returned partial status.",
                recoverable=True,
            ),
        }
        status = status_by_agent[agent_name]
        return {
            "tool_result": {
                "tool_result_id": result_id,
                "round_id": round_id,
                "agent_name": agent_name,
                "dispatch_id": request["dispatch_id"],
                "capability_id": request["capability_id"],
                "selected_route": request["route"],
                "tool_calls": [f"mcp.{agent_name}"],
                "raw_results": {"transport": "fake_mcp"},
                "structured_results": {"selected_route": route},
                "status": status,
                "failure": failure_by_status[status].model_dump(mode="json"),
                "evidence_units": [
                    {
                        "evidence_id": f"{result_id}:evidence",
                        "round_id": round_id,
                        "source_report_id": result_id,
                        "agent_name": agent_name,
                        "capability_id": request["capability_id"],
                        "claim": request["objective"],
                        "context": f"{agent_name} fake MCP test context",
                        "basis": "proxy" if agent_name == "macro" else "checklist",
                        "support": "supports" if status == "success" else "unresolved",
                        "limits": ["Fake MCP transport test payload."],
                        "observable": route,
                        "family": family_by_agent[agent_name],
                        "status": "present" if status == "success" else "partial",
                        "observable_tags": tags_by_agent[agent_name],
                        "summary": f"{agent_name} MCP adapter returned typed evidence.",
                    }
                ],
                "summary": f"{agent_name} MCP adapter returned typed evidence/status.",
            }
        }


class BrokenCapabilityDiscoveryTransport(FakeMcpTransport):
    def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        if tool_name == "list_capabilities":
            raise RuntimeError("capability server is unavailable")
        return super().call_tool(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
        )


def test_orchestrator_can_use_mcp_tool_adapters(tmp_path) -> None:
    transport = FakeMcpTransport()
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=4)
    deps = mcp_deps(config, transport)

    result = MechCALOrchestrator(
        config,
        deps=_fake_llm_deps(config, deps),
    ).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id="mcp_smoke_v2",
    )

    assert result.status == "finalized"
    assert ("aie-capability-mcp", "list_capabilities") in transport.calls
    assert ("aie-structure-mcp", "prepare_structure") in transport.calls
    assert ("aie-macro-mcp", "run_macro") in transport.calls
    assert ("aie-micro-mcp", "run_microscopic") in transport.calls
    assert ("aie-verifier-mcp", "run_verifier") not in transport.calls
    assert any(
        tool_name == "run_macro" and isinstance(arguments.get("execution_plan"), dict)
        for _, tool_name, arguments in transport.arguments
    )


def test_mcp_capability_discovery_failure_is_not_silent(tmp_path) -> None:
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=4)

    with pytest.raises(RuntimeError, match="MCP capability discovery failed"):
        mcp_deps(config, BrokenCapabilityDiscoveryTransport())


@pytest.mark.integration
def test_orchestrator_uses_real_stdio_mcp_server(tmp_path) -> None:
    pytest.importorskip("fastmcp")
    pytest.importorskip("mcp")
    config = OrchestratorConfig(run_base_dir=tmp_path, max_rounds=4)
    deps = mcp_deps(config, StdioMcpTransport())

    result = MechCALOrchestrator(
        config,
        deps=_fake_llm_deps(config, deps),
    ).run(
        smiles="C1=CC=CC=C1",
        user_query="Assess the likely AIE mechanism for this molecule.",
        case_id="stdio_mcp_smoke_v2",
    )

    assert result.status == "finalized"
    assert (tmp_path / "stdio_mcp_smoke_v2" / "final.json").is_file()
    assert (tmp_path / "stdio_mcp_smoke_v2" / "capabilities.json").is_file()
    assert "external_precedent" not in set(result.evidence_ledger.covered_families())


@pytest.mark.integration
def test_real_stdio_mcp_run_macro_supports_migrated_proxy_route() -> None:
    pytest.importorskip("fastmcp")
    pytest.importorskip("mcp")
    route = "screen_rotor_torsion_topology"
    request = DispatchRequest(
        dispatch_id=f"R001:dispatch:macro:{route}",
        round_id="R001",
        agent_name="macro",
        capability_id=f"macro.{route}",
        task=f"Collect {route} structural prior.",
        objective=f"Collect {route} structural prior.",
        evidence_goal_family="geometry_precondition",
        route=route,
    )

    payload = StdioMcpTransport().call_tool(
        server_name="aie-macro-mcp",
        tool_name="run_macro",
        arguments={
            "dispatch_request": request.model_dump(mode="json"),
            "case_run": _case_run_for_gate().model_dump(mode="json"),
            "schema_feedback": None,
        },
    )
    result = ToolExecutionResult.model_validate(payload["tool_result"])

    assert result.status == "success"
    assert result.selected_route == route
    assert result.structured_results["result_name"] == f"{route}_proxy"
    assert "structural_prior" in result.evidence_units[0].observable_tags


def test_openai_settings_default_to_deepseek_flash_without_reasoning_effort(monkeypatch) -> None:
    from mechcal.runtime.llm import OpenAICompatibleSettings

    monkeypatch.setenv("MECHCAL_OPENAI_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MECHCAL_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MECHCAL_OPENAI_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("MECHCAL_OPENAI_TEMPERATURE", raising=False)
    monkeypatch.delenv("MECHCAL_OPENAI_TOP_P", raising=False)
    monkeypatch.delenv("MECHCAL_OPENAI_SEED", raising=False)

    settings = OpenAICompatibleSettings.from_env()

    assert settings.model == "deepseek-v4-flash"
    assert settings.reasoning_effort is None
    assert settings.temperature == 0.0
    assert settings.top_p == 1.0
    assert settings.seed is None
    assert settings.missing_fields() == []


def test_openai_settings_can_omit_reasoning_effort(monkeypatch) -> None:
    from mechcal.runtime.llm import OpenAICompatibleSettings

    monkeypatch.setenv("MECHCAL_OPENAI_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MECHCAL_OPENAI_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("MECHCAL_OPENAI_REASONING_EFFORT", "default")

    settings = OpenAICompatibleSettings.from_env()

    assert settings.model == "deepseek-v4-flash"
    assert settings.reasoning_effort is None
    assert settings.missing_fields() == []


def test_openai_settings_accept_deterministic_generation_overrides(monkeypatch) -> None:
    from mechcal.runtime.llm import OpenAICompatibleSettings

    monkeypatch.setenv("MECHCAL_OPENAI_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("MECHCAL_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MECHCAL_OPENAI_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("MECHCAL_OPENAI_TEMPERATURE", "0.2")
    monkeypatch.setenv("MECHCAL_OPENAI_TOP_P", "0.9")
    monkeypatch.setenv("MECHCAL_OPENAI_SEED", "123")

    settings = OpenAICompatibleSettings.from_env()

    assert settings.temperature == 0.2
    assert settings.top_p == 0.9
    assert settings.seed == 123


def test_benchmark_and_legacy_report_exports(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "case_id,smiles,user_query\n"
        "benzene,C1=CC=CC=C1,Assess AIE mechanism.\n",
        encoding="utf-8",
    )

    from mechcal import benchmark as benchmark_module

    class FakeLlmBenchmarkOrchestrator(MechCALOrchestrator):
        def __init__(self, config, deps=None) -> None:  # type: ignore[no-untyped-def]
            super().__init__(config, deps=_fake_llm_deps(config))

    monkeypatch.setattr(
        benchmark_module,
        "MechCALOrchestrator",
        FakeLlmBenchmarkOrchestrator,
    )

    benchmark_result = run_benchmark(
        dataset_path=dataset,
        output_dir=tmp_path / "benchmark",
        limit=1,
        export_legacy=True,
    )
    run_dir = tmp_path / "benchmark" / "benzene"
    exported = write_legacy_report(
        load_case_run(run_dir),
        tmp_path / "benzene_compat.md",
    )

    assert benchmark_result.case_count == 1
    assert benchmark_result.finalized_count == 1
    assert (run_dir / "legacy_report.json").is_file()
    assert (run_dir / "legacy_report.md").is_file()
    assert exported.is_file()
    assert "AIE-MAS Report: benzene" in exported.read_text(encoding="utf-8")
