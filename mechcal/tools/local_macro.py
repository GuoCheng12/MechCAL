from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mechcal.schemas import (
    AgentExecutionPlan,
    ArtifactRecord,
    CaseRun,
    DispatchRequest,
    EvidenceUnit,
    ToolExecutionResult,
)
from mechcal.schemas.failures import FailureReport
from mechcal.utils.smiles import extract_smiles_features, scaffold_proxy

_LANTHANIDE_SYMBOLS = {
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
}
_TRANSITION_METAL_SYMBOLS = {
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
}
_HEAVY_ATOM_TRIPLET_SYMBOLS = {
    "Br",
    "I",
    "Se",
    "Te",
    "At",
    "Au",
    "Pt",
    "Ir",
    "Ru",
    "Os",
    "Re",
}
_TRIPLET_RELEVANT_HETERO_SYMBOLS = {"S", "P"}


@dataclass(frozen=True)
class MacroRouteResult:
    result_name: str
    observable_tags: list[str]
    summary: str
    metrics: dict[str, Any]
    basis: str = "proxy"
    support: str = "not_applicable"


class LocalMacroTool:
    def run(
        self,
        request: DispatchRequest,
        case_run: CaseRun,
        *,
        execution_plan: AgentExecutionPlan | None = None,
        schema_feedback: str | None = None,
    ) -> ToolExecutionResult:
        del schema_feedback
        tool_args = _bounded_macro_tool_args(execution_plan)
        base_metrics = self._base_metrics(case_run)
        handler = _ROUTE_HANDLERS.get(request.route)
        if handler is None:
            return self._unsupported_result(request)

        route_result = handler(base_metrics)
        route_metrics = {
            **route_result.metrics,
            **({"focus_tags": tool_args["focus_tags"]} if tool_args.get("focus_tags") else {}),
        }
        report_id = f"{request.round_id}:macro"
        artifact_refs = (
            [base_metrics["prepared_artifact_id"]]
            if base_metrics.get("prepared_artifact_id")
            else []
        )
        evidence = EvidenceUnit(
            evidence_id=f"{report_id}:{request.route}:structural_prior",
            round_id=request.round_id,
            source_report_id=report_id,
            agent_name="macro",
            capability_id=request.capability_id,
            claim=_macro_claim(
                request.route,
                route_result=route_result,
                base_metrics=base_metrics,
            ),
            context="single-molecule structure proxy",
            basis=route_result.basis,  # type: ignore[arg-type]
            support=route_result.support,  # type: ignore[arg-type]
            limits=[
                "Macro evidence is a bounded structural proxy, not direct "
                "photophysical or wet-lab evidence."
            ],
            observable=route_result.result_name,
            family="geometry_precondition",
            relation="neutral",
            status="present",
            artifact_refs=artifact_refs,
            observable_tags=route_result.observable_tags,
            summary=route_result.summary,
            metrics=route_metrics,
        )
        return ToolExecutionResult(
            tool_result_id=report_id,
            round_id=request.round_id,
            agent_name="macro",
            dispatch_id=request.dispatch_id,
            capability_id=request.capability_id,
            selected_route=request.route,
            tool_calls=[
                "local_macro.extract_smiles_features",
                f"local_macro.{request.route}",
            ],
            raw_results={
                "tool_args": tool_args,
                "base_metrics": base_metrics,
                route_result.result_name: route_metrics,
            },
            structured_results={
                "selected_route": request.route,
                "tool_args": tool_args,
                "result_name": route_result.result_name,
                "metrics": route_metrics,
                "structural_prior": {
                    route_result.result_name: route_metrics,
                },
            },
            status="success",
            evidence_units=[evidence],
            summary=(
                "Macro returned bounded structural-prior proxy evidence. These "
                "observations are inputs for Planner synthesis and are not direct "
                "mechanism evidence."
            ),
            failure=FailureReport(),
        )

    def _base_metrics(self, case_run: CaseRun) -> dict[str, Any]:
        smiles = case_run.input.smiles
        features = extract_smiles_features(smiles)
        prepared = _prepared_structure_artifact(case_run)
        rdkit = _rdkit_structure_metrics(smiles)
        atom_count = _first_present(
            prepared.metadata.get("atom_count") if prepared else None,
            rdkit.get("atom_count"),
            features.aromatic_atom_count + features.hetero_atom_count,
        )
        return {
            "structure_source": "prepared_structure" if prepared else "smiles_only_fallback",
            "smiles": smiles,
            "prepared_artifact_id": prepared.artifact_id if prepared else None,
            "prepared_file_paths": prepared.file_paths if prepared else {},
            "length": features.length,
            "atom_count": atom_count,
            "aromatic_atom_count": features.aromatic_atom_count,
            "hetero_atom_count": features.hetero_atom_count,
            "branch_point_count": features.branch_point_count,
            "double_bond_count": features.double_bond_count,
            "ring_digit_count": features.ring_digit_count,
            "donor_acceptor_proxy": features.donor_acceptor_proxy,
            "flexibility_proxy": features.flexibility_proxy,
            "conjugation_proxy": features.conjugation_proxy,
            "scaffold_proxy": scaffold_proxy(smiles),
            "conformer_count": _first_present(
                prepared.metadata.get("conformer_count") if prepared else None,
                0,
            ),
            "selected_conformer_id": (
                prepared.metadata.get("selected_conformer_id") if prepared else None
            ),
            **rdkit,
        }

    def _unsupported_result(self, request: DispatchRequest) -> ToolExecutionResult:
        report_id = f"{request.round_id}:macro"
        failure = FailureReport(
            kind="capability_unsupported",
            message=f"Unsupported macro route: {request.route}",
            recoverable=False,
            details={"route": request.route, "capability_id": request.capability_id},
        )
        evidence = EvidenceUnit(
            evidence_id=f"{report_id}:{request.route}:unsupported",
            round_id=request.round_id,
            source_report_id=report_id,
            agent_name="macro",
            capability_id=request.capability_id,
            claim=f"Macro route status observation: {request.route} is unsupported.",
            context="unsupported macro route",
            basis="checklist",
            support="unresolved",
            limits=["No macro proxy was computed because the requested route is unsupported."],
            observable=request.route,
            family="geometry_precondition",
            relation="unknown",
            status="unsupported",
            observable_tags=[request.route],
            summary=failure.message,
        )
        return ToolExecutionResult(
            tool_result_id=report_id,
            round_id=request.round_id,
            agent_name="macro",
            dispatch_id=request.dispatch_id,
            capability_id=request.capability_id,
            selected_route=request.route,
            status="unsupported",
            evidence_units=[evidence],
            failure=failure,
            summary=failure.message,
        )


def _macro_structure_scan(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "length",
            "aromatic_atom_count",
            "hetero_atom_count",
            "branch_point_count",
            "double_bond_count",
            "ring_digit_count",
            "donor_acceptor_proxy",
            "flexibility_proxy",
            "conjugation_proxy",
            "scaffold_proxy",
            "rotatable_bond_count",
            "torsion_candidate_count",
            "aromatic_ring_count",
            "ring_system_count",
            "formal_charge",
            "mol_logp",
            "hbd_count",
            "hba_count",
            "carbonyl_like_site_count",
        ],
    )
    return MacroRouteResult(
        result_name="macro_structure_scan_proxy",
        observable_tags=[
            "macro_structure_scan_proxy",
            "smiles_topology_proxy",
            "structural_prior",
        ],
        summary="Macro screened broad structural-prior proxy descriptors.",
        metrics=metrics,
    )


def _macro_claim(
    route: str,
    *,
    route_result: MacroRouteResult,
    base_metrics: dict[str, Any],
) -> str:
    aromatic = int(base_metrics.get("aromatic_atom_count") or 0)
    hetero = int(base_metrics.get("hetero_atom_count") or 0)
    double_bonds = int(base_metrics.get("double_bond_count") or 0)
    flexibility = float(base_metrics.get("flexibility_proxy") or 0.0)
    donor_acceptor = int(base_metrics.get("donor_acceptor_proxy") or 0)
    scaffold = str(base_metrics.get("scaffold_proxy") or "unknown_scaffold")
    sulfur_rich = "s" in str(base_metrics.get("smiles", "")).lower()

    if route == "macro_structure_scan":
        if hetero == 0 and aromatic >= 12 and double_bonds > 0:
            return (
                "SMILES-only structural scan reports a neutral aromatic alkene "
                "topology. It records topology only; Planner owns any AIE, "
                "photochemistry, or solution-versus-solid mechanism interpretation."
            )
        return (
            "SMILES-only structural scan reports aromaticity, heteroatom, conjugation, "
            "and flexibility descriptors. It does not assign photophysical behavior."
        )
    if route == "run_solid_state_emission_proxy":
        return (
            "Solid-state-emission proxy route reports aromatic and hydrophobic "
            "structure descriptors plus missing material-level observables. Planner "
            "owns any aggregate-emission or packing-restriction interpretation."
        )
    if route == "run_crystal_restriction_checklist":
        return (
            "Crystal restriction checklist records which packing, solid-state PL, "
            "time-resolved, or high-level excited-state observables are absent from "
            "the current SMILES-only run."
        )
    if route == "screen_esipt_structural_motif":
        motif = bool(route_result.metrics.get("esipt_motif_proxy"))
        pair_text = _esipt_pair_text(route_result.metrics)
        hbd = int(route_result.metrics.get("hbd_count") or 0)
        hba = int(route_result.metrics.get("hba_count") or 0)
        tautomer_count = route_result.metrics.get("tautomerizable_subgraph_proxy")
        return (
            "ESIPT structural screen observation: "
            + (
                f"{hbd} H-bond donor(s) and {hba} acceptor(s) were detected; "
                f"{pair_text}; tautomerizable-subgraph proxy is {tautomer_count}."
                if motif
                else (
                    f"{hbd} H-bond donor(s) and {hba} acceptor(s) were detected, "
                    "but no SMILES-level intramolecular proton-transfer donor/acceptor "
                    "pair proxy was found."
                )
            )
        )
    if route in {"screen_rotor_torsion_topology", "screen_rotor_rim_prior"}:
        rotatable = int(route_result.metrics.get("rotatable_bond_count") or 0)
        torsions = int(route_result.metrics.get("torsion_candidate_count") or 0)
        branches = int(route_result.metrics.get("branch_point_count") or 0)
        flex = float(route_result.metrics.get("flexibility_proxy") or 0.0)
        torsion_inventory = _short_inventory(
            route_result.metrics.get("rotatable_bond_inventory"),
            "bond",
        )
        return (
            "Rotor/torsion topology observation: "
            + (
                f"{rotatable} rotatable bonds, {torsions} torsion candidates, "
                f"{branches} branch points, and flexibility proxy {flex:.2f} "
                f"were detected{torsion_inventory}."
                if flexibility > 0
                else (
                    f"{rotatable} rotatable bonds and {torsions} torsion candidates "
                    f"were detected{torsion_inventory}; the structural flexibility "
                    "proxy is low or unavailable."
                )
            )
        )
    if route in {
        "screen_donor_acceptor_layout",
        "screen_donor_acceptor_architecture",
    }:
        hbd = int(route_result.metrics.get("hbd_count") or 0)
        hba = int(route_result.metrics.get("hba_count") or 0)
        conjugation = float(route_result.metrics.get("conjugation_proxy") or 0.0)
        hetero_count = int(route_result.metrics.get("hetero_atom_count") or 0)
        donor_atoms = _short_atom_list(route_result.metrics.get("donor_atom_symbols"))
        acceptor_atoms = _short_atom_list(
            route_result.metrics.get("acceptor_atom_symbols")
        )
        return (
            "Donor/acceptor structural observation: "
            + (
                f"donor/acceptor proxy={donor_acceptor}, hetero atoms={hetero_count}, "
                f"HBD={hbd}, HBA={hba}, conjugation proxy={conjugation:.2f}, "
                f"donor-like atoms={donor_atoms}, acceptor-like atoms={acceptor_atoms}."
                if donor_acceptor > 0
                else (
                    f"donor/acceptor proxy is low; hetero atoms={hetero_count}, "
                    f"HBD={hbd}, HBA={hba}, conjugation proxy={conjugation:.2f}, "
                    f"donor-like atoms={donor_atoms}, acceptor-like atoms={acceptor_atoms}."
                )
            )
        )
    if route == "screen_polar_binding_site_prior":
        hbd = int(route_result.metrics.get("hbd_count") or 0)
        hba = int(route_result.metrics.get("hba_count") or 0)
        formal_charge = int(route_result.metrics.get("formal_charge") or 0)
        carbonyl_like = bool(route_result.metrics.get("carbonyl_like_proxy"))
        polar_atoms = _short_atom_list(route_result.metrics.get("polar_atom_symbols"))
        carbonyl_count = int(route_result.metrics.get("carbonyl_like_site_count") or 0)
        return (
            "Polar binding-site structural-prior observation: "
            + (
                f"formal charge={formal_charge}, HBD={hbd}, HBA={hba}, "
                f"carbonyl-like proxy={carbonyl_like}, carbonyl-like site count="
                f"{carbonyl_count}, polar atoms={polar_atoms}; polar interaction-site "
                "proxy was detected."
                if route_result.metrics.get("polar_binding_site_proxy")
                else (
                    f"formal charge={formal_charge}, HBD={hbd}, HBA={hba}, "
                    f"carbonyl-like proxy={carbonyl_like}, carbonyl-like site count="
                    f"{carbonyl_count}, polar atoms={polar_atoms}; no strong "
                    "polar interaction-site proxy was detected."
                )
            )
            + " This is only a public structural screen; it is not guest uptake, "
            "binding, PET quenching, redox, or sensing evidence."
        )
    if route == "screen_metal_triplet_prior":
        metal_symbols = list(route_result.metrics.get("metal_atom_symbols") or [])
        heavy_symbols = list(route_result.metrics.get("heavy_atom_symbols") or [])
        triplet_symbols = list(
            route_result.metrics.get("triplet_relevant_hetero_symbols") or []
        )
        detected = [*metal_symbols, *heavy_symbols, *triplet_symbols]
        return (
            "Metal/triplet structural-prior observation: "
            + (
                "triplet- or metal-transfer-relevant atom classes were detected "
                f"({', '.join(sorted(set(map(str, detected))))})."
                if detected
                else "no metal, lanthanide, heavy-atom, sulfur, or phosphorus "
                "triplet prior was detected from the parsed structure."
            )
            + " This is a screening prior only, not phosphorescence, RTP, TADF, "
            "lanthanide-line, lifetime, triplet-energy, or energy-transfer evidence."
        )
    if route in {"screen_pi_stacking_prone_geometry", "run_dimer_packing_proxy"}:
        return (
            "Packing/contact proxy route records pi-contact, compactness, or dimer "
            "geometry descriptors. Planner owns any excimer, quenching, or packing "
            "assignment."
        )
    if route == "screen_aggregation_prone_scaffold":
        aromatic_atoms = int(route_result.metrics.get("aromatic_atom_count") or 0)
        aromatic_rings = int(route_result.metrics.get("aromatic_ring_count") or 0)
        logp = float(route_result.metrics.get("mol_logp") or 0.0)
        aggregation_proxy = bool(route_result.metrics.get("aggregation_prone_proxy"))
        return (
            "Aggregation-prone scaffold observation: "
            f"{aromatic_atoms} aromatic atoms, {aromatic_rings} aromatic rings, "
            f"and logP proxy {logp:.2f} were detected; aggregation-prone proxy is "
            f"{aggregation_proxy}."
        )
    if route == "run_water_fraction_aggregation_checklist":
        return (
            "Water-fraction aggregation checklist records that PL, DLS, and scattering "
            "controls are not present in the SMILES-only input."
        )
    if sulfur_rich:
        return (
            "Sulfur-containing aromatic topology was detected by the macro route. "
            "Planner owns any through-space sulfur-contact or heavy-atom interpretation."
        )
    return (
        f"Macro route {route} recorded bounded structural proxy descriptors "
        f"({scaffold}) without assigning a photophysical mechanism."
    )


def _screen_donor_acceptor_layout(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "donor_acceptor_proxy",
            "hetero_atom_count",
            "hbd_count",
            "hba_count",
            "conjugation_proxy",
            "aromatic_atom_count",
            "donor_atom_symbols",
            "acceptor_atom_symbols",
            "polar_atom_symbols",
        ],
    )
    metrics["donor_acceptor_partition_proxy"] = float(
        min(int(base["donor_acceptor_proxy"]), 1)
    )
    return MacroRouteResult(
        result_name="screen_donor_acceptor_layout_proxy",
        observable_tags=[
            "screen_donor_acceptor_layout_proxy",
            "charge_separation_proxy",
            "structural_prior",
        ],
        summary="Macro screened donor/acceptor layout as bounded structural prior.",
        metrics=metrics,
    )


def _screen_rotor_torsion_topology(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "rotatable_bond_count",
            "torsion_candidate_count",
            "branch_point_count",
            "flexibility_proxy",
            "rotatable_bond_inventory",
        ],
    )
    metrics["topology_availability"] = (
        "candidate_proxy_present"
        if int(metrics["torsion_candidate_count"] or 0) > 0
        else "checked_low_rotor_proxy"
    )
    return MacroRouteResult(
        result_name="screen_rotor_torsion_topology_proxy",
        observable_tags=[
            "screen_rotor_torsion_topology_proxy",
            "rotor_topology_proxy",
            "torsion_topology_proxy",
            "structural_prior",
        ],
        summary="Macro screened rotor/torsion topology as bounded structural prior.",
        metrics=metrics,
    )


def _screen_planarity_compactness(base: dict[str, Any]) -> MacroRouteResult:
    atom_count = float(base.get("atom_count") or base["length"])
    metrics = {
        "structure_source": base["structure_source"],
        "planarity_proxy": round(min(1.0, float(base["conjugation_proxy"]) / 10.0), 3),
        "compactness_proxy": round(max(0.0, 1.0 - min(atom_count / 120.0, 1.0)), 3),
        "principal_span_proxy": round(min(atom_count / 10.0, 20.0), 3),
        "conjugation_proxy": base["conjugation_proxy"],
        "aromatic_fraction_proxy": _fraction(base["aromatic_atom_count"], atom_count),
    }
    return MacroRouteResult(
        result_name="screen_planarity_compactness_proxy",
        observable_tags=[
            "screen_planarity_compactness_proxy",
            "planarity_proxy",
            "compactness_proxy",
            "structural_prior",
        ],
        summary="Macro screened planarity/compactness as bounded structural prior.",
        metrics=metrics,
    )


def _screen_intramolecular_hbond_preorganization(base: dict[str, Any]) -> MacroRouteResult:
    hbd = int(base.get("hbd_count") or 0)
    hba = int(base.get("hba_count") or 0)
    metrics = {
        "structure_source": base["structure_source"],
        "hbd_count": base.get("hbd_count"),
        "hba_count": base.get("hba_count"),
        "h_transfer_sites_proxy": int(bool(hbd and hba)),
        "tautomerizable_subgraph_proxy": base.get("tautomerizable_subgraph_proxy"),
        "preorganization_proxy_status": "motif_proxy_present"
        if hbd and hba
        else "motif_proxy_checked_absent",
    }
    return MacroRouteResult(
        result_name="screen_intramolecular_hbond_preorganization_proxy",
        observable_tags=[
            "screen_intramolecular_hbond_preorganization_proxy",
            "hbond_preorganization_proxy",
            "structural_prior",
        ],
        summary=(
            "Macro screened intramolecular H-bond preorganization as bounded "
            "structural prior."
        ),
        metrics=metrics,
    )


def _screen_conformer_geometry_proxy(base: dict[str, Any]) -> MacroRouteResult:
    conformer_count = int(base.get("conformer_count") or 0)
    metrics = {
        "structure_source": base["structure_source"],
        "conformer_count": conformer_count,
        "selected_conformer_id": base.get("selected_conformer_id"),
        "conformer_dispersion_proxy": 0.0,
        "single_conformer_limit_note": (
            "Prepared structure context is available."
            if conformer_count > 0
            else "Only SMILES-level conformer proxy context is available."
        ),
        "flexibility_proxy": base["flexibility_proxy"],
    }
    return MacroRouteResult(
        result_name="screen_conformer_geometry_proxy",
        observable_tags=[
            "screen_conformer_geometry_proxy",
            "conformer_geometry_proxy",
            "structural_prior",
        ],
        summary="Macro screened conformer geometry as bounded structural prior.",
        metrics=metrics,
    )


def _screen_neutral_aromatic_structure(base: dict[str, Any]) -> MacroRouteResult:
    atom_count = float(base.get("atom_count") or base["length"])
    metrics = {
        "structure_source": base["structure_source"],
        "aromatic_atom_count": base["aromatic_atom_count"],
        "aromatic_fraction_proxy": _fraction(base["aromatic_atom_count"], atom_count),
        "aromatic_ring_count": base.get("aromatic_ring_count"),
        "ring_system_count": base.get("ring_system_count"),
        "planarity_proxy": round(min(1.0, float(base["conjugation_proxy"]) / 10.0), 3),
        "neutral_aromatic_structural_prior": bool(
            int(base["aromatic_atom_count"]) > 0 and int(base["hetero_atom_count"]) == 0
        ),
    }
    return MacroRouteResult(
        result_name="screen_neutral_aromatic_structure_proxy",
        observable_tags=[
            "screen_neutral_aromatic_structure_proxy",
            "aromatic_core_dominance_proxy",
            "ring_system_rigidity_proxy",
            "structural_prior",
        ],
        summary="Macro screened neutral aromatic structure as bounded structural prior.",
        metrics=metrics,
    )


def _screen_donor_acceptor_architecture(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "donor_acceptor_proxy",
            "hetero_atom_count",
            "hba_count",
            "hbd_count",
            "conjugation_proxy",
            "formal_charge",
            "donor_atom_symbols",
            "acceptor_atom_symbols",
            "polar_atom_symbols",
        ],
    )
    metrics["architecture_proxy_status"] = (
        "donor_acceptor_proxy_present"
        if int(base.get("donor_acceptor_proxy") or 0) > 0
        else "donor_acceptor_proxy_checked_weak"
    )
    return MacroRouteResult(
        result_name="screen_donor_acceptor_architecture_proxy",
        observable_tags=[
            "screen_donor_acceptor_architecture_proxy",
            "donor_acceptor_topology",
            "ct_architecture_proxy",
            "structural_prior",
        ],
        summary="Macro screened donor-acceptor architecture as a bounded CT prior.",
        metrics=metrics,
    )


def _screen_cationic_targeting_prior(base: dict[str, Any]) -> MacroRouteResult:
    formal_charge = int(base.get("formal_charge") or 0)
    metrics = {
        "structure_source": base["structure_source"],
        "formal_charge": formal_charge,
        "hetero_atom_count": base.get("hetero_atom_count"),
        "cationic_prior": formal_charge > 0,
    }
    return MacroRouteResult(
        result_name="screen_cationic_targeting_prior_proxy",
        observable_tags=[
            "screen_cationic_targeting_prior_proxy",
            "cationic_targeting_prior",
            "structural_prior",
        ],
        summary="Macro screened cationic targeting priors from formal charge topology.",
        metrics=metrics,
    )


def _screen_lipophilicity_proxy(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "mol_logp",
            "aromatic_atom_count",
            "hetero_atom_count",
            "atom_count",
        ],
    )
    metrics["lipophilic_partition_proxy"] = (
        "high_proxy" if float(base.get("mol_logp") or 0.0) >= 5.0 else "bounded_proxy"
    )
    return MacroRouteResult(
        result_name="screen_lipophilicity_proxy",
        observable_tags=[
            "screen_lipophilicity_proxy",
            "lipophilicity_proxy",
            "partitioning_proxy",
            "structural_prior",
        ],
        summary="Macro screened lipophilicity as a bounded partitioning prior.",
        metrics=metrics,
    )


def _screen_polar_binding_site_prior(base: dict[str, Any]) -> MacroRouteResult:
    formal_charge = int(base.get("formal_charge") or 0)
    hbd = int(base.get("hbd_count") or 0)
    hba = int(base.get("hba_count") or 0)
    atom_symbols = [str(item) for item in base.get("atom_symbols", [])]
    carbonyl_like = int(base.get("double_bond_count") or 0) > 0 and "O" in atom_symbols
    charged_or_hbond = bool(formal_charge or hbd or hba >= 2)
    metrics = {
        "structure_source": base["structure_source"],
        "formal_charge": formal_charge,
        "hbd_count": hbd,
        "hba_count": hba,
        "hetero_atom_count": base.get("hetero_atom_count"),
        "carbonyl_like_proxy": carbonyl_like,
        "carbonyl_like_site_count": base.get("carbonyl_like_site_count"),
        "polar_atom_symbols": base.get("polar_atom_symbols", []),
        "donor_atom_symbols": base.get("donor_atom_symbols", []),
        "acceptor_atom_symbols": base.get("acceptor_atom_symbols", []),
        "polar_binding_site_proxy": bool(charged_or_hbond or carbonyl_like),
        "host_guest_followup_trigger": bool(formal_charge or hbd or hba >= 2),
        "pet_receptor_like_proxy": bool(carbonyl_like or formal_charge or hba >= 2),
    }
    return MacroRouteResult(
        result_name="screen_polar_binding_site_prior_proxy",
        observable_tags=[
            "screen_polar_binding_site_prior_proxy",
            "host_guest_interaction_prior",
            "pet_receptor_prior",
            "structural_prior",
        ],
        summary=(
            "Macro screened polar, charged, carbonyl, and H-bonding sites as "
            "bounded host-guest/PET interaction priors."
        ),
        metrics=metrics,
    )


def _screen_metal_triplet_prior(base: dict[str, Any]) -> MacroRouteResult:
    atom_symbols = [str(item) for item in base.get("atom_symbols", [])]
    metal_symbols = sorted(
        {
            symbol
            for symbol in atom_symbols
            if symbol in _LANTHANIDE_SYMBOLS or symbol in _TRANSITION_METAL_SYMBOLS
        }
    )
    heavy_symbols = sorted(
        {symbol for symbol in atom_symbols if symbol in _HEAVY_ATOM_TRIPLET_SYMBOLS}
    )
    triplet_hetero = sorted(
        {symbol for symbol in atom_symbols if symbol in _TRIPLET_RELEVANT_HETERO_SYMBOLS}
    )
    metrics = {
        "structure_source": base["structure_source"],
        "atom_symbols": atom_symbols[:24],
        "metal_atom_symbols": metal_symbols,
        "lanthanide_symbols": sorted(
            {symbol for symbol in atom_symbols if symbol in _LANTHANIDE_SYMBOLS}
        ),
        "transition_metal_symbols": sorted(
            {symbol for symbol in atom_symbols if symbol in _TRANSITION_METAL_SYMBOLS}
        ),
        "heavy_atom_symbols": heavy_symbols,
        "triplet_relevant_hetero_symbols": triplet_hetero,
        "metal_or_lanthanide_prior": bool(metal_symbols),
        "heavy_atom_triplet_prior": bool(heavy_symbols),
        "sulfur_phosphorus_triplet_prior": bool(triplet_hetero),
        "triplet_metal_followup_trigger": bool(
            metal_symbols or heavy_symbols or triplet_hetero
        ),
    }
    return MacroRouteResult(
        result_name="screen_metal_triplet_prior_proxy",
        observable_tags=[
            "screen_metal_triplet_prior_proxy",
            "metal_triplet_prior",
            "heavy_atom_triplet_prior",
            "structural_prior",
        ],
        summary=(
            "Macro screened metal, lanthanide, heavy-atom, sulfur, and phosphorus "
            "structural priors for triplet/metal energy-transfer follow-up."
        ),
        metrics=metrics,
    )


def _screen_rotor_rim_prior(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "rotatable_bond_count",
            "torsion_candidate_count",
            "flexibility_proxy",
            "branch_point_count",
        ],
    )
    metrics["rim_prior"] = int(base.get("torsion_candidate_count") or 0) > 0
    return MacroRouteResult(
        result_name="screen_rotor_rim_prior_proxy",
        observable_tags=[
            "screen_rotor_rim_prior_proxy",
            "rotor_rim_prior",
            "torsion_topology_proxy",
            "structural_prior",
        ],
        summary="Macro screened rotor/RIM priors as bounded structural topology.",
        metrics=metrics,
    )


def _screen_esipt_structural_motif(base: dict[str, Any]) -> MacroRouteResult:
    hbd = int(base.get("hbd_count") or 0)
    hba = int(base.get("hba_count") or 0)
    candidate_pairs = list(base.get("proton_transfer_pair_proxies") or [])
    metrics = {
        "structure_source": base["structure_source"],
        "hbd_count": hbd,
        "hba_count": hba,
        "tautomerizable_subgraph_proxy": base.get("tautomerizable_subgraph_proxy"),
        "proton_transfer_pair_proxy_count": len(candidate_pairs),
        "proton_transfer_pair_proxies": candidate_pairs[:5],
        "closest_proton_transfer_topological_distance": (
            candidate_pairs[0].get("topological_distance") if candidate_pairs else None
        ),
        "esipt_motif_proxy": bool(candidate_pairs) or bool(hbd and hba),
    }
    return MacroRouteResult(
        result_name="screen_esipt_structural_motif_proxy",
        observable_tags=[
            "screen_esipt_structural_motif_proxy",
            "esipt_motif_proxy",
            "structural_prior",
        ],
        summary="Macro screened ESIPT structural motifs as a bounded proxy.",
        metrics=metrics,
    )


def _screen_aggregation_prone_scaffold(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "aromatic_atom_count",
            "aromatic_ring_count",
            "conjugation_proxy",
            "mol_logp",
            "scaffold_proxy",
        ],
    )
    metrics["aggregation_prone_proxy"] = bool(
        int(base.get("aromatic_atom_count") or 0) >= 6
        and float(base.get("mol_logp") or 0.0) >= 3.0
    )
    return MacroRouteResult(
        result_name="screen_aggregation_prone_scaffold_proxy",
        observable_tags=[
            "screen_aggregation_prone_scaffold_proxy",
            "aggregation_scaffold_proxy",
            "solid_state_emission_checklist",
            "structural_prior",
        ],
        summary="Macro screened aggregation-prone scaffold features as bounded proxies.",
        metrics=metrics,
    )


def _screen_pi_stacking_prone_geometry(base: dict[str, Any]) -> MacroRouteResult:
    atom_count = float(base.get("atom_count") or base["length"])
    metrics = _pick(
        base,
        [
            "structure_source",
            "aromatic_atom_count",
            "aromatic_ring_count",
            "ring_system_count",
            "conjugation_proxy",
            "mol_logp",
        ],
    )
    metrics["planar_aromatic_surface_proxy"] = round(
        _fraction(base["aromatic_atom_count"], atom_count)
        * min(float(base.get("conjugation_proxy") or 0.0) / 10.0, 1.0),
        6,
    )
    metrics["pi_stacking_prone_proxy"] = (
        metrics["planar_aromatic_surface_proxy"] >= 0.25
        and int(base.get("aromatic_ring_count") or 0) >= 1
    )
    return MacroRouteResult(
        result_name="screen_pi_stacking_prone_geometry_proxy",
        observable_tags=[
            "screen_pi_stacking_prone_geometry_proxy",
            "pi_stacking_proxy",
            "material_level_proxy",
            "structural_prior",
        ],
        summary="Macro screened pi-stacking-prone geometry as a material-level proxy.",
        metrics=metrics,
    )


def _run_dimer_packing_proxy(base: dict[str, Any]) -> MacroRouteResult:
    atom_count = float(base.get("atom_count") or base["length"])
    aromatic_fraction = _fraction(base["aromatic_atom_count"], atom_count)
    metrics = {
        "structure_source": base["structure_source"],
        "aromatic_fraction_proxy": aromatic_fraction,
        "mol_logp": base.get("mol_logp"),
        "rotatable_bond_count": base.get("rotatable_bond_count"),
        "dimer_contact_score_proxy": round(
            aromatic_fraction
            + min(float(base.get("mol_logp") or 0.0) / 10.0, 0.5)
            - min(float(base.get("rotatable_bond_count") or 0.0) / 20.0, 0.3),
            6,
        ),
    }
    return MacroRouteResult(
        result_name="dimer_packing_proxy",
        observable_tags=[
            "run_dimer_packing_proxy",
            "dimer_packing_proxy",
            "material_level_proxy",
            "structural_prior",
        ],
        summary="Macro estimated a lightweight dimer packing/contact proxy.",
        metrics=metrics,
    )


def _run_aggregate_contact_proxy(base: dict[str, Any]) -> MacroRouteResult:
    metrics = _pick(
        base,
        [
            "structure_source",
            "aromatic_atom_count",
            "hetero_atom_count",
            "rotatable_bond_count",
            "mol_logp",
            "scaffold_proxy",
        ],
    )
    metrics["hydrophobic_contact_proxy"] = float(base.get("mol_logp") or 0.0) >= 3.0
    metrics["polar_disruption_proxy"] = int(base.get("hetero_atom_count") or 0) > 4
    return MacroRouteResult(
        result_name="aggregate_contact_proxy",
        observable_tags=[
            "run_aggregate_contact_proxy",
            "aggregate_contact_proxy",
            "material_level_proxy",
            "structural_prior",
        ],
        summary="Macro screened aggregate contact priors from bounded descriptors.",
        metrics=metrics,
    )


def _run_crystal_restriction_checklist(base: dict[str, Any]) -> MacroRouteResult:
    metrics = {
        "structure_source": base["structure_source"],
        "required_evidence": [
            "single_crystal_or_powder_packing",
            "solid_state_PL_or_quantum_yield",
            "temperature_or_viscosity_restriction_control",
        ],
        "available_proxy_inputs": [
            "aromaticity",
            "rotatable_bond_count",
            "planarity_proxy",
        ],
    }
    return MacroRouteResult(
        result_name="crystal_restriction_checklist",
        observable_tags=[
            "run_crystal_restriction_checklist",
            "crystal_restriction_checklist",
            "material_level_checklist",
            "structural_prior",
        ],
        summary="Macro returned a checklist for crystal/solid-state restriction evidence.",
        metrics=metrics,
        basis="checklist",
        support="unresolved",
    )


def _run_solid_state_emission_proxy(base: dict[str, Any]) -> MacroRouteResult:
    atom_count = float(base.get("atom_count") or base["length"])
    metrics = {
        "structure_source": base["structure_source"],
        "aromatic_fraction_proxy": _fraction(base["aromatic_atom_count"], atom_count),
        "rotor_burden_proxy": base.get("rotatable_bond_count"),
        "aggregation_prone_proxy": bool(
            int(base.get("aromatic_atom_count") or 0) >= 6
            and float(base.get("mol_logp") or 0.0) >= 3.0
        ),
        "solid_state_emission_claim_status": "proxy_only",
    }
    return MacroRouteResult(
        result_name="solid_state_emission_proxy",
        observable_tags=[
            "run_solid_state_emission_proxy",
            "solid_state_emission_proxy",
            "material_level_proxy",
            "structural_prior",
        ],
        summary="Macro screened solid-state emission plausibility as proxy-only evidence.",
        metrics=metrics,
    )


def _run_water_fraction_aggregation_checklist(base: dict[str, Any]) -> MacroRouteResult:
    metrics = {
        "structure_source": base["structure_source"],
        "required_evidence": [
            "water_fraction_PL_series",
            "DLS_or_nanoparticle_size",
            "absorption_scattering_control",
        ],
        "available_proxy_inputs": [
            "mol_logp",
            "aromaticity",
            "hetero_atom_count",
        ],
    }
    return MacroRouteResult(
        result_name="water_fraction_aggregation_checklist",
        observable_tags=[
            "run_water_fraction_aggregation_checklist",
            "water_fraction_aggregation_checklist",
            "material_level_checklist",
            "structural_prior",
        ],
        summary="Macro returned a wet-lab checklist for water-fraction aggregation evidence.",
        metrics=metrics,
        basis="checklist",
        support="unresolved",
    )


_ROUTE_HANDLERS: dict[str, Callable[[dict[str, Any]], MacroRouteResult]] = {
    "macro_structure_scan": _macro_structure_scan,
    "screen_donor_acceptor_layout": _screen_donor_acceptor_layout,
    "screen_rotor_torsion_topology": _screen_rotor_torsion_topology,
    "screen_planarity_compactness": _screen_planarity_compactness,
    "screen_intramolecular_hbond_preorganization": (
        _screen_intramolecular_hbond_preorganization
    ),
    "screen_conformer_geometry_proxy": _screen_conformer_geometry_proxy,
    "screen_neutral_aromatic_structure": _screen_neutral_aromatic_structure,
    "screen_donor_acceptor_architecture": _screen_donor_acceptor_architecture,
    "screen_cationic_targeting_prior": _screen_cationic_targeting_prior,
    "screen_lipophilicity_proxy": _screen_lipophilicity_proxy,
    "screen_polar_binding_site_prior": _screen_polar_binding_site_prior,
    "screen_metal_triplet_prior": _screen_metal_triplet_prior,
    "screen_rotor_rim_prior": _screen_rotor_rim_prior,
    "screen_esipt_structural_motif": _screen_esipt_structural_motif,
    "screen_aggregation_prone_scaffold": _screen_aggregation_prone_scaffold,
    "screen_pi_stacking_prone_geometry": _screen_pi_stacking_prone_geometry,
    "run_dimer_packing_proxy": _run_dimer_packing_proxy,
    "run_aggregate_contact_proxy": _run_aggregate_contact_proxy,
    "run_crystal_restriction_checklist": _run_crystal_restriction_checklist,
    "run_solid_state_emission_proxy": _run_solid_state_emission_proxy,
    "run_water_fraction_aggregation_checklist": _run_water_fraction_aggregation_checklist,
}


def _prepared_structure_artifact(case_run: CaseRun) -> ArtifactRecord | None:
    return next(
        (
            artifact
            for artifact in case_run.artifact_manifest.artifacts
            if artifact.kind == "prepared_structure"
            and artifact.status in {"available", "partial"}
        ),
        None,
    )


def _rdkit_structure_metrics(smiles: str) -> dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors
    except ModuleNotFoundError:
        return _empty_rdkit_metrics()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _empty_rdkit_metrics()
    atom_count = mol.GetNumAtoms()
    rotatable_bond_count = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    donor_acceptor_inventory = _donor_acceptor_atom_symbols(mol)
    return {
        "atom_symbols": atom_symbols,
        "atom_count": atom_count,
        "rotatable_bond_count": rotatable_bond_count,
        "torsion_candidate_count": rotatable_bond_count,
        "rotatable_bond_inventory": _rotatable_bond_inventory(mol)[:8],
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "mol_logp": float(Crippen.MolLogP(mol)),
        "aromatic_ring_count": _aromatic_ring_count(mol),
        "ring_system_count": _ring_system_count(mol),
        "hbd_count": int(Lipinski.NumHDonors(mol)),
        "hba_count": int(Lipinski.NumHAcceptors(mol)),
        "carbonyl_like_site_count": _carbonyl_like_site_count(mol),
        **donor_acceptor_inventory,
        "tautomerizable_subgraph_proxy": _tautomerizable_subgraph_proxy(Chem, mol),
        "proton_transfer_pair_proxies": _proton_transfer_pair_proxies(Chem, mol),
    }


def _empty_rdkit_metrics() -> dict[str, Any]:
    return {
        "atom_symbols": [],
        "atom_count": None,
        "rotatable_bond_count": 0,
        "torsion_candidate_count": 0,
        "formal_charge": 0,
        "mol_logp": None,
        "aromatic_ring_count": None,
        "ring_system_count": None,
        "rotatable_bond_inventory": [],
        "hbd_count": None,
        "hba_count": None,
        "carbonyl_like_site_count": 0,
        "donor_atom_symbols": [],
        "acceptor_atom_symbols": [],
        "polar_atom_symbols": [],
        "tautomerizable_subgraph_proxy": None,
        "proton_transfer_pair_proxies": [],
    }


def _tautomerizable_subgraph_proxy(Chem: Any, mol: Any) -> int | None:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ModuleNotFoundError:
        return None
    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = enumerator.Enumerate(mol)
    unique = {Chem.MolToSmiles(tautomer, canonical=True) for tautomer in tautomers}
    return max(len(unique) - 1, 0)


def _aromatic_ring_count(mol: Any) -> int:
    return sum(
        1
        for atom_indices in mol.GetRingInfo().AtomRings()
        if atom_indices
        and all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in atom_indices)
    )


def _ring_system_count(mol: Any) -> int:
    remaining = [set(atom_indices) for atom_indices in mol.GetRingInfo().AtomRings()]
    system_count = 0
    while remaining:
        system_count += 1
        merged = remaining.pop()
        connected = [ring for ring in remaining if ring & merged]
        while connected:
            merged |= connected.pop()
            remaining = [ring for ring in remaining if not (ring & merged)]
            connected = [ring for ring in remaining if ring & merged]
    return system_count


def _rotatable_bond_inventory(mol: Any) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if bond.IsInRing() or bond.GetBondTypeAsDouble() != 1.0:
            continue
        if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
            continue
        inventory.append(
            {
                "bond": (
                    f"{begin.GetSymbol()}{begin.GetIdx()}-"
                    f"{end.GetSymbol()}{end.GetIdx()}"
                ),
                "begin_symbol": begin.GetSymbol(),
                "end_symbol": end.GetSymbol(),
                "begin_degree": int(begin.GetDegree()),
                "end_degree": int(end.GetDegree()),
                "aryl_link": bool(begin.GetIsAromatic() or end.GetIsAromatic()),
            }
        )
    return inventory


def _donor_acceptor_atom_symbols(mol: Any) -> dict[str, list[str]]:
    donor_like: list[str] = []
    acceptor_like: list[str] = []
    polar: list[str] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in {"N", "O", "S", "P"}:
            continue
        label = f"{symbol}{atom.GetIdx()}"
        polar.append(label)
        if atom.GetFormalCharge() >= 0 and (
            atom.GetTotalNumHs() > 0 or symbol in {"N", "S", "P"}
        ):
            donor_like.append(label)
        if atom.GetFormalCharge() <= 0 and symbol in {"N", "O", "S"}:
            acceptor_like.append(label)
    return {
        "donor_atom_symbols": donor_like[:12],
        "acceptor_atom_symbols": acceptor_like[:12],
        "polar_atom_symbols": polar[:16],
    }


def _carbonyl_like_site_count(mol: Any) -> int:
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 2.0:
            continue
        symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
        if "O" in symbols and symbols & {"C", "S", "P"}:
            count += 1
    return count


def _proton_transfer_pair_proxies(Chem: Any, mol: Any) -> list[dict[str, object]]:
    donors: list[tuple[int, str]] = []
    acceptors: list[tuple[int, str]] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        if symbol in {"O", "N", "S"} and atom.GetTotalNumHs() > 0 and formal_charge <= 0:
            donors.append((atom.GetIdx(), symbol))
        if symbol in {"O", "N", "S"} and formal_charge <= 0:
            acceptors.append((atom.GetIdx(), symbol))
    if not donors or not acceptors:
        return []
    distances = Chem.GetDistanceMatrix(mol)
    pairs: list[dict[str, object]] = []
    for donor_index, donor_symbol in donors:
        for acceptor_index, acceptor_symbol in acceptors:
            if donor_index == acceptor_index:
                continue
            distance = int(distances[donor_index][acceptor_index])
            if 2 <= distance <= 6:
                pairs.append(
                    {
                        "donor_atom": f"{donor_symbol}{donor_index}",
                        "acceptor_atom": f"{acceptor_symbol}{acceptor_index}",
                        "pair_type": f"{donor_symbol}-H...{acceptor_symbol}",
                        "topological_distance": distance,
                    }
                )
    return sorted(
        pairs,
        key=lambda item: (
            int(item["topological_distance"]),
            str(item["pair_type"]),
            str(item["donor_atom"]),
            str(item["acceptor_atom"]),
        ),
    )[:8]


def _short_inventory(value: Any, noun: str) -> str:
    if not isinstance(value, list) or not value:
        return ""
    labels: list[str] = []
    for item in value[:4]:
        if isinstance(item, dict) and item.get(noun):
            labels.append(str(item[noun]))
    if not labels:
        return ""
    return " including " + ", ".join(labels)


def _short_atom_list(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "none"
    return ",".join(str(item) for item in value[:8])


def _esipt_pair_text(metrics: dict[str, Any]) -> str:
    pairs = metrics.get("proton_transfer_pair_proxies")
    if not isinstance(pairs, list) or not pairs:
        return "no short intramolecular proton donor/acceptor pair was resolved"
    first = pairs[0]
    if not isinstance(first, dict):
        return "intramolecular proton donor/acceptor pair proxy was detected"
    return (
        "nearest intramolecular proton-transfer pair proxy is "
        f"{first.get('pair_type', 'donor-H...acceptor')} "
        f"({first.get('donor_atom', 'donor')} to "
        f"{first.get('acceptor_atom', 'acceptor')}, "
        f"{first.get('topological_distance', 'unknown')} bonds apart)"
    )


def _pick(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys}


def _fraction(numerator: Any, denominator: float) -> float:
    return round(float(numerator or 0) / max(denominator, 1.0), 6)


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _bounded_macro_tool_args(
    execution_plan: AgentExecutionPlan | None,
) -> dict[str, Any]:
    raw = execution_plan.tool_args if execution_plan is not None else {}
    focus_tags = raw.get("focus_tags")
    if not isinstance(focus_tags, list):
        return {}
    tags = [str(item).strip() for item in focus_tags if str(item).strip()]
    return {"focus_tags": tags[:8]} if tags else {}
