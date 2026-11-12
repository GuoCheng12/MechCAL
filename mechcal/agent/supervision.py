"""Professor-student protocol: deterministic rules around the Worker (student).

The Planner (professor) decides the science of an assignment; the Worker (student) carries it
out and fixes technical problems. Everything here is plain code, so the rules hold regardless of
what the models write:

- every tool argument is classified technical or scientific (``ARG_CLASS``; a test keeps the table
  complete); the geometry source is checked through recorded artifact lineage;
- a student call that changes a fixed or already-used scientific setting, or leaves the assigned
  route, is intercepted (not executed, not charged) and must come back as ``needs_approval``;
- after a failure the student has at most ``MAX_RETRIES`` further attempts per route, each must
  change something, a timed-out route is not retried if the wall time left cannot exceed the
  failed attempt, and a not-applicable failure is never retried;
- the student's report is validated and corrected (e.g. ``done`` with unresolved warnings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_RETRIES = 2          # further attempts after the first failure of a route, per task
MAX_REVISIONS = 3        # professor revisions per task

M = "microscopic."
ARG_CLASS: dict[str, str] = {
    # technical: the student may change these to make a calculation work
    "max_wall_seconds": "technical", "max_opt_cycles": "technical", "fmax_ev_per_angstrom": "technical",
    "s1_nstates": "technical", "nstates": "technical", "nstates_per_spin": "technical", "td_tout": "technical",
    "grid_level": "technical", "scf_solver": "technical", "spatial_grid": "technical",
    "angles": "technical", "angles_degrees": "technical", "distances_angstrom": "technical", "values": "technical",
    "max_members": "technical", "max_candidates": "technical", "md_length_ps": "technical",
    "energy_window_kcal_mol": "technical", "search_level": "technical",
    # scientific: method, state, coordinate, partition, environment, input structure
    "functional": "scientific", "basis": "scientific", "solvents": "scientific", "state_root": "scientific",
    "dihedral_id": "scientific", "dihedral_atoms": "scientific", "atom_indices": "scientific",
    "coordinate": "scientific", "coordinate_mode": "scientific", "fragments": "scientific",
    "fragment_atom_indices": "scientific", "donor_atoms": "scientific", "acceptor_atoms": "scientific",
    "charge_schemes": "scientific", "measurements": "scientific", "conformer_ids": "scientific",
    "member_id": "scientific", "partner_artifact_id": "scientific",
    "source_artifact_id": "source",      # checked through lineage, not by value
}

# Tools a recipe runs internally (a student may call them directly inside that recipe's task).
RECIPE_TOOLS: dict[str, set[str]] = {
    "excited_states": {M + "run_baseline_bundle"},
    "charge_transfer": {M + "run_baseline_bundle", M + "analyze_hole_electron_from_bundle"},
    "bright_dark_states": {M + "run_bright_dark_state_ordering"},
    "torsion_response": {M + "list_rotatable_dihedrals", M + "run_torsion_brightness_coupling_scan"},
    "spin_states": {M + "compare_singlet_triplet_states", M + "evaluate_spin_orbit_couplings", M + "compare_spin_states_pyscf"},
    "redox_levels": {M + "estimate_vertical_ip_ea"},
    "vibrations": {M + "analyze_ground_state_vibrations"},
    "dimer_contacts": {M + "search_dimer_contacts"},
    "solvent_polarity": {M + "run_solvation_polarity_proxy"},
    "conformers": {M + "sample_conformers_crest"},
}
# Preparation steps any task may use (they produce inputs, not evidence).
PREREQUISITE_TOOLS = {M + "optimize_ground_state_geometry", M + "list_rotatable_dihedrals", M + "run_baseline_bundle"}
# With the Worker on, the Planner may itself run only single-step, parameter-free tools.
PLANNER_DIRECT_TOOLS = {M + "list_rotatable_dihedrals", M + "select_computed_geometry",
                        M + "extract_geometry_descriptors_from_bundle"}
GEOMETRY_KINDS = {"prepared_structure", "initial_structure"}
REPORT_STATUSES = ("done", "done_with_warnings", "blocked_technical", "needs_approval", "not_applicable")


@dataclass
class Attempt:
    n: int
    kind: str                       # recipe | tool
    name: str
    args: dict
    outcome: str                    # success | partial | failed | not_applicable | error | ...
    artifact_ids: list[str] = field(default_factory=list)
    failure_class: str | None = None
    failure_code: str | None = None
    failure: str | None = None
    warnings: list[dict] = field(default_factory=list)
    changed: dict = field(default_factory=dict)        # computed by the system vs the previous attempt of this route
    reason: str = ""                                    # the student's own explanation
    wall_seconds: float = 0.0

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "", [], {})}


@dataclass
class Task:
    task_id: str
    objective: str
    recipe: str | None
    tools: list[str]
    fixed_args: dict
    separates: list[str]
    expectation: Any
    created_step: int
    attempts: list[Attempt] = field(default_factory=list)
    instructions: list[dict] = field(default_factory=list)      # professor's revisions
    pinned: dict = field(default_factory=dict)                  # scientific values in force (fixed or first used)
    geometry: str | None = None                                 # pinned input geometry
    approved: list[dict] = field(default_factory=list)          # approved route/argument changes
    pending: dict | None = None                                 # change the student asked approval for
    interceptions: list[dict] = field(default_factory=list)
    reports: list[dict] = field(default_factory=list)
    retry_base: int = 0                                         # attempts before the last revision do not count
    wall_budget: float | None = None                            # tool seconds this task may use (set by the Planner)
    estimate: dict | None = None                                # cost estimate at delegation

    def route_allowed(self, kind: str, name: str) -> bool:
        if kind == "recipe":
            return name == self.recipe or any(a.get("recipe") == name for a in self.approved)
        allowed = set(self.tools) | PREREQUISITE_TOOLS | RECIPE_TOOLS.get(self.recipe or "", set())
        allowed |= {a["tool"] for a in self.approved if a.get("tool")}
        return name in allowed

    def route_attempts(self, kind: str, name: str) -> list[Attempt]:
        return [a for a in self.attempts[self.retry_base:] if a.kind == kind and a.name == name]

    def summary(self) -> dict:
        last = self.reports[-1] if self.reports else {}
        return {"task_id": self.task_id, "objective": self.objective[:160], "status": last.get("status", "open"),
                "attempts": len(self.attempts), "revisions": len(self.instructions),
                "findings": [f["finding_id"] for f in last.get("findings", [])]}


def recipe_tool_args(recipe: str, args: dict, recipe_args: dict) -> dict:
    """Recipe arguments translated to the tool argument names they feed (for classification)."""
    out = {}
    for key, value in args.items():
        target = recipe_args.get(recipe, {}).get(key)
        out[target[1] if target else key] = value
    return out


def _diff(old: dict, new: dict) -> dict:
    keys = set(old) | set(new)
    return {k: {"from": old.get(k), "to": new.get(k)} for k in sorted(keys) if old.get(k) != new.get(k)}


def check_call(task: Task, kind: str, name: str, args: dict, *, gateway, defaults: dict, accepted: set,
               recipe_args: dict) -> tuple[dict | None, dict]:
    """Return (interception or None, effective args). Effective args carry injected fixed settings.

    ``accepted``: argument names the tool (or recipe) takes; ``defaults``: the tool's defaults.
    """
    args = dict(args or {})
    if not task.route_allowed(kind, name):
        return {"code": "route_change", "needs_approval": True,
                "message": f"{kind} {name} is outside this assignment (assigned: recipe={task.recipe}, tools={task.tools}). "
                           "A different tool or route is a scientific change: report needs_approval with the proposed change."}, args

    reverse = {targ: rarg for rarg, (_, targ) in recipe_args.get(name, {}).items()} if kind == "recipe" else {}
    for key, value in task.fixed_args.items():          # fixed settings: inject when absent, intercept when changed
        if key == "source_artifact_id":
            continue
        target = key if key in accepted else reverse.get(key)
        if target is None:
            if kind == "recipe":
                tools = sorted(RECIPE_TOOLS.get(name, set()))
                return {"code": "recipe_cannot_honor", "needs_approval": False,
                        "message": f"Recipe {name} cannot set the fixed {key}={value!r}; call the tool directly ({tools})."}, args
            continue                                     # this tool has no such setting
        if target in args and args[target] != value:
            return {"code": "fixed_setting_changed", "needs_approval": True,
                    "message": f"{key} is fixed by the Planner to {value!r}; you used {args[target]!r}. Ask for approval."}, args
        args.setdefault(target, value)
    tool_view = recipe_tool_args(name, args, recipe_args) if kind == "recipe" else dict(args)
    # scientific settings already in force in this task (first use pins them)
    for key, value in tool_view.items():
        if ARG_CLASS.get(key) != "scientific":
            continue
        if key in task.pinned and task.pinned[key] != value:
            return {"code": "scientific_change", "needs_approval": True,
                    "message": f"{key} was {task.pinned[key]!r} in this task; changing it to {value!r} is a scientific "
                               "change. Report needs_approval with the proposed change and why."}, args
    for key, value in defaults.items():   # omitted scientific args fall back to the tool default
        if ARG_CLASS.get(key) == "scientific" and key not in tool_view and key in task.pinned and task.pinned[key] != value:
            return {"code": "scientific_change", "needs_approval": True,
                    "message": f"{key} is {task.pinned[key]!r} in this task; omitting it would use the default {value!r}."}, args
    # the target state must stay visible
    root = task.pinned.get("state_root", task.fixed_args.get("state_root"))
    for key in ("s1_nstates", "nstates"):
        if isinstance(root, int) and isinstance(tool_view.get(key), int) and tool_view[key] < root:
            return {"code": "target_state_dropped", "needs_approval": True,
                    "message": f"{key}={tool_view[key]} would no longer include the target state {root}."}, args
    # geometry lineage
    source = args.get("source_artifact_id")
    if task.geometry:
        if source is None and kind == "tool":
            return {"code": "source_required", "needs_approval": False,
                    "message": f"Give source_artifact_id explicitly: {task.geometry} or an artifact computed from it."}, args
        if source is None and kind == "recipe":
            args["source_artifact_id"] = task.geometry
        elif source is not None and not gateway.derived_from(source, task.geometry):
            return {"code": "geometry_change", "needs_approval": True,
                    "message": f"{source} is not computed from this task's geometry {task.geometry}. "
                               "Using another geometry is a scientific change."}, args
    # retries
    previous = task.route_attempts(kind, name)
    if any(a.failure_class == "not_applicable" for a in previous):
        return {"code": "not_applicable", "needs_approval": False,
                "message": f"{name} cannot treat this molecule ({previous[-1].failure}). Do not retry; report not_applicable."}, args
    failures = [a for a in previous if a.outcome not in {"success"} or any(w["student_fixable"] for w in a.warnings)]
    if failures and len(failures) > MAX_RETRIES:
        return {"code": "retry_limit", "needs_approval": False,
                "message": f"{name} failed {len(failures)} times in this task (limit: first attempt + {MAX_RETRIES} retries). "
                           "Report blocked_technical with each attempt and a suggested next step."}, args
    if any(a.args == args for a in previous):
        return {"code": "no_change", "needs_approval": False,
                "message": "Identical to an earlier attempt in this task; every retry must change something."}, args
    last = previous[-1] if previous else None
    if last and last.failure_code == "timeout":
        wall = args.get("max_wall_seconds", last.args.get("max_wall_seconds"))
        others_same = {k: v for k, v in args.items() if k != "max_wall_seconds"} == \
                      {k: v for k, v in last.args.items() if k != "max_wall_seconds"}
        prev_wall = last.args.get("max_wall_seconds") or last.wall_seconds
        if others_same and isinstance(prev_wall, (int, float)) and min(float(wall or 0), gateway.wall_seconds_left()) <= prev_wall:
            return {"code": "cannot_fit", "needs_approval": False,
                    "message": f"The last attempt timed out at {prev_wall:.0f} s and at most {gateway.wall_seconds_left():.0f} s "
                               "is left; the same settings cannot finish. Change a technical setting or report blocked_technical."}, args
    return None, args


def pin_after(task: Task, kind: str, name: str, args: dict, *, defaults: dict, recipe_args: dict,
              geometry_used: str | None) -> None:
    """After an executed call: pin scientific values and the geometry used."""
    view = recipe_tool_args(name, args, recipe_args) if kind == "recipe" else args
    for key, value in {**{k: v for k, v in defaults.items() if ARG_CLASS.get(k) == "scientific"}, **view}.items():
        if ARG_CLASS.get(key) == "scientific":
            task.pinned.setdefault(key, value)
    if geometry_used and not task.geometry:
        task.geometry = geometry_used


def changed_from_previous(task: Task, kind: str, name: str, args: dict) -> dict:
    previous = task.route_attempts(kind, name)
    return _diff(previous[-1].args, args) if previous else {}


def validate_report(task: Task, report: dict) -> tuple[dict, list[str]]:
    """System check of the student's report; returns the corrected report and the corrections made."""
    report = dict(report)
    corrections = []
    status = report.get("status")
    run = task.attempts[task.retry_base:]
    succeeded = [a for a in run if a.outcome in {"success", "partial"}]
    unresolved = [w for w in (succeeded[-1].warnings if succeeded else [])]
    last_failure = run[-1] if run and run[-1].outcome not in {"success", "partial"} else None
    if status not in REPORT_STATUSES:
        corrections.append(f"status {status!r} is not allowed")
        status = "blocked_technical"
    if status in {"done", "done_with_warnings"} and not succeeded:
        new = "not_applicable" if last_failure and last_failure.failure_class == "not_applicable" else "blocked_technical"
        corrections.append(f"{status} -> {new}: no attempt in this task succeeded")
        status = new
    elif status == "done" and unresolved:
        corrections.append("done -> done_with_warnings: the latest result has unresolved warnings")
        status = "done_with_warnings"
    if status == "not_applicable" and not any(a.failure_class == "not_applicable" for a in run):
        corrections.append("not_applicable -> blocked_technical: no failure was classified as not applicable")
        status = "blocked_technical"
    if status == "needs_approval" and not (report.get("proposed_change") or task.pending):
        corrections.append("needs_approval without a proposed change")
    report["status"] = status
    report["warnings"] = [w for w in unresolved] + [w for w in report.get("warnings", []) if isinstance(w, dict)]
    return report, corrections
