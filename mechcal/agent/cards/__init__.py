"""Capability cards: what each tool computes and what can be varied, plus molecule-specific applicability.

The static text lives in tool_cards.json (frozen with the system). Applicability is computed by code
from the structural facts only (elements, size, charge, electron parity, force-field preparation),
before the Planner's first turn; cost and input readiness are added by the agent each turn.
Cards never name a mechanism family and never recommend a tool for a purpose.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

CARDS_PATH = Path(__file__).with_name("tool_cards.json")


@lru_cache(maxsize=1)
def load_cards() -> dict:
    return json.loads(CARDS_PATH.read_text(encoding="utf-8"))


def cards_sha256() -> str:
    return hashlib.sha256(CARDS_PATH.read_bytes()).hexdigest()


def visible_tools() -> list[str]:
    return list(load_cards()["tools"])


def molecule_profile(facts: dict, prepared_kind: str) -> dict:
    """What applicability is judged on: only information derived from the SMILES."""
    s = facts["summary"]
    from rdkit import Chem
    mol = Chem.MolFromSmiles(facts["smiles"])
    electrons = sum(a.GetAtomicNum() for a in mol.GetAtoms()) + sum(a.GetTotalNumHs() for a in mol.GetAtoms()) - s["net_charge"]
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return {"elements": set(s["elements"]), "atoms": int(s["atoms_with_h"]), "charge": int(s["net_charge"]),
            "closed_shell": electrons % 2 == 0 and radicals == 0, "fragments": int(s["fragments"]),
            "force_field": prepared_kind == "prepared_structure"}


def applicability(tool: str, profile: dict) -> dict:
    """{'ok': bool, 'why': str} from the card's declared scope."""
    scope = load_cards()["tools"][tool].get("scope", {})
    reasons = []
    allowed = scope.get("elements")
    if allowed and not profile["elements"] <= set(allowed):
        reasons.append(f"elements {sorted(profile['elements'] - set(allowed))} are outside its scope")
    if scope.get("max_atoms") and profile["atoms"] > scope["max_atoms"]:
        reasons.append(f"{profile['atoms']} atoms exceed its limit of {scope['max_atoms']}")
    charge = scope.get("charge")
    if charge == "neutral" and profile["charge"] != 0:
        reasons.append(f"needs a neutral molecule (charge {profile['charge']:+d})")
    elif isinstance(charge, list) and not charge[0] <= profile["charge"] <= charge[1]:
        reasons.append(f"charge {profile['charge']:+d} is outside {charge[0]:+d}..{charge[1]:+d}")
    elif charge == "neutral unless the molecule is pure CHNO" and profile["charge"] != 0 \
            and not profile["elements"] <= {"C", "H", "N", "O"}:
        reasons.append("needs a neutral molecule unless it is pure CHNO")
    if scope.get("closed_shell") and not profile["closed_shell"]:
        reasons.append("needs a closed-shell molecule")
    if scope.get("needs_force_field") and not profile["force_field"]:
        reasons.append("no MMFF94/UFF force-field parameters for this molecule")
    if scope.get("single_molecule") and profile["fragments"] != 1:
        reasons.append("needs a single connected molecule")
    return {"ok": not reasons, **({"why": "; ".join(reasons)} if reasons else {})}


def professor_view(tool: str, profile: dict, args_spec: dict, arg_class: dict) -> dict:
    """Card for the Planner: full static text, applicability, the settings with their class and range."""
    card = load_cards()["tools"][tool]
    settings = {k: {**args_spec.get(k, {}), "what": v, "class": arg_class.get(k, "technical")}
                for k, v in card["you_can_vary"].items()}
    return {"tool": tool, "computes": card["computes"], "you_can_vary": settings, "returns": card["returns"],
            "feeds_into": card["feeds_into"], "input": card.get("scope", {}).get("input"),
            "applies_to_this_molecule": applicability(tool, profile), "does_not_tell": card["does_not_tell"]}


def recipe_view(name: str, description: str, args: dict, profile: dict) -> dict:
    recipe = load_cards()["recipes"].get(name, {})
    groups = recipe.get("requires_any", [])
    verdicts = [[(t, applicability(t, profile)) for t in g if t in load_cards()["tools"]] for g in groups]
    ok = not groups or any(all(v["ok"] for _, v in g) for g in verdicts)
    why = "; ".join(f"{t.split('.')[-1]}: {v['why']}" for g in verdicts for t, v in g if not v["ok"])
    return {"recipe": name, "combines": recipe.get("combines", []), "description": description,
            **({"args": args} if args else {}),
            "applies_to_this_molecule": {"ok": ok, **({"why": why} if not ok else {})}}


def _arg_spec(v: dict, defs: dict) -> dict:
    spec = {key: v[key] for key in ("type", "default", "minimum", "maximum", "enum", "minItems", "maxItems")
            if key in v and v[key] is not None}
    if v.get("description"):
        spec["description"] = v["description"][:220]
    items = v.get("items") or next((x.get("items") for x in v.get("anyOf", []) if x.get("items")), None)
    if isinstance(items, dict) and items.get("$ref"):
        items = defs.get(items["$ref"].rsplit("/", 1)[-1], items)
    if isinstance(items, dict) and items.get("enum"):
        spec["items_enum"] = items["enum"]
    if isinstance(items, dict) and items.get("properties"):        # list of objects: name the fields
        spec["item_fields"] = {f: fv.get("type") or "|".join(str(x.get("type")) for x in fv.get("anyOf", []))
                               for f, fv in items["properties"].items()}
        spec["item_required"] = items.get("required", [])
    if isinstance(items, dict) and items.get("type") == "array":
        spec["items"] = "array (list of lists)"
    if not spec.get("type") and v.get("anyOf"):
        spec["type"] = "|".join(str(x.get("type")) for x in v["anyOf"] if x.get("type"))
    return spec


def tool_entries(gateway) -> list[dict]:
    """Carded tools available in this deployment, with argument specs from the tools' own schemas.
    Also restricts the gateway to exactly these tools (the same set for every method)."""
    visible = set(visible_tools())
    ids = [i for i in gateway.tool_ids() if i in visible]
    entries = []
    for d in gateway.describe(ids):
        schema = d.get("tool_arg_schema") or {}
        args = {k: _arg_spec(v, schema.get("$defs") or {}) for k, v in (schema.get("properties") or {}).items()}
        static = load_cards()["tools"][d["capability_id"]]
        kinds = list(dict.fromkeys([*(d.get("required_artifact_kinds") or []), *(d.get("source_artifact_kinds") or [])]))
        entries.append({"tool": d["capability_id"], "status": d.get("status"), "computes": static["computes"],
                        "returns": static["returns"], "does_not_tell": static["does_not_tell"],
                        "input_kinds": kinds, "args": args})
    gateway.allowed = tuple(e["tool"] for e in entries)
    return entries


def catalog(entries: list[dict], profile: dict, arg_class: dict, kinds_present: set, cost=None) -> list[dict]:
    """The tool list every method sees: card text, settings, applicability, input readiness, cost."""
    out = []
    for e in entries:
        view = professor_view(e["tool"], profile, e["args"], arg_class)
        missing = [k for k in e["input_kinds"] if k not in kinds_present]
        if missing and len(missing) == len(e["input_kinds"]):
            view["input_ready"] = f"needs {' or '.join(missing)} (none yet)"
        if cost is not None:
            view.update(cost(e["tool"]))
        out.append(view)
    return out
