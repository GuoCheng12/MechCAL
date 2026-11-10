"""Live audit wire contract derived from persisted models and request scope."""

from jsonschema import Draft202012Validator

from mechcal.runtime.retry import ModelContractError
from mechcal.runtime.structured_output import _strict
from mechcal.schemas.mechanisms import PhotophysicsReview
from mechcal.schemas.contracts import object_variants
from mechcal.runtime.scoped_contracts import restrict_items


def audit_output_schema(payload):
    schema = PhotophysicsReview.model_json_schema()
    # Provenance is written by the runtime, not generated as an audit concern.
    schema["properties"].pop("context_notes", None)
    definitions = schema["$defs"]
    if payload["require_argument_assessments"]:
        definitions["MechanismArgumentAssessment"] = payload["argument_assessment_contract"]
        audit = definitions["MechanismSupportAudit"]
        audit["properties"]["argument_assessment"] = {"$ref": "#/$defs/MechanismArgumentAssessment"}
        for field in ("recommended_support_level", "boundary_patch"):
            audit["properties"][field] = {"type": "null"}
    ids = payload["allowed_support_ids"]
    if ids:
        definitions["MechanismSupportAudit"]["properties"]["support_id"] = {"type": "string", "enum": ids}
    audits = schema["properties"]["support_argument_audits"]
    if payload["require_complete_support_audit"]:
        audits.update(minItems=len(ids), maxItems=len(ids))
    if not ids:
        audits["maxItems"] = 0
    if payload.get("require_explicit_audit_routing"):
        schema["properties"]["routing"] = {"$ref": "#/$defs/AuditRouting"}
        routing = definitions["AuditRouting"]
        restrict_items(routing["properties"]["affected_support_ids"], ids)
        definitions["AuditRouting"] = object_variants(routing, [
            {"mode": {"type": "string", "const": "no_change"}, "affected_support_ids": {"maxItems": 0}},
            {"mode": {"type": "string", "const": "local_audit"}, "affected_support_ids": {"minItems": 1}},
            {"mode": {"type": "string", "const": "global"}},
        ])
    if "compact_evidence_ids" in payload:
        refs = sorted(set(payload["compact_evidence_ids"]) & set(payload["allowed_evidence_ids"]))
        for definition, fields in {"PhotophysicsCoverageAxis": ["evidence_refs"],
                                   "MechanismHypothesisCard": ["support_evidence_refs", "weakening_evidence_refs"]}.items():
            for name in fields:
                restrict_items(definitions[definition]["properties"][name], refs)
    schema = _strict(schema)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_audit_output(response, schema):
    errors = list(Draft202012Validator(schema).iter_errors(response))
    if errors:
        raise ModelContractError("Audit output schema violations:\n" + "\n".join(
            f"{list(error.absolute_path)}: {error.message}" for error in errors
        ))
