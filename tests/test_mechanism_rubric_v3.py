import hashlib
import json
from pathlib import Path

import pytest

from mechcal.eval import mechanism_ranking
from mechcal.mechanism_pool import MECHANISM_POOL


RUBRICS = Path(mechanism_ranking.__file__).parent / "rubrics"


def read_v3():
    return json.loads((RUBRICS / "mechanism_evidence_rubric_v3.json").read_text())


def test_v2_protocol_remains_frozen():
    assert hashlib.sha256(
        (RUBRICS / "mechanism_evidence_rubric_v2.json").read_bytes()
    ).hexdigest() == "70b40d6893a3077b7dceb4ef095779d628b1361b1c09b63c210131db5e9931dc"
    prompt = RUBRICS.parent / "prompts/mechanism_evidence_support_judge.md"
    assert hashlib.sha256(prompt.read_bytes()).hexdigest() == (
        "c476de71705da0439918d175e3ced4b5f0cfef8800fc7b13030803a87b547138"
    )
    assert mechanism_ranking.load_mechanism_evidence_rubric()[
        "mechanism_evidence_rubric_version"
    ] == "v2"
    assert mechanism_ranking.MECHANISM_EVIDENCE_SUPPORT_PROMPT_VERSION == "2026-06-05"


def test_v3_pool_and_decision_order():
    rubric = read_v3()
    assert rubric["mechanism_evidence_rubric_version"] == "v3"
    assert rubric["status"] == "draft_not_enabled"
    assert rubric["support_score_scale"] == {"strong": 1.0, "partial": 0.5, "unsupported": 0.0}
    assert tuple(rubric["mechanism_pool"]) == MECHANISM_POOL
    assert set(rubric["labels"]) == set(MECHANISM_POOL)
    policy = rubric["decision_policy"]
    assert policy["order"] == ["strong_support", "partial_support", "zero"]
    assert [policy[key]["score"] for key in policy["order"]] == [1.0, 0.5, 0.0]
    assert "unresolved record-availability" in policy["zero"]["all_of"][1]
    assert "not a fourth support level" in policy["missing_records"]


@pytest.mark.parametrize("label", MECHANISM_POOL)
def test_v3_each_route_requires_a_complete_combination(label):
    entry = read_v3()["labels"][label]
    route_ids = []
    for level in ("strong_support", "partial_support"):
        assert set(entry[level]) == {"any_of"}
        assert entry[level]["any_of"]
        for route in entry[level]["any_of"]:
            assert set(route) == {"id", "all_of"}
            assert len(route["all_of"]) >= 2
            assert all(isinstance(c, str) and c.strip() for c in route["all_of"])
            route_ids.append(route["id"])
    assert len(route_ids) == len(set(route_ids))
    assert entry["zero_support_examples"]
    assert entry["confounder_boundaries"]
    assert entry["typical_tools_or_proxies"]


def test_v3_verification_distinguishes_missing_from_contradicted():
    policy = read_v3()["verification_policy"]
    assert set(policy["outcomes"]) == {
        "consistent", "contradicted", "beyond_source", "unavailable"
    }
    assert "runner-controlled" in policy["tool_findings"]["source"]
    assert "SMILES" in policy["structural_findings"]["source"]
    assert "Absence of tool calls is not a penalty" in policy["zero_shot_fairness"]
    assert "Hidden references" in policy["experimental_findings"]["checks"][1]


def test_v3_rules_preserve_overlap_and_do_not_reward_tool_calls():
    rules = read_v3()["global_rules"]
    assert any("Evidence can support multiple coexisting mechanisms" in rule for rule in rules)
    assert any("Do not reward a tool name" in rule for rule in rules)
    assert any("Missing strong-support evidence does not automatically imply zero" in rule for rule in rules)
    assert any("absence of a stock caveat cannot erase valid partial support" in rule for rule in rules)


def test_v3_is_inspectable_but_cannot_be_sent_to_legacy_judge(monkeypatch):
    rubric = mechanism_ranking.load_mechanism_evidence_rubric(
        RUBRICS / "mechanism_evidence_rubric_v3.json"
    )

    def forbidden_client(*args, **kwargs):
        pytest.fail("A draft must be rejected before constructing the model client")

    monkeypatch.setattr(mechanism_ranking, "OpenAIJsonClient", forbidden_client)
    with pytest.raises(ValueError, match="requires source-verification integration"):
        mechanism_ranking.LLMMechanismEvidenceSupportJudge(rubric=rubric)


def test_documented_route_counts():
    entries = read_v3()["labels"].values()
    assert sum(len(e["strong_support"]["any_of"]) for e in entries) == 22
    assert sum(len(e["partial_support"]["any_of"]) for e in entries) == 20


def test_strong_support_does_not_require_complete_causal_validation():
    rubric = read_v3()
    rules = rubric["global_rules"]
    assert any("not proof of the complete causal chain" in rule for rule in rules)
    assert any("as universal prerequisites" in rule for rule in rules)
    assert any("Do not require exclusion of all plausible alternatives" in rule for rule in rules)
    assert len(rubric["decision_policy"]["strong_support"]["all_of"]) == 2
    rim = rubric["labels"]["RIM_RIR_RIV"]["strong_support"]["any_of"]
    assert len(rim[0]["all_of"]) == 2
    assert "Separate emission measurements" in rim[0]["all_of"][1]
    assert "not mandatory" in rim[0]["all_of"][1]
    assert "dynamics comparison is not mandatory" in rim[1]["all_of"][1]


def test_partial_raci_does_not_require_a_stock_caveat():
    entry = read_v3()["labels"]["RACI_CI_ACCESS"]
    conditions = entry["partial_support"]["any_of"][0]["all_of"]
    assert len(conditions) == 2
    assert "case-specific explanation" in conditions[1]
    assert any("Do not require the answer to contain a stock disclaimer" in text
               for text in entry["confounder_boundaries"])


def test_verification_requires_only_material_records():
    rubric = read_v3()
    scope = rubric["verification_policy"]["scope"]
    assert "Do not require tool logs for a structural fact checkable from SMILES" in scope
    assert "Missing auxiliary files or metadata" in scope
    assert "essential" in rubric["decision_policy"]["missing_records"]
