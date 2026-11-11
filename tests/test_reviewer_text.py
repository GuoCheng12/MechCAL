"""Scientific review is prose; the runtime owns its scope and storage envelope."""

import json

import pytest

from mechcal.runtime.command_loop import _comment, planner_payload, save
from mechcal.runtime.command_operations import OperationRejected, check_refs, protocol, put, revise
from mechcal.runtime.llm import OpenAIJsonClient, TextCompletionResult
from mechcal.schemas import CaseRun
from test_command_protocol import ScriptClient, case, judgment, orchestrator, store
from test_command_worker_text import chat_body
from test_responses_api import response_body, settings, wire  # noqa: F401

ROLES = ["coverage_reviewer", "reviewer", "final_reviewer"]
TEXT = 'The "weak" transition is not conclusive.\n\nConsider alternatives {A,B}.\nUnits and paths \\ remain text.'


def review_case(tmp_path, client, role):
    c = revise(case(), judgment())
    c = put(c, phase="review", review_role=role,
            review_changes=[change.model_dump(mode="json") for change in judgment().changes])
    o, s = orchestrator(tmp_path, reviews=True), store(tmp_path, c)
    o.deps.scientific_reviewer.llm_client = client
    return o, c, s


@pytest.mark.parametrize("role", ROLES)
def test_multiline_review_is_not_json_and_cannot_mutate_science(tmp_path, role):
    client = ScriptClient([TEXT])
    o, c, s = review_case(tmp_path, client, role)
    before = c.model_dump(mode="json")
    reviewed = _comment(o, c, s)
    feedback = protocol(reviewed)["review"]
    assert feedback == {"status": "available", "text": TEXT,
                        "reviewed_record_ids": ["A1", "input:smiles"]}
    assert not {"response_format", "tools", "tool_choice"} & client.calls[0].keys()
    assert "not a JSON" in client.calls[0]["system_prompt"]
    assert planner_payload(o, reviewed)["review_feedback"] == feedback
    for field in ("portfolio", "evidence_ledger", "claim_ledger", "artifact_manifest"):
        assert reviewed.model_dump(mode="json")[field] == before[field]
    for key in ("results", "arguments", "ranking", "revision", "rounds"):
        assert protocol(reviewed)[key] == protocol(c)[key]
    save(reviewed, s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert protocol(restored)["review"] == feedback
    events = [json.loads(line) for line in (s.root / "history/command_requests.jsonl").read_text().splitlines()]
    assert len(events) == 2 and all(e["transport"] == "plain_text" and e["role"] == role for e in events)
    assert events[-1]["raw_text"] == TEXT


@pytest.mark.parametrize("protocol_name", ["responses", "chat_completions"])
@pytest.mark.parametrize("role", ROLES)
def test_reviewer_wire_omits_output_schema_but_preserves_model_settings(wire, tmp_path, protocol_name, role):
    requests, replies = wire
    replies.append((200, response_body(TEXT) if protocol_name == "responses" else chat_body(TEXT)))
    client = OpenAIJsonClient(settings(api_protocol=protocol_name, reasoning_effort="high",
                                       max_tokens=1024, temperature=None, top_p=1.0))
    o, c, s = review_case(tmp_path, client, role)
    reviewed = _comment(o, c, s)
    assert protocol(reviewed)["review"]["text"] == TEXT
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    assert not {"response_format", "text", "tools", "tool_choice"} & body.keys()
    assert body.get("reasoning", {}).get("effort", body.get("reasoning_effort")) == "high"
    assert body.get("max_output_tokens", body.get("max_tokens")) == 1024


@pytest.mark.parametrize("reply", [
    TextCompletionResult(raw_text="", finish_reason="stop", usage={}),
    TextCompletionResult(raw_text=" \n\t ", finish_reason="stop", usage={}),
    TextCompletionResult(raw_text="Partial review", finish_reason="length", usage={}),
    TextCompletionResult(raw_text="Partial review", finish_reason="max_output_tokens", usage={}),
    TextCompletionResult(raw_text="Refusal", finish_reason="stop", response_error="Refused.", usage={}),
])
def test_unavailable_review_preserves_state_and_records_scope(tmp_path, reply):
    client = ScriptClient([reply])
    o, c, s = review_case(tmp_path, client, "final_reviewer")
    reviewed = _comment(o, c, s)
    feedback = protocol(reviewed)["review"]
    assert feedback["status"] == "unavailable" and "text" not in feedback
    assert feedback["reviewed_record_ids"] == ["A1", "input:smiles"]
    assert feedback["issues"][0]["code"] == "unavailable_review"
    assert protocol(reviewed)["arguments"] == protocol(c)["arguments"]
    assert protocol(reviewed)["results"] == protocol(c)["results"]
    assert reviewed.portfolio == c.portfolio and len(client.calls) == 1


def test_transport_failure_keeps_checkpoint_and_does_not_invent_feedback(tmp_path):
    o, c, s = review_case(tmp_path, ScriptClient([TimeoutError("offline timeout")]), "final_reviewer")
    c = save(c, s)
    with pytest.raises(TimeoutError):
        _comment(o, c, s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert restored == c
    events = [json.loads(line) for line in (s.root / "history/command_requests.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "transport_error" and events[-1]["role"] == "final_reviewer"


def test_text_cannot_inject_scope_refs_or_commands(tmp_path):
    text = '{"reviewed_record_ids":["foreign:E1"],"command":"compute"}\nAdvice mentioning foreign:E1.'
    o, c, s = review_case(tmp_path, ScriptClient([text]), "reviewer")
    reviewed = _comment(o, c, s)
    feedback = protocol(reviewed)["review"]
    assert feedback["text"] == text and feedback["reviewed_record_ids"] == ["A1", "input:smiles"]
    assert "refs" not in feedback and "command" not in feedback
    assert protocol(reviewed)["rounds"] == 0
    with pytest.raises(OperationRejected):
        check_refs(reviewed, ["foreign:E1"], ["refs"], scientific=True)


def test_old_review_checkpoint_still_reads_without_rewriting_refs(tmp_path):
    c = put(case(), review={"status": "available", "text": "Legacy advice.", "refs": ["input:smiles"]})
    s = store(tmp_path, c)
    save(c, s)
    restored = CaseRun.model_validate_json((s.root / "command_checkpoint.json").read_text())
    assert planner_payload(orchestrator(tmp_path), restored)["review_feedback"] == protocol(c)["review"]


def test_final_review_transport_interruption_resumes_without_repeating_judgment(tmp_path):
    o = orchestrator(tmp_path, reviews=True)
    final = {"command": {"kind": "finalize", "answer": "Offline bounded answer.", "limitations": []}}
    o.deps.planner.llm_client = ScriptClient([
        {"command": judgment().model_dump(mode="json")}, final,
    ])
    reviews = ScriptClient(["Initial coverage advice.", "Regular review.\n\nNo endorsement.",
                            TimeoutError("offline final review timeout")])
    o.deps.scientific_reviewer.llm_client = reviews
    with pytest.raises(TimeoutError):
        o.run(smiles="CCO", user_query="Offline review-resume fixture.", case_id="review-resume")
    checkpoint = tmp_path / "review-resume/command_checkpoint.json"
    saved = CaseRun.model_validate_json(checkpoint.read_text())
    assert saved.status == "paused" and protocol(saved)["phase"] == "review"
    assert protocol(saved)["review_role"] == "final_reviewer"
    o.deps.scientific_reviewer.llm_client = ScriptClient([TEXT])
    o.deps.planner.llm_client = ScriptClient([final])
    resumed = o.run(smiles="CCO", user_query="Offline review-resume fixture.", case_id="review-resume", resume=True)
    assert resumed.status == "finalized"
    for key in ("results", "arguments", "ranking", "revision", "rounds"):
        assert protocol(resumed)[key] == protocol(saved)[key]
    assert len(o.deps.scientific_reviewer.llm_client.calls) == 1
    assert len(o.deps.planner.llm_client.calls) == 1
    assert o.deps.planner.llm_client.calls[0]["payload"]["review_feedback"]["text"] == TEXT
