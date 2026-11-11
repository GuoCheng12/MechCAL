"""Commentary is prose, not a transport-success receipt or executable text."""

from dataclasses import replace
import json

import pytest

from mechcal.agents.command_worker import scoped_commentary_turn
from mechcal.runtime.command_compact import expand_text
from mechcal.runtime.command_loop import _comment, save
from mechcal.runtime.command_operations import protocol, put, record_index
from mechcal.runtime.snapshot_storage import read_case_run
from test_command_protocol import case, store, ScriptClient
from test_native_commands import NativeFake, reply
from test_reviewer_text import review_case


DSML = ('<\uff5c\uff5cDSML\uff5c\uff5c calls>\n'
        '<\uff5c\uff5cDSML\uff5c\uff5c invoke name="read_evidence">\n'
        '<\uff5c\uff5cDSML\uff5c\uff5c parameter name="queries">[]')


def prose(text, *, complete=True, request_id="review-text"):
    return {"request_id": request_id, "protocol": "responses", "complete": complete,
            "calls": [], "items": [{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text}]}], "text": text}


def turn(client, c, s, state, *, compact=True, role="reviewer"):
    scope = {"R1": record_index(c).available["R1"]}
    return scoped_commentary_turn(client, c, {"sections": {
        "arguments:A": {"text": "Check the recorded count and status.", "refs": ["R1"]}}},
        s, scope=scope, state=state, role=role, prompt_name="command_reviewer_text.md",
        compact_context=compact)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("role", ["worker", "reviewer", "final_reviewer"])
def test_text_tool_request_is_not_commentary_and_never_executes(tmp_path, compact, role):
    c = put(case(), results={"R1": {"status": "partial", "value": 1}})
    before = c.model_dump_json()
    fake = NativeFake([prose(DSML)])
    feedback, session = turn(fake, c, store(tmp_path, c), {"queries": 2}, compact=compact, role=role)
    assert feedback is None
    assert session["outcome"]["status"] == "rejected"
    assert session["outcome"]["issues"][0]["code"] == "tool_call_in_commentary"
    assert session["queries"] == 2 and session["corrections"] == 1
    assert not session.get("read_record_ids")
    assert c.model_dump_json() == before


@pytest.mark.parametrize("compact", [False, True])
def test_recovery_error_reaches_next_request_without_fabricated_tool_receipt(tmp_path, compact):
    c = put(case(), results={"R1": {"status": "partial", "value": 1}})
    s = store(tmp_path, c)
    fake = NativeFake([prose(DSML), prose("One returned member; overall job partial.")])
    _, session = turn(fake, c, s, {"queries": 2}, compact=compact)
    feedback, session = turn(fake, c, s, json.loads(json.dumps(session)), compact=compact)
    assert session is None and feedback["status"] == "available"
    material = expand_text(fake.requests[1]["payload"])
    assert material["delivery_error"]["executed"] is False
    assert material["delivery_error"]["issues"][0]["code"] == "tool_call_in_commentary"
    assert material["response_contract"]["query_tools"] == []
    assert material["response_contract"]["delivery"] == "natural_language"
    assert fake.requests[1]["exchange"]["calls"] == []
    assert fake.requests[1]["tools"] == []


def test_actual_read_after_text_error_uses_unchanged_allowance_and_sources(tmp_path):
    c = put(case(), results={"R1": {"status": "partial", "value": 1}})
    s = store(tmp_path, c)
    fake = NativeFake([prose(DSML), reply(arguments={"records": [{"ref": "R1", "paths": ["/value"]}]}),
                       prose("The returned value is 1; the overall calculation is partial.")])
    _, state = turn(fake, c, s, {"queries": 0})
    assert state["queries"] == 0
    _, state = turn(fake, c, s, state)
    assert state["queries"] == 1 and state["read_record_ids"] == ["R1"]
    feedback, state = turn(fake, c, s, state)
    assert state is None and feedback["read_record_ids"] == ["R1"]
    assert expand_text(fake.requests[2]["payload"])["query_result"]["status"] == "read"


def test_repeated_leak_exhausts_local_corrections_not_scientific_rounds(tmp_path):
    c = put(case(), results={"R1": {"status": "failed"}})
    before = c.model_dump_json()
    s = store(tmp_path, c)
    fake = NativeFake([prose(DSML)] * 3)
    state = {"queries": 2}
    for _ in range(3):
        feedback, state = turn(fake, c, s, state)
    assert feedback["status"] == "unavailable" and state is None
    assert "text" not in feedback
    assert feedback["failed_session"]["queries"] == 2
    assert len(fake.requests) == 3 and c.model_dump_json() == before


@pytest.mark.parametrize("text", ["", " ", "The source suggests a trend."])
def test_incomplete_or_empty_text_never_becomes_review(tmp_path, text):
    c = put(case(), results={"R1": {"status": "failed"}})
    feedback, session = turn(NativeFake([prose(text, complete=False)]), c, store(tmp_path, c), {"queries": 2})
    assert feedback["status"] == "unavailable" and session is None


def test_production_comment_resume_does_not_save_unexecuted_call_as_review(tmp_path):
    fake = NativeFake([prose(DSML), prose("The available structure supports only a hypothesis.")])
    o, c, s = review_case(tmp_path, fake, "final_reviewer")
    o.config = replace(o.config, command_native_tools=True)
    pending = _comment(o, c, s)
    assert protocol(pending)["phase"] == "review"
    assert not protocol(pending).get("final_review")
    save(pending, s)
    restored = read_case_run(s.root / "command_checkpoint.json")
    completed = _comment(o, restored, s)
    assert protocol(completed)["review"]["status"] == "available"
    assert protocol(completed)["review"]["text"] != DSML
    for key in ("results", "arguments", "ranking", "revision", "rounds"):
        assert protocol(completed)[key] == protocol(c)[key]


def test_legacy_plain_text_path_also_rejects_tool_markup(tmp_path):
    o, c, s = review_case(tmp_path, ScriptClient([DSML]), "final_reviewer")
    after = _comment(o, c, s)
    assert protocol(after)["review"]["status"] == "unavailable"
    assert "text" not in protocol(after)["review"]


def test_structured_review_still_requires_native_submission_after_queries(tmp_path):
    from test_computational_review import fixture, report
    c, o, s = fixture(tmp_path)
    fake = NativeFake([reply(call_id="read1"), reply(call_id="read2"), prose(DSML),
                       reply("record_review", report().model_dump(exclude={"kind"}), call_id="done")])
    o.deps.scientific_reviewer.llm_client = fake
    original = c
    for _ in range(3):
        c = _comment(o, c, s)
    assert protocol(c)["reviewer_session"]["queries"] == 2
    assert protocol(c)["phase"] == "review"
    c = _comment(o, c, s)
    material = expand_text(fake.requests[-1]["payload"])
    assert material["delivery_error"]["executed"] is False
    assert material["response_contract"]["delivery"] == "record_review"
    assert material["response_contract"]["query_tools"] == []
    assert {t["name"] for t in fake.requests[-1]["tools"]} == {"record_review"}
    assert protocol(c)["review"]["text"] == report().text
    for key in ("results", "arguments", "ranking", "revision", "rounds"):
        assert protocol(c)[key] == protocol(original)[key]


def test_previous_read_facts_survive_a_delivery_error(tmp_path):
    c = put(case(), results={"R1": {"status": "partial", "value": 1}})
    s = store(tmp_path, c)
    fake = NativeFake([reply(arguments={"records": [{"ref": "R1", "paths": ["/value"]}]}),
                       reply(arguments={"records": [{"ref": "R1", "paths": ["/status"]}]}, call_id="second"),
                       prose(DSML), prose("One returned value; the job is partial.")])
    state = {"queries": 0}
    for _ in range(4):
        feedback, state = turn(fake, c, s, state)
    assert feedback["status"] == "available" and feedback["read_record_ids"] == ["R1"]
    material = expand_text(fake.requests[-1]["payload"])
    assert material["query_turns_remaining"] == 0
    memory = json.dumps(material["working_memory"])
    assert '/value' in memory and '/status' in memory


def test_native_envelope_cannot_turn_serialized_call_text_into_review(tmp_path):
    from test_computational_review import fixture, report
    c, o, s = fixture(tmp_path)
    value = report().model_dump(exclude={"kind"})
    fake = NativeFake([reply("record_review", {**value, "text": DSML}, call_id="bad"),
                       reply("record_review", value, call_id="corrected")])
    o.deps.scientific_reviewer.llm_client = fake
    pending = _comment(o, c, s)
    assert protocol(pending)["phase"] == "review"
    issue = protocol(pending)["reviewer_session"]["outcome"]["issues"][0]
    assert issue["path"] == ["arguments", "text"] and issue["code"] == "tool_call_in_commentary"
    assert not protocol(pending).get("final_review")
    done = _comment(o, pending, s)
    assert protocol(done)["review"]["text"] == value["text"]
    assert protocol(done)["ranking"] == protocol(c)["ranking"]


@pytest.mark.parametrize("text", [
    DSML, '<|DSML|function_calls>\n<|DSML|invoke name="read_evidence">',
    '```xml\n<tool_call>{"name":"read_evidence"}</tool_call>\n```',
    '<function_calls><invoke name="read_evidence">',
    '{"tool_calls":[{"function":{"name":"read_evidence","arguments":"{}"}}]}',
    '```json\n{"name":"read_evidence","arguments":{}}\n```',
    '{"type":"function_call","name":"read_evidence","arguments":"{}"}',
    'A preliminary check.\n' + DSML,
])
def test_recognizable_call_envelopes_are_rejected_as_data(text):
    from mechcal.agents.commentary_delivery import delivery_issue
    assert delivery_issue(text)["code"] == "tool_call_in_commentary"


@pytest.mark.parametrize("text", [
    'Use read_evidence later if more detail is needed. This source supports a partial result.',
    'No queries remain. I cannot verify the missing state values.',
    'The error contained `<tool_call>` rather than a review.',
    '> <tool_call>\nThis quoted fragment was not executed.',
    '{"reviewed_record_ids":["foreign:E1"],"command":"compute"}\nAdvice mentioning foreign:E1.',
    'The calculation returned a JSON table, not an independent experiment.',
])
def test_ordinary_scientific_prose_or_quoted_syntax_remains_prose(text):
    from mechcal.agents.commentary_delivery import delivery_issue
    assert delivery_issue(text) is None
