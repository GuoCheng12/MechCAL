"""Transport normalization must never invent missing content or ignore another answer."""

import json

import pytest

from mechcal.runtime.json_transport import normalize_redundant_suffix

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
          "properties": {"answer": {"type": "string"}}}


@pytest.mark.parametrize("answer", ["ordinary", "braces } ] { inside", 'escaped \\\" quotes',
                                    "\u673a\u5236", "line\nline", ""])
@pytest.mark.parametrize("suffix", ["}", "}}}", "\n } ] \t }\n"])
def test_only_complete_schema_valid_object_can_be_normalized(answer, suffix):
    original = " \n" + json.dumps({"answer": answer})
    text, note = normalize_redundant_suffix(original + suffix, SCHEMA)
    assert text == original
    assert json.loads(text) == {"answer": answer}
    assert note == {"kind": "redundant_json_closing_suffix", "start_offset": len(original),
                    "removed_suffix": suffix, "schema_validated": True, "values_modified": False}


@pytest.mark.parametrize("text", [
    '{"answer":"ok"}', '{"answer":"ok"} \n',
    '{"answer":"ok"', '{"answer":', '{"answer":"unclosed',
    '{"answer":"ok" "other":2}', '{"answer":null}}', '{} }}',
    '{"answer":"ok","unknown":3}}', '{"answer":"ok"} {"answer":"different"}',
    '{"answer":"ok"} prose', '{"answer":"ok"} //comment',
    '```json\n{"answer":"ok"}\n```', '<tool>{"answer":"ok"}</tool>',
    '[{"answer":"ok"}]]', '{"answer":"ok"})',
])
def test_no_recovery_for_missing_content_invalid_schema_or_extra_messages(text):
    assert normalize_redundant_suffix(text, SCHEMA) == (text, None)
