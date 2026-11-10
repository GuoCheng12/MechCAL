import json
from pathlib import Path

import pytest

from mechcal.eval.tool_sources import (
    MAX_JUDGE_INPUT_BYTES, MAX_SOURCE_BYTES, build_tool_source_packet,
    check_judge_input_size, json_bytes,
)
from mechcal.eval.zero_shot_v3 import EvidenceSubmission, ZeroShotV3Judge


def test_complete_request_boundary_includes_schema_prompt_and_feedback():
    assert MAX_JUDGE_INPUT_BYTES is None and MAX_SOURCE_BYTES is None
    payload = {"evidence": ""}
    schema = {"type": "json_schema"}
    base = check_judge_input_size("prompt", payload, schema)
    payload["evidence"] = "x" * (200_000 - base)
    assert check_judge_input_size("prompt", payload, schema) == 200_000
    assert check_judge_input_size("prompt!", payload, schema) > 200_000
    assert check_judge_input_size("prompt", payload, schema, "correction") > 200_000
    with pytest.raises(ValueError, match="not sent"):
        check_judge_input_size("prompt!", payload, schema, max_bytes=200_000)
    with pytest.raises(ValueError, match="not sent"):
        check_judge_input_size("prompt", payload, schema, "correction", max_bytes=200_000)
