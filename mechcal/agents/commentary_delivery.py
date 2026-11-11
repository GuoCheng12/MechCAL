"""Recognize transport syntax accidentally emitted as prose; never execute it."""

import json
import re


# Match call envelopes at a line boundary, not ordinary mentions or quoted
# examples. Provider-native calls remain the only executable representation.
_CALL_MARKUP = re.compile(
    r"(?m)^\s*(?:```(?:xml|json)?\s*)?<(?:"
    r"(?:\|+|\uff5c+)DSML(?:\|+|\uff5c+)\s*(?:calls|function_calls|invoke)\b"
    r"|(?:tool_call|tool_calls|function_call|function_calls|invoke)\b)"
)


def delivery_issue(text, *, can_read=False):
    """Reject recognizable call serialization, without judging scientific prose."""
    serialized_call = bool(_CALL_MARKUP.search(text))
    body = text.strip()
    if body.startswith("```") and body.endswith("```"):
        body = body.partition("\n")[2].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(body)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        serialized_call |= ("tool_calls" in value or "function_call" in value
                            or ("name" in value and "arguments" in value)
                            or value.get("type") == "function_call")
    if not serialized_call:
        return None
    remedy = ("Use an offered native read tool if needed, or return the scientific commentary."
              if can_read else
              "No query tools remain. Return commentary from the supplied facts and state any unverified limits.")
    return {"path": ["text"], "code": "tool_call_in_commentary", "record_id": None,
            "message": "The text contains a serialized tool request, not completed commentary. "
                       "No tool was executed and no commentary was saved.",
            "remedy": remedy + " Do not write tool-call markup in the answer."}
