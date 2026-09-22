from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from mechcal.eval.prompts import load_eval_prompt
from mechcal.eval.schemas import SubjectRawOutput
from mechcal.mechanism_pool import MECHANISM_POOL
from mechcal.public_ids import normalize_public_case_id
from mechcal.runtime.llm import (
    OpenAICompatibleSettings,
    OpenAIJsonClient,
    _parse_json_object,
)

MECHANISM_STRUCTURE_LLM_PROMPT = "mechanism_structure_llm.md"
MECHANISM_STRUCTURE_LLM_PROMPT_VERSION = "2026-06-06"
MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT = (
    "mechanism_structure_llm_evidence_formatted.md"
)
MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT_VERSION = "2026-07-25"
MECHANISM_STRUCTURE_LLM_CHEM_R_TAGGED_PROMPT = (
    "mechanism_structure_llm_chem_r_tagged.md"
)
MECHANISM_STRUCTURE_LLM_CHEM_R_TAGGED_PROMPT_VERSION = "2026-07-26-v2"
MECHANISM_BASELINE_TRANSCRIPT_EXTRACTOR_PROMPT = (
    "mechanism_baseline_transcript_extractor.md"
)
MECHANISM_BASELINE_TRANSCRIPT_EXTRACTOR_PROMPT_VERSION = "2026-06-16"
CODEX_STRUCTURE_PRIOR_PROMPT_VERSION = "2026-07-10"


class CodexResponsesJsonClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str = "gpt-5.5",
        api_key: str | None = None,
        auth_path: Path | None = None,
        reasoning_effort: str = "high",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key or (
            _api_key_from_codex_auth(auth_path) if auth_path is not None else ""
        )
        if not self.api_key:
            raise ValueError("Pass an API key or an explicit authentication file.")

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_name: str = "mechanism_predictions",
    ) -> dict[str, Any]:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            default_headers={"User-Agent": "curl/8.0"},
            http_client=httpx.Client(timeout=self.timeout_seconds, trust_env=False),
        )
        response = client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            reasoning={"effort": self.reasoning_effort},
            text={"format": _mechanism_prediction_json_schema(schema_name)},
            max_output_tokens=2500,
            store=False,
            timeout=self.timeout_seconds,
        )
        content = getattr(response, "output_text", "") or "{}"
        return _parse_json_object(content)


class CodexStructurePriorMechanismBaseline:
    subject_id = "codex_structure_prior_llm"
    subject_kind = "codex_structure_prior_llm"
    prompt_version = CODEX_STRUCTURE_PRIOR_PROMPT_VERSION

    def __init__(
        self,
        *,
        client: CodexResponsesJsonClient,
        prompt_name: str = MECHANISM_STRUCTURE_LLM_EVIDENCE_FORMATTED_PROMPT,
    ) -> None:
        self.client = client
        self.prompt_name = prompt_name

    @property
    def model(self) -> str:
        return self.client.model

    def run(self, case: dict[str, object]) -> dict[str, object]:
        payload = build_structure_llm_mechanism_payload(case)
        raw_json = self.client.complete_json(
            system_prompt=load_eval_prompt(self.prompt_name),
            payload=payload,
        )
        return {
            "system_prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "public_payload": payload,
            "raw_json": raw_json,
            "model": self.model,
            "runtime": "openai_responses",
            "reasoning_effort": self.client.reasoning_effort,
        }


class StructureLLMMechanismBaseline:
    subject_id = "structure_llm"
    subject_kind = "structure_llm"
    prompt_version = MECHANISM_STRUCTURE_LLM_PROMPT_VERSION

    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
        subject_id: str | None = None,
        subject_kind: str | None = None,
        prompt_name: str = MECHANISM_STRUCTURE_LLM_PROMPT,
        prompt_version: str | None = None,
        strict_json_schema: bool = False,
        native_think_answer: bool = False,
        native_thinking_json: bool = False,
        intern_api_thinking: bool = False,
    ) -> None:
        self.client = client or OpenAIJsonClient(settings)
        self.subject_id = subject_id or self.subject_id
        self.subject_kind = subject_kind or self.subject_kind
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version or self.prompt_version
        self.strict_json_schema = strict_json_schema
        self.native_think_answer = native_think_answer
        self.native_thinking_json = native_thinking_json
        self.intern_api_thinking = intern_api_thinking

    @property
    def model(self) -> str:
        return self.client.settings.model

    def run(self, case: dict[str, object]) -> dict[str, object]:
        payload = build_structure_llm_mechanism_payload(case)
        complete_args: dict[str, object] = {
            "system_prompt": load_eval_prompt(self.prompt_name),
            "payload": payload,
        }
        if self.native_think_answer:
            payload["output_constraints"] = {
                "reasoning_format": "<think>...</think>",
                "answer_format": "<answer>{JSON}</answer>",
                "mechanism_prediction_count": 3,
                "distinct_mechanism_labels": True,
                "evidence_items_per_prediction": 1,
                "limitation_items_per_prediction": 1,
                "evidence_in_final_answer": True,
            }
            result = self.client.complete_json_result(
                **complete_args,
                omit_response_format=True,
                extra_body={
                    "guided_regex": _mechanism_prediction_tagged_regex()
                },
                answer_tag="answer",
                preserve_invalid_response=True,
            )
            return {
                "system_prompt_name": self.prompt_name,
                "prompt_version": self.prompt_version,
                "public_payload": payload,
                "raw_json": result.data,
                "raw_completion": result.raw_text,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "parse_error": result.parse_error,
                "model": self.model,
            }
        if self.native_thinking_json:
            payload["output_constraints"] = {
                "reasoning_mode": "native_model_thinking",
                "final_answer_format": "JSON object only after reasoning",
                "mechanism_prediction_count": 3,
                "distinct_mechanism_labels": True,
                "evidence_items_per_prediction": 1,
                "limitation_items_per_prediction": 1,
                "evidence_in_final_answer": True,
                "evidence_field_example": [
                    "Finding: ... Warrant: ... Boundary: ..."
                ],
                "limitations_field_example": ["One concise limitation."],
            }
            thinking_extra_body = (
                {"thinking_mode": True}
                if self.intern_api_thinking
                else {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "top_k": 50,
                }
            )
            result = self.client.complete_json_result(
                **complete_args,
                omit_response_format=True,
                extra_body=thinking_extra_body,
                parse_after_tag="think",
                preserve_invalid_response=True,
            )
            normalized_json = _normalize_native_thinking_prediction_json(result.data)
            return {
                "system_prompt_name": self.prompt_name,
                "prompt_version": self.prompt_version,
                "public_payload": payload,
                "raw_json": normalized_json,
                "native_final_json": result.data,
                "raw_completion": result.raw_text,
                "raw_reasoning": result.reasoning_text,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "parse_error": result.parse_error,
                "response_metadata": result.response_metadata,
                "model": self.model,
                "format_adapter": (
                    "intern_api_thinking_json_v1"
                    if self.intern_api_thinking
                    else "native_thinking_json_v1"
                ),
            }
        if self.strict_json_schema:
            payload["output_constraints"] = {
                "mechanism_prediction_count": 3,
                "evidence_items_per_prediction": 1,
                "limitation_items_per_prediction": 1,
                "concise_output": True,
                "include_final_summary": False,
            }
            complete_args["omit_response_format"] = True
            if case.get("_bounded_output"):
                complete_args["extra_body"] = {
                    "guided_grammar": _mechanism_prediction_compact_grammar()
                }
            else:
                complete_args["extra_body"] = {
                    "guided_json": _mechanism_prediction_exact_json_schema()
                }
        raw_json = self.client.complete_json(**complete_args)
        return {
            "system_prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "public_payload": payload,
            "raw_json": raw_json,
            "model": self.model,
        }


def build_structure_llm_mechanism_payload(case: dict[str, object]) -> dict[str, object]:
    return {
        "case_id": normalize_public_case_id(case.get("public_case_id")),
        "smiles": str(case["smiles"]),
        "task": str(case.get("task") or "Predict and rank likely AIE mechanisms."),
        "mechanism_pool": list(MECHANISM_POOL),
        "baseline_constraints": {
            "tools": "none",
            "web_search": "disabled",
            "private_answer_key_access": "none",
            "source_lookup": "none",
            "evidence_basis": "SMILES-only structure proxy",
        },
    }


def prediction_payload_from_structure_llm_mechanism_json(
    raw_json: dict[str, object],
) -> list[dict[str, object]]:
    predictions = raw_json.get("mechanism_predictions") or []
    if not isinstance(predictions, list):
        raise ValueError("structure_llm response must contain mechanism_predictions list.")
    payload: list[dict[str, object]] = []
    for index, item in enumerate(predictions, start=1):
        if not isinstance(item, dict):
            continue
        payload.append(
            {
                "prediction_id": f"P{index:03d}",
                "rank": _rank(item.get("rank"), default=index),
                "label": str(item.get("label") or ""),
                "confidence": _confidence(item.get("confidence")),
                "claim_status": _claim_status(item.get("claim_status")),
                "support_strength": _support_strength(item.get("support_strength")),
                "evidence_refs": [],
                "evidence": _string_list(item.get("evidence")),
                "limitations": _string_list(item.get("limitations")),
            }
        )
    return payload


def _normalize_native_thinking_prediction_json(
    raw_json: dict[str, Any],
) -> dict[str, Any]:
    predictions = raw_json.get("mechanism_predictions")
    if not isinstance(predictions, list):
        return raw_json

    normalized = dict(raw_json)
    normalized_predictions: list[object] = []
    for item in predictions:
        if not isinstance(item, dict):
            normalized_predictions.append(item)
            continue
        row = dict(item)
        row["evidence"] = _normalize_native_evidence(row.get("evidence"))
        limitations = row.get("limitations")
        if limitations is None:
            limitations = row.get("limitation")
        row["limitations"] = _string_list(limitations)
        normalized_predictions.append(row)
    normalized["mechanism_predictions"] = normalized_predictions
    return normalized


def _normalize_native_evidence(value: object) -> list[str]:
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = _normalize_native_evidence_record(item)
                if text:
                    normalized.append(text)
            else:
                text = str(item).strip()
                if text:
                    normalized.append(text)
        return normalized
    if not isinstance(value, dict):
        return _string_list(value)
    text = _normalize_native_evidence_record(value)
    return [text] if text else []


def _normalize_native_evidence_record(value: dict[object, object]) -> str:
    fields = {str(key).lower(): item for key, item in value.items()}
    parts = []
    for field, heading in (
        ("finding", "Finding"),
        ("warrant", "Warrant"),
        ("boundary", "Boundary"),
    ):
        text = str(fields.get(field) or "").strip()
        if text:
            parts.append(f"{heading}: {text}")
    return " ".join(parts)


class MechanismBaselineTranscriptExtractor:
    extractor_id = "mechanism_baseline_transcript_extractor_v1"
    prompt_version = MECHANISM_BASELINE_TRANSCRIPT_EXTRACTOR_PROMPT_VERSION

    def __init__(
        self,
        *,
        client: OpenAIJsonClient | None = None,
        settings: OpenAICompatibleSettings | None = None,
    ) -> None:
        self.client = client or OpenAIJsonClient(settings)

    @property
    def model(self) -> str:
        return self.client.settings.model

    def extract(
        self,
        raw_output: SubjectRawOutput,
        *,
        public_case_id: str | None = None,
        max_predictions: int = 5,
    ) -> dict[str, object]:
        raw_json = raw_output.raw_json or {}
        transcript = raw_json.get("transcript") if isinstance(raw_json, dict) else None
        final_answer = raw_json.get("final_answer") if isinstance(raw_json, dict) else None
        response = self.client.complete_json(
            system_prompt=load_eval_prompt(
                MECHANISM_BASELINE_TRANSCRIPT_EXTRACTOR_PROMPT
            ),
            payload={
                "case_id": normalize_public_case_id(
                    public_case_id or raw_output.metadata.get("public_case_id")
                ),
                "subject_id": raw_output.subject_id,
                "subject_kind": raw_output.subject_kind,
                "mechanism_pool": list(MECHANISM_POOL),
                "max_predictions": max_predictions,
                "raw_packet": {
                    "raw_text": raw_output.raw_text,
                    "transcript": transcript if isinstance(transcript, list) else [],
                    "final_answer": final_answer if isinstance(final_answer, str) else "",
                },
                "extraction_constraints": {
                    "private_answer_key_access": "none",
                    "hidden_reference_access": "none",
                    "source_lookup": "none",
                    "mode": "extract_only_from_raw_packet",
                },
            },
        )
        response["extractor_id"] = self.extractor_id
        response["extractor_prompt_version"] = self.prompt_version
        response["extractor_model"] = self.model
        return response


def prediction_payload_from_mechanism_extractor_json(
    raw_json: dict[str, object],
) -> list[dict[str, object]]:
    predictions = raw_json.get("mechanism_predictions") or []
    if not isinstance(predictions, list):
        raise ValueError("extractor response must contain mechanism_predictions list.")
    payload: list[dict[str, object]] = []
    for index, item in enumerate(predictions, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "")
        if label not in MECHANISM_POOL:
            continue
        evidence = _string_list(item.get("evidence"))
        source_snippets = _string_list(item.get("source_snippets"))
        if not evidence and source_snippets:
            evidence = source_snippets
        payload.append(
            {
                "prediction_id": f"P{index:03d}",
                "rank": _rank(item.get("rank"), default=index),
                "label": label,
                "confidence": _confidence(
                    item.get("differential_priority")
                    if item.get("differential_priority") is not None
                    else item.get("confidence")
                ),
                "differential_priority": _confidence(
                    item.get("differential_priority")
                    if item.get("differential_priority") is not None
                    else item.get("confidence")
                ),
                "claim_status": str(item.get("claim_status") or "candidate"),
                "support_strength": str(item.get("support_strength") or ""),
                "evidence_refs": [],
                "evidence": evidence,
                "limitations": _string_list(item.get("limitations")),
                "source_snippets": source_snippets,
            }
        )
    return payload


def _rank(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, float):
        return max(1, int(value))
    return default


def _confidence(value: object) -> float:
    if isinstance(value, int | float):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _claim_status(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {
        "candidate_requires_validation",
        "partially_supported_candidate",
        "supported",
        "supported_claim",
        "likely",
    }:
        return text
    return "likely"


def _support_strength(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"strong", "partial", "weak_or_proxy", "unsupported"}:
        return text
    return ""


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _api_key_from_codex_auth(path: Path) -> str:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    api_key = str(payload.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY missing from {path}")
    return api_key


def _mechanism_prediction_json_schema(
    name: str,
    *,
    min_predictions: int = 3,
    max_predictions: int = 5,
) -> dict[str, object]:
    mechanism_labels = list(MECHANISM_POOL)
    return {
        "type": "json_schema",
        "name": name,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "final_summary": {"type": "string"},
                "mechanism_predictions": {
                    "type": "array",
                    "minItems": min_predictions,
                    "maxItems": max_predictions,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string", "enum": mechanism_labels},
                            "rank": {"type": "integer", "minimum": 1, "maximum": 5},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "claim_status": {
                                "type": "string",
                                "enum": [
                                    "candidate_requires_validation",
                                    "partially_supported_candidate",
                                    "supported",
                                    "proxy_supported",
                                    "underdetermined_candidate",
                                    "weakened",
                                ],
                            },
                            "support_strength": {
                                "type": "string",
                                "enum": [
                                    "strong",
                                    "partial",
                                    "weak_or_proxy",
                                    "unsupported",
                                ],
                            },
                            "evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "limitations": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "label",
                            "rank",
                            "confidence",
                            "claim_status",
                            "support_strength",
                            "evidence",
                            "limitations",
                        ],
                    },
                },
            },
            "required": ["final_summary", "mechanism_predictions"],
        },
    }


def _mechanism_prediction_compact_grammar() -> str:
    labels = " | ".join(f'"\\"{label}\\""' for label in MECHANISM_POOL)
    root = (
        'root ::= "{" ws "\\"mechanism_predictions\\"" ws ":" ws "[" ws '
        'prediction1 ws "," ws prediction2 ws "," ws prediction3 ws "]" ws "}"'
    )

    def prediction(rank: int) -> str:
        return (
            f'prediction{rank} ::= "{{" ws "\\"label\\"" ws ":" ws label ws "," ws '
            f'"\\"rank\\"" ws ":" ws "{rank}" ws "," ws fields ws "}}"'
        )

    fields = (
        'fields ::= "\\"confidence\\"" ws ":" ws number ws "," ws '
        '"\\"claim_status\\"" ws ":" ws "\\"candidate_requires_validation\\"" ws '
        '"," ws "\\"support_strength\\"" ws ":" ws "\\"weak_or_proxy\\"" ws "," '
        'ws "\\"evidence\\"" ws ":" ws "[" ws evidence_string ws "]" ws "," ws '
        '"\\"limitations\\"" ws ":" ws "[" ws limitation_string ws "]"'
    )
    number = (
        'number ::= "0" fraction exponent | [1-9] [0-9]* fraction exponent | '
        '"-" [0-9] fraction exponent | "-" [1-9] [0-9]* fraction exponent'
    )
    rules = [
        root,
        *(prediction(rank) for rank in (1, 2, 3)),
        fields,
        f"label ::= {labels}",
        'evidence_string ::= "\\"" character{1,400} "\\""',
        'limitation_string ::= "\\"" character{1,160} "\\""',
        r'character ::= [^"\\\0-\x1f] | "\\" escape',
        (
            r'escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] '
            r"[A-Fa-f0-9] [A-Fa-f0-9]"
        ),
        number,
        'fraction ::= "" | "." [0-9] [0-9]*',
        'exponent ::= "" | "e" sign [0-9] [0-9]* | "E" sign [0-9] [0-9]*',
        'sign ::= "" | "+" | "-"',
        r"ws ::= [ \n\t]{0,4}",
    ]
    return "\n".join(rules)


def _mechanism_prediction_tagged_regex() -> str:
    confidence = r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)"
    text = r'[^"\\\r\n]+'
    evidence = (
        rf'"Finding: {text} Warrant: {text} Boundary: {text}"'
    )
    limitation = rf'"{text}"'

    def label_choice(labels: list[str]) -> str:
        return "(?:" + "|".join(re.escape(value) for value in labels) + ")"

    def prediction(rank: int, labels: list[str]) -> str:
        return (
            r'\{"label":"' + label_choice(labels) + rf'","rank":{rank},'
            rf'"confidence":{confidence},'
            r'"claim_status":"candidate_requires_validation",'
            r'"support_strength":"weak_or_proxy",'
            rf'"evidence":\[{evidence}\],'
            rf'"limitations":\[{limitation}\]\}}'
        )

    rank_one_choices: list[str] = []
    for first in MECHANISM_POOL:
        rank_two_choices: list[str] = []
        for second in MECHANISM_POOL:
            if second == first:
                continue
            third_choices = [
                label for label in MECHANISM_POOL if label not in {first, second}
            ]
            rank_two_choices.append(
                prediction(2, [second])
                + ","
                + prediction(3, third_choices)
            )
        rank_one_choices.append(
            prediction(1, [first])
            + ",(?:"
            + "|".join(rank_two_choices)
            + ")"
        )
    predictions = "(?:" + "|".join(rank_one_choices) + ")"
    return (
        r"<think>[^<]+</think>\s*<answer>"
        r'\{"mechanism_predictions":\['
        + predictions
        + r"\]\}</answer>"
    )


def _mechanism_prediction_exact_json_schema() -> dict[str, object]:
    def prediction(rank: int) -> dict[str, object]:
        one_string = {
            "type": "array",
            "prefixItems": [{"type": "string"}],
            "items": False,
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "label": {"type": "string", "enum": list(MECHANISM_POOL)},
                "rank": {"type": "integer", "enum": [rank]},
                "confidence": {"type": "number"},
                "claim_status": {
                    "type": "string",
                    "enum": ["candidate_requires_validation"],
                },
                "support_strength": {
                    "type": "string",
                    "enum": ["weak_or_proxy"],
                },
                "evidence": one_string,
                "limitations": one_string,
            },
            "required": [
                "label",
                "rank",
                "confidence",
                "claim_status",
                "support_strength",
                "evidence",
                "limitations",
            ],
        }

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mechanism_predictions": {
                "type": "array",
                "prefixItems": [prediction(rank) for rank in (1, 2, 3)],
                "items": False,
            }
        },
        "required": ["mechanism_predictions"],
    }
