from __future__ import annotations

import json
from typing import Any

from mechcal.runtime.llm import (
    OpenAICompatibleSettings,
    _add_generation_args,
    create_completion,
    response_text,
    validate_response,
)


class OpenAITextClient:
    def __init__(self, settings: OpenAICompatibleSettings | None = None) -> None:
        self.settings = settings or OpenAICompatibleSettings.from_env()

    def is_configured(self) -> bool:
        return not self.settings.missing_fields()

    def complete_text(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
    ) -> str:
        if not self.is_configured():
            missing = ", ".join(self.settings.missing_fields())
            raise RuntimeError(f"OpenAI-compatible settings are incomplete: {missing}")

        request_args: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
        }
        if self.settings.reasoning_effort is not None:
            request_args["reasoning_effort"] = self.settings.reasoning_effort
        _add_generation_args(request_args, self.settings)
        response = create_completion(
            self.settings, request_args, sdk_retries=self.settings.max_retries,
        )
        validate_response(response)
        return response_text(response).strip()
