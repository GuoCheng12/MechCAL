from __future__ import annotations

import json
from typing import Any

from mechcal.runtime.llm import OpenAICompatibleSettings


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

        import httpx
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            max_retries=self.settings.max_retries,
            http_client=httpx.Client(
                timeout=self.settings.timeout_seconds,
                trust_env=self.settings.trust_env,
            ),
        )
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
        response = client.chat.completions.create(
            **request_args,
            timeout=self.settings.timeout_seconds,
        )
        return (response.choices[0].message.content or "").strip()
