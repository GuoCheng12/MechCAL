from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StdioMcpServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0

    @classmethod
    def current_python_server(cls) -> StdioMcpServerConfig:
        return cls(command=sys.executable, args=["-m", "mechcal.mcp.server"])


class StdioMcpTransport:
    """Synchronous stdio MCP transport for the MechCAL orchestrator.

    The implementation opens a short-lived MCP session per tool call. That keeps
    the first production path simple and deterministic; a persistent async client
    can replace this class later without touching the orchestrator contract.
    """

    def __init__(self, config: StdioMcpServerConfig | None = None) -> None:
        self.config = config or StdioMcpServerConfig.current_python_server()

    def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        del server_name
        return asyncio.run(self._call_tool(tool_name, arguments))

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self._merged_env(self.config.env),
        )
        async with stdio_client(params) as (stdio, write):
            async with ClientSession(stdio, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=self.config.timeout_seconds)
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=self.config.timeout_seconds,
                )
        if getattr(result, "isError", False):
            raise RuntimeError(self._result_text(result) or f"MCP tool {tool_name} failed.")
        return self._result_payload(result)

    def _merged_env(self, overrides: Mapping[str, str]) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("FASTMCP_LOG_LEVEL", "CRITICAL")
        env.setdefault("PYTHONWARNINGS", "ignore")
        env.update({key: value for key, value in overrides.items() if value is not None})
        return env

    def _result_payload(self, result: Any) -> dict[str, Any]:
        structured = getattr(result, "structuredContent", None) or getattr(
            result,
            "structured_content",
            None,
        )
        if isinstance(structured, dict):
            return structured

        text = self._result_text(result)
        if text:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("MCP result did not contain a JSON object payload.")

    def _result_text(self, result: Any) -> str:
        texts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                texts.append(text)
        return "\n".join(texts)
