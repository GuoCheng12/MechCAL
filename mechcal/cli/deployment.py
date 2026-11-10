"""Portable configuration; no imports from research workspaces or saved runs."""

from contextlib import contextmanager
from dataclasses import fields
from importlib import resources
import json
import os
from pathlib import Path
import shutil
import tomllib
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mechcal.runtime import OrchestratorConfig
from mechcal.runtime.llm import OpenAICompatibleSettings
from mechcal.tools.qc_process import QCConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str
    model: str = Field(min_length=1)
    api_key_env: str = Field(default="DEEPSEEK_API_KEY", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    reasoning_effort: str | None = "high"
    api_protocol: Literal["responses", "chat_completions"] = "responses"
    timeout_seconds: float = Field(default=180, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int | None = None
    trust_env: bool = False
    normalize_json_suffix: bool = True

    @model_validator(mode="after")
    def validate_endpoint(self):
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("base_url must be an HTTP(S) endpoint")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Do not embed credentials or query parameters in base_url")
        return self

    def settings(self):
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ValueError(f"Set {self.api_key_env} before making a model request")
        return OpenAICompatibleSettings(**self.model_dump(exclude={"api_key_env"}), api_key=key)


class ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amesp_binary: str = ""
    multiwfn_binary: str = ""
    m2m_binary: str = ""
    amesp_timeout: float = Field(default=180, gt=0)
    amesp_threads: int = Field(default=1, ge=1)
    amesp_maxcore_mb: int = Field(default=1000, ge=1)


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelConfig
    runtime: dict[str, Any] = Field(default_factory=dict)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    qc: QCConfig = Field(default_factory=QCConfig)

    def orchestrator_config(self, output: Path):
        allowed = {f.name for f in fields(OrchestratorConfig)} - {"run_base_dir"}
        unknown = set(self.runtime) - allowed
        if unknown:
            raise ValueError(f"Unknown runtime options: {sorted(unknown)}")
        profile = json.loads(resources.files("mechcal").joinpath("profiles/command_v1.json").read_text())
        config = OrchestratorConfig(**{**profile, **self.runtime}, run_base_dir=output / "runs")
        if not config.command_native_tools:
            raise ValueError("This entry point requires the native command workflow")
        return config


def resolve_location(value: str, directory: Path, *, executable=False) -> str:
    if not value:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError("Unresolved environment variable in tool location")
    if executable and "/" not in expanded:
        found = shutil.which(expanded)
        if found:
            return str(Path(found).resolve())
    path = Path(expanded)
    return str((path if path.is_absolute() else directory / path).resolve())


def load_deployment(path: Path) -> Deployment:
    path = path.expanduser().resolve()
    with path.open("rb") as handle:
        config = Deployment.model_validate(tomllib.load(handle))
    for name in ("amesp_binary", "multiwfn_binary", "m2m_binary"):
        setattr(config.tools, name, resolve_location(getattr(config.tools, name), path.parent, executable=True))
    for name in ("xtb_binary", "crest_binary", "amesp_soc_binary", "dftb_binary", "pyscf_python", "pysoc_python"):
        setattr(config.qc, name, resolve_location(getattr(config.qc, name), path.parent, executable=True))
    for name in ("parameter_dir", "pysoc_source"):
        setattr(config.qc, name, resolve_location(getattr(config.qc, name), path.parent))
    config.orchestrator_config(Path("unused"))
    return config


def environment(config: Deployment, runtime: OrchestratorConfig, qc_path: Path) -> dict[str, str]:
    return {
        "MECHCAL_ENABLE_AMESP": str(runtime.enable_amesp).lower(),
        "MECHCAL_AMESP_BIN": config.tools.amesp_binary,
        "MECHCAL_MULTIWFN_BIN": config.tools.multiwfn_binary,
        "MECHCAL_AMESP_M2M_BIN": config.tools.m2m_binary,
        "MECHCAL_AMESP_TIMEOUT": str(config.tools.amesp_timeout),
        "MECHCAL_AMESP_NPARA": str(config.tools.amesp_threads),
        "MECHCAL_AMESP_MAXCORE_MB": str(config.tools.amesp_maxcore_mb),
        "MECHCAL_QC_CONFIG": str(qc_path),
    }


@contextmanager
def scoped_environment(values: dict[str, str]):
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def tool_checks(config: Deployment, runtime: OrchestratorConfig) -> list[dict]:
    checks = []
    for name, value in config.tools.model_dump().items():
        if not name.endswith("_binary"):
            continue
        required = name == "amesp_binary" and runtime.enable_amesp
        ok = bool(value and Path(value).is_file() and os.access(value, os.X_OK))
        checks.append({"tool": name, "configured": bool(value), "available": ok,
                       "error": (required or bool(value)) and not ok})
    if config.qc.enabled:
        for backend, field in {"xtb": "xtb_binary", "crest": "crest_binary", "dftb": "dftb_binary",
                               "pyscf_tda": "pyscf_python", "pysoc": "pysoc_python",
                               "amesp_soc": "amesp_soc_binary"}.items():
            configured = bool(getattr(config.qc, field))
            reason = config.qc.backend_reason(backend)
            disabled = backend == "amesp_soc" and not config.qc.allow_amesp_soc
            checks.append({"tool": backend, "configured": configured, "available": reason is None,
                           "reason": reason, "error": configured and not disabled and reason is not None})
    if bool(config.tools.multiwfn_binary) != bool(config.tools.m2m_binary):
        checks.append({"tool": "hole_electron", "error": True,
                       "reason": "Configure both Multiwfn and m2m, or neither"})
    return checks
