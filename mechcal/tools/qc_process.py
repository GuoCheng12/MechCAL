"""Isolated subprocess execution, whole-route deadlines and measured resources."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from pydantic import BaseModel, ConfigDict, Field


class QCError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class QCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    xtb_binary: str = ""
    dftb_binary: str = ""
    pyscf_python: str = ""
    pysoc_python: str = ""
    pysoc_source: str = ""
    parameter_dir: str = ""
    parameter_name: str = "mio-1-1"
    parameter_revision: str = ""
    threads: int = Field(default=4, ge=1, le=8)
    allow_excited_scan: bool = False

    @classmethod
    def from_environment(cls):
        path = os.environ.get("MECHCAL_QC_CONFIG")
        return cls.model_validate_json(Path(path).read_text()) if path else cls()

    def backend_reason(self, backend):
        binary = {"xtb": self.xtb_binary, "dftb": self.dftb_binary,
                  "pyscf": self.pyscf_python, "pysoc": self.pysoc_python}.get(backend)
        if not self.enabled:
            return "External QC trial is disabled."
        if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
            return f"{backend} executable is unavailable."
        if backend == "dftb" and not Path(self.parameter_dir).is_dir():
            return "DFTB Slater-Koster parameter directory is unavailable."
        if backend == "pysoc":
            from mechcal.tools.pysoc_source import source_reason
            return self.backend_reason("dftb") or source_reason(self.pysoc_source)
        return None


class QCJob:
    def __init__(self, workdir: Path, config: QCConfig, seconds: int):
        self.workdir, self.config = workdir, config
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.records = []
        workdir.mkdir(parents=True, exist_ok=False)

    def execute(self, argv: list[str], directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise QCError("timeout", "Whole-route wall time exhausted.")
        env = dict(os.environ, OMP_NUM_THREADS=str(self.config.threads),
                   MKL_NUM_THREADS=str(self.config.threads), OPENBLAS_NUM_THREADS="1",
                   OMP_STACKSIZE="256M")
        log, resources = directory / "process.log", directory / "resources.json"
        command = ["/usr/bin/time", "-f", '{"peak_rss_kb":%M,"user_seconds":%U,"system_seconds":%S}',
                   "-o", str(resources), *argv]
        start = time.monotonic()
        record = {"argv": argv, "directory": str(directory), "threads": self.config.threads}
        try:
            with log.open("w") as output:
                process = subprocess.Popen(command, cwd=directory, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=remaining)
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    raise
            record["returncode"] = code
            if resources.exists():
                for line in resources.read_text().splitlines():
                    if line.startswith("{"):
                        record.update(json.loads(line))
            if code:
                raise QCError("process_failed", f"Exit {code}; inspect {log}.")
            return log.read_text()
        except subprocess.TimeoutExpired as exc:
            record["timed_out"] = True
            raise QCError("timeout", f"Whole-route wall time exceeded; process group killed. See {log}.") from exc
        except OSError as exc:
            raise QCError("process_failed", str(exc)) from exc
        finally:
            record["wall_seconds"] = time.monotonic() - start
            record["log"] = str(log)
            self.records.append(record)
            (self.workdir / "process_records.json").write_text(json.dumps(self.records, indent=2))

    def timing(self):
        return {"total_wall_seconds": time.monotonic() - self.started,
                "compute_wall_seconds": sum(r["wall_seconds"] for r in self.records),
                "peak_process_rss_kb": max((r.get("peak_rss_kb", 0) for r in self.records), default=0),
                "cpu_threads": self.config.threads, "gpu_used": False,
                "process_count": len(self.records), "cache_hit": False}
