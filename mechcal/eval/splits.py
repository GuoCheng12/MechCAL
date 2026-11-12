"""Development/test split of PhotoMechBench (by paper cluster; see examples/make_dev_test_split.py).

Test cases are held out: the agent must not be run on them before the system is frozen. The CLI and
the batch runner refuse a test case unless explicitly allowed (``--allow-test-case`` / ``--allow-test``).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SPLIT_PATH = Path(__file__).resolve().parents[2] / "benchmarks/photomechbench/splits/raw_v2_119_dev_test.json"


@lru_cache(maxsize=1)
def load_split(path: str = str(SPLIT_PATH)) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_of(case_id: str) -> str | None:
    """'dev', 'test', or None for a case outside the benchmark (e.g. an ad hoc SMILES)."""
    split = load_split()
    if case_id in {d["case_id"] for d in split["dev"]}:
        return "dev"
    return "test" if case_id in set(split["test"]) else None


def dev_cases() -> list[str]:
    return [d["case_id"] for d in load_split()["dev"]]


HELD_OUT = ("{case} is a held-out test case. Test cases are run only after the system (tools, cards, prompts, "
            "rubric) is frozen; pass {flag} to run it deliberately.")
