"""Off-peak gating of large runs (DeepSeek peak: 01-04 and 06-10 UTC, Monday-Friday)."""

from datetime import datetime, timezone

from mechcal.agent import schedule as sc

UTC = timezone.utc
FRI = lambda h, m=0: datetime(2026, 11, 13, h, m, tzinfo=UTC)          # Friday 2026-11-13


def test_peak_windows_and_weekends():
    assert sc.is_peak(FRI(2)) and sc.is_peak(FRI(9, 59)) and sc.is_peak(FRI(1))
    assert not sc.is_peak(FRI(4)) and not sc.is_peak(FRI(5, 30)) and not sc.is_peak(FRI(10)) and not sc.is_peak(FRI(23))
    assert not sc.is_peak(datetime(2026, 11, 14, 2, tzinfo=UTC))             # Saturday
    beijing = datetime(2026, 11, 13, 15, tzinfo=sc.BEIJING)                    # 15:00 Beijing = 07:00 UTC
    assert sc.is_peak(beijing)


def test_start_needs_a_margin_before_the_next_peak():
    assert not sc.can_start(FRI(0, 30), 45)                 # peak at 01:00 in 30 min
    assert sc.can_start(FRI(0, 10), 45)
    assert sc.next_start(FRI(0, 30), 45) == FRI(4)           # after the 01:00-04:00 window
    assert sc.next_start(FRI(4), 45) == FRI(4)               # 04:00 start, next peak 06:00 (120 min away)
    assert sc.next_start(FRI(5, 30), 45) == FRI(10)
    assert sc.next_peak_start(FRI(23)) == datetime(2026, 11, 16, 1, tzinfo=UTC)   # Monday after the weekend


def test_wait_blocks_until_an_off_peak_start():
    clock = {"t": FRI(9)}
    lines = []

    def sleep(seconds):
        clock["t"] = clock["t"].replace(minute=clock["t"].minute) + __import__("datetime").timedelta(seconds=seconds)

    waited = sc.wait_for_off_peak(45, log=lines.append, now=lambda: clock["t"], sleep=sleep)
    assert clock["t"] >= FRI(10) and waited == 3600 and len(lines) == 1 and "18:00 Beijing" in lines[0]
    assert sc.wait_for_off_peak(45, log=lines.append, now=lambda: FRI(12), sleep=sleep) == 0


def test_judge_usage_goes_to_its_own_telemetry_file(tmp_path, monkeypatch):
    import os
    from mechcal.eval.llm_settings import record_judge_usage
    from mechcal.runtime.llm import TELEMETRY_ENV
    monkeypatch.delenv(TELEMETRY_ENV, raising=False)
    record_judge_usage(tmp_path / "judge_telemetry.jsonl")
    assert os.environ[TELEMETRY_ENV] == str(tmp_path / "judge_telemetry.jsonl")
    monkeypatch.delenv(TELEMETRY_ENV)


def test_token_usage_prices_peak_and_off_peak(tmp_path):
    import importlib.util, json
    spec = importlib.util.spec_from_file_location("tu", __import__("pathlib").Path(__file__).parents[1] / "examples/token_usage.py")
    tu = importlib.util.module_from_spec(spec); spec.loader.exec_module(tu)
    row = lambda ts: json.dumps({"timestamp_utc": ts, "usage": {"prompt_tokens": 1_000_000, "cached_tokens": 0,
                                                                "completion_tokens": 0, "total_tokens": 1_000_000}})
    (tmp_path / "t.jsonl").write_text(row("2026-11-13T02:00:00Z") + "\n" + row("2026-11-13T12:00:00Z") + "\n")
    c = tu.tally([tmp_path / "t.jsonl"])
    assert c["calls"] == 2 and abs(c["usd"] - (0.30 + 0.15)) < 1e-9 and c["peak_tokens"] == 1_000_000
