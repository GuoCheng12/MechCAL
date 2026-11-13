"""DeepSeek peak / off-peak windows, so large runs start only when the API is billed at half price.

Peak (from api-docs.deepseek.com/quick_start/pricing, read 2026-11-13): 01:00-04:00 and
06:00-10:00 UTC, Monday to Friday, except Chinese public holidays; that is 09:00-12:00 and
14:00-18:00 in Beijing. Every other hour is off-peak, weekends included. Public holidays are not
known to this module and are treated as weekdays (conservative: a run may wait when it need not).

A case is started only off-peak and only if the next peak window is at least ``margin_minutes``
away, so that it is likely to finish before the price doubles. Running cases are never stopped.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable

PEAK_UTC_HOURS = ((1, 4), (6, 10))        # [start, end) hours, Monday-Friday
BEIJING = timezone(timedelta(hours=8))


def is_peak(t: datetime) -> bool:
    t = t.astimezone(timezone.utc)
    return t.weekday() < 5 and any(a <= t.hour < b for a, b in PEAK_UTC_HOURS)


def _peak_starts(t: datetime):
    """Peak window starts from the day of ``t`` onward (UTC)."""
    day = t.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for d in range(8):
        base = day + timedelta(days=d)
        if base.weekday() < 5:
            for a, _ in PEAK_UTC_HOURS:
                yield base + timedelta(hours=a)


def next_peak_start(t: datetime) -> datetime:
    t = t.astimezone(timezone.utc)
    return next(s for s in _peak_starts(t) if s > t)


def can_start(t: datetime, margin_minutes: float = 45) -> bool:
    return not is_peak(t) and next_peak_start(t) - t.astimezone(timezone.utc) >= timedelta(minutes=margin_minutes)


def next_start(t: datetime, margin_minutes: float = 45) -> datetime:
    """Earliest time at or after ``t`` when a case may start (minute resolution)."""
    t = t.astimezone(timezone.utc).replace(second=0, microsecond=0)
    for _ in range(8 * 24 * 60):
        if can_start(t, margin_minutes):
            return t
        t += timedelta(minutes=1)
    raise RuntimeError("no off-peak start found within a week")


def wait_for_off_peak(margin_minutes: float = 45, *, log: Callable[[str], None] = print,
                      now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                      sleep: Callable[[float], None] = time.sleep) -> float:
    """Block until a case may start; returns the seconds waited."""
    waited = 0.0
    announced = False
    while not can_start(now(), margin_minutes):
        if not announced:
            at = next_start(now(), margin_minutes).astimezone(BEIJING)
            log(f"peak pricing: waiting until {at:%a %H:%M} Beijing time to start the next case")
            announced = True
        sleep(60)
        waited += 60
    return waited
