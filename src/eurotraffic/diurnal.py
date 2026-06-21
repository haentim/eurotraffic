"""Canonical diurnal (time-of-day) traffic profile.

Most cities in the dataset only provide a daily/annual aggregate (AADT) with no
hour-of-day breakdown. To feed a uniform time-of-day slider we distribute each
sensor's measured daily volume across 24 hours using a single canonical weekday
curve with the typical twin AM/PM commuter peaks and an overnight trough.

The weights sum to 1.0, so ``estimate_hourly(daily_total)`` conserves the daily
volume: ``sum(estimate_hourly(v)) == v``. Values are an estimate, clearly labeled
as such in the dashboard.
"""

from __future__ import annotations

# Relative share of a day's traffic in each hour 0..23. Shape based on typical
# urban weekday counting profiles (low overnight, ~8h morning peak, midday
# plateau, ~17h evening peak). Normalized below so it always sums to 1.0.
_RAW_WEIGHTS = [
    0.6,  # 00
    0.4,  # 01
    0.3,  # 02
    0.3,  # 03
    0.5,  # 04
    1.2,  # 05
    3.0,  # 06
    6.0,  # 07
    7.8,  # 08  morning peak
    6.2,  # 09
    4.8,  # 10
    4.6,  # 11
    4.9,  # 12
    4.9,  # 13
    5.0,  # 14
    5.8,  # 15
    7.0,  # 16
    8.2,  # 17  evening peak
    7.4,  # 18
    5.2,  # 19
    3.6,  # 20
    2.6,  # 21
    1.8,  # 22
    1.1,  # 23
]

_TOTAL = sum(_RAW_WEIGHTS)
DIURNAL_WEIGHTS: list[float] = [w / _TOTAL for w in _RAW_WEIGHTS]
HOURS: list[int] = list(range(24))


def estimate_hourly(daily_total: float) -> list[float]:
    """Spread a daily traffic total across 24 hours via the canonical curve."""
    return [daily_total * w for w in DIURNAL_WEIGHTS]
