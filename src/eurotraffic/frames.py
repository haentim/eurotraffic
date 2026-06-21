"""Polars helpers shared by adapters to produce the canonical long frame.

Two entry points:

* :func:`hourly_from_daily` — for cities that only have a daily/annual volume per
  sensor. Cross-joins each sensor with the 24-hour canonical curve (estimated).
* :func:`hourly_from_measured` — for cities with a real per-hour count per sensor.
"""

from __future__ import annotations

import polars as pl

from .diurnal import DIURNAL_WEIGHTS
from .schema import conform

# Per-sensor metadata columns every adapter assembles before expansion.
_SENSOR_COLS = ["city", "country", "sensor_id", "longitude", "latitude", "geometry"]


def _weights_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"hour": list(range(24)), "weight": DIURNAL_WEIGHTS},
        schema={"hour": pl.Int64, "weight": pl.Float64},
    )


def hourly_from_daily(sensors: pl.DataFrame, volume_col: str = "daily_volume") -> pl.DataFrame:
    """Expand one row per sensor (with a daily volume) into 24 hourly rows.

    ``sensors`` must contain ``_SENSOR_COLS`` plus ``volume_col``. Density per hour
    is ``daily_volume * weight`` so the 24 values sum back to the daily volume.
    """
    sensors = sensors.filter(pl.col(volume_col).is_not_null() & (pl.col(volume_col) > 0))
    out = (
        sensors.select(_SENSOR_COLS + [volume_col])
        .join(_weights_frame(), how="cross")
        .with_columns((pl.col(volume_col) * pl.col("weight")).alias("density"))
    )
    return conform(out)


def hourly_from_measured(long: pl.DataFrame) -> pl.DataFrame:
    """Conform a frame that already has ``hour`` and ``density`` per sensor.

    ``long`` must contain ``_SENSOR_COLS`` plus ``hour`` and ``density``. Use when
    real sub-daily counts were aggregated to an hour-of-day profile by the adapter.
    """
    return conform(long)
