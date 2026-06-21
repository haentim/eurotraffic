"""Canonical schema for the harmonized traffic table.

Every city adapter must return a Polars DataFrame with exactly these columns and
dtypes. ``build_db`` concatenates them and writes them to SQLite.
"""

from __future__ import annotations

import polars as pl

# Uniform long-format columns. One row per (city, sensor, hour).
COLUMNS: dict[str, pl.DataType] = {
    "city": pl.Utf8,
    "country": pl.Utf8,
    "sensor_id": pl.Utf8,
    "longitude": pl.Float64,
    "latitude": pl.Float64,
    "geometry": pl.Utf8,  # WKT in EPSG:4326 (POINT(...) or LINESTRING(...))
    "hour": pl.Int64,  # 0..23, the time-of-day slider dimension
    "density": pl.Float64,  # estimated vehicles/hour at this sensor for this hour
}

COLUMN_ORDER = list(COLUMNS)


def empty_frame() -> pl.DataFrame:
    """Return an empty DataFrame with the canonical schema."""
    return pl.DataFrame(schema=COLUMNS)


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Validate and coerce an adapter's output to the canonical schema.

    Raises if required columns are missing; reorders/casts the rest so the
    concatenation in ``build_db`` is always clean.
    """
    missing = set(COLUMN_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"adapter output missing columns: {sorted(missing)}")
    return df.select(
        [pl.col(name).cast(dtype) for name, dtype in COLUMNS.items()]
    )
