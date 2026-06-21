"""Lisbon adapter (tier-1, measured hourly).

``trafegoall.csv`` holds timestamped traffic readings. Rows with
``COD_PARAMETRO == '0VTH'`` are hourly traffic volumes (Volume de Tráfego
Horário); ``0TMD`` rows are daily means and are ignored. Coordinates are inline
(WGS84). Density per sensor/hour = both-direction total, averaged over all days.
"""

from __future__ import annotations

import polars as pl

from ..frames import hourly_from_measured
from ..geometry import points_to_wgs84
from ..registry import CityContext, register

_CSV = "trafegoall.csv"


@register("Lisbon", tier="measured", note="hourly volumes (trafegoall, COD_PARAMETRO=0VTH)")
def load(ctx: CityContext) -> pl.DataFrame:
    path = ctx.raw_dir / _CSV
    df = pl.read_csv(path, infer_schema_length=2000)

    df = (
        df.filter(pl.col("COD_PARAMETRO") == "0VTH")
        .with_columns(
            pl.col("DTM_LOCAL").str.slice(11, 2).cast(pl.Int64).alias("hour"),
            (
                pl.col("SMO_FROMTO_TOTAL").cast(pl.Float64, strict=False).fill_null(0)
                + pl.col("SMO_TOFROM_TOTAL").cast(pl.Float64, strict=False).fill_null(0)
            ).alias("count"),
        )
        .filter(pl.col("hour").is_between(0, 23))
    )

    density = df.group_by(["COD_SENSOR", "hour"]).agg(
        pl.col("count").mean().alias("density"),
        pl.col("LATITUDE").first().cast(pl.Float64),
        pl.col("LONGITUDE").first().cast(pl.Float64),
    )

    lons, lats, wkts = points_to_wgs84(
        density["LONGITUDE"].to_list(), density["LATITUDE"].to_list()
    )
    long = (
        density.with_columns(
            longitude=pl.Series(lons, dtype=pl.Float64),
            latitude=pl.Series(lats, dtype=pl.Float64),
            geometry=pl.Series(wkts, dtype=pl.Utf8),
        )
        .with_columns(
            pl.lit(ctx.city).alias("city"),
            pl.lit(ctx.country).alias("country"),
            pl.col("COD_SENSOR").cast(pl.Utf8).alias("sensor_id"),
        )
        .drop("COD_SENSOR", "LATITUDE", "LONGITUDE")
    )
    return hourly_from_measured(long)
