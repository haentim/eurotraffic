"""Helsinki adapter (tier-1, measured hourly).

``hki_liikennemaarat.csv`` holds per-station, per-direction, per-hour vehicle
counts across many years. Columns (``;`` separated):

* ``piste``        station id
* ``x_gk25``/``y_gk25``  coordinates in ETRS-GK25 (EPSG:3879)
* ``suunta``       direction
* ``aika``         time of day as HHMM (e.g. 0, 100, .. 2300; some 15-min values)
* ``vuosi``        year
* ``autot``        total motor vehicles in the interval

We sum directions and sub-hour intervals into an hourly total per year, then
average across years to get a representative hour-of-day density per station.
"""

from __future__ import annotations

import polars as pl

from ..frames import hourly_from_measured
from ..geometry import points_to_wgs84
from ..registry import CityContext, register

GK25 = "EPSG:3879"
_CSV = "hki_liikennemaarat.csv"


@register("Helsinki", tier="measured", note="hourly station counts (hki_liikennemaarat)")
def load(ctx: CityContext) -> pl.DataFrame:
    path = ctx.raw_dir / _CSV
    df = pl.read_csv(path, separator=";", encoding="utf8-lossy")

    df = (
        df.with_columns(
            (pl.col("aika") // 100).cast(pl.Int64).alias("hour"),
            pl.col("autot").cast(pl.Float64),
        )
        .filter(pl.col("hour").is_between(0, 23) & pl.col("autot").is_not_null())
    )

    # Hourly total per station/year (sum directions + sub-hour intervals), then
    # mean across years -> representative density.
    per_year = df.group_by(["piste", "vuosi", "hour"]).agg(
        pl.col("autot").sum().alias("hourly_total")
    )
    density = per_year.group_by(["piste", "hour"]).agg(
        pl.col("hourly_total").mean().alias("density")
    )

    # One coordinate pair per station, reprojected in one bulk call.
    station_xy = df.group_by("piste").agg(
        pl.col("x_gk25").first(), pl.col("y_gk25").first()
    )
    lons, lats, wkts = points_to_wgs84(
        station_xy["x_gk25"].cast(pl.Float64).to_list(),
        station_xy["y_gk25"].cast(pl.Float64).to_list(),
        GK25,
    )
    coords = station_xy.select("piste").with_columns(
        longitude=pl.Series(lons, dtype=pl.Float64),
        latitude=pl.Series(lats, dtype=pl.Float64),
        geometry=pl.Series(wkts, dtype=pl.Utf8),
    )

    long = (
        density.join(coords, on="piste", how="inner")
        .with_columns(
            pl.lit(ctx.city).alias("city"),
            pl.lit(ctx.country).alias("country"),
            pl.col("piste").cast(pl.Utf8).alias("sensor_id"),
        )
        .drop("piste")
    )
    return hourly_from_measured(long)
