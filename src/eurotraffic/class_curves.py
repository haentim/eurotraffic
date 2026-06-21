"""Derive per-highway-class 24h diurnal curves from the measured cities.

The measured-hourly adapters (Berlin, Helsinki, Lisbon) give real per-sensor
hour-of-day profiles. We attribute each measured sensor a highway class by its
nearest treated segment (treated GeoJSONs carry ``osm_type``), normalize each
sensor's 24h profile to sum 1, and average per class across all measured cities.

Classes with too few measured sensors fall back to the canonical curve in
``diurnal.py``. The result — ``{osm_type: [24 weights]}`` — lets every street get
a road-type-appropriate temporal shape (e.g. motorways peak differently than
residential streets) while still summing to 1 so it conserves the daily volume.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from shapely.geometry import shape
from sklearn.neighbors import BallTree

from .diurnal import DIURNAL_WEIGHTS
from .registry import ADAPTERS, CityContext, discover_cities

MIN_SENSORS_PER_CLASS = 8
_DEG2RAD = np.pi / 180.0


def _treated_class_points(ctx: CityContext) -> tuple[np.ndarray, list[str]]:
    """Return (lat/lon radians array, osm_type list) for a city's treated segments."""
    files = sorted(ctx.treated_dir.glob("*.geojson"))
    if not files:
        return np.empty((0, 2)), []
    data = json.loads(files[-1].read_text())
    pts, classes = [], []
    for feat in data.get("features", []):
        geom, props = feat.get("geometry"), feat.get("properties") or {}
        osm_type = props.get("osm_type")
        if geom is None or not osm_type:
            continue
        p = shape(geom).representative_point()
        pts.append((p.y * _DEG2RAD, p.x * _DEG2RAD))
        classes.append(str(osm_type))
    return np.array(pts), classes


def _attribute_classes(sensors: pl.DataFrame, ctx: CityContext) -> pl.DataFrame:
    """Add an ``osm_type`` column to per-sensor rows via nearest treated segment."""
    pts, classes = _treated_class_points(ctx)
    if len(classes) == 0:
        return sensors.with_columns(pl.lit(None, dtype=pl.Utf8).alias("osm_type"))
    tree = BallTree(pts, metric="haversine")
    query = np.column_stack(
        [sensors["latitude"].to_numpy() * _DEG2RAD,
         sensors["longitude"].to_numpy() * _DEG2RAD]
    )
    _, idx = tree.query(query, k=1)
    return sensors.with_columns(
        pl.Series("osm_type", [classes[i] for i in idx[:, 0]], dtype=pl.Utf8)
    )


def derive_class_curves() -> dict[str, list[float]]:
    measured = [c for c in discover_cities()
                if ADAPTERS.get(c.city) and ADAPTERS[c.city].tier == "measured"]

    # Accumulate normalized 24h profiles per class across all measured cities.
    profiles: dict[str, list[np.ndarray]] = {}
    for ctx in measured:
        long = ADAPTERS[ctx.city].func(ctx)  # sensor_id, lon, lat, hour, density
        sensors = long.select("sensor_id", "longitude", "latitude").unique()
        sensors = _attribute_classes(sensors, ctx)
        labelled = long.join(sensors, on="sensor_id", how="left")

        # 24-vector per sensor, normalized to sum 1.
        wide = (
            labelled.group_by("sensor_id", "osm_type")
            .agg(pl.col("density").sort_by("hour"))  # list length 24 (hours ordered)
        )
        for row in wide.iter_rows(named=True):
            cls = row["osm_type"]
            vec = np.array(row["density"], dtype=float)
            if cls is None or vec.size != 24 or vec.sum() <= 0:
                continue
            profiles.setdefault(cls, []).append(vec / vec.sum())

    curves: dict[str, list[float]] = {}
    for cls, vecs in profiles.items():
        if len(vecs) >= MIN_SENSORS_PER_CLASS:
            mean = np.mean(np.vstack(vecs), axis=0)
            curves[cls] = (mean / mean.sum()).tolist()
    return curves


def curve_for(osm_type: str | None, curves: dict[str, list[float]]) -> list[float]:
    """Class curve if learned, else the canonical fallback."""
    return curves.get(osm_type or "", DIURNAL_WEIGHTS)
