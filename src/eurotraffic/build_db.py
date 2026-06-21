"""Build the SQLite database of per-street estimated traffic density.

Run with: ``python -m eurotraffic.build_db`` (after ``python -m eurotraffic.model``).

For each city it:
  1. fetches the drivable OSM street network (cached),
  2. predicts each street's AADT with the gradient-boosted model,
  3. anchors streets that coincide with a measured sensor (by ``osmid``) to the
     measured AADT, and calibrates the remaining predictions to the city's level,
  4. writes one row per street.

Hourly density is derived in the app as ``aadt × class_diurnal[osm_type, hour]``.
Tables written: ``streets``, ``class_diurnal``, ``cities``.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import traceback

import joblib
import numpy as np
import polars as pl

from . import DB_PATH
from .class_curves import derive_class_curves
from .features import (
    CATEGORICAL_COLS,
    DEFAULT_CLASS_RANK,
    FEATURE_COLS,
    ROAD_CLASS_RANK,
    standardize_coords,
)
from .model import MODEL_PATH
from .network import fetch_edges
from .regularize import N_HOURS, regularize_hourly
from .registry import CityContext, discover_cities

# Per-hour flow-consistency regularization knobs.
REG_MU_FLOW = 0.2     # strictness of the "no dominating edge" flow constraint
REG_MU_TIME = 0.5     # temporal smoothness of the per-hour correction
MEASURED_WEIGHT = 80.0  # how strongly sensor-anchored streets resist being moved
# Capacity fallback per road class (lanes, maxspeed km/h) when OSM tags are missing.
_CAP_FALLBACK_LANES = 1.5
_CAP_FALLBACK_SPEED = 50.0
HOUR_COLS = [f"h{h}" for h in range(N_HOURS)]


def _load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle.get("cv_r2", float("nan"))


def _treated_osmid_aadt(ctx: CityContext) -> dict[int, float]:
    """Map ``osmid -> measured AADT`` from a city's latest treated GeoJSON."""
    files = sorted(ctx.treated_dir.glob("*.geojson"))
    if not files:
        return {}
    data = json.loads(files[-1].read_text())
    acc: dict[int, list[float]] = {}
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        osmid, aadt = props.get("osmid"), props.get("AADT") or props.get("AAWT")
        if osmid in (None, "") or aadt in (None, "", 0):
            continue
        try:
            acc.setdefault(int(float(osmid)), []).append(float(aadt))
        except (TypeError, ValueError):
            continue
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def _predict(model, edges: pl.DataFrame, country: str) -> np.ndarray:
    pdf = edges.select("osm_type", "lanes", "maxspeed", "oneway",
                       "latitude", "longitude").to_pandas()
    pdf["country"] = country
    # Per-city standardized coordinates, matching the training-time transform.
    pdf["x_norm"], pdf["y_norm"] = standardize_coords(pdf["longitude"], pdf["latitude"])
    X = pdf[FEATURE_COLS]
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    return np.expm1(model.predict(X))


def _capacity(edges: pl.DataFrame) -> np.ndarray:
    """Per-edge capacity proxy ``lanes × maxspeed`` (km/h), with class-median then
    constant fallback where OSM tags are missing."""
    e = edges.with_columns(
        pl.col("lanes").fill_null(pl.col("lanes").median().over("osm_type")),
        pl.col("maxspeed").fill_null(pl.col("maxspeed").median().over("osm_type")),
    ).with_columns(
        pl.col("lanes").fill_null(_CAP_FALLBACK_LANES),
        pl.col("maxspeed").fill_null(_CAP_FALLBACK_SPEED),
    )
    cap = (e["lanes"] * e["maxspeed"]).to_numpy()
    fallback = _CAP_FALLBACK_LANES * _CAP_FALLBACK_SPEED
    return np.where(np.isfinite(cap) & (cap > 0), cap, fallback)


def _score_city(ctx: CityContext, model, curves: dict) -> pl.DataFrame:
    edges = fetch_edges(ctx)
    pred = _predict(model, edges, ctx.country)

    osmid_aadt = _treated_osmid_aadt(ctx)
    aadt = pred.astype(float).copy()
    source = np.array(["predicted"] * len(aadt), dtype=object)

    # Anchor streets that coincide with a measured sensor; collect (pred, measured)
    # pairs for per-city calibration.
    matched_pred, matched_meas = [], []
    osmids_col = edges["osmids"].to_list()
    for i, ids_json in enumerate(osmids_col):
        vals = [osmid_aadt[o] for o in json.loads(ids_json) if o in osmid_aadt]
        if vals:
            m = float(np.mean(vals))
            matched_pred.append(pred[i])
            matched_meas.append(m)
            aadt[i] = m
            source[i] = "measured"

    # Multiplicative (log-space) calibration of predicted streets to city level.
    if matched_meas:
        offset = float(
            np.mean(np.log1p(matched_meas)) - np.mean(np.log1p(matched_pred))
        )
        pred_mask = source == "predicted"
        aadt[pred_mask] = np.expm1(np.log1p(aadt[pred_mask]) + offset)

    # Per-hour flow-consistency regularization. Starts from the prior per-hour field
    # (daily volume × class diurnal curve), enforces "no dominating edge" at each
    # vertex/hour, keeps sensor anchors pinned, lets high-capacity roads adjust more,
    # and keeps the hourly correction temporally smooth. Returns C[n, 24].
    C = regularize_hourly(
        aadt, edges["geometry"].to_list(), edges["osm_type"].to_list(), curves,
        capacity=_capacity(edges), measured=(source == "measured"),
        mu_flow=REG_MU_FLOW, mu_time=REG_MU_TIME, measured_weight=MEASURED_WEIGHT,
    )
    daily = C.sum(axis=1)  # regularized daily total (for ranking / size selection)
    hour_cols = [pl.Series(HOUR_COLS[h], np.rint(C[:, h]).astype(np.int64)) for h in range(N_HOURS)]

    return edges.with_columns(
        pl.lit(ctx.city).alias("city"),
        pl.lit(ctx.country).alias("country"),
        pl.Series("aadt", daily, dtype=pl.Float64),
        pl.Series("source", source.tolist(), dtype=pl.Utf8),
        pl.col("osm_type")
        .replace_strict(ROAD_CLASS_RANK, default=DEFAULT_CLASS_RANK, return_dtype=pl.Int64)
        .alias("class_rank"),
        *hour_cols,
    ).select(
        "street_id", "city", "country", "osm_type", "class_rank", "lanes", "maxspeed",
        "oneway", "length_m", "aadt", "source", "longitude", "latitude", "geometry",
        *HOUR_COLS,
    )


def build(buffer_km: float | None = None) -> None:
    model, cv_r2 = _load_model()
    print("Deriving road-class diurnal curves from measured cities...")
    curves = derive_class_curves()
    print(f"  learned curves for: {sorted(curves)}")

    frames: list[pl.DataFrame] = []
    city_rows: list[dict] = []
    for ctx in discover_cities():
        try:
            streets = _score_city(ctx, model, curves)
        except Exception as exc:  # noqa: BLE001 - keep building other cities
            print(f"  x {ctx.city}: FAILED ({exc})", file=sys.stderr)
            traceback.print_exc()
            continue
        n_meas = int((streets["source"] == "measured").sum())
        frames.append(streets)
        city_rows.append({
            "city": ctx.city, "country": ctx.country,
            "center_lat": streets["latitude"].median(),
            "center_lon": streets["longitude"].median(),
            "n_streets": streets.height, "n_measured": n_meas,
            "model_r2": float(cv_r2),
        })
        print(f"  + {ctx.city:14} {streets.height:7,} streets  ({n_meas} measured-anchored)")

    if not frames:
        raise SystemExit("No cities built; aborting.")

    streets = pl.concat(frames, how="vertical")
    cities = pl.DataFrame(city_rows).sort("city")
    ceilings = compute_class_ceilings(streets)
    _write_sqlite(streets, cities, ceilings)
    print(f"\nWrote {streets.height:,} streets for {cities.height} cities to {DB_PATH}")


def compute_class_ceilings(streets: pl.DataFrame) -> list[tuple]:
    """Per-(city, road class) color ceiling so the scale is relative to street size.

    Now that the per-hour field is stored directly, the ceiling is the p95 (robust to
    outliers) of each class's *peak-hour* density. Hour-independent, so the slider
    still varies colors while small streets light up at their own (lower) volumes.
    """
    peak_expr = pl.max_horizontal(HOUR_COLS).alias("peak")
    grouped = (
        streets.with_columns(peak_expr)
        .group_by(["city", "osm_type"])
        .agg(pl.col("peak").quantile(0.95).alias("p95"))
    )
    return [
        (r["city"], r["osm_type"], max(float(r["p95"] or 1.0), 1e-6))
        for r in grouped.iter_rows(named=True)
    ]


def _write_sqlite(streets, cities, ceilings) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    hour_cols_sql = ", ".join(f"{c} INTEGER" for c in HOUR_COLS)
    base_cols = [
        "street_id", "city", "country", "osm_type", "class_rank", "lanes",
        "maxspeed", "oneway", "length_m", "aadt", "source",
        "longitude", "latitude", "geometry",
    ]
    all_cols = base_cols + HOUR_COLS
    placeholders = ",".join(["?"] * len(all_cols))
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(
            f"""
            CREATE TABLE streets (
                street_id TEXT, city TEXT, country TEXT, osm_type TEXT,
                class_rank INTEGER, lanes REAL, maxspeed REAL, oneway REAL,
                length_m REAL, aadt REAL, source TEXT,
                longitude REAL, latitude REAL, geometry TEXT,
                {hour_cols_sql}
            );
            CREATE TABLE class_ceiling (city TEXT, osm_type TEXT, ceiling REAL);
            CREATE TABLE cities (
                city TEXT PRIMARY KEY, country TEXT,
                center_lat REAL, center_lon REAL,
                n_streets INTEGER, n_measured INTEGER, model_r2 REAL
            );
            """
        )
        con.executemany(
            f"INSERT INTO streets VALUES ({placeholders})",
            streets.select(all_cols).rows(),
        )
        con.executemany("INSERT INTO class_ceiling VALUES (?,?,?)", ceilings)
        con.executemany(
            "INSERT INTO cities VALUES (?,?,?,?,?,?,?)",
            cities.select("city", "country", "center_lat", "center_lon",
                          "n_streets", "n_measured", "model_r2").rows(),
        )
        # (city, class_rank, aadt DESC) lets the app pull the largest streets in a
        # frame (biggest road class first, ties by daily volume) via an index walk
        # instead of sorting the whole city on every pan/zoom.
        con.execute(
            "CREATE INDEX idx_streets_rank ON streets (city, class_rank, aadt DESC)"
        )
        con.commit()
    finally:
        con.close()


if __name__ == "__main__":
    from . import adapters  # noqa: F401  (registers measured adapters for curves)

    build()
