"""Export the monolithic DB into slim, per-city files for the static web app.

For each city we write a self-contained gzipped SQLite (`web/data/<slug>.sqlite.gz`)
containing only what the browser app reads, with geometry simplified and rounded to
shrink download size, plus a `web/data/cities.json` manifest for the dropdown,
centering and per-city bbox. Run: ``python -m eurotraffic.export_web``.

Reductions (all at build time): Douglas–Peucker geometry simplification, 5-decimal
WKT precision, column drop, integer types, gzip.
"""

from __future__ import annotations

import gzip
import json
import re
import sqlite3
import time

import shapely

from . import DB_PATH, PROJECT_ROOT
from .regularize import N_HOURS

WEB_DATA = PROJECT_ROOT / "web" / "data"
SIMPLIFY_TOL = 0.00005  # ~5 m in degrees
WKT_PRECISION = 5       # ~1 m
GZIP_LEVEL = 6


def slug(city: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", city.lower()).strip("-")


def _simplify_wkt(wkts: list[str]) -> list[str]:
    geoms = shapely.from_wkt(wkts)
    geoms = shapely.simplify(geoms, SIMPLIFY_TOL, preserve_topology=False)
    return list(shapely.to_wkt(geoms, rounding_precision=WKT_PRECISION))


HOUR_COLS = [f"h{h}" for h in range(N_HOURS)]


def _city_db_bytes(rows, ceiling) -> bytes:
    """Build a self-contained in-memory SQLite for one city and serialize it."""
    hour_decl = ", ".join(f"{c} INTEGER" for c in HOUR_COLS)
    n_cols = 7 + N_HOURS
    mem = sqlite3.connect(":memory:")
    mem.executescript(
        f"""
        CREATE TABLE streets (
            osm_type TEXT, class_rank INTEGER, aadt INTEGER, source INTEGER,
            longitude REAL, latitude REAL, geometry TEXT, {hour_decl}
        );
        CREATE TABLE class_ceiling (osm_type TEXT, ceiling REAL);
        """
    )
    mem.executemany(f"INSERT INTO streets VALUES ({','.join(['?'] * n_cols)})", rows)
    mem.executemany("INSERT INTO class_ceiling VALUES (?,?)", ceiling)
    mem.execute("CREATE INDEX idx_rank ON streets (class_rank, aadt DESC)")
    mem.commit()
    data = mem.serialize()
    mem.close()
    return data


def export() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(DB_PATH)
    cities = src.execute(
        "SELECT city, country, center_lat, center_lon, n_streets, n_measured, model_r2 "
        "FROM cities ORDER BY city"
    ).fetchall()
    manifest = []
    print(f"Exporting {len(cities)} cities to {WEB_DATA} ...")
    t0 = time.time()
    hour_sel = ", ".join(HOUR_COLS)
    for city, country, clat, clon, n_streets, n_measured, r2 in cities:
        raw = src.execute(
            f"SELECT osm_type, class_rank, aadt, source, longitude, latitude, geometry, "
            f"{hour_sel} FROM streets WHERE city = ?",
            (city,),
        ).fetchall()
        if not raw:
            continue
        wkts = _simplify_wkt([r[6] for r in raw])
        rows = [
            (
                r[0],
                int(r[1]) if r[1] is not None else 10,
                int(round(r[2] or 0)),
                1 if r[3] == "measured" else 0,
                r[4], r[5], w,
                *[int(r[7 + h] or 0) for h in range(N_HOURS)],
            )
            for r, w in zip(raw, wkts)
        ]
        lons = [r[4] for r in raw]
        lats = [r[5] for r in raw]
        bbox = [min(lons), min(lats), max(lons), max(lats)]
        ceiling = src.execute(
            "SELECT osm_type, ceiling FROM class_ceiling WHERE city = ?", (city,)
        ).fetchall()

        blob = gzip.compress(_city_db_bytes(rows, ceiling), GZIP_LEVEL)
        fname = f"{slug(city)}.sqlite.gz"
        (WEB_DATA / fname).write_bytes(blob)
        manifest.append({
            "city": city, "file": fname, "country": country,
            "center_lat": clat, "center_lon": clon,
            "n_streets": n_streets, "n_measured": n_measured,
            "model_r2": r2, "bbox": bbox,
        })
        print(f"  {city:14} {len(rows):7,} streets -> {len(blob)/1e6:6.2f} MB gz")

    (WEB_DATA / "cities.json").write_text(json.dumps({"cities": manifest}))
    total = sum((WEB_DATA / m["file"]).stat().st_size for m in manifest)
    print(f"\n{len(manifest)} cities, total {total/1e6:.1f} MB gz, in {time.time()-t0:.1f}s")
    print(f"Manifest: {WEB_DATA / 'cities.json'}")


if __name__ == "__main__":
    export()
