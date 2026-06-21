"""Generic fallback adapter: read a city's treated AADT/AAWT GeoJSON.

The harmonized ``treated/`` GeoJSONs only carry annual averages (AADT / AAWT), so
this adapter expands each feature's daily volume into a 24-hour *estimated* profile
via the canonical diurnal curve. It is the default for any city without a
dedicated raw-data adapter, guaranteeing all cities appear in the dashboard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
from shapely.geometry import shape

from ..frames import hourly_from_daily
from ..geometry import normalize
from ..registry import AdapterInfo, CityContext

# Volume property to use, in order of preference (AADT = all days, AAWT = weekday).
_VOLUME_KEYS = ("AADT", "AAWT")

_YEAR_RE = re.compile(r"(\d{4})")


def _latest_geojson(treated_dir: Path) -> Path | None:
    files = sorted(treated_dir.glob("*.geojson"))
    if not files:
        return None
    # Prefer the file with the most recent 4-digit year in its name.
    def year_key(p: Path) -> int:
        years = [int(y) for y in _YEAR_RE.findall(p.stem)]
        return max(years) if years else 0

    return max(files, key=year_key)


def _pick_volume(props: dict) -> float | None:
    for key in _VOLUME_KEYS:
        val = props.get(key)
        if val is not None:
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if f > 0:
                return f
    return None


def load(ctx: CityContext) -> pl.DataFrame:
    path = _latest_geojson(ctx.treated_dir)
    if path is None:
        raise FileNotFoundError(f"no treated geojson for {ctx.city}")

    data = json.loads(path.read_text())
    rows: list[dict] = []
    for i, feat in enumerate(data.get("features", [])):
        geom = feat.get("geometry")
        props = feat.get("properties") or {}
        if geom is None:
            continue
        volume = _pick_volume(props)
        if volume is None:
            continue
        lon, lat, wkt = normalize(shape(geom))  # treated geojson is already WGS84
        rows.append(
            {
                "city": ctx.city,
                "country": ctx.country,
                "sensor_id": str(props.get("osmid") or props.get("raw_name") or i),
                "longitude": lon,
                "latitude": lat,
                "geometry": wkt,
                "daily_volume": volume,
            }
        )

    if not rows:
        raise ValueError(f"no usable features in {path.name}")

    sensors = pl.DataFrame(rows)
    return hourly_from_daily(sensors)


FALLBACK = AdapterInfo(
    func=load,
    tier="estimated",
    note="treated AADT/AAWT annual average expanded with canonical diurnal curve",
)
