"""Fetch and cache each city's drivable street network from OpenStreetMap.

Uses osmnx over the city's *sensor bounding box + buffer* (keeps big cities
tractable while covering the measured area). Returns a cleaned Polars frame — one
row per street segment — with scalar OSM features ready for the density model, the
representative point, WGS84 WKT geometry, and the underlying ``osmid``s (for
anchoring measured sensors). Cleaned frames are cached to
``data/networks/<city>.parquet`` so rebuilds don't re-download.
"""

from __future__ import annotations

import json
import math

import geopandas as gpd
import osmnx as ox
import polars as pl
from shapely.geometry import shape

from . import PROJECT_ROOT
from .features import clean_highway, clean_lanes, clean_maxspeed, clean_oneway
from .geometry import normalize
from .registry import CityContext

NETWORK_CACHE = PROJECT_ROOT / "data" / "networks"
DEFAULT_BUFFER_KM = 2.0


def sensor_bbox(ctx: CityContext, buffer_km: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) around a city's treated sensor extent + buffer."""
    files = sorted(ctx.treated_dir.glob("*.geojson"))
    data = json.loads(files[-1].read_text())
    lons, lats = [], []
    for feat in data.get("features", []):
        g = feat.get("geometry")
        if not g:
            continue
        p = shape(g).representative_point()
        lons.append(p.x)
        lats.append(p.y)
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    mid_lat = (south + north) / 2
    dlat = buffer_km / 111.0
    dlon = buffer_km / (111.0 * max(math.cos(math.radians(mid_lat)), 0.1))
    return (west - dlon, south - dlat, east + dlon, north + dlat)


def _osmid_list(value) -> str:
    if isinstance(value, (list, tuple)):
        ids = value
    else:
        ids = [value]
    out = []
    for v in ids:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return json.dumps(out)


def _clean_edges(edges: gpd.GeoDataFrame, ctx: CityContext) -> pl.DataFrame:
    rows = []
    for i, e in enumerate(edges.itertuples(index=False)):
        geom = e.geometry
        if geom is None:
            continue
        lon, lat, wkt = normalize(geom)  # osmnx geometries are EPSG:4326
        rows.append(
            {
                "street_id": f"{ctx.city}:{i}",
                "osmids": _osmid_list(getattr(e, "osmid", None)),
                "osm_type": clean_highway(getattr(e, "highway", None)),
                "lanes": clean_lanes(getattr(e, "lanes", None)),
                "maxspeed": clean_maxspeed(getattr(e, "maxspeed", None)),
                "oneway": clean_oneway(getattr(e, "oneway", None)),
                "length_m": float(getattr(e, "length", 0.0) or 0.0),
                "longitude": lon,
                "latitude": lat,
                "geometry": wkt,
            }
        )
    return pl.DataFrame(rows)


def fetch_edges(ctx: CityContext, buffer_km: float = DEFAULT_BUFFER_KM) -> pl.DataFrame:
    """Cleaned street edges for a city (cached). Raises on download failure."""
    cache = NETWORK_CACHE / f"{ctx.city}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)

    bbox = sensor_bbox(ctx, buffer_km)
    graph = ox.graph_from_bbox(bbox, network_type="drive")
    edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()
    cleaned = _clean_edges(edges, ctx)

    NETWORK_CACHE.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(cache)
    return cleaned
