"""Geometry helpers: reproject to WGS84, derive a representative point, emit WKT.

Adapters call :func:`normalize` with a shapely geometry (or raw coordinates) and
a source CRS; it returns ``(longitude, latitude, wkt)`` in EPSG:4326. This keeps
every city's output consistent regardless of whether the raw data arrived as
GeoJSON points, projected national-grid coordinates, or shapefiles.
"""

from __future__ import annotations

from functools import lru_cache

from pyproj import Transformer
from shapely import to_wkt
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

WGS84 = "EPSG:4326"


@lru_cache(maxsize=64)
def _transformer(src_crs: str) -> Transformer:
    # always_xy keeps (lon, lat) / (x, y) ordering consistent.
    return Transformer.from_crs(src_crs, WGS84, always_xy=True)


def to_wgs84(geom: BaseGeometry, src_crs: str) -> BaseGeometry:
    """Reproject a shapely geometry to EPSG:4326. No-op if already WGS84."""
    if src_crs in (WGS84, "wgs84", "EPSG:4326", "4326"):
        return geom
    tf = _transformer(src_crs)
    return shapely_transform(lambda x, y, z=None: tf.transform(x, y), geom)


def representative_point(geom: BaseGeometry) -> Point:
    """A single representative point: the point itself, else the centroid."""
    if geom.geom_type == "Point":
        return geom
    # representative_point() is guaranteed to lie on the geometry (unlike centroid).
    return geom.representative_point()


def normalize(geom: BaseGeometry, src_crs: str = WGS84) -> tuple[float, float, str]:
    """Return ``(longitude, latitude, wkt)`` in WGS84 for any input geometry."""
    g = to_wgs84(geom, src_crs)
    pt = representative_point(g)
    return float(pt.x), float(pt.y), to_wkt(g, rounding_precision=6)


def point_from_xy(x: float, y: float, src_crs: str = WGS84) -> tuple[float, float, str]:
    """Convenience for adapters that only have a coordinate pair."""
    return normalize(Point(x, y), src_crs)


def points_to_wgs84(
    xs: list[float], ys: list[float], src_crs: str = WGS84
) -> tuple[list[float], list[float], list[str]]:
    """Vectorized point conversion: return parallel lon/lat/WKT lists.

    Reprojects all coordinates in one bulk pyproj call (NOT per-row inside a
    Polars UDF, which can crash the native extension). Use this from adapters.
    """
    if src_crs in (WGS84, "wgs84", "EPSG:4326", "4326"):
        lons, lats = list(xs), list(ys)
    else:
        tf = _transformer(src_crs)
        lons, lats = tf.transform(xs, ys)
        lons, lats = list(lons), list(lats)
    wkts = [f"POINT ({lon:.6f} {lat:.6f})" for lon, lat in zip(lons, lats)]
    return lons, lats, wkts


def line_from_coords(
    coords: list[tuple[float, float]], src_crs: str = WGS84
) -> tuple[float, float, str]:
    """Convenience for adapters that have an ordered list of coordinates."""
    return normalize(LineString(coords), src_crs)
