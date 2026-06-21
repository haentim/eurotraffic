"""Shared OSM feature engineering for the density model.

Both training (from treated GeoJSONs) and inference (from osmnx network edges)
must transform raw OSM tags identically, or the model sees inconsistent inputs.
This module is the single source of truth for that cleaning.

Feature parity note: treated training segments are mostly *points* (sensors) with
no length, while network edges have length. To keep features identical across
train/inference we deliberately exclude segment length and rely on tags + location.
"""

from __future__ import annotations

import re

# Model feature columns. ``osm_type`` and ``country`` are categorical.
FEATURE_COLS = ["osm_type", "country", "lanes", "maxspeed", "oneway", "latitude", "longitude"]
CATEGORICAL_COLS = ["osm_type", "country"]
TARGET = "aadt"

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Road-class "size" ranking — smaller number = bigger/more important road. Used to
# decide which streets to show when a map frame holds too many: largest classes
# first, ties broken by daily volume (AADT). Links rank just below their parent
# class; anything unknown sorts last.
ROAD_CLASS_RANK: dict[str, int] = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "motorway_link": 5,
    "trunk_link": 5,
    "unclassified": 6,
    "primary_link": 6,
    "secondary_link": 6,
    "residential": 7,
    "tertiary_link": 7,
    "living_street": 8,
    "busway": 8,
    "service": 9,
    "pedestrian": 9,
}
DEFAULT_CLASS_RANK = 10


def class_rank(osm_type) -> int:
    return ROAD_CLASS_RANK.get(clean_highway(osm_type) or "", DEFAULT_CLASS_RANK)


def _first(value):
    """OSM tags are sometimes lists (e.g. ``['residential','service']``)."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def clean_highway(value) -> str | None:
    v = _first(value)
    if v is None or v == "":
        return None
    return str(v)


def clean_lanes(value) -> float | None:
    v = _first(value)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        m = _NUM_RE.search(str(v))
        return float(m.group()) if m else None


def clean_maxspeed(value) -> float | None:
    """Return speed in km/h. Handles ``'30 mph'``, ``'50'``, lists, ``'walk'``."""
    v = _first(value)
    if v is None or v == "":
        return None
    s = str(v)
    m = _NUM_RE.search(s)
    if not m:
        return None
    speed = float(m.group())
    if "mph" in s.lower():
        speed *= 1.60934
    return speed


def clean_oneway(value) -> float | None:
    v = _first(value)
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return float(v)
    s = str(v).strip().lower()
    if s in ("yes", "true", "1", "-1"):
        return 1.0
    if s in ("no", "false", "0"):
        return 0.0
    return None


def feature_record(
    *, osm_type, country: str, lanes, maxspeed, oneway, latitude: float, longitude: float
) -> dict:
    """Assemble one cleaned feature row (raw tag values in, model features out)."""
    return {
        "osm_type": clean_highway(osm_type),
        "country": country,
        "lanes": clean_lanes(lanes),
        "maxspeed": clean_maxspeed(maxspeed),
        "oneway": clean_oneway(oneway),
        "latitude": float(latitude),
        "longitude": float(longitude),
    }
