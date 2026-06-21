"""Adapter registry and city discovery.

A city *adapter* turns one city's raw (or treated) files into the canonical long
frame. Cities without a dedicated adapter fall back to the generic treated-GeoJSON
adapter, so every city in the dataset is always covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

from . import DATASET_ROOT


@dataclass(frozen=True)
class CityContext:
    """Everything an adapter needs to locate a city's files."""

    city: str
    country: str
    city_dir: Path  # e.g. .../Germany/Berlin

    @property
    def raw_dir(self) -> Path:
        return self.city_dir / "raw"

    @property
    def treated_dir(self) -> Path:
        return self.city_dir / "treated"


Adapter = Callable[[CityContext], pl.DataFrame]


@dataclass(frozen=True)
class AdapterInfo:
    func: Adapter
    tier: str  # "measured" (real sub-daily data) or "estimated" (diurnal model)
    note: str = ""


# Populated by register(); city name -> AdapterInfo.
ADAPTERS: dict[str, AdapterInfo] = {}


def register(city: str, tier: str, note: str = "") -> Callable[[Adapter], Adapter]:
    """Decorator registering a city-specific adapter."""

    def deco(func: Adapter) -> Adapter:
        ADAPTERS[city] = AdapterInfo(func=func, tier=tier, note=note)
        return func

    return deco


def discover_cities() -> list[CityContext]:
    """Find every city that has a ``treated/`` folder under the dataset root."""
    cities: list[CityContext] = []
    for treated in sorted(DATASET_ROOT.glob("*/*/treated")):
        city_dir = treated.parent
        country = city_dir.parent.name
        cities.append(CityContext(city=city_dir.name, country=country, city_dir=city_dir))
    return cities


def adapter_for(city: str) -> AdapterInfo:
    """Return the registered adapter for a city, or the generic fallback."""
    from .adapters._treated_fallback import FALLBACK

    return ADAPTERS.get(city, FALLBACK)
