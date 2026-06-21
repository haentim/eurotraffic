# EuroTraffic

Interactive Shiny-for-Python dashboard of traffic density across European cities,
built on the *Harmonized Annual Averaged Traffic Data at Street Segment Level for
European Cities* dataset (`traffic-volume-data-EU-cities/`).

Pick a city, scrub the **time-of-day slider**, and a Folium/Leaflet map colors the
**whole drivable street network** by estimated traffic density for that hour.

## How it works

Every street gets a base **AADT** (daily volume) and a road-class **hourly shape**;
density = `aadt × class_diurnal[osm_type, hour]`.

1. **Density model** (`model.py`) — a `HistGradientBoostingRegressor` trained on the
   105k treated sensor segments to predict `log(AADT)` from OSM road features
   (`osm_type`, lanes, maxspeed, oneway, location). Within-city R² ≈ 0.62. Feature
   cleaning is shared by training and inference in `features.py`. Saved to
   `data/model.joblib`.

2. **Road-class diurnal curves** (`class_curves.py`) — real per-hour profiles from
   the measured cities (Berlin, Helsinki, Lisbon) are attributed a highway class by
   nearest treated segment and averaged into a normalized 24h curve per `osm_type`
   (fallback: canonical curve in `diurnal.py`).

3. **Network scoring** (`network.py` + `build_db.py`) — for each city, `osmnx`
   fetches the drivable network over the sensor bounding box + buffer (cached to
   `data/networks/<city>.parquet`). The model predicts every street's AADT; streets
   that coincide with a measured sensor (by `osmid`) are **anchored** to the measured
   value, and the remaining predictions are **calibrated** to the city's level.
   Output tables in `data/traffic.sqlite`: `streets`, `class_diurnal`,
   `class_ceiling`, `cities`.

3b. **Graph regularization** (`regularize.py`) — independent per-street predictions
    are smoothed over the street-network graph so connected segments of the same road
    stay consistent. Minimizes `Σ wᵢ(xᵢ−yᵢ)² + λ Σ_E (xᵢ−xⱼ)²` in log-AADT (sparse
    CG solve), where adjacency links **same-class** segments sharing an endpoint (a
    motorway isn't averaged into a residential street it crosses) and measured anchors
    carry a high weight so they barely move. Small inconsistencies remain by design.

4. **Dashboard** (`app/app.py`) — an interactive `ipyleaflet` map (via
   `shinywidgets`) colored by density. **All streets in the current frame are drawn;
   when a frame holds too many, only the largest are kept** — ranked by road-class
   size (`class_rank`), ties broken by daily volume (AADT). Panning/zooming
   re-queries the visible frame (index `idx_streets_rank`), so zooming in reveals the
   smaller streets while zoomed-out views show the arterials. The hour slider only
   recolors; the sidebar shows the measured-anchored share and the model's R².
   Colors are **relative to road class** (`class_ceiling` table: p95 AADT × class
   peak weight), so a busy residential street is highlighted like a busy motorway
   instead of staying blue under the city's absolute maximum.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .        # or: pip install -r requirements (deps in pyproject.toml)
```

## Source data

The upstream dataset is **not vendored** here. Clone it before running the
preprocessing pipeline:

```bash
scripts/fetch_data_repo.sh   # clones github.com/XavB64/traffic-volume-data-EU-cities
```

## Build the database

```bash
PYTHONPATH=src .venv/bin/python -m eurotraffic.model       # train + save data/model.joblib
PYTHONPATH=src .venv/bin/python -m eurotraffic.build_db    # fetch networks, score, write sqlite
```

The first `build_db` run downloads each city's OSM network (cached under
`data/networks/`); later runs reuse the cache. Delete a city's parquet to refetch.

## Run the dashboard

```bash
PYTHONPATH=src .venv/bin/python -m shiny run app/app.py
# open http://127.0.0.1:8000
```

## Measured-hourly adapters

The measured cities feed the road-class curves and AADT anchors. Each lives in
`src/eurotraffic/adapters/<city>.py`, decorated with
`@register("<City>", tier="measured", ...)` returning a Polars frame via
`frames.hourly_from_measured`; import it in `adapters/__init__.py`.

## Standalone web app (Pyodide / Shinylive)

A fully static, server-less version of the dashboard runs entirely in the browser
via Pyodide. The 660 MB DB is split into slim, simplified, gzipped **per-city** files
(`web/data/<city>.sqlite.gz`, ~78 MB total, London ~12 MB) fetched **on demand** when
a city is selected; everything else (querying, coloring, rendering) happens
client-side. Source: `web/app/app.py`; data export: `src/eurotraffic/export_web.py`.

```bash
.venv/bin/python -m pip install -e ".[web]"   # adds shinylive
scripts/build_web.sh                          # -> site/  (data co-located in site/data)
.venv/bin/python -m http.server --directory site 8000
# open http://127.0.0.1:8000  (serve from the site root)
```

Deploys to any static host and works offline once loaded.

### Deploy to GitHub Pages

`.github/workflows/deploy.yml` builds and publishes automatically on push to `main`
(enable Pages → "GitHub Actions" in the repo settings). The committed per-city files
in `web/data/` are bundled into `site/data/`; the app **auto-detects** the co-located
`data/` folder for both root- and project-subpath (`/eurotraffic/`) sites, so no data
URL needs configuring.

### Hosting data on a separate origin (optional)

To serve the per-city files from a CDN/object storage instead, build with a base URL
and deploy `web/data/*.sqlite.gz` there:

```bash
scripts/build_web.sh https://cdn.example.com/eurotraffic/data
```

This bakes the base into `web/app/config.json`; the app then fetches each city from
that URL.

Notes: the in-map hover box uses a Shiny-native output rather than `ipywidgets.HTML`
(which fails under Shinylive). `scripts/serve_coi.py` is an optional COOP/COEP server
for headless/CI browsers (a normal browser doesn't need it).
