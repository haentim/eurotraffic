# EuroTraffic — Session Handoff

A Shiny-for-Python dashboard over the *Harmonized Annual Averaged Traffic Data for
European Cities* dataset (`traffic-volume-data-EU-cities/`, 39 cities). It colors a
city's whole drivable street network by estimated traffic density, by time of day.

## Standalone Pyodide web app — BUILT & verified

A fully static, browser-only version (Shinylive + Pyodide) is implemented per
**[WEB_APP_PLAN.md](WEB_APP_PLAN.md)**. Verified headlessly: app boots in-browser,
ipyleaflet renders, per-city data fetches on demand, and city switching works.
- **Data export** `src/eurotraffic/export_web.py` → per-city `web/data/<city>.sqlite.gz`
  (geometry simplified ~5 m, 5-decimal coords, columns dropped, gzip). **78 MB total**
  for 39 cities (London 12 MB); `web/data/cities.json` manifest.
- **Browser app** `web/app/app.py` — self-contained (no eurotraffic/polars/shapely);
  async per-city fetch + connection cache; viewport rendering, relative colors, hour
  slider, play, hover. Hover uses a Shiny output (NOT `ipywidgets.HTML`, which fails
  under Shinylive — key finding).
- **Build** `scripts/build_web.sh` → `site/`; `pip install -e ".[web]"` adds shinylive.
  Configurable data source via `web/app/config.json` (`build_web.sh <base-url>`),
  default co-located `site/data`. `scripts/serve_coi.py` = optional COI server for
  headless browsers (plain `http.server` works for real browsers).
- Currently served (plain http.server) at `http://127.0.0.1:8014`.

## Current state — all built & running

- **DB built**: `data/traffic.sqlite` (660 MB) — **2,356,016 streets, 39 cities**,
  all OSM networks cached in `data/networks/`.
- **App running**: `http://127.0.0.1:8770` (ipyleaflet + shinywidgets).
- Everything verified except the live in-browser pan/zoom round-trip (no browser in
  this environment) — open the URL to confirm interactively.

## Pipeline (all working)

1. **Model** (`model.py`) — `HistGradientBoostingRegressor` predicts `log1p(AADT)`
   from OSM features (`features.py`: osm_type, country, lanes, maxspeed, oneway,
   lat/lon). Within-city R²≈**0.62**. Saved to `data/model.joblib`.
2. **Road-class diurnal curves** (`class_curves.py`) — learned from the measured
   cities (Berlin/Helsinki/Lisbon) by nearest-treated-segment class attribution.
3. **Network scoring** (`network.py` + `build_db.py`) — osmnx fetches each city's
   drivable network (sensor bbox + 2 km buffer, cached). Model predicts every
   street's AADT; streets coinciding with a measured sensor (by `osmid`) are
   **anchored** to the measured value; the rest are **calibrated** to city level.
4. **Dashboard** (`app/app.py`).

## Build / run commands
```bash
cd /home/tim/eurotraffic
PYTHONPATH=src .venv/bin/python -m eurotraffic.model       # train model
PYTHONPATH=src .venv/bin/python -m eurotraffic.build_db    # fetch/score/write DB (cached)
PYTHONPATH=src .venv/bin/python -m shiny run --port 8770 app/app.py
```
Build writes the DB only at the end. To restart the server, kill it **by PID**
(`kill $(pgrep -f 'shiny.run.*8770' | head -1)`) — note `pkill -f` self-matches its
own command line and returns a spurious exit 144.

## Dashboard behaviour (per user requests)

- **Viewport rendering**: all streets in the current map frame are drawn; when too
  many, only the **largest are kept** — ranked by road-class size (`class_rank` in
  `features.ROAD_CLASS_RANK`), ties broken by daily volume (AADT). Pan/zoom
  re-queries the frame via index `idx_streets_rank (city, class_rank, aadt DESC)`.
  Cap = `MAX_FEATURES=1500`. Query: 52 ms zoomed out, ~300 ms worst-case zoomed in.
- **Relative color scale**: color = `density / class_ceiling[city, osm_type]`, so
  smaller streets light up at their own (lower) volumes. `class_ceiling` = p95 AADT
  per (city, class) × class peak weight (hour-independent, so the slider still
  varies colors). Built by `build_db.compute_class_ceilings`.
- The hour slider only **recolors** (street selection is hour-independent).

## SQLite schema
- `streets(street_id, city, country, osm_type, class_rank, lanes, maxspeed, oneway,
  length_m, aadt, source['measured'|'predicted'], longitude, latitude, geometry[WKT])`
  — indexes: `idx_streets_rank (city, class_rank, aadt DESC)`.
- `class_diurnal(osm_type, hour, weight)` — 24 weights/class; `'_default'` = canonical.
- `class_ceiling(city, osm_type, ceiling)` — per-class color ceiling.
- `cities(city, country, center_lat, center_lon, n_streets, n_measured, model_r2)`.
- Density per street/hour = `aadt × class_diurnal[osm_type, hour]`.

## Verified
- 39 cities, no zero-anchor cities (min 13); predicted AADT by class sane
  (motorway ≫ tertiary ≫ residential) across London/Madrid/Lisbon/Barcelona.
- Viewport query correct + fast both regimes; relative coloring makes the busiest
  street of every class reach red.
- Server boots, serves 200; ipyleaflet/GeoJSON API usage valid.

## Possible follow-ups
- A handful of mis-predicted residential outliers (e.g. one London residential
  street at ~94k AADT); p95 ceilings already keep these from distorting the scale.
- Class diurnal curves cover 4 classes (motorway/primary/secondary/tertiary); others
  fall back to the canonical curve — could learn more if desired.
- DB rebuild from scratch re-downloads any networks missing from `data/networks/`.
