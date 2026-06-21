# Plan: Standalone browser app via Pyodide (Shinylive)

## Context

The EuroTraffic dashboard currently runs as a server-side Shiny-for-Python app
(`app/app.py`) backed by a 660 MB SQLite DB. We want a **standalone, fully static
web app** — no Python server — that runs entirely in the browser via **Pyodide**,
using **Shinylive** (the official "export a Shiny app to static WASM" tool).

Decisions (user-selected): cover **all 39 cities, fetched on demand** (one slim data
file per city); **port the full app** (viewport-reactive `ipyleaflet` map + hour
slider + play button) via Shinylive; ship **slimmed web data** (reduced precision,
dropped columns, gzip).

Two hard constraints shape the design:
1. **Data size** — 660 MB can't be bundled. We split it into per-city slim SQLite
   files fetched on demand and queried in-browser with stdlib `sqlite3` (available in
   Pyodide).
2. **`ipyleaflet` under Pyodide is unproven here** — it works in JupyterLite but
   Shinylive support is uncertain. So the very first step is a feasibility spike;
   the rest of the plan only proceeds if it passes (fallback noted).

Runtime deps are light (no polars/sklearn/osmnx): `sqlite3`, `shiny`, `shinywidgets`,
`ipywidgets`, `ipyleaflet`, `branca`. The web app will **drop `shapely`** by parsing
`LINESTRING` WKT into coordinate arrays directly (one small helper), shrinking the
Pyodide dependency set.

## Step 0 — Feasibility spike (do FIRST, gates everything)

Minimal Shinylive app: a `shinywidgets` `render_widget` returning an `ipyleaflet.Map`
with one `GeoJSON` layer + an `on_hover` handler. `shinylive export spike/ site/`,
serve `site/` with `python -m http.server`, open in a browser, confirm the Leaflet map
renders, the layer draws, and hover fires under Pyodide.
- **Pass** → continue with the full port below.
- **Fail** → fall back to a Pyodide + plain **Leaflet.js** page (Pyodide runs the
  SQL/query logic and passes GeoJSON to JS via the JS bridge), or `anywidget`. Same
  data pipeline; only the map layer differs. Decide with the user at that point.

## Architecture

```
web/
  app.py             # self-contained browser app (NO eurotraffic/polars imports)
  requirements.txt   # extra Pyodide packages: ipyleaflet, branca, shinywidgets
  data/              # generated: <city>.sqlite.gz per city + cities.json
scripts/build_web.sh # export data + shinylive export + copy data into site/
site/                # generated static output (gitignored) — deployable anywhere
```

### 1. Web data export + reduction — `src/eurotraffic/export_web.py` (new)

All shrinking happens **at build time** (no per-request work in the browser). Read
`data/traffic.sqlite`; for each city write a **self-contained slim SQLite**
`web/data/<city>.sqlite`, then gzip → `<city>.sqlite.gz`.

**Data-reduction preprocessing (the main size levers):**
- **Geometry simplification** — Douglas–Peucker (`shapely.simplify`, tol ≈ 0.00005°
  ≈ ~5 m, `preserve_topology=False`) collapses the many-vertex OSM linestrings to a
  few vertices each; visually identical at city zoom and the single biggest reduction
  (London's ~49 MB of geometry is dominated by vertex count).
- **Coordinate precision** — emit WKT at **5 decimals** (~1 m) via `shapely.set_precision`
  / `to_wkt(rounding_precision=5)`.
- **Column drop** — keep only what `app/app.py` reads at runtime:
  `streets(osm_type, class_rank, aadt, source, longitude, latitude, geometry)`;
  drop `street_id, country, lanes, maxspeed, oneway, length_m`.
- **Compact types** — `aadt` and `class_rank` stored as INTEGER; `source` as 0/1.
- **gzip** each file (sqlite text/blobs compress ~3–4×; servers also serve gzipped).
- Index `idx_rank (class_rank, aadt DESC)` (city implicit — one file per city).
- `class_diurnal(osm_type, hour, weight)` (copied whole; tiny) and
  `class_ceiling(osm_type, ceiling)` (this city's rows) so each file is self-sufficient.

Also emit `web/data/cities.json`: per city `{country, center_lat, center_lon,
n_streets, n_measured, model_r2, bbox:[w,s,e,n]}` (bbox from `MIN/MAX(longitude/
latitude)`), bundled into the app for the dropdown, centering, and the city-change
fallback bbox — no DB needed for metadata.

Reuses existing tables verbatim (`class_ceiling` from `build_db.compute_class_ceilings`,
`cities`). Expected sizes after simplification + column-drop + precision + gzip: most
cities ≲2 MB gz; London the heaviest but down to a few MB. Print a per-city size report.

### 2. Browser app — `web/app.py` (new, self-contained)

A trimmed copy of `app/app.py` adapted for the browser:
- **No `eurotraffic`/`polars` imports**; load `cities.json` at startup (bundled in the
  app dir, so present in the Pyodide FS) for `CITY_CHOICES`, centering, and `CITY_BBOX`.
- **Configurable data source** — resolve a base URL once at startup, in priority
  order: (1) `?data_base=<url>` query param, (2) injected `window.EUROTRAFFIC_DATA_BASE`
  (settable in the export's `index.html` or a tiny `config.js`), (3) default **`./data`**
  (co-located files when deployed locally/statically). The per-city URL is
  `f"{BASE}/{city}.sqlite.gz"`. This lets the same static app pull data from a local
  folder, a CDN, or object storage without rebuilding. `cities.json` is loaded from the
  same base.
- **On-demand data loading** (async): a `reactive.calc`/effect keyed on `input.city()`
  fetches `f"{BASE}/{city}.sqlite.gz"` via `pyodide.http.pyfetch` → `await resp.bytes()`
  → `gzip.decompress` → write to Pyodide MEMFS → `sqlite3.connect`. Cache one connection
  per city in a dict (each city fetched at most once per session); show a "loading…"
  state on first fetch.
- **Same query** as `app/app.py:_FRAME_SQL` minus the `city = :city` predicate (single-
  city DB), keeping the bbox filter + `class_rank, aadt` ordering + `class_diurnal`/
  `class_ceiling` joins. Reuse the relative-color and `class_ceiling` logic unchanged.
- **WKT → GeoJSON without shapely**: small helper parsing `LINESTRING (x y, …)` into
  `[[x,y],…]` (the data is all LineStrings); removes the shapely dependency.
- Keep the viewport-reactive `ipyleaflet` rendering, hover info box, hour slider, and
  play button exactly as in `app/app.py` (the `_redraw`/`_recenter`/`_tick` effects,
  `MAX_FEATURES` cap, `reactive_read(map.widget, "bounds")`).

### 3. Build script — `scripts/build_web.sh` (new)

1. `python -m eurotraffic.export_web` → regenerate `web/data/`.
2. `shinylive export web site` (shinylive bundles Pyodide + the app statically).
3. For **local/static deployment**, copy `web/data/` into `site/data/` (the default
   `./data` base resolves there). For a **remote base URL**, skip the copy and host
   `web/data/` separately (CDN/object storage) — the app points at it via
   `?data_base=` or `EUROTRAFFIC_DATA_BASE`. A `--data-base` flag controls this.
4. `web/requirements.txt` lists PyPI packages Pyodide must `micropip`-install
   (`ipyleaflet`, `branca`, `shinywidgets`); stdlib `sqlite3`/`gzip` and Pyodide-builtin
   packages need no listing.

### 4. Dependencies & docs

- Add `shinylive` to a `[project.optional-dependencies] web` group in `pyproject.toml`
  (build-time only).
- README: new "Standalone web app" section (build + `python -m http.server site`),
  noting it deploys to any static host (GitHub Pages, S3, etc.) and works offline once
  loaded.

## Implementation order

1. **Spike** (Step 0) — gate on ipyleaflet-in-Pyodide. STOP and consult if it fails.
2. `export_web.py` + generate `web/data/` (verify slim sizes, one self-contained file).
3. `web/app.py` against a local static server feeding `web/data/` (before shinylive,
   test the query + WKT parsing + async fetch logic with desktop Pyodide or a thin
   harness).
4. `scripts/build_web.sh` + `web/requirements.txt`; `shinylive export`; wire data copy.
5. README + pyproject updates.

## Verification

- **Spike**: ipyleaflet map renders + hover works in a browser from `shinylive export`.
- **Data**: `python -m eurotraffic.export_web` → assert each `<city>.sqlite.gz` opens,
  has all three tables, `hour` spans 0–23, and is materially smaller than the per-city
  slice of the monolith (per-city size report printed); `cities.json` has 39 entries
  with bboxes. Spot-check simplified geometry still looks right at city zoom (vertex
  count down, shape preserved).
- **Configurable source**: load the app with default `./data`, then with
  `?data_base=<other-host>/data` and confirm it fetches the city files from there.
- **End-to-end**: `scripts/build_web.sh`, then `python -m http.server` over `site/`,
  open in a browser:
  - dropdown lists 39 cities; selecting one fetches its file (visible in the Network
    tab) and renders the network;
  - pan/zoom re-queries the frame (zoom in reveals smaller streets);
  - hour slider + Play animate colors; hover shows "N veh/h — measured/estimated";
  - try a small city (Barcelona) and London (largest) for load-time + responsiveness.
- **Offline**: reload with cache disabled / serve from a second static host to confirm
  no server-side dependency.

## Risks / fallbacks

- **ipyleaflet under Pyodide** (primary) — mitigated by the Step-0 spike; fallback is
  Pyodide + plain Leaflet.js (same data path, JS-side rendering) or `anywidget`.
- **London download size** — if still heavy after slimming, optionally cap streets per
  city in `export_web` (top-K by `class_rank, aadt`) at the cost of deep-zoom detail.
- **Async loading UX** — first selection of a city blocks on fetch; surfaced via a
  loading indicator and per-city connection caching.
