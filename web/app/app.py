"""EuroTraffic — standalone browser app (Pyodide / Shinylive).

Runs entirely client-side. City metadata is bundled (`cities.json`); each city's
slim SQLite is fetched on demand from a configurable base URL, decompressed, and
queried in-browser. Mirrors the server app's viewport rendering, per-road-class
relative colors, hover, hour slider, and play button — but with NO `ipywidgets.HTML`
(it fails under Shinylive); hover is shown via a Shiny output instead.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import branca.colormap as cm
from ipyleaflet import GeoJSON, Map, basemaps
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, reactive_read, render_widget

# --------------------------------------------------------------- config / metadata
# Bundled alongside app.py; open relative to this file (cwd is not the app dir).
_HERE = Path(__file__).resolve().parent
CITIES = json.loads((_HERE / "cities.json").read_text())["cities"]
CITY_META = {c["city"]: c for c in CITIES}
CITY_CHOICES = {c["city"]: f"{c['city']} ({c['country']})" for c in CITIES}

MAX_FEATURES = 1500
COLORS = ["#2c7bb6", "#abd9e9", "#ffff8c", "#fdae61", "#d7191c"]
NORM_CMAP = cm.LinearColormap(COLORS, vmin=0, vmax=1)

_FRAME_SQL = """
SELECT s.geometry, s.source, s.osm_type, s.aadt,
       s.aadt * COALESCE(cd.weight, dd.weight) AS density
FROM (
    SELECT geometry, source, osm_type, aadt, class_rank FROM streets
    WHERE longitude BETWEEN :w AND :e AND latitude BETWEEN :s AND :n
    ORDER BY class_rank ASC, aadt DESC LIMIT :cap
) s
LEFT JOIN class_diurnal cd ON cd.osm_type = s.osm_type AND cd.hour = :h
LEFT JOIN class_diurnal dd ON dd.osm_type = '_default'  AND dd.hour = :h
ORDER BY s.class_rank ASC, s.aadt DESC
"""


_BASE: str | None = None


def data_base() -> str | None:
    """Configured data base URL, or None to fall back to auto-detected candidates.

    Pyodide runs in a web worker (no `window`), so relative `./data` doesn't resolve
    to the site root. Configure an explicit base by bundling `config.json`
    (`{"data_base": "https://cdn/.../data"}`) — written by `build_web.sh <base>`.
    """
    global _BASE
    if _BASE is not None:
        return _BASE or None
    base = ""
    try:
        cfg = json.loads((_HERE / "config.json").read_text())
        base = (cfg.get("data_base") or "").rstrip("/")
    except Exception:
        pass
    _BASE = base
    return base or None


# Cache of opened per-city connections + their class ceilings.
_CONNS: dict[str, tuple[sqlite3.Connection, dict]] = {}


def _candidate_urls(fname: str) -> list[str]:
    """URLs to try, configured base first, then layouts that work for a co-located
    `data/` folder regardless of where the worker resolves relative paths from."""
    urls = []
    cfg = data_base()
    if cfg:
        urls.append(f"{cfg}/{fname}")
    try:
        import js

        href = str(js.self.location.href)
        print(f"[eurotraffic] worker location: {href}")
        # Anchor at the site root = everything before '/shinylive/'. Works for both
        # root-hosted and GitHub-Pages-style subpath (/repo/) deployments.
        if "/shinylive/" in href:
            root = href.split("/shinylive/")[0].rstrip("/")
            urls.append(f"{root}/data/{fname}")
        origin = str(js.self.location.origin)
        urls.append(f"{origin}/data/{fname}")
    except Exception as exc:  # noqa: BLE001
        print(f"[eurotraffic] location detect failed: {exc}")
    urls += [f"/data/{fname}", f"./data/{fname}", f"../data/{fname}", f"data/{fname}"]
    return urls


async def get_city(city: str) -> tuple[sqlite3.Connection, dict]:
    """Fetch + open a city's SQLite (cached). Writes to MEMFS (no deserialize dep)."""
    if city in _CONNS:
        return _CONNS[city]
    from pyodide.http import pyfetch

    fname = CITY_META[city]["file"]
    last = "no attempt"
    for url in _candidate_urls(fname):
        try:
            resp = await pyfetch(url)
            if resp.status != 200:
                last = f"{url} -> HTTP {resp.status}"
                continue
            raw = await resp.bytes()
            path = f"/tmp/{fname.replace('/', '_')}.sqlite"
            with open(path, "wb") as fh:
                fh.write(gzip.decompress(raw))
            con = sqlite3.connect(path)
            ceilings = {ot: (cl or 1.0) for ot, cl in con.execute(
                "SELECT osm_type, ceiling FROM class_ceiling")}
            _CONNS[city] = (con, ceilings)
            print(f"[eurotraffic] loaded {fname} from {url}")
            return _CONNS[city]
        except Exception as exc:  # noqa: BLE001
            last = f"{url} -> {exc}"
    raise RuntimeError(f"could not load {fname} (last: {last})")


def _line_coords(wkt: str) -> list[list[float]]:
    """Parse 'LINESTRING (x y, x y, …)' → [[lon,lat], …] (GeoJSON order)."""
    inner = wkt[wkt.index("(") + 1 : wkt.rindex(")")]
    out = []
    for pair in inner.split(","):
        x, y = pair.split()[:2]
        out.append([float(x), float(y)])
    return out


def _features(con, ceilings, hour: int, bbox) -> dict:
    w, s, e, n = bbox
    rows = con.execute(
        _FRAME_SQL,
        {"w": w, "e": e, "s": s, "n": n, "h": hour, "cap": MAX_FEATURES},
    ).fetchall()
    feats = []
    for geometry, source, osm_type, aadt, density in rows:
        if not geometry:
            continue
        density = density or 0.0
        rel = min(density / (ceilings.get(osm_type, 1.0) or 1.0), 1.0)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": _line_coords(geometry)},
            "properties": {
                "color": NORM_CMAP(rel),
                "veh": int(round(density)),
                "kind": "measured" if source == 1 else "estimated",
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def _style(feature):
    return {"color": feature["properties"]["color"], "weight": 2.5, "opacity": 0.85}


# ------------------------------------------------------------------------------- ui
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("city", "City", CITY_CHOICES, selected="Berlin"),
        ui.input_slider("hour", "Time of day (hour)", min=0, max=23, value=8, step=1),
        ui.div(
            ui.input_action_button("play", "▶ Play", class_="btn-sm btn-primary"),
            ui.input_numeric("delay", "Seconds per hour", value=2.5, min=0.2, max=10, step=0.5),
            style="display:flex; gap:0.6em; align-items:flex-end; margin-bottom:0.6em",
        ),
        ui.output_ui("legend"),
        ui.output_text("status"),
        ui.output_ui("hover_info"),
        ui.p(
            "Tip: zoom in to reveal smaller roads — only the largest streets are "
            "shown when the view is too crowded.",
            style="font-size:0.8em; color:#777; margin-top:0.4em",
        ),
        width=340,
    ),
    output_widget("map"),
    title="EuroTraffic — European city traffic density by time of day",
    fillable=True,
)


# --------------------------------------------------------------------------- server
def server(input, output, session):
    state: dict = {"layer": None, "draw_city": None}
    playing = reactive.value(False)
    status_msg = reactive.value("")
    hover_msg = reactive.value("Hover a street for details.")

    def _on_hover(**kwargs):
        feat = kwargs.get("feature")
        if feat:
            p = feat["properties"]
            hover_msg.set(f"{p['veh']:,} veh/h — {p['kind']}")

    @reactive.effect
    @reactive.event(input.play)
    def _toggle_play():
        now = not playing()
        playing.set(now)
        ui.update_action_button("play", label="⏸ Pause" if now else "▶ Play")

    @reactive.effect
    def _tick():
        if not playing():
            return
        reactive.invalidate_later(max(float(input.delay() or 1.0), 0.2))
        with reactive.isolate():
            nxt = (int(input.hour()) + 1) % 24
        ui.update_slider("hour", value=nxt)

    @render_widget
    def map():
        with reactive.isolate():
            meta = CITY_META[input.city()]
        return Map(
            center=(meta["center_lat"], meta["center_lon"]),
            zoom=12, scroll_wheel_zoom=True, basemap=basemaps.CartoDB.Positron,
        )

    @reactive.calc
    async def city_conn():
        city = input.city()
        status_msg.set(f"Loading {city}…")
        try:
            con, ceilings = await get_city(city)
            status_msg.set("")
            return city, con, ceilings
        except Exception as exc:  # noqa: BLE001
            status_msg.set(f"Failed to load {city}: {exc}")
            raise

    @reactive.effect
    def _recenter():
        city = input.city()
        w = map.widget
        if w is None:
            return
        meta = CITY_META[city]
        w.center = (meta["center_lat"], meta["center_lon"])
        w.zoom = 12

    @reactive.effect
    async def _redraw():
        w = map.widget
        if w is None:
            return
        hour = int(input.hour())
        bounds = reactive_read(w, "bounds")
        try:
            city, con, ceilings = await city_conn()
        except Exception:
            return

        city_changed = state["draw_city"] != city
        state["draw_city"] = city
        if bounds and not city_changed:
            (south, west), (north, east) = bounds
            bbox = (west, south, east, north)
        else:
            bbox = tuple(CITY_META[city]["bbox"])

        data = _features(con, ceilings, hour, bbox)
        layer = GeoJSON(data=data, style_callback=_style, hover_style={"weight": 6})
        layer.on_hover(_on_hover)
        old = state["layer"]
        w.add(layer)
        state["layer"] = layer
        if old is not None:
            try:
                w.remove(old)
            except Exception:
                pass

    @render.text
    def status():
        return status_msg()

    @render.ui
    def hover_info():
        return ui.p(hover_msg(), style="font-weight:600;margin-top:0.3em")

    @render.ui
    def legend():
        hour = int(input.hour())
        NORM_CMAP.caption = f"Congestion relative to road type — {hour:02d}:00 (low → high)"
        NORM_CMAP.width = 290
        return ui.HTML(NORM_CMAP._repr_html_())


app = App(app_ui, server)
