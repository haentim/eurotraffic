"""EuroTraffic dashboard — pick a city, scrub the hour, see per-street traffic density.

The drivable street network is colored by estimated density. Each street has a base
AADT (measured where a sensor coincides, else predicted by a gradient-boosted model
from OSM road features) and an hourly shape `density = aadt × class_diurnal[type, hour]`.

Display logic: all streets in the current map frame are drawn; when a frame holds
too many, only the largest are kept — ranked by road-class size, ties broken by daily
volume (AADT). Panning/zooming re-queries the visible frame, so zooming in reveals
the smaller streets while zoomed-out views show the arterials.

Run with:  shiny run app/app.py   (DB built by `python -m eurotraffic.build_db`).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import branca.colormap as cm
from ipyleaflet import GeoJSON, Map, WidgetControl, basemaps
from ipywidgets import HTML
from shapely import from_wkt
from shapely.geometry import mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurotraffic import DB_PATH  # noqa: E402
from shiny import App, reactive, render, ui  # noqa: E402
from shinywidgets import output_widget, reactive_read, render_widget  # noqa: E402


# --------------------------------------------------------------------------- data
def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} not found. Build it first: python -m eurotraffic.build_db"
        )
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


_CON = _connect()

# Max streets drawn per frame. When the frame holds more, we keep the largest by
# (class_rank, aadt); when it holds fewer, all of them are shown.
MAX_FEATURES = 1500


def load_cities() -> list[dict]:
    rows = _CON.execute(
        "SELECT city, country, center_lat, center_lon, n_streets, n_measured, model_r2 "
        "FROM cities ORDER BY city"
    ).fetchall()
    return [dict(r) for r in rows]


def load_city_bbox() -> dict[str, tuple[float, float, float, float]]:
    rows = _CON.execute(
        "SELECT city, MIN(longitude), MIN(latitude), MAX(longitude), MAX(latitude) "
        "FROM streets GROUP BY city"
    ).fetchall()
    return {r[0]: (r[1], r[2], r[3], r[4]) for r in rows}


def load_class_ceiling() -> dict[tuple[str, str | None], float]:
    """Per-(city, road class) color ceiling, so the scale is *relative to street
    size*: a small street near its class's busy end is highlighted just like a
    motorway near its own — instead of staying blue under the city's motorway max.
    Ceilings are hour-independent (p95 AADT × class peak weight), so the time slider
    still varies colors. Built by ``build_db.compute_class_ceilings``."""
    rows = _CON.execute("SELECT city, osm_type, ceiling FROM class_ceiling").fetchall()
    return {(r[0], r[1]): (r[2] or 1.0) for r in rows}


# Largest streets in a frame: index walk by (class_rank, aadt) with a bbox filter,
# then the hourly class weight is applied to the capped set. See idx_streets_rank.
_FRAME_SQL = """
SELECT s.geometry, s.source, s.osm_type, s.aadt,
       s.aadt * COALESCE(cd.weight, dd.weight) AS density
FROM (
    SELECT geometry, source, osm_type, aadt, class_rank
    FROM streets
    WHERE city = :city
      AND longitude BETWEEN :w AND :e
      AND latitude  BETWEEN :s AND :n
    ORDER BY class_rank ASC, aadt DESC
    LIMIT :cap
) s
LEFT JOIN class_diurnal cd ON cd.osm_type = s.osm_type AND cd.hour = :h
LEFT JOIN class_diurnal dd ON dd.osm_type = '_default'  AND dd.hour = :h
ORDER BY s.class_rank ASC, s.aadt DESC
"""


def load_in_frame(city, hour, bbox) -> list[dict]:
    w, s, e, n = bbox
    rows = _CON.execute(
        _FRAME_SQL,
        {"city": city, "w": w, "e": e, "s": s, "n": n, "h": hour, "cap": MAX_FEATURES},
    ).fetchall()
    return [dict(r) for r in rows]


CITIES = load_cities()
CITY_BBOX = load_city_bbox()
CLASS_CEILING = load_class_ceiling()
CITY_CHOICES = {row["city"]: f"{row['city']} ({row['country']})" for row in CITIES}
CITY_META = {row["city"]: row for row in CITIES}

COLORS = ["#2c7bb6", "#abd9e9", "#ffff8c", "#fdae61", "#d7191c"]
# Colors map a 0..1 relative-congestion value; the per-street ceiling sets the scale.
NORM_CMAP = cm.LinearColormap(COLORS, vmin=0, vmax=1)


def frame_geojson(city: str, hour: int, bbox) -> tuple[dict, int]:
    """Build a GeoJSON FeatureCollection of the visible streets, colored relative to
    each street's road-class ceiling (so smaller streets light up at lower volumes)."""
    features = []
    for row in load_in_frame(city, hour, bbox):
        if not row["geometry"]:
            continue
        density = row["density"] or 0.0
        ceiling = CLASS_CEILING.get((city, row["osm_type"]), 1.0) or 1.0
        rel = min(density / ceiling, 1.0)
        # "measured" = AADT anchored to a real sensor (incl. counts derived from
        # non-vehicle-count measurements); otherwise model-estimated.
        kind = "measured" if row["source"] == "measured" else "estimated"
        features.append({
            "type": "Feature",
            "geometry": mapping(from_wkt(row["geometry"])),
            "properties": {"color": NORM_CMAP(rel), "veh": int(round(density)), "kind": kind},
        })
    return {"type": "FeatureCollection", "features": features}, len(features)


def _style(feature):
    return {
        "color": feature["properties"]["color"],
        "weight": 2.5,
        "opacity": 0.85,
    }


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
        ui.output_ui("city_note"),
        width=340,
    ),
    output_widget("map"),
    title="EuroTraffic — European city traffic density by time of day",
    fillable=True,
)


# --------------------------------------------------------------------------- server
def server(input, output, session):
    # Plain (non-reactive) holders: the current GeoJSON layer and the last city we
    # recentered to. Kept out of the reactive graph on purpose.
    state: dict = {"layer": None, "city": None, "draw_city": None, "info": None}
    playing = reactive.value(False)

    def _on_hover(**kwargs):
        feat = kwargs.get("feature")
        info = state["info"]
        if not feat or info is None:
            return
        p = feat.get("properties", {})
        info.value = (
            "<div style='padding:4px 8px;font:12px/1.3 sans-serif'>"
            f"<b>{p.get('veh', 0):,} veh/h</b><br>"
            f"<span style='color:#666'>{p.get('kind', '')}</span></div>"
        )

    @reactive.effect
    @reactive.event(input.play)
    def _toggle_play():
        now = not playing()
        playing.set(now)
        ui.update_action_button("play", label="⏸ Pause" if now else "▶ Play")

    @reactive.effect
    def _tick():
        """While playing, advance the hour every `delay` seconds, wrapping 23→0."""
        if not playing():
            return
        reactive.invalidate_later(max(float(input.delay() or 1.0), 0.2))
        with reactive.isolate():
            next_hour = (int(input.hour()) + 1) % 24
        ui.update_slider("hour", value=next_hour)

    @render_widget
    def map():
        # Build the map ONCE. Reading input.city() here without isolation would make
        # Shiny recreate the whole widget on every city switch, orphaning the layer
        # we track and crashing the session ("connection closed").
        with reactive.isolate():
            meta = CITY_META[input.city()]
        return Map(
            center=(meta["center_lat"], meta["center_lon"]),
            zoom=12,
            scroll_wheel_zoom=True,
            basemap=basemaps.CartoDB.Positron,
        )

    @reactive.effect
    def _recenter_on_city_change():
        """When the city changes, fly the map to it (bounds change then redraws)."""
        city = input.city()
        w = map.widget
        if w is None or state["city"] == city:
            return
        state["city"] = city
        meta = CITY_META[city]
        w.center = (meta["center_lat"], meta["center_lon"])
        w.zoom = 12

    @reactive.effect
    def _redraw():
        """Redraw streets whenever the frame, hour, or city changes."""
        w = map.widget
        if w is None:
            return
        city = input.city()
        hour = int(input.hour())
        bounds = reactive_read(w, "bounds")  # ((south, west), (north, east)) or ()
        # On a city switch the reported bounds still describe the previous city until
        # the map recenters, so draw the new city's full extent for that pass; once
        # the map has recentered, bounds update and we redraw to the actual viewport.
        city_changed = state["draw_city"] != city
        state["draw_city"] = city
        if bounds and not city_changed:
            (south, west), (north, east) = bounds
            bbox = (west, south, east, north)
        else:
            bbox = CITY_BBOX.get(city, (-180, -90, 180, 90))

        # Hover info box (top-right), created once.
        if state["info"] is None:
            state["info"] = HTML(
                "<div style='padding:4px 8px;font:12px sans-serif;color:#666'>"
                "Hover a street</div>"
            )
            w.add(WidgetControl(widget=state["info"], position="topright"))

        data, _ = frame_geojson(city, hour, bbox)
        new_layer = GeoJSON(data=data, style_callback=_style, hover_style={"weight": 6})
        new_layer.on_hover(_on_hover)
        old = state["layer"]
        w.add(new_layer)
        state["layer"] = new_layer
        if old is not None:
            try:
                w.remove(old)
            except Exception:
                pass

    @output
    @render.ui
    def legend():
        hour = int(input.hour())
        NORM_CMAP.caption = f"Congestion relative to road type — {hour:02d}:00 (low → high)"
        NORM_CMAP.width = 290  # fit inside the 340px sidebar
        return ui.HTML(NORM_CMAP._repr_html_())

    @output
    @render.ui
    def city_note():
        meta = CITY_META[input.city()]
        n_streets, n_meas = meta["n_streets"], meta["n_measured"]
        pct = (100 * n_meas / n_streets) if n_streets else 0
        return ui.tags.ul(
            ui.tags.li(
                f"{n_streets:,} streets; {n_meas:,} ({pct:.1f}%) anchored to measured "
                f"sensors, the rest predicted from OSM road features "
                f"(within-city R²={meta['model_r2']:.2f})."
            ),
            ui.tags.li(
                "All streets in view are shown; when too many, the largest are kept "
                f"(road-class size, then daily volume), up to {MAX_FEATURES:,}."
            ),
            ui.tags.li("Zoom in to reveal smaller streets."),
            ui.tags.li(
                "Colors are relative to each road class, so smaller streets light up "
                "at their own busy times."
            ),
            ui.tags.li("Hover a street for its vehicles/hour and measured-vs-estimated."),
            ui.tags.li("Press Play to animate through the day."),
            style="font-size:0.8em;color:#555;padding-left:1.1em;margin-top:0.4em",
        )


app = App(app_ui, server)
