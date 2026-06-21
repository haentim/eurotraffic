#!/usr/bin/env bash
# Build the standalone Pyodide/Shinylive site into ./site
#
#   scripts/build_web.sh             # data co-located in site/data (auto-detected base)
#   scripts/build_web.sh <base-url>  # host web/data separately; app reads from <base-url>
#
# The app auto-detects co-located data at <site>/data for both root- and
# subpath-hosted (GitHub Pages project) sites, so the default needs no base URL.
# Pass a <base-url> only to host the per-city files on a separate origin/CDN.
#
# Interpreter/tool can be overridden for CI:  PYTHON=python SHINYLIVE=shinylive scripts/build_web.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}";        [ -x "$PY" ] || PY=python
SHINYLIVE="${SHINYLIVE:-.venv/bin/shinylive}"; command -v "$SHINYLIVE" >/dev/null 2>&1 || SHINYLIVE=shinylive
DATA_BASE="${1:-local}"

# Regenerate per-city data only when the source DB is present (local dev). In CI the
# committed web/data/ is used as-is (the 660 MB monolith isn't checked in).
if [ -f data/traffic.sqlite ]; then
  echo "==> Exporting per-city web data"
  PYTHONPATH=src "$PY" -m eurotraffic.export_web
else
  echo "==> data/traffic.sqlite not found; using committed web/data/ (CI mode)"
  [ -f web/data/cities.json ] || { echo "ERROR: web/data/ missing; run export locally first"; exit 1; }
fi

echo "==> Bundling cities.json into the app"
cp web/data/cities.json web/app/cities.json

if [ "$DATA_BASE" = "local" ]; then
  echo '{"data_base": ""}' > web/app/config.json
else
  printf '{"data_base": "%s"}\n' "$DATA_BASE" > web/app/config.json
fi

echo "==> shinylive export -> site/"
"$SHINYLIVE" export web/app site

if [ "$DATA_BASE" = "local" ]; then
  echo "==> Copying per-city data into site/data"
  mkdir -p site/data
  cp web/data/*.sqlite.gz site/data/
  echo "Done. Serve from the site ROOT, e.g.:  $PY -m http.server --directory site 8000"
else
  echo "==> Remote base configured: $DATA_BASE (NOT copying data into site/)"
  echo "    Deploy web/data/*.sqlite.gz to: $DATA_BASE"
fi
