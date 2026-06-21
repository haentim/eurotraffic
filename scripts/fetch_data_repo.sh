#!/usr/bin/env bash
# Clone (or update) the source dataset repository this project is built on.
# The dataset is NOT vendored into this repo; fetch it here before running the
# preprocessing pipeline (eurotraffic.model / eurotraffic.build_db).
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="traffic-volume-data-EU-cities"
REPO="https://github.com/XavB64/traffic-volume-data-EU-cities"

if [ -d "$DIR/.git" ]; then
  echo "==> $DIR already present; pulling latest"
  git -C "$DIR" pull --ff-only
else
  echo "==> Cloning $REPO into $DIR"
  git clone --depth 1 "$REPO" "$DIR"
fi
echo "Done."
