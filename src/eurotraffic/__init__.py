"""EuroTraffic: harmonize European city traffic data into an hourly density profile."""

from pathlib import Path

# Repository / data locations resolved relative to this file so the package works
# regardless of the current working directory.
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATASET_ROOT = PROJECT_ROOT / "traffic-volume-data-EU-cities"
DB_PATH = PROJECT_ROOT / "data" / "traffic.sqlite"

__all__ = ["PACKAGE_ROOT", "PROJECT_ROOT", "DATASET_ROOT", "DB_PATH"]
