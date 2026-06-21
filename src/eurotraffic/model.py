"""Train a gradient-boosted model that predicts street traffic volume (AADT).

Training data is every treated GeoJSON feature that carries an ``AADT`` label and
OSM road features. The model predicts ``log1p(AADT)`` from the cleaned features in
``features.py``. Run with: ``python -m eurotraffic.model``.

The fitted estimator is persisted to ``data/model.joblib`` and consumed by
``build_db`` to score full street networks.
"""

from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import shape
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from . import DATASET_ROOT, PROJECT_ROOT
from .features import CATEGORICAL_COLS, FEATURE_COLS, feature_record

MODEL_PATH = PROJECT_ROOT / "data" / "model.joblib"


def _iter_training_rows():
    for treated in sorted(DATASET_ROOT.glob("*/*/treated")):
        city = treated.parent.name
        country = treated.parent.parent.name
        # One file per city is enough; pick the most recent by year in the name.
        files = sorted(treated.glob("*.geojson"))
        if not files:
            continue
        path = max(files, key=lambda p: max([int(y) for y in re.findall(r"\d{4}", p.stem)] or [0]))
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for feat in data.get("features", []):
            props = feat.get("properties") or {}
            geom = feat.get("geometry")
            aadt = props.get("AADT") or props.get("AAWT")
            if geom is None or aadt in (None, "", 0):
                continue
            try:
                aadt = float(aadt)
            except (TypeError, ValueError):
                continue
            if aadt <= 0:
                continue
            pt = shape(geom).representative_point()
            rec = feature_record(
                osm_type=props.get("osm_type"),
                country=country,
                lanes=props.get("osm_lanes"),
                maxspeed=props.get("osm_maxspeed"),
                oneway=props.get("osm_oneway"),
                latitude=pt.y,
                longitude=pt.x,
            )
            rec["aadt"] = aadt
            rec["city"] = city
            yield rec


def build_training_frame() -> pd.DataFrame:
    df = pd.DataFrame(_iter_training_rows())
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def make_estimator() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=500,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=CATEGORICAL_COLS,
        early_stopping=True,
        random_state=0,
    )


def _cv(X, y, splitter, groups=None) -> tuple[float, float]:
    r2s, maes = [], []
    for tr, te in splitter.split(X, y, groups):
        est = make_estimator()
        est.fit(X.iloc[tr], y[tr])
        pred = est.predict(X.iloc[te])
        r2s.append(r2_score(y[te], pred))
        maes.append(mean_absolute_error(np.expm1(y[te]), np.expm1(pred)))
    return float(np.mean(r2s)), float(np.mean(maes))


def _evaluate(df: pd.DataFrame) -> float:
    """Report two CV regimes and return the within-city (random) R².

    * grouped-by-city: transfer to a *brand-new* city (pessimistic; we never
      actually do this since each scored city has its own measured anchors).
    * random KFold: within-city prediction — the realistic scenario, because at
      score time we calibrate each city to its own measured sensors.
    """
    X = df[FEATURE_COLS]
    y = np.log1p(df["aadt"].to_numpy())

    n = min(5, df["city"].nunique())
    g_r2, g_mae = _cv(X, y, GroupKFold(n_splits=n), df["city"])
    r_r2, r_mae = _cv(X, y, KFold(n_splits=5, shuffle=True, random_state=0))

    print("  cross-validation:")
    print(f"    grouped-by-city  log-R² {g_r2:6.3f}   AADT MAE {g_mae:>8,.0f}/day  (unseen city)")
    print(f"    random KFold     log-R² {r_r2:6.3f}   AADT MAE {r_mae:>8,.0f}/day  (within-city)")
    return r_r2


def train_and_save() -> float:
    print("Building training frame from treated GeoJSONs...")
    df = build_training_frame()
    print(f"  {len(df):,} labeled segments across {df['city'].nunique()} cities")

    r2 = _evaluate(df)

    print("Fitting final model on all data...")
    est = make_estimator()
    est.fit(df[FEATURE_COLS], np.log1p(df["aadt"].to_numpy()))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": est, "cv_r2": r2, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH} (cv_r2={r2:.3f})")
    return r2


if __name__ == "__main__":
    train_and_save()
