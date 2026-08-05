"""Internal validation: temporal split, baselines, and model comparison.

The real task is predicting Nov-Dec 2025 from Jan-Oct history, so the split
mirrors it: train Jan-Aug, hold out Sep-Oct. A random split would leak future
market conditions (rates swing seasonally: $2.10/mile in January to $2.33 in
June) and overstate accuracy.

Reproduces the model comparison table in the report.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from pipeline import FEATURES, TARGET, add_features, build_lane_stats, load_dev

SPLIT_DATE = "2025-09-01"

MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree (depth 8)": DecisionTreeRegressor(
        max_depth=8, min_samples_leaf=30, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "HistGradientBoosting (sklearn)": HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, random_state=42),
    "LightGBM (selected)": lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=1500, learning_rate=0.03,
        num_leaves=63, min_child_samples=30, random_state=42, verbose=-1),
}


def main() -> None:
    dev = load_dev()
    train = dev[dev["date"] < SPLIT_DATE].copy()
    test = dev[dev["date"] >= SPLIT_DATE].copy()
    y_te = test[TARGET]
    print(f"train {len(train):,} (Jan-Aug) | test {len(test):,} (Sep-Oct)\n")

    # ---- Baselines: the bar any model must clear -------------------------
    print(f"{'Approach':34s} {'MAE':>9s} {'MAPE':>8s}")
    naive = test["quote_signal"] * test["distance"]
    print(f"{'Naive quote (quote_signal x dist)':34s} ${mean_absolute_error(y_te, naive):8.2f} "
          f"{mean_absolute_percentage_error(y_te, naive)*100:7.2f}%")

    lane_med = train.groupby(["pickup", "delivery"])[TARGET].median().rename("lane_med")
    t = test.merge(lane_med, on=["pickup", "delivery"], how="left")
    t["lane_med"] = t["lane_med"].fillna(train["rpm"].median() * t["distance"])
    print(f"{'Lane median rate':34s} ${mean_absolute_error(y_te, t.lane_med):8.2f} "
          f"{mean_absolute_percentage_error(y_te, t.lane_med)*100:7.2f}%")

    # ---- Model comparison on the same 10 features ------------------------
    lane_stats = build_lane_stats(train)
    tr = add_features(train, lane_stats, train)
    te = add_features(test, lane_stats, train)
    X_tr, X_te = tr[FEATURES], te[FEATURES]
    y_tr_log = np.log(train[TARGET])

    print()
    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(X_tr, y_tr_log)
        pred = np.exp(model.predict(X_te))
        mae = mean_absolute_error(y_te, pred)
        mape = mean_absolute_percentage_error(y_te, pred) * 100
        print(f"{name:34s} ${mae:8.2f} {mape:7.2f}%   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
