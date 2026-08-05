"""Data loading, cleaning, and feature engineering.

Design principle: keep it simple. Ten features, every one of them computable
from columns present in ALL three input files (pickup, delivery, distance,
equipment, weight, date). Market features (market_index, quote_signal),
coordinates, and raw city categoricals were tested and removed: they added
noise and mild overfitting rather than accuracy (see validate.py), and
dropping them means the December file needs no special handling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "posted_rate"

FEATURES = [
    "distance",          # #1 cost driver: more miles, more money
    "weight",            # heavier loads price slightly higher
    "month",             # yearly seasonality (rates peak May-June, soften in fall)
    "day_of_week",       # weekly rhythm (midweek peaks, weekend dips)
    "lane_rate_median",  # what this route has typically sold for
    "lane_rpm_median",   # same, per mile (transfers across load variations)
    "lane_count",        # how many loads back that estimate (trustworthiness)
    "equipment_Dry Van", "equipment_Flatbed", "equipment_Reefer",
]


def load_dev(path: str = "data/train_test.csv") -> pd.DataFrame:
    """Load labeled data; drop 103 corrupted labels (<$0.50/mile on 100+ mile
    hauls -- below operating cost, contradicted by their own quote_signal)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df["rpm"] = df[TARGET] / df["distance"]
    bad = (df["rpm"] < 0.5) & (df["distance"] > 100)
    return df.loc[~bad].copy()


def build_lane_stats(history: pd.DataFrame) -> pd.DataFrame:
    """Per-route price history. Fit on labeled TRAINING data only, so a
    holdout set never contributes to its own features."""
    return (
        history.groupby(["pickup", "delivery"])
        .agg(
            lane_rate_median=(TARGET, "median"),
            lane_rpm_median=("rpm", "median"),
            lane_count=(TARGET, "size"),
        )
        .reset_index()
    )


def add_features(frame: pd.DataFrame, lane_stats: pd.DataFrame,
                 history: pd.DataFrame) -> pd.DataFrame:
    """Build the 10 model inputs. `history` supplies fallback values
    (global medians) so unseen lanes and missing weights are handled."""
    out = frame.merge(lane_stats, on=["pickup", "delivery"], how="left")
    global_rpm = history["rpm"].median()
    out["lane_count"] = out["lane_count"].fillna(0)
    out["lane_rate_median"] = out["lane_rate_median"].fillna(global_rpm * out["distance"])
    out["lane_rpm_median"] = out["lane_rpm_median"].fillna(global_rpm)
    out["weight"] = out["weight"].fillna(history["weight"].median())
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out = pd.get_dummies(out, columns=["equipment"])
    for col in ("equipment_Dry Van", "equipment_Flatbed", "equipment_Reefer"):
        if col not in out.columns:  # a file might contain only one equipment type
            out[col] = False
    return out
