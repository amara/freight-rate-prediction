"""SHAP feature attribution for the selected model. Target is log-rate, so a
mean |SHAP| of 0.05 moves predictions ~5%."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import shap

from pipeline import FEATURES, TARGET, add_features, build_lane_stats, load_dev

SPLIT_DATE = "2025-09-01"
PARAMS = dict(objective="regression_l1", n_estimators=1500, learning_rate=0.03,
              num_leaves=63, min_child_samples=30, random_state=42, verbose=-1)


def main() -> None:
    dev = load_dev()
    train = dev[dev["date"] < SPLIT_DATE]
    test = dev[dev["date"] >= SPLIT_DATE]
    lane_stats = build_lane_stats(train)
    tr = add_features(train, lane_stats, train)
    te = add_features(test, lane_stats, train)

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(tr[FEATURES], np.log(train[TARGET]))

    sample = te[FEATURES].sample(2000, random_state=42)
    shap_values = shap.TreeExplainer(model).shap_values(sample)

    shap.summary_plot(shap_values, sample, plot_type="bar", max_display=10, show=False)
    plt.tight_layout(); plt.savefig("shap_bar.png", dpi=150); plt.close()
    shap.summary_plot(shap_values, sample, max_display=10, show=False)
    plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=150); plt.close()
    print("Wrote shap_bar.png and shap_beeswarm.png")


if __name__ == "__main__":
    main()
