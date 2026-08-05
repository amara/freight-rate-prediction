"""Train the final model on all cleaned development data and write both outputs:

1. validation_predictions.csv       (12,000 Nov-Dec loads)
2. data/december_chart_inputs.csv   (predicted_rate filled, 7 columns preserved)

Every feature is computable from columns present in both prediction files,
so no imputation of market features or coordinates is needed anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from pipeline import FEATURES, TARGET, add_features, build_lane_stats, load_dev

PARAMS = dict(objective="regression_l1", n_estimators=1500, learning_rate=0.03,
              num_leaves=63, min_child_samples=30, random_state=42, verbose=-1)


def main() -> None:
    dev = load_dev()
    lane_stats = build_lane_stats(dev)

    full = add_features(dev, lane_stats, dev)
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(full[FEATURES], np.log(full[TARGET]))

    # ---- Validation predictions ------------------------------------------
    val = pd.read_csv("data/validation.csv", parse_dates=["date"])
    val_f = add_features(val, lane_stats, dev)
    val_pred = np.exp(model.predict(val_f[FEATURES]))

    template = pd.read_csv("data/validation_predictions_template.csv")
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": val["load_id"], "predicted_rate": val_pred}),
        on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv("validation_predictions.csv", index=False)
    print(f"validation_predictions.csv written ({len(out):,} rows)")
    print(out["predicted_rate"].describe().round(2).to_string(), "\n")

    # ---- December fixed-lane predictions ---------------------------------
    dec = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])
    dec_f = add_features(dec, lane_stats, dev)
    dec_pred = np.exp(model.predict(dec_f[FEATURES]))

    dec_out = pd.read_csv("data/december_chart_inputs.csv")  # preserve 7 columns
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv("data/december_chart_inputs.csv", index=False)
    print(f"December predictions: {dec_pred.min():.0f} - {dec_pred.max():.0f} "
          f"| mean {dec_pred.mean():.0f}")

    lane = dev[(dev.pickup == "Lexington") & (dev.delivery == "Fort Wayne")]
    print(f"Lane history sanity check: {len(lane)} loads, median ${lane[TARGET].median():.0f}")


if __name__ == "__main__":
    main()
