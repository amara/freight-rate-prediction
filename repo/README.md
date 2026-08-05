# Freight Rate Prediction

Solution for the Spotter ML Engineer assessment: predict `posted_rate` for
12,000 Nov–Dec 2025 loads and a fixed Lexington → Fort Wayne Dry Van lane
across December 2025.

## Approach summary

- **Simple by design: 10 features**, every one computable from columns present
  in all three input files — distance, weight, equipment (3 dummies),
  month, day-of-week, and three route-history aggregates (median rate,
  median rate/mile, load count). Market features, coordinates, and raw city
  categoricals were tested and removed: they added noise, not accuracy.
- **Temporal validation split** (train Jan–Aug, hold out Sep–Oct), mirroring
  the real two-month forecast horizon. A random split would leak seasonal
  market conditions (rates swing $2.10–$2.33/mile across the year).
- **Model comparison on identical features** (Sep–Oct holdout MAE):
  Linear $279 · Decision Tree $148 · Random Forest $123 ·
  HistGradientBoosting $120 · **LightGBM $109.50 (4.10% MAPE, selected)** —
  vs $193.58 for the strongest non-ML baseline (route median price).
- **Data cleaning:** dropped 103 loads (~0.2%) priced below $0.50/mile on
  100+ mile hauls (label corruption); median-imputed ~300 missing weights.
- **December file needs no special handling** — all 10 features exist
  natively in its columns.

## Setup & run

```bash
python -m pip install -r requirements.txt

python src/validate.py        # baselines + 6-model comparison table
python src/train_predict.py   # writes validation_predictions.csv and fills
                              # data/december_chart_inputs.csv
python src/explain.py         # optional: SHAP plots

python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

## Layout

```
src/pipeline.py       cleaning + the 10-feature builder
src/validate.py       temporal split, baselines, model comparison
src/train_predict.py  final training + both prediction outputs
src/explain.py        SHAP attribution plots
data/                 provided inputs
```

Random seeds fixed; results reproducible.
