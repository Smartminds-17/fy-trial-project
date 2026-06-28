"""Export trained ML models for the FastAPI dashboard.

The working notebook reference is ``tz_economic_predictor.ipynb``. This script
keeps the live app reproducible by implementing the same dashboard-ready feature
engineering in plain Python, then saving the dashboard model files as:
- infl_rf.joblib
- fx_rf.joblib

Run:
  python export_models.py

Then run your predictor:
  python predict_realtime.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from project_config import FX_MODEL_PATH, INFL_MODEL_PATH, INPUT_CSV, MAIN_NOTEBOOK
from predict_realtime import (
    FX_FEATURES,
    INFL_FEATURES,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
    compute_features_like_notebook,
    train_test_payload_dates,
)


CSV_PATH = INPUT_CSV


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing training data CSV: {CSV_PATH}")
    if not MAIN_NOTEBOOK.exists():
        raise FileNotFoundError(f"Main notebook reference is missing: {MAIN_NOTEBOOK.name}")

    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["month"], format="%b-%y")
    df = df.sort_values("date").reset_index(drop=True)
    df = compute_features_like_notebook(df)
    df["inflation_next"] = df["inflation_pct"].shift(-1)
    df["usd_tzs_change_next"] = df["usd_tzs_change"].shift(-1)

    model_df = df.dropna(
        subset=INFL_FEATURES + FX_FEATURES + ["inflation_next", "usd_tzs_change_next"]
    ).reset_index(drop=True)
    model_df["target_date"] = model_df["date"] + pd.offsets.MonthBegin(1)
    train_df, test_df, live_df = train_test_payload_dates(model_df)

    if train_df.empty:
        raise ValueError(
            f"No training rows found for {TRAIN_START_DATE:%b-%y} through {TRAIN_END_DATE:%b-%y}."
        )

    X_infl_train = train_df[INFL_FEATURES]
    y_infl_train = train_df["inflation_next"]
    X_infl_test = test_df[INFL_FEATURES]
    y_infl_test = test_df["inflation_next"]

    X_fx_train = train_df[FX_FEATURES]
    y_fx_train = train_df["usd_tzs_change_next"]
    X_fx_test = test_df[FX_FEATURES]
    y_fx_test = test_df["usd_tzs_change_next"]

    infl_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42,
    )
    infl_model.fit(X_infl_train, y_infl_train)

    fx_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42,
    )
    fx_model.fit(X_fx_train, y_fx_train)

    # Optional sanity metrics
    print(f"Modeling target months: {TRAIN_START_DATE:%b-%y} through {TRAIN_END_DATE:%b-%y}")
    print(f"Training rows: {len(train_df)}")
    print("Testing target months: last 20% inside the 2018-2024 modeling window")
    print(f"Testing rows: {len(test_df)}")
    print(f"Live collected rows held out from metrics: {len(live_df)}")

    if len(test_df):
        infl_preds = infl_model.predict(X_infl_test)
        infl_rmse = np.sqrt(mean_squared_error(y_infl_test, infl_preds))
        infl_mae = mean_absolute_error(y_infl_test, infl_preds)
        infl_r2 = r2_score(y_infl_test, infl_preds)

        fx_preds_change = fx_model.predict(X_fx_test)
        fx_base_level = test_df["usd_tzs_rate"].values
        fx_preds_level = fx_base_level + fx_preds_change
        fx_actual_next_level = fx_base_level + y_fx_test.values
        fx_rmse = np.sqrt(mean_squared_error(fx_actual_next_level, fx_preds_level))
        fx_mae = mean_absolute_error(fx_actual_next_level, fx_preds_level)
        fx_r2 = r2_score(fx_actual_next_level, fx_preds_level)

        print(f"Inflation model test: RMSE={infl_rmse:.3f} MAE={infl_mae:.3f} R2={infl_r2:.3f}")
        print(f"FX model test: RMSE={fx_rmse:.2f} MAE={fx_mae:.2f} R2={fx_r2:.3f}")
    else:
        print("No 2018-2024 test rows are available yet.")

    joblib.dump(infl_model, INFL_MODEL_PATH)
    joblib.dump(fx_model, FX_MODEL_PATH)

    print("Exported models:")
    print("- infl_rf.joblib")
    print("- fx_rf.joblib")


if __name__ == "__main__":
    main()
