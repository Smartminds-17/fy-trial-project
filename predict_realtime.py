"""Generate next-month predictions for the dashboard from your real/latest data.

Expected input:
- A CSV with the raw monthly columns used in the main notebook
  (tz_economic_predictor.ipynb):
  - month, inflation_pct, usd_tzs_rate, brent_oil_usd, m2_bn_tzs,
    bot_policy_rate_pct

This script is designed to be easy to adapt:
- If your CSV already contains the engineered columns, set ASSUME_FEATURES=True.
- Otherwise, set ASSUME_FEATURES=False and it will recompute the dashboard
  features used by the main notebook workflow.

Outputs:
- predictions.json in the project root.

It also produces historical series so the dashboard can render year-by-year graphs.

Inflation series are rolling one-step-ahead predictions:
- for each month t (features from t), predict inflation_next = inflation_pct[t+1]

FX series are also rolling one-step-ahead predictions:
- for each month t (features from t), predict usd_tzs_change_next
- then convert to USD/TZS LEVEL for t+1 using usd_tzs_rate[t]

Rows whose prediction target is in 2025 or later are treated as live collected
inputs for forecasting, not as model test rows.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from project_config import (
    FX_MODEL_PATH,
    INFL_MODEL_PATH,
    INPUT_CSV,
    PREDICTIONS_JSON,
    PROJECT_ROOT,
)

# ---- Configure these ----
ASSUME_FEATURES = False
MODEL_START_DATE = pd.Timestamp("2018-01-01")
MODEL_END_DATE = pd.Timestamp("2024-12-01")
LIVE_START_DATE = MODEL_END_DATE + pd.offsets.MonthBegin(1)
MODEL_TRAIN_FRACTION = 0.8

# Backward-compatible names used by the app payload and export script.
TRAIN_START_DATE = MODEL_START_DATE
TRAIN_END_DATE = MODEL_END_DATE
TEST_START_DATE = LIVE_START_DATE

MODEL_INFL_PATH = INFL_MODEL_PATH
MODEL_FX_PATH = FX_MODEL_PATH

OUT_JSON = PREDICTIONS_JSON
# --------------------------


INFL_FEATURES = [
    "inflation_pct",
    "inflation_lag1",
    "brent_oil_usd",
    "m2_growth",
]

FX_FEATURES = [
    "usd_tzs_change",
    "usd_tzs_change_lag1",
    "brent_oil_usd",
    "m2_growth",
    "bot_policy_rate_pct",
]


def compute_features_like_notebook(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["usd_tzs_change"] = out["usd_tzs_rate"].diff()
    out["inflation_lag1"] = out["inflation_pct"].shift(1)
    out["usd_tzs_change_lag1"] = out["usd_tzs_change"].shift(1)
    out["brent_oil_lag1"] = out["brent_oil_usd"].shift(1)
    out["m2_growth"] = out["m2_bn_tzs"].pct_change()

    return out


def load_source_dataset() -> pd.DataFrame:
    """Load and date-sort the source CSV without changing its recorded values."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if "month" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["month"], format="%b-%y")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("CSV must contain 'month' (e.g. Jan-18) or 'date' column")

    return df.sort_values("date").reset_index(drop=True)


def load_models():
    missing = [path.name for path in (MODEL_INFL_PATH, MODEL_FX_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model file(s): {', '.join(missing)}. Run 'python export_models.py' first."
        )
    return joblib.load(MODEL_INFL_PATH), joblib.load(MODEL_FX_PATH)


def dataset_payload(offset: int = 0, limit: int = 100) -> dict:
    """Return a JSON-safe page of the source dataset for the web table."""
    source = pd.read_csv(INPUT_CSV)
    page = source.iloc[offset : offset + limit]
    rows = json.loads(page.to_json(orient="records"))
    return {
        "source": INPUT_CSV.name,
        "columns": list(source.columns),
        "total_rows": int(len(source)),
        "offset": offset,
        "limit": limit,
        "rows": rows,
    }


def train_test_payload_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by target month.

    Only 2018-2024 target months are used for model train/test evaluation.
    Later rows remain available as live collected data for prediction inputs.
    """
    with_target_dates = df.copy()
    with_target_dates["target_date"] = with_target_dates["date"] + pd.offsets.MonthBegin(1)

    model_df = with_target_dates[
        (with_target_dates["target_date"] >= TRAIN_START_DATE)
        & (with_target_dates["target_date"] <= TRAIN_END_DATE)
    ].reset_index(drop=True)

    split = int(len(model_df) * MODEL_TRAIN_FRACTION)
    split = min(max(split, 1), max(len(model_df) - 1, 1))

    train_df = model_df.iloc[:split].reset_index(drop=True)
    test_df = model_df.iloc[split:].reset_index(drop=True)
    live_df = with_target_dates[with_target_dates["target_date"] >= LIVE_START_DATE].reset_index(drop=True)

    return train_df, test_df, live_df


def _target_range_label(df: pd.DataFrame) -> str | None:
    if df.empty or "target_date" not in df.columns:
        return None
    first = pd.to_datetime(df["target_date"].iloc[0]).strftime("%b-%y")
    last = pd.to_datetime(df["target_date"].iloc[-1]).strftime("%b-%y")
    return first if first == last else f"{first} to {last}"


def _target_boundary_labels(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df.empty or "target_date" not in df.columns:
        return None, None
    first = pd.to_datetime(df["target_date"].iloc[0]).strftime("%b-%y")
    last = pd.to_datetime(df["target_date"].iloc[-1]).strftime("%b-%y")
    return first, last


def train_test_metadata(train_df: pd.DataFrame, test_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    train_start, train_end = _target_boundary_labels(train_df)
    test_start, test_end = _target_boundary_labels(test_df)
    return {
        "model_start": MODEL_START_DATE.strftime("%b-%y"),
        "model_end": MODEL_END_DATE.strftime("%b-%y"),
        "live_start": LIVE_START_DATE.strftime("%b-%y"),
        "train_start": train_start,
        "train_end": train_end,
        "train_period": _target_range_label(train_df),
        "test_start": test_start,
        "test_end": test_end,
        "test_period": _target_range_label(test_df),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "live_rows": int(len(live_df)),
        "split_rule": "Model train/test metrics use only 2018-2024 target months; 2025 onward is live collected data used for prediction inputs.",
    }


def display_model_name(model) -> str:
    names = {
        "GradientBoostingRegressor": "Gradient Boosting",
        "RandomForestRegressor": "Random Forest",
        "LinearRegression": "Linear Regression",
    }
    return names.get(type(model).__name__, type(model).__name__)


def feature_importance_payload(model, features: list[str]) -> tuple[list[dict], str]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        importance_type = "tree_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_).astype(float))
        importance_type = "absolute_coefficient"
    else:
        return [], "unavailable"

    total = float(values.sum())
    if total <= 0:
        normalized = np.zeros_like(values)
    else:
        normalized = values / total

    items = [
        {
            "name": feature,
            "value": float(value),
            "pct": float(pct * 100),
        }
        for feature, value, pct in zip(features, values, normalized)
    ]

    return sorted(items, key=lambda item: item["pct"], reverse=True), importance_type


def prediction_interval_95(prediction: float | None, residuals: np.ndarray) -> list[float] | None:
    clean_residuals = np.asarray(residuals, dtype=float)
    clean_residuals = clean_residuals[np.isfinite(clean_residuals)]
    if prediction is None or len(clean_residuals) < 2:
        return None

    margin = 1.96 * float(np.std(clean_residuals, ddof=1))
    return [float(prediction - margin), float(prediction + margin)]


def model_metrics_payload(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(actual) == 0 or len(predicted) == 0:
        return {"rmse": None, "mae": None, "r2": None}

    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def latest_manual_input_template() -> dict:
    """Return the latest real row and lag values needed for manual prediction."""
    df = load_source_dataset()
    if len(df) < 3:
        raise ValueError("At least three monthly rows are required to auto-fill prediction inputs.")

    current = df.iloc[-1]
    previous = df.iloc[-2]
    two_back = df.iloc[-3]

    fields = {
        "month": pd.to_datetime(current["date"]).strftime("%b-%y"),
        "inflation_pct": float(current["inflation_pct"]),
        "usd_tzs_rate": float(current["usd_tzs_rate"]),
        "brent_oil_usd": float(current["brent_oil_usd"]),
        "m2_bn_tzs": float(current["m2_bn_tzs"]),
        "bot_policy_rate_pct": float(current["bot_policy_rate_pct"]),
        "previous_inflation_pct": float(previous["inflation_pct"]),
        "previous_usd_tzs_rate": float(previous["usd_tzs_rate"]),
        "previous_brent_oil_usd": float(previous["brent_oil_usd"]),
        "previous_m2_bn_tzs": float(previous["m2_bn_tzs"]),
        "two_months_ago_usd_tzs_rate": float(two_back["usd_tzs_rate"]),
    }

    return {
        "source": INPUT_CSV.name,
        "input_month": fields["month"],
        "next_month": (pd.to_datetime(current["date"]) + pd.offsets.MonthBegin(1)).strftime("%b-%y"),
        "fields": fields,
    }


def _coerce_float(payload: dict, key: str, label: str) -> float:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"{label} is required.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    return parsed


def _next_month_label(month_label: str | None) -> str | None:
    if not month_label:
        return None
    for fmt in ("%b-%y", "%Y-%m"):
        try:
            month = pd.to_datetime(month_label, format=fmt)
            return (month + pd.offsets.MonthBegin(1)).strftime("%b-%y")
        except ValueError:
            continue
    try:
        month = pd.to_datetime(month_label)
    except (TypeError, ValueError):
        return None
    return (month + pd.offsets.MonthBegin(1)).strftime("%b-%y")


def manual_prediction_payload(payload: dict) -> dict:
    """Predict the next month from human-entered current and lag values."""
    values = {
        "inflation_pct": _coerce_float(payload, "inflation_pct", "Current inflation"),
        "usd_tzs_rate": _coerce_float(payload, "usd_tzs_rate", "Current USD/TZS rate"),
        "brent_oil_usd": _coerce_float(payload, "brent_oil_usd", "Current Brent oil price"),
        "m2_bn_tzs": _coerce_float(payload, "m2_bn_tzs", "Current broad money M2"),
        "bot_policy_rate_pct": _coerce_float(payload, "bot_policy_rate_pct", "Current BOT policy rate"),
        "previous_inflation_pct": _coerce_float(payload, "previous_inflation_pct", "Previous inflation"),
        "previous_usd_tzs_rate": _coerce_float(payload, "previous_usd_tzs_rate", "Previous USD/TZS rate"),
        "previous_brent_oil_usd": _coerce_float(payload, "previous_brent_oil_usd", "Previous Brent oil price"),
        "previous_m2_bn_tzs": _coerce_float(payload, "previous_m2_bn_tzs", "Previous broad money M2"),
        "two_months_ago_usd_tzs_rate": _coerce_float(
            payload,
            "two_months_ago_usd_tzs_rate",
            "USD/TZS rate two months ago",
        ),
    }
    if values["previous_m2_bn_tzs"] == 0:
        raise ValueError("Previous broad money M2 cannot be zero.")

    usd_tzs_change = values["usd_tzs_rate"] - values["previous_usd_tzs_rate"]
    usd_tzs_change_lag1 = values["previous_usd_tzs_rate"] - values["two_months_ago_usd_tzs_rate"]
    m2_growth = (values["m2_bn_tzs"] - values["previous_m2_bn_tzs"]) / values["previous_m2_bn_tzs"]

    infl_features = {
        "inflation_pct": values["inflation_pct"],
        "inflation_lag1": values["previous_inflation_pct"],
        "brent_oil_usd": values["brent_oil_usd"],
        "m2_growth": m2_growth,
    }
    fx_features = {
        "usd_tzs_change": usd_tzs_change,
        "usd_tzs_change_lag1": usd_tzs_change_lag1,
        "brent_oil_usd": values["brent_oil_usd"],
        "m2_growth": m2_growth,
        "bot_policy_rate_pct": values["bot_policy_rate_pct"],
    }

    infl_model, fx_model = load_models()
    infl_pred_next = float(infl_model.predict(pd.DataFrame([infl_features], columns=INFL_FEATURES))[0])
    fx_pred_next_change = float(fx_model.predict(pd.DataFrame([fx_features], columns=FX_FEATURES))[0])
    fx_pred_next_level = values["usd_tzs_rate"] + fx_pred_next_change

    input_month = str(payload.get("month") or "").strip() or None
    return {
        "input_month": input_month,
        "next_month": _next_month_label(input_month),
        "model_training": {
            "model_start": MODEL_START_DATE.strftime("%b-%y"),
            "model_end": MODEL_END_DATE.strftime("%b-%y"),
            "live_start": LIVE_START_DATE.strftime("%b-%y"),
        },
        "infl": {
            "pred_next_inflation": infl_pred_next,
        },
        "fx": {
            "latest_usdtzs_level": values["usd_tzs_rate"],
            "pred_next_usdtzs_level": fx_pred_next_level,
            "pred_next_usdtzs_change": fx_pred_next_change,
        },
        "engineered_features": {
            "inflation": infl_features,
            "fx": fx_features,
        },
    }


def build_inflation_history_payload(
    df: pd.DataFrame,
    model,
    start_from: pd.Timestamp,
) -> tuple[list[float], list[float], list[str]]:
    """Return (actual_series, predicted_series, months).

    predicted_series is rolling one-step-ahead prediction matching notebook target:
      inflation_next = inflation_pct.shift(-1)
    """

    tmp = df.copy()
    tmp["inflation_lag1"] = tmp["inflation_pct"].shift(1)
    tmp["usd_tzs_change"] = tmp["usd_tzs_rate"].diff()
    tmp["usd_tzs_change_lag1"] = tmp["usd_tzs_change"].shift(1)
    tmp["brent_oil_lag1"] = tmp["brent_oil_usd"].shift(1)
    tmp["m2_growth"] = tmp["m2_bn_tzs"].pct_change()

    tmp["inflation_next"] = tmp["inflation_pct"].shift(-1)

    model_df = tmp.dropna().reset_index(drop=True)
    model_df = model_df[model_df["date"] >= start_from].reset_index(drop=True)

    actual: list[float] = []
    predicted: list[float] = []
    months: list[str] = []

    for i in range(len(model_df)):
        row = model_df.iloc[i]
        x = row[INFL_FEATURES].to_frame().T
        pred_next = float(model.predict(x)[0])

        next_month_date = pd.to_datetime(row["date"]) + pd.offsets.MonthBegin(1)
        y_next = float(row["inflation_next"])

        months.append(next_month_date.strftime("%Y-%m"))
        actual.append(y_next)
        predicted.append(pred_next)

    return actual, predicted, months


def build_fx_level_history_payload(
    df: pd.DataFrame,
    model,
    start_from: pd.Timestamp,
) -> tuple[list[float], list[float], list[str]]:
    """Return (actual_level_series, predicted_level_series, months).

    Notebook target is:
      usd_tzs_change_next = usd_tzs_change.shift(-1)
    but dashboard wants USD/TZS LEVEL, so we convert:
      level[t+1] = level[t] + predicted_change_next
    """

    tmp = df.copy()
    tmp["usd_tzs_change"] = tmp["usd_tzs_rate"].diff()
    tmp["inflation_lag1"] = tmp["inflation_pct"].shift(1)
    tmp["usd_tzs_change_lag1"] = tmp["usd_tzs_change"].shift(1)
    tmp["brent_oil_lag1"] = tmp["brent_oil_usd"].shift(1)
    tmp["m2_growth"] = tmp["m2_bn_tzs"].pct_change()

    tmp["usd_tzs_change_next"] = tmp["usd_tzs_change"].shift(-1)

    model_df = tmp.dropna().reset_index(drop=True)
    model_df = model_df[model_df["date"] >= start_from].reset_index(drop=True)

    actual: list[float] = []
    predicted: list[float] = []
    months: list[str] = []

    for i in range(len(model_df)):
        row = model_df.iloc[i]
        x = row[FX_FEATURES].to_frame().T

        pred_change_next = float(model.predict(x)[0])
        next_month_date = pd.to_datetime(row["date"]) + pd.offsets.MonthBegin(1)

        # actual level for next month = current level + actual change_next
        current_level = float(row["usd_tzs_rate"])
        actual_change_next = float(row["usd_tzs_change_next"])
        actual_level_next = current_level + actual_change_next
        pred_level_next = current_level + pred_change_next

        months.append(next_month_date.strftime("%Y-%m"))
        actual.append(actual_level_next)
        predicted.append(pred_level_next)

    return actual, predicted, months


def generate_predictions(write_json: bool = False) -> dict:
    source_df = load_source_dataset()
    df = source_df.copy()

    if not ASSUME_FEATURES:
        df = compute_features_like_notebook(df)

    infl_model, fx_model = load_models()

    latest = df.tail(1).copy()

    for col in INFL_FEATURES + FX_FEATURES:
        if col not in latest.columns:
            raise ValueError(
                f"Missing column '{col}' in input for prediction. "
                "Either add it to your real dataset or set ASSUME_FEATURES=False to recompute features."
            )

    infl_pred_next = float(infl_model.predict(latest[INFL_FEATURES])[0])

    fx_pred_next_change = float(fx_model.predict(latest[FX_FEATURES])[0])
    latest_fx_level = float(latest["usd_tzs_rate"].iloc[0]) if "usd_tzs_rate" in latest.columns else None
    fx_pred_next_level = None if latest_fx_level is None else latest_fx_level + fx_pred_next_change

    eval_df = df.copy()
    eval_df["inflation_next"] = eval_df["inflation_pct"].shift(-1)
    eval_df["usd_tzs_change_next"] = eval_df["usd_tzs_change"].shift(-1)
    eval_df = eval_df.dropna(subset=INFL_FEATURES + FX_FEATURES + ["inflation_next", "usd_tzs_change_next"]).reset_index(drop=True)

    train_df, test_df, live_df = train_test_payload_dates(eval_df)

    infl_test_actual = test_df["inflation_next"].to_numpy(dtype=float)
    infl_test_pred = infl_model.predict(test_df[INFL_FEATURES]).astype(float) if len(test_df) else np.array([])
    infl_metrics = model_metrics_payload(infl_test_actual, infl_test_pred)
    infl_residuals = infl_test_actual - infl_test_pred
    infl_ci = prediction_interval_95(infl_pred_next, infl_residuals)

    fx_test_pred_change = fx_model.predict(test_df[FX_FEATURES]).astype(float) if len(test_df) else np.array([])
    fx_base_level = test_df["usd_tzs_rate"].to_numpy(dtype=float)
    fx_test_actual_level = fx_base_level + test_df["usd_tzs_change_next"].to_numpy(dtype=float)
    fx_test_pred_level = fx_base_level + fx_test_pred_change
    fx_metrics = model_metrics_payload(fx_test_actual_level, fx_test_pred_level)
    fx_residuals = fx_test_actual_level - fx_test_pred_level
    fx_ci = prediction_interval_95(fx_pred_next_level, fx_residuals)

    infl_importance, infl_importance_type = feature_importance_payload(infl_model, INFL_FEATURES)
    fx_importance, fx_importance_type = feature_importance_payload(fx_model, FX_FEATURES)

    # Inflation history start (earliest date with all required lag/diff)
    tmp_infl = df.copy()
    tmp_infl["inflation_lag1"] = tmp_infl["inflation_pct"].shift(1)
    tmp_infl["brent_oil_lag1"] = tmp_infl["brent_oil_usd"].shift(1)
    tmp_infl["usd_tzs_change"] = tmp_infl["usd_tzs_rate"].diff()
    tmp_infl["m2_growth"] = tmp_infl["m2_bn_tzs"].pct_change()

    first_valid_infl = tmp_infl.dropna(subset=["inflation_lag1", "brent_oil_lag1", "m2_growth", "usd_tzs_change"]).head(1)["date"].iloc[0]

    infl_actual_series, infl_pred_series, infl_months = build_inflation_history_payload(df, infl_model, first_valid_infl)

    # FX history start
    tmp_fx = df.copy()
    tmp_fx["usd_tzs_change"] = tmp_fx["usd_tzs_rate"].diff()
    tmp_fx["usd_tzs_change_lag1"] = tmp_fx["usd_tzs_change"].shift(1)
    tmp_fx["m2_growth"] = tmp_fx["m2_bn_tzs"].pct_change()

    first_valid_fx = tmp_fx.dropna(subset=["usd_tzs_change", "usd_tzs_change_lag1", "m2_growth"]).head(1)["date"].iloc[0]

    fx_actual_level_series, fx_pred_level_series, fx_months = build_fx_level_history_payload(df, fx_model, first_valid_fx)

    latest_month_label = pd.to_datetime(latest["date"].iloc[0]).strftime("%b-%y")
    next_month_label = (pd.to_datetime(latest["date"].iloc[0]) + pd.offsets.MonthBegin(1)).strftime("%b-%y")

    payload = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "data": {
            "input_csv": INPUT_CSV.name,
            "latest_month": latest_month_label,
            "next_month": next_month_label,
            **train_test_metadata(train_df, test_df, live_df),
        },
        "infl": {
            "model": {
                "name": display_model_name(infl_model),
                "class": type(infl_model).__name__,
            },
            "metrics": infl_metrics,
            "feature_importance": infl_importance,
            "feature_importance_type": infl_importance_type,
            "pred_ci_95": infl_ci,
            "latest_inflation": float(latest["inflation_pct"].iloc[0]),
            "pred_next_inflation": infl_pred_next,
            "history_months": infl_months,
            "history_actual": infl_actual_series,
            "history_predicted": infl_pred_series,
            "bot_policy_rate_pct": float(latest["bot_policy_rate_pct"].iloc[0]) if "bot_policy_rate_pct" in latest.columns else None,
        },
        "fx": {
            "model": {
                "name": display_model_name(fx_model),
                "class": type(fx_model).__name__,
            },
            "metrics": fx_metrics,
            "feature_importance": fx_importance,
            "feature_importance_type": fx_importance_type,
            "pred_ci_95": fx_ci,
            "latest_usdtzs_level": latest_fx_level,
            "latest_brent_oil_usd": float(latest["brent_oil_usd"].iloc[0]) if "brent_oil_usd" in latest.columns else None,
            "pred_next_usdtzs_level": fx_pred_next_level,
            "pred_next_usdtzs_change": fx_pred_next_change,
            "history_months": fx_months,
            "history_actual_level": fx_actual_level_series,
            "history_predicted_level": fx_pred_level_series,
        },
    }

    if write_json:
        OUT_JSON.write_text(json.dumps(payload, indent=2))

    return payload


def main() -> None:
    generate_predictions(write_json=True)
    print(f"Wrote predictions to: {OUT_JSON}")


if __name__ == "__main__":
    main()
