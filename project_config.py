"""Shared project file choices.

The working notebook is the reusable development notebook and the single
notebook kept in this project folder.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

MAIN_NOTEBOOK = PROJECT_ROOT / "tz_economic_predictor.ipynb"
INPUT_CSV = PROJECT_ROOT / "tz_economic_predictors.csv"
INFL_MODEL_PATH = PROJECT_ROOT / "infl_rf.joblib"
FX_MODEL_PATH = PROJECT_ROOT / "fx_rf.joblib"
PREDICTIONS_JSON = PROJECT_ROOT / "predictions.json"
