"""Shared project file choices.

The working notebook is the reusable development notebook and the single
notebook kept in this project folder.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT)).expanduser().resolve()

_PERSISTED_FILES = (
    "tz_economic_predictors.csv",
    "infl_rf.joblib",
    "fx_rf.joblib",
    "predictions.json",
    "data_sources.json",
)


def prepare_runtime_data() -> None:
    """Seed an empty persistent data directory from versioned app artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in _PERSISTED_FILES:
        source = PROJECT_ROOT / filename
        destination = DATA_DIR / filename
        if not destination.exists() and source.exists() and source != destination:
            shutil.copy2(source, destination)


prepare_runtime_data()

MAIN_NOTEBOOK = PROJECT_ROOT / "tz_economic_predictor.ipynb"
INPUT_CSV = DATA_DIR / "tz_economic_predictors.csv"
INFL_MODEL_PATH = DATA_DIR / "infl_rf.joblib"
FX_MODEL_PATH = DATA_DIR / "fx_rf.joblib"
PREDICTIONS_JSON = DATA_DIR / "predictions.json"
SOURCE_LOG_PATH = DATA_DIR / "data_sources.json"
