from __future__ import annotations

import hmac
import logging
import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
# import playwright

from data_crawler import source_status, update_dataset_and_models
from project_config import PROJECT_ROOT
from predict_realtime import (
    INPUT_CSV,
    OUT_JSON,
    dataset_payload,
    generate_predictions,
    latest_manual_input_template,
    manual_prediction_payload,
)


DASHBOARD_HTML = PROJECT_ROOT / "dashboard_live.html"
STYLE_CSS = PROJECT_ROOT / "style.css"
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="16" fill="#0071e3"/>
  <path d="M18 46V16h18c7 0 12 5 12 12s-5 12-12 12H27v6h-9zm9-15h8c3 0 5-2 5-4s-2-4-5-4h-8v8z" fill="#fff"/>
  <path d="M42 45l8 7" stroke="#ffd166" stroke-width="6" stroke-linecap="round"/>
</svg>"""

logger = logging.getLogger("fy_dashboard")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").strip().lower()
UPDATE_API_TOKEN = os.environ.get("UPDATE_API_TOKEN", "")
_request_times: dict[str, deque[float]] = defaultdict(deque)
_request_times_lock = Lock()

app = FastAPI(
    title="Proci EF-03 Dashboard API",
    description="FastAPI backend for the Tanzania inflation and USD/TZS forecast dashboard.",
    version="1.1.0",
)


def authorize_data_update(provided_token: str | None) -> None:
    """Keep model retraining private in production while preserving local use."""
    if not UPDATE_API_TOKEN:
        if ENVIRONMENT == "production":
            raise HTTPException(status_code=503, detail="Data updates are not configured.")
        return
    if not provided_token or not hmac.compare_digest(provided_token, UPDATE_API_TOKEN):
        raise HTTPException(status_code=401, detail="A valid update token is required.")


def enforce_rate_limit(request: Request, scope: str, limit: int, window_seconds: int) -> None:
    """Apply a small in-process limit to public compute-heavy endpoints."""
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    client_host = forwarded_for or (request.client.host if request.client else "unknown")
    key = f"{scope}:{client_host}"
    now = time.monotonic()
    cutoff = now - window_seconds
    with _request_times_lock:
        timestamps = _request_times[key]
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if len(timestamps) >= limit:
            raise HTTPException(status_code=429, detail="Too many requests. Try again shortly.")
        timestamps.append(now)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(
        DASHBOARD_HTML,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/dashboard", include_in_schema=False)
def dashboard_alias() -> FileResponse:
    return dashboard()


@app.get("/dataset", include_in_schema=False)
def dataset_page() -> FileResponse:
    return dashboard()


@app.get("/predict", include_in_schema=False)
def prediction_page() -> FileResponse:
    return dashboard()


@app.get("/style.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(
        STYLE_CSS,
        media_type="text/css",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> Response:
    return Response(
        content=FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/dataset")
def dataset(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> JSONResponse:
    try:
        return JSONResponse(dataset_payload(offset=offset, limit=limit))
    except Exception as exc:
        logger.exception("Dataset request failed")
        raise HTTPException(status_code=500, detail="Dataset request failed.") from exc


@app.get("/dataset.csv", include_in_schema=False)
def download_dataset() -> FileResponse:
    return FileResponse(
        INPUT_CSV,
        media_type="text/csv",
        filename=INPUT_CSV.name,
    )


@app.get("/api/data-sources")
def data_sources() -> JSONResponse:
    return JSONResponse(source_status())


@app.get("/api/predict-template")
def predict_template() -> JSONResponse:
    try:
        return JSONResponse(latest_manual_input_template())
    except Exception as exc:
        logger.exception("Prediction template request failed")
        raise HTTPException(status_code=500, detail="Prediction template request failed.") from exc


@app.post("/api/manual-prediction")
def manual_prediction(request: Request, payload: dict) -> JSONResponse:
    enforce_rate_limit(request, "manual-prediction", limit=30, window_seconds=60)
    try:
        return JSONResponse(manual_prediction_payload(payload))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Manual prediction failed")
        raise HTTPException(status_code=500, detail="Manual prediction failed.") from exc


@app.post("/api/update-data")
def update_data(x_update_token: str | None = Header(default=None)) -> JSONResponse:
    authorize_data_update(x_update_token)
    try:
        payload = update_dataset_and_models()
    except Exception as exc:
        logger.exception("Official data update failed")
        raise HTTPException(
            status_code=502,
            detail="Official data update failed. Check the service logs.",
        ) from exc

    return JSONResponse(payload)


@app.get("/api/predictions")
def predictions(
    refresh: bool = Query(
        default=False,
        description="Use false to read predictions.json when available. Use true to regenerate from the current CSV and model files.",
    ),
    x_update_token: str | None = Header(default=None),
) -> JSONResponse:
    if refresh:
        authorize_data_update(x_update_token)
    try:
        if refresh:
            payload = generate_predictions(write_json=True)
        else:
            if not OUT_JSON.exists():
                payload = generate_predictions(write_json=False)
            else:
                return FileResponse(OUT_JSON, media_type="application/json")
    except Exception as exc:
        logger.exception("Prediction request failed")
        raise HTTPException(status_code=500, detail="Prediction request failed.") from exc

    return JSONResponse(payload)


@app.get("/predictions.json", include_in_schema=False)
def predictions_json_compat() -> JSONResponse:
    return predictions(refresh=False)
