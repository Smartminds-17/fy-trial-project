from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response

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

app = FastAPI(
    title="Proci EF-03 Dashboard API",
    description="FastAPI backend for the Tanzania inflation and USD/TZS forecast dashboard.",
    version="1.0.0",
)


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
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/manual-prediction")
def manual_prediction(payload: dict) -> JSONResponse:
    try:
        return JSONResponse(manual_prediction_payload(payload))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/update-data")
def update_data() -> JSONResponse:
    try:
        payload = update_dataset_and_models()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Official data update failed: {exc}",
        ) from exc

    return JSONResponse(payload)


@app.get("/api/predictions")
def predictions(
    refresh: bool = Query(
        default=False,
        description="Use false to read predictions.json when available. Use true to regenerate from the current CSV and model files.",
    )
) -> JSONResponse:
    try:
        if refresh:
            payload = generate_predictions(write_json=True)
        else:
            if not OUT_JSON.exists():
                payload = generate_predictions(write_json=False)
            else:
                return FileResponse(OUT_JSON, media_type="application/json")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(payload)


@app.get("/predictions.json", include_in_schema=False)
def predictions_json_compat() -> JSONResponse:
    return predictions(refresh=False)
