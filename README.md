# FY Trial Forecast Dashboard

## Main Notebook

Use `tz_economic_predictor.ipynb` as the main working notebook. It is the
single notebook kept in this folder for reusable forecasting, report work, and
development experiments.

## Live App Pipeline

The website runs from the Python pipeline:

- `data_crawler.py` collects official Bank of Tanzania rows.
- `export_models.py` trains and tests models inside the Jan-18 through Dec-24 modeling window.
- `predict_realtime.py` generates dashboard and manual scenario predictions.
- `app.py` serves the FastAPI routes and dashboard.
- `dashboard_live.html` renders the website.

Run the app with:

```bash
.venv/bin/python -m uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```
