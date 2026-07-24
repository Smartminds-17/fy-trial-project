import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module


class DashboardApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_module.app)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_dashboard_and_predictions_are_available(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.get("/api/predictions?refresh=false")
        self.assertEqual(response.status_code, 200)
        self.assertIn("data", response.json())

    def test_update_requires_configured_token(self):
        original_token = app_module.UPDATE_API_TOKEN
        try:
            app_module.UPDATE_API_TOKEN = "test-secret"
            response = self.client.post("/api/update-data")
            self.assertEqual(response.status_code, 401)
            response = self.client.post(
                "/api/update-data", headers={"X-Update-Token": "wrong-secret"}
            )
            self.assertEqual(response.status_code, 401)
            response = self.client.get("/api/predictions?refresh=true")
            self.assertEqual(response.status_code, 401)
        finally:
            app_module.UPDATE_API_TOKEN = original_token

    def test_manual_prediction_rejects_missing_fields(self):
        response = self.client.post("/api/manual-prediction", json={})
        self.assertEqual(response.status_code, 422)
        self.assertIn("required", response.json()["detail"])


class RuntimeDataTests(unittest.TestCase):
    def test_versioned_seed_artifacts_exist(self):
        for filename in (
            "tz_economic_predictors.csv",
            "infl_rf.joblib",
            "fx_rf.joblib",
            "predictions.json",
            "data_sources.json",
        ):
            self.assertTrue((Path(__file__).parents[1] / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
