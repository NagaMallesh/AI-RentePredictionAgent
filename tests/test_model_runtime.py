import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.agent.model_runtime import (
    ModelLoadError,
    explain_contributions,
    load_linear_regression_pickle,
    predict_rent,
)


class TestModelRuntime(unittest.TestCase):
    def _create_model_file(self, payload: dict) -> Path:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp.close()
        path = Path(temp.name)
        with open(path, "wb") as file:
            pickle.dump(payload, file)
        return path

    def test_load_valid_multi_feature_model(self):
        payload = {
            "coefficients": [300, 0.8, 50, 25, 150],
            "intercept": 200,
            "n_features": 5,
            "saved_at": "2026-03-24T12:00:00",
            "model_type": "LinearRegression",
        }
        path = self._create_model_file(payload)
        model = load_linear_regression_pickle(path)

        self.assertEqual(model.n_features, 5)
        self.assertEqual(model.intercept, 200.0)
        np.testing.assert_allclose(model.coefficients, np.array(payload["coefficients"]))

    def test_reject_non_five_feature_model(self):
        payload = {
            "coefficients": [400],
            "intercept": 500,
            "n_features": 1,
        }
        path = self._create_model_file(payload)

        with self.assertRaises(ModelLoadError):
            load_linear_regression_pickle(path)

    def test_predict_and_explain(self):
        payload = {
            "coefficients": [300, 0.8, 50, 25, 150],
            "intercept": 200,
            "n_features": 5,
        }
        path = self._create_model_file(payload)
        model = load_linear_regression_pickle(path)

        features = np.array([2, 1000, 8, 3, 1], dtype=float)
        predicted = predict_rent(model, features)
        expected = 2 * 300 + 1000 * 0.8 + 8 * 50 + 3 * 25 + 1 * 150 + 200
        self.assertAlmostEqual(predicted, expected, places=6)

        contributions = explain_contributions(model, features)
        self.assertEqual(len(contributions), 5)
        self.assertIn("feature", contributions[0])


if __name__ == "__main__":
    unittest.main()
