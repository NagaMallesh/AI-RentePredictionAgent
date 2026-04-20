import os
import unittest
from unittest import mock

from src.agent.rent_prediction_agent import extract_listing_features, run_agent_task


class TestHybridRouting(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def test_extract_listing_features_from_structured_text(self):
        features = extract_listing_features(
            "2 bed 1200 sqft location 8 amenities 4 furnished 1 listed rent 2100"
        )

        self.assertIsNotNone(features)
        self.assertEqual(features["bedrooms"], 2.0)
        self.assertEqual(features["size_sqft"], 1200.0)
        self.assertEqual(features["location_score"], 8.0)
        self.assertEqual(features["amenities"], 4.0)
        self.assertEqual(features["furnished"], 1.0)
        self.assertEqual(features["listed_rent"], 2100.0)

    def test_run_agent_task_uses_model_only_path_when_features_exist(self):
        os.environ["RENT_AGENT_HYBRID_MODE"] = "true"
        with mock.patch("src.agent.rent_prediction_agent.predict_rent", return_value=1900.0), mock.patch(
            "src.agent.rent_prediction_agent.build_agent_executor"
        ) as mock_build_executor:
            response = run_agent_task("2 bed 1200 sqft listed 1700 furnished 0")

        self.assertIn("Estimated fair monthly rent", response)
        self.assertIn("underpriced", response)
        self.assertIn("--- DRAFT EMAIL ---", response)
        mock_build_executor.assert_not_called()

    def test_run_agent_task_respects_no_llm_fallback(self):
        os.environ["RENT_AGENT_HYBRID_MODE"] = "true"
        os.environ["RENT_AGENT_LLM_FALLBACK_ENABLED"] = "false"

        with self.assertRaises(RuntimeError):
            run_agent_task("Is this a good deal?")

    def test_run_agent_task_falls_back_to_llm_when_enabled(self):
        os.environ["RENT_AGENT_HYBRID_MODE"] = "true"
        os.environ["RENT_AGENT_LLM_FALLBACK_ENABLED"] = "true"

        mock_executor = mock.MagicMock()
        mock_executor.invoke.return_value = {
            "messages": [
                mock.Mock(content="LLM analysis")
            ]
        }

        with mock.patch("src.agent.rent_prediction_agent.build_agent_executor", return_value=mock_executor):
            response = run_agent_task("Is this a good deal?")

        self.assertEqual(response, "LLM analysis")


if __name__ == "__main__":
    unittest.main()
