import os
import unittest
from pathlib import Path

from src.agent.rent_prediction_agent import predict_rent_value


class TestLangChainToolAgent(unittest.TestCase):
    def test_predict_rent_value_tool(self):
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "rent_model.pkl"
        if not model_path.exists():
            self.skipTest("Model file not found for tool test")

        os.environ["RENT_AGENT_MODEL_PATH"] = str(model_path)
        value = predict_rent_value.invoke(
            {
                "bedrooms": 2,
                "size_sqft": 1200,
                "location_score": 8,
                "amenities": 4,
                "furnished": 1,
            }
        )
        self.assertIsInstance(value, float)
        self.assertGreater(value, 0)


if __name__ == "__main__":
    unittest.main()
