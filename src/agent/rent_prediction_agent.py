from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

try:
    from .model_runtime import load_linear_regression_pickle, predict_rent
except ImportError:
    from model_runtime import load_linear_regression_pickle, predict_rent


DEFAULT_LOCATION_SCORE = 7.5
DEFAULT_AMENITIES = 3.0
DEFAULT_FURNISHED = 0.0


def _default_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "models" / "rent_model.pkl"


@lru_cache(maxsize=1)
def _load_model_cached(model_path_str: str):
    return load_linear_regression_pickle(Path(model_path_str))


@tool
def predict_rent_value(
    bedrooms: float,
    size_sqft: float,
    location_score: float = DEFAULT_LOCATION_SCORE,
    amenities: float = DEFAULT_AMENITIES,
    furnished: int = int(DEFAULT_FURNISHED),
) -> float:
    """Predict fair monthly rent for a property using model features.

    Use this tool whenever property details are available.
    Inputs:
    - bedrooms
    - size_sqft
    - location_score (1 to 10, default 7.5)
    - amenities (count, default 3)
    - furnished (0 or 1, default 0)
    """
    model_path = os.getenv("RENT_AGENT_MODEL_PATH", str(_default_model_path()))
    model = _load_model_cached(model_path)

    feature_values = np.array(
        [
            float(bedrooms),
            float(size_sqft),
            float(location_score),
            float(amenities),
            float(furnished),
        ],
        dtype=float,
    )
    return float(predict_rent(model, feature_values))


def build_agent_executor(model_name: str | None = None):
    llm_model = model_name or os.getenv("RENT_AGENT_LLM_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://us.api.openai.com/v1")
    llm = ChatOpenAI(model=llm_model, temperature=0, base_url=openai_base_url)
    tools = [predict_rent_value]

    system_prompt = """
You are a Real Estate Assistant.
Use tools to estimate fair monthly rent and compare against listed rent.

When enough details are present, call predict_rent_value.
If optional values are missing, use defaults:
- location_score = 7.5
- amenities = 3
- furnished = 0

After getting tool output:
1) Explain the fair-rent estimate.
2) Compare listed price vs estimate.
3) Say if the listing seems underpriced, fair, or overpriced.
4) If underpriced, draft a short email inquiry.
""".strip()

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def run_agent_task(user_input: str, model_name: str | None = None) -> str:
    executor = build_agent_executor(model_name=model_name)
    response = executor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        }
    )

    messages = response.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content:
            if isinstance(content, list):
                return "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            return str(content)

    return str(response)
