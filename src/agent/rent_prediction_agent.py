from __future__ import annotations

import logging
import os
import time
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
DEFAULT_REQUEST_TIMEOUT_SECONDS = 45.0
DEFAULT_REQUEST_RETRIES = 2


logger = logging.getLogger(__name__)


def _default_model_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "models" / "rent_model.pkl"


def _request_timeout_seconds() -> float:
    raw_value = os.getenv("RENT_AGENT_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)).strip()
    try:
        parsed = float(raw_value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning(
            "Invalid RENT_AGENT_REQUEST_TIMEOUT_SECONDS=%s; using default %.1f",
            raw_value,
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        return DEFAULT_REQUEST_TIMEOUT_SECONDS


def _request_retries() -> int:
    raw_value = os.getenv("RENT_AGENT_REQUEST_RETRIES", str(DEFAULT_REQUEST_RETRIES)).strip()
    try:
        parsed = int(raw_value)
        if parsed < 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning("Invalid RENT_AGENT_REQUEST_RETRIES=%s; using default %d", raw_value, DEFAULT_REQUEST_RETRIES)
        return DEFAULT_REQUEST_RETRIES


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
    timeout_seconds = _request_timeout_seconds()
    retries = _request_retries()
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
        base_url=openai_base_url,
        timeout=timeout_seconds,
        max_retries=retries,
    )
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
4) If underpriced, output exactly --- DRAFT EMAIL --- on its own line, then write the draft email below it.
""".strip()

    return create_agent(model=llm, tools=tools, system_prompt=system_prompt)


def run_agent_task(user_input: str, model_name: str | None = None) -> str:
    executor = build_agent_executor(model_name=model_name)
    retries = _request_retries()
    last_error: Exception | None = None
    response = None

    for attempt in range(retries + 1):
        try:
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
            break
        except Exception as error:
            last_error = error
            logger.warning("Agent invoke attempt %d/%d failed: %s", attempt + 1, retries + 1, error)
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

    if response is None:
        raise RuntimeError(f"Agent request failed after {retries + 1} attempt(s): {last_error}")

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


if __name__ == "__main__":
    import sys
    from pathlib import Path

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except ImportError:
        pass

    try:
        from voice_output import speak_response
    except ImportError:
        from .voice_output import speak_response

    print("=" * 64)
    print("🏠 AI Rent Prediction Agent")
    print("=" * 64)

    while True:
        print("\nEnter listing details (or 'exit' to quit):")
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            sys.exit(0)
        if not user_input:
            continue
        try:
            output = run_agent_task(user_input)
            print(f"\n{output}")
            speak_response(output)
        except Exception as error:
            print(f"\nError: {error}")
