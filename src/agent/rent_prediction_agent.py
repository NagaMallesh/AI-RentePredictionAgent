from __future__ import annotations

import logging
import os
import re
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
EMAIL_DELIMITER = "--- DRAFT EMAIL ---"


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


def _hybrid_mode_enabled() -> bool:
    value = os.getenv("RENT_AGENT_HYBRID_MODE", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _llm_fallback_enabled() -> bool:
    value = os.getenv("RENT_AGENT_LLM_FALLBACK_ENABLED", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None

    try:
        return float(match.group(1))
    except (ValueError, TypeError, IndexError):
        return None


def _extract_furnished_flag(text: str) -> int | None:
    if re.search(r"\bunfurnished\b", text, flags=re.IGNORECASE):
        return 0

    explicit = re.search(
        r"\bfurnished\s*(?:[:=]|is)?\s*(1|0|true|false|yes|no|y|n)\b",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        value = explicit.group(1).strip().lower()
        return 1 if value in {"1", "true", "yes", "y"} else 0

    if re.search(r"\bfurnished\b", text, flags=re.IGNORECASE):
        return 1

    return None


def extract_listing_features(user_input: str) -> dict[str, float] | None:
    text = user_input.strip()
    if not text:
        return None

    bedrooms = _extract_float(
        r"\b(\d+(?:\.\d+)?)\s*(?:bedrooms?|beds?|br|bds?|b(?:\s|,|$))\b",
        text,
    )
    size_sqft = _extract_float(
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft|sq|sf|square\s*feet)\b",
        text,
    )
    location_score = _extract_float(
        r"\blocation(?:\s*score)?\s*(?:[:=]|is|of)?\s*(\d+(?:\.\d+)?)\b",
        text,
    )
    amenities = _extract_float(
        r"\bamenities?(?:\s*(?:count|number))?\s*(?:[:=]|is|of)?\s*(\d+(?:\.\d+)?)\b",
        text,
    )
    listed_rent = _extract_float(
        r"\b(?:listed|asking|list|price|rent)\s*(?:rent|price)?\s*(?:[:=]|is|at|of)?\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b",
        text,
    )
    if listed_rent is None:
        listed_rent = _extract_float(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b", text)
    furnished = _extract_furnished_flag(text)

    if bedrooms is None or size_sqft is None:
        return None

    return {
        "bedrooms": bedrooms,
        "size_sqft": size_sqft,
        "location_score": location_score if location_score is not None else DEFAULT_LOCATION_SCORE,
        "amenities": amenities if amenities is not None else DEFAULT_AMENITIES,
        "furnished": float(furnished if furnished is not None else int(DEFAULT_FURNISHED)),
        "listed_rent": listed_rent,
    }


def _compose_email_draft(fair_rent: float, listed_rent: float) -> str:
    return (
        "Hi,\n\n"
        "I am interested in this rental and would like to schedule a viewing. "
        f"Based on recent comparisons, the listing appears competitively priced around ${fair_rent:,.0f} "
        f"(currently listed at ${listed_rent:,.0f}), so I wanted to reach out quickly.\n\n"
        "Could you share available viewing times this week?\n\n"
        "Thanks,"
    )


def _build_model_only_response(features: dict[str, float]) -> str:
    estimated_rent = predict_rent_value.invoke(
        {
            "bedrooms": features["bedrooms"],
            "size_sqft": features["size_sqft"],
            "location_score": features["location_score"],
            "amenities": features["amenities"],
            "furnished": int(features["furnished"]),
        }
    )

    lines = [f"Estimated fair monthly rent: ${estimated_rent:,.2f}."]
    listed_rent = features.get("listed_rent")

    if listed_rent is None:
        lines.append("Listed rent was not provided, so only the fair-rent estimate is available.")
        return "\n".join(lines)

    difference = float(listed_rent) - float(estimated_rent)
    percent_diff = (difference / float(estimated_rent)) * 100 if estimated_rent else 0.0

    if difference < -50:
        classification = "underpriced"
        lines.append(
            f"The listing appears {classification}: ${abs(difference):,.2f} below estimate ({abs(percent_diff):.1f}% lower)."
        )
        lines.append(EMAIL_DELIMITER)
        lines.append(_compose_email_draft(float(estimated_rent), float(listed_rent)))
    elif difference > 50:
        classification = "overpriced"
        lines.append(
            f"The listing appears {classification}: ${abs(difference):,.2f} above estimate ({abs(percent_diff):.1f}% higher)."
        )
    else:
        lines.append("The listing appears fairly priced versus the model estimate.")

    return "\n".join(lines)


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
    if _hybrid_mode_enabled():
        features = extract_listing_features(user_input)
        if features is not None:
            logger.info("Hybrid route: using local rent model without LLM call")
            return _build_model_only_response(features)

        if not _llm_fallback_enabled():
            raise RuntimeError(
                "Could not extract required structured fields (bedrooms and size_sqft) from input, and LLM fallback is disabled"
            )

        logger.info("Hybrid route: falling back to LLM because structured extraction was insufficient")

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
