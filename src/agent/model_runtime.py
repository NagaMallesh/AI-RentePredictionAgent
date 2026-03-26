from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np


FEATURE_NAMES = ["bedrooms", "size_sqft", "location_score", "amenities", "furnished"]


@dataclass
class LoadedLinearModel:
    coefficients: np.ndarray
    intercept: float
    n_features: int
    saved_at: str
    model_type: str
    source_path: Path


class ModelLoadError(ValueError):
    pass


def _as_coeff_array(coefficients: Any) -> np.ndarray:
    if isinstance(coefficients, (int, float, np.number)):
        return np.array([float(coefficients)], dtype=float)
    if isinstance(coefficients, list):
        if len(coefficients) == 0:
            raise ModelLoadError("Model coefficients cannot be empty")
        return np.array(coefficients, dtype=float)
    raise ModelLoadError("Model coefficients must be a number or a list")


def load_linear_regression_pickle(model_path: Path) -> LoadedLinearModel:
    path = model_path.expanduser().resolve()
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    try:
        with open(path, "rb") as file:
            payload = pickle.load(file)
    except Exception as exc:
        raise ModelLoadError(f"Unable to load pickle model: {exc}") from exc

    if not isinstance(payload, dict):
        raise ModelLoadError("Invalid model format: expected dictionary payload")

    missing = [key for key in ("coefficients", "intercept") if key not in payload]
    if missing:
        raise ModelLoadError(f"Model payload missing required keys: {', '.join(missing)}")

    coefficients = _as_coeff_array(payload["coefficients"])

    try:
        intercept = float(payload["intercept"])
    except Exception as exc:
        raise ModelLoadError("Invalid intercept value in model payload") from exc

    payload_n_features = payload.get("n_features", len(coefficients))
    try:
        n_features = int(payload_n_features)
    except Exception as exc:
        raise ModelLoadError("Invalid n_features value in model payload") from exc

    if n_features != len(coefficients):
        raise ModelLoadError(
            f"Model mismatch: n_features={n_features} but coefficients={len(coefficients)}"
        )

    if n_features != 5:
        raise ModelLoadError(
            f"This agent supports only 5-feature rent models, but loaded model has {n_features} features"
        )

    return LoadedLinearModel(
        coefficients=coefficients,
        intercept=intercept,
        n_features=n_features,
        saved_at=str(payload.get("saved_at", "unknown")),
        model_type=str(payload.get("model_type", "LinearRegression")),
        source_path=path,
    )


def predict_rent(model: LoadedLinearModel, feature_values: np.ndarray) -> float:
    if feature_values.shape != (model.n_features,):
        raise ValueError(
            f"Expected {model.n_features} feature values, received shape {feature_values.shape}"
        )
    return float(np.dot(feature_values, model.coefficients) + model.intercept)


def explain_contributions(model: LoadedLinearModel, feature_values: np.ndarray) -> list[dict[str, float | str]]:
    contributions = []
    for index, feature_name in enumerate(FEATURE_NAMES):
        contribution = float(feature_values[index] * model.coefficients[index])
        contributions.append(
            {
                "feature": feature_name,
                "value": float(feature_values[index]),
                "coefficient": float(model.coefficients[index]),
                "contribution": contribution,
            }
        )
    contributions.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
    return contributions
