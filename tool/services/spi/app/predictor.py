"""Shared VTE model loading and prediction utilities."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


SERVICE_DIR = Path(__file__).resolve().parents[1]
MODEL_INFO_PATH = SERVICE_DIR / "models" / "model_info.json"


class PredictorConfigurationError(RuntimeError):
    """Raised when the model metadata is missing or inconsistent."""


class VTEPredictor:
    """Load the trained model once and expose thread-safe prediction helpers."""

    def __init__(self, model_info_path: Path = MODEL_INFO_PATH) -> None:
        self.model_info_path = model_info_path
        self.model_info = self._load_model_info(model_info_path)
        self.model_name = self._required_string("model_name")
        self.model_version = self._required_string("model_version")

        model_file = self._required_string("model_file")
        self.model_path = model_info_path.parent / model_file
        if not self.model_path.is_file():
            raise PredictorConfigurationError(
                f"VTE mean model file not found: {self.model_path}"
            )

        raw_features = self.model_info.get("input_features")
        if not isinstance(raw_features, dict) or not raw_features:
            raise PredictorConfigurationError(
                "model_info.json must contain a non-empty input_features object"
            )

        self.api_fields = tuple(raw_features.keys())
        self.model_columns: tuple[str, ...] = tuple(
            self._feature_string(field_name, config, "model_column")
            for field_name, config in raw_features.items()
        )
        self.training_ranges: dict[str, tuple[float, float]] = {}
        for field_name, config in raw_features.items():
            minimum = self._feature_number(field_name, config, "min")
            maximum = self._feature_number(field_name, config, "max")
            if minimum >= maximum:
                raise PredictorConfigurationError(
                    f"Invalid training range for {field_name}: [{minimum}, {maximum}]"
                )
            self.training_ranges[field_name] = (minimum, maximum)

        self._model = joblib.load(self.model_path)
        self._prediction_lock = threading.Lock()

    @staticmethod
    def _load_model_info(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise PredictorConfigurationError(f"Model info file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as file:
                content = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise PredictorConfigurationError(
                f"Unable to read model info file: {path}"
            ) from exc
        if not isinstance(content, dict):
            raise PredictorConfigurationError("model_info.json root must be an object")
        return content

    def _required_string(self, key: str) -> str:
        value = self.model_info.get(key)
        if not isinstance(value, str) or not value:
            raise PredictorConfigurationError(
                f"model_info.json field {key!r} must be a non-empty string"
            )
        return value

    @staticmethod
    def _feature_string(field_name: str, config: Any, key: str) -> str:
        if not isinstance(config, dict):
            raise PredictorConfigurationError(
                f"Feature configuration for {field_name} must be an object"
            )
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise PredictorConfigurationError(
                f"Feature {field_name} field {key!r} must be a non-empty string"
            )
        return value

    @staticmethod
    def _feature_number(field_name: str, config: Any, key: str) -> float:
        if not isinstance(config, dict):
            raise PredictorConfigurationError(
                f"Feature configuration for {field_name} must be an object"
            )
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PredictorConfigurationError(
                f"Feature {field_name} field {key!r} must be numeric"
            )
        number = float(value)
        if not math.isfinite(number):
            raise PredictorConfigurationError(
                f"Feature {field_name} field {key!r} must be finite"
            )
        return number

    def _ordered_values(self, values: Mapping[str, float]) -> list[float]:
        missing = [field for field in self.api_fields if field not in values]
        if missing:
            raise ValueError(f"Missing model input fields: {', '.join(missing)}")

        ordered_values = [float(values[field]) for field in self.api_fields]
        if not all(math.isfinite(value) for value in ordered_values):
            raise ValueError("All model input values must be finite")
        return ordered_values

    def predict(self, values: Mapping[str, float]) -> float:
        """Predict VTE mean for one set of process parameters."""

        model_input = pd.DataFrame(
            [self._ordered_values(values)],
            columns=self.model_columns,
        )
        with self._prediction_lock:
            prediction = float(self._model.predict(model_input)[0])
        if not math.isfinite(prediction):
            raise RuntimeError("Model returned a non-finite prediction")
        return prediction

    def check_training_domain(
        self, values: Mapping[str, float]
    ) -> tuple[bool, list[str]]:
        """Return whether all inputs are inside their metadata training ranges."""

        ordered_values = self._ordered_values(values)
        warnings: list[str] = []
        for field_name, value in zip(self.api_fields, ordered_values, strict=True):
            minimum, maximum = self.training_ranges[field_name]
            if value < minimum or value > maximum:
                warnings.append(
                    f"{field_name}={value} 超出训练范围 [{minimum}, {maximum}]"
                )
        return not warnings, warnings

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Return optimizer bounds in the model feature order."""

        return [self.training_ranges[field] for field in self.api_fields]
