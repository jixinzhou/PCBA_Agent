from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .metrics import build_process_metrics


DELIVERY_MODEL_FILENAME = "reflow_gpr_delivery_model.joblib"
CURVE_MODEL_FILENAME = "reflow_curve_bspline_lightgbm_model.joblib"
FEATURE_COLUMNS = [f"zone_mean_{index}" for index in range(1, 14)] + ["belt_speed"]


def build_feature_row(zone_means_c: Sequence[float], belt_speed_cm_min: float) -> pd.DataFrame:
    if len(zone_means_c) != 13:
        raise ValueError(f"zone_means_c 必须包含 13 个值，实际为 {len(zone_means_c)} 个")
    values = [float(value) for value in zone_means_c] + [float(belt_speed_cm_min)]
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def physical_time_axis(duration_s: float, sample_interval_s: float) -> np.ndarray:
    if sample_interval_s <= 0:
        raise ValueError("采样间隔必须大于 0")
    n_points = int(math.floor(duration_s / sample_interval_s)) + 1
    time_s = np.arange(n_points, dtype=float) * sample_interval_s
    if duration_s - time_s[-1] > 1e-9:
        time_s = np.append(time_s, duration_s)
    return time_s


def bspline_basis(tau: np.ndarray, n_basis: int, degree: int) -> np.ndarray:
    if n_basis <= degree:
        raise ValueError("B-spline 基函数数量必须大于阶数")
    n_internal = n_basis - degree - 1
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)
    knots = np.r_[np.zeros(degree + 1), internal, np.ones(degree + 1)]
    return BSpline.design_matrix(tau, knots, degree, extrapolate=True).toarray()


class DeliveryPredictor:
    def __init__(self, artifact: Mapping[str, Any]):
        self.artifact = dict(artifact)
        self.metric_names: list[str] = list(self.artifact["metric_names"])
        self.tc_models: Mapping[str, Mapping[str, Any]] = self.artifact["tc_models"]
        self.route_points: Mapping[str, Mapping[str, Any]] = self.artifact["route_points"]
        self.route_tcs: list[str] = list(self.artifact["route_tcs"])

    @classmethod
    def load(cls, model_path: str | Path) -> "DeliveryPredictor":
        return cls(joblib.load(model_path))

    def select_tc(
        self,
        component_x_mm: float,
        component_y_mm: float,
        component_volume_mm3: float,
    ) -> dict[str, Any]:
        if component_volume_mm3 < 0:
            raise ValueError("component_volume_mm3 不能为负数")

        log_volume = math.log1p(float(component_volume_mm3))
        ranked: list[dict[str, Any]] = []
        for tc in self.route_tcs:
            point = self.route_points[tc]
            d_volume = abs(log_volume - math.log1p(float(point["volume"])))
            d_position = math.hypot(
                float(component_x_mm) - float(point["x"]),
                float(component_y_mm) - float(point["y"]),
            )
            ranked.append(
                {
                    "tc": tc,
                    "ref": str(point.get("ref", "")),
                    "d_log_volume": d_volume,
                    "d_position": d_position,
                }
            )

        ranked.sort(
            key=lambda item: (item["d_log_volume"], item["d_position"], item["tc"])
        )
        return ranked[0]

    def predict_metrics(self, matched_tc: str, feature_row: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            for metric_name in self.metric_names:
                model = self.tc_models[matched_tc][metric_name]
                metrics[metric_name] = float(model.predict(feature_row)[0])
        return metrics


class CurvePredictor:
    def __init__(self, artifact: Mapping[str, Any]):
        self.artifact = dict(artifact)
        self.route_tcs: list[str] = list(self.artifact["route_tcs"])
        self.route_points: Mapping[str, Mapping[str, Any]] = self.artifact["route_points"]
        self.tc_models: Mapping[str, Mapping[str, Any]] = self.artifact["tc_models"]
        self.bspline_basis_count = int(self.artifact["bspline_basis_count"])
        self.bspline_degree = int(self.artifact["bspline_degree"])
        self.sample_interval_s = float(self.artifact.get("sample_interval_s", 0.25))
        self.process_length_cm = float(self.artifact.get("process_length_cm", 619.0))

    @classmethod
    def load(cls, model_path: str | Path) -> "CurvePredictor":
        return cls(joblib.load(model_path))

    def predict_curve(
        self,
        matched_tc: str,
        feature_row: pd.DataFrame,
        belt_speed_cm_min: float,
        output_interval_s: float,
    ) -> dict[str, Any]:
        if belt_speed_cm_min <= 0:
            raise ValueError("belt_speed_cm_min 必须大于 0")
        if output_interval_s + 1e-12 < self.sample_interval_s:
            raise ValueError(
                f"curve_downsample_interval_s 不能小于模型原始采样间隔 "
                f"{self.sample_interval_s:g} 秒"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            coefficients = self.tc_models[matched_tc]["regressor"].predict(feature_row)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)

        duration_s = self.process_length_cm / float(belt_speed_cm_min) * 60.0
        original_time_s = physical_time_axis(duration_s, self.sample_interval_s)
        tau = np.clip(original_time_s / duration_s, 0.0, 1.0)
        basis = bspline_basis(tau, self.bspline_basis_count, self.bspline_degree)
        original_temperature_c = basis @ coefficients[0]

        if math.isclose(output_interval_s, self.sample_interval_s, rel_tol=0.0, abs_tol=1e-12):
            output_time_s = original_time_s
            output_temperature_c = original_temperature_c
        else:
            output_time_s = physical_time_axis(duration_s, output_interval_s)
            output_temperature_c = np.interp(
                output_time_s,
                original_time_s,
                original_temperature_c,
            )

        return {
            "duration_s": round(float(duration_s), 6),
            "sample_interval_s": round(float(output_interval_s), 6),
            "time_s": [round(float(value), 6) for value in output_time_s],
            "temperature_c": [round(float(value), 6) for value in output_temperature_c],
        }


class PredictionService:
    def __init__(self, delivery: DeliveryPredictor, curve: CurvePredictor):
        self.delivery = delivery
        self.curve = curve
        self._validate_model_compatibility()

    @classmethod
    def load(cls, models_dir: str | Path) -> "PredictionService":
        model_path = Path(models_dir)
        delivery = DeliveryPredictor.load(model_path / DELIVERY_MODEL_FILENAME)
        curve = CurvePredictor.load(model_path / CURVE_MODEL_FILENAME)
        return cls(delivery, curve)

    def _validate_model_compatibility(self) -> None:
        if self.delivery.route_tcs != self.curve.route_tcs:
            raise ValueError("指标模型与曲线模型的 TC 路由列表不一致")
        for tc in self.delivery.route_tcs:
            delivery_point = self.delivery.route_points[tc]
            curve_point = self.curve.route_points[tc]
            for field in ("x", "y", "volume", "ref"):
                if delivery_point.get(field) != curve_point.get(field):
                    raise ValueError(f"指标模型与曲线模型的 {tc}.{field} 路由信息不一致")

    @property
    def route_tcs(self) -> list[str]:
        return list(self.delivery.route_tcs)

    @property
    def curve_sample_interval_s(self) -> float:
        return self.curve.sample_interval_s

    def predict_point(
        self,
        *,
        point_id: str,
        component_x_mm: float,
        component_y_mm: float,
        component_volume_mm3: float,
        feature_row: pd.DataFrame,
        belt_speed_cm_min: float,
        return_temperature_curve: bool,
        curve_downsample_interval_s: float,
    ) -> dict[str, Any]:
        route = self.delivery.select_tc(
            component_x_mm,
            component_y_mm,
            component_volume_mm3,
        )
        matched_tc = str(route["tc"])
        raw_metrics = self.delivery.predict_metrics(matched_tc, feature_row)
        metrics = build_process_metrics(raw_metrics)

        temperature_curve = None
        if return_temperature_curve:
            temperature_curve = self.curve.predict_curve(
                matched_tc,
                feature_row,
                belt_speed_cm_min,
                curve_downsample_interval_s,
            )

        return {
            "point_id": point_id,
            "matched_tc": matched_tc,
            "matched_ref": str(route["ref"]),
            "metrics": metrics,
            "temperature_curve": temperature_curve,
        }

