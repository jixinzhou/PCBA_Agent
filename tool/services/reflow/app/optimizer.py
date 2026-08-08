from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import MODEL_METRIC_TO_API_FIELD, PWI_LIMITS, calculate_pwi
from .predictor import DeliveryPredictor, FEATURE_COLUMNS
from .training_domain import TrainingDomain


SWARM_SIZE = 50
MAX_ITERATIONS = 100
RANDOM_SEED = 42


class OptimizationInputError(ValueError):
    """输入参数导致优化问题无可行解。"""


@dataclass(frozen=True)
class CandidateEvaluation:
    max_pwi: float
    peak_temperature_c: float
    time_above_217_s: float

    def response_dict(self) -> dict[str, float]:
        return {
            "max_pwi": round(self.max_pwi, 2),
            "peak_temperature_c": round(self.peak_temperature_c, 6),
            "time_above_217_s": round(self.time_above_217_s, 6),
        }


@dataclass(frozen=True)
class OptimizationResult:
    recommended_parameters: tuple[float, ...]
    before: CandidateEvaluation
    after: CandidateEvaluation
    target_reached: bool
    warnings: tuple[str, ...]


def response_number(value: float) -> int | float:
    numeric = float(value)
    if math.isclose(numeric, round(numeric), rel_tol=0.0, abs_tol=1e-9):
        return int(round(numeric))
    return round(numeric, 6)


class SinglePointPSOOptimizer:
    """在模型训练域内优化单个测点的回流焊参数。"""

    def __init__(
        self,
        predictor: DeliveryPredictor,
        training_domain: TrainingDomain,
        *,
        swarm_size: int = SWARM_SIZE,
        max_iterations: int = MAX_ITERATIONS,
        random_seed: int = RANDOM_SEED,
    ):
        self.predictor = predictor
        self.training_domain = training_domain
        self.swarm_size = int(swarm_size)
        self.max_iterations = int(max_iterations)
        self.random_seed = int(random_seed)
        if self.swarm_size < 2:
            raise ValueError("swarm_size 必须至少为 2")
        if self.max_iterations < 1:
            raise ValueError("max_iterations 必须至少为 1")

        zone_bounds = [
            (float(item["min"]), float(item["max"]))
            for item in training_domain.zone_ranges
        ]
        speed_bounds = (
            float(training_domain.belt_speed_range["min"]),
            float(training_domain.belt_speed_range["max"]),
        )
        self.bounds = np.asarray(zone_bounds + [speed_bounds], dtype=float)

    def optimize(
        self,
        *,
        component_x_mm: float,
        component_y_mm: float,
        component_volume_mm3: float,
        current_zone_means_c: Sequence[float],
        current_belt_speed_cm_min: float,
        mode: str,
        target_value: float | None,
        tolerance: float | None,
        zone_indexes: Sequence[int],
        adjust_belt_speed: bool,
    ) -> OptimizationResult:
        current = np.asarray(
            [float(value) for value in current_zone_means_c]
            + [float(current_belt_speed_cm_min)],
            dtype=float,
        )
        if current.shape != (14,):
            raise OptimizationInputError("current_parameters 必须包含 13 个温区和 1 个链速")

        adjustable_dims = sorted({int(index) - 1 for index in zone_indexes})
        if adjust_belt_speed:
            adjustable_dims.append(13)
        if not adjustable_dims:
            raise OptimizationInputError("没有可调整的工艺参数")

        adjustable_set = set(adjustable_dims)
        self._validate_fixed_parameters(current, adjustable_set)
        feasible_seed = self._build_feasible_seed(current, adjustable_set)

        route = self.predictor.select_tc(
            component_x_mm,
            component_y_mm,
            component_volume_mm3,
        )
        matched_tc = str(route["tc"])
        cache: dict[tuple[float, ...], CandidateEvaluation] = {}

        before_key = tuple(float(value) for value in current)
        self._evaluate_candidates([before_key], matched_tc, cache)
        before = cache[before_key]

        lower = self.bounds[adjustable_dims, 0]
        upper = self.bounds[adjustable_dims, 1]
        rng = np.random.default_rng(self.random_seed)
        positions = rng.uniform(
            low=lower,
            high=upper,
            size=(self.swarm_size, len(adjustable_dims)),
        )
        positions[0] = feasible_seed[adjustable_dims]
        if self.swarm_size > 1 and self._is_feasible_candidate(current, adjustable_set):
            positions[1] = np.clip(current[adjustable_dims], lower, upper)

        velocity = np.zeros_like(positions)
        velocity_limit = (upper - lower) * 0.2
        personal_best = positions.copy()
        personal_cost = self._evaluate_positions(
            positions,
            current,
            adjustable_dims,
            matched_tc,
            mode,
            target_value,
            tolerance,
            cache,
        )
        global_index = int(np.argmin(personal_cost))
        global_best = personal_best[global_index].copy()
        global_cost = float(personal_cost[global_index])

        for iteration in range(self.max_iterations):
            inertia = 0.9 - (0.9 - 0.4) * (iteration / self.max_iterations)
            r1 = rng.random(positions.shape)
            r2 = rng.random(positions.shape)
            velocity = (
                inertia * velocity
                + 1.5 * r1 * (personal_best - positions)
                + 1.5 * r2 * (global_best - positions)
            )
            velocity = np.clip(velocity, -velocity_limit, velocity_limit)
            positions = np.clip(positions + velocity, lower, upper)

            costs = self._evaluate_positions(
                positions,
                current,
                adjustable_dims,
                matched_tc,
                mode,
                target_value,
                tolerance,
                cache,
            )
            improved = costs < personal_cost
            if np.any(improved):
                personal_best[improved] = positions[improved]
                personal_cost[improved] = costs[improved]
                candidate_index = int(np.argmin(personal_cost))
                candidate_cost = float(personal_cost[candidate_index])
                if candidate_cost < global_cost:
                    global_best = personal_best[candidate_index].copy()
                    global_cost = candidate_cost

        recommended = self._expand_position(global_best, current, adjustable_dims)
        recommended_key = tuple(float(value) for value in recommended)
        self._evaluate_candidates([recommended_key], matched_tc, cache)
        after = cache[recommended_key]
        rounded_after_pwi = round(after.max_pwi, 2)

        if mode == "minimize_pwi":
            target_reached = rounded_after_pwi <= 100.0
        else:
            assert target_value is not None and tolerance is not None
            optimized_value = (
                after.peak_temperature_c
                if mode == "target_peak_temperature"
                else after.time_above_217_s
            )
            target_reached = (
                abs(optimized_value - target_value) <= tolerance + 1e-12
                and rounded_after_pwi <= 100.0
            )

        result_warnings: list[str] = []
        if not self._trend_is_valid(current):
            result_warnings.append("当前参数不满足温区非递减约束，推荐参数已修正该趋势")
        if not target_reached:
            result_warnings.append("在当前可调参数和训练域约束下未达到优化目标")

        return OptimizationResult(
            recommended_parameters=recommended_key,
            before=before,
            after=after,
            target_reached=target_reached,
            warnings=tuple(result_warnings),
        )

    def _validate_fixed_parameters(
        self,
        current: np.ndarray,
        adjustable_set: set[int],
    ) -> None:
        for dimension, value in enumerate(current):
            if dimension in adjustable_set:
                continue
            lower, upper = self.bounds[dimension]
            if value < lower or value > upper:
                name = f"Z{dimension + 1}" if dimension < 13 else "belt_speed_cm_min"
                raise OptimizationInputError(
                    f"不可调参数 {name}={value:g} 超出训练范围 [{lower:g}, {upper:g}]"
                )

    def _build_feasible_seed(
        self,
        current: np.ndarray,
        adjustable_set: set[int],
    ) -> np.ndarray:
        lower = self.bounds[:, 0].copy()
        upper = self.bounds[:, 1].copy()
        for dimension in range(14):
            if dimension in adjustable_set:
                lower[dimension] = math.ceil(lower[dimension])
                upper[dimension] = math.floor(upper[dimension])
            else:
                lower[dimension] = current[dimension]
                upper[dimension] = current[dimension]

        seed = current.copy()
        previous = -math.inf
        for dimension in range(12):
            candidate = max(lower[dimension], previous)
            if dimension in adjustable_set:
                candidate = math.ceil(candidate)
            if candidate > upper[dimension] + 1e-12:
                raise OptimizationInputError(
                    "当前不可调参数与训练域无法同时满足 Z1<=Z2<=...<=Z12"
                )
            seed[dimension] = candidate
            previous = candidate

        z13_candidate = lower[12]
        if 12 in adjustable_set:
            z13_candidate = math.ceil(z13_candidate)
        if z13_candidate > upper[12] + 1e-12:
            raise OptimizationInputError("Z13 在训练域中不存在可行整数值")

        if seed[11] + 1e-12 < z13_candidate:
            if 11 not in adjustable_set:
                raise OptimizationInputError("不可调的 Z12/Z13 无法满足 Z13<=Z12")
            raised_z12 = math.ceil(z13_candidate)
            if raised_z12 > upper[11] + 1e-12:
                raise OptimizationInputError("训练域中不存在满足 Z13<=Z12 的推荐参数")
            seed[11] = raised_z12
        seed[12] = z13_candidate

        speed_dimension = 13
        if speed_dimension in adjustable_set:
            seed[speed_dimension] = int(
                np.clip(round(current[speed_dimension]), lower[speed_dimension], upper[speed_dimension])
            )
        else:
            seed[speed_dimension] = current[speed_dimension]
        return seed

    def _is_feasible_candidate(
        self,
        parameters: np.ndarray,
        adjustable_set: set[int],
    ) -> bool:
        for dimension, value in enumerate(parameters):
            lower, upper = self.bounds[dimension]
            if value < lower - 1e-12 or value > upper + 1e-12:
                return False
            if dimension in adjustable_set and not math.isclose(
                value,
                round(value),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return False
        return self._trend_is_valid(parameters)

    @staticmethod
    def _trend_is_valid(parameters: Sequence[float]) -> bool:
        return all(
            float(parameters[index]) <= float(parameters[index + 1]) + 1e-12
            for index in range(11)
        ) and float(parameters[12]) <= float(parameters[11]) + 1e-12

    @staticmethod
    def _trend_violation(parameters: Sequence[float]) -> float:
        violation = sum(
            max(0.0, float(parameters[index]) - float(parameters[index + 1])) ** 2
            for index in range(11)
        )
        violation += max(0.0, float(parameters[12]) - float(parameters[11])) ** 2
        return violation

    def _expand_position(
        self,
        position: np.ndarray,
        current: np.ndarray,
        adjustable_dims: Sequence[int],
    ) -> np.ndarray:
        parameters = current.copy()
        for column, dimension in enumerate(adjustable_dims):
            lower, upper = self.bounds[dimension]
            parameters[dimension] = int(np.clip(round(position[column]), lower, upper))
        return parameters

    def _evaluate_positions(
        self,
        positions: np.ndarray,
        current: np.ndarray,
        adjustable_dims: Sequence[int],
        matched_tc: str,
        mode: str,
        target_value: float | None,
        tolerance: float | None,
        cache: dict[tuple[float, ...], CandidateEvaluation],
    ) -> np.ndarray:
        parameters = [
            tuple(float(value) for value in self._expand_position(position, current, adjustable_dims))
            for position in positions
        ]
        valid_parameters = [item for item in parameters if self._trend_is_valid(item)]
        self._evaluate_candidates(valid_parameters, matched_tc, cache)

        costs = np.empty(len(parameters), dtype=float)
        for index, item in enumerate(parameters):
            violation = self._trend_violation(item)
            if violation > 0:
                costs[index] = 1e12 + violation * 1e9
                continue
            costs[index] = self._objective_cost(
                cache[item],
                mode,
                target_value,
                tolerance,
            )
        return costs

    def _evaluate_candidates(
        self,
        candidates: Sequence[tuple[float, ...]],
        matched_tc: str,
        cache: dict[tuple[float, ...], CandidateEvaluation],
    ) -> None:
        missing = list(dict.fromkeys(candidate for candidate in candidates if candidate not in cache))
        if not missing:
            return

        frame = pd.DataFrame(missing, columns=FEATURE_COLUMNS)
        predictions: dict[str, np.ndarray] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            for model_metric, api_field in MODEL_METRIC_TO_API_FIELD.items():
                model = self.predictor.tc_models[matched_tc][model_metric]
                predictions[api_field] = np.asarray(model.predict(frame), dtype=float)

        for row_index, candidate in enumerate(missing):
            metric_values = {
                name: float(values[row_index]) for name, values in predictions.items()
            }
            pwi_values = [
                calculate_pwi(metric_values[name], lower, upper)
                for name, (lower, upper) in PWI_LIMITS.items()
            ]
            cache[candidate] = CandidateEvaluation(
                max_pwi=max(pwi_values),
                peak_temperature_c=metric_values["peak_temperature_c"],
                time_above_217_s=metric_values["time_above_217_s"],
            )

    @staticmethod
    def _objective_cost(
        evaluation: CandidateEvaluation,
        mode: str,
        target_value: float | None,
        tolerance: float | None,
    ) -> float:
        if mode == "minimize_pwi":
            return evaluation.max_pwi

        if target_value is None or tolerance is None:
            raise ValueError("目标模式缺少目标值或容差")
        value = (
            evaluation.peak_temperature_c
            if mode == "target_peak_temperature"
            else evaluation.time_above_217_s
        )
        distance = abs(value - target_value)
        normalized_distance = distance / tolerance
        if distance <= tolerance + 1e-12:
            # 进入目标容差后，以 max_pwi 为主要优化目标，目标中心距离只用于打破平局。
            return evaluation.max_pwi + normalized_distance * 1e-4
        # 目标容差是第一优先级；未进入容差的候选一定劣于已进入容差的候选。
        return 1e9 + (normalized_distance - 1.0) * 1e6 + evaluation.max_pwi

