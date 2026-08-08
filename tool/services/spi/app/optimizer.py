"""Differential-evolution optimization for solder-paste-printing parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from .predictor import VTEPredictor


TARGET_VTE = 100.0
TOLERANCE = 5.0
OPTIMIZER_NAME = "differential_evolution"
RANDOM_SEED = 42

# VTE error remains the primary objective. This small regularizer chooses solutions
# nearer to the current process settings when their predicted VTE is comparable.
PARAMETER_CHANGE_WEIGHT = 0.25


@dataclass(frozen=True)
class OptimizationResult:
    before_prediction: float
    recommended_parameters: dict[str, float]
    after_prediction: float
    objective_error: float
    target_reached: bool
    within_training_domain: bool
    warnings: list[str]


def optimize_parameters(
    predictor: VTEPredictor,
    current_parameters: dict[str, float],
) -> OptimizationResult:
    """Search model training bounds for a near-target, low-change parameter set."""

    fields = predictor.api_fields
    bounds = predictor.bounds
    current_vector = np.asarray(
        [current_parameters[field] for field in fields], dtype=float
    )
    lower_bounds = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper_bounds = np.asarray([bound[1] for bound in bounds], dtype=float)
    spans = upper_bounds - lower_bounds
    movement_reference = np.clip(current_vector, lower_bounds, upper_bounds)

    before_prediction = predictor.predict(current_parameters)
    _, warnings = predictor.check_training_domain(current_parameters)

    def vector_to_parameters(vector: np.ndarray) -> dict[str, float]:
        return {
            field: float(value)
            for field, value in zip(fields, vector, strict=True)
        }

    def objective(vector: np.ndarray) -> float:
        prediction = predictor.predict(vector_to_parameters(vector))
        vte_error = abs(prediction - TARGET_VTE)
        normalized_change = np.sqrt(
            np.mean(np.square((vector - movement_reference) / spans))
        )
        return float(vte_error + PARAMETER_CHANGE_WEIGHT * normalized_change)

    result = differential_evolution(
        objective,
        bounds=bounds,
        x0=movement_reference,
        seed=RANDOM_SEED,
        maxiter=100,
        popsize=15,
        tol=1e-7,
        atol=1e-7,
        polish=True,
        workers=1,
        updating="immediate",
    )

    recommended_parameters = vector_to_parameters(result.x)
    after_prediction = predictor.predict(recommended_parameters)
    objective_error = abs(after_prediction - TARGET_VTE)
    within_training_domain, domain_warnings = predictor.check_training_domain(
        recommended_parameters
    )
    warnings.extend(domain_warnings)
    if not result.success and objective_error > TOLERANCE:
        warnings.append(
            "差分进化算法已返回当前最优结果，但尚未达到目标容差。"
        )

    return OptimizationResult(
        before_prediction=before_prediction,
        recommended_parameters=recommended_parameters,
        after_prediction=after_prediction,
        objective_error=objective_error,
        target_reached=objective_error <= TOLERANCE,
        within_training_domain=within_training_domain,
        warnings=warnings,
    )
