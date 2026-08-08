"""Pydantic input contracts exposed to the Agent."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, populate_by_name=True)


class ClassificationToolInput(ToolInput):
    image_path: str = Field(..., min_length=1, description="Local JPG/JPEG/PNG path")
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    top_k: int = Field(default=3, ge=1, le=5)


class PrintingParameters(ToolInput):
    squeegee_pressure_kgf: float
    squeegee_speed_m_s: float
    separation_speed_m_s: float
    separation_distance_mm: float


class SpiPredictionToolInput(ToolInput):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    input: PrintingParameters


class SpiOptimizationInput(ToolInput):
    current_parameters: PrintingParameters


class SpiOptimizationToolInput(ToolInput):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    input: SpiOptimizationInput


class ReflowPoint(ToolInput):
    point_id: str = Field(..., min_length=1)
    component_x_mm: float
    component_y_mm: float
    component_volume_mm3: float = Field(..., ge=0)


class ReflowPredictionInput(ToolInput):
    points: list[ReflowPoint] = Field(..., min_length=1)
    zone_means_c: list[float] = Field(..., min_length=13, max_length=13)
    belt_speed_cm_min: float = Field(..., gt=0)

    @model_validator(mode="after")
    def unique_points(self) -> "ReflowPredictionInput":
        ids = [point.point_id for point in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("point_id must be unique")
        return self


class ReflowOptions(ToolInput):
    return_temperature_curve: bool = True
    curve_downsample_interval_s: float = Field(default=0.25, ge=0.25)


class ReflowPredictionToolInput(ToolInput):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    input: ReflowPredictionInput
    options: ReflowOptions = Field(default_factory=ReflowOptions)


class ReflowParameters(ToolInput):
    zone_means_c: list[float] = Field(..., min_length=13, max_length=13)
    belt_speed_cm_min: float = Field(..., gt=0)


OptimizationMode = Literal[
    "minimize_pwi",
    "target_peak_temperature",
    "target_time_above_217",
]


class ReflowOptimizationTarget(ToolInput):
    mode: OptimizationMode
    target_value_c: float | None = None
    tolerance_c: float | None = Field(default=None, gt=0)
    target_value_s: float | None = None
    tolerance_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def mode_fields(self) -> "ReflowOptimizationTarget":
        if self.mode == "minimize_pwi":
            if any(
                value is not None
                for value in (
                    self.target_value_c,
                    self.tolerance_c,
                    self.target_value_s,
                    self.tolerance_s,
                )
            ):
                raise ValueError("minimize_pwi does not accept target fields")
        elif self.mode == "target_peak_temperature":
            if self.target_value_c is None or self.tolerance_c is None:
                raise ValueError("peak target and tolerance are required")
            if not 230 <= self.target_value_c <= 250:
                raise ValueError("target_value_c must be within [230, 250]")
            if self.target_value_s is not None or self.tolerance_s is not None:
                raise ValueError("TAL target fields are not allowed in peak mode")
        else:
            if self.target_value_s is None or self.tolerance_s is None:
                raise ValueError("TAL target and tolerance are required")
            if not 35 <= self.target_value_s <= 90:
                raise ValueError("target_value_s must be within [35, 90]")
            if self.target_value_c is not None or self.tolerance_c is not None:
                raise ValueError("peak target fields are not allowed in TAL mode")
        return self


ZoneIndex = Annotated[int, Field(ge=1, le=13)]


class ReflowAdjustableParameters(ToolInput):
    zone_indexes: list[ZoneIndex]
    adjust_belt_speed: bool

    @model_validator(mode="after")
    def adjustable_fields(self) -> "ReflowAdjustableParameters":
        if len(self.zone_indexes) != len(set(self.zone_indexes)):
            raise ValueError("zone_indexes must be unique")
        if not self.zone_indexes and not self.adjust_belt_speed:
            raise ValueError("at least one parameter must be adjustable")
        return self


class ReflowOptimizationInput(ToolInput):
    points: list[ReflowPoint] = Field(..., min_length=1, max_length=1)
    current_parameters: ReflowParameters
    optimization_target: ReflowOptimizationTarget
    adjustable_parameters: ReflowAdjustableParameters


class ReflowOptimizationToolInput(ToolInput):
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    input: ReflowOptimizationInput
