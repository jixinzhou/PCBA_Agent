from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tool.common.schemas import ErrorDetail, HealthResponse


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PredictionPoint(StrictModel):
    point_id: str = Field(..., min_length=1, description="测点唯一标识，响应中原样返回")
    component_x_mm: float = Field(..., description="器件 X 坐标，单位 mm")
    component_y_mm: float = Field(..., description="器件 Y 坐标，单位 mm")
    component_volume_mm3: float = Field(..., ge=0, description="器件体积，单位 mm³")


class PredictionInput(StrictModel):
    points: list[PredictionPoint] = Field(..., min_length=1, description="需要预测的测点")
    zone_means_c: list[float] = Field(
        ...,
        min_length=13,
        max_length=13,
        description="Z1 到 Z13 的温区平均温度，单位 ℃",
    )
    belt_speed_cm_min: float = Field(..., gt=0, description="链速，单位 cm/min")

    @model_validator(mode="after")
    def validate_unique_point_ids(self) -> "PredictionInput":
        point_ids = [point.point_id for point in self.points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("points 中的 point_id 必须唯一")
        return self


class PredictionOptions(StrictModel):
    return_temperature_curve: bool = Field(default=True, description="是否返回温度曲线")
    curve_downsample_interval_s: float = Field(
        default=0.25,
        ge=0.25,
        description="温度曲线输出间隔，最小为模型原始采样间隔 0.25 秒",
    )


class PredictionRequest(StrictModel):
    request_id: str = Field(..., min_length=1, description="调用方生成的请求唯一标识")
    input_data: PredictionInput = Field(..., alias="input")
    options: PredictionOptions = Field(default_factory=PredictionOptions)


class ProcessMetrics(StrictModel):
    heating_slope_40_150_c_per_s: float
    heating_slope_200_217_c_per_s: float
    max_cooling_slope_c_per_s: float
    preheat_time_40_150_s: float
    soak_time_150_200_s: float
    time_above_217_s: float
    peak_temperature_c: float
    pwi: float
    status: Literal["qualified", "unqualified"]


class TemperatureCurve(StrictModel):
    duration_s: float
    sample_interval_s: float
    time_s: list[float]
    temperature_c: list[float]


class PointResult(StrictModel):
    point_id: str
    matched_tc: str
    matched_ref: str
    metrics: ProcessMetrics
    temperature_curve: TemperatureCurve | None


class OverallResult(StrictModel):
    max_pwi: float
    qualified: bool
    worst_point_id: str


class PredictionData(StrictModel):
    point_results: list[PointResult]
    overall: OverallResult
    within_training_domain: bool


class PredictionResponse(StrictModel):
    success: bool
    request_id: str | None
    api_version: str
    tool_name: str
    tool_version: str
    model_name: str
    model_version: str
    execution_time_ms: int
    data: PredictionData | None
    warnings: list[str]
    error: ErrorDetail | None


OptimizationMode = Literal[
    "minimize_pwi",
    "target_peak_temperature",
    "target_time_above_217",
]
ZoneIndex = Annotated[int, Field(ge=1, le=13)]


class CurrentParameters(StrictModel):
    zone_means_c: list[float] = Field(
        ...,
        min_length=13,
        max_length=13,
        description="当前 Z1 到 Z13 温区平均温度，单位 ℃",
    )
    belt_speed_cm_min: float = Field(..., gt=0, description="当前链速，单位 cm/min")


class OptimizationTarget(StrictModel):
    mode: OptimizationMode
    target_value_c: float | None = None
    tolerance_c: float | None = Field(default=None, gt=0)
    target_value_s: float | None = None
    tolerance_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_mode_parameters(self) -> "OptimizationTarget":
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
                raise ValueError("minimize_pwi 模式不能提供目标值或容差")
            return self

        if self.mode == "target_peak_temperature":
            if self.target_value_c is None or self.tolerance_c is None:
                raise ValueError(
                    "target_peak_temperature 模式必须提供 target_value_c 和 tolerance_c"
                )
            if not 230.0 <= self.target_value_c <= 250.0:
                raise ValueError("target_value_c 必须位于工艺范围 [230, 250]℃")
            if self.target_value_s is not None or self.tolerance_s is not None:
                raise ValueError("峰值温度目标模式不能提供 TAL 目标字段")
            return self

        if self.target_value_s is None or self.tolerance_s is None:
            raise ValueError(
                "target_time_above_217 模式必须提供 target_value_s 和 tolerance_s"
            )
        if not 35.0 <= self.target_value_s <= 90.0:
            raise ValueError("target_value_s 必须位于工艺范围 [35, 90] 秒")
        if self.target_value_c is not None or self.tolerance_c is not None:
            raise ValueError("TAL 目标模式不能提供峰值温度目标字段")
        return self


class AdjustableParameters(StrictModel):
    zone_indexes: list[ZoneIndex] = Field(..., description="允许调整的温区编号，范围 1-13")
    adjust_belt_speed: bool

    @model_validator(mode="after")
    def validate_adjustable_parameters(self) -> "AdjustableParameters":
        if len(self.zone_indexes) != len(set(self.zone_indexes)):
            raise ValueError("zone_indexes 不能包含重复编号")
        if not self.zone_indexes and not self.adjust_belt_speed:
            raise ValueError("至少需要允许调整一个温区或链速")
        return self


class OptimizationInput(StrictModel):
    points: list[PredictionPoint] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="第一版只支持单测点，因此必须且只能包含一个元素",
    )
    current_parameters: CurrentParameters
    optimization_target: OptimizationTarget
    adjustable_parameters: AdjustableParameters


class OptimizationRequest(StrictModel):
    request_id: str = Field(..., min_length=1, description="调用方生成的请求唯一标识")
    input_data: OptimizationInput = Field(..., alias="input")


class ParameterSet(StrictModel):
    zone_means_c: list[float | int]
    belt_speed_cm_min: float | int


class OptimizationBefore(ParameterSet):
    max_pwi: float
    peak_temperature_c: float
    time_above_217_s: float


class OptimizationAfter(StrictModel):
    max_pwi: float
    peak_temperature_c: float
    time_above_217_s: float


class OptimizationData(StrictModel):
    optimization_mode: OptimizationMode
    before: OptimizationBefore
    recommended_parameters: ParameterSet
    after: OptimizationAfter
    target_reached: bool
    within_training_domain: bool


class OptimizationResponse(StrictModel):
    success: bool
    request_id: str | None
    api_version: str
    tool_name: str
    tool_version: str
    model_name: str
    model_version: str
    execution_time_ms: int
    data: OptimizationData | None
    warnings: list[str]
    error: ErrorDetail | None
