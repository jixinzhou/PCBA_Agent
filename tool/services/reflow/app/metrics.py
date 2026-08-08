from __future__ import annotations

import math
from typing import Any, Mapping


MODEL_METRIC_TO_API_FIELD = {
    "斜率 1 ( 40-150°C)": "heating_slope_40_150_c_per_s",
    "斜率 2 (200-217°C)": "heating_slope_200_217_c_per_s",
    "温度最高下降斜率": "max_cooling_slope_c_per_s",
    "预热 40至150°C": "preheat_time_40_150_s",
    "恒温时间150至200°C": "soak_time_150_200_s",
    "回流时间 /217°C": "time_above_217_s",
    "最高温度": "peak_temperature_c",
}

# 已由用户确认的制程上下限。200-217℃ 升温斜率没有规格，不参与 PWI。
PWI_LIMITS = {
    "heating_slope_40_150_c_per_s": (0.0, 2.5),
    "max_cooling_slope_c_per_s": (-3.0, -1.0),
    "preheat_time_40_150_s": (60.0, 150.0),
    "soak_time_150_200_s": (60.0, 120.0),
    "time_above_217_s": (35.0, 90.0),
    "peak_temperature_c": (230.0, 250.0),
}


def calculate_pwi(value: float, lower: float, upper: float) -> float:
    """计算单项 PWI：中心为 0，规格边界为 100，越界时大于 100。"""
    half_range = (upper - lower) / 2.0
    if half_range <= 0:
        raise ValueError("PWI 规格上限必须大于下限")
    center = (upper + lower) / 2.0
    return abs(value - center) / half_range * 100.0


def build_process_metrics(raw_metrics: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in MODEL_METRIC_TO_API_FIELD if name not in raw_metrics]
    if missing:
        raise ValueError(f"指标模型缺少输出字段: {', '.join(missing)}")

    api_metrics: dict[str, float] = {}
    for model_name, api_name in MODEL_METRIC_TO_API_FIELD.items():
        value = float(raw_metrics[model_name])
        if not math.isfinite(value):
            raise ValueError(f"指标模型输出非有限值: {model_name}={value}")
        api_metrics[api_name] = value

    pwi_values = [
        calculate_pwi(api_metrics[name], lower, upper)
        for name, (lower, upper) in PWI_LIMITS.items()
    ]
    point_pwi = max(pwi_values)

    rounded_pwi = round(point_pwi, 2)
    result: dict[str, Any] = {
        name: round(value, 6) for name, value in api_metrics.items()
    }
    result["pwi"] = rounded_pwi
    # status 与响应中的 PWI 保持可直接校验的契约：pwi <= 100 即合格。
    result["status"] = "qualified" if rounded_pwi <= 100.0 else "unqualified"
    return result
