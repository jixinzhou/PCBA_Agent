from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence


class TrainingDomain:
    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        zone_ranges = artifact.get("zone_means_c")
        if not isinstance(zone_ranges, list) or len(zone_ranges) != 13:
            raise ValueError("training_domain.json 必须包含 13 个温区范围")
        self.zone_ranges = zone_ranges
        self.belt_speed_range = artifact["belt_speed_cm_min"]
        self.source = str(artifact.get("source", ""))

    @classmethod
    def load(cls, path: str | Path) -> "TrainingDomain":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls(json.load(file))

    def check(
        self,
        zone_means_c: Sequence[float],
        belt_speed_cm_min: float,
    ) -> tuple[bool, list[str]]:
        if len(zone_means_c) != 13:
            raise ValueError("zone_means_c 必须包含 13 个温区温度")

        warnings: list[str] = []
        for index, (value, limits) in enumerate(zip(zone_means_c, self.zone_ranges), start=1):
            numeric_value = float(value)
            lower = float(limits["min"])
            upper = float(limits["max"])
            if not math.isfinite(numeric_value):
                raise ValueError(f"Z{index} 必须是有限数值")
            if numeric_value < lower or numeric_value > upper:
                warnings.append(
                    f"Z{index}={numeric_value:g}℃ 超出训练范围 [{lower:g}, {upper:g}]℃"
                )

        speed = float(belt_speed_cm_min)
        speed_lower = float(self.belt_speed_range["min"])
        speed_upper = float(self.belt_speed_range["max"])
        if not math.isfinite(speed):
            raise ValueError("belt_speed_cm_min 必须是有限数值")
        if speed < speed_lower or speed > speed_upper:
            warnings.append(
                f"belt_speed_cm_min={speed:g} cm/min 超出训练范围 "
                f"[{speed_lower:g}, {speed_upper:g}] cm/min"
            )

        return not warnings, warnings

