from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFECT_ALIASES = {
    "insufficient_solder": ("insufficient solder", "少锡", "焊料不足"),
    "excessive_solder": ("excessive solder", "多锡", "焊料过多"),
    "short": ("short", "bridge", "bridging", "短路", "桥连"),
    "shifted_component": ("shifted component", "misalignment", "偏移", "移位"),
}


def infer_defect_from_text(text: str | None) -> str | None:
    lowered = (text or "").lower()
    for canonical, aliases in DEFECT_ALIASES.items():
        if any(alias.lower() in lowered for alias in aliases):
            return canonical
    return None


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def assess_prediction(tool_name: str, response: dict[str, Any]) -> tuple[str, str]:
    data = response.get("data") or {}
    if tool_name == "reflow_profile_prediction":
        qualified = (data.get("overall") or {}).get("qualified")
        if qualified is False:
            return "supported", "回流预测判定当前工艺窗口不合格，支持该回流候选路径。"
        if qualified is True:
            return "contradicted", "回流预测判定当前工艺窗口合格，当前数据不支持该回流候选路径。"
    if tool_name == "spi_vte_prediction":
        return "inconclusive", "VTE目标/阈值尚未批准，不能仅凭预测值确认或排除候选路径。"
    return "inconclusive", "Tool返回缺少可用于确定性判定的已批准字段。"


def unique_missing(paths: list[str], unavailable: list[str]) -> list[str]:
    blocked = set(unavailable)
    return sorted({path for path in paths if path not in blocked})
