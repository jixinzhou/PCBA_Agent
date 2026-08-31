from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


DEFECT_ALIASES = {
    "insufficient_solder": ("insufficient solder", "少锡", "焊料不足"),
    "excessive_solder": ("excessive solder", "多锡", "焊料过多"),
    "short": ("short", "bridge", "bridging", "短路", "桥连"),
    "shifted_component": ("shifted component", "misalignment", "偏移", "移位"),
}

SPI_VTE_LOW_THRESHOLD = 95.0
SPI_VTE_HIGH_THRESHOLD = 105.0
SPI_INSUFFICIENT_RELATIONSHIP = "REL-INSUFFICIENT-SOLDER-PRINTING"
SPI_EXCESSIVE_RELATIONSHIP = "REL-EXCESSIVE-SOLDER-PRINTING"


def spi_vte_in_target(response: dict[str, Any]) -> bool:
    data = response.get("data") or {}
    raw_vte = data.get("vte_mean")
    return bool(
        isinstance(raw_vte, (int, float))
        and math.isfinite(float(raw_vte))
        and data.get("within_training_domain") is True
        and SPI_VTE_LOW_THRESHOLD <= float(raw_vte) <= SPI_VTE_HIGH_THRESHOLD
    )


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


def assess_prediction(
    tool_name: str,
    response: dict[str, Any],
    *,
    relationship_id: str | None = None,
) -> tuple[str, str]:
    data = response.get("data") or {}
    if tool_name == "reflow_profile_prediction":
        qualified = (data.get("overall") or {}).get("qualified")
        if qualified is False:
            return "supported", "回流预测判定当前工艺窗口不合格，支持该回流候选路径。"
        if qualified is True:
            return "contradicted", "回流预测判定当前工艺窗口合格，当前数据不支持该回流候选路径。"
    if tool_name == "spi_vte_prediction":
        raw_vte = data.get("vte_mean")
        if not isinstance(raw_vte, (int, float)) or not math.isfinite(float(raw_vte)):
            return "inconclusive", "SPI预测未返回有效VTE均值，候选路径保持不确定。"
        vte = float(raw_vte)
        if data.get("within_training_domain") is not True:
            return (
                "inconclusive",
                f"SPI预测VTE={vte:.4f}%，但输入不在确认的训练域内，候选路径保持不确定。",
            )
        if relationship_id == SPI_INSUFFICIENT_RELATIONSHIP:
            if vte < SPI_VTE_LOW_THRESHOLD:
                return (
                    "supported",
                    f"SPI预测VTE={vte:.4f}%低于{SPI_VTE_LOW_THRESHOLD:.0f}%阈值，支持少锡印刷候选路径。",
                )
            return (
                "inconclusive",
                f"SPI预测VTE={vte:.4f}%未低于{SPI_VTE_LOW_THRESHOLD:.0f}%阈值，不能据此支持少锡印刷候选路径。",
            )
        if relationship_id == SPI_EXCESSIVE_RELATIONSHIP:
            if vte > SPI_VTE_HIGH_THRESHOLD:
                return (
                    "supported",
                    f"SPI预测VTE={vte:.4f}%高于{SPI_VTE_HIGH_THRESHOLD:.0f}%阈值，支持多锡印刷候选路径。",
                )
            return (
                "inconclusive",
                f"SPI预测VTE={vte:.4f}%未高于{SPI_VTE_HIGH_THRESHOLD:.0f}%阈值，不能据此支持多锡印刷候选路径。",
            )
        return "inconclusive", "当前SPI候选路径没有已批准的VTE方向性判定规则。"
    return "inconclusive", "Tool返回缺少可用于确定性判定的已批准字段。"


def unique_missing(paths: list[str], unavailable: list[str]) -> list[str]:
    blocked = set(unavailable)
    return sorted({path for path in paths if path not in blocked})
