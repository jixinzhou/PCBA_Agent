"""Live HTTP end-to-end verification for all five PCBA Agent Tools."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

from tool.agent_tools.classification import PCBADefectClassificationTool
from tool.agent_tools.reflow import (
    ReflowParameterOptimizationTool,
    ReflowProfilePredictionTool,
)
from tool.agent_tools.spi import SPIParameterOptimizationTool, SPIVTEPredictionTool


ERROR_KEYS = {
    "success",
    "request_id",
    "api_version",
    "tool_name",
    "tool_version",
    "model_name",
    "model_version",
    "execution_time_ms",
    "data",
    "warnings",
    "error",
}
ERROR_DETAIL_KEYS = {"code", "message", "details"}


def assert_error_schema(
    response: httpx.Response,
    expected_tool: str,
    expected_status: int = 422,
) -> dict[str, Any]:
    payload = response.json()
    assert response.status_code == expected_status, (response.status_code, payload)
    assert set(payload) == ERROR_KEYS, payload
    assert payload["success"] is False
    assert payload["data"] is None
    assert payload["tool_name"] == expected_tool
    assert set(payload["error"]) == ERROR_DETAIL_KEYS
    return payload


def assert_health(base_url: str, expected_tool: str) -> dict[str, Any]:
    with httpx.Client(timeout=120, trust_env=False) as client:
        response = client.get(f"{base_url}/health")
    response.raise_for_status()
    payload = response.json()
    assert payload["success"] is True
    assert payload["tool_name"] == expected_tool
    assert payload["data"]["status"] == "ready"
    assert payload["data"]["model_loaded"] is True
    return payload


def timed_call(name: str, call, report: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = call()
    report[name] = {
        "passed": True,
        "wall_time_ms": round((time.perf_counter() - started) * 1000),
        "request_id": result.get("request_id"),
        "execution_time_ms": result.get("execution_time_ms"),
    }
    return result


def run(
    image_path: Path,
    report_path: Path | None,
    *,
    aoi_url: str,
    reflow_url: str,
    spi_url: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "health": {},
        "tools": {},
        "errors": {},
        "http_errors": {},
    }

    report["health"]["aoi"] = assert_health(aoi_url, "pcba_defect_classification")
    report["health"]["reflow"] = assert_health(reflow_url, "reflow_profile_prediction")
    report["health"]["spi"] = assert_health(spi_url, "spi_vte_prediction")

    aoi_tool = PCBADefectClassificationTool(aoi_url, timeout_s=120)
    spi_predict = SPIVTEPredictionTool(spi_url, timeout_s=120)
    spi_optimize = SPIParameterOptimizationTool(spi_url, timeout_s=240)
    reflow_predict = ReflowProfilePredictionTool(reflow_url, timeout_s=180)
    reflow_optimize = ReflowParameterOptimizationTool(reflow_url, timeout_s=240)

    aoi_result = timed_call(
        "pcba_defect_classification",
        lambda: aoi_tool.invoke(
            {"image_path": str(image_path), "request_id": "E2E-AOI-001", "top_k": 3}
        ),
        report["tools"],
    )
    assert len(aoi_result["data"]["top_k"]) == 3

    printing_parameters = {
        "squeegee_pressure_kgf": 8.0,
        "squeegee_speed_m_s": 30.0,
        "separation_speed_m_s": 2.0,
        "separation_distance_mm": 2.0,
    }
    timed_call(
        "spi_vte_prediction",
        lambda: spi_predict.invoke(
            {"request_id": "E2E-SPI-PRED-001", "input": printing_parameters}
        ),
        report["tools"],
    )
    spi_optimized = timed_call(
        "spi_parameter_optimization",
        lambda: spi_optimize.invoke(
            {
                "request_id": "E2E-SPI-OPT-001",
                "input": {"current_parameters": printing_parameters},
            }
        ),
        report["tools"],
    )
    spi_verify = spi_predict.invoke(
        {
            "request_id": "E2E-SPI-VERIFY-001",
            "input": spi_optimized["data"]["recommended_parameters"],
        }
    )
    assert math.isclose(
        spi_verify["data"]["vte_mean"],
        spi_optimized["data"]["after"]["predicted_vte"],
        abs_tol=1e-4,
    )
    report["tools"]["spi_parameter_optimization"]["verified_by_prediction"] = True

    point = {
        "point_id": "P1",
        "component_x_mm": 117.729,
        "component_y_mm": 77.3908,
        "component_volume_mm3": 107,
    }
    current_parameters = {
        "zone_means_c": [
            135, 155, 165, 173, 180, 180, 190,
            210, 220, 230, 255, 270, 265,
        ],
        "belt_speed_cm_min": 95,
    }
    prediction_payload = {
        "request_id": "E2E-REFLOW-PRED-001",
        "input": {"points": [point], **current_parameters},
        "options": {"return_temperature_curve": False},
    }
    timed_call(
        "reflow_profile_prediction",
        lambda: reflow_predict.invoke(prediction_payload),
        report["tools"],
    )
    reflow_optimized = timed_call(
        "reflow_parameter_optimization",
        lambda: reflow_optimize.invoke(
            {
                "request_id": "E2E-REFLOW-OPT-001",
                "input": {
                    "points": [point],
                    "current_parameters": current_parameters,
                    "optimization_target": {"mode": "minimize_pwi"},
                    "adjustable_parameters": {
                        "zone_indexes": [8, 9, 10, 11, 12, 13],
                        "adjust_belt_speed": True,
                    },
                },
            }
        ),
        report["tools"],
    )
    recommended = reflow_optimized["data"]["recommended_parameters"]
    reflow_verify = reflow_predict.invoke(
        {
            "request_id": "E2E-REFLOW-VERIFY-001",
            "input": {"points": [point], **recommended},
            "options": {"return_temperature_curve": False},
        }
    )
    verified_metrics = reflow_verify["data"]["point_results"][0]["metrics"]
    assert math.isclose(
        verified_metrics["pwi"],
        reflow_optimized["data"]["after"]["max_pwi"],
        abs_tol=0.01,
    )
    report["tools"]["reflow_parameter_optimization"]["verified_by_prediction"] = True

    invalid_calls = [
        (aoi_url, "/api/v1/classify", "pcba_defect_classification"),
        (spi_url, "/api/v1/tools/spi/predict", "spi_vte_prediction"),
        (spi_url, "/api/v1/tools/spi/optimize", "spi_parameter_optimization"),
        (reflow_url, "/api/v1/reflow-profile/predict", "reflow_profile_prediction"),
        (reflow_url, "/api/v1/reflow-profile/optimize", "reflow_parameter_optimization"),
    ]
    for base_url, route, tool_name in invalid_calls:
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(f"{base_url}{route}", json={})
        payload = assert_error_schema(response, tool_name)
        report["errors"][tool_name] = {
            "passed": True,
            "status_code": response.status_code,
            "error_code": payload["error"]["code"],
        }

    unknown_routes = [
        (aoi_url, "pcba_defect_classification"),
        (spi_url, "spi_vte_prediction"),
        (reflow_url, "reflow_profile_prediction"),
    ]
    for base_url, tool_name in unknown_routes:
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.get(f"{base_url}/api/v1/unknown-route")
        payload = assert_error_schema(response, tool_name, expected_status=404)
        report["http_errors"][tool_name] = {
            "passed": True,
            "status_code": response.status_code,
            "error_code": payload["error"]["code"],
        }

    report["passed"] = True
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--aoi-url", default="http://127.0.0.1:8000")
    parser.add_argument("--reflow-url", default="http://127.0.0.1:8001")
    parser.add_argument("--spi-url", default="http://127.0.0.1:8002")
    args = parser.parse_args()
    result = run(
        args.image.resolve(),
        args.report.resolve() if args.report else None,
        aoi_url=args.aoi_url.rstrip("/"),
        reflow_url=args.reflow_url.rstrip("/"),
        spi_url=args.spi_url.rstrip("/"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
