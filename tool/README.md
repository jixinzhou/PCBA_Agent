# PCBA 模型服务与 Agent Tools

本目录是 5 个核心 Tool 的唯一正式实现入口。模型只能经由 FastAPI 和 Agent HTTP Tool 调用，不从 Agent 直接导入模型代码。

## 目录

```text
tool/
├─ common/                 # 三套服务共用的健康检查和错误响应 Schema
├─ services/
│  ├─ aoi/                 # 缺陷图片分类服务
│  ├─ spi/                 # SPI VTE 预测与参数优化服务
│  └─ reflow/              # 回流焊预测与参数优化服务
├─ agent_tools/            # 5 个框架无关的 Agent HTTP Tool
├─ scripts/                # 统一服务启动入口
├─ tests/e2e/              # 真实 HTTP 端到端测试
└─ requirements.txt        # PCB_Agent 已验证依赖版本
```

旧重复实现保存在 `archive/legacy_tools`，不再作为运行或修改入口。`数据资料`中的原始交付材料不属于正式服务代码。

## 服务地址

| 服务 | 基础 URL | Tool | 业务路由 |
|---|---|---|---|
| AOI | `http://127.0.0.1:8000` | `pcba_defect_classification` | `POST /api/v1/classify` |
| 回流焊 | `http://127.0.0.1:8001` | `reflow_profile_prediction` | `POST /api/v1/reflow-profile/predict` |
| 回流焊 | `http://127.0.0.1:8001` | `reflow_parameter_optimization` | `POST /api/v1/reflow-profile/optimize` |
| SPI | `http://127.0.0.1:8002` | `spi_vte_prediction` | `POST /api/v1/tools/spi/predict` |
| SPI | `http://127.0.0.1:8002` | `spi_parameter_optimization` | `POST /api/v1/tools/spi/optimize` |

三套服务都提供 `GET /health`、`GET /docs` 和 `GET /openapi.json`。

## 启动

所有命令都从项目根目录运行，并统一使用 Conda 环境 `PCB_Agent`。分别打开三个 PowerShell：

```powershell
.\tool\scripts\start_service.ps1 -Service aoi
.\tool\scripts\start_service.ps1 -Service reflow
.\tool\scripts\start_service.ps1 -Service spi
```

等价的直接命令：

```powershell
conda run -n PCB_Agent python -m uvicorn tool.services.aoi.app.main:app --host 0.0.0.0 --port 8000
conda run -n PCB_Agent python -m uvicorn tool.services.reflow.app.main:app --host 0.0.0.0 --port 8001
conda run -n PCB_Agent python -m uvicorn tool.services.spi.app.main:app --host 0.0.0.0 --port 8002
```

若端口被其他本机程序临时占用，可以用 `-Port` 覆盖启动端口，并通过对应的 `PCBA_*_BASE_URL` 环境变量通知 Agent Tool，例如：

```powershell
.\tool\scripts\start_service.ps1 -Service aoi -Port 18000
$env:PCBA_AOI_BASE_URL = "http://127.0.0.1:18000"
```

## 统一错误 Schema

所有 HTTP 4xx/5xx 响应使用完全相同的字段：

```json
{
  "success": false,
  "request_id": "REQ-001",
  "api_version": "v1",
  "tool_name": "spi_vte_prediction",
  "tool_version": "0.1.0",
  "model_name": "spi_gpr",
  "model_version": "0.1.0",
  "execution_time_ms": 1,
  "data": null,
  "warnings": [],
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数校验失败",
    "details": []
  }
}
```

`error.details` 始终存在；没有更多信息时为 `null`。

## Agent Tool

```python
from tool.agent_tools import TOOL_REGISTRY

result = TOOL_REGISTRY["spi_vte_prediction"].invoke(
    {
        "input": {
            "squeegee_pressure_kgf": 8.0,
            "squeegee_speed_m_s": 30.0,
            "separation_speed_m_s": 2.0,
            "separation_distance_mm": 2.0,
        }
    }
)
```

可通过环境变量覆盖地址：`PCBA_AOI_BASE_URL`、`PCBA_REFLOW_BASE_URL`、`PCBA_SPI_BASE_URL`。

## 测试

端到端测试说明见 `tests/e2e/README.md`，最近一次正式结果见 `tests/e2e/VALIDATION_REPORT.md`。测试会真实调用 5 个 Tool、检查 5 个校验错误，并对两个优化结果重新执行预测验证。
