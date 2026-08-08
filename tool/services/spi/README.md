# 焊膏印刷 VTE 预测与参数优化 Tool

## 1. 功能范围

同一个 FastAPI 服务加载一次 VTE 均值 GPR 模型，并对外提供两个接口：

```text
POST /api/v1/tools/spi/predict    # VTE 均值预测
POST /api/v1/tools/spi/optimize   # 工艺参数优化
```

优化器内部反复调用共享的预测器，不会重复加载模型。VTE 方差模型因交叉验证表现不可靠，未在 v1 中使用。

## 2. 文件结构

```text
spi/
├── app/
│   ├── main.py          # FastAPI 接口、请求响应模型及异常处理
│   ├── predictor.py     # 模型与元数据加载、预测及训练域检查
│   └── optimizer.py     # Differential Evolution 参数优化
├── README.md
├── requirements.txt
└── models/
    ├── model_info.json
    └── vte_mean_model.joblib
```

模型只在 `main.py` 的应用生命周期启动阶段加载一次。预测接口和优化接口共享同一个 `VTEPredictor` 实例。

## 3. 模型与优化配置

| 项目 | 内容 |
| --- | --- |
| 模型名称 | `spi_gpr` |
| 模型版本 | `0.1.0` |
| 模型算法 | Gaussian Process Regression |
| 训练样本数 | 31 |
| 5 折汇总 OOF R² | 0.9146 |
| 5 折汇总 OOF MAE | 2.0057 |
| 5 折汇总 OOF RMSE | 2.3522 |
| 优化器 | `differential_evolution` |
| 固定目标 | `target_vte = 100.0` |
| 达标容差 | `±5.0` |
| 随机种子 | `42` |

优化边界、API 字段顺序和模型列名均从 `models/model_info.json` 动态读取：

| API 字段 | 模型字段 | 单位 | 最小值 | 最大值 |
| --- | --- | --- | ---: | ---: |
| `squeegee_pressure_kgf` | 刮刀压力 | kgf | 5.2 | 10.8 |
| `squeegee_speed_m_s` | 刮刀速度 | m/s | 5.7 | 69.3 |
| `separation_speed_m_s` | 脱模速度 | m/s | 0.6 | 3.4 |
| `separation_distance_mm` | 脱模距离 | mm | 0.6 | 3.4 |

训练文件中的压力列记为 kg，本接口按实验数值直接对应到 `kgf` 字段，不额外换算。

优化目标由两部分组成：

```text
abs(predicted_vte - 100.0)
+ 0.25 × 参数相对当前值的归一化均方根改变量
```

VTE 误差是主目标；较小的参数变化惩罚用于在预测效果相近时优先选择更接近当前工艺设置的方案。推荐参数始终限制在模型训练范围内。

## 4. 安装与启动

项目使用 Conda 环境 `PCB_Agent`：

```powershell
conda activate PCB_Agent
cd "E:\PCBA智能体"
python -m uvicorn tool.services.spi.app.main:app --host 0.0.0.0 --port 8002
```

启动后：

- 健康检查：`http://127.0.0.1:8002/health`
- Swagger 文档：`http://127.0.0.1:8002/docs`
- OpenAPI JSON：`http://127.0.0.1:8002/openapi.json`
- 预测接口：`POST http://127.0.0.1:8002/api/v1/tools/spi/predict`
- 优化接口：`POST http://127.0.0.1:8002/api/v1/tools/spi/optimize`

健康检查成功响应：

```json
{
  "success": true,
  "request_id": null,
  "api_version": "v1",
  "tool_name": "spi_vte_prediction",
  "tool_version": "0.1.0",
  "model_name": "spi_gpr",
  "model_version": "0.1.0",
  "execution_time_ms": 0,
  "data": {
    "status": "ready",
    "model_loaded": true,
    "optimizer": "differential_evolution",
    "input_features": [
      "squeegee_pressure_kgf",
      "squeegee_speed_m_s",
      "separation_speed_m_s",
      "separation_distance_mm"
    ]
  },
  "warnings": [],
  "error": null
}
```

## 5. VTE 预测接口

### 请求

请求体必须严格包含 `request_id` 和 `input`，额外字段会被拒绝。

```json
{
  "request_id": "REQ-SPI-PRED-0001",
  "input": {
    "squeegee_pressure_kgf": 8.0,
    "squeegee_speed_m_s": 30.0,
    "separation_speed_m_s": 2.0,
    "separation_distance_mm": 2.0
  }
}
```

### 成功响应

```json
{
  "success": true,
  "request_id": "REQ-SPI-PRED-0001",
  "api_version": "v1",
  "tool_name": "spi_vte_prediction",
  "tool_version": "0.1.0",
  "model_name": "spi_gpr",
  "model_version": "0.1.0",
  "execution_time_ms": 18,
  "data": {
    "vte_mean": 94.6132,
    "vte_unit": "percent",
    "within_training_domain": true
  },
  "warnings": [],
  "error": null
}
```

`vte_mean` 保留 4 位小数，不会将大于 100 的模型结果截断。预测参数超出训练范围时仍执行模型预测，但 `within_training_domain` 为 `false`，并在 `warnings` 中列出越界字段。

## 6. 参数优化接口

### 请求

目标值、容差、算法和边界由服务内部固定或从模型元数据读取，调用方只提交当前参数。

```json
{
  "request_id": "REQ-SPI-OPT-0001",
  "input": {
    "current_parameters": {
      "squeegee_pressure_kgf": 8.0,
      "squeegee_speed_m_s": 30.0,
      "separation_speed_m_s": 2.0,
      "separation_distance_mm": 2.0
    }
  }
}
```

### 成功响应

推荐参数及预测数值均保留 4 位小数。下面的推荐值来自固定随机种子，但如果模型或优化配置更新，结果会相应变化。

```json
{
  "success": true,
  "request_id": "REQ-SPI-OPT-0001",
  "api_version": "v1",
  "tool_name": "spi_parameter_optimization",
  "tool_version": "0.1.0",
  "model_name": "spi_gpr",
  "model_version": "0.1.0",
  "optimizer": "differential_evolution",
  "execution_time_ms": 7823,
  "data": {
    "target_vte": 100.0,
    "tolerance": 5.0,
    "before": {
      "parameters": {
        "squeegee_pressure_kgf": 8.0,
        "squeegee_speed_m_s": 30.0,
        "separation_speed_m_s": 2.0,
        "separation_distance_mm": 2.0
      },
      "predicted_vte": 94.6132
    },
    "recommended_parameters": {
      "squeegee_pressure_kgf": 7.2071,
      "squeegee_speed_m_s": 39.0064,
      "separation_speed_m_s": 1.9523,
      "separation_distance_mm": 2.0468
    },
    "after": {
      "predicted_vte": 99.9998
    },
    "objective_error": 0.0002,
    "target_reached": true,
    "within_training_domain": true
  },
  "warnings": [],
  "error": null
}
```

`before.predicted_vte` 是当前参数的模型预测；`after.predicted_vte` 会使用响应中已经四舍五入的推荐参数重新计算，确保两者严格对应。`objective_error` 为 `abs(after.predicted_vte - target_vte)`。

如果当前参数本身超出训练范围，接口仍可执行优化，但会在 `warnings` 中标出当前参数的越界项；推荐参数仍受训练范围约束。

## 7. 错误响应

缺少字段、类型错误、非有限数值或出现额外字段时返回 HTTP 422，并保持统一外层结构。`tool_name` 会依据当前接口返回预测或优化工具名称。

```json
{
  "success": false,
  "request_id": "REQ-SPI-OPT-0001",
  "api_version": "v1",
  "tool_name": "spi_parameter_optimization",
  "tool_version": "0.1.0",
  "model_name": "spi_gpr",
  "model_version": "0.1.0",
  "execution_time_ms": 1,
  "data": null,
  "warnings": [],
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求 JSON 不符合接口定义，请检查必填字段、字段类型及额外字段。",
    "details": []
  }
}
```

若请求中无法解析出合法的 `request_id`，响应中的 `request_id` 为 `null`。未处理的服务异常返回 HTTP 500 和 `INTERNAL_ERROR`，不会向调用方暴露内部堆栈。

## 8. PowerShell 调用示例

```powershell
$body = @{
    request_id = "REQ-SPI-OPT-0001"
    input = @{
        current_parameters = @{
            squeegee_pressure_kgf = 8.0
            squeegee_speed_m_s = 30.0
            separation_speed_m_s = 2.0
            separation_distance_mm = 2.0
        }
    }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8002/api/v1/tools/spi/optimize" `
    -ContentType "application/json" `
    -Body $body
```
