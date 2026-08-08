
# 回流焊温度曲线与工艺指标预测接口

## 1. 功能说明

接口输入：

```text
测点位置/体积 + 13 个温区平均温度 + 链速
```

接口输出：

```text
各测点温度曲线 + Peak + TAL + 升降温指标 + PWI + 总体合格结论
```

服务同时加载以下两个交付模型：

- `models/reflow_gpr_delivery_model.joblib`：预测 7 项工艺指标。
- `models/reflow_curve_bspline_lightgbm_model.joblib`：预测完整温度曲线。

服务不依赖数据库，可作为独立目录部署。

## 2. 目录结构

```text
reflow/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ schemas.py
│  ├─ predictor.py
│  ├─ optimizer.py
│  ├─ metrics.py
│  └─ training_domain.py
├─ models/
│  ├─ reflow_gpr_delivery_model.joblib
│  ├─ reflow_curve_bspline_lightgbm_model.joblib
│  └─ training_domain.json
├─ requirements.txt
└─ README.md
```

## 3. 环境要求

- 推荐 Python 3.11 64 位。
- 模型验证环境的主要依赖版本已经固定在 `requirements.txt` 中。
- Windows、Linux 均可运行。

## 4. 安装与启动

项目统一使用 Conda 环境 `PCB_Agent`，不为本服务创建独立虚拟环境。从项目根目录执行：

```powershell
conda activate PCB_Agent
cd "E:\PCBA智能体"
python -m uvicorn tool.services.reflow.app.main:app --host 0.0.0.0 --port 8001
```

启动成功后可访问：

- Swagger：`http://127.0.0.1:8001/docs`
- OpenAPI JSON：`http://127.0.0.1:8001/openapi.json`
- 健康检查：`http://127.0.0.1:8001/health`

## 5. 健康检查

```http
GET /health
```

响应示例：

```json
{
  "success": true,
  "request_id": null,
  "api_version": "v1",
  "tool_name": "reflow_profile_prediction",
  "tool_version": "0.1.0",
  "model_name": "reflow_curve_model",
  "model_version": "0.1.0",
  "execution_time_ms": 0,
  "data": {
    "status": "ready",
    "model_loaded": true,
    "route_tcs": ["TC2", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9"],
    "curve_sample_interval_s": 0.25,
    "training_domain_source": "2_14-8_实验方案_69组_递增_5_V2.xlsx"
  },
  "warnings": [],
  "error": null
}
```

## 6. 预测接口

```http
POST /api/v1/reflow-profile/predict
Content-Type: application/json
```

### 6.1 请求示例

```json
{
  "request_id": "REQ-REFLOW-PRED-0001",
  "input": {
    "points": [
      {
        "point_id": "P1",
        "component_x_mm": 117.729,
        "component_y_mm": 77.3908,
        "component_volume_mm3": 107
      }
    ],
    "zone_means_c": [
      135, 155, 165, 173, 180, 180, 190,
      210, 220, 230, 255, 270, 265
    ],
    "belt_speed_cm_min": 95
  },
  "options": {
    "return_temperature_curve": true,
    "curve_downsample_interval_s": 0.25
  }
}
```

### 6.2 请求字段

| 字段                                    | 类型       | 必填 | 说明                             |
| --------------------------------------- | ---------- | ---: | -------------------------------- |
| `request_id`                          | string     |   是 | 请求唯一标识，响应中原样返回     |
| `input.points`                        | object[]   |   是 | 至少包含一个测点                 |
| `input.points[].point_id`             | string     |   是 | 测点唯一标识；同一请求内不得重复 |
| `input.points[].component_x_mm`       | number     |   是 | 器件 X 坐标，mm                  |
| `input.points[].component_y_mm`       | number     |   是 | 器件 Y 坐标，mm                  |
| `input.points[].component_volume_mm3` | number     |   是 | 器件体积，mm³，不能为负数       |
| `input.zone_means_c`                  | number[13] |   是 | 按 Z1 到 Z13 顺序传入            |
| `input.belt_speed_cm_min`             | number     |   是 | 链速，cm/min，必须大于 0         |
| `options.return_temperature_curve`    | boolean    |   否 | 默认`true`                     |
| `options.curve_downsample_interval_s` | number     |   否 | 默认且最小为`0.25` 秒          |

请求顶层、`input`、测点和 `options` 均不接受未定义的扩展字段。

### 6.3 成功响应示例

下面的曲线数组仅节选前三个采样点；真实响应返回完整数组。

```json
{
  "success": true,
  "request_id": "REQ-REFLOW-PRED-0001",
  "api_version": "v1",
  "tool_name": "reflow_profile_prediction",
  "tool_version": "0.1.0",
  "model_name": "reflow_curve_model",
  "model_version": "0.1.0",
  "execution_time_ms": 235,
  "data": {
    "point_results": [
      {
        "point_id": "P1",
        "matched_tc": "TC6",
        "matched_ref": "U26",
        "metrics": {
          "heating_slope_40_150_c_per_s": 1.850845,
          "heating_slope_200_217_c_per_s": 0.588755,
          "max_cooling_slope_c_per_s": -2.758467,
          "preheat_time_40_150_s": 93.736118,
          "soak_time_150_200_s": 108.56894,
          "time_above_217_s": 64.049919,
          "peak_temperature_c": 245.025046,
          "pwi": 75.85,
          "status": "qualified"
        },
        "temperature_curve": {
          "duration_s": 390.947368,
          "sample_interval_s": 0.25,
          "time_s": [0.0, 0.25, 0.5],
          "temperature_c": [27.451636, 27.583134, 27.720419]
        }
      }
    ],
    "overall": {
      "max_pwi": 75.85,
      "qualified": true,
      "worst_point_id": "P1"
    },
    "within_training_domain": true
  },
  "warnings": [],
  "error": null
}
```

实际模型验证中，上述输入返回 `matched_tc=TC6`、`matched_ref=U26`、`pwi=75.85`，曲线总时长约 `390.947368` 秒，共 1565 个 0.25 秒采样点（包含最终时刻）。

### 6.4 多测点调用

`points` 可同时传入多个测点。每个输入测点对应一条 `point_results`，顺序与输入顺序一致，并通过 `point_id` 关联。

```json
"points": [
  {
    "point_id": "P1",
    "component_x_mm": 117.729,
    "component_y_mm": 77.3908,
    "component_volume_mm3": 107
  },
  {
    "point_id": "P2",
    "component_x_mm": 26.035,
    "component_y_mm": 28.321,
    "component_volume_mm3": 12
  }
]
```

`overall.max_pwi` 是所有测点中的最大 PWI，`worst_point_id` 指向该测点。只有所有测点均合格时，`overall.qualified` 才为 `true`。

## 7. 温度曲线选项

模型原始曲线采样间隔为 0.25 秒。

- `return_temperature_curve=true`：返回温度曲线。
- `return_temperature_curve=false`：不执行曲线推理，`temperature_curve` 返回 `null`。
- `curve_downsample_interval_s=0.25`：返回原始采样间隔。
- 大于 0.25 秒：通过原始预测曲线线性插值得到指定输出时间点。
- 小于 0.25 秒：返回 HTTP 422 参数校验错误。

每条曲线始终满足：

```text
time_s.length == temperature_c.length
time_s[i] 对应 temperature_c[i]
```

最后一个时间点固定为物理过炉总时长，因此最后一段时间间隔可能小于配置的降采样间隔。

## 8. TC 匹配规则

`matched_tc` 和 `matched_ref` 是模型内部真实路由结果。TC2–TC9 的匹配顺序为：

1. 优先选择 `log1p(component_volume_mm3)` 最接近的测温点。
2. 体积距离相同时，再选择 X/Y 空间距离最近的测温点。

## 9. PWI 和合格判定

### 9.1 工艺上下限

| 指标                              |   下限 |   上限 | 是否参与 PWI |
| --------------------------------- | -----: | -----: | :----------: |
| `heating_slope_40_150_c_per_s`  |      0 |    2.5 |      是      |
| `heating_slope_200_217_c_per_s` | 未定义 | 未定义 |      否      |
| `max_cooling_slope_c_per_s`     |     -3 |     -1 |      是      |
| `preheat_time_40_150_s`         |     60 |    150 |      是      |
| `soak_time_150_200_s`           |     60 |    120 |      是      |
| `time_above_217_s`              |     35 |     90 |      是      |
| `peak_temperature_c`            |    230 |    250 |      是      |

### 9.2 计算公式

```text
规格中心 = (上限 + 下限) / 2
半规格宽度 = (上限 - 下限) / 2
单项 PWI = abs(预测值 - 规格中心) / 半规格宽度 × 100
测点 PWI = 6 个受控指标的单项 PWI 最大值
```

- PWI 小于 100：位于规格范围内。
- PWI 等于 100：到达规格边界，仍按合格处理。
- PWI 大于 100：至少一项指标越界，判定不合格。

计算过程使用模型输出的完整精度，响应中的测点 PWI 四舍五入到两位小数，`status` 根据响应中的两位小数 PWI 判定。

## 10. 训练域判断

`within_training_domain` 依据 `2_14-8_实验方案_69组_递增_5_V2.xlsx` 的 69 组训练工况判断。判断对象是实际进入两个回归模型的 14 个特征：Z1–Z13 和链速。

| 特征 |      训练范围 |
| ---- | ------------: |
| Z1   |    135–145℃ |
| Z2   |    145–155℃ |
| Z3   |    155–165℃ |
| Z4   |    165–173℃ |
| Z5   |    173–180℃ |
| Z6   |    180–190℃ |
| Z7   |    190–200℃ |
| Z8   |    210–220℃ |
| Z9   |    220–230℃ |
| Z10  |    230–240℃ |
| Z11  |    245–255℃ |
| Z12  |    260–270℃ |
| Z13  |    255–265℃ |
| 链速 | 85–95 cm/min |

所有特征均在各自范围内（包含边界）时返回 `true`。任一特征越界时仍执行预测，但返回 `false`，并在 `warnings` 中列出越界字段，例如：

```json
{
  "within_training_domain": false,
  "warnings": [
    "Z1=150℃ 超出训练范围 [135, 145]℃"
  ]
}
```

测点坐标和体积仅用于 TC 路由，不进入工艺指标或曲线回归模型，因此不参与该字段判断。

## 11. 错误响应

请求参数不合法时返回 HTTP 422；预测输入逻辑错误返回 HTTP 400；模型执行异常返回 HTTP 500。错误响应统一使用以下外壳：

```json
{
  "success": false,
  "request_id": "REQ-INVALID",
  "api_version": "v1",
  "tool_name": "reflow_profile_prediction",
  "tool_version": "0.1.0",
  "model_name": "reflow_curve_model",
  "model_version": "0.1.0",
  "execution_time_ms": 0,
  "data": null,
  "warnings": [],
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数校验失败",
    "details": []
  }
}
```

## 12. 测试调用样例

### 12.1 PowerShell

服务启动后，在另一个 PowerShell 窗口执行：

```powershell
$body = @{
    request_id = "REQ-REFLOW-PRED-0001"
    input = @{
        points = @(
            @{
                point_id = "P1"
                component_x_mm = 117.729
                component_y_mm = 77.3908
                component_volume_mm3 = 107
            }
        )
        zone_means_c = @(135, 155, 165, 173, 180, 180, 190, 210, 220, 230, 255, 270, 265)
        belt_speed_cm_min = 95
    }
    options = @{
        return_temperature_curve = $true
        curve_downsample_interval_s = 0.25
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8001/api/v1/reflow-profile/predict" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body $body
```

### 12.2 Python 标准库

该样例不需要额外安装 `requests`：

```python
import json
from urllib.request import Request, urlopen

payload = {
    "request_id": "REQ-REFLOW-PRED-0001",
    "input": {
        "points": [
            {
                "point_id": "P1",
                "component_x_mm": 117.729,
                "component_y_mm": 77.3908,
                "component_volume_mm3": 107,
            }
        ],
        "zone_means_c": [
            135, 155, 165, 173, 180, 180, 190,
            210, 220, 230, 255, 270, 265,
        ],
        "belt_speed_cm_min": 95,
    },
    "options": {
        "return_temperature_curve": True,
        "curve_downsample_interval_s": 0.25,
    },
}

request = Request(
    "http://127.0.0.1:8001/api/v1/reflow-profile/predict",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json; charset=utf-8"},
    method="POST",
)
with urlopen(request) as response:
    result = json.load(response)

print(json.dumps(result, ensure_ascii=False, indent=2))
```

## 13. 停止服务

在运行 Uvicorn 的终端中按 `Ctrl+C`。

## 14. 单测点参数优化接口

```http
POST /api/v1/reflow-profile/optimize
Content-Type: application/json
```

优化接口使用当前已加载的指标模型和整数粒子群算法，在模型训练域内搜索推荐的 13 温区和链速。第一版只支持一个测点，因此 `input.points` 必须且只能包含一个元素。

### 14.1 优化目标

支持以下三种模式：

| `mode`                    | 主要目标                              | `target_reached` 条件           |
| --------------------------- | ------------------------------------- | --------------------------------- |
| `minimize_pwi`            | 最小化单测点 6 项受控指标中的最大 PWI | 优化后`max_pwi <= 100`          |
| `target_peak_temperature` | Peak 进入目标容差，再最小化最大 PWI   | Peak 进入容差且`max_pwi <= 100` |
| `target_time_above_217`   | TAL 进入目标容差，再最小化最大 PWI    | TAL 进入容差且`max_pwi <= 100`  |

单测点 PWI 优化目标为：

```text
max_pwi = max(
  40–150℃升温斜率 PWI,
  最大降温斜率 PWI,
  预热时间 PWI,
  恒温时间 PWI,
  TAL PWI,
  峰值温度 PWI
)

优化目标 = minimize(max_pwi)
```

`200–217℃升温斜率`不参与 PWI。优化算法不再包含原多测点算法中的平均 PWI 或多测点鲁棒损失。

### 14.2 最小化 PWI 请求

```json
{
  "request_id": "REQ-REFLOW-OPT-0001",
  "input": {
    "points": [
      {
        "point_id": "P1",
        "component_x_mm": 117.729,
        "component_y_mm": 77.3908,
        "component_volume_mm3": 107
      }
    ],
    "current_parameters": {
      "zone_means_c": [
        135, 155, 165, 173, 180, 180, 190,
        210, 220, 230, 255, 270, 265
      ],
      "belt_speed_cm_min": 95
    },
    "optimization_target": {
      "mode": "minimize_pwi"
    },
    "adjustable_parameters": {
      "zone_indexes": [8, 9, 10, 11, 12, 13],
      "adjust_belt_speed": true
    }
  }
}
```

### 14.3 目标峰值温度请求

只需把 `optimization_target` 改为：

```json
{
  "mode": "target_peak_temperature",
  "target_value_c": 242,
  "tolerance_c": 2
}
```

要求：

- `target_value_c` 必须位于 230–250℃。
- `tolerance_c` 必须大于 0。
- 不得同时传入 TAL 目标字段。

### 14.4 目标 TAL 请求

```json
{
  "mode": "target_time_above_217",
  "target_value_s": 60,
  "tolerance_s": 5
}
```

要求：

- `target_value_s` 必须位于 35–90 秒。
- `tolerance_s` 必须大于 0。
- 不得同时传入峰值温度目标字段。

### 14.5 可调参数规则

`zone_indexes` 使用从 1 开始的温区编号：

```text
1 表示 Z1
...
13 表示 Z13
```

- 编号范围只能是 1–13，且不能重复。
- 未列入 `zone_indexes` 的温区保持当前值。
- `adjust_belt_speed=true` 时允许优化链速，否则链速保持当前值。
- 至少需要允许调整一个温区或链速。
- 可调参数的推荐值为整数。
- 不可调参数如果超出训练域，接口返回 HTTP 400。
- 如果固定参数导致温区趋势约束无可行解，接口返回 HTTP 400。

推荐参数必须满足：

```text
Z1 <= Z2 <= ... <= Z12
Z13 <= Z12
```

相邻温区允许相等。搜索范围采用第 10 节列出的 69 组训练数据范围，成功响应中的 `within_training_domain` 必须为 `true`。

算法内部参数固定为：

```text
swarm_size = 50
max_iterations = 100
random_seed = 42
```

因此相同模型、相同输入会得到可复现的结果。

### 14.6 成功响应示例

以下是交付模型对第 14.2 节请求的实际测试结果；`execution_time_ms` 会随机器性能变化。

```json
{
  "success": true,
  "request_id": "REQ-REFLOW-OPT-0001",
  "api_version": "v1",
  "tool_name": "reflow_parameter_optimization",
  "tool_version": "0.1.0",
  "model_name": "reflow_pso_optimizer",
  "model_version": "0.1.0",
  "execution_time_ms": 1261,
  "data": {
    "optimization_mode": "minimize_pwi",
    "before": {
      "zone_means_c": [
        135, 155, 165, 173, 180, 180, 190,
        210, 220, 230, 255, 270, 265
      ],
      "belt_speed_cm_min": 95,
      "max_pwi": 75.85,
      "peak_temperature_c": 245.025046,
      "time_above_217_s": 64.049919
    },
    "recommended_parameters": {
      "zone_means_c": [
        135, 155, 165, 173, 180, 180, 190,
        214, 222, 236, 246, 268, 261
      ],
      "belt_speed_cm_min": 93
    },
    "after": {
      "max_pwi": 62.41,
      "peak_temperature_c": 242.8966,
      "time_above_217_s": 67.761391
    },
    "target_reached": true,
    "within_training_domain": true
  },
  "warnings": [],
  "error": null
}
```

### 14.7 PowerShell 调用样例

服务启动后，在另一个 PowerShell 窗口执行：

```powershell
$body = @{
    request_id = "REQ-REFLOW-OPT-0001"
    input = @{
        points = @(
            @{
                point_id = "P1"
                component_x_mm = 117.729
                component_y_mm = 77.3908
                component_volume_mm3 = 107
            }
        )
        current_parameters = @{
            zone_means_c = @(135, 155, 165, 173, 180, 180, 190, 210, 220, 230, 255, 270, 265)
            belt_speed_cm_min = 95
        }
        optimization_target = @{
            mode = "minimize_pwi"
        }
        adjustable_parameters = @{
            zone_indexes = @(8, 9, 10, 11, 12, 13)
            adjust_belt_speed = $true
        }
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8001/api/v1/reflow-profile/optimize" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body $body
```

### 14.8 优化错误响应

请求字段校验错误返回 HTTP 422。输入虽然通过字段校验，但不可调参数越界或约束无可行解时返回 HTTP 400：

```json
{
  "success": false,
  "request_id": "REQ-REFLOW-OPT-INVALID",
  "api_version": "v1",
  "tool_name": "reflow_parameter_optimization",
  "tool_version": "0.1.0",
  "model_name": "reflow_pso_optimizer",
  "model_version": "0.1.0",
  "execution_time_ms": 1,
  "data": null,
  "warnings": [],
  "error": {
    "code": "OPTIMIZATION_INPUT_ERROR",
    "message": "不可调参数 Z1=150 超出训练范围 [135, 145]",
    "details": null
  }
}
```
