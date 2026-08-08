输入 JSON

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

`optimization_target.mode` 支持：

```text
minimize_pwi

target_peak_temperature

target_time_above_217
```

如果指定目标峰值温度：

```json
{
  "mode": "target_peak_temperature",
  "target_value_c": 242,
  "tolerance_c": 2
}
```

如果指定 TAL：

```json
{
  "mode": "target_time_above_217",
  "target_value_s": 60,
  "tolerance_s": 5
}
```

### 输出 JSON

```json
{
  "success": true,
  "request_id": "REQ-REFLOW-OPT-0001",
  "api_version": "v1",
  "tool_name": "reflow_parameter_optimization",
  "tool_version": "0.1.0",
  "model_name": "reflow_pso_optimizer",
  "model_version": "0.1.0",
  "execution_time_ms": 1850,

  "data": {
    "optimization_mode": "minimize_pwi",

    "before": {
      "zone_means_c": [
        135, 155, 165, 173, 180, 180, 190,
        210, 220, 230, 255, 270, 265
      ],
      "belt_speed_cm_min": 95,
      "max_pwi": 118.3,
      "peak_temperature_c": 232.4,
      "time_above_217_s": 41.2
    },

    "recommended_parameters": {
      "zone_means_c": [
        135, 155, 165, 173, 180, 180, 190,
        212, 224, 235, 258, 270, 265
      ],
      "belt_speed_cm_min": 92
    },

    "after": {
      "max_pwi": 76.5,
      "peak_temperature_c": 241.3,
      "time_above_217_s": 58.7
    },

    "target_reached": true,
    "within_training_domain": true
  },

  "warnings": [],
  "error": null
}
```

第一版就保持这个结构：**当前参数 + 测点 + 优化目标 → 推荐13温区和链速 → 返回优化前后 PWI、Peak、TAL 对比。**
