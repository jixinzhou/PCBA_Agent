# PCBA 缺陷分类接口文档

## 1. 接口启动

在 PowerShell 中执行：

```powershell
conda activate PCB_Agent
cd "E:\PCBA智能体"
python -m uvicorn tool.services.aoi.app.main:app --host 0.0.0.0 --port 8000
```

当控制台显示 `Application startup complete` 时，表示模型已经加载完成，接口可以接收请求。

启动后的访问地址：

| 地址 | 含义 |
|---|---|
| `http://127.0.0.1:8000/api/v1/classify` | PCBA 图片分类接口 |
| `http://127.0.0.1:8000/health` | 服务及模型健康检查 |
| `http://127.0.0.1:8000/docs` | Swagger 在线接口页面 |
| `http://127.0.0.1:8000/redoc` | ReDoc 在线接口页面 |

使用本机以外的设备调用时，将 `127.0.0.1` 替换为运行接口服务的计算机 IP 地址。

需要停止接口时，在运行 Uvicorn 的 PowerShell 窗口中按 `Ctrl+C`。

## 2. 图片分类接口

### 接口

```text
POST /api/v1/classify
Content-Type: multipart/form-data
```

### 输入

| 字段 | 类型 | 必填 | 含义 |
|---|---|---:|---|
| `image` | 文件 | 是 | 待分类的 PCBA 图片。只支持 JPG、JPEG、PNG，最大 10 MB。服务会校验扩展名、Content-Type 和真实图片格式。 |
| `request_id` | 字符串 | 是 | 本次请求的唯一标识，长度 1～128 个字符。成功时原样返回，便于关联请求和结果。 |
| `top_k` | 整数 | 否 | 返回置信度最高的候选类别数量，范围 1～5，默认值为 3。 |

输入的统一结构可表示为：

```json
{
  "image": "<上传的 jpg/jpeg/png 图片文件>",
  "request_id": "REQ-IMG-0001",
  "top_k": 3
}
```

实际请求中，图片和其他字段都使用 `multipart/form-data` 传输，`image` 不是 JSON 字符串。

请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/classify" \
  -F "image=@example.png;type=image/png" \
  -F "request_id=REQ-IMG-0001" \
  -F "top_k=3"
```

### 分类成功输出

```json
{
  "success": true,
  "request_id": "REQ-IMG-0001",
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 85,
  "data": {
    "status": "classified",
    "predicted_class": {
      "class_id": 1,
      "class_name_en": "insufficient_solder",
      "class_name_zh": "焊锡不足"
    },
    "confidence": 0.932,
    "top_k": [
      {
        "class_id": 1,
        "class_name_en": "insufficient_solder",
        "class_name_zh": "焊锡不足",
        "confidence": 0.932
      },
      {
        "class_id": 4,
        "class_name_en": "short",
        "class_name_zh": "短路/连锡",
        "confidence": 0.041
      },
      {
        "class_id": 3,
        "class_name_en": "shifted_component",
        "class_name_zh": "元件偏移",
        "confidence": 0.015
      }
    ],
    "low_confidence": false,
    "confidence_threshold": 0.6,
    "label_schema_version": "0.1.0"
  },
  "warnings": [],
  "error": null
}
```

### 低置信度输出

最高候选类别的置信度低于 `0.6` 时，接口仍返回最高候选和 Top-K，供人工复核，但 `status` 为 `manual_review`，不表示服务已确认该分类。

```json
{
  "success": true,
  "request_id": "REQ-IMG-0002",
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 76,
  "data": {
    "status": "manual_review",
    "predicted_class": {
      "class_id": 3,
      "class_name_en": "shifted_component",
      "class_name_zh": "元件偏移"
    },
    "confidence": 0.52,
    "top_k": [
      {
        "class_id": 3,
        "class_name_en": "shifted_component",
        "class_name_zh": "元件偏移",
        "confidence": 0.52
      },
      {
        "class_id": 2,
        "class_name_en": "normal",
        "class_name_zh": "正常",
        "confidence": 0.31
      },
      {
        "class_id": 1,
        "class_name_en": "insufficient_solder",
        "class_name_zh": "焊锡不足",
        "confidence": 0.1
      }
    ],
    "low_confidence": true,
    "confidence_threshold": 0.6,
    "label_schema_version": "0.1.0"
  },
  "warnings": [
    "Confidence is below the threshold; manual review is recommended."
  ],
  "error": null
}
```

### 输出字段含义

| 字段 | 含义 |
|---|---|
| `success` | 接口是否成功完成图片分类。 |
| `request_id` | 调用方传入的请求标识。参数校验失败且无法取得该字段时可能为 `null`。 |
| `api_version` | API 结构版本。 |
| `tool_name` | 工具名称，固定为 `pcba_defect_classification`。 |
| `tool_version` | 工具实现版本。 |
| `model_name` | 使用的模型结构，固定为 `efficientnet_b0`。 |
| `model_version` | 当前模型版本。 |
| `execution_time_ms` | 从接口接收请求到完成推理的耗时，单位为毫秒。 |
| `data.status` | `classified` 表示置信度达到阈值；`manual_review` 表示需要人工复核。 |
| `data.predicted_class` | 模型输出中置信度最高的候选类别。低置信度时它只表示最高候选，不表示最终确认。 |
| `data.confidence` | 最高候选类别的预测概率，范围为 0～1。 |
| `data.top_k` | 按置信度从高到低排列的候选类别，数量由输入 `top_k` 决定。 |
| `data.low_confidence` | 最高置信度是否低于 `confidence_threshold`。 |
| `data.confidence_threshold` | 低置信度判断阈值，当前固定为 0.6。 |
| `data.label_schema_version` | 中英文类别映射版本。 |
| `warnings` | 非阻断警告。低置信度时包含人工复核提示。 |
| `error` | 成功时为 `null`；失败时包含错误代码、说明和可选详情。 |

### 类别映射

| `class_id` | `class_name_en` | `class_name_zh` |
|---:|---|---|
| 0 | `excessive_solder` | 焊锡过量 |
| 1 | `insufficient_solder` | 焊锡不足 |
| 2 | `normal` | 正常 |
| 3 | `shifted_component` | 元件偏移 |
| 4 | `short` | 短路/连锡 |

### 失败输出

```json
{
  "success": false,
  "request_id": "REQ-IMG-0003",
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 0,
  "data": null,
  "warnings": [],
  "error": {
    "code": "UNSUPPORTED_IMAGE_TYPE",
    "message": "Only JPG, JPEG and PNG image files are supported.",
    "details": null
  }
}
```

| HTTP 状态码 | 错误代码 | 含义 |
|---:|---|---|
| 400 | `EMPTY_IMAGE` | 上传文件内容为空。 |
| 400 | `INVALID_IMAGE` | 文件无法被识别或读取为有效图片。 |
| 413 | `IMAGE_TOO_LARGE` | 图片超过 10 MB。 |
| 415 | `UNSUPPORTED_IMAGE_TYPE` | 扩展名、Content-Type 或真实格式不是 JPG/JPEG/PNG。 |
| 422 | `INVALID_REQUEST_ID` | `request_id` 只包含空白字符。 |
| 422 | `VALIDATION_ERROR` | 缺少字段，或 `request_id`、`top_k` 不符合限制。 |
| 500 | `INTERNAL_SERVER_ERROR` | 服务发生未预期错误。 |

## 3. 健康检查接口

### 接口与输入

```text
GET /health
```

该接口不需要输入参数。

### 输出

```json
{
  "success": true,
  "request_id": null,
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 0,
  "data": {
    "status": "ready",
    "model_loaded": true,
    "device": "cuda",
    "class_count": 5
  },
  "warnings": [],
  "error": null
}
```

| 字段 | 含义 |
|---|---|
| `data.status` | `ready` 表示服务已就绪。 |
| `data.model_loaded` | 模型是否已加载完成。 |
| `data.device` | 当前推理设备，值为 `cuda` 或 `cpu`。 |
| `data.class_count` | 模型输出类别数量。 |
