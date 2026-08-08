# cropped2.png

```json
{
  "success": true,
  "request_id": "REQ-API-0001",
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 13,
  "data": {
    "status": "classified",
    "predicted_class": {
      "class_id": 2,
      "class_name_en": "normal",
      "class_name_zh": "正常"
    },
    "confidence": 0.988449,
    "top_k": [
      {
        "class_id": 2,
        "class_name_en": "normal",
        "class_name_zh": "正常",
        "confidence": 0.988449
      },
      {
        "class_id": 0,
        "class_name_en": "excessive_solder",
        "class_name_zh": "焊锡过量",
        "confidence": 0.003859
      },
      {
        "class_id": 4,
        "class_name_en": "short",
        "class_name_zh": "短路/连锡",
        "confidence": 0.002942
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

# excessive (1).png

```json
{
  "success": true,
  "request_id": "REQ-API-0002",
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 15,
  "data": {
    "status": "classified",
    "predicted_class": {
      "class_id": 0,
      "class_name_en": "excessive_solder",
      "class_name_zh": "焊锡过量"
    },
    "confidence": 0.972767,
    "top_k": [
      {
        "class_id": 0,
        "class_name_en": "excessive_solder",
        "class_name_zh": "焊锡过量",
        "confidence": 0.972767
      },
      {
        "class_id": 1,
        "class_name_en": "insufficient_solder",
        "class_name_zh": "焊锡不足",
        "confidence": 0.011036
      },
      {
        "class_id": 4,
        "class_name_en": "short",
        "class_name_zh": "短路/连锡",
        "confidence": 0.00649
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

# insufficient (1).png

```json
{
  "success": true,
  "request_id": "REQ-API-0003",
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 16,
  "data": {
    "status": "classified",
    "predicted_class": {
      "class_id": 1,
      "class_name_en": "insufficient_solder",
      "class_name_zh": "焊锡不足"
    },
    "confidence": 0.985138,
    "top_k": [
      {
        "class_id": 1,
        "class_name_en": "insufficient_solder",
        "class_name_zh": "焊锡不足",
        "confidence": 0.985138
      },
      {
        "class_id": 3,
        "class_name_en": "shifted_component",
        "class_name_zh": "元件偏移",
        "confidence": 0.005309
      },
      {
        "class_id": 2,
        "class_name_en": "normal",
        "class_name_zh": "正常",
        "confidence": 0.004386
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

# short (1).png

```json
{
  "success": true,
  "request_id": "REQ-API-0004",
  "api_version": "v1",
  "tool_name": "pcba_defect_classification",
  "tool_version": "0.1.0",
  "model_name": "efficientnet_b0",
  "model_version": "0.1.0",
  "execution_time_ms": 14,
  "data": {
    "status": "classified",
    "predicted_class": {
      "class_id": 4,
      "class_name_en": "short",
      "class_name_zh": "短路/连锡"
    },
    "confidence": 0.97537,
    "top_k": [
      {
        "class_id": 4,
        "class_name_en": "short",
        "class_name_zh": "短路/连锡",
        "confidence": 0.97537
      },
      {
        "class_id": 3,
        "class_name_en": "shifted_component",
        "class_name_zh": "元件偏移",
        "confidence": 0.007557
      },
      {
        "class_id": 0,
        "class_name_en": "excessive_solder",
        "class_name_zh": "焊锡过量",
        "confidence": 0.006954
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
