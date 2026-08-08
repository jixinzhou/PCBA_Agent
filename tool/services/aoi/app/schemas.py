from typing import List, Optional

from pydantic import BaseModel, Field

from tool.common.schemas import ErrorDetail, HealthResponse


class ClassPrediction(BaseModel):
    class_id: int = Field(..., description="模型类别索引")
    class_name_en: str = Field(..., description="英文类别名称")
    class_name_zh: str = Field(..., description="中文类别名称")


class RankedPrediction(ClassPrediction):
    confidence: float = Field(..., ge=0.0, le=1.0, description="该类别的预测概率")


class ClassificationData(BaseModel):
    status: str = Field(..., description="classified 或 manual_review")
    predicted_class: ClassPrediction = Field(..., description="置信度最高的候选类别")
    confidence: float = Field(..., ge=0.0, le=1.0, description="最高候选类别的置信度")
    top_k: List[RankedPrediction] = Field(..., description="按置信度降序排列的候选类别")
    low_confidence: bool = Field(..., description="最高置信度是否低于阈值")
    confidence_threshold: float = Field(..., description="低置信度判断阈值")
    label_schema_version: str = Field(..., description="标签映射版本")


class ClassificationResponse(BaseModel):
    success: bool
    request_id: Optional[str]
    api_version: str
    tool_name: str
    tool_version: str
    model_name: str
    model_version: str
    execution_time_ms: int
    data: Optional[ClassificationData]
    warnings: List[str]
    error: Optional[ErrorDetail]


