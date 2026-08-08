from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import efficientnet_b0

from .config import DEVICE, MODEL_NAME, MODEL_PATH
from .label_mapping import class_name_zh, validate_class_names
from .preprocessing import build_inference_transform


def select_device(device_setting: str) -> torch.device:
    value = device_setting.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE is cuda, but CUDA is not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError("DEVICE must be one of: auto, cuda, cpu")


def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class ModelService:
    def __init__(self) -> None:
        self.device = select_device(DEVICE)
        self.model: nn.Module
        self.class_names: List[str]
        self.image_size: int
        self.transform: Any
        self._inference_lock = Lock()
        self._load()

    def _load(self) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

        checkpoint = torch.load(
            str(MODEL_PATH),
            map_location=self.device,
            weights_only=False,
        )
        required_keys = {
            "model_state_dict",
            "model_name",
            "class_names",
            "image_size",
        }
        missing = sorted(required_keys.difference(checkpoint))
        if missing:
            raise ValueError(f"Checkpoint is missing required keys: {missing}")
        if checkpoint["model_name"] != MODEL_NAME:
            raise ValueError(
                f"Expected model_name {MODEL_NAME}, got {checkpoint['model_name']}"
            )

        self.class_names = list(checkpoint["class_names"])
        validate_class_names(self.class_names)
        self.image_size = int(checkpoint["image_size"])
        if self.image_size != 224:
            raise ValueError(f"Expected image_size 224, got {self.image_size}")

        self.model = build_model(len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = build_inference_transform(self.image_size)

    def predict(self, image: Image.Image, top_k: int) -> Dict[str, Any]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._inference_lock, torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
            confidences, indices = torch.topk(probabilities, k=top_k)

        ranked = []
        for confidence, index in zip(
            confidences.detach().cpu().tolist(),
            indices.detach().cpu().tolist(),
        ):
            class_id = int(index)
            class_name_en = self.class_names[class_id]
            ranked.append(
                {
                    "class_id": class_id,
                    "class_name_en": class_name_en,
                    "class_name_zh": class_name_zh(class_name_en),
                    "confidence": round(float(confidence), 6),
                }
            )

        best = ranked[0]
        return {
            "predicted_class": {
                "class_id": best["class_id"],
                "class_name_en": best["class_name_en"],
                "class_name_zh": best["class_name_zh"],
            },
            "confidence": best["confidence"],
            "top_k": ranked,
        }

