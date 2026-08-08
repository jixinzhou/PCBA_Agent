from typing import Dict, List


EXPECTED_CLASS_NAMES: List[str] = [
    "excessive_solder",
    "insufficient_solder",
    "normal",
    "shifted_component",
    "short",
]

CLASS_NAMES_ZH: Dict[str, str] = {
    "excessive_solder": "焊锡过量",
    "insufficient_solder": "焊锡不足",
    "normal": "正常",
    "shifted_component": "元件偏移",
    "short": "短路/连锡",
}


def validate_class_names(class_names: List[str]) -> None:
    if class_names != EXPECTED_CLASS_NAMES:
        raise ValueError(
            "Checkpoint class_names do not match the service label schema: "
            f"expected {EXPECTED_CLASS_NAMES}, got {class_names}"
        )


def class_name_zh(class_name_en: str) -> str:
    try:
        return CLASS_NAMES_ZH[class_name_en]
    except KeyError as exc:
        raise ValueError(f"Unknown class name: {class_name_en}") from exc

