from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
import pypdfium2 as pdfium
from rapidocr import RapidOCR

from .page_quality import normalize_text


@dataclass(frozen=True)
class OcrResult:
    text: str
    elapsed_seconds: float
    average_score: float | None
    line_count: int
    lines: tuple["OcrLine", ...] = ()


@dataclass(frozen=True)
class OcrLine:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None


def _box_metrics(box: Iterable[Iterable[float]]) -> tuple[float, float, float]:
    points = list(box)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), (min(ys) + max(ys)) / 2, max(ys) - min(ys)


def group_ocr_lines(boxes: np.ndarray, texts: Iterable[str]) -> str:
    items = []
    for box, text in zip(boxes, texts):
        value = str(text).strip()
        if not value:
            continue
        x, center_y, height = _box_metrics(box)
        items.append({"x": x, "y": center_y, "height": max(height, 1.0), "text": value})
    if not items:
        return ""

    typical_height = median(item["height"] for item in items)
    tolerance = max(typical_height * 0.65, 6.0)
    items.sort(key=lambda item: (item["y"], item["x"]))
    rows: list[list[dict[str, float | str]]] = []
    row_centers: list[float] = []
    for item in items:
        if not rows or abs(float(item["y"]) - row_centers[-1]) > tolerance:
            rows.append([item])
            row_centers.append(float(item["y"]))
        else:
            rows[-1].append(item)
            row_centers[-1] = sum(float(entry["y"]) for entry in rows[-1]) / len(rows[-1])

    lines = []
    for row in rows:
        row.sort(key=lambda item: float(item["x"]))
        lines.append(" | ".join(str(item["text"]) for item in row))
    return normalize_text("\n".join(lines))


def build_ocr_lines(
    boxes: np.ndarray,
    texts: Iterable[str],
    scores: Iterable[float],
    image_width: int,
    image_height: int,
) -> tuple[OcrLine, ...]:
    lines: list[OcrLine] = []
    score_values = tuple(float(value) for value in scores)
    for index, (box, text) in enumerate(zip(boxes, texts)):
        value = str(text).strip()
        if not value:
            continue
        points = list(box)
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        bbox = (
            max(0.0, min(1.0, min(xs) / max(image_width, 1))),
            max(0.0, min(1.0, min(ys) / max(image_height, 1))),
            max(0.0, min(1.0, max(xs) / max(image_width, 1))),
            max(0.0, min(1.0, max(ys) / max(image_height, 1))),
        )
        confidence = score_values[index] if index < len(score_values) else None
        lines.append(OcrLine(value, bbox, confidence))
    lines.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return tuple(lines)


def render_page(path: Path, page_index: int, dpi: int):
    document = pdfium.PdfDocument(path)
    page = document[page_index]
    bitmap = page.render(scale=dpi / 72)
    image = bitmap.to_pil().convert("RGB")
    bitmap.close()
    page.close()
    document.close()
    return image


class LocalOcrEngine:
    def __init__(self) -> None:
        self._engine = RapidOCR()

    def recognize_page(self, path: Path, page_index: int, dpi: int) -> OcrResult:
        image = render_page(path, page_index, dpi)
        result = self._engine(np.asarray(image))
        texts = tuple(result.txts or ())
        boxes = result.boxes if result.boxes is not None else np.empty((0, 4, 2))
        scores = tuple(float(score) for score in (result.scores or ()))
        text = group_ocr_lines(boxes, texts)
        average_score = sum(scores) / len(scores) if scores else None
        lines = build_ocr_lines(boxes, texts, scores, image.width, image.height)
        return OcrResult(text, float(result.elapse or 0.0), average_score, len(texts), lines)
