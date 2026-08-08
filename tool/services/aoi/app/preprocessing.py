from typing import Tuple

from PIL import Image, ImageOps
from torchvision import transforms


IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class Letterbox:
    """Resize the longest side, then pad to a square without cropping."""

    def __init__(
        self,
        size: int,
        fill: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image.size}")

        scale = self.size / max(width, height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.Resampling.BILINEAR,
        )
        left = (self.size - resized.width) // 2
        top = (self.size - resized.height) // 2
        right = self.size - resized.width - left
        bottom = self.size - resized.height - top
        return ImageOps.expand(resized, (left, top, right, bottom), fill=self.fill)


def build_inference_transform(image_size: int):
    return transforms.Compose(
        [
            Letterbox(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

