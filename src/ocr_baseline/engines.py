from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps


# All supplied images are 1040x780 MyKad/identity-card photographs. Relative
# crops keep the baseline usable if the images are resized without pretending
# to solve document detection or perspective correction yet.
CROP_BOXES = {
    "id_number": (0.055, 0.18, 0.68, 0.36),
    "name": (0.055, 0.54, 0.68, 0.68),
    "address": (0.055, 0.66, 0.72, 0.96),
}


def crop_field(image: Image.Image, field: str) -> Image.Image:
    left, top, right, bottom = CROP_BOXES[field]
    width, height = image.size
    return image.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))


def preprocess(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    return gray.resize((gray.width * 2, gray.height * 2))


@dataclass
class OCRResult:
    raw: dict[str, object]
    blocks: list["OCRBlock"] = field(default_factory=list)


@dataclass(frozen=True)
class OCRBlock:
    """One OCR detection in full-image coordinates."""

    text: str
    bbox: tuple[tuple[float, float], ...]
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "bbox": [list(point) for point in self.bbox],
            "confidence": self.confidence,
        }


class BaseEngine:
    name = "base"

    def predict(self, image: Image.Image) -> OCRResult:
        raise NotImplementedError


class TesseractEngine(BaseEngine):
    name = "tesseract"

    def __init__(self) -> None:
        import pytesseract

        self.pytesseract = pytesseract

    def _read(self, image: Image.Image, psm: int) -> str:
        return self.pytesseract.image_to_string(
            preprocess(image),
            config=f"--oem 3 --psm {psm}",
            lang="eng",
        ).strip()

    def predict(self, image: Image.Image) -> OCRResult:
        return OCRResult(
            raw={
                "full": self._read(image, 11),
                "id_number": self._read(crop_field(image, "id_number"), 7),
                "name": self._read(crop_field(image, "name"), 7),
                "address": self._read(crop_field(image, "address"), 6),
            }
        )


class EasyOCREngine(BaseEngine):
    name = "easyocr"

    def __init__(self, gpu: bool = True) -> None:
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def _read_native(self, image: Image.Image) -> list[list[object]]:
        import numpy as np

        detections = self.reader.readtext(np.asarray(image), detail=1, paragraph=False)
        return [
            [
                [[float(point[0]), float(point[1])] for point in item[0]],
                str(item[1]).strip(),
                float(item[2]),
            ]
            for item in detections
            if str(item[1]).strip()
        ]

    @staticmethod
    def _native_to_blocks(native: list[list[object]]) -> list[OCRBlock]:
        blocks = [
            OCRBlock(
                text=str(item[1]),
                bbox=tuple((float(point[0]), float(point[1])) for point in item[0]),
                confidence=float(item[2]),
            )
            for item in native
        ]
        return sorted(
            blocks,
            key=lambda block: (
                min(point[1] for point in block.bbox),
                min(point[0] for point in block.bbox),
            ),
        )

    @staticmethod
    def _native_to_text(native: list[list[object]]) -> str:
        blocks = EasyOCREngine._native_to_blocks(native)
        return "\n".join(block.text for block in blocks)

    def _read(self, image: Image.Image) -> tuple[list[list[object]], list[OCRBlock]]:
        native = self._read_native(image)
        return native, self._native_to_blocks(native)

    @staticmethod
    def _blocks_to_text(blocks: list[OCRBlock]) -> str:
        return "\n".join(block.text for block in blocks)

    def predict(self, image: Image.Image) -> OCRResult:
        full_native, full_blocks = self._read(image)
        id_native, id_blocks = self._read(crop_field(image, "id_number"))
        name_native, name_blocks = self._read(crop_field(image, "name"))
        address_native, address_blocks = self._read(crop_field(image, "address"))
        return OCRResult(
            raw={
                "full": self._blocks_to_text(full_blocks),
                "id_number": self._blocks_to_text(id_blocks),
                "name": self._blocks_to_text(name_blocks),
                "address": self._blocks_to_text(address_blocks),
                "blocks": [block.to_dict() for block in full_blocks],
                "native_easyocr": {
                    "full": full_native,
                    "id_number": id_native,
                    "name": name_native,
                    "address": address_native,
                },
            },
            blocks=full_blocks,
        )


class TrOCRSmallEngine(BaseEngine):
    name = "trocr-small-printed"

    def __init__(self, device: str = "auto", model_name: str = "microsoft/trocr-small-printed") -> None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.torch = torch

    def _read(self, image: Image.Image) -> str:
        pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        with self.torch.inference_mode():
            generated_ids = self.model.generate(pixel_values.to(self.device), max_new_tokens=64)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def predict(self, image: Image.Image) -> OCRResult:
        # TrOCR is a line recognizer. The crops are intentionally used as a
        # simple line/field approximation; no extra detector or external data.
        return OCRResult(
            raw={
                "full": self._read(image),
                "id_number": self._read(crop_field(image, "id_number")),
                "name": self._read(crop_field(image, "name")),
                "address": self._read(crop_field(image, "address")),
            }
        )


def build_engine(
    name: str,
    device: str = "auto",
    trocr_model: str = "microsoft/trocr-small-printed",
) -> BaseEngine:
    if name == "tesseract":
        return TesseractEngine()
    if name == "easyocr":
        import torch

        return EasyOCREngine(gpu=device == "cuda" or (device == "auto" and torch.cuda.is_available()))
    if name in {"trocr", "trocr-small-printed"}:
        return TrOCRSmallEngine(device=device, model_name=trocr_model)
    raise ValueError(f"Unknown OCR engine: {name}")
