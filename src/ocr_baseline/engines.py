from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

# No LLM/VLM engines here by project rule -- conventional OCR only
# (Tesseract, EasyOCR, PaddleOCR are all explicitly allowed).


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


def _sort_reading_order(blocks: list[OCRBlock]) -> list[OCRBlock]:
    return sorted(
        blocks,
        key=lambda block: (
            min(point[1] for point in block.bbox),
            min(point[0] for point in block.bbox),
        ),
    )


class BaseEngine:
    name = "base"

    def predict(self, image: Image.Image) -> OCRResult:
        """Run full-image OCR. No fixed crops: card layout varies by document
        type/country, so line detection + downstream field parsing (see
        fields.py) has to do the localization work, not a hardcoded region.
        """

        raise NotImplementedError


class TesseractEngine(BaseEngine):
    name = "tesseract"

    def __init__(self) -> None:
        import pytesseract

        self.pytesseract = pytesseract

    def predict(self, image: Image.Image) -> OCRResult:
        text = self.pytesseract.image_to_string(image, config="--oem 3 --psm 11", lang="eng").strip()
        return OCRResult(raw={"full": text})


class EasyOCREngine(BaseEngine):
    name = "easyocr"

    def __init__(self, gpu: bool = True) -> None:
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def predict(self, image: Image.Image) -> OCRResult:
        import numpy as np

        detections = self.reader.readtext(np.asarray(image), detail=1, paragraph=False)
        blocks = _sort_reading_order(
            [
                OCRBlock(
                    text=str(text).strip(),
                    bbox=tuple((float(point[0]), float(point[1])) for point in box),
                    confidence=float(confidence),
                )
                for box, text, confidence in detections
                if str(text).strip()
            ]
        )
        full_text = "\n".join(block.text for block in blocks)
        return OCRResult(
            raw={"full": full_text, "blocks": [block.to_dict() for block in blocks]},
            blocks=blocks,
        )


class PaddleOCREngine(BaseEngine):
    """PaddleOCR PP-OCR detector + recognizer (+ angle classifier).

    Stronger than EasyOCR/Tesseract on rotated/skewed phone photos, which is
    most of what this dataset's card photography looks like.
    """

    name = "paddleocr"

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        from paddleocr import PaddleOCR

        self.reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    def predict(self, image: Image.Image) -> OCRResult:
        import numpy as np

        result = self.reader.ocr(np.asarray(image.convert("RGB")), cls=True)
        # PaddleOCR returns [None] (not []) for a page with zero detections.
        detections = (result[0] if result else None) or []
        blocks = _sort_reading_order(
            [
                OCRBlock(
                    text=str(text).strip(),
                    bbox=tuple((float(point[0]), float(point[1])) for point in box),
                    confidence=float(confidence),
                )
                for box, (text, confidence) in detections
                if str(text).strip()
            ]
        )
        full_text = "\n".join(block.text for block in blocks)
        return OCRResult(
            raw={"full": full_text, "blocks": [block.to_dict() for block in blocks]},
            blocks=blocks,
        )


def build_engine(name: str, device: str = "auto") -> BaseEngine:
    if name == "tesseract":
        return TesseractEngine()
    if name == "easyocr":
        import torch

        return EasyOCREngine(gpu=device == "cuda" or (device == "auto" and torch.cuda.is_available()))
    if name == "paddleocr":
        return PaddleOCREngine(use_gpu=device == "cuda")
    raise ValueError(f"Unknown OCR engine: {name}")
