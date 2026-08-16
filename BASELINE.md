# Malaysian MyKad OCR Baseline

## Dataset split

This first focused experiment uses only the 112 labeled Malaysian
MyKad/MyKid rows in the supplied export. In this export, those rows have a
non-empty address annotation; the later international specimen-card rows have
blank addresses. This is an annotation-level selector, not a visual country
classifier. The Malaysian subset contains 109 unique `(name, birth_date)`
identities.

The split is identity-disjoint, using seed 42. Therefore, repeated photographs
of one person cannot appear in different partitions.

| Split | Images | Identities |
|---|---:|---:|
| Train | 90 | 87 |
| Validation | 11 | 11 |
| Test | 11 | 11 |

The final table below is measured only on those 11 held-out Malaysian test
images. The test filenames are recorded in
`data/splits/malaysia/test.csv`.

## Models tested

- **Tesseract** — classical OCR baseline.
- **EasyOCR** — pretrained OCR detector and recognizer.
- **TrOCR-small-printed** — pretrained small printed-text OCR model.

No model was fine-tuned on this dataset. The models read the full card and
simple fixed field crops; the custom parser then processes the OCR transcript.

## Recommended Malaysian pipeline

EasyOCR is the selected engine because it provides text, bounding boxes, and
confidence scores in one pass. The `malaysia-structured` field parser then:

1. Groups EasyOCR boxes into reading-order lines.
2. Detects a MyKad/date anchor using the Malaysian parser.
3. Generates name and address candidates from nearby lines and spatial cues.
4. Scores candidates using OCR confidence, distance from the anchor, Malaysian
   name/address cues, and postcode presence.
5. Removes card metadata such as `WARGANEGARA`, `ISLAM`, and `LELAKI` from
   selected address lines.
6. Falls back to the fixed-crop parser if OCR boxes or anchors are missing.

Run it with:

```bash
PYTHONPATH=src python scripts/run_baseline.py \
  --engine easyocr \
  --field-parser malaysia-structured \
  --split test \
  --split-dir data/splits/malaysia \
  --image-root data/raw/images \
  --output-dir reports/malaysia
```

The structured output retains both the normalized OCR blocks and the native
EasyOCR detections in `raw.native_easyocr` (full card plus each crop), along
with selected candidates in the raw JSON artifact. This makes failure
analysis possible without rerunning OCR.

## Malaysian MyKad regex algorithm

Malaysian identity numbers use the twelve-digit `YYMMDD-PB-####` layout. The
parser below is deliberately OCR-tolerant, but only performs letter-to-digit
corrections inside a candidate that already has the MyKad shape:

```python
import re
from datetime import date

OCR_DIGIT_MAP = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "T": "1",
    "Z": "2", "S": "5", "G": "6", "B": "8",
}
OCR_DIGIT = r"[0-9OQDILTZSGB]"
SEP = r"[\s._:/\\|,–—-]*"


def digit_block(length: int) -> str:
    return OCR_DIGIT + "".join(SEP + OCR_DIGIT for _ in range(length - 1))


MYKAD_RE = re.compile(
    rf"(?<![A-Z0-9])"
    rf"(?P<birth>{digit_block(6)}){SEP}"
    rf"(?P<place>{digit_block(2)}){SEP}"
    rf"(?P<serial>{digit_block(4)})"
    rf"(?![A-Z0-9])",
    re.IGNORECASE,
)


def extract_mykad(ocr_text: str) -> tuple[str, str] | None:
    for match in MYKAD_RE.finditer(ocr_text.upper()):
        groups = match.groupdict()
        digits = {
            key: "".join(OCR_DIGIT_MAP.get(char, char)
                          for char in value if char.isdigit() or char in OCR_DIGIT_MAP)
            for key, value in groups.items()
        }
        yy, mm, dd = int(digits["birth"][:2]), int(digits["birth"][2:4]), int(digits["birth"][4:])
        year = 2000 + yy if yy <= date.today().year % 100 else 1900 + yy
        try:
            birth_date = date(year, mm, dd).isoformat()
        except ValueError:  # reject impossible dates such as 991332
            continue
        return digits["birth"] + digits["place"] + digits["serial"], birth_date
    return None
```

The repository implementation also accepts a date-only `YYMMDD` fallback when
the OCR crop omits the rest of the number, prefers a full 12-digit candidate,
and scores common Malaysian place codes `01`–`16` without rejecting extended
JPN codes. See [`src/ocr_baseline/mykad.py`](src/ocr_baseline/mykad.py).

For the final table, **birth-date exact accuracy** means the parsed ISO date
equals the annotation. **Valid MyKad regex rate** means the parser found a
valid date-bearing candidate. **Full MyKad regex rate** means it found the
complete 12-digit structure; it is an extraction-rate metric, not full-number
accuracy, because the supplied CSV does not contain the complete ID number.

Name and address are secondary baseline fields. Name is taken from the OCR
lines immediately after the detected ID number:

```python
def extract_name(ocr_text: str) -> str:
    lines = [normalize_text(line) for line in ocr_text.splitlines()]
    id_line = next(
        (i for i, line in enumerate(lines) if extract_birth_date(line)),
        None,
    )
    if id_line is None:
        return ""
    candidates = []
    for line in lines[id_line + 1:id_line + 5]:
        if line in STOP_NAME_LINES or not line:
            continue
        if any(char.isdigit() for char in line):
            break
        if len(line) >= 3:
            candidates.append(line)
    return " ".join(candidates)
```

The original baseline address path uses the relative crop
`(0.055, 0.66, 0.72, 0.96)`. The recommended structured path instead uses
OCR line geometry and postcode/address cues, with that crop retained as a
fallback.

## Generic engine baseline table

These results use the original fixed-crop field parser on the identity-disjoint
Malaysian test set (`n = 11`).

| Model | Name exact | Birth-date exact | Valid MyKad/date regex | Full MyKad regex | Address exact |
|---|---:|---:|---:|---:|---:|
| Tesseract | 9.1% | 27.3% | 36.4% | 18.2% | 0.0% |
| EasyOCR | 9.1% | 63.6% | 81.8% | 72.7% | 0.0% |
| TrOCR-small-printed | 0.0% | 9.1% | 9.1% | 9.1% | 0.0% |

The raw OCR transcripts and predictions are saved under
`reports/malaysia/`.

The raw JSON artifacts are intentionally kept local because they contain
identity-card names, addresses, and OCR text. This repository commit includes
the aggregate metrics and implementation, not those personal-data artifacts.

### CER and WER

CER and WER use the same uppercase, punctuation, and whitespace normalization
as the exact-match metrics. They are error rates, so lower is better; WER can
exceed 100% when a prediction contains many extra words.

| Model | Name CER | Name WER | Birth-date CER | Birth-date WER | Address CER | Address WER |
|---|---:|---:|---:|---:|---:|---:|
| Tesseract | 74.0% | 96.2% | 67.3% | 72.7% | 91.2% | 148.1% |
| EasyOCR | 84.9% | 85.6% | **22.7%** | **24.2%** | **62.4%** | **78.4%** |
| TrOCR-small-printed | 91.3% | 100.0% | 90.9% | 90.9% | 98.5% | 100.0% |

The metrics are also stored in each report under
`metrics.fields.<field>.character_error_rate` and
`metrics.fields.<field>.word_error_rate`.

The recommended EasyOCR structured result is reported separately below because
it uses OCR geometry and candidate selection rather than the generic fixed-crop
parser.

### EasyOCR parser comparison

Both rows below were run in the same local environment on the same 11-image
Malaysian test set. The structured parser improves name and exact address
selection while preserving birth-date performance.

| EasyOCR field parser | Name exact | Name CER | Name WER | Birth exact | Address exact | Address CER | Address WER |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-crop baseline | 9.1% | 85.2% | 85.6% | 63.6% | 0.0% | 62.4% | 78.4% |
| Malaysian structured | 9.1% | **52.8%** | **67.4%** | **63.6%** | **9.1%** | 68.0% | 81.4% |
