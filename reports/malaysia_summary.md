# Malaysian MyKad OCR Baseline

## Dataset split

The focused subset contains 112 annotated Malaysian MyKad/MyKid images and
109 unique identities. It was selected using the non-empty address annotation
in this export; the remaining labeled rows are international specimen-card
examples with blank address labels.

The identity-disjoint seed-42 split is:

| Split | Images | Identities |
|---|---:|---:|
| Train | 90 | 87 |
| Validation | 11 | 11 |
| Test | 11 | 11 |

## Models tested

- Tesseract
- EasyOCR
- TrOCR-small-printed

All are pretrained zero-shot baselines; none was fine-tuned on these cards.

EasyOCR is the selected engine for the next stage because it returns OCR text,
bounding boxes, and confidence scores. The Malaysian structured parser groups
those boxes into lines, detects the MyKad anchor, associates nearby name and
address candidates, validates them with Malaysian-specific cues, and falls
back to fixed crops when necessary.

## Regex algorithm

The custom parser targets Malaysian `YYMMDD-PB-####` numbers. It accepts OCR
separators and conservative glyph confusions (`O→0`, `I/L/T→1`, `S→5`, etc.)
only inside a candidate MyKad-shaped match. It validates the calendar date,
prefers complete 12-digit candidates, and supports a date-only fallback.

```python
candidate = parse_mykad(ocr_text)
if candidate:
    mykad_number = candidate.number       # 12 digits when fully recognized
    birth_date = candidate.birth_date     # normalized YYYY-MM-DD
```

The full implementation is in [`src/ocr_baseline/mykad.py`](../src/ocr_baseline/mykad.py).
Name extraction uses OCR lines after the detected MyKad number; address uses a
simple fixed crop `(0.055, 0.66, 0.72, 0.96)` and normalization.

## Generic engine baseline table

Measured with the original fixed-crop parser on the 11-image identity-disjoint
Malaysian test set:

| Model | Name exact | Birth-date exact | Valid MyKad/date regex | Full MyKad regex | Address exact |
|---|---:|---:|---:|---:|---:|
| Tesseract | 9.1% | 27.3% | 36.4% | 18.2% | 0.0% |
| EasyOCR | 9.1% | **63.6%** | **81.8%** | **72.7%** | 0.0% |
| TrOCR-small-printed | 0.0% | 9.1% | 9.1% | 9.1% | 0.0% |

Raw OCR output and predictions are retained locally as `*_test.json` artifacts;
the raw files are excluded from the public repository because they contain
personal identity-card data.

### CER and WER

These are normalized character and word error rates; lower is better.

| Model | Name CER | Name WER | Birth-date CER | Birth-date WER | Address CER | Address WER |
|---|---:|---:|---:|---:|---:|---:|
| Tesseract | 74.0% | 96.2% | 67.3% | 72.7% | 91.2% | 148.1% |
| EasyOCR | 84.9% | 85.6% | **22.7%** | **24.2%** | **62.4%** | **78.4%** |
| TrOCR-small-printed | 91.3% | 100.0% | 90.9% | 90.9% | 98.5% | 100.0% |

### EasyOCR parser comparison

| Parser | Name exact | Name CER | Name WER | Birth exact | Address exact | Address CER | Address WER |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-crop baseline | 9.1% | 85.2% | 85.6% | 63.6% | 0.0% | 62.4% | 78.4% |
| Malaysian structured | 9.1% | **52.8%** | **67.4%** | **63.6%** | **9.1%** | 68.0% | 81.4% |

The full structured raw report is retained locally as
`reports/malaysia/easyocr_malaysia_structured_test.json`; it preserves native
EasyOCR detections under `raw.native_easyocr`. It is not committed here because
the raw OCR includes personal identity-card data.
