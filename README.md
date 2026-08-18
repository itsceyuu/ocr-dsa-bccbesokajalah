# COMPFEST 18 DSA — Identity Document OCR baseline

KYC field extraction (`name`, `birth_date`, `address`) from identity
documents. The supplied export (`data/raw/`) has 732 images: 632 labeled in
`ground_truth.csv` (MyKad/MyKid cards, EU driving licences, passports, and
national ID specimens across dozens of countries/layouts) and 100 unlabeled.
Per the project rules, the **whole** dataset is in scope — no country/layout
subsetting.

## Constraints (from `data/raw/GUIDELINE FINAL PROJECT DSA COMPFEST 18.docx`)

- No external datasets, no manual image annotation (eval only).
- No LLM or VLM as an OCR pipeline component.
- Conventional OCR libraries are explicitly allowed: Tesseract, EasyOCR,
  PaddleOCR, and similar.
- Pretrained (zero-shot/inference-only), from-scratch training, or a mix —
  participant's choice.
- No AutoML / automated feature engineering platforms.

## Layout

```
src/ocr_baseline/
  data.py      # ground-truth loading (handles a CSV quoting quirk in the
               # export), identity-disjoint splitting, text normalization
  engines.py   # OCR engine wrappers: Tesseract, EasyOCR, PaddleOCR
  fields.py    # OCR text -> {name, birth_date, address}
  evaluate.py  # CER/WER/exact-match scoring, prediction harness
scripts/
  make_split.py    # deterministic identity-level train/val/test split
  run_baseline.py  # run one engine over a split and score it
```

## Run locally

```bash
uv pip install -e '.[ocr]'

python scripts/make_split.py   # writes data/splits/{train,val,test}.csv

PYTHONPATH=src python scripts/run_baseline.py \
  --engine paddleocr --split test \
  --split-dir data/splits --image-root data/raw/images \
  --output-dir reports
```

`--engine` is `tesseract`, `easyocr`, or `paddleocr`. Reports (with
per-field exact accuracy, CER, WER, and raw predictions) are written to
`reports/<engine>_<split>.json` — gitignored since predictions contain
identity-document PII.

## Current state

`fields.py` is a first, deliberately simple baseline: regex date extraction
(handles ISO, `DD.MM.YYYY`, and `DD MON YYYY` forms) plus line heuristics for
name/address. It is not tuned to any one document layout, since the dataset
has no single layout. Improving field selection — e.g. a trained line
classifier over OCR line features (position, keyword cues, digit density) —
is the next step, not a solved problem.

## Data download

```bash
uvx gdown --folder 'https://drive.google.com/drive/folders/1c843ExZujcHohdnndy_wfcT5hwA5iSCx' -O data/raw
```
