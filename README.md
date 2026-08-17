# COMPFEST 18 DSA — Malaysian MyKad OCR baseline

This repository contains the first reproducible baseline for the DSA final
project, focused on the 112 labeled Malaysian MyKad/MyKid images in the
provided export. The task is to extract `name`, `birth_date`, and `address`.
The complete export has 632 labeled images plus 100 unlabeled images, but the
international specimen-card rows are intentionally out of scope for this
first experiment.

The Malaysian split is identity-level: images with the same `(name,
birth_date)` stay in one partition. It contains 90 train, 11 validation, and
11 test images across 109 identities.

## Run locally

```bash
python scripts/make_split.py --subset malaysia --output-dir data/splits/malaysia
PYTHONPATH=src python scripts/run_baseline.py \
  --engine easyocr --split test \
  --split-dir data/splits/malaysia --image-root data/raw/images \
  --output-dir reports/malaysia
```

The recommended Malaysian path uses EasyOCR boxes and confidence scores for
spatial field selection:

```bash
PYTHONPATH=src python scripts/run_baseline.py \
  --engine easyocr --field-parser malaysia-structured --split test \
  --split-dir data/splits/malaysia --image-root data/raw/images \
  --output-dir reports/malaysia
```

Optional OCR engines are declared under the `ocr` extra:

```bash
uv pip install -e '.[ocr]'
PYTHONPATH=src python scripts/run_baseline.py --engine easyocr --split test \
  --split-dir data/splits/malaysia --output-dir reports/malaysia
PYTHONPATH=src python scripts/run_baseline.py --engine trocr-small-printed --split test \
  --split-dir data/splits/malaysia --output-dir reports/malaysia
```

The baseline uses fixed relative regions for the printed ID number, name, and
address, then applies an OCR-tolerant Malaysian `YYMMDD-PB-####` parser. It
handles common glyph confusions only within a candidate ID and rejects invalid
calendar dates. Reports are written to `reports/malaysia/` with field-level
exact accuracy, regex extraction rates, normalized edit distance, and raw OCR
artifacts. They also include normalized character error rate (CER) and word
error rate (WER). See [`BASELINE.md`](BASELINE.md) for the algorithm,
structured pipeline, and results.

## Data download

The original Drive folder is the source of truth. A convenient download path
is:

```bash
uvx gdown --folder 'https://drive.google.com/drive/folders/1c843ExZujcHohdnndy_wfcT5hwA5iSCx' -O data/raw
```

No external training dataset is used.
