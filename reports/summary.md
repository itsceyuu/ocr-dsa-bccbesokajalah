# Baseline run summary

Evaluation uses the identity-disjoint test split (`n=63`, seed `42`). The
baseline is zero-shot OCR: no labeled training images were used to fit the
pretrained engines. The address column is non-empty for 35 test rows and blank
for 28 rows, so both aggregate and non-empty-only behavior matter.

| engine | name exact | birth date exact | birth-date regex non-empty | address exact | address exact on non-empty labels |
|---|---:|---:|---:|---:|---:|
| Tesseract | 7.9% | 17.5% | 17.5% | 3.2% | 0.0% |
| EasyOCR | **22.2%** | **47.6%** | **52.4%** | **33.3%** | 0.0% |
| TrOCR small printed | 1.6% | 7.9% | 17.5% | 17.5% | 0.0% |

The aggregate address score is mostly blank-address detection: EasyOCR is
exact on 75.0% of blank-address rows. All three engines fail to reproduce a
non-empty address exactly with the current fixed crop, making address
localization/preprocessing the clearest next improvement.

Detailed JSON predictions and raw OCR text are in the engine-specific files in
this directory. The TrOCR run used the local `microsoft/trocr-small-printed`
snapshot on Vast AI with a stable Transformers environment; the host’s default
development Transformers checkout could not load its SentencePiece tokenizer.

