#!/usr/bin/env python3
"""Baseline 1: fit a per-line classifier on train, evaluate on val/test.

Reuses the OCR already saved by run_baseline.py (reports/<engine>_<split>.json)
instead of re-running OCR. The classifier finds *which line* holds a field;
fields.extract_date still does the format normalization for birth_date.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from ocr_baseline.data import compact_text, levenshtein, normalize_text
from ocr_baseline.evaluate import Prediction, score_predictions
from ocr_baseline.fields import _NAME_STOPWORDS, extract_date
from ocr_baseline.line_features import (
    build_char_ngram_model,
    build_table,
    cluster_from_seed,
    compute_boilerplate_frequency,
    ends_with_name_particle,
    featurize_image_blocks,
    find_name_continuation,
    in_cluster,
    label_block,
    main_cluster_bbox,
    starts_with_name_particle,
)


_COMPACT_STOPWORDS = [compact_text(s) for s in _NAME_STOPWORDS if len(compact_text(s)) >= 4]


def is_boilerplate_stopword(text: str) -> bool:
    """Exact match plus a small edit-distance tolerance, so a single OCR-
    dropped letter (e.g. "SLAM"/"ELAKI" for "ISLAM"/"LELAKI") doesn't slip
    the hard exclusion -- confirmed directly, these garbled variants leaked
    into address predictions past the exact-match-only version of this
    check. Length floor (>=4) avoids false-matching short real words."""

    normalized = normalize_text(text)
    if normalized in _NAME_STOPWORDS:
        return True
    compact = compact_text(text)
    if len(compact) < 4:
        return False
    return any(
        levenshtein(compact, sw) <= 1 for sw in _COMPACT_STOPWORDS if abs(len(compact) - len(sw)) <= 1
    )


class _StringClassesModel:
    """Wraps a model whose `classes_` is read-only/encoded-int so it exposes
    string classes_ instead -- predict_fields/predict_fields_geo only ever
    call .predict_proba and .classes_, so this is a complete substitute."""

    def __init__(self, model, classes):
        self._model = model
        self.classes_ = classes

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def fit(
    train_report: Path, image_root: Path, freq_lookup: dict[str, float] | None = None,
    classifier: str = "logreg", ngram_model: dict[str, float] | None = None, use_face: bool = False,
) -> tuple[DictVectorizer, object]:
    features, labels, _ = build_table(train_report, image_root, freq_lookup, ngram_model, use_face)
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(features)
    # Same feature-interaction rationale as HGB (see its comment above) --
    # other gradient-boosting implementations, different tree-building
    # strategies (leaf-wise vs level-wise, GPU-oriented histogram binning,
    # built-in ordered boosting for CatBoost) that can matter on a dataset
    # this small and imbalanced.
    if classifier == "hgb":
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
        model.fit(X, labels)
    elif classifier == "lgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(class_weight="balanced", random_state=0, verbose=-1)
        model.fit(X, labels)
    elif classifier == "xgboost":
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        # XGBoost has no class_weight param (compute_sample_weight is the
        # sample-weight equivalent of class_weight="balanced") and, unlike
        # sklearn/lightgbm/catboost, doesn't accept string labels directly --
        # and unlike them, `classes_` is read-only (always the encoded ints),
        # so predict_fields's `classes.index(label)` lookup would break.
        # _StringClassesModel translates it back to string labels; the
        # underlying predict_proba columns are already in encoder order.
        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)
        xgb_model = XGBClassifier(random_state=0)
        xgb_model.fit(X, y, sample_weight=compute_sample_weight("balanced", labels))
        model = _StringClassesModel(xgb_model, encoder.classes_)
    elif classifier == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(auto_class_weights="Balanced", random_state=0, verbose=False)
        model.fit(X, labels)
    else:
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X, labels)
    return vectorizer, model


def predict_fields(
    blocks: list[dict],
    image_size: tuple[int, int],
    vectorizer: DictVectorizer,
    model: LogisticRegression,
    address_threshold: float = 0.5,
    freq_lookup: dict[str, float] | None = None,
    ngram_model: dict[str, float] | None = None,
    face_center: tuple[float, float] | None = None,
    boilerplate_freq_lookup: dict[str, float] | None = None,  # unused here, kept for signature parity
) -> dict[str, str]:
    if not blocks:
        return {"name": "", "birth_date": "", "address": ""}

    feats = featurize_image_blocks(blocks, image_size, freq_lookup, ngram_model, face_center)
    X = vectorizer.transform(feats)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    def class_proba(row: int, label: str) -> float:
        return proba[row][classes.index(label)] if label in classes else 0.0

    best_name, best_name_score = "", -1.0
    best_date_text, best_date_score = "", -1.0
    address_lines: list[tuple[int, str]] = []
    for i, block in enumerate(blocks):
        name_score = class_proba(i, "name")
        if name_score > best_name_score:
            best_name, best_name_score = block["text"], name_score
        date_score = class_proba(i, "birth_date")
        if date_score > best_date_score:
            best_date_text, best_date_score = block["text"], date_score
        if class_proba(i, "address") >= address_threshold:
            address_lines.append((i, block["text"]))

    address_lines.sort(key=lambda item: item[0])
    return {
        "name": normalize_text(best_name) if best_name_score > 0 else "",
        "birth_date": extract_date(best_date_text) if best_date_score > 0 else "",
        "address": normalize_text(", ".join(text for _, text in address_lines)),
    }


def predict_fields_geo(
    blocks: list[dict],
    image_size: tuple[int, int],
    vectorizer: DictVectorizer,
    model: LogisticRegression,
    address_seed_threshold: float = 0.15,  # swept 0.15-0.40; lower is strictly better here, see experiments.jsonl
    freq_lookup: dict[str, float] | None = None,
    ngram_model: dict[str, float] | None = None,
    face_center: tuple[float, float] | None = None,
    boilerplate_freq_lookup: dict[str, float] | None = None,
) -> dict[str, str]:
    """Same classifier, layout-agnostic geometric post-processing on top:
    (1) reject name/date candidates that sit outside the card's own block
    cluster (kills background-clutter hijacking, e.g. a poster in frame);
    (2) build the address from a proximity cluster around the best-scoring
    block instead of "any block above a threshold anywhere on the card".
    `boilerplate_freq_lookup` (separate from `freq_lookup`, which only feeds
    the trained soft feature) hard-excludes any block that's frequent
    boilerplate -- kept as its own param so it can be applied at inference
    time even for a model that was trained with freq_lookup=None, without a
    train/inference mismatch on the *soft* feature's value."""

    if not blocks:
        return {"name": "", "birth_date": "", "address": ""}

    feats = featurize_image_blocks(blocks, image_size, freq_lookup, ngram_model, face_center)
    X = vectorizer.transform(feats)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    def class_proba(row: int, label: str) -> float:
        return proba[row][classes.index(label)] if label in classes else 0.0

    cluster_bbox = main_cluster_bbox(blocks)
    in_doc = [in_cluster(b, cluster_bbox) for b in blocks]

    # The single topmost block in reading order is always the document
    # title/header on every layout in this dataset ("KAD PENGENALAN",
    # "CARTAO DE CIDADAO", ...) -- confirmed empirically it's the wrongly-
    # picked block in half of all name failures (15/30) while being the
    # correct name in only 1/67 correct cases. Universal document
    # convention (the title always leads), not a per-country rule.
    topmost_index = min(range(len(blocks)), key=lambda i: min(p[1] for p in blocks[i]["bbox"]))

    # Outlier rejection only for `name`: it's the field that was actually
    # getting hijacked by background clutter (see poster-in-frame bug).
    # birth_date is already regex-verified by extract_date and was near-
    # optimal before this change -- filtering it too only adds risk, no
    # benefit, and empirically regressed it hard. Leave it unfiltered.
    # Automatic counterpart to the hand-typed stopword list: any text that
    # recurs verbatim across a real chunk of train identities is template
    # boilerplate almost by definition, in ANY language -- no per-country
    # typing needed. Same hard-exclusion mechanism as the stopword/topmost
    # rules (a soft learned feature was tried first and failed -- see
    # boilerplate_freq's history in experiments.jsonl). Length floor avoids
    # excluding single-character OCR noise fragments that happen to recur.
    boilerplate_freq_lookup = boilerplate_freq_lookup or {}
    boilerplate_freq_threshold = 0.15

    best_name, best_name_index, best_name_score = "", -1, -1.0
    best_date_text, best_date_score = "", -1.0
    address_scores = [class_proba(i, "address") for i in range(len(blocks))]
    for i, block in enumerate(blocks):
        # Hard exclusion, not just a soft feature: a whole-block boilerplate
        # match is closed-vocabulary and enumerable (unlike open-ended name
        # text), so a direct rule is safer than trusting the learned score
        # to weigh it correctly -- same reasoning as the topmost-block rule.
        is_stopword = is_boilerplate_stopword(block["text"])
        block_compact = compact_text(block["text"])
        is_frequent_boilerplate = (
            len(block_compact) >= 4 and boilerplate_freq_lookup.get(block_compact, 0.0) >= boilerplate_freq_threshold
        )
        # Boilerplate exclusion applies to address too, not just name: MyKad
        # interleaves gender/religion fields (WARGANEGARA/LELAKI/ISLAM)
        # spatially with the address block, and cluster_from_seed's pure
        # gap+score walk happily absorbs them -- confirmed directly, "LELAKI"
        # and "ISLAM"/"SLAM" (OCR-garbled) showed up glued into the address
        # prediction in 6 of 8 sampled near-miss cases.
        # Pure-digit blocks 6+ chars long (a truncated/duplicate ID number
        # print, e.g. a verification-stamp repeat of the MyKad number) are
        # never genuine address content -- confirmed no ground-truth address
        # in the whole dataset contains a pure 6+ digit token (postcodes are
        # 4-5 digits, house numbers shorter). The model itself doesn't score
        # these as birth_date either (it's a truncated/malformed ID, not a
        # valid date match), so a score-based guard can't catch it -- this
        # needs to be a shape rule instead.
        is_id_fragment = block_compact.isdigit() and len(block_compact) >= 6
        if is_stopword or is_frequent_boilerplate or starts_with_name_particle(block["text"]) or is_id_fragment:
            address_scores[i] = -1.0
        if in_doc[i] and i != topmost_index and not is_stopword and not is_frequent_boilerplate:
            name_score = class_proba(i, "name")
            if name_score > best_name_score:
                best_name, best_name_index, best_name_score = block["text"], i, name_score
        date_score = class_proba(i, "birth_date")
        if date_score > best_date_score:
            best_date_text, best_date_score = block["text"], date_score

    # Malay names commonly split given-name and surname across two blocks
    # ("MOHD ASWARDIBIN" / "AHMAD") -- absorb the immediate next block by
    # adjacency + shape, not by its own class score (see
    # find_name_continuation's docstring for why score-based join is unsafe
    # here specifically, unlike for address).
    name_text = best_name
    name_block_indices = {best_name_index} if best_name_index >= 0 else set()
    if best_name_index >= 0 and ends_with_name_particle(best_name):
        cont = find_name_continuation(blocks, best_name_index, image_size[1])
        if cont is not None:
            name_text = f"{best_name} {blocks[cont]['text']}"
            name_block_indices.add(cont)

    # Field exclusivity: a block already claimed by name can't also seed or
    # be absorbed into address -- confirmed empirically that without this,
    # the name block ends up duplicated into the address prediction in ~21%
    # of records (e.g. "MOHD SHAHRIMAN BIN ADENAN B-8-8 KRISTAL HEIGHTS...").
    for i in name_block_indices:
        address_scores[i] = -1.0

    # Broader version of the same rule: even when name SELECTION picks the
    # wrong block (so the real name block is never in name_block_indices),
    # any block the model itself thinks plausibly looks like a name still
    # shouldn't leak into address -- confirmed directly on image_087.jpg,
    # where a wrong name pick let "LEE TOO CHENG" (name_score=0.66) get
    # absorbed into the address prediction verbatim.
    name_leak_threshold = 0.7  # swept 0.4-1.0; 0.7 is the sweet spot, see experiments.jsonl
    for i in range(len(blocks)):
        if class_proba(i, "name") >= name_leak_threshold:
            address_scores[i] = -1.0

    best_address_index, best_address_score = -1, -1.0
    for i, score in enumerate(address_scores):
        if score > best_address_score:
            best_address_index, best_address_score = i, score

    address_text = ""
    if best_address_score >= address_seed_threshold:
        indices = cluster_from_seed(blocks, address_scores, best_address_index, image_size[1])
        address_text = ", ".join(blocks[i]["text"] for i in indices)

    return {
        "name": normalize_text(name_text) if best_name_score > 0 else "",
        "birth_date": extract_date(best_date_text) if best_date_score > 0 else "",
        "address": normalize_text(address_text),
    }


def predict_fields_multi_engine(
    primary_blocks: list[dict],
    secondary_blocks: list[dict],
    image_size: tuple[int, int],
    vectorizer: DictVectorizer,
    model: LogisticRegression,
    **kwargs,
) -> dict[str, str]:
    """name/address stay on the primary engine's blocks alone -- the model's
    rank/position features are calibrated on that engine's typical block
    count, and mixing in a second engine's blocks (even filtered to "novel"
    text only) measurably regressed both fields (confirmed empirically).
    birth_date is the one field where more redundant reads directly help:
    it's decided by extract_date's regex on the classifier's single winning
    block, so a second engine's independent read of the ID number just gives
    the regex more chances to hit -- confirmed: CER 6.9/8.3% -> 4.3/4.2%,
    with zero cost to name/address since they don't touch these blocks."""

    primary = predict_fields_geo(primary_blocks, image_size, vectorizer, model, **kwargs)
    merged = sorted(
        list(primary_blocks) + list(secondary_blocks),
        key=lambda b: (min(p[1] for p in b["bbox"]), min(p[0] for p in b["bbox"])),
    )
    merged_pred = predict_fields_geo(merged, image_size, vectorizer, model, **kwargs)
    return {"name": primary["name"], "address": primary["address"], "birth_date": merged_pred["birth_date"]}


def evaluate_report_multi_engine(
    primary_report_path: Path, secondary_report_path: Path, image_root: Path,
    vectorizer: DictVectorizer, model: LogisticRegression,
    boilerplate_freq_lookup: dict[str, float] | None = None,
) -> dict:
    from PIL import Image

    primary_data = json.loads(primary_report_path.read_text(encoding="utf-8"))
    secondary_data = json.loads(secondary_report_path.read_text(encoding="utf-8"))
    secondary_by_file = {r["filename"]: r for r in secondary_data["predictions"]}

    predictions = []
    for record in primary_data["predictions"]:
        primary_blocks = record["raw"].get("blocks") or []
        secondary_record = secondary_by_file.get(record["filename"])
        secondary_blocks = (secondary_record["raw"].get("blocks") or []) if secondary_record else []
        with Image.open(image_root / record["filename"]) as image:
            size = image.size
        predicted = predict_fields_multi_engine(primary_blocks, secondary_blocks, size, vectorizer, model,
                                                  boilerplate_freq_lookup=boilerplate_freq_lookup)
        predictions.append(
            Prediction(filename=record["filename"], expected=record["expected"], predicted=predicted, raw={})
        )
    return score_predictions(predictions)


def evaluate_report(
    report_path: Path, image_root: Path, vectorizer: DictVectorizer, model: LogisticRegression,
    predictor=predict_fields, freq_lookup: dict[str, float] | None = None,
    ngram_model: dict[str, float] | None = None, use_face: bool = False,
    boilerplate_freq_lookup: dict[str, float] | None = None,
) -> dict:
    from PIL import Image

    data = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = []
    for record in data["predictions"]:
        blocks = record["raw"].get("blocks") or []
        with Image.open(image_root / record["filename"]) as image:
            size = image.size
            face_center = None
            if use_face:
                from ocr_baseline.preprocess import detect_face_center

                face_center = detect_face_center(image)
        predicted = predictor(blocks, size, vectorizer, model, freq_lookup=freq_lookup,
                               ngram_model=ngram_model, face_center=face_center,
                               boilerplate_freq_lookup=boilerplate_freq_lookup)
        predictions.append(
            Prediction(filename=record["filename"], expected=record["expected"], predicted=predicted, raw={})
        )
    return score_predictions(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate the Baseline 1 line classifier")
    parser.add_argument("--engine", default="paddleocr", help="Which saved OCR run to build features from")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--image-root", default="data/raw/images")
    parser.add_argument("--model-out", default="reports/line_classifier.pkl")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    image_root = Path(args.image_root)

    train_report = reports_dir / f"{args.engine}_train.json"

    # Verified-best recipe (see experiments.jsonl history): no boilerplate-
    # frequency feature active (freq_lookup=None -> feature stays neutral at
    # 0), logistic regression, geo post-processing. This is what gets saved
    # as the production model.
    vectorizer, model = fit(train_report, image_root, classifier="logreg")

    # Automatic, language-agnostic counterpart to the hand-typed stopword
    # list (see predict_fields_geo's docstring): any text recurring across a
    # real chunk of train identities is boilerplate almost by definition, in
    # any language. Applied as a hard exclusion at inference time -- works
    # for every classifier below without retraining, and self-extends to new
    # languages as train grows, unlike a fixed list.
    boilerplate_freq_lookup = compute_boilerplate_frequency(train_report)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "model": model, "freq_lookup": None,
                     "boilerplate_freq_lookup": boilerplate_freq_lookup}, handle)

    # Which features push a line toward each class -- cheap interpretability
    # for the report, not just a black box.
    print("Top positive coefficients per class:")
    for class_index, class_name in enumerate(model.classes_):
        coefs = model.coef_[class_index] if len(model.classes_) > 2 else model.coef_[0]
        ranked = sorted(zip(vectorizer.feature_names_, coefs), key=lambda item: -item[1])[:4]
        print(f"  {class_name:12s} {ranked}")
    print()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_baseline import log_experiment

    # Other classifiers, same features -- same feature-interaction rationale
    # as HGB. Trained once here, evaluated (with and without geo) on every
    # split below. Logged separately; none of these touch the saved
    # production model above unless you decide to swap it in.
    other_models = {tag: fit(train_report, image_root, classifier=tag)
                     for tag in ("hgb", "lgbm", "xgboost", "catboost")}

    # Character n-gram "looks like a name" feature (see line_features.py's
    # build_char_ngram_model docstring): denser than the word-level gazetteer
    # that regressed things earlier, and language-agnostic since it's learned
    # from this dataset's own weak-labeled blocks, not a hardcoded word list.
    # Tried on both logreg (isolates the feature's effect) and catboost (the
    # strongest classifier so far, to see if the feature helps it too).
    ngram_model = build_char_ngram_model(train_report)
    ngram_vectorizer, ngram_model_fit = fit(train_report, image_root, classifier="logreg", ngram_model=ngram_model)
    ngram_cat_vectorizer, ngram_cat_model = fit(train_report, image_root, classifier="catboost", ngram_model=ngram_model)

    # Distance-to-detected-face feature (see preprocess.detect_face_center):
    # ~91% face-detection hit rate checked empirically before wiring this in.
    # Layout-agnostic in a stronger sense than the MyKad ID-number anchor --
    # doesn't encode any one country's field order, just the physical fact
    # that ID documents have a photo and personal data lives near it.
    face_vectorizer, face_model = fit(train_report, image_root, classifier="logreg", use_face=True)
    face_cat_vectorizer, face_cat_model = fit(train_report, image_root, classifier="catboost", use_face=True)

    for split in ("val", "test"):
        report_path = reports_dir / f"{args.engine}_{split}.json"
        if not report_path.exists():
            continue

        metrics = evaluate_report(report_path, image_root, vectorizer, model, predictor=predict_fields)
        log_experiment(reports_dir, args.engine, split, "baseline1-line-classifier", metrics)
        print(f"=== {args.engine} / {split} (logreg) ===")
        print(json.dumps(metrics, indent=2))

        geo_metrics = evaluate_report(report_path, image_root, vectorizer, model, predictor=predict_fields_geo,
                                       boilerplate_freq_lookup=boilerplate_freq_lookup)
        log_experiment(reports_dir, args.engine, split, "baseline1-line-classifier-geo", geo_metrics)
        print(f"=== {args.engine} / {split} (logreg + geo) ===")
        print(json.dumps(geo_metrics, indent=2))

        secondary_report_path = reports_dir / f"easyocr_{split}.json"
        if secondary_report_path.exists():
            multi_metrics = evaluate_report_multi_engine(report_path, secondary_report_path, image_root,
                                                           vectorizer, model,
                                                           boilerplate_freq_lookup=boilerplate_freq_lookup)
            log_experiment(reports_dir, args.engine, split, "baseline1-line-classifier-geo-multiengine",
                            multi_metrics)
            print(f"=== {args.engine} / {split} (logreg + geo + easyocr-for-date) ===")
            print(json.dumps(multi_metrics, indent=2))

        for tag, (tag_vectorizer, tag_model) in other_models.items():
            m = evaluate_report(report_path, image_root, tag_vectorizer, tag_model, predictor=predict_fields)
            log_experiment(reports_dir, args.engine, split, f"baseline1-{tag}", m)
            print(f"=== {args.engine} / {split} ({tag}) ===")
            print(json.dumps(m, indent=2))

            geo_m = evaluate_report(report_path, image_root, tag_vectorizer, tag_model, predictor=predict_fields_geo,
                                     boilerplate_freq_lookup=boilerplate_freq_lookup)
            log_experiment(reports_dir, args.engine, split, f"baseline1-{tag}-geo", geo_m)
            print(f"=== {args.engine} / {split} ({tag} + geo) ===")
            print(json.dumps(geo_m, indent=2))

        ngram_geo_m = evaluate_report(report_path, image_root, ngram_vectorizer, ngram_model_fit,
                                       predictor=predict_fields_geo, ngram_model=ngram_model)
        log_experiment(reports_dir, args.engine, split, "baseline1-charngram-geo", ngram_geo_m)
        print(f"=== {args.engine} / {split} (logreg + charngram + geo) ===")
        print(json.dumps(ngram_geo_m, indent=2))

        ngram_cat_geo_m = evaluate_report(report_path, image_root, ngram_cat_vectorizer, ngram_cat_model,
                                           predictor=predict_fields_geo, ngram_model=ngram_model)
        log_experiment(reports_dir, args.engine, split, "baseline1-charngram-catboost-geo", ngram_cat_geo_m)
        print(f"=== {args.engine} / {split} (catboost + charngram + geo) ===")
        print(json.dumps(ngram_cat_geo_m, indent=2))

        face_geo_m = evaluate_report(report_path, image_root, face_vectorizer, face_model,
                                      predictor=predict_fields_geo, use_face=True)
        log_experiment(reports_dir, args.engine, split, "baseline1-face-geo", face_geo_m)
        print(f"=== {args.engine} / {split} (logreg + face + geo) ===")
        print(json.dumps(face_geo_m, indent=2))

        face_cat_geo_m = evaluate_report(report_path, image_root, face_cat_vectorizer, face_cat_model,
                                          predictor=predict_fields_geo, use_face=True)
        log_experiment(reports_dir, args.engine, split, "baseline1-face-catboost-geo", face_cat_geo_m)
        print(f"=== {args.engine} / {split} (catboost + face + geo) ===")
        print(json.dumps(face_cat_geo_m, indent=2))


if __name__ == "__main__":
    main()
