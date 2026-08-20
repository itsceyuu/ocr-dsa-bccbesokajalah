"""Image preprocessing run before OCR: grayscale, clutter crop + perspective
correction (one step: warp to the detected document quad), and low-light-aware
CLAHE. cv2 does the contour/warp/CLAHE work -- no custom edge-detection code
to maintain.
"""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np
from PIL import Image


# ID cards/passports/licences all land roughly in this aspect-ratio band
# (MyKad is 1.586:1); accepting only this range rejects near-square or
# far-off-ratio false positives (stamps, forms, desk edges) that would
# otherwise be picked up as "the document" by area alone.
_CARD_ASPECT_RANGE = (1.2, 1.9)


def _find_document_quad(gray: np.ndarray) -> np.ndarray | None:
    """Largest plausible 4-corner, card-shaped contour, else None."""

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]
    # Lower area floor than before (was 0.2): this dataset also has cards
    # held at arm's length, small in frame against a plain background.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(contour) < 0.05 * image_area:
            break  # remaining contours are only smaller -- not the document
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        corners = _order_corners(quad)
        tl, tr, br, bl = corners
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        if height < 1:
            continue
        ratio = max(width, height) / min(width, height)
        if _CARD_ASPECT_RANGE[0] <= ratio <= _CARD_ASPECT_RANGE[1]:
            return quad
    return None


def _line_intersection(line_a: tuple[float, float], line_b: tuple[float, float]) -> np.ndarray | None:
    """Intersection of two Hough lines given as (rho, theta), or None if
    near-parallel (Dropbox's "reject acute-angle intersections" filter)."""

    rho_a, theta_a = line_a
    rho_b, theta_b = line_b
    a = np.array([[np.cos(theta_a), np.sin(theta_a)], [np.cos(theta_b), np.sin(theta_b)]])
    if abs(np.sin(theta_a - theta_b)) < 0.17:  # < ~10 degrees apart
        return None
    try:
        x, y = np.linalg.solve(a, np.array([rho_a, rho_b]))
    except np.linalg.LinAlgError:
        return None
    return np.array([x, y], dtype=np.float32)


def _find_document_quad_hough(gray: np.ndarray) -> np.ndarray | None:
    """Dropbox's document-scanner approach: Canny -> Hough lines -> intersect
    horizontal-ish with vertical-ish lines into corner candidates -> score
    each quadrilateral by how much of its perimeter actually sits on an
    edge. Survives a fragmented/textured boundary (e.g. wood-grain desk)
    that a single continuous contour (_find_document_quad) can't."""

    height, width = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=int(0.15 * min(height, width)))
    if lines is None or len(lines) < 4:
        return None
    lines = [tuple(line[0]) for line in lines[:20]]  # strongest first, cap combinations

    horizontal = [line for line in lines if abs(np.sin(line[1])) > 0.5]  # theta near 90 deg
    vertical = [line for line in lines if abs(np.sin(line[1])) <= 0.5]  # theta near 0 deg
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    margin = 0.1
    best_quad, best_score = None, 0.0
    for h1, h2 in combinations(horizontal[:8], 2):
        for v1, v2 in combinations(vertical[:8], 2):
            corners = [_line_intersection(h, v) for h in (h1, h2) for v in (v1, v2)]
            if any(c is None for c in corners):
                continue
            pts = np.array(corners, dtype=np.float32)
            if (pts[:, 0].min() < -margin * width or pts[:, 0].max() > width * (1 + margin)
                    or pts[:, 1].min() < -margin * height or pts[:, 1].max() > height * (1 + margin)):
                continue
            quad = cv2.convexHull(pts).reshape(-1, 2)
            if len(quad) != 4:
                continue
            ordered = _order_corners(quad)
            tl, tr, br, bl = ordered
            side_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
            side_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            if side_h < 1 or side_w < 1:
                continue
            ratio = max(side_w, side_h) / min(side_w, side_h)
            if not (_CARD_ASPECT_RANGE[0] <= ratio <= _CARD_ASPECT_RANGE[1]):
                continue
            mask = np.zeros_like(edges)
            cv2.polylines(mask, [np.clip(ordered, 0, [width - 1, height - 1]).astype(np.int32)], True, 255, 3)
            score = cv2.countNonZero(cv2.bitwise_and(mask, edges))
            if score > best_score:
                best_quad, best_score = ordered, score
    return best_quad


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """top-left, top-right, bottom-right, bottom-left."""

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def _warp(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = _order_corners(quad)
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 20 or height < 20:
        return bgr  # degenerate quad, keep the original image
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(bgr, matrix, (width, height))


def grayscale_image(image: Image.Image) -> Image.Image:
    return image.convert("L")


def perspective_correct_image(image: Image.Image) -> Image.Image:
    """Clutter crop + perspective correction only, no grayscale/CLAHE. Keeps
    color output so it's isolated from the other preprocessing steps."""

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    quad = _find_document_quad(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if quad is not None:
        bgr = _warp(bgr, quad)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def perspective_correct_image_hough(image: Image.Image) -> Image.Image:
    """Same as perspective_correct_image but using the Hough-line quad finder
    instead of the single-contour one -- see _find_document_quad_hough."""

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    quad = _find_document_quad_hough(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if quad is not None:
        bgr = _warp(bgr, quad)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


_LDRNET_PATH = "data/models/ldrnet"
_ldrnet_layer = None  # lazy-loaded singleton -- tensorflow is a heavy import


def _find_document_quad_ldrnet(bgr: np.ndarray) -> np.ndarray | None:
    """Pretrained LDRNet (github.com/niuwagege/LDRNet, MIT): a CNN trained
    specifically to regress a document's 4 corners, unlike the two Canny/
    Hough heuristics above. Ships its own pretrained weights -- no training
    or manual annotation on our side, same "pretrained model" category as
    the OCR engines themselves."""

    global _ldrnet_layer
    if _ldrnet_layer is None:
        import keras

        _ldrnet_layer = keras.layers.TFSMLayer(_LDRNET_PATH, call_endpoint="serving_default")

    height, width = bgr.shape[:2]
    resized = cv2.resize(bgr, (224, 224)).astype(np.float32) / 255.0
    out = _ldrnet_layer(np.expand_dims(resized, 0))["output_1"].numpy()[0]
    coords = out[:8]
    quad = np.array(
        [[coords[i] * width, coords[i + 1] * height] for i in range(0, 8, 2)], dtype=np.float32
    )
    return quad


def perspective_correct_image_ldrnet(image: Image.Image) -> Image.Image:
    """Same as perspective_correct_image but using pretrained LDRNet for
    corner detection instead of a classical heuristic."""

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    quad = _find_document_quad_ldrnet(bgr)
    if quad is not None:
        bgr = _warp(bgr, quad)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def preprocess_image(image: Image.Image) -> Image.Image:
    """Grayscale + clutter crop/perspective correction + low-light-aware CLAHE.

    Returns a grayscale PIL image; every OCR engine here accepts grayscale.
    """

    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    quad = _find_document_quad(gray)
    if quad is not None:
        gray = cv2.cvtColor(_warp(bgr, quad), cv2.COLOR_BGR2GRAY)

    # Low-light-aware: darker images get a stronger clip limit, well-lit ones
    # a gentle one so we don't blow out contrast that's already fine.
    clip_limit = 4.0 if gray.mean() < 90 else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(gray))


_FACE_CASCADE_PATH = "data/models/haarcascade_frontalface_default.xml"
_face_cascade = None  # lazy-loaded singleton, same pattern as LDRNet above


def detect_face_center(image: Image.Image) -> tuple[float, float] | None:
    """Center of the largest detected face, in original image pixel
    coordinates, or None if no face found (~91% hit rate on this dataset,
    checked empirically before wiring this in). Nearly every ID document
    worldwide places the name near the portrait photo -- unlike the MyKad-
    specific ID-number anchor, this doesn't encode any one country's layout,
    just the physical fact that ID cards have a face on them."""

    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)

    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest by area
    return (x + w / 2, y + h / 2)


if __name__ == "__main__":
    # ponytail self-check: runs on a synthetic bright + a synthetic dark
    # image, asserts output stays a valid same-mode-ish PIL image.
    bright = Image.new("RGB", (200, 120), (230, 230, 230))
    dark = Image.new("RGB", (200, 120), (20, 20, 20))
    for src in (bright, dark):
        out = preprocess_image(src)
        assert out.mode == "L"
        assert out.size[0] > 0 and out.size[1] > 0
        out = perspective_correct_image(src)
        assert out.mode == "RGB"
        assert out.size[0] > 0 and out.size[1] > 0
        out = perspective_correct_image_hough(src)
        assert out.mode == "RGB"
        assert out.size[0] > 0 and out.size[1] > 0
    # face detection: just assert it runs and returns the right shape/type
    # (a blank synthetic image has no face -- correctly returns None)
    assert detect_face_center(bright) is None
    print("ok")
