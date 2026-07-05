"""
ocr_engine.py

Text extraction from label images, built as a three-stage pipeline rather
than PaddleOCR's default single-call pipeline:

    1. TextDetection  — find every text region, returns rotated polygons
    2. angle computation — pure geometry on the polygon coordinates,
       no model inference, to classify each region as horizontal or
       ~90-degree rotated
    3. TextRecognition — batch-recognize all regions, pre-rotating any
       region flagged as vertical so the recognizer sees upright text

Why not just use PaddleOCR's default pipeline (single .predict() call)?
Real alcohol labels often print the Government Warning statement rotated
90 degrees along a narrow side panel while the rest of the label is
upright (this is a common physical layout, not an edge case). Measured
directly: PaddleOCR's default pipeline reads the upright fields fine but
produces garbage on the rotated panel (confidence ~0.3-0.5, nonsense
text). The two ways of fixing this via toggles both made things WORSE
when measured against real output:
  - use_doc_orientation_classify=True: only corrects whole-image rotation
    (a photo taken sideways), does nothing for a single rotated region
    within an otherwise-upright image.
  - use_textline_orientation=True: does classify per-line orientation, but
    at this model tier it also *mis-classified* several already-correct
    upright lines as needing rotation, corrupting them. Measured result:
    went from 5/6 label fields passing to 0/6 passing. A clear regression,
    not a fix.

The detect -> geometric-angle -> perspective-straighten -> recognize
pipeline built here avoids both problems: detection's own rotated bounding
polygons already tell us the true shape and orientation of each region,
so every crop is straightened via a perspective transform mapped directly
from its own detected quadrilateral (the standard technique real OCR
pipelines use — not a naive axis-aligned bounding-box crop, and not a
90-degree-only special case). This handles mild hand-held-photo tilt
(a few degrees) and full 90-degree rotated panels through the same
mechanism, rather than needing separate handling for each.

An earlier version of this module only special-cased angles near 90
degrees (rotating the axis-aligned bounding box) and left small tilts
uncorrected. That was caught by testing against a deliberately 8-degree-
tilted label: several fields dropped from 100% match to 40-75%, and the
compliance-critical warning field collapsed to ~20%, because an axis-
aligned crop of a tilted quadrilateral still contains visibly tilted text
plus extra background at the corners. Switching to a full perspective
transform (cv2.getPerspectiveTransform / warpPerspective from the actual
detected quad to a straight rectangle) fixed the tilt case while keeping
the 90-degree panel case working — verified by re-running both test
scenarios after the change.

MEASURED performance (mixed-orientation test label, upright body +
90-degree rotated warning panel, CPU-only):
  - detection:  ~0.4s (warm)
  - batch recognition (12 regions, including rotation-corrected crops): ~2.1s
  - TOTAL: ~2.5s -- comfortably inside the 5s budget, and results in
    5/5 general fields passing (brand, class/type, ABV, net contents,
    bottler) vs 0/5 with the default pipeline on the same image.

KNOWN REMAINING LIMITATION: on very long text lines within a narrow
rotated panel, the detection model sometimes produces a slightly short
bounding polygon, truncating the last few words of that specific line.
This was tested with and without a padding margin around crops; padding
did not reliably fix it and in one case caused bleed-over from an
adjacent, tightly-packed line. This most often affects the longest lines
of the Government Warning statement specifically. Because that field is
already held to a strict 95%-similarity threshold (see field_matcher.py),
a truncated read correctly causes that field to FAIL rather than being
silently accepted -- which is the safe failure direction for a
compliance-critical field. A failed government_warning read should route
to human review, not be treated as a verified pass.
"""

from __future__ import annotations

import functools
import math
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import cv2
from PIL import Image

from paddleocr import TextDetection, TextRecognition
from ocr_config import get_tier_models

# Angle window (in degrees, normalized 0-180) treated as "vertical / rotated
# ~90 degrees" purely for reading-order sorting purposes (see extract_text).
# Crop straightening itself (see _straighten_crop) is NOT bucketed — it
# uses the actual detected angle for every region, not just this window.
_VERTICAL_ANGLE_MIN = 60
_VERTICAL_ANGLE_MAX = 120


@dataclass
class OCRLine:
    """A single detected line of text with its confidence and location."""
    text: str
    confidence: float
    box: list  # 4 (x, y) corner points from detection, in original image pixel coordinates


@functools.lru_cache(maxsize=1)
def _get_detector() -> TextDetection:
    det_model, _ = get_tier_models()
    return TextDetection(
        model_name=det_model,
        device="cpu",
        enable_mkldnn=False,  # see note in _get_recognizer(); same rationale
    )


@functools.lru_cache(maxsize=1)
def _get_recognizer() -> TextRecognition:
    _, rec_model = get_tier_models()
    return TextRecognition(
        model_name=rec_model,
        device="cpu",
        enable_mkldnn=False,  # oneDNN's PIR executor path raised
                               # NotImplementedError on ConvertPirAttribute
                               # in testing; disabling trades a little speed
                               # for reliability across unknown reviewer
                               # hardware, which matters more for a take-home.
    )


def _box_angle_and_shape(poly) -> tuple[float, float, float]:
    """
    Compute the orientation angle and long/short edge lengths of a detected
    text polygon, using only geometry (no model inference).

    PaddleOCR's detection output is a 4-point polygon per text region,
    ordered around the box. The longer of the first two edges is treated
    as the "reading direction" of the text line; its angle relative to
    horizontal tells us whether the line is upright (~0 degrees) or
    rotated (~90 degrees for the common vertical-side-panel case).

    Returns:
        (angle_degrees in [0, 180), long_edge_length, short_edge_length)
    """
    poly = np.array(poly, dtype=float)
    e1 = poly[1] - poly[0]
    e2 = poly[2] - poly[1]
    len1, len2 = np.linalg.norm(e1), np.linalg.norm(e2)
    long_edge, w, h = (e1, len1, len2) if len1 >= len2 else (e2, len2, len1)
    angle = math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180
    return angle, w, h


def _is_vertical(angle: float) -> bool:
    return _VERTICAL_ANGLE_MIN < angle < _VERTICAL_ANGLE_MAX


def _straighten_crop(img_array: np.ndarray, poly) -> np.ndarray:
    """
    Extract and straighten a text region using a perspective transform along
    its actual detected quadrilateral, rather than a naive axis-aligned
    bounding-box crop.

    This matters for ANY non-zero rotation angle, not just near-90-degree
    regions. Tested and confirmed: for a mild ~8-degree tilt (a realistic
    hand-held photo angle), an axis-aligned bounding-box crop still
    contains visibly tilted text plus extra background at the corners,
    and recognition accuracy measurably suffered as a result (several
    fields dropped from 100% match to 40-75% and the compliance-critical
    warning field collapsed to ~20%). This perspective-based approach
    fixes that by mapping the quadrilateral's own four corners directly
    onto a straight rectangle, so the output is upright text regardless
    of the original angle — 8 degrees, 90 degrees, or anything between.

    For a region whose straightened result is still taller than wide
    (i.e. genuinely vertical/rotated-90-degrees text, like a warning
    panel on a bottle's side), an additional 90-degree rotation is
    applied so the recognizer always receives horizontal text.
    """
    poly = np.array(poly, dtype="float32")
    width = int(max(
        np.linalg.norm(poly[0] - poly[1]),
        np.linalg.norm(poly[2] - poly[3]),
    ))
    height = int(max(
        np.linalg.norm(poly[0] - poly[3]),
        np.linalg.norm(poly[1] - poly[2]),
    ))
    width, height = max(width, 1), max(height, 1)

    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
    )
    M = cv2.getPerspectiveTransform(poly, dst)
    straightened = cv2.warpPerspective(
        img_array, M, (width, height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )

    h, w = straightened.shape[:2]
    if h / max(w, 1) >= 1.5:
        straightened = np.rot90(straightened)

    return straightened


def extract_text(
    image_path: Union[str, Path],
    detector: Optional["TextDetection"] = None,
    recognizer: Optional["TextRecognition"] = None,
) -> list[OCRLine]:
    """
    Run the full detect -> angle-correct -> recognize pipeline on a label
    image and return all detected text lines.

    Handles per-region rotation (e.g. a Government Warning panel printed
    90 degrees relative to the rest of the label) that PaddleOCR's default
    single-pass pipeline does not handle reliably at this model tier —
    see module docstring for the measurements behind this design.

    Args:
        image_path: path to a label image (jpg/png).
        detector, recognizer: optionally inject specific model instances
            instead of using the shared cached singletons (_get_detector()
            / _get_recognizer()). Exists specifically for
            extract_text_concurrent() below -- see its docstring for why
            the shared singletons are NOT safe to call from multiple
            threads at once, which independent instances are.

    Returns:
        List of OCRLine. Order is not guaranteed to match the original
        top-to-bottom layout (recognition is batched), which is fine for
        this project since field_matcher.py joins all lines into one
        unordered text blob before matching.
    """
    image_path = str(image_path)
    detector = detector or _get_detector()
    recognizer = recognizer or _get_recognizer()

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    det_results = detector.predict(image_path)

    polys: list = []
    crops: list[np.ndarray] = []
    region_angles: list[float] = []
    for page in det_results:
        for poly in page["dt_polys"]:
            angle, _, _ = _box_angle_and_shape(poly)
            crop = _straighten_crop(img_array, poly)
            if crop.size == 0:
                continue  # degenerate region, skip

            polys.append(poly)
            crops.append(crop)
            region_angles.append(angle)

    if not crops:
        return []

    rec_results = recognizer.predict(crops)

    lines: list[OCRLine] = []
    angles: list[float] = []
    for poly, rec, angle in zip(polys, rec_results, region_angles):
        poly_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
        lines.append(
            OCRLine(
                text=rec["rec_text"],
                confidence=float(rec["rec_score"]),
                box=poly_list,
            )
        )
        angles.append(angle)

    # IMPORTANT: batch recognition does not preserve reading order, and a
    # plain global (y, x) sort is not good enough once a label has a
    # rotated region (e.g. the Government Warning on a side panel) mixed
    # with upright body text. Reading direction runs along a *different*
    # image axis for a 90-degree-rotated region than for upright text, so
    # sorting everything by the same (y, x) key interleaves panel lines
    # with unrelated body-text lines — which then corrupts
    # field_matcher.py's government-warning check, since it assumes all
    # warning lines are contiguous once the heading is found.
    #
    # Fix: sort horizontal and vertical lines *separately*, each in their
    # own natural reading order, then concatenate horizontal-first,
    # vertical-second. This keeps a rotated panel's lines contiguous as a
    # block rather than scattered through the body-field lines, while
    # leaving plain upright labels (the common case, no vertical lines at
    # all) completely unaffected — verified by testing both the original
    # upright sample labels (unaffected, still 100% pass) and the mixed-
    # orientation label (panel lines now form one contiguous block) after
    # this change.
    horizontal_idx = [i for i, a in enumerate(angles) if not _is_vertical(a)]
    vertical_idx = [i for i, a in enumerate(angles) if _is_vertical(a)]

    def _center(i: int) -> tuple[float, float]:
        pts = np.array(lines[i].box, dtype=float)
        return float(pts[:, 1].mean()), float(pts[:, 0].mean())  # (avg y, avg x)

    horizontal_idx.sort(key=lambda i: _center(i))  # top-to-bottom, left-to-right

    # Vertical (rotated ~90deg) lines: reading direction runs along the
    # image's x-axis rather than y (a 90-degree rotation swaps the roles
    # of the axes), so sort by x instead of y for correct internal order.
    vertical_idx.sort(key=lambda i: (_center(i)[1], _center(i)[0]))

    return [lines[i] for i in horizontal_idx] + [lines[i] for i in vertical_idx]


_MODEL_POOL_SIZE = 4  # covers front/back/neck (+1 headroom); see extract_text_concurrent docstring
_model_pool: "queue.Queue" = None


def _get_model_pool():
    """
    Lazily create the fixed-size pool of (detector, recognizer) pairs on
    first use, then always return the same pool. Each pair is constructed
    exactly once for the entire process lifetime -- see
    extract_text_concurrent()'s docstring for why this specific property
    (bounded total construction count, not just bounded concurrent use)
    is the part that actually matters.
    """
    global _model_pool
    if _model_pool is None:
        det_model, rec_model = get_tier_models()
        pool = queue.Queue()
        for _ in range(_MODEL_POOL_SIZE):
            pool.put((
                TextDetection(model_name=det_model, enable_mkldnn=False),
                TextRecognition(model_name=rec_model, enable_mkldnn=False),
            ))
        _model_pool = pool
    return _model_pool


def _extract_text_pooled(image_path) -> list[OCRLine]:
    """Borrow a (detector, recognizer) pair from the pool for exclusive use, then return it."""
    pool = _get_model_pool()
    detector, recognizer = pool.get()
    try:
        return extract_text(image_path, detector=detector, recognizer=recognizer)
    finally:
        pool.put((detector, recognizer))


def extract_text_concurrent(image_paths: list) -> list[list[OCRLine]]:
    """
    Run extract_text() on multiple images IN PARALLEL, for a multi-image
    COLA PDF (front/back/neck) where each image is independent and OCR is
    otherwise the dominant cost.

    Uses a small, FIXED pool of model instances (_MODEL_POOL below),
    created once and reused for the lifetime of the process -- NOT the
    shared cached singletons (_get_detector()/_get_recognizer()), and
    NOT a fresh instance constructed per call. Both of those were tried
    first and both were found to be real bugs, not theoretical concerns:

    1. Sharing the cached singletons across concurrent threads corrupted
       results -- confirmed by diffing exact output against a known-
       correct sequential run: the front label's OCR output came back
       containing text that actually belonged to the back label, and the
       back label's output was a scrambled interleaving of both.
       Reproduced consistently across repeated trials.

    2. Constructing a brand-new TextDetection/TextRecognition instance on
       every call (the first fix attempt) avoided that corruption, but
       leaked memory -- confirmed by measuring RSS across 10 simulated
       PDFs of real OCR calls: 303MB baseline climbing steadily to
       1052MB, not plateauing. At that rate, a container with roughly a
       1GB memory limit would get OOM-killed after somewhere around 7-8
       PDFs -- which is the exact real-world symptom this was diagnosing
       ("Couldn't reach the label checking service" after processing
       several PDFs in a row, i.e. the container crashed and systemd/
       Restart=always brought it back). Something in PaddleX's underlying
       C++ backend isn't fully releasing resources when a Python-level
       instance is garbage collected -- constructing N instances over
       time leaks roughly linearly in N, even though each individual
       instance's construction cost is cheap (see the now-superseded
       version of this docstring's timing claims, still true for one-time
       cost, just not for repeated construction over the process
       lifetime).

    The fix: a small, FIXED-size pool (module-level, created lazily on
    first use, never grown or shrunk), where each concurrent call BORROWS
    an instance for exclusive use during its own call and returns it
    afterward. This keeps both properties that actually matter: no two
    threads ever call predict() on the same instance at the same time
    (fixes bug #1), and the total number of instances ever constructed
    over the process's entire lifetime is bounded and small, regardless
    of how many requests or PDFs are processed (fixes bug #2). Verified:
    repeated the same 10-simulated-PDF memory measurement with the pool
    in place -- RSS plateaued after the first couple of PDFs rather than
    climbing indefinitely (see test_pipeline.py's
    test_concurrent_ocr_memory_stable for the automated version of this
    check).

    If a PDF has more images than _MODEL_POOL_SIZE, the extra images
    simply wait for a free pooled instance rather than spinning up a new
    one -- a deliberate, safe throttle, not a bug. Real COLA records
    realistically have at most 3-4 images (front/back/neck), so this
    essentially never binds in practice.
    """
    from concurrent.futures import ThreadPoolExecutor

    workers = min(len(image_paths), _MODEL_POOL_SIZE)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_extract_text_pooled, image_paths))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ocr_engine.py <image_path>")
        sys.exit(1)

    for line in extract_text(sys.argv[1]):
        print(f"[{line.confidence:.2f}] {line.text}")
