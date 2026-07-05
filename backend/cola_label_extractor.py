"""
cola_label_extractor.py

Extracts label images from a COLA PDF, matched against the form's OWN
declared "Image Type: X" / "Actual Dimensions: W x H" entries (see
cola_pdf_parser.py) rather than a generic size/aspect-ratio guess.

WHY: a COLA PDF can have multiple real label images -- front, back,
neck -- and the Government Warning is frequently on the BACK label, not
the front/brand image. An earlier version of this module picked "the
single largest embedded image," which is wrong whenever the back label
happens to be smaller than the front (confirmed on a real record: front
546x769px vs. back 288x516px -- back is smaller, but that's where the
warning actually is).

The fix: the PDF already tells us exactly how many real label images
there are and their true physical aspect ratio (from "Actual
Dimensions"). Matching each embedded image to the closest-aspect-ratio
declared entry does two things at once: correctly identifies ALL real
label images (front, back, AND neck, even when the neck label is a very
thin, unusually-shaped strip that a generic aspect-ratio filter would
have excluded), and naturally excludes non-label images like signatures
and TTB seals, since those have no corresponding declared entry to match
against at all -- no separate signature-detection heuristic needed.

Verified against real records: aspect-ratio matching cleanly separated
a signature (aspect ~3.0-3.1, no match) from front/back/neck labels
(aspects 0.5-1.1, each matching its declared counterpart), including a
real neck label with an extreme 0.125 aspect ratio that a fixed
MIN_ASPECT/MAX_ASPECT range would have wrongly excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

from cola_pdf_parser import ImageDimensions

MIN_AREA_PX = 20 * 20  # only excludes genuinely tiny icons/decoration; real filtering is aspect-match
ASPECT_MATCH_TOLERANCE = 0.20  # fraction. Calibrated against real records: a genuine front-label
# match measured 15.3% off from its declared aspect (likely a small margin/border baked into the
# raster that isn't part of the declared label dimensions), so 0.15 was too tight and wrongly
# excluded it. 0.20 catches that case with room to spare, while a real signature's aspect ratio
# (~3.0-4.1 in every record checked) stays roughly 20x further off than any real label match --
# no risk of a false match at this tolerance.

TTB_ERROR_SIGNATURES = [
    "An error has occurred in the system",
    "Unable to process request",
]


@dataclass
class ExtractedLabel:
    image_bytes: bytes
    width_px: int
    height_px: int
    ext: str
    image_type: str              # from the matched declared entry, e.g. "Back"
    width_inches: float          # from the matched declared entry -- real physical scale
    height_inches: float


def is_ttb_error_page(doc: fitz.Document) -> bool:
    first_page_text = doc[0].get_text() if len(doc) > 0 else ""
    return any(sig in first_page_text for sig in TTB_ERROR_SIGNATURES)


def _open(pdf_source: Union[Path, bytes]) -> fitz.Document:
    if isinstance(pdf_source, (bytes, bytearray)):
        return fitz.open(stream=pdf_source, filetype="pdf")
    return fitz.open(pdf_source)


def extract_all_label_images(
    pdf_source: Union[Path, bytes],
    declared_dimensions: list[ImageDimensions],
) -> tuple[list[ExtractedLabel], str]:
    """
    Extract every real label image from a COLA PDF, matched against its
    own declared Image Type/Dimensions entries.

    Args:
        pdf_source: a path to a PDF file, or raw PDF bytes.
        declared_dimensions: from cola_pdf_parser.parse_cola_pdf(...).image_dimensions.
            Required -- without at least one declared entry to match
            against, there's no reliable way to distinguish a label image
            from a signature or decorative graphic, so this returns
            "no_label_found" rather than guessing.

    Returns:
        (list of ExtractedLabel, status), status one of:
        "ok", "error_page", "no_label_found".
    """
    doc = _open(pdf_source)

    if is_ttb_error_page(doc):
        return [], "error_page"

    if not declared_dimensions:
        return [], "no_label_found"

    raw_images = []
    for page in doc:
        for img in page.get_images(full=True):
            base_image = doc.extract_image(img[0])
            w, h = base_image.get("width", 0), base_image.get("height", 0)
            if w * h >= MIN_AREA_PX:
                raw_images.append((w / h, base_image, w, h))

    if not raw_images:
        return [], "no_label_found"

    unclaimed = list(raw_images)
    results: list[ExtractedLabel] = []

    for decl in declared_dimensions:
        target_aspect = decl.width_inches / decl.height_inches
        best_idx, best_diff = None, float("inf")
        for i, (aspect, _, _, _) in enumerate(unclaimed):
            diff = abs(aspect - target_aspect) / target_aspect
            if diff < best_diff:
                best_idx, best_diff = i, diff

        if best_idx is not None and best_diff <= ASPECT_MATCH_TOLERANCE:
            aspect, base_image, w, h = unclaimed.pop(best_idx)
            results.append(ExtractedLabel(
                image_bytes=base_image["image"], width_px=w, height_px=h,
                ext=base_image["ext"], image_type=decl.image_type,
                width_inches=decl.width_inches, height_inches=decl.height_inches,
            ))

    if not results:
        return [], "no_label_found"

    return results, "ok"
