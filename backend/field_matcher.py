"""
field_matcher.py

Compares OCR-extracted label text against the field values submitted on a
COLA application, and produces a pass/fail verdict per field.

Two distinct matching strategies, driven directly by the stakeholder notes:

1. FUZZY matching (brand name, class/type, ABV, net contents, bottler
   name/address) — Dave's interview flagged "STONE'S THROW" (label) vs
   "Stone's Throw" (application) as a false mismatch under naive exact
   comparison. Real applications will have case differences, punctuation
   differences, and OCR noise that shouldn't fail an otherwise-correct
   label. We normalize and use rapidfuzz similarity scoring with a
   configurable threshold.

2. STRICT matching (the Government Warning statement only) — Jenny's
   interview was explicit: this one has to be word-for-word, in
   ALL CAPS, with "GOVERNMENT WARNING:" leading. She specifically flagged
   a real rejection over title-case ("Government Warning" instead of
   "GOVERNMENT WARNING"). Fuzzy-matching this field would silently defeat
   the exact compliance check it exists to enforce, so it deliberately
   bypasses the fuzzy path entirely.

KNOWN LIMITATION (documented, not hidden): OCR text output does not carry
font-weight information, so this module cannot verify the "bold" part of
Jenny's requirement — only the text content and casing. Detecting bold
would require image-level analysis (e.g. stroke-width heuristics on the
detected text region) which is out of scope for this prototype. Flagged
here and in the README rather than silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from rapidfuzz import fuzz

from ocr_engine import OCRLine

# The statutory Government Warning text required on all alcohol beverage
# labels (27 U.S.C. § 215). This is federally mandated legal text that
# every compliant label must reproduce verbatim — it is the thing being
# checked *for*, not reference material being copied from a third party.
GOVERNMENT_WARNING_TEXT = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth "
    "defects. (2) Consumption of alcoholic beverages impairs your ability to "
    "drive a car or operate machinery, and may cause health problems."
)

GOVERNMENT_WARNING_LEAD = "GOVERNMENT WARNING:"

FUZZY_MATCH_THRESHOLD = 88.0  # 0-100 similarity score; tuned, see README for rationale

# Per-field threshold overrides. Default (FUZZY_MATCH_THRESHOLD) applies unless
# a field is listed here. Currently only "bottler" is overridden -- see
# match_field_fuzzy() docstring for why this one field is structurally
# different from the others (calibrated against a real COLA application,
# not guessed).
FIELD_THRESHOLD_OVERRIDES: dict[str, float] = {
    "bottler": 65.0,
    "class_type": 70.0,
}

# Common unit words/abbreviations seen on alcohol labels, mapped to a
# canonical short form. Applications tend to spell units out ("750
# MILLILITERS"); printed labels almost always abbreviate them ("750mL").
# Without this, those two strings share zero overlapping tokens and score
# near 0 no matter how good the fuzzy algorithm is -- this was found by
# testing against a real COLA application/label pair, not a hypothetical.
_UNIT_ALIASES = {
    "milliliters": "ml", "millilitres": "ml", "milliliter": "ml", "millilitre": "ml",
    "liters": "l", "litres": "l", "liter": "l", "litre": "l",
    "fluid ounces": "floz", "fluid ounce": "floz", "fl oz": "floz", "fl. oz.": "floz",
    "ounces": "oz", "ounce": "oz",
    "gallons": "gal", "gallon": "gal",
}


def _normalize_units(text: str) -> str:
    """
    Replace known unit words with a canonical abbreviation, and collapse
    the space between a number and its unit, so "750 MILLILITERS" and
    "750ML" normalize to the same thing. Applied on top of _normalize().
    """
    for word, canonical in sorted(_UNIT_ALIASES.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"\b{re.escape(word)}\b", canonical, text)
    # collapse "750 ml" -> "750ml" so spacing differences don't matter either
    text = re.sub(r"(\d)\s+(ml|l|floz|oz|gal)\b", r"\1\2", text)
    return text


class MatchMethod(str, Enum):
    FUZZY = "fuzzy"
    STRICT = "strict"


@dataclass
class FieldResult:
    field_name: str
    expected_value: str
    matched: bool
    score: float  # 0-100
    method: MatchMethod
    detail: str = ""


@dataclass
class VerificationReport:
    overall_pass: bool
    fields: list[FieldResult] = field(default_factory=list)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip most punctuation for fuzzy comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s%./]", " ", text)  # keep %, ., / since they matter in ABV/volume fields
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _joined_blob(ocr_lines: list[OCRLine]) -> str:
    """All detected text joined into one searchable blob (case preserved)."""
    return " ".join(line.text for line in ocr_lines)


WARNING_LINE_OVERLAP_THRESHOLD = 0.4


def select_warning_lines(ocr_lines: list[OCRLine]) -> list[OCRLine]:
    """
    Identify which detected lines are (likely) part of the Government
    Warning statement, by content rather than position in the list.

    Shared by match_government_warning() (text/case checking) and
    label_size_requirements-based absolute size checks (which need the
    same set of lines to measure physical height on) so both use
    identical, tested line-selection logic rather than two independent
    implementations that could drift apart. See match_government_warning()
    docstring for why content-based selection replaced an earlier,
    position-based approach that broke on real rotated-panel labels.
    """
    warning_vocab = set(_normalize(GOVERNMENT_WARNING_TEXT).split())
    warning_lines = []
    for line in ocr_lines:
        line_words = set(_normalize(line.text).split())
        if not line_words:
            continue
        overlap = len(line_words & warning_vocab) / len(line_words)
        if overlap >= WARNING_LINE_OVERLAP_THRESHOLD:
            warning_lines.append(line)
    return warning_lines


def match_field_fuzzy(field_name: str, expected_value: str, ocr_lines: list[OCRLine]) -> FieldResult:
    """
    Fuzzy-match a single expected field value against the full OCR text blob.

    Takes the max of three rapidfuzz scores, each covering a different
    real-world mismatch pattern:
      - partial_ratio: a short expected value (e.g. "750 mL") matching a
        substring of the larger blob.
      - token_sort_ratio: word-order differences (order-invariant at the
        word level).
      - token_set_ratio: the expected value contains EXTRA words beyond
        what's on the label -- e.g. TTB's assigned class/type description
        "TABLE RED WINE" vs. a label that just says "Red Wine". Verified
        against a real COLA record: token_set_ratio scored this 100,
        while partial_ratio/token_sort_ratio both scored under 10.

    Unit words are normalized before scoring (see _normalize_units) so
    "750 MILLILITERS" (how applications write it) and "750mL" (how labels
    print it) aren't scored as unrelated strings.

    Threshold is per-field via FIELD_THRESHOLD_OVERRIDES, defaulting to
    FUZZY_MATCH_THRESHOLD. Two fields are currently overridden:
      - "bottler" (65 vs. default 88): a label is only required to print
        city/state/zip for the bottler, while the applicant's declared
        address on file usually includes the full street address and
        sometimes a corporate suffix ("INC."). Those words are
        legitimately absent from a fully compliant label, not evidence of
        a mismatch -- verified against a real COLA record where a genuine
        match scored 80 (missing exactly the street address and "INC.")
        while a deliberately wrong bottler scored 33, leaving a wide, safe
        margin for a lower threshold.
      - "class_type" (70 vs. default 88): TTB's assigned classification
        uses formal regulatory vocabulary ("TABLE RED WINE") that doesn't
        always literally appear on the label, which uses consumer-facing
        wording instead ("Red Wine"). Verified against the same real COLA
        record: genuine match scored 78.6, a genuinely wrong class/type
        scored 42.9, and an adjacent-but-wrong one (same beverage
        category, wrong subtype) scored 53.8 -- comfortable margin below
        70 for both wrong cases.
    """
    blob_norm = _normalize_units(_normalize(_joined_blob(ocr_lines)))
    expected_norm = _normalize_units(_normalize(expected_value))

    if not expected_norm:
        return FieldResult(field_name, expected_value, False, 0.0, MatchMethod.FUZZY,
                            detail="Expected value is empty.")

    partial = fuzz.partial_ratio(expected_norm, blob_norm)
    token_sort = fuzz.token_sort_ratio(expected_norm, blob_norm)
    token_set = fuzz.token_set_ratio(expected_norm, blob_norm)
    score = max(partial, token_sort, token_set)

    threshold = FIELD_THRESHOLD_OVERRIDES.get(field_name, FUZZY_MATCH_THRESHOLD)
    matched = score >= threshold
    detail = (
        f"partial_ratio={partial:.1f}, token_sort_ratio={token_sort:.1f}, "
        f"token_set_ratio={token_set:.1f}, threshold={threshold}"
    )
    return FieldResult(field_name, expected_value, matched, score, MatchMethod.FUZZY, detail)


def match_government_warning(ocr_lines: list[OCRLine]) -> FieldResult:
    """
    Strict, word-for-word + case-sensitive check for the Government Warning.

    Two checks, both required to pass:
      1. The lead-in "GOVERNMENT WARNING:" must appear verbatim, in all
         caps, somewhere in the raw (non-lowercased) OCR text. This is the
         check that catches Jenny's "Government Warning" title-case
         rejection example.
      2. The full statutory text, once whitespace/newline differences are
         collapsed, must match the required wording closely (case-sensitive
         for the lead-in, case-insensitive for the body — the statute
         doesn't currently mandate all-caps body text, only that the
         "GOVERNMENT WARNING:" heading itself be capitalized). We allow a
         small amount of OCR-noise tolerance on the body via fuzzy scoring,
         but at a much higher threshold than general fields, since this is
         a compliance-critical field, not a cosmetic one.

    Body candidate selection is CONTENT-based, not positional. An earlier
    version took "everything from the heading to the end of the line
    list" — that broke on a real label (Cascade Winery, TTB ID
    03211001000018) where the Government Warning runs along a rotated
    side panel: some warning body lines sorted BEFORE the heading in
    ocr_engine's output, so "heading onward" silently dropped them,
    collapsing the score from what should have been ~100 down to ~32.
    Scoring the ENTIRE blob instead (no isolation at all) was tried too,
    but that reintroduces a different, earlier-caught bug: unrelated
    label content (brand name, ABV, "Contains Sulfites", etc.) dilutes
    token_sort_ratio even when the warning itself is a perfect match.

    The fix: classify each detected line by vocabulary overlap with the
    known statutory text (what fraction of the line's own words appear
    in it), and treat any line clearing a threshold as part of the
    warning, regardless of where it landed in the list. Verified against
    the real label this was found on: correctly separates all 4 genuine
    warning lines (overlap 1.00 each) from the other 10 label lines
    (brand, ABV, address, etc. — the highest false-candidate was
    "Produced and bottled by" at 0.25, safely below the 0.4 threshold).
    """
    warning_lines = select_warning_lines(ocr_lines)

    if not warning_lines:
        return FieldResult(
            "government_warning", GOVERNMENT_WARNING_TEXT, False, 0.0, MatchMethod.STRICT,
            detail="'GOVERNMENT WARNING:' heading not found on label at all.",
        )

    candidate_raw = re.sub(r"\s+", " ", " ".join(l.text for l in warning_lines)).strip()

    lead_present_correct_case = GOVERNMENT_WARNING_LEAD in candidate_raw
    lead_present_any_case = GOVERNMENT_WARNING_LEAD.lower() in candidate_raw.lower()

    STRICT_BODY_THRESHOLD = 95.0

    if not lead_present_any_case:
        return FieldResult(
            "government_warning", GOVERNMENT_WARNING_TEXT, False, 0.0, MatchMethod.STRICT,
            detail="'GOVERNMENT WARNING:' heading not found on label at all.",
        )

    body_score = fuzz.token_sort_ratio(_normalize(candidate_raw), _normalize(GOVERNMENT_WARNING_TEXT))

    if not lead_present_correct_case:
        return FieldResult(
            "government_warning", GOVERNMENT_WARNING_TEXT, False, body_score, MatchMethod.STRICT,
            detail="Heading found but not in required ALL CAPS form "
                   "(e.g. 'Government Warning' instead of 'GOVERNMENT WARNING:'). "
                   "Note: bold-face cannot be verified from OCR text alone — see README.",
        )

    if body_score < STRICT_BODY_THRESHOLD:
        return FieldResult(
            "government_warning", GOVERNMENT_WARNING_TEXT, False, body_score, MatchMethod.STRICT,
            detail=f"Heading OK, but statutory body text does not match closely enough "
                   f"(score={body_score:.1f}, required>={STRICT_BODY_THRESHOLD}).",
        )

    return FieldResult(
        "government_warning", GOVERNMENT_WARNING_TEXT, True, body_score, MatchMethod.STRICT,
        detail="Heading present in correct case; statutory text matches.",
    )


# Ratio (0-1): warning text's median line-height must be at least this
# fraction of the rest of the label's median line-height, or it gets
# flagged as suspiciously small for human review. Calibrated against a
# real generated compliant label: that label's genuine warning-to-body
# height ratio measured ~0.8, so 0.6 leaves real headroom for normal
# label typography while still catching a warning shrunk well below the
# rest of the label's text -- Jenny Park's real complaint pattern
# ("smaller font... burying it in tiny text").
WARNING_SIZE_RATIO_THRESHOLD = 0.6


def _line_height(line: OCRLine) -> float:
    """
    Estimate a detected text line's height in pixels from its bounding
    polygon, as a proxy for font size. Uses the shorter of the box's two
    edge-pair lengths (works regardless of whether the line was upright
    or came from a rotated region, since ocr_engine.py stores each
    OCRLine's box as the ORIGINAL detected quadrilateral, not the
    straightened crop).
    """
    pts = np.array(line.box, dtype=float)
    if pts.shape[0] < 4:
        return 0.0
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return float(min(e1, e2))


def check_warning_text_size(ocr_lines: list[OCRLine]) -> FieldResult:
    """
    Flag a Government Warning that's suspiciously small relative to the
    rest of the label's text.

    IMPORTANT, honestly stated: this is a same-photo RELATIVE heuristic,
    not a certified measurement. The actual TTB minimum type-size rule
    (27 CFR 16.21) is specified in physical units (mm) relative to
    container size, and there is no reliable way to recover physical
    scale from an arbitrary uploaded photo without a size reference in
    frame. What this CAN do: compare the warning's pixel line-height
    against the rest of the label's pixel line-height in the same image,
    which catches the specific failure pattern Jenny Park described --
    using a noticeably smaller font specifically for the warning to bury
    it -- without claiming to certify absolute legal type size. Treat a
    failure here as "verify by eye," not an automatic legal violation.

    Depends on line order from ocr_engine.extract_text(): the Government
    Warning heading is located by text search, and every line from that
    point to the end of the list is treated as "warning text"; everything
    before it is treated as "the rest of the label." This mirrors the
    same order assumption match_government_warning() already relies on.
    """
    heading_idx = next(
        (i for i, l in enumerate(ocr_lines) if GOVERNMENT_WARNING_LEAD.lower() in l.text.lower()),
        None,
    )
    if heading_idx is None:
        return FieldResult(
            "government_warning_size", "proportional to rest of label", False, 0.0, MatchMethod.STRICT,
            detail="Could not locate the Government Warning heading, so its text size "
                   "could not be checked against the rest of the label.",
        )

    warning_lines = ocr_lines[heading_idx:]
    body_lines = ocr_lines[:heading_idx]

    if not body_lines:
        return FieldResult(
            "government_warning_size", "proportional to rest of label", False, 0.0, MatchMethod.STRICT,
            detail="No other label text detected to compare the warning's size against.",
        )

    warning_heights = [_line_height(l) for l in warning_lines if _line_height(l) > 0]
    body_heights = [_line_height(l) for l in body_lines if _line_height(l) > 0]

    if not warning_heights or not body_heights:
        return FieldResult(
            "government_warning_size", "proportional to rest of label", False, 0.0, MatchMethod.STRICT,
            detail="Could not measure text height reliably enough to check warning size.",
        )

    warning_median = float(np.median(warning_heights))
    body_median = float(np.median(body_heights))
    ratio = warning_median / body_median if body_median > 0 else 0.0

    matched = ratio >= WARNING_SIZE_RATIO_THRESHOLD
    score = min(ratio / WARNING_SIZE_RATIO_THRESHOLD, 1.0) * 100 if WARNING_SIZE_RATIO_THRESHOLD > 0 else 0.0

    detail = (
        f"Warning text ~{warning_median:.0f}px tall vs. ~{body_median:.0f}px for the rest of the "
        f"label ({ratio*100:.0f}% of body text size; flagged below {WARNING_SIZE_RATIO_THRESHOLD*100:.0f}%). "
        f"This is a same-photo pixel-height comparison, not a certified physical measurement -- "
        f"verify by eye."
    )
    return FieldResult(
        "government_warning_size", "proportional to rest of label", matched, score, MatchMethod.STRICT, detail
    )


def verify_label(expected_fields: dict[str, str], ocr_lines: list[OCRLine]) -> VerificationReport:
    """
    Run full verification for one label.

    Args:
        expected_fields: dict of field_name -> value as submitted on the
            COLA application, e.g. {"brand_name": "OLD TOM DISTILLERY",
            "class_type": "Kentucky Straight Bourbon Whiskey", ...}.
            The government warning is handled separately/automatically and
            should NOT be included in this dict.
        ocr_lines: output of ocr_engine.extract_text() for the label image.

    Returns:
        VerificationReport with a per-field breakdown and an overall pass/fail.
    """
    results = [
        match_field_fuzzy(name, value, ocr_lines)
        for name, value in expected_fields.items()
    ]
    results.append(match_government_warning(ocr_lines))
    results.append(check_warning_text_size(ocr_lines))

    overall_pass = all(r.matched for r in results)
    return VerificationReport(overall_pass=overall_pass, fields=results)


if __name__ == "__main__":
    # Quick manual smoke test against the sample label + Dave's real-world edge case.
    from ocr_engine import extract_text

    lines = extract_text("../sample_labels/old_tom_bourbon.png")

    expected = {
        "brand_name": "Old Tom Distillery",   # deliberately different case than the label
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "bottler": "Old Tom Distillery Co., Bardstown, KY",
    }

    report = verify_label(expected, lines)
    print(f"OVERALL: {'PASS' if report.overall_pass else 'FAIL'}\n")
    for r in report.fields:
        status = "OK  " if r.matched else "FAIL"
        print(f"[{status}] {r.field_name:20s} score={r.score:5.1f} ({r.method.value})  {r.detail}")
