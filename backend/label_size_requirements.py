"""
label_size_requirements.py

Real, physical (not relative) minimum type-size requirements for the
Government Warning statement, per 27 CFR 16.22(a) -- verified directly
against eCFR, not just taken on faith from a secondary summary:

    "(a) ... the statement required by Sec. 16.21 ... appear in a type
    size at least as large as specified in this paragraph, based on the
    net contents of the container: (1) 237 ml (8 fl. oz.) or less --
    1 mm; (2) more than 237 ml but not more than 3 liters -- 2 mm;
    (3) more than 3 liters -- 3 mm."
    -- 27 CFR 16.22(a), https://www.ecfr.gov/current/title-27/chapter-I/subchapter-A/part-16

Also from the same section (16.22(b), summarized not quoted): the words
"GOVERNMENT WARNING" must be capitalized and bold; the rest of the
statement must NOT be bold.

Why this exists as a SEPARATE module from field_matcher.py's existing
check_warning_text_size(): that function is a same-photo RELATIVE
heuristic (warning text vs. the rest of the label, in the same image,
no physical units) — useful when no physical scale reference is
available, which is the common case for an arbitrary uploaded photo.
This module does an ABSOLUTE check in real millimeters, which requires
knowing the image's physical scale (pixels per mm). That scale is only
available when we have an external reference for the image's real-world
size — for example, a COLA PDF's own "Actual Dimensions: W x H" label,
which is exactly what cola_pdf_parser.py will supply (planned, not yet
wired up as of this module's creation). Until that's connected, this
module's functions take px_per_mm as an explicit parameter rather than
computing it themselves.

NOT YET VERIFIED end-to-end against a real image with known physical
scale -- the arithmetic is tested with synthetic/mock scale values below.
Real validation happens once cola_pdf_parser.py can supply an actual
px_per_mm for a real label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_engine import OCRLine

ML_PER_FL_OZ = 29.5735

# 27 CFR 16.22(a) -- warning statement minimum type size by container size.
# List of (max_ml_for_this_tier, min_height_mm), checked in order.
WARNING_SIZE_TIERS_MM: list[tuple[float, float]] = [
    (237.0, 1.0),           # 237 mL (8 fl. oz.) or less
    (3000.0, 2.0),          # more than 237 mL, up to 3 L
    (float("inf"), 3.0),    # more than 3 L
]


def required_warning_height_mm(net_contents_ml: float) -> float:
    """Return the minimum required Government Warning type height in mm for a given container size."""
    for max_ml, min_mm in WARNING_SIZE_TIERS_MM:
        if net_contents_ml <= max_ml:
            return min_mm
    return WARNING_SIZE_TIERS_MM[-1][1]  # unreachable given the inf tier, but safe


def parse_net_contents_ml(text: str) -> float | None:
    """
    Best-effort parse of a net contents string into milliliters. Handles
    the common forms seen on labels and applications: "750 mL", "750ML",
    "750 MILLILITERS", "12 FL OZ", "1 L", "1.5 LITERS", "1 GAL".

    Returns None if the string can't be confidently parsed -- callers
    should treat that as "can't determine which size tier applies",
    not silently default to a tier that could be wrong.
    """
    import re
    text = text.strip().lower()
    match = re.search(r"([\d.]+)\s*(ml|milliliters?|millilitres?|l|liters?|litres?|"
                       r"fl\.?\s*oz\.?|fluid\s*ounces?|oz|ounces?|gal|gallons?)", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).replace(".", "").replace(" ", "")

    if unit.startswith("ml") or "millilit" in unit:
        return value
    if unit == "l" or "lit" in unit:
        return value * 1000.0
    if "floz" in unit or "fluid" in unit or unit in ("oz", "ounce", "ounces"):
        return value * ML_PER_FL_OZ
    if "gal" in unit:
        return value * 3785.41
    return None


@dataclass
class ScaleReference:
    """
    Physical scale for one label image: how many pixels correspond to one
    millimeter. Derived from a known physical dimension of the image
    (e.g. a COLA PDF's stated "Actual Dimensions: W x H") divided by the
    image's pixel dimensions along the same axis.
    """
    px_per_mm: float

    @classmethod
    def from_known_dimension(cls, pixel_length: float, physical_inches: float) -> "ScaleReference":
        physical_mm = physical_inches * 25.4
        if physical_mm <= 0:
            raise ValueError("physical_inches must be positive")
        return cls(px_per_mm=pixel_length / physical_mm)


def _line_height_px(line: OCRLine) -> float:
    """Same measurement as field_matcher._line_height -- short edge of the detected box."""
    pts = np.array(line.box, dtype=float)
    if pts.shape[0] < 4:
        return 0.0
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return float(min(e1, e2))


@dataclass
class AbsoluteSizeResult:
    matched: bool
    measured_mm: float
    required_mm: float
    detail: str


def check_warning_absolute_size(
    warning_lines: list[OCRLine],
    scale: ScaleReference,
    net_contents_ml: float,
) -> AbsoluteSizeResult:
    """
    Check the Government Warning's actual physical type size against the
    real 27 CFR 16.22(a) minimum for this container's net contents.

    Uses the SMALLEST line height among the warning lines, not the
    average/median -- the regulation sets a minimum, so the smallest
    line is the one that could actually violate it even if other lines
    are larger.
    """
    heights_px = [_line_height_px(l) for l in warning_lines if _line_height_px(l) > 0]
    if not heights_px:
        return AbsoluteSizeResult(
            matched=False, measured_mm=0.0, required_mm=required_warning_height_mm(net_contents_ml),
            detail="Could not measure warning text height.",
        )

    smallest_px = min(heights_px)
    measured_mm = smallest_px / scale.px_per_mm
    required_mm = required_warning_height_mm(net_contents_ml)

    matched = measured_mm >= required_mm
    detail = (
        f"Smallest warning line measures {measured_mm:.2f}mm "
        f"(required >= {required_mm}mm for a {net_contents_ml:.0f}mL container, "
        f"per 27 CFR 16.22(a))."
    )
    return AbsoluteSizeResult(matched=matched, measured_mm=measured_mm, required_mm=required_mm, detail=detail)
