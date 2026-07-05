"""
cola_pdf_parser.py

Parses the declared application fields out of a TTB Form 5100.31 (COLA)
PDF's text layer, for the fields that matter to label verification, plus
the label image(s)' stated physical dimensions (needed for the absolute
size check in label_size_requirements.py).

DELIBERATELY DOES NOT EXTRACT (by design, not oversight): phone number,
fax number, email address, the applicant's personal printed name, or any
signature. Those fields exist on the form but have no bearing on
label-vs-application verification, and there's no reason for a personal
name and phone number to end up living in a test-data folder. Only
business-identifying fields that are legally required to appear on the
label itself are extracted -- the same information the label is already
obligated to publicly display.

IMPORTANT: TTB Form 5100.31 has multiple real-world revisions with
DIFFERENT field numbering. Confirmed against two actual revisions:
  - OMB No. 1512-0092: fields numbered 1-21, with NET CONTENTS (11) and
    ALCOHOL CONTENT (12) declared on Part I.
  - OMB No. 1513-0020: fields numbered 1-20 (shifted by the addition of
    a "SOURCE OF PRODUCT" field), and does NOT declare Net Contents or
    Alcohol Content on Part I at all -- those only appear on the label
    itself in this revision.
Field-label regexes below match on the label TEXT ("BRAND NAME
(Required)"), not a hardcoded leading number, specifically because the
number shifts between revisions but the label wording doesn't. Net
contents/alcohol content are treated as OPTIONAL (None if not found,
not an error) for the same reason -- see ParsedCOLA.is_complete().

Also parses every "Image Type: X" / "Actual Dimensions: W x H" pair on
the form (there can be multiple -- front, back, neck -- not just one),
since the Government Warning is frequently on the BACK label, which is
often a smaller image than the front/brand label. Confirmed on a real
record: front image 546x769px (larger), back image 288x516px (smaller,
but this is where the warning actually is) -- a "just take the biggest
image" extraction strategy would silently grab the wrong one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


@dataclass
class ImageDimensions:
    image_type: str  # e.g. "Brand (front) or keg collar", "Back", "Neck"
    width_inches: float
    height_inches: float


@dataclass
class ParsedCOLA:
    ttb_id: Optional[str]
    brand_name: Optional[str]
    class_type: Optional[str]
    alcohol_content: Optional[str]   # may be None -- not declared on all form revisions
    net_contents: Optional[str]      # may be None -- not declared on all form revisions
    bottler: Optional[str]
    status: Optional[str]  # e.g. "APPROVED", "SURRENDERED" -- for filtering/context only
    image_dimensions: list[ImageDimensions] = field(default_factory=list)

    def is_complete(self) -> bool:
        """
        True if the fields that are ALWAYS present regardless of form
        revision were successfully parsed. net_contents/alcohol_content
        are intentionally excluded here -- they're legitimately absent
        from Part I on some real form revisions (see module docstring),
        so requiring them would reject valid records for no good reason.
        """
        return all([self.brand_name, self.class_type, self.bottler])

    def has_scale_reference(self) -> bool:
        """
        True if at least one image's physical dimensions were parsed.

        Not currently called by main.py's endpoint (which checks
        parsed.net_contents directly, per-image, since the multi-image
        refactor) -- kept as a small public utility on ParsedCOLA for
        future modularity, e.g. if a caller wants to check "does this
        record have any scale reference at all" before deciding whether
        to attempt the absolute size check, without needing to know the
        internal image_dimensions shape.
        """
        return len(self.image_dimensions) > 0

    def as_field_dict(self) -> dict[str, str]:
        """
        Shape expected by field_matcher.verify_label(). Only includes
        keys for fields that were actually found -- net_contents/
        alcohol_content are OMITTED (not sent as empty strings) when not
        declared on the application, so verify_label() simply doesn't
        check them rather than failing them for a mismatch that was never
        possible to declare in the first place.
        """
        d = {
            "brand_name": self.brand_name or "",
            "class_type": self.class_type or "",
            "bottler": self.bottler or "",
        }
        if self.alcohol_content:
            d["alcohol_content"] = self.alcohol_content
        if self.net_contents:
            d["net_contents"] = self.net_contents
        return d


def _extract_after_label(text: str, label_pattern: str) -> Optional[str]:
    """
    Find a field label (e.g. "BRAND NAME (Required)") and return the
    next non-empty line as its value. Standard TTB form layout: label on
    its own line(s), value on the line(s) immediately after. The pattern
    itself should NOT include a hardcoded leading field number -- match
    on the label text only, since numbering shifts between form revisions.
    """
    match = re.search(label_pattern, text)
    if not match:
        return None
    remainder = text[match.end():]
    for line in remainder.split("\n"):
        line = line.strip()
        if line:
            return line
    return None


def parse_cola_pdf(pdf_path: Path) -> ParsedCOLA:
    doc = fitz.open(pdf_path)
    full_text = "\n".join(page.get_text() for page in doc)

    ttb_id_match = re.search(r"TTB ID\s*\n\s*(\d+)", full_text)
    ttb_id = ttb_id_match.group(1) if ttb_id_match else None

    brand_name = _extract_after_label(full_text, r"\d+\.\s*BRAND NAME \(Required\)")

    bottler = None
    field_match = re.search(
        r"\d+\.\s*NAME AND ADDRESS OF APPLICANT.*?\(Required\)\s*\n", full_text, re.DOTALL
    )
    if field_match:
        remainder = full_text[field_match.end():]
        lines = []
        dba_name = None
        for line in remainder.split("\n"):
            line = line.strip()
            if not line:
                break
            if "(Used on label)" in line:
                # The DBA/trade name actually printed on the physical label,
                # which can differ from the legal permit-holder name above
                # (e.g. "Cutler's Artisan Spirits, Inc." on the permit vs.
                # "BLACK MARKET SPIRITS" on the actual label). This is
                # exactly what the label is expected to say, so it needs to
                # be part of the expected bottler text, not discarded.
                dba_name = line.replace("(Used on label)", "").strip()
                break
            if re.match(r"^\d+[a-z]?\.", line):
                break
            lines.append(line)
        if dba_name:
            lines.append(dba_name)
        if lines:
            bottler = ", ".join(lines)

    net_contents = _extract_after_label(full_text, r"\d+\.\s*NET CONTENTS")
    alcohol_content = _extract_after_label(full_text, r"\d+\.\s*ALCOHOL CONTENT")

    class_type = _extract_after_label(full_text, r"CLASS/TYPE DESCRIPTION")

    status_match = re.search(r"STATUS\s*\n\s*THE STATUS IS (\w+)", full_text)
    status = status_match.group(1) if status_match else None

    image_dimensions = []
    for type_match in re.finditer(r"Image Type:\s*\n\s*([^\n]+)", full_text):
        image_type = type_match.group(1).strip()
        remainder = full_text[type_match.end():type_match.end() + 200]
        dims_match = re.search(
            r"Actual Dimensions:\s*([\d.]+)\s*inches?\s*W\s*X\s*([\d.]+)\s*inches?\s*H",
            remainder, re.IGNORECASE,
        )
        if dims_match:
            image_dimensions.append(ImageDimensions(
                image_type=image_type,
                width_inches=float(dims_match.group(1)),
                height_inches=float(dims_match.group(2)),
            ))

    return ParsedCOLA(
        ttb_id=ttb_id, brand_name=brand_name, class_type=class_type,
        alcohol_content=alcohol_content, net_contents=net_contents,
        bottler=bottler, status=status, image_dimensions=image_dimensions,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cola_pdf_parser.py <pdf_path>")
        sys.exit(1)

    result = parse_cola_pdf(Path(sys.argv[1]))
    print(f"TTB ID:          {result.ttb_id}")
    print(f"Brand name:      {result.brand_name}")
    print(f"Class/type:      {result.class_type}")
    print(f"Alcohol content: {result.alcohol_content}")
    print(f"Net contents:    {result.net_contents}")
    print(f"Bottler:         {result.bottler}")
    print(f"Status:          {result.status}")
    print("Image dimensions:")
    for d in result.image_dimensions:
        print(f"  {d.image_type}: {d.width_inches}in W x {d.height_inches}in H")
    print(f"Complete:        {result.is_complete()}")
