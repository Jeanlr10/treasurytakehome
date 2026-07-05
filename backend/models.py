"""
models.py

Pydantic schemas that define the JSON shape of the API, sitting on top of
the plain-dataclass internals in field_matcher.py.

Why a separate layer instead of just serializing FieldResult/VerificationReport
directly: those dataclasses are internal implementation detail (e.g. MatchMethod
enum, raw score floats) and we want a stable, documented JSON contract for
whoever calls this API — including FastAPI's auto-generated OpenAPI docs in
the next stage. Keeping them separate also means we can change the internal
matching logic without breaking the response shape API consumers depend on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from field_matcher import FieldResult, VerificationReport


class LabelSubmission(BaseModel):
    """
    The fields as submitted on the COLA application, to be checked against
    what's actually printed on the label image.

    The government warning is intentionally NOT part of this model — it's
    a fixed statutory requirement, not something an applicant "submits" a
    value for, and is always checked automatically (see field_matcher.py).
    """
    brand_name: str = Field(..., examples=["Old Tom Distillery"])
    class_type: str = Field(..., examples=["Kentucky Straight Bourbon Whiskey"])
    alcohol_content: str = Field(..., examples=["45% Alc./Vol. (90 Proof)"])
    net_contents: str = Field(..., examples=["750 mL"])
    bottler: str = Field(..., examples=["Old Tom Distillery Co., Bardstown, KY"])
    country_of_origin: str | None = Field(
        default=None,
        description="Required for imports only; omit for domestic products.",
        examples=["Scotland"],
    )

    def as_field_dict(self) -> dict[str, str]:
        """Convert to the plain dict shape field_matcher.verify_label() expects."""
        data = {
            "brand_name": self.brand_name,
            "class_type": self.class_type,
            "alcohol_content": self.alcohol_content,
            "net_contents": self.net_contents,
            "bottler": self.bottler,
        }
        if self.country_of_origin:
            data["country_of_origin"] = self.country_of_origin
        return data


class FieldVerificationResult(BaseModel):
    field_name: str
    expected_value: str
    matched: bool
    score: float = Field(..., ge=0, le=100, description="Similarity score, 0-100")
    method: str = Field(..., description="'fuzzy' or 'strict'")
    detail: str = Field(default="", description="Human-readable explanation of the result")

    @classmethod
    def from_field_result(cls, r: FieldResult) -> "FieldVerificationResult":
        return cls(
            field_name=r.field_name,
            expected_value=r.expected_value,
            matched=r.matched,
            score=r.score,
            method=r.method.value,
            detail=r.detail,
        )


class LabelImage(BaseModel):
    image_type: str = Field(..., description="e.g. 'Brand (front) or keg collar', 'Back', 'Neck', or 'Uploaded photo'")
    data_uri: str = Field(..., description="base64 data: URI, directly usable as an <img src>")


class VerificationResponse(BaseModel):
    overall_pass: bool
    fields: list[FieldVerificationResult]
    processing_time_ms: float = Field(..., description="Wall-clock time for OCR + matching combined")
    detected_text: list[str] = Field(
        default_factory=list,
        description="Every line of text the OCR engine actually detected on the label(s), "
                    "for transparency -- lets a reviewer see what the system read regardless "
                    "of which fields passed or failed.",
    )
    label_images: list[LabelImage] = Field(
        default_factory=list,
        description="The label image(s) actually processed (e.g. front/back/neck from a COLA "
                    "PDF), returned so the frontend can display them without needing to persist "
                    "anything server-side. Empty for /api/verify, which already has the "
                    "uploaded photo client-side.",
    )

    @classmethod
    def from_report(cls, report: VerificationReport, processing_time_ms: float) -> "VerificationResponse":
        return cls(
            overall_pass=report.overall_pass,
            fields=[FieldVerificationResult.from_field_result(f) for f in report.fields],
            processing_time_ms=processing_time_ms,
        )


class ErrorResponse(BaseModel):
    """Shape for error responses (bad image, OCR failure, etc.) — used by the FastAPI layer next."""
    error: str
    detail: str | None = None
