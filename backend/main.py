"""
main.py

FastAPI backend for the label verifier. Three endpoints:

    POST /api/verify              - one label image + submitted field values
    POST /api/verify/batch        - multiple labels in one request (Janet's
                                     bulk-import use case from the stakeholder
                                     notes: importers dumping 200-300 label
                                     applications at once)
    POST /api/verify/from-cola-pdf - upload a COLA PDF directly; fields and
                                     the label image are both auto-extracted
                                     from it, and a real physical-units size
                                     check runs if the PDF states the label's
                                     actual dimensions

Run locally:
    uvicorn main:app --reload

Then see interactive docs at http://127.0.0.1:8000/docs (FastAPI
auto-generates these from the Pydantic models in models.py).

Design notes:
- Fields are accepted as individual Form(...) parameters rather than a
  single JSON body, because this endpoint needs to accept a file upload
  (multipart/form-data) in the same request — you cannot mix a JSON body
  and a file upload in one standard HTTP request. This is exactly the
  shape a plain HTML <form> with an <input type="file"> produces, which
  matters given Sarah Chen's "my mother could figure it out" bar for the
  eventual frontend.
- Every image is written to a temp file and cleaned up in a `finally`
  block. ocr_engine.extract_text() takes a path (it's also usable as a
  standalone CLI tool that way), so this keeps that interface unchanged
  rather than plumbing bytes-vs-path through every layer.
- No image is retained after the request completes. Combined with
  ocr_engine.py making zero outbound network calls per request, nothing
  here depends on or exposes data to an external service — relevant given
  the stakeholder notes' PII/document-retention concerns and the firewall
  restrictions from the earlier vendor pilot.
- Batch endpoint processes each label independently and never lets one
  bad image take down the whole batch — each item's success or failure
  is reported individually, matching Dave's "don't make my life harder"
  complaint: a single malformed upload in a 200-label batch shouldn't
  force a re-submission of the other 199.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from field_matcher import verify_label, select_warning_lines
from models import ErrorResponse, FieldVerificationResult, LabelImage, LabelSubmission, VerificationResponse
from ocr_engine import extract_text, extract_text_concurrent
from cola_pdf_parser import parse_cola_pdf
from cola_label_extractor import extract_all_label_images
from label_size_requirements import ScaleReference, check_warning_absolute_size, parse_net_contents_ml


def _file_to_data_uri(path: Path, mime: Optional[str] = None) -> str:
    mime = mime or mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _bytes_to_data_uri(data: bytes, ext: str) -> str:
    mime = mimetypes.guess_type(f"x.{ext}")[0] or "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"

app = FastAPI(
    title="TTB Label Verifier (Prototype)",
    description="AI-powered alcohol label verification — proof of concept.",
    version="0.1.0",
)

# Prototype-only: wide open CORS so a locally-served frontend (any port)
# can call this without extra config. Would be restricted to a known
# origin before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Basic liveness check. Does not touch the OCR engine (stays fast)."""
    return {"status": "ok"}


def _save_upload_to_temp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return Path(tmp.name)


def _run_verification(image_path: Path, submission: LabelSubmission) -> VerificationResponse:
    return _run_verification_from_fields(image_path, submission.as_field_dict())


def _run_verification_from_fields(image_path: Path, expected_fields: dict) -> VerificationResponse:
    """
    Like _run_verification, but takes a plain fields dict rather than a
    validated LabelSubmission. Used by the COLA-PDF-intake endpoint,
    where alcohol_content/net_contents may be legitimately absent (not
    every TTB form revision declares them -- see cola_pdf_parser.py) and
    LabelSubmission's Pydantic validation would incorrectly reject that
    as an incomplete submission rather than treating it as "nothing to
    check for this field."
    """
    import time
    t0 = time.perf_counter()
    lines = extract_text(image_path)
    report = verify_label(expected_fields, lines)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response = VerificationResponse.from_report(report, elapsed_ms)
    response.detected_text = [l.text for l in lines]
    return response


@app.post("/api/verify", response_model=VerificationResponse, responses={400: {"model": ErrorResponse}})
async def verify_single(
    image: UploadFile = File(..., description="Label image (jpg/png)"),
    brand_name: str = Form(...),
    class_type: str = Form(...),
    alcohol_content: str = Form(...),
    net_contents: str = Form(...),
    bottler: str = Form(...),
    country_of_origin: Optional[str] = Form(None),
):
    try:
        submission = LabelSubmission(
            brand_name=brand_name,
            class_type=class_type,
            alcohol_content=alcohol_content,
            net_contents=net_contents,
            bottler=bottler,
            country_of_origin=country_of_origin,
        )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid submission fields: {e}")

    tmp_path = None
    try:
        tmp_path = _save_upload_to_temp(image)
        response = _run_verification(tmp_path, submission)
        response.label_images = [LabelImage(
            image_type="Uploaded photo",
            data_uri=_file_to_data_uri(tmp_path),
        )]
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


@app.post("/api/verify/batch")
async def verify_batch(
    images: list[UploadFile] = File(..., description="Label images, same order as `submissions`"),
    submissions: str = Form(..., description="JSON array of label field objects, same order as `images`"),
):
    try:
        submissions_data = json.loads(submissions)
        if not isinstance(submissions_data, list):
            raise ValueError("`submissions` must be a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid `submissions` JSON: {e}")

    if len(submissions_data) != len(images):
        raise HTTPException(
            status_code=400,
            detail=f"Got {len(images)} images but {len(submissions_data)} submissions — "
                   f"these must be the same length and in matching order.",
        )

    results = []
    for i, (image, sub_data) in enumerate(zip(images, submissions_data)):
        tmp_path = None
        try:
            submission = LabelSubmission(**sub_data)
            tmp_path = _save_upload_to_temp(image)
            response = _run_verification(tmp_path, submission)
            results.append({
                "index": i,
                "filename": image.filename,
                "status": "ok",
                "result": response.model_dump(),
            })
        except ValidationError as e:
            results.append({
                "index": i, "filename": image.filename,
                "status": "error", "error": f"Invalid submission fields: {e}",
            })
        except Exception as e:
            results.append({
                "index": i, "filename": image.filename,
                "status": "error", "error": f"Could not process image: {e}",
            })
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    succeeded = sum(1 for r in results if r["status"] == "ok")
    return JSONResponse({
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    })


class ColaPdfError(Exception):
    """Raised by _process_cola_pdf for any known, diagnosable failure mode
    (error page, incomplete parse, no label found). Kept separate from
    HTTPException so the same processing logic can be reused by both the
    single-PDF endpoint (raises HTTPException) and the batch endpoint
    (catches this per-item without killing the whole batch)."""
    pass


def _process_cola_pdf(tmp_pdf_path: Path, tmp_img_paths: list[Path]) -> VerificationResponse:
    """
    Core COLA-PDF processing logic, shared by the single and batch
    endpoints. Appends any temp image files it creates to tmp_img_paths
    (caller is responsible for cleanup) so both endpoints clean up the
    same way. Raises ColaPdfError with a specific, human-readable message
    on any known failure mode.
    """
    pdf_bytes = tmp_pdf_path.read_bytes()
    parsed = parse_cola_pdf(tmp_pdf_path)

    # Check for a TTB system error page FIRST -- an invalid/unsupported ID
    # also fails to parse fields, but "this isn't a real record" is the
    # correct diagnosis, not "we couldn't parse the fields on this record".
    extracted_images, status = extract_all_label_images(pdf_bytes, parsed.image_dimensions)
    if status == "error_page":
        raise ColaPdfError(
            "This PDF is a TTB system error page, not a real COLA record "
            "(often means an invalid or unsupported/very old TTB ID)."
        )

    if not parsed.is_complete():
        missing = [k for k in ("brand_name", "class_type", "bottler") if not getattr(parsed, k)]
        raise ColaPdfError(
            f"Could not parse all required fields from this PDF (missing: "
            f"{', '.join(missing)}). This parser is tested against standard "
            f"electronically-filed COLA forms; older paper-filed scans or "
            f"unusual layouts may not parse cleanly."
        )

    if status == "no_label_found":
        raise ColaPdfError(
            "No plausible label image found in this PDF. This can be a "
            "genuine label-exempt application (no image expected), or a "
            "layout our extraction heuristic doesn't handle well -- check "
            "the source PDF manually."
        )

    # OCR each real label image in PARALLEL (front/back/neck), since each
    # is a physically distinct image and OCR is the dominant per-image
    # cost. Uses extract_text_concurrent() specifically -- NOT a naive
    # ThreadPoolExecutor over extract_text() calls, which was tested and
    # found to corrupt results by sharing model instances across threads
    # (see extract_text_concurrent()'s docstring in ocr_engine.py for the
    # measured before/after). Then merge the detected text for general
    # field matching, where a field could legitimately appear on either
    # image.
    import time
    t0 = time.perf_counter()

    img_paths = []
    for extracted in extracted_images:
        suffix = f".{extracted.ext}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(extracted.image_bytes)
            img_paths.append(Path(tmp.name))
    tmp_img_paths.extend(img_paths)

    all_results = extract_text_concurrent(img_paths)
    per_image_lines = list(zip(extracted_images, all_results))
    all_lines = [line for lines in all_results for line in lines]

    submission_fields = parsed.as_field_dict()
    report = verify_label(submission_fields, all_lines)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response = VerificationResponse.from_report(report, elapsed_ms)
    response.detected_text = [l.text for l in all_lines]
    response.label_images = [
        LabelImage(
            image_type=extracted.image_type,
            data_uri=_bytes_to_data_uri(extracted.image_bytes, extracted.ext),
        )
        for extracted, _ in per_image_lines
    ]

    # Bonus check, only possible because the PDF gave us a real physical
    # scale reference: certified absolute size compliance, not just the
    # same-photo relative heuristic /api/verify is limited to. Uses
    # whichever image the warning text was actually found on -- front
    # and back images can have different physical dimensions, so the
    # scale must come from the SAME image the measurement is taken on.
    if parsed.net_contents:
        net_ml = parse_net_contents_ml(parsed.net_contents)
        if net_ml is not None:
            for extracted, lines in per_image_lines:
                warning_lines = select_warning_lines(lines)
                if not warning_lines:
                    continue
                scale = ScaleReference.from_known_dimension(
                    pixel_length=extracted.height_px, physical_inches=extracted.height_inches,
                )
                size_result = check_warning_absolute_size(warning_lines, scale, net_ml)
                score = min(size_result.measured_mm / size_result.required_mm, 1.0) * 100 \
                    if size_result.required_mm > 0 else 0.0
                response.fields.append(FieldVerificationResult(
                    field_name="government_warning_absolute_size",
                    expected_value=f">= {size_result.required_mm}mm (27 CFR 16.22(a))",
                    matched=size_result.matched,
                    score=score,
                    method="strict",
                    detail=f"[{extracted.image_type}] {size_result.detail}",
                ))
                response.overall_pass = response.overall_pass and size_result.matched
                break  # only check the first image where the warning was actually found

    return response


@app.post("/api/verify/from-cola-pdf", response_model=VerificationResponse,
          responses={400: {"model": ErrorResponse}})
async def verify_from_cola_pdf(
    pdf: UploadFile = File(..., description="A COLA detail PDF (printed from TTB's Public COLA Registry)"),
):
    """
    Auto-intake mode: upload a COLA PDF directly (the same kind of file
    produced by printing a COLA detail page from TTB's Public COLA
    Registry) instead of manually typing in the five application fields.

    This endpoint:
      1. Parses the declared field values from the PDF's own text (brand,
         class/type, ABV, net contents, bottler) -- see cola_pdf_parser.py.
         Deliberately does NOT extract or retain phone/fax/email/personal
         name/signature.
      2. Extracts the actual label image embedded in the PDF -- see
         cola_label_extractor.py.
      3. Runs OCR on each image separately, merges the results, and runs
         the same matching pipeline as /api/verify against the combined
         text (a field can legitimately be printed on either image).
      4. If the PDF states physical dimensions for the image the warning
         was actually found on, ALSO runs a real, physical-units size
         check against 27 CFR 16.22(a) -- see label_size_requirements.py.
         Uses THAT image's own declared dimensions for the scale
         reference, not an arbitrary one, since different label images on
         the same product can have different physical sizes.

    Nothing from the PDF is retained after the request completes. Skips
    the bonus absolute-size check gracefully (not an error) if this form
    revision doesn't declare net contents on the application at all.
    """
    tmp_pdf_path = None
    tmp_img_paths: list[Path] = []
    try:
        tmp_pdf_path = _save_upload_to_temp(pdf)
        return _process_cola_pdf(tmp_pdf_path, tmp_img_paths)
    except ColaPdfError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process this PDF: {e}")
    finally:
        for p in [tmp_pdf_path, *tmp_img_paths]:
            if p and p.exists():
                p.unlink()


@app.post("/api/verify/batch-from-cola-pdf")
async def verify_batch_from_cola_pdf(
    pdfs: list[UploadFile] = File(..., description="Multiple COLA detail PDFs"),
):
    """
    Bulk version of /api/verify/from-cola-pdf -- Janet's "big importers
    dump 200-300 label applications at once" scenario from the
    stakeholder notes, for the COLA-PDF intake path specifically (the
    realistic bulk-intake shape: applications arrive as PDFs from the
    registry, not as loose photos).

    Each PDF is processed independently, exactly like the single-PDF
    endpoint -- one bad/unparseable PDF in a large batch doesn't take
    down the rest, same principle as /api/verify/batch.
    """
    results = []
    for i, pdf in enumerate(pdfs):
        tmp_pdf_path = None
        tmp_img_paths: list[Path] = []
        try:
            tmp_pdf_path = _save_upload_to_temp(pdf)
            response = _process_cola_pdf(tmp_pdf_path, tmp_img_paths)
            results.append({
                "index": i,
                "filename": pdf.filename,
                "status": "ok",
                "result": response.model_dump(),
            })
        except ColaPdfError as e:
            results.append({"index": i, "filename": pdf.filename, "status": "error", "error": str(e)})
        except Exception as e:
            results.append({
                "index": i, "filename": pdf.filename,
                "status": "error", "error": f"Could not process this PDF: {e}",
            })
        finally:
            for p in [tmp_pdf_path, *tmp_img_paths]:
                if p and p.exists():
                    p.unlink()

    succeeded = sum(1 for r in results if r["status"] == "ok")
    return JSONResponse({
        "total": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "results": results,
    })


# Serve the frontend from the SAME origin as the API. Mounted last, after
# every API route above, so explicit routes (/health, /api/verify, etc.)
# take precedence and this only catches everything else -- including "/"
# itself, via html=True, which serves frontend/index.html.
#
# This also means CORS is no longer needed in a real deployment (frontend
# and API share an origin) -- the wide-open CORSMiddleware above is left
# in place for local dev convenience (running the frontend on a separate
# port via `python -m http.server`, as documented earlier in this
# project), but a production deployment behind this mount doesn't
# actually need it for the shipped frontend to work.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
