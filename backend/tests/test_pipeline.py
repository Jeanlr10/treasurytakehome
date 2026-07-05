"""
tests/test_pipeline.py

Reproducible version of the ad-hoc checks run during development. Not a
pytest suite (kept dependency-free on purpose, at this stage) — run
directly with:

    python tests/test_pipeline.py

from inside backend/. Exits non-zero on any failure.

Covers, in order:
  1. Government warning: full correct statutory text -> must PASS
  2. Government warning: Jenny's real example (title-case heading) -> must FAIL
  3. Government warning: missing part (2) of the statutory text -> must FAIL
  4. Field matching: Dave's real example (case-only brand mismatch) -> must MATCH
  5. Field matching: genuinely different brand -> must NOT MATCH
  6. End-to-end: real OCR + matching against a real generated label image,
     for a compliant label -> overall PASS
  7. End-to-end: real OCR + matching against a label with the title-case
     warning defect -> overall FAIL, specifically on government_warning
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from field_matcher import (
    match_field_fuzzy, match_government_warning, verify_label, check_warning_text_size,
)
from label_size_requirements import (
    parse_net_contents_ml, required_warning_height_mm, ScaleReference,
    check_warning_absolute_size,
)
from cola_pdf_parser import parse_cola_pdf
from cola_label_extractor import extract_all_label_images
from ocr_engine import OCRLine, extract_text, extract_text_concurrent
from models import LabelSubmission, VerificationResponse
from pydantic import ValidationError
import json as _json
from fastapi.testclient import TestClient
from main import app as _app

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "sample_labels"

PASSED = 0
FAILED = 0
TIMINGS: list[tuple[str, str, float]] = []  # (group, check_name, elapsed_seconds)


def check(group: str, name: str, condition: bool, detail: str = "", elapsed: float | None = None):
    global PASSED, FAILED
    timing_str = f"  ({elapsed*1000:.1f} ms)" if elapsed is not None else ""
    if elapsed is not None:
        TIMINGS.append((group, name, elapsed))
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}{timing_str}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}{timing_str}  {detail}")


class timer:
    """Context manager: `with timer() as t: ...` then t.elapsed is seconds."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def mk(text: str) -> OCRLine:
    return OCRLine(text=text, confidence=0.99, box=[[0, 0], [1, 0], [1, 1], [0, 1]])


def test_government_warning_unit_cases():
    print("\n-- government warning: unit cases --")
    group = "matching_government_warning"

    full_warning_lines = [
        mk("GOVERNMENT WARNING:"),
        mk("(1) According to the Surgeon General, women should not"),
        mk("drink alcoholic beverages during pregnancy because of the risk of birth"),
        mk("defects."),
        mk("(2) Consumption of alcoholic beverages impairs your ability to"),
        mk("drive a car or operate machinery, and may cause health problems."),
    ]

    with timer() as t:
        r = match_government_warning(full_warning_lines)
    check(group, "full correct warning -> PASS", r.matched, r.detail, t.elapsed)

    titlecase_lines = [mk("Government Warning:")] + full_warning_lines[1:]
    with timer() as t:
        r = match_government_warning(titlecase_lines)
    check(group, "title-case heading (Jenny's example) -> FAIL", not r.matched, r.detail, t.elapsed)

    partial_lines = full_warning_lines[:4]
    with timer() as t:
        r = match_government_warning(partial_lines)
    check(group, "missing part (2) -> FAIL", not r.matched, r.detail, t.elapsed)


def test_field_matching_unit_cases():
    print("\n-- field matching: unit cases --")
    group = "matching_general_fields"

    label_lines = [mk("STONE'S THROW"), mk("Some Other Label Text Here")]

    with timer() as t:
        r = match_field_fuzzy("brand_name", "Stone's Throw", label_lines)
    check(group, "case-only mismatch (Dave's example) -> MATCH", r.matched, r.detail, t.elapsed)

    with timer() as t:
        r = match_field_fuzzy("brand_name", "Completely Different Brand", label_lines)
    check(group, "genuinely different brand -> NO MATCH", not r.matched, r.detail, t.elapsed)


def test_end_to_end_real_images():
    print("\n-- end-to-end: real OCR against generated label images --")

    expected = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "bottler": "Old Tom Distillery Co., Bardstown, KY",
    }

    pass_img = SAMPLE_DIR / "old_tom_bourbon_pass.png"
    with timer() as t_ocr:
        lines = extract_text(pass_img)
    with timer() as t_match:
        report = verify_label(expected, lines)
    check("ocr_extraction (cold, incl. model load)", f"OCR extraction ({pass_img.name})", True, elapsed=t_ocr.elapsed)
    check("full_verify_label", f"compliant label ({pass_img.name}) -> overall PASS",
          report.overall_pass, elapsed=t_match.elapsed)

    fail_img = SAMPLE_DIR / "old_tom_bourbon_titlecase_fail.png"
    with timer() as t_ocr:
        lines = extract_text(fail_img)
    with timer() as t_match:
        report = verify_label(expected, lines)
    warning_result = next(f for f in report.fields if f.field_name == "government_warning")
    check("ocr_extraction (warm)", f"OCR extraction ({fail_img.name})", True, elapsed=t_ocr.elapsed)
    check(
        "full_verify_label",
        f"title-case defect label ({fail_img.name}) -> overall FAIL",
        not report.overall_pass,
        elapsed=t_match.elapsed,
    )
    check(
        "matching_government_warning",
        "  ...specifically flagged on government_warning field, not a false field mismatch",
        not warning_result.matched,
        warning_result.detail,
    )


def test_warning_text_size():
    print("\n-- warning text size heuristic (Jenny's 'tiny text' complaint) --")
    group = "warning_text_size"

    expected = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "bottler": "Old Tom Distillery Co., Bardstown, KY",
    }

    # Legitimately compliant label: warning is naturally somewhat smaller
    # than body text (normal label typography) but should NOT be flagged.
    pass_img = SAMPLE_DIR / "old_tom_bourbon_pass.png"
    with timer() as t:
        lines = extract_text(pass_img)
        result = check_warning_text_size(lines)
    check(group, f"{pass_img.name}: normal proportions -> not flagged", result.matched,
          result.detail, elapsed=t.elapsed)

    # Deliberately tiny warning text: real violation pattern Jenny described.
    tiny_img = SAMPLE_DIR / "warning_tiny_text_fail.png"
    with timer() as t:
        lines = extract_text(tiny_img)
        result = check_warning_text_size(lines)
        full_report = verify_label(expected, lines)
    check(group, f"{tiny_img.name}: tiny warning text -> flagged", not result.matched,
          result.detail, elapsed=t.elapsed)
    check(group, f"{tiny_img.name}: text CONTENT still correct despite tiny size "
                 "(proves size check catches something content check alone would miss)",
          next(f for f in full_report.fields if f.field_name == "government_warning").matched)
    check(group, f"{tiny_img.name}: overall_pass correctly False due to size violation alone",
          not full_report.overall_pass)


def test_absolute_size_requirements():
    """
    label_size_requirements.py -- real physical minimum type-size checks
    per 27 CFR 16.22(a), verified directly against eCFR (not a secondary
    summary). NOT wired into the main verify_label() pipeline yet -- that
    requires a physical scale reference (e.g. a COLA PDF's stated "Actual
    Dimensions"), which cola_pdf_parser.py will supply in a later stage.
    This tests the module standalone, including one real end-to-end run
    using the actual Cascade Winery record's real stated dimensions.
    """
    print("\n-- absolute size requirements (27 CFR 16.22(a)) --")
    group = "absolute_size_requirements"

    with timer() as t:
        ok = (
            parse_net_contents_ml("750 mL") == 750.0
            and parse_net_contents_ml("750ML") == 750.0
            and parse_net_contents_ml("750 MILLILITERS") == 750.0
            and parse_net_contents_ml("1 L") == 1000.0
            and parse_net_contents_ml("garbage") is None
        )
    check(group, "parse_net_contents_ml handles common label/application formats", ok, elapsed=t.elapsed)

    with timer() as t:
        ok = (
            required_warning_height_mm(237) == 1.0
            and required_warning_height_mm(238) == 2.0
            and required_warning_height_mm(3000) == 2.0
            and required_warning_height_mm(3001) == 3.0
        )
    check(group, "required_warning_height_mm matches 27 CFR 16.22(a) tier boundaries exactly",
          ok, elapsed=t.elapsed)

    # Real end-to-end: actual PDF-stated physical dimensions + actual label image
    from field_matcher import GOVERNMENT_WARNING_TEXT, _normalize

    img = SAMPLE_DIR / "real_ttb_03211001000018_cascade_winery.png"
    with timer() as t:
        lines = extract_text(img)
        warning_vocab = set(_normalize(GOVERNMENT_WARNING_TEXT).split())
        warning_lines = [
            l for l in lines
            if (words := set(_normalize(l.text).split()))
            and len(words & warning_vocab) / len(words) >= 0.4
        ]
        # Real "Actual Dimensions: 3.5 inches W X 4 inches H" from the source PDF
        scale = ScaleReference.from_known_dimension(pixel_length=477, physical_inches=4.0)
        result = check_warning_absolute_size(warning_lines, scale, net_contents_ml=750.0)
    check(group, "real label: warning text physically meets 27 CFR 16.22(a) minimum for 750mL "
                 f"({result.measured_mm:.2f}mm measured >= {result.required_mm}mm required)",
          result.matched, result.detail, elapsed=t.elapsed)


def test_real_world_cola():
    """
    Uses a real, publicly-approved TTB COLA (Cascade Winery, TTB ID
    03211001000018, sourced via TTB's own Public COLA Registry -- see
    project history). Declared field values below are hardcoded from that
    real application's text (not synthetic), matched against the real
    label image extracted from the same PDF.

    This is what actually drove three real matcher calibration fixes:
    unit normalization (net_contents), a lower bottler threshold (labels
    legally omit the street address), and a lower class_type threshold
    (TTB's assigned classification uses different wording than the
    label). All three are exercised here so a future change can't
    silently regress them.
    """
    print("\n-- real-world validation: actual approved TTB COLA --")
    group = "real_world_cola"

    # Declared on the real application (TTB ID 03211001000018), business
    # fields only -- no personal contact info was ever extracted or kept.
    declared = {
        "brand_name": "CASCADE WINERY",
        "class_type": "TABLE RED WINE",
        "alcohol_content": "11.5",
        "net_contents": "750 MILLILITERS",
        "bottler": "CASCADE WINERY, CASCADE WINERY, INC., 6275 28TH ST, GRAND RAPIDS MI 49546",
    }

    img = SAMPLE_DIR / "real_ttb_03211001000018_cascade_winery.png"
    with timer() as t:
        lines = extract_text(img)
        report = verify_label(declared, lines)
    check(group, "real COLA: overall_pass (all 3 calibration fixes hold together)",
          report.overall_pass, elapsed=t.elapsed)

    scores = {r.field_name: r for r in report.fields}
    check(group, "  net_contents matches despite unit spelling difference (MILLILITERS vs mL)",
          scores["net_contents"].matched, scores["net_contents"].detail)
    check(group, "  bottler matches despite application having extra street address",
          scores["bottler"].matched, scores["bottler"].detail)
    check(group, "  class_type matches despite regulatory vs. consumer wording",
          scores["class_type"].matched, scores["class_type"].detail)
    check(group, "  government_warning matches (rotated-panel ordering fix holds on real data)",
          scores["government_warning"].matched, scores["government_warning"].detail)


def test_rotation_handling():
    """
    Documents CURRENT actual behavior on rotated/tilted labels, including a
    known, unresolved regression (see README "Known limitations"). This is
    intentional: two regressions were introduced and caught during
    development of this feature by testing after each change. Locking in
    current behavior here means any future change that makes things WORSE
    shows up as a test failure, even though the underlying limitation
    itself is not yet fixed.
    """
    print("\n-- rotation handling: current behavior (includes a known open issue) --")
    group = "rotation_handling"

    expected = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "bottler": "Old Tom Distillery Co., Bardstown, KY",
    }

    def general_fields_pass(report) -> bool:
        warning_related = {"government_warning", "government_warning_size"}
        return all(r.matched for r in report.fields if r.field_name not in warning_related)

    # Mild tilt (~8-20 degrees): a realistic hand-held-photo angle.
    # Should fully pass, including the warning field.
    for fname in ["rotation_tilt_8deg.png", "rotation_tilt_20deg.png"]:
        img = SAMPLE_DIR / fname
        with timer() as t:
            lines = extract_text(img)
            report = verify_label(expected, lines)
        check(group, f"{fname}: general fields pass", general_fields_pass(report), elapsed=t.elapsed)
        check(group, f"{fname}: warning field also passes", report.overall_pass)

    # Mixed orientation (upright body + 90-degree rotated warning panel):
    # general fields pass; warning field is a KNOWN, documented failure
    # (not a bug in this test -- this asserts the known-bad current state).
    img = SAMPLE_DIR / "rotation_mixed_panel.png"
    with timer() as t:
        lines = extract_text(img)
        report = verify_label(expected, lines)
    check(group, "rotation_mixed_panel.png: general fields still pass despite rotated panel",
          general_fields_pass(report), elapsed=t.elapsed)
    gw = next(r for r in report.fields if r.field_name == "government_warning")
    check(group, "rotation_mixed_panel.png: warning field KNOWN LIMITATION (currently fails, see README)",
          not gw.matched, gw.detail)

    # Full 90-degree whole-image rotation: KNOWN, documented regression.
    # general fields currently FAIL here -- this asserts that known-bad
    # state so a future fix is visible (test starts failing in the GOOD
    # direction) and a future further regression is also visible.
    img = SAMPLE_DIR / "rotation_full_90deg.png"
    with timer() as t:
        lines = extract_text(img)
        report = verify_label(expected, lines)
    check(group, "rotation_full_90deg.png: KNOWN LIMITATION (general fields currently fail, see README)",
          not general_fields_pass(report), elapsed=t.elapsed)


def test_api_endpoints():
    print("\n-- main.py: FastAPI endpoint tests --")
    group = "api_endpoints"
    client = TestClient(_app)

    sub = {
        "brand_name": "Old Tom Distillery",
        "class_type": "Kentucky Straight Bourbon Whiskey",
        "alcohol_content": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "bottler": "Old Tom Distillery Co., Bardstown, KY",
    }

    with timer() as t:
        r = client.get("/health")
    check(group, "GET /health -> 200", r.status_code == 200, elapsed=t.elapsed)

    pass_img = SAMPLE_DIR / "old_tom_bourbon_pass.png"
    with timer() as t:
        with open(pass_img, "rb") as f:
            r = client.post("/api/verify", files={"image": ("label.png", f, "image/png")}, data=sub)
    check(group, "POST /api/verify: compliant label -> 200 + overall_pass=True",
          r.status_code == 200 and r.json()["overall_pass"] is True, elapsed=t.elapsed)

    fail_img = SAMPLE_DIR / "old_tom_bourbon_titlecase_fail.png"
    with timer() as t:
        with open(fail_img, "rb") as f:
            r = client.post("/api/verify", files={"image": ("label.png", f, "image/png")}, data=sub)
    check(group, "POST /api/verify: defect label -> 200 + overall_pass=False",
          r.status_code == 200 and r.json()["overall_pass"] is False, elapsed=t.elapsed)

    with timer() as t:
        with open(pass_img, "rb") as f:
            r = client.post("/api/verify", files={"image": ("label.png", f, "image/png")},
                             data={"brand_name": "Old Tom Distillery"})  # missing required fields
    check(group, "POST /api/verify: missing required fields -> 4xx, not a crash",
          400 <= r.status_code < 500, elapsed=t.elapsed)

    with timer() as t:
        r = client.post("/api/verify",
                         files={"image": ("bad.png", b"not an image", "image/png")}, data=sub)
    check(group, "POST /api/verify: corrupt image -> 400, not a 500 crash",
          r.status_code == 400, elapsed=t.elapsed)

    with timer() as t:
        with open(pass_img, "rb") as f1, open(fail_img, "rb") as f2:
            r = client.post(
                "/api/verify/batch",
                files=[("images", ("a.png", f1, "image/png")), ("images", ("b.png", f2, "image/png"))],
                data={"submissions": _json.dumps([sub, sub])},
            )
    body = r.json() if r.status_code == 200 else {}
    check(group, "POST /api/verify/batch: 2 images, mixed pass/fail -> both processed correctly",
          r.status_code == 200 and body.get("succeeded") == 2
          and body["results"][0]["result"]["overall_pass"] is True
          and body["results"][1]["result"]["overall_pass"] is False,
          elapsed=t.elapsed)

    with timer() as t:
        with open(pass_img, "rb") as f1:
            r = client.post(
                "/api/verify/batch",
                files=[("images", ("a.png", f1, "image/png"))],
                data={"submissions": _json.dumps([sub, sub])},  # mismatched count on purpose
            )
    check(group, "POST /api/verify/batch: mismatched images/submissions count -> 400",
          r.status_code == 400, elapsed=t.elapsed)

    with timer() as t:
        with open(pass_img, "rb") as f1:
            r = client.post(
                "/api/verify/batch",
                files=[("images", ("good.png", f1, "image/png")),
                       ("images", ("bad.png", b"not an image", "image/png"))],
                data={"submissions": _json.dumps([sub, sub])},
            )
    body = r.json() if r.status_code == 200 else {}
    check(group, "POST /api/verify/batch: one corrupt image doesn't fail the whole batch",
          r.status_code == 200 and body.get("succeeded") == 1 and body.get("failed") == 1,
          elapsed=t.elapsed)


def test_concurrent_ocr_correctness():
    """
    extract_text_concurrent() -- parallel OCR across a multi-image COLA
    PDF's images (front/back/neck), used by the /api/verify/from-cola-pdf
    endpoint for multi-image records.

    This exists specifically because a naive ThreadPoolExecutor over
    extract_text() (which uses shared, cached model singletons) was
    tested and found to be a REAL correctness bug, not a theoretical
    one: concurrent inference calls on the same shared PaddleOCR
    instances produced cross-contaminated results across images (e.g.
    the front label's OCR output coming back containing text that
    actually belonged to the back label). extract_text_concurrent() gives
    each concurrent call its own independent model instances instead --
    this test locks in that the fix actually produces byte-identical
    text output to the sequential path, so a future change can't
    silently reintroduce the corruption.
    """
    print("\n-- concurrent OCR correctness (extract_text_concurrent) --")
    group = "concurrent_ocr_correctness"

    fixtures = Path(__file__).resolve().parent / "fixtures"
    throwback_pdf = fixtures / "real_cola_throwback_3images.pdf"

    parsed = parse_cola_pdf(throwback_pdf)
    pdf_bytes = throwback_pdf.read_bytes()
    extracted_images, status = extract_all_label_images(pdf_bytes, parsed.image_dimensions)

    import tempfile
    img_paths = []
    for img in extracted_images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img.ext}") as tmp:
            tmp.write(img.image_bytes)
            img_paths.append(Path(tmp.name))

    with timer() as t:
        sequential_results = [extract_text(p) for p in img_paths]
    sequential_texts = [sorted(l.text for l in r) for r in sequential_results]

    with timer() as t2:
        concurrent_results = extract_text_concurrent(img_paths)
    concurrent_texts = [sorted(l.text for l in r) for r in concurrent_results]

    check(group, "concurrent output text-identical to sequential, across all 3 images",
          concurrent_texts == sequential_texts, elapsed=t2.elapsed)
    # Informational only, not a pass/fail gate -- speed comparisons are
    # inherently noisy (shared CI/sandbox CPU contention, warm-cache
    # ordering effects from whatever ran before this test in the same
    # suite), and the thing that actually matters for safety is the
    # correctness check above. Manually verified the real speedup this
    # is meant to provide (3x+) in an isolated, controlled comparison --
    # see extract_text_concurrent()'s docstring in ocr_engine.py.
    print(f"  (informational) sequential={t.elapsed*1000:.0f}ms, concurrent={t2.elapsed*1000:.0f}ms")

    for p in img_paths:
        p.unlink()


def test_concurrent_ocr_memory_stable():
    """
    Regression test for a real memory leak found and fixed in this
    project: an earlier version of extract_text_concurrent() constructed
    a brand-new TextDetection/TextRecognition pair on every call, which
    leaked memory (something in PaddleX's underlying C++ backend doesn't
    fully release resources when a Python-level instance is garbage
    collected). This actually crashed the deployed production container
    -- after roughly 7-8 real COLA PDFs, the service started refusing
    connections, consistent with an OOM kill and Podman's Restart=always
    bringing it back, which is why the failures recurred rather than
    being permanent.

    The fix was a small, fixed-size model instance pool (_get_model_pool()
    in ocr_engine.py) -- this test locks in that the fix actually holds:
    RSS should plateau after the pool's one-time construction cost, not
    keep climbing per call. Threshold set at 120MB drift across the back
    half of the run, calibrated against REAL observed noise, not guessed:
    repeated manual runs measured drift of 14MB, 36MB, and 70MB across
    otherwise-identical conditions, purely from allocator/GC timing
    variance -- an initial 50MB threshold produced a false failure from
    this normal noise. 120MB keeps a wide, safe margin below the actual
    bug's severity (roughly 20MB of growth PER SINGLE PDF, continuously,
    with no sign of stopping -- ~180MB+ over this many iterations) while
    comfortably absorbing the observed noise range.
    """
    print("\n-- concurrent OCR memory stability (regression for a real production crash) --")
    group = "concurrent_ocr_memory_stable"

    import gc
    import os
    import tempfile

    import psutil

    fixtures = Path(__file__).resolve().parent / "fixtures"
    pdf_path = fixtures / "real_cola_throwback_3images.pdf"
    parsed = parse_cola_pdf(pdf_path)
    pdf_bytes = pdf_path.read_bytes()
    extracted_images, _ = extract_all_label_images(pdf_bytes, parsed.image_dimensions)

    img_paths = []
    for img in extracted_images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img.ext}") as tmp:
            tmp.write(img.image_bytes)
            img_paths.append(Path(tmp.name))

    process = psutil.Process(os.getpid())
    N = 10  # fewer than the 25 used during manual investigation, to keep the suite's runtime reasonable
    readings = []

    with timer() as t:
        for _ in range(N):
            results = extract_text_concurrent(img_paths)
            del results
            gc.collect()
            readings.append(process.memory_info().rss / 1024 / 1024)

    first_half_avg = sum(readings[:N // 2]) / (N // 2)
    second_half_avg = sum(readings[N // 2:]) / (N - N // 2)
    drift = second_half_avg - first_half_avg

    check(group, f"RSS plateaus rather than climbing per call "
                 f"(first-half avg={first_half_avg:.0f}MB, second-half avg={second_half_avg:.0f}MB, "
                 f"drift={drift:.0f}MB, threshold=120MB)",
          drift < 120, elapsed=t.elapsed)

    for p in img_paths:
        p.unlink()


def test_cola_pdf_intake():
    """
    cola_pdf_parser.py + cola_label_extractor.py + the new
    /api/verify/from-cola-pdf endpoint. Covers TWO real form revisions
    (OMB 1512-0092 and the newer 1513-0020, which has different field
    numbering and doesn't declare net contents/ABV on Part I at all) and
    multi-image records (front + back, and front + neck + back), plus a
    genuine TTB system error page.
    """
    print("\n-- COLA PDF intake: parser, extractor, and endpoint --")
    group = "cola_pdf_intake"

    fixtures = Path(__file__).resolve().parent / "fixtures"
    real_pdf = fixtures / "real_cola_cascade_winery.pdf"
    error_pdf = fixtures / "real_ttb_error_page.pdf"
    throwback_pdf = fixtures / "real_cola_throwback_3images.pdf"       # newer form rev, 3 images
    chaglasian_pdf = fixtures / "real_cola_chaglasian_2images.pdf"     # newer form rev, 2 images

    with timer() as t:
        parsed = parse_cola_pdf(real_pdf)
    check(group, "parse_cola_pdf: all fields parsed from a real COLA (older form rev)",
          parsed.is_complete(), elapsed=t.elapsed)
    check(group, "  Actual Dimensions parsed for absolute size check",
          parsed.has_scale_reference())

    with timer() as t:
        throwback_parsed = parse_cola_pdf(throwback_pdf)
    check(group, "parse_cola_pdf: newer form revision (no Part I net contents/ABV) still parses",
          throwback_parsed.is_complete(), elapsed=t.elapsed)
    check(group, "  net_contents/alcohol_content correctly absent, not a parse failure",
          throwback_parsed.net_contents is None and throwback_parsed.alcohol_content is None)
    check(group, "  DBA name ('(Used on label)') captured into bottler, not discarded",
          "BLACK MARKET SPIRITS" in (throwback_parsed.bottler or ""))
    check(group, "  all 3 declared images parsed (front, neck, back)",
          len(throwback_parsed.image_dimensions) == 3)

    with timer() as t:
        extracted, status = extract_all_label_images(real_pdf, parsed.image_dimensions)
    check(group, "extract_all_label_images: real record -> label extracted",
          status == "ok" and len(extracted) == 1, elapsed=t.elapsed)

    with timer() as t:
        throwback_images, tb_status = extract_all_label_images(
            throwback_pdf, throwback_parsed.image_dimensions)
    check(group, "extract_all_label_images: 3-image record -> all 3 correctly matched "
                 "(front/neck/back, not just 'largest image')",
          tb_status == "ok" and len(throwback_images) == 3, elapsed=t.elapsed)
    if tb_status == "ok":
        types_found = {img.image_type for img in throwback_images}
        check(group, "  signature/decoration correctly excluded (only declared types present)",
              types_found == {img.image_type for img in throwback_parsed.image_dimensions})

    with timer() as t:
        err_parsed = parse_cola_pdf(error_pdf)
        _, err_status = extract_all_label_images(error_pdf, err_parsed.image_dimensions)
    check(group, "extract_all_label_images: TTB error page correctly classified (not 'no label')",
          err_status == "error_page", elapsed=t.elapsed)

    client = TestClient(_app)

    with timer() as t:
        with open(real_pdf, "rb") as f:
            r = client.post("/api/verify/from-cola-pdf", files={"pdf": ("cola.pdf", f, "application/pdf")})
    body = r.json() if r.status_code == 200 else {}
    check(group, "POST /api/verify/from-cola-pdf: real record -> 200, all 8 checks incl. absolute size",
          r.status_code == 200 and body.get("overall_pass") is True and len(body.get("fields", [])) == 8,
          elapsed=t.elapsed)
    if r.status_code == 200:
        field_names = {f["field_name"] for f in body["fields"]}
        check(group, "  response includes government_warning_absolute_size",
              "government_warning_absolute_size" in field_names)

    with timer() as t:
        with open(chaglasian_pdf, "rb") as f:
            r = client.post("/api/verify/from-cola-pdf", files={"pdf": ("chaglasian.pdf", f, "application/pdf")})
    body = r.json() if r.status_code == 200 else {}
    check(group, "POST /api/verify/from-cola-pdf: 2-image record (warning on smaller BACK image) "
                 "-> government_warning found at all (not silently missed on wrong image)",
          r.status_code == 200 and any(
              f["field_name"] == "government_warning" and f["score"] > 0
              for f in body.get("fields", [])),
          elapsed=t.elapsed)

    with timer() as t:
        with open(error_pdf, "rb") as f:
            r = client.post("/api/verify/from-cola-pdf", files={"pdf": ("bad.pdf", f, "application/pdf")})
    check(group, "POST /api/verify/from-cola-pdf: TTB error page -> 400 with specific diagnosis",
          r.status_code == 400 and "system error page" in r.json().get("detail", ""),
          elapsed=t.elapsed)

    with timer() as t:
        r = client.post("/api/verify/from-cola-pdf",
                         files={"pdf": ("notapdf.pdf", b"not a pdf", "application/pdf")})
    check(group, "POST /api/verify/from-cola-pdf: corrupt upload -> 400, not a crash",
          r.status_code == 400, elapsed=t.elapsed)

    # Batch version: mix of two real, valid records and one real error page
    with timer() as t:
        with open(real_pdf, "rb") as f1, open(chaglasian_pdf, "rb") as f2, open(error_pdf, "rb") as f3:
            r = client.post(
                "/api/verify/batch-from-cola-pdf",
                files=[
                    ("pdfs", ("cascade.pdf", f1, "application/pdf")),
                    ("pdfs", ("chaglasian.pdf", f2, "application/pdf")),
                    ("pdfs", ("bad.pdf", f3, "application/pdf")),
                ],
            )
    body = r.json() if r.status_code == 200 else {}
    check(group, "POST /api/verify/batch-from-cola-pdf: 3 real PDFs (2 valid, 1 error page) -> "
                 "correctly isolates the bad one without failing the batch",
          r.status_code == 200 and body.get("total") == 3
          and body.get("succeeded") == 2 and body.get("failed") == 1
          and body["results"][0]["status"] == "ok"
          and body["results"][2]["status"] == "error",
          elapsed=t.elapsed)


def test_json_serialization():
    print("\n-- models.py: JSON serialization --")
    group = "json_serialization"

    submission = LabelSubmission(
        brand_name="Old Tom Distillery",
        class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45% Alc./Vol. (90 Proof)",
        net_contents="750 mL",
        bottler="Old Tom Distillery Co., Bardstown, KY",
    )

    pass_img = SAMPLE_DIR / "old_tom_bourbon_pass.png"
    with timer() as t_ocr:
        lines = extract_text(pass_img)
    with timer() as t_full:
        report = verify_label(submission.as_field_dict(), lines)
        response = VerificationResponse.from_report(report, t_ocr.elapsed * 1000)
        json_str = response.model_dump_json()
    check(group, "LabelSubmission -> verify_label -> VerificationResponse -> JSON",
          response.overall_pass and '"overall_pass":true' in json_str, elapsed=t_full.elapsed)
    check(group, "JSON round-trips back into VerificationResponse",
          VerificationResponse.model_validate_json(json_str).overall_pass == response.overall_pass)

    with timer() as t:
        try:
            LabelSubmission(brand_name="Old Tom Distillery", class_type="Bourbon")
            rejected = False
        except ValidationError:
            rejected = True
    check(group, "incomplete submission (missing fields) -> rejected by validation", rejected, elapsed=t.elapsed)


if __name__ == "__main__":
    print("Running label-verifier pipeline tests...")
    test_government_warning_unit_cases()
    test_field_matching_unit_cases()
    test_end_to_end_real_images()
    test_warning_text_size()
    test_absolute_size_requirements()
    test_real_world_cola()
    test_concurrent_ocr_correctness()
    test_concurrent_ocr_memory_stable()
    test_cola_pdf_intake()
    test_rotation_handling()
    test_api_endpoints()
    test_json_serialization()

    print(f"\n{PASSED} passed, {FAILED} failed")

    if TIMINGS:
        print("\n-- timing summary --")
        groups: dict[str, list[float]] = {}
        for group, name, elapsed in TIMINGS:
            groups.setdefault(group, []).append(elapsed)

        for group, times in groups.items():
            avg = sum(times) / len(times)
            print(f"  {group:20s} avg={avg*1000:8.1f} ms   (n={len(times)}, "
                  f"min={min(times)*1000:.1f} ms, max={max(times)*1000:.1f} ms)")

        overall_avg = sum(e for _, _, e in TIMINGS) / len(TIMINGS)
        print(f"  {'OVERALL':20s} avg={overall_avg*1000:8.1f} ms   (n={len(TIMINGS)})")

    sys.exit(1 if FAILED else 0)
