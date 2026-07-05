# TTB Label Verifier

An AI-powered prototype for verifying alcohol beverage labels against
their TTB (Alcohol and Tobacco Tax and Trade Bureau) application data —
built for a Treasury take-home exercise.

Upload a label photo alongside the declared application fields, or upload
a COLA (Certificate of Label Approval) application PDF directly and skip
manual data entry entirely. The system reads the label with OCR, checks
it against what was declared, strictly verifies the federally-mandated
Government Warning statement, and — when a source PDF supplies the
label's physical dimensions — runs a certified, CFR-cited check of the
warning statement's actual type size in millimeters.

**Live demo:** _add your deployed URL here_

## What it does

- **Two intake modes:** upload a label photo + type in the application
  fields, or upload one or more COLA PDFs and have both the fields and
  the label image(s) extracted automatically — select multiple files at
  once for bulk checking.
- **Field verification:** brand name, class/type, alcohol content, net
  contents, and bottler name/address — fuzzy-matched against the OCR'd
  label text, tolerant of real-world formatting differences (case,
  punctuation, unit abbreviations) without being tolerant of genuine
  mismatches.
- **Government Warning compliance:** strict, case-sensitive verification
  of the statutory warning text (27 U.S.C. § 215), plus a physical
  type-size check against 27 CFR 16.22(a) when a physical scale reference
  is available.
- **Batch processing** for multiple labels in one request.
- **Full transparency:** every result shows what was actually declared
  vs. detected, the raw OCR output, and a carousel of every label image
  processed — not just a pass/fail badge.

Tested throughout against real, publicly-sourced TTB COLA records (not
just synthetic data) — see [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md)
for specifics, including a real labeling defect this system caught that
TTB's own reviewers had independently documented.

## Quick start

**Docker is the intended way to run/host this** (see "Deployment" below)
— the steps in this section are for local development and testing
directly against the Python source, not the primary way to use it.

### 1. Install

```bash
cd backend
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads PaddleOCR model weights (one-time, ~a few hundred MB,
requires internet access). Everything after that runs fully offline — no
outbound calls per request.

*(Optional)* Defaults to the fast/mobile OCR tier, tuned for a 2020-era
CPU-only laptop per the brief. If you're on stronger hardware and running
this directly rather than via Docker:
```bash
python ocr_config.py balanced   # or: accurate
```
Prints the valid tier names and their measured speed/accuracy tradeoffs
if you run it with no argument or an invalid one.

### 2. Run

```bash
uvicorn main:app --reload
```

Open **`http://127.0.0.1:8000`** — one process serves both the API and
the frontend. Interactive API docs at `/docs`.

### 3. Test

```bash
python tests/test_pipeline.py
```

58 checks, including several run against real TTB records sourced from
the public COLA registry — see `tests/fixtures/`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /api/verify` | One label: photo + the five declared fields |
| `POST /api/verify/batch` | Multiple labels in a single request |
| `POST /api/verify/from-cola-pdf` | Upload a COLA PDF — fields and label image(s) auto-extracted |
| `POST /api/verify/batch-from-cola-pdf` | Multiple COLA PDFs in one request — bulk intake |

## Deployment

**Intended to be run via Docker** — a single container (`Dockerfile` at
the project root), API and frontend sharing one process/port, OCR models
baked into the image at build time so there's no slow first-request
download. The OCR tier is set at build time via `RUN python3 ocr_config.py
<tier>` in the `Dockerfile` (currently `balanced`) — no interactive
picker in this path by design; edit that line and rebuild to change it
(see "Quick start" above for the tier names and tradeoffs).

**A deliberate, acknowledged trade-off against the stated ~5-second
target**: Sarah Chen's interview notes are explicit that results need to
come back in about 5 seconds or the tool won't get used, based on a
prior vendor pilot that failed for exactly that reason. Measured, not
assumed: `fast` tier meets that bar (~2.5-4s/label); `balanced` (the
tier actually deployed here) measures ~7s/label, over it. Chosen anyway
for meaningfully better recognition accuracy — see `ocr_config.py` for
the per-tier numbers. This is a real trade-off, not an oversight; `fast`
is one Dockerfile-line-and-rebuild away if the 5-second target should
take priority instead (`RUN python3 ocr_config.py fast`).

```bash
docker build -t label-verifier .
docker run -p 7860:7860 label-verifier
```

Deploys as-is to Hugging Face Spaces (Docker SDK), Render, Railway,
Fly.io, or any container host — reads `$PORT` from the environment
(falls back to `7860`). See
[`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md#deployment-details) for
hosting options actually tried, including real performance numbers.

## Project structure

```
Dockerfile
backend/
├── main.py                     # FastAPI app + serves frontend/ as static files
├── ocr_engine.py                # detect -> straighten -> recognize OCR pipeline
├── ocr_config.py                # configurable OCR model tier (`python ocr_config.py <tier>`)
├── field_matcher.py             # fuzzy field matching + strict Government Warning check
├── label_size_requirements.py   # certified physical type-size check (27 CFR 16.22(a))
├── cola_pdf_parser.py            # parses a COLA PDF's declared fields
├── cola_label_extractor.py       # extracts label image(s) from a COLA PDF
├── models.py                    # Pydantic request/response schemas
├── requirements.txt
└── tests/
    ├── test_pipeline.py
    └── fixtures/                # real TTB COLA PDFs used in testing
frontend/
└── index.html                   # single-file HTML/CSS/JS, no build step
sample_labels/                   # synthetic test label images
docs/
└── DESIGN_NOTES.md               # full development narrative, decisions, and citations
```

## Approach, tools, and key assumptions

**Approach**
- OCR runs as a custom detect → perspective-straighten → recognize
  pipeline (PaddleOCR), rather than the default single-pass pipeline,
  specifically to handle rotated or tilted label text — a real, common
  layout on physical bottle labels, not an edge case.
- General fields use fuzzy string matching (rapidfuzz) tolerant of
  formatting differences; the Government Warning uses a separate, strict,
  case-sensitive path, since that field explicitly must not be forgiving.
- COLA PDF intake parses declared fields and extracts label images
  directly from a TTB application PDF, so a reviewer can skip manual data
  entry — tested against two different real TTB Form 5100.31 revisions.
  Multi-image records (front/back/neck) are OCR'd sequentially — a
  concurrent version was built and measured on real deployment hardware,
  and concurrency turned out to be 38% *slower*, not faster (likely GIL
  contention with no real OS-level parallelism to gain from it); reverted
  once that was confirmed rather than kept for its own sake. Full story,
  including two other real bugs found along the way, in the design notes.
- The physical size check computes a real pixels-per-mm scale from a
  PDF's stated label dimensions and checks the warning statement's actual
  measured height in millimeters against the cited CFR minimum — a
  certified measurement, not a proxy.

**Tools used**
- **PaddleOCR / PaddlePaddle** — text detection and recognition
- **rapidfuzz** — fuzzy string matching
- **FastAPI + Pydantic** — API layer and schemas
- **PyMuPDF (fitz)** — COLA PDF text and image extraction
- **Plain HTML/CSS/JS** — frontend, no framework or build step
- **Docker** — deployment

**Key assumptions**
- Target hardware is a 2020-era CPU-only laptop (per the brief); the
  default OCR tier is tuned for that and is configurable for stronger
  hardware.
- A same-photo relative size heuristic is used when no physical scale
  reference is available (a plain photo upload); the certified absolute
  measurement is only possible when a COLA PDF supplies its stated
  physical dimensions.
- Test data was sourced from TTB's own public COLA Registry (public
  record, CC0-licensed). Personal contact fields (phone, email,
  signature) from source applications are deliberately never extracted
  or retained — only label-relevant business fields are.

## Known limitations

- **Deployed at `balanced` tier (~7s/label), over the stakeholder-stated
  ~5-second target** (`fast` tier meets it, ~2.5-4s/label) — a
  deliberate trade for meaningfully better accuracy, not an oversight.
  See "Deployment" above for the one-line change to revert it.
- Full 90-degree label rotation and some mixed-orientation labels aren't
  reliably handled yet.
- Bold-face verification for "GOVERNMENT WARNING" isn't possible from OCR
  text alone (no font-weight information).
- Very long rotated text lines can occasionally get truncated by the
  recognition model — this fails safe (flags for human review) rather
  than silently passing.
- TTB's administrative class/type wording vs. a label's consumer-facing
  name, and imported-product bottler matching (foreign producer vs. US
  importer), are both real edge cases currently flagged for review rather
  than fully resolved.
- No frontend for the photo-batch endpoint (`/api/verify/batch`) — bulk
  upload is supported in the frontend for COLA PDFs only (the realistic
  bulk-intake shape), not loose photos.

Full detail on all of these, including what was tried and measured, is in
[`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md#known-limitations-in-full).

## Further reading

[`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) has the complete
development narrative: real bugs found and fixed using actual TTB
records, calibration decisions with before/after numbers, and citations
checked directly against source for every regulatory claim.
