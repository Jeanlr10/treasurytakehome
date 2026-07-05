# Design Notes

The full development narrative behind this prototype — real bugs found and
fixed against actual TTB records, calibration decisions with the numbers
behind them, and every regulatory citation checked directly against source.
The top-level [`README.md`](../README.md) has the concise version; this is
the detailed one.

## Contents
- [OCR pipeline design](#ocr-pipeline-design)
- [Model tier configuration](#model-tier-configuration)
- [Known, unresolved rotation regression](#known-unresolved-rotation-regression)
- [Fuzzy field matching](#fuzzy-field-matching)
- [Government Warning: strict content matching](#government-warning-strict-content-matching)
- [Government Warning: text size checks](#government-warning-text-size-checks)
- [Real-world validation](#real-world-validation)
- [Absolute (physical) size requirements](#absolute-physical-size-requirements)
- [COLA PDF intake](#cola-pdf-intake)
- [Frontend design notes](#frontend-design-notes)
- [Deployment details](#deployment-details)
- [Known limitations, in full](#known-limitations-in-full)

## OCR pipeline design

**Model choice: PaddleOCR TextDetection + TextRecognition as separate
stages, not the default single-pass pipeline.** Considered GLM-OCR (0.9B)
and Granite-Docling, but this environment is explicitly laptop-class,
CPU-only hardware per the stakeholder notes ("assume a 2020-era
workstation"). PaddleOCR's default single-call pipeline works fine for a
plain upright label, but real alcohol labels often print the Government
Warning rotated 90 degrees along a narrow side panel — a real, common
physical layout, not an edge case — and the default pipeline produces
garbage on that panel while reading the upright body fine. Splitting into
detect → geometric-angle-check → perspective-straighten → recognize (see
`ocr_engine.py`'s module docstring for the full history of what was tried
and measured) fixes that for the common case and also fixed mild
hand-held-photo tilt (a regression an earlier, simpler version of this
approach had — caught by testing, not assumed).

## Model tier configuration

Model tier is configurable (`ocr_config.py`) since the "2020 laptop"
baseline is a stated assumption, not a hard constraint — reviewers on
stronger hardware running the backend directly (not via Docker) can trade
speed for accuracy via `python ocr_config.py <tier>`. Confirmed via direct
testing that the long-line truncation issue (below) is identical across
tiers; picking a bigger model does not fix it.

An earlier version of this had an interactive tkinter GUI
(`setup_gui.py`) for picking the tier. Removed: this project is intended
to be run via Docker, which fixes the tier at build time by design and
never touched the GUI anyway, so it was serving an increasingly narrow
audience (only someone cloning the repo and running the backend directly
on stronger-than-baseline hardware) while adding a real dependency risk
(`tkinter` needs a separate system package on some platforms, documented
install failures were seen during this project's own development). The
GUI's only real logic was a validated write to the tier config file
(`save_tier()`, in `ocr_config.py`) — kept, and given its own small CLI
entry point in the same file, so removing the GUI didn't strand a
genuinely useful safety check (rejects an invalid tier name) behind a
dependency that added more risk than value.

## Known, unresolved rotation regression

A full 90-degree whole-image rotation (e.g. a label photographed sideways)
and the Government Warning field specifically on a mixed-orientation label
(upright body + rotated panel) do not currently work reliably. An earlier,
simpler version of this pipeline (bucket angles near 90 degrees, rotate
the axis-aligned bounding box by a fixed -90 degrees) handled both of
these correctly, but produced visibly tilted crops — and measurably worse
recognition — on mild tilts. Switching to a general perspective transform
(straighten along the actual detected quadrilateral, for any angle) fixed
the tilt case, but introduced a reading-direction ambiguity: straightening
a quad to the right aspect ratio doesn't guarantee the text reads
right-side-up rather than upside-down/mirrored, and pure box geometry has
no way to resolve that — it would need a learned orientation classifier
applied *only* to regions with a non-trivial original angle (tested;
applying it to every region regardless of angle actively corrupts
already-correct upright text). That targeted classifier step was not
implemented before development moved on to other priorities.

**If accurate handling of fully-rotated or mixed-orientation labels
matters for your use case, this needs further work before relying on
it** — for now, plain-upright and mildly-tilted labels (the more common
real-world case for hand-held photos) work well; strongly rotated labels
do not.

## Fuzzy field matching

Directly answers Dave Morrison's complaint from the stakeholder interview
notes: "STONE'S THROW" on a label vs "Stone's Throw" on the application is
the same brand, not a mismatch, and naive exact-string comparison would
wrongly reject it. `match_field_fuzzy` takes the max of three rapidfuzz
scores — `partial_ratio`, `token_sort_ratio`, `token_set_ratio` — against
a normalized (lowercased, punctuation-stripped, unit-aliased) text blob,
with per-field thresholds where the default (88) doesn't fit (see
"Real-world validation" below for how those were calibrated).

## Government Warning: strict content matching

Jenny Park was explicit that this one field is *not* supposed to be
forgiving — it has to be word-for-word, in ALL CAPS, and she gave a real
example of a rejection over title case ("Government Warning" instead of
"GOVERNMENT WARNING"). Routing this field through the fuzzy matcher would
silently defeat the exact compliance check it exists to enforce, so it's a
fully separate code path:
1. Heading `GOVERNMENT WARNING:` must appear case-sensitively.
2. The statutory body text (27 U.S.C. § 215) is compared at a much higher
   fuzzy threshold (95/100) than general fields, to allow minor OCR noise
   without accepting a substantively altered or incomplete warning.

Line selection for this check is **content-based, not positional** — an
earlier version took "everything from the heading to the end of the line
list," which broke on a real label (Cascade Winery, TTB ID
03211001000018) where the warning runs along a rotated side panel: some
body lines sorted *before* the heading in the OCR output, silently
dropping them and collapsing the score from ~100 to ~32. The fix
classifies each detected line by vocabulary overlap with the known
statutory text, and treats any line clearing a threshold as part of the
warning regardless of where it landed in the list — verified against the
real label this was found on (correctly separates all 4 genuine warning
lines from the other 10 unrelated label lines).

## Government Warning: text size checks

Jenny also described a separate failure pattern: applicants using a
noticeably smaller font specifically for the warning to bury it, even when
the wording itself is technically correct — she gave a real example of
catching this. Text content and text size are independent failure modes,
so this is checked as its own field (`government_warning_size`), not
folded into the content check. It compares the warning's median detected
line-height (pixels) against the median line-height of the rest of the
label's text in the same photo, flagging anything below 60% as "verify by
eye."

**Honestly scoped:** this is a same-photo relative comparison, not a
certified measurement — TTB's actual minimum type-size rule (27 CFR
16.22(a)) is specified in physical units (mm) relative to container size,
and there's no reliable way to recover physical scale from an arbitrary
photo without a size reference in frame. Tested against both a real
compliant label (75% ratio — safely above the threshold) and a
deliberately tiny-text violation (46% ratio, correctly flagged) — and
confirmed the violation case still passes the *content* check at 100%,
which is exactly why this needed to be a separate check rather than
folded into the existing one.

A second, **certified** check exists too — see "Absolute (physical) size
requirements" below.

## Real-world validation

Beyond the synthetic test labels, this project sourced real, publicly
approved TTB COLA records via TTB's own Public COLA Registry (public
record, CC0-licensed) to validate against actual data rather than only
hypotheticals. This found and fixed three real calibration gaps:

- **Unit normalization** — applications spell units out ("750
  MILLILITERS"), labels abbreviate them ("750mL"). Without normalizing
  common unit words to a canonical form first, those two strings share
  zero overlapping tokens no matter how good the fuzzy algorithm is.
- **`bottler` threshold lowered to 65** — labels are only required to
  print city/state/zip; the applicant's address on file usually includes
  the full street address and a corporate suffix. A genuine match on real
  data scored 80 (missing exactly the street + "INC."); a deliberately
  wrong bottler scored 33 — wide, safe margin.
- **`class_type` threshold lowered to 70** — TTB's assigned classification
  uses formal regulatory wording ("TABLE RED WINE") that doesn't always
  literally appear on the label ("Red Wine"). Genuine match scored 78.6;
  wrong and adjacent-wrong class/types scored 42.9 and 53.8.

The Government Warning content check was also independently re-validated
on this same real record: the rotated-panel ordering fix took its score
from 32.2 (broken) to 99.8 (correct), confirming the fix generalizes
beyond the synthetic test case that originally caught it.

This record (and three others — two different TTB Form 5100.31 revisions,
a 2-image record, a 3-image record, and a genuine TTB system error page)
are now permanent test fixtures (`tests/fixtures/`, exercised by
`test_real_world_cola` and `test_cola_pdf_intake` in
`tests/test_pipeline.py`), specifically so these fixes can't silently
regress. Only label-relevant business fields were ever extracted and
retained from source applications — no personal name, phone, email, or
signature.

**One genuinely interesting validation:** on a real record (Chaglasian,
TTB ID 19115001001298) with front and back label images, this system's
`government_warning` check scored 86.5 — just below the 95 threshold. TTB's
own qualifications section on that record states: *"a comas is missing
after machinery"* — this system independently flagged a real, TTB-documented
defect.

## Absolute (physical) size requirements

The relative heuristic above can't certify actual legal compliance,
because that requires real millimeters and there's normally no way to
recover an arbitrary photo's physical scale. A COLA PDF changes that: it
states the label's actual physical size directly (e.g. "Actual
Dimensions: 3.5 inches W X 4 inches H"), which, divided into the extracted
image's pixel dimensions, gives a real pixels-per-mm scale.

**27 CFR 16.22(a)**, verified directly against eCFR:

| Net contents | Minimum warning type height |
|---|---|
| 237 mL (8 fl. oz.) or less | 1 mm |
| More than 237 mL, up to 3 L | 2 mm |
| More than 3 L | 3 mm |

Same section: "GOVERNMENT WARNING" must be capitalized and bold; the rest
of the statement must NOT be bold — the asymmetric bold-face requirement
is still unverifiable from OCR text alone (see limitations); this module
checks size, not weight.

**Real end-to-end validation:** using the actual Cascade Winery PDF's
stated dimensions and the real extracted label image, the warning text
measures 2.98mm against a 2.0mm requirement for its 750mL container — a
real pass, computed from real physical units, not a relative guess.

This is wired into `POST /api/verify/from-cola-pdf` (that endpoint is what
supplies a `ScaleReference`, by parsing the "Actual Dimensions" line from
the source PDF). `/api/verify` (a plain photo upload with no known
physical scale) can't run this check and relies on the relative heuristic
instead. `label_size_requirements.py` also has its own standalone tests
independent of the endpoint.

## COLA PDF intake

`POST /api/verify/from-cola-pdf` lets a reviewer upload a COLA detail PDF
directly instead of manually typing in the application fields. To get
one: search TTB's Public COLA Registry
(`https://www.ttbonline.gov/colasonline/publicSearchColasBasic.do`, no
account needed), open a record, and print its detail page to PDF from a
browser (Ctrl/Cmd+P → Save as PDF).

This endpoint:
1. Parses declared field values from the PDF's text (`cola_pdf_parser.py`)
   — brand, class/type, bottler; ABV/net contents when the form revision
   declares them. Deliberately does **not** extract or retain
   phone/fax/email/personal name/signature, even though they're present
   on the form.
2. Extracts **every** real label image embedded in the PDF — front, back,
   neck, not just one (`cola_label_extractor.py`).
3. Runs OCR on each image separately and merges the results, then runs
   the same matching pipeline as `/api/verify` against the combined text.
4. If the PDF states physical dimensions for whichever image the warning
   was actually found on, also runs the certified absolute size check.

**Handles two real TTB Form 5100.31 revisions**, confirmed against actual
records, not assumed: OMB No. 1512-0092 (older, declares Net Contents and
Alcohol Content on Part I) and OMB No. 1513-0020 (newer, different field
numbering, and does **not** declare Net Contents/Alcohol Content on Part I
at all — those only appear on the label itself in this revision). Field
regexes match on label *text* ("BRAND NAME (Required)"), not a hardcoded
leading field number, specifically because numbering shifts between
revisions. `net_contents`/`alcohol_content` are optional and simply
omitted from matching when a revision doesn't declare them, rather than
blocking the whole PDF as unparseable.

**Multi-image extraction matches against the PDF's own declared "Image
Type" / "Actual Dimensions" entries, not a generic size heuristic.** An
earlier version picked "the single largest embedded image," which is
wrong whenever the back label — where the Government Warning usually is
— happens to be smaller than the front/brand image. Confirmed on a real
record (TTB ID 19115001001298): front image 546×769px, back image
288×516px — back is smaller, but that's where the warning actually is.
Matching each embedded image to its closest-aspect-ratio declared entry
fixes this, and as a side effect naturally excludes signatures/seals (they
have no declared entry to match against at all) without a separate
signature-detection heuristic — including correctly keeping an extreme
0.75in × 6in neck-label strip that a fixed aspect-ratio range would have
wrongly excluded.

**The "(Used on label)" DBA name is captured, not discarded.** Some
applicants' legal permit-holder name differs from the trade name actually
printed on the label (e.g. "Cutler's Artisan Spirits, Inc." on the permit
vs. "BLACK MARKET SPIRITS" on the real label) — the form explicitly flags
this with a "(Used on label)" marker, which now gets folded into the
expected bottler text instead of being thrown away.

Distinguishes three real failure modes with specific error messages,
verified against real PDFs (not just designed-and-hoped-for): a genuine
parse failure, a TTB system error page (invalid/unsupported ID — happens
on very old records), and no label image found (e.g. a label-exempt
application, which is legally allowed to have no image).

**Honest note on remaining match-quality edge cases**, found on real
records and not yet resolved (judgment calls about fuzzy-match thresholds,
not parsing bugs): TTB's administratively-assigned class/type wording
sometimes has little vocabulary overlap with a label's consumer-facing
product name (e.g. "OTHER (HERBS & SEEDS)" vs. "Fernet Liqueur"), and
imported products list the foreign producer as bottler on the label while
the application declares the US importer, which the current bottler
matcher doesn't specifically model. Both currently correctly get flagged
for human review rather than silently passed — arguably the right failure
direction for a compliance tool, but worth knowing about.

## Concurrent OCR for multi-image PDFs

Multi-image COLA PDFs (front/back/neck) originally ran OCR on each image
sequentially, which adds up: a real 3-image PDF measured ~44s total on a
CPU-only box, since a text-dense back label alone can take 10-15s (see
"Frontend design notes" below on why OCR time scales with detected line
count, not just image size). Since each image is independent, running
them in parallel is the obvious fix — but this genuinely was not a "just
add a ThreadPoolExecutor" change.

**A naive version of this was tested and found to be a real correctness
bug, not a theoretical concern.** `ocr_engine.py`'s detector/recognizer
are cached, shared singletons (`@lru_cache`). Running
`extract_text()` concurrently across threads, all hitting the same
shared model instances, produced **cross-contaminated results** —
confirmed by diffing exact text against a known-correct sequential run:
the front label's OCR output came back containing text that actually
belonged to the back label, and the back label's output was a scrambled
interleaving of both labels' real content. Reproduced consistently
across repeated trials, not a one-off fluke. For a compliance tool,
silently attributing one label's text to a different label is a serious
failure mode, not a minor glitch — this needed to be caught and fixed
before shipping, not shipped and hoped to be fine.

**The fix**: give each concurrent call its own independent
`TextDetection`/`TextRecognition` instances instead of sharing the
cached ones (`extract_text_concurrent()` in `ocr_engine.py`). This was
also measured, not assumed, to have negligible construction overhead —
PaddleX appears to cache the actual model weights at a lower level, so
building a new instance is cheap; it's specifically *concurrent
inference calls on one shared instance* that aren't thread-safe, not
model loading itself. Verified: byte-identical text output to sequential
`extract_text()` across repeated trials, at effectively the same
wall-clock cost as the (broken) shared-instance version — the fix cost
nothing.

**Real speedup measured in isolation** (unshared CPU, the same
conditions as an actual dedicated server, not a busy CI box): ~44s
sequential down to ~15s concurrent on a real 3-image PDF at the balanced
tier — bounded by the single slowest image (the dense back label), not
divided evenly by worker count, since the workload is uneven. Through
the real `/api/verify/from-cola-pdf` endpoint end-to-end: 14.2s wall
time, with field-match results identical to the very first time this PDF
was tested (before any of this concurrency work existed) — confirming
correctness through the full stack, not just the isolated OCR call.

**Honest note on the test suite's timing check**: `test_pipeline.py`'s
`test_concurrent_ocr_correctness` asserts the text output matches
sequential exactly (a real correctness gate), but deliberately does
**not** hard-assert that concurrent is faster — timing comparisons are
noisy under CPU contention (confirmed directly: in one full-suite run,
concurrent came back marginally *slower* than sequential, purely from
sharing the test machine with dozens of other CPU-heavy tests running
back to back). The speed number is printed for visibility, not gated on,
since a flaky test that fails under load for reasons unrelated to actual
correctness is worse than no test at all.

**A second, more serious performance issue was found on real deployment
hardware, not in any sandbox: thread oversubscription.** After deploying
the concurrent OCR fix above, a 16-core dedicated server measured
concurrent processing as *slower* than sequential, not faster --
diagnosed with real `podman inspect`/`nproc`/`env` output, not guessed.
Ruled out a container CPU quota first (`NanoCpus`/`CpuQuota`/`CpuPeriod`
all `0`, i.e. no limit, and the container correctly saw all 16 host
cores). The actual cause: no `OMP_NUM_THREADS` (or equivalent) was set
anywhere, so PaddlePaddle's OpenMP-based math kernels defaulted to
spawning one internal thread per visible core -- *per inference call*.
With 3 outer Python threads each independently triggering that, a
16-core box could see up to ~48 OS threads competing for 16 real cores,
which is worse than useless -- the context-switching/scheduling overhead
from that many competing threads outweighed any real parallel gain, and
plausibly outweighed running everything on just one thread. Fixed by
setting `OMP_NUM_THREADS=1` (`OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` alongside it, covering PaddlePaddle's possible
backends) in the `Dockerfile`, early enough to apply to both the
build-time model warm-up and runtime serving -- this makes the *outer*
`ThreadPoolExecutor` the only source of real parallelism, rather than
fighting an uncontrolled inner one. Not independently re-measured on the
16-core box after this fix (that verification is on whoever's running
this deployment to confirm) -- documented here as a real, diagnosed bug
and its fix, not as a claim that the fix was re-verified to actually
restore the expected speedup on that specific hardware.

**A third, genuinely serious issue: the fix for the original
thread-safety bug (constructing a brand-new model instance per call)
turned out to leak memory, and it was actually crashing the deployed
container in production before it was caught.** The real-world symptom:
after processing roughly 7-8 COLA PDFs in a row, the deployed service
started returning connection failures ("Couldn't reach the label
checking service") rather than HTTP errors -- consistent with the
container process itself dying (most likely OOM-killed) and Podman's
`Restart=always` bringing it back, which would explain why the failures
weren't permanent but recurred after a similar number of requests each
time.

Measured, not assumed: constructing a fresh `TextDetection`/
`TextRecognition` pair on every single call and monitoring RSS across 10
simulated PDFs (3 real images each, real OCR) showed memory climbing
steadily from a 303MB baseline to 1052MB with no sign of plateauing --
roughly 20MB of growth per simulated PDF, continuing linearly rather than
settling. At that rate, a container with something like a 1GB memory
limit would be expected to get OOM-killed after almost exactly the
7-8-PDF range being reported in production. Something in PaddleX's
underlying C++ backend is not fully releasing resources when a
Python-level `TextDetection`/`TextRecognition` instance is garbage
collected -- each individual instance's construction cost is cheap (as
documented in the now-superseded version of this fix), but the total
memory footprint across *all instances ever constructed over the
process's lifetime* grows without bound, which only shows up as a real
problem after enough requests accumulate -- exactly the kind of bug that
passes every quick correctness/speed check and only surfaces under
sustained real use.

**The fix: a small, fixed-size object pool** (`_get_model_pool()` /
`_extract_text_pooled()` in `ocr_engine.py`), sized to 4 (front/back/neck
plus one spare -- real COLA records realistically never exceed this).
Each `(detector, recognizer)` pair is constructed exactly once, for the
entire process's lifetime, the first time the pool is needed. Concurrent
calls borrow a pair for their own exclusive use via a thread-safe
`queue.Queue` and return it when done, which preserves the property that
actually matters for the original bug (no two threads ever call
`predict()` on the same instance at the same time) while adding the
property that matters for this one (the total number of instances ever
constructed is fixed and small, not proportional to the number of
requests served). If a PDF ever has more images than the pool size, the
extra images simply wait for a free pooled instance rather than spinning
up a new one -- a deliberate throttle, not a bug.

Re-ran the identical memory measurement with the pool in place, extended
to 25 simulated PDFs for higher confidence: RSS grew rapidly for the
first couple of PDFs (the pool's one-time construction cost, expected),
then plateaued -- first-5-PDF average 1221.6MB vs. last-5-PDF average
1257.6MB, a ~36MB total drift across 20 additional PDFs (noise-level,
consistent with normal allocator behavior, not an ongoing leak), a
dramatically different profile from the unbounded version's steady
~20MB-per-PDF climb with no plateau in sight. Also reconfirmed text
output is still byte-identical to sequential `extract_text()` with the
pooled implementation, so the correctness fix from before was not
undone in the process of fixing this.

## Frontend design notes

**Real performance finding, not a frontend bug:** OCR time scales with the
*number of detected text lines*, not just image size — a text-dense back
label (tasting notes, vineyard description, etc.) can take far longer than
a sparse front label. Measured on a real record: front image (11 lines)
took 3.7s, back image (54 lines) took 13.5s — same pipeline, same code,
just more text to recognize. The loading message reflects this honestly
("usually a few seconds, longer for labels with a lot of small text")
rather than promising a fixed short wait dense labels won't meet. No
per-image progress indicator exists yet for the multi-image PDF case, so
a 15-30s wait on a busy label currently looks identical to a fast 3s one
from the user's side, just longer — a future improvement could stream
per-image status.

**Two modes, one tab switcher:** "COLA PDF" (default landing tab — fields
and label image both auto-extracted) and "Photo + application details"
(upload a photo, type in the five fields). Same results view either way.

**Results always show what was actually expected/declared for each
field, not just pass/fail** — a reviewer can see "Expected: CASCADE
WINERY" whether that field passed or failed. A collapsible "What the
system read from the label" panel also shows the raw OCR output — this is
deliberately NOT attributed per-field (e.g. "this specific line is the
brand name"), since the fuzzy-matching architecture doesn't identify which
OCR line corresponds to which field; showing the honest full list lets a
reviewer cross-reference themselves rather than implying a precision the
system doesn't have.

**Label image carousel** alongside the results (side-by-side on screens
≥760px, stacked below on narrow viewports) — shows every image actually
processed, each captioned with its type. Images are returned as base64
data URIs, nothing persisted server-side. Also used per-item in the bulk
COLA-PDF result list (see "Bulk PDF upload" above) — refactored into a
reusable builder specifically to support that without duplicating the
carousel implementation.

**Design intent:** kept deliberately plain per the brief — Sarah mentioned
wanting something her 73-year-old mother could use, and Dave doesn't want
a tool that makes his job harder. One deliberate touch: the pass/fail
result renders as a stamp-style badge, a nod to how label approval has
historically worked (an actual rubber stamp on the COLA certificate) —
everything else stays quiet: one typeface, high contrast, large touch
targets, plain-language copy, visible focus states for keyboard use, and
`prefers-reduced-motion` respected.

Not built: a frontend for the photo-batch endpoint (`/api/verify/batch`)
— would need a genuinely different layout (a list of image+field rows),
left out to keep the page simple. Bulk upload *is* built for the COLA-PDF
path specifically — see "Bulk PDF upload" below.

## Bulk PDF upload

Scoped deliberately to PDFs only, not loose photos — Janet's "big
importers dump 200-300 label applications at once" scenario realistically
arrives as COLA PDFs pulled from the registry, not as a folder of loose
label photos, so that's the bulk-intake shape actually built.

Backend: `POST /api/verify/batch-from-cola-pdf` accepts multiple PDFs,
processes each independently through the same `_process_cola_pdf` helper
the single-PDF endpoint uses (refactored out specifically so both
endpoints share one code path rather than duplicating the OCR + matching
+ absolute-size logic), and reports each PDF's outcome separately — one
bad or unparseable PDF in a large batch doesn't take down the rest, same
principle as the existing photo-batch endpoint.

Frontend: the same "COLA PDF" dropzone accepts multiple files (`<input
multiple>`); selecting more than one routes to the batch endpoint
automatically rather than needing a separate mode/tab. Results render as
a collapsible list — one row per PDF, showing filename and pass/fail/error
at a glance, expandable to the full field-by-field detail (reusing the
exact same field-row rendering function as the single-PDF view, not a
separate implementation). Tested end-to-end with a real browser and three
real PDFs (two valid records, one genuine TTB error page) — correctly
processed the two valid ones and isolated the bad one with its specific
diagnosis, without the batch failing as a whole.

**Each expandable batch item also shows its own label image carousel**,
not just the single-result view. The carousel itself was refactored from
a single set of hardcoded DOM element IDs (`#carousel-img`,
`#carousel-caption`, etc. — fine when only one carousel could ever exist
on the page at once) into a `buildCarousel(images)` function that
constructs a fully self-contained element with its own state via
closure, so it can be instantiated independently any number of times —
once for the single-result view, once per row in a bulk result list.

**A real bug was caught here, not just designed around in advance**:
the first version inserted each item's carousel into its detail panel,
then called `renderFieldRows()` on that same panel to add the field
rows — but `renderFieldRows()` unconditionally clears its container's
`innerHTML` first (correct behavior for the single-result view, which
re-renders into the same container repeatedly), which silently wiped out
the carousel that had just been inserted moments earlier. Manual testing
caught this immediately (zero carousels rendered despite the API
response demonstrably containing `label_images`, confirmed by inspecting
the raw JSON directly) rather than being caught only by chance. Fixed by
giving field rows their own dedicated child container instead of sharing
the carousel's parent element — verified via direct DOM inspection
(`document.querySelectorAll('.batch-item-carousel')`) that both a
single-image item and a three-image item now render and navigate
correctly, with the three-image item's arrow controls confirmed to
actually change the displayed image and caption.

## Deployment details

The app is a single Docker container — `main.py` mounts `frontend/` as
static files and serves it from the same origin as the API, so there's
one service, one port, no CORS to configure in production.

**What the Dockerfile does, and why:**
- Installs `libgl1`/`libglib2.0-0`/`libgomp1` — `opencv-contrib-python`
  and `paddlepaddle`'s compiled core need these; the slim base image
  doesn't include them (both found via real build failures, not
  anticipated in advance).
- **Bakes the OCR models into the image at build time** (runs a real
  detection+recognition pass during `docker build`). Avoids a slow,
  fragile first request in production and a dependency on the deployed
  container's own network egress rules at runtime.
- **Tier is set explicitly in the Dockerfile**, via `RUN python3
  ocr_config.py <tier>` immediately before the model-warming step —
  writes the tier config file first, so the warm-up step downloads and
  bakes in the model files for whichever tier is actually being deployed,
  not always the default. Verified this wiring end-to-end, not just
  assumed: set the tier, ran the exact warm-up call the Dockerfile runs,
  and confirmed via the actual downloaded model directory names
  (`PP-OCRv6_medium_det`/`_rec` for `balanced`, matching what
  `get_tier_models()` resolves to) that the right models get pulled. To
  verify which tier a *running* deployed container actually has, check
  either the model directory it downloaded
  (`podman exec <container> ls /root/.paddlex/official_models/`) or the
  config file directly
  (`podman exec <container> cat /app/backend/ocr_tier_config.json`) —
  don't assume from the Dockerfile source alone that a given build
  actually has the tier you expect, since it's easy to change one
  without rebuilding/redeploying the other.

**Hosting options considered:**
- **Hugging Face Spaces (Docker SDK)** — free, no payment, deploys via
  `git push`. Works, but on the free CPU tier a real multi-image PDF took
  ~494 seconds to process in testing — a shared/oversubscribed tier isn't
  a good fit for this workload's actual latency needs.
- **Self-hosted (Podman + Quadlet + Cloudflare Tunnel)** — dramatically
  faster on dedicated hardware (the same multi-image PDF: ~6-8 seconds
  self-hosted vs. ~494 seconds on HF's free tier — roughly 60-80x). No
  port-forwarding or public IP needed; Cloudflare Tunnel provides a real
  HTTPS URL without exposing the home network directly.

## Known limitations, in full

- **Bold-face verification** — OCR text carries no font-weight
  information, so "GOVERNMENT WARNING must be bolder than surrounding
  text" can't be verified from text alone. Would need image-level stroke-
  width analysis; out of scope here.
- **Long rotated text line truncation** — on a mixed-orientation label,
  the longest Government Warning lines occasionally get cut short by a
  few words. Investigated in depth: not a cropping-boundary issue (tested
  with padding, no improvement), not an image-resize-width issue (tested
  `input_shape` 320–1280, identical truncation point), not tier-dependent
  (tested mobile vs. medium recognition models, identical truncation).
  Likely a fixed max-decode-length in the recognition model's CTC
  postprocessing — a deeper model-internals question than was in scope
  to resolve. Because the Government Warning field is held to a strict
  95% threshold, a truncated read correctly FAILS rather than being
  silently accepted — the safe failure direction for a compliance-
  critical field.
- **Full 90-degree rotation / mixed-orientation labels** — see "Known,
  unresolved rotation regression" above.
- **Class/type and imported-product bottler matching** — see "Honest
  note on remaining match-quality edge cases" under COLA PDF intake.
- **Photo-batch endpoint has no frontend** — API-only
  (`/api/verify/batch`); test via `/docs`, curl, or the automated test
  suite. Bulk upload for COLA PDFs *does* have a frontend — see "Bulk PDF
  upload" above.
- **No per-image progress indicator** in multi-image PDF mode — a slow
  request currently looks identical to a fast one, just longer.
