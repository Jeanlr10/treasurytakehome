# Label Verifier -- single-container deployment.
#
# Serves both the API (FastAPI, main.py) and the static frontend from
# ONE process/port -- main.py mounts ../frontend as static files, so
# there's no separate frontend server and no CORS to configure in
# production.
#
# OCR models are downloaded and baked into the image at BUILD time (see
# the "warm up" RUN step below), not left to happen on first request.
# This matters for two reasons: it avoids a slow, fragile first real
# request in production, and it avoids depending on the deployed
# container's own network access/egress rules at runtime (some hosts
# restrict outbound traffic more than their build environment does).

FROM python:3.12-slim

# PaddlePaddle's math kernels (OpenMP-based) default to spawning one
# internal thread PER VISIBLE CPU CORE for every single inference call,
# unless told otherwise. This was originally set to fix a real
# oversubscription bug when OCR ran across multiple outer Python threads
# at once (confirmed on real hardware: a 16-core box with no thread
# limit set made concurrent multi-image processing measurably SLOWER
# than sequential -- threads fighting each other for cores). That outer
# concurrency was later removed entirely (see docs/DESIGN_NOTES.md --
# it turned out to be a net loss even once oversubscription was fixed,
# likely GIL contention), so this setting is no longer protecting
# against multiple outer threads competing for cores. Left in place
# because removing it hasn't been tested: with no outer concurrency now,
# letting a single sequential call use multiple cores for its own math
# kernels COULD be faster than currently measured -- an open, real,
# not-yet-tested follow-up, not a settled question either way.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# opencv-contrib-python (used directly by ocr_engine.py for perspective
# transforms) needs a few system libraries not in the slim base image.
# libgomp1 is required separately by paddlepaddle's own compiled core
# (libpaddle.so links against it for OpenMP) -- confirmed by an actual
# build failure: "ImportError: libgomp.so.1: cannot open shared object
# file", not guessed preemptively.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer, cached across builds
# unless requirements.txt itself changes).
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r backend/requirements.txt

# Copy the actual application.
COPY backend ./backend
COPY frontend ./frontend
COPY sample_labels ./sample_labels

# Set the OCR tier BEFORE the warm-up step below, so the warm-up
# downloads/bakes in the model files for the tier actually being deployed
# -- not the default. Uses the same validated CLI a human would run
# locally (see ocr_config.py), just invoked at build time instead.
# Change "balanced" to "fast" or "accurate" to deploy a different tier --
# see ocr_config.py for the measured speed/accuracy tradeoffs of each.
RUN cd backend && python3 ocr_config.py balanced

# Bake OCR models into the image at build time (see module docstring
# above for why this matters). Uses a real sample label so both
# detection AND recognition models get downloaded and exercised, not
# just constructed. Reads whichever tier was just configured above.
RUN cd backend && python3 -c "from ocr_engine import extract_text; extract_text('/app/sample_labels/old_tom_bourbon_pass.png')"

# Most hosts (Hugging Face Spaces, Render, Railway, Fly.io) set $PORT at
# runtime; default to 7860 (Hugging Face Spaces' Docker SDK convention)
# if it isn't set.
ENV PORT=7860
EXPOSE 7860

WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
