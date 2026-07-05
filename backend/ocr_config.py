"""
ocr_config.py

Defines the available OCR model tiers and handles reading/writing the
reviewer's chosen tier to a small JSON config file.

Why this exists: this project was built assuming "2020-era laptop, CPU
only" as the baseline (per the take-home's stakeholder notes), and the
default model tier ("fast") is tuned for that. This is intended to be run
via the Docker container (see the project README), which always uses the
"fast" tier baked in at build time -- there's no interactive picker in
that path by design. If you're running the backend directly (not via
Docker) on stronger hardware and want a different tier, set it by hand:
write `{"tier": "balanced"}` or `{"tier": "accurate"}` to
`backend/ocr_tier_config.json` before starting the server. See TIERS
below for the exact valid tier names and their measured tradeoffs.

All timing numbers below are MEASURED (see ocr_engine.py's module
docstring and project history for methodology), not estimated, on a
CPU-only dev container. Re-benchmark on your own hardware — these are a
reference point, not a guarantee.

IMPORTANT, tested and confirmed: the known truncation issue on very long
rotated text lines (see ocr_engine.py) is IDENTICAL across the mobile and
medium tiers — it is not a model-size problem, so picking a heavier tier
here will NOT fix it. Tier choice affects general recognition accuracy
and speed, not that specific limitation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

CONFIG_PATH = Path(__file__).resolve().parent / "ocr_tier_config.json"

DEFAULT_TIER = "fast"


class TierSpec(TypedDict):
    label: str
    det_model: str
    rec_model: str
    description: str
    measured_speed: str


TIERS: dict[str, TierSpec] = {
    "fast": {
        "label": "Fast (mobile)",
        "det_model": "PP-OCRv5_mobile_det",
        "rec_model": "en_PP-OCRv5_mobile_rec",
        "description": (
            "Smallest models. Built for the take-home's stated baseline: "
            "a 2020-era CPU-only laptop. Default — use this unless you "
            "know you have stronger hardware."
        ),
        "measured_speed": "~2.5-4s per label (warm) on a CPU-only container",
    },
    "balanced": {
        "label": "Balanced (medium)",
        "det_model": "PP-OCRv6_medium_det",
        "rec_model": "PP-OCRv6_medium_rec",
        "description": (
            "Larger detection + recognition models. Somewhat more robust "
            "on noisy or low-quality images. Needs a reasonably modern "
            "CPU (or a few years newer than the 2020 baseline) to stay "
            "comfortably responsive."
        ),
        "measured_speed": "~7s per label (warm) on a CPU-only container",
    },
    "accurate": {
        "label": "Accurate (server)",
        "det_model": "PP-OCRv5_server_det",
        "rec_model": "PP-OCRv5_server_rec",
        "description": (
            "Largest models available for this pipeline. Best accuracy, "
            "slowest. Reasonable if you have a modern multi-core desktop "
            "or a GPU, and don't need sub-5-second responses."
        ),
        "measured_speed": "~7.5s per label (warm) on a CPU-only container",
    },
}


def load_tier() -> str:
    """Return the configured tier name, or DEFAULT_TIER if unset/invalid."""
    if not CONFIG_PATH.exists():
        return DEFAULT_TIER
    try:
        data = json.loads(CONFIG_PATH.read_text())
        tier = data.get("tier", DEFAULT_TIER)
        return tier if tier in TIERS else DEFAULT_TIER
    except (json.JSONDecodeError, OSError):
        return DEFAULT_TIER


def save_tier(tier: str) -> None:
    """Persist the chosen tier so ocr_engine.py picks it up on next run."""
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}. Valid options: {list(TIERS)}")
    CONFIG_PATH.write_text(json.dumps({"tier": tier}, indent=2))


def get_tier_models(tier: str | None = None) -> tuple[str, str]:
    """Return (det_model_name, rec_model_name) for the given tier (or the configured/default one)."""
    tier = tier or load_tier()
    spec = TIERS.get(tier, TIERS[DEFAULT_TIER])
    return spec["det_model"], spec["rec_model"]


if __name__ == "__main__":
    # Replaces the old tkinter-based setup_gui.py: same validated write via
    # save_tier(), same three tier choices, no GUI dependency. Docker is
    # this project's intended way to run/host it (tier fixed at build
    # time there by design) -- this is only for someone running the
    # backend directly on hardware where the default "fast" tier isn't
    # the right tradeoff.
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in TIERS:
        print(f"Usage: python ocr_config.py <tier>")
        print(f"Valid tiers: {', '.join(TIERS)}\n")
        for name, spec in TIERS.items():
            print(f"  {name}: {spec['description']} ({spec['measured_speed']})")
        sys.exit(1)

    save_tier(sys.argv[1])
    print(f"Saved tier '{sys.argv[1]}' to {CONFIG_PATH}. Takes effect next time the server starts.")
