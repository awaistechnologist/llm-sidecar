"""Image generation — one local service for every project on this machine.

Five projects here each drove mflux themselves, in two different ways and
with two different costs:

  - in-process (social-media-poster, inspiring-kids/flux-test) skips mflux's
    CallbackManager, so MLX's buffer cache is never capped and generation
    times swing — 22.5s to 77.6s measured for the same image
  - CLI-per-image (inspired-tidings, quran-bloom, andalus) does get a
    MemorySaver, which evicts the text encoder, so every prompt rebuilds the
    model — about 20s of tax per image

Plus three mflux versions, two repos borrowing another repo's virtualenv,
per-directory locks that could not stop two repos generating at once, and
five provenance formats. This is the single owner of mflux: one resident
model behind a capped cache, which avoids both costs, and everything else
asks it.

Callers name a **profile**, not a model:

    from llm_sidecar import Sidecar

    sc = Sidecar()
    r = sc.generate_image("a red apple on a wooden table", profile="standard")
    print(r.path)

A profile carries the model, steps, quantisation, licence, memory peak and
reference-image limit together, so upgrading a model upgrades every caller at
once instead of five times, differently. See profiles.py.

Content policy stays with the projects. The Quran imagery gate, the Little
Muslims modesty rules and the andalus anachronism list are editorial
judgements belonging to the repo that holds them; this service never weakens
a project's gate and does not try to own one.

mflux is an optional, Apple-Silicon-only extra. Import this module anywhere;
it only fails when you actually ask for an image.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from . import engine, manifest, profiles
from .profiles import (  # noqa: F401  (re-exported)
    DEFAULT_PROFILE,
    ImageError,
    LicenceError,
    ProfileError,
)
from .engine import ImageUnavailable, OutOfMemory  # noqa: F401

logger = logging.getLogger("llm_sidecar.images")

__all__ = [
    "generate", "ImageRequest", "ImageResult", "ImageError", "ProfileError",
    "LicenceError", "ImageUnavailable", "OutOfMemory", "DEFAULT_PROFILE",
]

MAX_SEED = 2 ** 31 - 1


@dataclass
class ImageRequest:
    prompt: str
    profile: str = DEFAULT_PROFILE
    negative: str | None = None
    refs: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    out_dir: str = "."
    stem: str = "image"
    # Exact output file name, for a caller with its own naming scheme.
    # Unset means <stem>_<seed>.png — see manifest.output_path.
    filename: str | None = None
    # Which project asked. Recorded in the manifest, because "who made this
    # and with what" is the question provenance exists to answer.
    project: str = ""
    # Whether the output will be used commercially. Checked against the
    # profile's licence *before* anything loads — FLUX.1-dev and Kontext are
    # non-commercial, and that is easy to forget at a call site.
    commercial: bool = True
    # Skip generation when an identical image is already there.
    skip_existing: bool = True
    # SeedVR2 only: target long edge in pixels.
    resolution: int | None = None


@dataclass
class ImageResult:
    path: str
    manifest_path: str
    profile: str
    model: str
    seed: int
    digest: str
    seconds: float
    warm: bool
    skipped: bool = False
    width: int = 0
    height: int = 0


def _resolve_refs(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        path = Path(p).expanduser()
        if not path.exists():
            raise ProfileError(f"Reference image not found: {path}")
        out.append(path)
    return out


def generate(req: ImageRequest, config: Config | None = None) -> ImageResult:
    """One image. Blocking, and serialised machine-wide by the engine lock."""
    profile = profiles.get(req.profile)
    refs = _resolve_refs(req.refs)
    # Licence and reference checks come before `require()` so a caller on a
    # machine without mflux still gets the *useful* error — "that profile is
    # non-commercial" beats "install an extra" when both are true.
    profiles.check(profile, commercial=req.commercial, refs=refs)
    engine.require()

    seed = req.seed if req.seed is not None else random.randint(1, MAX_SEED)
    width = req.width or profile.width
    height = req.height or profile.height
    steps = req.steps or profile.steps

    ref_hashes = [manifest.sha256_file(r) for r in refs]
    dg = manifest.digest(
        profile=profile.name, model=profile.model, prompt=req.prompt,
        negative=req.negative, steps=steps, width=width, height=height,
        seed=seed, ref_hashes=ref_hashes,
    )

    out_dir = Path(req.out_dir).expanduser()
    existing = out_dir / (req.filename or f"{req.stem}_{seed}.png")
    if req.skip_existing and existing.exists() and manifest.is_current(existing, dg):
        logger.info("up to date, skipping: %s", existing)
        return ImageResult(
            path=str(existing),
            manifest_path=str(existing) + ".json",
            profile=profile.name, model=profile.model, seed=seed, digest=dg,
            seconds=0.0, warm=True, skipped=True, width=width, height=height,
        )

    out = manifest.output_path(out_dir, req.stem, seed, filename=req.filename)
    timing = engine.run(
        profile, prompt=req.prompt, negative=req.negative, seed=seed,
        width=width, height=height, steps=steps, refs=refs,
        resolution=req.resolution, out=out, config=config,
    )

    payload = manifest.build(
        request={
            "project": req.project, "profile": req.profile, "prompt": req.prompt,
            "negative": req.negative, "refs": [str(r) for r in refs],
            "width": width, "height": height, "steps": steps,
            "commercial": req.commercial,
        },
        profile=profile, digest_value=dg, seed=seed, ref_hashes=ref_hashes,
        timing=timing, mflux_version=engine.mflux_version(),
    )
    mpath = manifest.write(out, payload)

    return ImageResult(
        path=str(out), manifest_path=str(mpath), profile=profile.name,
        model=profile.model, seed=seed, digest=dg,
        seconds=timing["generate_s"], warm=timing["warm"],
        width=width, height=height,
    )


def status() -> dict:
    """Engine state plus the profile table — one call for a caller deciding
    whether to bother."""
    return {**engine.status(), "profiles": profiles.describe()}
