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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Config
from . import engine, manifest, profiles
from .profiles import (  # noqa: F401  (re-exported)
    DEFAULT_PROFILE,
    ImageError,
    LicenceError,
    ProfileError,
)
from .engine import EngineFault, ImageUnavailable, MemoryBusy, OutOfMemory  # noqa: F401

logger = logging.getLogger("llm_sidecar.images")

__all__ = [
    "generate", "ImageRequest", "ImageResult", "ImageError", "ProfileError",
    "LicenceError", "ImageUnavailable", "OutOfMemory", "MemoryBusy",
    "EngineFault", "DEFAULT_PROFILE",
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
    # SeedVR2 only. Scales the image so its *short* edge becomes this many
    # pixels — mflux computes `scale = resolution / min(w, h)`. Documented
    # as the long edge once, which sent a consumer's first upscales to 3840
    # px wide.
    resolution: int | None = None
    # How long to wait for memory to free up before refusing with
    # MemoryBusy. Waiting is the default because the callers that hit it
    # are batch scripts, which mostly treat any error as fatal: one that
    # stops at the first refusal makes no images. 0 refuses at once — the
    # phone app does, because a person is looking at a spinner.
    wait_for_memory_s: float = 300


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



# How long to wait for the daemon to say it is alive. Short: this runs before
# every generation, and a daemon that cannot answer in a second is not one we
# want to hand a job to.
DAEMON_PROBE_TIMEOUT = 1.0
# How long to wait for the image itself. Generation is minutes, not seconds.
DAEMON_REQUEST_TIMEOUT = 1800.0
# Probe result cache. Without it a batch pays a round trip per image.
_daemon_seen: tuple[bool, float] | None = None
_DAEMON_TTL = 5.0


def _daemon_base(config: Config) -> str:
    return f"http://{config.daemon_host}:{config.daemon_port}"


def daemon_available(config: Config) -> bool:
    """Whether a local daemon is up and should own the model."""
    global _daemon_seen
    now = time.time()
    if _daemon_seen and (now - _daemon_seen[1]) < _DAEMON_TTL:
        return _daemon_seen[0]
    ok = False
    try:
        import httpx
        r = httpx.get(_daemon_base(config) + "/health", timeout=DAEMON_PROBE_TIMEOUT)
        ok = r.status_code == 200
    except Exception:
        ok = False
    _daemon_seen = (ok, now)
    return ok


# The daemon reports which exception it raised in this header, because an
# HTTP status cannot tell a licence refusal from a bad profile name and a
# caller caching on LicenceError should not have to parse prose.
ERROR_HEADER = "X-Sidecar-Error"
_ERRORS = {
    "ImageUnavailable": ImageUnavailable,
    "LicenceError": LicenceError,
    "ProfileError": ProfileError,
    "OutOfMemory": OutOfMemory,
    "MemoryBusy": MemoryBusy,
    "EngineFault": EngineFault,
    "ImageError": ImageError,
}


def _generate_via_daemon(req: ImageRequest, config: Config) -> ImageResult:
    """Hand the job to the daemon, which owns the model.

    A refusal is re-raised as the same exception the daemon raised, and is
    NOT retried locally: a non-commercial profile or a missing reference is
    just as wrong in this process, and falling back would quietly defeat the
    licence check."""
    import httpx

    headers = {}
    if config.daemon_token:
        headers["Authorization"] = f"Bearer {config.daemon_token}"

    r = httpx.post(_daemon_base(config) + "/v1/images/generate",
                   json=asdict(req), headers=headers,
                   timeout=DAEMON_REQUEST_TIMEOUT)
    if r.status_code == 200:
        return ImageResult(**r.json())

    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    kind = _ERRORS.get(r.headers.get(ERROR_HEADER, ""), ImageError)
    raise kind(str(detail))


def _resolve_refs(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        path = Path(p).expanduser()
        if not path.exists():
            raise ProfileError(f"Reference image not found: {path}")
        out.append(path)
    return out


def generate(req: ImageRequest, config: Config | None = None, *,
             allow_daemon: bool = True) -> ImageResult:
    """One image.

    If the local daemon is running, the job goes to it and this process
    never loads a model. That is the same arrangement every other capability
    here already uses: inference goes to Ollama, embeddings go to Ollama,
    and the heavy weights live in one process that owns them. Images were
    the exception only because mflux is a library rather than a server, so
    whichever process called it paid the ~12 GiB itself — and four MCP
    servers each answering an image request could load four copies.

    Without a daemon it still works, in-process, serialised by a
    machine-wide file lock. That is the fallback, not the design.

    `allow_daemon=False` forces the local path. The daemon's own route
    passes it, or it would call itself."""
    from .. import config as config_mod

    config = config or config_mod.load()
    if allow_daemon and daemon_available(config):
        logger.debug("delegating image generation to the daemon")
        return _generate_via_daemon(req, config)

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
        seed=seed, ref_hashes=ref_hashes, resolution=req.resolution,
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
        wait_for_memory_s=req.wait_for_memory_s,
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


def prepare(profile_name: str, config: Config | None = None, *,
            wait_for_memory_s: float = 0, measure: bool = True,
            allow_daemon: bool = True) -> dict:
    """Save a quantised copy of a profile's model so it loads without the
    full-precision peak (see engine.prepare), then make one image at the
    profile's size to measure what the prepared copy really peaks at.

    Goes to the daemon when one is running, like generate(): the daemon owns
    the models, and preparing beside it would hold two."""
    import tempfile

    import httpx

    from .. import config as config_mod

    config = config or config_mod.load()
    if allow_daemon and daemon_available(config):
        headers = {}
        if config.daemon_token:
            headers["Authorization"] = f"Bearer {config.daemon_token}"
        r = httpx.post(_daemon_base(config) + "/v1/images/prepare",
                       json={"profile": profile_name,
                             "wait_for_memory_s": wait_for_memory_s,
                             "measure": measure},
                       headers=headers, timeout=DAEMON_REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise _ERRORS.get(r.headers.get(ERROR_HEADER, ""), ImageError)(str(detail))

    profile = profiles.get(profile_name)
    out = engine.prepare(profile, config, wait_for_memory_s=wait_for_memory_s)
    if measure and not out.get("already"):
        with tempfile.TemporaryDirectory() as tmp:
            r = generate(ImageRequest(
                prompt="a quiet garden at dawn, soft gouache illustration",
                profile=profile_name, out_dir=tmp, stem="measure",
                project="llm-sidecar prepare", skip_existing=False,
                wait_for_memory_s=wait_for_memory_s,
            ), config, allow_daemon=False)
        out["measure_s"] = r.seconds
    info = engine.prepared_info(profile) or {}
    out["peak_gib"] = info.get("peak_gib")
    out["was_peak_gib"] = profile.peak_gib
    return out


def status(config: Config | None = None) -> dict:
    """Engine state plus the profile table — one call for a caller deciding
    whether to bother.

    `delegating` says whether a generation from *this* process would be
    handed to the daemon rather than loading a model here."""
    from .. import config as config_mod

    config = config or config_mod.load()
    return {
        **engine.status(),
        "prepared": {n: engine.prepared_info(p) for n, p in profiles.PROFILES.items()
                     if engine.prepared_info(p) is not None},
        "delegating_to_daemon": daemon_available(config),
        "profiles": profiles.describe(),
    }
