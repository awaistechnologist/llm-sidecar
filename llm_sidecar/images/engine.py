"""The worker — load one model, keep it warm, generate.

Three things this owns that no per-project script managed:

**One model resident at a time.** Loading a second set of weights beside the
first is how a 36 GiB machine starts swapping; `draft` alone peaks at 31 GiB.
So the engine holds exactly one, swaps when a different profile is asked for,
and drops it after an idle timeout.

**A machine-wide lock, not a per-directory one.** Two of the repos here used
`fcntl` locks scoped to their own output folder, which stops a repo racing
itself and does nothing about two repos racing each other — the case that
actually causes swapping. The lock is on the engine, held across the whole
generation, and taken through a file so it holds across processes too: the
daemon and a library-mode caller in some other repo are not the same process.

**Loading is the expensive half.** Measured on this machine rather than
assumed; see BENCH. That is the entire argument for a shared service over
`subprocess` per image, so the numbers are recorded rather than claimed.

mflux is an optional dependency — Apple Silicon only, and heavy. Nothing here
is imported until an image is actually requested, so the package installs and
runs on Linux and Windows with the image routes simply absent.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from ..config import CONFIG_DIR, Config
from .profiles import ImageError, Profile

logger = logging.getLogger("llm_sidecar.images")

# (module path, class name) per profile variant. Verified against mflux
# 0.24.0 by importing each one — the names are not guessable from the model
# key, and two of them live several packages deep.
VARIANTS: dict[str, tuple[str, str]] = {
    "flux1": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
    "kontext": ("mflux.models.flux.variants.kontext.flux_kontext", "Flux1Kontext"),
    "klein": ("mflux.models.flux2.variants", "Flux2Klein"),
    "klein-edit": ("mflux.models.flux2.variants", "Flux2KleinEdit"),
    "z-image-turbo": ("mflux.models.z_image.variants", "ZImageTurbo"),
    "seedvr2": ("mflux.models.seedvr2.variants.upscale.seedvr2", "SeedVR2"),
}

LOCK_FILE = CONFIG_DIR / "images.lock"

# Drop the resident model after this long unused. Long enough that a batch
# of images pays the load once, short enough that an idle daemon is not
# sitting on 10-31 GiB of someone's laptop.
IDLE_UNLOAD_S = 10 * 60


class ImageUnavailable(ImageError):
    """mflux is not installed, or not installable on this platform."""


class OutOfMemory(ImageError):
    """This machine cannot hold the profile's weights."""


def available() -> bool:
    """Whether image generation can run here at all."""
    try:
        import mflux  # noqa: F401
        return True
    except ImportError:
        return False


def require() -> None:
    if available():
        return
    import platform
    extra = ""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        extra = (" mflux is built on MLX and runs on Apple Silicon only, so "
                 "this machine cannot run local image generation.")
    raise ImageUnavailable(
        "Image generation needs the optional extra: "
        'pip install "llm-sidecar[images]".' + extra
    )


def mflux_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("mflux")
    except Exception:
        return None


# ── memory guard ──────────────────────────────────────────────────────────


def headroom(profile: Profile) -> dict:
    """Can this machine hold the profile's weights, with the OS still running?

    Reuses the hardware module the model picker already uses, so "will this
    fit" is answered the same way for a 19 GiB LLM and a 31 GiB diffusion
    model instead of twice, differently."""
    from .. import hardware

    usable = hardware.usable_gib()
    if usable is None:
        return {"verdict": "unknown", "needs_gib": profile.peak_gib, "usable_gib": None}
    if profile.peak_gib <= usable * 0.8:
        verdict = "fits"
    elif profile.peak_gib <= usable:
        verdict = "tight"
    else:
        verdict = "too_big"
    return {
        "verdict": verdict,
        "needs_gib": profile.peak_gib,
        "usable_gib": round(usable, 1),
        "headroom_gib": round(usable - profile.peak_gib, 1),
    }


def guard(profile: Profile, allow_tight: bool = True) -> dict:
    """Refuse before loading, rather than swapping for ten minutes.

    Ollama's failure mode taught this lesson on the LLM side: a model that
    does not fit is not refused, it is loaded and then crawls, which reads as
    a hang rather than an error."""
    h = headroom(profile)
    if h["verdict"] == "too_big" or (h["verdict"] == "tight" and not allow_tight):
        raise OutOfMemory(
            f"Profile {profile.name!r} peaks around {profile.peak_gib} GiB and this "
            f"machine has about {h['usable_gib']} GiB usable. It would swap rather "
            f"than run. Try a lighter profile, or close something."
        )
    return h


# ── the one resident model ────────────────────────────────────────────────


class _Resident:
    """At most one loaded model, guarded for both threads and processes."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.model = None
        self.loaded_at = 0.0
        self.last_used = 0.0
        self.load_s = 0.0
        self._lock = threading.RLock()

    def unload(self) -> None:
        with self._lock:
            if self.model is None:
                return
            logger.info("unloading image model %s", self.key)
            self.model = None
            self.key = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

    def expire_if_idle(self, idle_s: float = IDLE_UNLOAD_S) -> None:
        with self._lock:
            if self.model is not None and (time.time() - self.last_used) > idle_s:
                self.unload()

    def get(self, profile: Profile):
        """The model for this profile, loading or swapping as needed.

        Returns (model, load_s, warm). `warm` is the number that matters and
        `load_s` is close to meaningless: mflux constructs in well under a
        second and streams the weights in lazily during the first denoising
        step. Measured on an M3 Max, klein-4b q8 "loaded" in 0.6s and then
        spent 11.3s on step 1 against ~2s on steps 2-4. So the cost of a cold
        model hides inside the first generation, and the only honest way to
        show what keeping it resident saves is cold-vs-warm generate time."""
        key = f"{profile.variant}:{profile.model}:{profile.quantize}"
        with self._lock:
            if self.key == key and self.model is not None:
                self.last_used = time.time()
                return self.model, 0.0, True

            # A different profile: drop the old weights *first*. Holding both
            # briefly is exactly the peak this service exists to avoid.
            self.unload()

            module_path, class_name = VARIANTS[profile.variant]
            import importlib

            from mflux.models.common.resolution.config_resolution import ConfigResolution

            cls = getattr(importlib.import_module(module_path), class_name)
            cfg = ConfigResolution.resolve(profile.model)

            t = time.time()
            kwargs = {"model_config": cfg}
            if profile.quantize is not None:
                kwargs["quantize"] = profile.quantize
            self.model = cls(**kwargs)
            self.load_s = round(time.time() - t, 2)
            self.key = key
            self.loaded_at = self.last_used = time.time()
            logger.info("constructed image model %s in %.1fs", key, self.load_s)
            return self.model, self.load_s, False

    def status(self) -> dict:
        with self._lock:
            return {
                "loaded": self.key,
                "load_s": self.load_s if self.key else None,
                "idle_s": round(time.time() - self.last_used, 1) if self.key else None,
            }


RESIDENT = _Resident()


class _MachineLock:
    """One generation at a time across every process on this machine.

    A thread lock is not enough: the daemon and a library-mode caller in
    another repo are different processes, and it is precisely two repos
    generating at once that makes a 36 GiB machine swap.

    Degrades to thread-only if flock is unavailable (Windows), which is
    harmless — that platform cannot run mflux anyway."""

    def __init__(self) -> None:
        self._thread_lock = threading.Lock()
        self._fh = None

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            import fcntl
            LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._fh = LOCK_FILE.open("w", encoding="utf-8")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            self._fh.write(f"{os.getpid()}\n")
            self._fh.flush()
        except Exception:
            self._fh = None
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            finally:
                self._fh.close()
                self._fh = None
        self._thread_lock.release()
        return False


MACHINE_LOCK = _MachineLock()


# ── generation ────────────────────────────────────────────────────────────


def _call(model, profile: Profile, *, prompt: str, negative: str | None,
          seed: int, width: int, height: int, steps: int,
          refs: list[Path], resolution: int | None):
    """Dispatch to the variant's own signature.

    These genuinely differ, and the differences matter: the edit variant takes
    `image_paths` (plural, up to three — the consistency feature), everything
    else takes a single `image_path`, SeedVR2 takes no prompt at all, and only
    Z-Image honours a negative prompt."""
    if profile.variant == "seedvr2":
        return model.generate_image(
            seed=seed, image_path=str(refs[0]), resolution=resolution or 2048,
        )

    kwargs = {
        "seed": seed, "prompt": prompt, "num_inference_steps": steps,
        "width": width, "height": height,
    }
    if profile.guidance is not None:
        kwargs["guidance"] = profile.guidance

    if profile.variant == "klein-edit":
        if refs:
            kwargs["image_paths"] = [str(r) for r in refs]
    elif refs:
        kwargs["image_path"] = str(refs[0])

    if negative:
        if profile.supports_negative:
            kwargs["negative_prompt"] = negative
        else:
            # Several models accept the argument and ignore it. Saying so
            # beats letting a caller believe their reject list is working.
            logger.info("profile %s ignores negative prompts", profile.name)

    return model.generate_image(**kwargs)


def run(profile: Profile, *, prompt: str, negative: str | None, seed: int,
        width: int, height: int, steps: int, refs: list[Path],
        resolution: int | None, out: Path, config: Config | None = None) -> dict:
    """One image, start to finish. Returns timing and the path written."""
    require()
    guard(profile)

    with MACHINE_LOCK:
        model, load_s, warm = RESIDENT.get(profile)
        t = time.time()
        image = _call(model, profile, prompt=prompt, negative=negative, seed=seed,
                      width=width, height=height, steps=steps, refs=refs,
                      resolution=resolution)
        generate_s = round(time.time() - t, 2)

        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(path=str(out))
        RESIDENT.last_used = time.time()
        _trim_cache()

    if profile.post == "lineart":
        _lineart(out)

    return {"load_s": load_s, "generate_s": generate_s, "warm": warm,
            "path": str(out)}


def _trim_cache() -> None:
    """Release MLX's buffer cache after each image, keeping the weights.

    Measured, not assumed. Keeping a model resident was supposed to make the
    second image faster; on an M3 Max at 1024x1024 it made it *three times
    slower* — 35s cold, then 96s warm. The weights were not the problem: MLX
    holds freed buffers in a cache for reuse, that cache grows with every
    generation, and on a 36 GiB machine it eventually pushes the resident
    model towards swap.

    `clear_cache` drops the cache without touching live arrays, so the model
    stays loaded and the next generation starts from a clean allocator.

    It halves the problem rather than solving it: with this in place the
    second image went 96s -> 42s, but the third was still 85s against 34s
    cold, with MLX reporting 0.0 GiB cached throughout. So the cache was a
    contributor and something else is the cause. Keeping this call because
    the improvement is real and measured; not claiming it is the fix."""
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def _lineart(path: Path) -> None:
    """Threshold and thicken to a colouring page.

    A post-process rather than a prompt instruction, because asking a
    diffusion model for "clean black outlines on white" gets grey, broken
    strokes that print badly."""
    try:
        from PIL import Image, ImageFilter
    except ImportError:
        logger.warning("lineart post-process needs Pillow; left as generated")
        return
    img = Image.open(path).convert("L")
    img = img.point(lambda v: 255 if v > 128 else 0, mode="1").convert("L")
    img = img.filter(ImageFilter.MinFilter(3))
    img.save(path)


def status() -> dict:
    """What the engine is holding right now."""
    RESIDENT.expire_if_idle()
    out = {"available": available(), "mflux": mflux_version(), **RESIDENT.status()}
    if out["available"]:
        from .. import hardware
        out["usable_gib"] = hardware.usable_gib()
        # Reported because it is what made a resident model slower than a
        # cold one here, and "the second image took three times as long" is
        # otherwise an unexplainable symptom. active is the weights and live
        # arrays; cache is reusable freed buffers, and should stay small
        # between generations now that _trim_cache runs after each.
        try:
            import mlx.core as mx
            out["mlx_active_gib"] = round(mx.get_active_memory() / 2 ** 30, 2)
            out["mlx_cache_gib"] = round(mx.get_cache_memory() / 2 ** 30, 2)
        except Exception:
            pass
    return out
