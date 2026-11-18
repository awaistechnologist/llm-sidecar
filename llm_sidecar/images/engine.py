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

**A bounded MLX buffer cache.** The one that actually bit. MLX keeps freed
buffers for reuse and, uncapped, grew to 16.2 GiB beside 3.2 GiB of live
weights — after which the third 1024x1024 image took 77.6s following two at
~23s, on a machine that started 92% free. `cap_mlx_cache` removes the cliff.

That measurement also killed this module's original premise, which was that
loading is the expensive half and keeping a model warm is therefore the
point. It is not: mflux loads weights lazily, construction is under a
second, and a warm model measured no faster than a cold one. What holding
one model buys is a bound on how much is in memory at once — which on a
36 GiB laptop with a 31 GiB profile is reason enough, just not the reason
originally written here. Run `llm-sidecar image bench` rather than trusting
any of these numbers on a different machine.

mflux is an optional dependency — Apple Silicon only, and heavy. Nothing here
is imported until an image is actually requested, so the package installs and
runs on Linux and Windows with the image routes simply absent.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass
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

GIB = 1024 ** 3

# Ceiling on MLX's buffer cache — reusable freed buffers it would otherwise
# hoard. Left alone it reached 15.8 GiB here on a machine that started 92%
# free, and forced a thousand pageouts in a single generation. MLX gives the
# memory back when capped; the cap costs a little allocator churn and buys
# not being the process that pushes everything else to swap.
MLX_CACHE_LIMIT = 4 * GIB

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


def _fitting_profiles() -> list[str]:
    from .profiles import PROFILES
    return [n for n, p in PROFILES.items() if headroom(p)["verdict"] == "fits"]


def guard(profile: Profile, allow_tight: bool = True) -> dict:
    """Refuse before loading, rather than swapping for ten minutes.

    Ollama's failure mode taught this lesson on the LLM side: a model that
    does not fit is not refused, it is loaded and then crawls, which reads as
    a hang rather than an error."""
    h = headroom(profile)
    if h["verdict"] == "tight" and allow_tight:
        # Allowed, because "tight" still works and the ceiling is the
        # machine's, not a preference. Said out loud, because the symptom
        # otherwise is just an image that took three times as long, and the
        # cache cap cannot help when the working set alone nearly fills RAM.
        fits = ", ".join(n for n in _fitting_profiles()) or "none"
        logger.warning(
            "profile %s needs ~%.0f GiB and this machine has ~%.0f usable: it "
            "will run, slowly, and may swap. Profiles that fit comfortably: %s",
            profile.name, profile.peak_gib, h["usable_gib"] or 0, fits,
        )
    if h["verdict"] == "too_big" or (h["verdict"] == "tight" and not allow_tight):
        raise OutOfMemory(
            f"Profile {profile.name!r} peaks around {profile.peak_gib} GiB and this "
            f"machine has about {h['usable_gib']} GiB usable. It would swap rather "
            f"than run. Try a lighter profile, or close something."
        )
    return h


# ── the one resident model ────────────────────────────────────────────────


def _mx():
    """mlx.core, or None when mflux is not installed.

    One accessor rather than an `import mlx.core` at each call site, because
    `import a.b as x` resolves through the real package's attribute when it
    exists, so patching sys.modules in a test is silently ignored on a
    machine that actually has mlx. Tests patch this instead and behave the
    same on Linux CI, where there is no mlx at all."""
    try:
        import mlx.core as mx
        return mx
    except ImportError:
        return None


def cap_mlx_cache(peak_gib: float, usable_gib: float | None = None) -> float | None:
    """Bound MLX's buffer cache, just above the working set.

    The single most important call in here, and the size matters more than
    the fact of capping. MLX keeps freed buffers for reuse; left alone it
    grew to 16.2 GiB beside 3.2 GiB of live weights, and 1024x1024 images
    ran anywhere from 22.5s to 77.6s on a machine that started 92% free.

    But capping too tightly is its own tax. Measured on an M3 Max, four runs
    each, where the generation's peak allocation was 11.9 GiB:

        uncapped   22.5 - 77.6s    unpredictable
        4 GiB      35.8 - 48.7s
        8 GiB      31.8 - 40.3s
        12 GiB     23.1 - 23.9s    fastest and steadiest

    The pattern is not "smaller is safer". A cap below the peak working set
    forces MLX to release and re-acquire memory in the middle of a
    generation, and that churn is what costs the time. A cap just above it
    holds the working set without churning, and still refuses to balloon.

    So: size the cap to the profile's peak, not to a fraction of RAM. The
    headroom is 15%, and `observed` lets a caller feed back what a real
    generation actually peaked at, since a larger image needs more than the
    profile's nominal figure."""
    mx = _mx()
    if mx is None:
        return None

    if usable_gib is None:
        from .. import hardware
        usable_gib = hardware.usable_gib()

    limit = max(peak_gib * 1.15, 2.0)
    if usable_gib:
        # Never promise the allocator more than the machine can give.
        limit = min(limit, usable_gib * 0.8)
    mx.set_cache_limit(int(limit * GIB))
    logger.debug("MLX cache capped at %.1f GiB (peak %.1f)", limit, peak_gib)
    return round(limit, 1)


def reset_peak() -> None:
    """Start a generation's high-water mark from zero.

    `mx.get_peak_memory()` is a process-lifetime maximum, not a per-call
    one, and forgetting that was a real bug: one Z-Image generation peaking
    at 27.8 GiB left every later reading at 27.8 GiB, so retune_mlx_cache
    then sized a klein edit's cap off Z-Image's working set. The cap went to
    26.4 GiB for a 12 GiB profile and that run paged the most of any
    measured. Resetting per generation is what makes the figure mean
    anything."""
    mx = _mx()
    if mx is not None:
        mx.reset_peak_memory()


def retune_mlx_cache(profile: Profile) -> None:
    """Raise the cap if *this* generation peaked above the profile estimate.

    A profile's peak_gib is a figure for its usual size; ask for 2048x2048
    and the real peak is higher. Left alone the cap would then sit under the
    working set, which the numbers above show is the slow case.

    Only meaningful because reset_peak runs first — see the bug it fixes."""
    mx = _mx()
    if mx is None:
        return
    observed = mx.get_peak_memory() / GIB
    if observed > profile.peak_gib:
        logger.info("%s peaked at %.1f GiB (estimate %.1f); raising the cap",
                    profile.name, observed, profile.peak_gib)
        cap_mlx_cache(observed)


def _encoder_evicted(model) -> bool:
    """Whether MemorySaver has dropped this model's text encoder.

    Such a model can still decode but cannot encode another prompt, so it is
    spent. Variants without a text encoder at all (SeedVR2 upscaling) and
    ones using the small encoder are unaffected."""
    if model is None or not hasattr(model, "text_encoder"):
        return False
    if getattr(model, "small_text_encoder", False):
        return False
    return model.text_encoder is None


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
            mx = _mx()
            if mx is not None:
                mx.clear_cache()

    def expire_if_idle(self, idle_s: float = IDLE_UNLOAD_S) -> None:
        with self._lock:
            if self.model is not None and (time.time() - self.last_used) > idle_s:
                self.unload()

    @staticmethod
    def key_for(profile: Profile) -> str:
        """The resident-model identity. Two profiles sharing it need no swap."""
        return f"{profile.variant}:{profile.model}:{profile.quantize}"

    def get(self, profile: Profile):
        """The model for this profile, loading or swapping as needed.

        Returns (model, load_s, warm). `warm` is the number that matters and
        `load_s` is close to meaningless: mflux constructs in well under a
        second and streams the weights in lazily during the first denoising
        step. Measured on an M3 Max, klein-4b q8 "loaded" in 0.6s and then
        spent 11.3s on step 1 against ~2s on steps 2-4. So the cost of a cold
        model hides inside the first generation, and the only honest way to
        show what keeping it resident saves is cold-vs-warm generate time."""
        key = self.key_for(profile)
        with self._lock:
            if self.key == key and self.model is not None:
                if not _encoder_evicted(self.model):
                    self.last_used = time.time()
                    return self.model, 0.0, True
                # Spent: the encoder went when the last prompt was encoded.
                # Rebuilding is cheap because mflux loads weights lazily —
                # far cheaper than the paging that holding it caused.
                logger.debug("text encoder evicted; rebuilding %s", key)

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
            cap_mlx_cache(profile.peak_gib)
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



# How many times a waiting request may be overtaken before it goes next,
# whatever model is resident. Three is a deliberate compromise: it is enough
# for affinity to group a short run of same-profile work, and small enough
# that a steady stream of one profile cannot hold another off for long. At a
# minute an image, three skips is the most anyone waits extra.
MAX_SKIPS = 3


@dataclass
class _Waiter:
    seq: int
    key: str            # the model key this request needs
    skipped: int = 0


class _Scheduler:
    """Admission control that prefers whatever model is already loaded.

    Strict first-come-first-served has a pathological case, and it is not
    hypothetical — it was measured. With two consumers using the daemon
    with different profiles, the log showed a construct and an unload on
    *every single generation*: klein-edit, unload, z-image-turbo, unload,
    klein-edit, unload. Each of those swaps is a reload, and for Z-Image
    that alone costs about 40s (117s cold against 76s warm). FIFO
    guarantees the worst outcome whenever callers interleave, and no amount
    of "order your batch by profile" advice helps, because neither caller
    can see the other's queue.

    So when the worker frees, the oldest waiter needing the resident model
    goes first. Fairness is bounded rather than hoped for: a waiter
    overtaken MAX_SKIPS times goes next regardless of what is loaded.

    Two honest limits. This orders work *inside the daemon*; a library-mode
    caller in another process is serialised by the file lock, which has no
    priority and cannot be given one. And affinity is keyed on the model,
    not the profile, which is a feature — `characters-consistent` and
    `lineart` share klein-edit, so they are affine and swap nothing."""

    def __init__(self, max_skips: int = MAX_SKIPS) -> None:
        self._cv = threading.Condition()
        self._busy = False
        self._waiting: list[_Waiter] = []
        self._seq = 0
        self._max_skips = max_skips
        # Counters, so the effect is observable rather than asserted.
        self.served = 0
        self.swaps = 0
        self.overtakes = 0

    def _choose(self) -> _Waiter | None:
        """Who goes next. Caller holds the condition."""
        if not self._waiting:
            return None
        oldest = self._waiting[0]
        if oldest.skipped >= self._max_skips:
            return oldest
        resident = RESIDENT.key
        if resident is not None:
            for w in self._waiting:
                if w.key == resident:
                    return w
        return oldest

    @contextlib.contextmanager
    def slot(self, key: str):
        with self._cv:
            self._seq += 1
            me = _Waiter(seq=self._seq, key=key)
            self._waiting.append(me)
            # A new arrival can change who should go next, so re-evaluate.
            self._cv.notify_all()

            while True:
                if not self._busy and self._choose() is me:
                    for w in self._waiting:
                        if w is me:
                            break
                        w.skipped += 1
                        self.overtakes += 1
                    self._waiting.remove(me)
                    self._busy = True
                    if RESIDENT.key is not None and RESIDENT.key != key:
                        self.swaps += 1
                    self.served += 1
                    break
                # The timeout is a backstop against a lost wakeup, not the
                # mechanism; generations take seconds, so polling costs
                # nothing measurable.
                self._cv.wait(timeout=1.0)
        try:
            yield
        finally:
            with self._cv:
                self._busy = False
                self._cv.notify_all()

    def stats(self) -> dict:
        with self._cv:
            return {
                "waiting": len(self._waiting),
                "busy": self._busy,
                "served": self.served,
                "model_swaps": self.swaps,
                "overtakes": self.overtakes,
            }


SCHEDULER = _Scheduler()


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

    with SCHEDULER.slot(RESIDENT.key_for(profile)), MACHINE_LOCK:
        model, load_s, warm = RESIDENT.get(profile)
        reset_peak()
        t = time.time()
        image = _call(model, profile, prompt=prompt, negative=negative, seed=seed,
                      width=width, height=height, steps=steps, refs=refs,
                      resolution=resolution)
        generate_s = round(time.time() - t, 2)

        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(path=str(out))
        RESIDENT.last_used = time.time()
        retune_mlx_cache(profile)

    if profile.post == "lineart":
        _lineart(out)

    return {"load_s": load_s, "generate_s": generate_s, "warm": warm,
            "path": str(out)}


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
    out = {"available": available(), "mflux": mflux_version(),
           **RESIDENT.status(), "queue": SCHEDULER.stats()}
    if out["available"]:
        from .. import hardware
        out["usable_gib"] = hardware.usable_gib()
        # Reported because the cache is what made the third image take
        # 77.6s after two at ~23s, and "it got slower" is otherwise an
        # unexplainable symptom. active is the weights and live arrays;
        # cache is reusable freed buffers, and should now sit at or below
        # the figure cap_mlx_cache set rather than climbing past 16 GiB.
        mx = _mx()
        if mx is not None:
            out["mlx_active_gib"] = round(mx.get_active_memory() / 2 ** 30, 2)
            out["mlx_cache_gib"] = round(mx.get_cache_memory() / 2 ** 30, 2)
            # The last generation's working set — reset_peak runs before
            # each one, so this is per-generation rather than the
            # process-lifetime maximum MLX reports by default. It is the
            # number the cap is sized from, and several profiles' peak_gib
            # are still estimates inherited from other projects' notes.
            out["mlx_peak_gib"] = round(mx.get_peak_memory() / 2 ** 30, 2)
    return out
