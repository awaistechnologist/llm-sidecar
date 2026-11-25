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

# A model that takes most of the machine is dropped after a minute instead:
# a 27 GiB Z-Image sitting idle for ten minutes is ten minutes of every other
# app on a 36 GiB laptop being squeezed into the compressor.
BIG_IDLE_UNLOAD_S = 60
BIG_SHARE = 0.6  # of usable memory

# After this long idle, give MLX's buffer cache back but keep the weights.
# The cache is what makes back-to-back jobs steady; between batches it is
# just memory taken from everything else (12.4 GiB of it beside 3.2 GiB of
# klein weights, measured idle).
CACHE_TRIM_S = 60

# Kept free beyond the profile's peak when deciding whether to start a job.
SAFETY_GIB = 2.0

# How long to wait for memory to come back after asking Ollama to unload.
FREE_WAIT_S = 10.0
# How often to look again while a caller waits for memory.
WAIT_POLL_S = 5.0

# After a GPU fault, refuse image work this long. The fault is a symptom of
# a machine already thrashing; starting the next 28 GiB job straight into it
# is what turned a failed image into a kernel panic on 25 Nov 2026.
FAULT_COOLDOWN_S = 60
FAULT_FILE = CONFIG_DIR / "images.fault"


class ImageUnavailable(ImageError):
    """mflux is not installed, or not installable on this platform."""


class OutOfMemory(ImageError):
    """This machine cannot hold the profile's weights."""


class MemoryBusy(ImageError):
    """The machine could hold the weights, but not right now: other programs
    are using the memory. Retry later. Distinct from OutOfMemory, which no
    amount of waiting fixes."""

    retry_after_s = 60


class EngineFault(ImageError):
    """The GPU failed mid-generation and the engine is restarting. Retry
    shortly."""

    retry_after_s = 30


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
        # Generations in flight. Expiry must never fire during one: last_used
        # is stamped when a generation starts, so a job that runs longer than
        # IDLE_UNLOAD_S — a 28 GiB model on a swapping 36 GiB machine does —
        # looked idle to the next status poll, which unloaded it mid-job.
        self.in_use = 0
        self.peak_gib = 0.0
        self.trimmed = False
        self._lock = threading.RLock()
        self._reaper: threading.Thread | None = None

    def unload(self) -> None:
        with self._lock:
            if self.model is None:
                return
            logger.info("unloading image model %s", self.key)
            self.model = None
            self.key = None
            self.peak_gib = 0.0
            mx = _mx()
            if mx is not None:
                mx.clear_cache()

    def idle_limit(self) -> float:
        from .. import hardware
        usable = hardware.usable_gib()
        if usable and self.peak_gib > usable * BIG_SHARE:
            return BIG_IDLE_UNLOAD_S
        return IDLE_UNLOAD_S

    def expire_if_idle(self, idle_s: float | None = None) -> None:
        with self._lock:
            if self.in_use or self.model is None:
                return
            limit = self.idle_limit() if idle_s is None else idle_s
            if (time.time() - self.last_used) > limit:
                self.unload()

    def trim_if_idle(self, idle_s: float = CACHE_TRIM_S) -> bool:
        with self._lock:
            if (self.in_use or self.model is None or self.trimmed
                    or (time.time() - self.last_used) <= idle_s):
                return False
            mx = _mx()
            if mx is None:
                return False
            mx.clear_cache()
            self.trimmed = True
            logger.debug("trimmed MLX cache for idle %s", self.key)
            return True

    def start_reaper(self) -> None:
        """Expire on a timer, not only when someone polls status.

        Before this, expiry ran only inside status(), so a daemon whose
        callers never polled kept the last model resident indefinitely —
        27 GiB, in the case that crashed this machine."""
        with self._lock:
            if self._reaper is not None and self._reaper.is_alive():
                return

            def loop():
                while True:
                    time.sleep(15)
                    try:
                        self.expire_if_idle()
                        self.trim_if_idle()
                    except Exception:
                        logger.debug("idle reaper failed", exc_info=True)

            self._reaper = threading.Thread(target=loop, name="image-reaper", daemon=True)
            self._reaper.start()

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
            ready = prepared_path(profile)
            if ready is not None:
                # Already quantised on disk: no full-precision copy in memory.
                kwargs["model_path"] = str(ready)
            self.model = cls(**kwargs)
            cap_mlx_cache(profile.peak_gib)
            self.load_s = round(time.time() - t, 2)
            self.key = key
            self.peak_gib = profile.peak_gib
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


# ── prepared (pre-quantised) copies ───────────────────────────────────────
#
# mflux loads a model's full-precision weights and quantises them in memory,
# so for a moment both copies are resident. For Z-Image that is ~20 GiB of
# bf16 plus the q8 result: the 27.8 GiB peak, when the q8 weights alone are
# about a third of that. Saving the quantised weights once and loading those
# skips the full-precision copy entirely. Same weights, same images.


def _prepared_dir(profile: Profile) -> Path:
    from ..ledger import DATA_DIR
    safe = RESIDENT.key_for(profile).replace(":", "-").replace("/", "-")
    return DATA_DIR / "image-models" / safe


def prepared_info(profile: Profile) -> dict | None:
    if profile.quantize is None:
        return None
    marker = _prepared_dir(profile) / "sidecar.json"
    try:
        import json
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def prepared_path(profile: Profile) -> Path | None:
    return _prepared_dir(profile) if prepared_info(profile) is not None else None


def effective_peak(profile: Profile) -> float:
    """The profile's peak, or the measured one for its prepared copy."""
    info = prepared_info(profile)
    if info and info.get("peak_gib"):
        return float(info["peak_gib"])
    return profile.peak_gib


def _record_peak(profile: Profile) -> None:
    """Keep the highest peak seen from the prepared copy, so the memory
    check uses a measurement rather than the full-precision estimate."""
    info = prepared_info(profile)
    mx = _mx()
    if info is None or mx is None:
        return
    observed = round(mx.get_peak_memory() / GIB, 2)
    if observed <= 0 or observed <= (info.get("peak_gib") or 0):
        return
    import json
    info["peak_gib"] = observed
    info["peak_measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (_prepared_dir(profile) / "sidecar.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")
    logger.info("%s (prepared) peaked at %.1f GiB", profile.name, observed)


def prepare(profile: Profile, config: Config | None = None,
            wait_for_memory_s: float = 0) -> dict:
    """Save a quantised copy of the profile's model, once.

    Costs the full-precision peak one last time, so it takes the machine
    lock and the memory check like any job. Then generates one image at the
    profile's size so the prepared copy's real peak is measured, not guessed."""
    import json
    import shutil

    require()
    if profile.quantize is None:
        raise ImageError(f"{profile.name!r} is not quantised; nothing to prepare.")
    if prepared_info(profile) is not None:
        return {"profile": profile.name, "path": str(_prepared_dir(profile)),
                "already": True, **prepared_info(profile)}

    final = _prepared_dir(profile)
    partial = final.with_name(final.name + ".partial")
    with SCHEDULER.slot("prepare"), MACHINE_LOCK:
        ensure_memory(profile, config, wait_s=wait_for_memory_s)

        def work():
            import importlib

            from mflux.models.common.resolution.config_resolution import ConfigResolution

            RESIDENT.unload()
            module_path, class_name = VARIANTS[profile.variant]
            cls = getattr(importlib.import_module(module_path), class_name)
            t = time.time()
            model = cls(model_config=ConfigResolution.resolve(profile.model),
                        quantize=profile.quantize)
            shutil.rmtree(partial, ignore_errors=True)
            partial.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(partial))
            del model
            mx = _mx()
            if mx is not None:
                mx.clear_cache()
            shutil.rmtree(final, ignore_errors=True)
            partial.rename(final)
            (final / "sidecar.json").write_text(json.dumps({
                "profile": profile.name, "model": profile.model,
                "quantize": profile.quantize, "mflux": mflux_version(),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "save_s": round(time.time() - t, 1), "peak_gib": None,
            }, indent=2), encoding="utf-8")

        try:
            _on_mlx_thread(work)
        except Exception as e:
            shutil.rmtree(partial, ignore_errors=True)
            if is_gpu_fault(e):
                _on_fault(e)
                raise EngineFault("The GPU failed while preparing; retry in a minute.") from e
            raise

    size = sum(f.stat().st_size for f in final.rglob("*") if f.is_file())
    logger.info("prepared %s at %s (%.1f GiB)", profile.name, final, size / GIB)
    return {"profile": profile.name, "path": str(final), "already": False,
            "size_gib": round(size / GIB, 1)}


# ── memory right now, faults, and the MLX thread ──────────────────────────


def _mlx_held_gib() -> float:
    """What this process's MLX already holds, which loading frees first."""
    mx = _mx()
    if mx is None or RESIDENT.model is None:
        return 0.0
    try:
        return (mx.get_active_memory() + mx.get_cache_memory()) / GIB
    except Exception:
        return 0.0


def _unload_ollama(config: Config | None) -> list[str]:
    """Ask Ollama to drop every model it holds. Returns the names asked.

    keep_alive=0 unloads once any in-flight request finishes, so a chat that
    is mid-answer completes; the model reloads on its next use."""
    import httpx

    from .. import config as config_mod
    from .. import hardware

    cfg = config or config_mod.load()
    names = sorted(hardware.loaded_models(cfg))
    for name in names:
        try:
            httpx.post(f"{cfg.ollama_host}/api/generate",
                       json={"model": name, "keep_alive": 0}, timeout=5.0)
        except Exception:
            logger.debug("could not unload %s", name, exc_info=True)
    return names


def ensure_memory(profile: Profile, config: Config | None = None,
                  wait_s: float = 0) -> dict:
    """Refuse to start a job the machine cannot hold *right now*.

    guard() asks whether the machine could ever hold the profile. This asks
    whether it can at this moment, with whatever else is running. They differ
    by everything else on the laptop: zimage-negprompt passed guard() as
    "tight" with ~33 GiB usable, then loaded 27 GiB beside a browser, Ollama
    and Docker, and the machine panicked from memory starvation.

    In order: count what this process already holds (it is freed before
    loading), ask Ollama to unload its models if that is not enough, wait
    briefly for the memory to come back, then keep checking for up to
    `wait_s` (someone closing a browser), then refuse with MemoryBusy.

    Waiting happens with the machine lock held, so other image jobs queue
    behind it. That is deliberate: they need memory too, and letting a
    small job in first would only move the goalposts for the big one."""
    from .. import hardware

    if _fault_remaining() > 0:
        raise MemoryBusy(
            f"Recovering from a GPU fault; image generation resumes in "
            f"{int(_fault_remaining())}s.")

    held = _mlx_held_gib()
    need = max(0.0, profile.peak_gib - held) + SAFETY_GIB
    have = hardware.available_now_gib()
    if have is None or have >= need:
        return {"available_gib": have, "needed_gib": round(need, 1)}

    asked = _unload_ollama(config)
    if asked:
        logger.info("freeing memory for %s: asked Ollama to unload %s",
                    profile.name, ", ".join(asked))
        deadline = time.time() + FREE_WAIT_S
        while time.time() < deadline:
            time.sleep(1.0)
            have = hardware.available_now_gib() or 0.0
            if have >= need:
                return {"available_gib": have, "needed_gib": round(need, 1),
                        "unloaded": asked}

    if wait_s > 0:
        logger.warning(
            "waiting up to %.0fs for memory for %s: needs ~%.0f GiB, %.0f GiB free. "
            "Close an app to let it start.", wait_s, profile.name, need, have or 0)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(WAIT_POLL_S)
            have = hardware.available_now_gib() or 0.0
            if have >= need:
                logger.info("memory freed; starting %s", profile.name)
                return {"available_gib": have, "needed_gib": round(need, 1),
                        "waited": True}

    fits = ", ".join(n for n in _fitting_profiles()
                     if n != profile.name) or "none"
    raise MemoryBusy(
        f"Not enough free memory right now for {profile.name!r}: it needs about "
        f"{need:.0f} GiB and {have:.0f} GiB is free. Starting it anyway is what "
        f"makes the Mac swap and crash. Close something and retry, or use a "
        f"lighter profile ({fits}).")


def _fault_remaining() -> float:
    try:
        return max(0.0, FAULT_COOLDOWN_S - (time.time() - FAULT_FILE.stat().st_mtime))
    except OSError:
        return 0.0


def is_gpu_fault(e: BaseException) -> bool:
    """Errors after which this process's MLX state cannot be trusted."""
    msg = str(e)
    return ("Command buffer execution failed" in msg
            or "kIOGPUCommandBuffer" in msg
            or "There is no Stream(" in msg)


def _on_fault(e: BaseException) -> None:
    """Drop the model, start the cooldown, and restart if launchd will
    bring us back.

    A Metal command-buffer timeout leaves MLX's streams unusable, and every
    later job in the same process failed. A fresh process is the only clean
    state, so under the service (launchd, KeepAlive) the daemon exits a
    moment after answering. Run by hand, it stays up and refuses images
    until the cooldown passes."""
    logger.error("GPU fault during image generation: %s", e)
    try:
        RESIDENT.unload()
    except Exception:
        pass
    try:
        FAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        FAULT_FILE.write_text(f"{time.time()}\n{str(e)[:300]}\n", encoding="utf-8")
    except OSError:
        pass
    if os.environ.get("XPC_SERVICE_NAME", "").startswith("com.llm-sidecar"):
        logger.error("restarting the sidecar to recover the GPU (launchd restarts it)")
        threading.Timer(2.0, os._exit, args=(75,)).start()


# Every MLX call on one thread. MLX streams are per-thread: a model built on
# one of the server's worker threads and then used from another fails with
# "There is no Stream(cpu, N) in current thread". 35 image jobs failed that
# way here before this existed.
_MLX_EXECUTOR = None
_MLX_EXECUTOR_LOCK = threading.Lock()


def _on_mlx_thread(fn):
    global _MLX_EXECUTOR
    with _MLX_EXECUTOR_LOCK:
        if _MLX_EXECUTOR is None:
            from concurrent.futures import ThreadPoolExecutor
            _MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
    return _MLX_EXECUTOR.submit(fn).result()


def run(profile: Profile, *, prompt: str, negative: str | None, seed: int,
        width: int, height: int, steps: int, refs: list[Path],
        resolution: int | None, out: Path, config: Config | None = None,
        wait_for_memory_s: float = 0) -> dict:
    """One image, start to finish. Returns timing and the path written."""
    import dataclasses

    require()
    peak = effective_peak(profile)
    if peak != profile.peak_gib:
        # A prepared copy: the memory check, the guard and the cache cap
        # should all use what it actually peaks at.
        profile = dataclasses.replace(profile, peak_gib=peak)
    guard(profile)

    with SCHEDULER.slot(RESIDENT.key_for(profile)), MACHINE_LOCK:
        # Inside the lock: memory is measured when this job is next, not
        # when it joined the queue behind three others.
        ensure_memory(profile, config, wait_s=wait_for_memory_s)
        with RESIDENT._lock:
            RESIDENT.in_use += 1

        def work():
            model, load_s, warm = RESIDENT.get(profile)
            reset_peak()
            t = time.time()
            image = _call(model, profile, prompt=prompt, negative=negative, seed=seed,
                          width=width, height=height, steps=steps, refs=refs,
                          resolution=resolution)
            generate_s = round(time.time() - t, 2)
            out.parent.mkdir(parents=True, exist_ok=True)
            image.save(path=str(out))
            retune_mlx_cache(profile)
            _record_peak(profile)
            return load_s, warm, generate_s

        try:
            load_s, warm, generate_s = _on_mlx_thread(work)
        except Exception as e:
            if is_gpu_fault(e):
                _on_fault(e)
                raise EngineFault(
                    "The GPU failed during this image (usually memory pressure). "
                    "The image engine is restarting; retry in about a minute."
                ) from e
            raise
        finally:
            with RESIDENT._lock:
                RESIDENT.in_use -= 1
                RESIDENT.last_used = time.time()
                RESIDENT.trimmed = False
        RESIDENT.start_reaper()

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
