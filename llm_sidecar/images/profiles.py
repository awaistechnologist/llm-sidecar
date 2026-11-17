"""Profiles — what a project asks for instead of naming a model.

Five repos on this machine each hardcoded their own model, steps and
quantisation, and each drifted separately: one was still on q4 (which puts
fake signatures in klein output), one borrowed another repo's venv, and two
had the same model under different names. Asking for `draft` instead of
`mflux-generate-z-image-turbo --quantize 8 --steps 8` means the decision is
made once, here, and upgrading it upgrades every caller.

A profile also carries the facts that get forgotten at the call site: what
the licence permits, whether the weights are gated, whether a negative
prompt does anything, how many reference images the model accepts, and how
much memory it needs at peak. The last two are load-bearing — a profile that
needs 31 GiB on a 36 GiB machine cannot run beside anything else, and
`max_refs` is the difference between character consistency and a different
face every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import SidecarError


class ImageError(SidecarError):
    """Base for every error the image subsystem raises deliberately."""


class ProfileError(ImageError):
    """Unknown profile, or one refused for this request."""


class LicenceError(ImageError):
    """The profile's weights do not permit this use.

    Separate from ProfileError because it is not a mistake — the caller
    asked for something real and the answer is no. andalus and
    social-media-poster may ship commercially, and FLUX.1-dev may not."""


@dataclass(frozen=True)
class Profile:
    name: str
    # mflux model key, exactly as AVAILABLE_MODELS spells it. Verified
    # against the installed mflux rather than guessed — several of these
    # differ from the obvious name ("dev-kontext", not "kontext").
    model: str
    # The mflux class to construct. Deliberately separate from `model`:
    # flux2-klein-4b is one set of weights driven by two different classes,
    # Flux2Klein for text-to-image and Flux2KleinEdit for reference-guided
    # editing, and the profile is what chooses between them.
    variant: str
    steps: int
    quantize: int | None = 8
    width: int = 1024
    height: int = 1024
    guidance: float | None = None
    # Reference images for character/object consistency. 0 means the model
    # takes none, so asking for consistency here is a silent no-op.
    max_refs: int = 0
    # A negative prompt is *accepted* by several models and *ignored* by
    # most. Recorded so a caller is told rather than quietly disappointed.
    supports_negative: bool = False
    licence: str = "unknown"
    commercial_ok: bool = False
    # Weights that need a Hugging Face token and an accepted agreement.
    gated: bool = False
    # Peak resident memory, GiB, approximate. Checked against this machine
    # before loading — see engine.guard.
    peak_gib: float = 16.0
    # Set by `bench`, not guessed.
    measured_s: float | None = None
    note: str = ""
    # An editing profile needs an input image; a text-to-image one must not
    # be handed one silently.
    requires_refs: bool = False
    post: str = ""   # optional post-process step, see engine._post


PROFILES: dict[str, Profile] = {p.name: p for p in [
    Profile(
        name="draft", model="z-image-turbo", variant="z-image-turbo", steps=8, quantize=8,
        width=1024, height=768, supports_negative=True,
        licence="Apache-2.0", commercial_ok=True, gated=False, peak_gib=31.0,
        note="Ungated and commercially clear, with a negative prompt that "
             "actually works. The default for bulk work.",
    ),
    Profile(
        name="fast", model="schnell", variant="flux1", steps=4, quantize=8,
        licence="Apache-2.0", commercial_ok=True, gated=True, peak_gib=12.0,
        note="FLUX.1-schnell. Gated: needs a Hugging Face token. Accepts a "
             "negative prompt and ignores it.",
    ),
    Profile(
        name="cinematic", model="dev", variant="flux1", steps=20, quantize=8,
        licence="FLUX.1-dev non-commercial", commercial_ok=False, gated=True,
        peak_gib=14.0,
        note="Best-looking of the FLUX.1 line and non-commercial. Refused "
             "unless the request says commercial=False.",
    ),
    Profile(
        name="characters-txt2img", model="flux2-klein-4b", variant="klein", steps=4,
        quantize=8, guidance=1.0,
        licence="Apache-2.0", commercial_ok=True, peak_gib=10.0,
        note="Character sheets. Never q4 or q5 — lower quantisations put "
             "fake signatures in the output.",
    ),
    Profile(
        name="characters-consistent", model="flux2-klein-4b", variant="klein-edit", steps=4,
        quantize=8, guidance=1.0, max_refs=3, requires_refs=True,
        licence="Apache-2.0", commercial_ok=True, peak_gib=10.0,
        note="The same character across frames, from up to 3 reference "
             "images. The one capability no other profile has.",
    ),
    Profile(
        name="kontext-edit", model="dev-kontext", variant="kontext", steps=20, quantize=8,
        max_refs=1, requires_refs=True,
        licence="FLUX.1-Kontext-dev non-commercial", commercial_ok=False,
        gated=True, peak_gib=14.0,
        note="Continuity edits to an existing frame. Non-commercial.",
    ),
    Profile(
        name="lineart", model="flux2-klein-4b", variant="klein-edit", steps=4, quantize=8,
        guidance=1.0, max_refs=3, requires_refs=True, post="lineart",
        licence="Apache-2.0", commercial_ok=True, peak_gib=10.0,
        note="Colouring pages: klein edit, then threshold and dilate.",
    ),
    Profile(
        name="upscale", model="seedvr2-3b", variant="seedvr2", steps=0, quantize=None,
        max_refs=1, requires_refs=True, peak_gib=20.0,
        licence="Apache-2.0", commercial_ok=True,
        note="SeedVR2. ~132s to 2400px on an M3 Max.",
    ),
]}

DEFAULT_PROFILE = "draft"


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ProfileError(
            f"Unknown profile {name!r}. Available: {', '.join(sorted(PROFILES))}."
        ) from None


def check(profile: Profile, *, commercial: bool, refs: list) -> None:
    """Refuse a request the profile cannot honour, before loading 10 GiB.

    Every one of these is a failure the caller would otherwise discover
    minutes later, or — worse, in the licence case — not at all."""
    if commercial and not profile.commercial_ok:
        raise LicenceError(
            f"Profile {profile.name!r} uses {profile.licence} weights, which do "
            f"not permit commercial use. Pass commercial=False if this output "
            f"is not commercial, or use a profile that is: "
            f"{', '.join(n for n, p in PROFILES.items() if p.commercial_ok)}."
        )
    if profile.requires_refs and not refs:
        raise ProfileError(
            f"Profile {profile.name!r} edits an existing image and needs at "
            f"least one reference. Use "
            f"{'characters-txt2img' if 'character' in profile.name else 'draft'} "
            f"to generate from text alone."
        )
    if len(refs) > profile.max_refs:
        raise ProfileError(
            f"Profile {profile.name!r} takes at most {profile.max_refs} "
            f"reference image(s); {len(refs)} were given."
        )


def describe() -> list[dict]:
    """The profile table, for `llm-sidecar image profiles` and the API."""
    return [
        {
            "name": p.name, "model": p.model, "variant": p.variant,
            "steps": p.steps,
            "quantize": p.quantize, "size": f"{p.width}x{p.height}",
            "max_refs": p.max_refs, "supports_negative": p.supports_negative,
            "licence": p.licence, "commercial_ok": p.commercial_ok,
            "gated": p.gated, "peak_gib": p.peak_gib,
            "measured_s": p.measured_s, "note": p.note,
        }
        for p in PROFILES.values()
    ]
