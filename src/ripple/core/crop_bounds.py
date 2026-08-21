"""General-purpose utilities for determining and applying a crop region from a mask."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np

if TYPE_CHECKING:
    import torch

CropMode = Literal["none", "tight", "nice_size", "fixed_size"]


class CropBounds(TypedDict):
    """Inclusive pixel bounds of a crop region, in row/column coordinates."""

    min_y: int
    max_y: int
    min_x: int
    max_x: int


def get_crop_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(min_y, max_y, min_x, max_x)`` bounding box of True pixels.

    Interior False pixels (e.g. a defect region enclosed within an otherwise usable
    area) never shrink this box -- it is defined purely by the outermost extent of
    True pixels, so any such "holes" remain inside the returned bounds.

    Parameters
    ----------
    mask : np.ndarray
        Boolean 2-D array.

    Returns
    -------
    tuple[int, int, int, int]
        Tight bounding box in row/column coordinates.

    Raises
    ------
    ValueError
        If ``mask`` contains no True pixels.
    """
    rows: np.ndarray = np.any(mask, axis=1)
    cols: np.ndarray = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        raise ValueError("Mask contains no True pixels; cannot compute crop bounds.")

    min_y = int(np.argmax(rows))
    max_y = int(len(rows) - 1 - np.argmax(rows[::-1]))
    min_x = int(np.argmax(cols))
    max_x = int(len(cols) - 1 - np.argmax(cols[::-1]))

    return min_y, max_y, min_x, max_x


# ---------------------------------------------------------------------------
# Size-policy helpers
# ---------------------------------------------------------------------------


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round `value` up to the nearest multiple of `multiple` (no-op if multiple<=1)."""
    if multiple <= 1:
        return value
    remainder = value % multiple
    return value if remainder == 0 else value + (multiple - remainder)


def _center_span(
    lo: int, hi: int, target_len: int, valid_lo: int, valid_hi: int
) -> tuple[int, int]:
    """Grow the span ``[lo, hi]`` to `target_len`, centered, clamped to valid range.

    `target_len` is always >= the current span length, so this only grows.
    """
    current_len = hi - lo + 1
    delta = target_len - current_len
    delta_before = delta // 2
    delta_after = delta - delta_before

    new_lo = lo - delta_before
    new_hi = hi + delta_after

    if new_lo < valid_lo:
        shift = valid_lo - new_lo
        new_lo += shift
        new_hi += shift
    if new_hi > valid_hi:
        shift = new_hi - valid_hi
        new_lo -= shift
        new_hi -= shift

    new_lo = max(new_lo, valid_lo)
    new_hi = min(new_hi, valid_hi)
    return new_lo, new_hi


def _apply_round_to(
    bounds: tuple[int, int, int, int],
    round_to: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Grow `bounds` to a multiple of `round_to` per side, clamped to `image_shape`."""
    min_y, max_y, min_x, max_x = bounds
    height_img, width_img = image_shape
    height = max_y - min_y + 1
    width = max_x - min_x + 1

    target_h = _round_up_to_multiple(height, round_to)
    target_w = _round_up_to_multiple(width, round_to)

    new_min_y, new_max_y = _center_span(min_y, max_y, target_h, 0, height_img - 1)
    new_min_x, new_max_x = _center_span(min_x, max_x, target_w, 0, width_img - 1)
    return new_min_y, new_max_y, new_min_x, new_max_x


def _apply_target_shape(
    bounds: tuple[int, int, int, int],
    target_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Grow `bounds` to a centered window of exactly `target_shape`."""
    height_img, width_img = image_shape
    target_h, target_w = target_shape
    if target_h > height_img or target_w > width_img:
        raise ValueError(
            f"target_shape {target_shape} exceeds the frame size "
            f"{(height_img, width_img)}."
        )

    min_y, max_y, min_x, max_x = bounds
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    if target_h < height or target_w < width:
        raise ValueError(
            f"target_shape {target_shape} is smaller than the mask's tight "
            f"bounding box {(height, width)}; cannot crop to target_shape without "
            "cutting off part of the mask."
        )

    new_min_y, new_max_y = _center_span(min_y, max_y, target_h, 0, height_img - 1)
    new_min_x, new_max_x = _center_span(min_x, max_x, target_w, 0, width_img - 1)
    return new_min_y, new_max_y, new_min_x, new_max_x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def determine_crop_bounds(
    mask: np.ndarray,
    mode: CropMode = "tight",
    round_to: int = 1,
    target_shape: tuple[int, int] | None = None,
    divisible_by: int = 1,
) -> CropBounds:
    """Determine crop bounds for `mask` under the requested size policy.

    Bounds are always derived by growing outward from the tight bounding box of
    `mask` (see :func:`get_crop_bounds`) -- they never shrink. If `mask` has
    interior False regions (e.g. a defect area enclosed within a beam disk), those
    holes remain inside the crop; downstream masking (e.g. Poisson noise-fill) is
    responsible for handling them, not the crop bounds.

    Parameters
    ----------
    mask : np.ndarray
        Boolean 2-D array.
    mode : Literal["none", "tight", "nice_size", "fixed_size"]
        Crop size policy:

        - ``"none"``: no cropping; bounds span the full `mask` shape.
        - ``"tight"``: smallest bounding box containing `mask`.
        - ``"nice_size"``: as ``"tight"``, then grown so each side is a multiple
          of `round_to`.
        - ``"fixed_size"``: a window of exactly `target_shape`, centered on the
          mask's tight bounding box.
    round_to : int
        Multiple to round crop side lengths to. Only used when ``mode="nice_size"``.
        Default is 1 (no-op).
    target_shape : tuple[int, int] | None
        ``(height, width)`` of the desired crop window. Required when
        ``mode="fixed_size"``, otherwise ignored.
    divisible_by : int
        Side lengths are constrained to be an exact multiple of this value. Default is 1
        (no constraint). Ignored when ``mode="none"``.

    Returns
    -------
    CropBounds
        Inclusive ``(min_y, max_y, min_x, max_x)`` crop bounds, guaranteed to have side
        lengths that are exact multiples of `divisible_by` (except in ``mode="none"``).

    Raises
    ------
    ValueError
        If `mode` is ``"fixed_size"`` and `target_shape` is None, doesn't fit, or
        isn't an exact multiple of `divisible_by`; if `mask` has no True pixels;
        or if `mode` is not a recognised value.
    """
    height, width = mask.shape

    if mode == "none":
        return CropBounds(min_y=0, max_y=height - 1, min_x=0, max_x=width - 1)

    bounds = get_crop_bounds(mask)

    if mode == "tight":
        if divisible_by > 1:
            bounds = _apply_round_to(bounds, divisible_by, (height, width))
    elif mode == "nice_size":
        effective_round_to = math.lcm(round_to, divisible_by)
        bounds = _apply_round_to(bounds, effective_round_to, (height, width))
    elif mode == "fixed_size":
        if target_shape is None:
            raise ValueError("target_shape must be set when mode='fixed_size'.")
        if divisible_by > 1 and (
            target_shape[0] % divisible_by != 0 or target_shape[1] % divisible_by != 0
        ):
            raise ValueError(
                f"target_shape {target_shape} is not an exact multiple of "
                f"divisible_by ({divisible_by}) in both dimensions."
            )
        bounds = _apply_target_shape(bounds, target_shape, (height, width))
    else:
        raise ValueError(f"Unknown crop mode '{mode}'.")

    min_y, max_y, min_x, max_x = bounds
    return CropBounds(min_y=min_y, max_y=max_y, min_x=min_x, max_x=max_x)


def crop_movie(
    movie: torch.Tensor, min_y: int, max_y: int, min_x: int, max_x: int
) -> torch.Tensor:
    """Crop the trailing (height, width) dimensions of `movie` to inclusive bounds.

    Parameters
    ----------
    movie : torch.Tensor
        Tensor with shape ``(..., height, width)``, e.g. ``(n_frames, height, width)``
        or ``(height, width)``.
    min_y, max_y, min_x, max_x : int
        Inclusive crop bounds, e.g. from :func:`determine_crop_bounds`.

    Returns
    -------
    torch.Tensor
        A view into `movie` cropped to the requested region.
    """
    return movie[..., min_y : max_y + 1, min_x : max_x + 1]
