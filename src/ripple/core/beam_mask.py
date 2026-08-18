"""Core algorithm for estimating a DeCo-LACE beam mask from a raw movie."""

from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np
import torch
from skimage.measure import label, regionprops
from torch_fourier_filter.bandpass import bandpass_filter

from ripple.core.prepare_movie import DEFAULT_PREP_CHUNK_SIZE


class BeamMaskParams(TypedDict):
    """Fitted ellipse and crop-bound parameters returned by `estimate_beam_mask`."""

    center_y: float
    center_x: float
    axis1: float
    axis2: float
    angle_deg: float
    diameter_reduction: float
    image_shape_y: int
    image_shape_x: int
    crop_min_y: int
    crop_max_y: int
    crop_min_x: int
    crop_max_x: int
    threshold_method: str
    pixel_size: float


# ---------------------------------------------------------------------------
# Frame summation
# ---------------------------------------------------------------------------


def sum_movie_chunked(
    movie: torch.Tensor,
    device: torch.device,
    chunk_size: int = DEFAULT_PREP_CHUNK_SIZE,
) -> torch.Tensor:
    """Sum all frames of a movie, transferring `chunk_size` frames to device at once.

    Parameters
    ----------
    movie : torch.Tensor
        Movie tensor with shape ``(n_frames, height, width)``, on any device.
    device : torch.device
        Device each chunk is transferred to and summed on.
    chunk_size : int
        Number of frames to transfer/sum at a time.

    Returns
    -------
    torch.Tensor
        2-D float32 tensor of shape ``(height, width)`` on `device`, equal to the
        sum of all frames.
    """
    n_frames = movie.shape[0]
    chunk_size = max(1, min(chunk_size, n_frames))

    frame_sum = torch.zeros(movie.shape[-2:], dtype=torch.float32, device=device)
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = movie[start:end].to(device=device, dtype=torch.float32)
        frame_sum += chunk.sum(dim=0)

    return frame_sum


# ---------------------------------------------------------------------------
# Algorithm steps
# ---------------------------------------------------------------------------


def low_pass_filter(
    image: torch.Tensor,
    pixel_size: float,
    low_pass_resolution: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Apply a Fourier low-pass filter to a 2-D image.

    Parameters
    ----------
    image : torch.Tensor
        2-D float tensor to filter.
    pixel_size : float
        Pixel size in Angstroms per pixel.
    low_pass_resolution : float
        Low-pass cutoff resolution in Angstroms (larger = more blurring).
    device : torch.device | str | None
        Device the filtering runs on. If None, uses `image`'s current device.

    Returns
    -------
    torch.Tensor
        Filtered image as a float32 2-D tensor, on `device`.
    """
    cutoff = 0.5 * pixel_size / low_pass_resolution
    if device is not None:
        image = image.to(device=device, dtype=torch.float32)
    dft = torch.fft.rfft2(image)
    bp = bandpass_filter(
        low=0.0,
        high=float(cutoff),
        falloff=0.01,
        image_shape=(image.shape[0], image.shape[1]),
        rfft=True,
        fftshift=False,
    ).to(device=dft.device)
    filtered: torch.Tensor = torch.fft.irfft2(bp * dft)

    return filtered


def threshold_otsu(image: np.ndarray) -> float:
    """Compute a threshold using Otsu's method (minimizes intra-class variance).

    Parameters
    ----------
    image : np.ndarray
        2-D or N-D float array.

    Returns
    -------
    float
        Optimal threshold value.
    """
    hist, bin_edges = np.histogram(image.flatten(), bins=256, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    hist = hist / float(hist.sum())

    best_threshold = float(bin_centers[0])
    min_variance = float("inf")

    for i in range(1, len(bin_centers) - 1):
        w0 = float(hist[:i].sum())
        w1 = float(hist[i:].sum())
        if w0 == 0.0 or w1 == 0.0:
            continue
        mu0 = float((hist[:i] * bin_centers[:i]).sum()) / w0
        mu1 = float((hist[i:] * bin_centers[i:]).sum()) / w1
        var0 = float((hist[:i] * (bin_centers[:i] - mu0) ** 2).sum()) / w0
        var1 = float((hist[i:] * (bin_centers[i:] - mu1) ** 2).sum()) / w1
        variance = w0 * var0 + w1 * var1
        if variance < min_variance:
            min_variance = variance
            best_threshold = float(bin_centers[i])

    return best_threshold


def fit_ellipse(binary: np.ndarray) -> tuple[float, float, float, float, float]:
    """Fit an ellipse to the largest connected component of a binary image.

    Falls back to scikit-image region-moment fitting when fewer than 5 interior
    contour points remain (beam too severely clipped to recover arc geometry).

    Parameters
    ----------
    binary : np.ndarray
        Boolean 2-D array.

    Returns
    -------
    tuple[float, float, float, float, float]
        ``(center_y, center_x, axis1, axis2, angle_deg)`` where ``axis1`` is
        the semi-axis radius along ``angle_deg`` (cv2 ``axes[0]/2``), ``axis2``
        is the semi-axis perpendicular to it (cv2 ``axes[1]/2``), and
        ``angle_deg`` is the angle in degrees (clockwise from horizontal on
        screen).

    Raises
    ------
    ValueError
        If the binary image has no foreground pixels.
    """
    labeled: np.ndarray = label(binary)
    props = regionprops(labeled)
    if not props:
        raise ValueError("No connected components found in the binary image.")

    largest = max(props, key=lambda r: r.area)

    H, W = binary.shape
    region_mask = (labeled == largest.label).astype(np.uint8) * 255

    # Full-pixel contour (CHAIN_APPROX_NONE) gives a dense arc for a better fit.
    contours, _ = cv2.findContours(
        region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cnt = max(contours, key=len)
    # cv2 contour points: (N, 1, 2) with last dim = (col, row) = (x, y)
    pts = cnt.reshape(-1, 2).astype(np.float64)  # shape (N, 2): (col, row)

    # Drop points on and near the image boundary -- they trace the straight clipped
    # edge, not the real ellipse arc, and would pull the fit toward the border.
    border_margin = 16  # pixels
    x_col, y_row = pts[:, 0], pts[:, 1]
    interior = (
        (x_col > border_margin)
        & (x_col < W - border_margin)
        & (y_row > border_margin)
        & (y_row < H - border_margin)
    )
    interior_pts = pts[interior]

    # Fallback: region moments from scikit-image (only for extreme clipping).
    if len(interior_pts) < 5:
        cy, cx = largest.centroid
        # Convert regionprops orientation (CCW from row axis, radians) to cv2 angle
        # convention (CW from horizontal on screen, degrees): angle = 90 - degrees(ori).
        fallback_angle_deg = float(np.degrees(np.pi / 2.0 - largest.orientation))
        return (
            float(cy),
            float(cx),
            float(largest.axis_major_length) / 2.0,
            float(largest.axis_minor_length) / 2.0,
            fallback_angle_deg,
        )

    (cx, cy), (ax1, ax2), angle_deg = cv2.fitEllipse(
        interior_pts.astype(np.float32).reshape(-1, 1, 2)
    )

    return float(cy), float(cx), float(ax1) / 2.0, float(ax2) / 2.0, float(angle_deg)


def get_crop_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(min_y, max_y, min_x, max_x)`` bounding box of True pixels.

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


def make_ellipse_mask(
    shape: tuple[int, int],
    center_y: float,
    center_x: float,
    axis1: float,
    axis2: float,
    angle_deg: float,
    diameter_reduction: float = 0.0,
) -> np.ndarray:
    """Create a boolean ellipse mask from fitted parameters.

    Angle and axis conventions match the output of ``cv2.fitEllipse`` directly:
    ``axis1`` is the semi-axis along the ``angle_deg`` direction and ``axis2``
    is the semi-axis perpendicular to it. ``angle_deg`` is measured clockwise
    from the horizontal as seen on screen (where image rows increase downward),
    which is the same as counter-clockwise from the x-axis in standard
    mathematical (y-up) coordinates.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(height, width)`` of the output mask in pixels.
    center_y : float
        Row coordinate of the ellipse centroid in pixels.
    center_x : float
        Column coordinate of the ellipse centroid in pixels.
    axis1 : float
        Semi-axis radius in pixels along the ``angle_deg`` direction (before
        diameter reduction).
    axis2 : float
        Semi-axis radius in pixels perpendicular to ``angle_deg`` (before
        diameter reduction).
    angle_deg : float
        Angle of ``axis1`` from horizontal in degrees, clockwise on screen
        (cv2.fitEllipse convention).
    diameter_reduction : float
        Fractional shrink applied to both radii. ``0.0`` = no change;
        ``0.1`` reduces each axis by 10 %.

    Returns
    -------
    np.ndarray
        Boolean array of ``shape`` where ``True`` marks pixels inside the
        ellipse.

    Raises
    ------
    ValueError
        If ``diameter_reduction`` reduces either axis to ``<= 0``.
    """
    semi1 = axis1 * (1.0 - diameter_reduction)
    semi2 = axis2 * (1.0 - diameter_reduction)
    if semi1 <= 0.0 or semi2 <= 0.0:
        raise ValueError(
            f"diameter_reduction={diameter_reduction} reduces an ellipse axis to <= 0 "
            f"(axis1={axis1}, axis2={axis2})."
        )

    h, w = shape
    row_idx, col_idx = np.mgrid[:h, :w]
    dy: np.ndarray = row_idx.astype(np.float64) - center_y
    dx: np.ndarray = col_idx.astype(np.float64) - center_x

    theta = np.radians(angle_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    # In screen coordinates (x right, y down), a clockwise rotation of theta
    # from the horizontal maps the pixel offset (dx, dy) onto the two ellipse
    # axes as follows:
    proj1: np.ndarray = dx * cos_t + dy * sin_t
    proj2: np.ndarray = -dx * sin_t + dy * cos_t

    inside: np.ndarray = (proj1 / semi1) ** 2 + (proj2 / semi2) ** 2 <= 1.0
    return inside


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def estimate_beam_mask(
    frame_sum: torch.Tensor,
    pixel_size: float,
    threshold_method: str,
    diameter_reduction: float,
    low_pass_resolution: float,
    device: torch.device | str | None = None,
) -> BeamMaskParams:
    """Estimate a DeCo-LACE beam mask ellipse from a raw frame sum.

    Low-pass filters `frame_sum` (on `device`), thresholds it with Otsu's method,
    fits an ellipse to the largest connected component, and computes tight crop
    bounds for the (possibly shrunk) ellipse mask.

    Parameters
    ----------
    frame_sum : torch.Tensor
        2-D tensor, sum of all raw movie frames (e.g. from :func:`sum_movie_chunked`).
    pixel_size : float
        Pixel size in Angstroms per pixel.
    threshold_method : str
        Thresholding algorithm; currently only ``"otsu"`` is supported.
    diameter_reduction : float
        Fractional shrink applied to both ellipse radii before computing crop
        bounds and the returned mask parameters.
    low_pass_resolution : float
        Low-pass filter cutoff resolution in Angstroms, applied to `frame_sum`
        before thresholding.
    device : torch.device | str | None
        Device the low-pass filter runs on. If None, uses `frame_sum`'s current
        device.

    Returns
    -------
    BeamMaskParams
        Fitted ellipse and crop-bound parameters.

    Raises
    ------
    ValueError
        If `threshold_method` is not ``"otsu"``.
    """
    if threshold_method != "otsu":
        raise ValueError(
            f"Unsupported threshold_method '{threshold_method}'. Only 'otsu' is "
            "currently supported."
        )

    filtered = low_pass_filter(
        frame_sum, pixel_size, low_pass_resolution, device=device
    )
    filtered_np = filtered.detach().cpu().numpy()

    threshold = threshold_otsu(filtered_np)
    binary = filtered_np > threshold

    center_y, center_x, axis1, axis2, angle_deg = fit_ellipse(binary)

    h, w = int(binary.shape[0]), int(binary.shape[1])

    ellipse_mask = make_ellipse_mask(
        shape=(h, w),
        center_y=center_y,
        center_x=center_x,
        axis1=axis1,
        axis2=axis2,
        angle_deg=angle_deg,
        diameter_reduction=diameter_reduction,
    )

    min_y, max_y, min_x, max_x = get_crop_bounds(ellipse_mask)

    return {
        "center_y": center_y,
        "center_x": center_x,
        "axis1": axis1,
        "axis2": axis2,
        "angle_deg": angle_deg,
        "diameter_reduction": diameter_reduction,
        "image_shape_y": h,
        "image_shape_x": w,
        "crop_min_y": min_y,
        "crop_max_y": max_y,
        "crop_min_x": min_x,
        "crop_max_x": max_x,
        "threshold_method": threshold_method,
        "pixel_size": pixel_size,
    }
