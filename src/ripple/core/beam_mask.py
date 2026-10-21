"""Core algorithm for estimating a DeCo-LACE beam mask from a raw movie."""

from __future__ import annotations

from typing import TypedDict

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import label, regionprops
from torch_fourier_filter.bandpass import bandpass_filter

from ripple.core.crop_bounds import get_crop_bounds
from ripple.core.prepare_movie import DEFAULT_PREP_CHUNK_SIZE

Ellipse = tuple[float, float, float, float, float]  # cy, cx, axis1, axis2, angle_deg

# Number of rays cast from the beam center for the Sobel edge-ray fit
DEFAULT_N_RAYS = 720


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
    n_edge_points: int | None  # NOTE: Only non-None for "sobel" method
    n_inliers: int | None  # NOTE: Only non-None for "sobel" method


class BeamEdgeFit(TypedDict):
    """Ellipse and edge-point counts returned by `fit_beam_edge`."""

    center_y: float
    center_x: float
    axis1: float
    axis2: float
    angle_deg: float
    n_edge_points: int
    n_inliers: int


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
    # bandpass_filter takes cycles/pixel (Nyquist = 0.5).
    cutoff = pixel_size / low_pass_resolution
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
# Sobel edge-ray fit
# ---------------------------------------------------------------------------


def bin_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean a 2-D array by `factor`, cropping any remainder.

    Parameters
    ----------
    image : np.ndarray
        2-D array to bin.
    factor : int
        Binning factor along both axes. Values ``<= 1`` return an unbinned copy.

    Returns
    -------
    np.ndarray
        float32 array of shape ``(height // factor, width // factor)``.
    """
    if factor <= 1:
        return image.astype(np.float32)
    h = image.shape[0] // factor * factor
    w = image.shape[1] // factor * factor
    binned: np.ndarray = (
        image[:h, :w]
        .reshape(h // factor, factor, w // factor, factor)
        .mean((1, 3), dtype=np.float32)
    )
    return binned


def _background_stats(image: np.ndarray) -> tuple[float, float, float]:
    """Return ``(background, noise, beam_level)`` of a smoothed frame sum."""
    dark = image[image <= np.percentile(image, 10)]
    bg = float(np.median(dark))
    noise = float(1.4826 * np.median(np.abs(dark - bg))) + 1e-6
    return bg, noise, float(np.percentile(image, 95))


def _initial_center(image: np.ndarray, k: float = 8.0) -> tuple[float, float]:
    """Centroid ``(y, x)`` of the image saturated at background + `k` x noise."""
    bg, noise, _ = _background_stats(image)
    weight = np.clip((image - bg) / (k * noise), 0.0, 1.0)
    total = float(weight.sum())
    if total <= 0.0:
        return (image.shape[0] - 1) / 2.0, (image.shape[1] - 1) / 2.0
    yy, xx = np.indices(weight.shape)
    return float((yy * weight).sum()) / total, float((xx * weight).sum()) / total


def radial_edge_points(
    image: np.ndarray,
    center: tuple[float, float],
    n_rays: int = DEFAULT_N_RAYS,
    snr: float = 6.0,
    margin: int = 4,
    exit_level: float = 0.15,
) -> np.ndarray:
    """Find one beam-edge point per ray cast outward from `center`.

    Parameters
    ----------
    image : np.ndarray
        Smoothed 2-D frame sum, linear intensity.
    center : tuple[float, float]
        ``(y, x)`` origin of the rays, in pixels.
    n_rays : int
        Number of rays, evenly spaced in angle.
    snr : float
        Minimum edge strength, in units of the background gradient noise.
    margin : int
        Samples within `margin` pixels of the image border are ignored.
    exit_level : float
        Normalized intensity (0 = background, 1 = beam) above which a ray is considered
        to leave the detector while still inside the beam.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` array of edge points as ``(x, y)`` = ``(col, row)``, ``N <=
        n_rays``.
    """
    h, w = image.shape
    image = image.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    bg, _, beam_level = _background_stats(image)
    dark = image <= np.percentile(image, 10)
    grad_noise = float(np.std(np.r_[gx[dark], gy[dark]])) + 1e-9

    radii = np.arange(0.15 * min(h, w) / 2, np.hypot(h, w), 0.5)
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ux, uy = np.cos(theta)[:, None], np.sin(theta)[:, None]
    xs, ys = center[1] + ux * radii, center[0] + uy * radii
    inside = (
        (xs > margin) & (xs < w - 1 - margin) & (ys > margin) & (ys < h - 1 - margin)
    )

    # Bright-to-dark step along the outward direction, in noise units.
    drop = -(
        map_coordinates(gx, [ys, xs], order=1) * ux
        + map_coordinates(gy, [ys, xs], order=1) * uy
    )
    drop = np.where(inside, drop / grad_noise, 0.0)

    rows = np.arange(n_rays)
    peak = (drop > snr) & (drop >= np.roll(drop, 1, 1)) & (drop >= np.roll(drop, -1, 1))
    idx = drop.shape[1] - 1 - np.argmax(peak[:, ::-1], axis=1)  # outermost peak

    # In-image samples along a ray are contiguous since the frame is convex.
    last = np.maximum(inside.sum(axis=1) - 1, 0)
    exit_intensity = map_coordinates(image, [ys[rows, last], xs[rows, last]], order=1)
    exit_level_norm = (exit_intensity - bg) / max(beam_level - bg, 1e-9)

    keep = (drop[rows, idx] > snr) & (exit_level_norm < exit_level)
    points: np.ndarray = np.c_[xs[rows, idx], ys[rows, idx]][keep]
    return points


def _cv2_fit_ellipse(points_xy: np.ndarray) -> Ellipse:
    """``cv2.fitEllipse`` on ``(x, y)`` points, as ``(cy, cx, axis1, axis2, angle)``."""
    (cx, cy), (ax1, ax2), angle_deg = cv2.fitEllipse(
        points_xy.astype(np.float32).reshape(-1, 1, 2)
    )
    return float(cy), float(cx), float(ax1) / 2.0, float(ax2) / 2.0, float(angle_deg)


def _ellipse_residual(points_xy: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    """Approximate radial distance, in pixels, of each point from `ellipse`."""
    cy, cx, axis1, axis2, angle_deg = ellipse
    theta = np.radians(angle_deg)
    dx, dy = points_xy[:, 0] - cx, points_xy[:, 1] - cy
    proj1 = dx * np.cos(theta) + dy * np.sin(theta)
    proj2 = -dx * np.sin(theta) + dy * np.cos(theta)
    rho = np.sqrt((proj1 / axis1) ** 2 + (proj2 / axis2) ** 2)
    residual: np.ndarray = (rho - 1.0) * np.sqrt(axis1 * axis2)
    return residual


def _circle_from_3_points(points_xy: np.ndarray) -> tuple[float, float, float] | None:
    """Circle ``(x, y, r)`` through three points, or None if they are collinear."""
    (x1, y1), (x2, y2), (x3, y3) = points_xy
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    s1, s2, s3 = x1**2 + y1**2, x2**2 + y2**2, x3**2 + y3**2
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return float(ux), float(uy), float(np.hypot(x1 - ux, y1 - uy))


def robust_ellipse_fit(
    points_xy: np.ndarray,
    tol_frac: float = 0.02,
    n_iterations: int = 400,
    seed: int = 0,
) -> tuple[Ellipse, np.ndarray]:
    """Fit an ellipse to beam-edge points, rejecting non-rim outliers via RANSAC.

    Parameters
    ----------
    points_xy : np.ndarray
        ``(N, 2)`` candidate edge points as ``(x, y)``.
    tol_frac : float
        RANSAC inlier band as a fraction of the candidate circle radius.
    n_iterations : int
        Number of RANSAC iterations.
    seed : int
        Seed for the RANSAC sampler, so results are deterministic.

    Returns
    -------
    tuple[Ellipse, np.ndarray]
        ``(center_y, center_x, axis1, axis2, angle_deg)`` in the :func:`fit_ellipse`
        convention, and the boolean inlier mask over `points_xy`.

    Raises
    ------
    ValueError
        If fewer than 5 points are given or no consistent rim is found.
    """
    n_points = len(points_xy)
    if n_points < 5:
        raise ValueError(f"Only {n_points} beam-edge points found; need at least 5.")

    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    for _ in range(n_iterations):
        circle = _circle_from_3_points(
            points_xy[rng.choice(n_points, 3, replace=False)]
        )
        if circle is None:
            continue
        cx, cy, r = circle
        dist = np.hypot(points_xy[:, 0] - cx, points_xy[:, 1] - cy)
        inliers = np.abs(dist - r) < tol_frac * r
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 5:
        raise ValueError("No consistent beam rim found among the beam-edge points.")

    # NOTE: Converting median of residual statistics to MAD (median absolute deviation)
    #       under assumption of Gaussian noise. Taking 75th percentile yields
    #       sigma = 1.4826 * MAD.
    keep = best
    for _ in range(3):
        ellipse = _cv2_fit_ellipse(points_xy[keep])
        residual = _ellipse_residual(points_xy, ellipse)
        mad = 1.4826 * float(np.median(np.abs(residual[keep])))
        new_keep = np.abs(residual) < max(3.0 * mad, 1.0)
        if new_keep.sum() < 5:
            break
        keep = new_keep

    return ellipse, keep


def fit_beam_edge(
    frame_sum: np.ndarray,
    pixel_size: float,
    working_pixel_size: float = 8.0,
    smoothing_sigma: float = 120.0,
    n_rays: int = DEFAULT_N_RAYS,
) -> BeamEdgeFit:
    """Fit the beam ellipse to the outer Sobel edge of a raw frame sum.

    Parameters
    ----------
    frame_sum : np.ndarray
        2-D sum of all raw movie frames.
    pixel_size : float
        Pixel size of `frame_sum` in Angstroms per pixel.
    working_pixel_size : float
        Approximate pixel size, in Angstroms, the frame sum is binned to before edge
        detection.
    smoothing_sigma : float
        Standard deviation, in Angstroms, of the Gaussian applied after binning.
    n_rays : int
        Number of rays cast from the beam center.

    Returns
    -------
    BeamEdgeFit
        Fitted ellipse in `frame_sum` pixels (:func:`fit_ellipse` convention), plus
        the number of edge points found and the number kept as rim inliers.

    Raises
    ------
    ValueError
        If no beam rim can be found (e.g. no beam in the frame).
    """
    factor = max(1, round(working_pixel_size / pixel_size))
    image = gaussian_filter(
        bin_image(frame_sum, factor), smoothing_sigma / (pixel_size * factor)
    )

    center = _initial_center(image)
    for _ in range(2):  # second pass re-casts rays from the fitted center
        points = radial_edge_points(image, center, n_rays=n_rays)
        ellipse, keep = robust_ellipse_fit(points)
        center = (ellipse[0], ellipse[1])

    cy, cx, axis1, axis2, angle_deg = ellipse
    return {
        # Pixel-center convention when undoing the binning.
        "center_y": (cy + 0.5) * factor - 0.5,
        "center_x": (cx + 0.5) * factor - 0.5,
        "axis1": axis1 * factor,
        "axis2": axis2 * factor,
        "angle_deg": angle_deg,
        "n_edge_points": len(points),
        "n_inliers": int(keep.sum()),
    }


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
    working_pixel_size: float = 8.0,
    edge_smoothing_sigma: float = 120.0,
) -> BeamMaskParams:
    """Estimate a DeCo-LACE beam mask ellipse from a raw frame sum.

    Fits an ellipse to the beam with one of two methods, then computes the tight crop
    bounds for the (possibly shrunk) ellipse mask:

    - ``"sobel"``: bins and Gaussian-smooths `frame_sum`, then fits the ellipse to the
      outermost Sobel edge along rays cast from the beam center (see
      :func:`fit_beam_edge`). Robust to dark sample lying across the beam rim.
    - ``"otsu"``: low-pass filters `frame_sum` (on `device`), thresholds it with
      Otsu's method and fits an ellipse to the largest connected component.

    Parameters
    ----------
    frame_sum : torch.Tensor
        2-D tensor, sum of all raw movie frames (e.g. from :func:`sum_movie_chunked`).
    pixel_size : float
        Pixel size in Angstroms per pixel.
    threshold_method : str
        Ellipse estimation method, ``"sobel"`` or ``"otsu"``.
    diameter_reduction : float
        Fractional shrink applied to both ellipse radii before computing crop
        bounds and the returned mask parameters.
    low_pass_resolution : float
        Low-pass filter cutoff resolution in Angstroms, applied to `frame_sum`
        before thresholding. Only used by ``"otsu"``.
    device : torch.device | str | None
        Device the low-pass filter runs on. If None, uses `frame_sum`'s current
        device. Only used by ``"otsu"``; ``"sobel"`` runs on the CPU.
    working_pixel_size : float
        Approximate pixel size, in Angstroms, `frame_sum` is binned to before edge
        detection. Only used by ``"sobel"``.
    edge_smoothing_sigma : float
        Standard deviation, in Angstroms, of the Gaussian smoothing applied before edge
        detection. Only used by ``"sobel"``.

    Returns
    -------
    BeamMaskParams
        Fitted ellipse and tight crop bounds.

    Raises
    ------
    ValueError
        If `threshold_method` is not ``"sobel"`` or ``"otsu"``, or if no beam can be
        found in `frame_sum`.
    """
    n_edge_points: int | None = None
    n_inliers: int | None = None

    if threshold_method == "sobel":
        fit = fit_beam_edge(
            frame_sum.detach().cpu().numpy(),
            pixel_size,
            working_pixel_size=working_pixel_size,
            smoothing_sigma=edge_smoothing_sigma,
        )
        center_y, center_x = fit["center_y"], fit["center_x"]
        axis1, axis2, angle_deg = fit["axis1"], fit["axis2"], fit["angle_deg"]
        n_edge_points, n_inliers = fit["n_edge_points"], fit["n_inliers"]
    elif threshold_method == "otsu":
        filtered = low_pass_filter(
            frame_sum, pixel_size, low_pass_resolution, device=device
        )
        filtered_np = filtered.detach().cpu().numpy()

        threshold = threshold_otsu(filtered_np)
        binary = filtered_np > threshold

        center_y, center_x, axis1, axis2, angle_deg = fit_ellipse(binary)
    else:
        raise ValueError(
            f"Unsupported threshold_method '{threshold_method}'. Expected 'sobel' or "
            "'otsu'."
        )

    h, w = int(frame_sum.shape[0]), int(frame_sum.shape[1])

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
        "n_edge_points": n_edge_points,
        "n_inliers": n_inliers,
    }
