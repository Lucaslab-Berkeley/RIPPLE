"""Spectrally matched noise re-fill of a DeCo-LACE sum, and its 2DTM search mask.

A DeCo-LACE exposure lights an ellipse. Outside it (and in the fringe just inside its
rim) the pixels carry no sample signal: zeros, a Poisson fill at the in-beam rate, or
Fresnel fringes. A 2DTM search whitens the whole image with one radial curve and
normalises it by global statistics, so this region distorts the whitening and the noise
model, and its steps in low-resolution content produce false peaks at the boundary.

:func:`refill_sum` fixes this after summing:

1. The pixels outside a reduced beam ellipse are replaced by Gaussian noise with the
   beam interior's radial power spectrum, blended over a short taper so that the mean
   and variance are kept across it. The whole image is then stationary.
2. Contamination inside the beam (dense ice or metal blobs) is found as large connected
   regions that are dark in a band-pass of the sum.
3. The search mask (``True`` where a particle centre may lie) is the reduced ellipse
   minus the contamination dilated by a margin, e.g. a particle radius. A search that
   takes a per-micrograph mask (Panther-EM's ``mask``) then counts only these positions
   in its null, so its false-positive cut refers to the area searched.

:func:`beam_mask_in_sum_frame` places the ellipse fitted on the raw movie (a
:class:`~ripple.config.BeamMaskResult`) in a sum's pixel frame, after the movie's
output crop and Fourier cropping by the super-resolution factor.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy import ndimage

from ripple.core.beam_mask import make_ellipse_mask


class RefillResult(TypedDict):
    """Outputs of :func:`refill_sum`."""

    image: np.ndarray
    search_mask: np.ndarray
    beam_mask: np.ndarray
    contamination: np.ndarray
    stats: dict


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def beam_mask_in_sum_frame(
    sum_shape: tuple[int, int],
    center_y: float,
    center_x: float,
    axis1: float,
    axis2: float,
    angle_deg: float,
    output_crop_min_y: int,
    output_crop_min_x: int,
    binning: float,
    diameter_reduction: float = 0.0,
) -> np.ndarray:
    """Rasterise a raw-frame beam ellipse in the pixel frame of a cropped, binned sum.

    Parameters
    ----------
    sum_shape : tuple[int, int]
        ``(height, width)`` of the sum.
    center_y, center_x, axis1, axis2, angle_deg : float
        The ellipse in raw-frame pixels, in the ``cv2.fitEllipse`` convention of
        :func:`~ripple.core.beam_mask.make_ellipse_mask` (as stored in a
        :class:`~ripple.config.BeamMaskResult`, before any reduction).
    output_crop_min_y, output_crop_min_x : int
        Raw-frame origin of the movie's output crop.
    binning : float
        Raw-frame pixels per sum pixel (the super-resolution factor). Sum pixel ``j``
        samples raw position ``crop_min + binning * j``, as Fourier cropping does.
    diameter_reduction : float
        Fractional shrink of both radii, as in ``make_ellipse_mask``.

    Returns
    -------
    np.ndarray
        Boolean ``sum_shape`` array, ``True`` inside the (reduced) ellipse.
    """
    if binning <= 0:
        raise ValueError(f"binning must be positive, got {binning}")
    return make_ellipse_mask(
        shape=(int(sum_shape[0]), int(sum_shape[1])),
        center_y=(center_y - output_crop_min_y) / binning,
        center_x=(center_x - output_crop_min_x) / binning,
        axis1=axis1 / binning,
        axis2=axis2 / binning,
        angle_deg=angle_deg,
        diameter_reduction=diameter_reduction,
    )


# pylint: disable-next=too-many-locals
def radial_power_spectrum(
    image: np.ndarray, allowed: np.ndarray, box: int = 512, step: int = 256
) -> tuple[np.ndarray, np.ndarray, int]:
    """Radial power spectrum of Hann-windowed tiles that lie wholly inside ``allowed``.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)`` image.
    allowed : np.ndarray
        Boolean ``(H, W)``; a tile is used only if every pixel of it is allowed.
    box, step : int
        Tile size and stride, pixels.

    Returns
    -------
    freq : np.ndarray
        Frequency of each radial bin, cycles per pixel (bin ``i`` is ``i / box``).
    power : np.ndarray
        Mean power per bin (empty if no tile fits).
    n_tiles : int
        Tiles averaged.
    """
    window = np.outer(np.hanning(box), np.hanning(box))
    norm = float((window**2).sum())
    acc = np.zeros((box, box // 2 + 1))
    n_tiles = 0
    height, width = image.shape
    for y in range(0, height - box + 1, step):
        for x in range(0, width - box + 1, step):
            if allowed[y : y + box, x : x + box].all():
                tile = image[y : y + box, x : x + box].astype(np.float64)
                acc += np.abs(np.fft.rfft2((tile - tile.mean()) * window)) ** 2 / norm
                n_tiles += 1
    if n_tiles == 0:
        return np.zeros(0), np.zeros(0), 0
    k = np.sqrt(np.fft.fftfreq(box)[:, None] ** 2 + np.fft.rfftfreq(box)[None, :] ** 2)
    bins = np.round(k * box).astype(int).ravel()
    counts = np.maximum(np.bincount(bins), 1)
    power = np.bincount(bins, (acc / n_tiles).ravel()) / counts
    return np.arange(power.size) / box, power, n_tiles


def coloured_noise(
    shape: tuple[int, int],
    freq: np.ndarray,
    power: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian noise whose radial power spectrum is ``power`` at ``freq``."""
    k = np.sqrt(
        np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.rfftfreq(shape[1])[None, :] ** 2
    )
    amplitude = np.sqrt(np.interp(k, freq, power))
    white = np.fft.rfft2(rng.standard_normal(shape))
    return np.fft.irfft2(white * amplitude, s=shape)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def contamination_mask(
    image: np.ndarray,
    region: np.ndarray,
    band_px: tuple[float, float] = (8.0, 150.0),
    threshold_sd: float = 5.0,
    min_area_px: int = 1000,
    polarity: str = "dark",
) -> np.ndarray:
    """Large blobs inside ``region`` that stand out in a band-pass of the image.

    The band-pass is a difference of Gaussians with standard deviations ``band_px``.
    Pixels more than ``threshold_sd`` robust standard deviations (1.4826 x MAD, over
    ``region``) from its median, on the side given by ``polarity``, are grouped into
    connected regions; regions of at least ``min_area_px`` pixels are returned.

    Returns
    -------
    np.ndarray
        Boolean mask of the blobs (not dilated).
    """
    if polarity not in {"dark", "bright", "both"}:
        raise ValueError(f"polarity must be 'dark', 'bright' or 'both', got {polarity}")
    image = image.astype(np.float32, copy=False)
    band = ndimage.gaussian_filter(image, band_px[0]) - ndimage.gaussian_filter(
        image, band_px[1]
    )
    values = band[region]
    if values.size == 0:
        return np.zeros(image.shape, bool)
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median))) + 1e-12
    score = (band - median) / mad
    if polarity == "dark":
        candidate = score < -threshold_sd
    elif polarity == "bright":
        candidate = score > threshold_sd
    else:
        candidate = np.abs(score) > threshold_sd
    labels, n = ndimage.label(candidate & region)
    if n == 0:
        return np.zeros(image.shape, bool)
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, np.arange(1, n + 1))
    return np.isin(labels, 1 + np.flatnonzero(sizes >= min_area_px))


# pylint: disable-next=too-many-arguments,too-many-locals
def refill_sum(
    image: np.ndarray,
    beam_mask: np.ndarray,
    *,
    taper_sigma_px: float = 2.0,
    interior_erosion_px: int = 8,
    psd_box: int = 512,
    psd_step: int = 256,
    contamination: bool = True,
    contamination_band_px: tuple[float, float] = (8.0, 150.0),
    contamination_threshold_sd: float = 5.0,
    contamination_min_area_px: int = 1000,
    contamination_polarity: str = "dark",
    exclusion_radius_px: float = 100.0,
    border_px: int = 2,
    seed: int = 0,
) -> RefillResult:
    """Re-fill a sum outside its beam with matched noise and build its search mask.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W)`` sum.
    beam_mask : np.ndarray
        Boolean ``(H, W)``, ``True`` inside the (reduced) beam: the pixels to keep. Use
        a reduced ellipse (e.g. ``diameter_reduction`` 0.06) so the fringe inside the
        rim is replaced too.
    taper_sigma_px : float
        Standard deviation of the Gaussian blur of ``beam_mask`` that sets the blend
        weight ``w``: the output is ``mu + w (image - mu) + sqrt(1 - w^2) noise``, which
        keeps the mean and variance through the taper.
    interior_erosion_px : int
        The noise spectrum is measured on tiles inside ``beam_mask`` eroded by this
        many pixels and away from the (dilated) contamination.
    psd_box, psd_step : int
        Tile size and stride of the spectrum (see :func:`radial_power_spectrum`).
    contamination : bool
        Find contamination blobs (:func:`contamination_mask`) and leave them, dilated
        by ``exclusion_radius_px``, out of the search mask. They are not re-filled.
    contamination_band_px, contamination_threshold_sd, contamination_min_area_px, \
contamination_polarity
        Passed to :func:`contamination_mask`.
    exclusion_radius_px : float
        Dilation of the contamination, pixels; about a particle radius.
    border_px : int
        No search-mask pixel within this many pixels of the image border.
    seed : int
        Seed of the noise.

    Returns
    -------
    RefillResult
        ``image`` (float32, re-filled), ``search_mask`` (bool, ``True`` where a centre
        may lie), ``beam_mask`` (as given), ``contamination`` (the dilated blobs) and
        ``stats`` (fractions, tile counts and the re-fill's power relative to the
        interior's at a few resolutions, given in pixels per cycle).

    Raises
    ------
    ValueError
        If no tile of ``psd_box`` fits in the beam interior.
    """
    if image.ndim != 2 or image.shape != beam_mask.shape:
        raise ValueError("image and beam_mask must be 2-D arrays of the same shape")
    rng = np.random.default_rng(seed)
    data = image.astype(np.float64)
    beam = beam_mask.astype(bool)

    if contamination:
        blobs = contamination_mask(
            data,
            beam,
            band_px=contamination_band_px,
            threshold_sd=contamination_threshold_sd,
            min_area_px=contamination_min_area_px,
            polarity=contamination_polarity,
        )
        n_blobs = int(ndimage.label(blobs)[1])
        excluded = ndimage.distance_transform_edt(~blobs) <= exclusion_radius_px
    else:
        n_blobs = 0
        excluded = np.zeros_like(beam)

    interior = beam & ~excluded
    if interior_erosion_px > 0:
        interior = ndimage.binary_erosion(interior, iterations=interior_erosion_px)
    freq, power, n_tiles = radial_power_spectrum(data, interior, psd_box, psd_step)
    if n_tiles == 0:
        raise ValueError(
            f"no {psd_box}-px tile fits inside the beam interior; lower psd_box"
        )

    # one correction pass: the realised noise's spectrum is divided out
    noise = coloured_noise(data.shape, freq, power, rng)
    _, realised, _ = radial_power_spectrum(noise, np.ones_like(beam), psd_box, psd_step)
    power = power * power / np.maximum(realised, 1e-30)
    noise = coloured_noise(data.shape, freq, power, rng)

    mean = float(data[interior].mean())
    weight = np.clip(
        ndimage.gaussian_filter(beam.astype(np.float64), taper_sigma_px), 0, 1
    )
    refilled = mean + weight * (data - mean) + np.sqrt(1.0 - weight**2) * noise

    search = beam & ~excluded
    if border_px > 0:
        search[:border_px] = search[-border_px:] = False
        search[:, :border_px] = search[:, -border_px:] = False

    stats: dict = {
        "beam_fraction": float(beam.mean()),
        "search_fraction": float(search.mean()),
        "n_contamination_blobs": n_blobs,
        "spectrum_tiles": n_tiles,
    }
    stats.update(_power_check(refilled, interior, beam, psd_box, interior_erosion_px))

    return RefillResult(
        image=refilled.astype(np.float32),
        search_mask=search,
        beam_mask=beam,
        contamination=excluded,
        stats=stats,
    )


def _power_check(
    refilled: np.ndarray,
    interior: np.ndarray,
    beam: np.ndarray,
    psd_box: int,
    erosion_px: int,
) -> dict:
    """Re-fill power over interior power at a few resolutions (pixels per cycle)."""
    small = max(psd_box // 2, 32)
    outside = ndimage.binary_erosion(~beam, iterations=max(erosion_px, 1))
    f_in, p_in, _ = radial_power_spectrum(refilled, interior, small, small // 2)
    _, p_out, n_out = radial_power_spectrum(refilled, outside, small, small // 2)
    check: dict = {"outside_tiles": n_out}
    if n_out:
        ratios = {}
        for pixels_per_cycle in (64.0, 16.0, 8.0, 4.0, 3.0, 2.2):
            i = int(np.argmin(np.abs(f_in - 1.0 / pixels_per_cycle)))
            ratios[f"{pixels_per_cycle:g}"] = round(float(p_out[i] / p_in[i]), 3)
        check["refill_over_interior_power"] = ratios
    return check
