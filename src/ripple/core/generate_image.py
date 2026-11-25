"""Functions for generating an image from a movie."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_fourier_filter.dose_weight import (
    dose_weight_frame_chunk,
    dose_weight_normalization_grid,
)
from torch_fourier_rescale import fourier_rescale_2d

from ripple.core.dose_weighting import frame_weights_module

if TYPE_CHECKING:
    from torch_fourier_filter.frame_weights import FrameWeightTable

DEFAULT_DOSE_WEIGHT_CHUNK_SIZE = 8


def fourier_crop_movie(
    movie: torch.Tensor,
    pixel_size: float,
    factor: int,
) -> tuple[torch.Tensor, float]:
    """Fourier-crop every frame of a movie down by an integer factor.

    Parameters
    ----------
    movie: torch.Tensor
        Movie tensor (t, h, w) to crop, at `pixel_size` Angstroms/pixel.
    pixel_size: float
        Pixel size of `movie`, in Angstroms per pixel.
    factor: int
        Integer factor by which to reduce sampling. 1 is a no-op (returns
        `movie`, `pixel_size` unchanged).

    Returns
    -------
    tuple[torch.Tensor, float]
        The Fourier-cropped movie and its new pixel size (`pixel_size * factor`).
    """
    if factor == 1:
        return movie, pixel_size

    cropped_movie, new_spacing = fourier_rescale_2d(
        movie, source_spacing=pixel_size, target_spacing=pixel_size * factor
    )
    return cropped_movie, new_spacing[0]


def fourier_crop_shape(
    frame_shape: tuple[int, int],
    pixel_size: float,
    factor: int,
) -> tuple[tuple[int, int], float]:
    """Frame shape and pixel size after Fourier-cropping by an integer factor.

    Matches the output of :func:`fourier_crop_movie` (i.e. of
    ``torch_fourier_rescale.fourier_rescale_2d``) without doing the crop.

    Parameters
    ----------
    frame_shape: tuple[int, int]
        ``(height, width)`` of the frames to crop.
    pixel_size: float
        Pixel size of the frames, in Angstroms per pixel.
    factor: int
        Integer factor by which to reduce sampling.

    Returns
    -------
    tuple[tuple[int, int], float]
        The cropped ``(height, width)`` and the pixel size along the height axis
        (``pixel_size * height / cropped_height``).
    """
    if factor == 1:
        return frame_shape, pixel_size
    ratio = pixel_size / (pixel_size * factor)
    out_shape = (
        int(np.round(frame_shape[0] * ratio)),
        int(np.round(frame_shape[1] * ratio)),
    )
    return out_shape, pixel_size * (frame_shape[0] / out_shape[0])


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-branches
def generate_summed_images(
    movie: torch.Tensor,
    pixel_size: float,
    pre_exposure: float,
    fluence_per_frame: float,
    voltage: float,
    super_resolution_factor: int = 1,
    dose_weighted: bool = True,
    non_dose_weighted: bool = True,
    device: torch.device | str | None = None,
    chunk_size: int = DEFAULT_DOSE_WEIGHT_CHUNK_SIZE,
    frame_weights: FrameWeightTable | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, float]:
    """Sum a movie with and/or without dose weighting, in one chunked pass.

    Each chunk of frames is moved to `device`. If `super_resolution_factor` > 1, the
    chunk is Fourier-cropped with :func:`fourier_crop_movie` (so the result matches
    cropping the whole movie and then summing, for any version of
    ``torch_fourier_rescale``), without ever holding the whole cropped movie. The
    dose-weighted spectra are accumulated across chunks and inverse transformed once
    at the end. Dose weighting is linear, so this equals summing the per-frame
    dose-weighted images.

    Parameters
    ----------
    movie: torch.Tensor
        Real movie with shape ``(n_frames, height, width)``. May live on a different
        device than `device`.
    pixel_size: float
        Pixel size of `movie`, in Angstroms per pixel.
    pre_exposure: float
        The total pre-exposure in (e-/A^2) before the first frame of the movie.
    fluence_per_frame: float
        The dose per frame in electrons per Angstrom^2/frame.
    voltage: float
        The accelerating voltage in kilovolts.
    super_resolution_factor: int
        Integer factor by which to Fourier-crop the frames before summing. 1 (the
        default) sums at the native sampling.
    dose_weighted: bool
        Whether to compute the dose-weighted sum.
    non_dose_weighted: bool
        Whether to compute the plain (non-dose-weighted) sum.
    device: torch.device | str | None
        Device the FFT, cropping, and dose-weighting compute runs on. If None, uses
        `movie`'s device.
    chunk_size: int
        Number of frames transferred to `device` and processed at a time.
    frame_weights: FrameWeightTable | None
        Per-frame B-factor weights for the dose-weighted sum (one row per frame; see
        :func:`ripple.core.dose_weighting.movie_frame_weight_table`). If None (the
        default), the Grant & Grigorieff weights are used, as before.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None, float]
        ``(dw_image, non_dw_image, output_pixel_size)``. Each image is on `movie`'s
        device, or None if it was not requested. `output_pixel_size` is the pixel size
        of the summed images (see :func:`fourier_crop_shape`).
    """
    frame_shape = (movie.shape[-2], movie.shape[-1])
    compute_device = torch.device(device) if device is not None else movie.device
    n_frames = movie.shape[0]
    chunk_size = max(1, min(chunk_size, n_frames))

    crop = super_resolution_factor > 1
    out_shape, out_pixel_size = fourier_crop_shape(
        frame_shape, pixel_size, super_resolution_factor
    )
    complex_dtype = (
        torch.complex128 if movie.dtype == torch.float64 else torch.complex64
    )
    dft_shape = (out_shape[0], out_shape[1] // 2 + 1)

    if frame_weights is not None and frame_weights.n_frames != n_frames:
        raise ValueError(
            f"frame_weights has {frame_weights.n_frames} frames, the movie {n_frames}"
        )

    # Critical-exposure (or frequency) and normalization grids for dose weighting,
    # if requested. Only (H, W) shaped, not copied across all frames
    dose_weighting = None
    dw_dft = None
    if dose_weighted and frame_weights is not None:
        fwm = frame_weights_module()
        dose_weighting = fwm.frame_weights_normalization_grid(
            frame_weights,
            image_shape=out_shape,
            pixel_size=out_pixel_size,
            rfft=True,
            fftshift=False,
            device=compute_device,
            chunk_size=chunk_size,
        )
        dw_dft = torch.zeros(dft_shape, dtype=complex_dtype, device=compute_device)
    elif dose_weighted:
        dose_weighting = dose_weight_normalization_grid(
            image_shape=out_shape,
            pixel_size=out_pixel_size,
            n_frames=n_frames,
            pre_exposure=pre_exposure,
            dose_per_frame=fluence_per_frame,
            voltage=voltage,
            crit_exposure_bfactor=-1,
            rfft=True,
            fftshift=False,
            device=compute_device,
            chunk_size=chunk_size,
        )
        dw_dft = torch.zeros(dft_shape, dtype=complex_dtype, device=compute_device)

    non_dw_sum = None
    if non_dose_weighted:
        non_dw_sum = torch.zeros(out_shape, dtype=movie.dtype, device=compute_device)

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = movie[start:end].to(compute_device)
        if crop:
            chunk, out_pixel_size = fourier_crop_movie(
                chunk, pixel_size, super_resolution_factor
            )
            if tuple(chunk.shape[-2:]) != out_shape:
                raise RuntimeError(
                    f"Fourier crop gave frames of shape {tuple(chunk.shape[-2:])}, "
                    f"expected {out_shape}"
                )
        if non_dw_sum is not None:
            non_dw_sum += torch.sum(chunk, dim=0)
        if dw_dft is None or dose_weighting is None:
            continue  # plain sum only, no FFT needed
        chunk_dft = torch.fft.rfft2(chunk, dim=(-2, -1))  # pylint: disable=not-callable
        del chunk  # free the real-space chunk before dose weighting

        if frame_weights is not None:
            frequency, log_normalization = dose_weighting
            chunk_dft = frame_weights_module().frame_weights_chunk(
                chunk_dft,
                frame_start_idx=start,
                table=frame_weights,
                frequency=frequency,
                log_normalization=log_normalization,
                in_place=True,
            )
        else:
            critical_exposure, normalization = dose_weighting
            chunk_dft = dose_weight_frame_chunk(
                chunk_dft=chunk_dft,
                frame_start_idx=start,
                Ne=critical_exposure,
                normalization=normalization,
                pre_exposure=pre_exposure,
                dose_per_frame=fluence_per_frame,
                voltage=voltage,
                in_place=True,
            )
        dw_dft += torch.sum(chunk_dft, dim=0)

    dw_image = None
    if dw_dft is not None:
        dw_image = torch.fft.irfft2(  # pylint: disable=not-callable
            dw_dft, s=out_shape, dim=(-2, -1)
        )
        dw_image = dw_image.to(movie.device)

    non_dw_image = None
    if non_dw_sum is not None:
        non_dw_image = non_dw_sum.to(movie.device)

    return dw_image, non_dw_image, out_pixel_size


def generate_dose_weighted_image(
    movie: torch.Tensor,
    pixel_size: float,
    pre_exposure: float,
    fluence_per_frame: float,
    voltage: float,
    device: torch.device | str | None = None,
    chunk_size: int = DEFAULT_DOSE_WEIGHT_CHUNK_SIZE,
    frame_weights: FrameWeightTable | None = None,
    super_resolution_factor: int = 1,
) -> torch.Tensor:
    """Dose weight the movie and sum into a single 2D image.

    The dose-weighted spectra are accumulated over chunks and inverse transformed
    once (see :func:`generate_summed_images`).

    Parameters
    ----------
    movie: torch.Tensor
        The movie to dose weight. May live on different device than `device`.
    pixel_size: float
        The pixel size in Angstroms per pixel.
    pre_exposure: float
        The total pre-exposure in (e-/A^2) before the first frame of the movie.
    fluence_per_frame: float
        The dose per frame in electrons per Angstrom^2/frame.
    voltage: float
        The accelerating voltage in kilovolts.
    device: torch.device | str | None
        Device each chunk's FFT/dose-weight compute runs on. If None, uses `movie`'s
        current device.
    chunk_size: int
        Number of frames transferred to `compute_device` and processed at a time.
    frame_weights: FrameWeightTable | None
        Per-frame B-factor weights (one row per frame). If None (the default), the
        Grant & Grigorieff weights are used.
    super_resolution_factor: int
        Integer factor by which to Fourier-crop the frames before summing (see
        :func:`generate_summed_images`). Default 1 (no cropping).

    Returns
    -------
    torch.Tensor
    The dose weighted, summed image, on the same device as `movie`.
    """
    dw_image, _, _ = generate_summed_images(
        movie,
        pixel_size,
        pre_exposure,
        fluence_per_frame,
        voltage,
        super_resolution_factor=super_resolution_factor,
        dose_weighted=True,
        non_dose_weighted=False,
        device=device,
        chunk_size=chunk_size,
        frame_weights=frame_weights,
    )
    assert dw_image is not None
    return dw_image


def sum_movie(
    movie: torch.Tensor,
) -> torch.Tensor:
    """
    Sum the movie.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to sum.

    Returns
    -------
    torch.Tensor
        The summed movie.
    """
    return torch.sum(movie, dim=0)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _dose_weight_movie_to_image_streaming(
    movie: torch.Tensor,
    pixel_size: float,
    pre_exposure: float,
    dose_per_frame: float,
    voltage: float,
    chunk_size: int,
    frame_weights: FrameWeightTable | None = None,
) -> torch.Tensor:
    """
    Dose weight `movie` and sum to a single 2D image, one chunk at a time.

    Notes
    -----
    Only elementwise operations are performed so function should be safe to call under
    autograd. The `chunk_size` parameter controls how many frames are processed
    simultaneously. With `frame_weights`, the per-frame B-factor weights replace the
    G&G weights.
    """
    frame_shape = (movie.shape[-2], movie.shape[-1])
    n_frames = movie.shape[0]
    if frame_weights is not None and frame_weights.n_frames != n_frames:
        raise ValueError(
            f"frame_weights has {frame_weights.n_frames} frames, the movie {n_frames}"
        )

    if frame_weights is not None:
        grids = frame_weights_module().frame_weights_normalization_grid(
            frame_weights,
            image_shape=frame_shape,
            pixel_size=pixel_size,
            rfft=True,
            fftshift=False,
            device=movie.device,
            chunk_size=chunk_size,
        )
    else:
        grids = dose_weight_normalization_grid(
            image_shape=frame_shape,
            pixel_size=pixel_size,
            n_frames=n_frames,
            pre_exposure=pre_exposure,
            dose_per_frame=dose_per_frame,
            voltage=voltage,
            crit_exposure_bfactor=-1,
            rfft=True,
            fftshift=False,
            device=movie.device,
            chunk_size=chunk_size,
        )

    image_dw = torch.zeros(frame_shape, dtype=movie.dtype, device=movie.device)
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk_dft = torch.fft.rfft2(  # pylint: disable=not-callable
            movie[start:end], dim=(-2, -1), norm="ortho"
        )
        if frame_weights is not None:
            chunk_dft = frame_weights_module().frame_weights_chunk(
                chunk_dft,
                frame_start_idx=start,
                table=frame_weights,
                frequency=grids[0],
                log_normalization=grids[1],
            )
        else:
            chunk_dft = dose_weight_frame_chunk(
                chunk_dft=chunk_dft,
                frame_start_idx=start,
                Ne=grids[0],
                normalization=grids[1],
                pre_exposure=pre_exposure,
                dose_per_frame=dose_per_frame,
                voltage=voltage,
            )
        chunk_dw = torch.fft.irfft2(  # pylint: disable=not-callable
            chunk_dft, s=frame_shape, dim=(-2, -1), norm="ortho"
        )
        image_dw = image_dw + torch.sum(chunk_dw, dim=0)

    return image_dw


def dose_weight_memory_efficient(
    movie: torch.Tensor,
    pixel_size: float,
    pre_exposure: float = 0.0,
    dose_per_frame: float = 1.0,
    voltage: float = 300.0,
    memory_strategy: str = "checkpointing",
    chunk_size: int = 10,
    frame_weights: FrameWeightTable | None = None,
) -> torch.Tensor:
    """Dose weight a movie and sum into a single 2D image, differentiably.

    Parameters
    ----------
    movie : torch.Tensor
        Input movie tensor (t, h, w)
    pixel_size : float
        Pixel size in Angstroms
    pre_exposure : float
        Pre-exposure dose
    dose_per_frame : float
        Dose per frame
    voltage : float
        Acceleration voltage
    memory_strategy : str
        Either "full" (stream chunk-by-chunk, keeping this call's activations
        for backward) or "checkpointing" (same streaming forward pass, but
        wrapped in `torch.utils.checkpoint.checkpoint` so backward recomputes
        it instead of storing any of the per-chunk intermediates).
    chunk_size : int
        Number of frames processed at a time.
    frame_weights : FrameWeightTable | None
        Per-frame B-factor weights (one row per frame). If None (the default), the
        Grant & Grigorieff weights are used.

    Returns
    -------
    torch.Tensor
        The dose-weighted, frame-summed image (h, w).
    """
    if memory_strategy == "full":
        return _dose_weight_movie_to_image_streaming(
            movie,
            pixel_size,
            pre_exposure,
            dose_per_frame,
            voltage,
            chunk_size,
            frame_weights,
        )
    if memory_strategy == "checkpointing":
        return torch.utils.checkpoint.checkpoint(
            _dose_weight_movie_to_image_streaming,
            movie,
            pixel_size,
            pre_exposure,
            dose_per_frame,
            voltage,
            chunk_size,
            frame_weights,
            use_reentrant=False,
        )
    raise ValueError(f"Unknown memory strategy: {memory_strategy}")
