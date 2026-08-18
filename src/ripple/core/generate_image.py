"""Functions for generating an image from a movie."""

import torch
from torch_fourier_filter.dose_weight import (
    dose_weight_frame_chunk,
    dose_weight_normalization_grid,
)
from torch_fourier_rescale import fourier_rescale_2d

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


def generate_dose_weighted_image(
    movie: torch.Tensor,
    pixel_size: float,
    pre_exposure: float,
    fluence_per_frame: float,
    voltage: float,
    device: torch.device | str | None = None,
    chunk_size: int = DEFAULT_DOSE_WEIGHT_CHUNK_SIZE,
) -> torch.Tensor:
    """Dose weight the movie and sum into a single 2D image.

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
        Device each chunk's FFT/dose-weight/inverse-FFT compute runs on. If None, uses
        `movie`'s current device.
    chunk_size: int
        Number of frames transferred to `compute_device` and processed at a time.

    Returns
    -------
    torch.Tensor
    The dose weighted, summed image, on the same device as `movie`.
    """
    frame_shape = (movie.shape[-2], movie.shape[-1])
    compute_device = torch.device(device) if device is not None else movie.device

    n_frames = movie.shape[0]
    chunk_size = max(1, min(chunk_size, n_frames))

    # Only (H, W) shaped, not copied across all frames
    Ne, normalization = dose_weight_normalization_grid(
        image_shape=frame_shape,
        pixel_size=pixel_size,
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

    image_dw = torch.zeros(frame_shape, dtype=movie.dtype, device=movie.device)
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = movie[start:end].to(compute_device)
        chunk_dft = torch.fft.rfft2(chunk, dim=(-2, -1))  # pylint: disable=not-callable
        chunk_dft = dose_weight_frame_chunk(
            chunk_dft=chunk_dft,
            frame_start_idx=start,
            Ne=Ne,
            normalization=normalization,
            pre_exposure=pre_exposure,
            dose_per_frame=fluence_per_frame,
            voltage=voltage,
            in_place=True,
        )
        chunk_dw = torch.fft.irfft2(  # pylint: disable=not-callable
            chunk_dft, s=frame_shape, dim=(-2, -1)
        )
        image_dw += torch.sum(chunk_dw, dim=0).to(movie.device)

    return image_dw


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
) -> torch.Tensor:
    """
    Dose weight `movie` and sum to a single 2D image, one chunk at a time.

    Notes
    -----
    Only elementwise operations are performed so function should be safe to call under
    autograd. The `chunk_size` parameter controls how many frames are processed
    simultaneously.
    """
    frame_shape = (movie.shape[-2], movie.shape[-1])
    n_frames = movie.shape[0]

    Ne, normalization = dose_weight_normalization_grid(
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
        chunk_dft = dose_weight_frame_chunk(
            chunk_dft=chunk_dft,
            frame_start_idx=start,
            Ne=Ne,
            normalization=normalization,
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

    Returns
    -------
    torch.Tensor
        The dose-weighted, frame-summed image (h, w).
    """
    if memory_strategy == "full":
        return _dose_weight_movie_to_image_streaming(
            movie, pixel_size, pre_exposure, dose_per_frame, voltage, chunk_size
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
            use_reentrant=False,
        )
    raise ValueError(f"Unknown memory strategy: {memory_strategy}")
