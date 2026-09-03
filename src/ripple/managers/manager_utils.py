"""Utility functions for managers."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch_motion_correction import DeformationField

from ripple.core import fourier_crop_movie, generate_dose_weighted_image, sum_movie
from ripple.core.crop_bounds import CropBounds
from ripple.utils.data_io import (
    write_mrc_from_tensor,
    write_trajectory_to_csv,
)

if TYPE_CHECKING:
    from torch_motion_correction.optimization_state import OptimizationTracker

    from ripple.config import (
        BaseAlignmentConfig,
        ComputationalConfig,
        MovieConfig,
        OutputConfig,
    )


# Tuple of (movie, gain_map, dark_map, mask)
LoadedTensors = tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


def _load_or_move(
    value: torch.Tensor | None,
    loader: Callable[[], torch.Tensor | None],
    device: torch.device,
) -> torch.Tensor | None:
    """Return `value` moved to `device`, loading it via `loader()` first if None."""
    if value is None:
        value = loader()
    return value.to(device) if value is not None else None


# pylint: disable=too-many-arguments,too-many-positional-arguments
def load_missing_tensors(
    computational_config: "ComputationalConfig",
    movie_config: "MovieConfig",
    movie: torch.Tensor | None = None,
    gain_map: torch.Tensor | None = None,
    dark_map: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> LoadedTensors:
    """Load only the tensors that are not provided as arguments.

    Parameters
    ----------
    computational_config: ComputationalConfig
        Computational configuration containing device information.
    movie_config: MovieConfig
        Movie configuration containing movie, gain, dark map, and mask.
    movie: Optional[torch.Tensor]
        Movie tensor. If None, will be loaded from config.
    gain_map: Optional[torch.Tensor]
        Gain map tensor. If None, will be loaded from config.
    dark_map: Optional[torch.Tensor]
        Dark map tensor. If None, will be loaded from config.
    mask: Optional[torch.Tensor]
        Mask tensor with shape (height, width). If None, will be loaded from
        config (or remain None if no ``mask_path`` is configured).

    Returns
    -------
    LoadedTensors
        Tuple (movie, gain_map, dark_map, mask), with missing ones loaded from config.
    """
    device = computational_config.gpu_device

    # NOTE: the movie is deliberately *not* moved to `device` here. A raw movie can be
    # far larger than GPU memory. `MovieConfig.prepare` / `prepare_movie` transfers it
    # to `device` in frame chunks instead. When `movie` is supplied directly, no disk
    # read via `movie_config.movie` occurs at all.
    if movie is None:
        movie = movie_config.movie

    gain_map = _load_or_move(gain_map, lambda: movie_config.gain, device)
    dark_map = _load_or_move(dark_map, lambda: movie_config.dark, device)
    mask = _load_or_move(mask, lambda: movie_config.mask, device)

    return movie, gain_map, dark_map, mask


def load_initial_deformation_field(
    alignment_config: "BaseAlignmentConfig",
    device: torch.device,
    initial_deformation_field: DeformationField | None = None,
) -> DeformationField | None:
    """Return `initial_deformation_field`, loading it from config if not provided.

    Parameters
    ----------
    alignment_config: BaseAlignmentConfig
        Alignment configuration containing the deformation field path.
    device: torch.device
        Device to move the resulting deformation field to.
    initial_deformation_field: DeformationField | None
        Starting deformation field. If None, falls back to
        ``alignment_config.initial_deformation_field`` (loaded from disk, or
        None when no path is configured — the backend will zero-initialise).

    Returns
    -------
    DeformationField | None
        The resolved deformation field, moved to `device`, or None.
    """
    if initial_deformation_field is None:
        initial_deformation_field = alignment_config.initial_deformation_field
    return (
        initial_deformation_field.to(device)
        if initial_deformation_field is not None
        else None
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def prepare_movie_if_needed(
    movie_config: "MovieConfig",
    alignment_config: "BaseAlignmentConfig",
    movie: torch.Tensor,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    mask: torch.Tensor | None,
    device: torch.device,
    storage_device: torch.device | None = None,
    crop_bounds: CropBounds | None = None,
) -> torch.Tensor:
    """Prepare the movie unless `alignment_config.skip_movie_preparation` is set.

    Parameters
    ----------
    movie_config: MovieConfig
        Movie configuration providing the gain/dark/mask correction settings.
    alignment_config: BaseAlignmentConfig
        Alignment configuration; only ``skip_movie_preparation`` is read.
    movie: torch.Tensor
        Raw movie tensor.
    gain_map: torch.Tensor | None
        Gain map tensor, or None to skip gain correction.
    dark_map: torch.Tensor | None
        Dark map tensor, or None to skip dark correction.
    mask: torch.Tensor | None
        Mask with shape (height, width), or None to skip masking.
    device: torch.device
        Device each chunk's gain/dark/hot-pixel/mask compute runs on.
    storage_device: torch.device | None
        Device the returned, fully-prepared movie is stored on. If None, defaults to
        `device`.
    crop_bounds: CropBounds | None
        Inclusive ``(min_y, max_y, min_x, max_x)`` crop bounds applied to `movie`,
        `gain_map`, `dark_map`, and `mask` before any other preparation step. Ignored
        if ``skip_movie_preparation`` is True. If None, no cropping is applied.

    Returns
    -------
    torch.Tensor
        The prepared movie, or the original movie if ``skip_movie_preparation`` is
        True.
    """
    if alignment_config.skip_movie_preparation:
        return movie
    return movie_config.prepare(
        movie,
        gain_map,
        dark_map,
        mask=mask,
        device=device,
        storage_device=storage_device,
        crop_bounds=crop_bounds,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def save_results(
    output_config: "OutputConfig",
    movie_config: "MovieConfig",
    corrected_movie: torch.Tensor,
    updated_deformation_field: DeformationField,
    movie_prepared: torch.Tensor,
    trajectory: "OptimizationTracker",
    device: torch.device | None = None,
) -> None:
    """
    Save the results of the alignment.

    Parameters
    ----------
    output_config: OutputConfig
        Output configuration containing output paths.
    movie_config: MovieConfig
        Movie configuration containing pixel size, pre-exposure, fluence, voltage, and
        super-resolution settings.
    corrected_movie: torch.Tensor
        The corrected movie, at `movie_config.pixel_size` (native/super-resolution).
    updated_deformation_field: torch.Tensor
        The updated deformation field.
    movie_prepared: torch.Tensor
        The prepared movie.
    trajectory: Optional[OptimizationTracker]
        The trajectory of the alignment.
    device: torch.device | None
        Device the dose-weighting FFT compute runs on (chunked, frame-by-frame). If
        None, uses `corrected_movie`'s current device.

    Returns
    -------
    None
        None.
    """
    summation_movie = corrected_movie
    summation_pixel_size = movie_config.pixel_size
    needs_summation = (
        output_config.dw_sum_output_path is not None
        or output_config.non_dw_sum_output_path is not None
    )
    if needs_summation and movie_config.super_resolution_factor > 1:
        summation_movie, summation_pixel_size = fourier_crop_movie(
            corrected_movie,
            movie_config.pixel_size,
            movie_config.super_resolution_factor,
        )

    # Save DW sum is wanted
    if output_config.dw_sum_output_path is not None:
        dw_movie = generate_dose_weighted_image(
            summation_movie,
            summation_pixel_size,
            movie_config.pre_exposure,
            movie_config.fluence_per_frame,
            movie_config.voltage,
            device=device,
        )
        dw_movie = dw_movie.cpu()
        write_mrc_from_tensor(
            data=dw_movie,
            mrc_path=output_config.dw_sum_output_path,
            overwrite=output_config.allow_file_overwrite,
            pixel_size=summation_pixel_size,
        )

    # Save deformation field if wanted
    if output_config.deformation_field_output_path is not None:
        def_path = Path(output_config.deformation_field_output_path)
        if def_path.suffix in (".h5", ".hdf5"):
            updated_deformation_field.to_hdf5(def_path)
        else:
            updated_deformation_field.to_csv(def_path)

    # Save non-dw sum movie if wanted
    if output_config.non_dw_sum_output_path is not None:
        summed_movie = sum_movie(summation_movie)
        summed_movie = summed_movie.cpu()
        write_mrc_from_tensor(
            data=summed_movie,
            mrc_path=output_config.non_dw_sum_output_path,
            overwrite=output_config.allow_file_overwrite,
            pixel_size=summation_pixel_size,
        )

    # Save aligned movie if wanted
    if output_config.motion_corrected_movie_output_path is not None:
        corrected_movie = corrected_movie.cpu()
        write_mrc_from_tensor(
            data=corrected_movie,
            mrc_path=output_config.motion_corrected_movie_output_path,
            overwrite=output_config.allow_file_overwrite,
            pixel_size=movie_config.pixel_size,
        )

    # Save rendered, unaligned movie if wanted
    if output_config.rendered_movie_output_path is not None:
        rendered_movie = movie_prepared.cpu()
        write_mrc_from_tensor(
            data=rendered_movie,
            mrc_path=output_config.rendered_movie_output_path,
            overwrite=output_config.allow_file_overwrite,
            pixel_size=movie_config.pixel_size,
        )

    # Save loss trajectories if wanted
    if output_config.loss_trajectories_output_path is not None:
        write_trajectory_to_csv(
            trajectory=trajectory,
            file_path=output_config.loss_trajectories_output_path,
        )
