"""Utility functions for managers."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch_motion_correction import DeformationField

from ripple.core.crop_bounds import CropBounds
from ripple.core.dose_weighting import movie_frame_weight_table
from ripple.core.generate_image import (
    DEFAULT_DOSE_WEIGHT_CHUNK_SIZE,
    generate_summed_images,
)
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
        The prepared movie, or, if ``skip_movie_preparation`` is True, the original
        movie cast to float32 (no other correction applied).
    """
    if alignment_config.skip_movie_preparation:
        # Movies are loaded in their native dtype (often integer counts); alignment
        # needs float32. No copy is made if the movie is already float32.
        return movie.to(dtype=torch.float32)
    return movie_config.prepare(
        movie,
        gain_map,
        dark_map,
        mask=mask,
        device=device,
        storage_device=storage_device,
        crop_bounds=crop_bounds,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
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

    Every configured output is checked (see
    :meth:`~ripple.config.OutputConfig.check_output_paths`) before anything is
    written, so with ``allow_file_overwrite`` False no file is overwritten and no
    output is written if any already exists. The dose-weighted sum uses
    ``output_config.dose_weighting`` (G&G by default).

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
        Device the summation FFT/crop/dose-weighting compute runs on (chunked). If
        None, uses `corrected_movie`'s current device.

    Returns
    -------
    None
        None.
    """
    output_config.check_output_paths()

    # Both sums come from one chunked pass over the movie. For super-resolution
    # movies each chunk is Fourier-cropped to the output pixel size; fewer frames
    # per chunk keep the per-chunk memory the same as at native sampling.
    dw_movie = None
    summed_movie = None
    summation_pixel_size = movie_config.pixel_size
    needs_dw = output_config.dw_sum_output_path is not None
    needs_non_dw = output_config.non_dw_sum_output_path is not None
    if needs_dw or needs_non_dw:
        factor = movie_config.super_resolution_factor
        frame_weights = None
        if needs_dw:
            frame_weights = movie_frame_weight_table(
                output_config.dose_weighting,
                n_frames=corrected_movie.shape[0],
                pre_exposure=movie_config.pre_exposure,
                fluence_per_frame=movie_config.fluence_per_frame,
                voltage=movie_config.voltage,
            )
        dw_movie, summed_movie, summation_pixel_size = generate_summed_images(
            corrected_movie,
            movie_config.pixel_size,
            movie_config.pre_exposure,
            movie_config.fluence_per_frame,
            movie_config.voltage,
            super_resolution_factor=factor,
            dose_weighted=needs_dw,
            non_dose_weighted=needs_non_dw,
            device=device,
            chunk_size=max(1, DEFAULT_DOSE_WEIGHT_CHUNK_SIZE // factor**2),
            frame_weights=frame_weights,
        )

    # Save DW sum is wanted
    if output_config.dw_sum_output_path is not None:
        assert dw_movie is not None
        write_mrc_from_tensor(
            data=dw_movie.cpu(),
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
        assert summed_movie is not None
        write_mrc_from_tensor(
            data=summed_movie.cpu(),
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
