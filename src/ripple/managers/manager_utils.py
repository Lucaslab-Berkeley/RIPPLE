"""Utility functions for managers."""

from typing import TYPE_CHECKING

import torch
from torch_motion_correction import DeformationField

from ripple.core import generate_dose_weighted_image, sum_movie
from ripple.utils.data_io import (
    save_deformation_field,
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


# pylint: disable=too-many-arguments,too-many-positional-arguments
def load_missing_tensors(
    computational_config: "ComputationalConfig",
    movie_config: "MovieConfig",
    alignment_config: "BaseAlignmentConfig",
    movie: torch.Tensor | None = None,
    gain_map: torch.Tensor | None = None,
    dark_map: torch.Tensor | None = None,
    initial_deformation_field: DeformationField | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DeformationField | None]:
    """Load only the tensors that are not provided as arguments.

    Parameters
    ----------
    computational_config: ComputationalConfig
        Computational configuration containing device information.
    movie_config: MovieConfig
        Movie configuration containing movie, gain, and dark map.
    alignment_config: BaseAlignmentConfig
        Alignment configuration containing deformation field.
    movie: Optional[torch.Tensor]
        Movie tensor. If None, will be loaded from config.
    gain_map: Optional[torch.Tensor]
        Gain map tensor. If None, will be loaded from config.
    dark_map: Optional[torch.Tensor]
        Dark map tensor. If None, will be loaded from config.
    initial_deformation_field: DeformationField | None
        Starting deformation field. If None, falls back to
        ``alignment_config.initial_deformation_field`` (loaded from disk, or
        None when no path is configured — the backend will zero-initialise).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, DeformationField | None]
        Tuple of (movie, gain_map, dark_map, initial_deformation_field), with
        missing ones loaded from config.
    """
    device = computational_config.gpu_device

    if movie is None:
        movie = movie_config.movie
        if movie is not None:
            movie = movie.to(device)
    else:
        movie = movie.to(device)

    if gain_map is None:
        gain_map = movie_config.gain
        if gain_map is not None:
            gain_map = gain_map.to(device)
    else:
        gain_map = gain_map.to(device)

    if dark_map is None:
        dark_map = movie_config.dark
        if dark_map is not None:
            dark_map = dark_map.to(device)
    else:
        dark_map = dark_map.to(device)

    if initial_deformation_field is None:
        initial_deformation_field = alignment_config.initial_deformation_field
        if initial_deformation_field is not None:
            initial_deformation_field = initial_deformation_field.to(device)
    else:
        initial_deformation_field = initial_deformation_field.to(device)

    return movie, gain_map, dark_map, initial_deformation_field


# pylint: disable=too-many-arguments,too-many-positional-arguments
def save_results(
    output_config: "OutputConfig",
    movie_config: "MovieConfig",
    corrected_movie: torch.Tensor,
    updated_deformation_field: DeformationField,
    movie_prepared: torch.Tensor,
    trajectory: "OptimizationTracker",
) -> None:
    """
    Save the results of the alignment.

    Parameters
    ----------
    output_config: OutputConfig
        Output configuration containing output paths.
    movie_config: MovieConfig
        Movie configuration containing pixel size, pre-exposure, fluence, and voltage.
    corrected_movie: torch.Tensor
        The corrected movie.
    updated_deformation_field: torch.Tensor
        The updated deformation field.
    movie_prepared: torch.Tensor
        The prepared movie.
    trajectory: Optional[OptimizationTracker]
        The trajectory of the alignment.

    Returns
    -------
    None
        None.
    """
    # Save DW sum is wanted
    if output_config.dw_sum_output_path is not None:
        dw_movie = generate_dose_weighted_image(
            corrected_movie,
            movie_config.pixel_size,
            movie_config.pre_exposure,
            movie_config.fluence_per_frame,
            movie_config.voltage,
        )
        dw_movie = dw_movie.cpu()
        write_mrc_from_tensor(
            data=dw_movie,
            mrc_path=output_config.dw_sum_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save deformation field if wanted
    if output_config.deformation_field_output_path is not None:
        save_deformation_field(
            updated_deformation_field.data.cpu(),
            output_config.deformation_field_output_path,
        )

    # Save non-dw sum movie if wanted
    if output_config.non_dw_sum_output_path is not None:
        summed_movie = sum_movie(corrected_movie)
        summed_movie = summed_movie.cpu()
        write_mrc_from_tensor(
            data=summed_movie,
            mrc_path=output_config.non_dw_sum_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save aligned movie if wanted
    if output_config.motion_corrected_movie_output_path is not None:
        corrected_movie = corrected_movie.cpu()
        write_mrc_from_tensor(
            data=corrected_movie,
            mrc_path=output_config.motion_corrected_movie_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save rendered, unaligned movie if wanted
    if output_config.rendered_movie_output_path is not None:
        rendered_movie = movie_prepared.cpu()
        write_mrc_from_tensor(
            data=rendered_movie,
            mrc_path=output_config.rendered_movie_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save loss trajectories if wanted
    if output_config.loss_trajectories_output_path is not None:
        write_trajectory_to_csv(
            trajectory=trajectory,
            file_path=output_config.loss_trajectories_output_path,
        )
