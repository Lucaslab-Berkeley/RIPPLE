"""Core function for aligning frames of a cryo-EM movie."""

from typing import TYPE_CHECKING, Optional

import torch
from torch_motion_correction import (
    DeformationField,
    FourierFilterConfig,
    PatchSamplingConfig,
    correct_motion,
    estimate_local_motion,
)
from torch_motion_correction import OptimizationConfig as MotionOptimizationConfig

from .prepare_movie import prepare_core

if TYPE_CHECKING:
    from torch_motion_correction import OptimizationTracker


def core_align_frames(
    movie: torch.Tensor,
    initial_deformation_field: DeformationField | None,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    pixel_size: float,
    deformation_field_resolution: tuple[int, int, int],
    patch_sampling: PatchSamplingConfig,
    fourier_filter: FourierFilterConfig | None = None,
    optimization: MotionOptimizationConfig | None = None,
    multiply_gain: bool = True,
    loss_trajectories: bool = False,
    skip_movie_preparation: bool = False,
    do_correct_motion: bool = True,
    device: torch.device = None,
) -> tuple[
    torch.Tensor, DeformationField, torch.Tensor, Optional["OptimizationTracker"]
]:
    """Core function for aligning frames of a cryo-EM movie.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to align.
    initial_deformation_field: DeformationField | None
        Starting deformation field. Pass None to initialise shifts from zero.
    gain_map: torch.Tensor | None
        The gain map to apply to the movie.
    dark_map: torch.Tensor | None
        The dark map to apply to the movie.
    gain_flip: int
        The flip to apply to the gain map.
    gain_rot: int
        The rotation to apply to the gain map.
    pixel_size: float
        The pixel size in Angstroms per pixel.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw).
    patch_sampling: PatchSamplingConfig
        Patch extraction configuration (shape and overlap).
    fourier_filter: FourierFilterConfig
        Fourier-space filtering parameters (b_factor and frequency_range).
    optimization: MotionOptimizationConfig
        Gradient-based optimisation hyper-parameters.
    multiply_gain: bool
        Whether to multiply (True) or divide (False) by the gain map.
    loss_trajectories: bool
        Whether to return the optimisation trajectory.
    skip_movie_preparation: bool
        Whether to skip gain/dark correction.
    do_correct_motion: bool
        Whether to warp the movie with the estimated deformation field.
    device: torch.device
        Device to use for computation.

    Returns
    -------
    tuple[torch.Tensor, DeformationField, torch.Tensor, OptimizationTracker | None]
        (corrected_movie, updated_deformation_field, movie_prepared, trajectory)
    """
    torch.set_grad_enabled(True)
    movie_prepared = prepare_core(
        movie,
        gain_map,
        dark_map,
        gain_flip,
        gain_rot,
        multiply_gain,
        skip_movie_preparation,
    )

    result = estimate_local_motion(
        image=movie_prepared,
        pixel_spacing=pixel_size,
        deformation_field_resolution=deformation_field_resolution,
        patch_sampling=patch_sampling,
        initial_deformation_field=initial_deformation_field,
        fourier_filter=fourier_filter,
        optimization=optimization,
        return_trajectory=loss_trajectories,
    )

    if loss_trajectories:
        updated_deformation_field, trajectory = result
    else:
        updated_deformation_field, trajectory = result, None

    if do_correct_motion:
        corrected_movie = correct_motion(
            image=movie_prepared,
            deformation_field=updated_deformation_field,
            pixel_spacing=pixel_size,
            device=device,
        )
    else:
        corrected_movie = movie_prepared

    return corrected_movie, updated_deformation_field, movie_prepared, trajectory
