"""Core function for aligning frames of a cryo-EM movie."""

import torch
from torch_motion_correction import (
    DeformationField,
    FourierFilterConfig,
    PatchSamplingConfig,
    correct_motion,
    estimate_local_motion,
)
from torch_motion_correction import OptimizationConfig as MotionOptimizationConfig
from torch_motion_correction.optimization_state import OptimizationTracker


def core_align_frames(
    movie: torch.Tensor,
    initial_deformation_field: DeformationField | None,
    pixel_size: float,
    deformation_field_resolution: tuple[int, int, int],
    patch_sampling: PatchSamplingConfig,
    fourier_filter: FourierFilterConfig | None = None,
    optimization: MotionOptimizationConfig | None = None,
    do_correct_motion: bool = True,
    device: torch.device = None,
) -> tuple[torch.Tensor, DeformationField, torch.Tensor, OptimizationTracker]:
    """Core function for aligning frames of a cryo-EM movie.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to align. Must already be gain/dark corrected and
        mean-zero'd (i.e. the output of ``MovieConfig.prepare``).
    initial_deformation_field: DeformationField | None
        Starting deformation field. Pass None to initialise shifts from zero.
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
    do_correct_motion: bool
        Whether to warp the movie with the estimated deformation field.
    device: torch.device
        Device to use for computation.

    Returns
    -------
    tuple[torch.Tensor, DeformationField, torch.Tensor, OptimizationTracker]
        (corrected_movie, updated_deformation_field, movie_prepared, trajectory)
    """
    torch.set_grad_enabled(True)
    updated_deformation_field, trajectory = estimate_local_motion(
        image=movie,
        pixel_spacing=pixel_size,
        deformation_field_resolution=deformation_field_resolution,
        patch_sampling=patch_sampling,
        initial_deformation_field=initial_deformation_field,
        fourier_filter=fourier_filter,
        optimization=optimization,
    )

    if do_correct_motion:
        corrected_movie = correct_motion(
            image=movie,
            deformation_field=updated_deformation_field,
            pixel_spacing=pixel_size,
            device=device,
        )
    else:
        corrected_movie = movie

    return corrected_movie, updated_deformation_field, movie, trajectory
