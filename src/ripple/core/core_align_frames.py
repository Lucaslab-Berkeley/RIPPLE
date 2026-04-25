"""Core function for estimating motion in a cryo-EM movie."""

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


def core_estimate_motion(
    movie: torch.Tensor,
    initial_deformation_field: DeformationField | None,
    pixel_size: float,
    deformation_field_resolution: tuple[int, int, int],
    patch_sampling: PatchSamplingConfig,
    fourier_filter: FourierFilterConfig | None = None,
    optimization: MotionOptimizationConfig | None = None,
    device: torch.device = None,
) -> tuple[DeformationField, OptimizationTracker]:
    """Movie motion estimation using gradient-based fitting on cubic spline grid.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to align. Must already be gain/dark corrected and
        mean-zero'd (i.e. the output of ``MovieConfig.prepare``).
    initial_deformation_field: DeformationField | None
        Starting deformation field. Pass None to initialise shifts from zero.
        Typically the result of a fast cross-correlation pre-pass.
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
    device: torch.device
        Device to use for computation.

    Returns
    -------
    tuple[DeformationField, OptimizationTracker]
        The estimated deformation field (first element) and the optimization history for
        the fit (second element).
    """
    torch.set_grad_enabled(True)
    return estimate_local_motion(
        image=movie,
        pixel_spacing=pixel_size,
        deformation_field_resolution=deformation_field_resolution,
        patch_sampling=patch_sampling,
        initial_deformation_field=initial_deformation_field,
        fourier_filter=fourier_filter,
        optimization=optimization,
        device=device,
    )


def core_align_frames(
    movie: torch.Tensor,
    deformation_field: DeformationField,
    pixel_size: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Correct the motion of a cryo-EM movie using a deformation field.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to align.
    deformation_field: DeformationField
        The estimated deformation field.
    pixel_size: float
        The pixel size in Angstroms per pixel.
    device: torch.device
        Device to use for computation.

    Returns
    -------
    torch.Tensor
        The motion-corrected (aligned) movie.
    """
    return correct_motion(
        image=movie,
        deformation_field=deformation_field,
        pixel_spacing=pixel_size,
        device=device,
    )
