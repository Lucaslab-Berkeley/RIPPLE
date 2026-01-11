"""Core function for aligning frames of a cryo-EM movie."""

from typing import TYPE_CHECKING, Any, Literal, Optional

import torch
from torch_motion_correction import (
    correct_motion,
    estimate_local_motion,
)

from .prepare_movie import prepare_core

if TYPE_CHECKING:
    from torch_motion_correction import OptimizationTracker


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def core_align_frames(
    movie: torch.Tensor,
    deformation_field: torch.Tensor,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    pixel_size: float,
    deformation_field_resolution: tuple[int, int, int],
    patch_shape: tuple[int, int],
    multiply_gain: bool = True,
    loss_trajectories: bool = False,
    skip_movie_preparation: bool = False,
    n_iterations: int = 100,
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom",
    loss_type: Literal["mse", "cc", "ncc"] = "mse",
    optimizer_type: Literal["adam", "lbfgs"] = "adam",
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),
    optimizer_kwargs: dict[str, Any] | None = None,
    do_correct_motion: bool = True,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional["OptimizationTracker"]]:
    """
    Core function for aligning frames of a cryo-EM movie.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to align.
    deformation_field: torch.Tensor
        The deformation field to use for alignment.
    gain_map: torch.Tensor | None
        The gain map to apply to the movie. If None, the gain map will be
        initialized to zero.
    dark_map: Optional[torch.Tensor]
        The dark map to apply to the movie. If None, the dark map will be
        initialized to zero.
    gain_flip: int
        The flip to apply to the gain map.
    gain_rot: int
        The rotation to apply to the gain map.
    pixel_size: float
        The pixel size in Angstroms per pixel.
    deformation_field_resolution: tuple[int, int, int]
        The resolution of the deformation field in pixels (x, y, z).
    patch_shape: tuple[int, int]
        The shape of the patch in pixels (width, height).
    multiply_gain: bool
        Whether to multiply the movie by the gain map or divide the movie by the
        gain map.
    loss_trajectories: bool
        Whether to return the trajectory of the alignment.
    skip_movie_preparation: bool
        Whether to skip the movie preparation step.
    n_iterations: int
        The number of iterations to run the alignment.
    grid_type: Literal["catmull_rom", "bspline"]
        The type of grid to use for the alignment.
    loss_type: Literal["mse", "cc", "ncc"]
        The type of loss function to use for the alignment.
    optimizer_type: Literal["adam", "lbfgs"]
        The type of optimizer to use for the alignment.
    b_factor: float
        The b-factor to use for the alignment.
    frequency_range: tuple[float, float]
        The frequency range to use for the alignment.
    optimizer_kwargs: dict[str, Any] | None
        The optimizer kwargs to use for the alignment. If None, defaults to
        {"lr": 0.2}.
    do_correct_motion: bool
        Whether to correct the motion.
    device: torch.device
        The device to use for the alignment.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional["OptimizationTracker"]]
        Tuple of
        (corrected_movie, updated_deformation_field, movie_prepared, trajectory).
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
    # estimate the motion
    if loss_trajectories:
        updated_deformation_field, trajectory = estimate_local_motion(
            image=movie_prepared,
            pixel_spacing=pixel_size,
            deformation_field_resolution=deformation_field_resolution,
            initial_deformation_field=deformation_field,
            patch_shape=patch_shape,
            n_iterations=n_iterations,
            optimizer_type=optimizer_type,
            grid_type=grid_type,
            optimizer_kwargs=optimizer_kwargs,
            b_factor=b_factor,
            frequency_range=frequency_range,
            loss_type=loss_type,
            return_trajectory=loss_trajectories,
        )
    else:
        updated_deformation_field = estimate_local_motion(
            image=movie_prepared,
            pixel_spacing=pixel_size,
            deformation_field_resolution=deformation_field_resolution,
            initial_deformation_field=deformation_field,
            patch_shape=patch_shape,
            n_iterations=n_iterations,
            optimizer_type=optimizer_type,
            grid_type=grid_type,
            optimizer_kwargs=optimizer_kwargs,
            b_factor=b_factor,
            frequency_range=frequency_range,
            loss_type=loss_type,
            return_trajectory=loss_trajectories,
        )
        trajectory = None
    # correct the motion
    if do_correct_motion:
        corrected_movie = correct_motion(
            image=movie_prepared,
            deformation_grid=updated_deformation_field,
            pixel_spacing=pixel_size,
            grid_type=grid_type,
            device=device,
        )
    else:
        corrected_movie = movie_prepared

    return corrected_movie, updated_deformation_field, movie_prepared, trajectory
