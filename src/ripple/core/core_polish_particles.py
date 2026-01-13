"""Core function for polishing particles."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import einops
import pandas as pd
import torch
import tqdm
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d
from torch_motion_correction import correct_motion, correct_motion_two_grids
from torch_motion_correction.data_io import write_deformation_field_to_csv
from torch_motion_correction.deformation_field_utils import (
    resample_deformation_field,
)
from torch_motion_correction.optimization_state import OptimizationTracker

from ripple.utils.data_io import load_template_volume_from_config

from .core_utils import (
    _create_batch_configs,
    _filter_particles_by_quality,
    _make_differentiable_refine_manager,
    get_batch_mean_std_stacks,
)
from .generate_image import dose_weight_memory_efficient
from .motion_priors import (
    _build_physical_coords,
    _compute_physical_spacing,
    _create_exponential_sigma_a,
    _normalize_sigma_fluence,
    laplacian_compute,
    relion2019_compute,
)
from .prepare_movie import prepare_core


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def core_polish_particles(
    movie: torch.Tensor,  # (t, H, W)
    initial_deformation_field: torch.Tensor,
    refine_config_path: str,
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    pixel_size: float,
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    multiply_gain: bool = True,
    loss_trajectories: bool = False,
    skip_movie_preparation: bool = False,
    n_iterations: int = 100,
    optimizer_kwargs: dict[str, Any] | None = None,
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom",
    trajectory_kwargs: dict | None = None,
    correlation_batch_size: int = 20,
    do_correct_motion: bool = True,
    voltage: float = 300.0,
    particle_indices: pd.Index = None,
    device: torch.device = None,
    movie_extract: bool = False,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    particle_batch_size: int = 102,
    save_intermediate_fields: bool = False,
    intermediate_fields_dir: str = ".",
    prior_type: str = "relion",
    sigma_d: float = 5782.376953,
    sigma_v: float = 0.194826,
    sigma_a: float = 0.513517,
    alpha_spatial: float = 1e5,
    sigma_a_exponential: bool = False,
    sigma_a_amplitude: float = 2.0,
    sigma_a_decay: float = 0.1,
    sigma_a_offset: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, OptimizationTracker | None]:
    """
    Core function for polishing particles.

    Parameters
    ----------
    movie: torch.Tensor
        (t, H, W) movie to polish.
    initial_deformation_field: torch.Tensor
        (2, nt, nh, nw) initial deformation field.
    refine_config_path: str
        Path to the refine config file.
    var_image: torch.Tensor
        (t, H, W) variance image.
    mean_image: torch.Tensor
        (t, H, W) mean image.
    gain_map: torch.Tensor | None
        (H, W) gain map.
    dark_map: torch.Tensor | None
        (H, W) dark map.
    gain_flip: int
        Gain flip value.
    gain_rot: int
        Gain rotation value.
    pixel_size: float
        Pixel size in Angstroms.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw).
    pre_exposure: float
        Pre-exposure time in seconds.
    fluence_per_frame: float
        Fluence per frame in electrons per pixel.
    multiply_gain: bool
        Whether to multiply the gain map by the movie.
    loss_trajectories: bool
        Whether to return the optimization trajectory.
    skip_movie_preparation: bool
        Whether to skip the movie preparation step.
    n_iterations: int
        Number of iterations for the optimization process.
    optimizer_kwargs: dict[str, Any] | None
        Keyword arguments for the optimizer.
    grid_type: Literal["catmull_rom", "bspline"]
        Grid type to use for the deformation field.
    trajectory_kwargs: dict | None
        Keyword arguments for the trajectory tracking.
    correlation_batch_size: int
        Batch size for the correlation.
    do_correct_motion: bool
        Whether to correct the motion.
    voltage: float
        Voltage in kV.
    particle_indices: pd.Index | None
        Particle indices to use for the refinement.
    device: torch.device | None
        Device to perform computation on.
    movie_extract: bool
        Whether to extract the movie.
    loss_metric: str
        Loss metric to use for the refinement.
    min_snr: float
        Minimum SNR to use for the refinement.
    best_n: int
        Maximum number of particles to use for the refinement.
    particle_batch_size: int
        Number of particles to process per batch for gradient accumulation.
        Default is 100.
    save_intermediate_fields: bool
        Whether to save the intermediate fields.
    intermediate_fields_dir: str
        Directory to save the intermediate fields.
    prior_type: str
        Type of prior to use. Default is 'relion'.
    sigma_d: float
        Spatial correlation length in Angstroms for RELION prior.
        Default is 5782.376953.
    sigma_v: float
        Velocity magnitude scale in Å per unit fluence for RELION prior.
        Default is 0.194826.
    sigma_a: float
        Temporal smoothness parameter. Default is 0.513517.
    alpha_spatial: float
        Spatial smoothness strength for Laplacian prior. Default is 1e5.
    sigma_a_exponential: bool
        Whether to use exponential decay for sigma_a over frames. Default is False.
    sigma_a_amplitude: float
        Amplitude in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset. Default is 2.0.
    sigma_a_decay: float
        Decay rate in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset. Default is 0.1.
    sigma_a_offset: float
        Constant offset in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset. Default is 1.0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, OptimizationTracker]
        - Corrected movie (t, H, W)
        - Updated deformation field (2, nt, nh, nw)
        - Movie prepared (t, H, W)
        - Optimization trajectory (OptimizationTracker)
    """
    movie_prepared = prepare_core(
        movie,
        gain_map,
        dark_map,
        gain_flip,
        gain_rot,
        multiply_gain,
        skip_movie_preparation,
    )

    # Prepare common kwargs for estimation functions (shared between both methods)
    estimate_kwargs = {
        "image": movie_prepared,
        "var_image": var_image,
        "mean_image": mean_image,
        "deformation_field_resolution": deformation_field_resolution,
        "initial_deformation_field": initial_deformation_field,
        "refine_config_path": refine_config_path,
        "pre_exposure": pre_exposure,
        "fluence_per_frame": fluence_per_frame,
        "n_iterations": n_iterations,
        "optimizer_kwargs": optimizer_kwargs,
        "return_trajectory": loss_trajectories,
        "trajectory_kwargs": trajectory_kwargs,
        "correlation_batch_size": correlation_batch_size,
        "particle_indices": particle_indices,
        "device": device,
        "loss_metric": loss_metric,
        "min_snr": min_snr,
        "best_n": best_n,
        "save_intermediate_fields": save_intermediate_fields,
        "intermediate_fields_dir": intermediate_fields_dir,
    }
    # Prior parameters only for bayesian estimation
    prior_kwargs: dict[str, Any] = {
        "prior_type": prior_type,
        "sigma_d": sigma_d,
        "sigma_v": sigma_v,
        "sigma_a": sigma_a,
        "alpha_spatial": alpha_spatial,
        "sigma_a_exponential": sigma_a_exponential,
        "sigma_a_amplitude": sigma_a_amplitude,
        "sigma_a_decay": sigma_a_decay,
        "sigma_a_offset": sigma_a_offset,
    }

    # estimate the motion
    if loss_trajectories:
        if movie_extract:
            (
                updated_deformation_field,
                trajectory,
            ) = estimate_local_motion_2dtm_particles_bayesian(
                **estimate_kwargs,
                **prior_kwargs,
                particle_batch_size=particle_batch_size,
                pixel_spacing=pixel_size,
            )
        else:
            updated_deformation_field, trajectory = estimate_local_motion_2dtm_bayesian(
                **estimate_kwargs,
                **prior_kwargs,
                pixel_spacing=pixel_size,
                grid_type=grid_type,
                voltage=voltage,
            )
    else:
        if movie_extract:
            updated_deformation_field = estimate_local_motion_2dtm_particles_bayesian(
                **estimate_kwargs,
                **prior_kwargs,
                particle_batch_size=particle_batch_size,
                pixel_spacing=pixel_size,
            )
        else:
            updated_deformation_field = estimate_local_motion_2dtm_bayesian(
                **estimate_kwargs,
                **prior_kwargs,
                pixel_spacing=pixel_size,
                grid_type=grid_type,
                voltage=voltage,
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def estimate_local_motion_2dtm_bayesian(
    image: torch.Tensor,  # (t, H, W)
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    refine_config_path: str,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    n_iterations: int = 100,
    optimizer_kwargs: dict[str, Any] | None = None,
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom",
    return_trajectory: bool = False,
    trajectory_kwargs: dict | None = None,
    correlation_batch_size: int = 20,
    voltage: float = 300.0,
    particle_indices: pd.Index = None,
    device: torch.device = None,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    save_intermediate_fields: bool = False,
    intermediate_fields_dir: str = ".",
    prior_type: str = "relion",
    sigma_d: float = 5782.376953,
    sigma_v: float = 0.194826,
    sigma_a: float = 0.513517,
    alpha_spatial: float = 1e5,
    sigma_a_exponential: bool = False,
    sigma_a_amplitude: float = 2.0,
    sigma_a_decay: float = 0.1,
    sigma_a_offset: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, OptimizationTracker]:
    """Estimate motion (new method).

    Parameters
    ----------
    image: torch.Tensor
        (t, H, W) image to estimate motion from where t is the number of frames,
        H is the height, and W is the width.
    var_image: torch.Tensor
        (t, H, W) variance image to estimate motion from.
    mean_image: torch.Tensor
        (t, H, W) mean image to estimate motion from.
    pixel_spacing: float
        Pixel spacing in Angstroms.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw) where nt is the number of
        time points, nh is the number of control points in height, and nw is the
        number of control points in width.
    initial_deformation_field: torch.Tensor | None
        Initial deformation field to start from with shape (2, nt, nh, nw) where 2
        corresponds to (y, x) shifts. If None, initializes to zero shifts.
    refine_config_path: str
        Path to the refine config file.
    pre_exposure: float
        Pre-exposure time in seconds. Default is 0.0.
    fluence_per_frame: float
        Dose per frame in electrons per pixel. Default is 1.0.
    particle_indices: pd.Index = None,
        Particle indices to use for the refinement. If None, uses all particles.
    voltage: float = 300.0,
        Voltage in kV. Default is 300.0.
    loss_metric: str = "scaled_mip",
        Loss metric to use for the refinement. Default is "scaled_mip".
    min_snr: float = 0.0,
        Minimum SNR to use for the refinement. Default is 0.0.
    best_n: int = 10000000000,
        Maximum number of particles to use for the refinement. Default is 10000000000.
    correlation_batch_size: int = 20,
        Batch size for the correlation. Default is 20.
    device: torch.device = None,
        Device to perform computation on. If None, uses the device of the input image.
    n_iterations: int = 100,
        Number of iterations for the optimization process. Default is 100.
    optimizer_kwargs: dict[str, Any] | None = None,
        Keyword arguments for the optimizer. If None, uses defaults.
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom",
        Grid type to use for the deformation field. Default is "catmull_rom".
    return_trajectory: bool = False,
        Whether to return the optimization trajectory. Default is False. If true, a
        second return value will be provided which is an OptimizationTrajectory object.
    trajectory_kwargs: dict | None = None,
        Additional keyword arguments for the trajectory tracking. If None, uses
        defaults.
    save_intermediate_fields: bool = False,
        Whether to save the intermediate fields.
    intermediate_fields_dir: str
        Directory to save the intermediate fields.
    prior_type: str
        Type of motion prior: "relion" or "laplacian". Default is "relion".
    sigma_d: float
        Spatial correlation length in Angstroms (RELION only).
        Default is 5782.376953.
    sigma_v: float
        Velocity magnitude scale in Å per unit fluence (RELION only).
        Default is 0.194826.
    sigma_a: float
        Temporal smoothness in Å/(e-/Å²). Smaller = smoother.
        Default is 0.513517.
    alpha_spatial: float
        Spatial smoothness strength (Laplacian only). Larger = smoother.
        Default is 1e5.
    sigma_a_exponential: bool
        Use exponential decay for sigma_a over frames. Default is False.
    sigma_a_amplitude: float
        Amplitude in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset. Default is 2.0.
    sigma_a_decay: float
        Decay rate in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset.
        Default is 0.1 (1/(e-/Å²)).
    sigma_a_offset: float
        Constant offset in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset. Default is 1.0.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, OptimizationTracker]
        The estimated deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts. If `return_trajectory` is True, also returns an
        OptimizationTrajectory object containing the optimization history.
    """
    torch.set_grad_enabled(True)

    # Setup common parameters for motion estimation
    setup_result = _setup_estimate_motion(
        refine_config_path=refine_config_path,
        particle_indices=particle_indices,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        temp_dir=None,
        var_image=var_image,
        mean_image=mean_image,
        image=image,
        optimizer_kwargs=optimizer_kwargs,
        return_trajectory=return_trajectory,
        trajectory_kwargs=trajectory_kwargs,
        initial_deformation_field=initial_deformation_field,
        deformation_field_resolution=deformation_field_resolution,
        device=device,
        requires_grad=False,
        grid_type=grid_type,
    )
    refine_config_path = setup_result["refine_config_path"]
    particle_indices = setup_result["particle_indices"]
    template_volume = setup_result["template_volume"]
    var_image = setup_result["var_image"]
    mean_image = setup_result["mean_image"]
    image = setup_result["image"]
    optimizer_kwargs = setup_result["optimizer_kwargs"]
    trajectory = setup_result["trajectory"]
    deformation_field = setup_result["deformation_field"]
    deformation_field_data = setup_result["deformation_field_data"]

    print("Making new deformation field")
    new_deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution, n_channels=2
    ).to(device)
    print("New deformation field made")

    # Setup prior-specific parameters
    prior_params = _setup_priors(
        prior_type=prior_type,
        sigma_a_exponential=sigma_a_exponential,
        sigma_a=sigma_a,
        sigma_a_amplitude=sigma_a_amplitude,
        sigma_a_decay=sigma_a_decay,
        sigma_a_offset=sigma_a_offset,
        sigma_v=sigma_v,
        image=image,
        fluence_per_frame=fluence_per_frame,
        deformation_field_resolution=deformation_field_resolution,
        pixel_spacing=pixel_spacing,
        device=device,
    )
    image_coords = prior_params.get("image_coords")
    sigma_v_norm = prior_params.get("sigma_v_norm")
    sigma_a_norm = prior_params["sigma_a_norm"]
    spatial_spacing = prior_params.get("spatial_spacing")
    temporal_spacing = prior_params.get("temporal_spacing")

    motion_optimizer = torch.optim.Adam(
        params=new_deformation_field.parameters(),
        lr=optimizer_kwargs["lr"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )

    # "Training" loop going over all patched n_iterations times
    pbar = tqdm.tqdm(range(n_iterations))

    for iter_idx in pbar:
        if save_intermediate_fields:
            write_deformation_field_to_csv(
                new_deformation_field.data,
                f"{intermediate_fields_dir}/new_deformation_field_{iter_idx}.csv",
            )
        torch.cuda.empty_cache()

        print("Correcting motion with two grids")

        corrected_movie = correct_motion_two_grids(
            image=image,
            new_deformation_grid=new_deformation_field,
            base_deformation_grid=deformation_field,
            pixel_spacing=pixel_spacing,
            grad=True,
            device=device,
        )

        # dose weight this movie
        print("Dose weighting movie")
        dw_image = dose_weight_memory_efficient(
            corrected_movie,
            pixel_spacing,
            pre_exposure=pre_exposure,
            dose_per_frame=fluence_per_frame,
            voltage=voltage,
            memory_efficient=True,
            chunk_size=1,
            memory_strategy="full",
        )
        if dw_image.ndim == 2:
            dw_image = einops.rearrange(dw_image, "h w ->  1 h w")

        refine_manager = _make_differentiable_refine_manager(
            refine_config_path=refine_config_path,
        )

        backend_kwargs = refine_manager.make_differentiable_backend_kwargs(
            image_stack=dw_image,
            mean_stack=mean_image,
            std_stack=var_image,
            particle_indices=particle_indices,
            template_tensor=template_volume,
            images_are_particles=False,
        )

        result = refine_manager.get_refine_result(
            backend_kwargs,
            correlation_batch_size=correlation_batch_size,
            use_differentiable=True,
        )

        refined_mip = result["refined_cross_correlation"]
        refined_scaled_mip = result["refined_z_score"]

        # Select loss metric based on loss_metric parameter
        if loss_metric == "mip":
            loss_tensor = refined_mip
        elif loss_metric == "scaled_mip":
            loss_tensor = refined_scaled_mip
        else:
            raise ValueError(
                f"Unknown loss_metric: {loss_metric}. Must be 'mip' or 'scaled_mip'"
            )

        # Compute loss with motion priors
        loss = _compute_loss(
            loss_tensor=loss_tensor,
            prior_type=prior_type,
            deformation_field=deformation_field,
            batch_size=1,
            total_n_particles=1,
            image_coords=image_coords,
            sigma_d=sigma_d,
            sigma_v_norm=sigma_v_norm,
            sigma_a_norm=sigma_a_norm,
            alpha_spatial=alpha_spatial,
            spatial_spacing=spatial_spacing,
            temporal_spacing=temporal_spacing,
        )

        motion_optimizer.zero_grad()

        try:
            loss.backward()
        except RuntimeError as e:
            print("❌ Backward pass FAILED with error:")
            print(f"   {e!s}")
            raise

        motion_optimizer.step()

        # log loss
        if iter_idx % 1 == 0:
            print(f"{iter_idx}: mean cc = {-1 * loss.item()}")

        if trajectory is not None and trajectory.sample_this_step(iter_idx):
            trajectory.add_checkpoint(
                deformation_field=deformation_field_data,
                loss=loss,
                step=iter_idx,
            )
    # Return final deformation field
    final_deformation_field = new_deformation_field.data + deformation_field.data
    average_shift = torch.mean(final_deformation_field, dim=(1, 2, 3), keepdim=True)
    final_deformation_field = final_deformation_field - average_shift

    if return_trajectory:
        return final_deformation_field, trajectory
    return final_deformation_field


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def estimate_local_motion_2dtm_particles_bayesian(
    image: torch.Tensor,  # (t, H, W)
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    refine_config_path: str,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    n_iterations: int = 100,
    optimizer_kwargs: dict[str, Any] | None = None,
    return_trajectory: bool = False,
    trajectory_kwargs: dict | None = None,
    correlation_batch_size: int = 20,
    particle_batch_size: int = 102,
    particle_indices: pd.Index = None,
    device: torch.device = None,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    save_intermediate_fields: bool = False,
    intermediate_fields_dir: str = ".",
    prior_type: str = "relion",
    sigma_d: float = 5782.376953,
    sigma_v: float = 0.194826,
    sigma_a: float = 0.513517,
    alpha_spatial: float = 1e5,
    sigma_a_exponential: bool = False,
    sigma_a_amplitude: float = 2.0,
    sigma_a_decay: float = 0.1,
    sigma_a_offset: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, OptimizationTracker]:
    """Estimate motion (new method).

    Parameters
    ----------
    image: torch.Tensor
        (t, H, W) image to estimate motion from where t is the number of frames,
        H is the height, and W is the width.
    var_image: torch.Tensor
        (t, H, W) variance image to estimate motion from.
    mean_image: torch.Tensor
        (t, H, W) mean image to estimate motion from.
    pixel_spacing: float
        Pixel spacing in Angstroms.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw) where nt is the number of
        time points, nh is the number of control points in height, and nw is the
        number of control points in width.
    initial_deformation_field: torch.Tensor | None
        Initial deformation field to start from with shape (2, nt, nh, nw) where 2
        corresponds to (y, x) shifts. If None, initializes to zero shifts.
    refine_config_path: str
        Path to the refine config file.
    pre_exposure: float
        Pre-exposure time in seconds. Default is 0.0.
    fluence_per_frame: float
        Dose per frame in electrons per pixel. Default is 1.0.
    device: torch.device, optional
        Device to perform computation on. If None, uses the device of the input image.
    n_iterations: int
        Number of iterations for the optimization process. Default is 100.
    optimizer_kwargs: dict[str, Any] | None = None,
        Keyword arguments for the optimizer. If None, uses defaults.
    return_trajectory: bool
        Whether to return the optimization trajectory. Default is False. If true, a
        second return value will be provided which is an OptimizationTrajectory object.
    trajectory_kwargs: dict | None
        Additional keyword arguments for the trajectory tracking. If None, uses
        defaults.
    correlation_batch_size: int
        Batch size for the correlation. Default is 20.
    particle_indices: pd.Index = None,
        Particle indices to use for the refinement. If None, uses all particles.
    particle_batch_size: int
        Number of particles to process per batch for gradient accumulation.
        Default is 100.
    loss_metric: str
        Loss metric to use for the refinement. Default is "scaled_mip".
    min_snr: float
        Minimum SNR to use for the refinement. Default is 0.0.
    best_n: int
        Maximum number of particles to use for the refinement. Default is 10000000000.
    save_intermediate_fields: bool
        Whether to save the intermediate fields.
    intermediate_fields_dir: str
        Directory to save the intermediate fields.
    prior_type: str
        Type of motion prior: "relion" or "laplacian". Default is "relion".
    sigma_d: float
        Spatial correlation length in Angstroms (RELION only).
        Default is 5782.376953.
    sigma_v: float
        Velocity magnitude scale in Å per unit fluence (RELION only).
        Default is 0.194826.
    sigma_a: float
        Temporal smoothness in Å/(e-/Å²). Smaller = smoother.
        Default is 0.513517.
    alpha_spatial: float
        Spatial smoothness strength (Laplacian only). Larger = smoother.
        Default is 1e5.
    sigma_a_exponential: bool
        Use exponential decay for sigma_a over frames. Default is False.
    sigma_a_amplitude: float
        Amplitude in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset. Default is 2.0.
    sigma_a_decay: float
        Decay rate in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset.
        Default is 0.1 (1/(e-/Å²)).
    sigma_a_offset: float
        Constant offset in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset. Default is 1.0.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, OptimizationTracker]
        The estimated deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts. If `return_trajectory` is True, also returns an
        OptimizationTrajectory object containing the optimization history.
    """
    torch.set_grad_enabled(True)

    # Create temporary directory for filtering and batch configs
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_batch_"))

    # Setup common parameters for motion estimation
    setup_result = _setup_estimate_motion(
        refine_config_path=refine_config_path,
        particle_indices=particle_indices,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        temp_dir=temp_dir,
        var_image=var_image,
        mean_image=mean_image,
        image=image,
        optimizer_kwargs=optimizer_kwargs,
        return_trajectory=return_trajectory,
        trajectory_kwargs=trajectory_kwargs,
        initial_deformation_field=initial_deformation_field,
        deformation_field_resolution=deformation_field_resolution,
        device=device,
        requires_grad=True,
    )
    refine_config_path = setup_result["refine_config_path"]
    particle_indices = setup_result["particle_indices"]
    template_volume = setup_result["template_volume"]
    var_image = setup_result["var_image"]
    mean_image = setup_result["mean_image"]
    image = setup_result["image"]
    optimizer_kwargs = setup_result["optimizer_kwargs"]
    trajectory = setup_result["trajectory"]
    deformation_field = setup_result["deformation_field"]

    motion_optimizer = torch.optim.Adam(
        params=deformation_field.parameters(),
        lr=optimizer_kwargs["lr"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )

    # "Training" loop going over all patched n_iterations times
    pbar = tqdm.tqdm(range(n_iterations))

    # Create batch configs once before optimization loop
    # This will work with the already-filtered CSV
    batch_config_paths, batch_particle_indices = _create_batch_configs(
        refine_config_path=refine_config_path,
        particle_batch_size=particle_batch_size,
        temp_dir=temp_dir,
    )

    # Setup prior-specific parameters
    prior_params = _setup_priors(
        prior_type=prior_type,
        sigma_a_exponential=sigma_a_exponential,
        sigma_a=sigma_a,
        sigma_a_amplitude=sigma_a_amplitude,
        sigma_a_decay=sigma_a_decay,
        sigma_a_offset=sigma_a_offset,
        sigma_v=sigma_v,
        image=image,
        fluence_per_frame=fluence_per_frame,
        deformation_field_resolution=deformation_field_resolution,
        pixel_spacing=pixel_spacing,
        device=device,
    )
    image_coords = prior_params.get("image_coords")
    sigma_v_norm = prior_params.get("sigma_v_norm")
    sigma_a_norm = prior_params["sigma_a_norm"]
    spatial_spacing = prior_params.get("spatial_spacing")
    temporal_spacing = prior_params.get("temporal_spacing")

    # Calculate total number of particles across all batches
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)
    # Pre-compute mean/std stacks for all batches (they don't change across iterations)
    batch_mean_stacks, batch_std_stacks = get_batch_mean_std_stacks(
        batch_config_paths=batch_config_paths,
        batch_particle_indices=batch_particle_indices,
        mean_image=mean_image,
        var_image=var_image,
    )
    # Use try-finally to ensure cleanup of temp directory
    try:
        for iter_idx in pbar:
            if save_intermediate_fields:
                write_deformation_field_to_csv(
                    deformation_field.data,
                    f"{intermediate_fields_dir}/particle_deformation_field_{iter_idx}.csv",
                )

            torch.cuda.empty_cache()

            # Zero gradients once at the start of iteration
            motion_optimizer.zero_grad()
            accumulated_loss = 0.0

            # Process particles in batches for gradient accumulation
            for batch_config_path, batch_indices in zip(
                batch_config_paths, batch_particle_indices, strict=True
            ):
                # Create refine manager for this batch with batch-specific config
                batch_refine_manager = _make_differentiable_refine_manager(
                    refine_config_path=batch_config_path,
                )
                batch_particle_stack = batch_refine_manager.particle_stack
                batch_size = len(batch_indices[0])  # Get size from batch_indices

                # 1. Extract particle images for this batch
                image_stack_batch = (
                    batch_particle_stack.construct_image_stack_from_movie(
                        movie=image,
                        deformation_field=deformation_field,
                        pos_reference="top-left",
                        handle_bounds="pad",
                        padding_mode="reflect",
                        padding_value=0.0,
                        pre_exposure=pre_exposure,
                        fluence_per_frame=fluence_per_frame,
                    )
                )

                # 2. Reuse pre-computed mean/std stacks
                batch_mean_stack = batch_mean_stacks[batch_config_path]
                batch_std_stack = batch_std_stacks[batch_config_path]

                # 3. Create backend kwargs for this batch
                backend_kwargs = (
                    batch_refine_manager.make_differentiable_backend_kwargs(
                        image_stack=image_stack_batch,
                        mean_stack=batch_mean_stack,
                        std_stack=batch_std_stack,
                        particle_indices=batch_indices,
                        images_are_particles=True,
                        template_tensor=template_volume,
                    )
                )

                result = batch_refine_manager.get_refine_result(
                    backend_kwargs,
                    correlation_batch_size=correlation_batch_size,
                    use_differentiable=True,
                )

                refined_mip = result["refined_cross_correlation"]
                refined_scaled_mip = result["refined_z_score"]

                # Select loss metric based on loss_metric parameter
                if loss_metric == "mip":
                    loss_tensor = refined_mip
                elif loss_metric == "scaled_mip":
                    loss_tensor = refined_scaled_mip
                else:
                    raise ValueError(
                        "Unknown loss_metric: "
                        f"{loss_metric}. Must be 'mip' or 'scaled_mip'"
                    )

                # Compute loss with motion priors
                batch_loss = _compute_loss(
                    loss_tensor=loss_tensor,
                    prior_type=prior_type,
                    deformation_field=deformation_field,
                    batch_size=batch_size,
                    total_n_particles=total_n_particles,
                    image_coords=image_coords,
                    sigma_d=sigma_d,
                    sigma_v_norm=sigma_v_norm,
                    sigma_a_norm=sigma_a_norm,
                    alpha_spatial=alpha_spatial,
                    spatial_spacing=spatial_spacing,
                    temporal_spacing=temporal_spacing,
                )
                accumulated_loss += batch_loss.item()
                print(f"batch_loss: {batch_loss.item()}")
                # Backward pass - gradients accumulate across batches
                batch_loss.backward()

                # Clear intermediate tensors to save memory
                del image_stack_batch
                del batch_mean_stack
                del batch_std_stack
                del backend_kwargs
                del result
                del refined_scaled_mip
                del batch_loss
                del batch_refine_manager, batch_particle_stack
                torch.cuda.empty_cache()

            # Now take the optimizer step with accumulated gradients from all batches
            motion_optimizer.step()
            print(f"accumulated_loss: {accumulated_loss}")

            # log loss
            if iter_idx % 1 == 0:
                print(f"{iter_idx}: mean cc = {accumulated_loss}")

            if trajectory is not None and trajectory.sample_this_step(iter_idx):
                trajectory.add_checkpoint(
                    deformation_field=deformation_field.data,
                    loss=accumulated_loss,
                    step=iter_idx,
                )
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary batch configs at {temp_dir}")

    # Return final deformation field
    final_deformation_field = deformation_field.data
    average_shift = torch.mean(final_deformation_field, dim=(1, 2, 3), keepdim=True)
    final_deformation_field = final_deformation_field - average_shift

    if return_trajectory:
        return final_deformation_field, trajectory
    return final_deformation_field


def _setup_estimate_motion(
    refine_config_path: str,
    particle_indices: pd.Index | None,
    loss_metric: str,
    min_snr: float,
    best_n: int,
    temp_dir: Path | None,
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    image: torch.Tensor,
    optimizer_kwargs: dict[str, Any] | None,
    return_trajectory: bool,
    trajectory_kwargs: dict | None,
    initial_deformation_field: torch.Tensor | None,
    deformation_field_resolution: tuple[int, int, int],
    device: torch.device | None,
    requires_grad: bool = False,
    grid_type: Literal["catmull_rom", "bspline"] | None = None,
) -> dict[str, Any]:
    """Setup common parameters for motion estimation.

    Parameters
    ----------
    refine_config_path: str
        Path to the refine config file.
    particle_indices: pd.Index | None
        Particle indices to use for the refinement.
    loss_metric: str
        Loss metric to use for the refinement.
    min_snr: float
        Minimum SNR to use for the refinement.
    best_n: int
        Maximum number of particles to use for the refinement.
    temp_dir: Path | None
        Temporary directory for filtering. None for non-particles version.
    var_image: torch.Tensor
        (t, H, W) variance image.
    mean_image: torch.Tensor
        (t, H, W) mean image.
    image: torch.Tensor
        (t, H, W) image tensor.
    optimizer_kwargs: dict[str, Any] | None
        Keyword arguments for the optimizer.
    return_trajectory: bool
        Whether to return the optimization trajectory.
    trajectory_kwargs: dict | None
        Additional keyword arguments for the trajectory tracking.
    initial_deformation_field: torch.Tensor | None
        Initial deformation field to start from.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw).
    device: torch.device | None
        Device to perform computation on.
    requires_grad: bool
        Whether to require gradients on deformation_field_data. Default is False.
    grid_type: Literal["catmull_rom", "bspline"] | None
        Grid type to use. If None, uses catmull_rom. Default is None.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - refine_config_path: Updated refine config path
        - particle_indices: Updated particle indices
        - template_volume: Loaded template volume
        - var_image: Detached variance image
        - mean_image: Detached mean image
        - image: Detached image
        - optimizer_kwargs: Optimizer kwargs with defaults
        - trajectory: OptimizationTracker or None
        - deformation_field_data: Initialized/resampled deformation field data
        - deformation_field: Created deformation field grid (if grid_type provided)
    """
    # Filter particles by quality metrics
    refine_config_path, particle_indices = _filter_particles_by_quality(
        refine_config_path=refine_config_path,
        particle_indices=particle_indices,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        temp_dir=temp_dir,
    )

    template_volume = load_template_volume_from_config(refine_config_path)
    # Make sure var and mean image don't have gradients
    if var_image.requires_grad:
        var_image = var_image.clone().detach().requires_grad_(False)
    if mean_image.requires_grad:
        mean_image = mean_image.clone().detach().requires_grad_(False)
    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": 0.2}
    trajectory = None
    if return_trajectory:
        trajectory_kwargs = trajectory_kwargs if trajectory_kwargs is not None else {}
        trajectory = OptimizationTracker(**trajectory_kwargs)

    # Ensure image does NOT require gradients - only deformation field should
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)

    if initial_deformation_field is None:
        deformation_field_data = torch.zeros(
            size=(2, *deformation_field_resolution),
            device=device,
            requires_grad=requires_grad,
        )
    else:
        # Get the resampled deformation field
        deformation_field_data = resample_deformation_field(
            deformation_field=initial_deformation_field,
            target_resolution=(
                deformation_field_resolution[0],
                deformation_field_resolution[1],
                deformation_field_resolution[2],
            ),
        )
        deformation_field_data = deformation_field_data - torch.mean(
            deformation_field_data, dim=(1, 2, 3), keepdim=True
        )

        # Ensure gradients are enabled for optimization if requested
        if requires_grad:
            deformation_field_data = (
                deformation_field_data.clone().detach().requires_grad_(True)
            )

    result = {
        "refine_config_path": refine_config_path,
        "particle_indices": particle_indices,
        "template_volume": template_volume,
        "var_image": var_image,
        "mean_image": mean_image,
        "image": image,
        "optimizer_kwargs": optimizer_kwargs,
        "trajectory": trajectory,
        "deformation_field_data": deformation_field_data,
    }

    # Create deformation field if grid_type is provided
    if grid_type is not None:
        if grid_type == "catmull_rom":
            deformation_field = CubicCatmullRomGrid3d.from_grid_data(
                deformation_field_data
            ).to(device)
        elif grid_type == "bspline":
            deformation_field = CubicBSplineGrid3d.from_grid_data(
                deformation_field_data
            ).to(device)
        else:
            raise ValueError(
                f"Unknown grid_type: {grid_type}. Must be 'catmull_rom' or 'bspline'"
            )
        result["deformation_field"] = deformation_field
    else:
        # Default to catmull_rom if not specified
        deformation_field = CubicCatmullRomGrid3d.from_grid_data(
            deformation_field_data
        ).to(device)
        result["deformation_field"] = deformation_field

    return result


def _compute_loss(
    loss_tensor: torch.Tensor,
    prior_type: str,
    deformation_field: CubicCatmullRomGrid3d,
    batch_size: int,
    total_n_particles: int,
    image_coords: torch.Tensor | None = None,
    sigma_d: float | None = None,
    sigma_v_norm: float | None = None,
    sigma_a_norm: torch.Tensor | float | None = None,
    alpha_spatial: float | None = None,
    spatial_spacing: tuple[float, float] | None = None,
    temporal_spacing: float | None = None,
) -> torch.Tensor:
    """Compute loss with motion priors.

    Parameters
    ----------
    loss_tensor: torch.Tensor
        Loss tensor from refinement.
    prior_type: str
        Type of motion prior: "relion" or "laplacian".
    deformation_field: CubicCatmullRomGrid3d
        Deformation field grid.
    batch_size: int
        Size of current batch.
    total_n_particles: int
        Total number of particles across all batches.
    image_coords: torch.Tensor | None
        Physical coordinates for RELION prior. Required if prior_type is "relion".
    sigma_d: float | None
        Spatial correlation length for RELION prior. Required if prior_type is "relion".
    sigma_v_norm: float | None
        Normalized velocity magnitude scale for RELION prior.
        Required if prior_type is "relion".
    sigma_a_norm: torch.Tensor | float | None
        Normalized temporal smoothness parameter.
        Required for both prior types.
    alpha_spatial: float | None
        Spatial smoothness strength for Laplacian prior.
        Required if prior_type is "laplacian".
    spatial_spacing: tuple[float, float] | None
        Spatial spacing for Laplacian prior. Required if prior_type is "laplacian".
    temporal_spacing: float | None
        Temporal spacing for Laplacian prior. Required if prior_type is "laplacian".

    Returns
    -------
    torch.Tensor
        Computed loss value.
    """
    # Compute motion priors
    e_space = torch.tensor(0.0, device=deformation_field._data.device)
    e_time = torch.tensor(0.0, device=deformation_field._data.device)
    if prior_type == "relion":
        assert sigma_d is not None, "sigma_d is required for relion prior"
        assert sigma_v_norm is not None, "sigma_v_norm is required for relion prior"
        assert sigma_a_norm is not None, "sigma_a_norm is required for relion prior"
        # Convert tensor to float if needed
        sigma_a_val = (
            float(sigma_a_norm)
            if isinstance(sigma_a_norm, torch.Tensor)
            else sigma_a_norm
        )
        e_space, e_time = relion2019_compute(
            field=deformation_field._data,
            coords=image_coords,
            sigma_d=sigma_d,
            sigma_v=sigma_v_norm,
            sigma_a=sigma_a_val,
        )
    elif prior_type == "laplacian":
        assert sigma_a_norm is not None, "sigma_a_norm is required for laplacian prior"
        assert alpha_spatial is not None, (
            "alpha_spatial is required for laplacian prior"
        )
        # Convert tensor to float if needed
        sigma_a_val = (
            float(sigma_a_norm)
            if isinstance(sigma_a_norm, torch.Tensor)
            else sigma_a_norm
        )
        e_space, e_time = laplacian_compute(
            field=deformation_field._data,
            sigma_a=sigma_a_val,
            alpha=alpha_spatial,
            spatial_spacing=spatial_spacing,
            temporal_spacing=temporal_spacing,
        )

    e_space = e_space * batch_size / total_n_particles
    e_time = e_time * batch_size / total_n_particles

    # Compute loss for this batch (weighted by batch size for averaging)
    e_obs = -2 * torch.mean(loss_tensor) * batch_size / total_n_particles
    loss = e_obs + (e_space + e_time)
    print(f"e_obs: {e_obs.item()}")
    print(f"e_space: {e_space.item()}")
    print(f"e_time: {e_time.item()}")
    print(f"loss: {loss.item()}")

    return loss


def _setup_priors(
    prior_type: str,
    sigma_a_exponential: bool,
    sigma_a: float,
    sigma_a_amplitude: float,
    sigma_a_decay: float,
    sigma_a_offset: float,
    sigma_v: float,
    image: torch.Tensor,
    fluence_per_frame: float,
    deformation_field_resolution: tuple[int, int, int],
    pixel_spacing: float,
    device: torch.device,
) -> dict[str, Any]:
    """Setup prior-specific parameters for motion estimation.

    Parameters
    ----------
    prior_type: str
        Type of motion prior: "relion" or "laplacian".
    sigma_a_exponential: bool
        Whether to use exponential decay for sigma_a over frames.
    sigma_a: float
        Temporal smoothness parameter.
    sigma_a_amplitude: float
        Amplitude in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset.
    sigma_a_decay: float
        Decay rate in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset.
    sigma_a_offset: float
        Constant offset in exponential sigma_a formula:
        amplitude*exp(-decay_rate*fluence) + offset.
    sigma_v: float
        Velocity magnitude scale in Å per unit fluence for RELION prior.
    image: torch.Tensor
        (t, H, W) image tensor.
    fluence_per_frame: float
        Fluence per frame in electrons per pixel.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw).
    pixel_spacing: float
        Pixel spacing in Angstroms.
    device: torch.device
        Device to perform computation on.

    Returns
    -------
    dict[str, Any]
        Dictionary containing prior parameters:
        - sigma_A_norm: Normalized sigma_A (always present)
        - image_coords: Physical coordinates (only for relion prior)
        - sigma_V_norm: Normalized sigma_V (only for relion prior)
        - spatial_spacing: Spatial spacing (only for laplacian prior)
        - temporal_spacing: Temporal spacing (only for laplacian prior)
    """
    # Create fluence-dependent sigma_a if requested
    if sigma_a_exponential:
        total_fluence = fluence_per_frame * image.shape[0]
        sigma_a_tensor = _create_exponential_sigma_a(
            total_fluence=total_fluence,
            n_frames=deformation_field_resolution[0],
            amplitude=sigma_a_amplitude,
            decay_rate=sigma_a_decay,
            offset=sigma_a_offset,
            device=device,
        )
    else:
        sigma_a_tensor = sigma_a

    if prior_type == "relion":
        image_coords = _build_physical_coords(
            nh=deformation_field_resolution[1],
            nw=deformation_field_resolution[2],
            image_shape=image.shape[-2:],
            pixel_size=pixel_spacing,
            device=device,
        )
        # Normalize sigma parameters by fluence
        sigma_v_norm = _normalize_sigma_fluence(
            sigma_v,
            fluence_per_frame * image.shape[0],
            deformation_field_resolution[0],
        )
        if not sigma_a_exponential:
            sigma_a_norm = _normalize_sigma_fluence(
                sigma_a,
                fluence_per_frame * image.shape[0],
                deformation_field_resolution[0],
            )
        else:
            # Normalize the exponential tensor
            sigma_a_norm = _normalize_sigma_fluence(
                sigma_a_tensor,
                fluence_per_frame * image.shape[0],
                deformation_field_resolution[0],
            )
        return {
            "sigma_a_norm": sigma_a_norm,
            "image_coords": image_coords,
            "sigma_v_norm": sigma_v_norm,
        }
    if prior_type == "laplacian":
        # Compute physical spacing for Laplacian prior
        spatial_spacing, temporal_spacing = _compute_physical_spacing(
            image_shape=image.shape[-2:],
            pixel_size=pixel_spacing,
            grid_resolution=deformation_field_resolution,
            total_fluence=fluence_per_frame * image.shape[0],
        )
        # For Laplacian, use sigma_a_tensor directly (no normalization needed)
        sigma_a_norm = sigma_a_tensor
        return {
            "sigma_a_norm": sigma_a_norm,
            "spatial_spacing": spatial_spacing,
            "temporal_spacing": temporal_spacing,
        }
    raise ValueError(
        f"Unknown prior_type: {prior_type}. Must be 'relion' or 'laplacian'"
    )
