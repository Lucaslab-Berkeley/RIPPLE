"""Core function for polishing particles using L-BFGS optimizer."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import einops
import pandas as pd
import torch
import tqdm
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_motion_correction import correct_motion, correct_motion_two_grids
from torch_motion_correction.data_io import write_deformation_field_to_csv
from torch_motion_correction.optimization_state import OptimizationTracker

from .core_polish_particles import (
    _compute_loss,
    _setup_estimate_motion,
    _setup_priors,
)
from .core_utils import (
    _create_batch_configs,
    _get_particle_coordinates,
    _make_differentiable_refine_manager,
    compute_particle_shifts_from_deformation_field,
    get_batch_mean_std_stacks,
)
from .generate_image import dose_weight_memory_efficient
from .motion_priors import relion2019_eigendecompose
from .prepare_movie import prepare_core


def _check_convergence(
    loss_history: list[float],
    convergence_threshold: float,
    num_convergence_iterations: int,
) -> tuple[bool, list[float]]:
    """Check if optimization has converged based on loss history.

    Parameters
    ----------
    loss_history : list[float]
        List of loss values from recent iterations.
    convergence_threshold : float
        Threshold for absolute change in loss to consider converged.
    num_convergence_iterations : int
        Number of consecutive iterations where change must be below threshold.

    Returns
    -------
    tuple[bool, list[float]]
        (converged, trimmed_loss_history)
    """
    min_history_len = num_convergence_iterations + 1
    if len(loss_history) < min_history_len:
        return False, loss_history

    # Check if absolute change has been less than threshold
    # for the last num_convergence_iterations consecutive iterations
    converged = True
    hist_len = len(loss_history)
    start_idx = hist_len - num_convergence_iterations
    for j in range(start_idx, hist_len - 1):
        abs_change = abs(loss_history[j + 1] - loss_history[j])
        if abs_change > convergence_threshold:
            converged = False
            break

    # Keep only last (num_convergence_iterations + 1) values for efficiency
    trimmed_history = loss_history[-min_history_len:]
    return converged, trimmed_history


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements,unused-argument
def core_polish_particles_lbfgs(
    movie: torch.Tensor,  # (t, H, W)
    initial_deformation_field: torch.Tensor | None,
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
    optimizer_kwargs: dict[str, Any] | None = None,  # Not used for L-BFGS, kept for API compatibility
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
    optimization_mode: Literal["deformation_field", "particle_shifts"] = "deformation_field",
    initial_particle_shifts: torch.Tensor | None = None,  # (T, N, 2) if mode is particle_shifts
    # L-BFGS specific parameters
    learning_rate: float = 0.2,
    lbfgs_max_eval: int = 5,
    lbfgs_line_search_fn: str | None = "strong_wolfe",
    convergence_threshold: float = 0.0005,
    num_convergence_iterations: int = 5,
) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor], torch.Tensor, OptimizationTracker | None]:
    """
    Core function for polishing particles using L-BFGS optimizer.

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
        Keyword arguments for the optimizer. Not used for L-BFGS, kept for API
        compatibility. Default is None.
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
    optimization_mode: Literal["deformation_field", "particle_shifts"]
        Optimization mode. If "deformation_field", optimizes a deformation field grid.
        If "particle_shifts", optimizes particle shifts directly (T, N, 2).
        Default is "deformation_field".
    initial_particle_shifts: torch.Tensor | None
        Initial particle shifts with shape (T, N, 2) where T is number of frames
        and N is number of particles. Only used if optimization_mode is "particle_shifts".
        If None, initializes to zero shifts.
    learning_rate: float
        Learning rate for L-BFGS. Default is 0.2.
    lbfgs_max_eval: int
        Maximum closure evaluations per L-BFGS step. Default is 5.
    lbfgs_line_search_fn: str | None
        Line search function: "strong_wolfe" or None. Default is "strong_wolfe".
    convergence_threshold: float
        Threshold for convergence checking. Default is 0.0005.
    num_convergence_iterations: int
        Number of consecutive iterations for convergence. Default is 5.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor | dict, torch.Tensor, OptimizationTracker | None]
        - Corrected movie (t, H, W)
        - Updated deformation field (2, nt, nh, nw) or dict with "particle_shifts" key
        - Movie prepared (t, H, W)
        - Optimization trajectory (OptimizationTracker) or None
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

    # Prepare common kwargs for estimation functions
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
    # Only add optimization_mode and initial_particle_shifts for particle-based estimation
    if movie_extract:
        estimate_kwargs["optimization_mode"] = optimization_mode
        estimate_kwargs["initial_particle_shifts"] = initial_particle_shifts
    # Prior parameters
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
    # L-BFGS specific parameters
    lbfgs_kwargs: dict[str, Any] = {
        "learning_rate": learning_rate,
        "lbfgs_max_eval": lbfgs_max_eval,
        "lbfgs_line_search_fn": lbfgs_line_search_fn,
        "convergence_threshold": convergence_threshold,
        "num_convergence_iterations": num_convergence_iterations,
    }

    # estimate the motion
    if loss_trajectories:
        if movie_extract:
            result, trajectory = estimate_local_motion_2dtm_particles_lbfgs(
                **estimate_kwargs,
                **prior_kwargs,
                **lbfgs_kwargs,
                particle_batch_size=particle_batch_size,
                pixel_spacing=pixel_size,
            )
        else:
            result, trajectory = estimate_local_motion_2dtm_lbfgs(
                **estimate_kwargs,
                **prior_kwargs,
                **lbfgs_kwargs,
                pixel_spacing=pixel_size,
                grid_type=grid_type,
                voltage=voltage,
            )
    else:
        if movie_extract:
            result = estimate_local_motion_2dtm_particles_lbfgs(
                **estimate_kwargs,
                **prior_kwargs,
                **lbfgs_kwargs,
                particle_batch_size=particle_batch_size,
                pixel_spacing=pixel_size,
            )
        else:
            result = estimate_local_motion_2dtm_lbfgs(
                **estimate_kwargs,
                **prior_kwargs,
                **lbfgs_kwargs,
                pixel_spacing=pixel_size,
                grid_type=grid_type,
                voltage=voltage,
            )
        trajectory = None

    # Extract result based on mode
    if optimization_mode == "particle_shifts":
        if isinstance(result, dict):
            updated_particle_shifts = result["particle_shifts"]
        else:
            updated_particle_shifts = result
        # For particle_shifts, we don't correct motion at the movie level
        # The shifts are applied per-particle during extraction
        corrected_movie = movie_prepared
        # Return particle_shifts as dict for consistency
        updated_deformation_field = {"particle_shifts": updated_particle_shifts}
    else:
        updated_deformation_field = result
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
def estimate_local_motion_2dtm_lbfgs(
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
    # L-BFGS specific parameters
    learning_rate: float = 0.2,
    lbfgs_max_eval: int = 5,
    lbfgs_line_search_fn: str | None = "strong_wolfe",
    convergence_threshold: float = 0.0005,
    num_convergence_iterations: int = 5,
) -> torch.Tensor | tuple[torch.Tensor, OptimizationTracker]:
    """Estimate motion using L-BFGS optimizer (non-particle version).

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
    n_iterations: int
        Number of iterations for the optimization process. Default is 100.
    grid_type: Literal["catmull_rom", "bspline"]
        Grid type to use for the deformation field. Default is "catmull_rom".
    return_trajectory: bool
        Whether to return the optimization trajectory. Default is False.
    trajectory_kwargs: dict | None
        Additional keyword arguments for the trajectory tracking.
    correlation_batch_size: int
        Batch size for the correlation. Default is 20.
    voltage: float
        Voltage in kV. Default is 300.0.
    particle_indices: pd.Index
        Particle indices to use for the refinement. If None, uses all particles.
    device: torch.device
        Device to perform computation on. If None, uses the device of the input image.
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
    learning_rate: float
        Learning rate for L-BFGS. Default is 0.2.
    lbfgs_max_eval: int
        Maximum closure evaluations per L-BFGS step. Default is 5.
    lbfgs_line_search_fn: str | None
        Line search function: "strong_wolfe" or None. Default is "strong_wolfe".
    convergence_threshold: float
        Threshold for convergence checking. Default is 0.0005.
    num_convergence_iterations: int
        Number of consecutive iterations for convergence. Default is 5.

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
        optimizer_kwargs=None,  # Not used for L-BFGS
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
    trajectory = setup_result["trajectory"]
    deformation_field = setup_result["deformation_field"]
    deformation_field_data = setup_result["deformation_field_data"]

    print("Making new deformation field")
    new_deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution, n_channels=2
    ).to(device)
    # Ensure parameters are contiguous for L-BFGS
    for param in new_deformation_field.parameters():
        param.data = param.data.contiguous()
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
        optimization_mode="deformation_field",
        n_particles=None,
    )
    image_coords = prior_params.get("image_coords")
    sigma_v_norm = prior_params.get("sigma_v_norm")
    sigma_a_norm = prior_params["sigma_a_norm"]
    spatial_spacing = prior_params.get("spatial_spacing")
    temporal_spacing = prior_params.get("temporal_spacing")

    # Pre-compute eigendecomposition for RELION prior
    relion_lam = None
    relion_vecs = None
    if prior_type == "relion":
        assert image_coords is not None
        relion_lam, relion_vecs, _ = relion2019_eigendecompose(
            image_coords,
            sigma_d,
            sigma_v_norm,
            variance_threshold=0.999,  # Keep modes accounting for 99.9% of variance
        )

    # Create L-BFGS optimizer
    motion_optimizer = torch.optim.LBFGS(
        params=new_deformation_field.parameters(),
        lr=learning_rate,
        max_eval=lbfgs_max_eval,
        line_search_fn=lbfgs_line_search_fn,
    )

    # Training loop
    pbar = tqdm.tqdm(range(n_iterations))
    loss_history: list[float] = []

    for iter_idx in pbar:
        if save_intermediate_fields:
            write_deformation_field_to_csv(
                new_deformation_field.data,
                f"{intermediate_fields_dir}/new_deformation_field_{iter_idx}.csv",
            )
        torch.cuda.empty_cache()

        # Track closure calls for this iteration
        closure_call_count = [0]
        accumulated_loss_for_display = [0.0]

        # Define closure for L-BFGS
        def closure() -> torch.Tensor:
            closure_call_count[0] += 1
            motion_optimizer.zero_grad()

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
            optimization_var = {
                "variable": deformation_field,
                "type": "deformation_field",
                "data": deformation_field._data,
            }
            loss = _compute_loss(
                loss_tensor=loss_tensor,
                prior_type=prior_type,
                optimization_var=optimization_var,
                batch_size=1,
                total_n_particles=1,
                image_coords=image_coords,
                particle_coords=None,
                sigma_d=sigma_d,
                sigma_v_norm=sigma_v_norm,
                sigma_a_norm=sigma_a_norm,
                alpha_spatial=alpha_spatial,
                spatial_spacing=spatial_spacing,
                temporal_spacing=temporal_spacing,
                relion_lam=relion_lam,
                relion_vecs=relion_vecs,
            )

            accumulated_loss_for_display[0] = loss.item()
            loss.backward()
            return loss

        # Run L-BFGS step
        motion_optimizer.step(closure)

        # Log progress
        current_loss = accumulated_loss_for_display[0]
        pbar.set_description(
            f"Loss: {current_loss:.6f} (closures: {closure_call_count[0]})"
        )
        print(
            f"{iter_idx}: loss = {current_loss:.6f}, "
            f"closure calls = {closure_call_count[0]}"
        )

        # Check for convergence
        loss_history.append(current_loss)
        converged, loss_history = _check_convergence(
            loss_history, convergence_threshold, num_convergence_iterations
        )
        if converged:
            print(
                f"Converged at iteration {iter_idx}: loss change below "
                f"{convergence_threshold} for {num_convergence_iterations} iterations"
            )
            break

        if trajectory is not None and trajectory.sample_this_step(iter_idx):
            trajectory.add_checkpoint(
                deformation_field=deformation_field_data,
                loss=current_loss,
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
def estimate_local_motion_2dtm_particles_lbfgs(
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
    optimization_mode: Literal["deformation_field", "particle_shifts"] = "deformation_field",
    initial_particle_shifts: torch.Tensor | None = None,
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom",
    # L-BFGS specific parameters
    learning_rate: float = 0.2,
    lbfgs_max_eval: int = 5,
    lbfgs_line_search_fn: str | None = "strong_wolfe",
    convergence_threshold: float = 0.0005,
    num_convergence_iterations: int = 5,
) -> (
    torch.Tensor
    | dict[str, torch.Tensor]
    | tuple[torch.Tensor, OptimizationTracker]
    | tuple[dict[str, torch.Tensor], OptimizationTracker]
):
    """Estimate motion using L-BFGS optimizer.

    This is similar to estimate_local_motion_2dtm_particles_bayesian but uses
    L-BFGS optimizer instead of Adam. L-BFGS can converge faster for smooth
    optimization landscapes but requires a closure function.

    Parameters
    ----------
    image : torch.Tensor
        (t, H, W) image to estimate motion from.
    var_image : torch.Tensor
        (t, H, W) variance image.
    mean_image : torch.Tensor
        (t, H, W) mean image.
    pixel_spacing : float
        Pixel spacing in Angstroms.
    deformation_field_resolution : tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw).
    initial_deformation_field : torch.Tensor | None
        Initial deformation field (2, nt, nh, nw). If None, initializes to zero.
    refine_config_path : str
        Path to the refine config file.
    pre_exposure : float
        Pre-exposure in electrons per Angstrom squared. Default is 0.0.
    fluence_per_frame : float
        Fluence per frame in electrons per Angstrom squared. Default is 1.0.
    n_iterations : int
        Maximum number of outer iterations. Default is 100.
    return_trajectory : bool
        Whether to return optimization trajectory. Default is False.
    trajectory_kwargs : dict | None
        Keyword arguments for trajectory tracking.
    correlation_batch_size : int
        Batch size for correlation computation. Default is 20.
    particle_batch_size : int
        Number of particles per batch. Default is 102.
    particle_indices : pd.Index
        Particle indices to use.
    device : torch.device
        Device to use for computation.
    loss_metric : str
        Loss metric: "mip" or "scaled_mip". Default is "scaled_mip".
    min_snr : float
        Minimum SNR for particle filtering. Default is 0.0.
    best_n : int
        Maximum number of particles to use. Default is 10000000000.
    save_intermediate_fields : bool
        Whether to save intermediate deformation fields. Default is False.
    intermediate_fields_dir : str
        Directory for intermediate fields. Default is ".".
    prior_type : str
        Prior type: "relion" or "laplacian". Default is "relion".
    sigma_d : float
        Spatial correlation length (RELION). Default is 5782.376953.
    sigma_v : float
        Velocity magnitude scale (RELION). Default is 0.194826.
    sigma_a : float
        Temporal smoothness. Default is 0.513517.
    alpha_spatial : float
        Spatial smoothness (Laplacian). Default is 1e5.
    sigma_a_exponential : bool
        Use exponential sigma_a decay. Default is False.
    sigma_a_amplitude : float
        Amplitude for exponential sigma_a. Default is 2.0.
    sigma_a_decay : float
        Decay rate for exponential sigma_a. Default is 0.1.
    sigma_a_offset : float
        Offset for exponential sigma_a. Default is 1.0.
    optimization_mode : Literal["deformation_field", "particle_shifts"]
        Optimization mode. Default is "deformation_field".
    initial_particle_shifts : torch.Tensor | None
        Initial particle shifts (T, N, 2). Default is None.
    grid_type : Literal["catmull_rom", "bspline"]
        Grid type for deformation field. Default is "catmull_rom".
    learning_rate : float
        Learning rate for L-BFGS. Default is 0.2.
    lbfgs_max_eval : int
        Maximum closure evaluations per L-BFGS step. Default is 5.
    lbfgs_line_search_fn : str | None
        Line search function: "strong_wolfe" or None. Default is "strong_wolfe".
    convergence_threshold : float
        Threshold for convergence checking. Default is 0.0005.
    num_convergence_iterations : int
        Number of consecutive iterations for convergence. Default is 5.

    Returns
    -------
    torch.Tensor | dict[str, torch.Tensor] | tuple
        Final deformation field or particle shifts, optionally with trajectory.
    """
    torch.set_grad_enabled(True)

    # Create temporary directory for filtering and batch configs
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_batch_lbfgs_"))

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
        optimizer_kwargs=None,  # Not used for L-BFGS
        return_trajectory=return_trajectory,
        trajectory_kwargs=trajectory_kwargs,
        initial_deformation_field=initial_deformation_field,
        deformation_field_resolution=deformation_field_resolution,
        device=device,
        requires_grad=True,
        grid_type=grid_type,
    )
    refine_config_path = setup_result["refine_config_path"]
    particle_indices = setup_result["particle_indices"]
    template_volume = setup_result["template_volume"]
    var_image = setup_result["var_image"]
    mean_image = setup_result["mean_image"]
    image = setup_result["image"]
    trajectory = setup_result["trajectory"]
    grid_type_from_setup = setup_result.get("grid_type", "catmull_rom")

    # Create batch configs once before optimization loop
    batch_config_paths, batch_particle_indices = _create_batch_configs(
        refine_config_path=refine_config_path,
        particle_batch_size=particle_batch_size,
        temp_dir=temp_dir,
    )

    # Calculate total number of particles across all batches
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)

    # Initialize optimization variable based on mode
    if optimization_mode == "particle_shifts":
        if initial_particle_shifts is None:
            if initial_deformation_field is not None:
                particle_shifts = compute_particle_shifts_from_deformation_field(
                    deformation_field=initial_deformation_field,
                    movie=image,
                    refine_config_path=refine_config_path,
                    pixel_spacing=pixel_spacing,
                    grid_type=grid_type_from_setup,
                    device=device,
                    particle_indices=batch_particle_indices,
                )
                particle_shifts = particle_shifts.detach().requires_grad_(True)
            else:
                particle_shifts = torch.zeros(
                    (image.shape[0], total_n_particles, 2),
                    device=device,
                    requires_grad=True,
                )
        else:
            if initial_particle_shifts.shape[1] != total_n_particles:
                raise ValueError(
                    f"initial_particle_shifts shape[1] ({initial_particle_shifts.shape[1]}) "
                    f"does not match total_n_particles ({total_n_particles})"
                )
            particle_shifts = initial_particle_shifts.to(device).requires_grad_(True)

        optimization_var = {
            "variable": particle_shifts,
            "type": "particle_shifts",
            "optimizer_params": [particle_shifts],
            "data": particle_shifts,
        }

        # Get particle coordinates for priors
        refine_manager = _make_differentiable_refine_manager(refine_config_path)
        particle_stack = refine_manager.particle_stack
        particle_coords = _get_particle_coordinates(
            particle_stack=particle_stack,
            particle_indices=batch_particle_indices,
            pixel_spacing=pixel_spacing,
            device=device,
        )
    else:  # deformation_field
        deformation_field = setup_result["deformation_field"]
        # Ensure parameters are contiguous for L-BFGS
        for param in deformation_field.parameters():
            param.data = param.data.contiguous()
        optimization_var = {
            "variable": deformation_field,
            "type": "deformation_field",
            "optimizer_params": list(deformation_field.parameters()),
            "data": deformation_field._data,
        }
        particle_coords = None

    # Create L-BFGS optimizer
    # Note: max_iter is set by user from config (n_iterations parameter)
    motion_optimizer = torch.optim.LBFGS(
        params=optimization_var["optimizer_params"],
        lr=learning_rate,
        max_eval=lbfgs_max_eval,
        line_search_fn=lbfgs_line_search_fn,
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
        deformation_field_resolution=(
            deformation_field_resolution if optimization_mode == "deformation_field" else None
        ),
        pixel_spacing=pixel_spacing,
        device=device,
        optimization_mode=optimization_mode,
        n_particles=total_n_particles if optimization_mode == "particle_shifts" else None,
    )
    image_coords = prior_params.get("image_coords")
    if optimization_mode == "particle_shifts" and prior_type == "relion":
        image_coords = particle_coords
    sigma_v_norm = prior_params.get("sigma_v_norm")
    sigma_a_norm = prior_params["sigma_a_norm"]
    spatial_spacing = prior_params.get("spatial_spacing")
    temporal_spacing = prior_params.get("temporal_spacing")

    # Pre-compute mean/std stacks for all batches
    batch_mean_stacks, batch_std_stacks = get_batch_mean_std_stacks(
        batch_config_paths=batch_config_paths,
        batch_particle_indices=batch_particle_indices,
        mean_image=mean_image,
        var_image=var_image,
    )

    # Pre-compute eigendecomposition for RELION prior
    relion_lam = None
    relion_vecs = None
    if prior_type == "relion":
        if optimization_mode == "particle_shifts":
            assert particle_coords is not None
            coords_for_eigen = particle_coords
        else:
            assert image_coords is not None
            coords_for_eigen = image_coords

        relion_lam, relion_vecs, _ = relion2019_eigendecompose(
            coords_for_eigen,
            sigma_d,
            sigma_v_norm,
            variance_threshold=0.999,  # Keep modes accounting for 99.9% of variance
        )

    # Training loop
    pbar = tqdm.tqdm(range(n_iterations))
    loss_history: list[float] = []

    try:
        for iter_idx in pbar:
            # Save intermediate fields if requested (only for deformation_field mode)
            if save_intermediate_fields and optimization_mode == "deformation_field":
                write_deformation_field_to_csv(
                    optimization_var["variable"].data,
                    f"{intermediate_fields_dir}/particle_deformation_field_{iter_idx}.csv",
                )

            torch.cuda.empty_cache()

            # Track closure calls for this iteration
            closure_call_count = [0]
            accumulated_loss_for_display = [0.0]

            # Define closure for L-BFGS
            def closure() -> torch.Tensor:
                closure_call_count[0] += 1
                motion_optimizer.zero_grad()
                accumulated_loss = 0.0

                # Process particles in batches
                for batch_config_path, batch_indices in zip(
                    batch_config_paths, batch_particle_indices, strict=True
                ):
                    batch_refine_manager = _make_differentiable_refine_manager(
                        refine_config_path=batch_config_path,
                    )
                    batch_particle_stack = batch_refine_manager.particle_stack
                    batch_size = len(batch_indices[0])

                    # Extract particle images for this batch
                    if optimization_mode == "deformation_field":
                        image_stack_batch = (
                            batch_particle_stack.construct_image_stack_from_movie(
                                movie=image,
                                deformation_field=optimization_var["variable"],
                                pos_reference="top-left",
                                handle_bounds="pad",
                                padding_mode="reflect",
                                padding_value=0.0,
                                pre_exposure=pre_exposure,
                                fluence_per_frame=fluence_per_frame,
                            )
                        )
                    else:  # particle_shifts
                        batch_start_idx = sum(
                            len(batch_particle_indices[i][0])
                            for i in range(batch_config_paths.index(batch_config_path))
                        )
                        batch_end_idx = batch_start_idx + batch_size
                        batch_particle_shifts = optimization_var["variable"][
                            :, batch_start_idx:batch_end_idx, :
                        ]

                        image_stack_batch = (
                            batch_particle_stack.construct_image_stack_from_movie(
                                movie=image,
                                particle_shifts=batch_particle_shifts,
                                pos_reference="top-left",
                                handle_bounds="pad",
                                padding_mode="reflect",
                                padding_value=0.0,
                                pre_exposure=pre_exposure,
                                fluence_per_frame=fluence_per_frame,
                            )
                        )

                    batch_mean_stack = batch_mean_stacks[batch_config_path]
                    batch_std_stack = batch_std_stacks[batch_config_path]

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

                    if loss_metric == "mip":
                        loss_tensor = result["refined_cross_correlation"]
                    elif loss_metric == "scaled_mip":
                        loss_tensor = result["refined_z_score"]
                    else:
                        raise ValueError(f"Unknown loss_metric: {loss_metric}")

                    if optimization_mode == "particle_shifts":
                        batch_optimization_var = optimization_var.copy()
                    else:
                        batch_optimization_var = optimization_var

                    batch_loss = _compute_loss(
                        loss_tensor=loss_tensor,
                        prior_type=prior_type,
                        optimization_var=batch_optimization_var,
                        batch_size=batch_size,
                        total_n_particles=total_n_particles,
                        image_coords=image_coords,
                        particle_coords=particle_coords,
                        sigma_d=sigma_d,
                        sigma_v_norm=sigma_v_norm,
                        sigma_a_norm=sigma_a_norm,
                        alpha_spatial=alpha_spatial,
                        spatial_spacing=spatial_spacing,
                        temporal_spacing=temporal_spacing,
                        relion_lam=relion_lam,
                        relion_vecs=relion_vecs,
                    )

                    accumulated_loss += batch_loss.item()
                    print(f"batch_loss: {batch_loss.item()}")
                    batch_loss.backward()

                    # Clean up
                    del image_stack_batch, backend_kwargs, result, batch_loss
                    del batch_refine_manager, batch_particle_stack
                    torch.cuda.empty_cache()

                accumulated_loss_for_display[0] = accumulated_loss

                # Return loss tensor for L-BFGS
                return torch.tensor(
                    accumulated_loss, device=device, requires_grad=False
                )

            # Run L-BFGS step
            motion_optimizer.step(closure)

            # Log progress
            current_loss = accumulated_loss_for_display[0]
            pbar.set_description(
                f"Loss: {current_loss:.6f} (closures: {closure_call_count[0]})"
            )
            print(
                f"{iter_idx}: loss = {current_loss:.6f}, "
                f"closure calls = {closure_call_count[0]}"
            )

            # Check for convergence
            loss_history.append(current_loss)
            converged, loss_history = _check_convergence(
                loss_history, convergence_threshold, num_convergence_iterations
            )
            if converged:
                print(
                    f"Converged at iteration {iter_idx}: loss change below "
                    f"{convergence_threshold} for {num_convergence_iterations} iterations"
                )
                break

            # Add trajectory checkpoint
            if trajectory is not None and trajectory.sample_this_step(iter_idx):
                if optimization_mode == "deformation_field":
                    trajectory.add_checkpoint(
                        deformation_field=optimization_var["variable"].data,
                        loss=current_loss,
                        step=iter_idx,
                    )

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary batch configs at {temp_dir}")

    # Return final result based on mode
    if optimization_mode == "deformation_field":
        final_deformation_field = optimization_var["variable"].data
        average_shift = torch.mean(final_deformation_field, dim=(1, 2, 3), keepdim=True)
        final_deformation_field = final_deformation_field - average_shift
        if return_trajectory:
            return final_deformation_field, trajectory
        return final_deformation_field
    else:  # particle_shifts
        final_particle_shifts = optimization_var["variable"].detach()
        result_dict = {"particle_shifts": final_particle_shifts}
        if return_trajectory:
            return result_dict, trajectory
        return result_dict
