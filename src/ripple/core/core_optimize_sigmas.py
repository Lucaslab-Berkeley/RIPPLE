"""Core functions for optimizing sigma hyperparameters."""

import gc
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
import torch
from scipy.optimize import minimize
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_motion_correction.deformation_field_utils import (
    resample_deformation_field,
)

from ripple.utils.data_io import (
    load_template_volume_from_config,
    save_optimize_sigmas_to_json,
)

from .core_utils import (
    _create_batch_configs,
    _filter_particles_by_quality,
    _get_particle_coordinates,
    _make_differentiable_refine_manager,
    compute_particle_shifts_from_deformation_field,
    get_batch_mean_std_stacks,
)
from .motion_priors import (
    _build_physical_coords,
    _compute_physical_spacing,
    _create_exponential_sigma_a,
    _normalize_sigma_fluence,
    laplacian_compute,
    relion2019_compute,
)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def core_optimize_sigmas(
    optimize_algorithm: Literal["nelder-mead", "bayesian"],
    image: torch.Tensor,  # (t, H, W)
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    refine_config_path: str,
    optimize_particle_df_path: str | None = None,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    motion_iterations: int = 10,
    sigma_iterations: int = 20,
    optimizer_kwargs: dict[str, Any] | None = None,
    correlation_batch_size: int = 20,
    particle_batch_size: int = 102,
    particle_indices: pd.Index = None,
    device: torch.device = None,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    prior_type: str = "relion",
    init_sigma_a: float = 0.513517,
    init_alpha_spatial: float = 1e5,
    init_sigma_a_amplitude: float = 2.0,
    init_sigma_a_decay: float = 0.1,
    init_sigma_a_offset: float = 1.0,
    sigma_a_exponential: bool = False,
    init_sigma_d: float = 5782.376953,
    init_sigma_v: float = 0.194826,
    optimize_sigma_a: bool = True,
    optimize_alpha_spatial: bool = True,
    optimize_sigma_a_amplitude: bool = True,
    optimize_sigma_a_decay: bool = True,
    optimize_sigma_a_offset: bool = True,
    optimize_sigma_d: bool = True,
    optimize_sigma_v: bool = True,
    verbose: bool = True,
    # Output paths for saving results
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
    optimization_mode: Literal["deformation_field", "particle_shifts"] = "deformation_field",
    initial_particle_shifts: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Dispatcher function to run the appropriate sigma optimization algorithm.

    Parameters
    ----------
    optimize_algorithm : Literal["nelder-mead", "bayesian"]
        Algorithm to use for sigma optimization:
        - 'nelder-mead': Nelder-Mead (simplex) method
        - 'bayesian': Bayesian optimization using Optuna
    image : torch.Tensor
        (t, H, W) movie to estimate motion from
    var_image : torch.Tensor
        (t, H, W) variance image
    mean_image : torch.Tensor
        (t, H, W) mean image
    pixel_spacing : float
        Pixel spacing in Angstroms
    deformation_field_resolution : tuple[int, int, int]
        Resolution of deformation field (nt, nh, nw)
    initial_deformation_field : torch.Tensor | None
        Initial deformation field (2, nt, nh, nw) or None
    refine_config_path : str
        Path to refine config (training template)
    optimize_particle_df_path : str | None
        Path to particle dataframe config for validation loss computation.
        The validation template will be loaded from the template_volume_path
        in this YAML file. If None, uses the same particle dataframe and
        template as the motion loop.
        Default is None.
    pre_exposure : float
        Pre-exposure time in seconds. Default 0.0
    fluence_per_frame : float
        Fluence per frame in e/Å². Default 1.0
    motion_iterations : int
        Motion optimization iterations per sigma update. Default 10
    sigma_iterations : int
        Number of sigma optimization iterations/trials. Default 20
    optimizer_kwargs : dict
        Kwargs for motion optimizer. Default {"lr": 0.2}
    correlation_batch_size : int
        Batch size for correlation computation. Default 20
    particle_batch_size : int
        Batch size for particles. Default 102
    particle_indices : pd.Index
        Particle indices to use. Default None (all particles)
    device : torch.device
        Device to use. Default None (auto-detect)
    loss_metric : str
        Loss metric: "scaled_mip" or "cross_correlation". Default "scaled_mip"
    min_snr : float
        Minimum SNR threshold. Default 0.0
    best_n : int
        Maximum number of best particles to use. Default 10000000000
    prior_type : str
        "laplacian" or "relion". Default "relion"
    init_sigma_a : float
        Initial sigma_a (constant mode). Default 0.513517
    init_alpha_spatial : float
        Initial alpha_spatial. Default 1e5
    init_sigma_a_amplitude : float
        Initial amplitude in exponential. Default 2.0
    init_sigma_a_decay : float
        Initial decay rate in exponential. Default 0.1
    init_sigma_a_offset : float
        Initial offset in exponential. Default 1.0
    sigma_a_exponential : bool
        Use exponential sigma_a. Default False
    init_sigma_d : float
        Initial sigma_d. Default 5782.376953
    init_sigma_v : float
        Initial sigma_v. Default 0.194826
    optimize_sigma_a : bool
        Whether to optimize sigma_a. Default True
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial. Default True
    optimize_sigma_a_amplitude : bool
        Whether to optimize sigma_a_amplitude. Default True
    optimize_sigma_a_decay : bool
        Whether to optimize sigma_a_decay. Default True
    optimize_sigma_a_offset : bool
        Whether to optimize sigma_a_offset. Default True
    optimize_sigma_d : bool
        Whether to optimize sigma_d. Default True
    optimize_sigma_v : bool
        Whether to optimize sigma_v. Default True
    verbose : bool
        Print progress. Default True
    optimized_sigmas_output_path : str | None
        Path to save final optimized sigmas as JSON. Default None
    sigma_history_output_path : str | None
        Path to save sigma history (all iterations) as JSON. Default None
    training_history_output_path : str | None
        Path to save training loss history as JSON. Default None
    validation_history_output_path : str | None
        Path to save validation loss history as JSON. Default None
    optimization_mode : Literal["deformation_field", "particle_shifts"]
        Optimization mode. If "deformation_field", optimizes a deformation field grid.
        If "particle_shifts", optimizes particle shifts directly (T, N, 2).
        Default is "deformation_field".
    initial_particle_shifts : torch.Tensor | None
        Initial particle shifts with shape (T, N, 2) where T is number of frames
        and N is number of particles. Only used if optimization_mode is "particle_shifts".
        If None, initializes to zero shifts. Default is None.

    Returns
    -------
    dict
        - "optimized_sigmas": dict of optimized sigma values
        - "final_deformation_field": final deformation field or dict with "particle_shifts"
        - "validation_loss_history": list of validation losses
        - "training_loss_history": list of training losses
        - "sigma_history": list of sigma values at each iteration
    """
    common_kwargs = {
        "image": image,
        "var_image": var_image,
        "mean_image": mean_image,
        "pixel_spacing": pixel_spacing,
        "deformation_field_resolution": deformation_field_resolution,
        "initial_deformation_field": initial_deformation_field,
        "refine_config_path": refine_config_path,
        "optimize_particle_df_path": optimize_particle_df_path,
        "pre_exposure": pre_exposure,
        "fluence_per_frame": fluence_per_frame,
        "motion_iterations": motion_iterations,
        "optimizer_kwargs": optimizer_kwargs,
        "correlation_batch_size": correlation_batch_size,
        "particle_batch_size": particle_batch_size,
        "particle_indices": particle_indices,
        "device": device,
        "loss_metric": loss_metric,
        "min_snr": min_snr,
        "best_n": best_n,
        "prior_type": prior_type,
        "init_sigma_a": init_sigma_a,
        "init_alpha_spatial": init_alpha_spatial,
        "init_sigma_a_amplitude": init_sigma_a_amplitude,
        "init_sigma_a_decay": init_sigma_a_decay,
        "init_sigma_a_offset": init_sigma_a_offset,
        "sigma_a_exponential": sigma_a_exponential,
        "init_sigma_d": init_sigma_d,
        "init_sigma_v": init_sigma_v,
        "optimize_sigma_a": optimize_sigma_a,
        "optimize_alpha_spatial": optimize_alpha_spatial,
        "optimize_sigma_a_amplitude": optimize_sigma_a_amplitude,
        "optimize_sigma_a_decay": optimize_sigma_a_decay,
        "optimize_sigma_a_offset": optimize_sigma_a_offset,
        "optimize_sigma_d": optimize_sigma_d,
        "optimize_sigma_v": optimize_sigma_v,
        "verbose": verbose,
        "optimized_sigmas_output_path": optimized_sigmas_output_path,
        "sigma_history_output_path": sigma_history_output_path,
        "training_history_output_path": training_history_output_path,
        "validation_history_output_path": validation_history_output_path,
        "optimization_mode": optimization_mode,
        "initial_particle_shifts": initial_particle_shifts,
    }

    if optimize_algorithm == "nelder-mead":
        return optimize_sigmas_2dtm_nelder_mead(
            sigma_iterations=sigma_iterations,
            **common_kwargs,
        )
    if optimize_algorithm == "bayesian":
        return optimize_sigmas_2dtm_optuna(
            n_trials=sigma_iterations,
            **common_kwargs,
        )
    # This should never happen due to Literal type, but satisfy type checker
    raise ValueError(
        f"Unknown optimize_algorithm: {optimize_algorithm}. "
        "Must be 'nelder-mead' or 'bayesian'"
    )


def optimize_sigmas_2dtm_nelder_mead(
    image: torch.Tensor,  # (t, H, W)
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    refine_config_path: str,
    optimize_particle_df_path: str | None = None,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    motion_iterations: int = 10,
    sigma_iterations: int = 20,
    optimizer_kwargs: dict[str, Any] | None = None,
    correlation_batch_size: int = 20,
    particle_batch_size: int = 102,
    particle_indices: pd.Index = None,
    device: torch.device = None,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    prior_type: str = "relion",
    init_sigma_a: float = 0.513517,
    init_alpha_spatial: float = 1e5,
    init_sigma_a_amplitude: float = 2.0,
    init_sigma_a_decay: float = 0.1,
    init_sigma_a_offset: float = 1.0,
    sigma_a_exponential: bool = False,
    init_sigma_d: float = 5782.376953,
    init_sigma_v: float = 0.194826,
    optimize_sigma_a: bool = True,
    optimize_alpha_spatial: bool = True,
    optimize_sigma_a_amplitude: bool = True,
    optimize_sigma_a_decay: bool = True,
    optimize_sigma_a_offset: bool = True,
    optimize_sigma_d: bool = True,
    optimize_sigma_v: bool = True,
    verbose: bool = True,
    # Output paths for saving results
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
    optimization_mode: Literal["deformation_field", "particle_shifts"] = "deformation_field",
    initial_particle_shifts: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Optimize prior hyperparameters using Nelder-Mead method.

    Uses scipy.optimize.minimize with Nelder-Mead (simplex) method for the outer
    loop optimization of sigma parameters. The inner loop still uses Adam optimizer
    for motion estimation.

    This method is derivative-free and typically faster than gradient-based methods
    when function evaluations are expensive, as it requires fewer evaluations per
    iteration.

    Parameters
    ----------
    image : torch.Tensor
        (t, H, W) movie to estimate motion from
    var_image : torch.Tensor
        (t, H, W) variance image
    mean_image : torch.Tensor
        (t, H, W) mean image
    pixel_spacing : float
        Pixel spacing in Angstroms
    deformation_field_resolution : tuple[int, int, int]
        Resolution of deformation field (nt, nh, nw)
    initial_deformation_field : torch.Tensor | None
        Initial deformation field (2, nt, nh, nw) or None
    refine_config_path : str
        Path to refine config (training template)
    optimize_particle_df_path : str | None
        Path to particle dataframe config for validation loss computation.
        The validation template will be loaded from the template_volume_path
        in this YAML file. If None, uses the same particle dataframe and
        template as the motion loop.
        Default is None.
    pre_exposure : float
        Pre-exposure time in seconds. Default 0.0
    fluence_per_frame : float
        Fluence per frame in e/Å². Default 1.0
    motion_iterations : int
        Motion optimization iterations per sigma update. Default 10
    sigma_iterations : int
        Maximum number of Nelder-Mead iterations. Default 20
    optimizer_kwargs : dict
        Kwargs for motion optimizer. Default {"lr": 0.2}
    correlation_batch_size : int
        Batch size for correlation computation. Default 20
    particle_batch_size : int
        Batch size for particles. Default 102
    particle_indices : pd.Index
        Particle indices to use. Default None (all particles)
    device : torch.device
        Device to use. Default None (auto-detect)
    loss_metric : str
        Loss metric: "scaled_mip" or "cross_correlation". Default "scaled_mip"
    min_snr : float
        Minimum SNR threshold. Default 0.0
    best_n : int
        Maximum number of best particles to use. Default 10000000000
    prior_type : str
        "laplacian" or "relion". Default "relion"
    init_sigma_a : float
        Initial sigma_a (constant mode). Default 0.513517
    init_alpha_spatial : float
        Initial alpha_spatial. Default 1e5
    init_sigma_a_amplitude : float
        Initial amplitude in exponential. Default 2.0
    init_sigma_a_decay : float
        Initial decay rate in exponential. Default 0.1
    init_sigma_a_offset : float
        Initial offset in exponential. Default 1.0
    sigma_a_exponential : bool
        Use exponential sigma_a. Default False
    init_sigma_d : float
        Initial sigma_d. Default 5782.376953
    init_sigma_v : float
        Initial sigma_v. Default 0.194826
    optimize_sigma_a : bool
        Whether to optimize sigma_a. Default True
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial. Default True
    optimize_sigma_a_amplitude : bool
        Whether to optimize sigma_a_amplitude. Default True
    optimize_sigma_a_decay : bool
        Whether to optimize sigma_a_decay. Default True
    optimize_sigma_a_offset : bool
        Whether to optimize sigma_a_offset. Default True
    optimize_sigma_d : bool
        Whether to optimize sigma_d. Default True
    optimize_sigma_v : bool
        Whether to optimize sigma_v. Default True
    verbose : bool
        Print progress. Default True
    optimized_sigmas_output_path : str | None
        Path to save final optimized sigmas as JSON. Default None
    sigma_history_output_path : str | None
        Path to save sigma history (all iterations) as JSON. Default None
    training_history_output_path : str | None
        Path to save training loss history as JSON. Default None
    validation_history_output_path : str | None
        Path to save validation loss history as JSON. Default None

    Returns
    -------
    dict
        - "optimized_sigmas": dict of optimized sigma values
        - "final_deformation_field": final deformation field
        - "validation_loss_history": list of validation losses
        - "training_loss_history": list of training losses
        - "sigma_history": list of sigma values at each iteration
        - "best_validation_loss": best validation loss found
        - "best_sigma_iter": iteration with best validation loss
        - "best_sigma_params": best sigma parameters found
    """
    torch.set_grad_enabled(True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_sigma_opt_nelder_"))

    (
        refine_config_path,
        particle_indices,
        validation_template,
        template_volume,
        optimizer_kwargs,
        image,
        var_image,
        mean_image,
        param_names,
        initial_values,
        sigma_params,
        batch_config_paths,
        batch_particle_indices,
        total_n_particles,
        batch_mean_stacks,
        batch_std_stacks,
        validation_batch_config_paths,
        validation_batch_particle_indices,
        validation_total_n_particles,
        validation_batch_mean_stacks,
        validation_batch_std_stacks,
    ) = _setup_optimizer(
        image=image,
        var_image=var_image,
        mean_image=mean_image,
        refine_config_path=refine_config_path,
        optimize_particle_df_path=optimize_particle_df_path,
        particle_indices=particle_indices,
        device=device,
        temp_dir=temp_dir,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        optimizer_kwargs=optimizer_kwargs,
        particle_batch_size=particle_batch_size,
        prior_type=prior_type,
        sigma_a_exponential=sigma_a_exponential,
        optimize_sigma_a=optimize_sigma_a,
        optimize_alpha_spatial=optimize_alpha_spatial,
        optimize_sigma_a_amplitude=optimize_sigma_a_amplitude,
        optimize_sigma_a_decay=optimize_sigma_a_decay,
        optimize_sigma_a_offset=optimize_sigma_a_offset,
        optimize_sigma_d=optimize_sigma_d,
        optimize_sigma_v=optimize_sigma_v,
        init_sigma_a=init_sigma_a,
        init_alpha_spatial=init_alpha_spatial,
        init_sigma_a_amplitude=init_sigma_a_amplitude,
        init_sigma_a_decay=init_sigma_a_decay,
        init_sigma_a_offset=init_sigma_a_offset,
        init_sigma_d=init_sigma_d,
        init_sigma_v=init_sigma_v,
        use_dict=False,  # Nelder-Mead uses list
        cleanup_memory=True,  # Clean up memory after batch stacks
    )

    # Explicitly capture variables for closure (helps linter recognize them)
    _batch_mean_stacks = batch_mean_stacks
    _batch_std_stacks = batch_std_stacks
    _validation_batch_config_paths = validation_batch_config_paths
    _validation_batch_particle_indices = validation_batch_particle_indices
    _validation_batch_mean_stacks = validation_batch_mean_stacks
    _validation_batch_std_stacks = validation_batch_std_stacks
    _validation_total_n_particles = validation_total_n_particles
    _template_volume = template_volume
    _validation_template = validation_template

    validation_loss_history = []
    training_loss_history = []
    sigma_history = []

    # Best-point tracking
    best_validation_loss = None
    best_sigma_iter = None
    best_sigma_params = None

    # Counter for outer loop iterations (Nelder-Mead)
    # Use list to allow modification in nested function
    outer_iter_counter = [0]  # pylint: disable=unused-variable

    # Objective function for scipy.optimize.minimize
    def objective_function(x: np.ndarray) -> float:
        """Objective function for Nelder-Mead optimization.

        Args:
            x: numpy array of parameter values (in order of param_names)

        Returns
        -------
            validation_loss: float
        """
        # Increment and print outer iteration number
        outer_iter_counter[0] += 1
        print(f"Outer iteration (Nelder-Mead): {outer_iter_counter[0]}")

        # Set sigma parameters from x
        for i, param_name in enumerate(param_names):
            sigma_params[param_name] = float(x[i])

        # Run inner optimization with current sigmas
        deformation_field, accumulated_loss = _run_inner_optimization_common(
            initial_deformation_field=initial_deformation_field,
            deformation_field_resolution=deformation_field_resolution,
            device=device,
            optimizer_kwargs=optimizer_kwargs,
            image=image,
            batch_config_paths=batch_config_paths,
            batch_particle_indices=batch_particle_indices,
            batch_mean_stacks=_batch_mean_stacks,
            batch_std_stacks=_batch_std_stacks,
            template_volume=_template_volume,
            pixel_spacing=pixel_spacing,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            motion_iterations=motion_iterations,
            correlation_batch_size=correlation_batch_size,
            total_n_particles=total_n_particles,
            loss_metric=loss_metric,
            prior_type=prior_type,
            sigma_params=sigma_params,
            sigma_a_exponential=sigma_a_exponential,
            optimization_mode=optimization_mode,
            initial_particle_shifts=initial_particle_shifts,
            refine_config_path=refine_config_path,
        )

        # Compute validation loss
        # Use validation batch configs if available,
        # otherwise use training batch configs
        val_config_paths = (
            _validation_batch_config_paths
            if _validation_batch_config_paths is not None
            else batch_config_paths
        )
        val_particle_indices = (
            _validation_batch_particle_indices
            if _validation_batch_particle_indices is not None
            else batch_particle_indices
        )
        val_mean_stacks = (
            _validation_batch_mean_stacks
            if _validation_batch_mean_stacks is not None
            else _batch_mean_stacks
        )
        val_std_stacks = (
            _validation_batch_std_stacks
            if _validation_batch_std_stacks is not None
            else _batch_std_stacks
        )
        val_total_n_particles = (
            _validation_total_n_particles
            if _validation_total_n_particles is not None
            else total_n_particles
        )
        validation_loss = _compute_validation_loss_common(
            deformation_field_to_use=deformation_field,
            batch_config_paths=val_config_paths,
            batch_particle_indices=val_particle_indices,
            batch_mean_stacks=val_mean_stacks,
            batch_std_stacks=val_std_stacks,
            image=image,
            validation_template=_validation_template,
            total_n_particles=val_total_n_particles,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            loss_metric=loss_metric,
            correlation_batch_size=correlation_batch_size,
        )

        # Track history
        validation_loss_history.append(validation_loss)
        training_loss_history.append(accumulated_loss)

        # Track best point
        nonlocal best_validation_loss, best_sigma_iter, best_sigma_params
        if best_validation_loss is None or validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_sigma_iter = len(validation_loss_history) - 1
            best_sigma_params = {
                name: float(x[i]) for i, name in enumerate(param_names)
            }
            if verbose:
                print(f"✓ New best validation loss: {best_validation_loss:.6f}")

        # Record current sigmas (include fixed parameters too)
        current_sigmas = sigma_params.copy()
        sigma_history.append(current_sigmas.copy())

        if verbose:
            print(
                f"Iteration {len(validation_loss_history)}: Parameters: "
                f"{dict(zip(param_names, x, strict=True))}"
            )
            print(f"  Validation loss: {validation_loss:.6f}")

        del deformation_field
        gc.collect()
        torch.cuda.empty_cache()

        return validation_loss

    try:
        # Convert initial values to numpy array
        x0 = np.array(initial_values, dtype=np.float64)

        if verbose:
            print("=" * 70)
            print("NELDER-MEAD SIGMA OPTIMIZATION")
            print(f"Parameters to optimize: {param_names}")
            print(
                f"Initial values: {dict(zip(param_names, initial_values, strict=True))}"
            )
            print(f"Maximum iterations: {sigma_iterations}")
            print("=" * 70)

        # Run Nelder-Mead optimization
        result = minimize(
            objective_function,
            x0,
            method="Nelder-Mead",
            options={
                "maxiter": sigma_iterations,
                "xatol": 1e-6,  # Absolute tolerance for convergence
                "fatol": 1e-6,  # Function value tolerance
                "disp": verbose,
            },
        )

        if verbose:
            print("\n" + "=" * 70)
            print("NELDER-MEAD OPTIMIZATION COMPLETE")
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Final function value: {result.fun:.6f}")
            print(f"Number of iterations: {result.nit}")
            print(f"Number of function evaluations: {result.nfev}")
            print("=" * 70)

        # Extract optimized values
        optimized_values = result.x
        for i, param_name in enumerate(param_names):
            sigma_params[param_name] = float(optimized_values[i])
        # Get final deformation field with optimized parameters
        deformation_field, _ = _run_inner_optimization_common(
            initial_deformation_field=initial_deformation_field,
            deformation_field_resolution=deformation_field_resolution,
            device=device,
            optimizer_kwargs=optimizer_kwargs,
            image=image,
            batch_config_paths=batch_config_paths,
            batch_particle_indices=batch_particle_indices,
            batch_mean_stacks=batch_mean_stacks,
            batch_std_stacks=batch_std_stacks,
            template_volume=template_volume,
            pixel_spacing=pixel_spacing,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            motion_iterations=motion_iterations,
            correlation_batch_size=correlation_batch_size,
            total_n_particles=total_n_particles,
            loss_metric=loss_metric,
            prior_type=prior_type,
            sigma_params=sigma_params,
            sigma_a_exponential=sigma_a_exponential,
            optimization_mode=optimization_mode,
            initial_particle_shifts=initial_particle_shifts,
            refine_config_path=refine_config_path,
        )
        # Handle return value - can be dict with particle_shifts or deformation_field
        if isinstance(deformation_field, dict):
            final_deformation_field = deformation_field
        else:
            final_deformation_field = deformation_field.data
            final_deformation_field = final_deformation_field - torch.mean(
                final_deformation_field, dim=(1, 2, 3), keepdim=True
            )

        optimized_sigmas = sigma_params.copy()

        # Use best point if it's better than final point
        final_validation_loss = validation_loss_history[-1]
        if (
            best_validation_loss is not None
            and best_sigma_params is not None
            and final_validation_loss > best_validation_loss
        ):
            if verbose:
                print(f"\nFinal validation loss ({final_validation_loss:.6f})")
                print(f" is worse than best ({best_validation_loss:.6f})")
                print(f"Restoring best parameters from iteration {best_sigma_iter}")
            # Restore best parameters
            for name, best_val in best_sigma_params.items():
                sigma_params[name] = best_val
            # Re-run inner optimization with best parameters
            deformation_field, _ = _run_inner_optimization_common(
                initial_deformation_field=initial_deformation_field,
                deformation_field_resolution=deformation_field_resolution,
                device=device,
                optimizer_kwargs=optimizer_kwargs,
                image=image,
                batch_config_paths=batch_config_paths,
                batch_particle_indices=batch_particle_indices,
                batch_mean_stacks=batch_mean_stacks,
                batch_std_stacks=batch_std_stacks,
                template_volume=template_volume,
                pixel_spacing=pixel_spacing,
                pre_exposure=pre_exposure,
                fluence_per_frame=fluence_per_frame,
                motion_iterations=motion_iterations,
                correlation_batch_size=correlation_batch_size,
                total_n_particles=total_n_particles,
                loss_metric=loss_metric,
                prior_type=prior_type,
                sigma_params=sigma_params,
                sigma_a_exponential=sigma_a_exponential,
                optimization_mode=optimization_mode,
                initial_particle_shifts=initial_particle_shifts,
                refine_config_path=refine_config_path,
            )
            # Handle return value - can be dict with particle_shifts or deformation_field
            if isinstance(deformation_field, dict):
                final_deformation_field = deformation_field
            else:
                final_deformation_field = deformation_field.data
                final_deformation_field = final_deformation_field - torch.mean(
                    final_deformation_field, dim=(1, 2, 3), keepdim=True
                )
            optimized_sigmas = sigma_params.copy()

        # Print summary
        if (
            verbose
            and best_validation_loss is not None
            and best_sigma_params is not None
        ):
            print("\n" + "=" * 70)
            print("OPTIMIZATION SUMMARY")
            print("=" * 70)
            print(f"Best validation loss: {best_validation_loss:.6f}")
            print(f" at iteration {best_sigma_iter}")
            print(f"Final validation loss: {validation_loss_history[-1]:.6f}")
            print("Best parameters:")
            for name, val in best_sigma_params.items():
                print(f"  {name}: {val:.6f}")
            print("=" * 70)

    finally:
        del batch_mean_stacks, batch_std_stacks, template_volume, validation_template
        if validation_batch_mean_stacks is not None:
            del validation_batch_mean_stacks
        if validation_batch_std_stacks is not None:
            del validation_batch_std_stacks
        gc.collect()
        torch.cuda.empty_cache()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"Cleaned up temporary configs at {temp_dir}")

    # Save results to files if paths are specified
    save_optimize_sigmas_to_json(
        optimized_sigmas=optimized_sigmas,
        sigma_history=sigma_history,
        training_loss_history=training_loss_history,
        validation_loss_history=validation_loss_history,
        optimized_sigmas_output_path=optimized_sigmas_output_path,
        sigma_history_output_path=sigma_history_output_path,
        training_history_output_path=training_history_output_path,
        validation_history_output_path=validation_history_output_path,
        verbose=verbose,
    )

    result_dict = {
        "optimized_sigmas": optimized_sigmas,
        "final_deformation_field": final_deformation_field,
        "validation_loss_history": validation_loss_history,
        "training_loss_history": training_loss_history,
        "sigma_history": sigma_history,
    }

    # Add best-point information
    if best_validation_loss is not None:
        result_dict["best_validation_loss"] = best_validation_loss
        result_dict["best_sigma_iter"] = best_sigma_iter
        result_dict["best_sigma_params"] = best_sigma_params

    return result_dict


def optimize_sigmas_2dtm_optuna(
    image: torch.Tensor,  # (t, H, W)
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    refine_config_path: str,
    optimize_particle_df_path: str | None = None,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    motion_iterations: int = 10,
    n_trials: int = 50,
    optimizer_kwargs: dict[str, Any] | None = None,
    correlation_batch_size: int = 20,
    particle_batch_size: int = 102,
    particle_indices: pd.Index = None,
    device: torch.device = None,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    prior_type: str = "relion",
    init_sigma_a: float = 0.513517,
    init_alpha_spatial: float = 1e5,
    init_sigma_a_amplitude: float = 2.0,
    init_sigma_a_decay: float = 0.1,
    init_sigma_a_offset: float = 1.0,
    sigma_a_exponential: bool = False,
    init_sigma_d: float = 5782.376953,
    init_sigma_v: float = 0.194826,
    optimize_sigma_a: bool = True,
    optimize_alpha_spatial: bool = True,
    optimize_sigma_a_amplitude: bool = True,
    optimize_sigma_a_decay: bool = True,
    optimize_sigma_a_offset: bool = True,
    optimize_sigma_d: bool = True,
    optimize_sigma_v: bool = True,
    # Optuna-specific parameters
    study_name: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,  # Default: TPE
    pruner: optuna.pruners.BasePruner | None = None,  # Default: MedianPruner
    direction: str = "minimize",
    param_range_low: float = 0.25,  # Lower bound multiplier for parameter
    param_range_high: float = 4.0,  # Upper bound multiplier for parameter
    verbose: bool = True,
    # Output paths for saving results
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
    optimization_mode: Literal["deformation_field", "particle_shifts"] = "deformation_field",
    initial_particle_shifts: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Optimize prior hyperparameters using Optuna (Bayesian optimization).

    Uses Optuna's Tree-structured Parzen Estimator (TPE) for intelligent
    hyperparameter search. The inner loop still uses Adam optimizer for
    motion estimation.

    Advantages of Optuna:
    - Bayesian optimization: Learns from previous trials to focus on promising regions
    - Global search: Better at finding global minima than local optimizers
    - Handles noisy functions: Robust to noisy/expensive function evaluations
    - Pruning: Can stop unpromising trials early to save computation
    - Parallelization: Can run multiple trials in parallel
    - Automatic parameter handling: Supports continuous, discrete, categorical

    Parameters
    ----------
    image : torch.Tensor
        (t, H, W) movie to estimate motion from
    var_image : torch.Tensor
        (t, H, W) variance image
    mean_image : torch.Tensor
        (t, H, W) mean image
    pixel_spacing : float
        Pixel spacing in Angstroms
    deformation_field_resolution : tuple[int, int, int]
        Resolution of deformation field (nt, nh, nw)
    initial_deformation_field : torch.Tensor | None
        Initial deformation field (2, nt, nh, nw) or None
    refine_config_path : str
        Path to refine config (training template)
    optimize_particle_df_path : str | None
        Path to particle dataframe config for validation loss computation.
        The validation template will be loaded from the template_volume_path
        in this YAML file. If None, uses the same particle dataframe and
        template as the motion loop.
        Default is None.
    pre_exposure : float
        Pre-exposure time in seconds. Default 0.0
    fluence_per_frame : float
        Fluence per frame in e/Å². Default 1.0
    motion_iterations : int
        Motion optimization iterations per sigma update. Default 10
    n_trials : int
        Number of Optuna trials to run. Default 50
    optimizer_kwargs : dict
        Kwargs for motion optimizer. Default {"lr": 0.2}
    correlation_batch_size : int
        Batch size for correlation computation. Default 20
    particle_batch_size : int
        Batch size for particles. Default 102
    particle_indices : pd.Index
        Particle indices to use. Default None (all particles)
    device : torch.device
        Device to use. Default None (auto-detect)
    loss_metric : str
        Loss metric: "scaled_mip" or "cross_correlation". Default "scaled_mip"
    min_snr : float
        Minimum SNR threshold. Default 0.0
    best_n : int
        Maximum number of best particles to use. Default 10000000000
    prior_type : str
        "laplacian" or "relion". Default "relion"
    init_sigma_a : float
        Initial sigma_a (constant mode). Default 0.513517
    init_alpha_spatial : float
        Initial alpha_spatial. Default 1e5
    init_sigma_a_amplitude : float
        Initial amplitude in exponential. Default 2.0
    init_sigma_a_decay : float
        Initial decay rate in exponential. Default 0.1
    init_sigma_a_offset : float
        Initial offset in exponential. Default 1.0
    sigma_a_exponential : bool
        Use exponential sigma_a. Default False
    init_sigma_d : float
        Initial sigma_d. Default 5782.376953
    init_sigma_v : float
        Initial sigma_v. Default 0.194826
    optimize_sigma_a : bool
        Whether to optimize sigma_a. Default True
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial. Default True
    optimize_sigma_a_amplitude : bool
        Whether to optimize sigma_a_amplitude. Default True
    optimize_sigma_a_decay : bool
        Whether to optimize sigma_a_decay. Default True
    optimize_sigma_a_offset : bool
        Whether to optimize sigma_a_offset. Default True
    optimize_sigma_d : bool
        Whether to optimize sigma_d. Default True
    optimize_sigma_v : bool
        Whether to optimize sigma_v. Default True
    study_name : str | None
        Name for Optuna study. Default None (auto-generated)
    sampler : optuna.samplers.BaseSampler | None
        Optuna sampler. Default None (uses TPE)
    pruner : optuna.pruners.BasePruner | None
        Optuna pruner. Default None (uses MedianPruner)
    direction : str
        Optimization direction: "minimize" or "maximize". Default "minimize"
    param_range_low : float
        Lower bound multiplier for parameter search range. Parameters will be
        in range [initial_value * param_range_low, initial_value * param_range_high].
        Default 0.25 (i.e., 0.25x to 4x initial values).
    param_range_high : float
        Upper bound multiplier for parameter search range. Parameters will be
        in range [initial_value * param_range_low, initial_value * param_range_high].
        Default 4.0 (i.e., 0.25x to 4x initial values).
    verbose : bool
        Print progress. Default True
    optimized_sigmas_output_path : str | None
        Path to save final optimized sigmas as JSON. Default None
    sigma_history_output_path : str | None
        Path to save sigma history (all trials) as JSON. Default None
    training_history_output_path : str | None
        Path to save training loss history as JSON. Default None
    validation_history_output_path : str | None
        Path to save validation loss history as JSON. Default None
    optimization_mode : Literal["deformation_field", "particle_shifts"]
        Optimization mode. If "deformation_field", optimizes a deformation field grid.
        If "particle_shifts", optimizes particle shifts directly (T, N, 2).
        Default is "deformation_field".
    initial_particle_shifts : torch.Tensor | None
        Initial particle shifts with shape (T, N, 2) where T is number of frames
        and N is number of particles. Only used if optimization_mode is "particle_shifts".
        If None, initializes to zero shifts. Default is None.

    Returns
    -------
    dict
        - "optimized_sigmas": dict of optimized sigma values
        - "final_deformation_field": final deformation field or dict with "particle_shifts"
        - "validation_loss_history": list of validation losses
        - "training_loss_history": list of training losses
        - "sigma_history": list of sigma values at each trial
        - "best_validation_loss": best validation loss found
        - "best_trial": best trial number
        - "best_sigma_params": best sigma parameters found
        - "optuna_study": Optuna study object (for further analysis)
    """
    print("Optimizing sigmas using Optuna...")

    torch.set_grad_enabled(True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_sigma_opt_optuna_"))

    (
        refine_config_path,
        particle_indices,
        validation_template,
        template_volume,
        optimizer_kwargs,
        image,
        var_image,
        mean_image,
        param_names,
        initial_values,
        sigma_params,
        batch_config_paths,
        batch_particle_indices,
        total_n_particles,
        batch_mean_stacks,
        batch_std_stacks,
        validation_batch_config_paths,
        validation_batch_particle_indices,
        validation_total_n_particles,
        validation_batch_mean_stacks,
        validation_batch_std_stacks,
    ) = _setup_optimizer(
        image=image,
        var_image=var_image,
        mean_image=mean_image,
        refine_config_path=refine_config_path,
        optimize_particle_df_path=optimize_particle_df_path,
        particle_indices=particle_indices,
        device=device,
        temp_dir=temp_dir,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        optimizer_kwargs=optimizer_kwargs,
        particle_batch_size=particle_batch_size,
        prior_type=prior_type,
        sigma_a_exponential=sigma_a_exponential,
        optimize_sigma_a=optimize_sigma_a,
        optimize_alpha_spatial=optimize_alpha_spatial,
        optimize_sigma_a_amplitude=optimize_sigma_a_amplitude,
        optimize_sigma_a_decay=optimize_sigma_a_decay,
        optimize_sigma_a_offset=optimize_sigma_a_offset,
        optimize_sigma_d=optimize_sigma_d,
        optimize_sigma_v=optimize_sigma_v,
        init_sigma_a=init_sigma_a,
        init_alpha_spatial=init_alpha_spatial,
        init_sigma_a_amplitude=init_sigma_a_amplitude,
        init_sigma_a_decay=init_sigma_a_decay,
        init_sigma_a_offset=init_sigma_a_offset,
        init_sigma_d=init_sigma_d,
        init_sigma_v=init_sigma_v,
        use_dict=True,  # Optuna uses dict
        cleanup_memory=False,  # No memory cleanup needed for Optuna
    )

    validation_loss_history = []
    training_loss_history = []
    sigma_history = []

    # Objective function for Optuna
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Print outer iteration number (trial number)
        print(f"Outer iteration (Optuna trial): {trial.number + 1}")

        # Type assertion: initial_values is a dict for Optuna (use_dict=True)
        assert isinstance(initial_values, dict), (
            "initial_values must be a dict for Optuna"
        )
        initial_values_dict: dict[str, float] = initial_values

        # Suggest parameter values using log-uniform for wide ranges
        for param_name in param_names:
            initial_val = initial_values_dict[param_name]

            # Use log-uniform for parameters that span orders of magnitude
            if param_name in ("alpha_spatial", "sigma_d"):
                # These span large ranges, use log-uniform
                suggested_val = trial.suggest_float(
                    param_name,
                    low=max(initial_val * param_range_low, 1e-6),
                    high=initial_val * param_range_high,
                    log=True,
                )
            else:
                # Use uniform for smaller ranges
                suggested_val = trial.suggest_float(
                    param_name,
                    low=max(initial_val * param_range_low, 1e-6),
                    high=initial_val * param_range_high,
                    log=False,
                )

            sigma_params[param_name] = suggested_val

        # Run inner optimization with suggested sigmas
        deformation_field, accumulated_loss = _run_inner_optimization_common(
            initial_deformation_field=initial_deformation_field,
            deformation_field_resolution=deformation_field_resolution,
            device=device,
            optimizer_kwargs=optimizer_kwargs,
            image=image,
            batch_config_paths=batch_config_paths,
            batch_particle_indices=batch_particle_indices,
            batch_mean_stacks=batch_mean_stacks,
            batch_std_stacks=batch_std_stacks,
            template_volume=template_volume,
            pixel_spacing=pixel_spacing,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            motion_iterations=motion_iterations,
            correlation_batch_size=correlation_batch_size,
            total_n_particles=total_n_particles,
            loss_metric=loss_metric,
            prior_type=prior_type,
            sigma_params=sigma_params,
            sigma_a_exponential=sigma_a_exponential,
            optimization_mode=optimization_mode,
            initial_particle_shifts=initial_particle_shifts,
            refine_config_path=refine_config_path,
        )

        # Compute validation loss
        # Use validation batch configs if available,
        # otherwise use training batch configs
        val_config_paths = (
            validation_batch_config_paths
            if validation_batch_config_paths is not None
            else batch_config_paths
        )
        val_particle_indices = (
            validation_batch_particle_indices
            if validation_batch_particle_indices is not None
            else batch_particle_indices
        )
        val_mean_stacks = (
            validation_batch_mean_stacks
            if validation_batch_mean_stacks is not None
            else batch_mean_stacks
        )
        val_std_stacks = (
            validation_batch_std_stacks
            if validation_batch_std_stacks is not None
            else batch_std_stacks
        )
        val_total_n_particles = (
            validation_total_n_particles
            if validation_total_n_particles is not None
            else total_n_particles
        )
        validation_loss = _compute_validation_loss_common(
            deformation_field_to_use=deformation_field,
            batch_config_paths=val_config_paths,
            batch_particle_indices=val_particle_indices,
            batch_mean_stacks=val_mean_stacks,
            batch_std_stacks=val_std_stacks,
            image=image,
            validation_template=validation_template,
            total_n_particles=val_total_n_particles,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            loss_metric=loss_metric,
            correlation_batch_size=correlation_batch_size,
        )

        # Track history
        validation_loss_history.append(validation_loss)
        training_loss_history.append(accumulated_loss)

        # Record current sigmas (include fixed parameters too)
        current_sigmas = sigma_params.copy()
        sigma_history.append(current_sigmas.copy())

        if verbose:
            param_dict = dict(
                zip(param_names, [sigma_params[n] for n in param_names], strict=True)
            )
            print(f"Trial {trial.number}: Parameters: {param_dict}")
            print(f"  Validation loss: {validation_loss:.6f}")

        return validation_loss

    try:
        # Create Optuna study
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
        )

        if verbose:
            print("=" * 70)
            print("OPTUNA SIGMA OPTIMIZATION")
            print(f"Parameters to optimize: {param_names}")
            print(f"Initial values: {initial_values}")
            print(f"Number of trials: {n_trials}")
            print(f"Sampler: {type(sampler).__name__}")
            print(f"Pruner: {type(pruner).__name__}")
            print("=" * 70)

        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

        if verbose:
            print("\n" + "=" * 70)
            print("OPTUNA OPTIMIZATION COMPLETE")
            print(f"Number of trials: {len(study.trials)}")
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best validation loss: {study.best_value:.6f}")
            print("Best parameters:")
            for name, val in study.best_params.items():
                print(f"  {name}: {val:.6f}")
            print("=" * 70)

        # Extract best parameters
        best_params = study.best_params
        for name, val in best_params.items():
            sigma_params[name] = val

        # Get final deformation field with best parameters
        deformation_field, _ = _run_inner_optimization_common(
            initial_deformation_field=initial_deformation_field,
            deformation_field_resolution=deformation_field_resolution,
            device=device,
            optimizer_kwargs=optimizer_kwargs,
            image=image,
            batch_config_paths=batch_config_paths,
            batch_particle_indices=batch_particle_indices,
            batch_mean_stacks=batch_mean_stacks,
            batch_std_stacks=batch_std_stacks,
            template_volume=template_volume,
            pixel_spacing=pixel_spacing,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            motion_iterations=motion_iterations,
            correlation_batch_size=correlation_batch_size,
            total_n_particles=total_n_particles,
            loss_metric=loss_metric,
            prior_type=prior_type,
            sigma_params=sigma_params,
            sigma_a_exponential=sigma_a_exponential,
            optimization_mode=optimization_mode,
            initial_particle_shifts=initial_particle_shifts,
            refine_config_path=refine_config_path,
        )
        # Handle return value - can be dict with particle_shifts or deformation_field
        if isinstance(deformation_field, dict):
            final_deformation_field = deformation_field
        else:
            final_deformation_field = deformation_field.data
            final_deformation_field = final_deformation_field - torch.mean(
                final_deformation_field, dim=(1, 2, 3), keepdim=True
            )

        # Create optimized_sigmas from best trial parameters
        # (explicitly use best validation loss)
        optimized_sigmas = {}
        # Include all parameters (both optimized and fixed)
        for key, value in sigma_params.items():
            if key in best_params:
                # Use best trial's value for optimized parameters
                optimized_sigmas[key] = best_params[key]
            else:
                # Use fixed parameter value
                optimized_sigmas[key] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"Cleaned up temporary configs at {temp_dir}")

    # Save results to files if paths are specified
    save_optimize_sigmas_to_json(
        optimized_sigmas=optimized_sigmas,
        sigma_history=sigma_history,
        training_loss_history=training_loss_history,
        validation_loss_history=validation_loss_history,
        optimized_sigmas_output_path=optimized_sigmas_output_path,
        sigma_history_output_path=sigma_history_output_path,
        training_history_output_path=training_history_output_path,
        validation_history_output_path=validation_history_output_path,
        verbose=verbose,
    )

    result_dict = {
        "optimized_sigmas": optimized_sigmas,
        "final_deformation_field": final_deformation_field,
        "validation_loss_history": validation_loss_history,
        "training_loss_history": training_loss_history,
        "sigma_history": sigma_history,
        "best_validation_loss": study.best_value,
        "best_trial": study.best_trial.number,
        "best_sigma_params": study.best_params,
        "optuna_study": study,  # For further analysis/visualization
    }

    return result_dict


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _compute_validation_loss_common(
    deformation_field_to_use: CubicCatmullRomGrid3d | dict[str, torch.Tensor],
    batch_config_paths: list[str],
    batch_particle_indices: list[list[pd.Index]],
    batch_mean_stacks: dict[str, torch.Tensor],
    batch_std_stacks: dict[str, torch.Tensor],
    image: torch.Tensor,
    validation_template: torch.Tensor,
    total_n_particles: int,
    pre_exposure: float,
    fluence_per_frame: float,
    loss_metric: str,
    correlation_batch_size: int,
) -> float:
    """Compute validation loss with current deformation field or particle_shifts.

    Parameters
    ----------
    deformation_field_to_use : CubicCatmullRomGrid3d | dict[str, torch.Tensor]
        Deformation field to evaluate, or dict with "particle_shifts" key containing
        (T, N, 2) tensor
    batch_config_paths : list[str]
        List of batch config paths
    batch_particle_indices : list[list[pd.Index]]
        List of particle indices for each batch
    batch_mean_stacks : dict[str, torch.Tensor]
        Pre-computed mean stacks for each batch
    batch_std_stacks : dict[str, torch.Tensor]
        Pre-computed std stacks for each batch
    image : torch.Tensor
        Input movie (t, H, W)
    validation_template : torch.Tensor
        Validation template tensor
    total_n_particles : int
        Total number of particles
    pre_exposure : float
        Pre-exposure time
    fluence_per_frame : float
        Fluence per frame
    loss_metric : str
        Loss metric ("scaled_mip" or "cross_correlation")
    correlation_batch_size : int
        Batch size for correlation computation

    Returns
    -------
    float
        Validation loss value
    """
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (batch_config_path, batch_indices) in enumerate(
            zip(batch_config_paths, batch_particle_indices, strict=True)
        ):
            batch_size = len(batch_indices[0])
            batch_refine_manager = _make_differentiable_refine_manager(
                batch_config_path
            )
            actual_particle_count = batch_refine_manager.particle_stack.num_particles
            print(
                f"    Validation batch {batch_idx + 1}: "
                f"indices={batch_size}, "
                f"actual={actual_particle_count}, "
                f"config={Path(batch_config_path).name}"
            )
            batch_particle_stack = batch_refine_manager.particle_stack

            # Handle both deformation_field and particle_shifts
            if isinstance(deformation_field_to_use, dict):
                # Extract particle_shifts for this batch
                particle_shifts = deformation_field_to_use["particle_shifts"]
                # Compute batch start/end indices (same logic as in optimization loop)
                batch_start_idx = sum(
                    len(batch_particle_indices[i][0])
                    for i in range(
                        batch_config_paths.index(batch_config_path)
                    )
                )
                batch_end_idx = batch_start_idx + batch_size
                batch_particle_shifts = particle_shifts[
                    :, batch_start_idx:batch_end_idx, :
                ]
                
                image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                    movie=image,
                    particle_shifts=batch_particle_shifts,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="reflect",
                    padding_value=0.0,
                    pre_exposure=pre_exposure,
                    fluence_per_frame=fluence_per_frame,
                )
            else:
                image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                    movie=image,
                    deformation_field=deformation_field_to_use,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="reflect",
                    padding_value=0.0,
                    pre_exposure=pre_exposure,
                    fluence_per_frame=fluence_per_frame,
                )

            # Reuse pre-computed mean/std stacks (same as motion loop)
            batch_mean_stack = batch_mean_stacks[batch_config_path]
            batch_std_stack = batch_std_stacks[batch_config_path]

            backend_kwargs = batch_refine_manager.make_differentiable_backend_kwargs(
                image_stack=image_stack_batch,
                mean_stack=batch_mean_stack,
                std_stack=batch_std_stack,
                particle_indices=batch_indices,
                template_tensor=validation_template,
                images_are_particles=True,
            )
            result = batch_refine_manager.get_refine_result(
                backend_kwargs, correlation_batch_size, use_differentiable=True
            )

            val_loss_tensor = (
                result["refined_z_score"]
                if loss_metric == "scaled_mip"
                else result["refined_cross_correlation"]
            )
            val_loss += (
                -torch.mean(val_loss_tensor).item() * batch_size / total_n_particles
            )

            del (
                image_stack_batch,
                batch_mean_stack,
                batch_std_stack,
                backend_kwargs,
                result,
            )
            torch.cuda.empty_cache()
    return val_loss


def _collect_sigma_parameters(
    prior_type: str,
    sigma_a_exponential: bool,
    optimize_sigma_a: bool,
    optimize_alpha_spatial: bool,
    optimize_sigma_a_amplitude: bool,
    optimize_sigma_a_decay: bool,
    optimize_sigma_a_offset: bool,
    optimize_sigma_d: bool,
    optimize_sigma_v: bool,
    init_sigma_a: float,
    init_alpha_spatial: float,
    init_sigma_a_amplitude: float,
    init_sigma_a_decay: float,
    init_sigma_a_offset: float,
    init_sigma_d: float,
    init_sigma_v: float,
    use_dict: bool = False,  # True for optuna (dict), False for nelder-mead (list)
) -> tuple[
    list[str],  # param_names
    list[float] | dict[str, float],  # initial_values
    dict[str, float],  # sigma_params
]:
    """Collect sigma parameters for optimization.

    Parameters
    ----------
    prior_type : str
        Prior type ("laplacian" or "relion")
    sigma_a_exponential : bool
        Whether to use exponential sigma_a
    optimize_sigma_a : bool
        Whether to optimize sigma_a
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial
    optimize_sigma_a_amplitude : bool
        Whether to optimize sigma_a_amplitude
    optimize_sigma_a_decay : bool
        Whether to optimize sigma_a_decay
    optimize_sigma_a_offset : bool
        Whether to optimize sigma_a_offset
    optimize_sigma_d : bool
        Whether to optimize sigma_d
    optimize_sigma_v : bool
        Whether to optimize sigma_v
    init_sigma_a : float
        Initial sigma_a value
    init_alpha_spatial : float
        Initial alpha_spatial value
    init_sigma_a_amplitude : float
        Initial sigma_a_amplitude value
    init_sigma_a_decay : float
        Initial sigma_a_decay value
    init_sigma_a_offset : float
        Initial sigma_a_offset value
    init_sigma_d : float
        Initial sigma_d value
    init_sigma_v : float
        Initial sigma_v value
    use_dict : bool
        If True, return initial_values as dict (for optuna).
        If False, return as list (for nelder-mead). Default False.

    Returns
    -------
    tuple
        - param_names: list of parameter names to optimize
        - initial_values: list or dict of initial values
        - sigma_params: dict of all parameters (optimized and fixed)
    """
    param_names: list[str] = []
    if use_dict:
        initial_values_dict: dict[str, float] = {}
    else:
        initial_values_list: list[float] = []
    sigma_params: dict[str, float] = {}

    # Collect parameters to optimize
    if sigma_a_exponential:
        if optimize_sigma_a_amplitude:
            param_names.append("sigma_a_amplitude")
            if use_dict:
                initial_values_dict["sigma_a_amplitude"] = init_sigma_a_amplitude
            else:
                initial_values_list.append(init_sigma_a_amplitude)
            sigma_params["sigma_a_amplitude"] = init_sigma_a_amplitude
        else:
            sigma_params["sigma_a_amplitude"] = init_sigma_a_amplitude

        if optimize_sigma_a_decay:
            param_names.append("sigma_a_decay")
            if use_dict:
                initial_values_dict["sigma_a_decay"] = init_sigma_a_decay
            else:
                initial_values_list.append(init_sigma_a_decay)
            sigma_params["sigma_a_decay"] = init_sigma_a_decay
        else:
            sigma_params["sigma_a_decay"] = init_sigma_a_decay

        if optimize_sigma_a_offset:
            param_names.append("sigma_a_offset")
            if use_dict:
                initial_values_dict["sigma_a_offset"] = init_sigma_a_offset
            else:
                initial_values_list.append(init_sigma_a_offset)
            sigma_params["sigma_a_offset"] = init_sigma_a_offset
        else:
            sigma_params["sigma_a_offset"] = init_sigma_a_offset
    else:
        if optimize_sigma_a:
            param_names.append("sigma_a")
            if use_dict:
                initial_values_dict["sigma_a"] = init_sigma_a
            else:
                initial_values_list.append(init_sigma_a)
            sigma_params["sigma_a"] = init_sigma_a
        else:
            sigma_params["sigma_a"] = init_sigma_a

    if prior_type == "laplacian" and optimize_alpha_spatial:
        param_names.append("alpha_spatial")
        if use_dict:
            initial_values_dict["alpha_spatial"] = init_alpha_spatial
        else:
            initial_values_list.append(init_alpha_spatial)
        sigma_params["alpha_spatial"] = init_alpha_spatial
    else:
        sigma_params["alpha_spatial"] = init_alpha_spatial

    if prior_type == "relion":
        if optimize_sigma_d:
            param_names.append("sigma_d")
            if use_dict:
                initial_values_dict["sigma_d"] = init_sigma_d
            else:
                initial_values_list.append(init_sigma_d)
            sigma_params["sigma_d"] = init_sigma_d
        else:
            sigma_params["sigma_d"] = init_sigma_d

        if optimize_sigma_v:
            param_names.append("sigma_v")
            if use_dict:
                initial_values_dict["sigma_v"] = init_sigma_v
            else:
                initial_values_list.append(init_sigma_v)
            sigma_params["sigma_v"] = init_sigma_v
        else:
            sigma_params["sigma_v"] = init_sigma_v

    if len(param_names) == 0:
        raise ValueError("No sigma parameters selected for optimization!")

    return (
        param_names,
        initial_values_dict if use_dict else initial_values_list,
        sigma_params,
    )


# pylint: disable=too-many-arguments,too-many-locals
def _setup_prior_params(
    prior_type: str,
    sigma_a_exponential: bool,
    sigma_params: dict[str, Any],
    image: torch.Tensor,
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int] | None,
    fluence_per_frame: float,
    device: torch.device,
    optimization_mode: Literal["deformation_field", "particle_shifts"] = "deformation_field",
    n_particles: int | None = None,  # pylint: disable=unused-argument
) -> dict[str, Any]:
    """Set up prior parameters based on prior type.

    Parameters
    ----------
    prior_type : str
        "laplacian" or "relion"
    sigma_a_exponential : bool
        Whether to use exponential sigma_a
    sigma_params : dict[str, Any]
        Dictionary of sigma parameters (values can be float or torch.Tensor)
    image : torch.Tensor
        Input movie (t, H, W)
    pixel_spacing : float
        Pixel spacing in Angstroms
    deformation_field_resolution : tuple[int, int, int]
        Resolution of deformation field (nt, nh, nw)
    fluence_per_frame : float
        Fluence per frame
    device : torch.device
        Device to use

    Returns
    -------
    dict[str, Any]
        Dictionary containing prior parameters:
        - For "laplacian": spatial_spacing, temporal_spacing, sigma_a_tensor, alpha
        - For "relion": image_coords, sigma_d_val, sigma_v_norm, sigma_a_norm
    """

    def get_val(key: str) -> Any:
        """Get parameter value, handling both dict and tensor cases."""
        v = sigma_params.get(key)
        return abs(v) if isinstance(v, int | float) else v

    prior_params: dict[str, Any] = {}

    # Determine number of time frames
    if optimization_mode == "deformation_field":
        assert deformation_field_resolution is not None
        nt = deformation_field_resolution[0]
    else:  # particle_shifts
        nt = image.shape[0]

    if prior_type == "laplacian":
        if optimization_mode == "deformation_field":
            assert deformation_field_resolution is not None
            spatial_spacing, temporal_spacing = _compute_physical_spacing(
                image.shape[-2:],
                pixel_spacing,
                deformation_field_resolution,
                fluence_per_frame * image.shape[0],
            )
        else:  # particle_shifts
            spatial_spacing = None
            temporal_spacing = (
                fluence_per_frame * image.shape[0] / nt if nt > 0 else None
            )
        prior_params["spatial_spacing"] = spatial_spacing
        prior_params["temporal_spacing"] = temporal_spacing

        if sigma_a_exponential:
            amplitude = get_val("sigma_a_amplitude")
            decay_rate = get_val("sigma_a_decay")
            offset = get_val("sigma_a_offset")
            amplitude = (
                amplitude.item() if isinstance(amplitude, torch.Tensor) else amplitude
            )
            decay_rate = (
                decay_rate.item()
                if isinstance(decay_rate, torch.Tensor)
                else decay_rate
            )
            offset = offset.item() if isinstance(offset, torch.Tensor) else offset
            sigma_a_tensor = _create_exponential_sigma_a(
                fluence_per_frame * image.shape[0],
                nt,
                amplitude=amplitude,
                decay_rate=decay_rate,
                offset=offset,
                device=device,
            )
        else:
            sigma_a_tensor = get_val("sigma_a")
            sigma_a_tensor = (
                sigma_a_tensor.item()
                if isinstance(sigma_a_tensor, torch.Tensor)
                else sigma_a_tensor
            )
        prior_params["sigma_a_tensor"] = sigma_a_tensor

        alpha = get_val("alpha_spatial")
        alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        prior_params["alpha"] = alpha

    elif prior_type == "relion":
        if optimization_mode == "deformation_field":
            assert deformation_field_resolution is not None
            image_coords = _build_physical_coords(
                nh=deformation_field_resolution[1],
                nw=deformation_field_resolution[2],
                image_shape=image.shape[-2:],
                pixel_size=pixel_spacing,
                device=device,
            )
            prior_params["image_coords"] = image_coords
        else:  # particle_shifts
            # image_coords will be set from particle_coords separately
            prior_params["image_coords"] = None

        sigma_d_val = get_val("sigma_d")
        sigma_d_val = (
            sigma_d_val.item() if isinstance(sigma_d_val, torch.Tensor) else sigma_d_val
        )
        prior_params["sigma_d_val"] = sigma_d_val

        sigma_v_val = get_val("sigma_v")
        sigma_v_val = (
            sigma_v_val.item() if isinstance(sigma_v_val, torch.Tensor) else sigma_v_val
        )
        total_fluence = fluence_per_frame * image.shape[0]
        sigma_v_norm = _normalize_sigma_fluence(
            sigma_v_val,
            total_fluence,
            nt,
        )
        prior_params["sigma_v_norm"] = sigma_v_norm

        if sigma_a_exponential:
            amplitude = get_val("sigma_a_amplitude")
            decay_rate = get_val("sigma_a_decay")
            offset = get_val("sigma_a_offset")
            amplitude = (
                amplitude.item() if isinstance(amplitude, torch.Tensor) else amplitude
            )
            decay_rate = (
                decay_rate.item()
                if isinstance(decay_rate, torch.Tensor)
                else decay_rate
            )
            offset = offset.item() if isinstance(offset, torch.Tensor) else offset
            sigma_a_tensor = _create_exponential_sigma_a(
                total_fluence,
                nt,
                amplitude=amplitude,
                decay_rate=decay_rate,
                offset=offset,
                device=device,
            )
            sigma_a_norm = _normalize_sigma_fluence(
                sigma_a_tensor,
                total_fluence,
                nt,
            )
        else:
            sa = get_val("sigma_a")
            sa = sa.item() if isinstance(sa, torch.Tensor) else sa
            sigma_a_norm = _normalize_sigma_fluence(
                sa,
                total_fluence,
                nt,
            )
        prior_params["sigma_a_norm"] = sigma_a_norm

    return prior_params


# pylint: disable=too-many-arguments,too-many-locals,too-many-statements
def _setup_optimizer(
    image: torch.Tensor,
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    refine_config_path: str,
    optimize_particle_df_path: str | None,
    particle_indices: pd.Index | None,
    device: torch.device | None,
    temp_dir: Path,
    loss_metric: str,
    min_snr: float,
    best_n: int,
    optimizer_kwargs: dict[str, Any] | None,
    particle_batch_size: int,
    prior_type: str,
    sigma_a_exponential: bool,
    optimize_sigma_a: bool,
    optimize_alpha_spatial: bool,
    optimize_sigma_a_amplitude: bool,
    optimize_sigma_a_decay: bool,
    optimize_sigma_a_offset: bool,
    optimize_sigma_d: bool,
    optimize_sigma_v: bool,
    init_sigma_a: float,
    init_alpha_spatial: float,
    init_sigma_a_amplitude: float,
    init_sigma_a_decay: float,
    init_sigma_a_offset: float,
    init_sigma_d: float,
    init_sigma_v: float,
    use_dict: bool,
    cleanup_memory: bool = False,
) -> tuple[
    str,  # refine_config_path
    pd.Index,  # particle_indices
    torch.Tensor,  # validation_template
    torch.Tensor,  # template_volume
    dict[str, Any],  # optimizer_kwargs
    torch.Tensor,  # image (detached)
    torch.Tensor,  # var_image (detached)
    torch.Tensor,  # mean_image (detached)
    list[str],  # param_names
    list[float] | dict[str, float],  # initial_values
    dict[str, float],  # sigma_params
    list[str],  # batch_config_paths
    list[list[pd.Index]],  # batch_particle_indices
    int,  # total_n_particles
    dict[str, torch.Tensor],  # batch_mean_stacks
    dict[str, torch.Tensor],  # batch_std_stacks
    list[str] | None,  # validation_batch_config_paths
    list[list[pd.Index]] | None,  # validation_batch_particle_indices
    int | None,  # validation_total_n_particles
    dict[str, torch.Tensor] | None,  # validation_batch_mean_stacks
    dict[str, torch.Tensor] | None,  # validation_batch_std_stacks
]:
    """Set up common components for sigma optimization.

    Parameters
    ----------
    image : torch.Tensor
        (t, H, W) movie tensor
    var_image : torch.Tensor
        (t, H, W) variance image tensor
    mean_image : torch.Tensor
        (t, H, W) mean image tensor
    refine_config_path : str
        Path to refine config
    optimize_particle_df_path : str | None
        Path to particle dataframe config for validation loss computation.
        The validation template will be loaded from the template_volume_path
        in this YAML file. If None, uses the same particle dataframe and
        template as the motion loop.
    particle_indices : pd.Index | None
        Particle indices to use
    device : torch.device | None
        Device to use
    temp_dir : Path
        Temporary directory for batch configs
    loss_metric : str
        Loss metric: "scaled_mip" or "cross_correlation"
    min_snr : float
        Minimum SNR threshold
    best_n : int
        Maximum number of best particles to use
    optimizer_kwargs : dict[str, Any] | None
        Kwargs for motion optimizer
    particle_batch_size : int
        Batch size for particles
    prior_type : str
        "laplacian" or "relion"
    sigma_a_exponential : bool
        Use exponential sigma_a
    optimize_sigma_a : bool
        Whether to optimize sigma_a
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial
    optimize_sigma_a_amplitude : bool
        Whether to optimize sigma_a_amplitude
    optimize_sigma_a_decay : bool
        Whether to optimize sigma_a_decay
    optimize_sigma_a_offset : bool
        Whether to optimize sigma_a_offset
    optimize_sigma_d : bool
        Whether to optimize sigma_d
    optimize_sigma_v : bool
        Whether to optimize sigma_v
    init_sigma_a : float
        Initial sigma_a value
    init_alpha_spatial : float
        Initial alpha_spatial value
    init_sigma_a_amplitude : float
        Initial sigma_a_amplitude value
    init_sigma_a_decay : float
        Initial sigma_a_decay value
    init_sigma_a_offset : float
        Initial sigma_a_offset value
    init_sigma_d : float
        Initial sigma_d value
    init_sigma_v : float
        Initial sigma_v value
    use_dict : bool
        If True, return initial_values as dict (for optuna).
        If False, return as list (for nelder-mead).
    cleanup_memory : bool
        If True, call torch.cuda.empty_cache() and gc.collect() after
        computing batch stacks. Default False

    Returns
    -------
    tuple
        - refine_config_path: Updated refine config path
        - particle_indices: Filtered particle indices
        - validation_template: Validation template tensor
        - template_volume: Template volume tensor
        - optimizer_kwargs: Optimizer kwargs (with defaults)
        - image: Detached image tensor
        - var_image: Detached var_image tensor
        - mean_image: Detached mean_image tensor
        - param_names: List of parameter names to optimize
        - initial_values: List or dict of initial values
        - sigma_params: Dict of all parameters (optimized and fixed)
        - batch_config_paths: List of batch config paths
        - batch_particle_indices: List of particle indices for each batch
        - total_n_particles: Total number of particles
        - batch_mean_stacks: Dict of mean stacks for each batch
        - batch_std_stacks: Dict of std stacks for each batch
    """
    refine_config_path, particle_indices = _filter_particles_by_quality(
        refine_config_path=refine_config_path,
        particle_indices=particle_indices,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        temp_dir=temp_dir,
    )

    # Load validation template from optimize_particle_df_path if provided,
    # otherwise use the training template
    if optimize_particle_df_path is not None:
        validation_template = load_template_volume_from_config(
            optimize_particle_df_path
        )
    else:
        validation_template = load_template_volume_from_config(refine_config_path)

    # Load training template
    template_volume = load_template_volume_from_config(refine_config_path)

    # Move validation template to device if needed
    if device is not None:
        validation_template = validation_template.to(device)

    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": 0.2}

    if var_image.requires_grad:
        var_image = var_image.clone().detach().requires_grad_(False)
    if mean_image.requires_grad:
        mean_image = mean_image.clone().detach().requires_grad_(False)
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)

    # Build parameter list and initial values
    param_names, initial_values, sigma_params = _collect_sigma_parameters(
        prior_type=prior_type,
        sigma_a_exponential=sigma_a_exponential,
        optimize_sigma_a=optimize_sigma_a,
        optimize_alpha_spatial=optimize_alpha_spatial,
        optimize_sigma_a_amplitude=optimize_sigma_a_amplitude,
        optimize_sigma_a_decay=optimize_sigma_a_decay,
        optimize_sigma_a_offset=optimize_sigma_a_offset,
        optimize_sigma_d=optimize_sigma_d,
        optimize_sigma_v=optimize_sigma_v,
        init_sigma_a=init_sigma_a,
        init_alpha_spatial=init_alpha_spatial,
        init_sigma_a_amplitude=init_sigma_a_amplitude,
        init_sigma_a_decay=init_sigma_a_decay,
        init_sigma_a_offset=init_sigma_a_offset,
        init_sigma_d=init_sigma_d,
        init_sigma_v=init_sigma_v,
        use_dict=use_dict,
    )

    batch_config_paths, batch_particle_indices = _create_batch_configs(
        refine_config_path=refine_config_path,
        particle_batch_size=particle_batch_size,
        temp_dir=temp_dir,
        prefix="train_batch",
    )
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)
    print(
        f"Training particles: total={total_n_particles}, "
        f"batches={len(batch_particle_indices)}"
    )
    for batch_idx, batch_indices in enumerate(batch_particle_indices):
        batch_size = len(batch_indices[0])
        print(f"  Batch {batch_idx + 1}: {batch_size} particles")

    # Pre-compute mean/std stacks for all batches (they don't change across iterations)
    batch_mean_stacks, batch_std_stacks = get_batch_mean_std_stacks(
        batch_config_paths=batch_config_paths,
        batch_particle_indices=batch_particle_indices,
        mean_image=mean_image,
        var_image=var_image,
    )

    # Create validation batch configs if optimize_particle_df_path is provided
    if optimize_particle_df_path is not None:
        validation_batch_config_paths, validation_batch_particle_indices = (
            _create_batch_configs(
                refine_config_path=optimize_particle_df_path,
                particle_batch_size=particle_batch_size,
                temp_dir=temp_dir,
                prefix="val_batch",
            )
        )
        validation_total_n_particles = sum(
            len(indices[0]) for indices in validation_batch_particle_indices
        )
        print(
            f"Validation particles: total={validation_total_n_particles}, "
            f"batches={len(validation_batch_particle_indices)}"
        )
        for batch_idx, batch_indices in enumerate(validation_batch_particle_indices):
            batch_size = len(batch_indices[0])
            print(f"  Validation batch {batch_idx + 1}: {batch_size} particles")
        validation_batch_mean_stacks, validation_batch_std_stacks = (
            get_batch_mean_std_stacks(
                batch_config_paths=validation_batch_config_paths,
                batch_particle_indices=validation_batch_particle_indices,
                mean_image=mean_image,
                var_image=var_image,
            )
        )
    else:
        validation_batch_config_paths = None
        validation_batch_particle_indices = None
        validation_total_n_particles = None
        validation_batch_mean_stacks = None
        validation_batch_std_stacks = None

    if cleanup_memory:
        torch.cuda.empty_cache()
        gc.collect()

    return (
        refine_config_path,
        particle_indices,
        validation_template,
        template_volume,
        optimizer_kwargs,
        image,
        var_image,
        mean_image,
        param_names,
        initial_values,
        sigma_params,
        batch_config_paths,
        batch_particle_indices,
        total_n_particles,
        batch_mean_stacks,
        batch_std_stacks,
        validation_batch_config_paths,
        validation_batch_particle_indices,
        validation_total_n_particles,
        validation_batch_mean_stacks,
        validation_batch_std_stacks,
    )


def _run_inner_optimization_common(
    initial_deformation_field: torch.Tensor | None,
    deformation_field_resolution: tuple[int, int, int],
    device: torch.device,
    optimizer_kwargs: dict[str, Any],
    image: torch.Tensor,
    batch_config_paths: list[str],
    batch_particle_indices: list[list[pd.Index]],
    batch_mean_stacks: dict[str, torch.Tensor],
    batch_std_stacks: dict[str, torch.Tensor],
    template_volume: torch.Tensor,
    pixel_spacing: float,
    pre_exposure: float,
    fluence_per_frame: float,
    motion_iterations: int,
    correlation_batch_size: int,
    total_n_particles: int,
    loss_metric: str,
    prior_type: str,
    sigma_params: dict[str, Any],
    sigma_a_exponential: bool,
    optimization_mode: Literal["deformation_field", "particle_shifts"] = "deformation_field",
    initial_particle_shifts: torch.Tensor | None = None,
    refine_config_path: str | None = None,
) -> tuple[CubicCatmullRomGrid3d | dict[str, torch.Tensor], float]:
    """Run inner motion optimization loop with current sigma parameters.

    Parameters
    ----------
    initial_deformation_field : torch.Tensor | None
        Initial deformation field or None
    deformation_field_resolution : tuple[int, int, int]
        Resolution of deformation field (nt, nh, nw)
    device : torch.device
        Device to use
    optimizer_kwargs : dict[str, Any]
        Optimizer kwargs (must contain "lr")
    image : torch.Tensor
        Input movie (t, H, W)
    batch_config_paths : list[str]
        List of batch config paths
    batch_particle_indices : list[list[pd.Index]]
        List of particle indices for each batch
    batch_mean_stacks : dict[str, torch.Tensor]
        Pre-computed mean stacks for each batch
    batch_std_stacks : dict[str, torch.Tensor]
        Pre-computed std stacks for each batch
    template_volume : torch.Tensor
        Template volume tensor
    pixel_spacing : float
        Pixel spacing in Angstroms
    pre_exposure : float
        Pre-exposure time
    fluence_per_frame : float
        Fluence per frame
    motion_iterations : int
        Number of motion optimization iterations
    correlation_batch_size : int
        Batch size for correlation computation
    total_n_particles : int
        Total number of particles
    loss_metric : str
        Loss metric ("scaled_mip" or "cross_correlation")
    prior_type : str
        Prior type ("laplacian" or "relion")
    sigma_params : dict[str, Any]
        Dictionary of sigma parameters (values can be float or torch.Tensor)
    sigma_a_exponential : bool
        Whether to use exponential sigma_a

    Returns
    -------
    tuple[CubicCatmullRomGrid3d | dict[str, torch.Tensor], float]
        Tuple of (deformation_field or particle_shifts dict, accumulated_loss)
    """
    # Initialize optimization variable based on mode
    particle_coords = None
    if optimization_mode == "particle_shifts":
        assert refine_config_path is not None, (
            "refine_config_path is required for particle_shifts mode"
        )
        # Initialize particle_shifts
        if initial_particle_shifts is None:
            # Compute from initial_deformation_field if available
            if initial_deformation_field is not None:
                particle_shifts = compute_particle_shifts_from_deformation_field(
                    deformation_field=initial_deformation_field,
                    movie=image,
                    refine_config_path=refine_config_path,
                    pixel_spacing=pixel_spacing,
                    grid_type="catmull_rom",  # Default grid type
                    device=device,
                    particle_indices=batch_particle_indices,
                )
                # Detach and require grad to make it a leaf tensor
                particle_shifts = particle_shifts.detach().requires_grad_(True)
            else:
                # Initialize to zero if no deformation field provided
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
        # Initialize deformation field
        if initial_deformation_field is None:
            deformation_field_data = torch.zeros(
                size=(2, *deformation_field_resolution),
                device=device,
                requires_grad=True,
            )
        else:
            deformation_field_data = resample_deformation_field(
                initial_deformation_field, deformation_field_resolution
            )
            deformation_field_data = deformation_field_data - torch.mean(
                deformation_field_data, dim=(1, 2, 3), keepdim=True
            )
            deformation_field_data = deformation_field_data.detach().clone()

        deformation_field_data.requires_grad_(True)
        deformation_field = CubicCatmullRomGrid3d.from_grid_data(
            deformation_field_data
        ).to(device)
        optimization_var = {
            "variable": deformation_field,
            "type": "deformation_field",
            "optimizer_params": deformation_field.parameters(),
            "data": deformation_field._data,
        }

    motion_optimizer = torch.optim.Adam(
        params=optimization_var["optimizer_params"], lr=optimizer_kwargs["lr"]
    )

    # Setup prior params
    prior_params = _setup_prior_params(
        prior_type=prior_type,
        sigma_a_exponential=sigma_a_exponential,
        sigma_params=sigma_params,
        image=image,
        pixel_spacing=pixel_spacing,
        deformation_field_resolution=deformation_field_resolution if optimization_mode == "deformation_field" else None,
        fluence_per_frame=fluence_per_frame,
        device=device,
        optimization_mode=optimization_mode,
        n_particles=total_n_particles if optimization_mode == "particle_shifts" else None,
    )
    # For particle_shifts with RELION prior, update image_coords with particle_coords
    if optimization_mode == "particle_shifts" and prior_type == "relion":
        prior_params["image_coords"] = particle_coords

    # Pre-compute eigendecomposition for RELION prior (coords and sigmas don't change)
    relion_lam = None
    relion_vecs = None
    if prior_type == "relion":
        from ripple.core.motion_priors import relion2019_eigendecompose

        # Determine which coords to use based on optimization mode
        if optimization_mode == "particle_shifts":
            assert particle_coords is not None, (
                "particle_coords is required for particle_shifts mode with relion prior"
            )
            coords_for_eigen = particle_coords
        else:
            assert prior_params["image_coords"] is not None, (
                "image_coords is required for deformation_field mode with relion prior"
            )
            coords_for_eigen = prior_params["image_coords"]

        # Compute eigendecomposition once
        # top_k=0.2 means keep top 20% of modes (default)
        relion_lam, relion_vecs, _ = relion2019_eigendecompose(
            coords_for_eigen,
            prior_params["sigma_d_val"],
            prior_params["sigma_v_norm"],
            top_k=0.2,  # Keep top 20% of modes
        )

    # Inner loop: motion optimization
    accumulated_loss = 0.0
    for iter_idx in range(motion_iterations):
        print(f"  Inner iteration: {iter_idx + 1}/{motion_iterations}")
        motion_optimizer.zero_grad()
        batch_accumulated_loss = 0.0

        for batch_idx, (batch_config_path, batch_indices) in enumerate(
            zip(batch_config_paths, batch_particle_indices, strict=True)
        ):
            batch_size = len(batch_indices[0])
            batch_refine_manager = _make_differentiable_refine_manager(
                batch_config_path
            )
            if iter_idx == 0:  # Only print batch info on first iteration
                # Get actual number of particles from the stack
                actual_particle_count = (
                    batch_refine_manager.particle_stack.num_particles
                )
                print(
                    f"    Training batch {batch_idx + 1}: "
                    f"indices={batch_size}, "
                    f"actual={actual_particle_count}, "
                    f"config={Path(batch_config_path).name}"
                )
            batch_particle_stack = batch_refine_manager.particle_stack

            # Extract particle images based on optimization mode
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
                # Get the particle indices for this batch to slice particle_shifts
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

            # Reuse pre-computed mean/std stacks
            batch_mean_stack = batch_mean_stacks[batch_config_path]
            batch_std_stack = batch_std_stacks[batch_config_path]

            backend_kwargs = batch_refine_manager.make_differentiable_backend_kwargs(
                image_stack=image_stack_batch,
                mean_stack=batch_mean_stack,
                std_stack=batch_std_stack,
                particle_indices=batch_indices,
                template_tensor=template_volume,
                images_are_particles=True,
            )
            result = batch_refine_manager.get_refine_result(
                backend_kwargs, correlation_batch_size, use_differentiable=True
            )

            loss_tensor = (
                result["refined_z_score"]
                if loss_metric == "scaled_mip"
                else result["refined_cross_correlation"]
            )

            # Compute priors based on optimization mode
            if prior_type == "laplacian":
                if optimization_mode == "deformation_field":
                    field_data = optimization_var["data"]
                else:  # particle_shifts
                    # For laplacian, still need to reshape to (2, T, N, 1)
                    particle_shifts = optimization_var["variable"]  # (T, N, 2)
                    field_data = particle_shifts.permute(2, 0, 1).unsqueeze(-1)  # (2, T, N, 1)
                
                e_space, e_time = laplacian_compute(
                    field_data,
                    prior_params["sigma_a_tensor"],
                    prior_params["alpha"],
                    prior_params["spatial_spacing"],
                    prior_params["temporal_spacing"],
                )
            else:  # relion
                if optimization_mode == "deformation_field":
                    field_data = optimization_var["data"]
                    coords = prior_params["image_coords"]
                    e_space, e_time = relion2019_compute(
                        field_data,
                        coords,
                        prior_params["sigma_d_val"],
                        prior_params["sigma_v_norm"],
                        prior_params["sigma_a_norm"],
                        lam=relion_lam,
                        vecs=relion_vecs,
                        is_particle_shifts=False,
                    )
                else:  # particle_shifts
                    particle_shifts = optimization_var["variable"]  # (T, N, 2)
                    # Ensure particle_coords matches particle_shifts dtype
                    particle_coords_matched = particle_coords.to(dtype=particle_shifts.dtype)
                    e_space, e_time = relion2019_compute(
                        particle_shifts,  # (T, N, 2) - pass directly
                        particle_coords_matched,
                        prior_params["sigma_d_val"],
                        prior_params["sigma_v_norm"],
                        prior_params["sigma_a_norm"],
                        lam=relion_lam,
                        vecs=relion_vecs,
                        is_particle_shifts=True,  # Indicate this is particle_shifts format
                    )

            weight = batch_size / total_n_particles
            e_space = e_space * weight
            e_time = e_time * weight
            e_obs = -2 * torch.mean(loss_tensor) * weight

            batch_loss = e_obs + e_space + e_time
            batch_accumulated_loss += batch_loss.item()
            batch_loss.backward()

            # Delete everything
            del (
                image_stack_batch,
                batch_mean_stack,
                batch_std_stack,
                backend_kwargs,
                result,
                loss_tensor,
                batch_loss,
                e_obs,
                e_space,
                e_time,
            )
            del batch_refine_manager, batch_particle_stack

        torch.cuda.empty_cache()
        motion_optimizer.step()
        accumulated_loss += batch_accumulated_loss

        if iter_idx % 3 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Return final result based on mode
    if optimization_mode == "deformation_field":
        # Detach and clone the final deformation field to break computation graph
        final_deformation_data = optimization_var["variable"].data.detach().clone()

        # Clean up optimizer
        del motion_optimizer, optimization_var
        if prior_type == "laplacian":
            if isinstance(prior_params["sigma_a_tensor"], torch.Tensor):
                del prior_params["sigma_a_tensor"]
        else:
            del prior_params["image_coords"]
            if isinstance(prior_params["sigma_a_norm"], torch.Tensor):
                del prior_params["sigma_a_norm"]

        gc.collect()
        torch.cuda.empty_cache()

        deformation_field_clean = CubicCatmullRomGrid3d.from_grid_data(
            final_deformation_data
        ).to(device)

        return deformation_field_clean, accumulated_loss
    else:  # particle_shifts
        final_particle_shifts = optimization_var["variable"].detach().clone()

        # Clean up optimizer
        del motion_optimizer, optimization_var
        if prior_type == "laplacian":
            if isinstance(prior_params["sigma_a_tensor"], torch.Tensor):
                del prior_params["sigma_a_tensor"]
        else:
            if isinstance(prior_params["sigma_a_norm"], torch.Tensor):
                del prior_params["sigma_a_norm"]

        gc.collect()
        torch.cuda.empty_cache()

        return {"particle_shifts": final_particle_shifts}, accumulated_loss
