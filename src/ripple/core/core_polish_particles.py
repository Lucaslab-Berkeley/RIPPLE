"""Core function for polishing particles."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import einops
import pandas as pd
import torch
import tqdm
import yaml
from leopard_em.pydantic_models.managers import DifferentiableRefineManager
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d
from torch_motion_correction import correct_motion, correct_motion_two_grids
from torch_motion_correction.data_io import write_deformation_field_to_csv
from torch_motion_correction.deformation_field_utils import resample_deformation_field
from torch_motion_correction.optimization_state import OptimizationTracker

from ripple.utils.data_io import load_template_volume_from_config

from .generate_image import dose_weight_memory_efficient
from .prepare_movie import prepare_core
from .motion_priors import (
    _build_physical_coords,
    _compute_physical_spacing,
    _create_exponential_sigma_A,
    _normalize_sigma_fluence,
    laplacian_compute,
    relion2019_compute,
)


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
    # Sigma optimization parameters
    optimize_sigmas: bool = False,
    validation_template_path: str | None = None,
    sigma_iterations: int = 20,
    motion_iterations: int = 10,
    # Sigma optimization output paths
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
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
    optimize_sigmas: bool
        Whether to optimize sigma hyperparameters using validation template.
        Default is False.
    validation_template_path: str | None
        Path to validation template (.mrc) for sigma optimization.
        Required if optimize_sigmas is True.
    sigma_iterations: int
        Number of outer loop iterations for sigma optimization. Default is 20.
    motion_iterations: int
        Number of inner loop motion iterations per sigma update. Default is 10.
    optimized_sigmas_output_path: str | None
        Path to save final optimized sigmas as JSON. Default is None.
    sigma_history_output_path: str | None
        Path to save sigma history (all iterations) as JSON. Default is None.
    training_history_output_path: str | None
        Path to save training loss history as JSON. Default is None.
    validation_history_output_path: str | None
        Path to save validation loss history as JSON. Default is None.

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
    
    # Check if we should run sigma optimization instead
    if optimize_sigmas:
        if validation_template_path is None:
            raise ValueError(
                "validation_template_path must be provided when optimize_sigmas=True"
            )
        
        print("=" * 70)
        print("SIGMA OPTIMIZATION MODE")
        print(f"Validation template: {validation_template_path}")
        print(f"Sigma iterations: {sigma_iterations}")
        print(f"Motion iterations per sigma update: {motion_iterations}")
        print("=" * 70)
        
        result = optimize_sigmas_2dtm_bayesian(
            image=movie_prepared,
            var_image=var_image,
            mean_image=mean_image,
            pixel_spacing=pixel_size,
            deformation_field_resolution=deformation_field_resolution,
            initial_deformation_field=initial_deformation_field,
            refine_config_path=refine_config_path,
            validation_template_path=validation_template_path,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            motion_iterations=motion_iterations,
            sigma_iterations=sigma_iterations,
            optimizer_kwargs=optimizer_kwargs,
            particle_batch_size=particle_batch_size,
            particle_indices=particle_indices,
            device=device,
            loss_metric=loss_metric,
            min_snr=min_snr,
            best_n=best_n,
            # Output paths
            optimized_sigmas_output_path=optimized_sigmas_output_path,
            sigma_history_output_path=sigma_history_output_path,
            training_history_output_path=training_history_output_path,
            validation_history_output_path=validation_history_output_path,
        )
        
        print("\n" + "=" * 70)
        print("SIGMA OPTIMIZATION COMPLETE")
        print("=" * 70)
        print("\nOptimized sigma values:")
        for key, value in result["optimized_sigmas"].items():
            print(f"  {key}: {value:.6f}")
        print(f"\nFinal training loss: {result['training_loss_history'][-1]:.4f}")
        print(f"Final validation loss: {result['validation_loss_history'][-1]:.4f}")
        print("=" * 70)
        
        updated_deformation_field = result["final_deformation_field"]
        trajectory = None
        
        # Correct motion if requested
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

    # estimate the motion
    if loss_trajectories:
        if movie_extract:
            (
                updated_deformation_field,
                trajectory,
            ) = estimate_local_motion_2dtm_particles_bayesian(
                **estimate_kwargs, particle_batch_size=particle_batch_size, pixel_spacing=pixel_size
            )
        else:
            updated_deformation_field, trajectory = estimate_local_motion_2dtm(
                **estimate_kwargs,
                pixel_spacing=pixel_size,
                grid_type=grid_type,
                voltage=voltage,
            )
    else:
        if movie_extract:
            updated_deformation_field = estimate_local_motion_2dtm_particles_bayesian(
                **estimate_kwargs, particle_batch_size=particle_batch_size, pixel_spacing=pixel_size
            )
        else:
            updated_deformation_field = estimate_local_motion_2dtm(
                **estimate_kwargs,
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _filter_particles_by_quality(
    refine_config_path: str,
    particle_indices: list[pd.Index] | None,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    temp_dir: Path | None = None,
) -> tuple[str, list[pd.Index]]:
    """
    Filter particles based on quality metrics and create a temporary config/CSV.

    Parameters
    ----------
    refine_config_path : str
        Path to the refine config YAML file.
    particle_indices : list[pd.Index] | None
        Original particle indices to filter, or None to load all from CSV.
    loss_metric : str
        Metric column name to use for filtering ('mip' or 'scaled_mip').
    min_snr : float
        Minimum value of the loss_metric for a particle to be considered.
    best_n : int
        Maximum number of particles to use, selecting the top N by loss_metric.
    temp_dir : Path | None
        Temporary directory to use. If None, returns original config and indices.

    Returns
    -------
    tuple[str, list[pd.Index]]
        - Path to filtered config YAML (or original if no filtering needed)
        - Filtered particle indices as list[pd.Index] with shape (1, n_filtered)
    """
    # Load the YAML config to get the CSV path
    with open(refine_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    csv_path = config["particle_stack"]["df_path"]

    # Resolve relative paths
    config_dir = Path(refine_config_path).parent
    if not Path(csv_path).is_absolute():
        csv_path = str(config_dir / csv_path)

    # Load the particle dataframe
    df = pd.read_csv(csv_path, index_col=0)

    # If particle_indices provided, filter df to only those indices
    if particle_indices is not None and len(particle_indices) > 0:
        df = df.loc[particle_indices[0]]

    # Filter by minimum SNR if the metric column exists
    needs_filtering = False
    if loss_metric in df.columns:
        df_filtered = df[df[loss_metric] >= min_snr]

        # Select top best_n particles by loss_metric (highest values)
        if len(df_filtered) > best_n:
            df_filtered = df_filtered.nlargest(best_n, loss_metric)

        needs_filtering = len(df_filtered) < len(df)

        print(
            f"Filtered particles: {len(df)} -> {len(df_filtered)} "
            f"(min_{loss_metric}={min_snr}, best_n={best_n})"
        )
    else:
        print(f"Warning: '{loss_metric}' column not found in CSV. Using all particles.")
        df_filtered = df

    # If no filtering needed or no temp_dir provided, return original config
    if not needs_filtering or temp_dir is None:
        return refine_config_path, [df_filtered.index]

    # Create temporary filtered CSV
    filtered_csv_path = temp_dir / "filtered_particles.csv"
    df_filtered.to_csv(filtered_csv_path)

    # Create new config pointing to filtered CSV with absolute paths
    filtered_config = config.copy()
    filtered_config["particle_stack"] = config["particle_stack"].copy()
    filtered_config["particle_stack"]["df_path"] = str(filtered_csv_path)

    # Resolve template_volume_path to absolute
    if "template_volume_path" in filtered_config:
        template_path = Path(config["template_volume_path"])
        if not template_path.is_absolute():
            template_path = (config_dir / template_path).resolve()
        filtered_config["template_volume_path"] = str(template_path)

    filtered_config_path = temp_dir / "filtered_config.yaml"
    with open(filtered_config_path, "w", encoding="utf-8") as f:
        yaml.dump(filtered_config, f)

    # Return indices starting from 0 to match the new CSV
    filtered_indices = pd.Index(range(len(df_filtered)))
    return str(filtered_config_path), [filtered_indices]


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def estimate_local_motion_2dtm(
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

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, OptimizationTracker]
        The estimated deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts. If `return_trajectory` is True, also returns an
        OptimizationTrajectory object containing the optimization history.
    """
    torch.set_grad_enabled(True)

    # Filter particles by quality metrics (no temp CSV needed for non-particles version)
    refine_config_path, particle_indices = _filter_particles_by_quality(
        refine_config_path=refine_config_path,
        particle_indices=particle_indices,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        temp_dir=None,
    )

    # Make sure var and mean image don't have gradients
    template_volume = load_template_volume_from_config(refine_config_path)
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

    # Ensure image requires gradients for optimization
    # Ensure image does NOT require gradients - only deformation field should
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)

    print("Making new deformation field")
    new_deformation_field = CubicCatmullRomGrid3d(
        resolution=deformation_field_resolution, n_channels=2
    ).to(device)
    print("New deformation field made")

    if initial_deformation_field is None:
        deformation_field_data = torch.zeros(
            size=(2, *deformation_field_resolution),
            device=device,
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
        deformation_field_data = deformation_field_data - (
            torch.mean(deformation_field_data, dim=(1, 2, 3), keepdim=True)
        )

    print("Making deformation field")
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

        backend_kwargs = refine_manager.make_backend_core_function_kwargs(
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

        loss = -torch.mean(loss_tensor)
        print(f"loss: {loss}")

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


# pylint: disable=too-many-locals
def _create_batch_configs(
    refine_config_path: str,
    particle_batch_size: int,
    temp_dir: Path,
) -> tuple[list[str], list[list[pd.Index]]]:
    """
    Split the particle CSV into batches and create temporary config files.

    Parameters
    ----------
    refine_config_path : str
        Path to the original refine config YAML file.
    particle_batch_size : int
        Number of particles per batch.
    temp_dir : Path
        Temporary directory to store batch configs and CSVs.

    Returns
    -------
    tuple[list[str], list[list[pd.Index]]]
        - List of paths to batch config YAML files
        - List of batch particle indices.
        Each as list[pd.Index] with shape (1, n_particles_in_batch)
    """
    # Load the original config
    with open(refine_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get base directory of original config for resolving relative paths
    config_base_dir = Path(refine_config_path).parent.resolve()

    def resolve_path(path_str: str | None) -> str | None:
        """Resolve a path relative to the original config directory."""
        if path_str is None:
            return None
        path = Path(path_str)
        if not path.is_absolute():
            path = (config_base_dir / path).resolve()
        return str(path)

    # Get the CSV path from config and resolve it
    original_csv_path = resolve_path(config["particle_stack"]["df_path"])

    # Load the full particle dataframe
    df = pd.read_csv(original_csv_path, index_col=0)
    n_particles = len(df)
    n_batches = (n_particles + particle_batch_size - 1) // particle_batch_size

    batch_config_paths = []
    batch_particle_indices = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * particle_batch_size
        end_idx = min((batch_idx + 1) * particle_batch_size, n_particles)

        # Create batch dataframe
        batch_df = df.iloc[start_idx:end_idx]

        # Save batch CSV (this will have row indices 0 to len(batch_df)-1)
        batch_csv_path = temp_dir / f"batch_{batch_idx}_particles.csv"
        batch_df.to_csv(batch_csv_path)

        # Create batch particle indices
        # Each batch has indices from 0 to n_particles_in_batch
        batch_size = end_idx - start_idx
        batch_indices = pd.Index(range(batch_size))
        batch_particle_indices.append([batch_indices])

        # Create batch config with absolute paths
        batch_config = config.copy()
        batch_config["particle_stack"] = config["particle_stack"].copy()
        batch_config["particle_stack"]["df_path"] = str(batch_csv_path)

        # Resolve template_volume_path to absolute
        if "template_volume_path" in batch_config:
            batch_config["template_volume_path"] = resolve_path(
                config["template_volume_path"]
            )

        # Save batch config
        batch_config_path = temp_dir / f"batch_{batch_idx}_config.yaml"
        with open(batch_config_path, "w", encoding="utf-8") as f:
            yaml.dump(batch_config, f)

        batch_config_paths.append(str(batch_config_path))

    return batch_config_paths, batch_particle_indices


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def estimate_local_motion_2dtm_particles(
    image: torch.Tensor,  # (t, H, W)
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
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

    # Filter particles by quality metrics BEFORE batching
    # This creates a filtered CSV that batching will then split
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

    # Ensure image requires gradients for optimization
    # Ensure image does NOT require gradients - only deformation field should
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)

    if initial_deformation_field is None:
        deformation_field_data = torch.zeros(
            size=(2, *deformation_field_resolution),
            device=device,
            requires_grad=True,
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

        # Ensure gradients are enabled for optimization
        deformation_field_data = (
            deformation_field_data.clone().detach().requires_grad_(True)
        )

    # make the catmull rom grid
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field_data).to(
        device
    )

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

    # Calculate total number of particles across all batches
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)
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

                # 2. Extract mean/std for this batch using batch_indices
                h, w = batch_particle_stack.original_template_size
                box_h, box_w = batch_particle_stack.extracted_box_size
                extracted_box_size = (box_h - h + 1, box_w - w + 1)

                batch_mean_stack = batch_particle_stack.construct_image_stack(
                    images=mean_image,
                    indices=batch_indices,  # Use batch-specific indices
                    extraction_size=extracted_box_size,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="constant",
                    padding_value=0.0,
                )

                batch_std_stack = batch_particle_stack.construct_image_stack(
                    images=var_image,
                    indices=batch_indices,  # Use batch-specific indices
                    extraction_size=extracted_box_size,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="constant",
                    padding_value=1e10,
                )

                # 3. Create backend kwargs for this batch
                backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                    image_stack=image_stack_batch,
                    mean_stack=batch_mean_stack,
                    std_stack=batch_std_stack,
                    particle_indices=batch_indices,
                    images_are_particles=True,
                    template_tensor=template_volume,
                )

                result = batch_refine_manager.get_refine_result(
                    backend_kwargs,
                    correlation_batch_size=correlation_batch_size,
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

                # Compute loss for this batch (weighted by batch size for averaging)
                batch_loss = -torch.mean(loss_tensor) * batch_size / total_n_particles
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


def _make_differentiable_refine_manager(
    refine_config_path: str,
) -> DifferentiableRefineManager:
    """
    Make a differentiable refine manager from a particle results path.

    Parameters
    ----------
    refine_config_path: str
        Path to the refine config file.

    Returns
    -------
    DifferentiableRefineManager
        The differentiable refine manager.
    """
    refine_manager = DifferentiableRefineManager.from_yaml(refine_config_path)
    # override the movie_params here
    refine_manager.movie_config.enabled = False
    return refine_manager

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
    prior_type: str = "laplacian",
    sigma_D: float = 5000.0,
    sigma_V: float = 1.0,
    sigma_A: float = 1.0,
    alpha_spatial: float = 1e5,
    sigma_A_exponential: bool = True,
    sigma_A_amplitude: float = 2.0,
    sigma_A_decay: float = 0.1,
    sigma_A_offset: float = 1.0,
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
    sigma_D: float
        Spatial correlation length in Angstroms (RELION only). Default is 5000.0.
    sigma_V: float
        Velocity magnitude scale in Å per unit fluence (RELION only). Default is 1.0.
    sigma_A_exponential: bool
        Use exponential decay for sigma_A over frames. Default is False.
    sigma_A_amplitude: float
        Amplitude A in sigma_A[f] = A * exp(B * f). Default is 2.0.
    sigma_A_decay: float
        Decay rate B in sigma_A = A*exp(-B*fluence) + C. Default is 0.1 (1/(e-/Å²)).
    sigma_A_offset: float
        Constant offset C in sigma_A = A*exp(-B*fluence) + C. Default is 1.0.
    sigma_A: float
        Temporal smoothness in Å/(e-/Å²). Smaller = smoother. Default is 1.0.
    alpha_spatial: float
        Spatial smoothness strength (Laplacian only). Larger = smoother. Default is 1.0.

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

    # Filter particles by quality metrics BEFORE batching
    # This creates a filtered CSV that batching will then split
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

    # Ensure image requires gradients for optimization
    # Ensure image does NOT require gradients - only deformation field should
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)

    if initial_deformation_field is None:
        deformation_field_data = torch.zeros(
            size=(2, *deformation_field_resolution),
            device=device,
            requires_grad=True,
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

        # Ensure gradients are enabled for optimization
        deformation_field_data = (
            deformation_field_data.clone().detach().requires_grad_(True)
        )

    # make the catmull rom grid
    deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field_data).to(
        device
    )

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
    # Create fluence-dependent sigma_A if requested
    if sigma_A_exponential:
        total_fluence = fluence_per_frame * image.shape[0]
        sigma_A_tensor = _create_exponential_sigma_A(
            total_fluence=total_fluence,
            n_frames=deformation_field_resolution[0],
            A=sigma_A_amplitude,
            B=sigma_A_decay,
            C=sigma_A_offset,
            device=device,
        )
    else:
        sigma_A_tensor = sigma_A
    
    if prior_type == "relion":
        image_coords = _build_physical_coords(
            nh=deformation_field_resolution[1],
            nw=deformation_field_resolution[2],
            image_shape=image.shape[-2:],
            pixel_size=pixel_spacing,
            device=device,
        )
        # Normalize sigma parameters by fluence
        sigma_V_norm = _normalize_sigma_fluence(
            sigma_V, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
        )
        if not sigma_A_exponential:
            sigma_A_norm = _normalize_sigma_fluence(
                sigma_A, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
            )
        else:
            # Normalize the exponential tensor
            sigma_A_norm = _normalize_sigma_fluence(
                sigma_A_tensor, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
            )
    elif prior_type == "laplacian":
        # Compute physical spacing for Laplacian prior
        spatial_spacing, temporal_spacing = _compute_physical_spacing(
            image_shape=image.shape[-2:],
            pixel_size=pixel_spacing,
            grid_resolution=deformation_field_resolution,
            total_fluence=fluence_per_frame * image.shape[0],
        )
        # For Laplacian, use sigma_A_tensor directly (no normalization needed)
        sigma_A_norm = sigma_A_tensor
    else:
        raise ValueError(f"Unknown prior_type: {prior_type}. Must be 'relion' or 'laplacian'")

    # Calculate total number of particles across all batches
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)
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

                # 2. Extract mean/std for this batch using batch_indices
                h, w = batch_particle_stack.original_template_size
                box_h, box_w = batch_particle_stack.extracted_box_size
                extracted_box_size = (box_h - h + 1, box_w - w + 1)

                batch_mean_stack = batch_particle_stack.construct_image_stack(
                    images=mean_image,
                    indices=batch_indices,  # Use batch-specific indices
                    extraction_size=extracted_box_size,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="constant",
                    padding_value=0.0,
                )

                batch_std_stack = batch_particle_stack.construct_image_stack(
                    images=var_image,
                    indices=batch_indices,  # Use batch-specific indices
                    extraction_size=extracted_box_size,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="constant",
                    padding_value=1e10,
                )

                # 3. Create backend kwargs for this batch
                backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                    image_stack=image_stack_batch,
                    mean_stack=batch_mean_stack,
                    std_stack=batch_std_stack,
                    particle_indices=batch_indices,
                    images_are_particles=True,
                    template_tensor=template_volume,
                )

                result = batch_refine_manager.get_refine_result(
                    backend_kwargs,
                    correlation_batch_size=correlation_batch_size,
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

                # Compute motion priors
                if prior_type == "relion":
                    E_space, E_time = relion2019_compute(
                        field=deformation_field._data,
                        coords=image_coords,
                        sigma_D=sigma_D,
                        sigma_V=sigma_V_norm,
                        sigma_A=sigma_A_norm,
                    )
                elif prior_type == "laplacian":
                    E_space, E_time = laplacian_compute(
                        field=deformation_field._data,
                        sigma_A=sigma_A_norm,
                        alpha=alpha_spatial,
                        spatial_spacing=spatial_spacing,
                        temporal_spacing=temporal_spacing,
                    )
                
                E_space = E_space * batch_size / total_n_particles
                E_time = E_time * batch_size / total_n_particles

                # Compute loss for this batch (weighted by batch size for averaging)
                E_obs = -2*torch.mean(loss_tensor) * batch_size / total_n_particles
                batch_loss = E_obs + (E_space + E_time)
                print(f"E_obs: {E_obs.item()}")
                print(f"E_space: {E_space.item()}")
                print(f"E_time: {E_time.item()}")
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def optimize_sigmas_2dtm_bayesian(
    image: torch.Tensor,  # (t, H, W)
    var_image: torch.Tensor,
    mean_image: torch.Tensor,
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    refine_config_path: str,
    validation_template_path: str,
    pre_exposure: float = 0.0,
    fluence_per_frame: float = 1.0,
    motion_iterations: int = 10,
    sigma_iterations: int = 20,
    optimizer_kwargs: dict[str, Any] | None = None,
    sigma_optimizer_kwargs: dict[str, Any] | None = None,
    correlation_batch_size: int = 20,
    particle_batch_size: int = 102,
    particle_indices: pd.Index = None,
    device: torch.device = None,
    loss_metric: str = "scaled_mip",
    min_snr: float = 0.0,
    best_n: int = 10000000000,
    prior_type: str = "relion",
    init_sigma_A: float = 0.8,
    init_alpha_spatial: float = 1e5,
    init_sigma_A_amplitude: float = 2.0,
    init_sigma_A_decay: float = 0.1,
    init_sigma_A_offset: float = 1.0,
    sigma_A_exponential: bool = False,
    init_sigma_D: float = 5000.0,
    init_sigma_V: float = 0.5,
    optimize_sigma_A: bool = True,
    optimize_alpha_spatial: bool = True,
    optimize_sigma_A_amplitude: bool = True,
    optimize_sigma_A_decay: bool = True,
    optimize_sigma_A_offset: bool = True,
    optimize_sigma_D: bool = True,
    optimize_sigma_V: bool = True,
    # Anti-local-minima strategies
    perturbation_interval: int = 0,  # 0 = disabled, otherwise perturb every N iterations
    perturbation_scale: float = 0.1,  # Relative scale of random perturbation (0.1 = 10%)
    verbose: bool = True,
    # Output paths for saving results
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
) -> dict[str, Any]:
    """Optimize prior hyperparameters using a validation template.
    
    Performs bi-level optimization:
    1. Inner loop: Run motion estimation for motion_iterations with current sigmas
    2. Outer loop: Evaluate with validation template and update sigmas
    
    The validation template prevents overfitting of the prior parameters.
    
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
    validation_template_path : str
        Path to validation template (.mrc) for computing validation loss
    motion_iterations : int
        Motion optimization iterations per sigma update. Default 10
    sigma_iterations : int
        Number of sigma optimization iterations. Default 20
    optimizer_kwargs : dict
        Kwargs for motion optimizer. Default {"lr": 0.2}
    sigma_optimizer_kwargs : dict
        Kwargs for sigma optimizer. Default {"lr": 0.01}
    prior_type : str
        "laplacian" or "relion". Default "laplacian"
    init_sigma_A : float
        Initial sigma_A (constant mode). Default 1.0
    init_alpha_spatial : float
        Initial alpha_spatial. Default 1e5
    init_sigma_A_amplitude : float
        Initial A in exponential. Default 2.0
    init_sigma_A_decay : float
        Initial B in exponential. Default 0.1
    init_sigma_A_offset : float
        Initial C in exponential. Default 1.0
    sigma_A_exponential : bool
        Use exponential sigma_A. Default True
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
    """
    import mrcfile
    
    torch.set_grad_enabled(True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_sigma_opt_"))
    
    refine_config_path, particle_indices = _filter_particles_by_quality(
        refine_config_path=refine_config_path,
        particle_indices=particle_indices,
        loss_metric=loss_metric,
        min_snr=min_snr,
        best_n=best_n,
        temp_dir=temp_dir,
    )
    
    with mrcfile.open(validation_template_path, mode='r') as mrc:
        validation_template = torch.tensor(mrc.data.copy(), device=device, dtype=torch.float32)
    
    if optimizer_kwargs is None:
        optimizer_kwargs = {"lr": 0.2}
    if sigma_optimizer_kwargs is None:
        sigma_optimizer_kwargs = {"lr": 0.2}
    
    # Default per-parameter learning rate multipliers (relative to base lr)
    # Large-scale params need larger lr, small-scale params need smaller lr
    default_lr_multipliers = {
        "alpha_spatial": 10000.0,  # ~1e5 scale
        "sigma_D": 1000.0,         # ~5000 scale
        "sigma_V": 1.0,           # ~1 scale
        "sigma_A": 1.0,           # ~1 scale
        "sigma_A_amplitude": 1.0, # ~1-10 scale
        "sigma_A_decay": 0.1,     # ~0.1 scale
        "sigma_A_offset": 1.0,    # ~1 scale
    }
    
    if var_image.requires_grad:
        var_image = var_image.clone().detach().requires_grad_(False)
    if mean_image.requires_grad:
        mean_image = mean_image.clone().detach().requires_grad_(False)
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)
    
    # Initialize sigma parameters with per-parameter learning rates
    sigma_params = {}
    param_groups = []  # List of {"params": [tensor], "lr": lr} dicts
    base_lr = sigma_optimizer_kwargs.get("lr", 0.1)
    
    if sigma_A_exponential:
        if optimize_sigma_A_amplitude:
            sigma_params["sigma_A_amplitude"] = torch.tensor(
                init_sigma_A_amplitude, device=device, requires_grad=True, dtype=torch.float32
            )
            param_groups.append({
                "params": [sigma_params["sigma_A_amplitude"]],
                "lr": base_lr * default_lr_multipliers["sigma_A_amplitude"],
            })
        else:
            sigma_params["sigma_A_amplitude"] = init_sigma_A_amplitude
            
        if optimize_sigma_A_decay:
            sigma_params["sigma_A_decay"] = torch.tensor(
                init_sigma_A_decay, device=device, requires_grad=True, dtype=torch.float32
            )
            param_groups.append({
                "params": [sigma_params["sigma_A_decay"]],
                "lr": base_lr * default_lr_multipliers["sigma_A_decay"],
            })
        else:
            sigma_params["sigma_A_decay"] = init_sigma_A_decay
            
        if optimize_sigma_A_offset:
            sigma_params["sigma_A_offset"] = torch.tensor(
                init_sigma_A_offset, device=device, requires_grad=True, dtype=torch.float32
            )
            param_groups.append({
                "params": [sigma_params["sigma_A_offset"]],
                "lr": base_lr * default_lr_multipliers["sigma_A_offset"],
            })
        else:
            sigma_params["sigma_A_offset"] = init_sigma_A_offset
    else:
        if optimize_sigma_A:
            sigma_params["sigma_A"] = torch.tensor(
                init_sigma_A, device=device, requires_grad=True, dtype=torch.float32
            )
            param_groups.append({
                "params": [sigma_params["sigma_A"]],
                "lr": base_lr * default_lr_multipliers["sigma_A"],
            })
        else:
            sigma_params["sigma_A"] = init_sigma_A
    
    if prior_type == "laplacian" and optimize_alpha_spatial:
        sigma_params["alpha_spatial"] = torch.tensor(
            init_alpha_spatial, device=device, requires_grad=True, dtype=torch.float32
        )
        param_groups.append({
            "params": [sigma_params["alpha_spatial"]],
            "lr": base_lr * default_lr_multipliers["alpha_spatial"],
        })
    else:
        sigma_params["alpha_spatial"] = init_alpha_spatial
    
    if prior_type == "relion":
        if optimize_sigma_D:
            sigma_params["sigma_D"] = torch.tensor(
                init_sigma_D, device=device, requires_grad=True, dtype=torch.float32
            )
            param_groups.append({
                "params": [sigma_params["sigma_D"]],
                "lr": base_lr * default_lr_multipliers["sigma_D"],
            })
        else:
            sigma_params["sigma_D"] = init_sigma_D
        
        if optimize_sigma_V:
            sigma_params["sigma_V"] = torch.tensor(
                init_sigma_V, device=device, requires_grad=True, dtype=torch.float32
            )
            param_groups.append({
                "params": [sigma_params["sigma_V"]],
                "lr": base_lr * default_lr_multipliers["sigma_V"],
            })
        else:
            sigma_params["sigma_V"] = init_sigma_V
    
    if len(param_groups) == 0:
        raise ValueError("No sigma parameters selected for optimization!")
    
    # Create optimizer with per-parameter learning rates
    sigma_optimizer = torch.optim.Adam(param_groups)
    
    batch_config_paths, batch_particle_indices = _create_batch_configs(
        refine_config_path=refine_config_path,
        particle_batch_size=particle_batch_size,
        temp_dir=temp_dir,
    )
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)
    
    validation_loss_history = []
    training_loss_history = []
    sigma_history = []
    
    def get_val(key):
        v = sigma_params.get(key)
        return v.abs() if isinstance(v, torch.Tensor) else v
    
    try:
        # Compute initial validation loss (before any optimization)
        print("Computing initial validation loss (with initial deformation field)...")
        initial_validation_loss = 0.0
        with torch.no_grad():
            # Use initial deformation field or zeros if none provided
            if initial_deformation_field is None:
                init_field_data = torch.zeros(
                    size=(2, *deformation_field_resolution), device=device
                )
            else:
                init_field_data = resample_deformation_field(
                    initial_deformation_field, deformation_field_resolution
                )
                init_field_data = init_field_data - torch.mean(
                    init_field_data, dim=(1, 2, 3), keepdim=True
                )
            init_deformation_field = CubicCatmullRomGrid3d.from_grid_data(init_field_data).to(device)
            
            for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
                batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
                batch_particle_stack = batch_refine_manager.particle_stack
                batch_size = len(batch_indices[0])
                
                image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                    movie=image, deformation_field=init_deformation_field,
                    pos_reference="top-left", handle_bounds="pad",
                    padding_mode="reflect", padding_value=0.0,
                    pre_exposure=pre_exposure, fluence_per_frame=fluence_per_frame
                )
                
                h, w = batch_particle_stack.original_template_size
                box_h, box_w = batch_particle_stack.extracted_box_size
                extracted_box_size = (box_h - h + 1, box_w - w + 1)
                
                batch_mean_stack = batch_particle_stack.construct_image_stack(
                    images=mean_image,
                    indices=batch_indices,
                    extraction_size=extracted_box_size,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="constant",
                    padding_value=0.0,
                )
                batch_std_stack = batch_particle_stack.construct_image_stack(
                    images=var_image,
                    indices=batch_indices,
                    extraction_size=extracted_box_size,
                    pos_reference="top-left",
                    handle_bounds="pad",
                    padding_mode="constant",
                    padding_value=1e10,
                )
                
                backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                    image_stack=image_stack_batch,
                    mean_stack=batch_mean_stack,
                    std_stack=batch_std_stack,
                    particle_indices=batch_indices,
                    template_tensor=validation_template,
                    images_are_particles=True,
                )
                result = batch_refine_manager.get_refine_result(backend_kwargs, correlation_batch_size)
                
                val_loss = result["refined_z_score"] if loss_metric == "scaled_mip" else result["refined_cross_correlation"]
                initial_validation_loss += -torch.mean(val_loss).item() * batch_size / total_n_particles
                
                del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                torch.cuda.empty_cache()
        
        print(f"Initial validation loss (with initial field): {initial_validation_loss:.6f}")
        validation_loss_history.append(initial_validation_loss)  # Store as iteration -1
        
        sigma_pbar = tqdm.tqdm(range(sigma_iterations), desc="Sigma optimization")
        
        for sigma_iter in sigma_pbar:
            print(f"sigma_iter: {sigma_iter}")
            # Initialize deformation field
            if initial_deformation_field is None:
                deformation_field_data = torch.zeros(
                    size=(2, *deformation_field_resolution), device=device, requires_grad=True
                )
            else:
                deformation_field_data = resample_deformation_field(
                    initial_deformation_field, deformation_field_resolution
                )
                deformation_field_data = deformation_field_data - torch.mean(
                    deformation_field_data, dim=(1, 2, 3), keepdim=True
                )
                deformation_field_data = deformation_field_data.clone().detach().requires_grad_(True)
            
            deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field_data).to(device)
            motion_optimizer = torch.optim.Adam(deformation_field.parameters(), lr=optimizer_kwargs["lr"])
            
            # Setup prior params
            if prior_type == "laplacian":
                spatial_spacing, temporal_spacing = _compute_physical_spacing(
                    image.shape[-2:], pixel_spacing, deformation_field_resolution,
                    fluence_per_frame * image.shape[0]
                )
                if sigma_A_exponential:
                    A = get_val("sigma_A_amplitude")
                    B = get_val("sigma_A_decay")
                    C = get_val("sigma_A_offset")
                    A = A.item() if isinstance(A, torch.Tensor) else A
                    B = B.item() if isinstance(B, torch.Tensor) else B
                    C = C.item() if isinstance(C, torch.Tensor) else C
                    sigma_A_tensor = _create_exponential_sigma_A(
                        fluence_per_frame * image.shape[0], deformation_field_resolution[0],
                        A=A, B=B, C=C, device=device
                    )
                else:
                    sigma_A_tensor = get_val("sigma_A")
                    sigma_A_tensor = sigma_A_tensor.item() if isinstance(sigma_A_tensor, torch.Tensor) else sigma_A_tensor
                alpha = get_val("alpha_spatial")
                alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            elif prior_type == "relion":
                image_coords = _build_physical_coords(
                    deformation_field_resolution[1], deformation_field_resolution[2],
                    image.shape[-2:], pixel_spacing, device
                )
                # Get sigma_D and sigma_V values (may be tensors if optimizing)
                sigma_D_val = get_val("sigma_D")
                sigma_D_val = sigma_D_val.item() if isinstance(sigma_D_val, torch.Tensor) else sigma_D_val
                sigma_V_val = get_val("sigma_V")
                sigma_V_val = sigma_V_val.item() if isinstance(sigma_V_val, torch.Tensor) else sigma_V_val
                sigma_V_norm = _normalize_sigma_fluence(
                    sigma_V_val, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
                )
                if sigma_A_exponential:
                    A = get_val("sigma_A_amplitude")
                    B = get_val("sigma_A_decay")
                    C = get_val("sigma_A_offset")
                    A = A.item() if isinstance(A, torch.Tensor) else A
                    B = B.item() if isinstance(B, torch.Tensor) else B
                    C = C.item() if isinstance(C, torch.Tensor) else C
                    sigma_A_tensor = _create_exponential_sigma_A(
                        fluence_per_frame * image.shape[0], deformation_field_resolution[0],
                        A=A, B=B, C=C, device=device
                    )
                    sigma_A_norm = _normalize_sigma_fluence(
                        sigma_A_tensor, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
                    )
                else:
                    sa = get_val("sigma_A")
                    sa = sa.item() if isinstance(sa, torch.Tensor) else sa
                    sigma_A_norm = _normalize_sigma_fluence(
                        sa, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
                    )
            
            # Inner loop: motion optimization
            for iter_idx in range(motion_iterations):
                print(f"motion_iter: {iter_idx}")
                motion_optimizer.zero_grad()
                accumulated_loss = 0.0
                
                for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
                    batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
                    batch_particle_stack = batch_refine_manager.particle_stack
                    batch_size = len(batch_indices[0])
                    
                    image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                        movie=image, deformation_field=deformation_field,
                        pos_reference="top-left", handle_bounds="pad",
                        padding_mode="reflect", padding_value=0.0,
                        pre_exposure=pre_exposure, fluence_per_frame=fluence_per_frame
                    )
                    
                    h, w = batch_particle_stack.original_template_size
                    box_h, box_w = batch_particle_stack.extracted_box_size
                    extracted_box_size = (box_h - h + 1, box_w - w + 1)
                    
                    batch_mean_stack = batch_particle_stack.construct_image_stack(
                        images=mean_image,
                        indices=batch_indices,
                        extraction_size=extracted_box_size,
                        pos_reference="top-left",
                        handle_bounds="pad",
                        padding_mode="constant",
                        padding_value=0.0,
                    )
                    batch_std_stack = batch_particle_stack.construct_image_stack(
                        images=var_image,
                        indices=batch_indices,
                        extraction_size=extracted_box_size,
                        pos_reference="top-left",
                        handle_bounds="pad",
                        padding_mode="constant",
                        padding_value=1e10,
                    )
                    
                    template_volume = load_template_volume_from_config(batch_config_path)
                    backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                        image_stack=image_stack_batch,
                        mean_stack=batch_mean_stack,
                        std_stack=batch_std_stack,
                        particle_indices=batch_indices,
                        template_tensor=template_volume,
                        images_are_particles=True,
                    )
                    result = batch_refine_manager.get_refine_result(backend_kwargs, correlation_batch_size)
                    
                    loss_tensor = result["refined_z_score"] if loss_metric == "scaled_mip" else result["refined_cross_correlation"]
                    
                    if prior_type == "laplacian":
                        E_space, E_time = laplacian_compute(
                            deformation_field._data, sigma_A_tensor, alpha, spatial_spacing, temporal_spacing
                        )
                    else:
                        E_space, E_time = relion2019_compute(
                            deformation_field._data, image_coords, sigma_D_val, sigma_V_norm, sigma_A_norm
                        )
                    
                    E_space = E_space * batch_size / total_n_particles
                    E_time = E_time * batch_size / total_n_particles
                    E_obs = -2 * torch.mean(loss_tensor) * batch_size / total_n_particles
                    
                    batch_loss = E_obs + E_space + E_time
                    print(f"E_obs: {E_obs.item()}")
                    print(f"E_space: {E_space.item()}")
                    print(f"E_time: {E_time.item()}")
                    accumulated_loss += batch_loss.item()
                    print(f"batch_loss: {batch_loss.item()}")
                    batch_loss.backward()
                    
                    del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                    torch.cuda.empty_cache()
                
                motion_optimizer.step()
            
            training_loss_history.append(accumulated_loss)
            
            # Validation with held-out template
            validation_loss = 0.0
            with torch.no_grad():
                for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
                    batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
                    batch_particle_stack = batch_refine_manager.particle_stack
                    batch_size = len(batch_indices[0])
                    
                    image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                        movie=image, deformation_field=deformation_field,
                        pos_reference="top-left", handle_bounds="pad",
                        padding_mode="reflect", padding_value=0.0,
                        pre_exposure=pre_exposure, fluence_per_frame=fluence_per_frame
                    )
                    
                    h, w = batch_particle_stack.original_template_size
                    box_h, box_w = batch_particle_stack.extracted_box_size
                    extracted_box_size = (box_h - h + 1, box_w - w + 1)
                    
                    batch_mean_stack = batch_particle_stack.construct_image_stack(
                        images=mean_image,
                        indices=batch_indices,
                        extraction_size=extracted_box_size,
                        pos_reference="top-left",
                        handle_bounds="pad",
                        padding_mode="constant",
                        padding_value=0.0,
                    )
                    batch_std_stack = batch_particle_stack.construct_image_stack(
                        images=var_image,
                        indices=batch_indices,
                        extraction_size=extracted_box_size,
                        pos_reference="top-left",
                        handle_bounds="pad",
                        padding_mode="constant",
                        padding_value=1e10,
                    )
                    
                    backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                        image_stack=image_stack_batch,
                        mean_stack=batch_mean_stack,
                        std_stack=batch_std_stack,
                        particle_indices=batch_indices,
                        template_tensor=validation_template,
                        images_are_particles=True,
                    )
                    result = batch_refine_manager.get_refine_result(backend_kwargs, correlation_batch_size)
                    
                    val_loss = result["refined_z_score"] if loss_metric == "scaled_mip" else result["refined_cross_correlation"]
                    validation_loss += -torch.mean(val_loss).item() * batch_size / total_n_particles
                    
                    del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                    torch.cuda.empty_cache()
            
            print(f"validation_loss: {validation_loss}")

            validation_loss_history.append(validation_loss)

            # Update sigmas based on validation loss trend
            sigma_optimizer.zero_grad()
            if len(validation_loss_history) > 1:
                delta = validation_loss_history[-1] - validation_loss_history[-2]
                for group in param_groups:
                    for param in group["params"]:
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        param.grad.fill_(delta * 0.1)
                sigma_optimizer.step()
                with torch.no_grad():
                    for group in param_groups:
                        for param in group["params"]:
                            param.clamp_(min=1e-6)
            
            # Apply random perturbation to escape local minima
            if perturbation_interval > 0 and (sigma_iter + 1) % perturbation_interval == 0:
                print(f"Applying random perturbation (scale={perturbation_scale})")
                with torch.no_grad():
                    for group in param_groups:
                        for param in group["params"]:
                            noise = torch.randn_like(param) * param.abs() * perturbation_scale
                            param.add_(noise)
                            param.clamp_(min=1e-6)

            # Record history
            current_sigmas = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in sigma_params.items()}
            print(f"current_sigmas: {current_sigmas}")
            sigma_history.append(current_sigmas.copy())
            
            if verbose:
                sigma_pbar.set_postfix({"train": f"{accumulated_loss:.3f}", "val": f"{validation_loss:.3f}"})
        
        final_deformation_field = deformation_field.data
        final_deformation_field = final_deformation_field - torch.mean(final_deformation_field, dim=(1, 2, 3), keepdim=True)
        
        optimized_sigmas = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in sigma_params.items()}
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"Cleaned up temporary configs at {temp_dir}")
    
    # Save results to files if paths are specified
    import json
    if optimized_sigmas_output_path is not None:
        with open(optimized_sigmas_output_path, 'w', encoding='utf-8') as f:
            json.dump(optimized_sigmas, f, indent=2)
        if verbose:
            print(f"Saved optimized sigmas to: {optimized_sigmas_output_path}")

    if sigma_history_output_path is not None:
        with open(sigma_history_output_path, 'w', encoding='utf-8') as f:
            json.dump(sigma_history, f, indent=2)
        if verbose:
            print(f"Saved sigma history to: {sigma_history_output_path}")

    if training_history_output_path is not None:
        with open(training_history_output_path, 'w', encoding='utf-8') as f:
            json.dump(training_loss_history, f, indent=2)
        if verbose:
            print(f"Saved training history to: {training_history_output_path}")

    if validation_history_output_path is not None:
        with open(validation_history_output_path, 'w', encoding='utf-8') as f:
            json.dump(validation_loss_history, f, indent=2)
        if verbose:
            print(f"Saved validation history to: {validation_history_output_path}")
    
    return {
        "optimized_sigmas": optimized_sigmas,
        "final_deformation_field": final_deformation_field,
        "validation_loss_history": validation_loss_history,
        "training_loss_history": training_loss_history,
        "sigma_history": sigma_history,
    }
