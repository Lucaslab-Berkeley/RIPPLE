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

    # estimate the motion
    if loss_trajectories:
        if movie_extract:
            (
                updated_deformation_field,
                trajectory,
            ) = estimate_local_motion_2dtm_particles(
                **estimate_kwargs, particle_batch_size=particle_batch_size
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
            updated_deformation_field = estimate_local_motion_2dtm_particles(
                **estimate_kwargs, particle_batch_size=particle_batch_size
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

    # Create new config pointing to filtered CSV
    filtered_config = config.copy()
    filtered_config["particle_stack"] = config["particle_stack"].copy()
    filtered_config["particle_stack"]["df_path"] = str(filtered_csv_path)

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
    all_losses = []

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
        all_losses.append(loss.item())

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
    return final_deformation_field, all_losses


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

    # Get the CSV path from config
    original_csv_path = config["particle_stack"]["df_path"]

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

        # Create batch config (copy of original with updated df_path)
        batch_config = config.copy()
        batch_config["particle_stack"] = config["particle_stack"].copy()
        batch_config["particle_stack"]["df_path"] = str(batch_csv_path)

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
    all_losses = []

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
            all_losses.append(accumulated_loss)

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
    return final_deformation_field, all_losses


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
