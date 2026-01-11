"""Optimization functions for sigma parameter tuning."""

import itertools
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal
import time
import einops
import numpy as np
import optuna
import pandas as pd
import torch
import tqdm
import yaml
from scipy.optimize import minimize
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

from .motion_priors import (
    _build_physical_coords,
    _compute_physical_spacing,
    _create_exponential_sigma_A,
    _normalize_sigma_fluence,
    laplacian_compute,
    relion2019_compute,
)
from ripple.utils.data_io import load_template_volume_from_config
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_motion_correction.deformation_field_utils import resample_deformation_field


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
    init_sigma_A: float = 0.513517,
    init_alpha_spatial: float = 1e5,
    init_sigma_A_amplitude: float = 2.0,
    init_sigma_A_decay: float = 0.1,
    init_sigma_A_offset: float = 1.0,
    sigma_A_exponential: bool = False,
    init_sigma_D: float = 5782.376953,
    init_sigma_V: float = 0.194826,
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
        "sigma_D": 2000.0,         # ~5000 scale
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

    # Pre-compute mean/std stacks for all batches (they don't change across iterations)
    batch_mean_stacks = {}
    batch_std_stacks = {}
    for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
        batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
        batch_particle_stack = batch_refine_manager.particle_stack
            
        h, w = batch_particle_stack.original_template_size
        box_h, box_w = batch_particle_stack.extracted_box_size
        extracted_box_size = (box_h - h + 1, box_w - w + 1)
            
        batch_mean_stacks[batch_config_path] = batch_particle_stack.construct_image_stack(
            images=mean_image,
            indices=batch_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=0.0,
        )
        batch_std_stacks[batch_config_path] = batch_particle_stack.construct_image_stack(
            images=var_image,
            indices=batch_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=1e10,
        )
        
    
    validation_loss_history = []
    training_loss_history = []
    sigma_history = []
    
    # Best-point tracking
    best_validation_loss = None
    best_sigma_iter = None
    best_sigma_params = None
    
    def get_val(key):
        v = sigma_params.get(key)
        return v.abs() if isinstance(v, torch.Tensor) else v
    
    def compute_validation_loss(deformation_field_to_use):
        """Compute validation loss with current sigma parameters."""
        val_loss = 0.0
        with torch.no_grad():
            for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
                batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
                batch_particle_stack = batch_refine_manager.particle_stack
                batch_size = len(batch_indices[0])
                
                image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                    movie=image, deformation_field=deformation_field_to_use,
                    pos_reference="top-left", handle_bounds="pad",
                    padding_mode="reflect", padding_value=0.0,
                    pre_exposure=pre_exposure, fluence_per_frame=fluence_per_frame
                )
                
                # Reuse pre-computed mean/std stacks (same as motion loop)
                batch_mean_stack = batch_mean_stacks[batch_config_path]
                batch_std_stack = batch_std_stacks[batch_config_path]
                
                backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                    image_stack=image_stack_batch,
                    mean_stack=batch_mean_stack,
                    std_stack=batch_std_stack,
                    particle_indices=batch_indices,
                    template_tensor=validation_template,
                    images_are_particles=True,
                )
                result = batch_refine_manager.get_refine_result(backend_kwargs, correlation_batch_size)
                
                val_loss_tensor = result["refined_z_score"] if loss_metric == "scaled_mip" else result["refined_cross_correlation"]
                val_loss += -torch.mean(val_loss_tensor).item() * batch_size / total_n_particles
                
                del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                torch.cuda.empty_cache()
        return val_loss
    
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
                
                # Reuse pre-computed mean/std stacks (same as motion loop)
                batch_mean_stack = batch_mean_stacks[batch_config_path]
                batch_std_stack = batch_std_stacks[batch_config_path]
                
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
        
        def run_inner_optimization():
            """Run inner motion optimization loop with current sigma parameters.
            
            Returns:
                tuple: (deformation_field, accumulated_loss)
            """
            # Load template volume once (same for all batches)
            template_volume = load_template_volume_from_config(refine_config_path)
            
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
            for _ in range(motion_iterations):
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
                    # Reuse pre-computed mean/std stacks
                    batch_mean_stack = batch_mean_stacks[batch_config_path]
                    batch_std_stack = batch_std_stacks[batch_config_path]

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
                    accumulated_loss += batch_loss.item()
                    batch_loss.backward()
                    
                    del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                    torch.cuda.empty_cache()
                
                motion_optimizer.step()
            
            return deformation_field, accumulated_loss
        
        sigma_pbar = tqdm.tqdm(range(sigma_iterations), desc="Sigma optimization")
        
        for sigma_iter in sigma_pbar:
            print(f"sigma_iter: {sigma_iter}")
            
            # Run inner optimization with current sigma parameters
            deformation_field, accumulated_loss = run_inner_optimization()
            training_loss_history.append(accumulated_loss)
            
            # Validation with held-out template
            validation_loss = compute_validation_loss(deformation_field)
            print(f"validation_loss: {validation_loss}")

            validation_loss_history.append(validation_loss)
            
            # Best-point tracking
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_sigma_iter = sigma_iter
                best_sigma_params = {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in sigma_params.items()
                }
                if verbose:
                    print(f"✓ New best validation loss: {best_validation_loss:.6f} at iteration {best_sigma_iter}")

            # Update sigmas using per-parameter finite differences
            sigma_optimizer.zero_grad()
            if sigma_iter > 0:  # Need at least one previous iteration for comparison
                # Compute gradients for each parameter using finite differences
                # This is expensive but necessary since gradients don't flow through the validation loss
                finite_diff_step_relative = 1e-3  # Relative step size (0.1% of parameter value)
                
                if verbose:
                    print("Computing per-parameter gradients using finite differences...")
                
                for group in param_groups:
                    for param in group["params"]:
                        # Find which parameter this is
                        param_name = None
                        for name, p in sigma_params.items():
                            if p is param:
                                param_name = name
                                break
                        
                        if param_name is None:
                            continue
                        
                        # Store original value
                        param_val = param.item()
                        
                        # Compute step size (relative to parameter magnitude)
                        step = max(abs(param_val) * finite_diff_step_relative, 1e-6)
                        
                        if verbose:
                            print(f"  Computing gradient for {param_name} (current={param_val:.6f}, step={step:.6f})")
                        
                        # Perturb parameter upward and re-optimize
                        param.data.fill_(param_val + step)
                        # Re-run inner optimization with perturbed sigma
                        deformation_field_plus, _ = run_inner_optimization()
                        loss_plus = compute_validation_loss(deformation_field_plus)
                        
                        # Perturb parameter downward and re-optimize
                        param.data.fill_(param_val - step)
                        deformation_field_minus, _ = run_inner_optimization()
                        loss_minus = compute_validation_loss(deformation_field_minus)
                        
                        # Restore original parameter value
                        param.data.fill_(param_val)
                        
                        # Compute gradient: (f(x+h) - f(x-h)) / (2h)
                        grad = (loss_plus - loss_minus) / (2 * step)
                        param.grad = torch.tensor(grad, device=param.device, dtype=param.dtype)
                        
                        if verbose:
                            print(f"    loss_plus={loss_plus:.6f}, loss_minus={loss_minus:.6f}, grad={grad:.6f}")
                        
                        # Clean up
                        del deformation_field_plus, deformation_field_minus
                        torch.cuda.empty_cache()
                
                # Take optimizer step with computed gradients
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
        
        # Use best point if it's better than final point
        final_validation_loss = validation_loss_history[-1]
        if best_validation_loss is not None and final_validation_loss > best_validation_loss:
            if verbose:
                print(f"\nFinal validation loss ({final_validation_loss:.6f}) is worse than best ({best_validation_loss:.6f})")
                print(f"Restoring best parameters from iteration {best_sigma_iter}")
            # Restore best parameters
            for name, best_val in best_sigma_params.items():
                if name in sigma_params and isinstance(sigma_params[name], torch.Tensor):
                    sigma_params[name].data.fill_(best_val)
            # Re-run inner optimization with best parameters to get best deformation field
            deformation_field, _ = run_inner_optimization()
        
        final_deformation_field = deformation_field.data
        final_deformation_field = final_deformation_field - torch.mean(final_deformation_field, dim=(1, 2, 3), keepdim=True)
        
        optimized_sigmas = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in sigma_params.items()}
        
        # Print summary
        if verbose and best_validation_loss is not None:
            print("\n" + "=" * 70)
            print("OPTIMIZATION SUMMARY")
            print("=" * 70)
            print(f"Best validation loss: {best_validation_loss:.6f} (iteration {best_sigma_iter})")
            print(f"Final validation loss: {validation_loss_history[-1]:.6f}")
            print(f"Best parameters:")
            for name, val in best_sigma_params.items():
                print(f"  {name}: {val:.6f}")
            print("=" * 70)
        
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
    
    result = {
        "optimized_sigmas": optimized_sigmas,
        "final_deformation_field": final_deformation_field,
        "validation_loss_history": validation_loss_history,
        "training_loss_history": training_loss_history,
        "sigma_history": sigma_history,
    }
    
    # Add best-point information
    if best_validation_loss is not None:
        result["best_validation_loss"] = best_validation_loss
        result["best_sigma_iter"] = best_sigma_iter
        result["best_sigma_params"] = best_sigma_params
    
    return result


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def optimize_sigmas_2dtm_nelder_mead(
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
    init_sigma_A: float = 0.513517,
    init_alpha_spatial: float = 1e5,
    init_sigma_A_amplitude: float = 2.0,
    init_sigma_A_decay: float = 0.1,
    init_sigma_A_offset: float = 1.0,
    sigma_A_exponential: bool = False,
    init_sigma_D: float = 5782.376953,
    init_sigma_V: float = 0.194826,
    optimize_sigma_A: bool = True,
    optimize_alpha_spatial: bool = True,
    optimize_sigma_A_amplitude: bool = True,
    optimize_sigma_A_decay: bool = True,
    optimize_sigma_A_offset: bool = True,
    optimize_sigma_D: bool = True,
    optimize_sigma_V: bool = True,
    verbose: bool = True,
    # Output paths for saving results
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
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
    validation_template_path : str
        Path to validation template (.mrc) for computing validation loss
    motion_iterations : int
        Motion optimization iterations per sigma update. Default 10
    sigma_iterations : int
        Maximum number of Nelder-Mead iterations. Default 20
    optimizer_kwargs : dict
        Kwargs for motion optimizer. Default {"lr": 0.2}
    sigma_optimizer_kwargs : dict
        Not used for Nelder-Mead, kept for API compatibility
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
    init_sigma_A : float
        Initial sigma_A (constant mode). Default 0.88
    init_alpha_spatial : float
        Initial alpha_spatial. Default 1e5
    init_sigma_A_amplitude : float
        Initial A in exponential. Default 2.0
    init_sigma_A_decay : float
        Initial B in exponential. Default 0.1
    init_sigma_A_offset : float
        Initial C in exponential. Default 1.0
    sigma_A_exponential : bool
        Use exponential sigma_A. Default False
    init_sigma_D : float
        Initial sigma_D. Default 5080.0
    init_sigma_V : float
        Initial sigma_V. Default 0.58
    optimize_sigma_A : bool
        Whether to optimize sigma_A. Default True
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial. Default True
    optimize_sigma_A_amplitude : bool
        Whether to optimize sigma_A_amplitude. Default True
    optimize_sigma_A_decay : bool
        Whether to optimize sigma_A_decay. Default True
    optimize_sigma_A_offset : bool
        Whether to optimize sigma_A_offset. Default True
    optimize_sigma_D : bool
        Whether to optimize sigma_D. Default True
    optimize_sigma_V : bool
        Whether to optimize sigma_V. Default True
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
    import mrcfile
    
    torch.set_grad_enabled(True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_sigma_opt_nelder_"))
    
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
    
    if var_image.requires_grad:
        var_image = var_image.clone().detach().requires_grad_(False)
    if mean_image.requires_grad:
        mean_image = mean_image.clone().detach().requires_grad_(False)
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)
    
    # Build parameter list and initial values
    param_names = []
    initial_values = []
    sigma_params = {}  # Store all parameters (both optimized and fixed)
    
    # Collect parameters to optimize
    if sigma_A_exponential:
        if optimize_sigma_A_amplitude:
            param_names.append("sigma_A_amplitude")
            initial_values.append(init_sigma_A_amplitude)
            sigma_params["sigma_A_amplitude"] = init_sigma_A_amplitude
        else:
            sigma_params["sigma_A_amplitude"] = init_sigma_A_amplitude
            
        if optimize_sigma_A_decay:
            param_names.append("sigma_A_decay")
            initial_values.append(init_sigma_A_decay)
            sigma_params["sigma_A_decay"] = init_sigma_A_decay
        else:
            sigma_params["sigma_A_decay"] = init_sigma_A_decay
            
        if optimize_sigma_A_offset:
            param_names.append("sigma_A_offset")
            initial_values.append(init_sigma_A_offset)
            sigma_params["sigma_A_offset"] = init_sigma_A_offset
        else:
            sigma_params["sigma_A_offset"] = init_sigma_A_offset
    else:
        if optimize_sigma_A:
            param_names.append("sigma_A")
            initial_values.append(init_sigma_A)
            sigma_params["sigma_A"] = init_sigma_A
        else:
            sigma_params["sigma_A"] = init_sigma_A
    
    if prior_type == "laplacian" and optimize_alpha_spatial:
        param_names.append("alpha_spatial")
        initial_values.append(init_alpha_spatial)
        sigma_params["alpha_spatial"] = init_alpha_spatial
    else:
        sigma_params["alpha_spatial"] = init_alpha_spatial
    
    if prior_type == "relion":
        if optimize_sigma_D:
            param_names.append("sigma_D")
            initial_values.append(init_sigma_D)
            sigma_params["sigma_D"] = init_sigma_D
        else:
            sigma_params["sigma_D"] = init_sigma_D
            
        if optimize_sigma_V:
            param_names.append("sigma_V")
            initial_values.append(init_sigma_V)
            sigma_params["sigma_V"] = init_sigma_V
        else:
            sigma_params["sigma_V"] = init_sigma_V
    
    if len(param_names) == 0:
        raise ValueError("No sigma parameters selected for optimization!")
    
    batch_config_paths, batch_particle_indices = _create_batch_configs(
        refine_config_path=refine_config_path,
        particle_batch_size=particle_batch_size,
        temp_dir=temp_dir,
    )
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)

    # Pre-compute mean/std stacks for all batches (they don't change across iterations)
    batch_mean_stacks = {}
    batch_std_stacks = {}
    for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
        batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
        batch_particle_stack = batch_refine_manager.particle_stack
        
        h, w = batch_particle_stack.original_template_size
        box_h, box_w = batch_particle_stack.extracted_box_size
        extracted_box_size = (box_h - h + 1, box_w - w + 1)
        
        batch_mean_stacks[batch_config_path] = batch_particle_stack.construct_image_stack(
            images=mean_image,
            indices=batch_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=0.0,
        )
        batch_std_stacks[batch_config_path] = batch_particle_stack.construct_image_stack(
            images=var_image,
            indices=batch_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=1e10,
        )
    
    validation_loss_history = []
    training_loss_history = []
    sigma_history = []
    
    # Best-point tracking
    best_validation_loss = None
    best_sigma_iter = None
    best_sigma_params = None
    
    def get_val(key):
        """Get parameter value, handling both dict and tensor cases."""
        v = sigma_params.get(key)
        return abs(v) if isinstance(v, (int, float)) else v
    
    def compute_validation_loss(deformation_field_to_use):
        """Compute validation loss with current sigma parameters."""
        val_loss = 0.0
        with torch.no_grad():
            for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
                batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
                batch_particle_stack = batch_refine_manager.particle_stack
                batch_size = len(batch_indices[0])
                
                image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                    movie=image, deformation_field=deformation_field_to_use,
                    pos_reference="top-left", handle_bounds="pad",
                    padding_mode="reflect", padding_value=0.0,
                    pre_exposure=pre_exposure, fluence_per_frame=fluence_per_frame
                )
                
                # Reuse pre-computed mean/std stacks (same as motion loop)
                batch_mean_stack = batch_mean_stacks[batch_config_path]
                batch_std_stack = batch_std_stacks[batch_config_path]
                
                backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                    image_stack=image_stack_batch,
                    mean_stack=batch_mean_stack,
                    std_stack=batch_std_stack,
                    particle_indices=batch_indices,
                    template_tensor=validation_template,
                    images_are_particles=True,
                )
                result = batch_refine_manager.get_refine_result(backend_kwargs, correlation_batch_size)
                
                val_loss_tensor = result["refined_z_score"] if loss_metric == "scaled_mip" else result["refined_cross_correlation"]
                val_loss += -torch.mean(val_loss_tensor).item() * batch_size / total_n_particles
                
                del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                torch.cuda.empty_cache()
        return val_loss
    
    def run_inner_optimization():
        """Run inner motion optimization loop with current sigma parameters."""
        # Load template volume once (same for all batches)
        template_volume = load_template_volume_from_config(refine_config_path)
        
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
                sigma_A_tensor = _create_exponential_sigma_A(
                    fluence_per_frame * image.shape[0], deformation_field_resolution[0],
                    A=A, B=B, C=C, device=device
                )
            else:
                sigma_A_tensor = get_val("sigma_A")
            alpha = get_val("alpha_spatial")
        elif prior_type == "relion":
            image_coords = _build_physical_coords(
                deformation_field_resolution[1], deformation_field_resolution[2],
                image.shape[-2:], pixel_spacing, device
            )
            sigma_D_val = get_val("sigma_D")
            sigma_V_val = get_val("sigma_V")
            sigma_V_norm = _normalize_sigma_fluence(
                sigma_V_val, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
            )
            if sigma_A_exponential:
                A = get_val("sigma_A_amplitude")
                B = get_val("sigma_A_decay")
                C = get_val("sigma_A_offset")
                sigma_A_tensor = _create_exponential_sigma_A(
                    fluence_per_frame * image.shape[0], deformation_field_resolution[0],
                    A=A, B=B, C=C, device=device
                )
                sigma_A_norm = _normalize_sigma_fluence(
                    sigma_A_tensor, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
                )
            else:
                sa = get_val("sigma_A")
                sigma_A_norm = _normalize_sigma_fluence(
                    sa, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
                )
        
        # Inner loop: motion optimization
        accumulated_loss = 0.0
        for _ in range(motion_iterations):
            motion_optimizer.zero_grad()
            batch_accumulated_loss = 0.0
            
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
                
                # Reuse pre-computed mean/std stacks
                batch_mean_stack = batch_mean_stacks[batch_config_path]
                batch_std_stack = batch_std_stacks[batch_config_path]
                
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
                batch_accumulated_loss += batch_loss.item()
                batch_loss.backward()
                
                del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                torch.cuda.empty_cache()
            
            motion_optimizer.step()
            accumulated_loss += batch_accumulated_loss

        # Detach and clone the final deformation field to break computation graph
        final_deformation_data = deformation_field.data.clone().detach()
        
        # Clean up optimizer
        del motion_optimizer
        torch.cuda.empty_cache()
        
        # Return a new deformation field without gradients
        deformation_field_clean = CubicCatmullRomGrid3d.from_grid_data(
            final_deformation_data
        ).to(device)
        
        # Clean up original
        del deformation_field
        torch.cuda.empty_cache()

        return deformation_field_clean, accumulated_loss
    
    # Objective function for scipy.optimize.minimize
    def objective_function(x):
        """Objective function for Nelder-Mead optimization.
        
        Args:
            x: numpy array of parameter values (in order of param_names)
        
        Returns:
            validation_loss: float
        """
        # Set sigma parameters from x
        for i, param_name in enumerate(param_names):
            sigma_params[param_name] = float(x[i])
        
        # Run inner optimization with current sigmas
        deformation_field, accumulated_loss = run_inner_optimization()
        
        # Compute validation loss
        validation_loss = compute_validation_loss(deformation_field)
        
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
            print(f"Iteration {len(validation_loss_history)}: Parameters: {dict(zip(param_names, x))}")
            print(f"  Validation loss: {validation_loss:.6f}")

        del deformation_field
        torch.cuda.empty_cache()        
        
        return validation_loss
    
    try:
        # Convert initial values to numpy array
        x0 = np.array(initial_values, dtype=np.float64)
        
        if verbose:
            print("=" * 70)
            print("NELDER-MEAD SIGMA OPTIMIZATION")
            print(f"Parameters to optimize: {param_names}")
            print(f"Initial values: {dict(zip(param_names, initial_values))}")
            print(f"Maximum iterations: {sigma_iterations}")
            print("=" * 70)
        
        # Run Nelder-Mead optimization
        result = minimize(
            objective_function,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': sigma_iterations,
                'xatol': 1e-6,  # Absolute tolerance for convergence
                'fatol': 1e-6,  # Function value tolerance
                'disp': verbose,
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
        deformation_field, _ = run_inner_optimization()
        final_deformation_field = deformation_field.data
        final_deformation_field = final_deformation_field - torch.mean(
            final_deformation_field, dim=(1, 2, 3), keepdim=True
        )
        
        optimized_sigmas = sigma_params.copy()
        
        # Use best point if it's better than final point
        final_validation_loss = validation_loss_history[-1]
        if best_validation_loss is not None and final_validation_loss > best_validation_loss:
            if verbose:
                print(f"\nFinal validation loss ({final_validation_loss:.6f}) is worse than best ({best_validation_loss:.6f})")
                print(f"Restoring best parameters from iteration {best_sigma_iter}")
            # Restore best parameters
            for name, best_val in best_sigma_params.items():
                sigma_params[name] = best_val
            # Re-run inner optimization with best parameters
            deformation_field, _ = run_inner_optimization()
            final_deformation_field = deformation_field.data
            final_deformation_field = final_deformation_field - torch.mean(
                final_deformation_field, dim=(1, 2, 3), keepdim=True
            )
            optimized_sigmas = sigma_params.copy()
        
        # Print summary
        if verbose and best_validation_loss is not None:
            print("\n" + "=" * 70)
            print("OPTIMIZATION SUMMARY")
            print("=" * 70)
            print(f"Best validation loss: {best_validation_loss:.6f} (iteration {best_sigma_iter})")
            print(f"Final validation loss: {validation_loss_history[-1]:.6f}")
            print(f"Best parameters:")
            for name, val in best_sigma_params.items():
                print(f"  {name}: {val:.6f}")
            print("=" * 70)
        
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def optimize_sigmas_2dtm_optuna(
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
    n_trials: int = 50,
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
    init_sigma_A: float = 0.513517,
    init_alpha_spatial: float = 1e5,
    init_sigma_A_amplitude: float = 2.0,
    init_sigma_A_decay: float = 0.1,
    init_sigma_A_offset: float = 1.0,
    sigma_A_exponential: bool = False,
    init_sigma_D: float = 5782.376953,
    init_sigma_V: float = 0.194826,
    optimize_sigma_A: bool = True,
    optimize_alpha_spatial: bool = True,
    optimize_sigma_A_amplitude: bool = True,
    optimize_sigma_A_decay: bool = True,
    optimize_sigma_A_offset: bool = True,
    optimize_sigma_D: bool = True,
    optimize_sigma_V: bool = True,
    # Optuna-specific parameters
    study_name: str | None = None,
    sampler: optuna.samplers.BaseSampler | None = None,  # Default: TPE
    pruner: optuna.pruners.BasePruner | None = None,  # Default: MedianPruner
    direction: str = "minimize",
    param_range_low: float = 0.25,  # Lower bound multiplier for parameter search range (default: 0.25x initial value)
    param_range_high: float = 4.0,  # Upper bound multiplier for parameter search range (default: 4.0x initial value)
    verbose: bool = True,
    # Output paths for saving results
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
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
    validation_template_path : str
        Path to validation template (.mrc) for computing validation loss
    motion_iterations : int
        Motion optimization iterations per sigma update. Default 10
    n_trials : int
        Number of Optuna trials to run. Default 50
    optimizer_kwargs : dict
        Kwargs for motion optimizer. Default {"lr": 0.2}
    sigma_optimizer_kwargs : dict
        Not used for Optuna, kept for API compatibility
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
    init_sigma_A : float
        Initial sigma_A (constant mode). Default 0.88
    init_alpha_spatial : float
        Initial alpha_spatial. Default 1e5
    init_sigma_A_amplitude : float
        Initial A in exponential. Default 2.0
    init_sigma_A_decay : float
        Initial B in exponential. Default 0.1
    init_sigma_A_offset : float
        Initial C in exponential. Default 1.0
    sigma_A_exponential : bool
        Use exponential sigma_A. Default False
    init_sigma_D : float
        Initial sigma_D. Default 5080.0
    init_sigma_V : float
        Initial sigma_V. Default 0.58
    optimize_sigma_A : bool
        Whether to optimize sigma_A. Default True
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial. Default True
    optimize_sigma_A_amplitude : bool
        Whether to optimize sigma_A_amplitude. Default True
    optimize_sigma_A_decay : bool
        Whether to optimize sigma_A_decay. Default True
    optimize_sigma_A_offset : bool
        Whether to optimize sigma_A_offset. Default True
    optimize_sigma_D : bool
        Whether to optimize sigma_D. Default True
    optimize_sigma_V : bool
        Whether to optimize sigma_V. Default True
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
        searched in range [initial_value * param_range_low, initial_value * param_range_high].
        Default 0.25 (i.e., 0.25x to 4x initial values).
    param_range_high : float
        Upper bound multiplier for parameter search range. Parameters will be
        searched in range [initial_value * param_range_low, initial_value * param_range_high].
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
        
    Returns
    -------
    dict
        - "optimized_sigmas": dict of optimized sigma values
        - "final_deformation_field": final deformation field
        - "validation_loss_history": list of validation losses
        - "training_loss_history": list of training losses
        - "sigma_history": list of sigma values at each trial
        - "best_validation_loss": best validation loss found
        - "best_trial": best trial number
        - "best_sigma_params": best sigma parameters found
        - "optuna_study": Optuna study object (for further analysis)
    """
    import mrcfile
    print("Optimizing sigmas using Optuna...")
    
    torch.set_grad_enabled(True)
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_sigma_opt_optuna_"))
    
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
    
    if var_image.requires_grad:
        var_image = var_image.clone().detach().requires_grad_(False)
    if mean_image.requires_grad:
        mean_image = mean_image.clone().detach().requires_grad_(False)
    if image.requires_grad:
        image = image.clone().detach().requires_grad_(False)
    
    # Build parameter list and initial values
    param_names = []
    initial_values = {}
    sigma_params = {}  # Store all parameters (both optimized and fixed)
    
    # Collect parameters to optimize
    if sigma_A_exponential:
        if optimize_sigma_A_amplitude:
            param_names.append("sigma_A_amplitude")
            initial_values["sigma_A_amplitude"] = init_sigma_A_amplitude
            sigma_params["sigma_A_amplitude"] = init_sigma_A_amplitude
        else:
            sigma_params["sigma_A_amplitude"] = init_sigma_A_amplitude
            
        if optimize_sigma_A_decay:
            param_names.append("sigma_A_decay")
            initial_values["sigma_A_decay"] = init_sigma_A_decay
            sigma_params["sigma_A_decay"] = init_sigma_A_decay
        else:
            sigma_params["sigma_A_decay"] = init_sigma_A_decay
            
        if optimize_sigma_A_offset:
            param_names.append("sigma_A_offset")
            initial_values["sigma_A_offset"] = init_sigma_A_offset
            sigma_params["sigma_A_offset"] = init_sigma_A_offset
        else:
            sigma_params["sigma_A_offset"] = init_sigma_A_offset
    else:
        if optimize_sigma_A:
            param_names.append("sigma_A")
            initial_values["sigma_A"] = init_sigma_A
            sigma_params["sigma_A"] = init_sigma_A
        else:
            sigma_params["sigma_A"] = init_sigma_A
    
    if prior_type == "laplacian" and optimize_alpha_spatial:
        param_names.append("alpha_spatial")
        initial_values["alpha_spatial"] = init_alpha_spatial
        sigma_params["alpha_spatial"] = init_alpha_spatial
    else:
        sigma_params["alpha_spatial"] = init_alpha_spatial
    
    if prior_type == "relion":
        if optimize_sigma_D:
            param_names.append("sigma_D")
            initial_values["sigma_D"] = init_sigma_D
            sigma_params["sigma_D"] = init_sigma_D
        else:
            sigma_params["sigma_D"] = init_sigma_D
            
        if optimize_sigma_V:
            param_names.append("sigma_V")
            initial_values["sigma_V"] = init_sigma_V
            sigma_params["sigma_V"] = init_sigma_V
        else:
            sigma_params["sigma_V"] = init_sigma_V
    
    if len(param_names) == 0:
        raise ValueError("No sigma parameters selected for optimization!")
    
    batch_config_paths, batch_particle_indices = _create_batch_configs(
        refine_config_path=refine_config_path,
        particle_batch_size=particle_batch_size,
        temp_dir=temp_dir,
    )
    total_n_particles = sum(len(indices[0]) for indices in batch_particle_indices)

    # Pre-compute mean/std stacks for all batches (they don't change across iterations)
    batch_mean_stacks = {}
    batch_std_stacks = {}
    for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
        batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
        batch_particle_stack = batch_refine_manager.particle_stack
        
        h, w = batch_particle_stack.original_template_size
        box_h, box_w = batch_particle_stack.extracted_box_size
        extracted_box_size = (box_h - h + 1, box_w - w + 1)
        
        batch_mean_stacks[batch_config_path] = batch_particle_stack.construct_image_stack(
            images=mean_image,
            indices=batch_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=0.0,
        )
        batch_std_stacks[batch_config_path] = batch_particle_stack.construct_image_stack(
            images=var_image,
            indices=batch_indices,
            extraction_size=extracted_box_size,
            pos_reference="top-left",
            handle_bounds="pad",
            padding_mode="constant",
            padding_value=1e10,
        )
    
    validation_loss_history = []
    training_loss_history = []
    sigma_history = []
    
    def get_val(key):
        """Get parameter value, handling both dict and tensor cases."""
        v = sigma_params.get(key)
        return abs(v) if isinstance(v, (int, float)) else v
    
    def compute_validation_loss(deformation_field_to_use):
        """Compute validation loss with current sigma parameters."""
        val_loss = 0.0
        with torch.no_grad():
            for batch_config_path, batch_indices in zip(batch_config_paths, batch_particle_indices, strict=True):
                batch_refine_manager = _make_differentiable_refine_manager(batch_config_path)
                batch_particle_stack = batch_refine_manager.particle_stack
                batch_size = len(batch_indices[0])
                
                image_stack_batch = batch_particle_stack.construct_image_stack_from_movie(
                    movie=image, deformation_field=deformation_field_to_use,
                    pos_reference="top-left", handle_bounds="pad",
                    padding_mode="reflect", padding_value=0.0,
                    pre_exposure=pre_exposure, fluence_per_frame=fluence_per_frame
                )
                
                # Reuse pre-computed mean/std stacks (same as motion loop)
                batch_mean_stack = batch_mean_stacks[batch_config_path]
                batch_std_stack = batch_std_stacks[batch_config_path]
                
                backend_kwargs = batch_refine_manager.make_backend_core_function_kwargs(
                    image_stack=image_stack_batch,
                    mean_stack=batch_mean_stack,
                    std_stack=batch_std_stack,
                    particle_indices=batch_indices,
                    template_tensor=validation_template,
                    images_are_particles=True,
                )
                result = batch_refine_manager.get_refine_result(backend_kwargs, correlation_batch_size)
                
                val_loss_tensor = result["refined_z_score"] if loss_metric == "scaled_mip" else result["refined_cross_correlation"]
                val_loss += -torch.mean(val_loss_tensor).item() * batch_size / total_n_particles
                
                del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                torch.cuda.empty_cache()
        return val_loss
    
    def run_inner_optimization():
        """Run inner motion optimization loop with current sigma parameters."""
        # Load template volume once (same for all batches)
        template_volume = load_template_volume_from_config(refine_config_path)
        
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
                sigma_A_tensor = _create_exponential_sigma_A(
                    fluence_per_frame * image.shape[0], deformation_field_resolution[0],
                    A=A, B=B, C=C, device=device
                )
            else:
                sigma_A_tensor = get_val("sigma_A")
            alpha = get_val("alpha_spatial")
        elif prior_type == "relion":
            image_coords = _build_physical_coords(
                deformation_field_resolution[1], deformation_field_resolution[2],
                image.shape[-2:], pixel_spacing, device
            )
            sigma_D_val = get_val("sigma_D")
            sigma_V_val = get_val("sigma_V")
            sigma_V_norm = _normalize_sigma_fluence(
                sigma_V_val, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
            )
            if sigma_A_exponential:
                A = get_val("sigma_A_amplitude")
                B = get_val("sigma_A_decay")
                C = get_val("sigma_A_offset")
                sigma_A_tensor = _create_exponential_sigma_A(
                    fluence_per_frame * image.shape[0], deformation_field_resolution[0],
                    A=A, B=B, C=C, device=device
                )
                sigma_A_norm = _normalize_sigma_fluence(
                    sigma_A_tensor, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
                )
            else:
                sa = get_val("sigma_A")
                sigma_A_norm = _normalize_sigma_fluence(
                    sa, fluence_per_frame * image.shape[0], deformation_field_resolution[0]
                )
        
        # Inner loop: motion optimization
        accumulated_loss = 0.0
        for _ in range(motion_iterations):
            motion_optimizer.zero_grad()
            batch_accumulated_loss = 0.0
            
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
                
                # Reuse pre-computed mean/std stacks
                batch_mean_stack = batch_mean_stacks[batch_config_path]
                batch_std_stack = batch_std_stacks[batch_config_path]
                
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
                batch_accumulated_loss += batch_loss.item()
                batch_loss.backward()
                
                del image_stack_batch, batch_mean_stack, batch_std_stack, backend_kwargs, result
                torch.cuda.empty_cache()
            
            motion_optimizer.step()
            accumulated_loss += batch_accumulated_loss
        
        return deformation_field, accumulated_loss
    
    # Objective function for Optuna
    def objective(trial):
        """Objective function for Optuna optimization."""
        # Suggest parameter values using log-uniform for wide ranges
        for param_name in param_names:
            initial_val = initial_values[param_name]
            
            # Use log-uniform for parameters that span orders of magnitude
            if param_name == "alpha_spatial" or param_name == "sigma_D":
                # These span large ranges, use log-uniform
                suggested_val = trial.suggest_float(
                    param_name,
                    low=max(initial_val * param_range_low, 1e-6),
                    high=initial_val * param_range_high,
                    log=True
                )
            else:
                # Use uniform for smaller ranges
                suggested_val = trial.suggest_float(
                    param_name,
                    low=max(initial_val * param_range_low, 1e-6),
                    high=initial_val * param_range_high,
                    log=False
                )
            
            sigma_params[param_name] = suggested_val
        
        # Run inner optimization with suggested sigmas
        deformation_field, accumulated_loss = run_inner_optimization()
        
        # Compute validation loss
        validation_loss = compute_validation_loss(deformation_field)
        
        # Track history
        validation_loss_history.append(validation_loss)
        training_loss_history.append(accumulated_loss)
        
        # Record current sigmas (include fixed parameters too)
        current_sigmas = sigma_params.copy()
        sigma_history.append(current_sigmas.copy())
        
        if verbose:
            print(f"Trial {trial.number}: Parameters: {dict(zip(param_names, [sigma_params[n] for n in param_names]))}")
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
            print(f"Best parameters:")
            for name, val in study.best_params.items():
                print(f"  {name}: {val:.6f}")
            print("=" * 70)
        
        # Extract best parameters
        best_params = study.best_params
        for name, val in best_params.items():
            sigma_params[name] = val
        
        # Get final deformation field with best parameters
        deformation_field, _ = run_inner_optimization()
        final_deformation_field = deformation_field.data
        final_deformation_field = final_deformation_field - torch.mean(
            final_deformation_field, dim=(1, 2, 3), keepdim=True
        )
        
        # Create optimized_sigmas from best trial parameters (explicitly use best validation loss)
        optimized_sigmas = {}
        # Include all parameters (both optimized and fixed)
        for key in sigma_params.keys():
            if key in best_params:
                # Use best trial's value for optimized parameters
                optimized_sigmas[key] = best_params[key]
            else:
                # Use fixed parameter value
                optimized_sigmas[key] = (
                    sigma_params[key].item() if isinstance(sigma_params[key], torch.Tensor)
                    else sigma_params[key]
                )
        
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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
def optimize_sigmas_2dtm_multi_start(
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
    init_sigma_A: float = 0.88,
    init_alpha_spatial: float = 1e5,
    init_sigma_A_amplitude: float = 2.0,
    init_sigma_A_decay: float = 0.1,
    init_sigma_A_offset: float = 1.0,
    sigma_A_exponential: bool = False,
    init_sigma_D: float = 5080.0,
    init_sigma_V: float = 0.58,
    optimize_sigma_A: bool = True,
    optimize_alpha_spatial: bool = True,
    optimize_sigma_A_amplitude: bool = True,
    optimize_sigma_A_decay: bool = True,
    optimize_sigma_A_offset: bool = True,
    optimize_sigma_D: bool = True,
    optimize_sigma_V: bool = True,
    # Multi-start parameters
    n_values_per_param: int = 3,  # Number of values to try per parameter (default 3: 0.5x, 1x, 2x)
    exploration_factors: tuple[float, ...] = (0.5, 1.0, 2.0),  # Multipliers for initial values
    verbose: bool = True,
    # Output paths for saving results
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
) -> dict[str, Any]:
    """Optimize prior hyperparameters using multi-start grid search.
    
    This function performs a two-stage optimization:
    1. Quick grid search: Tests multiple initial sigma combinations with 1 sigma_iteration each
    2. Full optimization: Runs full optimization from the best starting point found in stage 1
    
    This approach helps escape local minima by exploring a wide parameter space first.
    
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
        Number of sigma optimization iterations for final optimization. Default 20
    optimizer_kwargs : dict
        Kwargs for motion optimizer. Default {"lr": 0.2}
    sigma_optimizer_kwargs : dict
        Kwargs for sigma optimizer. Default {"lr": 0.01}
    prior_type : str
        "laplacian" or "relion". Default "relion"
    init_sigma_A : float
        Initial sigma_A (constant mode). Default 0.88
    init_alpha_spatial : float
        Initial alpha_spatial. Default 1e5
    init_sigma_A_amplitude : float
        Initial A in exponential. Default 2.0
    init_sigma_A_decay : float
        Initial B in exponential. Default 0.1
    init_sigma_A_offset : float
        Initial C in exponential. Default 1.0
    sigma_A_exponential : bool
        Use exponential sigma_A. Default False
    init_sigma_D : float
        Initial sigma_D. Default 5080.0
    init_sigma_V : float
        Initial sigma_V. Default 0.58
    optimize_sigma_A : bool
        Whether to optimize sigma_A. Default True
    optimize_alpha_spatial : bool
        Whether to optimize alpha_spatial. Default True
    optimize_sigma_A_amplitude : bool
        Whether to optimize sigma_A_amplitude. Default True
    optimize_sigma_A_decay : bool
        Whether to optimize sigma_A_decay. Default True
    optimize_sigma_A_offset : bool
        Whether to optimize sigma_A_offset. Default True
    optimize_sigma_D : bool
        Whether to optimize sigma_D. Default True
    optimize_sigma_V : bool
        Whether to optimize sigma_V. Default True
    n_values_per_param : int
        Number of values to try per parameter. Default 3
    exploration_factors : tuple[float, ...]
        Multipliers for initial values. Default (0.5, 1.0, 2.0)
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
        - "grid_search_results": list of (params, validation_loss) from grid search
    """
    # Determine which parameters to optimize
    params_to_optimize = {}
    
    if sigma_A_exponential:
        if optimize_sigma_A_amplitude:
            params_to_optimize["sigma_A_amplitude"] = init_sigma_A_amplitude
        if optimize_sigma_A_decay:
            params_to_optimize["sigma_A_decay"] = init_sigma_A_decay
        if optimize_sigma_A_offset:
            params_to_optimize["sigma_A_offset"] = init_sigma_A_offset
    else:
        if optimize_sigma_A:
            params_to_optimize["sigma_A"] = init_sigma_A
    
    if prior_type == "laplacian" and optimize_alpha_spatial:
        params_to_optimize["alpha_spatial"] = init_alpha_spatial
    
    if prior_type == "relion":
        if optimize_sigma_D:
            params_to_optimize["sigma_D"] = init_sigma_D
        if optimize_sigma_V:
            params_to_optimize["sigma_V"] = init_sigma_V
    
    if len(params_to_optimize) == 0:
        raise ValueError("No sigma parameters selected for optimization!")
    
    if verbose:
        print("=" * 70)
        print("MULTI-START SIGMA OPTIMIZATION")
        print(f"Parameters to optimize: {list(params_to_optimize.keys())}")
        print(f"Values per parameter: {n_values_per_param}")
        print(f"Exploration factors: {exploration_factors}")
        total_combinations = len(exploration_factors) ** len(params_to_optimize)
        print(f"Total combinations to test: {total_combinations}")
        print("=" * 70)
    
    # Generate all combinations
    param_names = list(params_to_optimize.keys())
    param_values = [
        [init_val * factor for factor in exploration_factors]
        for init_val in params_to_optimize.values()
    ]
    
    all_combinations = list(itertools.product(*param_values))
    
    if verbose:
        print(f"\nStage 1: Quick grid search ({len(all_combinations)} combinations)")
        print("Running 1 sigma_iteration per combination...")
    
    # Stage 1: Quick grid search with 1 sigma_iteration each
    grid_search_results = []
    best_combination = None
    best_validation_loss = float('inf')
    
    temp_dir = Path(tempfile.mkdtemp(prefix="ripple_multi_start_"))
    
    try:
        for combo_idx, combination in enumerate(all_combinations):
            if verbose:
                print(f"\n[{combo_idx + 1}/{len(all_combinations)}] Testing combination:")
                for name, value in zip(param_names, combination, strict=True):
                    print(f"  {name}: {value:.6f}")
            
            # Build kwargs for this combination
            combo_kwargs = {}
            for name, value in zip(param_names, combination, strict=True):
                if name == "sigma_A":
                    combo_kwargs["init_sigma_A"] = value
                elif name == "alpha_spatial":
                    combo_kwargs["init_alpha_spatial"] = value
                elif name == "sigma_A_amplitude":
                    combo_kwargs["init_sigma_A_amplitude"] = value
                elif name == "sigma_A_decay":
                    combo_kwargs["init_sigma_A_decay"] = value
                elif name == "sigma_A_offset":
                    combo_kwargs["init_sigma_A_offset"] = value
                elif name == "sigma_D":
                    combo_kwargs["init_sigma_D"] = value
                elif name == "sigma_V":
                    combo_kwargs["init_sigma_V"] = value
            
            # Run quick evaluation (1 sigma_iteration)
            temp_output_dir = temp_dir / f"combo_{combo_idx}"
            temp_output_dir.mkdir(exist_ok=True)
            
            try:
                result = optimize_sigmas_2dtm_bayesian(
                    image=image,
                    var_image=var_image,
                    mean_image=mean_image,
                    pixel_spacing=pixel_spacing,
                    deformation_field_resolution=deformation_field_resolution,
                    initial_deformation_field=initial_deformation_field,
                    refine_config_path=refine_config_path,
                    validation_template_path=validation_template_path,
                    pre_exposure=pre_exposure,
                    fluence_per_frame=fluence_per_frame,
                    motion_iterations=motion_iterations,
                    sigma_iterations=1,  # Quick evaluation
                    optimizer_kwargs=optimizer_kwargs,
                    sigma_optimizer_kwargs=sigma_optimizer_kwargs,
                    correlation_batch_size=correlation_batch_size,
                    particle_batch_size=particle_batch_size,
                    particle_indices=particle_indices,
                    device=device,
                    loss_metric=loss_metric,
                    min_snr=min_snr,
                    best_n=best_n,
                    prior_type=prior_type,
                    sigma_A_exponential=sigma_A_exponential,
                    optimize_sigma_A=optimize_sigma_A,
                    optimize_alpha_spatial=optimize_alpha_spatial,
                    optimize_sigma_A_amplitude=optimize_sigma_A_amplitude,
                    optimize_sigma_A_decay=optimize_sigma_A_decay,
                    optimize_sigma_A_offset=optimize_sigma_A_offset,
                    optimize_sigma_D=optimize_sigma_D,
                    optimize_sigma_V=optimize_sigma_V,
                    perturbation_interval=0,  # Disable perturbation for quick search
                    verbose=False,  # Reduce verbosity for grid search
                    optimized_sigmas_output_path=None,  # Don't save intermediate results
                    sigma_history_output_path=None,
                    training_history_output_path=None,
                    validation_history_output_path=None,
                    **combo_kwargs,
                )
                
                # Get final validation loss
                final_val_loss = result["validation_loss_history"][-1]
                grid_search_results.append({
                    "params": dict(zip(param_names, combination, strict=True)),
                    "validation_loss": final_val_loss,
                })
                
                if verbose:
                    print(f"  Validation loss: {final_val_loss:.6f}")
                
                if final_val_loss < best_validation_loss:
                    best_validation_loss = final_val_loss
                    best_combination = dict(zip(param_names, combination, strict=True))
                    if verbose:
                        print(f"  ✓ New best!")
                
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error: {e}")
                grid_search_results.append({
                    "params": dict(zip(param_names, combination, strict=True)),
                    "validation_loss": float('inf'),
                    "error": str(e),
                })
        
        if best_combination is None:
            raise RuntimeError("All grid search combinations failed!")
        
        if verbose:
            print("\n" + "=" * 70)
            print("Stage 1 complete: Best starting point found")
            print("Best combination:")
            for name, value in best_combination.items():
                print(f"  {name}: {value:.6f}")
            print(f"Best validation loss: {best_validation_loss:.6f}")
            print("=" * 70)
            print(f"\nStage 2: Full optimization ({sigma_iterations} sigma_iterations)")
            print("Running full optimization from best starting point...")
        
        # Stage 2: Full optimization from best starting point
        best_kwargs = {}
        for name, value in best_combination.items():
            if name == "sigma_A":
                best_kwargs["init_sigma_A"] = value
            elif name == "alpha_spatial":
                best_kwargs["init_alpha_spatial"] = value
            elif name == "sigma_A_amplitude":
                best_kwargs["init_sigma_A_amplitude"] = value
            elif name == "sigma_A_decay":
                best_kwargs["init_sigma_A_decay"] = value
            elif name == "sigma_A_offset":
                best_kwargs["init_sigma_A_offset"] = value
            elif name == "sigma_D":
                best_kwargs["init_sigma_D"] = value
            elif name == "sigma_V":
                best_kwargs["init_sigma_V"] = value
        
        final_result = optimize_sigmas_2dtm_bayesian(
            image=image,
            var_image=var_image,
            mean_image=mean_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=deformation_field_resolution,
            initial_deformation_field=initial_deformation_field,
            refine_config_path=refine_config_path,
            validation_template_path=validation_template_path,
            pre_exposure=pre_exposure,
            fluence_per_frame=fluence_per_frame,
            motion_iterations=motion_iterations,
            sigma_iterations=sigma_iterations,  # Full optimization
            optimizer_kwargs=optimizer_kwargs,
            sigma_optimizer_kwargs=sigma_optimizer_kwargs,
            correlation_batch_size=correlation_batch_size,
            particle_batch_size=particle_batch_size,
            particle_indices=particle_indices,
            device=device,
            loss_metric=loss_metric,
            min_snr=min_snr,
            best_n=best_n,
            prior_type=prior_type,
            sigma_A_exponential=sigma_A_exponential,
            optimize_sigma_A=optimize_sigma_A,
            optimize_alpha_spatial=optimize_alpha_spatial,
            optimize_sigma_A_amplitude=optimize_sigma_A_amplitude,
            optimize_sigma_A_decay=optimize_sigma_A_decay,
            optimize_sigma_A_offset=optimize_sigma_A_offset,
            optimize_sigma_D=optimize_sigma_D,
            optimize_sigma_V=optimize_sigma_V,
            perturbation_interval=0,  # Can enable if desired
            verbose=verbose,
            optimized_sigmas_output_path=optimized_sigmas_output_path,
            sigma_history_output_path=sigma_history_output_path,
            training_history_output_path=training_history_output_path,
            validation_history_output_path=validation_history_output_path,
            **best_kwargs,
        )
        
        # Add grid search results to final output
        final_result["grid_search_results"] = grid_search_results
        final_result["best_starting_point"] = best_combination
        
        if verbose:
            print("\n" + "=" * 70)
            print("MULTI-START OPTIMIZATION COMPLETE")
            print("=" * 70)
        
        return final_result
        
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"Cleaned up temporary directory: {temp_dir}")
