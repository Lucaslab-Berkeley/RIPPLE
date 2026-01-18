"""Utility functions shared across core modules."""

from pathlib import Path

import pandas as pd
import torch
import yaml
from leopard_em.pydantic_models.managers import RefineTemplateManager
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_grid_utils import coordinate_grid


def get_batch_mean_std_stacks(
    batch_config_paths: list[str],
    batch_particle_indices: list[list[pd.Index]],
    mean_image: torch.Tensor,
    var_image: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Pre-compute mean and std stacks for all batches.

    These stacks don't change across iterations, so they can be computed once
    and reused.

    Parameters
    ----------
    batch_config_paths : list[str]
        List of paths to batch config files.
    batch_particle_indices : list[list[pd.Index]]
        List of particle indices for each batch.
    mean_image : torch.Tensor
        Mean image tensor (t, H, W).
    var_image : torch.Tensor
        Variance image tensor (t, H, W).

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        Tuple of (batch_mean_stacks, batch_std_stacks) dictionaries,
        keyed by batch_config_path.
    """
    batch_mean_stacks: dict[str, torch.Tensor] = {}
    batch_std_stacks: dict[str, torch.Tensor] = {}

    for batch_config_path, batch_indices in zip(
        batch_config_paths, batch_particle_indices, strict=True
    ):
        batch_refine_manager = _make_differentiable_refine_manager(
            refine_config_path=batch_config_path,
        )
        batch_particle_stack = batch_refine_manager.particle_stack

        h, w = batch_particle_stack.original_template_size
        box_h, box_w = batch_particle_stack.extracted_box_size
        extracted_box_size = (box_h - h + 1, box_w - w + 1)

        batch_mean_stacks[batch_config_path] = (
            batch_particle_stack.construct_image_stack(
                images=mean_image,
                indices=batch_indices,
                extraction_size=extracted_box_size,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="constant",
                padding_value=0.0,
            )
        )

        batch_std_stacks[batch_config_path] = (
            batch_particle_stack.construct_image_stack(
                images=var_image,
                indices=batch_indices,
                extraction_size=extracted_box_size,
                pos_reference="top-left",
                handle_bounds="pad",
                padding_mode="constant",
                padding_value=1e10,
            )
        )

    return batch_mean_stacks, batch_std_stacks


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

    # Resolve relative paths relative to current working directory
    if not Path(csv_path).is_absolute():
        csv_path = str(Path(csv_path).resolve())

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

    # Resolve template_volume_path to absolute (relative to current working directory)
    if "template_volume_path" in filtered_config:
        template_path = Path(config["template_volume_path"])
        if not template_path.is_absolute():
            template_path = template_path.resolve()
        filtered_config["template_volume_path"] = str(template_path)

    filtered_config_path = temp_dir / "filtered_config.yaml"
    with open(filtered_config_path, "w", encoding="utf-8") as f:
        yaml.dump(filtered_config, f)

    # Return indices starting from 0 to match the new CSV
    filtered_indices = pd.Index(range(len(df_filtered)))
    return str(filtered_config_path), [filtered_indices]


# pylint: disable=too-many-locals
def _create_batch_configs(
    refine_config_path: str,
    particle_batch_size: int,
    temp_dir: Path,
    prefix: str = "batch",
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
    prefix : str
        Prefix for batch file names to avoid collisions. Default "batch".

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

    def resolve_path(path_str: str | None) -> str | None:
        """Resolve a path relative to the current working directory."""
        if path_str is None:
            return None
        path = Path(path_str)
        if not path.is_absolute():
            path = path.resolve()
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
        batch_csv_path = temp_dir / f"{prefix}_{batch_idx}_particles.csv"
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
        batch_config_path = temp_dir / f"{prefix}_{batch_idx}_config.yaml"
        with open(batch_config_path, "w", encoding="utf-8") as f:
            yaml.dump(batch_config, f)

        batch_config_paths.append(str(batch_config_path))

    return batch_config_paths, batch_particle_indices


def _make_differentiable_refine_manager(
    refine_config_path: str,
) -> RefineTemplateManager:
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
    refine_manager = RefineTemplateManager.from_yaml(refine_config_path)
    # override the movie_params here
    refine_manager.movie_config.enabled = False
    return refine_manager


def _get_particle_coordinates(
    particle_stack,
    particle_indices: list[pd.Index],
    pixel_spacing: float,
    device: torch.device,
) -> torch.Tensor:
    """Get physical coordinates of particles in Angstroms.

    Parameters
    ----------
    particle_stack
        Particle stack from refine manager
    particle_indices: list[pd.Index]
        List of particle indices (one per batch)
    pixel_spacing: float
        Pixel spacing in Angstroms
    device: torch.device
        Device for tensors

    Returns
    -------
    torch.Tensor
        (N, 2) coordinates in Angstroms where N is total number of particles
    """
    # Get position columns
    y_col, x_col = particle_stack._get_position_reference_columns()

    # Get all particle indices across batches
    all_indices = []
    for batch_indices in particle_indices:
        # batch_indices is a list with one pd.Index
        batch_df_indices = particle_stack._df.index[batch_indices[0]].tolist()
        all_indices.extend(batch_df_indices)

    # Get positions in pixels
    pos_y_px = particle_stack._df.loc[all_indices, y_col].to_numpy()
    pos_x_px = particle_stack._df.loc[all_indices, x_col].to_numpy()

    # Convert to Angstroms
    # Use float32 to match particle_shifts dtype
    pos_y_angstrom = torch.tensor(
        pos_y_px * pixel_spacing, device=device, dtype=torch.float32
    )
    pos_x_angstrom = torch.tensor(
        pos_x_px * pixel_spacing, device=device, dtype=torch.float32
    )

    # Stack into (N, 2) coordinates
    coords = torch.stack([pos_y_angstrom, pos_x_angstrom], dim=-1)
    return coords


def compute_particle_shifts_from_deformation_field(
    deformation_field: torch.Tensor,  # (2, nt, nh, nw)
    movie: torch.Tensor,  # (t, H, W)
    refine_config_path: str,
    pixel_spacing: float,
    grid_type: str = "catmull_rom",
    device: torch.device | None = None,
    particle_indices: list[pd.Index] | None = None,
) -> torch.Tensor:
    """
    Compute particle shifts from deformation field as a tensor.

    Parameters
    ----------
    deformation_field : torch.Tensor
        Deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts.
    movie : torch.Tensor
        Movie tensor with shape (t, H, W) where t is number of frames.
    refine_config_path : str
        Path to the refine config YAML file.
    pixel_spacing : float
        Pixel spacing in Angstroms.
    grid_type : str
        Grid type to use for the deformation field. Must be 'catmull_rom' or 'bspline'.
        Default is 'catmull_rom'.
    device : torch.device | None
        Device to perform computation on. If None, uses the device of the input tensors.
    particle_indices : list[pd.Index] | None
        List of particle indices (one per batch). If None, uses all particles.

    Returns
    -------
    torch.Tensor
        Particle shifts tensor with shape (T, N, 2) where T is number of frames
        and N is number of particles. The last dimension is (y_shift, x_shift).
    """
    if device is None:
        device = deformation_field.device

    # Convert deformation field tensor to grid
    if grid_type == "catmull_rom":
        deformation_field_grid = CubicCatmullRomGrid3d.from_grid_data(
            deformation_field
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported grid_type: {grid_type}. "
            "Only 'catmull_rom' is currently supported."
        )

    # Get particle stack from refine manager
    refine_manager = _make_differentiable_refine_manager(refine_config_path)
    particle_stack = refine_manager.particle_stack

    # Get particle positions and other parameters
    pixel_sizes = particle_stack.get_pixel_size()
    y_col, x_col = particle_stack._get_position_reference_columns()
    h, w = particle_stack.original_template_size

    # Get particle indices
    if particle_indices is None:
        # Use all particles
        all_indices = particle_stack._df.index.tolist()
    else:
        # Get all particle indices across batches
        all_indices = []
        for batch_indices in particle_indices:
            batch_df_indices = particle_stack._df.index[batch_indices[0]].tolist()
            all_indices.extend(batch_df_indices)

    n_particles = len(all_indices)

    # Get positions
    pos_y = particle_stack._df.loc[all_indices, y_col].to_numpy()
    pos_x = particle_stack._df.loc[all_indices, x_col].to_numpy()

    # Convert to center positions
    # (as used in compute_frame_particle_shifts_from_deformation)
    pos_y_center = pos_y + h // 2
    pos_x_center = pos_x + w // 2
    pos_y_center = torch.tensor(pos_y_center, device=device)
    pos_x_center = torch.tensor(pos_x_center, device=device)

    # Get deformation field grid dimensions
    _, _, gh, gw = deformation_field.shape

    # Create pixel grid
    t, img_h, img_w = movie.shape
    pixel_grid = coordinate_grid(
        image_shape=(img_h, img_w),
        device=device,
    )

    # Normalized time values for each frame
    normalized_t = torch.linspace(0, 1, steps=t, device=device)

    # Get pixel spacing (handle both scalar and array cases)
    if hasattr(pixel_sizes, "__len__") and len(pixel_sizes) > 0:
        pixel_spacing_to_use = pixel_sizes[0].item()
    else:
        pixel_spacing_to_use = pixel_spacing

    # Initialize output tensor
    particle_shifts = torch.zeros((t, n_particles, 2), device=device, dtype=torch.float32)

    # Process each frame
    for frame_idx, movie_frame in enumerate(movie):
        normalized_t_value = normalized_t[frame_idx]

        # Compute shifts for all particles in this frame
        shifts = particle_stack.compute_frame_particle_shifts_from_deformation(
            movie_frame=movie_frame.to(device),
            deformation_field=deformation_field_grid,
            normalized_t_value=normalized_t_value,
            pixel_grid=pixel_grid,
            pixel_spacing=pixel_spacing_to_use,
            pos_y_center=pos_y_center,
            pos_x_center=pos_x_center,
            gh=gh,
            gw=gw,
        )

        # Store shifts (shifts is (N, 2) as (dy, dx))
        particle_shifts[frame_idx, :, :] = shifts

    return particle_shifts


def compute_particle_shifts(
    deformation_field: torch.Tensor,  # (2, nt, nh, nw)
    movie: torch.Tensor,  # (t, H, W)
    refine_config_path: str,
    pixel_spacing: float,
    grid_type: str = "catmull_rom",
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    Compute particle shifts from deformation field.

    Parameters
    ----------
    deformation_field : torch.Tensor
        Final deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts.
    movie : torch.Tensor
        Movie tensor with shape (t, H, W) where t is number of frames.
    refine_config_path : str
        Path to the refine config YAML file.
    pixel_spacing : float
        Pixel spacing in Angstroms.
    grid_type : str
        Grid type to use for the deformation field. Must be 'catmull_rom' or 'bspline'.
        Default is 'catmull_rom'.
    device : torch.device | None
        Device to perform computation on. If None, uses the device of the input tensors.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: particle_index, frame, y_shift, x_shift.
    """
    # Get particle shifts as tensor using the core function
    particle_shifts_tensor = compute_particle_shifts_from_deformation_field(
        deformation_field=deformation_field,
        movie=movie,
        refine_config_path=refine_config_path,
        pixel_spacing=pixel_spacing,
        grid_type=grid_type,
        device=device,
        particle_indices=None,  # Use all particles
    )

    # Get particle stack to get actual particle indices
    refine_manager = _make_differentiable_refine_manager(refine_config_path)
    particle_stack = refine_manager.particle_stack
    particle_indices = particle_stack._df.index.tolist()

    # Convert tensor to DataFrame
    particle_shifts_np = particle_shifts_tensor.cpu().numpy()  # (T, N, 2)
    t, n_particles, _ = particle_shifts_np.shape

    rows = []
    for frame_idx in range(t):
        for particle_idx in range(n_particles):
            y_shift = float(particle_shifts_np[frame_idx, particle_idx, 0])
            x_shift = float(particle_shifts_np[frame_idx, particle_idx, 1])
            # Use the actual particle index from the dataframe
            particle_index = particle_indices[particle_idx]
            rows.append({
                "particle_index": particle_index,
                "frame": frame_idx,
                "y_shift": y_shift,
                "x_shift": x_shift,
            })

    # Create DataFrame and return
    df = pd.DataFrame(rows)
    return df
