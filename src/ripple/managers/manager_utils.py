"""Utility functions for managers."""

from typing import TYPE_CHECKING, Optional

import pandas as pd
import torch

from ripple.core import generate_dose_weighted_image, sum_movie
from ripple.core.core_utils import compute_particle_shifts
from ripple.utils.data_io import (
    save_deformation_field,
    write_mrc_from_tensor,
    write_trajectory_to_csv,
)

if TYPE_CHECKING:
    from torch_motion_correction import OptimizationTracker

    from ripple.config import (
        BaseAlignmentConfig,
        ComputationalConfig,
        MovieConfig,
        OutputConfig,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def load_missing_tensors(
    computational_config: "ComputationalConfig",
    movie_config: "MovieConfig",
    alignment_config: "BaseAlignmentConfig",
    movie: torch.Tensor | None = None,
    gain_map: torch.Tensor | None = None,
    dark_map: torch.Tensor | None = None,
    deformation_field: torch.Tensor | None = None,
    skip_deformation_field: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Load only the tensors that are not provided as arguments.

    Parameters
    ----------
    computational_config: ComputationalConfig
        Computational configuration containing device information.
    movie_config: MovieConfig
        Movie configuration containing movie, gain, and dark map.
    alignment_config: BaseAlignmentConfig
        Alignment configuration containing deformation field.
    movie: Optional[torch.Tensor]
        Movie tensor. If None, will be loaded from config.
    gain_map: Optional[torch.Tensor]
        Gain map tensor. If None, will be loaded from config.
    dark_map: Optional[torch.Tensor]
        Dark map tensor. If None, will be loaded from config.
    deformation_field: Optional[torch.Tensor]
        Deformation field tensor. If None, will be loaded from config.
    skip_deformation_field: bool
        If True, skip loading deformation_field and return None for it.
        Default is False.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
        Tuple of (movie, gain_map, dark_map, deformation_field), with missing
        ones loaded from config. deformation_field will be None if
        skip_deformation_field is True.
    """
    device = computational_config.gpu_id

    if movie is None:
        movie = movie_config.movie
        # if still not none, move to device
        if movie is not None:
            movie = movie.to(device)
    else:
        movie = movie.to(device)

    if gain_map is None:
        gain_map = movie_config.gain
        if gain_map is not None:
            gain_map = gain_map.to(device)
    else:
        gain_map = gain_map.to(device)

    if dark_map is None:
        dark_map = movie_config.dark
        if dark_map is not None:
            dark_map = dark_map.to(device)
    else:
        dark_map = dark_map.to(device)

    if skip_deformation_field:
        deformation_field = None
    elif deformation_field is None:
        deformation_field = alignment_config.deformation_field
        if deformation_field is not None:
            deformation_field = deformation_field.to(device)
    else:
        deformation_field = deformation_field.to(device)

    return movie, gain_map, dark_map, deformation_field


# pylint: disable=too-many-arguments,too-many-positional-arguments
def save_results(
    output_config: "OutputConfig",
    movie_config: "MovieConfig",
    corrected_movie: torch.Tensor,
    updated_deformation_field: torch.Tensor | dict[str, torch.Tensor],
    movie_prepared: torch.Tensor,
    trajectory: Optional["OptimizationTracker"] = None,
    refine_config_path: str | None = None,
    grid_type: str = "catmull_rom",
    device: torch.device | None = None,
) -> None:
    """
    Save the results of the alignment.

    Parameters
    ----------
    output_config: OutputConfig
        Output configuration containing output paths.
    movie_config: MovieConfig
        Movie configuration containing pixel size, pre-exposure, fluence, and voltage.
    corrected_movie: torch.Tensor
        The corrected movie.
    updated_deformation_field: torch.Tensor | dict[str, torch.Tensor]
        The updated deformation field (tensor) or particle_shifts (dict with
        "particle_shifts" key containing (T, N, 2) tensor).
    movie_prepared: torch.Tensor
        The prepared movie.
    trajectory: Optional[OptimizationTracker]
        The trajectory of the alignment.
    refine_config_path: str | None
        Path to refine config YAML file. Required for particle shift computation
        when using deformation_field mode.
    grid_type: str
        Grid type used for deformation field ('catmull_rom' or 'bspline').
        Default is 'catmull_rom'.
    device: torch.device | None
        Device to use for computations. If None, uses the device of input tensors.

    Returns
    -------
    None
        None.
    """
    # Check if we have particle_shifts or deformation_field
    is_particle_shifts = isinstance(updated_deformation_field, dict)
    if is_particle_shifts:
        particle_shifts = updated_deformation_field.get("particle_shifts")
        if particle_shifts is None:
            raise ValueError(
                "updated_deformation_field is a dict but does not contain "
                "'particle_shifts' key"
            )
        deformation_field = None
    else:
        deformation_field = updated_deformation_field
        particle_shifts = None
    # Save DW sum is wanted
    if output_config.dw_sum_output_path is not None:
        dw_movie = generate_dose_weighted_image(
            corrected_movie,
            movie_config.pixel_size,
            movie_config.pre_exposure,
            movie_config.fluence_per_frame,
            movie_config.voltage,
        )
        dw_movie = dw_movie.cpu()
        write_mrc_from_tensor(
            data=dw_movie,
            mrc_path=output_config.dw_sum_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save deformation field if wanted (only if we have a deformation field)
    if output_config.deformation_field_output_path is not None:
        if deformation_field is None:
            raise ValueError(
                "deformation_field_output_path is specified but optimization "
                "result contains particle_shifts instead of deformation_field"
            )
        deformation_field_cpu = deformation_field.cpu()
        save_deformation_field(
            deformation_field_cpu,
            output_config.deformation_field_output_path,
        )

    # Save non-dw sum movie if wanted
    if output_config.non_dw_sum_output_path is not None:
        summed_movie = sum_movie(corrected_movie)
        summed_movie = summed_movie.cpu()
        write_mrc_from_tensor(
            data=summed_movie,
            mrc_path=output_config.non_dw_sum_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save aligned movie if wanted
    if output_config.motion_corrected_movie_output_path is not None:
        corrected_movie = corrected_movie.cpu()
        write_mrc_from_tensor(
            data=corrected_movie,
            mrc_path=output_config.motion_corrected_movie_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save rendered, unaligned movie if wanted
    if output_config.rendered_movie_output_path is not None:
        rendered_movie = movie_prepared.cpu()
        write_mrc_from_tensor(
            data=rendered_movie,
            mrc_path=output_config.rendered_movie_output_path,
            overwrite=output_config.allow_file_overwrite,
        )

    # Save loss trajectories if wanted
    if trajectory is not None:
        if output_config.loss_trajectories_output_path is not None:
            write_trajectory_to_csv(
                trajectory=trajectory,
                file_path=output_config.loss_trajectories_output_path,
            )

    # Save particle shifts if wanted (only for polishing with particles)
    if output_config.particle_shift_path is not None:
        if is_particle_shifts:
            # We already have particle_shifts, convert directly to DataFrame
            # particle_shifts is (T, N, 2) where T is frames, N is particles
            particle_shifts_np = particle_shifts.cpu().numpy()
            T, N, _ = particle_shifts_np.shape

            # Create DataFrame with columns: particle_index, frame, y_shift, x_shift
            rows = []
            for frame_idx in range(T):
                for particle_idx in range(N):
                    y_shift = float(particle_shifts_np[frame_idx, particle_idx, 0])
                    x_shift = float(particle_shifts_np[frame_idx, particle_idx, 1])
                    rows.append({
                        "particle_index": particle_idx,
                        "frame": frame_idx,
                        "y_shift": y_shift,
                        "x_shift": x_shift,
                    })
            particle_shifts_df = pd.DataFrame(rows)
        else:
            # Compute particle_shifts from deformation_field
            if refine_config_path is None:
                raise ValueError(
                    "particle_shift_path is specified but refine_config_path is not "
                    "provided. Particle shifts can only be computed when polishing "
                    "particles."
                )
            # Use the original movie (before correction) to compute shifts
            # since we want the shifts that were applied during correction
            particle_shifts_df = compute_particle_shifts(
                deformation_field=deformation_field,
                movie=movie_prepared,
                refine_config_path=refine_config_path,
                pixel_spacing=movie_config.pixel_size,
                grid_type=grid_type,
                device=device,
            )
        # Save to CSV
        particle_shifts_df.to_csv(
            output_config.particle_shift_path,
            index=False,
        )
