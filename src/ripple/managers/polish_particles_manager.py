"""Manager for polishing particles."""

from typing import TYPE_CHECKING, Any, ClassVar

import einops
import pandas as pd
import torch
import yaml
from pydantic import ConfigDict

from ripple.config import (
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
    PolishParticlesConfig,
)
from ripple.core import core_polish_particles
from ripple.managers import manager_utils
from ripple.utils.custom_types import BaseModelRIPPLE
from ripple.utils.data_io import read_mrc_to_tensor

if TYPE_CHECKING:
    from torch_motion_correction import OptimizationTracker


class PolishParticlesManager(BaseModelRIPPLE):
    """Manager for aligning frames of a cryo-EM movie."""

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)
    computational_config: ComputationalConfig
    movie_config: MovieConfig
    output_config: OutputConfig
    alignment_config: PolishParticlesConfig

    # pylint: disable=too-many-locals
    def setup_backend_kwargs(
        self,
        movie: torch.Tensor,
        gain_map: torch.Tensor,
        dark_map: torch.Tensor,
        deformation_field: torch.Tensor,
    ) -> dict[str, Any]:
        """Setup the backend kwargs for the align frames manager."""
        loss_trajectories = self.output_config.loss_trajectories_output_path is not None
        optimizer_kwargs = {"lr": self.alignment_config.learning_rate}
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.2}
        # Load YAML config to get the actual CSV path
        refine_config_yaml_path = self.alignment_config.particle_df_path
        with open(refine_config_yaml_path, encoding="utf-8") as f:
            refine_config = yaml.safe_load(f)
        df_path = refine_config["particle_stack"]["df_path"]

        var_image, mean_image, voltage, particle_indices = _load_refine_results(df_path)
        backend_kwargs = {
            "movie": movie,
            "initial_deformation_field": deformation_field,
            "refine_config_path": refine_config_yaml_path,
            "var_image": var_image,
            "mean_image": mean_image,
            "particle_indices": particle_indices,
            "gain_map": gain_map,
            "dark_map": dark_map,
            "gain_flip": self.movie_config.gain_flip,
            "gain_rot": self.movie_config.gain_rot,
            "pixel_size": self.movie_config.pixel_size,
            "deformation_field_resolution": (
                self.alignment_config.deformation_field_resolution
            ),
            "pre_exposure": self.movie_config.pre_exposure,
            "fluence_per_frame": self.movie_config.fluence_per_frame,
            "multiply_gain": self.movie_config.multiply_gain,
            "loss_trajectories": loss_trajectories,
            "skip_movie_preparation": self.alignment_config.skip_movie_preparation,
            "n_iterations": self.alignment_config.n_iterations,
            "optimizer_kwargs": optimizer_kwargs,
            "grid_type": self.alignment_config.grid_type,
            "voltage": voltage,
            "device": self.computational_config.gpu_id,
            "loss_metric": self.alignment_config.loss_metric,
            "min_snr": self.alignment_config.min_snr,
            "best_n": self.alignment_config.best_n,
        }
        return backend_kwargs

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def run_polish_particles(
        self,
        movie: torch.Tensor | None = None,
        gain_map: torch.Tensor | None = None,
        dark_map: torch.Tensor | None = None,
        deformation_field: torch.Tensor | None = None,
        movie_extract: bool = True,
        particle_batch_size: int = 100,
        save_intermediate_fields: bool = False,
        intermediate_fields_dir: str = ".",
    ) -> None:
        """Align the frames of a cryo-EM movie.

        Parameters
        ----------
        output_dataframe_path: str
            Path to save the output dataframe.
        movie: Optional[torch.Tensor]
            Movie tensor. If provided, will not be loaded from config.
        gain_map: Optional[torch.Tensor]
            Gain map tensor. If provided, will not be loaded from config.
        dark_map: Optional[torch.Tensor]
            Dark map tensor. If provided, will not be loaded from config.
        deformation_field: Optional[torch.Tensor]
            Deformation field tensor. If provided, will not be loaded from config.
        movie_extract: bool
            Whether to extract the movie.
        particle_batch_size: int
            Number of particles to process per batch for gradient accumulation.
        save_intermediate_fields: bool
            Whether to save the intermediate fields.
        intermediate_fields_dir: str | None
            Directory to save the intermediate fields.
        """
        (
            movie,
            gain_map,
            dark_map,
            deformation_field,
        ) = manager_utils.load_missing_tensors(
            self.computational_config,
            self.movie_config,
            self.alignment_config,
            movie,
            gain_map,
            dark_map,
            deformation_field,
        )
        core_kwargs = self.setup_backend_kwargs(
            movie, gain_map, dark_map, deformation_field
        )
        trajectory: OptimizationTracker | None = None
        corrected_movie, updated_deformation_field, movie_prepared, trajectory = (
            core_polish_particles(
                **core_kwargs,
                do_correct_motion=True,
                movie_extract=movie_extract,
                particle_batch_size=particle_batch_size,
                save_intermediate_fields=save_intermediate_fields,
                intermediate_fields_dir=intermediate_fields_dir,
            )
        )

        manager_utils.save_results(
            self.output_config,
            self.movie_config,
            corrected_movie,
            updated_deformation_field,
            movie_prepared,
            trajectory,
        )


def _load_refine_results(
    refine_config_path: str,
) -> tuple[torch.Tensor, torch.Tensor, float, list[pd.Index]]:
    """Load the var avg images from the refine config path.

    Parameters
    ----------
    refine_config_path: str
        Path to the refine config CSV file.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, float, list[pd.Index]]
        Tuple containing (variance_image, mean_image, voltage, indices_list)
        as tensors and voltage and indices_list.
        particle_indices is a single pandas Index
        with all row indices (shape: (n_rows,)).
    """
    # Load the refine config csv to dataframe
    df = pd.read_csv(
        refine_config_path,
        index_col=0,  # Use first column as index since it starts with a comma
        on_bad_lines="skip",  # Skip malformed lines (pandas >= 1.3)
    )
    var_image_path = df["correlation_variance_path"].iloc[0]
    mean_image_path = df["correlation_average_path"].iloc[0]
    voltage = df["voltage"].iloc[0]
    # Get all row indices as a single pandas Index
    # This will be a single Index object with shape (n_rows,) containing all row indices
    indices_list = []
    particle_indices = df.index
    indices_list.append(particle_indices)
    # Load MRC files and convert to tensors
    var_image = read_mrc_to_tensor(var_image_path)
    mean_image = read_mrc_to_tensor(mean_image_path)

    if var_image.ndim == 2:
        var_image = einops.rearrange(var_image, "h w -> 1 h w")
    if mean_image.ndim == 2:
        mean_image = einops.rearrange(mean_image, "h w -> 1 h w")

    return var_image, mean_image, voltage, indices_list
