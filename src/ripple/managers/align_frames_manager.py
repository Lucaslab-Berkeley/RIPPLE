"""Manager for aligning frames of a cryo-EM movie."""

from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch
from pydantic import ConfigDict

from ripple.config import (
    AlignFramesConfig,
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.core import core_align_frames
from ripple.managers import manager_utils
from ripple.utils.custom_types import BaseModelRIPPLE

if TYPE_CHECKING:
    from torch_motion_correction import OptimizationTracker


class AlignFramesManager(BaseModelRIPPLE):
    """Manager for aligning frames of a cryo-EM movie."""

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)
    computational_config: ComputationalConfig
    movie_config: MovieConfig
    output_config: OutputConfig
    alignment_config: AlignFramesConfig

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
        backend_kwargs = {
            "movie": movie,
            "deformation_field": deformation_field,
            "gain_map": gain_map,
            "dark_map": dark_map,
            "gain_flip": self.movie_config.gain_flip,
            "gain_rot": self.movie_config.gain_rot,
            "multiply_gain": self.movie_config.multiply_gain,
            "pixel_size": self.movie_config.pixel_size,
            "deformation_field_resolution": (
                self.alignment_config.deformation_field_resolution
            ),
            "loss_trajectories": loss_trajectories,
            "skip_movie_preparation": self.alignment_config.skip_movie_preparation,
            "n_iterations": self.alignment_config.n_iterations,
            "patch_shape": self.alignment_config.patch_shape,
            "grid_type": self.alignment_config.grid_type,
            "loss_type": self.alignment_config.loss_type,
            "optimizer_type": self.alignment_config.optimizer_type,
            "b_factor": self.alignment_config.b_factor,
            "frequency_range": self.alignment_config.frequency_range,
            "optimizer_kwargs": optimizer_kwargs,
            "device": self.computational_config.gpu_id,
        }
        return backend_kwargs

    def align_frames_last_pass(
        self,
        movie: torch.Tensor | None = None,
        gain_map: torch.Tensor | None = None,
        dark_map: torch.Tensor | None = None,
        deformation_field: torch.Tensor | None = None,
    ) -> None:
        """Align the frames of a cryo-EM movie.

        Parameters
        ----------
        movie: Optional[torch.Tensor]
            Movie tensor. If provided, will not be loaded from config.
        gain_map: Optional[torch.Tensor]
            Gain map tensor. If provided, will not be loaded from config.
        dark_map: Optional[torch.Tensor]
            Dark map tensor. If provided, will not be loaded from config.
        deformation_field: Optional[torch.Tensor]
            Deformation field tensor. If provided, will not be loaded from config.
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
            core_align_frames(**core_kwargs, do_correct_motion=True)
        )

        manager_utils.save_results(
            self.output_config,
            self.movie_config,
            corrected_movie,
            updated_deformation_field,
            movie_prepared,
            trajectory,
        )

    # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    def align_frames_first_passes(
        self,
        save_intermediate: bool = False,
        movie: torch.Tensor | None = None,
        gain_map: torch.Tensor | None = None,
        dark_map: torch.Tensor | None = None,
        deformation_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional["OptimizationTracker"]]:
        """Align the frames of a cryo-EM movie.

        Parameters
        ----------
        save_intermediate: bool
            Whether to save intermediate results.
        movie: Optional[torch.Tensor]
            Movie tensor. If provided, will not be loaded from config.
        gain_map: Optional[torch.Tensor]
            Gain map tensor. If provided, will not be loaded from config.
        dark_map: Optional[torch.Tensor]
            Dark map tensor. If provided, will not be loaded from config.
        deformation_field: Optional[torch.Tensor]
            Deformation field tensor. If provided, will not be loaded from config.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, Optional[dict[str, Any]]]
            Tuple of (updated_deformation_field, movie_prepared, trajectory).
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
        do_correct_motion = save_intermediate
        (
            corrected_movie,
            updated_deformation_field,
            movie_prepared,
            trajectory,
        ) = core_align_frames(
            **core_kwargs,
            do_correct_motion=do_correct_motion,
        )
        if save_intermediate:
            self.save_results(
                corrected_movie=corrected_movie,
                updated_deformation_field=updated_deformation_field,
                movie_prepared=movie_prepared,
                trajectory=trajectory,
            )

        return updated_deformation_field, movie_prepared, trajectory
