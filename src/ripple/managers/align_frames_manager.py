"""Manager for aligning frames of a cryo-EM movie."""

from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch
from pydantic import ConfigDict
from torch_motion_correction import DeformationField

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
        initial_deformation_field: DeformationField | None = None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for :func:`~ripple.core.core_align_frames`.

        Parameters
        ----------
        movie: torch.Tensor
            Prepared (gain/dark corrected, mean-zero'd) movie tensor.
        initial_deformation_field: DeformationField | None
            External override for the starting deformation field (e.g. the
            result of a previous alignment pass). When None the manager falls
            back to ``alignment_config.initial_deformation_field``, which loads
            from disk or returns None for zero-initialisation.
        """
        deformation_field = (
            initial_deformation_field
            if initial_deformation_field is not None
            else self.alignment_config.initial_deformation_field
        )
        loss_trajectories = self.output_config.loss_trajectories_output_path is not None
        return {
            "movie": movie,
            "initial_deformation_field": deformation_field,
            "pixel_size": self.movie_config.pixel_size,
            "deformation_field_resolution": (
                self.alignment_config.deformation_field_resolution
            ),
            "loss_trajectories": loss_trajectories,
            "patch_sampling": self.alignment_config.as_patch_sampling_config,
            "fourier_filter": self.alignment_config.as_fourier_filter_config,
            "optimization": self.alignment_config.as_optimization_config,
            "device": self.computational_config.gpu_device,
        }

    def align_frames_last_pass(
        self,
        movie: torch.Tensor | None = None,
        gain_map: torch.Tensor | None = None,
        dark_map: torch.Tensor | None = None,
        initial_deformation_field: DeformationField | None = None,
    ) -> None:
        """Run a final alignment pass and save all requested outputs.

        Parameters
        ----------
        movie: torch.Tensor | None
            Movie tensor. If None, loaded from config.
        gain_map: torch.Tensor | None
            Gain map tensor. If None, loaded from config.
        dark_map: torch.Tensor | None
            Dark map tensor. If None, loaded from config.
        initial_deformation_field: DeformationField | None
            Starting deformation field. If None, loaded from config.
        """
        movie, gain_map, dark_map, initial_deformation_field = (
            manager_utils.load_missing_tensors(
                self.computational_config,
                self.movie_config,
                self.alignment_config,
                movie,
                gain_map,
                dark_map,
                initial_deformation_field,
            )
        )
        if not self.alignment_config.skip_movie_preparation:
            movie = self.movie_config.prepare(movie, gain_map, dark_map)
        core_kwargs = self.setup_backend_kwargs(movie, initial_deformation_field)
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
        initial_deformation_field: DeformationField | None = None,
    ) -> tuple[DeformationField, torch.Tensor, Optional["OptimizationTracker"]]:
        """Run an alignment pass without final output saving (for multi-pass workflows).

        Parameters
        ----------
        save_intermediate: bool
            Whether to save intermediate results.
        movie: torch.Tensor | None
            Movie tensor. If None, loaded from config.
        gain_map: torch.Tensor | None
            Gain map tensor. If None, loaded from config.
        dark_map: torch.Tensor | None
            Dark map tensor. If None, loaded from config.
        initial_deformation_field: DeformationField | None
            Starting deformation field. If None, loaded from config.

        Returns
        -------
        tuple[DeformationField, torch.Tensor, OptimizationTracker | None]
            (updated_deformation_field, movie_prepared, trajectory)
        """
        movie, gain_map, dark_map, initial_deformation_field = (
            manager_utils.load_missing_tensors(
                self.computational_config,
                self.movie_config,
                self.alignment_config,
                movie,
                gain_map,
                dark_map,
                initial_deformation_field,
            )
        )
        if not self.alignment_config.skip_movie_preparation:
            movie = self.movie_config.prepare(movie, gain_map, dark_map)
        core_kwargs = self.setup_backend_kwargs(movie, initial_deformation_field)
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
            manager_utils.save_results(
                self.output_config,
                self.movie_config,
                corrected_movie,
                updated_deformation_field,
                movie_prepared,
                trajectory,
            )

        return updated_deformation_field, movie_prepared, trajectory
