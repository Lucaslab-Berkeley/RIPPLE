"""Manager for aligning frames of a cryo-EM movie."""

from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch
from pydantic import ConfigDict

from ripple.config import (
    AlignmentConfig,
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.core import core_align_frames, generate_dose_weighted_image, sum_movie
from ripple.utils.custom_types import BaseModelRIPPLE
from ripple.utils.data_io import (
    save_deformation_field,
    write_mrc_from_tensor,
    write_trajectory_to_csv,
)

if TYPE_CHECKING:
    from torch_motion_correction import OptimizationTracker


class AlignFramesManager(BaseModelRIPPLE):
    """Manager for aligning frames of a cryo-EM movie."""

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)
    computational_config: ComputationalConfig
    movie_config: MovieConfig
    output_config: OutputConfig
    alignment_config: AlignmentConfig

    def _load_missing_tensors(
        self,
        movie: torch.Tensor | None = None,
        gain_map: torch.Tensor | None = None,
        dark_map: torch.Tensor | None = None,
        deformation_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load only the tensors that are not provided as arguments.

        Parameters
        ----------
        movie: Optional[torch.Tensor]
            Movie tensor. If None, will be loaded from config.
        gain_map: Optional[torch.Tensor]
            Gain map tensor. If None, will be loaded from config.
        dark_map: Optional[torch.Tensor]
            Dark map tensor. If None, will be loaded from config.
        deformation_field: Optional[torch.Tensor]
            Deformation field tensor. If None, will be loaded from config.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (movie, gain_map, dark_map, deformation_field), with missing
            ones loaded from config.
        """
        device = self.computational_config.gpu_id

        if movie is None:
            movie = self.movie_config.movie
            # if still not none, move to device
            if movie is not None:
                movie = movie.to(device)
        else:
            movie = movie.to(device)

        if gain_map is None:
            gain_map = self.movie_config.gain
            if gain_map is not None:
                gain_map = gain_map.to(device)
        else:
            gain_map = gain_map.to(device)

        if dark_map is None:
            dark_map = self.movie_config.dark
            if dark_map is not None:
                dark_map = dark_map.to(device)
        else:
            dark_map = dark_map.to(device)

        if deformation_field is None:
            deformation_field = self.alignment_config.deformation_field
            if deformation_field is not None:
                deformation_field = deformation_field.to(device)
        else:
            deformation_field = deformation_field.to(device)

        return movie, gain_map, dark_map, deformation_field

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
        movie, gain_map, dark_map, deformation_field = self._load_missing_tensors(
            movie, gain_map, dark_map, deformation_field
        )
        core_kwargs = self.setup_backend_kwargs(
            movie, gain_map, dark_map, deformation_field
        )
        trajectory: OptimizationTracker | None = None
        corrected_movie, updated_deformation_field, movie_prepared, trajectory = (
            core_align_frames(**core_kwargs, do_correct_motion=True)
        )

        self.save_results(
            corrected_movie=corrected_movie,
            updated_deformation_field=updated_deformation_field,
            movie_prepared=movie_prepared,
            trajectory=trajectory,
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
        movie, gain_map, dark_map, deformation_field = self._load_missing_tensors(
            movie, gain_map, dark_map, deformation_field
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

    def save_results(
        self,
        corrected_movie: torch.Tensor,
        updated_deformation_field: torch.Tensor,
        movie_prepared: torch.Tensor,
        trajectory: Optional["OptimizationTracker"] = None,
    ) -> None:
        """
        Save the results of the alignment.

        Parameters
        ----------
        corrected_movie: torch.Tensor
            The corrected movie.
        updated_deformation_field: torch.Tensor
            The updated deformation field.
        movie_prepared: torch.Tensor
            The prepared movie.
        trajectory: Optional[dict[str, Any]]
            The trajectory of the alignment.

        Returns
        -------
        None
            None.
        """
        # Save DW sum is wanted
        if self.output_config.dw_sum_output_path is not None:
            dw_movie = generate_dose_weighted_image(
                corrected_movie,
                self.movie_config.pixel_size,
                self.movie_config.pre_exposure,
                self.movie_config.fluence_per_frame,
                self.movie_config.voltage,
            )
            dw_movie = dw_movie.cpu()
            write_mrc_from_tensor(
                data=dw_movie,
                mrc_path=self.output_config.dw_sum_output_path,
                overwrite=self.output_config.allow_file_overwrite,
            )

        # Save deformation field if wanted
        if self.output_config.deformation_field_output_path is not None:
            updated_deformation_field = updated_deformation_field.cpu()
            save_deformation_field(
                updated_deformation_field,
                self.output_config.deformation_field_output_path,
            )

        # Save non-dw sum movie if wanted
        if self.output_config.non_dw_sum_output_path is not None:
            summed_movie = sum_movie(corrected_movie)
            summed_movie = summed_movie.cpu()
            write_mrc_from_tensor(
                data=summed_movie,
                mrc_path=self.output_config.non_dw_sum_output_path,
                overwrite=self.output_config.allow_file_overwrite,
            )

        # Save aligned movie if wanted
        if self.output_config.motion_corrected_movie_output_path is not None:
            corrected_movie = corrected_movie.cpu()
            write_mrc_from_tensor(
                data=corrected_movie,
                mrc_path=self.output_config.motion_corrected_movie_output_path,
                overwrite=self.output_config.allow_file_overwrite,
            )

        # Save rendered, unaligned movie if wanted
        if self.output_config.rendered_movie_output_path is not None:
            rendered_movie = movie_prepared.cpu()
            write_mrc_from_tensor(
                data=rendered_movie,
                mrc_path=self.output_config.rendered_movie_output_path,
                overwrite=self.output_config.allow_file_overwrite,
            )

        # Save loss trajectories if wanted
        if trajectory is not None:
            if self.output_config.loss_trajectories_output_path is not None:
                write_trajectory_to_csv(
                    trajectory=trajectory,
                    file_path=self.output_config.loss_trajectories_output_path,
                )
