"""Manager for aligning frames of a cryo-EM movie."""

from typing import Any, ClassVar

import torch
from pydantic import ConfigDict
from torch_motion_correction import (
    DeformationField,
    OptimizationTracker,
    estimate_global_motion,
)

from ripple.config import (
    AlignFramesConfig,
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.core import core_align_frames, core_estimate_motion
from ripple.managers import manager_utils
from ripple.utils.custom_types import BaseModelRIPPLE


class AlignFramesManager(BaseModelRIPPLE):
    """Manager for aligning frames of a cryo-EM movie."""

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)
    computational_config: ComputationalConfig
    movie_config: MovieConfig
    output_config: OutputConfig
    alignment_config: AlignFramesConfig

    def _setup_estimation_kwargs(
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
        return {
            "movie": movie,
            "initial_deformation_field": deformation_field,
            "pixel_size": self.movie_config.pixel_size,
            "deformation_field_resolution": (
                self.alignment_config.deformation_field_resolution
            ),
            "patch_sampling": self.alignment_config.as_patch_sampling_config,
            "fourier_filter": self.alignment_config.as_fourier_filter_config,
            "optimization": self.alignment_config.as_optimization_config,
            "device": self.computational_config.gpu_device,
        }

    def prepare_movie(
        self,
        movie: torch.Tensor | None = None,
        gain_map: torch.Tensor | None = None,
        dark_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Load and prepare the movie (gain/dark correction, mean-zero).

        Parameters
        ----------
        movie: torch.Tensor | None
            Raw movie tensor. If None, loaded from ``movie_config``.
        gain_map: torch.Tensor | None
            Gain map tensor. If None, loaded from ``movie_config``.
        dark_map: torch.Tensor | None
            Dark map tensor. If None, loaded from ``movie_config``.

        Returns
        -------
        torch.Tensor
            Prepared movie tensor ready for motion estimation.
        """
        movie, gain_map, dark_map, _ = manager_utils.load_missing_tensors(
            self.computational_config,
            self.movie_config,
            self.alignment_config,
            movie,
            gain_map,
            dark_map,
            initial_deformation_field=None,
        )
        return self.movie_config.prepare(movie, gain_map, dark_map)

    def estimate_motion(
        self,
        movie: torch.Tensor,
        initial_deformation_field: DeformationField | None = None,
    ) -> tuple[DeformationField, OptimizationTracker]:
        """Estimate motion via optional XC pre-pass then gradient-based optimization.

        If ``alignment_config.use_xc_prepass`` is True and no
        ``initial_deformation_field`` is provided, a fast whole-image
        cross-correlation pass runs first to estimate global per-frame shifts.
        Those shifts seed the deformation field so the gradient optimizer
        starts from a good global motion estimate.

        Parameters
        ----------
        movie: torch.Tensor
            Prepared movie tensor (output of :meth:`prepare_movie`).
        initial_deformation_field: DeformationField | None
            Explicit starting field. When provided, the XC pre-pass is skipped.
            When None and ``use_xc_prepass`` is True, the XC pre-pass runs
            automatically.

        Returns
        -------
        tuple[DeformationField, OptimizationTracker]
            Estimated deformation field and optimization history.
        """
        device = self.computational_config.gpu_device
        movie = movie.to(device)

        if self.alignment_config.use_xc_prepass and initial_deformation_field is None:
            initial_deformation_field = estimate_global_motion(
                image=movie,
                pixel_spacing=self.movie_config.pixel_size,
                fourier_filter=self.alignment_config.as_fourier_filter_config,
                device=device,
            )

        kwargs = self._setup_estimation_kwargs(movie, initial_deformation_field)
        return core_estimate_motion(**kwargs)

    def correct_and_save(
        self,
        movie: torch.Tensor,
        deformation_field: DeformationField,
        trajectory: OptimizationTracker,
    ) -> None:
        """Warp the movie with the estimated deformation field and save outputs.

        Parameters
        ----------
        movie: torch.Tensor
            Prepared movie tensor (output of :meth:`prepare_movie`).
        deformation_field: DeformationField
            Estimated deformation field (output of :meth:`estimate_motion`).
        trajectory: OptimizationTracker
            Optimization history (output of :meth:`estimate_motion`).
        """
        device = self.computational_config.gpu_device
        corrected_movie = core_align_frames(
            movie=movie,
            deformation_field=deformation_field,
            pixel_size=self.movie_config.pixel_size,
            device=device,
        )
        manager_utils.save_results(
            self.output_config,
            self.movie_config,
            corrected_movie,
            deformation_field,
            movie,
            trajectory,
        )
