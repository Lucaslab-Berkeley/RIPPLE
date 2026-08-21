"""Manager for aligning frames of a cryo-EM movie."""

from typing import Any, ClassVar

import torch
from pydantic import ConfigDict
from teamtomo_basemodel import BaseModelTeamTomo
from torch_motion_correction import (
    DeformationField,
    OptimizationTracker,
    correct_motion,
    estimate_global_motion,
    estimate_local_motion,
)

from ripple.config import (
    AlignFramesConfig,
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.core.crop_bounds import CropBounds
from ripple.managers import manager_utils


class AlignFramesManager(BaseModelTeamTomo):
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
        """Build the kwargs for :func:`~torch_motion_correction.estimate_local_motion`.

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
            "image": movie,
            "initial_deformation_field": deformation_field,
            "pixel_spacing": self.movie_config.pixel_size,
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
        mask: torch.Tensor | None = None,
        crop_bounds: CropBounds | None = None,
    ) -> torch.Tensor:
        """Load and prepare the movie (gain/dark correction, mask, mean-zero).

        Honors ``alignment_config.skip_movie_preparation``. When True, the loaded movie
        is returned as-is with no gain/dark/mask/mean-zero/crop correction applied.

        Parameters
        ----------
        movie: torch.Tensor | None
            Raw movie tensor. If provided, it is used as-is and never read from
            disk (``movie_config.movie_path`` is ignored). If None, loaded from
            ``movie_config``.
        gain_map: torch.Tensor | None
            Gain map tensor. If None, loaded from ``movie_config``.
        dark_map: torch.Tensor | None
            Dark map tensor. If None, loaded from ``movie_config``.
        mask: torch.Tensor | None
            Mask with shape (height, width) multiplied uniformly into every frame. If
            None, loaded from ``movie_config`` (or left unset if no ``mask_path`` is
            configured).
        crop_bounds: CropBounds | None
            Inclusive ``(min_y, max_y, min_x, max_x)`` crop bounds applied to the
            movie, gain map, dark map, and mask before any other preparation step.
            If None, no cropping is applied.

        Returns
        -------
        torch.Tensor
            Prepared movie tensor ready for motion estimation.
        """
        movie, gain_map, dark_map, mask = manager_utils.load_missing_tensors(
            self.computational_config,
            self.movie_config,
            movie,
            gain_map,
            dark_map,
            mask,
        )
        return manager_utils.prepare_movie_if_needed(
            self.movie_config,
            self.alignment_config,
            movie,
            gain_map,
            dark_map,
            mask,
            self.computational_config.gpu_device,
            storage_device=self.computational_config.movie_storage_device,
            crop_bounds=crop_bounds,
        )

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
        # Check only one of 'use_xc_prepass' and 'initial_deformation_field' is not None
        if (
            self.alignment_config.use_xc_prepass
            and initial_deformation_field is not None
        ):
            raise ValueError(
                "Cannot use both 'use_xc_prepass' and 'initial_deformation_field'. "
                "Select only one (or neither) of them to pre-seed optimization stage."
            )

        device = self.computational_config.gpu_device

        if self.alignment_config.use_xc_prepass:
            initial_deformation_field = estimate_global_motion(
                image=movie,
                pixel_spacing=self.movie_config.pixel_size,
                fourier_filter=self.alignment_config.as_fourier_filter_config,
                device=device,
                downsample_factor=self.alignment_config.xc_prepass_downsample_factor,
            )

        kwargs = self._setup_estimation_kwargs(movie, initial_deformation_field)
        torch.set_grad_enabled(True)
        return estimate_local_motion(**kwargs)  # type: ignore[no-any-return]

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
        corrected_movie = correct_motion(
            image=movie,
            deformation_field=deformation_field,
            pixel_spacing=self.movie_config.pixel_size,
            device=device,
        )
        manager_utils.save_results(
            self.output_config,
            self.movie_config,
            corrected_movie,
            deformation_field,
            movie,
            trajectory,
            device=device,
        )
