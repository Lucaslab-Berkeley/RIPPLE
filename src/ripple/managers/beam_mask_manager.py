"""Manager for estimating a DeCo-LACE beam mask from a cryo-EM movie."""

from typing import ClassVar

import torch
from pydantic import ConfigDict

from ripple.config import (
    BeamMaskConfig,
    BeamMaskResult,
    ComputationalConfig,
    MovieConfig,
)
from ripple.core.beam_mask import estimate_beam_mask, sum_movie_chunked
from ripple.utils.custom_types import BaseModelRIPPLE


class BeamMaskManager(BaseModelRIPPLE):
    """Lightweight manager for estimating a DeCo-LACE beam mask from a cryo-EM movie."""

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)
    computational_config: ComputationalConfig
    movie_config: MovieConfig
    beam_mask_config: BeamMaskConfig

    def estimate(self, movie: torch.Tensor | None = None) -> BeamMaskResult:
        """Sum raw frames and fit the beam mask ellipse.

        Parameters
        ----------
        movie : torch.Tensor | None
            Raw movie tensor. If provided, it is used as-is. If None, loaded from
            ``movie_config`` (``movie_config.fluence``/``fluence_per_frame`` are not
            read by this manager and may be left at any value, e.g. 0.0, if the movie is
            only being used for beam mask estimation).

        Returns
        -------
        BeamMaskResult
            Fitted ellipse and crop-bound parameters.
        """
        if movie is None:
            movie = self.movie_config.movie

        device = self.computational_config.gpu_device
        frame_sum = sum_movie_chunked(movie, device=device)

        result_dict = estimate_beam_mask(
            frame_sum,
            self.movie_config.pixel_size,
            self.beam_mask_config.threshold_method,
            self.beam_mask_config.diameter_reduction,
            self.beam_mask_config.low_pass_resolution,
            device=device,
        )
        return BeamMaskResult(**result_dict)
