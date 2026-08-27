"""Manager for estimating a DeCo-LACE beam mask from a cryo-EM movie."""

from typing import ClassVar

import torch
from pydantic import ConfigDict
from teamtomo_basemodel import BaseModelTeamTomo

from ripple.config import (
    BeamMaskConfig,
    BeamMaskResult,
    ComputationalConfig,
    MovieConfig,
)
from ripple.core.beam_mask import (
    estimate_beam_mask,
    make_ellipse_mask,
    sum_movie_chunked,
)


class BeamMaskManager(BaseModelTeamTomo):
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

        ellipse_mask = make_ellipse_mask(
            shape=(result_dict["image_shape_y"], result_dict["image_shape_x"]),
            center_y=result_dict["center_y"],
            center_x=result_dict["center_x"],
            axis1=result_dict["axis1"],
            axis2=result_dict["axis2"],
            angle_deg=result_dict["angle_deg"],
            diameter_reduction=result_dict["diameter_reduction"],
        )
        output_bounds = self.beam_mask_config.crop_bounds_config.determine_bounds(
            ellipse_mask
        )

        return BeamMaskResult(
            **result_dict,
            output_crop_min_y=output_bounds["min_y"],
            output_crop_max_y=output_bounds["max_y"],
            output_crop_min_x=output_bounds["min_x"],
            output_crop_max_x=output_bounds["max_x"],
        )
