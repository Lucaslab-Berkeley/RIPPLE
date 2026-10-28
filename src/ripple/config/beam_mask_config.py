"""Configuration and result models for DeCo-LACE beam mask estimation."""

from typing import Literal

import torch
from pydantic import PositiveFloat
from teamtomo_basemodel import BaseModelTeamTomo

from ripple.config.crop_bounds_config import CropBoundsConfig
from ripple.core.beam_mask import make_ellipse_mask
from ripple.core.crop_bounds import CropBounds


class BeamMaskConfig(BaseModelTeamTomo):
    """Algorithm parameters for DeCo-LACE beam mask estimation.

    Parameters
    ----------
    threshold_method: Literal["sobel"]
        Method used to fit the beam ellipse to the frame sum. ``"sobel"`` fits the
        ellipse to the outermost Sobel edge along rays cast from the beam center, and is
        robust to dark sample lying across the beam rim. Currently only ``"sobel"`` is
        supported. Default is ``"sobel"``.
    diameter_reduction: float
        Fractional amount by which both ellipse radii are shrunk before creating the
        final mask. ``0.0`` applies no reduction; ``0.1`` shrinks each axis by 10% to
        exclude fringe artifacts at the beam edge. Default is 0.0.
    working_pixel_size: float
        Approximate pixel size, in Angstroms, the frame sum is binned to before edge
        detection. Default is 8.0.
    edge_smoothing_sigma: float
        Standard deviation, in Angstroms, of the Gaussian smoothing applied to the
        binned frame sum before edge detection. Default is 120.0.
    crop_bounds_config: CropBoundsConfig
        Size policy for cropping movie frames down to the region covered by the beam
        mask. Default disables cropping (``mode="none"``).
    """

    threshold_method: Literal["sobel"] = "sobel"
    diameter_reduction: float = 0.0
    working_pixel_size: PositiveFloat = 8.0
    edge_smoothing_sigma: PositiveFloat = 120.0
    crop_bounds_config: CropBoundsConfig = CropBoundsConfig()


class BeamMaskResult(BaseModelTeamTomo):
    """Fitted ellipse and crop bounds for a DeCo-LACE beam mask.

    Ellipse parameters are stored in the ``cv2.fitEllipse`` convention: ``axis1`` lies
    along ``angle_deg`` and ``axis2`` lies perpendicular to it. ``angle_deg`` is
    measured clockwise from the horizontal as seen on screen.

    Parameters
    ----------
    center_y: float
        Row coordinate of the ellipse centroid in pixels.
    center_x: float
        Column coordinate of the ellipse centroid in pixels.
    axis1: float
        Semi-axis radius in pixels along the ``angle_deg`` direction.
    axis2: float
        Semi-axis radius in pixels perpendicular to ``angle_deg``.
    angle_deg: float
        Angle of ``axis1`` from horizontal in degrees, clockwise on screen
        (cv2.fitEllipse convention).
    diameter_reduction: float
        Fractional shrink applied to both radii before computing the mask and
        crop bounds. ``0.0`` means no reduction.
    image_shape_y: int
        Full source frame height in pixels.
    image_shape_x: int
        Full source frame width in pixels.
    crop_min_y: int
        Top row of the tight bounding-box crop region.
    crop_max_y: int
        Bottom row of the crop region, inclusive.
    crop_min_x: int
        Left column of the tight bounding-box crop region.
    crop_max_x: int
        Right column of the crop region, inclusive.
    output_crop_min_y: int
        Top row of the requested output crop region (see `CropBoundsConfig`).
    output_crop_max_y: int
        Bottom row of the output crop region, inclusive.
    output_crop_min_x: int
        Left column of the output crop region.
    output_crop_max_x: int
        Right column of the output crop region, inclusive.
    threshold_method: str
        Method used to fit the beam ellipse (``"sobel"``).
    pixel_size: float
        Pixel size in Angstroms per pixel.
    n_edge_points: int
        Number of beam-edge points found (at most one per ray cast).
    n_inliers: int
        Number of edge points kept as beam-rim inliers.
        ``n_inliers / ripple.core.beam_mask.DEFAULT_N_RAYS`` is the fraction of the
        rim that was found.
    """

    center_y: float
    center_x: float
    axis1: float
    axis2: float
    angle_deg: float
    diameter_reduction: float = 0.0
    image_shape_y: int
    image_shape_x: int
    crop_min_y: int
    crop_max_y: int
    crop_min_x: int
    crop_max_x: int
    output_crop_min_y: int
    output_crop_max_y: int
    output_crop_min_x: int
    output_crop_max_x: int
    threshold_method: str
    pixel_size: float
    n_edge_points: int
    n_inliers: int

    @property
    def crop_bounds(self) -> CropBounds:
        """Tight bounding box as a CropBounds dict."""
        return CropBounds(
            min_y=self.crop_min_y,
            max_y=self.crop_max_y,
            min_x=self.crop_min_x,
            max_x=self.crop_max_x,
        )

    @property
    def output_crop_bounds(self) -> CropBounds:
        """Requested output crop region as a CropBounds dict."""
        return CropBounds(
            min_y=self.output_crop_min_y,
            max_y=self.output_crop_max_y,
            min_x=self.output_crop_min_x,
            max_x=self.output_crop_max_x,
        )

    def to_mask(self) -> torch.Tensor:
        """Regenerate the boolean beam mask from the stored ellipse parameters.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(image_shape_y, image_shape_x)`` where
            ``True`` marks pixels inside the (possibly shrunk) ellipse.
        """
        mask_np = make_ellipse_mask(
            shape=(self.image_shape_y, self.image_shape_x),
            center_y=self.center_y,
            center_x=self.center_x,
            axis1=self.axis1,
            axis2=self.axis2,
            angle_deg=self.angle_deg,
            diameter_reduction=self.diameter_reduction,
        )
        return torch.from_numpy(mask_np)
