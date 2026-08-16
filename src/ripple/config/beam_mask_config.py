"""Configuration and result models for DeCo-LACE beam mask estimation."""

from typing import Literal

import torch

from ripple.core.beam_mask import make_ellipse_mask
from ripple.utils.custom_types import BaseModelRIPPLE


class BeamMaskConfig(BaseModelRIPPLE):
    """Algorithm parameters for DeCo-LACE beam mask estimation.

    Parameters
    ----------
    threshold_method: Literal["otsu"]
        Thresholding algorithm used to binarise the low-pass-filtered frame sum.
        Currently only ``"otsu"`` is supported. Default is ``"otsu"``.
    diameter_reduction: float
        Fractional amount by which both ellipse radii are shrunk before creating the
        final mask. ``0.0`` applies no reduction; ``0.1`` shrinks each axis by 10% to
        exclude fringe artifacts at the beam edge. Default is 0.0.
    low_pass_resolution: float
        Low-pass filter cutoff, in Angstroms, applied to the frame sum before
        thresholding. Default is 100.0.
    """

    threshold_method: Literal["otsu"] = "otsu"
    diameter_reduction: float = 0.0
    low_pass_resolution: float = 100.0


class BeamMaskResult(BaseModelRIPPLE):
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
    threshold_method: str
        Thresholding algorithm used to binarise the filtered frame sum.
    pixel_size: float
        Pixel size in Angstroms per pixel.
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
    threshold_method: str
    pixel_size: float

    @property
    def crop_bounds(self) -> tuple[int, int, int, int]:
        """Tight bounding box as ``(min_y, max_y, min_x, max_x)``."""
        return self.crop_min_y, self.crop_max_y, self.crop_min_x, self.crop_max_x

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
