"""Configuration for cropping movie frames down to (a region around) a mask."""

from pydantic import model_validator
from teamtomo_basemodel import BaseModelTeamTomo
from typing_extensions import Self

from ripple.core.crop_bounds import CropMode


class CropBoundsConfig(BaseModelTeamTomo):
    """Size policy for cropping movie frames to the region covered by a mask.

    Parameters
    ----------
    mode: Literal["none", "tight", "nice_size", "fixed_size"]
        Crop size policy. ``"none"`` (default) disables cropping entirely. ``"tight"``
        crops to the smallest bounding box containing the mask. ``"nice_size"`` grows
        (or, for masks with excluded interior areas, shrinks) the tight crop so each
        side is a multiple of `round_to` -- useful for FFT-friendly sizes.
        ``"fixed_size"`` crops to an exact `target_shape` window centered on the mask
        -- useful for giving every micrograph in a collection the same output size.
    round_to: int
        Multiple crop side lengths are rounded to. Only used when
        ``mode="nice_size"``. Default is 1 (no-op).
    divisible_by: int
        Side lengths are constrained to be an exact multiple of this value, typically a
        movie's super-resolution factor.
    target_shape: tuple[int, int] | None
        ``(height, width)`` of the desired crop window. Required when
        ``mode="fixed_size"``, must be None otherwise. Default is None.
    """

    mode: CropMode = "none"
    round_to: int = 1
    divisible_by: int = 1
    target_shape: tuple[int, int] | None = None

    @model_validator(mode="after")  # type: ignore
    def validate_target_shape(self) -> Self:
        """Ensure `target_shape` is set iff `mode` is 'fixed_size', and divisible.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If `target_shape` is missing while `mode='fixed_size'`, set while
            `mode!='fixed_size'`, or isn't an exact multiple of `divisible_by` in both
            dimensions.
        """
        if self.mode == "fixed_size" and self.target_shape is None:
            raise ValueError("target_shape must be set when mode='fixed_size'.")
        if self.mode != "fixed_size" and self.target_shape is not None:
            raise ValueError("target_shape must be None unless mode='fixed_size'.")
        if (
            self.target_shape is not None
            and self.divisible_by > 1
            and (
                self.target_shape[0] % self.divisible_by != 0
                or self.target_shape[1] % self.divisible_by != 0
            )
        ):
            raise ValueError(
                f"target_shape {self.target_shape} is not an exact multiple of "
                f"divisible_by ({self.divisible_by}) in both dimensions."
            )
        return self
