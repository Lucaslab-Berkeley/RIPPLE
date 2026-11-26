"""Configuration of the DeCo-LACE sum re-fill (`ripple.core.sum_refill`)."""

from pathlib import Path
from typing import Literal

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)
from teamtomo_basemodel import BaseModelTeamTomo


class RefillConfig(BaseModelTeamTomo):
    """Re-fill a DeCo-LACE sum outside its beam and write its 2DTM search mask.

    Parameters
    ----------
    sum_path: str
        The sum to re-fill (MRC), e.g. a dose-weighted sum from the DeCo-LACE pipeline
        or from ``resum``.
    beam_mask_result_path: str | None
        JSON of the :class:`~ripple.config.BeamMaskResult` the DeCo-LACE pipeline wrote
        for this movie (``beam_mask_output_path``). The ellipse is placed in the sum's
        frame from its output crop and the binning. Give this or ``beam_mask_path``.
    beam_mask_path: str | None
        Instead: a mask image in the sum's frame (``.npy``, ``.mrc``, ``.tif``), nonzero
        inside the beam region to keep. It is used as given (``diameter_reduction`` does
        not apply).
    diameter_reduction: float
        Fractional shrink of the ellipse radii for the region kept and searched. The
        default 0.06 drops the fringe inside the rim of a DeCo-LACE beam.
    binning: float | None
        Raw-frame pixels per sum pixel. Default: the output crop's extent over the sum's
        shape (the super-resolution factor).
    taper_sigma_px: float
        Blur of the beam mask that sets the blend weight, sum pixels.
    interior_erosion_px: int
        Erosion of the beam (minus contamination) for the spectrum's tiles, pixels.
    psd_box: int
        Tile size of the interior spectrum, pixels.
    psd_step: int
        Tile stride, pixels.
    contamination: bool
        Find dark (or bright) blobs inside the beam and leave them out of the search
        mask.
    contamination_band_px: tuple[float, float]
        Gaussian standard deviations of the band-pass used to find them, pixels.
    contamination_threshold_sd: float
        Robust standard deviations from the band-pass median that mark a blob pixel.
    contamination_min_area_px: int
        Smallest blob kept, pixels.
    contamination_polarity: str
        ``"dark"``, ``"bright"`` or ``"both"``.
    exclusion_radius_px: float
        Dilation of the blobs in the search mask, pixels; about a particle radius.
    border_px: int
        No search position within this many pixels of the image border.
    seed: int
        Seed of the noise.
    refilled_sum_output_path: str
        Output MRC.
    search_mask_output_path: str
        Output search mask: ``.npy`` (boolean) or ``.mrc`` (0/1), ``True`` where a
        particle centre may lie.
    stats_output_path: str | None
        Optional JSON of the fractions, blob count and the re-fill's power check.
    allow_file_overwrite: bool
        Overwrite existing outputs.
    """

    sum_path: str
    beam_mask_result_path: str | None = None
    beam_mask_path: str | None = None
    diameter_reduction: float = Field(default=0.06, ge=0.0, lt=1.0)
    binning: PositiveFloat | None = None

    taper_sigma_px: PositiveFloat = 2.0
    interior_erosion_px: NonNegativeInt = 8
    psd_box: int = Field(default=512, ge=32)
    psd_step: int = Field(default=256, ge=1)

    contamination: bool = True
    contamination_band_px: tuple[PositiveFloat, PositiveFloat] = (8.0, 150.0)
    contamination_threshold_sd: PositiveFloat = 5.0
    contamination_min_area_px: NonNegativeInt = 1000
    contamination_polarity: Literal["dark", "bright", "both"] = "dark"
    exclusion_radius_px: NonNegativeFloat = 100.0
    border_px: NonNegativeInt = 2
    seed: int = 0

    refilled_sum_output_path: str
    search_mask_output_path: str
    stats_output_path: str | None = None
    allow_file_overwrite: bool = False

    @model_validator(mode="after")
    def _one_beam_source(self) -> "RefillConfig":
        if (self.beam_mask_result_path is None) == (self.beam_mask_path is None):
            raise ValueError(
                "give exactly one of beam_mask_result_path or beam_mask_path"
            )
        if Path(self.search_mask_output_path).suffix.lower() not in {".npy", ".mrc"}:
            raise ValueError("search_mask_output_path must end in .npy or .mrc")
        return self
