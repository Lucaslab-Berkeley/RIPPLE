"""Serialization and validation of movie parameters for 2DTM."""

import torch
from pydantic import field_validator

from ripple.core.prepare_movie import prepare_movie
from ripple.utils.custom_types import BaseModelRIPPLE
from ripple.utils.data_io import load_tensor_from_path, render_eer_to_tensor


class MovieConfig(BaseModelRIPPLE):
    """Serialization and validation of movie parameters for RIPPLE.

    Parameters
    ----------
    movie_path: str
        Path to the movie file.
    pixel_size: float
        Pixel size in Angstroms per pixel.
    fluence: float
        Total fluence in electrons per Angstrom^2.
    fluence_per_frame: float
        Fluence per frame in electrons per Angstrom^2/frame.
    pre_exposure: float
        Pre-exposure time in seconds.
    voltage: float
        Accelerating voltage in kilovolts.
        Default is 300.0 kV.
    gain_path: Optional[str]
        Path to the gain map file. If None, the gain map will be initialized to zero.
    dark_path: Optional[str]
        Path to the dark map file. If None, the dark map will be initialized to zero.
    gain_flip: int
        Flip the gain map.
        0: no flip
        1: flipY
        2: flipX
    gain_rot: int
        Rotate the gain map.
        0: no rotation
        1: 90 degrees
        2: 180 degrees
        3: 270 degrees
    multiply_gain: bool
        Whether to multiply the movie by the gain map or divide the movie by the
        gain map. Default is True.
    """

    movie_path: str
    pixel_size: float
    fluence: float
    fluence_per_frame: float
    pre_exposure: float = 0.0
    voltage: float = 300.0
    gain_path: str | None = None
    gain_flip: int = 0
    gain_rot: int = 0
    multiply_gain: bool = True
    dark_path: str | None = None

    @field_validator("gain_flip")  # type: ignore[misc]
    @classmethod
    def validate_gain_flip(cls, v: int) -> int:
        """Validate that gain_flip is 0, 1, or 2."""
        if v not in (0, 1, 2):
            raise ValueError(f"gain_flip must be 0, 1, or 2, got {v}")
        return v

    @field_validator("gain_rot")  # type: ignore[misc]
    @classmethod
    def validate_gain_rot(cls, v: int) -> int:
        """Validate that gain_rot is 0, 1, 2, or 3."""
        if v not in (0, 1, 2, 3):
            raise ValueError(f"gain_rot must be 0, 1, 2, or 3, got {v}")
        return v

    def prepare(
        self,
        movie: torch.Tensor,
        gain_map: torch.Tensor | None = None,
        dark_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply gain, dark, hot-pixel, and mean-zero corrections to a movie.

        Parameters
        ----------
        movie : torch.Tensor
            Raw movie tensor (frames x height x width).
        gain_map : torch.Tensor | None
            Gain map tensor. If None, gain correction is skipped.
        dark_map : torch.Tensor | None
            Dark map tensor. If None, dark correction is skipped.

        Returns
        -------
        torch.Tensor
            Corrected movie tensor.
        """
        return prepare_movie(
            movie, gain_map, dark_map, self.gain_flip, self.gain_rot, self.multiply_gain
        )

    @property
    def movie(self) -> torch.Tensor:
        """Get the movie tensor."""
        if not self.movie_path:
            raise ValueError("Movie path is not set.")
        if self.movie_path.endswith(".eer"):
            return render_eer_to_tensor(
                self.movie_path, self.fluence_per_frame, self.fluence
            )
        return load_tensor_from_path(self.movie_path, expected_ndim=3)

    @property
    def gain(self) -> torch.Tensor:
        """Get the gain tensor."""
        if self.gain_path is None:
            return None
        return load_tensor_from_path(self.gain_path, expected_ndim=2)

    @property
    def dark(self) -> torch.Tensor:
        """Get the dark tensor."""
        if self.dark_path is None:
            return None
        return load_tensor_from_path(self.dark_path, expected_ndim=2)
