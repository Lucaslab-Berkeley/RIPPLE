"""Serialization and validation of movie parameters for 2DTM."""

import torch
from pydantic import PositiveInt, field_validator
from teamtomo_basemodel import BaseModelTeamTomo

from ripple.core.crop_bounds import CropBounds
from ripple.core.prepare_movie import DEFAULT_PREP_CHUNK_SIZE, prepare_movie
from ripple.utils.data_io import load_tensor_from_path, render_eer_to_tensor


class MovieConfig(BaseModelTeamTomo):
    """Serialization and validation of movie parameters for RIPPLE.

    Parameters
    ----------
    movie_path: str
        Path to the movie file.
    pixel_size: float
        Pixel size of the movie, as collected, in Angstroms per pixel.
    super_resolution_factor: int
        Integer factor relating the native `pixel_size` to the desired output pixel
        size of the final micrograph, i.e.
        `output_pixel_size = pixel_size * super_resolution_factor`.
        Default is 1 (no super-resolution).
    fluence: float
        Total fluence in electrons per Angstrom^2.
    fluence_per_frame: float
        Fluence per frame in electrons per Angstrom^2/frame.
    pre_exposure: float
        The total pre-exposure in (e-/A^2) before the first frame of the movie.
    voltage: float
        Accelerating voltage in kilovolts.
        Default is 300.0 kV.
    gain_path: Optional[str]
        Path to the gain map file. If None, the gain map will be initialized to zero.
    dark_path: Optional[str]
        Path to the dark map file. If None, the dark map will be initialized to zero.
    mask_path: Optional[str]
        Path to a (height, width) mask file applied uniformly to every frame during
        preparation. If None (default), no masking is applied.
    mask_fill_noise: bool
        If True (and mask_path is set), pixels where the mask is 0 are replaced with
        per-frame Poisson noise instead of being zeroed out. Default is False.
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

    movie_path: str | None = None
    pixel_size: float
    super_resolution_factor: PositiveInt = 1
    fluence: float
    fluence_per_frame: float
    pre_exposure: float = 0.0
    voltage: float = 300.0
    gain_path: str | None = None
    gain_flip: int = 0
    gain_rot: int = 0
    multiply_gain: bool = True
    dark_path: str | None = None
    mask_path: str | None = None
    mask_fill_noise: bool = False

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
        mask: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        storage_device: torch.device | str | None = None,
        chunk_size: int = DEFAULT_PREP_CHUNK_SIZE,
        crop_bounds: CropBounds | None = None,
    ) -> torch.Tensor:
        """Apply gain, dark, hot-pixel, mask, and mean-zero corrections to a movie.

        Parameters
        ----------
        movie : torch.Tensor
            Raw movie tensor (frames x height x width).
        gain_map : torch.Tensor | None
            Gain map tensor. If None, gain correction is skipped.
        dark_map : torch.Tensor | None
            Dark map tensor. If None, dark correction is skipped.
        mask : torch.Tensor | None
            Mask with shape (height, width) applied uniformly to every frame. If None,
            no masking is applied. If ``self.mask_fill_noise`` is True, then masked out
            pixels (``mask == 0``) are replaced with per-frame Poisson. Otherwise, those
            pixel locations are zeroed.
        device : torch.device | str | None
            Device each chunk's gain/dark/hot-pixel/mask compute runs on. If None, uses
            `movie`'s current device.
        storage_device : torch.device | str | None
            Device the returned, fully-prepared movie is stored on. If None, defaults to
            `device`.
        chunk_size : int
            Number of frames to transfer/correct at a time.
        crop_bounds : CropBounds | None
            Inclusive ``(min_y, max_y, min_x, max_x)`` crop bounds applied to `movie`,
            `gain_map`, `dark_map`, and `mask` before any other preparation step. If
            None, no cropping is applied.

        Returns
        -------
        torch.Tensor
            Corrected movie tensor.
        """
        return prepare_movie(
            movie,
            gain_map,
            dark_map,
            self.gain_flip,
            self.gain_rot,
            self.multiply_gain,
            mask=mask,
            mask_fill_noise=self.mask_fill_noise,
            device=device,
            storage_device=storage_device,
            chunk_size=chunk_size,
            crop_bounds=crop_bounds,
        )

    @property
    def output_pixel_size(self) -> float:
        """Pixel size of the final micrograph, after Fourier-crop downsampling."""
        return self.pixel_size * int(self.super_resolution_factor)

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

    @staticmethod
    def _load_2d(path: str | None) -> torch.Tensor | None:
        """Load a (height, width) tensor from `path`, or None if `path` is unset."""
        if path is None:
            return None
        return load_tensor_from_path(path, expected_ndim=2)

    @property
    def gain(self) -> torch.Tensor | None:
        """Get the gain tensor."""
        return self._load_2d(self.gain_path)

    @property
    def mask(self) -> torch.Tensor | None:
        """Get the mask tensor."""
        return self._load_2d(self.mask_path)

    @property
    def dark(self) -> torch.Tensor | None:
        """Get the dark tensor."""
        return self._load_2d(self.dark_path)
