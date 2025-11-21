"""Configuration for alignment of frames of a cryo-EM movie."""

from typing import Annotated, Literal

import torch
from pydantic import Field, field_validator

from ripple.utils.custom_types import BaseModelRIPPLE
from ripple.utils.data_io import load_deformation_field

# Type alias for positive integer
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]


class BaseAlignmentConfig(BaseModelRIPPLE):
    """Base configuration for alignment operations.

    This class contains common parameters shared between different alignment
    operations (e.g., align_frames and polish_particles).

    Parameters
    ----------
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field in pixels (x, y, z).
    deformation_field_path: Optional[str]
        Path to the deformation field file. If None, the deformation field will be
        initialized to zero.
    n_iterations: int
        Number of optimization iterations. Default is 100.
    grid_type: Literal["catmull_rom", "bspline"]
        Type of interpolation grid. Must be 'catmull_rom' or 'bspline'.
        Default is 'catmull_rom'.
    optimizer_type: Literal["adam", "lbfgs"]
        Type of optimizer. Must be 'adam' or 'lbfgs'. Default is 'adam'.
    learning_rate: float
        Learning rate for optimization. Default is 0.2.
    skip_movie_preparation: bool
        Whether to skip the movie preparation step. Default is False.
    """

    deformation_field_resolution: tuple[PositiveInt, PositiveInt, PositiveInt]
    deformation_field_path: str | None = None
    n_iterations: PositiveInt = 100
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom"
    optimizer_type: Literal["adam", "lbfgs"] = "adam"
    learning_rate: float = 0.2
    skip_movie_preparation: bool = False

    @property
    def deformation_field(self) -> torch.Tensor:
        """Get the deformation field tensor."""
        if self.deformation_field_path is None:
            return torch.zeros(self.deformation_field_resolution, dtype=torch.float32)
        return load_deformation_field(self.deformation_field_path)


class AlignFramesConfig(BaseAlignmentConfig):
    """Configuration for alignment of frames of a cryo-EM movie.

    This extends BaseAlignmentConfig with parameters specific to frame alignment.

    Parameters
    ----------
    patch_shape: tuple[int, int]
        Shape of the patch in pixels (width, height). Default is (1024, 1024).
    loss_type: Literal["mse", "cc", "ncc"]
        Type of loss function. Must be 'mse' (mean square error), 'cc'
        (cross correlation), or 'ncc' (normalized cross correlation).
        Default is 'mse'.
    b_factor: float
        B-factor for filtering. Default is 500.
    frequency_range: tuple[float, float]
        Frequency range for filtering in Angstroms. First value must be
        larger than the second value. Default is (300, 10).
    """

    patch_shape: tuple[PositiveInt, PositiveInt] = (1024, 1024)
    loss_type: Literal["mse", "cc", "ncc"] = "mse"
    b_factor: float = 500
    frequency_range: tuple[PositiveFloat, PositiveFloat] = (300, 10)

    @field_validator("frequency_range")  # type: ignore[misc]
    @classmethod
    def validate_frequency_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate that the frequency_range has exactly 2 elements.

        Also validates that the first value is larger than the second.

        Parameters
        ----------
        v: tuple[float, float]
            The frequency_range tuple to validate.

        Returns
        -------
        tuple[float, float]
            The validated frequency_range tuple.

        Raises
        ------
        ValueError
            If frequency_range does not have exactly 2 elements or if the first
            value is not larger than the second.
        """
        if len(v) != 2:
            raise ValueError(
                f"frequency_range must have exactly 2 elements, got {len(v)}"
            )
        if v[0] <= v[1]:
            raise ValueError(
                f"frequency_range first value must be larger than second, "
                f"got {v[0]} <= {v[1]}"
            )
        return v


class PolishParticlesConfig(BaseAlignmentConfig):
    """Configuration for polishing particles.

    This extends BaseAlignmentConfig with parameters specific to particle polishing.

    Parameters
    ----------
    particle_df_path: str
        Path to the refine config file.
    loss_metric: Literal["mip", "scaled_mip"]
        Metric to use for particle quality filtering. Must be 'mip' or 'scaled_mip'.
        Default is 'scaled_mip'.
    min_snr: float
        Minimum value of the loss_metric for a particle to be considered.
        Particles below this threshold will be excluded. Default is 0.
    best_n: int
        Maximum number of particles to use for optimization, selecting the top N
        particles with the highest loss_metric values. Default is 10000000000
        (essentially unlimited).
    """

    particle_df_path: str
    loss_metric: Literal["mip", "scaled_mip"] = "scaled_mip"
    min_snr: float = 0.0
    best_n: PositiveInt = 10000000000
