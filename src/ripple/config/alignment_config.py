"""Configuration for alignment of frames of a cryo-EM movie."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from torch_motion_correction import (
    DeformationField,
    FourierFilterConfig,
    PatchSamplingConfig,
)
from torch_motion_correction import OptimizationConfig as MotionOptimizationConfig

from ripple.utils.custom_types import BaseModelRIPPLE

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
        Resolution of the deformation field in pixels (nt, nh, nw).
    deformation_field_path: Optional[str]
        Path to the deformation field CSV file. If None, the backend initialises
        shifts to zero.
    max_iterations: int
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
    early_stopping: bool
        Whether to enable plateau-style early stopping. Default is False.
    early_stopping_patience: int
        Steps without improvement before stopping. Default is 5.
    early_stopping_window_size: int
        Number of recent loss values averaged for smoothing. Default is 3.
    early_stopping_tolerance: float
        Minimum relative improvement to reset the patience counter. Default is 1e-5.
    """

    deformation_field_resolution: tuple[PositiveInt, PositiveInt, PositiveInt]
    deformation_field_path: str | None = None  # .csv or .hdf5/.h5
    max_iterations: PositiveInt = 100
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom"
    optimizer_type: Literal["adam", "lbfgs"] = "adam"
    learning_rate: float = 0.2
    skip_movie_preparation: bool = False
    early_stopping: bool = False
    early_stopping_patience: PositiveInt = 5
    early_stopping_window_size: PositiveInt = 3
    early_stopping_tolerance: float = 1e-5

    @property
    def initial_deformation_field(self) -> DeformationField | None:
        """Load and wrap the saved deformation field, or None to start from zero.

        Returns
        -------
        DeformationField | None
            - If no deformation_field_path is set, returns None to indicate to backend
            that shifts should be initialized to zero.
            - If a path is set, loads the tensor and wrapped into a
            :class:`~torch_motion_correction.DeformationField` so that the ``grid_type``
            travels with the data.
        """
        if self.deformation_field_path is None:
            return None

        path = Path(self.deformation_field_path)
        if path.suffix in (".h5", ".hdf5"):
            return DeformationField.from_hdf5(path)
        return DeformationField.from_csv(path, grid_type=self.grid_type)

    @property
    def as_optimization_config(self) -> MotionOptimizationConfig:
        """Build a :class:`~torch_motion_correction.OptimizationConfig`.

        Returns
        -------
        MotionOptimizationConfig
            The optimization config object to be passed to the motion correction
            backend.
        """
        return MotionOptimizationConfig(
            max_iterations=self.max_iterations,
            optimizer_type=self.optimizer_type,
            grid_type=self.grid_type,
            optimizer_kwargs={"lr": self.learning_rate},
            early_stopping=self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_window_size=self.early_stopping_window_size,
            early_stopping_tolerance=self.early_stopping_tolerance,
        )


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
    use_xc_prepass: bool
        Whether to run a fast cross-correlation pre-pass to estimate global
        per-frame shifts before the gradient-based optimization. The XC
        shifts seed the initial deformation field so the optimizer starts
        from a good global motion estimate. Default is True.
    """

    patch_shape: tuple[PositiveInt, PositiveInt] = (1024, 1024)
    loss_type: Literal["mse", "cc", "ncc"] = "mse"
    b_factor: float = 500
    frequency_range: tuple[PositiveFloat, PositiveFloat] = (300, 10)
    use_xc_prepass: bool = True

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

    # --- Backend config accessors --------------------------------------------

    @property
    def as_patch_sampling_config(self) -> PatchSamplingConfig:
        """Build a :class:`~torch_motion_correction.PatchSamplingConfig`."""
        return PatchSamplingConfig(patch_shape=self.patch_shape)

    @property
    def as_fourier_filter_config(self) -> FourierFilterConfig:
        """Build a :class:`~torch_motion_correction.FourierFilterConfig`."""
        return FourierFilterConfig(
            b_factor=self.b_factor,
            frequency_range=self.frequency_range,
        )

    @property
    def as_optimization_config(self) -> MotionOptimizationConfig:
        """Build a :class:`~torch_motion_correction.OptimizationConfig`.

        Extends the base implementation to include ``loss_type``.
        """
        return MotionOptimizationConfig(
            max_iterations=self.max_iterations,
            optimizer_type=self.optimizer_type,
            loss_type=self.loss_type,
            grid_type=self.grid_type,
            optimizer_kwargs={"lr": self.learning_rate},
            early_stopping=self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_window_size=self.early_stopping_window_size,
            early_stopping_tolerance=self.early_stopping_tolerance,
        )


class PriorConfig(BaseModelRIPPLE):
    """Configuration for motion priors.

    Parameters
    ----------
    prior_type: str
        Type of prior to use: "relion" or "laplacian". Default is "relion".
    sigma_A_exponential: bool
        Whether to use exponential decay for sigma_A over frames. Default is False.
    init_sigma_A: float
        sigma_A (temporal smoothness). Default is 0.513517.
    init_alpha_spatial: float
        alpha_spatial (spatial smoothness strength for Laplacian prior).
        Default is 1e5.
    init_sigma_D: float
        sigma_D (spatial correlation length in Angstroms for RELION prior).
        Default is 5782.376953.
    init_sigma_V: float
        sigma_V (velocity magnitude scale in Å per unit fluence for
        RELION prior). Default is 0.194826.
    init_sigma_A_amplitude: float
        Amplitude A in exponential sigma_A formula: A*exp(-B*fluence) + C.
        Default is 2.0.
    init_sigma_A_decay: float
        Decay rate B in exponential sigma_A formula: A*exp(-B*fluence) + C.
        Default is 0.1.
    init_sigma_A_offset: float
        Constant offset C in exponential sigma_A formula: A*exp(-B*fluence) + C.
        Default is 1.0.
    """

    prior_type: str = "relion"
    sigma_a_exponential: bool = False
    init_sigma_a: float = 0.513517
    init_alpha_spatial: float = 1e5
    init_sigma_d: float = 5782.376953
    init_sigma_v: float = 0.194826
    init_sigma_a_amplitude: float = 2.0
    init_sigma_a_decay: float = 0.1
    init_sigma_a_offset: float = 1.0


class OptimizationConfig(BaseModelRIPPLE):
    """Configuration for sigma optimization.

    Parameters
    ----------
    enabled: bool
        Whether to enable sigma optimization. Default is False.
    optimize_algorithm: Literal["nelder-mead", "bayesian"]
        Algorithm to use for sigma optimization. Options are:
        - 'nelder-mead': Nelder-Mead (simplex) method
        - 'bayesian': Bayesian optimization using Optuna
        Default is 'bayesian'.
    optimize_particle_df_path: str | None
        Path to particle dataframe config for validation loss computation.
        The validation template will be loaded from the template_volume_path
        in this YAML file. If None, uses the same particle dataframe and
        template as the motion loop.
        Default is None.
    sigma_iterations: int
        Number of outer loop iterations for sigma optimization. Default is 50.
    motion_iterations: int
        Number of inner loop motion iterations per sigma update. Default is 20.
    optimized_sigmas_output_path: str | None
        Path to save optimized sigma values. Default is None.
    sigma_history_output_path: str | None
        Path to save sigma optimization history. Default is None.
    training_history_output_path: str | None
        Path to save training history. Default is None.
    validation_history_output_path: str | None
        Path to save validation history. Default is None.
    """

    enabled: bool = False
    optimize_algorithm: Literal["nelder-mead", "bayesian"] = "bayesian"
    optimize_particle_df_path: str | None = None
    sigma_iterations: PositiveInt = 50
    motion_iterations: PositiveInt = 20
    optimized_sigmas_output_path: str | None = None
    sigma_history_output_path: str | None = None
    training_history_output_path: str | None = None
    validation_history_output_path: str | None = None


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
    prior_config: PriorConfig
        Configuration for motion priors. Defaults to PriorConfig().
    optimization_config: OptimizationConfig | None
        Configuration for sigma optimization. If None, optimization is disabled.
        Default is None.
    """

    particle_df_path: str
    loss_metric: Literal["mip", "scaled_mip"] = "scaled_mip"
    min_snr: float = 0.0
    best_n: PositiveInt = 10000000000

    # Nested configs with defaults
    prior_config: PriorConfig = Field(default_factory=PriorConfig)
    optimization_config: OptimizationConfig | None = None

    @property
    def optimize_sigmas(self) -> bool:
        """Whether sigma optimization is enabled."""
        return self.optimization_config is not None
