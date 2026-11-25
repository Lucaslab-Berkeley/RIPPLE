"""Manager for re-summing a movie from its saved deformation field."""

# prepare_and_correct_movie repeats the preparation call of manager_utils on purpose
# pylint: disable=duplicate-code

from pathlib import Path
from typing import ClassVar, Literal, Self

import torch
from pydantic import ConfigDict, model_validator
from teamtomo_basemodel import BaseModelTeamTomo
from torch_motion_correction import DeformationField, correct_motion

from ripple.config import ComputationalConfig, MovieConfig, OutputConfig

from . import manager_utils


def load_deformation_field(
    path: str | Path, grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom"
) -> DeformationField:
    """Load a deformation field saved by an alignment run (.csv, .h5 or .hdf5).

    Parameters
    ----------
    path : str | Path
        The field's file. HDF5 files carry their grid type; CSV files do not.
    grid_type : Literal["catmull_rom", "bspline"]
        Grid type of a CSV field: the ``grid_type`` the alignment used.

    Returns
    -------
    DeformationField
        The field.
    """
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        return DeformationField.from_hdf5(path)
    return DeformationField.from_csv(path, grid_type=grid_type)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def prepare_and_correct_movie(
    computational_config: ComputationalConfig,
    movie_config: MovieConfig,
    deformation_field: DeformationField,
    skip_movie_preparation: bool = False,
    movie: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prepare a movie as an alignment run does, then warp it with a saved field.

    Parameters
    ----------
    computational_config : ComputationalConfig
        Device for the compute; ``keep_movie_on_cpu`` keeps the prepared and
        corrected movies in CPU RAM and streams frames to the GPU, so
        super-resolution movies fit.
    movie_config : MovieConfig
        The movie and its preparation (gain, dark, mask, crop), as in the alignment
        run that estimated `deformation_field`.
    deformation_field : DeformationField
        The field the alignment run saved.
    skip_movie_preparation : bool
        As ``alignment_config.skip_movie_preparation`` in the alignment run: only
        cast the movie to float32.
    movie : torch.Tensor | None
        The raw movie, if already loaded. If None, read from ``movie_config``.

    Returns
    -------
    torch.Tensor
        The motion-corrected movie ``(n_frames, h, w)`` at ``movie_config.pixel_size``
        (not Fourier-cropped), on ``computational_config.movie_storage_device``.
    """
    device = computational_config.gpu_device
    movie, gain_map, dark_map, mask = manager_utils.load_missing_tensors(
        computational_config, movie_config, movie
    )
    if skip_movie_preparation:
        prepared = movie.to(
            device=computational_config.movie_storage_device, dtype=torch.float32
        )
    else:
        prepared = movie_config.prepare(
            movie,
            gain_map,
            dark_map,
            mask=mask,
            device=device,
            storage_device=computational_config.movie_storage_device,
        )
    del movie
    with torch.no_grad():
        return correct_motion(  # type: ignore[no-any-return, unused-ignore]
            image=prepared,
            deformation_field=deformation_field,
            pixel_spacing=movie_config.pixel_size,
            device=device,
        )


class ResumManager(BaseModelTeamTomo):
    """Re-sum a movie with new frame weights, from its saved deformation field.

    No motion is re-estimated: the movie is prepared as in the alignment run
    (``movie_config``), warped with the deformation field that run saved, and summed
    with ``output_config.dose_weighting`` (``gg``, ``default`` or a frame weight
    table). With ``gg`` and the alignment run's settings, the dose-weighted sum
    reproduces that run's ``dw_sum`` (up to the precision the field was saved at).

    Attributes
    ----------
    computational_config : ComputationalConfig
        Compute device. ``keep_movie_on_cpu: true`` streams frames to the GPU, so
        super-resolution movies fit.
    movie_config : MovieConfig
        The movie and its preparation, as in the alignment run.
    output_config : OutputConfig
        Sums to write (``dw_sum_output_path``, ``non_dw_sum_output_path``), the frame
        weights (``dose_weighting``), and optionally the motion-corrected movie
        (``motion_corrected_movie_output_path``). ``deformation_field_output_path``,
        ``loss_trajectories_output_path`` and ``rendered_movie_output_path`` must be
        unset: nothing is estimated, and the unaligned movie is not kept.
    deformation_field_path : str
        The field saved by the alignment run (.csv, .h5 or .hdf5).
    grid_type : Literal["catmull_rom", "bspline"]
        The alignment run's ``grid_type`` (CSV fields do not record it).
    skip_movie_preparation : bool
        The alignment run's ``skip_movie_preparation``.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)
    computational_config: ComputationalConfig = ComputationalConfig()
    movie_config: MovieConfig
    output_config: OutputConfig
    deformation_field_path: str
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom"
    skip_movie_preparation: bool = False

    @model_validator(mode="after")  # type: ignore
    def _check_outputs(self) -> Self:
        """Refuse outputs that only an alignment run produces."""
        if self.output_config.deformation_field_output_path is not None:
            raise ValueError(
                "resum does not estimate motion: unset "
                "output_config.deformation_field_output_path"
            )
        if self.output_config.loss_trajectories_output_path is not None:
            raise ValueError(
                "resum does not optimise anything: unset "
                "output_config.loss_trajectories_output_path"
            )
        if self.output_config.rendered_movie_output_path is not None:
            raise ValueError(
                "resum does not keep the unaligned movie: unset "
                "output_config.rendered_movie_output_path (render_movie_to_mrc.py "
                "writes it)"
            )
        return self

    @property
    def deformation_field(self) -> DeformationField:
        """The saved deformation field."""
        return load_deformation_field(self.deformation_field_path, self.grid_type)

    def corrected_movie(self, movie: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare the movie and warp it with the saved field.

        Parameters
        ----------
        movie : torch.Tensor | None
            The raw movie, if already loaded.

        Returns
        -------
        torch.Tensor
            The motion-corrected movie at the movie's own pixel size.
        """
        return prepare_and_correct_movie(
            self.computational_config,
            self.movie_config,
            self.deformation_field,
            self.skip_movie_preparation,
            movie=movie,
        )

    def run(self, movie: torch.Tensor | None = None) -> None:
        """Write the configured sums (and movies) from the saved field.

        Parameters
        ----------
        movie : torch.Tensor | None
            The raw movie, if already loaded. If None, read from ``movie_config``.
        """
        self.output_config.check_output_paths()
        corrected = self.corrected_movie(movie)
        manager_utils.save_results(
            self.output_config,
            self.movie_config,
            corrected,
            self.deformation_field,
            corrected,
            None,  # type: ignore[arg-type, unused-ignore]
            device=self.computational_config.gpu_device,
        )
