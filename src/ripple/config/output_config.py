"""Configuration for output files for RIPPLE."""

import os

from pydantic import model_validator
from typing_extensions import Self

from ripple.utils.custom_types import BaseModelRIPPLE


def check_file_path_and_permissions(path: str | None, allow_overwrite: bool) -> None:
    """Ensures path is writable and it does not exist, if `allow_overwrite` is False."""
    # If path is None, skip validation (no output file specified)
    if path is None:
        return

    # 1. Create path to file, if it does not exist
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # 2. Check write permissions
    if directory and not os.access(directory, os.W_OK):
        raise ValueError(
            f"Directory '{directory}' does not permit writing."
            f"Will be unable to write results to '{path}'."
        )

    # 3. Check if file exists
    if not allow_overwrite and os.path.exists(path):
        raise ValueError(
            f"File '{path}' already exists, but 'allow_file_overwrite' "
            "is False. Set 'allow_file_overwrite' to True to permit. "
            "overwriting.\n"
            "WARNING: Overwriting will delete the existing file(s)!"
        )


class OutputConfig(BaseModelRIPPLE):
    """Configuration for output files for RIPPLE."""

    allow_file_overwrite: bool = True
    dw_sum_output_path: str | None = None
    deformation_field_output_path: str | None = None
    motion_corrected_movie_output_path: str | None = None
    rendered_movie_output_path: str | None = None
    non_dw_sum_output_path: str | None = None
    loss_trajectories_output_path: str | None = None
    particle_shift_path: str | None = None

    @model_validator(mode="after")  # type: ignore
    def validate_paths(self) -> Self:
        """Validate output paths for write permissions and overwriting.

        Note: This method runs after instantiation, so attributes are already
        set. We can safely access them with `self`.

        Returns
        -------
        Self
            The validated instance.

        Raises
        ------
        ValueError
            If the output paths are not writable or do not permit overwriting.
        """
        # 1. Check write permissions and overwriting for each path
        paths = [
            self.dw_sum_output_path,
            self.deformation_field_output_path,
            self.motion_corrected_movie_output_path,
            self.rendered_movie_output_path,
            self.non_dw_sum_output_path,
            self.loss_trajectories_output_path,
            self.particle_shift_path,
        ]
        for path in paths:
            check_file_path_and_permissions(path, self.allow_file_overwrite)

        return self
