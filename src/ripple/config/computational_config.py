"""Computational configuration for RIPPLE."""

from typing import Annotated

import torch
from pydantic import BaseModel, Field, field_validator

# Type alias for non-negative integer
NonNegativeInt = Annotated[int, Field(ge=0)]


class ComputationalConfig(BaseModel):
    """Serialization of computational resources allocated for RIPPLE.

    Attributes
    ----------
    gpu_id : Optional[Union[int, str]]
        Field which specifies which GPU or CPU to use for computation.
        The following types of values are allowed:
        - A single integer, e.g. 0, which means to use GPU with ID 0.
        - A device specifier string, e.g. "cuda:0", which means to use GPU with ID 0.
        - The specific string "cpu" which means to use CPU.

        Note: Only a single GPU or CPU is supported (no multiprocessing).
    keep_movie_on_cpu : bool
        When True, the prepared movie and output frames stored on CPU RAM instead of
        `gpu_device`. Individual steps (XC pre-pass, local optimizer patch precompute,
        final correction) then stream frames to the GPU. Default is False.
    """

    # Type-hinting here ensures only single GPU or CPU (no lists allowed)
    gpu_id: str | NonNegativeInt | None = 0
    keep_movie_on_cpu: bool = False

    @field_validator("gpu_id")  # type: ignore[misc]
    @classmethod
    def validate_gpu_id(cls, v: str | int | None) -> str | int | None:
        """Validate that gpu_id is a single GPU or CPU (no lists or 'all').

        Parameters
        ----------
        v : Union[str, int]
            The gpu_id value to validate.

        Returns
        -------
        Union[str, int]
            The validated gpu_id value.

        Raises
        ------
        ValueError
            If a list or 'all' is provided, or if the string is not 'cpu' or a
            valid cuda device.
        """
        # Reject lists (though type system should catch this)
        if isinstance(v, list):
            raise ValueError(
                f"Lists are not allowed for gpu_id. Got {v}. "
                "Only a single GPU (int or 'cuda:X') or 'cpu' is supported."
            )

        # Reject 'all' option
        if v == "all":
            raise ValueError(
                "The 'all' option is not supported. "
                "Please specify a single GPU (int or 'cuda:X') or 'cpu'."
            )

        # Validate string format (should be 'cpu' or 'cuda:X')
        if isinstance(v, str) and v != "cpu" and not v.startswith("cuda:"):
            raise ValueError(
                f"Invalid string value for gpu_id: {v}. "
                "Expected 'cpu' or a cuda device specifier like 'cuda:0'."
            )

        return v

    @property
    def gpu_device(self) -> torch.device:
        """Convert requested GPU ID or CPU to a torch device object.

        Returns
        -------
        torch.device
            A single device (either GPU or CPU).
        """
        # Handle CPU case
        if self.gpu_id == "cpu":
            return torch.device("cpu")

        # Handle single GPU case
        if isinstance(self.gpu_id, int):
            return torch.device(f"cuda:{self.gpu_id}")
        if isinstance(self.gpu_id, str):
            # String should be a device specifier like "cuda:0"
            return torch.device(self.gpu_id)
        raise TypeError(
            f"Invalid type for gpu_id: {type(self.gpu_id)}. "
            "Expected int, str, or 'cpu'."
        )

    @property
    def movie_storage_device(self) -> torch.device:
        """Device the persistent, full-resolution prepared movie is stored on.

        Returns
        -------
        torch.device
            CPU if `keep_movie_on_cpu` is True, otherwise `gpu_device`.
        """
        return torch.device("cpu") if self.keep_movie_on_cpu else self.gpu_device
