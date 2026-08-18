"""Additional type definitions and hints for Pydantic models."""

from typing import Annotated

import torch
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

# Pydantic type-hint to exclude tensor from JSON schema/dump (still attribute)
ExcludedTensor = SkipJsonSchema[
    Annotated[torch.Tensor | None, Field(default=None, exclude=True)]
]
