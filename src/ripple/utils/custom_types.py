"""Additional type definitions and hints for Pydantic models."""

import json
import os
from typing import Annotated, ClassVar

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

# Pydantic type-hint to exclude tensor from JSON schema/dump (still attribute)
ExcludedTensor = SkipJsonSchema[
    Annotated[torch.Tensor | None, Field(default=None, exclude=True)]
]


class BaseModelRIPPLE(BaseModel):
    """Implementation of a Pydantic BaseModel with additional, useful methods.

    Currently, only additional import/export methods are implemented and this
    class can effectively be treated as the `pydantic.BaseModel` class.

    Attributes
    ----------
    None

    Methods
    -------
    from_json(json_path: str | os.PathLike) -> BaseModelRIPPLE
        Load a BaseModelRIPPLE subclass from a serialized JSON file.
    from_yaml(yaml_path: str | os.PathLike) -> BaseModelRIPPLE
        Load a BaseModelRIPPLE subclass from a serialized YAML file.
    to_json(json_path: str | os.PathLike) -> None
        Serialize the BaseModelRIPPLE subclass to a JSON file.
    to_yaml(yaml_path: str | os.PathLike) -> None
        Serialize the BaseModelRIPPLE subclass to a YAML file.
    """

    model_config: ClassVar = ConfigDict(extra="forbid")

    #####################################
    ### Import/instantiation methods ###
    #####################################

    @classmethod
    def from_json(cls, json_path: str | os.PathLike) -> "BaseModelRIPPLE":
        """Load a BaseModelRIPPLE subclass from a serialized JSON file.

        Parameters
        ----------
        json_path : str | os.PathLike
            Path to the JSON file to load.

        Returns
        -------
        BaseModelRIPPLE
            Instance of the BaseModelRIPPLE subclass loaded from the JSON file.
        """
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str | os.PathLike) -> "BaseModelRIPPLE":
        """Load a BaseModelRIPPLE subclass from a serialized YAML file.

        Parameters
        ----------
        yaml_path : str | os.PathLike
            Path to the YAML file to load.

        Returns
        -------
        BaseModelRIPPLE
            Instance of the BaseModelRIPPLE subclass loaded from the YAML file.
        """
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    ####################################
    ### Export/serialization methods ###
    ####################################

    def to_json(self, json_path: str | os.PathLike) -> None:
        """Serialize the BaseModelRIPPLE to a JSON file.

        Parameters
        ----------
        json_path : str | os.PathLike
            Path to the JSON file to save.

        Returns
        -------
        None
        """
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f)

    def to_yaml(self, yaml_path: str | os.PathLike) -> None:
        """Serialize the BaseModelRIPPLE to a YAML file.

        Parameters
        ----------
        yaml_path : str | os.PathLike
            Path to the YAML file to save.

        Returns
        -------
        None
        """
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f)
