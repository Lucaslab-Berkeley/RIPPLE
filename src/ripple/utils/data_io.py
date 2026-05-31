"""Utility functions dealing with basic data I/O operations."""

import json
import os
from pathlib import Path
from typing import Any

import dm4
import eerfile
import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml
from tifffile import TiffFile
from torch_motion_correction.optimization_state import OptimizationTracker


def render_eer_to_tensor(
    eer_path: str | os.PathLike | Path,
    fluence_per_frame: float,
    total_fluence: float,
) -> torch.Tensor:
    """Renders an EER file to a tensor.

    Parameters
    ----------
    eer_path : str | os.PathLike | Path
        Path to the EER file.
    fluence_per_frame : float
        Fluence per frame in electrons per Angstrom^2/frame.
    total_fluence : float
        Total fluence in electrons per Angstrom^2.

    Returns
    -------
    torch.Tensor
        The rendered EER data as a tensor.
    """
    movie_data = eerfile.render(
        eer_path, dose_per_output_frame=fluence_per_frame, total_fluence=total_fluence
    )
    return torch.tensor(movie_data, dtype=torch.float32)


def _load_tiff_array(tif_path: str | os.PathLike | Path) -> np.ndarray:
    with TiffFile(tif_path) as tiff:
        tif_frames = tiff.asarray()
    return np.asarray(tif_frames, dtype=np.float32)


def _load_mrc_array(mrc_path: str | os.PathLike | Path) -> np.ndarray:
    array = mrcfile.read(mrc_path, permissive=True)
    # Convert float16 to float32 for FFT compatibility
    if array.dtype == np.float16:
        array = array.astype(np.float32)
    return array


def _load_dm4_array(dm4_path: str | os.PathLike | Path) -> np.ndarray:
    with dm4.DM4File.open(dm4_path) as dm4file:
        tags = dm4file.read_directory()

        image_data_tag = (
            tags.named_subdirs["ImageList"]
            .unnamed_subdirs[1]
            .named_subdirs["ImageData"]
        )
        image_tag = image_data_tag.named_tags["Data"]

        x_dim = dm4file.read_tag_data(
            image_data_tag.named_subdirs["Dimensions"].unnamed_tags[0]
        )
        y_dim = dm4file.read_tag_data(
            image_data_tag.named_subdirs["Dimensions"].unnamed_tags[1]
        )

        image_array = np.array(dm4file.read_tag_data(image_tag), dtype=np.float32)
        image_array = np.reshape(image_array, (y_dim, x_dim))

        return image_array


def write_mrc_from_numpy(
    data: np.ndarray,
    mrc_path: str | os.PathLike | Path,
    mrc_header: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Writes a numpy array to an MRC file.

    NOTE: Writing header information is not currently implemented.

    Parameters
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    overwrite : bool
        Overwrite argument passed to mrcfile.new. Default is False.
    """
    if mrc_header is not None:
        raise NotImplementedError("Setting header info is not yet implemented.")

    with mrcfile.new(mrc_path, overwrite=overwrite) as mrc:
        mrc.set_data(data)


def write_mrc_from_tensor(
    data: torch.Tensor,
    mrc_path: str | os.PathLike | Path,
    mrc_header: dict | None = None,
    overwrite: bool = False,
) -> None:
    """Writes a tensor array to an MRC file.

    NOTE: Not currently implemented.

    Parameters
    ----------
    data : np.ndarray
        The data to write to the MRC file.
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.
    mrc_header : Optional[dict]
        Dictionary containing header information. Default is None.
    overwrite : bool
        Overwrite argument passed to mrcfile.new. Default is False.
    """
    write_mrc_from_numpy(data.numpy(), mrc_path, mrc_header, overwrite)


def load_array_from_path(
    file_path: str | os.PathLike | Path,
    expected_ndim: int | None = None,
    squeeze: bool = True,
) -> np.ndarray:
    """Load an array-like file into memory as a numpy array.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the file. Supported extensions: ``.mrc``, ``.tif``, ``.tiff``,
        ``.gain``, ``.dark``, ``.dm4``.
    expected_ndim : int | None
        Expected number of dimensions after optional squeezing. Default None
        skips shape validation.
    squeeze : bool
        Squeeze singleton dimensions before validating shape. Default True.

    Returns
    -------
    np.ndarray
        The loaded data as a float32 array.

    Raises
    ------
    ValueError
        If the file extension is not supported or the shape does not match.
    """
    path_str = str(file_path)

    if path_str.endswith(".mrc"):
        array = _load_mrc_array(path_str)
    elif any(path_str.endswith(ext) for ext in (".tif", ".tiff", ".gain", ".dark")):
        array = _load_tiff_array(path_str)
    elif path_str.endswith(".dm4"):
        array = _load_dm4_array(path_str)
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")

    if squeeze:
        array = np.squeeze(array)
    if expected_ndim is not None and len(array.shape) != expected_ndim:
        raise ValueError(
            f"Unexpected array shape for {file_path}. Got shape: {array.shape}"
        )

    return array


def load_tensor_from_path(
    file_path: str | os.PathLike | Path,
    expected_ndim: int | None = None,
    squeeze: bool = True,
) -> torch.Tensor:
    """Load an array-like file into memory as a tensor."""
    array = load_array_from_path(
        file_path,
        expected_ndim=expected_ndim,
        squeeze=squeeze,
    )
    return torch.tensor(array, dtype=torch.float32)


def write_trajectory_to_csv(
    trajectory: OptimizationTracker,
    file_path: str | os.PathLike | Path,
) -> None:
    """Helper function for saving a trajectory to a CSV file.

    Parameters
    ----------
    trajectory : OptimizationTracker
        The trajectory to save. Must have a `checkpoints` attribute.
    file_path : str | os.PathLike | Path
        Path to the CSV file.

    Returns
    -------
    None
    """
    data = trajectory.as_dict()

    # Write the loss trajectory to a CSV file
    df = pd.DataFrame(data["optimization_checkpoints"])
    df.to_csv(file_path, index=False)


def load_template_volume_from_config(
    refine_config_path: str,
) -> torch.Tensor:
    """
    Load the template volume from the refine config YAML file.

    Parameters
    ----------
    refine_config_path : str
        Path to the refine config YAML file.

    Returns
    -------
    torch.Tensor
        The template volume as a float32 tensor.
    """
    # Load the config YAML
    with open(refine_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get the template volume path
    template_volume_path = config.get("template_volume_path")
    if template_volume_path is None:
        raise ValueError(
            f"template_volume_path not found in config file: {refine_config_path}"
        )

    # Resolve relative paths relative to current working directory
    if not Path(template_volume_path).is_absolute():
        template_volume_path = str(Path(template_volume_path).resolve())

    # Read MRC file and convert to float32 tensor
    template_volume = load_tensor_from_path(template_volume_path)

    return template_volume


# pylint: disable=too-many-arguments,too-many-positional-arguments
def save_optimize_sigmas_to_json(
    optimized_sigmas: dict[str, Any],
    sigma_history: list[dict[str, Any]],
    training_loss_history: list[float],
    validation_loss_history: list[float],
    optimized_sigmas_output_path: str | None = None,
    sigma_history_output_path: str | None = None,
    training_history_output_path: str | None = None,
    validation_history_output_path: str | None = None,
    verbose: bool = True,
) -> None:
    """Save sigma optimization results to JSON files.

    Parameters
    ----------
    optimized_sigmas : dict[str, Any]
        Dictionary of optimized sigma values
    sigma_history : list[dict[str, Any]]
        List of sigma values at each iteration/trial
    training_loss_history : list[float]
        List of training losses at each iteration/trial
    validation_loss_history : list[float]
        List of validation losses at each iteration/trial
    optimized_sigmas_output_path : str | None
        Path to save final optimized sigmas as JSON. Default None
    sigma_history_output_path : str | None
        Path to save sigma history as JSON. Default None
    training_history_output_path : str | None
        Path to save training loss history as JSON. Default None
    validation_history_output_path : str | None
        Path to save validation loss history as JSON. Default None
    verbose : bool
        Print messages when saving files. Default True
    """
    if optimized_sigmas_output_path is not None:
        with open(optimized_sigmas_output_path, "w", encoding="utf-8") as f:
            json.dump(optimized_sigmas, f, indent=2)
        if verbose:
            print(f"Saved optimized sigmas to: {optimized_sigmas_output_path}")

    if sigma_history_output_path is not None:
        with open(sigma_history_output_path, "w", encoding="utf-8") as f:
            json.dump(sigma_history, f, indent=2)
        if verbose:
            print(f"Saved sigma history to: {sigma_history_output_path}")

    if training_history_output_path is not None:
        with open(training_history_output_path, "w", encoding="utf-8") as f:
            json.dump(training_loss_history, f, indent=2)
        if verbose:
            print(f"Saved training history to: {training_history_output_path}")

    if validation_history_output_path is not None:
        with open(validation_history_output_path, "w", encoding="utf-8") as f:
            json.dump(validation_loss_history, f, indent=2)
        if verbose:
            print(f"Saved validation history to: {validation_history_output_path}")
