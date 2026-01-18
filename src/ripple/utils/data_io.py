"""Utility functions dealing with basic data I/O operations."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import eerfile
import mrcfile
import numpy as np
import pandas as pd
import torch
import yaml
from tifffile import TiffFile
from torch_motion_correction import (
    read_deformation_field_from_csv,
    write_deformation_field_to_csv,
)

if TYPE_CHECKING:
    from torch_motion_correction import OptimizationTracker


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


def read_tif_to_tensor(tif_path: str | os.PathLike | Path) -> torch.Tensor:
    """Reads a TIFF file and returns the data as a tensor.

    Parameters
    ----------
    tif_path : str | os.PathLike | Path
        Path to the TIFF file.

    Returns
    -------
    torch.Tensor
        The TIFF data as a tensor, copied and converted to float32 if needed.
    """
    with TiffFile(tif_path) as tiff:
        tif_frames = tiff.asarray()
    return torch.tensor(tif_frames, dtype=torch.float32)


def read_mrc_to_numpy(mrc_path: str | os.PathLike | Path) -> np.ndarray:
    """Reads an MRC file and returns the data as a numpy array.

    Parameters
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    np.ndarray
        The MRC data as a numpy array, copied.
    """
    with mrcfile.open(mrc_path) as mrc:
        return mrc.data.copy()


def read_mrc_to_tensor(mrc_path: str | os.PathLike | Path) -> torch.Tensor:
    """Reads an MRC file and returns the data as a torch tensor.

    Parameters
    ----------
    mrc_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC data as a tensor, copied and converted to float32 if needed.
    """
    tensor = torch.tensor(read_mrc_to_numpy(mrc_path))
    # Convert float16 to float32 for FFT compatibility
    if tensor.dtype == torch.float16:
        tensor = tensor.to(torch.float32)
    return tensor


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


def load_mrc_image(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an two-dimensional MRC image into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC image as a tensor, converted to float32 for FFT compatibility.

    Raises
    ------
    ValueError
        If the MRC file is not two-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 2D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 2:
        raise ValueError(f"MRC file is not two-dimensional. Got shape: {tensor.shape}")

    return tensor


def load_mrc_movie(file_path: str | os.PathLike | Path) -> torch.Tensor:
    """Helper function for loading an three-dimensional MRC movie into a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the MRC file.

    Returns
    -------
    torch.Tensor
        The MRC movie as a tensor, converted to float32 for FFT compatibility.

    Raises
    ------
    ValueError
        If the MRC file is not three-dimensional.
    """
    tensor = read_mrc_to_tensor(file_path)

    # Check that tensor is 3D, squeezing if necessary
    tensor = tensor.squeeze()
    if len(tensor.shape) != 3:
        raise ValueError(
            f"MRC file is not three-dimensional. Got shape: {tensor.shape}"
        )

    return tensor


def load_deformation_field(
    file_path: str | os.PathLike | Path,
) -> torch.Tensor:
    """Helper function for loading a deformation field from a CSV file.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the CSV file.

    Returns
    -------
    torch.Tensor
        The deformation field as a tensor.
    """
    return read_deformation_field_from_csv(file_path)


def load_particle_shifts(
    file_path: str | os.PathLike | Path,
    n_frames: int,
    n_particles: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load particle shifts from CSV file.

    Parameters
    ----------
    file_path: str | os.PathLike | Path
        Path to CSV file with columns: particle_index, frame, y_shift, x_shift.
    n_frames: int
        Number of frames in the movie.
    n_particles: int
        Number of particles.
    device: torch.device | None
        Device to load tensor on. If None, uses CPU.

    Returns
    -------
    torch.Tensor
        Particle shifts tensor with shape (T, N, 2) where T is number of frames
        and N is number of particles. The last dimension is (y_shift, x_shift).
    """
    df = pd.read_csv(file_path)
    required_columns = ["particle_index", "frame", "y_shift", "x_shift"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV file must contain columns: {required_columns}. "
            f"Found: {list(df.columns)}"
        )

    # Initialize tensor with zeros
    particle_shifts = torch.zeros((n_frames, n_particles, 2), dtype=torch.float32)

    # Fill in shifts from DataFrame
    for _, row in df.iterrows():
        particle_idx = int(row["particle_index"])
        frame_idx = int(row["frame"])
        y_shift = float(row["y_shift"])
        x_shift = float(row["x_shift"])

        if particle_idx >= n_particles:
            raise ValueError(
                f"Particle index {particle_idx} exceeds number of particles {n_particles}"
            )
        if frame_idx >= n_frames:
            raise ValueError(
                f"Frame index {frame_idx} exceeds number of frames {n_frames}"
            )

        particle_shifts[frame_idx, particle_idx, 0] = y_shift
        particle_shifts[frame_idx, particle_idx, 1] = x_shift

    if device is not None:
        particle_shifts = particle_shifts.to(device)

    return particle_shifts


def save_deformation_field(
    deformation_field: torch.Tensor,
    file_path: str | os.PathLike | Path,
) -> None:
    """Helper function for saving a deformation field to a CSV file.

    Parameters
    ----------
    deformation_field : torch.Tensor
        The deformation field to save.
    file_path : str | os.PathLike | Path
        Path to the CSV file.

    Returns
    -------
    None
    """
    write_deformation_field_to_csv(deformation_field, file_path)


def write_trajectory_to_csv(
    trajectory: "OptimizationTracker",
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
    df = pd.DataFrame(
        [{"step": cp.step, "loss": cp.loss} for cp in trajectory.checkpoints]
    )
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
    template_volume = read_mrc_to_tensor(template_volume_path)

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
