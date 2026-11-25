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

# Number of output frames `eerfile.render` decodes at a time. Rendering all frames at
# once needs ~32 GB of RAM for a 4K movie with ~1800 EER frames; 9 frames at a time
# keeps the peak near 7.5 GB for about 1 s (~15%) more wall time.
DEFAULT_EER_RENDER_CHUNK_SIZE = 9

# File extensions (lower case) `load_array_from_path` reads, by loader
MRC_EXTENSIONS = (".mrc", ".mrcs")
TIFF_EXTENSIONS = (".tif", ".tiff", ".gain", ".dark")
DM4_EXTENSIONS = (".dm4",)

# File extensions (lower case) accepted as movies, including EER
MOVIE_EXTENSIONS = (".eer", ".mrc", ".mrcs", ".tif", ".tiff")

# Frames read at a time when only the sum of a movie's frames is needed. For TIFF
# this keeps tifffile's parallel decoding; for EER it is the number of EER frames
# (about 16 MB each at 4K once decoded).
DEFAULT_FRAME_SUM_CHUNK_SIZE = 8
DEFAULT_EER_FRAME_SUM_CHUNK_SIZE = 128


def render_eer_to_tensor(
    eer_path: str | os.PathLike | Path,
    fluence_per_frame: float,
    total_fluence: float,
    chunk_size: int | None = DEFAULT_EER_RENDER_CHUNK_SIZE,
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
    chunk_size : int | None
        Number of output frames decoded at a time. Lower values use less RAM. None
        decodes every EER frame at once (fastest, but needs tens of GB for a 4K movie).

    Returns
    -------
    torch.Tensor
        The rendered EER data as a uint16 tensor of electron counts.
    """
    movie_data = eerfile.render(
        eer_path,
        dose_per_output_frame=fluence_per_frame,
        total_fluence=total_fluence,
        chunk_size=chunk_size,
    )
    return _array_to_tensor(movie_data)


def sum_movie_frames(
    movie_path: str | os.PathLike | Path,
    chunk_size: int = DEFAULT_FRAME_SUM_CHUNK_SIZE,
    fluence_per_frame: float | None = None,
    total_fluence: float | None = None,
) -> torch.Tensor:
    """Sum a movie's frames, reading a few frames at a time.

    Gives the sum over frames of the movie as :class:`~ripple.config.MovieConfig`
    loads it (same orientation, so TIFF rows are flipped) without holding the whole
    movie in memory: peak memory is about one chunk of frames plus the sum.

    Parameters
    ----------
    movie_path : str | os.PathLike | Path
        Path to an EER, MRC/MRCS or TIFF movie.
    chunk_size : int
        Number of frames read at a time (MRC and TIFF). Default 8.
    fluence_per_frame : float | None
        EER only. With `total_fluence`, only the EER frames that rendering groups
        into whole output frames are summed, as :func:`render_eer_to_tensor` drops
        a final partial group. If either is None or 0, all EER frames are summed.
    total_fluence : float | None
        EER only; see `fluence_per_frame`.

    Returns
    -------
    torch.Tensor
        float32 ``(height, width)`` sum, on the CPU.
    """
    path_lower = str(movie_path).lower()
    if path_lower.endswith(".eer"):
        frame_sum = _sum_eer_frames(movie_path, fluence_per_frame, total_fluence)
    elif path_lower.endswith(MRC_EXTENSIONS):
        frame_sum = _sum_mrc_frames(movie_path, chunk_size)
    elif path_lower.endswith((".tif", ".tiff")):
        frame_sum = _sum_tiff_frames(movie_path, chunk_size)
    else:
        movie = load_array_from_path(movie_path, expected_ndim=3)
        frame_sum = movie.sum(axis=0, dtype=np.float32)
    return torch.from_numpy(frame_sum)


def _sum_tiff_frames(tif_path: str | os.PathLike | Path, chunk_size: int) -> np.ndarray:
    """Sum a TIFF movie's frames, decoding `chunk_size` pages at a time."""
    with TiffFile(tif_path) as tiff:
        n_pages = len(tiff.series[0].pages)
        frame_sum = None
        for start in range(0, n_pages, chunk_size):
            pages = range(start, min(start + chunk_size, n_pages))
            chunk = tiff.asarray(key=pages, series=0)
            chunk_sum = chunk.reshape(-1, *chunk.shape[-2:]).sum(
                axis=0, dtype=np.float32
            )
            frame_sum = chunk_sum if frame_sum is None else frame_sum + chunk_sum
    assert frame_sum is not None
    # Same orientation as the loaded movie (see _load_tiff_array)
    return np.ascontiguousarray(frame_sum[::-1])


def _sum_mrc_frames(mrc_path: str | os.PathLike | Path, chunk_size: int) -> np.ndarray:
    """Sum an MRC movie's frames, reading `chunk_size` frames at a time."""
    with mrcfile.mmap(mrc_path, mode="r") as mrc:
        frames = mrc.data.reshape(-1, *mrc.data.shape[-2:])
        frame_sum = np.zeros(frames.shape[-2:], dtype=np.float32)
        for start in range(0, frames.shape[0], chunk_size):
            frame_sum += frames[start : start + chunk_size].sum(
                axis=0, dtype=np.float32
            )
    return frame_sum


def _sum_eer_frames(
    eer_path: str | os.PathLike | Path,
    fluence_per_frame: float | None,
    total_fluence: float | None,
    chunk_size: int = DEFAULT_EER_FRAME_SUM_CHUNK_SIZE,
) -> np.ndarray:
    """Sum an EER movie's electron counts, decoding `chunk_size` EER frames at a time.

    EER frames are not flipped on read (see :meth:`ripple.config.MovieConfig.gain`).
    """
    with TiffFile(eer_path) as tiff:
        n_eer_frames = len(tiff.pages)
        n_summed = n_eer_frames
        if fluence_per_frame and total_fluence:
            # As eerfile.render: whole groups of EER frames make output frames
            dose_per_eer_frame = total_fluence / n_eer_frames
            per_output_frame = max(1, round(fluence_per_frame / dose_per_eer_frame))
            n_summed = (n_eer_frames // per_output_frame) * per_output_frame
        frame_sum = None
        for start in range(0, n_summed, chunk_size):
            chunk = tiff.asarray(key=range(start, min(start + chunk_size, n_summed)))
            chunk_sum = chunk.reshape(-1, *chunk.shape[-2:]).sum(
                axis=0, dtype=np.float32
            )
            frame_sum = chunk_sum if frame_sum is None else frame_sum + chunk_sum
    assert frame_sum is not None
    return frame_sum


def _array_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Wrap `array` as a tensor of the same dtype, without copying where possible.

    ``torch.from_numpy`` shares memory with `array` but needs a writable array in
    native byte order, so anything else (e.g. big-endian MRC data) is copied first.
    """
    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array)


def _load_tiff_array(tif_path: str | os.PathLike | Path) -> np.ndarray:
    """Read a TIFF file with its Y axis flipped, as IMOD, MotionCor2 and RELION do.

    TIFF stores the top row first and MRC the bottom row; those programs flip TIFF
    rows on read so both formats share one orientation. Doing the same means gain
    flip/rotation settings carry over from them, and sums of TIFF movies have the
    same orientation as theirs.
    """
    with TiffFile(tif_path) as tiff:
        array = tiff.asarray()
    # Flip one image at a time, in place, so the stack is never copied
    for image in array.reshape(-1, *array.shape[-2:]):
        image[:] = image[::-1].copy()
    return array


def _load_mrc_array(mrc_path: str | os.PathLike | Path) -> np.ndarray:
    array = mrcfile.read(mrc_path)
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
    pixel_size: float | None = None,
) -> None:
    """Writes a numpy array to an MRC file.

    NOTE: Writing header information, besides pixel/voxel size, is not currently
    implemented.

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
    pixel_size : float | None
        Isotropic pixel size, in Angstroms, for  MRC header's voxel size fields.
        If None (default), no voxel size is written.
    """
    if mrc_header is not None:
        raise NotImplementedError("Setting header info is not yet implemented.")

    with mrcfile.new(mrc_path, overwrite=overwrite) as mrc:
        mrc.set_data(data)
        if pixel_size is not None:
            mrc.voxel_size = pixel_size


def write_mrc_from_tensor(
    data: torch.Tensor,
    mrc_path: str | os.PathLike | Path,
    mrc_header: dict | None = None,
    overwrite: bool = False,
    pixel_size: float | None = None,
) -> None:
    """Writes a tensor array to an MRC file.

    NOTE: Writing header information, besides pixel/voxel size, is not currently
    implemented.

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
    pixel_size : float | None
        Isotropic pixel size, in Angstroms, for  MRC header's voxel size fields.
        If None (default), no voxel size is written.
    """
    write_mrc_from_numpy(data.numpy(), mrc_path, mrc_header, overwrite, pixel_size)


def load_array_from_path(
    file_path: str | os.PathLike | Path,
    expected_ndim: int | None = None,
    squeeze: bool = True,
) -> np.ndarray:
    """Load an array-like file into memory as a numpy array.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the file. Supported extensions (case-insensitive): ``.mrc``,
        ``.mrcs``, ``.tif``, ``.tiff``, ``.gain``, ``.dark``, ``.dm4``. TIFF files
        (including ``.gain`` and ``.dark``) are read with the Y axis flipped, as
        IMOD, MotionCor2 and RELION read them.
    expected_ndim : int | None
        Expected number of dimensions after optional squeezing. Default None
        skips shape validation.
    squeeze : bool
        Squeeze singleton dimensions before validating shape. Default True.

    Returns
    -------
    np.ndarray
        The loaded data. MRC and TIFF data keep the file's dtype (except float16
        MRC data, which is promoted to float32); DM4 data is float32.

    Raises
    ------
    ValueError
        If the file extension is not supported or the shape does not match.
    """
    path_str = str(file_path)
    path_lower = path_str.lower()

    if path_lower.endswith(MRC_EXTENSIONS):
        array = _load_mrc_array(path_str)
    elif path_lower.endswith(TIFF_EXTENSIONS):
        array = _load_tiff_array(path_str)
    elif path_lower.endswith(DM4_EXTENSIONS):
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
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """Load an array-like file into memory as a tensor.

    Parameters
    ----------
    file_path : str | os.PathLike | Path
        Path to the file; see :func:`load_array_from_path` for supported extensions.
    expected_ndim : int | None
        Expected number of dimensions after optional squeezing. Default None
        skips shape validation.
    squeeze : bool
        Squeeze singleton dimensions before validating shape. Default True.
    dtype : torch.dtype | None
        dtype of the returned tensor. Default float32. If None, the loaded dtype is
        kept (see :func:`load_array_from_path`) and the tensor shares memory with
        the loaded array, so e.g. an integer-count movie is not copied to float32.

    Returns
    -------
    torch.Tensor
        The loaded data.
    """
    array = load_array_from_path(
        file_path,
        expected_ndim=expected_ndim,
        squeeze=squeeze,
    )
    tensor = _array_to_tensor(array)
    return tensor if dtype is None else tensor.to(dtype)


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
