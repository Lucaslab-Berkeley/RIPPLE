"""Functions for preparing a movie for alignment."""

import torch
import torch.nn.functional as F

# Number of frames transferred to `device` and corrected at a time
DEFAULT_PREP_CHUNK_SIZE = 8

# 3x3 kernel that averages the 8-neighborhood of a pixel (center excluded)
_NEIGHBOR_AVERAGE_KERNEL = (
    torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]) / 8.0
)


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def prepare_movie(
    movie: torch.Tensor,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    multiply_gain: bool = True,
    device: torch.device | str | None = None,
    chunk_size: int = DEFAULT_PREP_CHUNK_SIZE,
) -> torch.Tensor:
    """Prepare the movie for alignment.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to prepare.
    gain_map: Optional[torch.Tensor]
        The gain map to apply to the movie. If None, the gain map will be
        initialized to zero.
    dark_map: Optional[torch.Tensor]
        The dark map to apply to the movie. If None, the dark map will be
        initialized to zero.
    gain_flip: int
        The flip to apply to the gain map.
    gain_rot: int
        The rotation to apply to the gain map.
    multiply_gain: bool = True
        Whether to multiply the movie by the gain map or divide the movie by the
        gain map.
    device: torch.device | str | None
        Device to prepare the movie on. If None, uses `movie`'s current device.
    chunk_size: int
        Number of frames to transfer/correct at a time.

    Returns
    -------
    torch.Tensor
    The prepared movie.
    """
    target_device = torch.device(device) if device is not None else movie.device

    n_frames = movie.shape[0]
    chunk_size = max(1, min(chunk_size, n_frames))
    prepared = torch.empty(movie.shape, dtype=torch.float32, device=target_device)

    # Iterate over each chunk of frames
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = movie[start:end].to(device=target_device, dtype=torch.float32)
        chunk = apply_gain(chunk, gain_map, gain_flip, gain_rot, multiply_gain)
        chunk = apply_dark(chunk, dark_map)
        chunk = remove_hot_pixels(chunk)
        prepared[start:end] = chunk

    # `prepared` is a buffer we own outright (not caller-supplied), so it is
    # safe to finish it off in place rather than allocating one more full copy.
    frame_means = torch.mean(prepared, dim=(1, 2), keepdim=True)
    prepared -= frame_means

    return prepared


def apply_gain(
    movie: torch.Tensor,
    gain_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    multiply_gain: bool = True,
) -> torch.Tensor:
    """
    Apply the gain map to the movie.

    Parameters
    ----------
    movie : torch.Tensor
        The movie to apply the gain map to.
    gain_map : torch.Tensor | None
        The gain map to apply to the movie. If None, returns the movie unchanged.
    gain_flip : int
        The flip to apply to the gain map.
        0: no flip
        1: flipY
        2: flipX
    gain_rot : int
        The rotation to apply to the gain map.
        0: no rotation
        1: 90 degrees
        2: 180 degrees
        3: 270 degrees
    multiply_gain : bool
        Whether to multiply the movie by the gain map or divide the movie by the
        gain map.

    Returns
    -------
    torch.Tensor
        The movie with the gain map applied, or the original movie if gain_map is None.
    """
    # If gain_map is None, return the movie unchanged
    if gain_map is None:
        return movie

    gain_map = gain_map.to(device=movie.device, dtype=movie.dtype)

    # Apply transformations to gain map
    if gain_flip == 1:
        gain_map = gain_map.flip(0)  # flipY
    elif gain_flip == 2:
        gain_map = gain_map.flip(1)  # flipX

    if gain_rot != 0:
        gain_map = torch.rot90(gain_map, k=-gain_rot)

    if multiply_gain:
        return movie * gain_map
    return movie / gain_map


def apply_dark(
    movie: torch.Tensor,
    dark_map: torch.Tensor | None,
) -> torch.Tensor:
    """
    Apply the dark map to the movie.

    Parameters
    ----------
    movie : torch.Tensor
        The movie to apply the dark map to.
    dark_map : torch.Tensor | None
        The dark map to apply to the movie. If None, returns the movie unchanged.

    Returns
    -------
    torch.Tensor
        The movie with the dark map applied, or the original movie if dark_map is None.
    """
    if dark_map is None:
        return movie
    return movie - dark_map.to(device=movie.device, dtype=movie.dtype)


def remove_hot_pixels(movie: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """Remove hot pixels from movie frames.

    Does so by replacing pixels that are more than `threshold` standard deviations
    above/below the mean (computed from the central 50% of each frame) with the average
    of their 8 neighboring pixels. Replacement vectorized into a single conv2d call.

    Args:
        movie: torch.Tensor
            Movie array with shape (n_frames, height, width)
        threshold: float
           Number of standard deviations above/below mean to consider as hot pixel

    Returns
    -------
        torch.Tensor
        The movie with hot pixels replaced.
    """
    _, height, width = movie.shape

    # Compute mean/std from the central 50% of each frame
    central = movie[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    frame_mean = torch.mean(central, dim=(1, 2), keepdim=True)
    frame_std = torch.std(central, dim=(1, 2), keepdim=True)

    hot_pixel_mask = (movie - frame_mean).abs() > threshold * frame_std
    if not bool(hot_pixel_mask.any()):
        return movie

    kernel = _NEIGHBOR_AVERAGE_KERNEL.to(device=movie.device, dtype=movie.dtype)
    neighbor_average = F.conv2d(
        movie.unsqueeze(1), kernel.view(1, 1, 3, 3), padding=1
    ).squeeze(1)

    return torch.where(hot_pixel_mask, neighbor_average, movie)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def prepare_core(
    movie: torch.Tensor,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    multiply_gain: bool,
    skip_movie_preparation: bool,
    device: torch.device | str | None = None,
    chunk_size: int = DEFAULT_PREP_CHUNK_SIZE,
) -> torch.Tensor:
    """
    Prepare the movie for core processing functions.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to prepare.
    gain_map: torch.Tensor | None
        The gain map to apply to the movie.
    dark_map: torch.Tensor | None
        The dark map to apply to the movie.
    gain_flip: int
        The flip to apply to the gain map.
    gain_rot: int
        The rotation to apply to the gain map.
    multiply_gain: bool
        Whether to multiply the movie by the gain map or divide the movie by the
        gain map.
    skip_movie_preparation: bool
        Whether to skip the movie preparation step.
    device: torch.device | str | None
        Device to prepare the movie on. If None, uses `movie`'s current device.
    chunk_size: int
        Number of frames to transfer/correct at a time.

    Returns
    -------
    torch.Tensor
        The prepared movie, or the original movie if skip_movie_preparation is True.
    """
    if not skip_movie_preparation:
        movie_prepared = prepare_movie(
            movie,
            gain_map,
            dark_map,
            gain_flip,
            gain_rot,
            multiply_gain,
            device=device,
            chunk_size=chunk_size,
        )
    else:
        movie_prepared = movie
    return movie_prepared
