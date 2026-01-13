"""Functions for preparing a movie for alignment."""

import torch


# pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
def prepare_movie(
    movie: torch.Tensor,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    multiply_gain: bool = True,
) -> torch.Tensor:
    """
    Prepare the movie for alignment.

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

    Returns
    -------
    torch.Tensor
    The prepared movie.
    """
    movie = apply_gain(movie, gain_map, gain_flip, gain_rot, multiply_gain)
    movie = apply_dark(movie, dark_map)
    movie = remove_hot_pixels(movie)
    movie = set_frames_mean_zero(movie)

    return movie


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
    return movie - dark_map


# pylint: disable=too-many-locals,too-many-nested-blocks
def remove_hot_pixels(movie: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """
    Remove hot pixels from movie frames.

    Does so by replacing pixels that are more than
    `threshold` standard deviations above/below the mean
    with a random adjacent pixel value.

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
    print(f"Removing hot pixels with threshold {threshold} standard deviations...")

    movie_corrected = movie.clone()
    n_frames, height, width = movie.shape

    for frame_idx in range(n_frames):
        frame = movie_corrected[frame_idx]

        # Calculate mean and std for this frame
        # just to this for middle 50% of image
        frame_mean = torch.mean(
            frame[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        )
        frame_std = torch.std(
            frame[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
        )

        # Find hot pixels (pixels above OR below threshold * std from mean)
        hot_pixel_mask = (frame > (frame_mean + threshold * frame_std)) | (
            frame < (frame_mean - threshold * frame_std)
        )
        hot_pixel_coords = torch.where(hot_pixel_mask)

        if len(hot_pixel_coords[0]) > 0:
            print(f"  Frame {frame_idx}: Found {len(hot_pixel_coords[0])} hot pixels")

            # Replace each hot pixel with a random adjacent pixel
            for y, x in zip(hot_pixel_coords[0], hot_pixel_coords[1], strict=False):
                # Define the 8-connected neighborhood bounds
                y_min = max(0, y - 1)
                y_max = min(height - 1, y + 1)
                x_min = max(0, x - 1)
                x_max = min(width - 1, x + 1)

                # Get adjacent pixels (excluding the hot pixel itself)
                adjacent_pixels = []
                for adj_y in range(y_min, y_max + 1):
                    for adj_x in range(x_min, x_max + 1):
                        if adj_y != y or adj_x != x:  # Exclude the hot pixel itself
                            adjacent_pixels.append(frame[adj_y, adj_x])

                # Replace with random adjacent pixel value
                if adjacent_pixels:
                    replacement_value = torch.randint(0, len(adjacent_pixels), (1,))
                    movie_corrected[frame_idx, y, x] = replacement_value

    return movie_corrected


def set_frames_mean_zero(movie: torch.Tensor) -> torch.Tensor:
    """
    Set each frame in the movie to have mean zero.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to set the frames mean to zero.

    Returns
    -------
    torch.Tensor
        The movie with each frame having mean zero.
    """
    print("Setting each frame to mean zero (vectorized)...")

    # Calculate mean for each frame along the spatial dimensions (axis=(1,2))
    frame_means = torch.mean(movie, axis=(1, 2), keepdim=True)

    # Subtract the mean from each frame using broadcasting
    movie_mean_zero = movie - frame_means

    n_frames = movie.shape[0]
    print(f"  Completed mean zero correction for {n_frames} frames")

    return movie_mean_zero


def prepare_core(
    movie: torch.Tensor,
    gain_map: torch.Tensor | None,
    dark_map: torch.Tensor | None,
    gain_flip: int,
    gain_rot: int,
    multiply_gain: bool,
    skip_movie_preparation: bool,
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

    Returns
    -------
    torch.Tensor
        The prepared movie, or the original movie if skip_movie_preparation is True.
    """
    if not skip_movie_preparation:
        movie_prepared = prepare_movie(
            movie, gain_map, dark_map, gain_flip, gain_rot, multiply_gain
        )
    else:
        movie_prepared = movie
    return movie_prepared
