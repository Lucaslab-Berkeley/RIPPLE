"""Functions for preparing a movie for alignment."""

import torch
import torch.nn.functional as F

from ripple.core.crop_bounds import CropBounds, crop_movie

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
    mask: torch.Tensor | None = None,
    mask_fill_noise: bool = False,
    device: torch.device | str | None = None,
    storage_device: torch.device | str | None = None,
    chunk_size: int = DEFAULT_PREP_CHUNK_SIZE,
    crop_bounds: CropBounds | None = None,
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
        Flip applied to gain and dark reference maps (0 = none, 1 = flipY, 2 = flipX).
    gain_rot: int
        Rotation applied to gain and dark reference maps (0–3, CCW quarter turns).
    multiply_gain: bool = True
        Whether to multiply the movie by the gain map or divide the movie by the
        gain map.
    mask: torch.Tensor | None
        Mask with shape (height, width) applied uniformly to every frame (e.g. a beam
        aperture or defect mask). If None, no masking is applied and `mask_fill_noise`
        has no effect.
    mask_fill_noise: bool
        If True (and `mask` is provided), pixels where `mask == 0` are replaced with
        per-frame Poisson noise instead of being zeroed out. Ignored if `mask` is None.
    device: torch.device | str | None
        Device each chunk's gain/dark/hot-pixel/mask compute runs on. If None, uses
        `movie`'s current device.
    storage_device: torch.device | str | None
        Device the returned `prepared` movie is stored on. If None, defaults to
        `device` (or `movie`'s device if `device` is also None).
    chunk_size: int
        Number of frames to transfer/correct at a time.
    crop_bounds: CropBounds | None
        Inclusive ``(min_y, max_y, min_x, max_x)`` crop bounds applied to `movie`,
        `gain_map`, `dark_map`, and `mask` after any gain/dark reference flip/rotation.
        If None, no cropping is applied.

    Returns
    -------
    torch.Tensor
    The prepared movie.
    """
    prep_gain_flip = gain_flip
    prep_gain_rot = gain_rot
    if gain_flip != 0 or gain_rot != 0:
        if gain_map is not None:
            gain_map = transform_reference_map(gain_map, gain_flip, gain_rot)
        if dark_map is not None:
            dark_map = transform_reference_map(dark_map, gain_flip, gain_rot)
        prep_gain_flip = 0
        prep_gain_rot = 0

    if crop_bounds is not None:
        movie = crop_movie(movie, crop_bounds)
        if gain_map is not None:
            gain_map = crop_movie(gain_map, crop_bounds)
        if dark_map is not None:
            dark_map = crop_movie(dark_map, crop_bounds)
        if mask is not None:
            mask = crop_movie(mask, crop_bounds)

    compute_device = torch.device(device) if device is not None else movie.device
    target_device = (
        torch.device(storage_device) if storage_device is not None else compute_device
    )

    n_frames = movie.shape[0]
    chunk_size = max(1, min(chunk_size, n_frames))
    prepared = torch.empty(movie.shape, dtype=torch.float32, device=target_device)

    # Iterate over each chunk of frames
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = movie[start:end].to(device=compute_device, dtype=torch.float32)
        chunk = apply_gain(
            chunk, gain_map, prep_gain_flip, prep_gain_rot, multiply_gain
        )
        chunk = apply_dark(chunk, dark_map)
        chunk = remove_hot_pixels(chunk)
        chunk = apply_mask(chunk, mask, fill_noise=mask_fill_noise)
        frame_means = torch.mean(chunk, dim=(1, 2), keepdim=True)
        chunk -= frame_means
        prepared[start:end] = chunk.to(target_device)

    return prepared


def transform_reference_map(
    reference_map: torch.Tensor,
    gain_flip: int,
    gain_rot: int,
) -> torch.Tensor:
    """Flip and rotate a gain or dark reference so it aligns with the movie grid.

    Parameters
    ----------
    reference_map : torch.Tensor
        2-D reference map with shape ``(height, width)`` (gain or dark).
    gain_flip : int
        Flip to apply: 0 = none, 1 = flipY, 2 = flipX.
    gain_rot : int
        Rotation to apply: 0 = none, 1 = 90°, 2 = 180°, 3 = 270° (CCW).

    Returns
    -------
    torch.Tensor
        Transformed reference map, same dtype/device as the input.
    """
    if gain_flip == 1:
        reference_map = reference_map.flip(0)  # flipY
    elif gain_flip == 2:
        reference_map = reference_map.flip(1)  # flipX

    if gain_rot != 0:
        reference_map = torch.rot90(reference_map, k=-gain_rot)

    return reference_map


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
    if gain_map is None:
        return movie

    gain_map = gain_map.to(device=movie.device, dtype=movie.dtype)
    gain_map = transform_reference_map(gain_map, gain_flip, gain_rot)

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


def apply_mask(
    movie: torch.Tensor,
    mask: torch.Tensor | None,
    fill_noise: bool = False,
) -> torch.Tensor:
    """Apply a (height, width) mask uniformly to every frame of the movie.

    Parameters
    ----------
    movie : torch.Tensor
        Movie tensor with shape (n_frames, height, width).
    mask : torch.Tensor | None
        Mask with shape (height, width) to apply to every frame. If None,
        returns the movie unchanged (`fill_noise` has no effect without a mask).
    fill_noise : bool
        If True, replace pixels where `mask == 0` with per-frame Poisson noise
        instead of multiplying the movie by `mask`. Ignored if `mask` is None.

    Returns
    -------
    torch.Tensor
        The masked (or noise-filled) movie, or the original movie if mask is None.

    Raises
    ------
    ValueError
        If mask's shape does not match the movie's per-frame (height, width) shape.
    """
    if mask is None:
        return movie
    if mask.shape != movie.shape[-2:]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match movie's per-frame "
            f"shape {tuple(movie.shape[-2:])}"
        )
    mask = mask.to(device=movie.device, dtype=movie.dtype)

    if not fill_noise:
        return movie * mask
    return _fill_masked_noise(movie, mask)


def _fill_masked_noise(movie: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace pixels where `mask == 0` with per-frame Poisson noise."""
    _, height, width = movie.shape
    central = movie[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    frame_lambda = torch.clamp(torch.mean(central, dim=(1, 2), keepdim=True), min=1e-6)
    noise = torch.poisson(frame_lambda.expand_as(movie))

    outside_mask = (mask == 0).unsqueeze(0).expand_as(movie)
    return torch.where(outside_mask, noise, movie)
