"""Tests for prepare_movie module."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import pytest
import torch

from ripple.core.prepare_movie import (
    apply_dark,
    apply_gain,
    apply_mask,
    prepare_movie,
    remove_hot_pixels,
)


@pytest.fixture
def sample_movie():
    """Create a sample movie tensor for testing."""
    # Create a simple 3D movie: 5 frames of 32x32 pixels
    return torch.randn(5, 32, 32, dtype=torch.float32)


@pytest.fixture
def sample_gain_map():
    """Create a sample gain map for testing."""
    return torch.ones(32, 32, dtype=torch.float32) * 2.0


@pytest.fixture
def sample_dark_map():
    """Create a sample dark map for testing."""
    return torch.ones(32, 32, dtype=torch.float32) * 0.5


# Tests for apply_gain
def test_apply_gain_none(sample_movie):
    """Test apply_gain with None gain_map returns original movie."""
    result = apply_gain(sample_movie, None, gain_flip=0, gain_rot=0, multiply_gain=True)
    assert torch.allclose(result, sample_movie)


def test_apply_gain_multiply(sample_movie, sample_gain_map):
    """Test apply_gain with multiply_gain=True."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=0, gain_rot=0, multiply_gain=True
    )
    expected = sample_movie * sample_gain_map
    assert torch.allclose(result, expected)


def test_apply_gain_divide(sample_movie, sample_gain_map):
    """Test apply_gain with multiply_gain=False (divide)."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=0, gain_rot=0, multiply_gain=False
    )
    expected = sample_movie / sample_gain_map
    assert torch.allclose(result, expected)


def test_apply_gain_flip_y(sample_movie, sample_gain_map):
    """Test apply_gain with flipY (gain_flip=1)."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=1, gain_rot=0, multiply_gain=True
    )
    flipped_gain = sample_gain_map.flip(0)
    expected = sample_movie * flipped_gain
    assert torch.allclose(result, expected)


def test_apply_gain_flip_x(sample_movie, sample_gain_map):
    """Test apply_gain with flipX (gain_flip=2)."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=2, gain_rot=0, multiply_gain=True
    )
    flipped_gain = sample_gain_map.flip(1)
    expected = sample_movie * flipped_gain
    assert torch.allclose(result, expected)


def test_apply_gain_rotation_90(sample_movie, sample_gain_map):
    """Test apply_gain with 90 degree rotation (gain_rot=1)."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=0, gain_rot=1, multiply_gain=True
    )
    rotated_gain = torch.rot90(sample_gain_map, k=-1)
    expected = sample_movie * rotated_gain
    assert torch.allclose(result, expected)


def test_apply_gain_rotation_180(sample_movie, sample_gain_map):
    """Test apply_gain with 180 degree rotation (gain_rot=2)."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=0, gain_rot=2, multiply_gain=True
    )
    rotated_gain = torch.rot90(sample_gain_map, k=-2)
    expected = sample_movie * rotated_gain
    assert torch.allclose(result, expected)


def test_apply_gain_rotation_270(sample_movie, sample_gain_map):
    """Test apply_gain with 270 degree rotation (gain_rot=3)."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=0, gain_rot=3, multiply_gain=True
    )
    rotated_gain = torch.rot90(sample_gain_map, k=-3)
    expected = sample_movie * rotated_gain
    assert torch.allclose(result, expected)


def test_apply_gain_flip_and_rotation(sample_movie, sample_gain_map):
    """Test apply_gain with both flip and rotation."""
    result = apply_gain(
        sample_movie, sample_gain_map, gain_flip=1, gain_rot=1, multiply_gain=True
    )
    # Should flip first, then rotate
    transformed_gain = torch.rot90(sample_gain_map.flip(0), k=-1)
    expected = sample_movie * transformed_gain
    assert torch.allclose(result, expected)


# Tests for apply_dark
def test_apply_dark_none(sample_movie):
    """Test apply_dark with None dark_map returns original movie."""
    result = apply_dark(sample_movie, None)
    assert torch.allclose(result, sample_movie)


def test_apply_dark_basic(sample_movie, sample_dark_map):
    """Test apply_dark with a dark map."""
    result = apply_dark(sample_movie, sample_dark_map)
    expected = sample_movie - sample_dark_map
    assert torch.allclose(result, expected)


# Tests for remove_hot_pixels
def test_remove_hot_pixels_no_hot_pixels(sample_movie):
    """Test remove_hot_pixels with a movie that has no hot pixels."""
    # Use a very high threshold so no pixels are considered hot
    result = remove_hot_pixels(sample_movie, threshold=100.0)

    # Should return a tensor of the same shape
    assert result.shape == sample_movie.shape


def test_remove_hot_pixels_with_hot_pixels():
    """Test remove_hot_pixels with a movie that has hot pixels."""
    # Create a movie with a clear hot pixel
    movie = torch.zeros(2, 10, 10, dtype=torch.float32)
    # Add a hot pixel in the middle of frame 0
    movie[0, 5, 5] = 1000.0  # Very high value

    # Use a low threshold to catch the hot pixel
    result = remove_hot_pixels(movie, threshold=3.0)

    # The hot pixel should be replaced (not equal to 1000.0)
    assert result[0, 5, 5] != 1000.0
    assert result.shape == movie.shape


def test_remove_hot_pixels_preserves_shape():
    """Test that remove_hot_pixels preserves movie shape."""
    movie = torch.randn(3, 20, 20, dtype=torch.float32)
    result = remove_hot_pixels(movie, threshold=10.0)
    assert result.shape == movie.shape


# Tests for apply_mask
def test_apply_mask_none(sample_movie):
    """Test apply_mask with None mask returns original movie."""
    result = apply_mask(sample_movie, None)
    assert torch.allclose(result, sample_movie)


def test_apply_mask_basic(sample_movie):
    """Test apply_mask multiplies the mask uniformly into every frame."""
    mask = torch.zeros(32, 32, dtype=torch.float32)
    mask[8:24, 8:24] = 1.0

    result = apply_mask(sample_movie, mask)
    expected = sample_movie * mask
    assert torch.allclose(result, expected)

    # Every frame should be masked identically
    for frame_idx in range(result.shape[0]):
        assert torch.allclose(result[frame_idx], sample_movie[frame_idx] * mask)


def test_apply_mask_shape_mismatch(sample_movie):
    """Test apply_mask raises ValueError when mask shape doesn't match frames."""
    mask = torch.ones(16, 16, dtype=torch.float32)
    with pytest.raises(ValueError, match="mask shape"):
        apply_mask(sample_movie, mask)


def test_apply_mask_fill_noise_none_mask_is_noop(sample_movie):
    """Test fill_noise has no effect when mask is None."""
    result = apply_mask(sample_movie, None, fill_noise=True)
    assert torch.allclose(result, sample_movie)


def test_apply_mask_fill_noise_replaces_masked_pixels():
    """Test fill_noise=True replaces mask==0 pixels with per-frame Poisson noise."""
    torch.manual_seed(0)
    movie = torch.full((2, 40, 40), 5.0, dtype=torch.float32)
    mask = torch.zeros(40, 40, dtype=torch.float32)
    mask[10:30, 10:30] = 1.0  # keep central region, noise-fill everything else

    result = apply_mask(movie, mask, fill_noise=True)

    # In-mask pixels are left untouched
    assert torch.allclose(result[:, mask == 1], movie[:, mask == 1])

    # Out-of-mask pixels are replaced with samples, not the original constant
    outside = result[:, mask == 0]
    assert not torch.allclose(outside, movie[:, mask == 0])

    # Poisson noise should be centered near the central-region lambda (5.0)
    assert torch.abs(outside.mean() - 5.0) < 1.0


def test_apply_mask_fill_noise_lambda_clamped_nonnegative():
    """Test the Poisson rate is clamped to >= 1.0, so noise stays non-negative."""
    torch.manual_seed(1)
    movie = torch.zeros(4, 20, 20, dtype=torch.float32)
    mask = torch.ones(20, 20, dtype=torch.float32)
    mask[0:5, 0:5] = 0.0

    result = apply_mask(movie, mask, fill_noise=True)
    outside = result[:, mask == 0]
    assert torch.all(outside >= 0)


# Tests for prepare_movie (integration)
def test_prepare_movie_basic(sample_movie, sample_gain_map, sample_dark_map):
    """Test prepare_movie with all components."""
    result = prepare_movie(
        sample_movie,
        gain_map=sample_gain_map,
        dark_map=sample_dark_map,
        gain_flip=0,
        gain_rot=0,
        multiply_gain=True,
    )

    # Should have same shape
    assert result.shape == sample_movie.shape

    # Should have mean zero frames (last step in prepare_movie)
    for frame_idx in range(result.shape[0]):
        frame_mean = torch.mean(result[frame_idx])
        assert torch.abs(frame_mean) < 1e-5


def test_prepare_movie_with_mask(sample_movie, sample_gain_map, sample_dark_map):
    """Test prepare_movie applies a mask uniformly and stays mean-zero."""
    mask = torch.zeros(32, 32, dtype=torch.float32)
    mask[8:24, 8:24] = 1.0

    result = prepare_movie(
        sample_movie,
        gain_map=sample_gain_map,
        dark_map=sample_dark_map,
        gain_flip=0,
        gain_rot=0,
        multiply_gain=True,
        mask=mask,
    )

    assert result.shape == sample_movie.shape

    # Mask is applied before the final mean-zero step, so masked-out pixels are
    # driven to a uniform value (the negated frame mean) rather than staying at 0.
    for frame_idx in range(result.shape[0]):
        masked_out = result[frame_idx][mask == 0]
        assert torch.allclose(masked_out, masked_out[0].expand_as(masked_out))

        frame_mean = torch.mean(result[frame_idx])
        assert torch.abs(frame_mean) < 1e-5


def test_prepare_movie_with_mask_fill_noise(sample_gain_map, sample_dark_map):
    """Test prepare_movie forwards mask_fill_noise to noise-fill masked pixels."""
    torch.manual_seed(0)
    movie = torch.full((3, 32, 32), 5.0, dtype=torch.float32)
    mask = torch.zeros(32, 32, dtype=torch.float32)
    mask[8:24, 8:24] = 1.0

    result = prepare_movie(
        movie,
        gain_map=None,
        dark_map=None,
        gain_flip=0,
        gain_rot=0,
        multiply_gain=True,
        mask=mask,
        mask_fill_noise=True,
    )

    assert result.shape == movie.shape

    # Noise-filled pixels should not be uniform across a frame (unlike the
    # zero-fill case, where masked-out pixels collapse to a single value).
    for frame_idx in range(result.shape[0]):
        masked_out = result[frame_idx][mask == 0]
        assert masked_out.std() > 0


def test_prepare_movie_no_gain_no_dark(sample_movie):
    """Test prepare_movie with no gain or dark maps."""
    result = prepare_movie(
        sample_movie,
        gain_map=None,
        dark_map=None,
        gain_flip=0,
        gain_rot=0,
        multiply_gain=True,
    )

    # Should have same shape
    assert result.shape == sample_movie.shape

    # Should have mean zero frames
    for frame_idx in range(result.shape[0]):
        frame_mean = torch.mean(result[frame_idx])
        assert torch.abs(frame_mean) < 1e-5
