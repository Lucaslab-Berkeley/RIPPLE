"""Tests for prepare_movie module."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import pytest
import torch

from ripple.core.prepare_movie import (
    apply_dark,
    apply_gain,
    prepare_core,
    prepare_movie,
    remove_hot_pixels,
    set_frames_mean_zero,
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


# Tests for set_frames_mean_zero
def test_set_frames_mean_zero(sample_movie):
    """Test that set_frames_mean_zero makes each frame have mean zero."""
    result = set_frames_mean_zero(sample_movie)

    # Check that result has same shape
    assert result.shape == sample_movie.shape

    # Check that each frame has mean approximately zero
    for frame_idx in range(result.shape[0]):
        frame_mean = torch.mean(result[frame_idx])
        assert torch.abs(frame_mean) < 1e-5, (
            f"Frame {frame_idx} mean is not zero: {frame_mean}"
        )


def test_set_frames_mean_zero_preserves_differences(sample_movie):
    """Test that set_frames_mean_zero preserves relative differences within frames."""
    result = set_frames_mean_zero(sample_movie)

    # The difference between any two pixels in a frame should be preserved
    # (just shifted by a constant)
    for frame_idx in range(sample_movie.shape[0]):
        original_frame = sample_movie[frame_idx]
        result_frame = result[frame_idx]

        # Difference between first two pixels should be the same
        diff_original = original_frame[0, 0] - original_frame[0, 1]
        diff_result = result_frame[0, 0] - result_frame[0, 1]
        assert torch.allclose(diff_original, diff_result)


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


# Tests for prepare_core
def test_prepare_core_skip_preparation(sample_movie):
    """Test prepare_core with skip_movie_preparation=True."""
    result = prepare_core(
        sample_movie,
        gain_map=None,
        dark_map=None,
        gain_flip=0,
        gain_rot=0,
        multiply_gain=True,
        skip_movie_preparation=True,
    )

    # Should return original movie unchanged
    assert torch.allclose(result, sample_movie)


def test_prepare_core_with_preparation(sample_movie, sample_gain_map):
    """Test prepare_core with skip_movie_preparation=False."""
    result = prepare_core(
        sample_movie,
        gain_map=sample_gain_map,
        dark_map=None,
        gain_flip=0,
        gain_rot=0,
        multiply_gain=True,
        skip_movie_preparation=False,
    )

    # Should have same shape
    assert result.shape == sample_movie.shape

    # Should have mean zero frames (from prepare_movie)
    for frame_idx in range(result.shape[0]):
        frame_mean = torch.mean(result[frame_idx])
        assert torch.abs(frame_mean) < 1e-5
