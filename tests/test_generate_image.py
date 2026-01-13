"""Tests for generate_image module."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import pytest
import torch

from ripple.core.generate_image import generate_dose_weighted_image, sum_movie


@pytest.fixture
def sample_movie():
    """Create a sample movie tensor for testing."""
    # Create a simple 3D movie: 5 frames of 32x32 pixels
    return torch.randn(5, 32, 32, dtype=torch.float32)


def test_sum_movie_basic(sample_movie):
    """Test that sum_movie sums along the first dimension."""
    result = sum_movie(sample_movie)

    # Should have shape (32, 32) - spatial dimensions only
    assert result.shape == (32, 32)

    # Should be sum of all frames
    expected = torch.sum(sample_movie, dim=0)
    assert torch.allclose(result, expected)


def test_sum_movie_single_frame():
    """Test sum_movie with a single frame."""
    movie = torch.randn(1, 16, 16, dtype=torch.float32)
    result = sum_movie(movie)

    assert result.shape == (16, 16)
    assert torch.allclose(result, movie[0])


def test_sum_movie_preserves_dtype(sample_movie):
    """Test that sum_movie preserves dtype."""
    result = sum_movie(sample_movie)
    assert result.dtype == sample_movie.dtype


def test_sum_movie_zeros():
    """Test sum_movie with all zeros."""
    movie = torch.zeros(3, 10, 10, dtype=torch.float32)
    result = sum_movie(movie)

    assert torch.allclose(result, torch.zeros(10, 10))


def test_generate_dose_weighted_image_basic(sample_movie):
    """Test generate_dose_weighted_image produces correct output shape."""
    result = generate_dose_weighted_image(
        movie=sample_movie,
        pixel_size=1.0,
        pre_exposure=0.0,
        fluence_per_frame=1.0,
        voltage=300.0,
    )

    # Should produce a 2D image (summed over frames)
    assert result.shape == (32, 32)
    assert isinstance(result, torch.Tensor)


def test_generate_dose_weighted_image_different_parameters(sample_movie):
    """Test generate_dose_weighted_image with different parameters."""
    result = generate_dose_weighted_image(
        movie=sample_movie,
        pixel_size=0.5,
        pre_exposure=1.0,
        fluence_per_frame=2.0,
        voltage=200.0,
    )

    # Should still produce correct shape
    assert result.shape == (32, 32)


def test_generate_dose_weighted_image_single_frame():
    """Test generate_dose_weighted_image with a single frame."""
    movie = torch.randn(1, 16, 16, dtype=torch.float32)
    result = generate_dose_weighted_image(
        movie=movie,
        pixel_size=1.0,
        pre_exposure=0.0,
        fluence_per_frame=1.0,
        voltage=300.0,
    )

    assert result.shape == (16, 16)
