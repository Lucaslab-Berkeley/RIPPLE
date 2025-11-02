"""Tests for ripple."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import pytest
import torch

import ripple
from ripple.core.core_align_frames import core_align_frames


def test_imports_with_version():
    """Test that ripple can be imported and has a version."""
    assert isinstance(ripple.__version__, str)


@pytest.fixture
def sample_movie():
    """Create a sample movie tensor for testing."""
    # Create a simple 3D movie: 10 frames of 64x64 pixels
    return torch.randn(10, 64, 64, dtype=torch.float32)


@pytest.fixture
def sample_deformation_field():
    """Create a sample deformation field for testing."""
    # Deformation field with shape (n_z, n_y, n_x) = (1, 8, 8)
    return torch.zeros(2, 1, 8, 8, dtype=torch.float32)


def test_core_align_frames_basic(sample_movie, sample_deformation_field):
    """Test basic functionality of core_align_frames."""
    corrected_movie, updated_deformation_field, movie_prepared, trajectory = (
        core_align_frames(
            movie=sample_movie,
            deformation_field=sample_deformation_field,
            gain_map=None,
            dark_map=None,
            gain_flip=0,
            gain_rot=0,
            pixel_size=1.0,
            deformation_field_resolution=(1, 8, 8),
            patch_shape=(32, 32),
            multiply_gain=True,
            loss_trajectories=False,
            skip_movie_preparation=False,
            n_iterations=5,  # Use fewer iterations for faster testing
            do_correct_motion=True,
        )
    )

    # Check that outputs have correct types and shapes
    assert isinstance(corrected_movie, torch.Tensor)
    assert isinstance(updated_deformation_field, torch.Tensor)
    assert isinstance(movie_prepared, torch.Tensor)
    assert trajectory is None  # No trajectory requested

    # Check shapes are preserved
    assert corrected_movie.shape == sample_movie.shape
    assert movie_prepared.shape == sample_movie.shape
    assert updated_deformation_field.shape == sample_deformation_field.shape


def test_core_align_frames_with_skip_preparation(
    sample_movie, sample_deformation_field
):
    """Test core_align_frames with skip_movie_preparation=True."""
    corrected_movie, _, movie_prepared, _ = core_align_frames(
        movie=sample_movie,
        deformation_field=sample_deformation_field,
        gain_map=None,
        dark_map=None,
        gain_flip=0,
        gain_rot=0,
        pixel_size=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_shape=(32, 32),
        skip_movie_preparation=True,
        n_iterations=5,
        do_correct_motion=True,
    )

    # When skipping preparation, movie_prepared should be the same as input movie
    assert torch.allclose(movie_prepared, sample_movie)
    assert corrected_movie.shape == sample_movie.shape


def test_core_align_frames_without_motion_correction(
    sample_movie, sample_deformation_field
):
    """Test core_align_frames with do_correct_motion=False."""
    corrected_movie, updated_deformation_field, movie_prepared, _ = core_align_frames(
        movie=sample_movie,
        deformation_field=sample_deformation_field,
        gain_map=None,
        dark_map=None,
        gain_flip=0,
        gain_rot=0,
        pixel_size=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_shape=(32, 32),
        n_iterations=5,
        do_correct_motion=False,
    )

    # When not correcting motion, corrected_movie should equal movie_prepared
    assert torch.allclose(corrected_movie, movie_prepared)
    assert updated_deformation_field.shape == sample_deformation_field.shape
