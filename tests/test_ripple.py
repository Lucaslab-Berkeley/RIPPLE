"""Tests for ripple."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import pytest
import torch
from torch_motion_correction import DeformationField, PatchSamplingConfig
from torch_motion_correction import OptimizationConfig as MotionOptimizationConfig
from torch_motion_correction.optimization_state import OptimizationTracker

import ripple
from ripple.core.core_align_frames import core_align_frames
from ripple.core.prepare_movie import prepare_movie


def test_imports_with_version():
    """Test that ripple can be imported and has a version."""
    assert isinstance(ripple.__version__, str)


@pytest.fixture
def sample_movie():
    """Create a sample movie tensor for testing."""
    return torch.randn(10, 64, 64, dtype=torch.float32)


@pytest.fixture
def sample_deformation_field():
    """Create a sample DeformationField for testing."""
    return DeformationField(
        data=torch.zeros(2, 1, 8, 8, dtype=torch.float32),
        grid_type="catmull_rom",
    )


@pytest.fixture
def patch_sampling():
    """Patch sampling config used across tests."""
    return PatchSamplingConfig(patch_shape=(32, 32))


@pytest.fixture
def fast_optimization():
    """Optimization config with few iterations for fast tests."""
    return MotionOptimizationConfig(n_iterations=5)


def test_core_align_frames_basic(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test basic functionality of core_align_frames."""
    corrected_movie, updated_deformation_field, movie_prepared, trajectory = (
        core_align_frames(
            movie=sample_movie,
            initial_deformation_field=sample_deformation_field,
            pixel_size=1.0,
            deformation_field_resolution=(1, 8, 8),
            patch_sampling=patch_sampling,
            optimization=fast_optimization,
            do_correct_motion=True,
        )
    )

    assert isinstance(corrected_movie, torch.Tensor)
    assert isinstance(updated_deformation_field, DeformationField)
    assert isinstance(movie_prepared, torch.Tensor)
    assert isinstance(trajectory, OptimizationTracker)

    assert corrected_movie.shape == sample_movie.shape
    assert movie_prepared.shape == sample_movie.shape
    assert updated_deformation_field.shape == sample_deformation_field.shape


def test_core_align_frames_without_motion_correction(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test core_align_frames with do_correct_motion=False."""
    corrected_movie, updated_deformation_field, movie_prepared, _ = core_align_frames(
        movie=sample_movie,
        initial_deformation_field=sample_deformation_field,
        pixel_size=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
        do_correct_motion=False,
    )

    assert torch.allclose(corrected_movie, movie_prepared)
    assert updated_deformation_field.shape == sample_deformation_field.shape


def test_core_align_frames_single_frame(
    sample_deformation_field, patch_sampling, fast_optimization
):
    """Test core_align_frames with a single frame movie."""
    single_frame_movie = torch.randn(1, 64, 64, dtype=torch.float32)

    corrected_movie, updated_deformation_field, movie_prepared, _ = core_align_frames(
        movie=single_frame_movie,
        initial_deformation_field=sample_deformation_field,
        pixel_size=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
        do_correct_motion=True,
    )

    assert corrected_movie.shape == single_frame_movie.shape
    assert movie_prepared.shape == single_frame_movie.shape
    assert updated_deformation_field.shape == sample_deformation_field.shape


def test_core_align_frames_with_gain_map(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test core_align_frames after gain map preparation."""
    gain_map = torch.ones(64, 64, dtype=torch.float32) * 2.0
    prepared = prepare_movie(sample_movie, gain_map, None, gain_flip=0, gain_rot=0)

    corrected_movie, _, movie_prepared, _ = core_align_frames(
        movie=prepared,
        initial_deformation_field=sample_deformation_field,
        pixel_size=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
        do_correct_motion=True,
    )

    assert corrected_movie.shape == sample_movie.shape
    assert movie_prepared.shape == sample_movie.shape


def test_core_align_frames_with_dark_map(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test core_align_frames after dark map preparation."""
    dark_map = torch.ones(64, 64, dtype=torch.float32) * 0.5
    prepared = prepare_movie(sample_movie, None, dark_map, gain_flip=0, gain_rot=0)

    corrected_movie, _, movie_prepared, _ = core_align_frames(
        movie=prepared,
        initial_deformation_field=sample_deformation_field,
        pixel_size=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
        do_correct_motion=True,
    )

    assert corrected_movie.shape == sample_movie.shape
    assert movie_prepared.shape == sample_movie.shape


def test_core_align_frames_zero_iterations(
    sample_movie, sample_deformation_field, patch_sampling
):
    """Test core_align_frames with zero iterations."""
    zero_optimization = MotionOptimizationConfig(n_iterations=0)

    corrected_movie, updated_deformation_field, movie_prepared, _ = core_align_frames(
        movie=sample_movie,
        initial_deformation_field=sample_deformation_field,
        pixel_size=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=zero_optimization,
        do_correct_motion=True,
    )

    assert corrected_movie.shape == sample_movie.shape
    assert movie_prepared.shape == sample_movie.shape
    assert updated_deformation_field.shape == sample_deformation_field.shape
