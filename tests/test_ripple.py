"""Tests for ripple."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import pytest
import torch
from torch_motion_correction import (
    DeformationField,
    PatchSamplingConfig,
    correct_motion,
    estimate_local_motion,
)
from torch_motion_correction import OptimizationConfig as MotionOptimizationConfig
from torch_motion_correction.optimization_state import OptimizationTracker

import ripple
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
    return MotionOptimizationConfig(max_iterations=5)


def test_estimate_motion_basic(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test basic functionality of estimate_local_motion."""
    deformation_field, trajectory = estimate_local_motion(
        image=sample_movie,
        initial_deformation_field=sample_deformation_field,
        pixel_spacing=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
    )

    assert isinstance(deformation_field, DeformationField)
    assert isinstance(trajectory, OptimizationTracker)
    assert deformation_field.shape == sample_deformation_field.shape


def test_estimate_motion_then_correct(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test that estimate_local_motion output can be used with correct_motion."""
    deformation_field, _ = estimate_local_motion(
        image=sample_movie,
        initial_deformation_field=sample_deformation_field,
        pixel_spacing=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
    )

    corrected = correct_motion(
        image=sample_movie,
        deformation_field=deformation_field,
        pixel_spacing=1.0,
    )

    assert corrected.shape == sample_movie.shape


def test_estimate_motion_single_frame(
    sample_deformation_field, patch_sampling, fast_optimization
):
    """Test estimate_motion with a single frame movie."""
    single_frame_movie = torch.randn(1, 64, 64, dtype=torch.float32)

    deformation_field, _ = estimate_local_motion(
        image=single_frame_movie,
        initial_deformation_field=sample_deformation_field,
        pixel_spacing=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
    )

    assert deformation_field.shape == sample_deformation_field.shape


def test_estimate_motion_with_gain_map(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test estimate_motion after gain map preparation."""
    gain_map = torch.ones(64, 64, dtype=torch.float32) * 2.0
    prepared = prepare_movie(sample_movie, gain_map, None, gain_flip=0, gain_rot=0)

    deformation_field, _ = estimate_local_motion(
        image=prepared,
        initial_deformation_field=sample_deformation_field,
        pixel_spacing=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
    )

    corrected = correct_motion(
        image=prepared,
        deformation_field=deformation_field,
        pixel_spacing=1.0,
    )
    assert corrected.shape == sample_movie.shape


def test_estimate_motion_with_dark_map(
    sample_movie, sample_deformation_field, patch_sampling, fast_optimization
):
    """Test estimate_motion after dark map preparation."""
    dark_map = torch.ones(64, 64, dtype=torch.float32) * 0.5
    prepared = prepare_movie(sample_movie, None, dark_map, gain_flip=0, gain_rot=0)

    deformation_field, _ = estimate_local_motion(
        image=prepared,
        initial_deformation_field=sample_deformation_field,
        pixel_spacing=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
    )

    corrected = correct_motion(
        image=prepared,
        deformation_field=deformation_field,
        pixel_spacing=1.0,
    )
    assert corrected.shape == sample_movie.shape


def test_estimate_motion_no_initial_field(
    sample_movie, patch_sampling, fast_optimization
):
    """Test estimate_motion with no initial deformation field (zero-initialized)."""
    deformation_field, trajectory = estimate_local_motion(
        image=sample_movie,
        initial_deformation_field=None,
        pixel_spacing=1.0,
        deformation_field_resolution=(1, 8, 8),
        patch_sampling=patch_sampling,
        optimization=fast_optimization,
    )

    assert isinstance(deformation_field, DeformationField)
    assert isinstance(trajectory, OptimizationTracker)
