"""Tests for motion_priors module utility functions."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import torch

from ripple.core.motion_priors import (
    _build_exponential_kernel,
    _build_gaussian_kernel_1d,
    _build_physical_coords,
    _create_exponential_sigma_a,
    _normalize_sigma_fluence,
    _pairwise_dist_matrix,
)


def test_pairwise_dist_matrix_basic():
    """Test _pairwise_dist_matrix with simple coordinates."""
    # Create 3 points: (0,0), (1,0), (0,1)
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    result = _pairwise_dist_matrix(coords)

    # Should be 3x3 matrix
    assert result.shape == (3, 3)

    # Diagonal should be (approximately) zero (distance from point to itself)
    assert torch.allclose(torch.diag(result), torch.zeros(3), atol=1e-5)

    # Should be symmetric
    assert torch.allclose(result, result.T)

    # Distance between (0,0) and (1,0) should be 1.0
    assert torch.allclose(result[0, 1], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(result[1, 0], torch.tensor(1.0), atol=1e-5)

    # Distance between (0,0) and (0,1) should be 1.0
    assert torch.allclose(result[0, 2], torch.tensor(1.0), atol=1e-5)

    # Distance between (1,0) and (0,1) should be sqrt(2)
    assert torch.allclose(result[1, 2], torch.tensor(2.0**0.5), atol=1e-5)


def test_pairwise_dist_matrix_single_point():
    """Test _pairwise_dist_matrix with a single point."""
    coords = torch.tensor([[5.0, 3.0]], dtype=torch.float32)
    result = _pairwise_dist_matrix(coords)

    assert result.shape == (1, 1)
    assert torch.allclose(result[0, 0], torch.tensor(0.0), atol=1e-5)


def test_pairwise_dist_matrix_symmetric():
    """Test that _pairwise_dist_matrix produces symmetric results."""
    coords = torch.randn(10, 2, dtype=torch.float32)
    result = _pairwise_dist_matrix(coords)

    # Should be symmetric
    assert torch.allclose(result, result.T)


def test_build_exponential_kernel_basic():
    """Test _build_exponential_kernel with simple parameters."""
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    sigma_d = 1.0
    sigma_v = 1.0

    result = _build_exponential_kernel(coords, sigma_d, sigma_v)

    # Should be 2x2 matrix
    assert result.shape == (2, 2)

    # Should be symmetric
    assert torch.allclose(result, result.T)

    # Diagonal should be sigma_v^2 (distance is 0)
    assert torch.allclose(result[0, 0], torch.tensor(sigma_v**2), atol=1e-5)
    assert torch.allclose(result[1, 1], torch.tensor(sigma_v**2), atol=1e-5)

    # Off-diagonal: distance is 1.0, so exp(-1.0/1.0) = exp(-1.0)
    expected_off_diag = sigma_v**2 * torch.exp(torch.tensor(-1.0))
    assert torch.allclose(result[0, 1], expected_off_diag, atol=1e-5)


def test_build_exponential_kernel_decay():
    """Test that _build_exponential_kernel decays with distance."""
    # Create points at increasing distances
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    sigma_d = 1.0
    sigma_v = 1.0

    result = _build_exponential_kernel(coords, sigma_d, sigma_v)

    # Kernel value should decrease with distance
    # result[0,1] should be > result[0,2] (closer points have higher kernel value)
    assert result[0, 1] > result[0, 2]

    # All values should be positive
    assert torch.all(result > 0)


def test_build_gaussian_kernel_1d_basic():
    """Test _build_gaussian_kernel_1d with simple parameters."""
    grid = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    sigma_space = 1.0
    sigma_strength = 1.0

    result = _build_gaussian_kernel_1d(grid, sigma_space, sigma_strength)

    # Should be 3x3x1 tensor (grid is (N, 1), so result is (N, N, 1))
    assert result.shape == (3, 3, 1)

    # Should be symmetric along first two dimensions
    assert torch.allclose(result[:, :, 0], result[:, :, 0].T)

    # Diagonal should be sigma_strength (distance is 0)
    assert torch.allclose(result[0, 0, 0], torch.tensor(sigma_strength), atol=1e-5)


def test_build_gaussian_kernel_1d_decay():
    """Test that _build_gaussian_kernel_1d decays with distance."""
    grid = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    sigma_space = 1.0
    sigma_strength = 1.0

    result = _build_gaussian_kernel_1d(grid, sigma_space, sigma_strength)

    # Kernel value should decrease with distance
    # result[0,1,0] should be > result[0,2,0]
    assert result[0, 1, 0] > result[0, 2, 0]

    # All values should be positive
    assert torch.all(result > 0)


def test_build_physical_coords_basic():
    """Test _build_physical_coords with basic parameters."""
    nh = 3
    nw = 4
    image_shape = (100, 80)  # 100 pixels high, 80 pixels wide
    pixel_size = 1.0
    device = torch.device("cpu")

    result = _build_physical_coords(nh, nw, image_shape, pixel_size, device)

    # Should have shape (nh * nw, 2) = (12, 2)
    assert result.shape == (nh * nw, 2)

    # First coordinate should be (0, 0)
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0]), atol=1e-5)

    # Last coordinate should be (100, 80) in Angstroms
    assert torch.allclose(result[-1], torch.tensor([100.0, 80.0]), atol=1e-5)


def test_build_physical_coords_pixel_size():
    """Test _build_physical_coords with different pixel size."""
    nh = 2
    nw = 2
    image_shape = (10, 10)
    pixel_size = 2.0  # 2 Angstroms per pixel
    device = torch.device("cpu")

    result = _build_physical_coords(nh, nw, image_shape, pixel_size, device)

    # Image is 10x10 pixels, so 20x20 Angstroms
    # With nh=2, nw=2, we get coordinates at (0,0), (0,20), (20,0), (20,20)
    assert torch.allclose(result[-1], torch.tensor([20.0, 20.0]), atol=1e-5)


def test_normalize_sigma_fluence_basic():
    """Test _normalize_sigma_fluence with basic parameters."""
    sigma = 1.0
    total_fluence = 10.0
    nt = 5

    result = _normalize_sigma_fluence(sigma, total_fluence, nt)

    # fluence_per_frame = 10.0 / 5 = 2.0
    # result = 1.0 * 2.0 = 2.0
    expected = 1.0 * (10.0 / 5)
    assert result == expected


def test_normalize_sigma_fluence_zero_frames():
    """Test _normalize_sigma_fluence edge case with single frame."""
    sigma = 1.0
    total_fluence = 10.0
    nt = 1

    result = _normalize_sigma_fluence(sigma, total_fluence, nt)

    # fluence_per_frame = 10.0 / 1 = 10.0
    expected = 1.0 * 10.0
    assert result == expected


def test_create_exponential_sigma_a_basic():
    """Test _create_exponential_sigma_a with basic parameters."""
    total_fluence = 10.0
    n_frames = 5
    amplitude = 2.0
    decay_rate = 0.1
    offset = 1.0

    result = _create_exponential_sigma_a(
        total_fluence, n_frames, amplitude, decay_rate, offset
    )

    # Should have shape (n_frames - 2) = 3
    assert result.shape == (3,)

    # All values should be positive
    assert torch.all(result > 0)

    # First value should be larger than last (decay)
    assert result[0] > result[-1]

    # First value should be approximately:
    # amplitude * exp(-decay_rate * fluence) + offset
    fluence_per_frame = total_fluence / n_frames
    first_fluence = 1.0 * fluence_per_frame
    expected_first = (
        amplitude * torch.exp(-decay_rate * torch.tensor(first_fluence)) + offset
    )
    assert torch.allclose(result[0], expected_first, atol=1e-3)


def test_create_exponential_sigma_a_decay():
    """Test that _create_exponential_sigma_a shows exponential decay."""
    total_fluence = 20.0
    n_frames = 10
    amplitude = 2.0
    decay_rate = 0.1
    offset = 1.0

    result = _create_exponential_sigma_a(
        total_fluence, n_frames, amplitude, decay_rate, offset
    )

    # Should have shape (n_frames - 2) = 8
    assert result.shape == (8,)

    # Values should decrease (decay)
    for i in range(len(result) - 1):
        assert result[i] > result[i + 1]

    # All values should be >= offset (minimum value)
    assert torch.all(result >= offset)


def test_create_exponential_sigma_a_no_decay():
    """Test _create_exponential_sigma_a with zero decay rate."""
    total_fluence = 10.0
    n_frames = 5
    amplitude = 2.0
    decay_rate = 0.0  # No decay
    offset = 1.0

    result = _create_exponential_sigma_a(
        total_fluence, n_frames, amplitude, decay_rate, offset
    )

    # With no decay, all values should be the same: amplitude + offset
    expected = amplitude + offset
    assert torch.allclose(result, torch.full_like(result, expected), atol=1e-5)


def test_create_exponential_sigma_a_minimum_frames():
    """Test _create_exponential_sigma_a with minimum number of frames."""
    total_fluence = 10.0
    n_frames = 3  # Minimum to get n_frames - 2 = 1
    amplitude = 2.0
    decay_rate = 0.1
    offset = 1.0

    result = _create_exponential_sigma_a(
        total_fluence, n_frames, amplitude, decay_rate, offset
    )

    # Should have shape (1,)
    assert result.shape == (1,)
    assert result[0] > offset
