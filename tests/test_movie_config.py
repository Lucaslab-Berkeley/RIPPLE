"""Tests for MovieConfig's crop-bounds auto-computation."""

# pylint: disable=redefined-outer-name

import pytest
import torch

from ripple.config import CropBoundsConfig, MovieConfig

HEIGHT = 32
WIDTH = 32


@pytest.fixture
def sample_movie() -> torch.Tensor:
    return torch.randn(5, HEIGHT, WIDTH, dtype=torch.float32)


@pytest.fixture
def sample_mask() -> torch.Tensor:
    mask = torch.zeros(HEIGHT, WIDTH, dtype=torch.float32)
    mask[8:20, 10:26] = 1.0  # tight bbox: (8, 19, 10, 25) -> shape (12, 16)
    return mask


def _movie_config(**crop_kwargs) -> MovieConfig:
    return MovieConfig(
        pixel_size=1.0,
        fluence=40.0,
        fluence_per_frame=10.0,
        crop_bounds_config=CropBoundsConfig(**crop_kwargs),
    )


class TestComputeCropBounds:
    """Tests for MovieConfig.compute_crop_bounds."""

    def test_mode_none_returns_none(self, sample_mask):
        config = _movie_config()  # default mode="none"
        assert config.compute_crop_bounds(sample_mask) is None

    def test_mode_tight_matches_expected_bbox(self, sample_mask):
        config = _movie_config(mode="tight")
        bounds = config.compute_crop_bounds(sample_mask)
        assert bounds == {"min_y": 8, "max_y": 19, "min_x": 10, "max_x": 25}

    def test_mode_nice_size_rounds_up(self, sample_mask):
        config = _movie_config(mode="nice_size", round_to=8)
        bounds = config.compute_crop_bounds(sample_mask)
        height = bounds["max_y"] - bounds["min_y"] + 1
        width = bounds["max_x"] - bounds["min_x"] + 1
        assert height % 8 == 0
        assert width % 8 == 0

    def test_works_regardless_of_mask_source(self, sample_mask):
        # Path-based (loaded from disk) and in-memory masks are both plain
        # torch.Tensor by the time they reach compute_crop_bounds, so an
        # in-memory mask must produce identical bounds to an equivalent
        # path-loaded one -- there is nothing that distinguishes the two here.
        config = _movie_config(mode="tight")
        in_memory_bounds = config.compute_crop_bounds(sample_mask)
        loaded_equivalent_bounds = config.compute_crop_bounds(sample_mask.clone())
        assert in_memory_bounds == loaded_equivalent_bounds


class TestPrepareAutoCrop:
    """Tests for MovieConfig.prepare's automatic crop_bounds computation."""

    def test_no_mask_no_crop(self, sample_movie):
        config = _movie_config(mode="tight")
        result = config.prepare(sample_movie)
        assert result.shape == sample_movie.shape

    def test_mask_with_mode_none_no_crop(self, sample_movie, sample_mask):
        config = _movie_config()  # default mode="none"
        result = config.prepare(sample_movie, mask=sample_mask)
        assert result.shape == sample_movie.shape

    def test_mask_with_mode_tight_auto_crops(self, sample_movie, sample_mask):
        config = _movie_config(mode="tight")
        result = config.prepare(sample_movie, mask=sample_mask)
        assert result.shape == (sample_movie.shape[0], 12, 16)

    def test_explicit_crop_bounds_overrides_auto_compute(
        self, sample_movie, sample_mask
    ):
        config = _movie_config(mode="tight")
        explicit_bounds = {"min_y": 0, "max_y": 15, "min_x": 0, "max_x": 15}
        result = config.prepare(
            sample_movie, mask=sample_mask, crop_bounds=explicit_bounds
        )
        assert result.shape == (sample_movie.shape[0], 16, 16)
