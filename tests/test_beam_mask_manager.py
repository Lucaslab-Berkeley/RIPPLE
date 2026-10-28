"""Tests for BeamMaskManager."""

# pylint: disable=redefined-outer-name

import mrcfile
import numpy as np
import pytest
import torch

from ripple.config import (
    AlignFramesConfig,
    BeamMaskConfig,
    ComputationalConfig,
    CropBoundsConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.core.beam_mask import make_ellipse_mask
from ripple.managers import AlignFramesManager, BeamMaskManager

N_FRAMES = 4
HEIGHT = 192
WIDTH = 256
PIXEL_SIZE = 8.0

# Nearly round beam, clipped by the top of the frame.
ELLIPSE = {
    "center_y": 100.0,
    "center_x": 130.0,
    "axis1": 104.0,
    "axis2": 100.0,
    "angle_deg": 40.0,
}


def _write_mrc(path, array: np.ndarray) -> str:
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(array)
    return str(path)


@pytest.fixture
def ellipse_mask() -> np.ndarray:
    return make_ellipse_mask(shape=(HEIGHT, WIDTH), **ELLIPSE)


@pytest.fixture
def movie_array(ellipse_mask) -> np.ndarray:
    """Poisson movie of the beam at ``PIXEL_SIZE`` A/px."""
    rng = np.random.default_rng(0)
    expected = 0.1 + 10.0 * ellipse_mask.astype(np.float32)
    return rng.poisson(expected, size=(N_FRAMES, HEIGHT, WIDTH)).astype(np.float32)


@pytest.fixture
def movie_config(tmp_path, movie_array) -> MovieConfig:
    """A MovieConfig backed by a small on-disk MRC movie.

    fluence/fluence_per_frame are irrelevant to beam mask estimation and left at 0.0.
    """
    return MovieConfig(
        movie_path=_write_mrc(tmp_path / "movie.mrc", movie_array),
        pixel_size=PIXEL_SIZE,
        fluence=0.0,
        fluence_per_frame=0.0,
    )


@pytest.fixture
def computational_config() -> ComputationalConfig:
    return ComputationalConfig(gpu_id="cpu")


@pytest.fixture
def beam_mask_config() -> BeamMaskConfig:
    return BeamMaskConfig(diameter_reduction=0.05)


@pytest.fixture
def manager(computational_config, movie_config, beam_mask_config) -> BeamMaskManager:
    return BeamMaskManager(
        computational_config=computational_config,
        movie_config=movie_config,
        beam_mask_config=beam_mask_config,
    )


class TestEstimate:
    """Tests for BeamMaskManager.estimate."""

    def test_loads_from_config_when_no_movie_given(self, manager):
        result = manager.estimate()

        assert result.image_shape_y == HEIGHT
        assert result.image_shape_x == WIDTH
        assert result.pixel_size == PIXEL_SIZE
        assert result.diameter_reduction == 0.05
        assert result.threshold_method == "sobel"

    def test_explicit_movie_tensor_skips_disk_load(self, manager, movie_array):
        manager.movie_config.movie_path = "/nonexistent/path/should-never-be-read.mrc"
        movie = torch.from_numpy(movie_array)

        result = manager.estimate(movie=movie)

        assert result.image_shape_y == HEIGHT
        assert result.image_shape_x == WIDTH

    def test_to_mask_recovers_beam_region(self, manager, ellipse_mask):
        result = manager.estimate()
        recovered = result.to_mask().numpy()

        # Recovered (shrunk) mask should mostly overlap the true beam region.
        intersection = np.logical_and(recovered, ellipse_mask).sum()
        assert intersection / recovered.sum() > 0.9

    def test_output_crop_bounds_respect_divisible_by(
        self, computational_config, movie_config
    ):
        # super_resolution_factor=3 is not a divisor of the tight bbox's natural
        # size for this ellipse -- crop_bounds_config.divisible_by must still
        # force both output crop side lengths to a multiple of 3.
        beam_mask_config = BeamMaskConfig(
            diameter_reduction=0.05,
            crop_bounds_config=CropBoundsConfig(mode="tight", divisible_by=3),
        )
        manager = BeamMaskManager(
            computational_config=computational_config,
            movie_config=movie_config,
            beam_mask_config=beam_mask_config,
        )

        result = manager.estimate()

        output_height = result.output_crop_max_y - result.output_crop_min_y + 1
        output_width = result.output_crop_max_x - result.output_crop_min_x + 1
        assert output_height % 3 == 0
        assert output_width % 3 == 0

    def test_recovers_ellipse(self, manager):
        result = manager.estimate()

        assert 0 < result.n_inliers <= result.n_edge_points
        center_offset = np.hypot(
            result.center_y - ELLIPSE["center_y"], result.center_x - ELLIPSE["center_x"]
        )
        assert center_offset < 2.0
        assert sorted([result.axis1, result.axis2]) == pytest.approx(
            sorted([ELLIPSE["axis1"], ELLIPSE["axis2"]]), rel=0.02
        )

    def test_raises_when_no_beam(self, manager):
        rng = np.random.default_rng(0)
        movie = rng.poisson(1.0, size=(N_FRAMES, HEIGHT, WIDTH)).astype(np.float32)

        with pytest.raises(ValueError, match="beam"):
            manager.estimate(movie=torch.from_numpy(movie))


class TestSharedTensorComposition:
    """Verify a single loaded movie tensor composes with AlignFramesManager."""

    def test_shared_movie_tensor_feeds_align_frames_manager(
        self, manager, movie_config, computational_config, movie_array
    ):
        movie = torch.from_numpy(movie_array)

        beam_mask_result = manager.estimate(movie=movie)

        align_movie_config = MovieConfig(
            pixel_size=1.0,
            fluence=40.0,
            fluence_per_frame=10.0,
            mask_fill_noise=True,
        )
        align_manager = AlignFramesManager(
            computational_config=computational_config,
            movie_config=align_movie_config,
            output_config=OutputConfig(),
            alignment_config=AlignFramesConfig(
                deformation_field_resolution=(N_FRAMES, 2, 2),
                max_iterations=1,
            ),
        )

        prepared = align_manager.prepare_movie(
            movie=movie, mask=beam_mask_result.to_mask()
        )

        assert prepared.shape == movie.shape

    def test_output_crop_bounds_feed_align_frames_manager_crop(
        self, computational_config, movie_config, movie_array
    ):
        """The beam mask's output_crop_* bounds physically crop the movie."""
        beam_mask_config = BeamMaskConfig(
            diameter_reduction=0.05,
            crop_bounds_config=CropBoundsConfig(mode="tight"),
        )
        beam_mask_manager = BeamMaskManager(
            computational_config=computational_config,
            movie_config=movie_config,
            beam_mask_config=beam_mask_config,
        )
        movie = torch.from_numpy(movie_array)
        beam_mask_result = beam_mask_manager.estimate(movie=movie)

        align_movie_config = MovieConfig(
            pixel_size=1.0, fluence=40.0, fluence_per_frame=10.0, mask_fill_noise=True
        )
        align_manager = AlignFramesManager(
            computational_config=computational_config,
            movie_config=align_movie_config,
            output_config=OutputConfig(),
            alignment_config=AlignFramesConfig(
                deformation_field_resolution=(N_FRAMES, 2, 2), max_iterations=1
            ),
        )

        prepared = align_manager.prepare_movie(
            movie=movie,
            mask=beam_mask_result.to_mask(),
            crop_bounds=beam_mask_result.output_crop_bounds,
        )

        expected_h = (
            beam_mask_result.output_crop_max_y - beam_mask_result.output_crop_min_y + 1
        )
        expected_w = (
            beam_mask_result.output_crop_max_x - beam_mask_result.output_crop_min_x + 1
        )
        assert prepared.shape == (N_FRAMES, expected_h, expected_w)
        assert (expected_h, expected_w) != (HEIGHT, WIDTH)
