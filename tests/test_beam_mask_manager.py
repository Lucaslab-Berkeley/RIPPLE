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
    MovieConfig,
    OutputConfig,
)
from ripple.core.beam_mask import make_ellipse_mask
from ripple.managers import AlignFramesManager, BeamMaskManager

N_FRAMES = 4
HEIGHT = 64
WIDTH = 64


def _write_mrc(path, array: np.ndarray) -> str:
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(array)
    return str(path)


@pytest.fixture
def ellipse_mask() -> np.ndarray:
    return make_ellipse_mask(
        shape=(HEIGHT, WIDTH),
        center_y=34.0,
        center_x=28.0,
        axis1=20.0,
        axis2=15.0,
        angle_deg=10.0,
    )


@pytest.fixture
def movie_array(ellipse_mask) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((N_FRAMES, HEIGHT, WIDTH)).astype(np.float32) * 0.1
    signal = ellipse_mask.astype(np.float32) * 5.0
    return noise + signal


@pytest.fixture
def movie_config(tmp_path, movie_array) -> MovieConfig:
    """A MovieConfig backed by a tiny on-disk MRC movie.

    fluence/fluence_per_frame are irrelevant to beam mask estimation and left at 0.0.
    """
    return MovieConfig(
        movie_path=_write_mrc(tmp_path / "movie.mrc", movie_array),
        pixel_size=1.0,
        fluence=0.0,
        fluence_per_frame=0.0,
    )


@pytest.fixture
def computational_config() -> ComputationalConfig:
    return ComputationalConfig(gpu_id="cpu")


@pytest.fixture
def beam_mask_config() -> BeamMaskConfig:
    return BeamMaskConfig(diameter_reduction=0.05, low_pass_resolution=10.0)


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
        assert result.pixel_size == 1.0
        assert result.diameter_reduction == 0.05
        assert result.threshold_method == "otsu"

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
