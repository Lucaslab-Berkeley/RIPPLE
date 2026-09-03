"""Tests for manager_utils.save_results, focused on summation-stage downsampling."""

# pylint: disable=redefined-outer-name

import mrcfile
import pytest
import torch

from ripple.config import MovieConfig, OutputConfig
from ripple.managers import manager_utils

N_FRAMES = 4
HEIGHT = 16
WIDTH = 16


@pytest.fixture
def corrected_movie() -> torch.Tensor:
    return torch.randn(N_FRAMES, HEIGHT, WIDTH, dtype=torch.float32) + 5.0


def _movie_config(**overrides) -> MovieConfig:
    defaults = {
        "pixel_size": 0.4,
        "fluence": 40.0,
        "fluence_per_frame": 10.0,
        "super_resolution_factor": 1,
    }
    defaults.update(overrides)
    return MovieConfig(**defaults)


def _output_config(tmp_path, **overrides) -> OutputConfig:
    defaults = {
        "dw_sum_output_path": str(tmp_path / "dw_sum.mrc"),
        "non_dw_sum_output_path": str(tmp_path / "non_dw_sum.mrc"),
        "motion_corrected_movie_output_path": str(tmp_path / "corrected.mrc"),
    }
    defaults.update(overrides)
    return OutputConfig(**defaults)


class TestSaveResultsDownsampling:
    """Tests for the Fourier-crop downsampling wired into save_results.

    Downsampling at the summation stage is inferred entirely from
    `super_resolution_factor`: a factor of 1 (the default) never crops, and any
    factor greater than 1 crops the dw_sum/non_dw_sum outputs down to
    `pixel_size * super_resolution_factor`.
    """

    def test_factor_one_keeps_native_resolution(self, tmp_path, corrected_movie):
        movie_config = _movie_config(super_resolution_factor=1)
        output_config = _output_config(tmp_path)

        manager_utils.save_results(
            output_config,
            movie_config,
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        with mrcfile.open(output_config.dw_sum_output_path) as mrc:
            assert mrc.data.shape == (HEIGHT, WIDTH)
            assert mrc.voxel_size.x == pytest.approx(0.4, abs=1e-5)
        with mrcfile.open(output_config.non_dw_sum_output_path) as mrc:
            assert mrc.data.shape == (HEIGHT, WIDTH)
            assert mrc.voxel_size.x == pytest.approx(0.4, abs=1e-5)

    def test_factor_greater_than_one_crops_both_summation_outputs(
        self, tmp_path, corrected_movie
    ):
        movie_config = _movie_config(super_resolution_factor=2)
        output_config = _output_config(tmp_path)

        manager_utils.save_results(
            output_config,
            movie_config,
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        with mrcfile.open(output_config.dw_sum_output_path) as mrc:
            assert mrc.data.shape == (HEIGHT // 2, WIDTH // 2)
            assert mrc.voxel_size.x == pytest.approx(0.8, abs=1e-5)
        with mrcfile.open(output_config.non_dw_sum_output_path) as mrc:
            assert mrc.data.shape == (HEIGHT // 2, WIDTH // 2)
            assert mrc.voxel_size.x == pytest.approx(0.8, abs=1e-5)

    def test_does_not_crop_motion_corrected_movie_output(
        self, tmp_path, corrected_movie
    ):
        """Only the summed micrograph outputs are downsampled, never the movie."""
        movie_config = _movie_config(super_resolution_factor=2)
        output_config = _output_config(tmp_path)

        manager_utils.save_results(
            output_config,
            movie_config,
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        with mrcfile.open(output_config.motion_corrected_movie_output_path) as mrc:
            assert mrc.data.shape == (N_FRAMES, HEIGHT, WIDTH)
            assert mrc.voxel_size.x == pytest.approx(0.4, abs=1e-5)

    def test_crop_only_computed_once_for_both_outputs(
        self, tmp_path, corrected_movie, monkeypatch
    ):
        """Both dw_sum and non_dw_sum must share a single Fourier-crop call."""
        movie_config = _movie_config(super_resolution_factor=2)
        output_config = _output_config(tmp_path)

        calls = []
        original = manager_utils.fourier_crop_movie

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(manager_utils, "fourier_crop_movie", _spy)

        manager_utils.save_results(
            output_config,
            movie_config,
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        assert len(calls) == 1

    def test_no_crop_computed_when_no_summation_output_requested(
        self, tmp_path, corrected_movie, monkeypatch
    ):
        movie_config = _movie_config(super_resolution_factor=2)
        output_config = _output_config(
            tmp_path, dw_sum_output_path=None, non_dw_sum_output_path=None
        )

        calls = []
        monkeypatch.setattr(
            manager_utils,
            "fourier_crop_movie",
            lambda *a, **k: calls.append((a, k)),
        )

        manager_utils.save_results(
            output_config,
            movie_config,
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        assert len(calls) == 0

    def test_no_crop_computed_when_factor_is_one(
        self, tmp_path, corrected_movie, monkeypatch
    ):
        movie_config = _movie_config(super_resolution_factor=1)
        output_config = _output_config(tmp_path)

        calls = []
        monkeypatch.setattr(
            manager_utils,
            "fourier_crop_movie",
            lambda *a, **k: calls.append((a, k)),
        )

        manager_utils.save_results(
            output_config,
            movie_config,
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        assert len(calls) == 0
