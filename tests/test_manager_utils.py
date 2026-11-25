"""Tests for manager_utils: save_results summation/downsampling and overwrite checks."""

# pylint: disable=redefined-outer-name

from pathlib import Path

import mrcfile
import pytest
import torch

from ripple.config import AlignFramesConfig, MovieConfig, OutputConfig
from ripple.core.generate_image import fourier_crop_movie, generate_dose_weighted_image
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

    def test_both_sums_computed_in_one_pass(
        self, tmp_path, corrected_movie, monkeypatch
    ):
        """dw_sum and non_dw_sum must come from a single pass over the movie."""
        movie_config = _movie_config(super_resolution_factor=2)
        output_config = _output_config(tmp_path)

        calls = []
        original = manager_utils.generate_summed_images

        def _spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(manager_utils, "generate_summed_images", _spy)

        manager_utils.save_results(
            output_config,
            movie_config,
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        assert len(calls) == 1
        assert calls[0]["dose_weighted"] and calls[0]["non_dose_weighted"]
        assert calls[0]["super_resolution_factor"] == 2

    def test_no_summation_when_no_summation_output_requested(
        self, tmp_path, corrected_movie, monkeypatch
    ):
        movie_config = _movie_config(super_resolution_factor=2)
        output_config = _output_config(
            tmp_path, dw_sum_output_path=None, non_dw_sum_output_path=None
        )

        calls = []
        monkeypatch.setattr(
            manager_utils,
            "generate_summed_images",
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

    @pytest.mark.parametrize("factor", [1, 2])
    @pytest.mark.parametrize("shape", [(N_FRAMES, HEIGHT, WIDTH), (5, 34, 30)])
    def test_sums_match_crop_then_sum(self, tmp_path, factor, shape):
        """Outputs equal Fourier-cropping the whole movie, then summing it."""
        movie = torch.randn(*shape, generator=torch.Generator().manual_seed(0)) + 5.0
        movie_config = _movie_config(super_resolution_factor=factor)
        output_config = _output_config(tmp_path)

        manager_utils.save_results(
            output_config,
            movie_config,
            movie,
            updated_deformation_field=None,
            movie_prepared=movie,
            trajectory=None,
        )

        cropped, pixel_size = fourier_crop_movie(movie, 0.4, factor)
        expected_dw = generate_dose_weighted_image(
            cropped,
            pixel_size,
            movie_config.pre_exposure,
            movie_config.fluence_per_frame,
            movie_config.voltage,
        )
        expected_sum = torch.sum(cropped, dim=0)
        for path, expected in (
            (output_config.dw_sum_output_path, expected_dw),
            (output_config.non_dw_sum_output_path, expected_sum),
        ):
            with mrcfile.open(path) as mrc:
                data = torch.from_numpy(mrc.data.copy())
                assert mrc.voxel_size.x == pytest.approx(pixel_size, abs=1e-5)
            assert data.shape == expected.shape
            assert torch.allclose(data, expected, atol=1e-5 * expected.abs().max())


class TestSaveResultsOverwrite:
    """Every configured output honours `allow_file_overwrite=False`."""

    @pytest.mark.parametrize(
        "existing_field",
        [
            "deformation_field_output_path",
            "loss_trajectories_output_path",
            "non_dw_sum_output_path",
        ],
    )
    def test_existing_output_blocks_all_writes(
        self, tmp_path, corrected_movie, existing_field
    ):
        output_config = OutputConfig(allow_file_overwrite=False)
        output_config.dw_sum_output_path = str(tmp_path / "dw_sum.mrc")
        output_config.non_dw_sum_output_path = str(tmp_path / "non_dw_sum.mrc")
        output_config.deformation_field_output_path = str(tmp_path / "field.csv")
        output_config.loss_trajectories_output_path = str(tmp_path / "loss.csv")
        existing = Path(getattr(output_config, existing_field))
        existing.write_text("keep me")

        with pytest.raises(ValueError, match="already exists"):
            manager_utils.save_results(
                output_config,
                _movie_config(),
                corrected_movie,
                updated_deformation_field=None,
                movie_prepared=corrected_movie,
                trajectory=None,
            )

        assert existing.read_text() == "keep me"
        assert not (tmp_path / "dw_sum.mrc").exists()

    def test_overwrite_allowed_replaces_existing(self, tmp_path, corrected_movie):
        output_config = OutputConfig(allow_file_overwrite=True)
        output_config.dw_sum_output_path = str(tmp_path / "dw_sum.mrc")
        (tmp_path / "dw_sum.mrc").write_text("old")

        manager_utils.save_results(
            output_config,
            _movie_config(),
            corrected_movie,
            updated_deformation_field=None,
            movie_prepared=corrected_movie,
            trajectory=None,
        )

        with mrcfile.open(output_config.dw_sum_output_path) as mrc:
            assert mrc.data.shape == (HEIGHT, WIDTH)


class TestOutputConfigChecks:
    """OutputConfig helpers for paths assigned after construction."""

    def test_existing_output_paths(self, tmp_path):
        output_config = OutputConfig(allow_file_overwrite=False)
        output_config.dw_sum_output_path = str(tmp_path / "dw.mrc")
        output_config.loss_trajectories_output_path = str(tmp_path / "loss.csv")
        (tmp_path / "loss.csv").write_text("")

        assert output_config.output_paths == [
            str(tmp_path / "dw.mrc"),
            str(tmp_path / "loss.csv"),
        ]
        assert output_config.existing_output_paths() == [str(tmp_path / "loss.csv")]

    def test_check_output_paths_catches_assigned_path(self, tmp_path):
        output_config = OutputConfig(allow_file_overwrite=False)
        output_config.deformation_field_output_path = str(tmp_path / "field.h5")
        output_config.check_output_paths()  # nothing exists yet

        (tmp_path / "field.h5").write_text("")
        with pytest.raises(ValueError, match="already exists"):
            output_config.check_output_paths()

    def test_check_output_paths_ignores_existing_when_overwrite_allowed(self, tmp_path):
        output_config = OutputConfig(allow_file_overwrite=True)
        output_config.dw_sum_output_path = str(tmp_path / "dw.mrc")
        (tmp_path / "dw.mrc").write_text("")
        output_config.check_output_paths()


class TestPrepareMovieIfNeeded:
    """skip_movie_preparation still hands back a float32 movie."""

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.uint16])
    def test_skip_preparation_casts_integer_movie_to_float32(self, dtype):
        movie = torch.arange(2 * 4 * 4).reshape(2, 4, 4).to(dtype)
        alignment_config = AlignFramesConfig(
            deformation_field_resolution=(2, 2, 2), skip_movie_preparation=True
        )

        result = manager_utils.prepare_movie_if_needed(
            _movie_config(),
            alignment_config,
            movie,
            None,
            None,
            None,
            torch.device("cpu"),
        )

        assert result.dtype == torch.float32
        assert torch.equal(result, movie.to(torch.float32))

    def test_skip_preparation_float32_movie_is_not_copied(self):
        movie = torch.randn(2, 4, 4)
        alignment_config = AlignFramesConfig(
            deformation_field_resolution=(2, 2, 2), skip_movie_preparation=True
        )

        result = manager_utils.prepare_movie_if_needed(
            _movie_config(),
            alignment_config,
            movie,
            None,
            None,
            None,
            torch.device("cpu"),
        )

        assert result is movie
