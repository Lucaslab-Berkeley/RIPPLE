"""Tests for AlignFramesManager.

All `torch_motion_correction` backend calls (`estimate_global_motion`,
`estimate_local_motion`, `correct_motion`) are monkeypatched with cheap stand-ins so
these tests never run real motion estimation/correction. Movie preparation itself
(`prepare_movie`) is exercised for real, but only on tiny (few-frame, 16x16) tensors,
so no expensive PyTorch computation occurs anywhere in this file.
"""

# pylint: disable=redefined-outer-name,protected-access

import mrcfile
import numpy as np
import pytest
import torch

from ripple.config import (
    AlignFramesConfig,
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.managers import align_frames_manager as afm_module
from ripple.managers.align_frames_manager import AlignFramesManager

N_FRAMES = 4
HEIGHT = 16
WIDTH = 16


@pytest.fixture(autouse=True)
def _restore_grad_state():
    """Restore torch's global grad-enabled state after each test.

    `estimate_motion` calls `torch.set_grad_enabled(True)` as a side effect; restore
    whatever was in effect before the test so this file doesn't leak global state
    into other test files (order-dependent test pollution).
    """
    previous = torch.is_grad_enabled()
    yield
    torch.set_grad_enabled(previous)


def _write_mrc(path, array: np.ndarray) -> str:
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(array)
    return str(path)


@pytest.fixture
def movie_array() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal((N_FRAMES, HEIGHT, WIDTH)) + 10.0).astype(np.float32)


@pytest.fixture
def gain_array() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)


@pytest.fixture
def dark_array() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)


@pytest.fixture
def mask_array() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mask[4:12, 4:12] = 1.0
    return mask


@pytest.fixture
def movie_config(
    tmp_path, movie_array, gain_array, dark_array, mask_array
) -> MovieConfig:
    """A MovieConfig backed by tiny on-disk MRC files for movie/gain/dark/mask."""
    return MovieConfig(
        movie_path=_write_mrc(tmp_path / "movie.mrc", movie_array),
        pixel_size=1.0,
        fluence=40.0,
        fluence_per_frame=10.0,
        gain_path=_write_mrc(tmp_path / "gain.mrc", gain_array),
        dark_path=_write_mrc(tmp_path / "dark.mrc", dark_array),
        mask_path=_write_mrc(tmp_path / "mask.mrc", mask_array),
    )


@pytest.fixture
def computational_config() -> ComputationalConfig:
    return ComputationalConfig(gpu_id="cpu")


@pytest.fixture
def output_config() -> OutputConfig:
    return OutputConfig()


@pytest.fixture
def alignment_config() -> AlignFramesConfig:
    return AlignFramesConfig(
        deformation_field_resolution=(N_FRAMES, 2, 2),
        max_iterations=1,
    )


@pytest.fixture
def manager(
    computational_config, movie_config, output_config, alignment_config
) -> AlignFramesManager:
    return AlignFramesManager(
        computational_config=computational_config,
        movie_config=movie_config,
        output_config=output_config,
        alignment_config=alignment_config,
    )


def _make_manager(
    computational_config, movie_config, output_config, alignment_config
) -> AlignFramesManager:
    return AlignFramesManager(
        computational_config=computational_config,
        movie_config=movie_config,
        output_config=output_config,
        alignment_config=alignment_config,
    )


# ---------------------------------------------------------------------------
# prepare_movie
# ---------------------------------------------------------------------------


class TestPrepareMovie:
    """Tests for AlignFramesManager.prepare_movie."""

    def test_loads_from_config_when_nothing_provided(self, manager):
        """With no arguments, movie/gain/dark/mask are all loaded from config."""
        result = manager.prepare_movie()

        assert result.shape == (N_FRAMES, HEIGHT, WIDTH)
        for frame in result:
            assert torch.abs(frame.mean()) < 1e-4

    def test_explicit_movie_tensor_skips_disk_load(self, manager, movie_config):
        """Passing `movie=` directly must never touch `movie_config.movie_path`."""
        movie_config.movie_path = "/nonexistent/path/should-never-be-read.mrc"
        raw_movie = torch.full((3, HEIGHT, WIDTH), 5.0)

        result = manager.prepare_movie(movie=raw_movie)

        assert result.shape == raw_movie.shape

    def test_explicit_gain_map_overrides_config(self, manager):
        """An explicit gain_map should be used instead of the one on disk."""
        raw_movie = torch.full((2, HEIGHT, WIDTH), 3.0)
        override_gain = torch.full((HEIGHT, WIDTH), 100.0)

        result_default = manager.prepare_movie(movie=raw_movie.clone())
        result_override = manager.prepare_movie(
            movie=raw_movie.clone(), gain_map=override_gain
        )

        assert not torch.allclose(result_default, result_override)

    def test_explicit_mask_overrides_config(self, manager):
        """An explicit mask should be used instead of the one on disk."""
        raw_movie = torch.full((2, HEIGHT, WIDTH), 3.0)
        all_ones_mask = torch.ones(HEIGHT, WIDTH)

        result_config_mask = manager.prepare_movie(movie=raw_movie.clone())
        result_override = manager.prepare_movie(
            movie=raw_movie.clone(), mask=all_ones_mask
        )

        assert not torch.allclose(result_config_mask, result_override)

    def test_result_moved_to_configured_device(self, manager):
        raw_movie = torch.randn(2, HEIGHT, WIDTH)
        result = manager.prepare_movie(movie=raw_movie)
        assert result.device == manager.computational_config.gpu_device

    def test_skip_movie_preparation_returns_movie_unchanged(
        self, computational_config, movie_config, output_config
    ):
        """`skip_movie_preparation=True` must bypass gain/dark/mask/mean-zero."""
        alignment_config = AlignFramesConfig(
            deformation_field_resolution=(3, 2, 2),
            skip_movie_preparation=True,
        )
        manager = _make_manager(
            computational_config, movie_config, output_config, alignment_config
        )
        raw_movie = torch.randn(3, HEIGHT, WIDTH)

        result = manager.prepare_movie(movie=raw_movie)

        assert torch.equal(result, raw_movie)

    def test_skip_movie_preparation_still_skips_disk_load_for_movie(
        self, computational_config, movie_config, output_config
    ):
        """Even when skipping preparation, an explicit movie must not be re-read."""
        movie_config.movie_path = "/nonexistent/path/should-never-be-read.mrc"
        alignment_config = AlignFramesConfig(
            deformation_field_resolution=(3, 2, 2),
            skip_movie_preparation=True,
        )
        manager = _make_manager(
            computational_config, movie_config, output_config, alignment_config
        )
        raw_movie = torch.randn(3, HEIGHT, WIDTH)

        result = manager.prepare_movie(movie=raw_movie)

        assert torch.equal(result, raw_movie)

    def test_crop_bounds_none_leaves_shape_unchanged(self, manager):
        """With crop_bounds=None (the default), the frame shape is untouched."""
        result = manager.prepare_movie()
        assert result.shape == (N_FRAMES, HEIGHT, WIDTH)

    def test_crop_bounds_crops_movie_gain_dark_and_mask(self, manager):
        """crop_bounds crops the movie (and loaded gain/dark/mask) consistently."""
        crop_bounds = {"min_y": 2, "max_y": 9, "min_x": 4, "max_x": 11}

        result = manager.prepare_movie(crop_bounds=crop_bounds)

        assert result.shape == (N_FRAMES, 8, 8)

    def test_crop_bounds_applies_to_explicit_movie_tensor(self, manager):
        """crop_bounds also crops an explicitly-provided movie tensor."""
        raw_movie = torch.randn(3, HEIGHT, WIDTH)
        crop_bounds = {"min_y": 0, "max_y": 7, "min_x": 0, "max_x": 7}

        result = manager.prepare_movie(movie=raw_movie, crop_bounds=crop_bounds)

        assert result.shape == (3, 8, 8)

    def test_skip_movie_preparation_ignores_crop_bounds(
        self, computational_config, movie_config, output_config
    ):
        """`skip_movie_preparation=True` bypasses cropping too."""
        alignment_config = AlignFramesConfig(
            deformation_field_resolution=(3, 2, 2),
            skip_movie_preparation=True,
        )
        manager = _make_manager(
            computational_config, movie_config, output_config, alignment_config
        )
        raw_movie = torch.randn(3, HEIGHT, WIDTH)
        crop_bounds = {"min_y": 0, "max_y": 7, "min_x": 0, "max_x": 7}

        result = manager.prepare_movie(movie=raw_movie, crop_bounds=crop_bounds)

        assert torch.equal(result, raw_movie)


# ---------------------------------------------------------------------------
# _setup_estimation_kwargs
# ---------------------------------------------------------------------------


class TestSetupEstimationKwargs:
    """Tests for AlignFramesManager._setup_estimation_kwargs."""

    def test_uses_explicit_field_over_config_default(self, manager):
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)
        sentinel = object()

        kwargs = manager._setup_estimation_kwargs(
            movie, initial_deformation_field=sentinel
        )

        assert kwargs["initial_deformation_field"] is sentinel
        assert kwargs["image"] is movie
        assert kwargs["pixel_spacing"] == manager.movie_config.pixel_size
        assert (
            kwargs["deformation_field_resolution"]
            == manager.alignment_config.deformation_field_resolution
        )
        assert kwargs["device"] == manager.computational_config.gpu_device

    def test_falls_back_to_config_default_when_none(self, manager):
        """With no explicit field and no configured path, falls back to None."""
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        kwargs = manager._setup_estimation_kwargs(movie, initial_deformation_field=None)

        assert kwargs["initial_deformation_field"] is None


# ---------------------------------------------------------------------------
# estimate_motion
# ---------------------------------------------------------------------------


class TestEstimateMotion:
    """Tests for AlignFramesManager.estimate_motion, with the backend mocked out."""

    def test_runs_xc_prepass_when_enabled_and_no_initial_field(
        self, manager, monkeypatch
    ):
        global_calls = []
        local_calls = []
        monkeypatch.setattr(
            afm_module,
            "estimate_global_motion",
            lambda **kw: global_calls.append(kw) or "GLOBAL_FIELD",
        )
        monkeypatch.setattr(
            afm_module,
            "estimate_local_motion",
            lambda **kw: local_calls.append(kw) or ("FINAL_FIELD", "TRAJECTORY"),
        )
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        result = manager.estimate_motion(movie)

        assert len(global_calls) == 1
        assert len(local_calls) == 1
        assert local_calls[0]["initial_deformation_field"] == "GLOBAL_FIELD"
        assert result == ("FINAL_FIELD", "TRAJECTORY")

    def test_skips_xc_prepass_when_explicit_initial_field_given(
        self, manager, monkeypatch
    ):
        global_calls = []
        local_calls = []
        monkeypatch.setattr(
            afm_module,
            "estimate_global_motion",
            lambda **kw: global_calls.append(kw) or "GLOBAL_FIELD",
        )
        monkeypatch.setattr(
            afm_module,
            "estimate_local_motion",
            lambda **kw: local_calls.append(kw) or ("FINAL_FIELD", "TRAJECTORY"),
        )
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        manager.estimate_motion(movie, initial_deformation_field="EXPLICIT_FIELD")

        assert len(global_calls) == 0
        assert local_calls[0]["initial_deformation_field"] == "EXPLICIT_FIELD"

    def test_skips_xc_prepass_when_disabled(
        self, computational_config, movie_config, output_config, monkeypatch
    ):
        alignment_config = AlignFramesConfig(
            deformation_field_resolution=(N_FRAMES, 2, 2),
            use_xc_prepass=False,
        )
        manager = _make_manager(
            computational_config, movie_config, output_config, alignment_config
        )
        global_calls = []
        local_calls = []
        monkeypatch.setattr(
            afm_module,
            "estimate_global_motion",
            lambda **kw: global_calls.append(kw) or "GLOBAL_FIELD",
        )
        monkeypatch.setattr(
            afm_module,
            "estimate_local_motion",
            lambda **kw: local_calls.append(kw) or ("FINAL_FIELD", "TRAJECTORY"),
        )
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        manager.estimate_motion(movie)

        assert len(global_calls) == 0
        assert local_calls[0]["initial_deformation_field"] is None

    def test_moves_movie_to_configured_device_before_estimation(
        self, manager, monkeypatch
    ):
        local_calls = []
        monkeypatch.setattr(afm_module, "estimate_global_motion", lambda **kw: None)
        monkeypatch.setattr(
            afm_module,
            "estimate_local_motion",
            lambda **kw: local_calls.append(kw) or ("FINAL_FIELD", "TRAJECTORY"),
        )
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        manager.estimate_motion(movie)

        assert local_calls[0]["image"].device == manager.computational_config.gpu_device

    def test_returns_backend_result_unchanged(self, manager, monkeypatch):
        monkeypatch.setattr(afm_module, "estimate_global_motion", lambda **kw: None)
        monkeypatch.setattr(
            afm_module,
            "estimate_local_motion",
            lambda **kw: ("FINAL_FIELD", "TRAJECTORY"),
        )
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        result = manager.estimate_motion(movie)

        assert result == ("FINAL_FIELD", "TRAJECTORY")


# ---------------------------------------------------------------------------
# correct_and_save
# ---------------------------------------------------------------------------


class TestCorrectAndSave:
    """Tests for AlignFramesManager.correct_and_save, with the backend mocked out."""

    def test_calls_correct_motion_with_expected_kwargs(self, manager, monkeypatch):
        correct_motion_calls = []
        monkeypatch.setattr(
            afm_module,
            "correct_motion",
            lambda **kw: correct_motion_calls.append(kw) or "CORRECTED_MOVIE",
        )
        monkeypatch.setattr(
            afm_module.manager_utils, "save_results", lambda *a, **k: None
        )
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        manager.correct_and_save(movie, "DEFORMATION_FIELD", "TRAJECTORY")

        assert len(correct_motion_calls) == 1
        call = correct_motion_calls[0]
        assert call["image"] is movie
        assert call["deformation_field"] == "DEFORMATION_FIELD"
        assert call["pixel_spacing"] == manager.movie_config.pixel_size
        assert call["device"] == manager.computational_config.gpu_device

    def test_calls_save_results_with_correct_motion_output(self, manager, monkeypatch):
        monkeypatch.setattr(
            afm_module, "correct_motion", lambda **kw: "CORRECTED_MOVIE"
        )
        save_results_calls = []
        monkeypatch.setattr(
            afm_module.manager_utils,
            "save_results",
            lambda *a, **k: save_results_calls.append((a, k)),
        )
        movie = torch.randn(N_FRAMES, HEIGHT, WIDTH)

        manager.correct_and_save(movie, "DEFORMATION_FIELD", "TRAJECTORY")

        assert len(save_results_calls) == 1
        args, kwargs = save_results_calls[0]
        assert kwargs == {"device": manager.computational_config.gpu_device}
        assert args[0] is manager.output_config
        assert args[1] is manager.movie_config
        assert args[2] == "CORRECTED_MOVIE"
        assert args[3] == "DEFORMATION_FIELD"
        assert args[4] is movie
        assert args[5] == "TRAJECTORY"
