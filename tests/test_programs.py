"""Tests for the example scripts in `programs/` (movie globbing, skipping, rendering).

The alignment and beam mask managers are replaced by cheap fakes in the directory
script tests, so no motion estimation runs here.
"""

# pylint: disable=redefined-outer-name,too-few-public-methods

import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest
import torch
import yaml

from ripple.config import OutputConfig

PROGRAMS_DIR = Path(__file__).resolve().parents[1] / "programs"
N_FRAMES = 4
HEIGHT = 16
WIDTH = 16


def _load_script(relative_path: str):
    path = PROGRAMS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def align_script():
    return _load_script("align_movie/run_align_movies.py")


@pytest.fixture(scope="module")
def decolace_script():
    return _load_script("decolace_pipeline/run_decolace_pipeline.py")


@pytest.fixture(scope="module")
def render_script():
    return _load_script("align_movie/render_movie_to_mrc.py")


def _write_mrc(path: Path, array: np.ndarray) -> str:
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(array)
    return str(path)


def _movie_array(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N_FRAMES, HEIGHT, WIDTH)) + 10.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Movie globbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script_name", ["align_script", "decolace_script"])
def test_get_movie_paths_accepts_all_movie_extensions_any_case(
    tmp_path, request, script_name
):
    script = request.getfixturevalue(script_name)
    names = ["a.eer", "b.EER", "c.mrc", "d.MRCS", "e.tif", "f.TIFF", "g.tiff"]
    for name in [*names, "notes.txt", "gain.gain", "h.mrc.bak"]:
        (tmp_path / name).write_text("")

    result = script.get_movie_paths(str(tmp_path))

    assert result == [str(tmp_path / name) for name in names]


# ---------------------------------------------------------------------------
# Skipping movies whose outputs already exist
# ---------------------------------------------------------------------------


class _RecordingAlignManager:
    """Stand-in for AlignFramesManager that records which movies it processed."""

    def __init__(self, output_config, movie_config=None, **_):
        self.output_config = output_config
        self.movie_config = movie_config or SimpleNamespace(movie_path=None)
        self.processed = []
        self._previous_prepared = None

    def prepare_movie(self, **_):
        self.output_config.check_output_paths()
        # The script must have freed the previous movie before preparing the next
        assert self._previous_prepared is None or self._previous_prepared() is None
        self.processed.append(self.movie_config.movie_path)
        prepared = torch.zeros(1)
        self._previous_prepared = weakref.ref(prepared)
        return prepared

    def estimate_motion(self, prepared):
        return "FIELD", "TRAJECTORY"

    def correct_and_save(self, prepared, deformation_field, trajectory):
        Path(self.output_config.deformation_field_output_path).write_text("done")


def _template_output_config(out_dir: Path) -> OutputConfig:
    return OutputConfig(
        allow_file_overwrite=False,
        dw_sum_output_path=str(out_dir / "dw_sum.mrc"),
        deformation_field_output_path=str(out_dir / "field.csv"),
    )


def test_align_script_skips_movies_with_existing_outputs(
    tmp_path, align_script, monkeypatch, capsys
):
    movie_dir = tmp_path / "movies"
    movie_dir.mkdir()
    for name in ("a.mrc", "b.mrc", "c.mrc"):
        (movie_dir / name).write_text("")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "b_field.csv").write_text("existing")

    fake = _RecordingAlignManager(_template_output_config(out_dir))
    monkeypatch.setattr(align_script, "MOVIE_DIR", str(movie_dir))
    monkeypatch.setattr(
        align_script,
        "AlignFramesManager",
        SimpleNamespace(from_yaml=lambda _path: fake),
    )

    align_script.main()

    assert fake.processed == [str(movie_dir / "a.mrc"), str(movie_dir / "c.mrc")]
    assert (out_dir / "b_field.csv").read_text() == "existing"
    assert f"Skipping {movie_dir / 'b.mrc'}" in capsys.readouterr().out


def test_decolace_script_skips_movies_with_existing_outputs(
    tmp_path, decolace_script, monkeypatch, capsys
):
    movie_dir = tmp_path / "movies"
    movie_dir.mkdir()
    for index, name in enumerate(("a.mrc", "b.mrc")):
        _write_mrc(movie_dir / name, _movie_array(index))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "a_dw_sum.mrc").write_text("existing")

    config = {
        "computational_config": {"gpu_id": "cpu"},
        "movie_config": {
            "pixel_size": 1.0,
            "fluence": 40.0,
            "fluence_per_frame": 10.0,
        },
        "beam_mask_config": {},
        "output_config": _template_output_config(out_dir).model_dump(),
        "alignment_config": {"deformation_field_resolution": [N_FRAMES, 2, 2]},
    }
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(yaml.safe_dump(config))

    managers = []

    def _make_align_manager(**kwargs):
        managers.append(_RecordingAlignManager(**kwargs))
        return managers[-1]

    class _FakeBeamMaskManager:
        def __init__(self, **_):
            pass

        def estimate_with_mask(self, movie):
            mask = torch.ones(movie.shape[-2:], dtype=torch.bool)
            return SimpleNamespace(output_crop_bounds=None), mask

    monkeypatch.setattr(decolace_script, "PIPELINE_YAML_PATH", str(yaml_path))
    monkeypatch.setattr(decolace_script, "MOVIE_DIR", str(movie_dir))
    monkeypatch.setattr(decolace_script, "AlignFramesManager", _make_align_manager)
    monkeypatch.setattr(decolace_script, "BeamMaskManager", _FakeBeamMaskManager)

    decolace_script.main()

    assert managers[0].processed == [str(movie_dir / "b.mrc")]
    assert (out_dir / "a_dw_sum.mrc").read_text() == "existing"
    assert f"Skipping {movie_dir / 'a.mrc'}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# render_movie_to_mrc.py
# ---------------------------------------------------------------------------


@pytest.fixture
def render_config(tmp_path) -> Path:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mask[2:10, 4:14] = 1.0
    (tmp_path / "unrelated_dw_sum.mrc").write_text("existing")
    config = {
        "computational_config": {"gpu_id": "cpu"},
        "movie_config": {
            "movie_path": _write_mrc(tmp_path / "movie.mrc", _movie_array()),
            "pixel_size": 1.5,
            "fluence": 40.0,
            "fluence_per_frame": 10.0,
            "gain_path": _write_mrc(
                tmp_path / "gain.mrc", np.full((HEIGHT, WIDTH), 2.0, np.float32)
            ),
            "dark_path": _write_mrc(
                tmp_path / "dark.mrc", np.full((HEIGHT, WIDTH), 0.5, np.float32)
            ),
            "mask_path": _write_mrc(tmp_path / "mask.mrc", mask),
            "crop_bounds_config": {"mode": "tight"},
        },
        "output_config": {
            "allow_file_overwrite": False,
            "rendered_movie_output_path": str(tmp_path / "rendered.mrc"),
            "dw_sum_output_path": str(tmp_path / "unrelated_dw_sum.mrc"),
        },
        "alignment_config": {"deformation_field_resolution": [N_FRAMES, 2, 2]},
    }
    yaml_path = tmp_path / "align.yaml"
    yaml_path.write_text(yaml.safe_dump(config))
    return yaml_path


def test_render_script_uses_manager_preparation_and_pixel_size(
    tmp_path, render_script, render_config
):
    manager = render_script.load_manager_from_yaml(str(render_config))
    expected = manager.prepare_movie()

    render_script.render_movie_to_mrc(manager)

    with mrcfile.open(str(tmp_path / "rendered.mrc")) as mrc:
        rendered = torch.from_numpy(mrc.data.copy())
        assert mrc.voxel_size.x == pytest.approx(1.5, abs=1e-5)
    # Mask crop bounds were applied (mask covers rows 2..9, columns 4..13)
    assert rendered.shape == (N_FRAMES, 8, 10)
    assert torch.allclose(rendered, expected, atol=1e-6)
    assert (tmp_path / "unrelated_dw_sum.mrc").read_text() == "existing"


def test_render_script_honours_allow_file_overwrite(
    tmp_path, render_script, render_config
):
    (tmp_path / "rendered.mrc").write_text("existing")

    with pytest.raises(ValueError, match="already exists"):
        render_script.render_movie_to_mrc(
            render_script.load_manager_from_yaml(str(render_config))
        )
    assert (tmp_path / "rendered.mrc").read_text() == "existing"
