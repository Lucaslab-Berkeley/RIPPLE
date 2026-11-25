"""Tests for ResumManager and programs/align_movie/run_resum.py."""

# pylint: disable=redefined-outer-name

import importlib.util
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
from torch_motion_correction import DeformationField, correct_motion

from ripple.config import ComputationalConfig, MovieConfig, OutputConfig
from ripple.core.dose_weighting import movie_frame_weight_table
from ripple.core.generate_image import generate_dose_weighted_image
from ripple.managers import ResumManager, manager_utils

PROGRAMS_DIR = Path(__file__).resolve().parents[1] / "programs"
N_FRAMES, H, W = 6, 32, 28

# torch_motion_correction's DeformationField.from_csv wraps a read-only pandas array
pytestmark = pytest.mark.filterwarnings(
    "ignore:The given NumPy array is not writable:UserWarning"
)


def _write_movie(path: Path, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    data = rng.poisson(3.0, size=(N_FRAMES, H, W)).astype(np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data)
    return str(path)


def _field(seed: int = 0) -> DeformationField:
    gen = torch.Generator().manual_seed(seed)
    data = 2.0 * torch.randn(2, N_FRAMES, 2, 2, generator=gen)
    return DeformationField(data=data, grid_type="catmull_rom")


def _movie_config(movie_path: str, **kw) -> MovieConfig:
    return MovieConfig(
        movie_path=movie_path,
        pixel_size=1.1,
        fluence=N_FRAMES * 1.5,
        fluence_per_frame=1.5,
        voltage=300.0,
        **kw,
    )


def _align_outputs(tmp_path, movie_path, field, stem="movie"):
    """What an alignment run writes: the field (CSV) and its G&G dw_sum."""
    movie_config = _movie_config(movie_path)
    computational = ComputationalConfig(gpu_id="cpu")
    movie, gain, dark, mask = manager_utils.load_missing_tensors(
        computational, movie_config
    )
    prepared = movie_config.prepare(movie, gain, dark, mask=mask, device="cpu")
    corrected = correct_motion(
        image=prepared, deformation_field=field, pixel_spacing=1.1, device="cpu"
    )
    field_path = tmp_path / f"{stem}_def_field.csv"
    out = OutputConfig(
        dw_sum_output_path=str(tmp_path / f"{stem}_dw_sum.mrc"),
        deformation_field_output_path=str(field_path),
    )
    manager_utils.save_results(out, movie_config, corrected, field, prepared, None)
    return field_path, tmp_path / f"{stem}_dw_sum.mrc", corrected


def _read(path) -> torch.Tensor:
    with mrcfile.open(str(path)) as mrc:
        return torch.as_tensor(mrc.data.copy())


def test_gg_resum_reproduces_the_alignment_dw_sum(tmp_path):
    movie_path = _write_movie(tmp_path / "movie.mrc")
    field_path, dw_sum_path, _ = _align_outputs(tmp_path, movie_path, _field())
    manager = ResumManager(
        computational_config=ComputationalConfig(gpu_id="cpu"),
        movie_config=_movie_config(movie_path),
        deformation_field_path=str(field_path),
        output_config=OutputConfig(
            dw_sum_output_path=str(tmp_path / "resum_gg.mrc"), dose_weighting="gg"
        ),
    )
    manager.run()
    resum, original = _read(tmp_path / "resum_gg.mrc"), _read(dw_sum_path)
    # the CSV field is saved as text, so equality is up to its precision
    corr = np.corrcoef(resum.ravel().numpy(), original.ravel().numpy())[0, 1]
    assert corr > 1 - 1e-6
    assert torch.allclose(resum, original, atol=1e-4 * original.abs().max())


def test_default_resum_uses_the_default_table(tmp_path):
    # torch_fourier_filter.frame_weights is in teamtomo/teamtomo#203, not yet released
    pytest.importorskip("torch_fourier_filter.frame_weights")
    movie_path = _write_movie(tmp_path / "movie.mrc")
    field_path, _, corrected = _align_outputs(tmp_path, movie_path, _field())
    manager = ResumManager(
        computational_config=ComputationalConfig(gpu_id="cpu"),
        movie_config=_movie_config(movie_path),
        deformation_field_path=str(field_path),
        output_config=OutputConfig(
            dw_sum_output_path=str(tmp_path / "resum_default.mrc"),
            non_dw_sum_output_path=str(tmp_path / "resum_plain.mrc"),
            dose_weighting="default",
        ),
    )
    manager.run()
    table = movie_frame_weight_table("default", N_FRAMES, 0.0, 1.5, 300.0)
    expected = generate_dose_weighted_image(
        corrected, 1.1, 0.0, 1.5, 300.0, frame_weights=table
    )
    resum = _read(tmp_path / "resum_default.mrc")
    assert torch.allclose(resum, expected, atol=1e-4 * expected.abs().max())
    plain = _read(tmp_path / "resum_plain.mrc")
    assert torch.allclose(plain, corrected.sum(0), atol=1e-4 * plain.abs().max())


@pytest.mark.parametrize(
    "field",
    [
        "deformation_field_output_path",
        "loss_trajectories_output_path",
        "rendered_movie_output_path",
    ],
)
def test_alignment_only_outputs_are_refused(tmp_path, field):
    with pytest.raises(ValueError, match="unset"):
        ResumManager(
            movie_config=_movie_config("movie.mrc"),
            deformation_field_path="field.csv",
            output_config=OutputConfig(**{field: str(tmp_path / "x")}),
        )


def _load_script(relative_path: str):
    path = PROGRAMS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resum_directory_prefixes_stems_and_skips(tmp_path, capsys):
    script = _load_script("align_movie/run_resum.py")
    movie_dir = tmp_path / "movies"
    movie_dir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for i, stem in enumerate(("a", "b", "c")):
        movie_path = _write_movie(movie_dir / f"{stem}.mrc", seed=i)
        if stem != "c":  # c has no saved field
            _align_outputs(out, movie_path, _field(i), stem=stem)
    (out / "b_fw_sum.mrc").write_text("existing")
    manager = ResumManager(
        computational_config=ComputationalConfig(gpu_id="cpu"),
        movie_config=_movie_config(None),
        deformation_field_path=str(out / "def_field.csv"),
        output_config=OutputConfig(
            allow_file_overwrite=False,
            dw_sum_output_path=str(out / "fw_sum.mrc"),
            dose_weighting="gg",
        ),
    )
    script.resum_directory(manager, str(movie_dir))
    printed = capsys.readouterr().out
    assert (out / "a_fw_sum.mrc").exists()
    assert (out / "b_fw_sum.mrc").read_text() == "existing"
    assert not (out / "c_fw_sum.mrc").exists()
    assert "no deformation field" in printed and "already exist" in printed
    corr = np.corrcoef(
        _read(out / "a_fw_sum.mrc").ravel().numpy(),
        _read(out / "a_dw_sum.mrc").ravel().numpy(),
    )[0, 1]
    assert corr > 1 - 1e-6
