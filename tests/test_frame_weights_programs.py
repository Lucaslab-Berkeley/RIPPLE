"""Tests for programs/frame_weights (movie discovery) and the B-factor plot."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROGRAMS_DIR = Path(__file__).resolve().parents[1] / "programs"


def _load_script(relative_path: str):
    path = PROGRAMS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_movies_from_directory_pairs_movies_with_fields(tmp_path, capsys):
    pytest.importorskip("torch_fourier_filter.frame_weights")
    script = _load_script("frame_weights/run_estimate_frame_weights.py")
    movies = tmp_path / "movies"
    movies.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    for name in ("a.tif", "b.EER", "c.mrc", "notes.txt"):
        (movies / name).write_text("")
    (out / "a_def_field.csv").write_text("")
    (out / "b_def_field.csv").write_text("")
    found = script.movies_from_directory(str(movies), str(out / "def_field.csv"))
    assert [m.stem for m in found] == ["a", "b"]
    assert found[0].deformation_field_path == str(out / "a_def_field.csv")
    assert "Skipping c.mrc" in capsys.readouterr().out


def test_example_config_parses(tmp_path, monkeypatch):
    pytest.importorskip("torch_fourier_filter.frame_weights")
    import yaml  # pylint: disable=import-outside-toplevel

    from ripple.managers import (  # pylint: disable=import-outside-toplevel
        EstimateFrameWeightsManager,
    )

    path = PROGRAMS_DIR / "frame_weights" / "estimate_frame_weights_example_config.yaml"
    config = yaml.safe_load(path.read_text())
    config["output"]["output_dir"] = str(tmp_path / "out")
    monkeypatch.chdir(tmp_path)
    manager = EstimateFrameWeightsManager(**config)
    assert manager.estimation.prior is None  # shrinkage is off by default
    assert manager.estimation.null_model == "analytic"


def test_plot_writes_a_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from ripple.utils.plot_frame_weights import (  # pylint: disable=import-outside-toplevel
        plot_frame_bfactors,
    )

    nf, nb = 12, 20
    k = np.linspace(0.02, 0.3, nb)
    b = -np.linspace(0, 200, nf)
    c = -np.linspace(0, 0.4, nf)
    mean = np.exp(b[:, None] * k[None] ** 2 / 4 + c[:, None])
    rng = np.random.default_rng(0)
    npz = tmp_path / "signal.npz"
    np.savez(
        npz,
        mean=mean,
        standard_error=0.05 * mean,
        frequency=k,
        bins_used=np.arange(2, nb),
        amplitude=np.ones(nb - 2),
        bfactor=b,
        log_scale=c,
        bootstrap_bfactor=b + rng.normal(0, 5, (20, nf)),
        bootstrap_log_scale=c + rng.normal(0, 0.02, (20, nf)),
        gg_weights=mean,
        gg_bfactor=b,
        gg_log_scale=c,
        default_bfactor=b,
        default_log_scale=c,
        envelope_frequency=k,
        envelope_relative=np.exp(-10 * k**2),
    )
    out = plot_frame_bfactors(npz, tmp_path / "plot.png", title="test")
    assert out.exists() and out.stat().st_size > 1000
