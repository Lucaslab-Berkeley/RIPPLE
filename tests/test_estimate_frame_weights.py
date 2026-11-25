"""End-to-end test of EstimateFrameWeightsManager on a synthetic movie (CPU).

Particles are pasted into a movie with Leopard-EM's own template projections (the
template-side filter included), each frame damaged by a known
``exp(B_f k^2 / 4 + C_f)``, plus white noise. The estimator should recover B_f and
C_f, and its correlation pieces should reproduce Leopard-EM's correlation.
"""

# pylint: disable=redefined-outer-name

import json

import mrcfile
import numpy as np
import pandas as pd
import pytest
import torch

# torch_fourier_filter.frame_weights is in teamtomo/teamtomo#203, not yet released
pytest.importorskip("torch_fourier_filter.frame_weights")

# pylint: disable=wrong-import-position
from torch_fourier_filter.frame_weights import (
    FrameWeightTable,
    TemplateEnvelope,
)

from ripple.core import frame_weights as fw
from ripple.core.particle_pieces import (
    particle_template_filter,
    rfft_grid_bins,
    template_projections,
)
from ripple.managers import EstimateFrameWeightsManager

PIXEL = 2.0
N_FRAMES, SIZE, T, WINDOW = 8, 1024, 48, 3
BOX = T + 2 * WINDOW
TRUE_B = -np.array([20.0, 0.0, 5.0, 20.0, 45.0, 80.0, 125.0, 180.0])
TRUE_C = -np.array([0.15, 0.0, 0.02, 0.05, 0.08, 0.11, 0.14, 0.17])


def _template(path):
    """An atom-like template: narrow Gaussians at random points of a ball.

    Its spectrum is broad, as a real map's: a frame's box is then noise-dominated at
    every frequency, as in real data. (The analytic null takes each frame's box power
    as noise; a template whose power sits at low frequency, with a strong signal,
    would make that power mostly signal and compress the fitted scales. The study's
    random-orientation null does the same.)
    """
    rng = np.random.default_rng(3)
    g = np.mgrid[:T, :T, :T].astype(np.float32) - T / 2
    vol = np.zeros((T, T, T), np.float32)
    atoms = rng.normal(0, T / 7, (400, 3))
    atoms = atoms[np.linalg.norm(atoms, axis=1) < T / 2.5]
    for z, y, x in atoms:
        vol += np.exp(-((g[0] - z) ** 2 + (g[1] - y) ** 2 + (g[2] - x) ** 2) / 2.0)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(vol)
        mrc.voxel_size = PIXEL


def _particles(path, micrograph, template):
    rng = np.random.default_rng(0)
    step = range(8, SIZE - BOX - 8, 60)
    pos = [(y, x) for y in step for x in step]
    n = len(pos)
    angles = rng.uniform([0, 0, 0], [360, 180, 360], (n, 3))
    df = pd.DataFrame(
        {
            "particle_index": np.arange(n),
            "mip": 10.0,
            "scaled_mip": np.linspace(12, 8, n),
            "refined_mip": 10.0,
            "refined_scaled_mip": np.linspace(12, 8, n),
            "correlation_mean": 0.0,
            "correlation_variance": 1.0,
            "total_correlations": 1,
            "pos_y": [p[0] for p in pos],
            "pos_x": [p[1] for p in pos],
            "pos_y_img": [p[0] + T // 2 for p in pos],
            "pos_x_img": [p[1] + T // 2 for p in pos],
            "pos_y_img_angstrom": [(p[0] + T // 2) * PIXEL for p in pos],
            "pos_x_img_angstrom": [(p[1] + T // 2) * PIXEL for p in pos],
            "phi": angles[:, 0],
            "theta": angles[:, 1],
            "psi": angles[:, 2],
            "relative_defocus": 0.0,
            "defocus_u": 600.0,
            "defocus_v": 550.0,
            "astigmatism_angle": 10.0,
            "pixel_size": PIXEL,
            "voltage": 300.0,
            "spherical_aberration": 2.7,
            "amplitude_contrast_ratio": 0.07,
            "phase_shift": 0.0,
            "ctf_B_factor": 0.0,
            "micrograph_path": str(micrograph),
            "template_path": str(template),
            "refined_pixel_size": PIXEL,
            "even_zernikes": np.nan,
            "odd_zernikes": np.nan,
            "mag_matrix": np.nan,
        }
    )
    for col in ("phi", "theta", "psi", "pos_x", "pos_y", "relative_defocus"):
        df[f"refined_{col}"] = df[col]  # as a refine_template result
    for col in ("pos_x_img", "pos_y_img", "pos_x_img_angstrom", "pos_y_img_angstrom"):
        df[f"refined_{col}"] = df[col]
    for name in ("mip", "scaled_mip", "psi", "theta", "phi", "defocus"):
        df[f"{name}_path"] = "unused.mrc"
    df["correlation_average_path"] = "unused.mrc"
    df["correlation_variance_path"] = "unused.mrc"
    df.to_csv(path)
    return df


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """A movie, particles and template; returns the manager kwargs."""
    tmp = tmp_path_factory.mktemp("synthetic")
    template = tmp / "template.mrc"
    _template(template)
    gen = torch.Generator().manual_seed(1)
    noise_sd = 1.0
    frames = noise_sd * torch.randn(N_FRAMES, SIZE, SIZE, generator=gen)
    movie = tmp / "aligned.mrc"
    micrograph = tmp / "movie_dw_sum.mrc"  # the particles' micrograph (not read)
    particles = tmp / "particles.csv"
    df = _particles(particles, micrograph, template)

    def kwargs(out):
        return {
            "computational_config": {"gpu_id": "cpu"},
            "movie_config": {
                "pixel_size": PIXEL,
                "fluence": N_FRAMES * 2.0,
                "fluence_per_frame": 2.0,
                "voltage": 300.0,
            },
            "movies": [
                {"aligned_movie_path": str(movie), "micrograph_path": str(micrograph)}
            ],
            "particles_path": str(particles),
            "template_path": str(template),
            "estimation": {"n_bootstrap": 30, "min_snr": 4.0, "n_verify": 2},
            "output": {"output_dir": str(out), "allow_file_overwrite": True},
        }

    # Leopard-EM's template-side filters for the noise-only sum, to synthesise with
    with mrcfile.new(str(movie), overwrite=True) as mrc:
        mrc.set_data(frames.numpy())
    manager = EstimateFrameWeightsManager(**kwargs(tmp / "setup"))
    sum_path = tmp / "noise_sum.mrc"
    with mrcfile.new(str(sum_path), overwrite=True) as mrc:
        mrc.set_data(frames.sum(0).numpy())
        mrc.voxel_size = PIXEL
    lkw = manager.leopard_kwargs(df, str(sum_path), tmp / "setup.csv", T)
    _, _, ky, kx = rfft_grid_bins(BOX, fw.radial_bin_edges(8))
    k2 = (ky[:, None] ** 2 + kx[None, :] ** 2) / (BOX * PIXEL) ** 2
    amp = torch.exp(
        torch.as_tensor(TRUE_B)[:, None, None] * k2[None] / 4
        + torch.as_tensor(TRUE_C)[:, None, None]
    ).float()
    for i in range(len(df)):
        filt, rot = particle_template_filter(lkw, i, "cpu")
        pp = template_projections(lkw, rot, filt, BOX)[0]
        img = torch.fft.irfftn(pp[None] * amp, s=(BOX, BOX), dim=(-2, -1))
        # per-pixel signal 0.2 of the noise in the best frame
        img = img * (0.2 * noise_sd / img[1, :T, :T].std())
        y, x = int(df.pos_y[i]), int(df.pos_x[i])
        frames[:, y : y + BOX, x : x + BOX] += img
    with mrcfile.new(str(movie), overwrite=True) as mrc:
        mrc.set_data(frames.numpy())
    return tmp, kwargs


def test_estimator_recovers_frame_bfactors(synthetic):
    tmp, kwargs = synthetic
    out = tmp / "out"
    diag = EstimateFrameWeightsManager(**kwargs(out)).run()

    checks = diag["log"]["movies"][0]["checks"]
    assert max(checks["cc_rel_err"]) < 1e-4  # pieces reproduce Leopard-EM's CC
    # the frames are whitened with the radial profile of Leopard-EM's template-side
    # filter, which a 32-pixel template samples coarsely (0.99998 on 512-px templates)
    assert min(checks["sum_box_vs_leopard_corr"]) > 0.95
    bf = np.asarray(diag["bfactor"])
    sd = np.asarray(diag["bfactor_sd"])
    truth_b = TRUE_B - TRUE_B.max()
    assert np.all(np.abs(bf - truth_b) < 4 * sd + 3.0), (bf, truth_b, sd)
    scale = np.asarray(diag["scale"])
    truth_s = np.exp(TRUE_C - TRUE_C.max())
    assert np.all(np.abs(scale - truth_s) < 4 * np.asarray(diag["scale_sd"]) + 0.02)

    table = FrameWeightTable.from_json(out / "frame_weights.json")
    assert table.n_frames == N_FRAMES and table.source == "estimated"
    assert np.allclose(table.bfactor.numpy(), bf)
    assert table.provenance["particles"] == diag["n_particles"]
    envelope = TemplateEnvelope.from_json(out / "template_envelope.json")
    assert envelope.correction is not None
    assert len(envelope.correction.log_values) == 5
    info = diag["envelope"]
    assert info["mean_z_matched_corrected"] >= info["mean_z_matched"] - 1e-9
    with open(out / "frame_weights_diagnostics.json", encoding="utf-8") as f:
        assert json.load(f)["frames"] == N_FRAMES


def test_existing_outputs_are_not_overwritten(synthetic):
    tmp, kwargs = synthetic
    out = tmp / "guarded"
    out.mkdir()
    (out / "frame_weights.json").write_text("{}")
    config = kwargs(out)
    config["output"]["allow_file_overwrite"] = False
    with pytest.raises(ValueError, match="allow_file_overwrite"):
        EstimateFrameWeightsManager(**config).run()


def test_no_matching_movie_raises(synthetic):
    tmp, kwargs = synthetic
    config = kwargs(tmp / "nomatch")
    config["movies"][0]["micrograph_path"] = str(tmp / "other.mrc")
    with pytest.raises(ValueError, match="no particle matched"):
        EstimateFrameWeightsManager(**config).run()


def test_leopard_import_keeps_autograd_on():
    """Leopard-EM's backend switches autograd off when imported; RIPPLE restores it."""
    from ripple.core.particle_pieces import keep_grad_mode

    with torch.enable_grad():
        with keep_grad_mode():
            torch.set_grad_enabled(False)  # what Leopard-EM's import does
        assert torch.is_grad_enabled()
