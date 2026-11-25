"""Tests for the per-frame B-factor fit and the envelope correction."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import torch

# torch_fourier_filter.frame_weights is in teamtomo/teamtomo#203, not yet released
pytest.importorskip("torch_fourier_filter.frame_weights")

# pylint: disable=wrong-import-position
from torch_fourier_filter.dose_weight import critical_exposure
from torch_fourier_filter.frame_weights import EnvelopeCorrection

from ripple.config.frame_weights_config import (
    FrameWeightsMovie,
    TemplateEnvelopeSource,
)
from ripple.core import frame_weights as fw

NF, NB = 12, 16
K = np.linspace(1 / 40, 1 / 4, NB)  # 1/A
K2 = K**2
TRUE_B = -np.r_[60.0, 10.0, 0.0, 0.0, 5.0, 15.0, 30.0, 50.0, 80.0, 110.0, 150.0, 200.0]
TRUE_C = -np.r_[0.25, 0.05, 0.0, 0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3]
AMP = 2.0 * np.exp(-30 * K2)


def model(b=TRUE_B, c=TRUE_C):
    return AMP[None, :] * np.exp(b[:, None] * K2[None, :] / 4 + c[:, None])


class TestFitBFactors:
    def test_recovers_noiseless_model(self):
        se = np.full((NF, NB), 0.01)
        fit = fw.fit_bfactors(model(), se, K2)
        assert np.allclose(fit.bfactor, TRUE_B, atol=1e-3)
        assert np.allclose(fit.log_scale, TRUE_C, atol=1e-5)
        assert fit.bfactor.max() == 0 and fit.log_scale.max() == 0
        assert np.allclose(fit.model(K2), model(), rtol=1e-5)

    @pytest.mark.parametrize("ref", [0, 3, 11])
    def test_gauge_frame_does_not_change_the_result(self, ref):
        rng = np.random.default_rng(1)
        se = 0.02 * model().mean() * np.ones((NF, NB))
        data = model() + se * rng.standard_normal((NF, NB))
        a = fw.fit_bfactors(data, se, K2, reference_frame=ref)
        b = fw.fit_bfactors(data, se, K2, reference_frame=2)
        assert np.allclose(a.bfactor, b.bfactor, atol=0.05)
        assert np.allclose(a.log_scale, b.log_scale, atol=1e-3)

    def test_every_frame_moves_without_a_prior(self):
        """A frame far from any default is fitted where the data put it."""
        b = TRUE_B.copy()
        b[2] = -400.0  # frame 2 is badly damaged in this dataset
        fit = fw.fit_bfactors(model(b=b), np.full((NF, NB), 1e-3), K2)
        assert fit.bfactor[2] == pytest.approx(-400.0, abs=0.5)

    def test_prior_shrinks_towards_the_reference(self):
        rng = np.random.default_rng(2)
        se = 0.05 * model().mean() * np.ones((NF, NB))
        data = model() + se * rng.standard_normal((NF, NB))
        prior_b = np.linspace(0, -100, NF)
        prior_c = np.linspace(0, -0.5, NF)
        free = fw.fit_bfactors(data, se, K2, reference_frame=2)
        strong = fw.fit_bfactors(
            data,
            se,
            K2,
            reference_frame=2,
            prior=fw.BFactorPrior(prior_b, prior_c, sd_bfactor=1e-3, sd_log_scale=1e-6),
        )
        expected_b = prior_b - prior_b[2]
        expected_c = prior_c - prior_c[2]
        assert np.allclose(strong.bfactor, expected_b - expected_b.max(), atol=0.05)
        assert np.allclose(strong.log_scale, expected_c - expected_c.max(), atol=1e-4)
        assert not np.allclose(free.bfactor, strong.bfactor, atol=1.0)


def test_select_bins_skips_empty_bins():
    mean = np.ones((4, 8))
    se = np.ones((4, 8))
    mean[:, 2] = se[:, 2] = 0.0  # no Fourier samples in this bin
    mean[:, 6] = 0.0  # no signal
    use, _ = fw.select_bins(mean, se, np.linspace(0.01, 0.3, 8), 1.5, 0.0)
    assert use.tolist() == [0, 1, 3, 4, 5]


def test_select_bins_takes_a_contiguous_run():
    mean = np.ones((4, 8))
    se = np.ones((4, 8))
    mean[:, 5] = 0.0  # a gap
    use, snr = fw.select_bins(mean, se, np.linspace(0.01, 0.3, 8), 1.5, 0.02)
    assert use.tolist() == [1, 2, 3, 4]
    assert snr[0] == pytest.approx(2.0)


def _synthetic_pieces(n=300, n_pos=9, seed=0, noise=1.0):
    """Particles whose per-frame signal follows TRUE_B and TRUE_C at the centre."""
    rng = np.random.default_rng(seed)
    c = noise * rng.standard_normal((n, NF, n_pos, NB))
    c[:, :, n_pos // 2, :] += 3.0 * model()[None] / model().max()
    cov = np.zeros((n, NB, NF, NF))
    cov[:, :, np.arange(NF), np.arange(NF)] = noise**2
    return torch.as_tensor(c), torch.as_tensor(cov)


def test_frame_signal_analytic_sd_and_positions():
    c, cov = _synthetic_pieces(n=50)
    w = np.full((NF, NB), 1 / np.sqrt(NF))
    sig, var, sd = fw.frame_signal(c, cov, w)
    assert torch.allclose(sd, torch.full_like(sd, np.sqrt(NB)))
    assert torch.allclose(var, torch.full_like(var, 1 / NB))
    # the even frames' signal is read where the odd frames peak (the centre)
    best = torch.einsum("pfqb,fb->pq", c[:, 1::2], torch.as_tensor(w[1::2])).argmax(1)
    assert torch.equal(sig[:, 0] * sd[:, None], c[torch.arange(50), 0, best])


def test_sampled_null_matches_analytic_for_white_noise():
    c, cov = _synthetic_pieces(n=20)
    rng = np.random.default_rng(3)
    samples = torch.as_tensor(rng.standard_normal((20, 4000, NF, NB)))
    w = np.full((NF, NB), 1 / np.sqrt(NF))
    _, var_a, sd_a = fw.frame_signal(c, cov, w)
    _, var_s, sd_s = fw.frame_signal_sampled_null(c, samples, w)
    assert torch.allclose(sd_s, sd_a, rtol=0.05)
    assert torch.allclose(var_s.mean(0), var_a.mean(0), rtol=0.05)
    mean, cov_s = fw.random_null_statistics(samples)
    assert mean.abs().max() < 0.1
    assert torch.allclose(cov_s.mean(0), cov[0], atol=0.05)


def test_end_to_end_recovers_bfactors_within_bootstrap_errors():
    c, cov = _synthetic_pieces(n=400, seed=4)
    w = np.full((NF, NB), 1 / np.sqrt(NF))
    sig, var, _ = fw.frame_signal(c, cov, w)
    summary = fw.frame_signal_summary(sig.numpy(), var.numpy(), K, 3.0, 0.0)
    use = summary.bins_used
    fit = fw.fit_bfactors(
        summary.mean[:, use], summary.standard_error[:, use], K[use] ** 2
    )
    groups = fw.groups_from_labels(np.arange(400) // 4)  # groups of 4 particles
    boot_b, boot_c = fw.bootstrap_bfactors(
        summary.weighted_signal, use, K[use] ** 2, groups, 40, fit.reference_frame
    )
    assert boot_b.shape == (40, NF)
    sd = boot_b.std(0)
    assert np.all(np.abs(fit.bfactor - TRUE_B) < 4 * sd + 1.0)
    assert np.all(np.abs(fit.log_scale - TRUE_C) < 4 * boot_c.std(0) + 0.01)


def test_groups_from_labels_keeps_first_appearance_order():
    groups = fw.groups_from_labels(["b", "a", "b", "c", "a"])
    assert [g.tolist() for g in groups] == [[0, 2], [1, 4], [3]]


class TestEnvelope:
    def test_spline_matches_torch_fourier_filter(self):
        knots = fw.knot_frequencies(5, 50.0, 2.9)
        lv = np.array([0.0, -0.1, 0.05, 0.12, -0.2])
        k = np.linspace(0.0, 0.5, 200)
        ours = fw.spline_correction(k, knots, lv)
        ref = EnvelopeCorrection(torch.as_tensor(knots), torch.as_tensor(lv))(
            torch.as_tensor(k)
        )
        assert np.allclose(ours, ref.numpy())
        uniform = EnvelopeCorrection.uniform_knots(5, 50.0, 2.9)
        assert np.allclose(knots, uniform.knot_frequency.numpy())

    def test_fit_finds_a_better_envelope(self):
        """With signal v_b ~ g_b and null var ~ 1, the best envelope is h ~ g."""
        rng = np.random.default_rng(5)
        k = np.linspace(1 / 50, 1 / 3, 24)
        g = np.exp(-((np.sqrt(k) - 0.35) ** 2) / 0.02)
        v = torch.as_tensor(g[None, None, :] + 0.05 * rng.standard_normal((200, 1, 24)))
        coeffs = fw.SumCoefficients(v=v, var=torch.ones(200, 24, dtype=torch.float64))
        knots = fw.knot_frequencies(5, 50.0, 2.9)
        lv, info = fw.fit_envelope_correction(coeffs, np.ones(24), k, knots)
        assert lv[0] == 0.0
        assert info["mean_z_corrected"] > info["mean_z_base"] * 1.05
        h = fw.spline_correction(k, knots, lv)
        assert np.corrcoef(h, g)[0, 1] > 0.9

    def test_sum_coefficients_shapes_and_whitening(self):
        c, cov = _synthetic_pieces(n=10)
        w = np.full((NF, NB), 1 / np.sqrt(NF))
        a = fw.sum_coefficients(c, cov, w, np.ones(NB))
        b = fw.sum_coefficients(c, cov, w, 2 * np.ones(NB))
        assert a.v.shape == (10, 9, NB) and a.var.shape == (10, NB)
        # a constant whitening change does not change z
        assert torch.allclose(a.z(np.ones(NB)), b.z(np.ones(NB)))


class TestConfig:
    def test_ttsim3d_envelope_matches_formula(self):
        k = np.linspace(0, 0.5, 50)
        src = TemplateEnvelopeSource(
            kind="ttsim3d", dose_end=50.0, modify_signal="rel_diff"
        )
        ne = critical_exposure(torch.as_tensor(k)).numpy()
        cum = 2 * ne * -np.expm1(-25.0 / ne) / 50.0
        assert np.allclose(src.curve(k), 2 * cum / (1 + cum))
        assert src.curve(np.array([0.0]))[0] == pytest.approx(1.0)
        assert np.all(TemplateEnvelopeSource(kind="none").curve(k) == 1)

    def test_movie_needs_one_source(self):
        with pytest.raises(ValueError, match="either"):
            FrameWeightsMovie(movie_path="m.tif")
        with pytest.raises(ValueError, match="either"):
            FrameWeightsMovie(
                movie_path="m.tif",
                deformation_field_path="f.csv",
                aligned_movie_path="a.mrc",
            )

    def test_movie_matches_particles_by_stem(self):
        movie = FrameWeightsMovie(
            movie_path="/x/mic_01.tif", deformation_field_path="f.csv"
        )
        paths = np.array(
            [
                "/out/mic_01_dw_sum.mrc",
                "/out/mic_011_dw_sum.mrc",
                "/o/mic_01.mrc",
                "/o/b.mrc",
            ]
        )
        assert movie.matches(paths).tolist() == [True, False, True, False]
        explicit = FrameWeightsMovie(
            aligned_movie_path="/x/a.mrc", micrograph_path="/o/b.mrc"
        )
        assert explicit.matches(paths).tolist() == [False, False, False, True]
