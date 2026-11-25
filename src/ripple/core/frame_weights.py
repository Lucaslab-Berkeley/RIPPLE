"""Per-frame B-factors and a template envelope correction, estimated from particles.

This is the fitting half of RIPPLE's ``estimate_frame_weights`` program; the
correlation pieces it fits come from :mod:`ripple.core.particle_pieces`.

For each particle i, frame f, radial bin b and position q in a small window around
the particle, ``c[i, f, q, b]`` is the template correlation restricted to the bin.
The particle's null (its correlation with the template where there is no particle)
has covariance ``noise_cov[i, b]`` between frames, here computed analytically from the
frames' power and the template's power per bin.

* :func:`frame_signal`: each frame's signal per bin, in units of the particle's null
  standard deviation. The position for the odd frames is chosen on the even frames
  and the reverse, so the maximum picks up no noise.
* :func:`frame_signal_summary`: the mean over particles, weighted as the best
  per-frame weights are (s / sigma^2), with standard errors, and the bins where the
  summed signal is clear.
* :func:`fit_bfactors`: RELION's model ``s[f, b] = A_b exp(B_f k_b^2 / 4 + C_f)``
  by weighted least squares, every frame free; an optional Gaussian prior shrinks
  each frame's B and C towards a reference table (off by default).
* :func:`bootstrap_bfactors`: errors from resampling particles or groups of them.
* :func:`fit_envelope_correction`: a smooth 5-knot correction to the template's
  matched envelope, chosen to maximise the particles' mean z on the weighted sum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped, unused-ignore]
from scipy.optimize import (  # type: ignore[import-untyped, unused-ignore]
    least_squares,
    minimize,
)
from scipy.sparse import csr_matrix  # type: ignore[import-untyped, unused-ignore]

KMAX_CORNER = 0.5 * math.sqrt(2.0)  # the corner of a 2D spectrum, cycles / px


def radial_bin_edges(n_bins: int = 48, kmax: float = KMAX_CORNER) -> np.ndarray:
    """Radial bin edges uniform in sqrt(|k|), in cycles / px, out to the corner.

    Parameters
    ----------
    n_bins : int
        Number of bins. Default 48.
    kmax : float
        Upper edge, cycles / px. Default the corner of the spectrum.

    Returns
    -------
    np.ndarray
        ``(n_bins + 1,)`` edges.
    """
    return np.asarray(kmax * np.linspace(0.0, 1.0, n_bins + 1) ** 2)


def bin_centres(edges: np.ndarray, pixel_size: float) -> np.ndarray:
    """Bin centres in 1/A."""
    edges = np.asarray(edges, float)
    return np.asarray(0.5 * (edges[:-1] + edges[1:]) / pixel_size)


def _as_tensor(x: Any, like: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=like.dtype, device=like.device)
    return torch.as_tensor(np.asarray(x), dtype=like.dtype, device=like.device)


def best_position_signal(
    c: torch.Tensor, ref_weights: np.ndarray | torch.Tensor
) -> torch.Tensor:
    """Each frame's correlation per bin at a position chosen on the other frames.

    The position for the odd frames is the best (under `ref_weights`) on the even
    frames, and the reverse, so the maximum picks up no noise of the frames it is
    read from (``dw_bfactor.frame_signal`` of the study this follows).

    Parameters
    ----------
    c : torch.Tensor
        ``(n, n_frames, n_positions, n_bins)`` correlation pieces.
    ref_weights : array-like
        ``(n_frames, n_bins)`` reference frame weights.

    Returns
    -------
    torch.Tensor
        ``(n, n_frames, n_bins)`` float64.
    """
    cc = c.double()
    nf = cc.shape[1]
    w = _as_tensor(ref_weights, cc)
    even = torch.arange(0, nf, 2, device=cc.device)
    odd = torch.arange(1, nf, 2, device=cc.device)
    best_for_odd = torch.einsum("pfqb,fb->pq", cc[:, even], w[even]).argmax(1)
    best_for_even = torch.einsum("pfqb,fb->pq", cc[:, odd], w[odd]).argmax(1)
    rows = torch.arange(cc.shape[0], device=cc.device)
    sig = torch.empty(cc.shape[0], nf, cc.shape[-1], dtype=cc.dtype, device=cc.device)
    sig[:, even] = cc[:, even][rows, :, best_for_even]
    sig[:, odd] = cc[:, odd][rows, :, best_for_odd]
    return sig


def frame_signal(
    c: torch.Tensor,
    noise_cov: torch.Tensor,
    ref_weights: np.ndarray | torch.Tensor,
    null_mean: torch.Tensor | None = None,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-particle signal per frame and bin, in units of each particle's null sd.

    Parameters
    ----------
    c : torch.Tensor
        ``(n, n_frames, n_positions, n_bins)`` correlation pieces.
    noise_cov : torch.Tensor
        ``(n, n_bins, n_frames, n_frames)`` null covariance of the pieces between
        frames, per particle and bin (bins independent).
    ref_weights : array-like
        ``(n_frames, n_bins)`` reference frame weights (for example G&G's), used to
        choose each particle's position (:func:`best_position_signal`) and to define
        its null sd.
    null_mean : torch.Tensor | None
        ``(n, n_frames, n_bins)`` null mean to subtract, or None for zero.
    chunk : int
        Particles processed at a time.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(signal, variance, sd)``: ``(n, n_frames, n_bins)`` signal and per-frame
        null variance, both in units of the particle's null sd under `ref_weights`,
        and that ``(n,)`` sd.
    """
    signals, variances, sds = [], [], []
    for c0 in range(0, c.shape[0], chunk):
        sig = best_position_signal(c[c0 : c0 + chunk], ref_weights)
        cov = noise_cov[c0 : c0 + chunk].to(sig)
        w = _as_tensor(ref_weights, sig)
        if null_mean is not None:
            sig = sig - null_mean[c0 : c0 + chunk].to(sig)
        sd = torch.sqrt(torch.einsum("fb,pbfg,gb->p", w, cov, w).clamp(min=1e-300))
        frame_var = torch.diagonal(cov, dim1=-2, dim2=-1).transpose(1, 2)
        signals.append(sig / sd[:, None, None])
        variances.append(frame_var / sd[:, None, None] ** 2)
        sds.append(sd)
    return torch.cat(signals), torch.cat(variances), torch.cat(sds)


def frame_signal_sampled_null(
    c: torch.Tensor,
    null_samples: torch.Tensor,
    ref_weights: np.ndarray | torch.Tensor,
    chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`frame_signal` with a sampled null (random orientations or decoys).

    As the study's ``dw_bfactor.frame_signal``: the null mean is subtracted, the sd is
    that of the reference-weighted correlation over the samples (so it includes any
    covariance between bins), and the per-frame variance is over the samples.

    Parameters
    ----------
    c : torch.Tensor
        ``(n, n_frames, n_positions, n_bins)`` correlation pieces.
    null_samples : torch.Tensor
        ``(n, n_samples, n_frames, n_bins)`` null correlations.
    ref_weights : array-like
        ``(n_frames, n_bins)`` reference weights.
    chunk : int
        Particles at a time.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(signal, variance, sd)`` as :func:`frame_signal`.
    """
    sig_all, var_all, sd_all = [], [], []
    for c0 in range(0, c.shape[0], chunk):
        x = null_samples[c0 : c0 + chunk].double()
        w = _as_tensor(ref_weights, x)
        sd = torch.einsum("prfb,fb->pr", x, w).std(1)
        sig = best_position_signal(c[c0 : c0 + chunk], w) - x.mean(1)
        sig_all.append(sig / sd[:, None, None])
        var_all.append(x.var(1) / sd[:, None, None] ** 2)
        sd_all.append(sd)
    return torch.cat(sig_all), torch.cat(var_all), torch.cat(sd_all)


def random_null_statistics(
    null_samples: torch.Tensor, chunk: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and between-frame covariance per bin of sampled null correlations.

    Parameters
    ----------
    null_samples : torch.Tensor
        ``(n, n_samples, n_frames, n_bins)`` null correlations per particle.
    chunk : int
        Particles at a time.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(mean, cov)``: ``(n, n_frames, n_bins)`` and ``(n, n_bins, n_frames,
        n_frames)``, for comparison with the analytic null.
    """
    means, covs = [], []
    for c0 in range(0, null_samples.shape[0], chunk):
        x = null_samples[c0 : c0 + chunk].double()
        mu = x.mean(1)
        d = x - mu[:, None]
        covs.append(torch.einsum("prfb,prgb->pbfg", d, d) / (x.shape[1] - 1))
        means.append(mu)
    return torch.cat(means), torch.cat(covs)


@dataclass
class FrameSignalSummary:
    """The particles' mean signal per frame and bin, and the bins fitted.

    Attributes
    ----------
    weighted_signal : np.ndarray
        ``(n, n_frames, n_bins)`` per-particle signal divided by each frame's relative
        noise variance (``s / sigma^2``, the best per-frame weight for frames with
        independent noise).
    mean : np.ndarray
        ``(n_frames, n_bins)`` mean over particles.
    standard_error : np.ndarray
        ``(n_frames, n_bins)`` standard error of the mean.
    noise_relative : np.ndarray
        ``(n_frames, n_bins)`` each frame's null variance relative to the mean frame.
    bins_used : np.ndarray
        Indices of the bins fitted: a contiguous run from the lowest bin at or above
        `min_frequency` whose summed signal is at least `min_snr` standard errors.
    snr : np.ndarray
        ``(n_bins,)`` the summed signal in standard errors.
    """

    weighted_signal: np.ndarray
    mean: np.ndarray
    standard_error: np.ndarray
    noise_relative: np.ndarray
    bins_used: np.ndarray
    snr: np.ndarray


def summarise_signal(
    weighted_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over particles (axis 0)."""
    n = weighted_signal.shape[0]
    mean = weighted_signal.mean(0)
    se = weighted_signal.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, se


def select_bins(
    mean: np.ndarray,
    se: np.ndarray,
    frequency: np.ndarray,
    min_snr: float = 5.0,
    min_frequency: float = 1 / 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The bins to fit: a contiguous run of clear signal from the lowest frequency.

    Bins without data (no Fourier samples in a small box, so a zero standard error)
    are skipped and do not end the run.

    Parameters
    ----------
    mean, se : np.ndarray
        ``(n_frames, n_bins)`` mean signal and standard error.
    frequency : np.ndarray
        ``(n_bins,)`` bin centres in 1/A.
    min_snr : float
        Minimum summed signal over frames, in standard errors.
    min_frequency : float
        Lowest frequency fitted, 1/A.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bins_used, snr)``.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = mean.sum(0) / np.sqrt((se**2).sum(0))
    snr = np.where(np.isfinite(snr), snr, 0.0)
    has_data = (se > 0).any(0)
    candidates = np.where(has_data & (frequency >= min_frequency))[0]
    run: list[int] = []
    for b in candidates:
        if snr[b] >= min_snr:
            run.append(int(b))
        elif run:
            break
    return np.asarray(run, dtype=int), snr


def frame_signal_summary(
    signal: np.ndarray,
    variance: np.ndarray,
    frequency: np.ndarray,
    min_snr: float = 5.0,
    min_frequency: float = 1 / 40.0,
) -> FrameSignalSummary:
    """Weight each frame's signal by its relative noise and summarise over particles.

    Parameters
    ----------
    signal, variance : np.ndarray
        ``(n, n_frames, n_bins)`` from :func:`frame_signal`.
    frequency : np.ndarray
        ``(n_bins,)`` bin centres in 1/A.
    min_snr, min_frequency : float
        Bin selection (see :func:`select_bins`).

    Returns
    -------
    FrameSignalSummary
    """
    noise = variance.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = noise / noise.mean(0, keepdims=True)
    rel = np.where(np.isfinite(rel) & (rel > 0), rel, 1.0)
    weighted = signal / rel[None]
    mean, se = summarise_signal(weighted)
    use, snr = select_bins(mean, se, frequency, min_snr, min_frequency)
    return FrameSignalSummary(weighted, mean, se, rel, use, snr)


@dataclass
class BFactorPrior:
    """A Gaussian prior on each frame's B and C, relative to a reference frame.

    The prior acts on differences from the fit's reference frame, so it does not
    fix the gauge: ``(B_f - B_ref) ~ N(B0_f - B0_ref, sd_bfactor^2)``, and the same
    for C.

    Attributes
    ----------
    bfactor, log_scale : np.ndarray
        ``(n_frames,)`` prior means (for example the default table on the movie's
        frames).
    sd_bfactor : float | None
        Prior sd of B in A^2, or None for no prior on B.
    sd_log_scale : float | None
        Prior sd of C, or None for no prior on C.
    """

    bfactor: np.ndarray
    log_scale: np.ndarray
    sd_bfactor: float | None = None
    sd_log_scale: float | None = None


@dataclass
class BFactorFit:
    """RELION's per-frame model fitted to the signal.

    Attributes
    ----------
    amplitude : np.ndarray
        ``(n_bins_used,)`` per-bin amplitude A_b.
    bfactor : np.ndarray
        ``(n_frames,)`` B_f in A^2, maximum 0.
    log_scale : np.ndarray
        ``(n_frames,)`` C_f, maximum 0.
    chi2_dof : float
        Reduced chi-squared of the data residuals.
    reference_frame : int
        The frame whose B and C were held at 0 during the fit (the gauge).
    """

    amplitude: np.ndarray
    bfactor: np.ndarray
    log_scale: np.ndarray
    chi2_dof: float
    reference_frame: int
    extra: dict = field(default_factory=dict)

    def model(self, k2: np.ndarray) -> np.ndarray:
        """``A_b exp(B_f k_b^2 / 4 + C_f)`` at the fitted bins, (n_frames, n_bins)."""
        return np.asarray(
            self.amplitude[None, :]
            * np.exp(self.bfactor[:, None] * k2[None, :] / 4 + self.log_scale[:, None])
        )


def default_reference_frame(mean: np.ndarray) -> int:
    """The frame with the most summed signal over the fitted bins."""
    return int(np.argmax(mean.sum(1)))


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-statements
def fit_bfactors(
    mean: np.ndarray,
    se: np.ndarray,
    k2: np.ndarray,
    reference_frame: int | None = None,
    prior: BFactorPrior | None = None,
    max_nfev: int = 20000,
) -> BFactorFit:
    """Weighted least squares ``s[f, b] = A_b exp(B_f k_b^2 / 4 + C_f)``.

    Every frame's B and C are free; one frame's are held at 0 during the fit to fix
    the gauge (the per-bin A_b absorbs the rest), and the result is shifted so the
    maximum B and the maximum C are 0.

    Parameters
    ----------
    mean, se : np.ndarray
        ``(n_frames, n_bins)`` signal and standard error at the fitted bins.
    k2 : np.ndarray
        ``(n_bins,)`` squared bin frequencies, 1/A^2.
    reference_frame : int | None
        The gauge frame. Default: the frame with the most signal.
    prior : BFactorPrior | None
        Optional shrinkage towards a reference table. None (the default) fits the
        data alone.
    max_nfev : int
        Iteration limit for ``scipy.optimize.least_squares``.

    Returns
    -------
    BFactorFit
    """
    nf, nb = mean.shape
    ref = default_reference_frame(mean) if reference_frame is None else reference_frame
    free = np.array([f for f in range(nf) if f != ref])
    nfree = len(free)
    a0 = np.log(np.clip(mean.mean(0), 1e-12, None))
    se = np.where(se > 0, se, np.inf)

    def unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bf = np.zeros(nf)
        cf = np.zeros(nf)
        bf[free] = p[nb : nb + nfree]
        cf[free] = p[nb + nfree :]
        return p[:nb], bf, cf

    prior_terms = []
    if prior is not None:
        if prior.sd_bfactor is not None:
            b0 = np.asarray(prior.bfactor, float)
            prior_terms.append(("B", (b0 - b0[ref])[free], prior.sd_bfactor))
        if prior.sd_log_scale is not None:
            c0 = np.asarray(prior.log_scale, float)
            prior_terms.append(("C", (c0 - c0[ref])[free], prior.sd_log_scale))

    def resid(p: np.ndarray) -> np.ndarray:
        la, bf, cf = unpack(p)
        model = np.exp(la[None, :] + bf[:, None] * k2[None, :] / 4 + cf[:, None])
        r = [((mean - model) / se).ravel()]
        for name, mu, sd in prior_terms:
            r.append(((bf if name == "B" else cf)[free] - mu) / sd)
        return np.concatenate(r)

    n_data = mean.size
    n_params = nb + 2 * nfree
    rows_b = np.repeat(np.arange(nf), nb)  # frame of each data residual (row-major)
    cols_b = np.tile(np.arange(nb), nf)  # bin of each data residual
    col_of_frame = np.full(nf, -1)
    col_of_frame[free] = np.arange(nfree)

    def jacobian(p: np.ndarray) -> csr_matrix:
        """Analytic, sparse Jacobian of `resid`.

        Each data residual depends on one A_b and one frame's B and C (sparse
        solvers also avoid dense BLAS, which is slow with many threads here).
        """
        la, bf, cf = unpack(p)
        model = np.exp(la[None, :] + bf[:, None] * k2[None, :] / 4 + cf[:, None])
        dm = (-model / se).ravel()
        on_free = col_of_frame[rows_b] >= 0
        fc = col_of_frame[rows_b][on_free]
        rows = np.arange(n_data)
        r_idx = [rows, rows[on_free], rows[on_free]]
        c_idx = [cols_b, nb + fc, nb + nfree + fc]
        vals = [dm, dm[on_free] * k2[cols_b[on_free]] / 4, dm[on_free]]
        n_rows = n_data
        for name, _, sd in prior_terms:
            offset = nb if name == "B" else nb + nfree
            r_idx.append(n_rows + np.arange(nfree))
            c_idx.append(offset + np.arange(nfree))
            vals.append(np.full(nfree, 1.0 / sd))
            n_rows += nfree
        return csr_matrix(
            (np.concatenate(vals), (np.concatenate(r_idx), np.concatenate(c_idx))),
            shape=(n_rows, n_params),
        )

    p0 = np.concatenate([a0, np.zeros(2 * nfree)])
    result = least_squares(
        resid,
        p0,
        jac=jacobian,
        method="trf",
        x_scale="jac",
        tr_solver="lsmr",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    la, bf, cf = unpack(result.x)
    shift_b, shift_c = bf.max(), cf.max()
    bf, cf = bf - shift_b, cf - shift_c
    la = la + shift_b * k2 / 4 + shift_c
    data_resid = result.fun[:n_data]
    chi2 = float((data_resid**2).sum() / max(1, n_data - result.x.size))
    return BFactorFit(
        amplitude=np.exp(la),
        bfactor=bf,
        log_scale=cf,
        chi2_dof=chi2,
        reference_frame=ref,
        extra={"success": bool(result.success), "nfev": int(result.nfev)},
    )


def bootstrap_bfactors(
    weighted_signal: np.ndarray,
    bins_used: np.ndarray,
    k2: np.ndarray,
    groups: list[np.ndarray],
    n_bootstrap: int = 200,
    reference_frame: int | None = None,
    prior: BFactorPrior | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit B and C on resampled particles (or groups of particles).

    Parameters
    ----------
    weighted_signal : np.ndarray
        ``(n, n_frames, n_bins)`` (:attr:`FrameSignalSummary.weighted_signal`).
    bins_used : np.ndarray
        The fitted bins.
    k2 : np.ndarray
        Squared frequencies of the fitted bins.
    groups : list[np.ndarray]
        Particle indices of each resampling unit (one particle each, or the
        particles of a micrograph or filament).
    n_bootstrap : int
        Number of resamples.
    reference_frame : int | None
        Gauge frame (as in the full fit).
    prior : BFactorPrior | None
        The prior of the full fit, if any.
    seed : int
        Random seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(n_bootstrap, n_frames)`` B and C of each resample (maximum 0).
    """
    rng = np.random.default_rng(seed)
    boot_b, boot_c = [], []
    for _ in range(n_bootstrap):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        mean, se = summarise_signal(weighted_signal[idx][:, :, bins_used])
        fit = fit_bfactors(mean, se, k2, reference_frame, prior)
        boot_b.append(fit.bfactor)
        boot_c.append(fit.log_scale)
    return np.asarray(boot_b), np.asarray(boot_c)


def groups_from_labels(labels: Any) -> list[np.ndarray]:
    """Particle indices for each distinct label, in order of first appearance."""
    labels = np.asarray(labels)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first)
    return [np.where(inverse == u)[0] for u in order]


# --- the template envelope correction ------------------------------------------------


def knot_frequencies(
    n_knots: int = 5, low_resolution: float = 50.0, high_resolution: float = 2.9
) -> np.ndarray:
    """Knots uniform in sqrt(k) between two resolutions (A), in 1/A."""
    s = np.linspace(
        math.sqrt(1 / low_resolution), math.sqrt(1 / high_resolution), n_knots
    )
    return np.asarray(s**2)


def spline_correction(
    frequency: np.ndarray, knots: np.ndarray, log_values: np.ndarray
) -> np.ndarray:
    """h(k): natural cubic spline of log h in sqrt(k) through the knots, flat outside.

    The same curve as ``torch_fourier_filter.frame_weights.EnvelopeCorrection``.
    """
    s_knots = np.sqrt(np.asarray(knots, float))
    lv = np.asarray(log_values, float)
    if np.allclose(lv, lv[0]):
        return np.full(np.shape(frequency), float(np.exp(lv[0])))
    spline = CubicSpline(s_knots, lv, bc_type="natural")
    s = np.sqrt(np.clip(np.asarray(frequency, float), 0, None))
    return np.asarray(np.exp(spline(np.clip(s, s_knots[0], s_knots[-1]))))


@dataclass
class SumCoefficients:
    """A weighted sum's correlation per bin, and its null variance, per particle.

    For frame weights ``W[f, b]`` and the sum's whitening relative to the frames'
    (``u2[b]``), the sum's correlation at position q is ``sum_b v[i, q, b]`` and its
    null variance ``sum_b var[i, b]``. A template envelope h multiplies both:
    ``z_i(h) = max_q (sum_b h_b v[i, q, b]) / sqrt(sum_b h_b^2 var[i, b])``.
    """

    v: torch.Tensor  # (n, n_positions, n_bins)
    var: torch.Tensor  # (n, n_bins)

    def z(self, h: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Per-particle z for envelope h (best position)."""
        hh = _as_tensor(h, self.v)
        num = torch.einsum("pqb,b->pq", self.v, hh)
        den = torch.sqrt((self.var * hh**2).sum(1)).clamp(min=1e-300)
        return (num / den[:, None]).max(1).values

    def subset(self, idx: np.ndarray) -> SumCoefficients:
        """The coefficients of particles `idx`."""
        i = torch.as_tensor(idx, device=self.v.device)
        return SumCoefficients(self.v[i], self.var[i])


def sum_coefficients(
    c: torch.Tensor,
    noise_cov: torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    u2: np.ndarray | torch.Tensor,
    null_mean: torch.Tensor | None = None,
    chunk: int = 256,
) -> SumCoefficients:
    """The coefficients of a weighted sum (see :class:`SumCoefficients`).

    Parameters
    ----------
    c : torch.Tensor
        ``(n, n_frames, n_positions, n_bins)`` correlation pieces.
    noise_cov : torch.Tensor
        ``(n, n_bins, n_frames, n_frames)`` null covariance.
    weights : array-like
        ``(n_frames, n_bins)`` frame weights of the sum.
    u2 : array-like
        ``(n, n_bins)`` or ``(n_bins,)``: the sum's whitening squared, relative to the
        whitening of the frames the pieces were computed on.
    null_mean : torch.Tensor | None
        ``(n, n_frames, n_bins)`` null mean, or None.
    chunk : int
        Particles at a time.

    Returns
    -------
    SumCoefficients
    """
    vs, vars_ = [], []
    for c0 in range(0, c.shape[0], chunk):
        cc = c[c0 : c0 + chunk].double()
        w = _as_tensor(weights, cc)
        uu = _as_tensor(u2, cc)
        if uu.ndim == 1:
            uu = uu[None].expand(cc.shape[0], -1)
        else:
            uu = uu[c0 : c0 + chunk]
        x = cc
        if null_mean is not None:
            x = cc - null_mean[c0 : c0 + chunk].to(cc)[:, :, None, :]
        v = torch.einsum("pfqb,fb->pqb", x, w) * uu[:, None, :]
        var = torch.einsum("fb,pbfg,gb->pb", w, noise_cov[c0 : c0 + chunk].to(cc), w)
        vs.append(v)
        vars_.append(var * uu**2)
    return SumCoefficients(torch.cat(vs), torch.cat(vars_))


def fit_envelope_correction(
    coefficients: SumCoefficients,
    base_envelope: np.ndarray,
    frequency: np.ndarray,
    knots: np.ndarray,
    max_iterations: int = 800,
) -> tuple[np.ndarray, dict]:
    """Log values of the smooth correction that maximise the particles' mean z.

    The template's envelope relative to the template the pieces were computed with
    is ``h = base_envelope * spline(log_values)``, the spline through `knots`
    (:func:`spline_correction`) with the first knot's value held at 0 (only the
    shape matters). Nelder-Mead from no correction, steps of 0.3 in log h.

    Parameters
    ----------
    coefficients : SumCoefficients
        The particles on the weighted sum.
    base_envelope : np.ndarray
        ``(n_bins,)`` the envelope before correction, relative to the template the
        pieces were computed with (the matched envelope divided by that template's
        own envelope).
    frequency : np.ndarray
        ``(n_bins,)`` bin centres, 1/A.
    knots : np.ndarray
        ``(n_knots,)`` knot frequencies, 1/A.
    max_iterations : int
        Nelder-Mead iteration limit.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(log_values, info)``: ``(n_knots,)`` log h at the knots (first 0), and the
        mean z before and after.
    """
    n_knots = len(knots)

    def h_of(p: np.ndarray) -> np.ndarray:
        lv = np.concatenate([[0.0], p])
        return np.asarray(base_envelope * spline_correction(frequency, knots, lv))

    def objective(p: np.ndarray) -> float:
        return -float(coefficients.z(h_of(p)).mean())

    x0 = np.zeros(n_knots - 1)
    simplex = np.vstack([x0, 0.3 * np.eye(n_knots - 1)])
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": max_iterations,
            "xatol": 1e-3,
            "fatol": 1e-5,
            "initial_simplex": simplex,
        },
    )
    log_values = np.concatenate([[0.0], result.x])
    info = {
        "mean_z_base": -objective(x0),
        "mean_z_corrected": -float(result.fun),
        "iterations": int(result.nit),
        "converged": bool(result.success),
    }
    return log_values, info
