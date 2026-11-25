"""Per-frame, per-bin template correlations at particles, through Leopard-EM.

For each particle of a Leopard-EM particle stack, at its pose and defocus, the
cross-correlation of the template projection with each aligned, whitened movie frame
is split into radial frequency bins, at the (2 w + 1)^2 positions within w pixels of
the particle's position. The template side is Leopard-EM's: the projection of the
template's Fourier slice, the CTF, the whitening filter Leopard-EM builds for the
frames' plain sum, the magnification matrix, and the projection's edge-mean removal.
With all frames and bins summed, the pieces reproduce Leopard-EM's own
cross-correlation of the summed box (up to its normalisation); the first few
particles are checked.

The null (the correlation of the template where it does not match) is analytic: for a
template whose phases are random relative to the image, the covariance between frames
f and g of the bin-b correlation is

    Cov_b(f, g) = S_b(f, g) Q_b / (n_b N^4)

with ``S_b(f, g) = sum_{k in b} Re F_f(k) conj F_g(k)`` the particle box's frame
cross-spectrum, ``Q_b`` the template projection's power in the bin, ``n_b`` the
number of Fourier samples in it and N the box size (rfft grid, Hermitian weights).
This replaces the 1,024 random orientations per particle of the study the estimator
comes from; ``n_random`` computes those as well, to check the analytic null.
"""

from __future__ import annotations

# torch.fft functions are callable; pylint cannot tell
# pylint: disable=not-callable
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd


@contextmanager
def keep_grad_mode() -> Iterator[None]:
    """Restore autograd's global mode after importing Leopard-EM.

    Leopard-EM's ``backend.core_match_template`` calls
    ``torch.set_grad_enabled(False)`` when it is imported, which would switch off
    autograd for the rest of the process (motion estimation and polishing need it).
    """
    enabled = torch.is_grad_enabled()
    try:
        yield
    finally:
        torch.set_grad_enabled(enabled)


def rfft_grid_bins(
    n: int, edges: np.ndarray, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bin index, Hermitian weight and integer frequencies of an n x n rfft grid.

    Parameters
    ----------
    n : int
        Real-space size (square).
    edges : np.ndarray
        Radial bin edges, cycles / px.
    device : torch.device | str | None
        Device of the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(bins, weight, ky, kx)``: ``(n, n // 2 + 1)`` bin index and weight (2 for
        columns standing for a +-k pair, 1 for the first column and, for even n, the
        last; 0 at the origin), and the integer frequencies ``ky (n,)`` and
        ``kx (n // 2 + 1,)``.
    """
    ky = torch.fft.fftfreq(n, device=device) * n
    kx = torch.fft.rfftfreq(n, device=device) * n
    kk = torch.sqrt((ky[:, None] / n) ** 2 + (kx[None, :] / n) ** 2)
    inner = torch.as_tensor(np.asarray(edges)[1:-1], device=device, dtype=kk.dtype)
    bins = torch.bucketize(kk.contiguous(), inner, right=True)
    weight = torch.full_like(kk, 2.0)
    weight[:, 0] = 1.0
    if n % 2 == 0:
        weight[:, -1] = 1.0
    weight[0, 0] = 0.0
    return bins, weight, ky, kx


def radial_profile(filter_rfft: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Radial mean of an (n, n // 2 + 1) rfft filter, against |k| in cycles / px."""
    n = filter_rfft.shape[-2]
    kk = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.rfftfreq(n)[None, :] ** 2)
    kk = kk.ravel()
    v = filter_rfft.detach().cpu().double().numpy().ravel()
    edges = np.arange(0, kk.max() + 2.0 / n, 1.0 / n)
    idx = np.digitize(kk, edges) - 1
    num = np.bincount(idx, v, minlength=len(edges))
    den = np.bincount(idx, minlength=len(edges))
    ok = den > 0
    centres = np.bincount(idx, kk, minlength=len(edges))[ok] / den[ok]
    return centres, num[ok] / den[ok]


class _SortedBins:  # pylint: disable=too-few-public-methods
    """Fourier samples of a grid sorted by radial bin, for per-bin contractions."""

    def __init__(self, bins: torch.Tensor, n_bins: int) -> None:
        flat = bins.reshape(-1)
        self.order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=n_bins)[:n_bins]
        bounds = torch.zeros(n_bins + 1, dtype=torch.long)
        bounds[1:] = torch.cumsum(counts.cpu(), 0)
        self.bounds = bounds.tolist()
        self.n_bins = n_bins

    def segments(self) -> list[tuple[int, int, int]]:
        """``(bin, start, stop)`` of every non-empty bin in the sorted order."""
        return [
            (b, self.bounds[b], self.bounds[b + 1])
            for b in range(self.n_bins)
            if self.bounds[b + 1] > self.bounds[b]
        ]


def frame_cross_spectra(
    frames_dft_sorted: torch.Tensor,
    sorted_bins: _SortedBins,
    weight_sorted: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """``S[b, f, g] = sum_{k in b} w(k) Re F_f(k) conj F_g(k)`` for sorted DFTs.

    Parameters
    ----------
    frames_dft_sorted : torch.Tensor
        ``(n_frames, n_samples)`` DFT samples ordered by bin (any device).
    sorted_bins : _SortedBins
        The bin segments.
    weight_sorted : torch.Tensor
        ``(n_samples,)`` Hermitian weights in the same order.
    device : torch.device | str
        Compute device.

    Returns
    -------
    torch.Tensor
        ``(n_bins, n_frames, n_frames)`` float64 on the CPU.
    """
    nf = frames_dft_sorted.shape[0]
    out = torch.zeros(sorted_bins.n_bins, nf, nf, dtype=torch.float64)
    for b, lo, hi in sorted_bins.segments():
        fb = frames_dft_sorted[:, lo:hi].to(device)
        fb = fb * torch.sqrt(weight_sorted[lo:hi].to(device))[None]
        out[b] = (fb @ fb.conj().T).real.double().cpu()
    return out


def template_projections(  # pylint: disable=too-many-locals
    kwargs: dict[str, Any],
    rotations: torch.Tensor,
    template_filter: torch.Tensor,
    box: int,
) -> torch.Tensor:
    """Leopard-EM's template projections, edge mean removed, as DFTs on a larger box.

    As Leopard-EM's correlation: the template's central slice (with the
    magnification matrix if any), DC removed, contrast inverted, times the
    template-side filter; in real space, centred, with the mean of its edge pixels
    subtracted. The projection fills the top-left of the ``box x box`` grid.

    Parameters
    ----------
    kwargs : dict
        Leopard-EM's backend kwargs (``template_dft``, ``mag_matrix``).
    rotations : torch.Tensor
        ``(r, 3, 3)`` rotation matrices.
    template_filter : torch.Tensor
        ``(t, t // 2 + 1)`` template-side filter (whitening times CTF).
    box : int
        Size of the grid (at least the template's).

    Returns
    -------
    torch.Tensor
        ``(r, box, box // 2 + 1)`` complex DFTs.
    """
    # pylint: disable=import-outside-toplevel
    from torch_fourier_slice import extract_central_slices_rfft_3d, transform_slice_2d

    dev = template_filter.device
    template_dft = kwargs["template_dft"].to(dev)
    mag = kwargs.get("mag_matrix")
    fs = extract_central_slices_rfft_3d(
        volume_rfft=template_dft, rotation_matrices=rotations
    )
    if isinstance(mag, torch.Tensor):
        fs = transform_slice_2d(
            projection_image_dfts=fs,
            rfft_shape=(template_dft.shape[1], template_dft.shape[2]),
            stack_shape=(rotations.shape[0],),
            transform_matrix=mag.to(dev),
        )
    fs = torch.fft.ifftshift(fs, dim=(-2,))
    fs[..., 0, 0] = 0
    proj = torch.fft.ifftshift(
        torch.fft.irfftn(-fs * template_filter, dim=(-2, -1)), dim=(-2, -1)
    )
    top, bottom = proj[..., 0, :], proj[..., -1, :]
    left, right = proj[..., 1:-1, 0], proj[..., 1:-1, -1]
    edge_mean = torch.cat([top, bottom, left, right], -1).mean(-1)
    pp = torch.fft.rfftn(proj - edge_mean[:, None, None], dim=(-2, -1), s=(box, box))
    pp[..., 0, 0] = 0
    return pp  # type: ignore[no-any-return, unused-ignore]


def particle_template_filter(
    kwargs: dict[str, Any], index: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """A particle's template-side filter (whitening times CTF) and rotation matrix.

    Parameters
    ----------
    kwargs : dict
        Leopard-EM's backend kwargs.
    index : int
        The particle's row.
    device : torch.device | str
        Device of the outputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(filter (t, t // 2 + 1), rotation (1, 3, 3))``.
    """
    # pylint: disable=import-outside-toplevel
    import roma

    with keep_grad_mode():
        from leopard_em.backend.utils import EULER_ANGLE_FMT
        from leopard_em.utils.ctf_utils import (
            calculate_ctf_filter_stack_full_args,
            move_ctf_kwargs_tensors_to_device,
        )

    dev = torch.device(device)
    ctf = calculate_ctf_filter_stack_full_args(
        defocus_u=kwargs["defocus_u"][index],
        defocus_v=kwargs["defocus_v"][index],
        astigmatism_angle=kwargs["defocus_angle"][index],
        defocus_offsets=torch.zeros(1, device=dev),
        pixel_size_offsets=torch.zeros(1, device=dev),
        **move_ctf_kwargs_tensors_to_device(kwargs["ctf_kwargs"], dev),
    )
    filt = (
        kwargs["projective_filters"][index].to(dev)
        * ctf.reshape(-1, *ctf.shape[-2:])[0]
    )
    rot = roma.euler_to_rotmat(
        EULER_ANGLE_FMT, kwargs["euler_angles"][index].to(dev)[None], degrees=True
    ).float()
    return filt, rot


@dataclass
class MoviePieces:  # pylint: disable=too-many-instance-attributes
    """Correlation pieces of one movie's particles.

    Attributes
    ----------
    c : torch.Tensor
        ``(n, n_frames, n_positions, n_bins)`` float32: the correlation per frame,
        position and bin. Position ``j`` is ``(j // (2w+1), j % (2w+1))`` in pixels
        from the window's corner; the particle's own position is the centre.
    noise_cov : torch.Tensor
        ``(n, n_bins, n_frames, n_frames)`` float32 analytic null covariance.
    template_power : torch.Tensor
        ``(n, n_bins)`` the projection's power per bin (box grid).
    frame_cross_spectra : torch.Tensor
        ``(n_bins, n_frames, n_frames)`` the whole micrograph's whitened frame
        cross-spectra (for the whitening of any weighted sum).
    particle_rows : np.ndarray
        Row positions (in the movie's particle table) of the particles kept.
    dropped_rows : np.ndarray
        Rows dropped because the box left the frame.
    null_random : torch.Tensor | None
        ``(n, n_random, n_frames, n_bins)`` float16 correlations of random
        orientations at the particle's position, if requested.
    checks : dict
        Agreement with Leopard-EM for the first particles.
    """

    c: torch.Tensor
    noise_cov: torch.Tensor
    template_power: torch.Tensor
    frame_cross_spectra: torch.Tensor
    particle_rows: np.ndarray
    dropped_rows: np.ndarray
    null_random: torch.Tensor | None = None
    checks: dict = field(default_factory=dict)


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-statements
def compute_movie_pieces(
    frames: torch.Tensor,
    kwargs: dict[str, Any],
    positions: np.ndarray,
    template_size: int,
    edges: np.ndarray,
    window: int = 3,
    device: torch.device | str = "cuda",
    n_random: int = 0,
    random_batch: int = 8,
    n_verify: int = 3,
    seed: int = 0,
) -> MoviePieces:
    """Correlation pieces and analytic null for one movie's particles.

    Parameters
    ----------
    frames : torch.Tensor
        ``(n_frames, h, w)`` aligned frames, already whitened with the filter in
        ``kwargs["projective_filters"]`` (see :func:`whiten_frames`), any device.
    kwargs : dict
        Leopard-EM's backend kwargs for the movie's particles
        (``template_dft``, ``projective_filters``, ``euler_angles``, ``defocus_u``,
        ``defocus_v``, ``defocus_angle``, ``ctf_kwargs``, ``mag_matrix`` and, for
        the check, ``particle_stack_dft`` of the first particles).
    positions : np.ndarray
        ``(n, 2)`` integer (y, x) template top-left positions (Leopard-EM's
        ``pos_y``, ``pos_x``) on the frames.
    template_size : int
        Template box size (square).
    edges : np.ndarray
        Radial bin edges, cycles / px.
    window : int
        Positions within this many pixels of the particle's are kept.
    device : torch.device | str
        Compute device.
    n_random : int
        Random orientations per particle for the empirical null (0: none).
    random_batch : int
        Orientations per batch for the empirical null.
    n_verify : int
        Particles checked against Leopard-EM's own correlation.
    seed : int
        Seed of the random orientations.

    Returns
    -------
    MoviePieces
    """
    # pylint: disable=import-outside-toplevel
    import roma

    with keep_grad_mode():
        from leopard_em.backend.cross_correlation import (
            do_batched_orientation_cross_correlate,
        )

    dev = torch.device(device)
    nf, height, width = frames.shape
    n_bins = len(edges) - 1
    box = template_size + 2 * window
    n_pos = (2 * window + 1) ** 2

    bins_box, wt_box, ky, kx = rfft_grid_bins(box, edges, dev)
    sorted_box = _SortedBins(bins_box.cpu(), n_bins)
    order = sorted_box.order.to(dev)
    wt_sorted = wt_box.reshape(-1)[order]
    n_per_bin = torch.zeros(n_bins, dtype=torch.float64)
    for b, lo, hi in sorted_box.segments():
        n_per_bin[b] = float(wt_sorted[lo:hi].sum())
    n_per_bin_dev = n_per_bin.clamp(min=1.0).to(dev)

    # phases of the window positions:
    # c(y, x) = Re sum_k w(k) X(k) e^{2 pi i k.(y, x) / N} / N^2
    py = torch.arange(2 * window + 1, device=dev, dtype=torch.float32)
    pyy, pxx = torch.meshgrid(py, py, indexing="ij")
    phase = (
        2
        * math.pi
        * (
            ky.reshape(-1, 1, 1)[:, None] * pyy[None, None]
            + kx.reshape(1, -1, 1, 1) * pxx[None, None]
        )
        / box
    )  # (box, box//2+1, 2w+1, 2w+1)
    e_pos = (wt_box[..., None, None] / box**2) * torch.exp(1j * phase)
    e_pos = e_pos.reshape(-1, n_pos)[order].to(torch.complex64)  # (F, P)
    del phase
    centre = n_pos // 2
    e_centre = e_pos[:, centre]

    template_dft = kwargs["template_dft"].to(dev)
    mag = kwargs.get("mag_matrix")
    mag = mag.to(dev) if isinstance(mag, torch.Tensor) else None
    gen = torch.Generator(device="cpu").manual_seed(seed)

    inside = (
        (positions[:, 0] - window >= 0)
        & (positions[:, 1] - window >= 0)
        & (positions[:, 0] - window + box <= height)
        & (positions[:, 1] - window + box <= width)
    )
    rows = np.where(inside)[0]
    dropped = np.where(~inside)[0]
    m = len(rows)
    c_out = torch.zeros(m, nf, n_pos, n_bins)
    cov_out = torch.zeros(m, n_bins, nf, nf)
    q_out = torch.zeros(m, n_bins)
    null = (
        torch.zeros(m, n_random, nf, n_bins, dtype=torch.float16) if n_random else None
    )
    checks: dict[str, list[float]] = {"sum_box_vs_leopard_corr": [], "cc_rel_err": []}
    segments = sorted_box.segments()

    for j, i in enumerate(rows):
        y0, x0 = int(positions[i, 0]) - window, int(positions[i, 1]) - window
        boxes = frames[:, y0 : y0 + box, x0 : x0 + box].to(dev, torch.float32)
        imgs = torch.fft.rfftn(boxes, dim=(-2, -1))  # (nf, box, box//2+1)
        imgs_sorted = imgs.reshape(nf, -1)[:, order]
        del boxes
        filt, rot = particle_template_filter(kwargs, int(i), dev)
        pp = template_projections(kwargs, rot, filt, box)[0]
        pp_sorted = pp.reshape(-1)[order]
        power = (pp_sorted.abs() ** 2) * wt_sorted
        x_sorted = imgs_sorted * pp_sorted.conj()[None]  # (nf, F)
        c_dev = torch.zeros(nf, n_pos, n_bins, device=dev)
        cov_dev = torch.zeros(n_bins, nf, nf, device=dev, dtype=torch.float64)
        q_dev = torch.zeros(n_bins, device=dev, dtype=torch.float64)
        for b, lo, hi in segments:
            q_dev[b] = power[lo:hi].double().sum()
            c_dev[:, :, b] = (x_sorted[:, lo:hi] @ e_pos[lo:hi]).real
            fb = imgs_sorted[:, lo:hi] * torch.sqrt(wt_sorted[lo:hi])[None]
            cov_dev[b] = (fb @ fb.conj().T).real.double()
        cov_dev *= (q_dev / (n_per_bin_dev * box**4))[:, None, None]
        c_out[j] = c_dev.cpu()
        cov_out[j] = cov_dev.float().cpu()
        q_out[j] = q_dev.float().cpu()
        verify = j < n_verify and i < kwargs["particle_stack_dft"].shape[0]
        if verify:
            ref_box = kwargs["particle_stack_dft"][i].to(dev)
            sel = bins_box > 0
            both = torch.stack(
                [
                    torch.view_as_real(imgs.sum(0)[sel]).reshape(-1),
                    torch.view_as_real(ref_box[sel]).reshape(-1),
                ]
            )
            checks["sum_box_vs_leopard_corr"].append(float(torch.corrcoef(both)[0, 1]))
            ref = do_batched_orientation_cross_correlate(
                image_dft=imgs.sum(0),
                template_dft=template_dft,
                rotation_matrices=rot,
                projective_filters=filt[None, None],
                apply_normalization=True,
                mag_matrix=mag,
            )[0, 0, 0, : 2 * window + 1, : 2 * window + 1].double()
            mine = c_out[j].sum((0, 2)).reshape(2 * window + 1, -1).double().to(dev)
            scale = float((mine * ref).sum() / (mine * mine).sum())
            err = float((scale * mine - ref).abs().max() / ref.abs().max())
            checks["cc_rel_err"].append(err)
        if null is not None:
            rand = roma.random_rotmat(n_random, generator=gen)
            for r0 in range(0, n_random, random_batch):
                rot_r = rand[r0 : r0 + random_batch].to(dev).float()
                pr = template_projections(kwargs, rot_r, filt, box).reshape(
                    rot_r.shape[0], -1
                )[:, order]
                xr = imgs_sorted[None] * pr.conj()[:, None]  # (r, nf, F)
                vals = torch.zeros(rot_r.shape[0], nf, n_bins, device=dev)
                for b, lo, hi in segments:
                    vals[..., b] = (xr[..., lo:hi] @ e_centre[lo:hi]).real
                null[j, r0 : r0 + rot_r.shape[0]] = vals.half().cpu()
        del imgs, imgs_sorted, x_sorted
    return MoviePieces(
        c=c_out,
        noise_cov=cov_out,
        template_power=q_out,
        frame_cross_spectra=torch.zeros(n_bins, nf, nf, dtype=torch.float64),
        particle_rows=rows,
        dropped_rows=dropped,
        null_random=null,
        checks=checks,
    )


def whiten_frames(
    frames: torch.Tensor,
    whitening: tuple[np.ndarray, np.ndarray],
    edges: np.ndarray,
    device: torch.device | str,
    chunk: int = 4,
) -> torch.Tensor:
    """Whiten frames in place with a radial filter; return their cross-spectra.

    Parameters
    ----------
    frames : torch.Tensor
        ``(n_frames, h, w)`` float32 frames (modified in place).
    whitening : tuple[np.ndarray, np.ndarray]
        ``(|k| in cycles / px, filter value)``, e.g. :func:`radial_profile` of
        Leopard-EM's template-side filter for the frames' plain sum.
    edges : np.ndarray
        Radial bin edges for the cross-spectra.
    device : torch.device | str
        Compute device.
    chunk : int
        Frames at a time.

    Returns
    -------
    torch.Tensor
        ``(n_bins, n_frames, n_frames)`` cross-spectra of the whitened frames.
    """
    dev = torch.device(device)
    nf, height, width = frames.shape
    kk = torch.sqrt(
        torch.fft.fftfreq(height, device=dev)[:, None] ** 2
        + torch.fft.rfftfreq(width, device=dev)[None, :] ** 2
    )
    kc, wc = (np.asarray(a, float) for a in whitening)
    wgrid = torch.as_tensor(
        np.interp(kk.double().cpu().numpy(), kc, wc), dtype=torch.float32, device=dev
    )
    wgrid[0, 0] = 0.0
    n_bins = len(edges) - 1
    inner = torch.as_tensor(np.asarray(edges)[1:-1], device=dev, dtype=kk.dtype)
    bins = torch.bucketize(kk.contiguous(), inner, right=True)
    weight = torch.full_like(kk, 2.0)
    weight[:, 0] = 1.0
    if width % 2 == 0:
        weight[:, -1] = 1.0
    weight[0, 0] = 0.0
    sorted_bins = _SortedBins(bins.cpu(), n_bins)
    order = sorted_bins.order.to(dev)
    dft_sorted = torch.empty(nf, kk.numel(), dtype=torch.complex64)
    for f0 in range(0, nf, chunk):
        dft = torch.fft.rfft2(frames[f0 : f0 + chunk].to(dev, torch.float32)) * wgrid
        frames[f0 : f0 + chunk] = torch.fft.irfft2(dft, s=(height, width)).to(
            frames.device
        )
        dft_sorted[f0 : f0 + chunk] = dft.reshape(dft.shape[0], -1)[:, order].cpu()
        del dft
    spectra = frame_cross_spectra(
        dft_sorted, sorted_bins, weight.reshape(-1)[order].cpu(), dev
    )
    del dft_sorted
    return spectra


def particle_positions(df: pd.DataFrame) -> np.ndarray:
    """(n, 2) integer (y, x) positions, refined if present (as Leopard-EM)."""
    y_col = "refined_pos_y" if "refined_pos_y" in df.columns else "pos_y"
    x_col = "refined_pos_x" if "refined_pos_x" in df.columns else "pos_x"
    return np.stack(
        [np.round(df[y_col].to_numpy()), np.round(df[x_col].to_numpy())], 1
    ).astype(np.int64)
