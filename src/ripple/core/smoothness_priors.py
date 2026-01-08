# smoothness.py
# ----------
# Provides temporal and spatial smoothness energies for control-point deformation fields.
# Expected deformation_field tensor shape: (2, nt, nh, nw)  (order: (y/x, time, h, w))
#
# Functions:
#   compute_temporal_energy(field, sigma_A, mode='velocity')
#   compute_spatial_energy(field, method='laplacian'|'spectral', **kwargs)
#   compute_energies(field, sigma_A, spatial_method='laplacian', spatial_kwargs={})
#
# Notes:
# - field is shifts per control-point per frame (absolute positions/displacements).
# - For temporal energy we define velocities v_f = s_{f+1} - s_f (intervals).
# - E_time follows eq. (10): sum_{p} sum_f ||v_f - v_{f-1}||^2 scaled by 1/sigma_A^2.
# - spectral spatial prior implements 0.5 * v^T Sigma^{-1} v per frame (separable Sigma).
# ----------

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F


# -------------------------
# Temporal smoothness (eq.10)
# -------------------------
def compute_temporal_energy(
    field: torch.Tensor,
    sigma_A: float,
    *,
    use_velocities: bool = True,
) -> torch.Tensor:
    """
    Compute temporal smoothness energy per eq. (10) using velocities.

    Parameters
    ----------
    field : torch.Tensor
        deformation field, shape (2, nt, nh, nw). This is the absolute shift s[f] per frame.
    sigma_A : float
        temporal smoothness scale (positive).
    use_velocities : bool
        If True (default), interpret temporal prior as penalizing differences of velocities:
           v_f = s[f+1] - s[f], and penalize (v_f - v_{f-1}).
        If False, penalize first differences of s (i.e., velocities directly):
           sum_f ||s[f+1] - s[f]||^2  (less common for eq.10).

    Returns
    -------
    E_time : torch.Tensor (scalar)
        Temporal energy (scalar tensor).
    """
    if sigma_A <= 0:
        raise ValueError("sigma_A must be positive.")

    # field: (2, nt, nh, nw)
    assert field.dim() == 4 and field.shape[0] == 2, "field must be (2, nt, nh, nw)"

    # compute velocities between frames: v_f = s[f+1] - s[f]
    # shape: (2, nt-1, nh, nw)
    v = field[:, 1:, :, :] - field[:, :-1, :, :]

    if use_velocities:
        # penalize differences of consecutive velocities: dv_f = v[f+1] - v[f]
        if v.shape[1] < 2:
            # not enough frames for second difference -> zero energy
            return torch.tensor(0.0, device=field.device, dtype=field.dtype)
        dv = v[:, 1:, :, :] - v[:, :-1, :, :]  # (2, nt-2, nh, nw)
        E_time = (1.0 / (sigma_A ** 2)) * torch.sum(dv * dv)
    else:
        # penalize velocities directly (sum ||v||^2)
        E_time = (1.0 / (sigma_A ** 2)) * torch.sum(v * v)

    return E_time


# -------------------------
# Laplacian (weighted) spatial energy
# -------------------------
def _shift_tensor(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    Shift a tensor with zero padding. t assumed shape (..., nh, nw, C) or (nh,nw) if 2D.
    We'll assume input shape (..., nh, nw, 2) in our use; implement for last-two dims.
    """
    # works for shape (..., H, W, C) or (..., H, W)
    # We'll use torch.roll then zero-out wrapped region to emulate zero padding.
    original_shape = t.shape
    ndims = t.dim()
    H = t.shape[-3]
    W = t.shape[-2] if ndims >= 3 else t.shape[-2]
    # For safety, assume last two dims are (H, W)
    shifted = torch.roll(t, shifts=(dy, dx), dims=(-3, -2))
    # zero out wrapped rows/cols
    if dy > 0:
        shifted[..., :dy, :, ...] = 0.0
    elif dy < 0:
        shifted[..., dy:, :, ...] = 0.0
    if dx > 0:
        shifted[..., :, :dx, ...] = 0.0
    elif dx < 0:
        shifted[..., :, dx:, ...] = 0.0
    return shifted


def compute_laplacian_spatial_energy(
    field: torch.Tensor,
    *,
    alpha: float = 1.0,
    neighbor_set: str = "4",
    sigma_kernel: Optional[float] = None,
    weight_by_image: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute spatial smoothness via weighted Laplacian:
      E_space = alpha * sum_{f,p,q} w_{pq} ||u_{p,f} - u_{q,f}||^2

    Parameters
    ----------
    field : torch.Tensor
        deformation field shape (2, nt, nh, nw)
    alpha : float
        global multiplier
    neighbor_set : '4' or '8'
        which offsets to include (4-neighbors or 8-neighbors)
    sigma_kernel : Optional[float]
        if provided, use gaussian kernel on offset distance to compute per-offset scalar weight:
            w = exp(-d^2 / (2*sigma_kernel^2))
    weight_by_image : Optional[torch.Tensor]
        if provided, should be shape (nt, nh, nw) or (nh, nw) and will modulate per-location weights
        (useful for edge-aware smoothing). Values expected in [0,1] where 1 == no attenuation.
    normalize : bool
        if True, normalize energy by sum of weights so alpha is comparable across kernels.

    Returns
    -------
    E_space : torch.Tensor (scalar)
    """
    assert field.dim() == 4 and field.shape[0] == 2

    _, nt, nh, nw = field.shape
    device = field.device
    dtype = field.dtype

    # define offsets
    if neighbor_set == "4":
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif neighbor_set == "8":
        offsets = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    else:
        raise ValueError("neighbor_set must be '4' or '8'")

    total = torch.tensor(0.0, device=device, dtype=dtype)
    total_weight = 0.0

    # base scalar weight per offset (distance-based)
    for (dy, dx) in offsets:
        d = math.sqrt(dy * dy + dx * dx)
        if sigma_kernel is None:
            w_off = 1.0
        else:
            w_off = math.exp(-(d * d) / (2.0 * (sigma_kernel ** 2)))

        # compute shifted field and difference
        # field shape currently (2, nt, nh, nw) -> move time axis to front for ease: (nt, nh, nw, 2)
        f_t = field.permute(1, 2, 3, 0)  # (nt, nh, nw, 2)
        f_shift = _shift_tensor(f_t, dy=dy, dx=dx)  # (nt, nh, nw, 2)
        diff = f_t - f_shift  # (nt, nh, nw, 2)
        sq = (diff * diff).sum(dim=-1)  # (nt, nh, nw)

        if weight_by_image is not None:
            # broadcast weight_by_image to (nt, nh, nw)
            w_map = weight_by_image
            if w_map.ndim == 2:
                w_map = w_map.unsqueeze(0).expand(nt, -1, -1)
            elif w_map.ndim == 3:
                pass
            else:
                raise ValueError("weight_by_image must be shape (nh,nw) or (nt,nh,nw)")
            total = total + (w_off * w_map * sq).sum()
            total_weight = total_weight + (w_off * w_map).sum().item()
        else:
            total = total + (w_off * sq).sum()
            total_weight = total_weight + w_off * nt * nh * nw

    E_space = alpha * total
    if normalize and total_weight > 0:
        E_space = E_space / (total_weight + 1e-12)

    return E_space


# -------------------------
# Spectral separable spatial energy (Kronecker covariance)
# -------------------------
def _rbf_1d(grid: torch.Tensor, sigma_space: float, sigma_strength: float = 1.0) -> torch.Tensor:
    """
    1D RBF kernel matrix for positions in `grid` (1D coordinates).
    Returns (N,N) matrix.
    """
    diff = grid.unsqueeze(0) - grid.unsqueeze(1)
    dist2 = diff * diff
    K = sigma_strength * torch.exp(-dist2 / (2.0 * (sigma_space ** 2)))
    return K


def compute_spectral_spatial_energy(
    field: torch.Tensor,
    *,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    sigma_space: float = 1.0,
    sigma_strength: float = 1.0,
    rx: Optional[int] = None,
    ry: Optional[int] = None,
    eps: float = 1e-12,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute spectral spatial energy E_space = 0.5 * sum_f v_f^T Sigma^{-1} v_f
    where Sigma = Kx ⊗ Ky is separable (Kronecker). This computes it by:
      1) computing 1D eigenpairs Kx = Ux Lambda_x Ux^T, Ky = Uy Lambda_y Uy^T
      2) projecting velocities into separable eigenbasis:
           a_f[i,j] = sum_{x,y} Ux[x,i] * Uy[y,j] * v_f[x,y]
      3) E_space = 0.5 * sum_f sum_{i,j,coord} a_f[i,j,coord]^2 / (lambda_x[i] * lambda_y[j] + eps)

    Parameters
    ----------
    field : torch.Tensor
        (2, nt, nh, nw) absolute shifts per frame
    nx, ny : Optional[int]
        number of grid points along x and y. If None, uses nh and nw from field.
    sigma_space, sigma_strength : floats
        parameters for RBF kernel Kx, Ky
    rx, ry : Optional[int]
        number of top modes to keep in x and y. If None, keep all.
    eps : float
        numerical regularizer for small eigenvalues.

    Returns
    -------
    E_space : scalar tensor
    info : dict with keys 'lam_x','lam_y','Ux','Uy', useful for reuse
    """
    assert field.dim() == 4 and field.shape[0] == 2
    _, nt, nh, nw = field.shape
    device = device or field.device
    dtype = field.dtype

    nx = nx or nh
    ny = ny or nw

    # build 1D grids (equally spaced in [0,1])
    xs = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)

    Kx = _rbf_1d(xs, sigma_space=sigma_space, sigma_strength=sigma_strength)
    Ky = _rbf_1d(ys, sigma_space=sigma_space, sigma_strength=sigma_strength)

    # eigendecompose (symmetric)
    lam_x_all, Ux_all = torch.linalg.eigh(Kx)  # ascending
    lam_y_all, Uy_all = torch.linalg.eigh(Ky)

    # sort descending
    ix = torch.argsort(lam_x_all, descending=True)
    iy = torch.argsort(lam_y_all, descending=True)
    lam_x_all = lam_x_all[ix]
    Ux_all = Ux_all[:, ix]
    lam_y_all = lam_y_all[iy]
    Uy_all = Uy_all[:, iy]

    if rx is None:
        rx = lam_x_all.shape[0]
    if ry is None:
        ry = lam_y_all.shape[0]

    lam_x = lam_x_all[:rx]  # (rx,)
    Ux = Ux_all[:, :rx]     # (nx, rx)
    lam_y = lam_y_all[:ry]  # (ry,)
    Uy = Uy_all[:, :ry]     # (ny, ry)

    # velocities v: (2, nt-1, nh, nw)
    if nt < 2:
        return torch.tensor(0.0, device=device, dtype=dtype), {
            "lam_x": lam_x,
            "lam_y": lam_y,
            "Ux": Ux,
            "Uy": Uy,
        }
    v = field[:, 1:, :, :] - field[:, :-1, :, :]  # (2, nt-1, nh, nw)

    # project to separable eigenbasis: a[f, i, j, coord] = sum_{x,y} Ux[x,i] * Uy[y,j] * v[coord,f,x,y]
    # Move coords to last dim: v -> (nt-1, nh, nw, 2)
    v_t = v.permute(1, 2, 3, 0)  # (nt-1, nh, nw, 2)
    # compute coefficients using einsum
    # a: (nt-1, rx, ry, 2)
    a = torch.einsum("xi,yj,fxyc->fijc", Ux, Uy, v_t)

    # eigenvalue products lam_x[i] * lam_y[j] -> (rx, ry)
    lam_prod = (lam_x.view(rx, 1) * lam_y.view(1, ry)).to(device=device, dtype=dtype)  # (rx, ry)

    # compute energy: 0.5 * sum_{f,i,j,c} a^2 / (lam_prod + eps)
    denom = lam_prod.unsqueeze(0).unsqueeze(-1) + eps  # (1, rx, ry, 1)
    E_space = 0.5 * torch.sum((a * a) / denom)

    info = {"lam_x": lam_x, "lam_y": lam_y, "Ux": Ux, "Uy": Uy}
    return E_space, info


# -------------------------
# Convenience wrapper
# -------------------------
def compute_energies(
    field: torch.Tensor,
    *,
    sigma_A: float,
    spatial_method: str = "laplacian",
    spatial_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Compute E_time and E_space for given field.

    Parameters
    ----------
    field : torch.Tensor
        (2, nt, nh, nw)
    sigma_A : float
        temporal strength
    spatial_method : 'laplacian' or 'spectral'
    spatial_kwargs : dict
        forwarded to the chosen spatial method.

    Returns
    -------
    (E_time, E_space, info)
    - E_time: scalar tensor
    - E_space: scalar tensor
    - info: dict with extra data (e.g., spectral info) for reuse
    """
    spatial_kwargs = spatial_kwargs or {}
    E_time = compute_temporal_energy(field, sigma_A)

    if spatial_method == "laplacian":
        E_space = compute_laplacian_spatial_energy(field, **spatial_kwargs)
        info = {}
    elif spatial_method == "spectral":
        E_space, info = compute_spectral_spatial_energy(field, **spatial_kwargs)
    else:
        raise ValueError("spatial_method must be 'laplacian' or 'spectral'")

    return E_time, E_space, info
