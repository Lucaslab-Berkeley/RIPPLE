# motion_priors.py
#
# Implements:
#   1. RELION 2019 Zivanov prior (exact GP with exponential kernel)
#   2. Separable Gaussian GP prior (fast Kronecker approximation)
#   3. Laplacian spatial smoothness
#
# All produce E_space and E_time compatible with eqs. (2)–(11)
# for a deformation field: field shape = (2, nt, nh, nw)
#
# Field is absolute shifts s[f,y,x] in Å or pixels.

import torch
import torch.nn.functional as F
import math


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def _pairwise_dist_matrix(coords):
    """
    coords: (P, 2)
    returns dists: (P, P)
    """
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (P, P, 2)
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)
    return dist


def _build_exponential_kernel(coords, sigma_D, sigma_V):
    """
    coords: (P,2)
    Sigma_V(p,q) = sigma_V * exp(-||p-q|| / sigma_D)
    """
    dist = _pairwise_dist_matrix(coords)  # (P,P)
    K = sigma_V*sigma_V * torch.exp(-dist / sigma_D)
    return K


def _build_gaussian_kernel_1d(grid, sigma_space, sigma_strength):
    diff = grid.unsqueeze(0) - grid.unsqueeze(1)
    dist2 = diff * diff
    return sigma_strength * torch.exp(-dist2 / (2 * sigma_space ** 2))


def _build_physical_coords(nh, nw, image_shape, pixel_size, device):
    """
    Returns coords in Angstroms: (P, 2)
    """
    H_px, W_px = image_shape
    H_A = H_px * pixel_size
    W_A = W_px * pixel_size

    yA = torch.linspace(0, H_A, steps=nh, device=device)
    xA = torch.linspace(0, W_A, steps=nw, device=device)

    yyA, xxA = torch.meshgrid(yA, xA, indexing='ij')
    coords = torch.stack([yyA.reshape(-1), xxA.reshape(-1)], dim=-1)

    return coords  # (P, 2)

def _normalize_sigma_fluence(sigma, total_fluence, nt):
    """
    Normalizes sigma_V by the fluence per frame.
    """
    fluence_per_frame = total_fluence / nt
    return sigma * fluence_per_frame


def _create_exponential_sigma_A(total_fluence, n_frames, A=2.0, B=0.1, C=1.0, device=None):
    """
    Create fluence-dependent sigma_A following exponential decay.
    
    sigma_A(fluence) = A * exp(-B * fluence) + C
    
    This allows more motion in early frames (typical for beam-induced motion)
    while maintaining stable minimum smoothness in later frames.
    
    Parameters
    ----------
    total_fluence : float
        Total accumulated fluence (e⁻/Å²)
    n_frames : int
        Number of frames (will create n_frames-2 values for velocity changes)
    A : float
        Amplitude parameter. Default is 2.0.
    B : float
        Decay rate (positive for decay, in units of 1/(e⁻/Å²)). Default is 0.1.
    C : float
        Constant offset (minimum sigma_A). Default is 1.0.
    device : torch.device, optional
        Device for the tensor
        
    Returns
    -------
    torch.Tensor
        Shape (n_frames-2,) containing sigma_A values for each acceleration
        (velocity change between frames)
        
    Notes
    -----
    Using fluence instead of frame index makes the decay independent of
    fluence_per_frame, ensuring consistent behavior across different
    dose rates.
    """
    # For n_frames, we have n_frames-1 velocities and n_frames-2 velocity changes
    # Compute fluence at each velocity change point
    fluence_per_frame = total_fluence / n_frames
    
    # Fluence at the midpoint of each velocity change interval
    # v[0]->v[1] happens around fluence 1.0*fluence_per_frame
    # v[1]->v[2] happens around fluence 2.0*fluence_per_frame, etc.
    frame_indices = torch.arange(n_frames - 2, dtype=torch.float32, device=device)
    fluence_values = (frame_indices + 1.0) * fluence_per_frame
    
    # sigma_A = A * exp(-B * fluence) + C
    sigma_A = A * torch.exp(-B * fluence_values) + C
    return sigma_A


def _compute_physical_spacing(image_shape, pixel_size, grid_resolution, total_fluence=None):
    """
    Compute physical spacing between grid points.
    
    image_shape: (H, W) in pixels
    pixel_size: Angstroms per pixel
    grid_resolution: (nt, nh, nw)
    total_fluence: total fluence in e-/Å² (optional, for temporal spacing)
    
    Returns:
        spatial_spacing: (dy, dx) in Angstroms
        temporal_spacing: dt in e-/Å² (or None if total_fluence not provided)
    """
    H_px, W_px = image_shape
    nt, nh, nw = grid_resolution
    
    # Physical image size
    H_A = H_px * pixel_size
    W_A = W_px * pixel_size
    
    # Spacing between grid points
    dy = H_A / (nh - 1) if nh > 1 else H_A
    dx = W_A / (nw - 1) if nw > 1 else W_A
    spatial_spacing = (dy, dx)
    
    # Temporal spacing (fluence per frame)
    if total_fluence is not None:
        dt = total_fluence / (nt - 1) if nt > 1 else total_fluence
        temporal_spacing = dt
    else:
        temporal_spacing = None
    
    return spatial_spacing, temporal_spacing

# --------------------------------------------------------------------------
# 1) EXACT RELION 2019 PRIOR (Exponential kernel)  --------------------------
# --------------------------------------------------------------------------

def relion2019_eigendecompose(coords, sigma_D, sigma_V, top_k=None):
    """
    coords: (P,2) coordinates of control points
    Returns eigenvalues λ, eigenvectors W, and basis B = W * sqrt(λ).
    top_k: if not None, truncate to top_k modes
    """

    P = coords.shape[0]
    K = _build_exponential_kernel(coords, sigma_D, sigma_V)  # (P,P)

    # eigendecompose kernel (symmetric PSD)
    vals, vecs = torch.linalg.eigh(K)

    # sort descending
    idx = torch.argsort(vals, descending=True)
    vals = vals[idx]
    vecs = vecs[:, idx]

    if top_k is not None:
        vals = vals[:top_k]
        vecs = vecs[:, :top_k]

    # basis b_i = sqrt(lambda_i) * w_i  (Eq. 3)
    B = vecs * torch.sqrt(vals + 1e-12)

    return vals, vecs, B


def relion2019_Espace(c):
    """
    E_space = mean |c_{i,f}|^2
    
    Using mean instead of sum for scale-invariance when number of control
    points differs from number of particles.
    
    c: (F, R, 2)   (R=number of eigenmodes)
    """
    return torch.mean(c * c)


def relion2019_Etime(c, lam, sigma_A):
    """
    Temporal smoothness in eigenmode basis.
    
    E_time = 1/sigma_A^2 * mean lam_i * |c_{i,f} - c_{i,f-1}|^2
    
    c: (F, R, 2) - coefficients in eigenmode basis (F frames → F-1 velocities)
    lam: (R,) - eigenvalues
    sigma_A: acceleration scale. Can be scalar or tensor of shape (F-2,) for
             frame-dependent smoothness (F-1 velocities → F-2 velocity changes).
    """
    diffs = c[1:] - c[:-1]   # (F-2, R, 2) - velocity changes
    lam = lam.view(1, -1, 1)
    
    # Handle sigma_A as scalar or tensor
    if isinstance(sigma_A, torch.Tensor) and sigma_A.ndim > 0:
        # sigma_A is (F-2,) tensor - reshape for broadcasting over (F-2, 1, 1)
        sigma_A_sq = (sigma_A ** 2).view(-1, 1, 1)
    else:
        sigma_A_sq = sigma_A ** 2
    
    return torch.mean(lam * (diffs * diffs) / sigma_A_sq)


# --------------------------------------------------------------------------
# 2) FAST SEPARABLE GAUSSIAN PRIOR (Kronecker) ------------------------------
# --------------------------------------------------------------------------

def separable_eigendecompose(nx, ny, sigma_space, sigma_strength, device):
    """
    Returns lam_x, Ux, lam_y, Uy
    """
    xs = torch.linspace(0, 1, nx, device=device)
    ys = torch.linspace(0, 1, ny, device=device)

    Kx = _build_gaussian_kernel_1d(xs, sigma_space, sigma_strength)
    Ky = _build_gaussian_kernel_1d(ys, sigma_space, sigma_strength)

    lam_x, Ux = torch.linalg.eigh(Kx)
    lam_y, Uy = torch.linalg.eigh(Ky)

    ix = torch.argsort(lam_x, descending=True)
    iy = torch.argsort(lam_y, descending=True)

    lam_x = lam_x[ix]
    lam_y = lam_y[iy]
    Ux = Ux[:, ix]
    Uy = Uy[:, iy]

    return lam_x, Ux, lam_y, Uy


def separable_project_velocities(v, Ux, Uy):
    """
    v: (F, ny, nx, 2)
    returns a: coefficients (F, rx, ry, 2)
    """
    return torch.einsum("xi,yj,fxyc->fijc", Ux, Uy, v)


def separable_Espace(a):
    """
    E_space = sum_f sum_ij |a_ij,f|^2
    Equivalent to eq. (5).
    """
    return torch.sum(a * a)


def separable_Etime(a, lam_x, lam_y, sigma_A):
    """
    Derived from RELION eq. (11):
    lam_ij = lam_x[i] * lam_y[j]
    E_time = 1/sigma_A^2 * sum_f sum_{i,j} lam_ij |a[f,i,j] - a[f-1,i,j]|^2
    """
    diffs = a[1:] - a[:-1]  # (F-1, rx, ry, 2)

    lam = lam_x.view(1, -1, 1, 1) * lam_y.view(1, 1, -1, 1)
    return (1.0 / sigma_A ** 2) * torch.sum(lam * (diffs * diffs))


# --------------------------------------------------------------------------
# 3) LAPLACIAN PRIOR --------------------------------------------------------
# --------------------------------------------------------------------------

def laplacian_Espace(field, alpha=1.0, spatial_spacing=None):
    """
    Spatial smoothness with physical distance weighting.
    
    field: (2, nt, nh, nw)
    alpha: regularization strength (dimensionless)
    spatial_spacing: (dy_physical, dx_physical) spacing in Angstroms
                     If None, assumes unit spacing
    
    E_space = alpha * mean |(u - u_neighbor)/distance|^2
            = alpha * mean |u - u_neighbor|^2 / distance^2
    
    Physical interpretation: closer points (smaller spacing) should be more correlated,
    so differences between them are penalized more (divide by smaller distance^2).
    """
    _, nt, nh, nw = field.shape
    field2 = field.permute(1, 2, 3, 0)  # (nt, nh, nw, 2)

    if spatial_spacing is None:
        dy_phys, dx_phys = 1.0, 1.0
    else:
        dy_phys, dx_phys = spatial_spacing

    total = 0.0
    count = 0
    
    # Vertical neighbors (dy direction)
    for dy in [1, -1]:
        rolled = torch.roll(field2, shifts=(dy, 0), dims=(1, 2))
        if dy > 0:
            rolled[:, :dy, :, :] = 0
        else:
            rolled[:, dy:, :, :] = 0
        diff = field2 - rolled
        # Weight by 1/dy_phys^2 (units: spatial gradient squared)
        total = total + torch.sum(diff * diff) / (dy_phys ** 2)
        count += diff.numel()
    
    # Horizontal neighbors (dx direction)
    for dx in [1, -1]:
        rolled = torch.roll(field2, shifts=(0, dx), dims=(1, 2))
        if dx > 0:
            rolled[:, :, :dx, :] = 0
        else:
            rolled[:, :, dx:, :] = 0
        diff = field2 - rolled
        # Weight by 1/dx_phys^2
        total = total + torch.sum(diff * diff) / (dx_phys ** 2)
        count += diff.numel()

    return alpha * (total / count)


def laplacian_Etime(field, sigma_A, temporal_spacing=None):
    """
    Temporal smoothness with physical fluence weighting.
    
    field: (2, nt, nh, nw)
    sigma_A: acceleration scale in Å/(e⁻/Å²) if temporal_spacing provided,
             otherwise Å/frame. Can be scalar or tensor of shape (nt-2,) for
             frame-dependent smoothness.
    temporal_spacing: fluence per frame (e⁻/Å²)
                      If None, assumes unit spacing
    
    E_time = 1/sigma_A^2 * mean |(v[f] - v[f-1])/dt|^2
           = 1/(sigma_A^2 * dt^2) * mean |v[f] - v[f-1]|^2
    
    Physical interpretation: frames closer in fluence (smaller dt) should have
    more correlated velocities, so velocity changes are penalized more (divide by smaller dt^2).
    """
    v = field[:, 1:] - field[:, :-1]         # (2, nt-1, nh, nw)
    dv = v[:, 1:] - v[:, :-1]                # (2, nt-2, nh, nw)
    
    if temporal_spacing is None:
        dt = 1.0
    else:
        dt = temporal_spacing
    
    # Handle sigma_A as scalar or tensor
    if isinstance(sigma_A, torch.Tensor) and sigma_A.ndim > 0:
        # sigma_A is (nt-2,) tensor - reshape for broadcasting
        # dv is (2, nt-2, nh, nw), we want sigma_A to broadcast over (1, nt-2, 1, 1)
        sigma_A_sq = (sigma_A ** 2).view(1, -1, 1, 1)
    else:
        sigma_A_sq = sigma_A ** 2
    
    # Weight by 1/dt^2 (units: dv/dt is acceleration)
    return torch.mean(dv * dv / (sigma_A_sq * dt ** 2))


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------

def relion2019_compute(field, coords, sigma_D, sigma_V, sigma_A, top_k=None):
    """
    field: (2, nt, P_y, P_x)
    coords: (P, 2)
    returns E_space, E_time
    """
    _, nt, nh, nw = field.shape
    P = nh * nw

    #field[:, 1:]: (2, nt-1, P_y, P_x) - frames 1 through nt-1
    #field[:, :-1]: (2, nt-1, P_y, P_x) - frames 0 through nt-1
    #Difference: (2, nt-1, P_y, P_x) - velocity between consecutive frames
    #.permute(1, 2, 3, 0): (nt-1, P_y, P_x, 2) - moves time first, components last
    #.reshape(nt-1, P, 2): (nt-1, P_y * P_x, 2) - flattens spatial grid
    # velocities: (F, P, 2)
    v = (field[:, 1:] - field[:, :-1]).permute(1, 2, 3, 0).reshape(nt-1, P, 2)

    # eigendecompose Σ_V
    lam, vecs, B = relion2019_eigendecompose(coords, sigma_D, sigma_V, top_k)
    #lam: (R,) - eigenvalues (R ≤ P, possibly truncated)
    #vecs: (P, R) - eigenvectors (each column is one eigenvector)
    #B: (P, R) - basis vectors

    # project velocities: c[f,i,d], using v[f,p,d] = sum_i B[p,i] c[f,i,d]
    #Bt = B.t()  # (R, P)
    #c = torch.einsum("rp,fpc->frc", Bt, v)  # (F-1, R, 2)

    c = torch.einsum("rp,fpc->frc", vecs.t(), v) / torch.sqrt(lam + 1e-12).view(1, -1, 1)
    #vecs.t(): (r, p) - eigenvector matrix transposed
    #v: (f, p, 2) - velocities, 2 for x and y components
    #c: (f, r, 2) - coefficients

    E_space = relion2019_Espace(c)
    E_time = relion2019_Etime(c, lam, sigma_A)

    return E_space, E_time


def separable_compute(field, lam_x, Ux, lam_y, Uy, sigma_A):
    """
    field: (2, nt, nh, nw)
    """
    _, nt, nh, nw = field.shape

    # velocities: (F, ny, nx, 2)
    v = (field[:, 1:] - field[:, :-1]).permute(1, 2, 3, 0)

    a = separable_project_velocities(v, Ux, Uy)
    E_space = separable_Espace(a)
    E_time = separable_Etime(a, lam_x, lam_y, sigma_A)

    return E_space, E_time


def laplacian_compute(field, sigma_A, alpha=1.0, spatial_spacing=None, temporal_spacing=None):
    """
    Compute Laplacian spatial and temporal energies with physical spacing.
    
    field: (2, nt, nh, nw)
    sigma_A: temporal smoothness parameter
    alpha: spatial smoothness strength
    spatial_spacing: (dy, dx) in Angstroms
    temporal_spacing: dt in fluence (e-/Å²)
    """
    E_space = laplacian_Espace(field, alpha, spatial_spacing)
    E_time = laplacian_Etime(field, sigma_A, temporal_spacing)
    return E_space, E_time
