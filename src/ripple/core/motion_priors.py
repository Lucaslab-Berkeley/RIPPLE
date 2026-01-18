"""Motion priors for deformation field regularization.

This module implements several motion prior models for regularizing deformation
fields in cryo-EM movie alignment:

1. RELION 2019 Zivanov prior: Exact Gaussian process with exponential kernel
2. Separable Gaussian GP prior: Fast Kronecker approximation
3. Laplacian spatial smoothness: Simple spatial and temporal smoothness constraints

All implementations produce E_space and E_time energy terms compatible with
equations (2)-(11) for a deformation field with shape (2, nt, nh, nw).

The deformation field represents absolute shifts s[f,y,x] in Angstroms or pixels,
where:
    - First dimension (2): y and x components
    - Second dimension (nt): Number of time frames
    - Third dimension (nh): Number of control points in height
    - Fourth dimension (nw): Number of control points in width
"""

import torch


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def _pairwise_dist_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    Builds a pairwise distance matrix between coordinates.

    Parameters
    ----------
    coords : torch.Tensor
        (P, 2) coordinates of control points

    Returns
    -------
    dist : torch.Tensor
        (P, P) pairwise distance matrix
    """
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (P, P, 2)
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)
    return dist


def _build_exponential_kernel(
    coords: torch.Tensor,
    sigma_d: float,
    sigma_v: float,
) -> torch.Tensor:
    """
    Builds an exponential kernel between coordinates.

    Parameters
    ----------
    coords : torch.Tensor
        (P, 2) coordinates of control points
    sigma_d : float
        Spatial correlation length
    sigma_v : float
        Velocity magnitude scale

    Returns
    -------
    kernel_matrix : torch.Tensor
        (P, P) exponential kernel matrix

    Notes
    -----
    Sigma_V(p,q) = sigma_v * exp(-||p-q|| / sigma_d)
    """
    dist = _pairwise_dist_matrix(coords)  # (P,P)
    kernel_matrix = sigma_v * sigma_v * torch.exp(-dist / sigma_d)
    return kernel_matrix


def _build_gaussian_kernel_1d(
    grid: torch.Tensor,
    sigma_space: float,
    sigma_strength: float,
) -> torch.Tensor:
    """
    Builds a Gaussian kernel between grid points.

    Parameters
    ----------
    grid : torch.Tensor
        (N, 1) grid points
    sigma_space : float
        Spatial correlation length
    sigma_strength : float
        Strength of the Gaussian kernel

    Returns
    -------
    K : torch.Tensor
        (N, N) Gaussian kernel matrix

    Notes
    -----
    K(p,q) = sigma_strength * exp(-||p-q||^2 / (2 * sigma_space^2))
    """
    diff = grid.unsqueeze(0) - grid.unsqueeze(1)
    dist2 = diff * diff
    return sigma_strength * torch.exp(-dist2 / (2 * sigma_space**2))


def _build_physical_coords(
    nh: int,
    nw: int,
    image_shape: tuple[int, int],
    pixel_size: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Builds physical coordinates in Angstroms.

    Parameters
    ----------
    nh : int
        Number of height pixels
    nw : int
        Number of width pixels
    image_shape : tuple[int, int]
        (H, W) image shape
    pixel_size : float
        Pixel size in Angstroms
    device : torch.device
        Device to use

    Returns
    -------
    coords : torch.Tensor
        (P, 2) coordinates in Angstroms
    """
    height_px, width_px = image_shape
    height_angstrom = height_px * pixel_size
    width_angstrom = width_px * pixel_size

    y_angstrom = torch.linspace(0, height_angstrom, steps=nh, device=device)
    x_angstrom = torch.linspace(0, width_angstrom, steps=nw, device=device)

    yy_angstrom, xx_angstrom = torch.meshgrid(y_angstrom, x_angstrom, indexing="ij")
    coords = torch.stack([yy_angstrom.reshape(-1), xx_angstrom.reshape(-1)], dim=-1)

    return coords  # (P, 2)


def _normalize_sigma_fluence(
    sigma: float | torch.Tensor,
    total_fluence: float,
    nt: int,
) -> float | torch.Tensor:
    """
    Normalizes sigma by the fluence per frame.

    Parameters
    ----------
    sigma : float | torch.Tensor
        Sigma value(s) to normalize. Can be scalar or tensor for per-frame values.
    total_fluence : float
        Total fluence in e-/Å²
    nt : int
        Number of frames

    Returns
    -------
    float | torch.Tensor
        Normalized sigma. Returns float if input is float, tensor if input is tensor.
    """
    fluence_per_frame = total_fluence / nt
    return sigma * fluence_per_frame


# pylint: disable=too-many-arguments,too-many-positional-arguments
def _create_exponential_sigma_a(
    total_fluence: float,
    n_frames: int,
    amplitude: float = 2.0,
    decay_rate: float = 0.1,
    offset: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create fluence-dependent sigma_A following exponential decay.

    sigma_A(fluence) = amplitude * exp(-decay_rate * fluence) + offset

    This allows more motion in early frames (typical for beam-induced motion)
    while maintaining stable minimum smoothness in later frames.

    Parameters
    ----------
    total_fluence : float
        Total accumulated fluence in electrons per Angstrom squared
    n_frames : int
        Number of frames (will create n_frames-2 values for velocity changes)
    amplitude : float, optional
        Amplitude parameter. Default is 2.0.
    decay_rate : float, optional
        Decay rate (positive for decay, in units of 1/(e⁻/Å²)). Default is 0.1.
    offset : float, optional
        Constant offset (minimum sigma_A). Default is 1.0.
    device : torch.device, optional
        Device for the tensor. Default is None.

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

    # sigma_a = amplitude * exp(-decay_rate * fluence) + offset
    sigma_a = amplitude * torch.exp(-decay_rate * fluence_values) + offset
    return sigma_a


# pylint: disable=too-many-locals
def _compute_physical_spacing(
    image_shape: tuple[int, int],
    pixel_size: float,
    grid_resolution: tuple[int, int, int],
    total_fluence: float | None = None,
) -> tuple[tuple[float, float], float | None]:
    """
    Compute physical spacing between grid points.

    Parameters
    ----------
    image_shape : tuple[int, int]
        (H, W) in pixels
    pixel_size : float
        Angstroms per pixel
    grid_resolution : tuple[int, int, int]
        (nt, nh, nw)
    total_fluence : float | None, optional
        Total fluence in e-/Å² (optional, for temporal spacing)

    Returns
    -------
    spatial_spacing : tuple[float, float]
        (dy, dx) in Angstroms
    temporal_spacing : float | None
        dt in e-/Å² (or None if total_fluence not provided)
    """
    height_px, width_px = image_shape
    nt, nh, nw = grid_resolution

    # Physical image size
    height_angstrom = height_px * pixel_size
    width_angstrom = width_px * pixel_size

    # Spacing between grid points
    dy = height_angstrom / (nh - 1) if nh > 1 else height_angstrom
    dx = width_angstrom / (nw - 1) if nw > 1 else width_angstrom
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


def relion2019_eigendecompose(
    coords: torch.Tensor,
    sigma_d: float,
    sigma_v: float,
    top_k: float | None = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Eigendecomposes the exponential kernel.

    Parameters
    ----------
    coords : torch.Tensor
        (P, 2) coordinates of control points
    sigma_d : float
        Spatial correlation length
    sigma_v : float
        Velocity magnitude scale
    top_k : float | None, optional
        Fraction of modes to keep (between 0 and 1). If None, keeps all modes.
        Default is 0.2 (keeps top 20% of modes).

    Returns
    -------
    vals : torch.Tensor
        (R,) eigenvalues
    vecs : torch.Tensor
        (P, R) eigenvectors
    B : torch.Tensor
        (P, R) basis vectors
    """
    kernel_matrix = _build_exponential_kernel(coords, sigma_d, sigma_v)  # (P,P)

    # eigendecompose kernel (symmetric PSD)
    vals, vecs = torch.linalg.eigh(kernel_matrix)  # pylint: disable=not-callable

    # sort descending
    idx = torch.argsort(vals, descending=True)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Truncate to top_k fraction of modes
    if top_k is not None:
        num_points = coords.shape[0]
        k = int(top_k * num_points)
        # Ensure at least 1 mode and at most num_points modes
        k = max(1, min(k, num_points))
        if k < len(vals):
            vals = vals[:k]
            vecs = vecs[:, :k]

    # Clamp eigenvalues to be non-negative (numerical errors can cause small negatives)
    vals = torch.clamp(vals, min=0.0)

    # basis b_i = sqrt(lambda_i) * w_i  (Eq. 3)
    basis_vectors = vecs * torch.sqrt(vals + 1e-12)

    return vals, vecs, basis_vectors


def relion2019_e_space(c: torch.Tensor) -> torch.Tensor:
    """
    Computes the spatial energy.

    Parameters
    ----------
    c : torch.Tensor
        (F, R, 2) coefficients in eigenmode basis

    Returns
    -------
    e_space : torch.Tensor
        Spatial energy

    Notes
    -----
    e_space = mean |c_{i,f}|^2
    """
    return torch.mean(c * c)


def relion2019_e_time(
    c: torch.Tensor,
    lam: torch.Tensor,
    sigma_a: float | torch.Tensor,
) -> torch.Tensor:
    """
    Temporal smoothness in eigenmode basis.

    Parameters
    ----------
    c : torch.Tensor
        (F, R, 2) coefficients in eigenmode basis
    lam : torch.Tensor
        (R,) eigenvalues
    sigma_a : float
        Acceleration scale

    Returns
    -------
    e_time : torch.Tensor
        Temporal energy

    Notes
    -----
    e_time = 1/sigma_A^2 * mean lam_i * |c_{i,f} - c_{i,f-1}|^2

    c: (F, R, 2) - coefficients in eigenmode basis (F frames → F-1 velocities)
    lam: (R,) - eigenvalues
    sigma_A: acceleration scale. Can be scalar or tensor of shape (F-2,) for
             frame-dependent smoothness (F-1 velocities → F-2 velocity changes).
    """
    diffs = c[1:] - c[:-1]  # (F-2, R, 2) - velocity changes
    lam = lam.view(1, -1, 1)

    # Handle sigma_a as scalar or tensor
    if isinstance(sigma_a, torch.Tensor) and sigma_a.ndim > 0:
        # sigma_a is (F-2,) tensor - reshape for broadcasting over (F-2, 1, 1)
        sigma_a_sq = (sigma_a**2).view(-1, 1, 1)
    else:
        sigma_a_sq = sigma_a**2

    return torch.mean(lam * (diffs * diffs) / sigma_a_sq)


# --------------------------------------------------------------------------
# 2) FAST SEPARABLE GAUSSIAN PRIOR (Kronecker) ------------------------------
# --------------------------------------------------------------------------


def separable_eigendecompose(
    nx: int,
    ny: int,
    sigma_space: float,
    sigma_strength: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Eigendecomposes the Gaussian kernels.

    Parameters
    ----------
    nx : int
        Number of x grid points
    ny : int
        Number of y grid points
    sigma_space : float
        Spatial correlation length
    sigma_strength : float
        Strength of the Gaussian kernel
    device : torch.device
        Device to use

    Returns
    -------
    lam_x : torch.Tensor
        (nx,) eigenvalues
    eigenvectors_x : torch.Tensor
        (nx, nx) eigenvectors
    lam_y : torch.Tensor
        (ny,) eigenvalues
    eigenvectors_y : torch.Tensor
        (ny, ny) eigenvectors
    """
    xs = torch.linspace(0, 1, nx, device=device)
    ys = torch.linspace(0, 1, ny, device=device)

    kernel_x = _build_gaussian_kernel_1d(xs, sigma_space, sigma_strength)
    kernel_y = _build_gaussian_kernel_1d(ys, sigma_space, sigma_strength)

    lam_x, eigenvectors_x = torch.linalg.eigh(kernel_x)  # pylint: disable=not-callable
    lam_y, eigenvectors_y = torch.linalg.eigh(kernel_y)  # pylint: disable=not-callable

    index_x = torch.argsort(lam_x, descending=True)
    index_y = torch.argsort(lam_y, descending=True)

    lam_x = lam_x[index_x]
    lam_y = lam_y[index_y]
    eigenvectors_x = eigenvectors_x[:, index_x]
    eigenvectors_y = eigenvectors_y[:, index_y]

    return lam_x, eigenvectors_x, lam_y, eigenvectors_y


def separable_project_velocities(
    v: torch.Tensor,
    eigenvectors_x: torch.Tensor,
    eigenvectors_y: torch.Tensor,
) -> torch.Tensor:
    """
    Projects velocities onto the eigenmodes.

    Parameters
    ----------
    v : torch.Tensor
        (F, ny, nx, 2) velocities
    eigenvectors_x : torch.Tensor
        (nx, nx) eigenvectors
    eigenvectors_y : torch.Tensor
        (ny, ny) eigenvectors

    Returns
    -------
    a : torch.Tensor
        (F, rx, ry, 2) coefficients
    """
    return torch.einsum("xi,yj,fxyc->fijc", eigenvectors_x, eigenvectors_y, v)


def separable_e_space(a: torch.Tensor) -> torch.Tensor:
    """
    Computes the spatial energy.

    Parameters
    ----------
    a : torch.Tensor
        (F, rx, ry, 2) coefficients

    Returns
    -------
    e_space : torch.Tensor
        Spatial energy

    Notes
    -----
    e_space = sum_f sum_ij |a_ij,f|^2
    Equivalent to eq. (5).
    """
    return torch.sum(a * a)


def separable_e_time(
    a: torch.Tensor,
    lam_x: torch.Tensor,
    lam_y: torch.Tensor,
    sigma_a: float,
) -> torch.Tensor:
    """
    Computes the temporal energy.

    Parameters
    ----------
    a : torch.Tensor
        (F, rx, ry, 2) coefficients
    lam_x : torch.Tensor
        (nx,) eigenvalues
    lam_y : torch.Tensor
        (ny,) eigenvalues
    sigma_a : float
        Acceleration scale

    Returns
    -------
    e_time : torch.Tensor
        Temporal energy

    Notes
    -----
    Derived from RELION eq. (11):
    lam_ij = lam_x[i] * lam_y[j]
    e_time = 1/sigma_A^2 * sum_f sum_{i,j} lam_ij |a[f,i,j] - a[f-1,i,j]|^2
    """
    diffs = a[1:] - a[:-1]  # (F-1, rx, ry, 2)

    lam = lam_x.view(1, -1, 1, 1) * lam_y.view(1, 1, -1, 1)
    return (1.0 / sigma_a**2) * torch.sum(lam * (diffs * diffs))


# --------------------------------------------------------------------------
# 3) LAPLACIAN PRIOR --------------------------------------------------------
# --------------------------------------------------------------------------


def laplacian_e_space(
    field: torch.Tensor,
    alpha: float = 1.0,
    spatial_spacing: tuple[float, float] | None = None,
) -> torch.Tensor:
    """
    Spatial smoothness with physical distance weighting.

    Parameters
    ----------
    field : torch.Tensor
        (2, nt, nh, nw)
    alpha : float, optional
        Regularization strength (dimensionless)
    spatial_spacing : tuple[float, float] | None, optional
        (dy_physical, dx_physical) spacing in Angstroms
        If None, assumes unit spacing

    Returns
    -------
    e_space : torch.Tensor
        Spatial energy

    Notes
    -----
    e_space = alpha * mean |(u - u_neighbor)/distance|^2
            = alpha * mean |u - u_neighbor|^2 / distance^2
    """
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
        total = total + torch.sum(diff * diff) / (dy_phys**2)
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
        total = total + torch.sum(diff * diff) / (dx_phys**2)
        count += diff.numel()

    return alpha * (total / count)


def laplacian_e_time(
    field: torch.Tensor,
    sigma_a: float | torch.Tensor,
    temporal_spacing: float | None = None,
) -> torch.Tensor:
    """
    Temporal smoothness with physical fluence weighting.

    Parameters
    ----------
    field : torch.Tensor
        (2, nt, nh, nw)
    sigma_a : float
        Acceleration scale in Å/(e⁻/Å²) if temporal_spacing provided,
        otherwise Å/frame. Can be scalar or tensor of shape (nt-2,) for
        frame-dependent smoothness.
    temporal_spacing : float | None, optional
        Fluence per frame (e⁻/Å²)
        If None, assumes unit spacing

    Returns
    -------
    e_time : torch.Tensor
        Temporal energy

    Notes
    -----
    e_time = 1/sigma_A^2 * mean |(v[f] - v[f-1])/dt|^2
           = 1/(sigma_A^2 * dt^2) * mean |v[f] - v[f-1]|^2
    """
    v = field[:, 1:] - field[:, :-1]  # (2, nt-1, nh, nw)
    dv = v[:, 1:] - v[:, :-1]  # (2, nt-2, nh, nw)

    if temporal_spacing is None:
        dt = 1.0
    else:
        dt = temporal_spacing

    # Handle sigma_a as scalar or tensor
    if isinstance(sigma_a, torch.Tensor) and sigma_a.ndim > 0:
        # sigma_a is (nt-2,) tensor - reshape for broadcasting
        # dv is (2, nt-2, nh, nw), we want sigma_a to broadcast over (1, nt-2, 1, 1)
        sigma_a_sq = (sigma_a**2).view(1, -1, 1, 1)
    else:
        sigma_a_sq = sigma_a**2

    # Weight by 1/dt^2 (units: dv/dt is acceleration)
    return torch.mean(dv * dv / (sigma_a_sq * dt**2))


# --------------------------------------------------------------------------
# Convenience wrappers
# --------------------------------------------------------------------------
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def relion2019_compute(
    field: torch.Tensor,
    coords: torch.Tensor,
    sigma_d: float,
    sigma_v: float,
    sigma_a: float | torch.Tensor,
    top_k: float | None = 0.2,
    lam: torch.Tensor | None = None,
    vecs: torch.Tensor | None = None,
    is_particle_shifts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the RELION 2019 energy.

    Parameters
    ----------
    field : torch.Tensor
        For deformation_field: (2, nt, nh, nw)
        For particle_shifts: (T, N, 2) where T is frames, N is particles
    coords : torch.Tensor
        (P, 2) coordinates
    sigma_d : float
        Spatial correlation length
    sigma_v : float
        Velocity magnitude scale
    sigma_a : float
        Acceleration scale
    top_k : float | None, optional
        Fraction of modes to keep (between 0 and 1). If None, keeps all modes.
        Default is 0.2 (keeps top 20% of modes).
    lam : torch.Tensor | None, optional
        Precomputed eigenvalues (R,). If None, will be computed from coords, sigma_d, sigma_v.
    vecs : torch.Tensor | None, optional
        Precomputed eigenvectors (P, R). If None, will be computed from coords, sigma_d, sigma_v.
        Must be provided if lam is provided.
    is_particle_shifts : bool, optional
        If True, field is in particle_shifts format (T, N, 2).
        If False, field is in deformation_field format (2, nt, nh, nw).
        Default is False.

    Returns
    -------
    e_space : torch.Tensor
        Spatial energy
    E_time : torch.Tensor
        Temporal energy

    Notes
    -----
    e_space = mean |c_{i,f}|^2
    e_time = 1/sigma_A^2 * mean lam_i * |c_{i,f} - c_{i,f-1}|^2
    """
    if is_particle_shifts:
        # field is (T, N, 2) where T is frames, N is particles
        # Compute velocities directly: (T-1, N, 2)
        v = field[1:] - field[:-1]  # (T-1, N, 2)
        num_points = v.shape[1]  # N
    else:
        # field is (2, nt, nh, nw) - deformation field format
        _, nt, nh, nw = field.shape
        num_points = nh * nw

        # field[:, 1:]: (2, nt-1, P_y, P_x) - frames 1 through nt-1
        # field[:, :-1]: (2, nt-1, P_y, P_x) - frames 0 through nt-1
        # Difference: (2, nt-1, P_y, P_x) - velocity between consecutive frames
        # .permute(1, 2, 3, 0): (nt-1, P_y, P_x, 2) - moves time first, components last
        # .reshape(nt-1, num_points, 2): (nt-1, P_y * P_x, 2) - flattens spatial grid
        # velocities: (F, num_points, 2)
        v = (
            (field[:, 1:] - field[:, :-1])
            .permute(1, 2, 3, 0)
            .reshape(nt - 1, num_points, 2)
        )

    # eigendecompose Σ_V (only if not provided)
    if lam is None or vecs is None:
        lam, vecs, _ = relion2019_eigendecompose(coords, sigma_d, sigma_v, top_k)
    # lam: (R,) - eigenvalues (R ≤ P, possibly truncated)
    # vecs: (P, R) - eigenvectors (each column is one eigenvector)
    # B: (P, R) - basis vectors

    # Clamp eigenvalues to be non-negative (numerical errors can cause small negatives)
    lam_clamped = torch.clamp(lam, min=0.0)

    # Compute sqrt(lam + eps) for division
    sqrt_lam = torch.sqrt(lam_clamped + 1e-12)

    # project velocities: c[f,i,d], using v[f,p,d] = sum_i B[p,i] c[f,i,d]
    # Bt = B.t()  # (R, P)
    # c = torch.einsum("rp,fpc->frc", Bt, v)  # (F-1, R, 2)

    c = torch.einsum("rp,fpc->frc", vecs.t(), v) / sqrt_lam.view(1, -1, 1)
    # vecs.t(): (r, p) - eigenvector matrix transposed
    # v: (f, p, 2) - velocities, 2 for x and y components
    # c: (f, r, 2) - coefficients

    e_space = relion2019_e_space(c)
    e_time = relion2019_e_time(c, lam_clamped, sigma_a)

    return e_space, e_time


# pylint: disable=too-many-arguments,too-many-positional-arguments
def separable_compute(
    field: torch.Tensor,
    lam_x: torch.Tensor,
    eigenvectors_x: torch.Tensor,
    lam_y: torch.Tensor,
    eigenvectors_y: torch.Tensor,
    sigma_a: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the separable energy.

    Parameters
    ----------
    field : torch.Tensor
        (2, nt, nh, nw)
    lam_x : torch.Tensor
        (nx,) eigenvalues
    eigenvectors_x : torch.Tensor
        (nx, nx) eigenvectors
    lam_y : torch.Tensor
        (ny,) eigenvalues
    eigenvectors_y : torch.Tensor
        (ny, ny) eigenvectors
    sigma_a : float
        Acceleration scale

    Returns
    -------
    e_space : torch.Tensor
        Spatial energy
    e_time : torch.Tensor
        Temporal energy

    Notes
    -----
    e_space = sum_f sum_ij |a_ij,f|^2
    e_time = 1/sigma_a^2 * sum_f sum_{i,j} lam_ij |a[f,i,j] - a[f-1,i,j]|^2
    """
    # velocities: (F, ny, nx, 2)
    v = (field[:, 1:] - field[:, :-1]).permute(1, 2, 3, 0)

    a = separable_project_velocities(v, eigenvectors_x, eigenvectors_y)
    e_space = separable_e_space(a)
    e_time = separable_e_time(a, lam_x, lam_y, sigma_a)

    return e_space, e_time


def laplacian_compute(
    field: torch.Tensor,
    sigma_a: float | torch.Tensor,
    alpha: float = 1.0,
    spatial_spacing: tuple[float, float] | None = None,
    temporal_spacing: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Laplacian spatial and temporal energies with physical spacing.

    Parameters
    ----------
    field : torch.Tensor
        (2, nt, nh, nw)
    sigma_a : float
        Temporal smoothness parameter
    alpha : float, optional
        Spatial smoothness strength
    spatial_spacing : tuple[float, float] | None, optional
        (dy, dx) in Angstroms
    temporal_spacing : float | None, optional
        dt in fluence (e-/Å²)

    Returns
    -------
    e_space : torch.Tensor
        Spatial energy
    E_time : torch.Tensor
        Temporal energy

    Notes
    -----
    e_space = alpha * mean |(u - u_neighbor)/distance|^2
            = alpha * mean |u - u_neighbor|^2 / distance^2
    e_time = 1/sigma_A^2 * mean |(v[f] - v[f-1])/dt|^2
           = 1/(sigma_A^2 * dt^2) * mean |v[f] - v[f-1]|^2
    """
    e_space = laplacian_e_space(field, alpha, spatial_spacing)
    e_time = laplacian_e_time(field, sigma_a, temporal_spacing)
    return e_space, e_time
