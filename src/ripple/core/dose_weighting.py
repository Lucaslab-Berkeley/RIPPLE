"""Choice of per-frame dose weights for a movie sum.

A sum is dose weighted in one of three ways, chosen by a single string:

* ``"gg"``: Grant & Grigorieff (2015) critical-exposure weights, normalised at each
  frequency by ``sqrt(sum_f w_f^2)``. This is the code path RIPPLE has always used,
  and it is unchanged.
* ``"default"``: RELION's per-frame model ``w_f(k) = exp(B_f k^2 / 4 + C_f)``, with
  B_f and C_f taken from the default curve in cumulative exposure that ships with
  ``torch_fourier_filter.frame_weights``. The weights are normalised in the same
  way as G&G's.
* a path to a ``FrameWeightTable`` JSON file, for example the output of RIPPLE's
  ``estimate_frame_weights`` program. The table is indexed by exposure, so it is
  resampled onto the movie's frames.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from types import ModuleType

    from torch_fourier_filter.frame_weights import FrameWeightTable

DOSE_WEIGHTING_GG = "gg"
DOSE_WEIGHTING_DEFAULT = "default"
DOSE_WEIGHTING_MODES = (DOSE_WEIGHTING_GG, DOSE_WEIGHTING_DEFAULT)


def frame_weights_module() -> ModuleType:
    """``torch_fourier_filter.frame_weights``, imported on first use.

    The G&G path does not need it, so RIPPLE still runs with a torch-fourier-filter
    that predates the module.

    Returns
    -------
    ModuleType
        The module.

    Raises
    ------
    ImportError
        If the installed torch-fourier-filter has no ``frame_weights`` module.
    """
    # pylint: disable-next=fixme
    # TODO: pin the torch-fourier-filter release that contains teamtomo/teamtomo#203
    # (the `frame_weights` module) in pyproject.toml once it is published, and import
    # it at module level.
    try:
        # pylint: disable=import-outside-toplevel
        from torch_fourier_filter import frame_weights
    except ImportError as err:
        raise ImportError(
            "per-frame B-factor weights need torch-fourier-filter with the "
            "frame_weights module (teamtomo/teamtomo#203); only dose_weighting 'gg' "
            "works with the installed version"
        ) from err
    return frame_weights  # type: ignore[no-any-return, unused-ignore]


def validate_dose_weighting(value: str) -> str:
    """Check a ``dose_weighting`` value: ``"gg"``, ``"default"`` or a table path.

    Parameters
    ----------
    value : str
        The configured value.

    Returns
    -------
    str
        ``value``, lower-cased if it names a mode.

    Raises
    ------
    ValueError
        If `value` is neither a mode nor a path to an existing ``.json`` file.
    """
    if value.lower() in DOSE_WEIGHTING_MODES:
        return value.lower()
    if not value.lower().endswith(".json"):
        raise ValueError(
            f"dose_weighting must be one of {DOSE_WEIGHTING_MODES} or a path to a "
            f"frame weight table (.json), got {value!r}"
        )
    if not os.path.isfile(value):
        raise ValueError(f"dose_weighting table {value!r} does not exist")
    return value


def movie_frame_weight_table(
    dose_weighting: str,
    n_frames: int,
    pre_exposure: float,
    fluence_per_frame: float,
    voltage: float,
) -> FrameWeightTable | None:
    """The frame weight table for a movie, or None for the G&G weights.

    Parameters
    ----------
    dose_weighting : str
        ``"gg"``, ``"default"`` or a path to a ``FrameWeightTable`` JSON file.
    n_frames : int
        Number of frames in the movie.
    pre_exposure : float
        Exposure before the first frame, in e-/A^2.
    fluence_per_frame : float
        Exposure per frame, in e-/A^2.
    voltage : float
        Acceleration voltage in kV. Only recorded in the table: the default curve
        is in physical exposure and fitted both 200 and 300 kV data.

    Returns
    -------
    FrameWeightTable | None
        None for ``"gg"`` (callers keep the existing G&G code path); otherwise a
        table with one row per frame, frame ``i`` covering the exposure
        ``pre_exposure + fluence_per_frame * [i, i + 1]``.
    """
    mode = validate_dose_weighting(dose_weighting)
    if mode == DOSE_WEIGHTING_GG:
        return None
    fwm = frame_weights_module()
    start, end = fwm.frame_exposures(n_frames, pre_exposure, fluence_per_frame)
    if mode == DOSE_WEIGHTING_DEFAULT:
        table = fwm.default_frame_bfactors(start, end, voltage)
    else:
        table = fwm.FrameWeightTable.from_json(dose_weighting).resample(start, end)
    return table  # type: ignore[no-any-return, unused-ignore]


def gg_frame_weights(
    frequency: torch.Tensor,
    n_frames: int,
    pre_exposure: float,
    fluence_per_frame: float,
    voltage: float,
) -> torch.Tensor:
    """RIPPLE's normalised G&G weights at the given frequencies.

    The same weights as ``torch_fourier_filter.dose_weight``'s G&G path, which
    RIPPLE's ``gg`` sums use: ``exp(-D_f / (2 Ne(k)))`` with D_f the exposure at
    the end of frame f (divided by 0.8 at 200 kV and 0.64 at 100 kV), normalised by
    ``sqrt(sum_f w_f(k)^2)``.

    Parameters
    ----------
    frequency : torch.Tensor
        ``(n_freq,)`` spatial frequencies in 1/A.
    n_frames : int
        Number of frames.
    pre_exposure : float
        Exposure before the first frame, in e-/A^2.
    fluence_per_frame : float
        Exposure per frame, in e-/A^2.
    voltage : float
        Acceleration voltage in kV.

    Returns
    -------
    torch.Tensor
        ``(n_frames, n_freq)`` weights, float64.
    """
    # pylint: disable=import-outside-toplevel
    from torch_fourier_filter.dose_weight import critical_exposure

    k = torch.as_tensor(frequency, dtype=torch.float64)
    doses = pre_exposure + fluence_per_frame * (
        torch.arange(n_frames, dtype=torch.float64) + 1
    )
    if abs(voltage - 200) <= 2:
        doses = doses / 0.8
    elif abs(voltage - 100) <= 2:
        doses = doses / 0.64
    ne = critical_exposure(k).clamp(min=1e-10)
    w = torch.exp(-doses[:, None] / (2 * ne[None, :]))
    return w / torch.sqrt((w**2).sum(0, keepdim=True)).clamp(min=1e-300)
