"""Configuration of the per-frame B-factor estimator (estimate_frame_weights)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
from pydantic import Field, PositiveInt, model_validator
from teamtomo_basemodel import BaseModelTeamTomo

from ripple.core.dose_weighting import frame_weights_module


class FrameWeightsMovie(BaseModelTeamTomo):
    """One movie whose particles enter the fit.

    Give either the movie and the deformation field its alignment run saved (the
    frames are prepared and warped as in that run, with no re-alignment), or an
    already aligned frame stack.

    Attributes
    ----------
    movie_path : str | None
        The raw movie. Its preparation (gain, dark, mask) comes from the manager's
        ``movie_config``.
    deformation_field_path : str | None
        The field the alignment run saved (.csv, .h5 or .hdf5).
    aligned_movie_path : str | None
        Instead of the two above: aligned frames (MRC), for example an alignment
        run's ``motion_corrected_movie_output_path``, at ``movie_config.pixel_size``.
    micrograph_path : str | None
        The particles' ``micrograph_path`` for this movie. If None, the particles
        whose micrograph file name starts with the movie's stem followed by ``_``
        (the naming of ``run_align_movies.py``), or equals the stem, are used.
    """

    movie_path: str | None = None
    deformation_field_path: str | None = None
    aligned_movie_path: str | None = None
    micrograph_path: str | None = None

    @model_validator(mode="after")  # type: ignore
    def _one_source(self) -> Self:
        """Require (movie and field) or aligned frames, not both."""
        raw = self.movie_path is not None and self.deformation_field_path is not None
        aligned = self.aligned_movie_path is not None
        if raw == aligned:
            raise ValueError(
                "give either movie_path with deformation_field_path, or "
                "aligned_movie_path"
            )
        return self

    @property
    def stem(self) -> str:
        """The movie's file stem (used to match particles and name outputs)."""
        path = (
            self.movie_path if self.movie_path is not None else self.aligned_movie_path
        )
        assert path is not None
        return os.path.splitext(os.path.basename(path))[0]

    def matches(self, micrograph_paths: np.ndarray) -> np.ndarray:
        """Boolean mask of the particles (by ``micrograph_path``) of this movie."""
        paths = np.asarray(micrograph_paths).astype(str)
        if self.micrograph_path is not None:
            target = os.path.abspath(self.micrograph_path)
            return np.array([os.path.abspath(p) == target for p in paths])
        names = np.array([os.path.basename(p) for p in paths])
        stems = np.array([os.path.splitext(n)[0] for n in names])
        stem = self.stem
        return np.array(
            [
                s == stem or n.startswith(f"{stem}_")
                for s, n in zip(stems, names, strict=True)
            ]
        )


class TemplateEnvelopeSource(BaseModelTeamTomo):
    """The radial envelope the search template already carries.

    The estimator's correlations use this template, so the envelope it writes (matched
    to the estimated weights, times the fitted correction) is found relative to it.

    Attributes
    ----------
    kind : Literal["none", "ttsim3d", "file"]
        ``none``: simulated without a dose filter. ``ttsim3d``: ttsim3d's cumulative
        G&G dose filter over ``[dose_start, dose_end]``, then ``modify_signal``.
        ``file``: a ``TemplateEnvelope`` JSON (for example the default envelope, or
        an earlier estimate) that the template was filtered with.
    dose_start, dose_end : float
        ttsim3d's dose range, e-/A^2 (as given to ttsim3d: at 200 kV the study's
        templates used 50 / 0.8).
    modify_signal : Literal["None", "sqrt", "rel_diff"]
        ttsim3d's ``dose_filter_modify_signal``.
    critical_bfactor : float
        ttsim3d's ``dose_filter_critical_bfactor`` (-1: G&G's critical exposure).
    path : str | None
        The envelope file for ``kind: file``.
    """

    kind: Literal["none", "ttsim3d", "file"] = "none"
    dose_start: float = 0.0
    dose_end: float = 50.0
    modify_signal: Literal["None", "sqrt", "rel_diff"] = "rel_diff"
    critical_bfactor: float = -1.0
    path: str | None = None

    @model_validator(mode="after")  # type: ignore
    def _check(self) -> Self:
        """A file envelope needs its path."""
        if self.kind == "file" and (self.path is None or not os.path.isfile(self.path)):
            raise ValueError("template_envelope.kind 'file' needs an existing path")
        return self

    def curve(self, frequency: np.ndarray) -> np.ndarray:
        """The envelope at `frequency` (1/A), 1 at k = 0."""
        k = torch.as_tensor(np.asarray(frequency, float), dtype=torch.float64)
        if self.kind == "none":
            return np.ones(k.shape)
        if self.kind == "file":
            assert self.path is not None
            envelope = frame_weights_module().TemplateEnvelope.from_json(self.path)
            return np.asarray(envelope(k).numpy())
        # pylint: disable=import-outside-toplevel
        from torch_fourier_filter.dose_weight import critical_exposure

        if self.critical_bfactor == -1:
            ne = critical_exposure(k)
        else:
            ne = 4 / (self.critical_bfactor * k.clamp(min=1e-10) ** 2)
        ne = ne.clamp(min=1e-10)
        # ttsim3d's 2 Ne (exp(-D0 / 2Ne) - exp(-D1 / 2Ne)) / D1, with expm1 so that it
        # stays accurate where Ne is huge (k -> 0)
        cum = (
            2
            * ne
            * torch.exp(-0.5 * self.dose_start / ne)
            * -torch.expm1(-0.5 * (self.dose_end - self.dose_start) / ne)
            / self.dose_end
        )
        if self.modify_signal == "sqrt":
            cum = torch.sqrt(cum)
        elif self.modify_signal == "rel_diff":
            cum = 2 * cum / (1 + cum)
        return np.asarray(cum.numpy())


class ParticleSelection(BaseModelTeamTomo):
    """Which particles of the stack enter the fit.

    Attributes
    ----------
    score_column : str | None
        Column to threshold, e.g. ``refined_scaled_mip`` or ``scaled_mip``. None
        keeps every particle (a Panther-EM export is already cut at its
        false-positive threshold).
    min_score : float | None
        Keep particles with ``score_column >= min_score``.
    max_particles : int | None
        Keep at most this many (highest score first, or the first rows).
    """

    score_column: str | None = None
    min_score: float | None = None
    max_particles: PositiveInt | None = None


class FrameWeightsEstimation(BaseModelTeamTomo):
    """Settings of the per-frame signal, the B-factor fit and the envelope fit.

    Attributes
    ----------
    n_bins : int
        Radial bins, uniform in sqrt(|k|) out to the spectrum's corner.
    position_window : int
        Correlations are kept within this many pixels of each particle; each half of
        the frames picks the position for the other half.
    reference_weights : Literal["gg", "default"]
        Frame weights that choose the position and define each particle's null sd.
    null_model : Literal["analytic", "random"]
        ``analytic``: from the frames' and the template's power per bin.
        ``random``: ``n_random`` random orientations per particle (slow; for checks).
    n_random : int
        Random orientations per particle for ``null_model: random``.
    min_snr : float
        Bins fitted: a contiguous run from ``min_resolution`` whose summed signal is
        at least this many standard errors.
    min_resolution : float
        Lowest resolution fitted, A.
    reference_frame : int | None
        Frame whose B and C are held at 0 during the fit (the gauge; the output is
        shifted to max B = max C = 0). Default: the frame with the most signal.
    n_bootstrap : int
        Bootstrap resamples for the errors.
    bootstrap_group : str | None
        Particle column whose distinct values are resampled together (e.g.
        ``micrograph_path``, or a filament id). None resamples particles.
    prior : str | None
        Optional shrinkage of every frame's B and C towards a table: ``default`` or
        a frame weight table path. Off (None) by default.
    prior_sd_bfactor, prior_sd_log_scale : float | None
        The prior's sd for B (A^2) and C. A term is used only if its sd is set.
    fit_envelope_correction : bool
        Fit the smooth correction on top of the matched template envelope.
    envelope_knots : int
        Knots of the correction, uniform in sqrt(k).
    envelope_knot_resolutions : tuple[float, float]
        Resolutions (A) of the first and last knot.
    n_verify : int
        Particles per movie checked against Leopard-EM's own correlation.
    seed : int
        Seed for the bootstrap and random orientations.
    """

    n_bins: PositiveInt = 48
    position_window: int = Field(default=3, ge=0)
    reference_weights: Literal["gg", "default"] = "gg"
    null_model: Literal["analytic", "random"] = "analytic"
    n_random: int = Field(default=1024, ge=2)
    min_snr: float = 5.0
    min_resolution: float = 40.0
    reference_frame: int | None = None
    n_bootstrap: int = Field(default=200, ge=0)
    bootstrap_group: str | None = None
    prior: str | None = None
    prior_sd_bfactor: float | None = None
    prior_sd_log_scale: float | None = None
    fit_envelope_correction: bool = True
    envelope_knots: int = Field(default=5, ge=2)
    envelope_knot_resolutions: tuple[float, float] = (50.0, 2.9)
    n_verify: int = Field(default=3, ge=0)
    seed: int = 0


class FrameWeightsOutput(BaseModelTeamTomo):
    """Where the estimator writes.

    Attributes
    ----------
    output_dir : str
        Directory for every output.
    allow_file_overwrite : bool
        Allow replacing existing outputs.
    frame_weights_name, template_envelope_name, plot_name, diagnostics_name : str
        File names of the frame weight table, the template envelope, the RELION-style
        plot and the diagnostics.
    save_pieces : bool
        Also save the per-particle correlation pieces (``pieces.pt``; large) so the
        fit can be repeated with other settings.
    """

    output_dir: str
    allow_file_overwrite: bool = False
    frame_weights_name: str = "frame_weights.json"
    template_envelope_name: str = "template_envelope.json"
    plot_name: str = "frame_bfactors.png"
    diagnostics_name: str = "frame_weights_diagnostics.json"
    save_pieces: bool = False

    def path(self, name: str) -> Path:
        """``output_dir / name``."""
        return Path(self.output_dir) / name

    @property
    def outputs(self) -> list[Path]:
        """The main output files."""
        return [
            self.path(self.frame_weights_name),
            self.path(self.template_envelope_name),
            self.path(self.plot_name),
            self.path(self.diagnostics_name),
        ]

    def check(self) -> None:
        """Create the directory; refuse existing outputs unless overwriting."""
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.access(self.output_dir, os.W_OK):
            raise ValueError(f"cannot write to {self.output_dir}")
        if not self.allow_file_overwrite:
            existing = [str(p) for p in self.outputs if p.exists()]
            if existing:
                raise ValueError(
                    f"outputs exist and allow_file_overwrite is false: {existing}"
                )
