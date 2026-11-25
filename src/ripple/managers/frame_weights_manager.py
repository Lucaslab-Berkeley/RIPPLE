"""Manager estimating per-frame B-factors and the template envelope from particles."""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mrcfile
import numpy as np
import pandas as pd
import torch
from pydantic import ConfigDict, Field
from teamtomo_basemodel import BaseModelTeamTomo

from ripple.config import ComputationalConfig, MovieConfig
from ripple.config.frame_weights_config import (
    FrameWeightsEstimation,
    FrameWeightsMovie,
    FrameWeightsOutput,
    ParticleSelection,
    TemplateEnvelopeSource,
)
from ripple.core import frame_weights as fw
from ripple.core.dose_weighting import frame_weights_module, gg_frame_weights
from ripple.core.generate_image import fourier_crop_movie
from ripple.core.particle_pieces import (
    MoviePieces,
    compute_movie_pieces,
    keep_grad_mode,
    particle_positions,
    radial_profile,
    whiten_frames,
)
from ripple.utils.data_io import load_tensor_from_path, write_mrc_from_tensor

from .resum_manager import load_deformation_field, prepare_and_correct_movie

if TYPE_CHECKING:
    from torch_fourier_filter.frame_weights import FrameWeightTable, TemplateEnvelope

DEFAULT_PREPROCESSING_FILTERS: dict[str, Any] = {
    "whitening_filter": {"enabled": True, "do_power_spectrum": True, "max_freq": 1.0}
}


def normalised_table_weights(
    table: FrameWeightTable, frequency: np.ndarray
) -> np.ndarray:
    """A table's weights at `frequency` (1/A), normalised by sqrt(sum_f w_f^2)."""
    la = table.log_amplitudes(torch.as_tensor(frequency, dtype=torch.float64)).numpy()
    la = la - la.max(0, keepdims=True)
    w = np.exp(la)
    return np.asarray(w / np.sqrt((w**2).sum(0, keepdims=True)))


def gg_model_fit(
    gg: np.ndarray, mean: np.ndarray, se: np.ndarray, k2: np.ndarray, ref: int
) -> fw.BFactorFit:
    """G&G's weights fitted with the same model, for comparison.

    The data follow G&G exactly, at the measured per-bin amplitude and with the
    measured errors.
    """
    scaled = gg * (mean.mean(0) / gg.mean(0))[None, :]
    return fw.fit_bfactors(scaled, se, k2, ref)


class EstimateFrameWeightsManager(BaseModelTeamTomo):
    """Estimate every frame's B and C, and the template envelope's correction.

    From particles that a search found (a Leopard-EM particle stack CSV, such as
    Panther-EM's ``export`` or a Leopard-EM ``refine_template`` result) on movies
    aligned by RIPPLE:

    1. each movie is prepared and warped with its saved deformation field (no
       re-alignment), or read as an aligned stack, and Fourier-cropped to the output
       pixel size;
    2. the frames are whitened with Leopard-EM's whitening for their plain sum, and
       each particle's template correlation is split by frame, position and radial
       bin, with an analytic null (:mod:`ripple.core.particle_pieces`);
    3. each frame's signal is read at a position chosen on the other half of the
       frames, and RELION's model ``A_b exp(B_f k^2 / 4 + C_f)`` is fitted by
       weighted least squares, every frame free, with bootstrap errors
       (:mod:`ripple.core.frame_weights`);
    4. the template envelope matched to the fitted weights is corrected by a smooth
       5-knot curve that maximises the particles' mean z on the re-weighted sum.

    Outputs (``output.output_dir``): the frame weight table (``dose_weighting`` of
    the align and resum programs), the template envelope (for ttsim3d), a RELION-style
    plot of B and scale per frame, and fit diagnostics.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True)
    computational_config: ComputationalConfig = ComputationalConfig(
        keep_movie_on_cpu=True
    )
    movie_config: MovieConfig
    skip_movie_preparation: bool = False
    grid_type: Literal["catmull_rom", "bspline"] = "catmull_rom"
    movies: list[FrameWeightsMovie] = Field(min_length=1)
    particles_path: str
    template_path: str
    template_envelope: TemplateEnvelopeSource = TemplateEnvelopeSource()
    particle_overrides: dict[str, Any] = Field(default_factory=dict)
    preprocessing_filters: dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_PREPROCESSING_FILTERS)
    )
    selection: ParticleSelection = ParticleSelection()
    estimation: FrameWeightsEstimation = FrameWeightsEstimation()
    output: FrameWeightsOutput

    # ------------------------------------------------------------------ inputs
    def load_particles(self) -> pd.DataFrame:
        """The particle stack, with overrides applied and the selection made."""
        df = pd.read_csv(self.particles_path, index_col=0)
        overrides = dict(self.particle_overrides)
        for column, value in overrides.items():
            df[column] = value
        sel = self.selection
        if sel.score_column is not None:
            if sel.score_column not in df.columns:
                raise ValueError(f"no column {sel.score_column!r} in the particles")
            if sel.min_score is not None:
                df = df[df[sel.score_column] >= sel.min_score]
            df = df.sort_values(sel.score_column, ascending=False, kind="stable")
        if sel.max_particles is not None:
            df = df.iloc[: sel.max_particles]
        return df.reset_index(drop=True)

    @property
    def output_pixel_size(self) -> float:
        """Pixel size of the frames the particles are on."""
        return self.movie_config.output_pixel_size

    def aligned_frames(self, movie: FrameWeightsMovie) -> torch.Tensor:
        """The movie's aligned frames at the output pixel size, float32, on the CPU."""
        comp = self.computational_config
        if movie.aligned_movie_path is not None:
            frames = load_tensor_from_path(movie.aligned_movie_path, expected_ndim=3)
            frames = frames.to(torch.float32)
        else:
            config = self.movie_config.model_copy(deep=True)
            config.movie_path = movie.movie_path
            assert movie.deformation_field_path is not None
            field = load_deformation_field(movie.deformation_field_path, self.grid_type)
            frames = prepare_and_correct_movie(
                comp, config, field, self.skip_movie_preparation
            )
        factor = self.movie_config.super_resolution_factor
        if factor > 1:
            out = []
            for f0 in range(0, frames.shape[0], 4):
                chunk, _ = fourier_crop_movie(
                    frames[f0 : f0 + 4].to(comp.gpu_device),
                    self.movie_config.pixel_size,
                    factor,
                )
                out.append(chunk.cpu())
            frames = torch.cat(out)
        return frames.cpu().contiguous()

    def leopard_kwargs(  # pylint: disable=too-many-locals
        self, df: pd.DataFrame, micrograph: str, work_csv: Path, template_size: int
    ) -> dict[str, Any]:
        """Leopard-EM's backend inputs for one micrograph's particles.

        The per-particle inputs (pose, defocus, CTF settings, magnification) come
        from Leopard-EM's particle stack for every particle. The image-side inputs
        (the template-side whitening filter, which is the same for every particle of
        a micrograph, and the whitened particle boxes, which are only used to check
        the pieces) come from Leopard-EM for the first few particles only, as
        Leopard-EM computes the micrograph's filter once per particle.
        """
        # pylint: disable=import-outside-toplevel
        with keep_grad_mode():
            from leopard_em.pydantic_models.config import PreprocessingFilters
            from leopard_em.pydantic_models.data_structures import ParticleStackCSV
            from leopard_em.utils.backend_setup import (
                setup_particle_backend_kwargs,
                setup_static_particle_kwargs,
            )
            from leopard_em.utils.data_io import load_template_tensor

        dev = self.computational_config.gpu_device
        d = df.copy()
        d["micrograph_path"] = micrograph
        d["template_path"] = self.template_path
        box = template_size + 2 * self.estimation.position_window
        n_image = min(len(d), max(1, self.estimation.n_verify))

        def stack_of(rows: pd.DataFrame, path: Path) -> Any:
            rows.to_csv(path)
            return ParticleStackCSV(
                df_path=str(path),
                extracted_box_size=(box, box),
                original_template_size=(template_size, template_size),
            )

        stack = stack_of(d, work_csv)
        image_stack = stack_of(
            d.iloc[:n_image], work_csv.with_name(f"{work_csv.stem}_first.csv")
        )
        template = load_template_tensor(template_volume_path=self.template_path).to(dev)
        with mrcfile.open(micrograph, permissive=True) as mrc:
            shape = mrc.data.shape[-2:]
        zeros = torch.zeros(1, *shape, device=dev)
        offsets = {
            "euler_angle_offsets": torch.zeros(1, 3, device=dev),
            "defocus_offsets": torch.zeros(1, device=dev),
            "pixel_size_offsets": torch.zeros(1, device=dev),
        }
        with torch.no_grad():
            kwargs = setup_static_particle_kwargs(
                particle_stack=stack,
                template=template,
                euler_angles=stack.get_euler_angles(True).to(dev),
                device_list=[dev],
                mean_stack=zeros,
                std_stack=zeros + 1,
                particle_indices=[pd.RangeIndex(len(d))],
                **offsets,
            )
            first = setup_particle_backend_kwargs(
                particle_stack=image_stack,
                template=template,
                preprocessing_filters=PreprocessingFilters(
                    **self.preprocessing_filters
                ),
                euler_angles=image_stack.get_euler_angles(True).to(dev),
                apply_global_filtering=True,
                device_list=[dev],
                mean_stack=zeros,
                std_stack=zeros + 1,
                particle_indices=[pd.RangeIndex(n_image)],
                **offsets,
            )
        filt = first["projective_filters"][0]
        kwargs["projective_filters"] = filt[None].expand(len(d), *filt.shape)
        kwargs["template_dft"] = first["template_dft"]
        kwargs["particle_stack_dft"] = first["particle_stack_dft"]
        return dict(kwargs)

    # --------------------------------------------------------------- the pieces
    def compute_pieces(  # pylint: disable=too-many-locals
        self, df: pd.DataFrame, log: dict[str, Any]
    ) -> tuple[list[MoviePieces], list[pd.DataFrame], np.ndarray]:
        """Correlation pieces of every movie's particles (see the class docstring)."""
        # pylint: disable=import-outside-toplevel
        with keep_grad_mode():
            from leopard_em.utils.data_io import load_template_tensor

        est = self.estimation
        out_dir = Path(self.output.output_dir)
        (out_dir / "work").mkdir(parents=True, exist_ok=True)
        template_size = int(
            load_template_tensor(template_volume_path=self.template_path).shape[-1]
        )
        edges = fw.radial_bin_edges(est.n_bins)
        pieces, tables = [], []
        claimed = np.zeros(len(df), dtype=bool)
        for movie in self.movies:
            mask = movie.matches(df["micrograph_path"].to_numpy()) & ~claimed
            claimed |= mask
            entry: dict[str, Any] = {
                "movie": movie.model_dump(),
                "particles": int(mask.sum()),
            }
            log["movies"].append(entry)
            if not mask.any():
                continue
            t0 = time.time()
            dfm = df[mask].reset_index(drop=True)
            if "pixel_size" in dfm.columns:
                ratio = float(dfm["pixel_size"].median()) / self.output_pixel_size
                entry["particle_pixel_size_ratio"] = ratio
                if abs(ratio - 1) > 0.005:
                    warnings.warn(
                        f"{movie.stem}: the particles' pixel_size is {ratio:.4f} x the "
                        "frames' (movie_config.pixel_size * super_resolution_factor)",
                        stacklevel=2,
                    )
            frames = self.aligned_frames(movie)
            plain_sum = frames.sum(0)
            sum_path = out_dir / "work" / f"{movie.stem}_frames_sum.mrc"
            write_mrc_from_tensor(
                plain_sum,
                str(sum_path),
                overwrite=True,
                pixel_size=self.output_pixel_size,
            )
            entry["sum_vs_particle_micrograph_corr"] = _micrograph_corr(
                plain_sum, dfm["micrograph_path"].iloc[0]
            )
            kw = self.leopard_kwargs(
                dfm,
                str(sum_path),
                out_dir / "work" / f"{movie.stem}_particles.csv",
                template_size,
            )
            whitening = radial_profile(kw["projective_filters"][0])
            spectra = whiten_frames(
                frames, whitening, edges, self.computational_config.gpu_device
            )
            mp = compute_movie_pieces(
                frames,
                kw,
                particle_positions(dfm),
                template_size,
                edges,
                window=est.position_window,
                device=self.computational_config.gpu_device,
                n_random=est.n_random if est.null_model == "random" else 0,
                n_verify=est.n_verify,
                seed=est.seed,
                noise_window=est.noise_window,
                null_mean=est.null_mean,
                n_envelope=est.n_envelope_orientations,
            )
            mp.frame_cross_spectra = spectra
            del frames, kw
            torch.cuda.empty_cache()
            entry.update(
                {
                    "frames": int(spectra.shape[-1]),
                    "kept": len(mp.particle_rows),
                    "dropped_at_edge": len(mp.dropped_rows),
                    "checks": mp.checks,
                    "seconds": time.time() - t0,
                }
            )
            pieces.append(mp)
            tables.append(dfm.iloc[mp.particle_rows].reset_index(drop=True))
        log["particles_without_movie"] = int((~claimed).sum())
        if not pieces:
            raise ValueError("no particle matched any movie")
        return pieces, tables, edges

    # --------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:  # pylint: disable=too-many-locals,too-many-statements
        """Estimate, and write every output. Returns the diagnostics."""
        self.output.check()
        t_start = time.time()
        est = self.estimation
        df = self.load_particles()
        log: dict[str, Any] = {"movies": [], "particles_selected": len(df)}
        pieces, tables, edges = self.compute_pieces(df, log)
        particles = pd.concat(tables, ignore_index=True)
        movie_index = np.concatenate([np.full(len(t), m) for m, t in enumerate(tables)])
        c = torch.cat([p.c for p in pieces])
        cov = torch.cat([p.noise_cov for p in pieces])
        # the search's null for the z objective (the envelope fit): its mean, and
        # its variance as the search's per-pixel variance
        null_mean = (
            torch.cat([p.null_mean for p in pieces])
            if all(p.null_mean is not None for p in pieces)
            else None
        )
        cov_z = (
            torch.cat([p.noise_cov_z for p in pieces])
            if all(p.noise_cov_z is not None for p in pieces)
            else cov
        )
        nf = c.shape[1]
        if any(p.c.shape[1] != nf for p in pieces):
            raise ValueError("every movie must have the same number of frames")

        fwm = frame_weights_module()
        mc = self.movie_config
        start, end = fwm.frame_exposures(nf, mc.pre_exposure, mc.fluence_per_frame)
        kb = fw.bin_centres(edges, self.output_pixel_size)
        default_table = fwm.default_frame_bfactors(start, end, mc.voltage)
        gg = gg_frame_weights(
            torch.as_tensor(kb), nf, mc.pre_exposure, mc.fluence_per_frame, mc.voltage
        ).numpy()
        ref_w = (
            gg
            if est.reference_weights == "gg"
            else normalised_table_weights(default_table, kb)
        )

        # 1. each frame's signal, in null-sd units
        if est.null_model == "random":
            nulls = torch.cat(
                [p.null_random for p in pieces if p.null_random is not None]
            )
            sig, var, sd = fw.frame_signal_sampled_null(c, nulls, ref_w)
        else:
            sig, var, sd = fw.frame_signal(c, cov, ref_w)
        summary = fw.frame_signal_summary(
            sig.numpy(), var.numpy(), kb, est.min_snr, 1.0 / est.min_resolution
        )
        use = summary.bins_used
        if len(use) < 2:
            raise ValueError(
                f"only {len(use)} bin(s) have summed signal >= {est.min_snr} SE: too "
                "few particles, or the particles do not match the template"
            )
        k2 = kb[use] ** 2

        # 2. B and C of every frame
        prior = self._prior(start, end)
        fit = fw.fit_bfactors(
            summary.mean[:, use],
            summary.standard_error[:, use],
            k2,
            est.reference_frame,
            prior,
        )
        if est.bootstrap_group is None:
            groups = [np.array([i]) for i in range(len(particles))]
        else:
            groups = fw.groups_from_labels(particles[est.bootstrap_group].to_numpy())
        boot_b, boot_c = fw.bootstrap_bfactors(
            summary.weighted_signal,
            use,
            k2,
            groups,
            est.n_bootstrap,
            fit.reference_frame,
            prior,
            est.seed,
        )
        b_sd = boot_b.std(0) if len(boot_b) else np.full(nf, np.nan)
        scale_sd = np.exp(boot_c).std(0) if len(boot_c) else np.full(nf, np.nan)
        gg_fit = gg_model_fit(
            gg[:, use],
            summary.mean[:, use],
            summary.standard_error[:, use],
            k2,
            fit.reference_frame,
        )
        micrographs = sorted(set(particles["micrograph_path"].astype(str)))
        table = fwm.FrameWeightTable(
            exposure_start=start,
            exposure_end=end,
            bfactor=fit.bfactor,
            log_scale=fit.log_scale,
            voltage=mc.voltage,
            source="estimated",
            provenance={
                "program": "RIPPLE estimate_frame_weights",
                "particles": len(particles),
                "micrographs": len(micrographs),
                "bootstrap_units": len(groups),
                "bfactor_sd": b_sd.tolist(),
                "log_scale_sd": (
                    boot_c.std(0) if len(boot_c) else b_sd * np.nan
                ).tolist(),
                "chi2_dof": fit.chi2_dof,
                "bins_used": [int(use[0]), int(use[-1])],
                "resolution_range_A": [float(1 / kb[use[0]]), float(1 / kb[use[-1]])],
                "reference_weights": est.reference_weights,
                "null_model": est.null_model,
                "noise_window": est.noise_window,
                "null_mean": est.null_mean,
                "prior": est.prior,
            },
        )

        # 3. the template envelope: matched to the table, times a fitted correction
        envelope, env_info = self._envelope(
            table, c, cov_z, sig, kb, movie_index, pieces, null_mean
        )

        # 4. outputs
        table.to_json(self.output.path(self.output.frame_weights_name))
        envelope.to_json(self.output.path(self.output.template_envelope_name))
        diagnostics = {
            "frames": nf,
            "pixel_size": self.output_pixel_size,
            "exposure_per_frame": mc.fluence_per_frame,
            "voltage": mc.voltage,
            "n_particles": len(particles),
            "particles_per_micrograph": {
                m: int((particles["micrograph_path"].astype(str) == m).sum())
                for m in micrographs
            },
            "bins_used": use.tolist(),
            "resolution_range_A": [float(1 / kb[use[0]]), float(1 / kb[use[-1]])],
            "chi2_dof": fit.chi2_dof,
            "reference_frame": fit.reference_frame,
            "bfactor": fit.bfactor.tolist(),
            "bfactor_sd": b_sd.tolist(),
            "scale": np.exp(fit.log_scale).tolist(),
            "scale_sd": scale_sd.tolist(),
            "gg_bfactor": gg_fit.bfactor.tolist(),
            "gg_scale": np.exp(gg_fit.log_scale).tolist(),
            "default_bfactor": (
                default_table.bfactor - default_table.bfactor.max()
            ).tolist(),
            "default_scale": torch.exp(
                default_table.log_scale - default_table.log_scale.max()
            ).tolist(),
            "noise_relative": summary.noise_relative[:, use].mean(1).tolist(),
            "null_sd_median": float(sd.median()),
            "bin_snr": summary.snr.tolist(),
            "envelope": env_info,
            "log": log,
            "seconds": time.time() - t_start,
        }
        with open(
            self.output.path(self.output.diagnostics_name), "w", encoding="utf-8"
        ) as f:
            json.dump(diagnostics, f, indent=1, default=_json_default)
        np.savez(
            self.output.path("frame_signal.npz"),
            mean=summary.mean,
            standard_error=summary.standard_error,
            frequency=kb,
            bins_used=use,
            amplitude=fit.amplitude,
            bfactor=fit.bfactor,
            log_scale=fit.log_scale,
            bootstrap_bfactor=boot_b,
            bootstrap_log_scale=boot_c,
            gg_weights=gg,
            gg_bfactor=gg_fit.bfactor,
            gg_log_scale=gg_fit.log_scale,
            default_bfactor=default_table.bfactor.numpy(),
            default_log_scale=default_table.log_scale.numpy(),
            envelope_frequency=np.asarray(env_info["frequency"]),
            envelope_relative=np.asarray(env_info["relative_to_template"]),
        )
        if self.output.save_pieces:
            torch.save(
                {
                    "c": c,
                    "noise_cov": cov,
                    "noise_cov_z": cov_z if cov_z is not cov else None,
                    "null_mean": null_mean,
                    "movie_index": movie_index,
                    "frame_cross_spectra": [p.frame_cross_spectra for p in pieces],
                    "edges": edges,
                    "particles": particles.to_dict("list"),
                },
                self.output.path("pieces.pt"),
            )
        try:
            # pylint: disable=import-outside-toplevel
            from ripple.utils.plot_frame_weights import plot_frame_bfactors

            plot_frame_bfactors(
                self.output.path("frame_signal.npz"),
                self.output.path(self.output.plot_name),
                title=f"{len(particles)} particles, {len(micrographs)} micrograph(s)",
            )
        except ImportError as err:  # matplotlib is optional
            diagnostics["plot"] = f"not written: {err}"
        return diagnostics

    def _prior(self, start: torch.Tensor, end: torch.Tensor) -> fw.BFactorPrior | None:
        est = self.estimation
        if est.prior is None:
            return None
        fwm = frame_weights_module()
        if est.prior == "default":
            table = fwm.default_frame_bfactors(start, end, self.movie_config.voltage)
        else:
            table = fwm.FrameWeightTable.from_json(est.prior).resample(start, end)
        return fw.BFactorPrior(
            bfactor=table.bfactor.numpy(),
            log_scale=table.log_scale.numpy(),
            sd_bfactor=est.prior_sd_bfactor,
            sd_log_scale=est.prior_sd_log_scale,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def _envelope(
        self,
        table: FrameWeightTable,
        c: torch.Tensor,
        cov: torch.Tensor,
        sig: torch.Tensor,
        kb: np.ndarray,
        movie_index: np.ndarray,
        pieces: list[MoviePieces],
        null_mean: torch.Tensor | None = None,
    ) -> tuple[TemplateEnvelope, dict[str, Any]]:
        """The matched envelope and its fitted correction (see the class docstring)."""
        est = self.estimation
        del sig
        weights = normalised_table_weights(table, kb)
        # whitening of the re-weighted sum relative to the frames' (plain sum's)
        u2 = np.zeros((len(movie_index), len(kb)))
        ones = np.ones(weights.shape[0])
        for m, p in enumerate(pieces):
            s = p.frame_cross_spectra.numpy()
            p_ref = np.einsum("f,bfg,g->b", ones, s, ones)
            p_w = np.einsum("fb,bfg,gb->b", weights, s, weights)
            u2[movie_index == m] = np.where(
                p_w > 0, p_ref / np.maximum(p_w, 1e-300), 1.0
            )
        coeffs = fw.sum_coefficients(c, cov, weights, u2, null_mean=null_mean)
        fwm = frame_weights_module()
        matched = fwm.matched_envelope(torch.as_tensor(kb), table).numpy()
        current = self.template_envelope.curve(kb)
        base = matched / np.where(current > 0, current, 1.0)
        knots = fw.knot_frequencies(est.envelope_knots, *est.envelope_knot_resolutions)
        info: dict[str, Any] = {
            "frequency": kb.tolist(),
            "matched": matched.tolist(),
            "template_current": current.tolist(),
            "knot_frequency": knots.tolist(),
            "mean_z_template_as_is": float(coeffs.z(np.ones_like(kb)).mean()),
            "mean_z_matched": float(coeffs.z(base).mean()),
        }
        correction = None
        log_values = np.zeros(len(knots))
        if est.fit_envelope_correction:
            log_values, fit_info = fw.fit_envelope_correction(coeffs, base, kb, knots)
            info.update(fit_info)
            correction = fwm.EnvelopeCorrection(
                knot_frequency=torch.as_tensor(knots),
                log_values=torch.as_tensor(log_values),
            )
        h = base * fw.spline_correction(kb, knots, log_values)
        info["log_values"] = log_values.tolist()
        info["relative_to_template"] = h.tolist()
        info["mean_z_matched_corrected"] = float(coeffs.z(h).mean())
        envelope = fwm.TemplateEnvelope(
            frame_weights=table,
            correction=correction,
            provenance={
                "program": "RIPPLE estimate_frame_weights",
                "template": os.path.abspath(self.template_path),
                "template_envelope_before": self.template_envelope.model_dump(),
                "particles": int(c.shape[0]),
                "mean_z": {
                    k: info[k]
                    for k in (
                        "mean_z_template_as_is",
                        "mean_z_matched",
                        "mean_z_matched_corrected",
                    )
                },
            },
        )
        return envelope, info


def _micrograph_corr(plain_sum: torch.Tensor, micrograph_path: str) -> float | None:
    """Correlation of the frames' sum with the particles' micrograph.

    A sanity check that the frames and the particle coordinates share one geometry.
    """
    if not os.path.isfile(str(micrograph_path)):
        return None
    with mrcfile.open(str(micrograph_path), permissive=True) as mrc:
        ref = np.asarray(mrc.data, np.float32)
    s = plain_sum.numpy()
    if ref.shape != s.shape:
        return None
    return float(np.corrcoef(s[::4, ::4].ravel(), ref[::4, ::4].ravel())[0, 1])


def _json_default(x: Any) -> Any:
    if isinstance(x, np.ndarray | torch.Tensor):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return str(x)
