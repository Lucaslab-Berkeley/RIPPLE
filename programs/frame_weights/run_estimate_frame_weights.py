r"""Estimate per-frame B-factors and the template envelope from particles.

Usage::

    python run_estimate_frame_weights.py estimate_frame_weights_example_config.yaml
    python run_estimate_frame_weights.py config.yaml \
        --movie-dir ../../example/movies \
        --deformation-field ../../example/output/def_field.csv

The particles are a Leopard-EM particle stack CSV (Panther-EM ``export``, or a
Leopard-EM ``refine_template`` result) found on the sums of RIPPLE-aligned movies.
Each movie is prepared and warped with the deformation field its alignment run saved
(no re-alignment), every particle's template correlation is split by frame and
spatial frequency, and RELION's per-frame model ``w_f(k) = exp(B_f k^2 / 4 + C_f)`` is
fitted to it, every frame free.

With ``--movie-dir``, the ``movies`` list of the config is replaced by every movie in
the directory, each with the field ``<stem>_<name>`` next to ``--deformation-field``
(the naming of ``run_align_movies.py``); particles are matched to movies by the
movie stem at the start of their ``micrograph_path`` file name.

Outputs (in ``output.output_dir``):

* ``frame_weights.json``: the frame weight table, for ``dose_weighting:`` in the align
  and resum configs (``programs/align_movie/run_resum.py`` re-sums without
  re-aligning);
* ``template_envelope.json``: the template envelope matched to the table, times the
  fitted 5-knot correction (for ttsim3d's matched dose filter, or
  ``ttsim3d-refilter`` of a map simulated without a dose filter);
* ``frame_bfactors.png``: RELION-style plot of B and scale per frame (needs
  matplotlib);
* ``frame_weights_diagnostics.json`` and ``frame_signal.npz``: the fit's inputs and
  diagnostics.
"""

import argparse
import json
import os
from pathlib import Path

from ripple.config.frame_weights_config import FrameWeightsMovie
from ripple.managers import EstimateFrameWeightsManager
from ripple.utils.data_io import MOVIE_EXTENSIONS

CONFIG_PATH = "estimate_frame_weights_example_config.yaml"


def movies_from_directory(
    movie_dir: str, field_template: str
) -> list[FrameWeightsMovie]:
    """Every movie in `movie_dir` that has a saved deformation field.

    Parameters
    ----------
    movie_dir : str
        Directory of movies (eer, mrc/mrcs, tif/tiff; any case).
    field_template : str
        The alignment config's ``deformation_field_output_path``; each movie's field
        is the same path with the file name prefixed by ``<stem>_``.

    Returns
    -------
    list[FrameWeightsMovie]
        One entry per movie with a field; movies without one are reported and skipped.
    """
    template = Path(field_template)
    movies = []
    for name in sorted(os.listdir(movie_dir)):
        if not name.lower().endswith(MOVIE_EXTENSIONS):
            continue
        stem = os.path.splitext(name)[0]
        field = template.with_name(f"{stem}_{template.name}")
        if not field.exists():
            print(f"Skipping {name}: no deformation field at {field}")
            continue
        movies.append(
            FrameWeightsMovie(
                movie_path=os.path.join(movie_dir, name),
                deformation_field_path=str(field),
            )
        )
    return movies


def main() -> None:
    """Run the estimator."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("config_path", nargs="?", default=CONFIG_PATH)
    parser.add_argument(
        "--movie-dir", default=None, help="use every movie in this directory"
    )
    parser.add_argument(
        "--deformation-field",
        default=None,
        help="with --movie-dir: the alignment's deformation_field_output_path",
    )
    args = parser.parse_args()
    manager = EstimateFrameWeightsManager.from_yaml(args.config_path)
    if args.movie_dir is not None:
        if args.deformation_field is None:
            parser.error("--movie-dir needs --deformation-field")
        manager.movies = movies_from_directory(args.movie_dir, args.deformation_field)
    diagnostics = manager.run()
    summary = {
        k: diagnostics[k]
        for k in ("n_particles", "resolution_range_A", "chi2_dof", "reference_frame")
    }
    summary["mean_z"] = {
        k: v for k, v in diagnostics["envelope"].items() if k.startswith("mean_z")
    }
    print(json.dumps(summary, indent=1))
    print(f"Wrote {manager.output.output_dir}")


if __name__ == "__main__":
    main()
