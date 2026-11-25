"""Re-sum aligned movies with new frame weights, without re-aligning them.

Each movie is prepared as in its alignment run, warped with the deformation field that
run saved, and summed with ``output_config.dose_weighting``: ``gg`` (Grant &
Grigorieff, as the alignment run's ``dw_sum``), ``default`` (per-frame B-factors from
the default exposure curve), or a frame weight table from
``programs/frame_weights/run_estimate_frame_weights.py``.

Usage::

    python run_resum.py resum_example_config.yaml                # the configured movie
    python run_resum.py resum_example_config.yaml --movie-dir ../../example/movies

With ``--movie-dir``, every movie in the directory (eer, mrc/mrcs, tif/tiff; any case)
is re-summed. As in ``run_align_movies.py``, the configured paths are templates whose
file names are prefixed with each movie's stem: ``deformation_field_path`` should be
the alignment config's ``deformation_field_output_path``, so ``def_field.csv`` finds
``<stem>_def_field.csv``. Movies whose outputs already exist are skipped unless
``allow_file_overwrite`` is true, and movies with no saved field are skipped.
"""

import argparse
import os
from pathlib import Path

from ripple.managers import ResumManager
from ripple.utils.data_io import MOVIE_EXTENSIONS

RESUM_YAML_PATH = "resum_example_config.yaml"


def get_movie_paths(directory: str) -> list[str]:
    """Get the sorted paths of the movies in the directory (extension any case)."""
    return sorted(
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(MOVIE_EXTENSIONS)
    )


def stem_prefixed(template: str | None, stem: str) -> str | None:
    """Prefix `template`'s basename (not its directory) with `stem`."""
    if template is None:
        return None
    template_path = Path(template)
    return str(template_path.with_name(f"{stem}_{template_path.name}"))


def resum_directory(manager: ResumManager, movie_dir: str) -> None:
    """Re-sum every movie in `movie_dir`, with stem-prefixed field and output paths.

    Parameters
    ----------
    manager : ResumManager
        Manager whose paths are templates (see the module docstring).
    movie_dir : str
        Directory of movies.
    """
    out_cfg = manager.output_config
    templates = {
        "dw_sum_output_path": out_cfg.dw_sum_output_path,
        "non_dw_sum_output_path": out_cfg.non_dw_sum_output_path,
        "motion_corrected_movie_output_path": (
            out_cfg.motion_corrected_movie_output_path
        ),
    }
    field_template = manager.deformation_field_path
    for movie_path in get_movie_paths(movie_dir):
        stem = os.path.splitext(os.path.basename(movie_path))[0]
        field_path = stem_prefixed(field_template, stem)
        assert field_path is not None
        if not os.path.exists(field_path):
            print(f"Skipping {movie_path}: no deformation field at {field_path}")
            continue
        manager.movie_config.movie_path = movie_path
        manager.deformation_field_path = field_path
        for name, template in templates.items():
            setattr(out_cfg, name, stem_prefixed(template, stem))
        existing = out_cfg.existing_output_paths()
        if existing and not out_cfg.allow_file_overwrite:
            print(
                f"Skipping {movie_path}: outputs already exist and "
                f"allow_file_overwrite is false: {', '.join(existing)}"
            )
            continue
        print(f"Re-summing {movie_path} with {out_cfg.dose_weighting} weights")
        manager.run()


def main() -> None:
    """Re-sum the configured movie, or every movie in a directory."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("config_path", nargs="?", default=RESUM_YAML_PATH)
    parser.add_argument(
        "--movie-dir",
        default=None,
        help="re-sum every movie in this directory (paths are stem-prefixed)",
    )
    args = parser.parse_args()
    manager = ResumManager.from_yaml(args.config_path)
    if args.movie_dir is None:
        manager.run()
    else:
        resum_directory(manager, args.movie_dir)


if __name__ == "__main__":
    main()
