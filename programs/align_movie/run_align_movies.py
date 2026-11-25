"""Run the align frames manager for a directory of movies.

Notes
-----
This script globs all movies in a directory (eer, mrc/mrcs, or tif/tiff; any case) and
applies the same alignment config to each movie, except the output paths which are
updated per-loop. If `allow_file_overwrite` is false, movies whose outputs already exist
are skipped. Use this script as an example for how to run RIPPLE on your data.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from ripple.config import OutputConfig
from ripple.managers import AlignFramesManager
from ripple.utils.data_io import MOVIE_EXTENSIONS

ALIGN_YAML_PATH = "align_movies_example_config.yaml"
MOVIE_DIR = "../../example/movies"


def get_movie_paths(directory: str) -> list[str]:
    """Get the sorted paths of the movies in the directory (extension any case)."""
    return sorted(
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(MOVIE_EXTENSIONS)
    )


def _stem_prefixed(template: str | None, stem: str) -> str | None:
    """Prefix `template`'s basename (not its directory) with `stem`."""
    if template is None:
        return None
    template_path = Path(template)
    return str(template_path.with_name(f"{stem}_{template_path.name}"))


@dataclass
class MovieOutputPaths:
    """Per-movie output paths, derived from the template `OutputConfig`."""

    dw_sum_output_path: str | None
    deformation_field_output_path: str | None
    motion_corrected_movie_output_path: str | None
    rendered_movie_output_path: str | None
    non_dw_sum_output_path: str | None
    loss_trajectories_output_path: str | None


def build_movie_outputs(
    movie_paths: list[str], output_templates: OutputConfig
) -> list[MovieOutputPaths]:
    """Build per-movie output paths by prefixing the template paths with each stem.

    Parameters
    ----------
    movie_paths: list[str]
        Input movie paths.
    output_templates: OutputConfig
        The template output configuration whose paths get stem-prefixed.

    Returns
    -------
    list[MovieOutputPaths]
        One entry per movie, in the same order as `movie_paths`.
    """
    outputs = []
    for movie_path in movie_paths:
        stem = os.path.splitext(os.path.basename(movie_path))[0]
        outputs.append(
            MovieOutputPaths(
                dw_sum_output_path=_stem_prefixed(
                    output_templates.dw_sum_output_path, stem
                ),
                deformation_field_output_path=_stem_prefixed(
                    output_templates.deformation_field_output_path, stem
                ),
                motion_corrected_movie_output_path=_stem_prefixed(
                    output_templates.motion_corrected_movie_output_path, stem
                ),
                rendered_movie_output_path=_stem_prefixed(
                    output_templates.rendered_movie_output_path, stem
                ),
                non_dw_sum_output_path=_stem_prefixed(
                    output_templates.non_dw_sum_output_path, stem
                ),
                loss_trajectories_output_path=_stem_prefixed(
                    output_templates.loss_trajectories_output_path, stem
                ),
            )
        )
    return outputs


def main():
    """Run alignment for all movies in the directory."""
    movie_paths = get_movie_paths(MOVIE_DIR)
    manager = AlignFramesManager.from_yaml(ALIGN_YAML_PATH)

    out_cfg = manager.output_config
    movie_outputs = build_movie_outputs(movie_paths, out_cfg)

    for movie_path, outputs in zip(movie_paths, movie_outputs, strict=True):
        manager.movie_config.movie_path = movie_path
        out_cfg.dw_sum_output_path = outputs.dw_sum_output_path
        out_cfg.deformation_field_output_path = outputs.deformation_field_output_path
        out_cfg.motion_corrected_movie_output_path = (
            outputs.motion_corrected_movie_output_path
        )
        out_cfg.rendered_movie_output_path = outputs.rendered_movie_output_path
        out_cfg.non_dw_sum_output_path = outputs.non_dw_sum_output_path
        out_cfg.loss_trajectories_output_path = outputs.loss_trajectories_output_path

        # Skip movies whose outputs already exist, rather than aborting the loop
        existing_outputs = out_cfg.existing_output_paths()
        if existing_outputs and not out_cfg.allow_file_overwrite:
            print(
                f"Skipping {movie_path}: outputs already exist and "
                f"allow_file_overwrite is false: {', '.join(existing_outputs)}"
            )
            continue

        # 1. Load and prepare (gain/dark correct, mean-zero). Checks first that all
        #    outputs can be written.
        prepared = manager.prepare_movie()

        # 2. Estimate motion: XC pre-pass seeds global shifts, then gradient
        #    optimizer refines at the configured deformation field resolution
        deformation_field, trajectory = manager.estimate_motion(prepared)

        # 3. Apply deformation field and write all configured outputs
        manager.correct_and_save(prepared, deformation_field, trajectory)

        # Free this movie before the next one is loaded
        del prepared, deformation_field, trajectory


if __name__ == "__main__":
    main()
