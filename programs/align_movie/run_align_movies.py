"""Run the align frames manager for a directory of movies.

Notes
-----
This script globs all movies in a directory (eer, tif, or mrc) and applies the same
alignment config to each movie, except the output paths which are updated per-loop. Use
this script as an example for how to run RIPPLE on your data.
"""

import os

from ripple.managers import AlignFramesManager

ALIGN_YAML_PATH = "align_movies_example_config.yaml"
MOVIE_DIR = "../../example/movies"


def get_movie_paths(directory: str) -> list[str]:
    """Get the paths of the movies in the directory."""
    movie_paths = []
    for file in os.listdir(directory):
        if file.endswith(".mrc") or file.endswith(".tif") or file.endswith(".eer"):
            movie_paths.append(os.path.join(directory, file))
    return movie_paths


def main():
    """Run alignment for all movies in the directory."""
    movie_paths = get_movie_paths(MOVIE_DIR)
    manager = AlignFramesManager.from_yaml(ALIGN_YAML_PATH)

    for movie_path in movie_paths:
        manager.movie_config.movie_path = movie_path

        # Derive output from base-name of movie. Only update fields that are not None
        stem = os.path.splitext(os.path.basename(movie_path))[0]
        out_cfg = manager.output_config

        def replace_path(path: str, stem: str = stem) -> str:
            """Replace the path with a new path that includes the stem."""
            return f"{stem}_{path}" if path is not None else None

        # fmt: off
        out_cfg.dw_sum_output_path                  = replace_path(out_cfg.dw_sum_output_path)  # noqa: E501
        out_cfg.deformation_field_output_path       = replace_path(out_cfg.deformation_field_output_path)  # noqa: E501
        out_cfg.motion_corrected_movie_output_path  = replace_path(out_cfg.motion_corrected_movie_output_path)  # noqa: E501
        out_cfg.rendered_movie_output_path          = replace_path(out_cfg.rendered_movie_output_path)  # noqa: E501
        out_cfg.non_dw_sum_output_path              = replace_path(out_cfg.non_dw_sum_output_path)  # noqa: E501
        out_cfg.loss_trajectories_output_path       = replace_path(out_cfg.loss_trajectories_output_path)  # noqa: E501
        # fmt: on

        # 1. Load and prepare (gain/dark correct, mean-zero)
        prepared = manager.prepare_movie()

        # 2. Estimate motion: XC pre-pass seeds global shifts, then gradient
        #    optimizer refines at the configured deformation field resolution
        deformation_field, trajectory = manager.estimate_motion(prepared)

        # 3. Apply deformation field and write all configured outputs
        manager.correct_and_save(prepared, deformation_field, trajectory)


if __name__ == "__main__":
    main()
