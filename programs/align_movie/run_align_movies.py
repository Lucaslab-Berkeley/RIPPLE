"""Run the align frames manager for a directory of movies."""

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

        # 1. Load and prepare (gain/dark correct, mean-zero)
        prepared = manager.prepare_movie()

        # 2. Estimate motion: XC pre-pass seeds global shifts, then gradient
        #    optimizer refines at the configured deformation field resolution
        deformation_field, trajectory = manager.estimate_motion(prepared)

        # 3. Apply deformation field and write all configured outputs
        manager.correct_and_save(prepared, deformation_field, trajectory)


if __name__ == "__main__":
    main()
