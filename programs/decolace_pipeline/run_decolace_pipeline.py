"""Run the fused DeCo-LACE pipeline for a directory of movies."""

import os

import yaml

from ripple.config import (
    AlignFramesConfig,
    BeamMaskConfig,
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.managers import AlignFramesManager, BeamMaskManager

PIPELINE_YAML_PATH = "decolace_pipeline_example_config.yaml"
MOVIE_DIR = "../../example/movies"


def get_movie_paths(directory: str) -> list[str]:
    """Get the paths of the movies in the directory."""
    movie_paths = []
    for file in os.listdir(directory):
        if file.endswith((".mrc", ".tif", ".eer")):
            movie_paths.append(os.path.join(directory, file))
    return movie_paths


def main() -> None:
    """Run beam mask estimation, motion estimation, and correction for each movie."""
    with open(PIPELINE_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    computational_config = ComputationalConfig(**config["computational_config"])
    movie_config = MovieConfig(**config["movie_config"])
    beam_mask_manager = BeamMaskManager(
        computational_config=computational_config,
        movie_config=movie_config,
        beam_mask_config=BeamMaskConfig(**config["beam_mask_config"]),
    )
    align_manager = AlignFramesManager(
        computational_config=computational_config,
        movie_config=movie_config,
        output_config=OutputConfig(**config["output_config"]),
        alignment_config=AlignFramesConfig(**config["alignment_config"]),
    )

    for movie_path in get_movie_paths(MOVIE_DIR):
        movie_config.movie_path = movie_path

        # 1. Load the raw movie once, from disk.
        movie = movie_config.movie

        # 2. Estimate the beam mask from the raw frame sum (reuses `movie`, no reload).
        beam_mask_result = beam_mask_manager.estimate(movie=movie)

        # 3. Prepare the movie for alignment (gain/dark correct, mean-zero), applying
        #    the beam mask as Poisson noise-fill outside the beam disk. Reuses `movie`
        #    again -- the movie is never read from disk a second time.
        prepared = align_manager.prepare_movie(
            movie=movie, mask=beam_mask_result.to_mask()
        )

        # 4. Estimate motion: XC pre-pass seeds global shifts, then gradient
        #    optimizer refines at the configured deformation field resolution.
        deformation_field, trajectory = align_manager.estimate_motion(prepared)

        # 5. Apply the deformation field and write all configured outputs.
        align_manager.correct_and_save(prepared, deformation_field, trajectory)


if __name__ == "__main__":
    main()
