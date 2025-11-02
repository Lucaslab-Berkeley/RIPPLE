"""Run the align frames manager for a directory of movies."""

# This will load the yaml.
# Also enter a dir
# Loop through the movies in dir and update yaml with movie name
# run the program as a chain, so don't have to load everything twice
import os

from ripple.managers import AlignFramesManager

# The config file for the match template
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
    """Main function to run the align frames manager for a directory of movies."""
    movie_paths = get_movie_paths(MOVIE_DIR)
    align_manager = AlignFramesManager.from_yaml(ALIGN_YAML_PATH)
    for movie_path in movie_paths:
        align_manager.movie_config.movie_path = movie_path
        align_manager.alignment_config.deformation_field_resolution = (54, 1, 1)
        deformation_field, movie_prepared, _ = align_manager.align_frames_first_passes(
            save_intermediate=False
        )
        # increase resolution for second step
        print("Aligning frames with increased resolution...")
        align_manager.alignment_config.deformation_field_resolution = (54, 4, 4)
        align_manager.alignment_config.skip_movie_preparation = True
        align_manager.align_frames_last_pass(
            movie=movie_prepared,
            deformation_field=deformation_field,
        )


if __name__ == "__main__":
    main()
