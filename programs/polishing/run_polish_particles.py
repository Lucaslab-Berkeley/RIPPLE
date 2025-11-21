"""Run the align frames manager for a directory of movies."""

# This will load the yaml.
# Also enter a dir
# Loop through the movies in dir and update yaml with movie name
# run the program as a chain, so don't have to load everything twice
import os

from ripple.managers import PolishParticlesManager

# The config file for the match template
POLISH_YAML_PATH = "polish_movies_example_config.yaml"
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
    polish_manager = PolishParticlesManager.from_yaml(POLISH_YAML_PATH)
    for movie_path in movie_paths:
        polish_manager.movie_config.movie_path = movie_path
        polish_manager.run_polish_particles(
            movie_extract=True,
            particle_batch_size=102,
            save_intermediate_fields=True,
        )


if __name__ == "__main__":
    main()
