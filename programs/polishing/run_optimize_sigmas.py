"""Run sigma hyperparameter optimization for particle polishing.

This script optimizes the prior hyperparameters (sigma_A, alpha_spatial, etc.)
using a validation template to prevent overfitting. The optimization uses
bi-level optimization:
  - Inner loop: Optimize deformation field with current sigmas
  - Outer loop: Update sigmas based on validation loss

Usage:
    python run_optimize_sigmas.py

Configuration is loaded from optimize_sigmas_example_config.yaml.
Set optimize_sigmas: true in the config to enable optimization.
"""

import os

from ripple.managers import PolishParticlesManager

# Config file for sigma optimization (should have optimize_sigmas: true)
OPTIMIZE_YAML_PATH = "optimize_sigmas_example_config.yaml"
MOVIE_DIR = "../../example/movies"


def get_movie_paths(directory: str) -> list[str]:
    """Get the paths of the movies in the directory."""
    movie_paths = []
    for file in os.listdir(directory):
        if file.endswith(".mrc") or file.endswith(".tif") or file.endswith(".eer"):
            movie_paths.append(os.path.join(directory, file))
    return movie_paths


def main():
    """Run sigma optimization for movies in directory."""
    movie_paths = get_movie_paths(MOVIE_DIR)
    polish_manager = PolishParticlesManager.from_yaml(OPTIMIZE_YAML_PATH)

    # Verify sigma optimization is enabled
    if not polish_manager.alignment_config.optimize_sigmas:
        print("WARNING: optimize_sigmas is False in config!")
        print("Set optimize_sigmas: true to run sigma optimization.")
        return

    for movie_path in movie_paths:
        print(f"\nProcessing: {movie_path}")
        polish_manager.movie_config.movie_path = movie_path
        polish_manager.run_polish_particles(
            movie_extract=True,
            particle_batch_size=102,
        )


if __name__ == "__main__":
    main()
