"""Render a movie to MRC file with gain and dark correction applied.

This script performs only the movie preparation step of the alignment pipeline
(cropping to the mask, dark and gain correction, hot pixel removal, masking or noise
fill, and mean-zero normalization), on the configured device, and writes the result to
an MRC file with the movie's pixel size. This is equivalent to what would be saved with
rendered_movie_output_path in the full alignment pipeline.

It reads the same YAML file as `run_align_movies.py`. Only
`output_config.rendered_movie_output_path` is checked and written; the other output
paths are ignored.
"""

import argparse

import yaml

from ripple.managers import AlignFramesManager
from ripple.utils.data_io import write_mrc_from_tensor


def load_manager_from_yaml(yaml_path: str) -> AlignFramesManager:
    """Load an align frames manager whose only output is the rendered movie.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file (same format as for
        `run_align_movies.py`).

    Returns
    -------
    AlignFramesManager
        Manager with every output path except `rendered_movie_output_path` dropped,
        so that an unrelated existing output cannot stop this script.
    """
    with open(yaml_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    output_data = config_data.get("output_config") or {}
    config_data["output_config"] = {
        "allow_file_overwrite": output_data.get("allow_file_overwrite", True),
        "rendered_movie_output_path": output_data.get("rendered_movie_output_path"),
    }
    return AlignFramesManager(**config_data)


def render_movie_to_mrc(manager: AlignFramesManager) -> None:
    """Render a movie to MRC file with the alignment preparation applied.

    Parameters
    ----------
    manager : AlignFramesManager
        Manager providing the movie, computational, alignment and output configs.
    """
    output_config = manager.output_config
    if output_config.rendered_movie_output_path is None:
        raise ValueError(
            "rendered_movie_output_path must be specified in output_config"
        )

    print("Loading and preparing movie (dark, gain, hot pixels, mask, mean-zero)...")
    movie_prepared = manager.prepare_movie()

    print(f"Saving rendered movie to {output_config.rendered_movie_output_path}...")
    write_mrc_from_tensor(
        data=movie_prepared.cpu(),
        mrc_path=output_config.rendered_movie_output_path,
        overwrite=output_config.allow_file_overwrite,
        pixel_size=manager.movie_config.pixel_size,
    )

    print("Done!")


def main():
    """Main function to render movie to MRC."""
    parser = argparse.ArgumentParser(
        description="Render a movie to MRC file with gain and dark correction applied."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML configuration file used for run_align_movies.py",
    )
    args = parser.parse_args()

    manager = load_manager_from_yaml(args.config_path)
    render_movie_to_mrc(manager)


if __name__ == "__main__":
    main()
