"""Run the fused DeCo-LACE pipeline for a directory of movies.

Notes
-----
Globs all movies in a directory (eer, mrc/mrcs, or tif/tiff; any case). If
`allow_file_overwrite` is false, movies whose outputs already exist are skipped.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from ripple.config import (
    AlignFramesConfig,
    BeamMaskConfig,
    BeamMaskResult,
    ComputationalConfig,
    MovieConfig,
    OutputConfig,
)
from ripple.managers import AlignFramesManager, BeamMaskManager
from ripple.utils.data_io import MOVIE_EXTENSIONS

PIPELINE_YAML_PATH = "decolace_pipeline_example_config.yaml"
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


def write_beam_mask_result(
    result: BeamMaskResult, template: str | None, movie_path: str
) -> None:
    """Write `result` as JSON to `template` prefixed with the movie's stem (if set)."""
    stem = os.path.splitext(os.path.basename(movie_path))[0]
    path = _stem_prefixed(template, stem)
    if path is not None:
        Path(path).write_text(result.model_dump_json(indent=1), encoding="utf-8")


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

    movie_paths = get_movie_paths(MOVIE_DIR)
    out_cfg = align_manager.output_config
    movie_outputs = build_movie_outputs(movie_paths, out_cfg)
    # optional: each movie's fitted ellipse and crop (a BeamMaskResult as JSON), for
    # programs/decolace_refill, which re-fills the sum outside the beam
    beam_mask_template = config.get("beam_mask_output_path")

    for movie_path, outputs in zip(movie_paths, movie_outputs, strict=True):
        movie_config.movie_path = movie_path
        out_cfg.dw_sum_output_path = outputs.dw_sum_output_path
        out_cfg.deformation_field_output_path = outputs.deformation_field_output_path
        out_cfg.motion_corrected_movie_output_path = (
            outputs.motion_corrected_movie_output_path
        )
        out_cfg.rendered_movie_output_path = outputs.rendered_movie_output_path
        out_cfg.non_dw_sum_output_path = outputs.non_dw_sum_output_path
        out_cfg.loss_trajectories_output_path = outputs.loss_trajectories_output_path

        # Skip movies whose outputs already exist, rather than aborting the loop. All
        # outputs are checked before any work starts on this movie.
        existing_outputs = out_cfg.existing_output_paths()
        if existing_outputs and not out_cfg.allow_file_overwrite:
            print(
                f"Skipping {movie_path}: outputs already exist and "
                f"allow_file_overwrite is false: {', '.join(existing_outputs)}"
            )
            continue
        out_cfg.check_output_paths()

        # 1. Load the raw movie once, from disk, in its native dtype.
        movie = movie_config.movie

        # 2. Estimate the beam mask from the raw frame sum (reuses `movie`, no reload).
        beam_mask_result, beam_mask = beam_mask_manager.estimate_with_mask(movie=movie)
        write_beam_mask_result(beam_mask_result, beam_mask_template, movie_path)

        # 3. Prepare the movie for alignment which does the following steps:
        #    a. Crops the movie to beam mask, if requested by CropBoundsConfig
        #    b. Applies dark and gain correction
        #    c. Sets mean of each frame to zero
        #    d. Fills pixels outside mask to Poisson noise with lambda equal to the
        #       average electron count per-pixel inside the beam mask
        prepared = align_manager.prepare_movie(
            movie=movie,
            mask=beam_mask,
            crop_bounds=beam_mask_result.output_crop_bounds,
        )

        # 4. Estimate motion: XC pre-pass seeds global shifts, then gradient
        #    optimizer refines at the configured deformation field resolution.
        deformation_field, trajectory = align_manager.estimate_motion(prepared)

        # 5. Apply the deformation field and write all configured outputs.
        align_manager.correct_and_save(prepared, deformation_field, trajectory)

        # Free this movie before the next one is loaded
        del movie, beam_mask, prepared, deformation_field, trajectory


if __name__ == "__main__":
    main()
