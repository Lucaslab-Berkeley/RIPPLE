"""Manager for re-filling a DeCo-LACE sum and writing its 2DTM search mask."""

import json
from pathlib import Path

import mrcfile
import numpy as np
from teamtomo_basemodel import BaseModelTeamTomo

from ripple.config.beam_mask_config import BeamMaskResult
from ripple.config.refill_config import RefillConfig
from ripple.core.sum_refill import RefillResult, beam_mask_in_sum_frame, refill_sum
from ripple.utils.data_io import load_array_from_path


class RefillManager(BaseModelTeamTomo):
    """Re-fill one sum (see :mod:`ripple.core.sum_refill`) and write the outputs."""

    refill_config: RefillConfig

    def beam_mask(self, sum_shape: tuple[int, int]) -> np.ndarray:
        """The beam region to keep, in the sum's frame."""
        cfg = self.refill_config
        if cfg.beam_mask_path is not None:
            if Path(cfg.beam_mask_path).suffix.lower() == ".npy":
                mask = np.load(cfg.beam_mask_path) != 0
            else:
                mask = load_array_from_path(cfg.beam_mask_path, expected_ndim=2) != 0
            if mask.shape != tuple(sum_shape):
                raise ValueError(
                    f"beam mask shape {mask.shape} differs from the sum's {sum_shape}"
                )
            return mask
        assert cfg.beam_mask_result_path is not None
        result = BeamMaskResult.model_validate_json(
            Path(cfg.beam_mask_result_path).read_text(encoding="utf-8")
        )
        binning = cfg.binning
        if binning is None:
            crop_h = result.output_crop_max_y - result.output_crop_min_y + 1
            crop_w = result.output_crop_max_x - result.output_crop_min_x + 1
            binning = 0.5 * (crop_h / sum_shape[0] + crop_w / sum_shape[1])
        return beam_mask_in_sum_frame(
            sum_shape,
            center_y=result.center_y,
            center_x=result.center_x,
            axis1=result.axis1,
            axis2=result.axis2,
            angle_deg=result.angle_deg,
            output_crop_min_y=result.output_crop_min_y,
            output_crop_min_x=result.output_crop_min_x,
            binning=binning,
            diameter_reduction=cfg.diameter_reduction,
        )

    def run(self) -> RefillResult:
        """Re-fill the sum; write it, the search mask and the stats."""
        cfg = self.refill_config
        outputs = [cfg.refilled_sum_output_path, cfg.search_mask_output_path]
        if cfg.stats_output_path:
            outputs.append(cfg.stats_output_path)
        existing = [p for p in outputs if Path(p).exists()]
        if existing and not cfg.allow_file_overwrite:
            raise FileExistsError(
                "allow_file_overwrite is false and outputs exist: "
                + ", ".join(existing)
            )
        with mrcfile.open(cfg.sum_path, permissive=True) as mrc:
            image = np.asarray(mrc.data, dtype=np.float32).copy()
            voxel = mrc.voxel_size
            pixel_size = float(voxel.x) if voxel.x > 0 else None
        if image.ndim != 2:
            raise ValueError(f"the sum must be a 2-D image, got shape {image.shape}")

        result = refill_sum(
            image,
            self.beam_mask(image.shape),
            taper_sigma_px=cfg.taper_sigma_px,
            interior_erosion_px=cfg.interior_erosion_px,
            psd_box=cfg.psd_box,
            psd_step=cfg.psd_step,
            contamination=cfg.contamination,
            contamination_band_px=cfg.contamination_band_px,
            contamination_threshold_sd=cfg.contamination_threshold_sd,
            contamination_min_area_px=cfg.contamination_min_area_px,
            contamination_polarity=cfg.contamination_polarity,
            exclusion_radius_px=cfg.exclusion_radius_px,
            border_px=cfg.border_px,
            seed=cfg.seed,
        )

        with mrcfile.new(cfg.refilled_sum_output_path, overwrite=True) as mrc:
            mrc.set_data(result["image"])
            if pixel_size is not None:
                mrc.voxel_size = pixel_size
        if Path(cfg.search_mask_output_path).suffix.lower() == ".npy":
            np.save(cfg.search_mask_output_path, result["search_mask"])
        else:
            with mrcfile.new(cfg.search_mask_output_path, overwrite=True) as mrc:
                mrc.set_data(result["search_mask"].astype(np.int8))
                if pixel_size is not None:
                    mrc.voxel_size = pixel_size
        if cfg.stats_output_path:
            Path(cfg.stats_output_path).write_text(
                json.dumps({"sum_path": cfg.sum_path, **result["stats"]}, indent=1),
                encoding="utf-8",
            )
        return result
