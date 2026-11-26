"""Tests for the DeCo-LACE sum re-fill and its search mask."""

import json

import mrcfile
import numpy as np
import pytest
from scipy import ndimage

from ripple.config import BeamMaskResult, RefillConfig
from ripple.core.beam_mask import make_ellipse_mask
from ripple.core.sum_refill import (
    beam_mask_in_sum_frame,
    coloured_noise,
    contamination_mask,
    radial_power_spectrum,
    refill_sum,
)
from ripple.managers import RefillManager

SHAPE = (768, 896)
BOX = 128


def _scene(seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coloured noise in an ellipse, other white noise outside, and one dark blob."""
    rng = np.random.default_rng(seed)
    freq = np.arange(BOX // 2 + 1) / BOX
    power = 1.0 / (1.0 + (freq / 0.08) ** 2) + 0.05
    inside_noise = coloured_noise(SHAPE, freq, power, rng)
    beam = make_ellipse_mask(SHAPE, 384.0, 448.0, 330.0, 300.0, 20.0)
    image = np.where(beam, inside_noise, 3.0 + 0.5 * rng.standard_normal(SHAPE))
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    blob = (yy - 300) ** 2 + (xx - 520) ** 2 <= 30**2
    image = image - 12.0 * ndimage.gaussian_filter(blob.astype(float), 3.0)
    return image.astype(np.float32), beam, blob


def test_beam_mask_in_sum_frame_scales_and_shifts():
    raw = {"center_y": 1100.0, "center_x": 1300.0, "axis1": 600.0, "axis2": 500.0}
    mask = beam_mask_in_sum_frame(
        (800, 900),
        **raw,
        angle_deg=30.0,
        output_crop_min_y=200,
        output_crop_min_x=300,
        binning=2.0,
        diameter_reduction=0.1,
    )
    expected = make_ellipse_mask(
        (800, 900), (1100 - 200) / 2, (1300 - 300) / 2, 300.0, 250.0, 30.0, 0.1
    )
    np.testing.assert_array_equal(mask, expected)
    assert mask[450, 500]  # the ellipse centre in the sum's frame
    with pytest.raises(ValueError, match="binning"):
        beam_mask_in_sum_frame((8, 8), 4, 4, 2, 2, 0, 0, 0, binning=0)


def test_contamination_mask_finds_the_dark_blob():
    image, beam, blob = _scene()
    found = contamination_mask(image, beam, band_px=(4.0, 60.0), min_area_px=200)
    assert found[300, 520]
    assert (found & ~ndimage.binary_dilation(blob, iterations=15)).sum() == 0
    bright = contamination_mask(image, beam, band_px=(4.0, 60.0), polarity="bright")
    assert not bright[300, 520]


def test_refill_matches_interior_spectrum_and_keeps_the_beam():
    image, _, blob = _scene()
    reduced = make_ellipse_mask(SHAPE, 384.0, 448.0, 330.0, 300.0, 20.0, 0.06)
    out = refill_sum(
        image,
        reduced,
        psd_box=BOX,
        psd_step=BOX // 2,
        contamination_band_px=(4.0, 60.0),
        contamination_min_area_px=200,
        exclusion_radius_px=40.0,
    )
    res = out["image"]
    assert res.dtype == np.float32 and np.isfinite(res).all()

    # deep inside the kept region the image is unchanged
    deep = ndimage.distance_transform_edt(reduced) > 12  # 6 taper sigmas
    np.testing.assert_allclose(res[deep], image[deep], atol=1e-4)

    # outside: the interior's mean, variance and spectrum, not the old fill's
    interior = deep & ~ndimage.binary_dilation(blob, iterations=50)
    outside = ndimage.distance_transform_edt(~reduced) > 12
    assert abs(res[outside].mean() - res[interior].mean()) < 0.1 * res[interior].std()
    assert res[outside].std() == pytest.approx(res[interior].std(), rel=0.15)
    _, p_in, n_in = radial_power_spectrum(res, interior, 64, 32)
    _, p_out, n_out = radial_power_spectrum(res, outside, 64, 32)
    assert n_in > 0 and n_out > 0
    ratio = p_out[2:30] / p_in[2:30]
    assert np.all((ratio > 0.6) & (ratio < 1.6)), ratio

    # search mask: inside the reduced ellipse, away from the blob and the border
    search = out["search_mask"]
    assert not search[~reduced].any()
    assert not search[300, 520] and not search[300, 555]
    assert search[384, 448]
    assert out["stats"]["n_contamination_blobs"] == 1
    assert not search[:2].any() and not search[:, -2:].any()


def test_refill_without_contamination_and_errors():
    image, beam, _ = _scene()
    out = refill_sum(image, beam, psd_box=BOX, psd_step=BOX, contamination=False)
    assert out["stats"]["n_contamination_blobs"] == 0
    assert out["search_mask"][300, 520]  # the blob is not excluded
    with pytest.raises(ValueError, match="no 4096-px tile"):
        refill_sum(image, beam, psd_box=4096)
    with pytest.raises(ValueError, match="same shape"):
        refill_sum(image, beam[:10])


def test_refill_config_needs_one_beam_source(tmp_path):
    base = {
        "sum_path": "s.mrc",
        "refilled_sum_output_path": str(tmp_path / "r.mrc"),
        "search_mask_output_path": str(tmp_path / "m.npy"),
    }
    with pytest.raises(ValueError, match="exactly one"):
        RefillConfig(**base)
    with pytest.raises(ValueError, match="exactly one"):
        RefillConfig(**base, beam_mask_path="a.npy", beam_mask_result_path="b.json")
    with pytest.raises(ValueError, match=r"\.npy or \.mrc"):
        RefillConfig(
            **{**base, "search_mask_output_path": "m.png"}, beam_mask_path="a.npy"
        )


def test_refill_manager_from_pipeline_json(tmp_path):
    image, _, _ = _scene()
    sum_path = tmp_path / "movie_dw_sum.mrc"
    with mrcfile.new(sum_path) as mrc:
        mrc.set_data(image)
        mrc.voxel_size = 1.05
    # the sum is the raw frame's output crop, Fourier-cropped by 2
    result = BeamMaskResult(
        center_y=2 * 384.0 + 100,
        center_x=2 * 448.0 + 60,
        axis1=660.0,
        axis2=600.0,
        angle_deg=20.0,
        diameter_reduction=0.0,
        image_shape_y=1800,
        image_shape_x=2000,
        crop_min_y=100,
        crop_max_y=1635,
        crop_min_x=60,
        crop_max_x=1851,
        output_crop_min_y=100,
        output_crop_max_y=100 + 2 * SHAPE[0] - 1,
        output_crop_min_x=60,
        output_crop_max_x=60 + 2 * SHAPE[1] - 1,
        threshold_method="sobel",
        pixel_size=0.525,
        n_edge_points=700,
        n_inliers=690,
    )
    beam_json = tmp_path / "movie_beam_mask.json"
    beam_json.write_text(result.model_dump_json())
    cfg = RefillConfig(
        sum_path=str(sum_path),
        beam_mask_result_path=str(beam_json),
        psd_box=BOX,
        psd_step=BOX // 2,
        contamination_band_px=(4.0, 60.0),
        contamination_min_area_px=200,
        exclusion_radius_px=40.0,
        refilled_sum_output_path=str(tmp_path / "refill.mrc"),
        search_mask_output_path=str(tmp_path / "mask.mrc"),
        stats_output_path=str(tmp_path / "stats.json"),
    )
    manager = RefillManager(refill_config=cfg)
    expected = make_ellipse_mask(SHAPE, 384.0, 448.0, 330.0, 300.0, 20.0, 0.06)
    np.testing.assert_array_equal(manager.beam_mask(SHAPE), expected)
    manager.run()
    with mrcfile.open(tmp_path / "refill.mrc") as mrc:
        assert mrc.data.shape == SHAPE
        assert mrc.voxel_size.x == pytest.approx(1.05)
    with mrcfile.open(tmp_path / "mask.mrc") as mrc:
        mask = mrc.data.astype(bool)
    assert mask[384, 448] and not mask[0, 0]
    assert (
        json.loads((tmp_path / "stats.json").read_text())["n_contamination_blobs"] == 1
    )
    with pytest.raises(FileExistsError):
        manager.run()

    # a mask image in the sum's frame instead
    np.save(tmp_path / "beam.npy", expected)
    cfg2 = cfg.model_copy(
        update={
            "beam_mask_result_path": None,
            "beam_mask_path": str(tmp_path / "beam.npy"),
            "allow_file_overwrite": True,
        }
    )
    np.testing.assert_array_equal(
        RefillManager(refill_config=cfg2).beam_mask(SHAPE), expected
    )
