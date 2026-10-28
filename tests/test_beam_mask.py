"""Tests for the core DeCo-LACE beam mask estimation algorithm."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import torch

from ripple.core.beam_mask import (
    DEFAULT_N_RAYS,
    estimate_beam_mask,
    fit_beam_edge,
    get_crop_bounds,
    make_ellipse_mask,
    sum_movie_chunked,
)

HEIGHT = 128
WIDTH = 128

# Synthetic DeCo-LACE beam, roughly a K3 super-resolution frame binned to ~8 A/px:
# ~4 % elliptical and clipped by the top and bottom of the detector.
BEAM_PIXEL_SIZE = 8.0
BEAM_SHAPE = (536, 754)
BEAM = {
    "center_y": 270.0,
    "center_x": 380.0,
    "axis1": 286.0,
    "axis2": 275.0,
    "angle_deg": 130.0,
}


def _synthetic_beam(
    scale: int = 1,
    dose: float = 50.0,
    background: float = 0.5,
    halo: float = 0.03,
    gradient: float = 0.0,
    dark_band: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Poisson frame sum of `BEAM` at ``BEAM_PIXEL_SIZE / scale`` A/px.

    `halo` is the amplitude (fraction of `dose`) of a faint exponential halo outside
    the aperture edge. `gradient` tilts the in-beam intensity linearly across x by
    +/- `gradient`. `dark_band` puts background-level sample across the right rim.
    """
    h, w = BEAM_SHAPE[0] * scale, BEAM_SHAPE[1] * scale
    cy = (BEAM["center_y"] + 0.5) * scale - 0.5
    cx = (BEAM["center_x"] + 0.5) * scale - 0.5
    a1, a2 = BEAM["axis1"] * scale, BEAM["axis2"] * scale
    theta = np.radians(BEAM["angle_deg"])

    yy, xx = np.indices((h, w), dtype=np.float64)
    dx, dy = xx - cx, yy - cy
    p1 = dx * np.cos(theta) + dy * np.sin(theta)
    p2 = -dx * np.sin(theta) + dy * np.cos(theta)
    rho = np.sqrt((p1 / a1) ** 2 + (p2 / a2) ** 2)

    beam = (rho <= 1.0) * (1.0 + gradient * dx / a1)
    if dark_band:
        beam[(dx > 0.6 * a1) & (np.abs(dy) < 0.15 * a1)] = 0.0
    decay = 65.0 * scale  # ~1000 super-resolution pixels
    halo_map = (rho > 1.0) * halo * np.exp(-(rho - 1.0) * np.sqrt(a1 * a2) / decay)

    rng = np.random.default_rng(seed)
    expected = background + dose * (beam + halo_map)
    return rng.poisson(expected).astype(np.float32)


def _assert_matches_beam(fit, scale: int = 1, rel_tol: float = 0.01) -> None:
    """Fitted ellipse matches `BEAM` (axes and centre within `rel_tol` of R)."""
    radius = np.sqrt(BEAM["axis1"] * BEAM["axis2"]) * scale
    true_axes = sorted([BEAM["axis1"] * scale, BEAM["axis2"] * scale])
    fit_axes = sorted([fit["axis1"], fit["axis2"]])
    assert fit_axes == pytest.approx(true_axes, rel=rel_tol)

    cy = (BEAM["center_y"] + 0.5) * scale - 0.5
    cx = (BEAM["center_x"] + 0.5) * scale - 0.5
    assert np.hypot(fit["center_y"] - cy, fit["center_x"] - cx) < rel_tol * radius


@pytest.fixture
def ellipse_params() -> dict:
    return {
        "shape": (HEIGHT, WIDTH),
        "center_y": 66.0,
        "center_x": 58.0,
        "axis1": 40.0,
        "axis2": 28.0,
        "angle_deg": 25.0,
    }


@pytest.fixture
def ellipse_mask(ellipse_params) -> np.ndarray:
    return make_ellipse_mask(**ellipse_params)


# ---------------------------------------------------------------------------
# sum_movie_chunked
# ---------------------------------------------------------------------------


class TestSumMovieChunked:
    """Tests for sum_movie_chunked."""

    def test_matches_direct_sum(self):
        movie = torch.randn(9, 16, 16)
        expected = movie.sum(dim=0)

        result = sum_movie_chunked(movie, device=torch.device("cpu"), chunk_size=4)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_chunk_size_does_not_affect_result(self):
        movie = torch.randn(7, 16, 16)

        device = torch.device("cpu")
        result_1 = sum_movie_chunked(movie, device=device, chunk_size=1)
        result_7 = sum_movie_chunked(movie, device=device, chunk_size=7)
        result_100 = sum_movie_chunked(movie, device=device, chunk_size=100)

        assert torch.allclose(result_1, result_7, atol=1e-5)
        assert torch.allclose(result_1, result_100, atol=1e-5)

    def test_result_on_requested_device(self):
        movie = torch.randn(3, 8, 8)
        result = sum_movie_chunked(movie, device=torch.device("cpu"))
        assert result.device == torch.device("cpu")
        assert result.shape == (8, 8)


# ---------------------------------------------------------------------------
# get_crop_bounds
# ---------------------------------------------------------------------------


class TestGetCropBounds:
    """Tests for get_crop_bounds."""

    def test_tight_bounding_box(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[4:9, 10:20] = True

        min_y, max_y, min_x, max_x = get_crop_bounds(mask)

        assert (min_y, max_y, min_x, max_x) == (4, 8, 10, 19)

    def test_raises_on_empty_mask(self):
        empty = np.zeros((10, 10), dtype=bool)
        with pytest.raises(ValueError, match="no True pixels"):
            get_crop_bounds(empty)


# ---------------------------------------------------------------------------
# make_ellipse_mask
# ---------------------------------------------------------------------------


class TestMakeEllipseMask:
    """Tests for make_ellipse_mask."""

    def test_center_pixel_inside_corner_outside(self, ellipse_params, ellipse_mask):
        cy, cx = int(ellipse_params["center_y"]), int(ellipse_params["center_x"])
        assert bool(ellipse_mask[cy, cx])
        assert not bool(ellipse_mask[0, 0])

    def test_diameter_reduction_shrinks_mask(self, ellipse_params, ellipse_mask):
        shrunk = make_ellipse_mask(**ellipse_params, diameter_reduction=0.3)
        assert shrunk.sum() < ellipse_mask.sum()
        # Shrunk mask should be fully contained within the original.
        assert np.logical_and(shrunk, ~ellipse_mask).sum() == 0

    def test_raises_when_diameter_reduction_collapses_axis(self, ellipse_params):
        with pytest.raises(ValueError, match="diameter_reduction"):
            make_ellipse_mask(**ellipse_params, diameter_reduction=1.0)


# ---------------------------------------------------------------------------
# fit_beam_edge
# ---------------------------------------------------------------------------


class TestFitBeamEdge:
    """Tests for the Sobel edge-ray beam fit."""

    def test_recovers_clipped_beam(self):
        """Beam clipped top and bottom: clipped rays are dropped, centre unbiased."""
        frame_sum = _synthetic_beam()

        fit = fit_beam_edge(frame_sum, pixel_size=BEAM_PIXEL_SIZE)

        _assert_matches_beam(fit)
        # Arcs clipped by the detector edge yield no edge points.
        assert fit["n_edge_points"] < DEFAULT_N_RAYS
        assert fit["n_inliers"] > 0.5 * DEFAULT_N_RAYS

    def test_strong_intensity_gradient_across_beam(self):
        frame_sum = _synthetic_beam(gradient=0.8)

        fit = fit_beam_edge(frame_sum, pixel_size=BEAM_PIXEL_SIZE)

        _assert_matches_beam(fit)

    def test_dark_sample_across_rim(self):
        """Background-level sample across the rim must not pull the fit inward."""
        frame_sum = _synthetic_beam(dark_band=True)

        fit = fit_beam_edge(frame_sum, pixel_size=BEAM_PIXEL_SIZE)

        _assert_matches_beam(fit)

    def test_bright_halo_does_not_bias_radius_outward(self):
        frame_sum = _synthetic_beam(halo=0.15)

        fit = fit_beam_edge(frame_sum, pixel_size=BEAM_PIXEL_SIZE)

        _assert_matches_beam(fit)

    def test_low_dose_beam(self):
        """100x lower dose: the ellipse is still recovered."""
        frame_sum = _synthetic_beam(dose=0.5, background=0.005)

        fit = fit_beam_edge(frame_sum, pixel_size=BEAM_PIXEL_SIZE)

        _assert_matches_beam(fit)

    def test_raises_on_pure_noise(self):
        rng = np.random.default_rng(0)
        frame_sum = rng.poisson(5.0, size=BEAM_SHAPE).astype(np.float32)

        with pytest.raises(ValueError, match="beam"):
            fit_beam_edge(frame_sum, pixel_size=BEAM_PIXEL_SIZE)

    @pytest.mark.parametrize("working_pixel_size", [4.0, 8.0, 16.0])
    def test_result_in_input_pixels_for_any_binning(self, working_pixel_size):
        """Fit on a 2 A/px frame sum is in input pixels whatever the binning."""
        scale = 4
        frame_sum = _synthetic_beam(scale=scale, dose=50.0 / scale**2)

        fit = fit_beam_edge(
            frame_sum,
            pixel_size=BEAM_PIXEL_SIZE / scale,
            working_pixel_size=working_pixel_size,
        )

        _assert_matches_beam(fit, scale=scale)


# ---------------------------------------------------------------------------
# estimate_beam_mask
# ---------------------------------------------------------------------------


class TestEstimateBeamMask:
    """Tests for the estimate_beam_mask orchestration function."""

    def test_full_pipeline_on_synthetic_frame_sum(self):
        frame_sum = torch.from_numpy(_synthetic_beam())

        result = estimate_beam_mask(
            frame_sum,
            pixel_size=BEAM_PIXEL_SIZE,
            threshold_method="sobel",
            diameter_reduction=0.0,
        )

        _assert_matches_beam(result)
        assert result["threshold_method"] == "sobel"
        assert result["pixel_size"] == BEAM_PIXEL_SIZE
        assert (result["image_shape_y"], result["image_shape_x"]) == BEAM_SHAPE
        assert 0 < result["n_inliers"] <= result["n_edge_points"] <= DEFAULT_N_RAYS
        # The beam is clipped top and bottom, but not left and right.
        assert result["crop_min_y"] == 0
        assert result["crop_max_y"] == BEAM_SHAPE[0] - 1
        assert 0 < result["crop_min_x"] < result["crop_max_x"] < BEAM_SHAPE[1] - 1

    def test_diameter_reduction_shrinks_crop_bounds(self):
        frame_sum = torch.from_numpy(_synthetic_beam())

        full, shrunk = (
            estimate_beam_mask(
                frame_sum,
                pixel_size=BEAM_PIXEL_SIZE,
                threshold_method="sobel",
                diameter_reduction=reduction,
            )
            for reduction in (0.0, 0.1)
        )

        assert shrunk["diameter_reduction"] == 0.1
        # Same fitted ellipse; only the mask and crop bounds shrink.
        assert shrunk["axis1"] == full["axis1"]
        assert shrunk["axis2"] == full["axis2"]
        # A 10 % shrink pulls the ellipse inside the top and bottom of the frame.
        assert full["crop_min_y"] < shrunk["crop_min_y"]
        assert shrunk["crop_max_y"] < full["crop_max_y"]
        assert full["crop_min_x"] < shrunk["crop_min_x"]
        assert shrunk["crop_max_x"] < full["crop_max_x"]

    def test_raises_on_unsupported_threshold_method(self):
        frame_sum = torch.from_numpy(_synthetic_beam())

        with pytest.raises(ValueError, match="Unsupported threshold_method"):
            estimate_beam_mask(
                frame_sum,
                pixel_size=BEAM_PIXEL_SIZE,
                threshold_method="otsu",
                diameter_reduction=0.0,
            )
