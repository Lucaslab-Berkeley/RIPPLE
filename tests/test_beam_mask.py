"""Tests for the core DeCo-LACE beam mask estimation algorithm."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import torch

from ripple.core.beam_mask import (
    DEFAULT_N_RAYS,
    estimate_beam_mask,
    fit_beam_edge,
    fit_ellipse,
    get_crop_bounds,
    low_pass_filter,
    make_ellipse_mask,
    sum_movie_chunked,
    threshold_otsu,
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


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union)


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
# low_pass_filter
# ---------------------------------------------------------------------------


class TestLowPassFilter:
    """Tests for low_pass_filter."""

    def test_preserves_uniform_image(self):
        """A constant (DC-only) image should pass through with its mean unchanged."""
        image = torch.full((HEIGHT, WIDTH), 5.0)

        result = low_pass_filter(image, pixel_size=1.0, low_pass_resolution=20.0)

        assert torch.allclose(result.mean(), image.mean(), atol=1e-3)

    def test_attenuates_high_frequency_noise(self):
        """Filtering should reduce the variance of pure high-frequency noise."""
        torch.manual_seed(0)
        noise = torch.randn(HEIGHT, WIDTH)

        result = low_pass_filter(noise, pixel_size=1.0, low_pass_resolution=20.0)

        assert result.std() < noise.std()

    def test_moves_to_requested_device(self):
        image = torch.randn(HEIGHT, WIDTH)
        result = low_pass_filter(
            image, pixel_size=1.0, low_pass_resolution=20.0, device=torch.device("cpu")
        )
        assert result.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# threshold_otsu
# ---------------------------------------------------------------------------


class TestThresholdOtsu:
    """Tests for threshold_otsu."""

    def test_separates_bimodal_clusters(self):
        rng = np.random.default_rng(0)
        low_cluster = rng.normal(loc=0.0, scale=0.5, size=5000)
        high_cluster = rng.normal(loc=10.0, scale=0.5, size=5000)
        image = np.concatenate([low_cluster, high_cluster])

        threshold = threshold_otsu(image)

        # The two clusters are well-separated, so any threshold strictly between
        # them correctly assigns every sample to its true cluster (Otsu may land
        # anywhere within the near-zero-density gap between the clusters).
        assert low_cluster.max() < threshold < high_cluster.min()


# ---------------------------------------------------------------------------
# fit_ellipse
# ---------------------------------------------------------------------------


class TestFitEllipse:
    """Tests for fit_ellipse."""

    def test_recovers_synthetic_ellipse(self, ellipse_params, ellipse_mask):
        """Fitted ellipse should closely reconstruct the original mask.

        cv2.fitEllipse can report an equivalent ellipse with axis1/axis2 swapped
        and angle offset by 90 degrees, so compare via mask IoU rather than raw
        parameters directly.
        """
        center_y, center_x, axis1, axis2, angle_deg = fit_ellipse(ellipse_mask)

        refit_mask = make_ellipse_mask(
            shape=ellipse_params["shape"],
            center_y=center_y,
            center_x=center_x,
            axis1=axis1,
            axis2=axis2,
            angle_deg=angle_deg,
        )

        assert _iou(ellipse_mask, refit_mask) > 0.95

    def test_raises_on_empty_mask(self):
        empty = np.zeros((HEIGHT, WIDTH), dtype=bool)
        with pytest.raises(ValueError, match="No connected components"):
            fit_ellipse(empty)

    def test_falls_back_for_severely_clipped_ellipse(self):
        """A beam mostly cut off by the frame edge should still fit via the fallback."""
        mask = make_ellipse_mask(
            shape=(HEIGHT, WIDTH),
            center_y=0.0,
            center_x=0.0,
            axis1=60.0,
            axis2=60.0,
            angle_deg=0.0,
        )

        center_y, center_x, axis1, axis2, _ = fit_ellipse(mask)

        assert axis1 > 0.0
        assert axis2 > 0.0
        assert center_y < HEIGHT / 2
        assert center_x < WIDTH / 2


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

    def test_full_pipeline_on_synthetic_frame_sum(self, ellipse_mask):
        frame_sum = torch.from_numpy(ellipse_mask.astype(np.float32)) * 100.0

        result = estimate_beam_mask(
            frame_sum,
            pixel_size=1.0,
            threshold_method="otsu",
            diameter_reduction=0.1,
            low_pass_resolution=10.0,
            device=torch.device("cpu"),
        )

        assert result["image_shape_y"] == HEIGHT
        assert result["image_shape_x"] == WIDTH
        assert result["threshold_method"] == "otsu"
        assert result["pixel_size"] == 1.0
        assert result["diameter_reduction"] == 0.1
        assert 0 <= result["crop_min_y"] < result["crop_max_y"] < HEIGHT
        assert 0 <= result["crop_min_x"] < result["crop_max_x"] < WIDTH

    def test_sobel_method_on_synthetic_frame_sum(self):
        frame_sum = torch.from_numpy(_synthetic_beam())

        result = estimate_beam_mask(
            frame_sum,
            pixel_size=BEAM_PIXEL_SIZE,
            threshold_method="sobel",
            diameter_reduction=0.0,
            low_pass_resolution=100.0,
        )

        _assert_matches_beam(result)
        assert result["threshold_method"] == "sobel"
        assert (result["image_shape_y"], result["image_shape_x"]) == BEAM_SHAPE
        assert result["n_inliers"] is not None
        assert result["n_edge_points"] is not None
        assert 0 < result["n_inliers"] <= result["n_edge_points"] <= DEFAULT_N_RAYS
        # The beam is clipped top and bottom, but not left and right.
        assert result["crop_min_y"] == 0
        assert result["crop_max_y"] == BEAM_SHAPE[0] - 1
        assert 0 < result["crop_min_x"] < result["crop_max_x"] < BEAM_SHAPE[1] - 1

    def test_sobel_beats_otsu_with_dark_sample_at_rim(self):
        """Dark sample at the rim notches Otsu's region; the Sobel fit stays round."""
        frame_sum = torch.from_numpy(_synthetic_beam(gradient=0.6, dark_band=True))

        def axis_ratio(method: str) -> float:
            result = estimate_beam_mask(
                frame_sum,
                pixel_size=BEAM_PIXEL_SIZE,
                threshold_method=method,
                diameter_reduction=0.0,
                low_pass_resolution=100.0,
            )
            axes = sorted([result["axis1"], result["axis2"]])
            return axes[0] / axes[1]

        true_ratio = BEAM["axis2"] / BEAM["axis1"]
        assert axis_ratio("sobel") == pytest.approx(true_ratio, abs=0.01)
        assert axis_ratio("otsu") < true_ratio - 0.02

    def test_otsu_reports_no_edge_counts(self, ellipse_mask):
        frame_sum = torch.from_numpy(ellipse_mask.astype(np.float32)) * 100.0

        result = estimate_beam_mask(
            frame_sum,
            pixel_size=1.0,
            threshold_method="otsu",
            diameter_reduction=0.0,
            low_pass_resolution=10.0,
        )

        assert result["n_edge_points"] is None
        assert result["n_inliers"] is None

    def test_raises_on_unsupported_threshold_method(self, ellipse_mask):
        frame_sum = torch.from_numpy(ellipse_mask.astype(np.float32)) * 100.0

        with pytest.raises(ValueError, match="Unsupported threshold_method"):
            estimate_beam_mask(
                frame_sum,
                pixel_size=1.0,
                threshold_method="valley",
                diameter_reduction=0.0,
                low_pass_resolution=10.0,
            )
