"""Tests for the core DeCo-LACE beam mask estimation algorithm."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import torch

from ripple.core.beam_mask import (
    estimate_beam_mask,
    fit_ellipse,
    get_crop_bounds,
    low_pass_filter,
    make_ellipse_mask,
    sum_movie_chunked,
    threshold_otsu,
)

HEIGHT = 128
WIDTH = 128


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
        # Default crop_mode="none": output bounds span the full frame.
        assert (result["output_crop_min_y"], result["output_crop_max_y"]) == (
            0,
            HEIGHT - 1,
        )
        assert (result["output_crop_min_x"], result["output_crop_max_x"]) == (
            0,
            WIDTH - 1,
        )

    def test_crop_mode_tight_matches_tight_crop_bounds(self, ellipse_mask):
        frame_sum = torch.from_numpy(ellipse_mask.astype(np.float32)) * 100.0

        result = estimate_beam_mask(
            frame_sum,
            pixel_size=1.0,
            threshold_method="otsu",
            diameter_reduction=0.1,
            low_pass_resolution=10.0,
            device=torch.device("cpu"),
            crop_mode="tight",
        )

        assert (result["output_crop_min_y"], result["output_crop_max_y"]) == (
            result["crop_min_y"],
            result["crop_max_y"],
        )
        assert (result["output_crop_min_x"], result["output_crop_max_x"]) == (
            result["crop_min_x"],
            result["crop_max_x"],
        )

    def test_crop_mode_nice_size_rounds_output_bounds(self, ellipse_mask):
        frame_sum = torch.from_numpy(ellipse_mask.astype(np.float32)) * 100.0

        result = estimate_beam_mask(
            frame_sum,
            pixel_size=1.0,
            threshold_method="otsu",
            diameter_reduction=0.1,
            low_pass_resolution=10.0,
            device=torch.device("cpu"),
            crop_mode="nice_size",
            crop_round_to=16,
        )

        height = result["output_crop_max_y"] - result["output_crop_min_y"] + 1
        width = result["output_crop_max_x"] - result["output_crop_min_x"] + 1
        assert height % 16 == 0
        assert width % 16 == 0

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
