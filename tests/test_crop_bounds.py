"""Tests for general-purpose mask crop-bound determination."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest
import torch

from ripple.core.crop_bounds import crop_movie, determine_crop_bounds, get_crop_bounds

HEIGHT = 40
WIDTH = 50


@pytest.fixture
def solid_mask() -> np.ndarray:
    """A single rectangular block with no holes, off-center in a larger frame."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[10:20, 15:35] = True
    return mask


@pytest.fixture
def holed_mask() -> np.ndarray:
    """The same block as `solid_mask`, with a small enclosed hole punched in it."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[10:20, 15:35] = True
    mask[13:17, 22:28] = False
    return mask


# ---------------------------------------------------------------------------
# get_crop_bounds (re-exported from ripple.core.beam_mask's original location)
# ---------------------------------------------------------------------------


class TestGetCropBounds:
    """Sanity checks for get_crop_bounds at its new canonical location."""

    def test_tight_bounding_box(self, solid_mask):
        assert get_crop_bounds(solid_mask) == (10, 19, 15, 34)

    def test_bounding_box_ignores_interior_holes(self, holed_mask, solid_mask):
        # A hole in the middle does not change the outer bounding box at all.
        assert get_crop_bounds(holed_mask) == get_crop_bounds(solid_mask)


# ---------------------------------------------------------------------------
# determine_crop_bounds
# ---------------------------------------------------------------------------


class TestDetermineCropBounds:
    """Tests for the determine_crop_bounds orchestration function."""

    def test_mode_none_returns_full_frame(self, solid_mask):
        bounds = determine_crop_bounds(solid_mask, mode="none")
        assert (bounds["min_y"], bounds["max_y"]) == (0, HEIGHT - 1)
        assert (bounds["min_x"], bounds["max_x"]) == (0, WIDTH - 1)

    def test_mode_tight_matches_get_crop_bounds(self, solid_mask):
        bounds = determine_crop_bounds(solid_mask, mode="tight")
        expected = get_crop_bounds(solid_mask)
        assert (bounds["min_y"], bounds["max_y"], bounds["min_x"], bounds["max_x"]) == (
            expected
        )

    def test_nice_size_grows_to_multiple_of_round_to(self, solid_mask):
        bounds = determine_crop_bounds(solid_mask, mode="nice_size", round_to=16)
        height = bounds["max_y"] - bounds["min_y"] + 1
        width = bounds["max_x"] - bounds["min_x"] + 1
        assert height % 16 == 0
        assert width % 16 == 0
        # Grown region still fully contains the original tight bounding box.
        tight = get_crop_bounds(solid_mask)
        assert bounds["min_y"] <= tight[0] and bounds["max_y"] >= tight[1]
        assert bounds["min_x"] <= tight[2] and bounds["max_x"] >= tight[3]
        assert 0 <= bounds["min_y"] and bounds["max_y"] < HEIGHT
        assert 0 <= bounds["min_x"] and bounds["max_x"] < WIDTH

    def test_fixed_size_centers_window_on_mask(self, solid_mask):
        bounds = determine_crop_bounds(
            solid_mask, mode="fixed_size", target_shape=(20, 30)
        )
        assert bounds["max_y"] - bounds["min_y"] + 1 == 20
        assert bounds["max_x"] - bounds["min_x"] + 1 == 30
        tight = get_crop_bounds(solid_mask)
        assert bounds["min_y"] <= tight[0] and bounds["max_y"] >= tight[1]
        assert bounds["min_x"] <= tight[2] and bounds["max_x"] >= tight[3]

    def test_fixed_size_clamps_near_border(self):
        mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        mask[0:5, 0:5] = True  # mask touches the top-left corner

        bounds = determine_crop_bounds(mask, mode="fixed_size", target_shape=(20, 20))

        assert bounds["min_y"] == 0
        assert bounds["min_x"] == 0
        assert bounds["max_y"] - bounds["min_y"] + 1 == 20
        assert bounds["max_x"] - bounds["min_x"] + 1 == 20

    def test_fixed_size_requires_target_shape(self, solid_mask):
        with pytest.raises(ValueError, match="target_shape must be set"):
            determine_crop_bounds(solid_mask, mode="fixed_size")

    def test_fixed_size_raises_when_smaller_than_tight_bbox(self, solid_mask):
        with pytest.raises(ValueError, match="smaller than the mask's tight"):
            determine_crop_bounds(solid_mask, mode="fixed_size", target_shape=(5, 5))

    def test_fixed_size_raises_when_larger_than_frame(self, solid_mask):
        with pytest.raises(ValueError, match="exceeds the frame size"):
            determine_crop_bounds(
                solid_mask, mode="fixed_size", target_shape=(HEIGHT + 1, WIDTH)
            )

    def test_unknown_mode_raises(self, solid_mask):
        with pytest.raises(ValueError, match="Unknown crop mode"):
            determine_crop_bounds(solid_mask, mode="bogus")  # type: ignore[arg-type]

    def test_raises_on_empty_mask(self):
        with pytest.raises(ValueError, match="no True pixels"):
            determine_crop_bounds(np.zeros((10, 10), dtype=bool), mode="tight")

    @pytest.mark.parametrize("mode", ["tight", "nice_size"])
    def test_interior_holes_never_shrink_the_crop(self, holed_mask, solid_mask, mode):
        # A hole inside the mask must never cause the crop to shrink relative to
        # what the same outer shape would produce without the hole -- the hole
        # stays inside the crop and is expected to be noise-filled downstream.
        holed_bounds = determine_crop_bounds(holed_mask, mode=mode, round_to=8)
        solid_bounds = determine_crop_bounds(solid_mask, mode=mode, round_to=8)
        assert holed_bounds == solid_bounds

    def test_interior_holes_never_shrink_fixed_size_crop(self, holed_mask):
        bounds = determine_crop_bounds(
            holed_mask, mode="fixed_size", target_shape=(20, 30)
        )
        assert bounds["max_y"] - bounds["min_y"] + 1 == 20
        assert bounds["max_x"] - bounds["min_x"] + 1 == 30
        tight = get_crop_bounds(holed_mask)
        assert bounds["min_y"] <= tight[0] and bounds["max_y"] >= tight[1]
        assert bounds["min_x"] <= tight[2] and bounds["max_x"] >= tight[3]


# ---------------------------------------------------------------------------
# determine_crop_bounds -- divisible_by (super-resolution factor constraint)
# ---------------------------------------------------------------------------


class TestDetermineCropBoundsDivisibleBy:
    """Tests for the divisible_by constraint (typically a super-resolution factor)."""

    def test_divisible_by_one_is_noop_for_tight(self, solid_mask):
        # solid_mask's tight bbox is (10, 19, 15, 34) -> shape (10, 20), already
        # sanity-checked against get_crop_bounds elsewhere in this file.
        bounds = determine_crop_bounds(solid_mask, mode="tight", divisible_by=1)
        assert bounds == determine_crop_bounds(solid_mask, mode="tight")

    def test_tight_grows_to_multiple_of_divisible_by(self, solid_mask):
        # Tight bbox height=10 is already a multiple of 4; width=20 isn't a
        # multiple of 3, so only the width should grow.
        bounds = determine_crop_bounds(solid_mask, mode="tight", divisible_by=3)
        height = bounds["max_y"] - bounds["min_y"] + 1
        width = bounds["max_x"] - bounds["min_x"] + 1
        assert height % 3 == 0
        assert width % 3 == 0
        tight = get_crop_bounds(solid_mask)
        assert bounds["min_y"] <= tight[0] and bounds["max_y"] >= tight[1]
        assert bounds["min_x"] <= tight[2] and bounds["max_x"] >= tight[3]

    def test_nice_size_satisfies_both_round_to_and_divisible_by(self, solid_mask):
        # round_to=8 and divisible_by=3 are coprime -- growing to a multiple of
        # just one would not satisfy the other, so the result must be a multiple
        # of lcm(8, 3) = 24 to satisfy both simultaneously. (Chosen small enough,
        # relative to the 40x50 frame, to grow without hitting the frame edge.)
        bounds = determine_crop_bounds(
            solid_mask, mode="nice_size", round_to=8, divisible_by=3
        )
        height = bounds["max_y"] - bounds["min_y"] + 1
        width = bounds["max_x"] - bounds["min_x"] + 1
        assert height % 8 == 0
        assert height % 3 == 0
        assert width % 8 == 0
        assert width % 3 == 0

    def test_fixed_size_accepts_divisible_target_shape(self, solid_mask):
        bounds = determine_crop_bounds(
            solid_mask, mode="fixed_size", target_shape=(20, 30), divisible_by=2
        )
        assert bounds["max_y"] - bounds["min_y"] + 1 == 20
        assert bounds["max_x"] - bounds["min_x"] + 1 == 30

    def test_fixed_size_rejects_non_divisible_target_shape(self, solid_mask):
        # 30 is not a multiple of 4 -- must fail loudly rather than silently
        # grow past the caller's explicit fixed_size request.
        with pytest.raises(ValueError, match="not an exact multiple of"):
            determine_crop_bounds(
                solid_mask, mode="fixed_size", target_shape=(20, 30), divisible_by=4
            )

    def test_mode_none_ignores_divisible_by(self, solid_mask):
        # "none" always spans the full (odd-or-not) frame; divisible_by is not a
        # cropping decision in this mode.
        bounds = determine_crop_bounds(solid_mask, mode="none", divisible_by=7)
        assert (bounds["min_y"], bounds["max_y"]) == (0, HEIGHT - 1)
        assert (bounds["min_x"], bounds["max_x"]) == (0, WIDTH - 1)


# ---------------------------------------------------------------------------
# crop_movie
# ---------------------------------------------------------------------------


class TestCropMovie:
    """Tests for crop_movie."""

    def test_crops_movie_tensor(self):
        movie = torch.arange(3 * HEIGHT * WIDTH, dtype=torch.float32).reshape(
            3, HEIGHT, WIDTH
        )
        bounds = {"min_y": 10, "max_y": 19, "min_x": 15, "max_x": 34}
        cropped = crop_movie(movie, bounds)

        assert cropped.shape == (3, 10, 20)
        assert torch.equal(cropped, movie[:, 10:20, 15:35])

    def test_crops_single_frame(self):
        frame = torch.arange(HEIGHT * WIDTH, dtype=torch.float32).reshape(HEIGHT, WIDTH)
        bounds = {"min_y": 0, "max_y": 4, "min_x": 0, "max_x": 4}
        cropped = crop_movie(frame, bounds)

        assert cropped.shape == (5, 5)
        assert torch.equal(cropped, frame[0:5, 0:5])
