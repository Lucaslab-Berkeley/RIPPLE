"""Tests for generate_image module."""

# pylint: disable=redefined-outer-name
# (pytest fixtures intentionally use same names as parameters)

import pytest
import torch
from torch_fourier_filter.dose_weight import dose_weight_movie

from ripple.core.generate_image import (
    dose_weight_memory_efficient,
    fourier_crop_movie,
    fourier_crop_shape,
    generate_dose_weighted_image,
    generate_summed_images,
    sum_movie,
)


@pytest.fixture
def sample_movie():
    """Create a sample movie tensor for testing."""
    # Create a simple 3D movie: 5 frames of 32x32 pixels
    return torch.randn(5, 32, 32, dtype=torch.float32)


def test_sum_movie_basic(sample_movie):
    """Test that sum_movie sums along the first dimension."""
    result = sum_movie(sample_movie)

    # Should have shape (32, 32) - spatial dimensions only
    assert result.shape == (32, 32)

    # Should be sum of all frames
    expected = torch.sum(sample_movie, dim=0)
    assert torch.allclose(result, expected)


def test_sum_movie_single_frame():
    """Test sum_movie with a single frame."""
    movie = torch.randn(1, 16, 16, dtype=torch.float32)
    result = sum_movie(movie)

    assert result.shape == (16, 16)
    assert torch.allclose(result, movie[0])


def test_sum_movie_preserves_dtype(sample_movie):
    """Test that sum_movie preserves dtype."""
    result = sum_movie(sample_movie)
    assert result.dtype == sample_movie.dtype


def test_sum_movie_zeros():
    """Test sum_movie with all zeros."""
    movie = torch.zeros(3, 10, 10, dtype=torch.float32)
    result = sum_movie(movie)

    assert torch.allclose(result, torch.zeros(10, 10))


def test_generate_dose_weighted_image_basic(sample_movie):
    """Test generate_dose_weighted_image produces correct output shape."""
    result = generate_dose_weighted_image(
        movie=sample_movie,
        pixel_size=1.0,
        pre_exposure=0.0,
        fluence_per_frame=1.0,
        voltage=300.0,
    )

    # Should produce a 2D image (summed over frames)
    assert result.shape == (32, 32)
    assert isinstance(result, torch.Tensor)


def test_generate_dose_weighted_image_different_parameters(sample_movie):
    """Test generate_dose_weighted_image with different parameters."""
    result = generate_dose_weighted_image(
        movie=sample_movie,
        pixel_size=0.5,
        pre_exposure=1.0,
        fluence_per_frame=2.0,
        voltage=200.0,
    )

    # Should still produce correct shape
    assert result.shape == (32, 32)


def test_generate_dose_weighted_image_single_frame():
    """Test generate_dose_weighted_image with a single frame."""
    movie = torch.randn(1, 16, 16, dtype=torch.float32)
    result = generate_dose_weighted_image(
        movie=movie,
        pixel_size=1.0,
        pre_exposure=0.0,
        fluence_per_frame=1.0,
        voltage=300.0,
    )

    assert result.shape == (16, 16)


def test_generate_dose_weighted_image_matches_reference(sample_movie):
    """Chunked streaming result must match a direct, whole-movie reference."""
    pixel_size, pre_exposure, fluence_per_frame, voltage = 1.2, 0.5, 1.5, 200.0

    result = generate_dose_weighted_image(
        movie=sample_movie,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        fluence_per_frame=fluence_per_frame,
        voltage=voltage,
        chunk_size=2,
    )

    frame_shape = (sample_movie.shape[-2], sample_movie.shape[-1])
    movie_dft = torch.fft.rfft2(sample_movie, dim=(-2, -1))
    movie_dw_dft = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=frame_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=fluence_per_frame,
        voltage=voltage,
        crit_exposure_bfactor=-1,
        rfft=True,
        fftshift=False,
    )
    expected = torch.sum(
        torch.fft.irfft2(movie_dw_dft, s=frame_shape, dim=(-2, -1)), dim=0
    )

    assert torch.allclose(result, expected, atol=1e-5)


def test_generate_dose_weighted_image_movie_stays_on_cpu(sample_movie):
    """The input movie tensor itself must never be moved off of CPU."""
    original_data_ptr = sample_movie.data_ptr()

    generate_dose_weighted_image(
        movie=sample_movie,
        pixel_size=1.0,
        pre_exposure=0.0,
        fluence_per_frame=1.0,
        voltage=300.0,
        chunk_size=2,
    )

    assert sample_movie.device.type == "cpu"
    assert sample_movie.data_ptr() == original_data_ptr


def test_dose_weight_memory_efficient_matches_reference(sample_movie):
    """ "full" strategy result must match a direct, whole-movie reference."""
    pixel_size, pre_exposure, dose_per_frame, voltage = 1.2, 0.5, 1.5, 200.0

    result = dose_weight_memory_efficient(
        sample_movie,
        pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        voltage=voltage,
        memory_strategy="full",
        chunk_size=2,
    )

    frame_shape = (sample_movie.shape[-2], sample_movie.shape[-1])
    movie_dft = torch.fft.rfft2(sample_movie, dim=(-2, -1), norm="ortho")
    movie_dw_dft = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=frame_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        voltage=voltage,
        crit_exposure_bfactor=-1,
        rfft=True,
        fftshift=False,
    )
    expected = torch.sum(
        torch.fft.irfft2(movie_dw_dft, s=frame_shape, dim=(-2, -1), norm="ortho"),
        dim=0,
    )

    assert result.shape == (32, 32)
    assert torch.allclose(result, expected, atol=1e-5)


def test_dose_weight_memory_efficient_checkpointing_matches_full(sample_movie):
    """ "checkpointing" must be numerically identical to "full", just recomputed."""
    kwargs = {
        "pixel_size": 1.2,
        "pre_exposure": 0.5,
        "dose_per_frame": 1.5,
        "voltage": 200.0,
        "chunk_size": 2,
    }

    full_result = dose_weight_memory_efficient(
        sample_movie, memory_strategy="full", **kwargs
    )
    checkpointed_result = dose_weight_memory_efficient(
        sample_movie, memory_strategy="checkpointing", **kwargs
    )

    assert torch.allclose(full_result, checkpointed_result, atol=1e-6)


def test_dose_weight_memory_efficient_unknown_strategy_raises(sample_movie):
    with pytest.raises(ValueError, match="Unknown memory strategy"):
        dose_weight_memory_efficient(
            sample_movie, pixel_size=1.0, memory_strategy="adaptive"
        )


@pytest.mark.parametrize("memory_strategy", ["full", "checkpointing"])
def test_dose_weight_memory_efficient_is_differentiable(memory_strategy):
    """Gradients must flow back to the input movie under both strategies."""
    movie = torch.randn(4, 16, 16, dtype=torch.float32, requires_grad=True)

    result = dose_weight_memory_efficient(
        movie,
        pixel_size=1.0,
        pre_exposure=0.0,
        dose_per_frame=1.0,
        voltage=300.0,
        memory_strategy=memory_strategy,
        chunk_size=2,
    )
    result.sum().backward()

    assert movie.grad is not None
    assert torch.all(torch.isfinite(movie.grad))
    assert torch.any(movie.grad != 0)


# ---------------------------------------------------------------------------
# fourier_crop_movie
# ---------------------------------------------------------------------------


def test_fourier_crop_movie_factor_one_is_noop(sample_movie):
    """factor=1 must return the exact same tensor object and unchanged pixel size."""
    result, new_pixel_size = fourier_crop_movie(sample_movie, pixel_size=1.5, factor=1)

    assert result is sample_movie
    assert new_pixel_size == 1.5


def test_fourier_crop_movie_halves_shape_and_doubles_pixel_size():
    movie = torch.randn(5, 32, 32, dtype=torch.float32)

    result, new_pixel_size = fourier_crop_movie(movie, pixel_size=0.4, factor=2)

    assert result.shape == (5, 16, 16)
    assert new_pixel_size == pytest.approx(0.8)


def test_fourier_crop_movie_preserves_mean():
    """Fourier cropping should preserve the DC component (per-frame mean)."""
    movie = torch.randn(3, 32, 32, dtype=torch.float32) + 5.0

    result, _ = fourier_crop_movie(movie, pixel_size=0.4, factor=2)

    assert torch.allclose(
        result.mean(dim=(-2, -1)), movie.mean(dim=(-2, -1)), atol=1e-3
    )


def test_fourier_crop_movie_all_frames_cropped_consistently():
    """Every frame in the batch should be cropped with the same operation."""
    frame = torch.randn(32, 32, dtype=torch.float32)
    movie = frame.unsqueeze(0).repeat(4, 1, 1)

    result, _ = fourier_crop_movie(movie, pixel_size=0.4, factor=2)

    for i in range(1, 4):
        assert torch.allclose(result[0], result[i])


# ---------------------------------------------------------------------------
# Single-pass summation (accumulated spectra, one inverse FFT per sum)
# ---------------------------------------------------------------------------

DW_PARAMS = {"pre_exposure": 0.5, "fluence_per_frame": 1.5, "voltage": 300.0}


def _per_frame_dw_sum(movie: torch.Tensor, pixel_size: float) -> torch.Tensor:
    """Reference: dose weight every frame, inverse FFT each frame, then sum."""
    frame_shape = (movie.shape[-2], movie.shape[-1])
    movie_dw_dft = dose_weight_movie(
        movie_dft=torch.fft.rfft2(movie, dim=(-2, -1)),
        image_shape=frame_shape,
        pixel_size=pixel_size,
        pre_exposure=DW_PARAMS["pre_exposure"],
        dose_per_frame=DW_PARAMS["fluence_per_frame"],
        voltage=DW_PARAMS["voltage"],
        crit_exposure_bfactor=-1,
        rfft=True,
        fftshift=False,
    )
    per_frame = torch.fft.irfft2(movie_dw_dft, s=frame_shape, dim=(-2, -1))
    return torch.sum(per_frame, dim=0)


def _assert_close(result: torch.Tensor, expected: torch.Tensor) -> None:
    scale = expected.abs().max()
    assert result.shape == expected.shape
    assert torch.allclose(result, expected, atol=1e-5 * scale, rtol=0)


@pytest.mark.parametrize("shape", [(7, 33, 30), (6, 31, 47)])
@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_generate_dose_weighted_image_matches_per_frame_inverse_fft(shape, chunk_size):
    """One inverse FFT of the accumulated spectra equals summing per-frame images."""
    movie = torch.randn(*shape, generator=torch.Generator().manual_seed(0)) + 3.0

    result = generate_dose_weighted_image(
        movie, pixel_size=1.1, chunk_size=chunk_size, **DW_PARAMS
    )

    _assert_close(result, _per_frame_dw_sum(movie, 1.1))


@pytest.mark.parametrize(
    "shape",
    [
        (5, 64, 48),  # even -> even
        (5, 66, 50),  # even -> odd
        (5, 65, 47),  # odd -> even
        (4, 63, 66),  # mixed parity per axis
    ],
)
@pytest.mark.parametrize("factor", [2, 3])
def test_generate_summed_images_super_resolution_matches_crop_then_sum(shape, factor):
    """Per-chunk Fourier crop must match Fourier-cropping the whole movie first."""
    pixel_size = 0.4
    movie = torch.randn(*shape, generator=torch.Generator().manual_seed(1)) + 5.0
    cropped, cropped_pixel_size = fourier_crop_movie(movie, pixel_size, factor)

    dw_image, sum_image, out_pixel_size = generate_summed_images(
        movie,
        pixel_size,
        super_resolution_factor=factor,
        chunk_size=2,
        **DW_PARAMS,
    )

    assert out_pixel_size == cropped_pixel_size
    _assert_close(dw_image, _per_frame_dw_sum(cropped, cropped_pixel_size))
    _assert_close(sum_image, torch.sum(cropped, dim=0))


def test_generate_summed_images_factor_one_plain_sum_is_exact(sample_movie):
    dw_image, sum_image, out_pixel_size = generate_summed_images(
        sample_movie, 1.5, chunk_size=2, **DW_PARAMS
    )

    assert out_pixel_size == 1.5
    assert torch.allclose(sum_image, torch.sum(sample_movie, dim=0), atol=1e-6)
    _assert_close(dw_image, _per_frame_dw_sum(sample_movie, 1.5))


@pytest.mark.parametrize("factor", [1, 2])
def test_generate_summed_images_only_computes_requested(sample_movie, factor):
    dw_only = generate_summed_images(
        sample_movie,
        1.0,
        super_resolution_factor=factor,
        non_dose_weighted=False,
        **DW_PARAMS,
    )
    sum_only = generate_summed_images(
        sample_movie,
        1.0,
        super_resolution_factor=factor,
        dose_weighted=False,
        **DW_PARAMS,
    )

    assert dw_only[0] is not None and dw_only[1] is None
    assert sum_only[0] is None and sum_only[1] is not None


@pytest.mark.parametrize("frame_shape", [(32, 32), (33, 31), (65, 47)])
@pytest.mark.parametrize("factor", [1, 2, 3])
def test_fourier_crop_shape_matches_fourier_crop_movie(frame_shape, factor):
    cropped, expected_pixel_size = fourier_crop_movie(
        torch.zeros(1, *frame_shape), 0.4, factor
    )

    out_shape, out_pixel_size = fourier_crop_shape(frame_shape, 0.4, factor)

    assert out_shape == tuple(cropped.shape[-2:])
    assert out_pixel_size == expected_pixel_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_generate_summed_images_gpu_memory_is_bounded_by_the_chunk():
    """Super-resolution sums keep a few chunks of frames on the GPU, not the stack."""
    device = torch.device("cuda")
    n_frames, height, width, chunk_size = 32, 1024, 1536, 2
    movie = torch.rand(n_frames, height, width, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    generate_summed_images(
        movie,
        0.5,
        super_resolution_factor=2,
        device=device,
        chunk_size=chunk_size,
        **DW_PARAMS,
    )
    torch.cuda.synchronize()

    extra = torch.cuda.max_memory_allocated() - before
    chunk_bytes = chunk_size * height * width * 4
    # Whole-stack cropping needed several copies of the 32-frame stack (16 chunks each)
    assert extra < 8 * chunk_bytes
