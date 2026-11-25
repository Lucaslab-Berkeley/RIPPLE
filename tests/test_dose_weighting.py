"""Tests for the dose_weighting option: G&G, the default table, or a table file."""

# pylint: disable=redefined-outer-name

import mrcfile
import pytest
import torch

# torch_fourier_filter.frame_weights is in teamtomo/teamtomo#203, not yet released
pytest.importorskip("torch_fourier_filter.frame_weights")

# pylint: disable=wrong-import-position
from torch_fourier_filter.dose_weight import (
    dose_weight_frame_chunk,
    dose_weight_movie,
    dose_weight_normalization_grid,
)
from torch_fourier_filter.frame_weights import (
    FrameWeightTable,
    default_frame_bfactors,
    frame_exposures,
)

from ripple.config import MovieConfig, OutputConfig
from ripple.core.dose_weighting import (
    gg_frame_weights,
    movie_frame_weight_table,
    validate_dose_weighting,
)
from ripple.core.generate_image import (
    dose_weight_memory_efficient,
    fourier_crop_movie,
    generate_dose_weighted_image,
    generate_summed_images,
)
from ripple.managers import manager_utils

N_FRAMES, H, W = 7, 24, 20


@pytest.fixture
def movie() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(N_FRAMES, H, W, dtype=torch.float32)


@pytest.fixture
def table_path(tmp_path):
    """A table on another fractionation (finer frames) than the test movie."""
    start, end = frame_exposures(3 * N_FRAMES, 0.0, 1.0 / 3)
    table = FrameWeightTable(
        exposure_start=start,
        exposure_end=end,
        bfactor=-12.0 * start,
        log_scale=-0.05 * start - 0.3 * (start < 0.5),
        voltage=300.0,
    )
    path = tmp_path / "frame_weights.json"
    table.to_json(path)
    return path


def _old_gg_dw_sum(movie, pixel_size, pre_exposure, fluence, voltage, chunk_size):
    """RIPPLE's G&G dose-weighted sum as written before the dose_weighting option."""
    shape = tuple(movie.shape[-2:])
    ne, norm = dose_weight_normalization_grid(
        image_shape=shape,
        pixel_size=pixel_size,
        n_frames=movie.shape[0],
        pre_exposure=pre_exposure,
        dose_per_frame=fluence,
        voltage=voltage,
        crit_exposure_bfactor=-1,
        rfft=True,
        fftshift=False,
        chunk_size=chunk_size,
    )
    acc = torch.zeros(shape[0], shape[1] // 2 + 1, dtype=torch.complex64)
    for start in range(0, movie.shape[0], chunk_size):
        chunk_dft = torch.fft.rfft2(movie[start : start + chunk_size], dim=(-2, -1))
        chunk_dft = dose_weight_frame_chunk(
            chunk_dft=chunk_dft,
            frame_start_idx=start,
            Ne=ne,
            normalization=norm,
            pre_exposure=pre_exposure,
            dose_per_frame=fluence,
            voltage=voltage,
            in_place=True,
        )
        acc += chunk_dft.sum(0)
    return torch.fft.irfft2(acc, s=shape, dim=(-2, -1))


class TestValidate:
    @pytest.mark.parametrize("value", ["gg", "GG", "default", "Default"])
    def test_modes(self, value):
        assert validate_dose_weighting(value) == value.lower()

    def test_existing_table(self, table_path):
        assert validate_dose_weighting(str(table_path)) == str(table_path)

    @pytest.mark.parametrize("value", ["", "relion", "weights.yaml"])
    def test_rejects_unknown(self, value):
        with pytest.raises(ValueError, match="dose_weighting must be"):
            validate_dose_weighting(value)

    def test_rejects_missing_table(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            validate_dose_weighting(str(tmp_path / "missing.json"))

    def test_output_config_field(self, table_path):
        assert OutputConfig().dose_weighting == "gg"
        assert OutputConfig(dose_weighting="DEFAULT").dose_weighting == "default"
        assert OutputConfig(dose_weighting=str(table_path)).dose_weighting == str(
            table_path
        )
        with pytest.raises(ValueError):
            OutputConfig(dose_weighting="nonsense")


class TestMovieTable:
    def test_gg_is_none(self):
        assert movie_frame_weight_table("gg", N_FRAMES, 0.0, 1.0, 300.0) is None

    def test_default_matches_package(self):
        table = movie_frame_weight_table("default", N_FRAMES, 2.0, 1.5, 200.0)
        start, end = frame_exposures(N_FRAMES, 2.0, 1.5)
        ref = default_frame_bfactors(start, end, 200.0)
        assert table.source == "default"
        assert torch.equal(table.exposure_start, start)
        assert torch.equal(table.exposure_end, end)
        assert torch.allclose(table.bfactor, ref.bfactor)
        assert torch.allclose(table.log_scale, ref.log_scale)

    def test_table_file_is_resampled(self, table_path):
        table = movie_frame_weight_table(str(table_path), N_FRAMES, 0.0, 1.0, 300.0)
        assert table.n_frames == N_FRAMES
        # B = -12 * start of the finer frames, interpolated at the coarse mid exposures
        mids = 0.5 * (table.exposure_start + table.exposure_end)
        fine = FrameWeightTable.from_json(table_path)
        expected = torch.as_tensor(
            __import__("numpy").interp(
                mids.numpy(), fine.exposure_mid.numpy(), fine.bfactor.numpy()
            )
        )
        assert torch.allclose(table.bfactor, expected)


class TestGGWeights:
    def test_matches_torch_fourier_filter(self):
        """gg_frame_weights reproduces the weights of RIPPLE's G&G sums."""
        shape, pixel_size = (32, 32), 1.1
        for voltage in (300.0, 200.0):
            ones = torch.ones(
                N_FRAMES, shape[0], shape[1] // 2 + 1, dtype=torch.complex64
            )
            w = dose_weight_movie(
                ones,
                shape,
                pixel_size,
                pre_exposure=0.5,
                dose_per_frame=1.3,
                voltage=voltage,
            ).real
            k = torch.fft.rfftfreq(shape[1]) / pixel_size  # the ky = 0 row
            mine = gg_frame_weights(k, N_FRAMES, 0.5, 1.3, voltage)
            assert torch.allclose(mine[:, 1:].float(), w[:, 0, 1:], rtol=1e-5)


class TestSums:
    @pytest.mark.parametrize("chunk_size", [1, 3, 8])
    def test_gg_path_bit_identical(self, movie, chunk_size):
        """frame_weights=None keeps the G&G sum exactly as before the option."""
        args = (1.05, 0.5, 1.2, 300.0)
        new = generate_dose_weighted_image(movie, *args, chunk_size=chunk_size)
        old = _old_gg_dw_sum(movie, *args, chunk_size=chunk_size)
        assert torch.equal(new, old)

    @pytest.mark.parametrize("chunk_size", [1, 3, 8])
    def test_frame_weights_match_dose_weight_movie(self, movie, chunk_size):
        """The chunked sum equals torch_fourier_filter's whole-movie weighting."""
        pixel_size = 0.9
        table = movie_frame_weight_table("default", N_FRAMES, 0.0, 6.0, 300.0)
        result = generate_dose_weighted_image(
            movie,
            pixel_size,
            0.0,
            6.0,
            300.0,
            chunk_size=chunk_size,
            frame_weights=table,
        )
        movie_dft = torch.fft.rfft2(movie, dim=(-2, -1))
        weighted = dose_weight_movie(movie_dft, (H, W), pixel_size, frame_weights=table)
        expected = torch.fft.irfft2(weighted.sum(0), s=(H, W), dim=(-2, -1))
        assert torch.allclose(result, expected, atol=1e-5)
        # and it differs from G&G
        gg = generate_dose_weighted_image(movie, pixel_size, 0.0, 6.0, 300.0)
        assert not torch.allclose(result, gg, atol=1e-3)

    def test_frame_count_mismatch_raises(self, movie):
        table = movie_frame_weight_table("default", N_FRAMES + 1, 0.0, 1.0, 300.0)
        with pytest.raises(ValueError, match="frames"):
            generate_dose_weighted_image(
                movie, 1.0, 0.0, 1.0, 300.0, frame_weights=table
            )
        with pytest.raises(ValueError, match="frames"):
            dose_weight_memory_efficient(
                movie, 1.0, memory_strategy="full", frame_weights=table
            )

    def test_super_resolution_uses_the_cropped_grid(self, movie):
        """Weights are applied on the output (cropped) grid, as for G&G."""
        table = movie_frame_weight_table("default", N_FRAMES, 0.0, 6.0, 300.0)
        dw, _, out_pixel = generate_summed_images(
            movie,
            0.5,
            0.0,
            6.0,
            300.0,
            super_resolution_factor=2,
            non_dose_weighted=False,
            chunk_size=2,
            frame_weights=table,
        )
        cropped, pixel = fourier_crop_movie(movie, 0.5, 2)
        expected = generate_dose_weighted_image(
            cropped, pixel, 0.0, 6.0, 300.0, frame_weights=table
        )
        assert out_pixel == pytest.approx(pixel)
        assert torch.allclose(dw, expected, atol=1e-5)

    @pytest.mark.parametrize("strategy", ["full", "checkpointing"])
    def test_streaming_path(self, movie, strategy):
        """The differentiable streaming sum agrees with the chunked sum."""
        table = movie_frame_weight_table("default", N_FRAMES, 0.0, 6.0, 300.0)
        x = movie.clone().requires_grad_(True)
        out = dose_weight_memory_efficient(
            x,
            0.9,
            0.0,
            6.0,
            300.0,
            memory_strategy=strategy,
            chunk_size=3,
            frame_weights=table,
        )
        expected = generate_dose_weighted_image(
            movie, 0.9, 0.0, 6.0, 300.0, frame_weights=table
        )
        assert torch.allclose(out.detach(), expected, atol=1e-5)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_save_results_uses_dose_weighting(tmp_path, movie, table_path):
    """save_results writes the dw_sum with the configured frame weights."""
    movie_config = MovieConfig(
        pixel_size=0.8, fluence=N_FRAMES * 1.0, fluence_per_frame=1.0, voltage=300.0
    )
    for mode in ("gg", "default", str(table_path)):
        out = tmp_path / f"dw_{len(mode)}.mrc"
        output_config = OutputConfig(dw_sum_output_path=str(out), dose_weighting=mode)
        manager_utils.save_results(
            output_config, movie_config, movie, None, movie, None, device="cpu"
        )
        with mrcfile.open(out) as m:
            written = torch.as_tensor(m.data.copy())
        table = movie_frame_weight_table(mode, N_FRAMES, 0.0, 1.0, 300.0)
        expected = generate_dose_weighted_image(
            movie, 0.8, 0.0, 1.0, 300.0, frame_weights=table
        )
        assert torch.allclose(written, expected, atol=1e-6)
