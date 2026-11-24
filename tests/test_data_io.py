"""Tests for data_io loading helpers and MovieConfig file-dispatch properties."""

# pylint: disable=redefined-outer-name

import mrcfile
import numpy as np
import pytest
import torch
from tifffile import imwrite

from ripple.config.movie_config import MovieConfig
from ripple.utils import data_io
from ripple.utils.data_io import (
    load_array_from_path,
    load_tensor_from_path,
    write_mrc_from_tensor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_data() -> np.ndarray:
    """A small 2-D float32 array."""
    return np.random.rand(16, 16).astype(np.float32)


@pytest.fixture
def movie_data() -> np.ndarray:
    """A small 3-D float32 array (frames x height x width)."""
    return np.random.rand(20, 16, 16).astype(np.float32)


@pytest.fixture
def mrc_image(tmp_path, image_data):
    """Write a 2-D MRC file; yield path and the original array."""
    path = tmp_path / "image.mrc"
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(image_data)
    return path, image_data


@pytest.fixture
def mrc_movie(tmp_path, movie_data):
    """Write a 3-D MRC file; yield path and the original array."""
    path = tmp_path / "movie.mrc"
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(movie_data)
    return path, movie_data


@pytest.fixture
def tif_image(tmp_path, image_data):
    """Write a 2-D TIFF file; yield path and the original array."""
    path = tmp_path / "image.tif"
    imwrite(str(path), image_data)
    return path, image_data


@pytest.fixture
def tif_movie(tmp_path, movie_data):
    """Write a 3-D TIFF file; yield path and the original array."""
    path = tmp_path / "movie.tif"
    imwrite(str(path), movie_data, photometric="minisblack")
    return path, movie_data


# ---------------------------------------------------------------------------
# load_tensor_from_path (images)
# ---------------------------------------------------------------------------


def test_load_tensor_from_path_image_mrc(mrc_image):
    path, data = mrc_image
    result = load_tensor_from_path(path, expected_ndim=2)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_tensor_from_path_image_tif(tif_image):
    path, data = tif_image
    result = load_tensor_from_path(path, expected_ndim=2)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_array_from_path_image_tif(tif_image):
    path, data = tif_image
    result = load_array_from_path(path, expected_ndim=2)
    assert isinstance(result, np.ndarray)
    assert result.shape == (16, 16)
    assert np.allclose(result, data)


def test_load_tensor_from_path_image_tiff_extension(tmp_path, image_data):
    path = tmp_path / "image.tiff"
    imwrite(str(path), image_data)
    result = load_tensor_from_path(path, expected_ndim=2)
    assert result.shape == torch.Size([16, 16])


def test_load_tensor_from_path_image_gain_extension(tmp_path, image_data):
    path = tmp_path / "image.gain"
    imwrite(str(path), image_data)
    result = load_tensor_from_path(path, expected_ndim=2)
    assert result.shape == torch.Size([16, 16])


def test_load_tensor_from_path_image_dark_extension(tmp_path, image_data):
    path = tmp_path / "image.dark"
    imwrite(str(path), image_data)
    result = load_tensor_from_path(path, expected_ndim=2)
    assert result.shape == torch.Size([16, 16])


def test_load_tensor_from_path_image_unsupported_extension(tmp_path):
    path = tmp_path / "image.xyz"
    path.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_tensor_from_path(path, expected_ndim=2)


# ---------------------------------------------------------------------------
# load_tensor_from_path (movies)
# ---------------------------------------------------------------------------


def test_load_tensor_from_path_movie_mrc(mrc_movie):
    path, data = mrc_movie
    result = load_tensor_from_path(path, expected_ndim=3)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([20, 16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_tensor_from_path_movie_tif(tif_movie):
    path, data = tif_movie
    result = load_tensor_from_path(path, expected_ndim=3)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([20, 16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_tensor_from_path_movie_unsupported_extension(tmp_path):
    path = tmp_path / "movie.eer"
    path.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_tensor_from_path(path, expected_ndim=3)


# ---------------------------------------------------------------------------
# MovieConfig.movie property
# ---------------------------------------------------------------------------


def _base_movie_config_kwargs(**overrides):
    defaults = {
        "movie_path": "placeholder.mrc",
        "pixel_size": 1.0,
        "fluence": 40.0,
        "fluence_per_frame": 2.0,
    }
    defaults.update(overrides)
    return defaults


def test_movie_config_movie_mrc(mrc_movie):
    path, data = mrc_movie
    cfg = MovieConfig(**_base_movie_config_kwargs(movie_path=str(path)))
    result = cfg.movie
    assert result.shape == torch.Size([20, 16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_movie_config_movie_tif(tif_movie):
    path, data = tif_movie
    cfg = MovieConfig(**_base_movie_config_kwargs(movie_path=str(path)))
    result = cfg.movie
    assert result.shape == torch.Size([20, 16, 16])
    assert torch.allclose(result, torch.tensor(data))


# ---------------------------------------------------------------------------
# MovieConfig.gain property
# ---------------------------------------------------------------------------


def test_movie_config_gain_none():
    cfg = MovieConfig(**_base_movie_config_kwargs(gain_path=None))
    assert cfg.gain is None


def test_movie_config_gain_mrc(mrc_image, tmp_path, movie_data):
    gain_path, gain_data = mrc_image
    movie_path = tmp_path / "movie.mrc"
    with mrcfile.new(str(movie_path), overwrite=True) as mrc:
        mrc.set_data(movie_data)
    cfg = MovieConfig(
        **_base_movie_config_kwargs(
            movie_path=str(movie_path), gain_path=str(gain_path)
        )
    )
    result = cfg.gain
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(gain_data))


def test_movie_config_gain_tif(tif_image, tmp_path, movie_data):
    gain_path, gain_data = tif_image
    movie_path = tmp_path / "movie.mrc"
    with mrcfile.new(str(movie_path), overwrite=True) as mrc:
        mrc.set_data(movie_data)
    cfg = MovieConfig(
        **_base_movie_config_kwargs(
            movie_path=str(movie_path), gain_path=str(gain_path)
        )
    )
    result = cfg.gain
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(gain_data))


# ---------------------------------------------------------------------------
# MovieConfig.dark property
# ---------------------------------------------------------------------------


def test_movie_config_dark_none():
    cfg = MovieConfig(**_base_movie_config_kwargs(dark_path=None))
    assert cfg.dark is None


def test_movie_config_dark_mrc(mrc_image, tmp_path, movie_data):
    dark_path, dark_data = mrc_image
    movie_path = tmp_path / "movie.mrc"
    with mrcfile.new(str(movie_path), overwrite=True) as mrc:
        mrc.set_data(movie_data)
    cfg = MovieConfig(
        **_base_movie_config_kwargs(
            movie_path=str(movie_path), dark_path=str(dark_path)
        )
    )
    result = cfg.dark
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(dark_data))


def test_movie_config_dark_tif(tif_image, tmp_path, movie_data):
    dark_path, dark_data = tif_image
    movie_path = tmp_path / "movie.mrc"
    with mrcfile.new(str(movie_path), overwrite=True) as mrc:
        mrc.set_data(movie_data)
    cfg = MovieConfig(
        **_base_movie_config_kwargs(
            movie_path=str(movie_path), dark_path=str(dark_path)
        )
    )
    result = cfg.dark
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(dark_data))


# ---------------------------------------------------------------------------
# MovieConfig super-resolution downsampling fields
# ---------------------------------------------------------------------------


def test_movie_config_super_resolution_defaults():
    cfg = MovieConfig(**_base_movie_config_kwargs(pixel_size=0.4))
    assert cfg.super_resolution_factor == 1
    assert cfg.output_pixel_size == pytest.approx(0.4)


def test_movie_config_output_pixel_size_scales_with_factor():
    cfg = MovieConfig(
        **_base_movie_config_kwargs(pixel_size=0.4, super_resolution_factor=2)
    )
    assert cfg.output_pixel_size == pytest.approx(0.8)


def test_movie_config_super_resolution_factor_must_be_positive():
    with pytest.raises(ValueError):
        MovieConfig(
            **_base_movie_config_kwargs(pixel_size=0.4, super_resolution_factor=0)
        )


# ---------------------------------------------------------------------------
# write_mrc_from_tensor pixel_size (voxel size) metadata
# ---------------------------------------------------------------------------


def test_write_mrc_from_tensor_writes_voxel_size(tmp_path):
    path = tmp_path / "out.mrc"
    data = torch.randn(16, 16, dtype=torch.float32)

    write_mrc_from_tensor(data=data, mrc_path=path, overwrite=True, pixel_size=0.8)

    with mrcfile.open(str(path)) as mrc:
        assert mrc.voxel_size.x == pytest.approx(0.8, abs=1e-5)
        assert mrc.voxel_size.y == pytest.approx(0.8, abs=1e-5)


def test_write_mrc_from_tensor_no_pixel_size_leaves_voxel_size_zero(tmp_path):
    path = tmp_path / "out.mrc"
    data = torch.randn(16, 16, dtype=torch.float32)

    write_mrc_from_tensor(data=data, mrc_path=path, overwrite=True)

    with mrcfile.open(str(path)) as mrc:
        assert mrc.voxel_size.x == 0


# ---------------------------------------------------------------------------
# File extensions (case-insensitive, .mrcs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["movie.mrcs", "movie.MRC", "movie.MRCS"])
def test_load_tensor_from_path_mrc_extensions(tmp_path, movie_data, name):
    path = tmp_path / name
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(movie_data)
    result = load_tensor_from_path(path, expected_ndim=3)
    assert torch.allclose(result, torch.tensor(movie_data))


@pytest.mark.parametrize("name", ["movie.TIF", "movie.TIFF", "movie.Tiff"])
def test_load_tensor_from_path_tiff_extensions(tmp_path, movie_data, name):
    path = tmp_path / name
    imwrite(str(path), movie_data, photometric="minisblack")
    result = load_tensor_from_path(path, expected_ndim=3)
    assert torch.allclose(result, torch.tensor(movie_data))


@pytest.mark.parametrize("name", ["movie.mrcs", "movie.TIFF"])
def test_movie_config_movie_accepts_extension(tmp_path, movie_data, name):
    path = tmp_path / name
    if name.endswith("mrcs"):
        with mrcfile.new(str(path), overwrite=True) as mrc:
            mrc.set_data(movie_data)
    else:
        imwrite(str(path), movie_data, photometric="minisblack")
    cfg = MovieConfig(**_base_movie_config_kwargs(movie_path=str(path)))
    assert torch.allclose(cfg.movie, torch.tensor(movie_data))


# ---------------------------------------------------------------------------
# Native dtype on load
# ---------------------------------------------------------------------------


def test_movie_config_movie_keeps_native_tiff_dtype(tmp_path):
    counts = np.random.default_rng(0).integers(0, 200, (6, 16, 16), dtype=np.uint8)
    path = tmp_path / "movie.tif"
    imwrite(str(path), counts, photometric="minisblack")
    cfg = MovieConfig(**_base_movie_config_kwargs(movie_path=str(path)))

    result = cfg.movie

    assert result.dtype == torch.uint8
    assert np.array_equal(result.numpy(), counts)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.uint16])
def test_movie_config_movie_keeps_native_mrc_dtype(tmp_path, dtype):
    counts = np.random.default_rng(0).integers(0, 100, (6, 16, 16)).astype(dtype)
    path = tmp_path / "movie.mrc"
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(counts)
    cfg = MovieConfig(**_base_movie_config_kwargs(movie_path=str(path)))

    result = cfg.movie

    assert result.dtype == torch.from_numpy(counts).dtype
    assert np.array_equal(result.numpy(), counts)


def test_gain_stays_float32_for_integer_file(tmp_path):
    """Gain/dark/mask references are still converted to float32 on load."""
    gain = np.full((16, 16), 3, dtype=np.uint8)
    path = tmp_path / "gain.tif"
    imwrite(str(path), gain)
    cfg = MovieConfig(**_base_movie_config_kwargs(gain_path=str(path)))
    assert cfg.gain.dtype == torch.float32
    assert torch.all(cfg.gain == 3.0)


def test_load_tensor_from_path_dtype_none_shares_memory(tmp_path, movie_data):
    path = tmp_path / "movie.tif"
    imwrite(str(path), movie_data, photometric="minisblack")
    result = load_tensor_from_path(path, expected_ndim=3, dtype=None)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor(movie_data))


@pytest.mark.parametrize(
    "array",
    [
        np.arange(12, dtype=">i2").reshape(3, 4),  # big-endian
        np.arange(12, dtype=np.uint16).reshape(3, 4),
    ],
)
def test_array_to_tensor_handles_byte_order_and_read_only(array):
    array.flags.writeable = False
    result = data_io._array_to_tensor(array)
    assert np.array_equal(result.numpy(), array)


# ---------------------------------------------------------------------------
# EER rendering
# ---------------------------------------------------------------------------


def test_movie_config_eer_passes_chunk_size_and_keeps_uint16(monkeypatch):
    calls = []

    def _fake_render(path, **kwargs):
        calls.append((path, kwargs))
        return np.ones((3, 8, 8), dtype=np.uint16)

    monkeypatch.setattr(data_io.eerfile, "render", _fake_render)
    cfg = MovieConfig(
        **_base_movie_config_kwargs(movie_path="movie.EER", eer_render_chunk_size=4)
    )

    result = cfg.movie

    assert result.dtype == torch.uint16
    assert calls[0][1]["chunk_size"] == 4
    assert calls[0][1]["dose_per_output_frame"] == 2.0
    assert calls[0][1]["total_fluence"] == 40.0


def test_movie_config_eer_chunk_size_default():
    cfg = MovieConfig(**_base_movie_config_kwargs())
    assert cfg.eer_render_chunk_size == data_io.DEFAULT_EER_RENDER_CHUNK_SIZE
