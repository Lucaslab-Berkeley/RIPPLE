"""Tests for data_io loading helpers and MovieConfig file-dispatch properties."""

# pylint: disable=redefined-outer-name

import mrcfile
import numpy as np
import pytest
import torch
from tifffile import imwrite

from ripple.config.movie_config import MovieConfig
from ripple.utils.data_io import load_image_from_path, load_movie_from_path

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
# load_image_from_path
# ---------------------------------------------------------------------------


def test_load_image_from_path_mrc(mrc_image):
    path, data = mrc_image
    result = load_image_from_path(path)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_image_from_path_tif(tif_image):
    path, data = tif_image
    result = load_image_from_path(path)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_image_from_path_tiff_extension(tmp_path, image_data):
    path = tmp_path / "image.tiff"
    imwrite(str(path), image_data)
    result = load_image_from_path(path)
    assert result.shape == torch.Size([16, 16])


def test_load_image_from_path_gain_extension(tmp_path, image_data):
    path = tmp_path / "image.gain"
    imwrite(str(path), image_data)
    result = load_image_from_path(path)
    assert result.shape == torch.Size([16, 16])


def test_load_image_from_path_dark_extension(tmp_path, image_data):
    path = tmp_path / "image.dark"
    imwrite(str(path), image_data)
    result = load_image_from_path(path)
    assert result.shape == torch.Size([16, 16])


def test_load_image_from_path_unsupported_extension(tmp_path):
    path = tmp_path / "image.xyz"
    path.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="Unsupported image file extension"):
        load_image_from_path(path)


# ---------------------------------------------------------------------------
# load_movie_from_path
# ---------------------------------------------------------------------------


def test_load_movie_from_path_mrc(mrc_movie):
    path, data = mrc_movie
    result = load_movie_from_path(path)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([20, 16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_movie_from_path_tif(tif_movie):
    path, data = tif_movie
    result = load_movie_from_path(path)
    assert isinstance(result, torch.Tensor)
    assert result.shape == torch.Size([20, 16, 16])
    assert torch.allclose(result, torch.tensor(data))


def test_load_movie_from_path_unsupported_extension(tmp_path):
    path = tmp_path / "movie.eer"
    path.write_bytes(b"dummy")
    with pytest.raises(ValueError, match="Unsupported movie file extension"):
        load_movie_from_path(path)


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
