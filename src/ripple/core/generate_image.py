"""Functions for generating an image from a movie."""

import torch
from torch_fourier_filter.dose_weight import dose_weight_movie


def generate_dose_weighted_image(
    movie: torch.Tensor,
    pixel_size: float,
    pre_exposure: float,
    fluence_per_frame: float,
    voltage: float,
) -> torch.Tensor:
    """
    Dose weight the movie.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to dose weight.
    pixel_size: float
        The pixel size in Angstroms per pixel.
    pre_exposure: float
        The pre-exposure time in seconds.
    fluence_per_frame: float
        The dose per frame in electrons per Angstrom^2/frame.
    voltage: float
        The accelerating voltage in kilovolts.

    Returns
    -------
    torch.Tensor
    The dose weighted movie.
    """
    # get the height and width from the last two dimensions
    frame_shape = (movie.shape[-2], movie.shape[-1])
    # FFT  each frame
    movie_dft = torch.fft.rfft2(movie, dim=(-2, -1))  # pylint: disable=not-callable
    # apply dose weight
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
    # inverse FFT
    movie_dw = torch.fft.irfft2(  # pylint: disable=not-callable
        movie_dw_dft, s=frame_shape, dim=(-2, -1)
    )
    image_dw = sum_movie(movie_dw)
    return image_dw


def sum_movie(
    movie: torch.Tensor,
) -> torch.Tensor:
    """
    Sum the movie.

    Parameters
    ----------
    movie: torch.Tensor
        The movie to sum.

    Returns
    -------
    torch.Tensor
        The summed movie.
    """
    return torch.sum(movie, dim=0)
