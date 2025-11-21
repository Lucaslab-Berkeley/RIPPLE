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


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def dose_weight_memory_efficient(
    movie: torch.Tensor,
    pixel_size: float,
    pre_exposure: float = 0.0,
    dose_per_frame: float = 1.0,
    voltage: float = 300.0,
    memory_strategy: str = "checkpointing",
    memory_efficient: bool = True,
    chunk_size: int = 10,
) -> torch.Tensor:
    """
    Apply dose weighting to a movie using the correct normalization.

    Since dose_weight_movie requires all frames for proper normalization,
    we use memory optimization strategies instead of chunking.

    Parameters
    ----------
    movie : torch.Tensor
        Input movie tensor (t, h, w)
    pixel_size : float
        Pixel size in Angstroms
    pre_exposure : float
        Pre-exposure dose
    dose_per_frame : float
        Dose per frame
    voltage : float
        Acceleration voltage
    memory_strategy : str
        Memory optimization strategy: 'full', 'checkpointing', 'adaptive'
    memory_efficient : bool
        Whether to use memory efficient strategy
    chunk_size : int
        Chunk size for memory efficient strategy

    Returns
    -------
    torch.Tensor
        The dose weighted movie as a float32 tensor.
    """
    frame_shape = (movie.shape[-2], movie.shape[-1])

    if memory_strategy == "full":
        # Direct computation with chunked FFT and inverse FFT
        n_frames = movie.shape[0]
        movie_dft_chunks = []

        # Process forward FFT in chunks
        for i in range(0, n_frames, chunk_size):
            chunk = movie[i : min(i + chunk_size, n_frames)]
            chunk_dft = torch.fft.rfft2(chunk, dim=(-2, -1), norm="ortho")  # pylint: disable=not-callable
            movie_dft_chunks.append(chunk_dft)
            del chunk, chunk_dft
            torch.cuda.empty_cache()

        # Concatenate all chunks
        movie_dft = torch.cat(movie_dft_chunks, dim=0)
        del movie_dft_chunks
        torch.cuda.empty_cache()

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
            memory_efficient=memory_efficient,
            chunk_size=chunk_size,
        )

        # Process inverse FFT in chunks to reduce memory
        n_frames = movie_dw_dft.shape[0]
        image_dw = None

        for i in range(0, n_frames, chunk_size):
            chunk_dw_dft = movie_dw_dft[i : min(i + chunk_size, n_frames)]
            chunk_dw = torch.fft.irfft2(  # pylint: disable=not-callable
                chunk_dw_dft, s=frame_shape, dim=(-2, -1), norm="ortho"
            )  # pylint: disable=not-callable

            if image_dw is None:
                image_dw = torch.sum(chunk_dw, dim=0)
            else:
                image_dw += torch.sum(chunk_dw, dim=0)

            del chunk_dw, chunk_dw_dft
            torch.cuda.empty_cache()

        return image_dw

    if memory_strategy == "checkpointing":
        # Use gradient checkpointing to reduce memory usage
        def _dose_weight_forward(movie: torch.Tensor) -> torch.Tensor:
            n_frames = movie.shape[0]
            movie_dft_chunks = []

            # Process forward FFT in chunks
            for i in range(0, n_frames, chunk_size):
                chunk = movie[i : min(i + chunk_size, n_frames)]
                chunk_dft = torch.fft.rfft2(chunk, dim=(-2, -1), norm="ortho")  # pylint: disable=not-callable
                movie_dft_chunks.append(chunk_dft)
                del chunk, chunk_dft
                torch.cuda.empty_cache()

            # Concatenate all chunks
            movie_dft = torch.cat(movie_dft_chunks, dim=0)
            del movie_dft_chunks
            torch.cuda.empty_cache()

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
                memory_efficient=memory_efficient,
                chunk_size=chunk_size,
            )

            # Process inverse FFT in chunks to reduce memory
            n_frames = movie_dw_dft.shape[0]
            image_dw = None

            for i in range(0, n_frames, chunk_size):
                chunk_dw_dft = movie_dw_dft[i : min(i + chunk_size, n_frames)]
                chunk_dw = torch.fft.irfft2(  # pylint: disable=not-callable
                    chunk_dw_dft, s=frame_shape, dim=(-2, -1), norm="ortho"
                )  # pylint: disable=not-callable

                if image_dw is None:
                    image_dw = torch.sum(chunk_dw, dim=0)
                else:
                    image_dw += torch.sum(chunk_dw, dim=0)

                del chunk_dw, chunk_dw_dft
                torch.cuda.empty_cache()

            return image_dw

        return torch.utils.checkpoint.checkpoint(_dose_weight_forward, movie)
    raise ValueError(f"Unknown memory strategy: {memory_strategy}")
